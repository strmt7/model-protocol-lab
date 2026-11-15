"""Cycle 116: CPU budget for checkpoint-controlled Jinja renders.

``_render_with_memory_cap`` bounds memory bombs (RLIMIT_AS), but a hostile
``chat_template`` can burn CPU without allocating -- e.g. nested loops
inside the template sandbox's range bound. Cycle 116 answered with a
SIGALRM wall-clock watchdog in-process (``_render_with_time_budget``).
W4 (RR-10) retired that mechanism: the watchdog could not arm off the
main thread or on Windows, so the burn ran unbounded in the host
process exactly where containment mattered most. Rendering now lives
in an owned worker process (``model_protocol_lab.render_worker``) and
the budget is a per-request deadline the *parent* enforces on every
platform and thread: on expiry the worker is killed and the request
reports :class:`WorkerDied`; the pair degrades to ``"unavailable"``.

These tests pin the successor contract at the level the originals
operated on:

* a real CPU burn through the real worker dies at the deadline, on
  the main thread and off it (the off-thread arm is the RR-10
  regression: the old watchdog never armed there);
* the retired in-process helpers are gone, so the unbounded path
  cannot silently return;
* legitimate renders and the memory cap are unaffected
  (``render_worker._render_once`` is the old capped render, moved);
* a render leaves the caller's signal state exactly as it found it —
  trivially true now that nothing installs a handler, and pinned so
  reintroducing signal machinery in the render path fails loudly.
"""

from __future__ import annotations

import json
import os
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

# Nested loops inside both range bounds that apply to a real render
# (the firewall's 1_000_000 literal cap; the transformers sandbox's
# MAX_RANGE of 100_000): 10**10 inner steps, ~1000 s of CPU.
BURN_TEMPLATE = (
    "{% for i in range(100000) %}"
    "{% for j in range(100000) %}"
    "{% set x = i + j %}"
    "{% endfor %}{% endfor %}done"
)

CLEAN_TEMPLATE = (
    "{% for message in messages %}"
    "<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def _write_checkpoint(root: Path, template: str) -> Path:
    """Write checkpoint."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    root.mkdir(parents=True, exist_ok=True)
    vocab = {f"tok{i}": i for i in range(15)}
    vocab["[UNK]"] = 15
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.add_special_tokens(["[UNK]", "[BOS]", "[EOS]"])
    tokenizer.save(str(root / "tokenizer.json"))
    (root / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": template}), encoding="utf-8"
    )
    return root


class _OkTokenizer:
    """Ok tokenizer."""

    @staticmethod
    def apply_chat_template(
        conversation: object, tokenize: bool = False, add_generation_prompt: bool = True
    ) -> str:
        """Apply chat template."""
        return "<|user|>hi<|assistant|>"


class TestRenderTimeBudget(unittest.TestCase):
    """Test render time budget."""

    def test_retired_inprocess_helpers_are_gone(self) -> None:
        """Test retired inprocess helpers are gone."""
        from model_protocol_lab import doctor

        for name in (
            "_render_with_time_budget",
            "_render_with_memory_cap",
            "_RenderTimeBudgetExceeded",
        ):
            self.assertFalse(hasattr(doctor, name), f"doctor.{name} must stay retired")

    def test_cpu_burn_dies_at_the_deadline(self) -> None:
        """Test cpu burn dies at the deadline."""
        from model_protocol_lab.render_worker import RenderSession, WorkerDied

        with TemporaryDirectory() as tmp:
            root = _write_checkpoint(Path(tmp), BURN_TEMPLATE)
            with RenderSession(
                str(root / "tokenizer.json"), BURN_TEMPLATE, None, budget_s=1.0
            ) as session:
                session.start()
                start = time.monotonic()
                with self.assertRaises(WorkerDied):
                    session.render([{"role": "user", "content": "hi"}])
                elapsed = time.monotonic() - start
        # Generous slack for loaded CI boxes; the point is ~1 s, not ~1000 s.
        self.assertLess(elapsed, 20.0, f"deadline did not kill the burn ({elapsed:.1f}s)")

    def test_cpu_burn_dies_at_the_deadline_off_main_thread(self) -> None:
        """Test cpu burn dies at the deadline off main thread."""
        # The RR-10 regression arm: the SIGALRM watchdog could never arm
        # here. The parent-side deadline can, and does.
        from model_protocol_lab.render_worker import RenderSession, WorkerDied

        with TemporaryDirectory() as tmp:
            root = _write_checkpoint(Path(tmp), BURN_TEMPLATE)
            outcome: list[str] = []

            def _work() -> None:
                """Work."""
                try:
                    with RenderSession(
                        str(root / "tokenizer.json"), BURN_TEMPLATE, None, budget_s=1.0
                    ) as session:
                        session.start()
                        session.render([{"role": "user", "content": "hi"}])
                    outcome.append("rendered")
                except WorkerDied:
                    outcome.append("died")

            worker = threading.Thread(target=_work)
            start = time.monotonic()
            worker.start()
            worker.join(timeout=30)
            elapsed = time.monotonic() - start
        self.assertFalse(worker.is_alive(), "worker thread hung")
        self.assertEqual(outcome, ["died"])
        self.assertLess(elapsed, 25.0)

    def test_legitimate_render_unaffected(self) -> None:
        """Test legitimate render unaffected."""
        from model_protocol_lab.render_worker import _render_once

        self.assertEqual(
            _render_once(_OkTokenizer(), [{"role": "user", "content": "hi"}]),
            "<|user|>hi<|assistant|>",
        )

    @unittest.skipUnless(
        os.name == "posix",
        "POSIX-only: the address-space cap does not exist on win32",
    )
    def test_memory_error_still_propagates(self) -> None:
        """Test memory error still propagates."""
        # The deadline wrapper composes with the memory cap inside the
        # worker's render helper; it must not mask it.
        from model_protocol_lab.render_worker import _render_once

        class _MemBomb:
            """Mem bomb."""

            @staticmethod
            def apply_chat_template(
                conversation: object,
                tokenize: bool = False,
                add_generation_prompt: bool = True,
            ) -> str:
                """Apply chat template."""
                return "x" * 500_000_000

        with self.assertRaises(MemoryError):
            _render_once(_MemBomb(), [])

    def test_render_leaves_signal_state_untouched(self) -> None:
        """Test render leaves signal state untouched."""
        # The old watchdog saved/restored the caller's SIGALRM handler
        # and timer around every render; the worker path installs
        # nothing, so both must be exactly as the caller left them.
        import signal

        from model_protocol_lab.render_worker import RenderSession

        if not hasattr(signal, "setitimer"):
            self.skipTest("requires setitimer (Unix)")

        def _handler(signum: int, frame: object) -> None:
            """Handler."""
            raise AssertionError("stray alarm fired")

        old = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _handler)
        try:
            with TemporaryDirectory() as tmp:
                root = _write_checkpoint(Path(tmp), CLEAN_TEMPLATE)
                with RenderSession(
                    str(root / "tokenizer.json"), CLEAN_TEMPLATE, None, budget_s=5.0
                ) as session:
                    session.start()
                    text = session.render([{"role": "user", "content": "hi"}])
            self.assertIn("<|im_start|>user", text)
            self.assertIs(signal.getsignal(signal.SIGALRM), _handler)
            delay, _interval = signal.getitimer(signal.ITIMER_REAL)
            self.assertEqual(delay, 0.0)
        finally:
            signal.signal(signal.SIGALRM, old)


if __name__ == "__main__":
    unittest.main()
