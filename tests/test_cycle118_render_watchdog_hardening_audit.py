"""Cycle 118: hardening audit of the render watchdog (cycle 116).

``_render_with_time_budget`` bounded checkpoint-controlled Jinja CPU
burns with a SIGALRM watchdog. Red-team review found two robustness
gaps:

1. A non-finite ``budget_s`` (``float("inf")``) reached ``setitimer``,
   which raises ``OverflowError`` -- not covered by the ``except
   (OSError, ValueError)`` fallback, so the defensive "fall back to the
   unbounded render" contract was violated.
2. Teardown disarmed the timer but did not block ``SIGALRM``: an alarm
   that became pending in that window would fire mid-restore, and
   because the budget exception was a ``ValueError`` the cleanup
   guards would swallow it, leaving the previous handler/timer
   inconsistent.

W4 (RR-10) retired the watchdog itself: rendering moved into an owned
worker process (``model_protocol_lab.render_worker``) whose deadline
the parent enforces with a timed queue wait. Both hazards are gone by
construction — there is no ``setitimer`` call to feed a bad budget to,
and no handler is ever installed, so there is no teardown window.
These tests pin the successor surface so neither hazard can regrow:

* the session refuses a non-positive or non-finite budget at
  construction (the old "non-finite disables the watchdog" escape
  hatch is deliberately gone — the deadline is structural now);
* the render path never touches the caller's signal mask, handler, or
  timer, and a caller's own alarm firing mid-render is delivered to
  the caller's handler and changes nothing.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

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


class TestWatchdogHardening(unittest.TestCase):
    """Test watchdog hardening."""

    def test_infinite_budget_refused(self) -> None:
        """Test infinite budget refused."""
        # setitimer(inf) raised OverflowError in the old design; the
        # session boundary refuses the value before it can reach any
        # wait, and "no deadline" is not an option anymore.
        from model_protocol_lab.render_worker import RenderSession

        with self.assertRaises(ValueError):
            RenderSession("unused.json", "t", None, budget_s=float("inf"))

    def test_nan_budget_refused(self) -> None:
        """Test nan budget refused."""
        from model_protocol_lab.render_worker import RenderSession

        with self.assertRaises(ValueError):
            RenderSession("unused.json", "t", None, budget_s=float("nan"))

    def test_nonpositive_budget_refused(self) -> None:
        """Test nonpositive budget refused."""
        # budget_s <= 0 used to disable the watchdog (a documented
        # escape hatch). The worker deadline is structural; the hatch
        # is gone and the value is a programming error.
        from model_protocol_lab.render_worker import RenderSession

        for bad in (0.0, -1.0):
            with self.assertRaises(ValueError):
                RenderSession("unused.json", "t", None, budget_s=bad)

    def test_signal_mask_not_leaked(self) -> None:
        """Test signal mask not leaked."""
        # The old teardown blocked SIGALRM across its restore and had
        # to prove the mask was put back. The worker path never touches
        # the mask; pin that it stays exactly as the caller left it.
        import signal

        from model_protocol_lab.render_worker import RenderSession

        if not hasattr(signal, "pthread_sigmask"):
            self.skipTest("requires pthread_sigmask (Unix)")
        before = signal.pthread_sigmask(signal.SIG_BLOCK, set())
        with TemporaryDirectory() as tmp:
            root = _write_checkpoint(Path(tmp), CLEAN_TEMPLATE)
            with RenderSession(
                str(root / "tokenizer.json"), CLEAN_TEMPLATE, None, budget_s=5.0
            ) as session:
                session.start()
                session.render([{"role": "user", "content": "hi"}])
        after = signal.pthread_sigmask(signal.SIG_BLOCK, set())
        self.assertEqual(before, after, "signal mask changed by the render path")
        self.assertNotIn(signal.SIGALRM, after, "SIGALRM left blocked after render")

    def test_caller_alarm_mid_render_is_the_callers(self) -> None:
        """Test caller alarm mid render is the callers."""
        # A SIGALRM becoming pending during a render (a caller's own
        # stale alarm, not the render path's — the render path has
        # none) is delivered to the caller's handler and corrupts
        # nothing: the render completes, the handler stays installed,
        # and no timer is left armed by the render path.
        import signal

        from model_protocol_lab.render_worker import RenderSession

        if not hasattr(signal, "setitimer"):
            self.skipTest("requires setitimer (Unix)")

        fired: list[int] = []

        def old_handler(signum: int, frame: object) -> None:
            """Record the caller's own alarm delivery."""
            fired.append(signum)

        prev = signal.signal(signal.SIGALRM, old_handler)
        try:
            with TemporaryDirectory() as tmp:
                root = _write_checkpoint(Path(tmp), CLEAN_TEMPLATE)
                with RenderSession(
                    str(root / "tokenizer.json"), CLEAN_TEMPLATE, None, budget_s=5.0
                ) as session:
                    signal.setitimer(signal.ITIMER_REAL, 0.3)
                    session.start()
                    text = session.render([{"role": "user", "content": "hi"}])
            self.assertIn("<|im_start|>user", text)
            self.assertEqual(fired, [signal.SIGALRM])
            self.assertIs(signal.getsignal(signal.SIGALRM), old_handler)
            delay, _interval = signal.getitimer(signal.ITIMER_REAL)
            self.assertEqual(delay, 0.0)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, prev)


if __name__ == "__main__":
    unittest.main()
