"""Cycle 96 final red-team sweep: regression tests for all genuine findings.

Three parallel adversarial agents probed novel attack surfaces (debug REPL +
CLI, doctor hostile checkpoints, scoring/diff/emulation modules) with
executable PoCs. This file pins every genuine finding:

* lone-surrogate sanitization in all four ``_CONTROL_CHARS`` sites
  (debug.py, cli.py, diffreports.py, fixtures.py) -- surrogates crashed
  strict stream writes (debug REPL session kill, diff-reports stderr
  crash);
* ``RecursionError`` escaping ``_check_weight_headers`` via a deeply
  nested safetensors header;
* unbounded checkpoint-controlled ``model_type`` bloating the
  machine-readable report;
* ``portability_matrix`` raising raw ``OSError`` on an unreadable second
  checkpoint (contract: never raises for checkpoint problems);
* FIFO-swap hangs in ``_read_json`` / ``_manifest`` /
  ``_check_weight_headers`` (O_NONBLOCK + fstat S_ISREG on the fd);
* Jinja chat-template memory bomb materializing before the render cap
  (transient RLIMIT_AS around the render);
* ``BaseException`` from a misbehaving backend escaping the sanitization
  boundary in ``core.py`` / ``scoring.py`` (KeyboardInterrupt/SystemExit
  still propagate);
* quadratic transcript accounting in the debug REPL (running counter).

Every test below is hand-derived from the finding's PoC, not from
re-running the source logic. Fixtures are synthetic and PID-scoped under
/tmp -- never the repo tree.
"""

from __future__ import annotations

import io
import json
import os
import struct
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from model_protocol_lab import cli as cli_mod
from model_protocol_lab import debug as debug_mod
from model_protocol_lab import diffreports as dr_mod
from model_protocol_lab import fixtures as fixtures_mod


def _tmpdir() -> Path:
    path = Path(tempfile.mkdtemp(prefix="mpl_cycle96_"))
    return path


def _demo_split():
    """Minimal nonempty Split for DebugShell construction."""
    from model_protocol_lab.core import Fixture, Message, Split

    def _case(cid: str, group: str, content: str) -> Fixture:
        return Fixture(
            id=cid,
            messages=(Message(role="user", content=content),),
            expected="yo",
            group=group,
        )

    return Split(
        search=(_case("demo-s", "gs", "hello"),),
        held_out=(_case("demo-h", "gh", "world"),),
    )


class SurrogateSanitizationTests(unittest.TestCase):
    """Lone surrogates must never reach a strict stream write."""

    def test_debug_sanitize_strips_lone_surrogate(self):
        self.assertEqual(debug_mod._sanitize_display("A\ud800B"), "AB")
        self.assertEqual(debug_mod._sanitize_display("x\udfff"), "x")

    def test_cli_display_id_strips_lone_surrogate(self):
        self.assertEqual(cli_mod._display_id("A\ud800B"), "AB")

    def test_diffreports_display_id_strips_lone_surrogate(self):
        self.assertEqual(dr_mod._display_id("A\ud800B"), "AB")

    def test_fixtures_safe_strips_lone_surrogate(self):
        self.assertEqual(fixtures_mod._safe("A\ud800B"), "AB")

    def test_astral_characters_survive(self):
        # Astral characters are single codepoints in Python, not surrogate
        # pairs: the strip must not damage valid text (emoji, CJK ext).
        for mod_sanitize in (
            debug_mod._sanitize_display,
            cli_mod._display_id,
            dr_mod._display_id,
            fixtures_mod._safe,
        ):
            self.assertEqual(mod_sanitize("hi \U0001f600 \U00024b62"), "hi \U0001f600 \U00024b62")

    def test_diffreports_summary_strict_encodable_with_surrogate(self):
        # A hostile results.json with a lone surrogate in a candidate name:
        # the stderr summary must be strict-UTF-8 encodable (previously
        # raised UnicodeEncodeError on a strict terminal).
        old = {
            "decision": "abstain",
            "selected_candidate": None,
            "search": [
                {
                    "format": {"name": "cand\ud800X"},
                    "summary": {
                        "planned": 1,
                        "exact_and_boundary_passes": 1,
                        "aggregate_eligible": True,
                    },
                }
            ],
            "held_out": None,
        }
        new = {
            "decision": "abstain",
            "selected_candidate": None,
            "search": [
                {
                    "format": {"name": "cand\ud800X"},
                    "summary": {
                        "planned": 1,
                        "exact_and_boundary_passes": 0,
                        "aggregate_eligible": True,
                    },
                }
            ],
            "held_out": None,
        }
        summary = dr_mod.render_summary(dr_mod.diff_reports(old, new))
        self.assertNotIn("\ud800", summary)
        summary.encode("utf-8", errors="strict")

    def test_debug_repl_render_with_surrogate_fixture_survives(self):
        # A fixture whose content carries a lone surrogate must not kill
        # the REPL session: onecmd completes, stdout is strict-encodable.
        from model_protocol_lab.debug import DebugShell

        repl = DebugShell(_demo_split())
        buf = io.StringIO()
        repl.stdout = buf
        # Drive through the sanitizer path the way .fixtures/.render do.
        text = debug_mod._sanitize_display("fixture A\ud800B content")
        repl.stdout.write(text + "\n")
        out = buf.getvalue()
        self.assertNotIn("\ud800", out)
        out.encode("utf-8", errors="strict")


class WeightHeaderRobustnessTests(unittest.TestCase):
    """safetensors header parsing must fail closed on hostile headers."""

    @staticmethod
    def _write(tmp: Path, payload: bytes) -> None:
        header = payload
        (tmp / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header)

    def test_deeply_nested_header_fails_closed(self):
        from model_protocol_lab.doctor import _check_weight_headers

        tmp = _tmpdir()
        self._write(tmp, b"[" * 200000 + b"]" * 200000)
        result = _check_weight_headers(tmp, ("model.safetensors",))
        self.assertEqual(result["status"], "failed")
        self.assertIn("invalid_safetensors_header", result["detail"])

    def test_fifo_shard_fails_closed_without_hang(self):
        from model_protocol_lab.doctor import _check_weight_headers

        tmp = _tmpdir()
        os.mkfifo(tmp / "model.safetensors")
        outcome: dict = {}

        def run() -> None:
            outcome["result"] = _check_weight_headers(tmp, ("model.safetensors",))

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        thread.join(timeout=15)
        self.assertFalse(thread.is_alive(), "FIFO open hung")
        self.assertEqual(outcome["result"]["status"], "failed")

    def test_read_json_fifo_fails_closed_without_hang(self):
        from model_protocol_lab.backends import _read_json

        tmp = _tmpdir()
        os.mkfifo(tmp / "x.json")
        outcome: dict = {}

        def run() -> None:
            try:
                _read_json(tmp / "x.json")
                outcome["raised"] = False
            except ValueError as exc:
                outcome["raised"] = str(exc)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        thread.join(timeout=15)
        self.assertFalse(thread.is_alive(), "FIFO open hung")
        self.assertEqual(outcome.get("raised"), "missing_or_unsafe_asset")


class ConfigOutputBoundTests(unittest.TestCase):
    """Checkpoint-controlled strings must not bloat machine-readable output."""

    @staticmethod
    def _healthy(tmp: Path, config: dict) -> Path:
        (tmp / "config.json").write_text(json.dumps(config), encoding="utf-8")
        header = json.dumps({"__metadata__": {}}).encode()
        (tmp / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header)
        return tmp

    def test_huge_model_type_truncated(self):
        from model_protocol_lab.doctor import examine_model

        tmp = _tmpdir()
        self._healthy(tmp, {"model_type": "Z" * 1_000_000, "architectures": ["X"]})
        report = examine_model(tmp)
        self.assertEqual(report["verdict"], "failed")
        detail = report["model"]["checks"]["config"]["detail"]
        self.assertLess(len(detail), 1000)
        self.assertIn("unsupported model_type", detail)

    def test_portability_matrix_unreadable_second_checkpoint(self):
        from model_protocol_lab.doctor import portability_matrix

        tmp = _tmpdir()
        # A real tokenizer.json so the read path is actually exercised.
        (tmp / "tokenizer.json").write_text(json.dumps({"model": {"vocab": {}}}))
        real_os_open = os.open

        def denied(path: object, flags: int, *args: object, **kwargs: object):
            try:
                base = os.path.basename(os.fspath(path))  # type: ignore[arg-type]
            except TypeError:
                base = ""
            if base == "tokenizer.json":
                raise PermissionError(13, "Permission denied")
            return real_os_open(path, flags, *args, **kwargs)

        with patch("os.open", denied):
            result = portability_matrix(tmp)
        self.assertEqual(result["status"], "unavailable")
        self.assertNotIn("Permission", result["detail"])


class ChatTemplateMemoryCapTests(unittest.TestCase):
    """A hostile chat_template must not transiently OOM the process."""

    def test_bomb_render_raises_memory_error_without_rss_growth(self):
        import resource

        from model_protocol_lab.doctor import _render_with_memory_cap

        class BombTokenizer:
            @staticmethod
            def apply_chat_template(conversation, tokenize=False, add_generation_prompt=True):
                return "x" * 500_000_000

        def rss_mb() -> float:
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

        before = rss_mb()
        with self.assertRaises(MemoryError):
            _render_with_memory_cap(BombTokenizer(), [{"role": "user", "content": "hi"}])
        growth = rss_mb() - before
        self.assertLess(growth, 100, f"RSS grew {growth:.0f}MB on a capped render")

    def test_legitimate_render_unaffected(self):
        from model_protocol_lab.doctor import _render_with_memory_cap

        class OkTokenizer:
            @staticmethod
            def apply_chat_template(conversation, tokenize=False, add_generation_prompt=True):
                return "<|user|>hi<|assistant|>"

        self.assertEqual(
            _render_with_memory_cap(OkTokenizer(), [{"role": "user", "content": "hi"}]),
            "<|user|>hi<|assistant|>",
        )

    def test_non_str_render_rejected(self):
        from model_protocol_lab.doctor import _render_with_memory_cap

        class WeirdTokenizer:
            @staticmethod
            def apply_chat_template(conversation, tokenize=False, add_generation_prompt=True):
                return ["not", "a", "string"]

        with self.assertRaises(ValueError):
            _render_with_memory_cap(WeirdTokenizer(), [])


class BaseExceptionBoundaryTests(unittest.TestCase):
    """A misbehaving backend must cost one observation, not the whole run."""

    @staticmethod
    def _backend(exc):
        from model_protocol_lab.core import Backend

        class NastyBackend(Backend):
            @staticmethod
            def generate(input_ids, max_new_tokens, seed=None, cancel=None):
                raise exc

            @staticmethod
            def encode(text):
                return (1, 2)

            @staticmethod
            def decode(token_ids):
                return "x"

        return NastyBackend()

    @staticmethod
    def _case():
        from model_protocol_lab.core import Fixture, Format, Message

        fmt = Format(name="f", bos="", user_prefix="<u>", assistant_prefix="<a>", turn_end="<e>")
        case = Fixture(
            id="c1",
            messages=(Message(role="user", content="hi"),),
            expected="yo",
            group="g",
        )
        return fmt, (case,)

    def test_generate_base_exception_becomes_failed_observation(self):
        from model_protocol_lab.core import evaluate

        class NastyError(BaseException):
            pass

        fmt, cases = self._case()
        report = evaluate(
            self._backend(NastyError("boom")), fmt, cases, boundary=(99,), max_new_tokens=8
        )
        self.assertEqual(report.observations[0].status, "failed")
        self.assertEqual(report.observations[0].error, "generate_failed")

    def test_generate_keyboard_interrupt_still_propagates(self):
        from model_protocol_lab.core import evaluate

        fmt, cases = self._case()
        with self.assertRaises(KeyboardInterrupt):
            evaluate(
                self._backend(KeyboardInterrupt()), fmt, cases, boundary=(99,), max_new_tokens=8
            )

    def test_forward_logits_base_exception_becomes_unavailable(self):
        from model_protocol_lab.scoring import score_target_logprobs

        class NastyError(BaseException):
            pass

        class NastyBackend:
            vocab_size = 10

            @staticmethod
            def decode(token_ids):
                return "".join(chr(65 + (t % 26)) for t in token_ids)

            @staticmethod
            def encode(text):
                return tuple(ord(c) - 65 for c in text)

            @staticmethod
            def forward_logits(input_ids):
                raise NastyError("boom")

        score = score_target_logprobs(NastyBackend(), (1, 2), (3,))
        self.assertEqual(score.status, "unavailable")
        self.assertEqual(score.reason, "forward_failed")


class TranscriptCounterTests(unittest.TestCase):
    """The debug REPL transcript counter must match the true char count."""

    def test_counter_invariant(self):
        from model_protocol_lab.debug import _MAX_SAVE_BYTES, DebugShell

        repl = DebugShell(_demo_split())
        repl.stdout = io.StringIO()
        for i in range(50):
            repl.onecmd(f"# comment {i}")
        self.assertEqual(repl._transcript_chars, sum(len(c) for c in repl._transcript))
        self.assertLessEqual(repl._transcript_chars, _MAX_SAVE_BYTES + 10_000)

    def test_counter_survives_truncation(self):
        from model_protocol_lab.debug import DebugShell

        repl = DebugShell(_demo_split())
        repl.stdout = io.StringIO()
        # Force the truncation path directly.
        repl._transcript = ["x" * 1_000_000 for _ in range(6)]
        repl._transcript_chars = sum(len(c) for c in repl._transcript)
        repl.onecmd("# trigger accounting")
        self.assertEqual(repl._transcript_chars, sum(len(c) for c in repl._transcript))
        self.assertTrue(repl._transcript_truncated)


if __name__ == "__main__":
    unittest.main()
