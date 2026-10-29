"""Cycle 10: red-team findings H1/M1/L1 -- input-bound and sanitization regressions.

Pins the three red-team fixes with deterministic stub-only tests (no torch,
no network, no model weights):

- H1: the token-id length cap is a named constant sized for the emulations'
  quadratic per-step-decode cost (``MAX_OUTPUT_IDS``); over-cap input is
  rejected with the fixed ``output_ids_too_long`` code, at-cap input is
  accepted and completes within a generous budget on the repo's linear
  reference stub, and hostile recorded observations are skipped (not hung)
  by ``divergence_section_data``.
- M1: stop texts carry a per-stop length bound and a total count bound
  (``MAX_STOP_TEXT_LEN`` / ``MAX_STOP_TEXTS``); over-length and over-count
  inputs are rejected with fixed codes, boundary values pass, dedup runs
  before the count check, and hostile-scale stop texts are rejected (not
  hung) through the diagnosis path and the ``diff-backends`` CLI path --
  including the ``cli.py`` second hop where a hostile tokenizer's
  ``decode`` feeds the same validation.
- L1: tokenizer-loader failures collapse to the fixed
  ``tokenizer_load_failed`` vocabulary (only exact-match allowlisted codes
  pass through); hostile exception bytes never reach the terminal, a
  non-ValueError/OSError parse failure produces no traceback, and the
  debug REPL's error path strips terminal-control characters.
"""

import argparse
import io
import time
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest import mock

from model_protocol_lab import diffbackends
from model_protocol_lab.debug import DebugShell, run_debug
from model_protocol_lab.diffbackends import (
    MAX_OUTPUT_IDS,
    MAX_PROMPT_IDS,
    MAX_STOP_TEXT_LEN,
    MAX_STOP_TEXTS,
    _checked_ids,
    _checked_stop_texts,
    diff_backends,
    divergence_section_data,
    run_diff_backends,
)
from model_protocol_lab.doctor import _loader_failure_code


class LinearStubBackend:
    """O(k) decode stub: the reference backend for quadratic-cost measurement.

    Whole-sequence decodes concatenate per-token texts, so one prefix
    decode costs O(prefix length) -- exactly the linear stub the H1 bound
    is measured against.
    """

    fingerprint = "cycle10-security-linear-stub-not-a-model"

    def __init__(self, size: int) -> None:
        self._texts = dict.fromkeys(range(size), "x")

    def decode(self, token_ids: tuple[int, ...]) -> str:
        """Decode."""
        return "".join(self._texts[token] for token in token_ids)


def _hostile_bytes() -> str:
    """Hostile bytes."""
    # Escape bytes a hostile tokenizer.json parse failure could echo into
    # an exception message: CSI sequences, a window-title set, and BEL.
    return "\x1b[2J\x1b]0;pwned\x07parse failed at byte 12: \xff\xfe evil"


# -- H1: quadratic CPU hang inside the documented-valid input range --------
# The old 131072 cap let one diff_backends call burn ~40 min on the linear
# reference stub (one backend.decode(prefix) per output step -- O(n^2)).


class OutputIdCapTests(unittest.TestCase):
    """Output id cap tests."""

    def test_caps_are_named_constants(self) -> None:
        """Test caps are named constants."""
        # No magic numbers: the bounds the emulations enforce are named.
        self.assertEqual(MAX_OUTPUT_IDS, 4096)
        self.assertEqual(MAX_PROMPT_IDS, 131072)
        self.assertIs(diffbackends.MAX_OUTPUT_IDS, MAX_OUTPUT_IDS)
        self.assertIs(diffbackends.MAX_PROMPT_IDS, MAX_PROMPT_IDS)

    def test_over_cap_output_ids_rejected(self) -> None:
        """Test over cap output ids rejected."""
        over = tuple([1] * (MAX_OUTPUT_IDS + 1))
        with self.assertRaises(ValueError) as ctx:
            _checked_ids(over, "output_ids", cap=MAX_OUTPUT_IDS)
        self.assertEqual(str(ctx.exception), "output_ids_too_long")

    def test_over_cap_rejected_through_diff_backends(self) -> None:
        """Test over cap rejected through diff backends."""
        backend = LinearStubBackend(MAX_OUTPUT_IDS + 1)
        with self.assertRaises(ValueError) as ctx:
            diff_backends(
                prompt_ids=(),
                output_ids=tuple([1] * (MAX_OUTPUT_IDS + 1)),
                stop_texts=("STOP",),
                eos_id=None,
                min_tokens=0,
                backend=backend,
            )
        self.assertEqual(str(ctx.exception), "output_ids_too_long")

    def test_at_cap_output_ids_accepted_and_fast(self) -> None:
        """Test at cap output ids accepted and fast."""
        # Worst-case single-stop cost at the cap on the linear reference
        # stub: measured ~2.5 s on the dev machine; the 60 s budget is
        # generous headroom for slow CI, not a performance claim.
        backend = LinearStubBackend(MAX_OUTPUT_IDS)
        ids = tuple(range(MAX_OUTPUT_IDS))
        start = time.perf_counter()
        result = diff_backends(
            prompt_ids=(),
            output_ids=ids,
            stop_texts=("STOP",),
            eos_id=None,
            min_tokens=0,
            backend=backend,
        )
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 60.0)
        self.assertIn(result["verdict"], ("agree", "diverge"))

    def test_prompt_cap_stays_generous(self) -> None:
        """Test prompt cap stays generous."""
        # Prompt tokens are decoded once each (linear cost) and lab runs
        # legitimately record prompts up to --max-context (32768), so the
        # prompt bound stays at the old 4x-window value.
        at_cap = tuple([1] * MAX_PROMPT_IDS)
        self.assertEqual(_checked_ids(at_cap, "prompt_ids", cap=MAX_PROMPT_IDS), at_cap)
        with self.assertRaises(ValueError) as ctx:
            _checked_ids(tuple([1] * (MAX_PROMPT_IDS + 1)), "prompt_ids", cap=MAX_PROMPT_IDS)
        self.assertEqual(str(ctx.exception), "prompt_ids_too_long")

    def test_hostile_observation_skipped_not_hung(self) -> None:
        """Test hostile observation skipped not hung."""
        # The divergence_section_data path (cli.py's backend_divergences
        # diagnosis over recorded observations) must skip an over-cap
        # recorded output, not burn quadratic time on it.
        backend = LinearStubBackend(MAX_OUTPUT_IDS + 1)
        observations = [
            {"fixture": "evil", "input_ids": [], "output_ids": [1] * (MAX_OUTPUT_IDS + 1)},
        ]
        start = time.perf_counter()
        section = divergence_section_data(
            observations=observations,
            stop_texts=("STOP",),
            eos_id=None,
            min_tokens=0,
            backend=backend,
        )
        elapsed = time.perf_counter() - start
        self.assertIsNone(section)  # no usable observations -> section omitted
        self.assertLess(elapsed, 10.0)


# -- M1: unbounded stop texts --------------------------------------------
# A 10 MB stop text balloons json.dumps(result) (embedded verbatim in
# records plus {stop_text!r} details) and the list-based dedup was O(m^2).


class StopTextBoundTests(unittest.TestCase):
    """Stop text bound tests."""

    def test_bounds_are_named_constants(self) -> None:
        """Test bounds are named constants."""
        self.assertEqual(MAX_STOP_TEXT_LEN, 1024)
        self.assertEqual(MAX_STOP_TEXTS, 16)

    def test_over_length_stop_rejected(self) -> None:
        """Test over length stop rejected."""
        evil = "x" * (MAX_STOP_TEXT_LEN + 1)
        with self.assertRaises(ValueError) as ctx:
            _checked_stop_texts((evil,))
        self.assertEqual(str(ctx.exception), "stop_text_too_long")

    def test_over_length_rejected_through_diff_backends(self) -> None:
        """Test over length rejected through diff backends."""
        backend = LinearStubBackend(8)
        with self.assertRaises(ValueError) as ctx:
            diff_backends(
                prompt_ids=(),
                output_ids=(1, 2, 3),
                stop_texts=("x" * (MAX_STOP_TEXT_LEN + 1),),
                eos_id=None,
                min_tokens=0,
                backend=backend,
            )
        self.assertEqual(str(ctx.exception), "stop_text_too_long")

    def test_over_count_stop_rejected(self) -> None:
        """Test over count stop rejected."""
        many = tuple(f"stop-{i}" for i in range(MAX_STOP_TEXTS + 1))
        with self.assertRaises(ValueError) as ctx:
            _checked_stop_texts(many)
        self.assertEqual(str(ctx.exception), "too_many_stop_texts")

    def test_boundary_values_accepted(self) -> None:
        """Test boundary values accepted."""
        at_len = "x" * MAX_STOP_TEXT_LEN
        self.assertEqual(_checked_stop_texts((at_len,)), (at_len,))
        at_count = tuple(f"stop-{i}" for i in range(MAX_STOP_TEXTS))
        self.assertEqual(_checked_stop_texts(at_count), at_count)
        # Boundary through the full emulation path.
        backend = LinearStubBackend(8)
        result = diff_backends(
            prompt_ids=(),
            output_ids=(1, 2, 3),
            stop_texts=at_count,
            eos_id=None,
            min_tokens=0,
            backend=backend,
        )
        self.assertEqual(len(result["cells"]), MAX_STOP_TEXTS)

    def test_dedup_runs_before_count_check(self) -> None:
        """Test dedup runs before count check."""
        # Duplicates collapse first (order-preserving): a hostile list of
        # repeated stops must not trip the count bound, and dedup itself
        # is linear (set-based), not O(m^2).
        bloated = ["STOP", "ZZZ"] * 5000
        start = time.perf_counter()
        checked = _checked_stop_texts(bloated)
        self.assertLess(time.perf_counter() - start, 5.0)
        self.assertEqual(checked, ("STOP", "ZZZ"))

    def test_hostile_scale_stop_rejected_through_diagnosis(self) -> None:
        """Test hostile scale stop rejected through diagnosis."""
        # 10 MB stop text via divergence_section_data: rejected with the
        # fixed code before any per-step decode loop runs -- no hang, and
        # the text never reaches json.dumps(result).
        backend = LinearStubBackend(8)
        observations = [{"fixture": "fx", "input_ids": [], "output_ids": [1, 2, 3]}]
        start = time.perf_counter()
        with self.assertRaises(ValueError) as ctx:
            divergence_section_data(
                observations=observations,
                stop_texts=("x" * 10_000_000,),
                eos_id=None,
                min_tokens=0,
                backend=backend,
            )
        self.assertLess(time.perf_counter() - start, 5.0)
        self.assertEqual(str(ctx.exception), "stop_text_too_long")

    def test_second_hop_hostile_decode_covered(self) -> None:
        """Test second hop hostile decode covered.

        cli.py's diagnosis path passes backend.decode(winner_boundary)
        into divergence_section_data as the stop text; a hostile
        tokenizer returning a huge string must hit the same validation
        (the call site degrades to "section omitted", never hangs).
        """

        class HostileDecodeBackend(LinearStubBackend):
            """Hostile decode backend."""

            @staticmethod
            def decode(token_ids: tuple[int, ...]) -> str:
                """Decode."""
                return "y" * 10_000_000

        backend = HostileDecodeBackend(8)
        boundary_text = backend.decode((1, 2, 3))
        self.assertEqual(len(boundary_text), 10_000_000)
        observations = [{"fixture": "fx", "input_ids": [], "output_ids": [1, 2, 3]}]
        try:
            divergence_section_data(
                observations=observations,
                stop_texts=(boundary_text,),
                eos_id=None,
                min_tokens=0,
                backend=backend,
            )
            section = "unexpected"
        except Exception:  # noqa: BLE001 -- mirrors the cli.py call site
            # Mirrors the cli.py call site: the section degrades alone.
            section = None
        self.assertIsNone(section)


class _CliStubTokenizer:
    """Minimal tokenizer stand-in for the diff-backends CLI path."""

    @staticmethod
    def encode(text: str, **kwargs: Any) -> tuple[int, ...]:
        """Encode."""
        return tuple(min(ord(ch), 127) for ch in text[:64])

    @staticmethod
    def decode(token_ids: list[int], **kwargs: Any) -> str:
        """Decode."""
        return "".join(chr(token_id) for token_id in token_ids)

    @staticmethod
    def get_vocab() -> dict[str, int]:
        """Get vocab."""
        return {chr(i): i for i in range(128)}


def _diff_cli_args(**overrides: Any) -> argparse.Namespace:
    """Diff cli args."""
    base: dict[str, Any] = {
        "min_tokens": 0,
        "eos_id": None,
        "stops": "STOP",
        "prompt_text": "the answer is",
        "output_text": "42 STOP hello",
        "model": "stub-checkpoint",
    }
    base.update(overrides)
    return argparse.Namespace(**base)


class DiffBackendsCliBoundTests(unittest.TestCase):
    """Diff backends cli bound tests."""

    @staticmethod
    def _run_cli(args: argparse.Namespace) -> tuple[int, str]:
        """Run cli."""
        parser = argparse.ArgumentParser(prog="mpl diff-backends")
        err = io.StringIO()
        with (
            mock.patch(
                "model_protocol_lab.doctor._build_primary_tokenizer",
                return_value=_CliStubTokenizer(),
            ),
            redirect_stderr(err),
        ):
            code = run_diff_backends(args, parser)
        return code, err.getvalue()

    def test_cli_rejects_over_length_stop(self) -> None:
        """Test cli rejects over length stop."""
        code, err = self._run_cli(_diff_cli_args(stops="x" * (MAX_STOP_TEXT_LEN + 1)))
        self.assertEqual(code, 1)
        self.assertIn("stop_text_too_long", err)
        self.assertNotIn("x" * 100, err)  # the hostile text is not echoed

    def test_cli_rejects_over_count_stops(self) -> None:
        """Test cli rejects over count stops."""
        stops = ",".join(f"s{i}" for i in range(MAX_STOP_TEXTS + 1))
        code, err = self._run_cli(_diff_cli_args(stops=stops))
        self.assertEqual(code, 1)
        self.assertIn("too_many_stop_texts", err)

    def test_cli_accepts_valid_stops(self) -> None:
        """Test cli accepts valid stops."""
        with mock.patch(
            "model_protocol_lab.doctor._build_primary_tokenizer",
            return_value=_CliStubTokenizer(),
        ):
            code = run_diff_backends(_diff_cli_args(stops="STOP,END"), argparse.ArgumentParser())
        self.assertEqual(code, 0)


# -- L1: hostile-checkpoint exception text reaches the terminal -----------

EVIL = _hostile_bytes()


class LoaderFailureCodeTests(unittest.TestCase):
    """Loader failure code tests."""

    def test_allowlisted_code_passes_through(self) -> None:
        """Test allowlisted code passes through."""
        self.assertEqual(
            _loader_failure_code(ValueError("oversized_tokenizer_file")),
            "oversized_tokenizer_file",
        )

    def test_hostile_value_error_collapses(self) -> None:
        """Test hostile value error collapses."""
        self.assertEqual(_loader_failure_code(ValueError(EVIL)), "tokenizer_load_failed")

    def test_non_value_error_collapses(self) -> None:
        """Test non value error collapses."""
        self.assertEqual(_loader_failure_code(RuntimeError(EVIL)), "tokenizer_load_failed")
        self.assertEqual(_loader_failure_code(OSError(EVIL)), "tokenizer_load_failed")


class DebugLoaderSanitizationTests(unittest.TestCase):
    """Debug loader sanitization tests."""

    @staticmethod
    def _run_debug_with_loader(exc: BaseException) -> tuple[int, str]:
        """Run debug with loader."""
        from model_protocol_lab.core import Fixture, Message, Split
        from model_protocol_lab.fixtures import split_to_json_text

        split = Split(
            search=(Fixture(id="s1", group="g1", messages=(Message("user", "hi"),), expected="x"),),
            held_out=(
                Fixture(id="h1", group="g2", messages=(Message("user", "q"),), expected="a"),
            ),
        )
        with TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "ds.json"
            dataset.write_text(split_to_json_text(split), encoding="utf-8")
            err = io.StringIO()
            with (
                mock.patch("model_protocol_lab.debug.attach_tokenizer", side_effect=exc),
                redirect_stderr(err),
            ):
                code = run_debug(
                    argparse.Namespace(dataset=dataset, model=Path(tmp) / "ckpt", script=None)
                )
        return code, err.getvalue()

    def test_hostile_value_error_fixed_code_no_bytes(self) -> None:
        """Test hostile value error fixed code no bytes."""
        code, err = self._run_debug_with_loader(ValueError(EVIL))
        self.assertEqual(code, 1)
        self.assertEqual(err, "debug: tokenizer_load_failed\n")
        self.assertNotIn("\x1b", err)

    def test_non_value_error_no_traceback(self) -> None:
        """Test non value error no traceback."""
        # Previously an uncaught traceback; now the fixed code, exit 1.
        code, err = self._run_debug_with_loader(RuntimeError(EVIL))
        self.assertEqual(code, 1)
        self.assertEqual(err, "debug: tokenizer_load_failed\n")
        self.assertNotIn("Traceback", err)
        self.assertNotIn("\x1b", err)

    def test_os_error_fixed_code(self) -> None:
        """Test os error fixed code."""
        code, err = self._run_debug_with_loader(OSError(EVIL))
        self.assertEqual(code, 1)
        self.assertEqual(err, "debug: tokenizer_load_failed\n")

    def test_onecmd_error_path_strips_control_chars(self) -> None:
        """Test onecmd error path strips control chars."""
        from model_protocol_lab.core import Fixture, Message, Split

        split = Split(
            search=(Fixture(id="s1", group="g1", messages=(Message("user", "hi"),), expected="x"),),
            held_out=(
                Fixture(id="h1", group="g2", messages=(Message("user", "q"),), expected="a"),
            ),
        )
        buf = io.StringIO()
        shell = DebugShell(
            split,
            dataset_label="test",
            tokenizer=None,
            stdin=io.StringIO(""),
            stdout=buf,
        )

        def boom(line: str) -> bool:
            """Boom."""
            raise RuntimeError(EVIL)

        shell._dispatch = boom  # type: ignore[method-assign]
        shell.onecmd(".evil")
        out = buf.getvalue()
        self.assertIn("error:", out)
        self.assertNotIn("\x1b", out)
        self.assertNotIn("\x07", out)
        # The message body survived sanitization (not swallowed).
        self.assertIn("parse failed", out)


class DiffBackendsLoaderSanitizationTests(unittest.TestCase):
    """Diff backends loader sanitization tests."""

    def test_hostile_parse_error_fixed_code(self) -> None:
        """Test hostile parse error fixed code."""
        args = _diff_cli_args()
        parser = argparse.ArgumentParser(prog="mpl diff-backends")
        err = io.StringIO()
        with (
            mock.patch(
                "model_protocol_lab.doctor._build_primary_tokenizer",
                side_effect=ValueError(EVIL),
            ),
            redirect_stderr(err),
        ):
            code = run_diff_backends(args, parser)
        self.assertEqual(code, 1)
        self.assertEqual(err.getvalue(), "diff-backends: tokenizer_load_failed\n")

    def test_non_value_error_no_traceback(self) -> None:
        """Test non value error no traceback."""
        args = _diff_cli_args()
        parser = argparse.ArgumentParser(prog="mpl diff-backends")
        err = io.StringIO()
        with (
            mock.patch(
                "model_protocol_lab.doctor._build_primary_tokenizer",
                side_effect=RuntimeError(EVIL),
            ),
            redirect_stderr(err),
        ):
            code = run_diff_backends(args, parser)
        self.assertEqual(code, 1)
        self.assertEqual(err.getvalue(), "diff-backends: tokenizer_load_failed\n")
        self.assertNotIn("Traceback", err.getvalue())


if __name__ == "__main__":
    unittest.main()
