"""Cycle 228 — package fixes: CLI test-hook hardening and the
contamination sweep budget.

Pins for security-audit-2 finding C-05 and bug-hunt finding F5:

* The shipped CLI's signal-ready test hook
  (``MODEL_PROTOCOL_LAB_TEST_SIGNAL_READY_FILE``) writes its marker
  with O_EXCL: it can no longer truncate a pre-existing file, while
  a fresh marker path still receives the marker (C-05).
* ``contamination_report`` runs under a wall-clock budget in the
  ``core.template_diff`` style. ``_bounded_ratio`` is exactly
  difflib's ratio whenever it completes; an exhausted sweep reports
  only completely evaluated held-out fixtures and carries
  ``budget_exhausted: True`` (absent on completed sweeps, so the
  long-standing exact report shape is unchanged), and the CLI
  warning surfaces the truncation instead of letting a partial
  prefix read as a completed clean sweep (F5).
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import random
import shutil
import tempfile
import time
import unittest
from difflib import SequenceMatcher
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest import mock

from model_protocol_lab import backends, cli, contamination, datasets, reporting
from model_protocol_lab.contamination import _bounded_ratio, contamination_report
from model_protocol_lab.core import Fixture, Generation, Message


def _fx(fid: str, user: str, expected: str = "expected") -> Fixture:
    """One user-message fixture."""
    return Fixture(fid, (Message("user", user),), expected, f"group-{fid}")


class TestBoundedRatio(unittest.TestCase):
    """F5: the bounded ratio is difflib's ratio whenever it completes."""

    def test_matches_difflib_on_a_battery(self) -> None:
        """Purpose: Prove exact equivalence with SequenceMatcher.ratio
        Inputs: Empty/identical/disjoint/fragmented/unicode pairs
        Outputs: Identical floats from both implementations.
        """
        rng = random.Random(7)  # noqa: S311 -- deterministic test seed
        random_text = " ".join(f"w{rng.randrange(50)}" for _ in range(300))
        random_text2 = " ".join(f"w{rng.randrange(50)}" for _ in range(300))
        cases = [
            ("", ""),
            ("", "abc"),
            ("abc", ""),
            ("abc", "abc"),
            ("hello world", "hello there world"),
            ("a" * 500, "a" * 499 + "b"),
            ("ab" * 300, "ba" * 300),
            (random_text, random_text2),
            ("ＡＢＣ unicode", "abc unicode"),
        ]
        for left, right in cases:
            want = SequenceMatcher(None, left, right, autojunk=False).ratio()
            got = _bounded_ratio(left, right, time.monotonic() + 60)
            self.assertEqual(got, want, (left[:20], right[:20]))

    def test_past_deadline_returns_none(self) -> None:
        """Purpose: Fail the pair closed when the deadline already passed
        Inputs: A deadline in the past
        Outputs: None, not a guessed score.
        """
        self.assertIsNone(_bounded_ratio("some text", "other text", time.monotonic() - 1))


class TestContaminationBudget(unittest.TestCase):
    """F5: the sweep is deadline-bound and says so when it stops."""

    def test_zero_deadline_exhausts_immediately(self) -> None:
        """Purpose: Pin the exhausted report shape
        Inputs: Non-empty splits with a zero budget
        Outputs: Empty pairs plus budget_exhausted, key present.
        """
        report = contamination_report([_fx("s1", "alpha")], [_fx("h1", "alpha")], _deadline_s=0)
        self.assertEqual(report, {"worst_tier": "clean", "pairs": [], "budget_exhausted": True})

    def test_completed_sweep_shape_unchanged(self) -> None:
        """Purpose: Pin the long-standing exact shape on completed sweeps
        Inputs: One duplicate pair, generous budget
        Outputs: The legacy dict exactly, with no budget key.
        """
        report = contamination_report([_fx("s1", "alpha beta")], [_fx("h1", "alpha beta")])
        self.assertNotIn("budget_exhausted", report)
        self.assertEqual(report["worst_tier"], "duplicate")
        self.assertEqual(len(report["pairs"]), 1)

    def test_exhaustion_keeps_only_completed_fixtures(self) -> None:
        """Purpose: Pin prefix semantics under a stubbed clock
        Inputs: A flagged first held fixture, then the clock jumps
        Outputs: Only the first fixture's pair, plus the budget flag.
        """
        # Call accounting for the implementation under test: one call
        # sets the deadline, one checks it before the pair, and the
        # identical pair's ratio completes after a single queue pop
        # (one more check) -- three zeros finish fixture h1; the next
        # call, h2's pre-pair check, jumps past the deadline.
        calls = iter([0.0] * 3 + [1e9] * 1000)
        fake_time = SimpleNamespace(monotonic=lambda: next(calls))
        search = [_fx("s1", "alpha beta gamma")]
        held = [_fx("h1", "alpha beta gamma"), _fx("h2", "delta epsilon zeta")]
        with mock.patch.object(contamination, "time", fake_time):
            report = contamination_report(search, held, _deadline_s=10.0)
        self.assertIs(report.get("budget_exhausted"), True)
        self.assertEqual([pair["held_out"] for pair in report["pairs"]], ["h1"])
        self.assertEqual(report["worst_tier"], "duplicate")

    def test_lane_a_dataset_is_bounded_and_flagged(self) -> None:
        """Purpose: Reproducer for the unbounded sweep (lane A, F5)
        Inputs: The within-caps 20+20 fixture dataset, 2 s budget
        Outputs: Returns promptly with budget_exhausted set.
        """
        rng = random.Random(42)  # noqa: S311 -- deterministic test seed

        def rand_text(nwords: int) -> str:
            """Random-word text."""
            return " ".join(f"w{rng.randrange(500)}" for _ in range(nwords))

        def fixture(i: int, group: str) -> dict[str, object]:
            """One dataset fixture dict."""
            return {
                "id": f"{group}{i}",
                "group": f"g{group}{i}",
                "messages": [{"role": "user", "content": rand_text(400)}],
                "expected": "a",
            }

        payload = {
            "search": [fixture(i, "s") for i in range(20)],
            "held_out": [fixture(i, "h") for i in range(20)],
        }
        split = datasets.parse_dataset(json.dumps(payload))
        start = time.perf_counter()
        report = contamination_report(split.search, split.held_out, _deadline_s=2.0)
        elapsed = time.perf_counter() - start
        self.assertIs(report.get("budget_exhausted"), True)
        # Budget 2 s plus at most one in-flight pair on this host class;
        # the unbounded sweep of this dataset exceeds 25 s.
        self.assertLess(elapsed, 15)

    def test_cli_warning_surfaces_exhaustion(self) -> None:
        """Purpose: Keep a partial prefix from reading as a clean sweep
        Inputs: An exhausted report with no flagged pairs
        Outputs: A stderr warning naming the budget stop.
        """
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            cli._warn_contamination(
                "inspect", {"worst_tier": "clean", "pairs": [], "budget_exhausted": True}
            )
        self.assertIn("partial prefix", stderr.getvalue())
        # And a completed clean report stays silent.
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            cli._warn_contamination("inspect", {"worst_tier": "clean", "pairs": []})
        self.assertEqual(stderr.getvalue(), "")

    def test_cli_warning_truncates_after_ten_pairs(self) -> None:
        """Purpose: Hold the documented ten-pair print cap on the warning
        Inputs: A contamination report with twelve flagged pairs
        Outputs: Ten pair lines plus the "2 more (see JSON)" trailer.
        """
        pair = {"held_out": "h1", "search": "s1", "ratio": 0.9, "jaccard": 0.8, "tier": "high"}
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            cli._warn_contamination("inspect", {"worst_tier": "high", "pairs": [pair] * 12})
        text = stderr.getvalue()
        self.assertIn("(12 pair(s))", text)
        self.assertEqual(text.count("held_out h1 ~ search s1"), 10)
        self.assertIn("... and 2 more (see JSON)", text)

    def test_cli_warning_skips_non_dict_pairs(self) -> None:
        """Purpose: Keep a malformed pair entry from breaking the warning
        Inputs: A pairs list mixing dict entries with a non-dict entry
        Outputs: Only the dict pairs print; the header counts all entries.
        """
        pair = {"held_out": "h1", "search": "s1", "ratio": 0.9, "jaccard": 0.8, "tier": "high"}
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            cli._warn_contamination(
                "inspect", {"worst_tier": "high", "pairs": [pair, "malformed", pair]}
            )
        text = stderr.getvalue()
        self.assertIn("(3 pair(s))", text)
        self.assertEqual(text.count("held_out h1 ~ search s1"), 2)

    def test_report_line_surfaces_exhaustion(self) -> None:
        """Purpose: Keep the HTML hygiene line honest under exhaustion
        Inputs: An exhausted clean-prefix report dict
        Outputs: A "partial" line, never the completed-clean claim.
        """
        line = reporting._contamination_line(
            {"worst_tier": "clean", "pairs": [], "budget_exhausted": True}
        )
        self.assertTrue(line.startswith("partial:"))
        clean = reporting._contamination_line({"worst_tier": "clean", "pairs": []})
        self.assertTrue(clean.startswith("clean:"))


class _BenignBackend:
    """Fake LocalHFBackend: completes immediately, no signals."""

    fingerprint = "cycle228-c05"
    eos_token_id = None

    def __init__(self, model: str, device: object = None, max_context: object = None) -> None:
        """Accept the real constructor's arguments."""
        self.calls = 0

    @staticmethod
    def encode(text: str) -> tuple[int, ...]:
        """Encode."""
        return (1, 2, 3)

    @staticmethod
    def decode(token_ids: tuple[int, ...]) -> str:
        """Decode."""
        return "yo"

    def generate(
        self, input_ids: tuple[int, ...], *, max_new_tokens: int, seed: int, cancel: Event
    ) -> Generation:
        """Generate."""
        self.calls += 1
        return Generation(tuple(range(max_new_tokens)), "limit")


class TestSignalReadyHook(unittest.TestCase):
    """C-05: the shipped hook cannot truncate a pre-existing file."""

    def setUp(self) -> None:
        """Scratch directory plus a minimal valid dataset."""
        self.tmp = Path(tempfile.mkdtemp(prefix="c228-c05-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.fixture = self.tmp / "fixture.json"
        self.fixture.write_text(
            json.dumps(
                {
                    "search": [
                        {
                            "id": "s0",
                            "group": "g1",
                            "messages": [{"role": "user", "content": "hi"}],
                            "expected": "yo",
                        }
                    ],
                    "held_out": [
                        {
                            "id": "h1",
                            "group": "g2",
                            "messages": [{"role": "user", "content": "hey"}],
                            "expected": "yo",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    def _run_cli(self, marker: Path) -> int:
        """Run the real cli.main run path with the hook env var set."""
        argv = [
            "run",
            "--model",
            "ckpt-dir",
            "--dataset",
            str(self.fixture),
            "--family",
            "chatml",
            "--boundary-ids",
            "198,199",
            "--output",
            str(self.tmp / "report"),
            "--seed",
            "7",
        ]
        env = {**os.environ, "MODEL_PROTOCOL_LAB_TEST_SIGNAL_READY_FILE": str(marker)}
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch.object(backends, "LocalHFBackend", _BenignBackend),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            return cli.main(argv)

    def test_preexisting_file_is_never_truncated(self) -> None:
        """Purpose: Prove the shipped binary cannot destroy file content
        Inputs: A sentinel file at the hook's env-var path, a full run
        Outputs: Sentinel bytes unchanged after the run.
        """
        sentinel = self.tmp / "sentinel.txt"
        sentinel.write_bytes(b"SENTINEL-DO-NOT-TRUNCATE")
        self._run_cli(sentinel)
        self.assertEqual(sentinel.read_bytes(), b"SENTINEL-DO-NOT-TRUNCATE")

    def test_fresh_marker_path_still_gets_the_marker(self) -> None:
        """Purpose: Positive control -- the hook still signals readiness
        Inputs: A marker path that does not exist yet, a full run
        Outputs: The marker file exists with the readiness content.
        """
        marker = self.tmp / "ready.marker"
        self.assertFalse(marker.exists())
        self._run_cli(marker)
        self.assertEqual(marker.read_text(encoding="utf-8"), "ready")


if __name__ == "__main__":
    unittest.main()
