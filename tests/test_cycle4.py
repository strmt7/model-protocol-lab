"""Cycle 4: first-run UX polish regressions.

- The `run` command's stdout now names the written report files (the
  first-run walk showed the bare decision JSON left new users guessing
  where the report went).
- The README's stated test count is pinned to the discovered suite size so
  the Cycle 3 drift (README said 491 while the suite was 505) cannot recur.
"""

import importlib.util
import io
import json
import re
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from model_protocol_lab.cli import main

HAS_HF = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "transformers", "safetensors", "tokenizers")
)

REPO_ROOT = Path(__file__).resolve().parent.parent


@unittest.skipUnless(HAS_HF, "the 'hf' extra is not installed")
class RunStdoutReportPathsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_adapter_integration import build_tiny_gpt2

        cls._tmp = TemporaryDirectory()
        cls.addClassCleanup(cls._tmp.cleanup)
        cls.model_dir = Path(cls._tmp.name) / "tiny"
        cls.model_dir.mkdir()
        build_tiny_gpt2(cls.model_dir)
        cls.dataset = REPO_ROOT / "examples" / "fixtures.json"

    def test_run_stdout_names_existing_report_files(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        output = Path(tmp.name) / "report"
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = main(
                [
                    "run",
                    "--model",
                    str(self.model_dir),
                    "--dataset",
                    str(self.dataset),
                    "--family",
                    "chatml",
                    "--boundary-ids",
                    "62",
                    "--max-new-tokens",
                    "8",
                    "--max-context",
                    "64",
                    "--device",
                    "cpu",
                    "--output",
                    str(output),
                ]
            )
        self.assertEqual(code, 2)  # random weights: honest abstention
        payload = json.loads(stdout.getvalue())
        self.assertIn(payload["decision"], ("no_candidate", "ambiguous", "held_out_failed"))
        for key, filename in (("results", "results.json"), ("report", "report.html")):
            self.assertIn(key, payload)
            path = Path(payload[key])
            self.assertTrue(path.is_file(), f"{key} path does not exist: {path}")
            self.assertEqual(path.name, filename)
            self.assertEqual(path.parent, output)


class ReadmeConsistencyTests(unittest.TestCase):
    def test_readme_test_count_matches_discovered_suite(self):
        text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        match = re.search(r"full test suite passes \((\d+) tests", text)
        self.assertIsNotNone(match, "README status line lost its test-count claim")
        # Count via pytest collection (the suite's actual runner). unittest
        # discovery alone misses pytest-style function tests, so it
        # undercounts once such files exist in the suite.
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--collect-only",
                "-q",
                "-p",
                "no:cacheprovider",
                "tests",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=180,
        )
        collected = None
        for line in proc.stdout.splitlines():
            m = re.search(r"(\d+) tests collected", line)
            if m:
                collected = int(m.group(1))
        self.assertIsNotNone(collected, "could not parse pytest collection count")
        self.assertEqual(int(match.group(1)), collected)


class MutationFoundTests(unittest.TestCase):
    def test_error_reason_detected_by_equality_not_identity(self):
        # Cosmic-ray survivor (Cycle 4, core.py `evaluate`): mutating
        # `result.reason == "error"` to `is "error"` survived because every
        # scripted backend used the interned literal. A backend may hand
        # back an equal-but-not-identical string, so the error branch must
        # use equality, never identity.
        from model_protocol_lab.core import Generation
        from tests.test_core import FakeBackend, case, ids, run_case

        reason = "".join(["err", "or"])
        self.assertEqual(reason, "error")
        backend = FakeBackend(Generation(ids("ok|"), reason, None))
        report = run_case(backend, cases=(case("first"),))
        observation = report.observations[0]
        self.assertEqual(
            (report.status, report.planned_count, report.attempted_count), ("failed", 1, 1)
        )
        self.assertEqual(observation.error, "backend_reported_error")
        self.assertEqual(observation.raw_reason, reason)
        self.assertFalse(report.aggregate_eligible)

    def test_evaluate_default_seed_is_zero(self):
        # Cosmic-ray survivor (Cycle 4, core.py `evaluate`): the
        # `seed: int = 0` default mutated to 1 survived because every test
        # passed `seed` explicitly. The default is a reproducibility
        # contract: omitting it must derive exactly the request seeds that
        # seed=0 derives.
        from model_protocol_lab.core import evaluate
        from tests.test_core import BASE, RAW, FakeBackend, case, ids

        def derived_seeds(**kwargs):
            backend = FakeBackend()
            evaluate(
                backend,
                BASE,
                (case("first"),),
                boundary=ids("|"),
                max_new_tokens=len(RAW),
                **kwargs,
            )
            return [record[2] for record in backend.generated]

        self.assertEqual(derived_seeds(), derived_seeds(seed=0))
        # Guard against a vacuous pass: seed=1 must actually differ, proving
        # the backend received a derived seed at all.
        self.assertNotEqual(derived_seeds(), derived_seeds(seed=1))


if __name__ == "__main__":
    unittest.main()
