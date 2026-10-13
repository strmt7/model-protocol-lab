"""End-to-end CLI tests: real subprocess entry points and full `run` flows.

The `run` tests use a tiny locally-generated GPT-2 (no downloads); they
exercise argument validation, backend wiring, report writing and exit codes
as one path.
"""

import importlib.util
import io
import json
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from model_protocol_lab.cli import main

HAS_HF = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "transformers", "safetensors", "tokenizers")
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_module(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "model_protocol_lab", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )


class ModuleEntryTests(unittest.TestCase):
    def test_module_catalog_via_subprocess(self):
        proc = run_module("catalog")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        entries = json.loads(proc.stdout)
        self.assertEqual({e["name"] for e in entries}, {"chatml", "llama3", "gemma"})

    def test_module_no_args_exits_two(self):
        proc = run_module()
        self.assertEqual(proc.returncode, 2)

    def test_console_script_entry_point(self):
        script = Path(sys.executable).parent / "model-protocol-lab"
        if not script.exists():
            self.skipTest("console script not installed in this environment")
        proc = subprocess.run(
            [str(script), "catalog"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)


@unittest.skipUnless(HAS_HF, "the 'hf' extra is not installed")
class RunCommandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_adapter_integration import build_tiny_gpt2

        cls._tmp = TemporaryDirectory()
        cls.addClassCleanup(cls._tmp.cleanup)
        cls.model_dir = Path(cls._tmp.name) / "tiny"
        cls.model_dir.mkdir()
        build_tiny_gpt2(cls.model_dir)
        cls.dataset = REPO_ROOT / "examples" / "fixtures.json"

    def _run(self, extra=()):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        output = Path(tmp.name) / "report"
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
                *extra,
            ]
        )
        return code, output

    def test_full_run_writes_report_and_returns_decision_code(self):
        with redirect_stdout(io.StringIO()):
            code, output = self._run()
        # Random weights: no candidate passes; the lab must abstain honestly.
        self.assertEqual(code, 2)
        results = json.loads((output / "results.json").read_text(encoding="utf-8"))
        self.assertIn(results["decision"], ("no_candidate", "ambiguous", "held_out_failed"))
        self.assertIn("candidate_fingerprint", results["search"][0])
        html = (output / "report.html").read_text(encoding="utf-8")
        self.assertIn("Model Protocol Lab", html)
        self.assertNotIn("http://", html)
        self.assertNotIn("https://", html)

    def test_full_run_with_mutations(self):
        with redirect_stdout(io.StringIO()):
            code, output = self._run(("--mutate",))
        self.assertEqual(code, 2)
        results = json.loads((output / "results.json").read_text(encoding="utf-8"))
        self.assertGreater(len(results["search"]), 1)

    def test_run_with_evidence_flag(self):
        with redirect_stdout(io.StringIO()):
            code, output = self._run(("--include-evidence",))
        self.assertEqual(code, 2)
        results = json.loads((output / "results.json").read_text(encoding="utf-8"))
        self.assertTrue(results["raw_evidence_included"])
        self.assertIn("observations", results["search"][0])

    def test_run_rejects_missing_model(self):
        with TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            code = main(
                [
                    "run",
                    "--model",
                    str(Path(tmp) / "missing"),
                    "--dataset",
                    str(self.dataset),
                    "--family",
                    "chatml",
                    "--boundary-ids",
                    "62",
                    "--output",
                    str(Path(tmp) / "out"),
                ]
            )
        self.assertEqual(code, 1)

    def test_keyboard_interrupt_maps_to_130(self):
        with (
            TemporaryDirectory() as tmp,
            patch("model_protocol_lab.cli.run_experiment", side_effect=KeyboardInterrupt),
            redirect_stdout(io.StringIO()),
        ):
            output = Path(tmp) / "report"
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
                    "--max-context",
                    "64",
                    "--output",
                    str(output),
                ]
            )
        self.assertEqual(code, 130)


if __name__ == "__main__":
    unittest.main()
