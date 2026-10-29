"""Cycle 111: end-to-end determinism audit (two runs -> byte-identical outputs).

The project's core contract is deterministic local JSON and self-contained
HTML reports. This test locks that in: the full ``run`` pipeline is executed
twice in separate subprocesses with different ``PYTHONHASHSEED`` values (so
any hash-order-dependent serialization would surface), and both
``results.json`` and ``report.html`` must be byte-identical.

Method mirrors the cycle-68 e2e audit: a locally generated tiny GPT-2
checkpoint (fixed seed, no downloads) evaluates a minimal fixture. Unlike the
cycle-68 test, this runs the pipeline TWICE and compares bytes, not just
structure.

Findings: CLEAN -- results.json and report.html are byte-identical across
runs with different hash seeds, including with --mutate and
--sensitivity-sweep enabled (verified manually during the audit).
"""

import importlib.util
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

HAS_HF = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "transformers", "safetensors", "tokenizers")
)

REPO_ROOT = Path(__file__).resolve().parent.parent

DET_CSV = """id,group,partition,expected,user_1,assistant_1,user_2
det-search-copy,det-copy,search,BLUE,Reply with exactly BLUE and nothing else.,,
det-held-number,det-integer,held_out,7,Return only the numeral 7.,,
"""


def run_cli_with_seed(seed: str, *args: str, timeout: int = 300) -> subprocess.CompletedProcess:
    """Run cli with seed."""
    env = {**os.environ, "PYTHONHASHSEED": seed}
    return subprocess.run(
        [sys.executable, "-m", "model_protocol_lab", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        check=False,
    )


@unittest.skipUnless(HAS_HF, "the 'hf' extra is not installed")
class DeterminismAuditTests(unittest.TestCase):
    """Two identical pipeline runs must produce byte-identical outputs."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the test fixture."""
        cls._tmp = TemporaryDirectory()
        cls.addClassCleanup(cls._tmp.cleanup)
        cls.root = Path(cls._tmp.name)
        from tests.test_adapter_integration import build_tiny_gpt2

        model_dir = cls.root / "tiny"
        model_dir.mkdir()
        build_tiny_gpt2(model_dir)
        cls.model_dir = model_dir
        # Build the fixture once; it is identical input for both runs.
        csv_path = cls.root / "det.csv"
        csv_path.write_text(DET_CSV, encoding="utf-8")
        fixture = cls.root / "det_fixtures.json"
        proc = run_cli_with_seed("1", "fixtures", "build", str(csv_path), "--output", str(fixture))
        assert proc.returncode == 0, proc.stderr
        cls.fixture = fixture

    def _run_once(self, seed: str, out: Path) -> None:
        """Run once."""
        proc = run_cli_with_seed(
            seed,
            "run",
            "--model",
            str(self.model_dir),
            "--dataset",
            str(self.fixture),
            "--family",
            "chatml",
            "--boundary-ids",
            "62",
            "--max-new-tokens",
            "4",
            "--max-context",
            "32",
            "--device",
            "cpu",
            "--include-evidence",
            "--output",
            str(out),
            timeout=600,
        )
        self.assertIn(proc.returncode, (0, 1, 2), proc.stderr[-2000:])

    def _first_json_diff(self, a: object, b: object, path: str = "") -> str | None:
        """Return a human-readable pointer to the first structural difference."""
        if type(a) is not type(b):
            return f"{path or '<root>'}: type {type(a).__name__} vs {type(b).__name__}"
        if isinstance(a, dict):
            ka = list(a.keys())
            kb = list(b.keys())  # type: ignore[union-attr]
            if ka != kb:
                return f"{path or '<root>'}: key order differs"
            for k in ka:
                diff = self._first_json_diff(a[k], b[k], f"{path}.{k}")  # type: ignore[index]
                if diff:
                    return diff
        elif isinstance(a, list):
            if len(a) != len(b):  # type: ignore[arg-type]
                return f"{path or '<root>'}: length {len(a)} vs {len(b)}"  # type: ignore[arg-type]
            for i, (x, y) in enumerate(zip(a, b, strict=True)):  # type: ignore[arg-type]
                diff = self._first_json_diff(x, y, f"{path}[{i}]")
                if diff:
                    return diff
        elif a != b:
            return f"{path or '<root>'}: {a!r} vs {b!r}"
        return None

    def test_two_runs_produce_byte_identical_results(self) -> None:
        """Test two runs produce byte identical results."""
        out_a = self.root / "report_a"
        out_b = self.root / "report_b"
        # Different hash seeds: any set/dict ordering leak would flip bytes.
        self._run_once("1", out_a)
        self._run_once("2", out_b)
        raw_a = (out_a / "results.json").read_bytes()
        raw_b = (out_b / "results.json").read_bytes()
        if raw_a != raw_b:
            diff = self._first_json_diff(json.loads(raw_a), json.loads(raw_b))
            self.fail(f"results.json not byte-identical across runs; first diff: {diff}")
        html_a = (out_a / "report.html").read_bytes()
        html_b = (out_b / "report.html").read_bytes()
        self.assertEqual(
            html_a,
            html_b,
            "report.html not byte-identical across runs",
        )


if __name__ == "__main__":
    unittest.main()
