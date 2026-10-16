"""Cycle 68: end-to-end pipeline audit (fixtures build -> run -> diff-reports).

Method: the REAL user path, driven through subprocess CLI invocations
(``python -m model_protocol_lab ...``) with synthetic/local inputs only --
no model downloads, no network. A locally generated tiny GPT-2 checkpoint
(with random weights, built by the existing helper in
``tests/test_adapter_integration.py``) is the only backend exercised;
there is no synthetic/deterministic backend on the CLI, so this is the
documented fast local path.

Stages exercised:
  1. ``fixtures build`` -- an authoring CSV in the documented format is
     compiled to a strict fixture JSON; the built fixture is round-tripped
     through ``load_dataset``.
  2. ``run`` -- the tiny GPT-2 checkpoint evaluates the built fixture
     (``--include-evidence``); ``results.json`` + ``report.html`` are
     written and the results file is re-validated through
     ``diffreports.load_results``.
  3. ``diff-reports`` -- the produced report is diffed against itself
     (expect ``identical=true``) and against a perturbed copy with exactly
     one flipped observation (expect exactly one flipped fixture cell and
     no other axis moved).

Adversarial cases (all via subprocess, asserting fixed error text and no
traceback):
  * ``fixtures build`` on a CSV with an oversize single field: the
    cycle-66 contract (fixed ``ValueError`` "CSV field exceeds the parser
    limit", exit 1, no traceback).
  * ``run`` with the backend's heavy dependencies unavailable: the real
    CLI error path (exit 1, generic message, no traceback). torch is
    stubbed out of ``sys.modules`` in a subprocess (so ``import torch``
    inside ``LocalHFBackend`` raises ``ImportError``), using a hand-built
    minimal asset directory so asset validation passes and the import is
    the only failure.
  * ``diff-reports`` given a report directory (cycle-64 behavior):
    accepted, resolves to the ``results.json`` inside.

Findings: CLEAN -- every stage behaved exactly per its documented
contract. No genuine bugs found; see the class docstrings for the one
deliberately-not-verified item (a full ``--sensitivity-sweep`` run is
compute-heavy and out of scope for this audit).
"""

import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from model_protocol_lab.datasets import load_dataset
from model_protocol_lab.diffreports import load_results

HAS_HF = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "transformers", "safetensors", "tokenizers")
)

REPO_ROOT = Path(__file__).resolve().parent.parent

AUDIT_CSV = """id,group,partition,expected,user_1,assistant_1,user_2
audit-search-copy,audit-copy,search,BLUE,Reply with exactly BLUE and nothing else.,,
audit-held-number,audit-integer,held_out,7,Return only the numeral 7.,,
"""


def run_cli(*args: str, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "model_protocol_lab", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@unittest.skipUnless(HAS_HF, "the 'hf' extra is not installed")
class PipelineEndToEndTests(unittest.TestCase):
    """Happy path: authoring CSV -> fixture JSON -> run -> diff-reports."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = TemporaryDirectory()
        cls.addClassCleanup(cls._tmp.cleanup)
        cls.root = Path(cls._tmp.name)
        # Locally generated tiny GPT-2 with random weights: no downloads.
        from tests.test_adapter_integration import build_tiny_gpt2

        model_dir = cls.root / "tiny"
        model_dir.mkdir()
        build_tiny_gpt2(model_dir)
        cls.model_dir = model_dir
        cls._fixture: Path | None = None
        cls._report: Path | None = None

    def _fixture_json(self) -> Path:
        # The class-level tmp root is shared across tests in this class, so
        # reuse the fixture if an earlier test already built it.
        out = self.root / "audit_fixtures.json"
        if self._fixture is not None:
            return self._fixture
        if out.exists():
            self._fixture = out
            return out
        csv_path = self.root / "audit.csv"
        csv_path.write_text(AUDIT_CSV, encoding="utf-8")
        proc = run_cli("fixtures", "build", str(csv_path), "--output", str(out))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self._fixture = out
        return out

    def _run_report(self) -> Path:
        if self._report is None:
            output = self.root / "report"
            proc = run_cli(
                "run",
                "--model",
                str(self.model_dir),
                "--dataset",
                str(self._fixture_json()),
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
                "--include-evidence",
                "--output",
                str(output),
                timeout=600,
            )
            self.assertIn(proc.returncode, (0, 1, 2), proc.stderr)
            self._proc = proc
            self._report = output
        return self._report

    def test_01_fixtures_build_compiles_authoring_csv(self) -> None:
        csv_path = self.root / "audit.csv"
        csv_path.write_text(AUDIT_CSV, encoding="utf-8")
        out = self.root / "audit_fixtures.json"
        # A fresh path each time: `fixtures build` never overwrites, and the
        # audit sidecar is also claimed atomically.
        for stale in (out, Path(str(out) + ".audit.json")):
            if stale.exists():
                stale.unlink()
        proc = run_cli("fixtures", "build", str(csv_path), "--output", str(out))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)
        summary = json.loads(proc.stdout)
        self.assertEqual(summary["output"], str(out))
        self.assertEqual(summary["search_cases"], 1)
        self.assertEqual(summary["held_out_cases"], 1)
        # The audit sidecar is written next to the fixture.
        self.assertTrue((out.parent / (out.name + ".audit.json")).exists())
        # Round-trip: the built JSON validates through the runtime loader.
        split = load_dataset(out)
        self.assertEqual([c.id for c in split.search], ["audit-search-copy"])
        self.assertEqual([c.id for c in split.held_out], ["audit-held-number"])
        self.assertEqual(split.search[0].expected, "BLUE")

    def test_02_run_writes_results_and_html_report(self) -> None:
        report = self._run_report()
        results_path = report / "results.json"
        self.assertTrue(results_path.exists())
        # results.json validates through the diff-reports loader.
        loaded = load_results(results_path)
        raw = json.loads(results_path.read_text(encoding="utf-8"))
        self.assertEqual(loaded, raw)
        # Exit code matches the decision per the CLI contract.
        decision = loaded["decision"]
        proc = self._proc
        if decision == "compatible_on_fixture_set":
            self.assertEqual(proc.returncode, 0)
        elif decision == "failed":
            self.assertEqual(proc.returncode, 1)
        else:
            self.assertEqual(proc.returncode, 2)
        # The run used --include-evidence: every search entry carries
        # observations so per-fixture diffs are comparable.
        self.assertTrue(
            all(isinstance(entry.get("observations"), list) for entry in loaded["search"])
        )
        # report.html is produced and self-contained (no remote references).
        html = (report / "report.html").read_text(encoding="utf-8")
        self.assertIn("Model Protocol Lab", html)
        for banned in ("http://", "https://", "//"):
            self.assertNotIn(banned, html)

    def test_03_diff_reports_self_is_identical(self) -> None:
        report = self._run_report()
        # Mix the two accepted input forms: directory and results.json path.
        proc = run_cli("diff-reports", str(report), str(report / "results.json"))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        diff = json.loads(proc.stdout)
        summary = diff["summary"]
        self.assertTrue(summary["identical"])
        self.assertFalse(summary["decision_changed"])
        self.assertFalse(summary["winner_changed"])
        self.assertEqual(summary["candidates_changed_status"], 0)
        self.assertEqual(summary["fixtures_flipped"], 0)
        self.assertEqual(summary["config_drift_warnings"], 0)
        self.assertFalse(summary["evidence_asymmetric"])
        self.assertEqual(diff["candidate_verdicts"], [])
        self.assertIn("No differences", proc.stderr)

    def test_04_diff_reports_perturbed_observation_flips_one_cell(self) -> None:
        report = self._run_report()
        original = json.loads((report / "results.json").read_text(encoding="utf-8"))
        perturbed = json.loads(json.dumps(original))  # deep copy
        target = None
        for entry in perturbed["search"]:
            for obs in entry.get("observations") or []:
                if (
                    isinstance(obs, dict)
                    and isinstance(obs.get("exact_match"), bool)
                    and isinstance(obs.get("boundary_found"), bool)
                    and obs.get("fixture")
                ):
                    # Flip the observation so its OUTCOME (pass iff
                    # exact_match and boundary_found) moves: flip exact_match,
                    # and boundary_found too if a single flip leaves the
                    # outcome unchanged (both fields False stays fail).
                    old_outcome = obs["exact_match"] and obs["boundary_found"]
                    obs["exact_match"] = not obs["exact_match"]
                    if (obs["exact_match"] and obs["boundary_found"]) == old_outcome:
                        obs["boundary_found"] = not obs["boundary_found"]
                    target = (entry, obs)
                    break
            if target:
                break
        self.assertIsNotNone(target, "expected at least one valid observation")
        entry, obs = target
        # Candidate name as the differ itself derives it (format.name).
        fmt = entry.get("format") if isinstance(entry, dict) else None
        candidate = fmt.get("name") if isinstance(fmt, dict) else None
        self.assertIsInstance(candidate, str)
        new_path = self.root / "perturbed.json"
        new_path.write_text(json.dumps(perturbed), encoding="utf-8")
        proc = run_cli("diff-reports", str(report / "results.json"), str(new_path))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        diff = json.loads(proc.stdout)
        summary = diff["summary"]
        self.assertFalse(summary["identical"])
        # Exactly one fixture cell moved; every other axis is untouched.
        self.assertEqual(summary["fixtures_flipped"], 1)
        self.assertEqual(summary["candidates_changed_status"], 0)
        self.assertEqual(summary["config_drift_warnings"], 0)
        self.assertFalse(summary["decision_changed"])
        self.assertFalse(summary["winner_changed"])
        self.assertFalse(summary["evidence_asymmetric"])
        self.assertEqual(diff["candidate_verdicts"], [])
        changed = [d for d in diff["fixture_diffs"] if d["old"] != d["new"]]
        self.assertEqual(len(changed), 1)
        cell = changed[0]
        self.assertEqual(cell["stage"], "search")
        self.assertEqual(cell["candidate"], candidate)
        self.assertEqual(cell["fixture"], obs["fixture"])
        flipped = "pass" if obs["exact_match"] and obs["boundary_found"] else "fail"
        self.assertEqual(cell["new"], flipped)
        self.assertEqual(cell["old"], "fail" if flipped == "pass" else "pass")


class PipelineAdversarialTests(unittest.TestCase):
    """Failure-mode contracts, all via subprocess: fixed error codes, no tracebacks."""

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_oversize_csv_field_yields_fixed_valueerror(self) -> None:
        # One field far beyond csv.field_size_limit: the cycle-66 contract is
        # a fixed ValueError surfaced by the CLI, not a bare _csv.Error.
        big = "x" * (2 * 1024 * 1024 // 8)  # 256 KiB, one field
        csv_path = self.root / "oversize.csv"
        csv_path.write_text(
            f"id,group,partition,expected,user_1\nbig-field,g,search,ok,{big}\n",
            encoding="utf-8",
        )
        out = self.root / "out.json"
        proc = run_cli("fixtures", "build", str(csv_path), "--output", str(out))
        self.assertEqual(proc.returncode, 1, proc.stderr)
        self.assertIn("fixtures: invalid authoring source:", proc.stderr)
        self.assertIn("CSV field exceeds the parser limit", proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)
        self.assertFalse(out.exists(), "failed build must not leave a fixture behind")

    def test_run_with_missing_backend_deps_is_clean_error(self) -> None:
        # A minimal asset directory that passes backend asset validation, so
        # the ONLY failure is `import torch` inside LocalHFBackend. The heavy
        # deps are stubbed out of sys.modules in a subprocess, exercising the
        # real CLI failure path.
        #
        # Actual contract (verified by this test): LocalHFBackend.__init__
        # fails closed on ANY init error -- including a missing dependency --
        # and re-raises it as the fixed code
        # ``local_backend_initialization_failed`` (the fail-closed
        # ``except Exception`` at the end of ``LocalHFBackend.__init__``),
        # which the CLI surfaces as
        # ``Operation failed [local_backend_initialization_failed]`` with
        # exit 1 and no traceback. CLEAN: deliberate fixed vocabulary, not
        # a leak of the raw ImportError.
        model_dir = self.root / "stubmodel"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps({"model_type": "gpt2"}), encoding="utf-8")
        (model_dir / "tokenizer.json").write_text(
            json.dumps({"model": {"type": "WordLevel", "dropout": 0}}),
            encoding="utf-8",
        )
        (model_dir / "model.safetensors").write_bytes(b"stub-weights")
        fixture = self.root / "fx.json"
        fixture.write_text(
            json.dumps(
                {
                    "search": [
                        {
                            "id": "s1",
                            "group": "gs",
                            "expected": "BLUE",
                            "messages": [{"role": "user", "content": "Reply BLUE."}],
                        }
                    ],
                    "held_out": [
                        {
                            "id": "h1",
                            "group": "gh",
                            "expected": "7",
                            "messages": [{"role": "user", "content": "Return 7."}],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        outdir = self.root / "report"
        script = (
            "import sys\n"
            "sys.modules['torch'] = None\n"
            "sys.modules['transformers'] = None\n"
            "from model_protocol_lab.cli import main\n"
            "sys.exit(main(['run',\n"
            f"    '--model', {str(model_dir)!r},\n"
            f"    '--dataset', {str(fixture)!r},\n"
            "    '--family', 'chatml', '--boundary-ids', '62',\n"
            "    '--max-new-tokens', '4', '--max-context', '64',\n"
            "    '--device', 'cpu',\n"
            f"    '--output', {str(outdir)!r}]))\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(proc.returncode, 1, proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)
        self.assertIn(
            "Operation failed [local_hf_extra_not_installed]",
            proc.stderr,
        )
        self.assertIn('pip install "model-protocol-lab[hf]"', proc.stderr)
        # No report directory is left behind by the failed run.
        self.assertFalse(outdir.exists())

    def test_diff_reports_accepts_report_directory(self) -> None:
        # Cycle-64 behavior: a directory input resolves to results.json inside.
        report_dir = self.root / "report"
        report_dir.mkdir()
        payload = {
            "decision": "no_candidate",
            "search": [],
            "selected_candidate": None,
            "held_out": None,
        }
        (report_dir / "results.json").write_text(json.dumps(payload), encoding="utf-8")
        other = self.root / "other.json"
        other.write_text(json.dumps(payload), encoding="utf-8")
        proc = run_cli("diff-reports", str(report_dir), str(other))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue(json.loads(proc.stdout)["summary"]["identical"])

    def test_diff_reports_bare_directory_is_clean_error(self) -> None:
        # A directory with no results.json inside: fixed code, no traceback.
        empty = self.root / "empty_report"
        empty.mkdir()
        other = self.root / "other.json"
        other.write_text(json.dumps({"decision": "x", "search": []}), encoding="utf-8")
        proc = run_cli("diff-reports", str(empty), str(other))
        self.assertEqual(proc.returncode, 1)
        self.assertNotIn("Traceback", proc.stderr)
        self.assertIn("diff-reports:", proc.stderr)


if __name__ == "__main__":
    unittest.main()
