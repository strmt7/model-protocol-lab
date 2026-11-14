"""W3-6 — the Windows portable workflow keeps its Phase-1 shape.

The workflow is evidence, not decoration: it is the only mechanism
that turns the W3 support matrix's Windows cells (hashed dev-hf
install, `pip check`, native mypy, base-layer collection) from
proxy-proven into CI-executed. These pins hold the workflow to the
scope it claims — one job, Windows, Python 3.12, the CI test job's
exact hashed install route, native mypy, a collect-only base-layer
step — and to what it must never become: the gate. The full suite
and its coverage floor stay Linux-only in ci.yml; a run step here
must never invoke coverage or unittest discovery.
"""

from __future__ import annotations

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "windows.yml"


def _load_yaml(path: Path) -> dict:
    """Load a YAML file (the workflow under audit) with yaml.safe_load."""
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _run_steps(doc: dict) -> list[str]:
    """Return the job's run-step scripts, in order."""
    job = doc["jobs"]["windows-base"]
    return [step["run"] for step in job["steps"] if "run" in step]


class TestWindowsPortableWorkflow(unittest.TestCase):
    """The windows.yml Phase-1 shape, pinned."""

    def setUp(self) -> None:
        """Load the workflow document once per test."""
        self.doc = _load_yaml(WORKFLOW)

    def test_parses_with_expected_name_and_single_job(self) -> None:
        """Purpose: The workflow exists, parses, and carries one job
        Inputs: .github/workflows/windows.yml
        Outputs: Name "Windows portable"; jobs == {windows-base}.
        """
        self.assertEqual(self.doc["name"], "Windows portable")
        self.assertEqual(set(self.doc["jobs"]), {"windows-base"})

    def test_job_platform_timeout_and_matrix(self) -> None:
        """Purpose: The cell is Windows, 3.12 only, and time-bounded
        Inputs: The windows-base job
        Outputs: windows-latest, timeout-minutes 60, matrix ["3.12"].
        """
        job = self.doc["jobs"]["windows-base"]
        self.assertEqual(job["runs-on"], "windows-latest")
        self.assertEqual(job["timeout-minutes"], 60)
        self.assertEqual(job["strategy"]["matrix"]["python-version"], ["3.12"])

    def test_permissions_are_read_only(self) -> None:
        """Purpose: Least privilege — the job needs no write grant
        Inputs: The workflow's top-level permissions
        Outputs: Exactly {contents: read}.
        """
        self.assertEqual(self.doc["permissions"], {"contents": "read"})

    def test_install_steps_are_the_ci_hashed_route(self) -> None:
        """Purpose: Installation is the CI test job's exact route
        Inputs: The job's run steps
        Outputs: pip.txt + dev-hf.txt under --require-hashes, the
            --no-deps editable install, and a pip check step.
        """
        runs = "\n".join(_run_steps(self.doc))
        self.assertIn("--require-hashes -r tools/requirements/pip.txt", runs)
        self.assertIn("--require-hashes -r tools/requirements/dev-hf.txt", runs)
        self.assertIn("--no-deps -e .", runs)
        self.assertIn("pip check", runs)

    def test_native_mypy_and_base_layer_collect_steps(self) -> None:
        """Purpose: The evidence steps are present and collect-only
        Inputs: The job's run steps
        Outputs: A native mypy step and a pytest --collect-only step
            whose file list comes from tools/layer_paths.py --layer base.
        """
        runs = _run_steps(self.doc)
        joined = "\n".join(runs)
        self.assertIn("python -m mypy model_protocol_lab", joined)
        collect = [step for step in runs if "--collect-only" in step]
        self.assertEqual(len(collect), 1)
        self.assertIn("tools/layer_paths.py --layer base", collect[0])
        self.assertIn("pytest", collect[0])

    def test_not_the_gate_guard(self) -> None:
        """Purpose: This workflow must never grow into the gate
        Inputs: The job's run steps
        Outputs: No step invokes coverage or unittest discovery.
        """
        joined = "\n".join(_run_steps(self.doc))
        self.assertNotIn("coverage", joined)
        self.assertNotIn("unittest", joined)


if __name__ == "__main__":
    unittest.main()
