"""Release-policy verifier contracts: exact-SHA evidence or rejection.

The bundles are recorded fixtures in the exact reduced shape the
GitHub API returns and tools/release_policy.py consumes (workflow
runs reduced to id/status/conclusion/head_sha; CI jobs to
name/status/conclusion). No test touches the network: acceptance and
every rejection mode are decided by the pure evaluation core, and the
CLI is exercised through its recorded-evidence mode.
"""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from tools.release_policy import REQUIRED_CI_JOB, evaluate_evidence, main

SHA = "0123456789abcdef0123456789abcdef01234567"
OTHER_SHA = "fedcba9876543210fedcba9876543210fedcba98"


def _run(run_id, conclusion="success", status="completed", head_sha=SHA):
    return {
        "id": run_id,
        "status": status,
        "conclusion": conclusion,
        "head_sha": head_sha,
    }


def _bundle(**overrides):
    bundle = {
        "sha": SHA,
        "expected_version": "0.2.0",
        "version_at_sha": "0.2.0",
        "runs": {"ci": _run(101), "secret_scan": _run(102), "codeql": _run(103)},
        "ci_jobs": [
            {
                "name": "lint and typecheck (no optional deps)",
                "status": "completed",
                "conclusion": "success",
            },
            {"name": REQUIRED_CI_JOB, "status": "completed", "conclusion": "success"},
        ],
    }
    bundle.update(overrides)
    return bundle


def _reasons(bundle):
    return evaluate_evidence(bundle)


class EvaluateEvidenceTests(unittest.TestCase):
    """The pure policy core: one test per acceptance/rejection mode."""

    def test_accepts_complete_exact_sha_evidence(self):
        """Purpose: a complete, successful, exact-SHA bundle is accepted.
        Inputs: the recorded all-success bundle. Outputs: no reasons.
        """
        self.assertEqual(_reasons(_bundle()), [])

    def test_rejects_absent_run(self):
        """Purpose: a workflow with no run for the SHA rejects as absent.
        Inputs: secret_scan run None. Outputs: an absent reason naming it.
        """
        bundle = _bundle()
        bundle["runs"]["secret_scan"] = None
        reasons = _reasons(bundle)
        self.assertTrue(
            any(r.startswith("absent:") and "secret_scan" in r for r in reasons),
            reasons,
        )

    def test_rejects_failed_run(self):
        """Purpose: a failed CI run rejects as failed.
        Inputs: ci conclusion failure. Outputs: a failed reason.
        """
        bundle = _bundle()
        bundle["runs"]["ci"] = _run(101, conclusion="failure")
        self.assertTrue(any(r.startswith("failed:") for r in _reasons(bundle)))

    def test_rejects_cancelled_run(self):
        """Purpose: a cancelled CodeQL run rejects as cancelled.
        Inputs: codeql conclusion cancelled. Outputs: a cancelled reason.
        """
        bundle = _bundle()
        bundle["runs"]["codeql"] = _run(103, conclusion="cancelled")
        self.assertTrue(any(r.startswith("cancelled:") for r in _reasons(bundle)))

    def test_rejects_skipped_run(self):
        """Purpose: a skipped secret-scan run rejects as skipped.
        Inputs: secret_scan conclusion skipped. Outputs: a skipped reason.
        """
        bundle = _bundle()
        bundle["runs"]["secret_scan"] = _run(102, conclusion="skipped")
        self.assertTrue(any(r.startswith("skipped:") for r in _reasons(bundle)))

    def test_rejects_incomplete_run(self):
        """Purpose: a still-running workflow rejects as incomplete.
        Inputs: ci status in_progress, conclusion None.
        Outputs: an incomplete reason.
        """
        bundle = _bundle()
        bundle["runs"]["ci"] = _run(101, conclusion=None, status="in_progress")
        self.assertTrue(any(r.startswith("incomplete:") for r in _reasons(bundle)))

    def test_rejects_wrong_sha_run(self):
        """Purpose: evidence bound to another commit rejects as wrong-sha.
        Inputs: ci run head_sha OTHER_SHA. Outputs: a wrong-sha reason.
        """
        bundle = _bundle()
        bundle["runs"]["ci"] = _run(101, head_sha=OTHER_SHA)
        self.assertTrue(any(r.startswith("wrong-sha:") for r in _reasons(bundle)))

    def test_rejects_missing_audit_job(self):
        """Purpose: a CI run without the audit job rejects as absent.
        Inputs: ci_jobs without REQUIRED_CI_JOB.
        Outputs: an absent reason naming the job.
        """
        bundle = _bundle(
            ci_jobs=[{"name": "tests", "status": "completed", "conclusion": "success"}]
        )
        reasons = _reasons(bundle)
        self.assertTrue(
            any(r.startswith("absent:") and REQUIRED_CI_JOB in r for r in reasons),
            reasons,
        )

    def test_rejects_failed_audit_job(self):
        """Purpose: a failed audit job rejects even when its run succeeded.
        Inputs: audit job conclusion failure. Outputs: job-not-success.
        """
        bundle = _bundle()
        bundle["ci_jobs"][1] = {
            "name": REQUIRED_CI_JOB,
            "status": "completed",
            "conclusion": "failure",
        }
        self.assertTrue(any(r.startswith("job-not-success:") for r in _reasons(bundle)))

    def test_rejects_unavailable_job_list(self):
        """Purpose: an unreadable CI job list rejects as absent.
        Inputs: ci_jobs None. Outputs: an absent reason.
        """
        self.assertTrue(any(r.startswith("absent:") for r in _reasons(_bundle(ci_jobs=None))))

    def test_rejects_wrong_version(self):
        """Purpose: a tree version differing from the claim rejects.
        Inputs: version_at_sha 0.1.0 vs expected 0.2.0.
        Outputs: a wrong-version reason.
        """
        reasons = _reasons(_bundle(version_at_sha="0.1.0"))
        self.assertTrue(any(r.startswith("wrong-version:") for r in reasons), reasons)

    def test_rejects_unreadable_version(self):
        """Purpose: an unreadable tree version rejects as absent.
        Inputs: version_at_sha None. Outputs: an absent reason.
        """
        self.assertTrue(
            any(r.startswith("absent:") for r in _reasons(_bundle(version_at_sha=None)))
        )


class CliRecordedEvidenceTests(unittest.TestCase):
    """The CLI against recorded evidence files (no network)."""

    def _write_bundle(self, directory, bundle):
        path = Path(directory) / "evidence.json"
        path.write_text(json.dumps(bundle), encoding="utf-8")
        return path

    def test_cli_accepts_recorded_bundle(self):
        """Purpose: the CLI accepts a complete recorded bundle.
        Inputs: evidence file + matching --sha/--version.
        Outputs: exit 0 and an accept receipt naming the evidence file.
        """
        with tempfile.TemporaryDirectory() as temp:
            path = self._write_bundle(temp, _bundle())
            out = io.StringIO()
            with redirect_stdout(out):
                code = main(["--sha", SHA, "--version", "0.2.0", "--evidence-json", str(path)])
            self.assertEqual(code, 0)
            receipt = json.loads(out.getvalue())
            self.assertEqual(receipt["verdict"], "accept")
            self.assertTrue(receipt["evidence_source"].startswith("evidence-file:"))
            self.assertEqual(receipt["runs"], {"ci": 101, "secret_scan": 102, "codeql": 103})

    def test_cli_rejects_bundle_for_other_sha(self):
        """Purpose: a bundle recorded for another commit rejects.
        Inputs: --sha OTHER_SHA against the SHA bundle.
        Outputs: exit 1 with a wrong-sha reason.
        """
        with tempfile.TemporaryDirectory() as temp:
            path = self._write_bundle(temp, _bundle())
            out = io.StringIO()
            with redirect_stdout(out):
                code = main(
                    [
                        "--sha",
                        OTHER_SHA,
                        "--version",
                        "0.2.0",
                        "--evidence-json",
                        str(path),
                    ]
                )
            self.assertEqual(code, 1)
            receipt = json.loads(out.getvalue())
            self.assertEqual(receipt["verdict"], "reject")
            self.assertTrue(any(r.startswith("wrong-sha:") for r in receipt["reasons"]))

    def test_cli_rejects_failed_bundle(self):
        """Purpose: the CLI rejects when recorded evidence failed.
        Inputs: bundle with a failed secret-scan run. Outputs: exit 1.
        """
        bundle = _bundle()
        bundle["runs"]["secret_scan"] = _run(102, conclusion="failure")
        with tempfile.TemporaryDirectory() as temp:
            path = self._write_bundle(temp, bundle)
            out = io.StringIO()
            with redirect_stdout(out):
                code = main(["--sha", SHA, "--version", "0.2.0", "--evidence-json", str(path)])
            self.assertEqual(code, 1)
            receipt = json.loads(out.getvalue())
            self.assertTrue(any(r.startswith("failed:") for r in receipt["reasons"]))


if __name__ == "__main__":
    unittest.main()
