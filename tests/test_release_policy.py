"""Release-policy verifier contracts: exact-SHA evidence or rejection.

The bundles are recorded fixtures in the exact reduced shape the
GitHub API returns and tools/release_policy.py consumes (workflow
runs reduced to id/status/conclusion/head_sha/workflow; CI jobs to
name/status/conclusion/run_id — the workflow file and the job's
run_id are the bindings the API path itself uses, so a recorded
bundle must carry them too). No test touches the network: acceptance
and every rejection mode are decided by the pure evaluation core,
and the CLI is exercised through its recorded-evidence mode.
"""

import base64
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from tools import release_policy
from tools.release_policy import REQUIRED_CI_JOB, evaluate_evidence, main

SHA = "0123456789abcdef0123456789abcdef01234567"
OTHER_SHA = "fedcba9876543210fedcba9876543210fedcba98"


def _run(run_id, conclusion="success", status="completed", head_sha=SHA, workflow="ci.yml"):
    return {
        "id": run_id,
        "status": status,
        "conclusion": conclusion,
        "head_sha": head_sha,
        "workflow": workflow,
    }


def _job(name, run_id=101, conclusion="success", status="completed"):
    return {"name": name, "status": status, "conclusion": conclusion, "run_id": run_id}


def _bundle(**overrides):
    bundle = {
        "sha": SHA,
        "expected_version": "0.2.0",
        "version_at_sha": "0.2.0",
        "runs": {
            "ci": _run(101),
            "secret_scan": _run(102, workflow="gitleaks.yml"),
            "codeql": _run(103, workflow="codeql.yml"),
        },
        "ci_jobs": [
            _job("lint and typecheck (no optional deps)"),
            _job(REQUIRED_CI_JOB),
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
        bundle["runs"]["codeql"] = _run(103, conclusion="cancelled", workflow="codeql.yml")
        self.assertTrue(any(r.startswith("cancelled:") for r in _reasons(bundle)))

    def test_rejects_skipped_run(self):
        """Purpose: a skipped secret-scan run rejects as skipped.
        Inputs: secret_scan conclusion skipped. Outputs: a skipped reason.
        """
        bundle = _bundle()
        bundle["runs"]["secret_scan"] = _run(102, conclusion="skipped", workflow="gitleaks.yml")
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

    def test_rejects_substituted_ci_run_id_t07(self):
        """Purpose (T07): a stale/foreign CI run ID substituted into a
        recorded bundle rejects, because the CI job list is still bound
        to the run it was fetched for.
        Inputs: ci run id 999 while every CI job records run_id 101.
        Outputs: an unbound reason naming the mismatch.
        """
        bundle = _bundle()
        bundle["runs"]["ci"] = _run(999)
        reasons = _reasons(bundle)
        self.assertTrue(any(r.startswith("unbound:") for r in reasons), reasons)

    def test_rejects_job_list_recorded_for_another_run(self):
        """Purpose: a CI job list fetched for a different run rejects
        as unbound even when the run entries are untouched.
        Inputs: all CI jobs record run_id 555, ci run id is 101.
        Outputs: an unbound reason.
        """
        bundle = _bundle()
        bundle["ci_jobs"] = [
            _job("lint and typecheck (no optional deps)", run_id=555),
            _job(REQUIRED_CI_JOB, run_id=555),
        ]
        reasons = _reasons(bundle)
        self.assertTrue(any(r.startswith("unbound:") for r in reasons), reasons)

    def test_rejects_run_from_wrong_workflow(self):
        """Purpose: a run produced by a different workflow file than
        the one its evidence key requires rejects as wrong-workflow.
        Inputs: secret_scan run recorded with workflow docs.yml.
        Outputs: a wrong-workflow reason naming secret_scan.
        """
        bundle = _bundle()
        bundle["runs"]["secret_scan"] = _run(102, workflow="docs.yml")
        reasons = _reasons(bundle)
        self.assertTrue(
            any(r.startswith("wrong-workflow:") and "secret_scan" in r for r in reasons),
            reasons,
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
        bundle["runs"]["secret_scan"] = _run(102, conclusion="failure", workflow="gitleaks.yml")
        with tempfile.TemporaryDirectory() as temp:
            path = self._write_bundle(temp, bundle)
            out = io.StringIO()
            with redirect_stdout(out):
                code = main(["--sha", SHA, "--version", "0.2.0", "--evidence-json", str(path)])
            self.assertEqual(code, 1)
            receipt = json.loads(out.getvalue())
            self.assertTrue(any(r.startswith("failed:") for r in receipt["reasons"]))

    def test_cli_rejects_substituted_run_id_t07(self):
        """Purpose (T07): the lane's executed attack — a genuine bundle
        whose CI run ID is replaced by another run's ID — rejects
        through the CLI, because the job list stays bound to the
        original run.
        Inputs: recorded bundle with ci run id 45358577365 substituted.
        Outputs: exit 1 and an unbound reason in the receipt.
        """
        bundle = _bundle()
        bundle["runs"]["ci"] = _run(45358577365)
        with tempfile.TemporaryDirectory() as temp:
            path = self._write_bundle(temp, bundle)
            out = io.StringIO()
            with redirect_stdout(out):
                code = main(["--sha", SHA, "--version", "0.2.0", "--evidence-json", str(path)])
            self.assertEqual(code, 1)
            receipt = json.loads(out.getvalue())
            self.assertEqual(receipt["verdict"], "reject")
            self.assertTrue(any(r.startswith("unbound:") for r in receipt["reasons"]), receipt)


class CliMalformedEvidenceTests(unittest.TestCase):
    """T28: malformed recorded evidence fails closed with exit 2.

    Every malformed class — unreadable file, bad JSON, wrong types,
    missing keys, hostile values — must end in the tool's documented
    "evidence could not be obtained" exit 2 with a named reason on
    stderr, never a traceback and never an accept.
    """

    def _run_cli(self, path):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["--sha", SHA, "--version", "0.2.0", "--evidence-json", str(path)])
        return code, out.getvalue(), err.getvalue()

    def _assert_fails_closed(self, path):
        code, _out, err = self._run_cli(path)
        self.assertEqual(code, 2, err)
        self.assertTrue(err.startswith("release-policy:"), err)
        self.assertNotIn("Traceback", err)

    def test_missing_evidence_file_fails_closed(self):
        """Purpose: a nonexistent evidence path exits 2, no traceback.
        Inputs: a path that does not exist. Outputs: exit 2.
        """
        with tempfile.TemporaryDirectory() as temp:
            self._assert_fails_closed(Path(temp) / "absent.json")

    def test_evidence_directory_fails_closed(self):
        """Purpose: a directory as the evidence path exits 2.
        Inputs: the temp directory itself. Outputs: exit 2.
        """
        with tempfile.TemporaryDirectory() as temp:
            self._assert_fails_closed(Path(temp))

    def test_invalid_utf8_evidence_fails_closed(self):
        """Purpose: non-UTF-8 evidence bytes exit 2, no traceback.
        Inputs: raw 0xff bytes. Outputs: exit 2.
        """
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "evidence.json"
            path.write_bytes(b"\xff\xfe{")
            self._assert_fails_closed(path)

    def test_malformed_bundle_sweep_fails_closed(self):
        """Purpose: a hostile sweep over the bundle grammar — bad JSON,
        non-object roots, missing keys, wrong types, bool-as-int,
        missing binding fields — each exits 2 with a named reason.
        Inputs: the payload list below. Outputs: exit 2 for every one.
        """
        valid = _bundle()

        def mutated(fn):
            bundle = json.loads(json.dumps(valid))
            fn(bundle)
            return json.dumps(bundle)

        payloads = {
            "not json": "this is not json {",
            "empty file": "",
            "json list root": "[]",
            "json scalar root": "42",
            "json null root": "null",
            "empty object": "{}",
            "missing runs": mutated(lambda b: b.pop("runs")),
            "missing ci_jobs": mutated(lambda b: b.pop("ci_jobs")),
            "missing version_at_sha": mutated(lambda b: b.pop("version_at_sha")),
            "sha wrong type": mutated(lambda b: b.update(sha=42)),
            "runs wrong type": mutated(lambda b: b.update(runs=[])),
            "run wrong type": mutated(lambda b: b["runs"].update(ci="run-101")),
            "run id wrong type": mutated(lambda b: b["runs"]["ci"].update(id="101")),
            "run id bool": mutated(lambda b: b["runs"]["ci"].update(id=True)),
            "run missing id": mutated(lambda b: b["runs"]["ci"].pop("id")),
            "run missing workflow": mutated(lambda b: b["runs"]["ci"].pop("workflow")),
            "run workflow wrong type": mutated(lambda b: b["runs"]["ci"].update(workflow=7)),
            "run head_sha wrong type": mutated(lambda b: b["runs"]["ci"].update(head_sha=42)),
            "ci_jobs wrong type": mutated(lambda b: b.update(ci_jobs={})),
            "job wrong type": mutated(lambda b: b.update(ci_jobs=["audit"])),
            "job missing run_id": mutated(lambda b: b["ci_jobs"][0].pop("run_id")),
            "job run_id wrong type": mutated(lambda b: b["ci_jobs"][0].update(run_id="101")),
            "job run_id bool": mutated(lambda b: b["ci_jobs"][0].update(run_id=True)),
            "version_at_sha wrong type": mutated(lambda b: b.update(version_at_sha=2.0)),
            "deeply nested json": "[" * 2000 + "]" * 2000,
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "evidence.json"
            for name, payload in payloads.items():
                with self.subTest(payload=name):
                    path.write_text(payload, encoding="utf-8")
                    self._assert_fails_closed(path)


class BuildBundleTests(unittest.TestCase):
    """The emit side: build_bundle records the binding fields.

    The GitHub API is replaced by a recorded-payload stub keyed on the
    URL, so the bundle build_bundle produces can be checked for the
    workflow identity on every run and the run_id on every CI job —
    the fields the evaluation core binds recorded evidence with.
    """

    def _payloads(self):
        def run_payload(run_id):
            return {
                "workflow_runs": [
                    {
                        "id": run_id,
                        "status": "completed",
                        "conclusion": "success",
                        "head_sha": SHA,
                        "created_at": "2026-01-01T00:00:00Z",
                    }
                ]
            }

        init_b64 = base64.b64encode(b'__version__ = "0.2.0"\n').decode("ascii")
        return {
            "workflows/ci.yml/runs": run_payload(101),
            "workflows/gitleaks.yml/runs": run_payload(102),
            "workflows/codeql.yml/runs": run_payload(103),
            "actions/runs/101/jobs": {
                "jobs": [
                    {
                        "name": "lint and typecheck (no optional deps)",
                        "status": "completed",
                        "conclusion": "success",
                        "run_id": 101,
                    },
                    {
                        "name": REQUIRED_CI_JOB,
                        "status": "completed",
                        "conclusion": "success",
                        "run_id": 101,
                    },
                ]
            },
            "contents/model_protocol_lab/__init__.py": {"content": init_b64},
        }

    def test_build_bundle_records_bindings_and_evaluates_clean(self):
        """Purpose: build_bundle's output carries the workflow file on
        each run and the run_id on each job, and the recorded bundle
        the rehearsal consumes evaluates to an accept.
        Inputs: the recorded API payloads above.
        Outputs: workflow/run_id fields present; no rejection reasons.
        """
        payloads = self._payloads()

        def fake_api_get(url, token):
            for marker, payload in payloads.items():
                if marker in url:
                    return payload
            raise AssertionError(f"unexpected API URL: {url}")

        original = release_policy._api_get
        release_policy._api_get = fake_api_get
        try:
            bundle = release_policy.build_bundle("owner/repo", SHA, "0.2.0", None)
        finally:
            release_policy._api_get = original
        self.assertEqual(bundle["runs"]["ci"]["workflow"], "ci.yml")
        self.assertEqual(bundle["runs"]["secret_scan"]["workflow"], "gitleaks.yml")
        self.assertEqual(bundle["runs"]["codeql"]["workflow"], "codeql.yml")
        self.assertTrue(all(job["run_id"] == 101 for job in bundle["ci_jobs"]))
        self.assertEqual(evaluate_evidence(bundle), [])


if __name__ == "__main__":
    unittest.main()
