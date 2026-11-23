"""W8 (RR-06): release.yml consumes the exact-SHA readiness receipt.

Audit finding RR-06 (docs/STABLE_RELEASE_LEDGER.md): the publication
path did not enforce the quality gates. The verifier has existed
since W1 (tools/release_policy.py) and release-readiness.yml runs
it on demand, but .github/workflows/release.yml never invoked it,
so a version tag would have published a commit whose CI, Secret
scan and CodeQL evidence for that exact SHA had failed — or never
existed. These pins read the shipped workflow as data (the
cycle-208/215 pattern): a readiness job must run the verifier
against the tagged commit and the tag's version before anything
is built, the build job must need it, and the job's grants must
be exactly the two reads the verifier performs. The verifier's
own verdict logic is pinned by tests/test_release_policy.py
against recorded fixtures; no test here executes a workflow or
touches the network.
"""

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "release.yml"
WORKFLOW = WORKFLOW_PATH.read_text(encoding="utf-8")

_SENTINEL = object()


def _load_workflow():
    """Parse release.yml with yaml.safe_load (the cycle-215 idiom)."""
    import yaml

    return yaml.safe_load(WORKFLOW)


def _needs(job):
    """Normalise a job's needs (string or list) to a list of names."""
    needs = job.get("needs", [])
    return [needs] if isinstance(needs, str) else list(needs)


def _verifier_step(job):
    """Return the step whose run invokes tools/release_policy.py."""
    for step in job.get("steps", []):
        if "tools/release_policy.py" in str(step.get("run", "")):
            return step
    return _SENTINEL


class ReleaseReceiptWiringTests(unittest.TestCase):
    """The release workflow is gated by the RR-06 readiness receipt."""

    def setUp(self) -> None:
        """Parse the workflow once per test."""
        self.doc = _load_workflow()
        self.jobs = self.doc["jobs"]

    def test_readiness_job_runs_verifier_for_tagged_sha_and_version(self) -> None:
        """The readiness job verifies the tagged commit at the tag's version."""
        self.assertIn("readiness", self.jobs)
        step = _verifier_step(self.jobs["readiness"])
        self.assertIsNot(step, _SENTINEL)
        run = str(step["run"])
        # The tagged commit is the candidate; the tag name (minus
        # the leading v) is the version the release would claim —
        # the same binding the build job's --expect-version uses.
        self.assertIn('--sha "$GITHUB_SHA"', run)
        self.assertIn('--version "${GITHUB_REF_NAME#v}"', run)
        # API mode is the verifier's default; it authenticates with
        # the workflow's own read-only token, never a stored secret.
        self.assertEqual(step["env"]["GITHUB_TOKEN"], "${{ secrets.GITHUB_TOKEN }}")
        self.assertNotIn("--evidence-json", run)

    def test_build_cannot_run_without_the_receipt(self) -> None:
        """Build needs readiness; publish still needs build (transitively gated)."""
        self.assertIn("readiness", _needs(self.jobs["build"]))
        self.assertIn("build", _needs(self.jobs["publish"]))

    def test_readiness_job_grants_only_the_verifiers_reads(self) -> None:
        """The job holds exactly contents: read and actions: read."""
        # The verifier reads workflow runs and jobs (actions: read)
        # and the candidate's __init__.py (contents: read); checkout
        # needs contents: read too. Anything more would hand the
        # pre-publication gate a grant the fence never uses.
        self.assertEqual(
            self.jobs["readiness"]["permissions"],
            {"contents": "read", "actions": "read"},
        )

    def test_receipt_is_written_and_uploaded(self) -> None:
        """The accepted receipt is kept as the release-receipt artifact."""
        step = _verifier_step(self.jobs["readiness"])
        self.assertIsNot(step, _SENTINEL)
        self.assertIn("--out release-receipt.json", str(step["run"]))
        uploads = [
            s
            for s in self.jobs["readiness"]["steps"]
            if str(s.get("uses", "")).startswith("actions/upload-artifact@")
        ]
        self.assertEqual(len(uploads), 1)
        self.assertEqual(uploads[0]["with"]["name"], "release-receipt")
        self.assertEqual(uploads[0]["with"]["path"], "release-receipt.json")

    def test_readiness_job_carries_the_frozen_posture(self) -> None:
        """Named job, integer timeout, credential-free checkout."""
        job = self.jobs["readiness"]
        self.assertEqual(job.get("name"), "verify exact-SHA readiness receipt")
        self.assertIsInstance(job.get("timeout-minutes"), int)
        self.assertGreater(job["timeout-minutes"], 0)
        checkouts = [
            s for s in job["steps"] if str(s.get("uses", "")).startswith("actions/checkout@")
        ]
        self.assertEqual(len(checkouts), 1)
        self.assertIs(checkouts[0]["with"]["persist-credentials"], False)


if __name__ == "__main__":
    unittest.main()
