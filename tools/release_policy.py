"""Release-policy verifier: bind a candidate commit to its quality evidence.

Purpose: given an exact commit SHA and the version a release would
claim, verify from the GitHub API that the required workflow evidence
exists for THAT SHA and succeeded — the CI run, the Secret scan run,
the CodeQL run, and the dependency/static security audit job inside
the CI run — and that the tree at that SHA carries the claimed
version. Any gap rejects the candidate: absent, incomplete, failed,
cancelled or skipped evidence, evidence bound to a different SHA, a
missing or unsuccessful audit job, or a version mismatch. On
acceptance it prints a JSON receipt naming the exact runs that vouch
for the exact commit.

This is the non-publishing fence of audit finding RR-06 (W1). The
tool never publishes, tags or uploads anything; the publishing path
(.github/workflows/release.yml) is unchanged and remains owner-gated.
The release-readiness workflow runs this verifier after re-running
the full gate at the candidate SHA.

Who uses it: the maintainer, on the morning a release is proposed, to
get one receipt binding a commit to its evidence instead of eyeballing
four workflow pages — and to have the absence of evidence fail closed
rather than pass unnoticed.

Recorded-evidence mode (--evidence-json, rehearsal only) consumes a
bundle in the shape build_bundle produces. The API path binds a run
ID to its commit by fetching it from the workflow's own run list
filtered to that SHA, and binds the CI job list to the CI run by
fetching it under that run's ID; a recorded bundle carries the same
bindings as data — each run records the workflow file that produced
it, each CI job records the run_id it was fetched for — and the
evaluator checks them, so a substituted run ID or a job list recorded
for another run rejects with a named reason. A bundle that cannot be
read, parsed or matched to that shape at all is not evidence: it
fails closed with exit 2, never a traceback and never an accept. The
bundle's authenticity still rests on its provenance (the receipt
names the evidence file as its source); the release path itself runs
in API mode, where the bindings come from the API directly.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

API_ROOT = "https://api.github.com"
DEFAULT_REPO = "strmt7/model-protocol-lab"
USER_AGENT = "model-protocol-lab-release-policy/1.0"

# Evidence keys mapped to the workflow files that produce them.
REQUIRED_WORKFLOWS: dict[str, str] = {
    "ci": "ci.yml",
    "secret_scan": "gitleaks.yml",
    "codeql": "codeql.yml",
}
# The CI job whose success the audit (RR-06) names explicitly.
REQUIRED_CI_JOB = "dependency and static security audit"

_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
_VERSION_PATTERN = re.compile(r"^__version__ = \"([^\"]+)\"", re.MULTILINE)


class PolicyError(RuntimeError):
    """Evidence could not be obtained at all (distinct from rejection)."""


def evaluate_evidence(bundle: dict[str, Any]) -> list[str]:
    """Return the named rejection reasons for an evidence bundle.

    An empty list accepts the candidate. The bundle shape is the one
    build_bundle produces and the test fixtures record:

        {"sha": str, "expected_version": str,
         "version_at_sha": str | None,
         "runs": {key: {"id": int, "status": str,
                        "conclusion": str | None,
                        "head_sha": str,
                        "workflow": str} | None},
         "ci_jobs": [{"name": str, "status": str,
                      "conclusion": str | None,
                      "run_id": int}] | None}

    Two bindings mirror how the API path obtains the evidence: a
    run's recorded workflow must be the workflow file its key stands
    for, and every CI job's recorded run_id must be the bundle's CI
    run — a job list fetched for any other run does not vouch for
    this one.
    """
    reasons: list[str] = []
    sha = bundle["sha"]
    runs = bundle["runs"]
    for key in REQUIRED_WORKFLOWS:
        run = runs.get(key)
        if run is None:
            reasons.append(f"absent: no {key} workflow run for {sha}")
            continue
        if run.get("head_sha") != sha:
            reasons.append(
                f"wrong-sha: {key} run {run.get('id')} is bound to {run.get('head_sha')}, not {sha}"
            )
            continue
        if run.get("workflow") != REQUIRED_WORKFLOWS[key]:
            reasons.append(
                f"wrong-workflow: {key} run {run.get('id')} was produced by "
                f"{run.get('workflow')}, not {REQUIRED_WORKFLOWS[key]}"
            )
            continue
        if run.get("status") != "completed":
            reasons.append(
                f"incomplete: {key} run {run.get('id')} has status "
                f"{run.get('status')}, not completed"
            )
            continue
        conclusion = run.get("conclusion")
        if conclusion != "success":
            label = {
                "failure": "failed",
                "cancelled": "cancelled",
                "skipped": "skipped",
            }.get(str(conclusion), "not-success")
            reasons.append(f"{label}: {key} run {run.get('id')} concluded {conclusion}")
    jobs = bundle.get("ci_jobs")
    if jobs is None:
        reasons.append("absent: CI job list for the CI run is unavailable")
    else:
        ci_run = runs.get("ci")
        if isinstance(ci_run, dict):
            ci_run_id = ci_run.get("id")
            for job in jobs:
                if job.get("run_id") != ci_run_id:
                    reasons.append(
                        f"unbound: CI job {job.get('name')!r} was recorded for "
                        f"run {job.get('run_id')}, not the CI run {ci_run_id} "
                        "in this bundle"
                    )
        audit_jobs = [j for j in jobs if j.get("name") == REQUIRED_CI_JOB]
        if not audit_jobs:
            reasons.append(f"absent: CI job {REQUIRED_CI_JOB!r} not in the CI run")
        else:
            job = audit_jobs[0]
            if job.get("status") != "completed":
                reasons.append(
                    f"incomplete: CI job {REQUIRED_CI_JOB!r} has status "
                    f"{job.get('status')}, not completed"
                )
            elif job.get("conclusion") != "success":
                reasons.append(
                    f"job-not-success: CI job {REQUIRED_CI_JOB!r} concluded {job.get('conclusion')}"
                )
    version_at_sha = bundle.get("version_at_sha")
    if version_at_sha is None:
        reasons.append(f"absent: package version at {sha} could not be read")
    elif version_at_sha != bundle["expected_version"]:
        reasons.append(
            f"wrong-version: expected {bundle['expected_version']}, but the "
            f"tree at {sha} carries {version_at_sha}"
        )
    return reasons


def _api_get(url: str, token: str | None) -> Any:
    # S310: every URL this tool opens is built from the fixed https
    # GitHub API root below; assert it so the invariant is checked,
    # not assumed.
    if not url.startswith(f"{API_ROOT}/"):
        raise PolicyError(f"refusing to open a non-GitHub-API URL: {url}")
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": USER_AGENT,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)  # noqa: S310  # nosec B310 -- URL asserted to the fixed https GitHub host above
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310  # nosec B310 -- URL asserted to the fixed https GitHub host above
            return json.load(response)
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        raise PolicyError(f"GitHub API read failed for {url}: {exc}") from exc


def _latest_run_for_sha(runs: list[dict[str, Any]], sha: str) -> dict[str, Any] | None:
    """The most recently created run bound to exactly this SHA, if any."""
    bound = [r for r in runs if r.get("head_sha") == sha]
    if not bound:
        return None
    return max(bound, key=lambda r: (str(r.get("created_at", "")), int(r.get("id", 0))))


def build_bundle(repo: str, sha: str, expected_version: str, token: str | None) -> dict[str, Any]:
    """Fetch the evidence bundle for a candidate from the GitHub API."""
    runs: dict[str, Any] = {}
    ci_run_id: int | None = None
    for key, workflow_file in REQUIRED_WORKFLOWS.items():
        payload = _api_get(
            f"{API_ROOT}/repos/{repo}/actions/workflows/{workflow_file}/runs"
            f"?head_sha={sha}&per_page=100",
            token,
        )
        run = _latest_run_for_sha(list(payload.get("workflow_runs", [])), sha)
        if run is None:
            runs[key] = None
            continue
        runs[key] = {
            "id": run.get("id"),
            "status": run.get("status"),
            "conclusion": run.get("conclusion"),
            "head_sha": run.get("head_sha"),
            "workflow": workflow_file,
        }
        if key == "ci":
            ci_run_id = int(run["id"])
    ci_jobs: list[dict[str, Any]] | None = None
    if ci_run_id is not None:
        payload = _api_get(
            f"{API_ROOT}/repos/{repo}/actions/runs/{ci_run_id}/jobs?per_page=100",
            token,
        )
        ci_jobs = [
            {
                "name": job.get("name"),
                "status": job.get("status"),
                "conclusion": job.get("conclusion"),
                "run_id": job.get("run_id"),
            }
            for job in payload.get("jobs", [])
        ]
    version_at_sha: str | None = None
    payload = _api_get(
        f"{API_ROOT}/repos/{repo}/contents/model_protocol_lab/__init__.py?ref={sha}",
        token,
    )
    content = payload.get("content")
    if isinstance(content, str):
        text = base64.b64decode(content).decode("utf-8", errors="replace")
        match = _VERSION_PATTERN.search(text)
        if match:
            version_at_sha = match.group(1)
    return {
        "sha": sha,
        "expected_version": expected_version,
        "version_at_sha": version_at_sha,
        "runs": runs,
        "ci_jobs": ci_jobs,
    }


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_bundle_shape(bundle: Any) -> None:
    """Raise PolicyError unless bundle matches the documented shape.

    Recorded evidence is untrusted input: every structural defect —
    a missing key, a wrong type, a binding field of the wrong type —
    means the evidence could not be obtained in the form the policy
    evaluates, so it is a PolicyError (exit 2), never a crash and
    never a policy verdict on a shape the evaluator cannot read.
    """

    def malformed(defect: str) -> PolicyError:
        return PolicyError(f"malformed evidence bundle: {defect}")

    if not isinstance(bundle, dict):
        raise malformed(f"top level must be a JSON object, got {type(bundle).__name__}")
    for key in ("sha", "runs", "ci_jobs", "version_at_sha"):
        if key not in bundle:
            raise malformed(f"missing key {key!r}")
    if not isinstance(bundle["sha"], str):
        raise malformed("'sha' must be a string")
    if "expected_version" in bundle and not isinstance(bundle["expected_version"], str):
        raise malformed("'expected_version' must be a string")
    if bundle["version_at_sha"] is not None and not isinstance(bundle["version_at_sha"], str):
        raise malformed("'version_at_sha' must be a string or null")
    runs = bundle["runs"]
    if not isinstance(runs, dict):
        raise malformed("'runs' must be an object")
    for key, run in runs.items():
        if run is None:
            continue
        if not isinstance(run, dict):
            raise malformed(f"runs.{key} must be an object or null")
        if key not in REQUIRED_WORKFLOWS:
            continue
        if not _is_int(run.get("id")):
            raise malformed(f"runs.{key}.id is missing or not an integer")
        for field in ("status", "conclusion", "head_sha"):
            if run.get(field) is not None and not isinstance(run.get(field), str):
                raise malformed(f"runs.{key}.{field} must be a string or null")
        if not isinstance(run.get("workflow"), str):
            raise malformed(f"runs.{key}.workflow is missing or not a string")
    jobs = bundle["ci_jobs"]
    if jobs is not None:
        if not isinstance(jobs, list):
            raise malformed("'ci_jobs' must be a list or null")
        for index, job in enumerate(jobs):
            if not isinstance(job, dict):
                raise malformed(f"ci_jobs[{index}] must be an object")
            if not isinstance(job.get("name"), str):
                raise malformed(f"ci_jobs[{index}].name is missing or not a string")
            for field in ("status", "conclusion"):
                if job.get(field) is not None and not isinstance(job.get(field), str):
                    raise malformed(f"ci_jobs[{index}].{field} must be a string or null")
            if not _is_int(job.get("run_id")):
                raise malformed(f"ci_jobs[{index}].run_id is missing or not an integer")


def _load_evidence_bundle(path: Path) -> dict[str, Any]:
    """Read and shape-check a recorded evidence bundle.

    Any failure — an unreadable file, invalid JSON, or a document
    that does not match the shape build_bundle produces — raises
    PolicyError, so the CLI fails closed with exit 2 (evidence could
    not be obtained) instead of crashing on garbage input.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise PolicyError(f"evidence bundle could not be read: {path}: {exc}") from exc
    try:
        bundle: dict[str, Any] = json.loads(text)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise PolicyError(f"evidence bundle is not valid JSON: {path}: {exc}") from exc
    _validate_bundle_shape(bundle)
    return bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify that a candidate commit carries successful, "
        "exact-SHA quality evidence (RR-06 readiness fence; never publishes)."
    )
    parser.add_argument("--sha", required=True, help="full 40-hex commit SHA")
    parser.add_argument("--version", required=True, help="version the release would claim")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="owner/name")
    parser.add_argument(
        "--evidence-json",
        type=Path,
        default=None,
        help="read a recorded evidence bundle from this file instead of "
        "the GitHub API (dry-run/rehearsal; the receipt names the source)",
    )
    parser.add_argument("--out", type=Path, default=None, help="also write the receipt here")
    args = parser.parse_args(argv)
    if not _SHA_PATTERN.fullmatch(args.sha):
        parser.error("--sha must be a full 40-character hex commit SHA")
    source = "api"
    try:
        if args.evidence_json is not None:
            bundle = _load_evidence_bundle(args.evidence_json)
            if bundle["sha"] != args.sha:
                print(
                    json.dumps(
                        {
                            "verdict": "reject",
                            "sha": args.sha,
                            "reasons": [
                                (
                                    "wrong-sha: evidence bundle is bound to "
                                    f"{bundle['sha']}, not {args.sha}"
                                )
                            ],
                        },
                        indent=2,
                    )
                )
                return 1
            bundle["expected_version"] = args.version
            source = f"evidence-file:{args.evidence_json}"
        else:
            token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
            bundle = build_bundle(args.repo, args.sha, args.version, token)
    except PolicyError as exc:
        print(f"release-policy: {exc}", file=sys.stderr)
        return 2
    reasons = evaluate_evidence(bundle)
    receipt = {
        "verdict": "accept" if not reasons else "reject",
        "sha": args.sha,
        "version": args.version,
        "evidence_source": source,
        "runs": {key: (run or {}).get("id") for key, run in bundle["runs"].items()},
        "reasons": reasons,
    }
    text = json.dumps(receipt, indent=2)
    print(text)
    if args.out is not None:
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0 if not reasons else 1


if __name__ == "__main__":
    raise SystemExit(main())
