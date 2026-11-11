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
                        "head_sha": str} | None},
         "ci_jobs": [{"name": str, "status": str,
                      "conclusion": str | None}] | None}
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
            bundle = json.loads(args.evidence_json.read_text(encoding="utf-8"))
            if bundle.get("sha") != args.sha:
                print(
                    json.dumps(
                        {
                            "verdict": "reject",
                            "sha": args.sha,
                            "reasons": [
                                (
                                    "wrong-sha: evidence bundle is bound to "
                                    f"{bundle.get('sha')}, not {args.sha}"
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
