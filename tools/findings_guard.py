"""Enforce the findings thresholds for one audited commit (cycle 217).

Stratos's spec, implemented literally:

- FAIL when open CodeQL code findings (GitHub code-scanning alerts
  with ``tool.name == "CodeQL"`` and ``state == "open"``, paginated)
  number more than two. Scorecard posture checks are not code
  findings and are excluded by the tool-name filter.
- FAIL when even one DeepSource finding exists for the audited
  commit, read through the ported ``tools/deepsource_inventory.py``
  (its ``total_count`` for a complete, source-matched manifest).
- A missing DeepSource credential is NOT-CONFIGURED: an explicit
  failure, never a silent pass. The guard cannot report "zero
  findings" for a signal it could not read.

The script is standard-library only so the guard workflow needs no
pip install. GitHub authentication comes from ``GITHUB_TOKEN``;
DeepSource authentication follows the inventory tool
(``DEEPSOURCE_TOKEN`` first, else the official CLI login).
No token value is ever printed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

CODEQL_LIMIT = 2
REPOSITORY = "strmt7/model-protocol-lab"


def count_codeql_findings(
    fetch: Callable[[int], list[dict[str, object]]],
) -> tuple[int, list[dict[str, object]]]:
    """Paginate the open code-scanning alerts; count CodeQL only."""
    findings: list[dict[str, object]] = []
    page = 1
    while True:
        batch = fetch(page)
        if not isinstance(batch, list):
            raise ValueError("code-scanning API returned a non-list page")
        for alert in batch:
            if not isinstance(alert, dict):
                raise ValueError("code-scanning API returned a non-object alert")
            tool = alert.get("tool")
            if isinstance(tool, dict) and tool.get("name") == "CodeQL":
                findings.append(alert)
        if len(batch) < 100:
            break
        page += 1
        if page > 1000:
            raise ValueError("code-scanning alert pagination exceeded its budget")
    return len(findings), findings


def github_page_fetcher(repo: str, token: str) -> Callable[[int], list[dict[str, object]]]:
    """Build a page fetcher over the code-scanning alerts API."""

    def fetch(page: int) -> list[dict[str, object]]:
        """Fetch."""
        url = (
            f"https://api.github.com/repos/{repo}/code-scanning/alerts"
            f"?state=open&per_page=100&page={page}"
        )
        request = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "User-Agent": "model-protocol-lab-Findings-Guard/1.0",
            },
        )
        if not url.startswith("https://api.github.com/"):
            raise ValueError("code-scanning API URL left the fixed GitHub host")
        try:
            # S310: the URL's host (api.github.com) and scheme (https)
            # are constants above; only the path carries the
            # operator-supplied repository slug (CI passes
            # GITHUB_REPOSITORY), so no attacker-controlled scheme or
            # host can reach urlopen here. The startswith assertion
            # above enforces that claim; it is not prose alone.
            with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310  # nosec B310 -- URL asserted to the fixed https GitHub host above
                if response.status != 200:
                    raise ValueError(f"code-scanning API HTTP {response.status}")
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise ValueError(f"code-scanning API HTTP {error.code}") from None

    return fetch


def deepsource_status(
    repository: str,
    commit: str,
    output_root: Path,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[str, int | None]:
    """Run the ported inventory; return (state, total_count).

    States: "ok" (complete manifest), "findings" (complete manifest
    with total_count >= 1), "not-configured" (no credential),
    "error" (any other inventory failure, with its diagnostic).
    The caller decides the exit code; this function never invents
    a zero.
    """
    if (
        not os.environ.get("DEEPSOURCE_TOKEN")
        and not (Path.home() / ".deepsource" / "config.toml").exists()
    ):
        return "not-configured", None
    output = output_root / f"deepsource-{commit[:12]}"
    tool = Path(__file__).resolve().parent / "deepsource_inventory.py"
    proc = runner(
        [
            sys.executable,
            str(tool),
            "--repository",
            repository,
            "--commit",
            commit,
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if proc.returncode != 0:
        # Fail closed on an inventory failure even if a previous run
        # left a complete-looking manifest in this output directory.
        return "error", None
    manifest_path = output / "manifest.json"
    if not manifest_path.exists():
        return "error", None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        return "error", None
    total = manifest.get("total_count")
    if type(total) is not int:
        return "error", None
    return ("findings" if total >= 1 else "ok"), total


def evaluate(
    repository: str,
    commit: str,
    fetch: Callable[[int], list[dict[str, object]]],
    output_root: Path,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, object]:
    """Run both legs; return the report (exit code under "exit")."""
    codeql_count, codeql_alerts = count_codeql_findings(fetch)
    codeql_pass = codeql_count <= CODEQL_LIMIT
    ds_state, ds_total = deepsource_status(repository, commit, output_root, runner)
    ds_pass = ds_state == "ok"
    failures = []
    if not codeql_pass:
        failures.append(f"CodeQL code findings {codeql_count} exceed limit {CODEQL_LIMIT}")
    if ds_state == "findings":
        failures.append(f"DeepSource findings {ds_total} exist (limit 0)")
    elif ds_state == "not-configured":
        failures.append("DeepSource NOT-CONFIGURED: no DEEPSOURCE_TOKEN or CLI login")
    elif ds_state == "error":
        failures.append("DeepSource inventory failed; see its retained manifest")
    return {
        "repository": repository,
        "commit": commit,
        "codeql_open": codeql_count,
        "codeql_rules": sorted(
            {
                str(a.get("rule", {}).get("id"))
                for a in codeql_alerts
                if isinstance(a.get("rule"), dict)
            }
        ),
        "codeql_pass": codeql_pass,
        "deepsource_state": ds_state,
        "deepsource_total": ds_total,
        "deepsource_pass": ds_pass,
        "pass": codeql_pass and ds_pass,
        "failures": failures,
    }


def main() -> int:
    """Evaluate the audited commit; print the report; exit 0 iff pass."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=REPOSITORY)
    parser.add_argument("--commit", default=os.environ.get("GITHUB_SHA", ""))
    parser.add_argument("--output-root", type=Path, default=Path("out/findings-guard"))
    args = parser.parse_args()
    token = os.environ.get("GITHUB_TOKEN", "")
    if not args.commit or not token:
        print(
            "findings-guard: --commit (or GITHUB_SHA) and GITHUB_TOKEN are required",
            file=sys.stderr,
        )
        return 2
    try:
        report = evaluate(
            args.repository,
            args.commit,
            github_page_fetcher(args.repository, token),
            args.output_root,
        )
    except (ValueError, OSError) as error:
        print(f"findings-guard: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["pass"]:
        for failure in report["failures"]:
            print(f"findings-guard: FAIL: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
