"""Shared memoized ``pytest --collect-only`` for the suite's self-checks.

``test_cycle4.test_readme_test_count_matches_discovered_suite`` (README
count pin) and
``test_ci_portability_guard.test_every_test_file_collected_by_pytest``
(every file contributes tests) each spawned a full collection
subprocess (~30-60s each). Both run in the same pytest/unittest session
during the full suite, so a single collection serves both: this module
runs it once per process and caches the (per-file counts, total) pair.

Only successful collections are cached; a failed collection raises
with diagnostics (the callers' retry loops are preserved by re-raising,
so transient failures still retry at the call site -- but a cached
success is never re-run).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

_CACHE: tuple[dict[str, int], int] | None = None


def pytest_collect_counts(repo_root: Path) -> tuple[dict[str, int], int]:
    """Return (per_file_test_counts, total_collected) for the suite.

    Runs ``pytest --collect-only -q -p no:cacheprovider tests/`` once
    per process; subsequent calls reuse the cached result. Raises
    RuntimeError with collection diagnostics after 3 failed attempts.
    """
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    last_diagnostics = ""
    for attempt in range(3):
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests/",
                    "--collect-only",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except subprocess.TimeoutExpired as exc:
            last_diagnostics = f"attempt {attempt + 1}: timed out after 300s: {exc}"
            continue
        if proc.returncode != 0:
            last_diagnostics = (
                f"attempt {attempt + 1}: returncode={proc.returncode}, "
                f"stderr_tail={(proc.stderr or proc.stdout)[-2000:]}"
            )
            continue
        counts: dict[str, int] = {}
        total: int | None = None
        for line in proc.stdout.splitlines():
            m = re.search(r"(\d+) tests collected", line)
            if m:
                total = int(m.group(1))
            # lines look like: tests/test_x.py::TestY::test_z
            if "::" in line:
                fname = line.split("::")[0].split("/")[-1]
                counts[fname] = counts.get(fname, 0) + 1
        if total is not None:
            _CACHE = (counts, total)
            return _CACHE
        last_diagnostics = (
            f"attempt {attempt + 1}: returncode={proc.returncode}, "
            f"no 'N tests collected' line, "
            f"stderr_tail={(proc.stderr or proc.stdout)[-2000:]}"
        )
    raise RuntimeError(
        "pytest --collect-only failed after 3 attempts; last diagnostics: " + last_diagnostics
    )
