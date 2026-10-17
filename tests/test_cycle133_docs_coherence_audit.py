"""Cycle 133: docs-coherence audit fixes — regression tests.

Pins two cycle-133 doc/behavior findings:

1. G5: ``python -m model_protocol_lab.cli --help`` must print usage and exit
   0 (the module previously had no ``__main__`` guard and printed nothing).
2. U2: user-facing docs (``docs/*.md`` except the chronological
   ``RESEARCH_LOG.md`` journal, whose dated entries are historical records)
   must not contain the ``mpl `` command shorthand — the installed console
   script is ``model-protocol-lab`` and ``mpl`` does not exist on PATH. The
   debug REPL's literal prompt (``mpl:<family> ...``) is a real in-code
   prompt string, not shorthand, and is out of scope for this check.
"""

from __future__ import annotations

import re
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_MPL_SHORTHAND = re.compile(r"`mpl (?!:)")


class TestCliModuleEntryPoint(unittest.TestCase):
    """`python -m model_protocol_lab.cli --help` prints usage, exit 0."""

    def test_module_help_exits_zero_with_usage(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-m", "model_protocol_lab.cli", "--help"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("model-protocol-lab", proc.stdout)
        self.assertIn("catalog", proc.stdout)

    def test_package_main_help_still_works(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-m", "model_protocol_lab", "--help"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("model-protocol-lab", proc.stdout)


class TestDocsCommandShorthand(unittest.TestCase):
    """User docs reference the real console script name, never `mpl `."""

    def test_no_mpl_shorthand_in_user_docs(self) -> None:
        offenders: list[str] = []
        for path in sorted((REPO_ROOT / "docs").glob("*.md")):
            if path.name == "RESEARCH_LOG.md":
                continue  # chronological journal: dated entries are history
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if _MPL_SHORTHAND.search(line):
                    offenders.append(f"{path.name}:{i}: {line.strip()[:80]}")
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
