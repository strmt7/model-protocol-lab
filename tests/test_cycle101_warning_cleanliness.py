"""Cycle 101: warning-cleanliness audit.

The full suite (6105 tests) passes with every warning promoted to an
error (``python -W error -m pytest``): no DeprecationWarning,
FutureWarning, ResourceWarning, or any other warning fires anywhere in
the package or its tests — verified 2026-10-16. This file pins that
property with fast, deterministic regression tests so a future change
that introduces a warning (e.g. a deprecated stdlib/dependency API)
fails loudly instead of rotting silently.

The checks run in a subprocess with ``-W error`` because warning
filters are most faithfully applied at interpreter startup.
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PKG = REPO_ROOT / "model_protocol_lab"


def _run_warnfree(code: str) -> subprocess.CompletedProcess[str]:
    """Run warnfree."""
    return subprocess.run(
        [sys.executable, "-W", "error", "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(REPO_ROOT),
        check=False,
    )


class WarningCleanlinessTests(unittest.TestCase):
    """Warning cleanliness tests."""

    def test_all_modules_import_warnfree(self):
        """Test all modules import warnfree."""
        mods = sorted(p.stem for p in PKG.glob("*.py") if p.name != "__init__.py")
        self.assertGreater(len(mods), 10)
        imports = "; ".join(f"import model_protocol_lab.{m}" for m in mods)
        proc = _run_warnfree(f"import model_protocol_lab; {imports}; print('ok')")
        self.assertEqual(
            proc.returncode,
            0,
            f"warning raised at import:\n{proc.stderr[-2000:]}",
        )
        self.assertIn("ok", proc.stdout)

    def test_doctor_smoke_warnfree(self):
        """Test doctor smoke warnfree."""
        code = (
            "import json, tempfile, pathlib;"
            "from model_protocol_lab.doctor import examine_model;"
            "d = pathlib.Path(tempfile.mkdtemp());"
            "(d / 'config.json').write_text(json.dumps("
            "{'model_type': 'gpt2', 'n_positions': 32}));"
            "(d / 'tokenizer.json').write_text(json.dumps("
            "{'version': '1.0', 'model': {'type': 'BPE'}}));"
            "r = examine_model(d);"
            "print(r['verdict'])"
        )
        proc = _run_warnfree(code)
        self.assertEqual(
            proc.returncode,
            0,
            f"warning raised during doctor smoke:\n{proc.stderr[-2000:]}",
        )

    def test_report_render_smoke_warnfree(self):
        """Test report render smoke warnfree."""
        code = (
            "from model_protocol_lab.reporting import render_html;"
            "h = render_html({'decision': 'abstained', 'search': []});"
            "assert h.startswith('<!doctype html>');"
            "print('ok')"
        )
        proc = _run_warnfree(code)
        self.assertEqual(
            proc.returncode,
            0,
            f"warning raised during report render:\n{proc.stderr[-2000:]}",
        )
        self.assertIn("ok", proc.stdout)

    def test_cli_help_warnfree(self):
        """Test cli help warnfree."""
        proc = subprocess.run(
            [sys.executable, "-W", "error", "-m", "model_protocol_lab", "--help"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(REPO_ROOT),
            check=False,
        )
        self.assertEqual(
            proc.returncode,
            0,
            f"warning raised during CLI startup:\n{proc.stderr[-2000:]}",
        )


if __name__ == "__main__":
    unittest.main()
