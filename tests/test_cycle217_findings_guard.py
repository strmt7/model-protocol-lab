"""Cycle 217: contract tests for tools/findings_guard.py.

The guard fails the workflow when open CodeQL code findings exceed
two, or when any DeepSource finding exists for the audited commit.
These tests prove both legs against mocked inputs, including the
boundaries (2 passes, 3 fails; Scorecard alerts never counted;
missing DeepSource credential fails closed, never as zero).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import findings_guard as guard

REPO = "strmt7/model-protocol-lab"
COMMIT = "a" * 40


def _alert(number: int, tool: str = "CodeQL", rule: str = "py/test-rule") -> dict:
    """Build one code-scanning alert dict in the API's shape."""
    return {"number": number, "tool": {"name": tool}, "rule": {"id": rule}}


def _fetch(pages):
    """Build a paginated fetch stub returning the given pages and recording its calls."""
    state = {"calls": []}

    def fetch(page: int):
        """Return the requested page (empty past the end) and record the call."""
        state["calls"].append(page)
        return pages[page - 1] if page <= len(pages) else []

    return fetch


def _runner(total: int | None, complete: bool = True):
    """Build a runner stub writing a manifest with the given total and completeness."""

    def run(cmd, **kwargs):
        """Stub run: create the output directory and write the manifest the guard reads."""
        out = Path(cmd[cmd.index("--output") + 1])
        out.mkdir(parents=True, exist_ok=True)
        manifest = {"complete": complete}
        if total is not None:
            manifest["total_count"] = total
        (out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        class Proc:
            """Stub process result with the fields the guard reads (returncode, stdout, stderr)."""

            returncode = 0
            stdout = ""
            stderr = ""

        return Proc()

    return run


class GuardTests(unittest.TestCase):
    """Findings-guard evaluation tests over stubbed fetches and runners."""

    def setUp(self):
        """Create the isolated root directory the guard evaluation runs against."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def _evaluate(self, pages, runner, env_extra=None):
        """Evaluate the guard with the stubbed fetch, runner and environment."""
        env = {"HOME": str(self.root)}
        if env_extra:
            env.update(env_extra)
        with patch.dict("os.environ", env, clear=True):
            return guard.evaluate(REPO, COMMIT, _fetch(pages), self.root, runner)

    def test_zero_codeql_and_zero_deepsource_pass(self):
        """Zero CodeQL findings and zero DeepSource findings pass the guard."""
        report = self._evaluate([[]], _runner(0), {"DEEPSOURCE_TOKEN": "x"})
        self.assertTrue(report["pass"])
        self.assertEqual(report["codeql_open"], 0)
        self.assertEqual(report["deepsource_state"], "ok")

    def test_two_codeql_findings_pass_three_fail(self):
        """Two CodeQL findings pass the guard; three fail it."""
        passing = self._evaluate([[_alert(1), _alert(2)]], _runner(0), {"DEEPSOURCE_TOKEN": "x"})
        self.assertTrue(passing["pass"])
        failing = self._evaluate(
            [[_alert(1), _alert(2), _alert(3)]], _runner(0), {"DEEPSOURCE_TOKEN": "x"}
        )
        self.assertFalse(failing["pass"])
        self.assertIn("exceed limit 2", failing["failures"][0])

    def test_scorecard_alerts_never_counted(self):
        """Scorecard alerts are never counted as CodeQL findings."""
        report = self._evaluate(
            [[_alert(1, "Scorecard", "PinnedDependenciesID")] * 15],
            _runner(0),
            {"DEEPSOURCE_TOKEN": "x"},
        )
        self.assertEqual(report["codeql_open"], 0)
        self.assertTrue(report["pass"])

    def test_codeql_pagination_counts_every_page(self):
        """CodeQL pagination counts the alerts across every page."""
        first = [_alert(n) for n in range(100)]
        second = [_alert(100), _alert(101)]
        report = self._evaluate([first, second], _runner(0), {"DEEPSOURCE_TOKEN": "x"})
        self.assertEqual(report["codeql_open"], 102)
        self.assertFalse(report["pass"])

    def test_single_deepsource_finding_fails(self):
        """A single DeepSource finding fails the guard."""
        report = self._evaluate([[]], _runner(1), {"DEEPSOURCE_TOKEN": "x"})
        self.assertFalse(report["pass"])
        self.assertEqual(report["deepsource_state"], "findings")
        self.assertIn("DeepSource findings 1 exist", report["failures"][0])

    def test_missing_credential_fails_closed_not_zero(self):
        """A missing DeepSource credential fails closed as not-configured, never as zero."""
        report = self._evaluate([[]], _runner(0))
        self.assertFalse(report["pass"])
        self.assertEqual(report["deepsource_state"], "not-configured")
        self.assertIsNone(report["deepsource_total"])
        self.assertIn("NOT-CONFIGURED", report["failures"][0])

    def test_incomplete_manifest_fails_closed(self):
        """An incomplete inventory manifest fails the guard closed as an error."""
        report = self._evaluate([[]], _runner(None, complete=False), {"DEEPSOURCE_TOKEN": "x"})
        self.assertFalse(report["pass"])
        self.assertEqual(report["deepsource_state"], "error")


if __name__ == "__main__":
    unittest.main()
