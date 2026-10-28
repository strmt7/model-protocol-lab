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
    return {"number": number, "tool": {"name": tool}, "rule": {"id": rule}}


def _fetch(pages):
    state = {"calls": []}

    def fetch(page: int):
        state["calls"].append(page)
        return pages[page - 1] if page <= len(pages) else []

    return fetch


def _runner(total: int | None, complete: bool = True):
    def run(cmd, **kwargs):
        out = Path(cmd[cmd.index("--output") + 1])
        out.mkdir(parents=True, exist_ok=True)
        manifest = {"complete": complete}
        if total is not None:
            manifest["total_count"] = total
        (out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        class Proc:
            returncode = 0
            stdout = ""
            stderr = ""

        return Proc()

    return run


class GuardTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def _evaluate(self, pages, runner, env_extra=None):
        env = {"HOME": str(self.root)}
        if env_extra:
            env.update(env_extra)
        with patch.dict("os.environ", env, clear=True):
            return guard.evaluate(REPO, COMMIT, _fetch(pages), self.root, runner)

    def test_zero_codeql_and_zero_deepsource_pass(self):
        report = self._evaluate([[]], _runner(0), {"DEEPSOURCE_TOKEN": "x"})
        self.assertTrue(report["pass"])
        self.assertEqual(report["codeql_open"], 0)
        self.assertEqual(report["deepsource_state"], "ok")

    def test_two_codeql_findings_pass_three_fail(self):
        passing = self._evaluate([[_alert(1), _alert(2)]], _runner(0), {"DEEPSOURCE_TOKEN": "x"})
        self.assertTrue(passing["pass"])
        failing = self._evaluate(
            [[_alert(1), _alert(2), _alert(3)]], _runner(0), {"DEEPSOURCE_TOKEN": "x"}
        )
        self.assertFalse(failing["pass"])
        self.assertIn("exceed limit 2", failing["failures"][0])

    def test_scorecard_alerts_never_counted(self):
        report = self._evaluate(
            [[_alert(1, "Scorecard", "PinnedDependenciesID")] * 15],
            _runner(0),
            {"DEEPSOURCE_TOKEN": "x"},
        )
        self.assertEqual(report["codeql_open"], 0)
        self.assertTrue(report["pass"])

    def test_codeql_pagination_counts_every_page(self):
        first = [_alert(n) for n in range(100)]
        second = [_alert(100), _alert(101)]
        report = self._evaluate([first, second], _runner(0), {"DEEPSOURCE_TOKEN": "x"})
        self.assertEqual(report["codeql_open"], 102)
        self.assertFalse(report["pass"])

    def test_single_deepsource_finding_fails(self):
        report = self._evaluate([[]], _runner(1), {"DEEPSOURCE_TOKEN": "x"})
        self.assertFalse(report["pass"])
        self.assertEqual(report["deepsource_state"], "findings")
        self.assertIn("DeepSource findings 1 exist", report["failures"][0])

    def test_missing_credential_fails_closed_not_zero(self):
        report = self._evaluate([[]], _runner(0))
        self.assertFalse(report["pass"])
        self.assertEqual(report["deepsource_state"], "not-configured")
        self.assertIsNone(report["deepsource_total"])
        self.assertIn("NOT-CONFIGURED", report["failures"][0])

    def test_incomplete_manifest_fails_closed(self):
        report = self._evaluate([[]], _runner(None, complete=False), {"DEEPSOURCE_TOKEN": "x"})
        self.assertFalse(report["pass"])
        self.assertEqual(report["deepsource_state"], "error")


if __name__ == "__main__":
    unittest.main()
