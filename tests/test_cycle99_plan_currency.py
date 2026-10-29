"""Cycle 99: DEVELOPMENT_PLAN.md currency regression test.

The plan was written when the lab was a proposal; its opening framed the
project as unimplemented. A 2026-10-16 status note now records that it has
been substantially implemented (v0.1.0). This test pins the note so future
edits cannot silently drop it and reintroduce the stale framing.
"""

from __future__ import annotations

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


class DevelopmentPlanCurrencyTests(unittest.TestCase):
    def test_status_note_present(self):
        text = (REPO_ROOT / "DEVELOPMENT_PLAN.md").read_text(encoding="utf-8")
        self.assertIn("Status note (2026-10-16, v0.1.0)", text)

    def test_status_note_records_implementation(self):
        text = (REPO_ROOT / "DEVELOPMENT_PLAN.md").read_text(encoding="utf-8")
        # The note must state the proposal phase is over.
        self.assertIn("substantially implemented", text)
        self.assertIn("RELEASE_READINESS_0.1.0.md", text)

    def test_no_bare_proposed_framing_in_first_paragraph(self):
        """The first content paragraph must not present the tool as
        merely proposed without the status-note qualification above it.
        """
        text = (REPO_ROOT / "DEVELOPMENT_PLAN.md").read_text(encoding="utf-8")
        # Status note must come before the historical "proposed" paragraph.
        note_pos = text.find("Status note (2026-10-16, v0.1.0)")
        proposed_pos = text.find("is a proposed local tool")
        self.assertGreaterEqual(note_pos, 0)
        self.assertGreaterEqual(proposed_pos, 0)
        self.assertLess(note_pos, proposed_pos)


if __name__ == "__main__":
    unittest.main()
