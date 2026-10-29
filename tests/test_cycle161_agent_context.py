"""Real-file context bounds, provenance, collision and mandatory-instruction contracts.

Adapted from strmt7/FlitzZip tools/test_agent_context.py (MIT).
"""

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import agent_context as context


class AgentContextTests(unittest.TestCase):
    """Agent context tests."""

    def setUp(self):
        """Purpose: Isolate state
        Inputs: temporary filesystem
        Outputs: owned checkout with mixed Markdown fences.
        """
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "guide.md"
        self.source.write_text(
            "# Guide\n"
            "intro\n"
            "## Selected\n"
            "required\n"
            "```md\n"
            "## Fake\n"
            "```\n"
            "### Nested\n"
            "more\n"
            "## Other\n"
            "end\n",
            encoding="utf-8",
        )

    @staticmethod
    def read_args(**changes):
        """Purpose: Select test extent
        Inputs: option overrides
        Outputs: ordinary CLI-equivalent namespace.
        """
        values = {
            "path": "guide.md",
            "section": "Selected",
            "start": 1,
            "end": None,
            "max_chars": 8000,
            "session": None,
            "force": False,
        }
        return argparse.Namespace(**(values | changes))

    def note_args(self, **changes):
        """Purpose: Select cited note
        Inputs: overrides
        Outputs: typed synthesis with explicit source and expiry.
        """
        values = {
            "key": "timing-decision",
            "summary_file": "summary.md",
            "source": ["guide.md"],
            "kind": "decision",
            "expires_hours": 24,
        }
        (self.root / "summary.md").write_text(
            "Use the exact measured counters.\n", encoding="utf-8"
        )
        return argparse.Namespace(**(values | changes))

    def test_section_includes_descendants_and_ignores_fenced_heading(self):
        """Purpose: Preserve required reading.
        Inputs: nested/fenced document.
        Outputs: exact complete section extent.
        """
        result = context.read_window(self.root, self.read_args())
        self.assertEqual((result["start"], result["end"]), (3, 9))
        self.assertEqual(result["status"], "complete")
        self.assertIn("9: more", result["content"])
        self.assertNotIn("10: ## Other", result["content"])
        self.assertEqual(
            [item["title"] for item in context.headings(self.source.read_text())],
            ["Guide", "Selected", "Nested", "Other"],
        )

    def test_partial_budget_reports_unread_and_never_enters_cache(self):
        """Purpose: Prevent false read coverage.
        Inputs: too-small budget/session.
        Outputs: unread range on every attempt.
        """
        args = self.read_args(max_chars=20, session="one-context")
        first = context.read_window(self.root, args)
        second = context.read_window(self.root, args)
        self.assertEqual(first["status"], "partial")
        self.assertEqual(second["status"], "partial")
        self.assertIsNotNone(first["unread"])
        self.assertNotIn("unchanged_in_explicit_session", (first["status"], second["status"]))

    def test_complete_cache_requires_same_context_range_and_bytes(self):
        """Purpose: Bind cached reads to exact bytes.
        Inputs: session read then changed source.
        Outputs: changed bytes are re-read, not served stale.
        """
        args = self.read_args(session="one-context")
        first = context.read_window(self.root, args)
        self.assertEqual(first["status"], "complete")
        second = context.read_window(self.root, args)
        self.assertEqual(second["status"], "unchanged_in_explicit_session")
        self.assertEqual(second["content"], "")
        self.source.write_text(
            self.source.read_text(encoding="utf-8") + "appended\n", encoding="utf-8"
        )
        third = context.read_window(self.root, args)
        self.assertEqual(third["status"], "complete")

    def test_source_and_state_reject_escape_binary_and_large_files(self):
        """Purpose: Contain reads to the checkout.
        Inputs: escape/binary/oversize/secret paths.
        Outputs: explicit refusal before access.
        """
        for bad in (
            "../escape.md",
            "/absolute.md",
            "a\\b.md",
            ".git/config",
            "secrets/key.md",
            ".env",
            "key.pem",
        ):
            with self.assertRaises(ValueError, msg=bad):
                context.relative_source_path(bad)
        binary = self.root / "binary.bin"
        binary.write_bytes(b"\x00\x01")
        with self.assertRaises(ValueError):
            context.read_source(self.root, "binary.bin")
        with self.assertRaises(ValueError):
            context.state_name("notes", "BAD KEY")

    def test_note_freshness_is_not_validation_and_source_change_withholds_summary(self):
        """Purpose: Keep notes honest.
        Inputs: note then changed source.
        Outputs: stale status withholds summary.
        """
        args = self.note_args()
        created = context.remember(self.root, args)
        fresh = context.recall(self.root, args.key)
        self.assertEqual(fresh["status"], "declared_sources_unchanged")
        self.assertIsNotNone(fresh["summary"])
        self.source.write_text("changed\n", encoding="utf-8")
        stale = context.recall(self.root, args.key)
        self.assertEqual(stale["status"], "stale")
        self.assertIsNone(stale["summary"])
        self.assertEqual(created["note"], context.state_name("notes", args.key))

    def test_expired_note_is_withheld_and_existing_note_is_never_overwritten(self):
        """Purpose: Enforce note TTL and immutability.
        Inputs: expired note and duplicate key.
        Outputs: stale withholding and exclusive-write refusal.
        """
        args = self.note_args()
        context.remember(self.root, args)
        with self.assertRaises(OSError):
            context.remember(self.root, args)
        name = context.state_name("notes", args.key)
        note = context.load_state(self.root, name)
        note["expires_at"] = "2000-01-01T00:00:00+00:00"
        context.write_state(self.root, name, note)
        self.assertEqual(context.recall(self.root, args.key)["reasons"], [{"reason": "expired"}])

    def test_startup_delivers_current_mandatory_skills_and_fails_if_missing(self):
        """Purpose: Check actual instruction delivery.
        Inputs: repository and missing local fixture.
        Outputs: full skills, explicit failure.
        """
        delivered = context.startup(context.ROOT)
        self.assertEqual(len(delivered["skills"]), 2)
        for skill in delivered["skills"]:
            actual, digest = context.read_source(context.ROOT, skill["path"])
            self.assertEqual((skill["instructions"], skill["sha256"]), (actual, digest))
        agents, digest = context.read_source(context.ROOT, "AGENTS.md")
        research = delivered["research"]
        self.assertIn(research["instructions"], agents)
        self.assertEqual(research["sha256"], digest)
        self.assertEqual(
            research["launcher_sha256"], context.read_source(context.ROOT, research["launcher"])[1]
        )
        (self.root / "AGENTS.md").write_text("Mandatory skills are missing.\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            context.startup(self.root)

    def test_startup_rejects_removed_research_rule_and_missing_portable_launcher(self):
        """Purpose: Prevent silent research-route removal.
        Inputs: real current instructions and missing tool/rule mutations.
        Outputs: offline startup fails; no crawler installation occurs.
        """
        agents, _ = context.read_source(context.ROOT, "AGENTS.md")
        for name in context.SKILLS:
            relative = f".agents/skills/{name}/SKILL.md"
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(context.read_source(context.ROOT, relative)[0], encoding="utf-8")
        (self.root / "AGENTS.md").write_bytes(agents.encode("utf-8"))
        with self.assertRaises(FileNotFoundError):
            context.startup(self.root)
        target = self.root / "tools/crawl4ai_tool.py"
        target.parent.mkdir()
        target.write_text(
            context.read_source(context.ROOT, "tools/crawl4ai_tool.py")[0], encoding="utf-8"
        )
        delivered = context.startup(self.root)
        (self.root / "AGENTS.md").write_bytes(
            agents.replace(delivered["research"]["instructions"], "").encode("utf-8")
        )
        with self.assertRaisesRegex(ValueError, "Crawl4AI"):
            context.startup(self.root)

    def test_startup_rejects_skill_routing_removal(self):
        """Purpose: Catch AGENTS.md that drops a skill path.
        Inputs: mandates present but skill file unlisted.
        Outputs: explicit routing failure.
        """
        agents, _ = context.read_source(context.ROOT, "AGENTS.md")
        (self.root / "AGENTS.md").write_text(
            agents.replace(".agents/skills/caveman/SKILL.md", ".agents/skills/elsewhere/SKILL.md"),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "routing"):
            context.startup(self.root)

    def test_cli_budget_errors_and_stale_exit_remain_machine_visible(self):
        """Purpose: Verify real command dispatch.
        Inputs: valid/invalid CLI and expired note.
        Outputs: JSON and distinct exit status.
        """
        args = self.note_args()
        context.remember(self.root, args)
        name = context.state_name("notes", args.key)
        note = context.load_state(self.root, name)
        note["expires_at"] = "2000-01-01T00:00:00+00:00"
        context.write_state(self.root, name, note)
        with patch.object(context, "ROOT", self.root):
            for argv, expected in (
                (["outline", "guide.md"], 0),
                (["read", "guide.md", "--max-chars", "0"], 2),
                (["recall", args.key], 3),
            ):
                with self.subTest(argv=argv), contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(context.main(argv), expected)
                    self.assertIsInstance(json.loads(output.getvalue()), dict)


if __name__ == "__main__":
    unittest.main()
