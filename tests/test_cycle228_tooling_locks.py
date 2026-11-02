"""Cycle 228 — tooling locks: complete hash-pinned closures, closed files.

Pins for security-audit-2 findings B-01/B-03/B-04/B-06/B-07:

* Every file in tools/requirements/ is a complete hash-pinned
  closure: each requirement entry carries at least one --hash, so
  --require-hashes installs verify every artifact (B-01, B-03).
* The crawl4ai and cocoindex locks cover their full transitive
  closures, including the marker-gated lines a Linux resolution
  never sees (Windows colorama/pywin32, the Requires-Python split
  pairs), and both tool wrappers install with --require-hashes.
* The four pre-existing locks are closed on every target the audit
  swept: build.txt's Python < 3.12 lines, the Windows colorama
  lines in dev.txt/dev-hf.txt/audit.txt, and audit.txt's own pip
  pin, which makes it standalone-closed (B-04).
* pylock.toml declares the single environment its wheels target,
  and the stale "CI installs the latest tools" mechanism text is
  gone from the lock header and the repo's agent instructions
  (B-06, B-07).
* The Chromium revision Playwright installs is read from the
  hash-verified wheel's browsers.json (the browser archive itself
  carries no upstream hash; see docs/crawl4ai.md).
"""

from __future__ import annotations

import json
import re
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import crawl4ai_tool as crawl_tool

REPO_ROOT = Path(__file__).resolve().parent.parent
REQ_DIR = REPO_ROOT / "tools" / "requirements"


def _entries(path: Path) -> list[tuple[str, list[str]]]:
    """Parse a requirements file into (requirement line, hash lines)."""
    entries: list[tuple[str, list[str]]] = []
    current: tuple[str, list[str]] | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "--hash=" in line:
            assert current is not None, f"hash line before any entry in {path}"
            current[1].append(line)
        else:
            current = (line.rstrip(" \\"), [])
            entries.append(current)
    return entries


def _entry(entries: list[tuple[str, list[str]]], name: str) -> list[tuple[str, list[str]]]:
    """Return the entries whose requirement names the given package."""
    pattern = re.compile(rf"^{re.escape(name)}==", re.IGNORECASE)
    return [entry for entry in entries if pattern.match(entry[0])]


class TestEveryLockIsFullyHashed(unittest.TestCase):
    """B-01/B-03: no requirement entry anywhere ships without hashes."""

    def test_all_requirement_files_fully_hashed(self) -> None:
        """Purpose: Keep --require-hashes meaningful in every lock
        Inputs: tools/requirements/*.txt
        Outputs: Every entry carries at least one --hash line.
        """
        files = sorted(REQ_DIR.glob("*.txt"))
        self.assertGreaterEqual(len(files), 8)
        for path in files:
            entries = _entries(path)
            self.assertTrue(entries, f"{path.name} parsed to no entries")
            for requirement, hashes in entries:
                self.assertTrue(hashes, f"{path.name}: entry without hashes: {requirement}")


class TestCrawl4aiLock(unittest.TestCase):
    """B-01: the crawl4ai lock is the full closure, honestly framed."""

    def setUp(self) -> None:
        """Load the crawl4ai lock entries."""
        self.entries = _entries(REQ_DIR / "crawl4ai.txt")

    def test_top_level_pins_present(self) -> None:
        """Purpose: Pin the reviewed top-level packages
        Inputs: Lock entries
        Outputs: Crawl4AI 0.9.5 and Playwright 1.64.1 entries.
        """
        self.assertEqual(len(_entry(self.entries, "Crawl4AI")), 1)
        self.assertIn("0.9.5", _entry(self.entries, "Crawl4AI")[0][0])
        self.assertEqual(len(_entry(self.entries, "playwright")), 1)
        self.assertIn("1.64.1", _entry(self.entries, "playwright")[0][0])
        # The closure is far larger than the two top-level packages.
        self.assertGreater(len(self.entries), 90)

    def test_requires_python_split_pairs(self) -> None:
        """Purpose: Pin the 3.11 vs 3.12+ split shape
        Inputs: Lock entries
        Outputs: numpy carries both marker-split lines.
        """
        numpy = _entry(self.entries, "numpy")
        self.assertEqual(len(numpy), 2)
        markers = {entry[0].split(";", 1)[1].strip() for entry in numpy}
        self.assertEqual(markers, {'python_version >= "3.12"', 'python_version < "3.12"'})

    def test_windows_colorama_line(self) -> None:
        """Purpose: Pin the foreign-marker line a Linux run never sees
        Inputs: Lock entries
        Outputs: colorama with tqdm's platform_system marker.
        """
        colorama = _entry(self.entries, "colorama")
        self.assertEqual(len(colorama), 1)
        self.assertIn('platform_system == "Windows"', colorama[0][0])


class TestCocoindexLock(unittest.TestCase):
    """B-03: the cocoindex lock exists and is the full closure."""

    def setUp(self) -> None:
        """Load the cocoindex lock entries."""
        self.entries = _entries(REQ_DIR / "cocoindex.txt")

    def test_top_level_pin_and_closure(self) -> None:
        """Purpose: Pin the reviewed CLI and its closure size
        Inputs: Lock entries
        Outputs: cocoindex-code 0.2.42 plus a full transitive closure.
        """
        code = _entry(self.entries, "cocoindex-code")
        self.assertEqual(len(code), 1)
        self.assertIn("0.2.42", code[0][0])
        self.assertGreater(len(self.entries), 120)

    def test_windows_lines(self) -> None:
        """Purpose: Pin the win32-only lines with their parents' markers
        Inputs: Lock entries
        Outputs: pywin32 (mcp) and colorama (typer) marker lines.
        """
        pywin32 = _entry(self.entries, "pywin32")
        self.assertEqual(len(pywin32), 1)
        self.assertIn('sys_platform == "win32"', pywin32[0][0])
        colorama = _entry(self.entries, "colorama")
        self.assertEqual(len(colorama), 1)
        self.assertIn('platform_system == "Windows"', colorama[0][0])


class TestWrapperHashEnforcement(unittest.TestCase):
    """B-01/B-03: both wrappers install from their locks with hashes."""

    def test_crawl4ai_check_lock_enforces_hashes(self) -> None:
        """Purpose: Bind check-lock to --require-hashes
        Inputs: Stubbed child runner
        Outputs: The dry-run argv carries the flag before -r.
        """
        calls: list[list[str]] = []
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(crawl_tool, "_run", lambda args, env, timeout=0: calls.append(args)),
        ):
            crawl_tool.check_lock(Path(temp))
        self.assertEqual(len(calls), 1)
        argv = calls[0]
        self.assertIn("--require-hashes", argv)
        self.assertEqual(argv[-2:], ["-r", str(crawl_tool.LOCK)])

    def test_crawl4ai_install_enforces_hashes_and_records_revision(self) -> None:
        """Purpose: Bind install to the lock and the Chromium revision
        Inputs: Stubbed children; a fake installed Chromium directory
        Outputs: Install argv hashed; receipt records the revision.
        """
        calls: list[list[str]] = []
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp)
            (home / "browsers" / "chromium-1234").mkdir(parents=True)
            with (
                patch.object(crawl_tool, "_run", lambda args, env, timeout=0: calls.append(args)),
                patch.object(crawl_tool, "verify_versions", lambda python, env: None),
                patch.object(crawl_tool, "_pinned_chromium_revision", lambda python, env: "1234"),
            ):
                python = crawl_tool.install(home)
            receipt = json.loads(
                (python.parent.parent / "install-receipt.json").read_text(encoding="utf-8")
            )
        self.assertIn("--require-hashes", calls[0])
        self.assertEqual(calls[0][-2:], ["-r", str(crawl_tool.LOCK)])
        self.assertEqual(receipt["chromium_revision"], "1234")

    def test_cocoindex_wrapper_installs_from_lock_with_hashes(self) -> None:
        """Purpose: Bind both cocoindex install paths to the lock
        Inputs: Wrapper source
        Outputs: uv and pip argv both carry --require-hashes -r LOCK.
        """
        source = (REPO_ROOT / "tools" / "cocoindex_agent_search.py").read_text(encoding="utf-8")
        self.assertIn('LOCK = ROOT / "tools" / "requirements" / "cocoindex.txt"', source)
        self.assertEqual(source.count('"--require-hashes"'), 2)
        self.assertNotIn("cocoindex-code[full]==", source)


class TestChromiumRevisionParsing(unittest.TestCase):
    """B-01: the revision comes from browsers.json, never a guess."""

    def test_reads_chromium_not_headless_shell(self) -> None:
        """Purpose: Select the chromium entry among browser entries
        Inputs: Synthetic browsers.json
        Outputs: The chromium revision.
        """
        payload = json.dumps(
            {
                "browsers": [
                    {"name": "chromium-headless-shell", "revision": "9999"},
                    {"name": "chromium", "revision": "1200"},
                    {"name": "firefox", "revision": "1500"},
                ]
            }
        )
        self.assertEqual(crawl_tool._chromium_revision(payload), "1200")

    def test_missing_or_nonnumeric_revision_fails(self) -> None:
        """Purpose: Fail closed when no revision is pinned
        Inputs: browsers.json without a usable chromium revision
        Outputs: RuntimeError.
        """
        with self.assertRaises(RuntimeError):
            crawl_tool._chromium_revision(json.dumps({"browsers": []}))
        with self.assertRaises(RuntimeError):
            crawl_tool._chromium_revision(
                json.dumps({"browsers": [{"name": "chromium", "revision": "tip"}]})
            )


class TestClosedPreexistingLocks(unittest.TestCase):
    """B-04: the four older locks are closed on the swept targets."""

    def test_build_lock_python_below_312_lines(self) -> None:
        """Purpose: Pin build.txt's hand-added < 3.12 closure
        Inputs: build.txt entries
        Outputs: backports.tarfile, importlib_metadata, zipp lines.
        """
        entries = _entries(REQ_DIR / "build.txt")
        for name in ("backports.tarfile", "importlib_metadata", "zipp"):
            found = _entry(entries, name)
            self.assertEqual(len(found), 1, name)
            self.assertIn('python_version < "3.12"', found[0][0], name)

    def test_windows_colorama_lines(self) -> None:
        """Purpose: Pin each file's colorama line with its parent's marker
        Inputs: dev.txt, dev-hf.txt, audit.txt entries
        Outputs: One colorama line each, markers as swept.
        """
        expectations = {
            "dev.txt": 'sys_platform == "win32"',
            "dev-hf.txt": 'platform_system == "Windows"',
            "audit.txt": 'platform_system == "Windows"',
        }
        for filename, marker in expectations.items():
            found = _entry(_entries(REQ_DIR / filename), "colorama")
            self.assertEqual(len(found), 1, filename)
            self.assertIn(marker, found[0][0], filename)

    def test_audit_lock_pins_pip_like_pip_txt(self) -> None:
        """Purpose: Make audit.txt standalone-closed via its own pip pin
        Inputs: audit.txt and pip.txt entries
        Outputs: Same pip version and the same hash set in both.
        """
        audit_pip = _entry(_entries(REQ_DIR / "audit.txt"), "pip")
        self.assertEqual(len(audit_pip), 1)
        own_pip = _entry(_entries(REQ_DIR / "pip.txt"), "pip")
        self.assertEqual(len(own_pip), 1)
        self.assertEqual(audit_pip[0][0], own_pip[0][0])
        self.assertEqual(set(audit_pip[0][1]), set(own_pip[0][1]))


class TestPylockEnvironmentAndMechanismText(unittest.TestCase):
    """B-06/B-07: the PEP 751 lock declares its target; text is true."""

    def test_pylock_declares_single_environment(self) -> None:
        """Purpose: Pin the lock's PEP 751 environment fields
        Inputs: pylock.toml
        Outputs: requires-python and environments name CPython 3.12/Linux.
        """
        data = tomllib.loads((REPO_ROOT / "pylock.toml").read_text(encoding="utf-8"))
        self.assertEqual(data["requires-python"], "==3.12.*")
        environments = data["environments"]
        self.assertEqual(len(environments), 1)
        self.assertIn('python_version == "3.12"', environments[0])
        self.assertIn('sys_platform == "linux"', environments[0])

    def test_no_latest_tools_claim_remains(self) -> None:
        """Purpose: Keep the corrected mechanism text from regressing
        Inputs: pylock.toml header and repo AGENTS.md
        Outputs: Neither claims CI installs the latest ruff/bandit.
        """
        pylock = (REPO_ROOT / "pylock.toml").read_text(encoding="utf-8")
        self.assertNotIn("deliberately installs the\n# latest", pylock)
        agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
        self.assertNotIn("latest bandit", agents)
        self.assertIn("tools/requirements/audit.txt", agents)


if __name__ == "__main__":
    unittest.main()
