"""Offline contracts for portable provisioning and honest Crawl4AI evidence.

Adapted from strmt7/FlitzZip tools/test_crawl4ai_tool.py (MIT).
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools import crawl4ai_research as research
from tools import crawl4ai_tool as tool


class CrawlerContracts(unittest.TestCase):
    """Test repository integration without installing packages or making network requests."""

    def test_check_lock_uses_an_empty_environment_and_preserves_failures(self):
        """Purpose: Detect masked dependency conflicts
        Inputs: Resolver outcomes
        Outputs: Isolation checks.
        """
        for failure in (False, True):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temp:
                home = Path(temp)
                observed = []

                def resolve(
                    # skipcq: PYL-W0102 - The list default binds the shared observed list by design.
                    command,
                    env,
                    timeout=900,
                    _calls=observed,
                    _home=home,
                    _failure=failure,
                ):
                    """Purpose: Inspect the resolver boundary
                    Inputs: Actual argv/env
                    Outputs: Result or error.
                    """
                    _calls.append(command)
                    self.assertTrue(Path(command[0]).parent.parent.is_relative_to(_home))
                    self.assertIn("--dry-run", command)
                    self.assertIn("--ignore-installed", command)
                    self.assertNotIn("--no-deps", command)
                    self.assertEqual(command[-2:], ["-r", str(tool.LOCK)])
                    self.assertEqual(env["PYTHONNOUSERSITE"], "1")
                    if _failure:
                        raise subprocess.CalledProcessError(1, command)

                with (
                    patch.object(tool.venv, "EnvBuilder") as builder,
                    patch.object(tool, "_run", side_effect=resolve),
                    patch.object(tool, "install") as install,
                ):
                    if failure:
                        with self.assertRaises(subprocess.CalledProcessError):
                            tool.check_lock(home)
                    else:
                        tool.check_lock(home)
                    builder.assert_called_once_with(with_pip=True)
                    install.assert_not_called()
                self.assertEqual(len(observed), 1)

    def test_check_lock_cli_does_not_install_the_runtime_or_browser(self):
        """Purpose: Separate validation from setup
        Inputs: Public command
        Outputs: Only resolver invoked.
        """
        with (
            patch.object(tool, "check_lock") as resolve,
            patch.object(tool, "install") as install,
        ):
            self.assertEqual(tool.main(["check-lock"]), 0)
            resolve.assert_called_once()
            install.assert_not_called()

    def test_install_cli_provisions_and_reports_version(self):
        """Purpose: Verify install dispatch
        Inputs: Public command
        Outputs: Provision invoked, JSON reported.
        """
        with tempfile.TemporaryDirectory() as temp:
            fake_python = Path(temp) / "bin" / "python"
            with (
                patch.object(tool, "install", return_value=fake_python) as install,
                contextlib.redirect_stdout(io.StringIO()) as output,
            ):
                self.assertEqual(tool.main(["install"]), 0)
            install.assert_called_once()
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["installed"], tool.VERSION)

    def test_doctor_cli_does_not_provision(self):
        """Purpose: Keep diagnostics side-effect free
        Inputs: Public command
        Outputs: Doctor only.
        """
        with (
            patch.object(tool, "doctor") as doctor,
            patch.object(tool, "install") as install,
        ):
            self.assertEqual(tool.main(["doctor"]), 0)
            doctor.assert_called_once()
            install.assert_not_called()

    def test_text_identity_ignores_only_line_endings(self):
        """Purpose: Reuse checkout text
        Inputs: Line endings/content changes
        Outputs: Identity assertions.
        """
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "lock.txt"
            path.write_bytes(b"package==1.0\n")
            identity = tool.text_identity(path)
            path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
            self.assertEqual(tool.text_identity(path), identity)
            path.write_bytes(path.read_bytes().replace(b"1.0", b"1.1"))
            self.assertNotEqual(tool.text_identity(path), identity)

    def test_cache_and_interpreter_paths(self):
        """Purpose: Check host-independent ownership
        Inputs: Temporary paths/platforms
        Outputs: Path assertions.
        """
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp).resolve()
            self.assertEqual(tool.cache_home(str(path)), path)
            with patch.object(tool.os, "name", "nt"):
                directory, python, identity = tool.environment_paths(path)
                self.assertEqual(python, directory / "Scripts/python.exe")
            with patch.object(tool.os, "name", "posix"):
                directory, python, other = tool.environment_paths(path)
                self.assertEqual(python, directory / "bin/python")
            self.assertEqual(identity, other)
            for platform, expected in (
                ("linux", "model-protocol-lab/crawl4ai"),
                ("darwin", "Library/Caches/model-protocol-lab/crawl4ai"),
            ):
                with (
                    patch.object(tool.sys, "platform", platform),
                    patch.dict(os.environ, {}, clear=True),
                ):
                    self.assertTrue(tool.cache_home().as_posix().endswith(expected))
        with self.assertRaises(ValueError):
            tool.cache_home("relative-path")
        with self.assertRaises(ValueError):
            tool.cache_home(str(tool.ROOT))

    def test_tool_environment_is_child_scoped_and_disables_telemetry(self):
        """Purpose: Keep settings child-scoped
        Inputs: Cache home
        Outputs: Telemetry off, parent intact.
        """
        before = dict(os.environ)
        home = Path.home() / "crawler-test"
        env = tool.tool_environment(home)
        self.assertEqual(dict(os.environ), before)
        self.assertEqual(env["DO_NOT_TRACK"], "1")
        self.assertEqual(env["LITELLM_TELEMETRY"], "False")
        self.assertEqual(env["HF_HUB_DISABLE_TELEMETRY"], "1")
        self.assertTrue(env["PLAYWRIGHT_BROWSERS_PATH"].endswith("browsers"))

    def test_verify_versions_rejects_mismatch(self):
        """Purpose: Enforce the reviewed pin
        Inputs: Version strings
        Outputs: Explicit failure on drift.
        """
        with tempfile.TemporaryDirectory() as temp:
            python = Path(temp) / "python"
            python.write_text("", encoding="utf-8")
            env = {}
            with patch.object(tool.subprocess, "run") as run:
                run.return_value.stdout = tool.VERSION + "\n"
                tool.verify_versions(python, env)
                run.return_value.stdout = "0.0.0\n"
                with self.assertRaisesRegex(ValueError, "does not match"):
                    tool.verify_versions(python, env)

    def test_install_reuses_qualified_receipt(self):
        """Purpose: Avoid redundant provisioning
        Inputs: Valid receipt
        Outputs: Interpreter reused, no venv build.
        """
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp)
            directory, python, identity = tool.environment_paths(home)
            directory.mkdir(parents=True)
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_text("", encoding="utf-8")
            (directory / "install-receipt.json").write_text(
                json.dumps({"schema": 1, "version": tool.VERSION, "lock_sha256": identity}),
                encoding="utf-8",
            )
            with patch.object(tool.venv, "EnvBuilder") as builder:
                self.assertEqual(tool.install(home), python)
                builder.assert_not_called()

    def test_sites_manifest_has_thirty_two_entries(self):
        """Purpose: Pin the qualification surface
        Inputs: Manifest file
        Outputs: 32 url/marker pairs.
        """
        manifest = json.loads((tool.ROOT / "tools/crawl4ai_sites.json").read_text(encoding="utf-8"))
        self.assertEqual(sorted(manifest.keys()), ["purpose", "schema", "sites"])
        self.assertEqual(len(manifest["sites"]), 32)
        for site in manifest["sites"]:
            self.assertTrue(site["url"].startswith("https://"))
            self.assertTrue(site["marker"])

    def test_inspect_result_rejects_challenges_empty_and_error_pages(self):
        """Purpose: Keep qualification honest
        Inputs: Synthetic results
        Outputs: Strict validity.
        """
        good = SimpleNamespace(
            success=True,
            status_code=200,
            markdown=SimpleNamespace(raw_markdown="# Title\n" + "content " * 100),
            error_message="",
        )
        self.assertTrue(research.inspect_result(good, "title")["success"])
        challenge = SimpleNamespace(
            success=True,
            status_code=200,
            markdown=SimpleNamespace(raw_markdown="Verify you are human\n" + "x" * 500),
            error_message="",
        )
        self.assertFalse(research.inspect_result(challenge)["success"])
        empty = SimpleNamespace(success=True, status_code=200, markdown=None, error_message="")
        self.assertFalse(research.inspect_result(empty)["success"])
        error = SimpleNamespace(
            success=False,
            status_code=500,
            markdown=SimpleNamespace(raw_markdown="x" * 500),
            error_message="boom",
        )
        row = research.inspect_result(error)
        self.assertFalse(row["success"])
        self.assertIn("boom", row["error"])

    def test_crawler_version_gate(self):
        """Purpose: Bind research to the reviewed pin
        Inputs: Installed version
        Outputs: Explicit mismatch failure.
        """
        with (
            patch("importlib.metadata.version", return_value="0.0.0"),
            self.assertRaisesRegex(ValueError, "does not match"),
        ):
            research.crawler()

    def test_source_identity_covers_integration_files(self):
        """Purpose: Bind receipts to exact bytes
        Inputs: Repo files
        Outputs: Digests for the four tracked paths.
        """
        identity = research.source_identity()
        self.assertEqual(
            sorted(identity.keys()),
            [
                "tools/crawl4ai_research.py",
                "tools/crawl4ai_sites.json",
                "tools/crawl4ai_tool.py",
                "tools/requirements/crawl4ai.txt",
            ],
        )
        for digest in identity.values():
            self.assertEqual(len(digest), 64)


if __name__ == "__main__":
    unittest.main()
