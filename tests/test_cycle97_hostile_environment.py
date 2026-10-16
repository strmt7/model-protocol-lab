"""Cycle 97: hostile-environment robustness audit.

The red-team sweeps (cycles 91, 96) covered hostile *inputs*. This module
covers hostile *environments*: the filesystem and OS conditions under
which the package's write paths must fail cleanly — no tracebacks leaking
internals, no partial files, no hangs.

We run as root in CI/dev, so chmod-based permission tests self-skip;
permission failures are simulated with unittest.mock on the exact syscalls,
which exercises the same error-handling paths.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from model_protocol_lab import fixtures as fixtures_mod
from model_protocol_lab import reporting


def _minimal_result() -> dict:
    return {
        "protocol": "test",
        "candidates": [],
        "search_fixtures": [],
        "heldout_fixtures": [],
    }


class PrepareReportDirAdversarialTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mpl97_"))

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_mkdir_permission_denied_raises_cleanly(self):
        """PermissionError during mkdir propagates as OSError, no partial dir."""
        target = self.tmp / "nope" / "deep"
        with (
            mock.patch.object(Path, "mkdir", side_effect=PermissionError(13, "denied")),
            self.assertRaises(OSError),
        ):
            reporting.prepare_report_dir(target)
        # Nothing was created.
        self.assertFalse((self.tmp / "nope").exists())

    def test_target_is_existing_file_raises_file_exists(self):
        f = self.tmp / "afile"
        f.write_text("x")
        with self.assertRaises(FileExistsError):
            reporting.prepare_report_dir(f)

    def test_existing_nonempty_dir_raises(self):
        d = self.tmp / "full"
        d.mkdir()
        (d / "something").write_text("x")
        with self.assertRaises(FileExistsError):
            reporting.prepare_report_dir(d)

    def test_concurrent_claim_second_fails_cleanly(self):
        """Two concurrent claims: exactly one wins, the other FileExistsError."""
        d = self.tmp / "race"
        reporting.prepare_report_dir(d)
        self.assertTrue(d.is_dir())
        with self.assertRaises(FileExistsError):
            reporting.prepare_report_dir(d)

    def test_allow_empty_existing_accepts_empty_dir(self):
        d = self.tmp / "empty"
        d.mkdir()
        out = reporting.prepare_report_dir(d, _allow_empty_existing=True)
        self.assertEqual(out, d)


class SaveReportAdversarialTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mpl97_"))

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_write_permission_denied_no_partial_files(self):
        """os.open failing -> OSError, and no half-written pair is left."""
        target = self.tmp / "rep"

        def _denied(path, flags, *a, **k):
            raise PermissionError(13, "denied")

        with mock.patch.object(os, "open", side_effect=_denied), self.assertRaises(OSError):
            reporting.save_report(_minimal_result(), target)
        # Either the dir was never created or it is empty.
        if target.exists():
            self.assertEqual(list(target.iterdir()), [])

    def test_second_file_failure_removes_first(self):
        """If the HTML write fails after results.json, both are removed."""
        target = self.tmp / "rep2"
        calls = {"n": 0}
        real_open = os.open

        def _fail_second(path, flags, *a, **k):
            calls["n"] += 1
            if calls["n"] == 2:
                raise OSError(28, "no space left")
            return real_open(path, flags, *a, **k)

        with mock.patch.object(os, "open", side_effect=_fail_second), self.assertRaises(OSError):
            reporting.save_report(_minimal_result(), target)
        # No half-built pair may remain.
        if target.exists():
            leftovers = list(target.iterdir())
            self.assertEqual(leftovers, [], f"half-built report pair left: {leftovers}")

    def test_unwritable_parent_raises_cleanly(self):
        with (
            mock.patch.object(Path, "mkdir", side_effect=PermissionError(13, "denied")),
            self.assertRaises(OSError),
        ):
            reporting.save_report(_minimal_result(), self.tmp / "sub" / "rep")


class WriteFixtureFileAdversarialTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mpl97_"))

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_permission_denied_raises_cleanly(self):
        # Contract: OSError is mapped to a clean ValueError (no raw
        # traceback internals, no partial file).
        with (
            mock.patch.object(os, "open", side_effect=PermissionError(13, "denied")),
            self.assertRaises(ValueError) as ctx,
        ):
            fixtures_mod.write_fixture_file("a,b\n1,2\n", self.tmp / "f.csv")
        self.assertIn("not writable", str(ctx.exception))
        self.assertFalse((self.tmp / "f.csv").exists())

    def test_existing_file_not_clobbered(self):
        """O_EXCL: never silently overwrite an existing fixture file."""
        p = self.tmp / "f.csv"
        p.write_text("original\n")
        # Contract: existing file -> clean ValueError, original untouched.
        with self.assertRaises(ValueError) as ctx:
            fixtures_mod.write_fixture_file("replacement\n", p)
        self.assertIn("already exists", str(ctx.exception))
        self.assertEqual(p.read_text(), "original\n")


if __name__ == "__main__":
    unittest.main()
