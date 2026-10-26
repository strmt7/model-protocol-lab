"""Cycle 211: defensive opens must survive a Windows attribute set.

Cycle 210's smoke lane caught a genuine shipped-code defect: on Windows,
``inspect`` died with an AttributeError traceback because ``load_dataset``
opened with ``os.O_RDONLY | os.O_NONBLOCK`` and CPython on Windows has no
``os.O_NONBLOCK`` attribute (``inspect`` catches ValueError only, so the
error escaped). The fix is ``getattr(os, "O_NONBLOCK", 0)`` (and the same
for ``O_NOFOLLOW``) at every defensive-open site.

These tests simulate the Windows attribute set by deleting both flags
from the ``os`` module for the duration of each call, then restoring
them. Without the fix every reader below raises AttributeError before
any file work; with it, every reader behaves exactly as on Unix. The
deletion is scoped to the test process and restored in ``finally`` (and
via ``addCleanup`` as a second net), so no other test can observe it.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

from model_protocol_lab import backends, datasets, diffreports, fixtures, pack
from model_protocol_lab.surfacenoise import _load_probe_json


def _dataset_text() -> str:
    doc = {
        "search": [
            {
                "id": "s1",
                "group": "g-search",
                "messages": [{"role": "user", "content": "hello search"}],
                "expected": "hi",
            }
        ],
        "held_out": [
            {
                "id": "h1",
                "group": "g-held",
                "messages": [{"role": "user", "content": "hello held"}],
                "expected": "hi",
            }
        ],
    }
    return json.dumps(doc)


class _WithoutUnixFlags:
    """Context manager deleting O_NONBLOCK/O_NOFOLLOW from ``os``."""

    _NAMES = ("O_NONBLOCK", "O_NOFOLLOW")

    def __init__(self, test: unittest.TestCase) -> None:
        self._saved: dict[str, int] = {}
        test.addCleanup(self._restore)

    def __enter__(self) -> "_WithoutUnixFlags":
        for name in self._NAMES:
            if hasattr(os, name):
                self._saved[name] = getattr(os, name)
                delattr(os, name)
        return self

    def __exit__(self, *exc: object) -> bool:
        self._restore()
        return False

    def _restore(self) -> None:
        for name, value in self._saved.items():
            setattr(os, name, value)
        self._saved.clear()


class TestWindowsOpenFlags(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory(prefix="mpl-c211-")
        self.tmp_path = Path(self._td.name)

    def tearDown(self) -> None:
        self._td.cleanup()

    def test_simulation_actually_removes_the_flags(self) -> None:
        # Guard against a vacuous suite: on a platform where the flags
        # already do not exist this still passes, but on Linux (where
        # CI runs) the deletion must take effect for the other tests
        # to prove anything.
        with _WithoutUnixFlags(self):
            self.assertFalse(hasattr(os, "O_NONBLOCK"))
            self.assertFalse(hasattr(os, "O_NOFOLLOW"))
        self.assertTrue(hasattr(os, "O_NONBLOCK"))
        self.assertTrue(hasattr(os, "O_NOFOLLOW"))

    def test_load_dataset_without_o_nonblock(self) -> None:
        text = _dataset_text()
        target = self.tmp_path / "ds.json"
        target.write_text(text, encoding="utf-8")
        with _WithoutUnixFlags(self):
            loaded = datasets.load_dataset(target)
        self.assertEqual(loaded, datasets.parse_dataset(text))
        self.assertEqual(loaded.search[0].id, "s1")

    def test_read_authoring_file_without_o_nonblock(self) -> None:
        target = self.tmp_path / "note.txt"
        target.write_text("authoring text", encoding="utf-8")
        with _WithoutUnixFlags(self):
            self.assertEqual(fixtures.read_authoring_file(target), "authoring text")

    def test_backend_open_regular_read_without_o_nonblock(self) -> None:
        target = self.tmp_path / "asset.json"
        target.write_text('{"k": 1}', encoding="utf-8")
        with _WithoutUnixFlags(self):
            fd = backends._open_regular_read(target)
            try:
                with os.fdopen(fd, "rb") as handle:
                    fd = -1
                    content = handle.read()
            finally:
                if fd != -1:
                    os.close(fd)
        self.assertEqual(content, b'{"k": 1}')

    def test_load_probe_json_without_o_nonblock(self) -> None:
        target = self.tmp_path / "probe.json"
        target.write_text('{"a": 1}', encoding="utf-8")
        with _WithoutUnixFlags(self):
            self.assertEqual(_load_probe_json(target), {"a": 1})

    def test_load_results_without_o_nonblock(self) -> None:
        target = self.tmp_path / "results.json"
        payload = {
            "decision": "compatible_on_fixture_set",
            "search": [],
            "selected_candidate": None,
            "held_out": None,
        }
        target.write_text(json.dumps(payload), encoding="utf-8")
        with _WithoutUnixFlags(self):
            self.assertEqual(diffreports.load_results(target), payload)

    def test_pack_readers_without_unix_flags(self) -> None:
        plain = self.tmp_path / "entry.txt"
        plain.write_bytes(b"pack-entry")
        bundle = self.tmp_path / "bundle.zip"
        bundle.write_bytes(
            pack.build_bundle_bytes(
                pack.manifest_bytes(pack.build_manifest([], {})),
                [],
            )
        )
        with _WithoutUnixFlags(self):
            self.assertEqual(
                pack._read_regular_file(plain, too_large_code="x", cap=64),
                b"pack-entry",
            )
            self.assertEqual(pack.load_bundle_bytes(bundle), bundle.read_bytes())
