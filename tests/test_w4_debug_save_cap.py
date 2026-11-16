"""W4 push 8 (RTI-07): the debug save cap is denominated in bytes.

The audit executed the defect: the components of a ``.save`` were
each checked against ``_MAX_SAVE_BYTES`` separately -- the existing
file's size and the transcript's length -- but the encoded total
was not, and the transcript length was counted in characters, not
in the UTF-8 bytes actually written. An existing file of
4,999,999 bytes plus a small transcript produced a 5,000,764-byte
file, over the cap the checks claimed to enforce; a multi-byte
transcript could exceed it on its own. The cap now denominates
everything in bytes: the transcript's running accounting, the
truncation budget, the transcript check, and the append total
(existing bytes + transcript bytes) that the final file must fit.
"""

import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from model_protocol_lab import debug as debug_module
from model_protocol_lab.core import Fixture, Message, Split
from model_protocol_lab.debug import DebugShell

CAP = 100


def _make_shell():
    """Make shell."""
    split = Split(
        search=(
            Fixture(
                id="s1",
                group="g1",
                messages=(Message("user", "a"),),
                expected="d",
            ),
        ),
        held_out=(
            Fixture(
                id="h1",
                group="g2",
                messages=(Message("user", "q"),),
                expected="a",
            ),
        ),
    )
    buf = io.StringIO()
    shell = DebugShell(
        split,
        dataset_label="test",
        tokenizer=None,
        stdin=io.StringIO(""),
        stdout=buf,
    )
    return shell, buf


def _run(shell, buf, *commands):
    """Run."""
    buf.seek(0)
    buf.truncate(0)
    shell.cmdqueue.extend(commands)
    shell.cmdloop()
    return buf.getvalue()


class DebugSaveCapTests(unittest.TestCase):
    """Test debug save cap."""

    def setUp(self):
        """Set up."""
        self._patch = mock.patch.object(debug_module, "_MAX_SAVE_BYTES", CAP)
        self._patch.start()

    def tearDown(self):
        """Tear down."""
        self._patch.stop()

    def test_append_total_over_cap_is_refused(self):
        """Test append total over cap is refused."""
        shell, buf = _make_shell()
        shell._transcript = ["x" * 20]
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "t.txt"
            target.write_bytes(b"y" * 90)
            out = _run(shell, buf, f".save {target}", ".quit")
            self.assertIn("would exceed 5MB", out)
            self.assertEqual(target.stat().st_size, 90)

    def test_multibyte_transcript_counts_bytes_not_chars(self):
        """Test multibyte transcript counts bytes not chars."""
        shell, buf = _make_shell()
        shell._transcript = ["é" * 60]  # 60 chars, 120 UTF-8 bytes
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "t.txt"
            out = _run(shell, buf, f".save {target}", ".quit")
            self.assertIn("transcript too large to save (over 5MB)", out)
            self.assertFalse(target.exists())

    def test_multibyte_append_under_cap_saves_exact_bytes(self):
        """Test multibyte append under cap saves exact bytes."""
        shell, buf = _make_shell()
        shell._transcript = ["é" * 20]  # 40 UTF-8 bytes
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "t.txt"
            target.write_bytes(b"y" * 10)
            out = _run(shell, buf, f".save {target}", ".quit")
            self.assertIn("saved 20 chars of transcript", out)
            self.assertEqual(target.stat().st_size, 50)

    def test_ascii_append_at_cap_boundary_saves(self):
        """Test ascii append at cap boundary saves."""
        shell, buf = _make_shell()
        shell._transcript = ["x" * 10]
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "t.txt"
            target.write_bytes(b"y" * 90)
            out = _run(shell, buf, f".save {target}", ".quit")
            self.assertIn("saved 10 chars of transcript", out)
            self.assertEqual(target.stat().st_size, 100)

    def test_truncation_budget_is_byte_denominated(self):
        """Test truncation budget is byte denominated."""
        shell, buf = _make_shell()
        # The echo of this (failing) command is the transcript's most
        # recent content: ~90 characters but ~130 UTF-8 bytes, so a
        # char-denominated budget never truncates it while a
        # byte-denominated one must.
        _run(shell, buf, ".fixture " + "é" * 40, ".quit")
        joined = "".join(shell._transcript)
        self.assertLessEqual(len(joined.encode("utf-8")), CAP)
        self.assertTrue(shell._transcript_truncated)
        self.assertTrue(shell._transcript[0].startswith("[... transcript truncated ...]"))


if __name__ == "__main__":
    unittest.main()
