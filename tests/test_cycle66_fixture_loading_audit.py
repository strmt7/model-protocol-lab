"""Cycle 66 -- adversarial audit of fixture LOADING.

Scope
-----
The loader that builds :class:`Split` objects from real files:

- ``model_protocol_lab/fixtures.py``: ``read_authoring_file`` (bounded UTF-8
  read with non-regular-file refusal, ~lines 60-97) and ``parse_fixture_csv``
  (the CSV authoring format, ~lines 110-208);
- ``model_protocol_lab/datasets.py``: ``load_dataset`` / ``parse_dataset``
  (~lines 85-138), re-audited adversarially only for gaps left by the
  hardening suites.

Method
------
Every case builds a hostile input and asserts the EXACT observable behavior:
the loaded split contents, or the exact fixed ``ValueError`` message string.
Where ``docs/USAGE.md`` documents a behavior it is cross-checked; doc drift
is flagged in this docstring but no docs were edited.

Findings
--------
REAL bugs (reported, not fixed -- repo source untouched):

- BUG-1 (medium), ``fixtures.py`` ``parse_fixture_csv``: a single CSV field
  larger than the csv module's default field limit (131072 chars) raises
  ``_csv.Error`` -- NOT a ``ValueError`` -- even though the file is under
  the 2 MB cap and the documented contract is "Raises ValueError naming the
  offending row". The module's own 16384-char cell check never runs because
  the csv parser explodes first. CLI impact: ``fixtures build`` catches only
  ``ValueError`` (cli.py ``except ValueError`` around the build), so such an
  input crashed with an unhandled traceback instead of the clean
  ``fixtures: invalid authoring source`` message. FIXED: the parser loop now
  catches ``csv.Error`` and raises ``ValueError("CSV field exceeds the
  parser limit near row N")``.
- BUG-2 (low), ``fixtures.py`` ``read_authoring_file`` and ``datasets.py``
  ``load_dataset``: a read-time ``OSError`` after a successful open was not
  normalized to ``ValueError``. ``read_authoring_file("/proc/self/mem")``
  and ``load_dataset("/proc/self/mem")`` pass the fstat regular-file check
  (``S_ISREG`` is true for /proc files) and then ``handle.read()`` raises
  ``OSError: [Errno 5] Input/output error``. Both CLI entry points
  (``inspect``, ``fixtures build``) catch only ``ValueError``. FIXED: the
  read is wrapped and OSError maps to the existing "Unreadable ..." codes.

CLEAN (verified, no bug):
- The partition column is honored exactly and case-sensitively; wrong-case
  labels are rejected with a row-numbered error.
- Whitespace-padded IDs are distinct and stored verbatim, matching the
  documented "no whitespace stripping" rule; NFC/NFD duplicate IDs, group
  overlap and identical-message leakage are all caught via the Split
  discipline (NFC-normalized, byte-exact storage preserved).
- CRLF, quoted commas/quotes/newlines parse; missing trailing columns pad
  with "" so the conversation ends at the first empty cell; row numbers are
  correct (a multiline quoted record reports its last line); all caps hold
  (2 MB file, 200 rows total, 100 per partition, 21 message columns,
  16384 chars/cell) with exact acceptance at each boundary.
- FIFO / directory / /dev/null / symlink-to-FIFO / dangling symlink /
  missing / null-byte / non-path inputs all raise ``ValueError`` with fixed
  messages; UTF-16-with-BOM and latin-1 are refused as non-UTF-8; BOM-less
  UTF-16 decodes to NUL-containing garbage but is then rejected by the
  header check (loud, never silently accepted); deeply nested JSON becomes
  ``ValueError("Dataset nesting too deep")``.

Minor observations (not bugs):
- A whitespace-only line (``"   "``) is treated as a data row and fails with
  "fixture id is blank" rather than being skipped as a blank line. Loud and
  safe, but ``docs/USAGE.md`` says "blank lines between rows carry no
  fixture" -- minor doc-vs-reality drift on what counts as blank.
- Oversize id/group cells pass ``parse_fixture_csv`` and fail only at
  ``split_to_json_text`` (already enshrined by an existing hardening test);
  the module docstring's "16,384 characters per cell" does not mention the
  staging. Minor.
- A null byte in the path raises bare ``ValueError`` from the interpreter
  (the ``OSError`` handler does not catch it) in both readers -- still a
  ``ValueError``, matching the existing loose contract for that input.
  The exact message is CPython-version-dependent ("embedded null byte" on
  older releases, "embedded null character" on newer ones); the test
  matches the stable core of both.

Mutation gate: 4 mutants on /tmp copies -- (1) partition assignment ignored,
(2) partition validation dropped, (3) NFC normalization removed from
``Split``, (4) authoring size cap dropped -- each caught by >=1 test here.
"""

import os
import tempfile
import unicodedata
import unittest
from pathlib import Path

from model_protocol_lab import fixtures as fixtures_mod
from model_protocol_lab.datasets import MAX_BYTES, load_dataset, parse_dataset
from model_protocol_lab.fixtures import parse_fixture_csv, read_authoring_file

HDR = "id,group,partition,expected,user_1,assistant_1,user_2"
GOOD_SEARCH = "a,g1,search,x,hello,,"
GOOD_HELD = "b,g2,held_out,y,hi,,"


def _doc(search_row: str, held_row: str) -> str:
    """Doc."""
    return f"{HDR}\n{search_row}\n{held_row}\n"


def _expect_error(text: str) -> str:
    """Expect error."""
    try:
        parse_fixture_csv(text)
    except ValueError as exc:
        return str(exc)
    raise AssertionError("expected ValueError from parse_fixture_csv")


class TestParseFixtureCsvAdversarial(unittest.TestCase):
    """Test parse fixture csv adversarial."""

    @staticmethod
    def test_baseline_two_rows():
        """Test baseline two rows."""
        split, empty = parse_fixture_csv(_doc(GOOD_SEARCH, GOOD_HELD))
        assert [f.id for f in split.search] == ["a"]
        assert [f.id for f in split.held_out] == ["b"]
        assert [(m.role, m.content) for m in split.search[0].messages] == [("user", "hello")]
        assert empty == []

    def test_partition_label_is_case_sensitive(self):
        """Test partition label is case sensitive."""
        for label in ["Search", "SEARCH", "held out", "search ", "held_OUT"]:
            with self.subTest(label=label):
                message = _expect_error(_doc(f"a,g1,{label},x,hello,,", GOOD_HELD))
                assert message == (
                    f"row 2 (id 'a'): partition must be 'search' or 'held_out', got {label!r}"
                )

    @staticmethod
    def test_empty_string_id_rejected_with_row_number():
        """Test empty string id rejected with row number."""
        assert _expect_error(_doc(",g1,search,x,hello,,", GOOD_HELD)) == (
            "row 2: fixture id is blank"
        )

    @staticmethod
    def test_whitespace_only_id_rejected_with_row_number():
        """Test whitespace only id rejected with row number."""
        assert _expect_error(_doc("  ,g1,search,x,hello,,", GOOD_HELD)) == (
            "row 2: fixture id is blank"
        )

    @staticmethod
    def test_whitespace_padded_ids_are_distinct_and_verbatim():
        """Test whitespace padded ids are distinct and verbatim."""
        split, _ = parse_fixture_csv(_doc("ab,g1,search,x,hello,,", " ab,g2,held_out,y,hi,,"))
        assert split.search[0].id == "ab"
        assert split.held_out[0].id == " ab"

    @staticmethod
    def test_case_differing_ids_are_distinct():
        """Test case differing ids are distinct."""
        split, _ = parse_fixture_csv(_doc("Ab,g1,search,x,hello,,", "ab,g2,held_out,y,hi,,"))
        assert [f.id for f in split.search] == ["Ab"]
        assert [f.id for f in split.held_out] == ["ab"]

    @staticmethod
    def test_nfc_nfd_duplicate_ids_rejected():
        """Test nfc nfd duplicate ids rejected."""
        nfc = unicodedata.normalize("NFC", "caf\u00e9")
        nfd = unicodedata.normalize("NFD", "caf\u00e9")
        assert nfc != nfd
        message = _expect_error(_doc(f"{nfc},g1,search,x,hello,,", f"{nfd},g2,held_out,y,hi,,"))
        assert message == (
            "CSV fixtures violate the split discipline: Fixture IDs must be globally unique"
        )

    @staticmethod
    def test_exact_duplicate_ids_same_partition_rejected():
        """Test exact duplicate ids same partition rejected."""
        text = HDR + "\na,g1,search,x,hello,,\na,g9,search,z,hey,,\n" + GOOD_HELD + "\n"
        assert _expect_error(text) == (
            "CSV fixtures violate the split discipline: Fixture IDs must be globally unique"
        )

    @staticmethod
    def test_nfd_group_overlap_rejected():
        """Test nfd group overlap rejected."""
        nfc = unicodedata.normalize("NFC", "caf\u00e9")
        nfd = unicodedata.normalize("NFD", "caf\u00e9")
        message = _expect_error(_doc(f"a,{nfc},search,x,hello,,", f"b,{nfd},held_out,y,hi,,"))
        assert message == (
            "CSV fixtures violate the split discipline: Related fixtures cannot cross the split"
        )

    @staticmethod
    def test_nfd_message_leakage_rejected():
        """Test nfd message leakage rejected."""
        nfc = unicodedata.normalize("NFC", "caf\u00e9")
        nfd = unicodedata.normalize("NFD", "caf\u00e9")
        message = _expect_error(_doc(f"a,g1,search,x,{nfc},,", f"b,g2,held_out,y,{nfd},,"))
        assert message == (
            "CSV fixtures violate the split discipline: "
            "Identical message inputs cannot cross the split"
        )

    @staticmethod
    def test_nfd_data_kept_byte_exact():
        """Test nfd data kept byte exact."""
        nfd = unicodedata.normalize("NFD", "caf\u00e9")
        split, _ = parse_fixture_csv(_doc(f"a,g1,search,x,{nfd},,", GOOD_HELD))
        assert split.search[0].messages[0].content == nfd
        assert split.search[0].messages[0].content != unicodedata.normalize("NFC", nfd)

    @staticmethod
    def test_header_only_file():
        """Test header only file."""
        assert _expect_error(HDR + "\n") == "CSV has no 'search' fixtures"

    @staticmethod
    def test_header_only_held_out_reports_search_first():
        """Test header only held out reports search first."""
        assert _expect_error(HDR + "\nb,g2,held_out,y,hi,,\n") == "CSV has no 'search' fixtures"

    @staticmethod
    def test_missing_trailing_columns_pad_with_empty():
        """Test missing trailing columns pad with empty."""
        split, _ = parse_fixture_csv(_doc("a,g1,search,x,hello", GOOD_HELD))
        assert [(m.role, m.content) for m in split.search[0].messages] == [("user", "hello")]

    @staticmethod
    def test_row_shorter_than_fixed_columns_needs_user_1():
        """Test row shorter than fixed columns needs user 1."""
        assert _expect_error(_doc("a,g1,search", GOOD_HELD)) == (
            "row 2 (id 'a'): fixture needs a user_1 message"
        )

    @staticmethod
    def test_extra_cell_rejected():
        """Test extra cell rejected."""
        assert _expect_error(_doc("a,g1,search,x,hello,,,EXTRA", GOOD_HELD)) == (
            "row 2: 8 cells, header has 7"
        )

    @staticmethod
    def test_bom_prefixed_header_rejected():
        """Test bom prefixed header rejected."""
        message = _expect_error("\ufeff" + _doc(GOOD_SEARCH, GOOD_HELD))
        assert message == (
            "CSV header must start with 'id,group,partition,expected', "
            "got '\\ufeffid,group,partition,expected'"
        )

    @staticmethod
    def test_crlf_accepted():
        """Test crlf accepted."""
        text = HDR + "\r\n" + GOOD_SEARCH + "\r\n" + GOOD_HELD + "\r\n"
        split, _ = parse_fixture_csv(text)
        assert [f.id for f in split.search] == ["a"]
        assert [f.id for f in split.held_out] == ["b"]

    @staticmethod
    def test_quoted_comma_and_quote_verbatim():
        """Test quoted comma and quote verbatim."""
        text = (
            HDR + "\n"
            '"a","g1","search","x","say, hi","",""\n'
            '"q","g3","search","y","say ""yo""","",""\n' + GOOD_HELD + "\n"
        )
        split, _ = parse_fixture_csv(text)
        contents = [f.messages[0].content for f in split.search]
        assert contents == ["say, hi", 'say "yo"']

    @staticmethod
    def test_quoted_newline_verbatim():
        """Test quoted newline verbatim."""
        text = HDR + '\n"a","g1","search","x","line1\nline2","",""\n' + GOOD_HELD + "\n"
        split, _ = parse_fixture_csv(text)
        assert split.search[0].messages[0].content == "line1\nline2"

    @staticmethod
    def test_multiline_record_error_reports_record_end_line():
        """Test multiline record error reports record end line."""
        text = HDR + '\n"a","g1","search","x","ok","","after\ngap"\n' + GOOD_HELD + "\n"
        assert _expect_error(text) == (
            "row 3 (id 'a'): empty message followed by content at 'user_2'"
        )

    @staticmethod
    def test_gap_then_content_rejected():
        """Test gap then content rejected."""
        assert _expect_error(_doc("a,g1,search,x,hello,,LATE", GOOD_HELD)) == (
            "row 2 (id 'a'): empty message followed by content at 'user_2'"
        )

    @staticmethod
    def test_conversation_ending_with_assistant_rejected():
        """Test conversation ending with assistant rejected."""
        assert _expect_error(_doc("a,g1,search,x,hello,resp,", GOOD_HELD)) == (
            "row 2 (id 'a'): conversation must end with a user message"
        )

    @staticmethod
    def test_missing_user_1_rejected():
        """Test missing user 1 rejected."""
        assert _expect_error(_doc("a,g1,search,x,,,", GOOD_HELD)) == (
            "row 2 (id 'a'): fixture needs a user_1 message"
        )

    @staticmethod
    def test_whitespace_only_user_1_kept_verbatim():
        """Test whitespace only user 1 kept verbatim."""
        split, _ = parse_fixture_csv(_doc("a,g1,search,x,   ,,", GOOD_HELD))
        assert split.search[0].messages[0].content == "   "

    @staticmethod
    def test_whitespace_only_line_is_not_a_blank_line():
        """Test whitespace only line is not a blank line."""
        message = _expect_error(HDR + "\na,g1,search,x,hello,,\n   \n" + GOOD_HELD + "\n")
        assert message == "row 3: fixture id is blank"

    @staticmethod
    def test_cell_exactly_at_text_limit_accepted():
        """Test cell exactly at text limit accepted."""
        cell = "Z" * 16384
        split, _ = parse_fixture_csv(_doc(f"a,g1,search,x,{cell},,", GOOD_HELD))
        assert split.search[0].messages[0].content == cell

    @staticmethod
    def test_cell_over_text_limit_rejected():
        """Test cell over text limit rejected."""
        cell = "Z" * 16385
        assert _expect_error(_doc(f"a,g1,search,x,{cell},,", GOOD_HELD)) == (
            "row 2 (id 'a'): text field exceeds size limit"
        )

    @staticmethod
    def test_expected_over_text_limit_rejected():
        """Test expected over text limit rejected."""
        assert _expect_error(_doc(f"a,g1,search,{'E' * 16385},hello,,", GOOD_HELD)) == (
            "row 2 (id 'a'): text field exceeds size limit"
        )

    @staticmethod
    def test_21_message_columns_accepted():
        """Test 21 message columns accepted."""
        cols = ",".join(
            f"user_{(i // 2) + 1}" if i % 2 == 0 else f"assistant_{(i // 2) + 1}" for i in range(21)
        )
        header = f"id,group,partition,expected,{cols}"
        body = ",".join("m" for _ in range(21))
        split, _ = parse_fixture_csv(
            f"{header}\na,g1,search,x,{body}\nb,g2,held_out,y,{body.replace('m', 'n')}\n"
        )
        assert len(split.search[0].messages) == 21
        assert split.search[0].messages[-1].role == "user"

    @staticmethod
    def test_22_message_columns_rejected():
        """Test 22 message columns rejected."""
        cols = ",".join(
            f"user_{(i // 2) + 1}" if i % 2 == 0 else f"assistant_{(i // 2) + 1}" for i in range(22)
        )
        assert _expect_error(f"id,group,partition,expected,{cols}\n") == (
            "CSV needs 1-21 message columns after 'expected', got 22"
        )

    @staticmethod
    def test_header_without_message_columns_rejected():
        """Test header without message columns rejected."""
        assert _expect_error("id,group,partition,expected\na,g1,search,x\n") == (
            "CSV needs 1-21 message columns after 'expected', got 0"
        )

    @staticmethod
    def test_bad_header_column_order():
        """Test bad header column order."""
        message = _expect_error("id,group,expected,partition,user_1\n" + GOOD_SEARCH + "\n")
        assert message == (
            "CSV header must start with 'id,group,partition,expected', "
            "got 'id,group,expected,partition'"
        )

    @staticmethod
    def test_uppercase_header_column_rejected():
        """Test uppercase header column rejected."""
        message = _expect_error("ID,group,partition,expected,user_1\n" + GOOD_SEARCH + "\n")
        assert message == (
            "CSV header must start with 'id,group,partition,expected', "
            "got 'ID,group,partition,expected'"
        )

    @staticmethod
    def test_exactly_200_rows_accepted():
        """Test exactly 200 rows accepted."""
        rows = "".join(f"s{i},gs{i},search,x,hello\n" for i in range(100))
        rows += "".join(f"h{i},gh{i},held_out,y,hi\n" for i in range(100))
        split, _ = parse_fixture_csv("id,group,partition,expected,user_1\n" + rows)
        assert len(split.search) == 100
        assert len(split.held_out) == 100

    @staticmethod
    def test_201_rows_rejected():
        """Test 201 rows rejected."""
        rows = "".join(f"s{i},gs{i},search,x,hello\n" for i in range(101))
        rows += "".join(f"h{i},gh{i},held_out,y,hi\n" for i in range(100))
        assert _expect_error("id,group,partition,expected,user_1\n" + rows) == (
            "CSV holds more than 200 fixtures"
        )

    @staticmethod
    def test_101_search_fixtures_rejected():
        """Test 101 search fixtures rejected."""
        rows = "".join(f"s{i},gs{i},search,x,hello\n" for i in range(101))
        rows += "b,g2,held_out,y,hi\n"
        assert _expect_error("id,group,partition,expected,user_1\n" + rows) == (
            "CSV has more than 100 'search' fixtures"
        )

    def test_non_string_source_rejected(self):
        """Test non string source rejected."""
        with self.assertRaises(ValueError) as exc:
            parse_fixture_csv(b"bytes")
        assert str(exc.exception) == "CSV source must be text"

    def test_source_over_size_limit_rejected(self):
        """Test source over size limit rejected."""
        with self.assertRaises(ValueError) as exc:
            parse_fixture_csv("x" * (MAX_BYTES + 1))
        assert str(exc.exception) == "CSV source exceeds size limit"

    @staticmethod
    def test_source_exactly_at_size_limit_accepted():
        """Test source exactly at size limit accepted."""
        base = _doc(GOOD_SEARCH, GOOD_HELD)
        padding = "\n" * (MAX_BYTES - len(base.encode("utf-8")))
        split, _ = parse_fixture_csv(base + padding)
        assert [f.id for f in split.search] == ["a"]

    @staticmethod
    def test_empty_expected_ids_reported():
        """Test empty expected ids reported."""
        _, empty = parse_fixture_csv(_doc("a,g1,search,,hello,,", GOOD_HELD))
        assert empty == ["a"]

    def test_oversize_single_field_raises_valueerror(self):
        """Test oversize single field raises valueerror."""
        text = _doc(f"a,g1,search,x,{'Z' * 200_000},,", GOOD_HELD)
        assert len(text.encode("utf-8")) < MAX_BYTES
        with self.assertRaisesRegex(ValueError, "exceeds the parser limit"):
            parse_fixture_csv(text)

    def setUp(self):
        """Set up the test fixture."""
        self._td = tempfile.TemporaryDirectory(prefix="mpl-c107-")
        self.tmp_path = Path(self._td.name)

    def tearDown(self):
        """Tear down the test fixture."""
        self._td.cleanup()


class TestReadAuthoringFileAdversarial(unittest.TestCase):
    """Test read authoring file adversarial."""

    def _write(self, name, data: bytes):
        """Write."""
        tmp_path = self.tmp_path
        target = tmp_path / name
        target.write_bytes(data)
        return target

    def test_roundtrip_small_file(self):
        """Test roundtrip small file."""
        target = self._write("cases.csv", _doc(GOOD_SEARCH, GOOD_HELD).encode())
        assert read_authoring_file(target) == _doc(GOOD_SEARCH, GOOD_HELD)

    def test_str_and_path_both_accepted(self):
        """Test str and path both accepted."""
        target = self._write("cases.csv", b"x")
        assert read_authoring_file(str(target)) == "x"
        assert read_authoring_file(target) == "x"

    def test_utf16_with_bom_refused(self):
        """Test utf16 with bom refused."""
        target = self._write("u.csv", _doc(GOOD_SEARCH, GOOD_HELD).encode("utf-16"))
        with self.assertRaises(ValueError) as exc:
            read_authoring_file(target)
        assert str(exc.exception) == "Authoring file is not valid UTF-8"

    def test_bomless_utf16_decodes_then_fails_header_check(self):
        """Test bomless utf16 decodes then fails header check."""
        target = self._write("u.csv", _doc(GOOD_SEARCH, GOOD_HELD).encode("utf-16-le"))
        text = read_authoring_file(target)
        assert "\x00" in text
        message = _expect_error(text)
        # _safe() strips the NULs from the message, so the two sides look
        # identical while still differing -- confusing but loud, not silent.
        assert message == (
            "CSV header must start with 'id,group,partition,expected', "
            "got 'id,group,partition,expected'"
        )

    def test_latin1_refused(self):
        """Test latin1 refused."""
        target = self._write("l.csv", "a,g1,search,x,h\xe9llo\n".encode("latin-1"))
        with self.assertRaises(ValueError) as exc:
            read_authoring_file(target)
        assert str(exc.exception) == "Authoring file is not valid UTF-8"

    def test_utf8_bom_read_ok_but_header_rejected(self):
        """Test utf8 bom read ok but header rejected."""
        target = self._write("b.csv", ("\ufeff" + _doc(GOOD_SEARCH, GOOD_HELD)).encode("utf-8"))
        text = read_authoring_file(target)
        assert text.startswith("\ufeff")
        assert _expect_error(text) == (
            "CSV header must start with 'id,group,partition,expected', "
            "got '\\ufeffid,group,partition,expected'"
        )

    def test_over_size_limit_refused(self):
        """Test over size limit refused."""
        target = self._write("big.csv", b"a" * (MAX_BYTES + 1))
        with self.assertRaises(ValueError) as exc:
            read_authoring_file(target)
        assert str(exc.exception) == "Authoring file exceeds size limit"

    def test_exactly_size_limit_accepted(self):
        """Test exactly size limit accepted."""
        target = self._write("exact.csv", b"a" * MAX_BYTES)
        assert len(read_authoring_file(target)) == MAX_BYTES

    def test_empty_file_reads_empty(self):
        """Test empty file reads empty."""
        target = self._write("empty.csv", b"")
        assert read_authoring_file(target) == ""

    def test_non_regular_files_refused(self):
        """Test non regular files refused."""
        tmp_path = self.tmp_path
        for kind in ["fifo", "dir", "devnull", "fifo_symlink"]:
            with self.subTest(kind=kind):
                # Unique names per subtest: subTests share one tmp_path (pytest gave
                # each parametrized case a fresh dir).
                if kind == "fifo":
                    target = tmp_path / "p_fifo.csv"
                    os.mkfifo(target)
                elif kind == "dir":
                    target = tmp_path / "sub"
                    target.mkdir()
                elif kind == "devnull":
                    target = "/dev/null"
                else:
                    fifo = tmp_path / "p_link.csv"
                    os.mkfifo(fifo)
                    target = tmp_path / "link.csv"
                    target.symlink_to(fifo)
                with self.assertRaises(ValueError) as exc:
                    read_authoring_file(target)
                assert str(exc.exception) == "Authoring path is not a regular file"

    def test_missing_file(self):
        """Test missing file."""
        tmp_path = self.tmp_path
        with self.assertRaises(ValueError) as exc:
            read_authoring_file(tmp_path / "nope.csv")
        assert str(exc.exception) == "Unreadable authoring file"

    def test_dangling_symlink(self):
        """Test dangling symlink."""
        tmp_path = self.tmp_path
        target = tmp_path / "dangling.csv"
        target.symlink_to(tmp_path / "nope.csv")
        with self.assertRaises(ValueError) as exc:
            read_authoring_file(target)
        assert str(exc.exception) == "Unreadable authoring file"

    def test_symlink_loop(self):
        """Test symlink loop."""
        tmp_path = self.tmp_path
        first = tmp_path / "loop1.csv"
        second = tmp_path / "loop2.csv"
        first.symlink_to(second)
        second.symlink_to(first)
        with self.assertRaises(ValueError) as exc:
            read_authoring_file(first)
        assert str(exc.exception) == "Unreadable authoring file"

    def test_unreadable_file(self):
        """Test unreadable file."""
        target = self._write("noperm.csv", b"x")
        target.chmod(0o000)
        try:
            if os.access(target, os.R_OK):
                self.skipTest("process can still read the file (e.g. running as root)")
            with self.assertRaises(ValueError) as exc:
                read_authoring_file(target)
            assert str(exc.exception) == "Unreadable authoring file"
        finally:
            target.chmod(0o600)

    def test_null_byte_in_path(self):
        """Test null byte in path."""
        tmp_path = self.tmp_path
        # CPython changed the message across versions: "embedded null byte"
        # (<=3.12.3, 3.11) vs "open: embedded null character in path"
        # (newer 3.12 patches, 3.13+). Match the stable core of both.
        with self.assertRaisesRegex(ValueError, "embedded null (byte|character)"):
            read_authoring_file(tmp_path / "a\0b.csv")

    def test_non_path_types_rejected(self):
        """Test non path types rejected."""
        for bad in [123, b"bytes", None]:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError) as exc:
                    read_authoring_file(bad)
                assert str(exc.exception) == "Invalid authoring path"

    @unittest.skipIf(not os.path.exists("/proc/self/mem"), "Linux /proc only")
    def test_read_error_after_open_normalized_to_valueerror(self):
        """Test read error after open normalized to valueerror."""
        with self.assertRaisesRegex(ValueError, "Unreadable authoring file"):
            read_authoring_file("/proc/self/mem")

    def setUp(self):
        """Set up the test fixture."""
        self._td = tempfile.TemporaryDirectory(prefix="mpl-c107-")
        self.tmp_path = Path(self._td.name)

    def tearDown(self):
        """Tear down the test fixture."""
        self._td.cleanup()


class TestDatasetsLoaderAdversarial(unittest.TestCase):
    """Test datasets loader adversarial."""

    @staticmethod
    def _doc():
        """Doc."""
        import json

        def fx(i, content):
            """Fx."""
            return {
                "id": f"id{i}",
                "group": f"g{i}",
                "messages": [{"role": "user", "content": content}],
                "expected": "x",
            }

        return json.dumps({"search": [fx(0, "aaa")], "held_out": [fx(1, "bbb")]})

    def test_deeply_nested_arrays_become_valueerror(self):
        """Test deeply nested arrays become valueerror."""
        nested = "[" * 300_000 + "]" * 300_000
        text = (
            '{"search":[{"id":"a","group":"g","messages":' + nested + ',"expected":"x"}],'
            '"held_out":[{"id":"b","group":"h","messages":'
            '[{"role":"user","content":"hi"}],"expected":"y"}]}'
        )
        assert len(text.encode("utf-8")) < MAX_BYTES
        with self.assertRaises(ValueError) as exc:
            parse_dataset(text)
        assert str(exc.exception) == "Dataset nesting too deep"

    def test_text_over_size_limit_rejected(self):
        """Test text over size limit rejected."""
        with self.assertRaises(ValueError) as exc:
            parse_dataset(" " * (MAX_BYTES + 1))
        assert str(exc.exception) == "Dataset exceeds size limit"

    def test_text_exactly_at_size_limit_parses(self):
        """Test text exactly at size limit parses."""
        text = self._doc().ljust(MAX_BYTES)
        assert len(text.encode("utf-8")) == MAX_BYTES
        split = parse_dataset(text)
        assert [f.id for f in split.search] == ["id0"]

    def test_csv_and_json_loaders_agree_on_nfc_discipline(self):
        """Test csv and json loaders agree on nfc discipline."""
        nfc = unicodedata.normalize("NFC", "caf\u00e9")
        nfd = unicodedata.normalize("NFD", "caf\u00e9")
        csv_text = _doc(f"{nfc},g1,search,x,hello,,", f"{nfd},g2,held_out,y,hi,,")
        with self.assertRaisesRegex(ValueError, "Fixture IDs must be globally unique"):
            parse_fixture_csv(csv_text)
        import json

        payload = {
            "search": [
                {
                    "id": nfc,
                    "group": "g1",
                    "messages": [{"role": "user", "content": "hello"}],
                    "expected": "x",
                }
            ],
            "held_out": [
                {
                    "id": nfd,
                    "group": "g2",
                    "messages": [{"role": "user", "content": "hi"}],
                    "expected": "y",
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "Fixture IDs must be globally unique"):
            parse_dataset(json.dumps(payload))

    def test_file_roundtrip_through_both_readers(self):
        """Test file roundtrip through both readers."""
        tmp_path = self.tmp_path
        target = tmp_path / "cases.csv"
        target.write_text(_doc(GOOD_SEARCH, GOOD_HELD), encoding="utf-8")
        split, _ = parse_fixture_csv(read_authoring_file(target))
        assert [f.id for f in split.search] == ["a"]
        assert [f.id for f in split.held_out] == ["b"]
        # The built JSON must satisfy the strict runtime loader by construction.
        rebuilt = fixtures_mod.split_to_json_text(split)
        reloaded = load_dataset(self._write_json(rebuilt))
        assert [f.id for f in reloaded.search] == ["a"]

    def _write_json(self, text):
        """Write json."""
        tmp_path = self.tmp_path
        target = tmp_path / "fixtures.json"
        target.write_text(text, encoding="utf-8")
        return target

    @unittest.skipIf(not os.path.exists("/proc/self/mem"), "Linux /proc only")
    def test_load_dataset_read_error_normalized_to_valueerror(self):
        """Test load dataset read error normalized to valueerror."""
        with self.assertRaisesRegex(ValueError, "Unreadable fixture file"):
            load_dataset("/proc/self/mem")

    def setUp(self):
        """Set up the test fixture."""
        self._td = tempfile.TemporaryDirectory(prefix="mpl-c107-")
        self.tmp_path = Path(self._td.name)

    def tearDown(self):
        """Tear down the test fixture."""
        self._td.cleanup()


if __name__ == "__main__":
    unittest.main()
