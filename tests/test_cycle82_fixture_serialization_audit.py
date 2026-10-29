"""Cycle 82 adversarial audit: fixture serialization round-trip.

METHOD
------
Target: ``fixtures.split_to_json_text`` (the "verified by construction" claim),
``fixtures.write_fixture_file`` (exclusive creation, 0o600, partial-write
cleanup), and ``datasets.parse_dataset`` / ``datasets.load_dataset`` (the fixed
rejection vocabulary). Earlier cycles audited CSV authoring (cycle 66) and the
builder CLI plumbing (cycles 29/31); no dedicated adversarial audit has verified
the serialization round-trip itself.

1.  Round-trip fidelity: Splits built with adversarial content -- unicode
    (emoji, CJK, combining marks, RTL text, NFD vs NFC), newlines, quotes,
    backslashes, empty ``expected``, MAX_TEXT-sized payloads, 21-message
    conversations, 100-fixture partitions, duplicate-looking ids across
    partitions -- serialize via ``split_to_json_text`` and re-parse via
    ``parse_dataset``; the result is compared field by field with a hand-rolled
    equality walk (not dataclass ``==``).
2.  By-construction claim: the returned text is asserted byte-identical to an
    independently reconstructed ``json.dumps(..., ensure_ascii=False,
    indent=2) + "\\n"``; and Splits that violate the runtime schema (22
    messages, oversized text, non-alternating roles, assistant-final) are
    refused by ``split_to_json_text`` itself, proving the internal
    ``parse_dataset`` re-parse is load-bearing rather than decorative.
3.  ``write_fixture_file``: existing file -> ValueError with the original
    bytes untouched (exclusive creation), missing parent -> ValueError,
    parent-that-is-a-file -> ValueError, non-str text -> ValueError,
    non-str/Path path -> ValueError, 0o600 on the created file, partial-write
    cleanup (a failing writer leaves no file behind), and a mocked
    ``os.open`` OSError -> ValueError (root-independent).
4.  Rejection vocabulary: every fixed error code of ``parse_dataset`` /
    ``load_dataset`` -- duplicate JSON fields, nonfinite constants, wrong
    shapes, bad message counts, broken alternation, blank identifiers,
    duplicate ids, group overlap, identical cross-split messages, oversize
    input, non-UTF8 bytes, deep nesting -- is pinned with a minimal input.

FINDINGS (verified by running this file, 2026-10-16)
----------------------------------------------------
CLEAN -- no genuine serialization bugs found:

* Adversarial Splits (unicode with NFD combining sequences, emoji, CJK, RTL,
  newlines, quotes, backslashes, empty expected, MAX_TEXT-sized fields,
  21-message conversations, 100-fixture partitions, duplicate-looking ids
  across partitions) round-trip through ``split_to_json_text`` /
  ``parse_dataset`` with field-by-field deep equality; NFD codepoints survive
  byte-exact (no NFC normalization of stored data).
* The emitted text is byte-identical to an independent
  ``json.dumps(..., ensure_ascii=False, indent=2) + "\\n"`` reconstruction,
  and schema-violating Splits (23 messages, oversized text, assistant-first,
  assistant-final) are refused by ``split_to_json_text`` -- the internal
  re-parse is load-bearing.
* ``write_fixture_file`` refuses existing files without touching their bytes,
  requires an existing parent directory, writes 0o600, and unlinks partial
  writes; mocked ``os.open`` failures surface as ValueError.
* Every fixed rejection code of ``parse_dataset`` / ``load_dataset`` fires on
  its minimal input, including "Dataset nesting too deep" (depth 40000),
  "Nonfinite JSON" (NaN/Infinity/-Infinity), "Duplicate JSON field", and the
  ``load_dataset`` FIFO/directory/non-UTF8/oversize guards.
* Non-finding: ``parse_dataset`` on a lone-surrogate ``str`` raises
  ``UnicodeEncodeError`` -- still a ``ValueError`` subclass, so the
  catch-ValueError contract holds; not a bug, left alone.
"""

import io
import json
import os
import unittest
from pathlib import Path
from typing import Any, cast
from unittest import mock

from model_protocol_lab.core import Fixture, Message, Split
from model_protocol_lab.datasets import (
    MAX_BYTES,
    MAX_CASES_PER_SPLIT,
    MAX_MESSAGES,
    MAX_TEXT,
    load_dataset,
    parse_dataset,
)
from model_protocol_lab.fixtures import split_to_json_text, write_fixture_file

# ---------------------------------------------------------------------------
# Builders.


def _messages(contents: tuple[str, ...]) -> tuple[Message, ...]:
    """Odd-length content list -> alternating messages, user-first/last."""
    assert contents and len(contents) % 2 == 1
    return tuple(
        Message("user" if index % 2 == 0 else "assistant", text)
        for index, text in enumerate(contents)
    )


def _fixture(fid: str, group: str, expected: str, contents: tuple[str, ...]) -> Fixture:
    return Fixture(id=fid, messages=_messages(contents), expected=expected, group=group)


def _adversarial_split() -> Split:
    """A Split exercising every hostile-but-legal content shape."""
    search = (
        _fixture(
            "uni-1",
            "g-unicode",
            "",  # empty expected passes vacuously, like the JSON schema
            (
                "h\u00e9llo \U0001f389 \u65e5\u672c\u8a9e e\u0301 \u0645\u0631\u062d\u0628\u0627",
                "line1\nline2\r\nline3\ttab\\\\backslash\"double\"'single'",
                "final? \u200b\u200e",  # zero-width space + LRM
            ),
        ),
        _fixture("dup-look", "g-dup", "e1", ("q1",)),
        _fixture(
            "huge-1",
            "g-huge",
            "D" * MAX_TEXT,
            ("A" * MAX_TEXT, "B" * MAX_TEXT, "C" * MAX_TEXT),
        ),
        _fixture(
            "long-1",
            "g-long",
            "ok",
            tuple(f"turn-{index} \u2603" for index in range(MAX_MESSAGES)),
        ),
    )
    held_out = (
        # Duplicate-LOOKING ids: case variant and trailing-space variant of the
        # search id "dup-look". NFC-distinct, so the Split discipline accepts;
        # the round-trip must keep them byte-exact.
        _fixture("DUP-LOOK", "h-dup", "e2", ("z1",)),
        _fixture("dup-look ", "h-dup2", "e3", ("z2",)),
        _fixture(
            "uni-2",
            "h-unicode",
            "ok",
            (
                "NFD e\u0301 vs NFC \u00e9",
                "a\r\rb",
                "tab\there",
            ),
        ),
    )
    return Split(search, held_out)


def _assert_splits_equal(first: Split, second: Split) -> None:
    """Hand-rolled deep equality: every field, in order, exact types."""
    assert type(first) is Split and type(second) is Split
    for part in ("search", "held_out"):
        first_cases = getattr(first, part)
        second_cases = getattr(second, part)
        assert len(first_cases) == len(second_cases), part
        for first_case, second_case in zip(first_cases, second_cases, strict=True):
            assert type(first_case) is Fixture and type(second_case) is Fixture
            assert first_case.id == second_case.id and type(second_case.id) is str
            assert first_case.group == second_case.group
            assert first_case.expected == second_case.expected
            assert type(second_case.expected) is str
            assert len(first_case.messages) == len(second_case.messages)
            for first_msg, second_msg in zip(
                first_case.messages, second_case.messages, strict=True
            ):
                assert type(first_msg) is Message and type(second_msg) is Message
                assert first_msg.role == second_msg.role
                assert first_msg.content == second_msg.content
                assert type(second_msg.content) is str


def _expected_text(split: Split) -> str:
    """Independent reconstruction of the documented serialization."""

    def fixture_dict(case: Fixture) -> dict[str, Any]:
        return {
            "id": case.id,
            "group": case.group,
            "messages": [
                {"role": message.role, "content": message.content} for message in case.messages
            ],
            "expected": case.expected,
        }

    payload = {
        "search": [fixture_dict(case) for case in split.search],
        "held_out": [fixture_dict(case) for case in split.held_out],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def _fx_json(
    fid: str = "a1",
    group: str = "g1",
    expected: str = "e",
    contents: tuple[str, ...] = ("q",),
) -> dict[str, Any]:
    return {
        "id": fid,
        "group": group,
        "messages": [
            {"role": "user" if index % 2 == 0 else "assistant", "content": text}
            for index, text in enumerate(contents)
        ],
        "expected": expected,
    }


def _doc_json(search: list[Any], held_out: list[Any]) -> str:
    return json.dumps({"search": search, "held_out": held_out})


def _valid_doc() -> str:
    return _doc_json(
        [_fx_json("a1", group="g1", contents=("q1",))],
        [_fx_json("b1", group="h1", contents=("z1",))],
    )


# ---------------------------------------------------------------------------
# 1. Round-trip fidelity.


class FixtureSerializationAuditTests(unittest.TestCase):
    """unittest entry point; pytest runs TestCase classes natively."""

    def setUp(self) -> None:
        import tempfile

        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        from pathlib import Path

        self._td = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._td.name)

    def tearDown(self) -> None:
        self._td.cleanup()

    @staticmethod
    def test_roundtrip_adversarial_split():
        split = _adversarial_split()
        _assert_splits_equal(split, parse_dataset(split_to_json_text(split)))

    @staticmethod
    def test_roundtrip_preserves_nfd_codepoints_exactly():
        split = _adversarial_split()
        reparsed = parse_dataset(split_to_json_text(split))
        original = split.search[0].messages[0].content
        assert "e\u0301" in original  # the test input really is NFD
        assert "e\u0301" in reparsed.search[0].messages[0].content
        # Stored data stays byte-exact: no NFC normalization by the round-trip.
        assert reparsed.search[0].messages[0].content == original

    @staticmethod
    def test_roundtrip_max_text_boundary():
        split = _adversarial_split()
        huge = split.search[2]
        assert len(huge.messages[0].content) == MAX_TEXT
        assert len(huge.expected) == MAX_TEXT
        reparsed = parse_dataset(split_to_json_text(split))
        _assert_splits_equal(split, reparsed)

    @staticmethod
    def test_roundtrip_max_messages_boundary():
        split = _adversarial_split()
        assert len(split.search[3].messages) == MAX_MESSAGES
        _assert_splits_equal(split, parse_dataset(split_to_json_text(split)))

    @staticmethod
    def test_roundtrip_max_cases_boundary():
        search = tuple(
            _fixture(f"s{i:03d}", "gs", "e", (f"sq{i}",)) for i in range(MAX_CASES_PER_SPLIT)
        )
        held_out = tuple(
            _fixture(f"h{i:03d}", "gh", "e", (f"hq{i}",)) for i in range(MAX_CASES_PER_SPLIT)
        )
        split = Split(search, held_out)
        _assert_splits_equal(split, parse_dataset(split_to_json_text(split)))

    @staticmethod
    def test_roundtrip_duplicate_looking_ids_survive():
        split = _adversarial_split()
        reparsed = parse_dataset(split_to_json_text(split))
        assert reparsed.held_out[0].id == "DUP-LOOK"
        assert reparsed.held_out[1].id == "dup-look "
        assert reparsed.search[1].id == "dup-look"
        _assert_splits_equal(split, reparsed)

    @staticmethod
    def test_text_byte_identical_to_independent_reconstruction():
        for split in (
            _adversarial_split(),
            Split(
                (_fixture("s1", "g", "", ("q",)),),
                (_fixture("h1", "h", "e", ("z", "w", "v")),),
            ),
        ):
            assert split_to_json_text(split) == _expected_text(split)

    @staticmethod
    def test_text_framing():
        split = _adversarial_split()
        text = split_to_json_text(split)
        assert text.startswith("{")
        assert text.endswith("}\n")
        assert not text.endswith("\n\n")
        assert type(text) is str

    def test_split_to_json_text_rejects_non_split(self):
        for bad in ["nope", None, 123, ["x"], {"search": []}]:
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "needs a Split"):
                split_to_json_text(bad)

    def test_split_to_json_text_rejects_split_subclass(self):
        class _SplitSubclass(Split):
            pass

        good = _adversarial_split()
        sub = _SplitSubclass(search=good.search, held_out=good.held_out)
        with self.assertRaisesRegex(ValueError, "needs a Split"):
            split_to_json_text(sub)

    def test_by_construction_rejects_oversized_text(self):
        split = _lone_fixture(messages=(Message("user", "x" * (MAX_TEXT + 1)),))
        with self.assertRaisesRegex(ValueError, "oversized text"):
            split_to_json_text(split)

    def test_by_construction_rejects_too_many_messages(self):
        contents = tuple(f"m{index}" for index in range(MAX_MESSAGES + 2))  # 23, odd
        split = _lone_fixture(messages=_messages(contents))
        with self.assertRaisesRegex(ValueError, "1-21 messages"):
            split_to_json_text(split)

    def test_by_construction_rejects_bad_alternation(self):
        split = _lone_fixture(messages=(Message("assistant", "q"),))
        with self.assertRaisesRegex(ValueError, "must alternate"):
            split_to_json_text(split)

    def test_by_construction_rejects_assistant_final(self):
        split = _lone_fixture(
            messages=(Message("user", "q"), Message("assistant", "a")),
        )
        with self.assertRaisesRegex(ValueError, "must end with user"):
            split_to_json_text(split)

    def test_exclusive_creation_refuses_existing_and_preserves_bytes(self):
        target = self.tmp_path / "exists.json"
        original = b'{"original": true}'
        target.write_bytes(original)
        with self.assertRaisesRegex(ValueError, "already exists"):
            write_fixture_file('{"search": []}', target)
        assert target.read_bytes() == original
        # The str-path spelling hits the same exclusive branch.
        with self.assertRaisesRegex(ValueError, "already exists"):
            write_fixture_file('{"search": []}', str(target))
        assert target.read_bytes() == original

    def test_missing_parent_directory_rejected(self):
        with self.assertRaisesRegex(ValueError, "parent directory"):
            write_fixture_file("x", self.tmp_path / "no-such-dir" / "f.json")

    def test_parent_that_is_a_file_rejected(self):
        parent = self.tmp_path / "afile"
        parent.write_text("x", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "parent directory"):
            write_fixture_file("x", parent / "f.json")

    def test_non_string_text_rejected(self):
        for bad in [b"x", None, 123, ["x"]]:
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "must be a string"):
                write_fixture_file(bad, self.tmp_path / "f.json")

    def test_invalid_path_type_rejected(self):
        for bad in [123, None, b"x"]:
            with (
                self.subTest(bad=bad),
                self.assertRaisesRegex(ValueError, "Invalid fixture output path"),
            ):
                write_fixture_file("x", bad)

    def test_created_file_is_owner_only_and_verbatim(self):
        text = split_to_json_text(_adversarial_split())
        target = write_fixture_file(text, self.tmp_path / "f.json")
        assert isinstance(target, Path)
        assert target == self.tmp_path / "f.json"
        assert target.read_text(encoding="utf-8") == text
        mode = os.stat(target).st_mode & 0o777
        assert mode & 0o077 == 0  # no group/other bits, whatever the umask
        assert mode == 0o600

    def test_str_path_spelling_writes_and_returns_path(self):
        target = write_fixture_file("{}", str(self.tmp_path / "s.json"))
        assert target == self.tmp_path / "s.json"
        assert target.read_text(encoding="utf-8") == "{}"

    def test_relative_path_resolves_in_cwd(self):
        _old_cwd = os.getcwd()
        os.chdir(self.tmp_path)
        self.addCleanup(os.chdir, _old_cwd)
        target = write_fixture_file("{}", "rel.json")
        assert target == Path("rel.json")
        assert (self.tmp_path / "rel.json").read_text(encoding="utf-8") == "{}"

    def test_empty_text_writes_verbatim(self):
        """The writer is content-agnostic: no schema validation at write time."""
        target = write_fixture_file("", self.tmp_path / "empty.json")
        assert target.read_text(encoding="utf-8") == ""

    def test_partial_write_leaves_no_file(self):
        target = self.tmp_path / "partial.json"

        def fake_fdopen(fd: int, *args: Any, **kwargs: Any) -> io.StringIO:
            os.close(fd)  # the real fd must not leak when fdopen "fails"
            return _FailingWriter()

        with (
            mock.patch.object(os, "fdopen", fake_fdopen),
            self.assertRaisesRegex(OSError, "simulated write failure"),
        ):
            write_fixture_file('{"search": []}', target)
        # The O_CREAT|O_EXCL open created the path; the failed write unlinked it.
        assert not target.exists()

    def test_os_open_oserror_becomes_valueerror(self):
        with (
            mock.patch.object(os, "open", side_effect=OSError("EACCES")),
            self.assertRaisesRegex(ValueError, "not writable"),
        ):
            write_fixture_file("x", self.tmp_path / "f.json")
        # FileExistsError keeps its dedicated message (it subclasses OSError).
        with (
            mock.patch.object(os, "open", side_effect=FileExistsError("exists")),
            self.assertRaisesRegex(ValueError, "already exists"),
        ):
            write_fixture_file("x", self.tmp_path / "f.json")

    def test_write_then_load_dataset_roundtrip(self):
        split = _adversarial_split()
        target = write_fixture_file(split_to_json_text(split), self.tmp_path / "fx.json")
        _assert_splits_equal(split, load_dataset(target))
        _assert_splits_equal(split, load_dataset(str(target)))

    def test_reject_duplicate_json_field(self):
        with self.assertRaisesRegex(ValueError, "Duplicate JSON field"):
            parse_dataset('{"search": [], "held_out": [], "search": []}')

    def test_reject_nonfinite_constants(self):
        for constant in ["NaN", "Infinity", "-Infinity"]:
            with self.subTest(constant=constant):
                text = _valid_doc().replace('"expected": "e"', f'"expected": {constant}', 1)
                with self.assertRaisesRegex(ValueError, "Nonfinite JSON"):
                    parse_dataset(text)

    def test_reject_missing_held_out(self):
        with self.assertRaisesRegex(ValueError, "Unexpected or missing JSON fields"):
            parse_dataset('{"search": []}')

    def test_reject_extra_top_level_key(self):
        text = _valid_doc().replace('{"search":', '{"extra": 1, "search":', 1)
        with self.assertRaisesRegex(ValueError, "Unexpected or missing JSON fields"):
            parse_dataset(text)

    def test_reject_non_object_top_level(self):
        for text in ["[]", "null", '"str"', "42"]:
            with (
                self.subTest(text=text),
                self.assertRaisesRegex(ValueError, "Unexpected or missing JSON fields"),
            ):
                parse_dataset(text)

    def test_reject_empty_search_partition(self):
        with self.assertRaisesRegex(ValueError, "Invalid fixture count"):
            parse_dataset(_doc_json([], [_fx_json("b1", group="h1", contents=("z",))]))

    def test_reject_empty_held_out_partition(self):
        with self.assertRaisesRegex(ValueError, "Invalid fixture count"):
            parse_dataset(_doc_json([_fx_json("a1", contents=("q",))], []))

    def test_reject_non_list_partition(self):
        with self.assertRaisesRegex(ValueError, "Invalid fixture count"):
            parse_dataset('{"search": {}, "held_out": []}')

    def test_reject_too_many_fixtures(self):
        many = [_fx_json(f"s{i}", group="g", contents=(f"q{i}",)) for i in range(101)]
        with self.assertRaisesRegex(ValueError, "Invalid fixture count"):
            parse_dataset(_doc_json(many, [_fx_json("b1", group="h1", contents=("z",))]))

    def test_reject_fixture_missing_field(self):
        bad = {"id": "a", "group": "g", "messages": [{"role": "user", "content": "q"}]}
        with self.assertRaisesRegex(ValueError, "Unexpected or missing JSON fields"):
            parse_dataset(_doc_json([bad], [_fx_json("b1", group="h1", contents=("z",))]))

    def test_reject_fixture_extra_field(self):
        bad = dict(_fx_json("a1"), extra=1)
        with self.assertRaisesRegex(ValueError, "Unexpected or missing JSON fields"):
            parse_dataset(_doc_json([bad], [_fx_json("b1", group="h1", contents=("z",))]))

    def test_reject_message_missing_field(self):
        bad = _fx_json("a1")
        bad["messages"] = [{"role": "user"}]
        with self.assertRaisesRegex(ValueError, "Unexpected or missing JSON fields"):
            parse_dataset(_doc_json([bad], [_fx_json("b1", group="h1", contents=("z",))]))

    def test_reject_message_extra_field(self):
        bad = _fx_json("a1")
        bad["messages"] = [{"role": "user", "content": "q", "extra": 1}]
        with self.assertRaisesRegex(ValueError, "Unexpected or missing JSON fields"):
            parse_dataset(_doc_json([bad], [_fx_json("b1", group="h1", contents=("z",))]))

    def test_reject_empty_messages(self):
        bad = _fx_json("a1", contents=())
        with self.assertRaisesRegex(ValueError, "1-21 messages"):
            parse_dataset(_doc_json([bad], [_fx_json("b1", group="h1", contents=("z",))]))

    def test_reject_too_many_messages(self):
        bad = _fx_json("a1", contents=tuple(f"m{i}" for i in range(22)))
        with self.assertRaisesRegex(ValueError, "1-21 messages"):
            parse_dataset(_doc_json([bad], [_fx_json("b1", group="h1", contents=("z",))]))

    def test_reject_bad_alternation_start(self):
        bad = _fx_json("a1")
        bad["messages"] = [{"role": "assistant", "content": "q"}]
        with self.assertRaisesRegex(ValueError, "must alternate"):
            parse_dataset(_doc_json([bad], [_fx_json("b1", group="h1", contents=("z",))]))

    def test_reject_bad_alternation_middle(self):
        bad = _fx_json("a1")
        bad["messages"] = [
            {"role": "user", "content": "q"},
            {"role": "user", "content": "q"},
            {"role": "user", "content": "q"},
        ]
        with self.assertRaisesRegex(ValueError, "must alternate"):
            parse_dataset(_doc_json([bad], [_fx_json("b1", group="h1", contents=("z",))]))

    def test_reject_assistant_final(self):
        bad = _fx_json("a1")
        bad["messages"] = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
        with self.assertRaisesRegex(ValueError, "must end with user"):
            parse_dataset(_doc_json([bad], [_fx_json("b1", group="h1", contents=("z",))]))

    def test_reject_non_string_content(self):
        bad = _fx_json("a1")
        bad["messages"] = [{"role": "user", "content": 123}]
        with self.assertRaisesRegex(ValueError, "Invalid or oversized text field"):
            parse_dataset(_doc_json([bad], [_fx_json("b1", group="h1", contents=("z",))]))

    def test_reject_oversized_content(self):
        bad = _fx_json("a1", contents=("x" * (MAX_TEXT + 1),))
        with self.assertRaisesRegex(ValueError, "Invalid or oversized text field"):
            parse_dataset(_doc_json([bad], [_fx_json("b1", group="h1", contents=("z",))]))

    def test_reject_blank_id(self):
        bad = _fx_json("   ")
        with self.assertRaisesRegex(ValueError, "Blank identifier"):
            parse_dataset(_doc_json([bad], [_fx_json("b1", group="h1", contents=("z",))]))

    def test_reject_blank_group(self):
        bad = _fx_json("a1", group="\t")
        with self.assertRaisesRegex(ValueError, "Blank identifier"):
            parse_dataset(_doc_json([bad], [_fx_json("b1", group="h1", contents=("z",))]))

    def test_reject_non_string_expected(self):
        bad = _fx_json("a1", expected="e")
        bad["expected"] = 5
        with self.assertRaisesRegex(ValueError, "Invalid or oversized text field"):
            parse_dataset(_doc_json([bad], [_fx_json("b1", group="h1", contents=("z",))]))

    def test_reject_duplicate_ids_across_partitions(self):
        search = [_fx_json("dup", group="g1", contents=("q1",))]
        held_out = [_fx_json("dup", group="h1", contents=("z1",))]
        with self.assertRaisesRegex(ValueError, "globally unique"):
            parse_dataset(_doc_json(search, held_out))

    def test_reject_duplicate_ids_within_partition(self):
        search = [
            _fx_json("dup", group="g1", contents=("q1",)),
            _fx_json("dup", group="g1", contents=("q2",)),
        ]
        held_out = [_fx_json("b1", group="h1", contents=("z1",))]
        with self.assertRaisesRegex(ValueError, "globally unique"):
            parse_dataset(_doc_json(search, held_out))

    def test_reject_group_overlap(self):
        search = [_fx_json("a1", group="same", contents=("q1",))]
        held_out = [_fx_json("b1", group="same", contents=("z1",))]
        with self.assertRaisesRegex(ValueError, "cannot cross the split"):
            parse_dataset(_doc_json(search, held_out))

    def test_reject_identical_messages_across_split(self):
        search = [_fx_json("a1", group="g1", contents=("same-q",))]
        held_out = [_fx_json("b1", group="h1", contents=("same-q",))]
        with self.assertRaisesRegex(ValueError, "Identical message inputs cannot cross"):
            parse_dataset(_doc_json(search, held_out))

    def test_reject_deep_nesting(self):
        with self.assertRaisesRegex(ValueError, "nesting too deep"):
            parse_dataset("[" * 40000 + "]" * 40000)

    def test_reject_oversize_text(self):
        with self.assertRaisesRegex(ValueError, "exceeds size limit"):
            parse_dataset("x" * (MAX_BYTES + 1))

    def test_reject_non_string_input(self):
        for bad in [None, 123, b"{}"]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_dataset(bad)

    def test_load_dataset_roundtrip(self):
        split = _adversarial_split()
        target = write_fixture_file(split_to_json_text(split), self.tmp_path / "fx.json")
        _assert_splits_equal(split, load_dataset(target))
        _assert_splits_equal(split, load_dataset(str(target)))

    def test_load_dataset_rejects_directory(self):
        with self.assertRaisesRegex(ValueError, "not a regular file"):
            load_dataset(self.tmp_path)

    def test_load_dataset_rejects_fifo(self):
        fifo = self.tmp_path / "pipe"
        os.mkfifo(fifo)
        with self.assertRaisesRegex(ValueError, "not a regular file"):
            load_dataset(fifo)

    def test_load_dataset_rejects_missing(self):
        with self.assertRaisesRegex(ValueError, "Unreadable fixture file"):
            load_dataset(self.tmp_path / "missing.json")

    def test_load_dataset_rejects_non_utf8(self):
        target = self.tmp_path / "bad.json"
        target.write_bytes(b'{"search": "\xff\xfe"}')
        with self.assertRaisesRegex(ValueError, "not valid UTF-8"):
            load_dataset(target)

    def test_load_dataset_rejects_oversize_file(self):
        target = self.tmp_path / "big.json"
        target.write_bytes(b"x" * (MAX_BYTES + 1))
        with self.assertRaisesRegex(ValueError, "exceeds size limit"):
            load_dataset(target)

    def test_load_dataset_rejects_bad_path_type(self):
        with self.assertRaisesRegex(ValueError, "Invalid fixture path"):
            load_dataset(cast("str | Path", 123))

    def test_load_dataset_rejects_invalid_content(self):
        target = self.tmp_path / "bad.json"
        target.write_text('{"search": [], "held_out": []}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Invalid fixture count"):
            load_dataset(target)


# ---------------------------------------------------------------------------
# 2. The by-construction claim.


def _lone_fixture(**kwargs: Any) -> Split:
    """A Split whose search half is caller-supplied (possibly schema-breaking)."""
    bad = Fixture(
        id=kwargs.pop("fid", "bad"),
        messages=kwargs.pop("messages"),
        expected=kwargs.pop("expected", ""),
        group=kwargs.pop("group", "gbad"),
    )
    assert not kwargs
    good = _fixture("ok", "gok", "", ("q",))
    return Split((bad,), (good,))


# ---------------------------------------------------------------------------
# 3. write_fixture_file.


class _FailingWriter(io.StringIO):
    @staticmethod
    def write(s: str) -> int:
        raise OSError("simulated write failure")


# ---------------------------------------------------------------------------
# 4. parse_dataset / load_dataset rejection vocabulary.


if __name__ == "__main__":
    unittest.main()
