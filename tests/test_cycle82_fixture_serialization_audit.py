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
from pathlib import Path
from typing import Any, cast
from unittest import mock

import pytest

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


def test_roundtrip_adversarial_split() -> None:
    split = _adversarial_split()
    _assert_splits_equal(split, parse_dataset(split_to_json_text(split)))


def test_roundtrip_preserves_nfd_codepoints_exactly() -> None:
    split = _adversarial_split()
    reparsed = parse_dataset(split_to_json_text(split))
    original = split.search[0].messages[0].content
    assert "e\u0301" in original  # the test input really is NFD
    assert "e\u0301" in reparsed.search[0].messages[0].content
    # Stored data stays byte-exact: no NFC normalization by the round-trip.
    assert reparsed.search[0].messages[0].content == original


def test_roundtrip_max_text_boundary() -> None:
    split = _adversarial_split()
    huge = split.search[2]
    assert len(huge.messages[0].content) == MAX_TEXT
    assert len(huge.expected) == MAX_TEXT
    reparsed = parse_dataset(split_to_json_text(split))
    _assert_splits_equal(split, reparsed)


def test_roundtrip_max_messages_boundary() -> None:
    split = _adversarial_split()
    assert len(split.search[3].messages) == MAX_MESSAGES
    _assert_splits_equal(split, parse_dataset(split_to_json_text(split)))


def test_roundtrip_max_cases_boundary() -> None:
    search = tuple(
        _fixture(f"s{i:03d}", "gs", "e", (f"sq{i}",)) for i in range(MAX_CASES_PER_SPLIT)
    )
    held_out = tuple(
        _fixture(f"h{i:03d}", "gh", "e", (f"hq{i}",)) for i in range(MAX_CASES_PER_SPLIT)
    )
    split = Split(search, held_out)
    _assert_splits_equal(split, parse_dataset(split_to_json_text(split)))


def test_roundtrip_duplicate_looking_ids_survive() -> None:
    split = _adversarial_split()
    reparsed = parse_dataset(split_to_json_text(split))
    assert reparsed.held_out[0].id == "DUP-LOOK"
    assert reparsed.held_out[1].id == "dup-look "
    assert reparsed.search[1].id == "dup-look"
    _assert_splits_equal(split, reparsed)


# ---------------------------------------------------------------------------
# 2. The by-construction claim.


def test_text_byte_identical_to_independent_reconstruction() -> None:
    for split in (
        _adversarial_split(),
        Split(
            (_fixture("s1", "g", "", ("q",)),),
            (_fixture("h1", "h", "e", ("z", "w", "v")),),
        ),
    ):
        assert split_to_json_text(split) == _expected_text(split)


def test_text_framing() -> None:
    split = _adversarial_split()
    text = split_to_json_text(split)
    assert text.startswith("{")
    assert text.endswith("}\n")
    assert not text.endswith("\n\n")
    assert type(text) is str


@pytest.mark.parametrize("bad", ["nope", None, 123, ["x"], {"search": []}])
def test_split_to_json_text_rejects_non_split(bad: Any) -> None:
    with pytest.raises(ValueError, match="needs a Split"):
        split_to_json_text(bad)


def test_split_to_json_text_rejects_split_subclass() -> None:
    class _SplitSubclass(Split):
        pass

    good = _adversarial_split()
    sub = _SplitSubclass(search=good.search, held_out=good.held_out)
    with pytest.raises(ValueError, match="needs a Split"):
        split_to_json_text(sub)


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


def test_by_construction_rejects_oversized_text() -> None:
    split = _lone_fixture(messages=(Message("user", "x" * (MAX_TEXT + 1)),))
    with pytest.raises(ValueError, match="oversized text"):
        split_to_json_text(split)


def test_by_construction_rejects_too_many_messages() -> None:
    contents = tuple(f"m{index}" for index in range(MAX_MESSAGES + 2))  # 23, odd
    split = _lone_fixture(messages=_messages(contents))
    with pytest.raises(ValueError, match="1-21 messages"):
        split_to_json_text(split)


def test_by_construction_rejects_bad_alternation() -> None:
    split = _lone_fixture(messages=(Message("assistant", "q"),))
    with pytest.raises(ValueError, match="must alternate"):
        split_to_json_text(split)


def test_by_construction_rejects_assistant_final() -> None:
    split = _lone_fixture(
        messages=(Message("user", "q"), Message("assistant", "a")),
    )
    with pytest.raises(ValueError, match="must end with user"):
        split_to_json_text(split)


# ---------------------------------------------------------------------------
# 3. write_fixture_file.


def test_exclusive_creation_refuses_existing_and_preserves_bytes(tmp_path: Path) -> None:
    target = tmp_path / "exists.json"
    original = b'{"original": true}'
    target.write_bytes(original)
    with pytest.raises(ValueError, match="already exists"):
        write_fixture_file('{"search": []}', target)
    assert target.read_bytes() == original
    # The str-path spelling hits the same exclusive branch.
    with pytest.raises(ValueError, match="already exists"):
        write_fixture_file('{"search": []}', str(target))
    assert target.read_bytes() == original


def test_missing_parent_directory_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="parent directory"):
        write_fixture_file("x", tmp_path / "no-such-dir" / "f.json")


def test_parent_that_is_a_file_rejected(tmp_path: Path) -> None:
    parent = tmp_path / "afile"
    parent.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="parent directory"):
        write_fixture_file("x", parent / "f.json")


@pytest.mark.parametrize("bad", [b"x", None, 123, ["x"]])
def test_non_string_text_rejected(bad: Any, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be a string"):
        write_fixture_file(bad, tmp_path / "f.json")


@pytest.mark.parametrize("bad", [123, None, b"x"])
def test_invalid_path_type_rejected(bad: Any, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Invalid fixture output path"):
        write_fixture_file("x", bad)


def test_created_file_is_owner_only_and_verbatim(tmp_path: Path) -> None:
    text = split_to_json_text(_adversarial_split())
    target = write_fixture_file(text, tmp_path / "f.json")
    assert isinstance(target, Path)
    assert target == tmp_path / "f.json"
    assert target.read_text(encoding="utf-8") == text
    mode = os.stat(target).st_mode & 0o777
    assert mode & 0o077 == 0  # no group/other bits, whatever the umask
    assert mode == 0o600


def test_str_path_spelling_writes_and_returns_path(tmp_path: Path) -> None:
    target = write_fixture_file("{}", str(tmp_path / "s.json"))
    assert target == tmp_path / "s.json"
    assert target.read_text(encoding="utf-8") == "{}"


def test_relative_path_resolves_in_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    target = write_fixture_file("{}", "rel.json")
    assert target == Path("rel.json")
    assert (tmp_path / "rel.json").read_text(encoding="utf-8") == "{}"


def test_empty_text_writes_verbatim(tmp_path: Path) -> None:
    """The writer is content-agnostic: no schema validation at write time."""
    target = write_fixture_file("", tmp_path / "empty.json")
    assert target.read_text(encoding="utf-8") == ""


class _FailingWriter(io.StringIO):
    def write(self, s: str) -> int:
        raise OSError("simulated write failure")


def test_partial_write_leaves_no_file(tmp_path: Path) -> None:
    target = tmp_path / "partial.json"

    def fake_fdopen(fd: int, *args: Any, **kwargs: Any) -> io.StringIO:
        os.close(fd)  # the real fd must not leak when fdopen "fails"
        return _FailingWriter()

    with (
        mock.patch.object(os, "fdopen", fake_fdopen),
        pytest.raises(OSError, match="simulated write failure"),
    ):
        write_fixture_file('{"search": []}', target)
    # The O_CREAT|O_EXCL open created the path; the failed write unlinked it.
    assert not target.exists()


def test_os_open_oserror_becomes_valueerror(tmp_path: Path) -> None:
    with (
        mock.patch.object(os, "open", side_effect=OSError("EACCES")),
        pytest.raises(ValueError, match="not writable"),
    ):
        write_fixture_file("x", tmp_path / "f.json")
    # FileExistsError keeps its dedicated message (it subclasses OSError).
    with (
        mock.patch.object(os, "open", side_effect=FileExistsError("exists")),
        pytest.raises(ValueError, match="already exists"),
    ):
        write_fixture_file("x", tmp_path / "f.json")


def test_write_then_load_dataset_roundtrip(tmp_path: Path) -> None:
    split = _adversarial_split()
    target = write_fixture_file(split_to_json_text(split), tmp_path / "fx.json")
    _assert_splits_equal(split, load_dataset(target))
    _assert_splits_equal(split, load_dataset(str(target)))


# ---------------------------------------------------------------------------
# 4. parse_dataset / load_dataset rejection vocabulary.


def test_reject_duplicate_json_field() -> None:
    with pytest.raises(ValueError, match="Duplicate JSON field"):
        parse_dataset('{"search": [], "held_out": [], "search": []}')


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_reject_nonfinite_constants(constant: str) -> None:
    text = _valid_doc().replace('"expected": "e"', f'"expected": {constant}', 1)
    with pytest.raises(ValueError, match="Nonfinite JSON"):
        parse_dataset(text)


def test_reject_missing_held_out() -> None:
    with pytest.raises(ValueError, match="Unexpected or missing JSON fields"):
        parse_dataset('{"search": []}')


def test_reject_extra_top_level_key() -> None:
    text = _valid_doc().replace('{"search":', '{"extra": 1, "search":', 1)
    with pytest.raises(ValueError, match="Unexpected or missing JSON fields"):
        parse_dataset(text)


@pytest.mark.parametrize("text", ["[]", "null", '"str"', "42"])
def test_reject_non_object_top_level(text: str) -> None:
    with pytest.raises(ValueError, match="Unexpected or missing JSON fields"):
        parse_dataset(text)


def test_reject_empty_search_partition() -> None:
    with pytest.raises(ValueError, match="Invalid fixture count"):
        parse_dataset(_doc_json([], [_fx_json("b1", group="h1", contents=("z",))]))


def test_reject_empty_held_out_partition() -> None:
    with pytest.raises(ValueError, match="Invalid fixture count"):
        parse_dataset(_doc_json([_fx_json("a1", contents=("q",))], []))


def test_reject_non_list_partition() -> None:
    with pytest.raises(ValueError, match="Invalid fixture count"):
        parse_dataset('{"search": {}, "held_out": []}')


def test_reject_too_many_fixtures() -> None:
    many = [_fx_json(f"s{i}", group="g", contents=(f"q{i}",)) for i in range(101)]
    with pytest.raises(ValueError, match="Invalid fixture count"):
        parse_dataset(_doc_json(many, [_fx_json("b1", group="h1", contents=("z",))]))


def test_reject_fixture_missing_field() -> None:
    bad = {"id": "a", "group": "g", "messages": [{"role": "user", "content": "q"}]}
    with pytest.raises(ValueError, match="Unexpected or missing JSON fields"):
        parse_dataset(_doc_json([bad], [_fx_json("b1", group="h1", contents=("z",))]))


def test_reject_fixture_extra_field() -> None:
    bad = dict(_fx_json("a1"), extra=1)
    with pytest.raises(ValueError, match="Unexpected or missing JSON fields"):
        parse_dataset(_doc_json([bad], [_fx_json("b1", group="h1", contents=("z",))]))


def test_reject_message_missing_field() -> None:
    bad = _fx_json("a1")
    bad["messages"] = [{"role": "user"}]
    with pytest.raises(ValueError, match="Unexpected or missing JSON fields"):
        parse_dataset(_doc_json([bad], [_fx_json("b1", group="h1", contents=("z",))]))


def test_reject_message_extra_field() -> None:
    bad = _fx_json("a1")
    bad["messages"] = [{"role": "user", "content": "q", "extra": 1}]
    with pytest.raises(ValueError, match="Unexpected or missing JSON fields"):
        parse_dataset(_doc_json([bad], [_fx_json("b1", group="h1", contents=("z",))]))


def test_reject_empty_messages() -> None:
    bad = _fx_json("a1", contents=())
    with pytest.raises(ValueError, match="1-21 messages"):
        parse_dataset(_doc_json([bad], [_fx_json("b1", group="h1", contents=("z",))]))


def test_reject_too_many_messages() -> None:
    bad = _fx_json("a1", contents=tuple(f"m{i}" for i in range(22)))
    with pytest.raises(ValueError, match="1-21 messages"):
        parse_dataset(_doc_json([bad], [_fx_json("b1", group="h1", contents=("z",))]))


def test_reject_bad_alternation_start() -> None:
    bad = _fx_json("a1")
    bad["messages"] = [{"role": "assistant", "content": "q"}]
    with pytest.raises(ValueError, match="must alternate"):
        parse_dataset(_doc_json([bad], [_fx_json("b1", group="h1", contents=("z",))]))


def test_reject_bad_alternation_middle() -> None:
    bad = _fx_json("a1")
    bad["messages"] = [
        {"role": "user", "content": "q"},
        {"role": "user", "content": "q"},
        {"role": "user", "content": "q"},
    ]
    with pytest.raises(ValueError, match="must alternate"):
        parse_dataset(_doc_json([bad], [_fx_json("b1", group="h1", contents=("z",))]))


def test_reject_assistant_final() -> None:
    bad = _fx_json("a1")
    bad["messages"] = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]
    with pytest.raises(ValueError, match="must end with user"):
        parse_dataset(_doc_json([bad], [_fx_json("b1", group="h1", contents=("z",))]))


def test_reject_non_string_content() -> None:
    bad = _fx_json("a1")
    bad["messages"] = [{"role": "user", "content": 123}]
    with pytest.raises(ValueError, match="Invalid or oversized text field"):
        parse_dataset(_doc_json([bad], [_fx_json("b1", group="h1", contents=("z",))]))


def test_reject_oversized_content() -> None:
    bad = _fx_json("a1", contents=("x" * (MAX_TEXT + 1),))
    with pytest.raises(ValueError, match="Invalid or oversized text field"):
        parse_dataset(_doc_json([bad], [_fx_json("b1", group="h1", contents=("z",))]))


def test_reject_blank_id() -> None:
    bad = _fx_json("   ")
    with pytest.raises(ValueError, match="Blank identifier"):
        parse_dataset(_doc_json([bad], [_fx_json("b1", group="h1", contents=("z",))]))


def test_reject_blank_group() -> None:
    bad = _fx_json("a1", group="\t")
    with pytest.raises(ValueError, match="Blank identifier"):
        parse_dataset(_doc_json([bad], [_fx_json("b1", group="h1", contents=("z",))]))


def test_reject_non_string_expected() -> None:
    bad = _fx_json("a1", expected="e")
    bad["expected"] = 5
    with pytest.raises(ValueError, match="Invalid or oversized text field"):
        parse_dataset(_doc_json([bad], [_fx_json("b1", group="h1", contents=("z",))]))


def test_reject_duplicate_ids_across_partitions() -> None:
    search = [_fx_json("dup", group="g1", contents=("q1",))]
    held_out = [_fx_json("dup", group="h1", contents=("z1",))]
    with pytest.raises(ValueError, match="globally unique"):
        parse_dataset(_doc_json(search, held_out))


def test_reject_duplicate_ids_within_partition() -> None:
    search = [
        _fx_json("dup", group="g1", contents=("q1",)),
        _fx_json("dup", group="g1", contents=("q2",)),
    ]
    held_out = [_fx_json("b1", group="h1", contents=("z1",))]
    with pytest.raises(ValueError, match="globally unique"):
        parse_dataset(_doc_json(search, held_out))


def test_reject_group_overlap() -> None:
    search = [_fx_json("a1", group="same", contents=("q1",))]
    held_out = [_fx_json("b1", group="same", contents=("z1",))]
    with pytest.raises(ValueError, match="cannot cross the split"):
        parse_dataset(_doc_json(search, held_out))


def test_reject_identical_messages_across_split() -> None:
    search = [_fx_json("a1", group="g1", contents=("same-q",))]
    held_out = [_fx_json("b1", group="h1", contents=("same-q",))]
    with pytest.raises(ValueError, match="Identical message inputs cannot cross"):
        parse_dataset(_doc_json(search, held_out))


def test_reject_deep_nesting() -> None:
    # 40000-deep nesting is ~80 KB, far under the 2 MB cap; the C scanner
    # exhausts the Python stack around depth ~9000 on 3.11-3.13.
    with pytest.raises(ValueError, match="nesting too deep"):
        parse_dataset("[" * 40000 + "]" * 40000)


def test_reject_oversize_text() -> None:
    with pytest.raises(ValueError, match="exceeds size limit"):
        parse_dataset("x" * (MAX_BYTES + 1))


@pytest.mark.parametrize("bad", [None, 123, b"{}"])
def test_reject_non_string_input(bad: Any) -> None:
    with pytest.raises(ValueError):
        parse_dataset(bad)


def test_load_dataset_roundtrip(tmp_path: Path) -> None:
    split = _adversarial_split()
    target = write_fixture_file(split_to_json_text(split), tmp_path / "fx.json")
    _assert_splits_equal(split, load_dataset(target))
    _assert_splits_equal(split, load_dataset(str(target)))


def test_load_dataset_rejects_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a regular file"):
        load_dataset(tmp_path)


def test_load_dataset_rejects_fifo(tmp_path: Path) -> None:
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="not a regular file"):
        load_dataset(fifo)


def test_load_dataset_rejects_missing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unreadable fixture file"):
        load_dataset(tmp_path / "missing.json")


def test_load_dataset_rejects_non_utf8(tmp_path: Path) -> None:
    target = tmp_path / "bad.json"
    target.write_bytes(b'{"search": "\xff\xfe"}')
    with pytest.raises(ValueError, match="not valid UTF-8"):
        load_dataset(target)


def test_load_dataset_rejects_oversize_file(tmp_path: Path) -> None:
    target = tmp_path / "big.json"
    target.write_bytes(b"x" * (MAX_BYTES + 1))
    with pytest.raises(ValueError, match="exceeds size limit"):
        load_dataset(target)


def test_load_dataset_rejects_bad_path_type() -> None:
    with pytest.raises(ValueError, match="Invalid fixture path"):
        load_dataset(cast("str | Path", 123))


def test_load_dataset_rejects_invalid_content(tmp_path: Path) -> None:
    target = tmp_path / "bad.json"
    target.write_text('{"search": [], "held_out": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid fixture count"):
        load_dataset(target)
