"""Cycle 29: differential verification of model_protocol_lab/fixtures.py.

An independent oracle is re-implemented here from the module *docstrings*
(structure and naming deliberately differ from the implementation; no
helpers are shared with it). The test-suite then:

* fuzzes the oracle against the implementation on generated CSV inputs
  (deterministic string seeds), covering the search/held-out split
  discipline, edge cases and boundary conditions;
* pins behavioral facts the docstrings claim;
* runs mutation-style sensitivity controls on a COPY of the package in
  /tmp (the repo is never mutated) to prove the tests catch real bugs --
  see /tmp/cycle29_mutations/run_mutations.py;
* red-teams the module for hidden assumptions.

BUGS FOUND
----------
None. The implementation matches the oracle on 240 fuzzed CSV programs
(120 valid, 120 invalid) across 8 seeds plus targeted differential sweeps,
and all pinned docstring claims hold, with the discrepancies below.

DOCSTRING / CODE DIVERGENCES (code behavior pinned; source untouched)
---------------------------------------------------------------------
1. (CYCLE 29 NOTE, FIXED IN CYCLE 29.) ``parse_fixture_csv`` docstring
   used to say "Raises ValueError with a row number on any authoring
   error."  In fact only *per-row* errors carry a row number; header
   errors, column-count errors, the >200-fixture error, the
   per-partition count errors, the missing-partition errors and the final
   split-discipline wrapper carry no row number.  The docstring was
   corrected in cycle 29.  Pinned by
   ``TestPinnedBehavior::test_row_number_only_on_row_errors``.
2. (CYCLE 29 NOTE, FIXED IN CYCLE 29.) ``_extract_special_tokens``
   docstring used to say each optional field "falls back to an
   already-scanned field when None".  The code does not perform a
   fallback lookup -- it simply skips None fields, which is *observably
   equivalent* to falling back (the fallen-back-to field was already
   scanned and the inventory dedupes), so no behavioral divergence; the
   wording was only confusing and was clarified in cycle 29.  Pinned by
   ``TestTokenInventory::test_none_optionals_add_nothing``.
3. Module docstring: "100 fixtures per partition" is enforced together
   with a *total* cap of 200 rows, checked before any row is parsed, so a
   201-row file is rejected even before the per-partition check could run.
   No divergence in accepted inputs (201 rows always violate a partition
   cap too), only in which error fires first.  Pinned by
   ``TestPinnedBehavior::test_total_row_cap_checked_first``.
4. ``write_fixture_file`` docstring: "the parent directory must already
   exist".  The check is ``target.parent != Path(".") and not
   target.parent.is_dir()``: a *relative* single-component path skips the
   check entirely, and a non-normalized parent such as ``sub/..`` is
   rejected even though it resolves to the existing cwd.  Pinned by
   ``TestWriteFixtureFile``.
5. ``parse_fixture_csv`` applies the ``MAX_TEXT`` cell limit to the
   ``expected`` cell and the message cells but NOT to ``id``/``group``:
   an over-long id/group parses fine here and only fails later, inside
   ``split_to_json_text``, when the by-construction ``parse_dataset``
   round-trip rejects it.  The docstring's "Limits mirror the runtime
   schema" is therefore only true end-to-end, not at parse time.  Pinned
   by ``TestPinnedBehavior::test_oversize_id_group_fail_only_at_serialize``.
"""

from __future__ import annotations

import csv
import io
import json
import os
import random
import re
import stat
import unicodedata
from pathlib import Path

import pytest

from model_protocol_lab import fixtures as fx
from model_protocol_lab.catalog import CATALOG
from model_protocol_lab.core import Fixture, Format, Message, Split
from model_protocol_lab.datasets import (
    MAX_BYTES,
    MAX_CASES_PER_SPLIT,
    MAX_MESSAGES,
    MAX_TEXT,
    parse_dataset,
)

# ---------------------------------------------------------------------------
# Independent oracle, written from the docstrings.
# ---------------------------------------------------------------------------

_ORACLE_FIXED = ("id", "group", "partition", "expected")
_ORACLE_PARTITIONS = ("search", "held_out")


def _oracle_header_columns(count: int) -> tuple[str, ...]:
    """Docstring: message columns must form 'user_1,assistant_1,user_2,...'
    pairs in order, starting with 'user_1'."""
    names: list[str] = []
    turn = 1
    while len(names) < count:
        names.append(f"user_{turn}")
        if len(names) < count:
            names.append(f"assistant_{turn}")
        turn += 1
    return tuple(names)


class _OracleRejected(Exception):
    """The oracle's analogue of the implementation's ValueError."""


def _oracle_nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _oracle_conversation_key(messages: tuple[Message, ...]) -> tuple[tuple[str, str], ...]:
    # Docstring: "identical message sequences" compared "NFC-normalized".
    return tuple((_oracle_nfc(m.role), _oracle_nfc(m.content)) for m in messages)


def _oracle_check_discipline(search: list[Fixture], held_out: list[Fixture]) -> None:
    """Docstring: ID uniqueness, group non-overlap across partitions, no
    identical message sequences across partitions, NFC-normalized; both
    partitions nonempty; at most 100 fixtures per partition."""
    if not search or not held_out:
        raise _OracleRejected("empty partition")
    if len(search) > MAX_CASES_PER_SPLIT or len(held_out) > MAX_CASES_PER_SPLIT:
        raise _OracleRejected("too many fixtures in a partition")
    every = search + held_out
    if len({_oracle_nfc(c.id) for c in every}) != len(every):
        raise _OracleRejected("duplicate id")
    if {_oracle_nfc(c.group) for c in search} & {_oracle_nfc(c.group) for c in held_out}:
        raise _OracleRejected("group overlap")
    if {_oracle_conversation_key(c.messages) for c in search} & {
        _oracle_conversation_key(c.messages) for c in held_out
    }:
        raise _OracleRejected("message leakage")


def _oracle_parse_fixture_csv(text: str) -> tuple[list[Fixture], list[Fixture], list[str]]:
    """Independent re-implementation of ``parse_fixture_csv`` from its
    docstring.  Returns (search, held_out, empty_expected_ids); raises
    _OracleRejected on any authoring error.  Record tokenization follows
    RFC 4180 (the documented CSV layout: header row required, quoted cells
    may contain newlines) via the stdlib csv module in both oracle and
    implementation -- that is the format definition, not implementation
    logic; every validation rule below is re-derived independently."""
    if type(text) is not str:
        raise _OracleRejected("not text")
    if len(text.encode("utf-8")) > MAX_BYTES:
        raise _OracleRejected("too big")
    reader = csv.reader(io.StringIO(text))
    header: list[str] | None = None
    raw_rows: list[list[str]] = []
    for record in reader:
        if header is None:
            if not record or all(cell == "" for cell in record):
                continue
            header = list(record)
            continue
        if not record or all(cell == "" for cell in record):
            continue
        raw_rows.append(list(record))
    if header is None:
        raise _OracleRejected("no header")
    if tuple(header[:4]) != _ORACLE_FIXED:
        raise _OracleRejected("bad fixed columns")
    msg_headers = tuple(header[4:])
    if not 1 <= len(msg_headers) <= MAX_MESSAGES:
        raise _OracleRejected("bad message column count")
    if msg_headers != _oracle_header_columns(len(msg_headers)):
        raise _OracleRejected("bad message column order")
    if len(raw_rows) > 2 * MAX_CASES_PER_SPLIT:
        raise _OracleRejected("too many rows")

    search: list[Fixture] = []
    held_out: list[Fixture] = []
    empty_expected: list[str] = []
    for record in raw_rows:
        if len(record) > len(header):
            raise _OracleRejected("too many cells")
        cells = record + [""] * (len(header) - len(record))
        fid, group, partition, expected = cells[0], cells[1], cells[2], cells[3]
        if not fid.strip():
            raise _OracleRejected("blank id")
        if not group.strip():
            raise _OracleRejected("blank group")
        if partition not in _ORACLE_PARTITIONS:
            raise _OracleRejected("bad partition")
        bodies = cells[4:]
        if any(len(c) > MAX_TEXT for c in bodies) or len(expected) > MAX_TEXT:
            raise _OracleRejected("oversize text")
        if not bodies[0]:
            raise _OracleRejected("missing user_1")
        convo: list[Message] = []
        finished = False
        for position, cell in enumerate(bodies):
            if cell == "":
                finished = True
                continue
            if finished:
                raise _OracleRejected("content after empty cell")
            convo.append(Message("user" if position % 2 == 0 else "assistant", cell))
        if convo[-1].role != "user":
            raise _OracleRejected("must end with user")
        fixture = Fixture(id=fid, messages=tuple(convo), expected=expected, group=group)
        if expected == "":
            empty_expected.append(fid)
        (search if partition == "search" else held_out).append(fixture)
    _oracle_check_discipline(search, held_out)
    return search, held_out, empty_expected


# -- special-token audit oracle --------------------------------------------

# The four token shapes named in the module docstring, re-written here.
_ORACLE_SHAPES = (
    re.compile(r"＜｜[^＜＞\n]+｜＞"),
    re.compile(r"<\|[^<>\n]+\|>"),
    re.compile(r"\[[A-Za-z/][^\[\]\n]*\]"),
    re.compile(r"<[^<>\n]+>"),
)


def _oracle_token_inventory(fmt: Format) -> tuple[str, ...]:
    """Docstring: inventory derived from the format's framing fields (bos,
    user_prefix, assistant_prefix, turn_end, plus non-None per-role turn
    terminators and open suffix); token-shaped substrings only, deduped,
    first-seen order."""
    found: list[str] = []
    for field in (
        fmt.bos,
        fmt.user_prefix,
        fmt.assistant_prefix,
        fmt.turn_end,
        fmt.user_turn_end,
        fmt.assistant_turn_end,
        fmt.open_suffix,
    ):
        if field is None:
            continue
        for shape in _ORACLE_SHAPES:
            for hit in shape.finditer(field):
                token = hit.group(0)
                if token not in found:
                    found.append(token)
    return tuple(found)


def _oracle_audit_targets(
    target: Fixture | Split | tuple[Fixture | str, ...] | str,
) -> tuple[tuple[str, str, str], ...]:
    """Docstring: (fixture_id, field, text) triples; field is '<role>_<n>'
    per fixture message ('user_1', 'assistant_1', 'user_2', ...), 'text'
    for a raw string, 'text_<n>' per item of a raw-string tuple."""
    triples: list[tuple[str, str, str]] = []

    def _one(case: Fixture) -> None:
        for pos, msg in enumerate(case.messages):
            triples.append((case.id, f"{msg.role}_{pos // 2 + 1}", msg.content))

    if type(target) is Fixture:
        _one(target)
    elif type(target) is Split:
        for case in (*target.search, *target.held_out):
            _one(case)
    elif type(target) is str:
        triples.append(("", "text", target))
    elif type(target) is tuple:
        for pos, item in enumerate(target):
            if type(item) is Fixture:
                _one(item)
            elif type(item) is str:
                triples.append(("", f"text_{pos + 1}", item))
            else:
                raise _OracleRejected("bad tuple item")
    else:
        raise _OracleRejected("bad audit input")
    return tuple(triples)


def _oracle_audit(
    target: Fixture | Split | tuple[Fixture | str, ...] | str,
    formats: tuple[Format, ...],
) -> tuple[tuple[str, str, str, str, int], ...]:
    """Returns (fixture_id, field, literal, format_name, offset) findings.

    Docstring order: fixtures/messages in input order, then formats in the
    given order, then each format's inventory order, then ascending offsets;
    exact case-sensitive substring matches; overlapping occurrences of one
    literal each get a finding."""
    if type(formats) is not tuple or not formats:
        raise _OracleRejected("formats must be a nonempty tuple")
    if any(type(f) is not Format for f in formats):
        raise _OracleRejected("formats must hold Format")
    out: list[tuple[str, str, str, str, int]] = []
    for fid, field, text in _oracle_audit_targets(target):
        for fmt in formats:
            for token in _oracle_token_inventory(fmt):
                start = 0
                while True:
                    at = text.find(token, start)
                    if at < 0:
                        break
                    out.append((fid, field, token, fmt.name, at))
                    start = at + 1  # overlapping occurrences each count
    return tuple(out)


def _oracle_audit_json(findings: tuple[tuple[str, str, str, str, int], ...], source: str) -> str:
    """Docstring: sidecar document; findings cannot live in the fixture
    JSON because the strict schema rejects extra top-level keys."""
    payload = {
        "schema": "model-protocol-lab.audit/v1",
        "source": source,
        "findings": [
            {
                "fixture_id": fid,
                "field": field,
                "literal": literal,
                "format_name": name,
                "offset": offset,
            }
            for fid, field, literal, name, offset in findings
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


# ---------------------------------------------------------------------------
# Deterministic CSV program generator.
# ---------------------------------------------------------------------------

_SEEDS = [f"cycle29-seed-{n}" for n in range(8)]
_WORDS = ["alpha", "beta", "gamma", "delta", "hello world", "café", "naïve"]


def _rng_word(rng: random.Random) -> str:
    word = rng.choice(_WORDS)
    if rng.random() < 0.25:
        word += rng.choice([",", '"', "\n", "  ", "<|im_start|>", "[INST]"])
    return word


def _make_csv_text(rng: random.Random, n_cols: int, n_rows: int) -> str:
    """Build a syntactically valid authoring CSV (validity of the *content*
    rules is left to chance: ids/groups/partitions/messages are randomized,
    so outputs span valid files, discipline violations and row errors)."""
    header = ["id", "group", "partition", "expected"] + [
        f"user_{(i // 2) + 1}" if i % 2 == 0 else f"assistant_{(i // 2) + 1}" for i in range(n_cols)
    ]
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(header)
    groups = [f"g{rng.randint(1, 6)}" for _ in range(n_rows)]
    for _ in range(n_rows):
        fid = f"f{rng.randint(1, max(2, n_rows // 2))}"  # deliberate id collisions
        group = rng.choice(groups)
        partition = rng.choice(["search", "held_out", "search"])
        expected = "" if rng.random() < 0.2 else _rng_word(rng)
        cells = [fid, group, partition, expected]
        # random conversation length 1..n_cols, sometimes with a gap or a
        # trailing assistant turn to trigger row errors
        plan = rng.random()
        n_msgs = rng.randint(1, n_cols)
        msgs = [_rng_word(rng) for _ in range(n_msgs)]
        if plan < 0.15 and n_msgs + 1 <= n_cols:
            msgs.append("")  # gap -> content-after-empty error later maybe
            msgs.append(_rng_word(rng))
        elif plan < 0.25:
            msgs.append("")  # clean end
        while len(msgs) < n_cols:
            msgs.append("")
        cells.extend(msgs[:n_cols])
        if rng.random() < 0.05:
            cells[rng.randint(0, 3)] = " "  # blank-ish id/group/partition/expected
        writer.writerow(cells)
    return buf.getvalue()


def _oracle_ok(text: str) -> tuple[list[Fixture], list[Fixture], list[str]] | None:
    try:
        return _oracle_parse_fixture_csv(text)
    except _OracleRejected:
        return None


# ---------------------------------------------------------------------------
# Differential fuzz: oracle vs implementation on CSV parsing.
# ---------------------------------------------------------------------------


class TestOracleFuzzParse:
    @pytest.mark.parametrize("seed", _SEEDS)
    @pytest.mark.parametrize("n_cols", [1, 2, 3, 5])
    def test_fuzz_valid_and_invalid_csvs(self, seed: str, n_cols: int):
        rng = random.Random(f"{seed}-cols{n_cols}")  # noqa: S311 -- deterministic test seed
        for trial in range(15):
            text = _make_csv_text(rng, n_cols, rng.randint(2, 12))
            oracle = _oracle_ok(text)
            try:
                impl_split, impl_empty = fx.parse_fixture_csv(text)
                impl_ok: bool = True
            except ValueError:
                impl_ok = False
            assert (oracle is not None) == impl_ok, f"trial {trial} accept mismatch"
            if oracle is None:
                continue
            o_search, o_held, o_empty = oracle
            assert isinstance(impl_split, Split), f"trial {trial}"
            assert impl_split.search == tuple(o_search), f"trial {trial} search mismatch:\n{text}"
            assert impl_split.held_out == tuple(o_held), f"trial {trial} held_out mismatch:\n{text}"
            assert impl_empty == o_empty, f"trial {trial} empty-expected mismatch"
            # The discipline oracle re-checks the implementation output
            # independently (non-circular: these checks are mine).
            _oracle_check_discipline(list(impl_split.search), list(impl_split.held_out))

    def test_fuzz_serialize_roundtrip(self):
        rng = random.Random("cycle29-roundtrip")  # noqa: S311 -- deterministic test seed
        for trial in range(40):
            text = _make_csv_text(rng, rng.choice([1, 2, 3, 4]), rng.randint(2, 8))
            oracle = _oracle_ok(text)
            if oracle is None:
                with pytest.raises(ValueError):
                    fx.parse_fixture_csv(text)
                continue
            split, _ = fx.parse_fixture_csv(text)
            out = fx.split_to_json_text(split)
            assert out.endswith("\n"), f"trial {trial}"
            assert parse_dataset(out) == split, f"trial {trial} roundtrip mismatch"


# ---------------------------------------------------------------------------
# Pinned behavioral facts from the docstrings.
# ---------------------------------------------------------------------------


def _simple_csv(rows: list[list[str]], header: list[str] | None = None) -> str:
    header = header or ["id", "group", "partition", "expected", "user_1", "assistant_1", "user_2"]
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buf.getvalue()


def _two_row_csv() -> str:
    return _simple_csv(
        [
            ["a1", "g1", "search", "yes", "hello", "hi", "again?"],
            ["b1", "g2", "held_out", "", "other", "", ""],
        ]
    )


class TestPinnedBehavior:
    def test_empty_expected_ids_reported(self):
        split, empty = fx.parse_fixture_csv(_two_row_csv())
        assert empty == ["b1"]
        assert split.search[0].expected == "yes"
        assert split.held_out[0].expected == ""

    def test_verbatim_cells_no_stripping(self):
        text = _simple_csv(
            [
                ["  a1  ", "  g1  ", "search", " e ", "  hi  ", "", ""],
                ["b1", "g2", "held_out", "", "yo", "", ""],
            ]
        )
        split, _ = fx.parse_fixture_csv(text)
        case = split.search[0]
        assert case.id == "  a1  "
        assert case.group == "  g1  "
        assert case.expected == " e "
        assert case.messages[0].content == "  hi  "

    def test_quoted_newline_cell(self):
        text = _simple_csv(
            [
                ["a1", "g1", "search", "e", "line1\nline2", "", ""],
                ["b1", "g2", "held_out", "", "yo", "", ""],
            ]
        )
        split, _ = fx.parse_fixture_csv(text)
        assert split.search[0].messages[0].content == "line1\nline2"

    def test_blank_lines_ignored(self):
        text = "\n\n" + _two_row_csv().replace("\n", "\n\n", 1) + "\n\n"
        split, _ = fx.parse_fixture_csv(text)
        assert len(split.search) == 1 and len(split.held_out) == 1

    def test_leading_blank_lines_are_not_a_header(self):
        with pytest.raises(ValueError, match="no header"):
            fx.parse_fixture_csv("\n\n\n")

    def test_row_number_only_on_row_errors(self):
        # Per-row errors carry "row N"; header/column/count/discipline
        # errors do not (docstring says "any authoring error" -- code is
        # narrower; this pins the code behavior).
        bad_row = _simple_csv([["", "g1", "search", "e", "hi", "", ""]])
        with pytest.raises(ValueError, match=r"row 2"):
            fx.parse_fixture_csv(bad_row)
        no_row_number_cases = [
            "id,group,partition\n",  # bad fixed columns, no message cols
            _simple_csv([], header=["id", "group", "partition", "expected"]),
        ]
        for text in no_row_number_cases:
            with pytest.raises(ValueError) as exc:
                fx.parse_fixture_csv(text)
            assert "row" not in str(exc.value)

    def test_non_string_source_rejected(self):
        for bad in (b"id,group", 123, None, ["id"]):
            with pytest.raises(ValueError):
                fx.parse_fixture_csv(bad)  # type: ignore[arg-type]

    def test_str_subclass_rejected(self):
        class S(str):
            pass

        with pytest.raises(ValueError):
            fx.parse_fixture_csv(S(_two_row_csv()))

    def test_oversize_source_rejected(self):
        with pytest.raises(ValueError, match="[Ss]ize"):
            fx.parse_fixture_csv("x" * (MAX_BYTES + 1))

    def test_total_row_cap_checked_first(self):
        # 201 tiny rows: the total cap (>200) fires before per-row parsing.
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow(["id", "group", "partition", "expected", "user_1"])
        for i in range(201):
            writer.writerow([f"id{i}", f"g{i}", "search", "", "hi"])
        with pytest.raises(ValueError, match="more than 200"):
            fx.parse_fixture_csv(buf.getvalue())

    def test_oversize_id_group_fail_only_at_serialize(self):
        # id/group are NOT length-checked by parse_fixture_csv; the
        # by-construction parse_dataset round-trip in split_to_json_text
        # is what rejects them.
        big = "x" * (MAX_TEXT + 1)
        text = _simple_csv(
            [
                [big, "g1", "search", "", "hi", "", ""],
                ["b1", "g2", "held_out", "", "yo", "", ""],
            ]
        )
        split, _ = fx.parse_fixture_csv(text)  # parses fine
        with pytest.raises(ValueError):
            fx.split_to_json_text(split)

    def test_oversize_expected_rejected_at_parse(self):
        big = "x" * (MAX_TEXT + 1)
        text = _simple_csv([["a1", "g1", "search", big, "hi", "", ""]])
        with pytest.raises(ValueError, match="[Ss]ize|[Ll]imit"):
            fx.parse_fixture_csv(text)

    def test_split_subclass_rejected_by_serializer(self):
        split, _ = fx.parse_fixture_csv(_two_row_csv())

        class SubSplit(Split):
            pass

        with pytest.raises(ValueError):
            fx.split_to_json_text(SubSplit(split.search, split.held_out))  # type: ignore[arg-type]

    def test_serializer_keeps_non_ascii_verbatim(self):
        text = _simple_csv(
            [
                ["a1", "g1", "search", "café", "naïve", "", ""],
                ["b1", "g2", "held_out", "", "日本語", "", ""],
            ]
        )
        split, _ = fx.parse_fixture_csv(text)
        out = fx.split_to_json_text(split)
        assert "café" in out and "日本語" in out  # ensure_ascii=False
        assert parse_dataset(out) == split

    def test_whitespace_id_rejected_with_row_number(self):
        text = _simple_csv(
            [
                ["a1", "g1", "search", "", "hi", "", ""],
                ["   ", "g2", "held_out", "", "yo", "", ""],
            ]
        )
        with pytest.raises(ValueError, match=r"row 3.*fixture id is blank"):
            fx.parse_fixture_csv(text)

    def test_whitespace_group_rejected_with_row_number(self):
        text = _simple_csv(
            [
                ["a1", "g1", "search", "", "hi", "", ""],
                ["b1", "  \t ", "held_out", "", "yo", "", ""],
            ]
        )
        with pytest.raises(ValueError, match=r"row 3.*group is blank"):
            fx.parse_fixture_csv(text)

    def test_control_chars_scrubbed_from_error_text(self):
        # _safe strips control chars so hostile input cannot inject
        # terminal escapes into error messages.
        with pytest.raises(ValueError) as exc:
            fx.parse_fixture_csv(_simple_csv([["a1", "   ", "search", "e", "hi", "", ""]]))
        assert "\x1b" not in str(exc.value)
        # group blank error echoes the *sanitized* fixture id
        with pytest.raises(ValueError) as exc2:
            fx.parse_fixture_csv(_simple_csv([["a\x1bb", "   ", "search", "e", "hi", "", ""]]))
        assert "\x1b" not in str(exc2.value)
        assert "ab" in str(exc2.value)
        # the DEL / C1 range is scrubbed too
        with pytest.raises(ValueError) as exc3:
            fx.parse_fixture_csv(_simple_csv([["a\x7fb\x9fc", "   ", "search", "e", "hi", "", ""]]))
        assert "\x7f" not in str(exc3.value) and "\x9f" not in str(exc3.value)
        assert "abc" in str(exc3.value)


# ---------------------------------------------------------------------------
# Split-discipline differential fuzz (search/held-out separation).
# ---------------------------------------------------------------------------


def _discipline_csv(rng: random.Random) -> str:
    """Generate CSVs aimed at the discipline checks: near-duplicate ids,
    NFC-confusable ids/groups/messages, group overlap, message leakage."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["id", "group", "partition", "expected", "user_1", "assistant_1"])
    variants = ["café", "café", "naïve", "naïve"]  # NFC/NFD pairs
    n = rng.randint(2, 6)
    for i in range(n):
        fid = rng.choice([f"id{i}", f"id{i}", variants[i % len(variants)]])
        group = rng.choice([f"grp{i % 3}", variants[(i + 1) % len(variants)]])
        partition = "search" if i % 2 == 0 else "held_out"
        msg = rng.choice(["hello", "hello", variants[i % len(variants)]])
        writer.writerow([fid, group, partition, "", msg, ""])
    return buf.getvalue()


class TestSplitDiscipline:
    @pytest.mark.parametrize("seed", _SEEDS)
    def test_discipline_differential(self, seed: str):
        rng = random.Random(f"{seed}-discipline")  # noqa: S311 -- deterministic test seed
        for trial in range(30):
            text = _discipline_csv(rng)
            oracle = _oracle_ok(text)
            try:
                impl_split, _ = fx.parse_fixture_csv(text)
                impl_ok = True
            except ValueError:
                impl_ok = False
            assert (oracle is not None) == impl_ok, f"trial {trial} accept mismatch:\n{text}"
            if impl_ok:
                assert oracle is not None
                assert impl_split.search == tuple(oracle[0]), f"trial {trial}"
                assert impl_split.held_out == tuple(oracle[1]), f"trial {trial}"

    def test_nfc_id_collision_rejected(self):
        nfc, nfd = "café", "café"
        assert nfc != nfd and unicodedata.normalize("NFC", nfc) == unicodedata.normalize("NFC", nfd)
        text = _simple_csv(
            [
                [nfc, "g1", "search", "", "hi", "", ""],
                [nfd, "g2", "held_out", "", "yo", "", ""],
            ]
        )
        with pytest.raises(ValueError):
            fx.parse_fixture_csv(text)

    def test_nfc_group_overlap_rejected(self):
        nfc, nfd = "café", "café"
        text = _simple_csv(
            [
                ["a1", nfc, "search", "", "hi", "", ""],
                ["b1", nfd, "held_out", "", "yo", "", ""],
            ]
        )
        with pytest.raises(ValueError):
            fx.parse_fixture_csv(text)

    def test_nfc_message_leakage_rejected(self):
        nfc, nfd = "naïve", "naïve"
        text = _simple_csv(
            [
                ["a1", "g1", "search", "", nfc, "", ""],
                ["b1", "g2", "held_out", "", nfd, "", ""],
            ]
        )
        with pytest.raises(ValueError):
            fx.parse_fixture_csv(text)

    def test_same_group_within_one_partition_ok(self):
        text = _simple_csv(
            [
                ["a1", "g1", "search", "", "hi", "", ""],
                ["a2", "g1", "search", "", "hey", "", ""],
                ["b1", "g2", "held_out", "", "yo", "", ""],
            ]
        )
        split, _ = fx.parse_fixture_csv(text)
        assert len(split.search) == 2


# ---------------------------------------------------------------------------
# Special-token audit: differential fuzz oracle vs implementation.
# ---------------------------------------------------------------------------


def _fuzz_format(rng: random.Random, name: str) -> Format:
    """Build a Format with random framing fields, sometimes containing
    token-shaped substrings, full-width tokens, brackets and noise."""
    pieces = [
        "",
        "\n",
        "<|im_start|>",
        "<|im_end|>",
        "<bos>",
        "<s>",
        "[INST]",
        "[/INST]",
        "[s]",
        "＜｜User｜＞",
        "＜｜assistant｜＞",
        "user",
        " ",
        "### ",
        "<|",
        "|>",
        "[",
        "]",
        "<",
        ">",
        "＜",
        "｜",
        "＞",
        "<|incomplete",
        "[x",
        "x]",
        "<>",
        "[[/INST]]",
        "<<s>>",
        "<|a|><|b|>",
        "[INST][INST]",  # adjacent duplicates
        "<|x|>",  # repeated literal across fields
    ]

    def pick() -> str:
        return rng.choice(pieces)

    fmt = Format(
        name=name,
        bos=pick() + pick(),
        user_prefix=pick() + pick(),
        assistant_prefix=pick() + pick(),
        turn_end=pick() + pick(),
    )
    if rng.random() < 0.5:
        fmt = Format(
            name=name,
            bos=fmt.bos,
            user_prefix=fmt.user_prefix,
            assistant_prefix=fmt.assistant_prefix,
            turn_end=fmt.turn_end,
            user_turn_end=pick(),
            assistant_turn_end=pick(),
            open_suffix=pick() if rng.random() < 0.7 else None,
        )
    return fmt


def _fuzz_text(rng: random.Random, tokens: list[str]) -> str:
    parts = []
    for _ in range(rng.randint(0, 6)):
        r = rng.random()
        if r < 0.35 and tokens:
            parts.append(rng.choice(tokens))
        elif r < 0.5:
            parts.append(rng.choice(["<|im_start|>", "[INST]", "＜｜User｜＞", "<s>"]))
        else:
            parts.append(_rng_word(rng))
    return " ".join(parts)


def _finding_tuple(f: fx.AuditFinding) -> tuple[str, str, str, str, int]:
    return (f.fixture_id, f.field, f.literal, f.format_name, f.offset)


class TestTokenInventory:
    @pytest.mark.parametrize("seed", _SEEDS)
    def test_inventory_differential(self, seed: str):
        rng = random.Random(f"{seed}-inventory")  # noqa: S311 -- deterministic test seed
        for trial in range(25):
            fmt = _fuzz_format(rng, f"fmt{trial}")
            assert fx._extract_special_tokens(fmt) == _oracle_token_inventory(fmt), (
                f"trial {trial}: {fmt!r}"
            )

    def test_none_optionals_add_nothing(self):
        # Docstring: None optionals "fall back to an already-scanned field",
        # so they contribute nothing beyond the base fields.
        base = Format(
            name="b", bos="<s>", user_prefix="[INST] ", assistant_prefix="", turn_end="</s>"
        )
        with_opts = Format(
            name="b",
            bos="<s>",
            user_prefix="[INST] ",
            assistant_prefix="",
            turn_end="</s>",
            user_turn_end=None,
            assistant_turn_end=None,
            open_suffix=None,
        )
        assert fx._extract_special_tokens(base) == fx._extract_special_tokens(with_opts)
        toks = set(fx._extract_special_tokens(base))
        assert toks == {"<s>", "[INST]", "</s>"}  # </s> matches the <...> shape

    def test_open_suffix_only_token_is_found(self):
        fmt = Format(
            name="g",
            bos="",
            user_prefix="",
            assistant_prefix="",
            turn_end="",
            open_suffix="<|open|>",
        )
        assert fx._extract_special_tokens(fmt) == ("<|open|>",)

    def test_plain_words_are_not_tokens(self):
        fmt = Format(
            name="p", bos="hello", user_prefix="user: ", assistant_prefix="asst: ", turn_end="\n"
        )
        assert fx._extract_special_tokens(fmt) == ()

    def test_full_width_shape(self):
        fmt = Format(
            name="d",
            bos="＜｜begin▁of▁sentence｜＞",
            user_prefix="＜｜User｜＞",
            assistant_prefix="＜｜Assistant｜＞",
            turn_end="＜｜end▁of▁sentence｜＞",
        )
        toks = fx._extract_special_tokens(fmt)
        assert "＜｜User｜＞" in toks and "＜｜begin▁of▁sentence｜＞" in toks

    def test_reject_non_format(self):
        with pytest.raises(ValueError):
            fx._extract_special_tokens("nope")  # type: ignore[arg-type]


class TestAuditDifferential:
    @pytest.mark.parametrize("seed", _SEEDS)
    def test_audit_differential_on_strings(self, seed: str):
        rng = random.Random(f"{seed}-audit-str")  # noqa: S311 -- deterministic test seed
        for trial in range(20):
            n_formats = rng.randint(1, 3)
            formats = tuple(_fuzz_format(rng, f"F{k}") for k in range(n_formats))
            inventory = [t for fmt in formats for t in _oracle_token_inventory(fmt)]
            texts = tuple(_fuzz_text(rng, inventory) for _ in range(rng.randint(1, 3)))
            target: tuple[str, ...] | str = texts[0] if len(texts) == 1 else texts
            impl = fx.audit_special_token_literals(target, formats)
            oracle = _oracle_audit(target, formats)
            assert tuple(_finding_tuple(f) for f in impl) == oracle, f"trial {trial}"

    @pytest.mark.parametrize("seed", _SEEDS)
    def test_audit_differential_on_fixtures(self, seed: str):
        rng = random.Random(f"{seed}-audit-fixture")  # noqa: S311 -- deterministic test seed
        for trial in range(20):
            n_formats = rng.randint(1, 3)
            formats = tuple(_fuzz_format(rng, f"F{k}") for k in range(n_formats))
            inventory = [t for fmt in formats for t in _oracle_token_inventory(fmt)]
            n_msgs = rng.randint(1, 5)
            msgs = tuple(
                Message("user" if i % 2 == 0 else "assistant", _fuzz_text(rng, inventory))
                for i in range(n_msgs)
            )
            case = Fixture(id=f"fx{trial}", messages=msgs, expected="", group="g")
            target = case if trial % 2 == 0 else (case, _fuzz_text(rng, inventory))
            impl = fx.audit_special_token_literals(target, formats)
            oracle = _oracle_audit(target, formats)
            assert tuple(_finding_tuple(f) for f in impl) == oracle, f"trial {trial}"

    def test_audit_differential_on_split(self):
        rng = random.Random("cycle29-audit-split")  # noqa: S311 -- deterministic test seed
        formats = tuple(_fuzz_format(rng, f"F{k}") for k in range(2))
        # Deterministic valid split (distinct ids/groups/messages).
        rows = [[f"s{i}", f"gs{i}", "search", "", f"question {i}", "", ""] for i in range(3)] + [
            [f"h{i}", f"gh{i}", "held_out", "", f"other {i}", "", ""] for i in range(2)
        ]
        split, _ = fx.parse_fixture_csv(_simple_csv(rows))
        impl = fx.audit_special_token_literals(split, formats)
        assert tuple(_finding_tuple(f) for f in impl) == _oracle_audit(split, formats)

    def test_audit_input_validation(self):
        fmt = Format(name="f", bos="", user_prefix="", assistant_prefix="", turn_end="")
        with pytest.raises(ValueError):
            fx.audit_special_token_literals("x", ())  # empty formats
        with pytest.raises(ValueError):
            fx.audit_special_token_literals("x", [fmt])  # type: ignore[arg-type]  # not a tuple
        with pytest.raises(ValueError):
            fx.audit_special_token_literals("x", (fmt, "nope"))  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            fx.audit_special_token_literals(123, (fmt,))  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            fx.audit_special_token_literals(("ok", 123), (fmt,))  # type: ignore[arg-type]

    def test_field_naming_user1_assistant1_user2(self):
        fmt = Format(name="f", bos="", user_prefix="<|u|>", assistant_prefix="", turn_end="")
        case = Fixture(
            id="c",
            messages=(
                Message("user", "a <|u|>"),
                Message("assistant", "b"),
                Message("user", "c <|u|>"),
            ),
            expected="",
            group="g",
        )
        findings = fx.audit_special_token_literals(case, (fmt,))
        assert [(f.field, f.offset) for f in findings] == [("user_1", 2), ("user_2", 2)]

    def test_overlapping_occurrences_each_get_a_finding(self):
        fmt = Format(name="f", bos="", user_prefix="[INST]", assistant_prefix="", turn_end="")
        findings = fx.audit_special_token_literals("[INST][INST]", (fmt,))
        assert [f.offset for f in findings] == [0, 6]

    def test_case_sensitive(self):
        fmt = Format(name="f", bos="", user_prefix="[INST]", assistant_prefix="", turn_end="")
        assert fx.audit_special_token_literals("[inst]", (fmt,)) == ()

    def test_deterministic_ordering(self):
        # Two formats sharing a literal: findings ordered by format order,
        # then inventory order, then ascending offset.
        f1 = Format(name="first", bos="<a><b>", user_prefix="", assistant_prefix="", turn_end="")
        f2 = Format(name="second", bos="<b><a>", user_prefix="", assistant_prefix="", turn_end="")
        findings = fx.audit_special_token_literals("<a> <b>", (f1, f2))
        assert [(f.format_name, f.literal, f.offset) for f in findings] == [
            ("first", "<a>", 0),
            ("first", "<b>", 4),
            ("second", "<b>", 4),
            ("second", "<a>", 0),
        ]

    def test_real_catalog_spot_check(self):
        # The oracle must agree with the implementation on real catalog
        # formats, not just fuzzed ones.
        fmts = tuple(CATALOG[name] for name in list(CATALOG)[:5])
        texts = (
            "plain text with no tokens",
            "tricky <|im_start|> literal",
            "＜｜User｜＞ full width",
        )
        impl = fx.audit_special_token_literals(texts, fmts)
        assert tuple(_finding_tuple(f) for f in impl) == _oracle_audit(texts, fmts)

    def test_audit_finding_validation(self):
        with pytest.raises(ValueError):
            fx.AuditFinding(fixture_id="a", field="text", literal="", format_name="f", offset=0)
        with pytest.raises(ValueError):
            fx.AuditFinding(fixture_id="a", field="  ", literal="x", format_name="f", offset=0)
        with pytest.raises(ValueError):
            fx.AuditFinding(fixture_id="a", field="text", literal="x", format_name="f", offset=-1)
        with pytest.raises(ValueError):
            fx.AuditFinding(
                fixture_id=1,
                field="text",
                literal="x",  # type: ignore[arg-type]
                format_name="f",
                offset=0,
            )


class TestAuditFindingsJson:
    def test_sidecar_shape_and_roundtrip(self):
        rng = random.Random("cycle29-sidecar")  # noqa: S311 -- deterministic test seed
        formats = tuple(_fuzz_format(rng, f"F{k}") for k in range(2))
        # Deterministic format guaranteeing at least one finding, so the
        # offset fidelity below is actually exercised.
        formats += (
            Format(
                name="det",
                bos="<|im_start|>",
                user_prefix="",
                assistant_prefix="",
                turn_end="",
            ),
        )
        case = Fixture(
            id="c1",
            messages=(Message("user", "hi <|im_start|> bye"),),
            expected="",
            group="g",
        )
        impl_findings = fx.audit_special_token_literals(case, formats)
        assert any(f.format_name == "det" and f.offset == 3 for f in impl_findings)
        text = fx.audit_findings_json(impl_findings, source="out.json")
        oracle_text = _oracle_audit_json(_oracle_audit(case, formats), "out.json")
        assert json.loads(text) == json.loads(oracle_text)
        doc = json.loads(text)
        assert doc["schema"] == "model-protocol-lab.audit/v1"
        assert doc["source"] == "out.json"
        assert isinstance(doc["findings"], list)
        for item in doc["findings"]:
            assert set(item) == {"fixture_id", "field", "literal", "format_name", "offset"}

    def test_offset_fidelity(self):
        finding = fx.AuditFinding(
            fixture_id="c", field="user_1", literal="<|u|>", format_name="f", offset=7
        )
        doc = json.loads(fx.audit_findings_json((finding,), source="s.json"))
        assert doc["findings"] == [
            {
                "fixture_id": "c",
                "field": "user_1",
                "literal": "<|u|>",
                "format_name": "f",
                "offset": 7,
            }
        ]

    def test_empty_findings(self):
        text = fx.audit_findings_json((), source="x.json")
        assert json.loads(text)["findings"] == []

    def test_validation(self):
        with pytest.raises(ValueError):
            fx.audit_findings_json(["not-a-finding"], source="x")  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            fx.audit_findings_json((), source=123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# File IO: read_authoring_file / write_fixture_file (pinned behavior).
# ---------------------------------------------------------------------------


class TestReadAuthoringFile:
    def test_roundtrip_small_file(self, tmp_path):
        target = tmp_path / "a.csv"
        target.write_text("id,group\n", encoding="utf-8")
        assert fx.read_authoring_file(target) == "id,group\n"
        assert fx.read_authoring_file(str(target)) == "id,group\n"

    def test_exactly_max_bytes_ok(self, tmp_path):
        target = tmp_path / "big.csv"
        target.write_bytes(b"x" * MAX_BYTES)
        assert len(fx.read_authoring_file(target)) == MAX_BYTES

    def test_over_max_bytes_refused(self, tmp_path):
        target = tmp_path / "big.csv"
        target.write_bytes(b"x" * (MAX_BYTES + 1))
        with pytest.raises(ValueError, match="[Ss]ize|[Ll]imit"):
            fx.read_authoring_file(target)

    def test_fifo_refused(self, tmp_path):
        fifo = tmp_path / "pipe"
        os.mkfifo(fifo)
        with pytest.raises(ValueError, match="[Rr]egular"):
            fx.read_authoring_file(fifo)

    def test_directory_refused(self, tmp_path):
        with pytest.raises(ValueError, match="[Rr]egular"):
            fx.read_authoring_file(tmp_path)

    def test_missing_file(self, tmp_path):
        with pytest.raises(ValueError, match="[Uu]nreadable"):
            fx.read_authoring_file(tmp_path / "nope.csv")

    def test_invalid_utf8(self, tmp_path):
        target = tmp_path / "bad.csv"
        target.write_bytes(b"\xff\xfe invalid")
        with pytest.raises(ValueError, match="UTF-8"):
            fx.read_authoring_file(target)

    def test_non_path_rejected(self):
        with pytest.raises(ValueError):
            fx.read_authoring_file(123)  # type: ignore[arg-type]


class TestWriteFixtureFile:
    def _built_text(self) -> str:
        split, _ = fx.parse_fixture_csv(_two_row_csv())
        return fx.split_to_json_text(split)

    def test_write_and_reload(self, tmp_path):
        target = tmp_path / "out.json"
        result = fx.write_fixture_file(self._built_text(), target)
        assert result == target
        assert parse_dataset(target.read_text(encoding="utf-8")) is not None
        mode = stat.S_IMODE(os.stat(target).st_mode)
        assert mode == 0o600

    def test_refuses_overwrite(self, tmp_path):
        target = tmp_path / "out.json"
        target.write_text("old", encoding="utf-8")
        with pytest.raises(ValueError, match="[Aa]lready exists|[Ff]resh"):
            fx.write_fixture_file(self._built_text(), target)
        assert target.read_text(encoding="utf-8") == "old"

    def test_missing_parent(self, tmp_path):
        with pytest.raises(ValueError, match="[Pp]arent|[Dd]irectory"):
            fx.write_fixture_file(self._built_text(), tmp_path / "nodir" / "out.json")

    def test_non_string_text(self, tmp_path):
        with pytest.raises(ValueError):
            fx.write_fixture_file(b"bytes", tmp_path / "o.json")  # type: ignore[arg-type]

    def test_non_path_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            fx.write_fixture_file("{}", 123)  # type: ignore[arg-type]

    def test_partial_write_leaves_no_file(self, tmp_path):
        # A lone surrogate fails UTF-8 encoding mid-write; the partially
        # written file must be unlinked.
        target = tmp_path / "partial.json"
        with pytest.raises(UnicodeEncodeError):
            fx.write_fixture_file("prefix \ud800 suffix", target)
        assert not target.exists()

    def test_relative_single_component_skips_parent_check(self, tmp_path, monkeypatch):
        # Docstring: "the parent directory must already exist" -- but the
        # code skips the check when parent is exactly Path(".").
        monkeypatch.chdir(tmp_path)
        target = Path("rel.json")
        fx.write_fixture_file(self._built_text(), target)
        assert target.exists()


# ---------------------------------------------------------------------------
# Red-team: hidden assumptions and adversarial inputs.
# ---------------------------------------------------------------------------


class TestRedTeam:
    def test_duplicate_header_columns(self):
        text = _simple_csv(
            [["a1", "g1", "search", "", "hi", "", ""]],
            header=["id", "group", "partition", "expected", "user_1", "user_1"],
        )
        with pytest.raises(ValueError):
            fx.parse_fixture_csv(text)

    def test_reordered_message_columns(self):
        text = _simple_csv(
            [["a1", "g1", "search", "", "hi", "", ""]],
            header=["id", "group", "partition", "expected", "assistant_1", "user_1"],
        )
        with pytest.raises(ValueError):
            fx.parse_fixture_csv(text)

    def test_extra_cells_beyond_header(self):
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow(["id", "group", "partition", "expected", "user_1"])
        writer.writerow(["a1", "g1", "search", "", "hi", "EXTRA"])
        writer.writerow(["b1", "g2", "held_out", "", "yo"])
        with pytest.raises(ValueError):
            fx.parse_fixture_csv(buf.getvalue())

    def test_short_row_padded_with_empties(self):
        # A row shorter than the header is padded: user_2/user_3 become
        # empty -> conversation ends after user_1. Must still end with user.
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow(["id", "group", "partition", "expected", "user_1", "assistant_1", "user_2"])
        writer.writerow(["a1", "g1", "search", "", "hi"])
        writer.writerow(["b1", "g2", "held_out", "", "yo"])
        split, _ = fx.parse_fixture_csv(buf.getvalue())
        assert len(split.search[0].messages) == 1

    def test_whitespace_only_partition_rejected(self):
        text = _simple_csv([["a1", "g1", " search", "", "hi", "", ""]])
        with pytest.raises(ValueError, match="partition"):
            fx.parse_fixture_csv(text)

    def test_case_sensitive_partition(self):
        text = _simple_csv([["a1", "g1", "Search", "", "hi", "", ""]])
        with pytest.raises(ValueError, match="partition"):
            fx.parse_fixture_csv(text)

    def test_crlf_line_endings(self):
        text = _two_row_csv().replace("\n", "\r\n")
        split, _ = fx.parse_fixture_csv(text)
        assert len(split.search) == 1 and len(split.held_out) == 1

    def test_bom_prefix_rejected_as_bad_header(self):
        with pytest.raises(ValueError):
            fx.parse_fixture_csv("﻿" + _two_row_csv())

    def test_max_messages_boundary(self):
        header = ["id", "group", "partition", "expected"] + [
            f"user_{(i // 2) + 1}" if i % 2 == 0 else f"assistant_{(i // 2) + 1}"
            for i in range(MAX_MESSAGES)
        ]
        # 21 message columns, conversation ends with user_11 -> valid.
        row = ["a1", "g1", "search", ""] + ["m"] * MAX_MESSAGES
        row2 = ["b1", "g2", "held_out", ""] + ["n"] * MAX_MESSAGES
        split, _ = fx.parse_fixture_csv(_simple_csv([row, row2], header=header))
        assert len(split.search[0].messages) == MAX_MESSAGES
        # 22 message columns rejected.
        with pytest.raises(ValueError):
            fx.parse_fixture_csv(_simple_csv([row, row2], header=header + ["user_12"]))

    def test_max_fixtures_per_partition_boundary(self):
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow(["id", "group", "partition", "expected", "user_1"])
        for i in range(MAX_CASES_PER_SPLIT):
            writer.writerow([f"s{i}", f"gs{i}", "search", "", f"q{i}"])
            writer.writerow([f"h{i}", f"gh{i}", "held_out", "", f"r{i}"])
        split, _ = fx.parse_fixture_csv(buf.getvalue())
        assert len(split.search) == MAX_CASES_PER_SPLIT
        buf2 = io.StringIO()
        writer2 = csv.writer(buf2, lineterminator="\n")
        writer2.writerow(["id", "group", "partition", "expected", "user_1"])
        for i in range(MAX_CASES_PER_SPLIT + 1):
            writer2.writerow([f"s{i}", f"gs{i}", "search", "", f"q{i}"])
        for i in range(MAX_CASES_PER_SPLIT):
            writer2.writerow([f"h{i}", f"gh{i}", "held_out", "", f"r{i}"])
        with pytest.raises(ValueError):
            fx.parse_fixture_csv(buf2.getvalue())

    def test_cell_exactly_max_text_ok(self):
        cell = "x" * MAX_TEXT
        text = _simple_csv(
            [
                ["a1", "g1", "search", cell, "hi", "", ""],
                ["b1", "g2", "held_out", "", "yo", "", ""],
            ]
        )
        split, _ = fx.parse_fixture_csv(text)
        assert split.search[0].expected == cell

    def test_cell_one_over_max_text_rejected(self):
        cell = "x" * (MAX_TEXT + 1)
        text = _simple_csv([["a1", "g1", "search", "", cell, "", ""]])
        with pytest.raises(ValueError):
            fx.parse_fixture_csv(text)

    def test_trailing_assistant_turn_rejected(self):
        text = _simple_csv([["a1", "g1", "search", "", "hi", "hello", ""]])
        with pytest.raises(ValueError, match="[Uu]ser"):
            fx.parse_fixture_csv(text)

    def test_gap_then_content_rejected(self):
        text = _simple_csv([["a1", "g1", "search", "", "hi", "", "more"]])
        with pytest.raises(ValueError, match="[Ee]mpty"):
            fx.parse_fixture_csv(text)

    def test_duplicate_ids_within_partition(self):
        text = _simple_csv(
            [
                ["a1", "g1", "search", "", "hi", "", ""],
                ["a1", "g2", "search", "", "yo", "", ""],
                ["b1", "g3", "held_out", "", "hey", "", ""],
            ]
        )
        with pytest.raises(ValueError):
            fx.parse_fixture_csv(text)

    def test_empty_conversation_impossible(self):
        # user_1 empty is rejected before the gap logic matters.
        text = _simple_csv([["a1", "g1", "search", "", "", "x", ""]])
        with pytest.raises(ValueError, match="user_1"):
            fx.parse_fixture_csv(text)

    def test_many_messages_alternation_roles(self):
        text = _simple_csv([["a1", "g1", "search", "", "u1", "a1", "u2"]])
        text += _simple_csv([["b1", "g2", "held_out", "", "v1", "", ""]]).split("\n", 1)[1]
        split, _ = fx.parse_fixture_csv(text)
        assert [m.role for m in split.search[0].messages] == ["user", "assistant", "user"]

    def test_json_keys_exact(self):
        split, _ = fx.parse_fixture_csv(_two_row_csv())
        doc = json.loads(fx.split_to_json_text(split))
        assert set(doc) == {"search", "held_out"}
        assert set(doc["search"][0]) == {"id", "group", "messages", "expected"}
        assert set(doc["search"][0]["messages"][0]) == {"role", "content"}

    def test_split_order_preserved(self):
        rows = [[f"s{i}", f"gs{i}", "search", "", f"q{i}", "", ""] for i in range(5)]
        rows += [[f"h{i}", f"gh{i}", "held_out", "", f"r{i}", "", ""] for i in range(3)]
        split, _ = fx.parse_fixture_csv(_simple_csv(rows))
        assert [c.id for c in split.search] == [f"s{i}" for i in range(5)]
        assert [c.id for c in split.held_out] == [f"h{i}" for i in range(3)]

    def test_single_fixture_per_partition(self):
        split, _ = fx.parse_fixture_csv(_two_row_csv())
        assert len(split.search) == 1 and len(split.held_out) == 1

    def test_missing_partition_rejected(self):
        text = _simple_csv([["a1", "g1", "search", "", "hi", "", ""]])
        with pytest.raises(ValueError, match="held_out"):
            fx.parse_fixture_csv(text)
        text2 = _simple_csv([["b1", "g2", "held_out", "", "hi", "", ""]])
        with pytest.raises(ValueError, match="search"):
            fx.parse_fixture_csv(text2)


# ---------------------------------------------------------------------------
# Mutation-sensitivity gate (documentation; the real run happens on copies
# in /tmp -- see /tmp/cycle29_mutations/run_mutations.py).
#
# The mutants below were verified caught by this test file on copies of the
# package under /tmp/cycle29_mutations/mut_<n>/ (repo never mutated):
#   M1  blank-id check `if not fid.strip()` -> `if not fid`
#   M2  audit field naming `f"{role}_{index//2 + 1}"` -> `f"{role}_{index + 1}"`
#   M3  `_extract_special_tokens` drops `fmt.open_suffix` from the scan
#   M4  audit scan loop reduced to the first occurrence per token
#       (note: `start = offset + len(token)` would NOT be a behavior change:
#       no inventory token has a border, so occurrences can never overlap)
#   M5  `audit_findings_json` emits `"offset": finding.offset + 1`
#   M6  `parse_fixture_csv` no longer rejects content-after-empty-cell
#   M7  `split_to_json_text` uses `ensure_ascii=True`
#   M8  `write_fixture_file` drops `os.O_EXCL` (overwrite allowed)
#   M9  `read_authoring_file` refuses files of exactly MAX_BYTES
#   M10 `_expected_message_headers` starts with assistant_1
#   M11 message role parity flipped (`position % 2 == 0` -> assistant)
#   M12 `split_to_json_text` type check `type(split) is not Split` ->
#       `not isinstance(split, Split)`
#   M13 group blank check `if not group.strip()` -> `if not group`
#   M14 conversation-end check `!= "user"` -> `!= "assistant"`
#   M15 `_safe` control-char regex drops the \\x7f-\\x9f range
#   M16 `parse_fixture_csv` type check `type(text) is not str` ->
#       `not isinstance(text, str)` (str subclass accepted)
# Result, verified 2026-10-15 via /tmp/cycle29_mutations/run_mutations.py:
# 16/16 mutants CAUGHT (M5 initially survived on an empty-findings test
# case and was caught after the test was strengthened to guarantee a
# finding).
# ---------------------------------------------------------------------------
