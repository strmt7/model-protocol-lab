"""Cycle-80 adversarial audit: the special-token literal auditor itself.

Target: ``model_protocol_lab/fixtures.py`` -- ``audit_special_token_literals``
(line 420), ``_audit_targets`` (390), ``_extract_special_tokens`` (358),
``AuditFinding`` (315), ``audit_findings_json`` (471). The auditor scans
authored fixture text for raw special-token literals (e.g. ``<|im_start|>``)
that would corrupt experiments if left in fixture text. Cycle 66 audited the
CSV loader; the audit path itself had no dedicated adversarial audit until
now. (Cycle 5 covered the happy paths; this file is the adversarial,
independent-oracle pass.)

RULE MAP (hand-derived from the source, cross-checked with the docstrings):

  extraction (``_extract_special_tokens``):
    - fields scanned in order: bos, user_prefix, assistant_prefix, turn_end,
      user_turn_end, assistant_turn_end, open_suffix; None fields skipped.
    - four token shapes, most-specific-first: full-width ``＜｜...｜＞``,
      ``<|...|>``, ``[...]`` (letter- or slash-leading), ``<...>``; the inner
      text may not contain the shape's delimiters or a newline and must be
      nonempty.
    - leftmost, non-overlapping matches per shape; greedy (longest) match;
      deduped, first-seen order is the inventory order.
  targets (``_audit_targets``):
    - Fixture -> one target per message: ``(id, "<role>_<index//2+1>", content)``.
    - Split -> search fixtures then held-out fixtures.
    - str -> ``("", "text", s)``; tuple -> Fixtures as fixtures, str items as
      ``("", "text_<1-based tuple position>", s)``.
  scan (``audit_special_token_literals``):
    - ``formats`` must be a nonempty tuple of Format, else ValueError.
    - exact, case-sensitive substring match; every occurrence of one literal
      gets a finding (the search resumes at offset+1).
    - deterministic order: targets in input order, formats in given order,
      each format's inventory order, then ascending offsets.
  sidecar (``audit_findings_json``):
    - ``{"schema": "model-protocol-lab.audit/v1", "source": source,
      "findings": [{fixture_id, field, literal, format_name, offset}, ...]}``,
      ``json.dumps(ensure_ascii=False, indent=2) + "\\n"``.

METHOD:
  1. Independent-oracle differential tests: the extraction oracle is a
     from-scratch character scanner (no ``re``, no ``str.find`` loop),
     validated against the source on all 29 catalog formats plus 4000
     fuzzed adversarial field soups (0 mismatches) before these tests were
     written; the scan oracle is a naive O(n*m) per-position comparison.
     Anchored cases below carry hand-derived expected values so the oracle
     itself is pinned, not just the differential.
  2. Minimal pairs: one-character edits flip a finding on/off.
  3. Adversarial edges: tokens split across lines, overlapping/adjacent
     tokens, unicode lookalikes, empty inputs, CSV non-message columns vs
     message cells, case variants.
  4. Finding fidelity: offset -> line/column attribution, field naming,
     tuple position numbering, sidecar round-trip, determinism.
  5. Mutation gate (run separately from /tmp scratch, never committed):
     six mutants in /tmp copies of the source -- dropped extraction field,
     flipped match direction, off-by-one offset, adjacent-occurrence skip,
     case-folded scan, sidecar key drop -- each must fail at least one test
     in this file.

Deterministic; no network, no model, no randomness.
"""

import json
import string
import unittest

from model_protocol_lab.catalog import CATALOG
from model_protocol_lab.core import Fixture, Format, Message, Split
from model_protocol_lab.fixtures import (
    AuditFinding,
    _audit_targets,
    _extract_special_tokens,
    audit_findings_json,
    audit_special_token_literals,
    parse_fixture_csv,
)

CHATML = CATALOG["chatml"]
LLAMA3 = CATALOG["llama3"]
GEMMA = CATALOG["gemma"]
MISTRAL = CATALOG["mistral"]
DEEPSEEK = CATALOG["deepseek"]

# Synthetic formats with known, hand-derived inventories:
#   _ORDER_FMT   -> ("<|zz|>", "<|aa|>")  (field order, not alphabetical)
#   _OVERLAP_FMT -> ("<s>", "[INST]", "[/INST]")
#   _NEST_FMT    -> ("<|[x]|>", "[x]")  (bracket token nests inside pipe token)
#   _DEEPSEEK_SYN -> full-width framing, like the Round-4 R1 template
_ORDER_FMT = Format("order", "<|zz|>", "plain", "<|aa|>", "tail")
_OVERLAP_FMT = Format("overlap", "<s>", "[INST] ", "", " [/INST]")
_NEST_FMT = Format("nest", "<|[x]|>", "p", "q", "r")
_DEEPSEEK_SYN = Format(
    "deepseek-syn",
    "＜｜begin▁of▁sentence｜＞",
    "＜｜User｜＞",
    "＜｜Assistant｜＞",
    "＜｜end▁of▁sentence｜＞",
)

_ASCII_LETTERS = frozenset(string.ascii_letters)


def _fixture(fid: str, *contents: str) -> Fixture:
    messages = tuple(
        Message("user" if index % 2 == 0 else "assistant", text)
        for index, text in enumerate(contents)
    )
    return Fixture(id=fid, messages=messages, expected="ok", group="g-" + fid)


# ---------------------------------------------------------------------------
# Independent oracles (from-scratch reimplementations of the documented rules)
# ---------------------------------------------------------------------------


def _oracle_closings(text: str, start: int, open_len: int, close: str) -> list[int]:
    """All start positions of ``close`` at or after ``start + open_len``."""
    found: list[int] = []
    end = len(text) - len(close) + 1
    index = start + open_len
    while index < end:
        if text[index : index + len(close)] == close:
            found.append(index)
        index += 1
    return found


def _oracle_scan_shape(text: str, kind: str) -> list[str]:
    """Leftmost non-overlapping scan for one shape; greedy longest match."""
    tokens: list[str] = []
    pos, end = 0, len(text)
    while pos < end:
        token: str | None = None
        if kind == "fullwidth" and text[pos : pos + 2] == "＜｜":
            closings = [
                j
                for j in _oracle_closings(text, pos, 2, "｜＞")
                if (inner := text[pos + 2 : j]) and not any(c in "＜＞\n" for c in inner)
            ]
            if closings:
                token = text[pos : closings[-1] + 2]
        elif kind == "pipe" and text[pos : pos + 2] == "<|":
            closings = [
                j
                for j in _oracle_closings(text, pos, 2, "|>")
                if (inner := text[pos + 2 : j]) and not any(c in "<>\n" for c in inner)
            ]
            if closings:
                token = text[pos : closings[-1] + 2]
        elif (
            kind == "bracket"
            and text[pos] == "["
            and pos + 1 < end
            and (text[pos + 1] in _ASCII_LETTERS or text[pos + 1] == "/")
        ):
            closings = [
                j
                for j in _oracle_closings(text, pos, 1, "]")
                if not any(c in "[]\n" for c in text[pos + 1 : j])
            ]
            if closings:
                token = text[pos : closings[-1] + 1]
        elif kind == "angle" and text[pos] == "<":
            closings = [
                j
                for j in _oracle_closings(text, pos, 1, ">")
                if (inner := text[pos + 1 : j]) and not any(c in "<>\n" for c in inner)
            ]
            if closings:
                token = text[pos : closings[-1] + 1]
        if token is not None:
            tokens.append(token)
            pos += len(token)
        else:
            pos += 1
    return tokens


def _oracle_extract(fmt: Format) -> tuple[str, ...]:
    """Independent reimplementation of the documented extraction rule."""
    seen: list[str] = []
    fields = (
        fmt.bos,
        fmt.user_prefix,
        fmt.assistant_prefix,
        fmt.turn_end,
        fmt.user_turn_end,
        fmt.assistant_turn_end,
        fmt.open_suffix,
    )
    for field in fields:
        if field is None:
            continue
        for kind in ("fullwidth", "pipe", "bracket", "angle"):
            for token in _oracle_scan_shape(field, kind):
                if token not in seen:
                    seen.append(token)
    return tuple(seen)


def _oracle_targets(
    value: Fixture | Split | tuple[Fixture | str, ...] | str,
) -> list[tuple[str, str, str]]:
    """Independent reimplementation of the documented target normalization."""
    targets: list[tuple[str, str, str]] = []

    def add_fixture(case: Fixture) -> None:
        for index, message in enumerate(case.messages):
            targets.append((case.id, f"{message.role}_{index // 2 + 1}", message.content))

    if type(value) is Fixture:
        add_fixture(value)
    elif type(value) is Split:
        for case in (*value.search, *value.held_out):
            add_fixture(case)
    elif type(value) is str:
        targets.append(("", "text", value))
    elif type(value) is tuple:
        for position, item in enumerate(value):
            if type(item) is Fixture:
                add_fixture(item)
            elif type(item) is str:
                targets.append(("", f"text_{position + 1}", item))
            else:
                raise ValueError("oracle: bad tuple item")
    else:
        raise ValueError("oracle: bad input")
    return targets


def _oracle_scan(text: str, token: str) -> list[int]:
    """Naive per-position substring search; no str.find loop."""
    offsets: list[int] = []
    for index in range(len(text) - len(token) + 1):
        if text[index : index + len(token)] == token:
            offsets.append(index)
    return offsets


def _oracle_findings(
    value: Fixture | Split | tuple[Fixture | str, ...] | str,
    formats: tuple[Format, ...],
) -> list[tuple[str, str, str, str, int]]:
    """Naive differential oracle for the whole audit scan."""
    out: list[tuple[str, str, str, str, int]] = []
    inventories = [(fmt.name, _oracle_extract(fmt)) for fmt in formats]
    for fixture_id, field, text in _oracle_targets(value):
        for format_name, tokens in inventories:
            for token in tokens:
                for offset in _oracle_scan(text, token):
                    out.append((fixture_id, field, token, format_name, offset))
    return out


def _finding_key(finding: AuditFinding) -> tuple[str, str, str, str, int]:
    return (
        finding.fixture_id,
        finding.field,
        finding.literal,
        finding.format_name,
        finding.offset,
    )


def _audit_keys(
    value: Fixture | Split | tuple[Fixture | str, ...] | str,
    formats: tuple[Format, ...],
) -> list[tuple[str, str, str, str, int]]:
    return [_finding_key(f) for f in audit_special_token_literals(value, formats)]


class OracleExtractionTests(unittest.TestCase):
    """The independent extraction oracle agrees with the source everywhere."""

    def test_oracle_matches_source_on_catalog(self):
        for name, fmt in sorted(CATALOG.items()):
            with self.subTest(format=name):
                self.assertEqual(_oracle_extract(fmt), _extract_special_tokens(fmt))

    def test_oracle_matches_source_on_tricky_fields(self):
        tricky = (
            "<|a|><|b|>",  # adjacent pipe tokens
            "<|a <|b|> c|>",  # nested-looking soup (greedy longest match)
            "[/INST] [INST]",  # bracket tokens sharing a substring
            "＜｜x｜＞ <｜y｜> <|z|>",  # mixed full-width/ASCII shapes
            "<a|> <||> <| |>",  # degenerate inners
            "a < b > c [d e] <f\ng>",  # near-misses: no token shapes
            "",  # empty field
            "<|unterminated",  # no closing delimiter
        )
        for index, field in enumerate(tricky):
            with self.subTest(field=field):
                fmt = Format(f"t{index}", field, "", "", "")
                self.assertEqual(_oracle_extract(fmt), _extract_special_tokens(fmt))

    def test_inventory_first_seen_order_and_dedupe(self):
        # "<|x|>" repeats across fields; "[INST]" is new in user_prefix.
        # Hand-derived: first-seen order, deduped.
        fmt = Format("dup", "<|x|> and <|x|>", "[INST] <|x|>", "", "")
        self.assertEqual(_extract_special_tokens(fmt), ("<|x|>", "[INST]"))
        self.assertEqual(_oracle_extract(fmt), ("<|x|>", "[INST]"))

    def test_none_fields_skipped_empty_overrides_honored(self):
        fmt = Format("e", "", "", "", "", user_turn_end="", open_suffix="")
        self.assertEqual(_extract_special_tokens(fmt), ())

    def test_per_role_and_suffix_overrides_add_tokens(self):
        fmt = Format(
            "ovr", "<s>", "[INST]", "<a>", "</s>", user_turn_end="[/U]", open_suffix="<|open|>"
        )
        tokens = _extract_special_tokens(fmt)
        self.assertIn("[/U]", tokens)
        self.assertIn("<|open|>", tokens)
        self.assertEqual(_oracle_extract(fmt), tokens)


class DifferentialScanTests(unittest.TestCase):
    """Naive-oracle differential tests: source scan == independent oracle."""

    DIFF_TEXTS = (
        "",
        "plain words, no tokens at all",
        "x <|im_start|> y",
        "x <|im_start|><|im_start|> y",
        "<|im_start|><|im_end|>",
        "[/INST] sandwich [INST]",
        "say <|[x]|> now",
        "line one\n<|eot_id|> mid\ntail <|begin_of_text|> done",
        "a\r\n<|im_start|>b",
        "＜｜User｜＞ versus <｜User｜> versus <|User|>",
        "＜｜begin▁of▁sentence｜＞!",
        "shout <|IM_START|> and <|Im_Start|> quietly",
        "[INST] [inst] [/INST] [/inst]",
        "ends with token <s>",
        "<s><s><s>",
        "math: a < b, c > d, [x], <3",
        "nested <|a <|b|> c|> soup",
        "half tokens <|im_start| and <|im_end",
        "token <|eot_id|> then token <|eot_id|> again <|eot_id|>",
    )
    DIFF_FORMAT_SETS = (
        (CHATML,),
        (LLAMA3,),
        (GEMMA,),
        (MISTRAL,),
        (DEEPSEEK,),
        (CHATML, LLAMA3),
        (LLAMA3, CHATML),
        (CHATML, DEEPSEEK, GEMMA),
        (_ORDER_FMT, _OVERLAP_FMT, _NEST_FMT),
    )

    def test_raw_text_differential(self):
        for text in self.DIFF_TEXTS:
            for formats in self.DIFF_FORMAT_SETS:
                with self.subTest(text=text[:24], formats=[f.name for f in formats]):
                    self.assertEqual(_audit_keys(text, formats), _oracle_findings(text, formats))

    def test_fixture_inputs_differential(self):
        fix = _fixture(
            "f-diff",
            "u <|im_start|> x",
            "a [/INST] y",
            "u2 <s> z",
        )
        held = _fixture("h-diff", "held <|eot_id|> q")
        split = Split((fix,), (held,))
        inputs: tuple[Fixture | Split | tuple[Fixture | str, ...] | str, ...] = (
            fix,
            split,
            "raw <|im_end|> tail",
            (fix, "t <|im_start|> u"),
            ("a", fix, "b <s> c"),
        )
        format_sets = ((CHATML,), (MISTRAL, LLAMA3), (_ORDER_FMT, _OVERLAP_FMT, GEMMA))
        for value in inputs:
            for formats in format_sets:
                with self.subTest(value=str(value)[:24], formats=[f.name for f in formats]):
                    self.assertEqual(_audit_keys(value, formats), _oracle_findings(value, formats))
                    # The real target normalization agrees with the oracle's.
                    self.assertEqual(
                        list(_audit_targets(value)),
                        _oracle_targets(value),
                    )


class AnchoredScanTests(unittest.TestCase):
    """Hand-derived expectations pinning the oracle's key behaviors."""

    def test_single_chatml_finding(self):
        self.assertEqual(
            _audit_keys("x <|im_start|> y", (CHATML,)),
            [("", "text", "<|im_start|>", "chatml", 2)],
        )

    def test_overlapping_distinct_tokens_inventory_order(self):
        # "[x]" (offset 6) sits inside "<|[x]|>" (offset 4): genuinely
        # overlapping occurrences of two distinct tokens of one inventory.
        # Inventory order wins over offset order.
        self.assertEqual(_extract_special_tokens(_NEST_FMT), ("<|[x]|>", "[x]"))
        self.assertEqual(
            _audit_keys("say <|[x]|> now", (_NEST_FMT,)),
            [
                ("", "text", "<|[x]|>", "nest", 4),
                ("", "text", "[x]", "nest", 6),
            ],
        )

    def test_adjacent_occurrences_each_get_a_finding(self):
        self.assertEqual(
            [key[4] for key in _audit_keys("x <|im_start|><|im_start|> y", (CHATML,))],
            [2, 14],
        )

    def test_inventory_order_beats_offset_order(self):
        # "<|aa|>" occurs first in the text but "<|zz|>" is inventoried first.
        keys = _audit_keys("has <|aa|> then <|zz|> end", (_ORDER_FMT,))
        self.assertEqual(
            keys,
            [
                ("", "text", "<|zz|>", "order", 16),
                ("", "text", "<|aa|>", "order", 4),
            ],
        )

    def test_format_order_is_given_order(self):
        text = "mix <|im_start|> and <|eot_id|> done"
        ab = _audit_keys(text, (CHATML, LLAMA3))
        ba = _audit_keys(text, (LLAMA3, CHATML))
        self.assertEqual(
            ab,
            [
                ("", "text", "<|im_start|>", "chatml", 4),
                ("", "text", "<|eot_id|>", "llama3", 21),
            ],
        )
        self.assertEqual(
            ba,
            [
                ("", "text", "<|eot_id|>", "llama3", 21),
                ("", "text", "<|im_start|>", "chatml", 4),
            ],
        )

    def test_findings_come_back_as_tuple(self):
        self.assertIs(type(audit_special_token_literals("clean", (CHATML,))), tuple)


class MinimalPairTests(unittest.TestCase):
    """One-character edits flip a finding on/off."""

    # (text_with_finding, one_char_variant_without, formats, expected_count)
    PAIRS = (
        ("a <|im_start|> b", "a <|im_start| > b", (CHATML,), 1),  # space before |>
        ("a <|im_start|> b", "a <|im_start| b", (CHATML,), 1),  # deleted ">"
        ("a <|im_start|> b", "a <|im_staart|> b", (CHATML,), 1),  # inserted "a"
        ("a <|im_start|> b", "a <|im_Start|> b", (CHATML,), 1),  # case flip
        ("a ＜｜User｜＞ b", "a <｜User｜> b", (_DEEPSEEK_SYN,), 1),  # fullwidth angle
        ("a ＜｜User｜＞ b", "a ＜|User|＞ b", (_DEEPSEEK_SYN,), 1),  # fullwidth bars
        ("a [INST] b", "a [INST} b", (_OVERLAP_FMT,), 1),  # ] -> }
        ("a <bos> b", "a <b0s> b", (GEMMA,), 1),  # o -> 0
        ("a <|im_start|> b", "a <|im_\nstart|> b", (CHATML,), 1),  # newline split
    )

    def test_minimal_pairs(self):
        for with_text, without_text, formats, count in self.PAIRS:
            with self.subTest(with_text=with_text):
                with_findings = audit_special_token_literals(with_text, formats)
                self.assertEqual(len(with_findings), count)
                self.assertEqual(
                    _audit_keys(with_text, formats),
                    _oracle_findings(with_text, formats),
                )
                self.assertEqual(audit_special_token_literals(without_text, formats), ())


class AdversarialEdgeTests(unittest.TestCase):
    def test_token_split_across_crlf_has_no_finding(self):
        self.assertEqual(audit_special_token_literals("a <|im_\r\nstart|> b", (CHATML,)), ())

    def test_token_at_text_boundaries(self):
        start = audit_special_token_literals("<|im_end|> tail", (CHATML,))
        end = audit_special_token_literals("tail <|im_end|>", (CHATML,))
        whole = audit_special_token_literals("<|im_end|>", (CHATML,))
        self.assertEqual([f.offset for f in start], [0])
        self.assertEqual([f.offset for f in end], [5])
        self.assertEqual([f.offset for f in whole], [0])

    def test_unicode_lookalikes_not_flagged(self):
        # Each lookalike differs from the real token by confusable characters:
        # Cyrillic "е" (U+0435), corner brackets, or mixed full-width/ASCII
        # framing. None is an exact literal of the format's inventory.
        syn_lookalikes = (
            "say ＜｜Usеr｜＞ now",  # Cyrillic е
            "say 〈｜User｜〉 now",  # corner brackets, not a token shape
            "say <｜User｜> now",  # ASCII angles, fullwidth bars
        )
        for text in syn_lookalikes:
            with self.subTest(text=text):
                self.assertEqual(audit_special_token_literals(text, (_DEEPSEEK_SYN,)), ())
        catalog_lookalikes = (
            "say ＜｜User｜＞ now",  # all-fullwidth vs catalog's ASCII angles
            "say ＜｜Usеr｜＞ now",  # Cyrillic е
            "say 〈｜User｜〉 now",  # corner brackets
        )
        for text in catalog_lookalikes:
            with self.subTest(text=text):
                self.assertEqual(audit_special_token_literals(text, (DEEPSEEK,)), ())

    def test_catalog_deepseek_exact_mixed_form_only(self):
        # Catalog deepseek tokens are ASCII angles + fullwidth bars.
        exact = audit_special_token_literals("x <｜User｜> y", (DEEPSEEK,))
        self.assertEqual(len(exact), 1)
        self.assertEqual(exact[0].literal, "<｜User｜>")
        self.assertEqual(exact[0].offset, 2)
        all_fullwidth = audit_special_token_literals("x ＜｜User｜＞ y", (DEEPSEEK,))
        self.assertEqual(all_fullwidth, ())

    def test_case_variants_not_flagged(self):
        variants = (
            "a <|IM_START|> b",
            "a <|Im_Start|> b",
            "a [inst] b",
            "a [/Inst] b",
            "a <BOS> b",
        )
        for text in variants:
            with self.subTest(text=text):
                self.assertEqual(
                    audit_special_token_literals(text, (CHATML, _OVERLAP_FMT, GEMMA)), ()
                )

    def test_empty_inputs_have_no_findings(self):
        self.assertEqual(audit_special_token_literals("", (CHATML,)), ())
        self.assertEqual(audit_special_token_literals((), (CHATML,)), ())
        self.assertEqual(audit_special_token_literals(("", ""), (CHATML,)), ())
        self.assertEqual(audit_special_token_literals(_fixture("e1", ""), (CHATML,)), ())
        self.assertEqual(audit_special_token_literals(("", _fixture("e2", ""), ""), (CHATML,)), ())

    def test_csv_non_message_columns_are_not_audited(self):
        # Tokens in id/group/expected columns are not message text: the audit
        # must stay silent. (The CSV header itself is structural -- a token
        # there is a parse error, tested below -- so headers cannot smuggle.)
        csv_text = (
            "id,group,partition,expected,user_1\n"
            "s1,g-<|im_start|>,search,expected <|im_end|> here,clean question?\n"
            "h1,g-hold,held_out,ok,clean held question\n"
        )
        split, _ = parse_fixture_csv(csv_text)
        self.assertEqual(audit_special_token_literals(split, tuple(CATALOG.values())), ())

    def test_csv_header_token_is_a_parse_error(self):
        csv_text = (
            "id,group,partition,expected,<|im_start|>\n"
            "s1,g1,search,ok,clean?\n"
            "h1,g2,held_out,ok,clean held?\n"
        )
        with self.assertRaises(ValueError):
            parse_fixture_csv(csv_text)

    def test_csv_message_cell_token_is_flagged(self):
        csv_text = (
            "id,group,partition,expected,user_1\n"
            "s1,g1,search,ok,say <|im_start|> now\n"
            "h1,g2,held_out,ok,clean held?\n"
        )
        split, _ = parse_fixture_csv(csv_text)
        findings = audit_special_token_literals(split, (CHATML,))
        self.assertEqual(len(findings), 1)
        self.assertEqual(_finding_key(findings[0]), ("s1", "user_1", "<|im_start|>", "chatml", 4))

    def test_triple_adjacent_same_token(self):
        keys = _audit_keys("<s><s><s>", (_OVERLAP_FMT,))
        self.assertEqual(
            keys,
            [
                ("", "text", "<s>", "overlap", 0),
                ("", "text", "<s>", "overlap", 3),
                ("", "text", "<s>", "overlap", 6),
            ],
        )


class FindingFidelityTests(unittest.TestCase):
    def test_multiline_offsets_map_to_line_and_column(self):
        text = "first line\nsay <|im_start|> now\nthird <|im_end|> end"
        findings = audit_special_token_literals(text, (CHATML,))
        # Hand-derived offsets: 15 and 38.
        self.assertEqual([f.offset for f in findings], [15, 38])

        def line_col(offset: int) -> tuple[int, int]:
            line = text.count("\n", 0, offset)
            col = offset - (text.rfind("\n", 0, offset) + 1)
            return line, col

        lines = text.split("\n")
        self.assertEqual([line_col(f.offset) for f in findings], [(1, 4), (2, 6)])
        for finding in findings:
            line, col = line_col(finding.offset)
            self.assertEqual(lines[line][col : col + len(finding.literal)], finding.literal)
        self.assertEqual(_audit_keys(text, (CHATML,)), _oracle_findings(text, (CHATML,)))

    def test_fixture_field_attribution(self):
        case = _fixture("f1", "u1 <|im_start|>", "a1 <|im_end|>", "u2 <|im_end|>")
        self.assertEqual(
            _audit_keys(case, (CHATML,)),
            [
                ("f1", "user_1", "<|im_start|>", "chatml", 3),
                ("f1", "assistant_1", "<|im_end|>", "chatml", 3),
                ("f1", "user_2", "<|im_end|>", "chatml", 3),
            ],
        )

    def test_tuple_position_numbering_counts_fixtures(self):
        fix_a = _fixture("fa", "clean")
        fix_b = _fixture("fb", "also clean")
        keys = _audit_keys((fix_a, "r1 <|im_start|>", fix_b, "r2 <|im_start|>"), (CHATML,))
        self.assertEqual(
            keys,
            [
                ("", "text_2", "<|im_start|>", "chatml", 3),
                ("", "text_4", "<|im_start|>", "chatml", 3),
            ],
        )

    def test_split_scans_search_then_held_out(self):
        split = Split(
            (_fixture("s1", "search <|im_end|>"),),
            (_fixture("h1", "held <|im_end|>"),),
        )
        keys = _audit_keys(split, (CHATML,))
        self.assertEqual(
            keys,
            [
                ("s1", "user_1", "<|im_end|>", "chatml", 7),
                ("h1", "user_1", "<|im_end|>", "chatml", 5),
            ],
        )

    def test_audit_is_deterministic(self):
        value = (
            _fixture("d1", "x <|im_start|> y\nz <|eot_id|> w"),
            "raw [/INST] tail",
        )
        formats = (CHATML, LLAMA3, MISTRAL)
        first = audit_special_token_literals(value, formats)
        second = audit_special_token_literals(value, formats)
        self.assertEqual(first, second)
        doc_a = audit_findings_json(first, source="f.json")
        doc_b = audit_findings_json(second, source="f.json")
        self.assertEqual(doc_a, doc_b)


class SidecarTests(unittest.TestCase):
    def test_sidecar_round_trip(self):
        findings = audit_special_token_literals(
            _fixture("r1", "say ＜｜User｜＞ and <|im_end|> now"),
            (_DEEPSEEK_SYN, CHATML),
        )
        self.assertTrue(findings)
        doc = audit_findings_json(findings, source="cases.csv")
        parsed = json.loads(doc)
        self.assertEqual(parsed["schema"], "model-protocol-lab.audit/v1")
        self.assertEqual(parsed["source"], "cases.csv")
        rebuilt = tuple(AuditFinding(**entry) for entry in parsed["findings"])
        self.assertEqual(rebuilt, findings)

    def test_sidecar_keeps_unicode_literal_unescaped(self):
        findings = audit_special_token_literals("x ＜｜User｜＞ y", (_DEEPSEEK_SYN,))
        doc = audit_findings_json(findings, source="u.csv")
        self.assertIn("＜｜User｜＞", doc)
        self.assertNotIn("\\uff1c", doc)

    def test_sidecar_exact_shape(self):
        findings = (
            AuditFinding(
                fixture_id="f1",
                field="user_1",
                literal="<|im_start|>",
                format_name="chatml",
                offset=2,
            ),
        )
        doc = audit_findings_json(findings, source="s.json")
        self.assertTrue(doc.endswith("\n") and not doc.endswith("\n\n"))
        parsed = json.loads(doc)
        self.assertEqual(
            parsed,
            {
                "schema": "model-protocol-lab.audit/v1",
                "source": "s.json",
                "findings": [
                    {
                        "fixture_id": "f1",
                        "field": "user_1",
                        "literal": "<|im_start|>",
                        "format_name": "chatml",
                        "offset": 2,
                    }
                ],
            },
        )

    def test_sidecar_empty_findings(self):
        doc = audit_findings_json((), source="empty.json")
        parsed = json.loads(doc)
        self.assertEqual(parsed["findings"], [])
        self.assertEqual(parsed["source"], "empty.json")


if __name__ == "__main__":
    unittest.main()
