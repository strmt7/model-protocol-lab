"""Cycle 221: record-notation probe (restated payload-
serialization candidate, fresh review BUILD-as-amended,
coordinator #65).

Spec citations are to docs/PAYLOADSERIALIZATION_DESIGN.md
(the restated candidate text of record is design Part II,
incorporated as amended) and to the fresh review's eight
amendments in the design doc's fresh-review adjudication
section (Part III).

Every amendment has named evidence here: the RNG call
sequence is re-executed literally and compared table for
table (amendment A); the instantiated-key overlap is shown
present in the generated panel with no redraw, and its
analytic rate is pinned (amendment B); the echo and parsed
diagnostics are shown independent in both directions on the
executed section 4.4 counterexample shape (amendment C); a
payload whose cell input_tokens disagree with its record arm
tokens is rejected by the mirror (amendment D); the section
6.1 power figures are re-derived by exact enumeration at
their stated marginal bases, including the corrected base-.90
panel figure (amendment E); the rendered section prints the
resolution beside the discordance count (amendment F); the
byte-cap relationships are measured, with the ASCII headroom
shown strictly below the candidate's printed 4.0x and the
non-ASCII-stressed panel shown over the cap (amendment G);
the flag census is re-derived by parser instantiation in
tests/test_cycle179_cli_dispatch.py's neighbourhood and in
the research log (amendment H).
"""

from __future__ import annotations

import copy
import importlib.util
import io
import json
import re
import threading
import unittest
from contextlib import redirect_stdout
from fractions import Fraction
from math import comb
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from model_protocol_lab import cli, diffreports, experiments, reporting, scorerlens
from model_protocol_lab import recordnotation as rn
from model_protocol_lab.core import Fixture, Format, Generation, Message

_CHATML = Format("chatml", "", "<|im_start|>user\n", "<|im_start|>assistant\n", "<|im_end|>\n")
_LLAMA3 = Format(
    "llama3",
    "<|begin_of_text|>",
    "<|start_header_id|>user<|end_header_id|>\n\n",
    "<|start_header_id|>assistant<|end_header_id|>\n\n",
    "<|eot_id|>",
)

_PROSE_RE = re.compile(
    r"code is ([\w-]+), the label is ([\w-]+)(?:, the mark is [\w-]+)?, and the zone is ([\w-]+)"
)
_JSON_RE = re.compile(
    r'"code": "([\w-]+)", "label": "([\w-]+)",(?: "mark": "[\w-]+",)? "zone": "([\w-]+)"'
)


class _WordBackend:
    """Word-token scripted backend.

    Modes: ``perfect`` (the expected answer parsed out of
    whichever rendering is present, boundary on; a non-value
    answer closed-book), ``json_drop`` (the record's label
    value instead of its code on the JSON arm only),
    ``closed_book_leak`` (the seed-0 record-00 expected value
    when no rendering is present, screening exactly that
    case), ``no_boundary`` (the expected answer with no
    boundary token anywhere -- evaluate takes the whole padded
    output as the answer, so cells fail as other_failure; true
    boundary_only cells are exercised at the aggregate and
    validator level, as in the sibling suites), ``explode``
    (generate raises).
    """

    fingerprint = "sha256:test-record-notation"

    def __init__(self, mode: str = "perfect") -> None:
        self.mode = mode
        self.calls = 0
        self.vocab: dict[str, int] = {}
        self.rev: dict[int, str] = {}

    def encode(self, text):
        ids = []
        for token in text.replace("\n", " \n ").split(" "):
            if not token:
                continue
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab) + 1
                self.rev[self.vocab[token]] = token
            ids.append(self.vocab[token])
        return tuple(ids)

    def decode(self, ids):
        return " ".join(self.rev.get(int(i), "?") for i in ids)

    def generate(self, input_ids, *, max_new_tokens, seed=None, cancel=None):
        self.calls += 1
        if self.mode == "explode":
            raise RuntimeError("boom")
        text = self.decode(list(input_ids))
        match = re.search(r"whose zone is ([\w-]+)\?", text)
        key = match.group(1) if match else None
        answer = None
        label = None
        if key is not None:
            for hit in _PROSE_RE.finditer(text):
                if hit.group(3) == key:
                    answer, label = hit.group(1), hit.group(2)
            for hit in _JSON_RE.finditer(text):
                if hit.group(3) == key:
                    answer, label = hit.group(1), hit.group(2)
        if answer is None:
            if self.mode == "closed_book_leak":
                answer = rn.generate_cases(0)[0]["expected"]
            else:
                answer = "unknownword"
        elif self.mode == "json_drop" and '"code":' in text:
            answer = label
        boundary = self.encode("endmark")[0]
        filler = self.encode("fillerword")[0]
        drop_boundary = self.mode == "no_boundary"
        out = list(self.encode(answer))
        if not drop_boundary:
            out.append(boundary)
        pad = filler if drop_boundary else boundary
        out = (out + [pad] * max_new_tokens)[:max_new_tokens]
        return Generation(token_ids=tuple(out), reason="limit")


def _payload(backend=None, formats=(_CHATML, _LLAMA3), seed=0, **stage_kwargs):
    backend = backend if backend is not None else _WordBackend()
    stage_kwargs.setdefault("max_context", 8192)
    stage = experiments.record_notation_stage(
        backend,
        formats,
        boundary=tuple(backend.encode("endmark")),
        seed=seed,
        **stage_kwargs,
    )
    return experiments.record_notation_analysis(stage, tuple(f.name for f in formats))


def _cell(verdict, digest, boundary=True, tokens=100, text=None, headline=True):
    cell = {
        "verdict": verdict,
        "boundary_found": boundary,
        "boundary_text_match": None,
        "input_tokens": tokens,
        "answer_digest": digest,
    }
    if headline:
        cell["answer_text"] = text
    return cell


def _record(case, prose_cell, json_cell, screen_cell, *, same_ids=False, details=None):
    expected_digest = rn.digest_text(str(case["expected"]))
    id_prose = rn.digest_text(f"ids-prose-{case['fixture_id']}")
    id_json = id_prose if same_ids else rn.digest_text(f"ids-json-{case['fixture_id']}")
    return {
        "fixture": case["fixture_id"],
        "n_records": len(case["table"]),
        "n_fields": len(case["table"][0]),
        "expected_digest": expected_digest,
        "content_digests": {arm: rn.digest_text(rn.arm_content(case, arm)) for arm in rn.ARM_KEYS},
        "arm_tokens": {"prose": 100, "canonical_json": 90, "closed_book": 20},
        "id_digests": {
            "prose": id_prose,
            "canonical_json": id_json,
            "closed_book": rn.digest_text(f"ids-closed-{case['fixture_id']}"),
        },
        "ids_identical": same_ids,
        "id_length_delta": -10,
        "cells": {
            "prose": prose_cell,
            "canonical_json": json_cell,
            "closed_book": screen_cell,
        },
        "cell_details": details or {"prose": "", "canonical_json": "", "closed_book": ""},
    }


def _pass_cell(case, tokens=100):
    return _cell(True, rn.digest_text(str(case["expected"])), True, tokens, str(case["expected"]))


def _screen_fail(case):
    return _cell(False, rn.digest_text("unknownword"), True, 20, headline=False)


class FrozenMaterialTests(unittest.TestCase):
    def test_words_fields_and_templates_frozen(self):
        self.assertEqual(len(rn.WORDS), 24)
        self.assertEqual(rn.WORDS[0], "amber")
        self.assertEqual(rn.WORDS[-1], "willow")
        self.assertEqual(rn.FIELDS_3, ("code", "label", "zone"))
        self.assertEqual(rn.FIELDS_4, ("code", "label", "mark", "zone"))
        self.assertEqual(rn.ANSWER_FIELD, "code")
        self.assertEqual(rn.KEY_FIELD, "zone")
        self.assertEqual(
            rn.QUESTION_TEMPLATE, "What is the code of the record whose zone is {key}?"
        )
        self.assertEqual(rn.CONTRACT, "Answer with ONLY the value.")
        self.assertEqual(rn.ARM_KEYS, ("prose", "canonical_json", "closed_book"))
        self.assertEqual(rn.HEADLINE_ARMS, ("prose", "canonical_json"))
        self.assertEqual(rn.CASE_COUNT, 24)
        self.assertEqual(rn.CODE_SPACE, 216000)
        self.assertEqual(rn.GENERATION_BUDGET, 1728)
        self.assertEqual(rn.HEADLINE_GENERATIONS, 1152)
        self.assertEqual(rn.SCREEN_GENERATIONS, 576)
        self.assertEqual(rn.MAX_GROUND_RECORDS, 576)
        self.assertEqual(rn.MAX_RECORDS_BYTES, 12 * 1024 * 1024)

    def test_table_shapes_cycle(self):
        shapes = [rn.table_shape(i) for i in range(24)]
        self.assertEqual([n for n, _ in shapes], [4, 5, 6] * 8)
        self.assertEqual([len(fields) for _, fields in shapes], [3, 4, 3, 4, 3, 4] * 4)

    def test_values_match_grammar_and_are_distinct(self):
        pattern = re.compile(r"^[a-z]+-[0-9]{4}$")
        for case in rn.generate_cases(3):
            values = [record[field] for record in case["table"] for field in sorted(record)]
            self.assertEqual(len(values), len(set(values)))
            for value in values:
                self.assertRegex(value, pattern)
                self.assertIn(value.split("-")[0], rn.WORDS)
                self.assertTrue(1000 <= int(value.split("-")[1]) <= 9999)

    def test_generation_is_deterministic_and_seed_sensitive(self):
        self.assertEqual(rn.generate_cases(5), rn.generate_cases(5))
        self.assertNotEqual(rn.generate_cases(5), rn.generate_cases(6))
        ids = [case["fixture_id"] for case in rn.generate_cases(5)]
        self.assertEqual(ids, [f"record-{i:02d}" for i in range(24)])

    def test_rng_call_sequence_is_the_frozen_one(self):
        # Amendment A: the seed-to-table mapping rests on this
        # exact call sequence -- per value draw the word first,
        # then the number; duplicates consumed and discarded;
        # row-major fill over alphabetical fields; the queried
        # index is the next draw after the table is complete.
        # Re-executed literally here and compared table for
        # table against the generator.
        import random

        for seed in (0, 1, 7):
            expected_cases = []
            for index in range(24):
                n_records, fields = rn.table_shape(index)
                rng = random.Random(seed * 1000 + index)  # noqa: S311 # nosec B311
                values = []
                seen = set()
                while len(values) < n_records * len(fields):
                    word = rn.WORDS[rng.randrange(24)]
                    number = rng.randrange(1000, 10000)
                    value = f"{word}-{number}"
                    if value in seen:
                        continue
                    seen.add(value)
                    values.append(value)
                table = [
                    {field: values[r * len(fields) + j] for j, field in enumerate(fields)}
                    for r in range(n_records)
                ]
                query = rng.randrange(n_records)
                expected_cases.append((table, query))
            for case, (table, query) in zip(rn.generate_cases(seed), expected_cases, strict=True):
                self.assertEqual(case["table"], table)
                self.assertEqual(case["query_index"], query)
                self.assertEqual(case["key"], table[query]["zone"])
                self.assertEqual(case["expected"], table[query]["code"])
        self.assertIn("rng.randrange(24)", rn.GENERATOR_SPEC)
        self.assertIn("rng.randrange(1000, 10000)", rn.GENERATOR_SPEC)
        self.assertIn("rng.randrange(n_records)", rn.GENERATOR_SPEC)

    def test_worked_example_byte_identical(self):
        # The candidate's section 3.3 worked example, byte for
        # byte (four 3-field records).
        table = [
            {"code": "amber-1001", "label": "birch-2002", "zone": "cedar-3003"},
            {"code": "copper-4004", "label": "falcon-5005", "zone": "garnet-6006"},
            {"code": "harbor-7007", "label": "indigo-8008", "zone": "juniper-9009"},
            {"code": "lagoon-1010", "label": "magnet-2020", "zone": "nickel-3030"},
        ]
        prose = rn.render_prose(table)
        self.assertEqual(
            prose,
            "The code is amber-1001, the label is birch-2002, and the zone is "
            "cedar-3003. The code is copper-4004, the label is falcon-5005, and "
            "the zone is garnet-6006. The code is harbor-7007, the label is "
            "indigo-8008, and the zone is juniper-9009. The code is lagoon-1010, "
            "the label is magnet-2020, and the zone is nickel-3030.",
        )
        self.assertEqual(len(prose), 317)
        canonical = rn.render_json(table)
        self.assertEqual(
            canonical,
            '[{"code": "amber-1001", "label": "birch-2002", "zone": "cedar-3003"}, '
            '{"code": "copper-4004", "label": "falcon-5005", "zone": "garnet-6006"}, '
            '{"code": "harbor-7007", "label": "indigo-8008", "zone": "juniper-9009"}, '
            '{"code": "lagoon-1010", "label": "magnet-2020", "zone": "nickel-3030"}]',
        )
        self.assertEqual(len(canonical), 286)

    def test_worst_case_lengths_match_design(self):
        # Section 3.4: a 6-record 4-field table of longest
        # values measures prose 653 / canonical JSON 594.
        words = ["saffron", "juniper"]
        table = [
            {field: f"{words[(r + j) % 2]}-9999" for j, field in enumerate(rn.FIELDS_4)}
            for r in range(6)
        ]
        # Distinctness is a generator property, not a renderer
        # one; the length measurement uses the frozen shapes.
        self.assertEqual(len(rn.render_prose(table)), 653)
        self.assertEqual(len(rn.render_json(table)), 594)

    def test_parse_inverts_render_on_generated_panel(self):
        # Injectivity, executed: parse after render is the
        # identity on every generated table.
        for seed in (0, 1, 2, 3):
            for case in rn.generate_cases(seed):
                self.assertEqual(rn.parse_prose(rn.render_prose(case["table"])), case["table"])

    def test_answer_key_overlap_exists_and_is_not_redrawn(self):
        # Amendment B: the answer word equals the key word in
        # about 1/24 of cases by construction; the population
        # is NOT redrawn -- an overlapping case must be present
        # in the generated panel exactly as drawn.
        overlaps = 0
        total = 0
        for seed in range(20):
            for case in rn.generate_cases(seed):
                total += 1
                if case["expected"].split("-")[0] == case["key"].split("-")[0]:
                    overlaps += 1
        self.assertGreater(overlaps, 0)
        self.assertLess(overlaps, total // 4)
        audit = rn.audit_answer_embedding()
        self.assertEqual(audit["answer_key_same_word_rate"], 8999 / 215999)
        self.assertAlmostEqual(audit["answer_key_same_word_rate"], 0.0417, places=3)

    def test_embedding_audit_values(self):
        audit = rn.audit_answer_embedding()
        self.assertEqual(audit["template_violations"], 0)
        self.assertEqual(audit["key_word_prefix_cases"], 24)
        self.assertEqual(audit["key_word_prefix_of"], 24)
        protocol = rn.protocol_object(0, "fp")
        self.assertEqual(protocol["embedding_audit"], audit)

    def test_protocol_object_and_fingerprints(self):
        protocol = rn.protocol_object(0, "fp-a")
        self.assertEqual(protocol["seed"], 0)
        self.assertEqual(protocol["backend_fingerprint"], "fp-a")
        self.assertNotIn("fingerprint", protocol)
        full = dict(protocol)
        full["fingerprint"] = rn.protocol_fingerprint(0, "fp-a")
        self.assertEqual(full["fingerprint"], rn.protocol_fingerprint(0, "fp-a"))
        self.assertNotEqual(full["fingerprint"], rn.protocol_fingerprint(0, "fp-b"))
        self.assertEqual(rn.protocol_object(0, "fp-a")["ground_sha256"], rn.ground_sha256(0))
        self.assertNotEqual(rn.ground_sha256(0), rn.ground_sha256(1))
        self.assertEqual(
            rn.protocol_object(0, "fp-a")["case_material_sha256"],
            rn.case_material_sha256(),
        )
        self.assertEqual(protocol["generation_budget"], 1728)
        self.assertEqual(protocol["max_records_bytes"], 12 * 1024 * 1024)

    def test_validate_case_accepts_generated_and_rejects_tampering(self):
        for case in rn.generate_cases(2):
            self.assertTrue(rn.validate_case(case, 2))
        case = rn.generate_cases(2)[0]
        tampered = copy.deepcopy(case)
        tampered["expected"] = "amber-0000"
        self.assertFalse(rn.validate_case(tampered, 2))
        tampered = copy.deepcopy(case)
        tampered["table"][0]["code"] = "amber-0000"
        self.assertFalse(rn.validate_case(tampered, 2))
        tampered = copy.deepcopy(case)
        tampered["query_index"] = (case["query_index"] + 1) % len(case["table"])
        self.assertFalse(rn.validate_case(tampered, 2))
        self.assertFalse(rn.validate_case(case, 3))

    def test_records_bytes_budget_relationships(self):
        # Amendment G: the candidate's printed "4.0x margin"
        # was false. Measured on synthetic worst-case records
        # in the stage's record shape: ASCII answer texts of
        # the tree's 2,048-character cap give headroom strictly
        # below 4.0x; non-ASCII-stressed answer texts exceed
        # the cap outright (the stage then fails closed with
        # the frozen budget detail).
        base = _payload()["candidates"]["chatml"]["records"]

        def stressed(fill):
            records = []
            for record in base:
                clone = copy.deepcopy(record)
                for arm in rn.HEADLINE_ARMS:
                    cell = clone["cells"][arm]
                    cell["answer_text"] = fill * 2048
                    cell["answer_digest"] = rn.digest_text(cell["answer_text"])
                    cell["verdict"] = False
                records.append(clone)
            return {f"cand-{i:02d}": records for i in range(24)}

        ascii_bytes = rn.records_bytes(stressed("x"))
        self.assertLess(ascii_bytes, rn.MAX_RECORDS_BYTES)
        ratio = rn.MAX_RECORDS_BYTES / ascii_bytes
        self.assertGreater(ratio, 3.9)
        self.assertLess(ratio, 4.0)
        cjk_bytes = rn.records_bytes(stressed("\u754c"))
        self.assertGreater(cjk_bytes, rn.MAX_RECORDS_BYTES)


class ConstructionTests(unittest.TestCase):
    def test_arm_contents(self):
        case = rn.generate_cases(0)[0]
        question = rn.QUESTION_TEMPLATE.format(key=case["key"])
        screen = rn.arm_content(case, "closed_book")
        self.assertEqual(screen, f"{question} {rn.CONTRACT}")
        prose = rn.arm_content(case, "prose")
        self.assertTrue(prose.startswith("The code is "))
        self.assertTrue(prose.endswith(f"\n\n{question} {rn.CONTRACT}"))
        canonical = rn.arm_content(case, "canonical_json")
        self.assertTrue(canonical.startswith('[{"code": '))
        self.assertTrue(canonical.endswith(f"\n\n{question} {rn.CONTRACT}"))

    def test_content_digests_distinct_across_arms(self):
        case = rn.generate_cases(1)[5]
        digests = {rn.digest_text(rn.arm_content(case, arm)) for arm in rn.ARM_KEYS}
        self.assertEqual(len(digests), 3)


class DecompositionTests(unittest.TestCase):
    def test_decompose_cell_classes(self):
        expected_digest = rn.digest_text("amber-1001")
        passing = _cell(True, expected_digest, True, 100, "amber-1001")
        self.assertEqual(rn.decompose_cell(passing, expected_digest), "pass")
        boundary_only = _cell(False, expected_digest, False, 100, "amber-1001")
        self.assertEqual(rn.decompose_cell(boundary_only, expected_digest), "boundary_only")
        wrong = _cell(False, rn.digest_text("birch-2002"), True, 100, "birch-2002")
        self.assertEqual(rn.decompose_cell(wrong, expected_digest), "other_failure")
        empty = _cell(False, None, False, 100, "")
        self.assertEqual(rn.decompose_cell(empty, expected_digest), "other_failure")

    def test_decompose_pair_classes(self):
        digest_a = rn.digest_text("amber-1001")
        digest_b = rn.digest_text("birch-2002")
        same_digest = (
            _cell(False, digest_a, False, 100, "x"),
            _cell(True, digest_a, True, 90, "x"),
        )
        self.assertEqual(rn.decompose_pair(*same_digest), "boundary_only")
        both_boundary = (
            _cell(False, digest_a, True, 100, "x"),
            _cell(True, digest_b, True, 90, "x"),
        )
        self.assertEqual(rn.decompose_pair(*both_boundary), "exact_only")
        mixed = (
            _cell(False, digest_a, False, 100, "x"),
            _cell(True, digest_b, True, 90, "x"),
        )
        self.assertEqual(rn.decompose_pair(*mixed), "mixed")

    def test_echo_and_parsed_are_independent(self):
        # Amendment C: the candidate's section 4.4 claim that
        # the parsed diagnostic is a subset of the echo
        # diagnostic is FALSE, executed here in both
        # directions. A JSON answer whose value carries the
        # escaped hyphen (\u002d as six literal characters)
        # parses to the expected value while the raw text does
        # not contain it; a prose answer containing the value
        # echoes without parsing.
        expected = "amber-1001"
        escaped = '{"code": "amber\\u002d1001"}'
        self.assertNotIn(expected, escaped)
        self.assertTrue(rn._parsed_hit(escaped, expected))
        prose = "The code is amber-1001, the label is birch-2002."
        self.assertIn(expected, prose)
        self.assertFalse(rn._parsed_hit(prose, expected))
        self.assertFalse(rn._parsed_hit("not json", expected))
        self.assertFalse(rn._parsed_hit('{"label": "amber-1001"}', expected))
        self.assertTrue(rn._parsed_hit('{"code": "amber-1001"}', expected))


def _sign_p(i: int, j: int) -> Fraction:
    d = i + j
    if d == 0:
        return Fraction(1)
    m = min(i, j)
    tail = sum(Fraction(comb(d, t)) for t in range(m + 1)) / Fraction(2) ** d
    return min(Fraction(1), 2 * tail)


def _fire_rates(n, p_both, p_prose, p_json):
    """Exact multinomial enumeration of the rejected rules,
    following the candidate's oc_tables.py method."""
    p_neither = 1 - p_both - p_prose - p_json
    rules = {
        "rcf": Fraction(0),
        "rule_a": Fraction(0),
        "mcnemar": Fraction(0),
        "sign_bonf": Fraction(0),
    }
    for i in range(n + 1):
        for j in range(n + 1 - i):
            for k in range(n + 1 - i - j):
                m = n - i - j - k
                coef = comb(n, i) * comb(n - i, j) * comb(n - i - j, k)
                prob = coef * p_prose**i * p_json**j * p_both**k * p_neither**m
                if not prob:
                    continue
                d = i + j
                if n >= 6 and max(i, j) >= 2:
                    rules["rcf"] += prob
                if d >= 4:
                    rules["rule_a"] += prob
                if d >= 6:
                    p = _sign_p(i, j)
                    if p <= Fraction(1, 20):
                        rules["mcnemar"] += prob
                    if p <= Fraction(1, 480):
                        rules["sign_bonf"] += prob
    return rules


def _indep(pa, pb):
    return (pa * pb, pa * (1 - pb), (1 - pa) * pb)


def _paired(p, phi):
    disc = p * (1 - p) * (1 - phi)
    return (p * p + phi * p * (1 - p), disc, disc)


def _delta(delta):
    return ((1 - delta) * Fraction(9, 10), delta / 2, delta / 2)


def _mcnemar_power(p_high, p_low, n, alpha):
    """The same McNemar rule by conditional enumeration (the
    d >= 6 floor is redundant: a two-sided sign p <= 0.05 first
    occurs at d = 6). Used for the N = 96 panel, where the
    multinomial loop is needlessly wide."""
    a = p_high * (1 - p_low)
    b = (1 - p_high) * p_low
    pd = a + b
    pi = a / pd
    total = Fraction(0)
    for k in range(n + 1):
        pk = Fraction(comb(n, k)) * pd**k * (1 - pd) ** (n - k)
        if not pk:
            continue
        fire = sum(
            Fraction(comb(k, x)) * pi**x * (1 - pi) ** (k - x)
            for x in range(k + 1)
            if _sign_p(x, k - x) <= alpha
        )
        total += pk * fire
    return total


class OperatingCharacteristicsTests(unittest.TestCase):
    """Amendment E: every printed figure re-derived by exact
    enumeration at its stated marginal basis."""

    OC = rn.OPERATING_CHARACTERISTICS

    def assertFigure(self, key, derived):
        self.assertEqual(round(float(derived), 4), self.OC[key], key)

    def test_transplant_and_rule_a_worlds(self):
        competent = _fire_rates(24, *_indep(Fraction(9, 10), Fraction(9, 10)))
        self.assertFigure("transplant_fire_competent_independent", competent["rcf"])
        self.assertFigure("rule_a_fire_competent_independent", competent["rule_a"])
        self.assertFigure("mcnemar_fire_competent_independent", competent["mcnemar"])
        self.assertEqual(
            round(float(24 * competent["rcf"]), 2),
            round(self.OC["transplant_familywise_expected"], 2),
        )
        self.assertEqual(
            round(float(1 - (1 - competent["rcf"]) ** 24), 4),
            round(self.OC["transplant_familywise_any"], 4),
        )
        paired = _fire_rates(24, *_paired(Fraction(9, 10), Fraction(1, 2)))
        self.assertFigure("transplant_fire_competent_paired_phi05", paired["rcf"])
        self.assertFigure("rule_a_fire_competent_paired_phi05", paired["rule_a"])
        self.assertFigure("mcnemar_fire_competent_paired_phi05", paired["mcnemar"])
        delta = _fire_rates(24, *_delta(Fraction(1, 20)))
        self.assertFigure("transplant_fire_delta_005", delta["rcf"])
        self.assertFigure("rule_a_fire_delta_005", delta["rule_a"])
        self.assertEqual(
            round(float(24 * delta["rule_a"]), 2),
            round(self.OC["rule_a_familywise_expected"], 2),
        )
        self.assertEqual(
            round(float(1 - (1 - delta["rule_a"]) ** 24), 3),
            round(self.OC["rule_a_familywise_any"], 3),
        )

    def test_mcnemar_powers_carry_their_marginal_bases(self):
        # The 25-point column of the candidate's table runs on
        # marginals .85/.60 and the 20-point column on .90/.70;
        # at the table's own .90 base the 25-point power is
        # 0.4000. All three are stated, none is interchangeable.
        gap25 = _fire_rates(24, *_indep(Fraction(85, 100), Fraction(60, 100)))
        self.assertFigure("mcnemar_power_gap25_marginals_85_60", gap25["mcnemar"])
        self.assertFigure("sign_bonferroni_power_gap25_marginals_85_60", gap25["sign_bonf"])
        gap20 = _fire_rates(24, *_indep(Fraction(90, 100), Fraction(70, 100)))
        self.assertFigure("mcnemar_power_gap20_marginals_90_70", gap20["mcnemar"])
        self.assertFigure("sign_bonferroni_power_gap20_marginals_90_70", gap20["sign_bonf"])
        base90 = _fire_rates(24, *_indep(Fraction(90, 100), Fraction(65, 100)))
        self.assertFigure("mcnemar_power_gap25_base_90", base90["mcnemar"])
        # The two enumerations of the same rule agree.
        self.assertEqual(
            _mcnemar_power(Fraction(85, 100), Fraction(60, 100), 24, Fraction(1, 20)),
            gap25["mcnemar"],
        )

    def test_panel96_figures(self):
        alpha = Fraction(1, 480)
        self.assertFigure(
            "panel96_bonferroni_power_gap25_marginals_85_60",
            _mcnemar_power(Fraction(85, 100), Fraction(60, 100), 96, alpha),
        )
        self.assertFigure(
            "panel96_bonferroni_power_gap25_base_90",
            _mcnemar_power(Fraction(90, 100), Fraction(65, 100), 96, alpha),
        )
        self.assertFigure(
            "panel96_bonferroni_false_fire",
            _mcnemar_power(Fraction(90, 100), Fraction(90, 100), 96, alpha),
        )
        self.assertEqual(self.OC["panel96_generations"], 6912)
        self.assertGreater(
            self.OC["panel96_bonferroni_power_gap25_base_90"],
            self.OC["panel96_bonferroni_power_gap25_marginals_85_60"],
        )

    def test_winner_and_screen_figures(self):
        self.assertEqual(
            round(float(Fraction(95, 100) ** 24), 3), self.OC["winner_pass_probability_95"]
        )
        self.assertEqual(
            round(float(Fraction(90, 100) ** 24), 3), self.OC["winner_pass_probability_90"]
        )
        self.assertEqual(
            round(float(Fraction(80, 100) ** 24), 3), self.OC["winner_pass_probability_80"]
        )
        self.assertEqual(self.OC["screen_guess_probability"], 1 / 216000)
        self.assertEqual(self.OC["screen_expected_exclusions_per_candidate"], 24 / 216000)
        self.assertEqual(self.OC["screen_expected_exclusions_per_panel"], 576 / 216000)


class AggregateTests(unittest.TestCase):
    def setUp(self):
        self.cases = rn.generate_cases(0)

    def aggregate(self, records):
        return rn.aggregate_candidate(records, self.cases)

    def test_discordance_direction_decomposition_and_resolution(self):
        c = self.cases
        records = [
            _record(
                c[0],
                _pass_cell(c[0]),
                _cell(False, rn.digest_text("birch-2002"), True, 90, "birch-2002"),
                _screen_fail(c[0]),
            ),
            _record(
                c[1],
                _cell(False, rn.digest_text("birch-2002"), True, 100, "birch-2002"),
                _pass_cell(c[1], 90),
                _screen_fail(c[1]),
            ),
            _record(c[2], _pass_cell(c[2]), _pass_cell(c[2], 90), _screen_fail(c[2])),
            _record(
                c[3],
                _cell(False, rn.digest_text("birch-2002"), True, 100, "birch-2002"),
                _cell(False, rn.digest_text("cedar-3003"), True, 90, "cedar-3003"),
                _screen_fail(c[3]),
            ),
        ]
        agg = self.aggregate(records)
        head = agg["headline"]
        self.assertEqual(head["n_paired"], 4)
        self.assertEqual(head["discordant"], 2)
        self.assertEqual(head["n_prose_only"], 1)
        self.assertEqual(head["n_canonical_json_only"], 1)
        self.assertEqual(head["decomposition"], {"exact_only": 2, "boundary_only": 0, "mixed": 0})
        self.assertEqual(head["resolution_points"], 25.0)
        self.assertEqual(head["prose_rate"], 0.5)
        self.assertEqual(head["canonical_json_rate"], 0.5)
        self.assertEqual(agg["n_cases"], 4)
        self.assertEqual(agg["n_incomplete"], 0)

    def test_screened_and_ids_identical_excluded_from_headline(self):
        c = self.cases
        screened = _record(
            c[0],
            _pass_cell(c[0]),
            _pass_cell(c[0], 90),
            _cell(True, rn.digest_text(str(c[0]["expected"])), True, 20, headline=False),
        )
        same_ids = _record(
            c[1],
            _pass_cell(c[1]),
            _cell(False, rn.digest_text("birch-2002"), True, 90, "birch-2002"),
            _screen_fail(c[1]),
            same_ids=True,
        )
        live = _record(c[2], _pass_cell(c[2]), _pass_cell(c[2], 90), _screen_fail(c[2]))
        agg = self.aggregate([screened, same_ids, live])
        self.assertEqual(agg["audit"]["n_screened"], 1)
        self.assertEqual(agg["headline"]["n_paired"], 1)
        self.assertEqual(agg["headline"]["discordant"], 0)
        self.assertEqual(agg["ids"]["n_ids_identical"], 1)
        self.assertEqual(agg["ids"]["n_same_ids_verdict_flips"], 1)
        # The unscreened sensitivity relaxes only the screen:
        # it counts the screened pair, never the ids-identical
        # one.
        unscreened = agg["sensitivities"]["unscreened"]
        self.assertEqual(unscreened["n_paired"], 2)
        self.assertEqual(unscreened["discordant"], 0)

    def test_resolution_none_when_no_pairs(self):
        c = self.cases
        screened = _record(
            c[0],
            _pass_cell(c[0]),
            _pass_cell(c[0], 90),
            _cell(True, rn.digest_text(str(c[0]["expected"])), True, 20, headline=False),
        )
        agg = self.aggregate([screened])
        head = agg["headline"]
        self.assertEqual(head["n_paired"], 0)
        self.assertIsNone(head["resolution_points"])
        self.assertIsNone(head["prose_rate"])
        self.assertIsNone(head["canonical_json_rate"])

    def test_echo_and_parsed_rates_over_failures_only(self):
        c = self.cases
        expected = str(c[0]["expected"])
        echo_fail = _record(
            c[0],
            _cell(False, rn.digest_text("prose text"), True, 100, f"The code is {expected} today"),
            _pass_cell(c[0], 90),
            _screen_fail(c[0]),
        )
        parsed_fail = _record(
            c[1],
            _pass_cell(c[1]),
            _cell(
                False,
                rn.digest_text("json text"),
                True,
                90,
                '{"code": "' + str(c[1]["expected"]) + '"}',
            ),
            _screen_fail(c[1]),
        )
        passing = _record(c[2], _pass_cell(c[2]), _pass_cell(c[2], 90), _screen_fail(c[2]))
        agg = self.aggregate([echo_fail, parsed_fail, passing])
        prose = agg["arms"]["prose"]
        self.assertEqual(prose["n_failures"], 1)
        self.assertEqual(prose["echo_rate"], 1.0)
        self.assertEqual(prose["parsed_rate"], 0.0)
        canonical = agg["arms"]["canonical_json"]
        self.assertEqual(canonical["n_failures"], 1)
        self.assertEqual(canonical["parsed_rate"], 1.0)
        self.assertEqual(canonical["echo_rate"], 1.0)
        screen = agg["arms"]["closed_book"]
        self.assertIsNone(screen["echo_rate"])
        self.assertIsNone(screen["parsed_rate"])

    def test_regrade_sensitivities(self):
        c = self.cases
        expected = str(c[0]["expected"])
        fenced = _record(
            c[0],
            _cell(False, rn.digest_text("fence"), True, 100, f"```\n{expected}\n```"),
            _pass_cell(c[0], 90),
            _screen_fail(c[0]),
        )
        agg = self.aggregate([fenced])
        self.assertEqual(agg["headline"]["discordant"], 1)
        self.assertEqual(agg["sensitivities"]["regrade_fence_discordant"], 0)
        self.assertEqual(agg["sensitivities"]["regrade_r1_discordant"], 1)

    def test_arm_attrition_counts(self):
        c = self.cases
        details = {"prose": "arm_context_exceeded", "canonical_json": "", "closed_book": ""}
        record = _record(c[0], None, _pass_cell(c[0], 90), _screen_fail(c[0]), details=details)
        agg = self.aggregate([record])
        prose = agg["arms"]["prose"]
        self.assertEqual(prose["n_context_exceeded"], 1)
        self.assertEqual(prose["n_completed"], 0)
        self.assertIsNone(prose["rate"])
        self.assertEqual(agg["n_incomplete"], 1)
        self.assertEqual(agg["headline"]["n_paired"], 0)


class StageTests(unittest.TestCase):
    def test_perfect_payload(self):
        payload = _payload()
        self.assertTrue(payload["available"])
        self.assertEqual(payload["detail"], "")
        self.assertEqual(payload["n_candidates"], 2)
        self.assertEqual(payload["n_cases"], 24)
        entry = payload["candidates"]["chatml"]
        self.assertEqual(entry["headline"]["discordant"], 0)
        self.assertEqual(entry["headline"]["n_paired"], 24)
        self.assertEqual(entry["headline"]["prose_rate"], 1.0)
        self.assertEqual(entry["headline"]["canonical_json_rate"], 1.0)
        self.assertEqual(entry["headline"]["resolution_points"], 100.0 / 24)
        self.assertEqual(entry["audit"]["n_screened"], 0)
        self.assertEqual(entry["ids"]["n_ids_identical"], 0)
        protocol = payload["protocol"]
        self.assertEqual(protocol["generation_budget"], 1728)
        self.assertEqual(
            protocol["fingerprint"], rn.protocol_fingerprint(0, _WordBackend.fingerprint)
        )

    def test_headline_cells_carry_text_screen_cells_do_not(self):
        payload = _payload()
        record = payload["candidates"]["chatml"]["records"][0]
        prose = record["cells"]["prose"]
        self.assertIn("answer_text", prose)
        self.assertEqual(rn.digest_text(prose["answer_text"]), prose["answer_digest"])
        self.assertTrue(prose["verdict"])
        screen = record["cells"]["closed_book"]
        self.assertNotIn("answer_text", screen)
        self.assertEqual(
            set(record),
            {
                "fixture",
                "n_records",
                "n_fields",
                "expected_digest",
                "content_digests",
                "arm_tokens",
                "id_digests",
                "ids_identical",
                "id_length_delta",
                "cells",
                "cell_details",
            },
        )
        self.assertEqual(
            record["id_length_delta"],
            record["arm_tokens"]["canonical_json"] - record["arm_tokens"]["prose"],
        )
        self.assertEqual(prose["input_tokens"], record["arm_tokens"]["prose"])

    def test_json_drop_payload(self):
        payload = _payload(_WordBackend("json_drop"))
        entry = payload["candidates"]["chatml"]
        head = entry["headline"]
        self.assertEqual(head["discordant"], 24)
        self.assertEqual(head["n_prose_only"], 24)
        self.assertEqual(head["n_canonical_json_only"], 0)
        self.assertEqual(head["decomposition"]["exact_only"], 24)
        self.assertEqual(head["canonical_json_rate"], 0.0)

    def test_closed_book_leak_screens_exactly_one(self):
        payload = _payload(_WordBackend("closed_book_leak"))
        entry = payload["candidates"]["chatml"]
        self.assertEqual(entry["audit"]["n_screened"], 1)
        self.assertEqual(entry["headline"]["n_paired"], 23)
        self.assertEqual(entry["headline"]["resolution_points"], 100.0 / 23)
        self.assertEqual(entry["sensitivities"]["unscreened"]["n_paired"], 24)

    def test_no_boundary_payload(self):
        payload = _payload(_WordBackend("no_boundary"))
        entry = payload["candidates"]["chatml"]
        self.assertEqual(entry["headline"]["discordant"], 0)
        self.assertEqual(
            entry["arms"]["prose"]["decomposition"],
            {"pass": 0, "boundary_only": 0, "other_failure": 24},
        )
        self.assertEqual(entry["arms"]["prose"]["rate"], 0.0)

    def test_exploding_backend_yields_evaluation_failed_cells(self):
        payload = _payload(_WordBackend("explode"))
        self.assertTrue(payload["available"])
        entry = payload["candidates"]["chatml"]
        self.assertEqual(entry["arms"]["prose"]["n_evaluation_failed"], 24)
        self.assertEqual(entry["arms"]["prose"]["n_completed"], 0)
        self.assertEqual(entry["headline"]["n_paired"], 0)
        self.assertEqual(entry["n_incomplete"], 24)

    def test_context_exceeded_cells(self):
        payload = _payload(max_context=1)
        entry = payload["candidates"]["chatml"]
        self.assertEqual(entry["arms"]["prose"]["n_context_exceeded"], 24)
        self.assertEqual(entry["arms"]["closed_book"]["n_context_exceeded"], 24)
        self.assertEqual(entry["headline"]["n_paired"], 0)

    def test_candidate_budget_detail(self):
        formats = tuple(Format(f"f{i:02d}", "", "u", "a", "e") for i in range(25))
        payload = _payload(formats=formats)
        self.assertFalse(payload["available"])
        self.assertEqual(payload["detail"], "record_notation_record_budget_exceeded")

    def test_cancelled_stage(self):
        event = threading.Event()
        event.set()
        payload = _payload(cancel=event)
        self.assertFalse(payload["available"])
        self.assertEqual(payload["detail"], "cancelled")

    def test_boundary_containment_screen_fails_closed(self):
        backend = _WordBackend()
        stage = experiments.record_notation_stage(
            backend,
            (_CHATML,),
            boundary=tuple(backend.encode("the")),
            seed=0,
            max_context=8192,
        )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "record_notation_screen_failed")

    def test_case_invariant_screen_fails_closed(self):
        with mock.patch.object(rn, "validate_case", return_value=False):
            payload = _payload()
        self.assertFalse(payload["available"])
        self.assertEqual(payload["detail"], "record_notation_screen_failed")
        self.assertEqual(payload["audit"]["by_reason"], {"case_invariant": 24})

    def test_special_token_screen_fails_closed(self):
        with mock.patch(
            "model_protocol_lab.fixtures.audit_special_token_literals",
            return_value=["<|im_end|>"],
        ):
            payload = _payload()
        self.assertFalse(payload["available"])
        self.assertEqual(payload["detail"], "record_notation_screen_failed")
        self.assertEqual(payload["audit"]["special_token_findings"], 1)
        self.assertEqual(payload["audit"]["by_reason"], {"special_token_literal": 1})


class ValidatorTests(unittest.TestCase):
    def setUp(self):
        self.payload = _payload()

    def test_real_payload_validates(self):
        self.assertTrue(diffreports._valid_record_notation(self.payload))
        diff = diffreports.diff_record_notation(self.payload, _payload())
        self.assertEqual(diff["available"], "both")
        self.assertFalse(diff["changed"])
        self.assertTrue(diff["comparable"])

    def _tampered(self, mutate):
        payload = copy.deepcopy(self.payload)
        mutate(payload)
        return payload

    def test_tamper_battery(self):
        def protocol_fingerprint(payload):
            payload["protocol"]["fingerprint"] = "0" * 64

        def ground_digest(payload):
            payload["protocol"]["ground_sha256"] = "0" * 64

        def content_digest(payload):
            payload["candidates"]["chatml"]["records"][0]["content_digests"]["prose"] = "0" * 64

        def arm_tokens(payload):
            payload["candidates"]["chatml"]["records"][0]["arm_tokens"]["prose"] += 1

        def cell_input_tokens(payload):
            # Amendment D: the cell's input_tokens are bound to
            # the record's arm token count; a payload whose
            # token fields were fabricated fails the mirror.
            payload["candidates"]["chatml"]["records"][0]["cells"]["prose"]["input_tokens"] += 1

        def id_length_delta(payload):
            payload["candidates"]["chatml"]["records"][0]["id_length_delta"] += 1

        def ids_identical_flip(payload):
            payload["candidates"]["chatml"]["records"][0]["ids_identical"] = True

        def answer_text_swap(payload):
            payload["candidates"]["chatml"]["records"][0]["cells"]["prose"]["answer_text"] = (
                "tampered-0000"
            )

        def screen_cell_with_text(payload):
            payload["candidates"]["chatml"]["records"][0]["cells"]["closed_book"]["answer_text"] = (
                "tampered-0000"
            )

        def verdict_flip(payload):
            payload["candidates"]["chatml"]["records"][0]["cells"]["prose"]["verdict"] = False

        def aggregate_discordant(payload):
            payload["candidates"]["chatml"]["headline"]["discordant"] += 1

        def candidate_audit_screened(payload):
            payload["candidates"]["chatml"]["audit"]["n_screened"] += 1

        def records_truncated(payload):
            payload["candidates"]["chatml"]["records"] = payload["candidates"]["chatml"]["records"][
                :23
            ]

        def extra_top_level_key(payload):
            payload["surprise"] = True

        def n_cases_tamper(payload):
            payload["n_cases"] = 23

        def expected_digest(payload):
            payload["candidates"]["chatml"]["records"][0]["expected_digest"] = "0" * 64

        for mutate in (
            protocol_fingerprint,
            ground_digest,
            content_digest,
            arm_tokens,
            cell_input_tokens,
            id_length_delta,
            ids_identical_flip,
            answer_text_swap,
            screen_cell_with_text,
            verdict_flip,
            aggregate_discordant,
            candidate_audit_screened,
            records_truncated,
            extra_top_level_key,
            n_cases_tamper,
            expected_digest,
        ):
            payload = self._tampered(mutate)
            self.assertFalse(diffreports._valid_record_notation(payload), mutate.__name__)
            diff = diffreports.diff_record_notation(payload, _payload())
            self.assertEqual(diff["available"], "new", mutate.__name__)

    def test_boundary_only_cell_validates_in_a_real_record_set(self):
        # A boundary_only headline cell in an otherwise real
        # payload: rebuild the aggregates through the producer
        # and the mirror must accept (the pair decomposition
        # reads it as boundary_only, never as a value failure).
        payload = _payload()
        entry = payload["candidates"]["chatml"]
        record = entry["records"][0]
        cell = record["cells"]["prose"]
        cell["verdict"] = False
        cell["boundary_found"] = False
        aggregates = rn.aggregate_candidate(entry["records"], rn.generate_cases(0))
        aggregates.update({"status": "available", "detail": "", "records": entry["records"]})
        payload["candidates"]["chatml"] = aggregates
        self.assertTrue(diffreports._valid_record_notation(payload))
        self.assertEqual(aggregates["arms"]["prose"]["decomposition"]["boundary_only"], 1)
        self.assertEqual(aggregates["headline"]["decomposition"]["boundary_only"], 1)
        self.assertEqual(aggregates["headline"]["discordant"], 1)

    def test_mirror_regrades_match_scorerlens(self):
        samples = [
            "amber-1001",
            "  amber-1001 \n",
            "a\r\nb\rc",
            "x\n\n\n\ny",
            "```\namber-1001\n```",
            "```python\namber-1001\n```",
            "``\namber-1001\n``",
            "amber-1001\n```",
        ]
        for text in samples:
            self.assertEqual(
                diffreports._recordnotation_scorer_r1(text), scorerlens._norm_r1(text), text
            )
            self.assertEqual(
                diffreports._recordnotation_scorer_fence(text),
                scorerlens._fence_strip(text),
                text,
            )

    def test_diff_drift_and_comparability(self):
        dropped = _payload(_WordBackend("json_drop"))
        diff = diffreports.diff_record_notation(self.payload, dropped)
        self.assertTrue(diff["changed"])
        self.assertTrue(diff["comparable"])
        metrics = {entry["metric"] for entry in diff["candidate_drift"]}
        self.assertIn("discordant", metrics)
        self.assertIn("n_prose_only", metrics)
        other_seed = _payload(seed=1)
        diff = diffreports.diff_record_notation(self.payload, other_seed)
        self.assertTrue(diff["protocol_drift"])
        self.assertTrue(diff["generator_drift"])
        self.assertFalse(diff["comparable"])
        self.assertTrue(diff["changed"])

    def test_diff_availability(self):
        unavailable = {"available": False, "detail": "record_notation_unavailable"}
        diff = diffreports.diff_record_notation(unavailable, self.payload)
        self.assertEqual(diff["available"], "new")
        self.assertFalse(diff["changed"])
        diff = diffreports.diff_record_notation(self.payload, unavailable)
        self.assertEqual(diff["available"], "old")
        diff = diffreports.diff_record_notation(unavailable, unavailable)
        self.assertEqual(diff["available"], "neither")
        single = _payload(formats=(_CHATML,))
        diff = diffreports.diff_record_notation(single, self.payload)
        self.assertEqual(diff["candidates_added"], ["llama3"])
        self.assertTrue(diff["changed"])


class ReportingTests(unittest.TestCase):
    def test_section_empty_without_payload(self):
        self.assertEqual(reporting._record_notation_section({}), "")
        self.assertEqual(
            reporting._record_notation_section(
                {"record_notation": {"available": False, "detail": "x"}}
            ),
            "",
        )

    def test_section_renders_readouts(self):
        section = reporting._record_notation_section({"record_notation": _payload()})
        self.assertIn('<section class="card" id="record-notation">', section)
        self.assertIn("Headline over 24 paired cases", section)
        self.assertIn("discordant 0, of which prose-only passes 0", section)
        # Amendment F: the resolution prints beside the count.
        self.assertIn("Resolution:", section)
        self.assertIn("4.2 points", section)
        # Amendment B: the instantiated-key overlap is printed.
        self.assertIn("4.17 percent", section)
        # Amendment C: neither diagnostic contains the other.
        self.assertIn("neither a subset of the other", section)
        # Amendment E: the marginal bases are stated.
        self.assertIn("marginals .85/.60", section)
        self.assertIn("0.833", section)
        # Researcher condition C: the scope phrase is printed.
        self.assertIn("for this checkpoint and this candidate's template family", section)
        self.assertIn("no winner", section)

    def test_section_escapes_candidate_names(self):
        fmt = Format("a<b", "", "u", "a", "e")
        payload = _payload(formats=(fmt,))
        section = reporting._record_notation_section({"record_notation": payload})
        self.assertIn("a&lt;b", section)
        self.assertNotIn("<td>a<b</td>", section)


class CliTests(unittest.TestCase):
    def test_flag_registered(self):
        parser = cli.build_parser()
        texts = []
        for action in parser._subparsers._group_actions[0].choices["run"]._actions:
            texts.extend(action.option_strings)
        self.assertIn("--record-notation", texts)

    def test_wrapper_fallback(self):
        result: dict = {}
        with mock.patch.object(
            experiments, "record_notation_stage", side_effect=RuntimeError("boom")
        ):
            cli._run_record_notation_probe(
                _WordBackend(),
                (_CHATML,),
                result,
                boundary=(1,),
                boundary_map=None,
                max_new_tokens=64,
                max_context=8192,
                seed=0,
                cancel=None,
            )
        self.assertEqual(
            result["record_notation"],
            {"available": False, "detail": "record_notation_unavailable"},
        )

    def test_wrapper_populates_result(self):
        result: dict = {}
        backend = _WordBackend()
        cli._run_record_notation_probe(
            backend,
            (_CHATML,),
            result,
            boundary=tuple(backend.encode("endmark")),
            boundary_map=None,
            max_new_tokens=64,
            max_context=8192,
            seed=0,
            cancel=None,
        )
        self.assertTrue(result["record_notation"]["available"])
        self.assertTrue(diffreports._valid_record_notation(result["record_notation"]))


class RenderParseGapTests(unittest.TestCase):
    """Gap pass: the prose renderer's 1- and 2-field sentence
    forms (the generated panel only ever has 3-4 fields), the
    inverse parser's three rejection raises, the unknown-arm
    raise, and the embedding audit's counting line (the frozen
    lists yield 0 by construction; a patched word list proves
    the zero is counted, not hardcoded)."""

    def test_one_and_two_field_sentences(self):
        one = [{"code": "amber-1001"}]
        self.assertEqual(rn.render_prose(one), "The code is amber-1001.")
        self.assertEqual(rn.parse_prose(rn.render_prose(one)), one)
        two = [{"code": "amber-1001", "label": "birch-2002"}]
        self.assertEqual(
            rn.render_prose(two), "The code is amber-1001 and the label is birch-2002."
        )
        self.assertEqual(rn.parse_prose(rn.render_prose(two)), two)

    def test_parse_prose_rejects_non_renderings(self):
        with self.assertRaises(ValueError):
            rn.parse_prose("The code is amber-1001")  # no final period
        with self.assertRaises(ValueError):
            rn.parse_prose("A code is amber-1001.")  # no 'The ' opening
        with self.assertRaises(ValueError):
            rn.parse_prose("The code amber-1001.")  # segment without ' is '

    def test_arm_content_unknown_arm(self):
        case = rn.generate_case(0, 0)
        with self.assertRaises(ValueError):
            rn.arm_content(case, "yaml")

    def test_embedding_audit_counts_collisions(self):
        with mock.patch.object(rn, "WORDS", ["code"]):
            audit = rn.audit_answer_embedding()
        self.assertGreaterEqual(audit["template_violations"], 1)
        self.assertEqual(rn.EMBEDDING_AUDIT["template_violations"], 0)


class ValidateCaseGapTests(unittest.TestCase):
    """Gap pass: validate_case's reject branches, one doctoring
    per branch. The regeneration-equality check gates the later
    branches, so the generator is patched to return the doctored
    case itself (equality then holds and the named branch is the
    one that fires). The key-uniqueness branch is NOT here: it
    is subsumed by the value-distinctness branch (a duplicated
    zone is a duplicated value) and is documented as
    provably unreachable in VALIDATION.md."""

    def _patched(self, case):
        with mock.patch.object(rn, "generate_case", return_value=case):
            return rn.validate_case(case, 0)

    def _case(self, index=0):
        return copy.deepcopy(rn.generate_case(0, index))

    def test_index_out_of_range_and_malformed(self):
        case = self._case()
        case["fixture_id"] = "record-24"
        self.assertFalse(rn.validate_case(case, 0))
        self.assertFalse(rn.validate_case({}, 0))
        self.assertFalse(rn.validate_case({"fixture_id": "record-0x"}, 0))

    def test_regeneration_mismatch(self):
        case = self._case()
        case["table"][0]["label"] = "zebra-1234"
        self.assertFalse(rn.validate_case(case, 0))

    def test_table_length(self):
        case = self._case()
        case["table"] = case["table"][:-1]
        self.assertFalse(self._patched(case))

    def test_field_set(self):
        case = self._case()
        case["table"][0]["zones"] = case["table"][0].pop("zone")
        self.assertFalse(self._patched(case))

    def test_duplicate_value(self):
        case = self._case()
        case["table"][1]["label"] = case["table"][0]["label"]
        self.assertFalse(self._patched(case))

    def test_value_shape(self):
        case = self._case()
        case["table"][0]["code"] = "amber-999"
        self.assertFalse(self._patched(case))

    def test_value_word_not_in_list(self):
        case = self._case()
        case["table"][0]["code"] = "zebra-1234"
        self.assertFalse(self._patched(case))

    def test_expected_absent(self):
        case = self._case()
        case["expected"] = case["table"][0]["label"]
        self.assertFalse(self._patched(case))

    def test_query_out_of_range(self):
        case = self._case()
        case["query_index"] = len(case["table"])
        self.assertFalse(self._patched(case))

    def test_query_binding(self):
        case = self._case()
        case["query_index"] = (case["query_index"] + 1) % len(case["table"])
        self.assertFalse(self._patched(case))

    def test_renderings_must_diverge(self):
        case = self._case()
        with (
            mock.patch.object(rn, "generate_case", return_value=case),
            mock.patch.object(rn, "render_json", side_effect=lambda table: rn.render_prose(table)),
        ):
            self.assertFalse(rn.validate_case(case, 0))

    def test_parse_inverse_must_hold(self):
        case = self._case()
        with (
            mock.patch.object(rn, "generate_case", return_value=case),
            mock.patch.object(rn, "parse_prose", return_value=[]),
        ):
            self.assertFalse(rn.validate_case(case, 0))

    def test_arm_content_bindings(self):
        real = rn.arm_content
        for bad_arm in ("prose", rn.SCREEN_ARM):
            case = self._case()

            def fake(case_, arm, _bad=bad_arm):
                return "junk" if arm == _bad else real(case_, arm)

            with (
                mock.patch.object(rn, "generate_case", return_value=case),
                mock.patch.object(rn, "arm_content", side_effect=fake),
            ):
                self.assertFalse(rn.validate_case(case, 0), bad_arm)


class AggregateGapTests(unittest.TestCase):
    """Gap pass: the aggregate's tolerance of malformed records
    (non-dict cells/details/digests maps), the non-integer token
    delta fallback, and the regrade's missing-answer-text path."""

    def test_malformed_records_are_inert(self):
        good = copy.deepcopy(_payload()["candidates"]["chatml"]["records"][0])
        del good["cells"]["prose"]["answer_text"]
        malformed = {
            "fixture": "record-01",
            "cells": [],
            "cell_details": [],
            "id_digests": [],
            "arm_tokens": {},
            "expected_digest": "x",
        }
        aggregate = rn.aggregate_candidate([malformed, good], rn.generate_cases(0))
        self.assertEqual(aggregate["ids"]["id_length_delta"][0], 0)
        self.assertEqual(aggregate["arms"]["prose"]["n_completed"], 1)
        self.assertIn("regrade_r1_discordant", aggregate["sensitivities"])


class CellCancelGapTests(unittest.TestCase):
    """Gap pass: the cell's cancel raises before and after the
    evaluation (the stage-level cancel test trips the stage's
    own check first)."""

    def _args(self):
        backend = _WordBackend()
        case = rn.generate_case(0, 0)
        content = rn.arm_content(case, "prose")
        fixture = Fixture(
            id=case["fixture_id"],
            messages=(Message("user", content),),
            expected=str(case["expected"]),
            group=rn.GROUP,
        )
        ids = tuple(backend.encode(_CHATML.render(fixture.messages)))
        return backend, fixture, ids

    def _call(self, backend, fixture, ids, event):
        return experiments._record_notation_cell(
            backend,
            _CHATML,
            fixture,
            ids,
            boundary_ids=tuple(backend.encode("endmark")),
            max_new_tokens=64,
            max_context=8192,
            seed=0,
            cancel=event,
            carry_text=True,
        )

    def test_cancel_before_evaluate(self):
        backend, fixture, ids = self._args()
        event = threading.Event()
        event.set()
        with self.assertRaises(experiments._RecordNotationCancelled):
            self._call(backend, fixture, ids, event)

    def test_cancel_after_evaluate(self):
        event = threading.Event()

        class CancellingBackend(_WordBackend):
            def generate(self, *args, **kwargs):
                event.set()
                return super().generate(*args, **kwargs)

        backend, fixture, ids = self._args()
        backend = CancellingBackend()
        with self.assertRaises(experiments._RecordNotationCancelled):
            self._call(backend, fixture, ids, event)


def _stage(backend=None, formats=(_CHATML, _LLAMA3), seed=0):
    backend = backend if backend is not None else _WordBackend()
    return experiments.record_notation_stage(
        backend,
        formats,
        boundary=tuple(backend.encode("endmark")),
        seed=seed,
        max_context=8192,
    )


def _unavailable_stage():
    stage = _stage()
    stage["candidates"] = {
        "gemma": {"status": "unavailable", "detail": "candidate_load_failed", "records": []}
    }
    return stage


class AnalysisGapTests(unittest.TestCase):
    """Gap pass: the analysis's unavailable-candidate branch,
    and the same entry carried through the mirror."""

    def test_analysis_marks_unavailable_candidate(self):
        stage = _stage()
        stage["candidates"]["gemma"] = {
            "status": "unavailable",
            "detail": "candidate_load_failed",
            "records": [],
        }
        analysis = experiments.record_notation_analysis(stage, ("chatml", "llama3", "gemma"))
        self.assertEqual(
            analysis["candidates"]["gemma"],
            {"status": "unavailable", "detail": "candidate_load_failed", "records": []},
        )
        self.assertTrue(diffreports._valid_record_notation(analysis))


class RecomputeGapTests(unittest.TestCase):
    """Gap pass: the mirror recompute's reject branches, one
    tamper per branch, asserted at the recompute itself. The
    total-records cap branch is NOT here: the per-candidate
    record count and the candidate cap bound the total first,
    and it is documented as provably unreachable in
    VALIDATION.md."""

    def setUp(self):
        self.payload = _payload()

    def _recomputed(self, mutate):
        payload = copy.deepcopy(self.payload)
        mutate(payload)
        return diffreports._recordnotation_recompute(payload)

    def test_protocol_branches(self):
        self.assertIsNone(self._recomputed(lambda p: p.update(protocol=[])))
        self.assertIsNone(self._recomputed(lambda p: p["protocol"].update(seed="0")))
        self.assertIsNone(self._recomputed(lambda p: p["protocol"].update(seed=1)))

    def test_candidate_set_branches(self):
        self.assertIsNone(self._recomputed(lambda p: p.update(candidates={})))
        many = {f"c{i}": {"status": "unavailable", "detail": "x", "records": []} for i in range(25)}
        self.assertIsNone(self._recomputed(lambda p: p.update(candidates=many)))
        self.assertIsNone(self._recomputed(lambda p: p["candidates"].update(chatml=[])))
        self.assertIsNone(
            self._recomputed(lambda p: p["candidates"]["chatml"].update(status="pending"))
        )
        self.assertIsNone(self._recomputed(lambda p: p["candidates"]["chatml"].update(detail="x")))
        self.assertIsNone(self._recomputed(lambda p: p["candidates"]["chatml"].update(extra=1)))
        self.assertIsNone(
            self._recomputed(
                lambda p: p["candidates"]["chatml"].update(
                    records=p["candidates"]["chatml"]["records"][:-1]
                )
            )
        )

    def test_unavailable_entry_branches(self):
        stage = _stage()
        stage["candidates"]["gemma"] = {
            "status": "unavailable",
            "detail": "candidate_load_failed",
            "records": [],
        }
        payload = experiments.record_notation_analysis(stage, ("chatml", "llama3", "gemma"))
        bad_records = copy.deepcopy(payload)
        bad_records["candidates"]["gemma"]["records"] = [{}]
        self.assertIsNone(diffreports._recordnotation_recompute(bad_records))
        bad_keys = copy.deepcopy(payload)
        bad_keys["candidates"]["gemma"]["extra"] = 1
        self.assertIsNone(diffreports._recordnotation_recompute(bad_keys))
        only = experiments.record_notation_analysis(_unavailable_stage(), ("gemma",))
        self.assertIsNone(diffreports._recordnotation_recompute(only))

    def test_records_bytes_cap(self):
        # The byte cap is enforced on the serialised records
        # before any per-record validation, so structurally
        # unexamined padding trips it.
        payload = copy.deepcopy(self.payload)
        payload["candidates"]["chatml"]["records"] = [{"pad": "x" * 600_000}] * 24
        self.assertIsNone(diffreports._recordnotation_recompute(payload))

    def test_record_branches(self):
        def rec(p):
            return p["candidates"]["chatml"]["records"][0]

        self.assertIsNone(self._recomputed(lambda p: rec(p).pop("cells")))
        self.assertIsNone(self._recomputed(lambda p: rec(p).update(fixture="record-99")))
        self.assertIsNone(
            self._recomputed(lambda p: rec(p).update(n_records=rec(p)["n_records"] + 1))
        )
        self.assertIsNone(self._recomputed(lambda p: rec(p).update(expected_digest="0" * 64)))
        self.assertIsNone(self._recomputed(lambda p: rec(p).update(cells={})))
        self.assertIsNone(
            self._recomputed(lambda p: rec(p).update(ids_identical=not rec(p)["ids_identical"]))
        )
        self.assertIsNone(self._recomputed(lambda p: rec(p)["arm_tokens"].update(prose=-1)))
        self.assertIsNone(self._recomputed(lambda p: rec(p)["id_digests"].update(prose="zz")))
        self.assertIsNone(
            self._recomputed(lambda p: rec(p)["content_digests"].update(prose="0" * 64))
        )
        self.assertIsNone(self._recomputed(lambda p: rec(p)["cell_details"].update(prose="bogus")))
        self.assertIsNone(self._recomputed(lambda p: rec(p)["cells"].update(prose=None)))
        self.assertIsNone(
            self._recomputed(lambda p: rec(p)["cell_details"].update(prose="arm_evaluation_failed"))
        )
        self.assertIsNone(
            self._recomputed(lambda p: rec(p).update(id_length_delta=rec(p)["id_length_delta"] + 1))
        )

    def test_cell_none_with_detail_continues(self):
        payload = copy.deepcopy(self.payload)
        record = payload["candidates"]["chatml"]["records"][0]
        record["cells"]["closed_book"] = None
        record["cell_details"]["closed_book"] = "arm_evaluation_failed"

        # The recompute walks past the screen cell (its continue
        # branch) and the rejection lands at the aggregate
        # comparison in the validity check.
        self.assertIsNotNone(diffreports._recordnotation_recompute(payload))
        self.assertFalse(diffreports._valid_record_notation(payload))

    def test_ids_identical_flip_is_recomputed(self):
        stage = _stage()
        record = stage["candidates"]["chatml"]["records"][0]
        cell = record["cells"]["canonical_json"]
        cell["answer_text"] = "bogus-0000"
        cell["answer_digest"] = rn.digest_text("bogus-0000")
        cell["verdict"] = False
        cell["input_tokens"] = record["arm_tokens"]["prose"]
        record["id_digests"]["canonical_json"] = record["id_digests"]["prose"]
        record["arm_tokens"]["canonical_json"] = record["arm_tokens"]["prose"]
        record["id_length_delta"] = 0
        record["ids_identical"] = True
        analysis = experiments.record_notation_analysis(stage, ("chatml", "llama3"))
        ids = analysis["candidates"]["chatml"]["ids"]
        self.assertEqual(ids["n_same_ids_verdict_flips"], 1)
        self.assertGreaterEqual(ids["n_ids_identical"], 1)
        self.assertTrue(diffreports._valid_record_notation(analysis))


class ReportingGapTests(unittest.TestCase):
    """Gap pass: the report section's guard returns (non-dict
    candidates, non-dict entries, unavailable rows, non-dict
    arms, None rates, and a missing operating-characteristics
    block)."""

    def test_candidates_not_a_dict(self):
        result = {"record_notation": {"available": True, "candidates": []}}
        self.assertEqual(reporting._record_notation_section(result), "")

    def test_unavailable_and_nondict_entries(self):
        result = {
            "record_notation": {
                "available": True,
                "candidates": {
                    "gemma": {"status": "unavailable", "detail": "x", "records": []},
                    "broken": None,
                },
                "operating_characteristics": {},
            }
        }
        section = reporting._record_notation_section(result)
        self.assertIn("gemma", section)
        self.assertIn("unavailable", section)

    def test_none_rates_and_missing_arm(self):
        payload = copy.deepcopy(_payload())
        entry = payload["candidates"]["chatml"]
        entry["arms"]["prose"] = None
        entry["arms"]["canonical_json"]["rate"] = None
        entry["headline"]["resolution_points"] = None
        section = reporting._record_notation_section({"record_notation": payload})
        self.assertIn("n/a", section)

    def test_missing_operating_characteristics(self):
        payload = copy.deepcopy(_payload())
        payload["operating_characteristics"] = {}
        section = reporting._record_notation_section({"record_notation": payload})
        self.assertIn("n/a", section)


_HAS_HF = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "transformers", "safetensors", "tokenizers")
)


@unittest.skipUnless(_HAS_HF, "the 'hf' extra is not installed")
class CliE2ETests(unittest.TestCase):
    """Gap pass: the run command's record-notation dispatch,
    driven end to end against the tiny locally generated GPT-2
    (the test_e2e harness shape, with the tree's 64-token
    context so headline cells resolve as context-exceeded and
    the screen cells generate)."""

    @classmethod
    def setUpClass(cls):
        from tests.test_adapter_integration import build_tiny_gpt2

        cls._tmp = TemporaryDirectory()
        cls.addClassCleanup(cls._tmp.cleanup)
        cls.model_dir = Path(cls._tmp.name) / "tiny"
        cls.model_dir.mkdir()
        build_tiny_gpt2(cls.model_dir)
        cls.dataset = Path(__file__).resolve().parent.parent / "examples" / "fixtures.json"

    def test_run_with_record_notation_flag(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        output = Path(tmp.name) / "report"
        with redirect_stdout(io.StringIO()):
            code = cli.main(
                [
                    "run",
                    "--model",
                    str(self.model_dir),
                    "--dataset",
                    str(self.dataset),
                    "--family",
                    "chatml",
                    "--boundary-ids",
                    "62",
                    "--max-new-tokens",
                    "8",
                    "--max-context",
                    "64",
                    "--device",
                    "cpu",
                    "--output",
                    str(output),
                    "--record-notation",
                ]
            )
        self.assertIn(code, (0, 2))
        results = json.loads((output / "results.json").read_text(encoding="utf-8"))
        self.assertTrue(results["record_notation"]["available"])
        self.assertTrue(diffreports._valid_record_notation(results["record_notation"]))


if __name__ == "__main__":
    unittest.main()
