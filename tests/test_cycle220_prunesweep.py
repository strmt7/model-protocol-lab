"""Cycle 220: prune-sweep probe (research wave-11 rank 4;
fresh review BUILD-as-amended, coordinator #62).

Spec citations are to docs/PRUNESWEEP_DESIGN.md (the
candidate text of record is research section 3, incorporated
as design Part I) and to the fresh review's amendments A-I
in the design doc's fresh-review adjudication section.

Every guard is proven engaged: the corrected operating
characteristics are re-derived by exact enumeration in this
file (amendment A -- the candidate's printed 0 / 40,000 null
row was false); boundary_only dose72 failures are shown NOT
to count toward A while boundary attrition counts them
(amendment B); the invariant battery is re-executed on the
printed lists (amendment C); the acrostic and negation
screens are exercised (amendment D); the drop key is pinned
to the eligible-list index (amendment E); the tail72 rival
reading prints exactly when the frozen rule says so
(amendment F); the two rejected shape gates are re-derived
by an exact DP with the graded interpolation printed
(amendment I).
"""

from __future__ import annotations

import copy
import re
import threading
import unittest
from fractions import Fraction
from math import comb
from unittest import mock

from model_protocol_lab import diffreports, experiments, reporting
from model_protocol_lab import prunesweep as ps
from model_protocol_lab.confidentiality import mcnemar_exact_p
from model_protocol_lab.core import Format, Generation

_CHATML = Format("chatml", "", "<|im_start|>user\n", "<|im_start|>assistant\n", "<|im_end|>\n")
_LLAMA3 = Format(
    "llama3",
    "<|begin_of_text|>",
    "<|start_header_id|>user<|end_header_id|>\n\n",
    "<|start_header_id|>assistant<|end_header_id|>\n\n",
    "<|eot_id|>",
)


class _WordBackend:
    """Word-token scripted backend.

    Modes: ``perfect`` (the expected answer with boundary on
    every passage cell; a non-value answer closed-book),
    ``tail72_drop`` (a different pool value at tail72 only),
    ``dose72_drop`` (a different pool value at dose72 only),
    ``both72_drop`` (a different pool value at dose72 and
    tail72), ``no_boundary`` (the expected answer with no
    boundary token anywhere), ``amber_closed`` (answers the
    prune-00 value closed-book, screening exactly that case),
    ``explode`` (generate raises).
    """

    fingerprint = "sha256:test-prune-sweep"

    def __init__(self, mode: str = "perfect") -> None:
        self.mode = mode
        self.calls = 0
        self.vocab: dict[str, int] = {}
        self.rev: dict[int, str] = {}

    def encode(self, text):
        """Encode text as word ids over the frozen test vocabulary."""
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
        """Decode word ids back to text, rendering unknown ids as '?'."""
        return " ".join(self.rev.get(int(i), "?") for i in ids)

    def generate(self, input_ids, *, max_new_tokens, seed=None, cancel=None):
        """Scripted generate: answer from the decoded prompt per the mode; 'explode' raises."""
        self.calls += 1
        if self.mode == "explode":
            raise RuntimeError("boom")
        text = self.decode(list(input_ids))
        match = re.search(r"code is (\w+)\.", text)
        value = match.group(1) if match else None
        answer = value if value is not None else "unknownword"
        boundary = self.encode("endmark")[0]
        filler = self.encode("fillerword")[0]
        pm = re.search(r"answer the question\.\s*(.*?)\s*What is the", text, re.S)
        passage = pm.group(1) if pm else ""
        passage_words = len(passage.split()) if passage else 0
        ends_with_fact = value is not None and passage.rstrip().endswith(f"code is {value}.")
        scattered72 = passage_words == 78 and not ends_with_fact
        tail72_shape = passage_words == 78 and ends_with_fact
        if (
            (self.mode == "dose72_drop" and scattered72)
            or (self.mode == "both72_drop" and passage_words == 78)
            or (self.mode == "tail72_drop" and tail72_shape)
        ):
            idx = ps.VALUES.index(value)
            answer = ps.VALUES[(idx + 1) % len(ps.VALUES)]
        if self.mode == "amber_closed" and value is None:
            answer = "amber"
        drop_boundary = self.mode == "no_boundary"
        out = list(self.encode(answer))
        if not drop_boundary:
            out.append(boundary)
        pad = filler if drop_boundary else boundary
        out = (out + [pad] * max_new_tokens)[:max_new_tokens]
        return Generation(token_ids=tuple(out), reason="limit")


def _payload(backend=None, formats=(_CHATML, _LLAMA3), seed=0, **stage_kwargs):
    """Run the prune-sweep stage over the scripted formats and return the payload."""
    backend = backend if backend is not None else _WordBackend()
    stage_kwargs.setdefault("max_context", 8192)
    stage = experiments.prune_sweep_stage(
        backend,
        formats,
        boundary=tuple(backend.encode("endmark")),
        seed=seed,
        **stage_kwargs,
    )
    return experiments.prune_sweep_analysis(stage, tuple(f.name for f in formats))


def _cell(verdict: bool, digest: str | None, boundary: bool = True, tokens: int = 40):
    """Build one record cell dict for the synthetic aggregate fixtures."""
    return {
        "verdict": verdict,
        "boundary_found": boundary,
        "boundary_text_match": None,
        "input_tokens": tokens,
        "answer_digest": digest,
    }


def _record(
    index: int,
    verdicts: dict[str, bool | None],
    *,
    dose72_digest: str = "expected",
    ids_identical: bool = False,
    tokens: dict[str, int] | None = None,
) -> dict:
    """One synthetic record for aggregate-level tests.

    ``verdicts`` maps arm -> True/False, or None for an
    incomplete cell. ``dose72_digest`` selects the dose72
    answer digest class: "expected", "other" (a different
    pool value), "junk" (a non-pool string), "none".
    """
    case = ps.generate_case(0, index)
    value = str(case["value"])
    expected_digest = ps.digest_text(value)
    digest_by_kind = {
        "expected": expected_digest,
        "other": ps.digest_text(ps.VALUES[(index + 1) % len(ps.VALUES)]),
        "junk": ps.digest_text("not-a-pool-value"),
        "none": None,
    }
    cells: dict[str, dict | None] = {}
    details: dict[str, str] = {}
    for arm in ps.ARM_KEYS:
        verdict = verdicts.get(arm, True)
        if verdict is None:
            cells[arm] = None
            details[arm] = "arm_evaluation_failed"
            continue
        details[arm] = ""
        digest = expected_digest if verdict else digest_by_kind["junk"]
        if arm == "dose72" and not verdict:
            digest = digest_by_kind[dose72_digest]
        if arm == "dose72" and verdict:
            digest = expected_digest
        boundary = True
        if not verdict and digest == expected_digest:
            boundary = False
        cells[arm] = _cell(bool(verdict), digest, boundary=boundary)
    if tokens is None:
        tokens = {arm: 200 - (ps.DOSE_BY_ARM.get(arm, 0)) for arm in ps.ARM_KEYS}
        tokens["tail72"] = 128
        tokens["closed_book"] = 20
    id_digests = {arm: ps.digest_text(f"ids-{index}-{arm}") for arm in ps.ARM_KEYS}
    if ids_identical:
        id_digests["dose72"] = id_digests["dose0"]
    return {
        "fixture": str(case["fixture_id"]),
        "topic": str(case["topic"]),
        "expected_digest": expected_digest,
        "arm_tokens": dict(tokens),
        "id_digests": id_digests,
        "ids_identical": ids_identical,
        "content_digests": {arm: ps.digest_text(f"content-{index}-{arm}") for arm in ps.ARM_KEYS},
        "dropped_shares": ps.dropped_set_shares(case, 0),
        "cells": cells,
        "cell_details": details,
    }


def _pass_verdicts(**overrides):
    """Per-arm verdict map for a synthetic record: all arms pass except closed_book, with overrides
    applied.
    """
    verdicts = dict.fromkeys(ps.ARM_KEYS, True)
    verdicts["closed_book"] = False
    verdicts.update(overrides)
    return verdicts


class FrozenMaterialTests(unittest.TestCase):
    """Design Part I sections 3.2-3.4 + amendments C/D/H:
    the frozen lists, texts, budgets, and vocabulary.
    """

    def test_vocabulary_is_frozen(self):
        """The decision, cell-detail and arm vocabularies equal the frozen tuples."""
        self.assertEqual(ps.DECISIONS, ("insufficient", "prune_drop_observed", "below_threshold"))
        self.assertEqual(
            ps.STAGE_DETAILS,
            ("cancelled", "prune_record_budget_exceeded", "prune_screen_failed"),
        )
        self.assertEqual(ps.CELL_DETAILS, ("", "arm_context_exceeded", "arm_evaluation_failed"))
        self.assertEqual(ps.GROUP, "prune")
        self.assertEqual(
            ps.ARM_KEYS, ("dose0", "dose18", "dose36", "dose72", "tail72", "closed_book")
        )
        self.assertEqual(ps.HEADLINE_ARMS, ("dose0", "dose18", "dose36", "dose72"))
        self.assertEqual(ps.DOSES, (0, 18, 36, 72))

    def test_counts_and_budgets(self):
        """Panel counts and budget constants equal the frozen design values."""
        self.assertEqual(ps.CASE_COUNT, 24)
        self.assertEqual(ps.ELIGIBLE_WORDS, 144)
        self.assertEqual(ps.MAX_GROUND_RECORDS, 576)
        self.assertEqual(ps.GENERATION_BUDGET, 3456)
        self.assertEqual(ps.HEADLINE_GENERATIONS, 2304)
        self.assertEqual(ps.MAX_RECORDS_BYTES, 12 * 1024 * 1024)
        self.assertEqual((ps.MIN_C, ps.MIN_A), (12, 5))

    def test_lists_are_frozen_and_disjoint(self):
        """The frozen word lists have their frozen sizes and are pairwise disjoint."""
        self.assertEqual(len(ps.TOPICS), 8)
        self.assertTrue(all(len(t.split(" ")) == 2 for t in ps.TOPICS))
        self.assertEqual(len(ps.VALUES), 24)
        self.assertEqual(len(set(ps.VALUES)), 24)
        pools = (ps.SUBJECTS, ps.OBJECTS, ps.PLACES, ps.TIMES)
        self.assertTrue(all(len(pool) == 12 for pool in pools))
        pool_words = {w for pool in pools for w in pool}
        topic_words = {w for t in ps.TOPICS for w in t.split(" ")}
        self.assertFalse(pool_words & set(ps.VALUES))
        self.assertFalse(pool_words & topic_words)
        self.assertFalse(pool_words & {"the", "code", "is"})
        # Amendment D: pools disjoint from instruction and
        # question vocabulary (topic words are the question's
        # by construction and are excluded on that side).
        instruction_words = set(ps.INSTRUCTION.lower().replace(".", "").split(" "))
        question_words = set(
            ps.QUESTION_TEMPLATE.lower().replace(".", "").replace("?", "").split(" ")
        ) - {"{topic}"}
        self.assertFalse(pool_words & instruction_words)
        self.assertFalse(pool_words & question_words)
        # No pool word or value carries a digit or is a
        # negation word; templates are exactly nine words.
        for word in pool_words | set(ps.VALUES):
            self.assertFalse(any(ch.isdigit() for ch in word), word)
            self.assertNotIn(word, ps.NEGATION_WORDS)
        for template in ps.TEMPLATES:
            filled = template.format(subject="s", object="o", place="p", time="t")
            self.assertEqual(len(filled.split(" ")), 9, template)

    def test_stem_screen_has_teeth_on_the_frozen_lists(self):
        # The value "quartz" and the topic word "quarry" share
        # the 4-char prefix "quar": the frozen TIMES pool must
        # not contain "quarter" (the build-time repair).
        """The stem screen has teeth on the frozen lists: near-miss time words and value stems are
        caught.
        """
        self.assertNotIn("quarter", ps.TIMES)
        for value in ps.VALUES:
            for pool in (ps.SUBJECTS, ps.OBJECTS, ps.PLACES, ps.TIMES):
                for word in pool:
                    self.assertNotEqual(word[:4], value[:4], (word, value))

    def test_protocol_object_and_fingerprint(self):
        """protocol_object binds the seed and backend fingerprint into the protocol dict."""
        protocol = ps.protocol_object(7, "fp-x")
        self.assertEqual(protocol["seed"], 7)
        self.assertEqual(protocol["backend_fingerprint"], "fp-x")
        self.assertEqual(protocol["ground_sha256"], ps.ground_sha256(7))
        self.assertEqual(protocol["case_material_sha256"], ps.case_material_sha256())
        fingerprint = ps.protocol_fingerprint(7, "fp-x")
        self.assertEqual(len(fingerprint), 64)
        self.assertNotEqual(fingerprint, ps.protocol_fingerprint(8, "fp-x"))
        self.assertNotEqual(ps.ground_sha256(7), ps.ground_sha256(8))

    def test_operating_characteristics_are_the_corrected_ones(self):
        # Amendment A: the candidate's printed null row
        # (0 / 40,000) was false; these are the exact-
        # enumeration values, re-derived in OperatingTests.
        """The shipped operating characteristics equal the corrected exact enumerations."""
        self.assertEqual(
            ps.OPERATING_CHARACTERISTICS,
            {
                "null_random_stated": 0.0121,
                "null_competent_stated": 0.0143,
                "null_jitter_stated": 0.000005,
                "null_competent_exchangeable_v09": 0.003,
                "null_competent_exchangeable_v05": 0.0004,
                "null_competent_exchangeable_v02": 0.000004,
                "power_q25": 0.3969,
                "power_q50": 0.9601,
                "power_q75": 0.9996,
                "familywise_expected_stated": 0.34,
                "familywise_any_stated": 0.29,
                "familywise_expected_exchangeable": 0.07,
                "familywise_any_exchangeable": 0.07,
                "shape_returns_cap_graded": 0.094,
                "shape_intermediate_support_threshold": 0.0,
            },
        )


class ConstructionTests(unittest.TestCase):
    """Design section 3.4 + amendments C/E: the generated
    panel and the frozen operator.
    """

    def test_panel_shape_and_topics(self):
        """generate_cases yields the 24 frozen fixture ids in order with the panel shape."""
        cases = ps.generate_cases(0)
        self.assertEqual(len(cases), 24)
        self.assertEqual([c["fixture_id"] for c in cases], [f"prune-{i:02d}" for i in range(24)])
        for i, case in enumerate(cases):
            self.assertEqual(case["topic"], ps.TOPICS[i % 8])
            self.assertEqual(case["value"], ps.VALUES[i])
            self.assertEqual(len(case["fact_sentence"].split(" ")), 6)

    def test_validate_case_battery_across_seeds(self):
        """Every generated case validates across the battery of seeds."""
        for seed in (0, 1, 7, 20261029):
            for case in ps.generate_cases(seed):
                self.assertTrue(ps.validate_case(case, seed), (seed, case["fixture_id"]))

    def test_validate_case_rejects_tampering(self):
        """validate_case rejects a case whose context sentences were tampered with."""
        case = ps.generate_case(0, 3)
        bad = copy.deepcopy(case)
        sentences = list(bad["context_sentences"])
        sentences[0] = sentences[1]
        bad["context_sentences"] = tuple(sentences)
        self.assertFalse(ps.validate_case(bad, 0))
        bad2 = copy.deepcopy(case)
        bad2["value"] = "quartz"
        self.assertFalse(ps.validate_case(bad2, 0))
        self.assertFalse(ps.validate_case(case, 1))
        self.assertFalse(ps.validate_case({"fixture_id": "prune-99"}, 0))

    def test_drop_order_key_is_the_eligible_index(self):
        # Amendment E: position = eligible-list index in
        # passage order; the key is sha256("seed:case:position").
        """drop_order is keyed by the seed/fixture hash over the eligible indices."""
        import hashlib

        order = ps.drop_order(5, "prune-03")
        expected = tuple(
            sorted(
                range(144),
                key=lambda p: hashlib.sha256(f"5:prune-03:{p}".encode()).hexdigest(),
            )
        )
        self.assertEqual(order, expected)
        self.assertEqual(sorted(order), list(range(144)))

    def test_nested_drops_and_tail_identity(self):
        """Dropped sets nest across doses and the tail72 set is the complement identity."""
        for seed in (0, 3):
            for case in ps.generate_cases(seed):
                cid = str(case["fixture_id"])
                d18 = ps.dropped_positions(cid, "dose18", seed)
                d36 = ps.dropped_positions(cid, "dose36", seed)
                d72 = ps.dropped_positions(cid, "dose72", seed)
                self.assertTrue(d18 < d36 < d72)
                self.assertEqual((len(d18), len(d36), len(d72)), (18, 36, 72))
                self.assertEqual(
                    ps.dropped_positions(cid, "tail72", seed), frozenset(range(72, 144))
                )
                self.assertEqual(ps.dropped_positions(cid, "dose0", seed), frozenset())

    def test_protected_parts_byte_identical_and_exact_doses(self):
        """Instruction, fact and question stay byte-identical across arms while doses drop exact
        word counts.
        """
        case = ps.generate_case(0, 0)
        base = ps.arm_content(case, "dose0", 0)
        lines = base.split("\n")
        self.assertEqual(lines[0], ps.INSTRUCTION)
        self.assertEqual(lines[2], ps.QUESTION_TEMPLATE.format(topic=case["topic"]))
        self.assertEqual(len(lines[1].split(" ")), 150)
        for arm, dose in (("dose18", 18), ("dose36", 36), ("dose72", 72), ("tail72", 72)):
            content = ps.arm_content(case, arm, 0)
            arm_lines = content.split("\n")
            self.assertEqual(arm_lines[0], lines[0])
            self.assertEqual(arm_lines[2], lines[2])
            self.assertIn(str(case["fact_sentence"]), arm_lines[1])
            self.assertEqual(150 - len(arm_lines[1].split(" ")), dose)
        closed = ps.arm_content(case, "closed_book", 0)
        self.assertEqual(closed, f"{ps.INSTRUCTION}\n{lines[2]}")

    def test_fact_position_and_tail_relocation(self):
        """The fact sentence sits at its frozen position in dose0 and relocates to the tail in
        tail72.
        """
        case = ps.generate_case(0, 0)
        passage0 = ps.arm_content(case, "dose0", 0).split("\n")[1]
        sentences0 = passage0.split(". ")
        self.assertTrue(sentences0[8].startswith("The harbor light code is"))
        tail = ps.arm_content(case, "tail72", 0).split("\n")[1]
        self.assertTrue(tail.endswith(str(case["fact_sentence"])))

    def test_tail72_overlap_figure_seed0(self):
        # Amendment C: recomputed on the build's own lists --
        # case 0 overlaps in exactly 32 words (both review
        # reconstructions agree), the panel mean is 35.92.
        """At seed 0 the tail72/dose72 overlap figures match the frozen values."""
        overlaps = []
        for case in ps.generate_cases(0):
            order = ps.drop_order(0, str(case["fixture_id"]))
            overlaps.append(len(set(order[:72]) & set(range(72, 144))))
        self.assertEqual(overlaps[0], 32)
        self.assertEqual(round(sum(overlaps) / len(overlaps), 2), 35.92)
        self.assertEqual((min(overlaps), max(overlaps)), (29, 41))

    def test_dropped_shares_are_zero_on_the_panel(self):
        """Dropped-share accounting is zero on the generated panel: no protected category is ever
        dropped.
        """
        for seed in (0, 1):
            for case in ps.generate_cases(seed):
                shares = ps.dropped_set_shares(case, seed)
                for arm in ("dose18", "dose36", "dose72", "tail72"):
                    self.assertEqual(shares[arm], {"numeral": 0.0, "negation": 0.0})


class ClassificationTests(unittest.TestCase):
    """Design section 5: the frozen decision rule and its
    worked McNemar values.
    """

    def test_mcnemar_worked_values(self):
        """The exact McNemar p-values match the worked values for (5,0), (6,0) and (7,0)."""
        self.assertEqual(mcnemar_exact_p(5, 0), 0.0625)
        self.assertEqual(mcnemar_exact_p(6, 0), 0.03125)
        self.assertEqual(mcnemar_exact_p(7, 0), 0.015625)
        self.assertEqual(mcnemar_exact_p(8, 1), 0.0390625)
        self.assertEqual(mcnemar_exact_p(9, 1), 0.021484375)
        self.assertEqual(mcnemar_exact_p(10, 2), 0.03857421875)
        self.assertIsNone(mcnemar_exact_p(0, 0))

    def test_decision_truth_table(self):
        """classify_decision reproduces the frozen truth table over completed, discordant and p
        inputs.
        """
        self.assertEqual(ps.classify_decision(11, 24, 0), "insufficient")
        self.assertEqual(ps.classify_decision(0, 0, 0), "insufficient")
        # (5, 0) does not fire: p = 0.0625 -- the A >= 5 floor
        # is the rule's text, the McNemar conjunct is binding.
        self.assertEqual(ps.classify_decision(12, 5, 0), "below_threshold")
        self.assertEqual(ps.classify_decision(12, 6, 0), "prune_drop_observed")
        self.assertEqual(ps.classify_decision(24, 6, 0), "prune_drop_observed")
        self.assertEqual(ps.classify_decision(24, 10, 2), "prune_drop_observed")
        self.assertEqual(ps.classify_decision(24, 8, 1), "prune_drop_observed")
        # Reverse discordants suppress: (6, 6) gives p = 1.0.
        self.assertEqual(ps.classify_decision(24, 6, 6), "below_threshold")
        self.assertEqual(ps.classify_decision(24, 6, 3), "below_threshold")
        self.assertEqual(ps.classify_decision(24, 4, 0), "below_threshold")
        self.assertEqual(ps.classify_decision(24, 0, 0), "below_threshold")

    def test_decisions_vocabulary(self):
        """classify_decision only ever returns the frozen decision vocabulary over the full input
        grid.
        """
        for c in range(0, 25):
            for a in range(0, 25):
                for b in range(0, 6):
                    self.assertIn(ps.classify_decision(c, a, b), ps.DECISIONS)


class DecompositionTests(unittest.TestCase):
    """Design section 3.5 + amendment B: the digest-only
    decomposition in its frozen order.
    """

    def test_frozen_order(self):
        """Answer decomposition classifies digests in the frozen order: expected value, other pool
        value, junk.
        """
        expected = ps.digest_text("amber")
        other = ps.digest_text("birch")
        junk = ps.digest_text("not-a-pool-value")
        self.assertEqual(ps.decompose_cell(_cell(True, other), expected), "exact_pass")
        self.assertEqual(
            ps.decompose_cell(_cell(False, expected, boundary=False), expected),
            "boundary_only",
        )
        self.assertEqual(ps.decompose_cell(_cell(False, other), expected), "wrong_value")
        self.assertEqual(ps.decompose_cell(_cell(False, junk), expected), "non_value")
        self.assertEqual(ps.decompose_cell(_cell(False, None), expected), "non_value")


def _binom_pmf(n: int, p: Fraction) -> list[Fraction]:
    """Exact binomial probability mass function as Fractions (operating-characteristics oracle)."""
    return [Fraction(comb(n, k)) * p**k * (1 - p) ** (n - k) for k in range(n + 1)]


def _fire_stated(n_cases: int, p0: float, q: float, r: float) -> Fraction:
    """Exact enumeration, stated conditional basis (pass0
    Bernoulli p0; fail72|pass0 = q; pass72|fail0 = r).
    """
    total = Fraction(0)
    pc = _binom_pmf(n_cases, Fraction(p0).limit_denominator(1000))
    for c in range(n_cases + 1):
        if pc[c] == 0 or c < 12:
            continue
        pa = _binom_pmf(c, Fraction(q).limit_denominator(1000))
        pb = _binom_pmf(n_cases - c, Fraction(r).limit_denominator(1000))
        for a in range(c + 1):
            if a < 5 or pa[a] == 0:
                continue
            for b in range(n_cases - c + 1):
                if pb[b] == 0:
                    continue
                p = mcnemar_exact_p(a, b)
                if p is not None and p < 0.05:
                    total += pc[c] * pa[a] * pb[b]
    return total


def _fire_exchangeable(v: float, marginal: float, n: int = 24) -> Fraction:
    """Exact enumeration, exchangeable basis: per-case joint
    (pass, pass) = m - v, (pass, fail) = (fail, pass) = v,
    (fail, fail) = 1 - m - v.
    """
    from math import factorial

    m = Fraction(marginal).limit_denominator(1000)
    vv = Fraction(v).limit_denominator(1000)
    u = m - vv
    w = 1 - m - vv
    total = Fraction(0)
    for a in range(n + 1):
        for b in range(n + 1 - a):
            for npp in range(n + 1 - a - b):
                nff = n - a - b - npp
                c = npp + a
                if c < 12 or a < 5:
                    continue
                p = mcnemar_exact_p(a, b)
                if p is None or p >= 0.05:
                    continue
                coef = Fraction(
                    factorial(n), factorial(a) * factorial(b) * factorial(npp) * factorial(nff)
                )
                total += coef * u**npp * vv ** (a + b) * w**nff
    return total


class OperatingTests(unittest.TestCase):
    """Amendment A + I: every printed operating figure is
    re-derived here by exact enumeration (no Monte Carlo).
    """

    def test_stated_basis_nulls_and_power(self):
        """The stated-basis null and power figures equal independent Fraction re-derivations."""
        oc = ps.OPERATING_CHARACTERISTICS
        self.assertEqual(round(float(_fire_stated(24, 0.5, 0.5, 0.5)), 4), oc["null_random_stated"])
        self.assertEqual(
            round(float(_fire_stated(24, 0.9, 0.1, 0.1)), 4), oc["null_competent_stated"]
        )
        self.assertEqual(
            round(float(_fire_stated(24, 0.97, 0.02, 0.02)), 6), oc["null_jitter_stated"]
        )
        self.assertEqual(round(float(_fire_stated(24, 0.9, 0.25, 0.1)), 4), oc["power_q25"])
        self.assertEqual(round(float(_fire_stated(24, 0.9, 0.5, 0.1)), 4), oc["power_q50"])
        self.assertEqual(round(float(_fire_stated(24, 0.9, 0.75, 0.1)), 4), oc["power_q75"])

    def test_exchangeable_basis(self):
        """The exchangeable-basis figures equal independent re-derivations."""
        oc = ps.OPERATING_CHARACTERISTICS
        self.assertEqual(
            round(float(_fire_exchangeable(0.09, 0.9)), 4), oc["null_competent_exchangeable_v09"]
        )
        self.assertEqual(
            round(float(_fire_exchangeable(0.05, 0.9)), 4), oc["null_competent_exchangeable_v05"]
        )
        self.assertEqual(
            round(float(_fire_exchangeable(0.02, 0.9)), 6), oc["null_competent_exchangeable_v02"]
        )
        # The random labeller is the same model on both
        # bases and must give the same number.
        self.assertEqual(
            float(_fire_exchangeable(0.25, 0.5)), float(_fire_stated(24, 0.5, 0.5, 0.5))
        )

    def test_familywise(self):
        """The family-wise expectations equal 24 times the per-decision stated fire rate, as
        printed.
        """
        oc = ps.OPERATING_CHARACTERISTICS
        stated = float(_fire_stated(24, 0.9, 0.1, 0.1))
        self.assertEqual(round(24 * stated, 2), oc["familywise_expected_stated"])
        self.assertEqual(round(1 - (1 - stated) ** 24, 2), oc["familywise_any_stated"])
        exch = float(_fire_exchangeable(0.09, 0.9))
        self.assertEqual(round(24 * exch, 2), oc["familywise_expected_exchangeable"])
        self.assertEqual(round(1 - (1 - exch) ** 24, 2), oc["familywise_any_exchangeable"])

    def test_screening_shrinks_the_population(self):
        # Stated basis, frozen floors, smaller N (the design's
        # printed sensitivity rows).
        """Screening shrinks the effective population: fire figures at n=14, 16 and 17 match the
        re-derived values.
        """
        self.assertEqual(round(float(_fire_stated(16, 0.9, 0.1, 0.1)), 4), 0.0017)
        self.assertLess(abs(float(_fire_stated(17, 0.5, 0.5, 0.5)) - 0.00715), 1e-6)
        self.assertEqual(round(float(_fire_stated(14, 0.5, 0.5, 0.5)), 4), 0.0019)

    def test_firing_triples_census(self):
        """The census of firing triples is exactly 492 and no firing triple has A <= B."""
        triples = []
        for c in range(12, 25):
            for a in range(5, c + 1):
                for b in range(0, 24 - c + 1):
                    p = mcnemar_exact_p(a, b)
                    if p is not None and p < 0.05:
                        triples.append((c, a, b))
        self.assertEqual(len(triples), 492)
        self.assertTrue(all(a > b for _, a, b in triples))
        self.assertEqual(min(a for _, a, _ in triples), 6)
        self.assertEqual(max(b for _, _, b in triples), 6)

    def test_shape_gates_rederived_with_printed_interpolation(self):
        # Amendment I: the graded interpolation is
        # q_d = q72 * d / 72 (printed in the design); the
        # returns-cap figure is re-derived by an exact DP over
        # per-case outcome categories.
        """The shape-gate power figures equal the dynamic-programming re-derivation with the printed
        interpolation.
        """

        def shape_dp(q18, q36, q72, p0=0.9, r=0.1):
            """Dynamic-programming oracle for the shape-gate power computation."""
            cats = [(0, 0, 1, 0, 0, (1 - p0) * r), (0, 0, 0, 0, 0, (1 - p0) * (1 - r))]
            for f18 in (0, 1):
                for f36 in (0, 1):
                    for f72 in (0, 1):
                        pr = p0
                        pr *= q18 if f18 else 1 - q18
                        pr *= q36 if f36 else 1 - q36
                        pr *= q72 if f72 else 1 - q72
                        rr = 1 if (f18 or f36) and not f72 else 0
                        cats.append((1, f72, 0, f36, rr, pr))
            states = {(0, 0, 0, 0, 0): 1.0}
            for _ in range(24):
                nxt: dict = {}
                for (c, a, b, a36, rr), prob in states.items():
                    for dc, da, db, da36, dr, cp in cats:
                        if cp == 0:
                            continue
                        key = (c + dc, a + da, b + db, a36 + da36, rr + dr)
                        nxt[key] = nxt.get(key, 0.0) + prob * cp
                states = nxt
            fire = cap = sup = 0.0
            for (c, a, b, a36, rr), prob in states.items():
                if c < 12 or a < 5:
                    continue
                p = mcnemar_exact_p(a, b)
                if p is None or p >= 0.05:
                    continue
                fire += prob
                if rr <= 1:
                    cap += prob
                if a36 >= 2:
                    sup += prob
            return fire, cap, sup

        base, cap, _sup = shape_dp(0.125, 0.25, 0.5)
        self.assertEqual(round(base, 4), ps.OPERATING_CHARACTERISTICS["power_q50"])
        self.assertEqual(round(cap, 3), ps.OPERATING_CHARACTERISTICS["shape_returns_cap_graded"])
        base_t, _cap_t, sup_t = shape_dp(0.0, 0.0, 0.5)
        self.assertEqual(round(base_t, 4), ps.OPERATING_CHARACTERISTICS["power_q50"])
        self.assertEqual(
            sup_t, ps.OPERATING_CHARACTERISTICS["shape_intermediate_support_threshold"]
        )

    def test_operator_constants(self):
        # Amendment F: hypergeometric fact shift at dose72.
        """The fact-shift operator constants equal their closed forms (mean 36.0 words; sd from the
        variance).
        """
        self.assertEqual(ps.FACT_SHIFT_MEAN_WORDS, 36.0)
        var = 72 * 0.5 * 0.5 * (72 / 143)
        self.assertEqual(round(var**0.5, 2), ps.FACT_SHIFT_SD_WORDS)
        intact = Fraction(comb(72, 9), comb(144, 9))
        broken = 16 * (1 - 2 * intact)
        self.assertEqual(round(float(broken), 2), ps.EXPECTED_BROKEN_SENTENCES_DOSE72)
        self.assertEqual(round(float(16 * intact), 2), ps.EXPECTED_REMOVED_SENTENCES_DOSE72)


class AggregateTests(unittest.TestCase):
    """Design sections 5/6 + amendments B/F: the aggregate
    over synthetic records (executed shapes, not assumed).
    """

    @staticmethod
    def _panel(overrides_by_index):
        """Build a synthetic 24-record panel, applying per-index cell overrides."""
        records = []
        for i in range(24):
            verdicts = _pass_verdicts()
            kwargs = {}
            for key, val in overrides_by_index.get(i, {}).items():
                if key == "dose72_digest":
                    kwargs["dose72_digest"] = val
                elif key == "ids_identical":
                    kwargs["ids_identical"] = val
                else:
                    verdicts[key] = val
            records.append(_record(i, verdicts, **kwargs))
        return records

    def test_perfect_panel_below_threshold(self):
        """A perfect panel aggregates to below_threshold with the full population retained."""
        agg = ps.aggregate_candidate(self._panel({}))
        head = agg["headline"]
        self.assertEqual(head["n_population"], 24)
        self.assertEqual(head["direct_passes"], 24)
        self.assertEqual(head["drop_cases"], 0)
        self.assertEqual(head["reverse"], 0)
        self.assertEqual(head["decision"], "below_threshold")
        self.assertEqual(agg["shape"]["first_flip_histogram"]["never"], 24)
        self.assertEqual(agg["arms"]["dose0"]["rate"], 1.0)
        self.assertEqual(agg["n_cases"], 24)
        self.assertEqual(agg["n_incomplete"], 0)

    def test_drop_panel_fires(self):
        """Six dose72 discordant drops fire prune_drop_observed."""
        overrides = {i: {"dose72": False, "dose72_digest": "other"} for i in range(6)}
        agg = ps.aggregate_candidate(self._panel(overrides))
        head = agg["headline"]
        self.assertEqual(head["drop_cases"], 6)
        self.assertEqual(head["decision"], "prune_drop_observed")
        self.assertEqual(head["a_by_first_flip"], {"dose18": 0, "dose36": 0, "dose72": 6})
        self.assertEqual(agg["shape"]["dose_pass_rates"]["dose72"], 18 / 24)

    def test_boundary_only_never_counts_toward_a(self):
        # Amendment B: eight boundary-only dose72 failures
        # move boundary attrition only; A stays 0.
        """Boundary-only dose72 failures never count toward A (amendment B)."""
        overrides = {i: {"dose72": False, "dose72_digest": "expected"} for i in range(8)}
        agg = ps.aggregate_candidate(self._panel(overrides))
        head = agg["headline"]
        self.assertEqual(head["drop_cases"], 0)
        self.assertEqual(head["boundary_attrition"], 8)
        self.assertEqual(head["decision"], "below_threshold")
        decomp = agg["arms"]["dose72"]["decomposition"]
        self.assertEqual(decomp["boundary_only"], 8)
        self.assertEqual(decomp["exact_pass"], 16)

    def test_mixed_drop_and_boundary(self):
        """A mixed panel counts only the real drops toward A, not the boundary-only failures."""
        overrides = {}
        for i in range(4):
            overrides[i] = {"dose72": False, "dose72_digest": "junk"}
        for i in range(4, 8):
            overrides[i] = {"dose72": False, "dose72_digest": "expected"}
        agg = ps.aggregate_candidate(self._panel(overrides))
        head = agg["headline"]
        self.assertEqual(head["drop_cases"], 4)
        self.assertEqual(head["boundary_attrition"], 4)
        self.assertEqual(head["decision"], "below_threshold")

    def test_screened_cases_leave_the_population(self):
        """Closed-book-screened cases leave the headline population."""
        overrides = {i: {"closed_book": True} for i in range(3)}
        agg = ps.aggregate_candidate(self._panel(overrides))
        head = agg["headline"]
        self.assertEqual(head["n_screened"], 3)
        self.assertEqual(head["n_population"], 21)
        self.assertEqual(head["direct_passes"], 21)

    def test_closed_book_incomplete_cases_stay_in(self):
        """Cases with an incomplete closed-book cell stay in the population."""
        overrides = {i: {"closed_book": None} for i in range(2)}
        agg = ps.aggregate_candidate(self._panel(overrides))
        head = agg["headline"]
        self.assertEqual(head["n_closed_book_incomplete"], 2)
        self.assertEqual(head["n_population"], 24)
        self.assertEqual(agg["n_incomplete"], 2)

    def test_ids_identical_cases_are_excluded(self):
        """Cases whose rendered ids are identical across arms are excluded from the population."""
        overrides = {
            0: {"dose72": False, "dose72_digest": "other", "ids_identical": True},
            1: {"ids_identical": True},
        }
        agg = ps.aggregate_candidate(self._panel(overrides))
        head = agg["headline"]
        self.assertEqual(head["n_ids_identical"], 2)
        self.assertEqual(head["n_population"], 22)
        # The ids-identical case's dose72 flip is a rendering
        # anomaly, not a drop: A does not see it.
        self.assertEqual(head["drop_cases"], 0)

    def test_headline_incomplete_cases_are_excluded(self):
        """Cases with an incomplete headline cell are excluded from the population."""
        overrides = {0: {"dose36": None}}
        agg = ps.aggregate_candidate(self._panel(overrides))
        head = agg["headline"]
        self.assertEqual(head["n_headline_incomplete"], 1)
        self.assertEqual(head["n_population"], 23)

    def test_reverse_discordants_suppress(self):
        """Reverse discordants (dose0 fails while dose72 passes) suppress the fire decision."""
        overrides = {i: {"dose72": False, "dose72_digest": "other"} for i in range(6)}
        for i in range(6, 9):
            overrides[i] = {"dose0": False, "dose72": True}
        agg = ps.aggregate_candidate(self._panel(overrides))
        head = agg["headline"]
        self.assertEqual(head["reverse"], 3)
        self.assertEqual(head["drop_cases"], 6)
        self.assertEqual(head["decision"], "below_threshold")

    def test_shape_readout_returns_and_first_flip(self):
        """The shape readout reports per-dose returns and the first flip dose."""
        overrides = {
            0: {"dose18": False},
            1: {"dose18": False, "dose36": False},
            2: {"dose36": False, "dose72": False, "dose72_digest": "other"},
        }
        agg = ps.aggregate_candidate(self._panel(overrides))
        shape = agg["shape"]
        self.assertEqual(shape["returns"], 2)
        self.assertEqual(
            shape["first_flip_histogram"],
            {"dose18": 2, "dose36": 1, "dose72": 0, "never": 21},
        )
        head = agg["headline"]
        self.assertEqual(head["a_by_first_flip"]["dose36"], 1)
        self.assertEqual(shape["dose_pass_rates"]["dose18"], 22 / 24)

    def test_tail_contrast_and_rival_reading(self):
        """The tail contrast computes the rival word-count reading alongside the headline."""
        overrides = {i: {"dose72": False, "dose72_digest": "other"} for i in range(6)}
        agg = ps.aggregate_candidate(self._panel(overrides))
        tail = agg["tail_contrast"]
        self.assertEqual(tail["n_compared"], 24)
        self.assertEqual((tail["dose72_fails"], tail["tail72_fails"]), (6, 0))
        self.assertEqual((tail["dose72_only"], tail["tail72_only"]), (6, 0))
        self.assertFalse(tail["rival_reading"])
        both = {i: {"dose72": False, "tail72": False, "dose72_digest": "other"} for i in range(6)}
        agg2 = ps.aggregate_candidate(self._panel(both))
        tail2 = agg2["tail_contrast"]
        self.assertEqual((tail2["dose72_only"], tail2["tail72_only"]), (0, 0))
        self.assertEqual(tail2["dose72_fails"], 6)
        self.assertTrue(tail2["rival_reading"])

    def test_ids_block_deltas_and_same_ids_flips(self):
        """The ids block reports token deltas and flips when a record's arm ids coincide."""
        record = _record(0, _pass_verdicts(dose36=False), ids_identical=False)
        record["id_digests"]["dose36"] = record["id_digests"]["dose0"]
        agg = ps.aggregate_candidate([record])
        block = agg["ids"]["dose36"]
        self.assertEqual(block["n_ids_identical"], 1)
        self.assertEqual(block["n_same_ids_flips"], 1)
        self.assertEqual(block["id_length_delta_by_case"], [-36])
        self.assertEqual(block["max_abs_id_delta"], 36)
        self.assertEqual(agg["ids"]["dose72"]["id_length_delta_by_case"], [-72])

    def test_insufficient_when_population_small(self):
        """When failures shrink the completed population below the floor, the decision is
        insufficient.
        """
        overrides = {i: {"dose0": False} for i in range(13)}
        agg = ps.aggregate_candidate(self._panel(overrides))
        self.assertEqual(agg["headline"]["direct_passes"], 11)
        self.assertEqual(agg["headline"]["decision"], "insufficient")


class StageTests(unittest.TestCase):
    """Design sections 3.4/3.5/10: stage behaviour end to
    end on scripted backends.
    """

    def test_perfect_backend(self):
        """A perfect backend yields an available payload with all arms passing."""
        backend = _WordBackend()
        payload = _payload(backend)
        self.assertTrue(payload["available"])
        self.assertEqual(payload["n_candidates"], 2)
        self.assertEqual(payload["n_cases"], 24)
        self.assertEqual(backend.calls, 2 * 24 * 6)
        head = payload["candidates"]["chatml"]["headline"]
        self.assertEqual(head["direct_passes"], 24)
        self.assertEqual(head["decision"], "below_threshold")
        self.assertEqual(payload["protocol"]["ground_sha256"], ps.ground_sha256(0))

    def test_dose72_drop_fires(self):
        """A dose72-only drop fires prune_drop_observed for both candidates."""
        payload = _payload(_WordBackend("dose72_drop"))
        for name in ("chatml", "llama3"):
            head = payload["candidates"][name]["headline"]
            self.assertEqual(head["drop_cases"], 24)
            self.assertEqual(head["decision"], "prune_drop_observed")
            tail = payload["candidates"][name]["tail_contrast"]
            self.assertEqual(tail["dose72_only"], 24)
            self.assertFalse(tail["rival_reading"])

    def test_both72_drop_prints_rival_reading(self):
        """When tail72 drops together with dose72, the rival word-count reading is produced."""
        payload = _payload(_WordBackend("both72_drop"))
        tail = payload["candidates"]["chatml"]["tail_contrast"]
        self.assertEqual(tail["dose72_fails"], 24)
        self.assertEqual(tail["tail72_fails"], 24)
        self.assertTrue(tail["rival_reading"])

    def test_closed_book_screen_excludes_case(self):
        """A closed-book pass on the fact value screens exactly one case out."""
        payload = _payload(_WordBackend("amber_closed"))
        head = payload["candidates"]["chatml"]["headline"]
        self.assertEqual(head["n_screened"], 1)
        self.assertEqual(head["n_population"], 23)

    def test_no_boundary_backend_is_insufficient(self):
        # No boundary token anywhere: evaluate takes the whole
        # padded output as the answer, so the digest is a
        # non-value and no case passes dose0 -- C = 0. (A true
        # boundary_only cell needs the answer to BE the whole
        # output; that class is exercised at the decompose /
        # aggregate / validator level, as in the sibling
        # suites.)
        """A backend that never emits the boundary leaves the decision insufficient."""
        payload = _payload(_WordBackend("no_boundary"))
        entry = payload["candidates"]["chatml"]
        self.assertEqual(entry["headline"]["decision"], "insufficient")
        self.assertEqual(entry["headline"]["direct_passes"], 0)
        self.assertEqual(entry["arms"]["dose0"]["decomposition"]["non_value"], 24)
        self.assertEqual(entry["arms"]["dose72"]["decomposition"]["non_value"], 24)

    def test_exploding_backend_yields_evaluation_failed_cells(self):
        """An exploding backend yields evaluation-failed cells while the payload stays available."""
        payload = _payload(_WordBackend("explode"))
        self.assertTrue(payload["available"])
        arm = payload["candidates"]["chatml"]["arms"]["dose0"]
        self.assertEqual(arm["n_evaluation_failed"], 24)
        self.assertEqual(arm["n_completed"], 0)
        self.assertEqual(payload["candidates"]["chatml"]["headline"]["n_population"], 0)

    def test_context_exceeded_cells(self):
        """A tiny max_context marks all 24 dose0 cells context-exceeded."""
        payload = _payload(max_context=10)
        arm = payload["candidates"]["chatml"]["arms"]["dose0"]
        self.assertEqual(arm["n_context_exceeded"], 24)
        self.assertEqual(arm["n_completed"], 0)

    def test_budget_exceeded_fails_closed(self):
        """More candidates than the generation budget fails the stage closed."""
        formats = tuple(Format(f"cand-{i:02d}", "", "u:", "a:", "\n") for i in range(25))
        backend = _WordBackend()
        stage = experiments.prune_sweep_stage(
            backend,
            formats,
            boundary=tuple(backend.encode("endmark")),
            max_context=8192,
            seed=0,
        )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "prune_record_budget_exceeded")
        self.assertEqual(backend.calls, 0)

    def test_cancelled_stage(self):
        """A pre-set cancel event yields the cancelled, unavailable payload."""
        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        import threading

        cancel = threading.Event()
        cancel.set()
        backend = _WordBackend()
        stage = experiments.prune_sweep_stage(
            backend,
            (_CHATML,),
            boundary=tuple(backend.encode("endmark")),
            max_context=8192,
            seed=0,
            cancel=cancel,
        )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "cancelled")
        self.assertEqual(backend.calls, 0)

    def test_boundary_text_in_content_fails_the_screen(self):
        """Boundary text embedded in arm content fails the stage screen closed."""
        backend = _WordBackend()
        stage = experiments.prune_sweep_stage(
            backend,
            (_CHATML,),
            boundary=tuple(backend.encode("code")),
            max_context=8192,
            seed=0,
        )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "prune_screen_failed")
        self.assertEqual(backend.calls, 0)

    def test_case_invariant_screen_fails_closed(self):
        """A duplicated context sentence trips the case-invariant screen and fails closed."""
        backend = _WordBackend()
        original = ps.generate_cases

        def mutant(seed):
            """Mutate generate_cases to duplicate a context sentence."""
            cases = [dict(case) for case in original(seed)]
            sentences = list(cases[0]["context_sentences"])
            sentences[1] = sentences[0]
            cases[0]["context_sentences"] = tuple(sentences)
            return tuple(cases)

        ps.generate_cases = mutant  # type: ignore[misc]
        try:
            stage = experiments.prune_sweep_stage(
                backend,
                (_CHATML,),
                boundary=tuple(backend.encode("endmark")),
                max_context=8192,
                seed=0,
            )
        finally:
            ps.generate_cases = original  # type: ignore[misc]
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "prune_screen_failed")
        self.assertEqual(stage["audit"]["by_reason"], {"case_invariant": 1})
        self.assertEqual(backend.calls, 0)

    def test_special_token_screen_fails_closed(self):
        """Arm content containing a special token fails the stage screen closed."""
        backend = _WordBackend()
        original = ps.arm_content

        def mutant(case, arm, seed):
            """Mutate arm_content to inject a special token into a three-line arm text."""
            text = original(case, arm, seed)
            lines = text.split("\n")
            if len(lines) == 3:
                # Inject into the passage line of every passage
                # arm alike, so the case battery's dose word
                # counts still balance and only the audit can
                # catch it.
                lines[1] = lines[1] + " <|im_start|>"
                return "\n".join(lines)
            return text

        ps.arm_content = mutant  # type: ignore[misc]
        try:
            stage = experiments.prune_sweep_stage(
                backend,
                (_CHATML,),
                boundary=tuple(backend.encode("endmark")),
                max_context=8192,
                seed=0,
            )
        finally:
            ps.arm_content = original  # type: ignore[misc]
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "prune_screen_failed")
        self.assertGreater(stage["audit"]["special_token_findings"], 0)
        self.assertEqual(backend.calls, 0)

    def test_records_shape_and_content_binding(self):
        """Stage records carry the frozen shape, with content digests bound to the regenerated
        cases.
        """
        payload = _payload(seed=3)
        cases = ps.generate_cases(3)
        for entry in payload["candidates"].values():
            self.assertEqual(len(entry["records"]), 24)
            for record, case in zip(entry["records"], cases, strict=True):
                self.assertEqual(record["fixture"], case["fixture_id"])
                self.assertEqual(record["topic"], case["topic"])
                self.assertEqual(record["expected_digest"], ps.digest_text(str(case["value"])))
                self.assertEqual(set(record["cells"]), set(ps.ARM_KEYS))
                for arm in ps.ARM_KEYS:
                    self.assertEqual(
                        record["content_digests"][arm],
                        ps.digest_text(ps.arm_content(case, arm, 3)),
                    )
                    self.assertIsInstance(record["arm_tokens"][arm], int)
                self.assertFalse(record["ids_identical"])
                self.assertEqual(record["dropped_shares"], ps.dropped_set_shares(case, 3))

    def test_stage_rejects_duplicate_format_names(self):
        """Duplicate format names are rejected with ValueError."""
        backend = _WordBackend()
        with self.assertRaises(ValueError):
            experiments.prune_sweep_stage(
                backend,
                (_CHATML, _CHATML),
                boundary=tuple(backend.encode("endmark")),
                max_context=8192,
                seed=0,
            )


class ValidatorTests(unittest.TestCase):
    """The diff validator recompute standard: real payloads
    validate; every forgery class is rejected.
    """

    def test_real_payloads_validate(self):
        """Real payloads from every scripted mode validate under the diff mirror."""
        for mode in ("perfect", "dose72_drop", "both72_drop", "no_boundary", "explode"):
            payload = _payload(_WordBackend(mode))
            self.assertTrue(diffreports._valid_prune_sweep(payload), mode)

    def test_self_diff_is_unchanged(self):
        """Diffing a payload against itself reports both-available and unchanged."""
        payload = _payload(_WordBackend("dose72_drop"))
        diff = diffreports.diff_prune_sweep(payload, copy.deepcopy(payload))
        self.assertEqual(diff["available"], "both")
        self.assertFalse(diff["changed"])
        self.assertTrue(diff["comparable"])

    def test_changed_payloads_diff(self):
        """Diffing a perfect payload against a dose72-drop payload reports the change."""
        old = _payload(_WordBackend("perfect"))
        new = _payload(_WordBackend("dose72_drop"))
        diff = diffreports.diff_prune_sweep(old, new)
        self.assertTrue(diff["changed"])
        metrics = {(d["candidate"], d["metric"]) for d in diff["candidate_drift"]}
        self.assertIn(("chatml", "decision"), metrics)
        self.assertIn(("chatml", "drop_cases"), metrics)

    def test_one_sided_and_neither(self):
        """One-sided payloads diff as old/new only; two Nones diff as neither."""
        payload = _payload(_WordBackend("perfect"))
        self.assertEqual(diffreports.diff_prune_sweep(payload, None)["available"], "old")
        self.assertEqual(diffreports.diff_prune_sweep(None, payload)["available"], "new")
        self.assertEqual(diffreports.diff_prune_sweep(None, None)["available"], "neither")
        self.assertEqual(
            diffreports.diff_prune_sweep({"available": False}, payload)["available"], "new"
        )

    @staticmethod
    def _tampered(mutate):
        """Return a perfect payload mutated by the given forgery function."""
        payload = _payload(_WordBackend("perfect"))
        mutate(payload)
        return payload

    def test_tamper_battery(self):
        """Every forgery in the battery makes the mirror reject the payload."""

        def verdict_flip(p):
            """Forgery: flip one dose72 cell verdict to False."""
            p["candidates"]["chatml"]["records"][0]["cells"]["dose72"]["verdict"] = False

        def content_forgery(p):
            """Forgery: zero one dose72 content digest."""
            p["candidates"]["chatml"]["records"][0]["content_digests"]["dose72"] = "0" * 64

        def ids_forgery(p):
            """Forgery: copy the dose0 id digest over the dose72 one."""
            record = p["candidates"]["chatml"]["records"][0]
            record["id_digests"]["dose72"] = record["id_digests"]["dose0"]

        def ids_flag_flip(p):
            """Forgery: flip ids_identical to True."""
            p["candidates"]["chatml"]["records"][0]["ids_identical"] = True

        def shares_forgery(p):
            """Forgery: set a dropped share to 0.5."""
            p["candidates"]["chatml"]["records"][0]["dropped_shares"]["dose72"]["numeral"] = 0.5

        def token_forgery(p):
            """Forgery: increment one arm token count."""
            p["candidates"]["chatml"]["records"][0]["arm_tokens"]["dose72"] += 1

        def protocol_forgery(p):
            """Forgery: zero the protocol fingerprint."""
            p["protocol"]["fingerprint"] = "0" * 64

        def count_forgery(p):
            """Forgery: set n_cases to 23."""
            p["n_cases"] = 23

        def aggregate_forgery(p):
            """Forgery: rewrite the headline decision to prune_drop_observed."""
            p["candidates"]["chatml"]["headline"]["decision"] = "prune_drop_observed"

        def record_swap(p):
            """Forgery: swap the first two records."""
            records = p["candidates"]["chatml"]["records"]
            records[0], records[1] = records[1], records[0]

        def detail_forgery(p):
            """Forgery: null a cell while leaving its detail empty."""
            record = p["candidates"]["chatml"]["records"][0]
            record["cells"]["dose18"] = None
            record["cell_details"]["dose18"] = ""

        def audit_forgery(p):
            """Forgery: set a negative special-token finding count."""
            p["audit"]["special_token_findings"] = -1

        for mutate in (
            verdict_flip,
            content_forgery,
            ids_forgery,
            ids_flag_flip,
            shares_forgery,
            token_forgery,
            protocol_forgery,
            count_forgery,
            aggregate_forgery,
            record_swap,
            detail_forgery,
            audit_forgery,
        ):
            payload = self._tampered(mutate)
            self.assertFalse(diffreports._valid_prune_sweep(payload), mutate.__name__)
            diff = diffreports.diff_prune_sweep(payload, _payload(_WordBackend("perfect")))
            self.assertEqual(diff["available"], "new", mutate.__name__)

    def test_boundary_only_cell_validates_in_a_real_record_set(self):
        # A boundary_only dose72 cell in an otherwise real
        # payload: rebuild the aggregates through the producer
        # and the mirror must accept (attrition, not A).
        """A boundary-only cell in a real record set validates: a legitimate cell shape."""
        payload = _payload(_WordBackend("perfect"))
        entry = payload["candidates"]["chatml"]
        record = entry["records"][0]
        cell = record["cells"]["dose72"]
        cell["verdict"] = False
        cell["boundary_found"] = False
        aggregates = ps.aggregate_candidate(entry["records"])
        aggregates.update({"status": "available", "detail": "", "records": entry["records"]})
        payload["candidates"]["chatml"] = aggregates
        self.assertTrue(diffreports._valid_prune_sweep(payload))
        self.assertEqual(aggregates["headline"]["boundary_attrition"], 1)
        self.assertEqual(aggregates["headline"]["drop_cases"], 0)


class ReportingTests(unittest.TestCase):
    """Design section 6: the report section in printed
    order, with the corrected OC paragraph and the bundle
    fencing.
    """

    def test_section_renders_for_drop_payload(self):
        """The prune-sweep section renders for a drop payload under its card anchor."""
        payload = _payload(_WordBackend("dose72_drop"))
        html = reporting.render_html({"prune_sweep": payload})
        self.assertIn('<section class="card" id="prune-sweep">', html)
        self.assertIn('href="#prune-sweep"', html)
        section = reporting._prune_sweep_section({"prune_sweep": payload})
        self.assertIn("prune_drop_observed", section)
        self.assertIn("boundary attrition beside A", section)
        self.assertIn("0.0143", section)
        self.assertIn("36.0 words", section)
        self.assertIn("15.95 of the 16 context sentences", section)
        self.assertNotIn("word-count loss at 50 percent", section)

    def test_rival_wording_prints_only_when_rival(self):
        """The rival word-count wording prints for the both72 payload."""
        payload = _payload(_WordBackend("both72_drop"))
        section = reporting._prune_sweep_section({"prune_sweep": payload})
        self.assertIn("word-count loss at 50 percent of the eligible words", section)

    def test_section_empty_without_payload(self):
        """The section is empty when the payload is absent."""
        self.assertEqual(reporting._prune_sweep_section({}), "")
        self.assertEqual(
            reporting._prune_sweep_section(
                {"prune_sweep": {"available": False, "detail": "prune_unavailable"}}
            ),
            "",
        )

    def test_hostile_candidate_name_is_escaped(self):
        """A script-tag candidate name is HTML-escaped in the rendered section."""
        payload = _payload(_WordBackend("perfect"))
        entry = payload["candidates"].pop("chatml")
        payload["candidates"]["<script>alert(1)</script>"] = entry
        section = reporting._prune_sweep_section({"prune_sweep": payload})
        self.assertNotIn("<script>alert(1)</script>", section)
        self.assertIn("&lt;script&gt;", section)


class CliTests(unittest.TestCase):
    """Amendment H + the CLI surface: the flag exists, its
    help discloses the pruning in the first sentence and
    renders (no bare percent), and the wrapper never
    raises.
    """

    def test_flag_and_help(self):
        """The --prune-sweep flag is registered and its help renders with the first-sentence
        disclosure.
        """
        import argparse
        import io
        from contextlib import redirect_stdout

        from model_protocol_lab import cli

        parser = cli.build_parser()
        args = parser.parse_args(
            [
                "run",
                "--prune-sweep",
                "--model",
                "m",
                "--dataset",
                "d",
                "--family",
                "chatml",
                "--boundary-ids",
                "1",
                "--output",
                "o",
            ]
        )
        self.assertTrue(args.prune_sweep)
        # The run subparser's own help carries the disclosure
        # (rendering it exercises argparse's percent expansion
        # of help strings).
        # skipcq: PTC-W0063 - in a test, next() raising StopIteration is the intended failure
        subparsers = next(
            action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
        )
        run_parser = subparsers.choices["run"]
        self.assertIn("--prune-sweep", run_parser.format_help())
        buffer = io.StringIO()
        with redirect_stdout(buffer), self.assertRaises(SystemExit):
            parser.parse_args(["run", "--help"])
        self.assertIn("prunes passage words", buffer.getvalue())

    def test_wrapper_success_and_failure(self):
        """The CLI wrapper returns the stage payload on success and fails closed when the stage
        raises.
        """
        from model_protocol_lab import cli

        backend = _WordBackend()
        result: dict = {}
        cli._run_prune_sweep_probe(
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
        self.assertTrue(result["prune_sweep"]["available"])

        original = experiments.prune_sweep_stage

        def boom(*args, **kwargs):
            """Replacement stage that raises, exercising the wrapper's fail-closed path."""
            raise RuntimeError("stage exploded")

        experiments.prune_sweep_stage = boom  # type: ignore[misc]
        try:
            result2: dict = {}
            cli._run_prune_sweep_probe(
                backend,
                (_CHATML,),
                result2,
                boundary=tuple(backend.encode("endmark")),
                boundary_map=None,
                max_new_tokens=64,
                max_context=8192,
                seed=0,
                cancel=None,
            )
        finally:
            experiments.prune_sweep_stage = original  # type: ignore[misc]
        self.assertEqual(
            result2["prune_sweep"], {"available": False, "detail": "prune_unavailable"}
        )


class GapValidatorBatteryTests(unittest.TestCase):
    """Hostile-payload battery over the validator's reject
    branches (shape, types, bindings).
    """

    @staticmethod
    def _bad(mutate):
        """Return a perfect payload mutated by the given reject-battery forgery."""
        payload = _payload(_WordBackend("perfect"))
        mutate(payload)
        return payload

    def test_reject_battery(self):
        """Each malformed-payload forgery in the battery is rejected by the mirror validator."""

        def top_level_key(p):
            """Forgery: add an extra top-level key."""
            p["extra"] = 1

        def detail_not_empty(p):
            """Forgery: set a non-empty detail on an available payload."""
            p["detail"] = "x"

        def audit_missing_key(p):
            """Forgery: delete the audit by_reason map."""
            del p["audit"]["by_reason"]

        def audit_negative(p):
            """Forgery: set a negative audit exclusion count."""
            p["audit"]["n_audit_excluded"] = -1

        def candidate_unavailable_shape(p):
            """Forgery: give an unavailable candidate a records list."""
            p["candidates"]["chatml"] = {"status": "unavailable", "detail": "x", "records": [1]}

        def entry_missing_aggregate_key(p):
            """Forgery: delete the candidate headline block."""
            del p["candidates"]["chatml"]["headline"]

        def record_missing_field(p):
            """Forgery: delete a record's topic field."""
            del p["candidates"]["chatml"]["records"][0]["topic"]

        def record_wrong_topic(p):
            """Forgery: set a record topic outside the frozen topics."""
            p["candidates"]["chatml"]["records"][0]["topic"] = "summit trail"

        def record_wrong_expected(p):
            """Forgery: zero a record's expected digest."""
            p["candidates"]["chatml"]["records"][0]["expected_digest"] = "0" * 64

        def cell_field_missing(p):
            """Forgery: delete a cell's input_tokens field."""
            del p["candidates"]["chatml"]["records"][0]["cells"]["dose0"]["input_tokens"]

        def cell_verdict_not_bool(p):
            """Forgery: set a cell verdict to the int 1 instead of a bool."""
            p["candidates"]["chatml"]["records"][0]["cells"]["dose0"]["verdict"] = 1

        def cell_digest_malformed(p):
            """Forgery: set a malformed cell answer digest."""
            p["candidates"]["chatml"]["records"][0]["cells"]["dose0"]["answer_digest"] = "zz"

        def cell_input_tokens_mismatch(p):
            """Forgery: increment a cell's input_tokens."""
            p["candidates"]["chatml"]["records"][0]["cells"]["dose0"]["input_tokens"] += 1

        def detail_unknown(p):
            """Forgery: pair a null cell with an unknown detail string."""
            record = p["candidates"]["chatml"]["records"][0]
            record["cells"]["dose0"] = None
            record["cell_details"]["dose0"] = "arm_parse_failed"

        def tokens_negative(p):
            """Forgery: set a negative arm token count."""
            p["candidates"]["chatml"]["records"][0]["arm_tokens"]["dose0"] = -1

        def tokens_bool(p):
            """Forgery: set an arm token count to True."""
            p["candidates"]["chatml"]["records"][0]["arm_tokens"]["dose0"] = True

        def id_digest_malformed(p):
            """Forgery: set a malformed id digest."""
            p["candidates"]["chatml"]["records"][0]["id_digests"]["dose0"] = "nope"

        def seed_bool(p):
            """Forgery: set the protocol seed to True."""
            p["protocol"]["seed"] = True

        def n_candidates_wrong(p):
            """Forgery: set n_candidates to 5."""
            p["n_candidates"] = 5

        def protocol_not_dict(p):
            """Forgery: replace the protocol with a string."""
            p["protocol"] = "x"

        def too_many_candidates(p):
            """Forgery: inflate the candidate set past the cap."""
            for i in range(30):
                p["candidates"][f"c{i}"] = {
                    "status": "unavailable",
                    "detail": "x",
                    "records": [],
                }

        def entry_not_dict(p):
            """Forgery: replace a candidate entry with an int."""
            p["candidates"]["chatml"] = 42

        def entry_status_weird(p):
            """Forgery: set an unknown candidate status."""
            p["candidates"]["chatml"]["status"] = "weird"

        def records_not_list(p):
            """Forgery: replace the records list with a dict."""
            p["candidates"]["chatml"]["records"] = {}

        def records_short(p):
            """Forgery: truncate the records list to 23 entries."""
            p["candidates"]["chatml"]["records"] = p["candidates"]["chatml"]["records"][:23]

        def record_not_dict(p):
            """Forgery: replace a record with an int."""
            p["candidates"]["chatml"]["records"][0] = 42

        def content_digest_tail_wrong(p):
            """Forgery: zero the tail72 content digest."""
            p["candidates"]["chatml"]["records"][0]["content_digests"]["tail72"] = "0" * 64

        def dropped_shares_wrong(p):
            """Forgery: set a dropped share to 0.5."""
            p["candidates"]["chatml"]["records"][0]["dropped_shares"]["dose72"]["numeral"] = 0.5

        def cell_none_empty_detail(p):
            """Forgery: null a cell and empty its detail."""
            record = p["candidates"]["chatml"]["records"][0]
            record["cells"]["dose0"] = None
            record["cell_details"]["dose0"] = ""

        def cell_present_with_detail(p):
            """Forgery: attach a detail string to a present cell."""
            p["candidates"]["chatml"]["records"][0]["cell_details"]["dose0"] = (
                "arm_evaluation_failed"
            )

        def cell_input_tokens_string(p):
            """Forgery: set a cell's input_tokens to a string."""
            p["candidates"]["chatml"]["records"][0]["cells"]["dose0"]["input_tokens"] = "40"

        def by_reason_not_dict(p):
            """Forgery: replace the audit by_reason map with a string."""
            p["audit"]["by_reason"] = "x"

        def by_reason_int_key(p):
            """Forgery: key the by_reason map by an int."""
            p["audit"]["by_reason"] = {1: 2}

        def by_reason_negative_count(p):
            """Forgery: set a negative by_reason count."""
            p["audit"]["by_reason"] = {"case_invariant": -1}

        for mutate in (
            top_level_key,
            detail_not_empty,
            audit_missing_key,
            audit_negative,
            candidate_unavailable_shape,
            entry_missing_aggregate_key,
            record_missing_field,
            record_wrong_topic,
            record_wrong_expected,
            cell_field_missing,
            cell_verdict_not_bool,
            cell_digest_malformed,
            cell_input_tokens_mismatch,
            detail_unknown,
            tokens_negative,
            tokens_bool,
            id_digest_malformed,
            seed_bool,
            n_candidates_wrong,
            protocol_not_dict,
            too_many_candidates,
            entry_not_dict,
            entry_status_weird,
            records_not_list,
            records_short,
            record_not_dict,
            content_digest_tail_wrong,
            dropped_shares_wrong,
            cell_none_empty_detail,
            cell_present_with_detail,
            cell_input_tokens_string,
            by_reason_not_dict,
            by_reason_int_key,
            by_reason_negative_count,
        ):
            self.assertFalse(diffreports._valid_prune_sweep(self._bad(mutate)), mutate.__name__)


class GapReportingTests(unittest.TestCase):
    """The section renderer over hostile/fat payload
    shapes: it must never raise and must degrade to n/a.
    """

    def test_fat_payload_renders(self):
        """A fat payload with a non-numeric rate renders without raising, degrading to n/a."""
        payload = _payload(_WordBackend("dose72_drop"))
        entry = payload["candidates"]["chatml"]
        entry["arms"]["dose0"]["rate"] = "not-a-number"
        entry["headline"]["direct_passes"] = "many"
        entry["ids"]["dose72"]["id_length_delta_by_case"] = "not-a-list"
        entry["dropped_shares"]["dose72"] = {"numeral": "x"}
        section = reporting._prune_sweep_section({"prune_sweep": payload})
        self.assertIn("n/a", section)
        self.assertIn("prune_drop_observed", section)

    def test_unavailable_candidate_row(self):
        """An unavailable candidate renders as its own row in the section."""
        payload = _payload(_WordBackend("perfect"))
        payload["candidates"]["ghost"] = {
            "status": "unavailable",
            "detail": "prune_unavailable",
            "records": [],
        }
        section = reporting._prune_sweep_section({"prune_sweep": payload})
        self.assertIn("ghost", section)
        self.assertIn("unavailable", section)


class GapCoverageTests(unittest.TestCase):
    """Gap pass (cycle 220): the branches the first full
    gate run's coverage showed unreached -- validate_case's
    screen branches behind the regeneration gate (exercised
    by patching the regeneration to return the mutated case,
    exactly the list-edit failure mode those screens guard),
    the aggregate helpers' malformed-record paths, the
    stage's remaining precondition/screen paths, and the
    renderer's malformed-candidate paths.
    """

    @staticmethod
    def _case():
        """Return the seed-0 case-0 fixture."""
        return ps.generate_case(0, 0)

    def _with_case(self, mutate):
        """Validate a mutated copy of the seed-0 case, with generate_case patched."""
        case = copy.deepcopy(self._case())
        mutate(case)
        with mock.patch.object(ps, "generate_case", return_value=case):
            return ps.validate_case(case, 0)

    def test_draw_exhaustion_raises(self):
        """Exhausting the value draws raises ValueError."""
        template = ps.TEMPLATES[1]
        sentence = template.format(
            subject=ps.SUBJECTS[0],
            object=ps.OBJECTS[0],
            place=ps.PLACES[0],
            time=ps.TIMES[0],
        )
        with mock.patch.object(ps, "_sha", return_value="0" * 64), self.assertRaises(ValueError):
            ps._draw_sentence(0, "prune-00", 1, {sentence})

    def test_dropped_positions_unknown_arm_is_empty(self):
        """dropped_positions is empty for the closed-book arm and for unknown arms."""
        self.assertEqual(ps.dropped_positions("prune-00", "closed_book", 0), frozenset())
        self.assertEqual(ps.dropped_positions("prune-00", "bogus", 0), frozenset())

    def test_validate_duplicate_sentence(self):
        """validate_case rejects a duplicated context sentence."""

        def mutate(case):
            """Mutate: duplicate a context sentence."""
            case["context_sentences"] = case["context_sentences"][:15] + (
                case["context_sentences"][0],
            )

        self.assertFalse(self._with_case(mutate))

    def test_validate_short_sentence(self):
        """validate_case rejects a context sentence truncated below the minimum length."""

        def mutate(case):
            """Mutate: truncate sentence 3 to eight words."""
            sentences = list(case["context_sentences"])
            sentences[3] = " ".join(sentences[3].split(" ")[:8])
            case["context_sentences"] = tuple(sentences)

        self.assertFalse(self._with_case(mutate))

    def test_validate_short_fact(self):
        """validate_case rejects a fact sentence below the frozen word count."""

        def mutate(case):
            """Mutate: replace the fact sentence with a five-word one."""
            case["fact_sentence"] = "The harbor light code amber."

        self.assertFalse(self._with_case(mutate))

    def test_validate_value_in_context(self):
        """validate_case rejects the answer value appearing inside a context sentence."""

        def mutate(case):
            """Mutate: plant the answer value inside sentence 2."""
            sentences = list(case["context_sentences"])
            words = sentences[2].split(" ")
            words[4] = case["value"]
            sentences[2] = " ".join(words)
            case["context_sentences"] = tuple(sentences)

        self.assertFalse(self._with_case(mutate))

    def test_validate_stem_collision(self):
        """validate_case rejects a context word sharing the answer value's stem."""

        def mutate(case):
            """Mutate: plant a stem-colliding word in sentence 2."""
            sentences = list(case["context_sentences"])
            words = sentences[2].split(" ")
            words[4] = case["value"][:4] + "zzzz"
            sentences[2] = " ".join(words)
            case["context_sentences"] = tuple(sentences)

        self.assertFalse(self._with_case(mutate))

    def test_validate_pool_intersects_value(self):
        """validate_case rejects a word pool containing the answer value."""
        case = self._case()
        with mock.patch.object(ps, "_POOL_WORDS", ps._POOL_WORDS | {case["value"]}):
            self.assertFalse(ps.validate_case(case, 0))

    def test_validate_pool_intersects_fact_words(self):
        """validate_case rejects a word pool containing a fact-sentence word."""
        case = self._case()
        with mock.patch.object(ps, "_POOL_WORDS", ps._POOL_WORDS | {"code"}):
            self.assertFalse(ps.validate_case(case, 0))

    def test_validate_pool_intersects_instruction(self):
        """validate_case rejects a word pool containing an instruction word."""
        case = self._case()
        with mock.patch.object(ps, "_POOL_WORDS", ps._POOL_WORDS | {"passage"}):
            self.assertFalse(ps.validate_case(case, 0))

    def test_validate_digit_word(self):
        """validate_case rejects a context word containing digits."""

        def mutate(case):
            """Mutate: replace a sentence-5 word with 'abc123'."""
            sentences = list(case["context_sentences"])
            words = sentences[5].split(" ")
            words[2] = "abc123"
            sentences[5] = " ".join(words)
            case["context_sentences"] = tuple(sentences)

        self.assertFalse(self._with_case(mutate))

    def test_validate_negation_word(self):
        """validate_case rejects a negation word inside the context."""

        def mutate(case):
            """Mutate: replace a sentence-5 word with 'never'."""
            sentences = list(case["context_sentences"])
            words = sentences[5].split(" ")
            words[2] = "never"
            sentences[5] = " ".join(words)
            case["context_sentences"] = tuple(sentences)

        self.assertFalse(self._with_case(mutate))

    def test_validate_acrostic(self):
        """validate_case rejects a context whose sentence initials spell a pool value."""

        def mutate(case):
            """Mutate: rebuild the context so its sentence initials form an acrostic."""
            starts = ["Apples", "Melons", "Boxes", "Ember", "Rocks"]
            tails = [
                "morning",
                "midday",
                "evening",
                "weekdays",
                "season",
                "holiday",
                "weekend",
                "noon",
                "dusk",
                "dawn",
                "closing",
                "harvest",
                "springtime",
                "winter",
                "summer",
                "autumn",
            ]
            sentences = []
            for slot in range(16):
                first = starts[slot] if slot < len(starts) else "Tables"
                sentences.append(f"{first} quietly fills the wide wooden shelf each {tails[slot]}.")
            case["context_sentences"] = tuple(sentences)

        self.assertFalse(self._with_case(mutate))

    def test_validate_reversed_drop_order_still_nested(self):
        # Nesting is a set property of one permutation's
        # prefixes: any single returned order, even reversed,
        # has nested prefix sets, so the battery's nesting
        # line is a guard against an inconsistent order
        # function, not against a different order. Pin the
        # semantics honestly: a reversed (but consistent)
        # order still validates.
        """A reversed drop order still yields nested dropped sets that validate."""
        case = self._case()
        with mock.patch.object(ps, "drop_order", return_value=tuple(range(143, -1, -1))):
            self.assertTrue(ps.validate_case(case, 0))

    def test_validate_tail_set_mismatch(self):
        """validate_case rejects a tail dropped set that mismatches the dose72 set."""
        case = self._case()
        original = ps.dropped_positions

        def fake(case_id, arm, seed):
            """Fake dropped_positions returning a wrong tail set for the descriptive arm."""
            if arm == ps.DESCRIPTIVE_ARM:
                return frozenset({0})
            return original(case_id, arm, seed)

        with mock.patch.object(ps, "dropped_positions", side_effect=fake):
            self.assertFalse(ps.validate_case(case, 0))

    def test_validate_content_instruction_mismatch(self):
        """validate_case rejects arm content whose instruction differs from the case."""
        case = self._case()
        original = ps.arm_content

        def fake(case_arg, arm, seed):
            """Fake arm_content altering the dose0 instruction."""
            text = original(case_arg, arm, seed)
            if arm == "dose0":
                return "Wrong instruction line.\n" + text.split("\n", 1)[1]
            return text

        with mock.patch.object(ps, "arm_content", side_effect=fake):
            self.assertFalse(ps.validate_case(case, 0))

    def test_validate_content_fact_missing(self):
        """validate_case rejects arm content missing the fact sentence."""
        case = self._case()
        original = ps.arm_content

        def fake(case_arg, arm, seed):
            """Fake arm_content dropping the fact sentence from the dose18 arm."""
            text = original(case_arg, arm, seed)
            if arm == "dose18":
                lines = text.split("\n")
                lines[1] = lines[1].replace(
                    case_arg["fact_sentence"], "The harbor light code is gone."
                )
                return "\n".join(lines)
            return text

        with mock.patch.object(ps, "arm_content", side_effect=fake):
            self.assertFalse(ps.validate_case(case, 0))

    def test_validate_dose_count_mismatch(self):
        """validate_case rejects arm content with a wrong dropped-word count."""
        case = self._case()
        original = ps.arm_content

        def fake(case_arg, arm, seed):
            """Fake arm_content dropping the wrong word count in the dose36 arm."""
            text = original(case_arg, arm, seed)
            if arm == "dose36":
                lines = text.split("\n")
                lines[1] = lines[1] + " extra"
                return "\n".join(lines)
            return text

        with mock.patch.object(ps, "arm_content", side_effect=fake):
            self.assertFalse(ps.validate_case(case, 0))

    def test_aggregate_malformed_records(self):
        """Malformed records aggregate to an empty population instead of raising."""
        records = [
            {
                "fixture": "prune-00",
                "topic": "harbor light",
                "expected_digest": ps.digest_text("amber"),
                "arm_tokens": "not-a-dict",
                "id_digests": "not-a-dict",
                "ids_identical": False,
                "content_digests": {},
                "cells": "not-a-dict",
                "cell_details": "not-a-dict",
            }
        ]
        aggregates = ps.aggregate_candidate(records)
        self.assertEqual(aggregates["headline"]["n_population"], 0)
        self.assertEqual(aggregates["arms"]["dose0"]["n_completed"], 0)
        self.assertEqual(aggregates["ids"]["dose72"]["id_length_delta_by_case"], [0])
        self.assertEqual(aggregates["dropped_shares"]["dose72"], {"numeral": 0.0, "negation": 0.0})

    def test_aggregate_tail_incomplete_and_tail_only(self):
        """Tail-incomplete and tail-only records aggregate, with the tail contrast skipping
        incomplete cells.
        """
        records = [_record(index, _pass_verdicts()) for index in range(14)]
        # One population record with an incomplete tail cell
        # (skipped by the contrast), one where only the tail
        # fails (the tail72_only discordant direction).
        records[0] = _record(0, _pass_verdicts(tail72=None))
        records[1] = _record(1, _pass_verdicts(tail72=False))
        aggregates = ps.aggregate_candidate(records)
        self.assertEqual(aggregates["tail_contrast"]["n_compared"], 13)
        self.assertEqual(aggregates["tail_contrast"]["tail72_only"], 1)
        self.assertEqual(aggregates["tail_contrast"]["dose72_only"], 0)

    def test_stage_formats_must_be_tuple(self):
        """A non-tuple formats argument is rejected with ValueError."""
        backend = _WordBackend("perfect")
        with self.assertRaises(ValueError):
            experiments.prune_sweep_stage(
                backend,
                [_CHATML],
                boundary=tuple(backend.encode("endmark")),
                seed=0,
                max_new_tokens=64,
                max_context=8192,
            )

    def test_stage_cancel_during_generation(self):
        """Cancellation raised during generation yields the cancelled payload."""
        cancel = threading.Event()
        event = cancel

        class CancellingBackend(_WordBackend):
            """Word backend whose generate sets the cancel event before delegating."""

            def generate(self, input_ids, *, max_new_tokens, seed=None, cancel=None):
                """Generate that sets the cancel event, then delegates to the scripted backend."""
                event.set()
                return super().generate(
                    input_ids, max_new_tokens=max_new_tokens, seed=seed, cancel=cancel
                )

        backend = CancellingBackend("perfect")
        stage = experiments.prune_sweep_stage(
            backend,
            (_CHATML,),
            boundary=tuple(backend.encode("endmark")),
            seed=0,
            max_new_tokens=64,
            max_context=8192,
            cancel=cancel,
        )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "cancelled")

    def test_stage_undecodable_boundary_screens_nothing(self):
        """An undecodable boundary decode screens nothing and the stage proceeds."""

        class UndecodableBoundaryBackend(_WordBackend):
            """Word backend whose decode leaves the boundary ids undecoded."""

            def decode(self, ids):
                """Decode that leaves the boundary ids undecoded, delegating otherwise."""
                if list(ids) == list(self.encode("endmark")):
                    raise RuntimeError("no decode")
                return super().decode(ids)

        backend = UndecodableBoundaryBackend("perfect")
        stage = experiments.prune_sweep_stage(
            backend,
            (_CHATML,),
            boundary=tuple(backend.encode("endmark")),
            seed=0,
            max_new_tokens=64,
            max_context=8192,
        )
        self.assertTrue(stage["available"])
        self.assertEqual(backend.calls, 144)

    @staticmethod
    def _real_stage():
        """Run the real prune-sweep stage with a perfect scripted backend."""
        backend = _WordBackend("perfect")
        return experiments.prune_sweep_stage(
            backend,
            (_CHATML,),
            boundary=tuple(backend.encode("endmark")),
            seed=0,
            max_new_tokens=64,
            max_context=8192,
        )

    def test_analysis_detail_passthrough(self):
        """The analysis passes an unavailable stage's detail through unchanged."""
        stage = dict(self._real_stage(), available=False, detail="cancelled")
        out = experiments.prune_sweep_analysis(stage, ("chatml",))
        self.assertFalse(out["available"])
        self.assertEqual(out["detail"], "cancelled")

    def test_analysis_unavailable_candidate_passthrough(self):
        """The analysis passes an unavailable candidate entry through unchanged."""
        stage = self._real_stage()
        stage["candidates"]["chatml"] = {
            "status": "unavailable",
            "detail": "arm_evaluation_failed",
            "records": [],
        }
        out = experiments.prune_sweep_analysis(stage, ("chatml",))
        self.assertTrue(out["available"])
        self.assertEqual(out["candidates"]["chatml"]["status"], "unavailable")
        self.assertEqual(out["candidates"]["chatml"]["records"], [])

    def test_validate_malformed_case_fails_closed(self):
        """validate_case returns False for empty, partial and None cases."""
        self.assertFalse(ps.validate_case({}, 0))
        self.assertFalse(ps.validate_case({"fixture_id": "prune-00"}, 0))
        self.assertFalse(ps.validate_case(None, 0))
        self.assertFalse(ps.validate_case({"fixture_id": "bogus"}, 0))

    def test_aggregate_ids_identical_non_dict_digests(self):
        """Records with non-dict id digests aggregate without raising."""
        records = [_record(index, _pass_verdicts()) for index in range(3)]
        for record in records:
            record["id_digests"] = "not-a-dict"
        aggregates = ps.aggregate_candidate(records)
        self.assertEqual(aggregates["headline"]["n_ids_identical"], 0)
        self.assertEqual(aggregates["headline"]["n_population"], 3)

    def test_stage_cancel_set_by_encode(self):
        # The event lands during the first arm's encode --
        # after the stage's per-case check, before the first
        # generation -- so the cell's own pre-evaluation
        # cancel check is the one that fires.
        """A cancel event set during encode yields the cancelled payload."""
        cancel = threading.Event()

        class EncodeCancellingBackend(_WordBackend):
            """Word backend whose encode sets the cancel event on the instruction text."""

            def encode(self, text):
                """Encode that sets the cancel event when the instruction text is encoded."""
                if "Read the passage" in text:
                    cancel.set()
                return super().encode(text)

        backend = EncodeCancellingBackend("perfect")
        stage = experiments.prune_sweep_stage(
            backend,
            (_CHATML,),
            boundary=tuple(backend.encode("endmark")),
            seed=0,
            max_new_tokens=64,
            max_context=8192,
            cancel=cancel,
        )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "cancelled")
        self.assertEqual(backend.calls, 0)

    def test_validator_context_exceeded_payload(self):
        """A context-exceeded payload validates under the mirror."""
        payload = _payload(_WordBackend("perfect"), max_context=10)
        self.assertTrue(diffreports._valid_prune_sweep(payload))
        entry = payload["candidates"]["chatml"]
        self.assertEqual(entry["arms"]["dose0"]["n_context_exceeded"], 24)

    def test_tail72_drop_payload_validates(self):
        """A tail72-drop payload validates and its tail contrast is populated."""
        payload = _payload(_WordBackend("tail72_drop"))
        self.assertTrue(diffreports._valid_prune_sweep(payload))
        entry = payload["candidates"]["chatml"]
        self.assertEqual(entry["tail_contrast"]["tail72_only"], 24)
        self.assertEqual(entry["tail_contrast"]["dose72_only"], 0)
        self.assertEqual(entry["headline"]["decision"], "below_threshold")
        diff = diffreports.diff_prune_sweep(payload, copy.deepcopy(payload))
        self.assertFalse(diff["changed"])

    def test_ids_identical_payload_validates(self):
        """An ids-identical payload validates, with per-record id digests bound."""
        payload = _payload(_WordBackend("dose72_drop"))
        entry = payload["candidates"]["chatml"]
        for record in entry["records"]:
            record["id_digests"]["dose72"] = record["id_digests"]["dose0"]
            record["ids_identical"] = True
        rebuilt = ps.aggregate_candidate(entry["records"])
        rebuilt.update(
            {"status": entry["status"], "detail": entry["detail"], "records": entry["records"]}
        )
        payload["candidates"]["chatml"] = rebuilt
        self.assertTrue(diffreports._valid_prune_sweep(payload))
        self.assertEqual(rebuilt["headline"]["n_ids_identical"], 24)
        self.assertEqual(rebuilt["headline"]["n_population"], 0)
        self.assertEqual(rebuilt["ids"]["dose72"]["n_same_ids_flips"], 24)

    def test_diff_candidate_status_drift(self):
        """Diffing payloads whose candidate status drifted reports the drift."""
        old = _payload(_WordBackend("perfect"))
        new = copy.deepcopy(old)
        new["candidates"]["chatml"] = {
            "status": "unavailable",
            "detail": "prune_unavailable",
            "records": [],
        }
        self.assertTrue(diffreports._valid_prune_sweep(new))
        diff = diffreports.diff_prune_sweep(old, new)
        self.assertTrue(diff["changed"])
        self.assertIn(
            "status",
            {row["metric"] for row in diff["candidate_drift"] if row["candidate"] == "chatml"},
        )

    def test_mirror_direct_units(self):
        # The diffreports mirror's defensive units, called
        # directly: several are unreachable through
        # _valid_prune_sweep because the recompute's own
        # shape checks reject such records before the
        # aggregate runs (the same defensive posture as the
        # producer's helpers).
        """The mirror's digest and cell helpers behave on direct unit inputs, including the
        patched-digest path.
        """
        with mock.patch.object(diffreports, "_prunesweep_digest", return_value="0" * 64):
            template = diffreports._PRUNESWEEP_TEMPLATES[1]
            sentence = template.format(
                subject=diffreports._PRUNESWEEP_SUBJECTS[0],
                object=diffreports._PRUNESWEEP_OBJECTS[0],
                place=diffreports._PRUNESWEEP_PLACES[0],
                time=diffreports._PRUNESWEEP_TIMES[0],
            )
            with self.assertRaises(ValueError):
                diffreports._prunesweep_draw_sentence(0, "prune-00", 1, {sentence})
        self.assertEqual(
            diffreports._prunesweep_dropped_positions("prune-00", "bogus", 0), frozenset()
        )
        self.assertIsNone(diffreports._prunesweep_cell_get({"cells": "x"}, "dose0"))
        self.assertEqual(diffreports._prunesweep_detail_get({"cell_details": "x"}, "dose0"), "")
        self.assertFalse(diffreports._prunesweep_ids_identical({"id_digests": "x"}))
        aggregates = diffreports._prunesweep_aggregate([{}])
        self.assertEqual(aggregates["n_cases"], 1)
        self.assertEqual(aggregates["ids"]["dose72"]["id_length_delta_by_case"], [0])
        self.assertFalse(
            diffreports._prunesweep_cell_ok(
                {
                    "verdict": False,
                    "boundary_found": True,
                    "boundary_text_match": None,
                    "input_tokens": "40",
                    "answer_digest": None,
                },
                "0" * 64,
            )
        )

    def test_mirror_recompute_caps(self):
        """The mirror's recompute caps the candidate count it will process."""
        payload = _payload(_WordBackend("perfect"))
        many = copy.deepcopy(payload)
        for i in range(30):
            many["candidates"][f"c{i}"] = {"status": "unavailable", "detail": "x", "records": []}
        self.assertIsNone(diffreports._prunesweep_recompute(many))
        padded = _payload(_WordBackend("perfect"))
        padded["candidates"]["chatml"]["records"] = [{"x": "A" * 700000} for _ in range(24)]
        self.assertIsNone(diffreports._prunesweep_recompute(padded))

    def test_mirror_returns_and_tail_skip(self):
        """The mirror's returns readout skips tail-incomplete records as designed."""
        payload = _payload(_WordBackend("perfect"))
        entry = payload["candidates"]["chatml"]
        record0 = entry["records"][0]
        record0["cells"]["dose18"]["verdict"] = False
        record0["cells"]["dose18"]["answer_digest"] = ps.digest_text("not-a-pool-value")
        record1 = entry["records"][1]
        record1["cells"]["tail72"] = None
        record1["cell_details"]["tail72"] = "arm_evaluation_failed"
        rebuilt = ps.aggregate_candidate(entry["records"])
        rebuilt.update(
            {"status": entry["status"], "detail": entry["detail"], "records": entry["records"]}
        )
        payload["candidates"]["chatml"] = rebuilt
        self.assertTrue(diffreports._valid_prune_sweep(payload))
        self.assertEqual(rebuilt["shape"]["returns"], 1)
        self.assertEqual(rebuilt["tail_contrast"]["n_compared"], 23)

    def test_reporting_malformed_candidates(self):
        """The section renderer tolerates a non-dict candidate entry."""
        self.assertEqual(
            reporting._prune_sweep_section({"prune_sweep": {"available": True, "candidates": "x"}}),
            "",
        )
        payload = _payload(_WordBackend("perfect"))
        payload["candidates"]["bad"] = 42
        payload["candidates"]["weird"] = {
            "status": "available",
            "detail": "",
            "arms": {"dose0": 42, "dose18": {"decomposition": "x"}},
            "headline": {},
            "shape": {},
            "tail_contrast": {},
            "ids": {"dose72": 42},
            "dropped_shares": {},
            "records": [],
        }
        section = reporting._prune_sweep_section({"prune_sweep": payload})
        self.assertIn("prune-sweep", section)


if __name__ == "__main__":
    unittest.main()
