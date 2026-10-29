"""Cycle 218: instruction-distance probe (``run
--instruction-distance``).

One frozen format directive governs a fixed fact+question
block on generated cases; six arms vary only where neutral
filler sits. The load-bearing pins, each citing the design
(docs/INSTRUCTIONDISTANCE_DESIGN.md, the candidate text as
amended by the fresh review, coordinator #59):

* Frozen material byte-for-byte (section 3): the constraint
  is 64 characters (amendment A: the candidate's printed
  "61 chars" was wrong), 24 cases (amendment A: not 20),
  topics absent from the neutral corpus as case-folded
  substrings, corpus digit-free.
* Grammar (section 5.2, amendments B2/B3): compliance is
  fullmatch ``\\A[0-9]+\\Z`` over the span, ASCII digits only;
  the 15 executed decidability spans from the adjudication
  classify identically from span text and from the recorded
  digests.
* Boundary disposition (amendment B1, negation section 4.2 as
  shipped): a boundary-missing compliant-correct cell is
  ``boundary_only`` -- never in D, never in S or T; other
  boundary-missing cells grammar-classify over the
  whole-output span.
* Amendment C: classes derive from digests (answer digest +
  non-digit-projection digest); the descriptive integers
  cannot move a class -- the adjudication's forgery case
  (digit_count 3 vs 10, identical digests) is detectable /
  inert here.
* The (10, 6, 5) rule is a total function of D, S, T
  (section 6); amendment D's ``arm_screened`` is in the
  per-cell vocabulary and counted in the aggregates.
* The stage fails closed (screen, budget, cancel, context)
  and the diff validator recomputes the decomposition from
  records alone.
"""

import copy
import re
import threading
import unittest
from unittest import mock

from model_protocol_lab import diffreports, experiments
from model_protocol_lab import instructiondistance as idm
from model_protocol_lab import reporting as reporting_mod
from model_protocol_lab.backends import Generation
from model_protocol_lab.core import Format
from model_protocol_lab.distraction import NEUTRAL_CORPUS

_CHATML = Format(
    name="chatml",
    bos="",
    user_prefix="u:",
    assistant_prefix="a:",
    turn_end="</s>",
)


class _WordBackend:
    """Word-token scripted backend with switchable behaviours.

    Modes: ``perfect`` (answer the planted value, digits only,
    on every arm), ``drop`` (on separated arms only -- the
    first sentence after the directive is not the fact --
    answer "The answer is {value}.", a noncompliant
    correct-value span), ``explode`` (generate raises).
    """

    fingerprint = "sha256:test-instruction-distance"

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
        found = re.findall(r"number is (\d+)\.", text)
        value = found[-1] if found else "000"
        answer = value
        if self.mode == "drop":
            idx = text.find(idm.CONSTRAINT)
            after = text[idx + len(idm.CONSTRAINT) :].strip() if idx >= 0 else ""
            if idx >= 0 and "number is" not in after.split(".")[0]:
                answer = f"The answer is {value}."
        boundary = self.encode("endmark")[0]
        out = list(self.encode(answer)) + [boundary]
        out = (out + [boundary] * max_new_tokens)[:max_new_tokens]
        return Generation(token_ids=tuple(out), reason="limit")


def _payload(backend=None, formats=(_CHATML,), seed=0, **stage_kwargs):
    """Run the instruction-distance stage over the scripted formats and return the payload."""
    backend = backend if backend is not None else _WordBackend()
    stage_kwargs.setdefault("max_context", 8192)
    stage = experiments.instruction_distance_stage(
        backend,
        formats,
        boundary=tuple(backend.encode("endmark")),
        seed=seed,
        **stage_kwargs,
    )
    return experiments.instruction_distance_analysis(stage, tuple(f.name for f in formats))


def _cell(span, expected, boundary=True):
    """A completed cell built from a span, exactly as the
    stage builds it (digest-bound primitives, amendment C).
    """
    return {
        "verdict": (span == expected) and boundary,
        "boundary_found": boundary,
        "boundary_text_match": None,
        "input_tokens": 3,
        "answer_digest": idm.digest_text(span),
        "first_digit_run_digest": idm.first_digit_run_digest(span),
        "nondigit_projection_digest": idm.nondigit_projection_digest(span),
        "answer_len": len(span),
        "digit_count": idm.digit_count(span),
    }


def _record(cells, details=None, expected="417"):
    """Build one synthetic record dict with the given cells and token digests."""
    tokens = {
        "adj": 20,
        "sep_256": 276,
        "sep_1024": 1044,
        "disp_256": 276,
        "disp_1024": 1044,
        "adj_after": 20,
    }
    return {
        "fixture": "instructiondistance-00",
        "topic": "harbor",
        "expected_digest": idm.digest_text(expected),
        "arm_tokens": dict(tokens),
        "id_digests": {arm: idm.digest_text(f"ids-{arm}") for arm in idm.ARM_KEYS},
        "cells": cells,
        "cell_details": details or dict.fromkeys(idm.ARM_KEYS, ""),
    }


def _full_cells(adj="417", sep="417", disp="417", expected="417", **boundary):
    """Build a complete six-arm cell map for a synthetic record."""
    cells = {
        "adj": _cell(adj, expected, boundary.get("adj", True)),
        "sep_256": _cell(sep, expected, boundary.get("sep", True)),
        "sep_1024": _cell(sep, expected, boundary.get("sep", True)),
        "disp_256": _cell(disp, expected, boundary.get("disp", True)),
        "disp_1024": _cell(disp, expected, boundary.get("disp", True)),
        "adj_after": _cell(adj, expected),
    }
    return cells


class FrozenMaterialTests(unittest.TestCase):
    """Design section 3 + amendment A."""

    def test_constraint_byte_exact_and_64_chars(self):
        # Amendment A: the candidate printed "(61 chars)"; the
        # executed length is 64 and the build freezes 64.
        """The frozen constraint text is byte-exact and exactly 64 characters long."""
        self.assertEqual(
            idm.CONSTRAINT,
            "Reply with only the number: no units, no sentence, nothing else.",
        )
        self.assertEqual(len(idm.CONSTRAINT), 64)

    def test_counts_floors_and_budgets(self):
        """Panel counts, doses, arm count and budget constants equal the frozen design values."""
        self.assertEqual(idm.CASE_COUNT, 24)
        self.assertEqual(idm.DOSES, (256, 1024))
        self.assertEqual(len(idm.ARM_KEYS), 6)
        self.assertEqual(idm.MAX_GROUND_RECORDS, 576)
        self.assertEqual(idm.GENERATION_BUDGET, 3456)
        self.assertEqual((idm.MIN_D, idm.MIN_S, idm.MIN_MARGIN), (10, 6, 5))
        self.assertEqual(idm.SIZING_ENCODE_BUDGET, 512)

    def test_topics_absent_from_neutral_corpus(self):
        # Section 3.1 pin: topics absent from the corpus as
        # case-folded substrings. The first candidate list had
        # four collisions (meadow/orchard/timber/hazel),
        # replaced before freezing.
        """None of the 24 frozen topics appears in the neutral corpus."""
        corpus = " ".join(NEUTRAL_CORPUS).casefold()
        self.assertEqual(len(idm.TOPICS), 24)
        for topic in idm.TOPICS:
            self.assertNotIn(topic.casefold(), corpus)

    def test_neutral_corpus_carries_no_digits(self):
        # The sizing/closed-book premise (adjudication side
        # finding 3): 71 sentences, zero digits of any kind.
        """The 71-sentence neutral corpus contains no digit characters."""
        self.assertEqual(len(NEUTRAL_CORPUS), 71)
        joined = "".join(NEUTRAL_CORPUS)
        self.assertFalse(any(ch.isdigit() for ch in joined))

    def test_generated_cases_shape(self):
        """generate_cases yields 24 cases with 24 distinct values."""
        cases = idm.generate_cases(0)
        self.assertEqual(len(cases), 24)
        self.assertEqual(len({case["value"] for case in cases}), 24)
        for index, case in enumerate(cases):
            self.assertEqual(case["fixture_id"], f"instructiondistance-{index:02d}")
            self.assertEqual(case["topic"], idm.TOPICS[index])
            self.assertTrue(100 <= int(case["value"]) <= 999)
            self.assertEqual(
                case["block"],
                f"The {case['topic']} number is {case['value']}.\n"
                f"What is the {case['topic']} number?",
            )
        self.assertEqual(idm.generate_cases(0), cases)

    def test_arm_layouts(self):
        """Arm layouts place the constraint and the block per the frozen arrangement."""
        case = idm.generate_cases(0)[0]
        block = case["block"]
        self.assertEqual(idm.arm_content(case, "adj"), idm.CONSTRAINT + "\n" + block)
        self.assertEqual(
            idm.arm_content(case, "sep_256", "FILL"),
            idm.CONSTRAINT + "\nFILL\n" + block,
        )
        self.assertEqual(
            idm.arm_content(case, "disp_1024", "FILL"),
            idm.CONSTRAINT + "\n" + block + "\nFILL",
        )
        self.assertEqual(idm.arm_content(case, "adj_after"), block + "\n" + idm.CONSTRAINT)

    def test_protocol_binds_rules_and_material(self):
        """protocol_object binds the sizing-rule digest and the frozen material."""
        protocol = idm.protocol_object(0, "fp")
        self.assertEqual(protocol["sizing_rule_sha256"], idm.digest_text(idm.SIZING_RULE_TEXT))
        self.assertEqual(
            protocol["screening_rule_sha256"], idm.digest_text(idm.SCREENING_RULE_TEXT)
        )
        self.assertEqual(protocol["ground_sha256"], idm.ground_sha256(0))
        # Amendment J: the sizing rule names the omitted
        # offset-verification phase rather than claiming the
        # needle transplant is verbatim.
        self.assertIn("no in-span offset-verification phase", idm.SIZING_RULE_TEXT)
        self.assertIn("arm_screened", idm.CELL_DETAILS)
        self.assertIn("arm_screened", idm.SCREENING_RULE_TEXT)


class GrammarTests(unittest.TestCase):
    """Design section 5.2 + amendments B2/B3/C: the
    adjudication's 15 executed decidability spans classify
    identically from the span and from the recorded digests.
    """

    SPANS = [
        ("417", "compliant_correct"),
        ("418", "compliant_wrong"),
        ("0417", "compliant_wrong"),
        ("14170", "compliant_wrong"),
        ("The answer is 417.", "noncompliant_correct_value"),
        ("417 crates", "noncompliant_correct_value"),
        ("1,417", "noncompliant_wrong"),
        ("I cannot find that number.", "noncompliant_wrong"),
        ("", "noncompliant_wrong"),
        ("417\n", "noncompliant_correct_value"),
        ("417 ", "noncompliant_correct_value"),
        ("٤١٧", "noncompliant_wrong"),
        ("４１７", "noncompliant_wrong"),
        ("٤١٧417", "noncompliant_correct_value"),
        ("١٢٣", "noncompliant_wrong"),
    ]

    def test_fifteen_decidability_spans_agree(self):
        """All fifteen decidability spans decompose to their expected classes."""
        for span, expected_class in self.SPANS:
            with self.subTest(span=span):
                self.assertEqual(idm.decompose_span(span, "417", True), expected_class)
                cell = _cell(span, "417")
                self.assertEqual(idm.decompose_cell(cell, idm.digest_text("417")), expected_class)

    def test_grammar_is_fullmatch_not_dollar_anchor(self):
        # Amendment B2: re.match(r"^[0-9]+$", "417\n") is True
        # in Python ($ matches before a trailing newline); the
        # frozen grammar is fullmatch, so the span is
        # noncompliant (its first run still keeps the value).
        """The span grammar is a fullmatch: a trailing newline or an empty span is not compliant."""
        self.assertFalse(idm.span_compliant("417\n"))
        self.assertTrue(idm.span_compliant("417"))
        self.assertFalse(idm.span_compliant(""))

    def test_digit_alphabet_is_ascii(self):
        # Amendment B3: str.isdigit() and \\d disagree on
        # "4²7" (superscript two); the pinned alphabet counts
        # ASCII only, and Unicode digits form no run.
        """The digit alphabet is ASCII only: superscripts and Arabic-Indic digits do not count."""
        self.assertEqual(idm.digit_count("4²7"), 2)
        self.assertEqual(idm.first_digit_run("4²7"), "4")
        self.assertEqual(idm.digit_count("٤١٧"), 0)
        self.assertIsNone(idm.first_digit_run("٤١٧"))
        self.assertEqual(idm.first_digit_run_digest("٤١٧"), "")

    def test_first_run_rulings(self):
        # Frozen rulings (section 5.2): grouping's first run is
        # not the value; a value inside a longer digit string
        # is compliant-wrong, never correct-value.
        """First-run and decomposition rulings for grouped and over-long digit runs."""
        self.assertEqual(idm.first_digit_run("1,417"), "1")
        self.assertEqual(idm.decompose_span("14170", "417", True), "compliant_wrong")

    def test_forgery_integers_cannot_move_a_class(self):
        # Amendment C / adjudication item 9: under the
        # candidate's bare-integer derivation, flipping
        # digit_count 3 -> 9 on this span (answer_len 9) flips
        # noncompliant_correct_value <-> compliant_wrong
        # undetectably. Here the class rides on digests, so
        # the tampered integers change nothing.
        """Forgery integers embedded in a span cannot move its decomposition class."""
        span = "ab 417 cd"
        self.assertEqual(len(span), 9)
        cell = _cell(span, "417")
        self.assertEqual(cell["digit_count"], 3)
        self.assertEqual(
            idm.decompose_cell(cell, idm.digest_text("417")), "noncompliant_correct_value"
        )
        forged = dict(cell, digit_count=9)
        self.assertEqual(
            idm.decompose_cell(forged, idm.digest_text("417")), "noncompliant_correct_value"
        )
        forged_empty = dict(cell, digit_count=0, answer_len=0)
        self.assertEqual(
            idm.decompose_cell(forged_empty, idm.digest_text("417")),
            "noncompliant_correct_value",
        )


class BoundaryDispositionTests(unittest.TestCase):
    """Amendment B1: the adjudication's item 6 -- three
    readings of boundary-missing membership gave three
    different labels on identical generations; the shipped
    disposition is negation section 4.2's.
    """

    def test_boundary_missing_classes(self):
        """With no boundary, spans decompose to boundary_only or compliant_wrong as ruled."""
        self.assertEqual(idm.decompose_span("417", "417", False), "boundary_only")
        self.assertEqual(idm.decompose_span("418", "417", False), "compliant_wrong")
        self.assertEqual(
            idm.decompose_span("The answer is 417.", "417", False),
            "noncompliant_correct_value",
        )
        self.assertEqual(idm.decompose_span("nope", "417", False), "noncompliant_wrong")

    def test_boundary_only_sep_is_not_a_loss(self):
        # 12 cases in D whose separated span is the correct
        # digits with the boundary missed: count-as-loss would
        # fire (S = 12); the shipped disposition reads S = 0.
        """A boundary-only separated cell is not counted as a loss in the aggregate."""
        records = [_record(_full_cells()) for _ in range(12)]
        for record in records:
            record["cells"]["sep_256"] = _cell("417", "417", boundary=False)
            record["cells"]["sep_1024"] = _cell("417", "417", boundary=False)
        agg = idm.aggregate_candidate(records)
        for dose in ("256", "1024"):
            self.assertEqual(agg["doses"][dose]["direct_passes"], 12)
            self.assertEqual(agg["doses"][dose]["separated_losses"], 0)
            self.assertEqual(agg["doses"][dose]["decision"], "below_threshold")
        self.assertEqual(agg["arms"]["sep_256"]["decomposition"]["boundary_only"], 12)

    def test_boundary_only_adj_never_enters_d(self):
        """A boundary-only adjacent cell never enters the D count."""
        records = [_record(_full_cells(adj="417")) for _ in range(12)]
        for record in records:
            record["cells"]["adj"] = _cell("417", "417", boundary=False)
        agg = idm.aggregate_candidate(records)
        for dose in ("256", "1024"):
            self.assertEqual(agg["doses"][dose]["direct_passes"], 0)
            self.assertEqual(agg["doses"][dose]["decision"], "insufficient")


class ClassifyTests(unittest.TestCase):
    """Design section 6: the (10, 6, 5) rule's edges."""

    def test_rule_boundaries(self):
        """classify_instruction_distance reproduces the frozen rule boundaries."""
        self.assertEqual(idm.classify_instruction_distance(9, 9, 0), "insufficient")
        self.assertEqual(
            idm.classify_instruction_distance(10, 6, 1), "instruction_distance_drop_observed"
        )
        self.assertEqual(idm.classify_instruction_distance(10, 6, 2), "below_threshold")
        self.assertEqual(idm.classify_instruction_distance(10, 5, 0), "below_threshold")
        self.assertEqual(
            idm.classify_instruction_distance(24, 24, 19),
            "instruction_distance_drop_observed",
        )
        # A pure constraint-to-generation effect yields S = T:
        # below_threshold even at total collapse (amendment F's
        # estimand restatement).
        self.assertEqual(idm.classify_instruction_distance(24, 24, 24), "below_threshold")


class AggregateTests(unittest.TestCase):
    """Aggregate recomputation tests over synthetic instruction-distance records."""

    def test_d_s_t_and_value_split(self):
        """The aggregate splits D into S and T and separates value-kept from value-lost losses."""
        records = []
        # 10 D cases: 6 sep losses (4 value-kept, 2 lost), 1 disp loss.
        for i in range(10):
            sep = "417"
            if i < 4:
                sep = "The answer is 417."
            elif i < 6:
                sep = "I cannot find that number."
            disp = "The answer is 417." if i == 0 else "417"
            records.append(_record(_full_cells(sep=sep, disp=disp)))
        agg = idm.aggregate_candidate(records)
        dose = agg["doses"]["256"]
        self.assertEqual(dose["n_population"], 10)
        self.assertEqual(dose["direct_passes"], 10)
        self.assertEqual(dose["separated_losses"], 6)
        self.assertEqual(dose["s_value_kept"], 4)
        self.assertEqual(dose["s_value_lost"], 2)
        self.assertEqual(dose["displaced_losses"], 1)
        self.assertEqual(dose["margin"], 5)
        self.assertEqual(dose["decision"], "instruction_distance_drop_observed")

    def test_population_requires_three_arm_completion(self):
        """A record enters the population only when the three compared arms completed."""
        cells = _full_cells()
        cells["disp_256"] = None
        details = dict.fromkeys(idm.ARM_KEYS, "")
        details["disp_256"] = "arm_evaluation_failed"
        agg = idm.aggregate_candidate([_record(cells, details)])
        self.assertEqual(agg["doses"]["256"]["n_population"], 0)
        self.assertEqual(agg["doses"]["1024"]["n_population"], 1)
        self.assertEqual(agg["n_incomplete"], 1)

    def test_reverse_discordants_counted_not_pooled(self):
        """Reverse discordants are counted in n_reverse, never pooled into the losses."""
        cells = _full_cells(adj="The answer is 417.", sep="417")
        agg = idm.aggregate_candidate([_record(cells)])
        self.assertEqual(agg["doses"]["256"]["n_reverse"], 1)
        # The case is not in D (adjacent not compliant-correct),
        # so the reverse discordant cannot soften S either.
        self.assertEqual(agg["doses"]["256"]["direct_passes"], 0)

    def test_ids_block_and_same_ids_flip(self):
        """The ids block flips when a record's separated ids coincide with the adjacent arm's."""
        record = _record(_full_cells(sep="The answer is 417."))
        record["id_digests"]["sep_256"] = record["id_digests"]["adj"]
        agg = idm.aggregate_candidate([record])
        ids = agg["ids"]["sep_256"]
        self.assertEqual(ids["n_ids_identical"], 1)
        self.assertEqual(ids["n_same_ids_compliance_flips"], 1)
        self.assertEqual(ids["id_length_delta_by_case"], [256])
        self.assertEqual(agg["ids"]["sep_1024"]["id_length_delta_by_case"], [1024])


class StageTests(unittest.TestCase):
    """Stage tests over scripted backends for the instruction-distance probe."""

    def test_perfect_backend_below_threshold(self):
        """A perfect backend yields an available payload classified below_threshold."""
        payload = _payload(_WordBackend("perfect"))
        self.assertTrue(payload["available"])
        cand = payload["candidates"]["chatml"]
        self.assertEqual(cand["n_cases"], 24)
        for dose in ("256", "1024"):
            block = cand["doses"][dose]
            self.assertEqual(block["direct_passes"], 24)
            self.assertEqual(block["separated_losses"], 0)
            self.assertEqual(block["decision"], "below_threshold")
        record = cand["records"][0]
        # Sizing exactness on the record itself: every sized
        # arm sits exactly its dose above the adjacent arm.
        tokens = record["arm_tokens"]
        for arm, dose in (
            ("sep_256", 256),
            ("sep_1024", 1024),
            ("disp_256", 256),
            ("disp_1024", 1024),
        ):
            self.assertEqual(tokens[arm] - tokens["adj"], dose)

    def test_drop_backend_fires_with_value_kept(self):
        """A drop backend fires the decision at both doses with the value kept."""
        payload = _payload(_WordBackend("drop"))
        cand = payload["candidates"]["chatml"]
        for dose in ("256", "1024"):
            block = cand["doses"][dose]
            self.assertEqual(block["separated_losses"], 24)
            self.assertEqual(block["s_value_kept"], 24)
            self.assertEqual(block["s_value_lost"], 0)
            self.assertEqual(block["displaced_losses"], 0)
            self.assertEqual(block["decision"], "instruction_distance_drop_observed")

    def test_explode_backend_scores_failed_never_crashes(self):
        """An exploding backend scores failed cells; the stage never crashes."""
        payload = _payload(_WordBackend("explode"))
        self.assertTrue(payload["available"])
        cand = payload["candidates"]["chatml"]
        self.assertEqual(cand["arms"]["adj"]["n_evaluation_failed"], 24)
        self.assertEqual(cand["doses"]["256"]["decision"], "insufficient")

    def test_context_exceeded_sized_arms_carry_no_construction(self):
        # adj renders ~23 word tokens; a 200-token context fits
        # adj + generation but never adj + 256 + generation.
        """Context-exceeded sized arms carry no construction in their records."""
        payload = _payload(_WordBackend("perfect"), max_context=200)
        cand = payload["candidates"]["chatml"]
        record = cand["records"][0]
        for arm in idm.SIZED_ARMS:
            self.assertEqual(record["cell_details"][arm], "arm_context_exceeded")
            self.assertIsNone(record["arm_tokens"][arm])
            self.assertIsNone(record["id_digests"][arm])
        self.assertIsNotNone(record["cells"]["adj"])
        self.assertEqual(cand["arms"]["sep_256"]["n_context_exceeded"], 24)

    def test_cancel_fails_closed(self):
        """A pre-set cancel event fails the stage closed as cancelled."""
        event = threading.Event()
        event.set()
        payload = _payload(_WordBackend("perfect"), cancel=event)
        self.assertFalse(payload["available"])
        self.assertEqual(payload["detail"], "cancelled")

    def test_boundary_text_in_generated_text_fails_stage(self):
        # The decoded boundary text "The" occurs in the frozen
        # fact template: the whole stage screens closed before
        # any generation (screening rule, stage level).
        """Boundary text inside generated text fails the stage screen."""
        backend = _WordBackend("perfect")
        stage = experiments.instruction_distance_stage(
            backend,
            (_CHATML,),
            boundary=tuple(backend.encode("The")),
            max_context=8192,
            seed=0,
        )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "instruction_distance_screen_failed")

    def test_screened_spans_are_per_cell(self):
        # Amendment D: with the span screen forced on, sized
        # cells land arm_screened (counted per arm), the stage
        # itself stays available, and unsized arms still run.
        """Filler screening applies per cell across the sized arms."""
        with mock.patch.object(idm, "filler_screened", return_value=True):
            payload = _payload(_WordBackend("perfect"))
        cand = payload["candidates"]["chatml"]
        for arm in idm.SIZED_ARMS:
            self.assertEqual(cand["arms"][arm]["n_screened"], 24)
            self.assertEqual(cand["arms"][arm]["n_completed"], 0)
        self.assertEqual(cand["arms"]["adj"]["n_completed"], 24)
        self.assertEqual(cand["doses"]["256"]["decision"], "insufficient")


class ValidatorTests(unittest.TestCase):
    """The diff validator recomputes the decomposition from
    records (cycle-190 digest-binding discipline).
    """

    @classmethod
    def setUpClass(cls):
        """Build the shared drop payload for the validator tests."""
        cls.payload = _payload(_WordBackend("drop"))

    def test_real_payload_is_valid(self):
        """A real payload validates under the diff mirror."""
        self.assertTrue(diffreports._valid_instruction_distance(self.payload))

    def test_tampered_aggregate_is_invalid(self):
        """A tampered aggregate count is rejected by the mirror."""
        bad = copy.deepcopy(self.payload)
        bad["candidates"]["chatml"]["doses"]["256"]["separated_losses"] = 0
        self.assertFalse(diffreports._valid_instruction_distance(bad))

    def test_tampered_protocol_is_invalid(self):
        """A tampered protocol field is rejected by the mirror."""
        bad = copy.deepcopy(self.payload)
        bad["protocol"]["min_s"] = 2
        self.assertFalse(diffreports._valid_instruction_distance(bad))

    def test_tampered_record_topic_is_invalid(self):
        """A record topic outside the frozen topics is rejected by the mirror."""
        bad = copy.deepcopy(self.payload)
        bad["candidates"]["chatml"]["records"][0]["topic"] = "quartz"
        self.assertFalse(diffreports._valid_instruction_distance(bad))

    def test_tampered_sized_delta_is_invalid(self):
        # The sizing acceptance identity is checked on the
        # stored counts: one token added to a separated arm
        # breaks the exact delta and fails validation.
        """A tampered sized-arm token delta is rejected by the mirror."""
        bad = copy.deepcopy(self.payload)
        bad["candidates"]["chatml"]["records"][0]["arm_tokens"]["sep_256"] += 1
        self.assertFalse(diffreports._valid_instruction_distance(bad))

    def test_tampered_cell_verdict_is_invalid(self):
        """A flipped cell verdict is rejected by the mirror."""
        bad = copy.deepcopy(self.payload)
        cell = bad["candidates"]["chatml"]["records"][0]["cells"]["adj"]
        cell["verdict"] = not cell["verdict"]
        self.assertFalse(diffreports._valid_instruction_distance(bad))

    def test_digit_count_forgery_cannot_move_the_diff(self):
        # Amendment C at payload level: tampering a
        # descriptive integer leaves every recomputed class --
        # and therefore the whole diff -- untouched, where the
        # candidate's bare-integer derivation would have
        # flipped the class undetectably.
        """A forged digit count cannot move the diff verdict."""
        forged = copy.deepcopy(self.payload)
        cell = forged["candidates"]["chatml"]["records"][0]["cells"]["sep_256"]
        self.assertEqual(cell["digit_count"], 3)
        cell["digit_count"] = 9  # "The answer is 417." has answer_len 18
        self.assertTrue(diffreports._valid_instruction_distance(forged))
        diff = diffreports.diff_instruction_distance(self.payload, forged)
        self.assertFalse(diff["changed"])

    def test_diff_decision_drift(self):
        """Diffing perfect against drop payloads reports the decision drift."""
        perfect = _payload(_WordBackend("perfect"))
        diff = diffreports.diff_instruction_distance(perfect, self.payload)
        self.assertEqual(diff["available"], "both")
        self.assertTrue(diff["changed"])
        metrics = {entry["metric"] for entry in diff["candidate_drift"]}
        self.assertIn("decision_256", metrics)
        self.assertIn("decision_1024", metrics)
        same = diffreports.diff_instruction_distance(self.payload, self.payload)
        self.assertFalse(same["changed"])

    def test_diff_reports_wiring(self):
        """The instruction-distance section is wired into the top-level diff_reports."""
        old = {"decision": "abstain", "instruction_distance": _payload(_WordBackend("perfect"))}
        new = {"decision": "abstain", "instruction_distance": self.payload}
        diff = diffreports.diff_reports(old, new)
        self.assertEqual(diff["instruction_distance"]["available"], "both")
        self.assertTrue(diff["summary"]["instruction_distance_changed"])
        self.assertFalse(diff["summary"]["identical"])
        empty = diffreports.diff_reports({"decision": "abstain"}, {"decision": "abstain"})
        self.assertEqual(empty["instruction_distance"]["available"], "neither")
        self.assertFalse(empty["summary"]["instruction_distance_changed"])


class ReportingTests(unittest.TestCase):
    """Report-section tests for the instruction-distance probe."""

    def test_section_renders_decision_and_oc(self):
        """The section renders the decision and the operating-characteristics paragraph."""
        payload = _payload(_WordBackend("drop"))
        html = reporting_mod.render_html(
            {"instruction_distance": payload, "decision": "x", "search": [], "held_out": []}
        )
        self.assertIn('<section class="card" id="instruction-distance">', html)
        self.assertIn('href="#instruction-distance"', html)
        self.assertIn("instruction_distance_drop_observed", html)
        # Amendment E figures ride beside every verdict.
        self.assertIn("0.28 false labels", html)
        self.assertIn("13.3%", html)
        self.assertIn("3.78 labels", html)
        # Amendment F: the 0.25 row is labelled below
        # resolution; no curve/threshold language anywhere.
        self.assertIn("below this panel's resolution", html)
        self.assertIn("24.6% at a 25% drop rate", html)
        section = html.split('id="instruction-distance"', 1)[1]
        self.assertNotIn("decay", section)
        self.assertNotIn("curve", section)

    def test_section_absent_without_payload(self):
        """The section is absent when the payload is missing."""
        html = reporting_mod.render_html({"decision": "x", "search": [], "held_out": []})
        self.assertNotIn('id="instruction-distance"', html)


class GapModuleTests(unittest.TestCase):
    """Cycle-218 gap pass (cycle-174 rule): the full-suite
    coverage misses inside instructiondistance.py, closed with
    targeted cases.
    """

    def test_generate_cases_rejects_non_int_seed(self):
        """generate_cases rejects a non-int seed with ValueError."""
        with self.assertRaises(ValueError):
            idm.generate_cases("0")

    def test_unknown_arm_raises(self):
        """Unknown arm names raise ValueError in the module helpers."""
        case = idm.generate_cases(0)[0]
        with self.assertRaises(ValueError):
            idm.arm_content(case, "bogus", "")
        with self.assertRaises(ValueError):
            idm.arm_dose("bogus")

    def test_filler_screened_boundary_text_path(self):
        # The value path is stage-tested; the boundary-text
        # path is the second disjunct of the screen.
        """filler_screened flags filler containing the boundary text and passes plain filler."""
        self.assertTrue(idm.filler_screened("the endmark sits here", "zzz", "endmark"))
        self.assertFalse(idm.filler_screened("plain filler", "zzz", "endmark"))

    def test_decompose_cell_requires_a_cell(self):
        """decompose_cell rejects a missing cell with ValueError."""
        with self.assertRaises(ValueError):
            idm.decompose_cell(None, idm.digest_text("417"))

    def test_aggregate_attrition_per_detail(self):
        """The aggregate reports attrition per cell detail across the sized arms."""
        cells = _full_cells()
        record = _record(
            cells,
            details={
                "adj": "",
                "sep_256": "arm_unmatchable",
                "sep_1024": "arm_context_exceeded",
                "disp_256": "arm_screened",
                "disp_1024": "arm_evaluation_failed",
                "adj_after": "",
            },
        )
        for arm in ("sep_256", "sep_1024", "disp_256", "disp_1024"):
            record["cells"][arm] = None
        agg = idm.aggregate_candidate([record])
        arms = agg["arms"]
        self.assertEqual(arms["sep_256"]["n_unmatchable"], 1)
        self.assertEqual(arms["sep_1024"]["n_context_exceeded"], 1)
        self.assertEqual(arms["disp_256"]["n_screened"], 1)
        self.assertEqual(arms["disp_1024"]["n_evaluation_failed"], 1)

    def test_aggregate_tolerates_non_dict_cells_and_details(self):
        """The aggregate tolerates non-dict cells and details without raising."""
        record = _record(_full_cells())
        record["cells"] = []
        record["cell_details"] = []
        agg = idm.aggregate_candidate([record])
        self.assertEqual(agg["n_cases"], 1)
        self.assertEqual(agg["doses"]["256"]["n_population"], 0)

    def test_same_ids_flip_counted(self):
        """A same-ids flip between the adjacent and separated arms is counted in the aggregate."""
        record = _record(_full_cells(adj="417", sep="The answer is 417."))
        record["id_digests"]["sep_256"] = record["id_digests"]["adj"]
        agg = idm.aggregate_candidate([record])
        ids = agg["ids"]["sep_256"]
        self.assertEqual(ids["n_ids_identical"], 1)
        self.assertEqual(ids["n_same_ids_compliance_flips"], 1)


class GapStageTests(unittest.TestCase):
    """Cycle-218 gap pass: stage fail-closed branches the
    happy-path stage tests never reach.
    """

    def test_formats_must_be_a_nonempty_tuple(self):
        """Formats must be a non-empty tuple; anything else raises ValueError."""
        backend = _WordBackend()
        with self.assertRaises(ValueError):
            experiments.instruction_distance_stage(
                backend, [_CHATML], boundary=(1,), seed=0, max_context=8192
            )
        with self.assertRaises(ValueError):
            experiments.instruction_distance_stage(
                backend, (), boundary=(1,), seed=0, max_context=8192
            )

    def test_duplicate_format_names_raise(self):
        """Duplicate format names raise ValueError."""
        with self.assertRaises(ValueError):
            experiments.instruction_distance_stage(
                _WordBackend(),
                (_CHATML, _CHATML),
                boundary=(1,),
                seed=0,
                max_context=8192,
            )

    def test_record_budget_exceeded_fails_closed(self):
        """Exceeding the ground-record budget fails the stage closed with the budget detail."""
        with mock.patch.object(idm, "MAX_GROUND_RECORDS", 10):
            stage = experiments.instruction_distance_stage(
                _WordBackend(), (_CHATML,), boundary=(1,), seed=0, max_context=8192
            )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "instruction_distance_record_budget_exceeded")

    def test_special_token_finding_fails_the_stage_closed(self):
        """A special-token finding fails the stage closed with the screen detail."""
        with mock.patch(
            "model_protocol_lab.fixtures.audit_special_token_literals",
            return_value=[{"literal": "<|x|>"}],
        ):
            stage = experiments.instruction_distance_stage(
                _WordBackend(), (_CHATML,), boundary=(1,), seed=0, max_context=8192
            )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "instruction_distance_screen_failed")
        self.assertEqual(stage["audit"]["special_token_findings"], 1)

    def test_undecodable_boundary_screens_nothing(self):
        """An undecodable boundary decode screens nothing and the payload stays available."""

        class _BadBoundaryDecode(_WordBackend):
            """Word backend whose decode raises on the boundary ids."""

            def decode(self, ids):
                """Decode that raises RuntimeError on the boundary (endmark) ids, delegating
                otherwise.
                """
                if tuple(ids) == tuple(self.encode("endmark")):
                    raise RuntimeError("undecodable")
                return super().decode(ids)

        payload = _payload(_BadBoundaryDecode())
        self.assertTrue(payload["available"])

    def test_sizing_budget_zero_makes_sized_cells_unmatchable(self):
        """A zero sizing-encode budget leaves the sized cells unmatchable; the payload stays
        available.
        """
        with mock.patch.object(idm, "SIZING_ENCODE_BUDGET", 0):
            payload = _payload(_WordBackend())
        self.assertTrue(payload["available"])
        arms = payload["candidates"]["chatml"]["arms"]
        self.assertEqual(arms["sep_256"]["n_unmatchable"], 24)
        record = payload["candidates"]["chatml"]["records"][0]
        self.assertIsNone(record["arm_tokens"]["sep_256"])
        self.assertIsNone(record["id_digests"]["sep_256"])

    def test_sizing_exhaustion_makes_sized_cells_unmatchable(self):
        """Sizing-encode exhaustion leaves the sized cells unmatchable."""
        with mock.patch(
            "model_protocol_lab.distraction.span_candidates",
            lambda stream, target, chars_per_token: [(0, "", 0)],
        ):
            payload = _payload(_WordBackend())
        self.assertTrue(payload["available"])
        arms = payload["candidates"]["chatml"]["arms"]
        self.assertEqual(arms["sep_1024"]["n_unmatchable"], 24)

    def test_midflight_cancel_unwinds_to_cancelled(self):
        """A cancel event set mid-generation unwinds the stage to cancelled."""
        event = threading.Event()

        class _CancellingBackend(_WordBackend):
            """Word backend whose generate sets the cancel event before delegating."""

            def generate(self, input_ids, *, max_new_tokens, seed=None, cancel=None):
                """Generate that sets the cancel event, then delegates to the scripted backend."""
                event.set()
                return super().generate(
                    input_ids, max_new_tokens=max_new_tokens, seed=seed, cancel=cancel
                )

        payload = _payload(_CancellingBackend(), cancel=event)
        self.assertFalse(payload["available"])
        self.assertEqual(payload["detail"], "cancelled")

    def test_cell_cancel_checks_and_ineligible_report(self):
        """Per-cell cancel checks fire and the ineligible report is produced."""
        from model_protocol_lab.core import Fixture, Message

        backend = _WordBackend()
        fixture = Fixture(
            id="gap-cell",
            messages=(Message("user", "hi"),),
            expected="417",
            group=idm.GROUP,
        )
        ids = tuple(backend.encode("hi"))
        boundary = tuple(backend.encode("endmark"))
        preset = threading.Event()
        preset.set()
        with self.assertRaises(experiments._InstructionDistanceCancelled):
            experiments._instruction_distance_cell(
                backend,
                _CHATML,
                fixture,
                ids,
                boundary_ids=boundary,
                max_new_tokens=64,
                max_context=8192,
                seed=0,
                cancel=preset,
            )

        fired = threading.Event()

        class _TripBackend(_WordBackend):
            """Word backend whose generate sets a trip event after delegating."""

            def generate(self, input_ids, *, max_new_tokens, seed=None, cancel=None):
                """Generate via the scripted backend, then set the trip event."""
                out = super().generate(
                    input_ids, max_new_tokens=max_new_tokens, seed=seed, cancel=cancel
                )
                fired.set()
                return out

        with self.assertRaises(experiments._InstructionDistanceCancelled):
            experiments._instruction_distance_cell(
                _TripBackend(),
                _CHATML,
                fixture,
                ids,
                boundary_ids=boundary,
                max_new_tokens=64,
                max_context=8192,
                seed=0,
                cancel=fired,
            )

        stub = mock.Mock(aggregate_eligible=False, observations=[])
        with mock.patch.object(experiments, "evaluate", return_value=stub):
            detail, cell = experiments._instruction_distance_cell(
                backend,
                _CHATML,
                fixture,
                ids,
                boundary_ids=boundary,
                max_new_tokens=64,
                max_context=8192,
                seed=0,
                cancel=None,
            )
        self.assertEqual(detail, "arm_evaluation_failed")
        self.assertIsNone(cell)

    def test_analysis_unavailable_stage_and_candidate(self):
        """The analysis passes an unavailable stage and candidate through, detail intact."""
        out = experiments.instruction_distance_analysis(
            {"available": False, "detail": "cancelled", "protocol": None, "audit": None},
            ("chatml",),
        )
        self.assertFalse(out["available"])
        self.assertEqual(out["detail"], "cancelled")
        backend = _WordBackend()
        stage = experiments.instruction_distance_stage(
            backend,
            (_CHATML,),
            boundary=tuple(backend.encode("endmark")),
            seed=0,
            max_context=8192,
        )
        stage["candidates"]["chatml"]["status"] = "unavailable"
        stage["candidates"]["chatml"]["detail"] = "arm_evaluation_failed"
        out = experiments.instruction_distance_analysis(stage, ("chatml",))
        self.assertEqual(out["candidates"]["chatml"]["status"], "unavailable")
        self.assertEqual(out["candidates"]["chatml"]["records"], [])


class GapValidatorTests(unittest.TestCase):
    """Cycle-218 gap pass: one-broken-identity-at-a-time
    battery over the diff validator's reject branches
    (cycle-174 fix pattern).
    """

    @classmethod
    def setUpClass(cls):
        """Build the shared drop payload for the gap validator battery."""
        cls.payload = _payload(_WordBackend("drop"))

    def _bad(self, mutate):
        """Assert the mirror rejects the shared payload after the given forgery."""
        bad = copy.deepcopy(self.payload)
        mutate(bad)
        self.assertFalse(diffreports._valid_instruction_distance(bad))

    def test_validator_battery(self):
        """Every forgery in the battery makes the mirror reject the payload."""
        cand = lambda p: p["candidates"]["chatml"]  # noqa: E731
        rec = lambda p: cand(p)["records"][0]  # noqa: E731
        cell = lambda p: rec(p)["cells"]["adj"]  # noqa: E731
        cases = {
            "payload detail": lambda p: p.update(detail="x"),
            "payload extra key": lambda p: p.update(extra=1),
            "n_candidates drift": lambda p: p.update(n_candidates=5),
            "n_cases drift": lambda p: p.update(n_cases=3),
            "audit key dropped": lambda p: p["audit"].pop("by_reason"),
            "audit negative excluded": lambda p: p["audit"].update(n_audit_excluded=-1),
            "audit findings not int": lambda p: p["audit"].update(special_token_findings="1"),
            "audit by_reason negative": lambda p: p["audit"].update(by_reason={"x": -1}),
            "protocol seed not int": lambda p: p["protocol"].update(seed="0"),
            "record fixture renamed": lambda p: rec(p).update(fixture="instructiondistance-99"),
            "record key dropped": lambda p: rec(p).pop("topic"),
            "record expected drift": lambda p: rec(p).update(
                expected_digest=idm.digest_text("999")
            ),
            "arm_tokens not dict": lambda p: rec(p).update(arm_tokens=[]),
            "adj tokens none": lambda p: rec(p)["arm_tokens"].update(adj=None),
            "sized tokens not int": lambda p: rec(p)["arm_tokens"].update(sep_256="279"),
            "sized digest dropped": lambda p: rec(p)["id_digests"].update(sep_256=None),
            "sized tokens dropped": lambda p: rec(p)["arm_tokens"].update(sep_256=None),
            "cell answer digest shape": lambda p: cell(p).update(answer_digest="xyz"),
            "cell run digest shape": lambda p: cell(p).update(first_digit_run_digest="zz"),
            "cell projection digest shape": lambda p: cell(p).update(
                nondigit_projection_digest="zz"
            ),
            "cell verdict not bool": lambda p: cell(p).update(verdict="yes"),
            "cell boundary not bool": lambda p: cell(p).update(boundary_found="yes"),
            "cell boundary match type": lambda p: cell(p).update(boundary_text_match=5),
            "cell input tokens negative": lambda p: cell(p).update(input_tokens=-1),
            "cell input tokens binding": lambda p: cell(p).update(
                input_tokens=cell(p)["input_tokens"] + 1
            ),
            "cell answer_len negative": lambda p: cell(p).update(answer_len=-1),
            "cell digit_count over len": lambda p: cell(p).update(digit_count=99),
            "cell invented detail": lambda p: rec(p)["cell_details"].update(adj="bogus"),
            "records shortened": lambda p: cand(p).update(records=cand(p)["records"][:23]),
            "arms aggregate drift": lambda p: cand(p)["arms"]["adj"].update(n_completed=0),
        }
        for name, mutate in cases.items():
            with self.subTest(case=name):
                self._bad(mutate)

    def test_diff_with_insufficient_payload(self):
        """Diffing against an insufficient (exploded) payload reports both-available."""
        exploded = _payload(_WordBackend("explode"))
        diff = diffreports.diff_instruction_distance(self.payload, exploded)
        self.assertEqual(diff["available"], "both")
        metrics = {entry["metric"]: entry for entry in diff["candidate_drift"]}
        self.assertEqual(metrics["decision_256"]["new"], "insufficient")

    def test_diff_asymmetric_availability(self):
        """Diffing with an unavailable side reports the asymmetric availability."""
        unavailable = {"available": False, "detail": "instruction_distance_unavailable"}
        self.assertEqual(
            diffreports.diff_instruction_distance(self.payload, unavailable)["available"],
            "old",
        )
        self.assertEqual(
            diffreports.diff_instruction_distance(unavailable, self.payload)["available"],
            "new",
        )
        self.assertEqual(
            diffreports.diff_instruction_distance(unavailable, unavailable)["available"],
            "neither",
        )


class GapReportingTests(unittest.TestCase):
    """Cycle-218 gap pass: the section's defensive branches
    over hostile payload shapes.
    """

    def test_section_hostile_payload(self):
        """The section renderer tolerates a hostile candidate payload without raising."""
        payload = _payload(_WordBackend("drop"))
        hostile = copy.deepcopy(payload)
        entry = hostile["candidates"]["chatml"]
        entry["doses"] = {"256": "bogus", "1024": entry["doses"]["1024"]}
        entry["ids"] = {"sep_256": "bogus", **entry["ids"]}
        entry["arms"]["adj"]["compliance_rate"] = "high"
        hostile["candidates"]["aaa_bogus"] = 42
        hostile["candidates"]["zzz_unavailable"] = {
            "status": "unavailable",
            "detail": "arm_evaluation_failed",
            "records": [],
        }
        html = reporting_mod.render_html(
            {"instruction_distance": hostile, "decision": "x", "search": [], "held_out": []}
        )
        self.assertIn('id="instruction-distance"', html)
        self.assertIn("unavailable", html)

    def test_section_candidates_not_a_dict(self):
        """A non-dict candidates payload renders no instruction-distance section."""
        html = reporting_mod.render_html(
            {
                "instruction_distance": {
                    "available": True,
                    "detail": "",
                    "candidates": [],
                },
                "decision": "x",
                "search": [],
                "held_out": [],
            }
        )
        self.assertNotIn('id="instruction-distance"', html)


if __name__ == "__main__":
    unittest.main()
