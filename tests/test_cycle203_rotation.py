"""Cycle 203: context rotation (wave-9 rank 1).

Pins the frozen contract in docs/ROTATION_DESIGN.md (design
v2, after red-team A's DO-NOT-BUILD verdict on v1 -- its two
FATAL findings were the any-of-five vs single-pair
cardinality mismatch in the classification's step 4 and the
all-fail/all-pass conflation in the invariance label -- and
red-team B's cross-check, which confirmed A's substantive
core by independent execution under alternative nulls,
corrected A's F1 fix and F4 witness, downgraded F2 to a
labelling defect, and added the reversal slot-displacement
confound (B-F1) and the arms-table labelling gap (B-F2); the
coordinator's adjudication adopted the matched-cardinality
mirror control set, the dominance step, the per-arm inventory
digest, and the slot-multiset answer to B-F1): the
seed-derived case generator and its redraw rules, the pool
pins, the arm orders and asked slots, the token-inventory
construction classification (derived, never stored), the
total classification function with its published joint
falsification table (pinned by enumeration), the stage's
fail-closed vocabulary and its two screening layers, the
full-recompute diff validator (one break at a time) with its
literal-table mirror, the diff, the reporting section, and
the CLI helper.
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import re
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from model_protocol_lab import backends, cli, diffreports, experiments, rotation
from model_protocol_lab import doctor as doctor_mod
from model_protocol_lab import reporting as reporting_mod
from model_protocol_lab.core import Format, Generation
from model_protocol_lab.diffreports import _valid_rotation, diff_rotation
from model_protocol_lab.reporting import render_html

FORMATS = (
    Format(name="rt-a", bos="", user_prefix="u:", assistant_prefix="a:", turn_end="</s>"),
    Format(name="rt-b", bos="", user_prefix="U>", assistant_prefix="A>", turn_end="<e>"),
)

_FACT_RE = re.compile(r"The (.+?) access code is ([a-z]+-\d+)\.")
_QUESTION_RE = re.compile(r"What is the (.+?) access code\?")


class _WordBackend:
    """Word-token scripted backend with switchable behaviours.

    Modes: ``perfect`` (answer the asked topic's planted code
    wherever it sits), ``slot_fail`` (answer wrongly whenever
    the asked fact is not the first block of the span -- a pure
    slot effect, which the mirror control set must absorb),
    ``wrong`` (every answer wrong), ``explode`` (generate
    raises). ``known`` maps a topic to a memorized value
    answered even with no fact in view -- the closed-book
    screen trigger. Word-level encoding makes every rotation
    of a case render to the same token multiset, so the
    inventory construction is exact for every span arm.
    """

    fingerprint = "sha256:test-rotation"

    def __init__(
        self,
        mode: str = "perfect",
        cancel_after: int | None = None,
        known: dict[str, str] | None = None,
    ):
        self.mode = mode
        self.cancel_after = cancel_after
        self.cancel_event: threading.Event | None = None
        self.known = known or {}
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
        if (
            self.cancel_after is not None
            and self.cancel_event is not None
            and self.calls >= self.cancel_after
        ):
            self.cancel_event.set()
        text = self.decode(list(input_ids))
        facts = _FACT_RE.findall(text)
        asked_match = _QUESTION_RE.search(text)
        asked = asked_match.group(1) if asked_match else ""
        answer = "unknown"
        for topic, value in facts:
            if topic == asked:
                answer = value
                if self.mode == "wrong" or (self.mode == "slot_fail" and facts[0][0] != asked):
                    answer = "wrong-0000"
                break
        else:
            if asked in self.known:
                answer = self.known[asked]
            elif self.mode == "wrong" and facts:
                answer = "wrong-0000"
        boundary = self.encode("endmark")[0]
        out = list(self.encode(answer)) + [boundary]
        out = (out + [boundary] * max_new_tokens)[:max_new_tokens]
        return Generation(token_ids=tuple(out), reason="limit")


def _payload(backend=None, formats=FORMATS, seed=0, **stage_kwargs):
    backend = backend if backend is not None else _WordBackend()
    stage_kwargs.setdefault("max_context", 4096)
    stage = experiments.rotation_stage(
        backend,
        formats,
        boundary=tuple(backend.encode("endmark")),
        seed=seed,
        **stage_kwargs,
    )
    return experiments.rotation_analysis(stage, tuple(f.name for f in formats))


def _payload_with_unavailable():
    """A coherent payload whose second candidate is an
    unavailable entry, built through the real analysis so the
    rest of the payload stays consistent."""
    backend = _WordBackend()
    stage = experiments.rotation_stage(
        backend,
        FORMATS,
        boundary=tuple(backend.encode("endmark")),
        max_context=4096,
        seed=0,
    )
    stage["candidates"]["rt-b"] = {
        "status": "unavailable",
        "detail": "candidate_evaluation_failed",
        "records": [],
    }
    return experiments.rotation_analysis(stage, tuple(f.name for f in FORMATS))


class TestProtocol(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(rotation.CASE_COUNT, 20)
        self.assertEqual(len(rotation.ARM_KEYS), 11)
        self.assertEqual(rotation.GENERATION_BUDGET, 5280)
        self.assertEqual(rotation.MAX_GROUND_RECORDS, 480)
        self.assertEqual(rotation.MAX_RECORDS_BYTES, 12 * 1024 * 1024)
        self.assertEqual(rotation.MIN_HEADLINE_POPULATION, 6)
        self.assertEqual(rotation.MIN_CONTROL_POPULATION, 6)
        self.assertEqual(
            rotation.CELL_DETAILS,
            ("", "arm_context_exceeded", "arm_evaluation_failed", "arm_screened"),
        )
        self.assertEqual(
            rotation.STAGE_DETAILS,
            ("cancelled", "rotation_record_budget_exceeded", "rotation_screen_failed"),
        )

    def test_protocol_object_and_fingerprint(self):
        protocol = rotation.protocol_object(3, "sha256:x")
        self.assertEqual(protocol["seed"], 3)
        self.assertEqual(protocol["backend_fingerprint"], "sha256:x")
        self.assertEqual(protocol["case_count"], 20)
        self.assertEqual(protocol["ground_sha256"], rotation.ground_sha256(3))
        fingerprint = rotation.protocol_fingerprint(3, "sha256:x")
        self.assertEqual(len(fingerprint), 64)
        self.assertNotEqual(fingerprint, rotation.protocol_fingerprint(4, "sha256:x"))
        self.assertNotEqual(fingerprint, rotation.protocol_fingerprint(3, "sha256:y"))


class TestGenerator(unittest.TestCase):
    def test_deterministic_and_seed_sensitive(self):
        self.assertEqual(rotation.generate_cases(0), rotation.generate_cases(0))
        self.assertNotEqual(rotation.generate_cases(0), rotation.generate_cases(1))

    def test_case_shape(self):
        for seed in (0, 1, 7):
            cases = rotation.generate_cases(seed)
            self.assertEqual(len(cases), 20)
            self.assertEqual(
                [c["fixture_id"] for c in cases],
                [f"rotation-{i:02d}" for i in range(20)],
            )
            asked_topics = set()
            asked_values = set()
            for case in cases:
                self.assertEqual(len(case["topics"]), 6)
                self.assertEqual(len(set(case["topics"])), 6)
                self.assertEqual(case["topic"], case["topics"][0])
                self.assertEqual(len(case["values"]), 6)
                self.assertEqual(len(set(case["values"])), 6)
                asked_topics.add(case["topics"][0])
                asked_values.add(case["values"][0])
            # Without replacement across cases, both streams.
            self.assertEqual(len(asked_topics), 20)
            self.assertEqual(len(asked_values), 20)

    def test_seed_must_be_int(self):
        with self.assertRaises(ValueError):
            rotation.generate_cases(True)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            rotation.generate_cases("0")  # type: ignore[arg-type]

    def test_redraw_paths_exercised(self):
        # Across a seed sweep, the topic redraw rules must
        # actually fire (a frozen rule that never fires is an
        # untested branch wearing a pin's clothes): the
        # asked-topic redraw (slot 20, counter > 0) and a
        # distractor redraw (slots 21..25, counter > 0). The
        # value redraw is too rare to reach naturally (the
        # value space is ~576k per block), so it gets its own
        # scripted test below.
        seen: set[tuple[int, int]] = set()
        original = rotation._derive

        def counting(seed, index, slot, counter):
            if counter > 0:
                seen.add((slot, counter))
            return original(seed, index, slot, counter)

        with mock.patch.object(rotation, "_derive", counting):
            for seed in range(50):
                rotation.generate_cases(seed)
        self.assertTrue(any(slot == 20 for slot, _ in seen), "asked redraw never fired")
        self.assertTrue(any(21 <= slot <= 25 for slot, _ in seen), "distractor redraw never fired")

    def test_value_redraw_scripted_collision(self):
        # Force blocks 0 and 1 of case 0 to draw the same value
        # at counter 0: the frozen rule increments ALL six
        # counters and redraws all six values, leaving the
        # topic stream untouched.
        calls: list[tuple[int, int, int]] = []

        def scripted(seed, index, slot, counter):
            calls.append((index, slot, counter))
            if slot == 20:
                return index
            if 21 <= slot <= 25:
                return index * 6 + slot + 1000 * counter
            if index == 0 and counter == 0 and slot in (0, 1, 2, 3):
                return 0  # blocks 0 and 1 collide: WORDS[0]-1000 twice
            return slot * 37 + index * 101 + counter * 13 + 5

        with mock.patch.object(rotation, "_derive", scripted):
            cases = rotation.generate_cases(0)
        self.assertTrue(
            any(slot < 20 and counter > 0 for _, slot, counter in calls),
            "value redraw never fired under the scripted collision",
        )
        case0 = cases[0]
        self.assertEqual(len(set(case0["values"])), 6)
        self.assertNotEqual(case0["values"][0], case0["values"][1])
        # The topics came from the topic stream alone (the
        # value redraw must not perturb them).
        self.assertEqual(case0["topics"][0], rotation.TOPICS[0])
        self.assertEqual(
            case0["topics"][1:],
            [rotation.TOPICS[(slot) % 30] for slot in (21, 22, 23, 24, 25)],
        )

    def test_pool_pins(self):
        # The frozen pool properties the design's containment
        # argument rests on, checked exhaustively (design F15:
        # containment is a pinned pool property, not prose).
        topics = rotation.TOPICS
        words = rotation.WORDS
        self.assertEqual(len(topics), 30)
        self.assertEqual(len(set(topics)), 30)
        self.assertEqual(len(words), 64)
        self.assertEqual(len(set(words)), 64)
        for topic in topics:
            for other in topics:
                if topic is not other:
                    self.assertNotIn(topic, other)
        for topic in topics:
            for token in topic.split(" "):
                self.assertGreaterEqual(len(token), 4)
                self.assertNotIn(token, words)
        for word in words:
            for other in words:
                if word is not other:
                    self.assertFalse(other.endswith(word) and len(other) > len(word), (word, other))

    def test_topic_codeword_prefix_disjoint(self):
        # No topic token shares a 4-character prefix with a code
        # word, and neither contains the other (the boundary
        # between a fact-shaped distractor and the asked fact
        # must stay decodable from the text).
        for topic in rotation.TOPICS:
            for token in topic.split(" "):
                for word in rotation.WORDS:
                    self.assertNotEqual(token[:4], word[:4], (token, word))
                    self.assertNotIn(token, word, (token, word))
                    self.assertNotIn(word, token, (token, word))


class TestArmConstruction(unittest.TestCase):
    def test_arm_orders_and_slots(self):
        expected_orders = {
            "rot_0": [0, 1, 2, 3, 4],
            "rot_1": [1, 2, 3, 4, 0],
            "rot_2": [2, 3, 4, 0, 1],
            "rot_3": [3, 4, 0, 1, 2],
            "rot_4": [4, 0, 1, 2, 3],
            "mir_1": [1, 0, 4, 3, 2],
            "mir_2": [2, 1, 0, 4, 3],
            "mir_3": [3, 2, 1, 0, 4],
            "mir_4": [4, 3, 2, 1, 0],
            "absent": [5, 1, 2, 3, 4],
        }
        for arm, order in expected_orders.items():
            self.assertEqual(rotation.arm_order(arm), tuple(order), arm)
        with self.assertRaises(ValueError):
            rotation.arm_order("closed_book")
        with self.assertRaises(ValueError):
            rotation.asked_slot("absent")
        # Asked slots: rotations and the control set share the
        # multiset {0,1,2,3,4} (the B-F1 slot-matching answer).
        self.assertEqual([rotation.asked_slot(f"rot_{r}") for r in range(5)], [0, 4, 3, 2, 1])
        self.assertEqual(
            sorted(rotation.asked_slot(arm) for arm in rotation.CONTROL_ARMS),
            [0, 1, 2, 3, 4],
        )
        self.assertEqual(
            sorted(rotation.asked_slot(arm) for arm in rotation.ROTATION_ARMS),
            [0, 1, 2, 3, 4],
        )
        self.assertEqual(rotation.asked_slot("mir_4"), 4)

    def test_arm_content(self):
        case = rotation.generate_cases(0)[0]
        content = rotation.arm_content(case, "rot_0")
        blocks = [case["blocks"][b] for b in range(5)]
        self.assertEqual(content, " ".join(blocks) + "\n\n" + case["question"])
        self.assertEqual(case["question"], f"What is the {case['topic']} access code?")
        content_absent = rotation.arm_content(case, "absent")
        self.assertNotIn(case["values"][0], content_absent)
        self.assertIn(case["values"][5], content_absent)
        self.assertEqual(rotation.arm_content(case, "closed_book"), case["question"])
        # A rotation moves whole blocks: the rot_2 content is
        # the same blocks in the rotated order.
        content_rot2 = rotation.arm_content(case, "rot_2")
        rotated = [case["blocks"][b] for b in (2, 3, 4, 0, 1)]
        self.assertEqual(content_rot2, " ".join(rotated) + "\n\n" + case["question"])

    def test_block_and_question_text(self):
        case = rotation.generate_cases(0)[0]
        self.assertEqual(
            case["blocks"][0],
            f"The {case['topics'][0]} access code is {case['values'][0]}.",
        )
        self.assertEqual(case["question"], f"What is the {case['topic']} access code?")
        self.assertEqual(case["expected"], case["values"][0])

    def test_inventory_digest(self):
        ids = (5, 3, 9, 3)
        self.assertEqual(rotation.inventory_digest(ids), rotation.digest_text("3,3,5,9"))
        self.assertEqual(rotation.inventory_digest((9, 5, 3, 3)), rotation.inventory_digest(ids))


def _record(fixture, verdicts, exact=True, screened_pass=False):
    """Fabricate one aggregate-level record: ``verdicts`` maps
    arm -> bool (present, completed) or None (incomplete). All
    span arms share one inventory digest (exact) unless
    ``exact`` is False, which perturbs rot_2's digest."""
    cells = {}
    details = {}
    for arm in rotation.ARM_KEYS:
        verdict = verdicts.get(arm)
        if verdict is None:
            cells[arm] = None
            details[arm] = "arm_evaluation_failed"
        else:
            cells[arm] = {
                "verdict": bool(verdict),
                "boundary_found": True,
                "boundary_text_match": True,
                "input_tokens": 10,
                "answer_digest": "a" * 64,
            }
            details[arm] = ""
    digests = dict.fromkeys(rotation.SPAN_ARMS, "b" * 64)
    if not exact:
        digests["rot_2"] = "c" * 64
    return {
        "fixture": fixture,
        "topic": "archive vault",
        "expected_digest": "a" * 64,
        "arm_tokens": dict.fromkeys(rotation.ARM_KEYS, 10),
        "arm_inventory_digests": digests,
        "cells": cells,
        "cell_details": details,
    }


def _all_arms(value):
    """Every arm at ``value``, except the closed-book screen
    cell, which stays False so the record is live (a closed-book
    pass would screen the record out of every population)."""
    verdicts = dict.fromkeys(rotation.ARM_KEYS, value)
    verdicts["closed_book"] = False
    return verdicts


class TestClassify(unittest.TestCase):
    def test_truth_table(self):
        c = rotation.classify_rotation
        # n_headline, flip, all_pass, all_fail, paired, flips_paired, flips_mirror
        self.assertEqual(c(5, 0, 5, 0, 5, 0, 0), "insufficient")
        self.assertEqual(c(6, 0, 2, 4, 6, 0, 0), "joint_collapse_dominant")
        self.assertEqual(c(6, 0, 4, 2, 6, 0, 0), "invariant_observed")
        self.assertEqual(c(6, 0, 3, 3, 6, 0, 0), "invariant_observed")
        self.assertEqual(c(6, 1, 3, 2, 5, 1, 0), "control_set_too_small")
        self.assertEqual(c(6, 2, 2, 2, 6, 2, 2), "within_mirror_control")
        self.assertEqual(c(6, 2, 2, 2, 6, 1, 2), "within_mirror_control")
        self.assertEqual(c(6, 2, 2, 2, 6, 2, 1), "rotation_sensitive")
        self.assertEqual(c(20, 20, 0, 0, 20, 20, 0), "rotation_sensitive")

    def test_enumeration_joint_table(self):
        # The published joint table, pinned over the full
        # feasible tuple space: the falsification boolean is
        # True exactly on {control_set_too_small,
        # within_mirror_control, rotation_sensitive}, every
        # label is reachable, and the classification is total.
        labels = set()
        falsified_labels = set()
        calm_labels = set()
        for n_headline in range(0, 11):
            for n_all_pass in range(0, n_headline + 1):
                for n_all_fail in range(0, n_headline - n_all_pass + 1):
                    n_flip = n_headline - n_all_pass - n_all_fail
                    for n_paired in range(0, n_headline + 1):
                        for n_fp in range(0, min(n_paired, n_flip) + 1):
                            if n_flip == 0 and n_fp != 0:
                                continue
                            for n_fm in range(0, n_paired + 1):
                                label = rotation.classify_rotation(
                                    n_headline,
                                    n_flip,
                                    n_all_pass,
                                    n_all_fail,
                                    n_paired,
                                    n_fp,
                                    n_fm,
                                )
                                falsified = rotation.invariance_falsified(n_headline, n_flip)
                                labels.add(label)
                                if falsified:
                                    falsified_labels.add(label)
                                else:
                                    calm_labels.add(label)
                                self.assertEqual(
                                    falsified,
                                    label
                                    in {
                                        "control_set_too_small",
                                        "within_mirror_control",
                                        "rotation_sensitive",
                                    },
                                    (
                                        n_headline,
                                        n_flip,
                                        n_all_pass,
                                        n_all_fail,
                                        n_paired,
                                        n_fp,
                                        n_fm,
                                        label,
                                    ),
                                )
        self.assertEqual(
            labels,
            {
                "insufficient",
                "joint_collapse_dominant",
                "invariant_observed",
                "control_set_too_small",
                "within_mirror_control",
                "rotation_sensitive",
            },
        )
        self.assertEqual(
            falsified_labels,
            {"control_set_too_small", "within_mirror_control", "rotation_sensitive"},
        )
        self.assertEqual(
            calm_labels,
            {"insufficient", "joint_collapse_dominant", "invariant_observed"},
        )

    def test_falsification_formula(self):
        self.assertTrue(rotation.invariance_falsified(6, 1))
        self.assertFalse(rotation.invariance_falsified(6, 0))
        self.assertFalse(rotation.invariance_falsified(5, 3))


class TestAggregateLabels(unittest.TestCase):
    """Every classification label reached through the real
    aggregator on fabricated records (the chirality patterns a
    scripted backend cannot produce from text alone are
    fabricated at the record level, where the aggregator's
    contract lives)."""

    def _aggregate(self, records):
        return rotation.aggregate_candidate(records)

    def test_invariant_observed(self):
        records = [_record(f"rotation-{i:02d}", _all_arms(True)) for i in range(8)]
        agg = self._aggregate(records)
        self.assertEqual(agg["classification"], "invariant_observed")
        self.assertFalse(agg["invariance_falsified"])
        self.assertEqual(agg["headline"]["invariance_rate"], 1.0)
        self.assertEqual(agg["slot_curve"]["slots"], {str(s): 8 for s in range(5)})

    def test_joint_collapse_dominant(self):
        records = [_record(f"rotation-{i:02d}", _all_arms(False)) for i in range(8)]
        agg = self._aggregate(records)
        self.assertEqual(agg["classification"], "joint_collapse_dominant")
        self.assertFalse(agg["invariance_falsified"])
        self.assertEqual(agg["headline"]["n_all_fail"], 8)

    def test_rotation_sensitive(self):
        # Rotations flip (rot_1 fails), the whole control set
        # agrees with rot_0: the mirror statistic stays at zero.
        verdicts = _all_arms(True)
        verdicts["rot_1"] = False
        records = [_record(f"rotation-{i:02d}", dict(verdicts)) for i in range(8)]
        agg = self._aggregate(records)
        self.assertEqual(agg["classification"], "rotation_sensitive")
        self.assertTrue(agg["invariance_falsified"])
        self.assertEqual(agg["mirror"]["n_flips_paired"], 8)
        self.assertEqual(agg["mirror"]["n_flips_mirror"], 0)
        self.assertEqual(agg["reversal"]["n_discordant"], 0)
        self.assertEqual(agg["headline"]["n_flip_from_pass"], 8)

    def test_within_mirror_control(self):
        # A pure slot effect: slot 4 fails in both sets
        # (rot_1 and mir_4 sit there), so the statistics tie.
        verdicts = _all_arms(True)
        verdicts["rot_1"] = False
        verdicts["mir_4"] = False
        records = [_record(f"rotation-{i:02d}", dict(verdicts)) for i in range(8)]
        agg = self._aggregate(records)
        self.assertEqual(agg["classification"], "within_mirror_control")
        self.assertEqual(agg["mirror"]["n_flips_paired"], 8)
        self.assertEqual(agg["mirror"]["n_flips_mirror"], 8)
        self.assertEqual(agg["reversal"]["n_discordant"], 8)

    def test_control_set_too_small(self):
        verdicts = _all_arms(True)
        verdicts["rot_1"] = False
        for arm in rotation.MIRROR_ARMS:
            verdicts[arm] = None
        records = [_record(f"rotation-{i:02d}", dict(verdicts)) for i in range(8)]
        agg = self._aggregate(records)
        self.assertEqual(agg["classification"], "control_set_too_small")
        self.assertTrue(agg["invariance_falsified"])
        self.assertEqual(agg["mirror"]["n_paired"], 0)

    def test_insufficient(self):
        records = [_record(f"rotation-{i:02d}", _all_arms(True)) for i in range(5)]
        agg = self._aggregate(records)
        self.assertEqual(agg["classification"], "insufficient")
        self.assertFalse(agg["invariance_falsified"])

    def test_exactness_gate_drops_headline(self):
        # One record's rot_2 inventory differs: the record
        # leaves the headline population (its arms rates still
        # count it), and the exactness share reports the loss.
        records = [_record(f"rotation-{i:02d}", _all_arms(True)) for i in range(8)]
        records[0] = _record("rotation-00", _all_arms(True), exact=False)
        agg = self._aggregate(records)
        self.assertEqual(agg["headline"]["n_headline"], 7)
        self.assertEqual(agg["arms"]["rot_2"]["n_completed"], 8)
        self.assertLess(agg["headline"]["id_exact_share"], 1.0)
        self.assertEqual(agg["headline"]["evaluable_share"], 7 / 8)

    def test_absent_false_recall(self):
        verdicts = _all_arms(False)
        verdicts["absent"] = True
        records = [_record(f"rotation-{i:02d}", dict(verdicts)) for i in range(4)]
        agg = self._aggregate(records)
        self.assertEqual(agg["absent"]["n_false_pass"], 4)
        self.assertEqual(agg["absent"]["rate"], 1.0)

    def test_aggregate_tolerates_malformed_record(self):
        agg = rotation.aggregate_candidate([{"fixture": "rotation-00", "cells": None}])
        self.assertEqual(agg["headline"]["n_headline"], 0)
        self.assertEqual(agg["n_closed_book_incomplete"], 1)


class TestMirrorAgreement(unittest.TestCase):
    """The diffreports literal-table mirror must agree with
    the producer module on every shared computation (a drift
    in either side's constants breaks the agreement)."""

    def test_mirror_generator(self):
        for seed in (0, 1, 2, 13):
            mirrored = diffreports._rotation_generate(seed)
            produced = rotation.generate_cases(seed)
            self.assertEqual(len(mirrored), len(produced))
            for m_case, p_case in zip(mirrored, produced, strict=True):
                self.assertEqual(m_case["fixture_id"], p_case["fixture_id"])
                self.assertEqual(m_case["topics"], p_case["topics"])
                self.assertEqual(m_case["values"], p_case["values"])

    def test_mirror_protocol(self):
        for seed in (0, 5):
            mirrored = diffreports._rotation_protocol(seed, "sha256:z")
            produced = dict(rotation.protocol_object(seed, "sha256:z"))
            produced["fingerprint"] = rotation.protocol_fingerprint(seed, "sha256:z")
            self.assertEqual(mirrored, produced)

    def test_mirror_classify(self):
        for args in (
            (5, 0, 5, 0, 5, 0, 0),
            (6, 0, 2, 4, 6, 0, 0),
            (6, 0, 4, 2, 6, 0, 0),
            (6, 1, 3, 2, 5, 1, 0),
            (6, 2, 2, 2, 6, 2, 2),
            (6, 2, 2, 2, 6, 2, 1),
            (20, 7, 9, 4, 18, 6, 3),
        ):
            self.assertEqual(
                diffreports._rotation_classify(*args), rotation.classify_rotation(*args)
            )
            self.assertEqual(
                diffreports._rotation_falsified(args[0], args[1]),
                rotation.invariance_falsified(args[0], args[1]),
            )

    def test_mirror_aggregate_and_selection_on_stage_records(self):
        payload = _payload()
        raw = {
            name: entry["records"]
            for name, entry in payload["candidates"].items()
            if entry["status"] == "available"
        }
        for records in raw.values():
            self.assertEqual(
                diffreports._rotation_aggregate(records),
                rotation.aggregate_candidate(records),
            )
        for arm in rotation.CONDITION_ARMS:
            self.assertEqual(
                diffreports._rotation_arm_selection(raw, arm),
                rotation.arm_selection(raw, arm),
            )


class TestStage(unittest.TestCase):
    def test_perfect_backend_invariant(self):
        payload = _payload()
        self.assertTrue(payload["available"])
        self.assertTrue(_valid_rotation(payload))
        self.assertEqual(payload["n_candidates"], 2)
        self.assertEqual(payload["n_cases"], 20)
        entry = payload["candidates"]["rt-a"]
        self.assertEqual(entry["n_screened"], 0)
        self.assertEqual(entry["n_boundary_screened"], 0)
        self.assertEqual(entry["n_closed_book_incomplete"], 0)
        self.assertEqual(entry["closed_book"]["pass_rate"], 0.0)
        headline = entry["headline"]
        self.assertEqual(headline["n_headline"], 20)
        self.assertEqual(headline["n_all_pass"], 20)
        self.assertEqual(headline["invariance_rate"], 1.0)
        self.assertEqual(headline["id_exact_share"], 1.0)
        self.assertEqual(entry["classification"], "invariant_observed")
        self.assertFalse(entry["invariance_falsified"])
        self.assertEqual(entry["slot_curve"]["slots"], {str(s): 20 for s in range(5)})
        self.assertEqual(entry["mirror"]["n_flips_mirror"], 0)
        self.assertEqual(entry["absent"]["n_false_pass"], 0)
        for arm in rotation.SPAN_ARMS:
            self.assertEqual(entry["arms"][arm]["rate"], 1.0, arm)
        # Every candidate passes everything: no unique winner.
        condition = payload["conditions"]["rot_0"]
        self.assertEqual(condition["decision"], "ambiguous")
        self.assertFalse(condition["flip"])

    def test_slot_fail_absorbed_by_mirror_control(self):
        # A pure slot effect flips the rotations AND the
        # control set identically (their asked-slot multisets
        # match), so the classification acquits -- the B-F1
        # settlement, pinned as behaviour rather than prose.
        payload = _payload(_WordBackend(mode="slot_fail"))
        self.assertTrue(_valid_rotation(payload))
        entry = payload["candidates"]["rt-a"]
        self.assertEqual(entry["headline"]["n_flip_any"], 20)
        self.assertEqual(entry["mirror"]["n_flips_paired"], 20)
        self.assertEqual(entry["mirror"]["n_flips_mirror"], 20)
        self.assertEqual(entry["classification"], "within_mirror_control")
        self.assertTrue(entry["invariance_falsified"])
        # Slot 0 passes everywhere; every other slot fails.
        self.assertEqual(
            entry["slot_curve"]["slots"],
            {"0": 20, "1": 0, "2": 0, "3": 0, "4": 0},
        )

    def test_wrong_backend_joint_collapse(self):
        payload = _payload(_WordBackend(mode="wrong"))
        self.assertTrue(_valid_rotation(payload))
        entry = payload["candidates"]["rt-a"]
        self.assertEqual(entry["headline"]["n_all_fail"], 20)
        self.assertEqual(entry["classification"], "joint_collapse_dominant")
        self.assertFalse(entry["invariance_falsified"])

    def test_stage_rejects_bad_formats(self):
        backend = _WordBackend()
        with self.assertRaises(ValueError):
            experiments.rotation_stage(backend, [], boundary=(1,), max_context=100)
        with self.assertRaises(ValueError):
            experiments.rotation_stage(backend, (), boundary=(1,), max_context=100)
        with self.assertRaises(ValueError):
            experiments.rotation_stage(backend, [FORMATS[0]], boundary=(1,), max_context=100)

    def test_duplicate_format_names_rejected(self):
        backend = _WordBackend()
        dup = (FORMATS[0], FORMATS[0])
        with self.assertRaises(ValueError):
            experiments.rotation_stage(backend, dup, boundary=(1,), max_context=100)

    def test_record_budget(self):
        backend = _WordBackend()
        formats = tuple(
            Format(name=f"f{i}", bos="", user_prefix="u:", assistant_prefix="a:", turn_end="</s>")
            for i in range(25)
        )
        stage = experiments.rotation_stage(
            backend, formats, boundary=(1,), max_context=4096, seed=0
        )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "rotation_record_budget_exceeded")

    def test_screen_failed_stage(self):
        # A boundary whose decoded text occurs inside the
        # frozen texts ("the" occurs in the question template)
        # screens the whole stage closed.
        backend = _WordBackend()
        boundary_map = {fmt.name: tuple(backend.encode("the")) for fmt in FORMATS}
        stage = experiments.rotation_stage(
            backend,
            FORMATS,
            boundary=tuple(backend.encode("endmark")),
            boundary_map=boundary_map,
            max_context=4096,
            seed=0,
        )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "rotation_screen_failed")

    def test_screen_audit_findings_fail_closed(self):
        backend = _WordBackend()
        with mock.patch(
            "model_protocol_lab.fixtures.audit_special_token_literals",
            return_value=(object(),),
        ):
            stage = experiments.rotation_stage(
                backend, FORMATS, boundary=(1,), max_context=4096, seed=0
            )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "rotation_screen_failed")

    def test_boundary_screened_case(self):
        # A boundary text that occurs in one case's rendered
        # blocks (crossing the topic/template seam, so it is in
        # no frozen literal) boundary-screens exactly the cases
        # whose blocks contain it: every cell is arm_screened
        # and the record enters no population.
        backend = _WordBackend()
        cases = rotation.generate_cases(0)
        expected_screened = sum(1 for c in cases if "archive vault" in c["topics"])
        self.assertGreaterEqual(expected_screened, 1)
        boundary_map = {fmt.name: tuple(backend.encode("vault access")) for fmt in FORMATS}
        payload = _payload(backend, boundary_map=boundary_map)
        self.assertTrue(payload["available"])
        self.assertTrue(_valid_rotation(payload))
        entry = payload["candidates"]["rt-a"]
        self.assertEqual(entry["n_boundary_screened"], expected_screened)
        self.assertEqual(entry["n_live"], 20 - expected_screened)
        screened_records = [
            r for r in entry["records"] if set(r["cell_details"].values()) == {"arm_screened"}
        ]
        self.assertEqual(len(screened_records), expected_screened)
        for record in screened_records:
            self.assertTrue(all(cell is None for cell in record["cells"].values()))
            # The construction readouts are still measured.
            self.assertEqual(set(record["arm_inventory_digests"]), set(rotation.SPAN_ARMS))

    def test_context_exceeded_cells(self):
        payload = _payload(max_context=8)
        self.assertTrue(payload["available"])
        self.assertTrue(_valid_rotation(payload))
        entry = payload["candidates"]["rt-a"]
        self.assertEqual(entry["classification"], "insufficient")
        self.assertEqual(entry["n_closed_book_incomplete"], 20)
        record = entry["records"][0]
        self.assertEqual(set(record["cell_details"].values()), {"arm_context_exceeded"})

    def test_exploding_backend_degrades_cells(self):
        payload = _payload(_WordBackend(mode="explode"))
        self.assertTrue(payload["available"])
        self.assertTrue(_valid_rotation(payload))
        record = payload["candidates"]["rt-a"]["records"][0]
        self.assertEqual(set(record["cell_details"].values()), {"arm_evaluation_failed"})

    def test_cancel_at_case_boundary(self):
        event = threading.Event()

        class _DecodeCancels(_WordBackend):
            def decode(self, ids):
                event.set()
                return super().decode(ids)

        backend = _DecodeCancels()
        stage = experiments.rotation_stage(
            backend,
            FORMATS,
            boundary=tuple(backend.encode("endmark")),
            max_context=4096,
            seed=0,
            cancel=event,
        )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "cancelled")

    def test_cancel_before_first_evaluation(self):
        event = threading.Event()

        class _EncodeCancels(_WordBackend):
            def encode(self, text):
                ids = super().encode(text)
                if len(text) > 20:  # arm renders only; setup encodes stay clean
                    event.set()
                return ids

        backend = _EncodeCancels()
        stage = experiments.rotation_stage(
            backend,
            FORMATS,
            boundary=tuple(backend.encode("endmark")),
            max_context=4096,
            seed=0,
            cancel=event,
        )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "cancelled")

    def test_cancel_at_arm_boundary(self):
        event = threading.Event()
        backend = _WordBackend()
        original = experiments._rotation_cell

        def wrapper(*args, **kwargs):
            outcome = original(*args, **kwargs)
            event.set()
            return outcome

        with mock.patch.object(experiments, "_rotation_cell", wrapper):
            stage = experiments.rotation_stage(
                backend,
                FORMATS,
                boundary=tuple(backend.encode("endmark")),
                max_context=4096,
                seed=0,
                cancel=event,
            )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "cancelled")

    def test_cancel_unwinds_stage(self):
        backend = _WordBackend(cancel_after=3)
        event = threading.Event()
        backend.cancel_event = event
        stage = experiments.rotation_stage(
            backend,
            FORMATS,
            boundary=tuple(backend.encode("endmark")),
            max_context=4096,
            seed=0,
            cancel=event,
        )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "cancelled")
        payload = experiments.rotation_analysis(stage, tuple(f.name for f in FORMATS))
        self.assertFalse(payload["available"])

    def test_analysis_unavailable_candidate(self):
        payload = _payload_with_unavailable()
        self.assertEqual(
            payload["candidates"]["rt-b"],
            {"status": "unavailable", "detail": "candidate_evaluation_failed", "records": []},
        )
        self.assertTrue(_valid_rotation(payload))


class TestScreeningAndConditions(unittest.TestCase):
    def test_closed_book_screen_excludes_case(self):
        first = rotation.generate_cases(0)[0]
        backend = _WordBackend(known={first["topic"]: first["values"][0]})
        payload = _payload(backend)
        self.assertTrue(_valid_rotation(payload))
        entry = payload["candidates"]["rt-a"]
        self.assertEqual(entry["n_screened"], 1)
        self.assertEqual(entry["closed_book"]["n_pass"], 1)
        self.assertEqual(entry["headline"]["n_headline"], 19)
        self.assertEqual(entry["n_live"], 19)

    def test_union_screening_across_candidates(self):
        # Selections run over the population screened for NO
        # candidate in the run, even when screening itself is
        # per-candidate: rt-a's closed-book pass on case 0 must
        # shrink rt-b's selection population too.
        first = rotation.generate_cases(0)[0]

        class _PerFormatBackend(_WordBackend):
            def generate(self, input_ids, **kwargs):
                text = self.decode(list(input_ids))
                if text.startswith("u:") and first["topic"] in text and "code is" not in text:
                    boundary = self.encode("endmark")[0]
                    out = list(self.encode(first["values"][0])) + [boundary]
                    out = (out + [boundary] * kwargs["max_new_tokens"])[: kwargs["max_new_tokens"]]
                    return Generation(token_ids=tuple(out), reason="limit")
                return super().generate(input_ids, **kwargs)

        payload = _payload(_PerFormatBackend())
        self.assertTrue(_valid_rotation(payload))
        self.assertEqual(payload["candidates"]["rt-a"]["n_screened"], 1)
        self.assertEqual(payload["candidates"]["rt-b"]["n_screened"], 0)
        shared = rotation.shared_live_fixtures(
            {n: e["records"] for n, e in payload["candidates"].items()}
        )
        self.assertEqual(len(shared), 19)
        self.assertNotIn(first["fixture_id"], shared)
        for arm in rotation.CONDITION_ARMS:
            self.assertFalse(payload["conditions"][arm]["withheld"], arm)

    def test_shared_live_fixtures_empty(self):
        self.assertEqual(rotation.shared_live_fixtures({}), frozenset())

    def test_selection_withheld_on_mismatched_exactness(self):
        # The per-arm exactness gate: if one candidate's rot_2
        # renders are not token-exact, the rot_2 selection is
        # withheld for the whole run while rot_0 still selects.
        def rec(fixture, verdict, exact=True):
            record = _record(fixture, _all_arms(verdict), exact=exact)
            record["cells"]["closed_book"] = {
                "verdict": False,
                "boundary_found": True,
                "boundary_text_match": True,
                "input_tokens": 10,
                "answer_digest": "d" * 64,
            }
            return record

        raw = {
            "a": [rec(f"rotation-{i:02d}", True) for i in range(3)],
            "b": [rec(f"rotation-{i:02d}", True, exact=False) for i in range(3)],
        }
        gated = rotation.arm_selection(raw, "rot_2")
        self.assertTrue(gated["withheld"])
        self.assertEqual(gated["detail"], "selection_withheld_mismatched_sets")
        self.assertIsNone(gated["flip"])
        baseline = rotation.arm_selection(raw, "rot_0")
        self.assertFalse(baseline["withheld"])
        self.assertEqual(baseline["flip"], False)


class TestValidatorBattery(unittest.TestCase):
    """One break at a time: each mutation of a coherent
    payload must flip _valid_rotation to False."""

    def _breaks(self):
        payload = _payload()

        def entry(payload):
            return payload["candidates"]["rt-a"]

        def record(payload):
            return entry(payload)["records"][0]

        def cell(payload):
            return record(payload)["cells"]["rot_1"]

        breaks = {
            "payload_extra_key": lambda p: p.update(extra=1),
            "detail_not_empty": lambda p: p.update(detail="x"),
            "n_cases_off": lambda p: p.update(n_cases=19),
            "n_candidates_off": lambda p: p.update(n_candidates=3),
            "protocol_fingerprint_forged": lambda p: p["protocol"].update(fingerprint="0" * 64),
            "protocol_seed_drift": lambda p: p["protocol"].update(seed=9),
            "classification_forged": lambda p: entry(p).update(classification="rotation_sensitive"),
            "falsified_forged": lambda p: entry(p).update(invariance_falsified=True),
            "aggregate_count_forged": lambda p: entry(p).update(n_live=19),
            "headline_count_forged": lambda p: entry(p)["headline"].update(n_headline=19),
            "condition_winner_forged": lambda p: p["conditions"]["rot_0"].update(winner="rt-a"),
            "record_topic_swapped": lambda p: record(p).update(topic="boiler room"),
            "record_fixture_swapped": lambda p: record(p).update(fixture="rotation-19"),
            "expected_digest_wrong": lambda p: record(p).update(expected_digest="0" * 64),
            "cell_verdict_flipped": lambda p: cell(p).update(verdict=False),
            "cell_input_tokens_off": lambda p: cell(p).update(
                input_tokens=cell(p)["input_tokens"] + 1
            ),
            "cell_answer_digest_none": lambda p: cell(p).update(answer_digest=None),
            "arm_tokens_bool": lambda p: record(p)["arm_tokens"].update(rot_0=True),
            "arm_tokens_missing_arm": lambda p: record(p)["arm_tokens"].pop("mir_3"),
            "inventory_digest_malformed": lambda p: record(p)["arm_inventory_digests"].update(
                rot_1="zz"
            ),
            "inventory_digest_missing_arm": lambda p: record(p)["arm_inventory_digests"].pop(
                "rot_4"
            ),
            "cell_present_with_detail": lambda p: record(p)["cell_details"].update(
                rot_1="arm_evaluation_failed"
            ),
            "cell_absent_with_empty_detail": lambda p: record(p)["cells"].update(rot_1=None),
            "cell_absent_with_bad_detail": lambda p: (
                record(p)["cells"].update(rot_1=None),
                record(p)["cell_details"].update(rot_1="bogus"),
            ),
            "records_short": lambda p: entry(p)["records"].pop(),
            "aggregate_extra_key": lambda p: entry(p).update(extra=1),
        }
        return payload, breaks

    def test_battery(self):
        payload, breaks = self._breaks()
        self.assertTrue(_valid_rotation(payload))
        for name, mutate in breaks.items():
            with self.subTest(break_=name):
                broken = copy.deepcopy(payload)
                mutate(broken)
                self.assertFalse(_valid_rotation(broken), name)

    def test_unavailable_entry_with_records_rejected(self):
        payload = _payload_with_unavailable()
        self.assertTrue(_valid_rotation(payload))
        broken = copy.deepcopy(payload)
        broken["candidates"]["rt-b"]["records"] = [{"fixture": "x"}]
        self.assertFalse(_valid_rotation(broken))


class TestDiff(unittest.TestCase):
    def test_identical_payloads_unchanged(self):
        payload = _payload()
        diff = diff_rotation(payload, copy.deepcopy(payload))
        self.assertEqual(diff["available"], "both")
        self.assertFalse(diff["changed"])
        self.assertTrue(diff["comparable"])

    def test_classification_drift(self):
        old = _payload()
        new = _payload(_WordBackend(mode="slot_fail"))
        diff = diff_rotation(old, new)
        self.assertTrue(diff["changed"])
        metrics = {(d["candidate"], d["metric"]) for d in diff["candidate_drift"]}
        self.assertIn(("rt-a", "classification"), metrics)
        self.assertIn(("rt-a", "invariance_falsified"), metrics)
        self.assertIn(("rt-a", "n_flip_any"), metrics)

    def test_foreign_backend_not_comparable(self):
        old = _payload()
        new = copy.deepcopy(old)
        protocol = dict(rotation.protocol_object(0, "sha256:other-backend"))
        protocol["fingerprint"] = rotation.protocol_fingerprint(0, "sha256:other-backend")
        new["protocol"] = protocol
        diff = diff_rotation(old, new)
        self.assertTrue(diff["backend_drift"])
        self.assertFalse(diff["comparable"])
        self.assertTrue(diff["changed"])

    def test_both_unavailable_no_status_drift(self):
        payload = _payload_with_unavailable()
        diff = diff_rotation(payload, copy.deepcopy(payload))
        self.assertFalse(diff["changed"])

    def test_status_drift(self):
        old = _payload_with_unavailable()
        new = _payload()
        diff = diff_rotation(old, new)
        self.assertTrue(diff["changed"])
        metrics = {(d["candidate"], d["metric"]) for d in diff["candidate_drift"]}
        self.assertIn(("rt-b", "status"), metrics)

    def test_missing_sides(self):
        payload = _payload()
        self.assertEqual(diff_rotation(None, None)["available"], "neither")
        self.assertEqual(diff_rotation(payload, None)["available"], "old")
        self.assertEqual(diff_rotation(None, payload)["available"], "new")
        self.assertFalse(diff_rotation(payload, None)["changed"])


class TestReportingAndCli(unittest.TestCase):
    def test_section_renders(self):
        payload = _payload()
        html = render_html({"rotation": payload})
        self.assertIn("Context rotation", html)
        self.assertIn("invariant_observed", html)
        self.assertIn("Mirror control over", html)
        self.assertIn("Fact-absent false recall", html)

    def test_section_absent_without_payload(self):
        self.assertNotIn('id="rotation"', render_html({}))
        self.assertNotIn('id="rotation"', render_html({"rotation": {"available": False}}))

    def test_section_hostile_payload(self):
        payload = _payload()
        hostile = {"rotation": copy.deepcopy(payload)}
        hostile["rotation"]["candidates"] = {
            "<script>alert(1)</script>": {"status": "available"},
        }
        html = render_html(hostile)
        self.assertNotIn("<script>alert(1)</script>", html)

    def test_section_na_rates(self):
        payload = _payload()
        sparse = copy.deepcopy(payload)
        entry = sparse["candidates"]["rt-a"]
        entry["headline"] = {"n_headline": 0}
        entry["closed_book"] = {}
        entry["absent"] = {}
        entry["mirror"] = {}
        entry["reversal"] = {}
        entry["slot_curve"] = {}
        html = render_html({"rotation": sparse})
        self.assertIn("Context rotation", html)

    def test_section_unavailable_row(self):
        payload = _payload_with_unavailable()
        html = render_html({"rotation": payload})
        self.assertIn("unavailable", html)

    def test_section_withheld_conditions(self):
        payload = _payload()
        for cond in payload["conditions"].values():
            cond["withheld"] = True
            cond["winner"] = None
        html = render_html({"rotation": payload})
        self.assertIn("selection withheld", html)

    def test_cli_helper_attaches_payload(self):
        sentinel = {"available": True, "detail": ""}
        with (
            mock.patch.object(experiments, "rotation_stage", return_value={"available": True}),
            mock.patch.object(experiments, "rotation_analysis", return_value=sentinel),
        ):
            result: dict = {}
            cli._run_rotation_probe(
                _WordBackend(),
                FORMATS,
                result,
                boundary=(1,),
                boundary_map=None,
                max_new_tokens=8,
                max_context=4096,
                seed=0,
                cancel=None,
            )
        self.assertIs(result["rotation"], sentinel)

    def test_cli_helper_never_raises(self):
        with mock.patch.object(experiments, "rotation_stage", side_effect=RuntimeError("boom")):
            result: dict = {}
            cli._run_rotation_probe(
                _WordBackend(),
                FORMATS,
                result,
                boundary=(1,),
                boundary_map=None,
                max_new_tokens=8,
                max_context=4096,
                seed=0,
                cancel=None,
            )
        self.assertEqual(result["rotation"], {"available": False, "detail": "rotation_unavailable"})


class TestGapCoverage(unittest.TestCase):
    """Full-run coverage left three defensive branches in
    rotation.py unexecuted; each is proven reachable here
    (a passing suite is not proof a guarded line ran)."""

    def test_arm_id_exact_malformed_record(self):
        self.assertFalse(rotation.arm_id_exact({"fixture": "rotation-00"}, "rot_1"))
        self.assertFalse(
            rotation.arm_id_exact({"arm_inventory_digests": "b" * 64, "arm_tokens": {}}, "rot_1")
        )
        # rot_0 is exact by definition, even on a malformed record.
        self.assertTrue(rotation.arm_id_exact({"fixture": "rotation-00"}, "rot_0"))

    def test_selection_withheld_on_empty_completed_set(self):
        raw = {
            "a": [_record(f"rotation-{i:02d}", _all_arms(True)) for i in range(3)],
            "b": [
                _record(
                    f"rotation-{i:02d}",
                    {**_all_arms(True), "rot_2": None},
                )
                for i in range(3)
            ],
        }
        gated = rotation.arm_selection(raw, "rot_2")
        self.assertTrue(gated["withheld"])
        self.assertEqual(gated["detail"], "selection_withheld_mismatched_sets")
        self.assertIsNone(gated["flip"])
        baseline = rotation.arm_selection(raw, "rot_0")
        self.assertFalse(baseline["withheld"])

    def test_selection_withheld_on_mismatched_completed_sets(self):
        # Non-empty but differing completed sets take the
        # mismatched-sets return (not the empty-set return):
        # candidate b completes only two of the three gated
        # fixtures on rot_2.
        b_records = []
        for i in range(3):
            verdicts = _all_arms(True)
            if i == 0:
                verdicts["rot_2"] = None
            b_records.append(_record(f"rotation-{i:02d}", verdicts))
        raw = {
            "a": [_record(f"rotation-{i:02d}", _all_arms(True)) for i in range(3)],
            "b": b_records,
        }
        gated = rotation.arm_selection(raw, "rot_2")
        self.assertTrue(gated["withheld"])
        self.assertEqual(gated["detail"], "selection_withheld_mismatched_sets")
        self.assertIsNone(gated["flip"])

    def test_mirror_exactness_share_drops(self):
        records = [_record(f"rotation-{i:02d}", _all_arms(True)) for i in range(8)]
        perturbed = _record("rotation-00", _all_arms(True))
        perturbed["arm_inventory_digests"] = dict.fromkeys(rotation.SPAN_ARMS, "b" * 64)
        perturbed["arm_inventory_digests"]["mir_2"] = "d" * 64
        records[0] = perturbed
        agg = rotation.aggregate_candidate(records)
        self.assertEqual(agg["mirror"]["n_paired"], 7)
        self.assertLess(agg["mirror"]["id_exact_share"], 1.0)
        # The headline is untouched: mirror exactness gates only
        # the mirror population, never the rotations.
        self.assertEqual(agg["headline"]["n_headline"], 8)


class TestValidatorGapBattery(unittest.TestCase):
    """Reject branches of the recompute validator and mirror
    paths the main battery did not reach (found by
    intersecting the full run's coverage with the new ranges
    -- the cycle-174 gap pattern): each break must flip
    _valid_rotation to False through its named branch, and
    each direct mirror call must produce the producer's
    result on the fabricated input."""

    def test_mirror_generator_value_redraw(self):
        # Force blocks 0 and 1 of case 0 to collide at counter
        # 0 in the MIRROR's generator: its whole-set redraw
        # (counters increment) must fire, as the producer's
        # scripted test pins for rotation.generate_cases.
        calls: list[tuple[int, int, int]] = []
        original = diffreports._rotation_derive

        def scripted(seed, index, slot, counter):
            calls.append((index, slot, counter))
            if index == 0 and counter == 0 and slot in (0, 1, 2, 3):
                return 0
            return original(seed, index, slot, counter)

        with mock.patch.object(diffreports, "_rotation_derive", scripted):
            mirrored = diffreports._rotation_generate(0)
        self.assertTrue(
            any(slot < 20 and counter > 0 for _, slot, counter in calls),
            "mirror value redraw never fired under the scripted collision",
        )
        self.assertEqual(len({c["values"][0] for c in mirrored}), 20)

    def test_mirror_aggregate_malformed_record(self):
        agg = diffreports._rotation_aggregate([{"fixture": "rotation-00", "cells": None}])
        self.assertEqual(agg["n_fixtures"], 1)
        self.assertEqual(agg["headline"]["n_headline"], 0)
        self.assertEqual(agg["n_closed_book_incomplete"], 1)
        self.assertFalse(
            diffreports._rotation_id_exact(
                {"arm_inventory_digests": None, "arm_tokens": None}, "rot_1"
            )
        )
        self.assertFalse(diffreports._rotation_boundary_screened({"cell_details": None}))

    def test_mirror_aggregate_absent_false_pass(self):
        verdicts = _all_arms(False)
        verdicts["absent"] = True
        records = [_record(f"rotation-{i:02d}", dict(verdicts)) for i in range(4)]
        agg = diffreports._rotation_aggregate(records)
        self.assertEqual(agg["absent"]["n_false_pass"], 4)
        self.assertEqual(agg["absent"]["rate"], 1.0)

    def test_mirror_selection_unique_winner(self):
        def rec(fixture, rot1):
            verdicts = _all_arms(True)
            verdicts["rot_1"] = rot1
            return _record(fixture, verdicts)

        raw = {
            "a": [rec(f"rotation-{i:02d}", True) for i in range(3)],
            "b": [rec(f"rotation-{i:02d}", False) for i in range(3)],
        }
        outcome = diffreports._rotation_arm_selection(raw, "rot_1")
        self.assertEqual(outcome["decision"], "selected")
        self.assertEqual(outcome["winner"], "a")
        self.assertFalse(outcome["withheld"])
        # Mismatched completed sets across candidates: the
        # selection is undefined (None), never a guess.
        mismatched = {
            "a": [rec(f"rotation-{i:02d}", True) for i in range(3)],
            "b": [rec(f"rotation-{i:02d}", True) for i in range(2)],
        }
        fixtures = frozenset(f"rotation-{i:02d}" for i in range(3))
        self.assertIsNone(diffreports._rotation_selection_over(mismatched, "rot_1", fixtures))

    def test_recompute_reject_branches(self):
        payload = _payload()

        def entry(p):
            return p["candidates"]["rt-a"]

        def record(p):
            return entry(p)["records"][0]

        def cell(p):
            return record(p)["cells"]["rot_1"]

        def total_records_over(p):
            base = copy.deepcopy(entry(p))
            for i in range(23):
                p["candidates"][f"c{i:02d}"] = copy.deepcopy(base)
            p["n_candidates"] = len(p["candidates"])

        breaks = {
            "protocol_not_dict": lambda p: p.update(protocol=[]),
            "protocol_seed_bool": lambda p: p["protocol"].update(seed=True),
            "protocol_fingerprint_not_str": lambda p: p["protocol"].update(backend_fingerprint=7),
            "candidates_empty": lambda p: p.update(candidates={}),
            "entry_not_dict": lambda p: p["candidates"].update({"rt-a": []}),
            "available_detail_nonempty": lambda p: entry(p).update(detail="x"),
            "records_bytes_over": lambda p: record(p).update(topic="x" * (13 * 1024 * 1024)),
            "total_records_over": total_records_over,
            "cells_not_dict": lambda p: record(p).update(cells=[]),
            "record_extra_key": lambda p: record(p).update(extra=1),
            "cell_extra_key": lambda p: cell(p).update(extra=1),
            "details_missing_arm": lambda p: record(p)["cell_details"].pop("mir_3"),
            "cell_verdict_int": lambda p: cell(p).update(verdict=1),
            "cell_boundary_text_match_str": lambda p: cell(p).update(boundary_text_match="x"),
            "cell_input_tokens_negative": lambda p: cell(p).update(input_tokens=-1),
            "cell_answer_digest_malformed": lambda p: cell(p).update(answer_digest="zz"),
        }
        self.assertTrue(_valid_rotation(payload))
        for name, mutate in breaks.items():
            with self.subTest(break_=name):
                broken = copy.deepcopy(payload)
                mutate(broken)
                self.assertFalse(_valid_rotation(broken), name)

    def test_unavailable_reject_branches(self):
        payload = _payload_with_unavailable()
        self.assertTrue(_valid_rotation(payload))
        extra_key = copy.deepcopy(payload)
        extra_key["candidates"]["rt-b"]["extra"] = 1
        self.assertFalse(_valid_rotation(extra_key))
        all_unavailable = copy.deepcopy(payload)
        all_unavailable["candidates"]["rt-a"] = {
            "status": "unavailable",
            "detail": "candidate_evaluation_failed",
            "records": [],
        }
        self.assertFalse(_valid_rotation(all_unavailable))


class _SecondFormatSlotFail(_WordBackend):
    """rt-b (the ``U>`` format) fails whenever the asked fact
    is not the first block; rt-a is perfect -- a unique winner
    on every non-trivial arm, which drives the selection's
    exactly-one-passer return in BOTH the producer and the
    recompute mirror."""

    def generate(self, input_ids, *, max_new_tokens, seed=None, cancel=None):
        text = self.decode(list(input_ids))
        self.mode = "slot_fail" if text.startswith("U>") else "perfect"
        return super().generate(input_ids, max_new_tokens=max_new_tokens, seed=seed, cancel=cancel)


class TestValidatorGapBattery2(unittest.TestCase):
    def test_unique_winner_selection_recomputed(self):
        payload = _payload(_SecondFormatSlotFail())
        self.assertTrue(payload["available"])
        self.assertTrue(_valid_rotation(payload))
        self.assertEqual(payload["conditions"]["rot_1"]["decision"], "selected")
        self.assertEqual(payload["conditions"]["rot_1"]["winner"], "rt-a")
        self.assertEqual(payload["conditions"]["rot_0"]["decision"], "ambiguous")


class TestStageGapBranches(unittest.TestCase):
    """Stage branches the main stage tests did not reach."""

    def test_boundary_decode_failure_screens_nothing(self):
        # A boundary that will not decode (singleton id) falls
        # back to an empty boundary text: the screen is inert,
        # the stage stays available, nothing is screened.
        class _SingletonUndecodable(_WordBackend):
            def decode(self, ids):
                if len(ids) == 1:
                    raise RuntimeError("singleton undecodable")
                return super().decode(ids)

        payload = _payload(_SingletonUndecodable())
        self.assertTrue(payload["available"])
        entry = payload["candidates"]["rt-a"]
        self.assertEqual(entry["n_boundary_screened"], 0)

    def test_cell_none_none_defensive_branch(self):
        # _rotation_cell's contract is exactly one of the pair
        # set; the stage's elif chain must still not invent a
        # cell or a detail when a (patched) helper returns
        # (None, None) -- the record carries the arm as absent
        # with an empty detail, which the validator then
        # rejects downstream rather than the stage guessing.
        with mock.patch.object(experiments, "_rotation_cell", return_value=(None, None)):
            payload = _payload()
        self.assertTrue(payload["available"])
        record = payload["candidates"]["rt-a"]["records"][0]
        self.assertTrue(all(c is None for c in record["cells"].values()))
        self.assertTrue(all(d == "" for d in record["cell_details"].values()))


class TestReportingGapBranches(unittest.TestCase):
    """Section branches the main reporting tests did not
    reach: malformed candidates/conditions containers and a
    non-dict condition entry."""

    def test_section_candidates_not_dict(self):
        html = render_html({"rotation": {"available": True, "candidates": [], "conditions": {}}})
        self.assertNotIn('id="rotation"', html)

    def test_section_entry_not_dict(self):
        html = render_html(
            {
                "rotation": {
                    "available": True,
                    "candidates": {"rt-a": "bogus"},
                    "conditions": {},
                }
            }
        )
        self.assertIn("Context rotation", html)

    def test_section_condition_not_dict_and_count_na(self):
        payload = _payload()
        hostile = copy.deepcopy(payload)
        hostile["conditions"]["rot_0"] = "bogus"
        hostile["candidates"]["rt-a"]["headline"]["n_headline"] = "20"
        html = render_html({"rotation": hostile})
        self.assertIn("Context rotation", html)


class _DispatchFakeHFBackend:
    """Stand-in for LocalHFBackend in the run flow (the
    cycle-179 harness pattern): no weights, no generations."""

    def __init__(self, model, device=None, max_context=None, dtype=None):
        self.model = model
        self.device = device
        self.max_context = max_context
        self.dtype = dtype

    eos_token_id = None


class TestMainDispatchGap(unittest.TestCase):
    """The --placement and --rotate call sites inside
    cli.main's post-hoc block, driven through the real run
    flow with the stage layer stubbed (cycle-179 pattern):
    both flags in one run must attach both payload keys."""

    def test_placement_and_rotation_dispatch(self):
        fixture_doc = {
            "search": [
                {
                    "id": "s1",
                    "group": "g1",
                    "messages": [{"role": "user", "content": "hi"}],
                    "expected": "yo",
                }
            ],
            "held_out": [
                {
                    "id": "h1",
                    "group": "g2",
                    "messages": [{"role": "user", "content": "hey"}],
                    "expected": "yo",
                }
            ],
        }
        result_doc = {
            "decision": "compatible_on_fixture_set",
            "selected_candidate": None,
            "search": [],
            "held_out": {},
        }
        sentinel = {"available": True, "detail": ""}
        with tempfile.TemporaryDirectory() as tmp:
            fixture_path = Path(tmp) / "fixture.json"
            fixture_path.write_text(json.dumps(fixture_doc), encoding="utf-8")
            outdir = Path(tmp) / "report"

            def fake_prepare(path):
                Path(path).mkdir(parents=True, exist_ok=False)
                return Path(path)

            with contextlib.ExitStack() as stack:
                stack.enter_context(
                    mock.patch.object(backends, "LocalHFBackend", _DispatchFakeHFBackend)
                )
                stack.enter_context(
                    mock.patch.object(cli, "run_experiment", return_value=result_doc)
                )
                stack.enter_context(
                    mock.patch.object(
                        cli,
                        "save_report",
                        return_value=(
                            str(outdir / "results.json"),
                            str(outdir / "report.md"),
                        ),
                    )
                )
                stack.enter_context(
                    mock.patch.object(doctor_mod, "template_conformance", return_value={"ok": True})
                )
                stack.enter_context(
                    mock.patch.object(
                        experiments, "budget_frontier", return_value={"available": False}
                    )
                )
                stack.enter_context(
                    mock.patch.object(reporting_mod, "prepare_report_dir", fake_prepare)
                )
                for stage_name in (
                    "placement_stage",
                    "placement_analysis",
                    "rotation_stage",
                    "rotation_analysis",
                ):
                    stack.enter_context(
                        mock.patch.object(experiments, stage_name, return_value=sentinel)
                    )
                argv = [
                    "run",
                    "--model",
                    "ckpt-dir",
                    "--dataset",
                    str(fixture_path),
                    "--family",
                    "chatml",
                    "--boundary-ids",
                    "198,199",
                    "--output",
                    str(outdir),
                    "--seed",
                    "7",
                    "--placement",
                    "--rotate",
                ]
                out, err = io.StringIO(), io.StringIO()
                try:
                    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                        code = cli.main(argv)
                    outcome = ("return", code)
                except SystemExit as exc:
                    outcome = ("exit", exc.code)
        self.assertEqual(outcome, ("return", 0))
        self.assertIs(result_doc["placement"], sentinel)
        self.assertIs(result_doc["rotation"], sentinel)


if __name__ == "__main__":
    unittest.main()
