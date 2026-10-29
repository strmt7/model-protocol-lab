"""Cycle 193: question/payload placement (wave-8 rank 5).

Pins the frozen contract in docs/PLACEMENT_DESIGN.md (design
v2, after red-team A's DO-NOT-BUILD verdict on v1 and red-team
B's cross-check, which refuted A's contradiction claim in part
-- the design's operative reading was total discordance -- while
confirming the population, joint-collapse, duplicate-name, and
screen-accounting findings; B's nine must-fixes were adopted
wholesale): the seed-derived case generator and its redraw
rules, the text-level arm construction, the sandwich control
computed on the headline's own paired population, the
both-fail-dominant joint-collapse state, the total
classification function and its coherence with the single
falsification formula, the union-screened per-arm selections,
the stage's fail-closed vocabulary, the full-recompute diff
validator (one break at a time) with its literal-table mirror,
the diff, the reporting section, and the CLI helper.
"""

from __future__ import annotations

import copy
import re
import threading
import unittest
from unittest import mock

from model_protocol_lab import cli, diffreports, experiments, placement
from model_protocol_lab.core import Format, Generation
from model_protocol_lab.diffreports import _valid_placement, diff_placement
from model_protocol_lab.reporting import render_html

FORMATS = (
    Format(name="pl-a", bos="", user_prefix="u:", assistant_prefix="a:", turn_end="</s>"),
    Format(name="pl-b", bos="", user_prefix="U>", assistant_prefix="A>", turn_end="<e>"),
)

_FACT_RE = re.compile(r"inspection code is ([a-z]+-\d+)\.")


class _WordBackend:
    """Word-token scripted backend with switchable behaviours.

    Modes: ``perfect`` (answer the planted code wherever it
    sits), ``pf_fail`` (answer wrongly when the question sits
    after the payload -- the sandwich still passes, because it
    also asks before, which is exactly what the sandwich
    control is supposed to absorb), ``split`` (payload-first
    always fails and the sandwich follows question-first on
    odd topic indices only, so the headline discordance exceeds
    the control level), ``wrong`` (every answer wrong),
    ``explode`` (generate raises). ``known`` maps a topic to a
    memorized value answered even with no payload in view --
    the closed-book screen trigger.
    """

    fingerprint = "sha256:test-placement"

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
        """Encode."""
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
        """Decode."""
        return " ".join(self.rev.get(int(i), "?") for i in ids)

    # skipcq: PY-R1000 - Gate-pinned frozen function; splitting breaks its verified contract.
    def generate(self, input_ids, *, max_new_tokens, seed=None, cancel=None):
        """Generate."""
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
        found = _FACT_RE.findall(text)
        topic_match = re.search(r"What is the (.+?) inspection code\?", text)
        topic = topic_match.group(1) if topic_match else ""
        if found:
            answer = found[-1]
            question_at = text.find("What is")
            fact_at = text.find("inspection code is")
            if self.mode == "wrong" or (self.mode == "pf_fail" and question_at > fact_at):
                answer = "wrong-0000"
            elif self.mode == "split":
                if question_at > fact_at and text.count("What is") == 1:
                    answer = "wrong-0000"  # payload-first arm
                elif text.count("What is") == 2:
                    index = placement.TOPICS.index(topic) if topic in placement.TOPICS else 0
                    if index % 2 == 0:
                        answer = "wrong-0000"  # sandwich follows payload-first
        elif topic in self.known:
            answer = self.known[topic]
        else:
            answer = "unknown"
        boundary = self.encode("endmark")[0]
        out = list(self.encode(answer)) + [boundary]
        out = (out + [boundary] * max_new_tokens)[:max_new_tokens]
        return Generation(token_ids=tuple(out), reason="limit")


def _payload(backend=None, formats=FORMATS, seed=0, **stage_kwargs):
    """Payload."""
    backend = backend if backend is not None else _WordBackend()
    stage_kwargs.setdefault("max_context", 4096)
    stage = experiments.placement_stage(
        backend,
        formats,
        boundary=tuple(backend.encode("endmark")),
        seed=seed,
        **stage_kwargs,
    )
    return experiments.placement_analysis(stage, tuple(f.name for f in formats))


def _payload_with_unavailable():
    """A coherent payload whose second candidate is an
    unavailable entry (the analysis shape for a candidate the
    stage could not evaluate), built through the real
    analysis so the rest of the payload stays consistent.
    """
    backend = _WordBackend()
    stage = experiments.placement_stage(
        backend,
        FORMATS,
        boundary=tuple(backend.encode("endmark")),
        max_context=4096,
        seed=0,
    )
    stage["candidates"]["pl-b"] = {
        "status": "unavailable",
        "detail": "candidate_evaluation_failed",
        "records": [],
    }
    return experiments.placement_analysis(stage, tuple(f.name for f in FORMATS))


class TestProtocol(unittest.TestCase):
    """Test protocol."""

    def test_constants(self):
        """Test constants."""
        self.assertEqual(
            placement.ARM_KEYS,
            ("question_first", "payload_first", "sandwich", "closed_book"),
        )
        self.assertEqual(placement.LAYOUT_ARMS, ("question_first", "payload_first", "sandwich"))
        self.assertEqual(placement.BASELINE_ARM, "payload_first")
        self.assertEqual(placement.CASE_COUNT, 24)
        self.assertEqual(placement.MIN_PAIRED_EVALUABLE, 6)
        self.assertEqual(placement.MIN_DISCORDANT, 2)
        self.assertEqual(placement.MAX_GROUND_RECORDS, 576)
        self.assertEqual(placement.GENERATION_BUDGET, 2304)

    def test_protocol_object_and_fingerprint(self):
        """Test protocol object and fingerprint."""
        proto = placement.protocol_object(3, "sha256:x")
        self.assertEqual(proto["seed"], 3)
        self.assertEqual(proto["ground_sha256"], placement.ground_sha256(3))
        self.assertNotEqual(placement.ground_sha256(3), placement.ground_sha256(4))
        self.assertEqual(
            placement.case_material_sha256(),
            placement.protocol_object(9, "y")["case_material_sha256"],
        )
        fp = placement.protocol_fingerprint(3, "sha256:x")
        self.assertEqual(len(fp), 64)
        self.assertNotEqual(fp, placement.protocol_fingerprint(3, "sha256:z"))


class TestGenerator(unittest.TestCase):
    """Test generator."""

    def test_deterministic_and_seed_sensitive(self):
        """Test deterministic and seed sensitive."""
        self.assertEqual(placement.generate_cases(7), placement.generate_cases(7))
        self.assertNotEqual(placement.generate_cases(7), placement.generate_cases(8))

    def test_case_shape(self):
        """Test case shape."""
        cases = placement.generate_cases(0)
        self.assertEqual(len(cases), 24)
        for index, case in enumerate(cases):
            self.assertEqual(case["fixture_id"], f"placement-{index:02d}")
            self.assertEqual(case["topic"], placement.TOPICS[index])
            first, second = case["sentence_indices"]
            self.assertNotEqual(first, second)
            self.assertEqual(
                case["payload"],
                placement.NEUTRAL_POOL[first]
                + " "
                + case["fact"]
                + " "
                + placement.NEUTRAL_POOL[second],
            )
            self.assertEqual(
                case["fact"],
                f"The {case['topic']} inspection code is {case['value']}.",
            )
            self.assertEqual(case["question"], f"What is the {case['topic']} inspection code?")
            self.assertEqual(case["expected"], case["value"])

    def test_seed_must_be_int(self):
        """Test seed must be int."""
        with self.assertRaises(ValueError):
            placement.generate_cases("7")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            placement.generate_cases(True)  # type: ignore[arg-type]

    def test_values_distinct_across_seeds(self):
        """Test values distinct across seeds."""
        for seed in range(60):
            values = [case["value"] for case in placement.generate_cases(seed)]
            self.assertEqual(len(set(values)), 24, f"seed {seed}")

    def test_redraw_path_exercised(self):
        """Test redraw path exercised."""
        # Seed 739 collides on a first-draw value, so its
        # generation takes the redraw branch (counter > 0);
        # seed 1322 is a second such seed. The mirror must
        # redraw identically -- a one-sided redraw rule would
        # desynchronize the case set and the ground hash.
        from model_protocol_lab.diffreports import _placement_generate

        for seed in (739, 1322, *range(60)):
            rows = placement.canonical_generated_rows(seed)
            mirror = [
                [c["fixture_id"], c["topic"], c["value"], list(c["sentence_indices"])]
                for c in _placement_generate(seed)
            ]
            self.assertEqual(rows, mirror, f"seed {seed}")

    def test_topic_word_stem_disjoint(self):
        """Test topic word stem disjoint."""
        # F10 residue: no topic token may share a 4-character
        # stem with a code word, and neither may prefix the
        # other -- a topic token must never read as part of the
        # code-word channel.
        topic_tokens = {token for topic in placement.TOPICS for token in topic.split()}
        for token in topic_tokens:
            for word in placement.WORDS:
                self.assertNotEqual(token[:4], word[:4], (token, word))
                self.assertFalse(token.startswith(word), (token, word))
                self.assertFalse(word.startswith(token), (token, word))

    def test_neutral_pool_screen_properties(self):
        """Test neutral pool screen properties."""
        for sentence in placement.NEUTRAL_POOL:
            self.assertFalse(any(ch.isdigit() for ch in sentence), sentence)
            for word in placement.WORDS:
                self.assertNotIn(word, sentence)
            for topic in placement.TOPICS:
                self.assertNotIn(topic, sentence)
            self.assertNotIn("inspection code", sentence)


class TestArmContent(unittest.TestCase):
    """Test arm content."""

    def test_arms_are_question_moves_only(self):
        """Test arms are question moves only."""
        case = placement.generate_cases(0)[0]
        question = case["question"]
        payload = case["payload"]
        joiner = placement.JOINER
        self.assertEqual(placement.arm_content(case, "question_first"), question + joiner + payload)
        self.assertEqual(placement.arm_content(case, "payload_first"), payload + joiner + question)
        self.assertEqual(
            placement.arm_content(case, "sandwich"),
            question + joiner + payload + joiner + question,
        )
        self.assertEqual(placement.arm_content(case, "closed_book"), question)
        self.assertEqual(placement.arm_content(case, "sandwich").count(question), 2)
        self.assertEqual(placement.arm_content(case, "question_first").count(case["fact"]), 1)
        with self.assertRaises(ValueError):
            placement.arm_content(case, "bogus")


class TestClassify(unittest.TestCase):
    """Test classify."""

    def test_truth_table(self):
        """Test truth table."""
        classify = placement.classify_placement
        # Below the paired floor nothing else is read.
        self.assertEqual(classify(5, 0, 0, 0, None, False), "insufficient")
        # Zero discordance, no joint collapse.
        self.assertEqual(classify(24, 0, 0, 0, None, False), "no_discordance_observed")
        # Zero discordance WITH joint collapse is not the
        # no-discordance label: the control decides.
        self.assertEqual(classify(24, 0, 0, 24, 3, True), "within_sandwich_control")
        self.assertEqual(classify(24, 0, 0, 3, None, True), "control_set_too_small")
        # The unreachable-from-aggregates state stays total on
        # direct calls.
        self.assertEqual(classify(24, 2, 0, 24, None, False), "control_unavailable")
        self.assertEqual(classify(24, 1, 1, 24, 2, False), "within_sandwich_control")
        self.assertEqual(classify(24, 2, 2, 24, 1, False), "discordant_balanced")
        self.assertEqual(classify(24, 3, 0, 24, 0, False), "discordant_asymmetric")
        self.assertEqual(classify(24, 1, 0, 24, 0, False), "below_threshold")

    def test_both_fail_dominant(self):
        """Test both fail dominant."""
        self.assertTrue(placement.both_fail_dominant(0, 0, 2))
        self.assertTrue(placement.both_fail_dominant(1, 0, 3))
        self.assertFalse(placement.both_fail_dominant(0, 0, 1))
        self.assertFalse(placement.both_fail_dominant(3, 0, 2))

    def test_falsification_coherence_enumeration(self):
        """Test falsification coherence enumeration."""
        # The registered claim: falsified exactly when the
        # classification lands on one of the two acquitting
        # labels at sufficient population. Enumerated over the
        # full input space, not sampled.
        for n_paired in (0, 5, 6, 12, 24):
            for qf_only in range(7):
                for pf_only in range(7):
                    for n_neither in (0, 1, 2, 5, 12):
                        joint = placement.both_fail_dominant(qf_only, pf_only, n_neither)
                        for n_control in (0, 5, 6, 24):
                            for level in (None, 0, 1, 3, 8):
                                label = placement.classify_placement(
                                    n_paired, qf_only, pf_only, n_control, level, joint
                                )
                                falsified = placement.effect_falsified(
                                    n_paired, qf_only, pf_only, n_control, level, joint
                                )
                                expected = (
                                    n_paired >= 6
                                    and not joint
                                    and label
                                    in (
                                        "no_discordance_observed",
                                        "within_sandwich_control",
                                    )
                                )
                                self.assertEqual(
                                    falsified,
                                    expected,
                                    (n_paired, qf_only, pf_only, n_neither, n_control, level),
                                )


class TestMirrorAgreement(unittest.TestCase):
    """Test mirror agreement."""

    def test_mirror_protocol(self):
        """Test mirror protocol."""
        for seed in (0, 1, 42):
            expected = dict(placement.protocol_object(seed, "sha256:m"))
            expected["fingerprint"] = placement.protocol_fingerprint(seed, "sha256:m")
            self.assertEqual(diffreports._placement_protocol(seed, "sha256:m"), expected)

    def test_mirror_classify_and_falsified(self):
        """Test mirror classify and falsified."""
        for args in [
            (24, 0, 0, 24, None, False),
            (24, 2, 2, 24, 1, False),
            (24, 3, 0, 24, 0, False),
            (24, 0, 0, 24, 3, True),
            (5, 0, 0, 0, None, False),
            (24, 1, 0, 24, 0, False),
        ]:
            self.assertEqual(
                diffreports._placement_classify(*args), placement.classify_placement(*args)
            )
            self.assertEqual(
                diffreports._placement_falsified(*args), placement.effect_falsified(*args)
            )

    def test_mirror_classify_control_states(self):
        """Test mirror classify control states."""
        for args in [(24, 2, 0, 3, None, False), (24, 2, 0, 24, None, False)]:
            self.assertEqual(
                diffreports._placement_classify(*args), placement.classify_placement(*args)
            )
            self.assertEqual(
                diffreports._placement_falsified(*args), placement.effect_falsified(*args)
            )

    def test_mirror_aggregate_malformed_record(self):
        """Test mirror aggregate malformed record."""
        records = [{"fixture": "placement-00", "cells": None}]
        self.assertEqual(
            diffreports._placement_aggregate(records),
            placement.aggregate_candidate(records),
        )

    def test_mirror_selection_outcomes(self):
        """Test mirror selection outcomes."""

        def record(fixture, qf, pf):
            """Record."""
            return {
                "fixture": fixture,
                "cells": {
                    "question_first": {"verdict": qf},
                    "payload_first": {"verdict": pf},
                    "sandwich": {"verdict": qf},
                    "closed_book": {"verdict": False},
                },
            }

        records = {
            "a": [record("placement-00", True, True), record("placement-01", True, False)],
            "b": [record("placement-00", True, True), record("placement-01", False, True)],
        }
        self.assertEqual(
            diffreports._placement_arm_selection(records, "question_first"),
            placement.arm_selection(records, "question_first"),
        )
        mismatched = {
            "a": records["a"],
            "b": [
                record("placement-00", True, True),
                {"fixture": "placement-01", "cells": {"question_first": None}},
            ],
        }
        self.assertEqual(
            diffreports._placement_arm_selection(mismatched, "question_first"),
            placement.arm_selection(mismatched, "question_first"),
        )

    def test_mirror_aggregate_on_stage_records(self):
        """Test mirror aggregate on stage records."""
        payload = _payload()
        for entry in payload["candidates"].values():
            records = entry["records"]
            self.assertEqual(
                diffreports._placement_aggregate(records),
                placement.aggregate_candidate(records),
            )
            self.assertEqual(
                diffreports._placement_arm_selection(
                    {n: e["records"] for n, e in payload["candidates"].items()},
                    "question_first",
                ),
                placement.arm_selection(
                    {n: e["records"] for n, e in payload["candidates"].items()},
                    "question_first",
                ),
            )


if __name__ == "__main__":
    unittest.main()


class TestStage(unittest.TestCase):
    """Test stage."""

    def test_perfect_backend_no_discordance(self):
        """Test perfect backend no discordance."""
        payload = _payload()
        self.assertTrue(payload["available"])
        self.assertTrue(_valid_placement(payload))
        self.assertEqual(payload["n_candidates"], 2)
        self.assertEqual(payload["n_cases"], 24)
        entry = payload["candidates"]["pl-a"]
        self.assertEqual(entry["n_screened"], 0)
        self.assertEqual(entry["n_closed_book_incomplete"], 0)
        self.assertEqual(entry["closed_book"]["pass_rate"], 0.0)
        headline = entry["headline"]
        self.assertEqual(headline["n_paired"], 24)
        self.assertEqual(headline["n_both"], 24)
        self.assertEqual(headline["classification"], "no_discordance_observed")
        self.assertTrue(entry["placement_effect_falsified"])
        self.assertEqual(entry["arms"]["question_first"]["rate"], 1.0)
        # Both candidates pass everything on every layout arm.
        condition = payload["conditions"]["question_first"]
        self.assertEqual(condition["decision"], "ambiguous")
        self.assertFalse(condition["flip"])

    def test_payload_first_failure_absorbed_by_sandwich(self):
        """Test payload first failure absorbed by sandwich."""
        # pf_fail: the payload-first arm fails everywhere, the
        # sandwich (which also asks first) passes everywhere, so
        # the headline discordance equals the control level and
        # the falsification formula acquits -- the F3 scope
        # limit, pinned as behaviour rather than prose.
        payload = _payload(_WordBackend(mode="pf_fail"))
        self.assertTrue(_valid_placement(payload))
        entry = payload["candidates"]["pl-a"]
        headline = entry["headline"]
        self.assertEqual(headline["n_qf_only"], 24)
        self.assertEqual(headline["n_pf_only"], 0)
        self.assertEqual(entry["control_level"], 24)
        self.assertEqual(headline["classification"], "within_sandwich_control")
        self.assertTrue(entry["placement_effect_falsified"])

    def test_split_sandwich_exceeds_control(self):
        """Test split sandwich exceeds control."""
        payload = _payload(_WordBackend(mode="split"))
        self.assertTrue(_valid_placement(payload))
        entry = payload["candidates"]["pl-a"]
        headline = entry["headline"]
        self.assertEqual(headline["n_qf_only"], 24)
        self.assertEqual(entry["control_level"], 12)
        self.assertEqual(headline["classification"], "discordant_asymmetric")
        self.assertFalse(entry["placement_effect_falsified"])

    def test_wrong_backend_joint_collapse(self):
        """Test wrong backend joint collapse."""
        payload = _payload(_WordBackend(mode="wrong"))
        self.assertTrue(_valid_placement(payload))
        entry = payload["candidates"]["pl-a"]
        headline = entry["headline"]
        self.assertEqual(headline["n_neither"], 24)
        self.assertTrue(headline["both_fail_dominant"])
        self.assertFalse(entry["placement_effect_falsified"])
        self.assertNotEqual(headline["classification"], "no_discordance_observed")

    def test_stage_rejects_bad_formats(self):
        """Test stage rejects bad formats."""
        backend = _WordBackend()
        with self.assertRaises(ValueError):
            experiments.placement_stage(backend, [], boundary=(1,), max_context=100)
        with self.assertRaises(ValueError):
            experiments.placement_stage(backend, (), boundary=(1,), max_context=100)
        with self.assertRaises(ValueError):
            experiments.placement_stage(backend, [FORMATS[0]], boundary=(1,), max_context=100)

    def test_duplicate_format_names_rejected(self):
        """Test duplicate format names rejected."""
        # Red-team F7: two formats sharing a name must not
        # silently collapse into one candidate.
        backend = _WordBackend()
        dup = (FORMATS[0], FORMATS[0])
        with self.assertRaises(ValueError):
            experiments.placement_stage(backend, dup, boundary=(1,), max_context=100)

    def test_record_budget(self):
        """Test record budget."""
        backend = _WordBackend()
        formats = tuple(
            Format(name=f"f{i}", bos="", user_prefix="u:", assistant_prefix="a:", turn_end="</s>")
            for i in range(25)
        )
        stage = experiments.placement_stage(
            backend, formats, boundary=(1,), max_context=4096, seed=0
        )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "placement_record_budget_exceeded")

    def test_screen_failed_stage(self):
        """Test screen failed stage."""
        # A boundary whose decoded text occurs inside the
        # generated texts screens the whole stage closed.
        backend = _WordBackend()
        boundary_map = {fmt.name: tuple(backend.encode("the")) for fmt in FORMATS}
        stage = experiments.placement_stage(
            backend,
            FORMATS,
            boundary=tuple(backend.encode("endmark")),
            boundary_map=boundary_map,
            max_context=4096,
            seed=0,
        )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "placement_screen_failed")

    def test_context_exceeded_cells(self):
        """Test context exceeded cells."""
        payload = _payload(max_context=8)
        self.assertTrue(payload["available"])
        self.assertTrue(_valid_placement(payload))
        entry = payload["candidates"]["pl-a"]
        self.assertEqual(entry["headline"]["classification"], "insufficient")
        self.assertEqual(entry["n_closed_book_incomplete"], 24)
        record = entry["records"][0]
        self.assertEqual(set(record["cell_details"].values()), {"arm_context_exceeded"})

    def test_exploding_backend_degrades_cells(self):
        """Test exploding backend degrades cells."""
        payload = _payload(_WordBackend(mode="explode"))
        self.assertTrue(payload["available"])
        self.assertTrue(_valid_placement(payload))
        record = payload["candidates"]["pl-a"]["records"][0]
        self.assertEqual(set(record["cell_details"].values()), {"arm_evaluation_failed"})

    def test_screen_audit_findings_fail_closed(self):
        """Test screen audit findings fail closed."""
        # The frozen-text audit half of the stage screen: the
        # frozen texts carry no token-shaped literals today,
        # so the branch is proven with the audit stubbed to
        # report a finding (the stubbed-guard precedent) --
        # the day a frozen text gains a bracketed marker, the
        # stage must fail closed instead of forging structure.
        backend = _WordBackend()
        with mock.patch(
            "model_protocol_lab.fixtures.audit_special_token_literals",
            return_value=(object(),),
        ):
            stage = experiments.placement_stage(
                backend, FORMATS, boundary=(1,), max_context=4096, seed=0
            )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "placement_screen_failed")

    def test_cancel_at_case_boundary(self):
        """Test cancel at case boundary."""
        # The event lands during setup (the boundary decode),
        # so the case-loop check is the one that unwinds.
        event = threading.Event()

        class _DecodeCancels(_WordBackend):
            """Decode cancels."""

            def decode(self, ids):
                """Decode."""
                event.set()
                return super().decode(ids)

        backend = _DecodeCancels()
        stage = experiments.placement_stage(
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
        """Test cancel before first evaluation."""
        # The event lands in the first arm's render encode, so
        # the cell's pre-evaluation check is the one that
        # raises (never an evaluation failure).
        event = threading.Event()

        class _EncodeCancels(_WordBackend):
            """Encode cancels."""

            def encode(self, text):
                """Encode."""
                ids = super().encode(text)
                if len(text) > 20:  # arm renders only; setup encodes stay clean
                    event.set()
                return ids

        backend = _EncodeCancels()
        stage = experiments.placement_stage(
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
        """Test cancel at arm boundary."""
        # The event lands just after a cell completes (the
        # wrapper sets it), so the next arm's loop check is
        # the one that raises.
        event = threading.Event()
        backend = _WordBackend()
        original = experiments._placement_cell

        def wrapper(*args, **kwargs):
            """Wrapper."""
            outcome = original(*args, **kwargs)
            event.set()
            return outcome

        with mock.patch.object(experiments, "_placement_cell", wrapper):
            stage = experiments.placement_stage(
                backend,
                FORMATS,
                boundary=tuple(backend.encode("endmark")),
                max_context=4096,
                seed=0,
                cancel=event,
            )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "cancelled")

    def test_cell_returning_neither_is_tolerated(self):
        """Test cell returning neither is tolerated.

        _placement_cell's contract sets exactly one of
        (detail, cell); the stage's elif structure is the
        defensive mirror of the needle stage, proven with
        the cell stubbed to return neither -- the record
        keeps an empty cell the validator then rejects,
        which is the fail-closed division of labour.
        """

        def neither(*args, **kwargs):
            """Neither."""
            return None, None

        backend = _WordBackend()
        with mock.patch.object(experiments, "_placement_cell", neither):
            stage = experiments.placement_stage(
                backend,
                FORMATS[:1],
                boundary=tuple(backend.encode("endmark")),
                max_context=4096,
                seed=0,
            )
        self.assertTrue(stage["available"])
        record = stage["candidates"]["pl-a"]["records"][0]
        self.assertIsNone(record["cells"]["question_first"])
        self.assertEqual(record["cell_details"]["question_first"], "")

    def test_analysis_unavailable_candidate(self):
        """Test analysis unavailable candidate."""
        payload = _payload_with_unavailable()
        self.assertEqual(
            payload["candidates"]["pl-b"],
            {"status": "unavailable", "detail": "candidate_evaluation_failed", "records": []},
        )
        self.assertTrue(_valid_placement(payload))

    def test_cancel_unwinds_stage(self):
        """Test cancel unwinds stage."""
        backend = _WordBackend(cancel_after=3)
        event = threading.Event()
        backend.cancel_event = event
        stage = experiments.placement_stage(
            backend,
            FORMATS,
            boundary=tuple(backend.encode("endmark")),
            max_context=4096,
            seed=0,
            cancel=event,
        )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "cancelled")
        payload = experiments.placement_analysis(stage, tuple(f.name for f in FORMATS))
        self.assertFalse(payload["available"])


class TestScreeningAndConditions(unittest.TestCase):
    """Test screening and conditions."""

    def test_closed_book_screen_excludes_case(self):
        """Test closed book screen excludes case."""
        # The backend has memorized the first case's value; the
        # closed-book cell passes, so the case leaves the live
        # population and the headline pairs over 23.
        first = placement.generate_cases(0)[0]
        backend = _WordBackend(known={first["topic"]: first["value"]})
        payload = _payload(backend)
        self.assertTrue(_valid_placement(payload))
        entry = payload["candidates"]["pl-a"]
        self.assertEqual(entry["n_screened"], 1)
        self.assertEqual(entry["closed_book"]["n_pass"], 1)
        self.assertEqual(entry["headline"]["n_paired"], 23)

    def test_union_screening_across_candidates(self):
        """Test union screening across candidates."""
        # F14: selections run over the population screened for
        # NO candidate in the run, even when screening itself
        # is per-candidate. Candidate A's closed-book pass on
        # case 0 must shrink candidate B's selection population
        # too -- visible because B's own aggregates keep all 24.
        first = placement.generate_cases(0)[0]

        class _PerFormatBackend(_WordBackend):
            """Per format backend."""

            def generate(self, input_ids, **kwargs):
                """Generate."""
                text = self.decode(list(input_ids))
                if text.startswith("u:") and first["topic"] in text and "code is" not in text:
                    boundary = self.encode("endmark")[0]
                    out = list(self.encode(first["value"])) + [boundary]
                    out = (out + [boundary] * kwargs["max_new_tokens"])[: kwargs["max_new_tokens"]]
                    return Generation(token_ids=tuple(out), reason="limit")
                return super().generate(input_ids, **kwargs)

        payload = _payload(_PerFormatBackend())
        self.assertTrue(_valid_placement(payload))
        self.assertEqual(payload["candidates"]["pl-a"]["n_screened"], 1)
        self.assertEqual(payload["candidates"]["pl-b"]["n_screened"], 0)
        shared = placement.shared_live_fixtures(
            {n: e["records"] for n, e in payload["candidates"].items()}
        )
        self.assertEqual(len(shared), 23)
        self.assertNotIn(first["fixture_id"], shared)
        for arm in placement.LAYOUT_ARMS:
            self.assertFalse(payload["conditions"][arm]["withheld"], arm)

    def test_shared_live_fixtures_empty(self):
        """Test shared live fixtures empty."""
        self.assertEqual(placement.shared_live_fixtures({}), frozenset())

    def test_aggregate_tolerates_malformed_record(self):
        """Test aggregate tolerates malformed record."""
        # The aggregator's cell accessor fails soft on a
        # non-dict cells field (the validator, not the
        # aggregator, is the layer that rejects such records).
        aggregate = placement.aggregate_candidate([{"fixture": "placement-00", "cells": None}])
        self.assertEqual(aggregate["headline"]["n_paired"], 0)
        self.assertEqual(aggregate["n_closed_book_incomplete"], 1)

    def test_selection_selected_and_flip(self):
        """Test selection selected and flip."""

        def record(fixture, qf, pf):
            """Record."""
            return {
                "fixture": fixture,
                "cells": {
                    "question_first": {"verdict": qf},
                    "payload_first": {"verdict": pf},
                    "sandwich": {"verdict": qf},
                    "closed_book": {"verdict": False},
                },
            }

        records = {
            "a": [record("placement-00", True, True), record("placement-01", True, False)],
            "b": [record("placement-00", True, True), record("placement-01", False, True)],
        }
        qf = placement.arm_selection(records, "question_first")
        self.assertEqual(qf["decision"], "selected")
        self.assertEqual(qf["winner"], "a")
        # Payload-first selects b on the same population: the
        # question-first winner is a genuine flip.
        self.assertTrue(qf["flip"])
        pf = placement.arm_selection(records, "payload_first")
        self.assertEqual(pf["winner"], "b")
        self.assertFalse(pf["flip"])  # the baseline never flips against itself

    def test_selection_withheld_on_mismatched_completed_sets(self):
        """Test selection withheld on mismatched completed sets."""
        # Same recorded fixtures, but candidate B never
        # completed the arm on one shared fixture: the
        # completed sets differ, so the selection is withheld
        # rather than compared across unequal denominators.
        records_a = [
            {"fixture": "placement-00", "cells": {"question_first": {"verdict": True}}},
            {"fixture": "placement-01", "cells": {"question_first": {"verdict": True}}},
        ]
        records_b = [
            {"fixture": "placement-00", "cells": {"question_first": {"verdict": True}}},
            {"fixture": "placement-01", "cells": {"question_first": None}},
        ]
        outcome = placement.arm_selection({"a": records_a, "b": records_b}, "question_first")
        self.assertTrue(outcome["withheld"])
        self.assertEqual(outcome["detail"], "selection_withheld_mismatched_sets")
        self.assertIsNone(outcome["flip"])


class TestValidatorBattery(unittest.TestCase):
    """One break at a time against a coherent payload."""

    @staticmethod
    def _breaks():
        """Breaks."""
        base = _payload()

        def mutate(fn):
            """Mutate."""
            payload = copy.deepcopy(base)
            fn(payload)
            return payload

        def rec(payload, name="pl-a", index=0):
            """Rec."""
            return payload["candidates"][name]["records"][index]

        yield "baseline_valid", base, True
        yield (
            "verdict_flip",
            mutate(
                lambda p: rec(p)["cells"]["question_first"].__setitem__(
                    "verdict", not rec(p)["cells"]["question_first"]["verdict"]
                )
            ),
            False,
        )
        yield (
            "answer_digest_foreign",
            mutate(
                lambda p: rec(p)["cells"]["question_first"].__setitem__(
                    "answer_digest", placement.digest_text("someone else's answer")
                )
            ),
            False,
        )
        yield (
            "input_tokens_off_by_one",
            mutate(
                lambda p: rec(p)["cells"]["payload_first"].__setitem__(
                    "input_tokens", rec(p)["cells"]["payload_first"]["input_tokens"] + 1
                )
            ),
            False,
        )
        yield (
            "arm_tokens_off_by_one",
            mutate(
                lambda p: rec(p)["arm_tokens"].__setitem__(
                    "sandwich", rec(p)["arm_tokens"]["sandwich"] + 1
                )
            ),
            False,
        )
        yield (
            "question_tokens_mismatch",
            mutate(lambda p: rec(p).__setitem__("question_tokens", rec(p)["question_tokens"] + 1)),
            False,
        )
        yield (
            "expected_digest_wrong",
            mutate(
                lambda p: rec(p).__setitem__("expected_digest", placement.digest_text("wrong-0000"))
            ),
            False,
        )
        yield "topic_wrong", mutate(lambda p: rec(p).__setitem__("topic", "boiler room")), False
        yield (
            "records_reordered",
            mutate(lambda p: p["candidates"]["pl-a"]["records"].reverse()),
            False,
        )
        yield (
            "protocol_seed_shifted",
            mutate(lambda p: p["protocol"].__setitem__("seed", p["protocol"]["seed"] + 1)),
            False,
        )
        yield (
            "ground_hash_zeroed",
            mutate(lambda p: p["protocol"].__setitem__("ground_sha256", "0" * 64)),
            False,
        )
        yield (
            "aggregate_screened_inflated",
            mutate(
                lambda p: p["candidates"]["pl-a"].__setitem__(
                    "n_screened", p["candidates"]["pl-a"]["n_screened"] + 1
                )
            ),
            False,
        )
        yield (
            "classification_forged",
            mutate(
                lambda p: p["candidates"]["pl-a"]["headline"].__setitem__(
                    "classification", "discordant_asymmetric"
                )
            ),
            False,
        )
        yield (
            "condition_winner_forged",
            mutate(lambda p: p["conditions"]["question_first"].__setitem__("winner", "pl-a")),
            False,
        )
        yield "n_cases_wrong", mutate(lambda p: p.__setitem__("n_cases", 23)), False
        yield "extra_top_level_key", mutate(lambda p: p.__setitem__("extra", 1)), False
        yield (
            "cell_with_detail",
            mutate(
                lambda p: rec(p)["cell_details"].__setitem__(
                    "question_first", "arm_evaluation_failed"
                )
            ),
            False,
        )
        yield (
            "missing_cell_no_detail",
            mutate(lambda p: rec(p)["cells"].__setitem__("sandwich", None)),
            False,
        )
        yield (
            "unavailable_with_records",
            mutate(
                lambda p: p["candidates"].__setitem__(
                    "ghost",
                    {
                        "status": "unavailable",
                        "detail": "candidate_evaluation_failed",
                        "records": [{}],
                    },
                )
            ),
            False,
        )
        yield ("protocol_not_dict", mutate(lambda p: p.__setitem__("protocol", "bogus")), False)
        yield ("candidates_empty", mutate(lambda p: p.__setitem__("candidates", {})), False)
        yield ("seed_bool", mutate(lambda p: p["protocol"].__setitem__("seed", True)), False)
        yield ("detail_field_set", mutate(lambda p: p.__setitem__("detail", "x")), False)
        yield ("n_candidates_wrong", mutate(lambda p: p.__setitem__("n_candidates", 99)), False)
        yield (
            "entry_not_dict",
            mutate(lambda p: p["candidates"].__setitem__("pl-a", "bogus")),
            False,
        )
        yield (
            "entry_detail_set",
            mutate(lambda p: p["candidates"]["pl-a"].__setitem__("detail", "x")),
            False,
        )
        yield (
            "entry_extra_key",
            mutate(lambda p: p["candidates"]["pl-a"].__setitem__("zzz", 1)),
            False,
        )
        yield (
            "records_not_list",
            mutate(lambda p: p["candidates"]["pl-a"].__setitem__("records", {})),
            False,
        )
        yield ("records_short", mutate(lambda p: p["candidates"]["pl-a"]["records"].pop()), False)
        yield ("record_extra_key", mutate(lambda p: rec(p).__setitem__("zzz", 1)), False)
        yield (
            "arm_tokens_missing_key",
            mutate(lambda p: rec(p)["arm_tokens"].__delitem__("sandwich")),
            False,
        )
        yield (
            "arm_tokens_negative",
            mutate(lambda p: rec(p)["arm_tokens"].__setitem__("sandwich", -1)),
            False,
        )
        yield (
            "cells_missing_key",
            mutate(lambda p: rec(p)["cells"].__delitem__("sandwich")),
            False,
        )
        yield (
            "details_missing_key",
            mutate(lambda p: rec(p)["cell_details"].__delitem__("sandwich")),
            False,
        )
        yield (
            "cell_extra_key",
            mutate(lambda p: rec(p)["cells"]["question_first"].__setitem__("zzz", 1)),
            False,
        )
        yield (
            "cell_verdict_not_bool",
            mutate(lambda p: rec(p)["cells"]["question_first"].__setitem__("verdict", 1)),
            False,
        )
        yield (
            "cell_boundary_text_match_type",
            mutate(
                lambda p: rec(p)["cells"]["question_first"].__setitem__(
                    "boundary_text_match", "yes"
                )
            ),
            False,
        )
        yield (
            "cell_input_tokens_type",
            mutate(lambda p: rec(p)["cells"]["question_first"].__setitem__("input_tokens", "12")),
            False,
        )
        yield (
            "cell_digest_malformed",
            mutate(lambda p: rec(p)["cells"]["question_first"].__setitem__("answer_digest", "zz")),
            False,
        )

        def _all_unavailable(p):
            """All unavailable."""
            for name in ("pl-a", "pl-b"):
                p["candidates"][name] = {
                    "status": "unavailable",
                    "detail": "candidate_evaluation_failed",
                    "records": [],
                }

        yield ("all_candidates_unavailable", mutate(_all_unavailable), False)

        def _total_cap(p):
            """Total cap."""
            entry = p["candidates"]["pl-a"]
            p["candidates"] = {f"c{i:02d}": copy.deepcopy(entry) for i in range(25)}

        yield ("total_records_cap", mutate(_total_cap), False)

        def _bytes_cap(p):
            """Bytes cap."""
            entry = p["candidates"]["pl-a"]
            p["candidates"] = {f"c{i:03d}": copy.deepcopy(entry) for i in range(650)}

        yield ("records_bytes_cap", mutate(_bytes_cap), False)

        def _unavailable_legit(p):
            """Unavailable legit."""
            p["candidates"]["pl-b"] = {
                "status": "unavailable",
                "detail": "candidate_evaluation_failed",
                "records": [],
            }
            records = {"pl-a": p["candidates"]["pl-a"]["records"]}
            p["conditions"] = {
                arm: placement.arm_selection(records, arm) for arm in placement.LAYOUT_ARMS
            }

        yield ("unavailable_candidate_legit", mutate(_unavailable_legit), True)

        def _none_digest_consistent(p):
            """None digest consistent."""
            record = p["candidates"]["pl-a"]["records"][0]
            cell = record["cells"]["question_first"]
            cell["answer_digest"] = None
            cell["boundary_found"] = False
            cell["verdict"] = False
            p["candidates"]["pl-a"].update(
                placement.aggregate_candidate(p["candidates"]["pl-a"]["records"])
            )
            records = {n: e["records"] for n, e in p["candidates"].items()}
            p["conditions"] = {
                arm: placement.arm_selection(records, arm) for arm in placement.LAYOUT_ARMS
            }

        yield ("answer_digest_none_consistent", mutate(_none_digest_consistent), True)

    def test_battery(self):
        """Test battery."""
        for name, payload, expected in self._breaks():
            with self.subTest(breakage=name):
                self.assertEqual(_valid_placement(payload), expected, name)


class TestDiff(unittest.TestCase):
    """Test diff."""

    def test_identical_payloads_unchanged(self):
        """Test identical payloads unchanged."""
        payload = _payload()
        diff = diff_placement(payload, copy.deepcopy(payload))
        self.assertEqual(diff["available"], "both")
        self.assertFalse(diff["changed"])
        self.assertTrue(diff["comparable"])

    def test_classification_drift(self):
        """Test classification drift."""
        old = _payload()
        new = _payload(_WordBackend(mode="split"))
        diff = diff_placement(old, new)
        self.assertTrue(diff["changed"])
        metrics = {entry["metric"] for entry in diff["candidate_drift"]}
        self.assertIn("classification", metrics)
        self.assertIn("placement_effect_falsified", metrics)

    def test_foreign_backend_not_comparable(self):
        """Test foreign backend not comparable."""
        old = _payload()
        new = _payload()
        protocol = dict(placement.protocol_object(0, "sha256:other-backend"))
        protocol["fingerprint"] = placement.protocol_fingerprint(0, "sha256:other-backend")
        new["protocol"] = protocol
        diff = diff_placement(old, new)
        self.assertEqual(diff["available"], "both")
        self.assertFalse(diff["comparable"])
        self.assertTrue(diff["backend_drift"])
        self.assertTrue(diff["changed"])

    def test_both_unavailable_no_status_drift(self):
        """Test both unavailable no status drift."""
        old = _payload_with_unavailable()
        diff = diff_placement(old, copy.deepcopy(old))
        self.assertEqual(diff["available"], "both")
        self.assertFalse(diff["changed"])
        self.assertEqual(diff["candidate_drift"], [])

    def test_status_drift(self):
        """Test status drift."""
        old = _payload()
        new = _payload_with_unavailable()
        diff = diff_placement(old, new)
        self.assertEqual(diff["available"], "both")
        self.assertTrue(diff["changed"])
        status_entries = [e for e in diff["candidate_drift"] if e["metric"] == "status"]
        self.assertEqual(len(status_entries), 1)
        self.assertEqual(status_entries[0]["candidate"], "pl-b")
        self.assertEqual(status_entries[0]["old"], "available")
        self.assertEqual(status_entries[0]["new"], "unavailable")

    def test_missing_sides(self):
        """Test missing sides."""
        payload = _payload()
        self.assertEqual(diff_placement(None, None)["available"], "neither")
        self.assertEqual(diff_placement(payload, None)["available"], "old")
        self.assertEqual(diff_placement(None, payload)["available"], "new")


class TestReportingAndCli(unittest.TestCase):
    """Test reporting and cli."""

    def test_section_renders(self):
        """Test section renders."""
        payload = _payload()
        html = render_html({"placement": payload})
        self.assertIn('<section class="card" id="placement">', html)
        self.assertIn("Question/payload placement", html)
        self.assertIn("no_discordance_observed", html)
        # Counts and digests only: no case values are rendered.
        first = placement.generate_cases(0)[0]
        self.assertNotIn(first["value"], html)

    def test_section_absent_without_payload(self):
        """Test section absent without payload."""
        html = render_html({})
        self.assertNotIn('<section class="card" id="placement">', html)

    def test_section_hostile_payload(self):
        """Test section hostile payload."""
        html = render_html({"placement": {"available": True, "candidates": []}})
        self.assertNotIn('<section class="card" id="placement">', html)

    def test_section_na_rates(self):
        """Test section na rates."""
        payload = _payload(max_context=8)
        html = render_html({"placement": payload})
        self.assertIn('<section class="card" id="placement">', html)
        self.assertIn("n/a", html)

    def test_section_unavailable_row(self):
        """Test section unavailable row."""
        html = render_html({"placement": _payload_with_unavailable()})
        self.assertIn('<section class="card" id="placement">', html)
        self.assertIn("unavailable", html)

    def test_section_withheld_and_missing_conditions(self):
        """Test section withheld and missing conditions."""
        payload = _payload()
        payload["conditions"]["question_first"] = {
            "withheld": True,
            "winner": None,
            "flip": None,
            "decision": "unavailable",
            "detail": "selection_withheld_mismatched_sets",
        }
        del payload["conditions"]["sandwich"]
        html = render_html({"placement": payload})
        self.assertIn("selection withheld", html)

    def test_section_sparse_available_entry(self):
        """Test section sparse available entry."""
        html = render_html(
            {
                "placement": {
                    "available": True,
                    "candidates": {"x": {"status": "available"}},
                    "conditions": {},
                }
            }
        )
        self.assertIn('<section class="card" id="placement">', html)
        self.assertIn("Sandwich control level n/a", html)

    def test_section_non_dict_candidate(self):
        """Test section non dict candidate."""
        html = render_html(
            {"placement": {"available": True, "candidates": {"x": "bogus"}, "conditions": {}}}
        )
        self.assertIn('<section class="card" id="placement">', html)

    def test_cli_helper_attaches_payload(self):
        """Test cli helper attaches payload."""
        backend = _WordBackend()
        result: dict = {}
        cli._run_placement_probe(
            backend,
            FORMATS,
            result,
            boundary=tuple(backend.encode("endmark")),
            boundary_map=None,
            max_new_tokens=64,
            max_context=4096,
            seed=0,
            cancel=None,
        )
        self.assertTrue(result["placement"]["available"])
        self.assertTrue(_valid_placement(result["placement"]))

    def test_cli_helper_never_raises(self):
        """Test cli helper never raises."""

        class _Boom:
            """Boom."""

            fingerprint = "sha256:boom"

            @staticmethod
            def encode(text):
                """Encode."""
                raise RuntimeError("boom")

        result: dict = {}
        cli._run_placement_probe(
            _Boom(),
            FORMATS,
            result,
            boundary=(1,),
            boundary_map=None,
            max_new_tokens=64,
            max_context=2048,
            seed=0,
            cancel=None,
        )
        self.assertEqual(
            result["placement"], {"available": False, "detail": "placement_unavailable"}
        )
