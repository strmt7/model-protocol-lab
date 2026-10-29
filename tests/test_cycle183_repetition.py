"""Cycle 183 -- prompt repetition x template (`run --repeat`).

The probe (wave-7 rank 4, arXiv:2512.14982) asks whether repeating
the prompt changes verdicts when the chat template is the variable:
x1 is the recorded native search (reused, never regenerated), x2
repeats the last user message inside the same turn, and pad is a
token-length-matched filler control, so the headline contrast
(x2 - pad) is repetition with length held constant.

Red-team A returned DO NOT BUILD AS SPECIFIED on the research
sketch (8 defects: the id-span invariant is false for ordinary
tokenizers, the prefill cell cannot express a pass rate, the
decision vocabularies were unmapped, the padding control was
unsatisfiable as specified, and more). Red-team B upheld A in
substance, amended ten gates, and added G19-G22 (selection
withholding, the wrapper's native-search gate, search-split-only
ground records, pad-share disclosure). This file pins the amended
design: reconstructive construction oracles, paired contrasts,
per-fixture degradation, the withholding rule, the validator's
full recompute, and three mutation gates over repetition.py.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from model_protocol_lab import cli as cli_module
from model_protocol_lab import diffreports, experiments, repetition
from model_protocol_lab.core import Fixture, Format, Generation, Message

REPO_ROOT = Path(__file__).resolve().parent.parent

FMT_A = Format(name="cand-a", bos="", user_prefix="u:", assistant_prefix="a:", turn_end="</s>")
FMT_B = Format(name="cand-b", bos="", user_prefix="U>", assistant_prefix="A>", turn_end="<e>")
FMT_THINK = Format(
    name="cand-think",
    bos="",
    user_prefix="u:",
    assistant_prefix="a:",
    turn_end="</s>",
    open_suffix="<think>",
)

CASE_A = Fixture(
    id="fx-a",
    group="g",
    messages=(Message(role="user", content="markone what is two plus two"),),
    expected="four",
)
CASE_B = Fixture(
    id="fx-b",
    group="g",
    messages=(Message(role="user", content="marktwo name a color"),),
    expected="blue",
)
CASE_C = Fixture(
    id="fx-c",
    group="g",
    messages=(Message(role="user", content="markthree name an animal"),),
    expected="cat",
)
CASES = (CASE_A, CASE_B, CASE_C)
EXPECTED_BY_MARKER = {"markone": "four", "marktwo": "blue", "markthree": "cat"}
# Full target contents per marker: an x2 prompt contains the
# content doubled around the joiner; a pad prompt contains it once.
# (Detecting the pad arm by the filler word "orchard" would fail
# whenever the sizing search matches with a short truncated unit.)
CONTENTS_BY_MARKER = {
    "markone": "markone what is two plus two",
    "marktwo": "marktwo name a color",
    "markthree": "markthree name an animal",
    "markrain": "markrain will it rain today",
}


class _WordBackend:
    """Word-level scripted backend: ids in first-seen order.

    The stage never generates x1; an x2 prompt contains the target
    content doubled around the joiner, a pad prompt contains it
    once (see CONTENTS_BY_MARKER for why filler-word detection is
    not used). Pass tables are keyed by (marker, arm). ``fail_on``
    holds (marker, arm) pairs whose generation raises;
    ``set_cancel_on`` is the 1-based generate call that sets the
    run's cancel event.
    """

    fingerprint = "sha256:scripted"

    def __init__(
        self,
        passes: dict | None = None,
        *,
        fail_on: frozenset = frozenset(),
        fail_fragments: tuple = (),
        set_cancel_on: int | None = None,
        boundary_id: int = 999,
        answers: dict | None = None,
    ) -> None:
        self._passes = passes or {}
        self._fail_on = fail_on
        self._fail_fragments = fail_fragments
        self._set_cancel_on = set_cancel_on
        self._boundary_id = boundary_id
        self._answers = answers or EXPECTED_BY_MARKER
        self._vocab: dict[str, int] = {}
        self.generate_calls = 0
        self.prompts: list[str] = []

    def encode(self, text: str) -> tuple[int, ...]:
        """Encode."""
        ids = []
        for word in text.split(" "):
            if word not in self._vocab:
                self._vocab[word] = len(self._vocab) + 1
            ids.append(self._vocab[word])
        return tuple(ids)

    def decode(self, ids: Any) -> str:
        """Decode."""
        inv = {v: k for k, v in self._vocab.items()}
        return " ".join(inv.get(i, "?") for i in ids)

    # skipcq: PY-R1000 - Gate-pinned frozen function; splitting breaks its verified contract.
    def generate(
        self, input_ids: Any, *, max_new_tokens: int, seed: int, cancel: Any = None
    ) -> Generation:
        """Generate."""
        self.generate_calls += 1
        prompt = self.decode(input_ids)
        self.prompts.append(prompt)
        if (
            self._set_cancel_on is not None
            and self.generate_calls == self._set_cancel_on
            and cancel is not None
        ):
            cancel.set()
        marker = next((m for m in self._answers if m in prompt), None)
        content = CONTENTS_BY_MARKER.get(marker) if marker is not None else None
        if content is not None:
            arm = "x2" if (content + "\n" + content) in prompt else "pad"
        else:
            arm = "pad" if "orchard" in prompt else "x2"
        if (marker, arm) in self._fail_on:
            raise RuntimeError("scripted generation failure")
        if (
            arm == "x2"
            and self._fail_fragments
            and all(fragment in prompt for fragment in self._fail_fragments)
        ):
            raise RuntimeError("scripted generation failure")
        ok = self._passes.get((marker, arm), False)
        answer = self._answers[marker] if ok and marker is not None else "wrong"
        # The Backend contract: reason "limit" means exactly the cap
        # was generated; the boundary cut discards the padding.
        ids = (*self.encode(answer), self._boundary_id)
        ids += (self._boundary_id,) * max(0, max_new_tokens - len(ids))
        return Generation(token_ids=ids, reason="limit")


class _SeamBackend:
    """Char-level backend whose "d\\n" digraph merges into one id.

    Models a merge-y BPE seam: the x1 content region stays locatable
    (no digraph inside it), the joiner region stays a single id, but
    the x2 junction merges, so the construction is text_level with
    recorded seam ids. Boundary id 7 decodes to a control char that
    cannot trip the filler screen.
    """

    fingerprint = "sha256:seam"

    def __init__(self, passes: dict | None = None) -> None:
        self._passes = passes or {}
        self.generate_calls = 0

    @staticmethod
    def encode(text: str) -> tuple[int, ...]:
        """Encode."""
        merged = text.replace("d\n", "\x01")
        return tuple(ord(ch) for ch in merged)

    @staticmethod
    def decode(ids: Any) -> str:
        """Decode."""
        return "".join(chr(i) for i in ids)

    def generate(
        self, input_ids: Any, *, max_new_tokens: int, seed: int, cancel: Any = None
    ) -> Generation:
        """Generate."""
        self.generate_calls += 1
        prompt = self.decode(input_ids)
        # The one seam scenario sizes its pad with a 16-char filler
        # prefix, which still contains "the lane".
        arm = "pad" if "the lane" in prompt else "x2"
        ok = self._passes.get(("solo", arm), False)
        answer = "four" if ok else "wrong"
        ids = (*self.encode(answer), 7)
        ids += (7,) * max(0, max_new_tokens - len(ids))
        return Generation(token_ids=ids, reason="limit")


def _entry(
    backend: Any,
    fmt: Format,
    verdicts: dict[str, bool],
    cases: tuple = CASES,
    *,
    eligible: bool = True,
    btm: Any = None,
) -> dict[str, Any]:
    """A recorded-native search entry whose x1 ids are the real renders."""
    observations = []
    for case in cases:
        observations.append(
            {
                "candidate": fmt.name,
                "fixture": case.id,
                "input_ids": list(backend.encode(fmt.render(case.messages))),
                "output_ids": [4, 999],
                "answer": "recorded",
                "exact_match": verdicts[case.id],
                "boundary_found": True,
                "boundary_text_match": btm,
                "raw_reason": "limit",
                "error": None,
                "status": "completed",
            }
        )
    return {
        "format": {"name": fmt.name},
        "boundary_token_ids": [999],
        "summary": {"aggregate_eligible": eligible},
        "observations": observations,
    }


def _run_stage(backend: Any, formats: tuple, cases: tuple, search: list, **kwargs: Any) -> dict:
    """Run stage."""
    kwargs.setdefault("boundary", (999,))
    kwargs.setdefault("max_context", 100_000)
    return experiments.repetition_stage(backend, formats, cases, search, **kwargs)


def _payload(backend: Any, formats: tuple, cases: tuple, search: list, **kwargs: Any) -> dict:
    """Payload."""
    stage = _run_stage(backend, formats, cases, search, **kwargs)
    return experiments.repetition_analysis(
        stage, "no_candidate", None, tuple(fmt.name for fmt in formats)
    )


class TestProtocolPins(unittest.TestCase):
    """Test protocol pins."""

    def test_protocol_constants_frozen(self):
        """Test protocol constants frozen."""
        self.assertEqual(repetition.PROTOCOL_VERSION, 1)
        self.assertEqual(repetition.JOINER, "\n")
        self.assertEqual(repetition.DETECTOR_VERSION, 1)
        self.assertEqual(repetition.SIZING_RULE, "whole-units-plus-one-truncated-suffix-v1")
        self.assertEqual(repetition.MAX_GROUND_RECORDS, 2400)
        self.assertEqual(repetition.SIZING_SEARCH_BUDGET, 512)
        self.assertEqual(
            hashlib.sha256(repetition.FILLER_UNIT.encode("utf-8")).hexdigest(),
            "60e23143d37e7443cf9c472a2ece340622052d782f6f67af315b46f6477df7af",
        )

    def test_fingerprint_pinned_and_id_order_free(self):
        """Test fingerprint pinned and id order free."""
        self.assertEqual(
            repetition.protocol_fingerprint(["fx-a", "fx-b"], "sha256:scripted"),
            repetition.protocol_fingerprint(["fx-b", "fx-a"], "sha256:scripted"),
        )
        self.assertEqual(
            repetition.protocol_fingerprint(["f1"], "sha256:scripted"),
            "7cadd875cb05c2827d720aba50129a2df3ffb933259b13a4f86149879349767a",
        )
        block = repetition.protocol_block(["f1"], "sha256:scripted")
        self.assertEqual(
            block["fingerprint"], repetition.protocol_fingerprint(["f1"], "sha256:scripted")
        )
        self.assertEqual(block["fixture_set_sha256"], repetition.fixture_set_sha256(["f1"]))

    def test_filler_unit_shape(self):
        """Test filler unit shape."""
        # Trailing space is deliberate (the truncated-suffix sizing
        # starts mid-flow, never at a hard stop).
        self.assertTrue(repetition.FILLER_UNIT.endswith(" "))
        self.assertFalse(any(ch.isdigit() for ch in repetition.FILLER_UNIT))


class TestDetector(unittest.TestCase):
    """Test detector."""

    def test_exact_halves_flagged(self):
        """Test exact halves flagged."""
        self.assertTrue(repetition.detect_pre_repeated("the cat sat the cat sat"))
        self.assertTrue(repetition.detect_pre_repeated("the cat satthe cat sat"))

    def test_ha_ha_ha_ha_flagged_by_clause_i(self):
        """Test ha ha ha ha flagged by clause i."""
        # B's pinned example: t = "ha ha", sep = " ".
        self.assertTrue(repetition.detect_pre_repeated("ha ha ha ha"))

    def test_sentence_repeat_flagged_via_period_inclusive_half(self):
        """Test sentence repeat flagged via period inclusive half."""
        # Red-team B's brief called this a non-example ("separator
        # is '. '"), but execution disproved the annotation: the
        # string decomposes as t + " " + t with t = "the cat sat."
        # (the period belongs to the repeated half), which clause
        # (i) admits. The frozen definition governs, not the brief.
        self.assertTrue(repetition.detect_pre_repeated("the cat sat. the cat sat."))
        # A near-repeat that differs inside the second half is not
        # flagged: no admitted decomposition exists.
        self.assertFalse(repetition.detect_pre_repeated("the cat sat. the dog sat."))

    def test_one_word_changed_not_flagged(self):
        """Test one word changed not flagged."""
        self.assertFalse(repetition.detect_pre_repeated("the cat sat the dog sat"))

    def test_adjacent_long_substring_flagged_by_clause_ii(self):
        """Test adjacent long substring flagged by clause ii."""
        # Clause (i) cannot fire (the halves differ); two adjacent
        # copies of a 10-char substring can.
        self.assertTrue(repetition.detect_pre_repeated("xxabcdefghijabcdefghijyy"))

    def test_short_repeats_not_flagged(self):
        """Test short repeats not flagged."""
        self.assertFalse(repetition.detect_pre_repeated("abcabcab"))
        self.assertFalse(repetition.detect_pre_repeated("short"))
        self.assertFalse(repetition.detect_pre_repeated(""))

    def test_whitespace_collapsed_before_detection(self):
        """Test whitespace collapsed before detection."""
        self.assertTrue(repetition.detect_pre_repeated("the  cat\nsat the cat sat"))


class TestPureHelpers(unittest.TestCase):
    """Test pure helpers."""

    def test_is_vacuous(self):
        """Test is vacuous."""
        self.assertTrue(repetition.is_vacuous(""))
        self.assertTrue(repetition.is_vacuous("  \n "))
        self.assertFalse(repetition.is_vacuous(" x "))

    def test_target_index_is_last_user(self):
        """Test target index is last user."""
        fixture = Fixture(
            id="m",
            group="g",
            messages=(
                Message(role="user", content="first"),
                Message(role="assistant", content="reply"),
                Message(role="user", content="second"),
            ),
            expected="x",
        )
        self.assertEqual(repetition.target_index(fixture), 2)
        self.assertEqual(repetition.target_content(fixture), "second")

    def test_target_index_requires_a_user_message(self):
        """Test target index requires a user message."""
        fixture = Fixture(
            id="m",
            group="g",
            messages=(Message(role="assistant", content="only"),),
            expected="x",
        )
        with self.assertRaises(ValueError):
            repetition.target_index(fixture)

    def test_build_arm_fixture_x2_doubles_last_user_only(self):
        """Test build arm fixture x2 doubles last user only."""
        fixture = Fixture(
            id="m",
            group="g",
            messages=(
                Message(role="user", content="same"),
                Message(role="assistant", content="ok"),
                Message(role="user", content="same"),
            ),
            expected="x",
        )
        arm = repetition.build_arm_fixture(fixture, "x2")
        self.assertEqual(arm.id, "m")
        self.assertEqual(arm.group, "g")
        self.assertEqual(arm.expected, "x")
        self.assertEqual(arm.messages[0].content, "same")
        self.assertEqual(arm.messages[2].content, "same\nsame")
        self.assertEqual(fixture.messages[2].content, "same")

    def test_build_arm_fixture_pad_and_x1(self):
        """Test build arm fixture pad and x1."""
        pad = repetition.build_arm_fixture(CASE_A, "pad", "FILL")
        self.assertTrue(pad.messages[-1].content.endswith("\nFILL"))
        x1 = repetition.build_arm_fixture(CASE_A, "x1")
        self.assertEqual(x1.messages, CASE_A.messages)
        with self.assertRaises(ValueError):
            repetition.build_arm_fixture(CASE_A, "x3")

    def test_is_echo_fixture(self):
        """Test is echo fixture."""
        echo = Fixture(
            id="e",
            group="g",
            messages=(Message(role="user", content="the answer is four, right?"),),
            expected="four",
        )
        self.assertTrue(repetition.is_echo_fixture(echo))
        self.assertFalse(repetition.is_echo_fixture(CASE_A))
        blank = Fixture(
            id="e2",
            group="g",
            messages=(Message(role="user", content="anything"),),
            expected=" ",
        )
        self.assertFalse(repetition.is_echo_fixture(blank))

    def test_filler_text_shape(self):
        """Test filler text shape."""
        unit = repetition.FILLER_UNIT
        self.assertEqual(repetition.filler_text(0, 0), "")
        self.assertEqual(repetition.filler_text(2, 3), unit * 2 + unit[:3])
        self.assertEqual(repetition.filler_text(0, 5), unit[:5])

    def test_filler_search_order_deterministic_capped(self):
        """Test filler search order deterministic capped."""
        first = repetition.filler_search_order(16, 40)
        self.assertEqual(first, repetition.filler_search_order(16, 40))
        self.assertEqual(len(first), len(set(first)))
        self.assertNotIn((0, 0), first)
        self.assertLessEqual(len(repetition.filler_search_order(1, 100_000)), 512)

    def test_screen_filler_both_directions(self):
        """Test screen filler both directions."""
        screened, candidates = repetition.screen_filler(
            {"fx-a": "rain", "fx-b": "blue"}, {"cand-a": "STOP"}
        )
        self.assertEqual(screened, frozenset({"fx-a"}))
        self.assertEqual(candidates, frozenset())
        screened2, candidates2 = repetition.screen_filler(
            {"fx-a": "the lane past the orchard was quiet"}, {"cand-a": "orchard"}
        )
        self.assertEqual(screened2, frozenset({"fx-a"}))
        self.assertEqual(candidates2, frozenset({"cand-a"}))

    def test_content_region(self):
        """Test content region."""
        self.assertEqual(repetition.content_region((1, 2, 3, 4, 5), (1, 2), (5,)), (3, 4))
        self.assertIsNone(repetition.content_region((1, 2, 3), (1, 9), (3,)))
        self.assertIsNone(repetition.content_region((1,), (1, 2), (3,)))

    def test_classify_construction(self):
        """Test classify construction."""
        self.assertEqual(
            repetition.classify_construction((1, 2), (1, 2, 9, 1, 2), (9,)),
            ("id_exact", ()),
        )
        kind, seam = repetition.classify_construction((1, 2), (1, 2, 8, 2), (9,))
        self.assertEqual(kind, "text_level")
        self.assertEqual(seam, (2, 8, 2))
        self.assertEqual(repetition.classify_construction(None, (1,), (9,)), ("text_level", ()))


def _record(
    fixture: str,
    *,
    vacuous: bool = False,
    pre: bool = False,
    x1: bool | None = True,
    x2: bool | None = True,
    pad: bool | None = True,
    pad_evaluable: bool = True,
    delta: int | None = 5,
) -> dict[str, Any]:
    """Record."""

    def cell(v: bool | None) -> dict[str, Any] | None:
        """Cell."""
        if v is None:
            return None
        return {
            "verdict": v,
            "boundary_found": True,
            "boundary_text_match": None,
            "input_tokens": 10,
        }

    return {
        "fixture": fixture,
        "vacuous": vacuous,
        "pre_repeated": pre,
        "echo_fixture": False,
        "construction": "vacuous" if vacuous else "id_exact",
        "seam_ids": [],
        "delta_tokens": delta,
        "filler_tokens": 5 if pad_evaluable else None,
        "pad_evaluable": pad_evaluable,
        "arms": {"x1": cell(x1), "x2": cell(x2), "pad": cell(pad)},
        "arm_details": {"x2": "", "pad": ""},
    }


class TestAggregate(unittest.TestCase):
    """Test aggregate."""

    def test_paired_denominators_per_contrast(self):
        """Test paired denominators per contrast."""
        records = [
            _record("f1", x1=True, x2=True, pad=False),
            _record("f2", x1=False, x2=True, pad=None, pad_evaluable=False),
            _record("f3", x1=True, x2=None, pad=True),
        ]
        agg = repetition.aggregate_candidate(records)
        self.assertEqual(agg["contrasts"]["x2_vs_x1"]["n_paired"], 2)
        self.assertEqual(agg["contrasts"]["x2_vs_x1"]["raw_gain"], 0.5)
        self.assertEqual(agg["contrasts"]["x2_vs_pad"]["n_paired"], 1)
        self.assertEqual(agg["contrasts"]["x2_vs_pad"]["repetition_effect"], 1.0)
        self.assertEqual(agg["contrasts"]["pad_vs_x1"]["n_paired"], 2)
        self.assertTrue(agg["evaluable_mismatch"]["x2_vs_pad"])
        # x1 completed everywhere; x2 missed f3 -> the sets differ.
        self.assertTrue(agg["evaluable_mismatch"]["x2_vs_x1"])
        self.assertEqual(agg["n_pad_evaluable"], 2)
        self.assertAlmostEqual(agg["pad_evaluable_share"], 2 / 3)

    def test_rates_none_when_nothing_paired(self):
        """Test rates none when nothing paired."""
        agg = repetition.aggregate_candidate([_record("f1", x2=None, pad=None)])
        self.assertIsNone(agg["contrasts"]["x2_vs_x1"]["raw_gain"])
        self.assertEqual(agg["contrasts"]["x2_vs_x1"]["n_paired"], 0)

    def test_vacuous_excluded_from_contrasts_but_counted(self):
        """Test vacuous excluded from contrasts but counted."""
        records = [
            _record("f1", x1=True, x2=True, pad=True),
            _record("f2", vacuous=True, x2=None, pad=None, pad_evaluable=False),
        ]
        agg = repetition.aggregate_candidate(records)
        self.assertEqual(agg["n_vacuous"], 1)
        self.assertEqual(agg["contrasts"]["x2_vs_x1"]["n_paired"], 1)

    def test_excluding_pre_repeated_is_a_subpopulation(self):
        """Test excluding pre repeated is a subpopulation."""
        records = [
            _record("f1", x1=False, x2=True, pad=True),
            _record("f2", pre=True, x1=True, x2=False, pad=False),
        ]
        agg = repetition.aggregate_candidate(records)
        self.assertEqual(agg["n_pre_repeated"], 1)
        excl = agg["contrasts_excluding_pre_repeated"]["x2_vs_x1"]
        self.assertEqual(excl["n_paired"], 1)
        self.assertEqual(excl["raw_gain"], 1.0)
        self.assertEqual(agg["contrasts"]["x2_vs_x1"]["raw_gain"], 0.0)

    def test_delta_signs_and_boundary_rates(self):
        """Test delta signs and boundary rates."""
        records = [
            _record("f1", delta=-2),
            _record("f2", delta=0),
            _record("f3", delta=7),
            _record("f4", delta=None),
        ]
        agg = repetition.aggregate_candidate(records)
        self.assertEqual(agg["delta_sign_counts"], {"negative": 1, "zero": 1, "positive": 1})
        self.assertEqual(agg["boundary_found_rates"]["x1"], 1.0)

    def test_arm_all_pass(self):
        """Test arm all pass."""
        self.assertTrue(repetition.arm_all_pass([_record("f1"), _record("f2")], "x2"))
        self.assertFalse(repetition.arm_all_pass([_record("f1"), _record("f2", x2=False)], "x2"))
        self.assertIsNone(repetition.arm_all_pass([_record("f1", x2=None)], "x2"))
        self.assertIsNone(repetition.arm_all_pass([_record("f1", vacuous=True, x2=None)], "x2"))


class TestConstructionOracle(unittest.TestCase):
    """Test construction oracle."""

    def test_x2_prompt_rebuilt_from_arm_fixture_not_replace(self):
        """Test x2 prompt rebuilt from arm fixture not replace."""
        # G1: identical first/last user contents. A replace-based
        # construction would double the FIRST turn; the stage's x2
        # prompt must equal the render of the fixture whose LAST
        # turn is doubled, written out literally here.
        fixture = Fixture(
            id="fx-twin",
            group="g",
            messages=(
                Message(role="user", content="markone ping"),
                Message(role="assistant", content="ok"),
                Message(role="user", content="markone ping"),
            ),
            expected="four",
        )
        backend = _WordBackend({("markone", "x2"): True, ("markone", "pad"): True})
        entry = _entry(backend, FMT_A, {"fx-twin": True}, cases=(fixture,))
        _run_stage(backend, (FMT_A,), (fixture,), [entry])
        expected = FMT_A.render(
            (
                Message(role="user", content="markone ping"),
                Message(role="assistant", content="ok"),
                Message(role="user", content="markone ping\nmarkone ping"),
            )
        )
        self.assertIn(expected, backend.prompts)
        wrongly_doubled_first = FMT_A.render(
            (
                Message(role="user", content="markone ping\nmarkone ping"),
                Message(role="assistant", content="ok"),
                Message(role="user", content="markone ping"),
            )
        )
        self.assertNotIn(wrongly_doubled_first, backend.prompts)

    def test_catalog_wide_construction_measured(self):
        """Test catalog wide construction measured."""
        # Every catalog family's base format runs the stage without
        # error; the construction class is measured per candidate,
        # and the resulting payload passes the diff validator.
        from model_protocol_lab.catalog import CATALOG, candidates

        backend = _WordBackend({("markone", "x2"): True, ("markone", "pad"): False})
        for family in CATALOG:
            fmt = candidates(family)[0]
            with self.subTest(family=family):
                entry = _entry(backend, fmt, {"fx-a": False}, cases=(CASE_A,))
                payload = _payload(backend, (fmt,), (CASE_A,), [entry])
                self.assertTrue(payload["available"])
                record = payload["candidates"][fmt.name]["records"][0]
                self.assertIn(record["construction"], ("id_exact", "text_level"))
                self.assertGreater(record["delta_tokens"], 0)
                self.assertTrue(diffreports._valid_repetition(payload))


class TestStage(unittest.TestCase):
    """Test stage."""

    def test_x1_reused_never_regenerated(self):
        """Test x1 reused never regenerated."""
        # G5: exactly the treated arms generate (2 arms x 3
        # fixtures); the x1 ground records are the recorded native
        # observations verbatim, including a None boundary_text_match.
        backend = _WordBackend(
            {(m, arm): True for m in EXPECTED_BY_MARKER for arm in ("x2", "pad")}
        )
        entry = _entry(backend, FMT_A, {"fx-a": True, "fx-b": False, "fx-c": True}, btm=None)
        payload = _payload(backend, (FMT_A,), CASES, [entry])
        self.assertEqual(backend.generate_calls, 6)
        records = payload["candidates"]["cand-a"]["records"]
        self.assertEqual([r["arms"]["x1"]["verdict"] for r in records], [True, False, True])
        self.assertIsNone(records[0]["arms"]["x1"]["boundary_text_match"])
        self.assertTrue(diffreports._valid_repetition(payload))

    def test_headline_contrast_length_held_constant(self):
        """Test headline contrast length held constant."""
        backend = _WordBackend(
            {
                ("markone", "x2"): True,
                ("marktwo", "x2"): True,
                ("markthree", "x2"): True,
            }
        )
        entry = _entry(backend, FMT_A, {"fx-a": False, "fx-b": True, "fx-c": False})
        payload = _payload(backend, (FMT_A,), CASES, [entry])
        agg = payload["candidates"]["cand-a"]
        self.assertAlmostEqual(agg["contrasts"]["x2_vs_x1"]["raw_gain"], 2 / 3)
        self.assertAlmostEqual(agg["contrasts"]["pad_vs_x1"]["length_effect"], -1 / 3)
        self.assertAlmostEqual(agg["contrasts"]["x2_vs_pad"]["repetition_effect"], 1.0)
        self.assertEqual(agg["pad_evaluable_share"], 1.0)

    def test_per_candidate_boundary_map(self):
        """Test per candidate boundary map."""
        # G10: each candidate is judged under its own boundary; the
        # backend emits only id 998, so cand-b (mapped to 999) finds
        # no boundary in its treated arms.
        backend = _WordBackend(
            {(m, arm): True for m in EXPECTED_BY_MARKER for arm in ("x2", "pad")},
            boundary_id=998,
        )
        entries = [
            _entry(backend, FMT_A, {"fx-a": True, "fx-b": True, "fx-c": True}),
            _entry(backend, FMT_B, {"fx-a": True, "fx-b": True, "fx-c": True}),
        ]
        payload = _payload(
            backend,
            (FMT_A, FMT_B),
            CASES,
            entries,
            boundary_map={"cand-a": (998,), "cand-b": (999,)},
        )
        rec_a = payload["candidates"]["cand-a"]["records"][0]
        rec_b = payload["candidates"]["cand-b"]["records"][0]
        self.assertTrue(rec_a["arms"]["x2"]["boundary_found"])
        self.assertFalse(rec_b["arms"]["x2"]["boundary_found"])
        self.assertFalse(rec_b["arms"]["x2"]["verdict"])

    def test_vacuous_fixture_excluded_and_never_generated(self):
        """Test vacuous fixture excluded and never generated."""
        # G14: whitespace-only content is vacuous by predicate, not
        # by render comparison; it generates nothing and pairs with
        # nothing, but it is counted.
        blank = Fixture(
            id="fx-blank",
            group="g",
            messages=(Message(role="user", content="   "),),
            expected="four",
        )
        backend = _WordBackend({("markone", "x2"): True, ("markone", "pad"): True})
        cases = (CASE_A, blank)
        entry = _entry(backend, FMT_A, {"fx-a": True, "fx-blank": True}, cases=cases)
        payload = _payload(backend, (FMT_A,), cases, [entry])
        self.assertEqual(backend.generate_calls, 2)
        agg = payload["candidates"]["cand-a"]
        self.assertEqual(agg["n_vacuous"], 1)
        record = agg["records"][1]
        self.assertEqual(record["construction"], "vacuous")
        self.assertEqual(record["arm_details"], {"x2": "vacuous", "pad": "vacuous"})
        self.assertEqual(agg["contrasts"]["x2_vs_x1"]["n_paired"], 1)
        self.assertTrue(diffreports._valid_repetition(payload))

    def test_screened_fixture_pad_skips_but_x2_runs(self):
        """Test screened fixture pad skips but x2 runs."""
        # G7: the expected answer "rain" occurs inside the filler
        # unit, so the pad arm would hand the model its answer; the
        # fixture is pad-screened (visible denominator shrink) while
        # x2 still measures.
        case = Fixture(
            id="fx-rain",
            group="g",
            messages=(Message(role="user", content="markrain will it rain today"),),
            expected="rain",
        )
        backend = _WordBackend(
            {("markrain", "x2"): True, ("markrain", "pad"): True},
            answers={"markrain": "rain"},
        )
        entry = _entry(backend, FMT_A, {"fx-rain": False}, cases=(case,))
        payload = _payload(backend, (FMT_A,), (case,), [entry])
        record = payload["candidates"]["cand-a"]["records"][0]
        self.assertEqual(record["arm_details"]["pad"], "pad_screened")
        self.assertFalse(record["pad_evaluable"])
        self.assertIsNotNone(record["arms"]["x2"])
        self.assertEqual(backend.generate_calls, 1)
        agg = payload["candidates"]["cand-a"]
        self.assertEqual(agg["contrasts"]["x2_vs_pad"]["n_paired"], 0)
        self.assertEqual(agg["contrasts"]["x2_vs_x1"]["n_paired"], 1)
        self.assertTrue(diffreports._valid_repetition(payload))

    def test_context_equality_measured_one_over_degrades_per_fixture(self):
        """Test context equality measured one over degrades per fixture."""
        # G15: the budget is the stage's max_context parameter (the
        # fake backend has no context limit of its own). Equality
        # with len(ids) + max_new_tokens is measured; the longer
        # fixture degrades alone, per fixture, both arms.
        long_case = Fixture(
            id="fx-long",
            group="g",
            messages=(
                Message(
                    role="user",
                    content="marklong " + " ".join(f"w{i}" for i in range(60)),
                ),
            ),
            expected="four",
        )
        backend = _WordBackend(
            {("markone", "x2"): True, ("markone", "pad"): True},
            answers={"markone": "four", "marklong": "four"},
        )
        self.assertFalse(hasattr(backend, "_context_limit"))
        cases = (CASE_A, long_case)
        entries = [_entry(backend, FMT_A, {"fx-a": True, "fx-long": True}, cases=cases)]
        ids_x2_short = len(
            backend.encode(FMT_A.render(repetition.build_arm_fixture(CASE_A, "x2").messages))
        )
        payload = _payload(
            backend,
            (FMT_A,),
            cases,
            entries,
            max_new_tokens=8,
            max_context=ids_x2_short + 8,
        )
        records = payload["candidates"]["cand-a"]["records"]
        self.assertIsNotNone(records[0]["arms"]["x2"])
        self.assertEqual(records[1]["arm_details"]["x2"], "arm_context_exceeded")
        self.assertIsNone(records[1]["arms"]["pad"])
        self.assertIn(records[1]["arm_details"]["pad"], ("arm_context_exceeded", "pad_unmatchable"))
        self.assertTrue(diffreports._valid_repetition(payload))

    def test_delta_zero_without_positive_filler_match_is_unmatchable(self):
        """Test delta zero without positive filler match is unmatchable."""
        # A one-word content doubles without growing the word count
        # (delta 0); every exact-length pad is a zero-filler
        # degenerate, so the pair is pad-unmatchable, never shipped
        # as a length control that controls nothing.
        case = Fixture(
            id="fx-one",
            group="g",
            messages=(Message(role="user", content="markone"),),
            expected="four",
        )
        backend = _WordBackend({("markone", "x2"): True})
        entry = _entry(backend, FMT_A, {"fx-one": True}, cases=(case,))
        payload = _payload(backend, (FMT_A,), (case,), [entry])
        record = payload["candidates"]["cand-a"]["records"][0]
        self.assertEqual(record["delta_tokens"], 0)
        self.assertEqual(record["arm_details"]["pad"], "pad_unmatchable")
        self.assertFalse(record["pad_evaluable"])
        self.assertTrue(diffreports._valid_repetition(payload))

    def test_negative_delta_overshoot_is_unmatchable(self):
        """Test negative delta overshoot is unmatchable.

        The recorded x1 is longer than the fresh x2 render (a
        squash tokenizer): the empty pad already overshoots the
        x2 length, so no filler search can match.
        """

        class _SquashBackend(_WordBackend):
            """Squash backend."""

            def encode(self, text: str) -> tuple[int, ...]:
                """Encode."""
                if text.count("markone") >= 2:
                    return (42,)
                return super().encode(text)

        backend = _SquashBackend({("markone", "x2"): True})
        entry = _entry(backend, FMT_A, {"fx-a": True}, cases=(CASE_A,))
        payload = _payload(backend, (FMT_A,), (CASE_A,), [entry])
        record = payload["candidates"]["cand-a"]["records"][0]
        self.assertLess(record["delta_tokens"], 0)
        self.assertEqual(record["arm_details"]["pad"], "pad_unmatchable")
        agg = payload["candidates"]["cand-a"]
        self.assertEqual(agg["delta_sign_counts"]["negative"], 1)
        self.assertTrue(diffreports._valid_repetition(payload))

    def test_candidate_unavailable_withholds_all_arms(self):
        """Test candidate unavailable withholds all arms."""
        backend = _WordBackend({(m, "x2"): True for m in EXPECTED_BY_MARKER})
        entries = [
            _entry(backend, FMT_A, {"fx-a": True, "fx-b": True, "fx-c": True}),
            _entry(backend, FMT_B, {"fx-a": True, "fx-b": True, "fx-c": True}, eligible=False),
        ]
        payload = _payload(backend, (FMT_A, FMT_B), CASES, entries)
        self.assertEqual(payload["candidates"]["cand-b"]["status"], "unavailable")
        self.assertEqual(payload["candidates"]["cand-b"]["detail"], "native_search_unavailable")
        for arm in ("x1", "x2", "pad"):
            self.assertEqual(payload["arms"][arm]["decision"], "unavailable")
            self.assertTrue(payload["arms"][arm]["withheld"])
            self.assertIsNone(payload["arms"][arm]["flip"])
        self.assertTrue(diffreports._valid_repetition(payload))

    def test_arm_failure_withholds_the_flip(self):
        """Test arm failure withholds the flip."""
        # G19's worked example: cand-b's x2 generation fails on one
        # fixture, so the completed sets differ across candidates.
        # A flip computed anyway would compare different populations;
        # the arm readout is withheld instead.
        backend = _WordBackend(
            {(m, arm): True for m in EXPECTED_BY_MARKER for arm in ("x2", "pad")},
            fail_fragments=("U>", "marktwo"),
        )
        entries = [
            _entry(backend, fmt, {"fx-a": True, "fx-b": True, "fx-c": True})
            for fmt in (FMT_A, FMT_B)
        ]
        payload = _payload(backend, (FMT_A, FMT_B), CASES, entries)
        record_b = payload["candidates"]["cand-b"]["records"][1]
        self.assertEqual(record_b["arm_details"]["x2"], "arm_evaluation_failed")
        self.assertTrue(payload["arms"]["x2"]["withheld"])
        self.assertIsNone(payload["arms"]["x2"]["flip"])
        self.assertEqual(payload["arms"]["x2"]["detail"], "selection_withheld_mismatched_sets")
        self.assertTrue(diffreports._valid_repetition(payload))

    def test_cancel_returns_no_partial_aggregates(self):
        """Test cancel returns no partial aggregates."""
        # G18: cancellation mid-stage returns the cancelled payload
        # with empty candidates -- no half-built aggregates.
        backend = _WordBackend(
            {(m, arm): True for m in EXPECTED_BY_MARKER for arm in ("x2", "pad")},
            set_cancel_on=1,
        )
        entry = _entry(backend, FMT_A, {"fx-a": True, "fx-b": True, "fx-c": True})
        event = threading.Event()
        stage = _run_stage(backend, (FMT_A,), CASES, [entry], cancel=event)
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "cancelled")
        self.assertEqual(stage["candidates"], {})
        payload = experiments.repetition_analysis(stage, "no_candidate", None, ("cand-a",))
        self.assertFalse(payload["available"])
        self.assertEqual(payload["detail"], "cancelled")

    def test_cell_budget_fires_before_any_generation(self):
        """Test cell budget fires before any generation."""
        backend = _WordBackend()
        formats = tuple(
            Format(name=f"f{i}", bos="", user_prefix="u:", assistant_prefix="a:", turn_end="</s>")
            for i in range(49)
        )
        cases = tuple(
            Fixture(
                id=f"c{i}",
                group="g",
                messages=(Message(role="user", content=f"q{i}"),),
                expected="x",
            )
            for i in range(50)
        )
        stage = _run_stage(backend, formats, cases, [])
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "repetition_cell_budget_exceeded")
        self.assertEqual(backend.generate_calls, 0)

    def test_filler_screen_failure_fails_closed(self):
        """Test filler screen failure fails closed."""
        # The shipped filler is clean against the shipped catalog,
        # so this guard is proven with a stubbed hostile unit (the
        # cycle-171 rule: a guard is proven with a stub or removed).
        fmt = Format(
            name="cand-x", bos="", user_prefix="u:", assistant_prefix="a:", turn_end="<|im_end|>"
        )
        backend = _WordBackend()
        entry = _entry(backend, fmt, {"fx-a": True}, cases=(CASE_A,))
        with mock.patch.object(repetition, "FILLER_UNIT", "pad <|im_end|> pad "):
            stage = _run_stage(backend, (fmt,), (CASE_A,), [entry])
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "filler_screen_failed")
        self.assertEqual(backend.generate_calls, 0)

    def test_seam_backend_records_text_level_with_seam_ids(self):
        """Test seam backend records text level with seam ids."""
        # G3: a merge-y tokenizer does not block measurement; the
        # construction is recorded text_level with the seam ids.
        case = Fixture(
            id="fx-seam",
            group="g",
            messages=(Message(role="user", content="markone says abcd"),),
            expected="four",
        )
        backend = _SeamBackend({("solo", "x2"): True, ("solo", "pad"): True})
        entry = _entry(backend, FMT_A, {"fx-seam": True}, cases=(case,))
        payload = _payload(backend, (FMT_A,), (case,), [entry], boundary=(7,))
        record = payload["candidates"]["cand-a"]["records"][0]
        self.assertEqual(record["construction"], "text_level")
        self.assertEqual(record["seam_ids"], [1, 109, 97])
        self.assertTrue(record["pad_evaluable"])
        # 17, not 16: the seam digraph also merges the empty pad's
        # trailing joiner, so the measured filler span on rendered
        # ids is one token longer than the raw character count.
        self.assertEqual(record["filler_tokens"], 17)
        self.assertTrue(diffreports._valid_repetition(payload))

    def test_think_groups_and_render_aliasing_disclosed(self):
        """Test think groups and render aliasing disclosed."""
        # G16: one contribution per candidate to its think group;
        # two candidates with identical framing share a render
        # signature and the payload says so.
        backend = _WordBackend(
            {
                ("markone", "x2"): True,
                ("marktwo", "x2"): True,
                ("markthree", "x2"): False,
                ("markone", "pad"): False,
                ("marktwo", "pad"): False,
                ("markthree", "pad"): False,
            }
        )
        clone = Format(
            name="cand-a2", bos="", user_prefix="u:", assistant_prefix="a:", turn_end="</s>"
        )
        formats = (FMT_THINK, FMT_A, clone)
        verdicts = {"fx-a": True, "fx-b": True, "fx-c": True}
        entries = [_entry(backend, fmt, verdicts) for fmt in formats]
        payload = _payload(backend, formats, CASES, entries)
        self.assertEqual(payload["n_candidates"], 3)
        self.assertEqual(payload["n_distinct_renders"], 2)
        think = payload["think_groups"]["think"]
        self.assertEqual(think["n_candidates"], 1)
        self.assertAlmostEqual(think["mean_repetition_effect"], 2 / 3)
        no_think = payload["think_groups"]["no_think"]
        self.assertEqual(no_think["n_candidates"], 2)
        self.assertEqual(no_think["n_distinct_renders"], 1)
        self.assertAlmostEqual(no_think["mean_repetition_effect_dedup"], 2 / 3)
        self.assertTrue(diffreports._valid_repetition(payload))


def _stage_dict(candidates: dict[str, Any]) -> dict[str, Any]:
    """Stage dict."""
    ids = sorted({r["fixture"] for e in candidates.values() for r in e.get("records", [])})
    return {
        "available": True,
        "detail": "",
        "protocol": repetition.protocol_block(ids or ["f1"], "sha256:hand"),
        "candidates": candidates,
    }


def _cand(records: list, *, think: bool = False, signature: str | None = "sig") -> dict[str, Any]:
    """Cand."""
    return {
        "status": "available",
        "detail": "",
        "think_channel": think,
        "render_signature": signature,
        "records": records,
    }


class TestAnalysis(unittest.TestCase):
    """Test analysis."""

    @staticmethod
    def _two_candidate_stage() -> dict[str, Any]:
        """Two candidate stage."""
        # Both pass x1 everywhere; under x2 only cand-a passes all.
        a_records = [_record("f1"), _record("f2")]
        b_records = [_record("f1"), _record("f2", x2=False)]
        return _stage_dict({"cand-a": _cand(a_records), "cand-b": _cand(b_records)})

    def test_mapped_decisions_do_not_flip(self):
        """Test mapped decisions do not flip."""
        # G6: the run vocabulary is mapped before comparison --
        # compatible_on_fixture_set and held_out_failed both mean
        # ("selected", winner); an arm re-selecting the same winner
        # is not a flip under either.
        stage = self._two_candidate_stage()
        for decision in ("compatible_on_fixture_set", "held_out_failed"):
            payload = experiments.repetition_analysis(
                stage, decision, "cand-a", ("cand-a", "cand-b")
            )
            self.assertEqual(payload["arms"]["x2"]["winner"], "cand-a")
            self.assertFalse(payload["arms"]["x2"]["flip"], decision)

    def test_verbatim_decision_can_flip(self):
        """Test verbatim decision can flip."""
        stage = self._two_candidate_stage()
        payload = experiments.repetition_analysis(stage, "no_candidate", None, ("cand-a", "cand-b"))
        self.assertTrue(payload["arms"]["x2"]["flip"])
        payload = experiments.repetition_analysis(
            stage, "compatible_on_fixture_set", "cand-b", ("cand-a", "cand-b")
        )
        self.assertTrue(payload["arms"]["x2"]["flip"])

    def test_x1_arm_never_reports_a_flip(self):
        """Test x1 arm never reports a flip."""
        stage = self._two_candidate_stage()
        payload = experiments.repetition_analysis(stage, "no_candidate", None, ("cand-a", "cand-b"))
        self.assertIsNone(payload["arms"]["x1"]["flip"])
        self.assertEqual(payload["arms"]["x1"]["decision"], "ambiguous")

    def test_unavailable_stage_passes_through(self):
        """Test unavailable stage passes through."""
        payload = experiments.repetition_analysis(
            {"available": False, "detail": "cancelled", "protocol": None},
            "no_candidate",
            None,
            ("cand-a",),
        )
        self.assertEqual(payload, {"available": False, "detail": "cancelled", "protocol": None})

    def test_think_group_dedup_counts_aliased_renders_once(self):
        """Test think group dedup counts aliased renders once."""
        # G16: two think candidates share one render signature;
        # the group reports both the per-candidate mean and the
        # deduplicated mean (first candidate by name wins the key).
        t1 = _cand([_record("f1", x2=True, pad=False)], think=True, signature="same")
        t2 = _cand(
            [_record("f1", x2=True, pad=False), _record("f2", x2=False, pad=False)],
            think=True,
            signature="same",
        )
        stage = _stage_dict({"cand-t1": t1, "cand-t2": t2})
        payload = experiments.repetition_analysis(
            stage, "no_candidate", None, ("cand-t1", "cand-t2")
        )
        group = payload["think_groups"]["think"]
        self.assertEqual(group["n_candidates"], 2)
        self.assertEqual(group["n_distinct_renders"], 1)
        self.assertAlmostEqual(group["mean_repetition_effect"], 0.75)
        self.assertAlmostEqual(group["mean_repetition_effect_dedup"], 1.0)


class TestValidatorAndDiff(unittest.TestCase):
    """Test validator and diff."""

    @staticmethod
    def _valid_payload() -> dict[str, Any]:
        """Valid payload."""
        backend = _WordBackend(
            {
                ("markone", "x2"): True,
                ("marktwo", "x2"): False,
                ("markthree", "x2"): True,
                ("markone", "pad"): False,
                ("marktwo", "pad"): False,
                ("markthree", "pad"): True,
            }
        )
        entries = [
            _entry(backend, FMT_A, {"fx-a": False, "fx-b": True, "fx-c": False}),
            _entry(backend, FMT_B, {"fx-a": True, "fx-b": True, "fx-c": False}),
        ]
        return _payload(backend, (FMT_A, FMT_B), CASES, entries)

    def test_real_payload_validates(self):
        """Test real payload validates."""
        self.assertTrue(diffreports._valid_repetition(self._valid_payload()))

    def test_tamper_battery_rejected(self):
        """Test tamper battery rejected."""
        import copy as _copy

        base = self._valid_payload()
        tampers = []

        def tamper(name, fn):
            """Tamper."""
            payload = _copy.deepcopy(base)
            fn(payload)
            tampers.append((name, payload))

        tamper(
            "record verdict",
            lambda p: p["candidates"]["cand-a"]["records"][0]["arms"]["x2"].__setitem__(
                "verdict", False
            ),
        )
        tamper("n_vacuous", lambda p: p["candidates"]["cand-a"].__setitem__("n_vacuous", 9))
        tamper(
            "raw_gain",
            lambda p: p["candidates"]["cand-a"]["contrasts"]["x2_vs_x1"].__setitem__(
                "raw_gain", 0.125
            ),
        )
        tamper("fingerprint", lambda p: p["protocol"].__setitem__("fingerprint", "0" * 64))
        tamper("fixture hash", lambda p: p["protocol"].__setitem__("fixture_set_sha256", "0" * 64))
        tamper("arm flip", lambda p: p["arms"]["x2"].__setitem__("flip", None))
        tamper(
            "think mean",
            lambda p: p["think_groups"]["no_think"].__setitem__("mean_repetition_effect", 3.5),
        )
        tamper(
            "pad share", lambda p: p["candidates"]["cand-a"].__setitem__("pad_evaluable_share", 0.5)
        )
        tamper(
            "btm type",
            lambda p: p["candidates"]["cand-a"]["records"][0]["arms"]["x1"].__setitem__(
                "boundary_text_match", "yes"
            ),
        )
        tamper(
            "duplicate fixture",
            lambda p: p["candidates"]["cand-a"]["records"].append(
                _copy.deepcopy(p["candidates"]["cand-a"]["records"][0])
            ),
        )
        tamper(
            "status flip", lambda p: p["candidates"]["cand-a"].__setitem__("status", "unavailable")
        )
        tamper(
            "signature type",
            lambda p: p["candidates"]["cand-a"].__setitem__("render_signature", ["x"]),
        )
        for name, payload in tampers:
            with self.subTest(tamper=name):
                self.assertFalse(diffreports._valid_repetition(payload), name)

    def test_coherent_foreign_protocol_version_validates_and_drifts(self):
        """Test coherent foreign protocol version validates and drifts."""
        payload = self._valid_payload()
        table = {
            "version": 99,
            "joiner": payload["protocol"]["joiner"],
            "filler_unit_sha256": payload["protocol"]["filler_unit_sha256"],
            "detector_version": payload["protocol"]["detector_version"],
            "sizing_rule": payload["protocol"]["sizing_rule"],
            "fixture_set_sha256": payload["protocol"]["fixture_set_sha256"],
            "backend_fingerprint": payload["protocol"]["backend_fingerprint"],
        }
        canonical = json.dumps(table, sort_keys=True, separators=(",", ":"))
        payload["protocol"]["version"] = 99
        payload["protocol"]["fingerprint"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        self.assertTrue(diffreports._valid_repetition(payload))
        diff = diffreports.diff_repetition(self._valid_payload(), payload)
        self.assertTrue(diff["protocol_drift"])
        self.assertFalse(diff["comparable"])
        self.assertTrue(diff["changed"])

    def test_diff_identical_payloads_unchanged(self):
        """Test diff identical payloads unchanged."""
        payload = self._valid_payload()
        diff = diffreports.diff_repetition(payload, payload)
        self.assertEqual(diff["available"], "both")
        self.assertFalse(diff["changed"])
        self.assertTrue(diff["comparable"])
        self.assertEqual(diff["effect_drift"], [])

    def test_diff_availability_shapes(self):
        """Test diff availability shapes."""
        payload = self._valid_payload()
        self.assertEqual(diffreports.diff_repetition(payload, None)["available"], "old")
        self.assertEqual(diffreports.diff_repetition(None, payload)["available"], "new")
        self.assertEqual(diffreports.diff_repetition(None, None)["available"], "neither")
        self.assertEqual(
            diffreports.diff_repetition(payload, {"available": False})["available"], "old"
        )

    def test_diff_effect_and_backend_drift(self):
        """Test diff effect and backend drift."""
        backend = _WordBackend({("markone", "x2"): True})
        backend.fingerprint = "sha256:other"
        entries = [_entry(backend, FMT_A, {"fx-a": False, "fx-b": True, "fx-c": False})]
        other = _payload(backend, (FMT_A,), CASES, entries)
        base_backend = _WordBackend({("markone", "x2"): False})
        base_entries = [_entry(base_backend, FMT_A, {"fx-a": False, "fx-b": True, "fx-c": False})]
        base = _payload(base_backend, (FMT_A,), CASES, base_entries)
        diff = diffreports.diff_repetition(base, other)
        self.assertTrue(diff["backend_drift"])
        self.assertFalse(diff["comparable"])
        metrics = {(d["candidate"], d["metric"]) for d in diff["effect_drift"]}
        self.assertIn(("cand-a", "repetition_effect"), metrics)
        self.assertIn(("cand-a", "raw_gain"), metrics)


class TestReporting(unittest.TestCase):
    """Test reporting."""

    @staticmethod
    def _payload() -> dict[str, Any]:
        """Payload."""
        backend = _WordBackend(
            {("markone", "x2"): True, ("marktwo", "x2"): True, ("markthree", "x2"): True}
        )
        entry = _entry(backend, FMT_A, {"fx-a": False, "fx-b": True, "fx-c": False})
        return _payload(backend, (FMT_A,), CASES, [entry])

    def test_section_renders_with_headline_effect(self):
        """Test section renders with headline effect."""
        from model_protocol_lab import reporting

        result = {"decision": "no_candidate", "search": [], "repetition": self._payload()}
        html = reporting.render_html(result)
        self.assertIn('id="repetition"', html)
        self.assertIn("Prompt repetition", html)

    def test_section_absent_without_available_payload(self):
        """Test section absent without available payload."""
        from model_protocol_lab import reporting

        html = reporting.render_html({"decision": "failed", "search": []})
        self.assertNotIn('id="repetition"', html)
        html = reporting.render_html(
            {
                "decision": "failed",
                "search": [],
                "repetition": {"available": False, "detail": "cancelled"},
            }
        )
        self.assertNotIn('id="repetition"', html)

    def test_section_escapes_hostile_candidate_names(self):
        """Test section escapes hostile candidate names."""
        from model_protocol_lab import reporting

        payload = self._payload()
        hostile = payload["candidates"].pop("cand-a")
        payload["candidates"]["<script>alert(1)</script>"] = hostile
        html = reporting.render_html(
            {"decision": "no_candidate", "search": [], "repetition": payload}
        )
        self.assertIn('id="repetition"', html)
        self.assertNotIn("<script>alert(1)</script>", html)

    def test_section_survives_malformed_payload(self):
        """Test section survives malformed payload."""
        from model_protocol_lab import reporting

        for bad in (
            {"available": True, "candidates": "junk"},
            {"available": True, "candidates": {"x": "junk"}},
            {"available": True, "candidates": {"x": {"status": "available", "contrasts": []}}},
            {"available": "yes", "candidates": {}},
        ):
            html = reporting.render_html(
                {"decision": "no_candidate", "search": [], "repetition": bad}
            )
            self.assertIsInstance(html, str)


class TestCliDispatch(unittest.TestCase):
    """Test cli dispatch."""

    @staticmethod
    def _write_fixtures(directory: str) -> Path:
        """Write fixtures."""
        path = Path(directory) / "fixture.json"
        path.write_text(
            json.dumps(
                {
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
            ),
            encoding="utf-8",
        )
        return path

    def _run_with_stage_recorder(self, result_doc: dict, stage_recorder: Any) -> tuple[Any, dict]:
        """Run with stage recorder."""
        from model_protocol_lab import backends
        from model_protocol_lab import doctor as doctor_mod

        captured: dict[str, Any] = {}

        class _FakeHFBackend:
            """Fake hf backend."""

            def __init__(self, model: Any, device: Any = None, max_context: Any = None) -> None:
                self.model = model

            eos_token_id = None

        def fake_experiment(backend: Any, formats: Any, split: Any, **kwargs: Any) -> dict:
            """Fake experiment."""
            captured["experiment_cancel"] = kwargs.get("cancel")
            return result_doc

        def fake_save(result: Any, outdir: Any, **kwargs: Any) -> tuple[str, str]:
            """Fake save."""
            captured["result"] = result
            return (str(Path(outdir) / "results.json"), str(Path(outdir) / "report.md"))

        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._write_fixtures(tmp)
            argv = [
                "run",
                "--model",
                "m",
                "--dataset",
                str(fixture),
                "--family",
                "llama3",
                "--boundary-ids",
                "124",
                "--output",
                str(Path(tmp) / "out"),
                "--repeat",
            ]
            with (
                mock.patch.object(backends, "LocalHFBackend", _FakeHFBackend),
                mock.patch.object(cli_module, "run_experiment", fake_experiment),
                mock.patch.object(cli_module, "save_report", fake_save),
                mock.patch.object(doctor_mod, "template_conformance", return_value={"ok": True}),
                mock.patch.object(experiments, "repetition_stage", stage_recorder),
            ):
                code = cli_module.main(argv)
        return code, captured

    def test_cancel_event_identity_reaches_the_stage(self):
        """Test cancel event identity reaches the stage."""
        # G18(a): the identical Event object the run created is
        # threaded dispatch -> wrapper -> stage.
        seen: dict[str, Any] = {}

        def recorder(backend: Any, formats: Any, cases: Any, search: Any, **kwargs: Any) -> dict:
            """Recorder."""
            seen["stage_cancel"] = kwargs.get("cancel")
            return {"available": False, "detail": "cancelled", "protocol": None, "candidates": {}}

        result_doc = {
            "decision": "no_candidate",
            "selected_candidate": None,
            "search": [{"format": {"name": "llama3"}}],
            "held_out": {},
        }
        code, captured = self._run_with_stage_recorder(result_doc, recorder)
        self.assertEqual(code, 2)  # abstention exit, by contract
        self.assertIsInstance(seen["stage_cancel"], threading.Event)
        self.assertIs(seen["stage_cancel"], captured["experiment_cancel"])
        self.assertEqual(
            captured["result"]["repetition"],
            {"available": False, "detail": "cancelled", "protocol": None},
        )

    def test_failed_native_search_gates_to_zero_generations(self):
        """Test failed native search gates to zero generations."""
        # G20: a failed run decision means the stage is never
        # invoked and the payload is the wrapper's gate detail.
        calls: list = []

        def recorder(*args: Any, **kwargs: Any) -> dict:
            """Recorder."""
            calls.append((args, kwargs))
            return {}

        result_doc = {
            "decision": "failed",
            "selected_candidate": None,
            "search": [{"format": {"name": "llama3"}}],
            "held_out": {},
        }
        _code, captured = self._run_with_stage_recorder(result_doc, recorder)
        self.assertEqual(calls, [])
        self.assertEqual(
            captured["result"]["repetition"],
            {"available": False, "detail": "native_search_incomplete"},
        )


def _exec_repetition_variant(source_text: str) -> Any:
    """Exec a repetition.py variant under the package namespace."""
    with tempfile.TemporaryDirectory(prefix="c183_mut_", dir="/tmp") as tmpdir:
        mod_path = os.path.join(tmpdir, "repetition_variant.py")
        with open(mod_path, "w", encoding="utf-8") as handle:
            handle.write(source_text)
        spec = importlib.util.spec_from_file_location(
            "model_protocol_lab.repetition_variant", mod_path
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules["model_protocol_lab.repetition_variant"] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop("model_protocol_lab.repetition_variant", None)
        return module


class TestRepetitionMutationGates(unittest.TestCase):
    """Tempdir gates over repetition.py (package-namespace exec:
    the module has a relative import, cycle-169/179 pattern).
    """

    def _run_gate(self, name: str, original: str, mutant: str, scenario: Any) -> None:
        """Run gate."""
        source = Path(repetition.__file__).read_text(encoding="utf-8")
        self.assertEqual(source.count(original), 1, f"gate {name}: anchor not unique")
        mutated = source.replace(original, mutant)
        self.assertNotEqual(mutated, source)
        variant = _exec_repetition_variant(mutated)
        scenario(repetition)  # passes on the real module: gate is non-vacuous
        with self.assertRaises(AssertionError, msg=f"gate {name}: mutant survived"):
            scenario(variant)

    def test_gate_joiner_in_id_exact(self):
        """Test gate joiner in id exact."""

        def scenario(mod: Any) -> None:
            """Scenario."""
            self.assertEqual(
                mod.classify_construction((1, 2), (1, 2, 9, 1, 2), (9,)),
                ("id_exact", ()),
            )
            kind, _seam = mod.classify_construction((1, 2), (1, 2, 1, 2), (9,))
            self.assertEqual(kind, "text_level")

        self._run_gate(
            "joiner_dropped",
            "if region_x2 == region_x1 + joiner_region + region_x1:",
            "if region_x2 == region_x1 + region_x1:",
            scenario,
        )

    def test_gate_vacuous_excluded_from_pairing(self):
        """Test gate vacuous excluded from pairing."""

        def scenario(mod: Any) -> None:
            """Scenario."""
            records = [
                _record("f1", x1=True, x2=True, pad=True),
                _record("f2", vacuous=True, x1=True, x2=True, pad=True),
            ]
            agg = mod.aggregate_candidate(records)
            self.assertEqual(agg["contrasts"]["x2_vs_x1"]["n_paired"], 1)

        self._run_gate(
            "vacuous_paired",
            '            if not r["vacuous"]\n            and _arm_verdict(r, arm_a) is not None',
            "            if _arm_verdict(r, arm_a) is not None",
            scenario,
        )

    def test_gate_detector_space_separator(self):
        """Test gate detector space separator."""

        def scenario(mod: Any) -> None:
            """Scenario."""
            self.assertTrue(mod.detect_pre_repeated("ha ha ha ha"))

        self._run_gate(
            "detector_sep",
            '    for sep in ("", " "):',
            '    for sep in ("",):',
            scenario,
        )


class TestValidatorStructuralBattery(unittest.TestCase):
    """One broken structural rule at a time against the validator.

    The tamper battery in TestValidatorAndDiff breaks aggregates;
    this battery breaks the ground-record grammar itself, so every
    early return in the recompute is exercised by a named case.
    """

    @staticmethod
    def _base() -> dict[str, Any]:
        """Base."""
        backend = _WordBackend(
            {("markone", "x2"): True, ("marktwo", "x2"): True, ("markthree", "x2"): True}
        )
        entry = _entry(backend, FMT_A, {"fx-a": True, "fx-b": True, "fx-c": True})
        return _payload(backend, (FMT_A,), CASES, [entry])

    def test_structural_rejects(self):
        """Test structural rejects."""
        import copy as _copy

        base = self._base()
        self.assertTrue(diffreports._valid_repetition(base))

        def rec(payload: dict, index: int = 0) -> dict:
            """Rec."""
            return payload["candidates"]["cand-a"]["records"][index]

        cases: list[tuple[str, Any]] = [
            ("protocol not dict", lambda p: p.__setitem__("protocol", "junk")),
            ("protocol joiner empty", lambda p: p["protocol"].__setitem__("joiner", "")),
            ("protocol version str", lambda p: p["protocol"].__setitem__("version", "1")),
            ("candidates not dict", lambda p: p.__setitem__("candidates", [])),
            ("candidate name not str", lambda p: p.__setitem__("candidates", {7: {}})),
            (
                "status vocab",
                lambda p: p["candidates"]["cand-a"].__setitem__("status", "weird"),
            ),
            (
                "think not bool",
                lambda p: p["candidates"]["cand-a"].__setitem__("think_channel", "yes"),
            ),
            (
                "records not list",
                lambda p: p["candidates"]["cand-a"].__setitem__("records", {}),
            ),
            (
                "record not dict",
                lambda p: p["candidates"]["cand-a"].__setitem__("records", ["junk"]),
            ),
            (
                "vacuous not bool",
                lambda p: rec(p).__setitem__("vacuous", "yes"),
            ),
            (
                "echo not bool",
                lambda p: rec(p).__setitem__("echo_fixture", 1),
            ),
            (
                "construction vocab",
                lambda p: rec(p).__setitem__("construction", "bogus"),
            ),
            (
                "vacuous construction mismatch",
                lambda p: (
                    rec(p).__setitem__("construction", "text_level")
                    or rec(p).__setitem__("vacuous", True)
                ),
            ),
            ("seam not list", lambda p: rec(p).__setitem__("seam_ids", "x")),
            ("seam non-int", lambda p: rec(p).__setitem__("seam_ids", ["x"])),
            ("delta str", lambda p: rec(p).__setitem__("delta_tokens", "5")),
            ("filler zero", lambda p: rec(p).__setitem__("filler_tokens", 0)),
            ("arms not dict", lambda p: rec(p).__setitem__("arms", [])),
            ("details not dict", lambda p: rec(p).__setitem__("arm_details", [])),
            ("arm key missing", lambda p: rec(p)["arms"].pop("pad")),
            ("cell not dict", lambda p: rec(p)["arms"].__setitem__("x2", 7)),
            (
                "verdict not bool",
                lambda p: rec(p)["arms"]["x2"].__setitem__("verdict", "yes"),
            ),
            (
                "boundary_found not bool",
                lambda p: rec(p)["arms"]["x2"].__setitem__("boundary_found", 1),
            ),
            (
                "input_tokens negative",
                lambda p: rec(p)["arms"]["x2"].__setitem__("input_tokens", -1),
            ),
            ("x1 missing cell", lambda p: rec(p)["arms"].__setitem__("x1", None)),
            (
                "detail vocab",
                lambda p: rec(p)["arm_details"].__setitem__("x2", "bogus"),
            ),
            (
                "cell present but detail set",
                lambda p: rec(p)["arm_details"].__setitem__("x2", "pad_screened"),
            ),
            (
                "pad evaluable but filler none",
                lambda p: rec(p).__setitem__("filler_tokens", None),
            ),
            (
                "pad cell without evaluable",
                lambda p: rec(p).__setitem__("pad_evaluable", False),
            ),
            (
                "vacuous record with treated cells",
                lambda p: (
                    rec(p).__setitem__("vacuous", True)
                    or rec(p).__setitem__("construction", "vacuous")
                ),
            ),
            (
                "missing cell with empty detail",
                lambda p: (
                    rec(p)["arms"].__setitem__("x2", None)
                    or rec(p)["arm_details"].__setitem__("x2", "")
                ),
            ),
            (
                "primary decision type",
                lambda p: p.__setitem__("primary_decision", 7),
            ),
            ("n_candidates tamper", lambda p: p.__setitem__("n_candidates", 5)),
            ("n_distinct tamper", lambda p: p.__setitem__("n_distinct_renders", 5)),
        ]
        for name, fn in cases:
            with self.subTest(case=name):
                payload = _copy.deepcopy(base)
                fn(payload)
                self.assertFalse(diffreports._valid_repetition(payload), name)

    def test_fixture_set_mismatch_across_candidates_rejected(self):
        """Test fixture set mismatch across candidates rejected."""
        backend = _WordBackend({("markone", "x2"): True})
        entries = [
            _entry(backend, FMT_A, {"fx-a": True, "fx-b": True, "fx-c": True}),
            _entry(backend, FMT_B, {"fx-a": True, "fx-b": True, "fx-c": True}),
        ]
        payload = _payload(backend, (FMT_A, FMT_B), CASES, entries)
        payload["candidates"]["cand-b"]["records"][0]["fixture"] = "fx-other"
        self.assertFalse(diffreports._valid_repetition(payload))

    def test_mapped_primary_payload_validates(self):
        """Test mapped primary payload validates."""
        # The validator's mapped branch: a run-vocabulary primary
        # decision is compared as ("selected", winner).
        backend = _WordBackend(
            {("markone", "x2"): True, ("marktwo", "x2"): True, ("markthree", "x2"): True}
        )
        entry = _entry(backend, FMT_A, {"fx-a": True, "fx-b": True, "fx-c": True})
        stage = _run_stage(backend, (FMT_A,), CASES, [entry])
        payload = experiments.repetition_analysis(
            stage, "compatible_on_fixture_set", "cand-a", ("cand-a",)
        )
        self.assertTrue(diffreports._valid_repetition(payload))
        self.assertFalse(payload["arms"]["x2"]["flip"])

    def test_diff_pad_share_drift_listed(self):
        """Test diff pad share drift listed."""
        backend = _WordBackend({("markone", "x2"): True})
        entry = _entry(backend, FMT_A, {"fx-a": True, "fx-b": True, "fx-c": True})
        full = _payload(backend, (FMT_A,), CASES, [entry])
        rain = Fixture(
            id="fx-rain",
            group="g",
            messages=(Message(role="user", content="markrain will it rain today"),),
            expected="rain",
        )
        backend2 = _WordBackend(
            {("markone", "x2"): True, ("markrain", "x2"): True},
            answers={"markone": "four", "markrain": "rain"},
        )
        cases = (CASE_A, CASE_B, rain)
        entry2 = _entry(backend2, FMT_A, {"fx-a": True, "fx-b": True, "fx-rain": True}, cases=cases)
        screened = _payload(backend2, (FMT_A,), cases, [entry2])
        diff = diffreports.diff_repetition(full, screened)
        metrics = {(d["candidate"], d["metric"]) for d in diff["effect_drift"]}
        self.assertIn(("cand-a", "pad_evaluable_share"), metrics)


class TestStageGapCoverage(unittest.TestCase):
    """Test stage gap coverage."""

    def test_stage_contract_value_errors(self):
        """Test stage contract value errors."""
        backend = _WordBackend()
        with self.assertRaises(ValueError):
            experiments.repetition_stage(
                backend, [FMT_A], CASES, [], boundary=(999,), max_context=100
            )
        with self.assertRaises(ValueError):
            experiments.repetition_stage(
                backend, (FMT_A,), [CASE_A], [], boundary=(999,), max_context=100
            )

    def test_search_entry_junk_and_missing_fixture(self):
        """Test search entry junk and missing fixture."""
        backend = _WordBackend({("markone", "x2"): True})
        good = _entry(backend, FMT_A, {"fx-a": True, "fx-b": True, "fx-c": True})
        stage = _run_stage(backend, (FMT_A,), CASES, ["junk", good])
        self.assertTrue(stage["available"])
        short = _entry(backend, FMT_A, {"fx-a": True, "fx-b": True, "fx-c": True})
        short["observations"] = short["observations"][:2]
        stage = _run_stage(backend, (FMT_A,), CASES, [short])
        self.assertEqual(stage["candidates"]["cand-a"]["detail"], "native_search_unavailable")
        malformed = _entry(backend, FMT_A, {"fx-a": True, "fx-b": True, "fx-c": True})
        malformed["observations"] = ["junk"]
        stage = _run_stage(backend, (FMT_A,), CASES, [malformed])
        self.assertEqual(stage["candidates"]["cand-a"]["detail"], "native_search_unavailable")

    def test_undecodable_boundary_screens_nothing(self):
        """Test undecodable boundary screens nothing."""

        class _BoundaryDecodeBackend(_WordBackend):
            """Boundary decode backend."""

            def decode(self, ids: Any) -> str:
                """Decode."""
                if tuple(ids) == (999,):
                    raise ValueError("undecodable")
                return super().decode(ids)

        backend = _BoundaryDecodeBackend({("markone", "x2"): True})
        entry = _entry(backend, FMT_A, {"fx-a": True, "fx-b": True, "fx-c": True})
        payload = _payload(backend, (FMT_A,), CASES, [entry])
        self.assertTrue(payload["available"])
        self.assertEqual(payload["candidates"]["cand-a"]["pad_evaluable_share"], 1.0)

    def test_cancel_already_set_at_stage_entry(self):
        """Test cancel already set at stage entry."""
        backend = _WordBackend()
        entry = _entry(backend, FMT_A, {"fx-a": True, "fx-b": True, "fx-c": True})
        event = threading.Event()
        event.set()
        stage = _run_stage(backend, (FMT_A,), CASES, [entry], cancel=event)
        self.assertEqual(stage["detail"], "cancelled")
        self.assertEqual(backend.generate_calls, 0)

    def test_non_list_search_and_nameless_entries(self):
        """Test non list search and nameless entries."""
        backend = _WordBackend({("markone", "x2"): True})
        entry = _entry(backend, FMT_A, {"fx-a": True, "fx-b": True, "fx-c": True})
        stage = _run_stage(backend, (FMT_A,), CASES, (entry,))
        self.assertEqual(stage["candidates"]["cand-a"]["detail"], "native_search_unavailable")
        stage = _run_stage(backend, (FMT_A,), CASES, [{"format": {"name": 7}}])
        self.assertEqual(stage["candidates"]["cand-a"]["detail"], "native_search_unavailable")

    def test_marker_ambiguous_render_skips_affixes(self):
        """Test marker ambiguous render skips affixes."""
        # A format whose own framing contains the marker makes the
        # marker render ambiguous (count != 1): affixes are skipped
        # and the construction is measured text_level, not guessed.
        fmt = Format(
            name="cand-marker",
            bos="",
            user_prefix="u:\x00mplrep\x00",
            assistant_prefix="a:",
            turn_end="</s>",
        )
        backend = _WordBackend({("markone", "x2"): True, ("markone", "pad"): True})
        entry = _entry(backend, fmt, {"fx-a": True}, cases=(CASE_A,))
        payload = _payload(backend, (fmt,), (CASE_A,), [entry])
        record = payload["candidates"]["cand-marker"]["records"][0]
        self.assertEqual(record["construction"], "text_level")
        self.assertTrue(diffreports._valid_repetition(payload))

    def test_marker_in_content_skips_affixes(self):
        """Test marker in content skips affixes."""
        case = Fixture(
            id="fx-marker",
            group="g",
            messages=(Message(role="user", content="markone \x00mplrep\x00 tail"),),
            expected="four",
        )
        backend = _WordBackend({("markone", "x2"): True, ("markone", "pad"): True})
        entry = _entry(backend, FMT_A, {"fx-marker": True}, cases=(case,))
        payload = _payload(backend, (FMT_A,), (case,), [entry])
        record = payload["candidates"]["cand-a"]["records"][0]
        self.assertEqual(record["construction"], "text_level")
        self.assertIsNotNone(record["delta_tokens"])
        self.assertTrue(diffreports._valid_repetition(payload))


class TestReportingDefensivePaths(unittest.TestCase):
    """Test reporting defensive paths."""

    def test_withheld_unavailable_and_junk_entries_render(self):
        """Test withheld unavailable and junk entries render."""
        from model_protocol_lab import reporting

        payload = {
            "available": True,
            "candidates": {
                "cand-a": {"status": "unavailable", "detail": "native_search_unavailable"},
                "junk": "not-a-dict",
            },
            "arms": {"x2": {"withheld": True}, "pad": "junk"},
            "think_groups": {},
        }
        html = reporting.render_html(
            {"decision": "no_candidate", "search": [], "repetition": payload}
        )
        self.assertIn('id="repetition"', html)
        self.assertIn("Selection comparison withheld for x2", html)
        self.assertIn("native search unavailable", html)

    def test_no_arms_payload_renders_without_arm_list(self):
        """Test no arms payload renders without arm list."""
        from model_protocol_lab import reporting

        backend = _WordBackend({("markone", "x2"): True})
        entry = _entry(backend, FMT_A, {"fx-a": True, "fx-b": True, "fx-c": True})
        payload = _payload(backend, (FMT_A,), CASES, [entry])
        del payload["arms"]
        html = reporting.render_html(
            {"decision": "no_candidate", "search": [], "repetition": payload}
        )
        self.assertIn('id="repetition"', html)
        self.assertNotIn("selection flip", html)
