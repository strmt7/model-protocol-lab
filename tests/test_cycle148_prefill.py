"""Cycle 148: assistant-prefill sensitivity experiment (wave-2 idea 2).

Does the winning protocol survive when the caller pre-fills the assistant's
opening tokens -- and which template families honor vs. ignore prefills?
Benign format-lock prefills are appended (as token ids) after each family's
generation prompt; the model generates the continuation and the scored span
is prefill + generated text. Honoring is detected exactly: a model that did
not accept the prefill as its own opening restarts its turn and duplicates
the prefill text.

Covers: prefill-spec validation, fixture construction, the honoring
predicate, the think-channel predicate (pinned per catalog family),
``PrefillBackend``, ``prefill_stage`` (scripted backend, fail-closed
cells), ``prefill_adherence`` (selection re-application, flips,
think-effect, malformed-input degradation), ``diff_prefill``
(availability, every changed disjunct pinned singly, hostile payloads),
the stderr summary lines, the HTML section (escaping), four tempdir-module
mutation gates, and a brute-force oracle differential for the pure
analysis.
"""

from __future__ import annotations

import importlib.util
import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from model_protocol_lab import core
from model_protocol_lab.catalog import CATALOG
from model_protocol_lab.core import Format, Message
from model_protocol_lab.diffreports import diff_prefill, diff_reports, render_summary
from model_protocol_lab.experiments import prefill_adherence, prefill_stage
from model_protocol_lab.prefill import (
    PREFILL_SPECS,
    PrefillBackend,
    build_prefill_fixtures,
    generation_tail,
    honors_prefill,
    opens_think_channel,
    validate_prefill_specs,
)
from model_protocol_lab.reporting import _prefill_section

CANDIDATES = ("alpha", "beta")
BOUNDARY = (7,)


def _format(name: str, **kwargs) -> Format:
    return Format(
        name=name,
        bos="",
        user_prefix=f"[{name}]",
        assistant_prefix="A:",
        turn_end="\n",
        **kwargs,
    )


def _case(fixture_id: str, expected: str) -> core.Fixture:
    return core.Fixture(
        id=fixture_id,
        messages=(Message("user", f"say expect:{expected}"),),
        expected=expected,
        group="g",
    )


class ScriptedPrefillBackend:
    """Deterministic backend with per-candidate prefill policies.

    ``policy`` maps candidate name to "honor" (emit the expected answer),
    "restart" (re-emit the prefill text, then the answer -- the model
    restarted its turn), "wrong" (emit an unrelated string), or "raise"
    (``generate`` raises, exercising the fail-closed cell path). The
    expected answer is parsed from the ``expect:`` marker in the prompt.
    """

    fingerprint = "cycle148-prefill-scripted"

    def __init__(self, policy, prefill_text="Answer: "):
        self._policy = dict(policy)
        self._prefill_text = prefill_text

    @staticmethod
    def encode(text):
        return tuple(map(ord, text))

    @staticmethod
    def decode(ids):
        return "".join(map(chr, ids))

    def generate(self, input_ids, *, max_new_tokens, seed, cancel):
        prompt = self.decode(input_ids)
        candidate = next((n for n in CANDIDATES if f"[{n}]" in prompt), None)
        if candidate is None:
            return core.Generation((), "error", "unknown")
        mode = self._policy.get(candidate, "honor")
        if mode == "raise":
            raise RuntimeError("generate boom")
        needle = "expect:"
        idx = prompt.rfind(needle)
        expected = prompt[idx + len(needle) :].split()[0] if idx >= 0 else "?"
        if mode == "honor":
            text = expected
        elif mode == "restart":
            text = self._prefill_text + expected
        else:
            text = "WRONG"
        ids = self.encode(text + chr(7))
        return core.Generation(ids + (1,) * (max_new_tokens - len(ids)), "limit")


def _run_stage(policy, prefill_id="answer-colon", cases=None, backend=None):
    spec = next(s for s in validate_prefill_specs(PREFILL_SPECS) if s["id"] == prefill_id)
    return prefill_stage(
        ScriptedPrefillBackend(policy, spec["text"]) if backend is None else backend,
        tuple(_format(c) for c in CANDIDATES),
        cases if cases is not None else (_case("f1", "GREEN"), _case("f2", "BLUE")),
        prefill=spec,
        boundary=BOUNDARY,
        max_new_tokens=16,
        seed=0,
    )


def _prefill_payload(**overrides):
    payload = {
        "available": True,
        "primary_winner": "alpha",
        "primary_decision": "selected",
        "n_fixtures": 2,
        "prefills": ["answer-colon", "final-answer"],
        "per_condition": {
            "answer-colon": {
                "decision": "selected",
                "winner": "alpha",
                "n_evaluable": 2,
                "per_candidate": {
                    "alpha": {
                        "honor_rate": 1.0,
                        "restart_rate": 0.0,
                        "think_channel": False,
                        "pass": True,
                    },
                    "beta": {
                        "honor_rate": 0.0,
                        "restart_rate": 1.0,
                        "think_channel": True,
                        "pass": False,
                    },
                },
            },
            "final-answer": {
                "decision": "no_candidate",
                "winner": None,
                "n_evaluable": 2,
                "per_candidate": {
                    "alpha": {
                        "honor_rate": 0.5,
                        "restart_rate": 0.5,
                        "think_channel": False,
                        "pass": False,
                    },
                    "beta": {
                        "honor_rate": 0.0,
                        "restart_rate": 1.0,
                        "think_channel": True,
                        "pass": False,
                    },
                },
            },
        },
        "flips": [
            {
                "prefill": "final-answer",
                "decision_from": "selected",
                "decision_to": "no_candidate",
                "winner_from": "alpha",
                "winner_to": None,
            }
        ],
        "think_effect": {
            "think_channel": {"mean_honor_rate": 0.0, "n_candidates": 1},
            "plain": {"mean_honor_rate": 0.75, "n_candidates": 1},
            "delta": -0.75,
        },
    }
    payload.update(overrides)
    return payload


class PrefillSpecValidationTests(unittest.TestCase):
    def test_valid_specs_round_trip(self):
        specs = validate_prefill_specs(PREFILL_SPECS)
        self.assertEqual([s["id"] for s in specs], ["answer-colon", "final-answer", "result-colon"])
        # Returned copies: mutating them must not affect the module constant.
        specs[0]["text"] = "MUT"
        self.assertEqual(PREFILL_SPECS[0]["text"], "Answer: ")

    def test_rejects_non_tuple(self):
        with self.assertRaises(ValueError):
            validate_prefill_specs(list(PREFILL_SPECS))
        with self.assertRaises(ValueError):
            validate_prefill_specs(())

    def test_rejects_bad_entries(self):
        with self.assertRaises(ValueError):
            validate_prefill_specs(({"id": "x"},))
        with self.assertRaises(ValueError):
            validate_prefill_specs(({"id": "x", "text": "  "},))
        with self.assertRaises(ValueError):
            validate_prefill_specs(({"id": "", "text": "Answer: "},))
        with self.assertRaises(ValueError):
            validate_prefill_specs(("nope",))

    def test_rejects_duplicate_ids(self):
        with self.assertRaises(ValueError):
            validate_prefill_specs(({"id": "x", "text": "A: "}, {"id": "x", "text": "B: "}))

    def test_rejects_non_ascii(self):
        with self.assertRaises(ValueError):
            validate_prefill_specs(({"id": "x", "text": "Antwort: \u00e4"},))


class HonorsPrefillTests(unittest.TestCase):
    def test_honored_continuation(self):
        self.assertTrue(honors_prefill("42", "Answer: "))
        self.assertTrue(honors_prefill("  42", "Answer: "))
        self.assertTrue(honors_prefill("Sure, the answer is 42", "Answer: "))

    def test_restarted_continuation(self):
        self.assertFalse(honors_prefill("Answer: 42", "Answer: "))
        self.assertFalse(honors_prefill("  Answer: 42", "Answer: "))
        # The colon guards against near-miss prefixes.
        self.assertTrue(honors_prefill("Answering your question: 42", "Answer: "))

    def test_empty_generation_has_no_duplication(self):
        self.assertTrue(honors_prefill("", "Answer: "))

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            honors_prefill(42, "Answer: ")
        with self.assertRaises(ValueError):
            honors_prefill("42", "   ")
        with self.assertRaises(ValueError):
            honors_prefill("42", None)


class ThinkChannelTests(unittest.TestCase):
    # Pinned per catalog family (catalog.py line references are from the
    # cycle-148 tree): the predicate is computed from each family's own
    # generation-prompt tail, so thinking on/off variants are labeled
    # independently.
    THINK_FAMILIES = {
        "deepseek41",  # open_suffix <Assistant><think>
        "glm53",  # open_suffix <|assistant|><think>
        "hy4",  # open_suffix ...<think:opensource>
        "k2horizon",  # open_suffix <|ifm|im_start|>assistant\n<ifm|think>\n
        "kimi3",  # open_suffix <|open|>message role="assistant"<|sep|><|open|>think<|sep|>
        "lfm25",  # open_suffix <|im_start|>assistant\n<think>\n
        "ling",  # open_suffix <role>ASSISTANT</role>\n<think>
        "nanbeige42",  # open_suffix <|im_start|>assistant\n<think>\n
        "nemotron35",  # open_suffix <|im_start|>assistant\n<think>\n
        "qwen38",  # open_suffix <|im_start|>assistant\n<think>\n
    }

    def test_all_catalog_families_pinned(self):
        for name, fmt in CATALOG.items():
            with self.subTest(family=name):
                self.assertEqual(
                    opens_think_channel(fmt),
                    name in self.THINK_FAMILIES,
                    f"family {name!r} tail={generation_tail(fmt)!r}",
                )

    def test_thinking_off_variants_are_plain(self):
        from model_protocol_lab.catalog import candidates as catalog_candidates

        for name in (
            "qwen38/thinking-off",
            "nanbeige42/thinking-off",
            "kolibri/thinking-off",
            "nemotron35/thinking-off",
            "ling/thinking-off",
            "minimax3/thinking-disabled",
            "hy4/no-think",
        ):
            family = name.split("/")[0]
            variants = {fmt.name: fmt for fmt in catalog_candidates(family, mutate=True)}
            self.assertIn(name, variants, f"variant {name} missing from catalog")
            self.assertFalse(opens_think_channel(variants[name]), f"{name} should be plain")

    def test_thinking_enabled_variant_is_think(self):
        from model_protocol_lab.catalog import candidates as catalog_candidates

        variants = {fmt.name: fmt for fmt in catalog_candidates("minimax3", mutate=True)}
        self.assertTrue(opens_think_channel(variants["minimax3/thinking-enabled"]))

    def test_closer_only_tail_is_plain(self):
        # laguna's tail closes a think block: no open channel.
        self.assertFalse(opens_think_channel(CATALOG["laguna"]))

    def test_rejects_non_format(self):
        with self.assertRaises(ValueError):
            opens_think_channel("qwen38")
        with self.assertRaises(ValueError):
            generation_tail(None)


class BuildPrefillFixturesTests(unittest.TestCase):
    def test_fixture_ids_and_groups(self):
        spec = {"id": "answer-colon", "text": "Answer: "}
        fixtures = build_prefill_fixtures((_case("f1", "GREEN"),), prefill=spec)
        self.assertEqual(fixtures[0].id, "prefill/answer-colon/f1")
        self.assertEqual(fixtures[0].group, "prefill-answer-colon")
        # Messages and expected are unchanged: the pipeline's exact_match
        # on the plain expected answer already grades the prefill run.
        self.assertEqual(fixtures[0].expected, "GREEN")
        self.assertEqual(fixtures[0].messages[0].content, "say expect:GREEN")

    def test_rejects_bad_input(self):
        spec = {"id": "answer-colon", "text": "Answer: "}
        with self.assertRaises(ValueError):
            build_prefill_fixtures((), prefill=spec)
        with self.assertRaises(ValueError):
            build_prefill_fixtures((_case("f1", "GREEN"),), prefill={"id": "x"})
        with self.assertRaises(ValueError):
            build_prefill_fixtures(["not-a-fixture"], prefill=spec)
        with self.assertRaises(ValueError):
            build_prefill_fixtures((_case("f1", "GREEN"), "not-a-fixture"), prefill=spec)


class PrefillBackendTests(unittest.TestCase):
    def test_encode_appends_prefill_ids(self):
        inner = ScriptedPrefillBackend({})
        wrapped = PrefillBackend(inner, (1, 2, 3), "answer-colon")
        self.assertEqual(wrapped.encode("hi"), inner.encode("hi") + (1, 2, 3))
        self.assertEqual(wrapped.prefill_ids, (1, 2, 3))

    def test_decode_and_generate_forward(self):
        inner = ScriptedPrefillBackend({})
        wrapped = PrefillBackend(inner, (9,), "x")
        self.assertEqual(wrapped.decode((65,)), "A")
        gen = wrapped.generate((1,), max_new_tokens=4, seed=0, cancel=None)
        self.assertIsInstance(gen, core.Generation)

    def test_fingerprint_marks_prefill(self):
        inner = ScriptedPrefillBackend({})
        wrapped = PrefillBackend(inner, (9,), "answer-colon")
        self.assertIn("+prefill:answer-colon", wrapped.fingerprint)
        self.assertTrue(wrapped.fingerprint.startswith(inner.fingerprint))

    def test_rejects_bad_construction(self):
        inner = ScriptedPrefillBackend({})
        with self.assertRaises(ValueError):
            PrefillBackend(None, (1,), "x")
        with self.assertRaises(ValueError):
            PrefillBackend(inner, (), "x")
        with self.assertRaises(ValueError):
            PrefillBackend(inner, (1, -2), "x")
        with self.assertRaises(ValueError):
            PrefillBackend(inner, (1,), "  ")

    def test_rejects_inner_without_string_fingerprint(self):
        class BadFingerprint:
            fingerprint = 42

            def generate(self, input_ids, *, max_new_tokens, seed, cancel):
                raise AssertionError("must not be called")

        with self.assertRaises(ValueError):
            PrefillBackend(BadFingerprint(), (1,), "x")


class PrefillStageTests(unittest.TestCase):
    def test_honor_restart_counts(self):
        stage = _run_stage({"alpha": "honor", "beta": "restart"})
        alpha = stage["per_candidate"]["alpha"]
        beta = stage["per_candidate"]["beta"]
        self.assertEqual((alpha["honored"], alpha["restarted"]), (2, 0))
        self.assertTrue(alpha["pass"])
        self.assertEqual((beta["honored"], beta["restarted"]), (0, 2))
        self.assertFalse(beta["pass"])
        self.assertEqual(stage["prefill_id"], "answer-colon")
        self.assertEqual(stage["prefill_text"], "Answer: ")
        self.assertEqual(stage["n_fixtures"], 2)
        self.assertFalse(alpha["think_channel"])

    def test_generate_failure_degrades_cell(self):
        stage = _run_stage({"alpha": "raise", "beta": "honor"})
        self.assertIsNone(stage["per_candidate"]["alpha"])
        self.assertIsNotNone(stage["per_candidate"]["beta"])

    def test_wrong_answer_honored_but_fails(self):
        # Honoring and passing are separate metrics: a model can honor
        # the prefill yet fail the fixture.
        stage = _run_stage({"alpha": "wrong", "beta": "honor"})
        alpha = stage["per_candidate"]["alpha"]
        self.assertEqual((alpha["honored"], alpha["restarted"]), (2, 0))
        self.assertFalse(alpha["pass"])

    def test_boundary_inside_prefill_fails_closed(self):
        evil_spec = {"id": "pipe", "text": "A|B "}
        stage = prefill_stage(
            ScriptedPrefillBackend({"alpha": "honor", "beta": "honor"}, "A|B "),
            tuple(_format(c) for c in CANDIDATES),
            (_case("f1", "GREEN"),),
            prefill=evil_spec,
            boundary=(ord("|"),),
            max_new_tokens=16,
            seed=0,
        )
        self.assertIsNone(stage["per_candidate"]["alpha"])
        self.assertIsNone(stage["per_candidate"]["beta"])

    def test_prefill_encode_failure_raises(self):
        class BadEncode(ScriptedPrefillBackend):
            @staticmethod
            def encode(text):
                raise RuntimeError("boom")

        with self.assertRaises(ValueError):
            prefill_stage(
                BadEncode({}),
                tuple(_format(c) for c in CANDIDATES),
                (_case("f1", "GREEN"),),
                prefill={"id": "answer-colon", "text": "Answer: "},
                boundary=BOUNDARY,
            )

    def test_prefill_decode_mismatch_raises(self):
        class BadDecode(ScriptedPrefillBackend):
            @staticmethod
            def decode(ids):
                return "MISMATCH"

        with self.assertRaises(ValueError):
            prefill_stage(
                BadDecode({}),
                tuple(_format(c) for c in CANDIDATES),
                (_case("f1", "GREEN"),),
                prefill={"id": "answer-colon", "text": "Answer: "},
                boundary=BOUNDARY,
            )

    def test_rejects_bad_inputs(self):
        backend = ScriptedPrefillBackend({})
        fmts = tuple(_format(c) for c in CANDIDATES)
        cases = (_case("f1", "GREEN"),)
        spec = {"id": "answer-colon", "text": "Answer: "}
        with self.assertRaises(ValueError):
            prefill_stage(None, fmts, cases, prefill=spec, boundary=BOUNDARY)
        with self.assertRaises(ValueError):
            prefill_stage(backend, (), cases, prefill=spec, boundary=BOUNDARY)
        with self.assertRaises(ValueError):
            prefill_stage(backend, fmts, (), prefill=spec, boundary=BOUNDARY)
        with self.assertRaises(ValueError):
            prefill_stage(backend, fmts, cases, prefill={"id": "x"}, boundary=BOUNDARY)
        with self.assertRaises(ValueError):
            prefill_stage(backend, fmts, cases, prefill=spec, boundary=())
        with self.assertRaises(ValueError):
            prefill_stage(backend, fmts, cases, prefill=spec, boundary=BOUNDARY, max_new_tokens=0)
        with self.assertRaises(ValueError):
            prefill_stage(backend, fmts, cases, prefill=spec, boundary=BOUNDARY, seed=-1)


class PrefillAdherenceTests(unittest.TestCase):
    def _stages(self):
        return {
            "answer-colon": _run_stage({"alpha": "honor", "beta": "restart"}),
            "final-answer": _run_stage(
                {"alpha": "honor", "beta": "honor"}, prefill_id="final-answer"
            ),
        }

    def test_selection_reapplied_per_prefill(self):
        result = prefill_adherence("alpha", "selected", CANDIDATES, self._stages())
        self.assertTrue(result["available"])
        self.assertEqual(result["prefills"], ["answer-colon", "final-answer"])
        cond = result["per_condition"]["answer-colon"]
        self.assertEqual((cond["decision"], cond["winner"]), ("selected", "alpha"))
        self.assertEqual(cond["per_candidate"]["alpha"]["honor_rate"], 1.0)
        self.assertEqual(cond["per_candidate"]["beta"]["restart_rate"], 1.0)

    def test_flips_against_primary(self):
        result = prefill_adherence("beta", "selected", CANDIDATES, self._stages())
        flips = result["flips"]
        self.assertEqual(len(flips), 2)
        by_prefill = {f["prefill"]: f for f in flips}
        self.assertEqual(by_prefill["answer-colon"]["winner_to"], "alpha")

    def test_unavailable_condition_skipped_in_flips(self):
        stages = self._stages()
        stages["answer-colon"]["per_candidate"] = {
            "alpha": None,
            "beta": None,
        }
        result = prefill_adherence("alpha", "selected", CANDIDATES, stages)
        self.assertEqual(result["per_condition"]["answer-colon"]["decision"], "unavailable")
        # The unavailable condition contributes no flip; the other
        # condition (ambiguous vs selected/alpha) still flips.
        self.assertEqual([f["prefill"] for f in result["flips"]], ["final-answer"])

    def test_ambiguous_and_no_candidate(self):
        stages = {
            "answer-colon": _run_stage({"alpha": "honor", "beta": "honor"}),
        }
        result = prefill_adherence("alpha", "selected", CANDIDATES, stages)
        cond = result["per_condition"]["answer-colon"]
        self.assertEqual(cond["decision"], "ambiguous")
        self.assertIsNone(cond["winner"])

    def test_think_effect_grouping(self):
        fmts = (
            _format("alpha"),
            _format("beta", open_suffix="<|im_start|>assistant\n<think>\n"),
        )
        spec = {"id": "answer-colon", "text": "Answer: "}
        stage = prefill_stage(
            ScriptedPrefillBackend({"alpha": "honor", "beta": "restart"}),
            fmts,
            (_case("f1", "GREEN"),),
            prefill=spec,
            boundary=BOUNDARY,
            max_new_tokens=16,
            seed=0,
        )
        result = prefill_adherence("alpha", "selected", ("alpha", "beta"), {"answer-colon": stage})
        effect = result["think_effect"]
        self.assertEqual(effect["think_channel"]["mean_honor_rate"], 0.0)
        self.assertEqual(effect["plain"]["mean_honor_rate"], 1.0)
        self.assertEqual(effect["delta"], -1.0)

    def test_malformed_inputs_degrade(self):
        good = self._stages()
        self.assertEqual(
            prefill_adherence("alpha", "selected", (), good)["detail"], "invalid_candidates"
        )
        self.assertEqual(
            prefill_adherence("alpha", "selected", ("alpha", "alpha"), good)["detail"],
            "invalid_candidates",
        )
        self.assertEqual(
            prefill_adherence("zzz", "selected", CANDIDATES, good)["detail"],
            "primary_winner_unknown",
        )
        self.assertEqual(
            prefill_adherence("alpha", "", CANDIDATES, good)["detail"],
            "primary_decision_invalid",
        )
        self.assertEqual(
            prefill_adherence("alpha", "selected", CANDIDATES, {})["detail"], "stages_invalid"
        )
        self.assertEqual(
            prefill_adherence("alpha", "selected", CANDIDATES, {"x": None})["detail"],
            "stage_x_invalid",
        )
        bad = self._stages()
        bad["answer-colon"]["per_candidate"]["alpha"]["honored"] = -1
        self.assertEqual(
            prefill_adherence("alpha", "selected", CANDIDATES, bad)["detail"],
            "stage_answer-colon_invalid",
        )
        bad2 = self._stages()
        bad2["final-answer"]["n_fixtures"] = 99
        for cell in bad2["final-answer"]["per_candidate"].values():
            cell["n"] = 99
        self.assertEqual(
            prefill_adherence("alpha", "selected", CANDIDATES, bad2)["detail"],
            "fixture_count_mismatch",
        )
        bad3 = self._stages()
        bad3["answer-colon"]["prefill_id"] = "other"
        self.assertEqual(
            prefill_adherence("alpha", "selected", CANDIDATES, bad3)["detail"],
            "stage_answer-colon_invalid",
        )


class DiffPrefillTests(unittest.TestCase):
    def test_identical_payloads_unchanged(self):
        diff = diff_prefill(_prefill_payload(), _prefill_payload())
        self.assertEqual(diff["available"], "both")
        self.assertFalse(diff["changed"])

    def test_nondict_flip_entries_skipped(self):
        # _valid_prefill accepts any list for flips; the key builder
        # skips non-dict entries instead of crashing.
        old = _prefill_payload()
        new = _prefill_payload()
        new["flips"] = ["junk", 42, None] + list(new["flips"])
        diff = diff_prefill(old, new)
        self.assertEqual(diff["available"], "both")
        self.assertFalse(diff["flips_changed"])
        self.assertFalse(diff["changed"])

    def test_condition_triplet_changed(self):
        new = _prefill_payload()
        new["per_condition"]["answer-colon"]["winner"] = "beta"
        diff = diff_prefill(_prefill_payload(), new)
        self.assertTrue(diff["changed"])
        cond = diff["condition_changed"]["answer-colon"]
        self.assertTrue(cond["changed"])
        self.assertEqual(cond["new_winner"], "beta")

    def test_single_cause_prefills_added(self):
        new = _prefill_payload()
        new["prefills"] = ["answer-colon", "final-answer", "result-colon"]
        new["per_condition"]["result-colon"] = {
            "decision": "selected",
            "winner": "alpha",
            "n_evaluable": 2,
            "per_candidate": {},
        }
        diff = diff_prefill(_prefill_payload(), new)
        self.assertTrue(diff["changed"])
        self.assertEqual(diff["prefills_added"], ["result-colon"])
        self.assertFalse(diff["prefills_removed"])
        self.assertFalse(diff["flips_changed"])
        self.assertFalse(diff["candidate_drift"])
        self.assertFalse(diff["think_rates"]["changed"])

    def test_single_cause_prefills_removed(self):
        new = _prefill_payload()
        new["prefills"] = ["answer-colon"]
        del new["per_condition"]["final-answer"]
        diff = diff_prefill(_prefill_payload(), new)
        self.assertTrue(diff["changed"])
        self.assertEqual(diff["prefills_removed"], ["final-answer"])

    def test_single_cause_flips_only(self):
        new = _prefill_payload()
        new["flips"] = []
        diff = diff_prefill(_prefill_payload(), new)
        self.assertTrue(diff["changed"])
        self.assertTrue(diff["flips_changed"])
        self.assertFalse(diff["candidate_drift"])
        self.assertFalse(diff["think_rates"]["changed"])
        self.assertFalse(any(c["changed"] for c in diff["condition_changed"].values()))

    def test_single_cause_candidate_drift_only(self):
        new = _prefill_payload()
        new["per_condition"]["answer-colon"]["per_candidate"]["alpha"]["honor_rate"] = 0.5
        diff = diff_prefill(_prefill_payload(), new)
        self.assertTrue(diff["changed"])
        self.assertEqual(len(diff["candidate_drift"]), 1)
        entry = diff["candidate_drift"][0]
        self.assertEqual((entry["candidate"], entry["prefill"]), ("alpha", "answer-colon"))
        self.assertFalse(diff["flips_changed"])
        self.assertFalse(diff["think_rates"]["changed"])

    def test_single_cause_candidates_added_only(self):
        new = _prefill_payload()
        new["per_condition"]["answer-colon"]["per_candidate"]["gamma"] = {
            "honor_rate": 1.0,
            "restart_rate": 0.0,
            "think_channel": False,
            "pass": True,
        }
        diff = diff_prefill(_prefill_payload(), new)
        self.assertTrue(diff["changed"])
        self.assertEqual(diff["candidates_added"], ["gamma"])

    def test_single_cause_candidates_removed_only(self):
        new = _prefill_payload()
        for condition in new["per_condition"].values():
            del condition["per_candidate"]["beta"]
        diff = diff_prefill(_prefill_payload(), new)
        self.assertTrue(diff["changed"])
        self.assertEqual(diff["candidates_removed"], ["beta"])

    def test_single_cause_think_rates_only(self):
        new = _prefill_payload()
        new["think_effect"]["think_channel"]["mean_honor_rate"] = 0.9
        diff = diff_prefill(_prefill_payload(), new)
        self.assertTrue(diff["changed"])
        self.assertTrue(diff["think_rates"]["changed"])
        self.assertFalse(diff["flips_changed"])
        self.assertFalse(diff["candidate_drift"])

    def test_single_cause_condition_changed_only(self):
        new = _prefill_payload()
        new["per_condition"]["answer-colon"]["decision"] = "ambiguous"
        new["per_condition"]["answer-colon"]["winner"] = None
        diff = diff_prefill(_prefill_payload(), new)
        self.assertTrue(diff["changed"])
        self.assertTrue(diff["condition_changed"]["answer-colon"]["changed"])
        self.assertFalse(diff["flips_changed"])

    def test_availability(self):
        self.assertEqual(diff_prefill(_prefill_payload(), {"available": False})["available"], "old")
        self.assertEqual(diff_prefill({"available": False}, _prefill_payload())["available"], "new")
        self.assertEqual(diff_prefill(None, None)["available"], "neither")

    def test_hostile_payloads_invalid(self):
        # Non-string map keys must fail validation (wave-2 C3 lesson).
        evil = _prefill_payload()
        evil["per_condition"]["answer-colon"]["per_candidate"][7] = {"honor_rate": 1.0}
        self.assertEqual(diff_prefill(evil, _prefill_payload())["available"], "new")
        # Non-string decision must fail validation.
        evil2 = _prefill_payload()
        evil2["per_condition"]["answer-colon"]["decision"] = 42
        self.assertEqual(diff_prefill(evil2, _prefill_payload())["available"], "new")
        # Bad rate type must fail validation.
        evil3 = _prefill_payload()
        evil3["per_condition"]["answer-colon"]["per_candidate"]["alpha"]["honor_rate"] = "high"
        self.assertEqual(diff_prefill(evil3, _prefill_payload())["available"], "new")
        # Non-bool think_channel must fail validation.
        evil4 = _prefill_payload()
        evil4["per_condition"]["answer-colon"]["per_candidate"]["alpha"]["think_channel"] = "yes"
        self.assertEqual(diff_prefill(evil4, _prefill_payload())["available"], "new")


class PrefillDiffIntegrationTests(unittest.TestCase):
    def _reports(self, prefill_old, prefill_new):
        base = {
            "decision": "select_best",
            "selected_candidate": "alpha",
            "search": [],
            "held_out": {},
        }
        old = dict(base, prefill=prefill_old)
        new = dict(base, prefill=prefill_new)
        return old, new

    def test_summary_flags_and_identical(self):
        old, new = self._reports(_prefill_payload(), _prefill_payload())
        diff = diff_reports(old, new)
        self.assertFalse(diff["summary"]["prefill_changed"])
        self.assertFalse(diff["summary"]["prefill_asymmetric"])
        self.assertTrue(diff["summary"]["identical"])

    def test_changed_feeds_identical(self):
        changed = _prefill_payload()
        changed["per_condition"]["answer-colon"]["winner"] = "beta"
        old, new = self._reports(_prefill_payload(), changed)
        diff = diff_reports(old, new)
        self.assertTrue(diff["summary"]["prefill_changed"])
        self.assertFalse(diff["summary"]["identical"])

    def test_asymmetric_feeds_identical(self):
        old, new = self._reports(_prefill_payload(), {"available": False})
        diff = diff_reports(old, new)
        self.assertTrue(diff["summary"]["prefill_asymmetric"])
        self.assertFalse(diff["summary"]["identical"])

    def test_stderr_summary_lines(self):
        changed = _prefill_payload()
        changed["per_condition"]["answer-colon"]["winner"] = "beta"
        old, new = self._reports(_prefill_payload(), changed)
        summary = render_summary(diff_reports(old, new))
        self.assertIn("Assistant prefill (answer-colon)", summary)
        self.assertIn("selected/alpha -> selected/beta", summary)

    def test_identical_axes_names_prefill(self):
        old, new = self._reports(_prefill_payload(), _prefill_payload())
        summary = render_summary(diff_reports(old, new))
        self.assertIn("assistant-prefill sensitivity", summary)


class PrefillSectionTests(unittest.TestCase):
    def _result(self):
        return {"prefill": _prefill_payload()}

    def test_renders_tables(self):
        section_html = _prefill_section(self._result())
        self.assertIn("Assistant-prefill sensitivity", section_html)
        self.assertIn("answer-colon", section_html)
        self.assertIn("Per-candidate honoring", section_html)
        self.assertIn("think_channel", section_html)

    def test_empty_when_absent(self):
        self.assertEqual(_prefill_section({}), "")
        self.assertEqual(_prefill_section({"prefill": {"available": False}}), "")

    def test_hostile_non_string_keys_do_not_crash(self):
        result = self._result()
        prefill = result["prefill"]
        prefill["per_condition"][7] = {
            "decision": "selected",
            "winner": "alpha",
            "per_candidate": {8: {"honor_rate": 1.0, "pass": True}},
        }
        section_html = _prefill_section(result)
        self.assertIn("Assistant-prefill sensitivity", section_html)
        self.assertIn("7", section_html)

    def test_escapes_hostile_candidate_names(self):
        import html as html_module

        result = self._result()
        evil = "<script>alert(1)</script>\x1b[31m"
        pc = result["prefill"]["per_condition"]["answer-colon"]["per_candidate"]
        pc[evil] = {
            "honor_rate": 1.0,
            "restart_rate": 0.0,
            "think_channel": False,
            "pass": True,
        }
        section_html = _prefill_section(result)
        self.assertNotIn("<script>", section_html)
        self.assertIn(html_module.escape(evil), section_html)


class PrefillMutationGateTests(unittest.TestCase):
    """Tempdir-module mutation gates (the real tree is never rewritten)."""

    def _run_gate(self, module_name, name, old_text, new_text, scenario):
        mod = __import__(f"model_protocol_lab.{module_name}", fromlist=["x"])
        source = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertEqual(source.count(old_text), 1, f"mutant {name}: anchor not unique")
        mutated = source.replace(old_text, new_text)
        with tempfile.TemporaryDirectory(prefix="c148_mut_", dir="/tmp") as tmpdir:
            mod_path = os.path.join(tmpdir, f"{module_name}_mut_{name}.py")
            with open(mod_path, "w", encoding="utf-8") as handle:
                handle.write(mutated)
            dotted = f"model_protocol_lab.{module_name}_mut_{name}"
            spec = importlib.util.spec_from_file_location(dotted, mod_path)
            assert spec is not None and spec.loader is not None
            mutant = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mutant)
            scenario(mod)  # passes on the real module: non-vacuous
            with self.assertRaises(AssertionError, msg=f"mutant {name} was NOT caught"):
                scenario(mutant)

    def _flip_scenario(self, mod):
        stages = {
            "answer-colon": {
                "prefill_id": "answer-colon",
                "prefill_text": "Answer: ",
                "n_fixtures": 2,
                "per_candidate": {
                    "alpha": {
                        "n": 2,
                        "n_completed": 2,
                        "honored": 2,
                        "restarted": 0,
                        "think_channel": False,
                        "pass": True,
                    },
                    "beta": {
                        "n": 2,
                        "n_completed": 2,
                        "honored": 0,
                        "restarted": 2,
                        "think_channel": False,
                        "pass": False,
                    },
                },
            }
        }
        result = mod.prefill_adherence("alpha", "ambiguous", CANDIDATES, stages)
        # The condition verdict (selected/alpha) differs from the primary
        # (ambiguous/None): exactly one flip must be recorded.
        assert len(result["flips"]) == 1, f"expected 1 flip, got {result['flips']}"

    def test_mutant_inverted_flip_detection_caught(self):
        self._run_gate(
            "experiments",
            "inverted_flip",
            """    for prefill_id in sorted(per_condition):
        verdict = per_condition[prefill_id]
        if verdict["decision"] == "unavailable":
            continue
        if (verdict["decision"], verdict["winner"]) != (primary_decision, primary_winner):""",  # noqa: E501
            """    for prefill_id in sorted(per_condition):
        verdict = per_condition[prefill_id]
        if verdict["decision"] == "unavailable":
            continue
        if (verdict["decision"], verdict["winner"]) == (primary_decision, primary_winner):  # MUTANT""",  # noqa: E501
            self._flip_scenario,
        )

    def _honor_scenario(self, mod):
        stage = mod.prefill_stage(
            ScriptedPrefillBackend({"alpha": "honor", "beta": "restart"}),
            tuple(_format(c) for c in CANDIDATES),
            (_case("f1", "GREEN"),),
            prefill={"id": "answer-colon", "text": "Answer: "},
            boundary=BOUNDARY,
            max_new_tokens=16,
            seed=0,
        )
        alpha = stage["per_candidate"]["alpha"]
        beta = stage["per_candidate"]["beta"]
        assert (alpha["honored"], alpha["restarted"]) == (1, 0), alpha
        assert (beta["honored"], beta["restarted"]) == (0, 1), beta

    def test_mutant_swapped_honor_branches_caught(self):
        self._run_gate(
            "experiments",
            "swapped_honor",
            """            if honors_prefill(obs.answer, prefill_text):
                honored += 1
            else:
                restarted += 1""",
            """            if honors_prefill(obs.answer, prefill_text):
                restarted += 1  # MUTANT
            else:
                honored += 1  # MUTANT""",
            self._honor_scenario,
        )

    def _diff_scenario(self, mod):
        old = _prefill_payload()
        new = _prefill_payload()
        new["flips"] = []
        diff = mod.diff_prefill(old, new)
        assert diff["changed"] is True, "flips-only change must set changed"
        assert diff["flips_changed"] is True

    def test_mutant_dropped_flips_disjunct_caught(self):
        self._run_gate(
            "diffreports",
            "dropped_flips_disjunct",
            '    old_flip_key = _flip_key(old.get("flips"))\n'
            '    new_flip_key = _flip_key(new.get("flips"))\n'
            "    flips_changed = old_flip_key != new_flip_key\n"
            "\n"
            "    def _candidate_rates(payload: Any) -> dict[str, dict[str, tuple[Any, Any]]]:\n"
            "        rates: dict[str, dict[str, tuple[Any, Any]]] = {}\n"
            '        per_condition = payload.get("per_condition") or {}\n'
            "        for prefill_id in sorted(per_condition):",
            '    old_flip_key = _flip_key(old.get("flips"))\n'
            '    new_flip_key = _flip_key(new.get("flips"))\n'
            "    flips_changed = old_flip_key == new_flip_key  # MUTANT\n"
            "\n"
            "    def _candidate_rates(payload: Any) -> dict[str, dict[str, tuple[Any, Any]]]:\n"
            "        rates: dict[str, dict[str, tuple[Any, Any]]] = {}\n"
            '        per_condition = payload.get("per_condition") or {}\n'
            "        for prefill_id in sorted(per_condition):",
            self._diff_scenario,
        )

    def _think_scenario(self, mod):
        old = _prefill_payload()
        new = _prefill_payload()
        new["think_effect"]["think_channel"]["mean_honor_rate"] = 0.9
        diff = mod.diff_prefill(old, new)
        assert diff["changed"] is True, "think-rate-only change must set changed"
        assert diff["think_rates"]["changed"] is True

    def test_mutant_dropped_think_disjunct_caught(self):
        self._run_gate(
            "diffreports",
            "dropped_think_disjunct",
            "think_changed = old_think != new_think",
            "think_changed = old_think == new_think  # MUTANT",
            self._think_scenario,
        )


def _oracle_prefill_adherence(primary_winner, primary_decision, candidates, stages):
    """Independent brute-force reimplementation of prefill_adherence.

    Written against the documented contract only: re-applies the
    conservative selection rule per prefill, diffs (decision, winner)
    against the primary, and averages honoring rates per
    think-channel group. Deliberately different control flow from the
    implementation so shared-logic slips surface as mismatches.
    """
    per_condition = {}
    for prefill_id in sorted(stages.keys()):
        stage = stages[prefill_id]
        cells = stage["per_candidate"]
        passing = [c for c in candidates if cells.get(c) is not None and cells[c]["pass"]]
        evaluable = [c for c in candidates if cells.get(c) is not None]
        if not evaluable:
            verdict = ("unavailable", None)
        elif len(passing) == 1:
            verdict = ("selected", passing[0])
        elif passing:
            verdict = ("ambiguous", None)
        else:
            verdict = ("no_candidate", None)
        per_candidate = {}
        for cand in candidates:
            cell = cells.get(cand)
            if cell is None:
                per_candidate[cand] = None
                continue
            done = cell["n_completed"]
            per_candidate[cand] = {
                "honor_rate": cell["honored"] / done if done else None,
                "restart_rate": cell["restarted"] / done if done else None,
                "think_channel": cell["think_channel"],
                "pass": cell["pass"],
            }
        per_condition[prefill_id] = {
            "decision": verdict[0],
            "winner": verdict[1],
            "n_evaluable": len(evaluable),
            "per_candidate": per_candidate,
        }
    flips = []
    for prefill_id in sorted(per_condition.keys()):
        cond = per_condition[prefill_id]
        if cond["decision"] == "unavailable":
            continue
        if (cond["decision"], cond["winner"]) != (primary_decision, primary_winner):
            flips.append(
                {
                    "prefill": prefill_id,
                    "decision_from": primary_decision,
                    "decision_to": cond["decision"],
                    "winner_from": primary_winner,
                    "winner_to": cond["winner"],
                }
            )
    think_effect = {}
    for label, wanted in (("think_channel", True), ("plain", False)):
        rates = []
        seen = set()
        for cond in per_condition.values():
            for cand, rc in cond["per_candidate"].items():
                if rc is not None and rc["think_channel"] is wanted:
                    seen.add(cand)
                    if rc["honor_rate"] is not None:
                        rates.append(rc["honor_rate"])
        think_effect[label] = {
            "mean_honor_rate": sum(rates) / len(rates) if rates else None,
            "n_candidates": len(seen),
        }
    means = [
        think_effect["think_channel"]["mean_honor_rate"],
        think_effect["plain"]["mean_honor_rate"],
    ]
    think_effect["delta"] = (
        means[0] - means[1] if all(type(m) in (int, float) for m in means) else None
    )
    return {
        "available": True,
        "primary_winner": primary_winner,
        "primary_decision": primary_decision,
        "n_fixtures": next(iter(stages.values()))["n_fixtures"],
        "prefills": sorted(stages.keys()),
        "per_condition": per_condition,
        "flips": flips,
        "think_effect": think_effect,
    }


class PrefillOracleTests(unittest.TestCase):
    def _random_stages(self, rng):
        prefill_ids = rng.sample(
            ["answer-colon", "final-answer", "result-colon"], rng.randint(1, 3)
        )
        stages = {}
        for pid in prefill_ids:
            per_candidate = {}
            for cand in CANDIDATES:
                roll = rng.random()
                if roll < 0.15:
                    per_candidate[cand] = None
                else:
                    n = rng.randint(1, 4)
                    completed = rng.randint(0, n)
                    honored = rng.randint(0, completed)
                    per_candidate[cand] = {
                        "n": n,
                        "n_completed": completed,
                        "honored": honored,
                        "restarted": completed - honored,
                        "think_channel": rng.random() < 0.4,
                        "pass": rng.random() < 0.5,
                    }
            stages[pid] = {
                "prefill_id": pid,
                "prefill_text": pid + " ",
                "n_fixtures": 4,
                "per_candidate": per_candidate,
            }
            # Keep n_fixtures consistent: every cell uses n=4.
            for cell in per_candidate.values():
                if cell is not None:
                    cell["n"] = 4
                    cell["n_completed"] = min(cell["n_completed"], 4)
                    cell["honored"] = min(cell["honored"], cell["n_completed"])
                    cell["restarted"] = cell["n_completed"] - cell["honored"]
        return stages

    def test_oracle_agrees_on_random_inputs(self):
        rng = random.Random(20261018)  # noqa: S311 -- deterministic test seed
        for trial in range(200):
            stages = self._random_stages(rng)
            primary_winner = rng.choice([None, *CANDIDATES])
            primary_decision = rng.choice(["selected", "ambiguous", "no_candidate"])
            if primary_winner is not None:
                primary_decision = "selected"
            got = prefill_adherence(primary_winner, primary_decision, CANDIDATES, stages)
            want = _oracle_prefill_adherence(primary_winner, primary_decision, CANDIDATES, stages)
            self.assertTrue(got["available"])
            self.assertEqual(got, want, f"oracle mismatch on trial {trial}")


class PrefillBranchCoverageTests(unittest.TestCase):
    def test_stage_records_think_channel_per_candidate(self):
        fmts = (
            _format("alpha"),
            _format("beta", open_suffix="<think>\n"),
        )
        stage = prefill_stage(
            ScriptedPrefillBackend({"alpha": "honor", "beta": "honor"}),
            fmts,
            (_case("f1", "GREEN"),),
            prefill={"id": "answer-colon", "text": "Answer: "},
            boundary=BOUNDARY,
            max_new_tokens=16,
            seed=0,
        )
        self.assertFalse(stage["per_candidate"]["alpha"]["think_channel"])
        self.assertTrue(stage["per_candidate"]["beta"]["think_channel"])

    def test_ineligible_report_degrades_cell(self):
        class Ineligible(ScriptedPrefillBackend):
            def generate(self, input_ids, *, max_new_tokens, seed, cancel):
                # Empty generation with reason "limit": _valid_generation
                # rejects it (limit requires exactly cap tokens), so the
                # observation fails, the report is not aggregate-eligible,
                # and the cell degrades to None -- fail-closed.
                return core.Generation((), "limit")

        stage = prefill_stage(
            Ineligible({"alpha": "honor"}),
            (_format("alpha"),),
            (_case("f1", "GREEN"),),
            prefill={"id": "answer-colon", "text": "Answer: "},
            boundary=BOUNDARY,
            max_new_tokens=16,
            seed=0,
        )
        self.assertIsNone(stage["per_candidate"]["alpha"])

    def test_adherence_zero_completed_rates_none(self):
        stages = {
            "answer-colon": {
                "prefill_id": "answer-colon",
                "prefill_text": "Answer: ",
                "n_fixtures": 1,
                "per_candidate": {
                    "alpha": {
                        "n": 1,
                        "n_completed": 0,
                        "honored": 0,
                        "restarted": 0,
                        "think_channel": False,
                        "pass": False,
                    },
                },
            }
        }
        result = prefill_adherence("alpha", "selected", ("alpha",), stages)
        rates = result["per_condition"]["answer-colon"]["per_candidate"]["alpha"]
        self.assertIsNone(rates["honor_rate"])
        self.assertIsNone(rates["restart_rate"])

    def test_adherence_empty_think_group_means_none(self):
        stages = {
            "answer-colon": {
                "prefill_id": "answer-colon",
                "prefill_text": "Answer: ",
                "n_fixtures": 1,
                "per_candidate": {
                    "alpha": {
                        "n": 1,
                        "n_completed": 1,
                        "honored": 1,
                        "restarted": 0,
                        "think_channel": False,
                        "pass": True,
                    },
                },
            }
        }
        result = prefill_adherence("alpha", "selected", ("alpha",), stages)
        effect = result["think_effect"]
        self.assertIsNone(effect["think_channel"]["mean_honor_rate"])
        self.assertIsNone(effect["delta"])

    def test_entry_validation_rejects_inconsistent_counts(self):
        stages = {
            "answer-colon": {
                "prefill_id": "answer-colon",
                "prefill_text": "Answer: ",
                "n_fixtures": 2,
                "per_candidate": {
                    "alpha": {
                        "n": 2,
                        "n_completed": 2,
                        "honored": 2,
                        "restarted": 1,  # honored + restarted != n_completed
                        "think_channel": False,
                        "pass": True,
                    },
                },
            }
        }
        result = prefill_adherence("alpha", "selected", ("alpha",), stages)
        self.assertFalse(result["available"])


class PrefillCliProbeTests(unittest.TestCase):
    def _result(self):
        return {"decision": "selected", "selected_candidate": "alpha"}

    def _call(self, result, backend, cases):
        from model_protocol_lab.cli import _run_prefill_probe

        _run_prefill_probe(
            backend,
            tuple(_format(c) for c in CANDIDATES),
            result,
            cases,
            boundary=BOUNDARY,
            boundary_map=None,
            max_new_tokens=16,
            seed=0,
        )

    def test_probe_helper_happy_path(self):
        result = self._result()
        self._call(
            result,
            ScriptedPrefillBackend({"alpha": "honor", "beta": "honor"}),
            (_case("f1", "GREEN"),),
        )
        prefill = result["prefill"]
        self.assertTrue(prefill["available"])
        self.assertEqual(prefill["prefills"], ["answer-colon", "final-answer", "result-colon"])
        # Both candidates honor and pass every prefill: each condition is
        # ambiguous, which differs from the primary (selected/alpha).
        self.assertEqual(len(prefill["flips"]), 3)
        flip = prefill["flips"][0]
        self.assertEqual(
            (flip["prefill"], flip["decision_from"], flip["decision_to"]),
            ("answer-colon", "selected", "ambiguous"),
        )

    def test_probe_helper_fail_closed(self):
        result = self._result()
        with mock.patch(
            "model_protocol_lab.prefill.validate_prefill_specs",
            side_effect=RuntimeError("boom"),
        ):
            self._call(result, ScriptedPrefillBackend({}), (_case("f1", "GREEN"),))
        self.assertEqual(
            result["prefill"],
            {"available": False, "detail": "prefill_unavailable"},
        )


if __name__ == "__main__":
    unittest.main()
