"""Cycle 154: position sensitivity (research wave-3 idea 1).

Tests the pure core (position.py): seeded option layouts, fixture
construction, primacy-index arithmetic; the experiments.position_stage
fail-closed contract; experiments.position_sensitivity post-hoc
analysis; the CLI probe wiring; the HTML section escaping; and the
diffreports.diff_position section. Follows the established per-cycle
pattern: brute-force oracle differentials, Hypothesis properties,
tempdir-module mutation gates, hostile-payload red-team.
"""

from __future__ import annotations

import copy
import importlib
import importlib.util
import json
import os
import random
import re
import tempfile
import unittest
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from model_protocol_lab import core
from model_protocol_lab.core import Format
from model_protocol_lab.diffreports import diff_position, diff_reports, render_summary
from model_protocol_lab.experiments import position_sensitivity, position_stage
from model_protocol_lab.position import (
    CONDITIONS,
    POSITION_SPECS,
    build_position_fixtures,
    layout_options,
    primacy_index,
    render_options,
    validate_position_specs,
)
from model_protocol_lab.reporting import _position_section

CANDIDATES = ("alpha", "beta")


def _spec(options=("w", "x", "y", "z"), correct="y", spec_id="s1"):
    return {
        "id": spec_id,
        "question": "Which one?",
        "options": options,
        "correct": correct,
    }


class ValidatePositionSpecsTests(unittest.TestCase):
    def test_valid_specs_pass(self):
        specs = validate_position_specs(POSITION_SPECS)
        self.assertEqual(len(specs), 8)
        self.assertIs(type(specs[0]["options"]), tuple)

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            validate_position_specs(())
        with self.assertRaises(ValueError):
            validate_position_specs([])

    def test_non_dict_rejected(self):
        with self.assertRaises(ValueError):
            validate_position_specs((POSITION_SPECS[0], "junk"))

    def test_blank_id_rejected(self):
        bad = dict(POSITION_SPECS[0])
        bad["id"] = "  "
        with self.assertRaises(ValueError):
            validate_position_specs((bad,))

    def test_blank_question_rejected(self):
        bad = dict(POSITION_SPECS[0])
        bad["question"] = ""
        with self.assertRaises(ValueError):
            validate_position_specs((bad,))

    def test_non_string_option_rejected(self):
        bad = dict(POSITION_SPECS[0])
        bad["options"] = ("a", 7, "c", "d")
        with self.assertRaises(ValueError):
            validate_position_specs((bad,))

    def test_blank_option_rejected(self):
        bad = dict(POSITION_SPECS[0])
        bad["options"] = ("a", " ", "c", "d")
        with self.assertRaises(ValueError):
            validate_position_specs((bad,))

    def test_duplicate_options_rejected(self):
        bad = dict(POSITION_SPECS[0])
        bad["options"] = ("a", "b", "a", "c")
        with self.assertRaises(ValueError):
            validate_position_specs((bad,))

    def test_single_option_rejected(self):
        bad = dict(POSITION_SPECS[0])
        bad["options"] = ("a",)
        with self.assertRaises(ValueError):
            validate_position_specs((bad,))

    def test_five_options_rejected(self):
        bad = dict(POSITION_SPECS[0])
        bad["options"] = ("a", "b", "c", "d", "e")
        with self.assertRaises(ValueError):
            validate_position_specs((bad,))

    def test_correct_not_in_options_rejected(self):
        bad = dict(POSITION_SPECS[0])
        bad["correct"] = "nope"
        with self.assertRaises(ValueError):
            validate_position_specs((bad,))

    def test_duplicate_id_rejected(self):
        with self.assertRaises(ValueError):
            validate_position_specs((POSITION_SPECS[0], POSITION_SPECS[0]))

    def test_two_and_three_options_accepted(self):
        specs = validate_position_specs(
            (
                _spec(options=("a", "b"), correct="a", spec_id="two"),
                _spec(options=("a", "b", "c"), correct="b", spec_id="three"),
            )
        )
        self.assertEqual(len(specs), 2)


class LayoutOptionsTests(unittest.TestCase):
    def test_correct_first_places_at_zero(self):
        layout = layout_options(("a", "b", "c", "d"), "c", condition="correct_first", seed=0)
        self.assertEqual(layout[0], "c")
        self.assertEqual(sorted(layout), ["a", "b", "c", "d"])

    def test_correct_last_places_at_end(self):
        layout = layout_options(("a", "b", "c", "d"), "c", condition="correct_last", seed=0)
        self.assertEqual(layout[-1], "c")
        self.assertEqual(sorted(layout), ["a", "b", "c", "d"])

    def test_correct_middle_places_at_midpoint(self):
        layout = layout_options(("a", "b", "c", "d"), "c", condition="correct_middle", seed=0)
        self.assertEqual(layout[len(layout) // 2], "c")

    def test_seeded_determinism(self):
        first = layout_options(("a", "b", "c", "d"), "c", condition="correct_middle", seed=42)
        second = layout_options(("a", "b", "c", "d"), "c", condition="correct_middle", seed=42)
        self.assertEqual(first, second)

    def test_correct_placement_invariant_under_seed(self):
        # Pre-registered falsifiable claim 3: the correct placement is
        # invariant under the distractor shuffle -- the shuffle only
        # reorders the other options.
        options = ("a", "b", "c", "d")
        for condition, expected in (
            ("correct_first", 0),
            ("correct_middle", 2),
            ("correct_last", 3),
        ):
            for seed in range(20):
                layout = layout_options(options, "c", condition=condition, seed=seed)
                self.assertEqual(layout.index("c"), expected, (condition, seed))

    def test_two_option_middle_is_last(self):
        # len(options) // 2 == 1 for two options: the middle slot is
        # the final slot.
        layout = layout_options(("a", "b"), "b", condition="correct_middle", seed=0)
        self.assertEqual(layout, ("a", "b"))
        self.assertEqual(layout.index("b"), 1)

    def test_bad_inputs_rejected(self):
        with self.assertRaises(ValueError):
            layout_options([], "a", condition="correct_first", seed=0)
        with self.assertRaises(ValueError):
            layout_options(("a", 1), "a", condition="correct_first", seed=0)
        with self.assertRaises(ValueError):
            layout_options(("a", "b"), "c", condition="correct_first", seed=0)
        with self.assertRaises(ValueError):
            layout_options(("a", "b"), "a", condition="sideways", seed=0)
        with self.assertRaises(ValueError):
            layout_options(("a", "b"), "a", condition="correct_first", seed=-1)
        with self.assertRaises(ValueError):
            layout_options(("a", "b"), "a", condition="correct_first", seed=2**64)


class RenderOptionsTests(unittest.TestCase):
    def test_render_format(self):
        text = render_options("Which?", ("b", "a", "c"))
        self.assertIn("Which?", text)
        self.assertIn("A. b", text)
        self.assertIn("B. a", text)
        self.assertIn("C. c", text)
        self.assertNotIn("D.", text)
        self.assertTrue(text.endswith("Reply with exactly the correct option text."))

    def test_bad_inputs_rejected(self):
        with self.assertRaises(ValueError):
            render_options("", ("a", "b"))
        with self.assertRaises(ValueError):
            render_options("Q", ("a",))
        with self.assertRaises(ValueError):
            render_options("Q", ("a", 1))


def _ref_layout(options, correct, condition, seed):
    """Independent oracle reimplementation of layout_options."""
    rng = random.Random(seed)  # noqa: S311 -- deterministic test seed
    distractors = [o for o in options if o != correct]
    rng.shuffle(distractors)
    target = {
        "correct_first": 0,
        "correct_middle": len(options) // 2,
        "correct_last": len(options) - 1,
    }[condition]
    return tuple(distractors[:target] + [correct] + distractors[target:])


class BuildPositionFixturesTests(unittest.TestCase):
    def test_one_fixture_per_spec_per_condition(self):
        specs = validate_position_specs(POSITION_SPECS)
        for condition in CONDITIONS:
            fixtures = build_position_fixtures(specs, condition=condition, seed=20261012)
            self.assertEqual(len(fixtures), 8)
            self.assertEqual(fixtures[0].id, f"position/{condition}/capital")
            self.assertEqual(fixtures[0].expected, "Oslo")
            self.assertEqual(fixtures[0].group, f"position-{condition}")

    def test_fixture_text_matches_oracle_layout(self):
        specs = validate_position_specs((_spec(),))
        for condition in CONDITIONS:
            for seed in (0, 1, 20261012):
                (fixture,) = build_position_fixtures(specs, condition=condition, seed=seed)
                expected_layout = _ref_layout(("w", "x", "y", "z"), "y", condition, seed)
                self.assertEqual(
                    fixture.messages[0].content, render_options("Which one?", expected_layout)
                )

    def test_brute_force_oracle_differential(self):
        rng = random.Random(0xC154)  # noqa: S311 -- deterministic test seed
        words = ["aa", "bb", "cc", "dd", "ee"]
        for _trial in range(200):
            n = rng.randint(2, 4)
            options = tuple(rng.sample(words, n))
            correct = rng.choice(options)
            seed = rng.randrange(2**32)
            condition = rng.choice(CONDITIONS)
            got = layout_options(options, correct, condition=condition, seed=seed)
            want = _ref_layout(options, correct, condition, seed)
            self.assertEqual(got, want, (options, correct, condition, seed))
            self.assertEqual(got.index(correct), want.index(correct))

    def test_bad_condition_rejected(self):
        specs = validate_position_specs(POSITION_SPECS)
        with self.assertRaises(ValueError):
            build_position_fixtures(specs, condition="sideways", seed=0)

    def test_bad_seed_rejected(self):
        specs = validate_position_specs(POSITION_SPECS)
        with self.assertRaises(ValueError):
            build_position_fixtures(specs, condition="correct_first", seed=-1)


class PrimacyIndexTests(unittest.TestCase):
    def test_pure_primacy(self):
        self.assertEqual(primacy_index(8, 8, 8, 0), 1.0)

    def test_pure_recency(self):
        self.assertEqual(primacy_index(8, 0, 8, 8), -1.0)

    def test_neutral(self):
        self.assertEqual(primacy_index(8, 4, 8, 4), 0.0)

    def test_mixed(self):
        self.assertAlmostEqual(primacy_index(4, 3, 4, 1), 0.5)

    def test_malformed_returns_none(self):
        self.assertIsNone(primacy_index(-1, 0, 8, 0))
        self.assertIsNone(primacy_index(8, 9, 8, 0))
        self.assertIsNone(primacy_index(0, 0, 8, 0))
        self.assertIsNone(primacy_index(8, 0, 8, 0.5))
        self.assertIsNone(primacy_index(8, 0, "8", 0))

    @given(
        st.integers(min_value=1, max_value=50),
        st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=100)
    def test_index_bounded(self, n_first, n_last):
        for correct_first in (0, n_first // 2, n_first):
            for correct_last in (0, n_last // 2, n_last):
                index = primacy_index(n_first, correct_first, n_last, correct_last)
                self.assertIsNotNone(index)
                self.assertGreaterEqual(index, -1.0)
                self.assertLessEqual(index, 1.0)


def _format(name: str) -> Format:
    return Format(
        name=name,
        bos="",
        user_prefix=f"[{name}]",
        assistant_prefix="A:",
        turn_end="\n",
    )


class ScriptedPositionBackend:
    """Deterministic backend: emits the option at ``emit_index``.

    ``emit_index`` 0 emulates primacy bias (first-listed option),
    -1 emulates recency bias (last-listed option). ``"raise"`` makes
    generate raise, exercising the fail-closed cell path.
    """

    fingerprint = "cycle154-position-scripted"

    def __init__(self, policy):
        self._policy = dict(policy)

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
        mode = self._policy.get(candidate, 0)
        if mode == "raise":
            raise RuntimeError("generate boom")
        options = [match.group(1) for match in re.finditer(r"^[A-D]\. (.+)$", prompt, re.MULTILINE)]
        if not options:
            return core.Generation((), "error", "unknown")
        text = options[mode]
        ids = self.encode(text + chr(7))
        return core.Generation(ids + (1,) * (max_new_tokens - len(ids)), "limit")


def _run_stage(policy, condition="correct_first", seed=20261012):
    specs = validate_position_specs(POSITION_SPECS)
    fixtures = build_position_fixtures(specs, condition=condition, seed=seed)
    return position_stage(
        ScriptedPositionBackend(policy),
        tuple(_format(c) for c in CANDIDATES),
        fixtures,
        condition=condition,
        boundary=(7,),
        max_new_tokens=16,
        seed=seed,
    )


class PositionStageTests(unittest.TestCase):
    def test_primacy_policy_passes_correct_first(self):
        stage = _run_stage({"alpha": 0, "beta": 0})
        self.assertEqual(stage["condition"], "correct_first")
        self.assertEqual(stage["n_fixtures"], 8)
        self.assertEqual(stage["per_candidate"]["alpha"]["correct"], 8)
        self.assertTrue(stage["per_candidate"]["alpha"]["pass"])

    def test_primacy_policy_fails_correct_last(self):
        stage = _run_stage({"alpha": 0, "beta": 0}, condition="correct_last")
        self.assertEqual(stage["per_candidate"]["alpha"]["correct"], 0)
        self.assertFalse(stage["per_candidate"]["alpha"]["pass"])

    def test_recency_policy_passes_correct_last(self):
        stage = _run_stage({"alpha": -1, "beta": -1}, condition="correct_last")
        self.assertTrue(stage["per_candidate"]["alpha"]["pass"])

    def test_middle_condition_scores_partially(self):
        # Primacy policy under correct_middle: the correct option is at
        # index 2, never the first -- zero correct.
        stage = _run_stage({"alpha": 0, "beta": 0}, condition="correct_middle")
        self.assertEqual(stage["per_candidate"]["alpha"]["correct"], 0)

    def test_raising_backend_cell_is_none(self):
        stage = _run_stage({"alpha": "raise", "beta": 0})
        self.assertIsNone(stage["per_candidate"]["alpha"])
        self.assertIsNotNone(stage["per_candidate"]["beta"])

    def test_bad_inputs_rejected(self):
        specs = validate_position_specs(POSITION_SPECS)
        fixtures = build_position_fixtures(specs, condition="correct_first", seed=0)
        fmts = tuple(_format(c) for c in CANDIDATES)
        backend = ScriptedPositionBackend({})
        with self.assertRaises(ValueError):
            position_stage(None, fmts, fixtures, condition="correct_first", boundary=(7,))
        with self.assertRaises(ValueError):
            position_stage(backend, (), fixtures, condition="correct_first", boundary=(7,))
        with self.assertRaises(ValueError):
            position_stage(backend, fmts, (), condition="correct_first", boundary=(7,))
        with self.assertRaises(ValueError):
            position_stage(backend, fmts, ("junk",), condition="correct_first", boundary=(7,))
        with self.assertRaises(ValueError):
            position_stage(backend, fmts, fixtures, condition="sideways", boundary=(7,))
        with self.assertRaises(ValueError):
            position_stage(backend, fmts, fixtures, condition="correct_first", boundary=())
        with self.assertRaises(ValueError):
            position_stage(
                backend,
                fmts,
                fixtures,
                condition="correct_first",
                boundary=(7,),
                max_new_tokens=0,
            )
        with self.assertRaises(ValueError):
            position_stage(
                backend,
                fmts,
                fixtures,
                condition="correct_first",
                boundary=(7,),
                seed=-1,
            )


def _run_analysis(policy):
    specs = validate_position_specs(POSITION_SPECS)
    stages = {}
    for condition in CONDITIONS:
        fixtures = build_position_fixtures(specs, condition=condition, seed=20261012)
        stages[condition] = position_stage(
            ScriptedPositionBackend(policy),
            tuple(_format(c) for c in CANDIDATES),
            fixtures,
            condition=condition,
            boundary=(7,),
            max_new_tokens=16,
            seed=20261012,
        )
    return position_sensitivity("alpha", "selected", CANDIDATES, stages)


class PositionSensitivityTests(unittest.TestCase):
    def test_primacy_indexes(self):
        result = _run_analysis({"alpha": 0, "beta": -1})
        self.assertTrue(result["available"])
        self.assertEqual(result["primacy_index"]["alpha"], 1.0)
        self.assertEqual(result["primacy_index"]["beta"], -1.0)

    def test_flips_detected(self):
        # Under correct_last the recency-biased beta wins; under
        # correct_first the primacy-biased alpha wins.
        result = _run_analysis({"alpha": 0, "beta": -1})
        flip_conditions = {flip["condition"] for flip in result["flips"]}
        self.assertIn("correct_last", flip_conditions)
        self.assertIn("correct_middle", flip_conditions)
        self.assertNotIn("correct_first", flip_conditions)

    def test_condition_effect(self):
        result = _run_analysis({"alpha": 0, "beta": -1})
        effect = result["condition_effect"]
        self.assertEqual(effect["n_candidates"], 2)
        self.assertEqual(effect["n_positive"], 1)
        self.assertEqual(effect["n_negative"], 1)
        self.assertAlmostEqual(effect["mean_index"], 0.0)

    def test_no_flips_when_all_conditions_agree(self):
        # Hand-built stages: alpha uniquely passes every condition, so
        # every condition matches the primary verdict.
        stages = {}
        for condition in CONDITIONS:
            stages[condition] = {
                "condition": condition,
                "n_fixtures": 8,
                "per_candidate": {
                    "alpha": {"n": 8, "correct": 8, "pass": True},
                    "beta": {"n": 8, "correct": 0, "pass": False},
                },
            }
        result = position_sensitivity("alpha", "selected", CANDIDATES, stages)
        self.assertTrue(result["available"])
        self.assertEqual(result["flips"], [])
        # Alpha passes every condition, so its primacy index is 0.0 --
        # the index measures placement *sensitivity*, not accuracy.
        self.assertEqual(result["primacy_index"]["alpha"], 0.0)
        self.assertEqual(result["primacy_index"]["beta"], 0.0)
        effect = result["condition_effect"]
        self.assertAlmostEqual(effect["mean_index"], 0.0)
        self.assertEqual(effect["n_positive"], 0)
        self.assertEqual(effect["n_negative"], 0)

    def test_missing_condition_degrades_index(self):
        specs = validate_position_specs(POSITION_SPECS)
        stages = {}
        for condition in ("correct_first", "correct_middle"):
            fixtures = build_position_fixtures(specs, condition=condition, seed=0)
            stages[condition] = position_stage(
                ScriptedPositionBackend({"alpha": 0, "beta": 0}),
                tuple(_format(c) for c in CANDIDATES),
                fixtures,
                condition=condition,
                boundary=(7,),
                max_new_tokens=16,
                seed=0,
            )
        result = position_sensitivity("alpha", "selected", CANDIDATES, stages)
        self.assertTrue(result["available"])
        self.assertIsNone(result["primacy_index"]["alpha"])
        self.assertIsNone(result["condition_effect"])

    def test_fail_closed_details(self):
        specs = validate_position_specs(POSITION_SPECS)
        fixtures = build_position_fixtures(specs, condition="correct_first", seed=0)
        fmts = tuple(_format(c) for c in CANDIDATES)
        stage = position_stage(
            ScriptedPositionBackend({}),
            fmts,
            fixtures,
            condition="correct_first",
            boundary=(7,),
            max_new_tokens=16,
            seed=0,
        )
        good = {"correct_first": stage}
        self.assertEqual(
            position_sensitivity("alpha", "selected", (), good)["detail"],
            "invalid_candidates",
        )
        self.assertEqual(
            position_sensitivity("alpha", "selected", ("a", "a"), good)["detail"],
            "invalid_candidates",
        )
        self.assertEqual(
            position_sensitivity("zzz", "selected", CANDIDATES, good)["detail"],
            "primary_winner_unknown",
        )
        self.assertEqual(
            position_sensitivity("alpha", "", CANDIDATES, good)["detail"],
            "primary_decision_invalid",
        )
        self.assertEqual(
            position_sensitivity("alpha", "selected", CANDIDATES, {})["detail"],
            "stages_invalid",
        )
        self.assertEqual(
            position_sensitivity("alpha", "selected", CANDIDATES, {7: stage})["detail"],
            "stages_invalid",
        )
        broken = copy.deepcopy(stage)
        broken["per_candidate"]["alpha"] = {"n": 8, "correct": 99, "pass": True}
        self.assertEqual(
            position_sensitivity("alpha", "selected", CANDIDATES, {"correct_first": broken})[
                "detail"
            ],
            "stage_correct_first_invalid",
        )
        mismatched = copy.deepcopy(stage)
        mismatched["n_fixtures"] = 7
        two = {"correct_first": stage, "correct_last": mismatched}
        self.assertEqual(
            position_sensitivity("alpha", "selected", CANDIDATES, two)["detail"],
            "fixture_count_mismatch",
        )

    def test_unavailable_condition_not_a_flip(self):
        # A condition where no candidate could be evaluated is
        # "unavailable", never a flip.
        stage = _run_stage({"alpha": "raise", "beta": "raise"})
        self.assertEqual(stage["per_candidate"], {"alpha": None, "beta": None})
        stages = {"correct_first": stage}
        result = position_sensitivity("alpha", "selected", CANDIDATES, stages)
        self.assertTrue(result["available"])
        self.assertEqual(result["per_condition"]["correct_first"]["decision"], "unavailable")
        self.assertEqual(result["flips"], [])


class PositionMutationGateTests(unittest.TestCase):
    """Tempdir-module mutation gates (the real tree is never rewritten)."""

    def _run_gate(self, module_name, name, old_text, new_text, scenario):
        mod = __import__(f"model_protocol_lab.{module_name}", fromlist=["x"])
        source = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertEqual(source.count(old_text), 1, f"mutant {name}: anchor not unique")
        mutated = source.replace(old_text, new_text)
        with tempfile.TemporaryDirectory(prefix="c154_mut_", dir="/tmp") as tmpdir:
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

    def _index_sign_scenario(self, mod):
        assert mod.primacy_index(8, 8, 8, 0) == 1.0
        assert mod.primacy_index(8, 0, 8, 8) == -1.0

    def test_mutant_flipped_index_sign_caught(self):
        self._run_gate(
            "position",
            "flipped_index",
            "    return correct_first / n_first - correct_last / n_last",
            "    return correct_last / n_last - correct_first / n_first  # MUTANT",
            self._index_sign_scenario,
        )

    def _boundary_gate_scenario(self, mod):
        specs = validate_position_specs(POSITION_SPECS)
        fixtures = build_position_fixtures(specs, condition="correct_first", seed=20261012)
        stage = mod.position_stage(
            ScriptedPositionBackend({"alpha": 0, "beta": 0}),
            tuple(_format(c) for c in CANDIDATES),
            fixtures,
            condition="correct_first",
            boundary=(7,),
            max_new_tokens=16,
            seed=20261012,
        )
        assert stage["per_candidate"]["alpha"]["correct"] == 8
        assert stage["per_candidate"]["beta"]["correct"] == 8

    def test_mutant_dropped_boundary_gate_caught(self):
        # The mutant counts non-exact observations as "correct": the
        # scenario pins the all-exact count, so it fails on the mutant.
        self._run_gate(
            "experiments",
            "position_no_exact",
            "        correct = sum(1 for obs in report.observations"
            " if obs.exact_match and obs.boundary_found)",
            "        correct = sum(1 for obs in report.observations"
            " if not obs.exact_match and obs.boundary_found)  # MUTANT",
            self._boundary_gate_scenario,
        )

    def _index_drift_scenario(self, mod):
        old = _position_payload(index_alpha=1.0)
        new = _position_payload(index_alpha=-1.0)
        assert mod.diff_position(old, new)["changed"] is True

    def test_mutant_dropped_index_drift_caught(self):
        self._run_gate(
            "diffreports",
            "dropped_index_drift",
            "        or index_drift\n",
            "        or []  # MUTANT\n",
            self._index_drift_scenario,
        )

    def _mean_drift_scenario(self, mod):
        old = _position_payload(mean_index=0.5)
        new = _position_payload(mean_index=-0.5)
        assert mod.diff_position(old, new)["changed"] is True

    def test_mutant_dropped_mean_drift_caught(self):
        self._run_gate(
            "diffreports",
            "dropped_mean_drift",
            "        or mean_index_changed\n",
            "        or False  # MUTANT\n",
            self._mean_drift_scenario,
        )


def _position_payload(index_alpha=1.0, mean_index=0.0):
    per_condition = {}
    for condition in CONDITIONS:
        per_condition[condition] = {
            "decision": "selected",
            "winner": "alpha",
            "n_evaluable": 2,
            "per_candidate": {
                "alpha": {"correct_rate": 1.0, "pass": True},
                "beta": {"correct_rate": 0.0, "pass": False},
            },
        }
    return {
        "available": True,
        "primary_winner": "alpha",
        "primary_decision": "selected",
        "n_fixtures": 8,
        "conditions": list(CONDITIONS),
        "per_condition": per_condition,
        "flips": [],
        "primacy_index": {"alpha": index_alpha, "beta": -1.0},
        "condition_effect": {
            "mean_index": mean_index,
            "n_candidates": 2,
            "n_positive": 1,
            "n_negative": 1,
        },
    }


class DiffPositionTests(unittest.TestCase):
    def test_identical_payloads_unchanged(self):
        payload = _position_payload()
        diff = diff_position(payload, copy.deepcopy(payload))
        self.assertEqual(diff["available"], "both")
        self.assertFalse(diff["changed"])

    def test_index_drift_only_still_changed(self):
        # Single-disjunct pin: only the primacy index moves.
        diff = diff_position(_position_payload(index_alpha=1.0), _position_payload(index_alpha=0.5))
        self.assertTrue(diff["changed"])
        self.assertEqual(len(diff["index_drift"]), 1)
        self.assertEqual(diff["index_drift"][0]["candidate"], "alpha")

    def test_mean_drift_only_still_changed(self):
        diff = diff_position(_position_payload(mean_index=0.5), _position_payload(mean_index=-0.5))
        self.assertTrue(diff["changed"])
        self.assertTrue(diff["mean_index"]["changed"])

    def test_condition_drift_only_still_changed(self):
        old = _position_payload()
        new = _position_payload()
        new["per_condition"]["correct_last"]["winner"] = "beta"
        diff = diff_position(old, new)
        self.assertTrue(diff["changed"])
        self.assertTrue(diff["condition_changed"]["correct_last"]["changed"])

    def test_flip_only_still_changed(self):
        old = _position_payload()
        new = _position_payload()
        new["flips"] = [
            {
                "condition": "correct_last",
                "decision_from": "selected",
                "decision_to": "no_candidate",
                "winner_from": "alpha",
                "winner_to": None,
            }
        ]
        diff = diff_position(old, new)
        self.assertTrue(diff["changed"])
        self.assertTrue(diff["flips_changed"])

    def test_candidate_rate_drift_only_still_changed(self):
        old = _position_payload()
        new = _position_payload()
        new["per_condition"]["correct_first"]["per_candidate"]["beta"]["correct_rate"] = 0.5
        diff = diff_position(old, new)
        self.assertTrue(diff["changed"])
        self.assertEqual(len(diff["candidate_drift"]), 1)

    def test_conditions_added_removed(self):
        old = _position_payload()
        new = _position_payload()
        del new["per_condition"]["correct_middle"]
        new["conditions"] = ["correct_first", "correct_last"]
        diff = diff_position(old, new)
        self.assertTrue(diff["changed"])
        self.assertEqual(diff["conditions_removed"], ["correct_middle"])

    def test_asymmetric(self):
        payload = _position_payload()
        self.assertEqual(diff_position(payload, None)["available"], "old")
        self.assertEqual(diff_position(None, payload)["available"], "new")
        self.assertEqual(diff_position(None, None)["available"], "neither")
        self.assertFalse(diff_position(payload, None)["changed"])

    def test_hostile_payloads_degrade(self):
        good = _position_payload()
        for hostile in (
            None,
            42,
            "junk",
            {"available": True},
            {"available": True, "conditions": "nope"},
            {"available": True, "conditions": [], "per_condition": {}},
            {**good, "primacy_index": {"alpha": "junk"}},
            {**good, "primacy_index": {7: 1.0}},
            {**good, "per_condition": {"correct_first": {"decision": 7}}},
            {**good, "flips": "junk"},
            {**good, "condition_effect": {"mean_index": "junk"}},
        ):
            diff = diff_position(good, hostile)
            self.assertEqual(diff["available"], "old", hostile)
            self.assertFalse(diff["changed"])

    def test_feeds_identical(self):
        old_report = {"decision": "x", "winner": "y", "position": _position_payload()}
        new_report = copy.deepcopy(old_report)
        diff = diff_reports(old_report, new_report)
        self.assertTrue(diff["summary"]["identical"])
        self.assertFalse(diff["summary"]["position_changed"])
        self.assertFalse(diff["summary"]["position_asymmetric"])
        # An index move alone breaks identical.
        new_report["position"]["primacy_index"]["alpha"] = 0.0
        diff = diff_reports(old_report, new_report)
        self.assertFalse(diff["summary"]["identical"])
        self.assertTrue(diff["summary"]["position_changed"])

    def test_summary_renders_hostile_names(self):
        old = _position_payload()
        new = _position_payload(index_alpha=-1.0)
        diff = diff_reports(
            {"decision": "x", "winner": "y", "position": old},
            {"decision": "x", "winner": "y", "position": new},
        )
        summary = render_summary(diff)
        self.assertIn("Primacy-index drift", summary)
        # Terminal escapes in candidate names are stripped on the stderr path.
        evil = _position_payload()
        evil["primacy_index"] = {"\x1b[31mevil": 1.0}
        diff = diff_reports(
            {"decision": "x", "winner": "y", "position": _position_payload()},
            {"decision": "x", "winner": "y", "position": evil},
        )
        self.assertNotIn("\x1b", render_summary(diff))


class PositionSectionTests(unittest.TestCase):
    def test_section_renders(self):
        result = _run_analysis({"alpha": 0, "beta": -1})
        html = _position_section({"position": result})
        self.assertIn('id="position-sensitivity"', html)
        self.assertIn("Primacy index", html)
        self.assertIn("+1.00", html)
        self.assertIn("-1.00", html)
        self.assertIn("differs", html)

    def test_section_absent_when_unavailable(self):
        self.assertEqual(_position_section({}), "")
        self.assertEqual(_position_section({"position": {"available": False}}), "")
        self.assertEqual(_position_section({"position": "junk"}), "")

    def test_section_escapes_hostile(self):
        result = _run_analysis({"alpha": 0, "beta": 0})
        hostile = copy.deepcopy(result)
        hostile["per_condition"]["correct_first"]["per_candidate"]["<script>alert(1)"] = {
            "correct_rate": 1.0,
            "pass": True,
        }
        hostile["primacy_index"] = {"<img src=x onerror=y>": 0.5}
        html = _position_section({"position": hostile})
        self.assertNotIn("<script>alert(1)", html)
        self.assertNotIn("<img src=x onerror=y>", html)
        self.assertIn("&lt;script&gt;", html)


class PositionCoverageTests(unittest.TestCase):
    """Branch coverage for the position renderers and stage edge paths."""

    def _reports(self, old_pos, new_pos):
        base = {"decision": "selected", "winner": "alpha"}
        return {**base, "position": old_pos}, {**base, "position": new_pos}

    def test_summary_position_only_in_old(self):
        old, new = self._reports(_position_payload(), None)
        summary = render_summary(diff_reports(old, new))
        self.assertIn("did not record the position probe", summary)

    def test_summary_neither_side_has_position(self):
        summary = render_summary(diff_reports({"decision": "x"}, {"decision": "x"}))
        self.assertNotIn("Position", summary)

    def test_summary_changed_condition_line(self):
        old = _position_payload()
        new = _position_payload()
        new["per_condition"]["correct_last"]["decision"] = "no_candidate"
        new["per_condition"]["correct_last"]["winner"] = None
        summary = render_summary(diff_reports(*self._reports(old, new)))
        self.assertIn("Position (correct_last):", summary)

    def test_summary_conditions_added_and_removed(self):
        old = _position_payload()
        new = _position_payload()
        del new["per_condition"]["correct_middle"]
        new["conditions"] = ["correct_first", "correct_last"]
        summary = render_summary(diff_reports(*self._reports(old, new)))
        self.assertIn("Position conditions removed: correct_middle", summary)

        old2 = _position_payload()
        new2 = _position_payload()
        new2["per_condition"]["correct_extra"] = {
            "decision": "selected",
            "winner": "alpha",
            "n_evaluable": 2,
            "per_candidate": {"alpha": {"correct_rate": 1.0, "pass": True}},
        }
        new2["conditions"] = [*CONDITIONS, "correct_extra"]
        summary = render_summary(diff_reports(*self._reports(old2, new2)))
        self.assertIn("Position conditions added: correct_extra", summary)

    def test_summary_flips_changed_line(self):
        old = _position_payload()
        new = _position_payload()
        new["flips"] = [
            {
                "condition": "correct_last",
                "decision_from": "selected",
                "decision_to": "no_candidate",
                "winner_from": "alpha",
                "winner_to": None,
            }
        ]
        summary = render_summary(diff_reports(*self._reports(old, new)))
        self.assertIn("Position flip pattern changed between runs.", summary)

    def test_summary_index_drift_truncates(self):
        old = _position_payload()
        new = _position_payload()
        old["primacy_index"] = {f"cand{i}": 1.0 for i in range(6)}
        new["primacy_index"] = {f"cand{i}": 0.0 for i in range(6)}
        summary = render_summary(diff_reports(*self._reports(old, new)))
        self.assertIn("Primacy-index drift (6):", summary)
        self.assertIn("... and 1 more", summary)

    def test_summary_mean_index_changed(self):
        old = _position_payload()
        new = _position_payload(mean_index=0.5)
        summary = render_summary(diff_reports(*self._reports(old, new)))
        self.assertIn("Mean primacy index:", summary)

    def test_summary_candidates_added_removed(self):
        old = _position_payload()
        new = _position_payload()
        new["per_condition"]["correct_first"]["per_candidate"]["gamma"] = {
            "correct_rate": 0.5,
            "pass": False,
        }
        summary = render_summary(diff_reports(*self._reports(old, new)))
        self.assertIn("Position candidates added: gamma", summary)

        old2 = _position_payload()
        new2 = _position_payload()
        for condition in new2["per_condition"].values():
            del condition["per_candidate"]["beta"]
        summary = render_summary(diff_reports(*self._reports(old2, new2)))
        self.assertIn("Position candidates removed: beta", summary)

    def test_section_hostile_structure_degrades(self):
        # per_condition not a dict -> empty section.
        self.assertEqual(_position_section({"position": {"available": True}}), "")
        self.assertEqual(
            _position_section({"position": {"available": True, "per_condition": []}}),
            "",
        )
        # Non-dict verdict / candidates / entries are skipped, not fatal.
        payload = _run_analysis({"alpha": 0, "beta": 0})
        payload["per_condition"]["correct_first"] = "junk"
        payload["per_condition"]["correct_last"]["per_candidate"] = "junk"
        payload["per_condition"]["correct_middle"]["per_candidate"]["alpha"] = "junk"
        html = _position_section({"position": payload})
        self.assertIn('id="position-sensitivity"', html)

    def test_section_non_numeric_rate_and_index(self):
        payload = _run_analysis({"alpha": 0, "beta": 0})
        payload["per_condition"]["correct_first"]["per_candidate"]["alpha"] = {
            "correct_rate": "junk",
            "pass": "maybe",
        }
        payload["primacy_index"] = {"alpha": "junk", "beta": None}
        payload["condition_effect"] = {"mean_index": "junk"}
        payload["flips"] = ["junk", {"condition": "correct_last"}]
        html = _position_section({"position": payload})
        self.assertIn("n/a", html)
        self.assertIn('id="position-sensitivity"', html)

    def test_section_missing_primacy_effect_flips(self):
        payload = _run_analysis({"alpha": 0, "beta": 0})
        del payload["primacy_index"]
        del payload["condition_effect"]
        del payload["flips"]
        html = _position_section({"position": payload})
        self.assertIn('id="position-sensitivity"', html)
        self.assertNotIn("Condition effect", html)

    def test_stage_evaluate_raise_degrades_cell(self):
        class EncodeBoomBackend(ScriptedPositionBackend):
            @staticmethod
            def encode(text):
                raise RuntimeError("encode boom")

        specs = validate_position_specs(POSITION_SPECS)
        fixtures = build_position_fixtures(specs, condition="correct_first", seed=0)
        stage = position_stage(
            EncodeBoomBackend({}),
            tuple(_format(c) for c in CANDIDATES),
            fixtures,
            condition="correct_first",
            boundary=(7,),
        )
        self.assertIsNone(stage["per_candidate"]["alpha"])

    def test_sensitivity_rejects_malformed_entries(self):
        specs = validate_position_specs(POSITION_SPECS)
        fixtures = build_position_fixtures(specs, condition="correct_first", seed=0)
        stage = position_stage(
            ScriptedPositionBackend({}),
            tuple(_format(c) for c in CANDIDATES),
            fixtures,
            condition="correct_first",
            boundary=(7,),
            max_new_tokens=16,
            seed=0,
        )
        for broken_entry, detail in (
            ("junk", "stage_correct_first_invalid"),
            ({"n": 7, "correct": 0, "pass": False}, "stage_correct_first_invalid"),
        ):
            broken = copy.deepcopy(stage)
            broken["per_candidate"]["alpha"] = broken_entry
            self.assertEqual(
                position_sensitivity("alpha", "selected", CANDIDATES, {"correct_first": broken})[
                    "detail"
                ],
                detail,
            )

    def test_sensitivity_rejects_malformed_stage_shapes(self):
        specs = validate_position_specs(POSITION_SPECS)
        fixtures = build_position_fixtures(specs, condition="correct_first", seed=0)
        stage = position_stage(
            ScriptedPositionBackend({}),
            tuple(_format(c) for c in CANDIDATES),
            fixtures,
            condition="correct_first",
            boundary=(7,),
            max_new_tokens=16,
            seed=0,
        )
        cases = [
            {"correct_first": "junk"},
            {"correct_first": {**stage, "condition": 7}},
            {"correct_first": {**stage, "n_fixtures": "8"}},
            {"correct_first": {**stage, "per_candidate": []}},
        ]
        for stages in cases:
            self.assertEqual(
                position_sensitivity("alpha", "selected", CANDIDATES, stages)["detail"],
                "stage_correct_first_invalid",
            )


class CliProbeTests(unittest.TestCase):
    def test_run_position_flag(self):
        from model_protocol_lab import cli

        self.assertTrue(hasattr(cli, "_run_position_probe"))
        probe = cli._run_position_probe
        result = {"decision": "selected", "selected_candidate": "alpha"}
        formats = tuple(_format(c) for c in CANDIDATES)
        probe(
            ScriptedPositionBackend({"alpha": 0, "beta": 0}),
            formats,
            result,
            boundary=(7,),
            boundary_map=None,
            max_new_tokens=16,
            seed=0,
        )
        position = result["position"]
        self.assertTrue(position["available"])
        self.assertEqual(position["n_fixtures"], 8)
        self.assertEqual(position["primacy_index"]["alpha"], 1.0)

    def test_probe_never_raises(self):
        from model_protocol_lab import cli

        result = {"decision": "selected"}
        cli._run_position_probe(
            None,
            (),
            result,
            boundary=None,
            boundary_map=None,
            max_new_tokens=16,
            seed=0,
        )
        self.assertEqual(result["position"]["detail"], "position_unavailable")


class SampleReportTests(unittest.TestCase):
    def test_sample_position_exercise_deterministic(self):
        script = Path(__file__).resolve().parent.parent / "examples" / "make_sample_report.py"
        spec = importlib.util.spec_from_file_location("make_sample_report_c154", script)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory(prefix="c154_sample_") as first:
            result_a = module.build_sample_report(Path(first) / "regen")
        with tempfile.TemporaryDirectory(prefix="c154_sample_") as second:
            result_b = module.build_sample_report(Path(second) / "regen")
        json_a = json.dumps(result_a["position"], sort_keys=True)
        json_b = json.dumps(result_b["position"], sort_keys=True)
        self.assertEqual(json_a, json_b)
        # Genuine signal: primacy-biased base vs recency-biased no-bos.
        self.assertTrue(result_a["position"]["available"])
        primacy = result_a["position"]["primacy_index"]
        values = {v for v in primacy.values() if v is not None}
        self.assertIn(1.0, values)
        self.assertIn(-1.0, values)
        self.assertTrue(result_a["position"]["flips"])

    def test_committed_sample_has_position(self):
        sample = Path(__file__).resolve().parent.parent / "examples" / "sample-report"
        results = json.loads((sample / "results.json").read_text(encoding="utf-8"))
        self.assertTrue(results["position"]["available"])
        self.assertIn("position-sensitivity", (sample / "report.html").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
