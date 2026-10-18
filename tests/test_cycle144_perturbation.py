"""Cycle 144: fixture-perturbation robustness experiment.

``perturbation.py`` provides deterministic seeded phrasing-perturbation
operators (pure, no model). ``experiments.perturbation_stage``
re-evaluates every search candidate on N perturbed fixture variants
via the existing evaluation pipeline; the pure
``experiments.perturbation_robustness`` re-applies the conservative
selection rule per variant and reports whether the primary winner
survives phrasing noise. Wired into ``mpl run --perturb`` (1-8), the
HTML "Fixture-perturbation robustness" section, results.json, and
``mpl diff-reports`` (``diff_perturbation`` with
``perturbation_changed`` / ``perturbation_asymmetric`` summary flags
feeding ``identical``).
"""

import html
import importlib.util
import os
import tempfile
import unicodedata
import unittest
from pathlib import Path

from model_protocol_lab import core, diffreports, experiments
from model_protocol_lab.core import Fixture, Format, Message
from model_protocol_lab.perturbation import (
    PERTURBATION_OPERATORS,
    perturb_fixture,
    perturb_fixtures,
    perturb_text,
    variant_plan,
)
from model_protocol_lab.reporting import _perturbation_section

CANDIDATES = ("alpha", "beta")
LONG_TEXT = (
    "Reply with exactly BLUE and nothing else, without any extra words. "
    "This is a deliberately long fixture prompt so seeded operators have "
    "plenty of characters to work with, deterministically."
)


def _fixture(fid, content, expected="BLUE"):
    return Fixture(
        id=fid,
        messages=(Message(role="user", content=content),),
        expected=expected,
        group=f"g-{fid}",
    )


def _format(name):
    return Format(
        name=name,
        bos="",
        user_prefix=f"[{name}]",
        assistant_prefix="A:",
        turn_end="\n",
    )


class ScriptedBackend:
    """Deterministic backend; answers from candidate behavior, not fixture text.

    Unlike the cycle-138 scripted backend this one must survive
    perturbed prompts, so it keys only on the (unperturbed) candidate
    prefix framing the prompt. ``behavior`` maps candidate name to
    "pass" / "fail" / "error".
    """

    fingerprint = "cycle144-perturbation-scripted"

    def __init__(self, behavior):
        self._behavior = dict(behavior)

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
        mode = self._behavior.get(candidate, "pass")
        if mode == "error":
            return core.Generation((), "error", "boom")
        text = "BLUE" if mode == "pass" else "WRONG"
        ids = self.encode(text + chr(7))
        return core.Generation(ids + (1,) * (max_new_tokens - len(ids)), "limit")


class PerturbTextTests(unittest.TestCase):
    def test_all_operators_change_suitable_input(self):
        for op in PERTURBATION_OPERATORS:
            with self.subTest(operator=op):
                out = perturb_text(LONG_TEXT, op, seed=42)
                self.assertNotEqual(out, LONG_TEXT, op)

    def test_deterministic_in_seed(self):
        for op in PERTURBATION_OPERATORS:
            with self.subTest(operator=op):
                self.assertEqual(
                    perturb_text(LONG_TEXT, op, seed=7),
                    perturb_text(LONG_TEXT, op, seed=7),
                    op,
                )

    def test_seed_sensitivity(self):
        # Two seeds must not silently produce the same perturbation.
        out1 = perturb_text(LONG_TEXT, "typo", seed=1)
        out2 = perturb_text(LONG_TEXT, "typo", seed=2)
        self.assertNotEqual(out1, out2)

    def test_unicode_variant_is_nfkc_foldable(self):
        out = perturb_text("GREEN 123", "unicode_variant", seed=3)
        self.assertNotEqual(out, "GREEN 123")
        # Fullwidth forms fold back to the ASCII originals under NFKC.
        self.assertEqual(unicodedata.normalize("NFKC", out), "GREEN 123")
        self.assertTrue(any(ord(c) > 127 for c in out))

    def test_typo_substitutes_different_letter(self):
        out = perturb_text("aaaa bbbb cccc dddd eeee ffff", "typo", seed=11)
        self.assertEqual(len(out), len("aaaa bbbb cccc dddd eeee ffff"))
        for a, b in zip("aaaa bbbb cccc dddd eeee ffff", out, strict=True):
            if a != b:
                self.assertTrue(a.isalpha() and b.isalpha())
                self.assertNotEqual(a.lower(), b.lower())

    def test_punctuation_strip_deterministic_and_complete(self):
        out = perturb_text("Hello, World! (yes?)", "punctuation_strip", seed=5)
        self.assertEqual(out, "Hello World yes")
        self.assertFalse(any(unicodedata.category(c).startswith("P") for c in out))

    def test_whitespace_all_three_modes(self):
        # Seeds pinned to each rng.randrange(3) mode: collapse, double,
        # newline-split. Deterministic by construction.
        self.assertEqual(perturb_text("a  b   c", "whitespace", seed=1), "a b c")
        self.assertEqual(perturb_text("a b", "whitespace", seed=0), "a  b")
        self.assertEqual(perturb_text("a b", "whitespace", seed=5), "a\nb")

    def test_fail_closed_on_would_be_empty(self):
        # Stripping every character must not produce an empty prompt.
        self.assertEqual(perturb_text("!!!", "punctuation_strip", seed=5), "!!!")
        self.assertEqual(perturb_text("   ", "whitespace", seed=5), "   ")

    def test_noop_when_operator_cannot_apply(self):
        self.assertEqual(perturb_text("abc", "punctuation_strip", seed=5), "abc")
        self.assertEqual(perturb_text("abc", "whitespace", seed=5), "abc")
        self.assertEqual(perturb_text("123", "typo", seed=5), "123")
        self.assertEqual(perturb_text("123", "case_flip", seed=5), "123")
        self.assertEqual(perturb_text("123", "unicode_variant", seed=5), "123")

    def test_validation(self):
        for bad_op in ("scream", "", None, 42):
            with self.assertRaises(ValueError, msg=repr(bad_op)):
                perturb_text("text", bad_op, seed=1)
        for bad_seed in (-1, 2**64, "1", None, True):
            with self.assertRaises(ValueError, msg=repr(bad_seed)):
                perturb_text("text", "typo", seed=bad_seed)
        with self.assertRaises(ValueError):
            perturb_text(123, "typo", seed=1)  # type: ignore[arg-type]


class PerturbFixtureTests(unittest.TestCase):
    def test_id_group_expected_preserved(self):
        fx = _fixture("S1", LONG_TEXT, expected="BLUE")
        out = perturb_fixture(fx, "typo", seed=9)
        self.assertEqual(out.id, "S1")
        self.assertEqual(out.group, "g-S1")
        self.assertEqual(out.expected, "BLUE")
        self.assertEqual(out.messages[0].role, "user")
        self.assertNotEqual(out.messages[0].content, LONG_TEXT)

    def test_all_messages_perturbed(self):
        fx = Fixture(
            id="M",
            messages=(
                Message(role="user", content=LONG_TEXT),
                Message(role="assistant", content=LONG_TEXT),
            ),
            expected="x",
            group="g",
        )
        out = perturb_fixture(fx, "case_flip", seed=4)
        for msg in out.messages:
            self.assertNotEqual(msg.content, LONG_TEXT)

    def test_perturb_fixtures_validates(self):
        cases = (_fixture("S1", LONG_TEXT),)
        out = perturb_fixtures(cases, "typo", seed=2)
        self.assertEqual(len(out), 1)
        self.assertIs(type(out), tuple)
        with self.assertRaises(ValueError):
            perturb_fixtures((), "typo", seed=2)
        with self.assertRaises(ValueError):
            perturb_fixture("nope", "typo", seed=2)  # type: ignore[arg-type]


class VariantPlanTests(unittest.TestCase):
    def test_round_robin_assignment(self):
        plan = variant_plan(5, seed=99)
        self.assertEqual([p["operator"] for p in plan], list(PERTURBATION_OPERATORS))
        self.assertEqual([p["variant_id"] for p in plan], ["v00", "v01", "v02", "v03", "v04"])

    def test_repeat_operators_with_fresh_seeds(self):
        plan = variant_plan(7, seed=99)
        seeds = [p["seed"] for p in plan]
        self.assertEqual(len(set(seeds)), 7)
        self.assertEqual(plan[5]["operator"], "typo")
        self.assertNotEqual(plan[0]["seed"], plan[5]["seed"])

    def test_deterministic(self):
        self.assertEqual(variant_plan(4, seed=1), variant_plan(4, seed=1))
        self.assertNotEqual(variant_plan(4, seed=1), variant_plan(4, seed=2))

    def test_custom_operator_subset(self):
        plan = variant_plan(3, ("typo", "case_flip"), seed=1)
        self.assertEqual([p["operator"] for p in plan], ["typo", "case_flip", "typo"])

    def test_validation(self):
        for bad in (0, 9, -1, "4", None):
            with self.assertRaises(ValueError, msg=repr(bad)):
                variant_plan(bad, seed=1)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            variant_plan(2, ("typo", "typo"), seed=1)
        with self.assertRaises(ValueError):
            variant_plan(2, ("typo", "nope"), seed=1)
        with self.assertRaises(ValueError):
            variant_plan(2, seed=-1)


def _stage(behavior, n_variants=2, operators=("punctuation_strip", "typo"), seed=11):
    cases = (_fixture("S1", LONG_TEXT), _fixture("S2", LONG_TEXT))
    formats = tuple(_format(n) for n in CANDIDATES)
    return experiments.perturbation_stage(
        ScriptedBackend(behavior),
        formats,
        cases,
        n_variants=n_variants,
        operators=operators,
        seed=seed,
        boundary=(7,),
        max_new_tokens=32,
    )


class PerturbationStageTests(unittest.TestCase):
    def test_outcomes_and_shape(self):
        out = _stage({"alpha": "pass", "beta": "fail"})
        self.assertEqual(out["n_variants"], 2)
        self.assertEqual(out["operators"], ["punctuation_strip", "typo"])
        self.assertEqual(sorted(out["variants"]), ["v00", "v01"])
        v00 = out["variants"]["v00"]
        self.assertEqual(v00["operator"], "punctuation_strip")
        self.assertEqual(v00["outcomes"], {"alpha": True, "beta": False})
        self.assertEqual(v00["n_fixtures"], 2)
        # punctuation_strip really did change both fixtures' content.
        self.assertEqual(v00["changed_fixtures"], 2)

    def test_changed_fixtures_honest_zero(self):
        # punctuation_strip on punctuation-free content changes nothing;
        # the stage must report 0, not silently drop the variant.
        cases = (_fixture("S1", "abc def"),)
        formats = (_format("alpha"),)
        out = experiments.perturbation_stage(
            ScriptedBackend({"alpha": "pass"}),
            formats,
            cases,
            n_variants=1,
            operators=("punctuation_strip",),
            seed=1,
            boundary=(7,),
        )
        self.assertEqual(out["variants"]["v00"]["changed_fixtures"], 0)
        self.assertTrue(out["variants"]["v00"]["outcomes"]["alpha"])

    def test_backend_error_records_none(self):
        out = _stage({"alpha": "error", "beta": "pass"})
        self.assertIsNone(out["variants"]["v00"]["outcomes"]["alpha"])
        self.assertTrue(out["variants"]["v00"]["outcomes"]["beta"])

    def test_deterministic(self):
        self.assertEqual(
            _stage({"alpha": "pass", "beta": "fail"}),
            _stage({"alpha": "pass", "beta": "fail"}),
        )

    def test_validation(self):
        cases = (_fixture("S1", LONG_TEXT),)
        formats = (_format("alpha"),)
        with self.assertRaises(ValueError):
            experiments.perturbation_stage(
                None,
                formats,
                cases,
                n_variants=1,
                boundary=(7,),  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            experiments.perturbation_stage(
                ScriptedBackend({}), formats, cases, n_variants=0, boundary=(7,)
            )
        with self.assertRaises(ValueError):
            experiments.perturbation_stage(
                ScriptedBackend({}), formats, cases, n_variants=9, boundary=(7,)
            )
        with self.assertRaises(ValueError):
            experiments.perturbation_stage(
                ScriptedBackend({}), (), cases, n_variants=1, boundary=(7,)
            )


def _robustness(stage, winner="alpha", decision="selected"):
    return experiments.perturbation_robustness(winner, decision, CANDIDATES, stage)


def _stage_dict(specs):
    """Hand-built stage dict: {vid: (operator, {candidate: bool|None}, changed, n)}."""
    variants = {}
    for vid, (op, outcomes, changed, n) in specs.items():
        variants[vid] = {
            "operator": op,
            "seed": 1,
            "changed_fixtures": changed,
            "n_fixtures": n,
            "outcomes": outcomes,
        }
    return {"n_variants": len(variants), "operators": ["typo"], "variants": variants}


class PerturbationRobustnessTests(unittest.TestCase):
    def test_stable_winner(self):
        stage = _stage_dict(
            {
                "v00": ("typo", {"alpha": True, "beta": False}, 2, 2),
                "v01": ("typo", {"alpha": True, "beta": False}, 2, 2),
            }
        )
        out = _robustness(stage)
        self.assertTrue(out["available"])
        self.assertEqual(out["stability"]["winner_retained"], 2)
        self.assertEqual(out["stability"]["decision_retained"], 2)
        self.assertEqual(out["stability"]["n_evaluable"], 2)
        self.assertEqual(out["flips"], [])
        self.assertEqual(out["per_variant"]["v00"]["decision"], "selected")
        self.assertEqual(out["per_variant"]["v00"]["winner"], "alpha")

    def test_flip_detected(self):
        stage = _stage_dict(
            {
                "v00": ("typo", {"alpha": True, "beta": False}, 2, 2),
                "v01": ("typo", {"alpha": False, "beta": True}, 2, 2),
            }
        )
        out = _robustness(stage)
        self.assertEqual(len(out["flips"]), 1)
        flip = out["flips"][0]
        self.assertEqual(flip["variant"], "v01")
        self.assertEqual(flip["winner_from"], "alpha")
        self.assertEqual(flip["winner_to"], "beta")
        self.assertEqual(out["stability"]["winner_retained"], 1)

    def test_ambiguous_variant(self):
        stage = _stage_dict({"v00": ("typo", {"alpha": True, "beta": True}, 2, 2)})
        out = _robustness(stage)
        self.assertEqual(out["per_variant"]["v00"]["decision"], "ambiguous")
        self.assertIsNone(out["per_variant"]["v00"]["winner"])
        self.assertEqual(len(out["flips"]), 1)

    def test_unavailable_variant_is_not_a_verdict(self):
        stage = _stage_dict(
            {
                "v00": ("typo", {"alpha": True, "beta": False}, 2, 2),
                "v01": ("typo", {"alpha": None, "beta": None}, 0, 2),
            }
        )
        out = _robustness(stage)
        self.assertEqual(out["per_variant"]["v01"]["decision"], "unavailable")
        self.assertEqual(out["stability"]["n_evaluable"], 1)
        self.assertEqual(out["stability"]["n_unavailable"], 1)
        # An unevaluable variant must not count as a flip.
        self.assertEqual(out["flips"], [])

    def test_no_primary_winner(self):
        stage = _stage_dict({"v00": ("typo", {"alpha": True, "beta": False}, 2, 2)})
        out = experiments.perturbation_robustness(None, "abstain", CANDIDATES, stage)
        self.assertTrue(out["available"])
        self.assertIsNone(out["stability"]["winner_retained"])
        self.assertIsNone(out["per_operator"]["typo"]["winner_retained"])
        self.assertEqual(out["stability"]["decision_retained"], 0)

    def test_per_operator_aggregation(self):
        stage = _stage_dict(
            {
                "v00": ("typo", {"alpha": True, "beta": False}, 2, 2),
                "v01": ("typo", {"alpha": False, "beta": True}, 1, 2),
                "v02": ("case_flip", {"alpha": True, "beta": False}, 2, 2),
            }
        )
        out = _robustness(stage)
        typo = out["per_operator"]["typo"]
        self.assertEqual(typo["variants"], 2)
        self.assertEqual(typo["evaluable"], 2)
        self.assertEqual(typo["winner_retained"], 1)
        self.assertAlmostEqual(typo["mean_changed_rate"], 0.75)
        self.assertEqual(out["per_operator"]["case_flip"]["winner_retained"], 1)

    def test_order_invariant(self):
        s1 = _stage_dict(
            {
                "v00": ("typo", {"alpha": True, "beta": False}, 2, 2),
                "v01": ("typo", {"alpha": False, "beta": True}, 2, 2),
            }
        )
        s2 = _stage_dict(
            {
                "v01": ("typo", {"alpha": False, "beta": True}, 2, 2),
                "v00": ("typo", {"alpha": True, "beta": False}, 2, 2),
            }
        )
        self.assertEqual(_robustness(s1), _robustness(s2))

    def test_malformed_input_unavailable(self):
        good = _stage_dict({"v00": ("typo", {"alpha": True, "beta": False}, 2, 2)})
        bad_stages = [
            None,
            "nope",
            {},
            {"variants": {}},
            {"variants": {"v00": "nope"}},
            {"variants": {1: good["variants"]["v00"]}},  # non-string key
            {"variants": {"v00": {"operator": "typo"}}},  # missing outcomes
            {
                "variants": {
                    "v00": {
                        "operator": "typo",
                        "outcomes": {"alpha": "yes", "beta": False},
                    }
                }
            },
        ]
        for bad in bad_stages:
            with self.subTest(bad=repr(bad)[:60]):
                out = experiments.perturbation_robustness("alpha", "selected", CANDIDATES, bad)
                self.assertFalse(out["available"])
        # Bad candidate tuples / primary winner / decision also degrade.
        for winner, decision, cands in [
            ("gamma", "selected", CANDIDATES),
            ("alpha", "", CANDIDATES),
            ("alpha", "selected", ("alpha", "alpha")),
            ("alpha", "selected", ()),
        ]:
            out = experiments.perturbation_robustness(winner, decision, cands, good)
            self.assertFalse(out["available"])

    def test_end_to_end_stage_to_robustness(self):
        stage = _stage({"alpha": "pass", "beta": "fail"})
        out = _robustness(stage)
        self.assertTrue(out["available"])
        self.assertEqual(out["n_variants"], 2)
        self.assertEqual(out["stability"]["winner_retained"], 2)
        self.assertEqual(out["flips"], [])


def _perturbation_payload(**overrides):
    payload = {
        "available": True,
        "primary_winner": "alpha",
        "primary_decision": "selected",
        "n_variants": 2,
        "operators": ["typo"],
        "per_variant": {
            "v00": {
                "operator": "typo",
                "decision": "selected",
                "winner": "alpha",
                "changed_fixtures": 2,
                "changed_rate": 1.0,
            },
            "v01": {
                "operator": "typo",
                "decision": "selected",
                "winner": "alpha",
                "changed_fixtures": 2,
                "changed_rate": 1.0,
            },
        },
        "flips": [],
        "stability": {
            "n_variants": 2,
            "n_evaluable": 2,
            "n_unavailable": 0,
            "winner_retained": 2,
            "decision_retained": 2,
        },
        "per_operator": {},
    }
    payload.update(overrides)
    return payload


def _reports(old_pert, new_pert):
    base = {
        "decision": "selected",
        "selected_candidate": "alpha",
        "search": [],
        "held_out": {},
    }
    old = dict(base, perturbation=old_pert)
    new = dict(base, perturbation=new_pert)
    return old, new


class DiffPerturbationTests(unittest.TestCase):
    def test_identical_payloads_no_change(self):
        p = _perturbation_payload()
        diff = diffreports.diff_perturbation(p, _perturbation_payload())
        self.assertEqual(diff["available"], "both")
        self.assertFalse(diff["changed"])
        self.assertFalse(diff["stability"]["changed"])
        self.assertEqual(diff["variant_drift"], [])

    def test_stability_drift(self):
        old = _perturbation_payload()
        new = _perturbation_payload()
        new["per_variant"]["v01"] = {
            "operator": "typo",
            "decision": "ambiguous",
            "winner": None,
            "changed_fixtures": 2,
            "changed_rate": 1.0,
        }
        new["stability"] = {
            "n_variants": 2,
            "n_evaluable": 2,
            "n_unavailable": 0,
            "winner_retained": 1,
            "decision_retained": 1,
        }
        diff = diffreports.diff_perturbation(old, new)
        self.assertTrue(diff["changed"])
        self.assertTrue(diff["stability"]["changed"])
        self.assertEqual(len(diff["variant_drift"]), 1)
        drift = diff["variant_drift"][0]
        self.assertEqual(drift["variant"], "v01")
        self.assertEqual(drift["old_winner"], "alpha")
        self.assertIsNone(drift["new_winner"])

    def test_variants_added_removed(self):
        old = _perturbation_payload()
        new = _perturbation_payload()
        new["per_variant"]["v02"] = {
            "operator": "typo",
            "decision": "selected",
            "winner": "alpha",
            "changed_fixtures": 1,
            "changed_rate": 0.5,
        }
        diff = diffreports.diff_perturbation(old, new)
        self.assertTrue(diff["changed"])
        self.assertEqual(diff["variants_added"], ["v02"])
        self.assertEqual(diff["variants_removed"], [])

    def test_operators_changed(self):
        old = _perturbation_payload()
        new = _perturbation_payload(operators=["typo", "case_flip"])
        diff = diffreports.diff_perturbation(old, new)
        self.assertTrue(diff["changed"])
        self.assertTrue(diff["operators_changed"])

    def test_asymmetric(self):
        diff = diffreports.diff_perturbation(_perturbation_payload(), {"available": False})
        self.assertEqual(diff["available"], "old")
        self.assertFalse(diff["changed"])
        diff = diffreports.diff_perturbation(None, None)
        self.assertEqual(diff["available"], "neither")

    def test_hostile_non_string_keys_degrade(self):
        hostile = _perturbation_payload()
        hostile["per_variant"] = {1: {"decision": "selected", "winner": "alpha"}}
        diff = diffreports.diff_perturbation(hostile, _perturbation_payload())
        # Old side fails structural validation -> asymmetric, never a crash.
        self.assertEqual(diff["available"], "new")
        self.assertFalse(diff["changed"])

    def test_summary_flags_feed_identical(self):
        old, new = _reports(_perturbation_payload(), _perturbation_payload())
        diff = diffreports.diff_reports(old, new)
        self.assertFalse(diff["summary"]["perturbation_changed"])
        self.assertFalse(diff["summary"]["perturbation_asymmetric"])
        self.assertTrue(diff["summary"]["identical"])
        self.assertIn("perturbation", diff)
        # A stability change breaks identical.
        new2 = _perturbation_payload()
        new2["stability"] = dict(new2["stability"], winner_retained=1)
        new2["per_variant"]["v01"] = dict(
            new2["per_variant"]["v01"], decision="ambiguous", winner=None
        )
        old2, new2r = _reports(_perturbation_payload(), new2)
        diff2 = diffreports.diff_reports(old2, new2r)
        self.assertTrue(diff2["summary"]["perturbation_changed"])
        self.assertFalse(diff2["summary"]["identical"])
        # Asymmetry alone breaks identical.
        old3, new3 = _reports(_perturbation_payload(), {"available": False})
        diff3 = diffreports.diff_reports(old3, new3)
        self.assertTrue(diff3["summary"]["perturbation_asymmetric"])
        self.assertFalse(diff3["summary"]["identical"])


class RenderPerturbationSummaryTests(unittest.TestCase):
    def test_stability_change_line(self):
        old = _perturbation_payload()
        new = _perturbation_payload()
        new["stability"] = dict(new["stability"], winner_retained=0, decision_retained=1)
        new["per_variant"]["v00"] = dict(
            new["per_variant"]["v00"], decision="no_candidate", winner=None
        )
        new["per_variant"]["v01"] = dict(
            new["per_variant"]["v01"], decision="no_candidate", winner=None
        )
        diff = diffreports.diff_perturbation(old, new)
        lines = diffreports.render_summary(
            {
                "old": {"source": "a"},
                "new": {"source": "b"},
                "config_drift": [],
                "decision": {"old": "selected", "new": "selected"},
                "winner": {"old": "alpha", "new": "alpha"},
                "frontier": {"available": "neither"},
                "held_out_floor": {"available": "neither"},
                "boundary_audit": {"available": "neither"},
                "selection_robustness": {"available": "neither"},
                "transfer": {"available": "neither"},
                "contamination": {"available": "neither"},
                "perturbation": diff,
                "candidate_verdicts": [],
                "evidence": {"old": False, "new": False},
                "summary": {
                    "identical": False,
                    "decision_changed": False,
                    "winner_changed": False,
                    "candidates_changed_status": 0,
                    "fixtures_flipped": 0,
                    "config_drift_warnings": 0,
                    "frontier_changed": False,
                },
            }
        )
        self.assertIn("Perturbation stability: 2/2 -> 0/2 variants retain the winner", lines)
        self.assertIn("Perturbation variant drift (2):", lines)

    def test_hostile_variant_id_sanitized_in_stderr(self):
        old = _perturbation_payload()
        new = _perturbation_payload()
        evil = "v01\x1b[31m"
        new["per_variant"][evil] = dict(
            new["per_variant"]["v00"], winner=None, decision="ambiguous"
        )
        diff = diffreports.diff_perturbation(old, new)
        lines = diffreports._render_perturbation_summary(diff)
        joined = "\n".join(lines)
        self.assertNotIn("\x1b[31m", joined)
        self.assertIn("v01", joined)


class PerturbationSectionTests(unittest.TestCase):
    def _result(self, **overrides):
        payload = _perturbation_payload(**overrides)
        return {"perturbation": payload}

    def test_section_renders(self):
        section_html = _perturbation_section(self._result())
        self.assertIn("Fixture-perturbation robustness", section_html)
        self.assertIn("2/2 evaluable variants retain the primary winner", section_html)
        self.assertIn("v00", section_html)
        self.assertIn("typo", section_html)

    def test_flip_badge(self):
        payload = _perturbation_payload()
        payload["per_variant"]["v01"] = dict(
            payload["per_variant"]["v01"], decision="ambiguous", winner=None
        )
        payload["flips"] = [
            {
                "variant": "v01",
                "operator": "typo",
                "decision_from": "selected",
                "decision_to": "ambiguous",
                "winner_from": "alpha",
                "winner_to": None,
            }
        ]
        section_html = _perturbation_section({"perturbation": payload})
        self.assertIn("1 variant(s) flip the primary verdict", section_html)
        self.assertIn("badge fail", section_html)

    def test_unavailable_renders_nothing(self):
        self.assertEqual(_perturbation_section({"perturbation": {"available": False}}), "")
        self.assertEqual(_perturbation_section({}), "")

    def test_malformed_renders_nothing(self):
        self.assertEqual(_perturbation_section({"perturbation": {"available": True}}), "")
        self.assertEqual(
            _perturbation_section({"perturbation": {"available": True, "per_variant": []}}),
            "",
        )

    def test_hostile_names_escaped(self):
        # Cycle-142 pattern: pin escaping at the integration point.
        evil = '<script>alert("x")</script>&"\x1b[31m'
        payload = _perturbation_payload(primary_winner=evil)
        payload["per_variant"] = {
            evil: {
                "operator": evil,
                "decision": "selected",
                "winner": evil,
                "changed_fixtures": 2,
                "changed_rate": 1.0,
            }
        }
        payload["per_operator"] = {
            evil: {"variants": 1, "evaluable": 1, "winner_retained": 1, "mean_changed_rate": 1.0}
        }
        section_html = _perturbation_section({"perturbation": payload})
        escaped = html.escape(evil)
        self.assertNotIn(evil, section_html)
        self.assertIn(escaped, section_html)

    def test_no_primary_winner_line(self):
        payload = _perturbation_payload(primary_winner=None)
        payload["stability"] = dict(payload["stability"], winner_retained=None)
        section_html = _perturbation_section({"perturbation": payload})
        self.assertIn("retain the primary decision", section_html)


class PerturbationMutationGateTests(unittest.TestCase):
    """In-place mutation gates use the tempdir-module pattern (no pyc poisoning)."""

    def _run_gate(self, name, old_text, new_text, scenario):
        from model_protocol_lab import perturbation as perturbation_mod

        source = Path(perturbation_mod.__file__).read_text(encoding="utf-8")
        self.assertEqual(source.count(old_text), 1, f"mutant {name}: anchor not unique")
        mutated = source.replace(old_text, new_text)
        with tempfile.TemporaryDirectory(prefix="c144_mut_", dir="/tmp") as tmpdir:
            mod_path = os.path.join(tmpdir, f"perturbation_mut_{name}.py")
            with open(mod_path, "w", encoding="utf-8") as handle:
                handle.write(mutated)
            # Dotted name gives the mutant package context so the
            # module's relative import (``from .core import ...``)
            # resolves against the real package.
            dotted = f"model_protocol_lab.perturbation_mut_{name}"
            spec = importlib.util.spec_from_file_location(dotted, mod_path)
            assert spec is not None and spec.loader is not None
            mutant = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mutant)
            scenario(perturbation_mod)  # passes on the real module: non-vacuous
            with self.assertRaises(AssertionError, msg=f"mutant {name} was NOT caught"):
                scenario(mutant)

    def test_mutant_guard_removed_caught(self):
        def scenario(mod):
            # Without the fail-closed guard, punctuation_strip on "!!!"
            # would return an empty prompt instead of the original.
            self.assertEqual(mod.perturb_text("!!!", "punctuation_strip", seed=5), "!!!")

        self._run_gate(
            "guard_removed",
            "    if not perturbed or not perturbed.strip():\n"
            "        return text\n"
            "    return perturbed",
            "    return perturbed  # MUTANT: guard removed",
            scenario,
        )

    def test_mutant_seed_ignored_caught(self):
        def scenario(mod):
            # Without seeding, two seeds must still... no: with seeding,
            # different seeds give different typos on long text.
            out1 = mod.perturb_text(LONG_TEXT, "typo", seed=1)
            out2 = mod.perturb_text(LONG_TEXT, "typo", seed=2)
            self.assertNotEqual(out1, out2)

        self._run_gate(
            "seed_ignored",
            "    rng = random.Random(seed)  # noqa: S311 # nosec B311\n",
            "    rng = random.Random(0)  # MUTANT: seed ignored\n",
            scenario,
        )


if __name__ == "__main__":
    unittest.main()
