"""Cycle 147: instruction-hierarchy adherence probe (wave-2 idea 1).

When a system instruction and a user instruction conflict, which one does
each chat template lead the model to follow -- and does the winning
protocol change under conflict? The lab never sends system messages, so
the system instruction is delivered as a privilege-marked block folded
into the user turn (before/after the user instruction); adherence is
scored per observation by exact string equality against the two expected
answers.

Covers: conflict-spec validation, privilege-block rendering, fixture
construction, the catalog-derived family treatment labels (pinned),
``hierarchy_stage`` (scripted backend, fail-closed cells),
``hierarchy_adherence`` (selection re-application, flips, position
effect, group means, malformed-input degradation), ``diff_hierarchy``
(availability, changed disjuncts, hostile payloads), the stderr summary
lines, the HTML section (escaping), four tempdir-module mutation gates,
and a brute-force oracle differential for the pure analysis.
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
from model_protocol_lab.diffreports import diff_hierarchy, diff_reports, render_summary
from model_protocol_lab.experiments import hierarchy_adherence, hierarchy_stage
from model_protocol_lab.hierarchy import (
    CONFLICT_SPECS,
    FAMILY_SYSTEM_TREATMENT,
    POSITIONS,
    PRIVILEGE_CLOSE,
    PRIVILEGE_OPEN,
    build_conflict_fixtures,
    family_of,
    privilege_block,
    system_treatment,
    validate_conflict_specs,
)
from model_protocol_lab.reporting import _hierarchy_section

CANDIDATES = ("alpha", "beta")


def _format(name: str) -> Format:
    """Format."""
    return Format(
        name=name,
        bos="",
        user_prefix=f"[{name}]",
        assistant_prefix="A:",
        turn_end="\n",
    )


class ScriptedHierarchyBackend:
    """Deterministic backend with per-candidate instruction-following policy.

    ``policy`` maps candidate name to "system" (emit the system answer),
    "user" (emit the user answer), or "neither" (emit an unrelated
    string). ``pairs`` maps a unique system-instruction substring to the
    (system_answer, user_answer) pair. "raise" policy makes ``generate``
    raise, exercising the fail-closed cell path.
    """

    fingerprint = "cycle147-hierarchy-scripted"

    def __init__(self, policy, pairs):
        self._policy = dict(policy)
        self._pairs = dict(pairs)

    @staticmethod
    def encode(text):
        """Encode."""
        return tuple(map(ord, text))

    @staticmethod
    def decode(ids):
        """Decode."""
        return "".join(map(chr, ids))

    def generate(self, input_ids, *, max_new_tokens, seed, cancel):
        """Generate."""
        prompt = self.decode(input_ids)
        candidate = next((n for n in CANDIDATES if f"[{n}]" in prompt), None)
        if candidate is None:
            return core.Generation((), "error", "unknown")
        mode = self._policy.get(candidate, "system")
        if mode == "raise":
            raise RuntimeError("generate boom")
        pair = next(
            (pair for needle, pair in self._pairs.items() if needle in prompt),
            None,
        )
        if pair is None:
            return core.Generation((), "error", "unknown")
        system_answer, user_answer = pair
        text = {"system": system_answer, "user": user_answer}.get(mode, "WRONG")
        ids = self.encode(text + chr(7))
        return core.Generation(ids + (1,) * (max_new_tokens - len(ids)), "limit")


def _pairs():
    """Pairs."""
    specs = validate_conflict_specs(CONFLICT_SPECS)
    return {spec["system"]: (spec["expected_system"], spec["expected_user"]) for spec in specs}


def _run_stage(policy, position="privilege_first"):
    """Run stage."""
    specs = validate_conflict_specs(CONFLICT_SPECS)
    fixtures, expectations = build_conflict_fixtures(specs, position=position)
    return hierarchy_stage(
        ScriptedHierarchyBackend(policy, _pairs()),
        tuple(_format(c) for c in CANDIDATES),
        fixtures,
        expectations,
        position=position,
        boundary=(7,),
        max_new_tokens=16,
        seed=0,
    )


class ConflictSpecValidationTests(unittest.TestCase):
    """Conflict spec validation tests."""

    def test_valid_specs_pass(self):
        """Test valid specs pass."""
        specs = validate_conflict_specs(CONFLICT_SPECS)
        self.assertEqual(len(specs), 8)

    def test_empty_rejected(self):
        """Test empty rejected."""
        with self.assertRaises(ValueError):
            validate_conflict_specs(())
        with self.assertRaises(ValueError):
            validate_conflict_specs([])

    def test_missing_key_rejected(self):
        """Test missing key rejected."""
        bad = dict(CONFLICT_SPECS[0])
        del bad["expected_user"]
        with self.assertRaises(ValueError):
            validate_conflict_specs((bad,))

    def test_blank_field_rejected(self):
        """Test blank field rejected."""
        bad = dict(CONFLICT_SPECS[0])
        bad["system"] = "   "
        with self.assertRaises(ValueError):
            validate_conflict_specs((bad,))

    def test_duplicate_id_rejected(self):
        """Test duplicate id rejected."""
        with self.assertRaises(ValueError):
            validate_conflict_specs((CONFLICT_SPECS[0], CONFLICT_SPECS[0]))

    def test_equal_answers_rejected(self):
        """Test equal answers rejected."""
        bad = dict(CONFLICT_SPECS[0])
        bad["expected_user"] = bad["expected_system"]
        with self.assertRaises(ValueError):
            validate_conflict_specs((bad,))

    def test_non_dict_rejected(self):
        """Test non dict rejected."""
        with self.assertRaises(ValueError):
            validate_conflict_specs((CONFLICT_SPECS[0], "junk"))


class PrivilegeBlockTests(unittest.TestCase):
    """Privilege block tests."""

    def test_block_format(self):
        """Test block format."""
        block = privilege_block("do the thing")
        self.assertTrue(block.startswith(PRIVILEGE_OPEN + "\n"))
        self.assertTrue(block.endswith("\n" + PRIVILEGE_CLOSE))
        self.assertIn("do the thing", block)

    def test_blank_rejected(self):
        """Test blank rejected."""
        with self.assertRaises(ValueError):
            privilege_block("  ")
        with self.assertRaises(ValueError):
            privilege_block(123)


class BuildConflictFixturesTests(unittest.TestCase):
    """Build conflict fixtures tests."""

    def test_positions_order_content(self):
        """Test positions order content."""
        specs = validate_conflict_specs(CONFLICT_SPECS)
        first, _exp_first = build_conflict_fixtures(specs, position="privilege_first")
        last, _exp_last = build_conflict_fixtures(specs, position="privilege_last")
        self.assertEqual(len(first), 8)
        content_first = first[0].messages[0].content
        content_last = last[0].messages[0].content
        block_at = content_first.index(PRIVILEGE_OPEN)
        user_at = content_first.index(specs[0]["user"])
        self.assertLess(block_at, user_at)
        block_at = content_last.index(PRIVILEGE_OPEN)
        user_at = content_last.index(specs[0]["user"])
        self.assertGreater(block_at, user_at)

    def test_expected_is_system_answer(self):
        """Test expected is system answer."""
        specs = validate_conflict_specs(CONFLICT_SPECS)
        fixtures, expectations = build_conflict_fixtures(specs, position="privilege_first")
        for fixture, spec in zip(fixtures, specs, strict=True):
            self.assertEqual(fixture.expected, spec["expected_system"])
            self.assertEqual(expectations[fixture.id]["expected_user"], spec["expected_user"])
            self.assertIsInstance(fixture.messages[0], Message)
            self.assertEqual(fixture.messages[0].role, "user")

    def test_ids_unique_and_grouped(self):
        """Test ids unique and grouped."""
        specs = validate_conflict_specs(CONFLICT_SPECS)
        fixtures, _ = build_conflict_fixtures(specs, position="privilege_first")
        ids = [f.id for f in fixtures]
        self.assertEqual(len(set(ids)), len(ids))
        self.assertTrue(all(f.group == "hierarchy-privilege_first" for f in fixtures))

    def test_bad_position_rejected(self):
        """Test bad position rejected."""
        specs = validate_conflict_specs(CONFLICT_SPECS)
        with self.assertRaises(ValueError):
            build_conflict_fixtures(specs, position="sideways")


class FamilyLabelTests(unittest.TestCase):
    """Family label tests."""

    def test_pinned_dedicated_families(self):
        """Test pinned dedicated families."""
        for family in (
            "mistral_large",
            "gemma4",
            "kimi3",
            "kolibri",
            "nemotron35",
            "muse_glimmer",
            "laguna",
            "ling",
            "minimax3",
            "nanbeige42",
            "qwen3",
            "glm53",
            "k2horizon",
            "hy4",
        ):
            self.assertEqual(system_treatment(family), "dedicated", family)

    def test_pinned_folded_families(self):
        """Test pinned folded families."""
        for family in ("gemma", "gemma3", "deepseek"):
            self.assertEqual(system_treatment(family), "folded", family)

    def test_unknown_by_default(self):
        """Test unknown by default."""
        for family in ("chatml", "llama3", "qwen38", "phi4", "muse_spark"):
            self.assertEqual(system_treatment(family), "unknown", family)
        self.assertEqual(system_treatment("not-a-family"), "unknown")
        self.assertEqual(system_treatment(123), "unknown")
        self.assertEqual(system_treatment(None), "unknown")

    def test_labels_cover_only_known_catalog_families(self):
        """Test labels cover only known catalog families."""
        for family in FAMILY_SYSTEM_TREATMENT:
            self.assertIn(family, CATALOG)

    def test_labels_are_valid_treatments(self):
        """Test labels are valid treatments."""
        for treatment in FAMILY_SYSTEM_TREATMENT.values():
            self.assertIn(treatment, ("dedicated", "folded"))

    def test_family_of(self):
        """Test family of."""
        self.assertEqual(family_of("llama3"), "llama3")
        self.assertEqual(family_of("llama3/toggle-bos"), "llama3")
        self.assertEqual(family_of("a/b/c"), "a")
        with self.assertRaises(ValueError):
            family_of("")
        with self.assertRaises(ValueError):
            family_of(42)


class HierarchyStageTests(unittest.TestCase):
    """Hierarchy stage tests."""

    def test_adherence_counts(self):
        """Test adherence counts."""
        stage = _run_stage({"alpha": "system", "beta": "user"})
        self.assertEqual(stage["position"], "privilege_first")
        self.assertEqual(stage["n_fixtures"], 8)
        alpha = stage["per_candidate"]["alpha"]
        self.assertEqual((alpha["system"], alpha["user"], alpha["neither"]), (8, 0, 0))
        self.assertIs(alpha["pass"], True)
        beta = stage["per_candidate"]["beta"]
        self.assertEqual((beta["system"], beta["user"], beta["neither"]), (0, 8, 0))
        self.assertIs(beta["pass"], False)

    def test_neither_counts(self):
        """Test neither counts."""
        stage = _run_stage({"alpha": "neither", "beta": "system"})
        alpha = stage["per_candidate"]["alpha"]
        self.assertEqual((alpha["system"], alpha["user"], alpha["neither"]), (0, 0, 8))
        self.assertIs(alpha["pass"], False)

    def test_fail_closed_on_generate_raise(self):
        """Test fail closed on generate raise."""
        stage = _run_stage({"alpha": "raise", "beta": "system"})
        self.assertIsNone(stage["per_candidate"]["alpha"])
        self.assertIsNotNone(stage["per_candidate"]["beta"])

    def test_stage_validates_inputs(self):
        """Test stage validates inputs."""
        specs = validate_conflict_specs(CONFLICT_SPECS)
        fixtures, expectations = build_conflict_fixtures(specs, position="privilege_first")
        formats = tuple(_format(c) for c in CANDIDATES)
        backend = ScriptedHierarchyBackend({"alpha": "system"}, _pairs())
        with self.assertRaises(ValueError):
            hierarchy_stage(
                backend, formats, fixtures, {}, position="privilege_first", boundary=(7,)
            )
        with self.assertRaises(ValueError):
            hierarchy_stage(
                backend,
                formats,
                fixtures,
                expectations,
                position="sideways",
                boundary=(7,),
            )
        with self.assertRaises(ValueError):
            hierarchy_stage(
                backend,
                formats,
                fixtures,
                expectations,
                position="privilege_first",
                boundary=(),
            )
        with self.assertRaises(ValueError):
            hierarchy_stage(
                None, formats, fixtures, expectations, position="privilege_first", boundary=(7,)
            )  # noqa: E501


def _adherence_stage(per_candidate, position="privilege_first", n_fixtures=8):
    """Adherence stage."""
    return {
        "position": position,
        "n_fixtures": n_fixtures,
        "per_candidate": per_candidate,
    }


def _entry(system, user, neither, passed):
    """Entry."""
    return {"n": 8, "system": system, "user": user, "neither": neither, "pass": passed}


class HierarchyAdherenceTests(unittest.TestCase):
    """Hierarchy adherence tests."""

    @staticmethod
    def _stages():
        """Stages."""
        return {
            "privilege_first": _adherence_stage(
                {
                    "alpha": _entry(8, 0, 0, True),
                    "beta": _entry(0, 8, 0, False),
                }
            ),
            "privilege_last": _adherence_stage(
                {
                    "alpha": _entry(4, 4, 0, False),
                    "beta": _entry(0, 8, 0, False),
                }
            ),
        }

    def test_selection_reapplied_per_condition(self):
        """Test selection reapplied per condition."""
        result = hierarchy_adherence("alpha", "selected", CANDIDATES, self._stages())
        self.assertTrue(result["available"])
        first = result["per_condition"]["privilege_first"]
        self.assertEqual((first["decision"], first["winner"]), ("selected", "alpha"))
        last = result["per_condition"]["privilege_last"]
        self.assertEqual((last["decision"], last["winner"]), ("no_candidate", None))

    def test_flips_against_primary(self):
        """Test flips against primary."""
        result = hierarchy_adherence("alpha", "selected", CANDIDATES, self._stages())
        flips = result["flips"]
        self.assertEqual(len(flips), 1)
        self.assertEqual(flips[0]["position"], "privilege_last")
        self.assertEqual(flips[0]["decision_from"], "selected")
        self.assertEqual(flips[0]["decision_to"], "no_candidate")

    def test_no_flip_when_matching(self):
        """Test no flip when matching."""
        stages = {
            "privilege_first": _adherence_stage(
                {"alpha": _entry(8, 0, 0, True), "beta": _entry(0, 8, 0, False)}
            )
        }
        result = hierarchy_adherence("alpha", "selected", CANDIDATES, stages)
        self.assertEqual(result["flips"], [])

    def test_ambiguous_and_unavailable(self):
        """Test ambiguous and unavailable."""
        stages = {
            "privilege_first": _adherence_stage(
                {"alpha": _entry(8, 0, 0, True), "beta": _entry(8, 0, 0, True)}
            ),
            "privilege_last": _adherence_stage({"alpha": None, "beta": None}),
        }
        result = hierarchy_adherence("alpha", "selected", CANDIDATES, stages)
        self.assertEqual(result["per_condition"]["privilege_first"]["decision"], "ambiguous")
        self.assertEqual(result["per_condition"]["privilege_last"]["decision"], "unavailable")
        # Unavailable conditions never appear as flips.
        self.assertEqual(
            result["flips"],
            [
                {
                    "position": "privilege_first",
                    "decision_from": "selected",
                    "decision_to": "ambiguous",
                    "winner_from": "alpha",
                    "winner_to": None,
                }
            ],
        )

    def test_position_effect(self):
        """Test position effect."""
        result = hierarchy_adherence("alpha", "selected", CANDIDATES, self._stages())
        effect = result["position_effect"]
        self.assertIsNotNone(effect)
        self.assertIs(effect["winner_changed"], True)
        # privilege_first mean system rate 0.5, privilege_last 0.25.
        self.assertAlmostEqual(effect["mean_system_rate_delta"], 0.25)

    def test_position_effect_absent_with_one_position(self):
        """Test position effect absent with one position."""
        stages = {
            "privilege_first": _adherence_stage(
                {"alpha": _entry(8, 0, 0, True), "beta": _entry(0, 8, 0, False)}
            )
        }
        result = hierarchy_adherence("alpha", "selected", CANDIDATES, stages)
        self.assertIsNone(result["position_effect"])

    def test_group_adherence_means(self):
        """Test group adherence means."""
        stages = {
            "privilege_first": _adherence_stage(
                {
                    "gemma3": _entry(8, 0, 0, True),
                    "gemma3/no-bos": _entry(8, 0, 0, True),
                    "kolibri": _entry(4, 4, 0, False),
                    "chatml": _entry(0, 8, 0, False),
                }
            )
        }
        cands = ("gemma3", "gemma3/no-bos", "kolibri", "chatml")
        result = hierarchy_adherence("gemma3", "selected", cands, stages)
        groups = result["group_adherence"]
        self.assertEqual(groups["folded"]["n_candidates"], 2)
        self.assertEqual(groups["folded"]["n_families"], 1)
        self.assertAlmostEqual(groups["folded"]["mean_system_rate"], 1.0)
        self.assertAlmostEqual(groups["dedicated"]["mean_system_rate"], 0.5)
        self.assertAlmostEqual(groups["unknown"]["mean_system_rate"], 0.0)

    def test_malformed_inputs_degrade(self):
        """Test malformed inputs degrade."""
        good = self._stages()
        self.assertEqual(
            hierarchy_adherence("alpha", "selected", (), good)["detail"],
            "invalid_candidates",
        )
        self.assertEqual(
            hierarchy_adherence("alpha", "selected", ("alpha", "alpha"), good)["detail"],
            "invalid_candidates",
        )
        self.assertEqual(
            hierarchy_adherence("gamma", "selected", CANDIDATES, good)["detail"],
            "primary_winner_unknown",
        )
        self.assertEqual(
            hierarchy_adherence("alpha", "", CANDIDATES, good)["detail"],
            "primary_decision_invalid",
        )
        self.assertEqual(
            hierarchy_adherence("alpha", "selected", CANDIDATES, {})["detail"],
            "stages_invalid",
        )
        self.assertEqual(
            hierarchy_adherence("alpha", "selected", CANDIDATES, {"p": {}})["detail"],
            "stage_p_invalid",
        )
        mismatch = dict(good)
        mismatch["privilege_last"] = _adherence_stage(
            {"alpha": _entry(8, 0, 0, True), "beta": _entry(0, 8, 0, False)},
            n_fixtures=4,
        )
        self.assertEqual(
            hierarchy_adherence("alpha", "selected", CANDIDATES, mismatch)["detail"],
            "fixture_count_mismatch",
        )
        bad_entry = {
            "privilege_first": _adherence_stage(
                {
                    "alpha": {"n": 8, "system": 5, "user": 5, "neither": 0, "pass": True},
                    "beta": _entry(0, 8, 0, False),
                }
            )
        }
        self.assertEqual(
            hierarchy_adherence("alpha", "selected", CANDIDATES, bad_entry)["detail"],
            "stage_privilege_first_invalid",
        )

    def test_rates_none_when_no_fixtures(self):
        """Test rates none when no fixtures."""
        stages = {
            "privilege_first": {
                "position": "privilege_first",
                "n_fixtures": 0,
                "per_candidate": {
                    "alpha": {"n": 0, "system": 0, "user": 0, "neither": 0, "pass": True},
                    "beta": None,
                },
            }
        }
        result = hierarchy_adherence("alpha", "selected", CANDIDATES, stages)
        entry = result["per_condition"]["privilege_first"]["per_candidate"]["alpha"]
        self.assertIsNone(entry["system_rate"])


def _hierarchy_payload(**overrides):
    """Hierarchy payload."""
    payload = {
        "available": True,
        "primary_winner": "alpha",
        "primary_decision": "selected",
        "n_fixtures": 8,
        "positions": ["privilege_first", "privilege_last"],
        "per_condition": {
            "privilege_first": {
                "decision": "selected",
                "winner": "alpha",
                "n_evaluable": 2,
                "per_candidate": {
                    "alpha": {
                        "system_rate": 1.0,
                        "user_rate": 0.0,
                        "neither_rate": 0.0,
                        "pass": True,
                    },
                    "beta": {
                        "system_rate": 0.0,
                        "user_rate": 1.0,
                        "neither_rate": 0.0,
                        "pass": False,
                    },
                },
            },
            "privilege_last": {
                "decision": "no_candidate",
                "winner": None,
                "n_evaluable": 2,
                "per_candidate": {
                    "alpha": {
                        "system_rate": 0.5,
                        "user_rate": 0.5,
                        "neither_rate": 0.0,
                        "pass": False,
                    },
                    "beta": {
                        "system_rate": 0.0,
                        "user_rate": 1.0,
                        "neither_rate": 0.0,
                        "pass": False,
                    },
                },
            },
        },
        "flips": [
            {
                "position": "privilege_last",
                "decision_from": "selected",
                "decision_to": "no_candidate",
                "winner_from": "alpha",
                "winner_to": None,
            }
        ],
        "position_effect": {"winner_changed": True, "mean_system_rate_delta": 0.25},
        "group_adherence": {
            "dedicated": {"n_candidates": 1, "n_families": 1, "mean_system_rate": 0.5},
            "folded": {"n_candidates": 1, "n_families": 1, "mean_system_rate": 1.0},
            "unknown": {"n_candidates": 0, "n_families": 0, "mean_system_rate": None},
        },
    }
    payload.update(overrides)
    return payload


class DiffHierarchyTests(unittest.TestCase):
    """Diff hierarchy tests."""

    def test_identical_payloads_unchanged(self):
        """Test identical payloads unchanged."""
        diff = diff_hierarchy(_hierarchy_payload(), _hierarchy_payload())
        self.assertEqual(diff["available"], "both")
        self.assertFalse(diff["changed"])

    def test_condition_triplet_changed(self):
        """Test condition triplet changed."""
        new = _hierarchy_payload()
        new["per_condition"]["privilege_first"]["winner"] = "beta"
        diff = diff_hierarchy(_hierarchy_payload(), new)
        self.assertTrue(diff["changed"])
        cond = diff["condition_changed"]["privilege_first"]
        self.assertTrue(cond["changed"])
        self.assertEqual(cond["new_winner"], "beta")

    def test_flips_changed(self):
        """Test flips changed."""
        new = _hierarchy_payload()
        new["flips"] = []
        diff = diff_hierarchy(_hierarchy_payload(), new)
        self.assertTrue(diff["changed"])
        self.assertTrue(diff["flips_changed"])

    def test_candidate_rate_drift(self):
        """Test candidate rate drift."""
        new = _hierarchy_payload()
        new["per_condition"]["privilege_last"]["per_candidate"]["alpha"]["system_rate"] = 0.75
        diff = diff_hierarchy(_hierarchy_payload(), new)
        self.assertTrue(diff["changed"])
        self.assertEqual(len(diff["candidate_drift"]), 1)
        entry = diff["candidate_drift"][0]
        self.assertEqual(entry["candidate"], "alpha")
        self.assertEqual(entry["position"], "privilege_last")

    def test_group_rates_changed(self):
        """Test group rates changed."""
        new = _hierarchy_payload()
        new["group_adherence"]["dedicated"]["mean_system_rate"] = 0.9
        diff = diff_hierarchy(_hierarchy_payload(), new)
        self.assertTrue(diff["changed"])
        self.assertTrue(diff["group_rates"]["changed"])

    def test_candidates_added_removed(self):
        """Test candidates added removed."""
        new = _hierarchy_payload()
        new["per_condition"]["privilege_first"]["per_candidate"]["gamma"] = {
            "system_rate": 1.0,
            "user_rate": 0.0,
            "neither_rate": 0.0,
            "pass": True,
        }
        diff = diff_hierarchy(_hierarchy_payload(), new)
        self.assertTrue(diff["changed"])
        self.assertEqual(diff["candidates_added"], ["gamma"])

    def test_positions_added(self):
        """Test positions added."""
        new = _hierarchy_payload()
        new["positions"] = ["privilege_first", "privilege_last", "privilege_middle"]
        new["per_condition"]["privilege_middle"] = {
            "decision": "selected",
            "winner": "alpha",
            "n_evaluable": 2,
            "per_candidate": {},
        }
        diff = diff_hierarchy(_hierarchy_payload(), new)
        self.assertTrue(diff["changed"])
        self.assertEqual(diff["positions_added"], ["privilege_middle"])

    def test_availability(self):
        """Test availability."""
        self.assertEqual(
            diff_hierarchy(_hierarchy_payload(), {"available": False})["available"], "old"
        )
        self.assertEqual(
            diff_hierarchy({"available": False}, _hierarchy_payload())["available"], "new"
        )
        self.assertEqual(diff_hierarchy(None, None)["available"], "neither")

    def test_hostile_payloads_invalid(self):
        """Test hostile payloads invalid."""
        # Non-string map keys must fail validation (wave-2 C3 lesson).
        evil = _hierarchy_payload()
        evil["per_condition"]["privilege_first"]["per_candidate"][7] = {"system_rate": 1.0}
        self.assertEqual(diff_hierarchy(evil, _hierarchy_payload())["available"], "new")
        # Non-string decision must fail validation.
        evil2 = _hierarchy_payload()
        evil2["per_condition"]["privilege_first"]["decision"] = 42
        self.assertEqual(diff_hierarchy(evil2, _hierarchy_payload())["available"], "new")
        # Bad rate type must fail validation.
        evil3 = _hierarchy_payload()
        evil3["per_condition"]["privilege_first"]["per_candidate"]["alpha"]["system_rate"] = "high"
        self.assertEqual(diff_hierarchy(evil3, _hierarchy_payload())["available"], "new")


class HierarchyDiffIntegrationTests(unittest.TestCase):
    """Hierarchy diff integration tests."""

    @staticmethod
    def _reports(hierarchy_old, hierarchy_new):
        """Reports."""
        base = {
            "decision": "select_best",
            "selected_candidate": "alpha",
            "search": [],
            "held_out": {},
        }
        old = dict(base, hierarchy=hierarchy_old)
        new = dict(base, hierarchy=hierarchy_new)
        return old, new

    def test_summary_flags_and_identical(self):
        """Test summary flags and identical."""
        old, new = self._reports(_hierarchy_payload(), _hierarchy_payload())
        diff = diff_reports(old, new)
        self.assertFalse(diff["summary"]["hierarchy_changed"])
        self.assertFalse(diff["summary"]["hierarchy_asymmetric"])
        self.assertTrue(diff["summary"]["identical"])

    def test_changed_feeds_identical(self):
        """Test changed feeds identical."""
        changed = _hierarchy_payload()
        changed["per_condition"]["privilege_first"]["winner"] = "beta"
        old, new = self._reports(_hierarchy_payload(), changed)
        diff = diff_reports(old, new)
        self.assertTrue(diff["summary"]["hierarchy_changed"])
        self.assertFalse(diff["summary"]["identical"])

    def test_asymmetric_feeds_identical(self):
        """Test asymmetric feeds identical."""
        old, new = self._reports(_hierarchy_payload(), {"available": False})
        diff = diff_reports(old, new)
        self.assertTrue(diff["summary"]["hierarchy_asymmetric"])
        self.assertFalse(diff["summary"]["identical"])

    def test_stderr_summary_lines(self):
        """Test stderr summary lines."""
        changed = _hierarchy_payload()
        changed["per_condition"]["privilege_first"]["winner"] = "beta"
        old, new = self._reports(_hierarchy_payload(), changed)
        summary = render_summary(diff_reports(old, new))
        self.assertIn("Instruction hierarchy (privilege_first)", summary)
        self.assertIn("selected/alpha -> selected/beta", summary)

    def test_identical_axes_names_hierarchy(self):
        """Test identical axes names hierarchy."""
        old, new = self._reports(_hierarchy_payload(), _hierarchy_payload())
        summary = render_summary(diff_reports(old, new))
        self.assertIn("instruction-hierarchy adherence", summary)


class HierarchySectionTests(unittest.TestCase):
    """Hierarchy section tests."""

    @staticmethod
    def _result():
        """Result."""
        return {"hierarchy": _hierarchy_payload()}

    def test_renders_tables(self):
        """Test renders tables."""
        section_html = _hierarchy_section(self._result())
        self.assertIn("Instruction-hierarchy adherence", section_html)
        self.assertIn("privilege_first", section_html)
        self.assertIn("Per-candidate adherence", section_html)
        self.assertIn("dedicated", section_html)

    def test_empty_when_absent(self):
        """Test empty when absent."""
        self.assertEqual(_hierarchy_section({}), "")
        self.assertEqual(_hierarchy_section({"hierarchy": {"available": False}}), "")

    def test_hostile_non_string_keys_do_not_crash(self):
        """Test hostile non string keys do not crash."""
        result = self._result()
        hierarchy = result["hierarchy"]
        hierarchy["per_condition"][7] = {
            "decision": "selected",
            "winner": "alpha",
            "per_candidate": {8: {"system_rate": 1.0, "pass": True}},
        }
        section_html = _hierarchy_section(result)
        self.assertIn("Instruction-hierarchy adherence", section_html)
        self.assertIn("7", section_html)

    def test_escapes_hostile_candidate_names(self):
        """Test escapes hostile candidate names."""
        import html as html_module

        result = self._result()
        evil = "<script>alert(1)</script>\x1b[31m"
        pc = result["hierarchy"]["per_condition"]["privilege_first"]["per_candidate"]
        pc[evil] = {
            "system_rate": 1.0,
            "user_rate": 0.0,
            "neither_rate": 0.0,
            "pass": True,
        }
        section_html = _hierarchy_section(result)
        self.assertNotIn("<script>", section_html)
        self.assertIn(html_module.escape(evil), section_html)


class HierarchyMutationGateTests(unittest.TestCase):
    """Tempdir-module mutation gates (the real tree is never rewritten)."""

    def _run_gate(self, module_name, name, old_text, new_text, scenario):
        """Run gate."""
        mod = __import__(f"model_protocol_lab.{module_name}", fromlist=["x"])
        source = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertEqual(source.count(old_text), 1, f"mutant {name}: anchor not unique")
        mutated = source.replace(old_text, new_text)
        with tempfile.TemporaryDirectory(prefix="c147_mut_", dir="/tmp") as tmpdir:
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

    @staticmethod
    def _flip_scenario(mod):
        """Flip scenario."""
        stages = {
            "privilege_first": {
                "position": "privilege_first",
                "n_fixtures": 8,
                "per_candidate": {
                    "alpha": {"n": 8, "system": 8, "user": 0, "neither": 0, "pass": True},
                    "beta": {"n": 8, "system": 0, "user": 8, "neither": 0, "pass": False},
                },
            }
        }
        result = mod.hierarchy_adherence("alpha", "ambiguous", CANDIDATES, stages)
        # The condition verdict (selected/alpha) differs from the primary
        # (ambiguous/None): exactly one flip must be recorded.
        assert len(result["flips"]) == 1, f"expected 1 flip, got {result['flips']}"

    def test_mutant_inverted_flip_detection_caught(self):
        """Test mutant inverted flip detection caught."""
        self._run_gate(
            "experiments",
            "inverted_flip",
            """    for position in sorted(per_condition):
        verdict = per_condition[position]
        if verdict["decision"] == "unavailable":
            continue
        if (verdict["decision"], verdict["winner"]) != (primary_decision, primary_winner):""",  # noqa: E501
            """    for position in sorted(per_condition):
        verdict = per_condition[position]
        if verdict["decision"] == "unavailable":
            continue
        if (verdict["decision"], verdict["winner"]) == (primary_decision, primary_winner):  # MUTANT""",  # noqa: E501
            self._flip_scenario,
        )

    @staticmethod
    def _diff_scenario(mod):
        """Diff scenario."""
        old = _hierarchy_payload()
        new = _hierarchy_payload()
        new["flips"] = []
        diff = mod.diff_hierarchy(old, new)
        assert diff["changed"] is True, "flips-only change must set changed"
        assert diff["flips_changed"] is True

    def test_mutant_dropped_flips_disjunct_caught(self):
        """Test mutant dropped flips disjunct caught."""
        self._run_gate(
            "diffreports",
            "dropped_flips_disjunct",
            'flips_changed = _flip_key(old.get("flips")) != _flip_key(new.get("flips"))',
            'flips_changed = _flip_key(old.get("flips")) == _flip_key(new.get("flips"))  # MUTANT',  # noqa: E501
            self._diff_scenario,
        )

    @staticmethod
    def _group_scenario(mod):
        """Group scenario."""
        old = _hierarchy_payload()
        new = _hierarchy_payload()
        new["group_adherence"]["dedicated"]["mean_system_rate"] = 0.9
        diff = mod.diff_hierarchy(old, new)
        assert diff["changed"] is True, "group-rate change must set changed"
        assert diff["group_rates"]["changed"] is True

    def test_mutant_dropped_group_disjunct_caught(self):
        """Test mutant dropped group disjunct caught."""
        self._run_gate(
            "diffreports",
            "dropped_group_disjunct",
            "group_changed = old_groups != new_groups",
            "group_changed = old_groups == new_groups  # MUTANT",
            self._group_scenario,
        )

    @staticmethod
    def _swap_scenario(mod):
        """Swap scenario."""
        specs = mod.validate_conflict_specs(mod.CONFLICT_SPECS)
        _, expectations = mod.build_conflict_fixtures(specs, position="privilege_first")
        fid = "hierarchy/privilege_first/color-word"
        assert expectations[fid]["expected_system"] == "azure", expectations[fid]
        assert expectations[fid]["expected_user"] == "emerald", expectations[fid]

    def test_mutant_swapped_expectations_caught(self):
        """Test mutant swapped expectations caught."""
        self._run_gate(
            "hierarchy",
            "swapped_expectations",
            """        expectations[fixture_id] = {
            "expected_system": spec["expected_system"],
            "expected_user": spec["expected_user"],
        }""",
            """        expectations[fixture_id] = {
            "expected_system": spec["expected_user"],
            "expected_user": spec["expected_system"],
        }  # MUTANT""",
            self._swap_scenario,
        )


def _oracle_adherence(primary_winner, primary_decision, candidates, stages):
    """Independent reimplementation of hierarchy_adherence (test oracle)."""
    per_condition = {}
    for position in sorted(stages):
        stage = stages[position]
        per_candidate = stage["per_candidate"]
        winners = [
            c
            for c in candidates
            if per_candidate.get(c) is not None and per_candidate[c]["pass"] is True
        ]
        evaluable = [c for c in candidates if per_candidate.get(c) is not None]
        if not evaluable:
            decision, winner = "unavailable", None
        elif len(winners) == 1:
            decision, winner = "selected", winners[0]
        elif winners:
            decision, winner = "ambiguous", None
        else:
            decision, winner = "no_candidate", None
        rates = {}
        for c in candidates:
            entry = per_candidate.get(c)
            if entry is None:
                rates[c] = None
            else:
                n = entry["n"]
                rates[c] = {
                    "system_rate": entry["system"] / n if n else None,
                    "user_rate": entry["user"] / n if n else None,
                    "neither_rate": entry["neither"] / n if n else None,
                    "pass": entry["pass"],
                }
        per_condition[position] = {
            "decision": decision,
            "winner": winner,
            "n_evaluable": len(evaluable),
            "per_candidate": rates,
        }
    flips = []
    for position in sorted(per_condition):
        verdict = per_condition[position]
        if verdict["decision"] == "unavailable":
            continue
        if (verdict["decision"], verdict["winner"]) != (primary_decision, primary_winner):
            flips.append(
                {
                    "position": position,
                    "decision_from": primary_decision,
                    "decision_to": verdict["decision"],
                    "winner_from": primary_winner,
                    "winner_to": verdict["winner"],
                }
            )
    return per_condition, flips


class HierarchyOracleTests(unittest.TestCase):
    """Brute-force differential: the pure analysis vs an independent oracle."""

    def test_oracle_agrees_on_random_stages(self):
        """Test oracle agrees on random stages."""
        rng = random.Random(20261018)  # noqa: S311 -- deterministic test seed
        for _ in range(200):
            n_candidates = rng.randint(1, 4)
            candidates = tuple(f"cand{i}" for i in range(n_candidates))
            n_positions = rng.randint(1, 2)
            positions = rng.sample(list(POSITIONS), n_positions)
            n_fixtures = rng.randint(0, 6)
            stages = {}
            for position in positions:
                per_candidate = {}
                for cand in candidates:
                    if rng.random() < 0.2:
                        per_candidate[cand] = None
                    else:
                        system = rng.randint(0, n_fixtures)
                        user = rng.randint(0, n_fixtures - system)
                        neither = n_fixtures - system - user
                        per_candidate[cand] = {
                            "n": n_fixtures,
                            "system": system,
                            "user": user,
                            "neither": neither,
                            "pass": rng.random() < 0.5,
                        }
                stages[position] = {
                    "position": position,
                    "n_fixtures": n_fixtures,
                    "per_candidate": per_candidate,
                }
            primary_winner = rng.choice([None, *candidates])
            primary_decision = rng.choice(["selected", "ambiguous", "no_candidate"])
            result = hierarchy_adherence(primary_winner, primary_decision, candidates, stages)
            self.assertTrue(result["available"])
            oracle_conditions, oracle_flips = _oracle_adherence(
                primary_winner, primary_decision, candidates, stages
            )
            self.assertEqual(result["per_condition"], oracle_conditions)
            self.assertEqual(result["flips"], oracle_flips)
            self.assertEqual(result["positions"], sorted(positions))
            self.assertEqual(result["n_fixtures"], n_fixtures)


class HierarchyBranchCoverageTests(unittest.TestCase):
    """Pin the fail-closed and defensive branches (coverage gate)."""

    def test_validator_rejects_each_malformed_shape(self):
        """Test validator rejects each malformed shape."""

        def _positions_not_list(payload):
            """Positions not list."""
            payload.update(positions="privilege_first")

        def _per_condition_key_mismatch(payload):
            """Per condition key mismatch."""
            payload["per_condition"].pop("privilege_last")

        def _condition_not_dict(payload):
            """Condition not dict."""
            payload["per_condition"].update(privilege_first="x")

        def _winner_not_str(payload):
            """Winner not str."""
            payload["per_condition"]["privilege_first"].update(winner=42)

        def _entry_not_dict(payload):
            """Entry not dict."""
            payload["per_condition"]["privilege_first"]["per_candidate"].update(alpha="x")

        def _pass_not_bool(payload):
            """Pass not bool."""
            entry = payload["per_condition"]["privilege_first"]["per_candidate"]["alpha"]
            entry["pass"] = "yes"

        def _groups_not_dict(payload):
            """Groups not dict."""
            payload.update(group_adherence=[])

        def _group_not_dict(payload):
            """Group not dict."""
            payload["group_adherence"].update(dedicated="x")

        def _group_label_not_str(payload):
            """Group label not str."""
            payload["group_adherence"][7] = {
                "n_candidates": 0,
                "n_families": 0,
                "mean_system_rate": None,
            }

        def _mean_not_number(payload):
            """Mean not number."""
            payload["group_adherence"]["dedicated"].update(mean_system_rate="high")

        def _n_candidates_not_int(payload):
            """N candidates not int."""
            payload["group_adherence"]["dedicated"].update(n_candidates="1")

        def _n_families_not_int(payload):
            """N families not int."""
            payload["group_adherence"]["dedicated"].update(n_families=1.5)

        cases = {
            "positions_not_list": _positions_not_list,
            "per_condition_key_mismatch": _per_condition_key_mismatch,
            "condition_not_dict": _condition_not_dict,
            "winner_not_str": _winner_not_str,
            "entry_not_dict": _entry_not_dict,
            "pass_not_bool": _pass_not_bool,
            "groups_not_dict": _groups_not_dict,
            "group_not_dict": _group_not_dict,
            "group_label_not_str": _group_label_not_str,
            "mean_not_number": _mean_not_number,
            "n_candidates_not_int": _n_candidates_not_int,
            "n_families_not_int": _n_families_not_int,
        }
        for name, mutate in cases.items():
            with self.subTest(name=name):
                evil = _hierarchy_payload()
                mutate(evil)
                diff = diff_hierarchy(evil, _hierarchy_payload())
                self.assertEqual(diff["available"], "new", name)

    def test_validator_accepts_none_entry(self):
        """Test validator accepts none entry."""
        # A None per-candidate entry is skipped, not invalid.
        payload = _hierarchy_payload()
        payload["per_condition"]["privilege_first"]["per_candidate"]["alpha"] = None
        diff = diff_hierarchy(payload, _hierarchy_payload())
        self.assertEqual(diff["available"], "both")

    def test_adherence_rejects_malformed_stages(self):
        """Test adherence rejects malformed stages."""

        def _bad(**overrides):
            """Bad."""
            stage = _adherence_stage({"alpha": _entry(8, 0, 0, True)})
            stage.update(overrides)
            return {"privilege_first": stage}

        cases = [
            ("stage_not_dict", {"privilege_first": "x"}),
            ("position_not_str", _bad(position=42)),
            ("n_fixtures_not_int", _bad(n_fixtures="8")),
            ("per_candidate_bad_entry", _bad(per_candidate={"alpha": "x"})),
        ]
        for name, stages in cases:
            with self.subTest(name=name):
                result = hierarchy_adherence("alpha", "selected", CANDIDATES, stages)
                self.assertFalse(result["available"])
                self.assertEqual(result["detail"], "stage_privilege_first_invalid", name)

    def test_flip_key_skips_non_dict_entries(self):
        """Test flip key skips non dict entries."""
        old = _hierarchy_payload()
        new = _hierarchy_payload()
        new["flips"] = [None, old["flips"][0]]
        diff = diff_hierarchy(old, new)
        self.assertEqual(diff["available"], "both")
        self.assertFalse(diff["flips_changed"])

    def test_adherence_rejects_bad_entry_shapes(self):
        """Test adherence rejects bad entry shapes."""

        def _stage_with(entry):
            """Stage with."""
            return {"privilege_first": _adherence_stage({"alpha": entry})}

        bad_entries = {
            "n_mismatch": {"n": 7, "system": 7, "user": 0, "neither": 0, "pass": True},
            "negative_count": {
                "n": 8,
                "system": -1,
                "user": 9,
                "neither": 0,
                "pass": False,
            },
            "sum_mismatch": {
                "n": 8,
                "system": 3,
                "user": 3,
                "neither": 3,
                "pass": False,
            },
        }
        for name, entry in bad_entries.items():
            with self.subTest(name=name):
                result = hierarchy_adherence("alpha", "selected", CANDIDATES, _stage_with(entry))
                self.assertFalse(result["available"])
                self.assertEqual(result["detail"], "stage_privilege_first_invalid")

    def test_adherence_rejects_non_dict_per_candidate(self):
        """Test adherence rejects non dict per candidate."""
        for per_candidate in ("x", {7: _entry(8, 0, 0, True)}):
            with self.subTest(per_candidate=per_candidate):
                stage = _adherence_stage({"alpha": _entry(8, 0, 0, True)})
                stage["per_candidate"] = per_candidate
                result = hierarchy_adherence(
                    "alpha", "selected", CANDIDATES, {"privilege_first": stage}
                )
                self.assertFalse(result["available"])
                self.assertEqual(result["detail"], "stage_privilege_first_invalid")

    def test_render_summary_positions_added_removed(self):
        """Test render summary positions added removed."""
        base = {
            "decision": "select_best",
            "selected_candidate": "alpha",
            "search": [],
            "held_out": {},
        }
        old = _hierarchy_payload()
        old["positions"] = ["privilege_first"]
        old["per_condition"] = {"privilege_first": old["per_condition"]["privilege_first"]}
        new = _hierarchy_payload()
        summary = render_summary(diff_reports(dict(base, hierarchy=old), dict(base, hierarchy=new)))
        self.assertIn("Hierarchy positions added", summary)
        summary = render_summary(diff_reports(dict(base, hierarchy=new), dict(base, hierarchy=old)))
        self.assertIn("Hierarchy positions removed", summary)

    def test_render_summary_flips_changed_line(self):
        """Test render summary flips changed line."""
        base = {
            "decision": "select_best",
            "selected_candidate": "alpha",
            "search": [],
            "held_out": {},
        }
        new = _hierarchy_payload()
        new["flips"] = []
        summary = render_summary(
            diff_reports(
                dict(base, hierarchy=_hierarchy_payload()),
                dict(base, hierarchy=new),
            )
        )
        self.assertIn("flip pattern changed", summary)

    def test_section_tolerates_malformed_optionals(self):
        """Test section tolerates malformed optionals."""
        result = {"hierarchy": _hierarchy_payload()}
        hierarchy = result["hierarchy"]
        hierarchy["flips"] = "not-a-list"
        hierarchy["position_effect"] = "not-a-dict"
        hierarchy["group_adherence"] = "not-a-dict"
        html = _hierarchy_section(result)
        self.assertIn("Instruction-hierarchy adherence", html)
        self.assertNotIn("correlation readout", html)

    def test_section_skips_non_dict_flip_entries(self):
        """Test section skips non dict flip entries."""
        result = {"hierarchy": _hierarchy_payload()}
        flips = result["hierarchy"]["flips"]
        result["hierarchy"]["flips"] = [None, flips[0]]
        html = _hierarchy_section(result)
        self.assertIn("privilege_last", html)
        old = _hierarchy_payload()
        new = _hierarchy_payload()
        new["flips"] = [None, old["flips"][0]]
        diff = diff_hierarchy(old, new)
        self.assertEqual(diff["available"], "both")
        self.assertFalse(diff["flips_changed"])

    def test_render_summary_neither_and_non_dict(self):
        """Test render summary neither and non dict."""
        from model_protocol_lab.diffreports import _render_hierarchy_summary

        self.assertEqual(_render_hierarchy_summary(None), [])
        self.assertEqual(_render_hierarchy_summary("not-a-dict"), [])
        base = {
            "decision": "select_best",
            "selected_candidate": "alpha",
            "search": [],
            "held_out": {},
        }
        diff = diff_reports(
            dict(base, hierarchy={"available": False}),
            dict(base, hierarchy={"available": False}),
        )
        self.assertEqual(diff["hierarchy"]["available"], "neither")
        summary = render_summary(diff)
        self.assertNotIn("Hierarchy adherence drift", summary)

    def test_render_summary_drift_candidates_groups(self):
        """Test render summary drift candidates groups."""
        base = {
            "decision": "select_best",
            "selected_candidate": "alpha",
            "search": [],
            "held_out": {},
        }
        old = _hierarchy_payload()
        new = _hierarchy_payload()
        for pos in ("privilege_first", "privilege_last"):
            # Rate drift for alpha plus two extra drifting candidates
            # (six drift entries exercises the "> 5" truncation line).
            new["per_condition"][pos]["per_candidate"]["alpha"]["system_rate"] = 0.25
            for cand, old_rate in (("gamma", 0.9), ("delta", 0.8)):
                old["per_condition"][pos]["per_candidate"][cand] = {
                    "system_rate": old_rate,
                    "user_rate": 0.1,
                    "neither_rate": 0.0,
                    "pass": True,
                }
                new["per_condition"][pos]["per_candidate"][cand] = {
                    "system_rate": 0.1,
                    "user_rate": 0.9,
                    "neither_rate": 0.0,
                    "pass": False,
                }
            # Candidate removed on the new side; epsilon added on the new side.
            del new["per_condition"][pos]["per_candidate"]["beta"]
            new["per_condition"][pos]["per_candidate"]["epsilon"] = {
                "system_rate": 0.2,
                "user_rate": 0.8,
                "neither_rate": 0.0,
                "pass": False,
            }
        # Group means drift, including a None -> numeric transition.
        new["group_adherence"]["dedicated"]["mean_system_rate"] = 0.9
        new["group_adherence"]["unknown"]["mean_system_rate"] = 0.5
        summary = render_summary(diff_reports(dict(base, hierarchy=old), dict(base, hierarchy=new)))
        self.assertIn("Hierarchy adherence drift (6):", summary)
        self.assertIn("... and 1 more", summary)
        self.assertIn("Hierarchy candidates removed: beta", summary)
        self.assertIn("Hierarchy candidates added: epsilon", summary)
        self.assertIn("dedicated-frame mean system rate", summary)
        self.assertIn("(none) -> 50.0%", summary)

    def test_render_summary_one_sided_hierarchy(self):
        """Test render summary one sided hierarchy."""
        base = {
            "decision": "select_best",
            "selected_candidate": "alpha",
            "search": [],
            "held_out": {},
        }
        old = dict(base, hierarchy=_hierarchy_payload())
        new = dict(base, hierarchy={"available": False})
        summary = render_summary(diff_reports(old, new))
        self.assertIn("did not record the hierarchy probe", summary)

    def test_section_group_note_absent_without_numeric_rates(self):
        """Test section group note absent without numeric rates."""
        result = {"hierarchy": _hierarchy_payload()}
        groups = result["hierarchy"]["group_adherence"]
        groups["dedicated"]["mean_system_rate"] = None
        groups["folded"]["mean_system_rate"] = "high"
        html = _hierarchy_section(result)
        self.assertIn("Instruction-hierarchy adherence", html)
        self.assertNotIn("correlation readout", html)

    def test_section_skips_non_dict_group(self):
        """Test section skips non dict group."""
        result = {"hierarchy": _hierarchy_payload()}
        result["hierarchy"]["group_adherence"]["dedicated"] = "x"
        html = _hierarchy_section(result)
        self.assertIn("Instruction-hierarchy adherence", html)
        self.assertNotIn("correlation readout", html)


class HierarchyCliProbeTests(unittest.TestCase):
    """Hierarchy cli probe tests."""

    @staticmethod
    def _result():
        """Result."""
        return {"decision": "selected", "selected_candidate": "alpha"}

    @staticmethod
    def _call(result, backend):
        """Call."""
        from model_protocol_lab.cli import _run_hierarchy_probe

        _run_hierarchy_probe(
            backend,
            tuple(_format(c) for c in CANDIDATES),
            result,
            boundary=(7,),
            boundary_map=None,
            max_new_tokens=16,
            seed=0,
        )

    def test_probe_helper_happy_path(self):
        """Test probe helper happy path."""
        result = self._result()
        self._call(
            result,
            ScriptedHierarchyBackend({"alpha": "system", "beta": "system"}, _pairs()),
        )
        hierarchy = result["hierarchy"]
        self.assertTrue(hierarchy["available"])
        self.assertEqual(hierarchy["positions"], ["privilege_first", "privilege_last"])

    def test_probe_helper_fail_closed(self):
        """Test probe helper fail closed."""
        result = self._result()
        with mock.patch(
            "model_protocol_lab.hierarchy.validate_conflict_specs",
            side_effect=RuntimeError("boom"),
        ):
            self._call(result, ScriptedHierarchyBackend({}, _pairs()))
        self.assertEqual(
            result["hierarchy"],
            {"available": False, "detail": "hierarchy_unavailable"},
        )


if __name__ == "__main__":
    unittest.main()


class HierarchyGroupLabelSanitizationTests(unittest.TestCase):
    """Wave-3 red-team A F2: hostile group_adherence labels must not
    inject terminal escapes into the stderr summary.
    """

    def test_group_label_terminal_escapes_stripped(self):
        """Test group label terminal escapes stripped."""
        base = {
            "decision": "select_best",
            "selected_candidate": "alpha",
            "search": [],
            "held_out": {},
        }
        evil = "\x1b[2J\x1b[31mHIER\x1b[0m"
        old = _hierarchy_payload()
        new = _hierarchy_payload()
        new["group_adherence"][evil] = {"n_candidates": 1, "n_families": 1, "mean_system_rate": 0.9}
        old["group_adherence"][evil] = {"n_candidates": 1, "n_families": 1, "mean_system_rate": 0.1}
        text = render_summary(diff_reports(dict(base, hierarchy=old), dict(base, hierarchy=new)))
        self.assertNotIn("\x1b", text)
        self.assertIn("HIER", text)
