"""Cycle 150: synthetic-history selection stability (research wave-2 idea 5).

Does the winning protocol survive when fixtures are asked after synthetic
prior conversation turns -- is protocol selection robust to conversational
context, or does history flip the winner? Fixed benign synthetic turns are
prepended to every fixture (one and three exchanges); the selection rule is
re-applied per history condition.

Covers: history-fixture construction and validation, the mechanical
delimiter-explicitness labels (pinned), ``synthhistory_stage`` (scripted
backend, fail-closed cells), ``synthhistory_stability`` (selection
re-application, flips, group means, malformed-input degradation),
``diff_synthhistory`` (availability, changed disjuncts, hostile payloads),
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

from model_protocol_lab import core, diffreports
from model_protocol_lab.catalog import CATALOG
from model_protocol_lab.core import Fixture, Format, Message
from model_protocol_lab.diffreports import diff_synthhistory, render_summary
from model_protocol_lab.experiments import synthhistory_stability, synthhistory_stage
from model_protocol_lab.reporting import _synthhistory_section
from model_protocol_lab.synthhistory import (
    CONDITIONS,
    DELIMITER_LABELS,
    SYNTHETIC_TURNS,
    TURN_COUNTS,
    build_history_fixtures,
    candidate_delimiter_label,
    delimiter_explicitness,
    family_delimiter_explicitness,
    prepend_history,
    validate_history_fixtures,
)

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


def _case(fixture_id: str, expected: str | None = None) -> Fixture:
    """Case."""
    return Fixture(
        id=fixture_id,
        messages=(Message(role="user", content=f"PROMPT-{fixture_id}"),),
        expected=f"ANS-{fixture_id}" if expected is None else expected,
        group="g",
    )


class ScriptedHistoryBackend:
    """Deterministic backend with per-cell history policy.

    ``policy`` maps (candidate, fixture_id) to "pass" (emit the
    fixture's expected answer), "fail" (emit an unrelated string), or
    "raise" (``generate`` raises, exercising the fail-closed cell path).
    The candidate is identified by its ``[name]`` user prefix; the
    fixture by its ``PROMPT-<id>`` marker.
    """

    fingerprint = "cycle150-synthhistory-scripted"

    def __init__(self, policy):
        self._policy = dict(policy)

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
        import re

        prompt = self.decode(input_ids)
        candidate = next((n for n in CANDIDATES if f"[{n}]" in prompt), None)
        match = re.search(r"PROMPT-([A-Za-z0-9_-]+)", prompt)
        fixture_id = match.group(1) if match else None
        if candidate is None or fixture_id is None:
            return core.Generation((), "error", "unknown")
        mode = self._policy.get((candidate, fixture_id), "pass")
        if mode == "raise":
            raise RuntimeError("generate boom")
        text = f"ANS-{fixture_id}" if mode == "pass" else "WRONG"
        ids = self.encode(text + chr(7))
        return core.Generation(ids + (1,) * (max_new_tokens - len(ids)), "limit")


def _run_stage(policy, condition="history_2", fixtures=None):
    """Run stage."""
    base = fixtures if fixtures is not None else (_case("f1"), _case("f2"))
    built, _k = build_history_fixtures(base, condition=condition)
    return synthhistory_stage(
        ScriptedHistoryBackend(policy),
        tuple(_format(c) for c in CANDIDATES),
        built,
        condition=condition,
        boundary=(7,),
        max_new_tokens=16,
        seed=0,
    )


class HistoryConstructionTests(unittest.TestCase):
    """History construction tests."""

    def test_conditions_and_turn_counts(self):
        """Test conditions and turn counts."""
        self.assertEqual(CONDITIONS, ("history_2", "history_6"))
        self.assertEqual(TURN_COUNTS, {"history_2": 2, "history_6": 6})

    def test_synthetic_turns_are_whole_exchanges(self):
        """Test synthetic turns are whole exchanges."""
        self.assertEqual(len(SYNTHETIC_TURNS) % 2, 0)
        self.assertGreater(len(SYNTHETIC_TURNS), 0)
        for i, turn in enumerate(SYNTHETIC_TURNS):
            self.assertEqual(turn.role, "user" if i % 2 == 0 else "assistant")

    def test_synthetic_turns_avoid_answer_vocabulary(self):
        """Test synthetic turns avoid answer vocabulary."""
        # The turns are pure context: they must not contain any of the
        # sample-fixture answer keywords, or scripted backends could
        # misattribute them.
        for turn in SYNTHETIC_TURNS:
            for keyword in ("BLUE", "GREEN", "RED", "NORTH", "SOUTH"):
                self.assertNotIn(keyword, turn.content)
            self.assertNotIn("numeral 7", turn.content)

    def test_build_prepends_and_namespaces(self):
        """Test build prepends and namespaces."""
        base = (_case("f1", "ANS-f1"),)
        built, k = build_history_fixtures(base, condition="history_6")
        self.assertEqual(k, 6)
        self.assertEqual(len(built), 1)
        fixture = built[0]
        self.assertEqual(fixture.id, "synthhistory/history_6/f1")
        self.assertEqual(fixture.group, "synthhistory-history_6")
        self.assertEqual(fixture.expected, "ANS-f1")
        self.assertEqual(len(fixture.messages), 7)
        self.assertEqual(fixture.messages[:6], SYNTHETIC_TURNS[:6])
        self.assertEqual(fixture.messages[6], base[0].messages[0])

    def test_build_history_2(self):
        """Test build history 2."""
        built, k = build_history_fixtures((_case("f1"),), condition="history_2")
        self.assertEqual(k, 2)
        self.assertEqual(len(built[0].messages), 3)

    def test_rejects_bad_condition(self):
        """Test rejects bad condition."""
        with self.assertRaises(ValueError):
            build_history_fixtures((_case("f1"),), condition="history_0")

    def test_rejects_non_tuple(self):
        """Test rejects non tuple."""
        with self.assertRaises(ValueError):
            build_history_fixtures([_case("f1")], condition="history_2")

    def test_rejects_empty(self):
        """Test rejects empty."""
        with self.assertRaises(ValueError):
            build_history_fixtures((), condition="history_2")

    def test_rejects_non_fixture(self):
        """Test rejects non fixture."""
        with self.assertRaises(ValueError):
            build_history_fixtures(("nope",), condition="history_2")

    def test_prepend_history_validation(self):
        """Test prepend history validation."""
        messages = (Message("user", "q"),)
        with self.assertRaises(ValueError):
            prepend_history(messages, 3)  # odd: partial exchange
        with self.assertRaises(ValueError):
            prepend_history(messages, 0)
        with self.assertRaises(ValueError):
            prepend_history(messages, len(SYNTHETIC_TURNS) + 2)
        with self.assertRaises(ValueError):
            prepend_history("nope", 2)

    def test_validate_history_fixtures(self):
        """Test validate history fixtures."""
        cases = (_case("f1"),)
        self.assertEqual(validate_history_fixtures(cases), cases)
        with self.assertRaises(ValueError):
            validate_history_fixtures(())
        with self.assertRaises(ValueError):
            validate_history_fixtures("nope")


class DelimiterLabelTests(unittest.TestCase):
    """Delimiter label tests."""

    # The labels are derived mechanically from the catalog's own Format
    # hypotheses (non-blank open AND close marker per turn type); pin the
    # exact split so any catalog change forces a conscious re-label.
    EXPECTED_EXPLICIT = sorted(
        [
            "chatml",
            "gemma",
            "gemma3",
            "gemma4",
            "hy4",
            "k2horizon",
            "kimi2",
            "kimi3",
            "kolibri",
            "laguna",
            "lfm25",
            "ling",
            "llama3",
            "llama4",
            "minimax3",
            "muse_glimmer",
            "muse_spark",
            "nanbeige42",
            "nemotron35",
            "phi4",
            "qwen3",
            "qwen38",
        ]
    )
    EXPECTED_LOOSE = sorted(
        [
            "deepseek",
            "deepseek41",
            "glm4",
            "glm45",
            "glm53",
            "mistral",
            "mistral_large",
        ]
    )

    def test_label_split_matches_catalog(self):
        """Test label split matches catalog."""
        explicit = sorted(f for f in CATALOG if family_delimiter_explicitness(f) == "explicit")
        loose = sorted(f for f in CATALOG if family_delimiter_explicitness(f) == "loose")
        self.assertEqual(explicit, self.EXPECTED_EXPLICIT)
        self.assertEqual(loose, self.EXPECTED_LOOSE)
        self.assertEqual(len(explicit) + len(loose), len(CATALOG))

    def test_labels_cover_all_values(self):
        """Test labels cover all values."""
        self.assertEqual(set(DELIMITER_LABELS), {"explicit", "loose", "unknown"})

    def test_structural_rule_spot_checks(self):
        """Test structural rule spot checks."""
        # chatml: open+close markers on both turn types.
        self.assertEqual(delimiter_explicitness(CATALOG["chatml"]), "explicit")
        # deepseek: user turn has no close marker.
        self.assertEqual(delimiter_explicitness(CATALOG["deepseek"]), "loose")
        # mistral: assistant turn has no open marker.
        self.assertEqual(delimiter_explicitness(CATALOG["mistral"]), "loose")
        # glm45: bare-newline close is not an unambiguous delimiter.
        self.assertEqual(delimiter_explicitness(CATALOG["glm45"]), "loose")

    def test_rejects_non_format(self):
        """Test rejects non format."""
        with self.assertRaises(ValueError):
            delimiter_explicitness("chatml")

    def test_unknown_family(self):
        """Test unknown family."""
        self.assertEqual(family_delimiter_explicitness("no-such-family"), "unknown")
        self.assertEqual(family_delimiter_explicitness(""), "unknown")
        self.assertEqual(family_delimiter_explicitness(None), "unknown")
        self.assertEqual(family_delimiter_explicitness(42), "unknown")

    def test_candidate_variant_uses_family(self):
        """Test candidate variant uses family."""
        self.assertEqual(candidate_delimiter_label("llama3/no-bos"), "explicit")
        self.assertEqual(candidate_delimiter_label("deepseek"), "loose")
        self.assertEqual(candidate_delimiter_label("no-such-family/x"), "unknown")
        with self.assertRaises(ValueError):
            candidate_delimiter_label("")
        with self.assertRaises(ValueError):
            candidate_delimiter_label(None)


class SynthhistoryStageTests(unittest.TestCase):
    """Synthhistory stage tests."""

    def test_all_pass(self):
        """Test all pass."""
        stage = _run_stage(
            {
                ("alpha", "f1"): "pass",
                ("alpha", "f2"): "pass",
                ("beta", "f1"): "pass",
                ("beta", "f2"): "pass",
            }
        )
        self.assertEqual(stage["condition"], "history_2")
        self.assertEqual(stage["n_turns"], 2)
        self.assertEqual(stage["n_fixtures"], 2)
        for candidate in CANDIDATES:
            cell = stage["per_candidate"][candidate]
            self.assertEqual(cell, {"n": 2, "passed": 2, "pass": True})

    def test_partial_pass(self):
        """Test partial pass."""
        stage = _run_stage(
            {
                ("alpha", "f1"): "pass",
                ("alpha", "f2"): "fail",
                ("beta", "f1"): "pass",
                ("beta", "f2"): "pass",
            }
        )
        self.assertEqual(stage["per_candidate"]["alpha"], {"n": 2, "passed": 1, "pass": False})
        self.assertEqual(stage["per_candidate"]["beta"], {"n": 2, "passed": 2, "pass": True})

    def test_failing_cell_is_fail_closed_not_skipped(self):
        """Test failing cell is fail closed not skipped."""
        stage = _run_stage(
            {
                ("alpha", "f1"): "raise",
                ("alpha", "f2"): "raise",
                ("beta", "f1"): "pass",
                ("beta", "f2"): "pass",
            }
        )
        self.assertIsNone(stage["per_candidate"]["alpha"])
        self.assertTrue(stage["per_candidate"]["beta"]["pass"])

    def test_history_6_condition(self):
        """Test history 6 condition."""
        stage = _run_stage(
            {
                ("alpha", "f1"): "pass",
                ("alpha", "f2"): "pass",
                ("beta", "f1"): "pass",
                ("beta", "f2"): "pass",
            },
            condition="history_6",
        )
        self.assertEqual(stage["n_turns"], 6)
        self.assertTrue(stage["per_candidate"]["alpha"]["pass"])

    def test_rejects_bad_condition(self):
        """Test rejects bad condition."""
        built, _ = build_history_fixtures((_case("f1"),), condition="history_2")
        with self.assertRaises(ValueError):
            synthhistory_stage(
                ScriptedHistoryBackend({}),
                (_format("alpha"),),
                built,
                condition="history_9",
                boundary=(7,),
            )

    def test_rejects_empty_fixtures(self):
        """Test rejects empty fixtures."""
        with self.assertRaises(ValueError):
            synthhistory_stage(
                ScriptedHistoryBackend({}),
                (_format("alpha"),),
                (),
                condition="history_2",
                boundary=(7,),
            )

    def test_rejects_bad_boundary(self):
        """Test rejects bad boundary."""
        built, _ = build_history_fixtures((_case("f1"),), condition="history_2")
        with self.assertRaises(ValueError):
            synthhistory_stage(
                ScriptedHistoryBackend({}),
                (_format("alpha"),),
                built,
                condition="history_2",
                boundary=(),
            )

    def test_rejects_no_backend(self):
        """Test rejects no backend."""
        built, _ = build_history_fixtures((_case("f1"),), condition="history_2")
        with self.assertRaises(ValueError):
            synthhistory_stage(
                None,
                (_format("alpha"),),
                built,
                condition="history_2",
                boundary=(7,),
            )


def _stage_payload(condition, cells, n_fixtures=2):
    """Stage payload."""
    return {
        "condition": condition,
        "n_turns": TURN_COUNTS[condition],
        "n_fixtures": n_fixtures,
        "per_candidate": cells,
    }


def _cell(passed, n=2):
    """Cell."""
    return {"n": n, "passed": passed, "pass": passed == n}


class SynthhistoryStabilityTests(unittest.TestCase):
    """Synthhistory stability tests."""

    @staticmethod
    def _stages(alpha_pass, beta_pass, condition="history_2"):
        """Stages."""
        return {
            condition: _stage_payload(
                condition,
                {"alpha": _cell(alpha_pass), "beta": _cell(beta_pass)},
            )
        }

    def test_unique_winner_selected(self):
        """Test unique winner selected."""
        stages = self._stages(2, 1)
        out = synthhistory_stability("alpha", "selected", CANDIDATES, stages)
        self.assertTrue(out["available"])
        cond = out["per_condition"]["history_2"]
        self.assertEqual((cond["decision"], cond["winner"]), ("selected", "alpha"))
        self.assertEqual(cond["n_evaluable"], 2)
        self.assertEqual(cond["per_candidate"]["alpha"], {"pass_rate": 1.0, "pass": True})
        self.assertEqual(cond["per_candidate"]["beta"], {"pass_rate": 0.5, "pass": False})

    def test_tie_is_ambiguous(self):
        """Test tie is ambiguous."""
        stages = self._stages(2, 2)
        out = synthhistory_stability("alpha", "selected", CANDIDATES, stages)
        cond = out["per_condition"]["history_2"]
        self.assertEqual((cond["decision"], cond["winner"]), ("ambiguous", None))

    def test_no_pass_is_no_candidate(self):
        """Test no pass is no candidate."""
        stages = self._stages(0, 1)
        out = synthhistory_stability("alpha", "selected", CANDIDATES, stages)
        cond = out["per_condition"]["history_2"]
        self.assertEqual((cond["decision"], cond["winner"]), ("no_candidate", None))

    def test_no_evaluable_is_unavailable(self):
        """Test no evaluable is unavailable."""
        stages = {"history_2": _stage_payload("history_2", {"alpha": None, "beta": None})}
        out = synthhistory_stability("alpha", "selected", CANDIDATES, stages)
        cond = out["per_condition"]["history_2"]
        self.assertEqual((cond["decision"], cond["winner"]), ("unavailable", None))
        self.assertEqual(cond["per_candidate"]["alpha"], None)

    def test_flip_detected(self):
        """Test flip detected."""
        stages = self._stages(2, 1)
        out = synthhistory_stability("beta", "selected", CANDIDATES, stages)
        self.assertEqual(len(out["flips"]), 1)
        flip = out["flips"][0]
        self.assertEqual(flip["condition"], "history_2")
        self.assertEqual((flip["decision_from"], flip["winner_from"]), ("selected", "beta"))
        self.assertEqual((flip["decision_to"], flip["winner_to"]), ("selected", "alpha"))

    def test_no_flip_when_same(self):
        """Test no flip when same."""
        stages = self._stages(2, 1)
        out = synthhistory_stability("alpha", "selected", CANDIDATES, stages)
        self.assertEqual(out["flips"], [])

    def test_unavailable_condition_excluded_from_flips(self):
        """Test unavailable condition excluded from flips."""
        stages = {
            "history_2": _stage_payload("history_2", {"alpha": None, "beta": None}),
            "history_6": _stage_payload("history_6", {"alpha": _cell(2), "beta": _cell(0)}),
        }
        out = synthhistory_stability("alpha", "selected", CANDIDATES, stages)
        self.assertEqual(out["flips"], [])
        self.assertEqual(out["conditions"], ["history_2", "history_6"])

    def test_group_pass_rates(self):
        """Test group pass rates."""
        # alpha is llama3 (explicit), beta is deepseek (loose).
        stages = {
            "history_2": _stage_payload("history_2", {"llama3": _cell(2), "deepseek": _cell(1)}),
            "history_6": _stage_payload("history_6", {"llama3": _cell(2), "deepseek": _cell(0)}),
        }
        out = synthhistory_stability("llama3", "selected", ("llama3", "deepseek"), stages)
        groups = out["group_pass_rates"]
        self.assertEqual(groups["explicit"]["n_candidates"], 1)
        self.assertAlmostEqual(groups["explicit"]["mean_pass_rate"], 1.0)
        self.assertAlmostEqual(groups["loose"]["mean_pass_rate"], 0.25)
        self.assertIsNone(groups["unknown"]["mean_pass_rate"])
        self.assertEqual(groups["unknown"]["n_candidates"], 0)

    def test_malformed_inputs_degrade(self):
        """Test malformed inputs degrade."""
        good = self._stages(2, 1)
        self.assertFalse(synthhistory_stability(None, "selected", (), good)["available"])
        self.assertFalse(
            synthhistory_stability("alpha", "selected", ("alpha", "alpha"), good)["available"]
        )
        self.assertFalse(synthhistory_stability("zzz", "selected", CANDIDATES, good)["available"])
        self.assertFalse(synthhistory_stability("alpha", "", CANDIDATES, good)["available"])
        self.assertFalse(synthhistory_stability("alpha", "selected", CANDIDATES, {})["available"])
        self.assertFalse(
            synthhistory_stability("alpha", "selected", CANDIDATES, {7: {}})["available"]
        )
        bad_stage = {"history_2": {"condition": "history_6"}}
        self.assertFalse(
            synthhistory_stability("alpha", "selected", CANDIDATES, bad_stage)["available"]
        )
        bad_entry = {
            "history_2": _stage_payload(
                "history_2", {"alpha": {"n": 2, "passed": -1, "pass": False}}
            )
        }
        self.assertFalse(
            synthhistory_stability("alpha", "selected", CANDIDATES, bad_entry)["available"]
        )
        mismatch = {
            "history_2": _stage_payload("history_2", {"alpha": _cell(2)}, n_fixtures=2),
            "history_6": _stage_payload("history_6", {"alpha": _cell(3, n=3)}, n_fixtures=3),
        }
        out = synthhistory_stability("alpha", "selected", CANDIDATES, mismatch)
        self.assertFalse(out["available"])
        self.assertEqual(out["detail"], "fixture_count_mismatch")

    # skipcq: PY-R1000 - Gate-pinned frozen function; splitting breaks its verified contract.
    def test_brute_force_oracle_differential(self):
        """Test brute force oracle differential."""
        # Independent reimplementation: selection re-application, flips,
        # and group means must match on randomized inputs.
        rng = random.Random(20261018)  # noqa: S311 -- deterministic test seed

        def oracle(primary_winner, primary_decision, candidates, stages):
            """Oracle."""
            per_condition = {}
            for condition in sorted(stages):
                cells = stages[condition]["per_candidate"]
                winners = [c for c in candidates if cells.get(c) is not None and cells[c]["pass"]]
                evaluable = [c for c in candidates if cells.get(c) is not None]
                if not evaluable:
                    decision, winner = "unavailable", None
                elif len(winners) == 1:
                    decision, winner = "selected", winners[0]
                else:
                    decision, winner = ("ambiguous", None) if winners else ("no_candidate", None)
                per_condition[condition] = (decision, winner)
            if primary_decision in ("compatible_on_fixture_set", "held_out_failed"):
                primary_verdict = ("selected", primary_winner)
            else:
                primary_verdict = (primary_decision, primary_winner)
            flips = [
                c
                for c, (d, w) in per_condition.items()
                if d != "unavailable" and (d, w) != primary_verdict
            ]
            return per_condition, flips

        families = ["llama3", "deepseek", "chatml", "mistral", "no-such-family"]
        for _ in range(200):
            n_c = rng.randint(1, 4)
            candidates = tuple(f"{rng.choice(families)}/{i}" for i in range(n_c))
            stages = {}
            for condition in CONDITIONS:
                cells = {}
                for cand in candidates:
                    roll = rng.random()
                    if roll < 0.15:
                        cells[cand] = None
                    else:
                        passed = rng.randint(0, 3)
                        cells[cand] = {"n": 3, "passed": passed, "pass": passed == 3}
                stages[condition] = _stage_payload(condition, cells, n_fixtures=3)
            primary_decision = rng.choice(
                [
                    "selected",
                    "ambiguous",
                    "no_candidate",
                    "compatible_on_fixture_set",
                    "held_out_failed",
                ]
            )
            primary_winner = (
                rng.choice(candidates)
                if primary_decision == "selected" and rng.random() < 0.7
                else None
            )
            expected_cond, expected_flips = oracle(
                primary_winner, primary_decision, candidates, stages
            )
            out = synthhistory_stability(primary_winner, primary_decision, candidates, stages)
            self.assertTrue(out["available"])
            for condition, (decision, winner) in expected_cond.items():
                got = out["per_condition"][condition]
                self.assertEqual((got["decision"], got["winner"]), (decision, winner))
            self.assertEqual(sorted(f["condition"] for f in out["flips"]), sorted(expected_flips))
            # Group means: recompute independently.
            for label in DELIMITER_LABELS:
                members = [c for c in candidates if candidate_delimiter_label(c) == label]
                rates = []
                for cand in members:
                    vals = []
                    for condition in CONDITIONS:
                        entry = stages[condition]["per_candidate"][cand]
                        if entry is not None:
                            vals.append(entry["passed"] / entry["n"])
                    if vals:
                        rates.append(sum(vals) / len(vals))
                group = out["group_pass_rates"][label]
                self.assertEqual(group["n_candidates"], len(rates))
                if rates:
                    self.assertAlmostEqual(group["mean_pass_rate"], sum(rates) / len(rates))
                else:
                    self.assertIsNone(group["mean_pass_rate"])


class BoundarylessBackend:
    """Emits the exact expected answer with no boundary token.

    Output is exactly ``max_new_tokens`` tokens (a valid "limit"
    generation) whose decode equals the fixture's expected answer, but
    the boundary token never appears: exact_match is True while
    boundary_found is False. Pins the conservative ``and`` in the
    stage's pass predicate.
    """

    fingerprint = "cycle150-boundaryless"

    def __init__(self, expected):
        self._expected = expected

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
        assert len(self._expected) == max_new_tokens
        return core.Generation(tuple(map(ord, self._expected)), "limit")


class SynthhistoryMutationGateTests(unittest.TestCase):
    """Tempdir-module mutation gates (the real tree is never rewritten)."""

    def _run_gate(self, module_name, name, old_text, new_text, scenario):
        """Run gate."""
        mod = __import__(f"model_protocol_lab.{module_name}", fromlist=["x"])
        source = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertEqual(source.count(old_text), 1, f"mutant {name}: anchor not unique")
        mutated = source.replace(old_text, new_text)
        with tempfile.TemporaryDirectory(prefix="c150_mut_", dir="/tmp") as tmpdir:
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
    def _delimiter_scenario(mod):
        """Delimiter scenario."""
        assert mod.delimiter_explicitness(CATALOG["deepseek"]) == "loose", (
            "deepseek has no user close marker"
        )
        assert mod.delimiter_explicitness(CATALOG["chatml"]) == "explicit"

    def test_mutant_flipped_delimiter_label_caught(self):
        """Test mutant flipped delimiter label caught."""
        self._run_gate(
            "synthhistory",
            "flipped_delimiter",
            '        return "loose"\n    return "explicit"',
            '        return "explicit"  # MUTANT\n    return "explicit"',
            self._delimiter_scenario,
        )

    @staticmethod
    def _prepend_scenario(mod):
        """Prepend scenario."""
        built, _ = mod.build_history_fixtures((_case("f1"),), condition="history_2")
        assert built[0].messages[0] == SYNTHETIC_TURNS[0], (
            "synthetic turns must precede the fixture messages"
        )

    def test_mutant_appended_history_caught(self):
        """Test mutant appended history caught."""
        self._run_gate(
            "synthhistory",
            "appended_history",
            "    return SYNTHETIC_TURNS[:k] + messages",
            "    return messages + SYNTHETIC_TURNS[:k]  # MUTANT",
            self._prepend_scenario,
        )

    @staticmethod
    def _pass_predicate_scenario(mod):
        """Pass predicate scenario."""
        expected = "A" * 16
        # The experiments mutant reuses the real synthhistory module
        # (only experiments.py is rewritten), so build fixtures from the
        # real module here.
        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        from model_protocol_lab.synthhistory import build_history_fixtures as real_build

        real_built, _ = real_build((_case("f1", expected),), condition="history_2")
        stage = mod.synthhistory_stage(
            BoundarylessBackend(expected),
            (_format("alpha"),),
            real_built,
            condition="history_2",
            boundary=(7,),
            max_new_tokens=16,
            seed=0,
        )
        cell = stage["per_candidate"]["alpha"]
        assert cell == {"n": 1, "passed": 0, "pass": False}, cell

    def test_mutant_loosened_pass_predicate_caught(self):
        """Test mutant loosened pass predicate caught."""
        self._run_gate(
            "experiments",
            "loosened_pass",
            "        passed = sum(1 for obs in report.observations if obs.exact_match and obs.boundary_found)",  # noqa: E501
            "        passed = sum(1 for obs in report.observations if obs.exact_match or obs.boundary_found)  # MUTANT",  # noqa: E501
            self._pass_predicate_scenario,
        )

    @staticmethod
    def _flips_disjunct_scenario(mod):
        """Flips disjunct scenario."""
        base = _synthhistory_payload(flips=[])
        flipped = _synthhistory_payload(
            flips=[
                {
                    "condition": "history_2",
                    "decision_from": "selected",
                    "decision_to": "no_candidate",
                    "winner_from": "alpha",
                    "winner_to": None,
                }
            ]
        )
        assert mod.diff_synthhistory(base, base)["changed"] is False
        assert mod.diff_synthhistory(base, flipped)["changed"] is True, (
            "a flips-only difference must set changed"
        )

    def test_mutant_dropped_flips_disjunct_caught(self):
        """Test mutant dropped flips disjunct caught."""
        self._run_gate(
            "diffreports",
            "dropped_flips_disjunct",
            '        or any(entry["changed"] for entry in condition_changed.values())\n'
            "        or flips_changed\n"
            "        or pass_rate_drift",
            '        or any(entry["changed"] for entry in condition_changed.values())\n'
            "        or pass_rate_drift  # MUTANT: flips_changed dropped",
            self._flips_disjunct_scenario,
        )


def _synthhistory_payload(**overrides):
    """Synthhistory payload."""
    payload = {
        "available": True,
        "primary_winner": "alpha",
        "primary_decision": "selected",
        "n_fixtures": 2,
        "conditions": ["history_2", "history_6"],
        "per_condition": {
            "history_2": {
                "decision": "selected",
                "winner": "alpha",
                "n_evaluable": 2,
                "per_candidate": {
                    "alpha": {"pass_rate": 1.0, "pass": True},
                    "beta": {"pass_rate": 0.5, "pass": False},
                },
            },
            "history_6": {
                "decision": "no_candidate",
                "winner": None,
                "n_evaluable": 2,
                "per_candidate": {
                    "alpha": {"pass_rate": 0.0, "pass": False},
                    "beta": {"pass_rate": 0.5, "pass": False},
                },
            },
        },
        "flips": [
            {
                "condition": "history_6",
                "decision_from": "selected",
                "decision_to": "no_candidate",
                "winner_from": "alpha",
                "winner_to": None,
            }
        ],
        "group_pass_rates": {
            "explicit": {"n_candidates": 1, "mean_pass_rate": 0.5},
            "loose": {"n_candidates": 1, "mean_pass_rate": 0.5},
            "unknown": {"n_candidates": 0, "mean_pass_rate": None},
        },
    }
    payload.update(overrides)
    return payload


class DiffSynthhistoryTests(unittest.TestCase):
    """Diff synthhistory tests."""

    def test_identical_not_changed(self):
        """Test identical not changed."""
        payload = _synthhistory_payload()
        diff = diff_synthhistory(payload, _synthhistory_payload())
        self.assertEqual(diff["available"], "both")
        self.assertFalse(diff["changed"])

    def test_verdict_drift_changed(self):
        """Test verdict drift changed."""
        old = _synthhistory_payload()
        new = _synthhistory_payload()
        new["per_condition"]["history_2"]["winner"] = "beta"
        new["per_condition"]["history_2"]["decision"] = "selected"
        diff = diff_synthhistory(old, new)
        self.assertTrue(diff["changed"])
        entry = diff["condition_changed"]["history_2"]
        self.assertTrue(entry["changed"])
        self.assertEqual(entry["old_winner"], "alpha")
        self.assertEqual(entry["new_winner"], "beta")

    def test_flips_only_changed(self):
        """Test flips only changed."""
        old = _synthhistory_payload(flips=[])
        new = _synthhistory_payload()
        diff = diff_synthhistory(old, new)
        self.assertTrue(diff["changed"])
        self.assertTrue(diff["flips_changed"])

    def test_pass_rate_drift_only_changed(self):
        """Test pass rate drift only changed."""
        old = _synthhistory_payload()
        new = _synthhistory_payload()
        new["per_condition"]["history_2"]["per_candidate"]["beta"]["pass_rate"] = 1.0
        diff = diff_synthhistory(old, new)
        self.assertTrue(diff["changed"])
        self.assertEqual(len(diff["pass_rate_drift"]), 1)
        drift = diff["pass_rate_drift"][0]
        self.assertEqual(drift["candidate"], "beta")
        self.assertEqual(drift["condition"], "history_2")

    def test_group_change_only_changed(self):
        """Test group change only changed."""
        old = _synthhistory_payload()
        new = _synthhistory_payload()
        new["group_pass_rates"]["explicit"]["mean_pass_rate"] = 0.75
        diff = diff_synthhistory(old, new)
        self.assertTrue(diff["changed"])
        self.assertTrue(diff["group_rates"]["changed"])

    def test_conditions_added_removed(self):
        """Test conditions added removed."""
        old = _synthhistory_payload()
        new = _synthhistory_payload()
        new["conditions"] = ["history_2"]
        del new["per_condition"]["history_6"]
        diff = diff_synthhistory(old, new)
        self.assertTrue(diff["changed"])
        self.assertEqual(diff["conditions_removed"], ["history_6"])

    def test_candidates_added(self):
        """Test candidates added."""
        old = _synthhistory_payload()
        new = _synthhistory_payload()
        new["per_condition"]["history_2"]["per_candidate"]["gamma"] = {
            "pass_rate": 1.0,
            "pass": True,
        }
        diff = diff_synthhistory(old, new)
        self.assertTrue(diff["changed"])
        self.assertEqual(diff["candidates_added"], ["gamma"])

    def test_asymmetric_when_one_side_missing(self):
        """Test asymmetric when one side missing."""
        diff = diff_synthhistory(_synthhistory_payload(), {"available": False})
        self.assertEqual(diff["available"], "old")
        self.assertFalse(diff["changed"])
        diff = diff_synthhistory({"available": False}, {"available": False})
        self.assertEqual(diff["available"], "neither")

    def test_hostile_payloads_degrade_not_raise(self):
        """Test hostile payloads degrade not raise."""
        hostile = _synthhistory_payload()
        hostile["per_condition"]["history_2"]["per_candidate"] = {
            "a": {"pass_rate": [1], "pass": "yes"}
        }
        diff = diff_synthhistory(hostile, _synthhistory_payload())
        self.assertEqual(diff["available"], "new")
        hostile2 = _synthhistory_payload()
        hostile2["per_condition"]["history_2"]["decision"] = ["selected"]
        diff = diff_synthhistory(hostile2, _synthhistory_payload())
        self.assertEqual(diff["available"], "new")
        hostile3 = _synthhistory_payload()
        hostile3["per_condition"] = {7: {}}
        diff = diff_synthhistory(hostile3, _synthhistory_payload())
        self.assertEqual(diff["available"], "new")

    def test_summary_flags_and_stderr(self):
        """Test summary flags and stderr."""
        old_report = {
            "decision": "selected",
            "selected_candidate": "alpha",
            "synthhistory": _synthhistory_payload(flips=[]),
        }
        new_report = {
            "decision": "selected",
            "selected_candidate": "alpha",
            "synthhistory": _synthhistory_payload(),
        }
        diff = diffreports.diff_reports(old_report, new_report)
        summary = diff["summary"]
        self.assertTrue(summary["synthhistory_changed"])
        self.assertFalse(summary["synthhistory_asymmetric"])
        self.assertFalse(summary["identical"])
        text = render_summary(diff)
        self.assertIn("flip pattern changed", text)

    def test_summary_identical_when_both_missing(self):
        """Test summary identical when both missing."""
        old_report = {"decision": "selected", "selected_candidate": "alpha"}
        new_report = {"decision": "selected", "selected_candidate": "alpha"}
        summary = diffreports.diff_reports(old_report, new_report)["summary"]
        self.assertFalse(summary["synthhistory_changed"])
        self.assertFalse(summary["synthhistory_asymmetric"])

    def test_summary_renders_asymmetric_unavailable(self):
        """Test summary renders asymmetric unavailable."""
        diff = diffreports.diff_reports(
            {
                "decision": "selected",
                "selected_candidate": "alpha",
                "synthhistory": _synthhistory_payload(),
            },
            {"decision": "selected", "selected_candidate": "alpha"},
        )
        text = render_summary(diff)
        self.assertIn("did not record the synthhistory experiment", text)

    def test_summary_renders_removed_and_drift(self):
        """Test summary renders removed and drift."""
        old_payload = _synthhistory_payload()
        new_payload = _synthhistory_payload()
        # Drop a condition and a candidate on the new side; shift a rate.
        del new_payload["per_condition"]["history_6"]
        new_payload["conditions"] = ["history_2"]
        for cond in new_payload["per_condition"].values():
            del cond["per_candidate"]["beta"]
            cond["per_candidate"]["alpha"]["pass_rate"] = 0.0
        new_payload["group_pass_rates"]["explicit"]["mean_pass_rate"] = 0.0
        diff = diffreports.diff_reports(
            {
                "decision": "selected",
                "selected_candidate": "alpha",
                "synthhistory": old_payload,
            },
            {
                "decision": "selected",
                "selected_candidate": "alpha",
                "synthhistory": new_payload,
            },
        )
        text = render_summary(diff)
        self.assertIn("History conditions removed: history_6", text)
        self.assertIn("History candidates removed: beta", text)
        self.assertIn("History pass-rate drift", text)
        self.assertIn("mean pass rate", text)


class SynthhistoryHtmlTests(unittest.TestCase):
    """Synthhistory html tests."""

    def test_section_renders(self):
        """Test section renders."""
        html = _synthhistory_section({"synthhistory": _synthhistory_payload()})
        self.assertIn("Synthetic-history selection stability", html)
        self.assertIn("history_6", html)
        self.assertIn("differs", html)

    def test_section_empty_when_unavailable(self):
        """Test section empty when unavailable."""
        self.assertEqual(_synthhistory_section({}), "")
        self.assertEqual(_synthhistory_section({"synthhistory": {"available": False}}), "")

    def test_section_escapes_hostile_values(self):
        """Test section escapes hostile values."""
        payload = _synthhistory_payload()
        payload["per_condition"]["history_2"]["winner"] = "<script>alert(1)</script>"
        payload["per_condition"]["history_2"]["per_candidate"] = {
            "<img src=x onerror=y>": {"pass_rate": 1.0, "pass": True}
        }
        html = _synthhistory_section({"synthhistory": payload})
        self.assertNotIn("<script>alert(1)</script>", html)
        self.assertNotIn("<img src=x onerror=y>", html)
        self.assertIn("&lt;script&gt;", html)

    def test_stderr_strips_terminal_escapes(self):
        """Test stderr strips terminal escapes."""
        payload = _synthhistory_payload()
        payload["per_condition"]["history_2"]["winner"] = "\x1b[31malpha"
        text = render_summary(
            diffreports.diff_reports(
                {
                    "decision": "selected",
                    "selected_candidate": "alpha",
                    "synthhistory": _synthhistory_payload(flips=[]),
                },
                {
                    "decision": "selected",
                    "selected_candidate": "alpha",
                    "synthhistory": payload,
                },
            )
        )
        self.assertNotIn("\x1b", text)


class SynthhistoryCliTests(unittest.TestCase):
    """Synthhistory cli tests."""

    def test_flag_registered(self):
        """Test flag registered."""
        from model_protocol_lab.cli import build_parser

        base = [
            "run",
            "--model",
            "m",
            "--dataset",
            "d",
            "--family",
            "llama3",
            "--boundary-ids",
            "1",
            "--output",
            "o",
        ]
        args = build_parser().parse_args(base + ["--synth-history"])
        self.assertTrue(args.synth_history)
        args = build_parser().parse_args(base)
        self.assertFalse(args.synth_history)


class SynthhistoryCoverageTests(unittest.TestCase):
    """Branch-coverage pins for the fail-closed/defensive paths."""

    @staticmethod
    def _built():
        """Built."""
        built, _ = build_history_fixtures((_case("f1"),), condition="history_2")
        return built

    def _stage_kwargs(self, **overrides):
        """Stage kwargs."""
        kwargs = {
            "backend": ScriptedHistoryBackend({}),
            "formats": (_format("alpha"),),
            "history_fixtures": self._built(),
            "condition": "history_2",
            "boundary": (7,),
            "max_new_tokens": 16,
            "seed": 0,
        }
        kwargs.update(overrides)
        return kwargs

    def test_stage_rejects_bad_formats(self):
        """Test stage rejects bad formats."""
        with self.assertRaises(ValueError):
            synthhistory_stage(**self._stage_kwargs(formats=["alpha"]))
        with self.assertRaises(ValueError):
            synthhistory_stage(**self._stage_kwargs(formats=()))

    def test_stage_rejects_non_fixture_history(self):
        """Test stage rejects non fixture history."""
        with self.assertRaises(ValueError):
            synthhistory_stage(**self._stage_kwargs(history_fixtures=("nope",)))

    def test_stage_rejects_bad_max_new_tokens(self):
        """Test stage rejects bad max new tokens."""
        with self.assertRaises(ValueError):
            synthhistory_stage(**self._stage_kwargs(max_new_tokens=0))
        with self.assertRaises(ValueError):
            synthhistory_stage(**self._stage_kwargs(max_new_tokens="16"))

    def test_stage_rejects_bad_seed(self):
        """Test stage rejects bad seed."""
        with self.assertRaises(ValueError):
            synthhistory_stage(**self._stage_kwargs(seed=-1))
        with self.assertRaises(ValueError):
            synthhistory_stage(**self._stage_kwargs(seed=2**64))

    def test_stage_backend_failure_degrades_cell(self):
        """Test stage backend failure degrades cell."""
        stage = _run_stage(
            {("alpha", "f1"): "raise", ("alpha", "f2"): "raise"},
        )
        self.assertIsNone(stage["per_candidate"]["alpha"])
        self.assertTrue(stage["per_candidate"]["beta"]["pass"])

    def test_entry_valid_branches(self):
        """Test entry valid branches."""
        from model_protocol_lab.experiments import _synthhistory_entry_valid

        self.assertTrue(_synthhistory_entry_valid(None, 2))
        self.assertFalse(_synthhistory_entry_valid("nope", 2))
        self.assertFalse(_synthhistory_entry_valid({"n": 3, "passed": 1, "pass": False}, 2))
        self.assertFalse(_synthhistory_entry_valid({"n": 2, "passed": 9, "pass": False}, 2))
        self.assertFalse(_synthhistory_entry_valid({"n": 2, "passed": 2, "pass": "yes"}, 2))
        self.assertTrue(_synthhistory_entry_valid({"n": 2, "passed": 2, "pass": True}, 2))

    @staticmethod
    def _bad_stages(stage):
        """Bad stages."""
        return synthhistory_stability("alpha", "selected", ("alpha",), stage)

    def test_stability_rejects_non_dict_stage(self):
        """Test stability rejects non dict stage."""
        out = self._bad_stages({"history_2": None})
        self.assertFalse(out["available"])
        self.assertEqual(out["detail"], "stage_history_2_invalid")

    def test_stability_rejects_condition_mismatch(self):
        """Test stability rejects condition mismatch."""
        bad = {
            "history_2": {
                "condition": "history_6",
                "n_fixtures": 1,
                "per_candidate": {},
            }
        }
        out = self._bad_stages(bad)
        self.assertFalse(out["available"])
        self.assertEqual(out["detail"], "stage_history_2_invalid")

    def test_stability_rejects_bad_count(self):
        """Test stability rejects bad count."""
        bad = {
            "history_2": {
                "condition": "history_2",
                "n_fixtures": -1,
                "per_candidate": {},
            }
        }
        out = self._bad_stages(bad)
        self.assertFalse(out["available"])

    def test_stability_rejects_non_dict_per_candidate(self):
        """Test stability rejects non dict per candidate."""
        bad = {
            "history_2": {
                "condition": "history_2",
                "n_fixtures": 1,
                "per_candidate": ["alpha"],
            }
        }
        out = self._bad_stages(bad)
        self.assertFalse(out["available"])

    def test_stability_rejects_bad_per_candidate(self):
        """Test stability rejects bad per candidate."""
        bad = {
            "history_2": {
                "condition": "history_2",
                "n_fixtures": 1,
                "per_candidate": {"alpha": "nope"},
            }
        }
        out = self._bad_stages(bad)
        self.assertFalse(out["available"])

    def test_section_empty_when_unavailable(self):
        """Test section empty when unavailable."""
        self.assertEqual(_synthhistory_section({"synthhistory": {"available": False}}), "")
        self.assertEqual(_synthhistory_section({}), "")
        self.assertEqual(
            _synthhistory_section({"synthhistory": {"available": True, "per_condition": "nope"}}),
            "",
        )

    def test_section_tolerates_malformed_verdicts(self):
        """Test section tolerates malformed verdicts."""
        payload = _synthhistory_payload()
        payload["per_condition"]["history_2"] = "nope"
        html = _synthhistory_section({"synthhistory": payload})
        self.assertIn("history_6", html)
        self.assertNotIn("history_2</span>", html)

    def test_section_tolerates_malformed_candidates(self):
        """Test section tolerates malformed candidates."""
        payload = _synthhistory_payload()
        cond = payload["per_condition"]["history_2"]
        cond["per_candidate"] = "nope"
        html = _synthhistory_section({"synthhistory": payload})
        self.assertIn("history_2", html)
        cond["per_candidate"] = {"alpha": "nope"}
        html = _synthhistory_section({"synthhistory": payload})
        self.assertIn("history_2", html)

    def test_section_tolerates_malformed_flips(self):
        """Test section tolerates malformed flips."""
        payload = _synthhistory_payload()
        payload["flips"] = ["nope", {"condition": "history_2"}]
        html = _synthhistory_section({"synthhistory": payload})
        self.assertIn("history_2", html)
        payload["flips"] = "nope"
        html = _synthhistory_section({"synthhistory": payload})
        self.assertIn("Synthetic-history selection stability", html)

    def test_section_tolerates_malformed_groups(self):
        """Test section tolerates malformed groups."""
        payload = _synthhistory_payload()
        payload["group_pass_rates"] = {"explicit": "nope"}
        html = _synthhistory_section({"synthhistory": payload})
        # Malformed group is skipped, the rest of the section still renders.
        self.assertIn("Synthetic-history selection stability", html)
        self.assertNotIn("<td>explicit</td>", html)
        payload["group_pass_rates"] = "nope"
        html = _synthhistory_section({"synthhistory": payload})
        self.assertIn("Synthetic-history selection stability", html)


class SynthhistoryValidatorTests(unittest.TestCase):
    """Hostile-payload validator tests (wave-2 C3 lesson)."""

    @staticmethod
    def _mutants():
        """Mutants."""
        good = _synthhistory_payload()
        cond = good["per_condition"]["history_2"]
        # skipcq: PY-W0070 - mutants are built from the base payload; appends are the battery
        mutants = []
        mutants.append(("not_a_dict", "nope"))
        mutants.append(("available_false", {**good, "available": False}))
        mutants.append(("bad_decision", {**good, "primary_decision": 7}))
        mutants.append(("bad_winner", {**good, "primary_winner": 7}))
        mutants.append(("bad_n_fixtures", {**good, "n_fixtures": 0}))
        mutants.append(("bad_conditions", {**good, "conditions": "history_2"}))
        mutants.append(("empty_conditions", {**good, "conditions": []}))
        mutants.append(("conditions_mismatch", {**good, "per_condition": {"history_2": cond}}))
        bad_cond = dict(cond, decision=7)
        mutants.append(
            (
                "bad_cond_decision",
                {
                    **good,
                    "per_condition": {
                        "history_2": bad_cond,
                        "history_6": good["per_condition"]["history_6"],
                    },
                },
            )
        )
        bad_cond2 = dict(cond, winner=7)
        mutants.append(
            (
                "bad_cond_winner",
                {
                    **good,
                    "per_condition": {
                        "history_2": bad_cond2,
                        "history_6": good["per_condition"]["history_6"],
                    },
                },
            )
        )
        bad_cond3 = dict(cond, n_evaluable=-1)
        mutants.append(
            (
                "bad_n_evaluable",
                {
                    **good,
                    "per_condition": {
                        "history_2": bad_cond3,
                        "history_6": good["per_condition"]["history_6"],
                    },
                },
            )
        )
        bad_cand = dict(cond["per_candidate"], alpha="nope")
        bad_cond4 = dict(cond, per_candidate=bad_cand)
        mutants.append(
            (
                "bad_candidate_entry",
                {
                    **good,
                    "per_condition": {
                        "history_2": bad_cond4,
                        "history_6": good["per_condition"]["history_6"],
                    },
                },
            )
        )
        bad_cand2 = dict(cond["per_candidate"], alpha={"pass_rate": "high", "pass": True})
        bad_cond5 = dict(cond, per_candidate=bad_cand2)
        mutants.append(
            (
                "bad_pass_rate",
                {
                    **good,
                    "per_condition": {
                        "history_2": bad_cond5,
                        "history_6": good["per_condition"]["history_6"],
                    },
                },
            )
        )
        bad_cand3 = dict(cond["per_candidate"], alpha={"pass_rate": 1.0, "pass": "yes"})
        bad_cond6 = dict(cond, per_candidate=bad_cand3)
        mutants.append(
            (
                "bad_pass_flag",
                {
                    **good,
                    "per_condition": {
                        "history_2": bad_cond6,
                        "history_6": good["per_condition"]["history_6"],
                    },
                },
            )
        )
        mutants.append(("bad_flips", {**good, "flips": "nope"}))
        mutants.append(("bad_flip_entry", {**good, "flips": ["nope"]}))
        mutants.append(("bad_groups", {**good, "group_pass_rates": "nope"}))
        mutants.append(("bad_group_entry", {**good, "group_pass_rates": {"explicit": "nope"}}))
        return mutants

    def test_hostile_payloads_degrade_to_asymmetric(self):
        """Test hostile payloads degrade to asymmetric."""
        good = _synthhistory_payload()
        for name, bad in self._mutants():
            with self.subTest(name=name):
                section = diff_synthhistory(bad, good)
                self.assertEqual(section["available"], "new", name)
                self.assertFalse(section["changed"], name)
                # Full-report level: the asymmetric flag feeds `identical`.
                full = diffreports.diff_reports(
                    {
                        "decision": "selected",
                        "selected_candidate": "alpha",
                        "synthhistory": bad,
                    },
                    {
                        "decision": "selected",
                        "selected_candidate": "alpha",
                        "synthhistory": good,
                    },
                )
                self.assertTrue(full["summary"]["synthhistory_asymmetric"], name)
                self.assertFalse(full["summary"]["synthhistory_changed"], name)
                self.assertFalse(full["summary"]["identical"], name)
                # Never raises; stderr rendering stays clean.
                text = render_summary(full)
                self.assertIn("did not record", text)


class SynthhistorySummaryRenderTests(unittest.TestCase):
    """Synthhistory summary render tests."""

    def test_summary_renders_conditions_added(self):
        """Test summary renders conditions added."""
        old_payload = _synthhistory_payload()
        del old_payload["per_condition"]["history_6"]
        old_payload["conditions"] = ["history_2"]
        diff = diffreports.diff_reports(
            {
                "decision": "selected",
                "selected_candidate": "alpha",
                "synthhistory": old_payload,
            },
            {
                "decision": "selected",
                "selected_candidate": "alpha",
                "synthhistory": _synthhistory_payload(),
            },
        )
        text = render_summary(diff)
        self.assertIn("History conditions added: history_6", text)

    def test_summary_truncates_long_drift(self):
        """Test summary truncates long drift."""
        from model_protocol_lab.diffreports import _render_synthhistory_summary

        drift = [
            {
                "candidate": f"c{i}",
                "condition": "history_2",
                "old_pass_rate": 1.0,
                "new_pass_rate": 0.0,
            }
            for i in range(8)
        ]
        lines = _render_synthhistory_summary(
            {
                "available": "both",
                "condition_changed": {},
                "conditions_added": [],
                "conditions_removed": [],
                "flips_changed": False,
                "pass_rate_drift": drift,
                "candidates_added": [],
                "candidates_removed": [],
                "group_rates": {"changed": False},
            }
        )
        text = "\n".join(lines)
        self.assertIn("... and 3 more", text)

    def test_summary_renders_nothing_for_non_dict(self):
        """Test summary renders nothing for non dict."""
        from model_protocol_lab.diffreports import _render_synthhistory_summary

        self.assertEqual(_render_synthhistory_summary(None), [])
        self.assertEqual(_render_synthhistory_summary("nope"), [])


class SynthhistoryRedteamRegressionTests(unittest.TestCase):
    """Regression tests for wave-3 red-team B findings."""

    def test_no_phantom_flip_when_winner_reselected(self):
        """Test no phantom flip when winner reselected."""
        # B1: the primary decision is the run-level verdict
        # ("compatible_on_fixture_set"); per-condition verdicts use the
        # selection vocabulary ("selected"). A condition that re-selects
        # the identical winner must NOT be recorded as a flip.
        stages = {
            "history_2": {
                "condition": "history_2",
                "n_fixtures": 2,
                "per_candidate": {
                    "llama3": {"n": 2, "passed": 2, "pass": True},
                    "mistral": {"n": 2, "passed": 1, "pass": False},
                },
            }
        }
        out = synthhistory_stability(
            "llama3", "compatible_on_fixture_set", ("llama3", "mistral"), stages
        )
        self.assertTrue(out["available"])
        self.assertEqual(out["per_condition"]["history_2"]["decision"], "selected")
        self.assertEqual(out["per_condition"]["history_2"]["winner"], "llama3")
        self.assertEqual(out["flips"], [])

    def test_genuine_flip_still_recorded(self):
        """Test genuine flip still recorded."""
        stages = {
            "history_6": {
                "condition": "history_6",
                "n_fixtures": 2,
                "per_candidate": {
                    "llama3": {"n": 2, "passed": 1, "pass": False},
                    "mistral": {"n": 2, "passed": 1, "pass": False},
                },
            }
        }
        out = synthhistory_stability(
            "llama3", "compatible_on_fixture_set", ("llama3", "mistral"), stages
        )
        self.assertEqual(len(out["flips"]), 1)
        flip = out["flips"][0]
        self.assertEqual(flip["condition"], "history_6")
        self.assertEqual(flip["decision_from"], "selected")
        self.assertEqual(flip["winner_from"], "llama3")
        self.assertEqual(flip["decision_to"], "no_candidate")
        self.assertIsNone(flip["winner_to"])

    def test_whitespace_candidates_rejected_not_raised(self):
        """Test whitespace candidates rejected not raised."""
        # B2: "never raises" contract -- blank names degrade to
        # invalid_candidates instead of raising via family_of.
        stages = {
            "history_2": {
                "condition": "history_2",
                "n_fixtures": 1,
                "per_candidate": {" ": {"n": 1, "passed": 1, "pass": True}},
            }
        }
        out = synthhistory_stability(" ", "selected", (" ",), stages)
        self.assertFalse(out["available"])
        self.assertEqual(out["detail"], "invalid_candidates")

    def test_zero_fixture_stage_rejected(self):
        """Test zero fixture stage rejected."""
        # B3: a stage with n_fixtures=0 cannot yield "selected" from zero
        # evidence; it degrades to invalid like the diff validator requires.
        stages = {
            "history_2": {
                "condition": "history_2",
                "n_fixtures": 0,
                "per_candidate": {"llama3": {"n": 0, "passed": 0, "pass": True}},
            }
        }
        out = synthhistory_stability("llama3", "compatible_on_fixture_set", ("llama3",), stages)
        self.assertFalse(out["available"])
        self.assertEqual(out["detail"], "stage_history_2_invalid")


class SynthhistoryGroupLabelSanitizationTests(unittest.TestCase):
    """Wave-3 red-team A F1: hostile group_pass_rates labels must not
    inject terminal escapes into the stderr summary.
    """

    @staticmethod
    def _evil_reports(label):
        """Evil reports."""

        def payload(rate):
            """Payload."""
            return {
                "available": True,
                "primary_decision": "selected",
                "primary_winner": "chatml",
                "n_fixtures": 4,
                "conditions": ["history_2"],
                "per_condition": {
                    "history_2": {
                        "decision": "selected",
                        "winner": "chatml",
                        "n_evaluable": 1,
                        "per_candidate": {"chatml": {"pass_rate": 1.0, "pass": True}},
                    }
                },
                "flips": [],
                "group_pass_rates": {
                    "explicit": {"n_candidates": 1, "mean_pass_rate": 1.0},
                    label: {"n_candidates": 1, "mean_pass_rate": rate},
                },
            }

        def results(rate):
            """Results."""
            return {
                "decision": "selected",
                "selected_candidate": "chatml",
                "search": [],
                "synthhistory": payload(rate),
            }

        return results(0.5), results(1.0)

    def test_group_label_terminal_escapes_stripped(self):
        """Test group label terminal escapes stripped."""
        old, new = self._evil_reports("\x1b[2J\x1b[31mPWNED\x1b[0m")
        text = render_summary(diffreports.diff_reports(old, new))
        self.assertNotIn("\x1b", text)
        self.assertIn("PWNED", text)  # visible text survives, escapes don't

    def test_group_label_bidi_controls_stripped(self):
        """Test group label bidi controls stripped."""
        old, new = self._evil_reports("safe-\u202e-EVIL")
        text = render_summary(diffreports.diff_reports(old, new))
        self.assertNotIn("\u202e", text)
