"""Cycle 155: stop-cause termination audit (research wave-3 idea 3).

Tests the pure core (stopcause.py): first-EOT positions, termination
cause classification with the documented stop-string-beats-EOS tie
rule, over-generation bleed arithmetic, cause-rate aggregation; the
experiments.stop_stage fail-closed contract; the CLI probe wiring;
the HTML section escaping; and the diffreports.diff_stop_cause
section. Follows the established per-cycle pattern: brute-force oracle
differentials, Hypothesis properties, tempdir-module mutation gates,
hostile-payload red-team.
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

from model_protocol_lab import diffreports
from model_protocol_lab.core import Fixture, Message
from model_protocol_lab.diffreports import diff_stop_cause
from model_protocol_lab.experiments import stop_stage
from model_protocol_lab.reporting import _stopcause_section
from model_protocol_lab.stopcause import (
    CAUSES,
    aggregate_cause_records,
    cause_rates,
    classify_cause,
    first_position,
    over_generated_tokens,
)

EOS_ID = 999
BOUNDARY = (124,)  # ord("|")


def _fixture(fid):
    return Fixture(
        id=fid,
        messages=(Message(role="user", content="Say the word."),),
        expected="WORD",
        group="g",
    )


def _obs(fixture_id, output_ids, status="completed", input_ids=(10, 20, 30)):
    return {
        "fixture": fixture_id,
        "status": status,
        "output_ids": list(output_ids),
        "input_ids": list(input_ids),
    }


def _entry(name, observations, boundary_ids=None):
    entry = {"format": {"name": name}, "observations": observations}
    if boundary_ids is not None:
        entry["boundary_token_ids"] = list(boundary_ids)
    return entry


def _result(entries, max_new_tokens=16):
    return {"max_new_tokens": max_new_tokens, "search": entries}


class ScriptedStopBackend:
    """ord-based codec; decode never fails."""

    fingerprint = "scripted-stop-v1"
    eos_token_id = EOS_ID

    @staticmethod
    def encode(text):
        return tuple(map(ord, text))

    @staticmethod
    def decode(token_ids):
        return "".join(map(chr, token_ids))

    def generate(self, input_ids, *, max_new_tokens, seed, cancel):
        raise AssertionError("stop_stage is post-hoc; generate must not run")


class UndecodableBackend(ScriptedStopBackend):
    @staticmethod
    def decode(token_ids):
        raise RuntimeError("no codec")


class FirstPositionTests(unittest.TestCase):
    def test_first_index(self):
        self.assertEqual(first_position((1, 5, 5, 9), 5), 1)

    def test_absent_returns_none(self):
        self.assertIsNone(first_position((1, 2, 3), 9))

    def test_empty_tuple(self):
        self.assertIsNone(first_position((), 9))

    def test_malformed_rejected(self):
        with self.assertRaises(ValueError):
            first_position([1, 2], 1)
        with self.assertRaises(ValueError):
            first_position((1, -2), 1)
        with self.assertRaises(ValueError):
            first_position((1, True), 1)
        with self.assertRaises(ValueError):
            first_position((1, 2), -1)
        with self.assertRaises(ValueError):
            first_position((1, 2), "1")


class ClassifyCauseTests(unittest.TestCase):
    def _kw(self, **over):
        base = {
            "output_len": 16,
            "max_new_tokens": 16,
            "eos_first": None,
            "stop_first": None,
            "eos_known": True,
            "stop_known": True,
        }
        base.update(over)
        return base

    def test_eos_earlier_wins(self):
        self.assertEqual(classify_cause(**self._kw(eos_first=2, stop_first=5)), "eos_id")

    def test_stop_earlier_wins(self):
        self.assertEqual(classify_cause(**self._kw(eos_first=7, stop_first=5)), "stop_string")

    def test_same_step_tie_goes_to_stop_string(self):
        # Documented precedence: stop string beats EOS in the same step
        # (vLLM port notes in backends.py).
        self.assertEqual(classify_cause(**self._kw(eos_first=4, stop_first=4)), "stop_string")

    def test_nothing_fired_cap_reached(self):
        self.assertEqual(classify_cause(**self._kw()), "max_length")

    def test_nothing_known_is_fail_closed(self):
        self.assertEqual(
            classify_cause(**self._kw(eos_known=False, stop_known=False)),
            "stop_unknown",
        )

    def test_eos_known_only(self):
        self.assertEqual(classify_cause(**self._kw(eos_first=3, stop_known=False)), "eos_id")
        self.assertEqual(classify_cause(**self._kw(stop_known=False)), "max_length")

    def test_stop_known_only(self):
        self.assertEqual(
            classify_cause(**self._kw(eos_known=False, stop_first=6)),
            "stop_string",
        )

    def test_short_output_no_fire_is_unknown(self):
        # Defensive: unreachable for completed lab observations (the
        # decoder always runs the cap), but never a phantom max_length.
        self.assertEqual(
            classify_cause(**self._kw(output_len=5, max_new_tokens=16)),
            "stop_unknown",
        )

    def test_malformed_rejected(self):
        with self.assertRaises(ValueError):
            classify_cause(**self._kw(output_len=-1))
        with self.assertRaises(ValueError):
            classify_cause(**self._kw(max_new_tokens=0))
        with self.assertRaises(ValueError):
            classify_cause(**self._kw(eos_first=16))
        with self.assertRaises(ValueError):
            classify_cause(**self._kw(stop_first=-1))
        with self.assertRaises(ValueError):
            classify_cause(**self._kw(eos_known="yes"))


class OverGeneratedTests(unittest.TestCase):
    def test_bleed_arithmetic(self):
        self.assertEqual(over_generated_tokens(16, 2), 13)
        self.assertEqual(over_generated_tokens(16, 15), 0)

    def test_no_eos_no_bleed(self):
        self.assertEqual(over_generated_tokens(16, None), 0)

    def test_malformed_rejected(self):
        with self.assertRaises(ValueError):
            over_generated_tokens(-1, None)
        with self.assertRaises(ValueError):
            over_generated_tokens(16, 16)
        with self.assertRaises(ValueError):
            over_generated_tokens(16, -1)


class CauseRatesTests(unittest.TestCase):
    def test_rates_sum_to_one(self):
        rates = cause_rates(["eos_id", "eos_id", "max_length", "stop_string"])
        self.assertAlmostEqual(sum(rates.values()), 1.0)
        self.assertEqual(rates["eos_id"], 0.5)
        self.assertEqual(rates["stop_unknown"], 0.0)

    def test_empty_is_all_zero(self):
        self.assertEqual(cause_rates([]), dict.fromkeys(CAUSES, 0.0))

    def test_unknown_cause_rejected(self):
        with self.assertRaises(ValueError):
            cause_rates(["eos_id", "mystery"])
        with self.assertRaises(ValueError):
            cause_rates("eos_id")


class AggregateTests(unittest.TestCase):
    def test_aggregation_math(self):
        records = [
            {"cause": "eos_id", "over_generated": 13},
            {"cause": "stop_string", "over_generated": 0},
            {"cause": "max_length", "over_generated": 0},
        ]
        agg = aggregate_cause_records(records)
        self.assertEqual(agg["n"], 3)
        self.assertEqual(agg["cause_counts"]["eos_id"], 1)
        self.assertAlmostEqual(agg["over_generation_rate"], 1 / 3)
        self.assertAlmostEqual(agg["would_stop_early_rate"], 2 / 3)
        self.assertAlmostEqual(agg["max_length_rate"], 1 / 3)

    def test_empty_aggregates(self):
        agg = aggregate_cause_records([])
        self.assertEqual(agg["n"], 0)
        self.assertEqual(agg["over_generation_rate"], 0.0)
        self.assertEqual(agg["would_stop_early_rate"], 0.0)

    def test_malformed_rejected(self):
        with self.assertRaises(ValueError):
            aggregate_cause_records("junk")
        with self.assertRaises(ValueError):
            aggregate_cause_records([{"cause": "eos_id"}])
        with self.assertRaises(ValueError):
            aggregate_cause_records([{"cause": "eos_id", "over_generated": -1}])
        with self.assertRaises(ValueError):
            aggregate_cause_records([{"cause": "nope", "over_generated": 0}])
        with self.assertRaises(ValueError):
            aggregate_cause_records(["not-a-dict"])


def _reference_classify(output_len, max_new_tokens, eos_first, stop_first, eos_known, stop_known):
    """Independent reimplementation of the classify_cause decision table."""
    if not eos_known and not stop_known:
        return "stop_unknown"
    candidates = []
    if eos_known and eos_first is not None:
        candidates.append((eos_first, "eos_id"))
    if stop_known and stop_first is not None:
        candidates.append((stop_first, "stop_string"))
    if candidates:
        best_step = min(step for step, _ in candidates)
        tied = [cause for step, cause in candidates if step == best_step]
        return "stop_string" if "stop_string" in tied else tied[0]
    if output_len >= max_new_tokens:
        return "max_length"
    return "stop_unknown"


class BruteForceOracleTests(unittest.TestCase):
    def test_classify_matches_reference_on_200_cases(self):
        rng = random.Random(0xC155)  # noqa: S311 -- deterministic test seed
        for _ in range(200):
            output_len = rng.randint(1, 32)
            max_new_tokens = rng.choice([8, 16, 32])
            eos_known = rng.random() < 0.7
            stop_known = rng.random() < 0.7
            eos_first = rng.randrange(output_len) if eos_known and rng.random() < 0.5 else None
            stop_first = rng.randrange(output_len) if stop_known and rng.random() < 0.5 else None
            got = classify_cause(
                output_len=output_len,
                max_new_tokens=max_new_tokens,
                eos_first=eos_first,
                stop_first=stop_first,
                eos_known=eos_known,
                stop_known=stop_known,
            )
            want = _reference_classify(
                output_len, max_new_tokens, eos_first, stop_first, eos_known, stop_known
            )
            self.assertEqual(got, want)
            self.assertIn(got, CAUSES)

    def test_first_position_matches_list_index(self):
        rng = random.Random(0xC155 + 1)  # noqa: S311 -- deterministic test seed
        for _ in range(200):
            ids = tuple(rng.randint(0, 20) for _ in range(rng.randint(0, 10)))
            token = rng.randint(0, 20)
            got = first_position(ids, token)
            want = ids.index(token) if token in ids else None
            self.assertEqual(got, want)


class StopcauseHypothesisTests(unittest.TestCase):
    @given(
        st.lists(st.sampled_from(CAUSES), min_size=1, max_size=20),
    )
    @settings(max_examples=100)
    def test_rates_always_sum_to_one(self, causes):
        self.assertAlmostEqual(sum(cause_rates(causes).values()), 1.0)

    @given(
        output_len=st.integers(min_value=1, max_value=64),
        cap=st.integers(min_value=1, max_value=64),
        eos_known=st.booleans(),
        stop_known=st.booleans(),
    )
    @settings(max_examples=200)
    def test_classify_never_leaves_vocabulary(self, output_len, cap, eos_known, stop_known):
        eos_first = 0 if eos_known else None
        stop_first = 0 if stop_known else None
        cause = classify_cause(
            output_len=output_len,
            max_new_tokens=cap,
            eos_first=eos_first if eos_first is not None and eos_first < output_len else None,
            stop_first=stop_first if stop_first is not None and stop_first < output_len else None,
            eos_known=eos_known,
            stop_known=stop_known,
        )
        self.assertIn(cause, CAUSES)

    @given(
        output_len=st.integers(min_value=1, max_value=64),
    )
    @settings(max_examples=100)
    def test_bleed_never_exceeds_output(self, output_len):
        for eos_first in (None, 0, output_len - 1):
            bleed = over_generated_tokens(output_len, eos_first)
            self.assertGreaterEqual(bleed, 0)
            self.assertLessEqual(bleed, output_len)


class StopStageTests(unittest.TestCase):
    def _run(self, entries, backend=None, eos_id=EOS_ID, **kw):
        backend = backend or ScriptedStopBackend()
        return stop_stage(
            _result(entries),
            (_fixture("f1"), _fixture("f2")),
            boundary=BOUNDARY,
            eos_id=eos_id,
            backend=backend,
            **kw,
        )

    def _ids(self, text):
        return tuple(map(ord, text))

    def test_eos_cause_and_bleed(self):
        # "AB" + EOT at index 2, then "C|": EOT fires before the
        # "|" stop text at index 4.
        out = self._ids("AB") + (EOS_ID,) + self._ids("C|") + self._ids(" " * 11)
        assert len(out) == 16
        payload = self._run(
            [_entry("alpha", [_obs("f1", out), _obs("f2", self._ids("XY|" + " " * 13))])]
        )
        pc = payload["per_candidate"]["alpha"]
        self.assertTrue(payload["available"])
        self.assertEqual(pc["cause_rates"]["eos_id"], 0.5)
        self.assertEqual(pc["cause_rates"]["stop_string"], 0.5)
        self.assertEqual(pc["over_generation_rate"], 0.5)
        self.assertEqual(pc["would_stop_early_rate"], 1.0)
        rec = pc["records"][0]
        self.assertEqual(rec["cause"], "eos_id")
        self.assertEqual(rec["eos_first"], 2)
        self.assertEqual(rec["over_generated"], 13)

    def test_max_length_cause(self):
        out = self._ids("Q" * 16)
        payload = self._run([_entry("alpha", [_obs("f1", out), _obs("f2", out)])])
        pc = payload["per_candidate"]["alpha"]
        self.assertEqual(pc["cause_rates"]["max_length"], 1.0)
        self.assertEqual(pc["max_length_rate"], 1.0)

    def test_stop_unknown_when_nothing_known(self):
        out = self._ids("Q" * 16)
        payload = self._run(
            [_entry("alpha", [_obs("f1", out), _obs("f2", out)])],
            backend=UndecodableBackend(),
            eos_id=None,
        )
        pc = payload["per_candidate"]["alpha"]
        self.assertEqual(pc["cause_rates"]["stop_unknown"], 1.0)
        self.assertFalse(pc["eos_known"])
        self.assertFalse(pc["stop_known"])

    def test_incomplete_observations_excluded_from_rates(self):
        good = self._ids("XY|" + " " * 13)
        payload = self._run([_entry("alpha", [_obs("f1", good), _obs("f2", (), status="failed")])])
        pc = payload["per_candidate"]["alpha"]
        self.assertEqual(pc["n_evaluable"], 1)
        self.assertEqual(pc["cause_rates"]["stop_string"], 1.0)
        self.assertFalse(pc["records"][1]["evaluable"])

    def test_misaligned_candidate_excluded(self):
        good = self._ids("XY|" + " " * 13)
        payload = self._run(
            [
                _entry("alpha", [_obs("f1", good), _obs("f2", good)]),
                _entry("beta", [_obs("f1", good)]),  # one observation short
            ]
        )
        self.assertIn("beta", payload["excluded_candidates"])
        self.assertNotIn("beta", payload["per_candidate"])

    def test_per_candidate_boundary_override(self):
        # Candidate-level boundary ids override the run default.
        out = self._ids("AB#DEF" + " " * 10)
        assert len(out) == 16
        payload = self._run(
            [_entry("alpha", [_obs("f1", out), _obs("f2", out)], boundary_ids=(ord("#"),))]
        )
        pc = payload["per_candidate"]["alpha"]
        self.assertEqual(pc["cause_rates"]["stop_string"], 1.0)
        self.assertEqual(pc["records"][0]["stop_first"], 2)

    def test_malformed_result_is_unavailable(self):
        backend = ScriptedStopBackend()
        cases = (_fixture("f1"), _fixture("f2"))
        self.assertEqual(
            stop_stage("junk", cases, boundary=BOUNDARY, backend=backend)["available"],
            False,
        )
        self.assertEqual(
            stop_stage({"search": []}, cases, boundary=BOUNDARY, backend=backend)["detail"],
            "no_search_entries",
        )
        self.assertEqual(
            stop_stage({"search": [{"nope": 1}]}, cases, boundary=BOUNDARY, backend=backend)[
                "detail"
            ],
            "no_evaluable_candidates",
        )

    def test_caller_contract_violations_raise(self):
        backend = ScriptedStopBackend()
        cases = (_fixture("f1"), _fixture("f2"))
        entries = [_entry("alpha", [_obs("f1", (1, 2)), _obs("f2", (1, 2))])]
        with self.assertRaises(ValueError):
            stop_stage(_result(entries), cases, boundary=BOUNDARY, backend=None)
        with self.assertRaises(ValueError):
            stop_stage(_result(entries), (), boundary=BOUNDARY, backend=backend)
        with self.assertRaises(ValueError):
            stop_stage(_result(entries), ("junk",), boundary=BOUNDARY, backend=backend)
        with self.assertRaises(ValueError):
            stop_stage(_result(entries), cases, boundary=(), backend=backend)
        with self.assertRaises(ValueError):
            stop_stage(_result(entries), cases, boundary=BOUNDARY, backend=backend, eos_id=-1)


def _stop_payload(**over):
    def _cand(eos_rate=0.5, stop_rate=0.5, bleed=0.5, early=1.0, maxlen=0.0):
        return {
            "n_observations": 2,
            "n_evaluable": 2,
            "eos_known": True,
            "stop_known": True,
            "cause_rates": {
                "eos_id": eos_rate,
                "stop_string": stop_rate,
                "max_length": maxlen,
                "stop_unknown": 0.0,
            },
            "over_generation_rate": bleed,
            "would_stop_early_rate": early,
            "max_length_rate": maxlen,
        }

    payload = {
        "available": True,
        "max_new_tokens": 16,
        "eos_known": True,
        "per_candidate": {"alpha": _cand(), "beta": _cand(eos_rate=0.0, stop_rate=1.0, bleed=0.0)},
        "excluded_candidates": [],
        "overall": {
            "cause_rates": {
                "eos_id": 0.25,
                "stop_string": 0.75,
                "max_length": 0.0,
                "stop_unknown": 0.0,
            }
        },
    }
    payload.update(over)
    return payload


class DiffStopCauseTests(unittest.TestCase):
    def test_identical_payloads_unchanged(self):
        payload = _stop_payload()
        diff = diff_stop_cause(payload, copy.deepcopy(payload))
        self.assertEqual(diff["available"], "both")
        self.assertFalse(diff["changed"])

    def test_cause_drift_only_still_changed(self):
        # Single-disjunct pin: only one cause rate moves.
        old = _stop_payload()
        new = _stop_payload()
        new["per_candidate"]["alpha"]["cause_rates"]["eos_id"] = 1.0
        new["per_candidate"]["alpha"]["cause_rates"]["stop_string"] = 0.0
        diff = diff_stop_cause(old, new)
        self.assertTrue(diff["changed"])
        self.assertEqual(len(diff["cause_drift"]), 2)
        self.assertEqual(diff["aggregate_drift"], [])

    def test_aggregate_drift_only_still_changed(self):
        old = _stop_payload()
        new = _stop_payload()
        new["per_candidate"]["alpha"]["over_generation_rate"] = 0.0
        diff = diff_stop_cause(old, new)
        self.assertTrue(diff["changed"])
        self.assertEqual(len(diff["aggregate_drift"]), 1)
        self.assertEqual(diff["aggregate_drift"][0]["metric"], "over_generation_rate")
        self.assertEqual(diff["cause_drift"], [])

    def test_candidates_added_only_still_changed(self):
        old = _stop_payload()
        new = _stop_payload()
        new["per_candidate"]["gamma"] = copy.deepcopy(new["per_candidate"]["alpha"])
        diff = diff_stop_cause(old, new)
        self.assertTrue(diff["changed"])
        self.assertEqual(diff["candidates_added"], ["gamma"])

    def test_candidates_removed_only_still_changed(self):
        old = _stop_payload()
        new = _stop_payload()
        del new["per_candidate"]["beta"]
        diff = diff_stop_cause(old, new)
        self.assertTrue(diff["changed"])
        self.assertEqual(diff["candidates_removed"], ["beta"])

    def test_asymmetric(self):
        payload = _stop_payload()
        self.assertEqual(diff_stop_cause(payload, None)["available"], "old")
        self.assertEqual(diff_stop_cause(None, payload)["available"], "new")
        self.assertEqual(diff_stop_cause(None, None)["available"], "neither")
        self.assertFalse(diff_stop_cause(payload, None)["changed"])

    def test_hostile_payloads_degrade_without_raising(self):
        good = _stop_payload()
        hostile = copy.deepcopy(good)
        hostile["per_candidate"][7] = {"cause_rates": {"eos_id": "x"}}
        diff = diff_stop_cause(good, hostile)
        self.assertEqual(diff["available"], "old")
        self.assertFalse(diff["changed"])
        # Renderer degrades to the one-sided unavailable line.
        lines = diffreports._render_stop_cause_summary(diff)
        self.assertTrue(any("did not record" in line for line in lines))
        self.assertEqual(diffreports._render_stop_cause_summary(None), [])
        self.assertEqual(diffreports._render_stop_cause_summary("junk"), [])

    def test_feeds_identical(self):
        old = _stop_payload()
        new = _stop_payload()
        new["per_candidate"]["alpha"]["cause_rates"]["eos_id"] = 1.0
        new["per_candidate"]["alpha"]["cause_rates"]["stop_string"] = 0.0
        base_old = {"decision": "select_best", "selected_candidate": "alpha", "search": []}
        base_new = copy.deepcopy(base_old)
        old_report = {**base_old, "stop_cause": old}
        new_report = {**base_new, "stop_cause": new}
        compared = diffreports.diff_reports(old_report, new_report)
        self.assertTrue(compared["summary"]["stop_cause_changed"])
        self.assertFalse(compared["summary"]["identical"])
        self.assertIn("stop_cause", compared)

    def test_identical_reports_stay_identical(self):
        payload = _stop_payload()
        base = {"decision": "select_best", "selected_candidate": "alpha", "search": []}
        old_report = {**base, "stop_cause": copy.deepcopy(payload)}
        new_report = {**base, "stop_cause": copy.deepcopy(payload)}
        compared = diffreports.diff_reports(old_report, new_report)
        self.assertFalse(compared["summary"]["stop_cause_changed"])
        self.assertFalse(compared["summary"]["stop_cause_asymmetric"])


class StopCauseMutationGateTests(unittest.TestCase):
    """Tempdir-module mutation gates (the real tree is never rewritten)."""

    def _run_gate(self, module_name, name, old_text, new_text, scenario):
        mod = __import__(f"model_protocol_lab.{module_name}", fromlist=["x"])
        source = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertEqual(source.count(old_text), 1, f"mutant {name}: anchor not unique")
        mutated = source.replace(old_text, new_text)
        with tempfile.TemporaryDirectory(prefix="c155_mut_", dir="/tmp") as tmpdir:
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

    def _tie_break_scenario(self, mod):
        # A same-step tie must classify stop_string (documented precedence).
        assert (
            mod.classify_cause(
                output_len=16,
                max_new_tokens=16,
                eos_first=4,
                stop_first=4,
                eos_known=True,
                stop_known=True,
            )
            == "stop_string"
        )

    def test_mutant_flipped_tie_break_caught(self):
        self._run_gate(
            "stopcause",
            "flipped_tie_break",
            'best = min(fired, key=lambda item: (item[0], 0 if item[1] == "stop_string" else 1))',
            'best = min(fired, key=lambda item: (item[0], 0 if item[1] == "eos_id" else 1))'
            "  # MUTANT",
            self._tie_break_scenario,
        )

    def _fail_closed_scenario(self, mod):
        assert (
            mod.classify_cause(
                output_len=16,
                max_new_tokens=16,
                eos_first=None,
                stop_first=None,
                eos_known=False,
                stop_known=False,
            )
            == "stop_unknown"
        )

    def test_mutant_dropped_fail_closed_caught(self):
        self._run_gate(
            "stopcause",
            "dropped_fail_closed",
            "    if not eos_known and not stop_known:\n        return STOP_UNKNOWN",
            '    if not eos_known and not stop_known:\n        return "max_length"  # MUTANT',
            self._fail_closed_scenario,
        )

    def _cause_drift_scenario(self, mod):
        old = _stop_payload()
        new = _stop_payload()
        new["per_candidate"]["alpha"]["cause_rates"]["eos_id"] = 1.0
        new["per_candidate"]["alpha"]["cause_rates"]["stop_string"] = 0.0
        assert mod.diff_stop_cause(old, new)["changed"] is True

    def test_mutant_dropped_cause_drift_caught(self):
        self._run_gate(
            "diffreports",
            "dropped_cause_drift",
            "        cause_drift or aggregate_drift or candidates_added or candidates_removed",
            "        aggregate_drift or candidates_added or candidates_removed  # MUTANT",
            self._cause_drift_scenario,
        )

    def _aggregate_drift_scenario(self, mod):
        old = _stop_payload()
        new = _stop_payload()
        new["per_candidate"]["alpha"]["over_generation_rate"] = 0.0
        assert mod.diff_stop_cause(old, new)["changed"] is True

    def test_mutant_dropped_aggregate_drift_caught(self):
        self._run_gate(
            "diffreports",
            "dropped_aggregate_drift",
            "        cause_drift or aggregate_drift or candidates_added or candidates_removed",
            "        cause_drift or candidates_added or candidates_removed  # MUTANT",
            self._aggregate_drift_scenario,
        )


class StopCauseSectionTests(unittest.TestCase):
    def test_renders_rates_and_bleed(self):
        html = _stopcause_section({"stop_cause": _stop_payload()})
        self.assertIn('id="stop-cause"', html)
        self.assertIn("alpha", html)
        self.assertIn("50%", html)

    def test_unavailable_renders_nothing(self):
        self.assertEqual(_stopcause_section({}), "")
        self.assertEqual(_stopcause_section({"stop_cause": {"available": False}}), "")
        self.assertEqual(_stopcause_section({"stop_cause": {"available": True}}), "")

    def test_hostile_candidate_name_escaped(self):
        payload = _stop_payload()
        payload["per_candidate"]["<script>alert(1)</script>"] = copy.deepcopy(
            payload["per_candidate"]["alpha"]
        )
        html = _stopcause_section({"stop_cause": payload})
        self.assertNotIn("<script>alert(1)</script>", html)
        self.assertIn("&lt;script&gt;", html)

    def test_renderer_constants_match_module(self):
        # The renderer's local cause vocabulary must track stopcause.CAUSES.
        from model_protocol_lab.reporting import _STOPCAUSE_CAUSES

        self.assertEqual(tuple(_STOPCAUSE_CAUSES), tuple(CAUSES))


class StopCauseSanitizationTests(unittest.TestCase):
    def test_stderr_summary_strips_terminal_escapes(self):
        old = _stop_payload()
        new = _stop_payload()
        new["per_candidate"]["\x1b[31malpha"] = copy.deepcopy(new["per_candidate"]["alpha"])
        new["per_candidate"]["\x1b[31malpha"]["cause_rates"]["eos_id"] = 1.0
        new["per_candidate"]["\x1b[31malpha"]["cause_rates"]["stop_string"] = 0.0
        diff = diff_stop_cause(old, new)
        lines = diffreports._render_stop_cause_summary(diff)
        blob = "\n".join(lines)
        self.assertNotIn("\x1b", blob)
        self.assertTrue(any("added" in line for line in lines))

    def test_summary_unavailable_sides(self):
        self.assertEqual(diffreports._render_stop_cause_summary({"available": "neither"}), [])
        lines = diffreports._render_stop_cause_summary({"available": "old"})
        self.assertTrue(any("did not record" in line for line in lines))


class StopCauseHypothesisDiffTests(unittest.TestCase):
    @given(
        st.dictionaries(
            st.text(min_size=1, max_size=8),
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            min_size=0,
            max_size=3,
        )
    )
    @settings(max_examples=100)
    def test_diff_never_raises_on_rate_maps(self, rates):
        old = _stop_payload()
        new = _stop_payload()
        new["per_candidate"]["alpha"]["cause_rates"] = dict(rates)
        diff = diff_stop_cause(old, new)
        self.assertIn(diff["available"], ("both", "old", "new", "neither"))
        lines = diffreports._render_stop_cause_summary(diff)
        self.assertIsInstance(lines, list)


class CliProbeTests(unittest.TestCase):
    def test_run_stop_cause_flag(self):
        from model_protocol_lab import cli

        self.assertTrue(hasattr(cli, "_run_stop_probe"))
        out = tuple(map(ord, "AB|" + " " * 13))
        assert len(out) == 16
        entries = [
            _entry("alpha", [_obs("f1", out), _obs("f2", out)]),
        ]
        result = {"decision": "selected", "search": entries, "max_new_tokens": 16}
        cli._run_stop_probe(
            ScriptedStopBackend(),
            result,
            (_fixture("f1"), _fixture("f2")),
            boundary=BOUNDARY,
            boundary_map=None,
            eos_id=EOS_ID,
            token_bytes=None,
        )
        payload = result["stop_cause"]
        self.assertTrue(payload["available"])
        self.assertEqual(payload["per_candidate"]["alpha"]["cause_rates"]["stop_string"], 1.0)

    def test_probe_never_raises(self):
        from model_protocol_lab import cli

        result = {"decision": "selected"}
        cli._run_stop_probe(
            None,
            result,
            (),
            boundary=None,
            boundary_map=None,
            eos_id=None,
        )
        self.assertEqual(result["stop_cause"]["detail"], "stop_unavailable")

    def test_flag_registered(self):
        from model_protocol_lab.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "run",
                "--stop-cause",
                "--model",
                "x",
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
        self.assertTrue(args.stop_cause)


class SampleReportTests(unittest.TestCase):
    def test_sample_stop_cause_exercise_deterministic(self):
        script = Path(__file__).resolve().parent.parent / "examples" / "make_sample_report.py"
        spec = importlib.util.spec_from_file_location("make_sample_report_c155", script)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory(prefix="c155_sample_") as first:
            result_a = module.build_sample_report(Path(first) / "regen")
        with tempfile.TemporaryDirectory(prefix="c155_sample_") as second:
            result_b = module.build_sample_report(Path(second) / "regen")
        json_a = json.dumps(result_a["stop_cause"], sort_keys=True)
        json_b = json.dumps(result_b["stop_cause"], sort_keys=True)
        self.assertEqual(json_a, json_b)
        # Genuine signal: the winner candidate emits the scripted EOT id
        # on the GREEN fixture; every other generation stops on the
        # boundary stop text.
        payload = result_a["stop_cause"]
        self.assertTrue(payload["available"])
        winner_rates = payload["per_candidate"]["llama3"]["cause_rates"]
        self.assertEqual(winner_rates["eos_id"], 0.5)
        self.assertEqual(winner_rates["stop_string"], 0.5)
        self.assertEqual(payload["per_candidate"]["llama3"]["over_generation_rate"], 0.5)
        for name, pc in payload["per_candidate"].items():
            if name != "llama3":
                self.assertEqual(pc["cause_rates"]["stop_string"], 1.0)

    def test_committed_sample_has_stop_cause(self):
        sample = Path(__file__).resolve().parent.parent / "examples" / "sample-report"
        results = json.loads((sample / "results.json").read_text(encoding="utf-8"))
        self.assertTrue(results["stop_cause"]["available"])
        html = (sample / "report.html").read_text(encoding="utf-8")
        self.assertIn('id="stop-cause"', html)
        self.assertIn('href="#stop-cause"', html)


class TocAnchorTests(unittest.TestCase):
    def test_stop_cause_anchor_resolves(self):
        sample = Path(__file__).resolve().parent.parent / "examples" / "sample-report"
        results = json.loads((sample / "results.json").read_text(encoding="utf-8"))
        from model_protocol_lab import reporting

        html = reporting.render_html(results)
        hrefs = set(re.findall(r'href="#([A-Za-z][\w-]*)"', html))
        ids = set(re.findall(r'id="([A-Za-z][\w-]*)"', html))
        self.assertIn("stop-cause", hrefs)
        self.assertIn("stop-cause", ids)
        self.assertEqual(sorted(hrefs - ids), [])


if __name__ == "__main__":
    unittest.main()
