"""Cycle 181: candidate aliasing audit (wave-7 research rank 2).

The design was red-teamed before code existed (wave-7 red-team A:
DO NOT BUILD AS SPECIFIED, 10 findings; red-team B upheld the
verdict, downgraded five severities, refuted one sub-claim about
the scorer-lens duplicate precedent, and amended six of the ten
build gates). These tests pin the corrected contract:

- A1': classes come from recorded ``input_ids`` only -- never the
  candidate fingerprint (which hashes the format's name), the
  format dict, or input text -- inside cohorts of identical
  completed fixture sets, over evaluable cells only.
- A2': within a measurement class, output digests and outcome
  fields agree per fixture; violations are determinism-suspect
  cells, and only outcome-suspect cells suppress the flag.
- A3': the alias-suspect flag fires exactly under its conditions,
  with withholding reasons in the fixed priority order.

Plus the profile's fail-closed joins, the stripped-artifact
contract, the validator's recompute battery, the diff, the HTML
section, the CLI dispatch (helper-level and through the real run
flow), a real ``run_experiment`` construction of the flag case,
and three tempdir mutation gates on the pure module's load-bearing
constructions (boundary in the measurement signature, the
nonempty evaluable predicate, boundary in the redundant count).
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from model_protocol_lab import aliasing, reporting
from model_protocol_lab import cli as cli_module
from model_protocol_lab.core import Format
from model_protocol_lab.diffreports import _valid_aliasing, diff_aliasing
from model_protocol_lab.experiments import aliasing_profile, candidate_fingerprint


def _obs(
    candidate: str,
    fixture: str,
    input_ids: Any,
    output_ids: Any = (5, 99),
    *,
    status: str = "completed",
    exact: bool = True,
    boundary_found: bool = True,
    btm: Any = True,
) -> dict[str, Any]:
    return {
        "candidate": candidate,
        "fixture": fixture,
        "input_ids": input_ids,
        "output_ids": output_ids,
        "answer": "x",
        "exact_match": exact,
        "boundary_found": boundary_found,
        "boundary_text_match": btm,
        "raw_reason": "limit" if status == "completed" else None,
        "error": None,
        "status": status,
    }


def _entry(
    name: str,
    observations: list[dict[str, Any]],
    *,
    boundary: Any = (99,),
    eligible: bool = True,
    planned: int = 2,
) -> dict[str, Any]:
    return {
        "format": {"name": name},
        "boundary_token_ids": list(boundary),
        "summary": {"aggregate_eligible": eligible, "planned": planned},
        "observations": observations,
    }


def _result(entries: list[dict[str, Any]], *, decision: str = "ambiguous") -> dict[str, Any]:
    return {
        "decision": decision,
        "selected_candidate": None,
        "backend_fingerprint": "sha256:backend",
        "seed": 7,
        "max_new_tokens": 8,
        "search": entries,
    }


def _aliased_pair(**kw: Any) -> dict[str, Any]:
    """Two measurement-identical passing candidates + one distinct failure."""
    entries = [
        _entry("A", [_obs("A", "f1", [1, 2]), _obs("A", "f2", [3])]),
        _entry("B", [_obs("B", "f1", [1, 2]), _obs("B", "f2", [3])]),
        _entry(
            "C",
            [_obs("C", "f1", [9]), _obs("C", "f2", [8], exact=False)],
        ),
    ]
    return _result(entries, **kw)


class TestDigestAndPredicate(unittest.TestCase):
    def test_ids_digest_canonical_form(self):
        self.assertEqual(aliasing.ids_digest([1, 2]), hashlib.sha256(b"[1,2]").hexdigest())
        self.assertEqual(aliasing.ids_digest((1, 2)), aliasing.ids_digest([1, 2]))
        self.assertNotEqual(aliasing.ids_digest([1, 2]), aliasing.ids_digest([2, 1]))

    def test_evaluable_ids_predicate(self):
        self.assertTrue(aliasing.is_evaluable_ids([0]))
        self.assertTrue(aliasing.is_evaluable_ids((3, 4)))
        self.assertFalse(aliasing.is_evaluable_ids([]))  # the abort sentinel
        self.assertFalse(aliasing.is_evaluable_ids(()))
        self.assertFalse(aliasing.is_evaluable_ids([True]))  # bool is not a token id
        self.assertFalse(aliasing.is_evaluable_ids([-1]))
        self.assertFalse(aliasing.is_evaluable_ids([1.0]))
        self.assertFalse(aliasing.is_evaluable_ids("12"))
        self.assertFalse(aliasing.is_evaluable_ids(None))


class TestAnalyzeCandidate(unittest.TestCase):
    def _analyze(self, observations: Any, **kw: Any) -> dict[str, Any]:
        return aliasing.analyze_candidate(
            observations,
            candidate_name="A",
            boundary_token_ids=[99],
            eligible=True,
            planned=2,
            **kw,
        )

    def test_missing_and_invalid_observations(self):
        self.assertEqual(
            self._analyze(None), {"available": False, "detail": "observations_missing"}
        )
        self.assertEqual(self._analyze("nope")["detail"], "observations_not_a_list")

    def test_boundary_invalid(self):
        out = aliasing.analyze_candidate(
            [],
            candidate_name="A",
            boundary_token_ids=[],
            eligible=True,
            planned=2,
        )
        self.assertEqual(out["detail"], "boundary_invalid")

    def test_duplicate_fixture_fails_closed(self):
        out = self._analyze([_obs("A", "f1", [1]), _obs("A", "f1", [2])])
        self.assertEqual(out["detail"], "duplicate_observation")

    def test_candidate_field_mismatch_fails_closed(self):
        out = self._analyze([_obs("B", "f1", [1])])
        self.assertEqual(out["detail"], "observation_candidate_mismatch")
        obs = _obs("A", "f1", [1])
        del obs["candidate"]
        self.assertEqual(self._analyze([obs])["detail"], "observation_candidate_mismatch")

    def test_malformed_observation(self):
        self.assertEqual(self._analyze([42])["detail"], "observation_malformed")
        obs = _obs("A", "f1", [1])
        obs["exact_match"] = "yes"
        self.assertEqual(self._analyze([obs])["detail"], "observation_malformed")
        obs = _obs("A", "f1", [1])
        obs["boundary_text_match"] = "yes"
        self.assertEqual(self._analyze([obs])["detail"], "observation_malformed")

    def test_completed_empty_input_is_recorded_ids_invalid(self):
        # The abort sentinel on a claimed-completed cell is an
        # artifact defect, never a class member (red-team A F3).
        out = self._analyze([_obs("A", "f1", ())])
        self.assertEqual(out["detail"], "recorded_ids_invalid")
        out = self._analyze([_obs("A", "f1", [1], output_ids="bad")])
        self.assertEqual(out["detail"], "recorded_ids_invalid")

    def test_non_completed_cells_carry_no_digests(self):
        out = self._analyze([_obs("A", "f1", [1]), _obs("A", "f2", (), status="failed")])
        self.assertTrue(out["available"])
        self.assertEqual(out["n_completed"], 1)
        self.assertEqual(out["n_not_completed"], 1)
        failed = out["records"][1]
        self.assertFalse(failed["evaluable"])
        self.assertIsNone(failed["input_sha256"])  # never a digest of []
        self.assertIsNone(failed["output_sha256"])

    def test_evaluable_record_digests(self):
        out = self._analyze([_obs("A", "f1", [1, 2], (5, 99))])
        record = out["records"][0]
        self.assertTrue(record["evaluable"])
        self.assertEqual(record["input_sha256"], aliasing.ids_digest([1, 2]))
        self.assertEqual(record["output_sha256"], aliasing.ids_digest([5, 99]))
        self.assertEqual(out["planned"], 2)
        self.assertEqual(out["boundary_token_ids"], [99])


class TestFingerprintTrap(unittest.TestCase):
    """Gate 4: classes ignore the fingerprint and the format dict."""

    def _fmt(self, name: str, **over: Any) -> Format:
        base: dict[str, Any] = {
            "name": name,
            "bos": "",
            "user_prefix": "U:",
            "assistant_prefix": "A:",
            "turn_end": "\n",
        }
        base.update(over)
        return Format(**base)

    def test_name_only_pair_aliased_despite_fingerprints(self):
        # Executed by both red teams: A/B differ only in name (and
        # repeat_bos under an empty bos), render byte-identically,
        # and have different fingerprints -- fingerprint classes
        # would be singletons forever.
        fmt_a = self._fmt("alpha")
        fmt_b = self._fmt("beta", repeat_bos=True)
        self.assertNotEqual(candidate_fingerprint(fmt_a), candidate_fingerprint(fmt_b))
        payload = aliasing_profile(_aliased_pair())
        self.assertEqual(payload["measurement_classes"], [["A", "B"], ["C"]])

    def test_turn_end_pair_aliased_on_single_turn_fixtures(self):
        fmt_d = self._fmt("delta", assistant_turn_end="XXX")
        fmt_e = self._fmt("epsilon", assistant_turn_end="YYY")
        self.assertNotEqual(candidate_fingerprint(fmt_d), candidate_fingerprint(fmt_e))
        result = _result(
            [
                _entry("delta", [_obs("delta", "f1", [1]), _obs("delta", "f2", [2])]),
                _entry("epsilon", [_obs("epsilon", "f1", [1]), _obs("epsilon", "f2", [2])]),
            ],
            decision="ambiguous",
        )
        payload = aliasing_profile(result)
        self.assertEqual(payload["n_effective_measurement"], 1)


class TestProfileClasses(unittest.TestCase):
    def test_global_alias_counts_and_pairs(self):
        payload = aliasing_profile(_aliased_pair())
        self.assertTrue(payload["available"])
        self.assertEqual(payload["n_candidates_recorded"], 3)
        self.assertEqual(payload["n_primary_cohort"], 3)
        self.assertEqual(payload["n_effective_render"], 2)
        self.assertEqual(payload["n_effective_measurement"], 2)
        self.assertEqual(payload["n_redundant_completed_observations"], 2)
        self.assertEqual(payload["n_redundant_render_observations"], 2)
        pair = next(p for p in payload["pairs"] if p["candidates"] == ["A", "B"])
        self.assertEqual(pair["n_common_completed"], 2)
        self.assertEqual(pair["n_shared"], 2)
        self.assertTrue(_valid_aliasing(payload))

    def test_boundary_refines_render_into_measurements(self):
        result = _aliased_pair()
        result["search"][1]["boundary_token_ids"] = [7]
        payload = aliasing_profile(result)
        self.assertEqual(payload["render_classes"], [["A", "B"], ["C"]])
        self.assertEqual(payload["n_effective_render"], 2)
        self.assertEqual(payload["n_effective_measurement"], 3)
        # Boundary-distinct work is not redundant as a measurement.
        self.assertEqual(payload["n_redundant_completed_observations"], 0)
        self.assertEqual(payload["n_redundant_render_observations"], 2)

    def test_partial_aliasing_visible_only_in_pairs(self):
        result = _result(
            [
                _entry("A", [_obs("A", "f1", [1]), _obs("A", "f2", [2])]),
                _entry("B", [_obs("B", "f1", [1]), _obs("B", "f2", [9])]),
                _entry("C", [_obs("C", "f1", [7]), _obs("C", "f2", [9])]),
            ],
            decision="no_candidate",
        )
        payload = aliasing_profile(result)
        self.assertEqual(payload["n_effective_render"], 3)  # no global class > 1
        self.assertEqual(payload["n_redundant_render_observations"], 2)
        shared = {(tuple(p["candidates"]), p["n_shared"]) for p in payload["pairs"]}
        self.assertIn((("A", "B"), 1), shared)
        self.assertIn((("B", "C"), 1), shared)
        self.assertIn((("A", "C"), 0), shared)

    def test_cohort_discipline_interrupter_excluded(self):
        result = _result(
            [
                _entry("A", [_obs("A", "f1", [1]), _obs("A", "f2", [2])]),
                _entry("B", [_obs("B", "f1", [1]), _obs("B", "f2", [2])]),
                _entry(
                    "Z",
                    [_obs("Z", "f1", [1]), _obs("Z", "f2", (), status="failed")],
                    eligible=False,
                    planned=2,
                ),
            ],
            decision="failed",
        )
        payload = aliasing_profile(result)
        self.assertTrue(payload["evaluable_mismatch"])
        self.assertEqual(payload["n_primary_cohort"], 2)
        self.assertEqual(payload["excluded_candidates"], ["Z"])
        self.assertEqual(payload["n_effective_measurement"], 1)
        zed = next(c for c in payload["candidates"] if c["candidate"] == "Z")
        self.assertFalse(zed["in_primary_cohort"])
        self.assertIsNone(zed["measurement_class_members"])
        self.assertTrue(_valid_aliasing(payload))

    def test_duplicate_candidate_entry_unavailable(self):
        result = _aliased_pair()
        result["search"].append(_entry("A", [_obs("A", "f1", [1, 2]), _obs("A", "f2", [3])]))
        payload = aliasing_profile(result)
        a_records = [c for c in payload["candidates"] if c["candidate"] == "A"]
        self.assertEqual(len(a_records), 1)
        self.assertEqual(a_records[0]["detail"], "duplicate_candidate_entry")
        self.assertEqual(payload["alias_suspect_detail"], "candidate_unavailable")
        self.assertTrue(_valid_aliasing(payload))

    def test_malformed_entry_and_top_level_shapes(self):
        self.assertEqual(
            aliasing_profile("nope"), {"available": False, "detail": "result_not_a_dict"}
        )
        self.assertEqual(aliasing_profile({})["detail"], "no_search_entries")
        payload = aliasing_profile({"search": [42]})
        self.assertTrue(payload["available"])
        self.assertEqual(payload["candidates"][0]["detail"], "entry_malformed")
        self.assertIsNone(payload["candidates"][0]["candidate"])

    def test_stripped_artifact_contract(self):
        # Amended gate 2: top-level stripped only when EVERY entry
        # lacks the observations key (the determinism test).
        stripped = {
            "search": [
                {"format": {"name": "A"}, "summary": {}},
                {"format": {"name": "B"}, "summary": {}},
            ]
        }
        self.assertEqual(
            aliasing_profile(stripped),
            {"available": False, "detail": "observations_stripped"},
        )
        partial = _aliased_pair()
        del partial["search"][2]["observations"]
        payload = aliasing_profile(partial)
        self.assertTrue(payload["available"])  # profile precedent: stays available
        self.assertEqual(payload["candidates"][2]["detail"], "observations_missing")

    def test_pair_matrix_skips_disjoint_completed_sets(self):
        result = _result(
            [
                _entry(
                    "A",
                    [_obs("A", "f1", [1]), _obs("A", "f2", (), status="failed")],
                    eligible=False,
                ),
                _entry(
                    "B",
                    [_obs("B", "f1", (), status="failed"), _obs("B", "f2", [2])],
                    eligible=False,
                ),
            ],
            decision="failed",
        )
        payload = aliasing_profile(result)
        self.assertEqual(payload["pairs"], [])
        self.assertTrue(_valid_aliasing(payload))

    def test_run_configuration_carried(self):
        payload = aliasing_profile(_aliased_pair())
        self.assertEqual(payload["decision"], "ambiguous")
        self.assertIsNone(payload["selected_candidate"])
        self.assertEqual(payload["backend_fingerprint"], "sha256:backend")
        self.assertEqual(payload["seed"], 7)
        self.assertEqual(payload["max_new_tokens"], 8)
        self.assertEqual(
            payload["fixture_set_sha256"],
            hashlib.sha256(b"f1\nf2").hexdigest(),
        )


class TestA1BruteForceOracle(unittest.TestCase):
    """A1': profile classes equal a naive oracle over raw recorded ids."""

    def test_classes_match_naive_oracle(self):
        import random

        rng = random.Random(181)  # noqa: S311 -- deterministic test seed
        for _ in range(60):
            names = ["A", "B", "C", "D"][: rng.randint(2, 4)]
            pools = [[1], [1], [2, 3], [4]]
            entries = []
            for name in names:
                ids = [rng.choice(pools) for _ in range(3)]
                entries.append(
                    _entry(
                        name,
                        [_obs(name, f"f{i}", ids[i]) for i in range(3)],
                        boundary=(99,) if rng.random() < 0.7 else (7,),
                        planned=3,
                    )
                )
            payload = aliasing_profile(_result(entries, decision="no_candidate"))
            self.assertTrue(_valid_aliasing(payload))
            # Naive oracle: group by (ids per fixture, boundary).
            groups: dict[Any, list[str]] = {}
            render: dict[Any, list[str]] = {}
            for entry in entries:
                key = tuple(tuple(o["input_ids"]) for o in entry["observations"])
                render.setdefault(key, []).append(entry["format"]["name"])
                groups.setdefault(key + (tuple(entry["boundary_token_ids"]),), []).append(
                    entry["format"]["name"]
                )
            expected_m = sorted((sorted(v) for v in groups.values()), key=lambda ns: ns[0])
            expected_r = sorted((sorted(v) for v in render.values()), key=lambda ns: ns[0])
            self.assertEqual(payload["measurement_classes"], expected_m)
            self.assertEqual(payload["render_classes"], expected_r)


class TestSuspectCells(unittest.TestCase):
    def test_output_only_suspect_reported_flag_not_suppressed(self):
        # Amended gate 6: a post-boundary-only output divergence is
        # a suspect cell, but every selection-relevant quantity is
        # identical, so the flag still fires.
        result = _aliased_pair()
        result["search"][1]["observations"][0]["output_ids"] = (5, 99, 42)
        payload = aliasing_profile(result)
        self.assertEqual(
            payload["determinism_suspect_cells"],
            [{"fixture": "f1", "candidates": ["A", "B"], "kind": "output"}],
        )
        self.assertEqual(payload["n_outcome_suspect_cells"], 0)
        self.assertTrue(payload["alias_suspect_abstention"])

    def test_outcome_suspect_in_class_suppresses_flag(self):
        # Class {A, B, C} measurement-identical; C's outcome differs
        # on f2 -- the aliasing explanation of the A/B tie is exactly
        # what the suspect cell refutes.
        entries = [
            _entry("A", [_obs("A", "f1", [1]), _obs("A", "f2", [2])]),
            _entry("B", [_obs("B", "f1", [1]), _obs("B", "f2", [2])]),
            _entry("C", [_obs("C", "f1", [1]), _obs("C", "f2", [2], exact=False)]),
        ]
        payload = aliasing_profile(_result(entries))
        self.assertEqual(payload["n_outcome_suspect_cells"], 1)
        self.assertFalse(payload["alias_suspect_abstention"])
        self.assertEqual(payload["alias_suspect_detail"], "determinism_suspect_in_class")
        self.assertTrue(_valid_aliasing(payload))


class TestFlag(unittest.TestCase):
    def test_fires_on_contained_tie(self):
        payload = aliasing_profile(_aliased_pair())
        self.assertTrue(payload["alias_suspect_abstention"])
        self.assertEqual(payload["alias_suspect_detail"], "")
        self.assertEqual(payload["alias_suspect_class_members"], ["A", "B"])
        self.assertEqual(payload["passing_candidates"], ["A", "B"])

    def test_decision_not_ambiguous(self):
        payload = aliasing_profile(_aliased_pair(decision="no_candidate"))
        self.assertEqual(payload["alias_suspect_detail"], "decision_not_ambiguous")

    def test_passing_set_inconsistent(self):
        entries = [
            _entry("A", [_obs("A", "f1", [1]), _obs("A", "f2", [2])]),
            _entry("B", [_obs("B", "f1", [9]), _obs("B", "f2", [8], exact=False)]),
        ]
        payload = aliasing_profile(_result(entries))
        self.assertEqual(payload["alias_suspect_detail"], "passing_set_inconsistent")
        self.assertTrue(_valid_aliasing(payload))

    def test_passing_spans_classes_never_fires(self):
        # Two genuinely different prompts both passing: a real tie
        # between distinct measurements, not an artifact (A F8.3).
        entries = [
            _entry("A", [_obs("A", "f1", [1]), _obs("A", "f2", [2])]),
            _entry("B", [_obs("B", "f1", [9]), _obs("B", "f2", [8])]),
        ]
        payload = aliasing_profile(_result(entries))
        self.assertEqual(payload["alias_suspect_detail"], "passing_spans_classes")
        self.assertTrue(_valid_aliasing(payload))

    def test_boundaries_differ_never_fires(self):
        result = _aliased_pair()
        result["search"][1]["boundary_token_ids"] = [7]
        payload = aliasing_profile(result)
        self.assertEqual(payload["alias_suspect_detail"], "boundaries_differ_within_passing_set")

    def test_priority_unavailable_beats_decision(self):
        result = _aliased_pair(decision="no_candidate")
        del result["search"][2]["observations"]
        payload = aliasing_profile(result)
        self.assertEqual(payload["alias_suspect_detail"], "candidate_unavailable")

    def test_flag_never_rewrites_decision(self):
        result = _aliased_pair()
        payload = aliasing_profile(result)
        self.assertEqual(payload["decision"], "ambiguous")
        self.assertIsNone(payload["selected_candidate"])
        # The profile is a pure read: the result dict is untouched.
        self.assertEqual(result["decision"], "ambiguous")
        self.assertIsNone(result["selected_candidate"])


class TestValidatorBattery(unittest.TestCase):
    """One broken identity at a time: the recompute must catch all."""

    def setUp(self):
        self.payload = aliasing_profile(_aliased_pair())
        self.assertTrue(_valid_aliasing(self.payload))

    def _tamper(self, mutate) -> None:  # noqa: ANN001
        payload = copy.deepcopy(self.payload)
        mutate(payload)
        self.assertFalse(_valid_aliasing(payload))

    def test_tamper_counts(self):
        self._tamper(lambda p: p.update(n_effective_render=9))
        self._tamper(lambda p: p.update(n_effective_measurement=9))
        self._tamper(lambda p: p.update(n_redundant_completed_observations=0))
        self._tamper(lambda p: p.update(n_redundant_render_observations=0))
        self._tamper(lambda p: p.update(n_primary_cohort=2))
        self._tamper(lambda p: p.update(n_candidates_recorded=4))
        self._tamper(lambda p: p.update(n_candidates_available=2))
        self._tamper(lambda p: p.update(n_determinism_suspect_cells=1))
        self._tamper(lambda p: p.update(n_outcome_suspect_cells=1))

    def test_tamper_classes_and_membership(self):
        self._tamper(lambda p: p.update(render_classes=[["A"], ["B"], ["C"]]))
        self._tamper(lambda p: p.update(measurement_classes=[["A", "B", "C"]]))
        self._tamper(lambda p: p["candidates"][0].update(render_class_members=["A"]))
        self._tamper(lambda p: p["candidates"][0].update(in_primary_cohort=False))
        self._tamper(lambda p: p.update(excluded_candidates=["C"]))

    def test_tamper_flag(self):
        self._tamper(lambda p: p.update(alias_suspect_abstention=False))
        self._tamper(lambda p: p.update(alias_suspect_detail="decision_not_ambiguous"))
        self._tamper(lambda p: p.update(alias_suspect_class_members=["A"]))
        self._tamper(lambda p: p.update(passing_candidates=["A"]))

    def test_tamper_records(self):
        def flip_digest(p):
            p["candidates"][0]["records"][0]["input_sha256"] = "0" * 64

        self._tamper(flip_digest)

        def drop_digest(p):
            p["candidates"][0]["records"][0]["output_sha256"] = None

        self._tamper(drop_digest)

        def duplicate_fixture(p):
            p["candidates"][0]["records"].append(dict(p["candidates"][0]["records"][0]))
            p["candidates"][0]["n_observations"] += 1

        self._tamper(duplicate_fixture)

        def wrong_counts(p):
            p["candidates"][0]["n_completed"] = 1

        self._tamper(wrong_counts)

        def bad_boundary(p):
            p["candidates"][0]["boundary_token_ids"] = []

        self._tamper(bad_boundary)

        def completed_not_evaluable(p):
            p["candidates"][0]["records"][0]["evaluable"] = False

        self._tamper(completed_not_evaluable)

    def test_tamper_header(self):
        self._tamper(lambda p: p.update(fixture_set_sha256="0" * 64))
        self._tamper(lambda p: p.update(evaluable_mismatch=True))
        self._tamper(lambda p: p.update(seed=True))  # bool is not an int here
        self._tamper(lambda p: p.update(decision=7))
        self._tamper(lambda p: p["method"].update(protocol=""))
        self._tamper(lambda p: p.update(pairs=[]))

    def test_cell_bound_rejects_oversize_payload(self):
        # 25 available candidates x 100 fixtures = 2500 cells > 2400.
        entries = []
        for i in range(25):
            entries.append(
                _entry(
                    f"c{i}",
                    [_obs(f"c{i}", f"f{j}", [i]) for j in range(100)],
                    planned=100,
                )
            )
        payload = aliasing_profile(_result(entries, decision="no_candidate"))
        self.assertFalse(_valid_aliasing(payload))

    def test_foreign_digest_version_validates(self):
        # Amended gate 7(d): recompute is equality-only, so a
        # coherent foreign digest version is internally valid.
        payload = copy.deepcopy(self.payload)
        payload["method"]["token_ids_digest"] = "sha256-json-intlist-v99"
        self.assertTrue(_valid_aliasing(payload))


class TestDiff(unittest.TestCase):
    def setUp(self):
        self.payload = aliasing_profile(_aliased_pair())

    def test_self_diff_unchanged(self):
        diff = diff_aliasing(self.payload, copy.deepcopy(self.payload))
        self.assertEqual(diff["available"], "both")
        self.assertFalse(diff["changed"])
        self.assertTrue(diff["comparable"])

    def test_partition_split_and_count_drift(self):
        other = copy.deepcopy(self.payload)
        other_result = _aliased_pair()
        other_result["search"][1]["observations"][0]["input_ids"] = [42]
        other = aliasing_profile(other_result)
        diff = diff_aliasing(self.payload, other)
        self.assertEqual(diff["pairs_split"], [["A", "B"]])
        metrics = {d["metric"] for d in diff["count_drift"]}
        self.assertIn("n_effective_measurement", metrics)
        self.assertTrue(diff["changed"])

    def test_boundary_drift(self):
        other_result = _aliased_pair()
        other_result["search"][0]["boundary_token_ids"] = [5]
        other = aliasing_profile(other_result)
        diff = diff_aliasing(self.payload, other)
        self.assertEqual(len(diff["boundary_drift"]), 1)
        self.assertEqual(diff["boundary_drift"][0]["candidate"], "A")

    def test_backend_drift_not_comparable(self):
        other = copy.deepcopy(self.payload)
        other["backend_fingerprint"] = "sha256:different"
        diff = diff_aliasing(self.payload, other)
        self.assertTrue(diff["backend_drift"])
        self.assertFalse(diff["comparable"])
        self.assertTrue(diff["changed"])

    def test_protocol_drift_foreign_version(self):
        other = copy.deepcopy(self.payload)
        other["method"]["token_ids_digest"] = "sha256-json-intlist-v99"
        diff = diff_aliasing(self.payload, other)
        self.assertTrue(diff["protocol_drift"])
        self.assertFalse(diff["comparable"])

    def test_flag_drift(self):
        other = aliasing_profile(_aliased_pair(decision="no_candidate"))
        diff = diff_aliasing(self.payload, other)
        self.assertTrue(diff["flag_changed"])

    def test_asymmetric_and_neither(self):
        diff = diff_aliasing(self.payload, None)
        self.assertEqual(diff["available"], "old")
        self.assertFalse(diff["changed"])
        diff = diff_aliasing(None, None)
        self.assertEqual(diff["available"], "neither")
        hostile = copy.deepcopy(self.payload)
        hostile["n_effective_render"] = 9
        diff = diff_aliasing(hostile, self.payload)
        self.assertEqual(diff["available"], "new")


class TestReportingSection(unittest.TestCase):
    def test_section_renders_counts_classes_flag(self):
        payload = aliasing_profile(_aliased_pair())
        html = reporting.render_html({"decision": "ambiguous", "search": [], "aliasing": payload})
        self.assertIn('id="aliasing"', html)
        self.assertIn("Alias-suspect abstention", html)
        self.assertIn("Effective measurement candidates", html)
        digest = payload["candidates"][0]["records"][0]["input_sha256"]
        self.assertNotIn(digest, html)  # digests are never rendered

    def test_section_absent_without_payload(self):
        html = reporting.render_html({"decision": "failed", "search": []})
        self.assertNotIn('id="aliasing"', html)
        html = reporting.render_html(
            {
                "decision": "failed",
                "search": [],
                "aliasing": {"available": False, "detail": "observations_stripped"},
            }
        )
        self.assertNotIn('id="aliasing"', html)

    def test_suspect_cells_lead_and_names_escaped(self):
        entries = [
            _entry("A<x>", [_obs("A<x>", "f1", [1]), _obs("A<x>", "f2", [2])]),
            _entry("B", [_obs("B", "f1", [1]), _obs("B", "f2", [2], exact=False)]),
            _entry("C", [_obs("C", "f1", [1]), _obs("C", "f2", [2])]),
        ]
        # A<x> and C share ids with B on both fixtures; B's f2 outcome
        # differs -> all three in one measurement class with a suspect.
        payload = aliasing_profile(_result(entries))
        html = reporting.render_html({"decision": "ambiguous", "search": [], "aliasing": payload})
        self.assertIn("Determinism-suspect cells", html)
        self.assertIn("A&lt;x&gt;", html)
        self.assertNotIn("A<x>", html)


class TestCliProbe(unittest.TestCase):
    def test_helper_attaches_payload(self):
        result = _aliased_pair()
        cli_module._run_aliasing_probe(result)
        self.assertIn("aliasing", result)
        self.assertTrue(result["aliasing"]["available"])
        self.assertTrue(result["aliasing"]["alias_suspect_abstention"])

    def test_helper_never_raises(self):
        result: dict[str, Any] = {}
        import model_protocol_lab.experiments as experiments_mod

        with mock.patch.object(
            experiments_mod, "aliasing_profile", side_effect=RuntimeError("boom")
        ):
            cli_module._run_aliasing_probe(result)
        self.assertEqual(result["aliasing"], {"available": False, "detail": "aliasing_unavailable"})


class TestRunExperimentConstruction(unittest.TestCase):
    """The flag case built through the real run_experiment (red-team
    A's demand): two formats whose renders coincide on the fixture
    set, a deterministic stub backend, decision ambiguous."""

    def test_flag_fires_on_real_run(self):
        from model_protocol_lab.core import Fixture, Generation, Message, Split
        from model_protocol_lab.experiments import run_experiment

        class _StubBackend:
            fingerprint = "sha256:stub"
            eos_token_id = None

            def encode(self, text: str) -> tuple[int, ...]:
                # Deliberately lossy: every prompt encodes to the
                # same ids, so distinct formats alias by construction.
                return (1, 2, 3)

            def generate(self, input_ids, *, max_new_tokens, seed, cancel):  # noqa: ANN001, ANN202
                return Generation((5, 99, 5, 99)[:max_new_tokens], "limit")

            def decode(self, ids) -> str:  # noqa: ANN001, ANN202
                return "yo"

        fmt_a = Format(name="alpha", bos="", user_prefix="U:", assistant_prefix="A:", turn_end="\n")
        fmt_b = Format(
            name="beta",
            bos="",
            user_prefix="U:",
            assistant_prefix="A:",
            turn_end="\n",
            repeat_bos=True,
        )
        self.assertNotEqual(candidate_fingerprint(fmt_a), candidate_fingerprint(fmt_b))
        case = Fixture(
            id="s1",
            group="g1",
            messages=(Message(role="user", content="hi"),),
            expected="yo",
        )
        held = Fixture(
            id="h1",
            group="g2",
            messages=(Message(role="user", content="hey"),),
            expected="yo",
        )
        split = Split(search=(case,), held_out=(held,))
        result = run_experiment(
            _StubBackend(),
            (fmt_a, fmt_b),
            split,
            boundary=(99,),
            max_new_tokens=4,
            seed=3,
        )
        self.assertEqual(result["decision"], "ambiguous")
        payload = aliasing_profile(result)
        self.assertTrue(payload["alias_suspect_abstention"])
        self.assertEqual(payload["alias_suspect_class_members"], ["alpha", "beta"])
        self.assertEqual(payload["n_effective_measurement"], 1)
        self.assertTrue(_valid_aliasing(payload))


class TestCliDispatchEndToEnd(unittest.TestCase):
    """The --aliasing branch driven through the real run flow
    (cycle-179 lesson: a flag can parse while its dispatch silently
    never runs)."""

    def test_aliasing_dispatches_through_main(self):
        from model_protocol_lab import backends, doctor, experiments
        from model_protocol_lab import reporting as reporting_mod

        with tempfile.TemporaryDirectory() as tmp:
            fixture_path = Path(tmp) / "fixture.json"
            fixture_path.write_text(
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
            outdir = Path(tmp) / "report"
            result_doc = _aliased_pair()
            result_doc["held_out"] = {}

            class _FakeHFBackend:
                def __init__(self, model, device=None, max_context=None):
                    self.model = model

                eos_token_id = None

            def fake_prepare(path):
                Path(path).mkdir(parents=True, exist_ok=False)
                return Path(path)

            with contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch.object(backends, "LocalHFBackend", _FakeHFBackend))
                stack.enter_context(
                    mock.patch.object(cli_module, "run_experiment", return_value=result_doc)
                )
                stack.enter_context(
                    mock.patch.object(
                        cli_module,
                        "save_report",
                        return_value=(str(outdir / "results.json"), str(outdir / "report.md")),
                    )
                )
                stack.enter_context(
                    mock.patch.object(doctor, "template_conformance", return_value={"ok": True})
                )
                stack.enter_context(
                    mock.patch.object(
                        experiments, "budget_frontier", return_value={"available": False}
                    )
                )
                stack.enter_context(
                    mock.patch.object(reporting_mod, "prepare_report_dir", fake_prepare)
                )
                out, err = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    code = cli_module.main(
                        [
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
                            "--aliasing",
                        ]
                    )
            # Exit 2 is the run's abstention code (cli.py: only
            # compatible_on_fixture_set exits 0); the probe payload
            # must be attached before that mapping is applied.
            self.assertEqual(code, 2)
            payload = result_doc["aliasing"]
            self.assertTrue(payload["available"])
            self.assertTrue(payload["alias_suspect_abstention"])


class TestMutationGates(unittest.TestCase):
    """Tempdir gates (cycle-179 pattern): each mutant of the pure
    module's load-bearing constructions must be killed by a named
    scenario, and each scenario must pass on the real module."""

    def _run_gate(self, name, old_text, new_text, scenario):  # noqa: ANN001
        import importlib.util
        import os
        import sys

        mod = aliasing
        source = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertEqual(source.count(old_text), 1, f"mutant {name}: anchor not unique")
        mutated = source.replace(old_text, new_text)
        with tempfile.TemporaryDirectory(prefix="c181_mut_") as tmpdir:
            mod_path = os.path.join(tmpdir, f"aliasing_mut_{name}.py")
            with open(mod_path, "w", encoding="utf-8") as handle:
                handle.write(mutated)
            dotted = f"model_protocol_lab.aliasing_mut_{name}"
            spec = importlib.util.spec_from_file_location(dotted, mod_path)
            assert spec is not None and spec.loader is not None
            mutant = importlib.util.module_from_spec(spec)
            sys.modules[dotted] = mutant
            try:
                spec.loader.exec_module(mutant)
            finally:
                sys.modules.pop(dotted, None)
            scenario(mod)  # passes on the real module: gate is non-vacuous
            with self.assertRaises(AssertionError, msg=f"mutant {name} was NOT caught"):
                scenario(mutant)

    def test_mutant_boundary_dropped_from_measurement_caught(self):
        def scenario(mod):  # noqa: ANN001
            members = [
                {
                    "candidate": "A",
                    "boundary_token_ids": [99],
                    "records": [
                        {
                            "fixture": "f1",
                            "evaluable": True,
                            "input_sha256": mod.ids_digest([1]),
                        }
                    ],
                },
                {
                    "candidate": "B",
                    "boundary_token_ids": [7],
                    "records": [
                        {
                            "fixture": "f1",
                            "evaluable": True,
                            "input_sha256": mod.ids_digest([1]),
                        }
                    ],
                },
            ]
            _render, measurement = mod.class_partitions(members)
            assert measurement == [["A"], ["B"]], measurement

        self._run_gate(
            "boundary_dropped",
            '        measurement_sig = render_sig + (tuple(record["boundary_token_ids"]),)',
            "        measurement_sig = render_sig  # MUTANT: boundary dropped",
            scenario,
        )

    def test_mutant_empty_ids_evaluable_caught(self):
        def scenario(mod):  # noqa: ANN001
            out = mod.analyze_candidate(
                [
                    {
                        "candidate": "A",
                        "fixture": "f1",
                        "input_ids": [],
                        "output_ids": [5],
                        "exact_match": False,
                        "boundary_found": False,
                        "boundary_text_match": None,
                        "status": "completed",
                    }
                ],
                candidate_name="A",
                boundary_token_ids=[99],
                eligible=True,
                planned=1,
            )
            assert out.get("detail") == "recorded_ids_invalid", out

        self._run_gate(
            "empty_evaluable",
            "    return _valid_ids(value, nonempty=True)",
            "    return _valid_ids(value, nonempty=False)  # MUTANT: sentinel admitted",
            scenario,
        )

    def test_mutant_redundant_ignores_boundary_caught(self):
        def scenario(mod):  # noqa: ANN001
            members = [
                {
                    "candidate": "A",
                    "boundary_token_ids": [99],
                    "records": [
                        {
                            "fixture": "f1",
                            "evaluable": True,
                            "input_sha256": mod.ids_digest([1]),
                        }
                    ],
                },
                {
                    "candidate": "B",
                    "boundary_token_ids": [7],
                    "records": [
                        {
                            "fixture": "f1",
                            "evaluable": True,
                            "input_sha256": mod.ids_digest([1]),
                        }
                    ],
                },
            ]
            measurement, render = mod.redundant_counts(members)
            assert measurement == 0 and render == 1, (measurement, render)

        self._run_gate(
            "redundant_boundary",
            '            measurement_tuples.add((cell["input_sha256"], '
            'tuple(record["boundary_token_ids"])))',
            '            measurement_tuples.add((cell["input_sha256"],))'
            "  # MUTANT: boundary dropped",
            scenario,
        )


if __name__ == "__main__":
    unittest.main()


class TestValidatorGapBattery(unittest.TestCase):
    """Structural rejects in the recompute's first pass: each
    malformed shape must be refused before any aggregate is
    compared (the cycle-174 gap-test pattern)."""

    def setUp(self):
        self.payload = aliasing_profile(_aliased_pair())
        self.assertTrue(_valid_aliasing(self.payload))

    def _tamper(self, mutate) -> None:  # noqa: ANN001
        payload = copy.deepcopy(self.payload)
        mutate(payload)
        self.assertFalse(_valid_aliasing(payload))

    def test_candidates_container(self):
        self._tamper(lambda p: p.update(candidates="x"))
        self._tamper(lambda p: p.update(candidates=[]))
        self._tamper(lambda p: p.pop("candidates"))
        self._tamper(lambda p: p["candidates"].__setitem__(0, 42))
        self._tamper(lambda p: p["candidates"][0].update(candidate=7))
        self._tamper(lambda p: p["candidates"][0].update(available="yes"))

    def test_duplicate_candidate_name_in_payload(self):
        def dup(p):
            p["candidates"].append(copy.deepcopy(p["candidates"][0]))
            p["n_candidates_recorded"] += 1

        self._tamper(dup)

    def test_unavailable_entry_needs_str_detail(self):
        def bad(p):
            p["candidates"][0] = {"candidate": "X", "available": False, "detail": 7}

        self._tamper(bad)

    def test_available_entry_field_types(self):
        self._tamper(lambda p: p["candidates"][0].update(eligible="yes"))
        self._tamper(lambda p: p["candidates"][0].update(planned=-1))
        self._tamper(lambda p: p["candidates"][0].update(planned="2"))
        self._tamper(lambda p: p["candidates"][0].update(boundary_token_ids="x"))
        self._tamper(lambda p: p["candidates"][0].update(boundary_token_ids=[True]))
        self._tamper(lambda p: p["candidates"][0].update(boundary_token_ids=[-1]))
        self._tamper(lambda p: p["candidates"][0].update(records="x"))
        self._tamper(lambda p: p["candidates"][0].update(n_observations=99))
        self._tamper(lambda p: p["candidates"][0].update(n_not_completed=9))
        self._tamper(lambda p: p["candidates"][0].update(n_evaluable=9))

    def test_record_field_types(self):
        self._tamper(lambda p: p["candidates"][0]["records"].__setitem__(0, 42))
        self._tamper(lambda p: p["candidates"][0]["records"][0].update(fixture=""))
        self._tamper(lambda p: p["candidates"][0]["records"][0].update(status=7))
        self._tamper(lambda p: p["candidates"][0]["records"][0].update(exact_match="yes"))
        self._tamper(lambda p: p["candidates"][0]["records"][0].update(boundary_text_match="yes"))
        self._tamper(lambda p: p["candidates"][0]["records"][0].update(input_sha256="g" * 64))
        self._tamper(lambda p: p["candidates"][0]["records"][0].update(output_sha256="g" * 64))
        self._tamper(lambda p: p["candidates"][0].update(measurement_class_members=["A"]))

    def test_header_field_types(self):
        self._tamper(lambda p: p.update(method="x"))
        self._tamper(lambda p: p["method"].update(scope=""))
        self._tamper(lambda p: p.update(max_new_tokens="8"))
        self._tamper(lambda p: p.update(selected_candidate=7))
        self._tamper(lambda p: p.update(backend_fingerprint=7))
        self._tamper(lambda p: p.update(fixture_set_sha256="short"))
        self._tamper(lambda p: p.update(evaluable_mismatch="yes"))

    def _interrupted_payload(self):
        result = _result(
            [
                _entry("A", [_obs("A", "f1", [1]), _obs("A", "f2", [2])]),
                _entry(
                    "Z",
                    [_obs("Z", "f1", [1]), _obs("Z", "f2", (), status="failed")],
                    eligible=False,
                ),
            ],
            decision="failed",
        )
        payload = aliasing_profile(result)
        self.assertTrue(_valid_aliasing(payload))
        return payload

    def test_evaluable_status_coupling(self):
        payload = self._interrupted_payload()
        zed = next(c for c in payload["candidates"] if c["candidate"] == "Z")
        failed_record = next(r for r in zed["records"] if r["status"] == "failed")
        failed_record["evaluable"] = True  # evaluable but not completed
        self.assertFalse(_valid_aliasing(payload))

    def test_non_evaluable_digest_must_be_none(self):
        payload = self._interrupted_payload()
        zed = next(c for c in payload["candidates"] if c["candidate"] == "Z")
        failed_record = next(r for r in zed["records"] if r["status"] == "failed")
        failed_record["input_sha256"] = "a" * 64
        self.assertFalse(_valid_aliasing(payload))

    def test_malformed_entry_payload_validates_and_diffs(self):
        payload = aliasing_profile({"search": [42, *_aliased_pair()["search"]]})
        self.assertTrue(_valid_aliasing(payload))
        diff = diff_aliasing(payload, copy.deepcopy(payload))
        self.assertEqual(diff["available"], "both")
        self.assertFalse(diff["changed"])

    def test_diff_with_excluded_candidates(self):
        payload = self._interrupted_payload()
        diff = diff_aliasing(payload, copy.deepcopy(payload))
        self.assertEqual(diff["available"], "both")
        self.assertFalse(diff["changed"])
        bigger = aliasing_profile(_aliased_pair())
        diff = diff_aliasing(payload, bigger)
        self.assertIn("Z", diff["candidates_removed"])
        self.assertIn("Z", diff["excluded_removed"])

    def test_pairs_joined(self):
        other_result = _aliased_pair()
        other_result["search"][1]["observations"][0]["input_ids"] = [42]
        other = aliasing_profile(other_result)
        diff = diff_aliasing(other, self.payload)
        self.assertEqual(diff["pairs_joined"], [["A", "B"]])


class TestReportingGaps(unittest.TestCase):
    """Hostile-payload rendering: the section's defensive branches
    must degrade, never raise and never leak (cycle-174 pattern)."""

    def _render(self, aliasing_payload: Any) -> str:
        return reporting.render_html(
            {"decision": "ambiguous", "search": [], "aliasing": aliasing_payload}
        )

    def test_candidates_not_a_list(self):
        html = self._render({"available": True, "candidates": "x"})
        self.assertNotIn('id="aliasing"', html)

    def test_hostile_section_payload(self):
        payload = aliasing_profile(_aliased_pair())
        payload["determinism_suspect_cells"] = [
            42,
            {"fixture": "f1", "candidates": "AB", "kind": "output"},
        ]
        payload["measurement_classes"] = [["A", "B"], "oops"]
        payload["render_classes"] = [["A"], 7]
        payload["candidates"] = [42, *payload["candidates"]]
        html = self._render(payload)
        self.assertIn('id="aliasing"', html)
        self.assertIn("n/a", html)

    def test_mismatch_and_status_lines(self):
        result = _result(
            [
                _entry("A", [_obs("A", "f1", [1]), _obs("A", "f2", [2])]),
                _entry(
                    "Z",
                    [_obs("Z", "f1", [1]), _obs("Z", "f2", (), status="failed")],
                    eligible=False,
                ),
            ],
            decision="failed",
        )
        del result["search"][1]["observations"]
        result["search"].append(
            _entry(
                "Y",
                [_obs("Y", "f1", [5]), _obs("Y", "f2", (), status="failed")],
                eligible=False,
            )
        )
        payload = aliasing_profile(result)
        html = self._render(payload)
        self.assertIn("no usable recorded observations", html)
        self.assertIn("outside the primary cohort", html)

    def test_mismatch_caution_line(self):
        result = _result(
            [
                _entry("A", [_obs("A", "f1", [1]), _obs("A", "f2", [2])]),
                _entry(
                    "Z",
                    [_obs("Z", "f1", [1]), _obs("Z", "f2", (), status="failed")],
                    eligible=False,
                ),
            ],
            decision="failed",
        )
        payload = aliasing_profile(result)
        self.assertTrue(payload["evaluable_mismatch"])
        html = self._render(payload)
        self.assertIn("not all", html)
        self.assertIn("measured on the same completed fixture set", html)
