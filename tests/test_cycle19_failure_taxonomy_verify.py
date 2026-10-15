"""Cycle 19: differential verification of classify_observation / failure_taxonomy.

Target: ``model_protocol_lab.experiments.classify_observation`` (one recorded
observation -> failure class or None) and ``failure_taxonomy`` (per-candidate
and held-out aggregation of those classes for the decision banner).

Method: an INDEPENDENT oracle re-derived from the two functions' docstrings
only. It never calls the implementation's helpers: it recomputes the pass
check, the backend-failure grouping, and the priority-ordered failure
predicates with its own flag-based structure, and it aggregates with its own
counting code. The differential harness then compares, per observation, the
returned class (or the raised exception type), and per result, the full
taxonomy dict.

Brief-vs-docstring discrepancies (docstrings win; the oracle follows the
docstrings and mirrors the implementation only where the docstrings are
silent, because the task requires the input contract verified from the code):

1. The brief describes malformed observations as falling under a "never raises
   on bad data" contract. No docstring promises that, and the implementation
   does raise: ValueError for a non-dict observation (``classify_observation``)
   or a non-dict result (``failure_taxonomy``) -- confirmed by the existing
   ``test_round3.py::test_classify_observation_rejects_non_dict``; a non-dict
   element inside a search entry's observation list propagates ValueError out
   of ``failure_taxonomy``; and a truthy non-iterable ``output_ids`` (e.g. an
   int) escapes as TypeError from the EOS-membership check. The oracle
   reproduces these observable raise behaviors.
2. The docstring's priority list ("backend failures first") does not name the
   ``cancelled`` status; the cancelled -> ``backend-error`` grouping is
   documented only in a code comment ("Operational, not a measurement").
3. Search-stage non-dict observations raise, but held-out-stage non-dict
   observations are silently skipped -- an asymmetry neither docstring
   documents. The oracle reproduces it.
"""

import copy
import importlib
import os
import random
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from model_protocol_lab import experiments as real
from model_protocol_lab.experiments import FAILURE_MODES

# ---------------------------------------------------------------------------
# Independent oracle, re-derived from the docstrings of experiments.py.
# Closed taxonomy, deterministic from the stored record. Priority (docstring):
# backend failures first, then the tokenization-mismatch diagnostic, then
# self-termination via EOS, then absence of any boundary signal. A passing
# observation (exact_match and boundary_found) maps to None. A wrong answer
# with a found boundary maps to no-boundary-in-N.
# ---------------------------------------------------------------------------

_ORACLE_MODES = (
    "no-boundary-in-N",
    "boundary-text-mismatch",
    "eos-without-boundary",
    "held-out-divergence",
    "backend-error",
)


def _oracle_classify_observation(obs, eos_id):
    """Independent re-derivation of ``classify_observation``.

    Flag-based predicate structure (computed up front, then a priority
    chain), versus the implementation's inline checks. The EOS-membership
    probe stays lazy inside its branch on purpose: the implementation only
    builds ``tuple(output_ids)`` after the earlier branches miss, so a
    malformed ``output_ids`` raises TypeError there and only there -- the
    oracle mirrors that observable timing.
    """
    if type(obs) is not dict:
        raise ValueError("observation must be a dict")
    passed = bool(obs.get("exact_match")) and bool(obs.get("boundary_found"))
    status = obs.get("status")
    backend_failure = (
        status == "failed" or obs.get("raw_reason") == "error" or status == "cancelled"
    )
    boundary_found = bool(obs.get("boundary_found"))
    text_mismatch = obs.get("boundary_text_match") is True
    if passed:
        return None
    if backend_failure:
        return "backend-error"
    if not boundary_found:
        if text_mismatch:
            return "boundary-text-mismatch"
        output_ids = obs.get("output_ids") or ()
        if eos_id is not None and eos_id in tuple(output_ids):
            return "eos-without-boundary"
        return "no-boundary-in-N"
    # Wrong answer with a found boundary: the expected answer never appeared
    # boundary-delimited within the token cap.
    return "no-boundary-in-N"


def _oracle_failure_taxonomy(result, eos_id):
    """Independent re-derivation of ``failure_taxonomy``.

    Own aggregation code: dict comprehensions instead of ``dict.fromkeys``,
    explicit local for the held-out observations, same observable outcomes --
    per-candidate zero-initialized counts over the closed vocabulary, last
    duplicate name wins, held-out classified only when the decision is
    exactly ``held_out_failed`` with every non-passing held-out observation
    forced to ``held-out-divergence``.
    """
    if type(result) is not dict:
        raise ValueError("result must be a dict")
    per_candidate = {}
    search = result.get("search") or []
    for entry in search:
        if type(entry) is not dict:
            continue
        fmt = entry.get("format")
        name = fmt.get("name") if type(fmt) is dict else None
        if type(name) is not str:
            continue
        counts = dict.fromkeys(_ORACLE_MODES, 0)
        observations = entry.get("observations")
        if type(observations) is list:
            for obs in observations:
                mode = _oracle_classify_observation(obs, eos_id)
                if mode is not None:
                    counts[mode] += 1
        per_candidate[name] = counts
    held_out = None
    if result.get("decision") == "held_out_failed":
        held_out = dict.fromkeys(_ORACLE_MODES, 0)
        held = result.get("held_out")
        held_observations = held.get("observations") if type(held) is dict else None
        if type(held_observations) is list:
            for obs in held_observations:
                if type(obs) is dict:
                    passed = bool(obs.get("exact_match")) and bool(obs.get("boundary_found"))
                    if not passed:
                        held_out["held-out-divergence"] += 1
    return {
        "modes": list(_ORACLE_MODES),
        "per_candidate": per_candidate,
        "held_out": held_out,
        "eos_id_considered": eos_id is not None,
    }


# ---------------------------------------------------------------------------
# Differential harness
# ---------------------------------------------------------------------------


def _classify_outcome(func, obs, eos_id):
    """Normalize one classification to ("ok", value) or ("raise", exc-name)."""
    try:
        return ("ok", func(obs, eos_id))
    except (ValueError, TypeError) as exc:
        return ("raise", type(exc).__name__)


def _taxonomy_outcome(func, result, eos_id):
    """Normalize one taxonomy call; KeyError is a possible mutant signal."""
    try:
        return ("ok", func(result, eos_id))
    except (ValueError, TypeError, KeyError) as exc:
        return ("raise", type(exc).__name__)


def _classify_corpus():
    """Deterministic cross-product corpus of (observation, eos_id) pairs."""
    corpus = []
    statuses = ["completed", "failed", "cancelled", "bogus", "", None]
    exacts = [True, False, None]
    founds = [True, False, None]
    text_matches = [True, False, None]
    raw_reasons = [None, "error", "other"]
    output_shapes = [None, [], [0], [1, 2], [2], [7, 8], (3,)]
    eos_ids = [None, 0, 2, 9]
    for status in statuses:
        for exact in exacts:
            for found in founds:
                for text_match in text_matches:
                    for raw_reason in raw_reasons:
                        for output_ids in output_shapes:
                            for eos_id in eos_ids:
                                corpus.append(
                                    (
                                        {
                                            "status": status,
                                            "exact_match": exact,
                                            "boundary_found": found,
                                            "boundary_text_match": text_match,
                                            "raw_reason": raw_reason,
                                            "output_ids": output_ids,
                                        },
                                        eos_id,
                                    )
                                )
    # Edge shapes beyond the cross product.
    edge_obs = [
        {},  # every key missing
        {"status": "completed"},  # almost every key missing
        {"exact_match": 1, "boundary_found": "yes"},  # truthy non-bool pass
        {"exact_match": 1, "boundary_found": 0},  # mixed truthiness
        {
            "status": "completed",
            "exact_match": True,
            "boundary_found": True,
            "output_ids": 5,
        },  # malformed output_ids, but passes first
        {"status": "failed", "output_ids": 5},  # malformed output_ids, backend-error first
        {"status": "completed", "boundary_found": False, "output_ids": 5},
        {"status": "completed", "boundary_found": False, "output_ids": "ab"},
        {"status": "completed", "boundary_found": False, "output_ids": {"k": 2}},
        {"status": "completed", "boundary_found": False, "output_ids": 0},
        {
            "status": "completed",
            "boundary_found": False,
            "boundary_text_match": 1,
            "output_ids": [2],
        },  # truthy-but-not-True mismatch flag
        {
            "status": "completed",
            "boundary_found": False,
            "boundary_text_match": True,
            "output_ids": [2],
            "unknown_future_key": object(),
        },
    ]
    for obs in edge_obs:
        for eos_id in (None, 2):
            corpus.append((obs, eos_id))
    # Non-dict observations: the verified input contract raises ValueError.
    for bad in (None, [], "x", 5, (1,), {"a": 1}.keys()):
        corpus.append((bad, None))
    corpus.append((type("DictSubclass", (dict,), {})({"status": "completed"}), None))
    return corpus


def _random_obs(rng):
    obs = {}
    if rng.random() < 0.9:
        obs["status"] = rng.choice(["completed", "failed", "cancelled", "bogus", None])
    if rng.random() < 0.9:
        obs["exact_match"] = rng.choice([True, False, None])
    if rng.random() < 0.9:
        obs["boundary_found"] = rng.choice([True, False, None])
    if rng.random() < 0.7:
        obs["boundary_text_match"] = rng.choice([True, False, None])
    if rng.random() < 0.7:
        obs["raw_reason"] = rng.choice([None, "error", "other"])
    if rng.random() < 0.8:
        obs["output_ids"] = rng.choice([None, [], [1, 2], [2], [5], (4,)])
    if rng.random() < 0.05:
        obs["output_ids"] = rng.choice([5, "ab"])  # malformed shapes
    if rng.random() < 0.1:
        obs["extra_unknown_key"] = "ignored"
    return obs


def _random_result(rng):
    result = {}
    search = []
    for _ in range(rng.randint(0, 3)):
        if rng.random() < 0.08:
            search.append(rng.choice(["junk", 42, None, ["x"], {"no": "format"}]))
            continue
        if rng.random() < 0.12:
            fmt = rng.choice([None, "junk", 42, {}, {"name": None}, {"name": 42}])
        else:
            # Small name pool -> occasional duplicate names (last wins).
            fmt = {"name": rng.choice(["a", "b", "c"])}
        entry = {"format": fmt}
        obs_roll = rng.random()
        if obs_roll < 0.15:
            pass  # observations key missing
        elif obs_roll < 0.25:
            entry["observations"] = rng.choice([None, "junk", 42, {"a": 1}])
        else:
            observations = [_random_obs(rng) for _ in range(rng.randint(0, 4))]
            if rng.random() < 0.08:
                observations.append(rng.choice([None, "junk", 42]))
            entry["observations"] = observations
        search.append(entry)
    roll = rng.random()
    if roll < 0.9:
        result["search"] = search
    elif roll < 0.95:
        result["search"] = rng.choice([None, "junk", 42])
    # else: search key missing entirely
    if rng.random() < 0.9:
        result["decision"] = rng.choice(
            ["held_out_failed", "compatible_on_fixture_set", "ambiguous", "cancelled", None]
        )
    held_roll = rng.random()
    if held_roll < 0.5:
        held_observations = [_random_obs(rng) for _ in range(rng.randint(0, 3))]
        if rng.random() < 0.1:
            held_observations.append(rng.choice([None, "junk"]))
        result["held_out"] = {"observations": held_observations}
    elif held_roll < 0.65:
        result["held_out"] = rng.choice([None, "junk", 42, {}, {"observations": "junk"}])
    # else: held_out key missing entirely
    return result


def _taxonomy_corpus():
    """Deterministic corpus of (result, eos_id) pairs, seeded RNG only."""
    rng = random.Random(0xC1A551F)  # noqa: S311 -- deterministic test seeding, not crypto
    corpus = []
    for _ in range(60):
        corpus.append((_random_result(rng), rng.choice([None, 0, 2, 99])))
    # Hand-built edges.
    corpus.append(({"search": [], "decision": "held_out_failed"}, None))
    corpus.append(
        (
            {
                "search": [
                    {"format": {"name": "a"}},
                    {"format": {"name": "a"}, "observations": None},
                    {
                        "format": {"name": "a"},
                        "observations": [
                            {
                                "status": "completed",
                                "exact_match": False,
                                "boundary_found": False,
                                "output_ids": [2],
                            }
                        ],
                    },
                ],
                "decision": "held_out_failed",
                "held_out": {
                    "observations": [
                        {"status": "completed", "exact_match": True, "boundary_found": True},
                        {"status": "failed", "exact_match": False, "boundary_found": False},
                    ]
                },
            },
            2,
        )
    )
    # Non-dict results: the verified input contract raises ValueError.
    corpus.append(("not-a-dict", None))
    corpus.append((None, 2))
    corpus.append(([], 0))
    return corpus


def _count_classify_mismatches(module):
    bad = 0
    for obs, eos_id in _classify_corpus():
        expected = _classify_outcome(_oracle_classify_observation, obs, eos_id)
        actual = _classify_outcome(module.classify_observation, obs, eos_id)
        if actual != expected:
            bad += 1
    return bad


def _count_taxonomy_mismatches(module):
    bad = 0
    for result, eos_id in _taxonomy_corpus():
        expected = _taxonomy_outcome(_oracle_failure_taxonomy, result, eos_id)
        actual = _taxonomy_outcome(module.failure_taxonomy, result, eos_id)
        if actual != expected:
            bad += 1
    return bad


class VocabularyTests(unittest.TestCase):
    def test_oracle_vocabulary_matches_module(self):
        # The oracle's closed vocabulary is restated independently; it must
        # stay byte-identical to the module's FAILURE_MODES.
        self.assertEqual(list(_ORACLE_MODES), list(FAILURE_MODES))
        self.assertEqual(len(set(_ORACLE_MODES)), 5)
        self.assertTrue(all(type(mode) is str for mode in _ORACLE_MODES))


class ClassifyDocstringTests(unittest.TestCase):
    """Docstring-derived unit cases: each class and each precedence rule."""

    def test_pass_returns_none(self):
        obs = {"status": "completed", "exact_match": True, "boundary_found": True}
        self.assertIsNone(_oracle_classify_observation(obs, None))
        self.assertIsNone(real.classify_observation(obs, None))

    def test_pass_uses_truthiness(self):
        obs = {"exact_match": 1, "boundary_found": "yes"}
        self.assertIsNone(_oracle_classify_observation(obs, None))
        self.assertIsNone(real.classify_observation(obs, None))

    def test_status_failed_is_backend_error(self):
        obs = {"status": "failed", "exact_match": False, "boundary_found": False}
        self.assertEqual(_oracle_classify_observation(obs, None), "backend-error")
        self.assertEqual(real.classify_observation(obs, None), "backend-error")

    def test_raw_reason_error_is_backend_error(self):
        obs = {
            "status": "completed",
            "raw_reason": "error",
            "exact_match": False,
            "boundary_found": False,
        }
        self.assertEqual(_oracle_classify_observation(obs, None), "backend-error")
        self.assertEqual(real.classify_observation(obs, None), "backend-error")

    def test_cancelled_grouped_with_backend_error(self):
        # Docstring gap (see module docstring): the cancelled grouping lives
        # only in a code comment; both oracle and implementation agree.
        obs = {"status": "cancelled", "exact_match": False, "boundary_found": False}
        self.assertEqual(_oracle_classify_observation(obs, None), "backend-error")
        self.assertEqual(real.classify_observation(obs, None), "backend-error")

    def test_backend_error_beats_text_mismatch(self):
        obs = {
            "status": "failed",
            "boundary_found": False,
            "boundary_text_match": True,
            "output_ids": [2],
        }
        self.assertEqual(_oracle_classify_observation(obs, 2), "backend-error")
        self.assertEqual(real.classify_observation(obs, 2), "backend-error")

    def test_backend_error_beats_eos(self):
        obs = {
            "status": "failed",
            "boundary_found": False,
            "output_ids": [2],
        }
        self.assertEqual(_oracle_classify_observation(obs, 2), "backend-error")
        self.assertEqual(real.classify_observation(obs, 2), "backend-error")

    def test_text_mismatch_beats_eos(self):
        obs = {
            "status": "completed",
            "boundary_found": False,
            "boundary_text_match": True,
            "output_ids": [2],
        }
        self.assertEqual(_oracle_classify_observation(obs, 2), "boundary-text-mismatch")
        self.assertEqual(real.classify_observation(obs, 2), "boundary-text-mismatch")

    def test_text_mismatch_requires_exact_true(self):
        # boundary_text_match uses `is True`, not truthiness.
        obs = {
            "status": "completed",
            "boundary_found": False,
            "boundary_text_match": 1,
            "output_ids": [2],
        }
        self.assertEqual(_oracle_classify_observation(obs, 2), "eos-without-boundary")
        self.assertEqual(real.classify_observation(obs, 2), "eos-without-boundary")

    def test_eos_without_boundary(self):
        obs = {
            "status": "completed",
            "exact_match": False,
            "boundary_found": False,
            "output_ids": [7, 2, 9],
        }
        self.assertEqual(_oracle_classify_observation(obs, 2), "eos-without-boundary")
        self.assertEqual(real.classify_observation(obs, 2), "eos-without-boundary")

    def test_eos_none_gives_no_boundary_in_n(self):
        obs = {
            "status": "completed",
            "exact_match": False,
            "boundary_found": False,
            "output_ids": [7, 2, 9],
        }
        self.assertEqual(_oracle_classify_observation(obs, None), "no-boundary-in-N")
        self.assertEqual(real.classify_observation(obs, None), "no-boundary-in-N")

    def test_eos_absent_gives_no_boundary_in_n(self):
        obs = {
            "status": "completed",
            "exact_match": False,
            "boundary_found": False,
            "output_ids": [7, 8],
        }
        self.assertEqual(_oracle_classify_observation(obs, 2), "no-boundary-in-N")
        self.assertEqual(real.classify_observation(obs, 2), "no-boundary-in-N")

    def test_empty_observation_maps_to_no_boundary_in_n(self):
        self.assertEqual(_oracle_classify_observation({}, 2), "no-boundary-in-N")
        self.assertEqual(real.classify_observation({}, 2), "no-boundary-in-N")

    def test_wrong_answer_with_found_boundary(self):
        # Docstring: the expected answer never appeared boundary-delimited
        # within the token cap -> no-boundary-in-N.
        obs = {"status": "completed", "exact_match": False, "boundary_found": True}
        self.assertEqual(_oracle_classify_observation(obs, 2), "no-boundary-in-N")
        self.assertEqual(real.classify_observation(obs, 2), "no-boundary-in-N")

    def test_non_dict_raises_value_error(self):
        for bad in ("nope", None, [], 5, (1,)):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    _oracle_classify_observation(bad, None)
                with self.assertRaises(ValueError):
                    real.classify_observation(bad, None)

    def test_malformed_output_ids_raise_type_error_in_eos_branch(self):
        # Docstring gap (see module docstring): a truthy non-iterable
        # output_ids escapes as TypeError, but only when the EOS branch is
        # actually reached -- backend-error and pass short-circuit first.
        short_circuit = [
            {"status": "failed", "output_ids": 5},
            {"exact_match": True, "boundary_found": True, "output_ids": 5},
        ]
        for obs in short_circuit:
            with self.subTest(obs=obs):
                self.assertEqual(
                    _classify_outcome(_oracle_classify_observation, obs, 2),
                    _classify_outcome(real.classify_observation, obs, 2),
                )
        reaching = {"status": "completed", "boundary_found": False, "output_ids": 5}
        self.assertEqual(
            _classify_outcome(_oracle_classify_observation, reaching, 2),
            ("raise", "TypeError"),
        )
        self.assertEqual(
            _classify_outcome(real.classify_observation, reaching, 2),
            ("raise", "TypeError"),
        )


class ClassifyFuzzTests(unittest.TestCase):
    def test_fuzz_classify_matches_oracle(self):
        corpus = _classify_corpus()
        for obs, eos_id in corpus:
            with self.subTest(obs=obs, eos_id=eos_id):
                expected = _classify_outcome(_oracle_classify_observation, obs, eos_id)
                actual = _classify_outcome(real.classify_observation, obs, eos_id)
                self.assertEqual(actual, expected)


class TaxonomyContractTests(unittest.TestCase):
    """Docstring-derived structural invariants on the implementation."""

    def test_taxonomy_shape(self):
        result = {"search": [], "decision": "no_candidate"}
        taxonomy = real.failure_taxonomy(result, None)
        self.assertEqual(taxonomy["modes"], list(FAILURE_MODES))
        self.assertEqual(taxonomy["per_candidate"], {})
        self.assertIsNone(taxonomy["held_out"])
        self.assertFalse(taxonomy["eos_id_considered"])

    def test_all_pass_candidate_gets_zero_counts(self):
        observations = [
            {"status": "completed", "exact_match": True, "boundary_found": True},
            {"status": "completed", "exact_match": 1, "boundary_found": "y"},
        ]
        result = {
            "search": [{"format": {"name": "a"}, "observations": observations}],
            "decision": "compatible_on_fixture_set",
        }
        taxonomy = real.failure_taxonomy(result, None)
        self.assertEqual(sum(taxonomy["per_candidate"]["a"].values()), 0)
        self.assertEqual(set(taxonomy["per_candidate"]["a"]), set(FAILURE_MODES))

    def test_counts_sum_to_nonpassing_observations(self):
        observations = [
            {"status": "completed", "exact_match": True, "boundary_found": True},
            {"status": "failed"},
            {"status": "completed", "boundary_found": False, "output_ids": [2]},
            {"status": "completed", "boundary_found": False, "boundary_text_match": True},
            {"status": "completed", "exact_match": False, "boundary_found": True},
        ]
        result = {
            "search": [{"format": {"name": "a"}, "observations": observations}],
            "decision": "x",
        }
        taxonomy = real.failure_taxonomy(result, 2)
        counts = taxonomy["per_candidate"]["a"]
        self.assertEqual(sum(counts.values()), 4)
        self.assertEqual(counts["backend-error"], 1)
        self.assertEqual(counts["eos-without-boundary"], 1)
        self.assertEqual(counts["boundary-text-mismatch"], 1)
        self.assertEqual(counts["no-boundary-in-N"], 1)

    def test_held_out_none_unless_winner_diverged(self):
        base = {"search": [], "held_out": {"observations": [{"status": "failed"}]}}
        for decision in ("compatible_on_fixture_set", "ambiguous", "no_candidate", None):
            result = dict(base, decision=decision)
            self.assertIsNone(real.failure_taxonomy(result, None)["held_out"])

    def test_held_out_zeros_when_stage_missing(self):
        result = {"search": [], "decision": "held_out_failed"}
        taxonomy = real.failure_taxonomy(result, None)
        self.assertIsNotNone(taxonomy["held_out"])
        self.assertEqual(sum(taxonomy["held_out"].values()), 0)

    def test_held_out_forces_divergence_regardless_of_failure(self):
        # Docstring: every non-passing held-out observation counts as
        # held-out-divergence regardless of its specific failure.
        held_observations = [
            {"status": "failed", "exact_match": False, "boundary_found": False},
            {"status": "completed", "boundary_found": False, "output_ids": [2]},
            {"status": "completed", "exact_match": True, "boundary_found": True},
        ]
        result = {
            "search": [],
            "decision": "held_out_failed",
            "held_out": {"observations": held_observations},
        }
        taxonomy = real.failure_taxonomy(result, 2)
        self.assertEqual(taxonomy["held_out"]["held-out-divergence"], 2)
        others = sum(
            value for mode, value in taxonomy["held_out"].items() if mode != "held-out-divergence"
        )
        self.assertEqual(others, 0)

    def test_eos_id_considered_flag(self):
        result = {"search": [], "decision": "x"}
        self.assertFalse(real.failure_taxonomy(result, None)["eos_id_considered"])
        self.assertTrue(real.failure_taxonomy(result, 0)["eos_id_considered"])
        self.assertTrue(real.failure_taxonomy(result, 2)["eos_id_considered"])

    def test_duplicate_candidate_names_last_wins(self):
        result = {
            "search": [
                {
                    "format": {"name": "a"},
                    "observations": [{"status": "failed"}],
                },
                {
                    "format": {"name": "a"},
                    "observations": [{"status": "completed", "boundary_found": False}],
                },
            ],
            "decision": "x",
        }
        taxonomy = real.failure_taxonomy(result, None)
        counts = taxonomy["per_candidate"]["a"]
        self.assertEqual(counts["backend-error"], 0)
        self.assertEqual(counts["no-boundary-in-N"], 1)

    def test_input_not_mutated(self):
        result, eos_id = _taxonomy_corpus()[1]
        snapshot = copy.deepcopy(result)
        real.failure_taxonomy(result, eos_id)
        _oracle_failure_taxonomy(copy.deepcopy(snapshot), eos_id)
        self.assertEqual(result, snapshot)

    def test_result_non_dict_raises(self):
        for bad in ("nope", None, [], 5):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    real.failure_taxonomy(bad, None)
                with self.assertRaises(ValueError):
                    _oracle_failure_taxonomy(bad, None)


class TaxonomyFuzzTests(unittest.TestCase):
    def test_fuzz_taxonomy_matches_oracle(self):
        for result, eos_id in _taxonomy_corpus():
            with self.subTest(result=result, eos_id=eos_id):
                expected = _taxonomy_outcome(_oracle_failure_taxonomy, result, eos_id)
                actual = _taxonomy_outcome(real.failure_taxonomy, result, eos_id)
                self.assertEqual(actual, expected)


# ---------------------------------------------------------------------------
# Sensitivity controls: mutate the implementation in a throwaway copy and
# prove the differential harness catches each mutation.
# ---------------------------------------------------------------------------

# PID-scoped: two pytest processes staging mutants concurrently must not share
# a directory (a fixed name raced: one process's rmtree deleted the other's
# in-use package copies -> import failures).
_MUTANT_ROOT = Path(tempfile.gettempdir()) / f"mpl_cycle19_taxonomy_mutants_{os.getpid()}"
_MUTANT_PATH_READY = False


def _build_mutant(package_name, replacements):
    """Copy the package under a fresh top-level name, apply textual mutations
    to its experiments.py, and import the mutated module."""
    global _MUTANT_PATH_READY
    src = Path(real.__file__).resolve().parent
    dest = _MUTANT_ROOT / package_name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns("__pycache__"))
    path = dest / "experiments.py"
    text = path.read_text(encoding="utf-8")
    for old, new, expected_count in replacements:
        found = text.count(old)
        assert found == expected_count, (package_name, old[:60], found, expected_count)
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")
    if not _MUTANT_PATH_READY:
        sys.path.insert(0, str(_MUTANT_ROOT))
        _MUTANT_PATH_READY = True
    return importlib.import_module(f"{package_name}.experiments")


_SWAP_PREDICATES = [
    (
        '        if obs.get("boundary_text_match") is True:\n'
        '            return "boundary-text-mismatch"\n'
        '        if eos_id is not None and eos_id in tuple(obs.get("output_ids") or ()):\n'
        '            return "eos-without-boundary"\n',
        '        if eos_id is not None and eos_id in tuple(obs.get("output_ids") or ()):\n'
        '            return "eos-without-boundary"\n'
        '        if obs.get("boundary_text_match") is True:\n'
        '            return "boundary-text-mismatch"\n',
        1,
    )
]

_DROP_CLASS = [
    (
        "        counts = dict.fromkeys(FAILURE_MODES, 0)\n",
        '        counts = {m: 0 for m in FAILURE_MODES if m != "eos-without-boundary"}\n',
        1,
    ),
    (
        "        held_out_counts = dict.fromkeys(FAILURE_MODES, 0)\n",
        '        held_out_counts = {m: 0 for m in FAILURE_MODES if m != "eos-without-boundary"}\n',
        1,
    ),
]

_HELDOUT_PER_OBS = [
    (
        "            for obs in observations:\n"
        "                if type(obs) is dict and not "
        '(obs.get("exact_match") and obs.get("boundary_found")):\n'
        '                    held_out_counts["held-out-divergence"] += 1\n',
        "            for obs in observations:\n"
        "                if type(obs) is dict:\n"
        "                    mode = classify_observation(obs, eos_id)\n"
        "                    if mode is not None:\n"
        "                        held_out_counts[mode] += 1\n",
        1,
    )
]


class SensitivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mut_swap = _build_mutant("mut_swap_predicates", _SWAP_PREDICATES)
        cls.mut_drop = _build_mutant("mut_drop_class", _DROP_CLASS)
        cls.mut_heldout = _build_mutant("mut_heldout_per_obs", _HELDOUT_PER_OBS)

    @classmethod
    def tearDownClass(cls):
        # Do not accumulate package copies in /tmp across runs.
        shutil.rmtree(_MUTANT_ROOT, ignore_errors=True)

    def test_real_implementation_has_zero_mismatches(self):
        self.assertEqual(_count_classify_mismatches(real), 0)
        self.assertEqual(_count_taxonomy_mismatches(real), 0)

    def test_swapped_predicates_caught(self):
        # Both the tokenization-mismatch and the EOS condition hold: the
        # docstring priority (mismatch before EOS) decides.
        obs = {
            "status": "completed",
            "boundary_found": False,
            "boundary_text_match": True,
            "output_ids": [2],
        }
        self.assertEqual(_oracle_classify_observation(obs, 2), "boundary-text-mismatch")
        self.assertEqual(real.classify_observation(obs, 2), "boundary-text-mismatch")
        self.assertEqual(self.mut_swap.classify_observation(obs, 2), "eos-without-boundary")
        self.assertGreater(_count_classify_mismatches(self.mut_swap), 0)

    def test_dropped_class_caught(self):
        result = {
            "search": [
                {
                    "format": {"name": "a"},
                    "observations": [
                        {
                            "status": "completed",
                            "exact_match": False,
                            "boundary_found": False,
                            "output_ids": [2],
                        }
                    ],
                }
            ],
            "decision": "x",
        }
        expected = _oracle_failure_taxonomy(result, 2)
        self.assertEqual(real.failure_taxonomy(result, 2), expected)
        self.assertEqual(expected["per_candidate"]["a"]["eos-without-boundary"], 1)
        with self.assertRaises(KeyError):
            self.mut_drop.failure_taxonomy(result, 2)
        self.assertGreater(_count_taxonomy_mismatches(self.mut_drop), 0)

    def test_heldout_per_observation_classification_caught(self):
        result = {
            "search": [],
            "decision": "held_out_failed",
            "held_out": {
                "observations": [
                    {"status": "failed", "exact_match": False, "boundary_found": False}
                ]
            },
        }
        expected = _oracle_failure_taxonomy(result, None)
        self.assertEqual(real.failure_taxonomy(result, None), expected)
        self.assertEqual(expected["held_out"]["held-out-divergence"], 1)
        mutated = self.mut_heldout.failure_taxonomy(result, None)
        self.assertEqual(mutated["held_out"]["held-out-divergence"], 0)
        self.assertEqual(mutated["held_out"]["backend-error"], 1)
        self.assertGreater(_count_taxonomy_mismatches(self.mut_heldout), 0)


if __name__ == "__main__":
    unittest.main()
