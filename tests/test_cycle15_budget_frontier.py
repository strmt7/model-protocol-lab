"""Cycle 15: differential verification of ``budget_frontier``.

``budget_frontier`` (experiments.py) claims an *exact* derivation of the
token-budget robustness frontier from the recorded search observations:
per-fixture minimal passing budgets (``cut + len(boundary)``), pass counts
via bisect on sorted minima, the conservative unique-winner ``decision_at``
rule, ``decision_steps``, ``flip_ranges`` and ``budget_robust``.

These tests check the implementation against an INDEPENDENT brute-force
oracle built from the docstring definition alone. The oracle never uses
the implementation's derivation -- no min-budget arithmetic, no bisect.
For every budget ``b`` in ``1..max_new_tokens`` it truncates each
observation's ``output_ids`` to ``b`` tokens and directly re-evaluates
pass/fail (``status == "completed"``, ``exact_match is True``,
``boundary_found is True``, and
``first_boundary(output_ids[:b], boundary) + len(boundary) <= b``), then
applies the unique-winner rule per budget. Everything else -- pass steps,
minimal budgets (found by scanning budgets, never by ``cut + len``
arithmetic), decision steps, flip ranges, robustness -- is recomputed from
those per-budget pass/fail outcomes only.

Coverage:

- seeded differential fuzz: 2-6 candidates, 1-8 fixtures, 1-64 budgets,
  per-candidate ``boundary_token_ids`` in absent/list/tuple forms,
  passing and failing observations, inconsistent flag combinations;
- hand-written edge cases: boundary at offset 0, output shorter than the
  boundary, multiple boundary occurrences, budgets below/above all minima,
  single candidate, all-ambiguous verdicts, native flags inconsistent
  with the observations;
- input-validation errors;
- sensitivity controls proving the fuzz actually catches real mistakes: a
  +1 shift of the minimal-budget derivation and an off-by-one in the
  decision counting are both detected.
"""

import random
import unittest
from unittest import mock

from model_protocol_lab import experiments
from model_protocol_lab.core import first_boundary
from model_protocol_lab.experiments import budget_frontier

N_FUZZ_SEEDS = 150


# ---------------------------------------------------------------------------
# Independent brute-force oracle (docstring definition only).
# ---------------------------------------------------------------------------


def _entry_boundary(entry, default):
    """Per-candidate boundary, exactly as the docstring specifies."""
    entry_boundary = entry.get("boundary_token_ids")
    if entry_boundary is None:
        return default
    return tuple(entry_boundary)


def _passes_at(observation, entry_boundary, budget):
    """Naive per-budget pass predicate: truncate to ``budget`` and re-evaluate.

    No derivation, no minima: the recorded output is cut to ``budget``
    tokens and the pass condition is checked directly on the truncated
    sequence.
    """
    if observation.get("status") != "completed":
        return False
    if observation.get("exact_match") is not True:
        return False
    if observation.get("boundary_found") is not True:
        return False
    raw = observation.get("output_ids")
    ids = tuple(raw) if isinstance(raw, (list, tuple)) else ()
    if not ids or any(type(token) is not int for token in ids):
        return False
    cut = first_boundary(ids[:budget], entry_boundary)
    return cut is not None and cut + len(entry_boundary) <= budget


def _oracle_budget_frontier(search, *, boundary, max_new_tokens, native_decision, native_winner):
    """Brute-force oracle: recompute the frontier from per-budget pass/fail."""
    names = []
    entry_boundaries = {}
    observations = {}
    for entry in search:
        name = entry["format"]["name"]
        names.append(name)
        entry_boundaries[name] = _entry_boundary(entry, boundary)
        observations[name] = entry["observations"]
    n_fixtures = len(observations[names[0]])

    def pass_count(name, budget):
        entry_boundary = entry_boundaries[name]
        return sum(
            1
            for observation in observations[name]
            if _passes_at(observation, entry_boundary, budget)
        )

    def decision_at(budget):
        winners = [name for name in names if pass_count(name, budget) == n_fixtures]
        if len(winners) == 1:
            return ("selected", winners[0])
        if not winners:
            return ("no_candidate", None)
        return ("ambiguous", None)

    if native_decision in ("compatible_on_fixture_set", "held_out_failed"):
        native_search = ("selected", native_winner)
    else:
        native_search = (native_decision, None)

    candidates = {}
    for name in names:
        steps = []
        previous = None
        for budget in range(1, max_new_tokens + 1):
            count = pass_count(name, budget)
            if count != previous:
                steps.append([budget, count])
                previous = count
        # Minimal passing budget per fixture: found by SCANNING budgets for
        # the first pass -- deliberately not ``cut + len(boundary)``.
        min_budgets = []
        for observation in observations[name]:
            entry_boundary = entry_boundaries[name]
            raw = observation.get("output_ids")
            ids = tuple(raw) if isinstance(raw, (list, tuple)) else ()
            scan_hi = len(ids) + len(entry_boundary) + 1
            found = next(
                (
                    budget
                    for budget in range(1, scan_hi + 1)
                    if _passes_at(observation, entry_boundary, budget)
                ),
                None,
            )
            min_budgets.append(found)
        candidates[name] = {
            "pass_steps": steps,
            "full_pass_budget": next(
                (budget for budget, count in steps if count == n_fixtures), None
            ),
            "min_budget_per_fixture": min_budgets,
        }

    decision_steps = []
    previous_decision = None
    for budget in range(1, max_new_tokens + 1):
        current = decision_at(budget)
        if current != previous_decision:
            decision_steps.append([budget, current[0], current[1]])
            previous_decision = current

    flip_ranges = []
    for budget in range(1, max_new_tokens + 1):
        if decision_at(budget) != native_search:
            if flip_ranges and flip_ranges[-1][1] == budget - 1:
                flip_ranges[-1][1] = budget
            else:
                flip_ranges.append([budget, budget])

    return {
        "native_search_decision": native_search[0],
        "native_winner": native_search[1],
        "candidates": candidates,
        "decision_steps": decision_steps,
        "flip_ranges": flip_ranges,
        "budget_robust": not flip_ranges,
    }


def _compare(impl, oracle):
    """Return a list of human-readable mismatches between implementation and oracle."""
    mismatches = []
    for key in ("native_search_decision", "native_winner"):
        if impl[key] != oracle[key]:
            mismatches.append(f"{key}: impl={impl[key]!r} oracle={oracle[key]!r}")
    for name, oracle_candidate in oracle["candidates"].items():
        impl_candidate = impl["candidates"][name]
        for key in ("pass_steps", "full_pass_budget", "min_budget_per_fixture"):
            if impl_candidate[key] != oracle_candidate[key]:
                mismatches.append(
                    f"candidates[{name!r}][{key}]: "
                    f"impl={impl_candidate[key]!r} oracle={oracle_candidate[key]!r}"
                )
    for key in ("decision_steps", "flip_ranges", "budget_robust"):
        if impl[key] != oracle[key]:
            mismatches.append(f"{key}: impl={impl[key]!r} oracle={oracle[key]!r}")
    return mismatches


# ---------------------------------------------------------------------------
# Seeded differential fuzz.
# ---------------------------------------------------------------------------


def _gen_observation(rng, fixture_id, alphabet, shared_boundary, entry_boundary, max_new_tokens):
    """Build one adversarial observation; flags are deliberately sometimes inconsistent."""
    boundary = entry_boundary if entry_boundary is not None else shared_boundary
    # Token stream: random length, sometimes longer than the budget cap.
    length = rng.randint(0, max_new_tokens + 8)
    ids = [rng.choice(alphabet) for _ in range(length)]
    plant_roll = rng.random()
    if plant_roll < 0.55 and length >= len(boundary):
        # Plant the boundary at a random offset; sometimes twice.
        offset = rng.randint(0, length - len(boundary))
        ids[offset : offset + len(boundary)] = list(boundary)
        if rng.random() < 0.3 and length >= len(boundary):
            offset2 = rng.randint(0, length - len(boundary))
            ids[offset2 : offset2 + len(boundary)] = list(boundary)
    elif plant_roll < 0.65 and length >= len(boundary) + 1:
        # Near-miss: boundary prefix present but incomplete.
        offset = rng.randint(0, length - len(boundary))
        ids[offset : offset + len(boundary) - 1] = list(boundary[:-1])
    # else: no boundary planted at all.
    observation = {
        "fixture": fixture_id,
        "status": "completed" if rng.random() < 0.85 else rng.choice(["error", "timeout"]),
        "exact_match": rng.random() < 0.7,
        "boundary_found": rng.random() < 0.7,
        "output_ids": ids,
    }
    # List-vs-tuple and degenerate id forms the implementation must tolerate.
    form_roll = rng.random()
    if form_roll < 0.1:
        observation["output_ids"] = tuple(ids)
    elif form_roll < 0.13:
        observation["output_ids"] = []
    elif form_roll < 0.15 and ids:
        bad = list(ids)
        bad[rng.randrange(len(bad))] = "7"
        observation["output_ids"] = bad
    return observation


def _gen_case(seed):
    """Deterministic adversarial budget_frontier input from ``seed``."""
    rng = random.Random(seed)  # noqa: S311 -- seeded deterministic fuzz, not crypto
    n_candidates = rng.randint(2, 6)
    n_fixtures = rng.randint(1, 8)
    max_new_tokens = rng.randint(1, 64)
    alphabet = list(range(rng.randint(2, 5)))
    shared_boundary = tuple(rng.choice(alphabet) for _ in range(rng.randint(1, 3)))
    fixture_ids = [f"f{i}" for i in range(n_fixtures)]
    search = []
    names = [f"cand-{i}" for i in range(n_candidates)]
    for name in names:
        # Per-candidate boundary: absent, list form, or tuple form.
        boundary_roll = rng.random()
        entry_boundary = None
        extra = {}
        if boundary_roll < 0.4:
            pass
        elif boundary_roll < 0.7:
            entry_boundary = [rng.choice(alphabet) for _ in range(rng.randint(1, 3))]
            extra = {"boundary_token_ids": entry_boundary}
        else:
            entry_boundary = tuple(rng.choice(alphabet) for _ in range(rng.randint(1, 3)))
            extra = {"boundary_token_ids": entry_boundary}
        observations = [
            _gen_observation(
                rng, fixture_id, alphabet, shared_boundary, entry_boundary, max_new_tokens
            )
            for fixture_id in fixture_ids
        ]
        search.append({"format": {"name": name}, "observations": observations, **extra})
    native_decision = rng.choice(
        ["compatible_on_fixture_set", "held_out_failed", "ambiguous", "no_candidate"]
    )
    if native_decision in ("compatible_on_fixture_set", "held_out_failed"):
        native_winner = rng.choice(names)
    else:
        native_winner = None
    return search, shared_boundary, max_new_tokens, native_decision, native_winner


class BudgetFrontierFuzzTests(unittest.TestCase):
    def test_oracle_matches_implementation_on_fuzz(self):
        for seed in range(N_FUZZ_SEEDS):
            with self.subTest(seed=seed):
                search, boundary, max_new_tokens, native_decision, native_winner = _gen_case(
                    0xC15E + seed
                )
                impl = budget_frontier(
                    search,
                    boundary=boundary,
                    max_new_tokens=max_new_tokens,
                    native_decision=native_decision,
                    native_winner=native_winner,
                )
                oracle = _oracle_budget_frontier(
                    search,
                    boundary=boundary,
                    max_new_tokens=max_new_tokens,
                    native_decision=native_decision,
                    native_winner=native_winner,
                )
                mismatches = _compare(impl, oracle)
                self.assertEqual(mismatches, [], f"seed {seed}: oracle mismatch")


class BudgetFrontierFuzzCorpusTests(unittest.TestCase):
    """A fixed corpus of adversarial seeds exercising rare corners."""

    def test_adversarial_corpus(self):
        # Seeds hand-picked for small alphabets (boundary collisions),
        # single fixtures, and tiny budgets.
        for seed in (1, 2, 3, 7, 11, 13, 42, 99, 1234, 99991):
            with self.subTest(seed=seed):
                search, boundary, max_new_tokens, native_decision, native_winner = _gen_case(
                    0xBAD0 + seed
                )
                impl = budget_frontier(
                    search,
                    boundary=boundary,
                    max_new_tokens=max_new_tokens,
                    native_decision=native_decision,
                    native_winner=native_winner,
                )
                oracle = _oracle_budget_frontier(
                    search,
                    boundary=boundary,
                    max_new_tokens=max_new_tokens,
                    native_decision=native_decision,
                    native_winner=native_winner,
                )
                self.assertEqual(_compare(impl, oracle), [], f"seed {seed}: oracle mismatch")


# ---------------------------------------------------------------------------
# Hand-written edge cases.
# ---------------------------------------------------------------------------


def _obs(fixture, output_ids, *, status="completed", exact_match=True, boundary_found=True):
    return {
        "fixture": fixture,
        "status": status,
        "exact_match": exact_match,
        "boundary_found": boundary_found,
        "output_ids": list(output_ids),
    }


def _entry(name, observations, boundary_token_ids=None):
    entry = {"format": {"name": name}, "observations": observations}
    if boundary_token_ids is not None:
        entry["boundary_token_ids"] = boundary_token_ids
    return entry


def _run(search, boundary, max_new_tokens, native_decision, native_winner):
    impl = budget_frontier(
        search,
        boundary=boundary,
        max_new_tokens=max_new_tokens,
        native_decision=native_decision,
        native_winner=native_winner,
    )
    oracle = _oracle_budget_frontier(
        search,
        boundary=boundary,
        max_new_tokens=max_new_tokens,
        native_decision=native_decision,
        native_winner=native_winner,
    )
    return impl, oracle


class BudgetFrontierEdgeCaseTests(unittest.TestCase):
    def assert_matches_oracle(
        self, search, boundary, max_new_tokens, native_decision, native_winner
    ):
        impl, oracle = _run(search, boundary, max_new_tokens, native_decision, native_winner)
        self.assertEqual(_compare(impl, oracle), [])
        return impl

    def test_boundary_at_offset_zero(self):
        search = [_entry("a", [_obs("f1", [5, 6, 1, 2])])]
        impl = self.assert_matches_oracle(search, (5, 6), 8, "no_candidate", None)
        candidate = impl["candidates"]["a"]
        self.assertEqual(candidate["min_budget_per_fixture"], [2])
        self.assertEqual(candidate["pass_steps"], [[1, 0], [2, 1]])
        self.assertEqual(candidate["full_pass_budget"], 2)

    def test_output_shorter_than_boundary_never_passes(self):
        search = [_entry("a", [_obs("f1", [1, 2])])]
        impl = self.assert_matches_oracle(search, (1, 2, 3), 8, "no_candidate", None)
        candidate = impl["candidates"]["a"]
        self.assertEqual(candidate["min_budget_per_fixture"], [None])
        self.assertEqual(candidate["pass_steps"], [[1, 0]])
        self.assertIsNone(candidate["full_pass_budget"])

    def test_multiple_occurrences_first_counts(self):
        search = [_entry("a", [_obs("f1", [9, 1, 2, 1, 2])])]
        impl = self.assert_matches_oracle(search, (1, 2), 8, "no_candidate", None)
        self.assertEqual(impl["candidates"]["a"]["min_budget_per_fixture"], [3])

    def test_truncation_cutoff_is_exact(self):
        # Boundary completes at token index 6 (cut=5, len=2): passes at 7, fails at 6.
        search = [_entry("a", [_obs("f1", [0, 0, 0, 0, 0, 1, 2, 9])])]
        impl = self.assert_matches_oracle(search, (1, 2), 10, "no_candidate", None)
        candidate = impl["candidates"]["a"]
        self.assertEqual(candidate["min_budget_per_fixture"], [7])
        self.assertEqual(candidate["pass_steps"], [[1, 0], [7, 1]])

    def test_noncompleted_observation_never_passes(self):
        search = [
            _entry(
                "a",
                [_obs("f1", [1, 2, 3], status="error")],
            )
        ]
        impl = self.assert_matches_oracle(search, (1, 2), 8, "no_candidate", None)
        self.assertEqual(impl["candidates"]["a"]["min_budget_per_fixture"], [None])

    def test_flag_lie_boundary_found_without_boundary(self):
        # boundary_found claims True but no boundary is present: still cannot pass.
        search = [_entry("a", [_obs("f1", [7, 7, 7], boundary_found=True)])]
        impl = self.assert_matches_oracle(search, (1, 2), 8, "no_candidate", None)
        self.assertEqual(impl["candidates"]["a"]["min_budget_per_fixture"], [None])

    def test_single_candidate_flip_ranges(self):
        # Candidate passes both fixtures from budget 3 on; native winner holds only there.
        search = [
            _entry("a", [_obs("f1", [9, 1, 2, 3]), _obs("f2", [8, 1, 2])]),
            _entry("b", [_obs("f1", [9, 9, 9]), _obs("f2", [8, 8, 8])]),
        ]
        impl = self.assert_matches_oracle(search, (1, 2), 8, "compatible_on_fixture_set", "a")
        self.assertEqual(impl["decision_steps"], [[1, "no_candidate", None], [3, "selected", "a"]])
        self.assertEqual(impl["flip_ranges"], [[1, 2]])
        self.assertFalse(impl["budget_robust"])
        self.assertEqual(impl["candidates"]["b"]["full_pass_budget"], None)

    def test_fully_robust_winner(self):
        # Boundary at offset 0 with length 1: passes at every budget 1..max.
        search = [
            _entry("a", [_obs("f1", [4, 0, 0]), _obs("f2", [4, 1, 1])]),
            _entry("b", [_obs("f1", [0, 0, 0]), _obs("f2", [0, 0, 0])]),
        ]
        impl = self.assert_matches_oracle(search, (4,), 8, "compatible_on_fixture_set", "a")
        self.assertEqual(impl["decision_steps"], [[1, "selected", "a"]])
        self.assertEqual(impl["flip_ranges"], [])
        self.assertTrue(impl["budget_robust"])

    def test_all_ambiguous_everywhere(self):
        search = [
            _entry("a", [_obs("f1", [4, 0])]),
            _entry("b", [_obs("f1", [4, 1])]),
        ]
        impl = self.assert_matches_oracle(search, (4,), 8, "ambiguous", None)
        self.assertEqual(impl["decision_steps"], [[1, "ambiguous", None]])
        self.assertEqual(impl["flip_ranges"], [])
        self.assertTrue(impl["budget_robust"])

    def test_budgets_below_all_minima(self):
        search = [
            _entry("a", [_obs("f1", [0, 0, 0, 0, 0, 1, 2])]),
            _entry("b", [_obs("f1", [0, 0, 0, 0, 0, 0, 0])]),
        ]
        impl = self.assert_matches_oracle(search, (1, 2), 4, "no_candidate", None)
        self.assertEqual(impl["decision_steps"], [[1, "no_candidate", None]])
        self.assertEqual(impl["flip_ranges"], [])
        self.assertTrue(impl["budget_robust"])

    def test_min_budget_beyond_max_new_tokens_recorded(self):
        # Minimal budget 7 exceeds the cap 4: recorded, but never reached.
        search = [_entry("a", [_obs("f1", [0, 0, 0, 0, 0, 1, 2, 9])])]
        impl = self.assert_matches_oracle(search, (1, 2), 4, "no_candidate", None)
        candidate = impl["candidates"]["a"]
        self.assertEqual(candidate["min_budget_per_fixture"], [7])
        self.assertIsNone(candidate["full_pass_budget"])
        self.assertEqual(candidate["pass_steps"], [[1, 0]])

    def test_ambiguous_flip_ranges(self):
        # Both candidates pass everything from budget 4; native claims a unique winner.
        search = [
            _entry("a", [_obs("f1", [9, 9, 9, 1, 2])]),
            _entry("b", [_obs("f1", [8, 8, 8, 1, 2])]),
        ]
        impl = self.assert_matches_oracle(search, (1, 2), 8, "compatible_on_fixture_set", "a")
        self.assertEqual(
            impl["decision_steps"],
            [[1, "no_candidate", None], [5, "ambiguous", None]],
        )
        self.assertEqual(impl["flip_ranges"], [[1, 8]])
        self.assertFalse(impl["budget_robust"])

    def test_list_vs_tuple_entry_boundary_equivalence(self):
        observations = [_obs("f1", [3, 1, 2, 7]), _obs("f2", [1, 2])]
        list_form = [_entry("a", observations, boundary_token_ids=[1, 2])]
        tuple_form = [_entry("a", observations, boundary_token_ids=(1, 2))]
        kwargs = {
            "boundary": (9,),
            "max_new_tokens": 8,
            "native_decision": "no_candidate",
            "native_winner": None,
        }
        impl_list, oracle_list = _run(list_form, **kwargs)
        impl_tuple, oracle_tuple = _run(tuple_form, **kwargs)
        self.assertEqual(impl_list, impl_tuple)
        self.assertEqual(_compare(impl_list, oracle_list), [])
        self.assertEqual(_compare(impl_tuple, oracle_tuple), [])
        self.assertEqual(impl_list["candidates"]["a"]["min_budget_per_fixture"], [3, 2])

    def test_entry_boundary_overrides_argument(self):
        # Entry boundary (7,) at offset 0 wins; argument boundary (9,) never occurs.
        search = [_entry("a", [_obs("f1", [7, 7, 7])], boundary_token_ids=(7,))]
        impl = self.assert_matches_oracle(search, (9,), 8, "no_candidate", None)
        self.assertEqual(impl["candidates"]["a"]["min_budget_per_fixture"], [1])


# ---------------------------------------------------------------------------
# Input validation.
# ---------------------------------------------------------------------------


class BudgetFrontierValidationTests(unittest.TestCase):
    @staticmethod
    def _valid_search():
        return [_entry("a", [_obs("f1", [1, 2])])]

    @staticmethod
    def _valid_kwargs():
        return {
            "boundary": (1, 2),
            "max_new_tokens": 8,
            "native_decision": "no_candidate",
            "native_winner": None,
        }

    def _raises(self, search, **overrides):
        kwargs = self._valid_kwargs()
        kwargs.update(overrides)
        with self.assertRaises(ValueError):
            budget_frontier(search, **kwargs)

    def test_empty_search_rejected(self):
        self._raises([])

    def test_non_list_search_rejected(self):
        self._raises({"format": {"name": "a"}})

    def test_non_dict_entry_rejected(self):
        self._raises(["nope"])

    def test_missing_format_name_rejected(self):
        self._raises([{"format": {}, "observations": [_obs("f1", [1, 2])]}])

    def test_duplicate_candidate_names_rejected(self):
        search = [_entry("a", [_obs("f1", [1, 2])]), _entry("a", [_obs("f1", [1, 2])])]
        self._raises(search)

    def test_empty_observations_rejected(self):
        self._raises([_entry("a", [])])

    def test_non_list_observations_rejected(self):
        self._raises([{"format": {"name": "a"}, "observations": {}}])

    def test_non_dict_observation_rejected(self):
        self._raises([_entry("a", ["nope"])])

    def test_fixture_mismatch_across_entries_rejected(self):
        search = [
            _entry("a", [_obs("f1", [1, 2])]),
            _entry("b", [_obs("f2", [1, 2])]),
        ]
        self._raises(search)

    def test_empty_boundary_rejected(self):
        self._raises(self._valid_search(), boundary=())

    def test_non_integer_boundary_rejected(self):
        self._raises(self._valid_search(), boundary=(1, "x"))

    def test_empty_entry_boundary_rejected(self):
        search = [_entry("a", [_obs("f1", [1, 2])], boundary_token_ids=[])]
        self._raises(search)

    def test_bad_max_new_tokens_rejected(self):
        for bad in (0, 4097, "8", 8.0):
            with self.subTest(bad=bad):
                self._raises(self._valid_search(), max_new_tokens=bad)

    def test_invalid_native_decision_rejected(self):
        self._raises(self._valid_search(), native_decision="selected")

    def test_native_winner_required_for_selected_decision(self):
        self._raises(
            self._valid_search(),
            native_decision="compatible_on_fixture_set",
            native_winner=None,
        )

    def test_native_winner_must_name_a_candidate(self):
        self._raises(
            self._valid_search(),
            native_decision="held_out_failed",
            native_winner="ghost",
        )

    def test_native_winner_forbidden_without_selection(self):
        self._raises(
            self._valid_search(),
            native_decision="ambiguous",
            native_winner="a",
        )


# ---------------------------------------------------------------------------
# Sensitivity controls: prove the differential test catches real mistakes.
# ---------------------------------------------------------------------------


class BudgetFrontierSensitivityTests(unittest.TestCase):
    """Mutation controls.

    Each test corrupts the implementation's derivation in a way a real bug
    would (a +1 shift of the minimal-budget formula; an off-by-one in the
    pass counting behind ``decision_at``) and asserts the oracle
    comparison flags the difference. If these fail, the fuzz is blind.
    """

    @staticmethod
    def _passing_case():
        search = [
            _entry("a", [_obs("f1", [7, 1, 2, 9]), _obs("f2", [1, 2, 0, 0])]),
            _entry("b", [_obs("f1", [7, 7, 7, 7]), _obs("f2", [0, 0, 0, 0])]),
        ]
        kwargs = {
            "boundary": (1, 2),
            "max_new_tokens": 8,
            "native_decision": "compatible_on_fixture_set",
            "native_winner": "a",
        }
        return search, kwargs

    def test_unmutated_implementation_matches_oracle(self):
        # Sanity: the control case itself is clean before mutation.
        search, kwargs = self._passing_case()
        impl = budget_frontier(search, **kwargs)
        oracle = _oracle_budget_frontier(search, **kwargs)
        self.assertEqual(_compare(impl, oracle), [])

    def test_min_budget_plus_one_shift_is_caught(self):
        # Realistic bug: ``min_budget = cut + len(boundary) + 1``.
        real_first_boundary = experiments.first_boundary

        def shifted(tokens, stop):
            cut = real_first_boundary(tokens, stop)
            return None if cut is None else cut + 1

        search, kwargs = self._passing_case()
        with mock.patch.object(experiments, "first_boundary", side_effect=shifted):
            mutated = budget_frontier(search, **kwargs)
        oracle = _oracle_budget_frontier(search, **kwargs)
        mismatches = _compare(mutated, oracle)
        self.assertTrue(
            mismatches,
            "oracle failed to catch a +1 shift of the minimal-budget derivation",
        )
        self.assertTrue(
            any("min_budget_per_fixture" in mismatch for mismatch in mismatches),
            f"shift not visible in minimal budgets: {mismatches}",
        )

    def test_decision_count_off_by_one_is_caught(self):
        # Realistic bug: pass counts undercount by one (decision flips).
        real_bisect_right = experiments.bisect_right

        def undercount(sorted_mins, budget):
            return max(0, real_bisect_right(sorted_mins, budget) - 1)

        search, kwargs = self._passing_case()
        with mock.patch.object(experiments, "bisect_right", side_effect=undercount):
            mutated = budget_frontier(search, **kwargs)
        oracle = _oracle_budget_frontier(search, **kwargs)
        mismatches = _compare(mutated, oracle)
        self.assertTrue(
            mismatches,
            "oracle failed to catch an off-by-one in the decision pass counting",
        )
        self.assertTrue(
            any("decision_steps" in mismatch for mismatch in mismatches),
            f"off-by-one not visible in decision steps: {mismatches}",
        )


if __name__ == "__main__":
    unittest.main()
