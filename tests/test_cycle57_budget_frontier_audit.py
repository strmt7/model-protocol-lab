"""Cycle 57: adversarial re-verification of ``budget_frontier``.

Audited module: ``model_protocol_lab/experiments.py::budget_frontier``
(budget sweep, frontier-point detection, aggregation), which derives the
token-budget robustness frontier from recorded search observations using
``model_protocol_lab/core.py::first_boundary`` for the per-fixture minimal
passing budgets.

Method: every expected value in the hand-pinned cases below was derived by
hand from the docstring contract (``min_budget = cut + len(boundary)``; a
frontier step at budget N>1 means the outcome at N differs from N-1; budget
1 is always the initial point; ``flip_ranges`` are the maximal budget
intervals where the per-budget decision differs from the native decision),
*not* by running the implementation. A second, fully independent reference
implementation (naive subsequence scan, per-budget truncation re-evaluation,
no ``cut + len`` arithmetic, no bisect) cross-checks the implementation on
seeded fuzz. The mutation gate (run outside this file, against /tmp copies
of the package) applies ~6 mutants and requires every one to be caught.

REAL-bug status: none found. The implementation's flip-budget convention is
exact on all adversarial inputs tried (see the pinned cases). The one
deliberate sharp edge documented here: budget 0 is never evaluated -- the
sweep is 1..max_new_tokens, so a frontier point at budget 1 is an *initial*
point, not a flip relative to a budget-0 outcome. There is no budget-0
outcome; budget 1 with min_budget == 1 pins ``pass_steps == [[1, 1]]`` with
no redundant earlier point.
"""

import random
import unittest

from model_protocol_lab.experiments import budget_frontier

N_FUZZ_SEEDS = 60

_ABSENT = object()  # sentinel: argument not supplied


# ---------------------------------------------------------------------------
# Builders.
# ---------------------------------------------------------------------------


def _obs(fixture, output_ids=_ABSENT, **flags):
    """Obs."""
    obs = {"fixture": fixture, "status": "completed", "exact_match": True, "boundary_found": True}
    if output_ids is not _ABSENT:
        obs["output_ids"] = output_ids
    obs.update(flags)
    return obs


def _entry(name, observations, boundary_token_ids=_ABSENT):
    """Entry."""
    entry = {"format": {"name": name}, "observations": list(observations)}
    if boundary_token_ids is not _ABSENT:
        entry["boundary_token_ids"] = boundary_token_ids
    return entry


def _run(bf, search, *, boundary, max_new_tokens, native_decision, native_winner):
    """Run."""
    return bf(
        search,
        boundary=boundary,
        max_new_tokens=max_new_tokens,
        native_decision=native_decision,
        native_winner=native_winner,
    )


# ---------------------------------------------------------------------------
# Independent reference implementation (docstring contract only).
#
# Pass at budget b <=> truncating the recorded output to b tokens still
# satisfies the native pass predicate, re-evaluated naively on the truncated
# sequence. Minimal budgets are found by *scanning* budgets, never by
# ``cut + len(boundary)`` arithmetic. Frontier points are the budgets where
# the outcome differs from the previous budget (budget 1 is the initial
# point). This never shares code paths with the implementation under test.
# ---------------------------------------------------------------------------


def _ref_first_cut(tokens, boundary):
    """Naive first-occurrence scan; independent of core.first_boundary."""
    stop = list(boundary)
    toks = list(tokens)
    for i in range(len(toks) - len(stop) + 1):
        if toks[i : i + len(stop)] == stop:
            return i
    return None


def _ref_passes_at(obs, boundary, budget):
    """Ref passes at."""
    if obs.get("status") != "completed":
        return False
    if obs.get("exact_match") is not True:
        return False
    if obs.get("boundary_found") is not True:
        return False
    raw = obs.get("output_ids")
    if not isinstance(raw, (list, tuple)):
        return False
    ids = list(raw)
    if not ids or any(type(t) is not int for t in ids):
        return False
    cut = _ref_first_cut(ids[:budget], boundary)
    return cut is not None and cut + len(boundary) <= budget


def _ref_entry_boundary(entry, default):
    """Ref entry boundary."""
    eb = entry.get("boundary_token_ids")
    return default if eb is None else tuple(eb)


def _ref_min_budget(obs, boundary, max_new_tokens):
    """Ref min budget."""
    # The true minimal budget is recorded even beyond max_new_tokens (the
    # implementation derives it from the full recorded output), so the scan
    # runs past max_new_tokens up to the output length, which bounds it:
    # cut + len(boundary) <= len(output_ids) whenever a boundary occurs.
    raw = obs.get("output_ids")
    ids = list(raw) if isinstance(raw, (list, tuple)) else []
    for b in range(1, max(max_new_tokens, len(ids)) + 1):
        if _ref_passes_at(obs, boundary, b):
            return b
    return None


# skipcq: PY-R1000 - Gate-pinned frozen function; splitting breaks its verified contract.
def _ref_frontier(search, *, boundary, max_new_tokens, native_decision, native_winner):
    """Ref frontier."""
    names = [e["format"]["name"] for e in search]
    eb = {n: _ref_entry_boundary(e, boundary) for n, e in zip(names, search, strict=True)}
    obss = {n: e["observations"] for n, e in zip(names, search, strict=True)}
    n_fix = len(obss[names[0]])
    mins = {n: [_ref_min_budget(o, eb[n], max_new_tokens) for o in obss[n]] for n in names}

    def count(name, b):
        """Count."""
        return sum(1 for m in mins[name] if m is not None and m <= b)

    def decision(b):
        """Decision."""
        winners = [n for n in names if count(n, b) == n_fix]
        if len(winners) == 1:
            return ("selected", winners[0])
        if not winners:
            return ("no_candidate", None)
        return ("ambiguous", None)

    pass_steps = {
        n: [[1, count(n, 1)]]
        + [[b, count(n, b)] for b in range(2, max_new_tokens + 1) if count(n, b) != count(n, b - 1)]
        for n in names
    }
    dsteps = [[1, *decision(1)]] + [
        [b, *decision(b)] for b in range(2, max_new_tokens + 1) if decision(b) != decision(b - 1)
    ]
    native_search = {
        "compatible_on_fixture_set": ("selected", native_winner),
        "held_out_failed": ("selected", native_winner),
        "ambiguous": ("ambiguous", None),
        "no_candidate": ("no_candidate", None),
    }[native_decision]
    flips, cur = [], None
    for b in range(1, max_new_tokens + 1):
        if decision(b) != native_search:
            cur = [b, b] if cur is None else [cur[0], b]
        elif cur is not None:
            flips.append(cur)
            cur = None
    if cur is not None:
        flips.append(cur)
    full_pass = {
        n: next((b for b in range(1, max_new_tokens + 1) if count(n, b) == n_fix), None)
        for n in names
    }
    return {
        "mins": mins,
        "pass_steps": pass_steps,
        "decision_steps": dsteps,
        "flip_ranges": flips,
        "budget_robust": not flips,
        "full_pass_budget": full_pass,
        "count": count,
        "decision": decision,
        "n_fixtures": n_fix,
    }


# ---------------------------------------------------------------------------
# Shared checks, parameterized on the budget_frontier callable so the
# mutation gate can run the identical audit against /tmp mutant copies.
# ---------------------------------------------------------------------------


def _check_matches_reference(
    bf, search, *, boundary, max_new_tokens, native_decision, native_winner
):
    """Check matches reference."""
    impl = _run(
        bf,
        search,
        boundary=boundary,
        max_new_tokens=max_new_tokens,
        native_decision=native_decision,
        native_winner=native_winner,
    )
    ref = _ref_frontier(
        search,
        boundary=boundary,
        max_new_tokens=max_new_tokens,
        native_decision=native_decision,
        native_winner=native_winner,
    )
    assert impl["max_new_tokens"] == max_new_tokens
    for name in [e["format"]["name"] for e in search]:
        c = impl["candidates"][name]
        assert c["min_budget_per_fixture"] == ref["mins"][name], name
        assert c["pass_steps"] == ref["pass_steps"][name], name
        assert c["full_pass_budget"] == ref["full_pass_budget"][name], name
    assert impl["decision_steps"] == ref["decision_steps"]
    assert impl["flip_ranges"] == ref["flip_ranges"]
    assert impl["budget_robust"] == ref["budget_robust"]
    return impl, ref


# skipcq: PY-R1000 - Gate-pinned frozen function; splitting breaks its verified contract.
def _check_invariants(impl, ref, max_new_tokens):
    """Check invariants."""
    # pass_steps: sorted, strictly increasing, minimal (no redundant points),
    # first point always at budget 1.
    for name, c in impl["candidates"].items():
        steps = c["pass_steps"]
        assert steps and steps[0][0] == 1, name
        budgets = [b for b, _ in steps]
        assert budgets == sorted(budgets) and len(set(budgets)) == len(budgets), name
        assert all(1 <= b <= max_new_tokens for b in budgets), name
        counts = [v for _, v in steps]
        assert all(counts[i] != counts[i + 1] for i in range(len(counts) - 1)), name
        # Exact flip convention: for N > 1, step at N <=> count differs N-1 vs N.
        for b, v in steps:
            assert ref["count"](name, b) == v, (name, b)
            if b > 1:
                assert ref["count"](name, b - 1) != v, (name, b)
        for b in range(2, max_new_tokens + 1):
            if ref["count"](name, b) != ref["count"](name, b - 1):
                assert b in budgets, (name, b)
        # full_pass_budget is the smallest budget with all fixtures passing.
        fp = c["full_pass_budget"]
        if fp is None:
            assert all(
                ref["count"](name, b) < ref["n_fixtures"] for b in range(1, max_new_tokens + 1)
            ), name
        else:
            assert ref["count"](name, fp) == ref["n_fixtures"], name
            assert all(ref["count"](name, b) < ref["n_fixtures"] for b in range(1, fp)), name
    # decision_steps: same shape invariants; exact flip convention on decisions.
    dsteps = impl["decision_steps"]
    assert dsteps and dsteps[0][0] == 1
    dbudgets = [p[0] for p in dsteps]
    assert dbudgets == sorted(dbudgets) and len(set(dbudgets)) == len(dbudgets)
    assert all(1 <= b <= max_new_tokens for b in dbudgets)
    for i, point in enumerate(dsteps):
        dec = (point[1], point[2])
        assert ref["decision"](point[0]) == dec, point
        if i:
            assert dec != (dsteps[i - 1][1], dsteps[i - 1][2]), point
        if point[0] > 1:
            assert ref["decision"](point[0] - 1) != dec, point
    for b in range(2, max_new_tokens + 1):
        if ref["decision"](b) != ref["decision"](b - 1):
            assert b in dbudgets, b
    # flip_ranges: sorted, disjoint, non-adjacent (maximally merged), in range;
    # every budget flips iff covered; budget_robust iff no ranges.
    flips = impl["flip_ranges"]
    assert flips == sorted(flips)
    for lo, hi in flips:
        assert 1 <= lo <= hi <= max_new_tokens, (lo, hi)
    for (lo1, hi1), (lo2, _) in zip(flips, flips[1:], strict=False):
        assert hi1 < lo2 and hi1 + 1 < lo2, ((lo1, hi1), lo2)
    native = (impl["native_search_decision"], impl["native_winner"])
    for b in range(1, max_new_tokens + 1):
        covered = any(lo <= b <= hi for lo, hi in flips)
        assert covered == (ref["decision"](b) != native), b
    assert impl["budget_robust"] == (not flips)


def _case_a(bf):
    """Step function: single candidate, boundary completes at budget 4."""
    search = [_entry("a", [_obs("f1", (1, 2, 7, 8, 9))])]
    impl, ref = _check_matches_reference(
        bf,
        search,
        boundary=(7, 8),
        max_new_tokens=8,
        native_decision="compatible_on_fixture_set",
        native_winner="a",
    )
    c = impl["candidates"]["a"]
    assert c["min_budget_per_fixture"] == [4]
    assert c["pass_steps"] == [[1, 0], [4, 1]]
    assert c["full_pass_budget"] == 4
    assert impl["decision_steps"] == [[1, "no_candidate", None], [4, "selected", "a"]]
    assert impl["flip_ranges"] == [[1, 3]]
    assert impl["budget_robust"] is False
    _check_invariants(impl, ref, 8)


def _case_b(bf):
    """Flat region: nothing ever passes."""
    search = [_entry("a", [_obs("f1", (1, 2, 3, 4), exact_match=False, boundary_found=False)])]
    impl, ref = _check_matches_reference(
        bf,
        search,
        boundary=(7, 8),
        max_new_tokens=8,
        native_decision="no_candidate",
        native_winner=None,
    )
    c = impl["candidates"]["a"]
    assert c["min_budget_per_fixture"] == [None]
    assert c["pass_steps"] == [[1, 0]]
    assert c["full_pass_budget"] is None
    assert impl["decision_steps"] == [[1, "no_candidate", None]]
    assert impl["flip_ranges"] == []
    assert impl["budget_robust"] is True
    _check_invariants(impl, ref, 8)


def _case_c(bf):
    """Flip at budget 1: boundary completes within a single token.

    Budget 1 is the initial point, not a flip vs budget 0 (budget 0 is never
    evaluated): pass_steps has exactly one entry, no redundant point.
    """
    search = [_entry("a", [_obs("f1", (5, 6, 7))])]
    impl, ref = _check_matches_reference(
        bf,
        search,
        boundary=(5,),
        max_new_tokens=4,
        native_decision="compatible_on_fixture_set",
        native_winner="a",
    )
    c = impl["candidates"]["a"]
    assert c["min_budget_per_fixture"] == [1]
    assert c["pass_steps"] == [[1, 1]]
    assert c["full_pass_budget"] == 1
    assert impl["decision_steps"] == [[1, "selected", "a"]]
    assert impl["flip_ranges"] == []
    assert impl["budget_robust"] is True
    _check_invariants(impl, ref, 4)


def _case_d(bf):
    """Multiple flips, non-monotonic decision: no_candidate -> B -> ambiguous."""
    search = [
        _entry("A", [_obs("f1", (1, 7, 3)), _obs("f2", (1, 2, 3, 4, 5, 6, 8, 9, 7))]),
        _entry("B", [_obs("f1", (1, 2, 3, 4, 7)), _obs("f2", (1, 2, 3, 4, 5, 7))]),
    ]
    impl, ref = _check_matches_reference(
        bf,
        search,
        boundary=(7,),
        max_new_tokens=10,
        native_decision="ambiguous",
        native_winner=None,
    )
    assert impl["candidates"]["A"]["min_budget_per_fixture"] == [2, 9]
    assert impl["candidates"]["B"]["min_budget_per_fixture"] == [5, 6]
    assert impl["candidates"]["A"]["pass_steps"] == [[1, 0], [2, 1], [9, 2]]
    assert impl["candidates"]["B"]["pass_steps"] == [[1, 0], [5, 1], [6, 2]]
    assert impl["candidates"]["A"]["full_pass_budget"] == 9
    assert impl["candidates"]["B"]["full_pass_budget"] == 6
    assert impl["decision_steps"] == [
        [1, "no_candidate", None],
        [6, "selected", "B"],
        [9, "ambiguous", None],
    ]
    # Flip intervals [1,5] (no_candidate) and [6,8] (selected B) are adjacent,
    # so they merge into one maximal range; budgets 9-10 match native.
    assert impl["flip_ranges"] == [[1, 8]]
    assert impl["budget_robust"] is False
    _check_invariants(impl, ref, 10)


def _case_e(bf):
    """Singleton budget range (max_new_tokens=1)."""
    search = [
        _entry("A", [_obs("f1", (7,))]),
        _entry("B", [_obs("f1", (1,), exact_match=False, boundary_found=False)]),
    ]
    impl, ref = _check_matches_reference(
        bf,
        search,
        boundary=(7,),
        max_new_tokens=1,
        native_decision="compatible_on_fixture_set",
        native_winner="A",
    )
    assert impl["candidates"]["A"]["pass_steps"] == [[1, 1]]
    assert impl["candidates"]["B"]["pass_steps"] == [[1, 0]]
    assert impl["decision_steps"] == [[1, "selected", "A"]]
    assert impl["flip_ranges"] == []
    assert impl["budget_robust"] is True
    _check_invariants(impl, ref, 1)


def _case_f(bf):
    """Minimal budget beyond max_new_tokens is recorded but yields no step."""
    search = [_entry("a", [_obs("f1", (1, 2, 7, 8, 9))])]
    impl, ref = _check_matches_reference(
        bf,
        search,
        boundary=(7, 8),
        max_new_tokens=3,
        native_decision="no_candidate",
        native_winner=None,
    )
    c = impl["candidates"]["a"]
    assert c["min_budget_per_fixture"] == [4]
    assert c["pass_steps"] == [[1, 0]]
    assert c["full_pass_budget"] is None
    assert impl["decision_steps"] == [[1, "no_candidate", None]]
    assert impl["flip_ranges"] == []
    assert impl["budget_robust"] is True
    _check_invariants(impl, ref, 3)


def _case_g(bf):
    """First boundary occurrence counts; boundary at offset 0."""
    search = [_entry("a", [_obs("f1", (7, 7, 7)), _obs("f2", (9, 7, 9, 7))])]
    impl, ref = _check_matches_reference(
        bf,
        search,
        boundary=(7,),
        max_new_tokens=4,
        native_decision="compatible_on_fixture_set",
        native_winner="a",
    )
    c = impl["candidates"]["a"]
    assert c["min_budget_per_fixture"] == [1, 2]
    assert c["pass_steps"] == [[1, 1], [2, 2]]
    assert c["full_pass_budget"] == 2
    _check_invariants(impl, ref, 4)


def _case_h(bf):
    """Inconsistent flags and malformed outputs never pass and never crash."""
    search = [
        _entry(
            "a",
            [
                _obs("f1", (1, 7, 8), boundary_found=False),  # flag lie
                _obs("f2", (1, 7, 8), status="failed"),  # not completed
                _obs("f3"),  # missing output_ids
                _obs("f4", (1, "x", 7)),  # non-int token
                _obs("f5", ()),  # empty output
            ],
        )
    ]
    impl, ref = _check_matches_reference(
        bf,
        search,
        boundary=(7,),
        max_new_tokens=6,
        native_decision="no_candidate",
        native_winner=None,
    )
    c = impl["candidates"]["a"]
    assert c["min_budget_per_fixture"] == [None] * 5
    assert c["pass_steps"] == [[1, 0]]
    assert c["full_pass_budget"] is None
    assert impl["flip_ranges"] == []
    assert impl["budget_robust"] is True
    _check_invariants(impl, ref, 6)


def _case_i(bf):
    """Per-candidate boundary_token_ids overrides the boundary argument."""
    search = [
        _entry("A", [_obs("f1", (1, 7, 8))], boundary_token_ids=[7]),
        _entry("B", [_obs("f1", (1, 7, 8))], boundary_token_ids=(7,)),
    ]
    impl, ref = _check_matches_reference(
        bf,
        search,
        boundary=(7, 8),
        max_new_tokens=6,
        native_decision="ambiguous",
        native_winner=None,
    )
    # With the 1-token boundary the min budget is 1 + 1 = 2, not 1 + 2 = 3.
    assert impl["candidates"]["A"]["min_budget_per_fixture"] == [2]
    assert impl["candidates"]["B"]["min_budget_per_fixture"] == [2]
    assert impl["candidates"]["A"]["pass_steps"] == [[1, 0], [2, 1]]
    assert impl["decision_steps"] == [[1, "no_candidate", None], [2, "ambiguous", None]]
    # At budget 1 the decision is no_candidate, which differs from native
    # ambiguous: the first budget is itself a flip range.
    assert impl["flip_ranges"] == [[1, 1]]
    assert impl["budget_robust"] is False
    _check_invariants(impl, ref, 6)


def _case_j(bf):
    """Flip-flip-back with adjacent flip intervals merged into one range."""
    search = [
        _entry("A", [_obs("f1", (1, 2, 7))]),
        _entry("B", [_obs("f1", (1, 2, 3, 4, 7))]),
    ]
    impl, ref = _check_matches_reference(
        bf, search, boundary=(7,), max_new_tokens=8, native_decision="ambiguous", native_winner=None
    )
    assert impl["decision_steps"] == [
        [1, "no_candidate", None],
        [3, "selected", "A"],
        [5, "ambiguous", None],
    ]
    assert impl["flip_ranges"] == [[1, 4]]
    assert impl["budget_robust"] is False
    _check_invariants(impl, ref, 8)


def _case_k(bf):
    """held_out_failed native decision uses the winner for flip comparison."""
    search = [
        _entry("A", [_obs("f1", (1, 2, 7))]),
        _entry("B", [_obs("f1", (1, 2, 3, 4, 7))]),
    ]
    impl, ref = _check_matches_reference(
        bf,
        search,
        boundary=(7,),
        max_new_tokens=8,
        native_decision="held_out_failed",
        native_winner="B",
    )
    assert impl["native_search_decision"] == "selected"
    assert impl["native_winner"] == "B"
    # Native search decision is ("selected", "B"): flips where decision differs.
    assert impl["decision_steps"] == [
        [1, "no_candidate", None],
        [3, "selected", "A"],
        [5, "ambiguous", None],
    ]
    # Every budget 1..8 differs from native ("selected", "B"): no_candidate at
    # 1-2, selected A at 3-4, ambiguous at 5-8 -- one merged flip range.
    assert impl["flip_ranges"] == [[1, 8]]
    assert impl["budget_robust"] is False
    _check_invariants(impl, ref, 8)


def _fuzz_search(rng):
    """Fuzz search."""
    vocab = [1, 2, 3, 5, 7, 8, 9]
    boundaries = [(7,), (7, 8), (1, 2, 3)]
    n_candidates = rng.randint(1, 3)
    n_fixtures = rng.randint(1, 5)
    max_new_tokens = rng.randint(1, 20)
    boundary = rng.choice(boundaries)
    search = []
    for ci in range(n_candidates):
        name = f"cand{ci}"
        # Sometimes a per-candidate boundary (list/tuple/absent forms).
        eb = rng.choice([_ABSENT, _ABSENT, list(rng.choice(boundaries)), rng.choice(boundaries)])
        observations = []
        for fi in range(n_fixtures):
            out = [rng.choice(vocab) for _ in range(rng.randint(0, 12))]
            if rng.random() < 0.5 and out:
                pos = rng.randrange(len(out))
                bnd = tuple(eb) if eb is not _ABSENT else boundary
                out[pos:pos] = list(bnd)
            obs = {
                "fixture": f"f{fi}",
                "output_ids": out,
                "status": "completed" if rng.random() < 0.85 else "failed",
                "exact_match": rng.random() < 0.7,
                "boundary_found": rng.random() < 0.7,
            }
            if rng.random() < 0.05:
                del obs["output_ids"]
            observations.append(obs)
        search.append(_entry(name, observations, boundary_token_ids=eb))
    native_decision = rng.choice(
        ["compatible_on_fixture_set", "held_out_failed", "ambiguous", "no_candidate"]
    )
    if native_decision in ("compatible_on_fixture_set", "held_out_failed"):
        native_winner = f"cand{rng.randrange(n_candidates)}"
    else:
        native_winner = None
    return search, boundary, max_new_tokens, native_decision, native_winner


def _check_fuzz(bf, seeds):
    """Check fuzz."""
    for seed in range(seeds):
        rng = random.Random(10_000 + seed)  # noqa: S311 -- deterministic fuzz, not crypto
        search, boundary, max_new_tokens, native_decision, native_winner = _fuzz_search(rng)
        impl, ref = _check_matches_reference(
            bf,
            search,
            boundary=boundary,
            max_new_tokens=max_new_tokens,
            native_decision=native_decision,
            native_winner=native_winner,
        )
        _check_invariants(impl, ref, max_new_tokens)
        # Determinism: identical output on repeat call.
        again = _run(
            bf,
            search,
            boundary=boundary,
            max_new_tokens=max_new_tokens,
            native_decision=native_decision,
            native_winner=native_winner,
        )
        assert again == impl, seed


def run_audit_checks(bf, fuzz_seeds=N_FUZZ_SEEDS):
    """Run the full adversarial audit against ``bf``; raises on any mismatch.

    The mutation gate calls this with mutant ``budget_frontier`` callables
    and requires an AssertionError for every mutant.
    """
    _case_a(bf)
    _case_b(bf)
    _case_c(bf)
    _case_d(bf)
    _case_e(bf)
    _case_f(bf)
    _case_g(bf)
    _case_h(bf)
    _case_i(bf)
    _case_j(bf)
    _case_k(bf)
    _check_fuzz(bf, fuzz_seeds)


# ---------------------------------------------------------------------------
# unittest entry points.
# ---------------------------------------------------------------------------


class StepFunctionFrontierTests(unittest.TestCase):
    """Step function frontier tests."""

    @staticmethod
    def test_step_function_exact_points():
        """Test step function exact points."""
        _case_a(budget_frontier)

    @staticmethod
    def test_flat_region_no_flips():
        """Test flat region no flips."""
        _case_b(budget_frontier)

    @staticmethod
    def test_flip_at_budget_one_is_initial_point():
        """Test flip at budget one is initial point."""
        _case_c(budget_frontier)

    @staticmethod
    def test_multiple_flips_non_monotonic_decision():
        """Test multiple flips non monotonic decision."""
        _case_d(budget_frontier)

    @staticmethod
    def test_singleton_budget_range():
        """Test singleton budget range."""
        _case_e(budget_frontier)

    @staticmethod
    def test_min_budget_beyond_max_recorded_without_step():
        """Test min budget beyond max recorded without step."""
        _case_f(budget_frontier)

    @staticmethod
    def test_first_boundary_occurrence_counts():
        """Test first boundary occurrence counts."""
        _case_g(budget_frontier)

    @staticmethod
    def test_flag_lies_and_malformed_outputs_never_pass():
        """Test flag lies and malformed outputs never pass."""
        _case_h(budget_frontier)

    @staticmethod
    def test_per_candidate_boundary_override():
        """Test per candidate boundary override."""
        _case_i(budget_frontier)

    @staticmethod
    def test_flip_flip_back_ranges_merged():
        """Test flip flip back ranges merged."""
        _case_j(budget_frontier)

    @staticmethod
    def test_held_out_failed_native_winner_comparison():
        """Test held out failed native winner comparison."""
        _case_k(budget_frontier)


class FrontierFuzzTests(unittest.TestCase):
    """Frontier fuzz tests."""

    @staticmethod
    def test_reference_agreement_and_invariants_on_fuzz():
        """Test reference agreement and invariants on fuzz."""
        _check_fuzz(budget_frontier, N_FUZZ_SEEDS)


class FrontierValidationTests(unittest.TestCase):
    """Frontier validation tests."""

    def test_empty_search_rejected(self):
        """Test empty search rejected."""
        with self.assertRaises(ValueError):
            budget_frontier(
                [],
                boundary=(7,),
                max_new_tokens=8,
                native_decision="no_candidate",
                native_winner=None,
            )

    def test_zero_max_new_tokens_rejected(self):
        """Test zero max new tokens rejected."""
        search = [_entry("a", [_obs("f1", (7,))])]
        with self.assertRaises(ValueError):
            budget_frontier(
                search,
                boundary=(7,),
                max_new_tokens=0,
                native_decision="no_candidate",
                native_winner=None,
            )


if __name__ == "__main__":
    unittest.main()
