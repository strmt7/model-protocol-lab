"""Cycle 129 audit: the frontier's post-hoc decision rule is exactly the
real selection rule.

``budget_frontier`` re-derives, for every smaller token budget, what the
selection decision *would have been* -- via its local ``decision_at``:
unique full-passer wins, otherwise ``no_candidate`` / ``ambiguous``. Every
frontier artifact (decision steps, flip spans, safe floor, attribution,
and their diff-reports comparisons in cycles 125-128) rests on that rule
matching the real pipeline: ``run_experiment`` collects winners as the
candidates with ``summary["exact_and_boundary_passes"] == planned_count``
and decides via ``_select_winner`` (exactly one winner wins; ties and
failures abstain).

Audit result: the rules are exactly consistent within the frontier's
documented contract (a completed search decision). The proof rests on
three load-bearing invariants, each pinned below:

1. ``exact_match`` is gated on ``status == "completed"`` at the single
   ``Observation`` construction site (core.py) -- so the frontier's
   ``status == "completed" and exact_match and boundary_found`` pass
   predicate is exactly ``_summary``'s ``exact_match and boundary_found``.
   The frontier's explicit status check is defense-in-depth for foreign
   payloads: it can only under-count, never over-count, relative to a
   hand-edited record.
2. ``aggregate_eligible`` means a completed report covering every planned
   fixture -- so the frontier's ``n_fixtures`` (observation count) is
   exactly the pipeline's ``planned_count``, and every candidate in the
   contract was eligible for ranking.
3. A natively-passing observation's boundary completes inside its recorded
   output, so ``min_budget = cut + len(boundary) <= max_new_tokens`` --
   the native full-passer set is exactly the post-hoc full-passer set at
   the sweep maximum, i.e. ``decision_at(max)`` is the recorded native
   decision by construction, not by luck.

Tests pin: end-to-end agreement of the post-hoc decision at max budget
with the recorded native decision across winner / no-candidate /
ambiguous real runs; the construction invariant on real observations;
min-budget range; the foreign-payload status gate; and decision-step
coverage of the sweep maximum. A mutation gate (applied during
development, then reverted) proves the tests are not vacuous.
"""

import unittest

from model_protocol_lab import core
from model_protocol_lab.experiments import budget_frontier, run_experiment

BOUNDARY = (7,)
"""Single-token boundary; the backend emits chr(7) as the turn terminator."""
MAX_NEW_TOKENS = 32
CANDIDATES = ("cand-A", "cand-B")


class ScriptedBackend:
    """Deterministic stand-in for a checkpoint.

    Behavior per (candidate, fixture): "pass" emits the expected answer
    plus the boundary; "fail_answer" emits a wrong answer plus the
    boundary; "fail_boundary" emits the expected answer with no boundary;
    "error" makes the backend report a failure.
    The fixture is identified by its unique question text inside the
    rendered prompt, so no seed bookkeeping is needed.
    """

    fingerprint = "cycle129-consistency"

    def __init__(self, expected, behavior):
        self._expected = dict(expected)
        self._behavior = dict(behavior)

    @staticmethod
    def encode(text):
        return tuple(map(ord, text))

    @staticmethod
    def decode(token_ids):
        return "".join(map(chr, token_ids))

    def generate(self, input_ids, *, max_new_tokens, seed, cancel):
        prompt = self.decode(input_ids)
        candidate = next((name for name in CANDIDATES if f"[{name}]" in prompt), None)
        fixture = next((fid for fid in self._expected if f"question-{fid}" in prompt), None)
        if candidate is None or fixture is None:
            return core.Generation((), "error", "unknown_request")
        mode = self._behavior.get((candidate, fixture), "pass")
        if mode == "error":
            return core.Generation((), "error", "context_limit_exceeded")
        expected = self._expected[fixture]
        if mode == "pass":
            text = expected + chr(BOUNDARY[0])
        elif mode == "fail_answer":
            text = expected + "WRONG" + chr(BOUNDARY[0])
        elif mode == "fail_boundary":
            text = expected
        else:  # pragma: no cover -- test bug, not backend behavior
            raise AssertionError(f"unknown behavior mode {mode!r}")
        ids = self.encode(text)
        assert len(ids) <= max_new_tokens, "fixture too long for the token cap"
        # The backend contract: reason="limit" means exactly the cap was
        # generated. Pad past the boundary; the answer region (before the
        # first boundary) is unchanged.
        padded = ids + (1,) * (max_new_tokens - len(ids))
        return core.Generation(padded, "limit")


def _run(behavior, seed=20261019):
    fixtures = {
        "S1": "GREEN",
        "S2": "BLUE",
    }
    split = core.Split(
        search=tuple(
            core.Fixture(
                id=fid,
                messages=(core.Message(role="user", content=f"question-{fid}"),),
                expected=answer,
                group=f"group-{fid}",
            )
            for fid, answer in fixtures.items()
        ),
        held_out=tuple(
            core.Fixture(
                id="H1",
                messages=(core.Message(role="user", content="question-H1"),),
                expected="RED",
                group="group-H1",
            )
            for _ in range(1)
        ),
    )
    formats = tuple(
        core.Format(
            name=name,
            bos="",
            user_prefix=f"[{name}]U:",
            assistant_prefix="A:",
            turn_end="\n",
        )
        for name in CANDIDATES
    )
    expected = dict(fixtures)
    expected["H1"] = "RED"
    backend = ScriptedBackend(expected, behavior)
    return run_experiment(
        backend,
        formats,
        split,
        boundary=BOUNDARY,
        max_new_tokens=MAX_NEW_TOKENS,
        seed=seed,
    )


def _native_search_decision(result):
    """Map the recorded run decision into the frontier's decision space."""
    decision = result["decision"]
    if decision in ("compatible_on_fixture_set", "held_out_failed"):
        return ("selected", result["selected_candidate"])
    if decision == "ambiguous":
        return ("ambiguous", None)
    return ("no_candidate", None)


def _decision_at_max(frontier):
    """Re-derive decision_at(max_new_tokens) from the recorded steps."""
    max_tokens = frontier["max_new_tokens"]
    last = None
    for step in frontier["decision_steps"]:
        if step[0] <= max_tokens:
            last = (step[1], step[2])
    assert last is not None, "decision_steps always covers budget 1"
    return last


def _frontier_for(result):
    return budget_frontier(
        result["search"],
        boundary=BOUNDARY,
        max_new_tokens=MAX_NEW_TOKENS,
        native_decision=result["decision"],
        native_winner=result.get("selected_candidate"),
    )


class TestDecisionAtMatchesNativeDecision(unittest.TestCase):
    def test_winner_run(self):
        result = _run({("cand-B", "S2"): "fail_answer"})
        self.assertEqual(result["decision"], "compatible_on_fixture_set")
        self.assertEqual(result["selected_candidate"], "cand-A")
        frontier = _frontier_for(result)
        self.assertEqual(_decision_at_max(frontier), ("selected", "cand-A"))
        self.assertEqual(_decision_at_max(frontier), _native_search_decision(result))

    def test_no_candidate_run(self):
        result = _run({("cand-A", "S1"): "fail_answer", ("cand-B", "S2"): "fail_boundary"})
        self.assertEqual(result["decision"], "no_candidate")
        frontier = _frontier_for(result)
        self.assertEqual(_decision_at_max(frontier), ("no_candidate", None))
        self.assertEqual(_decision_at_max(frontier), _native_search_decision(result))

    def test_ambiguous_run(self):
        result = _run({})
        self.assertEqual(result["decision"], "ambiguous")
        frontier = _frontier_for(result)
        self.assertEqual(_decision_at_max(frontier), ("ambiguous", None))
        self.assertEqual(_decision_at_max(frontier), _native_search_decision(result))

    def test_held_out_failed_run(self):
        result = _run({("cand-B", "S2"): "fail_answer", ("cand-A", "H1"): "fail_answer"})
        self.assertEqual(result["decision"], "held_out_failed")
        self.assertEqual(result["selected_candidate"], "cand-A")
        frontier = _frontier_for(result)
        # The frontier models the *search* stage: the native search
        # decision is still ("selected", "cand-A").
        self.assertEqual(_decision_at_max(frontier), ("selected", "cand-A"))
        self.assertEqual(_decision_at_max(frontier), _native_search_decision(result))


class TestLoadBearingInvariants(unittest.TestCase):
    def test_exact_match_implies_completed(self):
        # Invariant 1: the construction-site gate. If this ever breaks,
        # the frontier's pass predicate and _summary's diverge.
        for behavior in (
            {},
            {("cand-B", "S2"): "fail_answer"},
            {("cand-A", "S1"): "fail_boundary"},
        ):
            result = _run(behavior)
            for entry in result["search"]:
                for obs in entry["observations"]:
                    if obs["exact_match"]:
                        self.assertEqual(
                            obs["status"],
                            "completed",
                            f"exact_match without completed status: {obs!r}",
                        )

    def test_empty_expected_error_stays_failed(self):
        # The sharp edge of invariant 1: a failed observation records
        # answer "", so with an empty-expected fixture the status gate is
        # the ONLY thing keeping exact_match False. Dropping the gate
        # (the "unknown behavior mode" mutant's cousin) would flip this.
        backend = ScriptedBackend({"E1": ""}, {("cand-A", "E1"): "error"})
        fmt = core.Format(
            name="cand-A",
            bos="",
            user_prefix="[cand-A]U:",
            assistant_prefix="A:",
            turn_end="\n",
        )
        case = core.Fixture(
            id="E1",
            messages=(core.Message(role="user", content="question-E1"),),
            expected="",
            group="group-E1",
        )
        report = core.evaluate(
            backend,
            fmt,
            (case,),
            boundary=BOUNDARY,
            max_new_tokens=MAX_NEW_TOKENS,
            seed=7,
        )
        self.assertEqual(report.status, "failed")
        (obs,) = report.observations
        self.assertEqual(obs.status, "failed")
        self.assertFalse(obs.exact_match)

    def test_min_budgets_within_sweep(self):
        # Invariant 3: a passing observation's boundary completes inside
        # its recorded output, so min_budget <= max_new_tokens.
        result = _run({("cand-B", "S2"): "fail_answer"})
        frontier = _frontier_for(result)
        for name, data in frontier["candidates"].items():
            for min_budget in data["min_budget_per_fixture"]:
                if min_budget is not None:
                    self.assertTrue(
                        1 <= min_budget <= MAX_NEW_TOKENS,
                        f"{name}: min_budget {min_budget} outside sweep",
                    )

    def test_winner_fully_passes_every_fixture(self):
        # Invariant 2 (eligibility): the native winner's post-hoc pass set
        # is complete -- no None entries, one per fixture.
        result = _run({("cand-B", "S2"): "fail_answer"})
        frontier = _frontier_for(result)
        winner = result["selected_candidate"]
        mins = frontier["candidates"][winner]["min_budget_per_fixture"]
        self.assertEqual(len(mins), 2)
        self.assertTrue(all(mb is not None for mb in mins))

    def test_foreign_status_gate_fails_closed(self):
        # Defense-in-depth: a foreign/hand-edited record claiming
        # exact_match with a non-completed status is NOT counted as a
        # pass by the frontier. (_summary never sees foreign records; it
        # runs on live Observation objects where the construction gate
        # makes this case unrepresentable.)
        search = [
            {
                "format": {"name": "X"},
                "observations": [
                    {
                        "fixture": "f0",
                        "status": "failed",
                        "exact_match": True,
                        "boundary_found": True,
                        "output_ids": [1, BOUNDARY[0]],
                    }
                ],
            }
        ]
        frontier = budget_frontier(
            search,
            boundary=BOUNDARY,
            max_new_tokens=MAX_NEW_TOKENS,
            native_decision="no_candidate",
            native_winner=None,
        )
        self.assertEqual(frontier["candidates"]["X"]["min_budget_per_fixture"], [None])
        self.assertEqual(_decision_at_max(frontier), ("no_candidate", None))

    def test_decision_steps_cover_sweep_maximum(self):
        for behavior in ({}, {("cand-B", "S2"): "fail_answer"}):
            result = _run(behavior)
            frontier = _frontier_for(result)
            steps = frontier["decision_steps"]
            self.assertTrue(steps, "decision_steps is never empty")
            self.assertLessEqual(steps[-1][0], MAX_NEW_TOKENS)
            # The post-hoc decision is constant from the last step onward,
            # so decision_at(max) is well-defined from the steps alone.
            self.assertEqual(_decision_at_max(frontier)[0], steps[-1][1])


if __name__ == "__main__":
    unittest.main()
