"""Cycle 130 audit: held-out budget floor.

The token-budget frontier (cycles 121-129) answers "was the token cap
load-bearing?" for the *search* stage. This cycle completes the story for
the *held-out verification* stage: ``experiments.held_out_budget_floor``
recomputes, from the recorded held-out observations alone, the first
token budget from which the verification verdict holds unchanged -- no
model calls, by the same greedy prefix-determinism argument.

The held-out verdict is monotone in the budget (a natively-passing
observation passes at every budget >= its minimal budget and at no
smaller one), so it flips at most once, at the maximum of the per-fixture
minimal budgets. The payload records ``safe_budget_floor`` (that maximum,
or None when the held-out failed natively -- no budget achieves a passing
verification), ``budget_robust`` (verdict identical at every budget), and
the per-fixture minimal budgets. ``mpl run`` attaches it like the search
frontier, the held-out HTML section calls it out, and ``mpl diff-reports``
compares it between runs.

This cycle also extracts the shared ``_observation_min_budget`` helper
used by both ``budget_frontier`` and the new function; the existing
oracle suites (cycles 34/69) prove the extraction changed nothing.

Tests pin: the exact derivation (floor/robustness/verdict matrix),
input validation and fail-closed inconsistency checks, the shared
helper, end-to-end agreement on a real run, the HTML callout, and the
diff-reports wiring. A mutation gate (applied during development, then
reverted) proves the tests are not vacuous.
"""

import unittest

from model_protocol_lab import core
from model_protocol_lab.diffreports import (
    diff_held_out_floor,
    diff_reports,
    render_summary,
)
from model_protocol_lab.experiments import _observation_min_budget, held_out_budget_floor
from model_protocol_lab.reporting import _held_out_floor_callout

BOUNDARY = (7,)
MAX_NEW_TOKENS = 32


def _obs(fixture, min_budget):
    """A recorded held-out observation with the given minimal budget."""
    if min_budget is None:
        return {
            "fixture": fixture,
            "status": "completed",
            "exact_match": False,
            "boundary_found": False,
            "output_ids": [1, 2, 3],
        }
    return {
        "fixture": fixture,
        "status": "completed",
        "exact_match": True,
        "boundary_found": True,
        "output_ids": [1] * (min_budget - 1) + [BOUNDARY[0]],
    }


def _floor_payload(min_budgets, verdict="compatible_on_fixture_set", max_tokens=MAX_NEW_TOKENS):
    observations = [_obs(f"H{i}", mb) for i, mb in enumerate(min_budgets)]
    return held_out_budget_floor(
        observations,
        boundary=BOUNDARY,
        max_new_tokens=max_tokens,
        native_verdict=verdict,
    )


class TestHeldOutBudgetFloor(unittest.TestCase):
    def test_floor_is_max_of_min_budgets(self):
        payload = _floor_payload([3, 7, 5])
        self.assertEqual(payload["safe_budget_floor"], 7)
        self.assertFalse(payload["budget_robust"])
        self.assertEqual(payload["native_verdict"], "compatible_on_fixture_set")
        self.assertEqual(payload["min_budget_per_fixture"], [3, 7, 5])
        self.assertEqual(payload["fixture_ids"], ["H0", "H1", "H2"])
        self.assertEqual(payload["max_new_tokens"], MAX_NEW_TOKENS)

    def test_robust_when_everything_passes_at_one(self):
        payload = _floor_payload([1, 1])
        self.assertEqual(payload["safe_budget_floor"], 1)
        self.assertTrue(payload["budget_robust"])

    def test_failed_native_has_no_floor(self):
        payload = _floor_payload([3, None], verdict="held_out_failed")
        self.assertIsNone(payload["safe_budget_floor"])
        # A natively-failing fixture can never pass at a smaller budget,
        # so the failed verdict holds at every budget in the sweep.
        self.assertTrue(payload["budget_robust"])
        self.assertEqual(payload["min_budget_per_fixture"], [3, None])

    def test_single_fixture(self):
        payload = _floor_payload([4])
        self.assertEqual(payload["safe_budget_floor"], 4)
        self.assertFalse(payload["budget_robust"])

    def test_rejects_empty_observations(self):
        with self.assertRaises(ValueError):
            held_out_budget_floor(
                [],
                boundary=BOUNDARY,
                max_new_tokens=MAX_NEW_TOKENS,
                native_verdict="compatible_on_fixture_set",
            )

    def test_rejects_bad_boundary(self):
        with self.assertRaises(ValueError):
            held_out_budget_floor(
                [_obs("H0", 3)],
                boundary=(),
                max_new_tokens=MAX_NEW_TOKENS,
                native_verdict="compatible_on_fixture_set",
            )

    def test_rejects_bad_max_tokens(self):
        with self.assertRaises(ValueError):
            held_out_budget_floor(
                [_obs("H0", 3)],
                boundary=BOUNDARY,
                max_new_tokens=0,
                native_verdict="compatible_on_fixture_set",
            )

    def test_rejects_bad_verdict(self):
        with self.assertRaises(ValueError):
            held_out_budget_floor(
                [_obs("H0", 3)],
                boundary=BOUNDARY,
                max_new_tokens=MAX_NEW_TOKENS,
                native_verdict="ambiguous",
            )

    def test_rejects_non_dict_observation(self):
        with self.assertRaises(ValueError):
            held_out_budget_floor(
                ["nope"],
                boundary=BOUNDARY,
                max_new_tokens=MAX_NEW_TOKENS,
                native_verdict="compatible_on_fixture_set",
            )

    def test_rejects_inconsistent_compatible(self):
        with self.assertRaises(ValueError):
            held_out_budget_floor(
                [_obs("H0", 3), _obs("H1", None)],
                boundary=BOUNDARY,
                max_new_tokens=MAX_NEW_TOKENS,
                native_verdict="compatible_on_fixture_set",
            )

    def test_rejects_inconsistent_failed(self):
        with self.assertRaises(ValueError):
            held_out_budget_floor(
                [_obs("H0", 3)],
                boundary=BOUNDARY,
                max_new_tokens=MAX_NEW_TOKENS,
                native_verdict="held_out_failed",
            )

    def test_deterministic(self):
        observations = [_obs("H0", 3), _obs("H1", 7)]
        first = held_out_budget_floor(
            observations,
            boundary=BOUNDARY,
            max_new_tokens=MAX_NEW_TOKENS,
            native_verdict="compatible_on_fixture_set",
        )
        second = held_out_budget_floor(
            observations,
            boundary=BOUNDARY,
            max_new_tokens=MAX_NEW_TOKENS,
            native_verdict="compatible_on_fixture_set",
        )
        self.assertEqual(first, second)


class TestObservationMinBudget(unittest.TestCase):
    def test_passing_observation(self):
        obs = _obs("H0", 6)
        self.assertEqual(_observation_min_budget(obs, BOUNDARY), 6)

    def test_non_completed_excluded(self):
        obs = _obs("H0", 6)
        obs["status"] = "failed"
        self.assertIsNone(_observation_min_budget(obs, BOUNDARY))

    def test_exact_false_excluded(self):
        obs = _obs("H0", None)
        self.assertIsNone(_observation_min_budget(obs, BOUNDARY))

    def test_malformed_output_ids_excluded(self):
        for bad in ([], ["x"], [1, "x"], None):
            obs = _obs("H0", 6)
            obs["output_ids"] = bad
            self.assertIsNone(_observation_min_budget(obs, BOUNDARY), f"output_ids={bad!r}")

    def test_missing_boundary_excluded(self):
        obs = _obs("H0", 6)
        obs["output_ids"] = [1, 2, 3]
        self.assertIsNone(_observation_min_budget(obs, BOUNDARY))

    def test_non_dict_excluded(self):
        self.assertIsNone(_observation_min_budget("nope", BOUNDARY))


class ScriptedBackend:
    """Minimal deterministic backend for the end-to-end test."""

    fingerprint = "cycle130-held-out-floor"

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
        candidate = next((n for n in ("cand-A", "cand-B") if f"[{n}]" in prompt), None)
        fixture = next((fid for fid in self._expected if f"question-{fid}" in prompt), None)
        if candidate is None or fixture is None:
            return core.Generation((), "error", "unknown_request")
        mode = self._behavior.get((candidate, fixture), "pass")
        expected = self._expected[fixture]
        # Any non-"pass" mode fails: wrong answer, no boundary.
        text = expected + chr(BOUNDARY[0]) if mode == "pass" else expected + "WRONG"
        ids = self.encode(text)
        assert len(ids) <= max_new_tokens
        return core.Generation(ids + (1,) * (max_new_tokens - len(ids)), "limit")


def _real_run():
    from model_protocol_lab.experiments import run_experiment

    fixtures = {"S1": "GREEN", "S2": "BLUE", "H1": "RED", "H2": "CYAN"}
    split = core.Split(
        search=tuple(
            core.Fixture(
                id=fid,
                messages=(core.Message(role="user", content=f"question-{fid}"),),
                expected=answer,
                group=f"group-{fid}",
            )
            for fid, answer in (("S1", "GREEN"), ("S2", "BLUE"))
        ),
        held_out=tuple(
            core.Fixture(
                id=fid,
                messages=(core.Message(role="user", content=f"question-{fid}"),),
                expected=answer,
                group=f"group-{fid}",
            )
            for fid, answer in (("H1", "RED"), ("H2", "CYAN"))
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
        for name in ("cand-A", "cand-B")
    )
    backend = ScriptedBackend(fixtures, {("cand-B", "S2"): "fail"})
    return run_experiment(
        backend,
        formats,
        split,
        boundary=BOUNDARY,
        max_new_tokens=MAX_NEW_TOKENS,
        seed=20261020,
    )


class TestHeldOutFloorEndToEnd(unittest.TestCase):
    def test_real_run_floor(self):
        result = _real_run()
        self.assertEqual(result["decision"], "compatible_on_fixture_set")
        self.assertEqual(result["selected_candidate"], "cand-A")
        payload = held_out_budget_floor(
            result["held_out"]["observations"],
            boundary=tuple(result["selected_boundary_token_ids"]),
            max_new_tokens=result["max_new_tokens"],
            native_verdict=result["decision"],
        )
        # "RED" + boundary -> 4, "CYAN" + boundary -> 5.
        self.assertEqual(payload["min_budget_per_fixture"], [4, 5])
        self.assertEqual(payload["safe_budget_floor"], 5)
        self.assertFalse(payload["budget_robust"])


class TestHeldOutFloorHtml(unittest.TestCase):
    def test_robust_callout(self):
        html = _held_out_floor_callout(_floor_payload([1, 1]))
        self.assertIn("verification robust", html)
        self.assertIn("badge pass", html)

    def test_floor_callout(self):
        html = _held_out_floor_callout(_floor_payload([3, 7]))
        self.assertIn("verification floor", html)
        self.assertIn("&ge; 7 tokens", html)
        self.assertIn("from budget 7 through 32", html)

    def test_failed_callout(self):
        html = _held_out_floor_callout(_floor_payload([3, None], verdict="held_out_failed"))
        self.assertIn("verification failed", html)
        self.assertIn("badge fail", html)

    def test_unavailable_marker_renders_nothing(self):
        self.assertEqual(_held_out_floor_callout({"available": False, "detail": "x"}), "")

    def test_malformed_renders_nothing(self):
        for bad in (None, "x", 42, {}, {"native_verdict": "bogus"}):
            self.assertEqual(_held_out_floor_callout(bad), "")

    def test_deterministic(self):
        payload = _floor_payload([3, 7])
        self.assertEqual(_held_out_floor_callout(payload), _held_out_floor_callout(payload))


def _diff_base(floor_payload):
    return {
        "decision": "compatible_on_fixture_set",
        "selected_candidate": "cand-A",
        "search": [],
        "budget_frontier": {"available": False},
        "held_out_budget_floor": floor_payload,
    }


class TestHeldOutFloorDiff(unittest.TestCase):
    def test_identical_not_changed(self):
        payload = _floor_payload([3, 7])
        diff = diff_held_out_floor(payload, payload)
        self.assertFalse(diff["changed"])
        self.assertEqual(diff["available"], "both")

    def test_different_floors_changed(self):
        old = _floor_payload([3, 7])
        new = _floor_payload([3, 9])
        diff = diff_held_out_floor(old, new)
        self.assertTrue(diff["changed"])
        self.assertTrue(diff["safe_budget_floor"]["changed"])
        self.assertEqual(diff["safe_budget_floor"]["old"], 7)
        self.assertEqual(diff["safe_budget_floor"]["new"], 9)

    def test_robustness_flip_changed(self):
        old = _floor_payload([1, 1])
        new = _floor_payload([1, 7])
        diff = diff_held_out_floor(old, new)
        self.assertTrue(diff["changed"])
        self.assertTrue(diff["budget_robust"]["changed"])

    def test_one_sided_missing(self):
        old = {"available": False, "detail": "x"}
        new = _floor_payload([3, 7])
        diff = diff_held_out_floor(old, new)
        self.assertEqual(diff["available"], "new")
        self.assertFalse(diff["changed"])

    def test_neither_available(self):
        diff = diff_held_out_floor({"available": False}, {"available": False, "detail": "y"})
        self.assertEqual(diff["available"], "neither")
        self.assertFalse(diff["changed"])

    def test_malformed_payload_not_comparable(self):
        diff = diff_held_out_floor({"safe_budget_floor": "x"}, _floor_payload([3]))
        self.assertEqual(diff["available"], "new")

    def test_changed_feeds_identical(self):
        old = _diff_base(_floor_payload([3, 7]))
        same = _diff_base(_floor_payload([3, 7]))
        report = diff_reports(old, same, old_label="a", new_label="b")
        self.assertTrue(report["summary"]["identical"])
        new = _diff_base(_floor_payload([3, 9]))
        report = diff_reports(old, new, old_label="a", new_label="b")
        self.assertFalse(report["summary"]["identical"])
        self.assertTrue(report["summary"]["held_out_floor_changed"])

    def test_summary_lines(self):
        old = _diff_base(_floor_payload([3, 7]))
        new = _diff_base(_floor_payload([3, 9]))
        text = render_summary(diff_reports(old, new, old_label="a", new_label="b"))
        self.assertIn("Held-out verification floor: 7 -> 9", text)
        self.assertIn("Held-out verification budget-robust: no -> no", text)

    def test_summary_line_missing_side(self):
        old = _diff_base({"available": False, "detail": "x"})
        new = _diff_base(_floor_payload([3, 7]))
        text = render_summary(diff_reports(old, new, old_label="a", new_label="b"))
        self.assertIn(
            "Held-out budget floor comparison unavailable: old run did not record one.", text
        )

    def test_diff_deterministic(self):
        old = _floor_payload([3, 7])
        new = _floor_payload([3, 9])
        self.assertEqual(diff_held_out_floor(old, new), diff_held_out_floor(old, new))


if __name__ == "__main__":
    unittest.main()
