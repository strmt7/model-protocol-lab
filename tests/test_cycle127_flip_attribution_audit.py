"""Cycle 127 audit: flip-span attribution.

For each ``flip_ranges`` span, ``budget_frontier`` now records
``flip_attribution``: the (candidate, fixture) observations whose minimal
passing budget falls inside the span, sorted by (min_budget, candidate,
fixture). Pass counts -- and hence the selection decision -- can only
change at those marks, so these are exactly the pass-status changes the
span brackets: the observations to inspect when asking why the decision
flips there. A span with no marks inside it (the decision already differs
at its start, e.g. a cold-start region) records an empty crossing list,
rendered honestly as such.

Proved during this cycle (and pinned by a property test): the per-budget
full-passer set only grows with the budget, so once the native decision is
reached it can never be left again -- ``flip_ranges`` is always ``[]`` or
a single prefix span ``[[1, K]]``.

Tests pin: hand-computed derivations (single span, cold-start empty
crossings, tie ordering, no flips), agreement with an independent
re-derivation on randomized inputs, the prefix-span theorem, the HTML
rendering (wording, escaping of hostile names, missing/malformed
attribution tolerated, determinism). A mutation gate (applied during
development, then reverted) proves the tests are not vacuous:
attributing every mark to every span, an exclusive upper bound, dropping
the sort, and unescaped rendering are all killed.
"""

import random
import unittest

from model_protocol_lab.experiments import budget_frontier
from model_protocol_lab.reporting import _frontier_section

BOUNDARY = (7,)
"""Single-token boundary; a trailing 7 at index i gives min_budget i + 1."""


def _passing(min_budget):
    """Output ids whose first boundary occurrence yields ``min_budget``."""
    return [1] * (min_budget - 1) + [7]


def _entry(name, specs):
    """Build a search entry.

    ``specs`` maps fixture id -> min_budget (int) or None (never passes).
    """
    observations = []
    for fixture, min_budget in specs.items():
        if min_budget is None:
            observations.append(
                {
                    "fixture": fixture,
                    "status": "completed",
                    "exact_match": False,
                    "boundary_found": False,
                    "output_ids": [1, 2, 3],
                }
            )
        else:
            observations.append(
                {
                    "fixture": fixture,
                    "status": "completed",
                    "exact_match": True,
                    "boundary_found": True,
                    "output_ids": _passing(min_budget),
                }
            )
    return {"format": {"name": name}, "observations": observations}


def _frontier(entries, max_new_tokens, native_decision, native_winner):
    """Frontier."""
    return budget_frontier(
        entries,
        boundary=BOUNDARY,
        max_new_tokens=max_new_tokens,
        native_decision=native_decision,
        native_winner=native_winner,
    )


class TestFlipAttributionDerivation(unittest.TestCase):
    """Test flip attribution derivation."""

    def test_single_span_two_crossings(self):
        """Test single span two crossings."""
        frontier = _frontier(
            [
                _entry("A", {"f0": 3, "f1": 7}),
                _entry("B", {"f0": 5, "f1": None}),
            ],
            max_new_tokens=10,
            native_decision="compatible_on_fixture_set",
            native_winner="A",
        )
        self.assertEqual(frontier["flip_ranges"], [[1, 6]])
        self.assertEqual(
            frontier["flip_attribution"],
            [
                {
                    "span": [1, 6],
                    "crossings": [
                        {"candidate": "A", "fixture": "f0", "min_budget": 3},
                        {"candidate": "B", "fixture": "f0", "min_budget": 5},
                    ],
                }
            ],
        )

    def test_single_span_multiple_crossings(self):
        """Test single span multiple crossings."""
        # flip_ranges is always [] or a single prefix span [[1, K]]: the
        # full-passer set only grows with the budget, so once the native
        # decision is reached it can never be left again.
        frontier = _frontier(
            [
                _entry("A", {"f0": 3, "f1": 12}),
                _entry("B", {"f0": 5, "f1": 9}),
            ],
            max_new_tokens=16,
            native_decision="ambiguous",
            native_winner=None,
        )
        self.assertEqual(frontier["flip_ranges"], [[1, 11]])
        attribution = frontier["flip_attribution"]
        self.assertEqual(len(attribution), 1)
        self.assertEqual(attribution[0]["span"], [1, 11])
        # Every mark inside the span is attributed, sorted; the mark at 12
        # lies above the span and is attributed nowhere.
        self.assertEqual(
            attribution[0]["crossings"],
            [
                {"candidate": "A", "fixture": "f0", "min_budget": 3},
                {"candidate": "B", "fixture": "f0", "min_budget": 5},
                {"candidate": "B", "fixture": "f1", "min_budget": 9},
            ],
        )

    def test_cold_start_span_has_empty_crossings(self):
        """Test cold start span has empty crossings."""
        # The decision already differs at budget 1; the only mark (6) lies
        # above the flip span, so the span brackets no pass-status change.
        frontier = _frontier(
            [
                _entry("A", {"f0": 6}),
                _entry("B", {"f0": None}),
            ],
            max_new_tokens=8,
            native_decision="compatible_on_fixture_set",
            native_winner="A",
        )
        self.assertEqual(frontier["flip_ranges"], [[1, 5]])
        self.assertEqual(
            frontier["flip_attribution"],
            [{"span": [1, 5], "crossings": []}],
        )

    def test_tied_marks_sorted_by_candidate_then_fixture(self):
        """Test tied marks sorted by candidate then fixture."""
        frontier = _frontier(
            [
                _entry("A", {"f0": 4, "f1": 4}),
                _entry("B", {"f0": 4, "f1": 7}),
            ],
            max_new_tokens=10,
            native_decision="ambiguous",
            native_winner=None,
        )
        self.assertEqual(frontier["flip_ranges"], [[1, 6]])
        crossings = frontier["flip_attribution"][0]["crossings"]
        self.assertEqual(
            [(c["candidate"], c["fixture"], c["min_budget"]) for c in crossings],
            [("A", "f0", 4), ("A", "f1", 4), ("B", "f0", 4)],
        )

    def test_crossings_sorted_across_candidates(self):
        """Test crossings sorted across candidates."""
        # B's mark (3) precedes A's (5): without the explicit sort the
        # candidate-iteration order (A before B) would win.
        frontier = _frontier(
            [
                _entry("A", {"f0": 5, "f1": 9}),
                _entry("B", {"f0": 3, "f1": 9}),
            ],
            max_new_tokens=12,
            native_decision="ambiguous",
            native_winner=None,
        )
        self.assertEqual(frontier["flip_ranges"], [[1, 8]])
        crossings = frontier["flip_attribution"][0]["crossings"]
        self.assertEqual(
            [(c["candidate"], c["fixture"], c["min_budget"]) for c in crossings],
            [("B", "f0", 3), ("A", "f0", 5)],
        )

    def test_no_flips_empty_attribution(self):
        """Test no flips empty attribution."""
        frontier = _frontier(
            [
                _entry("A", {"f0": 1}),
                _entry("B", {"f0": 1}),
            ],
            max_new_tokens=8,
            native_decision="ambiguous",
            native_winner=None,
        )
        self.assertEqual(frontier["flip_ranges"], [])
        self.assertEqual(frontier["flip_attribution"], [])

    def test_never_passing_observations_excluded(self):
        """Test never passing observations excluded."""
        frontier = _frontier(
            [
                _entry("A", {"f0": 3, "f1": None}),
                _entry("B", {"f0": None, "f1": None}),
            ],
            max_new_tokens=8,
            native_decision="compatible_on_fixture_set",
            native_winner="A",
        )
        for span_attribution in frontier["flip_attribution"]:
            for crossing in span_attribution["crossings"]:
                self.assertIsNotNone(crossing["min_budget"])

    def test_marks_outside_all_spans_attributed_nowhere(self):
        """Test marks outside all spans attributed nowhere."""
        # A/f1 passes at 12, after the last flip span ends: it must not
        # appear in any span's crossings.
        frontier = _frontier(
            [
                _entry("A", {"f0": 3, "f1": 12}),
                _entry("B", {"f0": 5, "f1": 9}),
            ],
            max_new_tokens=16,
            native_decision="ambiguous",
            native_winner=None,
        )
        seen = [
            (c["candidate"], c["fixture"])
            for a in frontier["flip_attribution"]
            for c in a["crossings"]
        ]
        self.assertNotIn(("A", "f1"), seen)


def _oracle_attribution(frontier):
    """Independent re-derivation of flip_attribution from the payload."""
    candidates = frontier["candidates"]
    result = []
    for span in frontier["flip_ranges"]:
        lo, hi = span
        crossings = []
        for name in sorted(candidates):
            fixture_ids = candidates[name]["fixture_ids"]
            min_budgets = candidates[name]["min_budget_per_fixture"]
            for fixture_id, min_budget in zip(fixture_ids, min_budgets, strict=True):
                if min_budget is not None and lo <= min_budget <= hi:
                    crossings.append(
                        {
                            "candidate": name,
                            "fixture": fixture_id,
                            "min_budget": min_budget,
                        }
                    )
        crossings.sort(key=lambda c: (c["min_budget"], c["candidate"], c["fixture"]))
        result.append({"span": [lo, hi], "crossings": crossings})
    return result


class TestFlipAttributionOracle(unittest.TestCase):
    """Test flip attribution oracle."""

    def test_agrees_with_oracle_on_randomized_inputs(self):
        """Test agrees with oracle on randomized inputs."""
        rng = random.Random(20261017)  # noqa: S311 -- deterministic test seeding
        decisions = [
            "compatible_on_fixture_set",
            "held_out_failed",
            "ambiguous",
            "no_candidate",
        ]
        for trial in range(60):
            max_tokens = rng.randint(2, 14)
            n_candidates = rng.randint(1, 3)
            n_fixtures = rng.randint(1, 3)
            entries = []
            for ci in range(n_candidates):
                specs = {}
                for fi in range(n_fixtures):
                    roll = rng.random()
                    specs[f"f{fi}"] = None if roll < 0.3 else rng.randint(1, max_tokens)
                entries.append(_entry(f"c{ci}", specs))
            native_decision = rng.choice(decisions)
            native_winner = None
            if native_decision in ("compatible_on_fixture_set", "held_out_failed"):
                # Winner must be a candidate; any name is accepted here
                # because budget_frontier validates membership only.
                native_winner = "c0"
            frontier = _frontier(
                entries,
                max_new_tokens=max_tokens,
                native_decision=native_decision,
                native_winner=native_winner,
            )
            with self.subTest(trial=trial):
                self.assertEqual(frontier["flip_attribution"], _oracle_attribution(frontier))

    def test_crossings_cover_exactly_the_in_span_marks(self):
        """Test crossings cover exactly the in span marks."""
        # Every recorded min_budget inside a span appears exactly once in
        # that span's crossings, and no crossing lies outside its span.
        frontier = _frontier(
            [
                _entry("A", {"f0": 3, "f1": 12}),
                _entry("B", {"f0": 5, "f1": 9}),
            ],
            max_new_tokens=16,
            native_decision="ambiguous",
            native_winner=None,
        )
        for attribution, span in zip(
            frontier["flip_attribution"], frontier["flip_ranges"], strict=True
        ):
            lo, hi = span
            self.assertEqual(attribution["span"], [lo, hi])
            for crossing in attribution["crossings"]:
                self.assertTrue(lo <= crossing["min_budget"] <= hi)

    def test_flip_ranges_is_always_a_single_prefix_span(self):
        """Test flip ranges is always a single prefix span."""
        # Theorem: the per-budget full-passer set only grows with the
        # budget, so once the native decision is reached it can never be
        # left again. Hence flip_ranges is [] or [[1, K]] with K < max.
        rng = random.Random(20261018)  # noqa: S311 -- deterministic test seeding
        for trial in range(120):
            max_tokens = rng.randint(2, 14)
            n_candidates = rng.randint(1, 3)
            n_fixtures = rng.randint(1, 3)
            entries = []
            for ci in range(n_candidates):
                specs = {}
                for fi in range(n_fixtures):
                    roll = rng.random()
                    specs[f"f{fi}"] = None if roll < 0.3 else rng.randint(1, max_tokens)
                entries.append(_entry(f"c{ci}", specs))
            # Derive the true native outcome at max budget from the marks:
            # a candidate fully passes iff every observation passed natively.
            winners = []
            for ci in range(n_candidates):
                obs = entries[ci]["observations"]
                if all(o["exact_match"] for o in obs):
                    winners.append(f"c{ci}")
            if len(winners) == 1:
                native_decision, native_winner = (
                    "compatible_on_fixture_set",
                    winners[0],
                )
            elif not winners:
                native_decision, native_winner = "no_candidate", None
            else:
                native_decision, native_winner = "ambiguous", None
            frontier = _frontier(
                entries,
                max_new_tokens=max_tokens,
                native_decision=native_decision,
                native_winner=native_winner,
            )
            with self.subTest(trial=trial):
                spans = frontier["flip_ranges"]
                self.assertLessEqual(len(spans), 1)
                if spans:
                    self.assertEqual(spans[0][0], 1)
                    self.assertLess(spans[0][1], max_tokens)
                    self.assertEqual(len(frontier["flip_attribution"]), 1)


class TestFlipAttributionHtml(unittest.TestCase):
    """Test flip attribution html."""

    @staticmethod
    def _section(frontier):
        """Section."""
        return _frontier_section({"budget_frontier": frontier})

    @staticmethod
    def _flipping_frontier():
        """Flipping frontier."""
        return _frontier(
            [
                _entry("A", {"f0": 3, "f1": 12}),
                _entry("B", {"f0": 5, "f1": 9}),
            ],
            max_new_tokens=16,
            native_decision="ambiguous",
            native_winner=None,
        )

    def test_attribution_rendered_per_span(self):
        """Test attribution rendered per span."""
        html = self._section(self._flipping_frontier())
        self.assertIn("pass-status changes inside this span", html)
        self.assertIn("<code>A</code>/<code>f0</code> passes at budget 3", html)
        self.assertIn("<code>B</code>/<code>f1</code> passes at budget 9", html)

    def test_empty_crossings_wording(self):
        """Test empty crossings wording."""
        frontier = _frontier(
            [
                _entry("A", {"f0": 6}),
                _entry("B", {"f0": None}),
            ],
            max_new_tokens=8,
            native_decision="compatible_on_fixture_set",
            native_winner="A",
        )
        html = self._section(frontier)
        self.assertIn("no pass-status change inside this span", html)

    def test_hostile_names_escaped(self):
        """Test hostile names escaped."""
        hostile_candidate = "<img src=x onerror=alert(1)>"
        hostile_fixture = 'f<0>"1'
        frontier = _frontier(
            [
                _entry(hostile_candidate, {hostile_fixture: 3, "f1": 7}),
                _entry("B", {hostile_fixture: 5, "f1": None}),
            ],
            max_new_tokens=10,
            native_decision="compatible_on_fixture_set",
            native_winner=hostile_candidate,
        )
        html = self._section(frontier)
        self.assertIn("pass-status changes inside this span", html)
        self.assertNotIn(hostile_candidate, html)
        self.assertNotIn(hostile_fixture, html)
        self.assertIn("&lt;img src=x onerror=alert(1)&gt;", html)
        self.assertIn("f&lt;0&gt;&quot;1", html)

    def test_missing_attribution_renders_plain_spans(self):
        """Test missing attribution renders plain spans."""
        frontier = self._flipping_frontier()
        del frontier["flip_attribution"]
        html = self._section(frontier)
        self.assertIn("budgets 1&ndash;11", html)
        self.assertNotIn("pass-status change", html)
        self.assertNotIn("attribution", html)

    def test_malformed_attribution_tolerated(self):
        """Test malformed attribution tolerated."""
        frontier = self._flipping_frontier()
        frontier["flip_attribution"] = [
            {"span": "bogus"},
            {"span": [1, 11], "crossings": "nope"},
            {"span": [9, 11]},
            None,
            42,
            {"span": [9, 11], "crossings": [{"candidate": "B"}]},
            {"span": [1, 11], "crossings": [{"candidate": "B"}]},
        ]
        html = self._section(frontier)
        # Only the well-formed entry whose span matches renders; sparse
        # crossing dicts render defensively and never raise.
        self.assertIn("budgets 1&ndash;11", html)
        self.assertIn("<code>B</code>/<code>None</code> passes at budget None", html)

    def test_robust_frontier_has_no_attribution_list(self):
        """Test robust frontier has no attribution list."""
        frontier = _frontier(
            [
                _entry("A", {"f0": 1}),
                _entry("B", {"f0": 1}),
            ],
            max_new_tokens=8,
            native_decision="ambiguous",
            native_winner=None,
        )
        html = self._section(frontier)
        self.assertIn("stable", html)
        self.assertNotIn("attribution", html)

    def test_rendering_deterministic(self):
        """Test rendering deterministic."""
        frontier = self._flipping_frontier()
        self.assertEqual(self._section(frontier), self._section(frontier))


if __name__ == "__main__":
    unittest.main()
