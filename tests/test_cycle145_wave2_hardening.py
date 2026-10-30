"""Cycle 145: wave-2 adversarial-audit hardening.

Every finding below was reported by the wave-2 specialist team (audited
at 751468fdc9a1) and re-verified against the current tree before fixing:

- C1 (HIGH): ``margin_trajectory`` was O(B*C) -- the wave-1 point cap
  bounds total points, not candidates, so C candidates x 1 distinct point
  each meant C**2 work (~1.4s at C=2000, projecting to hours on a
  hostile few-MB payload). Restructured to an O((E+B) log (E+C)) event
  sweep with a lazily-deleted leader heap.
- C2: ``old_decision``/``new_decision`` were interpolated raw into the
  stderr summary by the transfer, robustness, and perturbation drift
  renderers, falsifying ``render_summary``'s sanitization comment.
- C3: non-string decision/winner/tier values crashed ``render_summary``
  with TypeError (``_display_id`` did regex-sub on non-str), contradicting
  the degrade-gracefully contract. ``_display_id`` is now total.
- C4/C5: ``diff_transfer``/``diff_contamination`` ``changed`` flags had
  unpinned disjuncts (cell-drift-only, tier-escalation-only); the same gap
  class existed in the new ``diff_perturbation`` (stability-only,
  variant-drift-only, removed-only).
- C6: ``_render_transfer_summary`` printed ``checkpoint_drift`` uncapped;
  now capped at 10 like ``cell_drift``.
"""

from __future__ import annotations

import random
import time
import unittest
from bisect import bisect_right
from typing import Any

from model_protocol_lab import diffreports
from model_protocol_lab.reporting import (
    _fixture_total,
    _frontier_series,
    margin_trajectory,
)


def _quadratic_reference(frontier: dict[str, Any]) -> list[tuple[int, Any, int]]:
    """The pre-C1 O(B*C) sweep, kept as a semantic oracle.

    Independent reimplementation of the old algorithm (per-budget loop
    over all candidates with bisect point lookups); the differential
    test below proves the event-sweep restructure did not change the
    emitted change points on 1400 seeded frontiers.
    """
    validated = _frontier_series(frontier)
    if validated is None:
        return []
    names, max_tokens = validated
    candidates = frontier["candidates"]
    steps_of: dict[str, list[tuple[int, int]]] = {}
    keys_of: dict[str, list[int]] = {}
    for name in names:
        data = candidates[name]
        by_budget: dict[int, int] = {}
        for point in data.get("pass_steps") or []:
            if (
                type(point) is list
                and len(point) == 2
                and type(point[0]) is int
                and type(point[1]) is int
                and 1 <= point[0] <= max_tokens
                and 0 <= point[1] <= _fixture_total(data)
            ):
                by_budget[point[0]] = point[1]
        steps_of[name] = sorted(by_budget.items())
        keys_of[name] = [b for b, _ in steps_of[name]]

    def _count_at(steps: list[tuple[int, int]], keys: list[int], budget: int) -> int:
        """Count at."""
        index = bisect_right(keys, budget) - 1
        return steps[index][1] if index >= 0 else 0

    change_budgets = sorted({b for ordered in steps_of.values() for b, _ in ordered})
    points: list[tuple[int, Any, int]] = []
    for budget in change_budgets:
        counts = {n: _count_at(steps_of[n], keys_of[n], budget) for n in names}
        top = max(counts.values())
        leaders = [n for n in names if counts[n] == top]
        if len(leaders) == 1:
            leader: Any = leaders[0]
            rest = [c for n, c in counts.items() if n != leader]
            margin = top - max(rest) if rest else 0
        else:
            leader, margin = None, 0
        if points and points[-1][1] == leader and points[-1][2] == margin:
            continue
        points.append((budget, leader, margin))
    return points


def _many_candidate_frontier(count: int) -> dict[str, Any]:
    """Many candidate frontier."""
    # The wave-2 C1 hostile shape: many candidates, one distinct point
    # each -- passes the total-point cap with B == C.
    return {
        "max_new_tokens": count,
        "candidates": {
            f"cand{i:05d}": {
                "fixture_ids": ["f1"],
                "pass_steps": [[i + 1, 1]],
            }
            for i in range(count)
        },
    }


class TestMarginTrajectoryComplexity(unittest.TestCase):
    """Test margin trajectory complexity."""

    def test_hostile_candidate_count_is_fast(self) -> None:
        """Test hostile candidate count is fast."""
        # C1: C=2000 x 1 point took ~1.4s before the restructure (O(C^2));
        # the event sweep does it in milliseconds. Generous bound.
        start = time.perf_counter()
        points = margin_trajectory(_many_candidate_frontier(2000))
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 10, f"margin_trajectory took {elapsed:.2f}s at C=2000")
        self.assertTrue(points)

    def test_scales_subquadratically(self) -> None:
        """Test scales subquadratically."""
        # Doubling the hostile shape must not quadruple the time.
        timings = []
        for count in (400, 800):
            frontier = _many_candidate_frontier(count)
            start = time.perf_counter()
            margin_trajectory(frontier)
            timings.append(time.perf_counter() - start)
        small, large = timings
        if small < 1e-3:
            self.skipTest(f"timer resolution too coarse ({small:.2e}s)")
        ratio = large / small
        self.assertLess(ratio, 3.0, f"scaling ratio {ratio:.2f} looks quadratic")

    def test_differential_against_quadratic_reference(self) -> None:
        """Test differential against quadratic reference."""
        # 1400 seeded frontiers incl. count revisits, decreases, ties,
        # duplicate budgets, and empty candidates: the restructure must
        # emit exactly the old change points.
        mismatches = 0
        for seed in (20261018, 1, 999):
            rng = random.Random(seed)  # noqa: S311 -- deterministic test seed
            for _ in range(400):
                n_cand = rng.randint(1, 8)
                max_tok = rng.randint(1, 40)
                candidates = {}
                for i in range(n_cand):
                    npts = rng.randint(0, 10)
                    pts = sorted(
                        {(rng.randint(1, max_tok), rng.randint(0, 6)) for _ in range(npts)}
                    )
                    candidates[f"c{i:02d}"] = {
                        "fixture_ids": ["f1"] * 7,
                        "pass_steps": [[b, c] for b, c in pts],
                    }
                frontier = {"max_new_tokens": max_tok, "candidates": candidates}
                if margin_trajectory(frontier) != _quadratic_reference(frontier):
                    mismatches += 1
        self.assertEqual(mismatches, 0)

    def test_differential_dense_revisits(self) -> None:
        """Test differential dense revisits."""
        # Adversarial for the heap: every budget changes every
        # candidate, with frequent count revisits and decreases.
        rng = random.Random(7)  # noqa: S311 -- deterministic test seed
        for _ in range(200):
            n_cand = rng.randint(2, 5)
            max_tok = 50
            candidates = {}
            for i in range(n_cand):
                pts = [[b, rng.randint(0, 6)] for b in range(1, max_tok + 1, rng.randint(1, 4))]
                candidates[f"c{i:02d}"] = {"fixture_ids": ["f1"] * 7, "pass_steps": pts}
            frontier = {"max_new_tokens": max_tok, "candidates": candidates}
            self.assertEqual(margin_trajectory(frontier), _quadratic_reference(frontier))


class TestDisplayIdTotal(unittest.TestCase):
    """Test display id total."""

    def test_non_string_values_render_without_crash(self) -> None:
        """Test non string values render without crash."""
        # C3: a hand-edited results.json may carry a dict winner, a list
        # tier, or an int decision; the summary must degrade, not crash.
        self.assertEqual(diffreports._display_id({"a": 1}), "{'a': 1}")
        self.assertEqual(diffreports._display_id(["x", "y"]), "['x', 'y']")
        self.assertEqual(diffreports._display_id(42), "42")
        self.assertEqual(diffreports._display_id(True), "True")
        self.assertEqual(diffreports._display_id(None), "(none)")
        self.assertEqual(diffreports._display_id("plain"), "plain")

    def test_control_chars_stripped_from_coerced_values(self) -> None:
        """Test control chars stripped from coerced values."""
        self.assertEqual(diffreports._display_id("a\x1b[2Jb"), "a[2Jb")
        self.assertNotIn("\x1b", diffreports._display_id(["\x1b[2J"]))
        # Lone surrogates are stripped, not passed through to the stream.
        self.assertEqual(diffreports._display_id("\ud800ok"), "ok")


def _hostile_decision_payload(decision: Any) -> dict[str, Any]:
    """Hostile decision payload."""
    return {
        "available": True,
        "primary_winner": "alpha",
        "transfer_verdict": "partial",
        "per_checkpoint": {"ckptA": {"decision": decision, "winner": "alpha"}},
        "matrix": {"alpha": {"ckptA": "transfers"}},
    }


class TestDecisionSanitization(unittest.TestCase):
    """Test decision sanitization."""

    @staticmethod
    def _render(diff: dict[str, Any]) -> str:
        """Render."""
        return "\n".join(diffreports._render_transfer_summary(diff))

    def test_transfer_decision_escapes_stripped(self) -> None:
        """Test transfer decision escapes stripped."""
        # C2: raw old_decision/new_decision interpolation injected
        # terminal escapes into stderr.
        old = _hostile_decision_payload("selected")
        new = _hostile_decision_payload("selected\x1b[2Jevil")
        diff = diffreports.diff_transfer(old, new)
        rendered = self._render(diff)
        self.assertNotIn("\x1b", rendered)
        self.assertIn("evil", rendered)

    def test_robustness_decision_escapes_stripped(self) -> None:
        """Test robustness decision escapes stripped."""

        def _payload(decision: Any) -> dict[str, Any]:
            """Payload."""
            return {
                "available": True,
                "primary_decision": "selected",
                "primary_winner": "alpha",
                "per_semantic": {
                    "s1": {"decision": decision, "winner": "alpha"},
                },
                "flips": [],
            }

        old, new = _payload("selected"), _payload("tie\x07evil")
        diff = diffreports.diff_selection_robustness(old, new)
        rendered = "\n".join(diffreports._render_robustness_summary(diff))
        self.assertNotIn("\x1b", rendered)
        self.assertNotIn("\x07", rendered)
        self.assertIn("evil", rendered)

    def test_perturbation_decision_escapes_stripped(self) -> None:
        """Test perturbation decision escapes stripped."""

        def _payload(decision: Any) -> dict[str, Any]:
            """Payload."""
            return {
                "available": True,
                "primary_winner": "alpha",
                "primary_decision": "selected",
                "n_variants": 1,
                "operators": ["typo"],
                "per_variant": {
                    "v00": {
                        "operator": "typo",
                        "decision": decision,
                        "winner": "alpha",
                        "changed_fixtures": 1,
                        "changed_rate": 1.0,
                    }
                },
                "stability": {
                    "n_variants": 1,
                    "n_evaluable": 1,
                    "n_unavailable": 0,
                    "winner_retained": 1,
                    "decision_retained": 1,
                },
                "flips": [],
            }

        old, new = _payload("selected"), _payload("selected\x1b[31mred")
        diff = diffreports.diff_perturbation(old, new)
        rendered = "\n".join(diffreports._render_perturbation_summary(diff))
        self.assertNotIn("\x1b", rendered)
        self.assertIn("red", rendered)

    def test_non_string_decisions_do_not_crash_render(self) -> None:
        """Test non string decisions do not crash render."""
        # C3 end-to-end: non-string decisions flow through render_summary.
        base = {
            "decision": "selected",
            "selected_candidate": "alpha",
            "search": [],
            "held_out": {},
        }
        old = _hostile_decision_payload("selected")
        new = _hostile_decision_payload({"nested": ["decision"]})
        diff = diffreports.diff_reports(
            dict(base, transfer=old),
            dict(base, transfer=new),
        )
        rendered = diffreports.render_summary(diff)
        self.assertNotIn("\x1b", rendered)
        self.assertIn("nested", rendered)


class TestChangedDisjunctPinning(unittest.TestCase):
    """Each ``changed`` disjunct must be pinned by a single-cause test.

    C4/C5: mutants dropping ``or cell_drift`` / ``or tier_changes``
    survived the whole suite because every existing test changed two
    things at once. The same gap class is pinned here for the new
    ``diff_perturbation`` disjuncts too.
    """

    def test_transfer_cell_drift_only_sets_changed(self) -> None:
        """Test transfer cell drift only sets changed."""
        old = _hostile_decision_payload("selected")
        new = _hostile_decision_payload("selected")
        new["matrix"] = {"alpha": {"ckptA": "fails_there"}}
        diff = diffreports.diff_transfer(old, new)
        self.assertEqual(diff["available"], "both")
        self.assertEqual(
            diff["cell_drift"],
            [
                {
                    "candidate": "alpha",
                    "checkpoint": "ckptA",
                    "old": "transfers",
                    "new": "fails_there",
                }
            ],
        )
        self.assertEqual(diff["checkpoint_drift"], [])
        self.assertFalse(diff["transfer_verdict"]["changed"])
        self.assertTrue(diff["changed"])

    def test_contamination_tier_escalation_only_sets_changed(self) -> None:
        """Test contamination tier escalation only sets changed."""

        def _payload(tier: str) -> dict[str, Any]:
            """Payload."""
            return {
                "worst_tier": "duplicate",
                "pairs": [
                    {"held_out": "h1", "search": "s1", "tier": "duplicate"},
                    {"held_out": "h2", "search": "s2", "tier": tier},
                ],
            }

        old, new = _payload("clean"), _payload("near_duplicate")
        diff = diffreports.diff_contamination(old, new)
        self.assertEqual(diff["available"], "both")
        self.assertFalse(diff["worst_tier"]["changed"])
        self.assertEqual(diff["pairs_added"], [])
        self.assertEqual(diff["pairs_removed"], [])
        self.assertEqual(len(diff["tier_changes"]), 1)
        self.assertTrue(diff["changed"])

    @staticmethod
    def _perturbation_payload(
        stability: dict[str, Any] | None = None,
        per_variant: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Perturbation payload."""
        return {
            "available": True,
            "primary_winner": "alpha",
            "primary_decision": "selected",
            "n_variants": 2,
            "operators": ["typo", "case_flip"],
            "per_variant": per_variant
            if per_variant is not None
            else {
                "v00": {
                    "operator": "typo",
                    "decision": "selected",
                    "winner": "alpha",
                    "changed_fixtures": 2,
                    "changed_rate": 1.0,
                },
                "v01": {
                    "operator": "case_flip",
                    "decision": "selected",
                    "winner": "alpha",
                    "changed_fixtures": 2,
                    "changed_rate": 1.0,
                },
            },
            "stability": stability
            if stability is not None
            else {
                "n_variants": 2,
                "n_evaluable": 2,
                "n_unavailable": 0,
                "winner_retained": 2,
                "decision_retained": 2,
            },
            "flips": [],
        }

    def test_perturbation_stability_only_sets_changed(self) -> None:
        """Test perturbation stability only sets changed."""
        # Same per-variant verdicts, different recorded stability: the
        # stability disjunct alone must flip ``changed``.
        old = self._perturbation_payload()
        new = self._perturbation_payload(
            stability={
                "n_variants": 2,
                "n_evaluable": 2,
                "n_unavailable": 0,
                "winner_retained": 1,
                "decision_retained": 2,
            }
        )
        diff = diffreports.diff_perturbation(old, new)
        self.assertTrue(diff["stability"]["changed"])
        self.assertEqual(diff["variant_drift"], [])
        self.assertTrue(diff["changed"])

    def test_perturbation_variant_drift_only_sets_changed(self) -> None:
        """Test perturbation variant drift only sets changed."""
        # One variant's winner flips while the recorded stability triplet
        # is unchanged: the variant-drift disjunct alone must flip
        # ``changed``.
        per_variant = {
            "v00": {
                "operator": "typo",
                "decision": "selected",
                "winner": "beta",
                "changed_fixtures": 2,
                "changed_rate": 1.0,
            },
            "v01": {
                "operator": "case_flip",
                "decision": "selected",
                "winner": "alpha",
                "changed_fixtures": 2,
                "changed_rate": 1.0,
            },
        }
        old = self._perturbation_payload()
        new = self._perturbation_payload(per_variant=per_variant)
        diff = diffreports.diff_perturbation(old, new)
        self.assertFalse(diff["stability"]["changed"])
        self.assertEqual(len(diff["variant_drift"]), 1)
        self.assertTrue(diff["changed"])

    def test_perturbation_removed_only_sets_changed(self) -> None:
        """Test perturbation removed only sets changed."""
        old = self._perturbation_payload()
        new = self._perturbation_payload()
        del new["per_variant"]["v01"]
        # v00 unchanged (no variant drift) and the recorded stability
        # triplet untouched: only the removed disjunct fires.
        diff = diffreports.diff_perturbation(old, new)
        self.assertFalse(diff["stability"]["changed"])
        self.assertEqual(diff["variant_drift"], [])
        self.assertEqual(diff["variants_removed"], ["v01"])
        self.assertTrue(diff["changed"])


class TestCheckpointDriftCap(unittest.TestCase):
    """Test checkpoint drift cap."""

    def test_checkpoint_drift_truncated_at_ten(self) -> None:
        """Test checkpoint drift truncated at ten."""
        # C6: the only drift renderer without a cap; now capped like
        # cell_drift.
        old = _hostile_decision_payload("selected")
        new = _hostile_decision_payload("selected")
        new["per_checkpoint"] = {
            f"ckpt{i:02d}": {"decision": "tie", "winner": None} for i in range(12)
        }
        old["per_checkpoint"] = {
            f"ckpt{i:02d}": {"decision": "selected", "winner": "alpha"} for i in range(12)
        }
        diff = diffreports.diff_transfer(old, new)
        self.assertEqual(len(diff["checkpoint_drift"]), 12)
        lines = diffreports._render_transfer_summary(diff)
        drift_lines = [line for line in lines if line.startswith("  ckpt")]
        self.assertEqual(len(drift_lines), 10)
        self.assertTrue(any("... and 2 more" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
