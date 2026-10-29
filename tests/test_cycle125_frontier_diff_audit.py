"""Cycle 125 audit: token-budget frontier comparison in diff-reports.

``diff_frontiers`` compares two recorded ``budget_frontier`` payloads:
sweep-range drift, budget-robustness flips, and the exact budget spans
where the post-hoc decision, post-hoc winner, or per-budget leader
(from ``reporting.margin_trajectory``) differ. It is wired into
``diff_reports`` (output dict + summary flags + ``identical``) and
``render_summary``. Tests pin hand-derived spans, availability handling,
escaping, determinism, and the integration; a mutation gate proves the
tests are not vacuous.
"""

import json
import subprocess
import sys
import unittest
from pathlib import Path

from model_protocol_lab.diffreports import (
    _diff_spans,
    diff_frontiers,
    diff_reports,
    render_summary,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _frontier(
    steps: dict,
    decision_steps: list,
    max_tokens: int = 10,
    robust: bool = False,
) -> dict:
    """Frontier."""
    candidates = {}
    for name, pts in steps.items():
        candidates[name] = {
            "fixture_ids": ["f1", "f2"],
            "pass_steps": [[b, c] for b, c in pts],
            "full_pass_budget": None,
        }
    return {
        "max_new_tokens": max_tokens,
        "candidates": candidates,
        "decision_steps": decision_steps,
        "budget_robust": robust,
    }


def _base(frontier: dict) -> dict:
    """Base."""
    return {
        "decision": "compatible_on_fixture_set",
        "selected_candidate": "alpha",
        "search": [],
        "budget_frontier": frontier,
    }


class TestDiffFrontiers(unittest.TestCase):
    """Test diff frontiers."""

    def test_identical_frontiers_no_changes(self) -> None:
        """Test identical frontiers no changes."""
        frontier = _frontier(
            {"alpha": [(1, 2)], "beta": [(1, 0), (5, 2)]},
            [[1, "selected", "alpha"]],
            robust=True,
        )
        diff = diff_frontiers(frontier, frontier)
        self.assertEqual(diff["available"], "both")
        self.assertFalse(diff["changed"])
        self.assertEqual(diff["decision_flips"], [])
        self.assertEqual(diff["winner_flips"], [])
        self.assertEqual(diff["leader_flips"], [])
        self.assertFalse(diff["max_new_tokens"]["changed"])
        self.assertFalse(diff["budget_robust"]["changed"])

    def test_leader_flip_span_hand_derived(self) -> None:
        """Test leader flip span hand derived."""
        # old: alpha leads 2-0 until beta ties 2-2 at budget 5.
        # new: beta takes the lead 2-1 at budget 3, tie at 5.
        old = _frontier(
            {"alpha": [(1, 2)], "beta": [(1, 0), (5, 2)]},
            [[1, "selected", "alpha"]],
        )
        new = _frontier(
            {"alpha": [(1, 1), (5, 2)], "beta": [(1, 0), (3, 2)]},
            [[1, "selected", "alpha"]],
        )
        diff = diff_frontiers(old, new)
        self.assertTrue(diff["changed"])
        self.assertEqual(diff["leader_flips"], [[3, 4, "alpha", "beta"]])
        self.assertEqual(diff["decision_flips"], [])
        self.assertEqual(diff["winner_flips"], [])

    def test_decision_and_winner_flip_spans(self) -> None:
        """Test decision and winner flip spans."""
        old = _frontier(
            {"alpha": [(1, 2)]},
            [[1, "selected", "alpha"], [6, "ambiguous", None]],
        )
        new = _frontier(
            {"alpha": [(1, 2)]},
            [[1, "selected", "alpha"], [4, "no_candidate", None]],
        )
        diff = diff_frontiers(old, new)
        # Decisions agree on 1-3; differ on 4-10 (selected vs no_candidate
        # on 4-5, ambiguous vs no_candidate on 6-10).
        self.assertEqual(
            diff["decision_flips"],
            [[4, 5, "selected", "no_candidate"], [6, 10, "ambiguous", "no_candidate"]],
        )
        self.assertEqual(
            diff["winner_flips"],
            [[4, 5, "alpha", None]],
        )

    def test_adjacent_spans_merge(self) -> None:
        """Test adjacent spans merge."""

        # Mutation-gate pin: _diff_spans must merge adjacent differing
        # spans whose (old, new) values agree. The `budget - 2` mutant
        # breaks the adjacency check, leaving two spans [2, 2] and
        # [3, 3] instead of the merged [2, 3].
        def old_at(budget: int) -> str:
            """Old at."""
            return "a"

        def new_at(budget: int) -> str:
            """New at."""
            return "b" if budget in (2, 3) else "a"

        self.assertEqual(
            _diff_spans(old_at, new_at, [1, 2, 3, 4], 4),
            [[2, 3, "a", "b"]],
        )

    def test_sweep_range_drift_and_common_prefix(self) -> None:
        """Test sweep range drift and common prefix."""
        old = _frontier({"alpha": [(1, 2)]}, [[1, "selected", "alpha"]], max_tokens=10)
        new = _frontier({"alpha": [(1, 2)]}, [[1, "selected", "alpha"]], max_tokens=16)
        diff = diff_frontiers(old, new)
        self.assertTrue(diff["max_new_tokens"]["changed"])
        self.assertEqual(diff["max_new_tokens"]["old"], 10)
        self.assertEqual(diff["max_new_tokens"]["new"], 16)
        self.assertTrue(diff["changed"])

    def test_robustness_flip(self) -> None:
        """Test robustness flip."""
        old = _frontier({"alpha": [(1, 2)]}, [[1, "selected", "alpha"]], robust=True)
        new = _frontier({"alpha": [(1, 2)]}, [[1, "selected", "alpha"]], robust=False)
        diff = diff_frontiers(old, new)
        self.assertTrue(diff["budget_robust"]["changed"])
        self.assertTrue(diff["changed"])

    def test_availability_variants(self) -> None:
        """Test availability variants."""
        good = _frontier({"alpha": [(1, 2)]}, [[1, "selected", "alpha"]])
        for old, new, expected in [
            (good, good, "both"),
            (good, None, "old"),
            (None, good, "new"),
            (None, None, "neither"),
            (good, {"candidates": {}}, "old"),
            ({"junk": True}, {"junk": True}, "neither"),
        ]:
            diff = diff_frontiers(old, new)
            self.assertEqual(diff["available"], expected, f"old={old} new={new}")
            self.assertFalse(diff["changed"])

    def test_malformed_decision_steps_skip_axis(self) -> None:
        """Test malformed decision steps skip axis."""
        old = _frontier({"alpha": [(1, 2)], "beta": [(1, 0)]}, "not-a-list")
        new = _frontier({"alpha": [(1, 2)], "beta": [(1, 0)]}, [[1, "selected", "alpha"]])
        diff = diff_frontiers(old, new)
        self.assertEqual(diff["available"], "both")
        # Decision/winner axes cannot be compared; the leader axis still can.
        self.assertEqual(diff["decision_flips"], [])
        self.assertEqual(diff["winner_flips"], [])

    def test_deterministic_json(self) -> None:
        """Test deterministic json."""
        old = _frontier(
            {"alpha": [(1, 2)], "beta": [(1, 0), (5, 2)]},
            [[1, "selected", "alpha"]],
        )
        new = _frontier(
            {"alpha": [(1, 1), (5, 2)], "beta": [(1, 0), (3, 2)]},
            [[1, "selected", "alpha"]],
        )
        first = json.dumps(diff_frontiers(old, new), sort_keys=True)
        # skipcq: PYL-W1114 - the reversed diff is the assertion: symmetry is what is audited
        second = json.dumps(diff_frontiers(new, old), sort_keys=True)
        self.assertEqual(json.dumps(diff_frontiers(old, new), sort_keys=True), first)
        self.assertNotEqual(first, second)  # direction matters


class TestDiffReportsIntegration(unittest.TestCase):
    """Test diff reports integration."""

    def test_frontier_change_breaks_identical(self) -> None:
        """Test frontier change breaks identical."""
        old = _frontier(
            {"alpha": [(1, 2)], "beta": [(1, 0), (5, 2)]},
            [[1, "selected", "alpha"]],
        )
        new = _frontier(
            {"alpha": [(1, 1), (5, 2)], "beta": [(1, 0), (3, 2)]},
            [[1, "selected", "alpha"]],
        )
        diff = diff_reports(_base(old), _base(new), old_label="a", new_label="b")
        self.assertTrue(diff["summary"]["frontier_changed"])
        self.assertFalse(diff["summary"]["identical"])
        self.assertIn("frontier", diff)
        self.assertEqual(diff["frontier"]["leader_flips"], [[3, 4, "alpha", "beta"]])

    def test_frontier_asymmetry_breaks_identical(self) -> None:
        """Test frontier asymmetry breaks identical."""
        old = _frontier({"alpha": [(1, 2)]}, [[1, "selected", "alpha"]])
        diff = diff_reports(_base(old), _base(None), old_label="a", new_label="b")
        self.assertTrue(diff["summary"]["frontier_asymmetric"])
        self.assertFalse(diff["summary"]["identical"])

    def test_neither_frontier_stays_identical(self) -> None:
        """Test neither frontier stays identical."""
        diff = diff_reports(_base(None), _base(None), old_label="a", new_label="b")
        self.assertFalse(diff["summary"]["frontier_changed"])
        self.assertFalse(diff["summary"]["frontier_asymmetric"])
        self.assertTrue(diff["summary"]["identical"])

    def test_identical_frontiers_keep_identical(self) -> None:
        """Test identical frontiers keep identical."""
        frontier = _frontier({"alpha": [(1, 2)]}, [[1, "selected", "alpha"]], robust=True)
        diff = diff_reports(_base(frontier), _base(frontier), old_label="a", new_label="b")
        self.assertTrue(diff["summary"]["identical"])
        self.assertIn("token-budget frontier", render_summary(diff))


class TestRenderSummary(unittest.TestCase):
    """Test render summary."""

    def test_flip_spans_rendered(self) -> None:
        """Test flip spans rendered."""
        old = _frontier(
            {"alpha": [(1, 2)], "beta": [(1, 0), (5, 2)]},
            [[1, "selected", "alpha"]],
        )
        new = _frontier(
            {"alpha": [(1, 1), (5, 2)], "beta": [(1, 0), (3, 2)]},
            [[1, "selected", "alpha"]],
        )
        text = render_summary(diff_reports(_base(old), _base(new), old_label="a", new_label="b"))
        self.assertIn("Per-budget leader differs (1 span(s)):", text)
        self.assertIn("budgets 3\u20134: alpha -> beta", text)
        self.assertIn("Budget-robust: no -> no", text)
        self.assertIn("frontier CHANGED", text)

    def test_unavailable_variants_rendered(self) -> None:
        """Test unavailable variants rendered."""
        good = _frontier({"alpha": [(1, 2)]}, [[1, "selected", "alpha"]])
        text = render_summary(diff_reports(_base(good), _base(None), old_label="a", new_label="b"))
        self.assertIn("did not record one", text)
        text = render_summary(diff_reports(_base(None), _base(None), old_label="a", new_label="b"))
        self.assertIn("not recorded by either run", text)

    def test_hostile_names_sanitized(self) -> None:
        """Test hostile names sanitized."""
        # render_summary writes to stderr (a terminal): the invariant is
        # that control characters and bidi spoofing are stripped, matching
        # _display_id's contract for every other operator-visible name.
        hostile = "evil\x1b[2J\u202ename"
        old = _frontier(
            {hostile: [(1, 2)], "beta": [(1, 0), (5, 2)]},
            [[1, "selected", hostile]],
        )
        new = _frontier(
            {hostile: [(1, 0)], "beta": [(1, 0), (3, 2)]},
            [[1, "selected", "beta"]],
        )
        text = render_summary(diff_reports(_base(old), _base(new), old_label="a", new_label="b"))
        self.assertNotIn("\x1b", text)
        self.assertNotIn("\u202e", text)
        # ESC and the bidi control are gone; the printable remainder is
        # inert on a terminal without its introducer.
        self.assertIn("evil[2Jname", text)


class TestMutationGate(unittest.TestCase):
    """Test mutation gate."""

    SOURCE = _REPO_ROOT / "model_protocol_lab" / "diffreports.py"

    MUTANTS = [
        # Winner flips silently dropped from the comparison.
        (
            "        winner_flips = _diff_spans(old_win_at, new_win_at, breakpoints, hi)",
            "        winner_flips = []",
        ),
        # Robustness change ignored.
        (
            '        "changed": old_robust != new_robust,',
            '        "changed": False,',
        ),
        # Span merge condition broken: adjacent spans never merge.
        (
            "            and spans[-1][1] == budget - 1",
            "            and spans[-1][1] == budget - 2",
        ),
        # Availability misreported: one-sided frontier claims "both".
        (
            '    elif old_ok:\n        available = "old"',
            '    elif old_ok:\n        available = "both"',
        ),
    ]

    def test_mutants_are_killed(self) -> None:
        """Test mutants are killed."""
        source = self.SOURCE.read_text()
        killed = []
        survived = []
        for original, mutant in self.MUTANTS:
            self.assertIn(original, source, f"mutant anchor missing: {original!r}")
            self.SOURCE.write_text(source.replace(original, mutant, 1))
            try:
                proc = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        __file__,
                        "-q",
                        "-p",
                        "no:cacheprovider",
                        # Exclude the gate itself from the subprocess run:
                        # without this, the subprocess re-runs this very
                        # test, stacking one more mutant per nesting level
                        # (this gate's 22-site anchor fanned out to 184
                        # processes). -k is used instead of --deselect
                        # because __file__ is absolute under unittest
                        # discovery while pytest normalizes node IDs to
                        # relative paths, so an absolute --deselect never
                        # matches. The kill verdict must come from the
                        # audit tests alone.
                        "-k",
                        "not TestMutationGate",
                    ],
                    capture_output=True,
                    text=True,
                    cwd=_REPO_ROOT,
                    check=False,
                )
            finally:
                self.SOURCE.write_text(source)
                # Invalidate the bytecode cache: a same-size mutant can
                # leave a .pyc whose (mtime, size) still validates
                # against the reverted source, poisoning later imports
                # with mutant bytecode (see the cycle-135 gate fix).
                for _pyc in self.SOURCE.parent.glob("__pycache__/diffreports.*.pyc"):
                    _pyc.unlink(missing_ok=True)
            if proc.returncode != 0:
                killed.append(mutant)
            else:
                survived.append(mutant)
        self.assertEqual(survived, [], f"surviving mutants: {survived}")
        self.assertEqual(len(killed), len(self.MUTANTS))
