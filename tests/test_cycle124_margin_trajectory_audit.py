"""Cycle 124 audit: selection margin trajectory over the token-budget sweep.

``margin_trajectory`` derives, from a recorded ``budget_frontier`` alone,
the per-budget leader and lead margin (leader pass count minus best
runner-up; 0 on a tied lead). The margin trajectory chart renders it as a
deterministic inline SVG whose step segments are colored by the leading
candidate (shared palette with the frontier chart). These tests pin the
decision-table semantics, the hand-derived geometry, escaping, and the
section wiring; a mutation gate at the bottom proves the tests are not
vacuous.
"""

import re
import unittest
from pathlib import Path

from model_protocol_lab import reporting
from model_protocol_lab.reporting import margin_trajectory

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _frontier() -> dict:
    return {
        "max_new_tokens": 10,
        "candidates": {
            "alpha": {
                "fixture_ids": ["f1", "f2"],
                "pass_steps": [[1, 0], [4, 1], [8, 2]],
                "full_pass_budget": 8,
            },
            "beta": {
                "fixture_ids": ["f1", "f2"],
                "pass_steps": [[1, 0], [6, 2]],
                "full_pass_budget": 6,
            },
        },
        "decision_steps": [[1, "no_candidate", None], [6, "no_candidate", None]],
    }


def _seg_paths(svg: str) -> list[tuple[str, str]]:
    return re.findall(r'<path d="([^"]+)" fill="none" stroke="([^"]+)"', svg)


class TestMarginTrajectorySemantics(unittest.TestCase):
    def test_hand_derived_trajectory(self) -> None:
        # b=1: 0-0 tie; b=4: alpha 1-0; b=6: beta 2-1; b=8: 2-2 tie.
        self.assertEqual(
            margin_trajectory(_frontier()),
            [(1, None, 0), (4, "alpha", 1), (6, "beta", 1), (8, None, 0)],
        )

    def test_single_candidate_has_zero_margin(self) -> None:
        frontier = {
            "max_new_tokens": 5,
            "candidates": {
                "solo": {
                    "fixture_ids": ["f1"],
                    "pass_steps": [[1, 1]],
                    "full_pass_budget": 1,
                }
            },
        }
        self.assertEqual(margin_trajectory(frontier), [(1, "solo", 0)])

    def test_three_way_tie_collapses_to_single_point(self) -> None:
        frontier = {
            "max_new_tokens": 4,
            "candidates": {
                name: {
                    "fixture_ids": ["f1"],
                    "pass_steps": [[1, 0]],
                    "full_pass_budget": None,
                }
                for name in ("a", "b", "c")
            },
        }
        self.assertEqual(margin_trajectory(frontier), [(1, None, 0)])

    def test_constant_margin_collapses_change_points(self) -> None:
        # alpha leads beta by exactly 1 at every change budget: one point.
        frontier = {
            "max_new_tokens": 8,
            "candidates": {
                "alpha": {
                    "fixture_ids": ["f1", "f2"],
                    "pass_steps": [[1, 1], [4, 2]],
                    "full_pass_budget": 4,
                },
                "beta": {
                    "fixture_ids": ["f1", "f2"],
                    "pass_steps": [[1, 0], [4, 1]],
                    "full_pass_budget": None,
                },
            },
        }
        self.assertEqual(margin_trajectory(frontier), [(1, "alpha", 1)])

    def test_malformed_input_yields_empty(self) -> None:
        self.assertEqual(margin_trajectory({}), [])
        self.assertEqual(margin_trajectory({"candidates": {}}), [])
        self.assertEqual(margin_trajectory({"candidates": {"a": {}}}), [])
        self.assertEqual(
            margin_trajectory({"candidates": {"a": {"fixture_ids": []}}, "max_new_tokens": 3}),
            [],
        )
        self.assertEqual(margin_trajectory("not-a-dict"), [])  # type: ignore[arg-type]

    def test_deterministic_across_calls(self) -> None:
        frontier = _frontier()
        self.assertEqual(margin_trajectory(frontier), margin_trajectory(frontier))


class TestMarginChartGeometry(unittest.TestCase):
    def test_segment_paths_hand_derived(self) -> None:
        # W=560 over budgets 1..10: x(b) = (b-1)*560//9.
        # y_max=1, H=240: y(0)=240, y(1)=0.
        svg = reporting._margin_chart_svg(_frontier())
        paths = _seg_paths(svg)
        self.assertEqual(len(paths), 4)
        # tie span [1,4): gray; alpha leads [4,6); beta leads [6,8); tie [8,10].
        self.assertEqual(paths[0], ("M 0 240 H 186 V 0", "#94a3b8"))
        self.assertEqual(paths[1], ("M 186 0 H 311 V 0", "#38bdf8"))
        self.assertEqual(paths[2], ("M 311 0 H 435 V 240", "#22c55e"))
        self.assertEqual(paths[3], ("M 435 240 H 560 V 240", "#94a3b8"))

    def test_leader_colors_match_frontier_palette_order(self) -> None:
        # Reversed dict insertion order must not change the assignment:
        # sorted-name order rules, shared with the frontier chart.
        frontier = _frontier()
        frontier["candidates"] = dict(reversed(list(frontier["candidates"].items())))
        paths = _seg_paths(reporting._margin_chart_svg(frontier))
        colors = {c for _, c in paths}
        self.assertEqual(colors, {"#94a3b8", "#38bdf8", "#22c55e"})
        # alpha (sorted first) keeps the first palette color.
        self.assertEqual(paths[1][1], "#38bdf8")

    def test_area_fill_present_but_not_url_based(self) -> None:
        svg = reporting._margin_chart_svg(_frontier())
        self.assertIn('fill-opacity="0.10"', svg)
        self.assertNotIn("url(", svg)

    def test_zero_baseline_emphasized(self) -> None:
        svg = reporting._margin_chart_svg(_frontier())
        # y(0)=240 is the bottom gridline; the zero tick uses the axis color.
        self.assertIn('y1="240" x2="560" y2="240" stroke="#475569"', svg)

    def test_malformed_returns_empty(self) -> None:
        self.assertEqual(reporting._margin_chart_svg({}), "")
        self.assertEqual(reporting._margin_chart_svg({"candidates": {}}), "")

    def test_byte_deterministic(self) -> None:
        frontier = _frontier()
        self.assertEqual(
            reporting._margin_chart_svg(frontier), reporting._margin_chart_svg(frontier)
        )

    def test_aria_covers_every_span(self) -> None:
        svg = reporting._margin_chart_svg(_frontier())
        match = re.search(r'aria-label="([^"]+)"', svg)
        self.assertIsNotNone(match)
        aria = match.group(1)
        for leader in ("tied", "alpha", "beta"):
            self.assertIn("leader " + leader, aria)
        self.assertIn("margin 1", aria)

    def test_hostile_candidate_name_escaped(self) -> None:
        frontier = {
            "max_new_tokens": 4,
            "candidates": {
                "<script>alert(1)</script>": {
                    "fixture_ids": ["f1"],
                    "pass_steps": [[1, 1]],
                    "full_pass_budget": 1,
                },
                "plain": {
                    "fixture_ids": ["f1"],
                    "pass_steps": [[1, 0]],
                    "full_pass_budget": None,
                },
            },
        }
        svg = reporting._margin_chart_svg(frontier)
        self.assertNotIn("<script>", svg)
        self.assertIn("&lt;script&gt;", svg)


class TestSectionWiring(unittest.TestCase):
    def test_margin_block_follows_frontier_chart(self) -> None:
        html = reporting._frontier_section({"budget_frontier": _frontier()})
        self.assertIn("<h3>Margin trajectory: how decisive was the lead</h3>", html)
        self.assertIn('class="margin-svg"', html)
        self.assertLess(
            html.index("Frontier chart: all candidates"),
            html.index("Margin trajectory: how decisive was the lead"),
        )
        self.assertLess(
            html.index("Margin trajectory: how decisive was the lead"),
            html.index("Pass curves by candidate"),
        )

    def test_section_without_frontier_has_no_margin_block(self) -> None:
        self.assertNotIn("margin-svg", reporting._frontier_section({}))
        self.assertNotIn("margin-svg", reporting._frontier_section({"budget_frontier": None}))


class TestMutationGate(unittest.TestCase):
    SOURCE = _REPO_ROOT / "model_protocol_lab" / "reporting.py"

    MUTANTS = [
        # Leader margin computed as runner-up minus leader (sign flip).
        ("margin = top_count - second[0]", "margin = second[0] - top_count"),
        # Tied lead reported with a nonzero margin.
        (
            "            leader = None\n            margin = 0",
            "            leader = None\n            margin = top_count",
        ),
        # Unique-leader test inverted: ties report a leader, leaders report None.
        ("elif second[0] == top_count:", "elif second[0] != top_count:"),
        # Margin chart y-axis inverted (margin grows downward).
        ("return (y_max - margin) * height // denom_y", "return margin * height // denom_y"),
    ]

    def test_mutants_are_killed(self) -> None:
        import subprocess
        import sys

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
                        # (cycle-125's 22-site anchor fanned out to 184
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
                )
            finally:
                self.SOURCE.write_text(source)
                # Invalidate the bytecode cache: a same-size mutant can
                # leave a .pyc whose (mtime, size) still validates
                # against the reverted source, poisoning later imports
                # with mutant bytecode (see the cycle-135 gate fix).
                for _pyc in self.SOURCE.parent.glob("__pycache__/reporting.*.pyc"):
                    _pyc.unlink(missing_ok=True)
            if proc.returncode != 0:
                killed.append(mutant)
            else:
                survived.append(mutant)
        self.assertEqual(survived, [], f"surviving mutants: {survived}")
        self.assertEqual(len(killed), len(self.MUTANTS))
