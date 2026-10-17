"""Cycle 122: audit of the decision ribbon in the frontier chart.

Target: the decision-ribbon band added to
``model_protocol_lab.reporting._frontier_chart_svg`` in this cycle — a
horizontal band under the chart's x-axis coloring every token-budget
span by the post-hoc selection outcome, drawn from the recorded
``decision_steps``.

Method
------
1. **Hand-derived geometry.** The two-candidate example frontier (shared
   with the cycle-121 audit) has decision spans [1,8] no_candidate and
   [9,10] selected; every ribbon ``<rect>`` is a literal derived on paper
   from the integer x-mapping. A second fixture exercises the ambiguous
   span.
2. **Legend/aria.** Decision swatches come first in the legend in fixed
   order (selected, ambiguous, no candidate — only those present); the
   aria-label states each decision span exactly.
3. **Robustness.** Missing or malformed ``decision_steps`` degrades to a
   chart without a ribbon instead of crashing; rendering never mutates
   the input.
4. **Mutation gate.** Three mutants in /tmp copies of reporting.py
   (PID-scoped; the repo is never mutated): swapped ribbon colors,
   off-by-one span ends, wrong legend order. Each is caught; an
   unmutated control passes.
"""

import copy
import importlib.util
import os
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from model_protocol_lab import reporting
from model_protocol_lab.reporting import _frontier_chart_svg

_REPO_ROOT = Path(reporting.__file__).resolve().parent.parent
_SCRATCH = Path(tempfile.gettempdir()) / f"cycle122_ribbon_{os.getpid()}"


def _example_frontier() -> dict:
    """Same two-candidate frontier as the cycle-121 audit, plus the
    decision steps the ribbon is drawn from.

    x(b) = (b-1)*560//9 -> x(1)=0, x(5)=248, x(9)=497, x(10)=560.
    Ribbon spans: [1,8] no_candidate (#64748b), [9,10] selected (#16a34a),
    at y = 240+38 = 278 with height 12.
    """
    return {
        "max_new_tokens": 10,
        "native_winner": "alpha",
        "decision_steps": [[1, "no_candidate", None], [9, "selected", "alpha"]],
        "candidates": {
            "beta": {
                "fixture_ids": ["f1", "f2", "f3"],
                "min_budget_per_fixture": [None, 7, 3],
                "pass_steps": [[1, 0], [3, 1], [7, 2]],
                "full_pass_budget": None,
            },
            "alpha": {
                "fixture_ids": ["f1", "f2", "f3"],
                "min_budget_per_fixture": [1, 4, 9],
                "pass_steps": [[1, 1], [4, 2], [9, 3]],
                "full_pass_budget": 9,
            },
        },
    }


def _ambiguous_frontier() -> dict:
    """Both candidates fully pass at budgets 5..10 -> ambiguous span."""
    return {
        "max_new_tokens": 10,
        "native_winner": None,
        "decision_steps": [[1, "selected", "alpha"], [5, "ambiguous", None]],
        "candidates": {
            "beta": {
                "fixture_ids": ["f1", "f2", "f3"],
                "pass_steps": [[1, 0], [5, 3]],
                "full_pass_budget": 5,
            },
            "alpha": {
                "fixture_ids": ["f1", "f2", "f3"],
                "pass_steps": [[1, 3]],
                "full_pass_budget": 1,
            },
        },
    }


def _ribbon_rects(svg: str) -> list[tuple[str, str, str]]:
    return re.findall(
        r'<rect x="(\d+)" y="278" width="(\d+)" height="12" fill="([^"]+)"/>',
        svg,
    )


def _load_mutant(source_path: Path, old: str, new: str, tag: str):
    workdir = _SCRATCH / tag
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    text = source_path.read_text(encoding="utf-8")
    if old:
        assert old in text, f"mutant anchor missing: {old[:60]!r}"
        text = text.replace(old, new, 1)
    (workdir / "reporting.py").write_text(text, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(
        f"cycle122_mutant_{tag}", workdir / "reporting.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


class TestRibbonGeometry(unittest.TestCase):
    def test_hand_derived_rects(self) -> None:
        rects = _ribbon_rects(_frontier_chart_svg(_example_frontier()))
        self.assertEqual(
            rects,
            [("0", "497", "#64748b"), ("497", "63", "#16a34a")],
        )

    def test_ribbon_tiles_full_width(self) -> None:
        # Spans tile [0, 560] exactly: no gaps, no overlaps.
        rects = _ribbon_rects(_frontier_chart_svg(_example_frontier()))
        pos = 0
        for x, w, _ in rects:
            self.assertEqual(int(x), pos)
            pos += int(w)
        self.assertEqual(pos, 560)

    def test_ambiguous_span(self) -> None:
        rects = _ribbon_rects(_frontier_chart_svg(_ambiguous_frontier()))
        self.assertEqual(
            rects,
            [("0", "248", "#16a34a"), ("248", "312", "#d97706")],
        )

    def test_decision_label_present(self) -> None:
        svg = _frontier_chart_svg(_example_frontier())
        self.assertIn(">decision</text>", svg)


class TestRibbonLegendAndAria(unittest.TestCase):
    def test_decision_swatches_first_in_fixed_order(self) -> None:
        svg = _frontier_chart_svg(_example_frontier())
        legend = re.findall(r"<text[^>]*>([^<]*)</text>", svg)
        # Last legend row items: decision swatches, then candidates.
        self.assertEqual(legend[-4:], ["selected", "no candidate", "beta", "alpha"])
        swatches = {
            name: fill
            for fill, name in re.findall(
                r'<rect x="\d+" y="\d+" width="10" height="10" '
                r'fill="([^"]+)"/><text[^>]*>([^<]+)</text>',
                svg,
            )
        }
        self.assertEqual(swatches["selected"], "#16a34a")
        self.assertEqual(swatches["no candidate"], "#64748b")

    def test_aria_label_states_decision_spans(self) -> None:
        svg = _frontier_chart_svg(_example_frontier())
        match = re.search(r'aria-label="([^"]+)"', svg)
        self.assertIsNotNone(match)
        assert match is not None
        label = match.group(1)
        self.assertIn("decision no candidate at budgets 1\u20138", label)
        self.assertIn("decision selected at budgets 9\u201310", label)


class TestRibbonRobustness(unittest.TestCase):
    def test_missing_decision_steps_renders_chart_without_ribbon(self) -> None:
        frontier = _example_frontier()
        del frontier["decision_steps"]
        svg = _frontier_chart_svg(frontier)
        self.assertEqual(_ribbon_rects(svg), [])
        self.assertIn('class="frontier-svg"', svg)
        # No decision swatches in the legend either.
        self.assertNotIn(">selected</text>", svg)

    def test_malformed_decision_steps_skipped(self) -> None:
        frontier = _example_frontier()
        frontier["decision_steps"] = [
            ["x", "selected", None],
            [3],
            None,
            [9, "selected", "alpha"],
            [2, 7, "nope"],
        ]
        svg = _frontier_chart_svg(frontier)
        # Only [9, "selected", "alpha"] is well-formed -> single span.
        rects = _ribbon_rects(svg)
        self.assertEqual(len(rects), 1)
        self.assertEqual(rects[0][2], "#16a34a")

    def test_does_not_mutate_input(self) -> None:
        frontier = _example_frontier()
        snapshot = copy.deepcopy(frontier)
        _frontier_chart_svg(frontier)
        self.assertEqual(frontier, snapshot)


class TestRibbonMutationGate(unittest.TestCase):
    SOURCE = _REPO_ROOT / "model_protocol_lab" / "reporting.py"

    def test_control_passes(self) -> None:
        module = _load_mutant(self.SOURCE, "", "", "control")
        rects = _ribbon_rects(module._frontier_chart_svg(_example_frontier()))
        self.assertEqual(rects, [("0", "497", "#64748b"), ("497", "63", "#16a34a")])

    def test_swapped_ribbon_colors_caught(self) -> None:
        module = _load_mutant(
            self.SOURCE,
            '    "selected": "#16a34a",\n    "ambiguous": "#d97706",\n'
            '    "no_candidate": "#64748b",',
            '    "selected": "#64748b",\n    "ambiguous": "#d97706",\n'
            '    "no_candidate": "#16a34a",',
            "ribbon_swap",
        )
        rects = _ribbon_rects(module._frontier_chart_svg(_example_frontier()))
        self.assertNotEqual(rects, [("0", "497", "#64748b"), ("497", "63", "#16a34a")])

    def test_off_by_one_span_end_caught(self) -> None:
        module = _load_mutant(
            self.SOURCE,
            "hi = dec_points[i + 1][0] - 1 if i + 1 < len(dec_points) else max_tokens",
            "hi = dec_points[i + 1][0] if i + 1 < len(dec_points) else max_tokens",
            "span_offbyone",
        )
        rects = _ribbon_rects(module._frontier_chart_svg(_example_frontier()))
        # First span would wrongly extend to budget 9 (x=497+62=559...).
        self.assertNotEqual(rects[0], ("0", "497", "#64748b"))

    def test_unknown_decision_falls_back_to_neutral(self) -> None:
        frontier = _example_frontier()
        frontier["decision_steps"] = [[1, "weird_outcome", None]]
        svg = _frontier_chart_svg(frontier)
        rects = _ribbon_rects(svg)
        self.assertEqual(len(rects), 1)
        self.assertEqual(rects[0][2], "#64748b")


if __name__ == "__main__":
    unittest.main()
