"""Cycle 121: audit of the multi-series token-budget frontier chart.

Target: ``model_protocol_lab.reporting._frontier_chart_svg`` (new in this
cycle) and its hook in ``_frontier_section``.

Method
------
1. **Hand-derived geometry.** A two-candidate frontier small enough to
   compute by hand: every ``<path d=...>`` string, stroke color/width,
   draw order, tick labels and legend order are literals derived on paper
   from the integer coordinate mapping -- never from calling the source.
2. **Determinism.** The same frontier rendered 3x in-process and in two
   fresh subprocesses with different ``PYTHONHASHSEED`` values must be
   byte-identical; candidate insertion order must not change the bytes.
3. **Robustness.** Degenerate frontiers (empty, zero budgets, zero
   fixtures, malformed steps) render nothing instead of crashing, and
   rendering never mutates the input dict.
4. **Injection safety.** Hostile candidate names are escaped in legend
   text and in the SVG ``aria-label``.
5. **Mutation gate.** Five mutants planted in /tmp *copies* of
   reporting.py (PID-scoped dirs; the repo is never mutated): unsorted
   legend, shifted x-mapping, rotated palette, unescaped legend name,
   flipped y-mapping. Every mutant is caught by at least one check; an
   unmutated control copy passes every check (proving the harness
   faithful).
"""

import copy
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from model_protocol_lab import reporting
from model_protocol_lab.reporting import _frontier_chart_svg

_REPO_ROOT = Path(reporting.__file__).resolve().parent.parent
_SCRATCH = Path(tempfile.gettempdir()) / f"cycle121_frontier_{os.getpid()}"


def _example_frontier() -> dict:
    """Two candidates, max_new_tokens=10, 3 fixtures.

    Hand-derived geometry (W=560, H=240):
      x(b) = (b-1)*560//9 -> x(1)=0, x(3)=124, x(4)=186, x(7)=373,
                             x(9)=497, x(10)=560
      y(c) = (3-c)*80     -> y(0)=240, y(1)=160, y(2)=80, y(3)=0
    Sorted names: alpha -> #38bdf8, beta -> #22c55e. Winner (alpha)
    draws last with stroke-width 3.
    Decision: no candidate fully passes at budgets 1..8, alpha alone at
    9..10 -> ribbon spans [1,8] no_candidate, [9,10] selected.
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


def _paths(svg: str) -> list[tuple[str, str, str]]:
    return re.findall(
        r'<path d="([^"]+)"[^>]*stroke="([^"]+)"[^>]*stroke-width="([^"]+)"',
        svg,
    )


def _load_mutant(source_path: Path, old: str, new: str, tag: str):
    """Load a mutated copy of reporting.py from a PID-scoped tmp dir."""
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
        f"cycle121_mutant_{tag}", workdir / "reporting.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


class TestHandDerivedGeometry(unittest.TestCase):
    def test_step_paths_colors_and_widths(self) -> None:
        svg = _frontier_chart_svg(_example_frontier())
        paths = _paths(svg)
        self.assertEqual(len(paths), 2)
        # beta draws first (not the winner): palette #22c55e, width 2.
        self.assertEqual(paths[0][1], "#22c55e")
        self.assertEqual(paths[0][2], "2")
        self.assertEqual(paths[0][0], "M 0 240 H 124 V 160 H 373 V 80 H 560")
        # alpha is the winner: palette #38bdf8, drawn last, width 3.
        self.assertEqual(paths[1][1], "#38bdf8")
        self.assertEqual(paths[1][2], "3")
        self.assertEqual(paths[1][0], "M 0 160 H 186 V 80 H 497 V 0 H 560")

    def test_step_markers_at_vertices(self) -> None:
        svg = _frontier_chart_svg(_example_frontier())
        circles = re.findall(r'<circle cx="(\d+)" cy="(\d+)"', svg)
        # alpha vertices: (0,160) (186,80) (497,0); beta: (0,240) (124,160)
        # (373,80). Draw order: beta markers, then alpha markers.
        self.assertEqual(
            circles,
            [
                ("0", "240"),
                ("124", "160"),
                ("373", "80"),
                ("0", "160"),
                ("186", "80"),
                ("497", "0"),
            ],
        )

    def test_tick_labels(self) -> None:
        svg = _frontier_chart_svg(_example_frontier())
        texts = re.findall(r"<text[^>]*>([^<]*)</text>", svg)
        for label in ("1", "3", "5", "7", "10", "0", "2", "token budget"):
            self.assertIn(label, texts, f"missing tick/label {label!r}")

    def test_legend_order_and_colors(self) -> None:
        svg = _frontier_chart_svg(_example_frontier())
        # Decision swatches key the ribbon and come first (fixed order:
        # selected, ambiguous, no candidate — only those present), then
        # the candidates in draw order.
        legend = re.findall(r"<text[^>]*>([^<]*)</text>", svg)[-4:]
        self.assertEqual(legend, ["selected", "no candidate", "beta", "alpha"])
        # Legend swatch colors match the path stroke colors per candidate.
        swatches = {
            name: fill
            for fill, name in re.findall(
                r'<rect x="\d+" y="\d+" width="10" height="10" fill="([^"]+)"/>'
                r"<text[^>]*>([^<]+)</text>",
                svg,
            )
        }
        # Candidate swatches keep their palette colors (the legend now
        # also carries decision swatches for the ribbon; cycle 122).
        self.assertEqual(swatches["beta"], "#22c55e")
        self.assertEqual(swatches["alpha"], "#38bdf8")

    def test_aria_label_states_exact_data(self) -> None:
        svg = _frontier_chart_svg(_example_frontier())
        match = re.search(r'aria-label="([^"]+)"', svg)
        self.assertIsNotNone(match)
        assert match is not None
        label = match.group(1)
        self.assertIn("2 candidates", label)
        self.assertIn("3 fixtures", label)
        self.assertIn("budgets 1 to 10", label)
        self.assertIn("alpha: 1 passing at budget 1, full pass at budget 9", label)
        self.assertIn("beta: 0 passing at budget 1, never fully passes", label)


class TestDeterminism(unittest.TestCase):
    def test_byte_identical_in_process(self) -> None:
        first = _frontier_chart_svg(_example_frontier())
        for _ in range(3):
            self.assertEqual(_frontier_chart_svg(_example_frontier()), first)

    def test_insertion_order_independent(self) -> None:
        forward = _example_frontier()
        reversed_frontier = {
            "max_new_tokens": 10,
            "native_winner": "alpha",
            "decision_steps": [[1, "no_candidate", None], [9, "selected", "alpha"]],
            "candidates": dict(reversed(list(forward["candidates"].items()))),
        }
        self.assertEqual(_frontier_chart_svg(forward), _frontier_chart_svg(reversed_frontier))

    def test_byte_identical_across_hash_seeds(self) -> None:
        _SCRATCH.mkdir(parents=True, exist_ok=True)
        payload = _SCRATCH / "frontier.json"
        payload.write_text(json.dumps(_example_frontier()), encoding="utf-8")
        script = (
            "import json, sys; "
            "sys.path.insert(0, " + repr(str(_REPO_ROOT)) + "); "
            "from model_protocol_lab.reporting import _frontier_chart_svg; "
            "svg = _frontier_chart_svg(json.load(open(" + repr(str(payload)) + "))); "
            "sys.stdout.write(svg)"
        )
        outs = []
        for seed in ("1", "2"):
            proc = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                timeout=120,
                env={**os.environ, "PYTHONHASHSEED": seed},
                cwd=str(_REPO_ROOT),
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr[-1000:])
            outs.append(proc.stdout)
        self.assertEqual(outs[0], outs[1])
        self.assertEqual(outs[0], _frontier_chart_svg(_example_frontier()))

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(_SCRATCH, ignore_errors=True)


class TestRobustness(unittest.TestCase):
    def test_degenerate_inputs_render_nothing(self) -> None:
        bad = [
            {},
            {"candidates": {}, "max_new_tokens": 10},
            {"candidates": {"a": {}}, "max_new_tokens": 0},
            {"candidates": {"a": {}}, "max_new_tokens": -3},
            {
                "max_new_tokens": 10,
                "candidates": {
                    "a": {"fixture_ids": [], "pass_steps": [[1, 0]]},
                },
            },
            {
                "max_new_tokens": 10,
                "candidates": {
                    "a": {
                        "fixture_ids": ["f1"],
                        "pass_steps": [["x", 0], [1], None, [2, 5]],
                    },
                },
            },
        ]
        for frontier in bad:
            with self.subTest(frontier=frontier):
                self.assertEqual(_frontier_chart_svg(frontier), "")

    def test_does_not_mutate_input(self) -> None:
        frontier = _example_frontier()
        snapshot = copy.deepcopy(frontier)
        _frontier_chart_svg(frontier)
        self.assertEqual(frontier, snapshot)

    def test_legend_escapes_hostile_names(self) -> None:
        evil = '"><script>alert(1)</script>&"'
        frontier = {
            "max_new_tokens": 5,
            "native_winner": None,
            "candidates": {
                evil: {
                    "fixture_ids": ["f1"],
                    "pass_steps": [[1, 1]],
                    "full_pass_budget": 1,
                },
            },
        }
        svg = _frontier_chart_svg(frontier)
        self.assertNotIn("<script>", svg)
        self.assertIn("&lt;script&gt;", svg)
        self.assertIn("&amp;", svg)

    def test_single_budget_and_single_fixture(self) -> None:
        frontier = {
            "max_new_tokens": 1,
            "native_winner": "solo",
            "candidates": {
                "solo": {
                    "fixture_ids": ["f1"],
                    "pass_steps": [[1, 1]],
                    "full_pass_budget": 1,
                },
            },
        }
        svg = _frontier_chart_svg(frontier)
        paths = _paths(svg)
        self.assertEqual(len(paths), 1)
        # x(1)=0, y(1)=0, x(max)=0: a degenerate point path, still valid.
        self.assertEqual(paths[0][0], "M 0 0 H 0")
        self.assertEqual(paths[0][2], "3")


class TestSectionHook(unittest.TestCase):
    def test_frontier_section_embeds_chart(self) -> None:
        result = {"budget_frontier": _example_frontier()}
        html = reporting._frontier_section(result)
        self.assertIn("<h3>Frontier chart: all candidates</h3>", html)
        self.assertIn('class="frontier-svg"', html)
        self.assertIn('class="chart-wrap"', html)
        # The pre-existing spark table still renders after the chart.
        self.assertLess(
            html.index("Frontier chart: all candidates"),
            html.index("Pass curves by candidate"),
        )

    def test_section_without_frontier_unchanged(self) -> None:
        self.assertEqual(reporting._frontier_section({}), "")
        self.assertEqual(reporting._frontier_section({"budget_frontier": None}), "")


class TestMutationGate(unittest.TestCase):
    SOURCE = _REPO_ROOT / "model_protocol_lab" / "reporting.py"

    def test_control_passes(self) -> None:
        module = _load_mutant(self.SOURCE, "", "", "control")
        svg = module._frontier_chart_svg(_example_frontier())
        self.assertEqual(_paths(svg)[1][0], "M 0 160 H 186 V 80 H 497 V 0 H 560")

    def test_unsorted_legend_caught(self) -> None:
        # The sort lives in _sorted_str_keys (hostile-payload hardening):
        # mutating it to preserve insertion order must change the chart.
        module = _load_mutant(
            self.SOURCE,
            "    return sorted(key for key in candidates if type(key) is str)",
            "    return [key for key in candidates if type(key) is str]",
            "unsorted_legend",
        )
        forward = _example_frontier()
        reversed_frontier = {
            "max_new_tokens": 10,
            "native_winner": "alpha",
            "decision_steps": [[1, "no_candidate", None], [9, "selected", "alpha"]],
            "candidates": dict(reversed(list(forward["candidates"].items()))),
        }
        self.assertNotEqual(
            module._frontier_chart_svg(forward),
            module._frontier_chart_svg(reversed_frontier),
        )

    def test_shifted_x_mapping_caught(self) -> None:
        module = _load_mutant(
            self.SOURCE,
            "return (budget - 1) * width // denom_x",
            "return budget * width // max_tokens",
            "x_shift",
        )
        svg = module._frontier_chart_svg(_example_frontier())
        self.assertNotIn("M 0 160 H 186 V 80 H 497 V 0 H 560", svg)

    def test_rotated_palette_caught(self) -> None:
        module = _load_mutant(
            self.SOURCE,
            '    "#38bdf8",\n    "#22c55e",',
            '    "#22c55e",\n    "#38bdf8",',
            "palette_rot",
        )
        paths = _paths(module._frontier_chart_svg(_example_frontier()))
        # alpha must keep #38bdf8; the mutant swaps the first two colors.
        self.assertNotEqual(paths[1][1], "#38bdf8")

    def test_unescaped_legend_name_caught(self) -> None:
        module = _load_mutant(
            self.SOURCE,
            '+ escape(name)\n            + "</text>"',
            '+ name\n            + "</text>"',
            "no_escape",
        )
        evil = '"><script>alert(1)</script>'
        frontier = {
            "max_new_tokens": 5,
            "native_winner": None,
            "candidates": {
                evil: {
                    "fixture_ids": ["f1"],
                    "pass_steps": [[1, 1]],
                    "full_pass_budget": 1,
                }
            },
        }
        self.assertIn("<script>", module._frontier_chart_svg(frontier))

    def test_flipped_y_mapping_caught(self) -> None:
        module = _load_mutant(
            self.SOURCE,
            "return (y_total - count) * height // y_total",
            "return count * height // y_total",
            "y_flip",
        )
        svg = module._frontier_chart_svg(_example_frontier())
        self.assertNotIn("M 0 160 H 186 V 80 H 497 V 0 H 560", svg)


if __name__ == "__main__":
    unittest.main()
