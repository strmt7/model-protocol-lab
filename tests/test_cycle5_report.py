"""Cycle 5 report upgrades: self-contained property tests.

The report stays a single file with no remote references (mirroring
pytest-html's CSP-conscious discipline); sortable tables carry data-sort
keys; the TOC anchors resolve to rendered sections; the one inline script
is hash-allowlisted in the Content-Security-Policy; <details> accordions
are exclusive via name=; and the committed sample report regenerates
byte-identically through the repo script.
"""

import base64
import hashlib
import importlib.util
import re
import unittest
from html.parser import HTMLParser
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from model_protocol_lab.reporting import (
    _SCRIPT,
    _frontier_spark_table,
    _spark_svg,
    render_html,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "examples" / "make_sample_report.py"
COMMITTED = REPO_ROOT / "examples" / "sample-report"


class _ScriptCollector(HTMLParser):
    """Collect <script> element bodies without case-sensitive regexes.

    Cycle 217 (CodeQL py/bad-tag-filter): a lowercase-only
    ``<script>`` regex misses uppercase tags, so a renderer change
    that emitted ``<SCRIPT>`` would silently void the extraction.
    The parser lowercases tag names and reads script content as
    raw data, so every script element is found regardless of case.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.bodies: list[str] = []
        self._current: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "script":
            self._current = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._current is not None:
            self.bodies.append("".join(self._current))
            self._current = None

    def handle_data(self, data: str) -> None:
        if self._current is not None:
            self._current.append(data)


def _script_bodies(html: str) -> list[str]:
    collector = _ScriptCollector()
    collector.feed(html)
    collector.close()
    return collector.bodies


def _obs(fixture: str, passed: bool) -> dict[str, Any]:
    return {
        "fixture": fixture,
        "status": "completed",
        "exact_match": passed,
        "boundary_found": passed,
    }


def _rich_result() -> dict[str, Any]:
    """Result exercising every report section: config, search, outcome
    grid, decisive fixtures, conformance, stop matrix, budget frontier,
    sensitivity sweep and held-out."""
    return {
        "decision": "compatible_on_fixture_set",
        "selected_candidate": "cand-a",
        "selection_rule": "unique_all_search_pass",
        "measurement": "exact_answer_and_boundary",
        "seed": 42,
        "max_new_tokens": 16,
        "statistical_confidence": "not_estimated",
        "compute_comparison": "not_equal_compute_certified",
        "backend_fingerprint": "synthetic-backend",
        "search": [
            {
                "format": {"name": "cand-a"},
                "boundary_token_ids": [124],
                "candidate_fingerprint": "fp-a",
                "summary": {
                    "status": "completed",
                    "rate": 1.0,
                    "attempted": 2,
                    "planned": 2,
                    "exact_and_boundary_passes": 2,
                    "encoded_input_tokens": 100,
                    "retained_output_tokens": 32,
                },
                "observations": [_obs("s1", True), _obs("s2", True)],
            },
            {
                "format": {"name": "cand-b"},
                "boundary_token_ids": [124],
                "candidate_fingerprint": "fp-b",
                "summary": {
                    "status": "completed",
                    "rate": 0.5,
                    "attempted": 2,
                    "planned": 2,
                    "exact_and_boundary_passes": 1,
                    "encoded_input_tokens": 110,
                    "retained_output_tokens": 32,
                },
                "observations": [_obs("s1", True), _obs("s2", False)],
            },
        ],
        "held_out": {
            "summary": {
                "status": "completed",
                "rate": 1.0,
                "exact_and_boundary_passes": 2,
                "planned": 2,
                "aggregate_eligible": True,
            }
        },
        "template_conformance": {
            "status": "ok",
            "diffs": [
                {
                    "status": "ok",
                    "format": "cand-a",
                    "fixture": "s1",
                    "hand_tokens": 10,
                    "template_tokens": 10,
                    "diff": [],
                },
                {
                    "status": "ok",
                    "format": "cand-a",
                    "fixture": "s2",
                    "hand_tokens": 12,
                    "template_tokens": 14,
                    "diff": [
                        {
                            "hand_span": [0, 1],
                            "hand_ids": [1],
                            "template_span": [0, 1],
                            "template_ids": [2],
                        }
                    ],
                },
            ],
        },
        "stop_matrix": {
            "semantics": [
                {"id": "native", "label": "Native (lab)", "description": "Lab stop semantics."},
                {"id": "vllm", "label": "vLLM", "description": "vLLM stop semantics."},
            ],
            "search": [
                {
                    "fixture": "s1",
                    "native_pass": True,
                    "cells": {"vllm": {"available": True, "pass": True, "fired": False}},
                    "flips": [],
                },
            ],
        },
        "budget_frontier": {
            "max_new_tokens": 16,
            "budget_robust": False,
            "native_winner": "cand-a",
            "candidates": {
                "cand-a": {
                    "fixture_ids": ["s1", "s2"],
                    "min_budget_per_fixture": [3, 5],
                    "full_pass_budget": 5,
                    "pass_steps": [[1, 0], [3, 1], [5, 2]],
                },
                "cand-b": {
                    "fixture_ids": ["s1", "s2"],
                    "min_budget_per_fixture": [None, None],
                    "full_pass_budget": None,
                    "pass_steps": [[1, 0]],
                },
            },
            "decision_steps": [
                [1, "no_candidate", None],
                [5, "compatible_on_fixture_set", "cand-a"],
            ],
            "flip_ranges": [[1, 4]],
        },
        "sensitivity_sweep": {
            "depths": [
                {
                    "drop_bytes": 1,
                    "decision": "compatible_on_fixture_set",
                    "selected_candidate": "cand-a",
                    "winner_changed": False,
                },
                {
                    "drop_bytes": 2,
                    "decision": "ambiguous",
                    "selected_candidate": None,
                    "winner_changed": True,
                },
            ],
            "native_winner": "cand-a",
            "native_decision": "compatible_on_fixture_set",
            "flips_at": [2],
        },
    }


class NoRemoteReferencesTests(unittest.TestCase):
    def assert_no_remote_references(self, html: str) -> None:
        self.assertNotIn("http://", html)
        self.assertNotIn("https://", html)
        # No src= resource attributes anywhere (<script src>, <img>, ...).
        self.assertIsNone(re.search(r"<[^>]*\ssrc\s*=", html))
        # No protocol-relative or absolute URLs in any attribute value.
        for value in re.findall(r'="([^"]*)"', html):
            self.assertFalse(value.startswith("//"), f"protocol-relative URL: {value[:60]}")
            self.assertNotIn("://", value)
        # No external stylesheet/script hooks of any kind.
        self.assertNotIn("<link", html)
        self.assertNotIn("@import", html)

    def test_render_html_has_no_remote_references(self) -> None:
        self.assert_no_remote_references(render_html(_rich_result()))

    def test_render_html_minimal_has_no_remote_references(self) -> None:
        self.assert_no_remote_references(render_html({"decision": "failed", "search": []}))

    def test_committed_sample_report_has_no_remote_references(self) -> None:
        html = (COMMITTED / "report.html").read_text(encoding="utf-8")
        self.assert_no_remote_references(html)


class SortableTableTests(unittest.TestCase):
    def test_sortable_tables_carry_data_sort_keys(self) -> None:
        html = render_html(_rich_result())
        tables = re.findall(r'<table class="sortable">(.*?)</table>', html, re.DOTALL)
        # Search candidates, outcome grid, conformance, frontier curves, sweep.
        self.assertGreaterEqual(len(tables), 4)
        for table in tables:
            tbody = re.search(r"<tbody>(.*?)</tbody>", table, re.DOTALL)
            self.assertIsNotNone(tbody, "sortable table without tbody")
            assert tbody is not None
            cells = re.findall(r"<t[dh][^>]*>", tbody.group(1))
            self.assertGreater(len(cells), 0)
            for cell in cells:
                self.assertIn("data-sort=", cell, f"sortable cell lacks key: {cell[:70]}")

    def test_sort_script_is_single_and_inert_safe(self) -> None:
        html = render_html(_rich_result())
        bodies = _script_bodies(html)
        self.assertEqual(len(bodies), 1)
        body = bodies[0]
        self.assertIn("document.querySelectorAll", body)  # inert-safe guard
        self.assertIn("data-sort", body)
        self.assertNotIn("</script>", body)  # never breaks out of the element

    def test_numeric_keys_sort_numerically(self) -> None:
        # data-sort keys for counts/rates are plain numbers, which the
        # inline sorter compares numerically (parseFloat branch).
        html = render_html(_rich_result())
        self.assertIn('data-sort="1.000000"', html)
        self.assertIn('data-sort="0.500000"', html)


class TocTests(unittest.TestCase):
    def test_toc_anchors_resolve(self) -> None:
        html = render_html(_rich_result())
        nav = re.search(r'<nav class="toc".*?</nav>', html, re.DOTALL)
        self.assertIsNotNone(nav)
        assert nav is not None
        anchors = re.findall(r'href="#([^"]+)"', nav.group(0))
        self.assertGreaterEqual(len(anchors), 5)
        ids = set(re.findall(r'id="([^"]+)"', html))
        for anchor in anchors:
            self.assertIn(anchor, ids, f"TOC anchor #{anchor} has no target")

    def test_toc_skips_unrendered_sections(self) -> None:
        html = render_html({"decision": "failed", "search": []})
        nav = re.search(r'<nav class="toc".*?</nav>', html, re.DOTALL)
        self.assertIsNotNone(nav)
        assert nav is not None
        self.assertNotIn("#token-budget-frontier", nav.group(0))
        self.assertIn("#search-candidates", nav.group(0))


class CspScriptTests(unittest.TestCase):
    def test_csp_hash_allowlist_matches_emitted_script(self) -> None:
        html = render_html(_rich_result())
        bodies = _script_bodies(html)
        self.assertEqual(len(bodies), 1)
        body = bodies[0]
        # The emitted script is exactly the module constant: no per-render
        # mutation, so the hash is stable and byte-deterministic.
        self.assertEqual(body, _SCRIPT)
        digest = base64.b64encode(hashlib.sha256(body.encode("utf-8")).digest()).decode("ascii")
        csp = re.search(r'<meta http-equiv="Content-Security-Policy" content="([^"]*)"', html)
        self.assertIsNotNone(csp)
        assert csp is not None
        policy = csp.group(1)
        self.assertIn("default-src 'none'", policy)
        self.assertIn(f"'sha256-{digest}'", policy)


class SvgSparklineTests(unittest.TestCase):
    def test_svg_has_no_text_glyphs(self) -> None:
        svg = _spark_svg([0, 2, 4], 4, "label")
        self.assertIn("<svg", svg)
        self.assertIn("<rect", svg)
        for glyph in "▁▂▃▄▅▆▇█":
            self.assertNotIn(glyph, svg)

    def test_svg_degenerate_inputs(self) -> None:
        self.assertEqual(_spark_svg([], 4, "x"), "?")
        self.assertEqual(_spark_svg([1], 0, "x"), "?")

    def test_svg_is_deterministic(self) -> None:
        self.assertEqual(_spark_svg([0, 2, 4], 4, "l"), _spark_svg([0, 2, 4], 4, "l"))

    def test_svg_geometry_matches_quantized_counts(self) -> None:
        # counts [0,0,1,1,2,2,2,2] of 2 -> levels 0,0,4,4,8,8,8,8 on an 8-high grid.
        svg = _spark_svg([0, 0, 1, 1, 2, 2, 2, 2], 2, "l")
        self.assertEqual(svg.count("<rect"), 8)
        self.assertIn('height="8"', svg)
        self.assertIn('height="4"', svg)
        self.assertIn('height="0"', svg)

    def test_frontier_table_uses_svg_with_accessible_data_label(self) -> None:
        frontier = {
            "max_new_tokens": 8,
            "candidates": {
                "a": {
                    "fixture_ids": ["s1", "s2"],
                    "min_budget_per_fixture": [3, 5],
                    "full_pass_budget": 5,
                },
            },
        }
        html = _frontier_spark_table(frontier)
        self.assertIn("<svg", html)
        self.assertIn('aria-label="pass count', html)
        self.assertIn("over budgets 1 to 8", html)


class DetailsAccordionTests(unittest.TestCase):
    def test_details_groups_are_named(self) -> None:
        html = render_html(_rich_result())
        self.assertIn('name="candidate-details"', html)
        self.assertIn('name="conformance-details"', html)
        self.assertIn('name="config-details"', html)


class DeterminismTests(unittest.TestCase):
    def test_render_html_is_byte_deterministic(self) -> None:
        result = _rich_result()
        self.assertEqual(render_html(result), render_html(result))


class SampleReportRegenTests(unittest.TestCase):
    def test_sample_report_regenerates_byte_identically(self) -> None:
        spec = importlib.util.spec_from_file_location("make_sample_report", SCRIPT)
        self.assertIsNotNone(spec)
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        with TemporaryDirectory() as tmp:
            module.build_sample_report(Path(tmp) / "regen")
            for name in ("results.json", "report.html"):
                self.assertEqual(
                    (Path(tmp) / "regen" / name).read_bytes(),
                    (COMMITTED / name).read_bytes(),
                    name,
                )


class HostileAttributeEscapingTests(unittest.TestCase):
    def test_hostile_strings_neutralized_in_data_sort_attributes(self) -> None:
        # Pins the attribute-injection fix: hostile candidate/fixture names
        # must render escaped inside data-sort="..." attributes (and element
        # text), including the budget-frontier SVG sparkline section. This
        # bug class escaped once already; the fix must not regress silently.
        evil = '"><script>alert(1)</script><img src=x onerror=alert(2)>'
        result = _rich_result()
        for entry in result["search"]:
            entry["format"]["name"] = evil
            for obs in entry["observations"]:
                obs["fixture"] = evil
        frontier = result["budget_frontier"]
        frontier["candidates"] = {evil: frontier["candidates"]["cand-a"]}
        frontier["native_winner"] = evil
        html = render_html(result)
        self.assertNotIn(evil, html)
        self.assertIn("&lt;script&gt;", html)
        for match in re.finditer(r'data-sort="([^"]*)"', html):
            value = match.group(1)
            self.assertNotIn("<", value)
            self.assertNotIn(">", value)
            self.assertNotIn('"', value.replace("&quot;", ""))


if __name__ == "__main__":
    unittest.main()
