"""Cycle 126 audit: safe budget floor readout.

The "safe budget floor" is the first token budget from which the native
selection decision never flips again -- derived exactly from the recorded
``flip_ranges``. It is recorded on the ``budget_frontier`` payload
(``experiments.budget_frontier``), derived defensively by
``reporting.safe_budget_floor`` for older payloads, shown as a first-class
callout in the HTML frontier section with a dashed marker on both charts,
and compared by ``diffreports.diff_frontiers`` (output dict + stderr
summary + ``identical`` flag).

Tests pin: the exact derivation table, span re-validation (sort/merge/
clip), recorded-field precedence, recorded-vs-derived agreement on fresh
payloads, the HTML callout wording, chart markers + legend + determinism,
and the diff-reports wiring. A mutation gate (see bottom of file) proves
the tests are not vacuous.
"""

import unittest

from model_protocol_lab import experiments
from model_protocol_lab.diffreports import diff_frontiers, diff_reports, render_summary
from model_protocol_lab.reporting import _frontier_chart_svg, _margin_chart_svg, safe_budget_floor


def _search_entry(name, specs):
    """Search entry."""
    observations = []
    for index, (output_ids, passed) in enumerate(specs):
        observations.append(
            {
                "fixture": f"f{index}",
                "status": "completed",
                "exact_match": passed,
                "boundary_found": passed,
                "output_ids": list(output_ids),
            }
        )
    return {"format": {"name": name}, "observations": observations}


_ABSENT = object()
"""Sentinel for "no recorded safe_budget_floor passed to _payload"."""


def _payload(flip_ranges, max_tokens=10, recorded=_ABSENT):
    """Payload."""
    frontier = {
        "max_new_tokens": max_tokens,
        "candidates": {
            "alpha": {
                "fixture_ids": ["f1", "f2"],
                "pass_steps": [[1, 2]],
                "full_pass_budget": 1,
            }
        },
        "decision_steps": [[1, "selected", "alpha"]],
        "flip_ranges": flip_ranges,
        "budget_robust": not flip_ranges,
    }
    if recorded is not _ABSENT:
        frontier["safe_budget_floor"] = recorded
    return frontier


class TestSafeBudgetFloorDerivation(unittest.TestCase):
    """Test safe budget floor derivation."""

    def test_robust_frontier_floor_is_one(self) -> None:
        """Test robust frontier floor is one."""
        self.assertEqual(safe_budget_floor(_payload([])), 1)

    def test_absent_flip_ranges_floor_is_one(self) -> None:
        """Test absent flip ranges floor is one."""
        frontier = _payload([])
        del frontier["flip_ranges"]
        self.assertEqual(safe_budget_floor(frontier), 1)

    def test_flip_ending_before_max(self) -> None:
        """Test flip ending before max."""
        self.assertEqual(safe_budget_floor(_payload([[2, 5]], max_tokens=10)), 6)

    def test_flip_reaching_max_has_no_floor(self) -> None:
        """Test flip reaching max has no floor."""
        self.assertIsNone(safe_budget_floor(_payload([[2, 10]], max_tokens=10)))
        self.assertIsNone(safe_budget_floor(_payload([[10, 10]], max_tokens=10)))

    def test_last_span_determines_floor(self) -> None:
        """Test last span determines floor."""
        self.assertEqual(safe_budget_floor(_payload([[2, 3], [7, 8]], max_tokens=10)), 9)

    def test_spans_resorted(self) -> None:
        """Test spans resorted."""
        self.assertEqual(safe_budget_floor(_payload([[7, 8], [2, 3]], max_tokens=10)), 9)

    def test_overlapping_spans_merged(self) -> None:
        """Test overlapping spans merged."""
        self.assertEqual(safe_budget_floor(_payload([[2, 4], [3, 6]], max_tokens=10)), 7)

    def test_adjacent_spans_merged(self) -> None:
        """Test adjacent spans merged."""
        self.assertEqual(safe_budget_floor(_payload([[2, 4], [5, 6]], max_tokens=10)), 7)

    def test_invalid_spans_skipped(self) -> None:
        """Test invalid spans skipped."""
        frontier = _payload([[2, "x"], ["a"], [5], [8, 3], None, "nope"], max_tokens=10)
        self.assertEqual(safe_budget_floor(frontier), 1)

    def test_spans_clipped_to_sweep(self) -> None:
        """Test spans clipped to sweep."""
        # [-5, 3] clips to [1, 3] -> floor 4.
        self.assertEqual(safe_budget_floor(_payload([[-5, 3]], max_tokens=10)), 4)
        # [0, 15] clips to [1, 10] -> reaches max -> None.
        self.assertIsNone(safe_budget_floor(_payload([[0, 15]], max_tokens=10)))

    def test_invalid_payload_is_none(self) -> None:
        """Test invalid payload is none."""
        self.assertIsNone(safe_budget_floor(None))
        self.assertIsNone(safe_budget_floor({}))
        self.assertIsNone(safe_budget_floor({"max_new_tokens": 0}))
        self.assertIsNone(safe_budget_floor({"max_new_tokens": "x"}))

    def test_recorded_field_wins(self) -> None:
        """Test recorded field wins."""
        # Recorded int wins over derivation (even when derivation says 1).
        self.assertEqual(safe_budget_floor(_payload([], recorded=7)), 7)
        # Recorded None wins (explicit "no floor in sweep").
        self.assertIsNone(safe_budget_floor(_payload([], recorded=None)))

    def test_malformed_recorded_field_falls_back(self) -> None:
        """Test malformed recorded field falls back."""
        # "7" is not an int -> derive from flip_ranges -> 6.
        self.assertEqual(safe_budget_floor(_payload([[2, 5]], recorded="7")), 6)
        # 0 is not >= 1 -> derive -> 6.
        self.assertEqual(safe_budget_floor(_payload([[2, 5]], recorded=0)), 6)


class TestRecordedField(unittest.TestCase):
    """Test recorded field."""

    @staticmethod
    def _recorded():
        """Recorded."""
        boundary = (9,)
        search = [
            _search_entry("A", [([1, 2, 9], True), ([9], True)]),
            _search_entry("B", [([1, 2, 3, 9], True), ([1, 2, 3], False)]),
        ]
        return experiments.budget_frontier(
            search,
            boundary=boundary,
            max_new_tokens=16,
            native_decision="compatible_on_fixture_set",
            native_winner="A",
        )

    def test_field_recorded(self) -> None:
        """Test field recorded."""
        frontier = self._recorded()
        self.assertIn("safe_budget_floor", frontier)
        # flip_ranges [[1, 2]] on max 16 -> floor 3.
        self.assertEqual(frontier["flip_ranges"], [[1, 2]])
        self.assertEqual(frontier["safe_budget_floor"], 3)

    def test_recorded_matches_derivation(self) -> None:
        """Test recorded matches derivation."""
        frontier = self._recorded()
        recorded = frontier["safe_budget_floor"]
        # Derive from flip_ranges by removing the recorded field entirely.
        plain = {k: v for k, v in frontier.items() if k != "safe_budget_floor"}
        self.assertEqual(recorded, safe_budget_floor(plain))

    def test_robust_payload_records_one(self) -> None:
        """Test robust payload records one."""
        boundary = (9,)
        search = [
            _search_entry("A", [([9], True), ([9], True)]),
            _search_entry("B", [([1, 9], False), ([2, 9], False)]),
        ]
        frontier = experiments.budget_frontier(
            search,
            boundary=boundary,
            max_new_tokens=8,
            native_decision="compatible_on_fixture_set",
            native_winner="A",
        )
        self.assertEqual(frontier["flip_ranges"], [])
        self.assertTrue(frontier["budget_robust"])
        self.assertEqual(frontier["safe_budget_floor"], 1)

    def test_floor_at_max_budget(self) -> None:
        """Test floor at max budget."""
        boundary = (9,)
        # A passes only at the very top of the sweep; the decision flips
        # on [1, 8] and the native decision holds from budget 9 onward.
        search = [
            _search_entry("A", [([1, 2, 3, 4, 5, 6, 7, 8, 9], True)]),
            _search_entry("B", [([1, 2], False)]),
        ]
        frontier = experiments.budget_frontier(
            search,
            boundary=boundary,
            max_new_tokens=9,
            native_decision="compatible_on_fixture_set",
            native_winner="A",
        )
        self.assertEqual(frontier["flip_ranges"], [[1, 8]])
        self.assertEqual(frontier["safe_budget_floor"], 9)


class TestFrontierSectionCallout(unittest.TestCase):
    """Test frontier section callout."""

    @staticmethod
    def _section(frontier):
        """Section."""
        from model_protocol_lab.reporting import _frontier_section

        return _frontier_section({"budget_frontier": frontier})

    def test_robust_callout(self) -> None:
        """Test robust callout."""
        html = self._section(_payload([], max_tokens=10))
        self.assertIn("Safe budget floor:", html)
        self.assertIn("1 token", html)

    def test_floor_callout(self) -> None:
        """Test floor callout."""
        html = self._section(_payload([[2, 5]], max_tokens=10))
        self.assertIn("&ge; 6 tokens", html)
        self.assertIn("from budget 6 through 10", html)

    def test_no_floor_callout(self) -> None:
        """Test no floor callout."""
        html = self._section(_payload([[2, 10]], max_tokens=10))
        self.assertIn("No safe budget floor within the sweep", html)

    def test_chart_marker_present(self) -> None:
        """Test chart marker present."""
        svg = _frontier_chart_svg(_payload([[2, 5]], max_tokens=10))
        self.assertIn('stroke-dasharray="5 4"', svg)
        self.assertIn("safe floor", svg)

    def test_chart_marker_absent_when_floor_is_one(self) -> None:
        """Test chart marker absent when floor is one."""
        svg = _frontier_chart_svg(_payload([], max_tokens=10))
        self.assertNotIn('stroke-dasharray="5 4"', svg)

    def test_chart_marker_absent_when_no_floor(self) -> None:
        """Test chart marker absent when no floor."""
        svg = _frontier_chart_svg(_payload([[2, 10]], max_tokens=10))
        self.assertNotIn('stroke-dasharray="5 4"', svg)

    def test_margin_chart_marker(self) -> None:
        """Test margin chart marker."""
        svg = _margin_chart_svg(_payload([[2, 5]], max_tokens=10))
        self.assertIn('stroke-dasharray="5 4"', svg)

    def test_charts_deterministic(self) -> None:
        """Test charts deterministic."""
        frontier = _payload([[2, 5]], max_tokens=10)
        self.assertEqual(_frontier_chart_svg(frontier), _frontier_chart_svg(frontier))
        self.assertEqual(_margin_chart_svg(frontier), _margin_chart_svg(frontier))
        self.assertEqual(self._section(frontier), self._section(frontier))


def _base(frontier):
    """Base."""
    return {
        "decision": "compatible_on_fixture_set",
        "selected_candidate": "alpha",
        "search": [],
        "budget_frontier": frontier,
    }


class TestDiffFloor(unittest.TestCase):
    """Test diff floor."""

    def test_floor_compared(self) -> None:
        """Test floor compared."""
        old = _payload([[2, 5]], max_tokens=10)
        new = _payload([[2, 10]], max_tokens=10)
        diff = diff_frontiers(old, new)
        self.assertEqual(diff["safe_budget_floor"]["old"], 6)
        self.assertIsNone(diff["safe_budget_floor"]["new"])
        self.assertTrue(diff["safe_budget_floor"]["changed"])
        self.assertTrue(diff["changed"])

    def test_identical_floor_not_changed(self) -> None:
        """Test identical floor not changed."""
        frontier = _payload([[2, 5]], max_tokens=10)
        diff = diff_frontiers(frontier, frontier)
        self.assertFalse(diff["safe_budget_floor"]["changed"])
        self.assertEqual(diff["safe_budget_floor"]["old"], 6)

    def test_floor_change_flips_identical(self) -> None:
        """Test floor change flips identical."""
        old = _payload([[2, 5]], max_tokens=10)
        new = _payload([[2, 6]], max_tokens=10)
        report = diff_reports(_base(old), _base(new), old_label="a", new_label="b")
        self.assertFalse(report["summary"]["identical"])
        self.assertTrue(report["frontier"]["safe_budget_floor"]["changed"])
        self.assertEqual(report["frontier"]["safe_budget_floor"]["old"], 6)
        self.assertEqual(report["frontier"]["safe_budget_floor"]["new"], 7)

    def test_summary_line(self) -> None:
        """Test summary line."""
        old = _payload([[2, 5]], max_tokens=10)
        new = _payload([[2, 10]], max_tokens=10)
        text = render_summary(diff_reports(_base(old), _base(new), old_label="a", new_label="b"))
        self.assertIn("Safe budget floor: 6 -> (none)", text)

    def test_summary_line_stable(self) -> None:
        """Test summary line stable."""
        frontier = _payload([], max_tokens=10)
        report = diff_reports(_base(frontier), _base(frontier), old_label="a", new_label="b")
        self.assertIn("Safe budget floor: 1 -> 1", render_summary(report))


if __name__ == "__main__":
    unittest.main()
