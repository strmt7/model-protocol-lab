"""Cycle 128 audit: flip-attribution comparison in diff-reports.

``diffreports.diff_frontiers`` now compares ``flip_attribution`` per side:
two runs can share identical flip spans while the *drivers* of the flip
differ (e.g. a template tweak moves the flip from one fixture to
another), and the diff must not call that identical. The comparison
normalizes each side (malformed entries skipped, crossings sorted), so
crossing order never counts as a difference; a side without the field
(an older payload) compares as unknown. Differences feed the ``changed``
flag and ``identical`` verdict, and the stderr summary prints per-span
old/new crossing lines with terminal-control characters stripped.

Tests pin: identical attributions, same-span/different-crossings,
different spans, one-sided and two-sided missing fields, malformed
entries, order-insensitivity, the ``identical`` wiring, summary wording
and sanitization, and determinism. A mutation gate (applied during
development, then reverted) proves the tests are not vacuous.
"""

import unittest

from model_protocol_lab.diffreports import (
    _normalize_attribution,
    diff_frontiers,
    diff_reports,
    render_summary,
)

_ABSENT = object()
"""Sentinel for "no flip_attribution key in the synthetic payload"."""


def _frontier(attribution=_ABSENT, max_tokens=10):
    payload = {
        "max_new_tokens": max_tokens,
        "candidates": {
            "A": {
                "fixture_ids": ["f0"],
                "pass_steps": [[1, 0], [3, 1]],
                "full_pass_budget": 3,
            }
        },
        "decision_steps": [[1, "no_candidate", None], [3, "selected", "A"]],
        "flip_ranges": [[1, 2]],
        "budget_robust": False,
        "safe_budget_floor": 3,
    }
    if attribution is not _ABSENT:
        payload["flip_attribution"] = attribution
    return payload


def _crossing(candidate, fixture, min_budget):
    return {"candidate": candidate, "fixture": fixture, "min_budget": min_budget}


def _attribution(span, crossings):
    return [{"span": list(span), "crossings": list(crossings)}]


def _base(frontier):
    return {
        "decision": "compatible_on_fixture_set",
        "selected_candidate": "A",
        "search": [],
        "budget_frontier": frontier,
    }


class TestNormalizeAttribution(unittest.TestCase):
    def test_sorts_crossings(self):
        value = _attribution(
            [1, 2],
            [_crossing("B", "f0", 2), _crossing("A", "f1", 1)],
        )
        normalized = _normalize_attribution(value)
        self.assertEqual([c["candidate"] for c in normalized[0]["crossings"]], ["A", "B"])

    def test_absent_field_is_none(self):
        self.assertIsNone(_normalize_attribution(None))
        self.assertIsNone(_normalize_attribution("nope"))
        self.assertIsNone(_normalize_attribution({"span": [1, 2]}))

    def test_empty_list_stays_empty(self):
        self.assertEqual(_normalize_attribution([]), [])

    def test_malformed_entries_skipped(self):
        value = [
            {"span": "bogus"},
            {"span": [1, 2]},  # missing crossings -> kept, empty
            {"span": [1, 2], "crossings": "nope"},  # not a list -> kept, empty
            {"span": [1, 2], "crossings": [{"candidate": "A"}]},  # sparse -> skipped
            {"span": [1, 2], "crossings": [_crossing("A", "f0", 2)]},
            None,
            42,
        ]
        normalized = _normalize_attribution(value)
        self.assertEqual(len(normalized), 4)
        self.assertEqual(normalized[0]["crossings"], [])
        self.assertEqual(normalized[1]["crossings"], [])
        self.assertEqual(normalized[2]["crossings"], [])
        self.assertEqual(len(normalized[3]["crossings"]), 1)

    def test_bool_min_budget_rejected(self):
        value = _attribution([1, 2], [_crossing("A", "f0", True)])
        self.assertEqual(_normalize_attribution(value)[0]["crossings"], [])


class TestAttributionDiff(unittest.TestCase):
    def test_identical_attributions_not_changed(self):
        attribution = _attribution([1, 2], [_crossing("A", "f0", 2)])
        diff = diff_frontiers(_frontier(attribution), _frontier(attribution))
        self.assertFalse(diff["flip_attribution"]["changed"])
        self.assertFalse(diff["changed"])

    def test_same_spans_different_crossings_changed(self):
        old = _frontier(_attribution([1, 2], [_crossing("A", "f0", 2)]))
        new = _frontier(_attribution([1, 2], [_crossing("A", "f1", 2)]))
        diff = diff_frontiers(old, new)
        self.assertTrue(diff["flip_attribution"]["changed"])
        self.assertTrue(diff["changed"])
        self.assertEqual(
            diff["flip_attribution"]["old"][0]["crossings"],
            [_crossing("A", "f0", 2)],
        )
        self.assertEqual(
            diff["flip_attribution"]["new"][0]["crossings"],
            [_crossing("A", "f1", 2)],
        )

    def test_crossing_order_irrelevant(self):
        old = _frontier(_attribution([1, 2], [_crossing("A", "f0", 1), _crossing("B", "f0", 2)]))
        new = _frontier(_attribution([1, 2], [_crossing("B", "f0", 2), _crossing("A", "f0", 1)]))
        diff = diff_frontiers(old, new)
        self.assertFalse(diff["flip_attribution"]["changed"])

    def test_missing_attribution_one_side_changed(self):
        old = _frontier()  # no flip_attribution key
        new = _frontier(_attribution([1, 2], [_crossing("A", "f0", 2)]))
        diff = diff_frontiers(old, new)
        self.assertTrue(diff["flip_attribution"]["changed"])
        self.assertTrue(diff["changed"])
        self.assertIsNone(diff["flip_attribution"]["old"])
        self.assertIsNotNone(diff["flip_attribution"]["new"])

    def test_missing_attribution_both_sides_not_changed(self):
        diff = diff_frontiers(_frontier(), _frontier())
        self.assertFalse(diff["flip_attribution"]["changed"])
        self.assertFalse(diff["changed"])

    def test_changed_feeds_identical(self):
        old = _frontier(_attribution([1, 2], [_crossing("A", "f0", 2)]))
        same = _frontier(_attribution([1, 2], [_crossing("A", "f0", 2)]))
        report = diff_reports(_base(old), _base(same), old_label="a", new_label="b")
        self.assertTrue(report["summary"]["identical"])
        new = _frontier(_attribution([1, 2], [_crossing("A", "f1", 2)]))
        report = diff_reports(_base(old), _base(new), old_label="a", new_label="b")
        self.assertFalse(report["summary"]["identical"])
        self.assertTrue(report["summary"]["frontier_changed"])

    def test_summary_lines_for_changed_attribution(self):
        old = _frontier(_attribution([1, 2], [_crossing("A", "f0", 2)]))
        new = _frontier(_attribution([1, 2], [_crossing("A", "f1", 2)]))
        text = render_summary(diff_reports(_base(old), _base(new), old_label="a", new_label="b"))
        self.assertIn("Flip attribution differs:", text)
        self.assertIn("budgets 1\u20132 (old): A/f0 @ 2", text)
        self.assertIn("budgets 1\u20132 (new): A/f1 @ 2", text)

    def test_summary_line_for_missing_side(self):
        old = _frontier()
        new = _frontier(_attribution([1, 2], [_crossing("A", "f0", 2)]))
        text = render_summary(diff_reports(_base(old), _base(new), old_label="a", new_label="b"))
        self.assertIn("Flip attribution: not recorded by old run.", text)

    def test_summary_sanitizes_control_chars(self):
        old = _frontier(_attribution([1, 2], [_crossing("A\x1b[2J", "f\x00f0", 2)]))
        new = _frontier(_attribution([1, 2], [_crossing("A", "f0", 2)]))
        text = render_summary(diff_reports(_base(old), _base(new), old_label="a", new_label="b"))
        self.assertIn("A[2J/ff0 @ 2", text)
        self.assertNotIn("\x1b", text)
        self.assertNotIn("\x00", text)

    def test_empty_crossings_summary_wording(self):
        old = _frontier(_attribution([1, 2], []))
        new = _frontier(_attribution([1, 2], [_crossing("A", "f0", 2)]))
        text = render_summary(diff_reports(_base(old), _base(new), old_label="a", new_label="b"))
        self.assertIn("(no pass-status changes in span)", text)

    def test_diff_deterministic(self):
        old = _frontier(_attribution([1, 2], [_crossing("A", "f0", 2)]))
        new = _frontier(_attribution([1, 2], [_crossing("A", "f1", 2)]))
        first = diff_frontiers(old, new)
        second = diff_frontiers(old, new)
        self.assertEqual(first, second)
        self.assertEqual(
            render_summary(diff_reports(_base(old), _base(new), old_label="a", new_label="b")),
            render_summary(diff_reports(_base(old), _base(new), old_label="a", new_label="b")),
        )


if __name__ == "__main__":
    unittest.main()
