"""Cycle 9: automatic selection-margin and token-overhead diagnostics.

Covers ``reporting.selection_margin`` and ``reporting.token_overhead`` --
both pure functions of the recorded search outcomes -- plus their wiring
into the decision banner, the token-overhead card, and ``results.json``.
"""

import json
import sys
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, "tests")

from test_core import BASE, FakeBackend, ids  # noqa: E402
from test_round3 import split  # noqa: E402

from model_protocol_lab.core import Generation  # noqa: E402
from model_protocol_lab.experiments import run_experiment  # noqa: E402
from model_protocol_lab.reporting import (  # noqa: E402
    attach_diagnostics,
    render_html,
    save_report,
    selection_margin,
    token_overhead,
)

OK = Generation(ids("ok|"), "limit")


def _obs(fixture, passed, *, n_in=10, n_out=5, status="completed"):
    """Obs."""
    return {
        "fixture": fixture,
        "status": status,
        "exact_match": passed if status == "completed" else False,
        "boundary_found": passed if status == "completed" else False,
        "input_ids": list(range(n_in)),
        "output_ids": list(range(n_out)),
    }


def _entry(name, outcomes):
    """outcomes: list of (fixture_id, passed) or (fixture_id, passed, status)."""
    observations = []
    for outcome in outcomes:
        if len(outcome) == 2:
            fixture, passed = outcome
            observations.append(_obs(fixture, passed))
        else:
            fixture, passed, status = outcome
            observations.append(_obs(fixture, passed, status=status))
    return {"format": {"name": name}, "observations": observations}


def _result(decision, entries, selected=None):
    """Result."""
    return {"decision": decision, "selected_candidate": selected, "search": entries}


class SelectionMarginTests(unittest.TestCase):
    """Selection margin tests."""

    def test_unique_winner_margin_is_closest_runner_up(self) -> None:
        """Test unique winner margin is closest runner up."""
        result = _result(
            "compatible_on_fixture_set",
            [
                _entry("winner", [("f1", True), ("f2", True), ("f3", True)]),
                _entry("a", [("f1", True), ("f2", False), ("f3", True)]),
                _entry("b", [("f1", False), ("f2", True), ("f3", False)]),
            ],
            selected="winner",
        )
        self.assertEqual(
            selection_margin(result),
            {
                "value": 1,
                "direction": "to_tie",
                "runner_up": "a",
                "decisive_fixtures": ["f2"],
            },
        )

    def test_unique_winner_runner_up_tie_breaks_by_name(self) -> None:
        """Test unique winner runner up tie breaks by name."""
        result = _result(
            "held_out_failed",
            [
                _entry("winner", [("f1", True), ("f2", True)]),
                _entry("zeta", [("f1", False), ("f2", True)]),
                _entry("alpha", [("f1", True), ("f2", False)]),
            ],
            selected="winner",
        )
        margin = selection_margin(result)
        self.assertIsNotNone(margin)
        assert margin is not None
        self.assertEqual(margin["value"], 1)
        self.assertEqual(margin["runner_up"], "alpha")
        self.assertEqual(margin["decisive_fixtures"], ["f2"])

    def test_no_candidate_margin_is_closest_candidate(self) -> None:
        """Test no candidate margin is closest candidate."""
        result = _result(
            "no_candidate",
            [
                _entry("a", [("f1", False), ("f2", False), ("f3", True)]),
                _entry("b", [("f1", True), ("f2", False), ("f3", True)]),
            ],
        )
        self.assertEqual(
            selection_margin(result),
            {
                "value": 1,
                "direction": "to_winner",
                "runner_up": "b",
                "decisive_fixtures": ["f2"],
            },
        )

    def test_ambiguous_margin_counts_demotions(self) -> None:
        """Test ambiguous margin counts demotions."""
        fixtures = [("f1", True), ("f2", True), ("f3", True)]
        result = _result(
            "ambiguous",
            [_entry("a", fixtures), _entry("b", fixtures), _entry("c", fixtures)],
        )
        self.assertEqual(
            selection_margin(result),
            {
                "value": 2,
                "direction": "to_unique",
                "runner_up": None,
                "decisive_fixtures": ["f1", "f2", "f3"],
            },
        )

    def test_ambiguous_two_way_tie(self) -> None:
        """Test ambiguous two way tie."""
        result = _result(
            "ambiguous",
            [
                _entry("a", [("f1", True), ("f2", True)]),
                _entry("b", [("f1", True), ("f2", True)]),
                _entry("c", [("f1", True), ("f2", False)]),
            ],
        )
        margin = selection_margin(result)
        self.assertIsNotNone(margin)
        assert margin is not None
        self.assertEqual(margin["value"], 1)
        self.assertEqual(margin["runner_up"], None)
        # Only fixtures passed by every tied candidate keep the tie alive.
        self.assertEqual(margin["decisive_fixtures"], ["f1", "f2"])

    def test_failed_and_cancelled_have_no_margin(self) -> None:
        """Test failed and cancelled have no margin."""
        entries = [_entry("a", [("f1", True, "failed")])]
        self.assertIsNone(selection_margin(_result("failed", entries)))
        self.assertIsNone(selection_margin(_result("cancelled", entries)))

    def test_single_fixture_unique_winner(self) -> None:
        """Test single fixture unique winner."""
        result = _result(
            "compatible_on_fixture_set",
            [_entry("winner", [("only", True)]), _entry("runner", [("only", False)])],
            selected="winner",
        )
        margin = selection_margin(result)
        self.assertIsNotNone(margin)
        assert margin is not None
        self.assertEqual((margin["value"], margin["runner_up"]), (1, "runner"))

    def test_single_candidate_winner_has_undefined_margin(self) -> None:
        """Test single candidate winner has undefined margin."""
        result = _result(
            "compatible_on_fixture_set",
            [_entry("solo", [("f1", True), ("f2", True)])],
            selected="solo",
        )
        self.assertEqual(
            selection_margin(result),
            {"value": None, "direction": "to_tie", "runner_up": None, "decisive_fixtures": []},
        )

    def test_all_fail_no_candidate(self) -> None:
        """Test all fail no candidate."""
        result = _result(
            "no_candidate",
            [_entry("a", [("f1", False), ("f2", False), ("f3", False)])],
        )
        self.assertEqual(
            selection_margin(result),
            {
                "value": 3,
                "direction": "to_winner",
                "runner_up": "a",
                "decisive_fixtures": ["f1", "f2", "f3"],
            },
        )

    def test_incomplete_observation_counts_as_failing(self) -> None:
        """Test incomplete observation counts as failing."""
        result = _result(
            "compatible_on_fixture_set",
            [
                _entry("winner", [("f1", True), ("f2", True)]),
                _entry("runner", [("f1", True, "completed"), ("f2", False, "failed")]),
            ],
            selected="winner",
        )
        margin = selection_margin(result)
        self.assertIsNotNone(margin)
        assert margin is not None
        self.assertEqual(margin["value"], 1)
        self.assertEqual(margin["decisive_fixtures"], ["f2"])

    def test_ambiguous_without_two_all_pass_candidates_is_absent(self) -> None:
        """Test ambiguous without two all pass candidates is absent."""
        # Inconsistent records cannot yield a meaningful margin.
        result = _result(
            "ambiguous",
            [_entry("a", [("f1", True), ("f2", False)])],
        )
        self.assertIsNone(selection_margin(result))

    def test_margin_rejects_non_dict(self) -> None:
        """Test margin rejects non dict."""
        with self.assertRaises(ValueError):
            selection_margin([])  # type: ignore[arg-type]


class TokenOverheadTests(unittest.TestCase):
    """Token overhead tests."""

    def test_hand_computed_means_and_tax(self) -> None:
        """Test hand computed means and tax."""
        result = _result(
            "compatible_on_fixture_set",
            [
                _entry("cheap", [("f1", True), ("f2", True)]),
                _entry("pricey", [("f1", True), ("f2", True)]),
            ],
            selected="pricey",
        )
        # cheap: prompt [10, 10] -> mean 10.0/max 10; output [5, 5] -> 5.0.
        # pricey: prompt [10, 10] -> 10.0 (same helper default).
        cheap_entry, pricey_entry = result["search"]
        cheap_entry["observations"][0]["input_ids"] = list(range(8))
        cheap_entry["observations"][1]["input_ids"] = list(range(12))
        pricey_entry["observations"][0]["input_ids"] = list(range(30))
        pricey_entry["observations"][1]["input_ids"] = list(range(30))
        overhead = token_overhead(result)
        self.assertEqual(
            overhead["per_candidate"]["cheap"],
            {"mean_prompt_tokens": 10.0, "max_prompt_tokens": 12, "mean_output_tokens": 5.0},
        )
        self.assertEqual(
            overhead["per_candidate"]["pricey"],
            {"mean_prompt_tokens": 30.0, "max_prompt_tokens": 30, "mean_output_tokens": 5.0},
        )
        self.assertEqual(overhead["cheapest_candidate"], "cheap")
        # (30 - 10) / 10 * 100
        self.assertEqual(overhead["winner_tax_vs_cheapest_pct"], 200.0)

    def test_winner_is_cheapest_reports_zero_tax(self) -> None:
        """Test winner is cheapest reports zero tax."""
        result = _result(
            "compatible_on_fixture_set",
            [_entry("a", [("f1", True)]), _entry("b", [("f1", True)])],
            selected="a",
        )
        result["search"][1]["observations"][0]["input_ids"] = list(range(40))
        overhead = token_overhead(result)
        self.assertEqual(overhead["cheapest_candidate"], "a")
        self.assertEqual(overhead["winner_tax_vs_cheapest_pct"], 0.0)

    def test_cheapest_tie_breaks_by_name(self) -> None:
        """Test cheapest tie breaks by name."""
        result = _result(
            "ambiguous",
            [_entry("zed", [("f1", True)]), _entry("abc", [("f1", True)])],
        )
        overhead = token_overhead(result)
        self.assertEqual(overhead["cheapest_candidate"], "abc")
        self.assertIsNone(overhead["winner_tax_vs_cheapest_pct"])

    def test_no_winner_reports_null_tax(self) -> None:
        """Test no winner reports null tax."""
        result = _result("no_candidate", [_entry("a", [("f1", False)])])
        overhead = token_overhead(result)
        self.assertEqual(overhead["cheapest_candidate"], "a")
        self.assertIsNone(overhead["winner_tax_vs_cheapest_pct"])

    def test_incomplete_observations_excluded(self) -> None:
        """Test incomplete observations excluded."""
        entry = _entry("a", [("f1", True, "completed"), ("f2", False, "failed")])
        entry["observations"][1]["input_ids"] = list(range(1000))
        overhead = token_overhead(_result("failed", [entry]))
        stats = overhead["per_candidate"]["a"]
        self.assertEqual(stats["mean_prompt_tokens"], 10.0)
        self.assertEqual(stats["max_prompt_tokens"], 10)

    def test_candidate_without_completed_observations_gets_null_stats(self) -> None:
        """Test candidate without completed observations gets null stats."""
        entry = _entry("a", [("f1", False, "failed")])
        overhead = token_overhead(_result("failed", [entry]))
        self.assertEqual(
            overhead["per_candidate"]["a"],
            {"mean_prompt_tokens": None, "max_prompt_tokens": None, "mean_output_tokens": None},
        )
        self.assertIsNone(overhead["cheapest_candidate"])

    def test_prompt_counts_match_synthetic_tokenizer_encoding(self) -> None:
        """len(input_ids) == len(tokenizer.encode(rendered_prompt)).

        The FakeBackend encodes with the codepoint codec ``ids`` (the
        synthetic stand-in for the checkpoint tokenizer); the overhead
        card must report exactly those encoded lengths.
        """
        second = replace(BASE, name="second")
        backend = FakeBackend(generation=OK)
        result = run_experiment(
            backend, (BASE, second), split(), boundary=ids("|"), max_new_tokens=3
        )
        overhead = token_overhead(result)
        cases = split().search
        for candidate, name in ((BASE, "base"), (second, "second")):
            expected = sum(len(ids(candidate.render(case.messages))) for case in cases) / len(cases)
            self.assertEqual(overhead["per_candidate"][name]["mean_prompt_tokens"], expected)
            expected_max = max(len(ids(candidate.render(case.messages))) for case in cases)
            self.assertEqual(overhead["per_candidate"][name]["max_prompt_tokens"], expected_max)

    def test_overhead_rejects_non_dict(self) -> None:
        """Test overhead rejects non dict."""
        with self.assertRaises(ValueError):
            token_overhead("nope")  # type: ignore[arg-type]


class DiagnosticsWiringTests(unittest.TestCase):
    """Diagnostics wiring tests."""

    def test_attach_diagnostics_adds_both_keys_without_mutating(self) -> None:
        """Test attach diagnostics adds both keys without mutating."""
        result = _result(
            "compatible_on_fixture_set",
            [
                _entry("winner", [("f1", True), ("f2", True)]),
                _entry("runner", [("f1", False), ("f2", True)]),
            ],
            selected="winner",
        )
        before = deepcopy(result)
        enriched = attach_diagnostics(result)
        self.assertEqual(result, before)
        self.assertEqual(enriched["selection_margin"], selection_margin(result))
        self.assertEqual(enriched["token_overhead"], token_overhead(result))
        for key in before:
            self.assertEqual(enriched[key], before[key])

    def test_decision_fields_unchanged_by_diagnostics(self) -> None:
        """The decision function output is identical with/without the features."""
        second = replace(BASE, name="second")
        backend = FakeBackend(generation=OK)
        result = run_experiment(
            backend, (BASE, second), split(), boundary=ids("|"), max_new_tokens=3
        )
        decision_before = (result["decision"], result["selected_candidate"])
        summary_before = deepcopy([entry["summary"] for entry in result["search"]])
        enriched = attach_diagnostics(result)
        html = render_html(enriched)
        with TemporaryDirectory() as tmp:
            json_path, _html_path = save_report(enriched, Path(tmp) / "report")
            saved = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual((result["decision"], result["selected_candidate"]), decision_before)
        self.assertEqual([entry["summary"] for entry in result["search"]], summary_before)
        self.assertEqual((enriched["decision"], enriched["selected_candidate"]), decision_before)
        self.assertEqual((saved["decision"], saved["selected_candidate"]), decision_before)
        self.assertIn("selection_margin", saved)
        self.assertIn("token_overhead", saved)
        self.assertIn("Selection margin:", html)
        self.assertIn("Token overhead", html)

    def test_failed_run_banner_has_no_margin_line(self) -> None:
        """Test failed run banner has no margin line."""
        html = render_html({"decision": "failed", "search": []})
        self.assertNotIn("Selection margin:", html)
        self.assertIn("Token overhead", html)
        self.assertIn("no search candidates were recorded", html)

    def test_banner_line_content_and_escaping(self) -> None:
        """Test banner line content and escaping."""
        from model_protocol_lab.reporting import _SCRIPT  # noqa: E402

        evil = "<script>alert(1)</script>"
        result = _result(
            "compatible_on_fixture_set",
            [_entry("winner", [("f1", True)]), _entry(evil, [("f1", False)])],
            selected="winner",
        )
        html = render_html(result)
        self.assertIn("Selection margin: 1 fixture (runner-up: ", html)
        self.assertIn("&lt;script&gt;", html)
        # The hostile name must not introduce any script element beyond the
        # single allowlisted inline table sorter.
        scrubbed = html.replace("<script>" + _SCRIPT + "</script>", "")
        self.assertNotIn("<script>", scrubbed)

    def test_overhead_card_disclaimer_and_tax_line(self) -> None:
        """Test overhead card disclaimer and tax line."""
        result = _result(
            "compatible_on_fixture_set",
            [_entry("a", [("f1", True)]), _entry("b", [("f1", True)])],
            selected="b",
        )
        result["search"][1]["observations"][0]["input_ids"] = list(range(20))
        html = render_html(result)
        # README disclaimer mirrored verbatim.
        self.assertIn(
            "Token counts in reports are retained-input/output counts, not equal-compute "
            "accounting.",
            html,
        )
        self.assertIn("Winner token tax vs cheapest", html)
        self.assertIn("100.0%", html)

    def test_save_report_is_byte_deterministic(self) -> None:
        """Test save report is byte deterministic."""
        second = replace(BASE, name="second")
        backend = FakeBackend(generation=OK)
        result = run_experiment(
            backend, (BASE, second), split(), boundary=ids("|"), max_new_tokens=3
        )
        with TemporaryDirectory() as tmp:
            first = Path(tmp) / "first"
            second_dir = Path(tmp) / "second"
            save_report(result, first)
            save_report(result, second_dir)
            for name in ("results.json", "report.html"):
                self.assertEqual(
                    (first / name).read_bytes(), (second_dir / name).read_bytes(), name
                )

    def test_overhead_html_has_no_remote_references(self) -> None:
        """Test overhead html has no remote references."""
        result = _result(
            "compatible_on_fixture_set",
            [_entry("a", [("f1", True)]), _entry("b", [("f1", False)])],
            selected="a",
        )
        html = render_html(result)
        self.assertNotIn("http://", html)
        self.assertNotIn("https://", html)


if __name__ == "__main__":
    unittest.main()
