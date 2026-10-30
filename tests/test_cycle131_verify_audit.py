"""Cycle 131 audit: independent results-integrity replay (`mpl verify`).

``mpl verify`` re-derives an experiment's recorded decision, winner, and
post-hoc diagnostics from the stored ``results.json`` artifact alone,
through code paths independent of the ones that produced them (N-version
cross-check): the selection rule is re-implemented as
``verify.oracle_select_winner`` instead of re-calling
``experiments._select_winner``, and the budget frontier / held-out floor
are recomputed from the recorded observations and compared for exact
equality.

Tests pin: the oracle's agreement with ``_select_winner`` (Hypothesis
cross-test), exact replay on genuine artifacts (diagnostics computed by
the real pipeline functions), tamper detection on the decision, the
winner, the stored frontier and the stored floor, candidate-name
uniqueness, search/held-out split separation, the cancelled-run and
stripped-observations "unavailable" honesty paths, schema fail-closed,
and the CLI exit-code contract. A mutation gate proves the tests are not
vacuous.
"""

import json
import subprocess
import sys
import time
import unittest
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from model_protocol_lab import verify
from model_protocol_lab.diffreports import (
    _step_function,
    _valid_frontier,
    diff_frontiers,
)
from model_protocol_lab.experiments import (
    _select_winner,
    budget_frontier,
    held_out_budget_floor,
)
from model_protocol_lab.reporting import MAX_FRONTIER_STEP_POINTS, margin_trajectory

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SOURCE = Path(verify.__file__)

BOUNDARY = [7]
MAX_NEW_TOKENS = 32


def _obs(fixture: str, passed: bool) -> dict:
    """Obs."""
    if passed:
        # Boundary [7] first occurs at offset 2 -> minimal budget 3.
        return {
            "fixture": fixture,
            "status": "completed",
            "exact_match": True,
            "boundary_found": True,
            "output_ids": [1, 2, 7, 3],
        }
    return {
        "fixture": fixture,
        "status": "completed",
        "exact_match": False,
        "boundary_found": False,
        "output_ids": [1, 2, 3],
    }


def _entry(name: str, n_pass: int, n_total: int = 2) -> dict:
    """Entry."""
    observations = [_obs(f"s{i}", i < n_pass) for i in range(n_total)]
    return {
        "format": {"name": name},
        "candidate_fingerprint": "fp-" + name,
        "boundary_token_ids": list(BOUNDARY),
        "summary": {
            "status": "completed",
            "planned": n_total,
            "attempted": n_total,
            "aggregate_eligible": True,
            "exact_and_boundary_passes": n_pass,
            "rate": n_pass / n_total,
            "encoded_input_tokens": 10,
            "retained_output_tokens": 20,
            "failure_code": "none" if n_pass == n_total else "answer_mismatch",
        },
        "observations": observations,
    }


def _held_out(n_pass: int, n_total: int = 2) -> dict:
    """Held out."""
    observations = [_obs(f"h{i}", i < n_pass) for i in range(n_total)]
    return {
        "summary": {
            "status": "completed",
            "planned": n_total,
            "attempted": n_total,
            "aggregate_eligible": True,
            "exact_and_boundary_passes": n_pass,
            "rate": n_pass / n_total,
            "encoded_input_tokens": 5,
            "retained_output_tokens": 8,
            "failure_code": "none" if n_pass == n_total else "answer_mismatch",
        },
        "observations": observations,
    }


def _artifact(
    decision: str,
    entries: list,
    selected: str | None,
    held_out: dict | None,
) -> dict:
    """Build an artifact exactly as `mpl run` would write it.

    The diagnostic sections are computed by the real pipeline functions
    with the same arguments the CLI uses, so the replay checks exercise
    genuine pipeline output rather than hand-shaped payloads.
    """
    data = {
        "seed": 0,
        "max_new_tokens": MAX_NEW_TOKENS,
        "boundary_token_ids": list(BOUNDARY),
        "eos_token_id": 9,
        "backend_fingerprint": "fake-backend",
        "statistical_confidence": "not_estimated",
        "compute_comparison": "not_equal_compute_certified",
        "decision": decision,
        "search": entries,
        "held_out": held_out,
        "selected_candidate": selected,
        "failure_taxonomy": {"decision": decision, "modes": []},
        "raw_evidence_included": True,
    }
    if selected is not None:
        entry = next(e for e in entries if e["format"]["name"] == selected)
        data["selected_boundary_token_ids"] = list(entry["boundary_token_ids"])
    if decision in (
        "compatible_on_fixture_set",
        "held_out_failed",
        "ambiguous",
        "no_candidate",
    ):
        data["budget_frontier"] = budget_frontier(
            entries,
            boundary=tuple(BOUNDARY),
            max_new_tokens=MAX_NEW_TOKENS,
            native_decision=decision,
            native_winner=selected,
        )
    else:
        data["budget_frontier"] = {"available": False, "detail": "search_incomplete"}
    if (
        decision in ("compatible_on_fixture_set", "held_out_failed")
        and held_out is not None
        and selected is not None
    ):
        data["held_out_budget_floor"] = held_out_budget_floor(
            held_out["observations"],
            boundary=tuple(data["selected_boundary_token_ids"]),
            max_new_tokens=MAX_NEW_TOKENS,
            native_verdict=decision,
        )
    else:
        data["held_out_budget_floor"] = {
            "available": False,
            "detail": "held_out_incomplete",
        }
    return data


def _compatible() -> dict:
    """Compatible."""
    return _artifact(
        "compatible_on_fixture_set",
        [_entry("alpha", 2), _entry("beta", 1)],
        "alpha",
        _held_out(2),
    )


def _check_status(report: dict, check_id: str) -> str:
    """Check status."""
    for check in report["checks"]:
        if check["id"] == check_id:
            return check["status"]
    raise AssertionError("check missing: " + check_id)


class TestOracle(unittest.TestCase):
    """Test oracle."""

    def test_single_winner_selects(self) -> None:
        """Test single winner selects."""
        self.assertEqual(verify.oracle_select_winner(["alpha"]), ("selected", "alpha"))

    def test_no_winners_abstains(self) -> None:
        """Test no winners abstains."""
        self.assertEqual(verify.oracle_select_winner([]), ("no_candidate", None))

    def test_tie_abstains(self) -> None:
        """Test tie abstains."""
        self.assertEqual(verify.oracle_select_winner(["alpha", "beta"]), ("ambiguous", None))

    @given(st.lists(st.text(min_size=1, max_size=12), unique=True, max_size=5))
    @settings(max_examples=100)
    def test_oracle_agrees_with_select_winner(self, names: list) -> None:
        """Test oracle agrees with select winner."""
        # N-version cross-check: the independent oracle must agree with the
        # pipeline's selection function on every generated winner list.
        self.assertEqual(verify.oracle_select_winner(names), _select_winner(names, False))

    def test_cancelled_is_outside_the_oracle(self) -> None:
        """Test cancelled is outside the oracle."""
        # The cancel signal is not recorded in results.json; the pipeline
        # maps it to ("cancelled", None) while the oracle has no cancel
        # input. verify reports such runs "unavailable", never re-derived.
        self.assertEqual(_select_winner(["alpha"], True), ("cancelled", None))


class TestValidArtifacts(unittest.TestCase):
    """Test valid artifacts."""

    def test_compatible_artifact_verifies_ok(self) -> None:
        """Test compatible artifact verifies ok."""
        report = verify.verify_results(_compatible())
        self.assertEqual(report["verdict"], "ok")
        for check in report["checks"]:
            self.assertEqual(check["status"], "passed", "check not passed: " + check["id"])

    def test_ambiguous_artifact_verifies_ok(self) -> None:
        """Test ambiguous artifact verifies ok."""
        data = _artifact("ambiguous", [_entry("alpha", 2), _entry("beta", 2)], None, None)
        report = verify.verify_results(data)
        self.assertEqual(report["verdict"], "ok")
        self.assertEqual(_check_status(report, "selection_decision"), "passed")

    def test_no_candidate_artifact_verifies_ok(self) -> None:
        """Test no candidate artifact verifies ok."""
        data = _artifact("no_candidate", [_entry("alpha", 1), _entry("beta", 0)], None, None)
        report = verify.verify_results(data)
        self.assertEqual(report["verdict"], "ok")

    def test_held_out_failed_artifact_verifies_ok(self) -> None:
        """Test held out failed artifact verifies ok."""
        data = _artifact(
            "held_out_failed",
            [_entry("alpha", 2), _entry("beta", 0)],
            "alpha",
            _held_out(1),
        )
        report = verify.verify_results(data)
        self.assertEqual(report["verdict"], "ok")

    def test_search_abort_failed_is_rederivable(self) -> None:
        """Test search abort failed is rederivable."""
        entry = _entry("alpha", 0)
        entry["summary"]["aggregate_eligible"] = False
        entry["summary"]["status"] = "failed"
        data = _artifact("failed", [entry], None, None)
        report = verify.verify_results(data)
        self.assertEqual(report["verdict"], "ok")
        self.assertEqual(_check_status(report, "selection_decision"), "passed")

    def test_cancelled_after_search_is_unavailable_not_failed(self) -> None:
        """Test cancelled after search is unavailable not failed."""
        data = _artifact("cancelled", [_entry("alpha", 2), _entry("beta", 0)], None, None)
        report = verify.verify_results(data)
        self.assertEqual(report["verdict"], "ok")
        self.assertEqual(_check_status(report, "selection_decision"), "unavailable")

    def test_stripped_observations_mark_dependent_checks_unavailable(self) -> None:
        """Test stripped observations mark dependent checks unavailable."""
        data = _compatible()
        for entry in data["search"]:
            del entry["observations"]
        del data["held_out"]["observations"]
        report = verify.verify_results(data)
        self.assertEqual(report["verdict"], "ok")
        self.assertEqual(_check_status(report, "budget_frontier_replay"), "unavailable")
        self.assertEqual(_check_status(report, "held_out_floor_replay"), "unavailable")
        self.assertEqual(_check_status(report, "split_separation"), "unavailable")
        # The decision replay needs only summaries, so it still passes.
        self.assertEqual(_check_status(report, "selection_decision"), "passed")


class TestTamperDetection(unittest.TestCase):
    """Test tamper detection."""

    def test_flipped_decision_detected(self) -> None:
        """Test flipped decision detected."""
        data = _artifact("ambiguous", [_entry("alpha", 2), _entry("beta", 2)], None, None)
        data["decision"] = "compatible_on_fixture_set"
        data["selected_candidate"] = "alpha"
        report = verify.verify_results(data)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(_check_status(report, "selection_decision"), "failed")

    def test_swapped_winner_detected(self) -> None:
        """Test swapped winner detected."""
        data = _compatible()
        data["selected_candidate"] = "beta"
        report = verify.verify_results(data)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(_check_status(report, "selection_decision"), "failed")

    def test_mutated_frontier_detected(self) -> None:
        """Test mutated frontier detected."""
        data = _compatible()
        frontier = data["budget_frontier"]
        first_candidate = next(iter(frontier["candidates"]))
        frontier["candidates"][first_candidate]["pass_steps"][0][1] += 1
        report = verify.verify_results(data)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(_check_status(report, "budget_frontier_replay"), "failed")

    def test_mutated_held_out_floor_detected(self) -> None:
        """Test mutated held out floor detected."""
        data = _compatible()
        data["held_out_budget_floor"]["safe_budget_floor"] = 1
        report = verify.verify_results(data)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(_check_status(report, "held_out_floor_replay"), "failed")

    def test_duplicate_candidate_names_detected(self) -> None:
        """Test duplicate candidate names detected."""
        data = _compatible()
        data["search"].append(_entry("alpha", 2))
        report = verify.verify_results(data)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(_check_status(report, "candidates_unique"), "failed")

    def test_summary_inconsistency_detected(self) -> None:
        """Test summary inconsistency detected."""
        data = _compatible()
        data["search"][0]["summary"]["exact_and_boundary_passes"] = 99
        report = verify.verify_results(data)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(_check_status(report, "summaries_consistent"), "failed")

    def test_search_held_out_overlap_detected(self) -> None:
        """Test search held out overlap detected."""
        data = _compatible()
        # Smuggle a search fixture id into the held-out set.
        data["held_out"]["observations"][0]["fixture"] = "s0"
        report = verify.verify_results(data)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(_check_status(report, "split_separation"), "failed")

    def test_candidate_fixture_mismatch_detected(self) -> None:
        """Test candidate fixture mismatch detected."""
        data = _compatible()
        data["search"][1]["observations"][0]["fixture"] = "other-fixture"
        report = verify.verify_results(data)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(_check_status(report, "split_separation"), "failed")

    def test_wrong_selected_boundary_detected(self) -> None:
        """Test wrong selected boundary detected."""
        data = _compatible()
        data["selected_boundary_token_ids"] = [999]
        report = verify.verify_results(data)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(_check_status(report, "selected_boundary"), "failed")

    def test_overcounted_passes_do_not_create_winners(self) -> None:
        """Test overcounted passes do not create winners."""
        # passes > planned is a summary inconsistency, but the replay must
        # still use exact equality (not >=) when deriving winners: beta
        # must not become a winner here.
        data = _artifact("no_candidate", [_entry("alpha", 1), _entry("beta", 0)], None, None)
        data["search"][1]["summary"]["exact_and_boundary_passes"] = 3
        report = verify.verify_results(data)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(_check_status(report, "summaries_consistent"), "failed")
        self.assertEqual(_check_status(report, "selection_decision"), "passed")

    def test_unknown_decision_vocabulary_rejected(self) -> None:
        """Test unknown decision vocabulary rejected."""
        data = _compatible()
        data["decision"] = "super-compatible"
        report = verify.verify_results(data)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(_check_status(report, "schema"), "failed")

    def test_missing_search_rejected(self) -> None:
        """Test missing search rejected."""
        data = _compatible()
        del data["search"]
        report = verify.verify_results(data)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(_check_status(report, "schema"), "failed")
        self.assertEqual(_check_status(report, "selection_decision"), "unavailable")

    def test_non_object_rejected(self) -> None:
        """Test non object rejected."""
        report = verify.verify_results(["not", "an", "object"])
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(_check_status(report, "schema"), "failed")

    def test_hostile_candidate_name_sanitized_in_detail(self) -> None:
        """Test hostile candidate name sanitized in detail."""
        data = _compatible()
        data["search"].append(_entry("alpha\x1b[31m-evil", 2))
        data["search"].append(_entry("alpha\x1b[31m-evil", 1))
        report = verify.verify_results(data)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(_check_status(report, "candidates_unique"), "failed")
        detail = next(c["detail"] for c in report["checks"] if c["id"] == "candidates_unique")
        self.assertNotIn("\x1b", detail)
        self.assertIn("alpha", detail)


class TestCli(unittest.TestCase):
    """Test cli."""

    @staticmethod
    def _write(directory: Path, data: dict) -> Path:
        """Write."""
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "results.json"
        path.write_text(json.dumps(data, ensure_ascii=True), encoding="utf-8")
        return directory

    @staticmethod
    def _run(*args: str) -> subprocess.CompletedProcess:
        """Run."""
        return subprocess.run(
            [sys.executable, "-m", "model_protocol_lab", *args],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

    def test_verify_ok_exit_0(self) -> None:
        """Test verify ok exit 0."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            target = self._write(Path(tmp) / "run", _compatible())
            proc = self._run("verify", str(target))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["verdict"], "ok")
        self.assertIn("verdict ok", proc.stderr)

    def test_verify_accepts_results_json_directly(self) -> None:
        """Test verify accepts results json directly."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            target = self._write(Path(tmp) / "run", _compatible())
            proc = self._run("verify", str(target / "results.json"))
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_verify_tampered_exit_2(self) -> None:
        """Test verify tampered exit 2."""
        import tempfile

        data = _compatible()
        data["decision"] = "no_candidate"
        with tempfile.TemporaryDirectory() as tmp:
            target = self._write(Path(tmp) / "run", data)
            proc = self._run("verify", str(target))
        self.assertEqual(proc.returncode, 2, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["verdict"], "failed")
        self.assertIn("FAILED: selection_decision", proc.stderr)

    def test_verify_missing_path_exit_1(self) -> None:
        """Test verify missing path exit 1."""
        proc = self._run("verify", "/nonexistent/path/results.json")
        self.assertEqual(proc.returncode, 1)
        self.assertIn("verify:", proc.stderr)

    def test_verify_invalid_json_exit_1(self) -> None:
        """Test verify invalid json exit 1."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.json"
            path.write_text("{not json", encoding="utf-8")
            proc = self._run("verify", str(path))
        self.assertEqual(proc.returncode, 1)
        self.assertIn("verify:", proc.stderr)


class TestHostilePayloadRobustness(unittest.TestCase):
    """Regression tests for the perf findings folded into this cycle.

    `mpl diff-reports` (and, through it, the frontier chart and margin
    trajectory) consumes foreign `budget_frontier` payloads. Two genuine
    issues, both reproduced before fixing: (1A) the trajectory/span
    derivations were superlinear in the step-point count with nothing
    capping it -- a crafted payload hung the comparison for hours;
    (1B) duplicate `decision_steps` budgets with mixed None/str winners
    crashed `diff_frontiers` with an uncaught TypeError.
    """

    @staticmethod
    def _foreign_frontier(n_points: int, max_tokens: int = 10**9) -> dict:
        """Foreign frontier."""
        pts = [[i + 1, (i * 7) % 3] for i in range(n_points)]
        return {
            "max_new_tokens": max_tokens,
            "native_decision": "ambiguous",
            "native_winner": None,
            "candidates": {
                "a": {
                    "fixture_ids": ["f1", "f2", "f3"],
                    "pass_steps": pts,
                    "full_pass_budget": None,
                }
            },
            "decision_steps": [[1, "ambiguous", None]],
            "flip_ranges": [],
            "budget_robust": True,
            "safe_budget_floor": 1,
            "flip_attribution": [],
        }

    def test_oversized_frontier_degrades_instead_of_hanging(self) -> None:
        """Test oversized frontier degrades instead of hanging."""
        big = self._foreign_frontier(MAX_FRONTIER_STEP_POINTS + 1)
        start = time.monotonic()
        self.assertEqual(margin_trajectory(big), [])
        self.assertFalse(_valid_frontier(big))
        small = self._foreign_frontier(3, 32)
        result = diff_frontiers(big, small)
        elapsed = time.monotonic() - start
        # The oversized side degrades to one-sided instead of hanging.
        self.assertEqual(result["available"], "new")
        self.assertFalse(result["changed"])
        self.assertLess(elapsed, 10.0)

    def test_large_capped_frontier_still_diffs_exactly(self) -> None:
        """Test large capped frontier still diffs exactly."""
        # 50k points is under the cap: the comparison must complete
        # (the old quadratic code needed minutes here) and stay exact.
        old = self._foreign_frontier(50_000, 10**9)
        new = self._foreign_frontier(50_000, 10**9)
        new["candidates"]["a"]["pass_steps"][-1][1] = 2
        start = time.monotonic()
        result = diff_frontiers(old, new)
        elapsed = time.monotonic() - start
        self.assertEqual(result["available"], "both")
        self.assertLess(elapsed, 60.0)

    def test_duplicate_decision_budgets_do_not_crash(self) -> None:
        """Test duplicate decision budgets do not crash."""
        old = self._foreign_frontier(3, 32)
        new = self._foreign_frontier(3, 32)
        old["decision_steps"] = [[1, "selected", "a"], [1, "selected", None]]
        new["decision_steps"] = [[1, "selected", "a"]]
        # Used to raise TypeError: '<' not supported between NoneType and
        # str, escaping run_diff_reports' ValueError-only catch.
        result = diff_frontiers(old, new)
        self.assertEqual(result["decision_flips"], [])
        self.assertEqual(result["winner_flips"], [])

    def test_first_duplicate_budget_wins(self) -> None:
        """Test first duplicate budget wins."""
        old = self._foreign_frontier(3, 32)
        new = self._foreign_frontier(3, 32)
        old["decision_steps"] = [[1, "selected", "a"], [1, "ambiguous", None]]
        new["decision_steps"] = [[1, "selected", "a"]]
        result = diff_frontiers(old, new)
        self.assertEqual(result["decision_flips"], [])
        self.assertEqual(result["winner_flips"], [])

    def test_step_function_value_at(self) -> None:
        """Test step function value at."""
        fn = _step_function([(1, "a"), (5, "b"), (10, "c")])
        self.assertEqual(fn(0), ("unknown", None))
        self.assertEqual(fn(1), "a")
        self.assertEqual(fn(4), "a")
        self.assertEqual(fn(5), "b")
        self.assertEqual(fn(10**12), "c")

    def test_step_function_tolerates_mixed_value_types(self) -> None:
        """Test step function tolerates mixed value types."""
        # Duplicate budgets can no longer reach the sort, but the sort
        # itself must never compare values. Duplicate-budget lookup keeps
        # the long-standing "last wins" semantic (bisect_right), unchanged
        # by this cycle.
        fn = _step_function([(1, None), (1, "a"), (2, "b")])
        self.assertEqual(fn(1), "a")
        self.assertEqual(fn(2), "b")


class TestDefensiveBranches(unittest.TestCase):
    """Cover verify's fail-closed branches: every malformed shape must
    produce a failed/unavailable check, never an exception.
    """

    def test_schema_rejects_each_bad_field(self) -> None:
        """Test schema rejects each bad field."""
        for field, bad in (
            ("selected_candidate", 123),
            ("max_new_tokens", "x"),
            ("boundary_token_ids", "x"),
            ("failure_taxonomy", None),
        ):
            with self.subTest(field=field):
                data = _compatible()
                data[field] = bad
                report = verify.verify_results(data)
                self.assertEqual(report["verdict"], "failed")
                self.assertEqual(_check_status(report, "schema"), "failed")

    def test_non_dict_search_entry_never_crashes(self) -> None:
        """Test non dict search entry never crashes."""
        data = _compatible()
        data["search"].append("not-a-dict")
        report = verify.verify_results(data)  # must not raise
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(_check_status(report, "candidates_unique"), "failed")
        self.assertEqual(_check_status(report, "selection_decision"), "failed")

    def test_selected_not_among_candidates(self) -> None:
        """Test selected not among candidates."""
        data = _compatible()
        data["selected_candidate"] = "ghost"
        report = verify.verify_results(data)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(_check_status(report, "candidates_unique"), "failed")
        self.assertEqual(_check_status(report, "selected_boundary"), "failed")

    def test_entry_without_summary(self) -> None:
        """Test entry without summary."""
        data = _compatible()
        del data["search"][0]["summary"]
        report = verify.verify_results(data)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(_check_status(report, "summaries_consistent"), "failed")

    def test_summary_planned_wrong_type(self) -> None:
        """Test summary planned wrong type."""
        data = _compatible()
        data["search"][0]["summary"]["planned"] = "two"
        report = verify.verify_results(data)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(_check_status(report, "summaries_consistent"), "failed")

    def test_held_out_missing_for_selected(self) -> None:
        """Test held out missing for selected."""
        data = _compatible()
        data["held_out"] = None
        report = verify.verify_results(data)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(_check_status(report, "selection_decision"), "failed")

    def test_held_out_summary_missing(self) -> None:
        """Test held out summary missing."""
        data = _compatible()
        data["held_out"] = {"observations": []}
        report = verify.verify_results(data)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(_check_status(report, "selection_decision"), "failed")

    def test_held_out_abort_keeps_winner(self) -> None:
        """Test held out abort keeps winner."""
        # A real held-abort run records the winner but the decision is the
        # held-out status; the CLI degrades both post-hoc diagnostics to
        # unavailable markers (their recompute functions reject the
        # non-completed verdicts).
        data = _compatible()
        data["held_out"]["summary"]["aggregate_eligible"] = False
        data["held_out"]["summary"]["status"] = "failed"
        data["decision"] = "failed"
        data["failure_taxonomy"] = {"decision": "failed", "modes": []}
        data["budget_frontier"] = {"available": False, "detail": "frontier_unavailable"}
        data["held_out_budget_floor"] = {
            "available": False,
            "detail": "held_out_floor_unavailable",
        }
        report = verify.verify_results(data)
        self.assertEqual(report["verdict"], "ok")
        self.assertEqual(_check_status(report, "selection_decision"), "passed")

    def test_frontier_unavailable_without_reason_fails(self) -> None:
        """Test frontier unavailable without reason fails."""
        data = _compatible()
        data["budget_frontier"] = {"available": False}
        report = verify.verify_results(data)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(_check_status(report, "budget_frontier_replay"), "failed")

    def test_frontier_malformed_fails(self) -> None:
        """Test frontier malformed fails."""
        data = _compatible()
        data["budget_frontier"] = [1, 2]
        report = verify.verify_results(data)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(_check_status(report, "budget_frontier_replay"), "failed")

    def test_frontier_recompute_error_fails_closed(self) -> None:
        """Test frontier recompute error fails closed."""
        data = _compatible()
        # Fixture mismatch across entries: budget_frontier raises ValueError.
        data["search"][1]["observations"][0]["fixture"] = "other"
        report = verify.verify_results(data)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(_check_status(report, "budget_frontier_replay"), "failed")

    def test_floor_unavailable_without_reason_fails(self) -> None:
        """Test floor unavailable without reason fails."""
        data = _compatible()
        data["held_out_budget_floor"] = {"available": False}
        report = verify.verify_results(data)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(_check_status(report, "held_out_floor_replay"), "failed")

    def test_floor_malformed_fails(self) -> None:
        """Test floor malformed fails."""
        data = _compatible()
        data["held_out_budget_floor"] = "x"
        report = verify.verify_results(data)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(_check_status(report, "held_out_floor_replay"), "failed")

    def test_floor_bad_held_out_boundary_fails(self) -> None:
        """Test floor bad held out boundary fails."""
        data = _compatible()
        data["selected_boundary_token_ids"] = "x"
        report = verify.verify_results(data)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(_check_status(report, "held_out_floor_replay"), "failed")

    def test_non_dict_observation_never_crashes(self) -> None:
        """Test non dict observation never crashes."""
        data = _compatible()
        data["search"][0]["observations"][0] = "not-a-dict"
        report = verify.verify_results(data)  # must not raise
        self.assertEqual(report["verdict"], "failed")

    @staticmethod
    def _run_in_process(target) -> tuple:
        """Run in process."""
        import argparse
        import io
        from contextlib import redirect_stderr, redirect_stdout

        args = argparse.Namespace(
            results=target,
            bundle=None,
            expect_digest=None,
            determinism=False,
            model=None,
            device="cpu",
            max_context=2048,
        )
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = verify.run_verify(args, None)
        text = out.getvalue()
        return code, (json.loads(text) if text.strip() else None), err.getvalue()

    def test_run_verify_in_process_ok(self) -> None:
        """Test run verify in process ok."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.json"
            path.write_text(json.dumps(_compatible()), encoding="utf-8")
            code, payload, err = self._run_in_process(path)
        self.assertEqual(code, 0)
        self.assertEqual(payload["verdict"], "ok")
        self.assertIn("verdict ok", err)

    def test_run_verify_in_process_failed(self) -> None:
        """Test run verify in process failed."""
        import tempfile

        data = _compatible()
        data["decision"] = "no_candidate"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            code, payload, err = self._run_in_process(path)
        self.assertEqual(code, 2)
        self.assertEqual(payload["verdict"], "failed")
        self.assertIn("FAILED: selection_decision", err)

    def test_run_verify_in_process_unreadable(self) -> None:
        """Test run verify in process unreadable."""
        code, _payload, err = self._run_in_process("/nonexistent/x.json")
        self.assertEqual(code, 1)
        self.assertIn("verify:", err)

    def test_run_verify_summary_lists_unavailable(self) -> None:
        """Test run verify summary lists unavailable."""
        import tempfile

        data = _compatible()
        for entry in data["search"]:
            del entry["observations"]
        del data["held_out"]["observations"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            code, _payload, err = self._run_in_process(path)
        self.assertEqual(code, 0)
        self.assertIn("unavailable: budget_frontier_replay", err)

    def test_build_verify_parser_registers(self) -> None:
        """Test build verify parser registers."""
        import argparse

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        verify.build_verify_parser(sub)
        args = parser.parse_args(["verify", "some/path"])
        self.assertEqual(args.command, "verify")
        self.assertEqual(str(args.results), "some/path")

    def test_entry_without_own_boundary_falls_back(self) -> None:
        """Test entry without own boundary falls back."""
        data = _compatible()
        del data["search"][0]["boundary_token_ids"]
        report = verify.verify_results(data)
        self.assertEqual(_check_status(report, "selected_boundary"), "passed")

    def test_boundary_key_set_without_selected_candidate(self) -> None:
        """Test boundary key set without selected candidate."""
        data = _artifact("ambiguous", [_entry("alpha", 2), _entry("beta", 2)], None, None)
        data["selected_boundary_token_ids"] = [7]
        report = verify.verify_results(data)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(_check_status(report, "selected_boundary"), "failed")

    def test_missing_frontier_section_is_unavailable(self) -> None:
        """Test missing frontier section is unavailable."""
        data = _compatible()
        del data["budget_frontier"]
        report = verify.verify_results(data)
        self.assertEqual(report["verdict"], "ok")
        self.assertEqual(_check_status(report, "budget_frontier_replay"), "unavailable")

    def test_missing_floor_section_is_unavailable(self) -> None:
        """Test missing floor section is unavailable."""
        data = _compatible()
        del data["held_out_budget_floor"]
        report = verify.verify_results(data)
        self.assertEqual(report["verdict"], "ok")
        self.assertEqual(_check_status(report, "held_out_floor_replay"), "unavailable")


class TestMutationGate(unittest.TestCase):
    """Test mutation gate."""

    SOURCE = _SOURCE
    MUTANTS = [
        # Oracle weakened: a tie would then "select" the first winner.
        (
            "    if len(winner_names) == 1:",
            "    if len(winner_names) >= 1:",
        ),
        # Exact replay comparison inverted: genuine artifacts would fail.
        (
            "    return json.dumps(first, sort_keys=True, ensure_ascii=True) == json.dumps(\n"
            "        second, sort_keys=True, ensure_ascii=True\n"
            "    )",
            "    return json.dumps(first, sort_keys=True, ensure_ascii=True) != json.dumps(\n"
            "        second, sort_keys=True, ensure_ascii=True\n"
            "    )",
        ),
        # Split-overlap detection inverted: contaminated artifacts pass.
        (
            "        if overlap:",
            "        if not overlap:",
        ),
        # Winner derivation weakened: partial passers would count as winners.
        (
            '        and entry["summary"].get("exact_and_boundary_passes")'
            ' == entry["summary"].get("planned")',
            '        and entry["summary"].get("exact_and_boundary_passes")'
            ' >= entry["summary"].get("planned")',
        ),
    ]

    def test_mutants_are_killed(self) -> None:
        """Test mutants are killed."""
        source = self.SOURCE.read_text(encoding="utf-8")
        killed = []
        survived = []
        for original, mutant in self.MUTANTS:
            self.assertIn(original, source, f"mutant anchor missing: {original!r}")
            self.SOURCE.write_text(source.replace(original, mutant, 1), encoding="utf-8")
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
                    timeout=600,
                    check=False,
                )
            finally:
                self.SOURCE.write_text(source, encoding="utf-8")
                # Invalidate the bytecode cache: a same-size mutant can
                # leave a .pyc whose (mtime, size) still validates
                # against the reverted source, poisoning later imports
                # with mutant bytecode (see the cycle-135 gate fix).
                for _pyc in self.SOURCE.parent.glob("__pycache__/verify.*.pyc"):
                    _pyc.unlink(missing_ok=True)
            if proc.returncode != 0:
                killed.append(mutant)
            else:
                survived.append(mutant)
        self.assertEqual(survived, [], f"surviving mutants: {survived}")
        self.assertEqual(len(killed), len(self.MUTANTS))


if __name__ == "__main__":
    unittest.main()
