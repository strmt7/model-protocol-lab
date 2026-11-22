"""W7-6: the R4 ``diff-reports --gate`` verdict and JUnit XML reports.

The contract is the R4 adjudication (``docs`` research lane, R4):
``--gate`` is opt-in; without it the diff-reports output is
byte-identical to the pre-gate contract (pinned here against goldens
recorded from the unmodified code). Gate mode adds a ``gate`` object
to the same JSON and makes the exit code the branch signal: 0 for
``no_change``/``improvement``/``change``, 2 for
``regression``/``unavailable``, 1 for operational failure. The
threshold resolves flag > ``MODEL_PROTOCOL_LAB_DIFF_MAX_REGRESSIONS``
> default 0, and a malformed env value fails closed (exit 1, never a
silent fallback). ``--junit`` on diff-reports and verify writes JUnit
XML derived from the same report dict; an unwritable path exits 1
after the normal outputs print and leaves no partial file.

Classifier expectations are derived from the adjudication's event
definitions (fixture pass->fail flips, verdict changes toward a
lower quality rank, decision changes away from
``compatible_on_fixture_set``, winner loss; improvements are the
exact mirrors), never from the implementation's own output.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock

from model_protocol_lab import cli
from model_protocol_lab import pack as pk
from model_protocol_lab import verify as vf
from model_protocol_lab.diffreports_core import (
    _checks_junit_suites,
    _classify_gate,
    _diff_junit_suites,
    run_diff_reports,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE = REPO_ROOT / "examples" / "sample-report" / "results.json"
GOLDEN_DIR = Path(__file__).resolve().parent / "data"
ENV_VAR = "MODEL_PROTOCOL_LAB_DIFF_MAX_REGRESSIONS"


def _sample() -> dict:
    return json.loads(SAMPLE.read_text(encoding="utf-8"))


def _perturbed() -> dict:
    """The golden pair-B input: winner flips and one verdict degrades."""
    payload = _sample()
    payload["selected_candidate"] = "chatml"
    for entry in payload["search"]:
        summary = entry.get("summary") or {}
        if summary.get("planned") and summary.get("planned") == summary.get(
            "exact_and_boundary_passes"
        ):
            summary["exact_and_boundary_passes"] = summary["planned"] - 1
            break
    return payload


def _run_diff(argv: list[str], cwd: Path) -> tuple[int, str, str]:
    parser = cli.build_parser()
    args = parser.parse_args(["diff-reports", *argv])
    stdout, stderr = io.StringIO(), io.StringIO()
    with (
        contextlib.chdir(cwd),
        contextlib.redirect_stdout(stdout),
        contextlib.redirect_stderr(stderr),
    ):
        code = run_diff_reports(args, parser)
    return code, stdout.getvalue(), stderr.getvalue()


class _PairDir(unittest.TestCase):
    """A temp dir holding the golden pair as bare old.json/new.json."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        (self.dir / "old.json").write_text(json.dumps(_sample()), encoding="utf-8")
        (self.dir / "new.json").write_text(json.dumps(_perturbed()), encoding="utf-8")

    def tearDown(self) -> None:
        self._tmp.cleanup()


def _diff_dict(**overrides) -> dict:
    diff = {
        "summary": {"identical": False, "evidence_asymmetric": False},
        "fixture_diffs": [],
        "candidate_verdicts": [],
        "decision": {"changed": False, "old": None, "new": None},
        "winner": {"changed": False, "old": None, "new": None},
    }
    diff.update(overrides)
    return diff


class ClassifierTests(unittest.TestCase):
    """Every taxonomy cell, from the adjudication's event definitions."""

    def verdict(self, diff: dict, max_regressions: int = 0) -> dict:
        return _classify_gate(diff, max_regressions, "default")

    def test_identical_is_no_change(self):
        diff = _diff_dict()
        diff["summary"]["identical"] = True
        self.assertEqual(self.verdict(diff)["verdict"], "no_change")

    def test_asymmetric_evidence_is_unavailable(self):
        diff = _diff_dict()
        diff["summary"]["evidence_asymmetric"] = True
        self.assertEqual(self.verdict(diff)["verdict"], "unavailable")

    def test_fixture_flip_down_is_regression(self):
        diff = _diff_dict(
            fixture_diffs=[
                {"old": "pass", "new": "fail"},
            ]
        )
        gate = self.verdict(diff)
        self.assertEqual((gate["verdict"], gate["regressions"]), ("regression", 1))

    def test_fixture_flip_up_is_improvement(self):
        diff = _diff_dict(fixture_diffs=[{"old": "fail", "new": "pass"}])
        gate = self.verdict(diff)
        self.assertEqual((gate["verdict"], gate["improvements"]), ("improvement", 1))

    def test_fixture_absent_transition_is_not_a_flip(self):
        diff = _diff_dict(fixture_diffs=[{"old": "absent", "new": "fail"}])
        gate = self.verdict(diff)
        self.assertEqual((gate["regressions"], gate["improvements"]), (0, 0))
        self.assertEqual(gate["verdict"], "change")

    def test_verdict_rank_edges(self):
        # pass(3) > fail(2) > unknown(1) > absent(0): toward lower is a
        # regression, toward higher an improvement, in every pairing.
        cases = [
            ("pass", "fail", "regression"),
            ("pass", "unknown", "regression"),
            ("pass", "absent", "regression"),
            ("fail", "unknown", "regression"),
            ("fail", "absent", "regression"),
            ("unknown", "absent", "regression"),
            ("absent", "unknown", "improvement"),
            ("absent", "fail", "improvement"),
            ("absent", "pass", "improvement"),
            ("unknown", "fail", "improvement"),
            ("unknown", "pass", "improvement"),
            ("fail", "pass", "improvement"),
        ]
        for old, new, expected in cases:
            with self.subTest(old=old, new=new):
                diff = _diff_dict(
                    candidate_verdicts=[{"old": old, "new": new}],
                )
                self.assertEqual(self.verdict(diff)["verdict"], expected)

    def test_equal_unranked_and_nonstring_verdicts_are_not_events(self):
        # The event definition counts only changes between two ranked
        # verdicts: an unchanged verdict, a value outside the rank
        # vocabulary, and a non-string value are all skipped, never
        # counted in either direction.
        diff = _diff_dict(
            candidate_verdicts=[
                {"old": "pass", "new": "pass"},
                {"old": "pass", "new": "not-a-verdict"},
                {"old": None, "new": "fail"},
            ]
        )
        gate = self.verdict(diff)
        self.assertEqual((gate["regressions"], gate["improvements"]), (0, 0))
        self.assertEqual(gate["verdict"], "change")

    def test_decision_away_from_compatible_is_regression(self):
        diff = _diff_dict(
            decision={
                "changed": True,
                "old": "compatible_on_fixture_set",
                "new": "abstain",
            }
        )
        self.assertEqual(self.verdict(diff)["verdict"], "regression")

    def test_decision_toward_compatible_is_improvement(self):
        diff = _diff_dict(
            decision={
                "changed": True,
                "old": "abstain",
                "new": "compatible_on_fixture_set",
            }
        )
        self.assertEqual(self.verdict(diff)["verdict"], "improvement")

    def test_decision_between_other_values_is_neither(self):
        diff = _diff_dict(decision={"changed": True, "old": "abstain", "new": "incompatible"})
        gate = self.verdict(diff)
        self.assertEqual((gate["regressions"], gate["improvements"]), (0, 0))
        self.assertEqual(gate["verdict"], "change")

    def test_winner_loss_is_regression_gain_is_improvement_swap_is_neither(self):
        loss = _diff_dict(winner={"changed": True, "old": "a", "new": None})
        self.assertEqual(self.verdict(loss)["verdict"], "regression")
        gain = _diff_dict(winner={"changed": True, "old": None, "new": "a"})
        self.assertEqual(self.verdict(gain)["verdict"], "improvement")
        swap = _diff_dict(winner={"changed": True, "old": "a", "new": "b"})
        self.assertEqual(self.verdict(swap)["verdict"], "change")

    def test_tolerated_regressions_pass_as_change_with_counts_visible(self):
        diff = _diff_dict(fixture_diffs=[{"old": "pass", "new": "fail"}])
        gate = self.verdict(diff, max_regressions=1)
        self.assertEqual(gate["verdict"], "change")
        self.assertEqual(gate["regressions"], 1)
        self.assertEqual(gate["max_regressions"], 1)

    def test_regression_dominates_improvement(self):
        diff = _diff_dict(
            fixture_diffs=[
                {"old": "pass", "new": "fail"},
                {"old": "fail", "new": "pass"},
            ]
        )
        gate = self.verdict(diff)
        self.assertEqual(gate["verdict"], "regression")
        self.assertEqual((gate["regressions"], gate["improvements"]), (1, 1))


class GoldenIdentityTests(_PairDir):
    """Default mode is byte-identical to the pre-gate contract."""

    def test_pair_a_default_matches_golden(self):
        code, out, err = _run_diff(["old.json", "old.json"], self.dir)
        self.assertEqual(code, 0)
        self.assertEqual(
            out,
            (GOLDEN_DIR / "w7_r4_golden_a_stdout.txt").read_text(encoding="utf-8"),
        )
        self.assertEqual(
            err,
            (GOLDEN_DIR / "w7_r4_golden_a_stderr.txt").read_text(encoding="utf-8"),
        )

    def test_pair_b_default_matches_golden(self):
        code, out, err = _run_diff(["old.json", "new.json"], self.dir)
        self.assertEqual(code, 0)
        self.assertEqual(
            out,
            (GOLDEN_DIR / "w7_r4_golden_b_stdout.txt").read_text(encoding="utf-8"),
        )
        self.assertEqual(
            err,
            (GOLDEN_DIR / "w7_r4_golden_b_stderr.txt").read_text(encoding="utf-8"),
        )

    def test_gate_mode_adds_only_the_gate_object(self):
        _, default_out, _ = _run_diff(["old.json", "new.json"], self.dir)
        code, gate_out, gate_err = _run_diff(["old.json", "new.json", "--gate"], self.dir)
        self.assertEqual(code, 2)
        payload = json.loads(gate_out)
        gate = payload.pop("gate")
        self.assertEqual(payload, json.loads(default_out))
        self.assertEqual(gate["verdict"], "regression")
        _, _, default_err = _run_diff(["old.json", "new.json"], self.dir)
        self.assertEqual(gate_err, default_err)


class GateWrapperTests(_PairDir):
    def test_identical_pair_gates_no_change_exit_0(self):
        code, out, _ = _run_diff(["old.json", "old.json", "--gate"], self.dir)
        self.assertEqual(code, 0)
        gate = json.loads(out)["gate"]
        self.assertEqual(
            gate,
            {
                "verdict": "no_change",
                "regressions": 0,
                "improvements": 0,
                "max_regressions": 0,
                "threshold_source": "default",
            },
        )

    def test_perturbed_pair_gates_regression_exit_2(self):
        code, out, _ = _run_diff(["old.json", "new.json", "--gate"], self.dir)
        self.assertEqual(code, 2)
        gate = json.loads(out)["gate"]
        # llama3 search pass->fail and held_out pass->absent regress;
        # chatml held_out absent->pass improves; the winner swap is
        # neither. Counts from the pair's known diff, by hand.
        self.assertEqual(gate["verdict"], "regression")
        self.assertEqual((gate["regressions"], gate["improvements"]), (2, 1))

    def test_flag_threshold_tolerates_and_reports_flag_source(self):
        code, out, _ = _run_diff(
            ["old.json", "new.json", "--gate", "--max-regressions", "2"], self.dir
        )
        self.assertEqual(code, 0)
        gate = json.loads(out)["gate"]
        self.assertEqual(gate["verdict"], "change")
        self.assertEqual(gate["threshold_source"], "flag")
        self.assertEqual(gate["max_regressions"], 2)

    def test_env_threshold_and_flag_precedence(self):
        with mock.patch.dict(os.environ, {ENV_VAR: "2"}):
            code, out, _ = _run_diff(["old.json", "new.json", "--gate"], self.dir)
            self.assertEqual(code, 0)
            gate = json.loads(out)["gate"]
            self.assertEqual(gate["threshold_source"], "env")
            code, out, _ = _run_diff(
                ["old.json", "new.json", "--gate", "--max-regressions", "0"],
                self.dir,
            )
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(out)["gate"]["threshold_source"], "flag")

    def test_malformed_env_fails_closed(self):
        # The last shape passes the canonical grammar but exceeds
        # CPython's integer-string conversion limit (BH15-2): it
        # must fail closed like every other unusable value, never
        # escape as a raw ValueError traceback.
        for raw in ("abc", "-1", "", " 2", "2.0", "007", "9" * 5000):
            with self.subTest(raw=raw):
                with mock.patch.dict(os.environ, {ENV_VAR: raw}):
                    code, out, err = _run_diff(["old.json", "new.json", "--gate"], self.dir)
                self.assertEqual(code, 1)
                self.assertEqual(out, "")
                self.assertIn(ENV_VAR, err)

    def test_max_regressions_without_gate_is_a_parser_error(self):
        parser = cli.build_parser()
        args = parser.parse_args(["diff-reports", "old.json", "new.json", "--max-regressions", "1"])
        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as caught,
        ):
            run_diff_reports(args, parser)
        self.assertEqual(caught.exception.code, 2)

    def test_negative_max_regressions_is_a_parser_error(self):
        parser = cli.build_parser()
        args = parser.parse_args(
            ["diff-reports", "old.json", "new.json", "--gate", "--max-regressions", "-1"]
        )
        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as caught,
        ):
            run_diff_reports(args, parser)
        self.assertEqual(caught.exception.code, 2)


class NoParserWrapperTests(unittest.TestCase):
    """Hand-built Namespaces called with no parser (the cycle-225 /
    W2-boundary caller shape): the refusals must still fire cleanly
    instead of dereferencing a missing parser."""

    def _args(self, **overrides) -> argparse.Namespace:
        base = {
            "gate": False,
            "junit": None,
            "max_regressions": None,
            "old_results": "old.json",
            "new_results": "new.json",
            "all": False,
        }
        base.update(overrides)
        return argparse.Namespace(**base)

    def test_max_regressions_without_gate_no_parser_returns_2(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = run_diff_reports(self._args(max_regressions=1), None)
        self.assertEqual(code, 2)
        self.assertIn("--max-regressions requires --gate", stderr.getvalue())

    def test_negative_max_regressions_no_parser_raises_systemexit_2(self):
        stderr = io.StringIO()
        with (
            contextlib.redirect_stderr(stderr),
            self.assertRaises(SystemExit) as caught,
        ):
            run_diff_reports(self._args(gate=True, max_regressions=-1), None)
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("--max-regressions must be a non-negative integer", stderr.getvalue())


class EvidenceGateTests(_PairDir):
    """Fixture-level evidence: flips count; one-sided evidence cannot judge."""

    def _with_observations(self, payload: dict, flip: bool) -> dict:
        for entry in payload["search"]:
            fmt = entry.get("format") or {}
            if fmt.get("name") != "llama3":
                continue
            entry["observations"] = [
                {
                    "fixture": "fx-one",
                    "exact_match": not (flip and True),
                    "boundary_found": True,
                },
                {"fixture": "fx-two", "exact_match": True, "boundary_found": True},
            ]
            return payload
        raise AssertionError("sample has no llama3 search entry")

    def test_fixture_flip_counts_and_appears_in_junit(self):
        old = self._with_observations(_sample(), flip=False)
        new = self._with_observations(_sample(), flip=True)
        (self.dir / "old.json").write_text(json.dumps(old), encoding="utf-8")
        (self.dir / "new.json").write_text(json.dumps(new), encoding="utf-8")
        junit = self.dir / "diff.xml"
        code, out, _ = _run_diff(
            ["old.json", "new.json", "--all", "--gate", "--junit", "diff.xml"],
            self.dir,
        )
        self.assertEqual(code, 2)
        gate = json.loads(out)["gate"]
        self.assertGreaterEqual(gate["regressions"], 1)
        root = ET.parse(junit).getroot()  # noqa: S314
        fixture_cases = [
            case
            for suite in root
            for case in suite
            if case.get("name", "").startswith("fixture fx-one")
        ]
        self.assertEqual(len(fixture_cases), 1)
        self.assertEqual([child.tag for child in fixture_cases[0]], ["failure"])

    def test_one_sided_evidence_gates_unavailable(self):
        old = self._with_observations(_sample(), flip=False)
        new = _perturbed()
        (self.dir / "old.json").write_text(json.dumps(old), encoding="utf-8")
        (self.dir / "new.json").write_text(json.dumps(new), encoding="utf-8")
        junit = self.dir / "diff.xml"
        code, out, _ = _run_diff(
            ["old.json", "new.json", "--gate", "--junit", "diff.xml"], self.dir
        )
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(out)["gate"]["verdict"], "unavailable")
        root = ET.parse(junit).getroot()  # noqa: S314
        verdict_cases = [
            case for suite in root for case in suite if case.get("name", "").startswith("verdict")
        ]
        self.assertTrue(verdict_cases)
        for case in verdict_cases:
            self.assertEqual([child.tag for child in case], ["skipped"])


class JUnitDiffTests(_PairDir):
    def test_pair_b_junit_counts_match_the_diff(self):
        code, out, _ = _run_diff(["old.json", "new.json", "--junit", "diff.xml"], self.dir)
        self.assertEqual(code, 0)
        root = ET.parse(self.dir / "diff.xml").getroot()  # noqa: S314
        self.assertEqual(root.tag, "testsuites")
        self.assertEqual(
            (root.get("tests"), root.get("failures"), root.get("skipped")),
            ("3", "2", "0"),
        )
        suites = {suite.get("name"): suite for suite in root}
        self.assertEqual(set(suites), {"chatml", "llama3"})
        llama_failures = [
            case for case in suites["llama3"] if any(child.tag == "failure" for child in case)
        ]
        self.assertEqual(len(llama_failures), 2)

    def test_unwritable_junit_exits_1_after_printing_json(self):
        target = self.dir / "missing-dir" / "diff.xml"
        code, out, err = _run_diff(["old.json", "new.json", "--junit", str(target)], self.dir)
        self.assertEqual(code, 1)
        self.assertIn('"summary"', out)
        self.assertIn("junit_unwritable", err)
        self.assertFalse(target.exists())

    def test_hostile_candidate_name_still_parses(self):
        hostile = "evil\x01<>&\"'\ud800name"
        payload_old = {
            "decision": "abstain",
            "selected_candidate": None,
            "search": [
                {
                    "format": {"name": hostile},
                    "summary": {
                        "planned": 1,
                        "exact_and_boundary_passes": 1,
                        "aggregate_eligible": True,
                    },
                }
            ],
        }
        payload_new = json.loads(json.dumps(payload_old))
        payload_new["search"][0]["summary"]["exact_and_boundary_passes"] = 0
        (self.dir / "old.json").write_text(json.dumps(payload_old), encoding="utf-8")
        (self.dir / "new.json").write_text(json.dumps(payload_new), encoding="utf-8")
        code, _, _ = _run_diff(["old.json", "new.json", "--junit", "diff.xml"], self.dir)
        self.assertEqual(code, 0)
        root = ET.parse(self.dir / "diff.xml").getroot()  # noqa: S314
        names = [suite.get("name") for suite in root]
        self.assertEqual(len(names), 1)
        self.assertIn("\ufffd", names[0])


class JUnitFixtureTransitionTests(unittest.TestCase):
    """The JUnit fixture-case mapping beyond the golden pair's flips:
    a fail->pass fixture passes, and a transition through a
    non-pass/fail outcome is skipped as not comparable (the suite
    never claims fixture coverage the evidence does not support)."""

    def test_fail_to_pass_passes_and_incomparable_is_skipped(self):
        diff = _diff_dict(
            fixture_diffs=[
                {
                    "candidate": "chatml",
                    "fixture": "fx-up",
                    "stage": "search",
                    "old": "fail",
                    "new": "pass",
                },
                {
                    "candidate": "chatml",
                    "fixture": "fx-side",
                    "stage": "search",
                    "old": "pass",
                    "new": "absent",
                },
            ]
        )
        (suite,) = _diff_junit_suites(diff, None)
        by_name = {case["name"]: case for case in suite["cases"]}
        self.assertEqual(by_name["fixture fx-up (search)"]["status"], "passed")
        side = by_name["fixture fx-side (search)"]
        self.assertEqual(side["status"], "skipped")
        self.assertIn("outcome not comparable as pass/fail", side["message"])


class JUnitVerifySuiteTests(unittest.TestCase):
    """The verify-style suite builder's message fallbacks."""

    def test_failed_check_without_detail_falls_back_to_its_id(self):
        (suite,) = _checks_junit_suites(
            "verify", "verify", [{"id": "chk-9", "status": "failed", "detail": ""}]
        )
        (case,) = suite["cases"]
        self.assertEqual(case["status"], "failed")
        self.assertEqual(case["message"], "chk-9")

    def test_passed_check_carries_no_message(self):
        (suite,) = _checks_junit_suites(
            "verify", "verify", [{"id": "chk-1", "status": "passed", "detail": "ok"}]
        )
        (case,) = suite["cases"]
        self.assertEqual((case["status"], case["message"]), ("passed", ""))


class JUnitVerifyTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run_verify(self, argv: list[str]) -> tuple[int, str, str]:
        parser = cli.build_parser()
        args = parser.parse_args(["verify", *argv])
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = vf.run_verify(args, parser)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_verify_junit_counts_match_json_checks(self):
        junit = self.dir / "verify.xml"
        code, out, _ = self._run_verify([str(SAMPLE), "--junit", str(junit)])
        self.assertEqual(code, 0)
        report = json.loads(out)
        root = ET.parse(junit).getroot()  # noqa: S314
        (suite,) = list(root)
        self.assertEqual(suite.get("name"), str(SAMPLE))
        cases = list(suite)
        self.assertEqual(len(cases), len(report["checks"]))
        self.assertTrue(all(c.get("classname") == "verify" for c in cases))
        failed = sum(1 for c in report["checks"] if c["status"] == "failed")
        unavailable = sum(1 for c in report["checks"] if c["status"] == "unavailable")
        self.assertEqual(int(suite.get("failures")), failed)
        self.assertEqual(int(suite.get("skipped")), unavailable)

    def test_verify_unwritable_junit_exits_1_after_printing(self):
        target = self.dir / "missing" / "verify.xml"
        code, out, err = self._run_verify([str(SAMPLE), "--junit", str(target)])
        self.assertEqual(code, 1)
        self.assertIn('"verdict"', out)
        self.assertIn("junit_unwritable", err)
        self.assertFalse(target.exists())

    def test_bundle_junit_uses_the_seal_surface(self):
        report_dir = self.dir / "report"
        report_dir.mkdir()
        sample_dir = SAMPLE.parent
        for name in ("results.json", "report.html"):
            (report_dir / name).write_bytes((sample_dir / name).read_bytes())
        bundle_bytes, _manifest, _digest = pk.pack_bundle(report_dir, [])
        bundle = self.dir / "bundle.mplpack"
        bundle.write_bytes(bundle_bytes)
        junit = self.dir / "seal.xml"
        parser = cli.build_parser()
        args = parser.parse_args(["verify", "--bundle", str(bundle), "--junit", str(junit)])
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
            code = vf.run_verify(args, parser)
        self.assertEqual(code, 0)
        report = json.loads(stdout.getvalue())
        root = ET.parse(junit).getroot()  # noqa: S314
        (suite,) = list(root)
        self.assertEqual(suite.get("name"), str(bundle))
        cases = list(suite)
        self.assertEqual(len(cases), len(report["checks"]))
        self.assertTrue(all(c.get("classname") == "seal" for c in cases))

    def test_bundle_unwritable_junit_exits_1_after_printing(self):
        report_dir = self.dir / "report"
        report_dir.mkdir()
        sample_dir = SAMPLE.parent
        for name in ("results.json", "report.html"):
            (report_dir / name).write_bytes((sample_dir / name).read_bytes())
        bundle_bytes, _manifest, _digest = pk.pack_bundle(report_dir, [])
        bundle = self.dir / "bundle.mplpack"
        bundle.write_bytes(bundle_bytes)
        target = self.dir / "missing" / "seal.xml"
        parser = cli.build_parser()
        args = parser.parse_args(["verify", "--bundle", str(bundle), "--junit", str(target)])
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = vf.run_verify(args, parser)
        self.assertEqual(code, 1)
        self.assertIn('"verdict"', stdout.getvalue())
        self.assertIn("junit_unwritable", stderr.getvalue())
        self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
