"""W7-4 pins for the shared mutation-gate verdict (RR-15).

The verdict logic in ``tests/_mutation_gate.py`` is exercised
here with a dependency-injected fake runner: fabricated child
outcomes, staged source files in a TemporaryDirectory, no real
mutant, no real source file, no real pytest child. This file is
therefore not itself a mutation gate and can never orphan a
mutant in the tree. The cases are the census's P1-P12, including
the audit's exact negative control (a child exiting 2 with a
collection failure and zero executed tests, which the pre-W7-4
gates counted as four kills).
"""

from __future__ import annotations

import ast
import tempfile
import unittest
from pathlib import Path

from tests._mutation_gate import (
    ChildOutcome,
    MutantSpec,
    Verdict,
    _parse_junit,
    classify_child,
    run_mutation_gate,
)

AUDIT = frozenset({"AuditTests::test_kills_it", "AuditTests::test_also_kills_it"})


def outcome(rc, failures=(), errors=(), executed=(), timed_out=False, parse_ok=True):
    """Fabricate one child outcome."""
    return ChildOutcome(
        returncode=rc,
        failures=frozenset(failures),
        errors=frozenset(errors),
        executed=frozenset(executed),
        timed_out=timed_out,
        parse_ok=parse_ok,
    )


class ClassifyChildTests(unittest.TestCase):
    """The pure classifier, case by case (census P1-P10 + signals)."""

    def test_p1_exit2_collection_failure_is_inconclusive(self):
        """P1: exit 2, zero executed — the audit's negative control."""
        self.assertIs(
            classify_child(outcome(2, executed=(), parse_ok=False), AUDIT),
            Verdict.INCONCLUSIVE,
        )

    def test_p2_exit5_no_tests_collected_is_inconclusive(self):
        """P2: exit 5 with an empty outcome set."""
        self.assertIs(
            classify_child(outcome(5, executed=(), parse_ok=False), AUDIT),
            Verdict.INCONCLUSIVE,
        )

    def test_p3_exit4_usage_error_is_inconclusive(self):
        """P3: exit 4 (a registered nodeid no longer exists)."""
        self.assertIs(classify_child(outcome(4), AUDIT), Verdict.INCONCLUSIVE)

    def test_exit3_and_signal_death_are_inconclusive(self):
        """Exit 3 (internal error) and -11 (SIGSEGV) are inconclusive."""
        self.assertIs(classify_child(outcome(3), AUDIT), Verdict.INCONCLUSIVE)
        self.assertIs(classify_child(outcome(-11), AUDIT), Verdict.INCONCLUSIVE)

    def test_p5_failures_disjoint_from_audit_set_is_inconclusive(self):
        """P5: an unrelated test failed — not this mutant's kill."""
        self.assertIs(
            classify_child(
                outcome(1, failures={"Other::test_x"}, executed={"Other::test_x"} | AUDIT),
                AUDIT,
            ),
            Verdict.INCONCLUSIVE,
        )

    def test_p6_error_entry_is_inconclusive_even_with_named_failure(self):
        """P6: an <error> outcome poisons the run's evidence."""
        self.assertIs(
            classify_child(
                outcome(
                    1,
                    failures={"AuditTests::test_kills_it"},
                    errors={"AuditTests::test_broken"},
                    executed=AUDIT,
                ),
                AUDIT,
            ),
            Verdict.INCONCLUSIVE,
        )

    def test_p7_unparseable_outcome_is_inconclusive(self):
        """P7: exit 1 but the outcome set is empty/unparseable."""
        self.assertIs(
            classify_child(outcome(1, executed=(), parse_ok=False), AUDIT),
            Verdict.INCONCLUSIVE,
        )

    def test_p8_clean_pass_with_all_audit_tests_is_survived(self):
        """P8: exit 0, every audit test executed and passed."""
        self.assertIs(
            classify_child(outcome(0, executed=AUDIT | {"Other::test_y"}), AUDIT),
            Verdict.SURVIVED,
        )

    def test_p9_clean_pass_missing_an_audit_test_is_inconclusive(self):
        """P9: exit 0 but a registered audit test never ran."""
        self.assertIs(
            classify_child(outcome(0, executed={"AuditTests::test_kills_it"}), AUDIT),
            Verdict.INCONCLUSIVE,
        )

    def test_p10_named_failure_is_killed(self):
        """P10: exit 1, a registered audit test failed, no errors."""
        self.assertIs(
            classify_child(
                outcome(
                    1,
                    failures={"AuditTests::test_kills_it"},
                    executed=AUDIT | {"Other::test_y"},
                ),
                AUDIT,
            ),
            Verdict.KILLED,
        )


class ParseJunitSkippedTests(unittest.TestCase):
    """BH15-1: a skipped testcase is not an executed testcase.

    The helper's rule is that survival requires every registered
    audit test executed, and the baseline requires the same. A
    testcase carrying a ``<skipped/>`` child (a skip or xfail
    marker) never executed its assertions, so ``_parse_junit``
    must keep it out of ``executed`` — otherwise a baseline
    accepts a run whose audit test never ran, and a mutant whose
    only registered killer is skipped classifies SURVIVED
    instead of INCONCLUSIVE. Synthetic JUnit only; no live gate.
    """

    def _parse(self, body: str):
        """Write one synthetic JUnit document and parse it."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "out.xml"
        path.write_text(
            '<?xml version="1.0" encoding="utf-8"?>\n'
            f'<testsuites><testsuite name="pytest">{body}</testsuite></testsuites>\n',
            encoding="utf-8",
        )
        return _parse_junit(path)

    def test_skipped_testcase_is_not_executed(self):
        """A skipped audit testcase stays out of the executed set."""
        failures, errors, executed, ok = self._parse(
            '<testcase classname="mod.AuditTests" name="test_kills_it">'
            '<skipped type="pytest.skip" message="environment-gated"/>'
            "</testcase>"
            '<testcase classname="mod.AuditTests" name="test_also_kills_it"/>'
        )
        self.assertEqual(executed, frozenset({"AuditTests::test_also_kills_it"}))
        self.assertEqual(failures, frozenset())
        self.assertEqual(errors, frozenset())
        self.assertTrue(ok)

    def test_all_skipped_is_not_a_parseable_execution(self):
        """Every testcase skipped: nothing executed, parse not ok."""
        _failures, _errors, executed, ok = self._parse(
            '<testcase classname="mod.AuditTests" name="test_kills_it">'
            '<skipped type="pytest.xfail" message="expected failure"/>'
            "</testcase>"
        )
        self.assertEqual(executed, frozenset())
        self.assertFalse(ok)

    def test_skipped_audit_test_cannot_support_a_survived_verdict(self):
        """rc 0 with the audit test skipped classifies INCONCLUSIVE."""
        _failures, _errors, executed, ok = self._parse(
            '<testcase classname="mod.AuditTests" name="test_kills_it">'
            '<skipped type="pytest.skip" message="environment-gated"/>'
            "</testcase>"
            '<testcase classname="mod.Other" name="test_y"/>'
        )
        verdict = classify_child(
            outcome(0, executed=executed, parse_ok=ok),
            frozenset({"AuditTests::test_kills_it"}),
        )
        self.assertIs(verdict, Verdict.INCONCLUSIVE)

    def test_failure_testcase_still_parsed_alongside_a_skip(self):
        """Skips do not disturb the failure accounting of real runs."""
        failures, _errors, executed, ok = self._parse(
            '<testcase classname="mod.AuditTests" name="test_kills_it">'
            '<skipped type="pytest.skip" message="environment-gated"/>'
            "</testcase>"
            '<testcase classname="mod.AuditTests" name="test_also_kills_it">'
            "<failure>assert</failure>"
            "</testcase>"
        )
        self.assertEqual(failures, frozenset({"AuditTests::test_also_kills_it"}))
        self.assertEqual(executed, frozenset({"AuditTests::test_also_kills_it"}))
        self.assertTrue(ok)


class FakeRunner:
    """A runner stub: outcomes scripted by the tag embedded in argv."""

    def __init__(self, script):
        """Script maps argv tag -> ChildOutcome."""
        self.script = script
        self.calls: list[str] = []

    def __call__(self, argv, junit_path, timeout, cwd):
        """Return the scripted outcome for argv's tag."""
        tag = argv[-1]
        self.calls.append(tag)
        return self.script[tag]


class RunMutationGateTests(unittest.TestCase):
    """Aggregation: baselines void, timeouts don't abort (P4/P11/P12)."""

    def _specs(self, root: Path, names=("m1",)):
        """Stage fake source files and specs in a temp directory."""
        specs = []
        for name in names:
            source = root / f"{name}.py"
            source.write_text("value = old + 1\n", encoding="utf-8")
            specs.append(
                MutantSpec(
                    name=name,
                    source=source,
                    anchor="value = old + 1",
                    replacement="value = old + 2",
                    audit_tests=tuple(sorted(AUDIT)),
                )
            )
        return specs

    def _run(self, script, names=("m1",), shared_baseline=True):
        """Run the gate over staged specs with a scripted runner."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        specs = self._specs(root, names)
        runner = FakeRunner(script)

        def argv_for(spec):
            if spec is None:
                return ["pytest", "baseline"] if shared_baseline else None
            return ["pytest", spec.name]

        result = run_mutation_gate(
            specs,
            argv_for=argv_for,
            timeout=10,
            cwd=root,
            junit_dir=root,
            runner=runner,
        )
        return result, runner, specs

    def test_p4_timeout_is_inconclusive_and_remaining_mutants_still_classified(self):
        """P4: one mutant times out; the other is still classified."""
        kill = outcome(
            1, failures={"AuditTests::test_kills_it"}, executed=AUDIT | {"Other::test_y"}
        )
        result, _runner, _specs = self._run(
            {
                "baseline": outcome(0, executed=AUDIT),
                "m1": outcome(None, timed_out=True, parse_ok=False),
                "m2": kill,
            },
            names=("m1", "m2"),
        )
        self.assertEqual(result.verdicts, {"m1": Verdict.INCONCLUSIVE, "m2": Verdict.KILLED})
        self.assertFalse(result.passed)
        self.assertIn("inconclusive", result.summary())

    def test_p11_failed_baseline_voids_the_gate(self):
        """P11: a failing baseline counts no mutant verdict at all."""
        result, runner, _specs = self._run(
            {
                "baseline": outcome(1, failures={"AuditTests::test_kills_it"}, executed=AUDIT),
                "m1": outcome(1, failures={"AuditTests::test_kills_it"}, executed=AUDIT),
            }
        )
        self.assertFalse(result.baseline_ok)
        self.assertEqual(result.verdicts, {})
        self.assertNotIn("m1", runner.calls)
        self.assertFalse(result.passed)

    def test_p12_happy_path_passes(self):
        """P12: baseline passes and every mutant is killed by name."""
        kill = outcome(1, failures={"AuditTests::test_also_kills_it"}, executed=AUDIT)
        result, _runner, specs = self._run(
            {"baseline": outcome(0, executed=AUDIT), "m1": kill, "m2": kill},
            names=("m1", "m2"),
        )
        self.assertTrue(result.passed)
        # Restoration invariant: staged sources are byte-identical.
        for spec in specs:
            self.assertEqual(spec.source.read_text(encoding="utf-8"), "value = old + 1\n")

    def test_survived_reported_distinctly_from_inconclusive(self):
        """A survived mutant fails the gate as survived, not inconclusive."""
        result, _runner, _specs = self._run(
            {
                "baseline": outcome(0, executed=AUDIT),
                "m1": outcome(0, executed=AUDIT),
            }
        )
        self.assertEqual(result.verdicts, {"m1": Verdict.SURVIVED})
        self.assertIn("survived: ['m1']", result.summary())
        self.assertNotIn("inconclusive", result.summary())

    def test_anchor_not_unique_fails_closed_before_any_child(self):
        """Anchor preflight: a duplicated anchor voids the gate."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        source = root / "m1.py"
        source.write_text("value = old + 1\nvalue = old + 1\n", encoding="utf-8")
        spec = MutantSpec(
            name="m1",
            source=source,
            anchor="value = old + 1",
            replacement="value = old + 2",
            audit_tests=tuple(sorted(AUDIT)),
        )
        runner = FakeRunner({})
        result = run_mutation_gate(
            [spec],
            argv_for=lambda s: ["pytest", "baseline"] if s is None else ["pytest", s.name],
            timeout=10,
            cwd=root,
            junit_dir=root,
            runner=runner,
        )
        self.assertFalse(result.baseline_ok)
        self.assertEqual(runner.calls, [])

    def test_per_spec_baseline_shape(self):
        """Targets-shaped gates baseline each spec's own argv."""
        kill = outcome(1, failures={"AuditTests::test_kills_it"}, executed=AUDIT)
        result, runner, _specs = self._run(
            {"m1": kill, "baseline_m1": outcome(0, executed=AUDIT)},
            shared_baseline=False,
        )
        # The fake's per-spec baseline is keyed by the same argv as
        # the mutant run; the script returns the kill for both, so
        # the baseline (which requires rc 0) voids — proving the
        # per-spec baseline ran the spec's argv first.
        self.assertFalse(result.baseline_ok)
        self.assertEqual(runner.calls, ["m1"])


class GateWiringTests(unittest.TestCase):
    """The six in-place gates delegate to the shared helper."""

    GATES = (
        "test_cycle124_margin_trajectory_audit.py",
        "test_cycle125_frontier_diff_audit.py",
        "test_cycle131_verify_audit.py",
        "test_cycle134_boundary_audit.py",
        "test_cycle135_marker_isolation.py",
        "test_cycle136_template_firewall.py",
    )

    def test_each_gate_uses_run_mutation_gate(self):
        """Every gate file calls run_mutation_gate (no hand-rolled verdict)."""
        tests_dir = Path(__file__).parent
        for fname in self.GATES:
            with self.subTest(gate=fname):
                source = (tests_dir / fname).read_text(encoding="utf-8")
                self.assertIn("tests._mutation_gate", source)
                self.assertIn("run_mutation_gate", source)
                # No gate method may branch on a raw child return
                # code: the string of the old verdict shape is gone.
                self.assertNotIn("returncode != 0", source)
                ast.parse(source)  # the gate file still parses


if __name__ == "__main__":
    unittest.main()
