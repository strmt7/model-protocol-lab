"""Shared verdict machinery for the in-place mutation gates (RR-15).

Before W7-4, six test files each hand-rolled the same gate: rewrite
a real source file with a mutant, spawn a child pytest, and count
the mutant as killed iff the child's exit code was non-zero. That
verdict is unsound — a collection error (exit 2), a usage error
(exit 4), no tests collected (exit 5), a timeout, or an unrelated
test's failure all count as "kills" under it, and no gate ran an
unmutated baseline control (audit finding RR-15; the audit's
negative control counted 4 kills from a child that executed zero
semantic tests).

This module is the single implementation of the repaired rule:

* every mutant is a :class:`MutantSpec` — a stable name, the real
  file it mutates, an anchor that must occur exactly once, the
  replacement text, and the registered audit set: the named tests
  in the gate's own file that execution proved kill this mutant;
* the gate first runs the child against the unmutated source
  (the baseline) and requires it to pass with every registered
  audit test executed — a failed baseline voids the whole gate;
* each mutant's child outcome is classified by the pure function
  :func:`classify_child` from the child's exit code and its
  parsed JUnit outcomes: KILLED only when the run completed with
  test failures (exit 1), no errors, and at least one registered
  audit test among the failures; SURVIVED only when the run
  passed (exit 0) with every audit test executed; everything
  else — exit 2/3/4/5, a signal death, a timeout, an error entry,
  an empty or unparseable outcome set, failures disjoint from
  the audit set, an audit test that never ran — is INCONCLUSIVE,
  which fails the gate without counting as a kill or a survival;
* the source file is restored after every mutant and the restore
  is verified by hash, so a gate can never silently leave the
  tree mutated (its ``__pycache__`` entry is invalidated too, per
  the same-size-mutant bytecode lesson).

The classification is a pure function of the parsed outcome, so
``tests/test_w7_mutation_gate_verdict.py`` pins it (including the
audit's exit-2 negative control) with a fake runner and never
touches a real source file.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class Verdict(Enum):
    """One mutant's adjudicated outcome."""

    KILLED = "killed"
    SURVIVED = "survived"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class MutantSpec:
    """A registered mutant: what it changes and who must catch it."""

    name: str
    source: Path
    anchor: str
    replacement: str
    audit_tests: tuple[str, ...]  # "ClassName::test_method" in the gate's file


@dataclass(frozen=True)
class ChildOutcome:
    """A child run's parsed outcome (the classifier's only input)."""

    returncode: int | None  # None iff the child timed out
    failures: frozenset[str]
    errors: frozenset[str]
    executed: frozenset[str]
    timed_out: bool
    parse_ok: bool  # a JUnit report existed, parsed, and was non-empty


def classify_child(outcome: ChildOutcome, expected: frozenset[str]) -> Verdict:
    """Classify one mutant child run (the RR-15 repaired rule).

    ``expected`` is the mutant's registered audit set. The rule is
    deliberately asymmetric: evidence of a kill must be positive
    (a named audit test failed in a clean test-failure run), and
    evidence of survival must be complete (every audit test ran
    and passed). Anything ambiguous is INCONCLUSIVE.
    """
    if outcome.timed_out or outcome.returncode is None:
        return Verdict.INCONCLUSIVE
    if outcome.returncode == 0:
        if expected and expected <= outcome.executed and not outcome.errors and outcome.parse_ok:
            return Verdict.SURVIVED
        return Verdict.INCONCLUSIVE
    if outcome.returncode != 1:
        # 2 interrupted/collection, 3 internal, 4 usage, 5 no tests
        # collected, negative = killed by a signal: none of these
        # can evidence a behavioural kill.
        return Verdict.INCONCLUSIVE
    if not outcome.parse_ok or not outcome.executed or outcome.errors:
        return Verdict.INCONCLUSIVE
    if outcome.failures & expected:
        return Verdict.KILLED
    return Verdict.INCONCLUSIVE


def _parse_junit(path: Path) -> tuple[frozenset[str], frozenset[str], frozenset[str], bool]:
    """Parse a JUnit report into (failures, errors, executed, ok)."""
    try:
        # The report is written by the gate's own child process
        # into the gate's private temp dir — trusted local output,
        # never an external payload.
        tree = ET.parse(path)  # noqa: S314
    except (OSError, ET.ParseError):
        return frozenset(), frozenset(), frozenset(), False
    failures: set[str] = set()
    errors: set[str] = set()
    executed: set[str] = set()
    for tc in tree.getroot().iter("testcase"):
        classname = tc.get("classname") or ""
        ident = f"{classname.split('.')[-1]}::{tc.get('name')}"
        # A testcase carrying <skipped/> (a skip or xfail marker)
        # never executed its assertions: it is not evidence that
        # the test ran, so it stays out of `executed` (BH15-1).
        # Counting it let a baseline accept a run whose audit test
        # never ran and let a mutant classify SURVIVED on a killer
        # that never fired.
        if tc.find("skipped") is not None:
            continue
        executed.add(ident)
        if tc.find("failure") is not None:
            failures.add(ident)
        if tc.find("error") is not None:
            errors.add(ident)
    return frozenset(failures), frozenset(errors), frozenset(executed), bool(executed)


def run_child(argv: list[str], junit_path: Path, timeout: float, cwd: Path) -> ChildOutcome:
    """Run one child with JUnit capture; timeouts become outcomes."""
    if junit_path.exists():
        junit_path.unlink()
    try:
        proc = subprocess.run(
            [*argv, f"--junitxml={junit_path}"],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=timeout,
            check=False,
        )
        returncode: int | None = proc.returncode
        timed_out = False
    except subprocess.TimeoutExpired:
        returncode = None
        timed_out = True
    failures, errors, executed, parse_ok = _parse_junit(junit_path)
    if timed_out:
        parse_ok = False
    return ChildOutcome(
        returncode=returncode,
        failures=failures,
        errors=errors,
        executed=executed,
        timed_out=timed_out,
        parse_ok=parse_ok,
    )


@dataclass
class GateResult:
    """The aggregated outcome of one gate run."""

    baseline_ok: bool
    baseline_detail: str = ""
    verdicts: dict[str, Verdict] = field(default_factory=dict)
    failures: dict[str, frozenset[str]] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Pass iff baseline ok and verdicts exist and are all KILLED."""
        return (
            self.baseline_ok
            and bool(self.verdicts)
            and all(v is Verdict.KILLED for v in self.verdicts.values())
        )

    def summary(self) -> str:
        """Failure message distinguishing survived from inconclusive."""
        if not self.baseline_ok:
            return f"mutation gate baseline inconclusive: {self.baseline_detail}"
        survived = sorted(n for n, v in self.verdicts.items() if v is Verdict.SURVIVED)
        inconclusive = sorted(n for n, v in self.verdicts.items() if v is Verdict.INCONCLUSIVE)
        parts = []
        if survived:
            parts.append(f"survived: {survived}")
        if inconclusive:
            parts.append(f"inconclusive: {inconclusive}")
        return "; ".join(parts) if parts else "mutation gate failed"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_mutation_gate(
    specs: list[MutantSpec],
    *,
    argv_for,
    timeout: float,
    cwd: Path,
    junit_dir: Path,
    runner=run_child,
) -> GateResult:
    """Run one gate: baseline first, then each mutant in place.

    ``argv_for`` maps a spec (or ``None`` for a shared baseline) to
    the child argv WITHOUT the junit flag (the runner appends it).
    When ``argv_for(None)`` returns a list, that single baseline
    run must execute every spec's audit tests; when it returns
    ``None``, the baseline is per-spec (each spec's own argv run
    against the unmutated source). A baseline that does not pass
    cleanly voids the gate: no mutant is attempted and the result
    carries no verdicts.
    """
    # Anchor preflight: every anchor occurs exactly once.
    for spec in specs:
        text = spec.source.read_text(encoding="utf-8")
        count = text.count(spec.anchor)
        if count != 1:
            return GateResult(
                baseline_ok=False,
                baseline_detail=(
                    f"anchor for {spec.name} occurs {count} times in "
                    f"{spec.source.name} (must be exactly 1)"
                ),
            )
    expected_all: frozenset[str] = frozenset().union(*(frozenset(s.audit_tests) for s in specs))
    shared = argv_for(None)
    if shared is not None:
        outcome = runner(shared, junit_dir / "baseline.xml", timeout, cwd)
        if not (
            outcome.returncode == 0
            and outcome.parse_ok
            and not outcome.errors
            and expected_all <= outcome.executed
        ):
            return GateResult(
                baseline_ok=False,
                baseline_detail=(
                    f"shared baseline rc={outcome.returncode} "
                    f"errors={sorted(outcome.errors)} "
                    f"missing={sorted(expected_all - outcome.executed)}"
                ),
            )
    else:
        for spec in specs:
            outcome = runner(argv_for(spec), junit_dir / f"baseline_{spec.name}.xml", timeout, cwd)
            expected = frozenset(spec.audit_tests)
            if not (
                outcome.returncode == 0
                and outcome.parse_ok
                and not outcome.errors
                and expected <= outcome.executed
            ):
                return GateResult(
                    baseline_ok=False,
                    baseline_detail=(
                        f"baseline for {spec.name} rc={outcome.returncode} "
                        f"errors={sorted(outcome.errors)} "
                        f"missing={sorted(expected - outcome.executed)}"
                    ),
                )
    result = GateResult(baseline_ok=True)
    for spec in specs:
        original_bytes = spec.source.read_bytes()
        original_hash = _sha256(original_bytes)
        text = original_bytes.decode("utf-8")
        outcome: ChildOutcome
        try:
            spec.source.write_text(text.replace(spec.anchor, spec.replacement, 1), encoding="utf-8")
            outcome = runner(argv_for(spec), junit_dir / f"mutant_{spec.name}.xml", timeout, cwd)
        finally:
            spec.source.write_bytes(original_bytes)
            # A same-size mutant can leave a .pyc whose (mtime, size)
            # still validates against the restored source, poisoning
            # later imports with mutant bytecode; invalidate it.
            for pyc in spec.source.parent.glob(f"__pycache__/{spec.source.stem}.*.pyc"):
                pyc.unlink(missing_ok=True)
            restored = _sha256(spec.source.read_bytes())
            if restored != original_hash:
                raise AssertionError(
                    f"mutation gate failed to restore {spec.source}: "
                    "source bytes differ from the pre-gate hash"
                )
        result.verdicts[spec.name] = classify_child(outcome, frozenset(spec.audit_tests))
        result.failures[spec.name] = outcome.failures
    return result


def file_child_argv(test_file: str, gate_class: str) -> list[str]:
    """The whole-file child argv shared by the file-shaped gates.

    ``-k`` (never ``--deselect`` with an absolute path — pytest
    normalizes node IDs to relative paths, so an absolute deselect
    silently matches nothing and the gate re-enters itself).
    """
    return [
        sys.executable,
        "-m",
        "pytest",
        test_file,
        "-q",
        "-p",
        "no:cacheprovider",
        "-k",
        f"not {gate_class}",
    ]
