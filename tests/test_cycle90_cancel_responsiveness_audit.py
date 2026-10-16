"""Cycle 90: adversarial cancellation-responsiveness audit.

METHOD
------
Cycle 89 wired SIGTERM to the same cooperative ``threading.Event`` cancel
as SIGINT around ``run_experiment`` in ``cli.py``. That fix is only useful
if the event is actually HONORED promptly through the stack, so this cycle
traces the event and then measures responsiveness by execution:

1. Static trace of the ``cancel`` event: ``cli.run`` signal handlers ->
   ``experiments.run_experiment`` -> ``core.evaluate`` -> ``Backend.generate``,
   plus every other loop that runs model compute: the per-format search
   loop, ``prompt_sensitivity_sweep``, ``logprob_diagnostic``,
   ``template_conformance`` diffs, the stop matrix, the budget frontier,
   and the backend-divergence section.
2. Execution measurement with a fake backend whose ``generate()`` sleeps
   ~5s total in 50ms increments, checking the event each increment (the
   cooperative contract a real backend honors): set the event from another
   thread after 0.5s and time ``run_experiment``. A second fake with a slow
   ``forward_logits`` does the same for ``logprob_diagnostic``.
3. The ``debug`` REPL and ``doctor --portability-matrix`` are checked for
   the same class of issue.

FINDINGS
--------
- CLEAN: ``run_experiment`` honors the event promptly. Measured: 0.5s to a
  "cancelled" decision vs ~10s uncancelled (ratio ~0.05, bound < 0.50).
  Checkpoints verified: ``evaluate()`` checks the event before every case
  and around ``generate()``/``decode()``; ``run_experiment`` drains through
  ``aggregate_eligible == False`` to decision ``"cancelled"``; the selected
  candidate stays None, held_out stays None, and ``finish()`` still
  attaches the failure taxonomy.
- REAL BUG (fixed this cycle): ``experiments.logprob_diagnostic`` accepted
  no cancel event and ran its full forward-pass loop (one forward per
  candidate x fixture) even with the event set -- measured 2.0s of forward
  passes with a pre-set event, returning ``available=True`` as if nothing
  happened. In production this runs after a Ctrl-C when
  ``--logprob-diagnostic`` is on, delaying the drain to the "cancelled"
  verdict. Fix: optional ``cancel`` event (same ``threading.Event``
  contract and ValueError validation as ``evaluate``/``run_experiment``),
  checked per fixture -- the forward-pass granularity -- returning
  ``available=False`` with the fixed ``detail`` ``"cancelled"``; candidates
  fully scored before the cancel are kept, the in-progress candidate is
  dropped (never partially aggregated). ``cli.run`` now passes its
  signal-driven event through.
- CLEAN: ``prompt_sensitivity_sweep`` forwards the event to every nested
  ``run_experiment`` and drains in milliseconds once set (measured with a
  pre-set event); no change needed.
- Deliberate exclusions, pinned by test (no cancel plumbing by design):
  ``debug`` REPL (never loads model weights; tokenizer/prompt-render only;
  KeyboardInterrupt exits the cmdloop cleanly), ``doctor``
  / ``--portability-matrix`` (bounded tokenizer loads and encode checks; no
  generate loops), ``stop_matrix`` / ``budget_frontier`` /
  ``divergence_section_data`` (post-hoc over recorded observations; the
  only backend call is a single ``decode`` of the boundary), and
  ``template_conformance`` (tokenizer-only local reads, no model calls).
  ``budget_frontier`` already degrades to a cheap ``"search_incomplete"``
  marker on cancelled runs.

Verified by running: 2026-10-16 -- this file green, plus the existing
logprob / sensitivity-sweep / CLI suites; ruff check, ruff format
--check, and mypy clean.
"""

import inspect
import threading
import time
import unittest
from threading import Event

from model_protocol_lab.core import (
    Fixture,
    Format,
    Generation,
    Message,
    Split,
    evaluate,
)
from model_protocol_lab.debug import run_debug
from model_protocol_lab.diffbackends import divergence_section_data
from model_protocol_lab.doctor import doctor, template_conformance
from model_protocol_lab.experiments import (
    budget_frontier,
    logprob_diagnostic,
    prompt_sensitivity_sweep,
    run_experiment,
    stop_matrix,
)

BOUNDARY = (999,)
GEN_SECONDS = 5.0
GEN_STEP = 0.05
MAX_NEW_TOKENS = 8

FMT = Format(name="f", bos="", user_prefix="U:", assistant_prefix="A:", turn_end="\n")


def make_split(n_search: int, n_held: int) -> Split:
    """Fixtures whose answers decode to "" under the null codec below."""

    def fx(prefix: str, i: int) -> Fixture:
        return Fixture(
            id=f"{prefix}-{i}",
            messages=(Message("user", f"q-{prefix}-{i}"),),
            expected="",
            group=f"{prefix}g{i}",
        )

    return Split(
        search=tuple(fx("s", i) for i in range(n_search)),
        held_out=tuple(fx("h", i) for i in range(n_held)),
    )


class SlowCooperativeBackend:
    """Fake backend: generate() sleeps GEN_SECONDS total in small increments,
    honoring the cancel event each increment like a cooperative real backend.
    """

    fingerprint = "cycle90-slow-cooperative-v1"

    def encode(self, text: str) -> tuple[int, ...]:
        return (1, 2, 3)

    def decode(self, token_ids: tuple[int, ...]) -> str:
        return ""

    def generate(
        self,
        input_ids: tuple[int, ...],
        *,
        max_new_tokens: int,
        seed: int,
        cancel: Event,
    ) -> Generation:
        if not isinstance(cancel, Event):
            return Generation((), "error", "invalid_cancellation_event")
        if cancel.is_set():
            return Generation((), "cancelled")
        steps = int(GEN_SECONDS / GEN_STEP)
        for _ in range(steps):
            time.sleep(GEN_STEP)
            if cancel.is_set():
                return Generation((), "cancelled")
        # Exactly max_new_tokens ids ("limit" requires len == cap); the
        # boundary sits at index 2, so decode("") == expected "" passes.
        payload = (5, 6, 999) + tuple(1000 + i for i in range(max_new_tokens - 3))
        return Generation(payload, "limit")


class SlowLogprobBackend:
    """Self-consistent ord/chr codec (so check_alignment passes) with a slow
    forward_logits. Optionally sets ``cancel`` during the first forward, so
    mid-run cancellation is deterministic, not timing-dependent.
    """

    fingerprint = "cycle90-slow-logprob-v1"

    def __init__(self, forward_seconds: float = 0.25, cancel: Event | None = None) -> None:
        self._forward_seconds = forward_seconds
        self._cancel = cancel
        self.forward_calls = 0

    def encode(self, text: str) -> tuple[int, ...]:
        return tuple(ord(c) for c in text)

    def decode(self, token_ids: tuple[int, ...]) -> str:
        return "".join(chr(i) for i in token_ids)

    def generate(
        self,
        input_ids: tuple[int, ...],
        *,
        max_new_tokens: int,
        seed: int,
        cancel: Event,
    ) -> Generation:
        raise AssertionError("logprob_diagnostic never calls generate")

    def forward_logits(self, input_ids: tuple[int, ...]) -> list[list[float]]:
        self.forward_calls += 1
        if self._cancel is not None and self.forward_calls == 1:
            self._cancel.set()
        time.sleep(self._forward_seconds)
        return [[1.0, 0.0]] * (len(input_ids) + 1)


def make_diag_inputs(n_fixtures: int = 3):
    """A completed search result plus matching fixtures for the diagnostic."""
    observations = [
        {
            "fixture": f"d-{i}",
            "status": "completed",
            "input_ids": [ord("q")],
            "exact_match": True,
            "boundary_found": True,
        }
        for i in range(n_fixtures)
    ]
    result = {"search": [{"format": {"name": "f"}, "observations": observations}]}
    fixtures = tuple(
        Fixture(
            id=f"d-{i}",
            messages=(Message("user", f"qd-{i}"),),
            expected="ab",
            group=f"dg{i}",
        )
        for i in range(n_fixtures)
    )
    return result, fixtures


class CancelWiringTest(unittest.TestCase):
    """Pin the cancel-event call graph: which functions accept the event."""

    def test_cancel_param_present_where_model_compute_runs(self) -> None:
        for func in (run_experiment, evaluate, prompt_sensitivity_sweep, logprob_diagnostic):
            with self.subTest(func=func.__name__):
                self.assertIn("cancel", inspect.signature(func).parameters)

    def test_logprob_diagnostic_rejects_non_event_cancel(self) -> None:
        result, fixtures = make_diag_inputs()
        with self.assertRaises(ValueError):
            logprob_diagnostic(
                SlowLogprobBackend(),
                result,
                fixtures,
                cancel="not-an-event",  # type: ignore[arg-type]
            )


class RunExperimentResponsivenessTest(unittest.TestCase):
    """Measure that run_experiment honors a mid-run cancel promptly."""

    def test_cancel_returns_well_before_uncancelled_runtime(self) -> None:
        split = make_split(1, 1)
        nominal = 2 * GEN_SECONDS  # one search generate + one held-out generate

        start = time.perf_counter()
        uncancelled = run_experiment(
            SlowCooperativeBackend(),
            (FMT,),
            split,
            boundary=BOUNDARY,
            max_new_tokens=MAX_NEW_TOKENS,
            cancel=Event(),
        )
        uncancelled_elapsed = time.perf_counter() - start

        # Sanity: the fake really slept (guards a vacuous fast fake making
        # the ratio below meaningless). Sleeps can only overshoot, so a
        # 0.9 factor is safe even on slow CI machines.
        self.assertGreaterEqual(uncancelled_elapsed, 0.9 * nominal)
        self.assertEqual(uncancelled["decision"], "compatible_on_fixture_set")

        cancel = Event()
        timer = threading.Timer(0.5, cancel.set)
        timer.start()
        try:
            start = time.perf_counter()
            cancelled = run_experiment(
                SlowCooperativeBackend(),
                (FMT,),
                make_split(1, 1),
                boundary=BOUNDARY,
                max_new_tokens=MAX_NEW_TOKENS,
                cancel=cancel,
            )
            cancelled_elapsed = time.perf_counter() - start
        finally:
            timer.cancel()

        self.assertEqual(cancelled["decision"], "cancelled")
        # Well before the uncancelled runtime: generous 0.5 ratio bound.
        self.assertLess(cancelled_elapsed, 0.5 * uncancelled_elapsed)
        # The in-flight generate was interrupted: its observation is kept
        # with the "cancelled" status, not dropped and not completed.
        observations = cancelled["search"][0]["observations"]
        self.assertTrue(observations)
        self.assertEqual(observations[-1]["status"], "cancelled")

    def test_cancelled_decision_shape(self) -> None:
        cancel = Event()
        cancel.set()  # pre-set: the run must not do any real work
        result = run_experiment(
            SlowCooperativeBackend(),
            (FMT,),
            make_split(1, 1),
            boundary=BOUNDARY,
            max_new_tokens=MAX_NEW_TOKENS,
            cancel=cancel,
        )
        self.assertEqual(result["decision"], "cancelled")
        self.assertIsNone(result["selected_candidate"])
        self.assertIsNone(result["held_out"])
        # finish() still ran: the failure taxonomy is attached ...
        self.assertIn("failure_taxonomy", result)
        # ... and with a pre-set event no work was done at all: the search
        # entry holds zero observations (evaluate aborts at its first
        # per-case checkpoint, before generate is ever called).
        self.assertEqual(result["search"][0]["observations"], [])

    def test_evaluate_preset_cancel_never_generates(self) -> None:
        backend = SlowCooperativeBackend()
        cancel = Event()
        cancel.set()
        start = time.perf_counter()
        report = evaluate(
            backend,
            FMT,
            make_split(1, 1).search,
            boundary=BOUNDARY,
            max_new_tokens=MAX_NEW_TOKENS,
            cancel=cancel,
        )
        elapsed = time.perf_counter() - start
        self.assertEqual(report.status, "cancelled")
        # No 5s generate ran: the per-case checkpoint fired immediately.
        self.assertLess(elapsed, GEN_SECONDS)


class LogprobDiagnosticCancelTest(unittest.TestCase):
    """The cycle-90 bug: logprob_diagnostic ignored cancellation entirely."""

    def test_preset_cancel_scores_nothing(self) -> None:
        backend = SlowLogprobBackend(forward_seconds=0.25)
        cancel = Event()
        cancel.set()
        result, fixtures = make_diag_inputs(3)
        start = time.perf_counter()
        diag = logprob_diagnostic(backend, result, fixtures, cancel=cancel)
        elapsed = time.perf_counter() - start
        self.assertEqual(backend.forward_calls, 0)
        self.assertLess(elapsed, 0.25)  # far below the 0.75s uncancelled cost
        self.assertFalse(diag["available"])
        self.assertEqual(diag["detail"], "cancelled")
        self.assertEqual(diag["candidates"], [])

    def test_mid_run_cancel_stops_at_next_fixture(self) -> None:
        # Deterministic: the backend sets the event during its first forward.
        cancel = Event()
        backend = SlowLogprobBackend(forward_seconds=0.25, cancel=cancel)
        result, fixtures = make_diag_inputs(3)
        diag = logprob_diagnostic(backend, result, fixtures, cancel=cancel)
        # Exactly one forward ran; the second fixture's check saw the event.
        self.assertEqual(backend.forward_calls, 1)
        self.assertFalse(diag["available"])
        self.assertEqual(diag["detail"], "cancelled")
        # The in-progress candidate is dropped, never partially aggregated.
        self.assertEqual(diag["candidates"], [])

    def test_no_cancel_runs_to_completion_unchanged(self) -> None:
        backend = SlowLogprobBackend(forward_seconds=0.05)
        result, fixtures = make_diag_inputs(3)
        diag = logprob_diagnostic(backend, result, fixtures)
        self.assertTrue(diag["available"])
        self.assertEqual(backend.forward_calls, 3)
        self.assertEqual(len(diag["candidates"]), 1)
        self.assertEqual(len(diag["candidates"][0]["fixture_scores"]), 3)


class SensitivitySweepDrainTest(unittest.TestCase):
    """The sweep forwards cancel to every nested run_experiment."""

    def test_preset_cancel_drains_promptly(self) -> None:
        cancel = Event()
        cancel.set()
        start = time.perf_counter()
        sweep = prompt_sensitivity_sweep(
            SlowCooperativeBackend(),
            (FMT,),
            make_split(1, 1),
            boundary=BOUNDARY,
            max_drop_bytes=2,
            max_new_tokens=MAX_NEW_TOKENS,
            cancel=cancel,
        )
        elapsed = time.perf_counter() - start
        self.assertEqual(sweep["native_decision"], "cancelled")
        # Native + 2 depths, each aborting at the first evaluate checkpoint.
        self.assertLess(elapsed, 5.0)


class DeliberateExclusionsTest(unittest.TestCase):
    """These run no model-compute loops, so they take no cancel event by
    design. Pinned so any future change to that boundary is conscious."""

    def test_no_cancel_plumbing_by_design(self) -> None:
        for func in (
            run_debug,
            doctor,
            template_conformance,
            stop_matrix,
            budget_frontier,
            divergence_section_data,
        ):
            with self.subTest(func=func.__name__):
                self.assertNotIn("cancel", inspect.signature(func).parameters)


if __name__ == "__main__":
    unittest.main()
