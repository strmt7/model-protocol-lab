"""W4-2 (RR-14a) — the run budget ledger in ``run_experiment``.

Before W4, per-cell caps existed (``max_new_tokens``) but nothing
accounted for a run's total token spend, and nothing bounded it: a
plan is a-priori bounded (<=24 formats x <=100+100 fixtures) yet a
single run could still spend 24 x 200 x (input + 4096) tokens with
no preflight check and no record. The ledger adds both:

* an optional ``max_total_tokens`` cap, enforced at ONE preflight
  refusal point — the estimate is a mathematical upper bound on
  actual spend (every planned search cell at ``len(input) +
  max_new_tokens``; held-out cells at the per-fixture maximum input
  over candidates, since the winner is unknown at preflight), so a
  run whose estimate exceeds the cap is refused before ANY
  generation, and a run that starts can never exceed its estimate;
* a ``budget`` block on every return path recording cap, estimate
  (``None`` when no cap was set — accounting without enforcement
  does not pay for a preflight pass), actual spend recomputed from
  the stored observations, planned vs executed cells.

Expected values in these tests come from the definitions (an
independent oracle renders + encodes through the public Format /
backend surface and sums the same bounds), never from the ledger's
own arithmetic.
"""

from __future__ import annotations

import unittest

from model_protocol_lab import core
from model_protocol_lab.experiments import run_experiment

BOUNDARY = (999,)
MAX_NEW_TOKENS = 40


class ScriptedBackend:
    """Deterministic backend: ord-encode; pass text + boundary, padded."""

    fingerprint = "w4-budget-ledger"

    def __init__(self, expected, behavior=None):
        """Init."""
        self._expected = dict(expected)
        self._behavior = dict(behavior or {})
        self.generate_calls = 0
        self.encode_calls = 0

    def encode(self, text):
        """Encode."""
        self.encode_calls += 1
        return tuple(map(ord, text))

    @staticmethod
    def decode(token_ids):
        """Decode."""
        return "".join(map(chr, token_ids))

    def generate(self, input_ids, *, max_new_tokens, seed, cancel):
        """Generate."""
        self.generate_calls += 1
        prompt = self.decode(input_ids)
        candidate = next((n for n in ("cand-A", "cand-B") if f"[{n}]" in prompt), None)
        fixture = next((fid for fid in self._expected if f"question-{fid}" in prompt), None)
        if candidate is None or fixture is None:
            return core.Generation((), "error", "unknown_request")
        mode = self._behavior.get((candidate, fixture), "pass")
        expected = self._expected[fixture]
        text = expected + chr(BOUNDARY[0]) if mode == "pass" else expected + "WRONG"
        ids = tuple(map(ord, text))
        assert len(ids) <= max_new_tokens
        return core.Generation(ids + (1,) * (max_new_tokens - len(ids)), "limit")


EXPECTED = {"S1": "GREEN", "S2": "BLUE", "H1": "RED", "H2": "CYAN"}
# cand-B fails S2, so cand-A is the unique winner and held-out runs.
BEHAVIOR = {("cand-B", "S2"): "fail"}


def _split():
    """Split."""

    def fixtures(ids):
        """Fixtures."""
        return tuple(
            core.Fixture(
                id=fid,
                messages=(core.Message(role="user", content=f"question-{fid}"),),
                expected=EXPECTED[fid],
                group=f"group-{fid}",
            )
            for fid in ids
        )

    return core.Split(search=fixtures(("S1", "S2")), held_out=fixtures(("H1", "H2")))


def _formats():
    """Formats."""
    return tuple(
        core.Format(
            name=name,
            bos="",
            user_prefix=f"[{name}]U:",
            assistant_prefix="A:",
            turn_end="\n",
        )
        for name in ("cand-A", "cand-B")
    )


def _oracle_estimate(backend, formats, split) -> int:
    """The definitional upper bound on a run's token spend.

    Search cells are exact (every candidate runs every search
    fixture): ``len(encode(render)) + max_new_tokens`` each. The
    held-out stage runs only the winner, unknown at preflight, so
    each held-out fixture contributes its maximum input over
    candidates plus ``max_new_tokens``.
    """
    total = 0
    for fmt in formats:
        for case in split.search:
            total += len(backend.encode(fmt.render(case.messages))) + MAX_NEW_TOKENS
    for case in split.held_out:
        worst = max(len(backend.encode(fmt.render(case.messages))) for fmt in formats)
        total += worst + MAX_NEW_TOKENS
    return total


def _recomputed_actual(result) -> tuple[int, int]:
    """Actual spend + executed cells, from the stored observations."""
    total = 0
    cells = 0
    for entry in result["search"]:
        for obs in entry["observations"]:
            total += len(obs["input_ids"]) + len(obs["output_ids"])
            cells += 1
    held = result.get("held_out") or {}
    for obs in held.get("observations", []):
        total += len(obs["input_ids"]) + len(obs["output_ids"])
        cells += 1
    return total, cells


def _run(backend, **kwargs):
    """Run."""
    return run_experiment(
        backend,
        _formats(),
        _split(),
        boundary=BOUNDARY,
        max_new_tokens=MAX_NEW_TOKENS,
        seed=7,
        **kwargs,
    )


class LedgerShapeTests(unittest.TestCase):
    """The ledger is recorded on every return path, cap or not."""

    def test_uncapped_run_records_accounting_without_estimate(self):
        """Test uncapped run records accounting without estimate."""
        backend = ScriptedBackend(EXPECTED, BEHAVIOR)
        result = _run(backend)
        self.assertEqual(result["decision"], "compatible_on_fixture_set")
        budget = result["budget"]
        self.assertIsNone(budget["max_total_tokens"])
        self.assertIsNone(budget["estimated_total_tokens"])
        self.assertTrue(budget["estimate_includes_held_out"])
        self.assertEqual(budget["cells_planned"], 2 * 2 + 2)
        actual, cells = _recomputed_actual(result)
        self.assertEqual(budget["actual_total_tokens"], actual)
        self.assertEqual(budget["cells_executed"], cells)
        self.assertEqual(cells, 6)

    def test_early_exit_run_records_partial_ledger(self):
        """Test early exit run records partial ledger."""
        # cand-A fails a search fixture (wrong answer, no boundary is
        # not the trigger — aggregate ineligibility is): make every
        # cand-A generation an error so its report is not eligible
        # and run_experiment returns after the first candidate.
        backend = ScriptedBackend(EXPECTED)
        original_generate = backend.generate

        def failing_generate(input_ids, *, max_new_tokens, seed, cancel):
            """Failing generate."""
            if "[cand-A]" in backend.decode(input_ids):
                backend.generate_calls += 1
                return core.Generation((), "error", "unknown_request")
            return original_generate(
                input_ids, max_new_tokens=max_new_tokens, seed=seed, cancel=cancel
            )

        backend.generate = failing_generate
        result = _run(backend)
        budget = result["budget"]
        actual, cells = _recomputed_actual(result)
        self.assertEqual(budget["actual_total_tokens"], actual)
        self.assertEqual(budget["cells_executed"], cells)
        self.assertLess(cells, budget["cells_planned"])


class PreflightTests(unittest.TestCase):
    """The cap is enforced once, before any generation."""

    def test_estimate_matches_definition_and_bounds_actual(self):
        """Test estimate matches definition and bounds actual."""
        backend = ScriptedBackend(EXPECTED)
        estimate = _oracle_estimate(backend, _formats(), _split())
        result = _run(ScriptedBackend(EXPECTED, BEHAVIOR), max_total_tokens=estimate)
        budget = result["budget"]
        self.assertEqual(budget["estimated_total_tokens"], estimate)
        self.assertEqual(budget["max_total_tokens"], estimate)
        actual, _cells = _recomputed_actual(result)
        self.assertEqual(budget["actual_total_tokens"], actual)
        self.assertLessEqual(actual, estimate)

    def test_refusal_at_estimate_minus_one_zero_generations(self):
        """Test refusal at estimate minus one zero generations."""
        backend = ScriptedBackend(EXPECTED)
        estimate = _oracle_estimate(ScriptedBackend(EXPECTED), _formats(), _split())
        with self.assertRaises(ValueError) as caught:
            _run(backend, max_total_tokens=estimate - 1)
        self.assertEqual(str(caught.exception), "run_budget_exceeded_preflight")
        self.assertEqual(backend.generate_calls, 0)

    def test_preflight_encode_failure_refuses_capped_run(self):
        """Test preflight encode failure refuses capped run."""
        backend = ScriptedBackend(EXPECTED)
        original_encode = backend.encode

        def flaky_encode(text):
            """Flaky encode."""
            if "question-S2" in text:
                raise RuntimeError("encode boom")
            return original_encode(text)

        backend.encode = flaky_encode
        with self.assertRaises(ValueError) as caught:
            _run(backend, max_total_tokens=10**9)
        self.assertEqual(str(caught.exception), "run_budget_preflight_failed")
        self.assertEqual(backend.generate_calls, 0)

    def test_preflight_not_run_without_cap(self):
        """Test preflight not run without cap."""
        # The same flaky encode without a cap: no preflight pass, so
        # the run reaches evaluate, whose per-cell sanitization
        # degrades the affected cells instead of refusing the run.
        backend = ScriptedBackend(EXPECTED)
        original_encode = backend.encode

        def flaky_encode(text):
            """Flaky encode."""
            if "question-S2" in text:
                raise RuntimeError("encode boom")
            return original_encode(text)

        backend.encode = flaky_encode
        result = _run(backend)
        self.assertIsNone(result["budget"]["estimated_total_tokens"])
        self.assertGreater(backend.generate_calls, 0)

    def test_cap_validation(self):
        """Test cap validation."""
        backend = ScriptedBackend(EXPECTED)
        for bad in (0, -5, True, 1.5, "100"):
            with self.assertRaises(ValueError, msg=f"cap={bad!r}"):
                _run(backend, max_total_tokens=bad)
        self.assertEqual(backend.generate_calls, 0)


if __name__ == "__main__":
    unittest.main()
