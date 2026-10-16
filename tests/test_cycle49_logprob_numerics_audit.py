"""Cycle 49: differential numerical audit of the logprob scoring math.

SCOPE
----
Numerically differentially verifies ``score_span_from_logits`` and
``score_target_logprobs`` in ``model_protocol_lab/scoring.py`` against an
independent pure-Python oracle (``math.log``/``math.exp`` only, no torch).
The teacher-forced logprob diagnostic (``experiments.logprob_diagnostic``)
delegates its math to ``score_target_logprobs``, so it shares the verified
path; its candidate-level aggregation (plain summation in
``_aggregate_fixture_scores``) is out of scope for this log-softmax audit.

DOCUMENTED CONTRACT UNDER TEST (re-derived from the module docstring)
--------------------------------------------------------------------
``score_span_from_logits(logits, prompt_len, target_ids)`` with a ``(T, V)``
logit tensor, ``T == prompt_len + len(target_ids)``:

* for target token ``i`` read row ``prompt_len - 1 + i`` (causal shift);
* log-softmax with the max-subtraction trick in float64:
  ``logprob = row[tid] - peak - log(sum(exp(row - peak)))``;
* accumulate the total in float64 over the *unrounded* per-token logprobs;
* report per-token values, the total, and ``total / n`` (``None`` when
  ``n == 0``), each float32-rounded; empty target -> ``((), 0.0, None)``.

METHOD
------
* Independent oracle below: pure Python, ``math.fsum`` for the most accurate
  reference summation, ``struct`` round-trip for float32 rounding.
* 300 seeded fuzz configurations: vocab in {2,3,5,16,50,128,257,512},
  prompt_len 1..8, target_len 0..64 (incl. empty), fp32 and fp64 inputs,
  logit scales {1, 5, 30}, plus spiky rows with dynamic range 60.
* Tolerances per comparison: ``|diff| <= max(2e-5, 4*ulp32(|expected|))``
  (covers a 1-ulp32 rounding-boundary flip between two float64 paths that
  agree to ~1e-15) and ``rel <= 1e-6`` wherever ``|expected| >= 1e-3``.
* Extra contract pins: every reported value is exactly float32-representable,
  nothing is NaN/inf, and ``fsum(per_token)`` agrees with the reported total
  within ``(n+1)`` ulp32 (sum-of-positions consistency).

MUTATION GATE
-------------
Six numerical mutants were applied to a /tmp copy of the package (never the
repo) and this suite was run against each: total-as-mean, off-by-one gather
row, log2 instead of ln in the normalizer, dropping the last target position,
skipping the per-token float32 rounding, and a mean denominator of n+1.
All six were caught (suite fails); the pristine /tmp copy passes.

NUMERICAL AUDIT RESULT (measured 2026-10-15, torch CPU, float64 engine path)
----------------------------------------------------------------------
Over the full sweep (6145 field comparisons: per-token logprobs, totals,
means across 300 fuzz configs + directed + end-to-end cases):

* max abs divergence REPO vs ORACLE: 0.0
* max rel divergence REPO vs ORACLE: 0.0

i.e. the engine's float64 log-softmax/gather/accumulate path agrees with the
independent pure-Python oracle bit-for-bit after float32 rounding on every
configuration, including extreme +-30 logits (dynamic range 60). No real
numerical discrepancy exists; there is nothing to document as REPO -> ORACLE.
Pinned tolerances (abs <= max(2e-5, 4*ulp32), rel <= 1e-6) hold with huge
margin and are enforced by ``test_zzz_divergence_report``.

RUNNER NOTE (cycle 106)
-----------------------
Converted from pytest-style module functions to ``unittest.TestCase`` so the
project's canonical CI gate (``python -m unittest discover``) executes this
suite; pytest runs ``TestCase`` classes natively, so both runners now cover
it. ``unittest`` runs methods alphabetically, while the divergence report
test depends on the fuzz tests having populated ``_DIVERGENCE`` first; it is
therefore named ``test_zzz_divergence_report`` to sort last, with the reason
documented here rather than left implicit.
"""

from __future__ import annotations

import math
import random
import struct
import unittest

import torch

from model_protocol_lab.scoring import score_span_from_logits, score_target_logprobs

# ---------------------------------------------------------------------------
# Independent pure-Python oracle (no torch).
# ---------------------------------------------------------------------------


def _f32_round(value: float) -> float:
    """Round to float32 exactly the way the engine's ``_f32`` does."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _ulp32(value: float) -> float:
    """One ulp of the float32 binade containing ``value``."""
    ax = abs(value)
    if ax == 0.0:
        return 2.0**-149
    if not math.isfinite(ax):
        return math.inf
    return 2.0 ** (math.floor(math.log2(ax)) - 23)


def oracle_score_span(
    rows: list[list[float]], prompt_len: int, target_ids: tuple[int, ...]
) -> tuple[tuple[float, ...], float, float | None]:
    """Reference implementation of the documented scoring contract.

    Written from the docstring, not from the implementation: causal-shifted
    row selection, log-softmax via the max-subtraction trick, float64
    accumulation of unrounded logprobs, float32-rounded reports.
    """
    per_token: list[float] = []
    total = 0.0
    for offset, token_id in enumerate(target_ids):
        row = rows[prompt_len - 1 + offset]
        peak = max(row)
        log_normalizer = math.log(math.fsum(math.exp(v - peak) for v in row))
        logprob = (row[token_id] - peak) - log_normalizer
        per_token.append(_f32_round(logprob))
        total += logprob
    mean = _f32_round(total / len(target_ids)) if target_ids else None
    return tuple(per_token), _f32_round(total), mean


# ---------------------------------------------------------------------------
# Divergence tracking and comparison helpers.
# ---------------------------------------------------------------------------

ABS_TOL = 2e-5
REL_TOL = 1e-6
_REL_FLOOR = 1e-3  # rel checked only where |expected| is safely nonzero

_DIVERGENCE = {"abs": 0.0, "rel": 0.0, "cases": 0}


def _record(expected: float, got: float) -> None:
    diff = abs(got - expected)
    denom = abs(expected)
    rel = diff / denom if denom >= _REL_FLOOR else diff / _REL_FLOOR
    if diff > _DIVERGENCE["abs"]:
        _DIVERGENCE["abs"] = diff
    if rel > _DIVERGENCE["rel"]:
        _DIVERGENCE["rel"] = rel
    _DIVERGENCE["cases"] += 1


def _assert_close(expected: float, got: float, *, what: str) -> None:
    """Pin the tight tolerance: 1-ulp32-boundary-safe abs, 1e-6 rel."""
    diff = abs(got - expected)
    bound = max(ABS_TOL, 4.0 * _ulp32(expected))
    assert diff <= bound, f"{what}: |{got!r} - {expected!r}| = {diff:.3e} > {bound:.3e}"
    if abs(expected) >= _REL_FLOOR:
        rel = diff / abs(expected)
        assert rel <= REL_TOL, f"{what}: rel {rel:.3e} > {REL_TOL:.0e}"
    _record(expected, got)


def _assert_score_tuple(
    expected: tuple[tuple[float, ...], float, float | None],
    got: tuple[tuple[float, ...], float, float | None],
    *,
    label: str,
) -> None:
    exp_per, exp_total, exp_mean = expected
    got_per, got_total, got_mean = got
    assert len(got_per) == len(exp_per), f"{label}: per-token length mismatch"
    for i, (e, g) in enumerate(zip(exp_per, got_per, strict=True)):
        _assert_close(e, g, what=f"{label} per_token[{i}]")
        assert g == _f32_round(g), f"{label} per_token[{i}] not float32-rounded"
        assert math.isfinite(g), f"{label} per_token[{i}] not finite"
    _assert_close(exp_total, got_total, what=f"{label} total")
    assert got_total == _f32_round(got_total), f"{label} total not float32-rounded"
    assert math.isfinite(got_total), f"{label} total not finite"
    assert (got_mean is None) == (exp_mean is None), f"{label} mean None-ness mismatch"
    if exp_mean is not None and got_mean is not None:
        _assert_close(exp_mean, got_mean, what=f"{label} mean")
        assert got_mean == _f32_round(got_mean), f"{label} mean not float32-rounded"
        assert math.isfinite(got_mean), f"{label} mean not finite"
    # Sum-of-positions consistency: fsum of the rounded per-token values must
    # agree with the reported (float64-accumulated, then rounded) total.
    if got_per:
        spread = math.fsum(got_per)
        bound = (len(got_per) + 1) * _ulp32(max(abs(got_total), 1.0)) + 1e-12
        assert abs(spread - got_total) <= bound, (
            f"{label}: sum(per_token)={spread!r} vs total={got_total!r}"
        )


# ---------------------------------------------------------------------------
# Fuzz-case generation (deterministic, version-independent: Python random only).
# ---------------------------------------------------------------------------

_VOCABS = (2, 3, 5, 16, 50, 128, 257, 512)
_TARGET_LENS = (0, 1, 2, 3, 5, 8, 13, 16, 32, 48, 64)


def _fuzz_case(rng: random.Random, *, spiky: bool) -> tuple[torch.Tensor, int, tuple[int, ...]]:
    vocab = rng.choice(_VOCABS)
    prompt_len = rng.randint(1, 8)
    target_len = rng.choice(_TARGET_LENS)
    dtype = rng.choice((torch.float32, torch.float64))
    scale = rng.choice((1.0, 5.0, 30.0))
    rows: list[list[float]] = []
    for _ in range(prompt_len + target_len):
        if spiky and rng.random() < 0.5:
            spike = rng.randrange(vocab)
            rows.append([-scale if j != spike else scale for j in range(vocab)])
        else:
            rows.append([rng.uniform(-scale, scale) for _ in range(vocab)])
    target_ids = tuple(rng.randrange(vocab) for _ in range(target_len))
    return torch.tensor(rows, dtype=dtype), prompt_len, target_ids


def _spike_rows(vocab: int, spikes: dict[int, int], scale: float = 30.0) -> torch.Tensor:
    rows = []
    n = max(spikes) + 1
    for r in range(n):
        spike = spikes.get(r, vocab - 1)
        rows.append([-scale if j != spike else scale for j in range(vocab)])
    return torch.tensor(rows, dtype=torch.float32)


class _FakeBackend:
    """Minimal backend: chr/ord codec with an exact roundtrip, fixed logits."""

    def __init__(self, vocab_size: int, logits: torch.Tensor) -> None:
        self.vocab_size = vocab_size
        self._logits = logits

    def encode(self, text: str) -> tuple[int, ...]:
        return tuple(ord(c) for c in text)

    def decode(self, ids: tuple[int, ...]) -> str:
        return "".join(chr(i) for i in ids)

    def forward_logits(self, input_ids: tuple[int, ...]) -> torch.Tensor:
        assert len(input_ids) == self._logits.shape[0]
        return self._logits


class LogprobNumericsAuditTests(unittest.TestCase):
    """unittest entry point; pytest runs TestCase classes natively."""

    def test_score_span_matches_oracle_fuzz(self) -> None:
        """300 seeded configs: repo vs oracle on per-token/total/mean."""
        for seed in range(6):
            with self.subTest(seed=seed):
                rng = random.Random(20261015 + seed)  # noqa: S311 -- deterministic test seed
                for case in range(50):
                    logits, prompt_len, target_ids = _fuzz_case(rng, spiky=(case % 4 == 0))
                    got = score_span_from_logits(
                        logits, prompt_len=prompt_len, target_ids=target_ids
                    )
                    rows64 = logits.to(torch.float64).tolist()
                    expected = oracle_score_span(rows64, prompt_len, target_ids)
                    _assert_score_tuple(
                        expected, got, label=f"seed={seed} case={case} V={logits.shape[1]}"
                    )

    def test_causal_shift_direction_spike(self) -> None:
        """Rows predicting each target spike at that target: logprobs ~ 0.

        Oracle-independent pin of the shift direction: with row
        ``prompt_len - 1 + i`` spiking at ``target[i]``, every per-token logprob
        must be ~0. A shift error in either direction would read ~-60 instead.
        """
        vocab, prompt_len = 8, 2
        target_ids = (3, 5, 1)
        logits = _spike_rows(vocab, {1: 3, 2: 5, 3: 1, 0: 7, 4: 7})
        per_token, total, mean = score_span_from_logits(
            logits, prompt_len=prompt_len, target_ids=target_ids
        )
        self.assertEqual(len(per_token), 3)
        for i, lp in enumerate(per_token):
            self.assertGreater(lp, -1e-6, f"position {i}: logprob {lp} shows wrong gather row")
        expected = oracle_score_span(logits.to(torch.float64).tolist(), prompt_len, target_ids)
        _assert_score_tuple(expected, (per_token, total, mean), label="spike")

    def test_uniform_logits_hand_computed(self) -> None:
        """All-zero rows: every logprob is exactly -log(V)."""
        for vocab in (2, 7, 512):
            with self.subTest(vocab=vocab):
                prompt_len, target_len = 3, 4
                logits = torch.zeros(prompt_len + target_len, vocab, dtype=torch.float32)
                target_ids = tuple(i % vocab for i in range(target_len))
                per_token, total, mean = score_span_from_logits(
                    logits, prompt_len=prompt_len, target_ids=target_ids
                )
                expect = _f32_round(-math.log(vocab))
                for lp in per_token:
                    self.assertEqual(lp, expect, f"V={vocab}: {lp!r} != -log(V)={expect!r}")
                _assert_score_tuple(
                    oracle_score_span(logits.to(torch.float64).tolist(), prompt_len, target_ids),
                    (per_token, total, mean),
                    label=f"uniform V={vocab}",
                )

    def test_extreme_logits_pm30(self) -> None:
        """Stress float precision: +-30 logits, dynamic range 60, V=512."""
        rng = random.Random(99)  # noqa: S311 -- deterministic test seed
        vocab, prompt_len, target_len = 512, 4, 61
        rows = [
            [30.0 if rng.random() < 0.5 else -30.0 for _ in range(vocab)]
            for _ in range(prompt_len + target_len)
        ]
        # One row with a single +30 spike among -30s (hardest normalizer).
        rows[prompt_len] = [-30.0] * vocab
        rows[prompt_len][17] = 30.0
        logits = torch.tensor(rows, dtype=torch.float32)
        target_ids = tuple(rng.randrange(vocab) for _ in range(target_len))
        got = score_span_from_logits(logits, prompt_len=prompt_len, target_ids=target_ids)
        expected = oracle_score_span(logits.to(torch.float64).tolist(), prompt_len, target_ids)
        _assert_score_tuple(expected, got, label="extreme +-30")

    def test_edge_cases_empty_single_vocab1(self) -> None:
        """Documented edges: empty target, single token, single-token vocab."""
        # Empty target -> ((), 0.0, None).
        logits = torch.zeros(2, 5, dtype=torch.float32)
        self.assertEqual(
            score_span_from_logits(logits, prompt_len=2, target_ids=()), ((), 0.0, None)
        )
        # Single target token.
        logits = torch.tensor([[0.1, 0.2, 0.3], [1.0, 2.0, 3.0]], dtype=torch.float32)
        got = score_span_from_logits(logits, prompt_len=1, target_ids=(2,))
        _assert_score_tuple(
            oracle_score_span(logits.to(torch.float64).tolist(), 1, (2,)),
            got,
            label="single",
        )
        # Vocab of size 1: the only logprob is 0.
        logits = torch.zeros(3, 1, dtype=torch.float32)
        per_token, total, mean = score_span_from_logits(logits, prompt_len=1, target_ids=(0, 0))
        self.assertEqual(per_token, (0.0, 0.0))
        self.assertEqual(total, 0.0)
        self.assertEqual(mean, 0.0)

    def test_batched_slices_match_oracle(self) -> None:
        """2-D slices of a batched tensor score identically to the oracle."""
        rng = random.Random(7)  # noqa: S311 -- deterministic test seed
        batch, vocab, prompt_len, target_len = 4, 64, 3, 9
        rows = [
            [[rng.uniform(-5.0, 5.0) for _ in range(vocab)] for _ in range(prompt_len + target_len)]
            for _ in range(batch)
        ]
        stacked = torch.tensor(rows, dtype=torch.float32)
        for b in range(batch):
            with self.subTest(batch=b):
                target_ids = tuple(rng.randrange(vocab) for _ in range(target_len))
                got = score_span_from_logits(
                    stacked[b], prompt_len=prompt_len, target_ids=target_ids
                )
                expected = oracle_score_span(
                    stacked[b].to(torch.float64).tolist(), prompt_len, target_ids
                )
                _assert_score_tuple(expected, got, label=f"batch[{b}]")

    def test_documented_rejections(self) -> None:
        good = torch.zeros(4, 8, dtype=torch.float32)
        with self.assertRaisesRegex(TypeError, "logits_must_be_tensor"):
            score_span_from_logits([[0.0] * 8] * 4, prompt_len=2, target_ids=(1, 2))
        with self.assertRaisesRegex(ValueError, "logits_must_be_2d"):
            score_span_from_logits(torch.zeros(4, 8, 2), prompt_len=2, target_ids=(1, 2))
        with self.assertRaisesRegex(ValueError, "prompt_len_must_be_positive"):
            score_span_from_logits(good, prompt_len=0, target_ids=(1, 2))
        with self.assertRaisesRegex(ValueError, "logits_length_mismatch"):
            score_span_from_logits(good, prompt_len=2, target_ids=(1,))
        with self.assertRaisesRegex(ValueError, "token_id_out_of_range"):
            score_span_from_logits(good, prompt_len=2, target_ids=(1, 8))
        bad = torch.zeros(4, 8, dtype=torch.float32)
        bad[1, 3] = float("inf")
        with self.assertRaisesRegex(ValueError, "non_finite_logits"):
            score_span_from_logits(bad, prompt_len=2, target_ids=(1, 2))
        bad2 = torch.zeros(4, 8, dtype=torch.float32)
        bad2[2, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non_finite_logits"):
            score_span_from_logits(bad2, prompt_len=2, target_ids=(1, 2))

    def test_score_target_logprobs_end_to_end(self) -> None:
        """Full engine path vs oracle: fields, counts, byte length, statuses."""
        for seed in range(3):
            with self.subTest(seed=seed):
                rng = random.Random(4242 + seed)  # noqa: S311 -- deterministic test seed
                for case in range(20):
                    vocab = rng.choice((16, 128, 512))
                    prompt_len = rng.randint(1, 6)
                    target_len = rng.choice((0, 1, 2, 7, 32))
                    scale = rng.choice((2.0, 30.0))
                    rows = [
                        [rng.uniform(-scale, scale) for _ in range(vocab)]
                        for _ in range(max(prompt_len + target_len, 1))
                    ]
                    logits = torch.tensor(rows, dtype=torch.float32)
                    prompt_ids = tuple(rng.randrange(1, vocab) for _ in range(prompt_len))
                    target_ids = tuple(rng.randrange(1, vocab) for _ in range(target_len))
                    backend = _FakeBackend(vocab, logits)
                    score = score_target_logprobs(backend, prompt_ids, target_ids)
                    if not target_ids:
                        self.assertEqual(score.status, "ok_empty")
                        self.assertEqual(score.per_token_logprobs, ())
                        self.assertEqual(score.total_logprob, 0.0)
                        self.assertIsNone(score.mean_logprob)
                        self.assertEqual(score.scored_token_count, 0)
                        continue
                    self.assertEqual(
                        score.status, "ok", f"case={case}: {score.status}/{score.reason}"
                    )
                    self.assertIsNone(score.reason)
                    expected = oracle_score_span(
                        logits.to(torch.float64).tolist(), prompt_len, target_ids
                    )
                    _assert_score_tuple(
                        expected,
                        (score.per_token_logprobs, score.total_logprob, score.mean_logprob),
                        label=f"e2e seed={seed} case={case}",
                    )
                    self.assertEqual(score.scored_token_count, len(target_ids))
                    self.assertEqual(score.prompt_token_count, prompt_len)
                    self.assertEqual(
                        score.target_byte_length,
                        len(backend.decode(target_ids).encode("utf-8")),
                    )

    def test_score_target_logprobs_empty_prompt_refused(self) -> None:
        backend = _FakeBackend(8, torch.zeros(2, 8))
        score = score_target_logprobs(backend, (), (1, 2))
        self.assertEqual(score.status, "unavailable")
        self.assertEqual(score.reason, "empty_prompt")
        self.assertIsNone(score.total_logprob)

    def test_zzz_divergence_report(self) -> None:
        """Assert the observed maxima stay within the pinned tolerances.

        Named to sort last under unittest's alphabetical method ordering: it
        depends on the fuzz tests above having populated ``_DIVERGENCE``.
        """
        self.assertGreater(_DIVERGENCE["cases"], 1000, "fuzz sweep did not record comparisons")
        self.assertLessEqual(
            _DIVERGENCE["abs"], ABS_TOL, f"max abs divergence {_DIVERGENCE['abs']:.3e}"
        )
        self.assertLessEqual(
            _DIVERGENCE["rel"], REL_TOL, f"max rel divergence {_DIVERGENCE['rel']:.3e}"
        )


if __name__ == "__main__":
    unittest.main()
