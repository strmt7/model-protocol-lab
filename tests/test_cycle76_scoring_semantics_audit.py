"""Cycle 76: adversarial audit of the logprob-scoring math in scoring.py.

Completes an audit started by a worker that confirmed a genuine bug by
direct experiment but ended before writing anything: ``_f32`` raised
``OverflowError`` for finite float64 log-probabilities outside the
float32 range (e.g. -6.8e38, which arises legitimately from fp32-range
logits via max-shift log-sum-exp) instead of saturating to +/-inf like
torch/numpy float32 conversion. Through ``score_target_logprobs`` the
``OverflowError`` collapsed to ``unavailable``/``forward_failed``,
mislabeling a computable score. Fixed in ``model_protocol_lab/scoring.py``
(``_f32`` now saturates per IEEE-754 round-to-nearest-even).

This file pins the fix and differentially audits the scoring math:

* ``_f32`` saturation: boundary values around FLT_MAX, infinities, NaN,
  and agreement with torch float32 conversion across magnitudes.
* End-to-end: a fake backend emitting fp32-range logits scores
  ``ok`` with ``-inf`` per-token logprobs (pre-fix: ``forward_failed``).
* Independent oracle: plain-Python log-softmax (explicit loops,
  ``math.log``/``math.exp``, independently coded float32 rounding) vs
  ``score_span_from_logits`` on seeded random logits -- exact equality.
* Minimal pairs: one-logit and one-target-id flips move the score the
  expected way; span off-by-one is rejected.
* Fail-closed: non-finite logits, shape/type rejections, torch absent,
  backend without ``forward_logits``, decode/encode failures -- fixed
  codes only, never tracebacks or silent NaNs.

Local-only: no model downloads, no network. torch is exercised (it is
installed in the venv); the torch-absent path is tested by blocking the
import.
"""

from __future__ import annotations

import math
import random
import struct
import sys
import unittest
from threading import Event

import torch

from model_protocol_lab.core import Generation
from model_protocol_lab.scoring import (
    _FLT_MAX,
    STATUS_OK,
    STATUS_UNAVAILABLE,
    _f32,
    check_alignment,
    score_span_from_logits,
    score_target_logprobs,
)


def _oracle_f32(value: float) -> float:
    """Independent float32 rounding, coded from the spec (not the source).

    Round-half-even via struct; finite out-of-range magnitudes saturate to
    +/-inf. Written without reference to scoring._f32's implementation.
    """
    try:
        return float(struct.unpack("<f", struct.pack("<f", value))[0])
    except OverflowError:
        return math.copysign(math.inf, value)


def _oracle_logprobs(
    rows: list[list[float]], prompt_len: int, target_ids: tuple[int, ...]
) -> tuple[tuple[float, ...], float, float | None]:
    """Plain-Python causal-shifted log-softmax with explicit loops."""
    per_token: list[float] = []
    total = 0.0
    for offset, token_id in enumerate(target_ids):
        row = rows[prompt_len - 1 + offset]
        peak = row[0]
        for x in row[1:]:
            if x > peak:
                peak = x
        normalizer = 0.0
        for x in row:
            normalizer += math.exp(x - peak)
        logprob = row[token_id] - peak - math.log(normalizer)
        per_token.append(_oracle_f32(logprob))
        total += logprob
    mean = _oracle_f32(total / len(target_ids)) if target_ids else None
    return tuple(per_token), _oracle_f32(total), mean


def _rand_logits(prompt_len: int, n_targets: int, vocab: int, seed: str) -> torch.Tensor:
    rng = random.Random(seed)  # noqa: S311 -- deterministic string seed
    rows = [[rng.uniform(-8.0, 8.0) for _ in range(vocab)] for _ in range(prompt_len + n_targets)]
    return torch.tensor(rows, dtype=torch.float32)


class CharBackend:
    """Tiny char-per-token codec; alignment round-trips by construction."""

    fingerprint = "char-scoring-backend"
    vocab_size = 4

    def __init__(self, logits: torch.Tensor) -> None:
        self._logits = logits

    @staticmethod
    def decode(ids: tuple[int, ...]) -> str:
        return "".join(chr(65 + i) for i in ids)

    @staticmethod
    def encode(text: str) -> tuple[int, ...]:
        return tuple(ord(c) - 65 for c in text)

    @staticmethod
    def generate(
        input_ids: tuple[int, ...],
        *,
        max_new_tokens: int,
        seed: int,
        cancel: Event,
    ) -> Generation:
        raise NotImplementedError("scoring audit never generates")

    def forward_logits(self, input_ids: tuple[int, ...]) -> torch.Tensor:
        return self._logits


class TestF32Saturation(unittest.TestCase):
    """Regression tests for the cycle-76 _f32 overflow bug."""

    def test_negative_overflow_saturates_to_neg_inf(self) -> None:
        # The exact reported case: finite float64 from fp32-range logits.
        result = _f32(-6.8e38)
        self.assertTrue(math.isinf(result) and result < 0)

    def test_positive_overflow_saturates_to_pos_inf(self) -> None:
        result = _f32(6.8e38)
        self.assertTrue(math.isinf(result) and result > 0)

    def test_just_above_flt_max_saturates(self) -> None:
        above = math.nextafter(_FLT_MAX, math.inf)
        self.assertEqual(_f32(above), math.inf)
        self.assertEqual(_f32(-above), -math.inf)

    def test_flt_max_itself_stays_finite(self) -> None:
        self.assertEqual(_f32(_FLT_MAX), _FLT_MAX)
        self.assertEqual(_f32(-_FLT_MAX), -_FLT_MAX)

    def test_infinities_pass_through(self) -> None:
        self.assertEqual(_f32(math.inf), math.inf)
        self.assertEqual(_f32(-math.inf), -math.inf)

    def test_nan_preserved(self) -> None:
        self.assertTrue(math.isnan(_f32(math.nan)))

    def test_ordinary_values_unchanged(self) -> None:
        for v in (0.0, 0.1, -2.5, 1e30, -1e-30, 3.14159):
            with self.subTest(value=v):
                self.assertEqual(_f32(v), struct.unpack("<f", struct.pack("<f", v))[0])

    def test_idempotent(self) -> None:
        for v in (-6.8e38, 6.8e38, _FLT_MAX, 0.1, float("nan")):
            with self.subTest(value=v):
                once = _f32(v)
                twice = _f32(once)
                if math.isnan(once):
                    self.assertTrue(math.isnan(twice))
                else:
                    self.assertEqual(twice, once)


class TestF32MatchesTorch(unittest.TestCase):
    """Differential: _f32 agrees with torch float32 conversion."""

    def test_agreement_across_magnitudes(self) -> None:
        rng = random.Random("cycle76-f32-torch")  # noqa: S311 -- deterministic
        values = [rng.uniform(-1e3, 1e3) for _ in range(50)]
        values += [
            -6.8e38,
            6.8e38,
            1e300,
            -1e300,
            1e39,
            -1e39,
            _FLT_MAX,
            -_FLT_MAX,
            0.0,
            math.inf,
            -math.inf,
        ]
        for v in values:
            with self.subTest(value=v):
                got = _f32(v)
                want = torch.tensor(v).float().item()
                if math.isnan(v):
                    continue
                if math.isnan(want):
                    self.assertTrue(math.isnan(got))
                else:
                    self.assertEqual(got, want)


class TestEndToEndOverflowScoring(unittest.TestCase):
    """The bug's pipeline impact: fp32-range logits must score ok."""

    @staticmethod
    def _extreme_backend() -> CharBackend:
        # Row 1 (the single target row) has fp32-range logits whose true
        # logprob for token 1 is ~-6.8e38: finite float64, out of f32 range.
        logits = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [3.4e38, -3.4e38, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        assert bool(torch.isfinite(logits).all().item())
        return CharBackend(logits)

    def test_extreme_logits_score_ok_with_neg_inf(self) -> None:
        # Target id 1 selects the -3.4e38 logit: true logprob ~ -6.8e38.
        score = score_target_logprobs(self._extreme_backend(), (0, 1), (1,))
        self.assertEqual(score.status, STATUS_OK)
        self.assertIsNone(score.reason)
        self.assertEqual(score.per_token_logprobs, (float("-inf"),))
        self.assertEqual(score.total_logprob, float("-inf"))
        self.assertEqual(score.mean_logprob, float("-inf"))
        self.assertEqual(score.scored_token_count, 1)

    def test_moderate_logits_unaffected(self) -> None:
        backend = CharBackend(torch.zeros(3, 4, dtype=torch.float32))
        score = score_target_logprobs(backend, (0, 1), (1,))
        self.assertEqual(score.status, STATUS_OK)
        # uniform row: logprob = -log(4) per token, f32-rounded
        want = _f32(-math.log(4.0))
        self.assertEqual(score.per_token_logprobs, (want,))
        self.assertEqual(score.total_logprob, want)


class TestScoringOracle(unittest.TestCase):
    """Independent plain-Python oracle vs score_span_from_logits."""

    def test_oracle_agreement_seeded(self) -> None:
        for seed, prompt_len, n_targets, vocab in (
            ("cycle76-a", 1, 1, 4),
            ("cycle76-b", 2, 3, 6),
            ("cycle76-c", 3, 2, 3),
        ):
            with self.subTest(seed=seed):
                logits = _rand_logits(prompt_len, n_targets, vocab, seed)
                target_ids = tuple(i % vocab for i in range(n_targets))
                got = score_span_from_logits(logits, prompt_len=prompt_len, target_ids=target_ids)
                rows = logits.to(torch.float64).tolist()
                want = _oracle_logprobs(rows, prompt_len, target_ids)
                self.assertEqual(got[0], want[0])
                self.assertEqual(got[1], want[1])
                self.assertEqual(got[2], want[2])

    def test_single_token_target(self) -> None:
        logits = _rand_logits(1, 1, 4, "cycle76-single")
        got = score_span_from_logits(logits, prompt_len=1, target_ids=(2,))
        rows = logits.to(torch.float64).tolist()
        want = _oracle_logprobs(rows, 1, (2,))
        self.assertEqual(got, want)

    def test_one_logit_flip_moves_score(self) -> None:
        logits = _rand_logits(2, 2, 4, "cycle76-flip")
        base = score_span_from_logits(logits, prompt_len=2, target_ids=(0, 1))
        flipped = logits.clone()
        flipped[1, 0] += 1.0
        moved = score_span_from_logits(flipped, prompt_len=2, target_ids=(0, 1))
        self.assertNotEqual(base[0], moved[0])
        # Flipping a prompt-only row leaves target logprobs untouched.
        flipped_prompt = logits.clone()
        flipped_prompt[0, 0] += 1.0
        same = score_span_from_logits(flipped_prompt, prompt_len=2, target_ids=(0, 1))
        self.assertEqual(base[0], same[0])

    def test_target_id_selects_gathered_token(self) -> None:
        logits = _rand_logits(1, 2, 4, "cycle76-gather")
        a = score_span_from_logits(logits, prompt_len=1, target_ids=(0, 0))
        b = score_span_from_logits(logits, prompt_len=1, target_ids=(3, 3))
        self.assertNotEqual(a[0], b[0])


class TestScoringRejections(unittest.TestCase):
    """Fixed rejection codes for malformed math inputs."""

    def test_non_finite_logits_rejected(self) -> None:
        logits = torch.zeros(2, 4, dtype=torch.float32)
        logits[0, 0] = float("inf")
        with self.assertRaises(ValueError) as ctx:
            score_span_from_logits(logits, prompt_len=1, target_ids=(0,))
        self.assertEqual(ctx.exception.args[0], "non_finite_logits")

    def test_nan_logits_rejected(self) -> None:
        logits = torch.zeros(2, 4, dtype=torch.float32)
        logits[1, 2] = float("nan")
        with self.assertRaises(ValueError) as ctx:
            score_span_from_logits(logits, prompt_len=1, target_ids=(0,))
        self.assertEqual(ctx.exception.args[0], "non_finite_logits")

    def test_length_mismatch_rejected(self) -> None:
        logits = torch.zeros(5, 4, dtype=torch.float32)
        with self.assertRaises(ValueError) as ctx:
            score_span_from_logits(logits, prompt_len=1, target_ids=(0,))
        self.assertEqual(ctx.exception.args[0], "logits_length_mismatch")

    def test_non_2d_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            score_span_from_logits(
                torch.zeros(4, dtype=torch.float32), prompt_len=1, target_ids=(0,)
            )
        self.assertEqual(ctx.exception.args[0], "logits_must_be_2d")

    def test_non_tensor_rejected(self) -> None:
        with self.assertRaises(TypeError) as ctx:
            score_span_from_logits([[0.0]], prompt_len=1, target_ids=(0,))  # type: ignore[arg-type]
        self.assertEqual(ctx.exception.args[0], "logits_must_be_tensor")

    def test_torch_absent_raises_fixed_code(self) -> None:
        logits = torch.zeros(2, 4, dtype=torch.float32)
        saved = sys.modules.get("torch")
        sys.modules["torch"] = None  # type: ignore[assignment]
        try:
            with self.assertRaises(RuntimeError) as ctx:
                score_span_from_logits(logits, prompt_len=1, target_ids=(0,))
        finally:
            if saved is not None:
                sys.modules["torch"] = saved
            else:
                del sys.modules["torch"]
        self.assertEqual(ctx.exception.args[0], "torch_not_installed")


class ExplodingBackend(CharBackend):
    @staticmethod
    def decode(ids: tuple[int, ...]) -> str:
        raise RuntimeError("decode exploded")


class NoLogitsBackend(CharBackend):
    forward_logits = None  # type: ignore[assignment]


class TestFailClosedPipeline(unittest.TestCase):
    """score_target_logprobs never raises and never guesses."""

    def test_backend_without_forward_logits(self) -> None:
        score = score_target_logprobs(NoLogitsBackend(torch.zeros(3, 4)), (0, 1), (1,))
        self.assertEqual(score.status, STATUS_UNAVAILABLE)
        self.assertEqual(score.reason, "logits_not_supported")

    def test_decode_failure_maps_to_fixed_code(self) -> None:
        score = score_target_logprobs(ExplodingBackend(torch.zeros(3, 4)), (0, 1), (1,))
        self.assertEqual(score.status, STATUS_UNAVAILABLE)
        self.assertEqual(score.reason, "decode_failed")

    def test_provider_exception_maps_to_fixed_code(self) -> None:
        class BadForward(CharBackend):
            @staticmethod
            def forward_logits(input_ids: tuple[int, ...]) -> torch.Tensor:
                # Whitelisted provider-vocabulary codes pass through;
                # anything else collapses (see next test).
                raise ValueError("non_finite_logits")

        score = score_target_logprobs(BadForward(torch.zeros(3, 4)), (0, 1), (1,))
        self.assertEqual(score.status, STATUS_UNAVAILABLE)
        self.assertEqual(score.reason, "non_finite_logits")

    def test_provider_arbitrary_exception_collapses(self) -> None:
        class BadForward(CharBackend):
            @staticmethod
            def forward_logits(input_ids: tuple[int, ...]) -> torch.Tensor:
                raise RuntimeError("something unprintable \x00")

        score = score_target_logprobs(BadForward(torch.zeros(3, 4)), (0, 1), (1,))
        self.assertEqual(score.status, STATUS_UNAVAILABLE)
        self.assertEqual(score.reason, "forward_failed")

    def test_empty_target_scores_ok_empty(self) -> None:
        backend = CharBackend(torch.zeros(2, 4))
        score = score_target_logprobs(backend, (0, 1), ())
        self.assertEqual(score.status, "ok_empty")
        self.assertEqual(score.total_logprob, 0.0)
        self.assertIsNone(score.mean_logprob)

    def test_alignment_ok_for_conforming_backend(self) -> None:
        backend = CharBackend(torch.zeros(3, 4))
        check = check_alignment(backend, (0, 1), (2,))
        self.assertTrue(check.ok)
        self.assertIsNone(check.reason)
        self.assertEqual((check.prompt_text, check.target_text), ("AB", "C"))


if __name__ == "__main__":
    unittest.main()
