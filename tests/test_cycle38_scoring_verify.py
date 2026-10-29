"""Cycle 38: verification of ``model_protocol_lab/scoring.py`` internals.

Scope: every helper and contract in ``scoring.py`` EXCEPT
``score_target_logprobs`` itself, which was differentially verified in
cycle 12 (and re-exercised in cycles 9/14). What is pinned here:

* ``_f32`` -- exact IEEE-754 binary32 rounding semantics (ties to even),
  idempotence, type, overflow/underflow/NaN behavior.
* ``score_span_from_logits`` -- every rejection code with its exact
  exception type, rejection precedence, boundary conditions (empty span,
  single-token vocab, prompt_len=1), determinism, masking discipline
  (prompt-predicting and past-the-span rows are never read), float64
  accumulation of *unrounded* per-token values, and hostile inputs
  (bools, numpy ints, lists, str subclasses).
* ``_safe_decode`` / ``_safe_encode`` -- sanitization boundaries: backend
  exceptions and non-conforming return values collapse to None.
* ``check_alignment`` -- all five §9.3 reason codes plus the ok path,
  with decoded texts pinned on every failure branch.
* ``_provider_code`` -- the whitelist is exact: every whitelisted code
  passes through, everything else (including str-subclass codes and
  non-string args) collapses to ``forward_failed``.
* ``_unavailable``, the three reason vocabularies (exact membership and
  mutual disjointness), the frozen dataclasses, the status constants,
  and ``__all__``.

Method: expectations are written from the docstrings and from
independent oracles (``struct``-based float32 rounding, naive
``math``-only log-softmax from the definition, never the engine's
max-shift formulation). Randomized checks use ``random.Random`` with
string seeds, so every run is deterministic.

Doc/code discrepancies found (source behavior NOT changed -- pinned
as-is, see the tests marked DISCREPANCY):
1. (CYCLE 38 NOTE, FIXED IN CYCLE 38.) ``score_span_from_logits``
   docstring used to say "Rejections raise ValueError with a fixed code",
   but a non-tensor ``logits`` argument raises
   ``TypeError("logits_must_be_tensor")``. The docstring was corrected in
   cycle 38 (wrong-type input is a caller bug, not a rejected
   measurement). The remaining rejections are ValueError as documented.
2. (Observation, not a discrepancy.) The whole-tensor finiteness check
   rejects NaN/Inf even in masked rows that are never gathered. This
   matches the module docstring ("NaN and infinity are detected before
   aggregation") and is pinned by ``test_masked_row_nan_still_rejected``.

Sensitivity: the file was validated against 12 hand-built mutants on a
copy of the package in /tmp (never the repo); every mutant is caught by
at least one test. See the cycle-38 report for the mutant list.
"""

from __future__ import annotations

import math
import random
import struct
import sys
import unittest

import numpy as np
import torch

from model_protocol_lab.scoring import (
    _REASON_ALIGNMENT,
    _REASON_ENGINE,
    _REASON_PROVIDER,
    STATUS_OK,
    STATUS_OK_EMPTY,
    STATUS_UNAVAILABLE,
    AlignmentCheck,
    TargetScore,
    _f32,
    _provider_code,
    _require_torch,
    _safe_decode,
    _safe_encode,
    _unavailable,
    check_alignment,
    score_span_from_logits,
)

# ---------------------------------------------------------------------------
# Independent oracles (written from the definitions, not from the engine).
# ---------------------------------------------------------------------------


def ref_f32(value: float) -> float:
    """The definition of binary32 rounding, via struct."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def ref_logprob_naive(row: list[float], token_id: int) -> float:
    """Naive log-softmax from the math definition: x_t - log(sum_v exp(x_v)).

    Deliberately avoids the engine's max-shift formulation. Only used with
    moderate-magnitude logits where the naive path is numerically safe.
    """
    return row[token_id] - math.log(math.fsum(math.exp(x) for x in row))


class FakeCodec:
    """Structural Backend double: scripted decode/encode with call logging."""

    def __init__(self) -> None:
        self.decode_fn = None
        self.encode_fn = None
        self.calls: list[tuple[str, object]] = []

    def decode(self, token_ids: tuple[int, ...]):
        """Decode."""
        # noqa: ANN001, ANN202
        self.calls.append(("decode", token_ids))
        assert self.decode_fn is not None, "decode not scripted"
        return self.decode_fn(token_ids)

    def encode(self, text: str):
        """Encode."""
        # noqa: ANN001, ANN202
        self.calls.append(("encode", text))
        assert self.encode_fn is not None, "encode not scripted"
        return self.encode_fn(text)


def scripted_codec(
    decode_map: dict,
    encode_map: dict,
    decode_raises: tuple = (),
    encode_raises: tuple = (),
) -> FakeCodec:
    """Build a codec from explicit maps; listed inputs raise RuntimeError."""
    codec = FakeCodec()

    def _decode(ids: tuple[int, ...]):
        """Decode."""
        if ids in decode_raises:
            raise RuntimeError("boom-decode")
        return decode_map[ids]

    def _encode(text: str):
        """Encode."""
        if text in encode_raises:
            raise RuntimeError("boom-encode")
        return encode_map[text]

    codec.decode_fn = _decode
    codec.encode_fn = _encode
    return codec


def ok_codec() -> FakeCodec:
    """A consistent tiny codec: (1,2)->'ab', (3,)->'c', 'abc'->(1,2,3)."""
    return scripted_codec(
        {(1, 2): "ab", (3,): "c"},
        {"ab": (1, 2), "c": (3,), "abc": (1, 2, 3)},
    )


# ---------------------------------------------------------------------------
# _f32 contract
# ---------------------------------------------------------------------------


class TestF32Contract(unittest.TestCase):
    """Test f32 contract."""

    def test_matches_struct_oracle_exactly(self) -> None:
        """Test matches struct oracle exactly."""
        values = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.1,
            -0.1,
            1 / 3,
            -1 / 3,
            math.pi,
            -math.pi,
            1e30,
            -1e30,
            1e-30,
            -1e-30,
            1.0000000596046448,  # near a float32 rounding boundary
            1.0000001788139343,
            123456.789,
            -987654.321,
            3.4028234663852886e38,  # FLT_MAX exactly
            1.1754943508222875e-38,  # FLT_MIN normal
            1e-45,  # subnormal territory
            float("inf"),
            float("-inf"),
        ]
        for v in values:
            with self.subTest(value=v):
                self.assertEqual(_f32(v), ref_f32(v))

    def test_nan_stays_nan(self) -> None:
        """Test nan stays nan."""
        out = _f32(float("nan"))
        self.assertTrue(math.isnan(out))
        self.assertIs(type(out), float)

    def test_returns_plain_float(self) -> None:
        """Test returns plain float."""
        self.assertIs(type(_f32(1.5)), float)
        self.assertIs(type(_f32(0.0)), float)

    def test_idempotent(self) -> None:
        """Test idempotent."""
        rng = random.Random("cycle38-f32-idempotent")  # noqa: S311 -- deterministic string seed for test fuzz
        for _ in range(200):
            v = rng.uniform(-1e6, 1e6)
            with self.subTest(value=v):
                self.assertEqual(_f32(_f32(v)), _f32(v))

    def test_round_half_to_even_at_tie(self) -> None:
        """Test round half to even at tie."""
        # 1.0 + 2**-24 is exactly representable in float64 and sits exactly
        # halfway between the float32 neighbors 1.0 and 1.0+2**-23.
        # Round-half-even picks 1.0 (even mantissa).
        tie = 1.0 + 2.0**-24
        self.assertEqual(_f32(tie), 1.0)
        self.assertEqual(_f32(-tie), -1.0)
        # 1.0 + 3*2**-24 is exactly 1.5 ulps above 1.0: a tie between the
        # odd-mantissa 1.0+2**-23 and the even-mantissa 1.0+2**-22, so
        # round-half-even picks 1.0+2**-22.
        self.assertEqual(_f32(1.0 + 3.0 * 2.0**-24), 1.0 + 2.0**-22)
        # Just below the tie rounds down.
        self.assertEqual(_f32(1.0 + 2.0**-24 - 2.0**-60), 1.0)

    def test_out_of_float32_range_saturates(self) -> None:
        """Test out of float32 range saturates."""
        # Finite float64 values outside the float32 range saturate to
        # +/-inf, matching IEEE-754 round-to-nearest-even and torch/numpy
        # float32 conversion. (struct.pack("<f") would raise OverflowError
        # here, which score_target_logprobs mislabeled as forward_failed
        # even though the score is computable -- fixed in cycle 76.)
        for v, want in (
            (1e300, math.inf),
            (-1e300, -math.inf),
            (1e39, math.inf),
            (-1e39, -math.inf),
            (6.8e38, math.inf),
            (-6.8e38, -math.inf),
        ):
            with self.subTest(value=v):
                self.assertEqual(_f32(v), want)

    def test_monotone_on_sample(self) -> None:
        """Test monotone on sample."""
        rng = random.Random("cycle38-f32-monotone")  # noqa: S311 -- deterministic string seed for test fuzz
        vals = sorted(rng.uniform(-1000, 1000) for _ in range(500))
        rounded = [_f32(v) for v in vals]
        for i in range(len(rounded) - 1):
            self.assertLessEqual(rounded[i], rounded[i + 1])


# ---------------------------------------------------------------------------
# score_span_from_logits: rejection codes and exception types
# ---------------------------------------------------------------------------


def _good_logits(prompt_len: int, target_ids: tuple[int, ...], vocab: int = 4) -> torch.Tensor:
    """Good logits."""
    t = prompt_len + len(target_ids)
    return torch.zeros(t, vocab, dtype=torch.float32)


class TestScoreSpanRejections(unittest.TestCase):
    """Test score span rejections."""

    def assert_rejects(self, code: str, exc_type: type, **kwargs) -> None:
        """Assert rejects."""
        # noqa: ANN003
        with self.assertRaises(exc_type) as ctx:
            score_span_from_logits(**kwargs)
        self.assertEqual(str(ctx.exception), code)

    def test_non_tensor_is_type_error(self) -> None:
        """Test non tensor is type error."""
        # (Was a DISCREPANCY; fixed in cycle 38: the docstring now
        # documents that wrong-type logits raise TypeError.) Pinned as-is.
        for bad in ([[0.0, 0.0]], None, "logits", 42, (0.1, 0.2)):
            with self.subTest(bad=type(bad).__name__):
                self.assert_rejects(
                    "logits_must_be_tensor",
                    TypeError,
                    logits=bad,
                    prompt_len=1,
                    target_ids=(0,),
                )

    def test_numpy_array_is_not_a_tensor(self) -> None:
        """Test numpy array is not a tensor."""
        self.assert_rejects(
            "logits_must_be_tensor",
            TypeError,
            logits=np.zeros((2, 4)),
            prompt_len=1,
            target_ids=(0,),
        )

    def test_wrong_ndim(self) -> None:
        """Test wrong ndim."""
        for shape in ((4,), (2, 4, 1), (2, 4, 1, 1)):
            with self.subTest(shape=shape):
                self.assert_rejects(
                    "logits_must_be_2d",
                    ValueError,
                    logits=torch.zeros(shape),
                    prompt_len=1,
                    target_ids=(0,),
                )

    def test_prompt_len_must_be_positive_int(self) -> None:
        """Test prompt len must be positive int."""
        bad_lens = [0, -1, -100, True, False, 1.0, 2.5, "1", None, np.int64(2)]
        for bad in bad_lens:
            with self.subTest(prompt_len=repr(bad)):
                self.assert_rejects(
                    "prompt_len_must_be_positive",
                    ValueError,
                    logits=_good_logits(2, (0, 1)),
                    prompt_len=bad,
                    target_ids=(0, 1),
                )

    def test_target_ids_must_be_exact_tuple_of_int(self) -> None:
        """Test target ids must be exact tuple of int."""
        bad_targets = [
            [0, 1],  # list
            "01",  # str
            None,
            (0.0, 1),  # float element
            (True, 1),  # bool element
            (0, np.int64(1)),  # numpy int element
            ((0,),),  # nested
        ]
        for bad in bad_targets:
            with self.subTest(target_ids=repr(bad)):
                self.assert_rejects(
                    "invalid_target_ids",
                    ValueError,
                    logits=torch.zeros(3, 4),
                    prompt_len=1,
                    target_ids=bad,
                )

    def test_tuple_subclass_rejected(self) -> None:
        """Test tuple subclass rejected."""

        class MyTuple(tuple):
            """My tuple."""

        self.assert_rejects(
            "invalid_target_ids",
            ValueError,
            logits=torch.zeros(3, 4),
            prompt_len=1,
            target_ids=MyTuple((0, 1)),
        )

    def test_empty_vocabulary(self) -> None:
        """Test empty vocabulary."""
        self.assert_rejects(
            "empty_vocabulary",
            ValueError,
            logits=torch.zeros(3, 0),
            prompt_len=2,
            target_ids=(0,),
        )

    def test_length_mismatch(self) -> None:
        """Test length mismatch."""
        for shape, plen, tids in [
            ((5, 4), 2, (0, 1)),  # T too big
            ((2, 4), 2, (0, 1)),  # T too small
            ((1, 4), 1, (0,)),  # T == prompt_len but target nonempty
        ]:
            with self.subTest(shape=shape, prompt_len=plen):
                self.assert_rejects(
                    "logits_length_mismatch",
                    ValueError,
                    logits=torch.zeros(shape),
                    prompt_len=plen,
                    target_ids=tids,
                )

    def test_token_id_out_of_range(self) -> None:
        """Test token id out of range."""
        for tids in [(-1,), (4,), (0, 99), (2**70,)]:
            with self.subTest(target_ids=tids):
                self.assert_rejects(
                    "token_id_out_of_range",
                    ValueError,
                    logits=_good_logits(1, tids, vocab=4),
                    prompt_len=1,
                    target_ids=tids,
                )

    def test_non_finite_logits(self) -> None:
        """Test non finite logits."""
        for bad_val, loc in [
            (float("nan"), (1, 0)),
            (float("inf"), (0, 2)),
            (float("-inf"), (2, 1)),
        ]:
            with self.subTest(value=bad_val, loc=loc):
                logits = _good_logits(2, (0, 1))
                logits[loc] = bad_val
                self.assert_rejects(
                    "non_finite_logits",
                    ValueError,
                    logits=logits,
                    prompt_len=2,
                    target_ids=(0, 1),
                )

    def test_masked_row_nan_still_rejected(self) -> None:
        """Test masked row nan still rejected."""
        # The finiteness check covers the whole tensor, including rows that
        # are never gathered (row 0 predicts a prompt token; the final row
        # predicts past the span). Matches the module docstring.
        logits = _good_logits(2, (0, 1))
        logits[0, 0] = float("nan")  # masked: predicts prompt token 1
        self.assert_rejects(
            "non_finite_logits", ValueError, logits=logits, prompt_len=2, target_ids=(0, 1)
        )
        logits = _good_logits(2, (0, 1))
        logits[3, 0] = float("inf")  # masked: predicts past the span
        self.assert_rejects(
            "non_finite_logits", ValueError, logits=logits, prompt_len=2, target_ids=(0, 1)
        )

    def test_rejection_precedence(self) -> None:
        """Test rejection precedence."""
        # empty_vocabulary beats length mismatch.
        self.assert_rejects(
            "empty_vocabulary",
            ValueError,
            logits=torch.zeros(7, 0),
            prompt_len=2,
            target_ids=(0, 1),
        )
        # ndim beats prompt_len.
        self.assert_rejects(
            "logits_must_be_2d",
            ValueError,
            logits=torch.zeros(4),
            prompt_len=0,
            target_ids=(0,),
        )
        # length mismatch beats token range.
        self.assert_rejects(
            "logits_length_mismatch",
            ValueError,
            logits=torch.zeros(5, 4),
            prompt_len=2,
            target_ids=(0, 99),
        )
        # token range beats non-finite logits.
        logits = _good_logits(1, (99,), vocab=4)
        logits[0, 0] = float("nan")
        self.assert_rejects(
            "token_id_out_of_range",
            ValueError,
            logits=logits,
            prompt_len=1,
            target_ids=(99,),
        )


# ---------------------------------------------------------------------------
# score_span_from_logits: values, masking, determinism, accumulation
# ---------------------------------------------------------------------------


class TestScoreSpanValues(unittest.TestCase):
    """Test score span values."""

    def test_empty_target_span(self) -> None:
        """Test empty target span."""
        per_token, total, mean = score_span_from_logits(
            torch.zeros(3, 4), prompt_len=3, target_ids=()
        )
        self.assertEqual(per_token, ())
        self.assertIs(type(per_token), tuple)
        self.assertEqual(total, 0.0)
        self.assertIs(type(total), float)
        self.assertIsNone(mean)

    def test_single_token_vocab_exact_zeros(self) -> None:
        """Test single token vocab exact zeros."""
        # V=1: every logprob is exactly 0.
        per_token, total, mean = score_span_from_logits(
            torch.zeros(4, 1), prompt_len=1, target_ids=(0, 0, 0)
        )
        self.assertEqual(per_token, (0.0, 0.0, 0.0))
        self.assertEqual(total, 0.0)
        self.assertEqual(mean, 0.0)

    def test_hand_computed_two_token(self) -> None:
        """Test hand computed two token."""
        # prompt_len=2, target=(2, 0); row 1 predicts token 2, row 2 predicts 0.
        logits = torch.tensor(
            [
                [9.0, 9.0, 9.0],  # masked: predicts prompt token
                [0.0, 0.0, 3.0],  # predicts target token 2
                [2.0, 0.0, 0.0],  # predicts target token 0
                [7.0, 7.0, 7.0],  # masked: predicts past the span
            ]
        )
        per_token, total, mean = score_span_from_logits(logits, prompt_len=2, target_ids=(2, 0))
        rows = logits.tolist()
        want = [
            ref_f32(ref_logprob_naive(rows[1], 2)),
            ref_f32(ref_logprob_naive(rows[2], 0)),
        ]
        self.assertEqual(list(per_token), want)
        unrounded_total = math.fsum([ref_logprob_naive(rows[1], 2), ref_logprob_naive(rows[2], 0)])
        self.assertEqual(total, ref_f32(unrounded_total))
        self.assertEqual(mean, ref_f32(unrounded_total / 2))

    def test_masked_rows_never_read(self) -> None:
        """Test masked rows never read."""
        rng = random.Random("cycle38-masking")  # noqa: S311 -- deterministic string seed for test fuzz
        prompt_len, n_tgt, vocab = 3, 4, 6
        target = tuple(rng.randrange(vocab) for _ in range(n_tgt))
        base = torch.tensor(
            [[rng.uniform(-2, 2) for _ in range(vocab)] for _ in range(prompt_len + n_tgt)]
        )
        first = score_span_from_logits(base, prompt_len=prompt_len, target_ids=target)
        for trial in range(10):
            masked = base.clone()
            for r in list(range(prompt_len - 1)) + [prompt_len + n_tgt - 1]:
                for c in range(vocab):
                    masked[r, c] = rng.uniform(-50, 50)
            second = score_span_from_logits(masked, prompt_len=prompt_len, target_ids=target)
            with self.subTest(trial=trial):
                self.assertEqual(first, second)

    def test_causal_shift_exact_row(self) -> None:
        """Test causal shift exact row."""
        # Only the predicting row may influence a token's score: give each
        # target row a unique spike and check the gathered values match the
        # spike rows, not their neighbors.
        vocab, prompt_len, n_tgt = 5, 2, 3
        target = (4, 1, 3)
        logits = torch.full((prompt_len + n_tgt, vocab), -100.0)
        for i, tid in enumerate(target):
            logits[prompt_len - 1 + i, tid] = 100.0
        per_token, _, _ = score_span_from_logits(logits, prompt_len=prompt_len, target_ids=target)
        for v in per_token:
            # log(1 / (1 + 4 * exp(-200))) underflows to -0.0 in float64.
            self.assertLess(abs(v), 1e-6)
            self.assertFalse(math.isnan(v))

    def test_deterministic(self) -> None:
        """Test deterministic."""
        rng = random.Random("cycle38-determinism")  # noqa: S311 -- deterministic string seed for test fuzz
        vocab, prompt_len, n_tgt = 7, 3, 5
        target = tuple(rng.randrange(vocab) for _ in range(n_tgt))
        logits = torch.tensor(
            [[rng.uniform(-5, 5) for _ in range(vocab)] for _ in range(prompt_len + n_tgt)]
        )
        first = score_span_from_logits(logits, prompt_len=prompt_len, target_ids=target)
        for _ in range(3):
            self.assertEqual(
                score_span_from_logits(logits.clone(), prompt_len=prompt_len, target_ids=target),
                first,
            )

    def test_reported_values_are_f32_representable(self) -> None:
        """Test reported values are f32 representable."""
        rng = random.Random("cycle38-f32-representable")  # noqa: S311 -- deterministic string seed for test fuzz
        for trial in range(30):
            vocab = rng.randint(2, 9)
            prompt_len = rng.randint(1, 4)
            n_tgt = rng.randint(1, 6)
            target = tuple(rng.randrange(vocab) for _ in range(n_tgt))
            logits = torch.tensor(
                [[rng.uniform(-8, 8) for _ in range(vocab)] for _ in range(prompt_len + n_tgt)]
            )
            per_token, total, mean = score_span_from_logits(
                logits, prompt_len=prompt_len, target_ids=target
            )
            with self.subTest(trial=trial):
                for v in per_token:
                    self.assertEqual(ref_f32(v), v)
                self.assertEqual(ref_f32(total), total)
                self.assertEqual(ref_f32(mean), mean)

    def test_total_accumulates_unrounded_float64(self) -> None:
        """Test total accumulates unrounded float64."""
        # 60k identical target rows with a per-token logprob that is NOT
        # exactly representable in binary32: rounding each per-token value
        # before accumulation shifts the total by ~1.5 half-ulps at the
        # total's magnitude, so the two designs round differently. This
        # separates "accumulate unrounded float64, round once" from
        # "round each token, then accumulate".
        n = 60000
        # float64 input: the engine converts to float64 exactly, so the
        # oracle below sees bit-identical row values (verified by reading
        # the row back from the tensor).
        logits = torch.tensor([[0.0, -17.3]] * (1 + n), dtype=torch.float64)
        row = logits[1].tolist()
        self.assertEqual(row, [0.0, -17.3])
        lp = ref_logprob_naive(row, 1)
        self.assertNotEqual(ref_f32(lp), lp)  # rounding actually loses info
        per_token, total, mean = score_span_from_logits(logits, prompt_len=1, target_ids=(1,) * n)
        # Reference mirrors the engine's sequential float64 accumulation of
        # the *unrounded* per-token values (naive oracle, moderate magnitudes).
        acc = 0.0
        for _ in range(n):
            acc += lp
        expected_total = ref_f32(acc)
        acc_rounded = 0.0
        for _ in range(n):
            acc_rounded += ref_f32(lp)
        rounded_first_total = ref_f32(acc_rounded)
        self.assertNotEqual(expected_total, rounded_first_total)  # test discriminates
        self.assertEqual(total, expected_total)
        self.assertEqual(mean, ref_f32(acc / n))
        self.assertTrue(all(v == ref_f32(lp) for v in per_token))

    def test_fuzz_against_naive_oracle(self) -> None:
        """Test fuzz against naive oracle."""
        rng = random.Random("cycle38-scoring-fuzz-v1")  # noqa: S311 -- deterministic string seed for test fuzz
        for trial in range(60):
            vocab = rng.randint(1, 8)
            prompt_len = rng.randint(1, 5)
            n_tgt = rng.randint(0, 6)
            target = tuple(rng.randrange(vocab) for _ in range(n_tgt))
            rows = [[rng.uniform(-5, 5) for _ in range(vocab)] for _ in range(prompt_len + n_tgt)]
            logits = torch.tensor(rows, dtype=torch.float64)
            per_token, total, mean = score_span_from_logits(
                logits, prompt_len=prompt_len, target_ids=target
            )
            with self.subTest(trial=trial):
                self.assertEqual(len(per_token), n_tgt)
                self.assertIs(type(per_token), tuple)
                unrounded = [
                    ref_logprob_naive(rows[prompt_len - 1 + i], tid) for i, tid in enumerate(target)
                ]
                for got, want in zip(per_token, unrounded, strict=True):
                    self.assertAlmostEqual(got, ref_f32(want), places=6)
                self.assertAlmostEqual(total, ref_f32(math.fsum(unrounded)), places=6)
                if n_tgt:
                    self.assertAlmostEqual(mean, ref_f32(math.fsum(unrounded) / n_tgt), places=6)
                else:
                    self.assertIsNone(mean)

    def test_extreme_logits_stable(self) -> None:
        """Test extreme logits stable."""
        # ±1e30 spikes must not overflow: the engine normalizes in float64
        # with a max shift.
        logits = torch.tensor(
            [
                [0.0, 0.0],
                [1e30, 0.0],  # predicts target token 0
                [0.0, -1e30],  # predicts target token 0
            ]
        )
        per_token, total, mean = score_span_from_logits(logits, prompt_len=1, target_ids=(0, 0))
        for v in per_token:
            self.assertTrue(math.isfinite(v))
            self.assertLessEqual(v, 0.0)
        self.assertTrue(math.isfinite(total))
        self.assertTrue(math.isfinite(mean))

    def test_accepts_float64_and_int_logits(self) -> None:
        """Test accepts float64 and int logits."""
        for dtype in (torch.float64, torch.int64):
            logits = torch.ones(3, 4, dtype=dtype)
            per_token, total, _mean = score_span_from_logits(
                logits, prompt_len=1, target_ids=(0, 1)
            )
            with self.subTest(dtype=str(dtype)):
                # Uniform rows -> logprob = -log(4) per token.
                want = ref_f32(-math.log(4))
                self.assertEqual(list(per_token), [want, want])
                self.assertIs(type(total), float)


# ---------------------------------------------------------------------------
# _safe_decode / _safe_encode sanitization boundaries
# ---------------------------------------------------------------------------


class TestSafeDecode(unittest.TestCase):
    """Test safe decode."""

    def test_passthrough(self) -> None:
        """Test passthrough."""
        codec = FakeCodec()
        codec.decode_fn = lambda ids: "hi"
        self.assertEqual(_safe_decode(codec, (1, 2)), "hi")

    def test_exception_becomes_none(self) -> None:
        """Test exception becomes none."""
        for exc in (RuntimeError("boom"), ValueError("nope"), KeyError("k")):
            codec = FakeCodec()

            def _raise(ids, _e=exc):
                """Raise."""
                raise _e

            codec.decode_fn = _raise
            with self.subTest(exc=type(exc).__name__):
                self.assertIsNone(_safe_decode(codec, (1,)))

    def test_non_str_becomes_none(self) -> None:
        """Test non str becomes none."""
        for bad in (123, None, b"bytes", ["a"], ("a",), 4.5):
            codec = FakeCodec()
            codec.decode_fn = lambda ids, _b=bad: _b
            with self.subTest(bad=repr(bad)):
                self.assertIsNone(_safe_decode(codec, (1,)))

    def test_str_subclass_becomes_none(self) -> None:
        """Test str subclass becomes none."""

        class MyStr(str):
            """My str."""

        codec = FakeCodec()
        codec.decode_fn = lambda ids: MyStr("x")
        self.assertIsNone(_safe_decode(codec, (1,)))

    def test_ids_passed_through_verbatim(self) -> None:
        """Test ids passed through verbatim."""
        seen: list = []
        codec = FakeCodec()
        codec.decode_fn = lambda ids: seen.append(ids) or ""
        _safe_decode(codec, (9, 8))
        self.assertEqual(seen, [(9, 8)])


class TestSafeEncode(unittest.TestCase):
    """Test safe encode."""

    def test_passthrough(self) -> None:
        """Test passthrough."""
        codec = FakeCodec()
        codec.encode_fn = lambda text: (4, 5)
        self.assertEqual(_safe_encode(codec, "hi"), (4, 5))

    def test_empty_tuple_is_valid(self) -> None:
        """Test empty tuple is valid."""
        codec = FakeCodec()
        codec.encode_fn = lambda text: ()
        self.assertEqual(_safe_encode(codec, ""), ())

    def test_exception_becomes_none(self) -> None:
        """Test exception becomes none."""
        codec = FakeCodec()

        def _raise(text):
            """Raise."""
            raise RuntimeError("boom")

        codec.encode_fn = _raise
        self.assertIsNone(_safe_encode(codec, "hi"))

    def test_non_tuple_becomes_none(self) -> None:
        """Test non tuple becomes none."""
        for bad in ([1, 2], "12", None, 7, {1: 2}):
            codec = FakeCodec()
            codec.encode_fn = lambda text, _b=bad: _b
            with self.subTest(bad=repr(bad)):
                self.assertIsNone(_safe_encode(codec, "hi"))

    def test_tuple_with_non_int_becomes_none(self) -> None:
        """Test tuple with non int becomes none."""
        for bad in ((1, "x"), (1.0, 2), (True, 2), ((1,),), (None,)):
            codec = FakeCodec()
            codec.encode_fn = lambda text, _b=bad: _b
            with self.subTest(bad=repr(bad)):
                self.assertIsNone(_safe_encode(codec, "hi"))

    def test_tuple_subclass_becomes_none(self) -> None:
        """Test tuple subclass becomes none."""

        class MyTuple(tuple):
            """My tuple."""

        codec = FakeCodec()
        codec.encode_fn = lambda text: MyTuple((1, 2))
        self.assertIsNone(_safe_encode(codec, "hi"))

    def test_text_passed_through_verbatim(self) -> None:
        """Test text passed through verbatim."""
        seen: list = []
        codec = FakeCodec()
        codec.encode_fn = lambda text: seen.append(text) or ()
        _safe_encode(codec, "héllo")
        self.assertEqual(seen, ["héllo"])


# ---------------------------------------------------------------------------
# check_alignment: all reason codes and the ok path
# ---------------------------------------------------------------------------


class TestCheckAlignment(unittest.TestCase):
    """Test check alignment."""

    def test_ok(self) -> None:
        """Test ok."""
        result = check_alignment(ok_codec(), (1, 2), (3,))
        self.assertIs(result.ok, True)
        self.assertIsNone(result.reason)
        self.assertEqual(result.prompt_text, "ab")
        self.assertEqual(result.target_text, "c")

    def test_ok_is_bool(self) -> None:
        """Test ok is bool."""
        result = check_alignment(ok_codec(), (1, 2), (3,))
        self.assertIs(type(result.ok), bool)

    def test_prompt_decode_failed(self) -> None:
        """Test prompt decode failed."""
        codec = scripted_codec({}, {}, decode_raises=((1, 2),))
        result = check_alignment(codec, (1, 2), (3,))
        self.assertEqual(result, AlignmentCheck(False, "decode_failed", "", ""))

    def test_target_decode_failed_keeps_prompt_text(self) -> None:
        """Test target decode failed keeps prompt text."""
        codec = scripted_codec({(1, 2): "ab"}, {"ab": (1, 2)}, decode_raises=((3,),))
        result = check_alignment(codec, (1, 2), (3,))
        self.assertEqual(result, AlignmentCheck(False, "decode_failed", "ab", ""))

    def test_decode_non_str_is_decode_failed(self) -> None:
        """Test decode non str is decode failed."""
        codec = scripted_codec({(1, 2): 12345}, {})
        result = check_alignment(codec, (1, 2), (3,))
        self.assertEqual(result.reason, "decode_failed")
        self.assertFalse(result.ok)

    def test_prompt_encode_failed(self) -> None:
        """Test prompt encode failed."""
        codec = scripted_codec({(1, 2): "ab", (3,): "c"}, {}, encode_raises=("ab",))
        result = check_alignment(codec, (1, 2), (3,))
        self.assertEqual(result, AlignmentCheck(False, "encode_failed", "ab", "c"))

    def test_prompt_roundtrip_mismatch(self) -> None:
        """Test prompt roundtrip mismatch."""
        codec = scripted_codec(
            {(1, 2): "ab", (3,): "c"},
            {"ab": (1, 9), "c": (3,)},  # prompt re-encodes differently
        )
        result = check_alignment(codec, (1, 2), (3,))
        self.assertEqual(result, AlignmentCheck(False, "prompt_roundtrip_mismatch", "ab", "c"))

    def test_target_roundtrip_mismatch(self) -> None:
        """Test target roundtrip mismatch."""
        codec = scripted_codec(
            {(1, 2): "ab", (3,): "c"},
            {"ab": (1, 2), "c": (3, 4)},  # target re-encodes differently
        )
        result = check_alignment(codec, (1, 2), (3,))
        self.assertEqual(result, AlignmentCheck(False, "target_roundtrip_mismatch", "ab", "c"))

    def test_target_encode_failed(self) -> None:
        """Test target encode failed."""
        codec = scripted_codec(
            {(1, 2): "ab", (3,): "c"},
            {"ab": (1, 2)},
            encode_raises=("c",),
        )
        result = check_alignment(codec, (1, 2), (3,))
        self.assertEqual(result, AlignmentCheck(False, "encode_failed", "ab", "c"))

    def test_combined_encode_failed(self) -> None:
        """Test combined encode failed."""
        codec = scripted_codec(
            {(1, 2): "ab", (3,): "c"},
            {"ab": (1, 2), "c": (3,)},
            encode_raises=("abc",),
        )
        result = check_alignment(codec, (1, 2), (3,))
        self.assertEqual(result, AlignmentCheck(False, "encode_failed", "ab", "c"))

    def test_boundary_merge_mismatch(self) -> None:
        """Test boundary merge mismatch."""
        # Each span round-trips, but the concatenated text re-tokenizes
        # differently: the classic straddling-token refusal.
        codec = scripted_codec(
            {(1, 2): "ab", (3,): "c"},
            {"ab": (1, 2), "c": (3,), "abc": (7, 8)},
        )
        result = check_alignment(codec, (1, 2), (3,))
        self.assertEqual(result, AlignmentCheck(False, "boundary_merge_mismatch", "ab", "c"))

    def test_combined_encode_non_tuple_is_encode_failed(self) -> None:
        """Test combined encode non tuple is encode failed."""
        codec = scripted_codec(
            {(1, 2): "ab", (3,): "c"},
            {"ab": (1, 2), "c": (3,), "abc": [1, 2, 3]},
        )
        result = check_alignment(codec, (1, 2), (3,))
        self.assertEqual(result.reason, "encode_failed")

    def test_genuine_splits_always_verify_fuzz(self) -> None:
        """Test genuine splits always verify fuzz."""
        # Liveness: genuine splits of a consistent codec must verify.
        rng = random.Random("cycle38-alignment-liveness")  # noqa: S311 -- deterministic string seed for test fuzz
        alphabet = [chr(65 + i) for i in range(26)]

        codec = FakeCodec()
        codec.decode_fn = lambda ids: "".join(alphabet[i % 26] for i in ids)
        codec.encode_fn = lambda text: tuple(alphabet.index(ch) for ch in text)

        for trial in range(50):
            n = rng.randint(1, 12)
            ids = tuple(rng.randrange(26) for _ in range(n))
            cut = rng.randint(0, n)
            with self.subTest(trial=trial, cut=cut):
                result = check_alignment(codec, ids[:cut], ids[cut:])
                self.assertTrue(result.ok)
                self.assertIsNone(result.reason)

    def test_does_not_require_backend_instance(self) -> None:
        """Test does not require backend instance."""
        # Backend is a structural protocol; a bare object with
        # decode/encode works.
        result = check_alignment(ok_codec(), (1, 2), (3,))
        self.assertTrue(result.ok)

    def test_missing_decode_method_is_decode_failed(self) -> None:
        """Test missing decode method is decode failed."""
        # Even a total protocol violation (no decode attribute at all) is
        # sanitized to the fixed code, never raised.
        result = check_alignment(object(), (1,), (2,))
        self.assertEqual(result, AlignmentCheck(False, "decode_failed", "", ""))

    def test_missing_encode_method_is_encode_failed(self) -> None:
        """Test missing encode method is encode failed."""

        class DecodeOnly:
            """Decode only."""

            @staticmethod
            def decode(token_ids):
                """Decode."""
                # noqa: ANN001, ANN202
                return "x"

        result = check_alignment(DecodeOnly(), (1,), (2,))
        self.assertEqual(result.reason, "encode_failed")
        self.assertFalse(result.ok)


class TestCheckAlignmentCallOrder(unittest.TestCase):
    """Test check alignment call order."""

    def test_ok_path_call_sequence(self) -> None:
        """Test ok path call sequence."""
        codec = ok_codec()
        check_alignment(codec, (1, 2), (3,))
        self.assertEqual(
            codec.calls,
            [
                ("decode", (1, 2)),
                ("decode", (3,)),
                ("encode", "ab"),
                ("encode", "c"),
                ("encode", "abc"),
            ],
        )

    def test_fail_fast_on_prompt_decode(self) -> None:
        """Test fail fast on prompt decode."""
        codec = scripted_codec({}, {}, decode_raises=((1, 2),))
        check_alignment(codec, (1, 2), (3,))
        self.assertEqual(codec.calls, [("decode", (1, 2))])

    def test_no_encode_calls_after_target_decode_failure(self) -> None:
        """Test no encode calls after target decode failure."""
        codec = scripted_codec({(1, 2): "ab"}, {"ab": (1, 2)}, decode_raises=((3,),))
        check_alignment(codec, (1, 2), (3,))
        self.assertEqual(codec.calls, [("decode", (1, 2)), ("decode", (3,))])

    def test_no_combined_encode_after_roundtrip_mismatch(self) -> None:
        """Test no combined encode after roundtrip mismatch."""
        codec = scripted_codec(
            {(1, 2): "ab", (3,): "c"},
            {"ab": (1, 9), "c": (3,)},
        )
        check_alignment(codec, (1, 2), (3,))
        self.assertEqual(
            codec.calls,
            [("decode", (1, 2)), ("decode", (3,)), ("encode", "ab")],
        )


# ---------------------------------------------------------------------------
# _provider_code whitelist
# ---------------------------------------------------------------------------


class TestProviderCode(unittest.TestCase):
    """Test provider code."""

    def test_every_whitelisted_code_passes_through(self) -> None:
        """Test every whitelisted code passes through."""
        for code in sorted(_REASON_PROVIDER):
            for exc_type in (ValueError, RuntimeError, Exception):
                with self.subTest(code=code, exc=exc_type.__name__):
                    self.assertEqual(_provider_code(exc_type(code)), code)

    def test_arbitrary_text_collapses(self) -> None:
        """Test arbitrary text collapses."""
        for bad in ("boom", "", "some backend diagnostic", "logits_must_be_tensor "):
            with self.subTest(bad=repr(bad)):
                self.assertEqual(_provider_code(ValueError(bad)), "forward_failed")
                self.assertEqual(_provider_code(RuntimeError(bad)), "forward_failed")

    def test_non_string_args_collapse(self) -> None:
        """Test non string args collapse."""
        for args in ((), (None,), (123,), ((1, 2),), (b"bytes",)):
            with self.subTest(args=repr(args)):
                self.assertEqual(_provider_code(ValueError(*args)), "forward_failed")

    def test_str_subclass_code_does_not_pass(self) -> None:
        """Test str subclass code does not pass."""

        # Conservative: only exact-str codes pass, never subclasses.
        class SneakyStr(str):
            """Sneaky str."""

        self.assertEqual(
            _provider_code(ValueError(SneakyStr("invalid_input_ids"))), "forward_failed"
        )

    def test_second_arg_ignored(self) -> None:
        """Test second arg ignored."""
        # Only args[0] is consulted; a whitelisted code in args[1] is not
        # enough, and junk in args[1] does not disturb a whitelisted args[0].
        self.assertEqual(_provider_code(ValueError("boom", "invalid_input_ids")), "forward_failed")
        self.assertEqual(
            _provider_code(ValueError("invalid_input_ids", "junk")), "invalid_input_ids"
        )


# ---------------------------------------------------------------------------
# _unavailable contract
# ---------------------------------------------------------------------------


class TestUnavailableContract(unittest.TestCase):
    """Test unavailable contract."""

    def test_fields(self) -> None:
        """Test fields."""
        score = _unavailable("empty_prompt", 12, 3)
        self.assertEqual(
            score,
            TargetScore(
                status=STATUS_UNAVAILABLE,
                reason="empty_prompt",
                per_token_logprobs=(),
                total_logprob=None,
                mean_logprob=None,
                scored_token_count=0,
                target_byte_length=12,
                prompt_token_count=3,
            ),
        )

    def test_likelihood_fields_are_none_not_zero(self) -> None:
        """Test likelihood fields are none not zero."""
        # An unavailable score is distinct from a legitimate score of zero.
        score = _unavailable("forward_failed", 0, 1)
        self.assertIsNone(score.total_logprob)
        self.assertIsNone(score.mean_logprob)
        self.assertEqual(score.per_token_logprobs, ())
        self.assertEqual(score.scored_token_count, 0)

    def test_counts_pass_through(self) -> None:
        """Test counts pass through."""
        score = _unavailable("logits_not_supported", 7, 9)
        self.assertEqual(score.target_byte_length, 7)
        self.assertEqual(score.prompt_token_count, 9)
        self.assertEqual(score.status, STATUS_UNAVAILABLE)


# ---------------------------------------------------------------------------
# Reason vocabularies: exact membership and disjointness
# ---------------------------------------------------------------------------


class TestReasonVocabularies(unittest.TestCase):
    """Test reason vocabularies."""

    def test_alignment_vocabulary_exact(self) -> None:
        """Test alignment vocabulary exact."""
        self.assertEqual(
            _REASON_ALIGNMENT,
            frozenset(
                {
                    "decode_failed",
                    "encode_failed",
                    "prompt_roundtrip_mismatch",
                    "target_roundtrip_mismatch",
                    "boundary_merge_mismatch",
                }
            ),
        )

    def test_provider_vocabulary_exact(self) -> None:
        """Test provider vocabulary exact."""
        self.assertEqual(
            _REASON_PROVIDER,
            frozenset(
                {
                    "invalid_input_ids",
                    "context_limit_exceeded",
                    "local_forward_failed",
                    "local_backend_not_initialized",
                    "logits_must_be_tensor",
                    "logits_must_be_2d",
                    "prompt_len_must_be_positive",
                    "invalid_target_ids",
                    "empty_vocabulary",
                    "logits_length_mismatch",
                    "token_id_out_of_range",
                    "non_finite_logits",
                }
            ),
        )

    def test_engine_vocabulary_exact(self) -> None:
        """Test engine vocabulary exact."""
        self.assertEqual(
            _REASON_ENGINE,
            frozenset({"empty_prompt", "logits_not_supported", "forward_failed"}),
        )

    def test_vocabularies_disjoint(self) -> None:
        """Test vocabularies disjoint."""
        self.assertTrue(_REASON_ALIGNMENT.isdisjoint(_REASON_PROVIDER))
        self.assertTrue(_REASON_ALIGNMENT.isdisjoint(_REASON_ENGINE))
        self.assertTrue(_REASON_PROVIDER.isdisjoint(_REASON_ENGINE))

    def test_span_rejection_codes_all_whitelisted(self) -> None:
        """Test span rejection codes all whitelisted."""
        # Every code score_span_from_logits can raise is in the provider
        # vocabulary, so _provider_code passes them through unchanged.
        for code in (
            "logits_must_be_tensor",
            "logits_must_be_2d",
            "prompt_len_must_be_positive",
            "invalid_target_ids",
            "empty_vocabulary",
            "logits_length_mismatch",
            "token_id_out_of_range",
            "non_finite_logits",
        ):
            with self.subTest(code=code):
                self.assertIn(code, _REASON_PROVIDER)

    def test_check_alignment_codes_all_in_alignment_vocab(self) -> None:
        """Test check alignment codes all in alignment vocab."""
        for code in (
            "decode_failed",
            "encode_failed",
            "prompt_roundtrip_mismatch",
            "target_roundtrip_mismatch",
            "boundary_merge_mismatch",
        ):
            with self.subTest(code=code):
                self.assertIn(code, _REASON_ALIGNMENT)


# ---------------------------------------------------------------------------
# Dataclass, constant, and export contracts
# ---------------------------------------------------------------------------


class TestDataclassContracts(unittest.TestCase):
    """Test dataclass contracts."""

    def test_target_score_frozen(self) -> None:
        """Test target score frozen."""
        score = _unavailable("forward_failed", 0, 0)
        with self.assertRaises(AttributeError):
            score.status = STATUS_OK  # type: ignore[misc]

    def test_alignment_check_frozen(self) -> None:
        """Test alignment check frozen."""
        check = AlignmentCheck(True, None, "a", "b")
        with self.assertRaises(AttributeError):
            check.ok = False  # type: ignore[misc]

    def test_alignment_check_equality(self) -> None:
        """Test alignment check equality."""
        self.assertEqual(
            AlignmentCheck(False, "decode_failed", "", ""),
            AlignmentCheck(ok=False, reason="decode_failed", prompt_text="", target_text=""),
        )

    def test_status_constants(self) -> None:
        """Test status constants."""
        self.assertEqual(STATUS_OK, "ok")
        self.assertEqual(STATUS_OK_EMPTY, "ok_empty")
        self.assertEqual(STATUS_UNAVAILABLE, "unavailable")

    def test_target_score_field_types(self) -> None:
        """Test target score field types."""
        score = TargetScore(
            status=STATUS_OK,
            reason=None,
            per_token_logprobs=(-1.5,),
            total_logprob=-1.5,
            mean_logprob=-1.5,
            scored_token_count=1,
            target_byte_length=2,
            prompt_token_count=3,
        )
        self.assertIs(type(score.per_token_logprobs), tuple)
        self.assertIs(type(score.total_logprob), float)
        self.assertIs(type(score.scored_token_count), int)


class TestExports(unittest.TestCase):
    """Test exports."""

    def test_all_contains_expected_names(self) -> None:
        """Test all contains expected names."""
        import model_protocol_lab.scoring as scoring_mod

        self.assertEqual(
            set(scoring_mod.__all__),
            {
                "AlignmentCheck",
                "ScoreStatus",
                "STATUS_OK",
                "STATUS_OK_EMPTY",
                "STATUS_UNAVAILABLE",
                "TargetScore",
                "check_alignment",
                "score_span_from_logits",
                "score_target_logprobs",
            },
        )

    def test_all_names_importable(self) -> None:
        """Test all names importable."""
        import model_protocol_lab.scoring as scoring_mod

        for name in scoring_mod.__all__:
            with self.subTest(name=name):
                self.assertTrue(hasattr(scoring_mod, name), name)


# ---------------------------------------------------------------------------
# _require_torch: missing-torch path
# ---------------------------------------------------------------------------


class TestRequireTorch(unittest.TestCase):
    """Test require torch."""

    @staticmethod
    def _hide_torch():
        """Hide torch."""
        real = sys.modules.get("torch")
        sys.modules["torch"] = None  # type: ignore[assignment]
        return real

    @staticmethod
    def _restore_torch(real) -> None:
        """Restore torch."""
        if real is not None:
            sys.modules["torch"] = real
        else:
            del sys.modules["torch"]

    def test_missing_torch_raises_fixed_code(self) -> None:
        """Test missing torch raises fixed code."""
        real = self._hide_torch()
        try:
            with self.assertRaises(RuntimeError) as ctx:
                _require_torch()
            self.assertEqual(str(ctx.exception), "torch_not_installed")
            self.assertIsInstance(ctx.exception.__cause__, ImportError)
        finally:
            self._restore_torch(real)

    def test_score_span_without_torch_raises_fixed_code(self) -> None:
        """Test score span without torch raises fixed code."""
        real = self._hide_torch()
        try:
            with self.assertRaises(RuntimeError) as ctx:
                score_span_from_logits(torch.zeros(2, 2), prompt_len=1, target_ids=(0,))
            self.assertEqual(str(ctx.exception), "torch_not_installed")
        finally:
            self._restore_torch(real)

    def test_torch_restored_afterwards(self) -> None:
        """Test torch restored afterwards."""
        # Sanity: the hide/restore dance above leaves torch usable.
        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        import torch as t2

        self.assertTrue(hasattr(t2, "zeros"))


if __name__ == "__main__":
    unittest.main()
