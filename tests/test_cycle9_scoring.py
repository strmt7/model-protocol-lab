"""Cycle 9: teacher-forced log-likelihood scoring engine (§9.3-§9.5).

Acceptance gate per DEVELOPMENT_PLAN.md §9.5: tiny deterministic logit
tensors for which every target token's probability is known by hand;
one-token and multi-token prompts/answers; padding on both sides where
supported; special tokens; empty targets; merge-sensitive boundaries.
Adapter scores are cross-checked against an INDEPENDENT reference
computation -- plain Python math written separately from the scoring
engine (naive log-softmax with no max-shift, a deliberately different code
path from ``scoring.score_span_from_logits``), never a wrapper around the
function under test.
"""

import importlib.util
import math
import struct
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from model_protocol_lab.scoring import (
    STATUS_OK,
    STATUS_OK_EMPTY,
    STATUS_UNAVAILABLE,
    TargetScore,
    check_alignment,
    score_span_from_logits,
    score_target_logprobs,
)

REQUIRED = ("torch", "transformers", "safetensors", "tokenizers")
HAS_HF = all(importlib.util.find_spec(name) is not None for name in REQUIRED)


def reference_f32(value: float) -> float:
    """Independent float32 rounding (struct, not the engine's helper)."""
    rounded: float = struct.unpack("<f", struct.pack("<f", value))[0]
    return rounded


def reference_logprobs(logit_rows, target_ids):
    """Independent reference: naive log-softmax in plain Python.

    Deliberately avoids the max-shift formulation used by the engine so the
    cross-check exercises a different floating-point path. ``logit_rows``
    are sequences of Python floats (e.g. from ``tensor.tolist()``).
    """
    out = []
    for row, token_id in zip(logit_rows, target_ids, strict=True):
        log_z = math.log(math.fsum(math.exp(x) for x in row))
        out.append(row[token_id] - log_z)
    return out


def reference_aggregate(logit_rows, target_ids):
    """Independent (per-token, total, mean) triple, float32-rounded."""
    per_token = [reference_f32(lp) for lp in reference_logprobs(logit_rows, target_ids)]
    total = reference_f32(math.fsum(reference_logprobs(logit_rows, target_ids)))
    mean = reference_f32(total / len(target_ids)) if target_ids else None
    return per_token, total, mean


class ScriptTokenizer:
    """Minimal deterministic tokenizer with controllable merge behavior.

    Greedy longest-match encoding over an explicit piece table, so tests
    can construct merge-sensitive boundaries: e.g. pieces
    ``{"a": 0, "b": 1, "ab": 2}`` encode ``"a"+"b"`` separately as
    ``(0, 1)`` but encode the concatenated ``"ab"`` as ``(2,)``.
    """

    def __init__(self, pieces: dict[str, int]) -> None:
        self._pieces = dict(pieces)
        self._by_id = {v: k for k, v in pieces.items()}

    def encode(self, text: str) -> tuple[int, ...]:
        ids: list[int] = []
        pos = 0
        ordered = sorted(self._pieces, key=len, reverse=True)
        while pos < len(text):
            for piece in ordered:
                if text.startswith(piece, pos):
                    ids.append(self._pieces[piece])
                    pos += len(piece)
                    break
            else:
                raise ValueError("unencodable_text")
        return tuple(ids)

    def decode(self, token_ids: tuple[int, ...]) -> str:
        return "".join(self._by_id[token_id] for token_id in token_ids)


class StubBackend:
    """Weight-free backend: scripted tokenizer plus a hand-built logit tensor."""

    fingerprint = "stub-scoring-backend-not-a-model"

    def __init__(
        self,
        pieces: dict[str, int],
        logits: torch.Tensor | None = None,
        *,
        vocab_size: int | None = None,
        forward_error: BaseException | None = None,
        with_forward_logits: bool = True,
    ) -> None:
        self._tokenizer = ScriptTokenizer(pieces)
        self._logits = logits
        self._forward_error = forward_error
        self.forward_calls: list[tuple[int, ...]] = []
        if vocab_size is not None:
            self.vocab_size = vocab_size
        if not with_forward_logits:
            # Simulate a backend whose protocol has no forward primitive.
            self.forward_logits = None  # type: ignore[assignment]

    def encode(self, text: str) -> tuple[int, ...]:
        return self._tokenizer.encode(text)

    def decode(self, token_ids: tuple[int, ...]) -> str:
        return self._tokenizer.decode(token_ids)

    def forward_logits(self, input_ids: tuple[int, ...]) -> torch.Tensor:
        self.forward_calls.append(tuple(input_ids))
        if self._forward_error is not None:
            raise self._forward_error
        if self._logits is None:
            raise RuntimeError("no_logits_scripted")
        return self._logits


MERGE_PIECES = {"a": 0, "b": 1, "c": 2, "ab": 3, "héllo": 9}


def stub_with_logits(logits: torch.Tensor, **kwargs) -> StubBackend:
    return StubBackend(MERGE_PIECES, logits, vocab_size=10, **kwargs)


class HandBuiltTensorTests(unittest.TestCase):
    """Pure-math layer driven by hand-built logit tensors (§9.5)."""

    def test_single_token_vocab_scores_exact_zero(self):
        # V=1: the only token has probability 1 by hand; log(1) == 0 exactly.
        # Two rows: the prompt row predicts the single target token.
        per_token, total, mean = score_span_from_logits(
            torch.tensor([[4.0], [4.0]]), prompt_len=1, target_ids=(0,)
        )
        self.assertEqual(per_token, (0.0,))
        self.assertEqual(total, 0.0)
        self.assertEqual(mean, 0.0)

    def test_one_token_prompt_one_token_answer_known_values(self):
        # Full sequence = [prompt, target]; the prompt row (position 0)
        # predicts the target token (position 1). Target row [1,2,3],
        # target id 2: by hand,
        #   logprob = 3 - (3 + log(e^-2 + e^-1 + 1)) = -log(e^-2+e^-1+1).
        logits = torch.tensor([[1.0, 2.0, 3.0], [9.0, 9.0, 9.0]])
        per_token, total, mean = score_span_from_logits(logits, prompt_len=1, target_ids=(2,))
        expected = reference_logprobs([[1.0, 2.0, 3.0]], [2])[0]
        self.assertAlmostEqual(per_token[0], reference_f32(expected), places=9)
        # Hand-derived closed form; looser tolerance since the naive formula
        # takes a different floating-point path than the max-shifted engine.
        self.assertAlmostEqual(
            per_token[0], math.log(1 / (math.exp(-2) + math.exp(-1) + 1)), places=6
        )
        self.assertEqual(total, per_token[0])
        self.assertEqual(mean, per_token[0])

    def test_multi_token_prompt_multi_token_answer(self):
        logits = torch.tensor(
            [
                [0.5, -0.5, 1.5, 0.0],
                [-1.0, 2.0, 0.25, -0.25],
                [0.1, 0.2, 0.3, 0.4],
                [1.0, 1.0, -2.0, 0.5],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        target_ids = (3, 0, 2)
        per_token, total, mean = score_span_from_logits(logits, prompt_len=2, target_ids=target_ids)
        rows = logits.tolist()
        # Causal shift: target token i is predicted by row (prompt_len - 1 + i),
        # i.e. rows 1, 2, 3 for target_ids (3, 0, 2).
        exp_per, exp_total, exp_mean = reference_aggregate(rows[1:4], target_ids)
        self.assertEqual(len(per_token), 3)
        for got, want in zip(per_token, exp_per, strict=True):
            self.assertAlmostEqual(got, want, places=9)
        self.assertAlmostEqual(total, exp_total, places=9)
        self.assertAlmostEqual(mean, exp_mean, places=9)

    def test_prompt_rows_are_masked_never_scored(self):
        # prompt_len=2, two target tokens: target 0 is predicted by row 1,
        # target 1 by row 2. Row 0 (predicts a prompt token) and row 3
        # (predicts past the span) are masked: changing them cannot move
        # the score, while the predicting rows 1-2 are gathered.
        base = torch.tensor(
            [
                [100.0, -100.0, 50.0],
                [0.2, -0.3, 0.9],
                [1.1, 0.4, -0.7],
                [-80.0, 90.0, -70.0],
            ]
        )
        masked = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.2, -0.3, 0.9],
                [1.1, 0.4, -0.7],
                [0.0, 0.0, 0.0],
            ]
        )
        first = score_span_from_logits(base, prompt_len=2, target_ids=(2, 0))
        second = score_span_from_logits(masked, prompt_len=2, target_ids=(2, 0))
        self.assertEqual(first, second)
        self.assertEqual(len(first[0]), 2)
        exp_per, _, _ = reference_aggregate(base.tolist()[1:3], (2, 0))
        for got, want in zip(first[0], exp_per, strict=True):
            self.assertAlmostEqual(got, want, places=9)

    def test_special_and_pad_like_ids_in_target_are_scored_per_policy(self):
        # Declared policy: no implicit unmasking -- every ID in the target
        # span is scored, including pad-like and special IDs on both sides.
        pad, special = 7, 4
        logits = torch.tensor(
            [
                [0.0] * 8,
                [0.3, -0.1, 0.0, 0.2, 1.5, -0.4, 0.7, -1.2],
                [0.1] * 8,
                [-0.5, 0.6, 0.0, 0.0, -0.2, 0.9, 0.0, 2.0],
                [0.2] * 8,
            ]
        )
        target_ids = (pad, special, pad)
        per_token, total, mean = score_span_from_logits(logits, prompt_len=2, target_ids=target_ids)
        exp_per, exp_total, exp_mean = reference_aggregate(logits.tolist()[1:4], target_ids)
        self.assertEqual(len(per_token), 3)
        for got, want in zip(per_token, exp_per, strict=True):
            self.assertAlmostEqual(got, want, places=9)
        self.assertAlmostEqual(total, exp_total, places=9)
        self.assertAlmostEqual(mean, exp_mean, places=9)

    def test_empty_target_span_scores_zero_tokens(self):
        per_token, total, mean = score_span_from_logits(
            torch.tensor([[0.5, -0.5], [0.1, 0.2]]), prompt_len=2, target_ids=()
        )
        self.assertEqual(per_token, ())
        self.assertEqual(total, 0.0)
        self.assertIsNone(mean)

    def test_rejects_nan_logits(self):
        with self.assertRaises(ValueError):
            score_span_from_logits(
                torch.tensor([[float("nan"), 0.0], [0.1, 0.2]]),
                prompt_len=1,
                target_ids=(0,),
            )

    def test_rejects_inf_logits(self):
        with self.assertRaises(ValueError):
            score_span_from_logits(
                torch.tensor([[float("inf"), 0.0], [0.1, 0.2]]),
                prompt_len=1,
                target_ids=(0,),
            )

    def test_rejects_out_of_range_token_id(self):
        with self.assertRaises(ValueError):
            score_span_from_logits(
                torch.tensor([[0.1, 0.2], [0.3, 0.4]]), prompt_len=1, target_ids=(2,)
            )
        with self.assertRaises(ValueError):
            score_span_from_logits(
                torch.tensor([[0.1, 0.2], [0.3, 0.4]]), prompt_len=1, target_ids=(-1,)
            )

    def test_rejects_length_mismatch(self):
        with self.assertRaises(ValueError):
            score_span_from_logits(torch.tensor([[0.1, 0.2]] * 3), prompt_len=1, target_ids=(0,))

    def test_rejects_empty_prompt_len(self):
        with self.assertRaises(ValueError):
            score_span_from_logits(torch.tensor([[0.1, 0.2]]), prompt_len=0, target_ids=(0,))

    def test_rejects_non_2d_logits(self):
        with self.assertRaises(ValueError):
            score_span_from_logits(torch.tensor([0.1, 0.2]), prompt_len=1, target_ids=(0,))

    def test_rejects_non_tensor_logits(self):
        with self.assertRaises(TypeError):
            score_span_from_logits([[0.1, 0.2]], prompt_len=1, target_ids=(0,))

    def test_deterministic_bit_identical(self):
        logits = torch.tensor([[0.7, -0.2, 1.3], [-0.9, 0.4, 0.4], [0.0, 0.0, 0.5]])
        first = score_span_from_logits(logits, prompt_len=1, target_ids=(2, 1))
        second = score_span_from_logits(logits, prompt_len=1, target_ids=(2, 1))
        self.assertEqual(first, second)


class AlignmentTests(unittest.TestCase):
    """Alignment verification and refusal behavior (§9.3)."""

    def test_aligned_pair_scores_against_reference(self):
        logits = torch.tensor(
            [
                [0.5, -1.0, 2.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        backend = stub_with_logits(logits)
        score = score_target_logprobs(backend, (0,), (2,))
        self.assertIsInstance(score, TargetScore)
        self.assertEqual(score.status, STATUS_OK)
        self.assertIsNone(score.reason)
        self.assertEqual(score.scored_token_count, 1)
        self.assertEqual(score.prompt_token_count, 1)
        self.assertEqual(score.target_byte_length, 1)  # "c"
        exp_per, exp_total, exp_mean = reference_aggregate(logits.tolist()[0:1], (2,))
        self.assertAlmostEqual(score.per_token_logprobs[0], exp_per[0], places=9)
        self.assertAlmostEqual(score.total_logprob, exp_total, places=9)
        self.assertAlmostEqual(score.mean_logprob, exp_mean, places=9)
        # Exactly one forward pass over prompt + target.
        self.assertEqual(backend.forward_calls, [(0, 2)])

    def test_multibyte_target_byte_length(self):
        logits = torch.zeros(2, 10)
        backend = stub_with_logits(logits)
        score = score_target_logprobs(backend, (0,), (9,))
        self.assertEqual(score.status, STATUS_OK)
        self.assertEqual(score.target_byte_length, len("héllo".encode()))
        self.assertEqual(score.target_byte_length, 6)

    def test_straddling_token_refuses_with_explicit_reason(self):
        # "a" + "b" encode separately as (0, 1) but "ab" encodes as (2,):
        # one token straddles the prompt/answer boundary.
        backend = stub_with_logits(torch.zeros(2, 10))
        score = score_target_logprobs(backend, (0,), (1,))
        self.assertEqual(score.status, STATUS_UNAVAILABLE)
        self.assertEqual(score.reason, "boundary_merge_mismatch")
        self.assertEqual(score.per_token_logprobs, ())
        self.assertIsNone(score.total_logprob)
        self.assertIsNone(score.mean_logprob)
        self.assertEqual(score.scored_token_count, 0)
        self.assertEqual(backend.forward_calls, [])

    def test_target_roundtrip_mismatch_refuses(self):
        # decode((0, 1)) == "ab" but encode("ab") == (3,) != (0, 1).
        backend = stub_with_logits(torch.zeros(3, 10))
        score = score_target_logprobs(backend, (3,), (0, 1))
        self.assertEqual(score.status, STATUS_UNAVAILABLE)
        self.assertEqual(score.reason, "target_roundtrip_mismatch")

    def test_prompt_roundtrip_mismatch_refuses(self):
        backend = stub_with_logits(torch.zeros(3, 10))
        score = score_target_logprobs(backend, (0, 1), (2,))
        self.assertEqual(score.status, STATUS_UNAVAILABLE)
        self.assertEqual(score.reason, "prompt_roundtrip_mismatch")

    def test_check_alignment_reports_decoded_spans(self):
        backend = stub_with_logits(torch.zeros(2, 10))
        check = check_alignment(backend, (0,), (2,))
        self.assertTrue(check.ok)
        self.assertIsNone(check.reason)
        self.assertEqual((check.prompt_text, check.target_text), ("a", "c"))
        bad = check_alignment(backend, (0,), (1,))
        self.assertFalse(bad.ok)
        self.assertEqual(bad.reason, "boundary_merge_mismatch")

    def test_empty_target_is_defined_not_an_error(self):
        backend = stub_with_logits(torch.zeros(1, 10))
        score = score_target_logprobs(backend, (0,), ())
        self.assertEqual(score.status, STATUS_OK_EMPTY)
        self.assertIsNone(score.reason)
        self.assertEqual(score.per_token_logprobs, ())
        self.assertEqual(score.total_logprob, 0.0)
        self.assertIsNone(score.mean_logprob)
        self.assertEqual(score.scored_token_count, 0)
        self.assertEqual(score.target_byte_length, 0)
        self.assertEqual(backend.forward_calls, [])

    def test_empty_prompt_is_refused(self):
        backend = stub_with_logits(torch.zeros(1, 10))
        score = score_target_logprobs(backend, (), (2,))
        self.assertEqual(score.status, STATUS_UNAVAILABLE)
        self.assertEqual(score.reason, "empty_prompt")

    def test_token_id_out_of_range_is_refused(self):
        backend = stub_with_logits(torch.zeros(2, 10))
        score = score_target_logprobs(backend, (0,), (10,))
        self.assertEqual(score.status, STATUS_UNAVAILABLE)
        self.assertEqual(score.reason, "token_id_out_of_range")
        self.assertEqual(backend.forward_calls, [])

    def test_backend_without_forward_logits_is_refused(self):
        backend = StubBackend(MERGE_PIECES, torch.zeros(2, 10), with_forward_logits=False)
        score = score_target_logprobs(backend, (0,), (2,))
        self.assertEqual(score.status, STATUS_UNAVAILABLE)
        self.assertEqual(score.reason, "logits_not_supported")

    def test_forward_failure_becomes_unavailable_not_exception(self):
        backend = stub_with_logits(
            torch.zeros(2, 10), forward_error=RuntimeError("simulated provider outage")
        )
        score = score_target_logprobs(backend, (0,), (2,))
        self.assertEqual(score.status, STATUS_UNAVAILABLE)
        # Arbitrary backend text never reaches the fixed reason vocabulary.
        self.assertEqual(score.reason, "forward_failed")

    def test_known_provider_code_passes_through(self):
        backend = stub_with_logits(
            torch.zeros(2, 10), forward_error=ValueError("context_limit_exceeded")
        )
        score = score_target_logprobs(backend, (0,), (2,))
        self.assertEqual(score.status, STATUS_UNAVAILABLE)
        self.assertEqual(score.reason, "context_limit_exceeded")

    def test_malformed_caller_input_raises(self):
        backend = stub_with_logits(torch.zeros(2, 10))
        with self.assertRaises(TypeError):
            score_target_logprobs(backend, [0], (2,))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            score_target_logprobs(backend, (0,), (-1,))
        with self.assertRaises(ValueError):
            score_target_logprobs(backend, (True,), (2,))

    def test_deterministic_two_runs_identical(self):
        logits = torch.tensor(
            [
                [0.7, -0.2, 1.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4],
                [-0.9, 0.4, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1],
            ]
        )
        backend = stub_with_logits(logits)
        first = score_target_logprobs(backend, (0,), (2,))
        second = score_target_logprobs(backend, (0,), (2,))
        self.assertEqual(first, second)


def build_tiny_bpe_gpt2(root: Path) -> None:
    """Minimal GPT-2 with a hand-controlled BPE merge table (local only).

    The merge ``a + b -> ab`` makes ``encode("a") + encode("b") !=
    encode("ab")`` -- a genuine merge-sensitive boundary -- while pairs
    like ``"a"`` / ``"c"`` concatenate cleanly for the aligned case.
    """
    import torch
    from tokenizers import Tokenizer, decoders
    from tokenizers.models import BPE
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(20261014)
    vocab = {"a": 0, "b": 1, "c": 2, "ab": 3, "<s>": 4, "</s>": 5}
    tokenizer = Tokenizer(BPE(vocab=vocab, merges=[("a", "b")], unk_token=None))
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.save(str(root / "tokenizer.json"))
    config = GPT2Config(
        vocab_size=len(vocab),
        n_positions=64,
        n_embd=16,
        n_layer=1,
        n_head=2,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    GPT2LMHeadModel(config).save_pretrained(str(root))


@unittest.skipUnless(HAS_HF, "the 'hf' extra is not installed")
class RealBackendScoringTests(unittest.TestCase):
    """Adapter scores vs the independent reference on a real LocalHFBackend."""

    @classmethod
    def setUpClass(cls):
        from model_protocol_lab.backends import LocalHFBackend

        cls._tmp = TemporaryDirectory()
        cls.addClassCleanup(cls._tmp.cleanup)
        cls.root = Path(cls._tmp.name)
        build_tiny_bpe_gpt2(cls.root)
        cls.backend = LocalHFBackend(cls.root, device="cpu", max_context=64)

    def test_forward_logits_contract(self):
        logits = self.backend.forward_logits((0, 2))
        self.assertIsInstance(logits, torch.Tensor)
        self.assertEqual(tuple(logits.shape), (2, 6))
        self.assertEqual(logits.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(logits).all().item()))
        self.assertEqual(self.backend.vocab_size, 6)

    def test_forward_logits_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            self.backend.forward_logits(())
        with self.assertRaises(ValueError):
            self.backend.forward_logits(tuple([0] * 65))
        with self.assertRaises(ValueError):
            self.backend.forward_logits((0, 99))

    def test_real_backend_score_matches_independent_reference(self):
        # prompt "a" -> (0,), target "c" -> (2,): "ac" encodes as (0, 2).
        score = score_target_logprobs(self.backend, (0,), (2,))
        self.assertEqual(score.status, STATUS_OK)
        self.assertIsNone(score.reason)
        self.assertEqual(score.scored_token_count, 1)
        self.assertEqual(score.prompt_token_count, 1)
        self.assertEqual(score.target_byte_length, 1)
        rows = self.backend.forward_logits((0, 2)).tolist()
        # Independent reference over the predicting row (row 0).
        exp_per, exp_total, exp_mean = reference_aggregate(rows[0:1], (2,))
        self.assertAlmostEqual(score.per_token_logprobs[0], exp_per[0], places=6)
        self.assertAlmostEqual(score.total_logprob, exp_total, places=6)
        self.assertAlmostEqual(score.mean_logprob, exp_mean, places=6)

    def test_real_backend_multi_token_target(self):
        # prompt "a" -> (0,), target "cb" -> (2, 1): "acb" encodes (0, 2, 1).
        score = score_target_logprobs(self.backend, (0,), (2, 1))
        self.assertEqual(score.status, STATUS_OK)
        self.assertEqual(score.scored_token_count, 2)
        self.assertEqual(score.target_byte_length, 2)
        rows = self.backend.forward_logits((0, 2, 1)).tolist()
        exp_per, exp_total, exp_mean = reference_aggregate(rows[0:2], (2, 1))
        self.assertEqual(len(score.per_token_logprobs), 2)
        for got, want in zip(score.per_token_logprobs, exp_per, strict=True):
            self.assertAlmostEqual(got, want, places=6)
        self.assertAlmostEqual(score.total_logprob, exp_total, places=6)
        self.assertAlmostEqual(score.mean_logprob, exp_mean, places=6)

    def test_real_backend_boundary_merge_refuses(self):
        # "a" -> (0,), "b" -> (1,), but "ab" -> (3,): straddling token.
        score = score_target_logprobs(self.backend, (0,), (1,))
        self.assertEqual(score.status, STATUS_UNAVAILABLE)
        self.assertEqual(score.reason, "boundary_merge_mismatch")
        self.assertIsNone(score.total_logprob)

    def test_real_backend_empty_target(self):
        score = score_target_logprobs(self.backend, (0,), ())
        self.assertEqual(score.status, STATUS_OK_EMPTY)
        self.assertEqual(score.total_logprob, 0.0)

    def test_real_backend_deterministic(self):
        first = score_target_logprobs(self.backend, (0,), (2, 1))
        second = score_target_logprobs(self.backend, (0,), (2, 1))
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
