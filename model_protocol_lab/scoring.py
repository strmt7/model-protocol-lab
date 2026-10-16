"""Teacher-forced target log-likelihood scoring (§9.3-§9.5).

This module is the pure scoring engine: given a full token sequence split
into a prompt span and a target (answer) span, it runs one forward pass and
gathers the causal-shifted log-probability of every target token. The logits
at position ``t - 1`` predict the token at position ``t``; labels are shifted
once and prompt positions are masked (never scored).

Declared masking policy: every token ID in the target span is scored exactly
once -- including pad-like or special IDs. The engine performs no implicit
unmasking; callers that pad must exclude padding from ``target_ids``
themselves. Prompt positions are never included in an answer-only metric.

Alignment (§9.3): the engine never assumes
``encode(prompt) + encode(answer) == encode(prompt + answer)``. Before any
forward pass it verifies, at the token-ID level, that the prompt span and the
target span each decode/re-encode to themselves and that encoding the
concatenated text reproduces the concatenated IDs. A token straddling the
prompt/answer boundary -- or any concatenation mismatch -- yields an
explicit ``unavailable`` status, never a failure and never a silent guess.

Score comparability (§9.4): every score reports the total log-likelihood,
the mean per scored token, the scored token count, and the target byte
length, so token counts that differ across candidates stay visible. NaN and
infinity are detected before aggregation, token IDs are validated against
the vocabulary, accumulation runs in float64, and reported values are
rounded to float32. Scoring is deterministic for a fixed logit tensor.

``TargetScore.status`` is one of ``"ok"``, ``"ok_empty"`` (empty target: a
defined zero-token score, not an error), or ``"unavailable"`` (explicit
refusal with a fixed ``reason`` code).
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Any, Literal

from .core import Backend

# Largest finite float32 value (exact): finite float64 magnitudes above
# this round to +/-inf under round-to-nearest-even.
_FLT_MAX = float.fromhex("0x1.fffffep+127")

ScoreStatus = Literal["ok", "ok_empty", "unavailable"]
"""Lifecycle of a target score: scored, defined-empty, or explicitly refused."""

STATUS_OK: ScoreStatus = "ok"
STATUS_OK_EMPTY: ScoreStatus = "ok_empty"
STATUS_UNAVAILABLE: ScoreStatus = "unavailable"

# Reason codes are a fixed vocabulary: backend/dependency text must never
# reach them. Codes produced by the engine itself are listed here; codes
# raised by a logits provider pass through only when they belong to the
# whitelisted provider vocabulary below.
_REASON_ALIGNMENT = frozenset(
    {
        "decode_failed",
        "encode_failed",
        "prompt_roundtrip_mismatch",
        "target_roundtrip_mismatch",
        "boundary_merge_mismatch",
    }
)
_REASON_PROVIDER = frozenset(
    {
        # LocalHFBackend.forward_logits fixed codes.
        "invalid_input_ids",
        "context_limit_exceeded",
        "local_forward_failed",
        "local_backend_not_initialized",
        # score_span_from_logits rejection codes.
        "logits_must_be_tensor",
        "logits_must_be_2d",
        "prompt_len_must_be_positive",
        "invalid_target_ids",
        "empty_vocabulary",
        "logits_length_mismatch",
        "token_id_out_of_range",
        "non_finite_logits",
    }
)
_REASON_ENGINE = frozenset(
    {
        "empty_prompt",
        "logits_not_supported",
        "forward_failed",
    }
)
# The three vocabularies are disjoint by construction: alignment codes name
# tokenizer-span problems, provider codes name forward/math rejections, and
# engine codes name orchestration refusals.


@dataclass(frozen=True)
class TargetScore:
    """Result of scoring one target span.

    ``total_logprob`` is the float64-accumulated sum of the per-token
    log-probabilities, reported float32-rounded; ``mean_logprob`` is the
    total divided by ``scored_token_count`` (None when zero tokens were
    scored -- an empty denominator is reported, not divided). For
    ``unavailable`` scores the likelihood fields are None: an invalid or
    unavailable score is distinct from a legitimate score of zero.
    ``target_byte_length`` is the UTF-8 byte length of the decoded target
    text. ``reason`` is None unless the status is ``unavailable``.
    """

    status: ScoreStatus
    reason: str | None
    per_token_logprobs: tuple[float, ...]
    total_logprob: float | None
    mean_logprob: float | None
    scored_token_count: int
    target_byte_length: int
    prompt_token_count: int


@dataclass(frozen=True)
class AlignmentCheck:
    """Outcome of the §9.3 token-ID alignment verification.

    ``ok`` is True only when the prompt span and the target span each
    decode/re-encode to themselves and the concatenated text encodes to the
    concatenated IDs. ``prompt_text``/``target_text`` are the decoded spans
    (empty when decoding itself failed).
    """

    ok: bool
    reason: str | None
    prompt_text: str
    target_text: str


def _f32(value: float) -> float:
    """Round a float64 value to the nearest float32, returned as float.

    Finite values outside the float32 range saturate to +/-inf, matching
    IEEE-754 round-to-nearest-even and torch/numpy float32 conversion
    semantics. (struct.pack alone would raise OverflowError here, which the
    scoring pipeline would then mislabel as ``forward_failed`` even though
    the score is computable.) NaN and the infinities pass through unchanged.
    """
    if math.isfinite(value) and abs(value) > _FLT_MAX:
        return math.copysign(math.inf, value)
    rounded: float = struct.unpack("<f", struct.pack("<f", value))[0]
    return rounded


def _require_torch() -> Any:
    """Import torch lazily: scoring stays importable without the hf extra."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch_not_installed") from exc
    return torch


def score_span_from_logits(
    logits: Any,
    *,
    prompt_len: int,
    target_ids: tuple[int, ...],
) -> tuple[tuple[float, ...], float, float | None]:
    """Gather causal-shifted target log-probabilities from a logit tensor.

    ``logits`` is a 2-D tensor of shape ``(T, V)`` holding one forward
    pass over ``prompt_ids + target_ids`` (so ``T`` must equal
    ``prompt_len + len(target_ids)``). The log-probability of target token
    ``i`` is read from row ``prompt_len - 1 + i`` -- the causal shift;
    prompt rows are masked by never being gathered.

    Returns ``(per_token_logprobs, total_logprob, mean_logprob)``: per-token
    values in target order, the float64-accumulated total, and the mean per
    scored token (None for an empty target span). Reported values are
    float32-rounded. Rejections raise ValueError with a fixed code; nothing
    is silently masked or guessed. The one exception is a non-tensor
    ``logits``, which raises ``TypeError("logits_must_be_tensor")`` --
    wrong-type input is a caller bug, not a rejected measurement.
    """
    torch = _require_torch()
    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits_must_be_tensor")
    if logits.ndim != 2:
        raise ValueError("logits_must_be_2d")
    if type(prompt_len) is not int or prompt_len < 1:
        raise ValueError("prompt_len_must_be_positive")
    if type(target_ids) is not tuple or any(type(t) is not int for t in target_ids):
        raise ValueError("invalid_target_ids")
    seq_len = int(logits.shape[0])
    vocab_size = int(logits.shape[1])
    if vocab_size < 1:
        raise ValueError("empty_vocabulary")
    if seq_len != prompt_len + len(target_ids):
        raise ValueError("logits_length_mismatch")
    if any(t < 0 or t >= vocab_size for t in target_ids):
        raise ValueError("token_id_out_of_range")
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError("non_finite_logits")
    work = logits.to(dtype=torch.float64)
    per_token: list[float] = []
    total = 0.0
    for offset, token_id in enumerate(target_ids):
        row = work[prompt_len - 1 + offset]
        peak = torch.max(row)
        log_normalizer = torch.log(torch.sum(torch.exp(row - peak)))
        logprob = float((row[token_id] - peak - log_normalizer).item())
        per_token.append(_f32(logprob))
        total += logprob
    mean = _f32(total / len(target_ids)) if target_ids else None
    return tuple(per_token), _f32(total), mean


def _safe_decode(backend: Backend, token_ids: tuple[int, ...]) -> str | None:
    try:
        text = backend.decode(token_ids)
    except Exception:  # noqa: BLE001 -- decode problems become fixed codes
        return None
    return text if type(text) is str else None


def _safe_encode(backend: Backend, text: str) -> tuple[int, ...] | None:
    try:
        ids = backend.encode(text)
    except Exception:  # noqa: BLE001 -- encode problems become fixed codes
        return None
    if type(ids) is not tuple or any(type(t) is not int for t in ids):
        return None
    return ids


def check_alignment(
    backend: Backend, prompt_ids: tuple[int, ...], target_ids: tuple[int, ...]
) -> AlignmentCheck:
    """Verify the prompt/target split at the token-ID level (§9.3).

    Decodes both spans, requires each to re-encode to itself (catching
    lossy decodes and tokenizer-added tokens), and requires the
    concatenated text to encode to the concatenated IDs (catching
    boundary-crossing merges: a token straddling prompt and answer text
    re-tokenizes differently and fails this check). Any failure yields
    ``ok=False`` with a fixed reason code.
    """
    prompt_text = _safe_decode(backend, prompt_ids)
    if prompt_text is None:
        return AlignmentCheck(False, "decode_failed", "", "")
    target_text = _safe_decode(backend, target_ids)
    if target_text is None:
        return AlignmentCheck(False, "decode_failed", prompt_text, "")
    re_prompt = _safe_encode(backend, prompt_text)
    if re_prompt is None:
        return AlignmentCheck(False, "encode_failed", prompt_text, target_text)
    if re_prompt != prompt_ids:
        return AlignmentCheck(False, "prompt_roundtrip_mismatch", prompt_text, target_text)
    re_target = _safe_encode(backend, target_text)
    if re_target is None:
        return AlignmentCheck(False, "encode_failed", prompt_text, target_text)
    if re_target != target_ids:
        return AlignmentCheck(False, "target_roundtrip_mismatch", prompt_text, target_text)
    combined = _safe_encode(backend, prompt_text + target_text)
    if combined is None:
        return AlignmentCheck(False, "encode_failed", prompt_text, target_text)
    if combined != prompt_ids + target_ids:
        return AlignmentCheck(False, "boundary_merge_mismatch", prompt_text, target_text)
    return AlignmentCheck(True, None, prompt_text, target_text)


def _unavailable(reason: str, target_byte_length: int, prompt_token_count: int) -> TargetScore:
    return TargetScore(
        status=STATUS_UNAVAILABLE,
        reason=reason,
        per_token_logprobs=(),
        total_logprob=None,
        mean_logprob=None,
        scored_token_count=0,
        target_byte_length=target_byte_length,
        prompt_token_count=prompt_token_count,
    )


def _provider_code(exc: BaseException) -> str:
    """Map a provider/math exception to a fixed reason code.

    Only whitelisted fixed-vocabulary codes pass through; anything else --
    including arbitrary backend text -- collapses to ``forward_failed``.
    """
    if exc.args and type(exc.args[0]) is str and exc.args[0] in _REASON_PROVIDER:
        return exc.args[0]
    return "forward_failed"


def score_target_logprobs(
    backend: Backend, prompt_ids: tuple[int, ...], target_ids: tuple[int, ...]
) -> TargetScore:
    """Score a target span by teacher forcing against ``backend`` (§9.3-§9.5).

    Runs one forward pass over ``prompt_ids + target_ids`` and returns the
    causal-shifted log-probability of every target token. The backend must
    expose ``forward_logits(input_ids)`` returning a ``(T, V)`` fp32 logit
    tensor (as :class:`LocalHFBackend.forward_logits` does); backends
    without it yield ``unavailable``/``logits_not_supported``.

    Empty ``target_ids`` score zero tokens with status ``ok_empty``: a
    defined outcome, not an error. An empty prompt is refused
    (``empty_prompt``): the first target token would have no predicting
    position. Alignment failures, forward failures, and invalid logits
    yield ``unavailable`` with a fixed reason -- never an exception and
    never a silent guess. Malformed caller input (non-tuple or non-int IDs)
    raises TypeError/ValueError instead, since that is a programming error
    rather than a property of the fixture.
    """
    if type(prompt_ids) is not tuple or type(target_ids) is not tuple:
        raise TypeError("token_id_sequences_must_be_tuples")
    for ids in (prompt_ids, target_ids):
        if any(type(t) is not int or t < 0 for t in ids):
            raise ValueError("invalid_token_ids")
    if not target_ids:
        return TargetScore(
            status=STATUS_OK_EMPTY,
            reason=None,
            per_token_logprobs=(),
            total_logprob=0.0,
            mean_logprob=None,
            scored_token_count=0,
            target_byte_length=0,
            prompt_token_count=len(prompt_ids),
        )
    if not prompt_ids:
        return _unavailable("empty_prompt", 0, 0)
    vocab_size = getattr(backend, "vocab_size", None)
    if (
        type(vocab_size) is int
        and vocab_size >= 1
        and any(t >= vocab_size for t in prompt_ids + target_ids)
    ):
        return _unavailable("token_id_out_of_range", 0, len(prompt_ids))
    alignment = check_alignment(backend, prompt_ids, target_ids)
    target_byte_length = len(alignment.target_text.encode("utf-8"))
    if not alignment.ok:
        reason = alignment.reason if alignment.reason in _REASON_ALIGNMENT else "forward_failed"
        return _unavailable(reason, target_byte_length, len(prompt_ids))
    provider = getattr(backend, "forward_logits", None)
    if not callable(provider):
        return _unavailable("logits_not_supported", target_byte_length, len(prompt_ids))
    try:
        logits = provider(prompt_ids + target_ids)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:  # noqa: BLE001 -- provider failures become fixed codes
        # As in core.py: a misbehaving logits provider raising BaseException
        # becomes one unavailable score, not a lost run. Interrupts propagate.
        return _unavailable(_provider_code(exc), target_byte_length, len(prompt_ids))
    try:
        per_token, total, mean = score_span_from_logits(
            logits, prompt_len=len(prompt_ids), target_ids=target_ids
        )
    except Exception as exc:  # noqa: BLE001 -- math rejections become fixed codes
        return _unavailable(_provider_code(exc), target_byte_length, len(prompt_ids))
    return TargetScore(
        status=STATUS_OK,
        reason=None,
        per_token_logprobs=per_token,
        total_logprob=total,
        mean_logprob=mean,
        scored_token_count=len(target_ids),
        target_byte_length=target_byte_length,
        prompt_token_count=len(prompt_ids),
    )


__all__ = [
    "AlignmentCheck",
    "ScoreStatus",
    "STATUS_OK",
    "STATUS_OK_EMPTY",
    "STATUS_UNAVAILABLE",
    "TargetScore",
    "check_alignment",
    "score_span_from_logits",
    "score_target_logprobs",
]
