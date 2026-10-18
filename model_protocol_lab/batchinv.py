"""Batch-invariance audit (research wave-3 idea 2).

The question: the lab's core invariant is "send exact token sequences,
compare exact outcomes" -- but GPU inference is not batch invariant.
Thinking Machines ("Defeating Nondeterminism in LLM Inference", Oct
2026) documents that kernel selection (GEMM vs GEMV), attention
reduction order, and KV-cache layout change per batch, so the *same
prompt* can produce different logits depending on what else ran
alongside it. Every prompt-format harness treats the serving path as a
fixed point; the lab sends exact token sequences, so batch composition
is a hidden confounder that could poison every comparison it publishes.
This module is the audit: greedy decoding of an identical token sequence
in isolation vs co-batched must produce byte-identical continuation
token ids.

Pre-registered claims (falsifiable):

1. No backend in the current tree exposes a batch path, so on the real
   local backend the stage records ``batch_unsupported``. The audit
   deliberately does NOT synthesize a batch out of sequential calls --
   that would be theater, not measurement.
2. Given a batch-capable backend, the comparison below detects any
   token-level divergence (proved by the brute-force oracle and the
   hostile-backend tests in ``test_cycle153_batchinv.py``).
3. The core falsifiable claim: for a batch-capable backend, solo and
   batched greedy decoding of identical token sequences produce
   byte-identical continuation ids. Any nonzero divergence rate
   falsifies the determinism assumption *for that backend* -- and is a
   result in itself.

Optional backend batch-path contract (probed, never required): a
backend exposes batch support by defining::

    def generate_batch(
        self,
        prompts: tuple[tuple[int, ...], ...],
        *,
        max_new_tokens: int,
        seed: int,
        cancel: Event,
    ) -> tuple[tuple[int, ...], ...]: ...

returning continuation-only token ids, one tuple per prompt, in order.
Backends without this attribute record ``batch_unsupported``; a batch
path that violates the contract (wrong arity, non-id values, raising)
records ``batch_contract_failed`` / ``batch_failed`` -- fail-closed,
never a crash and never silent data.

Everything here is pure: prompt validation and the solo-vs-batched
comparison need no model.
"""

from __future__ import annotations

from collections.abc import Mapping

from .core import _valid_token_ids

#: Detail code recorded when the backend exposes no batch path.
BATCH_UNSUPPORTED = "batch_unsupported"

#: Detail code recorded when the batch path violates its contract.
BATCH_CONTRACT_FAILED = "batch_contract_failed"

#: Detail code recorded when the batch path raises.
BATCH_FAILED = "batch_failed"

#: Verdict vocabulary for an available audit payload.
VERDICTS = ("batch_invariant", "batch_variant", "unavailable")


def validate_prompt_ids(prompts: object) -> dict[str, tuple[int, ...]]:
    """Validate the audit's prompt map ``{prompt_id: input_ids}``.

    Returns a plain ``dict`` copy. Raises ``ValueError`` on a
    non-dict, an empty dict, a non-string or blank key, or any
    input-id tuple that fails :func:`core._valid_token_ids`
    (nonempty required: an empty prompt is not an auditable
    sequence).
    """
    if type(prompts) is not dict or not prompts:
        raise ValueError("prompts must be a nonempty dict")
    clean: dict[str, tuple[int, ...]] = {}
    for key, ids in prompts.items():
        if type(key) is not str or not key.strip():
            raise ValueError("prompt ids must be keyed by nonempty strings")
        if not _valid_token_ids(ids, nonempty=True):
            raise ValueError(f"prompt {key!r} has invalid input ids")
        clean[key] = ids
    return clean


def compare_outputs(
    solo_ids: tuple[int, ...], batched_ids: tuple[int, ...]
) -> dict[str, int | bool | None]:
    """Compare one prompt's solo vs batched continuation ids.

    Returns ``{"diverged": bool, "first_divergence": int | None,
    "solo_len": int, "batched_len": int}``. ``first_divergence`` is the
    first index where the sequences differ; when one is a strict prefix
    of the other it is the length of the shorter sequence; it is
    ``None`` exactly when the sequences are identical. Raises
    ``ValueError`` on malformed id tuples.
    """
    if not _valid_token_ids(solo_ids) or not _valid_token_ids(batched_ids):
        raise ValueError("id sequences must be tuples of nonnegative ints")
    first: int | None = None
    for index, (solo_tok, batched_tok) in enumerate(zip(solo_ids, batched_ids, strict=False)):
        if solo_tok != batched_tok:
            first = index
            break
    else:
        if len(solo_ids) != len(batched_ids):
            first = min(len(solo_ids), len(batched_ids))
    return {
        "diverged": first is not None,
        "first_divergence": first,
        "solo_len": len(solo_ids),
        "batched_len": len(batched_ids),
    }


def compare_solo_vs_batched(
    solo: Mapping[str, tuple[int, ...]], batched: Mapping[str, tuple[int, ...]]
) -> dict[str, object]:
    """Compare full solo vs batched output maps, token-for-token.

    Both maps must cover exactly the same prompt ids. Returns
    ``{"n_prompts": int, "n_diverged": int, "divergence_rate": float,
    "per_prompt": {pid: compare_outputs(...)} }``. Raises
    ``ValueError`` on key-set mismatch or malformed values.
    """
    if not isinstance(solo, Mapping) or not isinstance(batched, Mapping):
        raise ValueError("solo and batched must be mappings")
    if set(solo) != set(batched):
        raise ValueError("solo and batched must cover the same prompt ids")
    per_prompt: dict[str, dict[str, int | bool | None]] = {}
    n_diverged = 0
    for pid in sorted(solo, key=str):
        entry = compare_outputs(solo[pid], batched[pid])
        per_prompt[str(pid)] = entry
        if entry["diverged"]:
            n_diverged += 1
    n_prompts = len(per_prompt)
    return {
        "n_prompts": n_prompts,
        "n_diverged": n_diverged,
        "divergence_rate": n_diverged / n_prompts if n_prompts else 0.0,
        "per_prompt": per_prompt,
    }


def audit_verdict(n_evaluable: int, n_diverged: int) -> str:
    """Verdict for an audit payload.

    ``"batch_invariant"`` when every evaluable prompt matched exactly,
    ``"batch_variant"`` on any divergence, ``"unavailable"`` when
    nothing was evaluable. Raises ``ValueError`` on bad counts.
    """
    if type(n_evaluable) is not int or n_evaluable < 0:
        raise ValueError("n_evaluable must be a nonnegative int")
    if type(n_diverged) is not int or n_diverged < 0 or n_diverged > n_evaluable:
        raise ValueError("n_diverged must be within [0, n_evaluable]")
    if n_evaluable == 0:
        return "unavailable"
    return "batch_variant" if n_diverged else "batch_invariant"
