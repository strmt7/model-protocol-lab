"""Prefix-cache sharability per template family (research wave-6 idea 2).

The question: when two templates tie on accuracy, which one's prompts
share more exact token prefixes -- i.e. which one would let a serving
stack's exact-prefix KV cache (vLLM automatic prefix caching) skip more
prefill work? The lab already records exact ``input_ids`` per
observation, so the measurement is purely post-hoc: zero model calls,
no backend, no timing (sharability is not latency; see scope below).

Definitions (pinned; red-team reviewed, cycle 168). Let P be one
candidate's evaluable prompts (see :func:`is_evaluable_prompt`).

- A *shared prefix* is a token sequence that is a prefix of at least
  two prompts in P. Token position k of a prompt is *cacheable* when
  the prompt's prefix of length k+1 is shared.
- ``potential`` counts cacheable tokens: an upper bound on realizable
  prefix-cache hits (every realizable hit's prefix was seen in an
  earlier request, hence is shared). The bound is eviction-robust (LRU
  only removes hits) but loose by exactly one first occurrence per
  shared prefix.
- ``achievable`` is exact, not a bound. For each shared prefix v with
  k_v sharing prompts, *every* request order (a permutation of P, each
  prompt served once, cold infinite dedicated cache) yields exactly
  k_v - 1 hits and 1 miss at v: the earliest sharer cannot have seen
  the prefix; every later sharer has. Summing over shared prefixes,
  hits = potential - (#shared prefixes), order-free. Red-team A found
  this after showing "potential" alone *misranks* templates against
  every realizable order (fragmented sharing inflates potential more);
  red-team B failed to break the formula (300 random prompt sets, all
  orders, zero mismatches).
- Block level models vLLM APC: blocks of ``block_size`` tokens
  (default 16) with chained content hashes -- block b of a prompt is
  reusable iff the whole prefix through block b was seen before, i.e.
  iff that block-length prefix is shared. Partial trailing blocks are
  never cached (vLLM design doc: "We only cache full blocks"), so they
  never count. The same per-identity argument gives exact achievable
  block hits = block occurrences - (#distinct shared block prefixes).

Pre-registered claims (falsifiable):

1. Sharability differs across candidates: the max-minus-min
   ``sharability_gap`` on the achievable block fraction is either
   exactly 0 (uniform -- the null hypothesis, reported, never
   asserted) or measurably positive, i.e. the template, not the model
   or the fixture set, decides how much prefill a prefix cache could
   skip.
2. The block-quantized number is the headline; the token-level number
   is a diagnostic that ignores block quantization and is always >=
   the block number at both levels (potential and achievable).

Scope limits (stated, not hidden): this measures prefix *sharability*,
a cost input -- not latency, not TTFT, not a production cost verdict
(absolute token counts, batching, HBM pressure, and decode length are
not modeled). The closed-world reading of "achievable" is a
permutation of the recorded prompts, each served once, into a cold,
infinite, dedicated cache; production interleaving, cross-workload
warming, repeated requests, and eviction can move real hits either
way relative to it (warming/repeats can exceed even "potential").
Model assumptions (vLLM fidelity): real block hashes also fold in
adapter ids, multimodal input hashes, and optional cache salts -- the
pure-token identity holds for single-model, no-adapter, unsalted
deployments; hash collisions are ignored; sliding-window-attention
models bound reusable prefix length (out of scope); and vLLM caps a
fully-cached prompt's hits at prompt_length - 1 (logits recomputation),
which with block-aligned allocation can cost one whole final block on
an exactly-duplicated block-aligned prompt -- so real hits are at most
the model's, by up to one block in that case only. Informational only:
never touches winner selection.
"""

from __future__ import annotations

from typing import Any

#: Pinned default block size: vLLM automatic prefix caching hashes and
#: caches full blocks of 16 tokens. A parameter (other engines differ),
#: recorded in every output so a number is reproducible.
DEFAULT_BLOCK_SIZE = 16


def is_evaluable_prompt(value: object) -> bool:
    """Whether ``value`` is a measurable prompt token sequence.

    Pinned predicate (cycle-168 red-team R10/R13): a list or tuple
    (reports serialize ``input_ids`` as lists; in-memory observations
    carry tuples), non-empty, whose elements are all exact ``int``
    (bools are not token ids) and non-negative. Anything else is
    excluded fail-closed by the aggregation -- visible via ``n`` vs
    ``n_evaluable``, never coerced and never crashing it.
    """
    sequence: Any = value
    if type(sequence) not in (list, tuple) or len(sequence) == 0:
        return False
    return all(type(token) is int and token >= 0 for token in sequence)


def _prefix_counts(prompts: list[tuple[int, ...]]) -> dict[tuple[int, ...], int]:
    """Trie counts: how many prompts pass through each prefix node."""
    counts: dict[tuple[int, ...], int] = {}
    for prompt in prompts:
        for end in range(1, len(prompt) + 1):
            prefix = prompt[:end]
            counts[prefix] = counts.get(prefix, 0) + 1
    return counts


def candidate_cache_profile(
    records: list[Any],
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> dict[str, Any]:
    """Aggregate one candidate's prefix-cache sharability.

    ``records`` are raw prompt values (any type); non-evaluable ones
    are excluded fail-closed (see :func:`is_evaluable_prompt`) and stay
    visible via ``n`` vs ``n_evaluable``. ``block_size`` is a caller
    contract: a non-positive or non-integer value raises
    ``ValueError``.

    Returns the cell dict: counts (``n``, ``n_evaluable``,
    ``total_tokens``, ``cacheable_tokens``, ``shared_prefixes``,
    ``achievable_tokens``, ``block_cacheable_blocks``,
    ``distinct_block_prefixes``, ``achievable_blocks``,
    ``n_duplicate_prompts``) and fractions (``mean_prompt_tokens``,
    ``potential_token_fraction``, ``achievable_token_fraction``,
    ``potential_block_fraction``, ``achievable_block_fraction``).
    Fractions and the mean are ``None`` when nothing is evaluable --
    never 0.0, which would falsely assert "no sharing" (cycle-168
    red-team R5).
    """
    if type(records) is not list:
        raise ValueError(f"records must be a list, got {type(records).__name__}")
    if type(block_size) is not int or block_size < 1:
        raise ValueError(f"block_size must be a positive int, got {block_size!r}")
    n = len(records)
    prompts = [tuple(record) for record in records if is_evaluable_prompt(record)]
    n_evaluable = len(prompts)
    if n_evaluable == 0:
        return {
            "n": n,
            "n_evaluable": 0,
            "total_tokens": 0,
            "mean_prompt_tokens": None,
            "cacheable_tokens": 0,
            "shared_prefixes": 0,
            "achievable_tokens": 0,
            # nosec B105 -- dict defaults are report-schema fields, never credentials
            "potential_token_fraction": None,  # nosec B105
            "achievable_token_fraction": None,  # nosec B105
            "block_cacheable_blocks": 0,
            "distinct_block_prefixes": 0,
            "achievable_blocks": 0,
            "potential_block_fraction": None,
            "achievable_block_fraction": None,
            "n_duplicate_prompts": 0,
        }
    counts = _prefix_counts(prompts)
    total_tokens = sum(len(prompt) for prompt in prompts)
    cacheable_tokens = 0
    for prompt in prompts:
        for end in range(1, len(prompt) + 1):
            if counts[prompt[:end]] >= 2:
                cacheable_tokens += 1
    shared_prefixes = sum(1 for sharers in counts.values() if sharers >= 2)
    # Exact per the module docstring: every shared prefix contributes
    # exactly one unavoidable first-occurrence miss in every order.
    achievable_tokens = cacheable_tokens - shared_prefixes
    block_cacheable_blocks = 0
    distinct_block_prefixes = 0
    for prefix, sharers in counts.items():
        if sharers >= 2 and len(prefix) % block_size == 0:
            distinct_block_prefixes += 1
            block_cacheable_blocks += sharers
    achievable_blocks = block_cacheable_blocks - distinct_block_prefixes
    n_duplicate_prompts = sum(1 for prompt in prompts if counts[prompt] >= 2)
    return {
        "n": n,
        "n_evaluable": n_evaluable,
        "total_tokens": total_tokens,
        "mean_prompt_tokens": total_tokens / n_evaluable,
        "cacheable_tokens": cacheable_tokens,
        "shared_prefixes": shared_prefixes,
        "achievable_tokens": achievable_tokens,
        "potential_token_fraction": cacheable_tokens / total_tokens,
        "achievable_token_fraction": achievable_tokens / total_tokens,
        "block_cacheable_blocks": block_cacheable_blocks,
        "distinct_block_prefixes": distinct_block_prefixes,
        "achievable_blocks": achievable_blocks,
        "potential_block_fraction": block_size * block_cacheable_blocks / total_tokens,
        "achievable_block_fraction": block_size * achievable_blocks / total_tokens,
        "n_duplicate_prompts": n_duplicate_prompts,
    }
