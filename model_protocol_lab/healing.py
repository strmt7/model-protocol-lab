r"""Token healing for greedy tokenization bias (research wave-2 idea 3).

Ports the *mechanism* of Hugging Face transformers'
``GenerationMixin.heal_tokens`` (``generation/utils.py``) into a
backend-agnostic, pure core:

1. Decode the prompt ids and re-encode the stripped text (re-canonicalization
   -- the step that repairs prompts affected by greedy tokenization bias).
2. Take the tail token; find every vocabulary token whose token-text starts
   with the tail token's text (whitespace represented as its tokenization,
   e.g. ``'\u0120'`` for GPT-2 -- exactly HF's ``ExtensionsTrie`` prefix
   search, here as a deterministic linear scan).
3. Trim the tail token and greedily generate one replacement token with a
   sequence bias of +10.0 on every extension and +1.0 on the original tail
   token (HF's constants), i.e. the model re-chooses the tail among tokens
   that extend the decoded tail text.

HF's documented contract (``GenerationConfig.token_healing``): "Heal tail
tokens of prompts by replacing them with their appropriate extensions. This
enhances the quality of completions for prompts affected by greedy
tokenization bias."

Deliberate adaptations from HF's ``heal_tokens`` (documented, not silent):

- HF decodes with ``skip_special_tokens=True`` because its pipeline re-adds
  specials afterwards. The lab's prompts are exact operator-constructed id
  sequences, so healing must preserve every id: decoding keeps specials
  (``skip_special_tokens=False`` in the backend adapter).
- HF substitutes BOS when the trimmed prompt is empty (its pipeline always
  has specials to fall back on). The lab has no implicit specials, so an
  empty trimmed prompt is a healing failure, not a BOS substitution -- the
  caller falls back to the native prompt and the fallback is counted.
- The whitespace prefix trick needs the tokenized form of ``" "``; when the
  vocabulary has no space token the search runs on the raw tail text.

Everything here is pure: tokenization, decoding, and the biased argmax are
injected callables, so the whole mechanism is testable without a model.
"""

from __future__ import annotations

from collections.abc import Callable

#: Sequence-bias added to every tail-text extension (HF's constant).
HEAL_EXTENSION_BIAS = 10.0
#: Extra bias on the original tail token, limiting aggressive healing
#: (HF's constant; e.g. keeps 'http' from healing to 'https').
HEAL_ORIGINAL_BIAS = 1.0


def normalize_tail_text(tail_text: str, space_token_text: str | None) -> str:
    r"""Map real spaces to their tokenization for the trie prefix search.

    HF replaces ``" "`` with the token text of the space token (e.g.
    ``'\u0120'``) so the prefix search finds tokens *prefixed with a
    whitespace*. No space token (or no spaces in the tail) leaves the text
    untouched. Non-string input is a programming error, not a runtime
    condition: raise.
    """
    if type(tail_text) is not str:
        raise ValueError("tail_text must be a string")
    if type(space_token_text) is str and space_token_text and " " in tail_text:
        return tail_text.replace(" ", space_token_text)
    return tail_text


def tail_extensions(vocab: dict[str, int], prefix: str) -> tuple[int, ...]:
    """Token ids whose token-text starts with ``prefix`` (sorted).

    The deterministic stand-in for HF's ``ExtensionsTrie.extensions``:
    a linear scan over the ``token -> id`` vocabulary. The tail token
    itself always matches (its text starts with itself), so the result
    has length >= 1 whenever the tail token is in the vocabulary.
    """
    if type(vocab) is not dict or type(prefix) is not str:
        raise ValueError("vocab must be a token->id dict and prefix a string")
    found: list[int] = []
    for text, token_id in vocab.items():
        if type(text) is not str or type(token_id) is not int:
            raise ValueError("vocab must map strings to integer ids")
        if text.startswith(prefix):
            found.append(token_id)
    return tuple(sorted(found))


def healing_biases(extensions: tuple[int, ...], tail_id: int) -> dict[int, float]:
    """Build the HF sequence-bias map for one healing step.

    Every extension gets ``HEAL_EXTENSION_BIAS``; the original tail token
    additionally gets ``HEAL_ORIGINAL_BIAS``. Mirrors HF's
    ``seq_bias = {(ext,): 10.0 ...}; seq_bias[(tail_id,)] += 1.0``.
    """
    if type(extensions) is not tuple or not extensions:
        raise ValueError("extensions must be a nonempty tuple of token ids")
    if type(tail_id) is not int:
        raise ValueError("tail_id must be an integer")
    biases = dict.fromkeys(extensions, HEAL_EXTENSION_BIAS)
    biases[tail_id] = biases.get(tail_id, 0.0) + HEAL_ORIGINAL_BIAS
    return biases


def heal_prompt_ids(
    input_ids: tuple[int, ...],
    *,
    decode_prompt: Callable[[tuple[int, ...]], str],
    decode_token: Callable[[int], str],
    reencode: Callable[[str], tuple[int, ...]],
    vocab: dict[str, int],
    space_token_text: str | None,
    biased_argmax: Callable[[tuple[int, ...], dict[int, float]], int],
) -> tuple[int, ...]:
    """Heal the tail of ``input_ids`` following HF's ``heal_tokens``.

    Returns the healed prompt ids ``trimmed + (healed_token,)``. Raises
    ``ValueError`` with a fixed code on any failure (empty input,
    undecodable tail, empty re-encoding, no usable extensions, empty
    trimmed prompt, bad argmax output): the caller falls back to the
    native prompt and counts the fallback -- healing never silently
    degrades into a phantom prompt.
    """
    if (
        type(input_ids) is not tuple
        or not input_ids
        or any(type(t) is not int or t < 0 for t in input_ids)
    ):
        raise ValueError("heal_invalid_input_ids")
    try:
        prompt_text = decode_prompt(input_ids)
    except Exception as exc:  # noqa: BLE001 -- injected callable; fail closed
        raise ValueError("heal_decode_failed") from exc
    if type(prompt_text) is not str:
        raise ValueError("heal_decode_failed")
    try:
        canonical = reencode(prompt_text.strip())
    except Exception as exc:  # noqa: BLE001 -- injected callable; fail closed
        raise ValueError("heal_reencode_failed") from exc
    if (
        type(canonical) is not tuple
        or not canonical
        or any(type(t) is not int or t < 0 for t in canonical)
    ):
        raise ValueError("heal_reencode_failed")
    tail_id = canonical[-1]
    try:
        tail_text = decode_token(tail_id)
    except Exception as exc:  # noqa: BLE001 -- injected callable; fail closed
        raise ValueError("heal_tail_decode_failed") from exc
    prefix = normalize_tail_text(tail_text, space_token_text)
    extensions = tail_extensions(vocab, prefix)
    if len(extensions) <= 1:
        # Only the tail token itself extends the tail text: no
        # alternatives to heal with. Mirrors HF's skip; the
        # re-canonicalized prompt is still the honest return value.
        return canonical
    biases = healing_biases(extensions, tail_id)
    trimmed = canonical[:-1]
    if not trimmed:
        # Deliberate divergence from HF (documented in the module
        # docstring): no implicit specials to fall back on.
        raise ValueError("heal_empty_trimmed_prompt")
    try:
        healed_token = biased_argmax(trimmed, biases)
    except Exception as exc:  # noqa: BLE001 -- injected callable; fail closed
        raise ValueError("heal_argmax_failed") from exc
    if type(healed_token) is not int or healed_token < 0:
        raise ValueError("heal_argmax_failed")
    return trimmed + (healed_token,)
