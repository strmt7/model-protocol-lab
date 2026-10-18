"""Assistant-prefill sensitivity experiment (research wave-2 idea 2).

The question: does the winning protocol survive when the caller pre-fills
the assistant's opening tokens -- and which template families honor vs.
ignore prefills?

Prefill is a first-class generation control (Anthropic's Messages API and
AWS Bedrock document prefilling the assistant's response; the attack
literature shows prefill overrides the initial-token distribution), yet no
prompt-format harness measures per-template prefill adherence or re-runs
*protocol selection* under prefill. This module is the lab's operational-
ization: benign format-lock prefills are appended (as raw token ids) after
each family's generation prompt, the model generates the continuation, and
the scored span is ``prefill_text + generated_text`` -- the deterministic
contract that the model "said" the prefill.

Architectural grounding (read before extending this module):

* The prefill joins at the *token-id* level: ``encode(prompt) +
  encode(prefill)``. This is deliberate -- the prefill is caller-supplied
  content appended after the rendered prompt, and id-concatenation keeps
  the exact token sequence auditable. (Token-level, not text-level:
  ``encode(prompt + prefill)`` could re-tokenize the seam.)
* The scored span is ``decode(prefill_ids + generated_ids)``. Token
  ``decode`` is a homomorphism over id-concatenation, so once the stage
  verifies ``decode(prefill_ids) == prefill_text`` (fail-closed), the
  scored span is exactly ``prefill_text + decoded_generation``.
* Grading needs no fixture transformation: ``prefill_text + generated ==
  prefill_text + expected`` iff ``generated == expected`` (exact-string
  prefix cancellation), so the pipeline's ``exact_match`` on the *plain*
  expected answer already grades the prefill run. A model that honors
  ``"Answer: "`` and emits ``"42"`` passes; a model that ignores the
  prefill (emits ``"The answer is 42"``) or restarts its turn (re-emits
  ``"Answer: 42"``) fails -- exactly as it should.
* Honoring is detected exactly, not heuristically: a model that did not
  accept the prefill as its own opening restarts its turn and re-emits
  its preferred opener, *duplicating* the prefill text at the start of
  its generation (:func:`honors_prefill`). The pass rate and the honoring
  rate are separate metrics: a model can honor the prefill yet still fail
  (wrong answer), or ignore it in a way that still passes nothing.

Two falsifiable claims, pre-registered here:

1. The selected winner under a format-locking prefill differs from the
   native winner for at least some runs (selection-level discriminative
   power of the prefill dimension).
2. Format-lock prefill honoring varies across candidates, and candidates
   whose generation prompt opens a reasoning/think channel honor them
   less: a think-channel opener puts the model in reasoning mode, which
   conflicts with an answer-format lock. The channel predicate
   (:func:`opens_think_channel`) is computed from each candidate's own
   ``Format`` (the generation-prompt tail), so thinking on/off variants
   of one family are labeled independently. Group means are reported as
   a correlation readout, not a causal claim.

Everything here is pure: spec validation, fixture construction, the
honoring predicate, the think-channel predicate, and the backend wrapper
need no model.
"""

from __future__ import annotations

import re
from threading import Event

from .core import Backend, Fixture, Format, Generation

#: Synthetic prefill specifications. Benign format locks only -- no
#: jailbreaks, no instruction overrides. The trailing space is part of
#: the contract: an honoring model emits exactly the plain expected
#: answer, and the scored span is ``text + expected`` with no seam
#: ambiguity. Texts are ASCII-only so the duplication check in
#: :func:`honors_prefill` stays exact.
PREFILL_SPECS: tuple[dict[str, str], ...] = (
    {"id": "answer-colon", "text": "Answer: "},
    {"id": "final-answer", "text": "Final answer: "},
    {"id": "result-colon", "text": "Result: "},
)

_SPEC_KEYS = ("id", "text")

#: Think-channel tags: ``<think>``, ``<|open|>think<|sep|>`` (the XTML
#: convention used by kimi3/minimax3 -- "think" delimited by ``<|``/``|>``
#: rather than inside one ``<...>`` pair), ``<think:opensource>``,
#: ``<ifm|think>`` -- any tag whose name contains "think". Closers
#: (``</think>``, ``<|close|>think<|sep|>``) are excluded by the negative
#: lookahead / alternation; the channel is open iff the *last* think tag
#: in the tail is an opener (a thinking-off template opens then closes
#: the block).
_THINK_TAG_RE = re.compile(r"(?:<(?!/)[^<>]*think[^<>]*>|<\|open\|>think<\|sep\|>)", re.IGNORECASE)
_ANY_THINK_TAG_RE = re.compile(
    r"(?:</?[^<>]*think[^<>]*>|<\|open\|>think<\|sep\|>|<\|close\|>think<\|sep\|>)",
    re.IGNORECASE,
)


def validate_prefill_specs(specs: object) -> tuple[dict[str, str], ...]:
    """Validate synthetic prefill specifications.

    Returns the specs as a tuple of plain dicts. Raises ``ValueError``
    on any malformed spec: not a nonempty tuple, non-dict entries,
    missing/blank ``id`` or ``text``, duplicate ids, or non-ASCII text
    (the honoring duplication check is exact only on ASCII).
    """
    if type(specs) is not tuple or not specs:
        raise ValueError("specs must be a nonempty tuple")
    seen: set[str] = set()
    for spec in specs:
        if type(spec) is not dict:
            raise ValueError("each spec must be a dict")
        for key in _SPEC_KEYS:
            value = spec.get(key)
            if type(value) is not str or not value.strip():
                raise ValueError(f"spec field {key!r} must be a nonblank string")
        spec_id = spec["id"]
        if spec_id in seen:
            raise ValueError(f"duplicate spec id {spec_id!r}")
        seen.add(spec_id)
        try:
            spec["text"].encode("ascii")
        except UnicodeEncodeError:
            raise ValueError(f"spec {spec_id!r}: text must be ASCII-only") from None
    return tuple(dict(spec) for spec in specs)


def build_prefill_fixtures(
    cases: tuple[Fixture, ...], *, prefill: dict[str, str]
) -> tuple[Fixture, ...]:
    """Wrap search fixtures for one prefill condition.

    Fixture ``messages`` and ``expected`` are unchanged (the pipeline's
    ``exact_match`` on the plain expected answer already grades the
    prefill run -- see the module docstring); only the fixture id gains
    the ``prefill/{id}/`` prefix for provenance and the group becomes
    ``prefill-{id}``. Raises ``ValueError`` on bad input.
    """
    (spec,) = validate_prefill_specs((prefill,))
    if type(cases) is not tuple or not cases:
        raise ValueError("cases must be a nonempty tuple of Fixture")
    if any(type(case) is not Fixture for case in cases):
        raise ValueError("cases must be a nonempty tuple of Fixture")
    prefill_id = spec["id"]
    return tuple(
        Fixture(
            id=f"prefill/{prefill_id}/{case.id}",
            messages=case.messages,
            expected=case.expected,
            group=f"prefill-{prefill_id}",
        )
        for case in cases
    )


def honors_prefill(generated: str, prefill_text: str) -> bool:
    """Whether the model accepted the prefill as its own opening.

    A model that ignored the prefill restarts its turn and re-emits its
    preferred opener, duplicating the prefill text at the start of its
    generation. Returns ``False`` on that duplication, ``True``
    otherwise. The comparison is exact after stripping leading
    whitespace from the generation and surrounding whitespace from the
    prefill (an empty generation trivially contains no duplication; the
    pass criterion still fails it). Raises ``ValueError`` on a non-
    string generation or a blank prefill.
    """
    if type(generated) is not str:
        raise ValueError("generated must be a string")
    if type(prefill_text) is not str or not prefill_text.strip():
        raise ValueError("prefill_text must be a nonblank string")
    return not generated.lstrip().startswith(prefill_text.strip())


def generation_tail(candidate: Format) -> str:
    """Literal text the model sees immediately before the prefill.

    The generation prompt ends with the open-reply suffix (``open_suffix``
    when set, else ``assistant_prefix``); the prefill is appended after
    it. Raises ``ValueError`` on a non-``Format`` candidate.
    """
    if type(candidate) is not Format:
        raise ValueError("candidate must be a Format")
    if candidate.open_suffix is not None:
        return candidate.open_suffix
    return candidate.assistant_prefix


def opens_think_channel(candidate: Format) -> bool:
    """Whether the generation prompt opens a reasoning/think channel.

    Inspects the generation-prompt tail (:func:`generation_tail`) for
    think-channel tags. The channel counts as open iff the *last* think
    tag is an opener: a thinking-off template (``<think>...</think>``)
    opens then closes the block, leaving no open channel at generation
    time. Computed from the candidate's own ``Format``, so thinking
    on/off variants of one family are labeled independently. Never
    raises on a ``Format``; raises ``ValueError`` otherwise.
    """
    tail = generation_tail(candidate)
    tags = _ANY_THINK_TAG_RE.findall(tail)
    if not tags:
        return False
    return _THINK_TAG_RE.fullmatch(tags[-1]) is not None


class PrefillBackend:
    """Backend wrapper appending a prefill after the encoded prompt.

    ``encode(text)`` returns ``inner.encode(text) + prefill_ids``; the
    prefill is caller-supplied tokens appended after the rendered
    prompt. ``generate`` forwards unchanged (the model generates the
    continuation *after* the prefill); ``decode`` forwards unchanged.
    The fingerprint gains ``+prefill:<id>`` so reports distinguish
    prefill runs. Fail-closed: raises ``ValueError`` on bad prefill ids
    or a blank prefill id at construction.
    """

    def __init__(self, inner: Backend, prefill_ids: tuple[int, ...], prefill_id: str) -> None:
        if inner is None:
            raise ValueError("inner backend must not be None")
        if (
            type(prefill_ids) is not tuple
            or not prefill_ids
            or any(type(token) is not int or token < 0 for token in prefill_ids)
        ):
            raise ValueError("prefill_ids must be a nonempty tuple of token ids")
        if type(prefill_id) is not str or not prefill_id.strip():
            raise ValueError("prefill_id must be a nonblank string")
        inner_fingerprint = getattr(inner, "fingerprint", None)
        if type(inner_fingerprint) is not str:
            raise ValueError("inner backend fingerprint must be a string")
        self._inner = inner
        self._prefill_ids = prefill_ids
        self._prefill_id = prefill_id
        # Settable instance attribute (not a read-only property): the
        # Backend protocol declares fingerprint as a settable variable.
        self.fingerprint = f"{inner_fingerprint}+prefill:{prefill_id}"

    @property
    def prefill_ids(self) -> tuple[int, ...]:
        return self._prefill_ids

    def encode(self, text: str) -> tuple[int, ...]:
        return self._inner.encode(text) + self._prefill_ids

    def decode(self, token_ids: tuple[int, ...]) -> str:
        return self._inner.decode(token_ids)

    def generate(
        self,
        input_ids: tuple[int, ...],
        *,
        max_new_tokens: int,
        seed: int,
        cancel: Event,
    ) -> Generation:
        return self._inner.generate(
            input_ids, max_new_tokens=max_new_tokens, seed=seed, cancel=cancel
        )
