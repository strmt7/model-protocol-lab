"""Whitespace canonicalization at template boundaries (research wave-3 idea 4).

The question: whitespace variants of the *same* template rendering
(trailing space vs not, leading newline vs not) tokenize to identical id
sequences and produce identical task verdicts? A production case study
(aicassindra, "Tokenization, in depth") traces three separate bugs to
whitespace at template/token boundaries -- a training template ending in
a lone space token, a serving template with a trailing space changing the
first generated token, an evaluator disagreeing on leading-space outputs
-- and arXiv:2405.17067 ("Tokenization Matters!") shows adversarial
tokenization degrading LLMs on purpose-built cases across 8 models.
Prompt harnesses treat the rendered string as atomic; the lab controls
the *exact token sequence* it sends, so it can audit whitespace ->
ids -> verdicts per template family and publish which families are
whitespace-fragile, exactly where.

The experiment: for each search candidate and each search fixture, the
rendered prompt is re-encoded under each of :data:`MUTATIONS`
(:func:`apply_mutation` is text-level, applied *before* encoding, so the
audited id sequence is exactly what the tokenizer produces for the
mutated string -- no seam re-tokenization games); id sequences are
compared exactly (:func:`compare_id_sequences`); fixtures whose ids
diverged are re-run through the normal ``core.evaluate()`` pipeline via
:class:`WsScanBackend` (an encode-time wrapper, mirroring
``PrefillBackend``) and their verdicts compared to the native ones --
the whitespace-fragility signal is a verdict *flip* on a divergent
variant.

Pre-registered claims (falsifiable):

1. ``apply_mutation`` is a pure function of ``(text, mutation)``;
   unknown mutation names raise ``ValueError`` (fail-closed, no silent
   no-op).
2. A whitespace variant flips a task verdict only when its id sequence
   diverged -- OR the backend is nondeterministic across runs. Same
   ids + flipped verdict is recorded separately
   (``n_same_ids_verdict_flips``) as a determinism finding, never as
   whitespace fragility.
3. ``aggregate_mutation_records`` rates stay in [0, 1]; a candidate is
   ``whitespace_robust`` only when every mutation produced zero id
   divergence and zero verdict flips (pinned by Hypothesis).

Everything here is pure: text mutation, id-sequence comparison, and
rate aggregation need no model. Tokenizer interaction enters only
through the injected ``encode`` callable at the stage level.
"""

from __future__ import annotations

from threading import Event

from .core import Backend, Generation

#: Whitespace mutations applied to the rendered prompt text, in canonical
#: order. Each name maps to a one-line description in
#: :data:`MUTATION_DESCRIPTIONS`.
MUTATIONS = (
    "trailing_space",
    "double_trailing_space",
    "trailing_newline",
    "trailing_tab",
    "leading_space",
    "leading_newline",
    "strip_trailing_space",
    "collapse_double_space",
)

#: Human-readable description per mutation (used by the HTML report).
MUTATION_DESCRIPTIONS = {
    "trailing_space": "append one trailing space",
    "double_trailing_space": "append two trailing spaces",
    "trailing_newline": "append one trailing newline",
    "trailing_tab": "append one trailing tab",
    "leading_space": "prepend one leading space",
    "leading_newline": "prepend one leading newline",
    "strip_trailing_space": "remove one trailing space when present",
    "collapse_double_space": "replace every double space with one (single left-to-right pass)",
}


def validate_mutations(names: object) -> tuple[str, ...]:
    """Validate a mutation list; returns the canonical tuple.

    ``names`` must be a nonempty tuple/list of known mutation names
    (duplicates rejected). Raises ``ValueError`` otherwise.
    """
    # isinstance (not type-is) so mypy narrows the element type below.
    if not isinstance(names, (tuple, list)) or not names:
        raise ValueError("mutations must be a nonempty tuple/list of mutation names")
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if type(name) is not str or name not in MUTATIONS:
            raise ValueError(f"unknown whitespace mutation {name!r}")
        if name in seen:
            raise ValueError(f"duplicate whitespace mutation {name!r}")
        seen.add(name)
        ordered.append(name)
    return tuple(ordered)


def apply_mutation(text: str, mutation: str) -> str:
    """Apply one whitespace mutation to rendered prompt text.

    Pure and total over valid inputs. Raises ``ValueError`` when
    ``text`` is not a string or ``mutation`` is not a known mutation
    name (fail-closed: never a silent no-op).
    """
    if type(text) is not str:
        raise ValueError("text must be a string")
    if mutation == "trailing_space":
        return text + " "
    if mutation == "double_trailing_space":
        return text + "  "
    if mutation == "trailing_newline":
        return text + "\n"
    if mutation == "trailing_tab":
        return text + "\t"
    if mutation == "leading_space":
        return " " + text
    if mutation == "leading_newline":
        return "\n" + text
    if mutation == "strip_trailing_space":
        return text[:-1] if text.endswith(" ") else text
    if mutation == "collapse_double_space":
        return text.replace("  ", " ")
    raise ValueError(f"unknown whitespace mutation {mutation!r}")


def _valid_token_ids(value: object) -> bool:
    return type(value) is tuple and all(type(token) is int and token >= 0 for token in value)


def compare_id_sequences(base_ids: object, variant_ids: object) -> bool:
    """Exact equality of two token-id sequences.

    Raises ``ValueError`` when either side is not a tuple of
    nonnegative ints (fail-closed: a malformed comparison is never
    reported as "identical").
    """
    if not _valid_token_ids(base_ids):
        raise ValueError("base_ids must be a tuple of nonnegative ints")
    if not _valid_token_ids(variant_ids):
        raise ValueError("variant_ids must be a tuple of nonnegative ints")
    return base_ids == variant_ids


def aggregate_mutation_records(records: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate validated per-mutation whitespace records.

    Each record is ``{"mutation": str, "n_fixtures": int,
    "n_evaluable": int, "n_ids_divergent": int, "n_verdict_flips": int,
    "n_same_ids_verdict_flips": int}``. Returns ``{"per_mutation": ...,
    "n_fragile_mutations": int, "max_ids_divergence_rate": float,
    "max_verdict_flip_rate": float, "whitespace_robust": bool}`` where a
    mutation is fragile when it flipped at least one verdict, and a
    candidate is whitespace-robust only when no mutation diverged in ids
    and no mutation flipped a verdict. Rates are 0.0 on empty
    denominators. Raises ``ValueError`` on malformed records.
    """
    if type(records) is not list:
        raise ValueError("records must be a list")
    per_mutation: dict[str, dict[str, object]] = {}
    max_ids_divergence_rate = 0.0
    max_verdict_flip_rate = 0.0
    for record in records:
        if type(record) is not dict:
            raise ValueError("records must be dicts")
        mutation = record.get("mutation")
        if type(mutation) is not str or mutation not in MUTATIONS:
            raise ValueError(f"record has unknown mutation {mutation!r}")
        if mutation in per_mutation:
            raise ValueError(f"duplicate record for mutation {mutation!r}")
        counts: dict[str, int] = {}
        for key in (
            "n_fixtures",
            "n_evaluable",
            "n_ids_divergent",
            "n_verdict_flips",
            "n_same_ids_verdict_flips",
        ):
            value = record.get(key)
            if type(value) is not int or value < 0:
                raise ValueError(f"record {key} must be a nonnegative int")
            counts[key] = value
        if counts["n_evaluable"] > counts["n_fixtures"]:
            raise ValueError("n_evaluable cannot exceed n_fixtures")
        if counts["n_ids_divergent"] > counts["n_fixtures"]:
            raise ValueError("n_ids_divergent cannot exceed n_fixtures")
        if counts["n_verdict_flips"] > counts["n_evaluable"]:
            raise ValueError("n_verdict_flips cannot exceed n_evaluable")
        n_fixtures = counts["n_fixtures"]
        n_evaluable = counts["n_evaluable"]
        ids_divergence_rate = counts["n_ids_divergent"] / n_fixtures if n_fixtures else 0.0
        verdict_flip_rate = counts["n_verdict_flips"] / n_evaluable if n_evaluable else 0.0
        per_mutation[mutation] = {
            "n_fixtures": n_fixtures,
            "n_evaluable": n_evaluable,
            "n_ids_divergent": counts["n_ids_divergent"],
            "ids_divergence_rate": ids_divergence_rate,
            "n_verdict_flips": counts["n_verdict_flips"],
            "verdict_flip_rate": verdict_flip_rate,
            "n_same_ids_verdict_flips": counts["n_same_ids_verdict_flips"],
            "fragile": counts["n_verdict_flips"] > 0,
        }
        if ids_divergence_rate > max_ids_divergence_rate:
            max_ids_divergence_rate = ids_divergence_rate
        if verdict_flip_rate > max_verdict_flip_rate:
            max_verdict_flip_rate = verdict_flip_rate
    n_fragile = sum(1 for entry in per_mutation.values() if entry["fragile"])
    return {
        "per_mutation": per_mutation,
        "n_fragile_mutations": n_fragile,
        "max_ids_divergence_rate": max_ids_divergence_rate,
        "max_verdict_flip_rate": max_verdict_flip_rate,
        "whitespace_robust": bool(per_mutation)
        and all(
            entry["n_ids_divergent"] == 0 and entry["n_verdict_flips"] == 0
            for entry in per_mutation.values()
        ),
    }


class WsScanBackend:
    """Backend wrapper mutating rendered whitespace before encoding.

    ``encode(text)`` returns ``inner.encode(apply_mutation(text,
    mutation))``; ``generate`` and ``decode`` forward unchanged. The
    fingerprint gains ``+wsscan:<mutation>`` so whitespace-variant runs
    are distinguishable in reports. Fail-closed: raises ``ValueError``
    on an unknown mutation or a non-string inner fingerprint at
    construction.
    """

    def __init__(self, inner: Backend, mutation: str) -> None:
        if inner is None:
            raise ValueError("inner backend must not be None")
        if type(mutation) is not str or mutation not in MUTATIONS:
            raise ValueError(f"unknown whitespace mutation {mutation!r}")
        inner_fingerprint = getattr(inner, "fingerprint", None)
        if type(inner_fingerprint) is not str:
            raise ValueError("inner backend fingerprint must be a string")
        self._inner = inner
        self._mutation = mutation
        # Settable instance attribute (not a read-only property): the
        # Backend protocol declares fingerprint as a settable variable.
        self.fingerprint = f"{inner_fingerprint}+wsscan:{mutation}"

    @property
    def mutation(self) -> str:
        return self._mutation

    def encode(self, text: str) -> tuple[int, ...]:
        return self._inner.encode(apply_mutation(text, self._mutation))

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
