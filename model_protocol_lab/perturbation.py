"""Deterministic fixture-phrasing perturbation operators.

Pure, seeded text transforms for the fixture perturbation robustness
experiment (cycle 144): they rewrite fixture *message content* while
leaving fixture ids, groups, and expected answers untouched, so the
exact-match scoring of the re-run stays meaningful. No model, no I/O,
no global state.

The research question: does the selected winner survive fixture
phrasing perturbations? A winner that only wins on the exact fixture
wording is overfitted to phrasing luck; a robust winner survives
typos, case noise, whitespace variation, Unicode quirks, and missing
punctuation. No other prompt-format lab measures this.

Every operator is a pure function of ``(text, operator, seed)``:
identical inputs always produce identical outputs. An operator that
cannot change a text (e.g. ``punctuation_strip`` on punctuation-free
text) or whose result would be empty/blank returns the original text
unchanged -- fail-closed, and reported honestly as a zero
``changed_fixtures`` count by the experiment stage rather than
silently dropped.
"""

from __future__ import annotations

import random
import string
import unicodedata
from typing import Any

from .core import Fixture, Message

PERTURBATION_OPERATORS: tuple[str, ...] = (
    "typo",
    "case_flip",
    "whitespace",
    "unicode_variant",
    "punctuation_strip",
)

_MAX_SEED = 2**64


def _check_text(text: Any) -> str:
    """Check text."""
    if type(text) is not str:
        raise ValueError("text must be a string")
    return text


def _check_seed(seed: Any) -> int:
    """Check seed."""
    if type(seed) is not int or isinstance(seed, bool) or not 0 <= seed < _MAX_SEED:
        raise ValueError("seed must be an unsigned 64-bit integer")
    return seed


def _check_operator(operator: Any) -> str:
    """Check operator."""
    for known in PERTURBATION_OPERATORS:
        if operator == known:
            return known
    raise ValueError(f"unknown perturbation operator: {operator!r}")


def _guard(text: str, perturbed: str) -> str:
    """Fail-closed: never return an empty/blank perturbation."""
    if not perturbed or not perturbed.strip():
        return text
    return perturbed


def _perturb_typo(text: str, rng: random.Random) -> str:
    """Seeded character substitutions, simulating typing errors."""
    alpha = [i for i, ch in enumerate(text) if ch.isalpha()]
    if not alpha:
        return text
    n_subs = max(1, len(alpha) // 16)
    chars = list(text)
    for pos in rng.sample(alpha, min(n_subs, len(alpha))):
        original = chars[pos]
        pool = [c for c in string.ascii_letters if c != original]
        chars[pos] = rng.choice(pool)
    return "".join(chars)


def _perturb_case_flip(text: str, rng: random.Random) -> str:
    """Flip the case of a seeded subset of cased characters."""
    return "".join(ch.swapcase() if ch.isalpha() and rng.random() < 0.35 else ch for ch in text)


def _perturb_whitespace(text: str, rng: random.Random) -> str:
    """Seeded whitespace variation: collapse, double, or newline-split."""
    if not any(ch.isspace() for ch in text):
        return text
    mode = rng.randrange(3)
    if mode == 0:
        return " ".join(text.split())
    if mode == 1:
        return text.replace(" ", "  ")
    return text.replace(" ", "\n")


def _perturb_unicode_variant(text: str, rng: random.Random) -> str:
    """Replace a seeded subset of ASCII alphanumerics with fullwidth forms.

    Fullwidth characters (U+FF01--U+FF5E) are the NFKC-foldable
    compatibility equivalents of ASCII ``!``--``~``: they look nearly
    identical to a human reader but tokenize differently under most
    tokenizers, which is exactly the phrasing-noise class this
    operator models. Non-ASCII input passes through untouched.
    """
    out: list[str] = []
    changed = False
    for ch in text:
        if "!" <= ch <= "~" and ch.isalnum() and rng.random() < 0.5:
            out.append(chr(ord(ch) + 0xFEE0))
            changed = True
        else:
            out.append(ch)
    return "".join(out) if changed else text


def _perturb_punctuation_strip(text: str, rng: random.Random) -> str:
    """Remove every Unicode punctuation character (category P*)."""
    _ = rng  # deterministic signature; the transform itself is fixed
    return "".join(ch for ch in text if not unicodedata.category(ch).startswith("P"))


_OPERATORS = {
    "typo": _perturb_typo,
    "case_flip": _perturb_case_flip,
    "whitespace": _perturb_whitespace,
    "unicode_variant": _perturb_unicode_variant,
    "punctuation_strip": _perturb_punctuation_strip,
}


def perturb_text(text: str, operator: str, *, seed: int) -> str:
    """Apply one perturbation operator to ``text``; deterministic in seed.

    Returns the perturbed text, or the original text unchanged when the
    operator cannot apply (no eligible characters) or would produce an
    empty/blank result. Raises ``ValueError`` on bad argument types.
    """
    text = _check_text(text)
    operator = _check_operator(operator)
    seed = _check_seed(seed)
    # Deterministic experiment seeding, not cryptographic randomness:
    # reproducibility from the run seed is the requirement.
    rng = random.Random(seed)  # noqa: S311 # nosec B311
    return _guard(text, _OPERATORS[operator](text, rng))


def perturb_fixture(fixture: Fixture, operator: str, *, seed: int) -> Fixture:
    """Perturb every message's content; id, group, expected preserved.

    The expected answer is deliberately never touched: re-running the
    evaluation pipeline on perturbed fixtures must score against the
    same expected answers, otherwise the per-variant verdicts would
    not be comparable to the primary run.
    """
    if type(fixture) is not Fixture:
        raise ValueError("fixture must be a Fixture")
    operator = _check_operator(operator)
    seed = _check_seed(seed)
    messages = tuple(
        Message(
            role=message.role,
            content=perturb_text(message.content, operator, seed=seed),
        )
        for message in fixture.messages
    )
    return Fixture(
        id=fixture.id,
        messages=messages,
        expected=fixture.expected,
        group=fixture.group,
    )


def perturb_fixtures(
    cases: tuple[Fixture, ...], operator: str, *, seed: int
) -> tuple[Fixture, ...]:
    """Perturb a whole fixture tuple with one operator and seed."""
    if type(cases) is not tuple or not cases or any(type(c) is not Fixture for c in cases):
        raise ValueError("cases must be a nonempty tuple of Fixture")
    operator = _check_operator(operator)
    seed = _check_seed(seed)
    return tuple(perturb_fixture(case, operator, seed=seed) for case in cases)


def variant_plan(
    n_variants: int,
    operators: tuple[str, ...] = PERTURBATION_OPERATORS,
    *,
    seed: int,
) -> tuple[dict[str, Any], ...]:
    """Deterministic per-variant operator/seed assignment.

    Operators are assigned round-robin, so ``n_variants <=
    len(operators)`` covers each operator exactly once; larger counts
    repeat operators with fresh seeds. The variant seed is ``(seed +
    index) mod 2**64``: adjacent variants never share a perturbation,
    and the whole plan is reproducible from the run seed alone.
    """
    if type(n_variants) is not int or isinstance(n_variants, bool) or not 1 <= n_variants <= 8:
        raise ValueError("n_variants must be an integer between 1 and 8")
    if (
        type(operators) is not tuple
        or not operators
        or any(op not in PERTURBATION_OPERATORS for op in operators)
        or len(set(operators)) != len(operators)
    ):
        raise ValueError("operators must be a nonempty tuple of unique known operators")
    seed = _check_seed(seed)
    return tuple(
        {
            "variant_id": f"v{i:02d}",
            "operator": operators[i % len(operators)],
            "seed": (seed + i) % _MAX_SEED,
        }
        for i in range(n_variants)
    )
