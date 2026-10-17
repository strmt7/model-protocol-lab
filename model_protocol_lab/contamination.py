"""Search/held-out contamination detection (pure, no model, no I/O).

The lab's central methodological claim is strict search/held-out
separation. :class:`core.Split` already rejects *byte-identical* message
inputs across the split, but a held-out fixture that is a near-duplicate
or paraphrase of a search fixture slips through and silently inflates
held-out pass rates. This module measures that gap mechanically.

Two complementary deterministic metrics are computed over a normalized
representation of each fixture (lowercased, whitespace-collapsed,
NFKC-normalized concatenation of user message contents plus the expected
answer):

* character-level normalized similarity via
  :class:`difflib.SequenceMatcher` with ``autojunk=False`` (deterministic;
  fixtures are small so the quadratic cost is bounded),
* token-3-gram Jaccard over the same normalized text.

Tiers are advisory -- they warn, never fail a run:

* ``duplicate``: ratio >= 0.98
* ``near_duplicate``: ratio >= 0.90
* ``paraphrase_suspect``: ratio >= 0.80 or Jaccard >= 0.85
* ``clean``: anything below

Unicode confusables must not evade detection: normalization applies NFKC
and strips format characters (category Cf: zero-width space, word
joiner, ...) before comparing, mirroring the sanitizer philosophy used
for terminal display. Thresholds are heuristic by nature and heavy
paraphrases will still evade; the report is a hygiene gate, not a proof.
"""

from __future__ import annotations

import unicodedata
from difflib import SequenceMatcher
from typing import Any

DUPLICATE_RATIO = 0.98
NEAR_DUPLICATE_RATIO = 0.90
PARAPHRASE_RATIO = 0.80
PARAPHRASE_JACCARD = 0.85

_TIERS = ("clean", "paraphrase_suspect", "near_duplicate", "duplicate")


def _normalize_text(text: str) -> str:
    """Normalize fixture text for comparison.

    NFKC folds fullwidth/confusable variants, format characters (Cf --
    zero-width space, soft hyphen is not Cf but is whitespace-collapsed,
    word joiner, ...) are stripped so they cannot evade detection, text is
    lowercased, and all whitespace runs collapse to a single space.
    """
    folded = unicodedata.normalize("NFKC", text)
    stripped = "".join(char for char in folded if unicodedata.category(char) != "Cf")
    return " ".join(stripped.lower().split())


def fixture_signature(fixture: Any) -> str:
    """Normalized comparison text for one fixture.

    Concatenation of the user message contents plus the expected answer.
    Assistant turns are excluded: they carry no input signal for the
    search/held-out separation question.
    """
    parts = [
        message.content for message in fixture.messages if getattr(message, "role", None) == "user"
    ]
    parts.append(getattr(fixture, "expected", ""))
    return _normalize_text("\n".join(part for part in parts if type(part) is str))


def _trigrams(text: str) -> set[tuple[str, ...]]:
    tokens = text.split()
    if len(tokens) < 3:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i : i + 3]) for i in range(len(tokens) - 2)}


def similarity_scores(left: str, right: str) -> tuple[float, float]:
    """(SequenceMatcher ratio, token-3-gram Jaccard) for normalized texts.

    Deterministic: ``autojunk=False`` keeps the ratio exact on small
    inputs, and both metrics are pure functions of the inputs.
    """
    ratio = SequenceMatcher(None, left, right, autojunk=False).ratio()
    grams_left, grams_right = _trigrams(left), _trigrams(right)
    union = grams_left | grams_right
    jaccard = len(grams_left & grams_right) / len(union) if union else 1.0
    return ratio, jaccard


def classify_pair(ratio: float, jaccard: float) -> str:
    """Tier for one fixture pair; thresholds are advisory, see module docs."""
    if ratio >= DUPLICATE_RATIO:
        return "duplicate"
    if ratio >= NEAR_DUPLICATE_RATIO:
        return "near_duplicate"
    if ratio >= PARAPHRASE_RATIO or jaccard >= PARAPHRASE_JACCARD:
        return "paraphrase_suspect"
    return "clean"


def contamination_report(search: Any, held_out: Any) -> dict[str, Any]:
    """Search/held-out contamination report; pure, never raises.

    For every held-out fixture, the maximum similarity to any search
    fixture is computed; pairs at or above ``paraphrase_suspect`` are
    reported. Output::

        {"worst_tier": "near_duplicate",
         "pairs": [{"held_out": "h-03", "search": "s-11",
                    "ratio": 0.87, "jaccard": 0.81,
                    "tier": "paraphrase_suspect"}, ...]}

    Pairs sort by tier severity (worst first), then ratio descending, then
    ids -- deterministic across processes and hash seeds. Empty or
    malformed inputs yield a clean empty report (fail-closed, advisory).
    """
    try:
        search_sigs = [(str(fixture.id), fixture_signature(fixture)) for fixture in search or ()]
        held_sigs = [(str(fixture.id), fixture_signature(fixture)) for fixture in held_out or ()]
    except (TypeError, AttributeError):
        return {"worst_tier": "clean", "pairs": []}
    pairs: list[dict[str, Any]] = []
    for held_id, held_sig in held_sigs:
        best: tuple[float, float, str, str] | None = None
        for search_id, search_sig in search_sigs:
            ratio, jaccard = similarity_scores(held_sig, search_sig)
            tier = classify_pair(ratio, jaccard)
            if best is None or (_TIERS.index(tier), ratio) > (
                _TIERS.index(best[3]),
                best[0],
            ):
                best = (ratio, jaccard, search_id, tier)
        if best is not None and _TIERS.index(best[3]) >= _TIERS.index("paraphrase_suspect"):
            ratio, jaccard, search_id, tier = best
            pairs.append(
                {
                    "held_out": held_id,
                    "search": search_id,
                    "ratio": round(ratio, 4),
                    "jaccard": round(jaccard, 4),
                    "tier": tier,
                }
            )
    pairs.sort(
        key=lambda pair: (
            -_TIERS.index(pair["tier"]),
            -pair["ratio"],
            pair["held_out"],
            pair["search"],
        )
    )
    worst = max((_TIERS.index(pair["tier"]) for pair in pairs), default=0)
    return {"worst_tier": _TIERS[worst], "pairs": pairs}
