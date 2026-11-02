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
  :class:`difflib.SequenceMatcher` with ``autojunk=False`` (deterministic),
  computed under a wall-clock budget -- the loader's caps (100 fixtures
  per split, 16,384-char fields) admit inputs whose all-pairs cost is
  minutes, so the sweep is deadline-bound rather than assumed small
  (see :func:`contamination_report`),
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

import time
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
    """Return the trigrams."""
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


def _bounded_ratio(left: str, right: str, deadline: float) -> float | None:
    """SequenceMatcher ratio under a wall-clock deadline, or None.

    Replicates difflib's matching-blocks queue with the clock checked
    between ``find_longest_match`` calls -- the construction
    ``core.template_diff`` established -- so a completed result is
    exactly ``SequenceMatcher(None, left, right, autojunk=False).ratio()``.
    Returns None when the deadline passes mid-computation; the caller
    treats the pair, and the sweep, as unevaluated rather than guessing
    a score from a partial match count.
    """
    total = len(left) + len(right)
    if total == 0:
        return 1.0
    matcher = SequenceMatcher(None, left, right, autojunk=False)
    queue: list[tuple[int, int, int, int]] = [(0, len(left), 0, len(right))]
    matches = 0
    while queue:
        if time.monotonic() > deadline:
            return None
        alo, ahi, blo, bhi = queue.pop()
        i, j, size = matcher.find_longest_match(alo, ahi, blo, bhi)
        if size:
            matches += size
            if alo < i and blo < j:
                queue.append((alo, i, blo, j))
            if i + size < ahi and j + size < bhi:
                queue.append((i + size, ahi, j + size, bhi))
    return 2.0 * matches / total


def classify_pair(ratio: float, jaccard: float) -> str:
    """Tier for one fixture pair; thresholds are advisory, see module docs."""
    if ratio >= DUPLICATE_RATIO:
        return "duplicate"
    if ratio >= NEAR_DUPLICATE_RATIO:
        return "near_duplicate"
    if ratio >= PARAPHRASE_RATIO or jaccard >= PARAPHRASE_JACCARD:
        return "paraphrase_suspect"
    return "clean"


def contamination_report(
    search: Any, held_out: Any, *, _deadline_s: float = 10.0
) -> dict[str, Any]:
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

    The sweep runs under a wall-clock budget (default 10 s, the
    ``core.template_diff`` precedent): the ratio is superlinear in
    fixture-text length and multiplies over all search x held-out
    pairs, and the dataset loader's caps admit inputs whose unbounded
    sweep takes minutes, which would hang the CLI commands embedding
    this report. The clock is checked between ``find_longest_match``
    calls and between pairs. When the budget is exhausted, the report
    covers only the held-out fixtures evaluated completely before the
    deadline -- the fixture in progress contributes nothing -- and
    carries ``"budget_exhausted": True``; the key is absent otherwise.
    An exhausted sweep's ``worst_tier`` describes the evaluated prefix
    only: it is never a completed clean sweep.
    """
    try:
        search_sigs = [(str(fixture.id), fixture_signature(fixture)) for fixture in search or ()]
        held_sigs = [(str(fixture.id), fixture_signature(fixture)) for fixture in held_out or ()]
    except (TypeError, AttributeError):
        return {"worst_tier": "clean", "pairs": []}
    deadline = time.monotonic() + _deadline_s
    search_grams = [_trigrams(sig) for _sig_id, sig in search_sigs]
    pairs: list[dict[str, Any]] = []
    exhausted = False
    for held_id, held_sig in held_sigs:
        held_grams = _trigrams(held_sig)
        best: tuple[float, float, str, str] | None = None
        for (search_id, search_sig), grams in zip(search_sigs, search_grams, strict=True):
            if time.monotonic() > deadline:
                exhausted = True
                break
            ratio = _bounded_ratio(held_sig, search_sig, deadline)
            if ratio is None:
                exhausted = True
                break
            union = held_grams | grams
            jaccard = len(held_grams & grams) / len(union) if union else 1.0
            tier = classify_pair(ratio, jaccard)
            if best is None or (_TIERS.index(tier), ratio) > (
                _TIERS.index(best[3]),
                best[0],
            ):
                best = (ratio, jaccard, search_id, tier)
        if exhausted:
            break
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
    report: dict[str, Any] = {"worst_tier": _TIERS[worst], "pairs": pairs}
    if exhausted:
        report["budget_exhausted"] = True
    return report
