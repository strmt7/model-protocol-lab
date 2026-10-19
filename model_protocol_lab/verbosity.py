"""Response verbosity per template family (research wave-4 idea 5).

The question: are response-length distributions (generated tokens per
fixture, on prompts with short stable answers) identical across the
template families? arXiv:2601.00624 ("Do Chatbot LLMs Talk Too Much?
The YapBench Benchmark") quantifies user-visible over-generation on
brevity-ideal prompts across 76 models and finds an order-of-magnitude
spread in median excess length -- but varies *models* on fixed prompts.
Verbosity-bias studies (e.g. the MT-Bench judge-bias section,
arXiv:2306.05685) treat length as a judge confound. Nobody varies the
chat template as the independent variable for verbosity. The lab
already records exact generated token counts per observation, so the
marginal cost of this experiment is zero extra model calls: it is
purely post-hoc over a run's observations.

Design: for each fixture whose expected answer is brevity-ideal
(see :func:`is_brevity_ideal`), ``excess = generated_tokens -
minimal_sufficient_tokens`` where the minimal count is the exact
tokenization of the expected answer (computed by the stage, which owns
the backend -- this module stays pure). Excess is clamped at zero by
construction: verbosity measures over-generation; emitting *less* than
the minimal answer is a different failure mode (already graded by
``exact_match``) and is out of scope for v1. Per candidate the module
reports the excess distribution (median / quartiles / IQR / mean /
max) plus the budget-saturation share -- the fraction of evaluable
observations that reached the token budget. A saturated generation's
measured length is a *lower bound* on true verbosity, so the saturation
share unconfounds this experiment from cycle 155's stop-cause audit
(over-generation vs max-length termination are otherwise confounded
channels).

Pre-registered claims (falsifiable):

1. Verbosity is uniform across candidates: the max-minus-min
   ``median_excess_gap`` over evaluable candidates is either ~0
   (uniform, the null hypothesis -- reported, never asserted) or
   measurably positive; i.e. the template, not the model or the
   prompt, decides how much the model yaps, with direct
   inference-cost consequences.
2. The distribution shape is reported, not just the median: quartiles
   and IQR travel with the median so a gap cannot hide behind a
   long-tailed family looking "typical" at the median.
3. Verbosity is a cost/behavior profile, never a quality judgment: the
   analysis names the most/least verbose candidates as behavior
   labels, and the report states explicitly that shorter is not
   better.

Quartile method (pinned): linear interpolation (Hyndman-Fan type 7,
the numpy ``'linear'`` default). For sorted values ``xs`` of length
``n``, the ``q``-quantile is at position ``h = (n - 1) * q``,
interpolated between ``xs[floor(h)]`` and ``xs[ceil(h)]``. Median is
``q = 0.5``; IQR = ``q(0.75) - q(0.25)``.

Everything here is pure: the brevity-ideal filter, excess arithmetic,
and distribution math need no model.
"""

from __future__ import annotations

import math
from typing import Any

#: Pinned brevity-ideal criterion: a stripped, single-line expected
#: answer of at most this many characters. Keeps minimal-sufficient
#: token counts in the single digits for typical tokenizers, so excess
#: is dominated by generation behavior rather than answer length.
BREVITY_IDEAL_MAX_CHARS = 32


def is_brevity_ideal(expected: object) -> bool:
    """Whether a fixture's expected answer qualifies as brevity-ideal.

    Pinned rule: ``expected`` is a string whose stripped form is
    non-empty, single-line, and at most
    :data:`BREVITY_IDEAL_MAX_CHARS` characters. Non-string, empty, or
    multi-line expected answers never qualify (fail-closed: they are
    excluded from the verbosity measurement, never coerced).
    """
    if type(expected) is not str:
        return False
    stripped = expected.strip()
    if not stripped:
        return False
    if "\n" in stripped or "\r" in stripped:
        return False
    return len(stripped) <= BREVITY_IDEAL_MAX_CHARS


def excess_tokens(generated: int, minimal: int) -> int:
    """Over-generation for one observation: ``max(0, generated - minimal)``.

    Clamped at zero by construction -- emitting fewer tokens than the
    minimal answer is under-generation, a different failure mode graded
    by ``exact_match`` and out of scope here. Raises ``ValueError`` on
    non-integer or negative inputs (fail-fast: token counts are never
    negative and never bools).
    """
    if type(generated) is not int or generated < 0:
        raise ValueError(f"generated must be a nonnegative int, got {generated!r}")
    if type(minimal) is not int or minimal < 0:
        raise ValueError(f"minimal must be a nonnegative int, got {minimal!r}")
    return generated - minimal if generated > minimal else 0


def _quantile(sorted_values: list[int], q: float) -> float:
    """Type-7 (linear interpolation) quantile of sorted non-empty ints."""
    n = len(sorted_values)
    if n == 1:
        return float(sorted_values[0])
    pos = (n - 1) * q
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return float(sorted_values[low])
    frac = pos - low
    return sorted_values[low] + frac * (sorted_values[high] - sorted_values[low])


def length_distribution(excesses: list[int]) -> dict[str, Any]:
    """Distribution summary of per-observation excess token counts.

    Returns ``{"n", "median_excess", "p25_excess", "p75_excess",
    "iqr_excess", "mean_excess", "max_excess"}``; the statistics are
    ``None`` when ``excesses`` is empty. Quartiles use the pinned
    linear-interpolation method (see module docstring). Raises
    ``ValueError`` on non-list input or non-integer/negative elements.
    """
    if type(excesses) is not list:
        raise ValueError(f"excesses must be a list, got {type(excesses).__name__}")
    for value in excesses:
        if type(value) is not int or value < 0:
            raise ValueError(f"excess values must be nonnegative ints, got {value!r}")
    n = len(excesses)
    if n == 0:
        return {
            "n": 0,
            "median_excess": None,
            "p25_excess": None,
            "p75_excess": None,
            "iqr_excess": None,
            "mean_excess": None,
            "max_excess": None,
        }
    ordered = sorted(excesses)
    p25 = _quantile(ordered, 0.25)
    median = _quantile(ordered, 0.5)
    p75 = _quantile(ordered, 0.75)
    return {
        "n": n,
        "median_excess": median,
        "p25_excess": p25,
        "p75_excess": p75,
        "iqr_excess": p75 - p25,
        "mean_excess": sum(ordered) / n,
        "max_excess": ordered[-1],
    }


def candidate_verbosity(
    records: list[tuple[int, int, bool]],
) -> dict[str, Any]:
    """Aggregate one candidate's per-observation verbosity records.

    Each record is ``(generated_tokens, minimal_tokens, capped)`` where
    ``capped`` marks a generation that reached the token budget (its
    measured length is a lower bound on true verbosity -- the cycle-155
    stop-cause confound, reported not hidden). Malformed records are
    excluded from the evaluable set (fail-closed: visible via ``n`` vs
    ``n_evaluable``, never silently dropped and never crashing the
    aggregation).

    Returns ``{"n", "n_evaluable", "median_excess", "p25_excess",
    "p75_excess", "iqr_excess", "mean_excess", "max_excess",
    "capped_share"}``; statistics and ``capped_share`` are ``None``
    when nothing is evaluable.
    """
    if type(records) is not list:
        raise ValueError(f"records must be a list, got {type(records).__name__}")
    n = len(records)
    excesses: list[int] = []
    capped_count = 0
    for record in records:
        if (
            type(record) is not tuple
            or len(record) != 3
            or type(record[0]) is not int
            or record[0] < 0
            or type(record[1]) is not int
            or record[1] < 0
            or type(record[2]) is not bool
        ):
            continue
        generated, minimal, capped = record
        excesses.append(excess_tokens(generated, minimal))
        if capped:
            capped_count += 1
    n_evaluable = len(excesses)
    distribution = length_distribution(excesses)
    # NOTE: "n" is set after the distribution spread: the
    # distribution's own "n" counts evaluable records, while the
    # cell's "n" counts every record seen (fail-closed visibility).
    return {
        **distribution,
        "n": n,
        "n_evaluable": n_evaluable,
        "capped_share": (capped_count / n_evaluable) if n_evaluable else None,
    }
