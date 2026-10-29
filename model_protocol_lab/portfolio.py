"""Oracle-mixture / protocol-portfolio gap analysis (research wave-2 idea 4).

The question: how much does committing to a single global winning
protocol leave on the table versus a per-fixture oracle -- and what is
the smallest protocol portfolio that covers every solvable fixture?

The routing literature (RouteLLM et al., 2024) shows per-request model
selection beats any single model on the cost--quality frontier -- but
routing is always *across models*. The lab's novel move is the routing
lens applied *within* the lab to *prompt protocols*: the per-fixture
oracle is the candidate (protocol) that passes each fixture, and the
gap quantifies what the conservative single-winner commitment costs.
This is pure post-hoc analysis over already-stored observations: zero
new model runs, fully deterministic.

Architectural grounding (read before extending this module):

* A candidate *passes* a fixture exactly when the run's own measurement
  says so: ``exact_match and boundary_found`` on the stored search
  observation. No re-scoring, no re-judgment.
* The per-fixture oracle winner set is the passing candidates for that
  fixture, ordered by family name (deterministic tie-break): the oracle
  is "any of these would have been a correct choice here", not a
  single anointed protocol.
* The covering portfolio is computed by **greedy set cover** with
  deterministic tie-breaks (lexicographically smallest candidate name
  wins every tie). Greedy is an *approximation* -- it is not guaranteed
  to find the true minimum -- and every surface that reports the
  portfolio says so. Reporting a "minimal portfolio" without that
  caveat would be dishonest.
* Only *solvable* fixtures (at least one passing candidate) are cover
  targets: no protocol can cover a fixture nobody passes, so unsolvable
  fixtures are reported separately, never folded into the portfolio.

Two falsifiable claims, pre-registered here:

1. ``gap >= 0`` always: the oracle passes every fixture the global
   winner passes, so the oracle rate can never fall below the winner's
   rate. Pinned by a Hypothesis property, not just asserted.
2. The covering portfolio is small and concentrated: a handful of
   protocols covers nearly all solvable fixtures. This is the empirical
   claim the analysis *reports*; the module asserts nothing about it.

Everything here is pure: validation, oracle rates, and greedy cover
need no model and never touch the network.
"""

from __future__ import annotations

from functools import partial

#: Passed fixture ids per candidate, as validated by
#: :func:`validate_passing`: candidate name -> sorted tuple of fixture
#: ids the candidate passes (exact answer AND explicit boundary).
PassingMap = dict[str, tuple[str, ...]]


def validate_fixture_ids(fixture_ids: object) -> tuple[str, ...]:
    """Validate the canonical fixture id tuple.

    Returns the ids as a tuple. Raises ``ValueError`` on a non-tuple,
    an empty tuple, a non-string or blank id, or duplicate ids.
    """
    if type(fixture_ids) is not tuple or not fixture_ids:
        raise ValueError("fixture_ids must be a nonempty tuple")
    for fixture_id in fixture_ids:
        if type(fixture_id) is not str or not fixture_id:
            raise ValueError("fixture ids must be nonblank strings")
    if len(set(fixture_ids)) != len(fixture_ids):
        raise ValueError("fixture ids must be unique")
    return fixture_ids


def validate_passing(passing: object, fixture_ids: tuple[str, ...]) -> PassingMap:
    """Validate candidate -> passed-fixture-ids and normalize it.

    Returns ``{candidate: sorted unique tuple of fixture ids}``. Raises
    ``ValueError`` on: a non-dict ``passing``; an empty map; non-string
    or blank candidate names; a non-tuple passed set; a non-string
    fixture id; or a passed fixture id outside ``fixture_ids``.
    """
    fixture_ids = validate_fixture_ids(fixture_ids)
    if type(passing) is not dict or not passing:
        raise ValueError("passing must be a nonempty dict")
    known = set(fixture_ids)
    normalized: PassingMap = {}
    for name, passed in passing.items():
        if type(name) is not str or not name:
            raise ValueError("candidate names must be nonblank strings")
        if type(passed) is not tuple:
            raise ValueError(f"candidate {name!r}: passed fixtures must be a tuple")
        for fixture_id in passed:
            if type(fixture_id) is not str or not fixture_id:
                raise ValueError(f"candidate {name!r}: fixture ids must be nonblank strings")
            if fixture_id not in known:
                raise ValueError(f"candidate {name!r}: unknown fixture id {fixture_id!r}")
        normalized[name] = tuple(sorted(set(passed)))
    return normalized


def per_fixture_oracle_winners(
    passing: PassingMap, fixture_ids: tuple[str, ...]
) -> dict[str, list[str]]:
    """Per-fixture oracle winner sets; deterministic.

    For every *solvable* fixture (at least one passing candidate) the
    sorted list of passing candidate names. The name sort is the
    deterministic tie-break: the oracle offers "any of these", not a
    single protocol. Returns only solvable fixtures; unsolvable
    fixtures are not oracle-able. Inputs must already be validated
    (see :func:`validate_passing`); raises ``ValueError`` otherwise.
    """
    passing = validate_passing(passing, fixture_ids)
    winners: dict[str, list[str]] = {}
    for fixture_id in fixture_ids:
        names = sorted(name for name, passed in passing.items() if fixture_id in passed)
        if names:
            winners[fixture_id] = names
    return winners


def oracle_rates(
    passing: PassingMap,
    fixture_ids: tuple[str, ...],
    winner: str | None,
) -> dict[str, object]:
    """Oracle pass rate, winner pass rate, and the gap between them.

    ``oracle_pass_rate`` is the fraction of fixtures with at least one
    passing candidate; ``winner_pass_rate`` is the winner's own fraction
    (``None`` when there is no global winner -- then the gap is the
    oracle rate itself). ``gap = oracle - winner`` is always ``>= 0``:
    the oracle passes every fixture the winner passes. Returns
    ``{"n_fixtures", "n_solvable", "solvable_fixture_ids",
    "unsolvable_fixture_ids", "oracle_pass_rate", "winner_pass_rate",
    "gap"}``. Raises ``ValueError`` on malformed input, including a
    winner name not present in ``passing``.
    """
    passing = validate_passing(passing, fixture_ids)
    if winner is not None:
        if type(winner) is not str or not winner:
            raise ValueError("winner must be a nonblank string or None")
        if winner not in passing:
            raise ValueError(f"unknown winner {winner!r}")
    solvable = [fid for fid in fixture_ids if any(fid in passed for passed in passing.values())]
    n = len(fixture_ids)
    oracle_rate = len(solvable) / n
    if winner is None:
        winner_rate: float | None = None
        gap = oracle_rate
    else:
        winner_rate = len(passing[winner]) / n
        gap = oracle_rate - winner_rate
    return {
        "n_fixtures": n,
        "n_solvable": len(solvable),
        "solvable_fixture_ids": list(solvable),
        "unsolvable_fixture_ids": [fid for fid in fixture_ids if fid not in solvable],
        "oracle_pass_rate": oracle_rate,
        "winner_pass_rate": winner_rate,
        "gap": gap,
    }


def _coverage_size(passing: PassingMap, uncovered: set[str], name: str) -> int:
    """How many still-uncovered fixtures one candidate covers."""
    return len(set(passing[name]) & uncovered)


def greedy_cover(passing: PassingMap, fixture_ids: tuple[str, ...]) -> list[str]:
    """Greedy covering protocol portfolio; deterministic.

    Repeatedly picks the candidate covering the most still-uncovered
    *solvable* fixtures; every tie is broken by lexicographically
    smallest candidate name, so the output is fully deterministic. This
    is a greedy *approximation* of the minimum set cover -- it is not
    guaranteed optimal, and callers must report it as such. Returns the
    portfolio members in greedy pick order (empty only when no fixture
    is solvable). Raises ``ValueError`` on malformed input.
    """
    passing = validate_passing(passing, fixture_ids)
    target = {fid for fid in fixture_ids if any(fid in passed for passed in passing.values())}
    covered: set[str] = set()
    portfolio: list[str] = []
    # Bounded by the candidate count: no infinite loop is possible, and
    # no unreachable guard is needed. ``target`` is solvable by
    # construction, so every iteration with uncovered fixtures finds a
    # candidate covering at least one of them; ``max`` over the
    # name-sorted candidates returns the lexicographically smallest
    # name among equal coverages (first maximal element wins), which is
    # the deterministic tie-break.
    for _ in range(len(passing)):
        uncovered = target - covered
        if not uncovered:
            break
        best = max(
            sorted(passing),
            key=partial(_coverage_size, passing, uncovered),
        )
        portfolio.append(best)
        covered |= set(passing[best])
    return portfolio
