"""Prompt-boundary canonicality audit (research backlog R2).

The Prompt Boundary Problem (PBP): a tokenizer encodes ``prompt_text`` and
``completion_text`` separately during a run, but the concatenated text may
re-encode to a *different* token sequence — the boundary between the two
spans is not canonical. ByteSampler (arXiv 2506.14123) is the exact fix for
this problem, and the companion measurement study (arXiv 2601.23223v1)
shows models predict the correct next token 60-95.6% less often at
unexpected token boundaries. No evaluation harness reports
boundary-canonicality as a template-quality metric; this module does,
using only artifacts the lab already stores.

The check is purely deterministic: ``encode(prompt_text + completion_text)``
must equal ``prompt_ids + completion_ids``. ``encode`` and ``decode`` are
injected callables (the backend's, which encode with no implicit BOS/EOS),
so the core is fully testable without any model or tokenizer.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BoundaryCheck:
    """Outcome of one prompt/completion boundary check."""

    has_problem: bool
    # Index into the id sequence where the joint encoding first diverges
    # from prompt_ids + completion_ids; None when there is no problem.
    first_divergence_index: int | None
    # False when encode/decode failed or inputs were unusable (excluded
    # from rate denominators, counted separately).
    usable: bool = True


def _exact_ids(values: Any) -> tuple[int, ...] | None:
    """A recorded id sequence, or None when it is not exactly ints.

    The lab's validators demand exact ``int`` for recorded token
    ids (``_checked_token_ids``, ``is_evaluable_prompt``): a bare
    ``int()`` conversion silently coerces strings and truncates
    floats into measurements of data that was never recorded, and
    an infinite float escapes as ``OverflowError`` — neither is a
    usable check (BH14-3). Bool members are not ids either
    (``type(True) is bool``).
    """
    if type(values) not in (list, tuple):
        return None
    if any(type(item) is not int for item in values):
        return None
    return tuple(values)


def check_boundary(
    prompt_text: str,
    prompt_ids: Sequence[int],
    completion_ids: Sequence[int],
    encode: Callable[[str], Sequence[int]],
    decode: Callable[[tuple[int, ...]], str],
) -> BoundaryCheck:
    """Check whether the prompt/completion boundary is token-canonical.

    Decodes ``completion_ids`` to the completion text, then requires
    ``encode(prompt_text + completion_text) == prompt_ids + completion_ids``.
    Any encode/decode failure (or non-str input) yields an unusable check,
    never a crash and never a false problem.
    """
    if type(prompt_text) is not str:
        return BoundaryCheck(has_problem=False, first_divergence_index=None, usable=False)
    prompt_seq = _exact_ids(prompt_ids)
    completion_seq = _exact_ids(completion_ids)
    if prompt_seq is None or completion_seq is None:
        return BoundaryCheck(has_problem=False, first_divergence_index=None, usable=False)
    try:
        completion_text = decode(completion_seq)
        if type(completion_text) is not str:
            return BoundaryCheck(has_problem=False, first_divergence_index=None, usable=False)
        joint = _exact_ids(encode(prompt_text + completion_text))
        if joint is None:
            return BoundaryCheck(has_problem=False, first_divergence_index=None, usable=False)
    except Exception:  # noqa: BLE001 -- backend encode/decode may raise; unusable, not fatal
        return BoundaryCheck(has_problem=False, first_divergence_index=None, usable=False)
    expected = prompt_seq + completion_seq
    if joint == expected:
        return BoundaryCheck(has_problem=False, first_divergence_index=None, usable=True)
    # First index where the sequences differ; a length mismatch diverges at
    # the end of the shorter sequence.
    limit = min(len(joint), len(expected))
    for i in range(limit):
        if joint[i] != expected[i]:
            return BoundaryCheck(has_problem=True, first_divergence_index=i, usable=True)
    return BoundaryCheck(has_problem=True, first_divergence_index=limit, usable=True)


def boundary_problem_rate(checks: Sequence[BoundaryCheck]) -> tuple[int, int, float | None]:
    """Aggregate usable checks into (n_problem, n_usable, rate).

    Rate is None when there are no usable checks.
    """
    usable = [c for c in checks if c.usable]
    n_problem = sum(1 for c in usable if c.has_problem)
    if not usable:
        return 0, 0, None
    return n_problem, len(usable), n_problem / len(usable)


def _family_of(candidate_name: object) -> str:
    """Base family of a candidate name ("chatml/x" -> "chatml")."""
    if type(candidate_name) is not str or not candidate_name:
        return "unknown"
    return candidate_name.split("/", 1)[0] or "unknown"


def audit_search_observations(
    search_entries: Sequence[dict[str, Any]],
    encode: Callable[[str], Sequence[int]],
    decode: Callable[[tuple[int, ...]], str],
) -> dict[str, Any]:
    """Compute the prompt-boundary-problem rate over search observations.

    ``search_entries`` are the ``result["search"]`` dicts (each carrying a
    ``format`` dict with a ``name`` and an ``observations`` list of dicts
    with ``input_text``/``input_ids``/``output_ids``). Returns a JSON-safe
    dict with the overall rate, per-candidate rates, and per-family rates.
    Observations without a recorded prompt text are unusable (excluded from
    denominators, counted separately). Never raises: backend encode/decode
    failures degrade individual checks to unusable.
    """
    per_candidate: dict[str, dict[str, Any]] = {}
    candidate_checks: dict[str, list[BoundaryCheck]] = {}
    candidate_counts: dict[str, int] = {}
    family_checks: dict[str, list[BoundaryCheck]] = {}
    n_observations = 0
    try:
        entries = list(search_entries)
    except TypeError:
        entries = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        fmt = entry.get("format")
        name = fmt.get("name") if isinstance(fmt, dict) else None
        if type(name) is not str or not name:
            continue
        observations = entry.get("observations")
        if not isinstance(observations, (list, tuple)):
            continue
        checks: list[BoundaryCheck] = []
        n_candidate_obs = 0
        for obs in observations:
            if not isinstance(obs, dict):
                continue
            n_observations += 1
            n_candidate_obs += 1
            prompt_text = obs.get("input_text")
            if type(prompt_text) is not str or not prompt_text:
                continue  # no recorded prompt: unusable, not a problem
            checks.append(
                check_boundary(
                    prompt_text,
                    obs.get("input_ids") or (),
                    obs.get("output_ids") or (),
                    encode,
                    decode,
                )
            )
        # Duplicate format names merge into the one candidate
        # rollup — counts and checks accumulate, never overwrite —
        # so the per-candidate sums conserve against the overall
        # counts (SA-02's principle one level down, BH14-4), the
        # same accumulation the per-family rollup already uses.
        candidate_checks.setdefault(name, []).extend(checks)
        candidate_counts[name] = candidate_counts.get(name, 0) + n_candidate_obs
        family_checks.setdefault(_family_of(name), []).extend(checks)
    for name, checks_for_name in candidate_checks.items():
        n_problem, n_usable, rate = boundary_problem_rate(checks_for_name)
        per_candidate[name] = {
            "n_observations": candidate_counts[name],
            "n_usable": n_usable,
            "n_problem": n_problem,
            "rate": rate,
        }
    all_checks = [c for checks in family_checks.values() for c in checks]
    n_problem, n_usable, rate = boundary_problem_rate(all_checks)
    per_family = {}
    for family in sorted(family_checks):
        fp, fu, fr = boundary_problem_rate(family_checks[family])
        per_family[family] = {"n_usable": fu, "n_problem": fp, "rate": fr}
    # Conservation (SA-02): every observation is either usable or
    # unusable — the no-prompt exclusions AND the checks that
    # degraded to unusable on a tokenizer failure. Subtracting all
    # appended checks (the pre-fix formula) made the failed checks
    # vanish from the accounting.
    n_unusable = n_observations - n_usable
    return {
        "available": n_usable > 0,
        "n_observations": n_observations,
        "n_usable": n_usable,
        "n_unusable": n_unusable,
        "n_problem": n_problem,
        "overall_rate": rate,
        "per_candidate": per_candidate,
        "per_family": per_family,
    }
