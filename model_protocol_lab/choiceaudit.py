"""Choice-scoring audit protocol: choice sets, verdicts, aggregation.

Why this exists: the lab's winner is chosen by greedy generation,
but the field's other scoring ontology -- teacher-forced
log-likelihood choice scoring, lm-eval's native multiple-choice
mode -- can pick a different answer on the same fixture and a
different template at the method level. Neither ontology is
privileged a priori, so the audit measures their disagreement
exactly: per search fixture a four-choice set is built from the
run's own fixtures (the expected answer plus three donor expecteds
screened so no donor is the same answer twice), every choice is
teacher-forced under every candidate over the recorded search
prompt, and the generation x choice discordance is reported per
fixture and per candidate. The audit is a second opinion: it
never selects and never feeds the selection rule.

Definitions that carry the design (red-team cross-checked,
cycle 195; the full contract is docs/CHOICE_AUDIT_DESIGN.md):

* Donors are new machinery modelled on leading's, not leading's
  reuse: leading assigns exactly one plant per fixture. The rule
  here takes the first three eligible distinct-class donors in
  ascending fixture-id order, screened with leading's own
  equivalence predicate (rung-4 plus folded containment) against
  every expected in the target's group. There is no
  numeric-perturbation fallback: a fabricated string is not a
  plausible choice, and the fallback would relieve shortfall
  selectively for numeric fixtures.
* A fixture's choice verdict exists only when every constructed
  cell scored ``ok`` with finite reported values. Partial sets
  are never argmaxed: unavailability correlates with the
  choice's surface form and with the candidate's rendering, so
  a surviving-subset argmax is an easier, different task.
* Three rankings of the same cells: total log-likelihood (the
  headline; lm-eval ``acc`` semantics), per-byte (reported total
  over target byte length; lm-eval ``acc_norm`` semantics), and
  per-token mean (the quantity the harness deliberately does not
  report -- carried because the engine produces it and the
  comparison is intra-checkpoint, and never labelled
  ``acc_norm``). Verdicts are computed from the reported
  (float32-rounded) values only; the reported mean is never
  re-derived from the reported total, because the engine rounds
  the mean from the unrounded total.
* The method-level headline is computed only over the common
  evaluable population (fixtures with a choice verdict under
  every candidate), floored by a probe-local constant, with
  ``n_common`` always reported: per-candidate denominators
  shrink unevenly by construction, and comparing accuracies
  across different fixture subsets is the pooled-eligibility
  trap two shipped probes already withhold winners over.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation
from typing import Any

from .core import Fixture
from .leading import equivalent_claim
from .scorerlens import NUMERIC_GRAMMAR, ladder_forms, rung_match

PROTOCOL_VERSION = 1

#: Donors per choice set, beside the true answer.
K_DONORS = 3

#: Frozen donor selection rule name (protocol data).
DONOR_RULE = "smallest-eligible-id-first-three-v1"

#: The equivalence screen, literally leading's predicate.
SCREEN = "scorerlens-r4-plus-containment-v1"

#: Reported rankings, in payload order. "byte" is lm-eval's
#: acc_norm semantics; "mean" is never labelled acc_norm.
RANKINGS: tuple[str, ...] = ("total", "byte", "mean")

#: Probe-local floor for the method-level headline. The value
#: mirrors leading's paired floor, but constants are never
#: imported across probes -- neither probe's constant may
#: silently move the other's.
MIN_COMMON_EVALUABLE = 6

#: Ground-record caps, checked before any scoring (the house
#: pattern). Records are per (candidate, fixture), so the cap
#: fits the legal maximum (24 candidates x 100 fixtures)
#: exactly: no legal scale is made unavailable by it.
MAX_RECORDS = 2400
MAX_RECORDS_BYTES = 12 * 1024 * 1024

DETAIL_BUDGET_EXCEEDED = "choiceaudit_cell_budget_exceeded"
DETAIL_BELOW_FLOOR = "choiceaudit_common_population_below_floor"
DETAIL_NO_EVALUABLE = "no_evaluable_fixtures"

#: A cell whose reported values are non-finite is demoted to
#: unavailable under this fixed reason (producer and validator
#: vocabulary alike): verbatim -inf would crash the
#: allow_nan=False report write.
REASON_NON_FINITE = "non_finite_score"

#: No usable recorded prompt for a (candidate, fixture).
REASON_OBSERVATION_MISSING = "observation_missing"

_NUMERIC_RE = re.compile(NUMERIC_GRAMMAR)


def text_digest(text: str) -> str:
    """SHA-256 of a choice text (records carry digests, not texts).

    ``surrogatepass`` keeps a pathological expected (a lone
    surrogate can survive the CSV loader) from crashing the
    audit; for every well-formed string the digest is the plain
    UTF-8 digest.
    """
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def is_numeric_answer(text: str) -> bool:
    """Whether ``text`` is numeric under the ladder's frozen grammar.

    Mirrors scorerlens' R4 value parse exactly (grammar gate on
    the R3 form, then a finite Decimal) using only its public
    data, so the stratum split and the screen can never drift
    apart.
    """
    form = ladder_forms(text)[3]
    if not _NUMERIC_RE.match(form):
        return False
    try:
        value = Decimal(form.replace(",", ""))
    except InvalidOperation:
        return False
    return value.is_finite()


def _exclusion(case: Fixture) -> str | None:
    """Why a fixture is excluded from every choice population.

    The scorer-lens exclusion, adopted verbatim: the expected
    normalises to empty at rung 1 (whose first step is a trim,
    so strip-empty is the same set). A blank true choice would
    score ``ok_empty`` with total 0.0 -- the maximum attainable
    -- and win its set by construction. Excluded fixtures are
    neither targets nor donors.
    """
    if ladder_forms(case.expected)[1] == "":
        return "blank_expected"
    return None


# skipcq: PY-R1000 - Gate-pinned frozen function; splitting breaks its verified contract.
def build_choice_sets(cases: Sequence[Fixture]) -> list[dict[str, Any]]:
    """The per-fixture choice construction block (pure, deterministic).

    One entry per search fixture, in split order: exclusion
    reason if any, the realised choices (true first, then donors
    in ascending fixture-id order), the distinct-choice count,
    and whether the set is complete (true + K_DONORS distinct
    donor classes). Shortfall is recorded, never repaired.
    """
    excluded = {case.id: _exclusion(case) for case in cases}
    group_expecteds: dict[str, list[str]] = {}
    for case in cases:
        group_expecteds.setdefault(case.group, []).append(case.expected)
    live = [case for case in cases if excluded[case.id] is None]
    block: list[dict[str, Any]] = []
    for case in cases:
        entry: dict[str, Any] = {
            "fixture": case.id,
            "group": case.group,
            "excluded": excluded[case.id],
            "numeric": is_numeric_answer(case.expected),
            "choices": [],
            "n_distinct_choices": 0,
            "constructible": False,
        }
        if excluded[case.id] is None:
            choices: list[dict[str, Any]] = [
                {
                    "source_fixture": None,
                    "group": case.group,
                    "digest": text_digest(case.expected),
                }
            ]
            chosen_texts = [case.expected]
            pool = sorted(
                (donor for donor in live if donor.group != case.group),
                key=lambda donor: donor.id,
            )
            for donor in pool:
                if len(choices) == 1 + K_DONORS:
                    break
                # Screen: non-equivalent to every expected in the
                # target's group (leading's group-wide rule).
                if any(
                    equivalent_claim(donor.expected, expected)
                    for expected in group_expecteds[case.group]
                ):
                    continue
                # Dedup: byte identity against any chosen text,
                # then rung-4 class against chosen donors (the
                # true answer's class is already screened out).
                if any(donor.expected == text for text in chosen_texts):
                    continue
                if any(rung_match(donor.expected, text, 4) for text in chosen_texts[1:]):
                    continue
                choices.append(
                    {
                        "source_fixture": donor.id,
                        "group": donor.group,
                        "digest": text_digest(donor.expected),
                    }
                )
                chosen_texts.append(donor.expected)
            entry["choices"] = choices
            entry["n_distinct_choices"] = len(choices)
            entry["constructible"] = len(choices) == 1 + K_DONORS
        block.append(entry)
    return block


def donor_map_sha256(block: Sequence[dict[str, Any]]) -> str:
    """Digest binding the whole construction block.

    Editing fixture B's expected changes fixture A's choice
    set, so the construction is part of the treatment and is
    digested as such (leading's donor-map lesson).
    """
    canonical = json.dumps(list(block), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def fixture_set_sha256(fixture_ids: Sequence[str]) -> str:
    """SHA-256 over the sorted fixture id set (leading's form)."""
    return hashlib.sha256("\n".join(sorted(fixture_ids)).encode("utf-8")).hexdigest()


def protocol_object(
    fixture_ids: Sequence[str],
    block: Sequence[dict[str, Any]],
    backend_fingerprint: str,
) -> dict[str, Any]:
    """The frozen protocol as data; the fingerprint covers it."""
    return {
        "version": PROTOCOL_VERSION,
        "donor_rule": DONOR_RULE,
        "screen": SCREEN,
        "k_donors": K_DONORS,
        "rankings": list(RANKINGS),
        "min_common_evaluable": MIN_COMMON_EVALUABLE,
        "max_records": MAX_RECORDS,
        "max_records_bytes": MAX_RECORDS_BYTES,
        "fixture_set_sha256": fixture_set_sha256(fixture_ids),
        "donor_map_sha256": donor_map_sha256(block),
        "backend_fingerprint": backend_fingerprint,
    }


def protocol_fingerprint(
    fixture_ids: Sequence[str],
    block: Sequence[dict[str, Any]],
    backend_fingerprint: str,
) -> str:
    """SHA-256 over the canonical protocol object."""
    canonical = json.dumps(
        protocol_object(fixture_ids, block, backend_fingerprint),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def generation_verdict(status: Any, exact_match: Any, boundary_found: Any) -> str:
    """The generation half of the discordance table, from stored facts.

    ``pass`` iff a completed observation with exact match and
    boundary found; ``fail`` iff completed without both;
    ``missing`` otherwise. In genuine runs exact_match is
    construction-gated on completion, so the status check costs
    nothing there -- it exists so a foreign payload cannot
    convert a missing measurement into a generation fail.
    """
    if status != "completed":
        return "missing"
    return "pass" if (exact_match is True and boundary_found is True) else "fail"


def ranking_values(cells: Sequence[dict[str, Any]], ranking: str) -> list[float]:
    """One ranking's per-choice values, from the reported cell values.

    The per-byte value is *defined* as the reported total over
    the target byte length, computed here so producer and
    validator agree bit-for-bit. Callers must have established
    evaluability (every cell ok and finite, byte length >= 1).
    """
    if ranking == "total":
        return [float(cell["total"]) for cell in cells]
    if ranking == "mean":
        return [float(cell["mean"]) for cell in cells]
    if ranking == "byte":
        return [float(cell["total"]) / int(cell["bytes"]) for cell in cells]
    raise ValueError(f"unknown ranking: {ranking!r}")


def rank_verdict(values: Sequence[float]) -> str:
    """``correct`` / ``tied`` / ``incorrect`` for the true choice (index 0).

    A tie is exact equality at the reported maximum with at
    least one other choice; ties are never broken and never
    counted correct.
    """
    top = max(values)
    if values[0] < top:
        return "incorrect"
    return "tied" if sum(1 for value in values if value == top) > 1 else "correct"


def record_evaluable(record: dict[str, Any], fixture_entry: dict[str, Any]) -> bool:
    """Whether a (candidate, fixture) record carries a choice verdict.

    Requires a constructible set, four recorded cells, and every
    cell exactly ``ok`` with finite reported total and mean.
    An ``ok_empty`` cell is not ``ok``: construction excludes
    empty choices, so one appearing anyway fails evaluability
    and is counted under its own status, never folded into
    alignment unavailability.
    """
    if not fixture_entry["constructible"]:
        return False
    cells = record["cells"]
    if len(cells) != 1 + K_DONORS:
        return False
    return all(
        cell["status"] == "ok"
        and type(cell["total"]) is float
        and math.isfinite(cell["total"])
        and type(cell["mean"]) is float
        and math.isfinite(cell["mean"])
        for cell in cells
    )


def _accuracy_block(verdicts: list[str]) -> dict[str, Any]:
    """Correct/incorrect/tied counts with the rate beside its denominator."""
    correct = sum(1 for verdict in verdicts if verdict == "correct")
    incorrect = sum(1 for verdict in verdicts if verdict == "incorrect")
    tied = sum(1 for verdict in verdicts if verdict == "tied")
    n = len(verdicts)
    return {
        "correct": correct,
        "incorrect": incorrect,
        "tied": tied,
        "n": n,
        "accuracy": (correct / n) if n else None,
        "detail": "" if n else DETAIL_NO_EVALUABLE,
    }


# skipcq: PY-R1000 - Gate-pinned frozen function; splitting breaks its verified contract.
def aggregate_candidate(
    records: list[dict[str, Any]], fixtures: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Per-candidate aggregates over its ground records.

    Every rate carries its denominator; a rate is None only at
    denominator zero, with a fixed detail. The discordance
    table is the total ranking's; the other rankings report
    accuracies and their verdict agreement with the total.
    """
    by_id = {entry["fixture"]: entry for entry in fixtures}
    verdicts: dict[str, list[str]] = {ranking: [] for ranking in RANKINGS}
    verdict_by_fixture: dict[str, dict[str, str]] = {}
    unavailable_reasons: dict[str, int] = {}
    n_constructible = 0
    n_evaluable = 0
    n_prompt_failures = 0
    n_generation_missing = 0
    paired = {
        "gen_pass_choice_correct": 0,  # nosec B105 -- verdict-count schema key, not a credential
        "gen_pass_choice_incorrect": 0,  # nosec B105 -- verdict-count schema key, not a credential
        "gen_fail_choice_correct": 0,
        "gen_fail_choice_incorrect": 0,
    }
    n_choice_tied = 0
    gen_fail_boundary_found = 0
    gen_fail_boundary_not_found = 0
    strata_verdicts: dict[str, list[str]] = {"numeric": [], "non_numeric": [], "length_matched": []}
    agreement: dict[str, dict[str, int]] = {
        "byte_vs_total": {"agree": 0, "disagree": 0},
        "mean_vs_total": {"agree": 0, "disagree": 0},
    }
    for record in records:
        entry = by_id[record["fixture"]]
        gen = generation_verdict(
            record["obs_status"], record["exact_match"], record["boundary_found"]
        )
        if gen == "missing":
            n_generation_missing += 1
        if not entry["constructible"]:
            continue
        n_constructible += 1
        if record["prompt_reason"] is not None:
            n_prompt_failures += 1
            reason = str(record["prompt_reason"])
            unavailable_reasons[reason] = unavailable_reasons.get(reason, 0) + 1
        if not record_evaluable(record, entry):
            for cell in record["cells"]:
                if cell["status"] == "unavailable":
                    reason = str(cell["reason"])
                    unavailable_reasons[reason] = unavailable_reasons.get(reason, 0) + 1
                elif cell["status"] != "ok":
                    unavailable_reasons[cell["status"]] = (
                        unavailable_reasons.get(cell["status"], 0) + 1
                    )
            continue
        n_evaluable += 1
        cells = record["cells"]
        per_ranking = {
            ranking: rank_verdict(ranking_values(cells, ranking)) for ranking in RANKINGS
        }
        verdict_by_fixture[record["fixture"]] = per_ranking
        for ranking in RANKINGS:
            verdicts[ranking].append(per_ranking[ranking])
        strata_verdicts["numeric" if entry["numeric"] else "non_numeric"].append(
            per_ranking["total"]
        )
        true_tokens = int(cells[0]["tokens"])
        if all(abs(int(cell["tokens"]) - true_tokens) <= 1 for cell in cells[1:]):
            strata_verdicts["length_matched"].append(per_ranking["total"])
        for key, ranking in (("byte_vs_total", "byte"), ("mean_vs_total", "mean")):
            same = per_ranking[ranking] == per_ranking["total"]
            agreement[key]["agree" if same else "disagree"] += 1
        if gen in ("pass", "fail"):
            total_verdict = per_ranking["total"]
            if total_verdict == "tied":
                n_choice_tied += 1
            else:
                paired[f"gen_{gen}_choice_{total_verdict}"] += 1
                if gen == "fail":
                    if record["boundary_found"] is True:
                        gen_fail_boundary_found += 1
                    else:
                        gen_fail_boundary_not_found += 1
    n_paired = sum(paired.values())
    n_discordant = paired["gen_pass_choice_incorrect"] + paired["gen_fail_choice_correct"]
    return {
        "n_fixtures": len(fixtures),
        "n_constructible": n_constructible,
        "n_evaluable": n_evaluable,
        "n_choice_unavailable": n_constructible - n_evaluable,
        "n_prompt_failures": n_prompt_failures,
        "unavailable_reasons": unavailable_reasons,
        "n_generation_missing": n_generation_missing,
        "accuracy": {ranking: _accuracy_block(verdicts[ranking]) for ranking in RANKINGS},
        "discordance": {
            "n_paired": n_paired,
            **paired,
            "n_discordant": n_discordant,
            "n_choice_tied": n_choice_tied,
            "gen_fail_boundary_found": gen_fail_boundary_found,
            "gen_fail_boundary_not_found": gen_fail_boundary_not_found,
        },
        "strata": {
            "numeric": _accuracy_block(strata_verdicts["numeric"]),
            "non_numeric": _accuracy_block(strata_verdicts["non_numeric"]),
            "length_matched": _accuracy_block(strata_verdicts["length_matched"]),
            "ranking_agreement": agreement,
        },
        "_verdicts": verdict_by_fixture,
    }


# skipcq: PY-R1000 - Gate-pinned frozen function; splitting breaks its verified contract.
def method_comparison(
    aggregates: dict[str, dict[str, Any]],
    candidate_names: Sequence[str],
    primary_decision: Any,
    primary_winner: Any,
) -> dict[str, Any]:
    """The method-level headline over the common evaluable population.

    ``aggregates`` maps candidate name to its
    :func:`aggregate_candidate` output (its ``_verdicts`` entry
    carries the per-fixture total verdicts). The agreement
    value is multi-state with frozen priority: withheld (below
    floor) -> native_abstain -> choice_tied -> agree/disagree.
    The native outcome is mapped exactly as leading maps it:
    ``compatible_on_fixture_set`` and ``held_out_failed`` map
    to ``(selected, selected_candidate)``; any other decision
    passes through with its winner.
    """
    evaluable_sets = [
        set(aggregates[name]["_verdicts"]) if name in aggregates else set()
        for name in candidate_names
    ]
    common = set.intersection(*evaluable_sets) if evaluable_sets else set()
    n_common = len(common)
    common_accuracy: dict[str, Any] = {}
    for name in candidate_names:
        per_fixture = aggregates[name]["_verdicts"] if name in aggregates else {}
        correct = sum(
            1 for fixture in common if per_fixture.get(fixture, {}).get("total") == "correct"
        )
        common_accuracy[name] = {
            "correct": correct,
            "n": n_common,
            "accuracy": (correct / n_common) if n_common else None,
        }
    if primary_decision in ("compatible_on_fixture_set", "held_out_failed"):
        mapped_decision: Any = "selected"
        mapped_winner: Any = primary_winner
    else:
        mapped_decision = primary_decision
        mapped_winner = primary_winner
    withheld = n_common < MIN_COMMON_EVALUABLE
    top_candidate: str | None = None
    top_tied: list[str] = []
    if not withheld and n_common:
        best = max(common_accuracy[name]["correct"] for name in candidate_names)
        leaders = sorted(
            name for name in candidate_names if common_accuracy[name]["correct"] == best
        )
        if len(leaders) == 1:
            top_candidate = leaders[0]
        else:
            top_tied = leaders
    if withheld:
        agreement = "withheld"
    elif mapped_decision != "selected" or type(mapped_winner) is not str:
        agreement = "native_abstain"
    elif top_tied:
        agreement = "choice_tied"
    elif top_candidate == mapped_winner:
        agreement = "agree"
    else:
        agreement = "disagree"
    return {
        "n_common": n_common,
        "floor": MIN_COMMON_EVALUABLE,
        "withheld": withheld,
        "detail": DETAIL_BELOW_FLOOR if withheld else "",
        "common_accuracy": common_accuracy,
        "choice_top_candidate": top_candidate,
        "choice_top_tied": top_tied,
        "primary_decision": mapped_decision,
        "primary_winner": mapped_winner,
        "agreement": agreement,
    }


def discordance_summary(aggregates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The probe-level falsification readout over all candidates.

    ``falsified`` is None when the paired population is empty:
    zero discordance over no cells falsifies nothing, and the
    payload says so instead of claiming the null.
    """
    n_cells = sum(entry["discordance"]["n_paired"] for entry in aggregates.values())
    n_discordant = sum(entry["discordance"]["n_discordant"] for entry in aggregates.values())
    return {
        "n_cells": n_cells,
        "n_discordant": n_discordant,
        "falsified": (n_discordant == 0) if n_cells else None,
    }


__all__ = [
    "DETAIL_BELOW_FLOOR",
    "DETAIL_BUDGET_EXCEEDED",
    "DETAIL_NO_EVALUABLE",
    "DONOR_RULE",
    "K_DONORS",
    "MAX_RECORDS",
    "MAX_RECORDS_BYTES",
    "MIN_COMMON_EVALUABLE",
    "PROTOCOL_VERSION",
    "RANKINGS",
    "REASON_NON_FINITE",
    "REASON_OBSERVATION_MISSING",
    "SCREEN",
    "aggregate_candidate",
    "build_choice_sets",
    "discordance_summary",
    "donor_map_sha256",
    "fixture_set_sha256",
    "generation_verdict",
    "is_numeric_answer",
    "method_comparison",
    "protocol_fingerprint",
    "protocol_object",
    "rank_verdict",
    "ranking_values",
    "record_evaluable",
    "text_digest",
]
