"""diffreports runlevel family: split from diffreports.py (W7-2).

Facades keep the public surface: import from model_protocol_lab.diffreports, not from this module.
"""

import hashlib
import json
from typing import Any

from .diffreports_core import (
    _diff_labeled_block,
    _diff_spans,
    _display_id,
    _float_convertible,
    _measurable_collision_cells,
    _normalize_attribution,
    _operator_view,
    _positive_number,
    _recomputed_equal,
    _share_in_unit,
    _stability_triplet,
    _step_function,
)


def _serving_ids_state(section: Any) -> tuple[bool, bool]:
    """(Any records reachable, any id streams carried) for a
    ``serving_backends`` section: its records sit at the section
    level, and the evidence the projection strips is the
    ``prompt_ids``/``returned_ids`` key pair on each record.
    """
    saw = False
    carried = False
    if type(section) is dict:
        records = section.get("records")
        if type(records) is list:
            for record in records:
                if type(record) is dict:
                    saw = True
                    if "prompt_ids" in record or "returned_ids" in record:
                        carried = True
    return saw, carried


def _valid_frontier(value: Any) -> bool:
    """Light validation: is this a comparable ``budget_frontier`` payload?

    Also bounds the total validated step-point count at
    ``reporting.MAX_FRONTIER_STEP_POINTS``: a foreign payload can carry
    millions of ``pass_steps`` points inside the 32 MiB ``results.json``
    cap, and the span/leader derivations are superlinear in that count --
    an unbounded hang (perf finding 1A). Beyond the bound the frontier
    degrades to one-sided instead of hanging the comparison. The import
    is lazy and absolute: this module is also loaded standalone by the
    cycle-67 mutation-gate harness (a bare copy in a temp dir with no
    parent package), where a module-level relative import fails.
    """
    from model_protocol_lab.reporting import MAX_FRONTIER_STEP_POINTS  # noqa: PLC0415

    if type(value) is not dict:
        return False
    candidates = value.get("candidates")
    if type(candidates) is not dict or not candidates:
        return False
    max_tokens = value.get("max_new_tokens")
    if type(max_tokens) is not int or max_tokens <= 0:
        return False
    # Shape before consumption (RR-02): the summary renderer maps
    # budget_robust through a {True/False/None} lookup, so a list
    # raised TypeError there after the diff itself had succeeded; a
    # scalar flip_ranges likewise raised inside the floor derivation.
    # A present, non-None field must carry its recorded type.
    robust = value.get("budget_robust")
    if robust is not None and type(robust) is not bool:
        return False
    flip_ranges = value.get("flip_ranges")
    if flip_ranges is not None and type(flip_ranges) is not list:
        return False
    found = False
    total_points = 0
    for data in candidates.values():
        if type(data) is not dict:
            continue
        # Shape before consumption (wave-8 F4): the `or []` idiom
        # rescues only falsy values, so a truthy scalar fixture_ids /
        # pass_steps raised TypeError inside this validator. A
        # non-list contributes no points -- the same reading
        # reporting._fixture_total gives a non-list fixture_ids.
        fixture_ids = data.get("fixture_ids")
        if type(fixture_ids) is not list or len(fixture_ids) <= 0:
            continue
        pass_steps = data.get("pass_steps")
        if type(pass_steps) is not list:
            continue
        for point in pass_steps:
            if (
                type(point) is list
                and len(point) == 2
                and type(point[0]) is int
                and type(point[1]) is int
                and 1 <= point[0] <= max_tokens
            ):
                found = True
                total_points += 1
                if total_points > MAX_FRONTIER_STEP_POINTS:
                    return False
    return found


def _budget_side(payload: Any, result: Any) -> dict[str, Any] | None:
    """Validate one side's budget ledger against its own observations.

    Returns the recorded ledger fields, or ``None`` when the payload
    is missing/malformed OR its recorded actuals disagree with the
    run's stored observations: a ledger that contradicts its own
    evidence is not comparable. The cross-field invariants are the
    producer's (``experiments.run_experiment``): an estimate is
    recorded exactly when a cap was set, a started run's estimate
    never exceeded its cap, and its actual spend never exceeded the
    estimate (the preflight bound is a true upper bound).
    """
    if type(payload) is not dict or type(result) is not dict:
        return None
    cap = payload.get("max_total_tokens")
    estimate = payload.get("estimated_total_tokens")
    actual = payload.get("actual_total_tokens")
    planned = payload.get("cells_planned")
    executed = payload.get("cells_executed")
    if cap is not None and (type(cap) is not int or cap < 1):
        return None
    if estimate is not None and (type(estimate) is not int or estimate < 0):
        return None
    if (cap is None) != (estimate is None):
        return None
    if cap is not None and estimate is not None and estimate > cap:
        return None
    if type(actual) is not int or actual < 0:
        return None
    if type(planned) is not int or planned < 0:
        return None
    if type(executed) is not int or executed < 0 or executed > planned:
        return None
    if payload.get("estimate_includes_held_out") is not True:
        return None
    search = result.get("search")
    if type(search) is not list:
        return None
    observation_lists: list[Any] = []
    for entry in search:
        if type(entry) is not dict or type(entry.get("observations")) is not list:
            return None
        observation_lists.append(entry["observations"])
    held = result.get("held_out")
    if type(held) is dict:
        if type(held.get("observations")) is not list:
            return None
        observation_lists.append(held["observations"])
    recomputed_actual = 0
    recomputed_cells = 0
    for observations in observation_lists:
        for obs in observations:
            if type(obs) is not dict:
                return None
            input_ids = obs.get("input_ids")
            output_ids = obs.get("output_ids")
            # Loaded documents carry lists; in-memory results carry
            # the tuples asdict produced. Both are the same document
            # at different pipeline stages, so both are accepted.
            # (isinstance, not type(): mypy narrows Any | None through
            # isinstance only — see the workspace lessons file.)
            if not isinstance(input_ids, (list, tuple)) or not isinstance(
                output_ids, (list, tuple)
            ):
                return None
            recomputed_actual += len(input_ids) + len(output_ids)
            recomputed_cells += 1
    if recomputed_actual != actual or recomputed_cells != executed:
        return None
    if estimate is not None and actual > estimate:
        return None
    return {
        "max_total_tokens": cap,
        "estimated_total_tokens": estimate,
        "actual_total_tokens": actual,
        "cells_planned": planned,
        "cells_executed": executed,
    }


def diff_budgets(old: Any, new: Any, old_result: Any, new_result: Any) -> dict[str, Any]:
    """Compare two runs' budget ledgers (the RR-14a mirror).

    Unlike the other section differs, this one receives the whole
    results dicts: the ledger's recorded actuals are only meaningful
    against the observations they claim to sum, so each side is
    first validated by recomputation from its own search/held-out
    observations (:func:`_budget_side`). A side whose ledger
    contradicts its observations — or is malformed or absent — is
    reported through the availability enum instead of compared, so a
    one-sided tamper surfaces as asymmetry and breaks the identical
    verdict rather than passing as a match. On evidence-excluded
    exports (the default public view strips observations) neither
    side can be recomputed, so the section degrades to unvalidated —
    the same documented behaviour as the other
    observation-recomputing sections; evidence exports are the
    diff inputs that carry the full check.
    """
    old_side = _budget_side(old, old_result)
    new_side = _budget_side(new, new_result)
    if old_side is not None and new_side is not None:
        available = "both"
    elif old_side is not None:
        available = "old"
    elif new_side is not None:
        available = "new"
    else:
        available = "neither"
    result: dict[str, Any] = {"available": available, "changed": False}
    if old_side is None or new_side is None:
        return result
    changed = False
    for field in (
        "max_total_tokens",
        "estimated_total_tokens",
        "actual_total_tokens",
        "cells_planned",
        "cells_executed",
    ):
        field_changed = old_side[field] != new_side[field]
        result[field] = {
            "old": old_side[field],
            "new": new_side[field],
            "changed": field_changed,
        }
        changed = changed or field_changed
    result["changed"] = changed
    return result


def diff_frontiers(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``budget_frontier`` payloads; deterministic.

    Answers "did the token-budget robustness story change between runs?":
    sweep-range drift, budget-robustness flips, and the exact budget spans
    where the post-hoc decision, post-hoc winner, or per-budget leader
    (from :func:`reporting.margin_trajectory`) differ -- plus, per flip
    span, whether the attributed (candidate, fixture) pass-status changes
    differ. A side without a recorded frontier is reported plainly
    instead of compared; malformed decision steps skip the decision/winner
    axes rather than inventing differences. No model, no network.
    """
    old_ok = _valid_frontier(old)
    new_ok = _valid_frontier(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    result: dict[str, Any] = {"available": available, "changed": False}
    if available != "both":
        return result
    old_max = old["max_new_tokens"]
    new_max = new["max_new_tokens"]
    result["max_new_tokens"] = {
        "old": old_max,
        "new": new_max,
        "changed": old_max != new_max,
    }
    old_robust = old.get("budget_robust")
    new_robust = new.get("budget_robust")
    result["budget_robust"] = {
        "old": old_robust,
        "new": new_robust,
        "changed": old_robust != new_robust,
    }

    def decision_points(frontier: dict[str, Any]) -> list[tuple[int, tuple[str, Any]]]:
        # Duplicate budgets are contradictory data (native payloads never
        # duplicate); the first occurrence wins so a foreign payload can
        # never crash the comparison (perf finding 1B).
        """Return the decision points."""
        points = []
        seen_budgets: set[int] = set()
        steps = frontier.get("decision_steps")
        for entry in steps if type(steps) is list else ():
            if (
                type(entry) is list
                and len(entry) == 3
                and type(entry[0]) is int
                and type(entry[1]) is str
                and 1 <= entry[0] <= frontier["max_new_tokens"]
                and entry[0] not in seen_budgets
            ):
                seen_budgets.add(entry[0])
                points.append((entry[0], (entry[1], entry[2])))
        return points

    old_dec = decision_points(old)
    new_dec = decision_points(new)
    hi = min(old_max, new_max)
    breakpoints = [b for b, _ in old_dec] + [b for b, _ in new_dec]
    decision_flips: list[list[Any]] = []
    winner_flips: list[list[Any]] = []
    if old_dec and new_dec:
        old_dec_at = _step_function([(b, v[0]) for b, v in old_dec])
        new_dec_at = _step_function([(b, v[0]) for b, v in new_dec])
        decision_flips = _diff_spans(old_dec_at, new_dec_at, breakpoints, hi)
        old_win_at = _step_function([(b, v[1]) for b, v in old_dec])
        new_win_at = _step_function([(b, v[1]) for b, v in new_dec])
        winner_flips = _diff_spans(old_win_at, new_win_at, breakpoints, hi)
    result["decision_flips"] = decision_flips
    result["winner_flips"] = winner_flips

    # Imported lazily and absolutely: this module is also loaded standalone
    # by the cycle-67 mutation-gate harness (a bare copy in a temp dir with
    # no parent package), where a module-level relative import fails. The
    # installed package is always importable there, so the absolute import
    # keeps a single source of truth for the trajectory logic.
    from model_protocol_lab.reporting import margin_trajectory, safe_budget_floor

    old_traj = margin_trajectory(old)
    new_traj = margin_trajectory(new)
    leader_flips: list[list[Any]] = []
    if old_traj and new_traj:
        old_lead_at = _step_function([(b, leader) for b, leader, _ in old_traj])
        new_lead_at = _step_function([(b, leader) for b, leader, _ in new_traj])
        leader_flips = _diff_spans(
            old_lead_at,
            new_lead_at,
            [b for b, _, _ in old_traj] + [b for b, _, _ in new_traj],
            hi,
        )
    result["leader_flips"] = leader_flips
    old_floor = safe_budget_floor(old)
    new_floor = safe_budget_floor(new)
    floor_changed = old_floor != new_floor
    result["safe_budget_floor"] = {
        "old": old_floor,
        "new": new_floor,
        "changed": floor_changed,
    }
    old_attr = _normalize_attribution(old.get("flip_attribution"))
    new_attr = _normalize_attribution(new.get("flip_attribution"))
    attr_changed = old_attr != new_attr
    result["flip_attribution"] = {
        "old": old_attr,
        "new": new_attr,
        "changed": attr_changed,
    }
    result["changed"] = bool(
        result["max_new_tokens"]["changed"]
        or result["budget_robust"]["changed"]
        or decision_flips
        or winner_flips
        or leader_flips
        or floor_changed
        or attr_changed
    )
    return result


def _valid_held_out_floor(value: Any) -> bool:
    """Light validation: is this a comparable ``held_out_budget_floor`` payload?"""
    if type(value) is not dict or value.get("available", True) is False:
        return False
    if value.get("native_verdict") not in (
        "compatible_on_fixture_set",
        "held_out_failed",
    ):
        return False
    max_tokens = value.get("max_new_tokens")
    if type(max_tokens) is not int or max_tokens <= 0:
        return False
    # Shape before consumption (RR-02): the summary renderer maps
    # budget_robust through a {True/False/None} lookup, so a list
    # raised TypeError after a valid diff had already been produced.
    robust = value.get("budget_robust")
    if robust is not None and type(robust) is not bool:
        return False
    floor = value.get("safe_budget_floor")
    return not (floor is not None and type(floor) is not int)


def _valid_boundary_audit(value: Any) -> bool:
    """Light validation: is this a comparable ``boundary_audit`` payload?"""
    if type(value) is not dict or value.get("available", True) is False:
        return False
    per_family = value.get("per_family")
    if type(per_family) is not dict:
        return False
    # Entry values are gated too (wave-8 F1): the differ reads
    # entry.get("rate"), and the `or {}` idiom there rescues only
    # falsy entries, so a truthy non-dict entry raised AttributeError.
    # The sibling validators gate their per-X values the same way
    # (_valid_perturbation, _valid_transfer's matrix).
    if not all(type(entry) is dict for entry in per_family.values()):
        return False
    rate = value.get("overall_rate")
    return rate is None or type(rate) in (int, float)


def diff_boundary_audit(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``boundary_audit`` payloads; deterministic.

    Answers "did the prompt-boundary canonicality story change between
    runs?": the overall PBP rate and the per-family rates. A side without
    a recorded audit is reported plainly instead of compared. No model,
    no network.
    """
    old_ok = _valid_boundary_audit(old)
    new_ok = _valid_boundary_audit(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    result: dict[str, Any] = {"available": available, "changed": False}
    if available != "both":
        return result
    old_rate = old.get("overall_rate")
    new_rate = new.get("overall_rate")
    old_families = old.get("per_family") or {}
    new_families = new.get("per_family") or {}
    family_drift: list[dict[str, Any]] = []
    for family in sorted(set(old_families) | set(new_families)):
        old_entry = old_families.get(family) or {}
        new_entry = new_families.get(family) or {}
        if old_entry.get("rate") != new_entry.get("rate"):
            family_drift.append(
                {
                    "family": family,
                    "old_rate": old_entry.get("rate"),
                    "new_rate": new_entry.get("rate"),
                }
            )
    result["overall_rate"] = {
        "old": old_rate,
        "new": new_rate,
        "changed": old_rate != new_rate,
    }
    result["family_drift"] = family_drift
    result["changed"] = bool(old_rate != new_rate or family_drift)
    return result


def _valid_selection_robustness(payload: Any) -> bool:
    """Structural check for a ``selection_robustness`` payload."""
    per_semantic = payload.get("per_semantic") if type(payload) is dict else None
    return (
        type(payload) is dict
        and payload.get("available") is True
        and type(per_semantic) is dict
        and all(type(k) is str for k in per_semantic)
        # Entry values gated (wave-8 F2): the differ reads
        # entry.get("decision") through the same `or {}` idiom as
        # boundary_audit -- a truthy non-dict entry raised
        # AttributeError there.
        and all(type(v) is dict for v in per_semantic.values())
        and type(payload.get("flips")) is list
    )


def _valid_contamination(payload: Any) -> bool:
    """Structural check for a ``contamination`` payload."""
    pairs = payload.get("pairs") if type(payload) is dict else None
    return (
        type(payload) is dict
        and type(payload.get("worst_tier")) is str
        and type(pairs) is list
        and all(type(p) is dict for p in pairs)
    )


def _contamination_pair_key(pair: dict[str, Any]) -> tuple[str, str] | None:
    """Identity key for a contamination pair, or None if malformed."""
    held = pair.get("held_out")
    search = pair.get("search")
    if type(held) is str and type(search) is str:
        return (held, search)
    return None


def diff_contamination(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``contamination`` payloads; deterministic.

    Answers "did fixture hygiene improve or degrade between runs?":
    the worst tier, and the set of flagged (held_out, search) pairs.
    Pairs are matched by ID tuple; added/removed pairs and tier
    escalations are reported. A side without a recorded contamination
    report is reported plainly. Malformed pairs are ignored (the
    report is advisory). No model, no network.
    """
    old_ok = _valid_contamination(old)
    new_ok = _valid_contamination(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    result: dict[str, Any] = {"available": available, "changed": False}
    if available != "both":
        return result
    old_worst = old.get("worst_tier")
    new_worst = new.get("worst_tier")
    old_pairs = {}
    for p in old.get("pairs") or []:
        key = _contamination_pair_key(p)
        if key is not None:
            old_pairs[key] = p.get("tier")
    new_pairs = {}
    for p in new.get("pairs") or []:
        key = _contamination_pair_key(p)
        if key is not None:
            new_pairs[key] = p.get("tier")
    old_keys = set(old_pairs)
    new_keys = set(new_pairs)
    pairs_added = sorted(new_keys - old_keys)
    pairs_removed = sorted(old_keys - new_keys)
    tier_changes = []
    for key in sorted(old_keys & new_keys):
        if old_pairs[key] != new_pairs[key]:
            tier_changes.append(
                {
                    "held_out": key[0],
                    "search": key[1],
                    "old_tier": old_pairs[key],
                    "new_tier": new_pairs[key],
                }
            )
    worst_changed = old_worst != new_worst
    result["worst_tier"] = {
        "old": old_worst,
        "new": new_worst,
        "changed": worst_changed,
    }
    result["pairs_added"] = [
        {"held_out": k[0], "search": k[1], "tier": new_pairs[k]} for k in pairs_added
    ]
    result["pairs_removed"] = [
        {"held_out": k[0], "search": k[1], "tier": old_pairs[k]} for k in pairs_removed
    ]
    result["tier_changes"] = tier_changes
    result["changed"] = bool(worst_changed or pairs_added or pairs_removed or tier_changes)
    return result


def diff_selection_robustness(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``selection_robustness`` payloads; deterministic.

    Answers "did the stop-semantics selection story change between runs?":
    the per-semantic (decision, winner) verdicts and the flip lists. A
    changed flip pattern between runs is the serving-stack migration
    signal: the same serving stack now selects a different winner than
    before. A side without a recorded analysis is reported plainly. No
    model, no network.
    """
    old_ok = _valid_selection_robustness(old)
    new_ok = _valid_selection_robustness(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    result: dict[str, Any] = {"available": available, "changed": False}
    if available != "both":
        return result
    old_ps = old.get("per_semantic") or {}
    new_ps = new.get("per_semantic") or {}
    verdict_drift: list[dict[str, Any]] = []
    for semantic in sorted(set(old_ps) | set(new_ps)):
        old_v = old_ps.get(semantic) or {}
        new_v = new_ps.get(semantic) or {}
        if (old_v.get("decision"), old_v.get("winner")) != (
            new_v.get("decision"),
            new_v.get("winner"),
        ):
            verdict_drift.append(
                {
                    "semantic": semantic,
                    "old_decision": old_v.get("decision"),
                    "old_winner": old_v.get("winner"),
                    "new_decision": new_v.get("decision"),
                    "new_winner": new_v.get("winner"),
                }
            )
    old_flips = old.get("flips") or []
    new_flips = new.get("flips") or []
    flips_changed = old_flips != new_flips
    result["verdict_drift"] = verdict_drift
    result["flips_changed"] = flips_changed
    result["changed"] = bool(verdict_drift or flips_changed)
    return result


def _valid_transfer(payload: Any) -> bool:
    """Structural check for a ``transfer`` payload."""
    per_checkpoint = payload.get("per_checkpoint") if type(payload) is dict else None
    matrix = payload.get("matrix") if type(payload) is dict else None
    return (
        type(payload) is dict
        and payload.get("available") is True
        and type(per_checkpoint) is dict
        and all(type(k) is str for k in per_checkpoint)
        # per_checkpoint values gated exactly like the matrix values
        # below (wave-8 F3): the differ reads entry.get("decision")
        # through the `or {}` idiom, which rescues only falsy
        # entries -- a truthy non-dict entry raised AttributeError.
        and all(type(v) is dict for v in per_checkpoint.values())
        and type(matrix) is dict
        and all(type(k) is str for k in matrix)
        and all(type(v) is dict and all(type(k) is str for k in v) for v in matrix.values())
    )


def diff_transfer(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``transfer`` payloads; deterministic.

    Answers "did the cross-checkpoint transfer story change between
    runs?": the transfer verdict, the per-checkpoint (decision, winner)
    verdicts, and the per-cell candidate x checkpoint verdicts. A changed
    transfer pattern after a checkpoint or fixture update is the
    fleet-portability regression signal. Checkpoints present on only one
    side are reported as added/removed. A side without a recorded
    transfer experiment is reported plainly. No model, no network.
    """
    old_ok = _valid_transfer(old)
    new_ok = _valid_transfer(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    result: dict[str, Any] = {"available": available, "changed": False}
    if available != "both":
        return result
    old_verdict = old.get("transfer_verdict")
    new_verdict = new.get("transfer_verdict")
    old_pc = old.get("per_checkpoint") or {}
    new_pc = new.get("per_checkpoint") or {}
    old_matrix = old.get("matrix") or {}
    new_matrix = new.get("matrix") or {}
    old_ckpts = sorted(old_pc)
    new_ckpts = sorted(new_pc)
    checkpoints_added = [c for c in new_ckpts if c not in old_pc]
    checkpoints_removed = [c for c in old_ckpts if c not in new_pc]
    checkpoint_drift: list[dict[str, Any]] = []
    for ckpt in sorted(set(old_ckpts) & set(new_ckpts)):
        old_v = old_pc.get(ckpt) or {}
        new_v = new_pc.get(ckpt) or {}
        if (old_v.get("decision"), old_v.get("winner")) != (
            new_v.get("decision"),
            new_v.get("winner"),
        ):
            checkpoint_drift.append(
                {
                    "checkpoint": ckpt,
                    "old_decision": old_v.get("decision"),
                    "old_winner": old_v.get("winner"),
                    "new_decision": new_v.get("decision"),
                    "new_winner": new_v.get("winner"),
                }
            )
    cell_drift: list[dict[str, Any]] = []
    for cand in sorted(set(old_matrix) | set(new_matrix)):
        old_row = old_matrix.get(cand) or {}
        new_row = new_matrix.get(cand) or {}
        for ckpt in sorted(set(old_row) | set(new_row)):
            if old_row.get(ckpt) != new_row.get(ckpt):
                cell_drift.append(
                    {
                        "candidate": cand,
                        "checkpoint": ckpt,
                        "old": old_row.get(ckpt),
                        "new": new_row.get(ckpt),
                    }
                )
    verdict_changed = old_verdict != new_verdict
    result["transfer_verdict"] = {
        "old": old_verdict,
        "new": new_verdict,
        "changed": verdict_changed,
    }
    result["checkpoint_drift"] = checkpoint_drift
    result["cell_drift"] = cell_drift
    result["checkpoints_added"] = checkpoints_added
    result["checkpoints_removed"] = checkpoints_removed
    result["changed"] = bool(
        verdict_changed
        or checkpoint_drift
        or cell_drift
        or checkpoints_added
        or checkpoints_removed
    )
    return result


def _valid_perturbation(payload: Any) -> bool:
    """Structural check for a ``perturbation`` payload."""
    per_variant = payload.get("per_variant") if type(payload) is dict else None
    stability = payload.get("stability") if type(payload) is dict else None
    return (
        type(payload) is dict
        and payload.get("available") is True
        and type(per_variant) is dict
        and all(type(k) is str for k in per_variant)
        and all(type(v) is dict for v in per_variant.values())
        and type(stability) is dict
    )


def diff_perturbation(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``perturbation`` payloads; deterministic.

    Answers "did the fixture-perturbation robustness story change
    between runs?": the stability triplet (variants retaining the
    primary winner / decision), the per-variant (decision, winner)
    verdicts, and the variant set. A changed stability pattern after
    a fixture or checkpoint update is the phrasing-fragility
    regression signal. Variants present on only one side are reported
    as added/removed; differing operator sets are flagged. A side
    without a recorded perturbation experiment is reported plainly.
    No model, no network.
    """
    old_ok = _valid_perturbation(old)
    new_ok = _valid_perturbation(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    result: dict[str, Any] = {"available": available, "changed": False}
    if available != "both":
        return result
    old_stability = _stability_triplet(old.get("stability"))
    new_stability = _stability_triplet(new.get("stability"))
    stability_changed = old_stability != new_stability
    result["stability"] = {
        "old": {
            "winner_retained": old_stability[0],
            "decision_retained": old_stability[1],
            "n_evaluable": old_stability[2],
        },
        "new": {
            "winner_retained": new_stability[0],
            "decision_retained": new_stability[1],
            "n_evaluable": new_stability[2],
        },
        "changed": stability_changed,
    }
    old_pv = old.get("per_variant") or {}
    new_pv = new.get("per_variant") or {}
    old_ids = sorted(old_pv)
    new_ids = sorted(new_pv)
    variants_added = [v for v in new_ids if v not in old_pv]
    variants_removed = [v for v in old_ids if v not in new_pv]
    variant_drift: list[dict[str, Any]] = []
    for vid in sorted(set(old_ids) & set(new_ids)):
        old_v = old_pv.get(vid) or {}
        new_v = new_pv.get(vid) or {}
        if (old_v.get("decision"), old_v.get("winner")) != (
            new_v.get("decision"),
            new_v.get("winner"),
        ):
            variant_drift.append(
                {
                    "variant": vid,
                    "operator": new_v.get("operator"),
                    "old_decision": old_v.get("decision"),
                    "old_winner": old_v.get("winner"),
                    "new_decision": new_v.get("decision"),
                    "new_winner": new_v.get("winner"),
                }
            )
    old_operators = old.get("operators")
    new_operators = new.get("operators")
    operators_changed = old_operators != new_operators
    result["variant_drift"] = variant_drift
    result["variants_added"] = variants_added
    result["variants_removed"] = variants_removed
    result["operators_changed"] = operators_changed
    result["changed"] = bool(
        stability_changed
        or variant_drift
        or variants_added
        or variants_removed
        or operators_changed
    )
    return result


def _valid_token_healing(payload: Any) -> bool:
    """Structural check for a ``token_healing`` payload.

    Also enforces the wave-2 C3 lesson at the value level: the compared
    fields must be bools/strings/None so the renderers never hit a
    TypeError on a hostile payload.
    """
    if type(payload) is not dict or payload.get("available") is not True:
        return False
    if type(payload.get("flipped")) is not bool:
        return False
    falsification = payload.get("falsification")
    if falsification is not None and type(falsification) is not dict:
        return False
    per_candidate = payload.get("per_candidate")
    if type(per_candidate) is not dict or not all(
        type(k) is str and type(v) is dict for k, v in per_candidate.items()
    ):
        return False
    for entry in per_candidate.values():
        for key in ("native", "healed"):
            value = entry.get(key)
            if value is not None and type(value) is not bool:
                return False
    return True


_STOPCAUSE_CAUSES = ("eos_id", "stop_string", "max_length", "stop_unknown")


def _valid_stop_cause(payload: Any) -> bool:
    """Structural check for a ``stop_cause`` payload.

    Enforces the wave-2 C3 lesson at the value level: the compared
    fields must be bools/strings/None/numbers so the renderers never
    hit a TypeError on a hostile payload, and map keys must be
    strings. A malformed side degrades to asymmetric instead of
    raising.
    """
    if type(payload) is not dict or payload.get("available") is not True:
        return False
    per_candidate = payload.get("per_candidate")
    if type(per_candidate) is not dict or not all(type(k) is str for k in per_candidate):
        return False
    for entry in per_candidate.values():
        if type(entry) is not dict:
            return False
        rates = entry.get("cause_rates")
        if type(rates) is not dict:
            return False
        for cause, rate in rates.items():
            if cause not in _STOPCAUSE_CAUSES:
                return False
            if type(rate) not in (int, float):
                return False
        for key in ("over_generation_rate", "would_stop_early_rate", "max_length_rate"):
            value = entry.get(key)
            if value is not None and type(value) not in (int, float):
                return False
    overall = payload.get("overall")
    return not (overall is not None and type(overall) is not dict)


def diff_stop_cause(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``stop_cause`` payloads; deterministic.

    Answers "did the stop-cause termination story change between
    runs?": per-candidate termination-cause rate drift (eos_id,
    stop_string, max_length, stop_unknown), over-generation-rate and
    would-stop-early-rate drift, and added/removed candidates. A newly
    eos-emitting candidate after a checkpoint or fixture update is the
    termination-behavior regression signal. A side without a recorded
    stop-cause audit is reported plainly. No model, no network.
    """
    old_ok = _valid_stop_cause(old)
    new_ok = _valid_stop_cause(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    result: dict[str, Any] = {"available": available, "changed": False}
    if available != "both":
        return result

    def _candidate_view(payload: Any) -> dict[str, dict[str, Any]]:
        """Return the candidate view."""
        view: dict[str, dict[str, Any]] = {}
        for name, entry in (payload.get("per_candidate") or {}).items():
            entry = entry or {}
            view[name] = {
                "cause_rates": {
                    cause: rate
                    for cause, rate in (entry.get("cause_rates") or {}).items()
                    if type(cause) is str
                },
                "over_generation_rate": entry.get("over_generation_rate"),
                "would_stop_early_rate": entry.get("would_stop_early_rate"),
                "max_length_rate": entry.get("max_length_rate"),
            }
        return view

    old_view = _candidate_view(old)
    new_view = _candidate_view(new)
    candidates_added = sorted(set(new_view) - set(old_view))
    candidates_removed = sorted(set(old_view) - set(new_view))
    cause_drift: list[dict[str, Any]] = []
    aggregate_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_view) & set(new_view)):
        old_rates = old_view[name]["cause_rates"]
        new_rates = new_view[name]["cause_rates"]
        for cause in sorted(set(old_rates) | set(new_rates)):
            if old_rates.get(cause) != new_rates.get(cause):
                cause_drift.append(
                    {
                        "candidate": name,
                        "cause": cause,
                        "old_rate": old_rates.get(cause),
                        "new_rate": new_rates.get(cause),
                    }
                )
        for metric in ("over_generation_rate", "would_stop_early_rate", "max_length_rate"):
            if old_view[name][metric] != new_view[name][metric]:
                aggregate_drift.append(
                    {
                        "candidate": name,
                        "metric": metric,
                        "old": old_view[name][metric],
                        "new": new_view[name][metric],
                    }
                )
    result["cause_drift"] = cause_drift
    result["aggregate_drift"] = aggregate_drift
    result["candidates_added"] = candidates_added
    result["candidates_removed"] = candidates_removed
    # NOTE: stop-cause-specific disjunct names keep this line textually
    # distinct from every other diff section's ``changed`` anchor
    # (mutation-gate uniqueness).
    result["changed"] = bool(
        cause_drift or aggregate_drift or candidates_added or candidates_removed
    )
    return result


def _valid_deliforge(payload: Any) -> bool:
    """Structural check for a ``deliforge`` delimiter-authority payload.

    Enforces the wave-2 C3 lesson at the value level: flip rates must
    be numbers/None in [0, 1], authority gaps numbers/None in [-1, 1],
    and map keys strings, so the renderers never hit a TypeError on a
    hostile payload.
    """
    if type(payload) is not dict or payload.get("available") is not True:
        return False
    payloads = payload.get("payloads")
    if type(payloads) is not list or not payloads or not all(type(p) is str for p in payloads):
        return False
    marker_indexes = payload.get("marker_indexes")
    if (
        type(marker_indexes) is not list
        or not marker_indexes
        or not all(type(i) is int for i in marker_indexes)
    ):
        return False
    per_candidate = payload.get("per_candidate")
    if type(per_candidate) is not dict or not all(type(k) is str for k in per_candidate):
        return False
    for entry in per_candidate.values():
        if type(entry) is not dict:
            return False
        cells = entry.get("cells")
        if type(cells) is not dict or not all(type(k) is str for k in cells):
            return False
        for cell in cells.values():
            if type(cell) is not dict or cell.get("supported") is not True:
                continue
            n_evaluable = cell.get("n_evaluable")
            if type(n_evaluable) is not int or n_evaluable < 0:
                return False
            for key in ("flip_rate_reserved", "flip_rate_subword"):
                rate = cell.get(key)
                if rate is not None and not (type(rate) in (int, float) and 0.0 <= rate <= 1.0):
                    return False
            gap = cell.get("authority_gap")
            if gap is not None and not (type(gap) in (int, float) and -1.0 <= gap <= 1.0):
                return False
        top = entry.get("max_authority_gap")
        if top is not None and not (type(top) in (int, float) and -1.0 <= top <= 1.0):
            return False
    excluded = payload.get("excluded_candidates")
    return type(excluded) is list and all(type(c) is str for c in excluded)


def diff_deliforge(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``deliforge`` delimiter-authority payloads; deterministic.

    Answers "did the delimiter-authority story change between runs?":
    per-candidate per-cell authority-gap drift, top-gap drift,
    added/removed candidates, and changes in the excluded set. A newly
    positive reserved-minus-subword gap after a template change is the
    regression signal -- the reserved encoding gaining verdict-flip
    authority it did not have before. A side without a recorded probe
    is reported plainly. No model, no network.
    """
    old_ok = _valid_deliforge(old)
    new_ok = _valid_deliforge(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    result: dict[str, Any] = {"available": available, "changed": False}
    if available != "both":
        return result

    def _candidate_view(payload: Any) -> dict[str, dict[str, Any]]:
        """Return the candidate view."""
        view: dict[str, dict[str, Any]] = {}
        for name, entry in (payload.get("per_candidate") or {}).items():
            # _valid_deliforge guarantees str keys and dict entries, so
            # no defensive skip is needed here (it would be dead code
            # and an uncoverable line).
            cells: dict[str, Any] = {}
            for key, cell in (entry.get("cells") or {}).items():
                cells[key] = (
                    cell.get("authority_gap")
                    if type(cell) is dict and cell.get("supported") is True
                    else None
                )
            view[name] = {
                "cells": cells,
                "max_authority_gap": entry.get("max_authority_gap"),
                "excluded": name in set(payload.get("excluded_candidates") or []),
            }
        return view

    old_view = _candidate_view(old)
    new_view = _candidate_view(new)
    deliforge_candidates_added = sorted(set(new_view) - set(old_view))
    deliforge_candidates_removed = sorted(set(old_view) - set(new_view))
    deliforge_gap_drift: list[dict[str, Any]] = []
    deliforge_top_gap_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_view) & set(new_view)):
        old_cells = old_view[name]["cells"]
        new_cells = new_view[name]["cells"]
        for key in sorted(set(old_cells) & set(new_cells)):
            old_gap = old_cells[key]
            new_gap = new_cells[key]
            if old_gap != new_gap:
                deliforge_gap_drift.append(
                    {"candidate": name, "cell": key, "old": old_gap, "new": new_gap}
                )
        old_top = old_view[name]["max_authority_gap"]
        new_top = new_view[name]["max_authority_gap"]
        if old_top != new_top:
            deliforge_top_gap_drift.append({"candidate": name, "old": old_top, "new": new_top})
    old_excluded = {name for name, v in old_view.items() if v["excluded"]}
    new_excluded = {name for name, v in new_view.items() if v["excluded"]}
    deliforge_excluded_added = sorted(new_excluded - old_excluded)
    deliforge_excluded_removed = sorted(old_excluded - new_excluded)
    result["gap_drift"] = deliforge_gap_drift
    result["top_gap_drift"] = deliforge_top_gap_drift
    result["candidates_added"] = deliforge_candidates_added
    result["candidates_removed"] = deliforge_candidates_removed
    result["excluded_added"] = deliforge_excluded_added
    result["excluded_removed"] = deliforge_excluded_removed
    # NOTE: deliforge-specific disjunct names keep this ``changed``
    # line textually distinct from every other diff section's anchor
    # (mutation-gate uniqueness).
    result["changed"] = bool(
        deliforge_gap_drift
        or deliforge_top_gap_drift
        or deliforge_candidates_added
        or deliforge_candidates_removed
        or deliforge_excluded_added
        or deliforge_excluded_removed
    )
    return result


def _valid_fewshot(payload: Any) -> bool:
    """Structural check for a ``fewshot`` few-shot x template payload.

    Enforces the wave-2 C3 lesson at the value level: gain must be
    numbers/None in [0, 1], order sensitivity numbers/None in [0, 0.5],
    per-order change rates numbers/None in [0, 1], arm decisions and
    winners strings/None, and map keys strings, so the renderers never
    hit a TypeError on a hostile payload.
    """
    if type(payload) is not dict or payload.get("available") is not True:
        return False
    demo_ids = payload.get("demo_ids")
    if type(demo_ids) is not list or not demo_ids or not all(type(d) is str for d in demo_ids):
        return False
    if type(payload.get("k")) is not int or type(payload.get("n_eval")) is not int:
        return False
    per_candidate = payload.get("per_candidate")
    if type(per_candidate) is not dict or not all(type(k) is str for k in per_candidate):
        return False
    for entry in per_candidate.values():
        if type(entry) is not dict:
            return False
        gain = entry.get("gain")
        if gain is not None and not (type(gain) in (int, float) and 0.0 <= gain <= 1.0):
            return False
        sensitivity = entry.get("order_sensitivity")
        if sensitivity is not None and not (
            type(sensitivity) in (int, float) and 0.0 <= sensitivity <= 0.5
        ):
            return False
        per_order = entry.get("per_order")
        if type(per_order) is not dict:
            return False
        for order in per_order.values():
            if type(order) is not dict:
                return False
            rate = order.get("change_rate")
            if rate is not None and not (type(rate) in (int, float) and 0.0 <= rate <= 1.0):
                return False
            for key in ("n_changed", "n_evaluable"):
                count = order.get(key)
                if type(count) is not int or count < 0:
                    return False
    per_arm = payload.get("per_arm")
    if type(per_arm) is not dict:
        return False
    for arm in per_arm.values():
        if type(arm) is not dict or type(arm.get("decision")) is not str:
            return False
        winner = arm.get("winner")
        if winner is not None and type(winner) is not str:
            return False
    flips = payload.get("flips")
    if type(flips) is not list or any(type(f) is not dict for f in flips):
        return False
    excluded = payload.get("excluded_candidates")
    return type(excluded) is list and all(type(c) is str for c in excluded)


def diff_fewshot(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``fewshot`` few-shot x template payloads; deterministic.

    Answers "did the few-shot story change between runs?": per-candidate
    gain drift and order-sensitivity drift, per-arm (decision, winner)
    drift, added/removed candidates, and changes in the excluded set. A
    newly high gain after a template change is the regression signal --
    demos now move verdicts they did not move before. A side without a
    recorded probe is reported plainly. No model, no network.
    """
    old_ok = _valid_fewshot(old)
    new_ok = _valid_fewshot(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    result: dict[str, Any] = {"available": available, "changed": False}
    if available != "both":
        return result

    def _candidate_view(payload: Any) -> dict[str, dict[str, Any]]:
        """Return the candidate view."""
        view: dict[str, dict[str, Any]] = {}
        for name, entry in (payload.get("per_candidate") or {}).items():
            # _valid_fewshot guarantees str keys and dict entries, so
            # no defensive skip is needed here (it would be dead code
            # and an uncoverable line).
            view[name] = {
                "gain": entry.get("gain"),
                "order_sensitivity": entry.get("order_sensitivity"),
                "excluded": name in set(payload.get("excluded_candidates") or []),
            }
        return view

    def _arm_view(payload: Any) -> dict[str, tuple[Any, Any]]:
        """Return the arm view."""
        return {
            arm: (cell.get("decision"), cell.get("winner"))
            for arm, cell in (payload.get("per_arm") or {}).items()
            if type(cell) is dict
        }

    old_view = _candidate_view(old)
    new_view = _candidate_view(new)
    fewshot_candidates_added = sorted(set(new_view) - set(old_view))
    fewshot_candidates_removed = sorted(set(old_view) - set(new_view))
    fewshot_gain_drift: list[dict[str, Any]] = []
    fewshot_sensitivity_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_view) & set(new_view)):
        old_gain = old_view[name]["gain"]
        new_gain = new_view[name]["gain"]
        if old_gain != new_gain:
            fewshot_gain_drift.append({"candidate": name, "old": old_gain, "new": new_gain})
        old_sens = old_view[name]["order_sensitivity"]
        new_sens = new_view[name]["order_sensitivity"]
        if old_sens != new_sens:
            fewshot_sensitivity_drift.append({"candidate": name, "old": old_sens, "new": new_sens})
    old_excluded = {name for name, v in old_view.items() if v["excluded"]}
    new_excluded = {name for name, v in new_view.items() if v["excluded"]}
    fewshot_excluded_added = sorted(new_excluded - old_excluded)
    fewshot_excluded_removed = sorted(old_excluded - new_excluded)
    old_arms = _arm_view(old)
    new_arms = _arm_view(new)
    fewshot_arm_drift: list[dict[str, Any]] = []
    for arm in sorted(set(old_arms) & set(new_arms)):
        if old_arms[arm] != new_arms[arm]:
            fewshot_arm_drift.append(
                {
                    "arm": arm,
                    "old_decision": old_arms[arm][0],
                    "new_decision": new_arms[arm][0],
                    "old_winner": old_arms[arm][1],
                    "new_winner": new_arms[arm][1],
                }
            )
    result["gain_drift"] = fewshot_gain_drift
    result["sensitivity_drift"] = fewshot_sensitivity_drift
    result["arm_drift"] = fewshot_arm_drift
    result["candidates_added"] = fewshot_candidates_added
    result["candidates_removed"] = fewshot_candidates_removed
    result["excluded_added"] = fewshot_excluded_added
    result["excluded_removed"] = fewshot_excluded_removed
    # NOTE: fewshot-specific disjunct names keep this ``changed``
    # line textually distinct from every other diff section's anchor
    # (mutation-gate uniqueness).
    result["changed"] = bool(
        fewshot_gain_drift
        or fewshot_sensitivity_drift
        or fewshot_arm_drift
        or fewshot_candidates_added
        or fewshot_candidates_removed
        or fewshot_excluded_added
        or fewshot_excluded_removed
    )
    return result


def _valid_verbosity(payload: Any) -> bool:
    """Structural check for a ``verbosity`` profile payload.

    Enforces the wave-2 C3 lesson at the value level: excess
    statistics must be numbers/None and non-negative, the capped
    share a number/None in [0, 1], quartiles ordered, and map keys
    strings, so the renderers never hit a TypeError on a hostile
    payload.
    """
    if type(payload) is not dict or payload.get("available") is not True:
        return False
    n_fixtures = payload.get("n_fixtures")
    if type(n_fixtures) is not int or n_fixtures < 0:
        return False
    n_brevity = payload.get("n_brevity_fixtures")
    if type(n_brevity) is not int or n_brevity < 0 or n_brevity > n_fixtures:
        return False
    per_candidate = payload.get("per_candidate")
    if type(per_candidate) is not dict or not all(type(k) is str for k in per_candidate):
        return False
    for entry in per_candidate.values():
        if entry is None:
            continue
        if type(entry) is not dict:
            return False
        for key in ("n", "n_evaluable"):
            value = entry.get(key)
            if type(value) is not int or value < 0:
                return False
        if entry["n_evaluable"] > entry["n"]:
            return False
        for key in (
            "median_excess",
            "p25_excess",
            "p75_excess",
            "iqr_excess",
            "mean_excess",
        ):
            value = entry.get(key)
            if value is not None and not (type(value) in (int, float) and value >= 0):
                return False
        max_excess = entry.get("max_excess")
        if max_excess is not None and (type(max_excess) is not int or max_excess < 0):
            return False
        capped_share = entry.get("capped_share")
        if capped_share is not None and not (
            type(capped_share) in (int, float) and 0.0 <= capped_share <= 1.0
        ):
            return False
        p25 = entry.get("p25_excess")
        median = entry.get("median_excess")
        p75 = entry.get("p75_excess")
        # isinstance narrows for mypy where type() in (...) does not;
        # bools were already rejected by the per-key loop above.
        if (
            entry["n_evaluable"] > 0
            and isinstance(p25, (int, float))
            and isinstance(median, (int, float))
            and isinstance(p75, (int, float))
            and not p25 <= median <= p75
        ):
            return False
    for key in ("most_verbose", "least_verbose", "capped_candidates", "excluded_candidates"):
        value = payload.get(key)
        if type(value) is not list or not all(type(c) is str for c in value):
            return False
    return True


def diff_verbosity(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``verbosity`` profile payloads; deterministic.

    Answers "did the verbosity story change between runs?":
    per-candidate median-excess drift, IQR drift, and
    budget-saturation-share drift, plus added/removed candidates and
    changes in the excluded set. A newly positive median-excess gap
    after a template change is the regression signal -- the template
    deciding how much the model generates. A side without a recorded
    probe is reported plainly. No model, no network.
    """
    old_ok = _valid_verbosity(old)
    new_ok = _valid_verbosity(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    result: dict[str, Any] = {"available": available, "changed": False}
    if available != "both":
        return result

    def _candidate_view(payload: Any) -> dict[str, dict[str, Any]]:
        """Return the candidate view."""
        view: dict[str, dict[str, Any]] = {}
        for name, entry in (payload.get("per_candidate") or {}).items():
            # _valid_verbosity guarantees str keys and dict/None
            # entries, so no defensive skip is needed here (it would
            # be dead code and an uncoverable line).
            if entry is None:
                view[name] = {
                    "median_excess": None,
                    "iqr_excess": None,
                    "capped_share": None,
                    "excluded": True,
                }
            else:
                view[name] = {
                    "median_excess": entry.get("median_excess"),
                    "iqr_excess": entry.get("iqr_excess"),
                    "capped_share": entry.get("capped_share"),
                    "excluded": name in set(payload.get("excluded_candidates") or []),
                }
        return view

    old_view = _candidate_view(old)
    new_view = _candidate_view(new)
    verbosity_candidates_added = sorted(set(new_view) - set(old_view))
    verbosity_candidates_removed = sorted(set(old_view) - set(new_view))
    verbosity_median_drift: list[dict[str, Any]] = []
    verbosity_iqr_drift: list[dict[str, Any]] = []
    verbosity_capped_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_view) & set(new_view)):
        old_entry = old_view[name]
        new_entry = new_view[name]
        for rate_key, bucket in (
            ("median_excess", verbosity_median_drift),
            ("iqr_excess", verbosity_iqr_drift),
            ("capped_share", verbosity_capped_drift),
        ):
            if old_entry[rate_key] != new_entry[rate_key]:
                bucket.append(
                    {
                        "candidate": name,
                        "old": old_entry[rate_key],
                        "new": new_entry[rate_key],
                    }
                )
    old_excluded = {name for name, v in old_view.items() if v["excluded"]}
    new_excluded = {name for name, v in new_view.items() if v["excluded"]}
    verbosity_excluded_added = sorted(new_excluded - old_excluded)
    verbosity_excluded_removed = sorted(old_excluded - new_excluded)
    result["median_drift"] = verbosity_median_drift
    result["iqr_drift"] = verbosity_iqr_drift
    result["capped_share_drift"] = verbosity_capped_drift
    result["candidates_added"] = verbosity_candidates_added
    result["candidates_removed"] = verbosity_candidates_removed
    result["excluded_added"] = verbosity_excluded_added
    result["excluded_removed"] = verbosity_excluded_removed
    # NOTE: verbosity-specific disjunct names keep this ``changed``
    # line textually distinct from every other diff section's anchor
    # (mutation-gate uniqueness).
    result["changed"] = bool(
        verbosity_median_drift
        or verbosity_iqr_drift
        or verbosity_capped_drift
        or verbosity_candidates_added
        or verbosity_candidates_removed
        or verbosity_excluded_added
        or verbosity_excluded_removed
    )
    return result


def _valid_refusals(payload: Any) -> bool:
    """Structural check for a ``refusals`` profile payload.

    Mirrors the fail-closed shape of the other profile validators:
    the method block must name the detector and carry the opener-list
    digest, counts must be non-negative ints with their arithmetic
    invariants intact (suspected = passed + failed, hits length =
    suspected count, rate in [0, 1] when present), so the diff and
    the renderers never hit a TypeError on a hostile payload.
    """
    if type(payload) is not dict or payload.get("available") is not True:
        return False
    method = payload.get("method")
    if type(method) is not dict:
        return False
    if type(method.get("detector")) is not str or not method.get("detector"):
        return False
    if type(method.get("openers_sha256")) is not str or not method.get("openers_sha256"):
        return False
    # Internal consistency, version-agnostic (cycle 176, wave-7
    # F6a): the digest must be the digest of the payload's OWN
    # opener list -- a foreign-but-coherent detector still
    # validates (protocol drift is the diff's job to report), a
    # list/digest mismatch is corruption, not drift.
    openers = method.get("openers")
    if (
        type(openers) is not list
        or not openers
        or not all(type(opener) is str and opener for opener in openers)
    ):
        return False
    try:
        openers_digest = hashlib.sha256("\n".join(openers).encode("utf-8")).hexdigest()
    except UnicodeEncodeError:
        # A lone-surrogate opener passes the str check above but can
        # never be the recorded digest's input (the producer's list
        # is real text); fail closed instead of raising (oversight
        # edge-case F1, the cycle-190 discipline).
        return False
    if method.get("openers_sha256") != openers_digest:
        return False
    if type(payload.get("evaluable_mismatch")) is not bool:
        return False
    candidates = payload.get("candidates")
    if type(candidates) is not list:
        return False
    n_candidates = payload.get("n_candidates")
    if type(n_candidates) is not int or n_candidates != len(candidates):
        return False
    n_available = 0
    seen_names: set[str] = set()
    for entry in candidates:
        if type(entry) is not dict:
            return False
        name = entry.get("candidate")
        if name is not None and type(name) is not str:
            return False
        if type(name) is str:
            # Duplicate names would collapse last-wins in the
            # diff's candidate view and manufacture phantom drift;
            # the producer never emits them (cycle 176, F6a).
            if name in seen_names:
                return False
            seen_names.add(name)
        if type(entry.get("available")) is not bool:
            return False
        if entry.get("available") is not True:
            if type(entry.get("detail")) is not str:
                return False
            continue
        n_available += 1
        counts: dict[str, int] = {}
        for key in (
            "n_observations",
            "n_completed",
            "n_boundary_found",
            "n_suspected",
            "n_suspected_passed",
            "n_suspected_failed",
        ):
            value = entry.get(key)
            if type(value) is not int or value < 0:
                return False
            counts[key] = value
        if counts["n_completed"] > counts["n_observations"]:
            return False
        if counts["n_boundary_found"] > counts["n_completed"]:
            return False
        if counts["n_suspected"] > counts["n_completed"]:
            return False
        if counts["n_suspected_passed"] + counts["n_suspected_failed"] != counts["n_suspected"]:
            return False
        rate = entry.get("suspected_rate")
        if rate is not None and not (type(rate) in (int, float) and 0.0 <= rate <= 1.0):
            return False
        if type(entry.get("rate_note")) is not str:
            return False
        hits = entry.get("hits")
        if type(hits) is not list or len(hits) != counts["n_suspected"]:
            return False
        for hit in hits:
            if type(hit) is not dict:
                return False
            fixture = hit.get("fixture")
            if fixture is not None and type(fixture) is not str:
                return False
            if type(hit.get("opener")) is not str or not hit.get("opener"):
                return False
            if hit.get("opener") not in openers:
                return False
            if type(hit.get("passed")) is not bool:
                return False
        # Derived-value recomputation (cycle 176, wave-7 F6a; the
        # answer-collision validator's standard): the rate is the
        # counts' ratio, the note vocabulary is the producer's, a
        # rate is present exactly when no note explains its absence,
        # and the hits' passed tally is the cross-tab's passed cell.
        note = entry.get("rate_note")
        if note not in ("", "candidate_not_eligible", "no_completed_observations"):
            return False
        if (rate is None) != (note != ""):
            return False
        if rate is not None and (
            counts["n_completed"] == 0 or rate != counts["n_suspected"] / counts["n_completed"]
        ):
            return False
        if note == "no_completed_observations" and counts["n_completed"] != 0:
            return False
        if sum(1 for hit in hits if hit.get("passed") is True) != counts["n_suspected_passed"]:
            return False
    n_candidates_available = payload.get("n_candidates_available")
    return type(n_candidates_available) is int and n_candidates_available == n_available


def diff_refusals(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``refusals`` profile payloads; deterministic.

    Answers "did the suspected-refusal story change between runs?":
    per-candidate rate drift with denominator drift alongside (a
    rate moved because its n moved is a different story from a rate
    moved on the same n), hit-set changes per candidate, candidates
    added/removed, candidates newly/no-longer excluded, an
    evaluable-mismatch change, and detector protocol drift (the
    frozen opener list or detector version differs -- rates across a
    detector change are not comparable, and the diff says so). A
    side without a recorded probe is reported plainly. No model, no
    network.
    """
    old_ok = _valid_refusals(old)
    new_ok = _valid_refusals(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    result: dict[str, Any] = {"available": available, "changed": False}
    if available != "both":
        return result
    protocol_drift = (old["method"].get("detector"), old["method"].get("openers_sha256")) != (
        new["method"].get("detector"),
        new["method"].get("openers_sha256"),
    )

    def _candidate_view(payload: Any) -> dict[str, dict[str, Any]]:
        """Return the candidate view."""
        view: dict[str, dict[str, Any]] = {}
        for entry in payload.get("candidates") or []:
            # _valid_refusals guarantees the entry shape, so no
            # defensive skip is needed here (it would be dead code
            # and an uncoverable line).
            name = entry.get("candidate")
            if type(name) is not str:
                continue
            if entry.get("available") is not True:
                view[name] = {"excluded": True}
                continue
            view[name] = {
                "excluded": False,
                "rate": entry.get("suspected_rate"),
                "n": entry.get("n_completed"),
                "hits": {
                    (hit.get("fixture"), hit.get("opener"), hit.get("passed"))
                    for hit in entry.get("hits") or []
                },
            }
        return view

    old_view = _candidate_view(old)
    new_view = _candidate_view(new)
    refusals_candidates_added = sorted(set(new_view) - set(old_view))
    refusals_candidates_removed = sorted(set(old_view) - set(new_view))
    refusals_rate_drift: list[dict[str, Any]] = []
    refusals_hits_added: list[dict[str, Any]] = []
    refusals_hits_removed: list[dict[str, Any]] = []
    for name in sorted(set(old_view) & set(new_view)):
        old_entry = old_view[name]
        new_entry = new_view[name]
        if old_entry["excluded"] or new_entry["excluded"]:
            continue
        if (old_entry["rate"], old_entry["n"]) != (new_entry["rate"], new_entry["n"]):
            refusals_rate_drift.append(
                {
                    "candidate": name,
                    "old_rate": old_entry["rate"],
                    "new_rate": new_entry["rate"],
                    "old_n": old_entry["n"],
                    "new_n": new_entry["n"],
                }
            )
        for fixture, opener, passed in sorted(
            new_entry["hits"] - old_entry["hits"], key=lambda h: (str(h[0]), str(h[1]), str(h[2]))
        ):
            refusals_hits_added.append(
                {"candidate": name, "fixture": fixture, "opener": opener, "passed": passed}
            )
        for fixture, opener, passed in sorted(
            old_entry["hits"] - new_entry["hits"], key=lambda h: (str(h[0]), str(h[1]), str(h[2]))
        ):
            refusals_hits_removed.append(
                {"candidate": name, "fixture": fixture, "opener": opener, "passed": passed}
            )
    old_excluded = {name for name, v in old_view.items() if v["excluded"]}
    new_excluded = {name for name, v in new_view.items() if v["excluded"]}
    refusals_excluded_added = sorted(new_excluded - old_excluded)
    refusals_excluded_removed = sorted(old_excluded - new_excluded)
    refusals_mismatch_changed = old.get("evaluable_mismatch") != new.get("evaluable_mismatch")
    result["rate_drift"] = refusals_rate_drift
    result["hits_added"] = refusals_hits_added
    result["hits_removed"] = refusals_hits_removed
    result["candidates_added"] = refusals_candidates_added
    result["candidates_removed"] = refusals_candidates_removed
    result["excluded_added"] = refusals_excluded_added
    result["excluded_removed"] = refusals_excluded_removed
    result["protocol_drift"] = protocol_drift
    result["mismatch_changed"] = refusals_mismatch_changed
    # NOTE: refusals-specific disjunct names keep this ``changed``
    # line textually distinct from every other diff section's anchor
    # (mutation-gate uniqueness).
    result["changed"] = bool(
        refusals_rate_drift
        or refusals_hits_added
        or refusals_hits_removed
        or refusals_candidates_added
        or refusals_candidates_removed
        or refusals_excluded_added
        or refusals_excluded_removed
        or refusals_mismatch_changed
        or protocol_drift
    )
    return result


def _valid_scorerlens(payload: Any) -> bool:
    """Structural check for a ``scorerlens`` profile payload.

    The full recompute standard (the wave-7 F6 cluster's bar): the
    ladder digest must be the digest of the payload's OWN
    ladder-as-data (a coherent foreign ladder validates and is
    reported as protocol drift by the diff; a ladder/digest
    mismatch is corruption), and every aggregate -- per-rung match
    and verdict counts, the surface-failure share, the cross-tabs,
    the evaluable-mismatch flag, and the per-rung winner readout
    including its withholding reason -- is re-derived here from
    the per-fixture ground records with independent code, so a
    payload cannot manufacture a winner its records do not support.
    This module is loaded standalone by the cycle-67 gate harness,
    so the recomputation uses only hashlib/json, never an import of
    the producer module.
    """
    if type(payload) is not dict or payload.get("available") is not True:
        return False
    method = payload.get("method")
    if type(method) is not dict:
        return False
    protocol = method.get("protocol")
    if type(protocol) is not str or not protocol:
        return False
    ladder = method.get("ladder")
    if type(ladder) is not dict or ladder.get("protocol") != protocol:
        return False
    ladder_sha = method.get("ladder_sha256")
    canonical = json.dumps(ladder, sort_keys=True, separators=(",", ":"))
    if ladder_sha != hashlib.sha256(canonical.encode("utf-8")).hexdigest():
        return False
    fixture_sha = payload.get("fixture_set_sha256")
    if (
        type(fixture_sha) is not str
        or len(fixture_sha) != 64
        or any(c not in "0123456789abcdef" for c in fixture_sha)
    ):
        return False
    if type(payload.get("evaluable_mismatch")) is not bool:
        return False
    native_winner = payload.get("native_winner")
    if native_winner is not None and type(native_winner) is not str:
        return False
    candidates = payload.get("candidates")
    if type(candidates) is not list or not candidates:
        return False
    n_candidates = payload.get("n_candidates")
    if type(n_candidates) is not int or n_candidates != len(candidates):
        return False
    n_available = 0
    seen_names: set[str] = set()
    evaluable_sets: list[frozenset[str]] = []
    available_entries: list[dict[str, Any]] = []
    for entry in candidates:
        if type(entry) is not dict:
            return False
        name = entry.get("candidate")
        if name is not None and type(name) is not str:
            return False
        if type(name) is str:
            if name in seen_names:
                return False
            seen_names.add(name)
        if type(entry.get("available")) is not bool:
            return False
        if entry.get("available") is not True:
            if type(entry.get("detail")) is not str:
                return False
            continue
        n_available += 1
        available_entries.append(entry)
        if type(entry.get("eligible")) is not bool:
            return False
        n_observations = entry.get("n_observations")
        n_completed_field = entry.get("n_completed")
        if (
            type(n_observations) is not int
            or n_observations < 0
            or type(n_completed_field) is not int
            or n_completed_field < 0
            or n_completed_field > n_observations
        ):
            return False
        records = entry.get("records")
        if type(records) is not list or len(records) != n_observations:
            return False
        n_completed = 0
        for record in records:
            if type(record) is not dict:
                return False
            if type(record.get("fixture")) is not str or not record.get("fixture"):
                return False
            for flag in ("completed", "boundary_found", "exact_match", "excluded"):
                if type(record.get(flag)) is not bool:
                    return False
            flip = record.get("first_flip_rung")
            if flip is not None and (type(flip) is not int or flip < 1 or flip > 4):
                return False
            via = record.get("flip_via")
            if via is not None and via not in ("nfc", "casefold"):
                return False
            if (via is not None) != (flip == 2):
                return False
            if record.get("exact_match") is True and flip is not None:
                return False
            if record.get("excluded") is True and flip is not None:
                return False
            for flag in ("expected_r3_altered", "newline_differs"):
                if type(record.get(flag)) is not bool:
                    return False
            if record.get("completed") is True:
                n_completed += 1
        if n_completed != n_completed_field:
            return False
        evaluable = [r for r in records if r["excluded"] is not True]
        evaluable_sets.append(frozenset(r["fixture"] for r in evaluable))
        match_counts: dict[str, int] = {}
        verdict_counts: dict[str, int] = {}
        for rung in (0, 1, 2, 3, 4):
            key = str(rung)
            match_counts[key] = sum(
                1
                for r in evaluable
                if r["exact_match"] is True
                or (r["first_flip_rung"] is not None and r["first_flip_rung"] <= rung)
            )
            verdict_counts[key] = sum(
                1
                for r in evaluable
                if r["completed"] is True
                and r["boundary_found"] is True
                and (
                    r["exact_match"] is True
                    or (r["first_flip_rung"] is not None and r["first_flip_rung"] <= rung)
                )
            )
        if entry.get("match_counts") != match_counts:
            return False
        if entry.get("verdict_counts") != verdict_counts:
            return False
        n_evaluable = len(evaluable)
        if entry.get("n_evaluable") != n_evaluable:
            return False
        if entry.get("n_excluded_empty_expected") != len(records) - n_evaluable:
            return False
        if entry.get("n_boundary_blocked") != sum(
            1 for r in evaluable if r["completed"] is True and r["boundary_found"] is not True
        ):
            return False
        n_failures = n_evaluable - verdict_counts["0"]
        n_recovered = verdict_counts["4"] - verdict_counts["0"]
        if entry.get("n_native_verdict_failures") != n_failures:
            return False
        if entry.get("n_recovered_by_r4") != n_recovered:
            return False
        flipped = [r for r in evaluable if r["first_flip_rung"] is not None]
        if entry.get("n_flips_boundary_blocked") != sum(
            1 for r in flipped if r["boundary_found"] is not True
        ):
            return False
        if entry.get("n_flips_expected_altered") != sum(
            1 for r in flipped if r["expected_r3_altered"] is True
        ):
            return False
        if entry.get("n_r1_flips_newline_changed") != sum(
            1 for r in flipped if r["first_flip_rung"] == 1 and r["newline_differs"] is True
        ):
            return False
        share = entry.get("surface_failure_share")
        note = entry.get("share_note")
        if note not in ("", "candidate_not_eligible", "no_native_failures"):
            return False
        if (share is None) != (note != ""):
            return False
        if share is not None and not _share_in_unit(share):
            return False
        if entry["eligible"] is not True:
            if note != "candidate_not_eligible":
                return False
        elif n_failures == 0:
            if note != "no_native_failures":
                return False
        elif share != n_recovered / n_failures:
            return False
    n_candidates_available = payload.get("n_candidates_available")
    if type(n_candidates_available) is not int or n_candidates_available != n_available:
        return False
    if payload.get("evaluable_mismatch") != (len(set(evaluable_sets)) > 1):
        return False
    winner_detail = payload.get("winner_detail")
    winners = payload.get("winner_by_rung")
    if winner_detail not in (
        "",
        "candidate_unavailable",
        "candidate_not_eligible",
        "evaluable_mismatch",
    ):
        return False
    if (winners is None) != (winner_detail != ""):
        return False
    if winners is None:
        # The withholding reason must be the derivable one, in the
        # producer's priority order -- a payload may not plead a
        # weaker reason than its records support.
        if n_available != len(candidates):
            expected_detail = "candidate_unavailable"
        elif any(e["eligible"] is not True for e in available_entries):
            expected_detail = "candidate_not_eligible"
        elif payload.get("evaluable_mismatch") is True:
            expected_detail = "evaluable_mismatch"
        else:
            return False
        return bool(winner_detail == expected_detail)
    if type(winners) is not dict or set(winners) != {"0", "1", "2", "3", "4"}:
        return False
    if n_available != len(candidates):
        return False
    if any(e["eligible"] is not True for e in available_entries):
        return False
    if payload.get("evaluable_mismatch") is True:
        return False
    for rung_key in ("0", "1", "2", "3", "4"):
        rung_winners = [
            e["candidate"]
            for e in available_entries
            if e["verdict_counts"][rung_key] == e["n_evaluable"] and e["n_evaluable"] > 0
        ]
        expected_winner = rung_winners[0] if len(rung_winners) == 1 else None
        if winners.get(rung_key) != expected_winner:
            return False
    return True


def diff_scorerlens(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``scorerlens`` regrade payloads; deterministic.

    Answers "did the scorer-lens story change between runs?":
    per-rung search-phase winner drift, per-candidate
    surface-failure-share drift with the failure denominators
    alongside, per-fixture first-flip-rung changes, candidates
    added/removed/excluded, an evaluable-mismatch change, ladder
    protocol drift, and fixture-set drift. When the fixture set or
    the ladder differs, winner and share drift are still listed
    but flagged ``comparable: False`` -- across a different exam,
    a moved winner conflates the exam change with the scorer's
    effect. A side without a recorded probe is reported plainly.
    No model, no network.
    """
    old_ok = _valid_scorerlens(old)
    new_ok = _valid_scorerlens(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    result: dict[str, Any] = {"available": available, "changed": False}
    if available != "both":
        return result
    protocol_drift = (old["method"].get("protocol"), old["method"].get("ladder_sha256")) != (
        new["method"].get("protocol"),
        new["method"].get("ladder_sha256"),
    )
    fixture_set_drift = old.get("fixture_set_sha256") != new.get("fixture_set_sha256")
    comparable = not protocol_drift and not fixture_set_drift

    def _candidate_view(payload: Any) -> dict[str, dict[str, Any]]:
        """Return the candidate view."""
        view: dict[str, dict[str, Any]] = {}
        for entry in payload.get("candidates") or []:
            # _valid_scorerlens guarantees the entry shape, so no
            # defensive skip is needed here.
            name = entry.get("candidate")
            if type(name) is not str:
                continue
            if entry.get("available") is not True:
                view[name] = {"excluded": True}
                continue
            view[name] = {
                "excluded": False,
                "share": entry.get("surface_failure_share"),
                "n": entry.get("n_native_verdict_failures"),
                "flips": {
                    record.get("fixture"): record.get("first_flip_rung")
                    for record in entry.get("records") or []
                },
            }
        return view

    old_view = _candidate_view(old)
    new_view = _candidate_view(new)
    scorerlens_candidates_added = sorted(set(new_view) - set(old_view))
    scorerlens_candidates_removed = sorted(set(old_view) - set(new_view))
    scorerlens_share_drift: list[dict[str, Any]] = []
    scorerlens_flip_changes: list[dict[str, Any]] = []
    for name in sorted(set(old_view) & set(new_view)):
        old_entry = old_view[name]
        new_entry = new_view[name]
        if old_entry["excluded"] or new_entry["excluded"]:
            continue
        if (old_entry["share"], old_entry["n"]) != (new_entry["share"], new_entry["n"]):
            scorerlens_share_drift.append(
                {
                    "candidate": name,
                    "old_share": old_entry["share"],
                    "new_share": new_entry["share"],
                    "old_n": old_entry["n"],
                    "new_n": new_entry["n"],
                }
            )
        for fixture in sorted(set(old_entry["flips"]) & set(new_entry["flips"])):
            if old_entry["flips"][fixture] != new_entry["flips"][fixture]:
                scorerlens_flip_changes.append(
                    {
                        "candidate": name,
                        "fixture": fixture,
                        "old_rung": old_entry["flips"][fixture],
                        "new_rung": new_entry["flips"][fixture],
                    }
                )
    old_excluded = {name for name, v in old_view.items() if v["excluded"]}
    new_excluded = {name for name, v in new_view.items() if v["excluded"]}
    scorerlens_excluded_added = sorted(new_excluded - old_excluded)
    scorerlens_excluded_removed = sorted(old_excluded - new_excluded)
    scorerlens_mismatch_changed = old.get("evaluable_mismatch") != new.get("evaluable_mismatch")
    scorerlens_winner_drift: list[dict[str, Any]] = []
    old_winners = old.get("winner_by_rung")
    new_winners = new.get("winner_by_rung")
    scorerlens_winner_readout_changed = (old_winners is None) != (new_winners is None)
    if type(old_winners) is dict and type(new_winners) is dict:
        for rung in ("0", "1", "2", "3", "4"):
            if old_winners.get(rung) != new_winners.get(rung):
                scorerlens_winner_drift.append(
                    {"rung": int(rung), "old": old_winners.get(rung), "new": new_winners.get(rung)}
                )
    result["winner_drift"] = scorerlens_winner_drift
    result["winner_readout_changed"] = scorerlens_winner_readout_changed
    result["share_drift"] = scorerlens_share_drift
    result["flip_changes"] = scorerlens_flip_changes
    result["candidates_added"] = scorerlens_candidates_added
    result["candidates_removed"] = scorerlens_candidates_removed
    result["excluded_added"] = scorerlens_excluded_added
    result["excluded_removed"] = scorerlens_excluded_removed
    result["protocol_drift"] = protocol_drift
    result["fixture_set_drift"] = fixture_set_drift
    result["comparable"] = comparable
    result["mismatch_changed"] = scorerlens_mismatch_changed
    # NOTE: scorerlens-specific disjunct names keep this ``changed``
    # line textually distinct from every other diff section's anchor
    # (mutation-gate uniqueness).
    result["changed"] = bool(
        scorerlens_winner_drift
        or scorerlens_winner_readout_changed
        or scorerlens_share_drift
        or scorerlens_flip_changes
        or scorerlens_candidates_added
        or scorerlens_candidates_removed
        or scorerlens_excluded_added
        or scorerlens_excluded_removed
        or scorerlens_mismatch_changed
        or protocol_drift
        or fixture_set_drift
    )
    return result


_SERVING_BACKENDS_HEX = frozenset("0123456789abcdef")


_SERVING_PROTOCOL_VERSION = "exec-backends-protocol-1"


_SERVING_STACK = "llamacpp"


_SERVING_MAX_RECORDS = 100_000


_SERVING_HARDWARE_CLASSES = frozenset({"cpu", "cuda", "metal", "rocm", "vulkan"})


_SERVING_FINISH_CLASSES = frozenset(
    {
        "stop",
        "length",
        "abort",
        "error",
        "repetition",
        "eos",
        "word",
        "none",
        "unclassified",
    }
)


_SERVING_CACHE_POSITIONS = frozenset(
    {"cold", "warm_repeat", "shared_prefix", "divergent_control", "standalone"}
)


_SERVING_VERDICTS = frozenset(
    {
        "agree",
        "diverge",
        "unobservable",
        "capability_absent",
        "predictor_detokenization_mismatch",
    }
)


_SERVING_DIVERGENCE_CLASSES = frozenset(
    {
        "KEPT_VS_STRIPPED",
        "FIRING_STEP_DIFFERS",
        "STREAM_TRUNCATED_MID_TOKEN",
        "BYTE_SURFACE_ONLY",
        "MIN_TOKENS_GATED",
        "NO_MIN_TOKENS_GATE",
        "EOS_PRECEDENCE",
        "STR_OVER_EOS_PRECEDENCE",
        "EOS_OVER_STR_PRECEDENCE",
        "MULTI_STOP_SELECTION",
        "SPURIOUS_STOP",
        "MISSING_STOP_AT_CAP",
        "MISSING_EOS_AT_CAP",
        "HARNESS_NORMALIZATION_RELIED",
        "OTHER",
    }
)


_SERVING_CELL_DETAILS = frozenset(
    {
        "",
        "capability_absent",
        "config_echo_mismatch",
        "prompt_echo_mismatch",
        "token_vocab_mismatch",
        "cache_protocol_contaminated",
        "predictor_detokenization_mismatch",
        "eos_attribution_mismatch",
        "stack_response_invalid",
        "stack_response_unclassified",
        "stack_execution_failed",
        "stack_config_unpinned",
        "serving_backends_record_budget_exceeded",
    }
)


_SERVING_CACHE_DETAILS = frozenset(
    {
        "",
        "not_attempted",
        "capability_absent",
        "cache_protocol_contaminated",
        "stack_identity_changed_during_run",
        "serving_backends_record_budget_exceeded",
    }
)


_SERVING_TIER_A = {
    "samplers": ["top_k", "top_p", "min_p", "temperature"],
    "temperature": 0.0,
    "top_k": 0,
    "top_p": 1.0,
    "min_p": 0.0,
    "repeat_penalty": 1.0,
    "dynatemp_range": 0.0,
    "n_probs": 0,
}


_SERVING_STACK_VLLM = "vllm"


_SERVING_STACKS = frozenset({_SERVING_STACK, _SERVING_STACK_VLLM})


_SERVING_TIER_A_VLLM = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 0,
    "min_p": 0.0,
    "repetition_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "n": 1,
    "detokenize": True,
    "skip_special_tokens": True,
    "logprobs": None,
}


_SERVING_VLLM_PROBE_SEED = 1234


_SERVING_VLLM_ABORT_MAX_TOKENS = 256


_SERVING_VLLM_ENV = {
    "VLLM_TARGET_DEVICE": "cpu",
    "VLLM_NO_USAGE_STATS": "1",
    "VLLM_DO_NOT_TRACK": "1",
    "HF_HUB_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
}


_SERVING_VLLM_CONFIG_KEYS = frozenset(
    {
        "protocol",
        "stack",
        "vllm_version",
        "torch_version",
        "transformers_version",
        "vllm_module_digest",
        "platform",
        "env",
        "engine",
        "default_sampling_params_digest",
        "detokenizer_variant",
        "declared_eos_ids",
        "tier_a_vllm",
        "abort_probe",
    }
)


_SERVING_VLLM_ENGINE_KEYS = frozenset(
    {
        "checkpoint_digest",
        "seed",
        "enforce_eager",
        "max_num_seqs",
        "max_model_len",
        "enable_prefix_caching",
        "dtype_resolved",
        "kv_cache_memory_bytes",
        "kv_cache_sizing_input",
        "block_size",
        "num_gpu_blocks",
    }
)


_SERVING_VLLM_ABORT_KEYS = frozenset(
    {
        "surface",
        "terminal_channel",
        "terminal_observed",
        "finish_reason",
        "returned_tokens",
        "elapsed_s",
    }
)


_SERVING_VLLM_DETOKENIZER_VARIANTS = frozenset(
    {"FastIncrementalDetokenizer", "SlowIncrementalDetokenizer"}
)


def _serving_hex64(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(char in _SERVING_BACKENDS_HEX for char in value)
    )


def _serving_canonical(payload: Any) -> str:
    """Canonical JSON (local copy; no imports from
    servingbackends.py anywhere in this mirror)."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _serving_sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _serving_ids_digest(ids: Any) -> str:
    return _serving_sha_text(_serving_canonical(list(ids)))


def _serving_int_list(value: Any) -> bool:
    return type(value) is list and all(
        type(item) is int and not isinstance(item, bool) and item >= 0 for item in value
    )


def _serving_valid_winner(value: Any, *, predicted: bool) -> bool:
    """Stop-winner shape: an index/id pair, never raw stop text."""
    if type(value) is not dict:
        return False
    kind = value.get("kind")
    if kind == "eos" or kind == "stop_token":
        return set(value) == {"kind", "id"} and type(value["id"]) is int and value["id"] >= 0
    if kind == "stop_text":
        if predicted:
            return (
                set(value) == {"kind", "index"}
                and type(value["index"]) is int
                and value["index"] >= 0
            )
        index = value.get("index")
        return (
            set(value) == {"kind", "index", "digest"}
            and (index is None or (type(index) is int and index >= 0))
            and _serving_hex64(value["digest"])
        )
    return False


def _serving_valid_record(
    record: Any, declared_eos: list[int], stack: str = _SERVING_STACK
) -> bool:
    """One execution record, recompute standard (cycle 243).

    Every stored leaf is type-gated before any membership test,
    lookup, or arithmetic (the cycle-241 discipline); digests are
    recomputed from the stored ids/config; the disposition is
    recomputed from the stored digest triple; EOS attribution is
    checked against the identity's declared set. The sampler
    expectation, the two producer-impossible fields, and the
    overshoot tolerance are per-stack (W6 Step 8): the vLLM arm
    legitimately reports a no-special-token digest and cache
    creation counts, and never overshoots its token cap.
    """
    if type(record) is not dict or set(record) != {
        "request_digest",
        "prompt_ids",
        "prompt_id_digest",
        "candidate",
        "fixture",
        "cache_position",
        "request_config",
        "returned_ids",
        "returned_id_digest",
        "returned_text_digest",
        "returned_text_digest_no_special",
        "finish_class",
        "stop_winner",
        "cache_n_reported",
        "tokens_cached_reported",
        "tokens_evaluated_reported",
        "cache_creation_tokens_reported",
        "mirror",
        "cell_detail",
    }:
        return False
    if not _serving_int_list(record["prompt_ids"]):
        return False
    if not _serving_hex64(record["prompt_id_digest"]):
        return False
    if _serving_ids_digest(record["prompt_ids"]) != record["prompt_id_digest"]:
        return False
    if type(record["candidate"]) is not str or not record["candidate"]:
        return False
    if type(record["fixture"]) is not str or not record["fixture"]:
        return False
    if (
        type(record["cache_position"]) is not str
        or record["cache_position"] not in _SERVING_CACHE_POSITIONS
    ):
        return False
    config = record["request_config"]
    if type(config) is not dict or set(config) != {
        "max_new_tokens",
        "stop_text_digests",
        "stop_token_ids",
        "eos_id",
        "include_stop_str_in_output",
        "ignore_eos",
        "min_tokens",
        "cache_salt_present",
        "cache_mode",
        "seed",
        "sampler",
    }:
        return False
    if type(config["max_new_tokens"]) is not int or not 1 <= config["max_new_tokens"] <= 4096:
        return False
    if type(config["stop_text_digests"]) is not list or not all(
        _serving_hex64(digest) for digest in config["stop_text_digests"]
    ):
        return False
    if not _serving_int_list(config["stop_token_ids"]):
        return False
    if config["eos_id"] is not None and (type(config["eos_id"]) is not int or config["eos_id"] < 0):
        return False
    for flag in ("include_stop_str_in_output", "ignore_eos", "cache_salt_present"):
        if type(config[flag]) is not bool:
            return False
    if type(config["min_tokens"]) is not int or config["min_tokens"] < 0:
        return False
    if (
        type(config["cache_mode"]) is not str
        or config["cache_mode"] not in _SERVING_CACHE_POSITIONS
    ):
        return False
    if type(config["seed"]) is not int or config["seed"] < 0:
        return False
    tier_a = _SERVING_TIER_A_VLLM if stack == _SERVING_STACK_VLLM else _SERVING_TIER_A
    sampler = config["sampler"]
    if type(sampler) is not dict or set(sampler) != set(tier_a) | {"seed"}:
        return False
    expected_sampler = {**tier_a, "seed": config["seed"]}
    if not _recomputed_equal(sampler, expected_sampler):
        return False
    if not _serving_hex64(record["request_digest"]):
        return False
    expected_request_digest = _serving_sha_text(
        _serving_canonical({"config": config, "prompt_id_digest": record["prompt_id_digest"]})
    )
    if record["request_digest"] != expected_request_digest:
        return False
    returned = record["returned_ids"]
    if returned is not None and not _serving_int_list(returned):
        return False
    if returned is None:
        if record["returned_id_digest"] is not None:
            return False
    elif not _serving_hex64(record["returned_id_digest"]) or (
        _serving_ids_digest(returned) != record["returned_id_digest"]
    ):
        return False
    if record["returned_text_digest"] is not None and not _serving_hex64(
        record["returned_text_digest"]
    ):
        return False
    if stack == _SERVING_STACK_VLLM:
        if record["returned_text_digest_no_special"] is not None and not _serving_hex64(
            record["returned_text_digest_no_special"]
        ):
            return False
        creation = record["cache_creation_tokens_reported"]
        if creation is not None and (type(creation) is not int or creation < 0):
            return False
    else:
        if record["returned_text_digest_no_special"] is not None:
            return False
        if record["cache_creation_tokens_reported"] is not None:
            return False
    if (
        type(record["finish_class"]) is not str
        or record["finish_class"] not in _SERVING_FINISH_CLASSES
    ):
        return False
    winner = record["stop_winner"]
    if winner is not None and not _serving_valid_winner(winner, predicted=False):
        return False
    for key in ("cache_n_reported", "tokens_cached_reported", "tokens_evaluated_reported"):
        value = record[key]
        if value is not None and (type(value) is not int or value < 0):
            return False
    if type(record["cell_detail"]) is not str or record["cell_detail"] not in _SERVING_CELL_DETAILS:
        return False
    mirror = record["mirror"]
    if type(mirror) is not dict or set(mirror) != {
        "predicted_firing_step",
        "predicted_winner",
        "precedence_tie",
        "local_text_digest",
        "stripped_text_digest",
        "server_detok_digest",
        "predicted_disposition",
        "observed_disposition",
        "fidelity_verdict",
        "divergence_class",
        "divergence_detail",
        "hf_stream",
    }:
        return False
    step = mirror["predicted_firing_step"]
    if step is not None and (type(step) is not int or step < 1):
        return False
    if mirror["predicted_winner"] is not None and not _serving_valid_winner(
        mirror["predicted_winner"], predicted=True
    ):
        return False
    if type(mirror["precedence_tie"]) is not bool:
        return False
    if not _serving_hex64(mirror["local_text_digest"]):
        return False
    for key in ("stripped_text_digest", "server_detok_digest"):
        if mirror[key] is not None and not _serving_hex64(mirror[key]):
            return False
    if mirror["predicted_disposition"] not in (None, "kept", "stripped"):
        return False
    if mirror["observed_disposition"] not in (None, "kept", "stripped", "unobservable"):
        return False
    if (
        type(mirror["fidelity_verdict"]) is not str
        or mirror["fidelity_verdict"] not in _SERVING_VERDICTS
    ):
        return False
    divergence_class = mirror["divergence_class"]
    if divergence_class is not None and (
        type(divergence_class) is not str or divergence_class not in _SERVING_DIVERGENCE_CLASSES
    ):
        return False
    if type(mirror["divergence_detail"]) is not str:
        return False
    stream = mirror["hf_stream"]
    if stream is not None:
        if type(stream) is not dict or set(stream) != {
            "match",
            "first_divergence",
            "shared_prefix",
            "recorded_length",
            "replayed_length",
            "class",
        }:
            return False
        if type(stream["match"]) is not bool:
            return False
        if stream["first_divergence"] is not None and (
            type(stream["first_divergence"]) is not int or stream["first_divergence"] < 0
        ):
            return False
        for key in ("shared_prefix", "recorded_length", "replayed_length"):
            if type(stream[key]) is not int or stream[key] < 0:
                return False
        if stream["match"]:
            if stream["first_divergence"] is not None or stream["class"] is not None:
                return False
        elif stream["class"] != "GENERATION_STREAM_DIVERGED":
            return False
    # Cross-field recomputes.
    verdict = mirror["fidelity_verdict"]
    if record["cell_detail"] == "capability_absent" and (
        returned is not None or verdict != "capability_absent"
    ):
        return False
    if verdict == "capability_absent" and record["cell_detail"] != "capability_absent":
        return False
    if (
        verdict == "diverge"
        and divergence_class is None
        and mirror["divergence_detail"] != "eos_attribution_mismatch"
    ):
        return False
    if verdict == "agree" and divergence_class is not None:
        return False
    if verdict == "agree":
        # Agreement is a claim about a prediction meeting an
        # observation; the record's own fields must not disprove
        # it (BH14-6). The producer diverges on a firing with no
        # prediction (SPURIOUS_STOP) and on a prediction with no
        # firing (MISSING_STOP_AT_CAP), is unobservable when the
        # predicted step is missing, and diverges when the steps
        # differ — so an agree record needs both sides present at
        # the same step. A stop-token winner the request never
        # listed (and that is not a declared EOS) could not have
        # been predicted from this record's own configuration.
        predicted_winner = mirror["predicted_winner"]
        if winner is not None and (predicted_winner is None or step is None):
            return False
        if winner is None and predicted_winner is not None:
            return False
        if winner is not None and returned is not None and step != len(returned):
            return False
        if (
            type(winner) is dict
            and winner.get("kind") == "stop_token"
            and type(winner.get("id")) is int
            and winner["id"] not in config["stop_token_ids"]
            and winner["id"] not in declared_eos
        ):
            return False
    if winner is not None and type(winner) is dict and winner.get("kind") == "stop_text":
        stack_digest = record["returned_text_digest"]
        if stack_digest is None:
            expected_disposition: Any = "unobservable"
        elif stack_digest == mirror["local_text_digest"]:
            expected_disposition = "kept"
        elif (
            mirror["stripped_text_digest"] is not None
            and stack_digest == mirror["stripped_text_digest"]
        ):
            expected_disposition = "stripped"
        else:
            expected_disposition = "unobservable"
        if mirror["observed_disposition"] != expected_disposition:
            return False
    elif mirror["observed_disposition"] is not None:
        # Without a stop-text winner there is no disposition to
        # observe: the producer reports None.
        return False
    if (
        winner is not None
        and type(winner) is dict
        and winner.get("kind") == "eos"
        and type(winner.get("id")) is int
        and winner["id"] not in declared_eos
        and (record["cell_detail"] != "eos_attribution_mismatch" or verdict != "diverge")
    ):
        return False
    overshoot = 0 if stack == _SERVING_STACK_VLLM else 1
    if (
        returned is not None
        and len(returned) > config["max_new_tokens"] + overshoot
        and record["cell_detail"] != "stack_response_invalid"
    ):
        return False
    return not (
        divergence_class == "FIRING_STEP_DIFFERS"
        and step is not None
        and returned is not None
        and step == len(returned)
    )


def _serving_recompute_rows(
    records: list[dict[str, Any]], stack: str = _SERVING_STACK
) -> list[dict[str, Any]]:
    """The stop-matrix companion derivation, re-implemented. The
    cells key is the payload's own stack (W6 Step 8)."""
    rows: list[dict[str, Any]] = []
    for record in records:
        if record.get("cache_position") != "standalone":
            continue
        mirror = record["mirror"]
        winner = record.get("stop_winner")
        available = record.get("cell_detail") != "capability_absent"
        # Mirror of the producer: an unavailable cell's fired state
        # is unobservable (None), never a measured negative (ORCH-04).
        fired = (winner is not None) if available else None
        returned = record.get("returned_ids")
        config = record.get("request_config") or {}
        if config.get("include_stop_str_in_output"):
            arm = "keep"
        elif mirror.get("hf_stream") is not None:
            arm = "strip"
        else:
            arm = "plain"
        cell = {
            "available": available,
            "fired": fired,
            "firing_step": len(returned) if fired and type(returned) is list else None,
            "disposition": mirror.get("observed_disposition"),
            "finish_class": record.get("finish_class"),
            "verdict": mirror.get("fidelity_verdict"),
            "detail": record.get("cell_detail", ""),
        }
        flips = (
            [mirror["divergence_class"]]
            if mirror.get("fidelity_verdict") == "diverge" and mirror.get("divergence_class")
            else []
        )
        rows.append(
            {
                "fixture": record.get("fixture"),
                "candidate": record.get("candidate"),
                "arm": arm,
                "cells": {stack: cell},
                "flips": flips,
            }
        )
    return rows


def _serving_recompute_fidelity(records: list[dict[str, Any]]) -> dict[str, Any]:
    """The §9.3 aggregation rule, re-implemented."""
    counts = {
        "agree": 0,
        "diverge": 0,
        "unobservable": 0,
        "capability_absent": 0,
        "predictor_detokenization_mismatch": 0,
    }
    classes: dict[str, int] = {}
    for record in records:
        verdict = record["mirror"]["fidelity_verdict"]
        if verdict in counts:
            counts[verdict] += 1
        else:
            counts["unobservable"] += 1
        if verdict == "diverge":
            divergence_class = record["mirror"]["divergence_class"]
            if type(divergence_class) is str and divergence_class:
                classes[divergence_class] = classes.get(divergence_class, 0) + 1
    total = len(records)
    if total == 0:
        verdict = "fidelity_unavailable"
    elif counts["diverge"]:
        verdict = "fidelity_diverge"
    elif (
        counts["unobservable"]
        or counts["capability_absent"]
        or counts["predictor_detokenization_mismatch"]
    ):
        verdict = "fidelity_unavailable"
    else:
        verdict = "fidelity_agree"
    return {
        "available": total > 0,
        "verdict": verdict,
        "counts": counts,
        "divergence_classes": classes,
    }


def _serving_valid_llama_config(
    config: dict[str, Any], identity: dict[str, Any], declared: list[int]
) -> bool:
    """The llamacpp identity config, exactly as the producer emits
    it (the pre-W6 checks, verbatim)."""
    if set(config) != {
        "protocol",
        "stack",
        "build_info",
        "total_slots",
        "model_path_digest",
        "checkpoint_sha256",
        "declared_eos_ids",
        "generation_settings_digest",
        "posture",
        "cache_probe",
        "tier_a",
    }:
        return False
    if config["protocol"] != _SERVING_PROTOCOL_VERSION or config["stack"] != _SERVING_STACK:
        return False
    if config["build_info"] != identity["stack_version"]:
        return False
    if config["total_slots"] != 1 or type(config["total_slots"]) is not int:
        return False
    if not _serving_hex64(config["model_path_digest"]):
        return False
    if config["checkpoint_sha256"] != identity["checkpoint_digest"]:
        return False
    if config["declared_eos_ids"] != declared:
        return False
    if config["generation_settings_digest"] is not None and not _serving_hex64(
        config["generation_settings_digest"]
    ):
        return False
    posture = config["posture"]
    if type(posture) is not dict or set(posture) != {
        "api_key_required",
        "slot_erase_available",
        "endpoint_slots",
        "ui_enabled",
    }:
        return False
    if posture["api_key_required"] is not True:
        return False
    for key in ("slot_erase_available", "endpoint_slots", "ui_enabled"):
        if type(posture[key]) is not bool:
            return False
    cache_probe = config["cache_probe"]
    if type(cache_probe) is not dict or set(cache_probe) != {"cold_cache_n", "warm_cache_n"}:
        return False
    for key in ("cold_cache_n", "warm_cache_n"):
        if cache_probe[key] is not None and (
            type(cache_probe[key]) is not int or cache_probe[key] < 0
        ):
            return False
    return _recomputed_equal(config["tier_a"], _SERVING_TIER_A)


def _serving_valid_vllm_config(
    config: dict[str, Any], identity: dict[str, Any], declared: list[int]
) -> bool:
    """The vLLM identity config: the closed key set fixed in
    advance for the adapter (digest/scalar-only), with the probed
    values the mirror pins independently — the six environment
    pins, the singleton eager engine shape, the Tier-A tuple at
    the probe seed, and the generate-family abort cell."""
    if set(config) != _SERVING_VLLM_CONFIG_KEYS:
        return False
    if config["protocol"] != _SERVING_PROTOCOL_VERSION or config["stack"] != _SERVING_STACK_VLLM:
        return False
    if config["vllm_version"] != identity["stack_version"]:
        return False
    for key in ("torch_version", "transformers_version"):
        if type(config[key]) is not str or not config[key]:
            return False
    if not _serving_hex64(config["vllm_module_digest"]):
        return False
    if config["platform"] != {"device_type": "cpu", "platform_class": "CpuPlatform"}:
        return False
    if config["env"] != _SERVING_VLLM_ENV:
        return False
    engine = config["engine"]
    if type(engine) is not dict or set(engine) != _SERVING_VLLM_ENGINE_KEYS:
        return False
    if engine["checkpoint_digest"] != identity["checkpoint_digest"]:
        return False
    if engine["seed"] != 0 or type(engine["seed"]) is not int:
        return False
    if engine["enforce_eager"] is not True or engine["enable_prefix_caching"] is not True:
        return False
    if engine["max_num_seqs"] != 1 or type(engine["max_num_seqs"]) is not int:
        return False
    if type(engine["max_model_len"]) is not int or engine["max_model_len"] < 1:
        return False
    if type(engine["dtype_resolved"]) is not str or not engine["dtype_resolved"]:
        return False
    kv_bytes = engine["kv_cache_memory_bytes"]
    if type(kv_bytes) is not int or kv_bytes <= 0:
        return False
    if engine["kv_cache_sizing_input"] != {"kwarg_bytes": kv_bytes}:
        return False
    if type(engine["block_size"]) is not int or engine["block_size"] < 1:
        return False
    if type(engine["num_gpu_blocks"]) is not int or engine["num_gpu_blocks"] < 1:
        return False
    if not _serving_hex64(config["default_sampling_params_digest"]):
        return False
    if type(config["detokenizer_variant"]) is not str or (
        config["detokenizer_variant"] not in _SERVING_VLLM_DETOKENIZER_VARIANTS
    ):
        return False
    if config["declared_eos_ids"] != declared:
        return False
    if not _recomputed_equal(
        config["tier_a_vllm"], {**_SERVING_TIER_A_VLLM, "seed": _SERVING_VLLM_PROBE_SEED}
    ):
        return False
    abort = config["abort_probe"]
    if type(abort) is not dict or set(abort) != _SERVING_VLLM_ABORT_KEYS:
        return False
    if abort["surface"] != "llm_engine.abort_request":
        return False
    if abort["terminal_channel"] != "generate_family":
        return False
    if type(abort["terminal_observed"]) is not bool:
        return False
    if abort["finish_reason"] not in (None, "abort"):
        return False
    if abort["returned_tokens"] is not None and (
        type(abort["returned_tokens"]) is not int or abort["returned_tokens"] < 0
    ):
        return False
    # Cross-field coherence (BH14-5): the fields only mean
    # something together — an observed terminal is a finished
    # "abort" terminal with the stream length it returned (at
    # most the cell's 256 tokens); an unobserved one carries
    # neither the reason nor the length.
    if abort["terminal_observed"]:
        if abort["finish_reason"] != "abort":
            return False
        if (
            type(abort["returned_tokens"]) is not int
            or abort["returned_tokens"] > _SERVING_VLLM_ABORT_MAX_TOKENS
        ):
            return False
    elif abort["finish_reason"] is not None or abort["returned_tokens"] is not None:
        return False
    elapsed = abort["elapsed_s"]
    return type(elapsed) in (int, float) and not isinstance(elapsed, bool) and elapsed >= 0


def _valid_serving_backends(payload: Any) -> bool:
    """Structural check for a ``serving_backends`` payload
    (recompute standard). The audit block and the fidelity
    per-fixture rows are shape-checked only: they carry stage
    observations (planned/executed counts, the fidelity entry's
    per-fixture stream comparisons) that the records alone do not
    recompute; everything the records do determine — digests, the
    identity fingerprint, the stop-matrix rows, the cache positions,
    the fidelity counts and verdict — is recomputed and compared.
    """
    if type(payload) is not dict or payload.get("available") is not True:
        return False
    if set(payload) != {
        "available",
        "detail",
        "protocol",
        "identity",
        "capabilities",
        "records",
        "stop_matrix",
        "cache",
        "fidelity",
        "audit",
    }:
        return False
    if type(payload["detail"]) is not str or payload["detail"] not in (
        "",
        "serving_backends_record_budget_exceeded",
    ):
        return False
    protocol = payload["protocol"]
    if (
        type(protocol) is not dict
        or set(protocol) != {"version"}
        or protocol["version"] != _SERVING_PROTOCOL_VERSION
    ):
        return False
    identity = payload["identity"]
    if type(identity) is not dict or set(identity) != {
        "stack",
        "stack_version",
        "checkpoint_digest",
        "checkpoint_bytes",
        "tokenizer_fingerprint",
        "hardware_class",
        "declared_eos_ids",
        "config",
        "config_fingerprint",
    }:
        return False
    if type(identity["stack"]) is not str or identity["stack"] not in _SERVING_STACKS:
        return False
    stack = identity["stack"]
    if type(identity["stack_version"]) is not str or not identity["stack_version"]:
        return False
    if not _serving_hex64(identity["checkpoint_digest"]):
        return False
    if type(identity["checkpoint_bytes"]) is not int or identity["checkpoint_bytes"] <= 0:
        return False
    tokenizer_fingerprint = identity["tokenizer_fingerprint"]
    if (
        type(tokenizer_fingerprint) is not str
        or not tokenizer_fingerprint.startswith("sha256:")
        or not _serving_hex64(tokenizer_fingerprint[len("sha256:") :])
    ):
        return False
    if (
        type(identity["hardware_class"]) is not str
        or identity["hardware_class"] not in _SERVING_HARDWARE_CLASSES
    ):
        return False
    declared = identity["declared_eos_ids"]
    if not _serving_int_list(declared) or not declared:
        return False
    if declared != sorted(set(declared)):
        return False
    config = identity["config"]
    if type(config) is not dict:
        return False
    if stack == _SERVING_STACK_VLLM:
        if not _serving_valid_vllm_config(config, identity, declared):
            return False
    elif not _serving_valid_llama_config(config, identity, declared):
        return False
    if not _serving_hex64(identity["config_fingerprint"]):
        return False
    if _serving_sha_text(_serving_canonical(config)) != identity["config_fingerprint"]:
        return False
    capabilities = payload["capabilities"]
    if type(capabilities) is not dict or set(capabilities) != {
        "raw_token_ids",
        "stop_token_ids",
        "cached_token_reporting",
        "token_id_echo",
        "prompt_echo",
        "min_tokens",
        "include_stop_str_in_output",
    }:
        return False
    if not all(type(value) is bool for value in capabilities.values()):
        return False
    records = payload["records"]
    if type(records) is not list or len(records) > _SERVING_MAX_RECORDS:
        return False
    for record in records:
        if not _serving_valid_record(record, declared, stack):
            return False
    stop_matrix = payload["stop_matrix"]
    if type(stop_matrix) is not dict or set(stop_matrix) != {"available", "rows"}:
        return False
    expected_rows = _serving_recompute_rows(records, stack)
    if stop_matrix["available"] is not bool(len(expected_rows)):
        return False
    if not _recomputed_equal(stop_matrix["rows"], expected_rows):
        return False
    cache = payload["cache"]
    if type(cache) is not dict or set(cache) != {"available", "detail", "positions"}:
        return False
    if type(cache["detail"]) is not str or cache["detail"] not in _SERVING_CACHE_DETAILS:
        return False
    expected_positions: dict[str, Any] = {}
    cache_records = [record for record in records if record["cache_position"] != "standalone"]
    for record in cache_records:
        if record["cell_detail"] in (
            "capability_absent",
            "serving_backends_record_budget_exceeded",
        ):
            continue
        if stack == _SERVING_STACK_VLLM:
            expected_positions[record["cache_position"]] = {
                "tokens_cached": record["tokens_cached_reported"],
                "cache_creation_tokens": record["cache_creation_tokens_reported"],
            }
        else:
            expected_positions[record["cache_position"]] = {
                "cache_n": record["cache_n_reported"],
                "tokens_cached": record["tokens_cached_reported"],
                "tokens_evaluated": record["tokens_evaluated_reported"],
            }
    if not _recomputed_equal(cache["positions"], expected_positions):
        return False
    if cache["available"] is not bool(len(expected_positions)):
        return False
    if cache["detail"] == "not_attempted" and cache_records:
        return False
    cold = next(
        (record for record in cache_records if record["cache_position"] == "cold"),
        None,
    )
    divergent = next(
        (record for record in cache_records if record["cache_position"] == "divergent_control"),
        None,
    )
    if stack == _SERVING_STACK_VLLM:
        block_size = config["engine"]["block_size"]
        contaminated = any(
            record is not None
            and type(record["tokens_cached_reported"]) is int
            and record["tokens_cached_reported"] >= block_size
            for record in (cold, divergent)
        )
    else:
        contaminated = any(
            record is not None
            and type(record["cache_n_reported"]) is int
            and record["cache_n_reported"] > 1
            for record in (cold, divergent)
        )
    if contaminated and cache["detail"] != "cache_protocol_contaminated":
        return False
    if cache["detail"] == "cache_protocol_contaminated" and not contaminated:
        return False
    fidelity = payload["fidelity"]
    if type(fidelity) is not dict or set(fidelity) != {
        "available",
        "detail",
        "verdict",
        "counts",
        "divergence_classes",
        "per_fixture",
    }:
        return False
    expected_fidelity = _serving_recompute_fidelity(records)
    for key in ("available", "verdict", "counts", "divergence_classes"):
        if not _recomputed_equal(fidelity[key], expected_fidelity[key]):
            return False
    if fidelity["detail"] != ("" if expected_fidelity["available"] else "not_attempted"):
        return False
    per_fixture = fidelity["per_fixture"]
    if type(per_fixture) is not list:
        return False
    for row in per_fixture:
        if type(row) is not dict or set(row) != {
            "fixture",
            "verdict",
            "divergences",
            "stream",
            "arms",
        }:
            return False
        if type(row["fixture"]) is not str or not row["fixture"]:
            return False
        if type(row["verdict"]) is not str or row["verdict"] not in _SERVING_VERDICTS:
            return False
        if type(row["divergences"]) is not list or type(row["arms"]) is not list:
            return False
        if not all(type(arm) is str and arm in _SERVING_VERDICTS for arm in row["arms"]):
            return False
        for finding in row["divergences"]:
            if type(finding) is not dict or set(finding) != {"class", "detail", "severity"}:
                return False
            if finding["class"] is not None and (
                type(finding["class"]) is not str
                or finding["class"] not in _SERVING_DIVERGENCE_CLASSES
            ):
                return False
            if type(finding["detail"]) is not str or finding["severity"] not in (
                "high",
                "medium",
                "low",
            ):
                return False
        stream = row["stream"]
        if stream is not None:
            if type(stream) is not dict or set(stream) != {
                "match",
                "first_divergence",
                "shared_prefix",
                "recorded_length",
                "replayed_length",
                "class",
            }:
                return False
            if type(stream["match"]) is not bool:
                return False
            if stream["class"] not in (None, "GENERATION_STREAM_DIVERGED"):
                return False
    audit = payload["audit"]
    if audit is not None:
        if type(audit) is not dict or set(audit) != {
            "n_records",
            "n_planned_cells",
            "n_executions",
            "budget_spent",
        }:
            return False
        for key in ("n_records", "n_planned_cells", "n_executions"):
            if type(audit[key]) is not int or audit[key] < 0:
                return False
        if type(audit["budget_spent"]) is not bool:
            return False
        if audit["n_records"] != len(records):
            return False
    return True


def diff_serving_backends(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``serving_backends`` payloads; deterministic.

    Answers "did the measured serving-stack story change between
    runs?": identity drift (a different stack build, checkpoint, or
    configuration fingerprint means the two runs measured different
    stacks — the comparison is then not comparable, with the drift
    still listed), the fidelity verdict, the cache companion's
    detail, and per-record drift keyed by request digest (records
    added/removed; verdict, finish, cache, and cell-detail changes
    on shared records). A side without a valid payload is reported
    plainly.
    """
    old_ok = _valid_serving_backends(old)
    new_ok = _valid_serving_backends(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    result: dict[str, Any] = {"available": available, "changed": False}
    if available != "both":
        return result
    identity_drift: list[dict[str, Any]] = []
    for field_name in (
        "stack",
        "stack_version",
        "checkpoint_digest",
        "tokenizer_fingerprint",
        "hardware_class",
        "config_fingerprint",
    ):
        if old["identity"][field_name] != new["identity"][field_name]:
            identity_drift.append(
                {
                    "field": field_name,
                    "old": old["identity"][field_name],
                    "new": new["identity"][field_name],
                }
            )
    old_records = {record["request_digest"]: record for record in old["records"]}
    new_records = {record["request_digest"]: record for record in new["records"]}
    records_added = sorted(set(new_records) - set(old_records))
    records_removed = sorted(set(old_records) - set(new_records))
    cache_metrics: tuple[tuple[str, str], ...]
    if old["identity"]["stack"] == _SERVING_STACK_VLLM:
        cache_metrics = (
            ("tokens_cached_reported", "tokens_cached_reported"),
            ("cache_creation_tokens_reported", "cache_creation_tokens_reported"),
        )
    else:
        cache_metrics = (("cache_n_reported", "cache_n_reported"),)
    record_drift: list[dict[str, Any]] = []
    for digest in sorted(set(old_records) & set(new_records)):
        old_record = old_records[digest]
        new_record = new_records[digest]
        for metric, old_value, new_value in (
            (
                "fidelity_verdict",
                old_record["mirror"]["fidelity_verdict"],
                new_record["mirror"]["fidelity_verdict"],
            ),
            ("finish_class", old_record["finish_class"], new_record["finish_class"]),
            *((metric, old_record[key], new_record[key]) for metric, key in cache_metrics),
            ("cell_detail", old_record["cell_detail"], new_record["cell_detail"]),
        ):
            if old_value != new_value:
                record_drift.append(
                    {
                        "request_digest": digest,
                        "metric": metric,
                        "old": old_value,
                        "new": new_value,
                    }
                )
    fidelity_drift = {
        "old": old["fidelity"]["verdict"],
        "new": new["fidelity"]["verdict"],
        "changed": old["fidelity"]["verdict"] != new["fidelity"]["verdict"],
    }
    cache_drift = {
        "old": old["cache"]["detail"],
        "new": new["cache"]["detail"],
        "changed": old["cache"]["detail"] != new["cache"]["detail"],
    }
    changed = bool(
        identity_drift
        or fidelity_drift["changed"]
        or cache_drift["changed"]
        or records_added
        or records_removed
        or record_drift
    )
    result.update(
        {
            "changed": changed,
            "comparable": not identity_drift,
            "identity_drift": identity_drift,
            "fidelity": fidelity_drift,
            "cache": cache_drift,
            "records_added": records_added,
            "records_removed": records_removed,
            "record_drift": record_drift,
        }
    )
    return result


_NEGATION_ARM_KEYS: tuple[str, ...] = ("direct", "negated", "flipped", "negated_alt")


_NEGATION_AGGREGATE_KEYS = frozenset(
    {
        "arms",
        "headline",
        "descriptive",
        "ids",
        "discordance",
        "operating_characteristics",
        "n_cases",
        "n_incomplete",
    }
)


def _valid_negation(payload: Any) -> bool:
    """Recompute-standard check for a ``negation`` payload (RR-03).

    The aggregation is a pure data function in its own module
    (``negation.aggregate_candidate``), shared with the stage;
    this validator independently gates the payload's shape --
    exact key sets, candidate/case counts against the frozen
    case generator -- and compares every stored aggregate
    against the recomputation from the stored records, so a
    tampered or drifted aggregate is refused. The protocol is
    recomputed, not merely type-checked (wave-10 F4): the
    stored object must equal the frozen instrument for its own
    seed/backend pair, field for field, and its fingerprint
    must be that object's fingerprint. The records are bound
    to the generated case set (wave-10 L4): one record per
    case, each record's skeleton and digests recomputed
    against its case. Malformed records fail closed: the
    recomputation runs under a guard and any error makes the
    payload invalid, never a crash (RR-02 totality).
    """
    if type(payload) is not dict or payload.get("available") is not True:
        return False
    if set(payload) != {
        "available",
        "detail",
        "protocol",
        "audit",
        "candidates",
        "n_candidates",
        "n_cases",
    }:
        return False
    if payload.get("detail") != "":
        return False
    protocol = payload.get("protocol")
    if type(protocol) is not dict:
        return False
    seed = protocol.get("seed")
    backend_fingerprint = protocol.get("backend_fingerprint")
    if type(seed) is not int or type(backend_fingerprint) is not str:
        return False
    if type(protocol.get("fingerprint")) is not str:
        return False
    audit = payload.get("audit")
    if audit is not None and type(audit) is not dict:
        return False
    candidates = payload.get("candidates")
    if type(candidates) is not dict:
        return False
    if not _recomputed_equal(payload.get("n_candidates"), len(candidates)):
        return False
    from . import negation as _ng

    try:
        cases = _ng.cases_for(seed)
        expected_protocol = _ng.protocol_object(seed, backend_fingerprint)
    except Exception:  # noqa: BLE001 -- fail closed: an ungeneratable seed invalidates
        return False
    if set(protocol) != set(expected_protocol) | {"fingerprint"}:
        return False
    for protocol_key, protocol_value in expected_protocol.items():
        if not _recomputed_equal(protocol.get(protocol_key), protocol_value):
            return False
    if protocol["fingerprint"] != _ng.protocol_fingerprint(seed, backend_fingerprint):
        return False
    if not _recomputed_equal(payload.get("n_cases"), len(cases)):
        return False
    case_by_id = {case["fixture_id"]: case for case in cases}
    for entry in candidates.values():
        if type(entry) is not dict:
            return False
        status = entry.get("status")
        if status == "unavailable":
            if set(entry) != {"status", "detail", "records"}:
                return False
            if type(entry.get("detail")) is not str or entry.get("records") != []:
                return False
            continue
        if status != "available":
            return False
        if set(entry) != _NEGATION_AGGREGATE_KEYS | {"status", "detail", "records"}:
            return False
        if entry.get("detail") != "":
            return False
        records = entry.get("records")
        if type(records) is not list:
            return False
        record_fixtures: list[str] = []
        for record in records:
            if type(record) is not dict or type(record.get("fixture")) is not str:
                return False
            record_fixtures.append(record["fixture"])
        if sorted(record_fixtures) != sorted(case_by_id):
            return False
        for record in records:
            case = case_by_id[record["fixture"]]
            for skeleton_key in ("operands", "larger_first", "stratum", "band"):
                if not _recomputed_equal(record.get(skeleton_key), case[skeleton_key]):
                    return False
            expected_digests = {arm: _ng.digest_text(case["contents"][arm]) for arm in _ng.ARM_KEYS}
            if not _recomputed_equal(record.get("content_digests"), expected_digests):
                return False
            if not _recomputed_equal(
                record.get("expected_digest"),
                _ng.digest_text(case["expected"]["direct"]),
            ):
                return False
        try:
            expected = _ng.aggregate_candidate(records)
        except Exception:  # noqa: BLE001 -- fail closed on malformed records
            return False
        for key, value in expected.items():
            if not _recomputed_equal(entry.get(key), value):
                return False
    return True


def diff_negation(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``negation`` payloads; deterministic.

    Answers "did the measured negation story change between
    runs?": protocol drift (a different seed/backend instrument
    means the two runs measured different case sets -- the
    comparison is then not comparable, with the drift still
    listed), the case-count axis, candidates added/removed,
    per-candidate status changes, headline decision and joint
    counts, per-arm rates and completions, discordance counts,
    and incomplete-case counts. A side without a valid payload
    is reported plainly.
    """
    old_ok = _valid_negation(old)
    new_ok = _valid_negation(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    result: dict[str, Any] = {"available": available, "changed": False}
    if available != "both":
        return result
    protocol_drift = old["protocol"].get("fingerprint") != new["protocol"].get("fingerprint")
    old_candidates = old["candidates"]
    new_candidates = new["candidates"]
    candidates_added = sorted(set(new_candidates) - set(old_candidates))
    candidates_removed = sorted(set(old_candidates) - set(new_candidates))
    candidate_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_candidates) & set(new_candidates)):
        old_entry = old_candidates[name]
        new_entry = new_candidates[name]
        if old_entry.get("status") != new_entry.get("status"):
            candidate_drift.append(
                {
                    "candidate": name,
                    "metric": "status",
                    "old": old_entry.get("status"),
                    "new": new_entry.get("status"),
                }
            )
            continue
        if old_entry.get("status") != "available":
            continue
        metrics: list[tuple[str, Any, Any]] = []
        for key in ("decision", "joint", "direct_passes", "n_population", "flipped_rate_on_d"):
            metrics.append(
                (f"headline.{key}", old_entry["headline"].get(key), new_entry["headline"].get(key))
            )
        for arm in _NEGATION_ARM_KEYS:
            for key in ("rate", "n_completed", "n_pass"):
                metrics.append(
                    (
                        f"arms.{arm}.{key}",
                        old_entry["arms"][arm].get(key),
                        new_entry["arms"][arm].get(key),
                    )
                )
        for key in ("boundary_only", "exact_only", "mixed"):
            metrics.append(
                (
                    f"discordance.{key}",
                    old_entry["discordance"].get(key),
                    new_entry["discordance"].get(key),
                )
            )
        metrics.append(
            ("n_incomplete", old_entry.get("n_incomplete"), new_entry.get("n_incomplete"))
        )
        for metric, old_value, new_value in metrics:
            if not _recomputed_equal(old_value, new_value):
                candidate_drift.append(
                    {"candidate": name, "metric": metric, "old": old_value, "new": new_value}
                )
    n_cases = {
        "old": old["n_cases"],
        "new": new["n_cases"],
        "changed": old["n_cases"] != new["n_cases"],
    }
    changed = bool(
        protocol_drift
        or candidates_added
        or candidates_removed
        or candidate_drift
        or n_cases["changed"]
    )
    result.update(
        {
            "changed": changed,
            "comparable": not protocol_drift,
            "protocol_drift": protocol_drift,
            "candidates_added": candidates_added,
            "candidates_removed": candidates_removed,
            "candidate_drift": candidate_drift,
            "n_cases": n_cases,
        }
    )
    return result


def _valid_cache_profile(payload: Any) -> bool:
    """Structural check for a ``cache_profile`` payload.

    Enforces the wave-2 C3 lesson at the value level: fractions must
    be numbers/None in [0, 1] (None exactly when nothing was
    evaluable), the integer identities the producer guarantees must
    hold (achievable = potential count - first occurrences), ordering
    invariants (block <= token, achievable <= potential) must hold,
    and map keys must be strings, so the renderers never hit a
    TypeError on a hostile payload. The recompute below multiplies
    and divides the counts into float expressions, so every count
    must also be float-convertible (wave-8 F5): an int beyond float
    range is outside the producer domain and fails validation here
    instead of overflowing mid-recompute.
    """
    if type(payload) is not dict or payload.get("available") is not True:
        return False
    block_size = payload.get("block_size")
    if type(block_size) is not int or block_size < 1 or not _float_convertible(block_size):
        return False
    n_observations = payload.get("n_observations")
    if (
        type(n_observations) is not int
        or n_observations < 0
        or not _float_convertible(n_observations)
    ):
        return False
    per_candidate = payload.get("per_candidate")
    if type(per_candidate) is not dict or not all(type(k) is str for k in per_candidate):
        return False
    for entry in per_candidate.values():
        if entry is None:
            continue
        if type(entry) is not dict:
            return False
        for key in ("n", "n_evaluable"):
            value = entry.get(key)
            if type(value) is not int or value < 0 or not _float_convertible(value):
                return False
        if entry["n_evaluable"] > entry["n"]:
            return False
        for key in (
            "total_tokens",
            "cacheable_tokens",
            "shared_prefixes",
            "achievable_tokens",
            "block_cacheable_blocks",
            "distinct_block_prefixes",
            "achievable_blocks",
            "n_duplicate_prompts",
        ):
            value = entry.get(key)
            if type(value) is not int or value < 0 or not _float_convertible(value):
                return False
        if entry["achievable_tokens"] != entry["cacheable_tokens"] - entry["shared_prefixes"]:
            return False
        if (
            entry["achievable_blocks"]
            != entry["block_cacheable_blocks"] - entry["distinct_block_prefixes"]
        ):
            return False
        if entry["cacheable_tokens"] > entry["total_tokens"]:
            return False
        if entry["n_duplicate_prompts"] > entry["n_evaluable"]:
            return False
        for key in (
            "potential_token_fraction",
            "achievable_token_fraction",
            "potential_block_fraction",
            "achievable_block_fraction",
        ):
            value = entry.get(key)
            if entry["n_evaluable"] == 0:
                if value is not None:
                    return False
            elif not _share_in_unit(value):
                return False
        mean = entry.get("mean_prompt_tokens")
        if entry["n_evaluable"] == 0:
            if mean is not None:
                return False
        elif not _positive_number(mean):
            return False
        if entry["n_evaluable"] > 0:
            # Recompute (cycle 177, wave-7 F6b): the mean and all
            # four fractions are the stored counts' ratios under
            # the payload's own block_size -- bit-exact, because the
            # producer computes the same expressions.
            total = entry["total_tokens"]
            if total <= 0:
                return False
            # The block fractions below convert the PRODUCT
            # block_size * blocks; two individually convertible
            # factors can compose past float range, so the products
            # are gated as composed values (wave-8 F5).
            if not _float_convertible(block_size * entry["block_cacheable_blocks"]):
                return False
            if not _float_convertible(block_size * entry["achievable_blocks"]):
                return False
            if mean != total / entry["n_evaluable"]:
                return False
            if entry["potential_token_fraction"] != entry["cacheable_tokens"] / total:
                return False
            if entry["achievable_token_fraction"] != entry["achievable_tokens"] / total:
                return False
            if (
                entry["potential_block_fraction"]
                != block_size * entry["block_cacheable_blocks"] / total
            ):
                return False
            if (
                entry["achievable_block_fraction"]
                != block_size * entry["achievable_blocks"] / total
            ):
                return False
        achievable_token = entry.get("achievable_token_fraction")
        potential_token = entry.get("potential_token_fraction")
        achievable_block = entry.get("achievable_block_fraction")
        potential_block = entry.get("potential_block_fraction")
        # isinstance narrows for mypy where type() in (...) does not;
        # bools were already rejected by the per-key loops above.
        if (
            isinstance(achievable_token, (int, float))
            and isinstance(potential_token, (int, float))
            and isinstance(achievable_block, (int, float))
            and isinstance(potential_block, (int, float))
            and not (
                achievable_block <= achievable_token <= potential_token
                and achievable_block <= potential_block <= potential_token
            )
        ):
            return False
    for key in ("most_sharable", "least_sharable", "excluded_candidates"):
        value = payload.get(key)
        if type(value) is not list or not all(type(c) is str for c in value):
            return False
    gap = payload.get("sharability_gap")
    if gap is not None and not (type(gap) in (int, float) and gap >= 0):
        return False
    if payload.get("uniform") is not None and type(payload.get("uniform")) is not bool:
        return False
    if type(payload.get("evaluable_mismatch")) is not bool:
        return False
    # Profile-level recomputation (cycle 177, wave-7 F6b): the
    # evaluable set, exclusions, gap, extremes, uniformity, and
    # the mismatch flag are all functions of the cells above,
    # recomputed with the producer's own definitions.
    evaluable: dict[str, float] = {}
    excluded: list[str] = []
    evaluable_counts: list[int] = []
    for name, entry in per_candidate.items():
        if entry is None or entry["n_evaluable"] == 0:
            excluded.append(name)
            continue
        evaluable[name] = float(entry["achievable_block_fraction"])
        evaluable_counts.append(entry["n_evaluable"])
    if payload.get("excluded_candidates") != sorted(excluded):
        return False
    if payload.get("evaluable_mismatch") != (len(set(evaluable_counts)) > 1):
        return False
    if not evaluable:
        return (
            gap is None
            and payload.get("uniform") is None
            and not payload["most_sharable"]
            and not payload["least_sharable"]
        )
    values = list(evaluable.values())
    if gap != max(values) - min(values):
        return False
    top = max(values)
    bottom = min(values)
    if payload["most_sharable"] != sorted(n for n, v in evaluable.items() if v == top):
        return False
    if payload["least_sharable"] != sorted(n for n, v in evaluable.items() if v == bottom):
        return False
    fraction_keys = (
        "achievable_block_fraction",
        "achievable_token_fraction",
        "potential_block_fraction",
        "potential_token_fraction",
    )
    first = per_candidate[next(iter(evaluable))]
    expected_uniform = all(
        all(per_candidate[name].get(key) == first.get(key) for name in evaluable)
        for key in fraction_keys
    )
    return payload.get("uniform") == expected_uniform


def diff_cache_profile(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``cache_profile`` payloads; deterministic.

    Answers "did the prefix-sharability story change between runs?":
    per-candidate drift on the headline metric (achievable block
    fraction) plus the token-level and potential diagnostics, and
    added/removed candidates and changes in the excluded set. A
    template edit that silently destroys prefix sharing is the
    regression signal -- prefill a prefix cache could have skipped
    comes back. A side without a recorded probe is reported plainly.
    No model, no network.
    """
    old_ok = _valid_cache_profile(old)
    new_ok = _valid_cache_profile(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    result: dict[str, Any] = {"available": available, "changed": False}
    if available != "both":
        return result

    def _candidate_view(payload: Any) -> dict[str, dict[str, Any]]:
        """Return the candidate view."""
        view: dict[str, dict[str, Any]] = {}
        for name, entry in (payload.get("per_candidate") or {}).items():
            # _valid_cache_profile guarantees str keys and dict/None
            # entries, so no defensive skip is needed here (it would
            # be dead code and an uncoverable line).
            if entry is None:
                view[name] = {
                    "achievable_block_fraction": None,
                    # nosec B105 -- view field default, never a credential
                    "achievable_token_fraction": None,  # nosec B105
                    "potential_block_fraction": None,
                    "excluded": True,
                }
            else:
                view[name] = {
                    "achievable_block_fraction": entry.get("achievable_block_fraction"),
                    "achievable_token_fraction": entry.get("achievable_token_fraction"),
                    "potential_block_fraction": entry.get("potential_block_fraction"),
                    "excluded": name in set(payload.get("excluded_candidates") or []),
                }
        return view

    old_view = _candidate_view(old)
    new_view = _candidate_view(new)
    cache_profile_candidates_added = sorted(set(new_view) - set(old_view))
    cache_profile_candidates_removed = sorted(set(old_view) - set(new_view))
    cache_profile_block_drift: list[dict[str, Any]] = []
    cache_profile_token_drift: list[dict[str, Any]] = []
    cache_profile_potential_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_view) & set(new_view)):
        old_entry = old_view[name]
        new_entry = new_view[name]
        for rate_key, bucket in (
            ("achievable_block_fraction", cache_profile_block_drift),
            ("achievable_token_fraction", cache_profile_token_drift),
            ("potential_block_fraction", cache_profile_potential_drift),
        ):
            if old_entry[rate_key] != new_entry[rate_key]:
                bucket.append(
                    {
                        "candidate": name,
                        "old": old_entry[rate_key],
                        "new": new_entry[rate_key],
                    }
                )
    old_excluded = {name for name, v in old_view.items() if v["excluded"]}
    new_excluded = {name for name, v in new_view.items() if v["excluded"]}
    cache_profile_excluded_added = sorted(new_excluded - old_excluded)
    cache_profile_excluded_removed = sorted(old_excluded - new_excluded)
    result["block_drift"] = cache_profile_block_drift
    result["token_drift"] = cache_profile_token_drift
    result["potential_drift"] = cache_profile_potential_drift
    result["candidates_added"] = cache_profile_candidates_added
    result["candidates_removed"] = cache_profile_candidates_removed
    result["excluded_added"] = cache_profile_excluded_added
    result["excluded_removed"] = cache_profile_excluded_removed
    # NOTE: cache-profile-specific disjunct names keep this ``changed``
    # line textually distinct from every other diff section's anchor
    # (mutation-gate uniqueness).
    result["changed"] = bool(
        cache_profile_block_drift
        or cache_profile_token_drift
        or cache_profile_potential_drift
        or cache_profile_candidates_added
        or cache_profile_candidates_removed
        or cache_profile_excluded_added
        or cache_profile_excluded_removed
    )
    return result


def _valid_confidentiality(payload: Any) -> bool:
    """Structural check for a ``confidentiality`` payload.

    Cells are validated by experiments._confidentiality_cell_valid, and
    every stored gap is recomputed from its paired rates with a float
    tolerance (red-team A R13.7: a stale or inconsistent stored gap is a
    hostile-payload trap). Counts must be consistent with the paired
    denominator, so the renderers never hit a TypeError or a lie.
    """
    from .experiments import _confidentiality_cell_valid

    if type(payload) is not dict or payload.get("available") is not True:
        return False
    protocol = payload.get("protocol")
    if type(protocol) is not dict:
        return False
    if type(protocol.get("fingerprint")) is not str:
        return False
    n_fixtures = payload.get("n_fixtures")
    if type(n_fixtures) is not int or n_fixtures < 1:
        return False
    per_candidate = payload.get("per_candidate")
    if type(per_candidate) is not dict or not all(type(k) is str for k in per_candidate):
        return False
    for entry in per_candidate.values():
        if not _confidentiality_cell_valid(entry):
            return False
        if type(entry) is dict:
            if entry.get("n") != n_fixtures:
                return False
            paired = entry["n_paired_evaluable"]
            if entry["treat_leaks"] > paired or entry["ctrl_leaks"] > paired:
                return False
            if entry["treat_verbatim"] > entry["treat_leaks"]:
                return False
            if entry["ctrl_verbatim"] > entry["ctrl_leaks"]:
                return False
            if (
                entry["discord_ctrl_only"] + entry["discord_treat_only"]
                > entry["treat_leaks"] + entry["ctrl_leaks"]
            ):
                return False
            if entry["gap"] is not None:
                expected = entry["ctrl_rate"] - entry["treat_rate"]
                if abs(float(entry["gap"]) - expected) > 1e-9:
                    return False
            # Generator invariant: gap is present exactly when the
            # candidate is evaluable AND the control rate is nonzero.
            if entry["gap"] is None:
                if entry["evaluable"] and not entry["base_rate_zero"]:
                    return False
            elif entry["base_rate_zero"] or not entry["evaluable"]:
                return False
    for key in ("uniform_nonpositive", "any_positive"):
        if type(payload.get(key)) is not bool:
            return False
    positive = payload.get("positive_candidates")
    if type(positive) is not list or not all(type(x) is str for x in positive):
        return False
    for key in ("most_resistant", "least_resistant"):
        value = payload.get(key)
        if value is not None and type(value) is not str:
            return False
    gap = payload.get("gap")
    if gap is not None and not (
        type(gap) in (int, float) and isinstance(gap, (int, float)) and -1.0 <= gap <= 1.0
    ):
        return False
    # Profile-level recomputation (cycle 177, wave-7 F6c): the
    # counts, extremes, spread, uniformity, and positive set are
    # functions of the cells, recomputed with the producer's own
    # definitions (measurable = dict cell with a gap; extremes
    # break ties by name, max-side last, min-side first).
    dict_cells = {name: e for name, e in per_candidate.items() if type(e) is dict}
    if payload.get("n_candidates") != len(per_candidate):
        return False
    if payload.get("n_cells") != len(dict_cells):
        return False
    measurable = {name: e for name, e in dict_cells.items() if e.get("gap") is not None}
    if payload.get("n_measurable") != len(measurable):
        return False
    if not measurable:
        # An available profile always has a measurable cell; the
        # producer reports unavailable otherwise.
        return False
    gaps = {name: float(e["gap"]) for name, e in measurable.items()}
    most = max(gaps, key=lambda name: (gaps[name], name))
    least = min(gaps, key=lambda name: (gaps[name], name))
    if payload.get("most_resistant") != most:
        return False
    if payload.get("least_resistant") != least:
        return False
    if gap != gaps[most] - gaps[least]:
        return False
    if payload.get("uniform_nonpositive") != all(g <= 0 for g in gaps.values()):
        return False
    expected_positive = sorted(name for name, e in measurable.items() if e.get("positive") is True)
    if positive != expected_positive:
        return False
    return payload.get("any_positive") == bool(expected_positive)


def diff_confidentiality(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``confidentiality`` payloads; deterministic.

    Public name retained for API compatibility (callers outside the
    diff pipeline, e.g. the cycle-169 tests, use it). The pipeline
    itself calls :func:`_diff_labeled_block` and carries the result
    as data, by key -- see :func:`_payload_section` for why.
    """
    return _diff_labeled_block(old, new)


def _valid_answer_collision(payload: Any) -> bool:
    """Structural check for an ``answer_collision`` payload.

    Cells are validated by experiments._answer_collision_cell_valid,
    every stored rate is recomputed from its counts with a float
    tolerance, and the profile-level fields are recomputed from the
    measurable cells -- a stale or inconsistent stored value is a
    hostile-payload trap, never rendered. Counts must be consistent
    with the fixture denominator, so the renderers never hit a
    TypeError or a lie.
    """
    from .experiments import _answer_collision_cell_valid

    if type(payload) is not dict or payload.get("available") is not True:
        return False
    protocol = payload.get("protocol")
    if type(protocol) is not dict:
        return False
    if type(protocol.get("fingerprint")) is not str:
        return False
    n_fixtures = payload.get("n_fixtures")
    if type(n_fixtures) is not int or n_fixtures < 1:
        return False
    per_candidate = payload.get("per_candidate")
    if type(per_candidate) is not dict or not all(type(k) is str for k in per_candidate):
        return False
    for entry in per_candidate.values():
        if not _answer_collision_cell_valid(entry):
            return False
        if type(entry) is dict and entry.get("n") != n_fixtures:
            return False
    if payload.get("n_candidates") != len(per_candidate):
        return False
    cells = {k: v for k, v in per_candidate.items() if type(v) is dict}
    if payload.get("n_cells") != len(cells):
        return False
    measurable = _measurable_collision_cells(per_candidate)
    if payload.get("n_measurable") != len(measurable):
        return False
    if type(payload.get("any_collision")) is not bool:
        return False
    if type(payload.get("uniform_intact")) is not bool:
        return False
    colliding = payload.get("colliding_candidates")
    if type(colliding) is not list or not all(type(x) is str for x in colliding):
        return False
    expected_colliding = sorted(
        name for name, cell in measurable.items() if cell.get("any_collision") is True
    )
    if colliding != expected_colliding:
        return False
    if payload["any_collision"] != bool(expected_colliding):
        return False
    if payload["uniform_intact"] != (
        bool(measurable) and all(cell.get("uniform_intact") is True for cell in measurable.values())
    ):
        return False
    rates = {
        name: cell["reserved"]["collision_rate"]
        for name, cell in measurable.items()
        if type(cell.get("reserved")) is dict
    }
    max_rate = payload.get("max_collision_rate")
    if not rates:
        return max_rate is None and payload.get("worst_candidate") is None
    if type(max_rate) is int and not _float_convertible(max_rate):
        # The tolerance comparison below converts max_rate to float;
        # an int beyond float range is a magnitude no producer emits
        # (the recomputed rates are shares in [0, 1]) -- invalid,
        # never an OverflowError (wave-8 F5).
        return False
    if (
        max_rate is None
        or not (type(max_rate) in (int, float) and isinstance(max_rate, (int, float)))
        or abs(float(max_rate) - max(float(r) for r in rates.values())) > 1e-9
    ):
        return False
    worst = max(rates, key=lambda name: (float(rates[name]), name))
    return payload.get("worst_candidate") == worst


def diff_answer_collision(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``answer_collision`` payloads; deterministic.

    Answers "did the detector-collision story change between runs?":
    protocol-fingerprint drift, per-configuration reserved-arm
    collision-rate drift, the colliding set, and the uniform-intact
    (regression-guard) flag. A tokenizer or boundary edit that starts
    truncating marker-bearing correct answers is the regression signal.
    A side without a recorded audit is reported plainly. No model, no
    network.
    """
    old_ok = _valid_answer_collision(old)
    new_ok = _valid_answer_collision(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    result: dict[str, Any] = {"available": available, "changed": False}
    if available != "both":
        return result
    protocol_drift = old["protocol"].get("fingerprint") != new["protocol"].get("fingerprint")
    old_cells = old.get("per_candidate") or {}
    new_cells = new.get("per_candidate") or {}
    answer_collision_candidates_added = sorted(set(new_cells) - set(old_cells))
    answer_collision_candidates_removed = sorted(set(old_cells) - set(new_cells))
    answer_collision_rate_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_cells) & set(new_cells)):
        old_entry = old_cells[name]
        new_entry = new_cells[name]
        old_rate = (
            old_entry["reserved"].get("collision_rate")
            if type(old_entry) is dict and type(old_entry.get("reserved")) is dict
            else None
        )
        new_rate = (
            new_entry["reserved"].get("collision_rate")
            if type(new_entry) is dict and type(new_entry.get("reserved")) is dict
            else None
        )
        if old_rate != new_rate:
            answer_collision_rate_drift.append(
                {"candidate": name, "old": old_rate, "new": new_rate}
            )
    answer_collision_colliding_added = sorted(
        set(new.get("colliding_candidates") or []) - set(old.get("colliding_candidates") or [])
    )
    answer_collision_colliding_removed = sorted(
        set(old.get("colliding_candidates") or []) - set(new.get("colliding_candidates") or [])
    )
    answer_collision_uniform_changed = old.get("uniform_intact") != new.get("uniform_intact")
    result["protocol_drift"] = protocol_drift
    result["rate_drift"] = answer_collision_rate_drift
    result["candidates_added"] = answer_collision_candidates_added
    result["candidates_removed"] = answer_collision_candidates_removed
    result["colliding_added"] = answer_collision_colliding_added
    result["colliding_removed"] = answer_collision_colliding_removed
    result["uniform_changed"] = answer_collision_uniform_changed
    # NOTE: answer-collision-specific disjunct names keep this
    # ``changed`` line textually distinct from every other diff
    # section's anchor (mutation-gate uniqueness).
    result["changed"] = bool(
        protocol_drift
        or answer_collision_rate_drift
        or answer_collision_candidates_added
        or answer_collision_candidates_removed
        or answer_collision_colliding_added
        or answer_collision_colliding_removed
        or answer_collision_uniform_changed
    )
    return result


def _valid_template_robustness(payload: Any) -> bool:
    """Structural check for a ``template_robustness`` payload.

    Preservation rates must be numbers/None, counts ints, candidate keys
    strings, so the renderers never hit a TypeError on a hostile payload.
    """

    def _ok_rate(value: Any) -> bool:
        """Return the ok rate."""
        return value is None or (type(value) in (int, float) and 0.0 <= value <= 1.0)

    def _ok_report(rep: Any) -> bool:
        """Return the ok report."""
        if type(rep) is not dict:
            return False
        if rep.get("status") != "ok":
            return True  # unavailable markers pass structurally
        if not _ok_rate(rep.get("preservation_rate")):
            return False
        for key in ("n_preserved", "n_preservation_denominator", "n_unevaluable"):
            if type(rep.get(key)) is not int:
                return False
        operators = rep.get("operators")
        if operators is not None:
            if type(operators) is not dict:
                return False
            for op in ("case_flip", "whitespace", "unicode_variant"):
                op_entry = operators.get(op)
                if op_entry is not None and type(op_entry) is not dict:
                    return False
        return True

    if type(payload) is not dict or payload.get("available") is not True:
        return False
    candidates = payload.get("candidates")
    if type(candidates) is not dict or not candidates:
        return False
    return all(type(k) is str and _ok_report(v) for k, v in candidates.items())


def diff_template_robustness(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``template_robustness`` payloads; deterministic.

    Answers "did the robustness story change between runs?": per-candidate
    preservation-rate drift, per-operator preservation drift, plus
    added/removed candidates. A newly dropped preservation rate after a
    template change is the regression signal -- the template losing
    correct verdicts to sloppy input. A side without a recorded probe is
    reported plainly. No model, no network.
    """
    old_ok = _valid_template_robustness(old)
    new_ok = _valid_template_robustness(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    result: dict[str, Any] = {"available": available, "changed": False}
    if available != "both":
        return result

    def _candidate_view(payload: Any) -> dict[str, dict[str, Any]]:
        """Return the candidate view."""
        view: dict[str, dict[str, Any]] = {}
        for name, entry in payload["candidates"].items():
            # _valid_template_robustness guarantees str keys and dict
            # entries, so no defensive skip is needed here (it would be
            # dead code and an uncoverable line).
            if entry.get("status") != "ok":
                view[name] = {"preservation_rate": None, "operators": {}}
            else:
                ops = entry.get("operators") or {}
                view[name] = {
                    "preservation_rate": entry.get("preservation_rate"),
                    "operators": {
                        op: (ops.get(op) or {}).get("preservation_rate")
                        for op in ("case_flip", "whitespace", "unicode_variant")
                    },
                }
        return view

    old_view = _candidate_view(old)
    new_view = _candidate_view(new)
    candidates_added = sorted(set(new_view) - set(old_view))
    candidates_removed = sorted(set(old_view) - set(new_view))
    preservation_drift: list[dict[str, Any]] = []
    operator_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_view) & set(new_view)):
        old_entry = old_view[name]
        new_entry = new_view[name]
        if old_entry["preservation_rate"] != new_entry["preservation_rate"]:
            preservation_drift.append(
                {
                    "candidate": name,
                    "old": old_entry["preservation_rate"],
                    "new": new_entry["preservation_rate"],
                }
            )
        for op in ("case_flip", "whitespace", "unicode_variant"):
            if old_entry["operators"].get(op) != new_entry["operators"].get(op):
                operator_drift.append(
                    {
                        "candidate": name,
                        "operator": op,
                        "old": old_entry["operators"].get(op),
                        "new": new_entry["operators"].get(op),
                    }
                )
    result["preservation_drift"] = preservation_drift
    result["operator_drift"] = operator_drift
    result["candidates_added"] = candidates_added
    result["candidates_removed"] = candidates_removed
    # NOTE: template-robustness-specific disjunct names keep this
    # ``changed`` line textually distinct from every other diff section's
    # anchor (mutation-gate uniqueness).
    result["changed"] = bool(
        preservation_drift or operator_drift or candidates_added or candidates_removed
    )
    return result


def _valid_ablation(payload: Any) -> bool:
    """Structural check for an ``ablation`` payload.

    Flip counts must be ints, load-bearing lists string lists, candidate
    keys strings, so the renderers never hit a TypeError on a hostile
    payload (the wave-2 C3 lesson at the value level). Derived values
    are recomputed (cycle 176, wave-7 F6d; the answer-collision
    validator's standard): flip counts are non-negative and satisfy
    n_flips = pass_to_fail + fail_to_pass = len(flipped_fixture_ids)
    with string ids (an unchecked id list is exactly how a hostile
    payload reached a renderer join), and load_bearing_fields is the
    producer's definition recomputed -- base passes all, arm does
    not hold the selection bar.
    """
    if type(payload) is not dict or payload.get("available") is not True:
        return False
    candidates = payload.get("candidates")
    if type(candidates) is not dict or not candidates:
        return False
    if not all(type(name) is str for name in candidates):
        return False
    for rep in candidates.values():
        if type(rep) is not dict:
            return False
        if rep.get("status") != "ok":
            continue  # unavailable markers pass structurally
        if type(rep.get("base_passes_all")) is not bool:
            return False
        arms = rep.get("arms")
        if type(arms) is not dict:
            return False
        for arm in arms.values():
            if type(arm) is not dict:
                return False
            if arm.get("status") != "ok":
                continue
            for key in ("n_flips", "pass_to_fail", "fail_to_pass"):
                value = arm.get(key)
                if type(value) is not int or value < 0:
                    return False
            if arm["pass_to_fail"] + arm["fail_to_pass"] != arm["n_flips"]:
                return False
            flipped = arm.get("flipped_fixture_ids")
            if (
                type(flipped) is not list
                or not all(type(fid) is str for fid in flipped)
                or len(flipped) != arm["n_flips"]
            ):
                return False
            n_fixtures = arm.get("n_fixtures")
            if type(n_fixtures) is not int or n_fixtures < arm["n_flips"]:
                return False
            if type(arm.get("selection_bar_held")) is not bool:
                return False
        load_bearing = rep.get("load_bearing_fields")
        if type(load_bearing) is not list or not all(type(f) is str for f in load_bearing):
            return False
        expected_load_bearing = [
            field
            for field, arm in arms.items()
            if type(arm) is dict
            and arm.get("status") == "ok"
            and rep["base_passes_all"]
            and arm.get("selection_bar_held") is False
        ]
        if load_bearing != expected_load_bearing:
            return False
        any_attribution = rep.get("any_attribution")
        if type(any_attribution) is not bool or any_attribution != any(
            type(arm) is dict and arm.get("status") == "ok" and arm.get("n_flips", 0) > 0
            for arm in arms.values()
        ):
            return False
    return True


def diff_ablation(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``ablation`` payloads; deterministic.

    Answers "did the causal story change between runs?": per-candidate,
    per-field flip-count drift, load-bearing-field set drift, plus
    added/removed candidates. A field that becomes load-bearing after a
    template edit is the regression signal -- the edit made the verdict
    depend on that field. A side without a recorded probe is reported
    plainly. No model, no network.
    """
    old_ok = _valid_ablation(old)
    new_ok = _valid_ablation(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    result: dict[str, Any] = {"available": available, "changed": False}
    if available != "both":
        return result

    def _candidate_view(payload: Any) -> dict[str, dict[str, Any]]:
        """Return the candidate view."""
        view: dict[str, dict[str, Any]] = {}
        for name, entry in payload["candidates"].items():
            # _valid_ablation guarantees str keys and dict entries, so no
            # defensive skip is needed here (it would be dead code and an
            # uncoverable line).
            if entry.get("status") != "ok":
                view[name] = {"flips": {}, "load_bearing": []}
            else:
                arms = entry.get("arms") or {}
                view[name] = {
                    "flips": {
                        field: (arm.get("n_flips") if type(arm) is dict else None)
                        for field, arm in arms.items()
                    },
                    "load_bearing": sorted(entry.get("load_bearing_fields") or []),
                }
        return view

    old_view = _candidate_view(old)
    new_view = _candidate_view(new)
    candidates_added = sorted(set(new_view) - set(old_view))
    candidates_removed = sorted(set(old_view) - set(new_view))
    attribution_drift: list[dict[str, Any]] = []
    load_bearing_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_view) & set(new_view)):
        old_entry = old_view[name]
        new_entry = new_view[name]
        for field in sorted(set(old_entry["flips"]) | set(new_entry["flips"])):
            old_flips = old_entry["flips"].get(field)
            new_flips = new_entry["flips"].get(field)
            if old_flips != new_flips:
                attribution_drift.append(
                    {
                        "candidate": name,
                        "field": field,
                        "old_n_flips": old_flips,
                        "new_n_flips": new_flips,
                    }
                )
        if old_entry["load_bearing"] != new_entry["load_bearing"]:
            load_bearing_drift.append(
                {
                    "candidate": name,
                    "old_load_bearing": old_entry["load_bearing"],
                    "new_load_bearing": new_entry["load_bearing"],
                }
            )
    result["attribution_drift"] = attribution_drift
    result["load_bearing_drift"] = load_bearing_drift
    result["candidates_added"] = candidates_added
    result["candidates_removed"] = candidates_removed
    # NOTE: ablation-specific disjunct names keep this ``changed`` line
    # textually distinct from every other diff section's anchor
    # (mutation-gate uniqueness).
    result["changed"] = bool(
        attribution_drift or load_bearing_drift or candidates_added or candidates_removed
    )
    return result


def _valid_portfolio(payload: Any) -> bool:
    """Light validation: is this a comparable ``portfolio`` payload?

    Structural checks only (key types and value shapes); a malformed
    side degrades to asymmetric instead of raising, per the module's
    degrade-gracefully contract. Mixed-type fixture/member keys would
    otherwise TypeError inside the set comparisons below.
    """
    if type(payload) is not dict or payload.get("available") is not True:
        return False
    if type(payload.get("n_fixtures")) is not int or payload["n_fixtures"] <= 0:
        return False
    oracle = payload.get("oracle")
    if type(oracle) is not dict:
        return False
    for key in ("gap", "oracle_pass_rate"):
        if type(oracle.get(key)) not in (int, float):
            return False
    winner_rate = oracle.get("winner_pass_rate")
    if winner_rate is not None and type(winner_rate) not in (int, float):
        return False
    solvable = oracle.get("solvable_fixture_ids")
    if type(solvable) is not list or not all(type(fid) is str for fid in solvable):
        return False
    winners = oracle.get("per_fixture_winners")
    if type(winners) is not dict or not all(type(key) is str for key in winners):
        return False
    for names in winners.values():
        if type(names) is not list or not all(type(name) is str for name in names):
            return False
    cover = payload.get("portfolio")
    if type(cover) is not dict:
        return False
    members = cover.get("members")
    if type(members) is not list or not all(type(member) is str for member in members):
        return False
    per_member = cover.get("per_member_covers")
    if type(per_member) is not dict or not all(type(key) is str for key in per_member):
        return False
    for fixtures in per_member.values():
        if type(fixtures) is not list or not all(type(fid) is str for fid in fixtures):
            return False
    return True


def diff_portfolio(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``portfolio`` payloads; deterministic.

    Answers "did the oracle-mixture story change between runs?": the
    oracle gap and rates, the solvable-fixture set, the greedy
    portfolio membership, the per-fixture oracle-winner sets, and each
    member's covered fixtures. A shrinking gap after a checkpoint or
    fixture update means the protocols are converging; a growing gap
    means the winner is increasingly specialized. Members present on
    only one side are reported as added/removed. A side without a
    recorded portfolio analysis is reported plainly. No model, no
    network.
    """
    old_ok = _valid_portfolio(old)
    new_ok = _valid_portfolio(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    result: dict[str, Any] = {"available": available, "changed": False}
    if available != "both":
        return result

    def _rate_triplet(payload: Any) -> tuple[Any, Any, Any]:
        """Return the rate triplet."""
        oracle = payload.get("oracle") or {}
        return (
            oracle.get("gap"),
            oracle.get("oracle_pass_rate"),
            oracle.get("winner_pass_rate"),
        )

    old_triplet = _rate_triplet(old)
    new_triplet = _rate_triplet(new)
    rates_shifted = old_triplet != new_triplet
    result["rates"] = {
        "old": {
            "gap": old_triplet[0],
            "oracle_pass_rate": old_triplet[1],
            "winner_pass_rate": old_triplet[2],
        },
        "new": {
            "gap": new_triplet[0],
            "oracle_pass_rate": new_triplet[1],
            "winner_pass_rate": new_triplet[2],
        },
        "changed": rates_shifted,
    }

    def _solvable(payload: Any) -> set[str]:
        """Return the solvable."""
        oracle = payload.get("oracle") or {}
        return set(oracle.get("solvable_fixture_ids") or [])

    old_solvable = _solvable(old)
    new_solvable = _solvable(new)
    solvable_added = sorted(new_solvable - old_solvable)
    solvable_removed = sorted(old_solvable - new_solvable)
    result["solvable_added"] = solvable_added
    result["solvable_removed"] = solvable_removed

    def _members(payload: Any) -> list[str]:
        """Return the members."""
        cover = payload.get("portfolio") or {}
        return list(cover.get("members") or [])

    old_members = _members(old)
    new_members = _members(new)
    members_added = [member for member in new_members if member not in old_members]
    members_removed = [member for member in old_members if member not in new_members]
    result["members_added"] = members_added
    result["members_removed"] = members_removed

    def _fixture_winners(payload: Any) -> dict[str, list[str]]:
        """Return the fixture winners."""
        oracle = payload.get("oracle") or {}
        winners = oracle.get("per_fixture_winners") or {}
        return {str(key): list(names) for key, names in winners.items()}

    old_winners = _fixture_winners(old)
    new_winners = _fixture_winners(new)
    fixture_winner_drift: list[dict[str, Any]] = []
    for fixture_id in sorted(set(old_winners) & set(new_winners)):
        if old_winners[fixture_id] != new_winners[fixture_id]:
            fixture_winner_drift.append(
                {
                    "fixture": fixture_id,
                    "old_winners": old_winners[fixture_id],
                    "new_winners": new_winners[fixture_id],
                }
            )
    result["fixture_winner_drift"] = fixture_winner_drift

    def _covers(payload: Any) -> dict[str, list[str]]:
        """Return the covers."""
        cover = payload.get("portfolio") or {}
        per_member = cover.get("per_member_covers") or {}
        return {str(key): sorted(fixtures) for key, fixtures in per_member.items()}

    old_covers = _covers(old)
    new_covers = _covers(new)
    cover_drift: list[dict[str, Any]] = []
    for member in sorted(set(old_covers) & set(new_covers)):
        if old_covers[member] != new_covers[member]:
            cover_drift.append(
                {
                    "member": member,
                    "old_covers": old_covers[member],
                    "new_covers": new_covers[member],
                }
            )
    result["cover_drift"] = cover_drift

    # NOTE: phrased with portfolio-specific names so this line stays
    # textually distinct from every other section's changed anchor
    # (mutation-gate uniqueness).
    result["changed"] = bool(
        rates_shifted
        or solvable_added
        or solvable_removed
        or members_added
        or members_removed
        or fixture_winner_drift
        or cover_drift
    )
    return result


def _valid_synthhistory(payload: Any) -> bool:
    """Structural check for a ``synthhistory`` payload.

    Enforces the wave-2 C3 lesson at the value level: the compared
    fields must be bools/strings/None/numbers so the renderers never hit
    a TypeError on a hostile payload, and map keys must be strings. A
    malformed side degrades to asymmetric instead of raising.
    """
    if type(payload) is not dict or payload.get("available") is not True:
        return False
    if type(payload.get("primary_decision")) is not str:
        return False
    winner = payload.get("primary_winner")
    if winner is not None and type(winner) is not str:
        return False
    if type(payload.get("n_fixtures")) is not int or payload["n_fixtures"] <= 0:
        return False
    conditions = payload.get("conditions")
    if (
        type(conditions) is not list
        or not conditions
        or not all(type(c) is str for c in conditions)
    ):
        return False
    per_condition = payload.get("per_condition")
    if type(per_condition) is not dict or set(per_condition) != set(conditions):
        return False
    for condition in per_condition.values():
        if type(condition) is not dict:
            return False
        if type(condition.get("decision")) is not str:
            return False
        cond_winner = condition.get("winner")
        if cond_winner is not None and type(cond_winner) is not str:
            return False
        if type(condition.get("n_evaluable")) is not int or condition["n_evaluable"] < 0:
            return False
        candidates = condition.get("per_candidate")
        if type(candidates) is not dict or not all(type(k) is str for k in candidates):
            return False
        for entry in candidates.values():
            if entry is None:
                continue
            if type(entry) is not dict:
                return False
            rate = entry.get("pass_rate")
            if rate is not None and type(rate) not in (int, float):
                return False
            if type(entry.get("pass")) is not bool:
                return False
    flips = payload.get("flips")
    if type(flips) is not list:
        return False
    for entry in flips:
        if type(entry) is not dict:
            return False
        if type(entry.get("condition")) is not str:
            return False
        if type(entry.get("decision_from")) is not str or type(entry.get("decision_to")) is not str:
            return False
        for key in ("winner_from", "winner_to"):
            if entry.get(key) is not None and type(entry.get(key)) is not str:
                return False
    groups = payload.get("group_pass_rates")
    if type(groups) is not dict:
        return False
    for group in groups.values():
        if type(group) is not dict:
            return False
        if type(group.get("n_candidates")) is not int or group["n_candidates"] < 0:
            return False
        mean = group.get("mean_pass_rate")
        if mean is not None and type(mean) not in (int, float):
            return False
    return True


def diff_synthhistory(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``synthhistory`` payloads; deterministic.

    Answers "did the synthetic-history stability story change between
    runs?": the per-condition (decision, winner) pairs, the winner-flip
    pattern against the primary run, per-candidate history pass-rate
    drift, and the explicit-vs-loose delimiter group-rate comparison. A
    changed flip pattern after a checkpoint or fixture update is the
    history-robustness regression signal. Candidates present on only
    one side are reported as added/removed. A side without a recorded
    synthetic-history experiment is reported plainly. No model, no
    network.
    """
    old_ok = _valid_synthhistory(old)
    new_ok = _valid_synthhistory(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    result: dict[str, Any] = {"available": available, "changed": False}
    if available != "both":
        return result
    old_conditions = old.get("conditions") or []
    new_conditions = new.get("conditions") or []
    conditions_added = [c for c in new_conditions if c not in old_conditions]
    conditions_removed = [c for c in old_conditions if c not in new_conditions]
    condition_changed: dict[str, dict[str, Any]] = {}
    for condition in sorted(set(old_conditions) & set(new_conditions)):
        old_verdict_pair = (old.get("per_condition") or {}).get(condition) or {}
        new_verdict_pair = (new.get("per_condition") or {}).get(condition) or {}
        old_verdict = (old_verdict_pair.get("decision"), old_verdict_pair.get("winner"))
        new_verdict = (new_verdict_pair.get("decision"), new_verdict_pair.get("winner"))
        changed = old_verdict != new_verdict
        condition_changed[condition] = {
            "old_decision": old_verdict[0],
            "old_winner": old_verdict[1],
            "new_decision": new_verdict[0],
            "new_winner": new_verdict[1],
            "changed": changed,
        }

    def _history_flip_key(flips: list[Any]) -> list[tuple[Any, ...]]:
        # ``flips`` entries are all dicts here: _valid_synthhistory rejects
        # anything else before this point, so no defensive branch is needed.
        """Return the history flip key."""
        keys = []
        for entry in flips:
            keys.append(
                (
                    entry.get("condition"),
                    entry.get("decision_from"),
                    entry.get("winner_from"),
                    entry.get("decision_to"),
                    entry.get("winner_to"),
                )
            )
        return sorted(keys, key=lambda key: tuple(str(part) for part in key))

    flips_changed = _history_flip_key(old.get("flips")) != _history_flip_key(new.get("flips"))

    def _candidate_pass_rates(payload: Any) -> dict[str, dict[str, Any]]:
        """Return the candidate pass rates."""
        rates: dict[str, dict[str, Any]] = {}
        per_condition = payload.get("per_condition") or {}
        for condition in sorted(per_condition):
            candidates = (per_condition.get(condition) or {}).get("per_candidate") or {}
            for name, entry in candidates.items():
                entry = entry or {}
                rates.setdefault(name, {})[condition] = entry.get("pass_rate")
        return rates

    old_rates = _candidate_pass_rates(old)
    new_rates = _candidate_pass_rates(new)
    candidates_added = sorted(set(new_rates) - set(old_rates))
    candidates_removed = sorted(set(old_rates) - set(new_rates))
    pass_rate_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_rates) & set(new_rates)):
        for condition in sorted(set(old_rates[name]) & set(new_rates[name])):
            old_rate = old_rates[name][condition]
            new_rate = new_rates[name][condition]
            if old_rate != new_rate:
                pass_rate_drift.append(
                    {
                        "candidate": name,
                        "condition": condition,
                        "old_pass_rate": old_rate,
                        "new_pass_rate": new_rate,
                    }
                )

    def _group_means(payload: Any) -> dict[str, Any]:
        """Return the group means."""
        groups = payload.get("group_pass_rates") or {}
        return {
            label: (group or {}).get("mean_pass_rate")
            for label, group in groups.items()
            if type(label) is str
        }

    old_groups = _group_means(old)
    new_groups = _group_means(new)
    group_changed = new_groups != old_groups  # history: swapped operand order
    result["conditions_added"] = conditions_added
    result["conditions_removed"] = conditions_removed
    result["condition_changed"] = condition_changed
    result["flips_changed"] = flips_changed
    result["pass_rate_drift"] = pass_rate_drift
    result["candidates_added"] = candidates_added
    result["candidates_removed"] = candidates_removed
    result["group_rates"] = {
        "old": old_groups,
        "new": new_groups,
        "changed": group_changed,
    }
    # NOTE: history-specific names keep this line textually distinct from
    # every other section's changed anchor (mutation-gate uniqueness).
    result["changed"] = bool(
        conditions_added
        or conditions_removed
        or any(entry["changed"] for entry in condition_changed.values())
        or flips_changed
        or pass_rate_drift
        or candidates_added
        or candidates_removed
        or group_changed
    )
    return result


def _valid_batchinv(payload: Any) -> bool:
    """Structural check for a ``batch_invariance`` payload.

    Enforces the wave-2 C3 lesson at the value level: the compared
    fields must be bools/strings/None/numbers so the renderers never
    hit a TypeError on a hostile payload, and map keys must be
    strings. A malformed side degrades to asymmetric instead of
    raising.
    """
    if type(payload) is not dict or payload.get("available") is not True:
        return False
    for key in ("n_prompts", "n_evaluable", "n_failed", "n_diverged"):
        if type(payload.get(key)) is not int or payload[key] < 0:
            return False
    if payload["n_evaluable"] + payload["n_failed"] != payload["n_prompts"]:
        return False
    if payload["n_diverged"] > payload["n_evaluable"]:
        return False
    rate = payload.get("divergence_rate")
    if payload["n_evaluable"]:
        if type(rate) not in (int, float):
            return False
    elif rate is not None:
        return False
    if payload.get("verdict") not in (
        "batch_invariant",
        "batch_variant",
        "unavailable",
    ):
        return False
    per_prompt = payload.get("per_prompt")
    if type(per_prompt) is not dict or not all(type(k) is str for k in per_prompt):
        return False
    for entry in per_prompt.values():
        if type(entry) is not dict:
            return False
        diverged = entry.get("diverged")
        if diverged is not None and type(diverged) is not bool:
            return False
        first = entry.get("first_divergence")
        if first is not None and (type(first) is not int or first < 0):
            return False
        for length_key in ("solo_len", "batched_len"):
            length = entry.get(length_key)
            if length is not None and (type(length) is not int or length < 0):
                return False
        error = entry.get("error")
        if error is not None and type(error) is not str:
            return False
    return True


def diff_batchinv(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``batch_invariance`` payloads; deterministic.

    Answers "did the batch-invariance story change between runs?": the
    audit verdict (invariant vs variant), the divergence-rate drift,
    and the diverged-prompt sets. A newly diverged prompt after a
    backend or fixture update is the batch-nondeterminism regression
    signal. A side without a recorded batch-invariance audit is
    reported plainly. No model, no network.
    """
    old_ok = _valid_batchinv(old)
    new_ok = _valid_batchinv(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    result: dict[str, Any] = {"available": available, "changed": False}
    if available != "both":
        return result

    def _diverged_set(payload: Any) -> set[str]:
        """Return the diverged set."""
        per_prompt = payload.get("per_prompt") or {}
        return {
            pid
            for pid, entry in per_prompt.items()
            if type(entry) is dict and entry.get("diverged") is True
        }

    verdict_changed = old.get("verdict") != new.get("verdict")
    divergence_rate_changed = old.get("divergence_rate") != new.get("divergence_rate")
    old_diverged = _diverged_set(old)
    new_diverged = _diverged_set(new)
    diverged_added = sorted(new_diverged - old_diverged)
    diverged_removed = sorted(old_diverged - new_diverged)
    old_prompts = set(old.get("per_prompt") or {})
    new_prompts = set(new.get("per_prompt") or {})
    prompts_added = sorted(new_prompts - old_prompts)
    prompts_removed = sorted(old_prompts - new_prompts)
    result["verdict_changed"] = verdict_changed
    result["divergence_rate_changed"] = divergence_rate_changed
    result["diverged_added"] = diverged_added
    result["diverged_removed"] = diverged_removed
    result["prompts_added"] = prompts_added
    result["prompts_removed"] = prompts_removed
    # NOTE: batchinv-specific names keep this line textually distinct
    # from every other section's changed anchor (mutation-gate
    # uniqueness).
    result["changed"] = bool(
        verdict_changed
        or divergence_rate_changed
        or diverged_added
        or diverged_removed
        or prompts_added
        or prompts_removed
    )
    return result


def _healing_verdict_triplet(payload: Any) -> tuple[Any, Any, Any]:
    """(flipped, healed_decision, healed_winner) for drift comparison."""
    if type(payload) is not dict:
        return (None, None, None)
    return (
        payload.get("flipped"),
        payload.get("healed_decision"),
        payload.get("healed_winner"),
    )


def diff_token_healing(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``token_healing`` payloads; deterministic.

    Answers "did the token-healing sensitivity story change between
    runs?": the flip verdict (did healing change the winner?), the
    healed (decision, winner), per-candidate (native, healed) verdicts,
    and the falsification accuracy (how many candidates the
    boundary-audit prediction held for). A changed flip pattern after a
    checkpoint or fixture update is the tokenization-fragility
    regression signal. Candidates present on only one side are reported
    as added/removed. A side without a recorded healing experiment is
    reported plainly. No model, no network.
    """
    old_ok = _valid_token_healing(old)
    new_ok = _valid_token_healing(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    result: dict[str, Any] = {"available": available, "changed": False}
    if available != "both":
        return result
    old_flip = _healing_verdict_triplet(old)
    new_flip = _healing_verdict_triplet(new)
    flip_changed = old_flip != new_flip
    result["flip"] = {
        "old": {
            "flipped": old_flip[0],
            "healed_decision": old_flip[1],
            "healed_winner": old_flip[2],
        },
        "new": {
            "flipped": new_flip[0],
            "healed_decision": new_flip[1],
            "healed_winner": new_flip[2],
        },
        "changed": flip_changed,
    }
    old_pc = old.get("per_candidate") or {}
    new_pc = new.get("per_candidate") or {}
    old_names = sorted(old_pc)
    new_names = sorted(new_pc)
    candidates_added = [c for c in new_names if c not in old_pc]
    candidates_removed = [c for c in old_names if c not in new_pc]
    candidate_drift: list[dict[str, Any]] = []
    for cand in sorted(set(old_names) & set(new_names)):
        old_e = old_pc.get(cand) or {}
        new_e = new_pc.get(cand) or {}
        old_pair = (old_e.get("native"), old_e.get("healed"))
        new_pair = (new_e.get("native"), new_e.get("healed"))
        if old_pair != new_pair:
            candidate_drift.append(
                {
                    "candidate": cand,
                    "old_native": old_pair[0],
                    "old_healed": old_pair[1],
                    "new_native": new_pair[0],
                    "new_healed": new_pair[1],
                }
            )
    old_fals = old.get("falsification") or {}
    new_fals = new.get("falsification") or {}
    old_acc = (old_fals.get("n_holds"), old_fals.get("n_comparable"))
    new_acc = (new_fals.get("n_holds"), new_fals.get("n_comparable"))
    falsification_changed = old_acc != new_acc
    result["falsification"] = {
        "old": {"n_holds": old_acc[0], "n_comparable": old_acc[1]},
        "new": {"n_holds": new_acc[0], "n_comparable": new_acc[1]},
        "changed": falsification_changed,
    }
    result["candidate_drift"] = candidate_drift
    result["candidates_added"] = candidates_added
    result["candidates_removed"] = candidates_removed
    result["changed"] = bool(
        flip_changed
        or candidate_drift
        or candidates_added
        or candidates_removed
        or falsification_changed
    )
    return result


def _render_contamination_summary(
    contamination_diff: dict[str, Any] | None,
) -> list[str]:
    """Contamination comparison lines for the stderr summary."""
    if type(contamination_diff) is not dict:
        return []
    available = contamination_diff.get("available")
    if available == "neither":
        return []
    if available != "both":
        missing = "new" if available == "old" else "old"
        return [
            "Contamination comparison unavailable: "
            f"{missing} run did not record the contamination report."
        ]
    lines: list[str] = []
    worst = contamination_diff.get("worst_tier") or {}
    if worst.get("changed"):
        lines.append(
            f"Contamination worst tier: {_display_id(worst.get('old'))} -> "
            f"{_display_id(worst.get('new'))}"
        )
    added = contamination_diff.get("pairs_added") or []
    removed = contamination_diff.get("pairs_removed") or []
    if added:
        lines.append(f"Contamination pairs added ({len(added)}):")
        for pair in added[:5]:
            lines.append(
                f"  {_display_id(pair['held_out'])} ~ {_display_id(pair['search'])} "
                f"({_display_id(pair.get('tier'))})"
            )
        if len(added) > 5:
            lines.append(f"  ... and {len(added) - 5} more")
    if removed:
        lines.append(f"Contamination pairs removed ({len(removed)}):")
        for pair in removed[:5]:
            lines.append(
                f"  {_display_id(pair['held_out'])} ~ {_display_id(pair['search'])} "
                f"({_display_id(pair.get('tier'))})"
            )
        if len(removed) > 5:
            lines.append(f"  ... and {len(removed) - 5} more")
    changes = contamination_diff.get("tier_changes") or []
    if changes:
        lines.append(f"Contamination tier changes ({len(changes)}):")
        for change in changes[:5]:
            lines.append(
                f"  {_display_id(change['held_out'])} ~ {_display_id(change['search'])}: "
                f"{_display_id(change.get('old_tier'))} -> {_display_id(change.get('new_tier'))}"
            )
    return lines


def _render_transfer_summary(transfer_diff: dict[str, Any] | None) -> list[str]:
    """Transfer comparison lines for the stderr summary."""
    if type(transfer_diff) is not dict:
        return []
    available = transfer_diff.get("available")
    if available == "neither":
        return []
    if available != "both":
        missing = "new" if available == "old" else "old"
        return [
            "Transfer comparison unavailable: "
            f"{missing} run did not record the transfer experiment."
        ]
    lines: list[str] = []
    verdict = transfer_diff.get("transfer_verdict") or {}
    if verdict.get("changed"):
        lines.append(
            f"Transfer verdict: {_display_id(verdict.get('old'))} -> "
            f"{_display_id(verdict.get('new'))}"
        )
    drift = transfer_diff.get("checkpoint_drift") or []
    if drift:
        lines.append(f"Transfer checkpoint drift ({len(drift)}):")
        for entry in drift[:10]:
            lines.append(
                f"  {_display_id(entry['checkpoint'])}: "
                f"{_display_id(entry['old_decision'])}/{_display_id(entry['old_winner'])} -> "
                f"{_display_id(entry['new_decision'])}/{_display_id(entry['new_winner'])}"
            )
        if len(drift) > 10:
            lines.append(f"  ... and {len(drift) - 10} more")
    cells = transfer_diff.get("cell_drift") or []
    if cells:
        lines.append(f"Transfer cell drift ({len(cells)}):")
        for entry in cells[:10]:
            lines.append(
                f"  {_display_id(entry['candidate'])}/{_display_id(entry['checkpoint'])}: "
                f"{_display_id(entry['old'])} -> {_display_id(entry['new'])}"
            )
        if len(cells) > 10:
            lines.append(f"  ... and {len(cells) - 10} more")
    added = transfer_diff.get("checkpoints_added") or []
    removed = transfer_diff.get("checkpoints_removed") or []
    if added:
        lines.append(f"Transfer checkpoints added: {', '.join(_display_id(c) for c in added)}")
    if removed:
        lines.append(f"Transfer checkpoints removed: {', '.join(_display_id(c) for c in removed)}")
    return lines


def _render_perturbation_summary(perturbation_diff: dict[str, Any] | None) -> list[str]:
    """Perturbation comparison lines for the stderr summary."""
    if type(perturbation_diff) is not dict:
        return []
    available = perturbation_diff.get("available")
    if available == "neither":
        return []
    if available != "both":
        missing = "new" if available == "old" else "old"
        return [
            "Perturbation comparison unavailable: "
            f"{missing} run did not record the perturbation experiment."
        ]
    lines: list[str] = []
    stability = perturbation_diff.get("stability") or {}
    if stability.get("changed"):
        old = stability.get("old") or {}
        new = stability.get("new") or {}

        def _count(value: Any) -> str:
            # Counts are ints (control-char-free); None means "not
            # applicable" (e.g. no primary winner to retain).
            """Return the count."""
            return "(n/a)" if value is None else str(value)

        lines.append(
            "Perturbation stability: "
            f"{_count(old.get('winner_retained'))}/{_count(old.get('n_evaluable'))} -> "
            f"{_count(new.get('winner_retained'))}/{_count(new.get('n_evaluable'))} "
            "variants retain the winner"
        )
    drift = perturbation_diff.get("variant_drift") or []
    if drift:
        lines.append(f"Perturbation variant drift ({len(drift)}):")
        for entry in drift[:5]:
            lines.append(
                f"  {_display_id(entry['variant'])} ({_display_id(entry.get('operator'))}): "
                f"{_display_id(entry['old_decision'])}/{_display_id(entry['old_winner'])} -> "
                f"{_display_id(entry['new_decision'])}/{_display_id(entry['new_winner'])}"
            )
        if len(drift) > 5:
            lines.append(f"  ... and {len(drift) - 5} more")
    added = perturbation_diff.get("variants_added") or []
    removed = perturbation_diff.get("variants_removed") or []
    if added:
        lines.append(f"Perturbation variants added: {', '.join(_display_id(v) for v in added)}")
    if removed:
        lines.append(f"Perturbation variants removed: {', '.join(_display_id(v) for v in removed)}")
    if perturbation_diff.get("operators_changed"):
        lines.append("Perturbation operator set changed between runs.")
    return lines


def _render_healing_summary(healing_diff: dict[str, Any] | None) -> list[str]:
    """Token-healing comparison lines for the stderr summary."""
    if type(healing_diff) is not dict:
        return []
    available = healing_diff.get("available")
    if available == "neither":
        return []
    if available != "both":
        missing = "new" if available == "old" else "old"
        return [
            "Token-healing comparison unavailable: "
            f"{missing} run did not record the token-healing experiment."
        ]
    lines: list[str] = []
    flip = healing_diff.get("flip") or {}
    if flip.get("changed"):
        old = flip.get("old") or {}
        new = flip.get("new") or {}

        def _verdict(side: dict[str, Any]) -> str:
            """Return the verdict."""
            winner = side.get("healed_winner")
            return (
                f"{_display_id(side.get('healed_decision'))}/"
                f"{_display_id('(none)' if winner is None else winner)}"
            )

        lines.append(
            f"Token-healing flip: {old.get('flipped')} -> {new.get('flipped')} "
            f"({_verdict(old)} -> {_verdict(new)})"
        )
    drift = healing_diff.get("candidate_drift") or []
    if drift:
        lines.append(f"Token-healing candidate drift ({len(drift)}):")
        for entry in drift[:5]:
            lines.append(
                f"  {_display_id(entry['candidate'])}: "
                f"native {_display_id(entry['old_native'])}->"
                f"{_display_id(entry['new_native'])}, healed "
                f"{_display_id(entry['old_healed'])}->{_display_id(entry['new_healed'])}"
            )
        if len(drift) > 5:
            lines.append(f"  ... and {len(drift) - 5} more")
    added = healing_diff.get("candidates_added") or []
    removed = healing_diff.get("candidates_removed") or []
    if added:
        lines.append(f"Token-healing candidates added: {', '.join(_display_id(c) for c in added)}")
    if removed:
        lines.append(
            f"Token-healing candidates removed: {', '.join(_display_id(c) for c in removed)}"
        )
    fals = healing_diff.get("falsification") or {}
    if fals.get("changed"):
        old = fals.get("old") or {}
        new = fals.get("new") or {}
        lines.append(
            "Token-healing falsification accuracy: "
            f"{old.get('n_holds')}/{old.get('n_comparable')} -> "
            f"{new.get('n_holds')}/{new.get('n_comparable')} predictions hold"
        )
    return lines


def _render_stop_cause_summary(stopcause_diff: dict[str, Any] | None) -> list[str]:
    """Stop-cause comparison lines for the stderr summary."""
    if type(stopcause_diff) is not dict:
        return []
    available = stopcause_diff.get("available")
    if available == "neither":
        return []
    if available != "both":
        missing = "new" if available == "old" else "old"
        return [
            f"Stop-cause comparison unavailable: {missing} run did not record the stop-cause probe."
        ]
    lines: list[str] = []

    def _fmt_rate(value: Any) -> str:
        """Return the fmt rate."""
        return (
            f"{value:.0%}"
            if type(value) in (int, float) and _float_convertible(value)
            else "(none)"
        )

    drift = stopcause_diff.get("cause_drift") or []
    if drift:
        lines.append(f"Stop-cause rate drift ({len(drift)}):")
        for entry in drift[:5]:
            lines.append(
                f"  {_display_id(entry['candidate'])}/{_display_id(entry.get('cause'))}: "
                f"{_fmt_rate(entry.get('old_rate'))} -> {_fmt_rate(entry.get('new_rate'))}"
            )
        if len(drift) > 5:
            lines.append(f"  ... and {len(drift) - 5} more")
    aggregate = stopcause_diff.get("aggregate_drift") or []
    if aggregate:
        lines.append(f"Stop-cause aggregate drift ({len(aggregate)}):")
        for entry in aggregate[:5]:
            lines.append(
                f"  {_display_id(entry['candidate'])}/{_display_id(entry.get('metric'))}: "
                f"{_fmt_rate(entry.get('old'))} -> {_fmt_rate(entry.get('new'))}"
            )
        if len(aggregate) > 5:
            lines.append(f"  ... and {len(aggregate) - 5} more")
    added = stopcause_diff.get("candidates_added") or []
    removed = stopcause_diff.get("candidates_removed") or []
    if added:
        lines.append(f"Stop-cause candidates added: {', '.join(_display_id(c) for c in added)}")
    if removed:
        lines.append(f"Stop-cause candidates removed: {', '.join(_display_id(c) for c in removed)}")
    return lines


def _render_negation_summary(negation_diff: dict[str, Any] | None) -> list[str]:
    """Negation comparison lines for the stderr summary."""
    if type(negation_diff) is not dict:
        return []
    available = negation_diff.get("available")
    if available == "neither":
        return []
    if available != "both":
        missing = "new" if available == "old" else "old"
        return [
            f"Negation comparison unavailable: {missing} run did not record the negation probe."
        ]
    lines: list[str] = []
    if negation_diff.get("protocol_drift"):
        lines.append(
            "Negation protocol drift: the runs used different negation instruments "
            "(seed or backend) -- not comparable."
        )
    drift = negation_diff.get("candidate_drift") or []
    decisions = [entry for entry in drift if entry.get("metric") == "headline.decision"]
    if decisions:
        lines.append(f"Negation headline decision drift ({len(decisions)}):")
        for entry in decisions[:5]:
            lines.append(
                f"  {_display_id(entry['candidate'])}: "
                f"{_display_id(entry.get('old'))} -> {_display_id(entry.get('new'))}"
            )
    others = [entry for entry in drift if entry.get("metric") != "headline.decision"]
    if others:
        lines.append(f"Negation measurement drift ({len(others)}):")
        for entry in others[:5]:
            lines.append(
                f"  {_display_id(entry['candidate'])}/{_display_id(entry.get('metric'))}: "
                f"{entry.get('old')} -> {entry.get('new')}"
            )
        if len(others) > 5:
            lines.append(f"  ... and {len(others) - 5} more")
    n_cases = negation_diff.get("n_cases") or {}
    if n_cases.get("changed"):
        lines.append(f"Negation case count changed: {n_cases.get('old')} -> {n_cases.get('new')}")
    added = negation_diff.get("candidates_added") or []
    removed = negation_diff.get("candidates_removed") or []
    if added:
        lines.append(f"Negation candidates added: {', '.join(_display_id(c) for c in added)}")
    if removed:
        lines.append(f"Negation candidates removed: {', '.join(_display_id(c) for c in removed)}")
    return lines


def _render_deliforge_summary(deliforge_diff: dict[str, Any] | None) -> list[str]:
    """Delimiter-authority comparison lines for the stderr summary.

    Every interpolated identifier passes through ``_display_id`` so a
    hostile candidate name cannot inject terminal escapes (the wave-3
    red-team A lesson).
    """
    if type(deliforge_diff) is not dict:
        return []
    available = deliforge_diff.get("available")
    if available == "neither":
        return []
    if available != "both":
        missing = "new" if available == "old" else "old"
        return [
            "Delimiter-authority comparison unavailable: "
            f"{missing} run did not record the deliforge probe."
        ]
    lines: list[str] = []

    def _fmt_gap(value: Any) -> str:
        """Return the fmt gap."""
        return (
            f"{value:+.2f}"
            if type(value) in (int, float) and _float_convertible(value)
            else "(none)"
        )

    drift = deliforge_diff.get("gap_drift") or []
    if drift:
        lines.append(f"Delimiter-authority gap drift ({len(drift)}):")
        for entry in drift[:5]:
            if type(entry) is not dict:
                continue
            lines.append(
                f"  {_display_id(entry.get('candidate'))}/{_display_id(entry.get('cell'))}: "
                f"{_fmt_gap(entry.get('old'))} -> {_fmt_gap(entry.get('new'))}"
            )
        if len(drift) > 5:
            lines.append(f"  ... and {len(drift) - 5} more")
    top_drift = deliforge_diff.get("top_gap_drift") or []
    for entry in top_drift[:3]:
        if type(entry) is not dict:
            continue
        lines.append(
            "Delimiter-authority top-gap drift "
            f"({_display_id(entry.get('candidate'))}): "
            f"{_fmt_gap(entry.get('old'))} -> {_fmt_gap(entry.get('new'))}"
        )
    added = deliforge_diff.get("candidates_added") or []
    removed = deliforge_diff.get("candidates_removed") or []
    if added:
        lines.append(
            f"Delimiter-probe candidates added: {', '.join(_display_id(c) for c in added)}"
        )
    if removed:
        lines.append(
            f"Delimiter-probe candidates removed: {', '.join(_display_id(c) for c in removed)}"
        )
    excl_added = deliforge_diff.get("excluded_added") or []
    excl_removed = deliforge_diff.get("excluded_removed") or []
    if excl_added:
        lines.append(
            f"Delimiter-probe newly excluded: {', '.join(_display_id(c) for c in excl_added)}"
        )
    if excl_removed:
        lines.append(
            f"Delimiter-probe no longer excluded: {', '.join(_display_id(c) for c in excl_removed)}"
        )
    return lines


def _render_fewshot_summary(fewshot_diff: dict[str, Any] | None) -> list[str]:
    """Few-shot x template comparison lines for the stderr summary.

    Every interpolated identifier passes through ``_display_id`` so a
    hostile candidate name cannot inject terminal escapes (the wave-3
    red-team A lesson).
    """
    if type(fewshot_diff) is not dict:
        return []
    available = fewshot_diff.get("available")
    if available == "neither":
        return []
    if available != "both":
        missing = "new" if available == "old" else "old"
        return [f"Few-shot comparison unavailable: {missing} run did not record the fewshot probe."]
    lines: list[str] = []

    def _fmt_gain(value: Any) -> str:
        """Return the fmt gain."""
        return (
            f"{value:.2f}"
            if type(value) in (int, float) and _float_convertible(value)
            else "(none)"
        )

    gain_drift = fewshot_diff.get("gain_drift") or []
    if gain_drift:
        lines.append(f"Few-shot gain drift ({len(gain_drift)}):")
        for entry in gain_drift[:5]:
            if type(entry) is not dict:
                continue
            lines.append(
                f"  {_display_id(entry.get('candidate'))}: "
                f"{_fmt_gain(entry.get('old'))} -> {_fmt_gain(entry.get('new'))}"
            )
        if len(gain_drift) > 5:
            lines.append(f"  ... and {len(gain_drift) - 5} more")
    sens_drift = fewshot_diff.get("sensitivity_drift") or []
    for entry in sens_drift[:3]:
        if type(entry) is not dict:
            continue
        lines.append(
            "Few-shot order-sensitivity drift "
            f"({_display_id(entry.get('candidate'))}): "
            f"{_fmt_gain(entry.get('old'))} -> {_fmt_gain(entry.get('new'))}"
        )
    arm_drift = fewshot_diff.get("arm_drift") or []
    for entry in arm_drift[:3]:
        if type(entry) is not dict:
            continue
        lines.append(
            f"Few-shot arm drift ({_display_id(entry.get('arm'))}): "
            f"{entry.get('old_decision')}/{_display_id(entry.get('old_winner'))} -> "
            f"{entry.get('new_decision')}/{_display_id(entry.get('new_winner'))}"
        )
    added = fewshot_diff.get("candidates_added") or []
    removed = fewshot_diff.get("candidates_removed") or []
    if added:
        lines.append(f"Few-shot candidates added: {', '.join(_display_id(c) for c in added)}")
    if removed:
        lines.append(f"Few-shot candidates removed: {', '.join(_display_id(c) for c in removed)}")
    excl_added = fewshot_diff.get("excluded_added") or []
    excl_removed = fewshot_diff.get("excluded_removed") or []
    if excl_added:
        lines.append(f"Few-shot newly excluded: {', '.join(_display_id(c) for c in excl_added)}")
    if excl_removed:
        lines.append(
            f"Few-shot no longer excluded: {', '.join(_display_id(c) for c in excl_removed)}"
        )
    return lines


def _render_answer_collision_summary(
    answer_collision_diff: dict[str, Any] | None,
) -> list[str]:
    """Answer-collision comparison lines for the stderr summary.

    Every interpolated identifier passes through ``_display_id`` so a
    hostile candidate name cannot inject terminal escapes. Collision
    rates are rendered as percentages; a protocol drift (fixture/
    classification-rule change) is reported first because it
    invalidates rate comparisons.
    """
    if type(answer_collision_diff) is not dict:
        return []
    available = answer_collision_diff.get("available")
    if available == "neither":
        return []
    if available != "both":
        missing = "new" if available == "old" else "old"
        return [
            "Answer-collision comparison unavailable: "
            f"{missing} run did not record the answer-collision audit."
        ]
    lines: list[str] = []
    if answer_collision_diff.get("protocol_drift"):
        lines.append(
            "Answer-collision protocol drift: the frozen audit protocol "
            "(fixtures/classification rule) changed between runs; "
            "collision-rate comparisons are not like-for-like."
        )

    def _fmt_rate(value: Any) -> str:
        """Return the fmt rate."""
        return (
            f"{value:.1%}"
            if type(value) in (int, float) and _float_convertible(value)
            else "(none)"
        )

    drift = answer_collision_diff.get("rate_drift") or []
    for entry in drift[:3]:
        if type(entry) is not dict:
            continue
        lines.append(
            f"Answer-collision rate drift ({_display_id(entry.get('candidate'))}): "
            f"{_fmt_rate(entry.get('old'))} -> {_fmt_rate(entry.get('new'))}"
        )
    if len(drift) > 3:
        lines.append(f"  ... and {len(drift) - 3} more rate drifts")
    added = answer_collision_diff.get("colliding_added") or []
    removed = answer_collision_diff.get("colliding_removed") or []
    if added:
        lines.append(
            "Answer-collision newly colliding: " + ", ".join(_display_id(c) for c in added)
        )
    if removed:
        lines.append(
            "Answer-collision no longer colliding: " + ", ".join(_display_id(c) for c in removed)
        )
    if answer_collision_diff.get("uniform_changed"):
        lines.append("Answer-collision uniform-intact (regression-guard) flag changed.")
    cand_added = answer_collision_diff.get("candidates_added") or []
    cand_removed = answer_collision_diff.get("candidates_removed") or []
    if cand_added:
        lines.append(
            "Answer-collision candidates added: " + ", ".join(_display_id(c) for c in cand_added)
        )
    if cand_removed:
        lines.append(
            "Answer-collision candidates removed: "
            + ", ".join(_display_id(c) for c in cand_removed)
        )
    return lines


def _render_template_robustness_summary(
    tr_diff: dict[str, Any] | None,
) -> list[str]:
    """Template-robustness comparison lines for the stderr summary.

    Every interpolated identifier passes through ``_display_id`` so a
    hostile candidate name cannot inject terminal escapes (the wave-3
    red-team A lesson). Rates render with two decimals.
    """
    if type(tr_diff) is not dict:
        return []
    available = tr_diff.get("available")
    if available == "neither":
        return []
    if available != "both":
        missing = "new" if available == "old" else "old"
        return [
            "Template-robustness comparison unavailable: "
            f"{missing} run did not record the template-robustness probe."
        ]
    lines: list[str] = []

    def _fmt_rate(value: Any) -> str:
        """Return the fmt rate."""
        return (
            f"{value:.2f}"
            if type(value) in (int, float) and _float_convertible(value)
            else "(none)"
        )

    drift = tr_diff.get("preservation_drift") or []
    for entry in drift[:3]:
        if type(entry) is not dict:
            continue
        lines.append(
            "Robustness preservation drift "
            f"({_display_id(entry.get('candidate'))}): "
            f"{_fmt_rate(entry.get('old'))} -> {_fmt_rate(entry.get('new'))}"
        )
    if len(drift) > 3:
        lines.append(f"  ... and {len(drift) - 3} more preservation drifts")
    op_drift = tr_diff.get("operator_drift") or []
    for entry in op_drift[:3]:
        if type(entry) is not dict:
            continue
        lines.append(
            "Robustness operator drift "
            f"({_display_id(entry.get('candidate'))}/{_display_id(entry.get('operator'))}): "
            f"{_fmt_rate(entry.get('old'))} -> {_fmt_rate(entry.get('new'))}"
        )
    if len(op_drift) > 3:
        lines.append(f"  ... and {len(op_drift) - 3} more operator drifts")
    added = tr_diff.get("candidates_added") or []
    removed = tr_diff.get("candidates_removed") or []
    if added:
        lines.append("Robustness candidates added: " + ", ".join(_display_id(c) for c in added))
    if removed:
        lines.append("Robustness candidates removed: " + ", ".join(_display_id(c) for c in removed))
    return lines


def _render_ablation_summary(
    ablation_diff: dict[str, Any] | None,
) -> list[str]:
    """Field-ablation comparison lines for the stderr summary.

    Every interpolated identifier passes through ``_display_id`` so a
    hostile candidate name cannot inject terminal escapes (the wave-3
    red-team A lesson).
    """
    if type(ablation_diff) is not dict:
        return []
    available = ablation_diff.get("available")
    if available == "neither":
        return []
    if available != "both":
        missing = "new" if available == "old" else "old"
        return [
            "Ablation comparison unavailable: "
            f"{missing} run did not record the field-ablation probe."
        ]
    lines: list[str] = []
    drift = ablation_diff.get("attribution_drift") or []
    if drift:
        lines.append(f"Ablation attribution drift ({len(drift)}):")
        for entry in drift[:5]:
            if type(entry) is not dict:
                continue
            lines.append(
                f"  {_display_id(entry.get('candidate'))}/"
                f"{_display_id(entry.get('field'))}: "
                f"{entry.get('old_n_flips')} -> {entry.get('new_n_flips')} flips"
            )
        if len(drift) > 5:
            lines.append(f"  ... and {len(drift) - 5} more")
    lb_drift = ablation_diff.get("load_bearing_drift") or []
    for entry in lb_drift[:3]:
        if type(entry) is not dict:
            continue
        lines.append(
            "Ablation load-bearing drift "
            f"({_display_id(entry.get('candidate'))}): "
            f"{entry.get('old_load_bearing')} -> {entry.get('new_load_bearing')}"
        )
    added = ablation_diff.get("candidates_added") or []
    removed = ablation_diff.get("candidates_removed") or []
    if added:
        lines.append("Ablation candidates added: " + ", ".join(_display_id(c) for c in added))
    if removed:
        lines.append("Ablation candidates removed: " + ", ".join(_display_id(c) for c in removed))
    return lines


def _render_verbosity_summary(verbosity_diff: dict[str, Any] | None) -> list[str]:
    """Verbosity profile comparison lines for the stderr summary.

    Every interpolated identifier passes through ``_display_id`` so a
    hostile candidate name cannot inject terminal escapes (the wave-3
    red-team A lesson). Excess values are token counts, rendered with
    one decimal.
    """
    if type(verbosity_diff) is not dict:
        return []
    available = verbosity_diff.get("available")
    if available == "neither":
        return []
    if available != "both":
        missing = "new" if available == "old" else "old"
        return [
            f"Verbosity comparison unavailable: {missing} run did not record the verbosity probe."
        ]
    lines: list[str] = []

    def _fmt_tokens(value: Any) -> str:
        """Return the fmt tokens."""
        return (
            f"{value:.1f}"
            if type(value) in (int, float) and _float_convertible(value)
            else "(none)"
        )

    for drift_key, label in (
        ("median_drift", "median-excess"),
        ("iqr_drift", "iqr-excess"),
        ("capped_share_drift", "capped-share"),
    ):
        drift = verbosity_diff.get(drift_key) or []
        for entry in drift[:3]:
            if type(entry) is not dict:
                continue
            lines.append(
                f"Verbosity {label} drift ({_display_id(entry.get('candidate'))}): "
                f"{_fmt_tokens(entry.get('old'))} -> {_fmt_tokens(entry.get('new'))}"
            )
        if len(drift) > 3:
            lines.append(f"  ... and {len(drift) - 3} more {label} drifts")
    added = verbosity_diff.get("candidates_added") or []
    removed = verbosity_diff.get("candidates_removed") or []
    if added:
        lines.append(f"Verbosity candidates added: {', '.join(_display_id(c) for c in added)}")
    if removed:
        lines.append(f"Verbosity candidates removed: {', '.join(_display_id(c) for c in removed)}")
    excl_added = verbosity_diff.get("excluded_added") or []
    excl_removed = verbosity_diff.get("excluded_removed") or []
    if excl_added:
        lines.append(f"Verbosity newly excluded: {', '.join(_display_id(c) for c in excl_added)}")
    if excl_removed:
        lines.append(
            f"Verbosity no longer excluded: {', '.join(_display_id(c) for c in excl_removed)}"
        )
    return lines


def _render_refusals_summary(refusals_diff: dict[str, Any] | None) -> list[str]:
    """Suspected-refusal comparison lines for the stderr summary.

    Every interpolated identifier passes through ``_display_id`` so a
    hostile candidate or fixture name cannot inject terminal escapes
    (the wave-3 red-team A lesson). Rates are percentages and travel
    with their denominators.
    """
    if type(refusals_diff) is not dict:
        return []
    available = refusals_diff.get("available")
    if available == "neither":
        return []
    if available != "both":
        missing = "new" if available == "old" else "old"
        return [
            f"Refusals comparison unavailable: {missing} run did not record the refusals probe."
        ]
    lines: list[str] = []
    if refusals_diff.get("protocol_drift"):
        lines.append(
            "Refusals detector protocol drift: the frozen opener list or detector "
            "version differs between runs; rates are not directly comparable."
        )

    def _fmt_rate(value: Any) -> str:
        """Return the fmt rate."""
        return (
            f"{value:.0%}"
            if type(value) in (int, float) and _float_convertible(value)
            else "(none)"
        )

    drift = refusals_diff.get("rate_drift") or []
    for entry in drift[:3]:
        if type(entry) is not dict:
            continue
        lines.append(
            f"Refusals rate drift ({_display_id(entry.get('candidate'))}): "
            f"{_fmt_rate(entry.get('old_rate'))} of {entry.get('old_n')} -> "
            f"{_fmt_rate(entry.get('new_rate'))} of {entry.get('new_n')}"
        )
    if len(drift) > 3:
        lines.append(f"  ... and {len(drift) - 3} more rate drifts")
    hits_added = refusals_diff.get("hits_added") or []
    hits_removed = refusals_diff.get("hits_removed") or []
    if hits_added:
        lines.append(f"Refusals hits added: {len(hits_added)}")
    if hits_removed:
        lines.append(f"Refusals hits removed: {len(hits_removed)}")
    added = refusals_diff.get("candidates_added") or []
    removed = refusals_diff.get("candidates_removed") or []
    if added:
        lines.append(f"Refusals candidates added: {', '.join(_display_id(c) for c in added)}")
    if removed:
        lines.append(f"Refusals candidates removed: {', '.join(_display_id(c) for c in removed)}")
    excl_added = refusals_diff.get("excluded_added") or []
    excl_removed = refusals_diff.get("excluded_removed") or []
    if excl_added:
        lines.append(f"Refusals newly excluded: {', '.join(_display_id(c) for c in excl_added)}")
    if excl_removed:
        lines.append(
            f"Refusals no longer excluded: {', '.join(_display_id(c) for c in excl_removed)}"
        )
    if refusals_diff.get("mismatch_changed"):
        lines.append("Refusals evaluable-mismatch status changed between runs.")
    return lines


def _render_cache_profile_summary(cache_profile_diff: dict[str, Any] | None) -> list[str]:
    """Cache-profile comparison lines for the stderr summary.

    Every interpolated identifier passes through ``_display_id`` so a
    hostile candidate name cannot inject terminal escapes (the wave-3
    red-team A lesson). Sharability fractions are rendered as
    percentages.
    """
    if type(cache_profile_diff) is not dict:
        return []
    available = cache_profile_diff.get("available")
    if available == "neither":
        return []
    if available != "both":
        missing = "new" if available == "old" else "old"
        return [
            "Cache-profile comparison unavailable: "
            f"{missing} run did not record the cache-profile probe."
        ]
    lines: list[str] = []

    def _fmt_share(value: Any) -> str:
        """Return the fmt share."""
        return (
            f"{value:.1%}"
            if type(value) in (int, float) and _float_convertible(value)
            else "(none)"
        )

    for drift_key, label in (
        ("block_drift", "achievable-block-share"),
        ("token_drift", "achievable-token-share"),
        ("potential_drift", "potential-block-share"),
    ):
        drift = cache_profile_diff.get(drift_key) or []
        for entry in drift[:3]:
            if type(entry) is not dict:
                continue
            lines.append(
                f"Cache-profile {label} drift ({_display_id(entry.get('candidate'))}): "
                f"{_fmt_share(entry.get('old'))} -> {_fmt_share(entry.get('new'))}"
            )
        if len(drift) > 3:
            lines.append(f"  ... and {len(drift) - 3} more {label} drifts")
    added = cache_profile_diff.get("candidates_added") or []
    removed = cache_profile_diff.get("candidates_removed") or []
    if added:
        lines.append(f"Cache-profile candidates added: {', '.join(_display_id(c) for c in added)}")
    if removed:
        lines.append(
            f"Cache-profile candidates removed: {', '.join(_display_id(c) for c in removed)}"
        )
    excl_added = cache_profile_diff.get("excluded_added") or []
    excl_removed = cache_profile_diff.get("excluded_removed") or []
    if excl_added:
        lines.append(
            f"Cache-profile newly excluded: {', '.join(_display_id(c) for c in excl_added)}"
        )
    if excl_removed:
        lines.append(
            f"Cache-profile no longer excluded: {', '.join(_display_id(c) for c in excl_removed)}"
        )
    return lines


def _render_synthhistory_summary(synthhistory_diff: dict[str, Any] | None) -> list[str]:
    """Synthetic-history comparison lines for the stderr summary."""
    if type(synthhistory_diff) is not dict:
        return []
    available = synthhistory_diff.get("available")
    if available == "neither":
        return []
    if available != "both":
        missing = "new" if available == "old" else "old"
        return [
            "Synthetic-history comparison unavailable: "
            f"{missing} run did not record the synthhistory experiment."
        ]
    lines: list[str] = []

    def _verdict(decision: Any, winner: Any) -> str:
        """Return the verdict."""
        return f"{_display_id(decision)}/{_display_id('(none)' if winner is None else winner)}"

    conditions = synthhistory_diff.get("condition_changed") or {}
    for condition in sorted(conditions):
        entry = conditions[condition]
        if entry.get("changed"):
            lines.append(
                f"Synthetic history ({_display_id(condition)}): "
                f"{_verdict(entry.get('old_decision'), entry.get('old_winner'))} -> "
                f"{_verdict(entry.get('new_decision'), entry.get('new_winner'))}"
            )
    added = synthhistory_diff.get("conditions_added") or []
    removed = synthhistory_diff.get("conditions_removed") or []
    if added:
        lines.append(f"History conditions added: {', '.join(_display_id(c) for c in added)}")
    if removed:
        lines.append(f"History conditions removed: {', '.join(_display_id(c) for c in removed)}")
    if synthhistory_diff.get("flips_changed"):
        lines.append("Synthetic-history flip pattern changed between runs.")

    def _fmt_rate(value: Any) -> str:
        """Return the fmt rate."""
        return (
            f"{value:.1%}"
            if type(value) in (int, float) and _float_convertible(value)
            else "(none)"
        )

    drift = synthhistory_diff.get("pass_rate_drift") or []
    if drift:
        lines.append(f"History pass-rate drift ({len(drift)}):")
        for entry in drift[:5]:
            lines.append(
                f"  {_display_id(entry['candidate'])} "
                f"({_display_id(entry.get('condition'))}): "
                f"{_fmt_rate(entry.get('old_pass_rate'))}->"
                f"{_fmt_rate(entry.get('new_pass_rate'))}"
            )
        if len(drift) > 5:
            lines.append(f"  ... and {len(drift) - 5} more")
    added_c = synthhistory_diff.get("candidates_added") or []
    removed_c = synthhistory_diff.get("candidates_removed") or []
    if added_c:
        lines.append(f"History candidates added: {', '.join(_display_id(c) for c in added_c)}")
    if removed_c:
        lines.append(f"History candidates removed: {', '.join(_display_id(c) for c in removed_c)}")
    groups = synthhistory_diff.get("group_rates") or {}
    if groups.get("changed"):
        old = groups.get("old") or {}
        new = groups.get("new") or {}
        for label in sorted(set(old) | set(new)):
            if old.get(label) != new.get(label):
                lines.append(
                    f"History {_display_id(label)} mean pass rate: "
                    f"{_fmt_rate(old.get(label))} -> {_fmt_rate(new.get(label))}"
                )
    return lines


def _render_batchinv_summary(batchinv_diff: dict[str, Any] | None) -> list[str]:
    """Batch-invariance comparison lines for the stderr summary."""
    if type(batchinv_diff) is not dict:
        return []
    available = batchinv_diff.get("available")
    if available == "neither":
        return []
    if available != "both":
        missing = "new" if available == "old" else "old"
        return [
            "Batch-invariance audit unavailable: "
            f"{missing} run did not record the batch_invariance experiment."
        ]
    lines: list[str] = []
    if batchinv_diff.get("verdict_changed"):
        lines.append("Batch-invariance verdict changed between runs.")
    if batchinv_diff.get("divergence_rate_changed"):
        lines.append("Batch-invariance divergence rate changed between runs.")
    diverged_added = batchinv_diff.get("diverged_added") or []
    if diverged_added:
        shown = ", ".join(_display_id(p) for p in diverged_added[:5])
        lines.append(f"Newly diverged prompts ({len(diverged_added)}): {shown}")
        if len(diverged_added) > 5:
            lines.append(f"  ... and {len(diverged_added) - 5} more")
    diverged_removed = batchinv_diff.get("diverged_removed") or []
    if diverged_removed:
        lines.append(f"Divergence resolved on {len(diverged_removed)} prompt(s).")
    prompts_added = batchinv_diff.get("prompts_added") or []
    if prompts_added:
        lines.append(f"Batch-audit prompts added: {len(prompts_added)}")
    prompts_removed = batchinv_diff.get("prompts_removed") or []
    if prompts_removed:
        lines.append(f"Batch-audit prompts removed: {len(prompts_removed)}")
    return lines


def _render_boundary_audit_summary(audit_diff: dict[str, Any]) -> list[str]:
    """Prompt-boundary canonicality comparison lines for the stderr summary."""
    available = audit_diff.get("available")
    if available == "neither":
        return []
    if available != "both":
        missing = "new" if available == "old" else "old"
        return [
            "Prompt-boundary canonicality comparison unavailable: "
            f"{missing} run did not record an audit."
        ]
    lines: list[str] = []

    def _fmt(rate: Any) -> str:
        """Return the fmt."""
        return (
            f"{rate:.1%}" if type(rate) in (int, float) and _float_convertible(rate) else "(none)"
        )

    overall = audit_diff["overall_rate"]
    lines.append(f"Prompt Boundary Problem rate: {_fmt(overall['old'])} -> {_fmt(overall['new'])}")
    drift = audit_diff.get("family_drift") or []
    if drift:
        lines.append(f"Family PBP-rate drift ({len(drift)}):")
        for entry in drift:
            lines.append(
                f"  {_display_id(entry['family'])}: "
                f"{_fmt(entry['old_rate'])} -> {_fmt(entry['new_rate'])}"
            )
    return lines


def _render_robustness_summary(robustness_diff: dict[str, Any] | None) -> list[str]:
    """Selection-robustness comparison lines for the stderr summary."""
    if type(robustness_diff) is not dict:
        return []
    available = robustness_diff.get("available")
    if available == "neither":
        return []
    if available != "both":
        missing = "new" if available == "old" else "old"
        return [
            "Selection-robustness comparison unavailable: "
            f"{missing} run did not record the analysis."
        ]
    lines: list[str] = []
    drift = robustness_diff.get("verdict_drift") or []
    if drift:
        lines.append(f"Stop-semantics selection drift ({len(drift)}):")
        for entry in drift:
            lines.append(
                f"  {_display_id(entry['semantic'])}: "
                f"{_display_id(entry['old_decision'])}/{_display_id(entry['old_winner'])} -> "
                f"{_display_id(entry['new_decision'])}/{_display_id(entry['new_winner'])}"
            )
    if robustness_diff.get("flips_changed"):
        lines.append("Selection flip pattern changed between runs.")
    return lines


def diff_held_out_floor(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``held_out_budget_floor`` payloads; deterministic.

    Answers "did the verification's token-budget story change between
    runs?": the floor from which the held-out verification passes
    unchanged, and whether the verification verdict was budget-robust.
    A side without a recorded floor is reported plainly instead of
    compared. No model, no network.
    """
    old_ok = _valid_held_out_floor(old)
    new_ok = _valid_held_out_floor(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    result: dict[str, Any] = {"available": available, "changed": False}
    if available != "both":
        return result
    old_floor = old.get("safe_budget_floor")
    new_floor = new.get("safe_budget_floor")
    old_robust = old.get("budget_robust")
    new_robust = new.get("budget_robust")
    floor_changed = old_floor != new_floor
    robust_changed = old_robust != new_robust
    result["safe_budget_floor"] = {
        "old": old_floor,
        "new": new_floor,
        "changed": floor_changed,
    }
    result["budget_robust"] = {
        "old": old_robust,
        "new": new_robust,
        "changed": robust_changed,
    }
    result["changed"] = bool(floor_changed or robust_changed)
    return result


def _render_held_out_floor_summary(floor_diff: dict[str, Any]) -> list[str]:
    """Held-out budget-floor comparison lines for the stderr summary."""
    available = floor_diff.get("available")
    if available == "neither":
        return []
    if available != "both":
        missing = "new" if available == "old" else "old"
        return [f"Held-out budget floor comparison unavailable: {missing} run did not record one."]
    lines: list[str] = []

    def _fmt(value: Any) -> str:
        """Return the fmt."""
        return str(value) if type(value) is int else "(none)"

    robust = floor_diff["budget_robust"]
    yes_no = {True: "yes", False: "no"}
    lines.append(
        "Held-out verification budget-robust: "
        f"{yes_no.get(robust['old'], robust['old'])} -> "
        f"{yes_no.get(robust['new'], robust['new'])}"
    )
    floor = floor_diff["safe_budget_floor"]
    lines.append(f"Held-out verification floor: {_fmt(floor['old'])} -> {_fmt(floor['new'])}")
    return lines


def _render_frontier_summary(frontier: dict[str, Any]) -> list[str]:
    """Token-budget frontier comparison lines for the stderr summary."""
    available = frontier.get("available")
    if available == "neither":
        return ["Token-budget frontier: not recorded by either run."]
    if available != "both":
        missing = "new" if available == "old" else "old"
        return [f"Token-budget frontier comparison unavailable: {missing} run did not record one."]
    lines: list[str] = []
    sweep = frontier["max_new_tokens"]
    if sweep["changed"]:
        lines.append(
            f"Frontier sweep range: {sweep['old']} -> {sweep['new']} token budgets "
            "(compared over the common prefix)"
        )
    robust = frontier["budget_robust"]
    yes_no = {True: "yes", False: "no"}
    lines.append(
        "Budget-robust: "
        f"{yes_no.get(robust['old'], robust['old'])} -> "
        f"{yes_no.get(robust['new'], robust['new'])}"
    )
    floor = frontier.get("safe_budget_floor") or {}

    def _fmt_floor(value: Any) -> str:
        """Return the fmt floor."""
        return str(value) if type(value) is int else "(none)"

    lines.append(
        f"Safe budget floor: {_fmt_floor(floor.get('old'))} -> {_fmt_floor(floor.get('new'))}"
    )
    for key, label in (
        ("decision_flips", "Post-hoc decision differs"),
        ("winner_flips", "Post-hoc winner differs"),
        ("leader_flips", "Per-budget leader differs"),
    ):
        spans = frontier.get(key) or []
        if not spans:
            continue
        lines.append(f"{label} ({len(spans)} span(s)):")
        for lo, hi, old_value, new_value in spans:
            old_text = "(none)" if old_value is None else str(old_value)
            new_text = "(none)" if new_value is None else str(new_value)
            lines.append(
                f"  budgets {lo}\u2013{hi}: {_display_id(old_text)} -> {_display_id(new_text)}"
            )
    attr = frontier.get("flip_attribution") or {}
    if attr.get("changed"):
        old_attr = attr.get("old")
        new_attr = attr.get("new")
        if old_attr is None or new_attr is None:
            missing = "old" if old_attr is None else "new"
            lines.append(f"Flip attribution: not recorded by {missing} run.")
        else:
            lines.append("Flip attribution differs:")
            for side_label, side_attr in (("old", old_attr), ("new", new_attr)):
                for entry in side_attr:
                    lo, hi = entry["span"]
                    crossings = (
                        "; ".join(
                            f"{_display_id(c['candidate'])}/{_display_id(c['fixture'])}"
                            f" @ {c['min_budget']}"
                            for c in entry["crossings"]
                        )
                        or "(no pass-status changes in span)"
                    )
                    lines.append(f"  budgets {lo}\u2013{hi} ({side_label}): {crossings}")
    if not frontier["changed"]:
        lines.append("Token-budget frontier: no differences.")
    return lines


SURFACE_NOISE_INSTRUMENT = "surface-noise-probe"


def _valid_surface_noise(payload: Any) -> bool:
    """Structural check for a ``probe.json`` surface-noise payload.

    The probe is a per-prompt instrument, so its diff lives outside the
    results.json dispatcher: it compares two standalone probe.json files
    (before/after a prompt edit), not two experiment runs.
    """
    if type(payload) is not dict:
        return False
    if payload.get("instrument") != SURFACE_NOISE_INSTRUMENT:
        return False
    if payload.get("status") != "ok":
        return False
    operators = payload.get("operators")
    if type(operators) is not dict or not all(type(k) is str for k in operators):
        return False
    for entry in operators.values():
        if type(entry) is not dict:
            return False
        stability = entry.get("stability")
        if stability is not None and (
            type(stability) not in (int, float) or isinstance(stability, bool)
        ):
            return False
        for key in ("n_effective", "n_total"):
            value = entry.get(key)
            if type(value) is not int or value < 0:
                return False
        nearest = entry.get("nearest_observed_flip")
        if nearest is not None and type(nearest) is not dict:
            return False
    if type(payload.get("prompt_sha256")) is not str:
        return False
    protocol = payload.get("protocol")
    if type(protocol) is not dict:
        return False
    if type(protocol.get("backend_fingerprint")) is not str:
        return False
    # The diff reads baseline.n_pass through a truthiness fallback
    # (``(payload.get("baseline") or {}).get("n_pass")``), so a truthy
    # non-dict baseline (a string, a list, a number) escaped as a raw
    # AttributeError (bug-hunt wave 2, F1). The producer's baseline is
    # a counts dict; validate the field like its siblings.
    baseline = payload.get("baseline")
    if baseline is not None:
        if type(baseline) is not dict:
            return False
        n_pass = baseline.get("n_pass")
        if n_pass is not None and (type(n_pass) is not int or n_pass < 0):
            return False
    return True


def diff_surface_noise(old: Any, new: Any) -> dict[str, Any]:
    """Compare two surface-noise probe.json payloads; deterministic.

    Answers "did the probe's story change between the two prompts?":
    per-operator stability drift and nearest-flip drift, baseline
    pass-count change, and — most importantly for the comparability rule
    — whether the prompt text or the protocol fingerprint changed at all.
    A side without a recorded probe is reported plainly. No model, no
    network.
    """
    old_ok = _valid_surface_noise(old)
    new_ok = _valid_surface_noise(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    result: dict[str, Any] = {
        "available": available,
        "changed": False,
        "identical": available == "neither",
    }
    if available != "both":
        return result

    # Comparability first: the probe's contract compares scores only
    # under identical (fixtures, backend, protocol-fingerprint) triples.
    protocol_drift = (
        old["protocol"].get("backend_fingerprint") != new["protocol"].get("backend_fingerprint")
        or old["protocol"].get("search_plan") != new["protocol"].get("search_plan")
        or old["protocol"].get("fixture_sha256") != new["protocol"].get("fixture_sha256")
    )
    prompt_changed = old["prompt_sha256"] != new["prompt_sha256"]

    old_view = _operator_view(old)
    new_view = _operator_view(new)
    operators_added = sorted(set(new_view) - set(old_view))
    operators_removed = sorted(set(old_view) - set(new_view))
    stability_drift: list[dict[str, Any]] = []
    nearest_flip_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_view) & set(new_view)):
        old_entry = old_view[name]
        new_entry = new_view[name]
        if old_entry["stability"] != new_entry["stability"]:
            stability_drift.append(
                {
                    "operator": name,
                    "old_stability": old_entry["stability"],
                    "new_stability": new_entry["stability"],
                }
            )
        if (
            old_entry["nearest_flip_distance"] != new_entry["nearest_flip_distance"]
            or old_entry["nearest_flip_variant"] != new_entry["nearest_flip_variant"]
        ):
            nearest_flip_drift.append(
                {
                    "operator": name,
                    "old": (
                        old_entry["nearest_flip_distance"],
                        old_entry["nearest_flip_variant"],
                    ),
                    "new": (
                        new_entry["nearest_flip_distance"],
                        new_entry["nearest_flip_variant"],
                    ),
                }
            )

    old_baseline = (old.get("baseline") or {}).get("n_pass")
    new_baseline = (new.get("baseline") or {}).get("n_pass")
    baseline_changed = old_baseline != new_baseline

    changed = bool(
        protocol_drift
        or prompt_changed
        or operators_added
        or operators_removed
        or stability_drift
        or nearest_flip_drift
        or baseline_changed
    )
    result.update(
        {
            "changed": changed,
            "identical": not changed,
            "protocol_drift": protocol_drift,
            "prompt_changed": prompt_changed,
            "baseline_n_pass": {"old": old_baseline, "new": new_baseline},
            "baseline_changed": baseline_changed,
            "operators_added": operators_added,
            "operators_removed": operators_removed,
            "stability_drift": stability_drift,
            "nearest_flip_drift": nearest_flip_drift,
        }
    )
    return result
