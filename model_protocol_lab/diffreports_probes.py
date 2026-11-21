"""diffreports probes family: split from diffreports.py (W7-2).

Facades keep the public surface: import from model_protocol_lab.diffreports, not from this module.
"""

import hashlib
import json
import math
import re
from typing import Any

from .diffreports_core import (
    _bool_or_none,
    _display_id,
    _float_convertible,
    _is_hex_digest,
    _recomputed_equal,
    _valid_primary_pair,
)


def _valid_hierarchy(payload: Any) -> bool:
    """Structural check for a ``hierarchy`` payload.

    Enforces the wave-2 C3 lesson at the value level: the compared
    fields must be bools/strings/None/numbers so the renderers never hit
    a TypeError on a hostile payload, and map keys must be strings.
    """
    if type(payload) is not dict or payload.get("available") is not True:
        return False
    positions = payload.get("positions")
    if type(positions) is not list or not positions or not all(type(p) is str for p in positions):
        return False
    per_condition = payload.get("per_condition")
    if type(per_condition) is not dict or set(per_condition) != set(positions):
        return False
    for condition in per_condition.values():
        if type(condition) is not dict:
            return False
        if type(condition.get("decision")) is not str:
            return False
        winner = condition.get("winner")
        if winner is not None and type(winner) is not str:
            return False
        candidates = condition.get("per_candidate")
        if type(candidates) is not dict or not all(type(k) is str for k in candidates):
            return False
        for entry in candidates.values():
            if entry is None:
                continue
            if type(entry) is not dict:
                return False
            for key in ("system_rate", "user_rate", "neither_rate"):
                rate = entry.get(key)
                if rate is not None and type(rate) not in (int, float):
                    return False
            if entry.get("pass") is not None and type(entry.get("pass")) is not bool:
                return False
    groups = payload.get("group_adherence")
    if type(groups) is not dict:
        return False
    for label, group in groups.items():
        if type(label) is not str or type(group) is not dict:
            return False
        mean = group.get("mean_system_rate")
        if mean is not None and type(mean) not in (int, float):
            return False
        if type(group.get("n_candidates")) is not int:
            return False
        if type(group.get("n_families")) is not int:
            return False
    return type(payload.get("flips")) is list


def _hierarchy_condition_triplet(payload: Any, position: str) -> tuple[Any, Any, Any]:
    """(decision, winner) for one hierarchy position, for drift comparison."""
    per_condition = payload.get("per_condition") or {}
    condition = per_condition.get(position) or {}
    return (condition.get("decision"), condition.get("winner"), position)


def diff_hierarchy(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``hierarchy`` payloads; deterministic.

    Answers "did the instruction-hierarchy adherence story change between
    runs?": the per-position (decision, winner) triplets, the winner-flip
    pattern against the primary run, per-candidate adherence-rate drift,
    and the dedicated-vs-folded group-rate comparison. A changed flip
    pattern after a checkpoint or fixture update is the
    instruction-following regression signal. Candidates present on only
    one side are reported as added/removed. A side without a recorded
    hierarchy probe is reported plainly. No model, no network.
    """
    old_ok = _valid_hierarchy(old)
    new_ok = _valid_hierarchy(new)
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
    old_positions = old.get("positions") or []
    new_positions = new.get("positions") or []
    positions_added = [p for p in new_positions if p not in old_positions]
    positions_removed = [p for p in old_positions if p not in new_positions]
    condition_changed: dict[str, dict[str, Any]] = {}
    for position in sorted(set(old_positions) & set(new_positions)):
        old_triplet = _hierarchy_condition_triplet(old, position)[:2]
        new_triplet = _hierarchy_condition_triplet(new, position)[:2]
        changed = old_triplet != new_triplet
        condition_changed[position] = {
            "old_decision": old_triplet[0],
            "old_winner": old_triplet[1],
            "new_decision": new_triplet[0],
            "new_winner": new_triplet[1],
            "changed": changed,
        }

    def _flip_key(flips: list[Any]) -> list[tuple[Any, ...]]:
        # ``flips`` is a list here: _valid_hierarchy rejects anything else
        # before this point, so no defensive branch is needed.
        """Return the flip key."""
        keys = []
        for entry in flips:
            if type(entry) is not dict:
                continue
            keys.append(
                (
                    entry.get("position"),
                    entry.get("decision_from"),
                    entry.get("winner_from"),
                    entry.get("decision_to"),
                    entry.get("winner_to"),
                )
            )
        return sorted(keys, key=lambda key: tuple(str(part) for part in key))

    flips_changed = _flip_key(old.get("flips")) != _flip_key(new.get("flips"))

    def _candidate_rates(payload: Any) -> dict[str, dict[str, tuple[Any, Any]]]:
        """Return the candidate rates."""
        rates: dict[str, dict[str, tuple[Any, Any]]] = {}
        per_condition = payload.get("per_condition") or {}
        for position in sorted(per_condition):
            candidates = (per_condition.get(position) or {}).get("per_candidate") or {}
            for name, entry in candidates.items():
                entry = entry or {}
                rates.setdefault(name, {})[position] = (
                    entry.get("system_rate"),
                    entry.get("user_rate"),
                )
        return rates

    old_rates = _candidate_rates(old)
    new_rates = _candidate_rates(new)
    candidates_added = sorted(set(new_rates) - set(old_rates))
    candidates_removed = sorted(set(old_rates) - set(new_rates))
    candidate_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_rates) & set(new_rates)):
        for position in sorted(set(old_rates[name]) & set(new_rates[name])):
            old_pair = old_rates[name][position]
            new_pair = new_rates[name][position]
            if old_pair != new_pair:
                candidate_drift.append(
                    {
                        "candidate": name,
                        "position": position,
                        "old_system_rate": old_pair[0],
                        "old_user_rate": old_pair[1],
                        "new_system_rate": new_pair[0],
                        "new_user_rate": new_pair[1],
                    }
                )

    def _group_means(payload: Any) -> dict[str, Any]:
        """Return the group means."""
        groups = payload.get("group_adherence") or {}
        return {
            label: (group or {}).get("mean_system_rate")
            for label, group in groups.items()
            if type(label) is str
        }

    old_groups = _group_means(old)
    new_groups = _group_means(new)
    group_changed = old_groups != new_groups
    result["positions_added"] = positions_added
    result["positions_removed"] = positions_removed
    result["condition_changed"] = condition_changed
    result["flips_changed"] = flips_changed
    result["candidate_drift"] = candidate_drift
    result["candidates_added"] = candidates_added
    result["candidates_removed"] = candidates_removed
    result["group_rates"] = {
        "old": old_groups,
        "new": new_groups,
        "changed": group_changed,
    }
    result["changed"] = bool(
        positions_added
        or positions_removed
        or any(entry["changed"] for entry in condition_changed.values())
        or flips_changed
        or candidate_drift
        or candidates_added
        or candidates_removed
        or group_changed
    )
    return result


def _valid_structured(payload: Any) -> bool:
    """Structural check for a ``structured`` adherence payload.

    Enforces the wave-2 C3 lesson at the value level: compared rate
    fields must be numbers/None in [0, 1] so the renderers never hit a
    TypeError on a hostile payload, and map keys must be strings. Also
    enforces the producer invariants documented by
    :func:`structured.candidate_adherence` -- ``adherent`` implies
    parse-valid and schema-valid, so ``adherent_rate <= parse_rate`` and
    ``schema_rate <= parse_rate`` -- so a hand-corrupted payload never
    renders a self-contradictory table.
    """
    if type(payload) is not dict or payload.get("available") is not True:
        return False
    n_fixtures = payload.get("n_fixtures")
    if type(n_fixtures) is not int or n_fixtures < 0:
        return False
    per_candidate = payload.get("per_candidate")
    if type(per_candidate) is not dict or not all(type(k) is str for k in per_candidate):
        return False
    for entry in per_candidate.values():
        if entry is None:
            continue
        if type(entry) is not dict:
            return False
        rates: dict[str, float | None] = {}
        for key in ("parse_rate", "schema_rate", "adherent_rate"):
            rate = entry.get(key)
            if rate is not None and not (type(rate) in (int, float) and 0.0 <= rate <= 1.0):
                return False
            rates[key] = float(rate) if rate is not None else None
        parse_rate = rates["parse_rate"]
        if parse_rate is not None:
            for key in ("schema_rate", "adherent_rate"):
                other = rates[key]
                if other is not None and other > parse_rate:
                    return False
    weakest = payload.get("weakest_candidates")
    if type(weakest) is not list or not all(type(w) is str for w in weakest):
        return False
    gap = payload.get("adherence_gap")
    if gap is None:
        return True
    return type(gap) in (int, float) and 0.0 <= gap <= 1.0


def diff_structured(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``structured`` adherence payloads; deterministic.

    Answers "did the structured-output adherence story change between
    runs?": per-candidate parse/schema/adherent rate drift, adherence-gap
    drift, weakest-candidate set changes, and added/removed candidates.
    A newly weak template after a template change is the regression
    signal. A side without a recorded probe is reported plainly. No
    model, no network.
    """
    old_ok = _valid_structured(old)
    new_ok = _valid_structured(new)
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
            # _valid_structured guarantees str keys and dict/None cells,
            # so no defensive skip is needed here (it would be dead
            # code and an uncoverable line).
            entry = entry or {}
            view[name] = {
                "parse_rate": entry.get("parse_rate"),
                "schema_rate": entry.get("schema_rate"),
                "adherent_rate": entry.get("adherent_rate"),
            }
        return view

    old_view = _candidate_view(old)
    new_view = _candidate_view(new)
    structured_candidates_added = sorted(set(new_view) - set(old_view))
    structured_candidates_removed = sorted(set(old_view) - set(new_view))
    structured_rate_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_view) & set(new_view)):
        for metric in ("parse_rate", "schema_rate", "adherent_rate"):
            old_value = old_view[name][metric]
            new_value = new_view[name][metric]
            if old_value != new_value:
                structured_rate_drift.append(
                    {
                        "candidate": name,
                        "metric": metric,
                        "old": old_value,
                        "new": new_value,
                    }
                )
    structured_gap_drift: list[dict[str, Any]] = []
    old_gap = old.get("adherence_gap")
    new_gap = new.get("adherence_gap")
    if old_gap != new_gap:
        structured_gap_drift.append({"old": old_gap, "new": new_gap})
    old_weakest = set(old.get("weakest_candidates") or [])
    new_weakest = set(new.get("weakest_candidates") or [])
    structured_weakest_added = sorted(new_weakest - old_weakest)
    structured_weakest_removed = sorted(old_weakest - new_weakest)
    result["rate_drift"] = structured_rate_drift
    result["gap_drift"] = structured_gap_drift
    result["weakest_added"] = structured_weakest_added
    result["weakest_removed"] = structured_weakest_removed
    result["candidates_added"] = structured_candidates_added
    result["candidates_removed"] = structured_candidates_removed
    # NOTE: structured-specific disjunct names keep this ``changed``
    # line textually distinct from every other diff section's anchor
    # (mutation-gate uniqueness).
    result["changed"] = bool(
        structured_rate_drift
        or structured_gap_drift
        or structured_weakest_added
        or structured_weakest_removed
        or structured_candidates_added
        or structured_candidates_removed
    )
    return result


def _valid_faketurn(payload: Any) -> bool:
    """Structural check for a ``faketurn`` history-tampering payload.

    Enforces the wave-2 C3 lesson at the value level: flip rates must
    be numbers/None in [0, 1], transfer gaps numbers/None in [-1, 1],
    token sources exactly "same"/"cross", and map keys strings, so
    the renderers never hit a TypeError on a hostile payload.
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
            if cell.get("token_source") not in ("same", "cross"):
                return False
            n_evaluable = cell.get("n_evaluable")
            if type(n_evaluable) is not int or n_evaluable < 0:
                return False
            rate = cell.get("flip_rate")
            if rate is not None and not (type(rate) in (int, float) and 0.0 <= rate <= 1.0):
                return False
        for key in ("same_flip_rate", "cross_flip_rate", "max_flip_rate"):
            rate = entry.get(key)
            if rate is not None and not (type(rate) in (int, float) and 0.0 <= rate <= 1.0):
                return False
        gap = entry.get("transfer_gap")
        if gap is not None and not (type(gap) in (int, float) and -1.0 <= gap <= 1.0):
            return False
    excluded = payload.get("excluded_candidates")
    return type(excluded) is list and all(type(c) is str for c in excluded)


def diff_faketurn(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``faketurn`` history-tampering payloads; deterministic.

    Answers "did the fake-turn tampering story change between runs?":
    per-candidate per-cell flip-rate drift, transfer-gap drift,
    provenance (same/cross) changes per cell, added/removed
    candidates, and changes in the excluded set. A newly positive
    cross-minus-same gap after a template change is the regression
    signal -- foreign delimiters gaining tampering authority they did
    not have before. A side without a recorded probe is reported
    plainly. No model, no network.
    """
    old_ok = _valid_faketurn(old)
    new_ok = _valid_faketurn(new)
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
            # _valid_faketurn guarantees str keys and dict entries, so
            # no defensive skip is needed here (it would be dead code
            # and an uncoverable line).
            cells: dict[str, Any] = {}
            for key, cell in (entry.get("cells") or {}).items():
                if type(cell) is dict and cell.get("supported") is True:
                    cells[key] = {
                        "flip_rate": cell.get("flip_rate"),
                        "token_source": cell.get("token_source"),
                    }
                else:
                    cells[key] = None
            view[name] = {
                "cells": cells,
                "transfer_gap": entry.get("transfer_gap"),
                "max_flip_rate": entry.get("max_flip_rate"),
                "excluded": name in set(payload.get("excluded_candidates") or []),
            }
        return view

    old_view = _candidate_view(old)
    new_view = _candidate_view(new)
    faketurn_candidates_added = sorted(set(new_view) - set(old_view))
    faketurn_candidates_removed = sorted(set(old_view) - set(new_view))
    faketurn_cell_drift: list[dict[str, Any]] = []
    faketurn_source_drift: list[dict[str, Any]] = []
    faketurn_gap_drift: list[dict[str, Any]] = []
    faketurn_top_rate_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_view) & set(new_view)):
        old_cells = old_view[name]["cells"]
        new_cells = new_view[name]["cells"]
        for key in sorted(set(old_cells) & set(new_cells)):
            old_cell = old_cells[key]
            new_cell = new_cells[key]
            if old_cell is None or new_cell is None:
                if old_cell != new_cell:
                    faketurn_source_drift.append(
                        {
                            "candidate": name,
                            "cell": key,
                            "old": "supported" if old_cell is not None else "unsupported",
                            "new": "supported" if new_cell is not None else "unsupported",
                        }
                    )
                continue
            if old_cell["flip_rate"] != new_cell["flip_rate"]:
                faketurn_cell_drift.append(
                    {
                        "candidate": name,
                        "cell": key,
                        "old": old_cell["flip_rate"],
                        "new": new_cell["flip_rate"],
                    }
                )
            if old_cell["token_source"] != new_cell["token_source"]:
                faketurn_source_drift.append(
                    {
                        "candidate": name,
                        "cell": key,
                        "old": old_cell["token_source"],
                        "new": new_cell["token_source"],
                    }
                )
        old_gap = old_view[name]["transfer_gap"]
        new_gap = new_view[name]["transfer_gap"]
        if old_gap != new_gap:
            faketurn_gap_drift.append({"candidate": name, "old": old_gap, "new": new_gap})
        old_top = old_view[name]["max_flip_rate"]
        new_top = new_view[name]["max_flip_rate"]
        if old_top != new_top:
            faketurn_top_rate_drift.append({"candidate": name, "old": old_top, "new": new_top})
    old_excluded = {name for name, v in old_view.items() if v["excluded"]}
    new_excluded = {name for name, v in new_view.items() if v["excluded"]}
    faketurn_excluded_added = sorted(new_excluded - old_excluded)
    faketurn_excluded_removed = sorted(old_excluded - new_excluded)
    result["cell_drift"] = faketurn_cell_drift
    result["gap_drift"] = faketurn_gap_drift
    result["top_rate_drift"] = faketurn_top_rate_drift
    result["source_drift"] = faketurn_source_drift
    result["candidates_added"] = faketurn_candidates_added
    result["candidates_removed"] = faketurn_candidates_removed
    result["excluded_added"] = faketurn_excluded_added
    result["excluded_removed"] = faketurn_excluded_removed
    # NOTE: faketurn-specific disjunct names keep this ``changed``
    # line textually distinct from every other diff section's anchor
    # (mutation-gate uniqueness).
    result["changed"] = bool(
        faketurn_cell_drift
        or faketurn_gap_drift
        or faketurn_top_rate_drift
        or faketurn_source_drift
        or faketurn_candidates_added
        or faketurn_candidates_removed
        or faketurn_excluded_added
        or faketurn_excluded_removed
    )
    return result


def _valid_toolcall(payload: Any) -> bool:
    """Structural check for a ``toolcall`` adherence payload.

    Enforces the wave-2 C3 lesson at the value level: the four rates
    must be numbers/None in [0, 1], the outcome counts non-negative
    ints, and map keys strings, so the renderers never hit a TypeError
    on a hostile payload.
    """
    if type(payload) is not dict or payload.get("available") is not True:
        return False
    n_fixtures = payload.get("n_fixtures")
    if type(n_fixtures) is not int or n_fixtures < 0:
        return False
    per_candidate = payload.get("per_candidate")
    if type(per_candidate) is not dict or not all(type(k) is str for k in per_candidate):
        return False
    for entry in per_candidate.values():
        if entry is None:
            continue
        if type(entry) is not dict:
            return False
        for key in ("n", "n_evaluable", "adherent", "parse_failed", "call_failed", "not_text"):
            value = entry.get(key)
            if type(value) is not int or value < 0:
                return False
        if entry["n_evaluable"] > entry["n"]:
            return False
        for key in ("selection_rate", "args_rate", "relevance_rate", "adherent_rate"):
            rate = entry.get(key)
            if rate is not None and not (type(rate) in (int, float) and 0.0 <= rate <= 1.0):
                return False
    excluded = payload.get("excluded_candidates")
    if type(excluded) is not list or not all(type(c) is str for c in excluded):
        return False
    weakest = payload.get("weakest_candidates")
    return type(weakest) is list and all(type(c) is str for c in weakest)


def diff_toolcall(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``toolcall`` adherence payloads; deterministic.

    Answers "did the tool-call adherence story change between runs?":
    per-candidate selection-rate drift, argument-rate drift, and
    relevance-rate drift, plus added/removed candidates and changes in
    the excluded set. A newly positive selection gap after a template
    change is the regression signal -- the template deciding which
    tool gets picked. A side without a recorded probe is reported
    plainly. No model, no network.
    """
    old_ok = _valid_toolcall(old)
    new_ok = _valid_toolcall(new)
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
            # _valid_toolcall guarantees str keys and dict/None entries,
            # so no defensive skip is needed here (it would be dead code
            # and an uncoverable line).
            if entry is None:
                view[name] = {
                    "selection_rate": None,
                    "args_rate": None,
                    "relevance_rate": None,
                    "excluded": True,
                }
            else:
                view[name] = {
                    "selection_rate": entry.get("selection_rate"),
                    "args_rate": entry.get("args_rate"),
                    "relevance_rate": entry.get("relevance_rate"),
                    "excluded": name in set(payload.get("excluded_candidates") or []),
                }
        return view

    old_view = _candidate_view(old)
    new_view = _candidate_view(new)
    toolcall_candidates_added = sorted(set(new_view) - set(old_view))
    toolcall_candidates_removed = sorted(set(old_view) - set(new_view))
    toolcall_selection_drift: list[dict[str, Any]] = []
    toolcall_args_drift: list[dict[str, Any]] = []
    toolcall_relevance_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_view) & set(new_view)):
        old_entry = old_view[name]
        new_entry = new_view[name]
        for rate_key, bucket in (
            ("selection_rate", toolcall_selection_drift),
            ("args_rate", toolcall_args_drift),
            ("relevance_rate", toolcall_relevance_drift),
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
    toolcall_excluded_added = sorted(new_excluded - old_excluded)
    toolcall_excluded_removed = sorted(old_excluded - new_excluded)
    result["selection_drift"] = toolcall_selection_drift
    result["args_drift"] = toolcall_args_drift
    result["relevance_drift"] = toolcall_relevance_drift
    result["candidates_added"] = toolcall_candidates_added
    result["candidates_removed"] = toolcall_candidates_removed
    result["excluded_added"] = toolcall_excluded_added
    result["excluded_removed"] = toolcall_excluded_removed
    # NOTE: toolcall-specific disjunct names keep this ``changed``
    # line textually distinct from every other diff section's anchor
    # (mutation-gate uniqueness).
    result["changed"] = bool(
        toolcall_selection_drift
        or toolcall_args_drift
        or toolcall_relevance_drift
        or toolcall_candidates_added
        or toolcall_candidates_removed
        or toolcall_excluded_added
        or toolcall_excluded_removed
    )
    return result


def _aliasing_recompute(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Re-derive every aliasing aggregate from the ground records.

    Independent code (the cycle-67 gate harness loads this module
    standalone): cohorts, classes, counts, pairs, suspect cells and
    the flag are all recomputed from the per-cell digest records,
    so a payload cannot manufacture a class structure or a fired
    flag its records do not support. This is an internal-
    consistency recompute -- it detects aggregates inconsistent
    with the ground records; it cannot detect fabricated ground
    records, the standing limitation of every house ground layer.
    Returns the recomputed structure, or None on any inconsistency
    with the payload's own fields (the caller compares).
    """
    candidates = payload.get("candidates")
    if type(candidates) is not list or not candidates:
        return None
    if payload.get("n_candidates_recorded") != len(candidates):
        return None
    available: list[dict[str, Any]] = []
    total_cells = 0
    seen_names: set[str] = set()
    for entry in candidates:
        if type(entry) is not dict:
            return None
        name = entry.get("candidate")
        if name is not None and type(name) is not str:
            return None
        if type(name) is str:
            if name in seen_names:
                return None
            seen_names.add(name)
        if type(entry.get("available")) is not bool:
            return None
        if entry.get("available") is not True:
            if type(entry.get("detail")) is not str:
                return None
            continue
        available.append(entry)
        if type(entry.get("eligible")) is not bool:
            return None
        planned = entry.get("planned")
        if planned is not None and (type(planned) is not int or planned < 0):
            return None
        boundary = entry.get("boundary_token_ids")
        if (
            type(boundary) is not list
            or not boundary
            or any(type(t) is not int or t < 0 for t in boundary)
        ):
            return None
        records = entry.get("records")
        if type(records) is not list or entry.get("n_observations") != len(records):
            return None
        total_cells += len(records)
        fixtures_seen: set[str] = set()
        n_completed = 0
        n_evaluable = 0
        for record in records:
            if type(record) is not dict:
                return None
            fixture = record.get("fixture")
            if type(fixture) is not str or not fixture or fixture in fixtures_seen:
                return None
            fixtures_seen.add(fixture)
            if type(record.get("status")) is not str:
                return None
            for field in ("evaluable", "exact_match", "boundary_found"):
                if type(record.get(field)) is not bool:
                    return None
            btm = record.get("boundary_text_match")
            if btm is not None and type(btm) is not bool:
                return None
            if record["status"] == "completed":
                n_completed += 1
            if record["evaluable"] is True:
                n_evaluable += 1
                if record["status"] != "completed":
                    return None
                if not _is_hex_digest(record.get("input_sha256")):
                    return None
                if not _is_hex_digest(record.get("output_sha256")):
                    return None
            else:
                if record["status"] == "completed":
                    return None
                if (
                    record.get("input_sha256") is not None
                    or record.get("output_sha256") is not None
                ):
                    return None
        if entry.get("n_completed") != n_completed:
            return None
        if entry.get("n_not_completed") != len(records) - n_completed:
            return None
        if entry.get("n_evaluable") != n_evaluable:
            return None
    if total_cells > 2400 or len(available) > 24:
        return None
    if payload.get("n_candidates_available") != len(available):
        return None

    def _fixtures(entry: dict[str, Any]) -> frozenset[str]:
        """Return the fixtures."""
        return frozenset(r["fixture"] for r in entry["records"] if r["evaluable"])

    mismatch = len({_fixtures(e) for e in available}) > 1
    groups: dict[frozenset[str], list[dict[str, Any]]] = {}
    for entry in available:
        groups.setdefault(_fixtures(entry), []).append(entry)

    def _rank(item: tuple[frozenset[str], list[dict[str, Any]]]) -> tuple[Any, ...]:
        """Return the rank."""
        fixtures, members = item
        return (
            -len(members),
            -len(fixtures),
            tuple(sorted(fixtures)),
            tuple(sorted(str(m.get("candidate")) for m in members)),
        )

    primary = sorted(groups.items(), key=_rank)[0][1] if groups else []
    primary_names = {str(e.get("candidate")) for e in primary}
    fixtures_sorted = sorted(_fixtures(primary[0])) if primary else []

    def _classes(key_with_boundary: bool) -> list[list[str]]:
        """Return the classes."""
        grouped: dict[tuple[Any, ...], list[str]] = {}
        for entry in primary:
            cells = {r["fixture"]: r for r in entry["records"] if r["evaluable"]}
            sig = tuple(cells[f]["input_sha256"] for f in fixtures_sorted)
            if key_with_boundary:
                sig = sig + (tuple(entry["boundary_token_ids"]),)
            grouped.setdefault(sig, []).append(str(entry.get("candidate")))
        return sorted((sorted(v) for v in grouped.values()), key=lambda names: names[0])

    render_classes = _classes(False)
    measurement_classes = _classes(True)
    redundant_measurement = 0
    redundant_render = 0
    for fixture in fixtures_sorted:
        m_tuples: set[tuple[Any, ...]] = set()
        r_tuples: set[Any] = set()
        for entry in primary:
            cell = next(
                (r for r in entry["records"] if r["evaluable"] and r["fixture"] == fixture),
                None,
            )
            if cell is None:  # cohort members share one evaluable fixture set
                raise ValueError("cohort member lacks an evaluable record for a cohort fixture")
            r_tuples.add(cell["input_sha256"])
            m_tuples.add((cell["input_sha256"], tuple(entry["boundary_token_ids"])))
        redundant_measurement += len(primary) - len(m_tuples)
        redundant_render += len(primary) - len(r_tuples)
    by_name = {str(e.get("candidate")): e for e in primary}
    suspects: list[dict[str, Any]] = []
    for names in measurement_classes:
        if len(names) < 2:
            continue
        for fixture in fixtures_sorted:
            outputs: set[Any] = set()
            outcomes: set[tuple[Any, ...]] = set()
            for name in names:
                cell = next(
                    (
                        r
                        for r in by_name[name]["records"]
                        if r["evaluable"] and r["fixture"] == fixture
                    ),
                    None,
                )
                if cell is None:  # cohort members share one evaluable fixture set
                    raise ValueError("cohort member lacks an evaluable record for a cohort fixture")
                outputs.add(cell["output_sha256"])
                outcomes.add(
                    (
                        cell["exact_match"],
                        cell["boundary_found"],
                        cell["boundary_text_match"],
                        cell["status"],
                    )
                )
            out_bad = len(outputs) > 1
            oc_bad = len(outcomes) > 1
            if out_bad or oc_bad:
                kind = "both" if out_bad and oc_bad else ("output" if out_bad else "outcome")
                suspects.append({"fixture": fixture, "candidates": list(names), "kind": kind})
    suspects.sort(key=lambda c: (c["fixture"], c["candidates"]))
    pairs: list[dict[str, Any]] = []
    for i, left in enumerate(available):
        left_cells = {r["fixture"]: r["input_sha256"] for r in left["records"] if r["evaluable"]}
        for right in available[i + 1 :]:
            right_cells = {
                r["fixture"]: r["input_sha256"] for r in right["records"] if r["evaluable"]
            }
            common = sorted(set(left_cells) & set(right_cells))
            if not common:
                continue
            shared = [f for f in common if left_cells[f] == right_cells[f]]
            pairs.append(
                {
                    "candidates": sorted([str(left.get("candidate")), str(right.get("candidate"))]),
                    "n_common_completed": len(common),
                    "n_shared": len(shared),
                    "shared_fixture_ids": shared,
                }
            )
    pairs.sort(key=lambda p: p["candidates"])
    passing = sorted(
        str(e.get("candidate"))
        for e in primary
        if e.get("eligible") is True
        and e.get("n_evaluable", 0) > 0
        and all(r["exact_match"] and r["boundary_found"] for r in e["records"] if r["evaluable"])
    )
    # The flag, recomputed in the producer's priority order.
    flag = False
    flag_detail = ""
    flag_members: list[str] = []
    if len(available) != len(candidates):
        flag_detail = "candidate_unavailable"
    elif payload.get("decision") != "ambiguous":
        flag_detail = "decision_not_ambiguous"
    elif len(passing) < 2:
        flag_detail = "passing_set_inconsistent"
    else:
        passing_set = set(passing)
        containing = [cls for cls in measurement_classes if passing_set <= set(cls)]
        if containing:
            cls = containing[0]
            if any(
                cell["kind"] in ("outcome", "both") and set(cell["candidates"]) == set(cls)
                for cell in suspects
            ):
                flag_detail = "determinism_suspect_in_class"
            else:
                flag = True
                flag_members = list(cls)
        elif not any(passing_set <= set(cls) for cls in render_classes):
            flag_detail = "passing_spans_classes"
        else:
            flag_detail = "boundaries_differ_within_passing_set"
    # A lone-surrogate fixture id cannot be UTF-8 encoded; the
    # recompute fails closed instead of raising (bug-hunt wave 2,
    # F6; the cycle-190 discipline, applied at every join-encode
    # digest site).
    try:
        fixture_set_sha256 = hashlib.sha256("\n".join(fixtures_sorted).encode("utf-8")).hexdigest()
    except UnicodeEncodeError:
        return None
    return {
        "evaluable_mismatch": mismatch,
        "primary_names": primary_names,
        "render_classes": render_classes,
        "measurement_classes": measurement_classes,
        "n_redundant_completed_observations": redundant_measurement,
        "n_redundant_render_observations": redundant_render,
        "determinism_suspect_cells": suspects,
        "pairs": pairs,
        "passing_candidates": passing,
        "alias_suspect_abstention": flag,
        "alias_suspect_detail": flag_detail,
        "alias_suspect_class_members": flag_members,
        "fixture_set_sha256": fixture_set_sha256,
        "excluded_candidates": sorted(
            str(e.get("candidate"))
            for e in available
            if str(e.get("candidate")) not in primary_names
        ),
    }


def _valid_aliasing(payload: Any) -> bool:
    """Structural check for an ``aliasing`` audit payload.

    The full recompute standard: every aggregate -- cohorts,
    render and measurement classes, both effective counts, the
    redundant counts, the pair matrix, the determinism-suspect
    cells, the passing set, and the alias-suspect flag including
    its withholding reason in the producer's priority order -- is
    re-derived from the per-cell digest ground records with
    independent code, so a payload cannot manufacture a class
    structure or a fired flag its records do not support. Digest
    handling is version-agnostic (equality only): a coherent
    foreign digest version validates internally and is reported
    as protocol drift by the diff.
    """
    if type(payload) is not dict or payload.get("available") is not True:
        return False
    method = payload.get("method")
    if type(method) is not dict:
        return False
    for key in ("protocol", "token_ids_digest", "scope"):
        if type(method.get(key)) is not str or not method.get(key):
            return False
    for key in ("decision", "selected_candidate", "backend_fingerprint"):
        value = payload.get(key)
        if value is not None and type(value) is not str:
            return False
    for key in ("seed", "max_new_tokens"):
        value = payload.get(key)
        if value is not None and type(value) is not int:
            return False
    if not _is_hex_digest(payload.get("fixture_set_sha256")):
        return False
    if type(payload.get("evaluable_mismatch")) is not bool:
        return False
    recomputed = _aliasing_recompute(payload)
    if recomputed is None:
        return False
    candidates = payload["candidates"]
    for entry in candidates:
        if entry.get("available") is not True:
            continue
        name = str(entry.get("candidate"))
        if entry.get("in_primary_cohort") is not (name in recomputed["primary_names"]):
            return False
        expected_render = next((cls for cls in recomputed["render_classes"] if name in cls), None)
        expected_measurement = next(
            (cls for cls in recomputed["measurement_classes"] if name in cls), None
        )
        if entry.get("render_class_members") != expected_render:
            return False
        if entry.get("measurement_class_members") != expected_measurement:
            return False
    checks = (
        ("evaluable_mismatch", recomputed["evaluable_mismatch"]),
        ("fixture_set_sha256", recomputed["fixture_set_sha256"]),
        ("excluded_candidates", recomputed["excluded_candidates"]),
        ("n_candidates_excluded", len(recomputed["excluded_candidates"])),
        ("n_primary_cohort", len(recomputed["primary_names"])),
        ("n_effective_render", len(recomputed["render_classes"])),
        ("n_effective_measurement", len(recomputed["measurement_classes"])),
        ("render_classes", recomputed["render_classes"]),
        ("measurement_classes", recomputed["measurement_classes"]),
        (
            "n_redundant_completed_observations",
            recomputed["n_redundant_completed_observations"],
        ),
        ("n_redundant_render_observations", recomputed["n_redundant_render_observations"]),
        ("determinism_suspect_cells", recomputed["determinism_suspect_cells"]),
        ("n_determinism_suspect_cells", len(recomputed["determinism_suspect_cells"])),
        (
            "n_outcome_suspect_cells",
            sum(
                1
                for c in recomputed["determinism_suspect_cells"]
                if c["kind"] in ("outcome", "both")
            ),
        ),
        ("pairs", recomputed["pairs"]),
        ("passing_candidates", recomputed["passing_candidates"]),
        ("alias_suspect_abstention", recomputed["alias_suspect_abstention"]),
        ("alias_suspect_detail", recomputed["alias_suspect_detail"]),
        ("alias_suspect_class_members", recomputed["alias_suspect_class_members"]),
    )
    return all(payload.get(key) == expected for key, expected in checks)


def diff_aliasing(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``aliasing`` audit payloads; deterministic.

    Answers "did the candidate-set integrity story change between
    runs?": measurement-partition drift over the common candidates
    (pairs joined / pairs split), per-candidate boundary drift,
    effective-count and suspect-count drift, the alias-suspect
    flag, candidates added/removed/excluded, and an
    evaluable-mismatch change. Classes are relative to the fixture
    set, the backend, and the digest definition, so any of those
    drifting flags the comparison ``comparable: False`` with the
    drift still listed. ``backend_drift`` means the recorded
    backend fingerprints differ -- tokenizer identity *not proven
    equal* (the fingerprint also binds device, context limit and
    dependency versions), never "a different tokenizer". A side
    without a recorded audit is reported plainly. No model, no
    network.
    """
    old_ok = _valid_aliasing(old)
    new_ok = _valid_aliasing(new)
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
    protocol_drift = (old["method"].get("protocol"), old["method"].get("token_ids_digest")) != (
        new["method"].get("protocol"),
        new["method"].get("token_ids_digest"),
    )
    fixture_set_drift = old.get("fixture_set_sha256") != new.get("fixture_set_sha256")
    backend_drift = old.get("backend_fingerprint") != new.get("backend_fingerprint")
    comparable = not protocol_drift and not fixture_set_drift and not backend_drift

    def _candidate_view(payload: Any) -> dict[str, dict[str, Any]]:
        """Return the candidate view."""
        view: dict[str, dict[str, Any]] = {}
        for entry in payload.get("candidates") or []:
            # _valid_aliasing guarantees the entry shape, so no
            # defensive skip is needed here.
            name = entry.get("candidate")
            if type(name) is not str:
                continue
            if entry.get("available") is not True or entry.get("in_primary_cohort") is not True:
                view[name] = {"excluded": True}
                continue
            view[name] = {
                "excluded": False,
                "members": entry.get("measurement_class_members") or [],
                "boundary": entry.get("boundary_token_ids"),
            }
        return view

    old_view = _candidate_view(old)
    new_view = _candidate_view(new)
    aliasing_candidates_added = sorted(set(new_view) - set(old_view))
    aliasing_candidates_removed = sorted(set(old_view) - set(new_view))
    old_excluded = {name for name, v in old_view.items() if v["excluded"]}
    new_excluded = {name for name, v in new_view.items() if v["excluded"]}
    aliasing_excluded_added = sorted(new_excluded - old_excluded)
    aliasing_excluded_removed = sorted(old_excluded - new_excluded)
    common = sorted(
        name
        for name in set(old_view) & set(new_view)
        if not old_view[name]["excluded"] and not new_view[name]["excluded"]
    )
    aliasing_pairs_joined: list[list[str]] = []
    aliasing_pairs_split: list[list[str]] = []
    aliasing_boundary_drift: list[dict[str, Any]] = []
    for i, left in enumerate(common):
        if old_view[left]["boundary"] != new_view[left]["boundary"]:
            aliasing_boundary_drift.append(
                {
                    "candidate": left,
                    "old_boundary": old_view[left]["boundary"],
                    "new_boundary": new_view[left]["boundary"],
                }
            )
        for right in common[i + 1 :]:
            old_together = right in old_view[left]["members"]
            new_together = right in new_view[left]["members"]
            if new_together and not old_together:
                aliasing_pairs_joined.append([left, right])
            elif old_together and not new_together:
                aliasing_pairs_split.append([left, right])
    aliasing_count_drift: list[dict[str, Any]] = []
    for metric in (
        "n_effective_render",
        "n_effective_measurement",
        "n_redundant_completed_observations",
        "n_determinism_suspect_cells",
    ):
        if old.get(metric) != new.get(metric):
            aliasing_count_drift.append(
                {"metric": metric, "old": old.get(metric), "new": new.get(metric)}
            )
    aliasing_flag_changed = old.get("alias_suspect_abstention") != new.get(
        "alias_suspect_abstention"
    )
    aliasing_mismatch_changed = old.get("evaluable_mismatch") != new.get("evaluable_mismatch")
    result["pairs_joined"] = aliasing_pairs_joined
    result["pairs_split"] = aliasing_pairs_split
    result["boundary_drift"] = aliasing_boundary_drift
    result["count_drift"] = aliasing_count_drift
    result["flag_changed"] = aliasing_flag_changed
    result["candidates_added"] = aliasing_candidates_added
    result["candidates_removed"] = aliasing_candidates_removed
    result["excluded_added"] = aliasing_excluded_added
    result["excluded_removed"] = aliasing_excluded_removed
    result["protocol_drift"] = protocol_drift
    result["fixture_set_drift"] = fixture_set_drift
    result["backend_drift"] = backend_drift
    result["comparable"] = comparable
    result["mismatch_changed"] = aliasing_mismatch_changed
    # NOTE: aliasing-specific disjunct names keep this ``changed``
    # line textually distinct from every other diff section's anchor
    # (mutation-gate uniqueness).
    result["changed"] = bool(
        aliasing_pairs_joined
        or aliasing_pairs_split
        or aliasing_boundary_drift
        or aliasing_count_drift
        or aliasing_flag_changed
        or aliasing_candidates_added
        or aliasing_candidates_removed
        or aliasing_excluded_added
        or aliasing_excluded_removed
        or aliasing_mismatch_changed
        or protocol_drift
        or fixture_set_drift
        or backend_drift
    )
    return result


_REPETITION_ARM_DETAILS = frozenset(
    {
        "",
        "vacuous",
        "arm_context_exceeded",
        "arm_evaluation_failed",
        "pad_screened",
        "pad_unmatchable",
    }
)


_REPETITION_CONTRASTS = (
    ("x2", "x1", "raw_gain"),
    ("pad", "x1", "length_effect"),
    ("x2", "pad", "repetition_effect"),
)


def _repetition_recompute(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Re-derive every repetition aggregate from the ground records.

    The full recompute standard (post-cycle-177): per-candidate
    paired contrasts, mismatch flags, boundary rates, pad shares,
    delta signs, the per-arm selection readout with its withholding
    rule, and the think-group means are all recomputed here with
    independent code, so a payload cannot publish an effect, a
    winner, or a flip its per-fixture records do not support. The
    protocol fingerprint is rebuilt from the payload's own protocol
    components and the record fixture ids — internal consistency,
    not verifiability against fabricated records (the aliasing
    standard). Returns None on any structural defect.
    """

    protocol = payload.get("protocol")
    if type(protocol) is not dict:
        return None
    for key in (
        "joiner",
        "filler_unit_sha256",
        "sizing_rule",
        "fixture_set_sha256",
        "backend_fingerprint",
        "fingerprint",
    ):
        if type(protocol.get(key)) is not str or not protocol.get(key):
            return None
    for key in ("version", "detector_version"):
        if type(protocol.get(key)) is not int:
            return None
    candidates = payload.get("candidates")
    if type(candidates) is not dict or not candidates:
        return None
    total_records = 0
    fixture_ids: list[str] | None = None
    per_candidate: dict[str, Any] = {}
    for name, entry in candidates.items():
        if type(name) is not str or type(entry) is not dict:
            return None
        if entry.get("status") not in ("available", "unavailable"):
            return None
        if type(entry.get("think_channel")) is not bool:
            return None
        signature = entry.get("render_signature")
        if signature is not None and type(signature) is not str:
            return None
        if entry["status"] == "unavailable":
            if entry.get("records") != []:
                return None
            per_candidate[name] = None
            continue
        records = entry.get("records")
        if type(records) is not list or not records:
            return None
        total_records += len(records)
        ids: list[str] = []
        for record in records:
            if type(record) is not dict or type(record.get("fixture")) is not str:
                return None
            ids.append(record["fixture"])
        if len(set(ids)) != len(ids):
            return None
        if fixture_ids is None:
            fixture_ids = sorted(ids)
        elif sorted(ids) != fixture_ids:
            return None
        for r in records:
            if type(r.get("vacuous")) is not bool or type(r.get("pre_repeated")) is not bool:
                return None
            if type(r.get("echo_fixture")) is not bool or type(r.get("pad_evaluable")) is not bool:
                return None
            if r.get("construction") not in ("id_exact", "text_level", "vacuous"):
                return None
            if r["vacuous"] != (r["construction"] == "vacuous"):
                return None
            seam = r.get("seam_ids")
            if type(seam) is not list or any(type(s) is not int for s in seam):
                return None
            delta = r.get("delta_tokens")
            if delta is not None and type(delta) is not int:
                return None
            filler = r.get("filler_tokens")
            if filler is not None and (type(filler) is not int or filler <= 0):
                return None
            arms = r.get("arms")
            details = r.get("arm_details")
            if type(arms) is not dict or type(details) is not dict:
                return None
            for arm in ("x1", "x2", "pad"):
                cell = arms.get(arm, "missing")
                if cell == "missing":
                    return None
                if cell is not None:
                    if type(cell) is not dict:
                        return None
                    if (
                        type(cell.get("verdict")) is not bool
                        or type(cell.get("boundary_found")) is not bool
                    ):
                        return None
                    btm = cell.get("boundary_text_match")
                    if btm is not None and type(btm) is not bool:
                        return None
                    tokens = cell.get("input_tokens")
                    if type(tokens) is not int or tokens < 0:
                        return None
            if arms["x1"] is None:
                return None
            for arm in ("x2", "pad"):
                detail = details.get(arm)
                if type(detail) is not str or detail not in _REPETITION_ARM_DETAILS:
                    return None
                if r["vacuous"]:
                    if arms[arm] is not None or detail != "vacuous":
                        return None
                elif arms[arm] is None:
                    if detail not in (
                        "arm_context_exceeded",
                        "arm_evaluation_failed",
                        "pad_screened",
                        "pad_unmatchable",
                    ):
                        return None
                elif detail != "":
                    return None
            if r["pad_evaluable"]:
                if filler is None:
                    return None
            elif arms["pad"] is not None:
                return None
        per_candidate[name] = records
    if total_records > 2400 or fixture_ids is None:
        return None
    try:
        fixture_hash = hashlib.sha256("\n".join(fixture_ids).encode("utf-8")).hexdigest()
    except UnicodeEncodeError:
        return None  # lone-surrogate fixture id: fail closed (F6)
    if protocol["fixture_set_sha256"] != fixture_hash:
        return None
    table = {
        "version": protocol["version"],
        "joiner": protocol["joiner"],
        "filler_unit_sha256": protocol["filler_unit_sha256"],
        "detector_version": protocol["detector_version"],
        "sizing_rule": protocol["sizing_rule"],
        "fixture_set_sha256": protocol["fixture_set_sha256"],
        "backend_fingerprint": protocol["backend_fingerprint"],
    }
    canonical = json.dumps(table, sort_keys=True, separators=(",", ":"))
    if protocol["fingerprint"] != hashlib.sha256(canonical.encode("utf-8")).hexdigest():
        return None

    def _verdict(record: dict[str, Any], arm: str) -> bool | None:
        """Return the verdict."""
        cell = record["arms"][arm]
        return None if cell is None else bool(cell["verdict"])

    def _contrast(
        recs: list[dict[str, Any]], arm_a: str, arm_b: str, effect: str
    ) -> dict[str, Any]:
        """Return the contrast."""
        paired = [
            r
            for r in recs
            if not r["vacuous"]
            and _verdict(r, arm_a) is not None
            and _verdict(r, arm_b) is not None
        ]
        n = len(paired)
        if n == 0:
            return {"n_paired": 0, f"rate_{arm_a}": None, f"rate_{arm_b}": None, effect: None}
        rate_a = sum(1 for r in paired if _verdict(r, arm_a)) / n
        rate_b = sum(1 for r in paired if _verdict(r, arm_b)) / n
        return {
            "n_paired": n,
            f"rate_{arm_a}": rate_a,
            f"rate_{arm_b}": rate_b,
            effect: rate_a - rate_b,
        }

    aggregates: dict[str, Any] = {}
    for name, records in per_candidate.items():
        if records is None:
            continue
        contrasts: dict[str, Any] = {}
        mismatches: dict[str, bool] = {}
        for arm_a, arm_b, effect in _REPETITION_CONTRASTS:
            key = f"{arm_a}_vs_{arm_b}"
            contrasts[key] = _contrast(records, arm_a, arm_b, effect)
            set_a = {
                r["fixture"] for r in records if not r["vacuous"] and _verdict(r, arm_a) is not None
            }
            set_b = {
                r["fixture"] for r in records if not r["vacuous"] and _verdict(r, arm_b) is not None
            }
            mismatches[key] = set_a != set_b
        clean = [r for r in records if not r["pre_repeated"]]
        contrasts_excl = {
            f"{a}_vs_{b}": _contrast(clean, a, b, e) for a, b, e in _REPETITION_CONTRASTS
        }
        boundary_rates: dict[str, Any] = {}
        for arm in ("x1", "x2", "pad"):
            cells = [r["arms"][arm] for r in records if r["arms"][arm] is not None]
            boundary_rates[arm] = (
                sum(1 for c in cells if c["boundary_found"]) / len(cells) if cells else None
            )
        signs = {"negative": 0, "zero": 0, "positive": 0}
        for r in records:
            d = r["delta_tokens"]
            if d is None:
                continue
            signs["negative" if d < 0 else "zero" if d == 0 else "positive"] += 1
        n_pad = sum(1 for r in records if r["pad_evaluable"])
        aggregates[name] = {
            "n_fixtures": len(records),
            "n_vacuous": sum(1 for r in records if r["vacuous"]),
            "n_pre_repeated": sum(1 for r in records if r["pre_repeated"]),
            "n_echo": sum(1 for r in records if r["echo_fixture"]),
            "n_pad_evaluable": n_pad,
            "pad_evaluable_share": n_pad / len(records),
            "delta_sign_counts": signs,
            "contrasts": contrasts,
            "contrasts_excluding_pre_repeated": contrasts_excl,
            "evaluable_mismatch": mismatches,
            "boundary_found_rates": boundary_rates,
        }
    primary_decision = payload.get("primary_decision")
    primary_winner = payload.get("primary_winner")
    mapped: tuple[Any, Any]
    if primary_decision in ("compatible_on_fixture_set", "held_out_failed"):
        mapped = ("selected", primary_winner)
    else:
        mapped = (primary_decision, primary_winner)
    arms_out: dict[str, Any] = {}
    for arm in ("x1", "x2", "pad"):
        withheld = False
        all_pass: dict[str, bool | None] = {}
        completed_sets: list[frozenset[str]] = []
        for name, records in per_candidate.items():
            if records is None:
                withheld = True
                all_pass[name] = None
                continue
            verdicts = [
                _verdict(r, arm)
                for r in records
                if not r["vacuous"] and _verdict(r, arm) is not None
            ]
            all_pass[name] = all(verdicts) if verdicts else None
            if all_pass[name] is None:
                withheld = True
            completed_sets.append(
                frozenset(
                    r["fixture"]
                    for r in records
                    if not r["vacuous"] and _verdict(r, arm) is not None
                )
            )
        if len(set(completed_sets)) > 1:
            withheld = True
        if withheld:
            arms_out[arm] = {
                "decision": "unavailable",
                "winner": None,
                "withheld": True,
                "detail": "selection_withheld_mismatched_sets",
                "flip": None,
            }
            continue
        passers = [n for n, v in all_pass.items() if v]
        if len(passers) == 1:
            decision, winner = "selected", passers[0]
        elif len(passers) > 1:
            decision, winner = "ambiguous", None
        else:
            decision, winner = "no_candidate", None
        arms_out[arm] = {
            "decision": decision,
            "winner": winner,
            "withheld": False,
            "detail": "",
            "flip": None if arm == "x1" else (decision, winner) != mapped,
        }
    think_groups: dict[str, Any] = {}
    for label, key in ((True, "think"), (False, "no_think")):
        effects: list[tuple[str, float, Any]] = []
        for name, records in per_candidate.items():
            if records is None:
                continue
            entry = candidates[name]
            if entry.get("think_channel") is not label:
                continue
            effect = aggregates[name]["contrasts"]["x2_vs_pad"]["repetition_effect"]
            if effect is not None:
                effects.append((name, float(effect), entry.get("render_signature")))
        dedup: dict[Any, float] = {}
        for c_name, c_effect, c_signature in sorted(effects):
            dedup.setdefault(c_signature or c_name, c_effect)
        think_groups[key] = {
            "n_candidates": len(effects),
            "n_distinct_renders": len(dedup),
            "mean_repetition_effect": (
                sum(e for _n, e, _s in effects) / len(effects) if effects else None
            ),
            "mean_repetition_effect_dedup": (sum(dedup.values()) / len(dedup) if dedup else None),
        }
    signatures = {
        candidates[name].get("render_signature")
        for name, records in per_candidate.items()
        if records is not None
    }
    return {
        "aggregates": aggregates,
        "arms": arms_out,
        "think_groups": think_groups,
        "n_candidates": len(candidates),
        "n_distinct_renders": len(signatures),
    }


def _valid_repetition(payload: Any) -> bool:
    """Structural check for a ``repetition`` payload (recompute standard)."""
    if type(payload) is not dict or payload.get("available") is not True:
        return False
    for key in ("primary_decision", "primary_winner"):
        value = payload.get(key)
        if value is not None and type(value) is not str:
            return False
    if not _valid_primary_pair(payload):
        return False
    recomputed = _repetition_recompute(payload)
    if recomputed is None:
        return False
    candidates = payload["candidates"]
    for name, expected in recomputed["aggregates"].items():
        entry = candidates[name]
        for key, value in expected.items():
            if not _recomputed_equal(entry.get(key), value):
                return False
    if not _recomputed_equal(payload.get("arms"), recomputed["arms"]):
        return False
    if not _recomputed_equal(payload.get("think_groups"), recomputed["think_groups"]):
        return False
    if not _recomputed_equal(payload.get("n_candidates"), recomputed["n_candidates"]):
        return False
    return _recomputed_equal(payload.get("n_distinct_renders"), recomputed["n_distinct_renders"])


def diff_repetition(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``repetition`` probe payloads; deterministic.

    Answers "did the repetition story change between runs?":
    per-candidate effect drift, per-arm selection/flip drift,
    candidates added/removed, and protocol / fixture-set / backend
    drift (any of which marks the comparison not comparable, with
    the drift still listed). A side without a valid payload is
    reported plainly. No model, no network.
    """
    old_ok = _valid_repetition(old)
    new_ok = _valid_repetition(new)
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
    repetition_protocol_drift = old["protocol"].get("fingerprint") != new["protocol"].get(
        "fingerprint"
    )
    repetition_fixture_set_drift = old["protocol"].get("fixture_set_sha256") != new["protocol"].get(
        "fixture_set_sha256"
    )
    repetition_backend_drift = old["protocol"].get("backend_fingerprint") != new["protocol"].get(
        "backend_fingerprint"
    )
    repetition_comparable = (
        not repetition_protocol_drift
        and not repetition_fixture_set_drift
        and not repetition_backend_drift
    )
    old_candidates = old["candidates"]
    new_candidates = new["candidates"]
    repetition_candidates_added = sorted(set(new_candidates) - set(old_candidates))
    repetition_candidates_removed = sorted(set(old_candidates) - set(new_candidates))
    repetition_effect_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_candidates) & set(new_candidates)):
        old_entry = old_candidates[name]
        new_entry = new_candidates[name]
        if old_entry.get("status") != "available" or new_entry.get("status") != "available":
            if old_entry.get("status") != new_entry.get("status"):
                repetition_effect_drift.append(
                    {
                        "candidate": name,
                        "metric": "status",
                        "old": old_entry.get("status"),
                        "new": new_entry.get("status"),
                    }
                )
            continue
        for key, metric in (
            ("x2_vs_pad", "repetition_effect"),
            ("pad_vs_x1", "length_effect"),
            ("x2_vs_x1", "raw_gain"),
        ):
            old_value = old_entry["contrasts"][key][metric]
            new_value = new_entry["contrasts"][key][metric]
            if old_value != new_value:
                repetition_effect_drift.append(
                    {"candidate": name, "metric": metric, "old": old_value, "new": new_value}
                )
        if old_entry.get("pad_evaluable_share") != new_entry.get("pad_evaluable_share"):
            repetition_effect_drift.append(
                {
                    "candidate": name,
                    "metric": "pad_evaluable_share",
                    "old": old_entry.get("pad_evaluable_share"),
                    "new": new_entry.get("pad_evaluable_share"),
                }
            )
    repetition_arm_drift: list[dict[str, Any]] = []
    for arm in ("x1", "x2", "pad"):
        old_arm = old["arms"].get(arm)
        new_arm = new["arms"].get(arm)
        if old_arm != new_arm:
            repetition_arm_drift.append({"arm": arm, "old": old_arm, "new": new_arm})
    repetition_mismatch_changed = any(
        old_candidates[name].get("evaluable_mismatch")
        != new_candidates[name].get("evaluable_mismatch")
        for name in set(old_candidates) & set(new_candidates)
        if old_candidates[name].get("status") == "available"
        and new_candidates[name].get("status") == "available"
    )
    result["effect_drift"] = repetition_effect_drift
    result["arm_drift"] = repetition_arm_drift
    result["candidates_added"] = repetition_candidates_added
    result["candidates_removed"] = repetition_candidates_removed
    result["protocol_drift"] = repetition_protocol_drift
    result["fixture_set_drift"] = repetition_fixture_set_drift
    result["backend_drift"] = repetition_backend_drift
    result["comparable"] = repetition_comparable
    result["mismatch_changed"] = repetition_mismatch_changed
    # NOTE: repetition-specific disjunct names keep this ``changed``
    # line textually distinct from every other diff section's anchor
    # (mutation-gate uniqueness).
    result["changed"] = bool(
        repetition_effect_drift
        or repetition_arm_drift
        or repetition_candidates_added
        or repetition_candidates_removed
        or repetition_mismatch_changed
        or repetition_protocol_drift
        or repetition_fixture_set_drift
        or repetition_backend_drift
    )
    return result


_DISTRACTION_CELL_DETAILS = frozenset(
    {
        "",
        "vacuous",
        "arm_context_exceeded",
        "arm_evaluation_failed",
        "arm_screened",
        "arm_unmatchable",
        "near_miss_undefined",
    }
)


_DISTRACTION_HEX = frozenset("0123456789abcdef")


def _distraction_digest_ok(value: Any) -> bool:
    """Return whether the distraction digest is well-formed."""
    return value is None or (
        type(value) is str and len(value) == 64 and all(char in _DISTRACTION_HEX for char in value)
    )


def _distraction_int_ok(value: Any) -> bool:
    """Return whether the distraction int is well-formed."""
    return type(value) is int and not isinstance(value, bool) and value >= 0


def _distraction_recompute(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Recompute a distraction payload from its ground records.

    Shape pre-checks run before any recompute loop (cycle-184
    G25): a hostile payload is rejected on its silhouette, not
    after its records have been accumulated. Everything derived
    — curves, first-flip state machine, capture and its neutral
    excess, per-condition selection with withholding, the
    protocol fingerprint — is then rebuilt with independent
    arithmetic and compared by _valid_distraction.
    """

    protocol = payload.get("protocol")
    if type(protocol) is not dict:
        return None
    raw_conditions = protocol.get("conditions")
    if type(raw_conditions) is not list or not raw_conditions:
        return None
    conditions: list[tuple[str, int, str]] = []
    cond_keys: list[str] = []
    for raw in raw_conditions:
        if type(raw) is not list or len(raw) != 3:
            return None
        kind, level, placement = raw
        if kind not in ("neutral", "near-miss"):
            return None
        if type(level) is not int or isinstance(level, bool) or level <= 0:
            return None
        if placement not in ("before", "after"):
            return None
        key = f"{kind}_{level}_{placement}"
        if key in cond_keys:
            return None
        conditions.append((kind, level, placement))
        cond_keys.append(key)
    cond_key_set = frozenset(cond_keys)
    level_by_key = {f"{k}_{lv}_{p}": lv for k, lv, p in conditions}
    kind_by_key = {f"{k}_{lv}_{p}": k for k, lv, p in conditions}
    candidates = payload.get("candidates")
    if type(candidates) is not dict or not candidates:
        return None
    total_records = 0
    records_bytes = 0
    fixture_ids: list[str] | None = None
    aggregates: dict[str, Any] = {}
    raw_records: dict[str, list[dict[str, Any]]] = {}
    for name, entry in candidates.items():
        if type(name) is not str or type(entry) is not dict:
            return None
        status = entry.get("status")
        if status not in ("available", "unavailable"):
            return None
        records = entry.get("records")
        if type(records) is not list:
            return None
        if status == "unavailable":
            if records != []:
                return None
            continue
        if len(records) > 100:
            return None
        total_records += len(records)
        if total_records > 2400:
            return None
        records_bytes += len(
            json.dumps(records, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        )
        if records_bytes > 12 * 1024 * 1024:
            return None
        ids: list[str] = []
        for record in records:
            if type(record) is not dict:
                return None
            fixture = record.get("fixture")
            if type(fixture) is not str or not fixture or fixture in ids:
                return None
            ids.append(fixture)
            for flag in ("vacuous", "echo_fixture", "near_miss_defined"):
                if type(record.get(flag)) is not bool:
                    return None
            if not _distraction_digest_ok(record.get("planted_digest")):
                return None
            if record["near_miss_defined"] != (record.get("planted_digest") is not None):
                return None
            native = record.get("native")
            if type(native) is not dict:
                return None
            if type(native.get("verdict")) is not bool:
                return None
            if type(native.get("boundary_found")) is not bool:
                return None
            if not _bool_or_none(native.get("boundary_text_match")):
                return None
            if not _distraction_int_ok(native.get("input_tokens")):
                return None
            if not _distraction_digest_ok(native.get("answer_digest")):
                return None
            cells = record.get("cells")
            details = record.get("cell_details")
            if type(cells) is not dict or set(cells) != cond_key_set:
                return None
            if type(details) is not dict or set(details) != cond_key_set:
                return None
            for key in cond_keys:
                detail = details[key]
                if type(detail) is not str or detail not in _DISTRACTION_CELL_DETAILS:
                    return None
                cell = cells[key]
                if record["vacuous"]:
                    if cell is not None or detail != "vacuous":
                        return None
                    continue
                if detail == "vacuous":
                    return None
                if kind_by_key[key] == "near-miss" and not record["near_miss_defined"]:
                    if cell is not None or detail != "near_miss_undefined":
                        return None
                    continue
                if detail == "near_miss_undefined":
                    return None
                if (cell is None) != (detail != ""):
                    return None
                if cell is None:
                    continue
                if type(cell) is not dict:
                    return None
                if type(cell.get("verdict")) is not bool:
                    return None
                if type(cell.get("boundary_found")) is not bool:
                    return None
                if not _bool_or_none(cell.get("boundary_text_match")):
                    return None
                if not _distraction_int_ok(cell.get("input_tokens")):
                    return None
                if cell["input_tokens"] - native["input_tokens"] != level_by_key[key]:
                    return None
                if not _distraction_digest_ok(cell.get("answer_digest")):
                    return None
                if cell.get("construction") not in ("id_exact", "text_level"):
                    return None
                seam = cell.get("seam_ids")
                if type(seam) is not list or len(seam) > 2:
                    return None
                if any(not _distraction_int_ok(token) for token in seam):
                    return None
                occurrences = cell.get("planted_occurrences")
                if kind_by_key[key] == "neutral":
                    if occurrences is not None:
                        return None
                elif not _distraction_int_ok(occurrences):
                    return None
        if fixture_ids is None:
            fixture_ids = ids
        elif fixture_ids != ids:
            return None
        raw_records[name] = records
        live = [r for r in records if not r["vacuous"]]
        curves: dict[str, Any] = {}
        first_flip: dict[str, Any] = {}
        curve_defs: list[tuple[str, str, list[int]]] = []
        seen_curves: set[tuple[str, str]] = set()
        for kind, _level, placement in conditions:
            if (kind, placement) not in seen_curves:
                seen_curves.add((kind, placement))
                curve_defs.append(
                    (
                        kind,
                        placement,
                        sorted(lv for k2, lv, p2 in conditions if k2 == kind and p2 == placement),
                    )
                )
        for kind, placement, levels in curve_defs:
            curve_key = f"{kind}_{placement}"
            points: dict[str, Any] = {}
            for level in levels:
                key = f"{kind}_{level}_{placement}"
                n_paired = 0
                n_pass = 0
                for record in live:
                    cell = record["cells"][key]
                    if cell is None:
                        continue
                    n_paired += 1
                    if cell["verdict"]:
                        n_pass += 1
                points[str(level)] = {
                    "n_paired": n_paired,
                    "n_pass": n_pass,
                    "rate": (n_pass / n_paired) if n_paired else None,
                }
            flip_counts = {str(level): 0 for level in levels}
            n_population = 0
            n_no_flip = 0
            n_returns = 0
            n_gain_population = 0
            n_gains = 0
            for record in live:
                verdicts: list[bool | None] = []
                for level in levels:
                    cell = record["cells"][f"{kind}_{level}_{placement}"]
                    verdicts.append(cell["verdict"] if cell is not None else None)
                if any(v is None for v in verdicts):
                    continue
                if record["native"]["verdict"]:
                    n_population += 1
                    flipped_at: int | None = None
                    for level, verdict in zip(levels, verdicts, strict=True):
                        if not verdict:
                            flipped_at = level
                            break
                    if flipped_at is None:
                        n_no_flip += 1
                    else:
                        flip_counts[str(flipped_at)] += 1
                        for level, verdict in zip(levels, verdicts, strict=True):
                            if level > flipped_at and verdict:
                                n_returns += 1
                else:
                    n_gain_population += 1
                    if any(verdicts):
                        n_gains += 1
            curves[curve_key] = points
            first_flip[curve_key] = {
                "n_population": n_population,
                "flip_counts": flip_counts,
                "n_no_flip": n_no_flip,
                "n_returns": n_returns,
                "n_gain_population": n_gain_population,
                "n_gains": n_gains,
            }
        capture: dict[str, Any] = {}
        for kind, level, placement in conditions:
            if kind != "near-miss":
                continue
            key = f"{kind}_{level}_{placement}"
            neutral_key = f"neutral_{level}_{placement}"
            n_population = 0
            n_captured = 0
            n_excess_population = 0
            n_excess = 0
            for record in live:
                if not record["near_miss_defined"]:
                    continue
                if not record["native"]["verdict"]:
                    continue
                cell = record["cells"][key]
                if cell is None:
                    continue
                n_population += 1
                captured = cell["answer_digest"] == record["planted_digest"]
                if captured:
                    n_captured += 1
                neutral_cell = record["cells"].get(neutral_key)
                if neutral_cell is not None:
                    n_excess_population += 1
                    if captured and neutral_cell["answer_digest"] != record["planted_digest"]:
                        n_excess += 1
            capture[f"{placement}_{level}"] = {
                "n_population": n_population,
                "n_captured": n_captured,
                "capture_rate": (n_captured / n_population) if n_population else None,
                "n_excess_population": n_excess_population,
                "n_excess": n_excess,
                "capture_excess_over_neutral": (
                    (n_excess / n_excess_population) if n_excess_population else None
                ),
            }
        n_defined = sum(1 for r in live if r["near_miss_defined"])
        aggregates[name] = {
            "curves": curves,
            "first_flip": first_flip,
            "capture": capture,
            "n_fixtures": len(records),
            "n_vacuous": len(records) - len(live),
            "n_echo": sum(1 for r in live if r["echo_fixture"]),
            "n_nearmiss_defined": n_defined,
            "nearmiss_defined_share": (n_defined / len(live)) if live else None,
        }
    if fixture_ids is None:
        return None
    try:
        expected_fixture_sha = hashlib.sha256(
            "\n".join(sorted(fixture_ids)).encode("utf-8")
        ).hexdigest()
    except UnicodeEncodeError:
        return None  # lone-surrogate fixture id: fail closed (F6)
    if protocol.get("fixture_set_sha256") != expected_fixture_sha:
        return None
    protocol_rebuilt = {
        "version": protocol.get("version"),
        "joiner": protocol.get("joiner"),
        "conditions": protocol.get("conditions"),
        "neutral_corpus_sha256": protocol.get("neutral_corpus_sha256"),
        "nearmiss_templates_sha256": protocol.get("nearmiss_templates_sha256"),
        "nearmiss_rule": protocol.get("nearmiss_rule"),
        "sizing_rule": protocol.get("sizing_rule"),
        "sizing_encode_budget": protocol.get("sizing_encode_budget"),
        "screening_rule_version": protocol.get("screening_rule_version"),
        "generation_budget": protocol.get("generation_budget"),
        "max_ground_records": protocol.get("max_ground_records"),
        "fixture_set_sha256": expected_fixture_sha,
        "backend_fingerprint": protocol.get("backend_fingerprint"),
    }
    canonical = json.dumps(
        protocol_rebuilt, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    if protocol.get("fingerprint") != hashlib.sha256(canonical.encode("utf-8")).hexdigest():
        return None
    primary_decision = payload.get("primary_decision")
    primary_winner = payload.get("primary_winner")
    mapped: tuple[Any, Any]
    if primary_decision in ("compatible_on_fixture_set", "held_out_failed"):
        mapped = ("selected", primary_winner)
    else:
        mapped = (primary_decision, primary_winner)
    conditions_out: dict[str, Any] = {}
    for key in cond_keys:
        completed_sets: list[frozenset[str]] = []
        all_pass: dict[str, bool | None] = {}
        withheld = False
        for cand_name, entry in candidates.items():
            if entry.get("status") != "available":
                withheld = True
                all_pass[cand_name] = None
                continue
            completed = frozenset(
                record["fixture"]
                for record in raw_records[cand_name]
                if not record["vacuous"] and record["cells"][key] is not None
            )
            completed_sets.append(completed)
            if not completed:
                all_pass[cand_name] = None
                withheld = True
            else:
                all_pass[cand_name] = all(
                    record["cells"][key]["verdict"]
                    for record in raw_records[cand_name]
                    if not record["vacuous"] and record["cells"][key] is not None
                )
        if len(set(completed_sets)) > 1:
            withheld = True
        if withheld:
            conditions_out[key] = {
                "decision": "unavailable",
                "detail": "selection_withheld_mismatched_sets",
                "winner": None,
                "withheld": True,
                "flip": None,
            }
            continue
        passers = [n for n in candidates if all_pass[n]]
        if len(passers) == 1:
            decision, winner = "selected", passers[0]
        elif len(passers) > 1:
            decision, winner = "ambiguous", None
        else:
            decision, winner = "no_candidate", None
        conditions_out[key] = {
            "decision": decision,
            "winner": winner,
            "withheld": False,
            "detail": "",
            "flip": (decision, winner) != mapped,
        }
    return {
        "aggregates": aggregates,
        "conditions": conditions_out,
        "n_candidates": len(candidates),
    }


def _valid_distraction(payload: Any) -> bool:
    """Structural check for a ``distraction`` payload (recompute standard)."""
    if type(payload) is not dict or payload.get("available") is not True:
        return False
    for key in ("primary_decision", "primary_winner"):
        value = payload.get(key)
        if value is not None and type(value) is not str:
            return False
    if not _valid_primary_pair(payload):
        return False
    recomputed = _distraction_recompute(payload)
    if recomputed is None:
        return False
    candidates = payload["candidates"]
    for name, expected in recomputed["aggregates"].items():
        entry = candidates[name]
        for key, value in expected.items():
            if not _recomputed_equal(entry.get(key), value):
                return False
    if not _recomputed_equal(payload.get("conditions"), recomputed["conditions"]):
        return False
    return _recomputed_equal(payload.get("n_candidates"), recomputed["n_candidates"])


def diff_distraction(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``distraction`` probe payloads; deterministic.

    Answers "did the dilution story change between runs?":
    per-candidate curve-point and capture drift, per-condition
    selection drift, candidates added/removed, and protocol /
    fixture-set / backend drift (any of which marks the
    comparison not comparable, with the drift still listed). A
    side without a valid payload is reported plainly.
    """
    old_ok = _valid_distraction(old)
    new_ok = _valid_distraction(new)
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
    distraction_protocol_drift = old["protocol"].get("fingerprint") != new["protocol"].get(
        "fingerprint"
    )
    distraction_fixture_set_drift = old["protocol"].get("fixture_set_sha256") != new[
        "protocol"
    ].get("fixture_set_sha256")
    distraction_backend_drift = old["protocol"].get("backend_fingerprint") != new["protocol"].get(
        "backend_fingerprint"
    )
    distraction_comparable = (
        not distraction_protocol_drift
        and not distraction_fixture_set_drift
        and not distraction_backend_drift
    )
    old_candidates = old["candidates"]
    new_candidates = new["candidates"]
    distraction_candidates_added = sorted(set(new_candidates) - set(old_candidates))
    distraction_candidates_removed = sorted(set(old_candidates) - set(new_candidates))
    distraction_curve_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_candidates) & set(new_candidates)):
        old_entry = old_candidates[name]
        new_entry = new_candidates[name]
        if old_entry.get("status") != "available" or new_entry.get("status") != "available":
            if old_entry.get("status") != new_entry.get("status"):
                distraction_curve_drift.append(
                    {
                        "candidate": name,
                        "metric": "status",
                        "old": old_entry.get("status"),
                        "new": new_entry.get("status"),
                    }
                )
            continue
        for curve_key, points in old_entry["curves"].items():
            for level, point in points.items():
                old_rate = point["rate"]
                new_rate = new_entry["curves"][curve_key][level]["rate"]
                if old_rate != new_rate:
                    distraction_curve_drift.append(
                        {
                            "candidate": name,
                            "metric": f"{curve_key}_{level}_rate",
                            "old": old_rate,
                            "new": new_rate,
                        }
                    )
        for cap_key, cap in old_entry["capture"].items():
            new_cap = new_entry["capture"][cap_key]
            if cap["capture_rate"] != new_cap["capture_rate"]:
                distraction_curve_drift.append(
                    {
                        "candidate": name,
                        "metric": f"capture_{cap_key}",
                        "old": cap["capture_rate"],
                        "new": new_cap["capture_rate"],
                    }
                )
            if cap["capture_excess_over_neutral"] != new_cap["capture_excess_over_neutral"]:
                distraction_curve_drift.append(
                    {
                        "candidate": name,
                        "metric": f"capture_excess_{cap_key}",
                        "old": cap["capture_excess_over_neutral"],
                        "new": new_cap["capture_excess_over_neutral"],
                    }
                )
        if old_entry.get("nearmiss_defined_share") != new_entry.get("nearmiss_defined_share"):
            distraction_curve_drift.append(
                {
                    "candidate": name,
                    "metric": "nearmiss_defined_share",
                    "old": old_entry.get("nearmiss_defined_share"),
                    "new": new_entry.get("nearmiss_defined_share"),
                }
            )
    distraction_condition_drift: list[dict[str, Any]] = []
    for key in old["conditions"]:
        if old["conditions"][key] != new["conditions"].get(key):
            distraction_condition_drift.append(
                {
                    "condition": key,
                    "old": old["conditions"][key],
                    "new": new["conditions"].get(key),
                }
            )
    result["curve_drift"] = distraction_curve_drift
    result["condition_drift"] = distraction_condition_drift
    result["candidates_added"] = distraction_candidates_added
    result["candidates_removed"] = distraction_candidates_removed
    result["protocol_drift"] = distraction_protocol_drift
    result["fixture_set_drift"] = distraction_fixture_set_drift
    result["backend_drift"] = distraction_backend_drift
    result["comparable"] = distraction_comparable
    # NOTE: distraction-specific disjunct names keep this ``changed``
    # line textually distinct from every other diff section's anchor
    # (mutation-gate uniqueness).
    result["changed"] = bool(
        distraction_curve_drift
        or distraction_condition_drift
        or distraction_candidates_added
        or distraction_candidates_removed
        or distraction_protocol_drift
        or distraction_fixture_set_drift
        or distraction_backend_drift
    )
    return result


_MULTITURN_FAMILIES = ("ledger", "shelf")


_MULTITURN_DEPTHS = (2, 3)


_MULTITURN_ARMS = ("full", "sharded", "scripted")


_MULTITURN_DETAILS = (
    "arm_context_exceeded",
    "arm_evaluation_failed",
    "arm_screened",
)


_MULTITURN_LABELS = ("A", "B", "C")


_MULTITURN_LEDGER_QUESTION = "What is the ledger total?"


_MULTITURN_SHELF_QUESTION = "Which shelf holds the most units? Answer with the shelf label."


_MULTITURN_JOINER = "\n\n"


_MULTITURN_SCRIPTED = "Noted."


_MULTITURN_TEMPLATES = (
    "Ledger entry: {v}.",
    "Ledger entry: {v}. " + _MULTITURN_LEDGER_QUESTION,
    "Shelf {label} holds {v} units.",
    "Shelf {label} holds {v} units. " + _MULTITURN_SHELF_QUESTION,
)


_MULTITURN_CELL_KEYS = {
    "verdict",
    "boundary_found",
    "boundary_text_match",
    "input_tokens",
    "answer_digest",
}


def _multiturn_derive(
    seed: int, family: str, depth: int, index: int, slot: int, counter: int
) -> int:
    """Return the multiturn derive."""

    material = f"multiturn-v1|{seed}|{family}|{depth}|{index}|{slot}|{counter}"
    return int.from_bytes(hashlib.sha256(material.encode("ascii")).digest()[:8], "big")


def _multiturn_generate(seed: int) -> list[dict[str, Any]]:
    """Mirror of the frozen fixture generator (design literals)."""
    fixtures: list[dict[str, Any]] = []
    for family in _MULTITURN_FAMILIES:
        for depth in _MULTITURN_DEPTHS:
            for index in range(12):
                values: list[int] = []
                if family == "ledger":
                    counter = 0
                    while True:
                        candidate_values: list[int] = []
                        for slot in range(depth):
                            derived = _multiturn_derive(seed, "ledger", depth, index, slot, counter)
                            magnitude = 1 + derived % 999
                            candidate_values.append(
                                magnitude if (derived >> 32) % 2 == 0 else -magnitude
                            )
                        if str(sum(candidate_values)) not in {
                            str(value) for value in candidate_values
                        }:
                            values = candidate_values
                            break
                        counter += 1
                else:
                    for slot in range(depth):
                        counter = 0
                        while True:
                            value = (
                                1
                                + _multiturn_derive(seed, "shelf", depth, index, slot, counter) % 99
                            )
                            if value not in values:
                                values.append(value)
                                break
                            counter += 1
                shards: list[str] = []
                for slot, value in enumerate(values):
                    final = slot == depth - 1
                    if family == "ledger":
                        text = f"Ledger entry: {value}."
                        if final:
                            text += f" {_MULTITURN_LEDGER_QUESTION}"
                    else:
                        text = f"Shelf {_MULTITURN_LABELS[slot]} holds {value} units."
                        if final:
                            text += f" {_MULTITURN_SHELF_QUESTION}"
                    shards.append(text)
                if family == "ledger":
                    expected = str(sum(values))
                else:
                    expected = _MULTITURN_LABELS[values.index(max(values))]
                fixtures.append(
                    {
                        "fixture_id": f"mt-{family}-{depth}-{index:02d}",
                        "family": family,
                        "depth": depth,
                        "shards": shards,
                        "values": values,
                        "expected": expected,
                    }
                )
    return fixtures


def _multiturn_digest(text: str) -> str:
    """Return the multiturn digest."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _multiturn_template_digest() -> str:
    """Return the multiturn template digest."""

    canonical = json.dumps(
        [list(_MULTITURN_TEMPLATES), _MULTITURN_JOINER, _MULTITURN_SCRIPTED],
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _multiturn_ground_sha(seed: int) -> str:
    """Return the multiturn ground sha."""

    rows = [
        [
            fixture["fixture_id"],
            fixture["family"],
            fixture["depth"],
            fixture["shards"],
            fixture["values"],
            fixture["expected"],
        ]
        for fixture in _multiturn_generate(seed)
    ]
    canonical = json.dumps(rows, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _multiturn_protocol(seed: int, backend_fingerprint: str) -> dict[str, Any]:
    """Return the multiturn protocol."""
    return {
        "version": 1,
        "families": list(_MULTITURN_FAMILIES),
        "depths": list(_MULTITURN_DEPTHS),
        "fixtures_per_family_depth": 12,
        "arms": list(_MULTITURN_ARMS),
        "template_digest": _multiturn_template_digest(),
        "ground_sha256": _multiturn_ground_sha(seed),
        "min_paired_evaluable": 6,
        "min_discordant": 2,
        "generation_budget": 5184,
        "max_ground_records": 1152,
        "max_records_bytes": 12 * 1024 * 1024,
        "seed": seed,
        "backend_fingerprint": backend_fingerprint,
    }


def _multiturn_mcnemar_p(a_only: int, b_only: int) -> float | None:
    """Two-sided exact McNemar p (the frozen formula, local copy)."""

    n = a_only + b_only
    if n == 0:
        return None
    k = min(a_only, b_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    result: float = min(1.0, 2.0 * tail)
    return result


def _multiturn_cell(record: dict[str, Any], arm: str) -> dict[str, Any] | None:
    """Return the multiturn cell."""
    cells = record.get("cells")
    if type(cells) is not dict:
        return None
    cell = cells.get(arm)
    return cell if type(cell) is dict else None


def _multiturn_verdict(record: dict[str, Any], arm: str) -> bool:
    """Return the multiturn verdict."""
    cell = _multiturn_cell(record, arm)
    return bool(cell["verdict"]) if cell is not None else False


def _multiturn_pair(records: list[dict[str, Any]], arm_a: str, arm_b: str) -> dict[str, int]:
    """Return the multiturn pair."""
    # bandit B105: the 'pass' keys are verdict counts in the frozen
    # payload schema, not credentials.
    counts = {
        "n_paired": 0,
        "a_only": 0,
        "b_only": 0,
        "both_pass": 0,
        "both_fail": 0,
    }  # nosec B105
    for record in records:
        cell_a = _multiturn_cell(record, arm_a)
        cell_b = _multiturn_cell(record, arm_b)
        if cell_a is None or cell_b is None:
            continue
        counts["n_paired"] += 1
        verdict_a = bool(cell_a["verdict"])
        verdict_b = bool(cell_b["verdict"])
        if verdict_a and verdict_b:
            counts["both_pass"] += 1
        elif verdict_a:
            counts["a_only"] += 1
        elif verdict_b:
            counts["b_only"] += 1
        else:
            counts["both_fail"] += 1
    return counts


def _multiturn_classify(depths: dict[str, Any]) -> str:
    """Mirror of the candidate classification (total function)."""
    evaluable = [block for block in depths.values() if block["n_paired"] >= 6]
    if not evaluable:
        return "insufficient_evidence"
    for block in evaluable:
        self_history = block["self_history"]
        if self_history["sh_only"] >= 2 and self_history["sh_only"] > self_history["sc_only"]:
            return "self_history_drift"
    for block in evaluable:
        if (
            max(
                block["delivery"]["drift"],
                block["delivery"]["gains"],
                block["scripted_contrast"]["drift"],
                block["scripted_contrast"]["gains"],
            )
            >= 2
        ):
            return "framing_only"
    for block in evaluable:
        if (
            block["delivery"]["drift"]
            + block["delivery"]["gains"]
            + block["scripted_contrast"]["drift"]
            + block["scripted_contrast"]["gains"]
            + block["self_history"]["sh_only"]
            + block["self_history"]["sc_only"]
            > 0
        ):
            return "below_threshold"
    return "no_drift"


def _multiturn_aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Independent recompute of ``multiturn.aggregate_candidate``."""
    depths_out: dict[str, Any] = {}
    for depth in _MULTITURN_DEPTHS:
        at_depth = [record for record in records if record["depth"] == depth]
        paired_records = [
            record
            for record in at_depth
            if all(_multiturn_cell(record, arm) is not None for arm in _MULTITURN_ARMS)
        ]
        n_paired = len(paired_records)
        completion: dict[str, Any] = {}
        for arm in _MULTITURN_ARMS:
            details: dict[str, int] = {}
            completed = 0
            for record in at_depth:
                if _multiturn_cell(record, arm) is not None:
                    completed += 1
                else:
                    detail = record["cell_details"][arm]
                    details[detail] = details.get(detail, 0) + 1
            completion[arm] = {
                "completed": completed,
                "defined": len(at_depth),
                "details": details,
            }
        passed: dict[str, int] = {}
        rates: dict[str, Any] = {}
        for arm in _MULTITURN_ARMS:
            n_pass = sum(1 for record in paired_records if _multiturn_verdict(record, arm))
            passed[arm] = n_pass
            rates[arm] = (n_pass / n_paired) if n_paired else None
        delivery = _multiturn_pair(paired_records, "full", "sharded")
        scripted_pair = _multiturn_pair(paired_records, "full", "scripted")
        # bandit B105: the 'pass' keys are verdict counts in the
        # frozen payload schema, not credentials.
        self_history = {
            "n_full_pass": 0,  # nosec B105
            "sh_only": 0,
            "sc_only": 0,
            "both_pass": 0,  # nosec B105
            "both_fail": 0,
        }
        for record in paired_records:
            if not _multiturn_verdict(record, "full"):
                continue
            self_history["n_full_pass"] += 1
            sharded = _multiturn_verdict(record, "sharded")
            scripted = _multiturn_verdict(record, "scripted")
            if sharded and scripted:
                self_history["both_pass"] += 1
            elif scripted:
                self_history["sh_only"] += 1
            elif sharded:
                self_history["sc_only"] += 1
            else:
                self_history["both_fail"] += 1
        per_family: dict[str, Any] = {}
        for family in _MULTITURN_FAMILIES:
            family_records = [record for record in paired_records if record["family"] == family]
            fam_delivery = _multiturn_pair(family_records, "full", "sharded")
            fam_scripted = _multiturn_pair(family_records, "full", "scripted")
            fam_self = {"sh_only": 0, "sc_only": 0}
            for record in family_records:
                if not _multiturn_verdict(record, "full"):
                    continue
                sharded = _multiturn_verdict(record, "sharded")
                scripted = _multiturn_verdict(record, "scripted")
                if scripted and not sharded:
                    fam_self["sh_only"] += 1
                elif sharded and not scripted:
                    fam_self["sc_only"] += 1
            per_family[family] = {
                "n_paired": len(family_records),
                "drift": fam_delivery["a_only"],
                "gains": fam_delivery["b_only"],
                "drift_scripted": fam_scripted["a_only"],
                "sh_only": fam_self["sh_only"],
                "sc_only": fam_self["sc_only"],
            }
        depths_out[str(depth)] = {
            "n_defined": len(at_depth),
            "n_paired": n_paired,
            "completion": completion,
            "passed": passed,
            "pass_rates": rates,
            "delivery": {
                "drift": delivery["a_only"],
                "gains": delivery["b_only"],
                "both_pass": delivery["both_pass"],
                "both_fail": delivery["both_fail"],
                "mcnemar_p": _multiturn_mcnemar_p(delivery["a_only"], delivery["b_only"]),
            },
            "scripted_contrast": {
                "drift": scripted_pair["a_only"],
                "gains": scripted_pair["b_only"],
                "both_pass": scripted_pair["both_pass"],
                "both_fail": scripted_pair["both_fail"],
                "mcnemar_p": _multiturn_mcnemar_p(scripted_pair["a_only"], scripted_pair["b_only"]),
            },
            "self_history": {
                **self_history,
                "net_excess": self_history["sh_only"] - self_history["sc_only"],
                "mcnemar_p": _multiturn_mcnemar_p(self_history["sh_only"], self_history["sc_only"]),
            },
            "per_family": per_family,
        }
    return {
        "depths": depths_out,
        "classification": _multiturn_classify(depths_out),
    }


def _multiturn_digest_ok(value: Any) -> bool:
    """Return whether the multiturn digest is well-formed."""
    return (
        type(value) is str
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _multiturn_cell_ok(cell: Any, arm: str, depth: int, expected: str) -> bool:
    """One cell's shape and identities, including the verdict
    binding ``verdict == (answer_digest == sha256(expected))
    and boundary_found`` and, for the sharded arm, the turns
    identities.
    """
    if type(cell) is not dict:
        return False
    keys = set(cell)
    if arm == "sharded":
        if keys != _MULTITURN_CELL_KEYS | {
            "intermediate_unbounded",
            "intermediate_empty",
            "turns",
        }:
            return False
    elif keys != _MULTITURN_CELL_KEYS:
        return False
    if type(cell["verdict"]) is not bool or type(cell["boundary_found"]) is not bool:
        return False
    if cell["boundary_text_match"] is not None and type(cell["boundary_text_match"]) is not bool:
        return False
    if type(cell["input_tokens"]) is not int or cell["input_tokens"] < 0:
        return False
    if not _multiturn_digest_ok(cell["answer_digest"]):
        return False
    verdict = (cell["answer_digest"] == _multiturn_digest(expected)) and cell["boundary_found"]
    if cell["verdict"] != verdict:
        return False
    if arm == "sharded":
        if type(cell["intermediate_unbounded"]) is not bool:
            return False
        if type(cell["intermediate_empty"]) is not bool:
            return False
        turns = cell["turns"]
        if type(turns) is not list or len(turns) != depth:
            return False
        for entry in turns:
            if type(entry) is not dict or set(entry) != {
                "input_tokens",
                "answer_digest",
                "boundary_found",
            }:
                return False
            if type(entry["input_tokens"]) is not int or entry["input_tokens"] < 0:
                return False
            if not _multiturn_digest_ok(entry["answer_digest"]):
                return False
            if type(entry["boundary_found"]) is not bool:
                return False
        last = turns[-1]
        if (
            last["input_tokens"] != cell["input_tokens"]
            or last["answer_digest"] != cell["answer_digest"]
            or last["boundary_found"] != cell["boundary_found"]
        ):
            return False
        empty_digest = _multiturn_digest("")
        if cell["intermediate_unbounded"] != any(
            not entry["boundary_found"] for entry in turns[:-1]
        ):
            return False
        if cell["intermediate_empty"] != any(
            entry["answer_digest"] == empty_digest for entry in turns[:-1]
        ):
            return False
    return True


def _multiturn_recompute(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Recompute a multiturn payload from its ground records.

    The protocol is rebuilt from this module's own literals and
    the mirror generator (never imported from multiturn.py) and
    compared whole; every record is checked against the
    mirror-regenerated fixture; every cell's verdict is
    re-derived from its digest identity; aggregates and
    selection readouts are recomputed and compared.
    """

    protocol = payload.get("protocol")
    if type(protocol) is not dict:
        return None
    seed = protocol.get("seed")
    backend_fingerprint = protocol.get("backend_fingerprint")
    if type(seed) is not int or type(backend_fingerprint) is not str:
        return None
    expected_protocol = _multiturn_protocol(seed, backend_fingerprint)
    fingerprint = hashlib.sha256(
        json.dumps(
            expected_protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()
    expected_protocol["fingerprint"] = fingerprint
    if not _recomputed_equal(protocol, expected_protocol):
        return None
    candidates = payload.get("candidates")
    if type(candidates) is not dict or not candidates:
        return None
    generated = _multiturn_generate(seed)
    by_id = {fixture["fixture_id"]: fixture for fixture in generated}
    total_records = 0
    records_bytes = 0
    aggregates: dict[str, Any] = {}
    raw_records: dict[str, list[dict[str, Any]]] = {}
    for name, entry in candidates.items():
        if type(entry) is not dict or set(entry) != {
            "status",
            "detail",
            "depths",
            "classification",
            "records",
        }:
            return None
        if entry["status"] != "available" or entry["detail"] != "":
            return None
        records = entry["records"]
        if type(records) is not list or len(records) != len(generated):
            return None
        total_records += len(records)
        records_bytes += len(
            json.dumps(records, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        )
        if records_bytes > 12 * 1024 * 1024:
            return None
        seen: set[str] = set()
        for record in records:
            if type(record) is not dict or set(record) != {
                "fixture",
                "family",
                "depth",
                "group",
                "shards",
                "values",
                "expected",
                "cells",
                "cell_details",
            }:
                return None
            if type(record["fixture"]) is not str:
                return None
            fixture = by_id.get(record["fixture"])
            if fixture is None or record["fixture"] in seen:
                return None
            seen.add(record["fixture"])
            if (
                record["family"] != fixture["family"]
                or record["depth"] != fixture["depth"]
                or record["group"] != "multiturn"
                or record["shards"] != fixture["shards"]
                or record["values"] != fixture["values"]
                or record["expected"] != fixture["expected"]
            ):
                return None
            cells = record["cells"]
            details = record["cell_details"]
            if type(cells) is not dict or set(cells) != set(_MULTITURN_ARMS):
                return None
            if type(details) is not dict or set(details) != set(_MULTITURN_ARMS):
                return None
            for arm in _MULTITURN_ARMS:
                cell = cells[arm]
                detail = details[arm]
                if cell is None:
                    if detail not in _MULTITURN_DETAILS:
                        return None
                else:
                    if detail != "":
                        return None
                    if not _multiturn_cell_ok(
                        cell, arm, int(fixture["depth"]), str(fixture["expected"])
                    ):
                        return None
        raw_records[name] = records
        aggregates[name] = _multiturn_aggregate(records)
    if total_records > 1152:
        return None
    # Selection readouts, recomputed with the analysis rule.
    selections: dict[str, Any] = {}
    for depth in _MULTITURN_DEPTHS:
        selections[str(depth)] = {}
        defined = {fixture["fixture_id"] for fixture in generated if fixture["depth"] == depth}
        for arm in ("full", "sharded"):
            completed_sets: list[frozenset[str]] = []
            all_pass: dict[str, bool | None] = {}
            withheld = False
            for name in candidates:
                completed = {
                    record["fixture"]
                    for record in raw_records[name]
                    if record["depth"] == depth and _multiturn_cell(record, arm) is not None
                }
                completed_sets.append(frozenset(completed))
                if completed != defined or not completed:
                    withheld = True
                    all_pass[name] = None
                    continue
                all_pass[name] = all(
                    _multiturn_verdict(record, arm)
                    for record in raw_records[name]
                    if record["depth"] == depth and record["fixture"] in completed
                )
            if len(set(completed_sets)) > 1:
                withheld = True
            if withheld:
                selections[str(depth)][arm] = {
                    "decision": "unavailable",
                    "detail": "selection_withheld_mismatched_sets",
                    "winner": None,
                    "withheld": True,
                }
                continue
            passers = [n for n in candidates if all_pass[n]]
            if len(passers) == 1:
                decision, winner = "selected", passers[0]
            elif len(passers) > 1:
                decision, winner = "ambiguous", None
            else:
                decision, winner = "no_candidate", None
            selections[str(depth)][arm] = {
                "decision": decision,
                "winner": winner,
                "withheld": False,
                "detail": "",
            }
    return {
        "aggregates": aggregates,
        "selections": selections,
        "n_candidates": len(candidates),
        "n_fixtures": len(generated),
    }


def _valid_multiturn(payload: Any) -> bool:
    """Structural check for a ``multiturn`` payload (recompute standard)."""
    if type(payload) is not dict or payload.get("available") is not True:
        return False
    if set(payload) != {
        "available",
        "detail",
        "protocol",
        "candidates",
        "selections",
        "n_candidates",
        "n_fixtures",
    }:
        return False
    if payload.get("detail") != "":
        return False
    recomputed = _multiturn_recompute(payload)
    if recomputed is None:
        return False
    candidates = payload["candidates"]
    for name, expected in recomputed["aggregates"].items():
        entry = candidates[name]
        if not _recomputed_equal(entry.get("depths"), expected["depths"]):
            return False
        if not _recomputed_equal(entry.get("classification"), expected["classification"]):
            return False
    if not _recomputed_equal(payload.get("selections"), recomputed["selections"]):
        return False
    if not _recomputed_equal(payload.get("n_candidates"), recomputed["n_candidates"]):
        return False
    return _recomputed_equal(payload.get("n_fixtures"), recomputed["n_fixtures"])


def diff_multiturn(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``multiturn`` probe payloads; deterministic.

    Answers "did the multi-turn drift story change between
    runs?": per-candidate classification drift, per-depth
    sharded pass-rate and self-history pair drift, selection
    readout drift, candidates added/removed, and protocol /
    generator / backend drift (any of which marks the
    comparison not comparable, with the drift still listed --
    the generated set is the treatment, so a changed generator
    is protocol drift). A side without a valid payload is
    reported plainly.
    """
    old_ok = _valid_multiturn(old)
    new_ok = _valid_multiturn(new)
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
    multiturn_protocol_drift = old["protocol"].get("fingerprint") != new["protocol"].get(
        "fingerprint"
    )
    multiturn_generator_drift = old["protocol"].get("ground_sha256") != new["protocol"].get(
        "ground_sha256"
    )
    multiturn_backend_drift = old["protocol"].get("backend_fingerprint") != new["protocol"].get(
        "backend_fingerprint"
    )
    multiturn_comparable = (
        not multiturn_protocol_drift
        and not multiturn_generator_drift
        and not multiturn_backend_drift
    )
    old_candidates = old["candidates"]
    new_candidates = new["candidates"]
    multiturn_candidates_added = sorted(set(new_candidates) - set(old_candidates))
    multiturn_candidates_removed = sorted(set(old_candidates) - set(new_candidates))
    multiturn_classification_drift: list[dict[str, Any]] = []
    multiturn_depth_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_candidates) & set(new_candidates)):
        old_entry = old_candidates[name]
        new_entry = new_candidates[name]
        if old_entry.get("status") != "available" or new_entry.get("status") != "available":
            if old_entry.get("status") != new_entry.get("status"):
                multiturn_classification_drift.append(
                    {
                        "candidate": name,
                        "metric": "status",
                        "old": old_entry.get("status"),
                        "new": new_entry.get("status"),
                    }
                )
            continue
        if old_entry.get("classification") != new_entry.get("classification"):
            multiturn_classification_drift.append(
                {
                    "candidate": name,
                    "metric": "classification",
                    "old": old_entry.get("classification"),
                    "new": new_entry.get("classification"),
                }
            )
        for depth in ("2", "3"):
            old_block = old_entry["depths"][depth]
            new_block = new_entry["depths"][depth]
            if old_block["pass_rates"]["sharded"] != new_block["pass_rates"]["sharded"]:
                multiturn_depth_drift.append(
                    {
                        "candidate": name,
                        "metric": f"depth_{depth}_sharded_rate",
                        "old": old_block["pass_rates"]["sharded"],
                        "new": new_block["pass_rates"]["sharded"],
                    }
                )
            old_pair = (
                old_block["self_history"]["sh_only"],
                old_block["self_history"]["sc_only"],
            )
            new_pair = (
                new_block["self_history"]["sh_only"],
                new_block["self_history"]["sc_only"],
            )
            if old_pair != new_pair:
                multiturn_depth_drift.append(
                    {
                        "candidate": name,
                        "metric": f"depth_{depth}_self_history_pair",
                        "old": list(old_pair),
                        "new": list(new_pair),
                    }
                )
    multiturn_selection_drift: list[dict[str, Any]] = []
    for depth in ("2", "3"):
        for arm in ("full", "sharded"):
            if old["selections"][depth][arm] != new["selections"][depth][arm]:
                multiturn_selection_drift.append(
                    {
                        "depth": depth,
                        "arm": arm,
                        "old": old["selections"][depth][arm],
                        "new": new["selections"][depth][arm],
                    }
                )
    result["classification_drift"] = multiturn_classification_drift
    result["depth_drift"] = multiturn_depth_drift
    result["selection_drift"] = multiturn_selection_drift
    result["candidates_added"] = multiturn_candidates_added
    result["candidates_removed"] = multiturn_candidates_removed
    result["protocol_drift"] = multiturn_protocol_drift
    result["generator_drift"] = multiturn_generator_drift
    result["backend_drift"] = multiturn_backend_drift
    result["comparable"] = multiturn_comparable
    result["changed"] = bool(
        multiturn_classification_drift
        or multiturn_depth_drift
        or multiturn_selection_drift
        or multiturn_candidates_added
        or multiturn_candidates_removed
        or multiturn_protocol_drift
        or multiturn_generator_drift
        or multiturn_backend_drift
    )
    return result


_NEEDLE_DOSES = (1024, 4096)


_NEEDLE_DEPTHS = (0, 25, 50, 75, 100)


_NEEDLE_CASE_COUNT = 20


_NEEDLE_FACT_TEMPLATE = "The {topic} code is {value}. "


_NEEDLE_QUESTION_TEMPLATE = "What is the {topic} code?"


_NEEDLE_TOPICS = (
    "astronomy society",
    "beekeeping circle",
    "chess club",
    "rowing team",
    "radio club",
    "pottery class",
    "fencing team",
    "archive office",
    "textile guild",
    "mountain rescue",
    "vineyard crew",
    "observatory staff",
    "printing workshop",
    "cycling club",
    "dance academy",
    "karate school",
    "model railway club",
    "stamp society",
    "hiking group",
    "choir society",
)


_NEEDLE_WORDS = (
    "ochre",
    "velvet",
    "cobalt",
    "harbor",
    "saffron",
    "tundra",
    "zephyr",
    "marble",
    "onyx",
    "pepper",
    "quartz",
    "riddle",
    "saddle",
    "temple",
    "umbrella",
    "violet",
    "walnut",
    "yonder",
    "zipper",
    "anchor",
    "beacon",
    "cinder",
    "dagger",
    "ember",
    "falcon",
    "garnet",
    "helmet",
    "inkwell",
    "jubilee",
    "kestrel",
    "basalt",
    "magnet",
    "nectar",
    "obelisk",
    "pumice",
    "quiver",
    "rampart",
    "scepter",
    "sienna",
    "utensil",
    "vortex",
    "willow",
    "yonks",
    "zenith",
    "almond",
    "birch",
    "copper",
    "drizzle",
    "elmwood",
    "fjord",
    "glacier",
    "mauve",
    "ivory",
    "juniper",
    "kraken",
    "linden",
    "meteor",
    "nickel",
    "opal",
    "pyrite",
    "quill",
    "russet",
    "sorrel",
    "topaz",
)


_NEEDLE_SIZING_RULE_TEXT = (
    "needle-phased-exact-v1: spans are whole sentences plus one "
    "truncated suffix (span_text) over the cycled neutral corpus "
    "stream, consumed in stream order; a truncated sentence "
    "counts as consumed and the next span continues after it. "
    "Treated cells size in two phases: prefix candidates in "
    "span_candidates closeness order, probed as (prefix + needle "
    "+ question) against the comparator (prefix + question), "
    "accepting the first whose D equals the target offset (the "
    "empty prefix at depth 0, no search); then suffix candidates "
    "continuing the stream, probed as the full arm render, "
    "accepting the first whose rendered delta over the dose-0 "
    "render equals the dose with D re-verified on the same ids "
    "(the empty suffix at depth 100, verified on a single "
    "render). Duplicate cells insert a middle phase between "
    "them: mid candidates continuing the stream, probed as "
    "(prefix + needle + mid + needle + question) against "
    "(prefix + needle + mid + question), accepting the first "
    "whose D equals the second target offset; the suffix phase "
    "then targets the duplicate delta (dose + needle_len) and "
    "re-verifies both offsets on the final ids. Absent cells "
    "size a single span against the question-only render. The "
    "encode budget counts every probe encode in the phases "
    "(two per prefix or middle probe, one per suffix or absent "
    "probe); the cached question-only, dose-0, and accepted-"
    "comparator renders are infrastructure and the joint final "
    "verification reuses the accepted renders, adding none."
)


_NEEDLE_SCREENING_RULE_TEXT = (
    "needle-screen-v1: per constructed filler span, case-folded: "
    "the span is screened when it contains the case value, or "
    "when the candidate's decoded boundary text occurs in it. "
    "The needle sentence itself is the plant and is never "
    "screened. Stage-level, the frozen texts (fact template "
    "with slots emptied, topics, words, neutral corpus) are "
    "audited for special-token literals against every candidate "
    "format, and the generated texts (fact template with slots "
    "emptied, topics, words) are checked for containment of "
    "any candidate's decoded boundary text; a hit fails the "
    "whole stage closed before any generation."
)


_NEEDLE_MIN_HEADLINE_POPULATION = 6


_NEEDLE_STREAM_CYCLES = 6


_NEEDLE_MAX_GROUND_RECORDS = 480


_NEEDLE_MAX_RECORDS_BYTES = 12 * 1024 * 1024


_NEEDLE_GENERATION_BUDGET = 7200


_NEEDLE_DETAILS = (
    "arm_context_exceeded",
    "arm_evaluation_failed",
    "arm_screened",
    "arm_unmatchable",
)


_NEEDLE_CELL_FIELDS = {
    "verdict",
    "boundary_found",
    "boundary_text_match",
    "input_tokens",
    "answer_digest",
    "needle_offsets",
}


def _needle_key_treated(dose: int, depth: int) -> str:
    """Return the needle key treated."""
    return f"dose_{dose}_depth_{depth}"


def _needle_key_absent(dose: int) -> str:
    """Return the needle key absent."""
    return f"absent_{dose}"


def _needle_key_duplicate(dose: int) -> str:
    """Return the needle key duplicate."""
    return f"duplicate_{dose}"


_NEEDLE_CELL_KEYS = (
    ("dose_0",)
    + tuple(_needle_key_treated(d, p) for d in _NEEDLE_DOSES for p in _NEEDLE_DEPTHS)
    + tuple(_needle_key_absent(d) for d in _NEEDLE_DOSES)
    + tuple(_needle_key_duplicate(d) for d in _NEEDLE_DOSES)
)


def _needle_digest(text: str) -> str:
    """Return the needle digest."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _needle_digest_ok(value: Any) -> bool:
    """Return whether the needle digest is well-formed."""
    return (
        type(value) is str
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _needle_derive(seed: int, index: int, slot: int, counter: int) -> int:
    """Return the needle derive."""

    material = f"{seed}|needle|{index}|{slot}|{counter}"
    return int.from_bytes(hashlib.sha256(material.encode("ascii")).digest()[:8], "big")


def _needle_generate(seed: int) -> list[dict[str, Any]]:
    """Mirror of the frozen case generation: slot 0 is the code
    word, slot 1 the number; a value colliding with an earlier
    case redraws both slots at the next counter.
    """
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index in range(_NEEDLE_CASE_COUNT):
        topic = _NEEDLE_TOPICS[index]
        counter = 0
        while True:
            word = _NEEDLE_WORDS[_needle_derive(seed, index, 0, counter) % len(_NEEDLE_WORDS)]
            number = 1000 + _needle_derive(seed, index, 1, counter) % 9000
            value = f"{word}-{number}"
            if value not in seen:
                break
            counter += 1
        seen.add(value)
        cases.append(
            {
                "fixture_id": f"needle-{index:02d}",
                "topic": topic,
                "value": value,
            }
        )
    return cases


def _needle_case_material_sha() -> str:
    """Return the needle case material sha."""

    canonical = json.dumps(
        [
            _NEEDLE_FACT_TEMPLATE,
            _NEEDLE_QUESTION_TEMPLATE,
            list(_NEEDLE_TOPICS),
            list(_NEEDLE_WORDS),
        ],
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _needle_ground_sha(seed: int) -> str:
    """Return the needle ground sha."""

    rows = [[case["fixture_id"], case["topic"], case["value"]] for case in _needle_generate(seed)]
    canonical = json.dumps(rows, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _needle_protocol(seed: int, backend_fingerprint: str) -> dict[str, Any]:
    """Return the needle protocol."""

    protocol = {
        "version": 1,
        "doses": list(_NEEDLE_DOSES),
        "depths": list(_NEEDLE_DEPTHS),
        "case_count": _NEEDLE_CASE_COUNT,
        "cell_keys": list(_NEEDLE_CELL_KEYS),
        "case_material_sha256": _needle_case_material_sha(),
        "ground_sha256": _needle_ground_sha(seed),
        "sizing_rule_sha256": _needle_digest(_NEEDLE_SIZING_RULE_TEXT),
        "sizing_encode_budget": 512,
        "screening_rule_sha256": _needle_digest(_NEEDLE_SCREENING_RULE_TEXT),
        "stream_cycles": _NEEDLE_STREAM_CYCLES,
        "min_headline_population": _NEEDLE_MIN_HEADLINE_POPULATION,
        "generation_budget": _NEEDLE_GENERATION_BUDGET,
        "max_ground_records": _NEEDLE_MAX_GROUND_RECORDS,
        "max_records_bytes": _NEEDLE_MAX_RECORDS_BYTES,
        "seed": seed,
        "backend_fingerprint": backend_fingerprint,
    }
    fingerprint = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
    ).hexdigest()
    protocol["fingerprint"] = fingerprint
    return protocol


def _needle_cell(record: dict[str, Any], key: str) -> dict[str, Any] | None:
    """Return the needle cell."""
    cells = record.get("cells")
    if type(cells) is not dict:
        return None
    cell = cells.get(key)
    return cell if type(cell) is dict else None


def _needle_verdict(record: dict[str, Any], key: str) -> bool:
    """Return the needle verdict."""
    cell = _needle_cell(record, key)
    return bool(cell["verdict"]) if cell is not None else False


def _needle_cell_ok(cell: Any, expected_digest: str) -> bool:
    """One cell's shape and the verdict binding
    ``verdict == (answer_digest == expected_digest) and
    boundary_found``. The expected digest is the record's,
    itself checked against the mirror-regenerated case.
    """
    if type(cell) is not dict or set(cell) != _NEEDLE_CELL_FIELDS:
        return False
    if type(cell["verdict"]) is not bool or type(cell["boundary_found"]) is not bool:
        return False
    if cell["boundary_text_match"] is not None and type(cell["boundary_text_match"]) is not bool:
        return False
    if type(cell["input_tokens"]) is not int or cell["input_tokens"] < 0:
        return False
    if cell["answer_digest"] is not None and not _needle_digest_ok(cell["answer_digest"]):
        return False
    offsets = cell["needle_offsets"]
    if type(offsets) is not list or any(type(o) is not int or o < 0 for o in offsets):
        return False
    verdict = (cell["answer_digest"] == expected_digest) and cell["boundary_found"]
    return bool(cell["verdict"] == verdict)


def _needle_expected_offsets(key: str) -> list[int] | None:
    """The frozen needle offsets a cell key must carry."""
    if key == "dose_0":
        return [0]
    for dose in _NEEDLE_DOSES:
        if key == _needle_key_absent(dose):
            return []
        if key == _needle_key_duplicate(dose):
            return [dose * 25 // 100, dose * 75 // 100]
        for depth in _NEEDLE_DEPTHS:
            if key == _needle_key_treated(dose, depth):
                return [dose * depth // 100]
    return None


def _needle_aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Mirror of needle.aggregate_candidate."""
    n_recorded = 0
    n_pass = 0
    for record in records:
        cell = _needle_cell(record, "dose_0")
        if cell is None:
            continue
        n_recorded += 1
        if cell["verdict"]:
            n_pass += 1
    dose0_control = {
        "n_recorded": n_recorded,
        "n_pass": n_pass,
        "rate": (n_pass / n_recorded) if n_recorded else None,
    }
    false_recall: dict[str, Any] = {}
    for dose in _NEEDLE_DOSES:
        key = _needle_key_absent(dose)
        n_evaluable = 0
        n_false_pass = 0
        for record in records:
            cell = _needle_cell(record, key)
            if cell is None:
                continue
            n_evaluable += 1
            if cell["verdict"]:
                n_false_pass += 1
        false_recall[str(dose)] = {
            "n_evaluable": n_evaluable,
            "n_false_pass": n_false_pass,
            "rate": (n_false_pass / n_evaluable) if n_evaluable else None,
        }
    duplicate: dict[str, Any] = {}
    for dose in _NEEDLE_DOSES:
        population = [
            record
            for record in records
            if _needle_verdict(record, "dose_0")
            and _needle_cell(record, _needle_key_duplicate(dose)) is not None
            and _needle_cell(record, _needle_key_treated(dose, 25)) is not None
            and _needle_cell(record, _needle_key_treated(dose, 50)) is not None
            and _needle_cell(record, _needle_key_treated(dose, 75)) is not None
        ]
        n_population = len(population)
        if not n_population:
            duplicate[str(dose)] = {
                "n_population": 0,
                "n_pass": 0,  # nosec B105 -- verdict count key, not a credential
                "pass_rate": None,  # nosec B105 -- verdict rate key, not a credential
                "rate_25": None,
                "rate_75": None,
                "independence_expected": None,
                "excess_over_independence": None,
                "n_mid_fail": 0,
                "n_rescued": 0,
                "rescue_rate": None,
            }
            continue
        dup_key = _needle_key_duplicate(dose)
        n_dup_pass = sum(1 for r in population if _needle_verdict(r, dup_key))
        rate_25 = (
            sum(1 for r in population if _needle_verdict(r, _needle_key_treated(dose, 25)))
            / n_population
        )
        rate_75 = (
            sum(1 for r in population if _needle_verdict(r, _needle_key_treated(dose, 75)))
            / n_population
        )
        expected = 1.0 - (1.0 - rate_25) * (1.0 - rate_75)
        pass_rate = n_dup_pass / n_population
        mid_fails = [r for r in population if not _needle_verdict(r, _needle_key_treated(dose, 50))]
        n_rescued = sum(1 for r in mid_fails if _needle_verdict(r, dup_key))
        duplicate[str(dose)] = {
            "n_population": n_population,
            "n_pass": n_dup_pass,
            "pass_rate": pass_rate,
            "rate_25": rate_25,
            "rate_75": rate_75,
            "independence_expected": expected,
            "excess_over_independence": pass_rate - expected,
            "n_mid_fail": len(mid_fails),
            "n_rescued": n_rescued,
            "rescue_rate": (n_rescued / len(mid_fails)) if mid_fails else None,
        }
    return {
        "dose0_control": dose0_control,
        "false_recall": false_recall,
        "duplicate": duplicate,
    }


def _needle_headline(
    records_by_candidate: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Mirror of needle.headline_readouts."""
    out: dict[str, Any] = {}
    by_fixture: dict[str, dict[str, dict[str, Any]]] = {
        name: {record["fixture"]: record for record in records}
        for name, records in records_by_candidate.items()
    }
    names = list(records_by_candidate)
    for dose in _NEEDLE_DOSES:
        keys = [_needle_key_treated(dose, depth) for depth in _NEEDLE_DEPTHS]
        population: list[str] = []
        if names:
            for record in records_by_candidate[names[0]]:
                fixture = record["fixture"]
                ok = True
                for name in names:
                    match = by_fixture[name].get(fixture)
                    if match is None or not _needle_verdict(match, "dose_0"):
                        ok = False
                        break
                    if any(_needle_cell(match, key) is None for key in keys):
                        ok = False
                        break
                if ok:
                    population.append(fixture)
        n_population = len(population)
        curves: dict[str, Any] = {}
        deficit: dict[str, Any] = {}
        first_failure: dict[str, Any] = {}
        for name in names:
            points: dict[str, Any] = {}
            rates: dict[int, float | None] = {}
            for depth in _NEEDLE_DEPTHS:
                key = _needle_key_treated(dose, depth)
                n_pass = sum(
                    1 for fixture in population if _needle_verdict(by_fixture[name][fixture], key)
                )
                rate = (n_pass / n_population) if n_population else None
                rates[depth] = rate
                points[str(depth)] = {"n_pass": n_pass, "rate": rate}
            curves[name] = points
            rate_0 = rates[0]
            rate_50 = rates[50]
            rate_100 = rates[100]
            if (
                n_population < _NEEDLE_MIN_HEADLINE_POPULATION
                or rate_0 is None
                or rate_50 is None
                or rate_100 is None
            ):
                deficit[name] = None
                first_failure[name] = None
                continue
            deficit[name] = (rate_0 + rate_100) / 2 - rate_50
            counts = {str(depth): 0 for depth in _NEEDLE_DEPTHS}
            n_no_failure = 0
            for fixture in population:
                record = by_fixture[name][fixture]
                first: int | None = None
                for depth in _NEEDLE_DEPTHS:
                    if not _needle_verdict(record, _needle_key_treated(dose, depth)):
                        first = depth
                        break
                if first is None:
                    n_no_failure += 1
                else:
                    counts[str(first)] += 1
            first_failure[name] = {"counts": counts, "n_no_failure": n_no_failure}
        out[str(dose)] = {
            "population": population,
            "n_population": n_population,
            "curves": curves,
            "deficit": deficit,
            "first_failure": first_failure,
        }
    return out


def _needle_selection_over(
    records_by_candidate: dict[str, list[dict[str, Any]]],
    key: str,
    fixtures: frozenset[str],
) -> dict[str, Any] | None:
    """Mirror of needle._selection_over."""
    completed_sets: list[frozenset[str]] = []
    all_pass: dict[str, bool | None] = {}
    for name, records in records_by_candidate.items():
        completed = frozenset(
            record["fixture"]
            for record in records
            if record["fixture"] in fixtures and _needle_cell(record, key) is not None
        )
        completed_sets.append(completed)
        if not completed:
            return None
        all_pass[name] = all(
            _needle_verdict(record, key) for record in records if record["fixture"] in completed
        )
    if len(set(completed_sets)) > 1:
        return None
    passers = [name for name, ok in all_pass.items() if ok]
    if len(passers) == 1:
        return {"decision": "selected", "winner": passers[0]}
    if len(passers) > 1:
        return {"decision": "ambiguous", "winner": None}
    return {"decision": "no_candidate", "winner": None}


def _needle_selection(
    records_by_candidate: dict[str, list[dict[str, Any]]], key: str
) -> dict[str, Any]:
    """Mirror of needle.cell_selection, including the
    paired-restricted flip baseline.
    """
    all_fixtures = frozenset(
        record["fixture"] for records in records_by_candidate.values() for record in records
    )
    own = _needle_selection_over(records_by_candidate, key, all_fixtures)
    if own is None:
        return {
            "decision": "unavailable",
            "detail": "selection_withheld_mismatched_sets",
            "winner": None,
            "withheld": True,
            "flip": None,
        }
    if key == "dose_0":
        flip: bool | None = False
    else:
        completed = frozenset(
            record["fixture"]
            for records in records_by_candidate.values()
            for record in records
            if _needle_cell(record, key) is not None
        )
        baseline = _needle_selection_over(records_by_candidate, "dose_0", completed)
        flip = None if baseline is None else own != baseline
    return {
        "decision": own["decision"],
        "detail": "",
        "winner": own["winner"],
        "withheld": False,
        "flip": flip,
    }


def _needle_recompute(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Recompute a needle payload from its ground records.

    The protocol is rebuilt from this module's own literals and
    the mirror generator (never imported from needle.py) and
    compared whole; every record is checked against the
    mirror-regenerated case in canonical order; every cell's
    verdict is re-derived from its digest identity; the delta
    identities and frozen needle offsets are checked wherever
    the cells they relate are both present; aggregates,
    headline readouts, and per-cell selections are recomputed
    and compared.
    """

    protocol = payload.get("protocol")
    if type(protocol) is not dict:
        return None
    seed = protocol.get("seed")
    backend_fingerprint = protocol.get("backend_fingerprint")
    if type(seed) is not int or type(backend_fingerprint) is not str:
        return None
    if not _recomputed_equal(protocol, _needle_protocol(seed, backend_fingerprint)):
        return None
    candidates = payload.get("candidates")
    if type(candidates) is not dict or not candidates:
        return None
    generated = _needle_generate(seed)
    total_records = 0
    records_bytes = 0
    aggregates: dict[str, Any] = {}
    raw_records: dict[str, list[dict[str, Any]]] = {}
    for name, entry in candidates.items():
        if type(entry) is not dict or set(entry) != {
            "status",
            "detail",
            "dose0_control",
            "false_recall",
            "duplicate",
            "records",
        }:
            return None
        if entry["status"] != "available" or entry["detail"] != "":
            return None
        records = entry["records"]
        if type(records) is not list or len(records) != len(generated):
            return None
        total_records += len(records)
        records_bytes += len(
            json.dumps(records, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        )
        if records_bytes > _NEEDLE_MAX_RECORDS_BYTES:
            return None
        for record, case in zip(records, generated, strict=True):
            if type(record) is not dict or set(record) != {
                "fixture",
                "topic",
                "expected_digest",
                "question_tokens",
                "cells",
                "cell_details",
            }:
                return None
            if record["fixture"] != case["fixture_id"] or record["topic"] != case["topic"]:
                return None
            expected_digest = _needle_digest(str(case["value"]))
            if record["expected_digest"] != expected_digest:
                return None
            if type(record["question_tokens"]) is not int or record["question_tokens"] < 0:
                return None
            cells = record["cells"]
            details = record["cell_details"]
            if type(cells) is not dict or set(cells) != set(_NEEDLE_CELL_KEYS):
                return None
            if type(details) is not dict or set(details) != set(_NEEDLE_CELL_KEYS):
                return None
            for key in _NEEDLE_CELL_KEYS:
                cell = cells[key]
                detail = details[key]
                if cell is None:
                    if detail not in _NEEDLE_DETAILS:
                        return None
                else:
                    if detail != "":
                        return None
                    if not _needle_cell_ok(cell, expected_digest):
                        return None
                    if cell["needle_offsets"] != _needle_expected_offsets(key):
                        return None
            dose0 = cells["dose_0"]
            if dose0 is not None and dose0["input_tokens"] <= record["question_tokens"]:
                return None
            needle_len = (
                dose0["input_tokens"] - record["question_tokens"] if dose0 is not None else None
            )
            for dose in _NEEDLE_DOSES:
                absent = cells[_needle_key_absent(dose)]
                if absent is not None and (
                    absent["input_tokens"] - record["question_tokens"] != dose
                ):
                    return None
                if dose0 is None:
                    continue
                for depth in _NEEDLE_DEPTHS:
                    treated = cells[_needle_key_treated(dose, depth)]
                    if treated is not None and (
                        treated["input_tokens"] - dose0["input_tokens"] != dose
                    ):
                        return None
                duplicate = cells[_needle_key_duplicate(dose)]
                if (
                    duplicate is not None
                    and needle_len is not None
                    and (duplicate["input_tokens"] - dose0["input_tokens"] != dose + needle_len)
                ):
                    return None
        raw_records[name] = records
        aggregates[name] = _needle_aggregate(records)
    if total_records > _NEEDLE_MAX_GROUND_RECORDS:
        return None
    return {
        "aggregates": aggregates,
        "readouts": _needle_headline(raw_records),
        "conditions": {key: _needle_selection(raw_records, key) for key in _NEEDLE_CELL_KEYS},
        "n_candidates": len(candidates),
        "n_cases": len(generated),
    }


def _valid_needle(payload: Any) -> bool:
    """Structural check for a ``needle`` payload (recompute standard)."""
    if type(payload) is not dict or payload.get("available") is not True:
        return False
    if set(payload) != {
        "available",
        "detail",
        "protocol",
        "candidates",
        "readouts",
        "conditions",
        "n_candidates",
        "n_cases",
    }:
        return False
    if payload.get("detail") != "":
        return False
    recomputed = _needle_recompute(payload)
    if recomputed is None:
        return False
    candidates = payload["candidates"]
    for name, expected in recomputed["aggregates"].items():
        entry = candidates[name]
        if not _recomputed_equal(entry.get("dose0_control"), expected["dose0_control"]):
            return False
        if not _recomputed_equal(entry.get("false_recall"), expected["false_recall"]):
            return False
        if not _recomputed_equal(entry.get("duplicate"), expected["duplicate"]):
            return False
    if not _recomputed_equal(payload.get("readouts"), recomputed["readouts"]):
        return False
    if not _recomputed_equal(payload.get("conditions"), recomputed["conditions"]):
        return False
    if not _recomputed_equal(payload.get("n_candidates"), recomputed["n_candidates"]):
        return False
    return _recomputed_equal(payload.get("n_cases"), recomputed["n_cases"])


def diff_needle(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``needle`` probe payloads; deterministic.

    Answers "did the depth story change between runs?":
    per-candidate deficit and dose-0 drift, false-recall and
    duplicate-excess drift, per-cell selection drift,
    candidates added/removed, and protocol / generator /
    backend drift (any of which marks the comparison not
    comparable, with the drift still listed -- the generated
    set is the treatment, so a changed generator is protocol
    drift). A side without a valid payload is reported plainly.
    """
    old_ok = _valid_needle(old)
    new_ok = _valid_needle(new)
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
    needle_protocol_drift = old["protocol"].get("fingerprint") != new["protocol"].get("fingerprint")
    needle_generator_drift = old["protocol"].get("ground_sha256") != new["protocol"].get(
        "ground_sha256"
    )
    needle_backend_drift = old["protocol"].get("backend_fingerprint") != new["protocol"].get(
        "backend_fingerprint"
    )
    needle_comparable = (
        not needle_protocol_drift and not needle_generator_drift and not needle_backend_drift
    )
    old_candidates = old["candidates"]
    new_candidates = new["candidates"]
    needle_candidates_added = sorted(set(new_candidates) - set(old_candidates))
    needle_candidates_removed = sorted(set(old_candidates) - set(new_candidates))
    needle_candidate_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_candidates) & set(new_candidates)):
        old_entry = old_candidates[name]
        new_entry = new_candidates[name]
        if old_entry.get("status") != "available" or new_entry.get("status") != "available":
            if old_entry.get("status") != new_entry.get("status"):
                needle_candidate_drift.append(
                    {
                        "candidate": name,
                        "metric": "status",
                        "old": old_entry.get("status"),
                        "new": new_entry.get("status"),
                    }
                )
            continue
        if old_entry["dose0_control"]["rate"] != new_entry["dose0_control"]["rate"]:
            needle_candidate_drift.append(
                {
                    "candidate": name,
                    "metric": "dose0_rate",
                    "old": old_entry["dose0_control"]["rate"],
                    "new": new_entry["dose0_control"]["rate"],
                }
            )
        for dose in ("1024", "4096"):
            old_deficit = old["readouts"][dose]["deficit"][name]
            new_deficit = new["readouts"][dose]["deficit"][name]
            if old_deficit != new_deficit:
                needle_candidate_drift.append(
                    {
                        "candidate": name,
                        "metric": f"dose_{dose}_deficit",
                        "old": old_deficit,
                        "new": new_deficit,
                    }
                )
            for depth in ("0", "50", "100"):
                old_rate = old["readouts"][dose]["curves"][name][depth]["rate"]
                new_rate = new["readouts"][dose]["curves"][name][depth]["rate"]
                if old_rate != new_rate:
                    needle_candidate_drift.append(
                        {
                            "candidate": name,
                            "metric": f"dose_{dose}_depth_{depth}_rate",
                            "old": old_rate,
                            "new": new_rate,
                        }
                    )
            old_floor = old_entry["false_recall"][dose]["rate"]
            new_floor = new_entry["false_recall"][dose]["rate"]
            if old_floor != new_floor:
                needle_candidate_drift.append(
                    {
                        "candidate": name,
                        "metric": f"dose_{dose}_false_recall",
                        "old": old_floor,
                        "new": new_floor,
                    }
                )
            old_excess = old_entry["duplicate"][dose]["excess_over_independence"]
            new_excess = new_entry["duplicate"][dose]["excess_over_independence"]
            if old_excess != new_excess:
                needle_candidate_drift.append(
                    {
                        "candidate": name,
                        "metric": f"dose_{dose}_duplicate_excess",
                        "old": old_excess,
                        "new": new_excess,
                    }
                )
    needle_condition_drift: list[dict[str, Any]] = []
    for key in _NEEDLE_CELL_KEYS:
        if old["conditions"][key] != new["conditions"][key]:
            needle_condition_drift.append(
                {
                    "condition": key,
                    "old": old["conditions"][key],
                    "new": new["conditions"][key],
                }
            )
    result["candidate_drift"] = needle_candidate_drift
    result["condition_drift"] = needle_condition_drift
    result["candidates_added"] = needle_candidates_added
    result["candidates_removed"] = needle_candidates_removed
    result["protocol_drift"] = needle_protocol_drift
    result["generator_drift"] = needle_generator_drift
    result["backend_drift"] = needle_backend_drift
    result["comparable"] = needle_comparable
    result["changed"] = bool(
        needle_candidate_drift
        or needle_condition_drift
        or needle_candidates_added
        or needle_candidates_removed
        or needle_protocol_drift
        or needle_generator_drift
        or needle_backend_drift
    )
    return result


_PLACEMENT_ARM_KEYS = (
    "question_first",
    "payload_first",
    "sandwich",
    "closed_book",
)


_PLACEMENT_LAYOUT_ARMS = ("question_first", "payload_first", "sandwich")


_PLACEMENT_BASELINE_ARM = "payload_first"


_PLACEMENT_CASE_COUNT = 24


_PLACEMENT_FACT_TEMPLATE = "The {topic} inspection code is {value}."


_PLACEMENT_QUESTION_TEMPLATE = "What is the {topic} inspection code?"


_PLACEMENT_TOPICS = (
    "archive vault",
    "boiler room",
    "loading dock",
    "records office",
    "north gate",
    "pump house",
    "storage annex",
    "mail room",
    "server closet",
    "roof access",
    "east stairwell",
    "water tower",
    "fuel depot",
    "tool library",
    "greenhouse bay",
    "ferry office",
    "signal box",
    "customs shed",
    "observatory dome",
    "print workshop",
    "harbor office",
    "tunnel control",
    "solar array",
    "wind station",
)


_PLACEMENT_WORDS = (
    "amber",
    "birch",
    "cobalt",
    "dune",
    "ember",
    "flint",
    "garnet",
    "jasper",
    "kestrel",
    "linden",
    "magnet",
    "nutmeg",
    "onyx",
    "sorrel",
    "quartz",
    "rowan",
    "saddle",
    "timber",
    "umber",
    "velvet",
    "walnut",
    "yonder",
    "zephyr",
    "copper",
)


_PLACEMENT_NEUTRAL_POOL = (
    "The corridor lights hum softly after the evening shift ends.",
    "A kettle sits on the counter beside a chipped blue mug.",
    "Rain taps the window above the wooden bench all morning.",
    "The notice board by the door holds a faded timetable.",
    "Someone left a grey umbrella leaning against the wall.",
    "The floor tiles are cool even on the warmest afternoons.",
    "A narrow shelf holds jars of screws, washers, and twine.",
    "The clock over the sink runs a few minutes fast.",
    "Dust gathers on the ledge where the plants used to sit.",
    "The hallway smells faintly of soap and old paper.",
    "A folded map rests on top of the filing cabinet.",
    "The radiator clicks twice before the heat comes on.",
    "Paper clips and rubber bands fill the shallow drawer.",
    "The yard outside is quiet except for sparrows at dawn.",
    "A chipped enamel sign hangs crooked by the entrance.",
    "The mop bucket stands in the corner by the sink.",
)


_PLACEMENT_MIN_PAIRED_EVALUABLE = 6


_PLACEMENT_MIN_DISCORDANT = 2


_PLACEMENT_MAX_GROUND_RECORDS = 24 * _PLACEMENT_CASE_COUNT


_PLACEMENT_GENERATION_BUDGET = len(_PLACEMENT_ARM_KEYS) * _PLACEMENT_MAX_GROUND_RECORDS


_PLACEMENT_MAX_RECORDS_BYTES = 12 * 1024 * 1024


_PLACEMENT_CELL_FIELDS = {
    "verdict",
    "boundary_found",
    "boundary_text_match",
    "input_tokens",
    "answer_digest",
}


_PLACEMENT_DETAILS = ("arm_context_exceeded", "arm_evaluation_failed")


_PLACEMENT_AGGREGATE_KEYS = {
    "arms",
    "closed_book",
    "headline",
    "sandwich",
    "control_level",
    "n_control",
    "placement_effect_falsified",
    "n_fixtures",
    "n_screened",
    "n_closed_book_incomplete",
}


def _placement_digest(text: str) -> str:
    """SHA-256 hex digest (local copy; no imports from
    placement.py anywhere in this mirror).
    """

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _placement_digest_ok(value: Any) -> bool:
    """Whether a stored answer digest is None or a well-formed
    SHA-256 hex digest (fails closed on lone surrogates, the
    cycle-190 discipline).
    """
    if value is None:
        return True
    if type(value) is not str or len(value) != 64:
        return False
    return all(ch in "0123456789abcdef" for ch in value)


def _placement_derive(seed: int, index: int, slot: int, counter: int) -> int:
    """Mirror of placement._derive (frozen material string)."""

    material = f"{seed}|placement|{index}|{slot}|{counter}"
    return int.from_bytes(hashlib.sha256(material.encode("ascii")).digest()[:8], "big")


def _placement_generate(seed: int) -> list[dict[str, Any]]:
    """Mirror of placement.generate_cases: the frozen case set
    for a seed, regenerated from the protocol's own seed so a
    fabricated set is rejected (the numform F2 lesson).
    """
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index in range(_PLACEMENT_CASE_COUNT):
        topic = _PLACEMENT_TOPICS[index]
        counter = 0
        while True:
            word = _PLACEMENT_WORDS[
                _placement_derive(seed, index, 0, counter) % len(_PLACEMENT_WORDS)
            ]
            number = 1000 + _placement_derive(seed, index, 1, counter) % 9000
            value = f"{word}-{number}"
            if value not in seen:
                break
            counter += 1
        seen.add(value)
        first = _placement_derive(seed, index, 2, 0) % len(_PLACEMENT_NEUTRAL_POOL)
        second = _placement_derive(seed, index, 3, 0) % len(_PLACEMENT_NEUTRAL_POOL)
        if second == first:
            second = (first + 1) % len(_PLACEMENT_NEUTRAL_POOL)
        cases.append(
            {
                "fixture_id": f"placement-{index:02d}",
                "topic": topic,
                "value": value,
                "sentence_indices": [first, second],
            }
        )
    return cases


def _placement_case_material_sha256() -> str:
    """Mirror of placement.case_material_sha256."""

    canonical = json.dumps(
        [
            _PLACEMENT_FACT_TEMPLATE,
            _PLACEMENT_QUESTION_TEMPLATE,
            list(_PLACEMENT_TOPICS),
            list(_PLACEMENT_WORDS),
            list(_PLACEMENT_NEUTRAL_POOL),
        ],
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return _placement_digest(canonical)


def _placement_ground_sha256(seed: int) -> str:
    """Mirror of placement.ground_sha256."""

    rows = [
        [case["fixture_id"], case["topic"], case["value"], list(case["sentence_indices"])]
        for case in _placement_generate(seed)
    ]
    canonical = json.dumps(rows, separators=(",", ":"), ensure_ascii=True)
    return _placement_digest(canonical)


def _placement_protocol(seed: int, backend_fingerprint: str) -> dict[str, Any]:
    """The placement protocol rebuilt from this module's own
    literals plus the payload's seed and backend fingerprint,
    fingerprint included (compared whole).
    """

    protocol: dict[str, Any] = {
        "version": 1,
        "arm_keys": list(_PLACEMENT_ARM_KEYS),
        "layout_arms": list(_PLACEMENT_LAYOUT_ARMS),
        "baseline_arm": _PLACEMENT_BASELINE_ARM,
        "case_count": _PLACEMENT_CASE_COUNT,
        "case_material_sha256": _placement_case_material_sha256(),
        "ground_sha256": _placement_ground_sha256(seed),
        "min_paired_evaluable": _PLACEMENT_MIN_PAIRED_EVALUABLE,
        "min_discordant": _PLACEMENT_MIN_DISCORDANT,
        "generation_budget": _PLACEMENT_GENERATION_BUDGET,
        "max_ground_records": _PLACEMENT_MAX_GROUND_RECORDS,
        "max_records_bytes": _PLACEMENT_MAX_RECORDS_BYTES,
        "seed": seed,
        "backend_fingerprint": backend_fingerprint,
    }
    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    protocol["fingerprint"] = _placement_digest(canonical)
    return protocol


def _placement_mcnemar_p(a_only: int, b_only: int) -> float | None:
    """Two-sided exact McNemar p (the frozen formula, local copy)."""

    n = a_only + b_only
    if n == 0:
        return None
    k = min(a_only, b_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    result: float = min(1.0, 2.0 * tail)
    return result


def _placement_cell_get(record: dict[str, Any], key: str) -> dict[str, Any] | None:
    """Return the placement cell get."""
    cells = record.get("cells")
    if not isinstance(cells, dict):
        return None
    cell = cells.get(key)
    return cell if isinstance(cell, dict) else None


def _placement_verdict(record: dict[str, Any], key: str) -> bool:
    """Return the placement verdict."""
    cell = _placement_cell_get(record, key)
    return bool(cell is not None and cell["verdict"])


def _placement_screened(record: dict[str, Any]) -> bool:
    """Return the placement screened."""
    cell = _placement_cell_get(record, "closed_book")
    return cell is not None and bool(cell["verdict"])


def _placement_pair_block(
    records: list[dict[str, Any]],
    key_a: str,
    key_b: str,
    restrict: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Mirror of placement._pair_block."""
    n_paired = 0
    n_a_only = 0
    n_b_only = 0
    n_both = 0
    n_pass_a = 0
    n_pass_b = 0
    for record in records:
        if restrict is not None and record.get("fixture") not in restrict:
            continue
        cell_a = _placement_cell_get(record, key_a)
        cell_b = _placement_cell_get(record, key_b)
        if cell_a is None or cell_b is None:
            continue
        n_paired += 1
        verdict_a = bool(cell_a["verdict"])
        verdict_b = bool(cell_b["verdict"])
        if verdict_a:
            n_pass_a += 1
        if verdict_b:
            n_pass_b += 1
        if verdict_a and verdict_b:
            n_both += 1
        elif verdict_a:
            n_a_only += 1
        elif verdict_b:
            n_b_only += 1
    return {
        "n_paired": n_paired,
        "n_a_only": n_a_only,
        "n_b_only": n_b_only,
        "n_both": n_both,
        "n_neither": n_paired - n_a_only - n_b_only - n_both,
        "rate_a": (n_pass_a / n_paired) if n_paired else None,
        "rate_b": (n_pass_b / n_paired) if n_paired else None,
        "mcnemar_p": _placement_mcnemar_p(n_a_only, n_b_only),
    }


def _placement_both_fail(qf_only: int, pf_only: int, n_neither: int) -> bool:
    """Mirror of placement.both_fail_dominant."""
    return n_neither >= max(qf_only, pf_only) and n_neither >= _PLACEMENT_MIN_DISCORDANT


def _placement_classify(
    n_paired: int,
    qf_only: int,
    pf_only: int,
    n_control: int,
    control_level: int | None,
    joint_collapse: bool,
) -> str:
    """Mirror of placement.classify_placement."""
    d = qf_only + pf_only
    if n_paired < _PLACEMENT_MIN_PAIRED_EVALUABLE:
        return "insufficient"
    if d == 0 and not joint_collapse:
        return "no_discordance_observed"
    if n_control < _PLACEMENT_MIN_PAIRED_EVALUABLE:
        return "control_set_too_small"
    if control_level is None:
        return "control_unavailable"
    if d <= control_level:
        return "within_sandwich_control"
    if qf_only == pf_only and qf_only >= _PLACEMENT_MIN_DISCORDANT:
        return "discordant_balanced"
    if max(qf_only, pf_only) >= _PLACEMENT_MIN_DISCORDANT:
        return "discordant_asymmetric"
    return "below_threshold"


def _placement_falsified(
    n_paired: int,
    qf_only: int,
    pf_only: int,
    n_control: int,
    control_level: int | None,
    joint_collapse: bool,
) -> bool:
    """Mirror of placement.effect_falsified."""
    if n_paired < _PLACEMENT_MIN_PAIRED_EVALUABLE or joint_collapse:
        return False
    d = qf_only + pf_only
    if d == 0:
        return True
    return (
        n_control >= _PLACEMENT_MIN_PAIRED_EVALUABLE
        and control_level is not None
        and d <= control_level
    )


def _placement_aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Mirror of placement.aggregate_candidate."""
    live = [r for r in records if not _placement_screened(r)]
    arms: dict[str, Any] = {}
    for key in _PLACEMENT_ARM_KEYS:
        n_completed = 0
        n_pass = 0
        for record in live:
            cell = _placement_cell_get(record, key)
            if cell is None:
                continue
            n_completed += 1
            if cell["verdict"]:
                n_pass += 1
        arms[key] = {
            "n_completed": n_completed,
            "n_pass": n_pass,
            "rate": (n_pass / n_completed) if n_completed else None,
        }
    cb_completed = 0
    cb_pass = 0
    for record in records:
        cell = _placement_cell_get(record, "closed_book")
        if cell is None:
            continue
        cb_completed += 1
        if cell["verdict"]:
            cb_pass += 1
    headline_raw = _placement_pair_block(live, "question_first", "payload_first")
    paired_ids = frozenset(
        str(record.get("fixture"))
        for record in live
        if _placement_cell_get(record, "question_first") is not None
        and _placement_cell_get(record, "payload_first") is not None
    )
    sb_pf = _placement_pair_block(live, "sandwich", "payload_first", paired_ids)
    sb_qf = _placement_pair_block(live, "sandwich", "question_first", paired_ids)
    n_control = sb_pf["n_paired"]
    control_level: int | None = None
    if n_control:
        control_level = max(
            sb_pf["n_a_only"] + sb_pf["n_b_only"],
            sb_qf["n_a_only"] + sb_qf["n_b_only"],
        )
    qf_only = headline_raw["n_a_only"]
    pf_only = headline_raw["n_b_only"]
    joint = _placement_both_fail(qf_only, pf_only, headline_raw["n_neither"])
    headline = {
        "n_paired": headline_raw["n_paired"],
        "n_qf_only": qf_only,
        "n_pf_only": pf_only,
        "n_both": headline_raw["n_both"],
        "n_neither": headline_raw["n_neither"],
        "rate_qf": headline_raw["rate_a"],
        "rate_pf": headline_raw["rate_b"],
        "mcnemar_p": headline_raw["mcnemar_p"],
        "both_fail_dominant": joint,
        "classification": _placement_classify(
            headline_raw["n_paired"], qf_only, pf_only, n_control, control_level, joint
        ),
    }
    return {
        "arms": arms,
        "closed_book": {
            "n_completed": cb_completed,
            "n_pass": cb_pass,
            "pass_rate": (cb_pass / cb_completed) if cb_completed else None,
        },
        "headline": headline,
        "sandwich": {
            "vs_payload_first": sb_pf,
            "vs_question_first": sb_qf,
        },
        "control_level": control_level,
        "n_control": n_control,
        "placement_effect_falsified": _placement_falsified(
            headline_raw["n_paired"], qf_only, pf_only, n_control, control_level, joint
        ),
        "n_fixtures": len(records),
        "n_screened": len(records) - len(live),
        "n_closed_book_incomplete": sum(
            1 for record in records if _placement_cell_get(record, "closed_book") is None
        ),
    }


def _placement_selection_over(
    records_by_candidate: dict[str, list[dict[str, Any]]],
    key: str,
    fixtures: frozenset[str],
) -> dict[str, Any] | None:
    """Mirror of placement._selection_over."""
    completed_sets: list[frozenset[str]] = []
    all_pass: dict[str, bool | None] = {}
    for name, records in records_by_candidate.items():
        completed = frozenset(
            str(record["fixture"])
            for record in records
            if record["fixture"] in fixtures and _placement_cell_get(record, key) is not None
        )
        completed_sets.append(completed)
        if not completed:
            return None
        all_pass[name] = all(
            _placement_verdict(record, key) for record in records if record["fixture"] in completed
        )
    if len(set(completed_sets)) > 1:
        return None
    passers = [name for name, ok in all_pass.items() if ok]
    if len(passers) == 1:
        return {"decision": "selected", "winner": passers[0]}
    if len(passers) > 1:
        return {"decision": "ambiguous", "winner": None}
    return {"decision": "no_candidate", "winner": None}


def _placement_arm_selection(
    records_by_candidate: dict[str, list[dict[str, Any]]],
    arm: str,
) -> dict[str, Any]:
    """Mirror of placement.arm_selection."""
    populations: list[frozenset[str]] = []
    for records in records_by_candidate.values():
        populations.append(
            frozenset(str(r.get("fixture")) for r in records if not _placement_screened(r))
        )
    shared: frozenset[str] = frozenset.intersection(*populations) if populations else frozenset()
    own = _placement_selection_over(records_by_candidate, arm, shared)
    if own is None:
        return {
            "decision": "unavailable",
            "detail": "selection_withheld_mismatched_sets",
            "winner": None,
            "withheld": True,
            "flip": None,
        }
    if arm == _PLACEMENT_BASELINE_ARM:
        flip: bool | None = False
    else:
        completed = frozenset(
            str(record["fixture"])
            for records in records_by_candidate.values()
            for record in records
            if record["fixture"] in shared and _placement_cell_get(record, arm) is not None
        )
        baseline = _placement_selection_over(
            records_by_candidate, _PLACEMENT_BASELINE_ARM, completed
        )
        flip = None if baseline is None else own != baseline
    return {
        "decision": own["decision"],
        "detail": "",
        "winner": own["winner"],
        "withheld": False,
        "flip": flip,
    }


def _placement_cell_ok(cell: Any, expected_digest: str) -> bool:
    """One cell's shape and the verdict binding
    ``verdict == (answer_digest == expected_digest) and
    boundary_found``. The expected digest is the record's,
    itself checked against the mirror-regenerated case.
    """
    if type(cell) is not dict or set(cell) != _PLACEMENT_CELL_FIELDS:
        return False
    if type(cell["verdict"]) is not bool or type(cell["boundary_found"]) is not bool:
        return False
    if cell["boundary_text_match"] is not None and type(cell["boundary_text_match"]) is not bool:
        return False
    if type(cell["input_tokens"]) is not int or cell["input_tokens"] < 0:
        return False
    if not _placement_digest_ok(cell["answer_digest"]):
        return False
    verdict = (cell["answer_digest"] == expected_digest) and cell["boundary_found"]
    return bool(cell["verdict"] == verdict)


def _placement_recompute(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Recompute a placement payload from its ground records.

    The validator carries its own generator mirror, templates,
    pools, and aggregators (no imports from placement.py): the
    protocol is rebuilt from frozen scalars plus the payload's
    seed and backend fingerprint and compared whole; the case
    set is regenerated from the seed and every record's
    construction is compared against it; every cell's verdict
    is re-derived from its digest identity; every completed
    cell's ``input_tokens`` must equal the record's
    ``arm_tokens`` for that arm; the aggregates and the per-arm
    selections are recomputed and compared.
    """

    protocol = payload.get("protocol")
    if type(protocol) is not dict:
        return None
    seed = protocol.get("seed")
    backend_fingerprint = protocol.get("backend_fingerprint")
    if type(seed) is not int or isinstance(seed, bool) or type(backend_fingerprint) is not str:
        return None
    if not _recomputed_equal(protocol, _placement_protocol(seed, backend_fingerprint)):
        return None
    candidates = payload.get("candidates")
    if type(candidates) is not dict or not candidates:
        return None
    generated = _placement_generate(seed)
    total_records = 0
    records_bytes = 0
    aggregates: dict[str, Any] = {}
    raw_records: dict[str, list[dict[str, Any]]] = {}
    for name, entry in candidates.items():
        if type(entry) is not dict:
            return None
        status = entry.get("status")
        if status == "unavailable":
            if set(entry) != {"status", "detail", "records"} or entry.get("records") != []:
                return None
            continue
        if status != "available" or entry.get("detail") != "":
            return None
        if set(entry) != _PLACEMENT_AGGREGATE_KEYS | {"status", "detail", "records"}:
            return None
        records = entry["records"]
        if type(records) is not list or len(records) != len(generated):
            return None
        total_records += len(records)
        records_bytes += len(
            json.dumps(records, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        )
        if records_bytes > _PLACEMENT_MAX_RECORDS_BYTES:
            return None
        for record, case in zip(records, generated, strict=True):
            if type(record) is not dict or set(record) != {
                "fixture",
                "topic",
                "expected_digest",
                "question_tokens",
                "arm_tokens",
                "cells",
                "cell_details",
            }:
                return None
            if record["fixture"] != case["fixture_id"] or record["topic"] != case["topic"]:
                return None
            expected_digest = _placement_digest(str(case["value"]))
            if record["expected_digest"] != expected_digest:
                return None
            arm_tokens = record["arm_tokens"]
            if type(arm_tokens) is not dict or set(arm_tokens) != set(_PLACEMENT_ARM_KEYS):
                return None
            if any(type(v) is not int or v < 0 for v in arm_tokens.values()):
                return None
            if record["question_tokens"] != arm_tokens["closed_book"]:
                return None
            cells = record["cells"]
            details = record["cell_details"]
            if type(cells) is not dict or set(cells) != set(_PLACEMENT_ARM_KEYS):
                return None
            if type(details) is not dict or set(details) != set(_PLACEMENT_ARM_KEYS):
                return None
            for arm in _PLACEMENT_ARM_KEYS:
                cell = cells[arm]
                detail = details[arm]
                if cell is None:
                    if detail not in _PLACEMENT_DETAILS:
                        return None
                else:
                    if detail != "":
                        return None
                    if not _placement_cell_ok(cell, expected_digest):
                        return None
                    if cell["input_tokens"] != arm_tokens[arm]:
                        return None
        raw_records[name] = records
        aggregates[name] = _placement_aggregate(records)
    if total_records > _PLACEMENT_MAX_GROUND_RECORDS:
        return None
    if not raw_records:
        return None
    return {
        "aggregates": aggregates,
        "conditions": {
            arm: _placement_arm_selection(raw_records, arm) for arm in _PLACEMENT_LAYOUT_ARMS
        },
        "n_candidates": len(candidates),
        "n_cases": len(generated),
    }


def _valid_placement(payload: Any) -> bool:
    """Structural check for a ``placement`` payload (recompute standard)."""
    if type(payload) is not dict or payload.get("available") is not True:
        return False
    if set(payload) != {
        "available",
        "detail",
        "protocol",
        "candidates",
        "conditions",
        "n_candidates",
        "n_cases",
    }:
        return False
    if payload.get("detail") != "":
        return False
    recomputed = _placement_recompute(payload)
    if recomputed is None:
        return False
    candidates = payload["candidates"]
    for name, expected in recomputed["aggregates"].items():
        entry = candidates[name]
        for key, value in expected.items():
            if not _recomputed_equal(entry.get(key), value):
                return False
    if not _recomputed_equal(payload.get("conditions"), recomputed["conditions"]):
        return False
    if not _recomputed_equal(payload.get("n_candidates"), recomputed["n_candidates"]):
        return False
    return _recomputed_equal(payload.get("n_cases"), recomputed["n_cases"])


def diff_placement(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``placement`` probe payloads; deterministic.

    Answers "did the placement story change between runs?":
    per-candidate classification and falsification drift, the
    directional discordant counts, control drift, screening
    drift, per-arm selection drift, candidates added/removed,
    and protocol / generator / backend drift (any of which
    marks the comparison not comparable, with the drift still
    listed -- the generated set is the treatment, so a changed
    generator is protocol drift). A side without a valid
    payload is reported plainly.
    """
    old_ok = _valid_placement(old)
    new_ok = _valid_placement(new)
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
    placement_protocol_drift = old["protocol"].get("fingerprint") != new["protocol"].get(
        "fingerprint"
    )
    placement_generator_drift = old["protocol"].get("ground_sha256") != new["protocol"].get(
        "ground_sha256"
    )
    placement_backend_drift = old["protocol"].get("backend_fingerprint") != new["protocol"].get(
        "backend_fingerprint"
    )
    placement_comparable = (
        not placement_protocol_drift
        and not placement_generator_drift
        and not placement_backend_drift
    )
    old_candidates = old["candidates"]
    new_candidates = new["candidates"]
    placement_candidates_added = sorted(set(new_candidates) - set(old_candidates))
    placement_candidates_removed = sorted(set(old_candidates) - set(new_candidates))
    placement_candidate_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_candidates) & set(new_candidates)):
        old_entry = old_candidates[name]
        new_entry = new_candidates[name]
        if old_entry.get("status") != "available" or new_entry.get("status") != "available":
            if old_entry.get("status") != new_entry.get("status"):
                placement_candidate_drift.append(
                    {
                        "candidate": name,
                        "metric": "status",
                        "old": old_entry.get("status"),
                        "new": new_entry.get("status"),
                    }
                )
            continue
        placement_metrics: list[tuple[str, Any, Any]] = [
            (
                "classification",
                old_entry["headline"]["classification"],
                new_entry["headline"]["classification"],
            ),
            (
                "placement_effect_falsified",
                old_entry["placement_effect_falsified"],
                new_entry["placement_effect_falsified"],
            ),
            (
                "n_qf_only",
                old_entry["headline"]["n_qf_only"],
                new_entry["headline"]["n_qf_only"],
            ),
            (
                "n_pf_only",
                old_entry["headline"]["n_pf_only"],
                new_entry["headline"]["n_pf_only"],
            ),
            (
                "control_level",
                old_entry["control_level"],
                new_entry["control_level"],
            ),
            (
                "n_screened",
                old_entry["n_screened"],
                new_entry["n_screened"],
            ),
        ]
        for metric, old_value, new_value in placement_metrics:
            if old_value != new_value:
                placement_candidate_drift.append(
                    {
                        "candidate": name,
                        "metric": metric,
                        "old": old_value,
                        "new": new_value,
                    }
                )
    placement_condition_drift: list[dict[str, Any]] = []
    for arm in _PLACEMENT_LAYOUT_ARMS:
        if old["conditions"][arm] != new["conditions"][arm]:
            placement_condition_drift.append(
                {
                    "condition": arm,
                    "old": old["conditions"][arm],
                    "new": new["conditions"][arm],
                }
            )
    result["candidate_drift"] = placement_candidate_drift
    result["condition_drift"] = placement_condition_drift
    result["candidates_added"] = placement_candidates_added
    result["candidates_removed"] = placement_candidates_removed
    result["protocol_drift"] = placement_protocol_drift
    result["generator_drift"] = placement_generator_drift
    result["backend_drift"] = placement_backend_drift
    result["comparable"] = placement_comparable
    result["changed"] = bool(
        placement_candidate_drift
        or placement_condition_drift
        or placement_candidates_added
        or placement_candidates_removed
        or placement_protocol_drift
        or placement_generator_drift
        or placement_backend_drift
    )
    return result


_INSTRUCTIONDISTANCE_ARM_KEYS = (
    "adj",
    "sep_256",
    "sep_1024",
    "disp_256",
    "disp_1024",
    "adj_after",
)


_INSTRUCTIONDISTANCE_SIZED_ARMS = ("sep_256", "sep_1024", "disp_256", "disp_1024")


_INSTRUCTIONDISTANCE_DOSES = (256, 1024)


_INSTRUCTIONDISTANCE_CASE_COUNT = 24


_INSTRUCTIONDISTANCE_CONSTRAINT = "Reply with only the number: no units, no sentence, nothing else."


_INSTRUCTIONDISTANCE_FACT_TEMPLATE = "The {topic} number is {value}."


_INSTRUCTIONDISTANCE_QUESTION_TEMPLATE = "What is the {topic} number?"


_INSTRUCTIONDISTANCE_TOPICS = (
    "harbor",
    "summit",
    "quarry",
    "tundra",
    "granite",
    "willow",
    "copper",
    "saddle",
    "walnut",
    "velvet",
    "cobalt",
    "marble",
    "cinder",
    "falcon",
    "garnet",
    "saffron",
    "indigo",
    "juniper",
    "lagoon",
    "magnet",
    "nickel",
    "onyx",
    "pepper",
    "quartz",
)


_INSTRUCTIONDISTANCE_MIN_D = 10


_INSTRUCTIONDISTANCE_MIN_S = 6


_INSTRUCTIONDISTANCE_MIN_MARGIN = 5


_INSTRUCTIONDISTANCE_MAX_GROUND_RECORDS = 24 * _INSTRUCTIONDISTANCE_CASE_COUNT


_INSTRUCTIONDISTANCE_GENERATION_BUDGET = (
    len(_INSTRUCTIONDISTANCE_ARM_KEYS) * _INSTRUCTIONDISTANCE_MAX_GROUND_RECORDS
)


_INSTRUCTIONDISTANCE_MAX_RECORDS_BYTES = 12 * 1024 * 1024


_INSTRUCTIONDISTANCE_SIZING_ENCODE_BUDGET = 512


_INSTRUCTIONDISTANCE_STREAM_CYCLES = 6


_INSTRUCTIONDISTANCE_CELL_FIELDS = {
    "verdict",
    "boundary_found",
    "boundary_text_match",
    "input_tokens",
    "answer_digest",
    "first_digit_run_digest",
    "nondigit_projection_digest",
    "answer_len",
    "digit_count",
}


_INSTRUCTIONDISTANCE_DETAILS = (
    "arm_context_exceeded",
    "arm_evaluation_failed",
    "arm_unmatchable",
    "arm_screened",
)


_INSTRUCTIONDISTANCE_UNSIZED_DETAILS = ("arm_context_exceeded", "arm_evaluation_failed")


_INSTRUCTIONDISTANCE_AGGREGATE_KEYS = {
    "arms",
    "doses",
    "ids",
    "operating_characteristics",
    "n_cases",
    "n_incomplete",
}


_INSTRUCTIONDISTANCE_SIZING_RULE_TEXT = (
    "instructiondistance-single-phase-exact-v1: spans are whole "
    "sentences plus one truncated suffix (span_text) over the "
    "cycled neutral corpus stream, every sized cell searching "
    "from the stream head; candidates are probed in "
    "span_candidates closeness order against the dose, each "
    "probe rendering the full arm content under the candidate's "
    "own template, accepting the first whose rendered token "
    "count minus the case's adjacent-arm rendered token count "
    "equals the dose exactly, verified on the accepted "
    "construction's final ids (the probe render IS the arm "
    "render, so acceptance and verification are one encode). "
    "One encode per probe; the per-cell encode budget is 512 "
    "and a cell with no exact construction inside it is "
    "arm_unmatchable, never approximated. Unlike needle's "
    "phased rule there is no in-span offset-verification phase: "
    "this instrument has no offset variable (the filler sits "
    "wholly between the directive and the block, or wholly "
    "after it) and the newline joiner insulates the span from "
    "the block it precedes or follows, so delta-exactness on "
    "the final render is the only acceptance condition."
)


_INSTRUCTIONDISTANCE_SCREENING_RULE_TEXT = (
    "instructiondistance-screen-v1: per constructed filler "
    "span, case-folded: the span is screened (the cell is "
    "arm_screened) when it contains the case value, or when "
    "the candidate's decoded boundary text occurs in it. The "
    "constraint and the case block are the treatment and are "
    "never screened per-cell. Stage-level, the frozen texts "
    "(constraint, fact and question templates, topics, neutral "
    "corpus) are audited for special-token literals against "
    "every candidate format, and the generated texts "
    "(constraint, templates with slots emptied, topics) are "
    "checked for containment of any candidate's decoded "
    "boundary text; a hit fails the whole stage closed before "
    "any generation."
)


_INSTRUCTIONDISTANCE_OPERATING_CHARACTERISTICS: dict[str, Any] = {
    "random_labeller_fire_rate": 0.0013,
    "no_effect_competent_fire_rate": 0.0058,
    "no_effect_paired_max_fire_rate": 0.0077,
    "power_drop_rate_0_25": 0.246,
    "power_drop_rate_0_50": 0.9088,
    "power_drop_rate_0_75": 0.9993,
    "pure_length_independent_fire_rate": 0.0788,
    "pure_length_fully_correlated_fire_rate": 0.0,
    "pure_length_negative_rho_max_fire_rate": 0.133,
    "familywise_decisions": 48,
    "familywise_no_effect_expected": 0.28,
    "familywise_no_effect_p_at_least_one": 0.24,
    "familywise_frailty_expected_low": 0.40,
    "familywise_frailty_expected_high": 0.59,
    "familywise_frailty_p_low": 0.33,
    "familywise_frailty_p_high": 0.44,
    "familywise_pure_length_expected": 3.78,
    "familywise_pure_length_p_at_least_one": 0.98,
    "verdict_availability_at_adjacent_cc_0_40": 0.511,
    "verdict_availability_at_adjacent_cc_0_30": 0.153,
}


_INSTRUCTIONDISTANCE_DECOMPOSITION_CLASSES = (
    "compliant_correct",
    "compliant_wrong",
    "noncompliant_correct_value",
    "noncompliant_wrong",
    "boundary_only",
)


_INSTRUCTIONDISTANCE_NONCOMPLIANT = ("noncompliant_correct_value", "noncompliant_wrong")


_INSTRUCTIONDISTANCE_COMPLIANT = ("compliant_correct", "compliant_wrong")


def _instructiondistance_digest(text: str) -> str:
    """SHA-256 hex digest (local copy; no imports from
    instructiondistance.py anywhere in this mirror).
    """

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _instructiondistance_digest_ok(value: Any) -> bool:
    """Whether a stored digest is a well-formed SHA-256 hex
    digest (this probe's answer digests are never None: a
    completed cell always carries a decoded span).
    """
    if type(value) is not str or len(value) != 64:
        return False
    return all(ch in "0123456789abcdef" for ch in value)


def _instructiondistance_run_digest_ok(value: Any) -> bool:
    """The first-digit-run digest field: "" (no ASCII digit
    run, amendment B3) or a well-formed digest.
    """
    if value == "":
        return True
    return _instructiondistance_digest_ok(value)


def _instructiondistance_derive(seed: int, index: int, slot: int, counter: int) -> int:
    """Mirror of instructiondistance._derive (frozen material
    string).
    """

    material = f"{seed}|instructiondistance|{index}|{slot}|{counter}"
    return int.from_bytes(hashlib.sha256(material.encode("ascii")).digest()[:8], "big")


def _instructiondistance_generate(seed: int) -> list[dict[str, Any]]:
    """Mirror of instructiondistance.generate_cases: the
    frozen case set for a seed, regenerated from the
    protocol's own seed so a fabricated set is rejected (the
    numform F2 lesson).
    """
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index in range(_INSTRUCTIONDISTANCE_CASE_COUNT):
        topic = _INSTRUCTIONDISTANCE_TOPICS[index]
        counter = 0
        while True:
            value = str(100 + _instructiondistance_derive(seed, index, 0, counter) % 900)
            if value not in seen:
                break
            counter += 1
        seen.add(value)
        cases.append(
            {
                "fixture_id": f"instructiondistance-{index:02d}",
                "topic": topic,
                "value": value,
            }
        )
    return cases


def _instructiondistance_case_material_sha256() -> str:
    """Mirror of instructiondistance.case_material_sha256."""

    canonical = json.dumps(
        [
            _INSTRUCTIONDISTANCE_CONSTRAINT,
            _INSTRUCTIONDISTANCE_FACT_TEMPLATE,
            _INSTRUCTIONDISTANCE_QUESTION_TEMPLATE,
            list(_INSTRUCTIONDISTANCE_TOPICS),
        ],
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return _instructiondistance_digest(canonical)


def _instructiondistance_ground_sha256(seed: int) -> str:
    """Mirror of instructiondistance.ground_sha256."""

    rows = [
        [case["fixture_id"], case["topic"], case["value"]]
        for case in _instructiondistance_generate(seed)
    ]
    canonical = json.dumps(rows, separators=(",", ":"), ensure_ascii=True)
    return _instructiondistance_digest(canonical)


def _instructiondistance_protocol(seed: int, backend_fingerprint: str) -> dict[str, Any]:
    """The instruction-distance protocol rebuilt from this
    module's own literals plus the payload's seed and backend
    fingerprint, fingerprint included (compared whole).
    """

    protocol: dict[str, Any] = {
        "version": 1,
        "arm_keys": list(_INSTRUCTIONDISTANCE_ARM_KEYS),
        "doses": list(_INSTRUCTIONDISTANCE_DOSES),
        "baseline_arm": "adj",
        "case_count": _INSTRUCTIONDISTANCE_CASE_COUNT,
        "case_material_sha256": _instructiondistance_case_material_sha256(),
        "ground_sha256": _instructiondistance_ground_sha256(seed),
        "sizing_rule_sha256": _instructiondistance_digest(_INSTRUCTIONDISTANCE_SIZING_RULE_TEXT),
        "sizing_encode_budget": _INSTRUCTIONDISTANCE_SIZING_ENCODE_BUDGET,
        "screening_rule_sha256": _instructiondistance_digest(
            _INSTRUCTIONDISTANCE_SCREENING_RULE_TEXT
        ),
        "stream_cycles": _INSTRUCTIONDISTANCE_STREAM_CYCLES,
        "min_d": _INSTRUCTIONDISTANCE_MIN_D,
        "min_s": _INSTRUCTIONDISTANCE_MIN_S,
        "min_margin": _INSTRUCTIONDISTANCE_MIN_MARGIN,
        "generation_budget": _INSTRUCTIONDISTANCE_GENERATION_BUDGET,
        "max_ground_records": _INSTRUCTIONDISTANCE_MAX_GROUND_RECORDS,
        "max_records_bytes": _INSTRUCTIONDISTANCE_MAX_RECORDS_BYTES,
        "seed": seed,
        "backend_fingerprint": backend_fingerprint,
    }
    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    protocol["fingerprint"] = _instructiondistance_digest(canonical)
    return protocol


def _instructiondistance_cell_get(record: dict[str, Any], key: str) -> dict[str, Any] | None:
    """Return the instructiondistance cell get."""
    cells = record.get("cells")
    if not isinstance(cells, dict):
        return None
    cell = cells.get(key)
    return cell if isinstance(cell, dict) else None


def _instructiondistance_decompose(cell: dict[str, Any], expected_digest: str) -> str:
    """Mirror of instructiondistance.decompose_cell: the
    class from the recorded digests alone (amendment C --
    compliance rides on the answer and non-digit-projection
    digests, never on the descriptive integer counts, so a
    forged digit_count cannot move a class).
    """
    empty = _instructiondistance_digest("")
    answer_digest = cell.get("answer_digest")
    compliant = (
        type(answer_digest) is str
        and answer_digest != empty
        and cell.get("nondigit_projection_digest") == empty
    )
    if compliant and answer_digest == expected_digest:
        return "compliant_correct" if cell.get("boundary_found") else "boundary_only"
    if compliant:
        return "compliant_wrong"
    run_digest = cell.get("first_digit_run_digest")
    if type(run_digest) is str and run_digest != "" and run_digest == expected_digest:
        return "noncompliant_correct_value"
    return "noncompliant_wrong"


def _instructiondistance_classify(direct_passes: int, separated: int, displaced: int) -> str:
    """Mirror of instructiondistance.classify_instruction_distance."""
    if direct_passes < _INSTRUCTIONDISTANCE_MIN_D:
        return "insufficient"
    if (
        separated >= _INSTRUCTIONDISTANCE_MIN_S
        and separated - displaced >= _INSTRUCTIONDISTANCE_MIN_MARGIN
    ):
        return "instruction_distance_drop_observed"
    return "below_threshold"


def _instructiondistance_aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Mirror of instructiondistance.aggregate_candidate."""
    arms: dict[str, Any] = {}
    for key in _INSTRUCTIONDISTANCE_ARM_KEYS:
        n_completed = 0
        n_pass = 0
        n_compliant = 0
        n_context = 0
        n_failed = 0
        n_unmatchable = 0
        n_screened = 0
        decomposition: dict[str, int] = dict.fromkeys(_INSTRUCTIONDISTANCE_DECOMPOSITION_CLASSES, 0)
        for record in records:
            details = record.get("cell_details")
            detail = details.get(key, "") if isinstance(details, dict) else ""
            if detail == "arm_context_exceeded":
                n_context += 1
            elif detail == "arm_evaluation_failed":
                n_failed += 1
            elif detail == "arm_unmatchable":
                n_unmatchable += 1
            elif detail == "arm_screened":
                n_screened += 1
            cell = _instructiondistance_cell_get(record, key)
            if cell is None:
                continue
            n_completed += 1
            if cell["verdict"]:
                n_pass += 1
            cls = _instructiondistance_decompose(cell, str(record.get("expected_digest", "")))
            decomposition[cls] = decomposition.get(cls, 0) + 1
            if cls in _INSTRUCTIONDISTANCE_COMPLIANT:
                n_compliant += 1
        arms[key] = {
            "n_completed": n_completed,
            "n_pass": n_pass,
            "rate": (n_pass / n_completed) if n_completed else None,
            "n_compliant": n_compliant,
            "compliance_rate": (n_compliant / n_completed) if n_completed else None,
            "n_context_exceeded": n_context,
            "n_evaluation_failed": n_failed,
            "n_unmatchable": n_unmatchable,
            "n_screened": n_screened,
            "decomposition": decomposition,
        }
    doses: dict[str, Any] = {}
    for dose in _INSTRUCTIONDISTANCE_DOSES:
        sep = f"sep_{dose}"
        disp = f"disp_{dose}"
        n_population = 0
        direct_passes = 0
        separated_losses = 0
        displaced_losses = 0
        s_value_kept = 0
        s_value_lost = 0
        n_reverse = 0
        for record in records:
            expected_digest = str(record.get("expected_digest", ""))
            adj = _instructiondistance_cell_get(record, "adj")
            sep_cell = _instructiondistance_cell_get(record, sep)
            disp_cell = _instructiondistance_cell_get(record, disp)
            if adj is not None and sep_cell is not None:
                adj_cls = _instructiondistance_decompose(adj, expected_digest)
                sep_cls = _instructiondistance_decompose(sep_cell, expected_digest)
                if (
                    adj_cls in _INSTRUCTIONDISTANCE_NONCOMPLIANT
                    and sep_cls in _INSTRUCTIONDISTANCE_COMPLIANT
                ):
                    n_reverse += 1
            if adj is None or sep_cell is None or disp_cell is None:
                continue
            n_population += 1
            if _instructiondistance_decompose(adj, expected_digest) != "compliant_correct":
                continue
            direct_passes += 1
            sep_cls = _instructiondistance_decompose(sep_cell, expected_digest)
            if sep_cls in _INSTRUCTIONDISTANCE_NONCOMPLIANT:
                separated_losses += 1
                if sep_cls == "noncompliant_correct_value":
                    s_value_kept += 1
                else:
                    s_value_lost += 1
            if (
                _instructiondistance_decompose(disp_cell, expected_digest)
                in _INSTRUCTIONDISTANCE_NONCOMPLIANT
            ):
                displaced_losses += 1
        doses[str(dose)] = {
            "n_population": n_population,
            "direct_passes": direct_passes,
            "separated_losses": separated_losses,
            "displaced_losses": displaced_losses,
            "margin": separated_losses - displaced_losses,
            "s_value_kept": s_value_kept,
            "s_value_lost": s_value_lost,
            "n_reverse": n_reverse,
            "decision": _instructiondistance_classify(
                direct_passes, separated_losses, displaced_losses
            ),
        }
    ids: dict[str, Any] = {}
    for key in _INSTRUCTIONDISTANCE_ARM_KEYS:
        if key == "adj":
            continue
        deltas: list[int] = []
        n_ids_identical = 0
        n_same_ids_flips = 0
        for record in records:
            arm_tokens = record.get("arm_tokens") or {}
            id_digests = record.get("id_digests") or {}
            adj_tokens = arm_tokens.get("adj")
            other_tokens = arm_tokens.get(key)
            if type(adj_tokens) is int and type(other_tokens) is int:
                deltas.append(other_tokens - adj_tokens)
            else:
                deltas.append(0)
            adj_digest = id_digests.get("adj")
            other_digest = id_digests.get(key)
            if type(adj_digest) is str and adj_digest != "" and adj_digest == other_digest:
                n_ids_identical += 1
                adj_cell = _instructiondistance_cell_get(record, "adj")
                other_cell = _instructiondistance_cell_get(record, key)
                if adj_cell is not None and other_cell is not None:
                    expected_digest = str(record.get("expected_digest", ""))
                    adj_compliant = (
                        _instructiondistance_decompose(adj_cell, expected_digest)
                        in _INSTRUCTIONDISTANCE_COMPLIANT
                    )
                    other_compliant = (
                        _instructiondistance_decompose(other_cell, expected_digest)
                        in _INSTRUCTIONDISTANCE_COMPLIANT
                    )
                    if adj_compliant != other_compliant:
                        n_same_ids_flips += 1
        ids[key] = {
            "id_length_delta_by_case": deltas,
            "max_abs_id_delta": max((abs(d) for d in deltas), default=0),
            "n_ids_identical": n_ids_identical,
            "n_same_ids_compliance_flips": n_same_ids_flips,
        }
    return {
        "arms": arms,
        "doses": doses,
        "ids": ids,
        "operating_characteristics": dict(_INSTRUCTIONDISTANCE_OPERATING_CHARACTERISTICS),
        "n_cases": len(records),
        "n_incomplete": sum(
            1
            for record in records
            if any(
                _instructiondistance_cell_get(record, arm) is None
                for arm in ("adj", "sep_256", "sep_1024", "disp_256", "disp_1024")
            )
        ),
    }


def _instructiondistance_cell_ok(cell: Any, expected_digest: str) -> bool:
    """One cell's shape and bindings: the verdict binding
    ``verdict == (answer_digest == expected_digest) and
    boundary_found`` plus the digest shapes of the
    amendment-C primitives (the descriptive counts are
    shape-checked and bounded by the span length, never
    trusted as verdict inputs).
    """
    if type(cell) is not dict or set(cell) != _INSTRUCTIONDISTANCE_CELL_FIELDS:
        return False
    if type(cell["verdict"]) is not bool or type(cell["boundary_found"]) is not bool:
        return False
    if cell["boundary_text_match"] is not None and type(cell["boundary_text_match"]) is not bool:
        return False
    if type(cell["input_tokens"]) is not int or cell["input_tokens"] < 0:
        return False
    if not _instructiondistance_digest_ok(cell["answer_digest"]):
        return False
    if not _instructiondistance_run_digest_ok(cell["first_digit_run_digest"]):
        return False
    if not _instructiondistance_digest_ok(cell["nondigit_projection_digest"]):
        return False
    if type(cell["answer_len"]) is not int or cell["answer_len"] < 0:
        return False
    if type(cell["digit_count"]) is not int or cell["digit_count"] < 0:
        return False
    if cell["digit_count"] > cell["answer_len"]:
        return False
    verdict = (cell["answer_digest"] == expected_digest) and cell["boundary_found"]
    return bool(cell["verdict"] == verdict)


def _instructiondistance_recompute(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Recompute an instruction-distance payload from its
    ground records.

    The validator carries its own generator mirror, frozen
    texts, and aggregators (no imports from
    instructiondistance.py): the protocol is rebuilt from
    frozen literals plus the payload's seed and backend
    fingerprint and compared whole; the case set is
    regenerated from the seed and every record is checked
    against it; every constructed sized arm's token count
    must sit exactly its dose above the adjacent arm's (the
    sizing rule's acceptance identity, checked on the stored
    counts); unconstructed arms (unmatchable, or
    context-exceeded before sizing) carry no token count and
    no id digest; every cell's verdict is re-derived from its
    digest identity and bound to the record's arm token
    count; the decomposition and the (D, S, T) decisions are
    recomputed from the recorded digests alone; the
    aggregates are recomputed and compared.
    """

    protocol = payload.get("protocol")
    if type(protocol) is not dict:
        return None
    seed = protocol.get("seed")
    backend_fingerprint = protocol.get("backend_fingerprint")
    if type(seed) is not int or isinstance(seed, bool) or type(backend_fingerprint) is not str:
        return None
    if not _recomputed_equal(protocol, _instructiondistance_protocol(seed, backend_fingerprint)):
        return None
    candidates = payload.get("candidates")
    if type(candidates) is not dict or not candidates:
        return None
    generated = _instructiondistance_generate(seed)
    total_records = 0
    records_bytes = 0
    aggregates: dict[str, Any] = {}
    for name, entry in candidates.items():
        if type(entry) is not dict:
            return None
        status = entry.get("status")
        if status == "unavailable":
            if set(entry) != {"status", "detail", "records"} or entry.get("records") != []:
                return None
            continue
        if status != "available" or entry.get("detail") != "":
            return None
        if set(entry) != _INSTRUCTIONDISTANCE_AGGREGATE_KEYS | {"status", "detail", "records"}:
            return None
        records = entry["records"]
        if type(records) is not list or len(records) != len(generated):
            return None
        total_records += len(records)
        records_bytes += len(
            json.dumps(records, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        )
        if records_bytes > _INSTRUCTIONDISTANCE_MAX_RECORDS_BYTES:
            return None
        for record, case in zip(records, generated, strict=True):
            if type(record) is not dict or set(record) != {
                "fixture",
                "topic",
                "expected_digest",
                "arm_tokens",
                "id_digests",
                "cells",
                "cell_details",
            }:
                return None
            if record["fixture"] != case["fixture_id"] or record["topic"] != case["topic"]:
                return None
            expected_digest = _instructiondistance_digest(str(case["value"]))
            if record["expected_digest"] != expected_digest:
                return None
            arm_tokens = record["arm_tokens"]
            id_digests = record["id_digests"]
            cells = record["cells"]
            details = record["cell_details"]
            for mapping in (arm_tokens, id_digests, cells, details):
                if type(mapping) is not dict or set(mapping) != set(_INSTRUCTIONDISTANCE_ARM_KEYS):
                    return None
            adj_tokens = arm_tokens["adj"]
            if type(adj_tokens) is not int or adj_tokens < 0:
                return None
            if type(arm_tokens["adj_after"]) is not int or arm_tokens["adj_after"] < 0:
                return None
            for arm in _INSTRUCTIONDISTANCE_ARM_KEYS:
                tokens = arm_tokens[arm]
                id_digest = id_digests[arm]
                if tokens is None:
                    # Unconstructed arm: no render exists to
                    # carry a token count or an id digest.
                    if id_digest is not None:
                        return None
                else:
                    if type(tokens) is not int or tokens < 0:
                        return None
                    if not _instructiondistance_digest_ok(id_digest):
                        return None
                    if arm in _INSTRUCTIONDISTANCE_SIZED_ARMS:
                        dose = 256 if arm.endswith("256") else 1024
                        if tokens - adj_tokens != dose:
                            return None
                cell = cells[arm]
                detail = details[arm]
                if cell is None:
                    allowed = (
                        _INSTRUCTIONDISTANCE_DETAILS
                        if arm in _INSTRUCTIONDISTANCE_SIZED_ARMS
                        else _INSTRUCTIONDISTANCE_UNSIZED_DETAILS
                    )
                    if detail not in allowed:
                        return None
                    if tokens is None and detail not in (
                        "arm_unmatchable",
                        "arm_context_exceeded",
                    ):
                        return None
                else:
                    if detail != "":
                        return None
                    if not _instructiondistance_cell_ok(cell, expected_digest):
                        return None
                    if cell["input_tokens"] != tokens:
                        return None
        aggregates[name] = _instructiondistance_aggregate(records)
    if total_records > _INSTRUCTIONDISTANCE_MAX_GROUND_RECORDS:
        return None
    if not aggregates:
        return None
    return {
        "aggregates": aggregates,
        "n_candidates": len(candidates),
        "n_cases": len(generated),
    }


def _valid_instruction_distance(payload: Any) -> bool:
    """Structural check for an ``instruction_distance``
    payload (recompute standard). The audit block is
    shape-checked only: its counts are stage observations
    (special-token findings against the run's own formats),
    not recomputable from records.
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
    audit = payload.get("audit")
    if type(audit) is not dict or set(audit) != {
        "n_audit_excluded",
        "by_reason",
        "special_token_findings",
    }:
        return False
    if type(audit["n_audit_excluded"]) is not int or audit["n_audit_excluded"] < 0:
        return False
    if type(audit["special_token_findings"]) is not int or audit["special_token_findings"] < 0:
        return False
    by_reason = audit["by_reason"]
    if type(by_reason) is not dict:
        return False
    for key, count in by_reason.items():
        if type(key) is not str or type(count) is not int or count < 0:
            return False
    recomputed = _instructiondistance_recompute(payload)
    if recomputed is None:
        return False
    candidates = payload["candidates"]
    for name, expected in recomputed["aggregates"].items():
        entry = candidates[name]
        for key, value in expected.items():
            if not _recomputed_equal(entry.get(key), value):
                return False
    if not _recomputed_equal(payload.get("n_candidates"), recomputed["n_candidates"]):
        return False
    return _recomputed_equal(payload.get("n_cases"), recomputed["n_cases"])


def diff_instruction_distance(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``instruction_distance`` probe payloads;
    deterministic.

    Answers "did the instruction-distance story change between
    runs?": per-candidate per-dose decision drift with the
    (D, S, T) counts that carry it, the same-ids compliance
    flip drift on the largest separated arm, candidates
    added/removed, and protocol / generator / backend drift
    (any of which marks the comparison not comparable, with
    the drift still listed -- the generated set is the
    treatment, so a changed generator is protocol drift). A
    side without a valid payload is reported plainly.
    """
    old_ok = _valid_instruction_distance(old)
    new_ok = _valid_instruction_distance(new)
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
    id_protocol_drift = old["protocol"].get("fingerprint") != new["protocol"].get("fingerprint")
    id_generator_drift = old["protocol"].get("ground_sha256") != new["protocol"].get(
        "ground_sha256"
    )
    id_backend_drift = old["protocol"].get("backend_fingerprint") != new["protocol"].get(
        "backend_fingerprint"
    )
    id_comparable = not id_protocol_drift and not id_generator_drift and not id_backend_drift
    old_candidates = old["candidates"]
    new_candidates = new["candidates"]
    id_candidates_added = sorted(set(new_candidates) - set(old_candidates))
    id_candidates_removed = sorted(set(old_candidates) - set(new_candidates))
    id_candidate_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_candidates) & set(new_candidates)):
        old_entry = old_candidates[name]
        new_entry = new_candidates[name]
        if old_entry.get("status") != "available" or new_entry.get("status") != "available":
            if old_entry.get("status") != new_entry.get("status"):
                id_candidate_drift.append(
                    {
                        "candidate": name,
                        "metric": "status",
                        "old": old_entry.get("status"),
                        "new": new_entry.get("status"),
                    }
                )
            continue
        id_metrics: list[tuple[str, Any, Any]] = []
        for dose_key in ("256", "1024"):
            old_dose = old_entry["doses"][dose_key]
            new_dose = new_entry["doses"][dose_key]
            id_metrics.extend(
                [
                    (f"decision_{dose_key}", old_dose["decision"], new_dose["decision"]),
                    (
                        f"direct_passes_{dose_key}",
                        old_dose["direct_passes"],
                        new_dose["direct_passes"],
                    ),
                    (
                        f"separated_losses_{dose_key}",
                        old_dose["separated_losses"],
                        new_dose["separated_losses"],
                    ),
                    (
                        f"displaced_losses_{dose_key}",
                        old_dose["displaced_losses"],
                        new_dose["displaced_losses"],
                    ),
                ]
            )
        id_metrics.append(
            (
                "n_same_ids_compliance_flips_sep_1024",
                old_entry["ids"]["sep_1024"]["n_same_ids_compliance_flips"],
                new_entry["ids"]["sep_1024"]["n_same_ids_compliance_flips"],
            )
        )
        for metric, old_value, new_value in id_metrics:
            if old_value != new_value:
                id_candidate_drift.append(
                    {"candidate": name, "metric": metric, "old": old_value, "new": new_value}
                )
    result["candidate_drift"] = id_candidate_drift
    result["candidates_added"] = id_candidates_added
    result["candidates_removed"] = id_candidates_removed
    result["protocol_drift"] = id_protocol_drift
    result["generator_drift"] = id_generator_drift
    result["backend_drift"] = id_backend_drift
    result["comparable"] = id_comparable
    result["changed"] = bool(
        id_candidate_drift
        or id_candidates_added
        or id_candidates_removed
        or id_protocol_drift
        or id_generator_drift
        or id_backend_drift
    )
    return result


_HISTORYFRAME_ARM_KEYS = ("own", "foreign_chatml", "foreign_llama3", "verbatim_chatml")


_HISTORYFRAME_HEADLINE_ARMS = ("own", "foreign_chatml", "foreign_llama3")


_HISTORYFRAME_CONTRASTS = ("foreign_chatml", "foreign_llama3")


_HISTORYFRAME_DONOR_FAMILIES = {"foreign_chatml": "chatml", "foreign_llama3": "llama3"}


_HISTORYFRAME_VERBATIM_ARM = "verbatim_chatml"


_HISTORYFRAME_VERBATIM_TWIN = "foreign_chatml"


_HISTORYFRAME_HISTORY_CONDITION = "history_6"


_HISTORYFRAME_HISTORY_TURNS = 6


_HISTORYFRAME_TURNS = (
    ("user", "Hello. I have a few short questions for you."),
    ("assistant", "Hello! I am ready. Please ask your questions one at a time."),
    ("user", "Thank you. I will keep each question short."),
    ("assistant", "Understood. I will keep my answers short as well."),
    ("user", "Let us begin whenever you are ready."),
    ("assistant", "I am ready. Please go ahead with your first question."),
)


_HISTORYFRAME_MIN_N = 10


_HISTORYFRAME_MIN_D = 6


_HISTORYFRAME_MAX_CANDIDATES = 24


_HISTORYFRAME_MAX_SEARCH_FIXTURES = 100


_HISTORYFRAME_MAX_GROUND_RECORDS = _HISTORYFRAME_MAX_CANDIDATES * _HISTORYFRAME_MAX_SEARCH_FIXTURES


_HISTORYFRAME_GENERATION_BUDGET = len(_HISTORYFRAME_ARM_KEYS) * _HISTORYFRAME_MAX_GROUND_RECORDS


_HISTORYFRAME_MAX_RECORDS_BYTES = 12 * 1024 * 1024


_HISTORYFRAME_CELL_FIELDS = {
    "verdict",
    "boundary_found",
    "boundary_text_match",
    "input_tokens",
    "answer_digest",
}


_HISTORYFRAME_DETAILS = ("arm_context_exceeded", "arm_evaluation_failed")


_HISTORYFRAME_STRATA = ("all_single", "all_fragmented", "mixed")


_HISTORYFRAME_DECOMPOSITION_CLASSES = ("boundary_only", "exact_only", "mixed")


_HISTORYFRAME_AGGREGATE_KEYS = {
    "arms",
    "contrasts",
    "verbatim",
    "operating_characteristics",
    "jitter_grid",
    "n_fixtures",
}


_HISTORYFRAME_OPERATING_CHARACTERISTICS: dict[str, Any] = {
    "random_labeller_fire_bound": 0.036,
    "competent_fire_bound": 0.031,
    "power_25_point_at_30": 0.52,
    "power_25_point_at_100": 0.988,
    "power_50_point_at_30": 0.983,
    "power_50_point_at_100": 1.0,
}


_HISTORYFRAME_JITTER_GRID = (
    {"epsilon": 0.01, "fire_rate": 0.0004},
    {"epsilon": 0.02, "fire_rate": 0.0048},
    {"epsilon": 0.05, "fire_rate": 0.0213},
)


_HISTORYFRAME_SEGMENT_RULE_TEXT = (
    "historyframe-segment-v1: seg(F, m, i) = (F.bos if i > 0 "
    "and F.repeat_bos else '') + (F.user_prefix if m.role is "
    "user else F.assistant_prefix) + m.content + (per-role end "
    "of F for m.role, the per-role field when it is not None "
    "else F.turn_end); open(B) = (B.bos if B.repeat_bos else "
    "'') + (B.open_suffix if B.open_suffix is not None else "
    "B.assistant_prefix); hybrid(A, B) = B.bos + seg(A) over "
    "the first 6 messages + seg(B) over the rest + open(B); "
    "verbatim(A, B) = A.bos + hybrid(A, B) with only the "
    "leading B.bos removed. Screens: S1 per-arm slice-swap "
    "identity pre-generation (each hybrid equals the native "
    "B render with only the history-segment slice exchanged; "
    "the verbatim arm exchanges the same slice under its "
    "defining leading-bos substitution; hybrid(B, B) equals "
    "the native render byte for byte); S2 "
    "special-token-literal audit over the frozen history texts "
    "and the donor and candidate framing literals; S3 budget "
    "caps. User fixture contents are never screened: they are "
    "byte-identical across arms, so they cannot manufacture a "
    "frame contrast."
)


def _historyframe_digest(text: str) -> str:
    """SHA-256 hex digest (local copy; no imports from
    historyframe.py anywhere in this mirror).
    """

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _historyframe_digest_ok(value: Any) -> bool:
    """Whether a stored digest is a well-formed SHA-256 hex
    digest.
    """
    if type(value) is not str or len(value) != 64:
        return False
    return all(ch in "0123456789abcdef" for ch in value)


def _historyframe_mcnemar_p(a: int, b: int) -> float | None:
    """Mirror of confidentiality.mcnemar_exact_p: two-sided
    exact McNemar p; None when there are no discordant pairs.
    """

    n = a + b
    if n == 0:
        return None
    k = min(a, b)
    tail: float = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    result: float = min(1.0, 2.0 * tail)
    return result


def _historyframe_classify(n: int, own_only: int, foreign_only: int) -> str:
    """Mirror of historyframe.classify_history_frame."""
    if n < _HISTORYFRAME_MIN_N:
        return "insufficient"
    d = own_only + foreign_only
    if d == 0:
        return "no_discordance_observed"
    if d < _HISTORYFRAME_MIN_D:
        return "below_threshold"
    if own_only == foreign_only:
        return "discordant_balanced"
    p = _historyframe_mcnemar_p(own_only, foreign_only)
    if p is not None and p < 0.05:
        return "discordant_asymmetric"
    return "below_threshold"


def _historyframe_random_fire_rate(n: int) -> float:
    """Mirror of historyframe.random_labeller_fire_rate
    (exact enumeration over the discordant count and its
    split under independent fair arms).
    """

    if type(n) is not int or n < _HISTORYFRAME_MIN_N:
        return 0.0
    total = 0.0
    for d in range(_HISTORYFRAME_MIN_D, n + 1):
        p_d = math.comb(n, d) * (0.5**n)
        for a in range(d + 1):
            b = d - a
            if a == b:
                continue
            p = _historyframe_mcnemar_p(a, b)
            if p is not None and p < 0.05:
                total += p_d * math.comb(d, a) * (0.5**d)
    return total


def _historyframe_familywise(rates: list[float]) -> dict[str, Any]:
    """Mirror of historyframe.familywise_expectation."""
    expected = sum(rates)
    survival = 1.0
    for rate in rates:
        survival *= 1.0 - rate
    return {
        "decisions": len(rates),
        "expected_false_labels": expected,
        "p_at_least_one": 1.0 - survival,
    }


def _historyframe_history_material_sha256() -> str:
    """Mirror of historyframe's history-material digest over
    the frozen transcript.
    """

    canonical = json.dumps(
        [[role, content] for role, content in _HISTORYFRAME_TURNS],
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return _historyframe_digest(canonical)


def _historyframe_protocol(
    seed: int, backend_fingerprint: str, fixture_ids: list[str]
) -> dict[str, Any]:
    """Mirror of historyframe.protocol_object, fingerprint
    included (compared whole).
    """

    protocol: dict[str, Any] = {
        "version": 1,
        "arm_keys": list(_HISTORYFRAME_ARM_KEYS),
        "headline_arms": list(_HISTORYFRAME_HEADLINE_ARMS),
        "contrasts": list(_HISTORYFRAME_CONTRASTS),
        "donor_families": dict(_HISTORYFRAME_DONOR_FAMILIES),
        "verbatim_arm": _HISTORYFRAME_VERBATIM_ARM,
        "verbatim_twin": _HISTORYFRAME_VERBATIM_TWIN,
        "history_condition": _HISTORYFRAME_HISTORY_CONDITION,
        "history_turns": _HISTORYFRAME_HISTORY_TURNS,
        "history_material_sha256": _historyframe_history_material_sha256(),
        "segment_rule_sha256": _historyframe_digest(_HISTORYFRAME_SEGMENT_RULE_TEXT),
        "min_n": _HISTORYFRAME_MIN_N,
        "min_d": _HISTORYFRAME_MIN_D,
        "generation_budget": _HISTORYFRAME_GENERATION_BUDGET,
        "max_ground_records": _HISTORYFRAME_MAX_GROUND_RECORDS,
        "max_records_bytes": _HISTORYFRAME_MAX_RECORDS_BYTES,
        "max_search_fixtures": _HISTORYFRAME_MAX_SEARCH_FIXTURES,
        "max_candidates": _HISTORYFRAME_MAX_CANDIDATES,
        "seed": seed,
        "backend_fingerprint": backend_fingerprint,
        "fixture_ids": list(fixture_ids),
    }
    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    protocol["fingerprint"] = _historyframe_digest(canonical)
    return protocol


def _historyframe_cell_get(record: dict[str, Any], key: str) -> dict[str, Any] | None:
    """Return the historyframe cell get."""
    cells = record.get("cells")
    if not isinstance(cells, dict):
        return None
    cell = cells.get(key)
    return cell if isinstance(cell, dict) else None


def _historyframe_decompose(own_cell: dict[str, Any], foreign_cell: dict[str, Any]) -> str:
    """Mirror of historyframe.decompose_pair."""
    if own_cell.get("answer_digest") == foreign_cell.get("answer_digest"):
        return "boundary_only"
    if own_cell.get("boundary_found") and foreign_cell.get("boundary_found"):
        return "exact_only"
    return "mixed"


def _historyframe_contrast(records: list[dict[str, Any]], contrast: str) -> dict[str, Any]:
    """Mirror of historyframe._contrast_aggregate. The
    ids-identical gate is derived from the recorded id
    digests (amendment A4), render-divergence from the
    recorded segment digests, and ``id_length_delta`` is
    recomputed from the recorded arm token counts -- a
    carried flag or delta is never trusted.
    """
    n_render_identical = 0
    n_ids_identical = 0
    n_same_ids_flips = 0
    n_paired = 0
    own_only = 0
    foreign_only = 0
    own_passes = 0
    foreign_passes = 0
    ungated_n = 0
    ungated_own_only = 0
    ungated_foreign_only = 0
    decomposition: dict[str, int] = dict.fromkeys(_HISTORYFRAME_DECOMPOSITION_CLASSES, 0)
    deltas: list[int] = []
    for record in records:
        segments = record.get("segment_digests")
        segments = segments if isinstance(segments, dict) else {}
        render_divergent = segments.get(contrast) != segments.get("own")
        arm_tokens = record.get("arm_tokens") or {}
        own_tokens = arm_tokens.get("own")
        foreign_tokens = arm_tokens.get(contrast)
        if type(own_tokens) is int and type(foreign_tokens) is int:
            deltas.append(own_tokens - foreign_tokens)
        own_cell = _historyframe_cell_get(record, "own")
        foreign_cell = _historyframe_cell_get(record, contrast)
        if own_cell is not None and foreign_cell is not None:
            ungated_n += 1
            if own_cell["verdict"] and not foreign_cell["verdict"]:
                ungated_own_only += 1
            elif foreign_cell["verdict"] and not own_cell["verdict"]:
                ungated_foreign_only += 1
        if not render_divergent:
            n_render_identical += 1
            continue
        id_digests = record.get("id_digests")
        id_digests = id_digests if isinstance(id_digests, dict) else {}
        own_digest = id_digests.get("own")
        foreign_digest = id_digests.get(contrast)
        ids_identical = (
            type(own_digest) is str and own_digest != "" and own_digest == foreign_digest
        )
        if ids_identical:
            n_ids_identical += 1
            if (
                own_cell is not None
                and foreign_cell is not None
                and own_cell["verdict"] != foreign_cell["verdict"]
            ):
                n_same_ids_flips += 1
            continue
        if own_cell is None or foreign_cell is None:
            continue
        n_paired += 1
        if own_cell["verdict"]:
            own_passes += 1
        if foreign_cell["verdict"]:
            foreign_passes += 1
        if own_cell["verdict"] and not foreign_cell["verdict"]:
            own_only += 1
            cls = _historyframe_decompose(own_cell, foreign_cell)
            decomposition[cls] = decomposition.get(cls, 0) + 1
        elif foreign_cell["verdict"] and not own_cell["verdict"]:
            foreign_only += 1
            cls = _historyframe_decompose(own_cell, foreign_cell)
            decomposition[cls] = decomposition.get(cls, 0) + 1
    applicable = n_render_identical < len(records)
    d = own_only + foreign_only
    classification = _historyframe_classify(n_paired, own_only, foreign_only)
    return {
        "applicable": applicable,
        "n_render_identical": n_render_identical,
        "n_ids_identical": n_ids_identical,
        "n_same_ids_flips": n_same_ids_flips,
        "n_paired": n_paired,
        "own_only": own_only,
        "foreign_only": foreign_only,
        "discordant": d,
        "own_passes_paired": own_passes,
        "foreign_passes_paired": foreign_passes,
        "paired_rate_own": (own_passes / n_paired) if n_paired else None,
        "paired_rate_foreign": (foreign_passes / n_paired) if n_paired else None,
        "ungated_n": ungated_n,
        "ungated_own_only": ungated_own_only,
        "ungated_foreign_only": ungated_foreign_only,
        "decomposition": decomposition,
        "classification": classification if applicable else None,
        "falsification": (classification == "discordant_asymmetric") if applicable else False,
        "mcnemar_p": _historyframe_mcnemar_p(own_only, foreign_only) if d else None,
        "random_fire_rate": _historyframe_random_fire_rate(n_paired) if applicable else None,
        "id_length_delta": deltas,
        "max_abs_id_delta": max((abs(x) for x in deltas), default=0),
    }


def _historyframe_verbatim(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Mirror of historyframe._verbatim_aggregate.
    Applicability is derived from the recorded bos digests
    (amendment A1): equal digests mean the arm is
    byte-identical to its twin and the readout is
    inapplicable.
    """
    applicable = all(
        type(record.get("own_bos_digest")) is str
        and type(record.get("donor_bos_digest")) is str
        and record.get("own_bos_digest") != record.get("donor_bos_digest")
        for record in records
    )
    n_paired = 0
    twin_only = 0
    verbatim_only = 0
    if applicable:
        for record in records:
            twin = _historyframe_cell_get(record, _HISTORYFRAME_VERBATIM_TWIN)
            verbatim = _historyframe_cell_get(record, _HISTORYFRAME_VERBATIM_ARM)
            if twin is None or verbatim is None:
                continue
            n_paired += 1
            if twin["verdict"] and not verbatim["verdict"]:
                twin_only += 1
            elif verbatim["verdict"] and not twin["verdict"]:
                verbatim_only += 1
    return {
        "applicable": applicable,
        "n_paired": n_paired,
        "twin_only": twin_only,
        "verbatim_only": verbatim_only,
        "discordant": twin_only + verbatim_only,
    }


def _historyframe_aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Mirror of historyframe.aggregate_candidate."""
    arms: dict[str, Any] = {}
    for key in _HISTORYFRAME_ARM_KEYS:
        n_completed = 0
        n_pass = 0
        n_context = 0
        n_failed = 0
        for record in records:
            details = record.get("cell_details")
            detail = details.get(key, "") if isinstance(details, dict) else ""
            if detail == "arm_context_exceeded":
                n_context += 1
            elif detail == "arm_evaluation_failed":
                n_failed += 1
            cell = _historyframe_cell_get(record, key)
            if cell is None:
                continue
            n_completed += 1
            if cell["verdict"]:
                n_pass += 1
        arms[key] = {
            "n_completed": n_completed,
            "n_pass": n_pass,
            "rate": (n_pass / n_completed) if n_completed else None,
            "n_context_exceeded": n_context,
            "n_evaluation_failed": n_failed,
        }
    return {
        "arms": arms,
        "contrasts": {
            contrast: _historyframe_contrast(records, contrast)
            for contrast in _HISTORYFRAME_CONTRASTS
        },
        "verbatim": _historyframe_verbatim(records),
        "operating_characteristics": dict(_HISTORYFRAME_OPERATING_CHARACTERISTICS),
        "jitter_grid": [dict(row) for row in _HISTORYFRAME_JITTER_GRID],
        "n_fixtures": len(records),
    }


def _historyframe_shared(
    records_by_candidate: dict[str, list[dict[str, Any]]],
) -> frozenset[str]:
    """Mirror of historyframe.shared_fixtures."""
    populations: list[frozenset[str]] = []
    for records in records_by_candidate.values():
        populations.append(frozenset(str(r.get("fixture")) for r in records))
    if not populations:
        return frozenset()
    return frozenset.intersection(*populations)


def _historyframe_verdict(record: dict[str, Any], key: str) -> bool:
    """Mirror of historyframe._verdict."""
    cell = _historyframe_cell_get(record, key)
    return bool(cell is not None and cell["verdict"])


def _historyframe_selection_over(
    records_by_candidate: dict[str, list[dict[str, Any]]],
    key: str,
    fixtures: frozenset[str],
) -> dict[str, Any] | None:
    """Mirror of historyframe._selection_over."""
    completed_sets: list[frozenset[str]] = []
    all_pass: dict[str, bool | None] = {}
    for name, records in records_by_candidate.items():
        completed = frozenset(
            str(record["fixture"])
            for record in records
            if record["fixture"] in fixtures and _historyframe_cell_get(record, key) is not None
        )
        completed_sets.append(completed)
        if not completed:
            return None
        all_pass[name] = all(
            _historyframe_verdict(record, key)
            for record in records
            if record["fixture"] in completed
        )
    if len(set(completed_sets)) > 1:
        return None
    passers = [name for name, ok in all_pass.items() if ok]
    if len(passers) == 1:
        return {"decision": "selected", "winner": passers[0]}
    if len(passers) > 1:
        return {"decision": "ambiguous", "winner": None}
    return {"decision": "no_candidate", "winner": None}


def _historyframe_arm_selection(
    records_by_candidate: dict[str, list[dict[str, Any]]],
    arm: str,
) -> dict[str, Any]:
    """Mirror of historyframe.arm_selection."""
    shared = _historyframe_shared(records_by_candidate)
    own = _historyframe_selection_over(records_by_candidate, arm, shared) if shared else None
    if own is None:
        return {
            "decision": "unavailable",
            "detail": "selection_withheld_mismatched_sets",
            "winner": None,
            "withheld": True,
        }
    return {"decision": own["decision"], "detail": "", "winner": own["winner"], "withheld": False}


def _historyframe_winner_headline(baseline: Any, other: Any) -> dict[str, Any]:
    """Mirror of historyframe.winner_headline."""
    if not isinstance(baseline, dict) or not isinstance(other, dict):
        return {
            "headline": "winner_undefined",
            "detail": "selection_missing",
            "winner_changed": None,
        }
    if baseline.get("withheld") is True or other.get("withheld") is True:
        return {
            "headline": "winner_undefined",
            "detail": "selection_withheld_mismatched_sets",
            "winner_changed": None,
        }
    baseline_selected = baseline.get("decision") == "selected"
    other_selected = other.get("decision") == "selected"
    if baseline_selected and other_selected:
        if baseline.get("winner") != other.get("winner"):
            return {"headline": "winner_changed", "detail": "", "winner_changed": True}
        return {"headline": "winner_stable", "detail": "", "winner_changed": False}
    if baseline_selected != other_selected:
        return {"headline": "winner_abstention_asymmetric", "detail": "", "winner_changed": False}
    return {"headline": "winner_undefined", "detail": "", "winner_changed": False}


def _historyframe_primary_flips(conditions: dict[str, Any], primary: Any) -> list[dict[str, Any]]:
    """Mirror of historyframe.primary_flips."""
    if not isinstance(primary, dict):
        return []
    decision = primary.get("decision")
    winner = primary.get("winner")
    if decision in ("compatible_on_fixture_set", "held_out_failed"):
        verdict = ("selected", winner)
    elif type(decision) is str and decision:
        verdict = (decision, winner)
    else:
        return []
    flips: list[dict[str, Any]] = []
    for arm in _HISTORYFRAME_HEADLINE_ARMS:
        selection = conditions.get(arm) if isinstance(conditions, dict) else None
        if not isinstance(selection, dict) or selection.get("withheld") is True:
            continue
        if (selection.get("decision"), selection.get("winner")) != verdict:
            flips.append(
                {
                    "arm": arm,
                    "decision_from": verdict[0],
                    "decision_to": selection.get("decision"),
                    "winner_from": verdict[1],
                    "winner_to": selection.get("winner"),
                }
            )
    return flips


def _historyframe_cell_ok(cell: Any, expected_digest: str) -> bool:
    """One cell's shape and verdict binding: ``verdict ==
    (answer_digest == expected_digest) and boundary_found``,
    with the answer digest either a well-formed digest or
    None (no answer string was decoded).
    """
    if type(cell) is not dict or set(cell) != _HISTORYFRAME_CELL_FIELDS:
        return False
    if type(cell["verdict"]) is not bool or type(cell["boundary_found"]) is not bool:
        return False
    if cell["boundary_text_match"] is not None and type(cell["boundary_text_match"]) is not bool:
        return False
    if type(cell["input_tokens"]) is not int or cell["input_tokens"] < 0:
        return False
    if cell["answer_digest"] is not None and not _historyframe_digest_ok(cell["answer_digest"]):
        return False
    verdict = (cell["answer_digest"] == expected_digest) and cell["boundary_found"]
    return bool(cell["verdict"] == verdict)


def _historyframe_recompute(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Recompute a history-frame payload from its ground
    records.

    The validator carries its own frozen constants (donor
    pair, transcript, floors, segment-rule text) and rebuilds
    the protocol from them plus the payload's seed, backend
    fingerprint, and fixture ids, comparing it whole (no
    imports from historyframe.py). The fixture set itself is
    user data and cannot be regenerated, so the mirror binds
    everything the records make decidable: per record, the
    fixture id against the protocol's set, the bos-digest
    pair constant across the candidate (amendment A1's
    applicability is derived, never carried), digest shapes
    for every arm's segment/hybrid/id digests, arm token
    counts bound to each completed cell's ``input_tokens``,
    the verdict binding per cell, and the budget caps; then
    every aggregate, the per-arm selections, the winner
    headlines, the primary flips, and the family-wise block
    are recomputed from the records and compared.
    """

    protocol = payload.get("protocol")
    if type(protocol) is not dict:
        return None
    seed = protocol.get("seed")
    backend_fingerprint = protocol.get("backend_fingerprint")
    fixture_ids = protocol.get("fixture_ids")
    if type(seed) is not int or isinstance(seed, bool):
        return None
    if type(backend_fingerprint) is not str:
        return None
    if (
        type(fixture_ids) is not list
        or not fixture_ids
        or len(fixture_ids) > _HISTORYFRAME_MAX_SEARCH_FIXTURES
        or any(type(fid) is not str for fid in fixture_ids)
        or len(set(fixture_ids)) != len(fixture_ids)
    ):
        return None
    if not _recomputed_equal(
        protocol, _historyframe_protocol(seed, backend_fingerprint, fixture_ids)
    ):
        return None
    candidates = payload.get("candidates")
    if type(candidates) is not dict or not candidates:
        return None
    if len(candidates) > _HISTORYFRAME_MAX_CANDIDATES:
        return None
    total_records = 0
    records_bytes = 0
    aggregates: dict[str, Any] = {}
    records_by_candidate: dict[str, list[dict[str, Any]]] = {}
    for name, entry in candidates.items():
        if type(entry) is not dict:
            return None
        status = entry.get("status")
        if status == "unavailable":
            if set(entry) != {"status", "detail", "records"} or entry.get("records") != []:
                return None
            continue
        if status != "available" or entry.get("detail") != "":
            return None
        if set(entry) != _HISTORYFRAME_AGGREGATE_KEYS | {"status", "detail", "records"}:
            return None
        records = entry["records"]
        if type(records) is not list or len(records) != len(fixture_ids):
            return None
        total_records += len(records)
        records_bytes += len(
            json.dumps(records, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        )
        if records_bytes > _HISTORYFRAME_MAX_RECORDS_BYTES:
            return None
        bos_pair: tuple[Any, Any] | None = None
        for record, fixture_id in zip(records, fixture_ids, strict=True):
            if type(record) is not dict or set(record) != {
                "fixture",
                "expected_digest",
                "own_bos_digest",
                "donor_bos_digest",
                "segment_digests",
                "hybrid_digests",
                "arm_tokens",
                "id_digests",
                "cells",
                "cell_details",
            }:
                return None
            if record["fixture"] != fixture_id:
                return None
            if not _historyframe_digest_ok(record["expected_digest"]):
                return None
            if not _historyframe_digest_ok(record["own_bos_digest"]):
                return None
            if not _historyframe_digest_ok(record["donor_bos_digest"]):
                return None
            pair = (record["own_bos_digest"], record["donor_bos_digest"])
            if bos_pair is None:
                bos_pair = pair
            elif pair != bos_pair:
                # The bos pair is a per-candidate constant (the
                # candidate's bos and the frozen donor's); a
                # record disagreeing with its siblings is a
                # forgery or corruption, never data.
                return None
            for mapping_key in ("segment_digests", "hybrid_digests", "id_digests"):
                mapping = record[mapping_key]
                if type(mapping) is not dict or set(mapping) != set(_HISTORYFRAME_ARM_KEYS):
                    return None
                if any(not _historyframe_digest_ok(value) for value in mapping.values()):
                    return None
            arm_tokens = record["arm_tokens"]
            if type(arm_tokens) is not dict or set(arm_tokens) != set(_HISTORYFRAME_ARM_KEYS):
                return None
            if any(type(value) is not int or value < 0 for value in arm_tokens.values()):
                return None
            cells = record["cells"]
            details = record["cell_details"]
            for mapping in (cells, details):
                if type(mapping) is not dict or set(mapping) != set(_HISTORYFRAME_ARM_KEYS):
                    return None
            for arm in _HISTORYFRAME_ARM_KEYS:
                cell = cells[arm]
                detail = details[arm]
                if cell is None:
                    if detail not in _HISTORYFRAME_DETAILS:
                        return None
                else:
                    if detail != "":
                        return None
                    if not _historyframe_cell_ok(cell, record["expected_digest"]):
                        return None
                    if cell["input_tokens"] != arm_tokens[arm]:
                        return None
        aggregates[name] = _historyframe_aggregate(records)
        records_by_candidate[name] = records
    if total_records > _HISTORYFRAME_MAX_GROUND_RECORDS:
        return None
    if not aggregates:
        return None
    conditions = {
        arm: _historyframe_arm_selection(records_by_candidate, arm)
        for arm in _HISTORYFRAME_HEADLINE_ARMS
    }
    selection = {
        contrast: _historyframe_winner_headline(conditions["own"], conditions[contrast])
        for contrast in _HISTORYFRAME_CONTRASTS
    }
    rates: list[float] = []
    for entry_agg in aggregates.values():
        for contrast in _HISTORYFRAME_CONTRASTS:
            block = entry_agg["contrasts"][contrast]
            if block["random_fire_rate"] is not None:
                rates.append(float(block["random_fire_rate"]))
    return {
        "aggregates": aggregates,
        "conditions": conditions,
        "selection": selection,
        "flips": _historyframe_primary_flips(conditions, payload.get("primary")),
        "familywise": _historyframe_familywise(rates),
        "n_candidates": len(candidates),
        "n_fixtures": len(fixture_ids),
    }


def _historyframe_stratum_ok(stratum: Any) -> bool:
    """Shape check for the run-level stratum block: its
    per-literal token-real values are backend observations,
    not recomputable from records, so -- like the audit
    block -- they are shape-checked only.
    """
    if type(stratum) is not dict or set(stratum) != set(_HISTORYFRAME_CONTRASTS):
        return False
    for arm in _HISTORYFRAME_CONTRASTS:
        block = stratum[arm]
        if type(block) is not dict or set(block) != {"label", "literals"}:
            return False
        if block["label"] not in _HISTORYFRAME_STRATA:
            return False
        literals = block["literals"]
        if type(literals) is not dict:
            return False
        for literal, real in literals.items():
            if type(literal) is not str or not literal or type(real) is not bool:
                return False
        values = list(literals.values())
        if values and all(values):
            expected = "all_single"
        elif values and not any(values):
            expected = "all_fragmented"
        elif values:
            expected = "mixed"
        else:
            expected = "all_single"
        if block["label"] != expected:
            return False
    return True


def _valid_history_frame(payload: Any) -> bool:
    """Structural check for a ``history_frame`` payload
    (recompute standard). The audit block and the stratum
    literals are shape-checked only: their values are stage
    observations against the run's own formats and backend,
    not recomputable from records.
    """
    if type(payload) is not dict or payload.get("available") is not True:
        return False
    if set(payload) != {
        "available",
        "detail",
        "protocol",
        "audit",
        "stratum",
        "candidates",
        "conditions",
        "selection",
        "primary",
        "flips",
        "familywise",
        "n_candidates",
        "n_fixtures",
    }:
        return False
    if payload.get("detail") != "":
        return False
    audit = payload.get("audit")
    if type(audit) is not dict or set(audit) != {
        "n_audit_excluded",
        "by_reason",
        "special_token_findings",
    }:
        return False
    if type(audit["n_audit_excluded"]) is not int or audit["n_audit_excluded"] < 0:
        return False
    if type(audit["special_token_findings"]) is not int or audit["special_token_findings"] < 0:
        return False
    by_reason = audit["by_reason"]
    if type(by_reason) is not dict:
        return False
    for key, count in by_reason.items():
        if type(key) is not str or type(count) is not int or count < 0:
            return False
    if not _historyframe_stratum_ok(payload.get("stratum")):
        return False
    primary = payload.get("primary")
    if primary is not None:
        if type(primary) is not dict or set(primary) != {"decision", "winner"}:
            return False
        if primary["decision"] is not None and type(primary["decision"]) is not str:
            return False
        if primary["winner"] is not None and type(primary["winner"]) is not str:
            return False
    recomputed = _historyframe_recompute(payload)
    if recomputed is None:
        return False
    candidates = payload["candidates"]
    for name, expected in recomputed["aggregates"].items():
        entry = candidates[name]
        for key, value in expected.items():
            if not _recomputed_equal(entry.get(key), value):
                return False
    for key in ("conditions", "selection", "flips", "familywise", "n_candidates", "n_fixtures"):
        if not _recomputed_equal(payload.get(key), recomputed[key]):
            return False
    return True


def diff_history_frame(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``history_frame`` probe payloads;
    deterministic.

    Answers "did the history-frame story change between
    runs?": per-candidate per-contrast classification drift
    with the paired counts and decomposition that carry it,
    verbatim applicability drift, winner-headline drift per
    contrast, encoding-stratum drift per foreign family,
    candidates added/removed, and protocol / history-material
    / backend drift (any of which marks the comparison not
    comparable, with the drift still listed -- the frozen
    transcript and the fixture set are part of the
    treatment). A side without a valid payload is reported
    plainly.
    """
    old_ok = _valid_history_frame(old)
    new_ok = _valid_history_frame(new)
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
    hf_protocol_drift = old["protocol"].get("fingerprint") != new["protocol"].get("fingerprint")
    hf_generator_drift = old["protocol"].get("history_material_sha256") != new["protocol"].get(
        "history_material_sha256"
    ) or old["protocol"].get("fixture_ids") != new["protocol"].get("fixture_ids")
    hf_backend_drift = old["protocol"].get("backend_fingerprint") != new["protocol"].get(
        "backend_fingerprint"
    )
    hf_comparable = not hf_protocol_drift and not hf_generator_drift and not hf_backend_drift
    hf_stratum_drift: list[dict[str, Any]] = []
    for arm in _HISTORYFRAME_CONTRASTS:
        old_label = old["stratum"][arm]["label"]
        new_label = new["stratum"][arm]["label"]
        if old_label != new_label:
            hf_stratum_drift.append({"family": arm, "old": old_label, "new": new_label})
    old_candidates = old["candidates"]
    new_candidates = new["candidates"]
    hf_candidates_added = sorted(set(new_candidates) - set(old_candidates))
    hf_candidates_removed = sorted(set(old_candidates) - set(new_candidates))
    hf_candidate_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_candidates) & set(new_candidates)):
        old_entry = old_candidates[name]
        new_entry = new_candidates[name]
        if old_entry.get("status") != "available" or new_entry.get("status") != "available":
            if old_entry.get("status") != new_entry.get("status"):
                hf_candidate_drift.append(
                    {
                        "candidate": name,
                        "metric": "status",
                        "old": old_entry.get("status"),
                        "new": new_entry.get("status"),
                    }
                )
            continue
        hf_metrics: list[tuple[str, Any, Any]] = []
        for contrast in _HISTORYFRAME_CONTRASTS:
            old_block = old_entry["contrasts"][contrast]
            new_block = new_entry["contrasts"][contrast]
            hf_metrics.extend(
                [
                    (
                        f"classification_{contrast}",
                        old_block["classification"],
                        new_block["classification"],
                    ),
                    (f"n_paired_{contrast}", old_block["n_paired"], new_block["n_paired"]),
                    (f"own_only_{contrast}", old_block["own_only"], new_block["own_only"]),
                    (
                        f"foreign_only_{contrast}",
                        old_block["foreign_only"],
                        new_block["foreign_only"],
                    ),
                    (
                        f"boundary_only_{contrast}",
                        old_block["decomposition"]["boundary_only"],
                        new_block["decomposition"]["boundary_only"],
                    ),
                    (
                        f"exact_only_{contrast}",
                        old_block["decomposition"]["exact_only"],
                        new_block["decomposition"]["exact_only"],
                    ),
                    (
                        f"mixed_{contrast}",
                        old_block["decomposition"]["mixed"],
                        new_block["decomposition"]["mixed"],
                    ),
                ]
            )
        hf_metrics.append(
            (
                "verbatim_applicable",
                old_entry["verbatim"]["applicable"],
                new_entry["verbatim"]["applicable"],
            )
        )
        for metric, old_value, new_value in hf_metrics:
            if old_value != new_value:
                hf_candidate_drift.append(
                    {"candidate": name, "metric": metric, "old": old_value, "new": new_value}
                )
    hf_selection_drift: list[dict[str, Any]] = []
    for contrast in _HISTORYFRAME_CONTRASTS:
        old_headline = old["selection"][contrast]["headline"]
        new_headline = new["selection"][contrast]["headline"]
        if old_headline != new_headline:
            hf_selection_drift.append(
                {"contrast": contrast, "old": old_headline, "new": new_headline}
            )
    result["candidate_drift"] = hf_candidate_drift
    result["candidates_added"] = hf_candidates_added
    result["candidates_removed"] = hf_candidates_removed
    result["selection_drift"] = hf_selection_drift
    result["stratum_drift"] = hf_stratum_drift
    result["protocol_drift"] = hf_protocol_drift
    result["generator_drift"] = hf_generator_drift
    result["backend_drift"] = hf_backend_drift
    result["comparable"] = hf_comparable
    result["changed"] = bool(
        hf_candidate_drift
        or hf_candidates_added
        or hf_candidates_removed
        or hf_selection_drift
        or hf_stratum_drift
        or hf_protocol_drift
        or hf_generator_drift
        or hf_backend_drift
    )
    return result


_PRUNESWEEP_ARM_KEYS = ("dose0", "dose18", "dose36", "dose72", "tail72", "closed_book")


_PRUNESWEEP_HEADLINE_ARMS = ("dose0", "dose18", "dose36", "dose72")


_PRUNESWEEP_DESCRIPTIVE_ARM = "tail72"


_PRUNESWEEP_SCREEN_ARM = "closed_book"


_PRUNESWEEP_DOSES = (0, 18, 36, 72)


_PRUNESWEEP_CASE_COUNT = 24


_PRUNESWEEP_CONTEXT_SENTENCES = 16


_PRUNESWEEP_WORDS_PER_SENTENCE = 9


_PRUNESWEEP_ELIGIBLE_WORDS = _PRUNESWEEP_CONTEXT_SENTENCES * _PRUNESWEEP_WORDS_PER_SENTENCE


_PRUNESWEEP_FACT_CONTEXT_SLOTS = 8


_PRUNESWEEP_MIN_C = 12


_PRUNESWEEP_MIN_A = 5


_PRUNESWEEP_INSTRUCTION = "Read the passage and answer the question."


_PRUNESWEEP_FACT_TEMPLATE = "The {topic} code is {value}."


_PRUNESWEEP_QUESTION_TEMPLATE = "What is the {topic} code? Reply with only the code word."


_PRUNESWEEP_TOPICS = (
    "harbor light",
    "summit trail",
    "quarry gate",
    "tundra post",
    "granite arch",
    "willow ford",
    "copper mill",
    "saddle pass",
)


_PRUNESWEEP_VALUES = (
    "amber",
    "birch",
    "cobalt",
    "dune",
    "ember",
    "flint",
    "garnet",
    "hazel",
    "ivory",
    "jasper",
    "kelvin",
    "linden",
    "magnet",
    "nickel",
    "onyx",
    "pepper",
    "quartz",
    "rowan",
    "saffron",
    "timber",
    "umber",
    "violet",
    "walnut",
    "yonder",
)


_PRUNESWEEP_SUBJECTS = (
    "archive",
    "ledger",
    "cabinet",
    "chapel",
    "meadow",
    "workshop",
    "station",
    "garden",
    "library",
    "market",
    "office",
    "depot",
)


_PRUNESWEEP_OBJECTS = (
    "forms",
    "letters",
    "tools",
    "charts",
    "samples",
    "records",
    "parcels",
    "volumes",
    "tickets",
    "bundles",
    "reports",
    "crates",
)


_PRUNESWEEP_PLACES = (
    "annex",
    "attic",
    "cellar",
    "landing",
    "terrace",
    "alcove",
    "gallery",
    "storeroom",
    "corridor",
    "balcony",
    "vestibule",
    "loft",
)


_PRUNESWEEP_TIMES = (
    "morning",
    "midday",
    "evening",
    "weekdays",
    "season",
    "harvest",
    "holiday",
    "weekend",
    "noon",
    "dusk",
    "dawn",
    "closing",
)


_PRUNESWEEP_TEMPLATES = (
    "The {subject} holds {object} inside the {place} each {time}.",
    "A {subject} keeps {object} near the {place} during {time}.",
    "Workers quietly move {object} past the {place} during {time}.",
    "Each {time} the {subject} stores {object} under the {place}.",
)


_PRUNESWEEP_NEGATION_WORDS = frozenset(
    {
        "no",
        "not",
        "never",
        "none",
        "cannot",
        "without",
        "neither",
        "nor",
        "nothing",
        "nobody",
        "nowhere",
        "hardly",
        "barely",
        "scarcely",
    }
)


_PRUNESWEEP_MAX_CANDIDATES = 24


_PRUNESWEEP_MAX_GROUND_RECORDS = _PRUNESWEEP_MAX_CANDIDATES * _PRUNESWEEP_CASE_COUNT


_PRUNESWEEP_GENERATION_BUDGET = len(_PRUNESWEEP_ARM_KEYS) * _PRUNESWEEP_MAX_GROUND_RECORDS


_PRUNESWEEP_HEADLINE_GENERATIONS = len(_PRUNESWEEP_HEADLINE_ARMS) * _PRUNESWEEP_MAX_GROUND_RECORDS


_PRUNESWEEP_MAX_RECORDS_BYTES = 12 * 1024 * 1024


_PRUNESWEEP_CELL_FIELDS = {
    "verdict",
    "boundary_found",
    "boundary_text_match",
    "input_tokens",
    "answer_digest",
}


_PRUNESWEEP_DETAILS = ("arm_context_exceeded", "arm_evaluation_failed")


_PRUNESWEEP_DECOMPOSITION_CLASSES = (
    "exact_pass",
    "boundary_only",
    "wrong_value",
    "non_value",
)


_PRUNESWEEP_OPERATING_CHARACTERISTICS: dict[str, Any] = {
    "null_random_stated": 0.0121,
    "null_competent_stated": 0.0143,
    "null_jitter_stated": 0.000005,
    "null_competent_exchangeable_v09": 0.003,
    "null_competent_exchangeable_v05": 0.0004,
    "null_competent_exchangeable_v02": 0.000004,
    "power_q25": 0.3969,
    "power_q50": 0.9601,
    "power_q75": 0.9996,
    "familywise_expected_stated": 0.34,
    "familywise_any_stated": 0.29,
    "familywise_expected_exchangeable": 0.07,
    "familywise_any_exchangeable": 0.07,
    "shape_returns_cap_graded": 0.094,
    "shape_intermediate_support_threshold": 0.0,
}


_PRUNESWEEP_AGGREGATE_KEYS = {
    "arms",
    "headline",
    "shape",
    "tail_contrast",
    "ids",
    "dropped_shares",
    "operating_characteristics",
    "n_cases",
    "n_incomplete",
}


def _prunesweep_digest(text: str) -> str:
    """Return the prunesweep digest."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _prunesweep_digest_ok(value: Any) -> bool:
    """Return whether the prunesweep digest is well-formed."""

    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _prunesweep_value_digests() -> frozenset[str]:
    """Return the prunesweep value digests."""
    return frozenset(_prunesweep_digest(value) for value in _PRUNESWEEP_VALUES)


def _prunesweep_draw_sentence(seed: int, case_id: str, slot: int, seen: set[str]) -> str:
    """Mirror of prunesweep._draw_sentence."""
    template = _PRUNESWEEP_TEMPLATES[slot % len(_PRUNESWEEP_TEMPLATES)]
    counter = 0
    while counter < 10000:
        digest = _prunesweep_digest(f"{seed}:{case_id}:slot:{slot}:{counter}")
        text = template.format(
            subject=_PRUNESWEEP_SUBJECTS[int(digest[0:8], 16) % len(_PRUNESWEEP_SUBJECTS)],
            object=_PRUNESWEEP_OBJECTS[int(digest[8:16], 16) % len(_PRUNESWEEP_OBJECTS)],
            place=_PRUNESWEEP_PLACES[int(digest[16:24], 16) % len(_PRUNESWEEP_PLACES)],
            time=_PRUNESWEEP_TIMES[int(digest[24:32], 16) % len(_PRUNESWEEP_TIMES)],
        )
        if text not in seen:
            return text
        counter += 1
    raise ValueError("prune-sweep sentence draw did not terminate")


def _prunesweep_generate(seed: int) -> list[dict[str, Any]]:
    """Mirror of prunesweep.generate_cases: the frozen case
    set for a seed, regenerated from the protocol's own seed
    so a fabricated set is rejected (the numform F2 lesson).
    """
    cases: list[dict[str, Any]] = []
    for index in range(_PRUNESWEEP_CASE_COUNT):
        case_id = f"prune-{index:02d}"
        topic = _PRUNESWEEP_TOPICS[index % len(_PRUNESWEEP_TOPICS)]
        value = _PRUNESWEEP_VALUES[index]
        seen: set[str] = set()
        sentences: list[str] = []
        for slot in range(_PRUNESWEEP_CONTEXT_SENTENCES):
            text = _prunesweep_draw_sentence(seed, case_id, slot, seen)
            seen.add(text)
            sentences.append(text)
        cases.append(
            {
                "fixture_id": case_id,
                "topic": topic,
                "value": value,
                "expected": value,
                "context_sentences": sentences,
                "fact_sentence": _PRUNESWEEP_FACT_TEMPLATE.format(topic=topic, value=value),
            }
        )
    return cases


def _prunesweep_drop_order(seed: int, case_id: str) -> tuple[int, ...]:
    """Mirror of prunesweep.drop_order (amendment E: the
    position is the eligible-list index in passage order).
    """
    return tuple(
        sorted(
            range(_PRUNESWEEP_ELIGIBLE_WORDS),
            key=lambda position: _prunesweep_digest(f"{seed}:{case_id}:{position}"),
        )
    )


def _prunesweep_dropped_positions(case_id: str, arm: str, seed: int) -> frozenset[int]:
    """Mirror of prunesweep.dropped_positions."""
    if arm == _PRUNESWEEP_DESCRIPTIVE_ARM:
        return frozenset(range(_PRUNESWEEP_ELIGIBLE_WORDS // 2, _PRUNESWEEP_ELIGIBLE_WORDS))
    dose = {"dose0": 0, "dose18": 18, "dose36": 36, "dose72": 72}.get(arm)
    if dose is None:
        return frozenset()
    return frozenset(_prunesweep_drop_order(seed, case_id)[:dose])


def _prunesweep_arm_content(case: dict[str, Any], arm: str, seed: int) -> str:
    """Mirror of prunesweep.arm_content."""
    question = _PRUNESWEEP_QUESTION_TEMPLATE.format(topic=case["topic"])
    if arm == _PRUNESWEEP_SCREEN_ARM:
        return f"{_PRUNESWEEP_INSTRUCTION}\n{question}"
    dropped = _prunesweep_dropped_positions(str(case["fixture_id"]), arm, seed)
    kept: dict[int, str] = {}
    for slot, sentence in enumerate(case["context_sentences"]):
        words = sentence.split(" ")
        surviving = [
            word
            for offset, word in enumerate(words)
            if slot * _PRUNESWEEP_WORDS_PER_SENTENCE + offset not in dropped
        ]
        if surviving:
            kept[slot] = " ".join(surviving)
    sentences = [kept[slot] for slot in range(_PRUNESWEEP_FACT_CONTEXT_SLOTS) if slot in kept]
    sentences.append(str(case["fact_sentence"]))
    sentences.extend(
        kept[slot]
        for slot in range(_PRUNESWEEP_FACT_CONTEXT_SLOTS, _PRUNESWEEP_CONTEXT_SENTENCES)
        if slot in kept
    )
    passage = " ".join(sentences)
    return f"{_PRUNESWEEP_INSTRUCTION}\n{passage}\n{question}"


def _prunesweep_dropped_shares(case: dict[str, Any], seed: int) -> dict[str, Any]:
    """Mirror of prunesweep.dropped_set_shares."""
    words = [word for sentence in case["context_sentences"] for word in sentence.split(" ")]

    def _classes(word: str) -> tuple[bool, bool]:
        """Return the classes."""
        bare = (word[:-1] if word.endswith(".") else word).lower()
        return any(char.isdigit() for char in bare), bare in _PRUNESWEEP_NEGATION_WORDS

    shares: dict[str, Any] = {}
    for arm in (*_PRUNESWEEP_HEADLINE_ARMS[1:], _PRUNESWEEP_DESCRIPTIVE_ARM):
        dropped = _prunesweep_dropped_positions(str(case["fixture_id"]), arm, seed)
        numeral = negation = 0
        for position in dropped:
            is_numeral, is_negation = _classes(words[position])
            numeral += 1 if is_numeral else 0
            negation += 1 if is_negation else 0
        total = len(dropped)
        shares[arm] = {
            "numeral": numeral / total if total else 0.0,
            "negation": negation / total if total else 0.0,
        }
    return shares


def _prunesweep_case_material_sha256() -> str:
    """Mirror of prunesweep.case_material_sha256."""

    material = {
        "instruction": _PRUNESWEEP_INSTRUCTION,
        "fact_template": _PRUNESWEEP_FACT_TEMPLATE,
        "question_template": _PRUNESWEEP_QUESTION_TEMPLATE,
        "topics": list(_PRUNESWEEP_TOPICS),
        "values": list(_PRUNESWEEP_VALUES),
        "subjects": list(_PRUNESWEEP_SUBJECTS),
        "objects": list(_PRUNESWEEP_OBJECTS),
        "places": list(_PRUNESWEEP_PLACES),
        "times": list(_PRUNESWEEP_TIMES),
        "templates": list(_PRUNESWEEP_TEMPLATES),
        "negation_words": sorted(_PRUNESWEEP_NEGATION_WORDS),
        "doses": list(_PRUNESWEEP_DOSES),
    }
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return _prunesweep_digest(canonical)


def _prunesweep_ground_sha256(seed: int) -> str:
    """Mirror of prunesweep.ground_sha256."""

    rows = [
        [
            case["fixture_id"],
            case["topic"],
            case["value"],
            list(case["context_sentences"]),
            case["fact_sentence"],
        ]
        for case in _prunesweep_generate(seed)
    ]
    canonical = json.dumps(rows, separators=(",", ":"), ensure_ascii=True)
    return _prunesweep_digest(canonical)


def _prunesweep_protocol(seed: int, backend_fingerprint: str) -> dict[str, Any]:
    """The prune-sweep protocol rebuilt from this module's
    own literals plus the payload's seed and backend
    fingerprint, fingerprint included (compared whole).
    """

    protocol: dict[str, Any] = {
        "version": 1,
        "arm_keys": list(_PRUNESWEEP_ARM_KEYS),
        "headline_arms": list(_PRUNESWEEP_HEADLINE_ARMS),
        "descriptive_arm": _PRUNESWEEP_DESCRIPTIVE_ARM,
        "screen_arm": _PRUNESWEEP_SCREEN_ARM,
        "doses": list(_PRUNESWEEP_DOSES),
        "case_count": _PRUNESWEEP_CASE_COUNT,
        "eligible_words": _PRUNESWEEP_ELIGIBLE_WORDS,
        "case_material_sha256": _prunesweep_case_material_sha256(),
        "ground_sha256": _prunesweep_ground_sha256(seed),
        "min_c": _PRUNESWEEP_MIN_C,
        "min_a": _PRUNESWEEP_MIN_A,
        "generation_budget": _PRUNESWEEP_GENERATION_BUDGET,
        "headline_generations": _PRUNESWEEP_HEADLINE_GENERATIONS,
        "max_ground_records": _PRUNESWEEP_MAX_GROUND_RECORDS,
        "max_records_bytes": _PRUNESWEEP_MAX_RECORDS_BYTES,
        "max_candidates": _PRUNESWEEP_MAX_CANDIDATES,
        "seed": seed,
        "backend_fingerprint": backend_fingerprint,
    }
    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    protocol["fingerprint"] = _prunesweep_digest(canonical)
    return protocol


def _prunesweep_mcnemar_p(a: int, b: int) -> float | None:
    """Two-sided exact McNemar p (the confidentiality formula,
    local, as in the sibling mirrors).
    """

    n = a + b
    if n == 0:
        return None
    k = min(a, b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    result: float = min(1.0, 2.0 * tail)
    return result


def _prunesweep_classify(direct_passes: int, drop_cases: int, reverse: int) -> str:
    """Mirror of prunesweep.classify_decision."""
    if direct_passes < _PRUNESWEEP_MIN_C:
        return "insufficient"
    p = _prunesweep_mcnemar_p(drop_cases, reverse)
    if drop_cases >= _PRUNESWEEP_MIN_A and p is not None and p < 0.05:
        return "prune_drop_observed"
    return "below_threshold"


def _prunesweep_decompose(cell: dict[str, Any], expected_digest: str) -> str:
    """Mirror of prunesweep.decompose_cell: the class from
    the recorded digest alone, in the frozen order
    (exact_pass, boundary_only, wrong_value, non_value --
    amendment B).
    """
    if cell.get("verdict") is True:
        return "exact_pass"
    digest = cell.get("answer_digest")
    if digest is not None and digest == expected_digest:
        return "boundary_only"
    if digest is not None and digest in _prunesweep_value_digests():
        return "wrong_value"
    return "non_value"


def _prunesweep_cell_get(record: dict[str, Any], key: str) -> dict[str, Any] | None:
    """Return the prunesweep cell get."""
    cells = record.get("cells")
    if not isinstance(cells, dict):
        return None
    cell = cells.get(key)
    return cell if isinstance(cell, dict) else None


def _prunesweep_detail_get(record: dict[str, Any], key: str) -> str:
    """Return the prunesweep detail get."""
    details = record.get("cell_details")
    if not isinstance(details, dict):
        return ""
    detail = details.get(key, "")
    return detail if type(detail) is str else ""


def _prunesweep_verdict(record: dict[str, Any], key: str) -> bool:
    """Return the prunesweep verdict."""
    cell = _prunesweep_cell_get(record, key)
    return bool(cell is not None and cell["verdict"])


def _prunesweep_screened(record: dict[str, Any]) -> bool:
    """Return the prunesweep screened."""
    cell = _prunesweep_cell_get(record, _PRUNESWEEP_SCREEN_ARM)
    return bool(cell is not None and cell["verdict"])


def _prunesweep_ids_identical(record: dict[str, Any]) -> bool:
    """Mirror of prunesweep._ids_identical: derived from the
    recorded id digests alone.
    """
    digests = record.get("id_digests")
    if not isinstance(digests, dict):
        return False
    return digests.get("dose0") is not None and digests.get("dose0") == digests.get("dose72")


def _prunesweep_first_flip(record: dict[str, Any]) -> str:
    """Return the prunesweep first flip."""
    for arm in _PRUNESWEEP_HEADLINE_ARMS[1:]:
        if not _prunesweep_verdict(record, arm):
            return arm
    return "never"


def _prunesweep_aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Mirror of prunesweep.aggregate_candidate."""
    arms: dict[str, Any] = {}
    for key in _PRUNESWEEP_ARM_KEYS:
        n_completed = 0
        n_pass = 0
        n_context = 0
        n_failed = 0
        decomposition = dict.fromkeys(_PRUNESWEEP_DECOMPOSITION_CLASSES, 0)
        for record in records:
            detail = _prunesweep_detail_get(record, key)
            if detail == "arm_context_exceeded":
                n_context += 1
            elif detail == "arm_evaluation_failed":
                n_failed += 1
            cell = _prunesweep_cell_get(record, key)
            if cell is None:
                continue
            n_completed += 1
            if cell["verdict"]:
                n_pass += 1
            decomposition[_prunesweep_decompose(cell, str(record.get("expected_digest", "")))] += 1
        arms[key] = {
            "n_completed": n_completed,
            "n_pass": n_pass,
            "rate": (n_pass / n_completed) if n_completed else None,
            "n_context_exceeded": n_context,
            "n_evaluation_failed": n_failed,
            "decomposition": decomposition,
        }
    complete = [
        record
        for record in records
        if all(_prunesweep_cell_get(record, arm) is not None for arm in _PRUNESWEEP_HEADLINE_ARMS)
    ]
    screened = [record for record in complete if _prunesweep_screened(record)]
    unscreened = [record for record in complete if not _prunesweep_screened(record)]
    ids_excluded = [record for record in unscreened if _prunesweep_ids_identical(record)]
    population = [record for record in unscreened if not _prunesweep_ids_identical(record)]
    direct = [record for record in population if _prunesweep_verdict(record, "dose0")]
    drop_cases = 0
    boundary_attrition = 0
    a_by_first_flip = dict.fromkeys(_PRUNESWEEP_HEADLINE_ARMS[1:], 0)
    for record in direct:
        cell72 = _prunesweep_cell_get(record, "dose72")
        if cell72 is None or cell72["verdict"]:
            continue
        klass = _prunesweep_decompose(cell72, str(record.get("expected_digest", "")))
        if klass == "boundary_only":
            boundary_attrition += 1
        elif klass in ("wrong_value", "non_value"):
            drop_cases += 1
            a_by_first_flip[_prunesweep_first_flip(record)] += 1
    reverse = sum(
        1
        for record in population
        if not _prunesweep_verdict(record, "dose0") and _prunesweep_verdict(record, "dose72")
    )
    headline = {
        "n_population": len(population),
        "n_headline_incomplete": len(records) - len(complete),
        "n_screened": len(screened),
        "n_closed_book_incomplete": sum(
            1 for record in records if _prunesweep_cell_get(record, _PRUNESWEEP_SCREEN_ARM) is None
        ),
        "n_ids_identical": len(ids_excluded),
        "direct_passes": len(direct),
        "drop_cases": drop_cases,
        "boundary_attrition": boundary_attrition,
        "reverse": reverse,
        "decision": _prunesweep_classify(len(direct), drop_cases, reverse),
        "a_by_first_flip": a_by_first_flip,
    }
    histogram = dict.fromkeys(_PRUNESWEEP_HEADLINE_ARMS[1:], 0)
    histogram["never"] = 0
    returns = 0
    for record in direct:
        histogram[_prunesweep_first_flip(record)] += 1
        if (
            not _prunesweep_verdict(record, "dose18") or not _prunesweep_verdict(record, "dose36")
        ) and _prunesweep_verdict(record, "dose72"):
            returns += 1
    shape = {
        "first_flip_histogram": histogram,
        "returns": returns,
        "dose_pass_rates": {
            arm: (
                (
                    sum(1 for record in population if _prunesweep_verdict(record, arm))
                    / len(population)
                )
                if population
                else None
            )
            for arm in _PRUNESWEEP_HEADLINE_ARMS
        },
    }
    n_compared = 0
    dose72_fails = 0
    tail72_fails = 0
    dose72_only = 0
    tail72_only = 0
    for record in population:
        if _prunesweep_cell_get(record, _PRUNESWEEP_DESCRIPTIVE_ARM) is None:
            continue
        n_compared += 1
        fail72 = not _prunesweep_verdict(record, "dose72")
        fail_tail = not _prunesweep_verdict(record, _PRUNESWEEP_DESCRIPTIVE_ARM)
        dose72_fails += 1 if fail72 else 0
        tail72_fails += 1 if fail_tail else 0
        if fail72 and not fail_tail:
            dose72_only += 1
        elif fail_tail and not fail72:
            tail72_only += 1
    p_tail = _prunesweep_mcnemar_p(dose72_only, tail72_only)
    rival = dose72_fails > 0 and (
        dose72_only + tail72_only == 0 or (p_tail is not None and p_tail >= 0.05)
    )
    tail_contrast = {
        "n_compared": n_compared,
        "dose72_fails": dose72_fails,
        "tail72_fails": tail72_fails,
        "dose72_only": dose72_only,
        "tail72_only": tail72_only,
        "rival_reading": rival,
    }
    ids: dict[str, Any] = {}
    for arm in (*_PRUNESWEEP_HEADLINE_ARMS[1:], _PRUNESWEEP_DESCRIPTIVE_ARM):
        deltas: list[int] = []
        n_ids_identical = 0
        n_same_ids_flips = 0
        for record in records:
            tokens = record.get("arm_tokens")
            tokens = tokens if isinstance(tokens, dict) else {}
            digests = record.get("id_digests")
            digests = digests if isinstance(digests, dict) else {}
            treated = tokens.get(arm)
            baseline = tokens.get("dose0")
            if type(treated) is int and type(baseline) is int:
                deltas.append(treated - baseline)
            else:
                deltas.append(0)
            base_digest = digests.get("dose0")
            if type(base_digest) is str and base_digest != "" and base_digest == digests.get(arm):
                n_ids_identical += 1
                base_cell = _prunesweep_cell_get(record, "dose0")
                arm_cell = _prunesweep_cell_get(record, arm)
                if (
                    base_cell is not None
                    and arm_cell is not None
                    and base_cell["verdict"] != arm_cell["verdict"]
                ):
                    n_same_ids_flips += 1
        ids[arm] = {
            "id_length_delta_by_case": deltas,
            "max_abs_id_delta": max((abs(delta) for delta in deltas), default=0),
            "n_ids_identical": n_ids_identical,
            "n_same_ids_flips": n_same_ids_flips,
        }
    share_arms = (*_PRUNESWEEP_HEADLINE_ARMS[1:], _PRUNESWEEP_DESCRIPTIVE_ARM)
    dropped_shares: dict[str, Any] = {}
    for arm in share_arms:
        numeral_total = 0.0
        negation_total = 0.0
        counted = 0
        for record in records:
            shares = record.get("dropped_shares")
            if not isinstance(shares, dict) or not isinstance(shares.get(arm), dict):
                continue
            numeral_total += float(shares[arm].get("numeral", 0.0))
            negation_total += float(shares[arm].get("negation", 0.0))
            counted += 1
        dropped_shares[arm] = {
            "numeral": numeral_total / counted if counted else 0.0,
            "negation": negation_total / counted if counted else 0.0,
        }
    return {
        "arms": arms,
        "headline": headline,
        "shape": shape,
        "tail_contrast": tail_contrast,
        "ids": ids,
        "dropped_shares": dropped_shares,
        "operating_characteristics": dict(_PRUNESWEEP_OPERATING_CHARACTERISTICS),
        "n_cases": len(records),
        "n_incomplete": sum(
            1
            for record in records
            if any(_prunesweep_cell_get(record, arm) is None for arm in _PRUNESWEEP_ARM_KEYS)
        ),
    }


def _prunesweep_cell_ok(cell: Any, expected_digest: str) -> bool:
    """One cell's shape and bindings: exact field set, the
    verdict binding ``verdict == (answer_digest ==
    expected_digest) and boundary_found``, and the answer
    digest's shape (None when the backend produced no
    answer).
    """
    if type(cell) is not dict or set(cell) != _PRUNESWEEP_CELL_FIELDS:
        return False
    if type(cell["verdict"]) is not bool or type(cell["boundary_found"]) is not bool:
        return False
    if cell["boundary_text_match"] is not None and type(cell["boundary_text_match"]) is not bool:
        return False
    if type(cell["input_tokens"]) is not int or cell["input_tokens"] < 0:
        return False
    digest = cell["answer_digest"]
    if digest is not None and not _prunesweep_digest_ok(digest):
        return False
    verdict = (digest == expected_digest) and cell["boundary_found"]
    return bool(cell["verdict"] == verdict)


def _prunesweep_recompute(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Recompute a prune-sweep payload from its ground
    records.

    The validator carries its own generator mirror, frozen
    texts, drop permutation, and aggregators (no imports
    from prunesweep.py): the protocol is rebuilt from frozen
    literals plus the payload's seed and backend fingerprint
    and compared whole; the case set is regenerated from the
    seed and every record is checked against it in protocol
    order; every record's per-arm content digests are
    rebound to the regenerated arm contents, so the drop
    sets, the protected parts, and the closed-book content
    are all re-derived; the record's ids_identical flag must
    equal the derivation from its recorded id digests; the
    dropped-set class shares must equal the regenerated
    cases' shares; every cell's verdict is re-derived from
    its digest identity and bound to the record's arm token
    count; the decomposition, the headline decision, the
    shape readout, and the tail contrast are recomputed
    from the recorded digests alone; the aggregates are
    recomputed and compared.
    """

    protocol = payload.get("protocol")
    if type(protocol) is not dict:
        return None
    seed = protocol.get("seed")
    backend_fingerprint = protocol.get("backend_fingerprint")
    if type(seed) is not int or isinstance(seed, bool) or type(backend_fingerprint) is not str:
        return None
    if not _recomputed_equal(protocol, _prunesweep_protocol(seed, backend_fingerprint)):
        return None
    candidates = payload.get("candidates")
    if type(candidates) is not dict or not candidates:
        return None
    if len(candidates) > _PRUNESWEEP_MAX_CANDIDATES:
        return None
    generated = _prunesweep_generate(seed)
    total_records = 0
    records_bytes = 0
    aggregates: dict[str, Any] = {}
    for name, entry in candidates.items():
        if type(entry) is not dict:
            return None
        status = entry.get("status")
        if status == "unavailable":
            if set(entry) != {"status", "detail", "records"} or entry.get("records") != []:
                return None
            continue
        if status != "available" or entry.get("detail") != "":
            return None
        if set(entry) != _PRUNESWEEP_AGGREGATE_KEYS | {"status", "detail", "records"}:
            return None
        records = entry["records"]
        if type(records) is not list or len(records) != len(generated):
            return None
        total_records += len(records)
        records_bytes += len(
            json.dumps(records, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        )
        if records_bytes > _PRUNESWEEP_MAX_RECORDS_BYTES:
            return None
        expected_shares = [_prunesweep_dropped_shares(case, seed) for case in generated]
        for record, case, case_shares in zip(records, generated, expected_shares, strict=True):
            if type(record) is not dict or set(record) != {
                "fixture",
                "topic",
                "expected_digest",
                "arm_tokens",
                "id_digests",
                "ids_identical",
                "content_digests",
                "dropped_shares",
                "cells",
                "cell_details",
            }:
                return None
            if record["fixture"] != case["fixture_id"] or record["topic"] != case["topic"]:
                return None
            expected_digest = _prunesweep_digest(str(case["value"]))
            if record["expected_digest"] != expected_digest:
                return None
            arm_tokens = record["arm_tokens"]
            id_digests = record["id_digests"]
            content_digests = record["content_digests"]
            cells = record["cells"]
            details = record["cell_details"]
            for mapping in (arm_tokens, id_digests, content_digests, cells, details):
                if type(mapping) is not dict or set(mapping) != set(_PRUNESWEEP_ARM_KEYS):
                    return None
            if record["ids_identical"] is not (id_digests["dose0"] == id_digests["dose72"]):
                return None
            if not _recomputed_equal(record["dropped_shares"], case_shares):
                return None
            for arm in _PRUNESWEEP_ARM_KEYS:
                tokens = arm_tokens[arm]
                if type(tokens) is not int or isinstance(tokens, bool) or tokens < 0:
                    return None
                if not _prunesweep_digest_ok(id_digests[arm]):
                    return None
                if content_digests[arm] != _prunesweep_digest(
                    _prunesweep_arm_content(case, arm, seed)
                ):
                    return None
                cell = cells[arm]
                detail = details[arm]
                if detail not in ("", *_PRUNESWEEP_DETAILS):
                    return None
                if cell is None:
                    if detail == "":
                        return None
                    continue
                if detail != "":
                    return None
                if not _prunesweep_cell_ok(cell, expected_digest):
                    return None
                if cell["input_tokens"] != tokens:
                    return None
        aggregates[name] = _prunesweep_aggregate(records)
    if total_records > _PRUNESWEEP_MAX_GROUND_RECORDS:
        return None
    if not aggregates:
        return None
    return {
        "aggregates": aggregates,
        "n_candidates": len(candidates),
        "n_cases": len(generated),
    }


def _valid_prune_sweep(payload: Any) -> bool:
    """Structural check for a ``prune_sweep`` payload
    (recompute standard). The audit block is shape-checked
    only: its counts are stage observations (special-token
    findings against the run's own formats), not recomputable
    from records.
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
    audit = payload.get("audit")
    if type(audit) is not dict or set(audit) != {
        "n_audit_excluded",
        "by_reason",
        "special_token_findings",
    }:
        return False
    if type(audit["n_audit_excluded"]) is not int or audit["n_audit_excluded"] < 0:
        return False
    if type(audit["special_token_findings"]) is not int or audit["special_token_findings"] < 0:
        return False
    by_reason = audit["by_reason"]
    if type(by_reason) is not dict:
        return False
    for key, count in by_reason.items():
        if type(key) is not str or type(count) is not int or count < 0:
            return False
    recomputed = _prunesweep_recompute(payload)
    if recomputed is None:
        return False
    candidates = payload["candidates"]
    for name, expected in recomputed["aggregates"].items():
        entry = candidates[name]
        for key, value in expected.items():
            if not _recomputed_equal(entry.get(key), value):
                return False
    if not _recomputed_equal(payload.get("n_candidates"), recomputed["n_candidates"]):
        return False
    return _recomputed_equal(payload.get("n_cases"), recomputed["n_cases"])


def diff_prune_sweep(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``prune_sweep`` probe payloads;
    deterministic.

    Answers "did the prune-sweep story change between
    runs?": per-candidate headline drift (decision with the
    C / A / B counts and boundary attrition that carry it),
    the tail72 contrast drift with the rival-reading flag,
    the same-ids flip drift on dose72, candidates
    added/removed, and protocol / generator / backend drift
    (any of which marks the comparison not comparable, with
    the drift still listed -- the generated panel is the
    treatment, so a changed generator is protocol drift). A
    side without a valid payload is reported plainly.
    """
    old_ok = _valid_prune_sweep(old)
    new_ok = _valid_prune_sweep(new)
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
    ps_protocol_drift = old["protocol"].get("fingerprint") != new["protocol"].get("fingerprint")
    ps_generator_drift = old["protocol"].get("ground_sha256") != new["protocol"].get(
        "ground_sha256"
    ) or old["protocol"].get("case_material_sha256") != new["protocol"].get("case_material_sha256")
    ps_backend_drift = old["protocol"].get("backend_fingerprint") != new["protocol"].get(
        "backend_fingerprint"
    )
    ps_comparable = not ps_protocol_drift and not ps_generator_drift and not ps_backend_drift
    old_candidates = old["candidates"]
    new_candidates = new["candidates"]
    ps_candidates_added = sorted(set(new_candidates) - set(old_candidates))
    ps_candidates_removed = sorted(set(old_candidates) - set(new_candidates))
    ps_candidate_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_candidates) & set(new_candidates)):
        old_entry = old_candidates[name]
        new_entry = new_candidates[name]
        if old_entry.get("status") != "available" or new_entry.get("status") != "available":
            if old_entry.get("status") != new_entry.get("status"):
                ps_candidate_drift.append(
                    {
                        "candidate": name,
                        "metric": "status",
                        "old": old_entry.get("status"),
                        "new": new_entry.get("status"),
                    }
                )
            continue
        ps_metrics: list[tuple[str, Any, Any]] = [
            ("decision", old_entry["headline"]["decision"], new_entry["headline"]["decision"]),
            (
                "direct_passes",
                old_entry["headline"]["direct_passes"],
                new_entry["headline"]["direct_passes"],
            ),
            (
                "drop_cases",
                old_entry["headline"]["drop_cases"],
                new_entry["headline"]["drop_cases"],
            ),
            (
                "boundary_attrition",
                old_entry["headline"]["boundary_attrition"],
                new_entry["headline"]["boundary_attrition"],
            ),
            ("reverse", old_entry["headline"]["reverse"], new_entry["headline"]["reverse"]),
            (
                "n_population",
                old_entry["headline"]["n_population"],
                new_entry["headline"]["n_population"],
            ),
            (
                "dose72_only",
                old_entry["tail_contrast"]["dose72_only"],
                new_entry["tail_contrast"]["dose72_only"],
            ),
            (
                "tail72_only",
                old_entry["tail_contrast"]["tail72_only"],
                new_entry["tail_contrast"]["tail72_only"],
            ),
            (
                "rival_reading",
                old_entry["tail_contrast"]["rival_reading"],
                new_entry["tail_contrast"]["rival_reading"],
            ),
            (
                "n_same_ids_flips_dose72",
                old_entry["ids"]["dose72"]["n_same_ids_flips"],
                new_entry["ids"]["dose72"]["n_same_ids_flips"],
            ),
        ]
        for metric, old_value, new_value in ps_metrics:
            if old_value != new_value:
                ps_candidate_drift.append(
                    {"candidate": name, "metric": metric, "old": old_value, "new": new_value}
                )
    result["candidate_drift"] = ps_candidate_drift
    result["candidates_added"] = ps_candidates_added
    result["candidates_removed"] = ps_candidates_removed
    result["protocol_drift"] = ps_protocol_drift
    result["generator_drift"] = ps_generator_drift
    result["backend_drift"] = ps_backend_drift
    result["comparable"] = ps_comparable
    result["changed"] = bool(
        ps_candidate_drift
        or ps_candidates_added
        or ps_candidates_removed
        or ps_protocol_drift
        or ps_generator_drift
        or ps_backend_drift
    )
    return result


def _valid_prefill(payload: Any) -> bool:
    """Structural check for a ``prefill`` payload.

    Enforces the wave-2 C3 lesson at the value level: the compared
    fields must be bools/strings/None/numbers so the renderers never hit
    a TypeError on a hostile payload, and map keys must be strings.
    """
    if type(payload) is not dict or payload.get("available") is not True:
        return False
    prefills = payload.get("prefills")
    if type(prefills) is not list or not prefills or not all(type(p) is str for p in prefills):
        return False
    per_condition = payload.get("per_condition")
    if type(per_condition) is not dict or set(per_condition) != set(prefills):
        return False
    for condition in per_condition.values():
        if type(condition) is not dict:
            return False
        if type(condition.get("decision")) is not str:
            return False
        winner = condition.get("winner")
        if winner is not None and type(winner) is not str:
            return False
        candidates = condition.get("per_candidate")
        if type(candidates) is not dict or not all(type(k) is str for k in candidates):
            return False
        for entry in candidates.values():
            if entry is None:
                continue
            if type(entry) is not dict:
                return False
            for key in ("honor_rate", "restart_rate"):
                rate = entry.get(key)
                if rate is not None and type(rate) not in (int, float):
                    return False
            think = entry.get("think_channel")
            if think is not None and type(think) is not bool:
                return False
            if entry.get("pass") is not None and type(entry.get("pass")) is not bool:
                return False
    think_effect = payload.get("think_effect")
    if type(think_effect) is not dict:
        return False
    for label, group in think_effect.items():
        if type(label) is not str:
            return False
        if label == "delta":
            if group is not None and type(group) not in (int, float):
                return False
            continue
        if type(group) is not dict:
            return False
        mean = group.get("mean_honor_rate")
        if mean is not None and type(mean) not in (int, float):
            return False
        if type(group.get("n_candidates")) is not int:
            return False
    return type(payload.get("flips")) is list


def _prefill_condition_triplet(payload: Any, prefill_id: str) -> tuple[Any, Any, Any]:
    """(decision, winner) for one prefill condition, for drift comparison."""
    per_condition = payload.get("per_condition") or {}
    condition = per_condition.get(prefill_id) or {}
    return (condition.get("decision"), condition.get("winner"), prefill_id)


def diff_prefill(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``prefill`` payloads; deterministic.

    Answers "did the assistant-prefill sensitivity story change between
    runs?": the per-prefill (decision, winner) triplets, the winner-flip
    pattern against the native run, per-candidate honoring-rate drift,
    and the think-channel vs plain group-rate comparison. A changed
    flip pattern after a checkpoint or fixture update is the
    prefill-sensitivity regression signal. Candidates present on only
    one side are reported as added/removed. A side without a recorded
    prefill experiment is reported plainly. No model, no network.
    """
    old_ok = _valid_prefill(old)
    new_ok = _valid_prefill(new)
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
    old_prefills = old.get("prefills") or []
    new_prefills = new.get("prefills") or []
    prefills_added = [p for p in new_prefills if p not in old_prefills]
    prefills_removed = [p for p in old_prefills if p not in new_prefills]
    condition_changed: dict[str, dict[str, Any]] = {}
    for prefill_id in sorted(set(old_prefills) & set(new_prefills)):
        old_triplet = _prefill_condition_triplet(old, prefill_id)[:2]
        new_triplet = _prefill_condition_triplet(new, prefill_id)[:2]
        changed = old_triplet != new_triplet
        condition_changed[prefill_id] = {
            "old_decision": old_triplet[0],
            "old_winner": old_triplet[1],
            "new_decision": new_triplet[0],
            "new_winner": new_triplet[1],
            "changed": changed,
        }

    def _flip_key(flips: list[Any]) -> list[tuple[Any, ...]]:
        # ``flips`` is a list here: _valid_prefill rejects anything else
        # before this point, so no defensive branch is needed.
        """Return the flip key."""
        keys = []
        for entry in flips:
            if type(entry) is not dict:
                continue
            keys.append(
                (
                    entry.get("prefill"),
                    entry.get("decision_from"),
                    entry.get("winner_from"),
                    entry.get("decision_to"),
                    entry.get("winner_to"),
                )
            )
        return sorted(keys, key=lambda key: tuple(str(part) for part in key))

    # NOTE: phrased to stay textually distinct from diff_hierarchy's
    # flips_changed line (a cycle-147 mutation-gate anchor).
    old_flip_key = _flip_key(old.get("flips"))
    new_flip_key = _flip_key(new.get("flips"))
    flips_changed = old_flip_key != new_flip_key

    def _candidate_rates(payload: Any) -> dict[str, dict[str, tuple[Any, Any]]]:
        rates: dict[str, dict[str, tuple[Any, Any]]] = {}
        per_condition = payload.get("per_condition") or {}
        for prefill_id in sorted(per_condition):
            candidates = (per_condition.get(prefill_id) or {}).get("per_candidate") or {}
            for name, entry in candidates.items():
                entry = entry or {}
                rates.setdefault(name, {})[prefill_id] = (
                    entry.get("honor_rate"),
                    entry.get("restart_rate"),
                )
        return rates

    old_rates = _candidate_rates(old)
    new_rates = _candidate_rates(new)
    candidates_added = sorted(set(new_rates) - set(old_rates))
    candidates_removed = sorted(set(old_rates) - set(new_rates))
    candidate_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_rates) & set(new_rates)):
        for prefill_id in sorted(set(old_rates[name]) & set(new_rates[name])):
            old_pair = old_rates[name][prefill_id]
            new_pair = new_rates[name][prefill_id]
            if old_pair != new_pair:
                candidate_drift.append(
                    {
                        "candidate": name,
                        "prefill": prefill_id,
                        "old_honor_rate": old_pair[0],
                        "old_restart_rate": old_pair[1],
                        "new_honor_rate": new_pair[0],
                        "new_restart_rate": new_pair[1],
                    }
                )

    def _think_means(payload: Any) -> dict[str, Any]:
        """Return the think means."""
        effect = payload.get("think_effect") or {}
        return {
            label: (group or {}).get("mean_honor_rate")
            for label, group in effect.items()
            if type(label) is str and label != "delta"
        }

    old_think = _think_means(old)
    new_think = _think_means(new)
    think_changed = old_think != new_think
    result["prefills_added"] = prefills_added
    result["prefills_removed"] = prefills_removed
    result["condition_changed"] = condition_changed
    result["flips_changed"] = flips_changed
    result["candidate_drift"] = candidate_drift
    result["candidates_added"] = candidates_added
    result["candidates_removed"] = candidates_removed
    result["think_rates"] = {
        "old": old_think,
        "new": new_think,
        "changed": think_changed,
    }
    result["changed"] = bool(
        prefills_added
        or prefills_removed
        or any(entry["changed"] for entry in condition_changed.values())
        or flips_changed
        or candidate_drift
        or candidates_added
        or candidates_removed
        or think_changed
    )
    return result


def _render_hierarchy_summary(hierarchy_diff: dict[str, Any] | None) -> list[str]:
    """Instruction-hierarchy comparison lines for the stderr summary."""
    if type(hierarchy_diff) is not dict:
        return []
    available = hierarchy_diff.get("available")
    if available == "neither":
        return []
    if available != "both":
        missing = "new" if available == "old" else "old"
        return [
            "Instruction-hierarchy comparison unavailable: "
            f"{missing} run did not record the hierarchy probe."
        ]
    lines: list[str] = []

    def _verdict(decision: Any, winner: Any) -> str:
        """Return the verdict."""
        return f"{_display_id(decision)}/{_display_id('(none)' if winner is None else winner)}"

    conditions = hierarchy_diff.get("condition_changed") or {}
    for position in sorted(conditions):
        entry = conditions[position]
        if entry.get("changed"):
            lines.append(
                f"Instruction hierarchy ({_display_id(position)}): "
                f"{_verdict(entry.get('old_decision'), entry.get('old_winner'))} -> "
                f"{_verdict(entry.get('new_decision'), entry.get('new_winner'))}"
            )
    added = hierarchy_diff.get("positions_added") or []
    removed = hierarchy_diff.get("positions_removed") or []
    if added:
        lines.append(f"Hierarchy positions added: {', '.join(_display_id(p) for p in added)}")
    if removed:
        lines.append(f"Hierarchy positions removed: {', '.join(_display_id(p) for p in removed)}")
    if hierarchy_diff.get("flips_changed"):
        lines.append("Instruction-hierarchy flip pattern changed between runs.")

    def _fmt_rate(value: Any) -> str:
        """Return the fmt rate."""
        return (
            f"{value:.1%}"
            if type(value) in (int, float) and _float_convertible(value)
            else "(none)"
        )

    drift = hierarchy_diff.get("candidate_drift") or []
    if drift:
        lines.append(f"Hierarchy adherence drift ({len(drift)}):")
        for entry in drift[:5]:
            lines.append(
                f"  {_display_id(entry['candidate'])} "
                f"({_display_id(entry.get('position'))}): system "
                f"{_fmt_rate(entry.get('old_system_rate'))}->"
                f"{_fmt_rate(entry.get('new_system_rate'))}, user "
                f"{_fmt_rate(entry.get('old_user_rate'))}->"
                f"{_fmt_rate(entry.get('new_user_rate'))}"
            )
        if len(drift) > 5:
            lines.append(f"  ... and {len(drift) - 5} more")
    added_c = hierarchy_diff.get("candidates_added") or []
    removed_c = hierarchy_diff.get("candidates_removed") or []
    if added_c:
        lines.append(f"Hierarchy candidates added: {', '.join(_display_id(c) for c in added_c)}")
    if removed_c:
        lines.append(
            f"Hierarchy candidates removed: {', '.join(_display_id(c) for c in removed_c)}"
        )
    groups = hierarchy_diff.get("group_rates") or {}
    if groups.get("changed"):
        old = groups.get("old") or {}
        new = groups.get("new") or {}
        for label in sorted(set(old) | set(new)):
            if old.get(label) != new.get(label):
                lines.append(
                    f"Hierarchy {_display_id(label)}-frame mean system rate: "
                    f"{_fmt_rate(old.get(label))} -> {_fmt_rate(new.get(label))}"
                )
    return lines


def _render_structured_summary(structured_diff: dict[str, Any] | None) -> list[str]:
    """Structured-output adherence comparison lines for the stderr summary."""
    if type(structured_diff) is not dict:
        return []
    available = structured_diff.get("available")
    if available == "neither":
        return []
    if available != "both":
        missing = "new" if available == "old" else "old"
        return [
            "Structured-output adherence comparison unavailable: "
            f"{missing} run did not record the structured probe."
        ]
    lines: list[str] = []

    def _fmt_rate(value: Any) -> str:
        """Return the fmt rate."""
        return (
            f"{value:.0%}"
            if type(value) in (int, float) and _float_convertible(value)
            else "(none)"
        )

    drift = structured_diff.get("rate_drift") or []
    if drift:
        lines.append(f"Structured adherence drift ({len(drift)}):")
        for entry in drift[:5]:
            if type(entry) is not dict:
                continue
            lines.append(
                f"  {_display_id(entry.get('candidate'))}/{_display_id(entry.get('metric'))}: "
                f"{_fmt_rate(entry.get('old'))} -> {_fmt_rate(entry.get('new'))}"
            )
        if len(drift) > 5:
            lines.append(f"  ... and {len(drift) - 5} more")
    gap_drift = structured_diff.get("gap_drift") or []
    for entry in gap_drift[:3]:
        if type(entry) is not dict:
            continue
        lines.append(
            "Structured adherence-gap drift: "
            f"{_fmt_rate(entry.get('old'))} -> {_fmt_rate(entry.get('new'))}"
        )
    for key, label in (
        ("weakest_added", "newly weakest"),
        ("weakest_removed", "no longer weakest"),
    ):
        names = structured_diff.get(key) or []
        if names:
            lines.append(f"Structured {label}: {', '.join(_display_id(c) for c in names)}")
    added = structured_diff.get("candidates_added") or []
    removed = structured_diff.get("candidates_removed") or []
    if added:
        lines.append(
            f"Structured-probe candidates added: {', '.join(_display_id(c) for c in added)}"
        )
    if removed:
        lines.append(
            f"Structured-probe candidates removed: {', '.join(_display_id(c) for c in removed)}"
        )
    return lines


def _render_faketurn_summary(faketurn_diff: dict[str, Any] | None) -> list[str]:
    """Fake-turn history-tampering comparison lines for the stderr summary.

    Every interpolated identifier passes through ``_display_id`` so a
    hostile candidate name cannot inject terminal escapes (the wave-3
    red-team A lesson).
    """
    if type(faketurn_diff) is not dict:
        return []
    available = faketurn_diff.get("available")
    if available == "neither":
        return []
    if available != "both":
        missing = "new" if available == "old" else "old"
        return [
            f"Fake-turn comparison unavailable: {missing} run did not record the faketurn probe."
        ]
    lines: list[str] = []

    def _fmt_rate(value: Any) -> str:
        """Return the fmt rate."""
        return (
            f"{value:.2f}"
            if type(value) in (int, float) and _float_convertible(value)
            else "(none)"
        )

    def _fmt_gap(value: Any) -> str:
        """Return the fmt gap."""
        return (
            f"{value:+.2f}"
            if type(value) in (int, float) and _float_convertible(value)
            else "(none)"
        )

    cell_drift = faketurn_diff.get("cell_drift") or []
    if cell_drift:
        lines.append(f"Fake-turn flip-rate drift ({len(cell_drift)}):")
        for entry in cell_drift[:5]:
            if type(entry) is not dict:
                continue
            lines.append(
                f"  {_display_id(entry.get('candidate'))}/{_display_id(entry.get('cell'))}: "
                f"{_fmt_rate(entry.get('old'))} -> {_fmt_rate(entry.get('new'))}"
            )
        if len(cell_drift) > 5:
            lines.append(f"  ... and {len(cell_drift) - 5} more")
    gap_drift = faketurn_diff.get("gap_drift") or []
    for entry in gap_drift[:3]:
        if type(entry) is not dict:
            continue
        lines.append(
            "Fake-turn transfer-gap drift "
            f"({_display_id(entry.get('candidate'))}): "
            f"{_fmt_gap(entry.get('old'))} -> {_fmt_gap(entry.get('new'))}"
        )
    source_drift = faketurn_diff.get("source_drift") or []
    for entry in source_drift[:3]:
        if type(entry) is not dict:
            continue
        lines.append(
            f"Fake-turn provenance drift ({_display_id(entry.get('candidate'))}/"
            f"{_display_id(entry.get('cell'))}): "
            f"{_display_id(entry.get('old'))} -> {_display_id(entry.get('new'))}"
        )
    added = faketurn_diff.get("candidates_added") or []
    removed = faketurn_diff.get("candidates_removed") or []
    if added:
        lines.append(f"Fake-turn candidates added: {', '.join(_display_id(c) for c in added)}")
    if removed:
        lines.append(f"Fake-turn candidates removed: {', '.join(_display_id(c) for c in removed)}")
    excl_added = faketurn_diff.get("excluded_added") or []
    excl_removed = faketurn_diff.get("excluded_removed") or []
    if excl_added:
        lines.append(f"Fake-turn newly excluded: {', '.join(_display_id(c) for c in excl_added)}")
    if excl_removed:
        lines.append(
            f"Fake-turn no longer excluded: {', '.join(_display_id(c) for c in excl_removed)}"
        )
    return lines


def _render_toolcall_summary(toolcall_diff: dict[str, Any] | None) -> list[str]:
    """Tool-call adherence comparison lines for the stderr summary.

    Every interpolated identifier passes through ``_display_id`` so a
    hostile candidate name cannot inject terminal escapes (the wave-3
    red-team A lesson).
    """
    if type(toolcall_diff) is not dict:
        return []
    available = toolcall_diff.get("available")
    if available == "neither":
        return []
    if available != "both":
        missing = "new" if available == "old" else "old"
        return [
            f"Tool-call comparison unavailable: {missing} run did not record the tool-call probe."
        ]
    lines: list[str] = []

    def _fmt_rate(value: Any) -> str:
        """Return the fmt rate."""
        return (
            f"{value:.2f}"
            if type(value) in (int, float) and _float_convertible(value)
            else "(none)"
        )

    for drift_key, label in (
        ("selection_drift", "selection-rate"),
        ("args_drift", "args-rate"),
        ("relevance_drift", "relevance-rate"),
    ):
        drift = toolcall_diff.get(drift_key) or []
        for entry in drift[:3]:
            if type(entry) is not dict:
                continue
            lines.append(
                f"Tool-call {label} drift ({_display_id(entry.get('candidate'))}): "
                f"{_fmt_rate(entry.get('old'))} -> {_fmt_rate(entry.get('new'))}"
            )
        if len(drift) > 3:
            lines.append(f"  ... and {len(drift) - 3} more {label} drifts")
    added = toolcall_diff.get("candidates_added") or []
    removed = toolcall_diff.get("candidates_removed") or []
    if added:
        lines.append(f"Tool-call candidates added: {', '.join(_display_id(c) for c in added)}")
    if removed:
        lines.append(f"Tool-call candidates removed: {', '.join(_display_id(c) for c in removed)}")
    excl_added = toolcall_diff.get("excluded_added") or []
    excl_removed = toolcall_diff.get("excluded_removed") or []
    if excl_added:
        lines.append(f"Tool-call newly excluded: {', '.join(_display_id(c) for c in excl_added)}")
    if excl_removed:
        lines.append(
            f"Tool-call no longer excluded: {', '.join(_display_id(c) for c in excl_removed)}"
        )
    return lines


def _render_prefill_summary(prefill_diff: dict[str, Any] | None) -> list[str]:
    """Assistant-prefill comparison lines for the stderr summary."""
    if type(prefill_diff) is not dict:
        return []
    available = prefill_diff.get("available")
    if available == "neither":
        return []
    if available != "both":
        missing = "new" if available == "old" else "old"
        return [
            "Assistant-prefill comparison unavailable: "
            f"{missing} run did not record the prefill experiment."
        ]
    lines: list[str] = []

    def _verdict(decision: Any, winner: Any) -> str:
        """Return the verdict."""
        return f"{_display_id(decision)}/{_display_id('(none)' if winner is None else winner)}"

    conditions = prefill_diff.get("condition_changed") or {}
    for prefill_id in sorted(conditions):
        entry = conditions[prefill_id]
        if entry.get("changed"):
            lines.append(
                f"Assistant prefill ({_display_id(prefill_id)}): "
                f"{_verdict(entry.get('old_decision'), entry.get('old_winner'))} -> "
                f"{_verdict(entry.get('new_decision'), entry.get('new_winner'))}"
            )
    added = prefill_diff.get("prefills_added") or []
    removed = prefill_diff.get("prefills_removed") or []
    if added:
        lines.append(f"Prefill conditions added: {', '.join(_display_id(p) for p in added)}")
    if removed:
        lines.append(f"Prefill conditions removed: {', '.join(_display_id(p) for p in removed)}")
    if prefill_diff.get("flips_changed"):
        lines.append("Assistant-prefill flip pattern changed between runs.")

    def _fmt_rate(value: Any) -> str:
        """Return the fmt rate."""
        return (
            f"{value:.1%}"
            if type(value) in (int, float) and _float_convertible(value)
            else "(none)"
        )

    drift = prefill_diff.get("candidate_drift") or []
    if drift:
        lines.append(f"Prefill honoring drift ({len(drift)}):")
        for entry in drift[:5]:
            lines.append(
                f"  {_display_id(entry['candidate'])} "
                f"({_display_id(entry.get('prefill'))}): honor "
                f"{_fmt_rate(entry.get('old_honor_rate'))}->"
                f"{_fmt_rate(entry.get('new_honor_rate'))}, restart "
                f"{_fmt_rate(entry.get('old_restart_rate'))}->"
                f"{_fmt_rate(entry.get('new_restart_rate'))}"
            )
        if len(drift) > 5:
            lines.append(f"  ... and {len(drift) - 5} more")
    added_c = prefill_diff.get("candidates_added") or []
    removed_c = prefill_diff.get("candidates_removed") or []
    if added_c:
        lines.append(f"Prefill candidates added: {', '.join(_display_id(c) for c in added_c)}")
    if removed_c:
        lines.append(f"Prefill candidates removed: {', '.join(_display_id(c) for c in removed_c)}")
    think = prefill_diff.get("think_rates") or {}
    if think.get("changed"):
        old = think.get("old") or {}
        new = think.get("new") or {}
        for label in sorted(set(old) | set(new)):
            if old.get(label) != new.get(label):
                lines.append(
                    f"Prefill {_display_id(label)} mean honor rate: "
                    f"{_fmt_rate(old.get(label))} -> {_fmt_rate(new.get(label))}"
                )
    return lines
