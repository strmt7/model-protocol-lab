"""Cycle 67 -- adversarial audit of diff-reports comparison semantics.

Scope
-----
``model_protocol_lab/diffreports.py``, everything after ``load_results``
(``load_results`` hostile-input handling was covered in cycle 64):
``diff_reports`` comparison semantics, the helpers it composes
(``_verdict``, ``_search_verdicts``, ``_held_verdict``,
``_fixture_outcomes``, ``_stage_evidence``, ``_config_snapshot``,
``_drift``, ``_fixture_set_drift``), and the ``render_summary``
"identical axes" / evidence-unavailable lines (the rest of the render
text is covered by ``tests/test_cycle13_diffreports.py``).

Method
------
1. Rules-as-found: the comparison rules below were read from the code,
   then cross-checked by executing probes (see COMPARISON RULES).
2. Hand-written adversarial pairs with EXACT expected outputs
   (non-tautological: every assertion names the full expected diff,
   including a one-field "minimal pair" per compared section).
3. An INDEPENDENT oracle written from the rules (different structure and
   control flow from the implementation) plus seeded differential
   fuzzing: randomized report pairs with scripted mutations must produce
   byte-identical comparison dicts from oracle and implementation, for
   both ``all_fixtures`` settings.
4. Mutation gate on /tmp copies of the module (>=3 semantic mutants);
   every pinned test must catch its mutant.
5. Red-team inputs: duplicate fixture IDs, duplicate candidate names,
   fixtures missing on one side or both directions, deeply nested
   observation structures, unicode IDs, case-sensitivity, evidence
   gating edge cases.

COMPARISON RULES (as found in diffreports.py, verified by execution)
-------------------------------------------------------------------
- decision: exact ``!=`` on the raw values ("select_best" vs "abstain").
- winner: ``selected_candidate`` with non-str values and ``""``
  normalized to None (matching ``_held_verdict``),
  BUT the empty string is kept as-is (``""`` vs missing/``None`` counts
  as "winner changed"). This differs from ``_held_verdict``, which
  treats ``""`` as absent -- see BUG-1 below.
- candidate verdicts (search): first occurrence of each candidate name
  wins; verdict "pass" iff summary has int planned/passes, bool
  ``aggregate_eligible`` True, planned > 0, and passes == planned;
  "fail" covers partial passes, ineligible runs, and planned <= 0;
  anything malformed -> "unknown" (bool planned is NOT int, so it is
  "unknown"). Missing on one side -> "absent".
- candidate verdicts (held_out): winner name vs held_out summary
  verdict; compared over the union of the two winner names.
- per-fixture outcomes: evidence-gated. A side "has evidence" iff ANY
  stage (search or held_out) yields at least one candidate with a
  non-empty fixture->outcome map. Per-fixture cells are compared only
  when BOTH sides have evidence. Matching is EXACT: fixture IDs and
  candidate names compare by plain string equality (case-sensitive,
  whitespace-sensitive, no Unicode normalization); ordering is
  normalized by sorting, so reordered fixtures/observations/candidates
  compare identical. "no fuzzy matching" = exact ID matching.
- fixture outcome: "pass" iff ``exact_match`` and ``boundary_found``
  are both True (bools required; anything else -> row skipped and
  counted, never silent). Duplicate fixture IDs inside one candidate's
  observations are LAST-wins.
- duplicate candidate names: DELIBERATE inconsistency, documented in
  the ``_fixture_outcomes`` docstring -- verdicts are FIRST-wins
  (``_search_verdicts`` skips repeat names) but evidence is LAST-wins
  (``_stage_evidence`` has no repeat-name guard; a later entry with no
  usable evidence silently keeps the earlier outcomes). ``mpl run``
  never emits duplicates; hostile-input behavior only.
- held-out evidence requires a non-empty ``selected_candidate``;
  held_out observations without a winner are ignored entirely.
- config drift fields, in fixed order: families (base name before "/"),
  candidates (sorted deduped names), max_new_tokens (int or None),
  boundaries (name -> token-id list; last VALID entry wins on duplicate
  names), backend_fingerprint (str or None); then ``planned:<name>``
  for candidates present on both sides (None compares as its own value);
  then ``fixtures:search`` / ``fixtures:held_out`` fixture-ID set drift,
  only when both sides carry any-stage evidence.
- summary: candidates_changed_status counts DISTINCT candidates across
  both stages; fixtures_flipped counts only pass<->fail transitions
  ("absent" transitions are changed diffs but not flips);
  identical requires decision, winner, verdicts, fixture diffs, config
  drift all clean AND evidence symmetry (one side with evidence and the
  other without -> not identical, with fixture comparison suppressed).
- unknown top-level keys are ignored; empty observations list and
  missing observations key both mean "no evidence".
- render_summary "identical" line names the compared axes; the
  per-fixture axis is listed only when both sides carried evidence.

Findings
--------
REAL bug (FOUND AND FIXED by the coordinator -- see below):

- BUG-1 (low), ``diffreports.py`` ``diff_reports`` (:361-364): the
  winner section normalizes only non-str ``selected_candidate`` values
  to None, keeping ``""`` as-is, while ``_held_verdict`` (:197) treats
  ``""`` as absent. Trigger: old ``selected_candidate=""``, new key
  missing/null, everything else identical -> ``winner`` section
  ``{"old": "", "new": None, "changed": true}``, ``winner_changed``
  true, ``identical`` false, with ``candidate_verdicts`` empty (both
  held verdicts absent) and a confusing ``Winner:  -> (none)`` render
  line. Two winnerless runs compare as "winner CHANGED" on one axis
  and "no verdict change" on the other -- the module's two
  normalizations of the same field disagree. ``mpl run`` never emits
  ``""`` (``None`` or a real name per experiments.py:993/1036), so
  this is malformed/hand-crafted-input territory; severity low. Found
  by the differential fuzz (the oracle initially normalized ``""``
  like ``_held_verdict``); pinned by
  ``test_bug1_empty_string_winner_vs_absent_counts_as_changed``. FIXED:
  the winner extraction now normalizes ``""`` to ``None``, matching
  ``_held_verdict`` -- ``""`` vs absent is identical on every axis.

CLEAN (verified, no bug): every other compared axis behaved exactly
per the rules above across all minimal pairs, the 4x4 evidence
presence matrix, the red-team inputs, and all 800 fuzzed comparisons:

Deliberate/documented behaviors (NOT bugs):
- first-wins verdicts vs last-wins evidence on duplicate candidate
  names (documented in the ``_fixture_outcomes`` docstring; same class
  covers ``boundaries`` last-valid-wins in ``_config_snapshot``).
- cross-stage evidence (old has only search evidence, new only
  held-out): ``fixtures:search``/``fixtures:held_out`` drift entries
  and "absent" fixture diffs are still emitted (documented in the
  ``_fixture_set_drift`` docstring NOTE).
- ``fixtures_flipped`` excludes absent<->pass transitions by design.
- bool ``planned``/``max_new_tokens`` degrade to "unknown"/None via
  the exact ``type(x) is int`` checks.

Deliberately NOT verified here: ``load_results`` hostile inputs
(cycle 64), full ``render_summary`` text (cycle 13), CLI plumbing
(cycles 13/31), the separate diff-backends module.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import tempfile
import unicodedata
import unittest
from pathlib import Path
from typing import Any

from model_protocol_lab import (
    cacheprofile,
    choiceaudit,
    confidence,
    dateline,
    density,
    diffreports,
    distraction,
    experiments,
    fieldorder,
    leading,
    multiturn,
    needle,
    numform,
    placement,
    precision,
    refusals,
    repetition,
    rotation,
    scorerlens,
)
from model_protocol_lab.core import Fixture, Format, Generation, Message

# ---------------------------------------------------------------------------
# Builders: canonical reports for the adversarial pairs.
# ---------------------------------------------------------------------------


def _obs(fixture: str, exact: bool = True, boundary: bool = True) -> dict[str, Any]:
    """Obs."""
    return {"fixture": fixture, "exact_match": exact, "boundary_found": boundary}


def _entry(
    name: str,
    planned: Any = 2,
    passes: Any = 2,
    eligible: Any = True,
    observations: Any = None,
    boundary_ids: Any = None,
    summary: Any = "auto",
) -> dict[str, Any]:
    """Entry."""
    entry: dict[str, Any] = {"format": {"name": name}}
    if summary == "auto":
        entry["summary"] = {
            "planned": planned,
            "exact_and_boundary_passes": passes,
            "aggregate_eligible": eligible,
        }
    elif summary is not None:
        entry["summary"] = summary
    if observations is not None:
        entry["observations"] = observations
    if boundary_ids is not None:
        entry["boundary_token_ids"] = boundary_ids
    return entry


def _base_report() -> dict[str, Any]:
    """Canonical report both sides of a minimal pair start from."""
    return {
        "decision": "select_best",
        "selected_candidate": "alpha",
        "max_new_tokens": 64,
        "backend_fingerprint": "fp-1",
        "search": [
            _entry(
                "alpha",
                observations=[_obs("f1"), _obs("f2", True, False)],
                boundary_ids=[10, 20],
            ),
            _entry(
                "beta/mut",
                passes=1,
                observations=[_obs("f1", False, True), _obs("f2")],
                boundary_ids=[10, 30],
            ),
        ],
        "held_out": {
            "summary": {
                "planned": 1,
                "exact_and_boundary_passes": 1,
                "aggregate_eligible": True,
            },
            "observations": [_obs("h1")],
        },
    }


def _identical_expected(all_fixtures: bool = False) -> dict[str, Any]:
    """Full expected diff for two canonical base reports."""
    cells = [
        {"stage": "held_out", "candidate": "alpha", "fixture": "h1", "old": "pass", "new": "pass"},
        {"stage": "search", "candidate": "alpha", "fixture": "f1", "old": "pass", "new": "pass"},
        {"stage": "search", "candidate": "alpha", "fixture": "f2", "old": "fail", "new": "fail"},
        {
            "stage": "search",
            "candidate": "beta/mut",
            "fixture": "f1",
            "old": "fail",
            "new": "fail",
        },
        {
            "stage": "search",
            "candidate": "beta/mut",
            "fixture": "f2",
            "old": "pass",
            "new": "pass",
        },
    ]
    return {
        "tool": "diff-reports",
        "old": {"source": "old"},
        "new": {"source": "new"},
        "decision": {"old": "select_best", "new": "select_best", "changed": False},
        "winner": {"old": "alpha", "new": "alpha", "changed": False},
        "frontier": {"available": "neither", "changed": False},
        "held_out_floor": {"available": "neither", "changed": False},
        "boundary_audit": {"available": "neither", "changed": False},
        "selection_robustness": {"available": "neither", "changed": False},
        "transfer": {"available": "neither", "changed": False},
        "contamination": {"available": "neither", "changed": False},
        "perturbation": {"available": "neither", "changed": False},
        "token_healing": {"available": "neither", "changed": False},
        "hierarchy": {"available": "neither", "changed": False},
        "prefill": {"available": "neither", "changed": False},
        "portfolio": {"available": "neither", "changed": False},
        "synthhistory": {"available": "neither", "changed": False},
        "batch_invariance": {"available": "neither", "changed": False},
        "position": {"available": "neither", "changed": False},
        "stop_cause": {"available": "neither", "changed": False},
        "ws_canonical": {"available": "neither", "changed": False},
        "structured": {"available": "neither", "changed": False},
        "deliforge": {"available": "neither", "changed": False},
        "fewshot": {"available": "neither", "changed": False},
        "faketurn": {"available": "neither", "changed": False},
        "toolcall": {"available": "neither", "changed": False},
        "verbosity": {"available": "neither", "changed": False},
        "refusals": {"available": "neither", "changed": False},
        "scorerlens": {"available": "neither", "changed": False},
        "aliasing": {"available": "neither", "changed": False},
        "repetition": {"available": "neither", "changed": False},
        "distraction": {"available": "neither", "changed": False},
        "leading": {"available": "neither", "changed": False},
        "numform": {"available": "neither", "changed": False},
        "dateline": {"available": "neither", "changed": False},
        "multiturn": {"available": "neither", "changed": False},
        "needle": {"available": "neither", "changed": False},
        "placement": {"available": "neither", "changed": False},
        "canonical_form": {"available": "neither", "changed": False},
        "instruction_distance": {"available": "neither", "changed": False},
        "history_frame": {"available": "neither", "changed": False},
        "prune_sweep": {"available": "neither", "changed": False},
        "record_notation": {"available": "neither", "changed": False},
        "invisible_chars": {"available": "neither", "changed": False},
        "confusable_skeleton": {"available": "neither", "changed": False},
        "choice_audit": {"available": "neither", "changed": False},
        "precision": {"available": "neither", "changed": False},
        "confidence": {"available": "neither", "changed": False},
        "rotation": {"available": "neither", "changed": False},
        "density": {"available": "neither", "changed": False},
        "field_order": {"available": "neither", "changed": False},
        "cache_profile": {"available": "neither", "changed": False},
        "template_robustness": {"available": "neither", "changed": False},
        "ablation": {"available": "neither", "changed": False},
        "confidentiality": {"available": "neither", "changed": False},
        "answer_collision": {"available": "neither", "changed": False},
        "config_drift": [],
        "candidate_verdicts": [],
        "evidence": {
            "old": True,
            "new": True,
            "skipped_observations": {"old": 0, "new": 0},
        },
        "fixture_diffs": cells if all_fixtures else [],
        "fixture_diffs_include_unchanged": all_fixtures,
        "summary": {
            "decision_changed": False,
            "winner_changed": False,
            "candidates_changed_status": 0,
            "fixtures_flipped": 0,
            "config_drift_warnings": 0,
            "evidence_asymmetric": False,
            "frontier_changed": False,
            "frontier_asymmetric": False,
            "held_out_floor_changed": False,
            "held_out_floor_asymmetric": False,
            "boundary_audit_changed": False,
            "boundary_audit_asymmetric": False,
            "selection_robustness_changed": False,
            "selection_robustness_asymmetric": False,
            "transfer_changed": False,
            "transfer_asymmetric": False,
            "contamination_changed": False,
            "contamination_asymmetric": False,
            "perturbation_changed": False,
            "perturbation_asymmetric": False,
            "healing_changed": False,
            "healing_asymmetric": False,
            "hierarchy_changed": False,
            "hierarchy_asymmetric": False,
            "prefill_changed": False,
            "prefill_asymmetric": False,
            "portfolio_changed": False,
            "portfolio_asymmetric": False,
            "synthhistory_changed": False,
            "synthhistory_asymmetric": False,
            "batch_invariance_changed": False,
            "batch_invariance_asymmetric": False,
            "position_changed": False,
            "position_asymmetric": False,
            "stop_cause_changed": False,
            "stop_cause_asymmetric": False,
            "ws_canonical_changed": False,
            "ws_canonical_asymmetric": False,
            "structured_changed": False,
            "structured_asymmetric": False,
            "deliforge_changed": False,
            "deliforge_asymmetric": False,
            "fewshot_changed": False,
            "fewshot_asymmetric": False,
            "faketurn_changed": False,
            "faketurn_asymmetric": False,
            "toolcall_changed": False,
            "toolcall_asymmetric": False,
            "verbosity_changed": False,
            "verbosity_asymmetric": False,
            "refusals_changed": False,
            "refusals_asymmetric": False,
            "scorerlens_changed": False,
            "scorerlens_asymmetric": False,
            "aliasing_changed": False,
            "aliasing_asymmetric": False,
            "repetition_changed": False,
            "repetition_asymmetric": False,
            "distraction_changed": False,
            "distraction_asymmetric": False,
            "leading_changed": False,
            "leading_asymmetric": False,
            "numform_changed": False,
            "numform_asymmetric": False,
            "dateline_changed": False,
            "dateline_asymmetric": False,
            "multiturn_changed": False,
            "multiturn_asymmetric": False,
            "needle_changed": False,
            "needle_asymmetric": False,
            "placement_changed": False,
            "placement_asymmetric": False,
            "canonical_form_changed": False,
            "canonical_form_asymmetric": False,
            "instruction_distance_changed": False,
            "instruction_distance_asymmetric": False,
            "history_frame_changed": False,
            "history_frame_asymmetric": False,
            "prune_sweep_changed": False,
            "prune_sweep_asymmetric": False,
            "record_notation_changed": False,
            "record_notation_asymmetric": False,
            "invisible_chars_changed": False,
            "invisible_chars_asymmetric": False,
            "confusable_skeleton_changed": False,
            "confusable_skeleton_asymmetric": False,
            "choice_audit_changed": False,
            "choice_audit_asymmetric": False,
            "precision_changed": False,
            "precision_asymmetric": False,
            "confidence_changed": False,
            "confidence_asymmetric": False,
            "rotation_changed": False,
            "rotation_asymmetric": False,
            "density_changed": False,
            "density_asymmetric": False,
            "field_order_changed": False,
            "field_order_asymmetric": False,
            "cache_profile_changed": False,
            "cache_profile_asymmetric": False,
            "template_robustness_changed": False,
            "template_robustness_asymmetric": False,
            "ablation_changed": False,
            "ablation_asymmetric": False,
            "confidentiality_changed": False,
            "confidentiality_asymmetric": False,
            "answer_collision_changed": False,
            "answer_collision_asymmetric": False,
            "identical": True,
        },
    }


# ---------------------------------------------------------------------------
# Independent oracle, written from the COMPARISON RULES above.
# Structure and control flow deliberately differ from the implementation
# (profiles-first, comprehensions instead of nested accumulation loops).
# ---------------------------------------------------------------------------


def _o_verdict(summary: Any) -> str:
    """O verdict."""
    if type(summary) is not dict:
        return "unknown"
    planned = summary.get("planned")
    passes = summary.get("exact_and_boundary_passes")
    eligible = summary.get("aggregate_eligible")
    if type(planned) is not int or type(passes) is not int or type(eligible) is not bool:
        return "unknown"
    if not eligible or planned <= 0:
        return "fail"
    return "pass" if passes == planned else "fail"


def _o_name(entry: Any) -> str | None:
    """O name."""
    if type(entry) is not dict:
        return None
    fmt = entry.get("format")
    name = fmt.get("name") if type(fmt) is dict else None
    return name if type(name) is str and name else None


def _o_base_family(name: str) -> str:
    """O base family."""
    return name.split("/", 1)[0]


def _o_outcomes(observations: Any) -> tuple[dict[str, str], int]:
    """O outcomes."""
    found: dict[str, str] = {}
    skipped = 0
    if type(observations) is not list:
        return found, 0
    for row in observations:
        valid = (
            type(row) is dict
            and type(row.get("fixture")) is str
            and row["fixture"]
            and type(row.get("exact_match")) is bool
            and type(row.get("boundary_found")) is bool
        )
        if valid:
            found[row["fixture"]] = (
                "pass" if (row["exact_match"] and row["boundary_found"]) else "fail"
            )
        else:
            skipped += 1
    return found, skipped


def _o_search_profile(result: dict[str, Any]) -> tuple[dict[str, str], dict[str, int | None]]:
    """O search profile."""
    verdicts: dict[str, str] = {}
    planned: dict[str, int | None] = {}
    for entry in result.get("search") or []:
        name = _o_name(entry)
        if name is None or name in verdicts:
            continue
        summary = entry.get("summary") if type(entry) is dict else None
        verdicts[name] = _o_verdict(summary)
        raw = summary.get("planned") if type(summary) is dict else None
        planned[name] = raw if type(raw) is int else None
    return verdicts, planned


def _o_held_profile(result: dict[str, Any]) -> tuple[str | None, str]:
    """O held profile."""
    winner = result.get("selected_candidate")
    if type(winner) is not str or not winner:
        return None, "absent"
    held = result.get("held_out")
    summary = held.get("summary") if type(held) is dict else None
    return winner, _o_verdict(summary)


def _o_evidence(result: dict[str, Any], stage: str) -> tuple[dict[str, dict[str, str]], int, bool]:
    """O evidence."""
    per_candidate: dict[str, dict[str, str]] = {}
    skipped = 0
    if stage == "search":
        for entry in result.get("search") or []:
            name = _o_name(entry)
            if name is None:
                continue
            observations = entry.get("observations") if type(entry) is dict else None
            outcomes, skipped_here = _o_outcomes(observations)
            skipped += skipped_here
            if type(observations) is list and outcomes:
                per_candidate[name] = outcomes
    else:
        winner = result.get("selected_candidate")
        held = result.get("held_out")
        observations = held.get("observations") if type(held) is dict else None
        outcomes, skipped = _o_outcomes(observations)
        if type(winner) is str and winner and type(observations) is list and outcomes:
            per_candidate[winner] = outcomes
    return per_candidate, skipped, bool(per_candidate)


def _o_snapshot(result: dict[str, Any]) -> dict[str, Any]:
    """O snapshot."""
    names = sorted({_o_name(e) for e in result.get("search") or []} - {None})
    bounds: dict[str, list[int]] = {}
    for entry in result.get("search") or []:
        name = _o_name(entry)
        ids = entry.get("boundary_token_ids") if type(entry) is dict else None
        if name is not None and type(ids) is list and all(type(token) is int for token in ids):
            bounds[name] = list(ids)
    max_new = result.get("max_new_tokens")
    fingerprint = result.get("backend_fingerprint")
    return {
        "families": sorted({_o_base_family(n) for n in names}),
        "candidates": names,
        "max_new_tokens": max_new if type(max_new) is int else None,
        "boundaries": {n: bounds[n] for n in sorted(bounds)},
        "backend_fingerprint": fingerprint if type(fingerprint) is str else None,
    }


def _o_drift(old_snap: dict[str, Any], new_snap: dict[str, Any]) -> list[dict[str, Any]]:
    """O drift."""
    fields = ("families", "candidates", "max_new_tokens", "boundaries", "backend_fingerprint")
    return [
        {"field": field, "old": old_snap[field], "new": new_snap[field]}
        for field in fields
        if old_snap[field] != new_snap[field]
    ]


def _o_frontier_valid(value: Any) -> bool:
    """Oracle-side availability check, mirroring the implementation's rule."""
    if type(value) is not dict:
        return False
    candidates = value.get("candidates")
    if type(candidates) is not dict or not candidates:
        return False
    max_tokens = value.get("max_new_tokens")
    if type(max_tokens) is not int or max_tokens <= 0:
        return False
    for data in candidates.values():
        if type(data) is not dict or len(data.get("fixture_ids") or []) <= 0:
            continue
        for point in data.get("pass_steps") or []:
            if (
                type(point) is list
                and len(point) == 2
                and type(point[0]) is int
                and type(point[1]) is int
                and 1 <= point[0] <= max_tokens
            ):
                return True
    return False


def _o_frontier(old: Any, new: Any) -> tuple[dict[str, Any], bool, bool]:
    """Oracle frontier section: availability from the rules.

    Span-level comparison (decision/winner/leader flips) is covered by the
    dedicated cycle-125 audit with hand-derived spans and its own mutation
    gate; the differential fuzz inputs here never carry budget_frontier
    payloads, so the oracle asserts the availability contract and
    changed=False, which the implementation guarantees whenever the
    comparison is one-sided or absent.
    """
    old_ok = _o_frontier_valid(old)
    new_ok = _o_frontier_valid(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    return {"available": available, "changed": False}, False, available not in ("both", "neither")


def _o_held_out_floor_valid(value: Any) -> bool:
    """Oracle-side availability check, mirroring the implementation's rule."""
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
    floor = value.get("safe_budget_floor")
    return not (floor is not None and type(floor) is not int)


def _o_held_out_floor(old: Any, new: Any) -> tuple[dict[str, Any], bool, bool]:
    """Oracle held-out-floor section: availability and field comparison."""
    old_ok = _o_held_out_floor_valid(old)
    new_ok = _o_held_out_floor_valid(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    diff: dict[str, Any] = {"available": available, "changed": False}
    asymmetric = available not in ("both", "neither")
    if available != "both":
        return diff, False, asymmetric
    floor_changed = old.get("safe_budget_floor") != new.get("safe_budget_floor")
    robust_changed = old.get("budget_robust") != new.get("budget_robust")
    diff["safe_budget_floor"] = {
        "old": old.get("safe_budget_floor"),
        "new": new.get("safe_budget_floor"),
        "changed": floor_changed,
    }
    diff["budget_robust"] = {
        "old": old.get("budget_robust"),
        "new": new.get("budget_robust"),
        "changed": robust_changed,
    }
    changed = bool(floor_changed or robust_changed)
    diff["changed"] = changed
    return diff, changed, asymmetric


def _o_boundary_audit_valid(value: Any) -> bool:
    """Oracle-side availability check, mirroring the implementation's rule."""
    if type(value) is not dict or value.get("available", True) is False:
        return False
    if type(value.get("per_family")) is not dict:
        return False
    rate = value.get("overall_rate")
    return rate is None or type(rate) in (int, float)


def _o_boundary_audit(old: Any, new: Any) -> tuple[dict[str, Any], bool, bool]:
    """Oracle boundary-audit section: availability and rate comparison."""
    old_ok = _o_boundary_audit_valid(old)
    new_ok = _o_boundary_audit_valid(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    diff: dict[str, Any] = {"available": available, "changed": False}
    asymmetric = available not in ("both", "neither")
    if available != "both":
        return diff, False, asymmetric
    old_rate = old.get("overall_rate")
    new_rate = new.get("overall_rate")
    old_families = old.get("per_family") or {}
    new_families = new.get("per_family") or {}
    family_drift = []
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
    rate_changed = old_rate != new_rate
    diff["overall_rate"] = {"old": old_rate, "new": new_rate, "changed": rate_changed}
    diff["family_drift"] = family_drift
    changed = bool(rate_changed or family_drift)
    diff["changed"] = changed
    return diff, changed, asymmetric


def _o_selection_robustness_valid(value: Any) -> bool:
    """Oracle-side availability check, mirroring the implementation's rule."""
    ps = value.get("per_semantic") if type(value) is dict else None
    return (
        type(value) is dict
        and value.get("available") is True
        and type(ps) is dict
        and all(type(k) is str for k in ps)
        and type(value.get("flips")) is list
    )


def _o_selection_robustness(old: Any, new: Any) -> tuple[dict[str, Any], bool, bool]:
    """Oracle selection-robustness section: verdict drift and flip changes."""
    old_ok = _o_selection_robustness_valid(old)
    new_ok = _o_selection_robustness_valid(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    diff: dict[str, Any] = {"available": available, "changed": False}
    asymmetric = available not in ("both", "neither")
    if available != "both":
        return diff, False, asymmetric
    old_ps = old.get("per_semantic") or {}
    new_ps = new.get("per_semantic") or {}
    verdict_drift = []
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
    flips_changed = (old.get("flips") or []) != (new.get("flips") or [])
    diff["verdict_drift"] = verdict_drift
    diff["flips_changed"] = flips_changed
    changed = bool(verdict_drift or flips_changed)
    diff["changed"] = changed
    return diff, changed, asymmetric


def _o_transfer_valid(value: Any) -> bool:
    """Oracle-side availability check, mirroring the implementation's rule."""
    pc = value.get("per_checkpoint") if type(value) is dict else None
    mx = value.get("matrix") if type(value) is dict else None
    return (
        type(value) is dict
        and value.get("available") is True
        and type(pc) is dict
        and all(type(k) is str for k in pc)
        and type(mx) is dict
        and all(type(k) is str for k in mx)
        and all(type(v) is dict and all(type(k) is str for k in v) for v in mx.values())
    )


def _o_contamination_valid(value: Any) -> bool:
    """Oracle-side availability check, mirroring the implementation's rule."""
    pairs = value.get("pairs") if type(value) is dict else None
    return (
        type(value) is dict
        and type(value.get("worst_tier")) is str
        and type(pairs) is list
        and all(type(p) is dict for p in pairs)
    )


def _o_perturbation_valid(value: Any) -> bool:
    """Oracle-side availability check, mirroring the implementation's rule."""
    pv = value.get("per_variant") if type(value) is dict else None
    st = value.get("stability") if type(value) is dict else None
    return (
        type(value) is dict
        and value.get("available") is True
        and type(pv) is dict
        and all(type(k) is str for k in pv)
        and all(type(v) is dict for v in pv.values())
        and type(st) is dict
    )


def _o_healing_valid(value: Any) -> bool:
    """Oracle-side availability check, mirroring the implementation's rule."""
    pc = value.get("per_candidate") if type(value) is dict else None
    return (
        type(value) is dict
        and value.get("available") is True
        and type(value.get("flipped")) is bool
        and type(pc) is dict
        and all(type(k) is str and type(v) is dict for k, v in pc.items())
        and all(
            (e.get("native") is None or type(e.get("native")) is bool)
            and (e.get("healed") is None or type(e.get("healed")) is bool)
            for e in pc.values()
        )
    )


def _o_healing(old: Any, new: Any) -> tuple[dict[str, Any], bool, bool]:
    """Oracle token-healing section: flip verdict, candidate, and set drift."""
    old_ok = _o_healing_valid(old)
    new_ok = _o_healing_valid(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    diff: dict[str, Any] = {"available": available, "changed": False}
    asymmetric = available not in ("both", "neither")
    if available != "both":
        return diff, False, asymmetric

    def _triplet(payload: Any) -> tuple[Any, Any, Any]:
        """Triplet."""
        if type(payload) is not dict:
            return (None, None, None)
        return (
            payload.get("flipped"),
            payload.get("healed_decision"),
            payload.get("healed_winner"),
        )

    old_t, new_t = _triplet(old), _triplet(new)
    flip_changed = old_t != new_t
    diff["flip"] = {
        "old": {
            "flipped": old_t[0],
            "healed_decision": old_t[1],
            "healed_winner": old_t[2],
        },
        "new": {
            "flipped": new_t[0],
            "healed_decision": new_t[1],
            "healed_winner": new_t[2],
        },
        "changed": flip_changed,
    }
    old_pc, new_pc = old.get("per_candidate") or {}, new.get("per_candidate") or {}
    old_names, new_names = sorted(old_pc), sorted(new_pc)
    candidates_added = [c for c in new_names if c not in old_pc]
    candidates_removed = [c for c in old_names if c not in new_pc]
    candidate_drift = [
        {
            "candidate": cand,
            "old_native": (old_pc.get(cand) or {}).get("native"),
            "old_healed": (old_pc.get(cand) or {}).get("healed"),
            "new_native": (new_pc.get(cand) or {}).get("native"),
            "new_healed": (new_pc.get(cand) or {}).get("healed"),
        }
        for cand in sorted(set(old_names) & set(new_names))
        if (
            ((old_pc.get(cand) or {}).get("native"), (old_pc.get(cand) or {}).get("healed"))
            != (
                (new_pc.get(cand) or {}).get("native"),
                (new_pc.get(cand) or {}).get("healed"),
            )
        )
    ]
    old_fals = old.get("falsification") or {}
    new_fals = new.get("falsification") or {}
    old_acc = (old_fals.get("n_holds"), old_fals.get("n_comparable"))
    new_acc = (new_fals.get("n_holds"), new_fals.get("n_comparable"))
    falsification_changed = old_acc != new_acc
    diff["falsification"] = {
        "old": {"n_holds": old_acc[0], "n_comparable": old_acc[1]},
        "new": {"n_holds": new_acc[0], "n_comparable": new_acc[1]},
        "changed": falsification_changed,
    }
    diff["candidate_drift"] = candidate_drift
    diff["candidates_added"] = candidates_added
    diff["candidates_removed"] = candidates_removed
    changed = bool(
        flip_changed
        or candidate_drift
        or candidates_added
        or candidates_removed
        or falsification_changed
    )
    diff["changed"] = changed
    return diff, changed, asymmetric


def _o_hierarchy_valid(value: Any) -> bool:
    """Oracle-side availability check, mirroring the implementation's rule."""
    if type(value) is not dict or value.get("available") is not True:
        return False
    positions = value.get("positions")
    if type(positions) is not list or not positions or not all(type(p) is str for p in positions):
        return False
    per_condition = value.get("per_condition")
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
    groups = value.get("group_adherence")
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
    return type(value.get("flips")) is list


def _o_hierarchy(old: Any, new: Any) -> tuple[dict[str, Any], bool, bool]:
    """Oracle hierarchy section: condition triplets, flips, rates, groups."""
    old_ok = _o_hierarchy_valid(old)
    new_ok = _o_hierarchy_valid(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    diff: dict[str, Any] = {"available": available, "changed": False}
    asymmetric = available not in ("both", "neither")
    if available != "both":
        return diff, False, asymmetric
    old_positions = old.get("positions") or []
    new_positions = new.get("positions") or []
    positions_added = [p for p in new_positions if p not in old_positions]
    positions_removed = [p for p in old_positions if p not in new_positions]
    condition_changed = {}
    for position in sorted(set(old_positions) & set(new_positions)):
        old_cond = (old.get("per_condition") or {}).get(position) or {}
        new_cond = (new.get("per_condition") or {}).get(position) or {}
        old_triplet = (old_cond.get("decision"), old_cond.get("winner"))
        new_triplet = (new_cond.get("decision"), new_cond.get("winner"))
        condition_changed[position] = {
            "old_decision": old_triplet[0],
            "old_winner": old_triplet[1],
            "new_decision": new_triplet[0],
            "new_winner": new_triplet[1],
            "changed": old_triplet != new_triplet,
        }

    def _flip_key(flips: Any) -> list[tuple[Any, ...]]:
        """Flip key."""
        if type(flips) is not list:
            return []
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
        """Candidate rates."""
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
    candidate_drift = []
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
        """Group means."""
        groups = payload.get("group_adherence") or {}
        return {
            label: (group or {}).get("mean_system_rate")
            for label, group in groups.items()
            if type(label) is str
        }

    old_groups = _group_means(old)
    new_groups = _group_means(new)
    group_changed = old_groups != new_groups
    diff["positions_added"] = positions_added
    diff["positions_removed"] = positions_removed
    diff["condition_changed"] = condition_changed
    diff["flips_changed"] = flips_changed
    diff["candidate_drift"] = candidate_drift
    diff["candidates_added"] = candidates_added
    diff["candidates_removed"] = candidates_removed
    diff["group_rates"] = {
        "old": old_groups,
        "new": new_groups,
        "changed": group_changed,
    }
    changed = bool(
        positions_added
        or positions_removed
        or any(entry["changed"] for entry in condition_changed.values())
        or flips_changed
        or candidate_drift
        or candidates_added
        or candidates_removed
        or group_changed
    )
    diff["changed"] = changed
    return diff, changed, asymmetric


def _o_position_valid(value: Any) -> bool:
    """Oracle-side availability check, mirroring the implementation's rule."""
    if type(value) is not dict or value.get("available") is not True:
        return False
    conditions = value.get("conditions")
    if (
        type(conditions) is not list
        or not conditions
        or not all(type(c) is str for c in conditions)
    ):
        return False
    per_condition = value.get("per_condition")
    if type(per_condition) is not dict or set(per_condition) != set(conditions):
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
            rate = entry.get("correct_rate")
            if rate is not None and type(rate) not in (int, float):
                return False
            if entry.get("pass") is not None and type(entry.get("pass")) is not bool:
                return False
    primacy = value.get("primacy_index")
    if type(primacy) is not dict or not all(type(k) is str for k in primacy):
        return False
    for index_value in primacy.values():
        if index_value is not None and type(index_value) not in (int, float):
            return False
    effect = value.get("condition_effect")
    if effect is not None:
        if type(effect) is not dict:
            return False
        mean = effect.get("mean_index")
        if mean is not None and type(mean) not in (int, float):
            return False
        for key in ("n_candidates", "n_positive", "n_negative"):
            if type(effect.get(key)) is not int:
                return False
    return type(value.get("flips")) is list


def _o_position(old: Any, new: Any) -> tuple[dict[str, Any], bool, bool]:
    """Oracle position section: condition pairs, flips, rates, index drift."""
    old_ok = _o_position_valid(old)
    new_ok = _o_position_valid(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    diff: dict[str, Any] = {"available": available, "changed": False}
    asymmetric = available not in ("both", "neither")
    if available != "both":
        return diff, False, asymmetric
    old_conditions = old.get("conditions") or []
    new_conditions = new.get("conditions") or []
    conditions_added = [c for c in new_conditions if c not in old_conditions]
    conditions_removed = [c for c in old_conditions if c not in new_conditions]
    condition_changed = {}
    for condition in sorted(set(old_conditions) & set(new_conditions)):
        old_cond = (old.get("per_condition") or {}).get(condition) or {}
        new_cond = (new.get("per_condition") or {}).get(condition) or {}
        old_pair = (old_cond.get("decision"), old_cond.get("winner"))
        new_pair = (new_cond.get("decision"), new_cond.get("winner"))
        condition_changed[condition] = {
            "old_decision": old_pair[0],
            "old_winner": old_pair[1],
            "new_decision": new_pair[0],
            "new_winner": new_pair[1],
            "changed": old_pair != new_pair,
        }

    def _flip_key(flips: Any) -> list[tuple[Any, ...]]:
        """Flip key."""
        if type(flips) is not list:
            return []
        keys = []
        for entry in flips:
            if type(entry) is not dict:
                continue
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

    flips_changed = _flip_key(old.get("flips")) != _flip_key(new.get("flips"))

    def _candidate_rates(payload: Any) -> dict[str, dict[str, Any]]:
        """Candidate rates."""
        rates: dict[str, dict[str, Any]] = {}
        per_condition = payload.get("per_condition") or {}
        for condition in sorted(per_condition):
            candidates = (per_condition.get(condition) or {}).get("per_candidate") or {}
            for name, entry in candidates.items():
                entry = entry or {}
                rates.setdefault(name, {})[condition] = entry.get("correct_rate")
        return rates

    old_rates = _candidate_rates(old)
    new_rates = _candidate_rates(new)
    candidates_added = sorted(set(new_rates) - set(old_rates))
    candidates_removed = sorted(set(old_rates) - set(new_rates))
    candidate_drift = []
    for name in sorted(set(old_rates) & set(new_rates)):
        for condition in sorted(set(old_rates[name]) & set(new_rates[name])):
            old_rate = old_rates[name][condition]
            new_rate = new_rates[name][condition]
            if old_rate != new_rate:
                candidate_drift.append(
                    {
                        "candidate": name,
                        "condition": condition,
                        "old_correct_rate": old_rate,
                        "new_correct_rate": new_rate,
                    }
                )

    def _index_map(payload: Any) -> dict[str, Any]:
        """Index map."""
        primacy = payload.get("primacy_index") or {}
        return {k: v for k, v in primacy.items() if type(k) is str}

    old_indexes = _index_map(old)
    new_indexes = _index_map(new)
    index_drift = []
    for name in sorted(set(old_indexes) & set(new_indexes)):
        if old_indexes[name] != new_indexes[name]:
            index_drift.append(
                {
                    "candidate": name,
                    "old_primacy_index": old_indexes[name],
                    "new_primacy_index": new_indexes[name],
                }
            )

    def _mean_index(payload: Any) -> Any:
        """Mean index."""
        effect = payload.get("condition_effect") or {}
        return effect.get("mean_index")

    old_mean = _mean_index(old)
    new_mean = _mean_index(new)
    mean_index_changed = old_mean != new_mean
    diff["conditions_added"] = conditions_added
    diff["conditions_removed"] = conditions_removed
    diff["condition_changed"] = condition_changed
    diff["flips_changed"] = flips_changed
    diff["candidate_drift"] = candidate_drift
    diff["candidates_added"] = candidates_added
    diff["candidates_removed"] = candidates_removed
    diff["index_drift"] = index_drift
    diff["mean_index"] = {"old": old_mean, "new": new_mean, "changed": mean_index_changed}
    changed = bool(
        conditions_added
        or conditions_removed
        or any(entry["changed"] for entry in condition_changed.values())
        or flips_changed
        or candidate_drift
        or candidates_added
        or candidates_removed
        or index_drift
        or mean_index_changed
    )
    diff["changed"] = changed
    return diff, changed, asymmetric


def _o_verbosity_valid(value: Any) -> bool:
    """Oracle verbosity validator: mirrors diffreports._valid_verbosity."""
    if type(value) is not dict or value.get("available") is not True:
        return False
    n_fixtures = value.get("n_fixtures")
    if type(n_fixtures) is not int or n_fixtures < 0:
        return False
    n_brevity = value.get("n_brevity_fixtures")
    if type(n_brevity) is not int or n_brevity < 0 or n_brevity > n_fixtures:
        return False
    per_candidate = value.get("per_candidate")
    if type(per_candidate) is not dict or not all(type(k) is str for k in per_candidate):
        return False
    for entry in per_candidate.values():
        if entry is None:
            continue
        if type(entry) is not dict:
            return False
        for key in ("n", "n_evaluable"):
            count = entry.get(key)
            if type(count) is not int or count < 0:
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
            stat = entry.get(key)
            if stat is not None and not (type(stat) in (int, float) and stat >= 0):
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
        if (
            entry["n_evaluable"] > 0
            and type(p25) in (int, float)
            and type(median) in (int, float)
            and type(p75) in (int, float)
            and not p25 <= median <= p75
        ):
            return False
    for key in ("most_verbose", "least_verbose", "capped_candidates", "excluded_candidates"):
        value_list = value.get(key)
        if type(value_list) is not list or not all(type(c) is str for c in value_list):
            return False
    return True


def _o_verbosity(old: Any, new: Any) -> tuple[dict[str, Any], bool, bool]:
    """Oracle verbosity section: median/iqr/capped-share drift.

    Independent reimplementation: dict-comprehension views rather
    than the implementation's accumulation loops, with the drift
    buckets filled in the implementation's per-candidate field order.
    """
    old_ok = _o_verbosity_valid(old)
    new_ok = _o_verbosity_valid(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    diff: dict[str, Any] = {"available": available, "changed": False}
    asymmetric = available not in ("both", "neither")
    if available != "both":
        return diff, False, asymmetric

    def _view(payload: Any) -> dict[str, dict[str, Any]]:
        # Mirrors the implementation: a None cell (candidate missing
        # from the stage) counts as excluded, exactly like the
        # profile's excluded_candidates semantics.
        """View."""
        out: dict[str, dict[str, Any]] = {}
        for name, entry in (payload.get("per_candidate") or {}).items():
            if entry is None:
                out[name] = {
                    "median_excess": None,
                    "iqr_excess": None,
                    "capped_share": None,
                    "excluded": True,
                }
            else:
                out[name] = {
                    "median_excess": entry.get("median_excess"),
                    "iqr_excess": entry.get("iqr_excess"),
                    "capped_share": entry.get("capped_share"),
                    "excluded": name in set(payload.get("excluded_candidates") or []),
                }
        return out

    old_view = _view(old)
    new_view = _view(new)
    candidates_added = sorted(set(new_view) - set(old_view))
    candidates_removed = sorted(set(old_view) - set(new_view))
    # Same per-candidate field order as the implementation's loop.
    median_drift: list[dict[str, Any]] = []
    iqr_drift: list[dict[str, Any]] = []
    capped_share_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_view) & set(new_view)):
        for field, bucket in (
            ("median_excess", median_drift),
            ("iqr_excess", iqr_drift),
            ("capped_share", capped_share_drift),
        ):
            if old_view[name][field] != new_view[name][field]:
                bucket.append(
                    {
                        "candidate": name,
                        "old": old_view[name][field],
                        "new": new_view[name][field],
                    }
                )
    old_excluded = {name for name, v in old_view.items() if v["excluded"]}
    new_excluded = {name for name, v in new_view.items() if v["excluded"]}
    excluded_added = sorted(new_excluded - old_excluded)
    excluded_removed = sorted(old_excluded - new_excluded)
    diff["median_drift"] = median_drift
    diff["iqr_drift"] = iqr_drift
    diff["capped_share_drift"] = capped_share_drift
    diff["candidates_added"] = candidates_added
    diff["candidates_removed"] = candidates_removed
    diff["excluded_added"] = excluded_added
    diff["excluded_removed"] = excluded_removed
    changed = bool(
        median_drift
        or iqr_drift
        or capped_share_drift
        or candidates_added
        or candidates_removed
        or excluded_added
        or excluded_removed
    )
    diff["changed"] = changed
    return diff, changed, asymmetric


def _o_refusals_valid(value: Any) -> bool:
    """Oracle refusals validator: mirrors diffreports._valid_refusals."""
    if type(value) is not dict or value.get("available") is not True:
        return False
    method = value.get("method")
    if type(method) is not dict:
        return False
    if type(method.get("detector")) is not str or not method.get("detector"):
        return False
    digest = method.get("openers_sha256")
    if type(digest) is not str or not digest:
        return False
    openers = method.get("openers")
    if (
        type(openers) is not list
        or not openers
        or not all(type(opener) is str and opener for opener in openers)
    ):
        return False
    if digest != hashlib.sha256("\n".join(openers).encode("utf-8")).hexdigest():
        return False
    if type(value.get("evaluable_mismatch")) is not bool:
        return False
    candidates = value.get("candidates")
    if type(candidates) is not list:
        return False
    if type(value.get("n_candidates")) is not int or value.get("n_candidates") != len(candidates):
        return False
    n_available = 0
    seen_names: set = set()
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
        counts = {}
        for key in (
            "n_observations",
            "n_completed",
            "n_boundary_found",
            "n_suspected",
            "n_suspected_passed",
            "n_suspected_failed",
        ):
            count = entry.get(key)
            if type(count) is not int or count < 0:
                return False
            counts[key] = count
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
    if type(value.get("n_candidates_available")) is not int:
        return False
    return value.get("n_candidates_available") == n_available


def _o_refusals(old: Any, new: Any) -> tuple[dict[str, Any], bool, bool]:
    """Oracle refusals section: rate/hit drift, protocol drift.

    Independent reimplementation: set-based hit comparison and
    comprehension-built drift lists rather than the implementation's
    accumulation loops, in the implementation's output order.
    """
    old_ok = _o_refusals_valid(old)
    new_ok = _o_refusals_valid(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    diff: dict[str, Any] = {"available": available, "changed": False}
    asymmetric = available not in ("both", "neither")
    if available != "both":
        return diff, False, asymmetric
    protocol_drift = (old["method"].get("detector"), old["method"].get("openers_sha256")) != (
        new["method"].get("detector"),
        new["method"].get("openers_sha256"),
    )

    def _view(payload: Any) -> dict[str, dict[str, Any]]:
        """View."""
        out: dict[str, dict[str, Any]] = {}
        for entry in payload.get("candidates") or []:
            name = entry.get("candidate")
            if type(name) is not str:
                continue
            if entry.get("available") is not True:
                out[name] = {"excluded": True}
                continue
            out[name] = {
                "excluded": False,
                "rate": entry.get("suspected_rate"),
                "n": entry.get("n_completed"),
                "hits": {
                    (hit.get("fixture"), hit.get("opener"), hit.get("passed"))
                    for hit in entry.get("hits") or []
                },
            }
        return out

    old_view = _view(old)
    new_view = _view(new)
    candidates_added = sorted(set(new_view) - set(old_view))
    candidates_removed = sorted(set(old_view) - set(new_view))

    def _hit_rows(names: Any, source: Any, other: Any) -> list[dict[str, Any]]:
        """Hit rows."""
        rows: list[dict[str, Any]] = []
        for cand in names:
            delta = source[cand]["hits"] - other[cand]["hits"]
            for fixture, opener, passed in sorted(
                delta, key=lambda h: (str(h[0]), str(h[1]), str(h[2]))
            ):
                rows.append(
                    {"candidate": cand, "fixture": fixture, "opener": opener, "passed": passed}
                )
        return rows

    shared = sorted(set(old_view) & set(new_view))
    comparable = [n for n in shared if not old_view[n]["excluded"] and not new_view[n]["excluded"]]
    rate_drift = [
        {
            "candidate": n,
            "old_rate": old_view[n]["rate"],
            "new_rate": new_view[n]["rate"],
            "old_n": old_view[n]["n"],
            "new_n": new_view[n]["n"],
        }
        for n in comparable
        if (old_view[n]["rate"], old_view[n]["n"]) != (new_view[n]["rate"], new_view[n]["n"])
    ]
    hits_added = _hit_rows(comparable, new_view, old_view)
    hits_removed = _hit_rows(comparable, old_view, new_view)
    old_excluded = {n for n, v in old_view.items() if v["excluded"]}
    new_excluded = {n for n, v in new_view.items() if v["excluded"]}
    excluded_added = sorted(new_excluded - old_excluded)
    excluded_removed = sorted(old_excluded - new_excluded)
    mismatch_changed = old.get("evaluable_mismatch") != new.get("evaluable_mismatch")
    diff["rate_drift"] = rate_drift
    diff["hits_added"] = hits_added
    diff["hits_removed"] = hits_removed
    diff["candidates_added"] = candidates_added
    diff["candidates_removed"] = candidates_removed
    diff["excluded_added"] = excluded_added
    diff["excluded_removed"] = excluded_removed
    diff["protocol_drift"] = protocol_drift
    diff["mismatch_changed"] = mismatch_changed
    changed = bool(
        rate_drift
        or hits_added
        or hits_removed
        or candidates_added
        or candidates_removed
        or excluded_added
        or excluded_removed
        or mismatch_changed
        or protocol_drift
    )
    diff["changed"] = changed
    return diff, changed, asymmetric


def _o_scorerlens_valid(value: Any) -> bool:
    """Oracle scorerlens validator: mirrors diffreports._valid_scorerlens."""
    if type(value) is not dict or value.get("available") is not True:
        return False
    method = value.get("method")
    if type(method) is not dict:
        return False
    protocol = method.get("protocol")
    if type(protocol) is not str or not protocol:
        return False
    ladder = method.get("ladder")
    if type(ladder) is not dict or ladder.get("protocol") != protocol:
        return False
    canonical = json.dumps(ladder, sort_keys=True, separators=(",", ":"))
    if method.get("ladder_sha256") != hashlib.sha256(canonical.encode("utf-8")).hexdigest():
        return False
    fixture_sha = value.get("fixture_set_sha256")
    if (
        type(fixture_sha) is not str
        or len(fixture_sha) != 64
        or any(c not in "0123456789abcdef" for c in fixture_sha)
    ):
        return False
    if type(value.get("evaluable_mismatch")) is not bool:
        return False
    native_winner = value.get("native_winner")
    if native_winner is not None and type(native_winner) is not str:
        return False
    candidates = value.get("candidates")
    if type(candidates) is not list or not candidates:
        return False
    if type(value.get("n_candidates")) is not int or value.get("n_candidates") != len(candidates):
        return False
    n_available = 0
    seen_names: set = set()
    evaluable_sets: list = []
    available_entries: list = []
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
        for record in records:
            if type(record) is not dict:
                return False
            if type(record.get("fixture")) is not str or not record.get("fixture"):
                return False
            if any(
                type(record.get(flag)) is not bool
                for flag in ("completed", "boundary_found", "exact_match", "excluded")
            ):
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
            if any(
                type(record.get(flag)) is not bool
                for flag in ("expected_r3_altered", "newline_differs")
            ):
                return False
        if sum(1 for r in records if r["completed"] is True) != n_completed_field:
            return False
        evaluable = [r for r in records if r["excluded"] is not True]
        evaluable_sets.append(frozenset(r["fixture"] for r in evaluable))

        def _matches(r: Any, rung: int) -> bool:
            """Matches."""
            return r["exact_match"] is True or (
                r["first_flip_rung"] is not None and r["first_flip_rung"] <= rung
            )

        match_counts = {str(k): sum(1 for r in evaluable if _matches(r, k)) for k in range(5)}
        verdict_counts = {
            str(k): sum(
                1
                for r in evaluable
                if r["completed"] is True and r["boundary_found"] is True and _matches(r, k)
            )
            for k in range(5)
        }
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
        if entry["eligible"] is not True:
            if note != "candidate_not_eligible":
                return False
        elif n_failures == 0:
            if note != "no_native_failures":
                return False
        elif share != n_recovered / n_failures:
            return False
        if share is not None and not (type(share) in (int, float) and 0.0 <= share <= 1.0):
            return False
    if type(value.get("n_candidates_available")) is not int:
        return False
    if value.get("n_candidates_available") != n_available:
        return False
    if value.get("evaluable_mismatch") != (len(set(evaluable_sets)) > 1):
        return False
    winner_detail = value.get("winner_detail")
    winners = value.get("winner_by_rung")
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
        if n_available != len(candidates):
            expected_detail = "candidate_unavailable"
        elif any(e["eligible"] is not True for e in available_entries):
            expected_detail = "candidate_not_eligible"
        elif value.get("evaluable_mismatch") is True:
            expected_detail = "evaluable_mismatch"
        else:
            return False
        return winner_detail == expected_detail
    if type(winners) is not dict or set(winners) != {"0", "1", "2", "3", "4"}:
        return False
    if n_available != len(candidates):
        return False
    if any(e["eligible"] is not True for e in available_entries):
        return False
    if value.get("evaluable_mismatch") is True:
        return False
    for rung in ("0", "1", "2", "3", "4"):
        rung_winners = [
            e["candidate"]
            for e in available_entries
            if e["verdict_counts"][rung] == e["n_evaluable"] and e["n_evaluable"] > 0
        ]
        if winners.get(rung) != (rung_winners[0] if len(rung_winners) == 1 else None):
            return False
    return True


def _o_scorerlens(old: Any, new: Any) -> tuple[dict[str, Any], bool, bool]:
    """Oracle scorerlens section: winner/share/flip drift, protocol drift.

    Independent reimplementation: view-based comparison and
    comprehension-built drift lists, in the implementation's
    output order.
    """
    old_ok = _o_scorerlens_valid(old)
    new_ok = _o_scorerlens_valid(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    diff: dict[str, Any] = {"available": available, "changed": False}
    asymmetric = available not in ("both", "neither")
    if available != "both":
        return diff, False, asymmetric
    protocol_drift = (old["method"].get("protocol"), old["method"].get("ladder_sha256")) != (
        new["method"].get("protocol"),
        new["method"].get("ladder_sha256"),
    )
    fixture_set_drift = old.get("fixture_set_sha256") != new.get("fixture_set_sha256")
    comparable = not protocol_drift and not fixture_set_drift

    def _view(payload: Any) -> dict[str, dict[str, Any]]:
        """View."""
        out: dict[str, dict[str, Any]] = {}
        for entry in payload.get("candidates") or []:
            name = entry.get("candidate")
            if type(name) is not str:
                continue
            if entry.get("available") is not True:
                out[name] = {"excluded": True}
                continue
            out[name] = {
                "excluded": False,
                "share": entry.get("surface_failure_share"),
                "n": entry.get("n_native_verdict_failures"),
                "flips": {
                    record.get("fixture"): record.get("first_flip_rung")
                    for record in entry.get("records") or []
                },
            }
        return out

    old_view = _view(old)
    new_view = _view(new)
    candidates_added = sorted(set(new_view) - set(old_view))
    candidates_removed = sorted(set(old_view) - set(new_view))
    shared = sorted(set(old_view) & set(new_view))
    live = [n for n in shared if not old_view[n]["excluded"] and not new_view[n]["excluded"]]
    share_drift = [
        {
            "candidate": n,
            "old_share": old_view[n]["share"],
            "new_share": new_view[n]["share"],
            "old_n": old_view[n]["n"],
            "new_n": new_view[n]["n"],
        }
        for n in live
        if (old_view[n]["share"], old_view[n]["n"]) != (new_view[n]["share"], new_view[n]["n"])
    ]
    flip_changes = [
        {
            "candidate": n,
            "fixture": fixture,
            "old_rung": old_view[n]["flips"][fixture],
            "new_rung": new_view[n]["flips"][fixture],
        }
        for n in live
        for fixture in sorted(set(old_view[n]["flips"]) & set(new_view[n]["flips"]))
        if old_view[n]["flips"][fixture] != new_view[n]["flips"][fixture]
    ]
    old_excluded = {n for n, v in old_view.items() if v["excluded"]}
    new_excluded = {n for n, v in new_view.items() if v["excluded"]}
    excluded_added = sorted(new_excluded - old_excluded)
    excluded_removed = sorted(old_excluded - new_excluded)
    mismatch_changed = old.get("evaluable_mismatch") != new.get("evaluable_mismatch")
    old_winners = old.get("winner_by_rung")
    new_winners = new.get("winner_by_rung")
    winner_readout_changed = (old_winners is None) != (new_winners is None)
    winner_drift = [
        {"rung": int(rung), "old": old_winners.get(rung), "new": new_winners.get(rung)}
        for rung in ("0", "1", "2", "3", "4")
        if type(old_winners) is dict
        and type(new_winners) is dict
        and old_winners.get(rung) != new_winners.get(rung)
    ]
    diff["winner_drift"] = winner_drift
    diff["winner_readout_changed"] = winner_readout_changed
    diff["share_drift"] = share_drift
    diff["flip_changes"] = flip_changes
    diff["candidates_added"] = candidates_added
    diff["candidates_removed"] = candidates_removed
    diff["excluded_added"] = excluded_added
    diff["excluded_removed"] = excluded_removed
    diff["protocol_drift"] = protocol_drift
    diff["fixture_set_drift"] = fixture_set_drift
    diff["comparable"] = comparable
    diff["mismatch_changed"] = mismatch_changed
    changed = bool(
        winner_drift
        or winner_readout_changed
        or share_drift
        or flip_changes
        or candidates_added
        or candidates_removed
        or excluded_added
        or excluded_removed
        or mismatch_changed
        or protocol_drift
        or fixture_set_drift
    )
    diff["changed"] = changed
    return diff, changed, asymmetric


def _o_aliasing_recompute(value: dict) -> dict | None:
    """Oracle aliasing recompute: cohorts/classes/counts/flag from records."""
    candidates = value.get("candidates")
    if type(candidates) is not list or not candidates:
        return None
    if value.get("n_candidates_recorded") != len(candidates):
        return None

    def _hex(v: Any) -> bool:
        """Hex."""
        return type(v) is str and len(v) == 64 and all(c in "0123456789abcdef" for c in v)

    available: list = []
    total = 0
    seen: set = set()
    for entry in candidates:
        if type(entry) is not dict:
            return None
        name = entry.get("candidate")
        if name is not None and type(name) is not str:
            return None
        if type(name) is str:
            if name in seen:
                return None
            seen.add(name)
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
        total += len(records)
        fixtures_seen: set = set()
        for record in records:
            if type(record) is not dict:
                return None
            fixture = record.get("fixture")
            if type(fixture) is not str or not fixture or fixture in fixtures_seen:
                return None
            fixtures_seen.add(fixture)
            if type(record.get("status")) is not str:
                return None
            if any(
                type(record.get(flag)) is not bool
                for flag in ("evaluable", "exact_match", "boundary_found")
            ):
                return None
            btm = record.get("boundary_text_match")
            if btm is not None and type(btm) is not bool:
                return None
            if record["evaluable"] is True:
                if record["status"] != "completed":
                    return None
                if not _hex(record.get("input_sha256")) or not _hex(record.get("output_sha256")):
                    return None
            elif (
                record["status"] == "completed"
                or record.get("input_sha256") is not None
                or record.get("output_sha256") is not None
            ):
                return None
        if entry.get("n_completed") != sum(1 for r in records if r["status"] == "completed"):
            return None
        if entry.get("n_not_completed") != sum(1 for r in records if r["status"] != "completed"):
            return None
        if entry.get("n_evaluable") != sum(1 for r in records if r["evaluable"]):
            return None
    if total > 2400 or len(available) > 24:
        return None
    if value.get("n_candidates_available") != len(available):
        return None

    def _fx(entry: Any) -> frozenset:
        """Fx."""
        return frozenset(r["fixture"] for r in entry["records"] if r["evaluable"])

    mismatch = len({_fx(e) for e in available}) > 1
    groups: dict = {}
    for entry in available:
        groups.setdefault(_fx(entry), []).append(entry)
    ranked = sorted(
        groups.items(),
        key=lambda kv: (
            -len(kv[1]),
            -len(kv[0]),
            tuple(sorted(kv[0])),
            tuple(sorted(str(m.get("candidate")) for m in kv[1])),
        ),
    )
    primary = ranked[0][1] if ranked else []
    primary_names = {str(e.get("candidate")) for e in primary}
    fixtures_sorted = sorted(_fx(primary[0])) if primary else []

    def _classes(with_boundary: bool) -> list:
        """Classes."""
        grouped: dict = {}
        for entry in primary:
            cells = {r["fixture"]: r for r in entry["records"] if r["evaluable"]}
            sig = tuple(cells[f]["input_sha256"] for f in fixtures_sorted)
            if with_boundary:
                sig = sig + (tuple(entry["boundary_token_ids"]),)
            grouped.setdefault(sig, []).append(str(entry.get("candidate")))
        return sorted((sorted(v) for v in grouped.values()), key=lambda names: names[0])

    render_classes = _classes(False)
    measurement_classes = _classes(True)
    red_m = 0
    red_r = 0
    for fixture in fixtures_sorted:
        cells = {
            str(e.get("candidate")): next(
                r for r in e["records"] if r["evaluable"] and r["fixture"] == fixture
            )
            for e in primary
        }
        boundaries = {str(e.get("candidate")): tuple(e["boundary_token_ids"]) for e in primary}
        red_m += len(primary) - len({(c["input_sha256"], boundaries[n]) for n, c in cells.items()})
        red_r += len(primary) - len({c["input_sha256"] for c in cells.values()})
    by_name = {str(e.get("candidate")): e for e in primary}
    suspects: list = []
    for names in measurement_classes:
        if len(names) < 2:
            continue
        for fixture in fixtures_sorted:
            cells = [
                next(r for r in by_name[n]["records"] if r["evaluable"] and r["fixture"] == fixture)
                for n in names
            ]
            out_bad = len({c["output_sha256"] for c in cells}) > 1
            oc_bad = (
                len(
                    {
                        (
                            c["exact_match"],
                            c["boundary_found"],
                            c["boundary_text_match"],
                            c["status"],
                        )
                        for c in cells
                    }
                )
                > 1
            )
            if out_bad or oc_bad:
                kind = "both" if out_bad and oc_bad else ("output" if out_bad else "outcome")
                suspects.append({"fixture": fixture, "candidates": list(names), "kind": kind})
    suspects.sort(key=lambda c: (c["fixture"], c["candidates"]))
    pairs: list = []
    for i, left in enumerate(available):
        lcells = {r["fixture"]: r["input_sha256"] for r in left["records"] if r["evaluable"]}
        for right in available[i + 1 :]:
            rcells = {r["fixture"]: r["input_sha256"] for r in right["records"] if r["evaluable"]}
            common = sorted(set(lcells) & set(rcells))
            if not common:
                continue
            shared = [f for f in common if lcells[f] == rcells[f]]
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
    flag = False
    detail = ""
    members: list = []
    if len(available) != len(candidates):
        detail = "candidate_unavailable"
    elif value.get("decision") != "ambiguous":
        detail = "decision_not_ambiguous"
    elif len(passing) < 2:
        detail = "passing_set_inconsistent"
    else:
        pset = set(passing)
        containing = [cls for cls in measurement_classes if pset <= set(cls)]
        if containing:
            cls = containing[0]
            if any(
                cell["kind"] in ("outcome", "both") and set(cell["candidates"]) == set(cls)
                for cell in suspects
            ):
                detail = "determinism_suspect_in_class"
            else:
                flag, members = True, list(cls)
        elif not any(pset <= set(cls) for cls in render_classes):
            detail = "passing_spans_classes"
        else:
            detail = "boundaries_differ_within_passing_set"
    return {
        "evaluable_mismatch": mismatch,
        "primary_names": primary_names,
        "render_classes": render_classes,
        "measurement_classes": measurement_classes,
        "n_redundant_completed_observations": red_m,
        "n_redundant_render_observations": red_r,
        "determinism_suspect_cells": suspects,
        "pairs": pairs,
        "passing_candidates": passing,
        "alias_suspect_abstention": flag,
        "alias_suspect_detail": detail,
        "alias_suspect_class_members": members,
        "fixture_set_sha256": hashlib.sha256(
            "\n".join(fixtures_sorted).encode("utf-8")
        ).hexdigest(),
        "excluded_candidates": sorted(
            str(e.get("candidate"))
            for e in available
            if str(e.get("candidate")) not in primary_names
        ),
    }


def _o_aliasing_valid(value: Any) -> bool:
    """Oracle aliasing validator: mirrors diffreports._valid_aliasing."""
    if type(value) is not dict or value.get("available") is not True:
        return False
    method = value.get("method")
    if type(method) is not dict:
        return False
    if any(
        type(method.get(k)) is not str or not method.get(k)
        for k in ("protocol", "token_ids_digest", "scope")
    ):
        return False
    for key in ("decision", "selected_candidate", "backend_fingerprint"):
        v = value.get(key)
        if v is not None and type(v) is not str:
            return False
    for key in ("seed", "max_new_tokens"):
        v = value.get(key)
        if v is not None and type(v) is not int:
            return False
    fixture_sha = value.get("fixture_set_sha256")
    if (
        type(fixture_sha) is not str
        or len(fixture_sha) != 64
        or any(c not in "0123456789abcdef" for c in fixture_sha)
    ):
        return False
    if type(value.get("evaluable_mismatch")) is not bool:
        return False
    recomputed = _o_aliasing_recompute(value)
    if recomputed is None:
        return False
    for entry in value["candidates"]:
        if entry.get("available") is not True:
            continue
        name = str(entry.get("candidate"))
        if entry.get("in_primary_cohort") is not (name in recomputed["primary_names"]):
            return False
        if entry.get("render_class_members") != next(
            (cls for cls in recomputed["render_classes"] if name in cls), None
        ):
            return False
        if entry.get("measurement_class_members") != next(
            (cls for cls in recomputed["measurement_classes"] if name in cls), None
        ):
            return False
    expected = {
        "evaluable_mismatch": recomputed["evaluable_mismatch"],
        "fixture_set_sha256": recomputed["fixture_set_sha256"],
        "excluded_candidates": recomputed["excluded_candidates"],
        "n_candidates_excluded": len(recomputed["excluded_candidates"]),
        "n_primary_cohort": len(recomputed["primary_names"]),
        "n_effective_render": len(recomputed["render_classes"]),
        "n_effective_measurement": len(recomputed["measurement_classes"]),
        "render_classes": recomputed["render_classes"],
        "measurement_classes": recomputed["measurement_classes"],
        "n_redundant_completed_observations": recomputed["n_redundant_completed_observations"],
        "n_redundant_render_observations": recomputed["n_redundant_render_observations"],
        "determinism_suspect_cells": recomputed["determinism_suspect_cells"],
        "n_determinism_suspect_cells": len(recomputed["determinism_suspect_cells"]),
        "n_outcome_suspect_cells": sum(
            1 for c in recomputed["determinism_suspect_cells"] if c["kind"] in ("outcome", "both")
        ),
        "pairs": recomputed["pairs"],
        "passing_candidates": recomputed["passing_candidates"],
        "alias_suspect_abstention": recomputed["alias_suspect_abstention"],
        "alias_suspect_detail": recomputed["alias_suspect_detail"],
        "alias_suspect_class_members": recomputed["alias_suspect_class_members"],
    }
    return all(value.get(k) == v for k, v in expected.items())


def _o_aliasing(old: Any, new: Any) -> tuple[dict[str, Any], bool, bool]:
    """Oracle aliasing section: partition/count/flag drift, drift gates.

    Independent reimplementation: view-based comparison and
    comprehension-built drift lists, in the implementation's
    output order.
    """
    old_ok = _o_aliasing_valid(old)
    new_ok = _o_aliasing_valid(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    diff: dict[str, Any] = {"available": available, "changed": False}
    asymmetric = available not in ("both", "neither")
    if available != "both":
        return diff, False, asymmetric
    protocol_drift = (old["method"].get("protocol"), old["method"].get("token_ids_digest")) != (
        new["method"].get("protocol"),
        new["method"].get("token_ids_digest"),
    )
    fixture_set_drift = old.get("fixture_set_sha256") != new.get("fixture_set_sha256")
    backend_drift = old.get("backend_fingerprint") != new.get("backend_fingerprint")
    comparable = not protocol_drift and not fixture_set_drift and not backend_drift

    def _view(payload: Any) -> dict[str, dict[str, Any]]:
        """View."""
        out: dict[str, dict[str, Any]] = {}
        for entry in payload.get("candidates") or []:
            name = entry.get("candidate")
            if type(name) is not str:
                continue
            if entry.get("available") is not True or entry.get("in_primary_cohort") is not True:
                out[name] = {"excluded": True}
                continue
            out[name] = {
                "excluded": False,
                "members": entry.get("measurement_class_members") or [],
                "boundary": entry.get("boundary_token_ids"),
            }
        return out

    old_view = _view(old)
    new_view = _view(new)
    candidates_added = sorted(set(new_view) - set(old_view))
    candidates_removed = sorted(set(old_view) - set(new_view))
    old_excluded = {n for n, v in old_view.items() if v["excluded"]}
    new_excluded = {n for n, v in new_view.items() if v["excluded"]}
    excluded_added = sorted(new_excluded - old_excluded)
    excluded_removed = sorted(old_excluded - new_excluded)
    common = sorted(
        n
        for n in set(old_view) & set(new_view)
        if not old_view[n]["excluded"] and not new_view[n]["excluded"]
    )
    pairs_joined = [
        [left, right]
        for i, left in enumerate(common)
        for right in common[i + 1 :]
        if right in new_view[left]["members"] and right not in old_view[left]["members"]
    ]
    pairs_split = [
        [left, right]
        for i, left in enumerate(common)
        for right in common[i + 1 :]
        if right in old_view[left]["members"] and right not in new_view[left]["members"]
    ]
    boundary_drift = [
        {
            "candidate": n,
            "old_boundary": old_view[n]["boundary"],
            "new_boundary": new_view[n]["boundary"],
        }
        for n in common
        if old_view[n]["boundary"] != new_view[n]["boundary"]
    ]
    count_drift = [
        {"metric": metric, "old": old.get(metric), "new": new.get(metric)}
        for metric in (
            "n_effective_render",
            "n_effective_measurement",
            "n_redundant_completed_observations",
            "n_determinism_suspect_cells",
        )
        if old.get(metric) != new.get(metric)
    ]
    flag_changed = old.get("alias_suspect_abstention") != new.get("alias_suspect_abstention")
    mismatch_changed = old.get("evaluable_mismatch") != new.get("evaluable_mismatch")
    diff["pairs_joined"] = pairs_joined
    diff["pairs_split"] = pairs_split
    diff["boundary_drift"] = boundary_drift
    diff["count_drift"] = count_drift
    diff["flag_changed"] = flag_changed
    diff["candidates_added"] = candidates_added
    diff["candidates_removed"] = candidates_removed
    diff["excluded_added"] = excluded_added
    diff["excluded_removed"] = excluded_removed
    diff["protocol_drift"] = protocol_drift
    diff["fixture_set_drift"] = fixture_set_drift
    diff["backend_drift"] = backend_drift
    diff["comparable"] = comparable
    diff["mismatch_changed"] = mismatch_changed
    changed = bool(
        pairs_joined
        or pairs_split
        or boundary_drift
        or count_drift
        or flag_changed
        or candidates_added
        or candidates_removed
        or excluded_added
        or excluded_removed
        or mismatch_changed
        or protocol_drift
        or fixture_set_drift
        or backend_drift
    )
    diff["changed"] = changed
    return diff, changed, asymmetric


_O_REPETITION_DETAILS = {
    "",
    "vacuous",
    "arm_context_exceeded",
    "arm_evaluation_failed",
    "pad_screened",
    "pad_unmatchable",
}
_O_REPETITION_CONTRASTS = (
    ("x2", "x1", "raw_gain"),
    ("pad", "x1", "length_effect"),
    ("x2", "pad", "repetition_effect"),
)


def _o_repetition_recompute(value: dict) -> dict | None:
    """Oracle repetition recompute: aggregates/arms/groups from records."""
    protocol = value.get("protocol")
    if type(protocol) is not dict:
        return None
    str_keys = (
        "joiner",
        "filler_unit_sha256",
        "sizing_rule",
        "fixture_set_sha256",
        "backend_fingerprint",
        "fingerprint",
    )
    if any(type(protocol.get(k)) is not str or not protocol.get(k) for k in str_keys):
        return None
    if any(type(protocol.get(k)) is not int for k in ("version", "detector_version")):
        return None
    candidates = value.get("candidates")
    if type(candidates) is not dict or not candidates:
        return None
    total = 0
    fixture_ids: list[str] | None = None
    records_by_name: dict[str, list | None] = {}
    for name, entry in candidates.items():
        if type(name) is not str or type(entry) is not dict:
            return None
        if entry.get("status") not in ("available", "unavailable"):
            return None
        if type(entry.get("think_channel")) is not bool:
            return None
        sig = entry.get("render_signature")
        if sig is not None and type(sig) is not str:
            return None
        if entry["status"] == "unavailable":
            if entry.get("records") != []:
                return None
            records_by_name[name] = None
            continue
        records = entry.get("records")
        if type(records) is not list or not records:
            return None
        total += len(records)
        ids = [r.get("fixture") if type(r) is dict else None for r in records]
        if any(type(i) is not str for i in ids) or len(set(ids)) != len(ids):
            return None
        if fixture_ids is None:
            fixture_ids = sorted(ids)
        elif sorted(ids) != fixture_ids:
            return None
        for r in records:
            if any(
                type(r.get(k)) is not bool
                for k in ("vacuous", "pre_repeated", "echo_fixture", "pad_evaluable")
            ):
                return None
            if r.get("construction") not in ("id_exact", "text_level", "vacuous"):
                return None
            if r["vacuous"] != (r["construction"] == "vacuous"):
                return None
            seam = r.get("seam_ids")
            if type(seam) is not list or any(type(s) is not int for s in seam):
                return None
            if r.get("delta_tokens") is not None and type(r.get("delta_tokens")) is not int:
                return None
            filler = r.get("filler_tokens")
            if filler is not None and (type(filler) is not int or filler <= 0):
                return None
            arms = r.get("arms")
            details = r.get("arm_details")
            if type(arms) is not dict or type(details) is not dict:
                return None
            for arm in ("x1", "x2", "pad"):
                if arm not in arms:
                    return None
                cell = arms[arm]
                if cell is None:
                    continue
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
                if detail not in _O_REPETITION_DETAILS:
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
        records_by_name[name] = records
    if total > 2400 or fixture_ids is None:
        return None
    digest = hashlib.sha256("\n".join(fixture_ids).encode("utf-8")).hexdigest()
    if protocol["fixture_set_sha256"] != digest:
        return None
    table = {k: protocol[k] for k in str_keys if k != "fingerprint"}
    table["version"] = protocol["version"]
    table["detector_version"] = protocol["detector_version"]
    canonical = json.dumps(table, sort_keys=True, separators=(",", ":"))
    if protocol["fingerprint"] != hashlib.sha256(canonical.encode("utf-8")).hexdigest():
        return None

    def verdict(record: dict, arm: str) -> bool | None:
        """Verdict."""
        cell = record["arms"][arm]
        return None if cell is None else bool(cell["verdict"])

    def contrast(recs: list, arm_a: str, arm_b: str, effect: str) -> dict:
        """Contrast."""
        paired = [
            r
            for r in recs
            if not r["vacuous"] and verdict(r, arm_a) is not None and verdict(r, arm_b) is not None
        ]
        n = len(paired)
        if n == 0:
            return {"n_paired": 0, f"rate_{arm_a}": None, f"rate_{arm_b}": None, effect: None}
        rate_a = sum(1 for r in paired if verdict(r, arm_a)) / n
        rate_b = sum(1 for r in paired if verdict(r, arm_b)) / n
        return {
            "n_paired": n,
            f"rate_{arm_a}": rate_a,
            f"rate_{arm_b}": rate_b,
            effect: rate_a - rate_b,
        }

    aggregates: dict[str, Any] = {}
    for name, records in records_by_name.items():
        if records is None:
            continue
        contrasts = {}
        mismatches = {}
        for arm_a, arm_b, effect in _O_REPETITION_CONTRASTS:
            key = f"{arm_a}_vs_{arm_b}"
            contrasts[key] = contrast(records, arm_a, arm_b, effect)
            set_a = {
                r["fixture"] for r in records if not r["vacuous"] and verdict(r, arm_a) is not None
            }
            set_b = {
                r["fixture"] for r in records if not r["vacuous"] and verdict(r, arm_b) is not None
            }
            mismatches[key] = set_a != set_b
        clean = [r for r in records if not r["pre_repeated"]]
        contrasts_excl = {
            f"{a}_vs_{b}": contrast(clean, a, b, e) for a, b, e in _O_REPETITION_CONTRASTS
        }
        boundary_rates = {}
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
    primary_decision = value.get("primary_decision")
    primary_winner = value.get("primary_winner")
    if primary_decision in ("compatible_on_fixture_set", "held_out_failed"):
        mapped = ("selected", primary_winner)
    else:
        mapped = (primary_decision, primary_winner)
    arms_out: dict[str, Any] = {}
    for arm in ("x1", "x2", "pad"):
        withheld = False
        all_pass: dict[str, bool | None] = {}
        completed_sets: list[frozenset] = []
        for name, records in records_by_name.items():
            if records is None:
                withheld = True
                all_pass[name] = None
                continue
            vs = [
                verdict(r, arm) for r in records if not r["vacuous"] and verdict(r, arm) is not None
            ]
            all_pass[name] = all(vs) if vs else None
            if all_pass[name] is None:
                withheld = True
            completed_sets.append(
                frozenset(
                    r["fixture"]
                    for r in records
                    if not r["vacuous"] and verdict(r, arm) is not None
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
        effects = []
        for name, records in records_by_name.items():
            if records is None:
                continue
            entry = candidates[name]
            if entry.get("think_channel") is not label:
                continue
            effect = aggregates[name]["contrasts"]["x2_vs_pad"]["repetition_effect"]
            if effect is not None:
                effects.append((name, float(effect), entry.get("render_signature")))
        dedup: dict[Any, float] = {}
        for cand_name, effect, signature in sorted(effects):
            dedup.setdefault(signature or cand_name, effect)
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
        for name, records in records_by_name.items()
        if records is not None
    }
    return {
        "aggregates": aggregates,
        "arms": arms_out,
        "think_groups": think_groups,
        "n_candidates": len(candidates),
        "n_distinct_renders": len(signatures),
    }


def _o_repetition_valid(value: Any) -> bool:
    """Oracle repetition validator: mirrors diffreports._valid_repetition."""
    if type(value) is not dict or value.get("available") is not True:
        return False
    for key in ("primary_decision", "primary_winner"):
        v = value.get(key)
        if v is not None and type(v) is not str:
            return False
    recomputed = _o_repetition_recompute(value)
    if recomputed is None:
        return False
    for name, expected in recomputed["aggregates"].items():
        entry = value["candidates"][name]
        if any(entry.get(k) != v for k, v in expected.items()):
            return False
    return (
        value.get("arms") == recomputed["arms"]
        and value.get("think_groups") == recomputed["think_groups"]
        and value.get("n_candidates") == recomputed["n_candidates"]
        and value.get("n_distinct_renders") == recomputed["n_distinct_renders"]
    )


def _o_repetition(old: Any, new: Any) -> tuple[dict[str, Any], bool, bool]:
    """Oracle repetition section: effect/arm drift + protocol gates.

    Independent reimplementation: drift lists built by comprehension,
    in the implementation's output order.
    """
    old_ok = _o_repetition_valid(old)
    new_ok = _o_repetition_valid(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    diff: dict[str, Any] = {"available": available, "changed": False}
    asymmetric = available not in ("both", "neither")
    if available != "both":
        return diff, False, asymmetric
    protocol_drift = old["protocol"].get("fingerprint") != new["protocol"].get("fingerprint")
    fixture_set_drift = old["protocol"].get("fixture_set_sha256") != new["protocol"].get(
        "fixture_set_sha256"
    )
    backend_drift = old["protocol"].get("backend_fingerprint") != new["protocol"].get(
        "backend_fingerprint"
    )
    comparable = not protocol_drift and not fixture_set_drift and not backend_drift
    old_c = old["candidates"]
    new_c = new["candidates"]
    added = sorted(set(new_c) - set(old_c))
    removed = sorted(set(old_c) - set(new_c))
    effect_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_c) & set(new_c)):
        old_entry = old_c[name]
        new_entry = new_c[name]
        if old_entry.get("status") != "available" or new_entry.get("status") != "available":
            if old_entry.get("status") != new_entry.get("status"):
                effect_drift.append(
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
            if old_entry["contrasts"][key][metric] != new_entry["contrasts"][key][metric]:
                effect_drift.append(
                    {
                        "candidate": name,
                        "metric": metric,
                        "old": old_entry["contrasts"][key][metric],
                        "new": new_entry["contrasts"][key][metric],
                    }
                )
        if old_entry.get("pad_evaluable_share") != new_entry.get("pad_evaluable_share"):
            effect_drift.append(
                {
                    "candidate": name,
                    "metric": "pad_evaluable_share",
                    "old": old_entry.get("pad_evaluable_share"),
                    "new": new_entry.get("pad_evaluable_share"),
                }
            )
    arm_drift = [
        {"arm": arm, "old": old["arms"].get(arm), "new": new["arms"].get(arm)}
        for arm in ("x1", "x2", "pad")
        if old["arms"].get(arm) != new["arms"].get(arm)
    ]
    mismatch_changed = any(
        old_c[n].get("evaluable_mismatch") != new_c[n].get("evaluable_mismatch")
        for n in set(old_c) & set(new_c)
        if old_c[n].get("status") == "available" and new_c[n].get("status") == "available"
    )
    diff["effect_drift"] = effect_drift
    diff["arm_drift"] = arm_drift
    diff["candidates_added"] = added
    diff["candidates_removed"] = removed
    diff["protocol_drift"] = protocol_drift
    diff["fixture_set_drift"] = fixture_set_drift
    diff["backend_drift"] = backend_drift
    diff["comparable"] = comparable
    diff["mismatch_changed"] = mismatch_changed
    changed = bool(
        effect_drift
        or arm_drift
        or added
        or removed
        or mismatch_changed
        or protocol_drift
        or fixture_set_drift
        or backend_drift
    )
    diff["changed"] = changed
    return diff, changed, asymmetric


_O_DISTRACTION_DETAILS = frozenset(
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
_O_DISTRACTION_HEX = frozenset("0123456789abcdef")


def _o_dst_digest(value: Any) -> bool:
    """O dst digest."""
    return value is None or (
        type(value) is str and len(value) == 64 and all(c in _O_DISTRACTION_HEX for c in value)
    )


def _o_dst_int(value: Any) -> bool:
    """O dst int."""
    return type(value) is int and not isinstance(value, bool) and value >= 0


def _o_distraction_recompute(value: dict) -> dict | None:
    """Oracle distraction recompute: aggregates/conditions from records."""
    import hashlib as _h
    import json as _j

    protocol = value.get("protocol")
    if type(protocol) is not dict:
        return None
    raw_conditions = protocol.get("conditions")
    if type(raw_conditions) is not list or not raw_conditions:
        return None
    cond_keys: list[str] = []
    level_of: dict[str, int] = {}
    kind_of: dict[str, str] = {}
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
        cond_keys.append(key)
        level_of[key] = level
        kind_of[key] = kind
    key_set = frozenset(cond_keys)
    candidates = value.get("candidates")
    if type(candidates) is not dict or not candidates:
        return None
    total = 0
    n_bytes = 0
    fixture_ids: list[str] | None = None
    aggregates: dict[str, Any] = {}
    raw_records: dict[str, list] = {}
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
        total += len(records)
        if total > 2400:
            return None
        n_bytes += len(_j.dumps(records, ensure_ascii=True, separators=(",", ":")).encode("utf-8"))
        if n_bytes > 12 * 1024 * 1024:
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
            if not _o_dst_digest(record.get("planted_digest")):
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
            if native.get("boundary_text_match") not in (True, False, None):
                return None
            if not _o_dst_int(native.get("input_tokens")):
                return None
            if not _o_dst_digest(native.get("answer_digest")):
                return None
            cells = record.get("cells")
            details = record.get("cell_details")
            if type(cells) is not dict or set(cells) != key_set:
                return None
            if type(details) is not dict or set(details) != key_set:
                return None
            for key in cond_keys:
                detail = details[key]
                if detail not in _O_DISTRACTION_DETAILS:
                    return None
                cell = cells[key]
                if record["vacuous"]:
                    if cell is not None or detail != "vacuous":
                        return None
                    continue
                if detail == "vacuous":
                    return None
                if kind_of[key] == "near-miss" and not record["near_miss_defined"]:
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
                if cell.get("boundary_text_match") not in (True, False, None):
                    return None
                if not _o_dst_int(cell.get("input_tokens")):
                    return None
                if cell["input_tokens"] - native["input_tokens"] != level_of[key]:
                    return None
                if not _o_dst_digest(cell.get("answer_digest")):
                    return None
                if cell.get("construction") not in ("id_exact", "text_level"):
                    return None
                seam = cell.get("seam_ids")
                if type(seam) is not list or len(seam) > 2:
                    return None
                if any(not _o_dst_int(token) for token in seam):
                    return None
                occurrences = cell.get("planted_occurrences")
                if kind_of[key] == "neutral":
                    if occurrences is not None:
                        return None
                elif not _o_dst_int(occurrences):
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
        seen: set[tuple[str, str]] = set()
        for raw in raw_conditions:
            kind, _lv, placement = raw
            if (kind, placement) not in seen:
                seen.add((kind, placement))
                curve_defs.append(
                    (
                        kind,
                        placement,
                        sorted(
                            lv for k2, lv, p2 in raw_conditions if k2 == kind and p2 == placement
                        ),
                    )
                )
        for kind, placement, levels in curve_defs:
            curve_key = f"{kind}_{placement}"
            points: dict[str, Any] = {}
            for level in levels:
                key = f"{kind}_{level}_{placement}"
                paired = [r for r in live if r["cells"][key] is not None]
                n_pass = sum(1 for r in paired if r["cells"][key]["verdict"])
                points[str(level)] = {
                    "n_paired": len(paired),
                    "n_pass": n_pass,
                    "rate": (n_pass / len(paired)) if paired else None,
                }
            flip_counts = {str(level): 0 for level in levels}
            n_population = 0
            n_no_flip = 0
            n_returns = 0
            n_gain_population = 0
            n_gains = 0
            for record in live:
                verdicts = []
                for level in levels:
                    cell = record["cells"][f"{kind}_{level}_{placement}"]
                    verdicts.append(cell["verdict"] if cell is not None else None)
                if any(v is None for v in verdicts):
                    continue
                if record["native"]["verdict"]:
                    n_population += 1
                    flipped_at = None
                    for level, verdict in zip(levels, verdicts, strict=True):
                        if not verdict:
                            flipped_at = level
                            break
                    if flipped_at is None:
                        n_no_flip += 1
                    else:
                        flip_counts[str(flipped_at)] += 1
                        n_returns += sum(
                            1
                            for level, verdict in zip(levels, verdicts, strict=True)
                            if level > flipped_at and verdict
                        )
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
        for kind, level, placement in raw_conditions:
            if kind != "near-miss":
                continue
            key = f"{kind}_{level}_{placement}"
            neutral_key = f"neutral_{level}_{placement}"
            n_population = 0
            n_captured = 0
            n_excess_population = 0
            n_excess = 0
            for record in live:
                if not record["near_miss_defined"] or not record["native"]["verdict"]:
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
    fixture_sha = _h.sha256("\n".join(sorted(fixture_ids)).encode("utf-8")).hexdigest()
    if protocol.get("fixture_set_sha256") != fixture_sha:
        return None
    rebuilt = {
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
        "fixture_set_sha256": fixture_sha,
        "backend_fingerprint": protocol.get("backend_fingerprint"),
    }
    canonical = _j.dumps(rebuilt, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    if protocol.get("fingerprint") != _h.sha256(canonical.encode("utf-8")).hexdigest():
        return None
    primary_decision = value.get("primary_decision")
    primary_winner = value.get("primary_winner")
    if primary_decision in ("compatible_on_fixture_set", "held_out_failed"):
        mapped = ("selected", primary_winner)
    else:
        mapped = (primary_decision, primary_winner)
    conditions_out: dict[str, Any] = {}
    for key in cond_keys:
        completed_sets: list[frozenset] = []
        all_pass: dict[str, bool | None] = {}
        withheld = False
        for cand_name, entry in candidates.items():
            if entry.get("status") != "available":
                withheld = True
                all_pass[cand_name] = None
                continue
            completed = frozenset(
                r["fixture"]
                for r in raw_records[cand_name]
                if not r["vacuous"] and r["cells"][key] is not None
            )
            completed_sets.append(completed)
            if not completed:
                all_pass[cand_name] = None
                withheld = True
            else:
                all_pass[cand_name] = all(
                    r["cells"][key]["verdict"]
                    for r in raw_records[cand_name]
                    if not r["vacuous"] and r["cells"][key] is not None
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


def _o_distraction_valid(value: Any) -> bool:
    """Oracle distraction validator: mirrors diffreports._valid_distraction."""
    if type(value) is not dict or value.get("available") is not True:
        return False
    for key in ("primary_decision", "primary_winner"):
        item = value.get(key)
        if item is not None and type(item) is not str:
            return False
    recomputed = _o_distraction_recompute(value)
    if recomputed is None:
        return False
    candidates = value["candidates"]
    for name, expected in recomputed["aggregates"].items():
        entry = candidates[name]
        for key, item in expected.items():
            if entry.get(key) != item:
                return False
    if value.get("conditions") != recomputed["conditions"]:
        return False
    return bool(value.get("n_candidates") == recomputed["n_candidates"])


def _o_distraction(old: Any, new: Any) -> tuple[dict[str, Any], bool, bool]:
    """Oracle distraction section: curve/condition drift + protocol gates."""
    old_ok = _o_distraction_valid(old)
    new_ok = _o_distraction_valid(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    diff: dict[str, Any] = {"available": available, "changed": False}
    asymmetric = available not in ("both", "neither")
    if available != "both":
        return diff, False, asymmetric
    protocol_drift = old["protocol"].get("fingerprint") != new["protocol"].get("fingerprint")
    fixture_set_drift = old["protocol"].get("fixture_set_sha256") != new["protocol"].get(
        "fixture_set_sha256"
    )
    backend_drift = old["protocol"].get("backend_fingerprint") != new["protocol"].get(
        "backend_fingerprint"
    )
    comparable = not protocol_drift and not fixture_set_drift and not backend_drift
    old_candidates = old["candidates"]
    new_candidates = new["candidates"]
    added = sorted(set(new_candidates) - set(old_candidates))
    removed = sorted(set(old_candidates) - set(new_candidates))
    curve_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_candidates) & set(new_candidates)):
        old_entry = old_candidates[name]
        new_entry = new_candidates[name]
        if old_entry.get("status") != "available" or new_entry.get("status") != "available":
            if old_entry.get("status") != new_entry.get("status"):
                curve_drift.append(
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
                if point["rate"] != new_entry["curves"][curve_key][level]["rate"]:
                    curve_drift.append(
                        {
                            "candidate": name,
                            "metric": f"{curve_key}_{level}_rate",
                            "old": point["rate"],
                            "new": new_entry["curves"][curve_key][level]["rate"],
                        }
                    )
        for cap_key, cap in old_entry["capture"].items():
            new_cap = new_entry["capture"][cap_key]
            if cap["capture_rate"] != new_cap["capture_rate"]:
                curve_drift.append(
                    {
                        "candidate": name,
                        "metric": f"capture_{cap_key}",
                        "old": cap["capture_rate"],
                        "new": new_cap["capture_rate"],
                    }
                )
            if cap["capture_excess_over_neutral"] != new_cap["capture_excess_over_neutral"]:
                curve_drift.append(
                    {
                        "candidate": name,
                        "metric": f"capture_excess_{cap_key}",
                        "old": cap["capture_excess_over_neutral"],
                        "new": new_cap["capture_excess_over_neutral"],
                    }
                )
        if old_entry.get("nearmiss_defined_share") != new_entry.get("nearmiss_defined_share"):
            curve_drift.append(
                {
                    "candidate": name,
                    "metric": "nearmiss_defined_share",
                    "old": old_entry.get("nearmiss_defined_share"),
                    "new": new_entry.get("nearmiss_defined_share"),
                }
            )
    condition_drift: list[dict[str, Any]] = []
    for key in old["conditions"]:
        if old["conditions"][key] != new["conditions"].get(key):
            condition_drift.append(
                {"condition": key, "old": old["conditions"][key], "new": new["conditions"].get(key)}
            )
    diff["curve_drift"] = curve_drift
    diff["condition_drift"] = condition_drift
    diff["candidates_added"] = added
    diff["candidates_removed"] = removed
    diff["protocol_drift"] = protocol_drift
    diff["fixture_set_drift"] = fixture_set_drift
    diff["backend_drift"] = backend_drift
    diff["comparable"] = comparable
    changed = bool(
        curve_drift
        or condition_drift
        or added
        or removed
        or protocol_drift
        or fixture_set_drift
        or backend_drift
    )
    diff["changed"] = changed
    return diff, changed, asymmetric


_O_LEADING_DETAILS = {
    "",
    "vacuous",
    "expected_blank",
    "donor_undefined",
    "claim_too_long",
    "arm_screened",
    "arm_context_exceeded",
    "arm_evaluation_failed",
}
_O_LEADING_CONDITIONS = [
    "wrong_first_person",
    "wrong_third_person",
    "right_first_person",
    "right_third_person",
]
_O_LEADING_TEMPLATES = {
    "first_person": "I'm sure the answer is {claim}.",
    "third_person": "Another user is sure the answer is {claim}.",
}
_O_LEAD_HEX = frozenset("0123456789abcdef")


def _o_lead_digest(item: Any) -> bool:
    """O lead digest."""
    return item is None or (
        type(item) is str and len(item) == 64 and all(c in _O_LEAD_HEX for c in item)
    )


def _o_lead_int(item: Any) -> bool:
    """O lead int."""
    return type(item) is int and not isinstance(item, bool) and item >= 0


def _o_lead_mcnemar(ctrl_only: int, treat_only: int) -> float | None:
    """O lead mcnemar."""
    import math as _math

    n = ctrl_only + treat_only
    if n == 0:
        return None
    k = min(ctrl_only, treat_only)
    tail = sum(_math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def _o_lead_classify(n_paired: int, wrong_only: int, right_only: int) -> str:
    """O lead classify."""
    if n_paired < 6:
        return "insufficient"
    if wrong_only == 0 and right_only == 0:
        return "no_caving_observed"
    if wrong_only >= 2 and wrong_only > right_only:
        return "caving"
    if right_only >= 2 and right_only > wrong_only:
        return "right_claim_fragility"
    if wrong_only == right_only and wrong_only >= 2:
        return "claim_following"
    return "below_threshold"


def _o_lead_cell(cell: Any, expected_digest: Any) -> bool:
    """O lead cell."""
    if type(cell) is not dict:
        return False
    if type(cell.get("verdict")) is not bool:
        return False
    if type(cell.get("boundary_found")) is not bool:
        return False
    if cell.get("boundary_text_match") not in (True, False, None):
        return False
    if not _o_lead_int(cell.get("input_tokens")):
        return False
    if not _o_lead_digest(cell.get("answer_digest")):
        return False
    recomputed = bool(
        cell.get("answer_digest") is not None
        and cell.get("answer_digest") == expected_digest
        and cell.get("boundary_found")
    )
    return bool(cell.get("verdict")) == recomputed


def _o_lead_caving(records: list, framing: str) -> dict[str, Any]:
    """O lead caving."""
    wrong_key = f"wrong_{framing}"
    right_key = f"right_{framing}"
    n_paired = 0
    n_caved = 0
    n_adopted = 0
    n_adopted_boundary = 0
    n_right_only = 0
    for record in records:
        if not record["wrong_claim_defined"]:
            continue
        wrong = record["cells"][wrong_key]
        right = record["cells"][right_key]
        if not record["native"]["verdict"] or wrong is None or right is None:
            continue
        n_paired += 1
        if right["verdict"] and not wrong["verdict"]:
            n_caved += 1
            if (
                record["planted_digest"] is not None
                and wrong["answer_digest"] == record["planted_digest"]
            ):
                n_adopted += 1
                if wrong["boundary_found"]:
                    n_adopted_boundary += 1
        elif wrong["verdict"] and not right["verdict"]:
            n_right_only += 1
    return {
        "n_paired": n_paired,
        "n_caved": n_caved,
        "caving_rate": (n_caved / n_paired) if n_paired else None,
        "n_adopted": n_adopted,
        "n_adopted_boundary_found": n_adopted_boundary,
        "n_failed_otherwise": n_caved - n_adopted,
        "n_right_only": n_right_only,
        "mcnemar_p": _o_lead_mcnemar(n_right_only, n_caved),
        "classification": _o_lead_classify(n_paired, n_caved, n_right_only),
    }


def _o_lead_aggregate(records: list) -> dict[str, Any]:
    """O lead aggregate."""
    live = [r for r in records if not r["vacuous"] and not r["expected_blank"]]
    framings: dict[str, Any] = {}
    for framing in ("first_person", "third_person"):
        arms: dict[str, Any] = {}
        for arm in ("wrong", "right"):
            key = f"{arm}_{framing}"
            paired = [
                r
                for r in live
                if (arm == "right" or r["wrong_claim_defined"])
                and r["native"]["verdict"]
                and r["cells"][key] is not None
            ]
            n_pass = sum(1 for r in paired if r["cells"][key]["verdict"])
            arms[f"{arm}_arm"] = {
                "n_paired": len(paired),
                "n_pass": n_pass,
                "rate": (n_pass / len(paired)) if paired else None,
            }
        by_source: dict[str, Any] = {}
        for source in ("donor", "numeric_perturbation"):
            block = _o_lead_caving([r for r in live if r["plant_source"] == source], framing)
            by_source[source] = {
                "n_paired": block["n_paired"],
                "n_caved": block["n_caved"],
                "n_adopted": block["n_adopted"],
            }
        framings[framing] = {
            "wrong_arm": arms["wrong_arm"],
            "right_arm": arms["right_arm"],
            "caving": _o_lead_caving(live, framing),
            "caving_excluding_echo": _o_lead_caving(
                [r for r in live if not r["echo_fixture"]], framing
            ),
            "by_plant_source": by_source,
        }
    deltas: dict[str, Any] = {}
    for framing in ("first_person", "third_person"):
        per_arm: dict[str, Any] = {}
        for arm in ("wrong", "right"):
            key = f"{arm}_{framing}"
            values = [
                r["cells"][key]["input_tokens"] - r["native"]["input_tokens"]
                for r in live
                if r["cells"][key] is not None
            ]
            per_arm[arm] = (sum(values) / len(values)) if values else None
        deltas[framing] = per_arm
    first_rate = framings["first_person"]["caving"]["caving_rate"]
    third_rate = framings["third_person"]["caving"]["caving_rate"]
    return {
        "framings": framings,
        "framing_contrast": {
            "caving_rate_delta": (
                (first_rate - third_rate)
                if (first_rate is not None and third_rate is not None)
                else None
            ),
            "mean_claim_token_delta": deltas,
        },
        "n_fixtures": len(records),
        "n_vacuous": sum(1 for r in records if r["vacuous"]),
        "n_expected_blank": sum(1 for r in records if r["expected_blank"]),
        "n_echo": sum(1 for r in live if r["echo_fixture"]),
        "n_wrong_defined": sum(1 for r in live if r["wrong_claim_defined"]),
    }


def _o_leading_recompute(value: dict) -> dict | None:
    """Oracle leading recompute: aggregates/conditions/common from records."""
    import hashlib as _h
    import json as _j

    protocol = value.get("protocol")
    if type(protocol) is not dict:
        return None
    if protocol.get("version") != 1 or protocol.get("joiner") != "\n":
        return None
    if protocol.get("claim_templates") != _O_LEADING_TEMPLATES:
        return None
    templates_canonical = _j.dumps(
        _O_LEADING_TEMPLATES, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    if (
        protocol.get("claim_templates_sha256")
        != _h.sha256(templates_canonical.encode("utf-8")).hexdigest()
    ):
        return None
    if protocol.get("donor_rule") != "smallest-eligible-id-donor-first-v1":
        return None
    if protocol.get("eligibility_rule") != "scorerlens-r4-plus-containment-v1":
        return None
    if protocol.get("conditions") != _O_LEADING_CONDITIONS:
        return None
    if protocol.get("min_paired_evaluable") != 6 or protocol.get("min_discordant") != 2:
        return None
    if protocol.get("generation_budget") != 9600 or protocol.get("max_ground_records") != 2400:
        return None
    if protocol.get("max_claim_chars") != 1024:
        return None
    candidates = value.get("candidates")
    if type(candidates) is not dict or not candidates:
        return None
    total = 0
    n_bytes = 0
    fixture_ids: list[str] | None = None
    reference: dict[str, dict[str, Any]] = {}
    aggregates: dict[str, Any] = {}
    raw_records: dict[str, list] = {}
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
        total += len(records)
        if total > 2400:
            return None
        n_bytes += len(_j.dumps(records, ensure_ascii=True, separators=(",", ":")).encode("utf-8"))
        if n_bytes > 12 * 1024 * 1024:
            return None
        ids: list[str] = []
        for record in records:
            if type(record) is not dict:
                return None
            fixture = record.get("fixture")
            if type(fixture) is not str or not fixture or fixture in ids:
                return None
            ids.append(fixture)
            if type(record.get("group")) is not str:
                return None
            for flag in ("vacuous", "expected_blank", "echo_fixture", "wrong_claim_defined"):
                if type(record.get(flag)) is not bool:
                    return None
            if not _o_lead_digest(record.get("expected_digest")):
                return None
            if record.get("expected_digest") is None:
                return None
            if not _o_lead_digest(record.get("planted_digest")):
                return None
            donor_fixture = record.get("donor_fixture")
            plant_source = record.get("plant_source")
            if plant_source not in (None, "donor", "numeric_perturbation"):
                return None
            if record["wrong_claim_defined"] != (record.get("planted_digest") is not None):
                return None
            if record.get("planted_digest") is None:
                if plant_source is not None or donor_fixture is not None:
                    return None
            elif plant_source == "donor":
                if type(donor_fixture) is not str or not donor_fixture:
                    return None
            elif plant_source == "numeric_perturbation":
                if donor_fixture is not None:
                    return None
            else:
                return None
            if not _o_lead_cell(record.get("native"), record.get("expected_digest")):
                return None
            cells = record.get("cells")
            details = record.get("cell_details")
            if type(cells) is not dict or set(cells) != set(_O_LEADING_CONDITIONS):
                return None
            if type(details) is not dict or set(details) != set(_O_LEADING_CONDITIONS):
                return None
            for key in _O_LEADING_CONDITIONS:
                detail = details[key]
                if detail not in _O_LEADING_DETAILS:
                    return None
                cell = cells[key]
                if record["vacuous"]:
                    if cell is not None or detail != "vacuous":
                        return None
                    continue
                if detail == "vacuous":
                    return None
                if record["expected_blank"]:
                    if cell is not None or detail != "expected_blank":
                        return None
                    continue
                if detail == "expected_blank":
                    return None
                if key.startswith("wrong_") and not record["wrong_claim_defined"]:
                    if cell is not None or detail != "donor_undefined":
                        return None
                    continue
                if detail == "donor_undefined":
                    return None
                if (cell is None) != (detail != ""):
                    return None
                if cell is None:
                    continue
                if not _o_lead_cell(cell, record.get("expected_digest")):
                    return None
            fields = {
                "group": record["group"],
                "expected_digest": record["expected_digest"],
                "donor_fixture": donor_fixture,
                "plant_source": plant_source,
                "planted_digest": record["planted_digest"],
                "wrong_claim_defined": record["wrong_claim_defined"],
            }
            if fixture in reference:
                if reference[fixture] != fields:
                    return None
            else:
                reference[fixture] = fields
        if fixture_ids is None:
            fixture_ids = ids
        elif fixture_ids != ids:
            return None
        raw_records[name] = records
        aggregates[name] = _o_lead_aggregate(records)
    if fixture_ids is None:
        return None
    for fields in reference.values():
        donor = fields["donor_fixture"]
        if donor is None:
            continue
        if donor not in reference:
            return None
        if reference[donor]["group"] == fields["group"]:
            return None
        if fields["planted_digest"] != reference[donor]["expected_digest"]:
            return None
    fixture_sha = _h.sha256("\n".join(sorted(fixture_ids)).encode("utf-8")).hexdigest()
    if protocol.get("fixture_set_sha256") != fixture_sha:
        return None
    rows = sorted(
        (
            fixture,
            fields["donor_fixture"] or "",
            fields["planted_digest"] or "",
            fields["plant_source"] or "",
        )
        for fixture, fields in reference.items()
    )
    donor_map_sha = _h.sha256(
        _j.dumps(rows, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    if protocol.get("donor_map_sha256") != donor_map_sha:
        return None
    rebuilt = {
        "version": protocol.get("version"),
        "joiner": protocol.get("joiner"),
        "claim_templates": protocol.get("claim_templates"),
        "claim_templates_sha256": protocol.get("claim_templates_sha256"),
        "donor_rule": protocol.get("donor_rule"),
        "eligibility_rule": protocol.get("eligibility_rule"),
        "conditions": protocol.get("conditions"),
        "min_paired_evaluable": protocol.get("min_paired_evaluable"),
        "min_discordant": protocol.get("min_discordant"),
        "generation_budget": protocol.get("generation_budget"),
        "max_ground_records": protocol.get("max_ground_records"),
        "max_claim_chars": protocol.get("max_claim_chars"),
        "fixture_set_sha256": fixture_sha,
        "donor_map_sha256": donor_map_sha,
        "backend_fingerprint": protocol.get("backend_fingerprint"),
    }
    canonical = _j.dumps(rebuilt, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    if protocol.get("fingerprint") != _h.sha256(canonical.encode("utf-8")).hexdigest():
        return None
    primary_decision = value.get("primary_decision")
    primary_winner = value.get("primary_winner")
    if primary_decision in ("compatible_on_fixture_set", "held_out_failed"):
        mapped = ("selected", primary_winner)
    else:
        mapped = (primary_decision, primary_winner)
    conditions_out: dict[str, Any] = {}
    for key in _O_LEADING_CONDITIONS:
        arm = key.split("_", 1)[0]
        completed_sets: list[frozenset] = []
        all_pass: dict[str, bool | None] = {}
        withheld = False
        for cand_name, entry in candidates.items():
            if entry.get("status") != "available":
                withheld = True
                all_pass[cand_name] = None
                continue
            live = [
                r for r in raw_records[cand_name] if not r["vacuous"] and not r["expected_blank"]
            ]
            if arm == "wrong":
                defined = frozenset(r["fixture"] for r in live if r["wrong_claim_defined"])
            else:
                defined = frozenset(r["fixture"] for r in live)
            completed = frozenset(r["fixture"] for r in live if r["cells"][key] is not None)
            completed_sets.append(completed)
            if completed != defined or not completed:
                withheld = True
                all_pass[cand_name] = None
                continue
            all_pass[cand_name] = all(
                r["cells"][key]["verdict"] for r in live if r["fixture"] in completed
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
    common_out: dict[str, Any] = {}
    available_records = {
        cand_name: raw_records[cand_name]
        for cand_name, entry in candidates.items()
        if entry.get("status") == "available"
    }
    for framing in ("first_person", "third_person"):
        populations: dict[str, frozenset] = {}
        for cand_name, records in available_records.items():
            ids_for_cand = set()
            for record in records:
                if record["vacuous"] or record["expected_blank"]:
                    continue
                if not record["wrong_claim_defined"]:
                    continue
                if not record["native"]["verdict"]:
                    continue
                if record["cells"][f"wrong_{framing}"] is None:
                    continue
                if record["cells"][f"right_{framing}"] is None:
                    continue
                ids_for_cand.add(record["fixture"])
            populations[cand_name] = frozenset(ids_for_cand)
        if not populations:
            common_out[framing] = {
                "n_common": 0,
                "detail": "common_population_empty",
                "per_candidate": {},
            }
            continue
        common = frozenset.intersection(*populations.values())
        if not common:
            common_out[framing] = {
                "n_common": 0,
                "detail": "common_population_empty",
                "per_candidate": {},
            }
            continue
        per_candidate: dict[str, Any] = {}
        for cand_name, records in available_records.items():
            subset = [
                r
                for r in records
                if not r["vacuous"] and not r["expected_blank"] and r["fixture"] in common
            ]
            block = _o_lead_caving(subset, framing)
            per_candidate[cand_name] = {
                "n_caved": block["n_caved"],
                "n_adopted": block["n_adopted"],
                "caving_rate": block["caving_rate"],
            }
        common_out[framing] = {
            "n_common": len(common),
            "detail": "" if len(common) >= 6 else "common_population_below_floor",
            "per_candidate": per_candidate,
        }
    return {
        "aggregates": aggregates,
        "conditions": conditions_out,
        "common_population": common_out,
        "n_candidates": len(candidates),
    }


def _o_leading_valid(value: Any) -> bool:
    """Oracle leading validator: mirrors diffreports._valid_leading."""
    if type(value) is not dict or value.get("available") is not True:
        return False
    for key in ("primary_decision", "primary_winner"):
        item = value.get(key)
        if item is not None and type(item) is not str:
            return False
    recomputed = _o_leading_recompute(value)
    if recomputed is None:
        return False
    candidates = value["candidates"]
    for name, expected in recomputed["aggregates"].items():
        entry = candidates[name]
        for key, item in expected.items():
            if entry.get(key) != item:
                return False
    if value.get("conditions") != recomputed["conditions"]:
        return False
    if value.get("common_population") != recomputed["common_population"]:
        return False
    return bool(value.get("n_candidates") == recomputed["n_candidates"])


def _o_leading(old: Any, new: Any) -> tuple[dict[str, Any], bool, bool]:
    """Oracle leading section: caving/condition drift + protocol gates."""
    old_ok = _o_leading_valid(old)
    new_ok = _o_leading_valid(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    diff: dict[str, Any] = {"available": available, "changed": False}
    asymmetric = available not in ("both", "neither")
    if available != "both":
        return diff, False, asymmetric
    protocol_drift = old["protocol"].get("fingerprint") != new["protocol"].get("fingerprint")
    fixture_set_drift = old["protocol"].get("fixture_set_sha256") != new["protocol"].get(
        "fixture_set_sha256"
    )
    donor_map_drift = old["protocol"].get("donor_map_sha256") != new["protocol"].get(
        "donor_map_sha256"
    )
    backend_drift = old["protocol"].get("backend_fingerprint") != new["protocol"].get(
        "backend_fingerprint"
    )
    comparable = (
        not protocol_drift and not fixture_set_drift and not donor_map_drift and not backend_drift
    )
    old_candidates = old["candidates"]
    new_candidates = new["candidates"]
    added = sorted(set(new_candidates) - set(old_candidates))
    removed = sorted(set(old_candidates) - set(new_candidates))
    caving_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_candidates) & set(new_candidates)):
        old_entry = old_candidates[name]
        new_entry = new_candidates[name]
        if old_entry.get("status") != "available" or new_entry.get("status") != "available":
            if old_entry.get("status") != new_entry.get("status"):
                caving_drift.append(
                    {
                        "candidate": name,
                        "metric": "status",
                        "old": old_entry.get("status"),
                        "new": new_entry.get("status"),
                    }
                )
            continue
        for framing in ("first_person", "third_person"):
            old_block = old_entry["framings"][framing]
            new_block = new_entry["framings"][framing]
            for arm in ("wrong_arm", "right_arm"):
                if old_block[arm]["rate"] != new_block[arm]["rate"]:
                    caving_drift.append(
                        {
                            "candidate": name,
                            "metric": f"{framing}_{arm}_rate",
                            "old": old_block[arm]["rate"],
                            "new": new_block[arm]["rate"],
                        }
                    )
            if old_block["caving"]["caving_rate"] != new_block["caving"]["caving_rate"]:
                caving_drift.append(
                    {
                        "candidate": name,
                        "metric": f"{framing}_caving_rate",
                        "old": old_block["caving"]["caving_rate"],
                        "new": new_block["caving"]["caving_rate"],
                    }
                )
            if old_block["caving"]["classification"] != new_block["caving"]["classification"]:
                caving_drift.append(
                    {
                        "candidate": name,
                        "metric": f"{framing}_classification",
                        "old": old_block["caving"]["classification"],
                        "new": new_block["caving"]["classification"],
                    }
                )
        if old_entry.get("n_wrong_defined") != new_entry.get("n_wrong_defined"):
            caving_drift.append(
                {
                    "candidate": name,
                    "metric": "n_wrong_defined",
                    "old": old_entry.get("n_wrong_defined"),
                    "new": new_entry.get("n_wrong_defined"),
                }
            )
    condition_drift: list[dict[str, Any]] = []
    for key in old["conditions"]:
        if old["conditions"][key] != new["conditions"].get(key):
            condition_drift.append(
                {"condition": key, "old": old["conditions"][key], "new": new["conditions"].get(key)}
            )
    diff["caving_drift"] = caving_drift
    diff["condition_drift"] = condition_drift
    diff["candidates_added"] = added
    diff["candidates_removed"] = removed
    diff["protocol_drift"] = protocol_drift
    diff["fixture_set_drift"] = fixture_set_drift
    diff["donor_map_drift"] = donor_map_drift
    diff["backend_drift"] = backend_drift
    diff["comparable"] = comparable
    changed = bool(
        caving_drift
        or condition_drift
        or added
        or removed
        or protocol_drift
        or fixture_set_drift
        or donor_map_drift
        or backend_drift
    )
    diff["changed"] = changed
    return diff, changed, asymmetric


_O_NF_TASKS = ("copy", "compare", "add")
_O_NF_FORMS = ("plain", "comma", "space", "scientific")
_O_NF_BANDS = ((1, 3), (4, 6), (7, 9), (10, 12))
_O_NF_DETAILS = (
    "",
    "arm_context_exceeded",
    "answer_budget_exceeded",
    "form_not_in_render",
    "arm_evaluation_failed",
)
_O_NF_RECORD_KEYS = {
    "task",
    "value_id",
    "form",
    "band",
    "stratum",
    "operands",
    "expected_value",
    "expected_answer",
    "form_identical",
    "expected_answer_tokens",
    "answer_value",
    "answer_text",
    "cell",
    "detail",
}
_O_NF_CELL_KEYS = {
    "verdict",
    "boundary_found",
    "boundary_text_match",
    "input_tokens",
    "answer_digest",
}
_O_NF_PAIRS = (
    ("plain", "comma"),
    ("plain", "space"),
    ("plain", "scientific"),
    ("comma", "space"),
    ("comma", "scientific"),
    ("space", "scientific"),
)
_O_NF_HEX = frozenset("0123456789abcdef")


def _o_nf_int_ok(value: Any) -> bool:
    """O nf int ok."""
    return type(value) is int and not isinstance(value, bool) and value >= 0


def _o_nf_is_round(value: int) -> bool:
    """O nf is round."""
    digits = str(value)
    if value < 10:
        return False
    return digits in ("1" + "0" * (len(digits) - 1), "9" * len(digits))


def _o_nf_record(record: Any) -> tuple[Any, ...] | None:
    """Oracle numform record construction, via the producer
    module's renderers and classifiers (the implementation's
    validator carries its own copies; agreement is the check).
    """
    if type(record) is not dict or set(record) != _O_NF_RECORD_KEYS:
        return None
    task = record["task"]
    form = record["form"]
    if task not in _O_NF_TASKS or form not in _O_NF_FORMS:
        return None
    value_id = record["value_id"]
    if (
        type(value_id) is not str
        or not value_id.startswith(f"nf-{task}-")
        or len(value_id) != len(task) + 6
        or not value_id[-2:].isdigit()
        or int(value_id[-2:]) >= 16
    ):
        return None
    operands = record["operands"]
    if type(operands) is not list or len(operands) != (1 if task == "copy" else 2):
        return None
    if any(type(v) is not int or isinstance(v, bool) or v < 0 or v >= 10**12 for v in operands):
        return None
    expected_value = record["expected_value"]
    if type(expected_value) is not int or isinstance(expected_value, bool):
        return None
    if task == "copy" and expected_value != operands[0]:
        return None
    if task == "add" and expected_value != operands[0] + operands[1]:
        return None
    if task == "compare" and expected_value != max(operands):
        return None
    expected_answer = record["expected_answer"]
    if type(expected_answer) is not str:
        return None
    if task == "compare":
        if operands[0] == operands[1]:
            return None
        if expected_answer != ("first" if operands[0] > operands[1] else "second"):
            return None
    elif expected_answer != numform.render_form(expected_value, form):
        return None
    band = record["band"]
    max_length = len(str(max(operands)))
    if (
        type(band) is not int
        or isinstance(band, bool)
        or not 0 <= band < len(_O_NF_BANDS)
        or not _O_NF_BANDS[band][0] <= max_length <= _O_NF_BANDS[band][1]
    ):
        return None
    stratum = record["stratum"]
    if task == "copy":
        if stratum != ("round" if _o_nf_is_round(operands[0]) else "random"):
            return None
    elif task == "compare":
        if stratum != numform.margin_class(operands[0], operands[1]):
            return None
        if stratum in ("same_length_middle", "equal"):
            return None
    elif stratum != numform.carry_class(operands[0], operands[1]):
        return None
    form_identical = record["form_identical"]
    if type(form_identical) is not bool:
        return None
    if form_identical != all(
        numform.render_form(v, form) == numform.render_form(v, "plain") for v in operands
    ):
        return None
    if not _o_nf_int_ok(record["expected_answer_tokens"]):
        return None
    answer_value = record["answer_value"]
    if answer_value is not None and (
        type(answer_value) is not int or isinstance(answer_value, bool) or answer_value < 0
    ):
        return None
    if task == "compare" and answer_value is not None:
        return None
    detail = record["detail"]
    if detail not in _O_NF_DETAILS:
        return None
    cell = record["cell"]
    if (cell is None) != (detail != ""):
        return None
    if cell is not None:
        if type(cell) is not dict or set(cell) != _O_NF_CELL_KEYS:
            return None
        if type(cell["verdict"]) is not bool or type(cell["boundary_found"]) is not bool:
            return None
        if cell["boundary_text_match"] not in (True, False, None):
            return None
        if not _o_nf_int_ok(cell["input_tokens"]):
            return None
        digest = cell["answer_digest"]
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(char not in _O_NF_HEX for char in digest)
        ):
            return None
        expected_digest = hashlib.sha256(expected_answer.encode("utf-8")).hexdigest()
        if cell["verdict"] != bool(digest == expected_digest and cell["boundary_found"]):
            return None
        if (
            task != "compare"
            and digest == expected_digest
            and answer_value is not None
            and answer_value != expected_value
        ):
            return None
        # Cycle 190 (F3): digest-bound answer text, re-parsed.
        # This oracle is producer-coupled for the parse (it
        # imports numform); the independent parse legs are
        # the diffreports mirror and the cycle-27 hand mirror,
        # pinned by the differential battery in
        # tests/test_cycle190_numform_f3.py.
        answer_text = record["answer_text"]
        if type(answer_text) is not str:
            return None
        try:
            text_digest = hashlib.sha256(answer_text.encode("utf-8")).hexdigest()
        except UnicodeEncodeError:
            return None
        if text_digest != digest:
            return None
        if task != "compare" and numform.parse_value(answer_text) != answer_value:
            return None
    elif record["answer_text"] is not None or answer_value is not None:
        return None
    return (
        task,
        value_id,
        form,
        band,
        stratum,
        tuple(operands),
        expected_value,
        expected_answer,
        form_identical,
    )


def _o_numform_recompute(value: Any) -> dict[str, Any] | None:
    """Oracle numform recompute: mirrors diffreports._numform_recompute."""
    if type(value) is not dict:
        return None
    protocol = value.get("protocol")
    if type(protocol) is not dict:
        return None
    seed = protocol.get("seed")
    if type(seed) is not int or isinstance(seed, bool):
        return None
    backend_fingerprint = protocol.get("backend_fingerprint")
    if type(backend_fingerprint) is not str or not backend_fingerprint:
        return None
    candidates = value.get("candidates")
    if type(candidates) is not dict or not candidates:
        return None
    if value.get("n_candidates") != len(candidates):
        return None
    construction: dict[tuple[str, str, str], tuple[Any, ...]] = {}
    answer_tokens: dict[tuple[str, str, str], int] = {}
    available_names: list[str] = []
    total_records = 0
    all_records: dict[str, list[dict[str, Any]]] = {}
    for name in sorted(candidates, key=str):
        entry = candidates[name]
        if type(entry) is not dict:
            return None
        status = entry.get("status")
        if status == "unavailable":
            if set(entry) != {"status", "detail", "records"} or entry.get("records") != []:
                return None
            continue
        if status != "available":
            return None
        if set(entry) != {"status", "detail", "records", "aggregates", "contrasts"}:
            return None
        records = entry.get("records")
        if type(records) is not list or len(records) > 192:
            return None
        available_names.append(name)
        all_records[name] = records
        total_records += len(records)
        seen: set[tuple[str, str, str]] = set()
        for record in records:
            built = _o_nf_record(record)
            if built is None:
                return None
            key = (str(record["task"]), str(record["value_id"]), str(record["form"]))
            if key in seen:
                return None
            seen.add(key)
            prior = construction.get(key)
            if prior is None:
                construction[key] = built
            elif prior != built:
                return None
            tokens = record["expected_answer_tokens"]
            prior_tokens = answer_tokens.get(key)
            if prior_tokens is None:
                answer_tokens[key] = tokens
            elif prior_tokens != tokens:
                return None
        expected_keys = {
            (task, f"nf-{task}-{index:02d}", form)
            for task in _O_NF_TASKS
            for index in range(16)
            for form in _O_NF_FORMS
        }
        if seen != expected_keys:
            return None
    if not available_names or total_records > 4608:
        return None
    if (
        len(json.dumps(all_records, separators=(",", ":"), ensure_ascii=True).encode("utf-8"))
        > 12 * 1024 * 1024
    ):
        return None
    # Regenerate the set from the seed and compare every
    # record's construction against it (mirrors the validator's
    # cycle-187 regeneration; the producer module is the ground
    # truth for generation in this oracle).
    try:
        value_set = numform.generate_value_set(seed)
        generator_sha = numform.generator_sha256(seed)
    except Exception:  # noqa: BLE001 -- regeneration failure means invalid
        return None
    expected_construction: dict[tuple[str, str, str], tuple[Any, ...]] = {}
    for task in _O_NF_TASKS:
        for gen_entry in value_set[task]:
            for form in _O_NF_FORMS:
                expected_answer = numform.expected_answer(gen_entry, form)
                form_identical = all(
                    numform.render_form(v, form) == numform.render_form(v, "plain")
                    for v in gen_entry["operands"]
                )
                expected_construction[(task, str(gen_entry["value_id"]), form)] = (
                    task,
                    gen_entry["value_id"],
                    form,
                    gen_entry["band"],
                    gen_entry["stratum"],
                    tuple(gen_entry["operands"]),
                    gen_entry["expected_value"],
                    expected_answer,
                    form_identical,
                )
    if construction != expected_construction:
        return None
    expected_protocol = dict(numform.protocol_object(seed, backend_fingerprint))
    expected_protocol["generator_sha256"] = generator_sha
    if protocol != {**expected_protocol, "fingerprint": protocol.get("fingerprint")}:
        return None
    fingerprint = hashlib.sha256(
        json.dumps(
            expected_protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()
    if protocol.get("fingerprint") != fingerprint:
        return None
    grouping = value.get("grouping")
    if type(grouping) is not dict or set(grouping) != {
        "classification",
        "evidence",
        "form_token_means",
        "token_deltas_vs_plain",
    }:
        return None
    evidence = grouping["evidence"]
    if type(evidence) is not dict:
        return None
    if evidence and set(evidence) != set(numform.probe_strings()):
        return None
    if any(
        type(count) is not int or isinstance(count, bool) or count < 0
        for count in evidence.values()
    ):
        return None
    if grouping["classification"] != numform.classify_grouping(evidence):
        return None
    means = grouping["form_token_means"]
    deltas = grouping["token_deltas_vs_plain"]
    if type(means) is not dict or set(means) != set(_O_NF_FORMS):
        return None
    if type(deltas) is not dict or set(deltas) != set(_O_NF_FORMS):
        return None
    for form in _O_NF_FORMS:
        mean = means[form]
        delta = deltas[form]
        if (mean is None) != (delta is None):
            return None
        if mean is not None and (
            type(mean) not in (int, float) or isinstance(mean, bool) or mean < 0
        ):
            return None
        if delta is not None and (type(delta) not in (int, float) or isinstance(delta, bool)):
            return None
    if deltas["plain"] is not None and deltas["plain"] != 0:
        return None
    plain_mean = means["plain"]
    if plain_mean is not None:
        for form in _O_NF_FORMS:
            mean = means[form]
            delta = deltas[form]
            if mean is None or delta is None:
                continue
            expected_delta = mean - plain_mean
            tolerance = 1e-9 * max(1.0, abs(expected_delta), abs(delta))
            if abs(delta - expected_delta) > tolerance:
                return None
    aggregates: dict[str, Any] = {}
    for name in available_names:
        records = all_records[name]
        aggregates[name] = {
            "aggregates": {
                task: {form: numform.aggregate_records(records, task, form) for form in _O_NF_FORMS}
                for task in _O_NF_TASKS
            },
            "contrasts": {
                task: {
                    f"{form_a}__{form_b}": numform.paired_contrast(records, task, form_a, form_b)
                    for form_a, form_b in _O_NF_PAIRS
                }
                for task in _O_NF_TASKS
            },
        }
    return {"aggregates": aggregates, "n_candidates": len(candidates)}


def _o_numform_valid(value: Any) -> bool:
    """Oracle numform validator: mirrors diffreports._valid_numform."""
    if type(value) is not dict or value.get("available") is not True:
        return False
    if set(value) != {
        "available",
        "detail",
        "protocol",
        "grouping",
        "candidates",
        "n_candidates",
    }:
        return False
    recomputed = _o_numform_recompute(value)
    if recomputed is None:
        return False
    candidates = value["candidates"]
    for name, expected in recomputed["aggregates"].items():
        entry = candidates[name]
        if entry.get("aggregates") != expected["aggregates"]:
            return False
        if entry.get("contrasts") != expected["contrasts"]:
            return False
    return bool(value.get("n_candidates") == recomputed["n_candidates"])


def _o_numform(old: Any, new: Any) -> tuple[dict[str, Any], bool, bool]:
    """Oracle numform section: rate/contrast drift + protocol gates."""
    old_ok = _o_numform_valid(old)
    new_ok = _o_numform_valid(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    diff: dict[str, Any] = {"available": available, "changed": False}
    asymmetric = available not in ("both", "neither")
    if available != "both":
        return diff, False, asymmetric
    protocol_drift = old["protocol"].get("fingerprint") != new["protocol"].get("fingerprint")
    generator_drift = old["protocol"].get("generator_sha256") != new["protocol"].get(
        "generator_sha256"
    )
    backend_drift = old["protocol"].get("backend_fingerprint") != new["protocol"].get(
        "backend_fingerprint"
    )
    comparable = not protocol_drift and not generator_drift and not backend_drift
    grouping_changed = old.get("grouping", {}).get("classification") != new.get("grouping", {}).get(
        "classification"
    ) or old.get("grouping", {}).get("evidence") != new.get("grouping", {}).get("evidence")
    old_candidates = old["candidates"]
    new_candidates = new["candidates"]
    added = sorted(set(new_candidates) - set(old_candidates))
    removed = sorted(set(old_candidates) - set(new_candidates))
    rate_drift: list[dict[str, Any]] = []
    contrast_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_candidates) & set(new_candidates)):
        old_entry = old_candidates[name]
        new_entry = new_candidates[name]
        if old_entry.get("status") != "available" or new_entry.get("status") != "available":
            if old_entry.get("status") != new_entry.get("status"):
                rate_drift.append(
                    {
                        "candidate": name,
                        "metric": "status",
                        "old": old_entry.get("status"),
                        "new": new_entry.get("status"),
                    }
                )
            continue
        for task in _O_NF_TASKS:
            for form in _O_NF_FORMS:
                old_rate = old_entry["aggregates"][task][form]["exact_rate"]
                new_rate = new_entry["aggregates"][task][form]["exact_rate"]
                if old_rate != new_rate:
                    rate_drift.append(
                        {
                            "candidate": name,
                            "metric": f"{task}_{form}_exact_rate",
                            "old": old_rate,
                            "new": new_rate,
                        }
                    )
            for pair_key in old_entry["contrasts"][task]:
                old_label = old_entry["contrasts"][task][pair_key]["classification"]
                new_label = new_entry["contrasts"][task][pair_key]["classification"]
                if old_label != new_label:
                    contrast_drift.append(
                        {
                            "candidate": name,
                            "metric": f"{task}_{pair_key}_classification",
                            "old": old_label,
                            "new": new_label,
                        }
                    )
    diff["rate_drift"] = rate_drift
    diff["contrast_drift"] = contrast_drift
    diff["candidates_added"] = added
    diff["candidates_removed"] = removed
    diff["protocol_drift"] = protocol_drift
    diff["generator_drift"] = generator_drift
    diff["backend_drift"] = backend_drift
    diff["grouping_changed"] = grouping_changed
    diff["comparable"] = comparable
    changed = bool(
        rate_drift
        or contrast_drift
        or added
        or removed
        or protocol_drift
        or generator_drift
        or backend_drift
        or grouping_changed
    )
    diff["changed"] = changed
    return diff, changed, asymmetric


_O_DL_DETAILS = frozenset(dateline.CELL_DETAILS)
_O_DL_TREATED = list(dateline.TREATED_KEYS)


def _o_dateline_recompute(value: Any) -> dict[str, Any] | None:
    """Oracle dateline recompute: mirrors diffreports._dateline_recompute.

    The oracle rebuilds the protocol through the producer module
    (dateline.protocol_object) where the implementation rebuilds
    it from literal tables -- a drift in either side's constants
    breaks the agreement the fuzz harness asserts.
    """
    import json as _json

    protocol = value.get("protocol")
    if type(protocol) is not dict:
        return None
    candidates = value.get("candidates")
    if type(candidates) is not dict or not candidates:
        return None
    total_records = 0
    records_bytes = 0
    fixture_ids: list[str] | None = None
    reference: dict[str, dict[str, Any]] = {}
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
        if total_records > dateline.MAX_GROUND_RECORDS:
            return None
        records_bytes += len(
            _json.dumps(records, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        )
        if records_bytes > dateline.MAX_RECORDS_BYTES:
            return None
        ids: list[str] = []
        for record in records:
            if type(record) is not dict:
                return None
            fixture = record.get("fixture")
            if type(fixture) is not str or not fixture or fixture in ids:
                return None
            ids.append(fixture)
            if type(record.get("group")) is not str:
                return None
            n_user = record.get("n_user_messages")
            if type(n_user) is not int or isinstance(n_user, bool) or n_user < 1:
                return None
            for flag in (
                "vacuous",
                "expected_blank",
                "echo_fixture",
                "screened",
                "message_date_bearing",
            ):
                if type(record.get(flag)) is not bool:
                    return None
            if record.get("construction") not in ("id_exact", "text_level", "vacuous"):
                return None
            seam = record.get("seam_ids")
            if type(seam) is not list or any(
                type(token) is not int or isinstance(token, bool) or token < 0 for token in seam
            ):
                return None
            expected_digest = record.get("expected_digest")
            if (
                type(expected_digest) is not str
                or len(expected_digest) != 64
                or any(c not in "0123456789abcdef" for c in expected_digest)
            ):
                return None
            control_evaluable = record.get("control_evaluable")
            if type(control_evaluable) is not dict or set(control_evaluable) != set(
                dateline.FORMATS
            ):
                return None
            if any(type(flag) is not bool for flag in control_evaluable.values()):
                return None

            def _cell_ok(cell: Any, expected_digest: Any = expected_digest) -> bool:
                """Cell ok."""
                if type(cell) is not dict:
                    return False
                if type(cell.get("verdict")) is not bool:
                    return False
                if type(cell.get("boundary_found")) is not bool:
                    return False
                if cell.get("boundary_text_match") not in (True, False, None):
                    return False
                tokens = cell.get("input_tokens")
                if type(tokens) is not int or isinstance(tokens, bool) or tokens < 0:
                    return False
                digest = cell.get("answer_digest")
                if digest is not None and (
                    type(digest) is not str
                    or len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest)
                ):
                    return False
                recomputed = bool(
                    digest is not None and digest == expected_digest and cell.get("boundary_found")
                )
                return bool(cell.get("verdict")) == recomputed

            if not _cell_ok(record.get("native")):
                return None
            cells = record.get("cells")
            details = record.get("cell_details")
            if type(cells) is not dict or set(cells) != set(_O_DL_TREATED):
                return None
            if type(details) is not dict or set(details) != set(_O_DL_TREATED):
                return None
            class_detail: str | None = None
            if record["vacuous"]:
                class_detail = "vacuous"
            elif record["expected_blank"]:
                class_detail = "expected_blank"
            elif record["screened"]:
                class_detail = "screened"
            elif record["message_date_bearing"]:
                class_detail = "message_date_bearing"
            for arm_key in _O_DL_TREATED:
                detail = details[arm_key]
                if detail not in _O_DL_DETAILS:
                    return None
                cell = cells[arm_key]
                if class_detail is not None:
                    if cell is not None or detail != class_detail:
                        return None
                    continue
                if detail in (
                    "vacuous",
                    "expected_blank",
                    "screened",
                    "message_date_bearing",
                ):
                    return None
                if (cell is None) != (detail != ""):
                    return None
                if arm_key in dateline.CONTROL_KEYS:
                    fmt = arm_key.removeprefix("nodate_")
                    if not control_evaluable[fmt]:
                        if cell is not None or detail not in (
                            "control_unmatchable",
                            "control_screened",
                        ):
                            return None
                        continue
                    if detail == "control_unmatchable":
                        return None
                    if cell is not None:
                        anchor = cells[f"anchor_{fmt}"]
                        if anchor is not None and (cell["input_tokens"] != anchor["input_tokens"]):
                            return None
                if cell is None:
                    continue
                if not _cell_ok(cell):
                    return None
            fields = {
                "group": record["group"],
                "n_user_messages": record["n_user_messages"],
                "expected_digest": record["expected_digest"],
                "vacuous": record["vacuous"],
                "expected_blank": record["expected_blank"],
                "echo_fixture": record["echo_fixture"],
                "screened": record["screened"],
                "message_date_bearing": record["message_date_bearing"],
            }
            if fixture in reference:
                if reference[fixture] != fields:
                    return None
            else:
                reference[fixture] = fields
        if fixture_ids is None:
            fixture_ids = ids
        elif fixture_ids != ids:
            return None
        raw_records[name] = records
        aggregates[name] = dateline.aggregate_candidate(records)
    if fixture_ids is None:
        return None
    backend_fingerprint = protocol.get("backend_fingerprint")
    if type(backend_fingerprint) is not str:
        return None
    expected_protocol = dict(dateline.protocol_object(fixture_ids, backend_fingerprint))
    expected_protocol["fingerprint"] = dateline.protocol_fingerprint(
        fixture_ids, backend_fingerprint
    )
    if protocol != expected_protocol:
        return None
    primary_decision = value.get("primary_decision")
    primary_winner = value.get("primary_winner")
    if primary_decision in ("compatible_on_fixture_set", "held_out_failed"):
        mapped: tuple[Any, Any] = ("selected", primary_winner)
    else:
        mapped = (primary_decision, primary_winner)
    arms_out: dict[str, Any] = {}
    for arm_key in _O_DL_TREATED:
        completed_sets: list[frozenset[str]] = []
        all_pass: dict[str, bool | None] = {}
        withheld = False
        for cand_name, entry in candidates.items():
            if entry.get("status") != "available":
                withheld = True
                all_pass[cand_name] = None
                continue
            live = dateline.live_records(raw_records[cand_name])
            if arm_key in dateline.CONTROL_KEYS:
                control_format = arm_key.removeprefix("nodate_")
                defined = frozenset(
                    r["fixture"] for r in live if r["control_evaluable"][control_format]
                )
            else:
                defined = frozenset(r["fixture"] for r in live)
            completed = frozenset(r["fixture"] for r in live if r["cells"][arm_key] is not None)
            completed_sets.append(completed)
            if completed != defined or not completed:
                withheld = True
                all_pass[cand_name] = None
                continue
            all_pass[cand_name] = all(
                r["cells"][arm_key]["verdict"] for r in live if r["fixture"] in completed
            )
        if len(set(completed_sets)) > 1:
            withheld = True
        if withheld:
            arms_out[arm_key] = {
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
        arms_out[arm_key] = {
            "decision": decision,
            "winner": winner,
            "withheld": False,
            "detail": "",
            "flip": (decision, winner) != mapped,
        }
    available_records = {
        cand_name: raw_records[cand_name]
        for cand_name, entry in candidates.items()
        if entry.get("status") == "available"
    }
    common_out = dateline.common_population(available_records)
    n_multi_turn = sum(1 for fields in reference.values() if fields["n_user_messages"] > 1)
    return {
        "aggregates": aggregates,
        "arms": arms_out,
        "common_population": common_out,
        "n_candidates": len(candidates),
        "n_search_fixtures": len(fixture_ids),
        "n_multi_turn_fixtures": n_multi_turn,
    }


def _o_dateline_valid(value: Any) -> bool:
    """Oracle dateline validator: mirrors diffreports._valid_dateline."""
    if type(value) is not dict or value.get("available") is not True:
        return False
    for key in ("primary_decision", "primary_winner"):
        item = value.get(key)
        if item is not None and type(item) is not str:
            return False
    recomputed = _o_dateline_recompute(value)
    if recomputed is None:
        return False
    candidates = value["candidates"]
    for name, expected in recomputed["aggregates"].items():
        entry = candidates[name]
        for key, item in expected.items():
            if entry.get(key) != item:
                return False
    if value.get("arms") != recomputed["arms"]:
        return False
    if value.get("common_population") != recomputed["common_population"]:
        return False
    if value.get("n_candidates") != recomputed["n_candidates"]:
        return False
    if value.get("n_search_fixtures") != recomputed["n_search_fixtures"]:
        return False
    return bool(value.get("n_multi_turn_fixtures") == recomputed["n_multi_turn_fixtures"])


def _o_dateline(old: Any, new: Any) -> tuple[dict[str, Any], bool, bool]:
    """Oracle dateline section: mirrors diffreports.diff_dateline."""
    old_ok = _o_dateline_valid(old)
    new_ok = _o_dateline_valid(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    diff: dict[str, Any] = {"available": available, "changed": False}
    asymmetric = available not in ("both", "neither")
    if available != "both":
        return diff, False, asymmetric
    protocol_drift = old["protocol"].get("fingerprint") != new["protocol"].get("fingerprint")
    fixture_set_drift = old["protocol"].get("fixture_set_sha256") != new["protocol"].get(
        "fixture_set_sha256"
    )
    backend_drift = old["protocol"].get("backend_fingerprint") != new["protocol"].get(
        "backend_fingerprint"
    )
    comparable = not protocol_drift and not fixture_set_drift and not backend_drift
    old_candidates = old["candidates"]
    new_candidates = new["candidates"]
    added = sorted(set(new_candidates) - set(old_candidates))
    removed = sorted(set(old_candidates) - set(new_candidates))
    loss_drift: list[dict[str, Any]] = []
    pair_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_candidates) & set(new_candidates)):
        old_entry = old_candidates[name]
        new_entry = new_candidates[name]
        if old_entry.get("status") != "available" or new_entry.get("status") != "available":
            if old_entry.get("status") != new_entry.get("status"):
                loss_drift.append(
                    {
                        "candidate": name,
                        "metric": "status",
                        "old": old_entry.get("status"),
                        "new": new_entry.get("status"),
                    }
                )
            continue
        if old_entry.get("date_content_falsified") != new_entry.get("date_content_falsified"):
            loss_drift.append(
                {
                    "candidate": name,
                    "metric": "date_content_falsified",
                    "old": old_entry.get("date_content_falsified"),
                    "new": new_entry.get("date_content_falsified"),
                }
            )
        for value_key in dateline.VALUE_KEYS:
            old_block = old_entry["date_loss"][value_key]
            new_block = new_entry["date_loss"][value_key]
            if old_block["loss_rate"] != new_block["loss_rate"]:
                loss_drift.append(
                    {
                        "candidate": name,
                        "metric": f"{value_key}_loss_rate",
                        "old": old_block["loss_rate"],
                        "new": new_block["loss_rate"],
                    }
                )
        for fmt in dateline.FORMATS:
            for pair, old_block in old_entry["pairs_by_format"][fmt].items():
                new_block = new_entry["pairs_by_format"][fmt][pair]
                if old_block["classification"] != new_block["classification"]:
                    pair_drift.append(
                        {
                            "candidate": name,
                            "metric": f"{fmt}_{pair}_classification",
                            "old": old_block["classification"],
                            "new": new_block["classification"],
                        }
                    )
    arm_drift: list[dict[str, Any]] = []
    for arm_key in old["arms"]:
        if old["arms"][arm_key] != new["arms"].get(arm_key):
            arm_drift.append(
                {
                    "arm": arm_key,
                    "old": old["arms"][arm_key],
                    "new": new["arms"].get(arm_key),
                }
            )
    diff["loss_drift"] = loss_drift
    diff["pair_drift"] = pair_drift
    diff["arm_drift"] = arm_drift
    diff["candidates_added"] = added
    diff["candidates_removed"] = removed
    diff["protocol_drift"] = protocol_drift
    diff["fixture_set_drift"] = fixture_set_drift
    diff["backend_drift"] = backend_drift
    diff["comparable"] = comparable
    changed = bool(
        loss_drift
        or pair_drift
        or arm_drift
        or added
        or removed
        or protocol_drift
        or fixture_set_drift
        or backend_drift
    )
    diff["changed"] = changed
    return diff, changed, asymmetric


_O_MT_DETAILS = frozenset({"arm_context_exceeded", "arm_evaluation_failed", "arm_screened"})
_O_MT_ARMS = ("full", "sharded", "scripted")
_O_MT_CELL_KEYS = frozenset(
    {"verdict", "boundary_found", "boundary_text_match", "input_tokens", "answer_digest"}
)
_O_MT_HEX = frozenset("0123456789abcdef")


def _o_multiturn_digest_ok(item: Any) -> bool:
    """O multiturn digest ok."""
    return type(item) is str and len(item) == 64 and all(c in _O_MT_HEX for c in item)


def _o_multiturn_cell_ok(
    cell: Any, arm: str, depth: int, expected_digest: str, empty_digest: str
) -> bool:
    """O multiturn cell ok."""
    if type(cell) is not dict:
        return False
    keys = frozenset(cell)
    if arm == "sharded":
        if keys != _O_MT_CELL_KEYS | {
            "intermediate_unbounded",
            "intermediate_empty",
            "turns",
        }:
            return False
    elif keys != _O_MT_CELL_KEYS:
        return False
    if type(cell["verdict"]) is not bool or type(cell["boundary_found"]) is not bool:
        return False
    if cell["boundary_text_match"] not in (True, False, None):
        return False
    if type(cell["input_tokens"]) is not int or cell["input_tokens"] < 0:
        return False
    if not _o_multiturn_digest_ok(cell["answer_digest"]):
        return False
    if cell["verdict"] != bool(cell["answer_digest"] == expected_digest and cell["boundary_found"]):
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
            if not _o_multiturn_digest_ok(entry["answer_digest"]):
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
        if cell["intermediate_unbounded"] != any(not e["boundary_found"] for e in turns[:-1]):
            return False
        if cell["intermediate_empty"] != any(
            e["answer_digest"] == empty_digest for e in turns[:-1]
        ):
            return False
    return True


def _o_multiturn_recompute(value: Any) -> dict[str, Any] | None:
    """Oracle multiturn recompute: mirrors diffreports._multiturn_recompute.

    The oracle rebuilds the protocol and the fixture set through
    the producer module (multiturn) where the implementation
    rebuilds them from literal tables -- a drift in either side's
    constants breaks the agreement the fuzz harness asserts.
    """
    protocol = value.get("protocol")
    if type(protocol) is not dict:
        return None
    seed = protocol.get("seed")
    backend_fp = protocol.get("backend_fingerprint")
    if type(seed) is not int or type(backend_fp) is not str:
        return None
    expected_protocol = dict(multiturn.protocol_object(seed, backend_fp))
    expected_protocol["fingerprint"] = multiturn.protocol_fingerprint(seed, backend_fp)
    if protocol != expected_protocol:
        return None
    candidates = value.get("candidates")
    if type(candidates) is not dict or not candidates:
        return None
    generated = multiturn.generate_fixtures(seed)
    by_id = {f["fixture_id"]: f for f in generated}
    empty_digest = hashlib.sha256(b"").hexdigest()
    total = 0
    records_bytes = 0
    raw: dict[str, list[dict[str, Any]]] = {}
    aggregates: dict[str, Any] = {}
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
        total += len(records)
        records_bytes += len(
            json.dumps(records, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        )
        if records_bytes > multiturn.MAX_RECORDS_BYTES:
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
            if type(cells) is not dict or set(cells) != set(_O_MT_ARMS):
                return None
            if type(details) is not dict or set(details) != set(_O_MT_ARMS):
                return None
            expected_digest = hashlib.sha256(str(fixture["expected"]).encode("utf-8")).hexdigest()
            for arm in _O_MT_ARMS:
                cell = cells[arm]
                detail = details[arm]
                if cell is None:
                    if detail not in _O_MT_DETAILS:
                        return None
                else:
                    if detail != "":
                        return None
                    if not _o_multiturn_cell_ok(
                        cell, arm, int(fixture["depth"]), expected_digest, empty_digest
                    ):
                        return None
        raw[name] = records
        aggregates[name] = multiturn.aggregate_candidate(records)
    if total > multiturn.MAX_GROUND_RECORDS:
        return None
    selections: dict[str, Any] = {}
    for depth in multiturn.DEPTHS:
        selections[str(depth)] = {}
        defined = {f["fixture_id"] for f in generated if f["depth"] == depth}
        for arm in ("full", "sharded"):
            completed_sets: list[frozenset[str]] = []
            all_pass: dict[str, bool | None] = {}
            withheld = False
            for name in candidates:
                completed = {
                    r["fixture"]
                    for r in raw[name]
                    if r["depth"] == depth and type(r["cells"][arm]) is dict
                }
                completed_sets.append(frozenset(completed))
                if completed != defined or not completed:
                    withheld = True
                    all_pass[name] = None
                    continue
                all_pass[name] = all(
                    bool(r["cells"][arm]["verdict"])
                    for r in raw[name]
                    if r["depth"] == depth and r["fixture"] in completed
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


def _o_multiturn_valid(value: Any) -> bool:
    """Oracle multiturn validator: mirrors diffreports._valid_multiturn."""
    if type(value) is not dict or value.get("available") is not True:
        return False
    if set(value) != {
        "available",
        "detail",
        "protocol",
        "candidates",
        "selections",
        "n_candidates",
        "n_fixtures",
    }:
        return False
    if value.get("detail") != "":
        return False
    recomputed = _o_multiturn_recompute(value)
    if recomputed is None:
        return False
    candidates = value["candidates"]
    for name, expected in recomputed["aggregates"].items():
        entry = candidates[name]
        if entry.get("depths") != expected["depths"]:
            return False
        if entry.get("classification") != expected["classification"]:
            return False
    if value.get("selections") != recomputed["selections"]:
        return False
    if value.get("n_candidates") != recomputed["n_candidates"]:
        return False
    return bool(value.get("n_fixtures") == recomputed["n_fixtures"])


def _o_multiturn(old: Any, new: Any) -> tuple[dict[str, Any], bool, bool]:
    """Oracle multiturn section: mirrors diffreports.diff_multiturn."""
    old_ok = _o_multiturn_valid(old)
    new_ok = _o_multiturn_valid(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    diff: dict[str, Any] = {"available": available, "changed": False}
    asymmetric = available not in ("both", "neither")
    if available != "both":
        return diff, False, asymmetric
    protocol_drift = old["protocol"].get("fingerprint") != new["protocol"].get("fingerprint")
    generator_drift = old["protocol"].get("ground_sha256") != new["protocol"].get("ground_sha256")
    backend_drift = old["protocol"].get("backend_fingerprint") != new["protocol"].get(
        "backend_fingerprint"
    )
    comparable = not protocol_drift and not generator_drift and not backend_drift
    old_candidates = old["candidates"]
    new_candidates = new["candidates"]
    added = sorted(set(new_candidates) - set(old_candidates))
    removed = sorted(set(old_candidates) - set(new_candidates))
    classification_drift: list[dict[str, Any]] = []
    depth_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_candidates) & set(new_candidates)):
        old_entry = old_candidates[name]
        new_entry = new_candidates[name]
        if old_entry.get("status") != "available" or new_entry.get("status") != "available":
            if old_entry.get("status") != new_entry.get("status"):
                classification_drift.append(
                    {
                        "candidate": name,
                        "metric": "status",
                        "old": old_entry.get("status"),
                        "new": new_entry.get("status"),
                    }
                )
            continue
        if old_entry.get("classification") != new_entry.get("classification"):
            classification_drift.append(
                {
                    "candidate": name,
                    "metric": "classification",
                    "old": old_entry.get("classification"),
                    "new": new_entry.get("classification"),
                }
            )
        for depth_key in ("2", "3"):
            old_block = old_entry["depths"][depth_key]
            new_block = new_entry["depths"][depth_key]
            if old_block["pass_rates"]["sharded"] != new_block["pass_rates"]["sharded"]:
                depth_drift.append(
                    {
                        "candidate": name,
                        "metric": f"depth_{depth_key}_sharded_rate",
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
                depth_drift.append(
                    {
                        "candidate": name,
                        "metric": f"depth_{depth_key}_self_history_pair",
                        "old": list(old_pair),
                        "new": list(new_pair),
                    }
                )
    selection_drift: list[dict[str, Any]] = []
    for depth_key in ("2", "3"):
        for arm in ("full", "sharded"):
            if old["selections"][depth_key][arm] != new["selections"][depth_key][arm]:
                selection_drift.append(
                    {
                        "depth": depth_key,
                        "arm": arm,
                        "old": old["selections"][depth_key][arm],
                        "new": new["selections"][depth_key][arm],
                    }
                )
    diff["classification_drift"] = classification_drift
    diff["depth_drift"] = depth_drift
    diff["selection_drift"] = selection_drift
    diff["candidates_added"] = added
    diff["candidates_removed"] = removed
    diff["protocol_drift"] = protocol_drift
    diff["generator_drift"] = generator_drift
    diff["backend_drift"] = backend_drift
    diff["comparable"] = comparable
    changed = bool(
        classification_drift
        or depth_drift
        or selection_drift
        or added
        or removed
        or protocol_drift
        or generator_drift
        or backend_drift
    )
    diff["changed"] = changed
    return diff, changed, asymmetric


_O_ND_DETAILS = frozenset(
    {"arm_context_exceeded", "arm_evaluation_failed", "arm_screened", "arm_unmatchable"}
)
_O_ND_CELL_KEYS = frozenset(
    {
        "verdict",
        "boundary_found",
        "boundary_text_match",
        "input_tokens",
        "answer_digest",
        "needle_offsets",
    }
)
_O_ND_HEX = frozenset("0123456789abcdef")


def _o_needle_digest_ok(item: Any) -> bool:
    """O needle digest ok."""
    return type(item) is str and len(item) == 64 and all(c in _O_ND_HEX for c in item)


def _o_needle_expected_offsets(key: str) -> list[int] | None:
    """O needle expected offsets."""
    if key == "dose_0":
        return [0]
    for dose in needle.DOSES:
        if key == needle.cell_key_absent(dose):
            return []
        if key == needle.cell_key_duplicate(dose):
            return [dose * 25 // 100, dose * 75 // 100]
        for depth in needle.DEPTHS:
            if key == needle.cell_key_treated(dose, depth):
                return [needle.target_offset(dose, depth)]
    return None


def _o_needle_cell_ok(cell: Any, expected_digest: str) -> bool:
    """O needle cell ok."""
    if type(cell) is not dict or frozenset(cell) != _O_ND_CELL_KEYS:
        return False
    if type(cell["verdict"]) is not bool or type(cell["boundary_found"]) is not bool:
        return False
    if cell["boundary_text_match"] not in (True, False, None):
        return False
    if type(cell["input_tokens"]) is not int or cell["input_tokens"] < 0:
        return False
    if cell["answer_digest"] is not None and not _o_needle_digest_ok(cell["answer_digest"]):
        return False
    offsets = cell["needle_offsets"]
    if type(offsets) is not list or any(type(o) is not int or o < 0 for o in offsets):
        return False
    return bool(
        cell["verdict"] == bool(cell["answer_digest"] == expected_digest and cell["boundary_found"])
    )


def _o_needle_recompute(value: Any) -> dict[str, Any] | None:
    """Oracle needle recompute: mirrors diffreports._needle_recompute.

    The oracle rebuilds the protocol and the case set through
    the producer module (needle) where the implementation
    rebuilds them from literal tables -- a drift in either
    side's constants breaks the agreement the fuzz harness
    asserts. Records are checked in canonical generation
    order, the order the implementation's validator enforces.
    """
    protocol = value.get("protocol")
    if type(protocol) is not dict:
        return None
    seed = protocol.get("seed")
    backend_fp = protocol.get("backend_fingerprint")
    if type(seed) is not int or type(backend_fp) is not str:
        return None
    expected_protocol = dict(needle.protocol_object(seed, backend_fp))
    expected_protocol["fingerprint"] = needle.protocol_fingerprint(seed, backend_fp)
    if protocol != expected_protocol:
        return None
    candidates = value.get("candidates")
    if type(candidates) is not dict or not candidates:
        return None
    generated = needle.generate_cases(seed)
    total = 0
    records_bytes = 0
    raw: dict[str, list[dict[str, Any]]] = {}
    aggregates: dict[str, Any] = {}
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
        total += len(records)
        records_bytes += len(
            json.dumps(records, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        )
        if records_bytes > needle.MAX_RECORDS_BYTES:
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
            expected_digest = hashlib.sha256(str(case["value"]).encode("utf-8")).hexdigest()
            if record["expected_digest"] != expected_digest:
                return None
            if type(record["question_tokens"]) is not int or record["question_tokens"] < 0:
                return None
            cells = record["cells"]
            details = record["cell_details"]
            if type(cells) is not dict or set(cells) != set(needle.CELL_KEYS):
                return None
            if type(details) is not dict or set(details) != set(needle.CELL_KEYS):
                return None
            for key in needle.CELL_KEYS:
                cell = cells[key]
                detail = details[key]
                if cell is None:
                    if detail not in _O_ND_DETAILS:
                        return None
                else:
                    if detail != "":
                        return None
                    if not _o_needle_cell_ok(cell, expected_digest):
                        return None
                    if cell["needle_offsets"] != _o_needle_expected_offsets(key):
                        return None
            dose0 = cells["dose_0"]
            if dose0 is not None and dose0["input_tokens"] <= record["question_tokens"]:
                return None
            needle_len = (
                dose0["input_tokens"] - record["question_tokens"] if dose0 is not None else None
            )
            for dose in needle.DOSES:
                absent = cells[needle.cell_key_absent(dose)]
                if absent is not None and (
                    absent["input_tokens"] - record["question_tokens"] != dose
                ):
                    return None
                if dose0 is None:
                    continue
                for depth in needle.DEPTHS:
                    treated = cells[needle.cell_key_treated(dose, depth)]
                    if treated is not None and (
                        treated["input_tokens"] - dose0["input_tokens"] != dose
                    ):
                        return None
                duplicate = cells[needle.cell_key_duplicate(dose)]
                if (
                    duplicate is not None
                    and needle_len is not None
                    and (duplicate["input_tokens"] - dose0["input_tokens"] != dose + needle_len)
                ):
                    return None
        raw[name] = records
        aggregates[name] = needle.aggregate_candidate(records)
    if total > needle.MAX_GROUND_RECORDS:
        return None
    return {
        "aggregates": aggregates,
        "readouts": needle.headline_readouts(raw),
        "conditions": {key: needle.cell_selection(raw, key) for key in needle.CELL_KEYS},
        "n_candidates": len(candidates),
        "n_cases": len(generated),
    }


def _o_needle_valid(value: Any) -> bool:
    """Oracle needle validator: mirrors diffreports._valid_needle."""
    if type(value) is not dict or value.get("available") is not True:
        return False
    if set(value) != {
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
    if value.get("detail") != "":
        return False
    recomputed = _o_needle_recompute(value)
    if recomputed is None:
        return False
    candidates = value["candidates"]
    for name, expected in recomputed["aggregates"].items():
        entry = candidates[name]
        if entry.get("dose0_control") != expected["dose0_control"]:
            return False
        if entry.get("false_recall") != expected["false_recall"]:
            return False
        if entry.get("duplicate") != expected["duplicate"]:
            return False
    if value.get("readouts") != recomputed["readouts"]:
        return False
    if value.get("conditions") != recomputed["conditions"]:
        return False
    if value.get("n_candidates") != recomputed["n_candidates"]:
        return False
    return bool(value.get("n_cases") == recomputed["n_cases"])


def _o_needle(old: Any, new: Any) -> tuple[dict[str, Any], bool, bool]:
    """Oracle needle section: mirrors diffreports.diff_needle."""
    old_ok = _o_needle_valid(old)
    new_ok = _o_needle_valid(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    diff: dict[str, Any] = {"available": available, "changed": False}
    asymmetric = available not in ("both", "neither")
    if available != "both":
        return diff, False, asymmetric
    protocol_drift = old["protocol"].get("fingerprint") != new["protocol"].get("fingerprint")
    generator_drift = old["protocol"].get("ground_sha256") != new["protocol"].get("ground_sha256")
    backend_drift = old["protocol"].get("backend_fingerprint") != new["protocol"].get(
        "backend_fingerprint"
    )
    comparable = not protocol_drift and not generator_drift and not backend_drift
    old_candidates = old["candidates"]
    new_candidates = new["candidates"]
    added = sorted(set(new_candidates) - set(old_candidates))
    removed = sorted(set(old_candidates) - set(new_candidates))
    candidate_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_candidates) & set(new_candidates)):
        old_entry = old_candidates[name]
        new_entry = new_candidates[name]
        if old_entry.get("status") != "available" or new_entry.get("status") != "available":
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
        if old_entry["dose0_control"]["rate"] != new_entry["dose0_control"]["rate"]:
            candidate_drift.append(
                {
                    "candidate": name,
                    "metric": "dose0_rate",
                    "old": old_entry["dose0_control"]["rate"],
                    "new": new_entry["dose0_control"]["rate"],
                }
            )
        for dose_key in ("1024", "4096"):
            old_deficit = old["readouts"][dose_key]["deficit"][name]
            new_deficit = new["readouts"][dose_key]["deficit"][name]
            if old_deficit != new_deficit:
                candidate_drift.append(
                    {
                        "candidate": name,
                        "metric": f"dose_{dose_key}_deficit",
                        "old": old_deficit,
                        "new": new_deficit,
                    }
                )
            for depth_key in ("0", "50", "100"):
                old_rate = old["readouts"][dose_key]["curves"][name][depth_key]["rate"]
                new_rate = new["readouts"][dose_key]["curves"][name][depth_key]["rate"]
                if old_rate != new_rate:
                    candidate_drift.append(
                        {
                            "candidate": name,
                            "metric": f"dose_{dose_key}_depth_{depth_key}_rate",
                            "old": old_rate,
                            "new": new_rate,
                        }
                    )
            old_floor = old_entry["false_recall"][dose_key]["rate"]
            new_floor = new_entry["false_recall"][dose_key]["rate"]
            if old_floor != new_floor:
                candidate_drift.append(
                    {
                        "candidate": name,
                        "metric": f"dose_{dose_key}_false_recall",
                        "old": old_floor,
                        "new": new_floor,
                    }
                )
            old_excess = old_entry["duplicate"][dose_key]["excess_over_independence"]
            new_excess = new_entry["duplicate"][dose_key]["excess_over_independence"]
            if old_excess != new_excess:
                candidate_drift.append(
                    {
                        "candidate": name,
                        "metric": f"dose_{dose_key}_duplicate_excess",
                        "old": old_excess,
                        "new": new_excess,
                    }
                )
    condition_drift: list[dict[str, Any]] = []
    for key in needle.CELL_KEYS:
        if old["conditions"][key] != new["conditions"][key]:
            condition_drift.append(
                {
                    "condition": key,
                    "old": old["conditions"][key],
                    "new": new["conditions"][key],
                }
            )
    diff["candidate_drift"] = candidate_drift
    diff["condition_drift"] = condition_drift
    diff["candidates_added"] = added
    diff["candidates_removed"] = removed
    diff["protocol_drift"] = protocol_drift
    diff["generator_drift"] = generator_drift
    diff["backend_drift"] = backend_drift
    diff["comparable"] = comparable
    changed = bool(
        candidate_drift
        or condition_drift
        or added
        or removed
        or protocol_drift
        or generator_drift
        or backend_drift
    )
    diff["changed"] = changed
    return diff, changed, asymmetric


def _o_placement_digest_ok(item: Any) -> bool:
    """O placement digest ok."""
    if item is None:
        return True
    if type(item) is not str or len(item) != 64:
        return False
    return all(ch in "0123456789abcdef" for ch in item)


def _o_placement_cell_ok(cell: Any, expected_digest: str) -> bool:
    """O placement cell ok."""
    if type(cell) is not dict or set(cell) != {
        "verdict",
        "boundary_found",
        "boundary_text_match",
        "input_tokens",
        "answer_digest",
    }:
        return False
    if type(cell["verdict"]) is not bool or type(cell["boundary_found"]) is not bool:
        return False
    if cell["boundary_text_match"] is not None and type(cell["boundary_text_match"]) is not bool:
        return False
    if type(cell["input_tokens"]) is not int or cell["input_tokens"] < 0:
        return False
    if not _o_placement_digest_ok(cell["answer_digest"]):
        return False
    verdict = (cell["answer_digest"] == expected_digest) and cell["boundary_found"]
    return bool(cell["verdict"] == verdict)


def _o_placement_recompute(value: Any) -> dict[str, Any] | None:
    """Oracle placement recompute: mirrors
    diffreports._placement_recompute. The oracle rebuilds the
    protocol and the case set through the producer module
    (placement) where the implementation rebuilds them from
    literal tables -- a drift in either side's constants breaks
    the agreement the fuzz harness asserts.
    """
    import hashlib as _hashlib
    import json as _json

    protocol = value.get("protocol")
    if type(protocol) is not dict:
        return None
    seed = protocol.get("seed")
    backend_fp = protocol.get("backend_fingerprint")
    if type(seed) is not int or isinstance(seed, bool) or type(backend_fp) is not str:
        return None
    expected_protocol = dict(placement.protocol_object(seed, backend_fp))
    expected_protocol["fingerprint"] = placement.protocol_fingerprint(seed, backend_fp)
    if protocol != expected_protocol:
        return None
    candidates = value.get("candidates")
    if type(candidates) is not dict or not candidates:
        return None
    generated = placement.generate_cases(seed)
    total = 0
    records_bytes = 0
    aggregates: dict[str, Any] = {}
    raw: dict[str, list[dict[str, Any]]] = {}
    for name, entry in candidates.items():
        if type(entry) is not dict:
            return None
        if entry.get("status") == "unavailable":
            if set(entry) != {"status", "detail", "records"} or entry.get("records") != []:
                return None
            continue
        if entry.get("status") != "available" or entry.get("detail") != "":
            return None
        if set(entry) != {
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
            "status",
            "detail",
            "records",
        }:
            return None
        records = entry["records"]
        if type(records) is not list or len(records) != len(generated):
            return None
        total += len(records)
        records_bytes += len(
            _json.dumps(records, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        )
        if records_bytes > placement.MAX_RECORDS_BYTES:
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
            expected_digest = _hashlib.sha256(case["value"].encode("utf-8")).hexdigest()
            if record["expected_digest"] != expected_digest:
                return None
            arm_tokens = record["arm_tokens"]
            if type(arm_tokens) is not dict or set(arm_tokens) != set(placement.ARM_KEYS):
                return None
            if any(type(v) is not int or v < 0 for v in arm_tokens.values()):
                return None
            if record["question_tokens"] != arm_tokens["closed_book"]:
                return None
            cells = record["cells"]
            details = record["cell_details"]
            if type(cells) is not dict or set(cells) != set(placement.ARM_KEYS):
                return None
            if type(details) is not dict or set(details) != set(placement.ARM_KEYS):
                return None
            for arm in placement.ARM_KEYS:
                cell = cells[arm]
                detail = details[arm]
                if cell is None:
                    if detail not in placement.CELL_DETAILS:
                        return None
                else:
                    if detail != "":
                        return None
                    if not _o_placement_cell_ok(cell, expected_digest):
                        return None
                    if cell["input_tokens"] != arm_tokens[arm]:
                        return None
        raw[name] = records
        aggregates[name] = placement.aggregate_candidate(records)
    if total > placement.MAX_GROUND_RECORDS:
        return None
    if not raw:
        return None
    return {
        "aggregates": aggregates,
        "conditions": {arm: placement.arm_selection(raw, arm) for arm in placement.LAYOUT_ARMS},
        "n_candidates": len(candidates),
        "n_cases": len(generated),
    }


def _o_placement_valid(value: Any) -> bool:
    """Oracle placement validator: mirrors diffreports._valid_placement."""
    if type(value) is not dict or value.get("available") is not True:
        return False
    if set(value) != {
        "available",
        "detail",
        "protocol",
        "candidates",
        "conditions",
        "n_candidates",
        "n_cases",
    }:
        return False
    if value.get("detail") != "":
        return False
    recomputed = _o_placement_recompute(value)
    if recomputed is None:
        return False
    candidates = value["candidates"]
    for name, expected in recomputed["aggregates"].items():
        entry = candidates[name]
        for key, val in expected.items():
            if entry.get(key) != val:
                return False
    if value.get("conditions") != recomputed["conditions"]:
        return False
    if value.get("n_candidates") != recomputed["n_candidates"]:
        return False
    return bool(value.get("n_cases") == recomputed["n_cases"])


def _o_placement(old: Any, new: Any) -> tuple[dict[str, Any], bool, bool]:
    """Oracle placement section: mirrors diffreports.diff_placement."""
    old_ok = _o_placement_valid(old)
    new_ok = _o_placement_valid(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    diff: dict[str, Any] = {"available": available, "changed": False}
    asymmetric = available not in ("both", "neither")
    if available != "both":
        return diff, False, asymmetric
    protocol_drift = old["protocol"].get("fingerprint") != new["protocol"].get("fingerprint")
    generator_drift = old["protocol"].get("ground_sha256") != new["protocol"].get("ground_sha256")
    backend_drift = old["protocol"].get("backend_fingerprint") != new["protocol"].get(
        "backend_fingerprint"
    )
    comparable = not protocol_drift and not generator_drift and not backend_drift
    old_candidates = old["candidates"]
    new_candidates = new["candidates"]
    added = sorted(set(new_candidates) - set(old_candidates))
    removed = sorted(set(old_candidates) - set(new_candidates))
    candidate_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_candidates) & set(new_candidates)):
        old_entry = old_candidates[name]
        new_entry = new_candidates[name]
        if old_entry.get("status") != "available" or new_entry.get("status") != "available":
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
        metrics = [
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
            ("n_qf_only", old_entry["headline"]["n_qf_only"], new_entry["headline"]["n_qf_only"]),
            ("n_pf_only", old_entry["headline"]["n_pf_only"], new_entry["headline"]["n_pf_only"]),
            ("control_level", old_entry["control_level"], new_entry["control_level"]),
            ("n_screened", old_entry["n_screened"], new_entry["n_screened"]),
        ]
        for metric, old_value, new_value in metrics:
            if old_value != new_value:
                candidate_drift.append(
                    {"candidate": name, "metric": metric, "old": old_value, "new": new_value}
                )
    condition_drift: list[dict[str, Any]] = []
    for arm in placement.LAYOUT_ARMS:
        if old["conditions"][arm] != new["conditions"][arm]:
            condition_drift.append(
                {"condition": arm, "old": old["conditions"][arm], "new": new["conditions"][arm]}
            )
    diff["candidate_drift"] = candidate_drift
    diff["condition_drift"] = condition_drift
    diff["candidates_added"] = added
    diff["candidates_removed"] = removed
    diff["protocol_drift"] = protocol_drift
    diff["generator_drift"] = generator_drift
    diff["backend_drift"] = backend_drift
    diff["comparable"] = comparable
    changed = bool(
        candidate_drift
        or condition_drift
        or added
        or removed
        or protocol_drift
        or generator_drift
        or backend_drift
    )
    diff["changed"] = changed
    return diff, changed, asymmetric


# ---------------------------------------------------------------------------
# Canonical-form oracle (cycle 209): mirrors
# diffreports.diff_canonicalform with an independent
# reimplementation -- frozen tables, generator, aggregates,
# selections, and winner headline are all rebuilt here, never
# imported from the producer or the mirror, so a drift in any
# side's constants breaks the agreement the tests assert.
# ---------------------------------------------------------------------------

_O_CF_TOPICS = (
    "café",
    "naïve",
    "Zürich",
    "crème",
    "señor",
    "façade",
    "résumé",
    "piñata",
    "jalapeño",
    "soufflé",
    "touché",
    "protégé",
    "fiancé",
    "blasé",
    "cliché",
    "exposé",
    "saké",
    "pâté",
    "brûlée",
    "décor",
    "élan",
    "über",
    "schön",
    "tête",
)
_O_CF_WORDS = (
    "alder",
    "basalt",
    "cedar",
    "drift",
    "elm",
    "fern",
    "glade",
    "heath",
    "iris",
    "jade",
    "kelp",
    "loom",
    "moss",
    "nova",
    "ochre",
    "pine",
    "quill",
    "reed",
    "sage",
    "tide",
    "urn",
    "vale",
    "wren",
    "yew",
)
_O_CF_POOL = (
    "The hôtel lobby smells faintly of coffee and old paper.",
    "A forêt trail winds past the lake before sunrise.",
    "The côte road curves above the harbour all morning.",
    "The fête banners hang above the market stalls at dusk.",
    "A silver locket with a fête ribbon rests on the wooden dresser.",
    "Letters from his mère fill the shallow drawer.",
    "The théâtre posters cover the wall beside the ticket desk.",
    "A matinée crowd gathers outside the cinéma doors.",
    "The musée catalogue lists every painting by number.",
    "Rain taps the lycée windows above the courtyard all day.",
    "The opéra programme rests on top of the piano.",
    "A séance of quiet reading fills the long afternoon.",
    "The Noël market smells of pine and roasted chestnuts.",
    "Dust gathers on the crèche figures in the attic box.",
)
_O_CF_FACT = "The {topic} inspection code is {value}."
_O_CF_QUESTION = "What is the {topic} inspection code?"
_O_CF_ARM_KEYS = ("nfc", "nfd")
_O_CF_BASELINE = "nfc"
_O_CF_CASE_COUNT = 24
_O_CF_MIN_PAIRED = 6
_O_CF_MIN_DISCORDANT = 2
_O_CF_MAX_RECORDS = 24 * 24
_O_CF_BUDGET = 2 * _O_CF_MAX_RECORDS
_O_CF_MAX_BYTES = 12 * 1024 * 1024
_O_CF_CELL_FIELDS = {
    "verdict",
    "boundary_found",
    "boundary_text_match",
    "input_tokens",
    "answer_digest",
}
_O_CF_DETAILS = ("arm_context_exceeded", "arm_evaluation_failed")
_O_CF_AGGREGATE_KEYS = {
    "arms",
    "ids",
    "headline",
    "invariance_falsified",
    "n_cases",
    "n_incomplete",
}


def _o_cf_digest(text: str) -> str:
    """O cf digest."""
    import hashlib as _hashlib

    return _hashlib.sha256(text.encode("utf-8")).hexdigest()


def _o_cf_digest_ok(value: Any) -> bool:
    """O cf digest ok."""
    if value is None:
        return True
    if type(value) is not str or len(value) != 64:
        return False
    return all(ch in "0123456789abcdef" for ch in value)


def _o_cf_derive(seed: int, index: int, slot: int, counter: int) -> int:
    """O cf derive."""
    import hashlib as _hashlib

    material = f"{seed}|canonicalform|{index}|{slot}|{counter}"
    return int.from_bytes(_hashlib.sha256(material.encode("ascii")).digest()[:8], "big")


def _o_cf_generate(seed: int) -> list[dict[str, Any]]:
    """O cf generate."""
    import unicodedata as _unicodedata

    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index in range(_O_CF_CASE_COUNT):
        topic = _O_CF_TOPICS[index]
        counter = 0
        while True:
            word = _O_CF_WORDS[_o_cf_derive(seed, index, 0, counter) % len(_O_CF_WORDS)]
            number = 1000 + _o_cf_derive(seed, index, 1, counter) % 9000
            value = f"{word}-{number}"
            if value not in seen:
                break
            counter += 1
        seen.add(value)
        first = _o_cf_derive(seed, index, 2, 0) % len(_O_CF_POOL)
        second = _o_cf_derive(seed, index, 3, 0) % len(_O_CF_POOL)
        if second == first:
            second = (first + 1) % len(_O_CF_POOL)
        fact = _O_CF_FACT.format(topic=topic, value=value)
        payload = f"{_O_CF_POOL[first]} {fact} {_O_CF_POOL[second]}"
        content = f"{payload} {_O_CF_QUESTION.format(topic=topic)}"
        cases.append(
            {
                "fixture_id": f"canonicalform-{index:02d}",
                "topic": topic,
                "value": value,
                "sentence_indices": [first, second],
                "content": content,
                "nfc_content": _unicodedata.normalize("NFC", content),
                "nfd_content": _unicodedata.normalize("NFD", content),
            }
        )
    return cases


def _o_cf_case_material_sha256() -> str:
    """O cf case material sha256."""
    import json as _json

    canonical = _json.dumps(
        [
            _O_CF_FACT,
            _O_CF_QUESTION,
            list(_O_CF_TOPICS),
            list(_O_CF_WORDS),
            list(_O_CF_POOL),
        ],
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return _o_cf_digest(canonical)


def _o_cf_ground_sha256(seed: int) -> str:
    """O cf ground sha256."""
    import json as _json

    rows = [
        [
            case["fixture_id"],
            case["topic"],
            case["value"],
            list(case["sentence_indices"]),
            _o_cf_digest(case["nfc_content"]),
            _o_cf_digest(case["nfd_content"]),
        ]
        for case in _o_cf_generate(seed)
    ]
    canonical = _json.dumps(rows, separators=(",", ":"), ensure_ascii=True)
    return _o_cf_digest(canonical)


def _o_cf_protocol(seed: int, backend_fingerprint: str) -> dict[str, Any]:
    """O cf protocol."""
    import json as _json

    protocol: dict[str, Any] = {
        "version": 1,
        "arm_keys": list(_O_CF_ARM_KEYS),
        "baseline_arm": _O_CF_BASELINE,
        "case_count": _O_CF_CASE_COUNT,
        "case_material_sha256": _o_cf_case_material_sha256(),
        "ground_sha256": _o_cf_ground_sha256(seed),
        "min_paired_evaluable": _O_CF_MIN_PAIRED,
        "min_discordant": _O_CF_MIN_DISCORDANT,
        "generation_budget": _O_CF_BUDGET,
        "max_ground_records": _O_CF_MAX_RECORDS,
        "max_records_bytes": _O_CF_MAX_BYTES,
        "seed": seed,
        "backend_fingerprint": backend_fingerprint,
    }
    canonical = _json.dumps(protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    protocol["fingerprint"] = _o_cf_digest(canonical)
    return protocol


def _o_cf_mcnemar_p(a_only: int, b_only: int) -> float | None:
    """O cf mcnemar p."""
    import math as _math

    n = a_only + b_only
    if n == 0:
        return None
    k = min(a_only, b_only)
    tail = sum(_math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def _o_cf_cell_get(record: dict[str, Any], key: str) -> dict[str, Any] | None:
    """O cf cell get."""
    cells = record.get("cells")
    if not isinstance(cells, dict):
        return None
    cell = cells.get(key)
    return cell if isinstance(cell, dict) else None


def _o_cf_verdict(record: dict[str, Any], key: str) -> bool:
    """O cf verdict."""
    cell = _o_cf_cell_get(record, key)
    return bool(cell is not None and cell["verdict"])


def _o_cf_pair_block(records: list[dict[str, Any]]) -> dict[str, Any]:
    """O cf pair block."""
    n_paired = 0
    n_nfc_only = 0
    n_nfd_only = 0
    n_both = 0
    n_pass_nfc = 0
    n_pass_nfd = 0
    for record in records:
        if record.get("ids_identical") is not False:
            continue
        cell_nfc = _o_cf_cell_get(record, "nfc")
        cell_nfd = _o_cf_cell_get(record, "nfd")
        if cell_nfc is None or cell_nfd is None:
            continue
        n_paired += 1
        verdict_nfc = bool(cell_nfc["verdict"])
        verdict_nfd = bool(cell_nfd["verdict"])
        if verdict_nfc:
            n_pass_nfc += 1
        if verdict_nfd:
            n_pass_nfd += 1
        if verdict_nfc and verdict_nfd:
            n_both += 1
        elif verdict_nfc:
            n_nfc_only += 1
        elif verdict_nfd:
            n_nfd_only += 1
    return {
        "n_paired": n_paired,
        "n_nfc_only": n_nfc_only,
        "n_nfd_only": n_nfd_only,
        "n_both": n_both,
        "n_neither": n_paired - n_nfc_only - n_nfd_only - n_both,
        "rate_nfc": (n_pass_nfc / n_paired) if n_paired else None,
        "rate_nfd": (n_pass_nfd / n_paired) if n_paired else None,
        "mcnemar_p": _o_cf_mcnemar_p(n_nfc_only, n_nfd_only),
    }


def _o_cf_classify(n_paired: int, nfc_only: int, nfd_only: int) -> str:
    """O cf classify."""
    d = nfc_only + nfd_only
    if n_paired < _O_CF_MIN_PAIRED:
        return "insufficient"
    if d == 0:
        return "no_discordance_observed"
    if nfc_only == nfd_only and nfc_only >= _O_CF_MIN_DISCORDANT:
        return "discordant_balanced"
    if max(nfc_only, nfd_only) >= _O_CF_MIN_DISCORDANT:
        return "discordant_asymmetric"
    return "below_threshold"


def _o_cf_falsified(classification: str) -> bool:
    """O cf falsified."""
    return classification in ("discordant_balanced", "discordant_asymmetric")


def _o_cf_aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """O cf aggregate."""
    arms: dict[str, Any] = {}
    for key in _O_CF_ARM_KEYS:
        n_completed = 0
        n_pass = 0
        for record in records:
            cell = _o_cf_cell_get(record, key)
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
    n_ids_identical = 0
    n_ids_divergent = 0
    deltas: list[int] = []
    for record in records:
        if record.get("ids_identical") is True:
            n_ids_identical += 1
        elif record.get("ids_identical") is False:
            n_ids_divergent += 1
        delta = record.get("id_length_delta")
        deltas.append(delta if type(delta) is int else 0)
    n_same_ids_flips = 0
    for record in records:
        if record.get("ids_identical") is not True:
            continue
        cell_nfc = _o_cf_cell_get(record, "nfc")
        cell_nfd = _o_cf_cell_get(record, "nfd")
        if cell_nfc is None or cell_nfd is None:
            continue
        if bool(cell_nfc["verdict"]) != bool(cell_nfd["verdict"]):
            n_same_ids_flips += 1
    headline_raw = _o_cf_pair_block(records)
    classification = _o_cf_classify(
        headline_raw["n_paired"], headline_raw["n_nfc_only"], headline_raw["n_nfd_only"]
    )
    headline = {
        "n_paired": headline_raw["n_paired"],
        "n_nfc_only": headline_raw["n_nfc_only"],
        "n_nfd_only": headline_raw["n_nfd_only"],
        "n_both": headline_raw["n_both"],
        "n_neither": headline_raw["n_neither"],
        "rate_nfc": headline_raw["rate_nfc"],
        "rate_nfd": headline_raw["rate_nfd"],
        "mcnemar_p": headline_raw["mcnemar_p"],
        "classification": classification,
    }
    return {
        "arms": arms,
        "ids": {
            "n_ids_identical": n_ids_identical,
            "n_ids_divergent": n_ids_divergent,
            "n_same_ids_verdict_flips": n_same_ids_flips,
            "id_length_delta_by_case": deltas,
            "max_abs_id_delta": max((abs(d) for d in deltas), default=0),
        },
        "headline": headline,
        "invariance_falsified": _o_cf_falsified(classification),
        "n_cases": len(records),
        "n_incomplete": sum(
            1
            for record in records
            if _o_cf_cell_get(record, "nfc") is None or _o_cf_cell_get(record, "nfd") is None
        ),
    }


def _o_cf_shared(records_by_candidate: dict[str, list[dict[str, Any]]]) -> frozenset[str]:
    """O cf shared."""
    populations: list[frozenset[str]] = []
    for records in records_by_candidate.values():
        populations.append(frozenset(str(r.get("fixture")) for r in records))
    if not populations:
        return frozenset()
    return frozenset.intersection(*populations)


def _o_cf_selection_over(
    records_by_candidate: dict[str, list[dict[str, Any]]],
    key: str,
    fixtures: frozenset[str],
) -> dict[str, Any] | None:
    """O cf selection over."""
    completed_sets: list[frozenset[str]] = []
    all_pass: dict[str, bool | None] = {}
    for name, records in records_by_candidate.items():
        completed = frozenset(
            str(record["fixture"])
            for record in records
            if record["fixture"] in fixtures and _o_cf_cell_get(record, key) is not None
        )
        completed_sets.append(completed)
        if not completed:
            return None
        all_pass[name] = all(
            _o_cf_verdict(record, key) for record in records if record["fixture"] in completed
        )
    if len(set(completed_sets)) > 1:
        return None
    passers = [name for name, ok in all_pass.items() if ok]
    if len(passers) == 1:
        return {"decision": "selected", "winner": passers[0]}
    if len(passers) > 1:
        return {"decision": "ambiguous", "winner": None}
    return {"decision": "no_candidate", "winner": None}


def _o_cf_arm_selection(
    records_by_candidate: dict[str, list[dict[str, Any]]],
    arm: str,
) -> dict[str, Any]:
    """O cf arm selection."""
    shared = _o_cf_shared(records_by_candidate)
    own = _o_cf_selection_over(records_by_candidate, arm, shared)
    if own is None:
        return {
            "decision": "unavailable",
            "detail": "selection_withheld_mismatched_sets",
            "winner": None,
            "withheld": True,
            "flip": None,
        }
    if arm == _O_CF_BASELINE:
        flip: bool | None = False
    else:
        completed = frozenset(
            str(record["fixture"])
            for records in records_by_candidate.values()
            for record in records
            if record["fixture"] in shared and _o_cf_cell_get(record, arm) is not None
        )
        baseline = _o_cf_selection_over(records_by_candidate, _O_CF_BASELINE, completed)
        flip = None if baseline is None else own != baseline
    return {
        "decision": own["decision"],
        "detail": "",
        "winner": own["winner"],
        "withheld": False,
        "flip": flip,
    }


def _o_cf_winner_headline(conditions: dict[str, Any]) -> dict[str, Any]:
    """O cf winner headline."""
    nfc = conditions.get("nfc") if isinstance(conditions, dict) else None
    nfd = conditions.get("nfd") if isinstance(conditions, dict) else None
    if not isinstance(nfc, dict) or not isinstance(nfd, dict):
        return {
            "headline": "winner_undefined",
            "detail": "selection_missing",
            "winner_changed": None,
        }
    if nfc.get("withheld") is True or nfd.get("withheld") is True:
        return {
            "headline": "winner_undefined",
            "detail": "selection_withheld_mismatched_sets",
            "winner_changed": None,
        }
    nfc_selected = nfc.get("decision") == "selected"
    nfd_selected = nfd.get("decision") == "selected"
    if nfc_selected and nfd_selected:
        if nfc.get("winner") != nfd.get("winner"):
            return {"headline": "winner_changed", "detail": "", "winner_changed": True}
        return {"headline": "winner_stable", "detail": "", "winner_changed": False}
    if nfc_selected != nfd_selected:
        return {
            "headline": "winner_abstention_asymmetric",
            "detail": "",
            "winner_changed": False,
        }
    return {"headline": "winner_undefined", "detail": "", "winner_changed": False}


def _o_cf_cell_ok(cell: Any, expected_digest: str) -> bool:
    """O cf cell ok."""
    if type(cell) is not dict or set(cell) != _O_CF_CELL_FIELDS:
        return False
    if type(cell["verdict"]) is not bool or type(cell["boundary_found"]) is not bool:
        return False
    if cell["boundary_text_match"] is not None and type(cell["boundary_text_match"]) is not bool:
        return False
    if type(cell["input_tokens"]) is not int or cell["input_tokens"] < 0:
        return False
    if not _o_cf_digest_ok(cell["answer_digest"]):
        return False
    verdict = (cell["answer_digest"] == expected_digest) and cell["boundary_found"]
    return bool(cell["verdict"] == verdict)


def _o_cf_recompute(payload: dict[str, Any]) -> dict[str, Any] | None:
    """O cf recompute."""
    import json as _json

    protocol = payload.get("protocol")
    if type(protocol) is not dict:
        return None
    seed = protocol.get("seed")
    backend_fingerprint = protocol.get("backend_fingerprint")
    if type(seed) is not int or isinstance(seed, bool) or type(backend_fingerprint) is not str:
        return None
    if protocol != _o_cf_protocol(seed, backend_fingerprint):
        return None
    candidates = payload.get("candidates")
    if type(candidates) is not dict or not candidates:
        return None
    generated = _o_cf_generate(seed)
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
        if set(entry) != _O_CF_AGGREGATE_KEYS | {"status", "detail", "records"}:
            return None
        records = entry["records"]
        if type(records) is not list or len(records) != len(generated):
            return None
        total_records += len(records)
        records_bytes += len(
            _json.dumps(records, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        )
        if records_bytes > _O_CF_MAX_BYTES:
            return None
        for record, case in zip(records, generated, strict=True):
            if type(record) is not dict or set(record) != {
                "fixture",
                "topic",
                "expected_digest",
                "nfc_tokens",
                "nfd_tokens",
                "ids_identical",
                "id_length_delta",
                "nfc_content_digest",
                "nfd_content_digest",
                "cells",
                "cell_details",
            }:
                return None
            if record["fixture"] != case["fixture_id"] or record["topic"] != case["topic"]:
                return None
            expected_digest = _o_cf_digest(str(case["value"]))
            if record["expected_digest"] != expected_digest:
                return None
            if record["nfc_content_digest"] != _o_cf_digest(case["nfc_content"]):
                return None
            if record["nfd_content_digest"] != _o_cf_digest(case["nfd_content"]):
                return None
            if record["nfc_content_digest"] == record["nfd_content_digest"]:
                return None
            nfc_tokens = record["nfc_tokens"]
            nfd_tokens = record["nfd_tokens"]
            if type(nfc_tokens) is not int or nfc_tokens < 0:
                return None
            if type(nfd_tokens) is not int or nfd_tokens < 0:
                return None
            if type(record["ids_identical"]) is not bool:
                return None
            if record["ids_identical"] and nfc_tokens != nfd_tokens:
                return None
            if record["id_length_delta"] != nfc_tokens - nfd_tokens:
                return None
            cells = record["cells"]
            details = record["cell_details"]
            if type(cells) is not dict or set(cells) != set(_O_CF_ARM_KEYS):
                return None
            if type(details) is not dict or set(details) != set(_O_CF_ARM_KEYS):
                return None
            for arm in _O_CF_ARM_KEYS:
                cell = cells[arm]
                detail = details[arm]
                if cell is None:
                    if detail not in _O_CF_DETAILS:
                        return None
                else:
                    if detail != "":
                        return None
                    if not _o_cf_cell_ok(cell, expected_digest):
                        return None
                    arm_tokens = nfc_tokens if arm == "nfc" else nfd_tokens
                    if cell["input_tokens"] != arm_tokens:
                        return None
        raw_records[name] = records
        aggregates[name] = _o_cf_aggregate(records)
    if total_records > _O_CF_MAX_RECORDS:
        return None
    if not raw_records:
        return None
    conditions = {arm: _o_cf_arm_selection(raw_records, arm) for arm in _O_CF_ARM_KEYS}
    return {
        "aggregates": aggregates,
        "conditions": conditions,
        "selection": {
            "nfc": conditions["nfc"],
            "nfd": conditions["nfd"],
            "winner": _o_cf_winner_headline(conditions),
        },
        "n_candidates": len(candidates),
        "n_cases": len(generated),
    }


def _o_canonicalform_valid(value: Any) -> bool:
    """Oracle canonical-form validator: mirrors
    diffreports._valid_canonicalform.
    """
    if type(value) is not dict or value.get("available") is not True:
        return False
    if set(value) != {
        "available",
        "detail",
        "protocol",
        "candidates",
        "conditions",
        "selection",
        "n_candidates",
        "n_cases",
    }:
        return False
    if value.get("detail") != "":
        return False
    recomputed = _o_cf_recompute(value)
    if recomputed is None:
        return False
    candidates = value["candidates"]
    for name, expected in recomputed["aggregates"].items():
        entry = candidates[name]
        for key, val in expected.items():
            if entry.get(key) != val:
                return False
    if value.get("conditions") != recomputed["conditions"]:
        return False
    if value.get("selection") != recomputed["selection"]:
        return False
    if value.get("n_candidates") != recomputed["n_candidates"]:
        return False
    return bool(value.get("n_cases") == recomputed["n_cases"])


def _o_canonicalform(old: Any, new: Any) -> tuple[dict[str, Any], bool, bool]:
    """Oracle canonical-form section: mirrors
    diffreports.diff_canonicalform.
    """
    old_ok = _o_canonicalform_valid(old)
    new_ok = _o_canonicalform_valid(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    diff: dict[str, Any] = {"available": available, "changed": False}
    asymmetric = available not in ("both", "neither")
    if available != "both":
        return diff, False, asymmetric
    protocol_drift = old["protocol"].get("fingerprint") != new["protocol"].get("fingerprint")
    generator_drift = old["protocol"].get("ground_sha256") != new["protocol"].get("ground_sha256")
    backend_drift = old["protocol"].get("backend_fingerprint") != new["protocol"].get(
        "backend_fingerprint"
    )
    comparable = not protocol_drift and not generator_drift and not backend_drift
    old_candidates = old["candidates"]
    new_candidates = new["candidates"]
    added = sorted(set(new_candidates) - set(old_candidates))
    removed = sorted(set(old_candidates) - set(new_candidates))
    candidate_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_candidates) & set(new_candidates)):
        old_entry = old_candidates[name]
        new_entry = new_candidates[name]
        if old_entry.get("status") != "available" or new_entry.get("status") != "available":
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
        metrics = [
            (
                "classification",
                old_entry["headline"]["classification"],
                new_entry["headline"]["classification"],
            ),
            (
                "invariance_falsified",
                old_entry["invariance_falsified"],
                new_entry["invariance_falsified"],
            ),
            (
                "n_nfc_only",
                old_entry["headline"]["n_nfc_only"],
                new_entry["headline"]["n_nfc_only"],
            ),
            (
                "n_nfd_only",
                old_entry["headline"]["n_nfd_only"],
                new_entry["headline"]["n_nfd_only"],
            ),
            (
                "n_ids_identical",
                old_entry["ids"]["n_ids_identical"],
                new_entry["ids"]["n_ids_identical"],
            ),
            (
                "n_same_ids_verdict_flips",
                old_entry["ids"]["n_same_ids_verdict_flips"],
                new_entry["ids"]["n_same_ids_verdict_flips"],
            ),
        ]
        for metric, old_value, new_value in metrics:
            if old_value != new_value:
                candidate_drift.append(
                    {"candidate": name, "metric": metric, "old": old_value, "new": new_value}
                )
    condition_drift: list[dict[str, Any]] = []
    for arm in _O_CF_ARM_KEYS:
        if old["conditions"][arm] != new["conditions"][arm]:
            condition_drift.append(
                {"condition": arm, "old": old["conditions"][arm], "new": new["conditions"][arm]}
            )
    selection_drift: list[dict[str, Any]] = []
    if old["selection"] != new["selection"]:
        selection_drift.append(
            {
                "selection": "winner_headline",
                "old": old["selection"]["winner"],
                "new": new["selection"]["winner"],
            }
        )
    diff["candidate_drift"] = candidate_drift
    diff["condition_drift"] = condition_drift
    diff["selection_drift"] = selection_drift
    diff["candidates_added"] = added
    diff["candidates_removed"] = removed
    diff["protocol_drift"] = protocol_drift
    diff["generator_drift"] = generator_drift
    diff["backend_drift"] = backend_drift
    diff["comparable"] = comparable
    changed = bool(
        candidate_drift
        or condition_drift
        or selection_drift
        or added
        or removed
        or protocol_drift
        or generator_drift
        or backend_drift
    )
    diff["changed"] = changed
    return diff, changed, asymmetric


# ---------------------------------------------------------------------------
# Cycle 218: instruction-distance section oracle -- mirrors
# diffreports.diff_instruction_distance with an independent
# reimplementation -- frozen tables, generator, decomposition,
# aggregates, and decisions are all rebuilt here, never
# imported from the producer or the mirror, so a drift in any
# side's constants breaks the agreement the tests assert.
# ---------------------------------------------------------------------------

_O_ID_ARM_KEYS = (
    "adj",
    "sep_256",
    "sep_1024",
    "disp_256",
    "disp_1024",
    "adj_after",
)
_O_ID_SIZED_ARMS = ("sep_256", "sep_1024", "disp_256", "disp_1024")
_O_ID_DOSES = (256, 1024)
_O_ID_CASE_COUNT = 24
_O_ID_CONSTRAINT = "Reply with only the number: no units, no sentence, nothing else."
_O_ID_FACT = "The {topic} number is {value}."
_O_ID_QUESTION = "What is the {topic} number?"
_O_ID_TOPICS = (
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
_O_ID_MIN_D = 10
_O_ID_MIN_S = 6
_O_ID_MIN_MARGIN = 5
_O_ID_MAX_RECORDS = 24 * _O_ID_CASE_COUNT
_O_ID_BUDGET = len(_O_ID_ARM_KEYS) * _O_ID_MAX_RECORDS
_O_ID_MAX_BYTES = 12 * 1024 * 1024
_O_ID_SIZING_BUDGET = 512
_O_ID_STREAM_CYCLES = 6
_O_ID_CELL_FIELDS = {
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
_O_ID_DETAILS = (
    "arm_context_exceeded",
    "arm_evaluation_failed",
    "arm_unmatchable",
    "arm_screened",
)
_O_ID_UNSIZED_DETAILS = ("arm_context_exceeded", "arm_evaluation_failed")
_O_ID_AGGREGATE_KEYS = {
    "arms",
    "doses",
    "ids",
    "operating_characteristics",
    "n_cases",
    "n_incomplete",
}
_O_ID_SIZING_RULE_TEXT = (
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
_O_ID_SCREENING_RULE_TEXT = (
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
_O_ID_OC: dict[str, Any] = {
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
_O_ID_CLASSES = (
    "compliant_correct",
    "compliant_wrong",
    "noncompliant_correct_value",
    "noncompliant_wrong",
    "boundary_only",
)
_O_ID_NONCOMPLIANT = ("noncompliant_correct_value", "noncompliant_wrong")
_O_ID_COMPLIANT = ("compliant_correct", "compliant_wrong")


def _o_id_digest(text: str) -> str:
    """O id digest."""
    import hashlib as _hashlib

    return _hashlib.sha256(text.encode("utf-8")).hexdigest()


def _o_id_digest_ok(value: Any) -> bool:
    """O id digest ok."""
    if type(value) is not str or len(value) != 64:
        return False
    return all(ch in "0123456789abcdef" for ch in value)


def _o_id_run_digest_ok(value: Any) -> bool:
    """O id run digest ok."""
    if value == "":
        return True
    return _o_id_digest_ok(value)


def _o_id_derive(seed: int, index: int, slot: int, counter: int) -> int:
    """O id derive."""
    import hashlib as _hashlib

    material = f"{seed}|instructiondistance|{index}|{slot}|{counter}"
    return int.from_bytes(_hashlib.sha256(material.encode("ascii")).digest()[:8], "big")


def _o_id_generate(seed: int) -> list[dict[str, Any]]:
    """O id generate."""
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index in range(_O_ID_CASE_COUNT):
        topic = _O_ID_TOPICS[index]
        counter = 0
        while True:
            value = str(100 + _o_id_derive(seed, index, 0, counter) % 900)
            if value not in seen:
                break
            counter += 1
        seen.add(value)
        cases.append(
            {"fixture_id": f"instructiondistance-{index:02d}", "topic": topic, "value": value}
        )
    return cases


def _o_id_case_material_sha256() -> str:
    """O id case material sha256."""
    import json as _json

    canonical = _json.dumps(
        [_O_ID_CONSTRAINT, _O_ID_FACT, _O_ID_QUESTION, list(_O_ID_TOPICS)],
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return _o_id_digest(canonical)


def _o_id_ground_sha256(seed: int) -> str:
    """O id ground sha256."""
    import json as _json

    rows = [[case["fixture_id"], case["topic"], case["value"]] for case in _o_id_generate(seed)]
    canonical = _json.dumps(rows, separators=(",", ":"), ensure_ascii=True)
    return _o_id_digest(canonical)


def _o_id_protocol(seed: int, backend_fingerprint: str) -> dict[str, Any]:
    """O id protocol."""
    import json as _json

    protocol: dict[str, Any] = {
        "version": 1,
        "arm_keys": list(_O_ID_ARM_KEYS),
        "doses": list(_O_ID_DOSES),
        "baseline_arm": "adj",
        "case_count": _O_ID_CASE_COUNT,
        "case_material_sha256": _o_id_case_material_sha256(),
        "ground_sha256": _o_id_ground_sha256(seed),
        "sizing_rule_sha256": _o_id_digest(_O_ID_SIZING_RULE_TEXT),
        "sizing_encode_budget": _O_ID_SIZING_BUDGET,
        "screening_rule_sha256": _o_id_digest(_O_ID_SCREENING_RULE_TEXT),
        "stream_cycles": _O_ID_STREAM_CYCLES,
        "min_d": _O_ID_MIN_D,
        "min_s": _O_ID_MIN_S,
        "min_margin": _O_ID_MIN_MARGIN,
        "generation_budget": _O_ID_BUDGET,
        "max_ground_records": _O_ID_MAX_RECORDS,
        "max_records_bytes": _O_ID_MAX_BYTES,
        "seed": seed,
        "backend_fingerprint": backend_fingerprint,
    }
    canonical = _json.dumps(protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    protocol["fingerprint"] = _o_id_digest(canonical)
    return protocol


def _o_id_cell_get(record: dict[str, Any], key: str) -> dict[str, Any] | None:
    """O id cell get."""
    cells = record.get("cells")
    if not isinstance(cells, dict):
        return None
    cell = cells.get(key)
    return cell if isinstance(cell, dict) else None


def _o_id_decompose(cell: dict[str, Any], expected_digest: str) -> str:
    """O id decompose."""
    empty = _o_id_digest("")
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


def _o_id_classify(direct_passes: int, separated: int, displaced: int) -> str:
    """O id classify."""
    if direct_passes < _O_ID_MIN_D:
        return "insufficient"
    if separated >= _O_ID_MIN_S and separated - displaced >= _O_ID_MIN_MARGIN:
        return "instruction_distance_drop_observed"
    return "below_threshold"


def _o_id_aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """O id aggregate."""
    arms: dict[str, Any] = {}
    for key in _O_ID_ARM_KEYS:
        n_completed = 0
        n_pass = 0
        n_compliant = 0
        counts = {
            "arm_context_exceeded": 0,
            "arm_evaluation_failed": 0,
            "arm_unmatchable": 0,
            "arm_screened": 0,
        }
        decomposition: dict[str, int] = dict.fromkeys(_O_ID_CLASSES, 0)
        for record in records:
            details = record.get("cell_details")
            detail = details.get(key, "") if isinstance(details, dict) else ""
            if detail in counts:
                counts[detail] += 1
            cell = _o_id_cell_get(record, key)
            if cell is None:
                continue
            n_completed += 1
            if cell["verdict"]:
                n_pass += 1
            cls = _o_id_decompose(cell, str(record.get("expected_digest", "")))
            decomposition[cls] = decomposition.get(cls, 0) + 1
            if cls in _O_ID_COMPLIANT:
                n_compliant += 1
        arms[key] = {
            "n_completed": n_completed,
            "n_pass": n_pass,
            "rate": (n_pass / n_completed) if n_completed else None,
            "n_compliant": n_compliant,
            "compliance_rate": (n_compliant / n_completed) if n_completed else None,
            "n_context_exceeded": counts["arm_context_exceeded"],
            "n_evaluation_failed": counts["arm_evaluation_failed"],
            "n_unmatchable": counts["arm_unmatchable"],
            "n_screened": counts["arm_screened"],
            "decomposition": decomposition,
        }
    doses: dict[str, Any] = {}
    for dose in _O_ID_DOSES:
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
            adj = _o_id_cell_get(record, "adj")
            sep_cell = _o_id_cell_get(record, sep)
            disp_cell = _o_id_cell_get(record, disp)
            if (
                adj is not None
                and sep_cell is not None
                and (
                    _o_id_decompose(adj, expected_digest) in _O_ID_NONCOMPLIANT
                    and _o_id_decompose(sep_cell, expected_digest) in _O_ID_COMPLIANT
                )
            ):
                n_reverse += 1
            if adj is None or sep_cell is None or disp_cell is None:
                continue
            n_population += 1
            if _o_id_decompose(adj, expected_digest) != "compliant_correct":
                continue
            direct_passes += 1
            sep_cls = _o_id_decompose(sep_cell, expected_digest)
            if sep_cls in _O_ID_NONCOMPLIANT:
                separated_losses += 1
                if sep_cls == "noncompliant_correct_value":
                    s_value_kept += 1
                else:
                    s_value_lost += 1
            if _o_id_decompose(disp_cell, expected_digest) in _O_ID_NONCOMPLIANT:
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
            "decision": _o_id_classify(direct_passes, separated_losses, displaced_losses),
        }
    ids: dict[str, Any] = {}
    for key in _O_ID_ARM_KEYS:
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
                adj_cell = _o_id_cell_get(record, "adj")
                other_cell = _o_id_cell_get(record, key)
                if adj_cell is not None and other_cell is not None:
                    expected_digest = str(record.get("expected_digest", ""))
                    if (_o_id_decompose(adj_cell, expected_digest) in _O_ID_COMPLIANT) != (
                        _o_id_decompose(other_cell, expected_digest) in _O_ID_COMPLIANT
                    ):
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
        "operating_characteristics": dict(_O_ID_OC),
        "n_cases": len(records),
        "n_incomplete": sum(
            1
            for record in records
            if any(
                _o_id_cell_get(record, arm) is None
                for arm in ("adj", "sep_256", "sep_1024", "disp_256", "disp_1024")
            )
        ),
    }


def _o_id_cell_ok(cell: Any, expected_digest: str) -> bool:
    """O id cell ok."""
    if type(cell) is not dict or set(cell) != _O_ID_CELL_FIELDS:
        return False
    if type(cell["verdict"]) is not bool or type(cell["boundary_found"]) is not bool:
        return False
    if cell["boundary_text_match"] is not None and type(cell["boundary_text_match"]) is not bool:
        return False
    if type(cell["input_tokens"]) is not int or cell["input_tokens"] < 0:
        return False
    if not _o_id_digest_ok(cell["answer_digest"]):
        return False
    if not _o_id_run_digest_ok(cell["first_digit_run_digest"]):
        return False
    if not _o_id_digest_ok(cell["nondigit_projection_digest"]):
        return False
    if type(cell["answer_len"]) is not int or cell["answer_len"] < 0:
        return False
    if type(cell["digit_count"]) is not int or cell["digit_count"] < 0:
        return False
    if cell["digit_count"] > cell["answer_len"]:
        return False
    verdict = (cell["answer_digest"] == expected_digest) and cell["boundary_found"]
    return bool(cell["verdict"] == verdict)


def _o_id_recompute(payload: dict[str, Any]) -> dict[str, Any] | None:
    """O id recompute."""
    import json as _json

    protocol = payload.get("protocol")
    if type(protocol) is not dict:
        return None
    seed = protocol.get("seed")
    backend_fingerprint = protocol.get("backend_fingerprint")
    if type(seed) is not int or isinstance(seed, bool) or type(backend_fingerprint) is not str:
        return None
    if protocol != _o_id_protocol(seed, backend_fingerprint):
        return None
    candidates = payload.get("candidates")
    if type(candidates) is not dict or not candidates:
        return None
    generated = _o_id_generate(seed)
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
        if set(entry) != _O_ID_AGGREGATE_KEYS | {"status", "detail", "records"}:
            return None
        records = entry["records"]
        if type(records) is not list or len(records) != len(generated):
            return None
        total_records += len(records)
        records_bytes += len(
            _json.dumps(records, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        )
        if records_bytes > _O_ID_MAX_BYTES:
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
            expected_digest = _o_id_digest(str(case["value"]))
            if record["expected_digest"] != expected_digest:
                return None
            arm_tokens = record["arm_tokens"]
            id_digests = record["id_digests"]
            cells = record["cells"]
            details = record["cell_details"]
            for mapping in (arm_tokens, id_digests, cells, details):
                if type(mapping) is not dict or set(mapping) != set(_O_ID_ARM_KEYS):
                    return None
            adj_tokens = arm_tokens["adj"]
            if type(adj_tokens) is not int or adj_tokens < 0:
                return None
            if type(arm_tokens["adj_after"]) is not int or arm_tokens["adj_after"] < 0:
                return None
            for arm in _O_ID_ARM_KEYS:
                tokens = arm_tokens[arm]
                id_digest = id_digests[arm]
                if tokens is None:
                    if id_digest is not None:
                        return None
                else:
                    if type(tokens) is not int or tokens < 0:
                        return None
                    if not _o_id_digest_ok(id_digest):
                        return None
                    if arm in _O_ID_SIZED_ARMS:
                        dose = 256 if arm.endswith("256") else 1024
                        if tokens - adj_tokens != dose:
                            return None
                cell = cells[arm]
                detail = details[arm]
                if cell is None:
                    allowed = _O_ID_DETAILS if arm in _O_ID_SIZED_ARMS else _O_ID_UNSIZED_DETAILS
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
                    if not _o_id_cell_ok(cell, expected_digest):
                        return None
                    if cell["input_tokens"] != tokens:
                        return None
        aggregates[name] = _o_id_aggregate(records)
    if total_records > _O_ID_MAX_RECORDS:
        return None
    if not aggregates:
        return None
    return {
        "aggregates": aggregates,
        "n_candidates": len(candidates),
        "n_cases": len(generated),
    }


# ---------------------------------------------------------------------
# History-frame oracle (cycle 219): an independent
# re-implementation of diffreports._valid_history_frame /
# diff_history_frame from docs/HISTORYFRAME_DESIGN.md section
# 6 and the fresh review's amendments A1-A10. Frozen literals
# are transcribed from the design text; aggregates are
# recomputed from the records alone (the per-arm id-digest
# gate of amendment A4 is re-derived here from the recorded
# digests, so a forged payload disagrees with this oracle
# exactly where it disagrees with the implementation).
# ---------------------------------------------------------------------

_O_HF_ARMS = ("own", "foreign_chatml", "foreign_llama3", "verbatim_chatml")
_O_HF_CONTRASTS = ("foreign_chatml", "foreign_llama3")
_O_HF_DONORS = {"foreign_chatml": "chatml", "foreign_llama3": "llama3"}
_O_HF_MIN_N = 10
_O_HF_MIN_D = 6
_O_HF_CELL_FIELDS = {
    "verdict",
    "boundary_found",
    "boundary_text_match",
    "input_tokens",
    "answer_digest",
}
_O_HF_CELL_DETAILS = {"", "arm_context_exceeded", "arm_evaluation_failed"}
_O_HF_TURNS = (
    ("user", "Hello. I have a few short questions for you."),
    ("assistant", "Hello! I am ready. Please ask your questions one at a time."),
    ("user", "Thank you. I will keep each question short."),
    ("assistant", "Understood. I will keep my answers short as well."),
    ("user", "Let us begin whenever you are ready."),
    ("assistant", "I am ready. Please go ahead with your first question."),
)
_O_HF_SEGMENT_RULE = (
    "historyframe-segment-v1: seg(F, m, i) = (F.bos if i > 0 and F.repeat_bos else '')"
    " + (F.user_prefix if m.role is user else F.assistant_prefix) + m.content"
    " + (per-role end of F for m.role, the per-role field when it is not None else"
    " F.turn_end); open(B) = (B.bos if B.repeat_bos else '') + (B.open_suffix if"
    " B.open_suffix is not None else B.assistant_prefix); hybrid(A, B) = B.bos"
    " + seg(A) over the first 6 messages + seg(B) over the rest + open(B);"
    " verbatim(A, B) = A.bos + hybrid(A, B) with only the leading B.bos removed."
    " Screens: S1 per-arm slice-swap identity pre-generation (each hybrid equals"
    " the native B render with only the history-segment slice exchanged; the"
    " verbatim arm exchanges the same slice under its defining leading-bos"
    " substitution; hybrid(B, B) equals the native render byte for byte); S2"
    " special-token-literal audit over the frozen history texts and the donor"
    " and candidate framing literals; S3 budget caps. User fixture contents are"
    " never screened: they are byte-identical across arms, so they cannot"
    " manufacture a frame contrast."
)
_O_HF_OC = {
    "random_labeller_fire_bound": 0.036,
    "competent_fire_bound": 0.031,
    "power_25_point_at_30": 0.52,
    "power_25_point_at_100": 0.988,
    "power_50_point_at_30": 0.983,
    "power_50_point_at_100": 1.0,
}
_O_HF_JITTER = [
    {"epsilon": 0.01, "fire_rate": 0.0004},
    {"epsilon": 0.02, "fire_rate": 0.0048},
    {"epsilon": 0.05, "fire_rate": 0.0213},
]
_O_HF_MAX_BYTES = 12 * 1024 * 1024


def _o_hf_digest_ok(value: Any) -> bool:
    """O hf digest ok."""
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _o_hf_mcnemar_p(a: int, b: int) -> float | None:
    """O hf mcnemar p."""
    d = a + b
    if d == 0:
        return None
    k = min(a, b)
    tail = sum(math.comb(d, i) for i in range(k + 1)) / (2**d)
    return min(1.0, 2.0 * tail)


def _o_hf_classify(n: int, a: int, b: int) -> str:
    """O hf classify."""
    if n < _O_HF_MIN_N:
        return "insufficient"
    d = a + b
    if d == 0:
        return "no_discordance_observed"
    if d < _O_HF_MIN_D:
        return "below_threshold"
    if a == b:
        return "discordant_balanced"
    p = _o_hf_mcnemar_p(a, b)
    if p is not None and p < 0.05:
        return "discordant_asymmetric"
    return "below_threshold"


def _o_hf_fire_rate(n: int) -> float:
    """O hf fire rate."""
    if type(n) is not int or n < _O_HF_MIN_N:
        return 0.0
    total = 0.0
    for d in range(_O_HF_MIN_D, n + 1):
        p_d = math.comb(n, d) * (0.5**n)
        for a in range(d + 1):
            b = d - a
            if a == b:
                continue
            p = _o_hf_mcnemar_p(a, b)
            if p is not None and p < 0.05:
                total += p_d * math.comb(d, a) * (0.5**d)
    return total


def _o_hf_familywise(rates: list[float]) -> dict[str, Any]:
    """O hf familywise."""
    product = 1.0
    for rate in rates:
        product *= 1.0 - rate
    return {
        "decisions": len(rates),
        "expected_false_labels": sum(rates),
        "p_at_least_one": 1.0 - product,
    }


def _o_hf_material_sha() -> str:
    """O hf material sha."""
    raw = json.dumps(
        [[role, content] for role, content in _O_HF_TURNS],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _o_hf_protocol(seed: int, backend_fp: str, fixture_ids: list) -> dict[str, Any]:
    """O hf protocol."""
    protocol = {
        "version": 1,
        "arm_keys": list(_O_HF_ARMS),
        "headline_arms": ["own", "foreign_chatml", "foreign_llama3"],
        "contrasts": list(_O_HF_CONTRASTS),
        "donor_families": dict(_O_HF_DONORS),
        "verbatim_arm": "verbatim_chatml",
        "verbatim_twin": "foreign_chatml",
        "history_condition": "history_6",
        "history_turns": 6,
        "history_material_sha256": _o_hf_material_sha(),
        "segment_rule_sha256": hashlib.sha256(_O_HF_SEGMENT_RULE.encode("utf-8")).hexdigest(),
        "min_n": _O_HF_MIN_N,
        "min_d": _O_HF_MIN_D,
        "generation_budget": 9600,
        "max_ground_records": 2400,
        "max_records_bytes": _O_HF_MAX_BYTES,
        "max_search_fixtures": 100,
        "max_candidates": 24,
        "seed": seed,
        "backend_fingerprint": backend_fp,
        "fixture_ids": list(fixture_ids),
    }
    canonical = json.dumps(protocol, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    protocol["fingerprint"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return protocol


def _o_hf_cell(record: dict, arm: str):
    """O hf cell."""
    cells = record.get("cells")
    if not isinstance(cells, dict):
        return None
    cell = cells.get(arm)
    return cell if isinstance(cell, dict) else None


def _o_hf_decompose(own: dict, foreign: dict) -> str:
    """O hf decompose."""
    if own["answer_digest"] == foreign["answer_digest"]:
        return "boundary_only"
    if own["boundary_found"] and foreign["boundary_found"]:
        return "exact_only"
    return "mixed"


def _o_hf_contrast(records: list, contrast: str) -> dict[str, Any]:
    """O hf contrast."""
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
    decomposition = {"boundary_only": 0, "exact_only": 0, "mixed": 0}
    deltas: list[int] = []
    for record in records:
        segments = record.get("segment_digests")
        segments = segments if isinstance(segments, dict) else {}
        render_divergent = segments.get(contrast) != segments.get("own")
        tokens = record.get("arm_tokens")
        tokens = tokens if isinstance(tokens, dict) else {}
        own_tokens = tokens.get("own")
        foreign_tokens = tokens.get(contrast)
        if type(own_tokens) is int and type(foreign_tokens) is int:
            deltas.append(own_tokens - foreign_tokens)
        own_cell = _o_hf_cell(record, "own")
        foreign_cell = _o_hf_cell(record, contrast)
        if own_cell is not None and foreign_cell is not None:
            ungated_n += 1
            if own_cell["verdict"] and not foreign_cell["verdict"]:
                ungated_own_only += 1
            elif foreign_cell["verdict"] and not own_cell["verdict"]:
                ungated_foreign_only += 1
        if not render_divergent:
            n_render_identical += 1
            continue
        digests = record.get("id_digests")
        digests = digests if isinstance(digests, dict) else {}
        own_digest = digests.get("own")
        foreign_digest = digests.get(contrast)
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
            decomposition[_o_hf_decompose(own_cell, foreign_cell)] += 1
        elif foreign_cell["verdict"] and not own_cell["verdict"]:
            foreign_only += 1
            decomposition[_o_hf_decompose(own_cell, foreign_cell)] += 1
    applicable = n_render_identical < len(records)
    d = own_only + foreign_only
    classification = _o_hf_classify(n_paired, own_only, foreign_only) if applicable else None
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
        "classification": classification,
        "falsification": classification == "discordant_asymmetric",
        "mcnemar_p": _o_hf_mcnemar_p(own_only, foreign_only) if d else None,
        "random_fire_rate": _o_hf_fire_rate(n_paired) if applicable else None,
        "id_length_delta": deltas,
        "max_abs_id_delta": max((abs(x) for x in deltas), default=0),
    }


def _o_hf_verbatim(records: list) -> dict[str, Any]:
    """O hf verbatim."""
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
            twin = _o_hf_cell(record, "foreign_chatml")
            verbatim = _o_hf_cell(record, "verbatim_chatml")
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


def _o_hf_aggregate(records: list) -> dict[str, Any]:
    """O hf aggregate."""
    arms: dict[str, Any] = {}
    for key in _O_HF_ARMS:
        n_completed = 0
        n_pass = 0
        n_context = 0
        n_failed = 0
        for record in records:
            details = record.get("cell_details")
            detail = details.get(key) if isinstance(details, dict) else None
            if detail == "arm_context_exceeded":
                n_context += 1
            elif detail == "arm_evaluation_failed":
                n_failed += 1
            cell = _o_hf_cell(record, key)
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
        "contrasts": {c: _o_hf_contrast(records, c) for c in _O_HF_CONTRASTS},
        "verbatim": _o_hf_verbatim(records),
        "operating_characteristics": dict(_O_HF_OC),
        "jitter_grid": [dict(row) for row in _O_HF_JITTER],
        "n_fixtures": len(records),
    }


def _o_hf_shared(by_candidate: dict) -> frozenset:
    """O hf shared."""
    shared = None
    for records in by_candidate.values():
        ids = frozenset(record["fixture"] for record in records)
        shared = ids if shared is None else shared & ids
    return shared if shared is not None else frozenset()


def _o_hf_selection_over(by_candidate: dict, arm: str, fixtures: frozenset):
    """O hf selection over."""
    completed_sets = []
    all_pass: dict[str, Any] = {}
    for name, records in by_candidate.items():
        completed = frozenset(
            str(record["fixture"])
            for record in records
            if record["fixture"] in fixtures and _o_hf_cell(record, arm) is not None
        )
        completed_sets.append(completed)
        if not completed:
            return None
        all_pass[name] = all(
            bool(_o_hf_cell(record, arm)["verdict"])
            for record in records
            if record["fixture"] in completed and _o_hf_cell(record, arm) is not None
        )
    if len(set(completed_sets)) > 1:
        return None
    passers = [name for name, ok in all_pass.items() if ok]
    if len(passers) == 1:
        return {"decision": "selected", "winner": passers[0]}
    if len(passers) > 1:
        return {"decision": "ambiguous", "winner": None}
    return {"decision": "no_candidate", "winner": None}


def _o_hf_arm_selection(by_candidate: dict, arm: str) -> dict[str, Any]:
    """O hf arm selection."""
    shared = _o_hf_shared(by_candidate)
    selection = _o_hf_selection_over(by_candidate, arm, shared) if shared else None
    if selection is None:
        return {
            "decision": "unavailable",
            "detail": "selection_withheld_mismatched_sets",
            "winner": None,
            "withheld": True,
        }
    return {
        "decision": selection["decision"],
        "detail": "",
        "winner": selection["winner"],
        "withheld": False,
    }


def _o_hf_winner_headline(baseline: dict, arm: dict) -> dict[str, Any]:
    """O hf winner headline."""
    if not isinstance(baseline, dict) or not isinstance(arm, dict):
        return {
            "headline": "winner_undefined",
            "detail": "selection_missing",
            "winner_changed": None,
        }
    if baseline.get("withheld") is True or arm.get("withheld") is True:
        return {
            "headline": "winner_undefined",
            "detail": "selection_withheld_mismatched_sets",
            "winner_changed": None,
        }
    baseline_selected = baseline.get("decision") == "selected"
    arm_selected = arm.get("decision") == "selected"
    if baseline_selected and arm_selected:
        if baseline.get("winner") != arm.get("winner"):
            return {"headline": "winner_changed", "detail": "", "winner_changed": True}
        return {"headline": "winner_stable", "detail": "", "winner_changed": False}
    if baseline_selected != arm_selected:
        return {"headline": "winner_abstention_asymmetric", "detail": "", "winner_changed": False}
    return {"headline": "winner_undefined", "detail": "", "winner_changed": False}


def _o_hf_primary_flips(conditions: dict, primary: Any) -> list:
    """O hf primary flips."""
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
    flips = []
    for arm in ("own", "foreign_chatml", "foreign_llama3"):
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


def _o_hf_stratum_ok(stratum: Any) -> bool:
    """O hf stratum ok."""
    if type(stratum) is not dict or set(stratum) != set(_O_HF_CONTRASTS):
        return False
    for arm in _O_HF_CONTRASTS:
        block = stratum[arm]
        if type(block) is not dict or set(block) != {"label", "literals"}:
            return False
        if block["label"] not in ("all_single", "all_fragmented", "mixed"):
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


def _o_hf_recompute(payload: Any):
    """O hf recompute."""
    if type(payload) is not dict:
        return None
    protocol = payload.get("protocol")
    if type(protocol) is not dict:
        return None
    seed = protocol.get("seed")
    backend_fp = protocol.get("backend_fingerprint")
    fixture_ids = protocol.get("fixture_ids")
    if type(seed) is not int or isinstance(seed, bool):
        return None
    if type(backend_fp) is not str:
        return None
    if (
        type(fixture_ids) is not list
        or not fixture_ids
        or len(fixture_ids) > 100
        or any(type(item) is not str for item in fixture_ids)
        or len(set(fixture_ids)) != len(fixture_ids)
    ):
        return None
    n_fixtures = len(fixture_ids)
    if protocol != _o_hf_protocol(seed, backend_fp, fixture_ids):
        return None
    if not _o_hf_stratum_ok(payload.get("stratum")):
        return None
    candidates = payload.get("candidates")
    if type(candidates) is not dict or not candidates:
        return None
    if len(candidates) > 24:
        return None
    total_records = 0
    records_bytes = 0
    aggregates: dict[str, Any] = {}
    by_candidate: dict[str, list] = {}
    rates: list[float] = []
    for name, entry in candidates.items():
        if type(entry) is not dict:
            return None
        if entry.get("status") == "unavailable":
            if set(entry) != {"status", "detail", "records"} or entry.get("records") != []:
                return None
            aggregates[name] = {}
            continue
        if entry.get("status") != "available" or entry.get("detail") != "":
            return None
        if set(entry) != {
            "arms",
            "contrasts",
            "verbatim",
            "operating_characteristics",
            "jitter_grid",
            "n_fixtures",
            "status",
            "detail",
            "records",
        }:
            return None
        records = entry.get("records")
        if type(records) is not list or len(records) != n_fixtures:
            return None
        total_records += len(records)
        records_bytes += len(
            json.dumps(records, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        )
        if records_bytes > _O_HF_MAX_BYTES:
            return None
        own_bos = donor_bos = None
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
            if not _o_hf_digest_ok(record["expected_digest"]):
                return None
            if own_bos is None:
                own_bos = record["own_bos_digest"]
                donor_bos = record["donor_bos_digest"]
            elif record["own_bos_digest"] != own_bos or record["donor_bos_digest"] != donor_bos:
                return None
            if not _o_hf_digest_ok(own_bos) or not _o_hf_digest_ok(donor_bos):
                return None
            for key in ("segment_digests", "hybrid_digests", "id_digests"):
                mapping = record[key]
                if type(mapping) is not dict or set(mapping) != set(_O_HF_ARMS):
                    return None
                if any(not _o_hf_digest_ok(digest) for digest in mapping.values()):
                    return None
            for key in ("arm_tokens", "cells", "cell_details"):
                mapping = record[key]
                if type(mapping) is not dict or set(mapping) != set(_O_HF_ARMS):
                    return None
            for arm in _O_HF_ARMS:
                tokens = record["arm_tokens"][arm]
                if type(tokens) is not int or isinstance(tokens, bool) or tokens < 0:
                    return None
                detail = record["cell_details"][arm]
                if detail not in _O_HF_CELL_DETAILS:
                    return None
                cell = record["cells"][arm]
                if cell is None:
                    if detail == "":
                        return None
                    continue
                if detail != "":
                    return None
                if type(cell) is not dict or set(cell) != _O_HF_CELL_FIELDS:
                    return None
                if type(cell["verdict"]) is not bool or type(cell["boundary_found"]) is not bool:
                    return None
                if (
                    cell["boundary_text_match"] is not None
                    and type(cell["boundary_text_match"]) is not bool
                ):
                    return None
                if cell["input_tokens"] != tokens:
                    return None
                digest = cell["answer_digest"]
                if digest is not None and not _o_hf_digest_ok(digest):
                    return None
                if cell["verdict"] != (
                    digest == record["expected_digest"] and cell["boundary_found"]
                ):
                    return None
        aggregates[name] = _o_hf_aggregate(records)
        by_candidate[name] = records
        for contrast in _O_HF_CONTRASTS:
            rate = aggregates[name]["contrasts"][contrast]["random_fire_rate"]
            if rate is not None:
                rates.append(rate)
    if total_records > 2400:
        return None
    if not aggregates:
        return None
    conditions = {
        arm: _o_hf_arm_selection(by_candidate, arm)
        for arm in ("own", "foreign_chatml", "foreign_llama3")
    }
    selection = {
        contrast: _o_hf_winner_headline(conditions["own"], conditions[contrast])
        for contrast in _O_HF_CONTRASTS
    }
    return {
        "aggregates": aggregates,
        "conditions": conditions,
        "selection": selection,
        "flips": _o_hf_primary_flips(conditions, payload.get("primary")),
        "familywise": _o_hf_familywise(rates),
        "n_candidates": len(candidates),
        "n_fixtures": n_fixtures,
    }


def _o_historyframe_valid(value: Any) -> bool:
    """Oracle history-frame validator: mirrors
    diffreports._valid_history_frame.
    """
    if type(value) is not dict or value.get("available") is not True:
        return False
    if set(value) != {
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
    if value.get("detail") != "":
        return False
    audit = value.get("audit")
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
    primary = value.get("primary")
    if primary is not None:
        if type(primary) is not dict or set(primary) != {"decision", "winner"}:
            return False
        if primary["decision"] is not None and type(primary["decision"]) is not str:
            return False
        if primary["winner"] is not None and type(primary["winner"]) is not str:
            return False
    recomputed = _o_hf_recompute(value)
    if recomputed is None:
        return False
    candidates = value["candidates"]
    for name, expected in recomputed["aggregates"].items():
        entry = candidates[name]
        for key, val in expected.items():
            if entry.get(key) != val:
                return False
    for key in ("conditions", "selection", "flips", "familywise", "n_candidates", "n_fixtures"):
        if value.get(key) != recomputed[key]:
            return False
    return True


def _o_historyframe(old: Any, new: Any) -> tuple[dict[str, Any], bool, bool]:
    """Oracle history-frame section: mirrors
    diffreports.diff_history_frame.
    """
    old_ok = _o_historyframe_valid(old)
    new_ok = _o_historyframe_valid(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    diff: dict[str, Any] = {"available": available, "changed": False}
    asymmetric = available not in ("both", "neither")
    if available != "both":
        return diff, False, asymmetric
    protocol_drift = old["protocol"].get("fingerprint") != new["protocol"].get("fingerprint")
    generator_drift = old["protocol"].get("history_material_sha256") != new["protocol"].get(
        "history_material_sha256"
    ) or old["protocol"].get("fixture_ids") != new["protocol"].get("fixture_ids")
    backend_drift = old["protocol"].get("backend_fingerprint") != new["protocol"].get(
        "backend_fingerprint"
    )
    comparable = not protocol_drift and not generator_drift and not backend_drift
    stratum_drift: list[dict[str, Any]] = []
    for arm in _O_HF_CONTRASTS:
        if old["stratum"][arm]["label"] != new["stratum"][arm]["label"]:
            stratum_drift.append(
                {
                    "family": arm,
                    "old": old["stratum"][arm]["label"],
                    "new": new["stratum"][arm]["label"],
                }
            )
    old_candidates = old["candidates"]
    new_candidates = new["candidates"]
    added = sorted(set(new_candidates) - set(old_candidates))
    removed = sorted(set(old_candidates) - set(new_candidates))
    candidate_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_candidates) & set(new_candidates)):
        old_entry = old_candidates[name]
        new_entry = new_candidates[name]
        if old_entry.get("status") != "available" or new_entry.get("status") != "available":
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
        metrics: list[tuple[str, Any, Any]] = []
        for contrast in _O_HF_CONTRASTS:
            old_block = old_entry["contrasts"][contrast]
            new_block = new_entry["contrasts"][contrast]
            metrics.extend(
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
        metrics.append(
            (
                "verbatim_applicable",
                old_entry["verbatim"]["applicable"],
                new_entry["verbatim"]["applicable"],
            )
        )
        for metric, old_value, new_value in metrics:
            if old_value != new_value:
                candidate_drift.append(
                    {"candidate": name, "metric": metric, "old": old_value, "new": new_value}
                )
    selection_drift: list[dict[str, Any]] = []
    for contrast in _O_HF_CONTRASTS:
        if old["selection"][contrast]["headline"] != new["selection"][contrast]["headline"]:
            selection_drift.append(
                {
                    "contrast": contrast,
                    "old": old["selection"][contrast]["headline"],
                    "new": new["selection"][contrast]["headline"],
                }
            )
    diff["candidate_drift"] = candidate_drift
    diff["candidates_added"] = added
    diff["candidates_removed"] = removed
    diff["selection_drift"] = selection_drift
    diff["stratum_drift"] = stratum_drift
    diff["protocol_drift"] = protocol_drift
    diff["generator_drift"] = generator_drift
    diff["backend_drift"] = backend_drift
    diff["comparable"] = comparable
    changed = bool(
        candidate_drift
        or added
        or removed
        or selection_drift
        or stratum_drift
        or protocol_drift
        or generator_drift
        or backend_drift
    )
    diff["changed"] = changed
    return diff, changed, asymmetric


# ---------------------------------------------------------------------
# Prune-sweep oracle (cycle 220): an independent
# re-implementation of diffreports._valid_prune_sweep /
# diff_prune_sweep from docs/PRUNESWEEP_DESIGN.md and the
# fresh review's amendments A-I. Frozen literals are
# transcribed from the design text; the case generator,
# drop order, protocol, and aggregates are recomputed here
# from the frozen rules alone, so a forged payload
# disagrees with this oracle exactly where it disagrees
# with the implementation.
# ---------------------------------------------------------------------

_O_PS_ARMS = ("dose0", "dose18", "dose36", "dose72", "tail72", "closed_book")
_O_PS_HEADLINE = ("dose0", "dose18", "dose36", "dose72")
_O_PS_SHARE_ARMS = ("dose18", "dose36", "dose72", "tail72")
_O_PS_DOSES = {"dose0": 0, "dose18": 18, "dose36": 36, "dose72": 72}
_O_PS_MIN_C = 12
_O_PS_MIN_A = 5
_O_PS_CELL_FIELDS = {
    "verdict",
    "boundary_found",
    "boundary_text_match",
    "input_tokens",
    "answer_digest",
}
_O_PS_CELL_DETAILS = {"", "arm_context_exceeded", "arm_evaluation_failed"}
_O_PS_DECOMPOSITION = ("exact_pass", "boundary_only", "wrong_value", "non_value")
_O_PS_INSTRUCTION = "Read the passage and answer the question."
_O_PS_TOPICS = (
    "harbor light",
    "summit trail",
    "quarry gate",
    "tundra post",
    "granite arch",
    "willow ford",
    "copper mill",
    "saddle pass",
)
_O_PS_VALUES = (
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
_O_PS_SUBJECTS = (
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
_O_PS_OBJECTS = (
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
_O_PS_PLACES = (
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
_O_PS_TIMES = (
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
_O_PS_TEMPLATES = (
    "The {subject} holds {object} inside the {place} each {time}.",
    "A {subject} keeps {object} near the {place} during {time}.",
    "Workers quietly move {object} past the {place} during {time}.",
    "Each {time} the {subject} stores {object} under the {place}.",
)
_O_PS_NEGATION = frozenset(
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
_O_PS_OC = {
    "null_random_stated": 0.0121,
    "null_competent_stated": 0.0143,
    "null_jitter_stated": 0.000005,
    "null_competent_exchangeable_v09": 0.003,
    "null_competent_exchangeable_v05": 0.0004,
    "null_competent_exchangeable_v02": 0.000004,
    "familywise_expected_stated": 0.34,
    "familywise_any_stated": 0.29,
    "familywise_expected_exchangeable": 0.07,
    "familywise_any_exchangeable": 0.07,
    "power_q25": 0.3969,
    "power_q50": 0.9601,
    "power_q75": 0.9996,
    "shape_returns_cap_graded": 0.094,
    "shape_intermediate_support_threshold": 0.0,
}
_O_PS_MAX_BYTES = 12 * 1024 * 1024
_O_PS_VALUE_DIGESTS = frozenset(
    hashlib.sha256(value.encode("utf-8")).hexdigest() for value in _O_PS_VALUES
)


def _o_ps_digest_ok(value: Any) -> bool:
    """O ps digest ok."""
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _o_ps_sha(text: str) -> str:
    """O ps sha."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _o_ps_draw_sentence(seed: int, case_id: str, slot: int, seen: set) -> str:
    """O ps draw sentence."""
    template = _O_PS_TEMPLATES[slot % 4]
    counter = 0
    while True:
        digest = _o_ps_sha(f"{seed}:{case_id}:slot:{slot}:{counter}")
        subject = _O_PS_SUBJECTS[int(digest[0:8], 16) % 12]
        obj = _O_PS_OBJECTS[int(digest[8:16], 16) % 12]
        place = _O_PS_PLACES[int(digest[16:24], 16) % 12]
        when = _O_PS_TIMES[int(digest[24:32], 16) % 12]
        sentence = template.format(subject=subject, object=obj, place=place, time=when)
        if sentence not in seen:
            return sentence
        counter += 1
        if counter > 10000:
            raise ValueError("prune_sentence_draw_exhausted")


def _o_ps_generate_case(seed: int, index: int) -> dict[str, Any]:
    """O ps generate case."""
    case_id = f"prune-{index:02d}"
    topic = _O_PS_TOPICS[index % len(_O_PS_TOPICS)]
    value = _O_PS_VALUES[index]
    seen: set = set()
    sentences = []
    for slot in range(16):
        sentence = _o_ps_draw_sentence(seed, case_id, slot, seen)
        seen.add(sentence)
        sentences.append(sentence)
    fact = f"The {topic} code is {value}."
    return {
        "case_id": case_id,
        "topic": topic,
        "value": value,
        "context_sentences": tuple(sentences),
        "fact_sentence": fact,
        "passage_sentences": tuple(sentences[:8]) + (fact,) + tuple(sentences[8:]),
    }


def _o_ps_generate_cases(seed: int) -> tuple:
    """O ps generate cases."""
    return tuple(_o_ps_generate_case(seed, index) for index in range(24))


def _o_ps_drop_order(seed: int, case_id: str) -> tuple:
    """O ps drop order."""
    return tuple(sorted(range(144), key=lambda position: _o_ps_sha(f"{seed}:{case_id}:{position}")))


def _o_ps_dropped(case_id: str, arm: str, seed: int) -> frozenset:
    """O ps dropped."""
    if arm == "tail72":
        return frozenset(range(72, 144))
    return frozenset(_o_ps_drop_order(seed, case_id)[: _O_PS_DOSES[arm]])


def _o_ps_arm_content(case: dict, arm: str, seed: int) -> str:
    """O ps arm content."""
    question = f"What is the {case['topic']} code? Reply with only the code word."
    if arm == "closed_book":
        return "\n".join((_O_PS_INSTRUCTION, question))
    dropped = _o_ps_dropped(case["case_id"], arm, seed)
    kept = []
    position = 0
    for index, sentence in enumerate(case["passage_sentences"]):
        words = sentence.split(" ")
        if index == 8:
            kept.extend(words)
            continue
        for offset, word in enumerate(words):
            if position + offset not in dropped:
                kept.append(word)
        position += len(words)
    return "\n".join((_O_PS_INSTRUCTION, " ".join(kept), question))


def _o_ps_word_classes(word: str) -> tuple:
    """O ps word classes."""
    bare = word.strip(".,;:!?\"'").lower()
    return (any(char.isdigit() for char in word), bare in _O_PS_NEGATION)


def _o_ps_shares(case: dict, seed: int) -> dict[str, Any]:
    """O ps shares."""
    words = []
    for sentence in case["context_sentences"]:
        words.extend(sentence.split(" "))
    shares: dict[str, Any] = {}
    for arm in _O_PS_SHARE_ARMS:
        dropped_words = [
            words[position] for position in sorted(_o_ps_dropped(case["case_id"], arm, seed))
        ]
        if not dropped_words:
            shares[arm] = {"numeral": 0.0, "negation": 0.0}
            continue
        numeral = sum(1 for word in dropped_words if _o_ps_word_classes(word)[0])
        negation = sum(1 for word in dropped_words if _o_ps_word_classes(word)[1])
        shares[arm] = {
            "numeral": numeral / len(dropped_words),
            "negation": negation / len(dropped_words),
        }
    return shares


def _o_ps_case_material() -> str:
    """O ps case material."""
    material = {
        "instruction": _O_PS_INSTRUCTION,
        "fact_template": "The {topic} code is {value}.",
        "question_template": "What is the {topic} code? Reply with only the code word.",
        "topics": list(_O_PS_TOPICS),
        "values": list(_O_PS_VALUES),
        "subjects": list(_O_PS_SUBJECTS),
        "objects": list(_O_PS_OBJECTS),
        "places": list(_O_PS_PLACES),
        "times": list(_O_PS_TIMES),
        "templates": list(_O_PS_TEMPLATES),
        "negation_words": sorted(_O_PS_NEGATION),
        "doses": [0, 18, 36, 72],
    }
    raw = json.dumps(material, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return _o_ps_sha(raw)


def _o_ps_ground(seed: int) -> str:
    """O ps ground."""
    rows = [
        [
            case["case_id"],
            case["topic"],
            case["value"],
            list(case["context_sentences"]),
            case["fact_sentence"],
        ]
        for case in _o_ps_generate_cases(seed)
    ]
    raw = json.dumps(rows, ensure_ascii=True, separators=(",", ":"))
    return _o_ps_sha(raw)


def _o_ps_protocol(seed: int, backend_fp: str) -> dict[str, Any]:
    """O ps protocol."""
    protocol = {
        "version": 1,
        "arm_keys": list(_O_PS_ARMS),
        "headline_arms": list(_O_PS_HEADLINE),
        "descriptive_arm": "tail72",
        "screen_arm": "closed_book",
        "doses": [0, 18, 36, 72],
        "case_count": 24,
        "eligible_words": 144,
        "case_material_sha256": _o_ps_case_material(),
        "ground_sha256": _o_ps_ground(seed),
        "min_c": _O_PS_MIN_C,
        "min_a": _O_PS_MIN_A,
        "generation_budget": 3456,
        "headline_generations": 2304,
        "max_ground_records": 576,
        "max_records_bytes": _O_PS_MAX_BYTES,
        "max_candidates": 24,
        "seed": seed,
        "backend_fingerprint": backend_fp,
    }
    canonical = json.dumps(protocol, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    protocol["fingerprint"] = _o_ps_sha(canonical)
    return protocol


def _o_ps_mcnemar_p(a: int, b: int):
    """O ps mcnemar p."""
    d = a + b
    if d == 0:
        return None
    k = min(a, b)
    tail = sum(math.comb(d, i) for i in range(k + 1)) / (2**d)
    return min(1.0, 2.0 * tail)


def _o_ps_classify(direct: int, drop: int, reverse: int) -> str:
    """O ps classify."""
    if direct < _O_PS_MIN_C:
        return "insufficient"
    p = _o_ps_mcnemar_p(drop, reverse)
    if drop >= _O_PS_MIN_A and p is not None and p < 0.05:
        return "prune_drop_observed"
    return "below_threshold"


def _o_ps_cell(record: dict, arm: str):
    """O ps cell."""
    cells = record.get("cells")
    if not isinstance(cells, dict):
        return None
    cell = cells.get(arm)
    return cell if isinstance(cell, dict) else None


def _o_ps_detail(record: dict, arm: str) -> str:
    """O ps detail."""
    details = record.get("cell_details")
    if not isinstance(details, dict):
        return ""
    detail = details.get(arm, "")
    return detail if type(detail) is str else ""


def _o_ps_verdict(record: dict, arm: str) -> bool:
    """O ps verdict."""
    cell = _o_ps_cell(record, arm)
    return bool(cell is not None and cell["verdict"])


def _o_ps_decompose(cell: dict, expected_digest: str) -> str:
    """O ps decompose."""
    if cell.get("verdict") is True:
        return "exact_pass"
    digest = cell.get("answer_digest")
    if digest is not None and digest == expected_digest:
        return "boundary_only"
    if digest is not None and digest in _O_PS_VALUE_DIGESTS:
        return "wrong_value"
    return "non_value"


def _o_ps_first_flip(record: dict) -> str:
    """O ps first flip."""
    for arm in _O_PS_HEADLINE[1:]:
        if not _o_ps_verdict(record, arm):
            return arm
    return "never"


def _o_ps_aggregate(records: list) -> dict[str, Any]:
    """O ps aggregate."""
    arms: dict[str, Any] = {}
    for key in _O_PS_ARMS:
        n_completed = 0
        n_pass = 0
        n_context = 0
        n_failed = 0
        decomposition = dict.fromkeys(_O_PS_DECOMPOSITION, 0)
        for record in records:
            detail = _o_ps_detail(record, key)
            if detail == "arm_context_exceeded":
                n_context += 1
            elif detail == "arm_evaluation_failed":
                n_failed += 1
            cell = _o_ps_cell(record, key)
            if cell is None:
                continue
            n_completed += 1
            if cell["verdict"]:
                n_pass += 1
            decomposition[_o_ps_decompose(cell, str(record.get("expected_digest", "")))] += 1
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
        if all(_o_ps_cell(record, arm) is not None for arm in _O_PS_HEADLINE)
    ]

    def _screened(record: dict) -> bool:
        """Screened."""
        cell = _o_ps_cell(record, "closed_book")
        return bool(cell is not None and cell["verdict"])

    screened = [record for record in complete if _screened(record)]
    unscreened = [record for record in complete if not _screened(record)]
    ids_excluded = []
    population = []
    for record in unscreened:
        digests = record.get("id_digests")
        identical = (
            isinstance(digests, dict)
            and digests.get("dose0") is not None
            and digests.get("dose0") == digests.get("dose72")
        )
        if identical:
            ids_excluded.append(record)
        else:
            population.append(record)
    direct = [record for record in population if _o_ps_verdict(record, "dose0")]
    drop_cases = 0
    boundary_attrition = 0
    a_by_first_flip = dict.fromkeys(_O_PS_HEADLINE[1:], 0)
    for record in direct:
        cell72 = _o_ps_cell(record, "dose72")
        if cell72 is None or cell72["verdict"]:
            continue
        klass = _o_ps_decompose(cell72, str(record.get("expected_digest", "")))
        if klass == "boundary_only":
            boundary_attrition += 1
        elif klass in ("wrong_value", "non_value"):
            drop_cases += 1
            a_by_first_flip[_o_ps_first_flip(record)] += 1
    reverse = sum(
        1
        for record in population
        if not _o_ps_verdict(record, "dose0") and _o_ps_verdict(record, "dose72")
    )
    headline = {
        "n_population": len(population),
        "n_headline_incomplete": len(records) - len(complete),
        "n_screened": len(screened),
        "n_closed_book_incomplete": sum(
            1 for record in records if _o_ps_cell(record, "closed_book") is None
        ),
        "n_ids_identical": len(ids_excluded),
        "direct_passes": len(direct),
        "drop_cases": drop_cases,
        "boundary_attrition": boundary_attrition,
        "reverse": reverse,
        "decision": _o_ps_classify(len(direct), drop_cases, reverse),
        "a_by_first_flip": a_by_first_flip,
    }
    histogram = dict.fromkeys(_O_PS_HEADLINE[1:], 0)
    histogram["never"] = 0
    returns = 0
    for record in direct:
        histogram[_o_ps_first_flip(record)] += 1
        if (
            not _o_ps_verdict(record, "dose18") or not _o_ps_verdict(record, "dose36")
        ) and _o_ps_verdict(record, "dose72"):
            returns += 1
    shape = {
        "first_flip_histogram": histogram,
        "returns": returns,
        "dose_pass_rates": {
            arm: (
                sum(1 for record in population if _o_ps_verdict(record, arm)) / len(population)
                if population
                else None
            )
            for arm in _O_PS_HEADLINE
        },
    }
    n_compared = 0
    dose72_fails = 0
    tail72_fails = 0
    dose72_only = 0
    tail72_only = 0
    for record in population:
        if _o_ps_cell(record, "tail72") is None:
            continue
        n_compared += 1
        fail72 = not _o_ps_verdict(record, "dose72")
        fail_tail = not _o_ps_verdict(record, "tail72")
        dose72_fails += 1 if fail72 else 0
        tail72_fails += 1 if fail_tail else 0
        if fail72 and not fail_tail:
            dose72_only += 1
        elif fail_tail and not fail72:
            tail72_only += 1
    p_tail = _o_ps_mcnemar_p(dose72_only, tail72_only)
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
    for arm in (*_O_PS_HEADLINE[1:], "tail72"):
        deltas: list = []
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
                base_cell = _o_ps_cell(record, "dose0")
                arm_cell = _o_ps_cell(record, arm)
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
    dropped_shares: dict[str, Any] = {}
    for arm in _O_PS_SHARE_ARMS:
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
        "operating_characteristics": dict(_O_PS_OC),
        "n_cases": len(records),
        "n_incomplete": sum(
            1 for record in records if any(_o_ps_cell(record, arm) is None for arm in _O_PS_ARMS)
        ),
    }


def _o_ps_recompute(payload: Any):
    """O ps recompute."""
    if type(payload) is not dict:
        return None
    protocol = payload.get("protocol")
    if type(protocol) is not dict:
        return None
    seed = protocol.get("seed")
    backend_fp = protocol.get("backend_fingerprint")
    if type(seed) is not int or isinstance(seed, bool):
        return None
    if type(backend_fp) is not str:
        return None
    if protocol != _o_ps_protocol(seed, backend_fp):
        return None
    cases = _o_ps_generate_cases(seed)
    if len(cases) != protocol["case_count"]:
        return None
    candidates = payload.get("candidates")
    if type(candidates) is not dict or not candidates:
        return None
    if len(candidates) > 24:
        return None
    total_records = 0
    records_bytes = 0
    aggregates: dict[str, Any] = {}
    for name, entry in candidates.items():
        if type(entry) is not dict:
            return None
        if entry.get("status") == "unavailable":
            if set(entry) != {"status", "detail", "records"} or entry.get("records") != []:
                return None
            aggregates[name] = {}
            continue
        if entry.get("status") != "available" or entry.get("detail") != "":
            return None
        if set(entry) != {
            "arms",
            "headline",
            "shape",
            "tail_contrast",
            "ids",
            "dropped_shares",
            "operating_characteristics",
            "n_cases",
            "n_incomplete",
            "status",
            "detail",
            "records",
        }:
            return None
        records = entry.get("records")
        if type(records) is not list or len(records) != 24:
            return None
        total_records += len(records)
        records_bytes += len(
            json.dumps(records, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        )
        if records_bytes > _O_PS_MAX_BYTES:
            return None
        for record, case in zip(records, cases, strict=True):
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
            if record["fixture"] != case["case_id"] or record["topic"] != case["topic"]:
                return None
            if record["expected_digest"] != _o_ps_sha(case["value"]):
                return None
            for key in ("arm_tokens", "id_digests", "content_digests", "cells", "cell_details"):
                mapping = record[key]
                if type(mapping) is not dict or set(mapping) != set(_O_PS_ARMS):
                    return None
            if record["content_digests"] != {
                arm: _o_ps_sha(_o_ps_arm_content(case, arm, seed)) for arm in _O_PS_ARMS
            }:
                return None
            derived_identical = record["id_digests"].get("dose0") is not None and record[
                "id_digests"
            ].get("dose0") == record["id_digests"].get("dose72")
            if record["ids_identical"] is not derived_identical:
                return None
            if record["dropped_shares"] != _o_ps_shares(case, seed):
                return None
            for arm in _O_PS_ARMS:
                tokens = record["arm_tokens"][arm]
                if type(tokens) is not int or isinstance(tokens, bool) or tokens < 0:
                    return None
                if not _o_ps_digest_ok(record["id_digests"][arm]):
                    return None
                if not _o_ps_digest_ok(record["content_digests"][arm]):
                    return None
                detail = record["cell_details"][arm]
                if detail not in _O_PS_CELL_DETAILS:
                    return None
                cell = record["cells"][arm]
                if cell is None:
                    if detail == "":
                        return None
                    continue
                if detail != "":
                    return None
                if type(cell) is not dict or set(cell) != _O_PS_CELL_FIELDS:
                    return None
                if type(cell["verdict"]) is not bool or type(cell["boundary_found"]) is not bool:
                    return None
                if (
                    cell["boundary_text_match"] is not None
                    and type(cell["boundary_text_match"]) is not bool
                ):
                    return None
                if cell["input_tokens"] != tokens:
                    return None
                digest = cell["answer_digest"]
                if digest is not None and not _o_ps_digest_ok(digest):
                    return None
                if cell["verdict"] != (
                    digest == record["expected_digest"] and cell["boundary_found"]
                ):
                    return None
        aggregates[name] = _o_ps_aggregate(records)
    if total_records > 576:
        return None
    if not aggregates:
        return None
    return {"aggregates": aggregates, "n_candidates": len(candidates), "n_cases": 24}


def _o_prunesweep_valid(value: Any) -> bool:
    """Oracle prune-sweep validator: mirrors
    diffreports._valid_prune_sweep.
    """
    if type(value) is not dict or value.get("available") is not True:
        return False
    if set(value) != {
        "available",
        "detail",
        "protocol",
        "audit",
        "candidates",
        "n_candidates",
        "n_cases",
    }:
        return False
    if value.get("detail") != "":
        return False
    audit = value.get("audit")
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
    recomputed = _o_ps_recompute(value)
    if recomputed is None:
        return False
    candidates = value["candidates"]
    for name, expected in recomputed["aggregates"].items():
        entry = candidates[name]
        for key, val in expected.items():
            if entry.get(key) != val:
                return False
    return all(value.get(key) == recomputed[key] for key in ("n_candidates", "n_cases"))


def _o_prunesweep(old: Any, new: Any) -> tuple[dict[str, Any], bool, bool]:
    """Oracle prune-sweep section: mirrors
    diffreports.diff_prune_sweep.
    """
    old_ok = _o_prunesweep_valid(old)
    new_ok = _o_prunesweep_valid(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    diff: dict[str, Any] = {"available": available, "changed": False}
    asymmetric = available not in ("both", "neither")
    if available != "both":
        return diff, False, asymmetric
    protocol_drift = old["protocol"].get("fingerprint") != new["protocol"].get("fingerprint")
    generator_drift = old["protocol"].get("ground_sha256") != new["protocol"].get(
        "ground_sha256"
    ) or old["protocol"].get("case_material_sha256") != new["protocol"].get("case_material_sha256")
    backend_drift = old["protocol"].get("backend_fingerprint") != new["protocol"].get(
        "backend_fingerprint"
    )
    comparable = not protocol_drift and not generator_drift and not backend_drift
    old_candidates = old["candidates"]
    new_candidates = new["candidates"]
    added = sorted(set(new_candidates) - set(old_candidates))
    removed = sorted(set(old_candidates) - set(new_candidates))
    candidate_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_candidates) & set(new_candidates)):
        old_entry = old_candidates[name]
        new_entry = new_candidates[name]
        if old_entry.get("status") != "available" or new_entry.get("status") != "available":
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
        metrics: list[tuple[str, Any, Any]] = [
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
        for metric, old_value, new_value in metrics:
            if old_value != new_value:
                candidate_drift.append(
                    {"candidate": name, "metric": metric, "old": old_value, "new": new_value}
                )
    diff["candidate_drift"] = candidate_drift
    diff["candidates_added"] = added
    diff["candidates_removed"] = removed
    diff["protocol_drift"] = protocol_drift
    diff["generator_drift"] = generator_drift
    diff["backend_drift"] = backend_drift
    diff["comparable"] = comparable
    changed = bool(
        candidate_drift or added or removed or protocol_drift or generator_drift or backend_drift
    )
    diff["changed"] = changed
    return diff, changed, asymmetric


# ---------------------------------------------------------------------------
# Record notation (cycle 221): independent oracle for
# diffreports._valid_record_notation / diff_record_notation,
# re-implemented from docs/PAYLOADSERIALIZATION_DESIGN.md
# Part II as amended (Part III). The generator's RNG call
# sequence is frozen (amendment A); the mirror binds every
# completed cell's input_tokens to the record's arm token
# count and every headline cell's answer_text to its
# answer_digest (amendment D).


_O_RN_WORDS = (
    "amber",
    "birch",
    "cedar",
    "copper",
    "coral",
    "falcon",
    "garnet",
    "harbor",
    "indigo",
    "juniper",
    "lagoon",
    "magnet",
    "marble",
    "nickel",
    "onyx",
    "pepper",
    "quartz",
    "saddle",
    "saffron",
    "summit",
    "tundra",
    "velvet",
    "walnut",
    "willow",
)
_O_RN_FIELDS_3 = ("code", "label", "zone")
_O_RN_FIELDS_4 = ("code", "label", "mark", "zone")
_O_RN_QUESTION_TEMPLATE = "What is the code of the record whose zone is {key}?"
_O_RN_CONTRACT = "Answer with ONLY the value."
_O_RN_GENERATOR_SPEC = (
    "rng = random.Random(seed * 1000 + i); per value draw, in order: "
    "word = WORDS[rng.randrange(24)], then number = rng.randrange(1000, 10000); "
    "a duplicate value is consumed and discarded; the table fills row-major, "
    "fields in alphabetical order; the queried record index is the next draw, "
    "rng.randrange(n_records), after the table is complete"
)
_O_RN_HEADLINE = ("prose", "canonical_json")
_O_RN_ARMS = ("prose", "canonical_json", "closed_book")
_O_RN_CELL_DETAILS = ("", "arm_context_exceeded", "arm_evaluation_failed")
_O_RN_HEADLINE_CELL_FIELDS = {
    "verdict",
    "boundary_found",
    "boundary_text_match",
    "input_tokens",
    "answer_digest",
    "answer_text",
}
_O_RN_SCREEN_CELL_FIELDS = {
    "verdict",
    "boundary_found",
    "boundary_text_match",
    "input_tokens",
    "answer_digest",
}
_O_RN_MAX_BYTES = 12 * 1024 * 1024
_O_RN_OC = {
    "transplant_fire_competent_independent": 0.8897,
    "transplant_fire_competent_paired_phi05": 0.5088,
    "transplant_fire_delta_005": 0.2276,
    "transplant_familywise_expected": 21.35,
    "transplant_familywise_any": 1.0,
    "rule_a_fire_competent_independent": 0.65,
    "rule_a_fire_competent_paired_phi05": 0.1648,
    "rule_a_fire_delta_005": 0.0298,
    "rule_a_familywise_expected": 0.71,
    "rule_a_familywise_any": 0.516,
    "mcnemar_fire_competent_independent": 0.0061,
    "mcnemar_fire_competent_paired_phi05": 0.0005,
    "mcnemar_power_gap25_marginals_85_60": 0.3547,
    "mcnemar_power_gap20_marginals_90_70": 0.2598,
    "mcnemar_power_gap25_base_90": 0.4,
    "sign_bonferroni_power_gap25_marginals_85_60": 0.0542,
    "sign_bonferroni_power_gap20_marginals_90_70": 0.0242,
    "panel96_bonferroni_false_fire": 0.0008,
    "panel96_bonferroni_power_gap25_marginals_85_60": 0.7452,
    "panel96_bonferroni_power_gap25_base_90": 0.8329,
    "panel96_generations": 6912,
    "winner_pass_probability_95": 0.292,
    "winner_pass_probability_90": 0.08,
    "winner_pass_probability_80": 0.005,
    "screen_guess_probability": 1 / 216000,
    "screen_expected_exclusions_per_candidate": 24 / 216000,
    "screen_expected_exclusions_per_panel": 576 / 216000,
}
_O_RN_AGGREGATE_KEYS = {
    "arms",
    "audit",
    "headline",
    "ids",
    "sensitivities",
    "operating_characteristics",
    "n_cases",
    "n_incomplete",
}


def _o_rn_sha(text: str) -> str:
    """O rn sha."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _o_rn_digest_ok(value: Any) -> bool:
    """O rn digest ok."""
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _o_rn_table_shape(index: int):
    """O rn table shape."""
    return 4 + (index % 3), (_O_RN_FIELDS_4 if index % 2 else _O_RN_FIELDS_3)


def _o_rn_generate_cases(seed: int) -> list:
    """O rn generate cases."""
    cases = []
    for index in range(24):
        n_records, fields = _o_rn_table_shape(index)
        rng = random.Random(seed * 1000 + index)  # noqa: S311 # nosec B311
        values: list = []
        seen: set = set()
        while len(values) < n_records * len(fields):
            word = _O_RN_WORDS[rng.randrange(len(_O_RN_WORDS))]
            number = rng.randrange(1000, 10000)
            value = f"{word}-{number}"
            if value in seen:
                continue
            seen.add(value)
            values.append(value)
        table = [
            {field: values[r * len(fields) + j] for j, field in enumerate(fields)}
            for r in range(n_records)
        ]
        query_index = rng.randrange(n_records)
        cases.append(
            {
                "fixture_id": f"record-{index:02d}",
                "table": table,
                "query_index": query_index,
                "key": table[query_index]["zone"],
                "expected": table[query_index]["code"],
            }
        )
    return cases


def _o_rn_render_json(table: list) -> str:
    """O rn render json."""
    return json.dumps(table, sort_keys=True, separators=(", ", ": "), allow_nan=False)


def _o_rn_render_prose(table: list) -> str:
    """O rn render prose."""
    sentences = []
    for record in table:
        parts = [f"the {field} is {record[field]}" for field in sorted(record)]
        if len(parts) == 1:
            sentence = f"The {parts[0][4:]}."
        elif len(parts) == 2:
            sentence = f"The {parts[0][4:]} and {parts[1]}."
        else:
            sentence = "The " + ", ".join([parts[0][4:], *parts[1:-1]]) + f", and {parts[-1]}."
        sentences.append(sentence)
    return " ".join(sentences)


def _o_rn_arm_content(case: dict, arm: str) -> str:
    """O rn arm content."""
    question = _O_RN_QUESTION_TEMPLATE.format(key=case["key"])
    if arm == "closed_book":
        return f"{question} {_O_RN_CONTRACT}"
    if arm == "prose":
        rendering = _o_rn_render_prose(case["table"])
    else:
        rendering = _o_rn_render_json(case["table"])
    return f"{rendering}\n\n{question} {_O_RN_CONTRACT}"


def _o_rn_embedding_audit() -> dict:
    """O rn embedding audit."""
    tokens = set(_O_RN_FIELDS_4)
    tokens.update(re.findall(r"[a-z]+", _O_RN_QUESTION_TEMPLATE.replace("{key}", " ").lower()))
    tokens.update(re.findall(r"[a-z]+", _O_RN_CONTRACT.lower()))
    violations = 0
    for token in tokens:
        for word in _O_RN_WORDS:
            if token[:4] == word[:4] or word.startswith(token) or token.startswith(word):
                violations += 1
    return {
        "template_violations": violations,
        "key_word_prefix_cases": len(_O_RN_WORDS),
        "key_word_prefix_of": len(_O_RN_WORDS),
        "answer_key_same_word_rate": 8999 / 215999,
    }


def _o_rn_case_material() -> str:
    """O rn case material."""
    material = {
        "words": list(_O_RN_WORDS),
        "fields_3": list(_O_RN_FIELDS_3),
        "fields_4": list(_O_RN_FIELDS_4),
        "answer_field": "code",
        "key_field": "zone",
        "question_template": _O_RN_QUESTION_TEMPLATE,
        "contract": _O_RN_CONTRACT,
        "generator_spec": _O_RN_GENERATOR_SPEC,
        "case_count": 24,
        "arm_keys": list(_O_RN_ARMS),
        "embedding_audit": _o_rn_embedding_audit(),
    }
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return _o_rn_sha(canonical)


def _o_rn_ground(seed: int) -> str:
    """O rn ground."""
    rows = [
        [
            case["fixture_id"],
            [[[field, record[field]] for field in sorted(record)] for record in case["table"]],
            case["query_index"],
            case["key"],
            case["expected"],
        ]
        for case in _o_rn_generate_cases(seed)
    ]
    canonical = json.dumps(rows, separators=(",", ":"), ensure_ascii=True)
    return _o_rn_sha(canonical)


def _o_rn_protocol(seed: int, backend_fp: str) -> dict:
    """O rn protocol."""
    protocol = {
        "version": 1,
        "arm_keys": list(_O_RN_ARMS),
        "headline_arms": list(_O_RN_HEADLINE),
        "screen_arm": "closed_book",
        "answer_field": "code",
        "key_field": "zone",
        "case_count": 24,
        "code_space": 216000,
        "case_material_sha256": _o_rn_case_material(),
        "ground_sha256": _o_rn_ground(seed),
        "embedding_audit": _o_rn_embedding_audit(),
        "generation_budget": 1728,
        "headline_generations": 1152,
        "screen_generations": 576,
        "max_ground_records": 576,
        "max_records_bytes": _O_RN_MAX_BYTES,
        "max_candidates": 24,
        "seed": seed,
        "backend_fingerprint": backend_fp,
    }
    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    protocol["fingerprint"] = _o_rn_sha(canonical)
    return protocol


def _o_rn_scorer_r1(text: str) -> str:
    """O rn scorer r1."""
    out = text.strip()
    out = out.replace("\r\n", "\n").replace("\r", "\n")
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out


def _o_rn_scorer_fence(text: str) -> str:
    """O rn scorer fence."""
    lines = text.split("\n")
    if len(lines) >= 3 and re.fullmatch(r"```[^`]*", lines[0]) and lines[-1] == "```":
        return "\n".join(lines[1:-1]).strip("\n")
    return text


def _o_rn_cell(record: dict, arm: str):
    """O rn cell."""
    cells = record.get("cells")
    if not isinstance(cells, dict):
        return None
    cell = cells.get(arm)
    return cell if isinstance(cell, dict) else None


def _o_rn_detail(record: dict, arm: str) -> str:
    """O rn detail."""
    details = record.get("cell_details")
    if not isinstance(details, dict):
        return ""
    detail = details.get(arm, "")
    return detail if type(detail) is str else ""


def _o_rn_verdict(record: dict, arm: str) -> bool:
    """O rn verdict."""
    cell = _o_rn_cell(record, arm)
    return bool(cell is not None and cell["verdict"])


def _o_rn_screened(record: dict) -> bool:
    """O rn screened."""
    cell = _o_rn_cell(record, "closed_book")
    return bool(cell is not None and cell["verdict"])


def _o_rn_ids_identical(record: dict) -> bool:
    """O rn ids identical."""
    digests = record.get("id_digests")
    if not isinstance(digests, dict):
        return False
    return digests.get("prose") is not None and digests.get("prose") == digests.get(
        "canonical_json"
    )


def _o_rn_decompose_cell(cell: dict, expected_digest: str) -> str:
    """O rn decompose cell."""
    if cell.get("verdict") is True:
        return "pass"
    digest = cell.get("answer_digest")
    if digest is not None and digest == expected_digest:
        return "boundary_only"
    return "other_failure"


def _o_rn_decompose_pair(prose_cell: dict, json_cell: dict) -> str:
    """O rn decompose pair."""
    if prose_cell.get("answer_digest") == json_cell.get("answer_digest"):
        return "boundary_only"
    if prose_cell.get("boundary_found") and json_cell.get("boundary_found"):
        return "exact_only"
    return "mixed"


def _o_rn_parsed_hit(answer_text: str, expected: str) -> bool:
    """O rn parsed hit."""
    try:
        parsed = json.loads(answer_text)
    except ValueError:
        return False
    return type(parsed) is dict and parsed.get("code") == expected


def _o_rn_regrade_verdict(cell: dict, expected: str, transform) -> bool:
    """O rn regrade verdict."""
    text = cell.get("answer_text")
    if type(text) is not str:
        return False
    return transform(text) == expected and bool(cell.get("boundary_found"))


def _o_rn_aggregate(records: list, cases: list) -> dict:
    """O rn aggregate."""
    expected_by_fixture = {str(case["fixture_id"]): str(case["expected"]) for case in cases}
    arms: dict = {}
    for key in _O_RN_ARMS:
        n_completed = 0
        n_pass = 0
        n_context = 0
        n_failed = 0
        decomposition = dict.fromkeys(("pass", "boundary_only", "other_failure"), 0)
        n_failures = 0
        echo_hits = 0
        parsed_hits = 0
        for record in records:
            detail = _o_rn_detail(record, key)
            if detail == "arm_context_exceeded":
                n_context += 1
            elif detail == "arm_evaluation_failed":
                n_failed += 1
            cell = _o_rn_cell(record, key)
            if cell is None:
                continue
            n_completed += 1
            if cell["verdict"]:
                n_pass += 1
            decomposition[_o_rn_decompose_cell(cell, str(record.get("expected_digest", "")))] += 1
            if key in _O_RN_HEADLINE and not cell["verdict"]:
                n_failures += 1
                text = cell.get("answer_text")
                expected = expected_by_fixture.get(str(record.get("fixture", "")), "")
                if type(text) is str:
                    if expected in text:
                        echo_hits += 1
                    if _o_rn_parsed_hit(text, expected):
                        parsed_hits += 1
        arms[key] = {
            "n_completed": n_completed,
            "n_pass": n_pass,
            "rate": (n_pass / n_completed) if n_completed else None,
            "n_context_exceeded": n_context,
            "n_evaluation_failed": n_failed,
            "decomposition": decomposition,
            "n_failures": n_failures,
            "echo_rate": ((echo_hits / n_failures) if n_failures else None)
            if key in _O_RN_HEADLINE
            else None,
            "parsed_rate": ((parsed_hits / n_failures) if n_failures else None)
            if key in _O_RN_HEADLINE
            else None,
        }
    both = [
        record
        for record in records
        if _o_rn_cell(record, "prose") is not None
        and _o_rn_cell(record, "canonical_json") is not None
    ]
    unscreened_pairs = [record for record in both if not _o_rn_ids_identical(record)]
    live_pairs = [record for record in unscreened_pairs if not _o_rn_screened(record)]

    def _discordance(pairs: list) -> dict:
        """Discordance."""
        prose_only = 0
        json_only = 0
        decomposition = dict.fromkeys(("exact_only", "boundary_only", "mixed"), 0)
        for record in pairs:
            prose_cell = _o_rn_cell(record, "prose")
            json_cell = _o_rn_cell(record, "canonical_json")
            if prose_cell is None or json_cell is None:
                continue
            if bool(prose_cell["verdict"]) == bool(json_cell["verdict"]):
                continue
            if prose_cell["verdict"]:
                prose_only += 1
            else:
                json_only += 1
            decomposition[_o_rn_decompose_pair(prose_cell, json_cell)] += 1
        return {
            "n_paired": len(pairs),
            "discordant": prose_only + json_only,
            "n_prose_only": prose_only,
            "n_canonical_json_only": json_only,
            "decomposition": decomposition,
        }

    headline_counts = _discordance(live_pairs)
    headline = {
        **headline_counts,
        "prose_rate": (
            (sum(1 for record in live_pairs if _o_rn_verdict(record, "prose")) / len(live_pairs))
            if live_pairs
            else None
        ),
        "canonical_json_rate": (
            (
                sum(1 for record in live_pairs if _o_rn_verdict(record, "canonical_json"))
                / len(live_pairs)
            )
            if live_pairs
            else None
        ),
        "resolution_points": (100.0 / len(live_pairs)) if live_pairs else None,
    }
    deltas: list = []
    n_same_ids_flips = 0
    for record in records:
        tokens = record.get("arm_tokens")
        tokens = tokens if isinstance(tokens, dict) else {}
        prose_tokens = tokens.get("prose")
        json_tokens = tokens.get("canonical_json")
        if type(prose_tokens) is int and type(json_tokens) is int:
            deltas.append(json_tokens - prose_tokens)
        else:
            deltas.append(0)
        if _o_rn_ids_identical(record):
            prose_cell = _o_rn_cell(record, "prose")
            json_cell = _o_rn_cell(record, "canonical_json")
            if (
                prose_cell is not None
                and json_cell is not None
                and prose_cell["verdict"] != json_cell["verdict"]
            ):
                n_same_ids_flips += 1
    ids = {
        "n_ids_identical": sum(1 for record in records if _o_rn_ids_identical(record)),
        "id_length_delta": deltas,
        "max_abs_id_delta": max((abs(delta) for delta in deltas), default=0),
        "n_same_ids_verdict_flips": n_same_ids_flips,
    }
    regrade: dict = {}
    for label, transform in (("r1", _o_rn_scorer_r1), ("fence", _o_rn_scorer_fence)):
        count = 0
        for record in live_pairs:
            prose_cell = _o_rn_cell(record, "prose")
            json_cell = _o_rn_cell(record, "canonical_json")
            if prose_cell is None or json_cell is None:
                continue
            expected = expected_by_fixture.get(str(record.get("fixture", "")), "")
            if _o_rn_regrade_verdict(prose_cell, expected, transform) != _o_rn_regrade_verdict(
                json_cell, expected, transform
            ):
                count += 1
        regrade[label] = count
    sensitivities = {
        "unscreened": _discordance(unscreened_pairs),
        "regrade_r1_discordant": regrade["r1"],
        "regrade_fence_discordant": regrade["fence"],
    }
    return {
        "arms": arms,
        "audit": {
            "n_screened": sum(1 for record in records if _o_rn_screened(record)),
            "n_closed_book_incomplete": sum(
                1 for record in records if _o_rn_cell(record, "closed_book") is None
            ),
        },
        "headline": headline,
        "ids": ids,
        "sensitivities": sensitivities,
        "operating_characteristics": dict(_O_RN_OC),
        "n_cases": len(records),
        "n_incomplete": sum(
            1 for record in records if any(_o_rn_cell(record, arm) is None for arm in _O_RN_ARMS)
        ),
    }


def _o_rn_recompute(payload: Any):
    """O rn recompute."""
    if type(payload) is not dict:
        return None
    protocol = payload.get("protocol")
    if type(protocol) is not dict:
        return None
    seed = protocol.get("seed")
    backend_fp = protocol.get("backend_fingerprint")
    if type(seed) is not int or isinstance(seed, bool):
        return None
    if type(backend_fp) is not str:
        return None
    if protocol != _o_rn_protocol(seed, backend_fp):
        return None
    cases = _o_rn_generate_cases(seed)
    if len(cases) != protocol["case_count"]:
        return None
    candidates = payload.get("candidates")
    if type(candidates) is not dict or not candidates:
        return None
    if len(candidates) > 24:
        return None
    total_records = 0
    records_bytes = 0
    aggregates: dict = {}
    for name, entry in candidates.items():
        if type(entry) is not dict:
            return None
        if entry.get("status") == "unavailable":
            if set(entry) != {"status", "detail", "records"} or entry.get("records") != []:
                return None
            continue
        if entry.get("status") != "available" or entry.get("detail") != "":
            return None
        if set(entry) != _O_RN_AGGREGATE_KEYS | {"status", "detail", "records"}:
            return None
        records = entry.get("records")
        if type(records) is not list or len(records) != len(cases):
            return None
        total_records += len(records)
        records_bytes += len(
            json.dumps(records, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        )
        if records_bytes > _O_RN_MAX_BYTES:
            return None
        for record, case in zip(records, cases, strict=True):
            if type(record) is not dict or set(record) != {
                "fixture",
                "n_records",
                "n_fields",
                "expected_digest",
                "content_digests",
                "arm_tokens",
                "id_digests",
                "ids_identical",
                "id_length_delta",
                "cells",
                "cell_details",
            }:
                return None
            if record["fixture"] != case["fixture_id"]:
                return None
            if record["n_records"] != len(case["table"]) or record["n_fields"] != len(
                case["table"][0]
            ):
                return None
            if record["expected_digest"] != _o_rn_sha(str(case["expected"])):
                return None
            for key in ("arm_tokens", "id_digests", "content_digests", "cells", "cell_details"):
                mapping = record[key]
                if type(mapping) is not dict or set(mapping) != set(_O_RN_ARMS):
                    return None
            if record["content_digests"] != {
                arm: _o_rn_sha(_o_rn_arm_content(case, arm)) for arm in _O_RN_ARMS
            }:
                return None
            if record["ids_identical"] is not (
                record["id_digests"]["prose"] == record["id_digests"]["canonical_json"]
            ):
                return None
            for arm in _O_RN_ARMS:
                tokens = record["arm_tokens"][arm]
                if type(tokens) is not int or isinstance(tokens, bool) or tokens < 0:
                    return None
                if not _o_rn_digest_ok(record["id_digests"][arm]):
                    return None
                detail = record["cell_details"][arm]
                if detail not in _O_RN_CELL_DETAILS:
                    return None
                cell = record["cells"][arm]
                if cell is None:
                    if detail == "":
                        return None
                    continue
                if detail != "":
                    return None
                headline = arm in _O_RN_HEADLINE
                fields = _O_RN_HEADLINE_CELL_FIELDS if headline else _O_RN_SCREEN_CELL_FIELDS
                if type(cell) is not dict or set(cell) != fields:
                    return None
                if type(cell["verdict"]) is not bool or type(cell["boundary_found"]) is not bool:
                    return None
                if (
                    cell["boundary_text_match"] is not None
                    and type(cell["boundary_text_match"]) is not bool
                ):
                    return None
                if type(cell["input_tokens"]) is not int or cell["input_tokens"] != tokens:
                    return None
                digest = cell["answer_digest"]
                if digest is not None and not _o_rn_digest_ok(digest):
                    return None
                if headline:
                    text = cell["answer_text"]
                    if text is None:
                        if digest is not None:
                            return None
                    elif type(text) is not str or _o_rn_sha(text) != digest:
                        return None
                if cell["verdict"] != (
                    digest == record["expected_digest"] and cell["boundary_found"]
                ):
                    return None
            if record["id_length_delta"] != (
                record["arm_tokens"]["canonical_json"] - record["arm_tokens"]["prose"]
            ):
                return None
        aggregates[name] = _o_rn_aggregate(records, cases)
    if total_records > 576:
        return None
    if not aggregates:
        return None
    return {"aggregates": aggregates, "n_candidates": len(candidates), "n_cases": len(cases)}


def _o_recordnotation_valid(value: Any) -> bool:
    """Oracle record-notation validator: mirrors
    diffreports._valid_record_notation.
    """
    if type(value) is not dict or value.get("available") is not True:
        return False
    if set(value) != {
        "available",
        "detail",
        "protocol",
        "audit",
        "candidates",
        "n_candidates",
        "n_cases",
    }:
        return False
    if value.get("detail") != "":
        return False
    audit = value.get("audit")
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
    recomputed = _o_rn_recompute(value)
    if recomputed is None:
        return False
    candidates = value["candidates"]
    for name, expected in recomputed["aggregates"].items():
        entry = candidates[name]
        for key, val in expected.items():
            if entry.get(key) != val:
                return False
    if value.get("n_candidates") != recomputed["n_candidates"]:
        return False
    return bool(value.get("n_cases") == recomputed["n_cases"])


def _o_recordnotation(old: Any, new: Any) -> tuple[dict[str, Any], bool, bool]:
    """Oracle record-notation section: mirrors
    diffreports.diff_record_notation.
    """
    old_ok = _o_recordnotation_valid(old)
    new_ok = _o_recordnotation_valid(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    diff: dict[str, Any] = {"available": available, "changed": False}
    asymmetric = available not in ("both", "neither")
    if available != "both":
        return diff, False, asymmetric
    protocol_drift = old["protocol"].get("fingerprint") != new["protocol"].get("fingerprint")
    generator_drift = old["protocol"].get("ground_sha256") != new["protocol"].get(
        "ground_sha256"
    ) or old["protocol"].get("case_material_sha256") != new["protocol"].get("case_material_sha256")
    backend_drift = old["protocol"].get("backend_fingerprint") != new["protocol"].get(
        "backend_fingerprint"
    )
    comparable = not protocol_drift and not generator_drift and not backend_drift
    old_candidates = old["candidates"]
    new_candidates = new["candidates"]
    added = sorted(set(new_candidates) - set(old_candidates))
    removed = sorted(set(old_candidates) - set(new_candidates))
    candidate_drift: list = []
    for name in sorted(set(old_candidates) & set(new_candidates)):
        old_entry = old_candidates[name]
        new_entry = new_candidates[name]
        if old_entry.get("status") != "available" or new_entry.get("status") != "available":
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
        metrics: list = [
            ("n_screened", old_entry["audit"]["n_screened"], new_entry["audit"]["n_screened"]),
            (
                "n_closed_book_incomplete",
                old_entry["audit"]["n_closed_book_incomplete"],
                new_entry["audit"]["n_closed_book_incomplete"],
            ),
            ("n_paired", old_entry["headline"]["n_paired"], new_entry["headline"]["n_paired"]),
            (
                "discordant",
                old_entry["headline"]["discordant"],
                new_entry["headline"]["discordant"],
            ),
            (
                "n_prose_only",
                old_entry["headline"]["n_prose_only"],
                new_entry["headline"]["n_prose_only"],
            ),
            (
                "n_canonical_json_only",
                old_entry["headline"]["n_canonical_json_only"],
                new_entry["headline"]["n_canonical_json_only"],
            ),
            (
                "prose_rate",
                old_entry["headline"]["prose_rate"],
                new_entry["headline"]["prose_rate"],
            ),
            (
                "canonical_json_rate",
                old_entry["headline"]["canonical_json_rate"],
                new_entry["headline"]["canonical_json_rate"],
            ),
            (
                "decomposition_exact_only",
                old_entry["headline"]["decomposition"]["exact_only"],
                new_entry["headline"]["decomposition"]["exact_only"],
            ),
            (
                "decomposition_boundary_only",
                old_entry["headline"]["decomposition"]["boundary_only"],
                new_entry["headline"]["decomposition"]["boundary_only"],
            ),
            (
                "decomposition_mixed",
                old_entry["headline"]["decomposition"]["mixed"],
                new_entry["headline"]["decomposition"]["mixed"],
            ),
            (
                "prose_echo_rate",
                old_entry["arms"]["prose"]["echo_rate"],
                new_entry["arms"]["prose"]["echo_rate"],
            ),
            (
                "prose_parsed_rate",
                old_entry["arms"]["prose"]["parsed_rate"],
                new_entry["arms"]["prose"]["parsed_rate"],
            ),
            (
                "canonical_json_echo_rate",
                old_entry["arms"]["canonical_json"]["echo_rate"],
                new_entry["arms"]["canonical_json"]["echo_rate"],
            ),
            (
                "canonical_json_parsed_rate",
                old_entry["arms"]["canonical_json"]["parsed_rate"],
                new_entry["arms"]["canonical_json"]["parsed_rate"],
            ),
            (
                "n_ids_identical",
                old_entry["ids"]["n_ids_identical"],
                new_entry["ids"]["n_ids_identical"],
            ),
            (
                "max_abs_id_delta",
                old_entry["ids"]["max_abs_id_delta"],
                new_entry["ids"]["max_abs_id_delta"],
            ),
            (
                "n_same_ids_verdict_flips",
                old_entry["ids"]["n_same_ids_verdict_flips"],
                new_entry["ids"]["n_same_ids_verdict_flips"],
            ),
            (
                "unscreened_discordant",
                old_entry["sensitivities"]["unscreened"]["discordant"],
                new_entry["sensitivities"]["unscreened"]["discordant"],
            ),
            (
                "regrade_r1_discordant",
                old_entry["sensitivities"]["regrade_r1_discordant"],
                new_entry["sensitivities"]["regrade_r1_discordant"],
            ),
            (
                "regrade_fence_discordant",
                old_entry["sensitivities"]["regrade_fence_discordant"],
                new_entry["sensitivities"]["regrade_fence_discordant"],
            ),
        ]
        for metric, old_value, new_value in metrics:
            if old_value != new_value:
                candidate_drift.append(
                    {"candidate": name, "metric": metric, "old": old_value, "new": new_value}
                )
    diff["candidate_drift"] = candidate_drift
    diff["candidates_added"] = added
    diff["candidates_removed"] = removed
    diff["protocol_drift"] = protocol_drift
    diff["generator_drift"] = generator_drift
    diff["backend_drift"] = backend_drift
    diff["comparable"] = comparable
    changed = bool(
        candidate_drift or added or removed or protocol_drift or generator_drift or backend_drift
    )
    diff["changed"] = changed
    return diff, changed, asymmetric


# ---------------------------------------------------------------------------
# Invisible characters (cycle 223): independent oracle for
# diffreports._valid_invisible_chars / diff_invisible_chars,
# re-implemented from docs/INVISIBLECHARS_DESIGN.md as
# amended (A1-A10). The generator's digest-to-integer
# conversion is frozen (amendment A1); every cell carries
# digest-bound answer_text; the headline is the guarded
# classification over the id-divergent paired population
# (amendments A3/A4); the decomposition is the ordered
# decision list (amendment A5) with the echo/form steps on
# the five invisible arms only (amendment A6).


_O_IC_TOPICS = (
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
    "pepper",
    "quartz",
    "birch",
)
_O_IC_WORDS = (
    "alder",
    "basalt",
    "cedar",
    "drift",
    "elm",
    "fern",
    "glade",
    "heath",
    "iris",
    "jade",
    "kelp",
    "loom",
    "moss",
    "nova",
    "ochre",
    "pine",
    "quill",
    "reed",
    "sage",
    "tide",
    "urn",
    "vale",
    "wren",
    "yew",
)
_O_IC_NEUTRAL_STATEMENTS = (
    "The weather changed slowly during the afternoon hours.",
    "A wooden fence runs along the edge of the property.",
    "She placed the documents inside a plain folder yesterday.",
    "The train arrived later than the schedule suggested.",
    "Fresh bread was cooling on the counter near the sink.",
    "He repaired the bicycle with tools from the garage.",
    "The bookshop extends its opening hours during summer.",
    "Several chairs were arranged around the long table.",
    "The recipe calls for a pinch of salt and herbs.",
    "They walked along the path beside the lake at dawn.",
    "The committee postponed its decision until next week.",
    "A thick layer of dust covered the old furniture.",
    "The children played quietly in the courtyard after lunch.",
    "His handwriting was difficult to read in places.",
    "The farmer harvested the fields before the rain came.",
    "An old map hung framed on the wall of the hallway.",
    "The accountant reviewed the figures a second time.",
    "Birds gathered on the rooftops as the sun went down.",
    "The mechanic inspected the engine and changed the oil.",
    "A narrow staircase led up to the attic.",
    "The orchestra rehearsed the piece several times over.",
    "Someone left a bicycle leaning against the railing.",
    "The soup simmered gently on the stove all morning.",
    "Visitors signed the guest book near the entrance.",
)
_O_IC_ARM_CODEPOINTS = {
    "control": None,
    "zwsp": "\u200b",
    "vs16": "\ufe0f",
    "tag": "\U000e0001",
    "softhyphen": "\u00ad",
    "wj": "\u2060",
    "visible_hyphen": "-",
}
_O_IC_ARM_KEYS = ("control", "zwsp", "vs16", "tag", "softhyphen", "wj", "visible_hyphen")
_O_IC_TREATED_ARMS = _O_IC_ARM_KEYS[1:]
_O_IC_INVISIBLE_ARMS = ("zwsp", "vs16", "tag", "softhyphen", "wj")
_O_IC_CASE_COUNT = 24
_O_IC_MIN_D = 10
_O_IC_MIN_J = 6
_O_IC_MAX_INCOMPLETE = 6
_O_IC_REDRAW_LIMIT = 1000
_O_IC_GENERATOR_SPEC = (
    "draw(seed, index, slot, counter) = int.from_bytes("
    "sha256(f'{seed}|invisiblechars|{index}|{slot}|{counter}')"
    ".digest()[:8], 'big'); slot 0: word = WORDS[draw % len(WORDS)]; "
    "slot 1: number = 1000 + draw % 9000; a value colliding with an "
    "earlier case redraws both slots at counter+1, giving up after "
    "1000 consecutive redraws (the stage fails closed); slots 2/3: "
    "neutral indices = draw % len(NEUTRAL_STATEMENTS), the second "
    "advancing +1 on collision"
)
_O_IC_MAX_CANDIDATES = 24
_O_IC_MAX_GROUND_RECORDS = _O_IC_MAX_CANDIDATES * _O_IC_CASE_COUNT
_O_IC_GENERATION_BUDGET = len(_O_IC_ARM_KEYS) * _O_IC_MAX_GROUND_RECORDS
_O_IC_MAX_RECORDS_BYTES = 12 * 1024 * 1024
_O_IC_CELL_FIELDS = {
    "verdict",
    "boundary_found",
    "boundary_text_match",
    "input_tokens",
    "answer_digest",
    "answer_text",
}
_O_IC_DETAILS = ("arm_context_exceeded", "arm_evaluation_failed")
_O_IC_CELL_CLASSES = (
    "pass",
    "boundary_only",
    "wrong_value",
    "echo_invisible",
    "form_only",
    "non_answer",
)
_O_IC_PAIR_CLASSES = ("exact_only", "boundary_only", "mixed")
_O_IC_OC = {
    "fire_competent_independent": 0.0173,
    "fire_competent_paired": 0.0044,
    "fire_weak_independent": 0.1743,
    "fire_random_independent": 0.5503,
    "power_drop10_base90": 0.2542,
    "power_drop25_base90": 0.8159,
    "power_drop25_base85": 0.8761,
    "power_drop25_paired": 0.6981,
    "power_drop50_base90": 0.999,
    "sensitivity18_drop25": 0.5214,
    "sensitivity18_competent": 0.0038,
    "panel_sweep_16_drop25": 0.3904,
    "panel_sweep_20_drop25": 0.6395,
    "transplant_fire_competent_independent": 0.8897,
    "familywise_competent_expected_120": 2.07,
    "familywise_competent_any_120": 0.8764,
    "familywise_competent_expected_144": 2.49,
    "familywise_competent_any_144": 0.9187,
    "familywise_paired_expected_120": 0.53,
    "familywise_paired_any_120": 0.4139,
    "familywise_random_expected_120": 66.04,
    "world_a_guarded_fire": 0.0014,
    "world_b_unguarded_below_threshold": 0.7285,
    "world_b_guarded_below_threshold": 0.0133,
    "world_c_guarded_fire": 0.8932,
    "world_c_unguarded_fire": 0.9408,
}
_O_IC_AGGREGATE_KEYS = {
    "arms",
    "ids",
    "headline",
    "operating_characteristics",
    "n_cases",
    "n_incomplete",
}


def _o_ic_sha(text: str) -> str:
    """O ic sha."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _o_ic_digest_ok(value: Any) -> bool:
    """O ic digest ok."""
    if type(value) is not str or len(value) != 64:
        return False
    return all(ch in "0123456789abcdef" for ch in value)


def _o_ic_derive(seed: int, index: int, slot: int, counter: int) -> int:
    """O ic derive."""
    material = f"{seed}|invisiblechars|{index}|{slot}|{counter}"
    return int.from_bytes(hashlib.sha256(material.encode("ascii")).digest()[:8], "big")


def _o_ic_generate(seed: int):
    """O ic generate."""
    cases: list = []
    seen: set = set()
    for index in range(_O_IC_CASE_COUNT):
        counter = 0
        while True:
            word = _O_IC_WORDS[_o_ic_derive(seed, index, 0, counter) % len(_O_IC_WORDS)]
            number = 1000 + _o_ic_derive(seed, index, 1, counter) % 9000
            value = f"{word}-{number}"
            if value not in seen:
                break
            counter += 1
            if counter > _O_IC_REDRAW_LIMIT:
                return None
        seen.add(value)
        first = _o_ic_derive(seed, index, 2, 0) % len(_O_IC_NEUTRAL_STATEMENTS)
        second = _o_ic_derive(seed, index, 3, 0) % len(_O_IC_NEUTRAL_STATEMENTS)
        if second == first:
            second = (first + 1) % len(_O_IC_NEUTRAL_STATEMENTS)
        cases.append(
            {
                "fixture_id": f"invisiblechars-{index:02d}",
                "topic": _O_IC_TOPICS[index],
                "value": value,
                "expected": value,
                "neutral_indices": [first, second],
            }
        )
    return cases


def _o_ic_fact(case: dict) -> str:
    """O ic fact."""
    return f"The {case['topic']} inspection code is {case['value']}."


def _o_ic_question(case: dict) -> str:
    """O ic question."""
    return f"What is the {case['topic']} inspection code?"


def _o_ic_treated_fact(case: dict, code_point: str) -> str:
    """O ic treated fact."""
    topic = str(case["topic"])
    word, _, number = str(case["value"]).partition("-")
    treated_topic = topic[:2] + code_point + topic[2:]
    treated_value = word[:2] + code_point + word[2:] + "-" + number
    return f"The {treated_topic} inspection code is {treated_value}."


def _o_ic_arm_content(case: dict, arm: str) -> str:
    """O ic arm content."""
    first, second = (int(i) for i in case["neutral_indices"])
    code_point = _O_IC_ARM_CODEPOINTS[arm]
    fact = _o_ic_fact(case) if code_point is None else _o_ic_treated_fact(case, code_point)
    return (
        f"{_O_IC_NEUTRAL_STATEMENTS[first]} {fact} "
        f"{_O_IC_NEUTRAL_STATEMENTS[second]} {_o_ic_question(case)}"
    )


def _o_ic_population_audit() -> dict:
    """O ic population audit."""
    arms: dict = {}
    ok = True
    for arm in (*_O_IC_INVISIBLE_ARMS, "visible_hyphen"):
        code_point = _O_IC_ARM_CODEPOINTS[arm]
        try:
            name = unicodedata.name(code_point)
        except ValueError:
            name = ""
        category = unicodedata.category(code_point)
        combining = unicodedata.combining(code_point)
        satisfies = bool(name) and category in ("Cf", "Mn") and combining == 0
        if arm in _O_IC_INVISIBLE_ARMS and not satisfies:
            ok = False
        arms[arm] = {
            "code_point": f"U+{ord(code_point):04X}",
            "name": name,
            "category": category,
            "combining_class": combining,
            "satisfies_predicate": satisfies,
        }
    return {"arms": arms, "ok": ok, "unicode_version": unicodedata.unidata_version}


def _o_ic_case_material_sha256() -> str:
    """O ic case material sha256."""
    material = {
        "topics": list(_O_IC_TOPICS),
        "words": list(_O_IC_WORDS),
        "neutral_statements": list(_O_IC_NEUTRAL_STATEMENTS),
        "arm_codepoints": {arm: _O_IC_ARM_CODEPOINTS[arm] for arm in _O_IC_ARM_KEYS},
        "generator_spec": _O_IC_GENERATOR_SPEC,
        "case_count": _O_IC_CASE_COUNT,
        "arm_keys": list(_O_IC_ARM_KEYS),
        "min_d": _O_IC_MIN_D,
        "min_j": _O_IC_MIN_J,
        "max_incomplete": _O_IC_MAX_INCOMPLETE,
        "redraw_limit": _O_IC_REDRAW_LIMIT,
        "population_audit": _o_ic_population_audit(),
    }
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return _o_ic_sha(canonical)


def _o_ic_ground_sha256(seed: int):
    """O ic ground sha256."""
    generated = _o_ic_generate(seed)
    if generated is None:
        return None
    rows = [
        [case["fixture_id"], case["topic"], case["value"], list(case["neutral_indices"])]
        for case in generated
    ]
    canonical = json.dumps(rows, separators=(",", ":"), ensure_ascii=True)
    return _o_ic_sha(canonical)


def _o_ic_protocol(seed: int, backend_fingerprint: str):
    """O ic protocol."""
    ground = _o_ic_ground_sha256(seed)
    if ground is None:
        return None
    protocol: dict = {
        "version": 1,
        "arm_keys": list(_O_IC_ARM_KEYS),
        "treated_arms": list(_O_IC_TREATED_ARMS),
        "invisible_arms": list(_O_IC_INVISIBLE_ARMS),
        "case_count": _O_IC_CASE_COUNT,
        "min_d": _O_IC_MIN_D,
        "min_j": _O_IC_MIN_J,
        "max_incomplete": _O_IC_MAX_INCOMPLETE,
        "case_material_sha256": _o_ic_case_material_sha256(),
        "ground_sha256": ground,
        "population_audit": _o_ic_population_audit(),
        "generation_budget": _O_IC_GENERATION_BUDGET,
        "max_ground_records": _O_IC_MAX_GROUND_RECORDS,
        "max_records_bytes": _O_IC_MAX_RECORDS_BYTES,
        "max_candidates": _O_IC_MAX_CANDIDATES,
        "seed": seed,
        "backend_fingerprint": backend_fingerprint,
    }
    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    protocol["fingerprint"] = _o_ic_sha(canonical)
    return protocol


def _o_ic_cell(record: dict, key: str):
    """O ic cell."""
    cells = record.get("cells")
    if not isinstance(cells, dict):
        return None
    cell = cells.get(key)
    return cell if isinstance(cell, dict) else None


def _o_ic_detail(record: dict, key: str) -> str:
    """O ic detail."""
    details = record.get("cell_details")
    if not isinstance(details, dict):
        return ""
    detail = details.get(key, "")
    return detail if type(detail) is str else ""


def _o_ic_verdict(record: dict, key: str) -> bool:
    """O ic verdict."""
    cell = _o_ic_cell(record, key)
    return bool(cell is not None and cell["verdict"])


def _o_ic_ids_identical(record: dict, arm: str) -> bool:
    """O ic ids identical."""
    digests = record.get("id_digests")
    if not isinstance(digests, dict):
        return False
    control = digests.get("control")
    return type(control) is str and control != "" and control == digests.get(arm)


def _o_ic_decompose_cell(cell: dict, expected: str, expected_digest: str, arm: str) -> str:
    """O ic decompose cell."""
    if cell.get("verdict") is True:
        return "pass"
    digest = cell.get("answer_digest")
    if digest is not None and digest == expected_digest:
        return "boundary_only"
    text = cell.get("answer_text")
    if type(text) is not str:
        return "non_answer"
    if text != expected and re.match(r"^[a-z]+-[0-9]{4}$", text):
        word, _, number = text.partition("-")
        if word in _O_IC_WORDS and 1000 <= int(number) <= 9999:
            return "wrong_value"
    code_point = _O_IC_ARM_CODEPOINTS[arm]
    if arm in _O_IC_INVISIBLE_ARMS and code_point is not None:
        word, _, number = expected.partition("-")
        site_echo = word[:2] + code_point + word[2:] + "-" + number
        if text == site_echo:
            return "echo_invisible"
        if text != expected and text.replace(code_point, "") == expected:
            return "form_only"
    return "non_answer"


def _o_ic_decompose_pair(control_cell: dict, arm_cell: dict) -> str:
    """O ic decompose pair."""
    if control_cell.get("answer_digest") == arm_cell.get("answer_digest"):
        return "boundary_only"
    if control_cell.get("boundary_found") and arm_cell.get("boundary_found"):
        return "exact_only"
    return "mixed"


def _o_ic_classify(control_passes: int, wrong_values: int, n_incomplete: int) -> str:
    """O ic classify."""
    if n_incomplete > _O_IC_MAX_INCOMPLETE:
        return "insufficient"
    if control_passes < _O_IC_MIN_D:
        return "insufficient"
    if wrong_values >= _O_IC_MIN_J:
        return "discordant_asymmetric"
    if wrong_values == 0:
        return "no_discordance_observed" if n_incomplete == 0 else "insufficient"
    return "below_threshold" if wrong_values + n_incomplete < _O_IC_MIN_J else "insufficient"


def _o_ic_rate(numerator: int, denominator: int):
    """O ic rate."""
    return (numerator / denominator) if denominator else None


def _o_ic_aggregate(records: list, cases: list) -> dict:
    """O ic aggregate."""
    expected_by_fixture = {str(case["fixture_id"]): str(case["expected"]) for case in cases}
    arms: dict = {}
    for key in _O_IC_ARM_KEYS:
        n_completed = 0
        n_pass = 0
        n_context = 0
        n_failed = 0
        decomposition = dict.fromkeys(_O_IC_CELL_CLASSES, 0)
        for record in records:
            detail = _o_ic_detail(record, key)
            if detail == "arm_context_exceeded":
                n_context += 1
            elif detail == "arm_evaluation_failed":
                n_failed += 1
            cell = _o_ic_cell(record, key)
            if cell is None:
                continue
            n_completed += 1
            if cell["verdict"]:
                n_pass += 1
            decomposition[
                _o_ic_decompose_cell(
                    cell,
                    expected_by_fixture.get(str(record.get("fixture", "")), ""),
                    str(record.get("expected_digest", "")),
                    key,
                )
            ] += 1
        arms[key] = {
            "n_completed": n_completed,
            "n_pass": n_pass,
            "rate": _o_ic_rate(n_pass, n_completed),
            "n_context_exceeded": n_context,
            "n_evaluation_failed": n_failed,
            "decomposition": decomposition,
        }
    ids: dict = {}
    for arm in _O_IC_TREATED_ARMS:
        deltas: list = []
        n_ids_identical = 0
        n_same_ids_flips = 0
        for record in records:
            tokens = record.get("arm_tokens")
            tokens = tokens if isinstance(tokens, dict) else {}
            treated = tokens.get(arm)
            baseline = tokens.get("control")
            if type(treated) is int and type(baseline) is int:
                deltas.append(treated - baseline)
            else:
                deltas.append(0)
            if _o_ic_ids_identical(record, arm):
                n_ids_identical += 1
                control_cell = _o_ic_cell(record, "control")
                arm_cell = _o_ic_cell(record, arm)
                if (
                    control_cell is not None
                    and arm_cell is not None
                    and control_cell["verdict"] != arm_cell["verdict"]
                ):
                    n_same_ids_flips += 1
        ids[arm] = {
            "n_ids_identical": n_ids_identical,
            "n_ids_divergent": len(records) - n_ids_identical,
            "id_length_delta": deltas,
            "max_abs_id_delta": max((abs(delta) for delta in deltas), default=0),
            "n_same_ids_verdict_flips": n_same_ids_flips,
        }
    headline: dict = {}
    for arm in _O_IC_TREATED_ARMS:
        domain = [record for record in records if not _o_ic_ids_identical(record, arm)]
        paired = [
            record
            for record in domain
            if _o_ic_cell(record, "control") is not None and _o_ic_cell(record, arm) is not None
        ]
        n_incomplete = len(domain) - len(paired)
        control_passes = 0
        wrong_values = 0
        reverse = 0
        non_answer = 0
        pair_decomposition = dict.fromkeys(_O_IC_PAIR_CLASSES, 0)
        failing_classes = dict.fromkeys(_O_IC_CELL_CLASSES, 0)
        for record in paired:
            control_cell = _o_ic_cell(record, "control")
            arm_cell = _o_ic_cell(record, arm)
            if control_cell is None or arm_cell is None:
                continue
            expected = expected_by_fixture.get(str(record.get("fixture", "")), "")
            expected_digest = str(record.get("expected_digest", ""))
            arm_class = _o_ic_decompose_cell(arm_cell, expected, expected_digest, arm)
            if arm_class == "non_answer":
                non_answer += 1
            if control_cell["verdict"]:
                control_passes += 1
                if arm_class == "wrong_value":
                    wrong_values += 1
            elif arm_cell["verdict"]:
                reverse += 1
            if control_cell["verdict"] != arm_cell["verdict"]:
                pair_decomposition[_o_ic_decompose_pair(control_cell, arm_cell)] += 1
                failing = arm_cell if control_cell["verdict"] else control_cell
                failing_arm = arm if control_cell["verdict"] else "control"
                failing_classes[
                    _o_ic_decompose_cell(failing, expected, expected_digest, failing_arm)
                ] += 1
        headline[arm] = {
            "label": _o_ic_classify(control_passes, wrong_values, n_incomplete),
            "n_population": len(paired),
            "n_incomplete": n_incomplete,
            "control_passes": control_passes,
            "wrong_values": wrong_values,
            "n_reverse": reverse,
            "control_rate": _o_ic_rate(control_passes, len(paired)),
            "arm_rate": _o_ic_rate(
                sum(1 for record in paired if _o_ic_verdict(record, arm)), len(paired)
            ),
            "n_non_answer": non_answer,
            "pair_decomposition": pair_decomposition,
            "failing_classes": failing_classes,
        }
    return {
        "arms": arms,
        "ids": ids,
        "headline": headline,
        "operating_characteristics": dict(_O_IC_OC),
        "n_cases": len(records),
        "n_incomplete": sum(
            1
            for record in records
            if any(_o_ic_cell(record, arm) is None for arm in _O_IC_ARM_KEYS)
        ),
    }


def _o_ic_cell_ok(cell: Any, expected_digest: str) -> bool:
    """O ic cell ok."""
    if type(cell) is not dict or set(cell) != _O_IC_CELL_FIELDS:
        return False
    if type(cell["verdict"]) is not bool or type(cell["boundary_found"]) is not bool:
        return False
    if cell["boundary_text_match"] is not None and type(cell["boundary_text_match"]) is not bool:
        return False
    if type(cell["input_tokens"]) is not int or cell["input_tokens"] < 0:
        return False
    digest = cell["answer_digest"]
    if digest is not None and not _o_ic_digest_ok(digest):
        return False
    text = cell["answer_text"]
    if text is None:
        if digest is not None:
            return False
    elif type(text) is not str or _o_ic_sha(text) != digest:
        return False
    verdict = (digest == expected_digest) and cell["boundary_found"]
    return bool(cell["verdict"] == verdict)


def _o_ic_recompute(payload: Any):
    """O ic recompute."""
    if type(payload) is not dict:
        return None
    protocol = payload.get("protocol")
    if type(protocol) is not dict:
        return None
    seed = protocol.get("seed")
    backend_fingerprint = protocol.get("backend_fingerprint")
    if type(seed) is not int or isinstance(seed, bool) or type(backend_fingerprint) is not str:
        return None
    expected_protocol = _o_ic_protocol(seed, backend_fingerprint)
    if expected_protocol is None or protocol != expected_protocol:
        return None
    candidates = payload.get("candidates")
    if type(candidates) is not dict or not candidates:
        return None
    if len(candidates) > _O_IC_MAX_CANDIDATES:
        return None
    generated = _o_ic_generate(seed)
    if generated is None:
        return None
    total_records = 0
    records_bytes = 0
    aggregates: dict = {}
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
        if set(entry) != _O_IC_AGGREGATE_KEYS | {"status", "detail", "records"}:
            return None
        records = entry["records"]
        if type(records) is not list or len(records) != len(generated):
            return None
        total_records += len(records)
        records_bytes += len(
            json.dumps(records, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        )
        if records_bytes > _O_IC_MAX_RECORDS_BYTES:
            return None
        for record, case in zip(records, generated, strict=True):
            if type(record) is not dict or set(record) != {
                "fixture",
                "topic",
                "expected_digest",
                "content_digests",
                "arm_tokens",
                "id_digests",
                "ids_identical",
                "id_length_delta",
                "cells",
                "cell_details",
            }:
                return None
            if record["fixture"] != case["fixture_id"]:
                return None
            if record["topic"] != case["topic"]:
                return None
            expected_digest = _o_ic_sha(str(case["expected"]))
            if record["expected_digest"] != expected_digest:
                return None
            arm_tokens = record["arm_tokens"]
            id_digests = record["id_digests"]
            content_digests = record["content_digests"]
            ids_identical = record["ids_identical"]
            id_length_delta = record["id_length_delta"]
            cells = record["cells"]
            details = record["cell_details"]
            for mapping in (arm_tokens, id_digests, content_digests, cells, details):
                if type(mapping) is not dict or set(mapping) != set(_O_IC_ARM_KEYS):
                    return None
            for mapping in (ids_identical, id_length_delta):
                if type(mapping) is not dict or set(mapping) != set(_O_IC_TREATED_ARMS):
                    return None
            for arm in _O_IC_TREATED_ARMS:
                if ids_identical[arm] is not (id_digests[arm] == id_digests["control"]):
                    return None
                if id_length_delta[arm] != arm_tokens[arm] - arm_tokens["control"]:
                    return None
            for arm in _O_IC_ARM_KEYS:
                tokens = arm_tokens[arm]
                if type(tokens) is not int or isinstance(tokens, bool) or tokens < 0:
                    return None
                if not _o_ic_digest_ok(id_digests[arm]):
                    return None
                if content_digests[arm] != _o_ic_sha(_o_ic_arm_content(case, arm)):
                    return None
                cell = cells[arm]
                detail = details[arm]
                if detail not in ("", *_O_IC_DETAILS):
                    return None
                if cell is None:
                    if detail == "":
                        return None
                    continue
                if detail != "":
                    return None
                if not _o_ic_cell_ok(cell, expected_digest):
                    return None
                if cell["input_tokens"] != tokens:
                    return None
        aggregates[name] = _o_ic_aggregate(records, generated)
    if total_records > _O_IC_MAX_GROUND_RECORDS:
        return None
    if not aggregates:
        return None
    return {
        "aggregates": aggregates,
        "n_candidates": len(candidates),
        "n_cases": len(generated),
    }


def _o_invisiblechars_valid(value: Any) -> bool:
    """Oracle invisible-chars validator: mirrors
    diffreports._valid_invisible_chars.
    """
    if type(value) is not dict or value.get("available") is not True:
        return False
    if set(value) != {
        "available",
        "detail",
        "protocol",
        "audit",
        "candidates",
        "n_candidates",
        "n_cases",
    }:
        return False
    if value.get("detail") != "":
        return False
    audit = value.get("audit")
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
    recomputed = _o_ic_recompute(value)
    if recomputed is None:
        return False
    candidates = value["candidates"]
    for name, expected in recomputed["aggregates"].items():
        entry = candidates[name]
        for key, val in expected.items():
            if entry.get(key) != val:
                return False
    if value.get("n_candidates") != recomputed["n_candidates"]:
        return False
    return bool(value.get("n_cases") == recomputed["n_cases"])


def _o_invisiblechars(old: Any, new: Any) -> tuple[dict[str, Any], bool, bool]:
    """Oracle invisible-chars section: mirrors
    diffreports.diff_invisible_chars.
    """
    old_ok = _o_invisiblechars_valid(old)
    new_ok = _o_invisiblechars_valid(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    diff: dict[str, Any] = {"available": available, "changed": False}
    asymmetric = available not in ("both", "neither")
    if available != "both":
        return diff, False, asymmetric
    protocol_drift = old["protocol"].get("fingerprint") != new["protocol"].get("fingerprint")
    generator_drift = old["protocol"].get("ground_sha256") != new["protocol"].get(
        "ground_sha256"
    ) or old["protocol"].get("case_material_sha256") != new["protocol"].get("case_material_sha256")
    backend_drift = old["protocol"].get("backend_fingerprint") != new["protocol"].get(
        "backend_fingerprint"
    )
    comparable = not protocol_drift and not generator_drift and not backend_drift
    old_candidates = old["candidates"]
    new_candidates = new["candidates"]
    added = sorted(set(new_candidates) - set(old_candidates))
    removed = sorted(set(old_candidates) - set(new_candidates))
    candidate_drift: list = []
    for name in sorted(set(old_candidates) & set(new_candidates)):
        old_entry = old_candidates[name]
        new_entry = new_candidates[name]
        if old_entry.get("status") != "available" or new_entry.get("status") != "available":
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
        metrics: list = []
        for arm in _O_IC_ARM_KEYS:
            metrics.append(
                (f"{arm}_rate", old_entry["arms"][arm]["rate"], new_entry["arms"][arm]["rate"])
            )
        for arm in _O_IC_TREATED_ARMS:
            old_head = old_entry["headline"][arm]
            new_head = new_entry["headline"][arm]
            old_ids = old_entry["ids"][arm]
            new_ids = new_entry["ids"][arm]
            metrics.extend(
                [
                    (f"{arm}_label", old_head["label"], new_head["label"]),
                    (
                        f"{arm}_control_passes",
                        old_head["control_passes"],
                        new_head["control_passes"],
                    ),
                    (f"{arm}_wrong_values", old_head["wrong_values"], new_head["wrong_values"]),
                    (f"{arm}_n_reverse", old_head["n_reverse"], new_head["n_reverse"]),
                    (f"{arm}_n_population", old_head["n_population"], new_head["n_population"]),
                    (f"{arm}_n_incomplete", old_head["n_incomplete"], new_head["n_incomplete"]),
                    (f"{arm}_n_non_answer", old_head["n_non_answer"], new_head["n_non_answer"]),
                    (f"{arm}_control_rate", old_head["control_rate"], new_head["control_rate"]),
                    (f"{arm}_arm_rate", old_head["arm_rate"], new_head["arm_rate"]),
                    (
                        f"{arm}_pair_exact_only",
                        old_head["pair_decomposition"]["exact_only"],
                        new_head["pair_decomposition"]["exact_only"],
                    ),
                    (
                        f"{arm}_pair_boundary_only",
                        old_head["pair_decomposition"]["boundary_only"],
                        new_head["pair_decomposition"]["boundary_only"],
                    ),
                    (
                        f"{arm}_pair_mixed",
                        old_head["pair_decomposition"]["mixed"],
                        new_head["pair_decomposition"]["mixed"],
                    ),
                    (
                        f"{arm}_n_ids_identical",
                        old_ids["n_ids_identical"],
                        new_ids["n_ids_identical"],
                    ),
                    (
                        f"{arm}_max_abs_id_delta",
                        old_ids["max_abs_id_delta"],
                        new_ids["max_abs_id_delta"],
                    ),
                    (
                        f"{arm}_n_same_ids_verdict_flips",
                        old_ids["n_same_ids_verdict_flips"],
                        new_ids["n_same_ids_verdict_flips"],
                    ),
                ]
            )
        for metric, old_value, new_value in metrics:
            if old_value != new_value:
                candidate_drift.append(
                    {"candidate": name, "metric": metric, "old": old_value, "new": new_value}
                )
    diff["candidate_drift"] = candidate_drift
    diff["candidates_added"] = added
    diff["candidates_removed"] = removed
    diff["protocol_drift"] = protocol_drift
    diff["generator_drift"] = generator_drift
    diff["backend_drift"] = backend_drift
    diff["comparable"] = comparable
    changed = bool(
        candidate_drift or added or removed or protocol_drift or generator_drift or backend_drift
    )
    diff["changed"] = changed
    return diff, changed, asymmetric


# diffreports._valid_confusable_skeleton / diff_confusable_skeleton,
# re-implemented from docs/CONFUSABLESKELETON_DESIGN.md as
# amended (A1-A9). The population is frozen from the pinned
# Unicode 15.0.0 confusables data (the smallest-code-point
# rule); the rarity assignment follows the name-prefix
# script rule (amendment A2); every cell carries
# digest-bound answer_text; the headline is the guarded
# classification over the id-divergent paired population;
# the decomposition is the ordered decision list with the
# position-aware rarity form_only predicate (amendment A3).


_O_CS_TOPICS = (
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
    "pepper",
    "quartz",
    "birch",
)
_O_CS_WORDS = (
    "alder",
    "basalt",
    "cedar",
    "drift",
    "elm",
    "fern",
    "glade",
    "heath",
    "iris",
    "jade",
    "kelp",
    "loom",
    "moss",
    "nova",
    "ochre",
    "pine",
    "quill",
    "reed",
    "sage",
    "tide",
    "urn",
    "vale",
    "wren",
    "yew",
)
_O_CS_NEUTRAL_STATEMENTS = (
    "The weather changed slowly during the afternoon hours.",
    "A wooden fence runs along the edge of the property.",
    "She placed the documents inside a plain folder yesterday.",
    "The train arrived later than the schedule suggested.",
    "Fresh bread was cooling on the counter near the sink.",
    "He repaired the bicycle with tools from the garage.",
    "The bookshop extends its opening hours during summer.",
    "Several chairs were arranged around the long table.",
    "The recipe calls for a pinch of salt and herbs.",
    "They walked along the path beside the lake at dawn.",
    "The committee postponed its decision until next week.",
    "A thick layer of dust covered the old furniture.",
    "The children played quietly in the courtyard after lunch.",
    "His handwriting was difficult to read in places.",
    "The farmer harvested the fields before the rain came.",
    "An old map hung framed on the wall of the hallway.",
    "The accountant reviewed the figures a second time.",
    "Birds gathered on the rooftops as the sun went down.",
    "The mechanic inspected the engine and changed the oil.",
    "A narrow staircase led up to the attic.",
    "The orchestra rehearsed the piece several times over.",
    "Someone left a bicycle leaning against the railing.",
    "The soup simmered gently on the stove all morning.",
    "Visitors signed the guest book near the entrance.",
)
_O_CS_FROZEN_REPLACEMENTS = {
    "a": "\u03b1",
    "c": "\u0441",
    "d": "\u0501",
    "e": "\u0435",
    "h": "\u04bb",
    "i": "\u03b9",
    "j": "\u0458",
    "o": "\u03bf",
    "p": "\u03c1",
    "q": "\u051b",
    "r": "\u0433",
    "s": "\u0455",
    "u": "\u03c5",
    "v": "\u03bd",
    "w": "\u0461",
    "y": "\u03b3",
}
_O_CS_RARITY_CONTROLS = {
    "a": "\u03b6",
    "c": "\u0434",
    "d": "\u0434",
    "e": "\u0434",
    "h": "\u0434",
    "i": "\u03b6",
    "j": "\u0434",
    "o": "\u03b6",
    "p": "\u03b6",
    "q": "\u0434",
    "r": "\u0434",
    "s": "\u0434",
    "u": "\u03b6",
    "v": "\u03b6",
    "w": "\u0434",
    "y": "\u03b6",
}
_O_CS_CONTROL_CHARACTERS = ("\u0434", "\u03b6")
_O_CS_LETTER_SKELETONS = {
    "a": "a",
    "b": "b",
    "c": "c",
    "d": "d",
    "e": "e",
    "f": "f",
    "g": "g",
    "h": "h",
    "i": "i",
    "j": "j",
    "k": "k",
    "l": "l",
    "m": "rn",
    "n": "n",
    "o": "o",
    "p": "p",
    "q": "q",
    "r": "r",
    "s": "s",
    "t": "t",
    "u": "u",
    "v": "v",
    "w": "w",
    "y": "y",
    "z": "z",
}
_O_CS_POPULATION_COUNTS = {
    "a": (23, 2),
    "b": (18, 0),
    "c": (21, 1),
    "d": (19, 1),
    "e": (19, 2),
    "f": (18, 0),
    "g": (18, 0),
    "h": (17, 1),
    "i": (37, 4),
    "j": (17, 1),
    "k": (13, 0),
    "l": (73, 0),
    "m": (17, 0),
    "n": (15, 0),
    "o": (75, 3),
    "p": (29, 2),
    "q": (16, 1),
    "r": (19, 1),
    "s": (20, 1),
    "t": (13, 0),
    "u": (27, 1),
    "v": (29, 2),
    "w": (22, 2),
    "y": (30, 3),
    "z": (16, 0),
}
_O_CS_PANEL_LETTERS = tuple(sorted(_O_CS_POPULATION_COUNTS))
_O_CS_CONFUSABLES_VERSION = "15.0.0"
_O_CS_CONFUSABLES_SHA256 = "2b10130885c3370b101c52d7baedc452ab7f0e257b86c1e52ee657ecfc29ce64"
_O_CS_CONFUSABLES_DERIVED_SHA256 = (
    "d367290bc0867e6b484c68370530bdd1a08b6b32404601b8c7accaf83e05628d"
)
_O_CS_ARM_KEYS = ("control", "single_site", "whole_word", "rarity_control")
_O_CS_TREATED_ARMS = _O_CS_ARM_KEYS[1:]
_O_CS_CONFUSABLE_ARMS = ("single_site", "whole_word")
_O_CS_CASE_COUNT = 24
_O_CS_MIN_D = 10
_O_CS_MIN_J = 6
_O_CS_MAX_INCOMPLETE = 6
_O_CS_REDRAW_LIMIT = 1000
_O_CS_GENERATOR_SPEC = (
    "draw(seed, index, slot, counter) = int.from_bytes("
    "sha256(f'{seed}|confusableskeleton|{index}|{slot}|{counter}')"
    ".digest()[:8], 'big'); slot 0: word = WORDS[draw % len(WORDS)]; "
    "slot 1: number = 1000 + draw % 9000; a value colliding with an "
    "earlier case redraws both slots at counter+1, giving up after "
    "1000 consecutive redraws (the stage fails closed); slots 2/3: "
    "neutral indices = draw % len(NEUTRAL_STATEMENTS), the second "
    "advancing +1 on collision"
)
_O_CS_MAX_CANDIDATES = 24
_O_CS_MAX_GROUND_RECORDS = _O_CS_MAX_CANDIDATES * _O_CS_CASE_COUNT
_O_CS_GENERATION_BUDGET = len(_O_CS_ARM_KEYS) * _O_CS_MAX_GROUND_RECORDS
_O_CS_MAX_RECORDS_BYTES = 12 * 1024 * 1024
_O_CS_CELL_FIELDS = {
    "verdict",
    "boundary_found",
    "boundary_text_match",
    "input_tokens",
    "answer_digest",
    "answer_text",
}
_O_CS_DETAILS = ("arm_context_exceeded", "arm_evaluation_failed")
_O_CS_CELL_CLASSES = (
    "pass",
    "boundary_only",
    "wrong_value",
    "echo_confusable",
    "echo_control",
    "form_only",
    "non_answer",
)
_O_CS_PAIR_CLASSES = ("exact_only", "boundary_only", "mixed")
_O_CS_OC = {
    "fire_competent_independent": 0.0173,
    "fire_competent_paired": 0.0044,
    "fire_weak_independent": 0.1743,
    "fire_random_independent": 0.5503,
    "power_drop10_base90": 0.2542,
    "power_drop25_base90": 0.8159,
    "power_drop25_base85": 0.8761,
    "power_drop25_paired": 0.6981,
    "power_drop50_base90": 0.999,
    "sensitivity18_drop25": 0.5214,
    "sensitivity18_competent": 0.0038,
    "panel_sweep_16_drop25": 0.3904,
    "panel_sweep_20_drop25": 0.6395,
    "transplant_fire_competent_independent": 0.8897,
    "familywise_competent_expected_72": 1.2437,
    "familywise_competent_any_72": 0.7148,
    "familywise_paired_expected_72": 0.3199,
    "familywise_paired_any_72": 0.2743,
    "familywise_random_expected_72": 39.6221,
    "world_a_guarded_fire": 0.0055,
    "world_a_unguarded_fire": 0.0186,
    "world_b_unguarded_below_threshold": 0.7277,
    "world_b_guarded_below_threshold": 0.0117,
    "world_c_guarded_fire": 0.8893,
    "world_c_unguarded_fire": 0.9399,
}
_O_CS_AGGREGATE_KEYS = {
    "arms",
    "ids",
    "dose",
    "headline",
    "operating_characteristics",
    "n_cases",
    "n_incomplete",
}


def _o_cs_sha(text: str) -> str:
    """O ic sha."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _o_cs_digest_ok(value: Any) -> bool:
    """O ic digest ok."""
    if type(value) is not str or len(value) != 64:
        return False
    return all(ch in "0123456789abcdef" for ch in value)


def _o_cs_derive(seed: int, index: int, slot: int, counter: int) -> int:
    """O ic derive."""
    material = f"{seed}|confusableskeleton|{index}|{slot}|{counter}"
    return int.from_bytes(hashlib.sha256(material.encode("ascii")).digest()[:8], "big")


def _o_cs_generate(seed: int):
    """O ic generate."""
    cases: list = []
    seen: set = set()
    for index in range(_O_CS_CASE_COUNT):
        counter = 0
        while True:
            word = _O_CS_WORDS[_o_cs_derive(seed, index, 0, counter) % len(_O_CS_WORDS)]
            number = 1000 + _o_cs_derive(seed, index, 1, counter) % 9000
            value = f"{word}-{number}"
            if value not in seen:
                break
            counter += 1
            if counter > _O_CS_REDRAW_LIMIT:
                return None
        seen.add(value)
        first = _o_cs_derive(seed, index, 2, 0) % len(_O_CS_NEUTRAL_STATEMENTS)
        second = _o_cs_derive(seed, index, 3, 0) % len(_O_CS_NEUTRAL_STATEMENTS)
        if second == first:
            second = (first + 1) % len(_O_CS_NEUTRAL_STATEMENTS)
        cases.append(
            {
                "fixture_id": f"confusableskeleton-{index:02d}",
                "topic": _O_CS_TOPICS[index],
                "value": value,
                "expected": value,
                "neutral_indices": [first, second],
            }
        )
    return cases


def _o_cs_fact(case: dict) -> str:
    """O ic fact."""
    return f"The {case['topic']} inspection code is {case['value']}."


def _o_cs_question(case: dict) -> str:
    """O ic question."""
    return f"What is the {case['topic']} inspection code?"


def _o_cs_eligible_count(word: str) -> int:
    """O cs eligible count."""
    return sum(1 for char in word if char in _O_CS_FROZEN_REPLACEMENTS)


def _o_cs_treated_word(word: str, arm: str) -> str:
    """O cs treated word."""
    if arm == "control":
        return word
    if arm == "whole_word":
        return "".join(_O_CS_FROZEN_REPLACEMENTS.get(char, char) for char in word)
    table = _O_CS_FROZEN_REPLACEMENTS if arm == "single_site" else _O_CS_RARITY_CONTROLS
    for index, char in enumerate(word):
        if char in table:
            return word[:index] + table[char] + word[index + 1 :]
    return word


def _o_cs_treated_fact(case: dict, arm: str) -> str:
    """O cs treated fact."""
    word, _, number = str(case["value"]).partition("-")
    treated_topic = _o_cs_treated_word(str(case["topic"]), arm)
    treated_value = _o_cs_treated_word(word, arm) + "-" + number
    return f"The {treated_topic} inspection code is {treated_value}."


def _o_cs_arm_content(case: dict, arm: str) -> str:
    """O cs arm content."""
    first, second = (int(i) for i in case["neutral_indices"])
    fact = _o_cs_fact(case) if arm == "control" else _o_cs_treated_fact(case, arm)
    return (
        f"{_O_CS_NEUTRAL_STATEMENTS[first]} {fact} "
        f"{_O_CS_NEUTRAL_STATEMENTS[second]} {_o_cs_question(case)}"
    )


def _o_cs_skeleton_form(word: str) -> str:
    """O cs skeleton form."""
    out: list = []
    inverse = {replacement: letter for letter, replacement in _O_CS_FROZEN_REPLACEMENTS.items()}
    for char in word:
        if char in _O_CS_LETTER_SKELETONS:
            out.append(_O_CS_LETTER_SKELETONS[char])
        elif char in inverse:
            out.append(_O_CS_LETTER_SKELETONS[inverse[char]])
        else:
            out.append(char)
    return "".join(out)


def _o_cs_script_of(char: str) -> str:
    """O cs script of."""
    name = unicodedata.name(char, "")
    if name.startswith("CYRILLIC"):
        return "CYRILLIC"
    if name.startswith("GREEK"):
        return "GREEK"
    return ""


def _o_cs_population_audit() -> dict:
    """O cs population audit."""
    letters: dict = {}
    ok = True
    for letter in _O_CS_PANEL_LETTERS:
        replacement = _O_CS_FROZEN_REPLACEMENTS.get(letter)
        control = _O_CS_RARITY_CONTROLS.get(letter)
        skeleton_equal, eligible = _O_CS_POPULATION_COUNTS[letter]
        entry: dict = {
            "skeleton_equal_sources": skeleton_equal,
            "eligible_sources": eligible,
            "replacement": None,
            "replacement_name": "",
            "script": "",
            "control": None,
            "control_name": "",
            "satisfies": replacement is None,
        }
        if replacement is not None and control is not None:
            name = unicodedata.name(replacement, "")
            script = _o_cs_script_of(replacement)
            control_name = unicodedata.name(control, "")
            satisfies = (
                name.startswith(("CYRILLIC SMALL LETTER", "GREEK SMALL LETTER"))
                and unicodedata.category(replacement) == "Ll"
                and unicodedata.normalize("NFC", replacement)
                == unicodedata.normalize("NFD", replacement)
                == replacement
                and unicodedata.normalize("NFKC", replacement) == replacement
                and unicodedata.decomposition(replacement) == ""
                and len(replacement.encode("utf-8")) == 2
                and _o_cs_script_of(control) == script
                and control in _O_CS_CONTROL_CHARACTERS
            )
            entry.update(
                {
                    "replacement": f"U+{ord(replacement):04X}",
                    "replacement_name": name,
                    "script": script,
                    "control": f"U+{ord(control):04X}",
                    "control_name": control_name,
                    "satisfies": satisfies,
                }
            )
            if not satisfies:
                ok = False
        letters[letter] = entry
    controls: dict = {}
    for control in _O_CS_CONTROL_CHARACTERS:
        name = unicodedata.name(control, "")
        distinct = (
            control not in _O_CS_FROZEN_REPLACEMENTS.values()
            and control not in _O_CS_LETTER_SKELETONS
            and _o_cs_skeleton_form(control) == control
        )
        satisfies = (
            bool(name)
            and unicodedata.category(control) == "Ll"
            and unicodedata.normalize("NFC", control)
            == unicodedata.normalize("NFD", control)
            == control
            and unicodedata.normalize("NFKC", control) == control
            and unicodedata.decomposition(control) == ""
            and len(control.encode("utf-8")) == 2
            and distinct
        )
        controls[f"U+{ord(control):04X}"] = {
            "name": name,
            "script": _o_cs_script_of(control),
            "skeleton_distinct": distinct,
            "satisfies": satisfies,
        }
        if not satisfies:
            ok = False
    words = (*_O_CS_TOPICS, *_O_CS_WORDS)
    words_ok = all(_o_cs_eligible_count(word) >= 1 for word in words)
    invariance = all(
        _o_cs_skeleton_form(_o_cs_treated_word(word, arm)) == _o_cs_skeleton_form(word)
        for word in words
        for arm in _O_CS_CONFUSABLE_ARMS
    )
    rarity_differs = all(
        _o_cs_skeleton_form(_o_cs_treated_word(word, "rarity_control")) != _o_cs_skeleton_form(word)
        for word in words
    )
    length_ok = all(
        len(_o_cs_treated_word(word, arm)) == len(word)
        for word in words
        for arm in _O_CS_ARM_KEYS[1:]
    )
    ok = ok and words_ok and invariance and rarity_differs and length_ok
    return {
        "letters": letters,
        "controls": controls,
        "every_word_eligible": words_ok,
        "skeleton_invariance": invariance,
        "rarity_skeleton_differs": rarity_differs,
        "length_preserved": length_ok,
        "ok": ok,
        "unicode_version": unicodedata.unidata_version,
        "confusables_version": _O_CS_CONFUSABLES_VERSION,
    }


def _o_cs_table_sha256() -> str:
    """O cs table sha256."""
    material = {
        "frozen_replacements": _O_CS_FROZEN_REPLACEMENTS,
        "rarity_controls": _O_CS_RARITY_CONTROLS,
        "control_characters": list(_O_CS_CONTROL_CHARACTERS),
        "letter_skeletons": _O_CS_LETTER_SKELETONS,
        "population_counts": {
            letter: list(counts) for letter, counts in _O_CS_POPULATION_COUNTS.items()
        },
        "confusables_version": _O_CS_CONFUSABLES_VERSION,
        "confusables_sha256": _O_CS_CONFUSABLES_SHA256,
        "confusables_derived_sha256": _O_CS_CONFUSABLES_DERIVED_SHA256,
    }
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return _o_cs_sha(canonical)


def _o_cs_case_material_sha256() -> str:
    """O cs case material sha256."""
    material = {
        "topics": list(_O_CS_TOPICS),
        "words": list(_O_CS_WORDS),
        "neutral_statements": list(_O_CS_NEUTRAL_STATEMENTS),
        "arm_keys": list(_O_CS_ARM_KEYS),
        "generator_spec": _O_CS_GENERATOR_SPEC,
        "case_count": _O_CS_CASE_COUNT,
        "table_sha256": _o_cs_table_sha256(),
        "min_d": _O_CS_MIN_D,
        "min_j": _O_CS_MIN_J,
        "max_incomplete": _O_CS_MAX_INCOMPLETE,
        "redraw_limit": _O_CS_REDRAW_LIMIT,
        "population_audit": _o_cs_population_audit(),
    }
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return _o_cs_sha(canonical)


def _o_cs_ground_sha256(seed: int):
    """O ic ground sha256."""
    generated = _o_cs_generate(seed)
    if generated is None:
        return None
    rows = [
        [case["fixture_id"], case["topic"], case["value"], list(case["neutral_indices"])]
        for case in generated
    ]
    canonical = json.dumps(rows, separators=(",", ":"), ensure_ascii=True)
    return _o_cs_sha(canonical)


def _o_cs_protocol(seed: int, backend_fingerprint: str):
    """O ic protocol."""
    ground = _o_cs_ground_sha256(seed)
    if ground is None:
        return None
    protocol: dict = {
        "version": 1,
        "arm_keys": list(_O_CS_ARM_KEYS),
        "treated_arms": list(_O_CS_TREATED_ARMS),
        "table_sha256": _o_cs_table_sha256(),
        "confusables_version": _O_CS_CONFUSABLES_VERSION,
        "case_count": _O_CS_CASE_COUNT,
        "min_d": _O_CS_MIN_D,
        "min_j": _O_CS_MIN_J,
        "max_incomplete": _O_CS_MAX_INCOMPLETE,
        "case_material_sha256": _o_cs_case_material_sha256(),
        "ground_sha256": ground,
        "population_audit": _o_cs_population_audit(),
        "generation_budget": _O_CS_GENERATION_BUDGET,
        "max_ground_records": _O_CS_MAX_GROUND_RECORDS,
        "max_records_bytes": _O_CS_MAX_RECORDS_BYTES,
        "max_candidates": _O_CS_MAX_CANDIDATES,
        "seed": seed,
        "backend_fingerprint": backend_fingerprint,
    }
    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    protocol["fingerprint"] = _o_cs_sha(canonical)
    return protocol


def _o_cs_cell(record: dict, key: str):
    """O ic cell."""
    cells = record.get("cells")
    if not isinstance(cells, dict):
        return None
    cell = cells.get(key)
    return cell if isinstance(cell, dict) else None


def _o_cs_detail(record: dict, key: str) -> str:
    """O ic detail."""
    details = record.get("cell_details")
    if not isinstance(details, dict):
        return ""
    detail = details.get(key, "")
    return detail if type(detail) is str else ""


def _o_cs_verdict(record: dict, key: str) -> bool:
    """O ic verdict."""
    cell = _o_cs_cell(record, key)
    return bool(cell is not None and cell["verdict"])


def _o_cs_ids_identical(record: dict, arm: str) -> bool:
    """O ic ids identical."""
    digests = record.get("id_digests")
    if not isinstance(digests, dict):
        return False
    control = digests.get("control")
    return type(control) is str and control != "" and control == digests.get(arm)


def _o_cs_decompose_cell(cell: dict, expected: str, expected_digest: str, arm: str) -> str:
    """O cs decompose cell."""
    if cell.get("verdict") is True:
        return "pass"
    digest = cell.get("answer_digest")
    if digest is not None and digest == expected_digest:
        return "boundary_only"
    text = cell.get("answer_text")
    if type(text) is not str:
        return "non_answer"
    if text != expected and re.match(r"^[a-z]+-[0-9]{4}$", text):
        word, _, number = text.partition("-")
        if word in _O_CS_WORDS and 1000 <= int(number) <= 9999:
            return "wrong_value"
    if arm in _O_CS_CONFUSABLE_ARMS:
        word, _, number = expected.partition("-")
        if text == _o_cs_treated_word(word, arm) + "-" + number:
            return "echo_confusable"
        if text != expected:
            inverse = {
                replacement: letter for letter, replacement in _O_CS_FROZEN_REPLACEMENTS.items()
            }
            if "".join(inverse.get(char, char) for char in text) == expected:
                return "form_only"
    elif arm == "rarity_control":
        word, _, number = expected.partition("-")
        if text == _o_cs_treated_word(word, arm) + "-" + number:
            return "echo_control"
        if (
            text != expected
            and len(text) == len(expected)
            and any(char in _O_CS_CONTROL_CHARACTERS for char in text)
            and all(
                char == expected_char or char in _O_CS_CONTROL_CHARACTERS
                for char, expected_char in zip(text, expected, strict=True)
            )
        ):
            return "form_only"
    return "non_answer"


def _o_cs_decompose_pair(control_cell: dict, arm_cell: dict) -> str:
    """O ic decompose pair."""
    if control_cell.get("answer_digest") == arm_cell.get("answer_digest"):
        return "boundary_only"
    if control_cell.get("boundary_found") and arm_cell.get("boundary_found"):
        return "exact_only"
    return "mixed"


def _o_cs_classify(control_passes: int, wrong_values: int, n_incomplete: int) -> str:
    """O ic classify."""
    if n_incomplete > _O_CS_MAX_INCOMPLETE:
        return "insufficient"
    if control_passes < _O_CS_MIN_D:
        return "insufficient"
    if wrong_values >= _O_CS_MIN_J:
        return "discordant_asymmetric"
    if wrong_values == 0:
        return "no_discordance_observed" if n_incomplete == 0 else "insufficient"
    return "below_threshold" if wrong_values + n_incomplete < _O_CS_MIN_J else "insufficient"


def _o_cs_rate(numerator: int, denominator: int):
    """O ic rate."""
    return (numerator / denominator) if denominator else None


def _o_cs_aggregate(records: list, cases: list) -> dict:
    """O ic aggregate."""
    expected_by_fixture = {str(case["fixture_id"]): str(case["expected"]) for case in cases}
    arms: dict = {}
    for key in _O_CS_ARM_KEYS:
        n_completed = 0
        n_pass = 0
        n_context = 0
        n_failed = 0
        decomposition = dict.fromkeys(_O_CS_CELL_CLASSES, 0)
        for record in records:
            detail = _o_cs_detail(record, key)
            if detail == "arm_context_exceeded":
                n_context += 1
            elif detail == "arm_evaluation_failed":
                n_failed += 1
            cell = _o_cs_cell(record, key)
            if cell is None:
                continue
            n_completed += 1
            if cell["verdict"]:
                n_pass += 1
            decomposition[
                _o_cs_decompose_cell(
                    cell,
                    expected_by_fixture.get(str(record.get("fixture", "")), ""),
                    str(record.get("expected_digest", "")),
                    key,
                )
            ] += 1
        arms[key] = {
            "n_completed": n_completed,
            "n_pass": n_pass,
            "rate": _o_cs_rate(n_pass, n_completed),
            "n_context_exceeded": n_context,
            "n_evaluation_failed": n_failed,
            "decomposition": decomposition,
        }
    ids: dict = {}
    for arm in _O_CS_TREATED_ARMS:
        deltas: list = []
        n_ids_identical = 0
        n_same_ids_flips = 0
        for record in records:
            tokens = record.get("arm_tokens")
            tokens = tokens if isinstance(tokens, dict) else {}
            treated = tokens.get(arm)
            baseline = tokens.get("control")
            if type(treated) is int and type(baseline) is int:
                deltas.append(treated - baseline)
            else:
                deltas.append(0)
            if _o_cs_ids_identical(record, arm):
                n_ids_identical += 1
                control_cell = _o_cs_cell(record, "control")
                arm_cell = _o_cs_cell(record, arm)
                if (
                    control_cell is not None
                    and arm_cell is not None
                    and control_cell["verdict"] != arm_cell["verdict"]
                ):
                    n_same_ids_flips += 1
        ids[arm] = {
            "n_ids_identical": n_ids_identical,
            "n_ids_divergent": len(records) - n_ids_identical,
            "id_length_delta": deltas,
            "max_abs_id_delta": max((abs(delta) for delta in deltas), default=0),
            "n_same_ids_verdict_flips": n_same_ids_flips,
        }
    dose_eligible = 0
    dose_letters = 0
    for case in cases:
        word, _, _number = str(case["value"]).partition("-")
        for treated in (str(case["topic"]), word):
            dose_eligible += _o_cs_eligible_count(treated)
            dose_letters += len(treated)
    dose = {
        "whole_word": {
            "eligible_letters": dose_eligible,
            "total_letters": dose_letters,
            "fraction": _o_cs_rate(dose_eligible, dose_letters),
        }
    }
    headline: dict = {}
    for arm in _O_CS_TREATED_ARMS:
        domain = [record for record in records if not _o_cs_ids_identical(record, arm)]
        paired = [
            record
            for record in domain
            if _o_cs_cell(record, "control") is not None and _o_cs_cell(record, arm) is not None
        ]
        n_incomplete = len(domain) - len(paired)
        control_passes = 0
        wrong_values = 0
        reverse = 0
        non_answer = 0
        pair_decomposition = dict.fromkeys(_O_CS_PAIR_CLASSES, 0)
        failing_classes = dict.fromkeys(_O_CS_CELL_CLASSES, 0)
        for record in paired:
            control_cell = _o_cs_cell(record, "control")
            arm_cell = _o_cs_cell(record, arm)
            if control_cell is None or arm_cell is None:
                continue
            expected = expected_by_fixture.get(str(record.get("fixture", "")), "")
            expected_digest = str(record.get("expected_digest", ""))
            arm_class = _o_cs_decompose_cell(arm_cell, expected, expected_digest, arm)
            if arm_class == "non_answer":
                non_answer += 1
            if control_cell["verdict"]:
                control_passes += 1
                if arm_class == "wrong_value":
                    wrong_values += 1
            elif arm_cell["verdict"]:
                reverse += 1
            if control_cell["verdict"] != arm_cell["verdict"]:
                pair_decomposition[_o_cs_decompose_pair(control_cell, arm_cell)] += 1
                failing = arm_cell if control_cell["verdict"] else control_cell
                failing_arm = arm if control_cell["verdict"] else "control"
                failing_classes[
                    _o_cs_decompose_cell(failing, expected, expected_digest, failing_arm)
                ] += 1
        headline[arm] = {
            "label": _o_cs_classify(control_passes, wrong_values, n_incomplete),
            "n_population": len(paired),
            "n_incomplete": n_incomplete,
            "control_passes": control_passes,
            "wrong_values": wrong_values,
            "n_reverse": reverse,
            "control_rate": _o_cs_rate(control_passes, len(paired)),
            "arm_rate": _o_cs_rate(
                sum(1 for record in paired if _o_cs_verdict(record, arm)), len(paired)
            ),
            "n_non_answer": non_answer,
            "pair_decomposition": pair_decomposition,
            "failing_classes": failing_classes,
        }
    return {
        "arms": arms,
        "ids": ids,
        "dose": dose,
        "headline": headline,
        "operating_characteristics": dict(_O_CS_OC),
        "n_cases": len(records),
        "n_incomplete": sum(
            1
            for record in records
            if any(_o_cs_cell(record, arm) is None for arm in _O_CS_ARM_KEYS)
        ),
    }


def _o_cs_cell_ok(cell: Any, expected_digest: str) -> bool:
    """O ic cell ok."""
    if type(cell) is not dict or set(cell) != _O_CS_CELL_FIELDS:
        return False
    if type(cell["verdict"]) is not bool or type(cell["boundary_found"]) is not bool:
        return False
    if cell["boundary_text_match"] is not None and type(cell["boundary_text_match"]) is not bool:
        return False
    if type(cell["input_tokens"]) is not int or cell["input_tokens"] < 0:
        return False
    digest = cell["answer_digest"]
    if digest is not None and not _o_cs_digest_ok(digest):
        return False
    text = cell["answer_text"]
    if text is None:
        if digest is not None:
            return False
    elif type(text) is not str or _o_cs_sha(text) != digest:
        return False
    verdict = (digest == expected_digest) and cell["boundary_found"]
    return bool(cell["verdict"] == verdict)


def _o_cs_recompute(payload: Any):
    """O ic recompute."""
    if type(payload) is not dict:
        return None
    protocol = payload.get("protocol")
    if type(protocol) is not dict:
        return None
    seed = protocol.get("seed")
    backend_fingerprint = protocol.get("backend_fingerprint")
    if type(seed) is not int or isinstance(seed, bool) or type(backend_fingerprint) is not str:
        return None
    expected_protocol = _o_cs_protocol(seed, backend_fingerprint)
    if expected_protocol is None or protocol != expected_protocol:
        return None
    candidates = payload.get("candidates")
    if type(candidates) is not dict or not candidates:
        return None
    if len(candidates) > _O_CS_MAX_CANDIDATES:
        return None
    generated = _o_cs_generate(seed)
    if generated is None:
        return None
    total_records = 0
    records_bytes = 0
    aggregates: dict = {}
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
        if set(entry) != _O_CS_AGGREGATE_KEYS | {"status", "detail", "records"}:
            return None
        records = entry["records"]
        if type(records) is not list or len(records) != len(generated):
            return None
        total_records += len(records)
        records_bytes += len(
            json.dumps(records, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        )
        if records_bytes > _O_CS_MAX_RECORDS_BYTES:
            return None
        for record, case in zip(records, generated, strict=True):
            if type(record) is not dict or set(record) != {
                "fixture",
                "topic",
                "expected_digest",
                "content_digests",
                "arm_tokens",
                "id_digests",
                "ids_identical",
                "id_length_delta",
                "table_pin",
                "cells",
                "cell_details",
            }:
                return None
            if record["fixture"] != case["fixture_id"]:
                return None
            if record["topic"] != case["topic"]:
                return None
            expected_digest = _o_cs_sha(str(case["expected"]))
            if record["expected_digest"] != expected_digest:
                return None
            table_pin = record["table_pin"]
            if type(table_pin) is not dict or set(table_pin) != {"version", "table_sha256"}:
                return None
            if table_pin["version"] != _O_CS_CONFUSABLES_VERSION:
                return None
            if table_pin["table_sha256"] != _o_cs_table_sha256():
                return None
            arm_tokens = record["arm_tokens"]
            id_digests = record["id_digests"]
            content_digests = record["content_digests"]
            ids_identical = record["ids_identical"]
            id_length_delta = record["id_length_delta"]
            cells = record["cells"]
            details = record["cell_details"]
            for mapping in (arm_tokens, id_digests, content_digests, cells, details):
                if type(mapping) is not dict or set(mapping) != set(_O_CS_ARM_KEYS):
                    return None
            for mapping in (ids_identical, id_length_delta):
                if type(mapping) is not dict or set(mapping) != set(_O_CS_TREATED_ARMS):
                    return None
            for arm in _O_CS_TREATED_ARMS:
                if ids_identical[arm] is not (id_digests[arm] == id_digests["control"]):
                    return None
                if id_length_delta[arm] != arm_tokens[arm] - arm_tokens["control"]:
                    return None
            for arm in _O_CS_ARM_KEYS:
                tokens = arm_tokens[arm]
                if type(tokens) is not int or isinstance(tokens, bool) or tokens < 0:
                    return None
                if not _o_cs_digest_ok(id_digests[arm]):
                    return None
                if content_digests[arm] != _o_cs_sha(_o_cs_arm_content(case, arm)):
                    return None
                cell = cells[arm]
                detail = details[arm]
                if detail not in ("", *_O_CS_DETAILS):
                    return None
                if cell is None:
                    if detail == "":
                        return None
                    continue
                if detail != "":
                    return None
                if not _o_cs_cell_ok(cell, expected_digest):
                    return None
                if cell["input_tokens"] != tokens:
                    return None
        aggregates[name] = _o_cs_aggregate(records, generated)
    if total_records > _O_CS_MAX_GROUND_RECORDS:
        return None
    if not aggregates:
        return None
    return {
        "aggregates": aggregates,
        "n_candidates": len(candidates),
        "n_cases": len(generated),
    }


def _o_confusableskeleton_valid(value: Any) -> bool:
    """Oracle invisible-chars validator: mirrors
    diffreports._valid_confusable_skeleton.
    """
    if type(value) is not dict or value.get("available") is not True:
        return False
    if set(value) != {
        "available",
        "detail",
        "protocol",
        "audit",
        "candidates",
        "n_candidates",
        "n_cases",
    }:
        return False
    if value.get("detail") != "":
        return False
    audit = value.get("audit")
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
    recomputed = _o_cs_recompute(value)
    if recomputed is None:
        return False
    candidates = value["candidates"]
    for name, expected in recomputed["aggregates"].items():
        entry = candidates[name]
        for key, val in expected.items():
            if entry.get(key) != val:
                return False
    if value.get("n_candidates") != recomputed["n_candidates"]:
        return False
    return bool(value.get("n_cases") == recomputed["n_cases"])


def _o_confusableskeleton(old: Any, new: Any) -> tuple[dict[str, Any], bool, bool]:
    """Oracle invisible-chars section: mirrors
    diffreports.diff_confusable_skeleton.
    """
    old_ok = _o_confusableskeleton_valid(old)
    new_ok = _o_confusableskeleton_valid(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    diff: dict[str, Any] = {"available": available, "changed": False}
    asymmetric = available not in ("both", "neither")
    if available != "both":
        return diff, False, asymmetric
    protocol_drift = old["protocol"].get("fingerprint") != new["protocol"].get("fingerprint")
    generator_drift = old["protocol"].get("ground_sha256") != new["protocol"].get(
        "ground_sha256"
    ) or old["protocol"].get("case_material_sha256") != new["protocol"].get("case_material_sha256")
    backend_drift = old["protocol"].get("backend_fingerprint") != new["protocol"].get(
        "backend_fingerprint"
    )
    comparable = not protocol_drift and not generator_drift and not backend_drift
    old_candidates = old["candidates"]
    new_candidates = new["candidates"]
    added = sorted(set(new_candidates) - set(old_candidates))
    removed = sorted(set(old_candidates) - set(new_candidates))
    candidate_drift: list = []
    for name in sorted(set(old_candidates) & set(new_candidates)):
        old_entry = old_candidates[name]
        new_entry = new_candidates[name]
        if old_entry.get("status") != "available" or new_entry.get("status") != "available":
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
        metrics: list = []
        for arm in _O_CS_ARM_KEYS:
            metrics.append(
                (f"{arm}_rate", old_entry["arms"][arm]["rate"], new_entry["arms"][arm]["rate"])
            )
        metrics.extend(
            [
                (
                    "whole_word_eligible_letters",
                    old_entry["dose"]["whole_word"]["eligible_letters"],
                    new_entry["dose"]["whole_word"]["eligible_letters"],
                ),
                (
                    "whole_word_total_letters",
                    old_entry["dose"]["whole_word"]["total_letters"],
                    new_entry["dose"]["whole_word"]["total_letters"],
                ),
                (
                    "whole_word_fraction",
                    old_entry["dose"]["whole_word"]["fraction"],
                    new_entry["dose"]["whole_word"]["fraction"],
                ),
            ]
        )
        for arm in _O_CS_TREATED_ARMS:
            old_head = old_entry["headline"][arm]
            new_head = new_entry["headline"][arm]
            old_ids = old_entry["ids"][arm]
            new_ids = new_entry["ids"][arm]
            metrics.extend(
                [
                    (f"{arm}_label", old_head["label"], new_head["label"]),
                    (
                        f"{arm}_control_passes",
                        old_head["control_passes"],
                        new_head["control_passes"],
                    ),
                    (f"{arm}_wrong_values", old_head["wrong_values"], new_head["wrong_values"]),
                    (f"{arm}_n_reverse", old_head["n_reverse"], new_head["n_reverse"]),
                    (f"{arm}_n_population", old_head["n_population"], new_head["n_population"]),
                    (f"{arm}_n_incomplete", old_head["n_incomplete"], new_head["n_incomplete"]),
                    (f"{arm}_n_non_answer", old_head["n_non_answer"], new_head["n_non_answer"]),
                    (f"{arm}_control_rate", old_head["control_rate"], new_head["control_rate"]),
                    (f"{arm}_arm_rate", old_head["arm_rate"], new_head["arm_rate"]),
                    (
                        f"{arm}_pair_exact_only",
                        old_head["pair_decomposition"]["exact_only"],
                        new_head["pair_decomposition"]["exact_only"],
                    ),
                    (
                        f"{arm}_pair_boundary_only",
                        old_head["pair_decomposition"]["boundary_only"],
                        new_head["pair_decomposition"]["boundary_only"],
                    ),
                    (
                        f"{arm}_pair_mixed",
                        old_head["pair_decomposition"]["mixed"],
                        new_head["pair_decomposition"]["mixed"],
                    ),
                    (
                        f"{arm}_n_ids_identical",
                        old_ids["n_ids_identical"],
                        new_ids["n_ids_identical"],
                    ),
                    (
                        f"{arm}_max_abs_id_delta",
                        old_ids["max_abs_id_delta"],
                        new_ids["max_abs_id_delta"],
                    ),
                    (
                        f"{arm}_n_same_ids_verdict_flips",
                        old_ids["n_same_ids_verdict_flips"],
                        new_ids["n_same_ids_verdict_flips"],
                    ),
                ]
            )
        for metric, old_value, new_value in metrics:
            if old_value != new_value:
                candidate_drift.append(
                    {"candidate": name, "metric": metric, "old": old_value, "new": new_value}
                )
    diff["candidate_drift"] = candidate_drift
    diff["candidates_added"] = added
    diff["candidates_removed"] = removed
    diff["protocol_drift"] = protocol_drift
    diff["generator_drift"] = generator_drift
    diff["backend_drift"] = backend_drift
    diff["comparable"] = comparable
    changed = bool(
        candidate_drift or added or removed or protocol_drift or generator_drift or backend_drift
    )
    diff["changed"] = changed
    return diff, changed, asymmetric


def _o_instructiondistance_valid(value: Any) -> bool:
    """Oracle instruction-distance validator: mirrors
    diffreports._valid_instruction_distance.
    """
    if type(value) is not dict or value.get("available") is not True:
        return False
    if set(value) != {
        "available",
        "detail",
        "protocol",
        "audit",
        "candidates",
        "n_candidates",
        "n_cases",
    }:
        return False
    if value.get("detail") != "":
        return False
    audit = value.get("audit")
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
    recomputed = _o_id_recompute(value)
    if recomputed is None:
        return False
    candidates = value["candidates"]
    for name, expected in recomputed["aggregates"].items():
        entry = candidates[name]
        for key, val in expected.items():
            if entry.get(key) != val:
                return False
    if value.get("n_candidates") != recomputed["n_candidates"]:
        return False
    return bool(value.get("n_cases") == recomputed["n_cases"])


def _o_instructiondistance(old: Any, new: Any) -> tuple[dict[str, Any], bool, bool]:
    """Oracle instruction-distance section: mirrors
    diffreports.diff_instruction_distance.
    """
    old_ok = _o_instructiondistance_valid(old)
    new_ok = _o_instructiondistance_valid(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    diff: dict[str, Any] = {"available": available, "changed": False}
    asymmetric = available not in ("both", "neither")
    if available != "both":
        return diff, False, asymmetric
    protocol_drift = old["protocol"].get("fingerprint") != new["protocol"].get("fingerprint")
    generator_drift = old["protocol"].get("ground_sha256") != new["protocol"].get("ground_sha256")
    backend_drift = old["protocol"].get("backend_fingerprint") != new["protocol"].get(
        "backend_fingerprint"
    )
    comparable = not protocol_drift and not generator_drift and not backend_drift
    old_candidates = old["candidates"]
    new_candidates = new["candidates"]
    added = sorted(set(new_candidates) - set(old_candidates))
    removed = sorted(set(old_candidates) - set(new_candidates))
    candidate_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_candidates) & set(new_candidates)):
        old_entry = old_candidates[name]
        new_entry = new_candidates[name]
        if old_entry.get("status") != "available" or new_entry.get("status") != "available":
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
        metrics: list[tuple[str, Any, Any]] = []
        for dose_key in ("256", "1024"):
            old_dose = old_entry["doses"][dose_key]
            new_dose = new_entry["doses"][dose_key]
            metrics.extend(
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
        metrics.append(
            (
                "n_same_ids_compliance_flips_sep_1024",
                old_entry["ids"]["sep_1024"]["n_same_ids_compliance_flips"],
                new_entry["ids"]["sep_1024"]["n_same_ids_compliance_flips"],
            )
        )
        for metric, old_value, new_value in metrics:
            if old_value != new_value:
                candidate_drift.append(
                    {"candidate": name, "metric": metric, "old": old_value, "new": new_value}
                )
    diff["candidate_drift"] = candidate_drift
    diff["candidates_added"] = added
    diff["candidates_removed"] = removed
    diff["protocol_drift"] = protocol_drift
    diff["generator_drift"] = generator_drift
    diff["backend_drift"] = backend_drift
    diff["comparable"] = comparable
    changed = bool(
        candidate_drift or added or removed or protocol_drift or generator_drift or backend_drift
    )
    diff["changed"] = changed
    return diff, changed, asymmetric


def _o_rotation_digest_ok(item: Any) -> bool:
    """O rotation digest ok."""
    if item is None:
        return True
    if type(item) is not str or len(item) != 64:
        return False
    return all(ch in "0123456789abcdef" for ch in item)


def _o_rotation_cell_ok(cell: Any, expected_digest: str) -> bool:
    """O rotation cell ok."""
    if type(cell) is not dict or set(cell) != {
        "verdict",
        "boundary_found",
        "boundary_text_match",
        "input_tokens",
        "answer_digest",
    }:
        return False
    if type(cell["verdict"]) is not bool or type(cell["boundary_found"]) is not bool:
        return False
    if cell["boundary_text_match"] is not None and type(cell["boundary_text_match"]) is not bool:
        return False
    if type(cell["input_tokens"]) is not int or cell["input_tokens"] < 0:
        return False
    if not _o_rotation_digest_ok(cell["answer_digest"]):
        return False
    verdict = (cell["answer_digest"] == expected_digest) and cell["boundary_found"]
    return bool(cell["verdict"] == verdict)


def _o_rotation_recompute(value: Any) -> dict[str, Any] | None:
    """Oracle rotation recompute: mirrors
    diffreports._rotation_recompute. The oracle rebuilds the
    protocol and the case set through the producer module
    (rotation) where the implementation rebuilds them from
    literal tables -- a drift in either side's constants breaks
    the agreement the fuzz harness asserts.
    """
    import hashlib as _hashlib
    import json as _json

    protocol = value.get("protocol")
    if type(protocol) is not dict:
        return None
    seed = protocol.get("seed")
    backend_fp = protocol.get("backend_fingerprint")
    if type(seed) is not int or isinstance(seed, bool) or type(backend_fp) is not str:
        return None
    expected_protocol = dict(rotation.protocol_object(seed, backend_fp))
    expected_protocol["fingerprint"] = rotation.protocol_fingerprint(seed, backend_fp)
    if protocol != expected_protocol:
        return None
    candidates = value.get("candidates")
    if type(candidates) is not dict or not candidates:
        return None
    generated = rotation.generate_cases(seed)
    total = 0
    records_bytes = 0
    aggregates: dict[str, Any] = {}
    raw: dict[str, list[dict[str, Any]]] = {}
    for name, entry in candidates.items():
        if type(entry) is not dict:
            return None
        if entry.get("status") == "unavailable":
            if set(entry) != {"status", "detail", "records"} or entry.get("records") != []:
                return None
            continue
        if entry.get("status") != "available" or entry.get("detail") != "":
            return None
        if set(entry) != {
            "arms",
            "closed_book",
            "absent",
            "headline",
            "slot_curve",
            "mirror",
            "reversal",
            "classification",
            "invariance_falsified",
            "n_fixtures",
            "n_screened",
            "n_boundary_screened",
            "n_closed_book_incomplete",
            "n_live",
            "status",
            "detail",
            "records",
        }:
            return None
        records = entry["records"]
        if type(records) is not list or len(records) != len(generated):
            return None
        total += len(records)
        records_bytes += len(
            _json.dumps(records, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        )
        if records_bytes > rotation.MAX_RECORDS_BYTES:
            return None
        for record, case in zip(records, generated, strict=True):
            if type(record) is not dict or set(record) != {
                "fixture",
                "topic",
                "expected_digest",
                "arm_tokens",
                "arm_inventory_digests",
                "cells",
                "cell_details",
            }:
                return None
            if record["fixture"] != case["fixture_id"] or record["topic"] != case["topic"]:
                return None
            expected_digest = _hashlib.sha256(case["values"][0].encode("utf-8")).hexdigest()
            if record["expected_digest"] != expected_digest:
                return None
            arm_tokens = record["arm_tokens"]
            if type(arm_tokens) is not dict or set(arm_tokens) != set(rotation.ARM_KEYS):
                return None
            if any(type(v) is not int or v < 0 for v in arm_tokens.values()):
                return None
            digests = record["arm_inventory_digests"]
            if type(digests) is not dict or set(digests) != set(rotation.SPAN_ARMS):
                return None
            if any(
                type(v) is not str or not _o_rotation_digest_ok(v) or v is None
                for v in digests.values()
            ):
                return None
            cells = record["cells"]
            details = record["cell_details"]
            if type(cells) is not dict or set(cells) != set(rotation.ARM_KEYS):
                return None
            if type(details) is not dict or set(details) != set(rotation.ARM_KEYS):
                return None
            for arm in rotation.ARM_KEYS:
                cell = cells[arm]
                detail = details[arm]
                if cell is None:
                    if detail not in rotation.CELL_DETAILS:
                        return None
                else:
                    if detail != "":
                        return None
                    if not _o_rotation_cell_ok(cell, expected_digest):
                        return None
                    if cell["input_tokens"] != arm_tokens[arm]:
                        return None
        raw[name] = records
        aggregates[name] = rotation.aggregate_candidate(records)
    if total > rotation.MAX_GROUND_RECORDS:
        return None
    if not raw:
        return None
    return {
        "aggregates": aggregates,
        "conditions": {arm: rotation.arm_selection(raw, arm) for arm in rotation.CONDITION_ARMS},
        "n_candidates": len(candidates),
        "n_cases": len(generated),
    }


def _o_rotation_valid(value: Any) -> bool:
    """Oracle rotation validator: mirrors diffreports._valid_rotation."""
    if type(value) is not dict or value.get("available") is not True:
        return False
    if set(value) != {
        "available",
        "detail",
        "protocol",
        "candidates",
        "conditions",
        "n_candidates",
        "n_cases",
    }:
        return False
    if value.get("detail") != "":
        return False
    recomputed = _o_rotation_recompute(value)
    if recomputed is None:
        return False
    candidates = value["candidates"]
    for name, expected in recomputed["aggregates"].items():
        entry = candidates[name]
        for key, val in expected.items():
            if entry.get(key) != val:
                return False
    if value.get("conditions") != recomputed["conditions"]:
        return False
    if value.get("n_candidates") != recomputed["n_candidates"]:
        return False
    return bool(value.get("n_cases") == recomputed["n_cases"])


def _o_rotation(old: Any, new: Any) -> tuple[dict[str, Any], bool, bool]:
    """Oracle rotation section: mirrors diffreports.diff_rotation."""
    old_ok = _o_rotation_valid(old)
    new_ok = _o_rotation_valid(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    diff: dict[str, Any] = {"available": available, "changed": False}
    asymmetric = available not in ("both", "neither")
    if available != "both":
        return diff, False, asymmetric
    protocol_drift = old["protocol"].get("fingerprint") != new["protocol"].get("fingerprint")
    generator_drift = old["protocol"].get("ground_sha256") != new["protocol"].get("ground_sha256")
    backend_drift = old["protocol"].get("backend_fingerprint") != new["protocol"].get(
        "backend_fingerprint"
    )
    comparable = not protocol_drift and not generator_drift and not backend_drift
    old_candidates = old["candidates"]
    new_candidates = new["candidates"]
    added = sorted(set(new_candidates) - set(old_candidates))
    removed = sorted(set(old_candidates) - set(new_candidates))
    candidate_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_candidates) & set(new_candidates)):
        old_entry = old_candidates[name]
        new_entry = new_candidates[name]
        if old_entry.get("status") != "available" or new_entry.get("status") != "available":
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
        metrics = [
            ("classification", old_entry["classification"], new_entry["classification"]),
            (
                "invariance_falsified",
                old_entry["invariance_falsified"],
                new_entry["invariance_falsified"],
            ),
            (
                "n_flip_any",
                old_entry["headline"]["n_flip_any"],
                new_entry["headline"]["n_flip_any"],
            ),
            (
                "n_headline",
                old_entry["headline"]["n_headline"],
                new_entry["headline"]["n_headline"],
            ),
            (
                "n_flips_mirror",
                old_entry["mirror"]["n_flips_mirror"],
                new_entry["mirror"]["n_flips_mirror"],
            ),
            (
                "n_discordant",
                old_entry["reversal"]["n_discordant"],
                new_entry["reversal"]["n_discordant"],
            ),
            ("n_screened", old_entry["n_screened"], new_entry["n_screened"]),
            (
                "n_boundary_screened",
                old_entry["n_boundary_screened"],
                new_entry["n_boundary_screened"],
            ),
        ]
        for metric, old_value, new_value in metrics:
            if old_value != new_value:
                candidate_drift.append(
                    {"candidate": name, "metric": metric, "old": old_value, "new": new_value}
                )
    condition_drift: list[dict[str, Any]] = []
    for arm in rotation.CONDITION_ARMS:
        if old["conditions"][arm] != new["conditions"][arm]:
            condition_drift.append(
                {"condition": arm, "old": old["conditions"][arm], "new": new["conditions"][arm]}
            )
    diff["candidate_drift"] = candidate_drift
    diff["condition_drift"] = condition_drift
    diff["candidates_added"] = added
    diff["candidates_removed"] = removed
    diff["protocol_drift"] = protocol_drift
    diff["generator_drift"] = generator_drift
    diff["backend_drift"] = backend_drift
    diff["comparable"] = comparable
    changed = bool(
        candidate_drift
        or condition_drift
        or added
        or removed
        or protocol_drift
        or generator_drift
        or backend_drift
    )
    diff["changed"] = changed
    return diff, changed, asymmetric


def _o_density_recompute(value: Any) -> dict[str, Any] | None:
    """Oracle density recompute: mirrors
    diffreports._density_recompute. The oracle rebuilds the
    protocol and the case set through the producer module
    (density) where the implementation rebuilds them from
    literal tables -- a drift in either side's constants breaks
    the agreement the fuzz harness asserts.
    """
    import json as _json

    protocol = value.get("protocol")
    if type(protocol) is not dict:
        return None
    seed = protocol.get("seed")
    backend_fp = protocol.get("backend_fingerprint")
    if type(seed) is not int or isinstance(seed, bool) or type(backend_fp) is not str:
        return None
    expected_protocol = dict(density.protocol_object(seed, backend_fp))
    expected_protocol["fingerprint"] = density.protocol_fingerprint(seed, backend_fp)
    if protocol != expected_protocol:
        return None
    candidates = value.get("candidates")
    if type(candidates) is not dict or not candidates:
        return None
    generated = density.generate_cases(seed)
    total = 0
    records_bytes = 0
    aggregates: dict[str, Any] = {}
    for name, entry in candidates.items():
        if type(entry) is not dict:
            return None
        if entry.get("status") == "unavailable":
            if set(entry) != {"status", "detail", "records"} or entry.get("records") != []:
                return None
            continue
        if entry.get("status") != "available":
            return None
        if set(entry) != {
            "available",
            "detail",
            "n_generated",
            "n_boundary_screened",
            "n_detail_screened",
            "n_no_boundary",
            "n_zero_word",
            "n_live",
            "by_density",
            "task_presence",
            "position_k16",
            "position_bands",
            "by_kind",
            "joint_k16",
            "sham",
            "discordance_by_position",
            "decision",
            "status",
            "records",
        }:
            return None
        records = entry["records"]
        if type(records) is not list or len(records) != len(density.ARM_KEYS) * len(generated):
            return None
        total += len(records)
        records_bytes += len(
            _json.dumps(records, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        )
        if records_bytes > density.MAX_DENSITY_PAYLOAD_BYTES:
            return None
        expected_order = [
            (case_index, arm) for case_index in range(len(generated)) for arm in density.ARM_KEYS
        ]
        for record, (case_index, arm) in zip(records, expected_order, strict=True):
            if type(record) is not dict or set(record) != {
                "candidate",
                "case_index",
                "arm",
                "detail",
                "case_screened",
                "boundary_found",
                "answer_digest",
                "word_hashes",
                "word_list_digest",
                "word_count",
                "input_tokens",
                "content_digest",
            }:
                return None
            if record["candidate"] != name:
                return None
            if record["case_index"] != case_index or record["arm"] != arm:
                return None
            if record["detail"] not in density.RECORD_DETAILS:
                return None
            if type(record["case_screened"]) is not bool:
                return None
            if type(record["boundary_found"]) is not bool:
                return None
            hashes = record["word_hashes"]
            if type(hashes) is not list or any(
                not (type(h) is str and len(h) == 64 and all(c in "0123456789abcdef" for c in h))
                for h in hashes
            ):
                return None
            if record["word_count"] != len(hashes):
                return None
            if record["word_list_digest"] != density.commitment_digest(list(hashes)):
                return None
            if type(record["input_tokens"]) is not int or record["input_tokens"] < 0:
                return None
            if record["answer_digest"] is not None and not (
                type(record["answer_digest"]) is str
                and len(record["answer_digest"]) == 64
                and all(c in "0123456789abcdef" for c in record["answer_digest"])
            ):
                return None
            case = generated[case_index]
            if record["content_digest"] != density.digest_text(density.arm_content(case, arm)):
                return None
            if record["detail"] != "" or record["case_screened"]:
                if hashes or record["answer_digest"] is not None or record["boundary_found"]:
                    return None
            elif record["answer_digest"] is None:
                return None
        aggregates[name] = density.aggregate_candidate(records, generated)
    if total > density.MAX_DENSITY_RECORDS:
        return None
    if not aggregates:
        return None
    return {
        "aggregates": aggregates,
        "n_candidates": len(candidates),
        "n_cases": len(generated),
    }


def _o_density_valid(value: Any) -> bool:
    """Oracle density validator: mirrors diffreports._valid_density."""
    if type(value) is not dict or value.get("available") is not True:
        return False
    if set(value) != {
        "available",
        "detail",
        "protocol",
        "candidates",
        "n_candidates",
        "n_cases",
    }:
        return False
    if value.get("detail") != "":
        return False
    recomputed = _o_density_recompute(value)
    if recomputed is None:
        return False
    candidates = value["candidates"]
    for name, expected in recomputed["aggregates"].items():
        entry = candidates[name]
        for key, val in expected.items():
            if entry.get(key) != val:
                return False
    if value.get("n_candidates") != recomputed["n_candidates"]:
        return False
    return bool(value.get("n_cases") == recomputed["n_cases"])


def _o_density(old: Any, new: Any) -> tuple[dict[str, Any], bool, bool]:
    """Oracle density section: mirrors diffreports.diff_density."""
    old_ok = _o_density_valid(old)
    new_ok = _o_density_valid(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    diff: dict[str, Any] = {"available": available, "changed": False}
    asymmetric = available not in ("both", "neither")
    if available != "both":
        return diff, False, asymmetric
    protocol_drift = old["protocol"].get("fingerprint") != new["protocol"].get("fingerprint")
    generator_drift = old["protocol"].get("ground_sha256") != new["protocol"].get("ground_sha256")
    backend_drift = old["protocol"].get("backend_fingerprint") != new["protocol"].get(
        "backend_fingerprint"
    )
    comparable = not protocol_drift and not generator_drift and not backend_drift
    old_candidates = old["candidates"]
    new_candidates = new["candidates"]
    added = sorted(set(new_candidates) - set(old_candidates))
    removed = sorted(set(old_candidates) - set(new_candidates))
    candidate_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_candidates) & set(new_candidates)):
        old_entry = old_candidates[name]
        new_entry = new_candidates[name]
        if old_entry.get("status") != "available" or new_entry.get("status") != "available":
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
        metrics = [
            ("verdict", old_entry["decision"]["verdict"], new_entry["decision"]["verdict"]),
            ("falsified", old_entry["decision"]["falsified"], new_entry["decision"]["falsified"]),
            ("f", old_entry["decision"]["f"], new_entry["decision"]["f"]),
            ("p", old_entry["decision"]["p"], new_entry["decision"]["p"]),
            (
                "discordant_cases",
                old_entry["decision"]["discordant_cases"],
                new_entry["decision"]["discordant_cases"],
            ),
            ("sign_p", old_entry["decision"]["sign_p"], new_entry["decision"]["sign_p"]),
            (
                "joint_satisfied",
                old_entry["joint_k16"]["satisfied"],
                new_entry["joint_k16"]["satisfied"],
            ),
            ("joint_live", old_entry["joint_k16"]["live"], new_entry["joint_k16"]["live"]),
            ("n_live", old_entry["n_live"], new_entry["n_live"]),
            (
                "n_boundary_screened",
                old_entry["n_boundary_screened"],
                new_entry["n_boundary_screened"],
            ),
            (
                "n_detail_screened",
                old_entry["n_detail_screened"],
                new_entry["n_detail_screened"],
            ),
            ("n_zero_word", old_entry["n_zero_word"], new_entry["n_zero_word"]),
        ]
        for metric, old_value, new_value in metrics:
            if old_value != new_value:
                candidate_drift.append(
                    {"candidate": name, "metric": metric, "old": old_value, "new": new_value}
                )
    diff["candidate_drift"] = candidate_drift
    diff["candidates_added"] = added
    diff["candidates_removed"] = removed
    diff["protocol_drift"] = protocol_drift
    diff["generator_drift"] = generator_drift
    diff["backend_drift"] = backend_drift
    diff["comparable"] = comparable
    changed = bool(
        candidate_drift or added or removed or protocol_drift or generator_drift or backend_drift
    )
    diff["changed"] = changed
    return diff, changed, asymmetric


def _o_field_order_recompute(value: Any) -> dict[str, Any] | None:
    """Oracle field-order recompute: mirrors
    diffreports._fo_recompute. The oracle rebuilds the
    protocol through the producer module (fieldorder) where
    the implementation rebuilds it from literal tables, checks
    every record in canonical (spec-table, arm) order against
    the re-rendered arm prompt, verifies every stored field
    value is the canonical form of its parse and the document
    digest consistent, and recomputes the aggregates through
    the producer's aggregation.
    """
    protocol = value.get("protocol")
    if type(protocol) is not dict:
        return None
    seed = protocol.get("seed")
    backend_fp = protocol.get("backend_fingerprint")
    if type(seed) is not int or isinstance(seed, bool) or type(backend_fp) is not str:
        return None
    expected_protocol = dict(fieldorder.protocol_object(seed, backend_fp))
    expected_protocol["fingerprint"] = fieldorder.protocol_fingerprint(seed, backend_fp)
    if protocol != expected_protocol:
        return None
    candidates = value.get("candidates")
    if type(candidates) is not dict or not candidates:
        return None
    record_keys = {
        "candidate",
        "spec_id",
        "arm",
        "detail",
        "case_screened",
        "boundary_found",
        "duplicate_keys",
        "answer_digest",
        "document_kind",
        "document_digest",
        "field_values",
        "input_tokens",
        "content_digest",
    }

    def _digest_ok(item: Any) -> bool:
        """Digest ok."""
        return (
            type(item) is str and len(item) == 64 and all(ch in "0123456789abcdef" for ch in item)
        )

    total = 0
    records_bytes = 0
    aggregates: dict[str, Any] = {}
    expected_order = [
        (spec["id"], arm) for spec in fieldorder.FIELD_ORDER_SPECS for arm in fieldorder.ARM_KEYS
    ]
    for name, entry in candidates.items():
        if type(entry) is not dict:
            return None
        status = entry.get("status")
        if status == "unavailable":
            if set(entry) != {"status", "detail", "records"} or entry.get("records") != []:
                return None
            continue
        if status != "available":
            return None
        if set(entry) != {
            "n_generated",
            "n_distinct_prompts",
            "n_boundary_screened",
            "n_detail_screened",
            "n_no_boundary",
            "n_live",
            "n_duplicate_keys",
            "by_arm",
            "per_field",
            "answer_by_arm",
            "whole_doc_by_arm",
            "pairing",
            "decision",
            "extraction_decision",
            "field_count_split",
            "status",
            "records",
        }:
            return None
        records = entry.get("records")
        if type(records) is not list or len(records) != len(expected_order):
            return None
        total += len(records)
        records_bytes += len(
            json.dumps(records, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        )
        if records_bytes > fieldorder.MAX_FIELD_ORDER_PAYLOAD_BYTES:
            return None
        for record, (spec_id, arm) in zip(records, expected_order, strict=True):
            if type(record) is not dict or set(record) != record_keys:
                return None
            if record["candidate"] != name:
                return None
            if record["spec_id"] != spec_id or record["arm"] != arm:
                return None
            if record["detail"] not in ("", "arm_evaluation_failed", "arm_context_exceeded"):
                return None
            if type(record["case_screened"]) is not bool:
                return None
            if type(record["boundary_found"]) is not bool:
                return None
            if type(record["duplicate_keys"]) is not bool:
                return None
            if type(record["input_tokens"]) is not int or record["input_tokens"] < 0:
                return None
            if record["answer_digest"] is not None and not _digest_ok(record["answer_digest"]):
                return None
            if not _digest_ok(record["content_digest"]):
                return None
            spec = fieldorder.spec_by_id(spec_id)
            if record["content_digest"] != fieldorder.digest_text(fieldorder.arm_prompt(spec, arm)):
                return None
            kind = record["document_kind"]
            if kind not in ("", "object", "non_object", "unparseable"):
                return None
            field_values = record["field_values"]
            if type(field_values) is not dict:
                return None
            if record["detail"] != "" or record["case_screened"]:
                if (
                    kind != ""
                    or field_values
                    or record["document_digest"] is not None
                    or record["answer_digest"] is not None
                    or record["boundary_found"]
                    or record["duplicate_keys"]
                ):
                    return None
                continue
            if record["answer_digest"] is None:
                return None
            if (
                sum(len(item) for item in field_values.values() if type(item) is str)
                > fieldorder.FIELD_VALUES_CAP
            ):
                return None
            for item in field_values.values():
                if item is None:
                    continue
                if type(item) is not str:
                    return None
                try:
                    parsed = json.loads(item)
                except (ValueError, TypeError, RecursionError):
                    return None
                if type(parsed) is float and not math.isfinite(parsed):
                    return None
                try:
                    if fieldorder._canonical_json(parsed) != item:
                        return None
                except (ValueError, TypeError, RecursionError):
                    return None
            if kind != "object":
                if field_values or record["document_digest"] is not None:
                    return None
            elif any(item is None for item in field_values.values()):
                if record["document_digest"] is not None:
                    return None
            else:
                document = {key: json.loads(item) for key, item in field_values.items()}
                try:
                    expected_digest = fieldorder.digest_text(fieldorder._canonical_json(document))
                except (ValueError, TypeError, RecursionError):
                    return None
                if record["document_digest"] != expected_digest:
                    return None
        aggregates[name] = fieldorder.aggregate_candidate(records)
    if total > fieldorder.MAX_FIELD_ORDER_RECORDS:
        return None
    if not aggregates:
        return None
    return {
        "aggregates": aggregates,
        "n_candidates": len(candidates),
        "n_specs": len(fieldorder.FIELD_ORDER_SPECS),
    }


def _o_field_order_valid(value: Any) -> bool:
    """Oracle field-order validator: mirrors
    diffreports._valid_field_order.
    """
    if type(value) is not dict or value.get("available") is not True:
        return False
    if set(value) != {
        "available",
        "detail",
        "protocol",
        "candidates",
        "n_candidates",
        "n_specs",
    }:
        return False
    if value.get("detail") != "":
        return False
    recomputed = _o_field_order_recompute(value)
    if recomputed is None:
        return False
    candidates = value["candidates"]
    for name, expected in recomputed["aggregates"].items():
        entry = candidates[name]
        for key, val in expected.items():
            if entry.get(key) != val:
                return False
    if value.get("n_candidates") != recomputed["n_candidates"]:
        return False
    return bool(value.get("n_specs") == recomputed["n_specs"])


def _o_field_order(old: Any, new: Any) -> tuple[dict[str, Any], bool, bool]:
    """Oracle field-order section: mirrors
    diffreports.diff_field_order.
    """
    old_ok = _o_field_order_valid(old)
    new_ok = _o_field_order_valid(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    diff: dict[str, Any] = {"available": available, "changed": False}
    asymmetric = available not in ("both", "neither")
    if available != "both":
        return diff, False, asymmetric
    protocol_drift = old["protocol"].get("fingerprint") != new["protocol"].get("fingerprint")
    generator_drift = old["protocol"].get("inventory_digest") != new["protocol"].get(
        "inventory_digest"
    )
    backend_drift = old["protocol"].get("backend_fingerprint") != new["protocol"].get(
        "backend_fingerprint"
    )
    comparable = not protocol_drift and not generator_drift and not backend_drift
    old_candidates = old["candidates"]
    new_candidates = new["candidates"]
    added = sorted(set(new_candidates) - set(old_candidates))
    removed = sorted(set(old_candidates) - set(new_candidates))
    candidate_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_candidates) & set(new_candidates)):
        old_entry = old_candidates[name]
        new_entry = new_candidates[name]
        if old_entry.get("status") != "available" or new_entry.get("status") != "available":
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
        metrics = [
            ("verdict", old_entry["decision"]["verdict"], new_entry["decision"]["verdict"]),
            (
                "direction",
                old_entry["decision"]["direction"],
                new_entry["decision"]["direction"],
            ),
            (
                "n_first_only",
                old_entry["decision"]["n_first_only"],
                new_entry["decision"]["n_first_only"],
            ),
            (
                "n_last_only",
                old_entry["decision"]["n_last_only"],
                new_entry["decision"]["n_last_only"],
            ),
            (
                "n_discordant",
                old_entry["decision"]["n_discordant"],
                new_entry["decision"]["n_discordant"],
            ),
            ("sign_p", old_entry["decision"]["sign_p"], new_entry["decision"]["sign_p"]),
            ("n_paired", old_entry["pairing"]["n_paired"], new_entry["pairing"]["n_paired"]),
            (
                "first_rate",
                old_entry["pairing"]["first_rate"],
                new_entry["pairing"]["first_rate"],
            ),
            (
                "last_rate",
                old_entry["pairing"]["last_rate"],
                new_entry["pairing"]["last_rate"],
            ),
            ("delta", old_entry["pairing"]["delta"], new_entry["pairing"]["delta"]),
            ("n_live", old_entry["n_live"], new_entry["n_live"]),
            (
                "n_boundary_screened",
                old_entry["n_boundary_screened"],
                new_entry["n_boundary_screened"],
            ),
            (
                "n_detail_screened",
                old_entry["n_detail_screened"],
                new_entry["n_detail_screened"],
            ),
            (
                "n_duplicate_keys",
                old_entry["n_duplicate_keys"],
                new_entry["n_duplicate_keys"],
            ),
        ]
        for arm in fieldorder.ARM_KEYS:
            metrics.append(
                (
                    f"answer_rate_{arm}",
                    old_entry["answer_by_arm"][arm]["rate"],
                    new_entry["answer_by_arm"][arm]["rate"],
                )
            )
        for metric, old_value, new_value in metrics:
            if old_value != new_value:
                candidate_drift.append(
                    {"candidate": name, "metric": metric, "old": old_value, "new": new_value}
                )
    diff["candidate_drift"] = candidate_drift
    diff["candidates_added"] = added
    diff["candidates_removed"] = removed
    diff["protocol_drift"] = protocol_drift
    diff["generator_drift"] = generator_drift
    diff["backend_drift"] = backend_drift
    diff["comparable"] = comparable
    changed = bool(
        candidate_drift or added or removed or protocol_drift or generator_drift or backend_drift
    )
    diff["changed"] = changed
    return diff, changed, asymmetric


def _o_choiceaudit_recompute(value: Any) -> dict[str, Any] | None:
    """Oracle choice-audit recompute: mirrors
    diffreports._choiceaudit_recompute. The oracle rebuilds the
    protocol and the aggregates through the producer module
    (choiceaudit) where the implementation rebuilds them from
    literal tables; the construction block and every record are
    checked structurally and every stored float is asserted
    float32-representable. A drift in either side's constants
    breaks the agreement.
    """
    import math as _math
    import struct as _struct

    def _f32(v: float) -> float:
        """F32."""
        if _math.isfinite(v) and abs(v) > float.fromhex("0x1.fffffep+127"):
            return _math.copysign(_math.inf, v)
        return _struct.unpack("<f", _struct.pack("<f", v))[0]

    protocol = value.get("protocol")
    block = value.get("fixtures")
    if type(protocol) is not dict or type(block) is not list or not block:
        return None
    fixture_ids: list[str] = []
    seen_ids: set[str] = set()
    for entry in block:
        if type(entry) is not dict or set(entry) != {
            "fixture",
            "group",
            "excluded",
            "numeric",
            "choices",
            "n_distinct_choices",
            "constructible",
        }:
            return None
        if type(entry["fixture"]) is not str or type(entry["group"]) is not str:
            return None
        if entry["fixture"] in seen_ids:
            return None
        seen_ids.add(entry["fixture"])
        fixture_ids.append(entry["fixture"])
        if entry["excluded"] not in (None, "blank_expected"):
            return None
        if type(entry["numeric"]) is not bool or type(entry["constructible"]) is not bool:
            return None
        choices = entry["choices"]
        if type(choices) is not list or entry["n_distinct_choices"] != len(choices):
            return None
        if entry["excluded"] is not None:
            if choices or entry["constructible"]:
                return None
            continue
        if entry["constructible"] != (len(choices) == 1 + choiceaudit.K_DONORS):
            return None
        digests: list[str] = []
        for index, choice in enumerate(choices):
            if type(choice) is not dict or set(choice) != {
                "source_fixture",
                "group",
                "digest",
            }:
                return None
            if type(choice["group"]) is not str:
                return None
            digest = choice["digest"]
            if (
                type(digest) is not str
                or len(digest) != 64
                or any(ch not in "0123456789abcdef" for ch in digest)
            ):
                return None
            digests.append(digest)
            if index == 0:
                if choice["source_fixture"] is not None or choice["group"] != entry["group"]:
                    return None
            else:
                if type(choice["source_fixture"]) is not str:
                    return None
                if choice["group"] == entry["group"]:
                    return None
        if len(set(digests)) != len(digests):
            return None
    by_id = {entry["fixture"]: entry for entry in block}
    for entry in block:
        for choice in entry["choices"][1:]:
            donor_entry = by_id.get(choice["source_fixture"])
            if donor_entry is None or donor_entry["excluded"] is not None:
                return None
            if donor_entry["group"] != choice["group"]:
                return None
    backend_fp = protocol.get("backend_fingerprint")
    if type(backend_fp) is not str:
        return None
    expected_protocol = dict(choiceaudit.protocol_object(fixture_ids, block, backend_fp))
    expected_protocol["fingerprint"] = choiceaudit.protocol_fingerprint(
        fixture_ids, block, backend_fp
    )
    if protocol != expected_protocol:
        return None
    candidates = value.get("candidates")
    if type(candidates) is not dict or not candidates:
        return None
    total = 0
    records_bytes = 0
    aggregates: dict[str, Any] = {}
    raw_aggregates: dict[str, Any] = {}
    for name, entry in candidates.items():
        if type(entry) is not dict:
            return None
        if entry.get("status") == "unavailable":
            if set(entry) != {"status", "detail", "records"} or entry.get("records") != []:
                return None
            continue
        if entry.get("status") != "available" or entry.get("detail") != "":
            return None
        records = entry["records"]
        if type(records) is not list or len(records) != len(block):
            return None
        total += len(records)
        records_bytes += len(
            json.dumps(records, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        )
        if records_bytes > choiceaudit.MAX_RECORDS_BYTES:
            return None
        for record, fixture_entry in zip(records, block, strict=True):
            if type(record) is not dict or set(record) != {
                "fixture",
                "obs_status",
                "exact_match",
                "boundary_found",
                "prompt_reason",
                "cells",
            }:
                return None
            if record["fixture"] != fixture_entry["fixture"]:
                return None
            if record["obs_status"] is not None and type(record["obs_status"]) is not str:
                return None
            for fact in ("exact_match", "boundary_found"):
                if record[fact] is not None and type(record[fact]) is not bool:
                    return None
            if record["prompt_reason"] is not None and type(record["prompt_reason"]) is not str:
                return None
            cells = record["cells"]
            if type(cells) is not list:
                return None
            if record["prompt_reason"] is not None or not fixture_entry["constructible"]:
                if cells:
                    return None
                continue
            if len(cells) != 1 + choiceaudit.K_DONORS:
                return None
            for cell in cells:
                if type(cell) is not dict or set(cell) != {
                    "status",
                    "reason",
                    "total",
                    "mean",
                    "tokens",
                    "bytes",
                }:
                    return None
                if type(cell["tokens"]) is not int or type(cell["bytes"]) is not int:
                    return None
                if cell["status"] == "ok":
                    if cell["reason"] is not None:
                        return None
                    for cell_value in (cell["total"], cell["mean"]):
                        if (
                            type(cell_value) is not float
                            or not _math.isfinite(cell_value)
                            or _f32(cell_value) != cell_value
                        ):
                            return None
                    if cell["tokens"] < 1 or cell["bytes"] < 1:
                        return None
                elif cell["status"] == "ok_empty":
                    if (
                        cell["reason"] is not None
                        or cell["total"] != 0.0
                        or cell["mean"] is not None
                        or cell["tokens"] != 0
                        or cell["bytes"] != 0
                    ):
                        return None
                elif cell["status"] == "unavailable":
                    if type(cell["reason"]) is not str:
                        return None
                    if cell["total"] is not None or cell["mean"] is not None:
                        return None
                    if cell["tokens"] < 0 or cell["bytes"] < 0:
                        return None
                else:
                    return None
        raw_aggregates[name] = choiceaudit.aggregate_candidate(records, block)
        aggregates[name] = {
            key: val for key, val in raw_aggregates[name].items() if key != "_verdicts"
        }
    if total > choiceaudit.MAX_RECORDS:
        return None
    if not aggregates:
        return None
    method = value.get("method_comparison")
    if type(method) is not dict:
        return None
    expected_method = choiceaudit.method_comparison(
        raw_aggregates,
        tuple(sorted(candidates)),
        method.get("primary_decision"),
        method.get("primary_winner"),
    )
    return {
        "aggregates": aggregates,
        "method_comparison": expected_method,
        "discordance": choiceaudit.discordance_summary(raw_aggregates),
        "n_candidates": len(candidates),
        "n_fixtures": len(block),
    }


def _o_choiceaudit_valid(value: Any) -> bool:
    """Oracle choice-audit validator: mirrors diffreports._valid_choiceaudit."""
    if type(value) is not dict or value.get("available") is not True:
        return False
    if set(value) != {
        "available",
        "detail",
        "protocol",
        "fixtures",
        "candidates",
        "method_comparison",
        "discordance",
        "n_candidates",
        "n_fixtures",
    }:
        return False
    if value.get("detail") != "":
        return False
    recomputed = _o_choiceaudit_recompute(value)
    if recomputed is None:
        return False
    candidates = value["candidates"]
    for name, expected in recomputed["aggregates"].items():
        entry = candidates[name]
        if set(entry) != set(expected) | {"status", "detail", "records"}:
            return False
        for key, val in expected.items():
            if entry.get(key) != val:
                return False
    if value.get("method_comparison") != recomputed["method_comparison"]:
        return False
    if value.get("discordance") != recomputed["discordance"]:
        return False
    if value.get("n_candidates") != recomputed["n_candidates"]:
        return False
    return bool(value.get("n_fixtures") == recomputed["n_fixtures"])


def _o_choiceaudit(old: Any, new: Any) -> tuple[dict[str, Any], bool, bool]:
    """Oracle choice-audit section: mirrors diffreports.diff_choiceaudit."""
    old_ok = _o_choiceaudit_valid(old)
    new_ok = _o_choiceaudit_valid(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    diff: dict[str, Any] = {"available": available, "changed": False}
    asymmetric = available not in ("both", "neither")
    if available != "both":
        return diff, False, asymmetric
    protocol_drift = old["protocol"].get("fingerprint") != new["protocol"].get("fingerprint")
    donor_drift = old["protocol"].get("donor_map_sha256") != new["protocol"].get("donor_map_sha256")
    backend_drift = old["protocol"].get("backend_fingerprint") != new["protocol"].get(
        "backend_fingerprint"
    )
    comparable = not protocol_drift and not donor_drift and not backend_drift
    old_candidates = old["candidates"]
    new_candidates = new["candidates"]
    added = sorted(set(new_candidates) - set(old_candidates))
    removed = sorted(set(old_candidates) - set(new_candidates))
    candidate_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_candidates) & set(new_candidates)):
        old_entry = old_candidates[name]
        new_entry = new_candidates[name]
        if old_entry.get("status") != "available" or new_entry.get("status") != "available":
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
        metrics = [
            (
                "accuracy_total_correct",
                old_entry["accuracy"]["total"]["correct"],
                new_entry["accuracy"]["total"]["correct"],
            ),
            (
                "accuracy_total_n",
                old_entry["accuracy"]["total"]["n"],
                new_entry["accuracy"]["total"]["n"],
            ),
            (
                "n_discordant",
                old_entry["discordance"]["n_discordant"],
                new_entry["discordance"]["n_discordant"],
            ),
            ("n_evaluable", old_entry["n_evaluable"], new_entry["n_evaluable"]),
        ]
        for metric, old_value, new_value in metrics:
            if old_value != new_value:
                candidate_drift.append(
                    {"candidate": name, "metric": metric, "old": old_value, "new": new_value}
                )
    method_drift: list[dict[str, Any]] = []
    if old["method_comparison"] != new["method_comparison"]:
        method_drift.append({"old": old["method_comparison"], "new": new["method_comparison"]})
    if old["discordance"] != new["discordance"]:
        method_drift.append({"old": old["discordance"], "new": new["discordance"]})
    diff["candidate_drift"] = candidate_drift
    diff["method_drift"] = method_drift
    diff["candidates_added"] = added
    diff["candidates_removed"] = removed
    diff["protocol_drift"] = protocol_drift
    diff["donor_drift"] = donor_drift
    diff["backend_drift"] = backend_drift
    diff["comparable"] = comparable
    changed = bool(
        candidate_drift
        or method_drift
        or added
        or removed
        or protocol_drift
        or donor_drift
        or backend_drift
    )
    diff["changed"] = changed
    return diff, changed, asymmetric


def _o_precision_recompute(value: Any) -> dict[str, Any] | None:
    """Oracle precision recompute: mirrors
    diffreports._precision_recompute. The protocol digest is
    rebuilt through the producer module (precision) where the
    implementation rebuilds it from literal tables; every record
    and cell is checked structurally, the digest verdict binding
    is re-derived, the stage-attested arm-passed verdicts are
    consistency-checked against the record universe, and the
    derived aggregation (paired flip matrix, unpaired list,
    divergence and logprob summaries, paired-population
    selections, survival, falsification) is mirrored
    independently here. A drift in either side's constants
    breaks the agreement.
    """
    from model_protocol_lab import precision as _prec

    def _dig(v: Any) -> bool:
        """Dig."""
        return type(v) is str and len(v) == 64 and all(c in "0123456789abcdef" for c in v)

    def _cell_ok(cell: Any, expected_digest: str) -> bool:
        """Cell ok."""
        if type(cell) is not dict or set(cell) != {
            "answer_digest",
            "boundary_found",
            "logprob",
            "output_digest",
            "verdict",
        }:
            return False
        if not _dig(cell["answer_digest"]) or not _dig(cell["output_digest"]):
            return False
        if type(cell["boundary_found"]) is not bool or type(cell["verdict"]) is not bool:
            return False
        lp = cell["logprob"]
        if lp is not None and (
            type(lp) not in (int, float)
            or isinstance(lp, bool)
            or (type(lp) is float and (lp != lp or lp in (float("inf"), float("-inf"))))
        ):
            return False
        return cell["verdict"] == (
            cell["answer_digest"] == expected_digest and cell["boundary_found"]
        )

    protocol = value.get("protocol")
    if type(protocol) is not dict or set(protocol) != {
        "arm_dtype",
        "bf16_fingerprint",
        "boundaries",
        "candidates",
        "declared_dtype",
        "fingerprint",
        "fixture_set_sha256",
        "fp32_fingerprint",
        "max_new_tokens",
        "max_records",
        "max_records_bytes",
        "native_dtype",
        "seed",
        "version",
    }:
        return None
    if protocol["version"] != 1 or type(protocol["version"]) is not int:
        return None
    if type(protocol["seed"]) is not int or protocol["seed"] < 0:
        return None
    if type(protocol["max_new_tokens"]) is not int or not 1 <= protocol["max_new_tokens"] <= 4096:
        return None
    if protocol["max_records"] != 2400 or protocol["max_records_bytes"] != 12 * 1024 * 1024:
        return None
    if not _dig(protocol["fixture_set_sha256"]):
        return None
    candidates = protocol["candidates"]
    if (
        type(candidates) is not list
        or not candidates
        or any(type(n) is not str or not n for n in candidates)
        or len(set(candidates)) != len(candidates)
    ):
        return None
    boundaries = protocol["boundaries"]
    if type(boundaries) is not dict or set(boundaries) != set(candidates):
        return None
    for name in candidates:
        ids = boundaries[name]
        if type(ids) is not list or not ids or any(type(t) is not int or t < 0 for t in ids):
            return None
    for key in ("fp32_fingerprint", "bf16_fingerprint"):
        if type(protocol[key]) is not str or not protocol[key]:
            return None
    if protocol["fp32_fingerprint"] == protocol["bf16_fingerprint"]:
        return None
    if protocol["native_dtype"] != "float32" or protocol["arm_dtype"] != "bfloat16":
        return None
    if protocol["declared_dtype"] is not None and type(protocol["declared_dtype"]) is not str:
        return None
    if protocol["fingerprint"] != _prec.protocol_fingerprint(protocol):
        return None
    decision = value.get("primary_decision")
    winner = value.get("primary_winner")
    if decision not in {
        "compatible_on_fixture_set",
        "held_out_failed",
        "ambiguous",
        "no_candidate",
        "cancelled",
        "failed",
    }:
        return None
    blocks_in = value.get("candidates")
    if type(blocks_in) is not dict or set(blocks_in) != set(candidates):
        return None
    if decision in ("compatible_on_fixture_set", "held_out_failed"):
        if type(winner) is not str or winner not in blocks_in:
            return None
    elif winner is not None:
        return None
    records = value.get("records")
    if type(records) is not list or len(records) > protocol["max_records"]:
        return None
    blob = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    if len(blob.encode("utf-8")) > protocol["max_records_bytes"]:
        return None
    seen: set = set()
    universe: set = set()
    record_candidates: set = set()
    for record in records:
        if type(record) is not dict or set(record) != {
            "bf16",
            "candidate",
            "diverged",
            "expected_digest",
            "first_divergence",
            "fixture",
            "fp32",
        }:
            return None
        name = record["candidate"]
        fixture = record["fixture"]
        if type(name) is not str or name not in candidates or type(fixture) is not str:
            return None
        if (name, fixture) in seen:
            return None
        seen.add((name, fixture))
        universe.add(fixture)
        record_candidates.add(name)
        if not _dig(record["expected_digest"]):
            return None
        for arm in ("fp32", "bf16"):
            cell = record[arm]
            if cell is not None and not _cell_ok(cell, str(record["expected_digest"])):
                return None
        both = type(record["fp32"]) is dict and type(record["bf16"]) is dict
        if (record["diverged"] is None) == both:
            return None
        if record["diverged"] is False and record["first_divergence"] is not None:
            return None
        if record["diverged"] is True and (
            type(record["first_divergence"]) is not int or record["first_divergence"] < 0
        ):
            return None
    unavailable = value.get("unavailable_candidates")
    if (
        type(unavailable) is not list
        or unavailable != sorted(unavailable)
        or any(type(n) is not str for n in unavailable)
        or set(unavailable) & record_candidates
        or set(unavailable) | record_candidates != set(candidates)
    ):
        return None
    ordered = sorted(records, key=lambda r: (str(r["candidate"]), str(r["fixture"])))
    blocks: dict[str, Any] = {}
    flips_all: list = []
    total_paired = 0
    total_cells = {"pass_pass": 0, "pass_fail": 0, "fail_pass": 0, "fail_fail": 0}
    total_diverged = 0
    for name in candidates:
        stored = blocks_in[name]
        if type(stored) is not dict:
            return None
        own = [r for r in ordered if r["candidate"] == name]
        for arm in ("fp32", "bf16"):
            passed = stored.get(f"{arm}_passed")
            if passed not in (True, False, None):
                return None
            arm_fixtures = {str(r["fixture"]) for r in own if type(r[arm]) is dict}
            complete = arm_fixtures == universe
            all_pass = complete and all(r[arm]["verdict"] for r in own if type(r[arm]) is dict)
            if passed is True and not all_pass:
                return None
            if passed is False and not (complete and not all_pass):
                return None
            if passed is None and complete:
                return None
        paired = [r for r in own if type(r["fp32"]) is dict and type(r["bf16"]) is dict]
        cells = {"pass_pass": 0, "pass_fail": 0, "fail_pass": 0, "fail_fail": 0}
        flips: list = []
        n_diverged = 0
        deltas: list[float] = []
        for record in paired:
            fp32_v = bool(record["fp32"]["verdict"])
            bf16_v = bool(record["bf16"]["verdict"])
            if fp32_v and bf16_v:
                cells["pass_pass"] += 1
            elif fp32_v:
                cells["pass_fail"] += 1
                flips.append({"fixture": record["fixture"], "direction": "pass_to_fail"})
            elif bf16_v:
                cells["fail_pass"] += 1
                flips.append({"fixture": record["fixture"], "direction": "fail_to_pass"})
            else:
                cells["fail_fail"] += 1
            if record["diverged"] is True:
                n_diverged += 1
            fp32_lp = record["fp32"]["logprob"]
            bf16_lp = record["bf16"]["logprob"]
            if type(fp32_lp) is float and type(bf16_lp) is float:
                deltas.append(bf16_lp - fp32_lp)
        blocks[name] = {
            "fp32_passed": stored.get("fp32_passed"),
            "bf16_passed": stored.get("bf16_passed"),
            "n_paired": len(paired),
            "cells": cells,
            "flips": flips,
            "n_fp32_only": sum(1 for r in own if type(r["fp32"]) is dict and r["bf16"] is None),
            "n_bf16_only": sum(1 for r in own if type(r["bf16"]) is dict and r["fp32"] is None),
            "n_diverged": n_diverged,
            "divergence_rate": (n_diverged / len(paired)) if paired else None,
            "logprob": {
                "n": len(deltas),
                "mean_delta": (sum(deltas) / len(deltas)) if deltas else None,
                "min_delta": min(deltas) if deltas else None,
                "max_delta": max(deltas) if deltas else None,
            },
        }
        for flip in flips:
            flips_all.append(
                {"candidate": name, "fixture": flip["fixture"], "direction": flip["direction"]}
            )
        total_paired += len(paired)
        total_diverged += n_diverged
        for key in total_cells:
            total_cells[key] += cells[key]
    flips_all.sort(key=lambda f: (str(f["candidate"]), str(f["fixture"])))
    unpaired: list = []
    for record in ordered:
        if type(record["fp32"]) is dict and record["bf16"] is None:
            unpaired.append(
                {"candidate": record["candidate"], "fixture": record["fixture"], "arm": "fp32"}
            )
        elif type(record["bf16"]) is dict and record["fp32"] is None:
            unpaired.append(
                {"candidate": record["candidate"], "fixture": record["fixture"], "arm": "bf16"}
            )
    totals = {
        "n_paired": total_paired,
        "cells": total_cells,
        "n_flips": len(flips_all),
        "n_diverged": total_diverged,
        "divergence_rate": (total_diverged / total_paired) if total_paired else None,
    }
    falsified = None if total_paired == 0 else len(flips_all) == 0
    shared: frozenset | None = None
    mismatched = False
    for name in candidates:
        fp32_done = frozenset(
            str(r["fixture"]) for r in records if r["candidate"] == name and type(r["fp32"]) is dict
        )
        bf16_done = frozenset(
            str(r["fixture"]) for r in records if r["candidate"] == name and type(r["bf16"]) is dict
        )
        if fp32_done != bf16_done:
            mismatched = True
        pair = fp32_done & bf16_done
        shared = pair if shared is None else (shared & pair)
    if shared is None:
        shared = frozenset()

    def _select(arm: str) -> dict[str, Any] | None:
        """Select."""
        completed_sets: list = []
        all_pass: dict[str, bool] = {}
        for cand in candidates:
            cells_by_fixture = {
                str(r["fixture"]): r[arm]
                for r in records
                if r["candidate"] == cand and type(r[arm]) is dict
            }
            completed = frozenset(f for f in cells_by_fixture if f in shared)
            completed_sets.append(completed)
            if not completed:
                return None
            all_pass[cand] = all(
                _prec.cell_verdict(
                    cells_by_fixture[f],
                    str(
                        next(
                            r["expected_digest"]
                            for r in records
                            if r["candidate"] == cand and r["fixture"] == f
                        )
                    ),
                )
                for f in completed
            )
        if len(set(completed_sets)) > 1:
            return None
        passers = [n for n, ok in all_pass.items() if ok]
        if len(passers) == 1:
            return {"decision": "selected", "winner": passers[0]}
        if len(passers) > 1:
            return {"decision": "ambiguous", "winner": None}
        return {"decision": "no_candidate", "winner": None}

    if decision not in ("compatible_on_fixture_set", "held_out_failed"):
        selection = {
            "paired_population": sorted(shared),
            "fp32": None,
            "bf16": None,
            "survived": None,
            "withheld": "primary_not_winner_bearing",
        }
    elif mismatched:
        selection = {
            "paired_population": sorted(shared),
            "fp32": None,
            "bf16": None,
            "survived": None,
            "withheld": "mismatched_sets",
        }
    elif not shared:
        selection = {
            "paired_population": [],
            "fp32": None,
            "bf16": None,
            "survived": None,
            "withheld": "no_paired_fixtures",
        }
    else:
        fp32_sel = _select("fp32")
        bf16_sel = _select("bf16")
        if fp32_sel is None or bf16_sel is None:
            selection = {
                "paired_population": sorted(shared),
                "fp32": fp32_sel,
                "bf16": bf16_sel,
                "survived": None,
                "withheld": "mismatched_sets",
            }
        else:
            selection = {
                "paired_population": sorted(shared),
                "fp32": fp32_sel,
                "bf16": bf16_sel,
                "survived": fp32_sel == bf16_sel,
                "withheld": None,
            }
    return {
        "candidates": blocks,
        "unpaired": unpaired,
        "flips": flips_all,
        "totals": totals,
        "selection": selection,
        "precision_effect_falsified": falsified,
    }


def _o_precision_valid(value: Any) -> bool:
    """Oracle precision validator: mirrors diffreports._valid_precision."""
    if type(value) is not dict or value.get("available") is not True:
        return False
    if set(value) != {
        "available",
        "candidates",
        "detail",
        "flips",
        "precision_effect_falsified",
        "primary_decision",
        "primary_winner",
        "protocol",
        "records",
        "selection",
        "totals",
        "unavailable_candidates",
        "unpaired",
    }:
        return False
    if value.get("detail") is not None:
        return False
    recomputed = _o_precision_recompute(value)
    if recomputed is None:
        return False
    for key in (
        "candidates",
        "unpaired",
        "flips",
        "totals",
        "selection",
        "precision_effect_falsified",
    ):
        if value.get(key) != recomputed[key]:
            return False
    return True


def _o_precision(old: Any, new: Any) -> tuple[dict[str, Any], bool, bool]:
    """Oracle precision section: mirrors diffreports.diff_precision."""
    old_ok = _o_precision_valid(old)
    new_ok = _o_precision_valid(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    diff: dict[str, Any] = {"available": available, "changed": False}
    asymmetric = available not in ("both", "neither")
    if available != "both":
        return diff, False, asymmetric
    protocol_drift = old["protocol"].get("fingerprint") != new["protocol"].get("fingerprint")
    backend_drift = old["protocol"].get("fp32_fingerprint") != new["protocol"].get(
        "fp32_fingerprint"
    ) or old["protocol"].get("bf16_fingerprint") != new["protocol"].get("bf16_fingerprint")
    comparable = not protocol_drift and not backend_drift
    old_candidates = old["candidates"]
    new_candidates = new["candidates"]
    added = sorted(set(new_candidates) - set(old_candidates))
    removed = sorted(set(old_candidates) - set(new_candidates))
    candidate_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_candidates) & set(new_candidates)):
        old_entry = old_candidates[name]
        new_entry = new_candidates[name]
        metrics = [
            ("n_paired", old_entry["n_paired"], new_entry["n_paired"]),
            ("n_flips", len(old_entry["flips"]), len(new_entry["flips"])),
            ("n_diverged", old_entry["n_diverged"], new_entry["n_diverged"]),
            (
                "mean_logprob_delta",
                old_entry["logprob"]["mean_delta"],
                new_entry["logprob"]["mean_delta"],
            ),
            ("fp32_passed", old_entry["fp32_passed"], new_entry["fp32_passed"]),
            ("bf16_passed", old_entry["bf16_passed"], new_entry["bf16_passed"]),
        ]
        for metric, old_value, new_value in metrics:
            if old_value != new_value:
                candidate_drift.append(
                    {"candidate": name, "metric": metric, "old": old_value, "new": new_value}
                )
    selection_drift: list[dict[str, Any]] = []
    if old["selection"] != new["selection"]:
        selection_drift.append({"old": old["selection"], "new": new["selection"]})
    if old["precision_effect_falsified"] != new["precision_effect_falsified"]:
        selection_drift.append(
            {
                "metric": "precision_effect_falsified",
                "old": old["precision_effect_falsified"],
                "new": new["precision_effect_falsified"],
            }
        )
    diff["candidate_drift"] = candidate_drift
    diff["selection_drift"] = selection_drift
    diff["candidates_added"] = added
    diff["candidates_removed"] = removed
    diff["protocol_drift"] = protocol_drift
    diff["backend_drift"] = backend_drift
    diff["comparable"] = comparable
    changed = bool(
        candidate_drift or selection_drift or added or removed or protocol_drift or backend_drift
    )
    diff["changed"] = changed
    return diff, changed, asymmetric


_O_CONFIDENCE_GRAMMAR = re.compile(r"^Confidence: (0|[1-9][0-9]?|100)$")


def _o_confidence_recompute(value: Any) -> dict[str, Any] | None:
    """Oracle confidence recompute: mirrors
    diffreports._confidence_recompute. The protocol digest and
    the generated case set are rebuilt through the producer
    module (confidence) where the implementation rebuilds them
    from literal tables and its own mirror; every record and
    cell is checked structurally, the carried-line binding
    re-derives the confidence scalar, screens are checked for
    protocol/record consistency, and the derived aggregation
    (compliance, joint tables, gap, bins, flips, knowability)
    is mirrored independently here.
    """
    from model_protocol_lab import confidence as _cf

    def _dig(v: Any) -> bool:
        """Dig."""
        return type(v) is str and len(v) == 64 and all(c in "0123456789abcdef" for c in v)

    def _cell_ok(cell: Any, expected_digest: str) -> bool:
        """Cell ok."""
        if type(cell) is not dict or set(cell) != {
            "answer_digest",
            "boundary_found",
            "code",
            "confidence",
            "final_line",
            "final_line_length",
            "input_tokens",
            "span_digest",
            "verdict",
        }:
            return False
        if not _dig(cell["answer_digest"]) or not _dig(cell["span_digest"]):
            return False
        if type(cell["boundary_found"]) is not bool or type(cell["verdict"]) is not bool:
            return False
        if cell["verdict"] != (cell["span_digest"] == expected_digest and cell["boundary_found"]):
            return False
        if type(cell["input_tokens"]) is not int or cell["input_tokens"] < 1:
            return False
        code = cell["code"]
        if code not in {
            "ok",
            "missing_line",
            "malformed_line",
            "multiple_lines",
            "confidence_not_final",
        }:
            return False
        conf = cell["confidence"]
        line = cell["final_line"]
        length = cell["final_line_length"]
        if code == "ok":
            if cell["span_digest"] == cell["answer_digest"]:
                return False
            if type(conf) is not int or not 0 <= conf <= 100:
                return False
            if type(line) is not str:
                return False
            m = _O_CONFIDENCE_GRAMMAR.match(line)
            if m is None or int(m.group(1)) != conf:
                return False
            return length == len(line)
        if cell["span_digest"] != cell["answer_digest"]:
            return False
        if code == "malformed_line":
            if conf is not None or type(line) is not str or not 1 <= len(line) <= 32:
                return False
            if type(length) is not int or length < len(line) or length < 11:
                return False
            if length > len(line) and len(line) != 32:
                return False
            if not (line.startswith("Confidence:") or line.lower().startswith("confidence:")):
                return False
            return not (length == len(line) and _O_CONFIDENCE_GRAMMAR.match(line))
        return conf is None and line is None and length is None

    def _block(cells: list, n_population: int) -> dict[str, Any]:
        """Block."""
        compliant = [c for c in cells if c["code"] == "ok"]
        correct = [c for c in compliant if c["verdict"]]
        incorrect = [c for c in compliant if not c["verdict"]]

        def _mean(xs: list) -> float | None:
            """Mean."""
            return (sum(xs) / len(xs)) if xs else None

        mc = _mean([int(c["confidence"]) for c in correct])
        mi = _mean([int(c["confidence"]) for c in incorrect])
        if not compliant:
            gap, detail = None, "no_compliant_cells"
        elif not correct:
            gap, detail = None, "no_correct_compliant"
        elif not incorrect:
            gap, detail = None, "no_incorrect_compliant"
        else:
            gap, detail = float(mc) - float(mi), ""
        bins = []
        for lo, hi in ((0, 24), (25, 49), (50, 74), (75, 100)):
            members = [c for c in compliant if lo <= int(c["confidence"]) <= hi]
            nc = sum(1 for c in members if c["verdict"])
            bins.append(
                {
                    "lo": lo,
                    "hi": hi,
                    "n": len(members),
                    "n_correct": nc,
                    "accuracy": (nc / len(members)) if members else None,
                }
            )
        return {
            "n_population": n_population,
            "n_completed": len(cells),
            "n_compliant": len(compliant),
            "compliance_rate": (len(compliant) / len(cells)) if cells else None,
            "mean_confidence": _mean([int(c["confidence"]) for c in compliant]),
            "mean_correct": mc,
            "mean_incorrect": mi,
            "gap": gap,
            "gap_detail": detail,
            "joint": {
                "compliant_correct": len(correct),
                "compliant_incorrect": len(incorrect),
                "noncompliant": len(cells) - len(compliant),
            },
            "bins": bins,
        }

    protocol = value.get("protocol")
    if type(protocol) is not dict or set(protocol) != {
        "backend_fingerprint",
        "bin_edges",
        "boundaries",
        "candidates",
        "case_count",
        "control_cases",
        "fingerprint",
        "fixture_set_sha256",
        "max_control_records",
        "max_new_tokens",
        "max_records",
        "max_records_bytes",
        "min_paired_compliant",
        "screens",
        "seed",
        "suffix_sha256",
        "version",
    }:
        return None
    if protocol["version"] != 1 or type(protocol["version"]) is not int:
        return None
    if type(protocol["seed"]) is not int or protocol["seed"] < 0:
        return None
    if type(protocol["max_new_tokens"]) is not int or not 1 <= protocol["max_new_tokens"] <= 4096:
        return None
    if protocol["max_records"] != 2400 or protocol["max_control_records"] != 576:
        return None
    if protocol["max_records_bytes"] != 12 * 1024 * 1024:
        return None
    if protocol["case_count"] != 24 or protocol["min_paired_compliant"] != 6:
        return None
    if protocol["bin_edges"] != [[0, 24], [25, 49], [50, 74], [75, 100]]:
        return None
    if not _dig(protocol["suffix_sha256"]) or not _dig(protocol["fixture_set_sha256"]):
        return None
    candidates = protocol["candidates"]
    if (
        type(candidates) is not list
        or not candidates
        or any(type(n) is not str or not n for n in candidates)
        or len(set(candidates)) != len(candidates)
    ):
        return None
    boundaries = protocol["boundaries"]
    if type(boundaries) is not dict or set(boundaries) != set(candidates):
        return None
    for name in candidates:
        ids = boundaries[name]
        if type(ids) is not list or not ids or any(type(t) is not int or t < 0 for t in ids):
            return None
    if type(protocol["backend_fingerprint"]) is not str or not protocol["backend_fingerprint"]:
        return None
    screens = protocol["screens"]
    if type(screens) is not dict:
        return None
    for fid, screen in screens.items():
        if type(fid) is not str or screen not in {
            "expected_blank",
            "expected_confidence_shaped",
            "vacuous_last_user",
            "no_user_turn",
        }:
            return None
    if protocol["control_cases"] != _cf.case_protocol(protocol["seed"]):
        return None
    if protocol["fingerprint"] != _cf.protocol_fingerprint(protocol):
        return None
    decision = value.get("primary_decision")
    winner = value.get("primary_winner")
    if decision not in {
        "compatible_on_fixture_set",
        "held_out_failed",
        "ambiguous",
        "no_candidate",
        "cancelled",
        "failed",
    }:
        return None
    blocks_in = value.get("candidates")
    if type(blocks_in) is not dict or set(blocks_in) != set(candidates):
        return None
    if decision in ("compatible_on_fixture_set", "held_out_failed"):
        if type(winner) is not str or winner not in blocks_in:
            return None
    elif winner is not None:
        return None
    records = value.get("records")
    control_records = value.get("control_records")
    if type(records) is not list or len(records) > protocol["max_records"]:
        return None
    if type(control_records) is not list or len(control_records) > protocol["max_control_records"]:
        return None
    blob = json.dumps(
        {"records": records, "control_records": control_records},
        ensure_ascii=True,
        separators=(",", ":"),
    )
    if len(blob.encode("utf-8")) > protocol["max_records_bytes"]:
        return None
    case_expected = {e["id"]: e["expected_digest"] for e in protocol["control_cases"]}
    seen: set = set()
    universe: set = set()
    per_candidate: dict[str, set] = {name: set() for name in candidates}
    zero_cell = dict.fromkeys(candidates, True)
    for record in records:
        if type(record) is not dict or set(record) != {
            "candidate",
            "elicited",
            "expected_digest",
            "fixture",
            "native_verdict",
            "screen",
        }:
            return None
        name = record["candidate"]
        fixture = record["fixture"]
        if type(name) is not str or name not in candidates or type(fixture) is not str:
            return None
        if (name, fixture) in seen:
            return None
        seen.add((name, fixture))
        universe.add(fixture)
        per_candidate[name].add(fixture)
        if record["screen"] != screens.get(fixture, ""):
            return None
        if not _dig(record["expected_digest"]):
            return None
        if record["native_verdict"] not in (True, False, None):
            return None
        cell = record["elicited"]
        if record["screen"] != "" and cell is not None:
            return None
        if cell is not None:
            if not _cell_ok(cell, str(record["expected_digest"])):
                return None
            zero_cell[name] = False
    if any(per_candidate[name] != universe for name in candidates):
        return None
    seen_control: set = set()
    control_per_candidate: dict[str, set] = {name: set() for name in candidates}
    for record in control_records:
        if type(record) is not dict or set(record) != {
            "candidate",
            "case",
            "control",
            "expected_digest",
            "knowable",
        }:
            return None
        name = record["candidate"]
        case_id = record["case"]
        if type(name) is not str or name not in candidates or case_id not in case_expected:
            return None
        if (name, case_id) in seen_control:
            return None
        seen_control.add((name, case_id))
        control_per_candidate[name].add(case_id)
        if record["expected_digest"] != case_expected[case_id]:
            return None
        for arm in ("knowable", "control"):
            cell = record[arm]
            if cell is not None:
                if not _cell_ok(cell, str(record["expected_digest"])):
                    return None
                zero_cell[name] = False
    if any(control_per_candidate[name] != set(case_expected) for name in candidates):
        return None
    if value.get("unavailable_candidates") != sorted(n for n in candidates if zero_cell[n]):
        return None
    n_population = sum(1 for f in universe if screens.get(f, "") == "")
    blocks: dict[str, Any] = {}
    for name in candidates:
        own = [r for r in records if r["candidate"] == name]
        elicited = [r["elicited"] for r in own if type(r["elicited"]) is dict]
        operator = _block(elicited, n_population)
        flips = {"n_compared": 0, "pass_to_fail": 0, "fail_to_pass": 0}
        for record in own:
            cell = record["elicited"]
            native = record["native_verdict"]
            if type(cell) is not dict or type(native) is not bool:
                continue
            flips["n_compared"] += 1
            if native and not cell["verdict"]:
                flips["pass_to_fail"] += 1
            elif not native and cell["verdict"]:
                flips["fail_to_pass"] += 1
        operator["flips"] = flips
        own_control = [r for r in control_records if r["candidate"] == name]
        knowable = _block([r["knowable"] for r in own_control if type(r["knowable"]) is dict], 24)
        control_block = _block(
            [r["control"] for r in own_control if type(r["control"]) is dict], 24
        )
        paired = [
            r
            for r in own_control
            if type(r["knowable"]) is dict
            and type(r["control"]) is dict
            and r["knowable"]["code"] == "ok"
            and r["control"]["code"] == "ok"
        ]
        if len(paired) >= 6:
            km = sum(int(r["knowable"]["confidence"]) for r in paired) / len(paired)
            cm = sum(int(r["control"]["confidence"]) for r in paired) / len(paired)
            knowability = {
                "n_paired_compliant": len(paired),
                "knowable_mean": km,
                "control_mean": cm,
                "difference": km - cm,
                "falsified": cm >= km,
                "detail": "",
            }
        else:
            knowability = {
                "n_paired_compliant": len(paired),
                "knowable_mean": None,
                "control_mean": None,
                "difference": None,
                "falsified": None,
                "detail": "insufficient_paired_compliant",
            }
        blocks[name] = {
            "operator": operator,
            "generated": {
                "knowable": knowable,
                "control": control_block,
                "knowability": knowability,
            },
        }
    return {
        "candidates": blocks,
        "totals": {"n_records": len(records), "n_control_records": len(control_records)},
    }


def _o_confidence_valid(value: Any) -> bool:
    """Oracle confidence validator: mirrors diffreports._valid_confidence."""
    if type(value) is not dict or value.get("available") is not True:
        return False
    if set(value) != {
        "available",
        "candidates",
        "control_records",
        "detail",
        "primary_decision",
        "primary_winner",
        "protocol",
        "records",
        "totals",
        "unavailable_candidates",
    }:
        return False
    if value.get("detail") is not None:
        return False
    recomputed = _o_confidence_recompute(value)
    if recomputed is None:
        return False
    return all(value.get(key) == recomputed[key] for key in ("candidates", "totals"))


def _o_confidence(old: Any, new: Any) -> tuple[dict[str, Any], bool, bool]:
    """Oracle confidence section: mirrors diffreports.diff_confidence."""
    old_ok = _o_confidence_valid(old)
    new_ok = _o_confidence_valid(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    diff: dict[str, Any] = {"available": available, "changed": False}
    asymmetric = available not in ("both", "neither")
    if available != "both":
        return diff, False, asymmetric
    protocol_drift = old["protocol"].get("fingerprint") != new["protocol"].get("fingerprint")
    backend_drift = old["protocol"].get("backend_fingerprint") != new["protocol"].get(
        "backend_fingerprint"
    )
    comparable = not protocol_drift and not backend_drift
    old_candidates = old["candidates"]
    new_candidates = new["candidates"]
    added = sorted(set(new_candidates) - set(old_candidates))
    removed = sorted(set(old_candidates) - set(new_candidates))
    candidate_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_candidates) & set(new_candidates)):
        old_entry = old_candidates[name]
        new_entry = new_candidates[name]
        metrics = [
            (
                "operator_compliance_rate",
                old_entry["operator"]["compliance_rate"],
                new_entry["operator"]["compliance_rate"],
            ),
            ("operator_gap", old_entry["operator"]["gap"], new_entry["operator"]["gap"]),
            (
                "operator_flips",
                old_entry["operator"]["flips"]["pass_to_fail"]
                + old_entry["operator"]["flips"]["fail_to_pass"],
                new_entry["operator"]["flips"]["pass_to_fail"]
                + new_entry["operator"]["flips"]["fail_to_pass"],
            ),
            (
                "knowable_mean_confidence",
                old_entry["generated"]["knowable"]["mean_confidence"],
                new_entry["generated"]["knowable"]["mean_confidence"],
            ),
            (
                "control_mean_confidence",
                old_entry["generated"]["control"]["mean_confidence"],
                new_entry["generated"]["control"]["mean_confidence"],
            ),
            (
                "knowability_falsified",
                old_entry["generated"]["knowability"]["falsified"],
                new_entry["generated"]["knowability"]["falsified"],
            ),
        ]
        for metric, old_value, new_value in metrics:
            if old_value != new_value:
                candidate_drift.append(
                    {"candidate": name, "metric": metric, "old": old_value, "new": new_value}
                )
    diff["candidate_drift"] = candidate_drift
    diff["candidates_added"] = added
    diff["candidates_removed"] = removed
    diff["protocol_drift"] = protocol_drift
    diff["backend_drift"] = backend_drift
    diff["comparable"] = comparable
    changed = bool(candidate_drift or added or removed or protocol_drift or backend_drift)
    diff["changed"] = changed
    return diff, changed, asymmetric


def _o_cache_profile_valid(value: Any) -> bool:
    """Oracle cache-profile validator: mirrors diffreports._valid_cache_profile."""
    if type(value) is not dict or value.get("available") is not True:
        return False
    block_size = value.get("block_size")
    if type(block_size) is not int or block_size < 1:
        return False
    n_observations = value.get("n_observations")
    if type(n_observations) is not int or n_observations < 0:
        return False
    per_candidate = value.get("per_candidate")
    if type(per_candidate) is not dict or not all(type(k) is str for k in per_candidate):
        return False
    for entry in per_candidate.values():
        if entry is None:
            continue
        if type(entry) is not dict:
            return False
        for key in ("n", "n_evaluable"):
            count = entry.get(key)
            if type(count) is not int or count < 0:
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
            count = entry.get(key)
            if type(count) is not int or count < 0:
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
            share = entry.get(key)
            if entry["n_evaluable"] == 0:
                if share is not None:
                    return False
            elif not (type(share) in (int, float) and 0.0 <= share <= 1.0):
                return False
        mean = entry.get("mean_prompt_tokens")
        if entry["n_evaluable"] == 0:
            if mean is not None:
                return False
        elif not (type(mean) in (int, float) and mean > 0):
            return False
        if entry["n_evaluable"] > 0:
            # Recompute mirror (cycle 177, F6b).
            total = entry["total_tokens"]
            if total <= 0:
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
        if (
            entry["n_evaluable"] > 0
            and type(achievable_token) in (int, float)
            and type(potential_token) in (int, float)
            and type(achievable_block) in (int, float)
            and type(potential_block) in (int, float)
            and not (
                achievable_block <= achievable_token <= potential_token
                and achievable_block <= potential_block <= potential_token
            )
        ):
            return False
    for key in ("most_sharable", "least_sharable", "excluded_candidates"):
        value_list = value.get(key)
        if type(value_list) is not list or not all(type(c) is str for c in value_list):
            return False
    gap = value.get("sharability_gap")
    if gap is not None and not (type(gap) in (int, float) and gap >= 0):
        return False
    if value.get("uniform") is not None and type(value.get("uniform")) is not bool:
        return False
    if type(value.get("evaluable_mismatch")) is not bool:
        return False
    # Profile-level recomputation mirror (cycle 177, F6b).
    evaluable = {
        name: float(entry["achievable_block_fraction"])
        for name, entry in per_candidate.items()
        if entry is not None and entry["n_evaluable"] > 0
    }
    excluded = sorted(
        name for name, entry in per_candidate.items() if entry is None or entry["n_evaluable"] == 0
    )
    if value.get("excluded_candidates") != excluded:
        return False
    counts = {
        entry["n_evaluable"]
        for entry in per_candidate.values()
        if entry is not None and entry["n_evaluable"] > 0
    }
    if value.get("evaluable_mismatch") != (len(counts) > 1):
        return False
    if not evaluable:
        return (
            gap is None
            and value.get("uniform") is None
            and not value["most_sharable"]
            and not value["least_sharable"]
        )
    values = list(evaluable.values())
    if gap != max(values) - min(values):
        return False
    top = max(values)
    bottom = min(values)
    if value["most_sharable"] != sorted(n for n, v in evaluable.items() if v == top):
        return False
    if value["least_sharable"] != sorted(n for n, v in evaluable.items() if v == bottom):
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
    return value.get("uniform") == expected_uniform


def _o_cache_profile(old: Any, new: Any) -> tuple[dict[str, Any], bool, bool]:
    """Oracle cache-profile section: block/token/potential drift.

    Independent reimplementation: comprehension-built views rather
    than the implementation's accumulation loops, with the drift
    buckets filled in the implementation's per-candidate field order.
    """
    old_ok = _o_cache_profile_valid(old)
    new_ok = _o_cache_profile_valid(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    diff: dict[str, Any] = {"available": available, "changed": False}
    asymmetric = available not in ("both", "neither")
    if available != "both":
        return diff, False, asymmetric

    def _view(payload: Any) -> dict[str, dict[str, Any]]:
        # Mirrors the implementation: a None cell (candidate missing
        # from the stage) counts as excluded, exactly like the
        # profile's excluded_candidates semantics.
        """View."""
        out: dict[str, dict[str, Any]] = {}
        for name, entry in (payload.get("per_candidate") or {}).items():
            if entry is None:
                out[name] = {
                    "achievable_block_fraction": None,
                    "achievable_token_fraction": None,
                    "potential_block_fraction": None,
                    "excluded": True,
                }
            else:
                out[name] = {
                    "achievable_block_fraction": entry.get("achievable_block_fraction"),
                    "achievable_token_fraction": entry.get("achievable_token_fraction"),
                    "potential_block_fraction": entry.get("potential_block_fraction"),
                    "excluded": name in set(payload.get("excluded_candidates") or []),
                }
        return out

    old_view = _view(old)
    new_view = _view(new)
    candidates_added = sorted(set(new_view) - set(old_view))
    candidates_removed = sorted(set(old_view) - set(new_view))
    # Same per-candidate field order as the implementation's loop.
    block_drift: list[dict[str, Any]] = []
    token_drift: list[dict[str, Any]] = []
    potential_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_view) & set(new_view)):
        for field, bucket in (
            ("achievable_block_fraction", block_drift),
            ("achievable_token_fraction", token_drift),
            ("potential_block_fraction", potential_drift),
        ):
            if old_view[name][field] != new_view[name][field]:
                bucket.append(
                    {
                        "candidate": name,
                        "old": old_view[name][field],
                        "new": new_view[name][field],
                    }
                )
    old_excluded = {name for name, v in old_view.items() if v["excluded"]}
    new_excluded = {name for name, v in new_view.items() if v["excluded"]}
    excluded_added = sorted(new_excluded - old_excluded)
    excluded_removed = sorted(old_excluded - new_excluded)
    diff["block_drift"] = block_drift
    diff["token_drift"] = token_drift
    diff["potential_drift"] = potential_drift
    diff["candidates_added"] = candidates_added
    diff["candidates_removed"] = candidates_removed
    diff["excluded_added"] = excluded_added
    diff["excluded_removed"] = excluded_removed
    changed = bool(
        block_drift
        or token_drift
        or potential_drift
        or candidates_added
        or candidates_removed
        or excluded_added
        or excluded_removed
    )
    diff["changed"] = changed
    return diff, changed, asymmetric


def _o_prefill_valid(value: Any) -> bool:
    """Oracle-side availability check, mirroring the implementation's rule."""
    if type(value) is not dict or value.get("available") is not True:
        return False
    prefills = value.get("prefills")
    if type(prefills) is not list or not prefills or not all(type(p) is str for p in prefills):
        return False
    per_condition = value.get("per_condition")
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
    think_effect = value.get("think_effect")
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
    return type(value.get("flips")) is list


def _o_prefill(old: Any, new: Any) -> tuple[dict[str, Any], bool, bool]:
    """Oracle prefill section: condition triplets, flips, rates, think groups."""
    old_ok = _o_prefill_valid(old)
    new_ok = _o_prefill_valid(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    diff: dict[str, Any] = {"available": available, "changed": False}
    asymmetric = available not in ("both", "neither")
    if available != "both":
        return diff, False, asymmetric
    old_prefills = old.get("prefills") or []
    new_prefills = new.get("prefills") or []
    prefills_added = [p for p in new_prefills if p not in old_prefills]
    prefills_removed = [p for p in old_prefills if p not in new_prefills]
    condition_changed = {}
    for prefill_id in sorted(set(old_prefills) & set(new_prefills)):
        old_cond = (old.get("per_condition") or {}).get(prefill_id) or {}
        new_cond = (new.get("per_condition") or {}).get(prefill_id) or {}
        old_triplet = (old_cond.get("decision"), old_cond.get("winner"))
        new_triplet = (new_cond.get("decision"), new_cond.get("winner"))
        condition_changed[prefill_id] = {
            "old_decision": old_triplet[0],
            "old_winner": old_triplet[1],
            "new_decision": new_triplet[0],
            "new_winner": new_triplet[1],
            "changed": old_triplet != new_triplet,
        }

    def _flip_key(flips: Any) -> list[tuple[Any, ...]]:
        """Flip key."""
        if type(flips) is not list:
            return []
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

    flips_changed = _flip_key(old.get("flips")) != _flip_key(new.get("flips"))

    def _candidate_rates(payload: Any) -> dict[str, dict[str, tuple[Any, Any]]]:
        """Candidate rates."""
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
    candidate_drift = []
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
        """Think means."""
        effect = payload.get("think_effect") or {}
        return {
            label: (group or {}).get("mean_honor_rate")
            for label, group in effect.items()
            if type(label) is str and label != "delta"
        }

    old_think = _think_means(old)
    new_think = _think_means(new)
    think_changed = old_think != new_think
    diff["prefills_added"] = prefills_added
    diff["prefills_removed"] = prefills_removed
    diff["condition_changed"] = condition_changed
    diff["flips_changed"] = flips_changed
    diff["candidate_drift"] = candidate_drift
    diff["candidates_added"] = candidates_added
    diff["candidates_removed"] = candidates_removed
    diff["think_rates"] = {
        "old": old_think,
        "new": new_think,
        "changed": think_changed,
    }
    changed = bool(
        prefills_added
        or prefills_removed
        or any(entry["changed"] for entry in condition_changed.values())
        or flips_changed
        or candidate_drift
        or candidates_added
        or candidates_removed
        or think_changed
    )
    diff["changed"] = changed
    return diff, changed, asymmetric


def _o_portfolio_valid(value: Any) -> bool:
    """Oracle validator mirroring diffreports._valid_portfolio."""
    if type(value) is not dict or value.get("available") is not True:
        return False
    if type(value.get("n_fixtures")) is not int or value["n_fixtures"] <= 0:
        return False
    oracle = value.get("oracle")
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
    cover = value.get("portfolio")
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


def _o_portfolio(old: Any, new: Any) -> tuple[dict[str, Any], bool, bool]:
    """Oracle portfolio section: gap/rates, solvable set, membership, drift."""
    old_ok = _o_portfolio_valid(old)
    new_ok = _o_portfolio_valid(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    diff: dict[str, Any] = {"available": available, "changed": False}
    asymmetric = available not in ("both", "neither")
    if available != "both":
        return diff, False, asymmetric

    def _triplet(payload: Any) -> tuple[Any, Any, Any]:
        """Triplet."""
        oracle = payload.get("oracle") or {}
        return (
            oracle.get("gap"),
            oracle.get("oracle_pass_rate"),
            oracle.get("winner_pass_rate"),
        )

    old_triplet, new_triplet = _triplet(old), _triplet(new)
    rates_shifted = old_triplet != new_triplet
    diff["rates"] = {
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
    old_solvable = set((old.get("oracle") or {}).get("solvable_fixture_ids") or [])
    new_solvable = set((new.get("oracle") or {}).get("solvable_fixture_ids") or [])
    solvable_added = sorted(new_solvable - old_solvable)
    solvable_removed = sorted(old_solvable - new_solvable)
    diff["solvable_added"] = solvable_added
    diff["solvable_removed"] = solvable_removed
    old_members = list((old.get("portfolio") or {}).get("members") or [])
    new_members = list((new.get("portfolio") or {}).get("members") or [])
    members_added = [m for m in new_members if m not in old_members]
    members_removed = [m for m in old_members if m not in new_members]
    diff["members_added"] = members_added
    diff["members_removed"] = members_removed
    old_winners = {
        str(k): list(v)
        for k, v in ((old.get("oracle") or {}).get("per_fixture_winners") or {}).items()
    }
    new_winners = {
        str(k): list(v)
        for k, v in ((new.get("oracle") or {}).get("per_fixture_winners") or {}).items()
    }
    fixture_winner_drift = [
        {"fixture": fid, "old_winners": old_winners[fid], "new_winners": new_winners[fid]}
        for fid in sorted(set(old_winners) & set(new_winners))
        if old_winners[fid] != new_winners[fid]
    ]
    diff["fixture_winner_drift"] = fixture_winner_drift
    old_covers = {
        str(k): sorted(v)
        for k, v in ((old.get("portfolio") or {}).get("per_member_covers") or {}).items()
    }
    new_covers = {
        str(k): sorted(v)
        for k, v in ((new.get("portfolio") or {}).get("per_member_covers") or {}).items()
    }
    cover_drift = [
        {"member": m, "old_covers": old_covers[m], "new_covers": new_covers[m]}
        for m in sorted(set(old_covers) & set(new_covers))
        if old_covers[m] != new_covers[m]
    ]
    diff["cover_drift"] = cover_drift
    changed = bool(
        rates_shifted
        or solvable_added
        or solvable_removed
        or members_added
        or members_removed
        or fixture_winner_drift
        or cover_drift
    )
    diff["changed"] = changed
    return diff, changed, asymmetric


def _o_synthhistory_valid(value: Any) -> bool:
    """Oracle validator mirroring diffreports._valid_synthhistory."""
    if type(value) is not dict or value.get("available") is not True:
        return False
    if type(value.get("primary_decision")) is not str:
        return False
    winner = value.get("primary_winner")
    if winner is not None and type(winner) is not str:
        return False
    if type(value.get("n_fixtures")) is not int or value["n_fixtures"] <= 0:
        return False
    conditions = value.get("conditions")
    if (
        type(conditions) is not list
        or not conditions
        or not all(type(c) is str for c in conditions)
    ):
        return False
    per_condition = value.get("per_condition")
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
    flips = value.get("flips")
    if type(flips) is not list:
        return False
    for entry in flips:
        if type(entry) is not dict:
            return False
        if type(entry.get("condition")) is not str:
            return False
        if type(entry.get("decision_from")) is not str:
            return False
        if type(entry.get("decision_to")) is not str:
            return False
        for key in ("winner_from", "winner_to"):
            if entry.get(key) is not None and type(entry.get(key)) is not str:
                return False
    groups = value.get("group_pass_rates")
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


def _o_synthhistory(old: Any, new: Any) -> tuple[dict[str, Any], bool, bool]:
    """Oracle synthhistory section: condition verdicts, flips, rate drift."""
    old_ok = _o_synthhistory_valid(old)
    new_ok = _o_synthhistory_valid(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    diff: dict[str, Any] = {"available": available, "changed": False}
    asymmetric = available not in ("both", "neither")
    if available != "both":
        return diff, False, asymmetric
    old_conditions = old.get("conditions") or []
    new_conditions = new.get("conditions") or []
    conditions_added = [c for c in new_conditions if c not in old_conditions]
    conditions_removed = [c for c in old_conditions if c not in new_conditions]
    diff["conditions_added"] = conditions_added
    diff["conditions_removed"] = conditions_removed
    condition_changed: dict[str, dict[str, Any]] = {}
    for condition in sorted(set(old_conditions) & set(new_conditions)):
        old_c = (old.get("per_condition") or {}).get(condition) or {}
        new_c = (new.get("per_condition") or {}).get(condition) or {}
        old_verdict = (old_c.get("decision"), old_c.get("winner"))
        new_verdict = (new_c.get("decision"), new_c.get("winner"))
        changed = old_verdict != new_verdict
        condition_changed[condition] = {
            "old_decision": old_verdict[0],
            "old_winner": old_verdict[1],
            "new_decision": new_verdict[0],
            "new_winner": new_verdict[1],
            "changed": changed,
        }
    diff["condition_changed"] = condition_changed

    def _flip_key(flips: Any) -> list[tuple[Any, ...]]:
        """Flip key."""
        keys = []
        for entry in flips or []:
            if type(entry) is not dict:
                continue
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

    flips_changed = _flip_key(old.get("flips")) != _flip_key(new.get("flips"))
    diff["flips_changed"] = flips_changed

    def _rates(payload: Any) -> dict[str, dict[str, Any]]:
        """Rates."""
        rates: dict[str, dict[str, Any]] = {}
        per_condition = payload.get("per_condition") or {}
        for condition in sorted(per_condition):
            candidates = (per_condition.get(condition) or {}).get("per_candidate") or {}
            for name, entry in candidates.items():
                entry = entry or {}
                rates.setdefault(name, {})[condition] = entry.get("pass_rate")
        return rates

    old_rates, new_rates = _rates(old), _rates(new)
    candidates_added = sorted(set(new_rates) - set(old_rates))
    candidates_removed = sorted(set(old_rates) - set(new_rates))
    diff["candidates_added"] = candidates_added
    diff["candidates_removed"] = candidates_removed
    pass_rate_drift = [
        {
            "candidate": name,
            "condition": condition,
            "old_pass_rate": old_rates[name][condition],
            "new_pass_rate": new_rates[name][condition],
        }
        for name in sorted(set(old_rates) & set(new_rates))
        for condition in sorted(set(old_rates[name]) & set(new_rates[name]))
        if old_rates[name][condition] != new_rates[name][condition]
    ]
    diff["pass_rate_drift"] = pass_rate_drift

    def _group_means(payload: Any) -> dict[str, Any]:
        """Group means."""
        groups = payload.get("group_pass_rates") or {}
        return {
            label: (group or {}).get("mean_pass_rate")
            for label, group in groups.items()
            if type(label) is str
        }

    old_groups, new_groups = _group_means(old), _group_means(new)
    group_changed = old_groups != new_groups
    diff["group_rates"] = {"old": old_groups, "new": new_groups, "changed": group_changed}
    changed = bool(
        conditions_added
        or conditions_removed
        or any(entry["changed"] for entry in condition_changed.values())
        or flips_changed
        or pass_rate_drift
        or candidates_added
        or candidates_removed
        or group_changed
    )
    diff["changed"] = changed
    return diff, changed, asymmetric


def _o_batchinv_valid(value: Any) -> bool:
    """Oracle validator mirroring diffreports._valid_batchinv."""
    if type(value) is not dict or value.get("available") is not True:
        return False
    for key in ("n_prompts", "n_evaluable", "n_failed", "n_diverged"):
        if type(value.get(key)) is not int or value[key] < 0:
            return False
    if value["n_evaluable"] + value["n_failed"] != value["n_prompts"]:
        return False
    if value["n_diverged"] > value["n_evaluable"]:
        return False
    rate = value.get("divergence_rate")
    if value["n_evaluable"]:
        if type(rate) not in (int, float):
            return False
    elif rate is not None:
        return False
    if value.get("verdict") not in ("batch_invariant", "batch_variant", "unavailable"):
        return False
    per_prompt = value.get("per_prompt")
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


def _o_batchinv(old: Any, new: Any) -> tuple[dict[str, Any], bool, bool]:
    """Oracle batch_invariance section: verdict, rate, diverged-set drift."""
    old_ok = _o_batchinv_valid(old)
    new_ok = _o_batchinv_valid(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    diff: dict[str, Any] = {"available": available, "changed": False}
    asymmetric = available not in ("both", "neither")
    if available != "both":
        return diff, False, asymmetric

    def _diverged_set(payload: Any) -> set[str]:
        """Diverged set."""
        per_prompt = payload.get("per_prompt") or {}
        return {
            pid
            for pid, entry in per_prompt.items()
            if type(entry) is dict and entry.get("diverged") is True
        }

    verdict_changed = old.get("verdict") != new.get("verdict")
    diff["verdict_changed"] = verdict_changed
    divergence_rate_changed = old.get("divergence_rate") != new.get("divergence_rate")
    diff["divergence_rate_changed"] = divergence_rate_changed
    old_diverged, new_diverged = _diverged_set(old), _diverged_set(new)
    diverged_added = sorted(new_diverged - old_diverged)
    diverged_removed = sorted(old_diverged - new_diverged)
    diff["diverged_added"] = diverged_added
    diff["diverged_removed"] = diverged_removed
    old_prompts = set(old.get("per_prompt") or {})
    new_prompts = set(new.get("per_prompt") or {})
    prompts_added = sorted(new_prompts - old_prompts)
    prompts_removed = sorted(old_prompts - new_prompts)
    diff["prompts_added"] = prompts_added
    diff["prompts_removed"] = prompts_removed
    changed = bool(
        verdict_changed
        or divergence_rate_changed
        or diverged_added
        or diverged_removed
        or prompts_added
        or prompts_removed
    )
    diff["changed"] = changed
    return diff, changed, asymmetric


def _o_stop_cause_valid(value: Any) -> bool:
    """Oracle stop_cause validator: mirrors diffreports._valid_stop_cause."""
    if type(value) is not dict or value.get("available") is not True:
        return False
    per_candidate = value.get("per_candidate")
    if type(per_candidate) is not dict or not all(type(k) is str for k in per_candidate):
        return False
    for entry in per_candidate.values():
        if type(entry) is not dict:
            return False
        rates = entry.get("cause_rates")
        if type(rates) is not dict:
            return False
        for cause, rate in rates.items():
            if cause not in ("eos_id", "stop_string", "max_length", "stop_unknown"):
                return False
            if type(rate) not in (int, float):
                return False
        for key in ("over_generation_rate", "would_stop_early_rate", "max_length_rate"):
            agg_value = entry.get(key)
            if agg_value is not None and type(agg_value) not in (int, float):
                return False
    overall = value.get("overall")
    return not (overall is not None and type(overall) is not dict)


def _o_stop_cause(old: Any, new: Any) -> tuple[dict[str, Any], bool, bool]:
    """Oracle stop_cause section: cause-rate and aggregate drift."""
    old_ok = _o_stop_cause_valid(old)
    new_ok = _o_stop_cause_valid(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    diff: dict[str, Any] = {"available": available, "changed": False}
    asymmetric = available not in ("both", "neither")
    if available != "both":
        return diff, False, asymmetric

    def _view(payload: Any) -> dict[str, dict[str, Any]]:
        """View."""
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

    old_view, new_view = _view(old), _view(new)
    candidates_added = sorted(set(new_view) - set(old_view))
    candidates_removed = sorted(set(old_view) - set(new_view))
    diff["candidates_added"] = candidates_added
    diff["candidates_removed"] = candidates_removed
    cause_drift: list[dict[str, Any]] = []
    aggregate_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_view) & set(new_view)):
        old_rates, new_rates = old_view[name]["cause_rates"], new_view[name]["cause_rates"]
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
    diff["cause_drift"] = cause_drift
    diff["aggregate_drift"] = aggregate_drift
    changed = bool(cause_drift or aggregate_drift or candidates_added or candidates_removed)
    diff["changed"] = changed
    return diff, changed, asymmetric


def _o_ws_canonical_valid(value: Any) -> bool:
    """Oracle ws_canonical validator: mirrors diffreports._valid_ws_canonical."""
    if type(value) is not dict or value.get("available") is not True:
        return False
    mutations = value.get("mutations")
    if type(mutations) is not list or not mutations or not all(type(m) is str for m in mutations):
        return False
    per_candidate = value.get("per_candidate")
    if type(per_candidate) is not dict or not all(type(k) is str for k in per_candidate):
        return False
    for entry in per_candidate.values():
        if type(entry) is not dict:
            return False
        robust = entry.get("whitespace_robust")
        if robust is not None and type(robust) is not bool:
            return False
        cells = entry.get("per_mutation")
        if type(cells) is not dict:
            return False
        for mutation, cell in cells.items():
            if type(mutation) is not str or type(cell) is not dict:
                return False
            for key in ("ids_divergence_rate", "verdict_flip_rate"):
                rate_value = cell.get(key)
                if rate_value is not None and type(rate_value) not in (int, float):
                    return False
            for key in ("n_ids_divergent", "n_verdict_flips", "n_same_ids_verdict_flips"):
                count_value = cell.get(key)
                if count_value is not None and type(count_value) is not int:
                    return False
            fragile = cell.get("fragile")
            if fragile is not None and type(fragile) is not bool:
                return False
    return True


def _o_ws_canonical(old: Any, new: Any) -> tuple[dict[str, Any], bool, bool]:
    """Oracle ws_canonical section: id/flip drift and robust flips.

    Independent reimplementation: set-driven comparison rather than
    the implementation's nested accumulation loops.
    """
    old_ok = _o_ws_canonical_valid(old)
    new_ok = _o_ws_canonical_valid(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    diff: dict[str, Any] = {"available": available, "changed": False}
    asymmetric = available not in ("both", "neither")
    if available != "both":
        return diff, False, asymmetric

    def _cells(entry: Any) -> dict[str, dict[str, Any]]:
        """Cells."""
        raw = (entry or {}).get("per_mutation") or {}
        return {
            mutation: {
                "ids_divergence_rate": (cell or {}).get("ids_divergence_rate"),
                "verdict_flip_rate": (cell or {}).get("verdict_flip_rate"),
                "n_same_ids_verdict_flips": (cell or {}).get("n_same_ids_verdict_flips"),
                "fragile": (cell or {}).get("fragile"),
            }
            for mutation, cell in raw.items()
            if type(mutation) is str
        }

    old_cands = {
        name: {"whitespace_robust": (entry or {}).get("whitespace_robust"), "cells": _cells(entry)}
        for name, entry in (old.get("per_candidate") or {}).items()
    }
    new_cands = {
        name: {"whitespace_robust": (entry or {}).get("whitespace_robust"), "cells": _cells(entry)}
        for name, entry in (new.get("per_candidate") or {}).items()
    }
    candidates_added = sorted(set(new_cands) - set(old_cands))
    candidates_removed = sorted(set(old_cands) - set(new_cands))
    diff["candidates_added"] = candidates_added
    diff["candidates_removed"] = candidates_removed
    old_muts = sorted({m for m in (old.get("mutations") or []) if type(m) is str})
    new_muts = sorted({m for m in (new.get("mutations") or []) if type(m) is str})
    diff["mutations_added"] = sorted(set(new_muts) - set(old_muts))
    diff["mutations_removed"] = sorted(set(old_muts) - set(new_muts))
    metrics = (
        "ids_divergence_rate",
        "verdict_flip_rate",
        "n_same_ids_verdict_flips",
        "fragile",
    )
    rate_drift = [
        {
            "candidate": name,
            "mutation": mutation,
            "metric": metric,
            "old": old_cands[name]["cells"][mutation][metric],
            "new": new_cands[name]["cells"][mutation][metric],
        }
        for name in sorted(set(old_cands) & set(new_cands))
        for mutation in sorted(set(old_cands[name]["cells"]) & set(new_cands[name]["cells"]))
        for metric in metrics
        if old_cands[name]["cells"][mutation][metric] != new_cands[name]["cells"][mutation][metric]
    ]
    robust_flips = [
        {
            "candidate": name,
            "old_robust": old_cands[name]["whitespace_robust"],
            "new_robust": new_cands[name]["whitespace_robust"],
        }
        for name in sorted(set(old_cands) & set(new_cands))
        if old_cands[name]["whitespace_robust"] != new_cands[name]["whitespace_robust"]
    ]
    fragile_old = set(old.get("fragile_candidates") or [])
    fragile_new = set(new.get("fragile_candidates") or [])
    diff["rate_drift"] = rate_drift
    diff["robust_flips"] = robust_flips
    diff["fragile_added"] = sorted(fragile_new - fragile_old)
    diff["fragile_removed"] = sorted(fragile_old - fragile_new)
    changed = bool(
        rate_drift
        or robust_flips
        or diff["fragile_added"]
        or diff["fragile_removed"]
        or candidates_added
        or candidates_removed
        or diff["mutations_added"]
        or diff["mutations_removed"]
    )
    diff["changed"] = changed
    return diff, changed, asymmetric


def _o_structured_valid(value: Any) -> bool:
    """Oracle structured validator: mirrors diffreports._valid_structured."""
    if type(value) is not dict or value.get("available") is not True:
        return False
    n_fixtures = value.get("n_fixtures")
    if type(n_fixtures) is not int or n_fixtures < 0:
        return False
    per_candidate = value.get("per_candidate")
    if type(per_candidate) is not dict or not all(type(k) is str for k in per_candidate):
        return False
    for entry in per_candidate.values():
        if entry is None:
            continue
        if type(entry) is not dict:
            return False
        rates: dict[str, Any] = {}
        for key in ("parse_rate", "schema_rate", "adherent_rate"):
            rate = entry.get(key)
            if rate is not None and not (type(rate) in (int, float) and 0.0 <= rate <= 1.0):
                return False
            rates[key] = rate
        parse_rate = rates["parse_rate"]
        if parse_rate is not None:
            for key in ("schema_rate", "adherent_rate"):
                if rates[key] is not None and rates[key] > parse_rate:
                    return False
    weakest = value.get("weakest_candidates")
    if type(weakest) is not list or not all(type(w) is str for w in weakest):
        return False
    gap = value.get("adherence_gap")
    if gap is None:
        return True
    return type(gap) in (int, float) and 0.0 <= gap <= 1.0


def _o_structured(old: Any, new: Any) -> tuple[dict[str, Any], bool, bool]:
    """Oracle structured section: rate/gap drift and weakest-set changes.

    Independent reimplementation: set-driven comparison rather than
    the implementation's nested accumulation loops.
    """
    old_ok = _o_structured_valid(old)
    new_ok = _o_structured_valid(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    diff: dict[str, Any] = {"available": available, "changed": False}
    asymmetric = available not in ("both", "neither")
    if available != "both":
        return diff, False, asymmetric

    def _rates(entry: Any) -> dict[str, Any]:
        """Rates."""
        entry = entry or {}
        return {
            metric: entry.get(metric) for metric in ("parse_rate", "schema_rate", "adherent_rate")
        }

    old_cands = {name: _rates(entry) for name, entry in (old.get("per_candidate") or {}).items()}
    new_cands = {name: _rates(entry) for name, entry in (new.get("per_candidate") or {}).items()}
    candidates_added = sorted(set(new_cands) - set(old_cands))
    candidates_removed = sorted(set(old_cands) - set(new_cands))
    metrics = ("parse_rate", "schema_rate", "adherent_rate")
    rate_drift = [
        {
            "candidate": name,
            "metric": metric,
            "old": old_cands[name][metric],
            "new": new_cands[name][metric],
        }
        for name in sorted(set(old_cands) & set(new_cands))
        for metric in metrics
        if old_cands[name][metric] != new_cands[name][metric]
    ]
    gap_drift = []
    if old.get("adherence_gap") != new.get("adherence_gap"):
        gap_drift.append({"old": old.get("adherence_gap"), "new": new.get("adherence_gap")})
    weakest_added = sorted(
        set(new.get("weakest_candidates") or []) - set(old.get("weakest_candidates") or [])
    )
    weakest_removed = sorted(
        set(old.get("weakest_candidates") or []) - set(new.get("weakest_candidates") or [])
    )
    diff["rate_drift"] = rate_drift
    diff["gap_drift"] = gap_drift
    diff["weakest_added"] = weakest_added
    diff["weakest_removed"] = weakest_removed
    diff["candidates_added"] = candidates_added
    diff["candidates_removed"] = candidates_removed
    changed = bool(
        rate_drift
        or gap_drift
        or weakest_added
        or weakest_removed
        or candidates_added
        or candidates_removed
    )
    diff["changed"] = changed
    return diff, changed, asymmetric


def _o_deliforge_valid(value: Any) -> bool:
    """Oracle deliforge validator: mirrors diffreports._valid_deliforge."""
    if type(value) is not dict or value.get("available") is not True:
        return False
    payloads = value.get("payloads")
    if type(payloads) is not list or not payloads or not all(type(p) is str for p in payloads):
        return False
    marker_indexes = value.get("marker_indexes")
    if (
        type(marker_indexes) is not list
        or not marker_indexes
        or not all(type(i) is int for i in marker_indexes)
    ):
        return False
    per_candidate = value.get("per_candidate")
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
    excluded = value.get("excluded_candidates")
    return type(excluded) is list and all(type(c) is str for c in excluded)


def _o_deliforge(old: Any, new: Any) -> tuple[dict[str, Any], bool, bool]:
    """Oracle deliforge section: per-cell authority-gap drift.

    Independent reimplementation: dict-comprehension views rather than
    the implementation's nested accumulation loops.
    """
    old_ok = _o_deliforge_valid(old)
    new_ok = _o_deliforge_valid(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    diff: dict[str, Any] = {"available": available, "changed": False}
    asymmetric = available not in ("both", "neither")
    if available != "both":
        return diff, False, asymmetric

    def _view(payload: Any) -> dict[str, dict[str, Any]]:
        """View."""
        out: dict[str, dict[str, Any]] = {}
        for name, entry in (payload.get("per_candidate") or {}).items():
            entry = entry or {}
            cells = {
                key: (
                    cell.get("authority_gap")
                    if type(cell) is dict and cell.get("supported") is True
                    else None
                )
                for key, cell in (entry.get("cells") or {}).items()
            }
            out[name] = {
                "cells": cells,
                "max_authority_gap": entry.get("max_authority_gap"),
                "excluded": name in set(payload.get("excluded_candidates") or []),
            }
        return out

    old_view = _view(old)
    new_view = _view(new)
    candidates_added = sorted(set(new_view) - set(old_view))
    candidates_removed = sorted(set(old_view) - set(new_view))
    gap_drift = [
        {
            "candidate": name,
            "cell": key,
            "old": old_view[name]["cells"][key],
            "new": new_view[name]["cells"][key],
        }
        for name in sorted(set(old_view) & set(new_view))
        for key in sorted(set(old_view[name]["cells"]) & set(new_view[name]["cells"]))
        if old_view[name]["cells"][key] != new_view[name]["cells"][key]
    ]
    top_gap_drift = [
        {
            "candidate": name,
            "old": old_view[name]["max_authority_gap"],
            "new": new_view[name]["max_authority_gap"],
        }
        for name in sorted(set(old_view) & set(new_view))
        if old_view[name]["max_authority_gap"] != new_view[name]["max_authority_gap"]
    ]
    old_excluded = {name for name, v in old_view.items() if v["excluded"]}
    new_excluded = {name for name, v in new_view.items() if v["excluded"]}
    excluded_added = sorted(new_excluded - old_excluded)
    excluded_removed = sorted(old_excluded - new_excluded)
    diff["gap_drift"] = gap_drift
    diff["top_gap_drift"] = top_gap_drift
    diff["candidates_added"] = candidates_added
    diff["candidates_removed"] = candidates_removed
    diff["excluded_added"] = excluded_added
    diff["excluded_removed"] = excluded_removed
    changed = bool(
        gap_drift
        or top_gap_drift
        or candidates_added
        or candidates_removed
        or excluded_added
        or excluded_removed
    )
    diff["changed"] = changed
    return diff, changed, asymmetric


def _o_fewshot_valid(value: Any) -> bool:
    """Oracle fewshot validator: mirrors diffreports._valid_fewshot."""
    if type(value) is not dict or value.get("available") is not True:
        return False
    demo_ids = value.get("demo_ids")
    if type(demo_ids) is not list or not demo_ids or not all(type(d) is str for d in demo_ids):
        return False
    if type(value.get("k")) is not int or type(value.get("n_eval")) is not int:
        return False
    per_candidate = value.get("per_candidate")
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
    per_arm = value.get("per_arm")
    if type(per_arm) is not dict:
        return False
    for arm in per_arm.values():
        if type(arm) is not dict or type(arm.get("decision")) is not str:
            return False
        winner = arm.get("winner")
        if winner is not None and type(winner) is not str:
            return False
    flips = value.get("flips")
    if type(flips) is not list or any(type(f) is not dict for f in flips):
        return False
    excluded = value.get("excluded_candidates")
    return type(excluded) is list and all(type(c) is str for c in excluded)


def _o_fewshot(old: Any, new: Any) -> tuple[dict[str, Any], bool, bool]:
    """Oracle fewshot section: gain/sensitivity/arm drift.

    Independent reimplementation: dict-comprehension views rather than
    the implementation's nested accumulation loops.
    """
    old_ok = _o_fewshot_valid(old)
    new_ok = _o_fewshot_valid(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    diff: dict[str, Any] = {"available": available, "changed": False}
    asymmetric = available not in ("both", "neither")
    if available != "both":
        return diff, False, asymmetric

    def _view(payload: Any) -> dict[str, dict[str, Any]]:
        """View."""
        out: dict[str, dict[str, Any]] = {}
        for name, entry in (payload.get("per_candidate") or {}).items():
            entry = entry or {}
            out[name] = {
                "gain": entry.get("gain"),
                "order_sensitivity": entry.get("order_sensitivity"),
                "excluded": name in set(payload.get("excluded_candidates") or []),
            }
        return out

    def _arm_view(payload: Any) -> dict[str, tuple[Any, Any]]:
        """Arm view."""
        return {
            arm: (cell.get("decision"), cell.get("winner"))
            for arm, cell in (payload.get("per_arm") or {}).items()
            if type(cell) is dict
        }

    old_view = _view(old)
    new_view = _view(new)
    candidates_added = sorted(set(new_view) - set(old_view))
    candidates_removed = sorted(set(old_view) - set(new_view))
    gain_drift = [
        {"candidate": name, "old": old_view[name]["gain"], "new": new_view[name]["gain"]}
        for name in sorted(set(old_view) & set(new_view))
        if old_view[name]["gain"] != new_view[name]["gain"]
    ]
    sensitivity_drift = [
        {
            "candidate": name,
            "old": old_view[name]["order_sensitivity"],
            "new": new_view[name]["order_sensitivity"],
        }
        for name in sorted(set(old_view) & set(new_view))
        if old_view[name]["order_sensitivity"] != new_view[name]["order_sensitivity"]
    ]
    old_excluded = {name for name, v in old_view.items() if v["excluded"]}
    new_excluded = {name for name, v in new_view.items() if v["excluded"]}
    excluded_added = sorted(new_excluded - old_excluded)
    excluded_removed = sorted(old_excluded - new_excluded)
    old_arms = _arm_view(old)
    new_arms = _arm_view(new)
    arm_drift = [
        {
            "arm": arm,
            "old_decision": old_arms[arm][0],
            "new_decision": new_arms[arm][0],
            "old_winner": old_arms[arm][1],
            "new_winner": new_arms[arm][1],
        }
        for arm in sorted(set(old_arms) & set(new_arms))
        if old_arms[arm] != new_arms[arm]
    ]
    diff["gain_drift"] = gain_drift
    diff["sensitivity_drift"] = sensitivity_drift
    diff["arm_drift"] = arm_drift
    diff["candidates_added"] = candidates_added
    diff["candidates_removed"] = candidates_removed
    diff["excluded_added"] = excluded_added
    diff["excluded_removed"] = excluded_removed
    changed = bool(
        gain_drift
        or sensitivity_drift
        or arm_drift
        or candidates_added
        or candidates_removed
        or excluded_added
        or excluded_removed
    )
    diff["changed"] = changed
    return diff, changed, asymmetric


def _o_faketurn_valid(value: Any) -> bool:
    """Oracle faketurn validator: mirrors diffreports._valid_faketurn."""
    if type(value) is not dict or value.get("available") is not True:
        return False
    payloads = value.get("payloads")
    if type(payloads) is not list or not payloads or not all(type(p) is str for p in payloads):
        return False
    marker_indexes = value.get("marker_indexes")
    if (
        type(marker_indexes) is not list
        or not marker_indexes
        or not all(type(i) is int for i in marker_indexes)
    ):
        return False
    per_candidate = value.get("per_candidate")
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
    excluded = value.get("excluded_candidates")
    return type(excluded) is list and all(type(c) is str for c in excluded)


def _o_faketurn(old: Any, new: Any) -> tuple[dict[str, Any], bool, bool]:
    """Oracle faketurn section: flip-rate, provenance, and gap drift.

    Independent reimplementation: comprehension-built views rather
    than the implementation's accumulation loops.
    """
    old_ok = _o_faketurn_valid(old)
    new_ok = _o_faketurn_valid(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    diff: dict[str, Any] = {"available": available, "changed": False}
    asymmetric = available not in ("both", "neither")
    if available != "both":
        return diff, False, asymmetric

    def _view(payload: Any) -> dict[str, dict[str, Any]]:
        """View."""
        out: dict[str, dict[str, Any]] = {}
        for name, entry in (payload.get("per_candidate") or {}).items():
            entry = entry or {}
            cells = {
                key: (
                    {"flip_rate": cell.get("flip_rate"), "token_source": cell.get("token_source")}
                    if type(cell) is dict and cell.get("supported") is True
                    else None
                )
                for key, cell in (entry.get("cells") or {}).items()
            }
            out[name] = {
                "cells": cells,
                "transfer_gap": entry.get("transfer_gap"),
                "max_flip_rate": entry.get("max_flip_rate"),
                "excluded": name in set(payload.get("excluded_candidates") or []),
            }
        return out

    old_view = _view(old)
    new_view = _view(new)
    candidates_added = sorted(set(new_view) - set(old_view))
    candidates_removed = sorted(set(old_view) - set(new_view))
    # Single pass in the implementation's (name, key) iteration order:
    # supportedness flips and token-source changes interleave exactly
    # as diffreports.diff_faketurn appends them.
    cell_drift: list[dict[str, Any]] = []
    source_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_view) & set(new_view)):
        for key in sorted(set(old_view[name]["cells"]) & set(new_view[name]["cells"])):
            old_cell = old_view[name]["cells"][key]
            new_cell = new_view[name]["cells"][key]
            if old_cell is None or new_cell is None:
                if old_cell != new_cell:
                    source_drift.append(
                        {
                            "candidate": name,
                            "cell": key,
                            "old": "supported" if old_cell is not None else "unsupported",
                            "new": "supported" if new_cell is not None else "unsupported",
                        }
                    )
                continue
            if old_cell["flip_rate"] != new_cell["flip_rate"]:
                cell_drift.append(
                    {
                        "candidate": name,
                        "cell": key,
                        "old": old_cell["flip_rate"],
                        "new": new_cell["flip_rate"],
                    }
                )
            if old_cell["token_source"] != new_cell["token_source"]:
                source_drift.append(
                    {
                        "candidate": name,
                        "cell": key,
                        "old": old_cell["token_source"],
                        "new": new_cell["token_source"],
                    }
                )
    gap_drift = [
        {
            "candidate": name,
            "old": old_view[name]["transfer_gap"],
            "new": new_view[name]["transfer_gap"],
        }
        for name in sorted(set(old_view) & set(new_view))
        if old_view[name]["transfer_gap"] != new_view[name]["transfer_gap"]
    ]
    top_rate_drift = [
        {
            "candidate": name,
            "old": old_view[name]["max_flip_rate"],
            "new": new_view[name]["max_flip_rate"],
        }
        for name in sorted(set(old_view) & set(new_view))
        if old_view[name]["max_flip_rate"] != new_view[name]["max_flip_rate"]
    ]
    old_excluded = {name for name, v in old_view.items() if v["excluded"]}
    new_excluded = {name for name, v in new_view.items() if v["excluded"]}
    excluded_added = sorted(new_excluded - old_excluded)
    excluded_removed = sorted(old_excluded - new_excluded)
    diff["cell_drift"] = cell_drift
    diff["gap_drift"] = gap_drift
    diff["top_rate_drift"] = top_rate_drift
    diff["source_drift"] = source_drift
    diff["candidates_added"] = candidates_added
    diff["candidates_removed"] = candidates_removed
    diff["excluded_added"] = excluded_added
    diff["excluded_removed"] = excluded_removed
    changed = bool(
        cell_drift
        or gap_drift
        or top_rate_drift
        or source_drift
        or candidates_added
        or candidates_removed
        or excluded_added
        or excluded_removed
    )
    diff["changed"] = changed
    return diff, changed, asymmetric


def _o_toolcall_valid(value: Any) -> bool:
    """Oracle toolcall validator: mirrors diffreports._valid_toolcall."""
    if type(value) is not dict or value.get("available") is not True:
        return False
    n_fixtures = value.get("n_fixtures")
    if type(n_fixtures) is not int or n_fixtures < 0:
        return False
    per_candidate = value.get("per_candidate")
    if type(per_candidate) is not dict or not all(type(k) is str for k in per_candidate):
        return False
    for entry in per_candidate.values():
        if entry is None:
            continue
        if type(entry) is not dict:
            return False
        for key in ("n", "n_evaluable", "adherent", "parse_failed", "call_failed", "not_text"):
            count = entry.get(key)
            if type(count) is not int or count < 0:
                return False
        if entry["n_evaluable"] > entry["n"]:
            return False
        for key in ("selection_rate", "args_rate", "relevance_rate", "adherent_rate"):
            rate = entry.get(key)
            if rate is not None and not (type(rate) in (int, float) and 0.0 <= rate <= 1.0):
                return False
    excluded = value.get("excluded_candidates")
    if type(excluded) is not list or not all(type(c) is str for c in excluded):
        return False
    weakest = value.get("weakest_candidates")
    return type(weakest) is list and all(type(c) is str for c in weakest)


def _o_toolcall(old: Any, new: Any) -> tuple[dict[str, Any], bool, bool]:
    """Oracle toolcall section: selection/args/relevance rate drift.

    Independent reimplementation: comprehension-built views rather
    than the implementation's accumulation loops, with the drift
    buckets filled in the implementation's per-candidate rate order.
    """
    old_ok = _o_toolcall_valid(old)
    new_ok = _o_toolcall_valid(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    diff: dict[str, Any] = {"available": available, "changed": False}
    asymmetric = available not in ("both", "neither")
    if available != "both":
        return diff, False, asymmetric

    def _view(payload: Any) -> dict[str, dict[str, Any]]:
        # A None cell (candidate missing from the stage) counts as
        # excluded, mirroring the implementation and the profile's
        # excluded_candidates semantics.
        """View."""
        out: dict[str, dict[str, Any]] = {}
        for name, entry in (payload.get("per_candidate") or {}).items():
            if entry is None:
                out[name] = {
                    "selection_rate": None,
                    "args_rate": None,
                    "relevance_rate": None,
                    "excluded": True,
                }
            else:
                out[name] = {
                    "selection_rate": entry.get("selection_rate"),
                    "args_rate": entry.get("args_rate"),
                    "relevance_rate": entry.get("relevance_rate"),
                    "excluded": name in set(payload.get("excluded_candidates") or []),
                }
        return out

    old_view = _view(old)
    new_view = _view(new)
    candidates_added = sorted(set(new_view) - set(old_view))
    candidates_removed = sorted(set(old_view) - set(new_view))
    # Same per-candidate rate order as the implementation's loop.
    selection_drift: list[dict[str, Any]] = []
    args_drift: list[dict[str, Any]] = []
    relevance_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_view) & set(new_view)):
        for rate_key, bucket in (
            ("selection_rate", selection_drift),
            ("args_rate", args_drift),
            ("relevance_rate", relevance_drift),
        ):
            if old_view[name][rate_key] != new_view[name][rate_key]:
                bucket.append(
                    {
                        "candidate": name,
                        "old": old_view[name][rate_key],
                        "new": new_view[name][rate_key],
                    }
                )
    old_excluded = {name for name, v in old_view.items() if v["excluded"]}
    new_excluded = {name for name, v in new_view.items() if v["excluded"]}
    excluded_added = sorted(new_excluded - old_excluded)
    excluded_removed = sorted(old_excluded - new_excluded)
    diff["selection_drift"] = selection_drift
    diff["args_drift"] = args_drift
    diff["relevance_drift"] = relevance_drift
    diff["candidates_added"] = candidates_added
    diff["candidates_removed"] = candidates_removed
    diff["excluded_added"] = excluded_added
    diff["excluded_removed"] = excluded_removed
    changed = bool(
        selection_drift
        or args_drift
        or relevance_drift
        or candidates_added
        or candidates_removed
        or excluded_added
        or excluded_removed
    )
    diff["changed"] = changed
    return diff, changed, asymmetric


def _o_contamination(old: Any, new: Any) -> tuple[dict[str, Any], bool, bool]:
    """Oracle contamination section: worst tier and pair-set drift."""
    old_ok = _o_contamination_valid(old)
    new_ok = _o_contamination_valid(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    diff: dict[str, Any] = {"available": available, "changed": False}
    asymmetric = available not in ("both", "neither")
    if available != "both":
        return diff, False, asymmetric

    def _key(pair: dict[str, Any]) -> tuple[str, str] | None:
        """Key."""
        held = pair.get("held_out")
        search = pair.get("search")
        if type(held) is str and type(search) is str:
            return (held, search)
        return None

    old_worst = old.get("worst_tier")
    new_worst = new.get("worst_tier")
    old_pairs: dict[tuple[str, str], Any] = {}
    for p in old.get("pairs") or []:
        k = _key(p)
        if k is not None:
            old_pairs[k] = p.get("tier")
    new_pairs: dict[tuple[str, str], Any] = {}
    for p in new.get("pairs") or []:
        k = _key(p)
        if k is not None:
            new_pairs[k] = p.get("tier")
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
    diff["worst_tier"] = {
        "old": old_worst,
        "new": new_worst,
        "changed": worst_changed,
    }
    diff["pairs_added"] = [
        {"held_out": k[0], "search": k[1], "tier": new_pairs[k]} for k in pairs_added
    ]
    diff["pairs_removed"] = [
        {"held_out": k[0], "search": k[1], "tier": old_pairs[k]} for k in pairs_removed
    ]
    diff["tier_changes"] = tier_changes
    changed = bool(worst_changed or pairs_added or pairs_removed or tier_changes)
    diff["changed"] = changed
    return diff, changed, asymmetric


def _o_perturbation(old: Any, new: Any) -> tuple[dict[str, Any], bool, bool]:
    """Oracle perturbation section: stability, variant, and set drift."""
    old_ok = _o_perturbation_valid(old)
    new_ok = _o_perturbation_valid(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    diff: dict[str, Any] = {"available": available, "changed": False}
    asymmetric = available not in ("both", "neither")
    if available != "both":
        return diff, False, asymmetric

    def _triplet(stability: Any) -> tuple[Any, Any, Any]:
        """Triplet."""
        if type(stability) is not dict:
            return (None, None, None)
        return (
            stability.get("winner_retained"),
            stability.get("decision_retained"),
            stability.get("n_evaluable"),
        )

    old_t, new_t = _triplet(old.get("stability")), _triplet(new.get("stability"))
    stability_changed = old_t != new_t
    diff["stability"] = {
        "old": {
            "winner_retained": old_t[0],
            "decision_retained": old_t[1],
            "n_evaluable": old_t[2],
        },
        "new": {
            "winner_retained": new_t[0],
            "decision_retained": new_t[1],
            "n_evaluable": new_t[2],
        },
        "changed": stability_changed,
    }
    old_pv = old.get("per_variant") or {}
    new_pv = new.get("per_variant") or {}
    old_ids, new_ids = sorted(old_pv), sorted(new_pv)
    variants_added = [v for v in new_ids if v not in old_pv]
    variants_removed = [v for v in old_ids if v not in new_pv]
    variant_drift = [
        {
            "variant": vid,
            "operator": (new_pv.get(vid) or {}).get("operator"),
            "old_decision": (old_pv.get(vid) or {}).get("decision"),
            "old_winner": (old_pv.get(vid) or {}).get("winner"),
            "new_decision": (new_pv.get(vid) or {}).get("decision"),
            "new_winner": (new_pv.get(vid) or {}).get("winner"),
        }
        for vid in sorted(set(old_ids) & set(new_ids))
        if (
            ((old_pv.get(vid) or {}).get("decision"), (old_pv.get(vid) or {}).get("winner"))
            != (
                (new_pv.get(vid) or {}).get("decision"),
                (new_pv.get(vid) or {}).get("winner"),
            )
        )
    ]
    operators_changed = old.get("operators") != new.get("operators")
    diff["variant_drift"] = variant_drift
    diff["variants_added"] = variants_added
    diff["variants_removed"] = variants_removed
    diff["operators_changed"] = operators_changed
    changed = bool(
        stability_changed
        or variant_drift
        or variants_added
        or variants_removed
        or operators_changed
    )
    diff["changed"] = changed
    return diff, changed, asymmetric


def _o_transfer(old: Any, new: Any) -> tuple[dict[str, Any], bool, bool]:
    """Oracle transfer section: verdict, checkpoint, and cell drift."""
    old_ok = _o_transfer_valid(old)
    new_ok = _o_transfer_valid(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    diff: dict[str, Any] = {"available": available, "changed": False}
    asymmetric = available not in ("both", "neither")
    if available != "both":
        return diff, False, asymmetric
    old_verdict = old.get("transfer_verdict")
    new_verdict = new.get("transfer_verdict")
    old_pc = old.get("per_checkpoint") or {}
    new_pc = new.get("per_checkpoint") or {}
    old_matrix = old.get("matrix") or {}
    new_matrix = new.get("matrix") or {}
    checkpoints_added = [c for c in sorted(new_pc) if c not in old_pc]
    checkpoints_removed = [c for c in sorted(old_pc) if c not in new_pc]
    checkpoint_drift = []
    for ckpt in sorted(set(old_pc) & set(new_pc)):
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
    cell_drift = []
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
    diff["transfer_verdict"] = {
        "old": old_verdict,
        "new": new_verdict,
        "changed": verdict_changed,
    }
    diff["checkpoint_drift"] = checkpoint_drift
    diff["cell_drift"] = cell_drift
    diff["checkpoints_added"] = checkpoints_added
    diff["checkpoints_removed"] = checkpoints_removed
    changed = bool(
        verdict_changed
        or checkpoint_drift
        or cell_drift
        or checkpoints_added
        or checkpoints_removed
    )
    diff["changed"] = changed
    return diff, changed, asymmetric


def _oracle_diff(
    old: dict[str, Any],
    new: dict[str, Any],
    *,
    all_fixtures: bool = False,
    old_label: str = "old",
    new_label: str = "new",
) -> dict[str, Any]:
    """Independent reimplementation of ``diff_reports`` from the rules."""
    decision_changed = old["decision"] != new["decision"]
    # The winner section normalizes "" to None, matching _held_verdict
    # (which treats a falsy winner as absent).
    old_w_raw, new_w_raw = old.get("selected_candidate"), new.get("selected_candidate")
    old_w = old_w_raw if type(old_w_raw) is str and old_w_raw else None
    new_w = new_w_raw if type(new_w_raw) is str and new_w_raw else None
    winner_changed = old_w != new_w

    old_sv, old_pl = _o_search_profile(old)
    new_sv, new_pl = _o_search_profile(new)
    old_hw, old_hv = _o_held_profile(old)
    new_hw, new_hv = _o_held_profile(new)

    verdict_changes = [
        {
            "candidate": name,
            "stage": "search",
            "old": old_sv.get(name, "absent"),
            "new": new_sv.get(name, "absent"),
        }
        for name in sorted(set(old_sv) | set(new_sv))
        if old_sv.get(name, "absent") != new_sv.get(name, "absent")
    ]
    verdict_changes.extend(
        {
            "candidate": name,
            "stage": "held_out",
            "old": old_hv if name == old_hw else "absent",
            "new": new_hv if name == new_hw else "absent",
        }
        for name in sorted({w for w in (old_hw, new_hw) if w is not None})
        if (old_hv if name == old_hw else "absent") != (new_hv if name == new_hw else "absent")
    )

    old_ev = {stage: _o_evidence(old, stage) for stage in ("search", "held_out")}
    new_ev = {stage: _o_evidence(new, stage) for stage in ("search", "held_out")}
    old_ok = any(present for _, _, present in old_ev.values())
    new_ok = any(present for _, _, present in new_ev.values())
    skipped = {
        "old": sum(s for _, s, _ in old_ev.values()),
        "new": sum(s for _, s, _ in new_ev.values()),
    }

    changed_cells: list[dict[str, str]] = []
    listed_cells: list[dict[str, str]] = []
    if old_ok and new_ok:
        for stage in ("search", "held_out"):
            old_map, new_map = old_ev[stage][0], new_ev[stage][0]
            for name in sorted(set(old_map) | set(new_map)):
                old_fx = old_map.get(name, {})
                new_fx = new_map.get(name, {})
                for fixture in sorted(set(old_fx) | set(new_fx)):
                    cell = {
                        "stage": stage,
                        "candidate": name,
                        "fixture": fixture,
                        "old": old_fx.get(fixture, "absent"),
                        "new": new_fx.get(fixture, "absent"),
                    }
                    if cell["old"] != cell["new"]:
                        changed_cells.append(cell)
                    if all_fixtures or cell["old"] != cell["new"]:
                        listed_cells.append(cell)
    order = lambda e: (e["stage"], e["candidate"], e["fixture"])  # noqa: E731
    fixture_diffs = sorted(listed_cells, key=order)
    changed_sorted = sorted(changed_cells, key=order)
    flips = [
        c for c in changed_sorted if c["old"] in ("pass", "fail") and c["new"] in ("pass", "fail")
    ]

    config_drift = _o_drift(_o_snapshot(old), _o_snapshot(new))
    config_drift.extend(
        {"field": f"planned:{name}", "old": old_pl[name], "new": new_pl[name]}
        for name in sorted(set(old_pl) & set(new_pl))
        if old_pl[name] != new_pl[name]
    )
    if old_ok and new_ok:
        for stage in ("search", "held_out"):
            old_ids = sorted({fx for cand in old_ev[stage][0].values() for fx in cand})
            new_ids = sorted({fx for cand in new_ev[stage][0].values() for fx in cand})
            if old_ids != new_ids:
                config_drift.append({"field": f"fixtures:{stage}", "old": old_ids, "new": new_ids})

    evidence_asymmetric = old_ok != new_ok
    o_frontier, o_frontier_changed, o_frontier_asymmetric = _o_frontier(
        old.get("budget_frontier"), new.get("budget_frontier")
    )
    o_hof, o_hof_changed, o_hof_asymmetric = _o_held_out_floor(
        old.get("held_out_budget_floor"), new.get("held_out_budget_floor")
    )
    o_ba, o_ba_changed, o_ba_asymmetric = _o_boundary_audit(
        old.get("boundary_audit"), new.get("boundary_audit")
    )
    o_sr, o_sr_changed, o_sr_asymmetric = _o_selection_robustness(
        old.get("selection_robustness"), new.get("selection_robustness")
    )
    o_tr, o_tr_changed, o_tr_asymmetric = _o_transfer(old.get("transfer"), new.get("transfer"))
    o_ct, o_ct_changed, o_ct_asymmetric = _o_contamination(
        old.get("contamination"), new.get("contamination")
    )
    o_pb, o_pb_changed, o_pb_asymmetric = _o_perturbation(
        old.get("perturbation"), new.get("perturbation")
    )
    o_hl, o_hl_changed, o_hl_asymmetric = _o_healing(
        old.get("token_healing"), new.get("token_healing")
    )
    o_hy, o_hy_changed, o_hy_asymmetric = _o_hierarchy(old.get("hierarchy"), new.get("hierarchy"))
    o_pf, o_pf_changed, o_pf_asymmetric = _o_prefill(old.get("prefill"), new.get("prefill"))
    o_po, o_po_changed, o_po_asymmetric = _o_portfolio(old.get("portfolio"), new.get("portfolio"))
    o_sh, o_sh_changed, o_sh_asymmetric = _o_synthhistory(
        old.get("synthhistory"), new.get("synthhistory")
    )
    o_bi, o_bi_changed, o_bi_asymmetric = _o_batchinv(
        old.get("batch_invariance"), new.get("batch_invariance")
    )
    o_pos, o_pos_changed, o_pos_asymmetric = _o_position(old.get("position"), new.get("position"))
    o_sc, o_sc_changed, o_sc_asymmetric = _o_stop_cause(
        old.get("stop_cause"), new.get("stop_cause")
    )
    o_ws, o_ws_changed, o_ws_asymmetric = _o_ws_canonical(
        old.get("ws_canonical"), new.get("ws_canonical")
    )
    o_st, o_st_changed, o_st_asymmetric = _o_structured(
        old.get("structured"), new.get("structured")
    )
    o_dl, o_dl_changed, o_dl_asymmetric = _o_deliforge(old.get("deliforge"), new.get("deliforge"))
    o_fs, o_fs_changed, o_fs_asymmetric = _o_fewshot(old.get("fewshot"), new.get("fewshot"))
    o_ft, o_ft_changed, o_ft_asymmetric = _o_faketurn(old.get("faketurn"), new.get("faketurn"))
    o_tc, o_tc_changed, o_tc_asymmetric = _o_toolcall(old.get("toolcall"), new.get("toolcall"))
    o_vb, o_vb_changed, o_vb_asymmetric = _o_verbosity(old.get("verbosity"), new.get("verbosity"))
    o_rf, o_rf_changed, o_rf_asymmetric = _o_refusals(old.get("refusals"), new.get("refusals"))
    o_sl, o_sl_changed, o_sl_asymmetric = _o_scorerlens(
        old.get("scorerlens"), new.get("scorerlens")
    )
    o_al, o_al_changed, o_al_asymmetric = _o_aliasing(old.get("aliasing"), new.get("aliasing"))
    o_rep, o_rep_changed, o_rep_asymmetric = _o_repetition(
        old.get("repetition"), new.get("repetition")
    )
    o_dst, o_dst_changed, o_dst_asymmetric = _o_distraction(
        old.get("distraction"), new.get("distraction")
    )
    o_lead, o_lead_changed, o_lead_asymmetric = _o_leading(old.get("leading"), new.get("leading"))
    o_nf, o_nf_changed, o_nf_asymmetric = _o_numform(old.get("numform"), new.get("numform"))
    o_dt, o_dt_changed, o_dt_asymmetric = _o_dateline(old.get("dateline"), new.get("dateline"))
    o_mt, o_mt_changed, o_mt_asymmetric = _o_multiturn(old.get("multiturn"), new.get("multiturn"))
    o_nd, o_nd_changed, o_nd_asymmetric = _o_needle(old.get("needle"), new.get("needle"))
    o_pl, o_pl_changed, o_pl_asymmetric = _o_placement(old.get("placement"), new.get("placement"))
    o_cform, o_cform_changed, o_cform_asymmetric = _o_canonicalform(
        old.get("canonical_form"), new.get("canonical_form")
    )
    o_idist, o_idist_changed, o_idist_asymmetric = _o_instructiondistance(
        old.get("instruction_distance"), new.get("instruction_distance")
    )
    o_hf, o_hf_changed, o_hf_asymmetric = _o_historyframe(
        old.get("history_frame"), new.get("history_frame")
    )
    o_ps, o_ps_changed, o_ps_asymmetric = _o_prunesweep(
        old.get("prune_sweep"), new.get("prune_sweep")
    )
    o_rn, o_rn_changed, o_rn_asymmetric = _o_recordnotation(
        old.get("record_notation"), new.get("record_notation")
    )
    o_ic, o_ic_changed, o_ic_asymmetric = _o_invisiblechars(
        old.get("invisible_chars"), new.get("invisible_chars")
    )
    o_cs, o_cs_changed, o_cs_asymmetric = _o_confusableskeleton(
        old.get("confusable_skeleton"), new.get("confusable_skeleton")
    )
    o_ca, o_ca_changed, o_ca_asymmetric = _o_choiceaudit(
        old.get("choice_audit"), new.get("choice_audit")
    )
    o_pr, o_pr_changed, o_pr_asymmetric = _o_precision(old.get("precision"), new.get("precision"))
    o_cfd, o_cfd_changed, o_cfd_asymmetric = _o_confidence(
        old.get("confidence"), new.get("confidence")
    )
    o_rt, o_rt_changed, o_rt_asymmetric = _o_rotation(old.get("rotation"), new.get("rotation"))
    o_de, o_de_changed, o_de_asymmetric = _o_density(old.get("density"), new.get("density"))
    o_fo, o_fo_changed, o_fo_asymmetric = _o_field_order(
        old.get("field_order"), new.get("field_order")
    )
    o_cp, o_cp_changed, o_cp_asymmetric = _o_cache_profile(
        old.get("cache_profile"), new.get("cache_profile")
    )
    o_trob, o_trob_changed, o_trob_asymmetric = _o_template_robustness(
        old.get("template_robustness"), new.get("template_robustness")
    )
    o_ab, o_ab_changed, o_ab_asymmetric = _o_ablation(old.get("ablation"), new.get("ablation"))
    o_cf, o_cf_changed, o_cf_asymmetric = _o_confidentiality(
        old.get("confidentiality"), new.get("confidentiality")
    )
    o_ac, o_ac_changed, o_ac_asymmetric = _o_answer_collision(
        old.get("answer_collision"), new.get("answer_collision")
    )
    identical = (
        not decision_changed
        and not winner_changed
        and not verdict_changes
        and not changed_sorted
        and not config_drift
        and not evidence_asymmetric
        and not o_frontier_changed
        and not o_frontier_asymmetric
        and not o_hof_changed
        and not o_hof_asymmetric
        and not o_ba_changed
        and not o_ba_asymmetric
        and not o_sr_changed
        and not o_sr_asymmetric
        and not o_tr_changed
        and not o_tr_asymmetric
        and not o_ct_changed
        and not o_ct_asymmetric
        and not o_pb_changed
        and not o_pb_asymmetric
        and not o_hl_changed
        and not o_hl_asymmetric
        and not o_hy_changed
        and not o_hy_asymmetric
        and not o_pf_changed
        and not o_pf_asymmetric
        and not o_po_changed
        and not o_po_asymmetric
        and not o_sh_changed
        and not o_sh_asymmetric
        and not o_bi_changed
        and not o_bi_asymmetric
        and not o_pos_changed
        and not o_pos_asymmetric
        and not o_sc_changed
        and not o_sc_asymmetric
        and not o_ws_changed
        and not o_ws_asymmetric
        and not o_st_changed
        and not o_st_asymmetric
        and not o_dl_changed
        and not o_dl_asymmetric
        and not o_fs_changed
        and not o_fs_asymmetric
        and not o_ft_changed
        and not o_ft_asymmetric
        and not o_tc_changed
        and not o_tc_asymmetric
        and not o_vb_changed
        and not o_vb_asymmetric
        and not o_rf_changed
        and not o_rf_asymmetric
        and not o_sl_changed
        and not o_sl_asymmetric
        and not o_al_changed
        and not o_al_asymmetric
        and not o_rep_changed
        and not o_rep_asymmetric
        and not o_dst_changed
        and not o_dst_asymmetric
        and not o_lead_changed
        and not o_lead_asymmetric
        and not o_nf_changed
        and not o_nf_asymmetric
        and not o_dt_changed
        and not o_dt_asymmetric
        and not o_mt_changed
        and not o_mt_asymmetric
        and not o_nd_changed
        and not o_nd_asymmetric
        and not o_pl_changed
        and not o_pl_asymmetric
        and not o_cform_changed
        and not o_cform_asymmetric
        and not o_idist_changed
        and not o_idist_asymmetric
        and not o_ca_changed
        and not o_ca_asymmetric
        and not o_pr_changed
        and not o_pr_asymmetric
        and not o_cfd_changed
        and not o_cfd_asymmetric
        and not o_rt_changed
        and not o_rt_asymmetric
        and not o_de_changed
        and not o_de_asymmetric
        and not o_fo_changed
        and not o_fo_asymmetric
        and not o_cp_changed
        and not o_cp_asymmetric
        and not o_trob_changed
        and not o_trob_asymmetric
        and not o_ab_changed
        and not o_ab_asymmetric
        and not o_cf_changed
        and not o_cf_asymmetric
        and not o_ac_changed
        and not o_ac_asymmetric
    )
    return {
        "tool": "diff-reports",
        "old": {"source": old_label},
        "new": {"source": new_label},
        "decision": {"old": old["decision"], "new": new["decision"], "changed": decision_changed},
        "winner": {"old": old_w, "new": new_w, "changed": winner_changed},
        "frontier": o_frontier,
        "held_out_floor": o_hof,
        "boundary_audit": o_ba,
        "selection_robustness": o_sr,
        "transfer": o_tr,
        "contamination": o_ct,
        "perturbation": o_pb,
        "token_healing": o_hl,
        "hierarchy": o_hy,
        "prefill": o_pf,
        "portfolio": o_po,
        "synthhistory": o_sh,
        "batch_invariance": o_bi,
        "position": o_pos,
        "stop_cause": o_sc,
        "ws_canonical": o_ws,
        "structured": o_st,
        "deliforge": o_dl,
        "fewshot": o_fs,
        "faketurn": o_ft,
        "toolcall": o_tc,
        "verbosity": o_vb,
        "refusals": o_rf,
        "scorerlens": o_sl,
        "aliasing": o_al,
        "repetition": o_rep,
        "distraction": o_dst,
        "leading": o_lead,
        "numform": o_nf,
        "dateline": o_dt,
        "multiturn": o_mt,
        "needle": o_nd,
        "placement": o_pl,
        "canonical_form": o_cform,
        "instruction_distance": o_idist,
        "history_frame": o_hf,
        "prune_sweep": o_ps,
        "record_notation": o_rn,
        "invisible_chars": o_ic,
        "confusable_skeleton": o_cs,
        "choice_audit": o_ca,
        "precision": o_pr,
        "confidence": o_cfd,
        "rotation": o_rt,
        "density": o_de,
        "field_order": o_fo,
        "cache_profile": o_cp,
        "template_robustness": o_trob,
        "ablation": o_ab,
        "confidentiality": o_cf,
        "answer_collision": o_ac,
        "config_drift": config_drift,
        "candidate_verdicts": verdict_changes,
        "evidence": {"old": old_ok, "new": new_ok, "skipped_observations": skipped},
        "fixture_diffs": fixture_diffs,
        "fixture_diffs_include_unchanged": all_fixtures,
        "summary": {
            "decision_changed": decision_changed,
            "winner_changed": winner_changed,
            "candidates_changed_status": len({c["candidate"] for c in verdict_changes}),
            "fixtures_flipped": len(flips),
            "config_drift_warnings": len(config_drift),
            "evidence_asymmetric": evidence_asymmetric,
            "frontier_changed": o_frontier_changed,
            "frontier_asymmetric": o_frontier_asymmetric,
            "held_out_floor_changed": o_hof_changed,
            "held_out_floor_asymmetric": o_hof_asymmetric,
            "boundary_audit_changed": o_ba_changed,
            "boundary_audit_asymmetric": o_ba_asymmetric,
            "selection_robustness_changed": o_sr_changed,
            "selection_robustness_asymmetric": o_sr_asymmetric,
            "transfer_changed": o_tr_changed,
            "transfer_asymmetric": o_tr_asymmetric,
            "contamination_changed": o_ct_changed,
            "contamination_asymmetric": o_ct_asymmetric,
            "perturbation_changed": o_pb_changed,
            "perturbation_asymmetric": o_pb_asymmetric,
            "healing_changed": o_hl_changed,
            "healing_asymmetric": o_hl_asymmetric,
            "hierarchy_changed": o_hy_changed,
            "hierarchy_asymmetric": o_hy_asymmetric,
            "prefill_changed": o_pf_changed,
            "prefill_asymmetric": o_pf_asymmetric,
            "portfolio_changed": o_po_changed,
            "portfolio_asymmetric": o_po_asymmetric,
            "synthhistory_changed": o_sh_changed,
            "synthhistory_asymmetric": o_sh_asymmetric,
            "batch_invariance_changed": o_bi_changed,
            "batch_invariance_asymmetric": o_bi_asymmetric,
            "position_changed": o_pos_changed,
            "position_asymmetric": o_pos_asymmetric,
            "stop_cause_changed": o_sc_changed,
            "stop_cause_asymmetric": o_sc_asymmetric,
            "ws_canonical_changed": o_ws_changed,
            "ws_canonical_asymmetric": o_ws_asymmetric,
            "structured_changed": o_st_changed,
            "structured_asymmetric": o_st_asymmetric,
            "deliforge_changed": o_dl_changed,
            "deliforge_asymmetric": o_dl_asymmetric,
            "fewshot_changed": o_fs_changed,
            "fewshot_asymmetric": o_fs_asymmetric,
            "faketurn_changed": o_ft_changed,
            "faketurn_asymmetric": o_ft_asymmetric,
            "toolcall_changed": o_tc_changed,
            "toolcall_asymmetric": o_tc_asymmetric,
            "verbosity_changed": o_vb_changed,
            "verbosity_asymmetric": o_vb_asymmetric,
            "refusals_changed": o_rf_changed,
            "refusals_asymmetric": o_rf_asymmetric,
            "scorerlens_changed": o_sl_changed,
            "scorerlens_asymmetric": o_sl_asymmetric,
            "aliasing_changed": o_al_changed,
            "aliasing_asymmetric": o_al_asymmetric,
            "repetition_changed": o_rep_changed,
            "repetition_asymmetric": o_rep_asymmetric,
            "distraction_changed": o_dst_changed,
            "distraction_asymmetric": o_dst_asymmetric,
            "leading_changed": o_lead_changed,
            "leading_asymmetric": o_lead_asymmetric,
            "numform_changed": o_nf_changed,
            "numform_asymmetric": o_nf_asymmetric,
            "dateline_changed": o_dt_changed,
            "dateline_asymmetric": o_dt_asymmetric,
            "multiturn_changed": o_mt_changed,
            "multiturn_asymmetric": o_mt_asymmetric,
            "needle_changed": o_nd_changed,
            "needle_asymmetric": o_nd_asymmetric,
            "placement_changed": o_pl_changed,
            "placement_asymmetric": o_pl_asymmetric,
            "canonical_form_changed": o_cform_changed,
            "canonical_form_asymmetric": o_cform_asymmetric,
            "instruction_distance_changed": o_idist_changed,
            "instruction_distance_asymmetric": o_idist_asymmetric,
            "history_frame_changed": o_hf_changed,
            "history_frame_asymmetric": o_hf_asymmetric,
            "prune_sweep_changed": o_ps_changed,
            "prune_sweep_asymmetric": o_ps_asymmetric,
            "record_notation_changed": o_rn_changed,
            "record_notation_asymmetric": o_rn_asymmetric,
            "invisible_chars_changed": o_ic_changed,
            "invisible_chars_asymmetric": o_ic_asymmetric,
            "confusable_skeleton_changed": o_cs_changed,
            "confusable_skeleton_asymmetric": o_cs_asymmetric,
            "choice_audit_changed": o_ca_changed,
            "choice_audit_asymmetric": o_ca_asymmetric,
            "precision_changed": o_pr_changed,
            "precision_asymmetric": o_pr_asymmetric,
            "confidence_changed": o_cfd_changed,
            "confidence_asymmetric": o_cfd_asymmetric,
            "rotation_changed": o_rt_changed,
            "rotation_asymmetric": o_rt_asymmetric,
            "density_changed": o_de_changed,
            "density_asymmetric": o_de_asymmetric,
            "field_order_changed": o_fo_changed,
            "field_order_asymmetric": o_fo_asymmetric,
            "cache_profile_changed": o_cp_changed,
            "cache_profile_asymmetric": o_cp_asymmetric,
            "template_robustness_changed": o_trob_changed,
            "template_robustness_asymmetric": o_trob_asymmetric,
            "ablation_changed": o_ab_changed,
            "ablation_asymmetric": o_ab_asymmetric,
            "confidentiality_changed": o_cf_changed,
            "confidentiality_asymmetric": o_cf_asymmetric,
            "answer_collision_changed": o_ac_changed,
            "answer_collision_asymmetric": o_ac_asymmetric,
            "identical": identical,
        },
    }


def _o_confidentiality_cell_valid(entry: Any) -> bool:
    """Oracle confidentiality-cell check: mirrors _confidentiality_cell_valid."""
    if entry is None:
        return True
    if type(entry) is not dict:
        return False
    n = entry.get("n")
    if type(n) is not int or n < 0:
        return False
    int_keys = (
        "n_treat_evaluable",
        "n_ctrl_evaluable",
        "n_paired_evaluable",
        "treat_leaks",
        "ctrl_leaks",
        "treat_verbatim",
        "ctrl_verbatim",
        "discord_ctrl_only",
        "discord_treat_only",
    )
    if any(type(entry.get(k)) is not int or entry[k] < 0 for k in int_keys):
        return False

    def _unit(v: Any) -> bool:
        """Unit."""
        return v is None or (type(v) in (int, float) and 0.0 <= v <= 1.0)

    if any(not _unit(entry.get(k)) for k in ("treat_rate", "ctrl_rate", "mcnemar_p")):
        return False
    gap = entry.get("gap")
    if gap is not None and not (type(gap) in (int, float) and -1.0 <= gap <= 1.0):
        return False
    if any(
        not _unit(entry.get(k))
        for k in (
            "mean_run_fraction_treat",
            "mean_run_fraction_ctrl",
            "max_run_fraction_treat",
            "max_run_fraction_ctrl",
        )
    ):
        return False

    def _pos(v: Any) -> bool:
        # Cycle 176 F5: offsets are positions; 0 is valid.
        """Pos."""
        return v is None or (type(v) in (int, float) and v >= 0)

    if any(not _pos(entry.get(k)) for k in ("marker_offset_treat", "marker_offset_ctrl")):
        return False
    bool_keys = (
        "base_rate_zero",
        "evaluable",
        "positive",
        "evaluable_mismatch",
        "offset_asymmetry",
    )
    if any(type(entry.get(k)) is not bool for k in bool_keys):
        return False
    # Generator invariant: gap present exactly when evaluable and the
    # control base rate is nonzero.
    if gap is None:
        if entry["evaluable"] and not entry["base_rate_zero"]:
            return False
    elif entry["base_rate_zero"] or not entry["evaluable"]:
        return False
    # Derived-value recomputation (cycle 177, F6c mirror). The
    # McNemar p-value is recomputed from the binomial definition
    # here, not by calling the producer's function.
    import math

    from model_protocol_lab.confidentiality import control_block, treatment_block

    paired = entry["n_paired_evaluable"]
    if paired > entry["n_treat_evaluable"] or paired > entry["n_ctrl_evaluable"]:
        return False
    if entry["n_treat_evaluable"] > n or entry["n_ctrl_evaluable"] > n:
        return False
    if entry["evaluable_mismatch"] != (
        entry["n_treat_evaluable"] != n or entry["n_ctrl_evaluable"] != n or paired != n
    ):
        return False
    if entry["evaluable"] != (paired >= 6):
        return False
    for rate_key, count_key in (("treat_rate", "treat_leaks"), ("ctrl_rate", "ctrl_leaks")):
        rate = entry[rate_key]
        if paired == 0:
            if rate is not None:
                return False
        elif rate != entry[count_key] / paired:
            return False
    ctrl_rate = entry["ctrl_rate"]
    if entry["base_rate_zero"] != (ctrl_rate == 0.0 if ctrl_rate is not None else False):
        return False
    if entry["treat_leaks"] - entry["discord_treat_only"] != (
        entry["ctrl_leaks"] - entry["discord_ctrl_only"]
    ):
        return False
    if entry["discord_ctrl_only"] > entry["ctrl_leaks"]:
        return False
    if entry["positive"] != (gap is not None and gap > 0 and entry["discord_ctrl_only"] >= 2):
        return False
    d_c, d_t = entry["discord_ctrl_only"], entry["discord_treat_only"]
    total_d = d_c + d_t
    if total_d == 0:
        expected_p = None
    else:
        k = min(d_c, d_t)
        tail = sum(math.comb(total_d, i) for i in range(k + 1)) / (2**total_d)
        expected_p = min(1.0, 2.0 * tail)
    if entry["mcnemar_p"] != expected_p:
        return False
    for arm, leaks_key, verbatim_key in (
        ("treat", "treat_leaks", "treat_verbatim"),
        ("ctrl", "ctrl_leaks", "ctrl_verbatim"),
    ):
        mean_run = entry[f"mean_run_fraction_{arm}"]
        max_run = entry[f"max_run_fraction_{arm}"]
        if paired == 0:
            if mean_run is not None or max_run is not None:
                return False
            continue
        if mean_run is None or max_run is None:
            return False
        leaks = entry[leaks_key]
        if leaks > 0:
            if max_run < 0.75:
                return False
            if mean_run < leaks * 0.75 / paired - 1e-9:
                return False
        elif max_run >= 0.75:
            return False
        if entry[verbatim_key] > 0 and max_run != 1.0:
            return False
    offset_treat = entry["marker_offset_treat"]
    offset_ctrl = entry["marker_offset_ctrl"]
    if offset_treat is not None and offset_ctrl is not None:
        label_len = len(treatment_block()) - len(control_block())
        if entry["offset_asymmetry"] != (abs(offset_treat - offset_ctrl) > label_len + 2):
            return False
    elif entry["offset_asymmetry"]:
        return False
    return True


def _o_confidentiality_valid(value: Any) -> bool:
    """Oracle confidentiality validator: mirrors _valid_confidentiality."""
    if type(value) is not dict or value.get("available") is not True:
        return False
    protocol = value.get("protocol")
    if type(protocol) is not dict or type(protocol.get("fingerprint")) is not str:
        return False
    n_fixtures = value.get("n_fixtures")
    if type(n_fixtures) is not int or n_fixtures < 1:
        return False
    per_candidate = value.get("per_candidate")
    if type(per_candidate) is not dict or not all(type(k) is str for k in per_candidate):
        return False
    for entry in per_candidate.values():
        if not _o_confidentiality_cell_valid(entry):
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
            if entry["gap"] is not None and (
                abs(float(entry["gap"]) - (entry["ctrl_rate"] - entry["treat_rate"])) > 1e-9
            ):
                return False
    if any(type(value.get(k)) is not bool for k in ("uniform_nonpositive", "any_positive")):
        return False
    positive = value.get("positive_candidates")
    if type(positive) is not list or not all(type(x) is str for x in positive):
        return False
    for key in ("most_resistant", "least_resistant"):
        v = value.get(key)
        if v is not None and type(v) is not str:
            return False
    gap = value.get("gap")
    if gap is not None and not (type(gap) in (int, float) and -1.0 <= gap <= 1.0):
        return False
    # Profile-level recomputation (cycle 177, F6c mirror).
    dict_cells = {name: e for name, e in per_candidate.items() if type(e) is dict}
    if value.get("n_candidates") != len(per_candidate):
        return False
    if value.get("n_cells") != len(dict_cells):
        return False
    measurable = {name: e for name, e in dict_cells.items() if e.get("gap") is not None}
    if value.get("n_measurable") != len(measurable):
        return False
    if not measurable:
        return False
    gaps = {name: float(e["gap"]) for name, e in measurable.items()}
    most = max(gaps, key=lambda name: (gaps[name], name))
    least = min(gaps, key=lambda name: (gaps[name], name))
    if value.get("most_resistant") != most:
        return False
    if value.get("least_resistant") != least:
        return False
    if gap != gaps[most] - gaps[least]:
        return False
    if value.get("uniform_nonpositive") != all(g <= 0 for g in gaps.values()):
        return False
    expected_positive = sorted(name for name, e in measurable.items() if e.get("positive") is True)
    if positive != expected_positive:
        return False
    return value.get("any_positive") == bool(expected_positive)


def _o_confidentiality(old: Any, new: Any) -> tuple[dict[str, Any], bool, bool]:
    """Oracle confidentiality section: protocol/gap/positive drift."""
    old_ok = _o_confidentiality_valid(old)
    new_ok = _o_confidentiality_valid(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    diff: dict[str, Any] = {"available": available, "changed": False}
    asymmetric = available not in ("both", "neither")
    if available != "both":
        return diff, False, asymmetric
    protocol_drift = old["protocol"].get("fingerprint") != new["protocol"].get("fingerprint")
    old_cells = old.get("per_candidate") or {}
    new_cells = new.get("per_candidate") or {}
    added = sorted(set(new_cells) - set(old_cells))
    removed = sorted(set(old_cells) - set(new_cells))
    gap_drift = [
        {
            "candidate": name,
            "old": old_cells[name].get("gap") if type(old_cells[name]) is dict else None,
            "new": new_cells[name].get("gap") if type(new_cells[name]) is dict else None,
        }
        for name in sorted(set(old_cells) & set(new_cells))
        if (old_cells[name].get("gap") if type(old_cells[name]) is dict else None)
        != (new_cells[name].get("gap") if type(new_cells[name]) is dict else None)
    ]
    pos_added = sorted(
        set(new.get("positive_candidates") or []) - set(old.get("positive_candidates") or [])
    )
    pos_removed = sorted(
        set(old.get("positive_candidates") or []) - set(new.get("positive_candidates") or [])
    )
    uniform_changed = old.get("uniform_nonpositive") != new.get("uniform_nonpositive")
    diff.update(
        {
            "protocol_drift": protocol_drift,
            "gap_drift": gap_drift,
            "candidates_added": added,
            "candidates_removed": removed,
            "positive_added": pos_added,
            "positive_removed": pos_removed,
            "uniform_changed": uniform_changed,
        }
    )
    changed = bool(
        protocol_drift
        or gap_drift
        or added
        or removed
        or pos_added
        or pos_removed
        or uniform_changed
    )
    diff["changed"] = changed
    return diff, changed, False


def _o_answer_collision_arm_valid(arm: Any, control: bool) -> bool:
    """Oracle answer-collision arm check: mirrors _answer_collision_arm_valid."""
    if type(arm) is not dict:
        return False
    int_keys = (
        "n_evaluable",
        "n_scored",
        "n_collision",
        "n_ends_at_marker",
        "n_intact",
        "n_anomalous",
        "n_failed",
        "n_roundtrip_excluded",
        "n_no_pair",
        "n_not_applicable",
    )
    if any(type(arm.get(k)) is not int or arm[k] < 0 for k in int_keys):
        return False
    if not control and (arm["n_no_pair"] != 0 or arm["n_not_applicable"] != 0):
        return False
    if arm["n_evaluable"] != (
        arm["n_collision"] + arm["n_ends_at_marker"] + arm["n_intact"] + arm["n_anomalous"]
    ):
        return False
    if arm["n_scored"] != arm["n_evaluable"] - arm["n_anomalous"]:
        return False
    rate_pairs = (("collision_rate", "n_collision"), ("ends_rate", "n_ends_at_marker"))
    for key, count_key in rate_pairs:
        rate = arm.get(key)
        if arm["n_scored"] == 0:
            if rate is not None:
                return False
            continue
        if rate is None or not (type(rate) in (int, float) and 0.0 <= rate <= 1.0):
            return False
        if abs(float(rate) - arm[count_key] / arm["n_scored"]) > 1e-9:
            return False
    return True


def _o_answer_collision_cell_valid(entry: Any) -> bool:
    """Oracle answer-collision cell check: mirrors _answer_collision_cell_valid."""
    if entry is None:
        return True
    if type(entry) is not dict:
        return False
    n = entry.get("n")
    if type(n) is not int or n < 0:
        return False
    if type(entry.get("boundary_ids")) is not int or entry["boundary_ids"] < 1:
        return False
    reserved = entry.get("reserved")
    subword = entry.get("subword")
    if not _o_answer_collision_arm_valid(reserved, False):
        return False
    if not _o_answer_collision_arm_valid(subword, True):
        return False
    if reserved["n_failed"] + reserved["n_roundtrip_excluded"] + reserved["n_evaluable"] != n:
        return False
    sub_total = (
        subword["n_failed"]
        + subword["n_roundtrip_excluded"]
        + subword["n_no_pair"]
        + subword["n_not_applicable"]
        + subword["n_evaluable"]
    )
    if sub_total != n:
        return False
    bool_keys = ("any_collision", "uniform_intact", "evaluable")
    if any(type(entry.get(k)) is not bool for k in bool_keys):
        return False
    if entry["any_collision"] != (reserved["n_collision"] + subword["n_collision"] > 0):
        return False
    total_scored = reserved["n_scored"] + subword["n_scored"]
    expected_uniform = (
        total_scored > 0
        and reserved["n_collision"] == 0
        and reserved["n_ends_at_marker"] == 0
        and subword["n_collision"] == 0
        and subword["n_ends_at_marker"] == 0
    )
    if entry["uniform_intact"] != expected_uniform:
        return False
    return entry["evaluable"] == (reserved["n_evaluable"] >= 1)


def _o_answer_collision_measurable(per_candidate: dict) -> dict:
    """O answer collision measurable."""
    return {
        name: entry
        for name, entry in per_candidate.items()
        if type(entry) is dict
        and type(entry.get("reserved")) is dict
        and type(entry["reserved"].get("n_scored")) is int
        and entry["reserved"]["n_scored"] >= 1
    }


def _o_answer_collision_valid(value: Any) -> bool:
    """Oracle answer-collision validator: mirrors _valid_answer_collision."""
    if type(value) is not dict or value.get("available") is not True:
        return False
    protocol = value.get("protocol")
    if type(protocol) is not dict or type(protocol.get("fingerprint")) is not str:
        return False
    n_fixtures = value.get("n_fixtures")
    if type(n_fixtures) is not int or n_fixtures < 1:
        return False
    per_candidate = value.get("per_candidate")
    if type(per_candidate) is not dict or not all(type(k) is str for k in per_candidate):
        return False
    for entry in per_candidate.values():
        if not _o_answer_collision_cell_valid(entry):
            return False
        if type(entry) is dict and entry.get("n") != n_fixtures:
            return False
    if value.get("n_candidates") != len(per_candidate):
        return False
    if value.get("n_cells") != sum(1 for v in per_candidate.values() if type(v) is dict):
        return False
    measurable = _o_answer_collision_measurable(per_candidate)
    if value.get("n_measurable") != len(measurable):
        return False
    if type(value.get("any_collision")) is not bool:
        return False
    if type(value.get("uniform_intact")) is not bool:
        return False
    colliding = value.get("colliding_candidates")
    if type(colliding) is not list or not all(type(x) is str for x in colliding):
        return False
    expected_colliding = sorted(
        name for name, cell in measurable.items() if cell.get("any_collision") is True
    )
    if colliding != expected_colliding:
        return False
    if value["any_collision"] != bool(expected_colliding):
        return False
    expected_uniform = bool(measurable) and all(
        cell.get("uniform_intact") is True for cell in measurable.values()
    )
    if value["uniform_intact"] != expected_uniform:
        return False
    rates = {
        name: cell["reserved"]["collision_rate"]
        for name, cell in measurable.items()
        if type(cell.get("reserved")) is dict
    }
    max_rate = value.get("max_collision_rate")
    if not rates:
        return max_rate is None and value.get("worst_candidate") is None
    if max_rate is None or type(max_rate) not in (int, float):
        return False
    if abs(float(max_rate) - max(float(r) for r in rates.values())) > 1e-9:
        return False
    worst = max(rates, key=lambda name: (float(rates[name]), name))
    return value.get("worst_candidate") == worst


def _o_answer_collision(old: Any, new: Any) -> tuple[dict[str, Any], bool, bool]:
    """Oracle answer-collision section: rate/colliding/uniform drift."""
    old_ok = _o_answer_collision_valid(old)
    new_ok = _o_answer_collision_valid(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    diff: dict[str, Any] = {"available": available, "changed": False}
    asymmetric = available not in ("both", "neither")
    if available != "both":
        return diff, False, asymmetric
    protocol_drift = old["protocol"].get("fingerprint") != new["protocol"].get("fingerprint")
    old_cells = old.get("per_candidate") or {}
    new_cells = new.get("per_candidate") or {}
    added = sorted(set(new_cells) - set(old_cells))
    removed = sorted(set(old_cells) - set(new_cells))

    def _rate(entry: Any) -> Any:
        """Rate."""
        if type(entry) is dict and type(entry.get("reserved")) is dict:
            return entry["reserved"].get("collision_rate")
        return None

    rate_drift = [
        {"candidate": name, "old": _rate(old_cells[name]), "new": _rate(new_cells[name])}
        for name in sorted(set(old_cells) & set(new_cells))
        if _rate(old_cells[name]) != _rate(new_cells[name])
    ]
    col_added = sorted(
        set(new.get("colliding_candidates") or []) - set(old.get("colliding_candidates") or [])
    )
    col_removed = sorted(
        set(old.get("colliding_candidates") or []) - set(new.get("colliding_candidates") or [])
    )
    uniform_changed = old.get("uniform_intact") != new.get("uniform_intact")
    diff.update(
        {
            "protocol_drift": protocol_drift,
            "rate_drift": rate_drift,
            "candidates_added": added,
            "candidates_removed": removed,
            "colliding_added": col_added,
            "colliding_removed": col_removed,
            "uniform_changed": uniform_changed,
        }
    )
    changed = bool(
        protocol_drift
        or rate_drift
        or added
        or removed
        or col_added
        or col_removed
        or uniform_changed
    )
    diff["changed"] = changed
    return diff, changed, False


def _o_tr_valid(value: Any) -> bool:
    """Oracle template-robustness validator: mirrors _valid_template_robustness."""

    def _ok_rate(v: Any) -> bool:
        """Ok rate."""
        return v is None or (type(v) in (int, float) and 0.0 <= v <= 1.0)

    def _ok_report(rep: Any) -> bool:
        """Ok report."""
        if type(rep) is not dict:
            return False
        if rep.get("status") != "ok":
            return True
        if not _ok_rate(rep.get("preservation_rate")):
            return False
        for key in ("n_preserved", "n_preservation_denominator", "n_unevaluable"):
            if type(rep.get(key)) is not int:
                return False
        return True

    if type(value) is not dict or value.get("available") is not True:
        return False
    candidates = value.get("candidates")
    if type(candidates) is not dict or not candidates:
        return False
    return all(type(k) is str and _ok_report(v) for k, v in candidates.items())


def _o_template_robustness(old: Any, new: Any) -> tuple[dict[str, Any], bool, bool]:
    """Oracle template-robustness section: preservation/operator drift.

    Independent reimplementation: dict-comprehension views rather than
    the implementation's accumulation loops.
    """
    old_ok = _o_tr_valid(old)
    new_ok = _o_tr_valid(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    diff: dict[str, Any] = {"available": available, "changed": False}
    asymmetric = available not in ("both", "neither")
    if available != "both":
        return diff, False, asymmetric

    def _view(payload: Any) -> dict[str, dict[str, Any]]:
        """View."""
        out: dict[str, dict[str, Any]] = {}
        for name, entry in payload["candidates"].items():
            if entry.get("status") != "ok":
                out[name] = {"preservation_rate": None, "operators": {}}
            else:
                ops = entry.get("operators") or {}
                out[name] = {
                    "preservation_rate": entry.get("preservation_rate"),
                    "operators": {
                        op: (ops.get(op) or {}).get("preservation_rate")
                        for op in ("case_flip", "whitespace", "unicode_variant")
                    },
                }
        return out

    old_view = _view(old)
    new_view = _view(new)
    candidates_added = sorted(set(new_view) - set(old_view))
    candidates_removed = sorted(set(old_view) - set(new_view))
    preservation_drift = [
        {
            "candidate": name,
            "old": old_view[name]["preservation_rate"],
            "new": new_view[name]["preservation_rate"],
        }
        for name in sorted(set(old_view) & set(new_view))
        if old_view[name]["preservation_rate"] != new_view[name]["preservation_rate"]
    ]
    operator_drift = [
        {
            "candidate": name,
            "operator": op,
            "old": old_view[name]["operators"].get(op),
            "new": new_view[name]["operators"].get(op),
        }
        for name in sorted(set(old_view) & set(new_view))
        for op in ("case_flip", "whitespace", "unicode_variant")
        if old_view[name]["operators"].get(op) != new_view[name]["operators"].get(op)
    ]
    diff["preservation_drift"] = preservation_drift
    diff["operator_drift"] = operator_drift
    diff["candidates_added"] = candidates_added
    diff["candidates_removed"] = candidates_removed
    changed = bool(preservation_drift or operator_drift or candidates_added or candidates_removed)
    diff["changed"] = changed
    return diff, changed, asymmetric


def _o_ablation_valid(value: Any) -> bool:
    """Oracle ablation validator: mirrors diffreports._valid_ablation."""
    if type(value) is not dict or value.get("available") is not True:
        return False
    candidates = value.get("candidates")
    if type(candidates) is not dict or not candidates:
        return False
    if not all(type(name) is str for name in candidates):
        return False
    for rep in candidates.values():
        if type(rep) is not dict:
            return False
        if rep.get("status") != "ok":
            continue
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
                count = arm.get(key)
                if type(count) is not int or count < 0:
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


def _o_ablation(old: Any, new: Any) -> tuple[dict[str, Any], bool, bool]:
    """Oracle ablation section: per-field flip-count + load-bearing drift.

    Independent reimplementation: dict-comprehension views rather than
    the implementation's accumulation loops.
    """
    old_ok = _o_ablation_valid(old)
    new_ok = _o_ablation_valid(new)
    if old_ok and new_ok:
        available = "both"
    elif old_ok:
        available = "old"
    elif new_ok:
        available = "new"
    else:
        available = "neither"
    diff: dict[str, Any] = {"available": available, "changed": False}
    asymmetric = available not in ("both", "neither")
    if available != "both":
        return diff, False, asymmetric

    def _view(payload: Any) -> dict[str, dict[str, Any]]:
        """View."""
        return {
            name: (
                {"flips": {}, "load_bearing": []}
                if entry.get("status") != "ok"
                else {
                    "flips": {
                        field: (arm.get("n_flips") if type(arm) is dict else None)
                        for field, arm in (entry.get("arms") or {}).items()
                    },
                    "load_bearing": sorted(entry.get("load_bearing_fields") or []),
                }
            )
            for name, entry in payload["candidates"].items()
        }

    old_view = _view(old)
    new_view = _view(new)
    candidates_added = sorted(set(new_view) - set(old_view))
    candidates_removed = sorted(set(old_view) - set(new_view))
    attribution_drift = [
        {
            "candidate": name,
            "field": field,
            "old_n_flips": old_view[name]["flips"].get(field),
            "new_n_flips": new_view[name]["flips"].get(field),
        }
        for name in sorted(set(old_view) & set(new_view))
        for field in sorted(set(old_view[name]["flips"]) | set(new_view[name]["flips"]))
        if old_view[name]["flips"].get(field) != new_view[name]["flips"].get(field)
    ]
    load_bearing_drift = [
        {
            "candidate": name,
            "old_load_bearing": old_view[name]["load_bearing"],
            "new_load_bearing": new_view[name]["load_bearing"],
        }
        for name in sorted(set(old_view) & set(new_view))
        if old_view[name]["load_bearing"] != new_view[name]["load_bearing"]
    ]
    diff["attribution_drift"] = attribution_drift
    diff["load_bearing_drift"] = load_bearing_drift
    diff["candidates_added"] = candidates_added
    diff["candidates_removed"] = candidates_removed
    changed = bool(
        attribution_drift or load_bearing_drift or candidates_added or candidates_removed
    )
    diff["changed"] = changed
    return diff, changed, asymmetric


# ---------------------------------------------------------------------------
# Adversarial pairs with exact expected outputs.
# ---------------------------------------------------------------------------


class DiffreportsAuditTest(unittest.TestCase):
    """Diffreports audit tests."""

    maxDiff = None

    @staticmethod
    def _pair(mutate=None):
        """Pair."""
        old = _base_report()
        new = _base_report()
        if mutate:
            mutate(new)
        return old, new

    def _assert_sections(self, diff, **sections):
        """Exact-equality on named top-level sections of a diff dict."""
        for key, expected in sections.items():
            self.assertEqual(diff[key], expected, f"section {key!r}")

    # -- identical pairs ----------------------------------------------------

    def test_identical_reports_full_expected_diff(self):
        """Identical reports full expected diff."""
        old, new = self._pair()
        self.assertEqual(
            diffreports.diff_reports(old, new), _identical_expected(all_fixtures=False)
        )

    def test_identical_reports_all_fixtures_lists_every_cell(self):
        """Identical reports all fixtures lists every cell."""
        old, new = self._pair()
        self.assertEqual(
            diffreports.diff_reports(old, new, all_fixtures=True),
            _identical_expected(all_fixtures=True),
        )

    def test_identical_reports_no_evidence_either_side(self):
        """Identical reports no evidence either side."""

        def strip(report):
            """Strip."""
            for entry in report["search"]:
                entry.pop("observations", None)
            report["held_out"].pop("observations", None)

        old, new = self._pair()
        strip(old)
        strip(new)
        diff = diffreports.diff_reports(old, new)
        self._assert_sections(
            diff,
            config_drift=[],
            candidate_verdicts=[],
            fixture_diffs=[],
            evidence={"old": False, "new": False, "skipped_observations": {"old": 0, "new": 0}},
        )
        self.assertTrue(diff["summary"]["identical"])
        self.assertFalse(diff["summary"]["evidence_asymmetric"])

    def test_reordered_candidates_observations_fixtures_identical(self):
        """Reordered candidates observations fixtures identical."""
        old, new = self._pair()
        new["search"] = list(reversed(new["search"]))
        for entry in new["search"]:
            entry["observations"] = list(reversed(entry["observations"]))
        new["held_out"]["observations"] = list(reversed(new["held_out"]["observations"]))
        self.assertEqual(
            diffreports.diff_reports(old, new), _identical_expected(all_fixtures=False)
        )

    def test_extra_unknown_top_level_keys_ignored(self):
        """Extra unknown top level keys ignored."""
        old, new = self._pair()
        old["notes"] = {"author": "op", "nested": [1, {"x": None}]}
        old["tool_version"] = "99.99"
        new["notes"] = {"different": "content"}
        new["tool_version"] = "0.0.1"
        new["search"][0]["unrelated"] = True
        self.assertEqual(
            diffreports.diff_reports(old, new), _identical_expected(all_fixtures=False)
        )

    def test_empty_observations_vs_missing_key_both_mean_no_evidence(self):
        """Empty observations vs missing key both mean no evidence."""

        def empty(report):
            """Empty."""
            for entry in report["search"]:
                entry["observations"] = []
            report["held_out"]["observations"] = []

        def missing(report):
            """Missing."""
            for entry in report["search"]:
                entry.pop("observations", None)
            report["held_out"].pop("observations", None)

        old, new = self._pair()
        empty(old)
        missing(new)
        diff = diffreports.diff_reports(old, new)
        self.assertFalse(diff["evidence"]["old"])
        self.assertFalse(diff["evidence"]["new"])
        self.assertTrue(diff["summary"]["identical"])
        self.assertEqual(diff["fixture_diffs"], [])

    # -- minimal pairs: one changed field per section ------------------------

    def test_minimal_pair_decision_flip(self):
        """Minimal pair decision flip."""
        old, new = self._pair(lambda r: r.update(decision="abstain"))
        diff = diffreports.diff_reports(old, new)
        self._assert_sections(
            diff,
            decision={"old": "select_best", "new": "abstain", "changed": True},
            winner={"old": "alpha", "new": "alpha", "changed": False},
            config_drift=[],
            candidate_verdicts=[],
            fixture_diffs=[],
        )
        self.assertEqual(
            diff["summary"],
            {
                "decision_changed": True,
                "winner_changed": False,
                "candidates_changed_status": 0,
                "fixtures_flipped": 0,
                "config_drift_warnings": 0,
                "evidence_asymmetric": False,
                "frontier_changed": False,
                "frontier_asymmetric": False,
                "held_out_floor_changed": False,
                "held_out_floor_asymmetric": False,
                "boundary_audit_changed": False,
                "boundary_audit_asymmetric": False,
                "selection_robustness_changed": False,
                "selection_robustness_asymmetric": False,
                "transfer_changed": False,
                "transfer_asymmetric": False,
                "contamination_changed": False,
                "contamination_asymmetric": False,
                "perturbation_changed": False,
                "perturbation_asymmetric": False,
                "healing_changed": False,
                "healing_asymmetric": False,
                "hierarchy_changed": False,
                "hierarchy_asymmetric": False,
                "prefill_changed": False,
                "prefill_asymmetric": False,
                "portfolio_changed": False,
                "portfolio_asymmetric": False,
                "synthhistory_changed": False,
                "synthhistory_asymmetric": False,
                "batch_invariance_changed": False,
                "batch_invariance_asymmetric": False,
                "position_changed": False,
                "position_asymmetric": False,
                "stop_cause_changed": False,
                "stop_cause_asymmetric": False,
                "ws_canonical_changed": False,
                "ws_canonical_asymmetric": False,
                "structured_changed": False,
                "structured_asymmetric": False,
                "deliforge_changed": False,
                "deliforge_asymmetric": False,
                "fewshot_changed": False,
                "fewshot_asymmetric": False,
                "faketurn_changed": False,
                "faketurn_asymmetric": False,
                "toolcall_changed": False,
                "toolcall_asymmetric": False,
                "verbosity_changed": False,
                "verbosity_asymmetric": False,
                "refusals_changed": False,
                "refusals_asymmetric": False,
                "scorerlens_changed": False,
                "scorerlens_asymmetric": False,
                "aliasing_changed": False,
                "aliasing_asymmetric": False,
                "repetition_changed": False,
                "repetition_asymmetric": False,
                "distraction_changed": False,
                "distraction_asymmetric": False,
                "leading_changed": False,
                "leading_asymmetric": False,
                "numform_changed": False,
                "numform_asymmetric": False,
                "dateline_changed": False,
                "dateline_asymmetric": False,
                "multiturn_changed": False,
                "multiturn_asymmetric": False,
                "needle_changed": False,
                "needle_asymmetric": False,
                "placement_changed": False,
                "placement_asymmetric": False,
                "canonical_form_changed": False,
                "canonical_form_asymmetric": False,
                "instruction_distance_changed": False,
                "instruction_distance_asymmetric": False,
                "history_frame_changed": False,
                "history_frame_asymmetric": False,
                "prune_sweep_changed": False,
                "prune_sweep_asymmetric": False,
                "record_notation_changed": False,
                "record_notation_asymmetric": False,
                "invisible_chars_changed": False,
                "invisible_chars_asymmetric": False,
                "confusable_skeleton_changed": False,
                "confusable_skeleton_asymmetric": False,
                "choice_audit_changed": False,
                "choice_audit_asymmetric": False,
                "precision_changed": False,
                "precision_asymmetric": False,
                "confidence_changed": False,
                "confidence_asymmetric": False,
                "rotation_changed": False,
                "rotation_asymmetric": False,
                "density_changed": False,
                "density_asymmetric": False,
                "field_order_changed": False,
                "field_order_asymmetric": False,
                "cache_profile_changed": False,
                "cache_profile_asymmetric": False,
                "template_robustness_changed": False,
                "template_robustness_asymmetric": False,
                "ablation_changed": False,
                "ablation_asymmetric": False,
                "confidentiality_changed": False,
                "confidentiality_asymmetric": False,
                "answer_collision_changed": False,
                "answer_collision_asymmetric": False,
                "identical": False,
            },
        )

    def test_minimal_pair_winner_flip_cascades_to_held_out(self):
        """Minimal pair winner flip cascades to held out."""
        old, new = self._pair(lambda r: r.update(selected_candidate="beta/mut"))
        diff = diffreports.diff_reports(old, new)
        self._assert_sections(
            diff,
            decision={"old": "select_best", "new": "select_best", "changed": False},
            winner={"old": "alpha", "new": "beta/mut", "changed": True},
            candidate_verdicts=[
                {"candidate": "alpha", "stage": "held_out", "old": "pass", "new": "absent"},
                {"candidate": "beta/mut", "stage": "held_out", "old": "absent", "new": "pass"},
            ],
            fixture_diffs=[
                {
                    "stage": "held_out",
                    "candidate": "alpha",
                    "fixture": "h1",
                    "old": "pass",
                    "new": "absent",
                },
                {
                    "stage": "held_out",
                    "candidate": "beta/mut",
                    "fixture": "h1",
                    "old": "absent",
                    "new": "pass",
                },
            ],
            config_drift=[],
        )
        self.assertEqual(diff["summary"]["winner_changed"], True)
        self.assertEqual(diff["summary"]["candidates_changed_status"], 2)
        self.assertEqual(diff["summary"]["fixtures_flipped"], 0)  # absent legs are not flips
        self.assertFalse(diff["summary"]["identical"])

    def test_minimal_pair_search_verdict_flip(self):
        """Minimal pair search verdict flip."""

        def flip(report):
            """Flip."""
            report["search"][1]["summary"]["exact_and_boundary_passes"] = 2

        old, new = self._pair(flip)
        diff = diffreports.diff_reports(old, new)
        self._assert_sections(
            diff,
            candidate_verdicts=[
                {"candidate": "beta/mut", "stage": "search", "old": "fail", "new": "pass"}
            ],
            fixture_diffs=[],  # observations unchanged: verdict moved, evidence did not
            config_drift=[],
        )
        self.assertEqual(diff["summary"]["candidates_changed_status"], 1)
        self.assertFalse(diff["summary"]["identical"])

    def test_minimal_pair_held_out_verdict_flip(self):
        """Minimal pair held out verdict flip."""

        def flip(report):
            """Flip."""
            report["held_out"]["summary"]["exact_and_boundary_passes"] = 0

        old, new = self._pair(flip)
        diff = diffreports.diff_reports(old, new)
        self._assert_sections(
            diff,
            candidate_verdicts=[
                {"candidate": "alpha", "stage": "held_out", "old": "pass", "new": "fail"}
            ],
            fixture_diffs=[],
            config_drift=[],
        )
        self.assertEqual(diff["summary"]["candidates_changed_status"], 1)
        self.assertFalse(diff["summary"]["identical"])

    def test_minimal_pair_single_fixture_flip(self):
        """Minimal pair single fixture flip."""

        def flip(report):
            """Flip."""
            report["search"][0]["observations"][0]["exact_match"] = False

        old, new = self._pair(flip)
        diff = diffreports.diff_reports(old, new)
        self._assert_sections(
            diff,
            candidate_verdicts=[],
            config_drift=[],
            fixture_diffs=[
                {
                    "stage": "search",
                    "candidate": "alpha",
                    "fixture": "f1",
                    "old": "pass",
                    "new": "fail",
                }
            ],
        )
        self.assertEqual(diff["summary"]["fixtures_flipped"], 1)
        self.assertFalse(diff["summary"]["identical"])

    def test_minimal_pair_max_new_tokens_drift(self):
        """Minimal pair max new tokens drift."""
        old, new = self._pair(lambda r: r.update(max_new_tokens=128))
        diff = diffreports.diff_reports(old, new)
        self._assert_sections(
            diff,
            config_drift=[{"field": "max_new_tokens", "old": 64, "new": 128}],
            candidate_verdicts=[],
            fixture_diffs=[],
        )
        self.assertEqual(diff["summary"]["config_drift_warnings"], 1)
        self.assertFalse(diff["summary"]["identical"])

    def test_minimal_pair_boundary_drift(self):
        """Minimal pair boundary drift."""

        def change(report):
            """Change."""
            report["search"][0]["boundary_token_ids"] = [10, 21]

        old, new = self._pair(change)
        diff = diffreports.diff_reports(old, new)
        self.assertEqual(
            diff["config_drift"],
            [
                {
                    "field": "boundaries",
                    "old": {"alpha": [10, 20], "beta/mut": [10, 30]},
                    "new": {"alpha": [10, 21], "beta/mut": [10, 30]},
                }
            ],
        )
        self.assertFalse(diff["summary"]["identical"])

    def test_minimal_pair_backend_fingerprint_drift(self):
        """Minimal pair backend fingerprint drift."""
        old, new = self._pair(lambda r: r.update(backend_fingerprint="fp-2"))
        diff = diffreports.diff_reports(old, new)
        self.assertEqual(
            diff["config_drift"],
            [{"field": "backend_fingerprint", "old": "fp-1", "new": "fp-2"}],
        )
        self.assertFalse(diff["summary"]["identical"])

    def test_minimal_pair_candidate_rename_family_unchanged(self):
        """Minimal pair candidate rename family unchanged."""

        def rename(report):
            """Rename."""
            report["search"][1]["format"]["name"] = "beta/mut2"

        old, new = self._pair(rename)
        diff = diffreports.diff_reports(old, new)
        # families identical ("beta" both) -> candidates drift; the rename
        # also re-keys the boundaries snapshot, so it drifts too.
        self.assertEqual(
            diff["config_drift"],
            [
                {
                    "field": "candidates",
                    "old": ["alpha", "beta/mut"],
                    "new": ["alpha", "beta/mut2"],
                },
                {
                    "field": "boundaries",
                    "old": {"alpha": [10, 20], "beta/mut": [10, 30]},
                    "new": {"alpha": [10, 20], "beta/mut2": [10, 30]},
                },
            ],
        )
        self.assertEqual(
            diff["candidate_verdicts"],
            [
                {"candidate": "beta/mut", "stage": "search", "old": "fail", "new": "absent"},
                {"candidate": "beta/mut2", "stage": "search", "old": "absent", "new": "fail"},
            ],
        )
        self.assertEqual(
            diff["fixture_diffs"],
            [
                {
                    "stage": "search",
                    "candidate": "beta/mut",
                    "fixture": "f1",
                    "old": "fail",
                    "new": "absent",
                },
                {
                    "stage": "search",
                    "candidate": "beta/mut",
                    "fixture": "f2",
                    "old": "pass",
                    "new": "absent",
                },
                {
                    "stage": "search",
                    "candidate": "beta/mut2",
                    "fixture": "f1",
                    "old": "absent",
                    "new": "fail",
                },
                {
                    "stage": "search",
                    "candidate": "beta/mut2",
                    "fixture": "f2",
                    "old": "absent",
                    "new": "pass",
                },
            ],
        )
        # fixture ID sets per stage are unchanged -> no fixtures:* drift entry
        self.assertEqual(diff["summary"]["fixtures_flipped"], 0)
        self.assertEqual(diff["summary"]["candidates_changed_status"], 2)
        self.assertFalse(diff["summary"]["identical"])

    def test_minimal_pair_planned_count_drift_verdict_unchanged(self):
        """Minimal pair planned count drift verdict unchanged."""

        def change(report):
            """Change."""
            summary = report["search"][0]["summary"]
            summary["planned"] = 4
            summary["exact_and_boundary_passes"] = 4

        old, new = self._pair(change)
        diff = diffreports.diff_reports(old, new)
        self.assertEqual(diff["config_drift"], [{"field": "planned:alpha", "old": 2, "new": 4}])
        self.assertEqual(diff["candidate_verdicts"], [])  # still pass on both sides
        self.assertFalse(diff["summary"]["identical"])

    def test_minimal_pair_fixture_set_drift(self):
        """Minimal pair fixture set drift."""

        def add(report):
            """Add."""
            report["search"][0]["observations"].append(_obs("f3"))

        old, new = self._pair(add)
        diff = diffreports.diff_reports(old, new)
        self.assertEqual(
            diff["config_drift"],
            [{"field": "fixtures:search", "old": ["f1", "f2"], "new": ["f1", "f2", "f3"]}],
        )
        self.assertEqual(
            diff["fixture_diffs"],
            [
                {
                    "stage": "search",
                    "candidate": "alpha",
                    "fixture": "f3",
                    "old": "absent",
                    "new": "pass",
                }
            ],
        )
        self.assertEqual(diff["summary"]["fixtures_flipped"], 0)
        self.assertFalse(diff["summary"]["identical"])

    def test_minimal_pair_ineligible_summary_flips_verdict(self):
        """Minimal pair ineligible summary flips verdict."""

        def change(report):
            """Change."""
            report["search"][0]["summary"]["aggregate_eligible"] = False

        old, new = self._pair(change)
        diff = diffreports.diff_reports(old, new)
        self.assertEqual(
            diff["candidate_verdicts"],
            [{"candidate": "alpha", "stage": "search", "old": "pass", "new": "fail"}],
        )
        self.assertEqual(diff["fixture_diffs"], [])
        self.assertFalse(diff["summary"]["identical"])

    def test_minimal_pair_missing_summary_is_unknown_plus_planned_drift(self):
        """Minimal pair missing summary is unknown plus planned drift."""

        def change(report):
            """Change."""
            del report["search"][0]["summary"]

        old, new = self._pair(change)
        diff = diffreports.diff_reports(old, new)
        self.assertEqual(
            diff["candidate_verdicts"],
            [{"candidate": "alpha", "stage": "search", "old": "pass", "new": "unknown"}],
        )
        self.assertEqual(diff["config_drift"], [{"field": "planned:alpha", "old": 2, "new": None}])
        self.assertFalse(diff["summary"]["identical"])

    # -- evidence gating ------------------------------------------------------

    def test_evidence_asymmetry_suppresses_fixture_comparison(self):
        """Evidence asymmetry suppresses fixture comparison."""

        def strip(report):
            """Strip."""
            for entry in report["search"]:
                entry.pop("observations", None)
            report["held_out"].pop("observations", None)

        old, new = self._pair()
        strip(new)
        diff = diffreports.diff_reports(old, new)
        self.assertEqual(
            diff["evidence"],
            {"old": True, "new": False, "skipped_observations": {"old": 0, "new": 0}},
        )
        self.assertEqual(diff["fixture_diffs"], [])
        self.assertEqual(diff["config_drift"], [])  # fixtures:* drift also gated
        self.assertEqual(diff["candidate_verdicts"], [])  # verdicts need no evidence
        self.assertTrue(diff["summary"]["evidence_asymmetric"])
        self.assertFalse(diff["summary"]["identical"])

    def test_evidence_asymmetry_either_direction(self):
        """Evidence asymmetry either direction."""

        def strip(report):
            """Strip."""
            for entry in report["search"]:
                entry.pop("observations", None)
            report["held_out"].pop("observations", None)

        old, new = self._pair()
        strip(old)
        diff = diffreports.diff_reports(old, new)
        self.assertEqual(diff["evidence"]["old"], False)
        self.assertEqual(diff["evidence"]["new"], True)
        self.assertEqual(diff["fixture_diffs"], [])
        self.assertTrue(diff["summary"]["evidence_asymmetric"])
        self.assertFalse(diff["summary"]["identical"])

    def test_held_out_observations_without_winner_are_ignored(self):
        """Held out observations without winner are ignored."""
        old, new = self._pair()
        for report in (old, new):
            for entry in report["search"]:
                entry.pop("observations", None)
            report["selected_candidate"] = None
        diff = diffreports.diff_reports(old, new)
        self.assertFalse(diff["evidence"]["old"])
        self.assertFalse(diff["evidence"]["new"])
        self.assertTrue(diff["summary"]["identical"])

    def test_cross_stage_evidence_still_emits_absent_diffs_and_set_drift(self):
        # old: search evidence only; new: held-out evidence only.
        """Cross stage evidence still emits absent diffs and set drift."""
        old, new = self._pair()
        old["held_out"].pop("observations", None)
        for entry in new["search"]:
            entry.pop("observations", None)
        diff = diffreports.diff_reports(old, new)
        self.assertTrue(diff["evidence"]["old"])
        self.assertTrue(diff["evidence"]["new"])
        self.assertFalse(diff["summary"]["evidence_asymmetric"])
        self.assertEqual(
            diff["config_drift"],
            [
                {"field": "fixtures:search", "old": ["f1", "f2"], "new": []},
                {"field": "fixtures:held_out", "old": [], "new": ["h1"]},
            ],
        )
        self.assertEqual(
            diff["fixture_diffs"],
            [
                {
                    "stage": "held_out",
                    "candidate": "alpha",
                    "fixture": "h1",
                    "old": "absent",
                    "new": "pass",
                },
                {
                    "stage": "search",
                    "candidate": "alpha",
                    "fixture": "f1",
                    "old": "pass",
                    "new": "absent",
                },
                {
                    "stage": "search",
                    "candidate": "alpha",
                    "fixture": "f2",
                    "old": "fail",
                    "new": "absent",
                },
                {
                    "stage": "search",
                    "candidate": "beta/mut",
                    "fixture": "f1",
                    "old": "fail",
                    "new": "absent",
                },
                {
                    "stage": "search",
                    "candidate": "beta/mut",
                    "fixture": "f2",
                    "old": "pass",
                    "new": "absent",
                },
            ],
        )
        self.assertFalse(diff["summary"]["identical"])

    def test_evidence_presence_matrix(self):
        """Any-stage evidence per side gates the whole fixture comparison."""

        def build(presence):
            """Build."""
            report = _base_report()
            if presence in ("none", "held"):
                for entry in report["search"]:
                    entry.pop("observations", None)
            if presence in ("none", "search"):
                report["held_out"].pop("observations", None)
            return report

        for old_p in ("none", "search", "held", "both"):
            for new_p in ("none", "search", "held", "both"):
                with self.subTest(old=old_p, new=new_p):
                    old, new = build(old_p), build(new_p)
                    diff = diffreports.diff_reports(old, new)
                    oracle = _oracle_diff(old, new)
                    self.assertEqual(diff, oracle)
                    gate = (old_p != "none") and (new_p != "none")
                    if gate:
                        self.assertTrue(diff["evidence"]["old"])
                        self.assertTrue(diff["evidence"]["new"])
                    else:
                        self.assertEqual(diff["fixture_diffs"], [])
                    asym = (old_p != "none") != (new_p != "none")
                    self.assertEqual(diff["summary"]["evidence_asymmetric"], asym)
        # anchors with exact expected outcomes
        self.assertTrue(
            diffreports.diff_reports(build("none"), build("none"))["summary"]["identical"]
        )
        self.assertTrue(
            diffreports.diff_reports(build("search"), build("search"))["summary"]["identical"]
        )
        cross = diffreports.diff_reports(build("search"), build("held"))
        self.assertFalse(cross["summary"]["identical"])
        self.assertFalse(cross["summary"]["evidence_asymmetric"])
        self.assertTrue(len(cross["fixture_diffs"]) > 0)

    # -- verdict edge cases ---------------------------------------------------

    def test_verdict_bool_planned_is_unknown(self):
        """Verdict bool planned is unknown."""
        self.assertEqual(
            diffreports._verdict(
                {"planned": True, "exact_and_boundary_passes": True, "aggregate_eligible": True}
            ),
            "unknown",
        )

    def test_verdict_edges(self):
        """Verdict edges."""
        v = diffreports._verdict
        self.assertEqual(v(None), "unknown")
        self.assertEqual(v("x"), "unknown")
        self.assertEqual(v({}), "unknown")
        self.assertEqual(
            v({"planned": 0, "exact_and_boundary_passes": 0, "aggregate_eligible": True}), "fail"
        )
        self.assertEqual(
            v({"planned": -3, "exact_and_boundary_passes": 0, "aggregate_eligible": True}), "fail"
        )
        self.assertEqual(
            v({"planned": 2, "exact_and_boundary_passes": 3, "aggregate_eligible": True}), "fail"
        )
        self.assertEqual(
            v({"planned": 2, "exact_and_boundary_passes": 2, "aggregate_eligible": "yes"}),
            "unknown",
        )
        self.assertEqual(
            v({"planned": 2, "exact_and_boundary_passes": 2, "aggregate_eligible": True}), "pass"
        )

    # -- red team ---------------------------------------------------------------

    def test_duplicate_fixture_ids_last_wins(self):
        """Duplicate fixture ids last wins."""

        def dup(report):
            """Dup."""
            report["search"][0]["observations"] = [_obs("f1"), _obs("f1", False, True), _obs("f2")]

        old, new = self._pair(dup)  # dup applies to NEW: f1 last-wins fail, f2 pass
        diff = diffreports.diff_reports(old, new)
        self.assertEqual(
            diff["fixture_diffs"],
            [
                {
                    "stage": "search",
                    "candidate": "alpha",
                    "fixture": "f1",
                    "old": "pass",
                    "new": "fail",
                },
                {
                    "stage": "search",
                    "candidate": "alpha",
                    "fixture": "f2",
                    "old": "fail",
                    "new": "pass",
                },
            ],
        )
        self.assertEqual(diff["summary"]["fixtures_flipped"], 2)
        self.assertFalse(diff["summary"]["identical"])

    def test_duplicate_candidate_names_verdict_first_wins_evidence_last_wins(self):
        """Duplicate candidate names verdict first wins evidence last wins."""
        old, new = self._pair()
        old["search"] = [
            _entry("alpha", observations=[_obs("f1")]),  # verdict pass (first wins)
            _entry("alpha", passes=0, observations=[_obs("f1", False, True)]),  # evidence fail
            _entry("beta/mut", passes=1, observations=[_obs("f1", False, True), _obs("f2")]),
        ]
        diff = diffreports.diff_reports(old, new)
        # verdicts: alpha pass on both sides (first-wins) -> no verdict change
        self.assertEqual(diff["candidate_verdicts"], [])
        # evidence: last-wins -> old alpha f1 fail vs new pass; old lacks f2
        self.assertEqual(
            diff["fixture_diffs"],
            [
                {
                    "stage": "search",
                    "candidate": "alpha",
                    "fixture": "f1",
                    "old": "fail",
                    "new": "pass",
                },
                {
                    "stage": "search",
                    "candidate": "alpha",
                    "fixture": "f2",
                    "old": "absent",
                    "new": "fail",
                },
            ],
        )
        self.assertEqual(diff["summary"]["fixtures_flipped"], 1)
        self.assertFalse(diff["summary"]["identical"])

    def test_duplicate_candidate_later_entry_without_evidence_keeps_earlier(self):
        """Duplicate candidate later entry without evidence keeps earlier."""
        old, new = self._pair()
        old["search"] = [
            _entry("alpha", observations=[_obs("f1")]),
            _entry("alpha", passes=0),  # no observations: earlier outcomes stay
            _entry("beta/mut", passes=1, observations=[_obs("f1", False, True), _obs("f2")]),
        ]
        diff = diffreports.diff_reports(old, new)
        self.assertEqual(diff["candidate_verdicts"], [])
        # earlier outcomes stay: old alpha has {f1: pass}, new has {f1: pass, f2: fail}
        self.assertEqual(
            diff["fixture_diffs"],
            [
                {
                    "stage": "search",
                    "candidate": "alpha",
                    "fixture": "f2",
                    "old": "absent",
                    "new": "fail",
                }
            ],
        )
        self.assertEqual(diff["summary"]["fixtures_flipped"], 0)
        self.assertFalse(diff["summary"]["identical"])

    def test_fixture_missing_both_directions(self):
        """Fixture missing both directions."""

        def change(report):
            """Change."""
            report["search"][0]["observations"] = [_obs("f2"), _obs("f3")]

        old, new = self._pair(change)  # new alpha: [f2 pass, f3 pass]
        diff = diffreports.diff_reports(old, new)
        self.assertEqual(
            diff["fixture_diffs"],
            [
                {
                    "stage": "search",
                    "candidate": "alpha",
                    "fixture": "f1",
                    "old": "pass",
                    "new": "absent",
                },
                {
                    "stage": "search",
                    "candidate": "alpha",
                    "fixture": "f2",
                    "old": "fail",
                    "new": "pass",
                },
                {
                    "stage": "search",
                    "candidate": "alpha",
                    "fixture": "f3",
                    "old": "absent",
                    "new": "pass",
                },
            ],
        )
        self.assertEqual(
            diff["config_drift"],
            # fixture-ID set is unioned across candidates: beta/mut still has f1/f2
            [{"field": "fixtures:search", "old": ["f1", "f2"], "new": ["f1", "f2", "f3"]}],
        )
        self.assertEqual(diff["summary"]["fixtures_flipped"], 1)  # f2 fail->pass
        self.assertFalse(diff["summary"]["identical"])

    def test_fixture_ids_case_sensitive_no_normalization(self):
        """Fixture ids case sensitive no normalization."""

        def change(report):
            """Change."""
            report["search"][0]["observations"][0]["fixture"] = "F1"

        old, new = self._pair(change)
        diff = diffreports.diff_reports(old, new)
        self.assertEqual(
            diff["fixture_diffs"],
            [
                {
                    "stage": "search",
                    "candidate": "alpha",
                    "fixture": "F1",
                    "old": "absent",
                    "new": "pass",
                },
                {
                    "stage": "search",
                    "candidate": "alpha",
                    "fixture": "f1",
                    "old": "pass",
                    "new": "absent",
                },
            ],
        )
        self.assertFalse(diff["summary"]["identical"])

    def test_unicode_fixture_ids_exact_match_and_codepoint_order(self):
        """Unicode fixture ids exact match and codepoint order."""

        def change(report):
            """Change."""
            report["search"][0]["observations"].append(_obs("Ω-✓"))

        old, new = self._pair(change)
        diff = diffreports.diff_reports(old, new)
        self.assertEqual(
            diff["config_drift"],
            [{"field": "fixtures:search", "old": ["f1", "f2"], "new": ["f1", "f2", "Ω-✓"]}],
        )
        self.assertEqual(
            diff["fixture_diffs"],
            [
                {
                    "stage": "search",
                    "candidate": "alpha",
                    "fixture": "Ω-✓",
                    "old": "absent",
                    "new": "pass",
                }
            ],
        )
        self.assertFalse(diff["summary"]["identical"])

    def test_deeply_nested_observation_extras_ignored_malformed_counted(self):
        """Deeply nested observation extras ignored malformed counted."""

        def change(report):
            """Change."""
            report["search"][0]["observations"].extend(
                [
                    {
                        "fixture": "deep",
                        "exact_match": True,
                        "boundary_found": True,
                        "tokens": {"ids": [[1, 2], [3, {"x": [None, {"y": []}]}]]},
                        "meta": [{"a": 1}],
                    },
                    {"fixture": "bad", "exact_match": [True], "boundary_found": True},
                ]
            )

        _old, new = self._pair(change)
        diff = diffreports.diff_reports(new, new)  # sanity: self-diff identical
        self.assertTrue(diff["summary"]["identical"])
        old2, new2 = self._pair()
        change(old2)
        diff2 = diffreports.diff_reports(old2, new2)
        self.assertEqual(
            diff2["fixture_diffs"],
            [
                {
                    "stage": "search",
                    "candidate": "alpha",
                    "fixture": "deep",
                    "old": "pass",
                    "new": "absent",
                }
            ],
        )
        self.assertEqual(diff2["evidence"]["skipped_observations"], {"old": 1, "new": 0})
        self.assertEqual(diff2["summary"]["fixtures_flipped"], 0)
        self.assertFalse(diff2["summary"]["identical"])

    def test_non_list_observations_mean_no_evidence(self):
        """Non list observations mean no evidence."""
        old, new = self._pair()
        for report in (old, new):
            for entry in report["search"]:
                entry["observations"] = {"not": "a list"}
            report["held_out"]["observations"] = "nope"
        diff = diffreports.diff_reports(old, new)
        self.assertFalse(diff["evidence"]["old"])
        self.assertFalse(diff["evidence"]["new"])
        self.assertEqual(diff["evidence"]["skipped_observations"], {"old": 0, "new": 0})
        self.assertTrue(diff["summary"]["identical"])

    # -- BUG-1 pinning ---------------------------------------------------------

    def test_bug1_empty_string_winner_vs_absent_counts_as_changed(self):
        # Former BUG-1 (low, fixed), diffreports.py:361-364 vs :197: the
        # winner section used to keep selected_candidate="" as-is while
        # _held_verdict treats "" as absent, so two winnerless runs compared
        # as "winner CHANGED" with no held-out verdict change. The winner
        # extraction now normalizes "" to None, matching _held_verdict, so
        # "" vs absent is identical on every axis.
        """Bug1 empty string winner vs absent counts as changed."""
        old = {"decision": "a", "selected_candidate": "", "search": [], "held_out": None}
        new = {"decision": "a", "search": [], "held_out": None}
        diff = diffreports.diff_reports(old, new)
        self.assertEqual(diff["winner"], {"old": None, "new": None, "changed": False})
        self.assertEqual(diff["candidate_verdicts"], [])
        self.assertFalse(diff["summary"]["winner_changed"])
        self.assertTrue(diff["summary"]["identical"])
        # "" vs "" is stable too
        same = diffreports.diff_reports(old, old)
        self.assertEqual(same["winner"], {"old": None, "new": None, "changed": False})
        self.assertTrue(same["summary"]["identical"])

    # -- render_summary identical axes ------------------------------------------

    def test_render_summary_identical_axes_with_evidence(self):
        """Render summary identical axes with evidence."""
        old, new = self._pair()
        text = diffreports.render_summary(diffreports.diff_reports(old, new))
        self.assertIn(
            "No differences: the two runs are identical on every compared axis "
            "(decision, winner, candidate verdicts, configuration, per-fixture evidence).",
            text,
        )

    def test_render_summary_identical_axes_without_evidence(self):
        """Render summary identical axes without evidence."""
        old, new = self._pair()
        for report in (old, new):
            for entry in report["search"]:
                entry.pop("observations", None)
            report["held_out"].pop("observations", None)
        text = diffreports.render_summary(diffreports.diff_reports(old, new))
        self.assertIn(
            "No differences: the two runs are identical on every compared axis "
            "(decision, winner, candidate verdicts, configuration).",
            text,
        )

    def test_render_summary_asymmetric_evidence_line(self):
        """Render summary asymmetric evidence line."""
        old, new = self._pair()
        for entry in new["search"]:
            entry.pop("observations", None)
        new["held_out"].pop("observations", None)
        text = diffreports.render_summary(diffreports.diff_reports(old, new))
        self.assertIn(
            "Per-fixture comparison unavailable: new run has no --include-evidence observations",
            text,
        )
        self.assertIn(
            "Summary: decision unchanged, winner unchanged, frontier unchanged, "
            "0 candidate(s) changed verdict, "
            "0 fixture(s) flipped, 0 config drift warning(s).",
            text,
        )

    # -- load-level: non-finite numbers rejected ----------------------------------

    def test_nan_infinity_rejected_at_load(self):
        """Nan infinity rejected at load."""
        for token in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(token=token):
                payload = (
                    '{"decision": "a", "selected_candidate": "c", "search": [], '
                    f'"held_out": null, "x": {token}}}'
                )
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "results.json"
                    path.write_text(payload, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "results_nonfinite_number"):
                        diffreports.load_results(path)


# ---------------------------------------------------------------------------
# Differential fuzz: seeded randomized report pairs, oracle vs implementation.
# ---------------------------------------------------------------------------

_FUZZ_NAMES = ["alpha", "beta", "beta/mut", "gamma", "delta/x", "épsïlon", "Ω", "alpha"]
_FUZZ_FIXTURES = ["f1", "f2", "f3", "g-α", "h✓", " spaced ", "F1", "dup"]


def _fuzz_row(rng: random.Random) -> Any:
    """Fuzz row."""
    roll = rng.random()
    fixture = rng.choice(_FUZZ_FIXTURES)
    if roll < 0.60:
        row: Any = {
            "fixture": fixture,
            "exact_match": rng.choice([True, False]),
            "boundary_found": rng.choice([True, False]),
        }
        if rng.random() < 0.3:
            row["extra"] = {"nested": [[1, {"deep": [None, {"deeper": []}]}]]}
        return row
    if roll < 0.68:
        return {"fixture": fixture, "exact_match": True}  # missing boundary_found
    if roll < 0.74:
        return {"fixture": "", "exact_match": True, "boundary_found": True}  # empty id
    if roll < 0.80:
        return {"fixture": fixture, "exact_match": "yes", "boundary_found": True}  # non-bool
    if roll < 0.86:
        return {"not": "a row"}
    return [fixture]  # non-dict row


def _fuzz_entry(rng: random.Random) -> Any:
    """Fuzz entry."""
    if rng.random() < 0.04:
        return rng.choice(["junk", 42, None, [1]])  # non-dict search entry
    entry: dict[str, Any] = {}
    name = rng.choice(_FUZZ_NAMES)
    pick = rng.random()
    if pick < 0.85:
        entry["format"] = {"name": name}
    elif pick < 0.90:
        entry["format"] = {"name": ""}  # falsy -> unnamed
    elif pick < 0.95:
        entry["format"] = "not-a-dict"
    # else: no format key at all
    pick = rng.random()
    if pick < 0.70:
        entry["summary"] = {
            "planned": rng.choice([0, 1, 2, 3, 5, True, "2", None]),
            "exact_and_boundary_passes": rng.choice([0, 1, 2, 3, 5, True]),
            "aggregate_eligible": rng.choice([True, False, "yes", 1]),
        }
    elif pick < 0.78:
        entry["summary"] = {"planned": 2}  # partial summary
    elif pick < 0.84:
        entry["summary"] = None
    elif pick < 0.90:
        entry["summary"] = [1, 2]  # non-dict summary
    # else: missing summary
    pick = rng.random()
    if pick < 0.60:
        entry["observations"] = [_fuzz_row(rng) for _ in range(rng.randint(0, 4))]
    elif pick < 0.68:
        entry["observations"] = {"not": "a list"}
    elif pick < 0.74:
        entry["observations"] = "nope"
    # else: missing observations
    pick = rng.random()
    if pick < 0.60:
        entry["boundary_token_ids"] = [rng.randint(0, 50000) for _ in range(rng.randint(0, 3))]
    elif pick < 0.68:
        entry["boundary_token_ids"] = [1, True, "x"]  # invalid: not all ints
    elif pick < 0.74:
        entry["boundary_token_ids"] = "nope"
    # else: missing
    return entry


def _fuzz_transfer_section(rng: random.Random) -> dict[str, Any]:
    # Real transfer payloads (cycle 138/140 shape), sometimes malformed,
    # so the differential fuzz exercises the newest diff sections instead
    # of only the legacy axes (wave-2 verifier note C).
    """Fuzz transfer section."""
    ckpts = [f"ckpt{i}" for i in range(rng.randint(1, 3))]
    cands = [f"cand{i}" for i in range(rng.randint(1, 3))]
    verdicts = [
        "transfers",
        "wins_there",
        "ties_there",
        "fails_there",
        "checkpoint_incompatible",
        "unevaluable",
    ]
    payload: dict[str, Any] = {
        "available": True,
        "primary_winner": rng.choice(cands + [None]),
        "transfer_verdict": rng.choice(["full", "partial", "none", "mixed"]),
        "per_checkpoint": {
            c: {
                "decision": rng.choice(["selected", "no_candidate", "tie", 7]),
                "winner": rng.choice(cands + [None]),
            }
            for c in ckpts
        },
        "matrix": {cand: {c: rng.choice(verdicts) for c in ckpts} for cand in cands},
    }
    if rng.random() < 0.15:
        # Hostile: non-string checkpoint key degrades the side.
        payload["per_checkpoint"][rng.choice([1, 2.5])] = {"decision": "x"}
    if rng.random() < 0.10:
        payload["available"] = False
    return payload


def _fuzz_contamination_section(rng: random.Random) -> dict[str, Any]:
    """Fuzz contamination section."""
    tiers = ["clean", "low", "medium", "high", "near_duplicate", "duplicate"]
    pairs = [
        {
            "held_out": f"h{i}",
            "search": f"s{rng.randint(0, 2)}",
            "tier": rng.choice(tiers + [None, 5]),
        }
        for i in range(rng.randint(0, 3))
    ]
    if rng.random() < 0.10:
        pairs.append(rng.choice(["junk", 9, {"held_out": 1}]))
    payload: dict[str, Any] = {
        "worst_tier": rng.choice(tiers + [None, 3]),
        "pairs": pairs,
    }
    return payload


def _fuzz_robustness_section(rng: random.Random) -> dict[str, Any]:
    """Fuzz robustness section."""
    sems = [f"sem{i}" for i in range(rng.randint(1, 3))]
    return {
        "available": True,
        "semantics": sems,
        "per_semantic": {
            s: {
                "decision": rng.choice(["selected", "ambiguous", "no_candidate", "unavailable", 9]),
                "winner": rng.choice(["alpha", "beta", None]),
            }
            for s in sems
        },
        "per_candidate": {},
        "excluded_candidates": [],
        "unavailable_semantics": [],
        "flips": [],
    }


def _fuzz_perturbation_section(rng: random.Random) -> dict[str, Any]:
    """Fuzz perturbation section."""
    ops = ["typo", "case_flip", "whitespace", "unicode_variant", "punctuation_strip"]
    n = rng.randint(1, 3)
    per_variant = {
        f"v{i:02d}": {
            "operator": rng.choice(ops),
            "decision": rng.choice(["selected", "no_candidate", "ambiguous", 4]),
            "winner": rng.choice(["alpha", "beta", None]),
            "changed_fixtures": rng.randint(0, 3),
            "changed_rate": rng.choice([0.0, 0.5, 1.0]),
        }
        for i in range(n)
    }
    if rng.random() < 0.15:
        per_variant[rng.choice([7, 8.5])] = {"decision": "selected"}
    return {
        "available": True,
        "primary_winner": "alpha",
        "primary_decision": "selected",
        "n_variants": n,
        "operators": ops[:n],
        "per_variant": per_variant,
        "stability": {
            "n_variants": n,
            "n_evaluable": n,
            "n_unavailable": 0,
            "winner_retained": rng.randint(0, n),
            "decision_retained": rng.randint(0, n),
        },
        "flips": [],
    }


def _fuzz_healing_section(rng: random.Random) -> dict[str, Any]:
    """Fuzz healing section."""
    per_candidate = {
        name: {
            "native": rng.choice([True, False, None]),
            "healed": rng.choice([True, False, None]),
            "verdict_changed": rng.choice([True, False, None]),
        }
        for name in ("alpha", "beta")
    }
    if rng.random() < 0.15:
        per_candidate[rng.choice([7, 8.5])] = {"native": True}
    return {
        "available": True,
        "primary_winner": "alpha",
        "primary_decision": "selected",
        "healed_decision": rng.choice(["selected", "ambiguous", "no_candidate", 4]),
        "healed_winner": rng.choice(["alpha", "beta", None]),
        "flipped": rng.choice([True, False]),
        "per_candidate": per_candidate,
        "fixture_flips": [],
        "falsification": {
            "per_candidate": {},
            "n_comparable": rng.randint(0, 2),
            "n_holds": rng.randint(0, 2),
            "n_violations": 0,
            "prediction_holds": rng.choice([True, False, None]),
        },
        "n_healed": rng.randint(0, 4),
        "n_fallback": rng.randint(0, 2),
    }


def _fuzz_hierarchy_section(rng: random.Random) -> dict[str, Any]:
    """Fuzz hierarchy section."""
    per_condition = {}
    for position in ("privilege_first", "privilege_last"):
        per_candidate = {
            name: {
                "system_rate": rng.choice([0.0, 0.5, 1.0, None]),
                "user_rate": rng.choice([0.0, 0.5, 1.0, None]),
                "neither_rate": rng.choice([0.0, 0.5, 1.0, None]),
                "pass": rng.choice([True, False, None]),
            }
            for name in ("alpha", "beta")
        }
        if rng.random() < 0.15:
            per_candidate[rng.choice([7, 8.5])] = {"system_rate": 1.0}
        per_condition[position] = {
            "decision": rng.choice(["selected", "ambiguous", "no_candidate", 4]),
            "winner": rng.choice(["alpha", "beta", None]),
            "n_evaluable": rng.randint(0, 2),
            "per_candidate": per_candidate,
        }
    if rng.random() < 0.15:
        per_condition["privilege_first"]["per_candidate"] = "junk"
    groups = {
        label: {
            "n_candidates": rng.randint(0, 3),
            "n_families": rng.randint(0, 2),
            "mean_system_rate": rng.choice([0.0, 0.5, 1.0, None]),
        }
        for label in ("dedicated", "folded", "unknown")
    }
    return {
        "available": True,
        "primary_winner": "alpha",
        "primary_decision": "selected",
        "n_fixtures": rng.randint(0, 8),
        "positions": ["privilege_first", "privilege_last"],
        "per_condition": per_condition,
        "flips": [],
        "position_effect": None,
        "group_adherence": groups,
    }


def _fuzz_prefill_section(rng: random.Random) -> dict[str, Any]:
    """Fuzz prefill section."""
    per_condition = {}
    for prefill_id in ("answer-colon", "final-answer", "result-colon"):
        per_candidate = {
            name: {
                "honor_rate": rng.choice([0.0, 0.5, 1.0, None]),
                "restart_rate": rng.choice([0.0, 0.5, 1.0, None]),
                "think_channel": rng.choice([True, False, None]),
                "pass": rng.choice([True, False, None]),
            }
            for name in ("alpha", "beta")
        }
        if rng.random() < 0.15:
            per_candidate[rng.choice([7, 8.5])] = {"honor_rate": 1.0}
        per_condition[prefill_id] = {
            "decision": rng.choice(["selected", "ambiguous", "no_candidate", 4]),
            "winner": rng.choice(["alpha", "beta", None]),
            "n_evaluable": rng.randint(0, 2),
            "per_candidate": per_candidate,
        }
    if rng.random() < 0.15:
        per_condition["answer-colon"]["per_candidate"] = "junk"
    think_effect = {
        label: {
            "n_candidates": rng.randint(0, 3),
            "mean_honor_rate": rng.choice([0.0, 0.5, 1.0, None]),
        }
        for label in ("think_channel", "plain")
    }
    think_effect["delta"] = rng.choice([0.0, -0.5, None])
    return {
        "available": True,
        "primary_winner": "alpha",
        "primary_decision": "selected",
        "n_fixtures": rng.randint(0, 8),
        "prefills": ["answer-colon", "final-answer", "result-colon"],
        "per_condition": per_condition,
        "flips": [],
        "think_effect": think_effect,
    }


def _fuzz_portfolio_section(rng: random.Random) -> dict[str, Any]:
    """Fuzz portfolio section."""
    fixtures = ["f1", "f2", "f3"]
    per_fixture_winners = {}
    for fid in fixtures:
        if rng.random() < 0.85:
            per_fixture_winners[fid] = sorted(
                rng.sample(["alpha", "beta", "gamma"], rng.randint(1, 2))
            )
    members = sorted(rng.sample(["alpha", "beta", "gamma"], rng.randint(1, 3)))
    per_member_covers = {name: sorted(rng.sample(fixtures, rng.randint(0, 3))) for name in members}
    if rng.random() < 0.15:
        per_member_covers[rng.choice([7, 8.5])] = ["f1"]
    solvable = sorted(per_fixture_winners)
    gap = rng.choice([0.0, 0.25, 0.5, 1.0])
    return {
        "available": True,
        "n_fixtures": 3,
        "n_candidates": rng.randint(1, 3),
        "decision": rng.choice(["selected", "ambiguous", "no_candidate"]),
        "winner": rng.choice(["alpha", None]),
        "excluded_candidates": [],
        "oracle": {
            "n_solvable": len(solvable),
            "solvable_fixture_ids": solvable,
            "unsolvable_fixture_ids": [f for f in fixtures if f not in solvable],
            "oracle_pass_rate": len(solvable) / 3,
            "winner_pass_rate": rng.choice([0.0, 0.5, 1.0, None]),
            "gap": gap,
            "per_fixture_winners": per_fixture_winners,
        },
        "portfolio": {
            "size": len(members),
            "members": members,
            "per_member_covers": per_member_covers,
            "covered_fixture_ids": sorted(
                {
                    fid
                    for covers in per_member_covers.values()
                    if type(covers) is list
                    for fid in covers
                }
            ),
            "uncovered_fixture_ids": [f for f in fixtures if f not in solvable],
            "greedy_note": "greedy set-cover approximation; not guaranteed optimal",
        },
    }


def _fuzz_synthhistory_section(rng: random.Random) -> dict[str, Any]:
    """Fuzz synthhistory section."""
    conditions = ["history_2", "history_6"]
    per_condition = {}
    for condition in conditions:
        per_candidate = {}
        for name in ["alpha", "beta", "gamma/mut"]:
            roll = rng.random()
            if roll < 0.10:
                per_candidate[name] = None
            else:
                passed = rng.randint(0, 2)
                per_candidate[name] = {
                    "n": 2,
                    "passed": passed,
                    "pass": passed == 2,
                }
        if rng.random() < 0.10:
            per_candidate[rng.choice([7, 8.5])] = {"n": 2, "passed": 1, "pass": False}
        per_condition[condition] = {
            "decision": rng.choice(["selected", "ambiguous", "no_candidate", "unavailable"]),
            "winner": rng.choice(["alpha", "beta", None]),
            "n_evaluable": rng.randint(0, 3),
            "per_candidate": per_candidate,
        }
    flips = []
    if rng.random() < 0.5:
        flips.append(
            {
                "condition": rng.choice(conditions),
                "decision_from": "selected",
                "decision_to": rng.choice(["ambiguous", "no_candidate"]),
                "winner_from": "alpha",
                "winner_to": None,
            }
        )
    groups = {}
    for label in ("explicit", "loose", "unknown"):
        if rng.random() < 0.85:
            groups[label] = {
                "n_candidates": rng.randint(0, 3),
                "mean_pass_rate": rng.choice([0.0, 0.5, 1.0, None]),
            }
    return {
        "available": True,
        "primary_winner": rng.choice(["alpha", None]),
        "primary_decision": rng.choice(["selected", "ambiguous"]),
        "n_fixtures": 2,
        "conditions": conditions,
        "per_condition": per_condition,
        "flips": flips,
        "group_pass_rates": groups,
    }


def _fuzz_batchinv_section(rng: random.Random) -> dict[str, Any]:
    """Fuzz batchinv section."""
    pids = ["p1", "p2", "p3"]
    per_prompt = {}
    n_diverged = 0
    for pid in pids:
        roll = rng.random()
        if roll < 0.10:
            # Failed prompt: excluded from the divergence rate.
            per_prompt[pid] = {
                "diverged": None,
                "first_divergence": None,
                "solo_len": None,
                "batched_len": None,
                "error": "solo_failed",
            }
            continue
        diverged = rng.random() < 0.35
        if diverged:
            n_diverged += 1
        per_prompt[pid] = {
            "diverged": diverged,
            "first_divergence": rng.randint(0, 7) if diverged else None,
            "solo_len": 8,
            "batched_len": 8,
            "error": None,
        }
    if rng.random() < 0.10:
        # Hostile key: the validator must reject the whole payload.
        per_prompt[rng.choice([7, 8.5])] = dict(per_prompt["p1"])
    n_failed = sum(1 for e in per_prompt.values() if e.get("error"))
    n_evaluable = len(pids) - n_failed
    return {
        "available": True,
        "n_prompts": len(pids),
        "n_evaluable": n_evaluable,
        "n_failed": n_failed,
        "n_diverged": n_diverged,
        "divergence_rate": (n_diverged / n_evaluable) if n_evaluable else None,
        "verdict": rng.choice(["batch_invariant", "batch_variant", "unavailable"]),
        "per_prompt": per_prompt,
    }


def _fuzz_report(rng: random.Random) -> dict[str, Any]:
    """Fuzz report."""
    report: dict[str, Any] = {
        "decision": rng.choice(["select_best", "abstain", "select_best", "x"]),
        "selected_candidate": rng.choice(["alpha", "beta/mut", "", None, "Ω", 42]),
        "search": [_fuzz_entry(rng) for _ in range(rng.randint(0, 4))],
    }
    if rng.random() < 0.10:
        report["search"].append(rng.choice(["junk", 7]))
    if rng.random() < 0.80:
        report["max_new_tokens"] = rng.choice([32, 64, 128, None, True, "64"])
    if rng.random() < 0.80:
        report["backend_fingerprint"] = rng.choice(["fp1", "fp2", None, 7])
    pick = rng.random()
    if pick < 0.70:
        held: dict[str, Any] = {}
        if rng.random() < 0.80:
            held["summary"] = {
                "planned": rng.choice([0, 1, 2, True]),
                "exact_and_boundary_passes": rng.choice([0, 1, 2]),
                "aggregate_eligible": rng.choice([True, False]),
            }
        if rng.random() < 0.60:
            held["observations"] = [_fuzz_row(rng) for _ in range(rng.randint(0, 3))]
        report["held_out"] = held
    elif pick < 0.80:
        report["held_out"] = None
    # else: missing held_out
    if rng.random() < 0.30:
        report["unknown_key"] = {"nested": [1, 2, {"three": None}]}
    # Newest diff sections (transfer, contamination, selection
    # robustness, perturbation, token healing, hierarchy, prefill,
    # portfolio, synthhistory, batch_invariance, position, stop_cause,
    # ws_canonical, structured, deliforge, fewshot, faketurn): emitted
    # often enough that the differential fuzz exercises them, not just
    # the legacy axes.
    if rng.random() < 0.45:
        report["transfer"] = _fuzz_transfer_section(rng)
    if rng.random() < 0.45:
        report["contamination"] = _fuzz_contamination_section(rng)
    if rng.random() < 0.45:
        report["selection_robustness"] = _fuzz_robustness_section(rng)
    if rng.random() < 0.45:
        report["perturbation"] = _fuzz_perturbation_section(rng)
    if rng.random() < 0.45:
        report["token_healing"] = _fuzz_healing_section(rng)
    if rng.random() < 0.45:
        report["hierarchy"] = _fuzz_hierarchy_section(rng)
    if rng.random() < 0.45:
        report["prefill"] = _fuzz_prefill_section(rng)
    if rng.random() < 0.45:
        report["portfolio"] = _fuzz_portfolio_section(rng)
    if rng.random() < 0.45:
        report["synthhistory"] = _fuzz_synthhistory_section(rng)
    if rng.random() < 0.45:
        report["batch_invariance"] = _fuzz_batchinv_section(rng)
    if rng.random() < 0.45:
        report["position"] = _fuzz_position_section(rng)
    if rng.random() < 0.45:
        report["stop_cause"] = _fuzz_stop_cause_section(rng)
    if rng.random() < 0.45:
        report["ws_canonical"] = _fuzz_ws_canonical_section(rng)
    if rng.random() < 0.45:
        report["structured"] = _fuzz_structured_section(rng)
    if rng.random() < 0.45:
        report["deliforge"] = _fuzz_deliforge_section(rng)
    if rng.random() < 0.45:
        report["fewshot"] = _fuzz_fewshot_section(rng)
    if rng.random() < 0.45:
        report["faketurn"] = _fuzz_faketurn_section(rng)
    if rng.random() < 0.45:
        report["toolcall"] = _fuzz_toolcall_section(rng)
    if rng.random() < 0.45:
        report["verbosity"] = _fuzz_verbosity_section(rng)
    if rng.random() < 0.45:
        report["refusals"] = _fuzz_refusals_section(rng)
    if rng.random() < 0.45:
        report["scorerlens"] = _fuzz_scorerlens_section(rng)
    if rng.random() < 0.45:
        report["aliasing"] = _fuzz_aliasing_section(rng)
    if rng.random() < 0.45:
        report["repetition"] = _fuzz_repetition_section(rng)
    if rng.random() < 0.45:
        report["distraction"] = _fuzz_distraction_section(rng)
        report["leading"] = _fuzz_leading_section(rng)
    if rng.random() < 0.45:
        report["numform"] = _fuzz_numform_section(rng)
        report["dateline"] = _fuzz_dateline_section(rng)
        report["multiturn"] = _fuzz_multiturn_section(rng)
        report["needle"] = _fuzz_needle_section(rng)
        report["placement"] = _fuzz_placement_section(rng)
        report["choice_audit"] = _fuzz_choiceaudit_section(rng)
        report["precision"] = _fuzz_precision_section(rng)
        report["confidence"] = _fuzz_confidence_section(rng)
        report["rotation"] = _fuzz_rotation_section(rng)
    if rng.random() < 0.45:
        report["cache_profile"] = _fuzz_cache_profile_section(rng)
    if rng.random() < 0.45:
        report["confidentiality"] = _fuzz_confidentiality_section(rng)
    if rng.random() < 0.45:
        report["answer_collision"] = _fuzz_answer_collision_section(rng)
    return report


def _fuzz_position_section(rng: random.Random) -> dict[str, Any]:
    """Fuzz position section."""
    per_condition = {}
    for condition in ("correct_first", "correct_middle", "correct_last"):
        per_candidate = {
            name: {
                "correct_rate": rng.choice([0.0, 0.5, 1.0, None]),
                "pass": rng.choice([True, False, None]),
            }
            for name in ("alpha", "beta")
        }
        if rng.random() < 0.15:
            per_candidate[rng.choice([7, 8.5])] = {"correct_rate": 1.0}
        per_condition[condition] = {
            "decision": rng.choice(["selected", "ambiguous", "no_candidate", 4]),
            "winner": rng.choice(["alpha", "beta", None]),
            "n_evaluable": rng.randint(0, 2),
            "per_candidate": per_candidate,
        }
    if rng.random() < 0.15:
        per_condition["correct_first"]["per_candidate"] = "junk"
    return {
        "available": True,
        "primary_winner": "alpha",
        "primary_decision": "selected",
        "n_fixtures": rng.randint(0, 8),
        "conditions": ["correct_first", "correct_middle", "correct_last"],
        "per_condition": per_condition,
        "flips": [],
        "primacy_index": {
            "alpha": rng.choice([1.0, -0.5, 0.0, None]),
            "beta": rng.choice([0.25, None]),
        },
        "condition_effect": {
            "mean_index": rng.choice([0.5, -0.25, None]),
            "n_candidates": rng.randint(0, 2),
            "n_positive": rng.randint(0, 2),
            "n_negative": rng.randint(0, 2),
        },
    }


def _fuzz_stop_cause_section(rng: random.Random) -> dict[str, Any]:
    """Fuzz stop cause section."""
    per_candidate = {}
    for name in ("alpha", "beta"):
        rates = {
            cause: rng.choice([0.0, 0.25, 0.5, 1.0, None])
            for cause in ("eos_id", "stop_string", "max_length", "stop_unknown")
        }
        if rng.random() < 0.15:
            rates[rng.choice(["mystery", 7])] = 0.5
        per_candidate[name] = {
            "n_observations": rng.randint(0, 4),
            "n_evaluable": rng.randint(0, 4),
            "cause_rates": rates,
            "over_generation_rate": rng.choice([0.0, 0.5, 1.0, None]),
            "would_stop_early_rate": rng.choice([0.0, 1.0, None]),
            "max_length_rate": rng.choice([0.0, 0.5, None]),
        }
    if rng.random() < 0.15:
        per_candidate[rng.choice([7, 8.5])] = {"cause_rates": {"eos_id": 1.0}}
    return {
        "available": True,
        "max_new_tokens": rng.choice([16, 64]),
        "per_candidate": per_candidate,
        "overall": {"cause_rates": {"eos_id": 0.5}},
    }


def _fuzz_structured_section(rng: random.Random) -> dict[str, Any]:
    """Fuzz structured section."""
    per_candidate = {}
    for name in ("alpha", "beta"):
        if rng.random() < 0.1:
            per_candidate[name] = None
            continue
        per_candidate[name] = {
            "n": 9,
            "n_evaluable": rng.randint(0, 9),
            "parse_ok": rng.randint(0, 9),
            "schema_ok": rng.randint(0, 9),
            "adherent": rng.randint(0, 9),
            "parse_rate": rng.choice([0.0, 0.5, 1.0, None]),
            "schema_rate": rng.choice([0.0, 0.5, 1.0, None]),
            "adherent_rate": rng.choice([0.0, 0.5, 1.0, None]),
        }
    if rng.random() < 0.15:
        per_candidate[rng.choice([7, 8.5])] = {"parse_rate": 1.0}
    return {
        "available": True,
        "n_fixtures": 9,
        "per_candidate": per_candidate,
        "weakest_candidates": rng.choice([[], ["alpha"], ["beta"]]),
        "excluded_candidates": [],
        "adherence_gap": rng.choice([0.0, 0.5, 1.0, None]),
        "uniform": rng.choice([True, False, None]),
    }


def _fuzz_deliforge_section(rng: random.Random) -> dict[str, Any]:
    """Fuzz deliforge section."""
    per_candidate = {}
    for name in ("alpha", "beta"):
        if rng.random() < 0.1:
            per_candidate[name] = None
            continue
        cells = {}
        for payload in ("fake_assistant", "fake_system"):
            for marker_index in (0, 1):
                key = f"{payload}/{marker_index}"
                roll = rng.random()
                if roll < 0.15:
                    cells[key] = None
                elif roll < 0.30:
                    cells[key] = {
                        "supported": False,
                        "detail": rng.choice(
                            ["no_encoding_pair", "localization_failed", "decode_mismatch", 7]
                        ),
                    }
                else:
                    n_evaluable = rng.randint(0, 4)
                    cells[key] = {
                        "supported": True,
                        "n_evaluable": n_evaluable,
                        "n_flips_reserved": rng.randint(0, n_evaluable),
                        "n_flips_subword": rng.randint(0, n_evaluable),
                        "flip_rate_reserved": rng.choice([0.0, 0.5, 1.0, None]),
                        "flip_rate_subword": rng.choice([0.0, 0.5, 1.0, None]),
                        "authority_gap": rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0, None]),
                    }
        if rng.random() < 0.15:
            cells[rng.choice(["mystery", 7])] = {"supported": True}
        per_candidate[name] = {
            "cells": cells,
            "n_cells": len(cells),
            "n_evaluable_cells": rng.randint(0, 2),
            "mean_authority_gap": rng.choice([0.0, 0.5, None]),
            "max_authority_gap": rng.choice([0.0, 1.0, None]),
            "min_authority_gap": rng.choice([0.0, -1.0, None]),
        }
    if rng.random() < 0.15:
        per_candidate[rng.choice([7, 8.5])] = {"cells": {}}
    return {
        "available": True,
        "payloads": rng.choice([["fake_assistant"], ["fake_assistant", "fake_system"]]),
        "marker_indexes": rng.choice([[0], [0, 1]]),
        "per_candidate": per_candidate,
        "excluded_candidates": rng.choice([[], ["alpha"], ["beta"]]),
        "max_authority_gap": rng.choice([0.0, 0.5, 1.0, None]),
    }


def _fuzz_fewshot_section(rng: random.Random) -> dict[str, Any]:
    """Fuzz fewshot section."""
    per_candidate = {}
    for name in ("alpha", "beta"):
        roll = rng.random()
        if roll < 0.15:
            per_candidate[name] = None
            continue
        if roll < 0.25:
            # Hostile entry shape: the validator must reject it.
            per_candidate[name] = rng.choice([42, "x", {"gain": 1.5}])
            continue
        per_order = {}
        for arm in ("order_a", "order_b"):
            if rng.random() < 0.15:
                per_order[arm] = {"change_rate": rng.choice([1.5, "high"])}
            else:
                n_evaluable = rng.randint(0, 3)
                per_order[arm] = {
                    "change_rate": rng.choice([0.0, 0.5, 1.0, None]),
                    "n_changed": rng.randint(0, n_evaluable),
                    "n_evaluable": n_evaluable,
                }
        per_candidate[name] = {
            "gain": rng.choice([0.0, 0.5, 1.0, None]),
            "order_sensitivity": rng.choice([0.0, 0.25, 0.5, None]),
            "n_evaluable": rng.randint(0, 3),
            "per_order": per_order,
        }
    if rng.random() < 0.15:
        per_candidate[rng.choice([7, 8.5])] = {"gain": 0.0}
    per_arm = {}
    for arm in ("baseline", "order_a", "order_b"):
        per_arm[arm] = {
            "decision": rng.choice(["selected", "ambiguous", "no_candidate", 7]),
            "winner": rng.choice(["alpha", None, 42]),
        }
    return {
        "available": True,
        "k": rng.choice([2, "2"]),
        "demo_ids": rng.choice([["d0", "d1"], [], [1]]),
        "n_eval": rng.choice([2, 0]),
        "primary_winner": rng.choice(["alpha", None]),
        "primary_decision": "selected",
        "per_candidate": per_candidate,
        "excluded_candidates": rng.choice([[], ["alpha"], [1]]),
        "per_arm": per_arm,
        "flips": rng.choice([[], [{"arm": "order_a"}]]),
    }


def _fuzz_faketurn_section(rng: random.Random) -> dict[str, Any]:
    """Fuzz faketurn section."""
    per_candidate = {}
    for name in ("alpha", "beta"):
        if rng.random() < 0.1:
            per_candidate[name] = None
            continue
        cells = {}
        for payload in ("fake_assistant", "fake_system"):
            for marker_index in (0, 1):
                key = f"{payload}/{marker_index}"
                roll = rng.random()
                if roll < 0.15:
                    cells[key] = None
                elif roll < 0.30:
                    cells[key] = {
                        "supported": False,
                        "detail": rng.choice(
                            ["no_provenance", "ref_missing", "marker_count_mismatch", 7]
                        ),
                    }
                else:
                    n_evaluable = rng.randint(0, 4)
                    cells[key] = {
                        "supported": True,
                        "n_evaluable": n_evaluable,
                        "n_flips": rng.randint(0, n_evaluable),
                        "flip_rate": rng.choice([0.0, 0.5, 1.0, None]),
                        "token_source": rng.choice(["same", "cross"]),
                    }
                    if rng.random() < 0.1:
                        cells[key]["token_source"] = "native"  # hostile
        if rng.random() < 0.15:
            cells[rng.choice(["mystery", 7])] = {"supported": True}
        per_candidate[name] = {
            "cells": cells,
            "n_cells": len(cells),
            "n_evaluable_cells": rng.randint(0, 2),
            "same_flip_rate": rng.choice([0.0, 0.5, 1.0, None]),
            "cross_flip_rate": rng.choice([0.0, 0.5, 1.0, None]),
            "transfer_gap": rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0, None]),
            "mean_flip_rate": rng.choice([0.0, 0.5, None]),
            "max_flip_rate": rng.choice([0.0, 1.0, None]),
        }
    if rng.random() < 0.15:
        per_candidate[rng.choice([7, 8.5])] = {"cells": {}}
    return {
        "available": True,
        "payloads": rng.choice([["fake_assistant"], ["fake_assistant", "fake_system"]]),
        "marker_indexes": rng.choice([[0], [0, 1]]),
        "per_candidate": per_candidate,
        "excluded_candidates": rng.choice([[], ["alpha"], ["beta"]]),
        "max_transfer_gap": rng.choice([-1.0, 0.0, 1.0, None]),
        "max_flip_rate": rng.choice([0.0, 0.5, 1.0, None]),
    }


def _fuzz_toolcall_section(rng: random.Random) -> dict[str, Any]:
    """Fuzz toolcall section."""
    per_candidate = {}
    for name in ("alpha", "beta"):
        if rng.random() < 0.1:
            per_candidate[name] = None
            continue
        n_evaluable = rng.randint(0, 9)
        entry = {
            "n": 9,
            "n_evaluable": n_evaluable,
            "adherent": rng.randint(0, n_evaluable),
            "parse_failed": 0,
            "call_failed": 0,
            "not_text": 9 - n_evaluable,
            "selection_rate": rng.choice([0.0, 0.5, 1.0, None, "high"]),
            "args_rate": rng.choice([0.0, 0.5, 1.0, None]),
            "relevance_rate": rng.choice([0.0, 1.0, None]),
            "adherent_rate": rng.choice([0.0, 0.5, 1.0, None]),
        }
        if rng.random() < 0.1:
            entry["selection_rate"] = 1.5  # hostile: out of range
        if rng.random() < 0.1:
            entry[rng.choice(["n", 7])] = -1  # hostile: negative / bad key
        per_candidate[name] = entry
    if rng.random() < 0.15:
        per_candidate[rng.choice([7, 8.5])] = {"n": 1}  # hostile: non-string key
    return {
        "available": True,
        "n_fixtures": 9,
        "per_candidate": per_candidate,
        "weakest_candidates": rng.choice([[], ["alpha"], ["beta"], [7]]),
        "excluded_candidates": rng.choice([[], ["alpha"], ["beta"]]),
    }


def _fuzz_verbosity_section(rng: random.Random) -> dict[str, Any]:
    """Fuzz verbosity section."""
    per_candidate = {}
    for name in ("alpha", "beta"):
        if rng.random() < 0.1:
            per_candidate[name] = None
            continue
        n_evaluable = rng.randint(0, 9)
        median = rng.choice([0.0, 2.5, 11.5, None, "huge"])
        p25 = rng.choice([0.0, 1.0, None])
        p75 = rng.choice([3.0, 12.0, None])
        entry = {
            "n": 9,
            "n_evaluable": n_evaluable,
            "median_excess": median,
            "p25_excess": p25,
            "p75_excess": p75,
            "iqr_excess": rng.choice([0.0, 2.0, 9.5, None]),
            "mean_excess": rng.choice([0.0, 5.0, None]),
            "max_excess": rng.choice([0, 5, 20, None]),
            "capped_share": rng.choice([0.0, 0.5, 1.0, None, 1.5]),
        }
        if rng.random() < 0.1:
            entry["median_excess"] = -1.0  # hostile: negative excess
        if rng.random() < 0.1:
            entry["p25_excess"] = 99.0  # hostile: quartile ordering violated
        if rng.random() < 0.1:
            entry[rng.choice(["n", 7])] = -1  # hostile: negative / bad key
        per_candidate[name] = entry
    if rng.random() < 0.15:
        per_candidate[rng.choice([7, 8.5])] = {"n": 1}  # hostile: non-string key
    return {
        "available": True,
        "n_fixtures": 9,
        "n_brevity_fixtures": rng.choice([0, 2, 9, 10]),
        "per_candidate": per_candidate,
        "most_verbose": rng.choice([[], ["alpha"], ["beta"], [7]]),
        "least_verbose": rng.choice([[], ["alpha"]]),
        "capped_candidates": rng.choice([[], ["beta"]]),
        "excluded_candidates": rng.choice([[], ["alpha"], ["beta"]]),
    }


def _fuzz_refusals_section(rng: random.Random) -> dict[str, Any]:
    # Coherent cells are produced by the real analyzer over random
    # recorded observations (guaranteeing the count identities hold);
    # hostile mutations then try to break them, and oracle and
    # implementation must agree on every rejection.
    """Fuzz refusals section."""
    answers = [
        "I'm sorry, I cannot help with that.",
        "I can't be certain, but 7",
        "7",
        "Paris",
        "As an AI language model, I refuse.",
        "The capital is Paris.",
        "Sorry to hear that. The answer is 7.",
    ]
    candidates = []
    for name in ("alpha", "beta"):
        if rng.random() < 0.1:
            candidates.append(
                {"candidate": name, "available": False, "detail": "observations_missing"}
            )
            continue
        observations = []
        for index in range(rng.randint(0, 6)):
            observations.append(
                {
                    "fixture": f"fx-{index}",
                    "status": rng.choice(["completed", "completed", "completed", "failed"]),
                    "answer": rng.choice(answers),
                    "exact_match": rng.choice([True, False]),
                    "boundary_found": rng.choice([True, True, False]),
                }
            )
        entry = {"candidate": name}
        entry.update(refusals.analyze_candidate(observations, eligible=rng.random() < 0.8))
        if rng.random() < 0.08:
            entry["suspected_rate"] = 1.5  # hostile: rate out of range
        if rng.random() < 0.08:
            entry["n_suspected"] = entry["n_suspected"] + 1  # hostile: identity broken
        if rng.random() < 0.08:
            entry["hits"] = []  # hostile: hits/suspected mismatch
        candidates.append(entry)
    if rng.random() < 0.1:
        candidates.append({"candidate": 7, "available": False, "detail": "entry_malformed"})
    digest = refusals.OPENERS_SHA256
    if rng.random() < 0.2:
        digest = "0" * 64  # protocol drift between sides
    return {
        "available": True,
        "method": {
            "detector": refusals.DETECTOR_ID,
            "openers": list(refusals.REFUSAL_OPENERS),
            "openers_sha256": digest,
            "normalization": "fuzz",
            "scope": "fuzz",
        },
        "evaluable_mismatch": rng.choice([True, False]),
        "candidates": candidates,
        "n_candidates": len(candidates),
        "n_candidates_available": sum(1 for c in candidates if c.get("available") is True),
    }


def _fuzz_scorerlens_section(rng: random.Random) -> dict[str, Any]:
    # Coherent cells are produced by the real profile over random
    # recorded observations (guaranteeing every recompute identity
    # holds); hostile mutations then try to break them, and oracle
    # and implementation must agree on every rejection. A coherent
    # foreign ladder (correct digest over modified data) exercises
    # true protocol drift rather than corruption.
    """Fuzz scorerlens section."""
    expected = {"fx-0": "42", "fx-1": "Paris", "fx-2": "1,000", "fx-3": ""}
    answers = [
        "42",
        "42.",
        "paris",
        "```\n42\n```",
        "1000",
        "1,5",
        "Answer: 42",
        "",
        "?",
        "PARIS",
        "1,000",
    ]
    search = []
    for name in ("alpha", "beta"):
        observations = []
        for index in range(4):
            observations.append(
                {
                    "fixture": f"fx-{index}",
                    "status": rng.choice(["completed", "completed", "completed", "failed"]),
                    "answer": rng.choice(answers),
                    "boundary_found": rng.choice([True, True, False]),
                    "exact_match": False,
                }
            )
        search.append(
            {
                "format": {"name": name},
                "summary": {"aggregate_eligible": rng.random() < 0.8},
                "observations": observations,
            }
        )
    result = {"search": search, "selected_candidate": rng.choice(["alpha", "beta", None])}
    payload = experiments.scorerlens_profile(result, dict(expected))
    if rng.random() < 0.15:
        ladder = copy.deepcopy(scorerlens.LADDER_SPEC)
        ladder["decisions"] = dict(ladder["decisions"], percent="fuzz-foreign")
        payload["method"]["ladder"] = ladder
        payload["method"]["ladder_sha256"] = hashlib.sha256(
            json.dumps(ladder, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    if rng.random() < 0.1:
        payload["fixture_set_sha256"] = hashlib.sha256(b"other-fixtures").hexdigest()
    available = [c for c in payload.get("candidates", []) if c.get("available") is True]
    if available and rng.random() < 0.1:
        available[0]["verdict_counts"]["4"] += 1  # hostile: counts off records
    if available and rng.random() < 0.08:
        available[0]["surface_failure_share"] = 0.99  # hostile: share off counts
    if payload.get("winner_by_rung") and rng.random() < 0.08:
        payload["winner_by_rung"]["4"] = "ghost"  # hostile: manufactured winner
    return payload


def _fuzz_aliasing_section(rng: random.Random) -> dict[str, Any]:
    # Coherent payloads are produced by the real profile over random
    # recorded observations (guaranteeing every recompute identity
    # holds); hostile mutations then try to break them, and oracle
    # and implementation must agree on every rejection. A coherent
    # foreign digest version (records unchanged, version string
    # replaced) exercises true protocol drift rather than
    # corruption, because digest handling is equality-only.
    """Fuzz aliasing section."""
    id_pools = [[1, 2], [1, 2], [3], [4, 5], [1, 2]]
    search = []
    for name in ("alpha", "beta"):
        observations = []
        for index in range(3):
            ids = rng.choice(id_pools)
            observations.append(
                {
                    "candidate": name,
                    "fixture": f"fx-{index}",
                    "input_ids": list(ids),
                    "output_ids": [rng.randint(0, 9), 99],
                    "answer": "x",
                    "exact_match": rng.random() < 0.7,
                    "boundary_found": rng.random() < 0.85,
                    "boundary_text_match": rng.choice([True, False, None]),
                    "raw_reason": "limit",
                    "error": None,
                    "status": rng.choice(["completed", "completed", "completed", "failed"]),
                }
            )
        search.append(
            {
                "format": {"name": name},
                "boundary_token_ids": [99] if rng.random() < 0.8 else [7],
                "summary": {
                    "aggregate_eligible": rng.random() < 0.85,
                    "planned": 3,
                },
                "observations": observations,
            }
        )
    result = {
        "search": search,
        "decision": rng.choice(["ambiguous", "ambiguous", "no_candidate", "failed"]),
        "selected_candidate": None,
        "backend_fingerprint": "sha256:fuzz",
        "seed": 3,
        "max_new_tokens": 8,
    }
    payload = experiments.aliasing_profile(result)
    if rng.random() < 0.12:
        payload["method"]["token_ids_digest"] = "sha256-json-intlist-v99"  # coherent foreign
    if rng.random() < 0.1:
        payload["backend_fingerprint"] = "sha256:other-backend"
    available = [c for c in payload.get("candidates", []) if c.get("available") is True]
    if available and rng.random() < 0.1:
        available[0]["n_effective_render"] = 99  # hostile: count off records (wrong level)
    if available and rng.random() < 0.1:
        payload["n_effective_render"] = payload.get("n_effective_render", 0) + 1
    if rng.random() < 0.08:
        payload["alias_suspect_abstention"] = True  # hostile: manufactured flag
        payload["alias_suspect_detail"] = ""
    if available and rng.random() < 0.08:
        available[0]["records"][0]["input_sha256"] = "0" * 64  # hostile: digest off ids
    return payload


_FUZZ_REPETITION_CONTENTS = {
    "markone": "markone what is two plus two",
    "marktwo": "marktwo name a color",
    "markthree": "markthree name an animal",
}


class _FuzzRepetitionBackend:
    """Word-level scripted backend for the repetition fuzz builder.

    Pass tables are keyed by (fixture marker, arm): an x2 prompt
    contains the fixture content doubled around the joiner, a pad
    prompt contains it once. Word ids are assigned in first-seen
    order (never hash(), so the builder is stable across
    PYTHONHASHSEED values).
    """

    fingerprint = "sha256:fuzz"

    def __init__(self, passes: dict, expected_by_marker: dict) -> None:
        """Init  ."""
        self._passes = passes
        self._expected = expected_by_marker
        self._vocab: dict[str, int] = {}

    def encode(self, text: str) -> tuple[int, ...]:
        """Encode."""
        ids = []
        for word in text.split(" "):
            if word not in self._vocab:
                self._vocab[word] = len(self._vocab) + 1
            ids.append(self._vocab[word])
        return tuple(ids)

    def decode(self, ids: Any) -> str:
        """Decode."""
        inv = {v: k for k, v in self._vocab.items()}
        return " ".join(inv.get(i, "?") for i in ids)

    def generate(
        self, input_ids: Any, *, max_new_tokens: int, seed: int, cancel: Any = None
    ) -> Any:
        """Generate."""
        prompt = self.decode(input_ids)
        marker = next((m for m in self._expected if m in prompt), None)
        content = _FUZZ_REPETITION_CONTENTS.get(marker) if marker is not None else None
        if content is not None:
            arm = "x2" if (content + "\n" + content) in prompt else "pad"
        else:
            arm = "pad" if "orchard" in prompt else "x2"
        ok = self._passes.get((marker, arm), False)
        answer = self._expected[marker] if ok and marker is not None else "wrong"
        ids = (*self.encode(answer), 99)
        ids += (99,) * max(0, max_new_tokens - len(ids))
        return Generation(token_ids=ids, reason="limit")


def _fuzz_repetition_section(rng: random.Random) -> dict[str, Any]:
    # Coherent payloads are produced by the real stage + analysis over
    # a scripted backend (guaranteeing every recompute identity
    # holds); hostile mutations then try to break them, and oracle
    # and implementation must agree on every rejection. A coherent
    # foreign backend fingerprint (protocol rebuilt over the same
    # fixture ids) exercises backend drift rather than corruption.
    """Fuzz repetition section."""
    formats = (
        Format(name="fz-alpha", bos="", user_prefix="u:", assistant_prefix="a:", turn_end="</s>"),
        Format(name="fz-beta", bos="", user_prefix="U>", assistant_prefix="A>", turn_end="<e>"),
    )
    cases = (
        Fixture(
            id="fx-a",
            group="g",
            messages=(Message(role="user", content="markone what is two plus two"),),
            expected="four",
        ),
        Fixture(
            id="fx-b",
            group="g",
            messages=(Message(role="user", content="marktwo name a color"),),
            expected="blue",
        ),
        Fixture(
            id="fx-c",
            group="g",
            messages=(Message(role="user", content="markthree name an animal"),),
            expected="cat",
        ),
    )
    expected_by_marker = {"markone": "four", "marktwo": "blue", "markthree": "cat"}
    passes = {
        (marker, arm): rng.random() < 0.6 for marker in expected_by_marker for arm in ("x2", "pad")
    }
    backend = _FuzzRepetitionBackend(passes, expected_by_marker)
    search = []
    for fmt in formats:
        observations = []
        for case in cases:
            ok = rng.random() < 0.7
            observations.append(
                {
                    "candidate": fmt.name,
                    "fixture": case.id,
                    "input_ids": [1, 2, 3],
                    "output_ids": [4, 99],
                    "answer": case.expected if ok else "wrong",
                    "exact_match": ok,
                    "boundary_found": True,
                    "boundary_text_match": rng.choice([True, False, None]),
                    "raw_reason": "limit",
                    "error": None,
                    "status": "completed",
                }
            )
        search.append(
            {
                "format": {"name": fmt.name},
                "boundary_token_ids": [99],
                "summary": {"aggregate_eligible": rng.random() < 0.85},
                "observations": observations,
            }
        )
    stage = experiments.repetition_stage(
        backend, formats, cases, search, boundary=(99,), max_context=4096, seed=0
    )
    payload = experiments.repetition_analysis(
        stage, "no_candidate", None, tuple(fmt.name for fmt in formats)
    )
    available = [
        entry
        for entry in payload.get("candidates", {}).values()
        if entry.get("status") == "available"
    ]
    if available and rng.random() < 0.12:
        record = available[0]["records"][0]
        if record["arms"]["x2"] is not None:
            record["arms"]["x2"]["verdict"] = not record["arms"]["x2"]["verdict"]  # hostile
    if available and rng.random() < 0.10:
        available[0]["n_vacuous"] = available[0].get("n_vacuous", 0) + 1  # hostile
    if payload.get("available") is True and rng.random() < 0.08:
        payload["protocol"]["fingerprint"] = "0" * 64  # hostile: digest off components
    if payload.get("available") is True and rng.random() < 0.08:
        payload["arms"]["x2"]["flip"] = None  # hostile: flip inconsistent with records
    if payload.get("available") is True and rng.random() < 0.10:
        payload["protocol"] = repetition.protocol_block(
            [case.id for case in cases], "sha256:other-backend"
        )  # coherent foreign backend
    return payload


class _FuzzDistractionBackend:
    """Word-level scripted backend for the distraction fuzz builder.

    Answers are keyed by (marker, kind, placement, dose band);
    ``passes`` decides which cells answer correctly. Kind is
    detected by filler vocabulary (template words vs corpus
    words), placement by which side of the marker the filler
    sits on, and the band by prompt length (native prompts are
    ~30 tokens). Generations pad to the cap with the boundary
    id (reason "limit" means exactly the cap).
    """

    fingerprint = "sha256:fuzz-distraction"

    def __init__(self, passes, expected_by_marker):
        """Init  ."""
        self.passes = dict(passes)
        self.expected_by_marker = dict(expected_by_marker)
        self.vocab: dict[str, int] = {}
        self.rev: dict[int, str] = {}

    def encode(self, text):
        """Encode."""
        ids = []
        for token in text.replace("\n", " \n ").split(" "):
            if not token:
                continue
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab) + 1
                self.rev[self.vocab[token]] = token
            ids.append(self.vocab[token])
        return tuple(ids)

    def decode(self, ids):
        """Decode."""
        return " ".join(self.rev.get(int(i), "?") for i in ids)

    def generate(self, input_ids, *, max_new_tokens, seed=None, cancel=None):
        """Generate."""
        text = self.decode(list(input_ids))
        marker = None
        for candidate in self.expected_by_marker:
            if candidate in text:
                marker = candidate
                break
        kind = None
        filler_at = None
        for word in ("ledger", "timetable", "manifest", "kiln"):
            if word in text:
                kind = "near-miss"
                filler_at = text.index(word)
                break
        if kind is None:
            for word in ("harbour", "gardener", "ferries", "cobbles"):
                if word in text:
                    kind = "neutral"
                    filler_at = text.index(word)
                    break
        answer = self.expected_by_marker.get(marker, "dunno")
        if marker is not None and kind is not None:
            placement = "before" if filler_at < text.index(marker) else "after"
            total = len(input_ids)
            band = 64 if total < 150 else 256 if total < 600 else 1024
            if not self.passes.get((marker, kind, placement, band), True):
                answer = "8" if kind == "near-miss" and marker == "markone" else "wrong"
        out = list(self.encode(answer)) + [99]
        out = (out + [99] * max_new_tokens)[:max_new_tokens]
        return Generation(token_ids=tuple(out), reason="limit")


def _fuzz_distraction_section(rng: random.Random) -> dict[str, Any]:
    # Coherent payloads come from the real stage + analysis over a
    # scripted backend; hostile mutations then try to break them,
    # and oracle and implementation must agree on every rejection.
    # A coherent foreign backend fingerprint exercises backend
    # drift rather than corruption.
    """Fuzz distraction section."""
    formats = (
        Format(name="fz-alpha", bos="", user_prefix="u:", assistant_prefix="a:", turn_end="</s>"),
        Format(name="fz-beta", bos="", user_prefix="U>", assistant_prefix="A>", turn_end="<e>"),
    )
    cases = (
        Fixture(
            id="fx-a",
            group="g",
            messages=(Message(role="user", content="markone what is six plus one"),),
            expected="7",
        ),
        Fixture(
            id="fx-b",
            group="g",
            messages=(Message(role="user", content="marktwo name a color"),),
            expected="blue",
        ),
        Fixture(
            id="fx-c",
            group="g",
            messages=(Message(role="user", content="markthree name an animal"),),
            expected="cat",
        ),
    )
    expected_by_marker = {"markone": "7", "marktwo": "blue", "markthree": "cat"}
    passes = {
        (marker, kind, placement, band): rng.random() < 0.6
        for marker in expected_by_marker
        for kind in ("neutral", "near-miss")
        for placement in ("before", "after")
        for band in (64, 256, 1024)
    }
    backend = _FuzzDistractionBackend(passes, expected_by_marker)
    search = []
    for fmt in formats:
        observations = []
        for case in cases:
            ok = rng.random() < 0.7
            observations.append(
                {
                    "candidate": fmt.name,
                    "fixture": case.id,
                    "input_ids": list(backend.encode(fmt.render(case.messages))),
                    "output_ids": [4, 99],
                    "answer": case.expected if ok else "wrong",
                    "exact_match": ok,
                    "boundary_found": True,
                    "boundary_text_match": rng.choice([True, False, None]),
                    "raw_reason": "limit",
                    "error": None,
                    "status": "completed",
                }
            )
        search.append(
            {
                "format": {"name": fmt.name},
                "boundary_token_ids": [99],
                "summary": {"aggregate_eligible": rng.random() < 0.85},
                "observations": observations,
            }
        )
    stage = experiments.distraction_stage(
        backend, formats, cases, search, boundary=(99,), max_context=8192, seed=0
    )
    payload = experiments.distraction_analysis(
        stage, "no_candidate", None, tuple(fmt.name for fmt in formats)
    )
    available = [
        entry
        for entry in payload.get("candidates", {}).values()
        if entry.get("status") == "available"
    ]
    if available and rng.random() < 0.12:
        record = available[0]["records"][0]
        cell = record["cells"].get("neutral_64_before")
        if cell is not None:
            cell["verdict"] = not cell["verdict"]  # hostile
    if available and rng.random() < 0.10:
        available[0]["n_vacuous"] = available[0].get("n_vacuous", 0) + 1  # hostile
    if payload.get("available") is True and rng.random() < 0.08:
        payload["protocol"]["fingerprint"] = "0" * 64  # hostile: digest off components
    if payload.get("available") is True and rng.random() < 0.08:
        payload["conditions"]["neutral_64_before"]["flip"] = None  # hostile
    if payload.get("available") is True and rng.random() < 0.10:
        protocol = dict(
            distraction.protocol_object([case.id for case in cases], "sha256:other-backend")
        )
        protocol["fingerprint"] = distraction.protocol_fingerprint(
            [case.id for case in cases], "sha256:other-backend"
        )
        payload["protocol"] = protocol  # coherent foreign backend
    return payload


class _FuzzLeadingBackend:
    """Word-level scripted backend for the leading fuzz builder.

    Claim cells are detected by the frozen sentence shapes in
    the decoded prompt; ``adopt`` decides whether a wrong claim
    is answered with the claimed value, ``fail_right`` whether a
    right claim is answered wrongly anyway. Anything else
    answers the fixture's expected string.
    """

    fingerprint = "sha256:fuzz-leading"

    def __init__(self, adopt, fail_right, expected_by_marker):
        """Init  ."""
        self.adopt = dict(adopt)
        self.fail_right = dict(fail_right)
        self.expected_by_marker = dict(expected_by_marker)
        self.vocab: dict[str, int] = {}
        self.rev: dict[int, str] = {}

    def encode(self, text):
        """Encode."""
        ids = []
        for token in text.replace("\n", " \n ").split(" "):
            if not token:
                continue
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab) + 1
                self.rev[self.vocab[token]] = token
            ids.append(self.vocab[token])
        return tuple(ids)

    def decode(self, ids):
        """Decode."""
        return " ".join(self.rev.get(int(i), "?") for i in ids)

    def generate(self, input_ids, *, max_new_tokens, seed=None, cancel=None):
        """Generate."""
        text = self.decode(list(input_ids))
        marker = None
        for candidate in self.expected_by_marker:
            if candidate in text:
                marker = candidate
                break
        expected = self.expected_by_marker.get(marker, "dunno")
        answer = expected
        framing = None
        if "I'm sure the answer is" in text:
            framing = "first"
        elif "Another user is sure the answer is" in text:
            framing = "third"
        if marker is not None and framing is not None:
            claim = text.split("answer is ", 1)[1].split(" ", 1)[0].split(".", 1)[0]
            if claim != expected and self.adopt.get((marker, framing), False):
                answer = claim
            elif claim == expected and self.fail_right.get((marker, framing), False):
                answer = "wrong!"
        out = list(self.encode(answer)) + [99]
        out = (out + [99] * max_new_tokens)[:max_new_tokens]
        return Generation(token_ids=tuple(out), reason="limit")


def _fuzz_leading_section(rng: random.Random) -> dict[str, Any]:
    # Coherent payloads come from the real stage + analysis over
    # a scripted backend; hostile mutations then try to break
    # them, and oracle and implementation must agree on every
    # rejection. A coherent foreign backend fingerprint exercises
    # backend drift rather than corruption.
    """Fuzz leading section."""
    formats = (
        Format(name="fz-alpha", bos="", user_prefix="u:", assistant_prefix="a:", turn_end="</s>"),
        Format(name="fz-beta", bos="", user_prefix="U>", assistant_prefix="A>", turn_end="<e>"),
    )
    cases = (
        Fixture(
            id="fx-a",
            group="g1",
            messages=(Message(role="user", content="markone what is six plus one"),),
            expected="7",
        ),
        Fixture(
            id="fx-b",
            group="g2",
            messages=(Message(role="user", content="marktwo name a color"),),
            expected="blue",
        ),
        Fixture(
            id="fx-c",
            group="g1",
            messages=(Message(role="user", content="markthree what is four plus five"),),
            expected="9",
        ),
    )
    expected_by_marker = {"markone": "7", "marktwo": "blue", "markthree": "9"}
    adopt = {
        (marker, framing): rng.random() < 0.5
        for marker in expected_by_marker
        for framing in ("first", "third")
    }
    fail_right = {
        (marker, framing): rng.random() < 0.2
        for marker in expected_by_marker
        for framing in ("first", "third")
    }
    backend = _FuzzLeadingBackend(adopt, fail_right, expected_by_marker)
    search = []
    for fmt in formats:
        observations = []
        for case in cases:
            ok = rng.random() < 0.7
            observations.append(
                {
                    "candidate": fmt.name,
                    "fixture": case.id,
                    "input_ids": list(backend.encode(fmt.render(case.messages))),
                    "output_ids": [4, 99],
                    "answer": case.expected if ok else "wrong",
                    "exact_match": ok,
                    "boundary_found": True,
                    "boundary_text_match": rng.choice([True, False, None]),
                    "raw_reason": "limit",
                    "error": None,
                    "status": "completed",
                }
            )
        search.append(
            {
                "format": {"name": fmt.name},
                "boundary_token_ids": [99],
                "summary": {"aggregate_eligible": rng.random() < 0.85},
                "observations": observations,
            }
        )
    stage = experiments.leading_stage(
        backend, formats, cases, search, boundary=(99,), max_context=8192, seed=0
    )
    payload = experiments.leading_analysis(
        stage, "no_candidate", None, tuple(fmt.name for fmt in formats)
    )
    available = [
        entry
        for entry in payload.get("candidates", {}).values()
        if entry.get("status") == "available"
    ]
    if available and rng.random() < 0.12:
        record = available[0]["records"][0]
        cell = record["cells"].get("wrong_first_person")
        if cell is not None:
            cell["verdict"] = not cell["verdict"]  # hostile
    if available and rng.random() < 0.10:
        block = available[0]["framings"]["first_person"]["caving"]
        block["n_caved"] = block.get("n_caved", 0) + 1  # hostile
    if payload.get("available") is True and rng.random() < 0.08:
        payload["protocol"]["fingerprint"] = "0" * 64  # hostile: digest off components
    if payload.get("available") is True and rng.random() < 0.08:
        payload["conditions"]["wrong_first_person"]["flip"] = None  # hostile
    if payload.get("available") is True and rng.random() < 0.10:
        donor_map = leading.build_donor_map(cases)
        ids = [case.id for case in cases]
        protocol = dict(leading.protocol_object(ids, donor_map, "sha256:other-backend"))
        protocol["fingerprint"] = leading.protocol_fingerprint(
            ids, donor_map, "sha256:other-backend"
        )
        payload["protocol"] = protocol  # coherent foreign backend
    return payload


class _FuzzNumformBackend:
    """Word-token scripted backend for the numform fuzz builder.

    Answers the generated tasks by rule in one of four modes
    (faithful / plain-normalising / off-by-one / label-flipping)
    so coherent payloads exercise the full decomposition range.
    """

    fingerprint = "sha256:fuzz-numform"

    def __init__(self, mode: str):
        """Init  ."""
        self.vocab: dict[str, int] = {}
        self.rev: dict[int, str] = {}
        self.mode = mode

    def encode(self, text):
        """Encode."""
        ids = []
        for token in text.replace("\n", " \n ").split(" "):
            if not token:
                continue
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab) + 1
                self.rev[self.vocab[token]] = token
            ids.append(self.vocab[token])
        return tuple(ids)

    def decode(self, ids):
        """Decode."""
        return " ".join(self.rev.get(int(i), "?") for i in ids)

    def generate(self, input_ids, *, max_new_tokens, seed=None, cancel=None):
        """Generate."""
        text = self.decode(list(input_ids))
        if "Copy this number" in text:
            task = "copy"
        elif "Which of these" in text:
            task = "compare"
        else:
            task = "add"
        tokens = []
        for raw_token in text.split(" "):
            token = raw_token.split("<")[0]
            if any(ch.isdigit() for ch in token) and numform.parse_value(token) is not None:
                tokens.append(token)
        values = [numform.parse_value(token) for token in tokens]

        def form_of(token: str) -> str:
            """Form of."""
            if "," in token:
                return "comma"
            if "\u202f" in token:
                return "space"
            if "e" in token and any(ch.isdigit() for ch in token):
                return "scientific"
            return "plain"

        if task == "copy":
            value = values[-1]
            form = form_of(tokens[-1])
            if self.mode == "plain":
                answer = str(value)
            elif self.mode == "wrong_value":
                answer = numform.render_form(value + 1, form)
            else:
                answer = numform.render_form(value, form)
        elif task == "compare":
            first, second = values[-2], values[-1]
            answer = "first" if first > second else "second"
            if self.mode == "flip_label":
                answer = "second" if answer == "first" else "first"
        else:
            first, second = values[-2], values[-1]
            form = form_of(tokens[-2])
            total = first + second
            if self.mode == "plain":
                answer = str(total)
            elif self.mode == "wrong_value":
                answer = numform.render_form(total + 1, form)
            else:
                answer = numform.render_form(total, form)
        boundary = self.encode("<|im_end|>")[0]
        out = list(self.encode(answer)) + [boundary]
        out = (out + [boundary] * max_new_tokens)[:max_new_tokens]
        return Generation(token_ids=tuple(out), reason="limit")


class _FuzzDatelineBackend:
    """Word-level scripted backend for the dateline fuzz builder.

    Date cells are detected by frozen date substrings in the
    decoded prompt; ``fail_on`` holds (marker, substring) pairs
    whose cells are answered wrongly. Anything else answers the
    fixture's expected string, so untreated/native prompts pass.
    """

    fingerprint = "sha256:fuzz-dateline"

    def __init__(self, fail_on, expected_by_marker):
        """Init  ."""
        self.fail_on = set(fail_on)
        self.expected_by_marker = dict(expected_by_marker)
        self.vocab: dict[str, int] = {}
        self.rev: dict[int, str] = {}

    def encode(self, text):
        """Encode."""
        ids = []
        for token in text.replace("\n", " \n ").split(" "):
            if not token:
                continue
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab) + 1
                self.rev[self.vocab[token]] = token
            ids.append(self.vocab[token])
        return tuple(ids)

    def decode(self, ids):
        """Decode."""
        return " ".join(self.rev.get(int(i), "?") for i in ids)

    def generate(self, input_ids, *, max_new_tokens, seed=None, cancel=None):
        """Generate."""
        text = self.decode(list(input_ids))
        marker = None
        for candidate in self.expected_by_marker:
            if candidate in text:
                marker = candidate
                break
        answer = self.expected_by_marker.get(marker, "dunno")
        if marker is not None:
            for fail_marker, substring in self.fail_on:
                if fail_marker == marker and substring in text:
                    answer = "wrong!"
                    break
        out = list(self.encode(answer)) + [99]
        out = (out + [99] * max_new_tokens)[:max_new_tokens]
        return Generation(token_ids=tuple(out), reason="limit")


def _fuzz_dateline_section(rng: random.Random) -> dict[str, Any]:
    # Coherent payloads come from the real stage + analysis over
    # a scripted backend; hostile mutations then try to break
    # them, and oracle and implementation must agree on every
    # rejection. A coherent foreign backend fingerprint exercises
    # backend drift rather than corruption.
    """Fuzz dateline section."""
    formats = (
        Format(name="fz-alpha", bos="", user_prefix="u:", assistant_prefix="a:", turn_end="</s>"),
        Format(name="fz-beta", bos="", user_prefix="U>", assistant_prefix="A>", turn_end="<e>"),
    )
    cases = (
        Fixture(
            id="fx-a",
            group="g1",
            messages=(Message(role="user", content="markone name a color"),),
            expected="blue",
        ),
        Fixture(
            id="fx-b",
            group="g2",
            messages=(Message(role="user", content="marktwo name a port town"),),
            expected="harbour",
        ),
        Fixture(
            id="fx-c",
            group="g3",
            messages=(
                Message(role="user", content="markthree first part"),
                Message(role="assistant", content="noted"),
                Message(role="user", content="markthree second part"),
            ),
            expected="gamma",
        ),
        Fixture(
            id="fx-d",
            group="g4",
            messages=(Message(role="user", content="markfour pick a day number"),),
            expected="21",
        ),
    )
    expected_by_marker = {
        "markone": "blue",
        "marktwo": "harbour",
        "markthree": "gamma",
        "markfour": "21",
    }
    fail_substrings = ("2028-02-29", "29 February 2028", "Thursday, 2026-10-21", "2027-10-21")
    fail_on = {
        (marker, substring)
        for marker in expected_by_marker
        for substring in fail_substrings
        if rng.random() < 0.3
    }
    backend = _FuzzDatelineBackend(fail_on, expected_by_marker)
    search = []
    for fmt in formats:
        observations = []
        for case in cases:
            ok = rng.random() < 0.8
            observations.append(
                {
                    "candidate": fmt.name,
                    "fixture": case.id,
                    "input_ids": list(backend.encode(fmt.render(case.messages))),
                    "output_ids": [4, 99],
                    "answer": case.expected if ok else "wrong",
                    "exact_match": ok,
                    "boundary_found": True,
                    "boundary_text_match": rng.choice([True, False, None]),
                    "raw_reason": "limit",
                    "error": None,
                    "status": "completed",
                }
            )
        search.append(
            {
                "format": {"name": fmt.name},
                "boundary_token_ids": [99],
                "summary": {"aggregate_eligible": rng.random() < 0.85},
                "observations": observations,
            }
        )
    stage = experiments.dateline_stage(
        backend, formats, cases, search, boundary=(99,), max_context=8192, seed=0
    )
    payload = experiments.dateline_analysis(
        stage, "no_candidate", None, tuple(fmt.name for fmt in formats)
    )
    available = [
        entry
        for entry in payload.get("candidates", {}).values()
        if entry.get("status") == "available"
    ]
    if available and rng.random() < 0.12:
        record = available[0]["records"][0]
        cell = record["cells"].get("anchor_iso")
        if cell is not None:
            cell["verdict"] = not cell["verdict"]  # hostile
    if available and rng.random() < 0.10:
        block = available[0]["date_loss"]["anchor"]
        block["n_loss"] = block.get("n_loss", 0) + 1  # hostile
    if payload.get("available") is True and rng.random() < 0.08:
        payload["protocol"]["fingerprint"] = "0" * 64  # hostile: digest off components
    if payload.get("available") is True and rng.random() < 0.08:
        payload["arms"]["lie"]["flip"] = None  # hostile
    if payload.get("available") is True and rng.random() < 0.10:
        ids = [case.id for case in cases]
        protocol = dict(dateline.protocol_object(ids, "sha256:other-backend"))
        protocol["fingerprint"] = dateline.protocol_fingerprint(ids, "sha256:other-backend")
        payload["protocol"] = protocol  # coherent foreign backend
    return payload


class _FuzzMultiturnBackend:
    """Word-level scripted backend for the multiturn fuzz builder.

    Answers are parsed out of the decoded prompt: ledger totals
    are summed from the stated entries, the shelf question takes
    the argmax label, and intermediate turns get the running
    partial answer (a plausible self-generated history). Fail
    modes answer wrongly for deep ledgers or shelf questions.
    """

    fingerprint = "sha256:fuzz-multiturn"

    def __init__(self, fail_deep: bool, fail_shelf: bool):
        """Init  ."""
        self.fail_deep = fail_deep
        self.fail_shelf = fail_shelf
        self.vocab: dict[str, int] = {}
        self.rev: dict[int, str] = {}

    def encode(self, text):
        """Encode."""
        ids = []
        for token in text.replace("\n", " \n ").split(" "):
            if not token:
                continue
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab) + 1
                self.rev[self.vocab[token]] = token
            ids.append(self.vocab[token])
        return tuple(ids)

    def decode(self, ids):
        """Decode."""
        return " ".join(self.rev.get(int(i), "?") for i in ids)

    def generate(self, input_ids, *, max_new_tokens, seed=None, cancel=None):
        """Generate."""
        import re as _re

        text = self.decode(list(input_ids))
        entries = [int(v) for v in _re.findall(r"Ledger entry: (-?\d+)\.", text)]
        shelves = _re.findall(r"Shelf ([ABC]) holds (\d+) units\.", text)
        if "What is the ledger total?" in text:
            answer = str(sum(entries) + (1 if self.fail_deep and len(entries) >= 3 else 0))
        elif "Which shelf holds the most units?" in text:
            best = max(shelves, key=lambda pair: int(pair[1]))[0]
            answer = "A" if self.fail_shelf and best != "A" else best
        elif entries:
            answer = str(sum(entries))
        elif shelves:
            answer = max(shelves, key=lambda pair: int(pair[1]))[0]
        else:
            answer = "dunno"
        boundary = self.encode("endmark")[0]
        out = list(self.encode(answer)) + [boundary]
        out = (out + [boundary] * max_new_tokens)[:max_new_tokens]
        return Generation(token_ids=tuple(out), reason="limit")


def _fuzz_multiturn_section(rng: random.Random) -> dict[str, Any]:
    # Coherent payloads come from the real stage + analysis over
    # a scripted backend; hostile mutations then try to break
    # them, and oracle and implementation must agree on every
    # rejection. A coherent foreign backend fingerprint exercises
    # backend drift rather than corruption.
    """Fuzz multiturn section."""
    formats = (
        Format(name="fz-alpha", bos="", user_prefix="u:", assistant_prefix="a:", turn_end="</s>"),
        Format(name="fz-beta", bos="", user_prefix="U>", assistant_prefix="A>", turn_end="<e>"),
    )
    backend = _FuzzMultiturnBackend(fail_deep=rng.random() < 0.4, fail_shelf=rng.random() < 0.4)
    seed = rng.choice([0, 1, 2])
    stage = experiments.multiturn_stage(
        backend,
        formats,
        boundary=tuple(backend.encode("endmark")),
        max_context=8192,
        seed=seed,
    )
    payload = experiments.multiturn_analysis(stage, tuple(fmt.name for fmt in formats))
    available = [
        entry
        for entry in payload.get("candidates", {}).values()
        if entry.get("status") == "available"
    ]
    if available and rng.random() < 0.12:
        record = available[0]["records"][0]
        cell = record["cells"].get("full")
        if cell is not None:
            cell["verdict"] = not cell["verdict"]  # hostile
    if available and rng.random() < 0.10:
        available[0]["classification"] = "self_history_drift"  # hostile
    if available and rng.random() < 0.10:
        block = available[0]["depths"]["2"]
        block["n_paired"] = block.get("n_paired", 0) + 1  # hostile
    if payload.get("available") is True and rng.random() < 0.08:
        payload["protocol"]["fingerprint"] = "0" * 64  # hostile: digest off components
    if payload.get("available") is True and rng.random() < 0.10:
        protocol = dict(multiturn.protocol_object(seed, "sha256:other-backend"))
        protocol["fingerprint"] = multiturn.protocol_fingerprint(seed, "sha256:other-backend")
        payload["protocol"] = protocol  # coherent foreign backend
    return payload


class _FuzzNeedleBackend:
    """Byte-level scripted backend for the needle fuzz builder.

    Byte tokens keep the real stage fast (a full two-candidate
    run is about a second) while still exercising the sizing
    search end to end. The answer is the planted code parsed
    out of the decoded prompt; ``midfail`` mode answers wrongly
    when the needle sits in the middle stretch of the prompt,
    which produces genuine U-shaped curves and partial
    headline populations.
    """

    fingerprint = "sha256:fuzz-needle"

    def __init__(self, mode: str):
        """Init  ."""
        self.mode = mode

    @staticmethod
    def encode(text):
        """Encode."""
        return tuple(text.encode("utf-8"))

    @staticmethod
    def decode(ids):
        """Decode."""
        return bytes(int(i) % 256 for i in ids).decode("utf-8", "replace")

    def generate(self, input_ids, *, max_new_tokens, seed=None, cancel=None):
        """Generate."""
        import re as _re

        text = self.decode(list(input_ids))
        found = _re.findall(r"code is ([a-z]+-\d+)\.", text)
        if found:
            answer = found[-1]
            if self.mode == "midfail" and len(text) > 1000:
                at = text.find("code is")
                fraction = at / max(1, len(text))
                if 0.2 < fraction < 0.8:
                    answer = "wrong-0000"
        else:
            answer = "unknown"
        boundary = list(self.encode("endmark"))
        out = list(self.encode(answer)) + boundary
        return Generation(
            token_ids=tuple((out + boundary * max_new_tokens)[:max_new_tokens]),
            reason="limit",
        )


def _fuzz_needle_section(rng: random.Random) -> dict[str, Any]:
    # Coherent payloads come from the real stage + analysis over
    # a scripted backend; hostile mutations then try to break
    # them, and oracle and implementation must agree on every
    # rejection. A coherent foreign backend fingerprint exercises
    # backend drift rather than corruption.
    """Fuzz needle section."""
    formats = (
        Format(name="fz-alpha", bos="", user_prefix="u:", assistant_prefix="a:", turn_end="</s>"),
        Format(name="fz-beta", bos="", user_prefix="U>", assistant_prefix="A>", turn_end="<e>"),
    )
    backend = _FuzzNeedleBackend(rng.choice(["perfect", "midfail"]))
    seed = rng.choice([0, 1, 2])
    stage = experiments.needle_stage(
        backend,
        formats,
        boundary=tuple(backend.encode("endmark")),
        max_context=65536,
        seed=seed,
    )
    payload = experiments.needle_analysis(stage, tuple(fmt.name for fmt in formats))
    available = [
        entry
        for entry in payload.get("candidates", {}).values()
        if entry.get("status") == "available"
    ]
    if available and rng.random() < 0.12:
        record = available[0]["records"][0]
        cell = record["cells"].get("dose_1024_depth_50")
        if cell is not None:
            cell["verdict"] = not cell["verdict"]  # hostile
    if available and rng.random() < 0.10:
        record = available[0]["records"][1]
        cell = record["cells"].get("dose_1024_depth_25")
        if cell is not None:
            cell["input_tokens"] = cell["input_tokens"] + 1  # hostile: delta breaks
    if available and rng.random() < 0.10:
        payload["readouts"]["1024"]["deficit"]["fz-alpha"] = 0.5  # hostile
    if payload.get("available") is True and rng.random() < 0.08:
        payload["protocol"]["fingerprint"] = "0" * 64  # hostile: digest off components
    if payload.get("available") is True and rng.random() < 0.10:
        protocol = dict(needle.protocol_object(seed, "sha256:other-backend"))
        protocol["fingerprint"] = needle.protocol_fingerprint(seed, "sha256:other-backend")
        payload["protocol"] = protocol  # coherent foreign backend
    return payload


class _FuzzPlacementBackend:
    """Byte-level scripted backend for the placement fuzz
    builder. The answer is the planted inspection code parsed
    out of the decoded prompt; ``payloadfirst_fail`` mode
    answers wrongly when the question sits after the payload,
    which produces genuine one-directional discordance. The
    closed-book arm carries no fact, so it answers "unknown"
    and never trips the screen in either mode.
    """

    fingerprint = "sha256:fuzz-placement"

    def __init__(self, mode: str):
        """Init  ."""
        self.mode = mode

    @staticmethod
    def encode(text):
        """Encode."""
        return tuple(text.encode("utf-8"))

    @staticmethod
    def decode(ids):
        """Decode."""
        return bytes(int(i) % 256 for i in ids).decode("utf-8", "replace")

    def generate(self, input_ids, *, max_new_tokens, seed=None, cancel=None):
        """Generate."""
        import re as _re

        text = self.decode(list(input_ids))
        found = _re.findall(r"inspection code is ([a-z]+-\d+)\.", text)
        if found:
            answer = found[-1]
            if self.mode == "payloadfirst_fail":
                question_at = text.find("What is")
                fact_at = text.find("inspection code is")
                if fact_at != -1 and question_at > fact_at:
                    answer = "wrong-0000"
        else:
            answer = "unknown"
        boundary = list(self.encode("endmark"))
        out = list(self.encode(answer)) + boundary
        return Generation(
            token_ids=tuple((out + boundary * max_new_tokens)[:max_new_tokens]),
            reason="limit",
        )


def _fuzz_placement_section(rng: random.Random) -> dict[str, Any]:
    # Coherent payloads come from the real stage + analysis over
    # a scripted backend; hostile mutations then try to break
    # them, and oracle and implementation must agree on every
    # rejection. A coherent foreign backend fingerprint exercises
    # backend drift rather than corruption.
    """Fuzz placement section."""
    formats = (
        Format(name="fz-alpha", bos="", user_prefix="u:", assistant_prefix="a:", turn_end="</s>"),
        Format(name="fz-beta", bos="", user_prefix="U>", assistant_prefix="A>", turn_end="<e>"),
    )
    backend = _FuzzPlacementBackend(rng.choice(["perfect", "payloadfirst_fail"]))
    seed = rng.choice([0, 1, 2])
    stage = experiments.placement_stage(
        backend,
        formats,
        boundary=tuple(backend.encode("endmark")),
        max_context=65536,
        seed=seed,
    )
    payload = experiments.placement_analysis(stage, tuple(fmt.name for fmt in formats))
    available = [
        entry
        for entry in payload.get("candidates", {}).values()
        if entry.get("status") == "available"
    ]
    if available and rng.random() < 0.12:
        record = available[0]["records"][0]
        cell = record["cells"].get("question_first")
        if cell is not None:
            cell["verdict"] = not cell["verdict"]  # hostile
    if available and rng.random() < 0.10:
        record = available[0]["records"][1]
        cell = record["cells"].get("payload_first")
        if cell is not None:
            cell["input_tokens"] = cell["input_tokens"] + 1  # hostile: token binding breaks
    if available and rng.random() < 0.10:
        available[0]["headline"]["classification"] = "bogus"  # hostile
    if payload.get("available") is True and rng.random() < 0.08:
        payload["protocol"]["fingerprint"] = "0" * 64  # hostile: digest off components
    if payload.get("available") is True and rng.random() < 0.10:
        protocol = dict(placement.protocol_object(seed, "sha256:other-backend"))
        protocol["fingerprint"] = placement.protocol_fingerprint(seed, "sha256:other-backend")
        payload["protocol"] = protocol  # coherent foreign backend
    return payload


class _FuzzRotationBackend:
    """Byte-level scripted backend for the rotation fuzz builder.
    The answer is the asked topic's planted access code parsed
    out of the decoded prompt; ``slot_fail`` mode answers
    wrongly whenever the asked fact is not the first block of
    the span, which produces genuine rotation flips. The
    absent arm carries no asked fact and the closed-book arm
    carries no fact at all, so both answer "unknown" and never
    trip the screen in either mode.
    """

    fingerprint = "sha256:fuzz-rotation"

    def __init__(self, mode: str):
        """Init  ."""
        self.mode = mode

    @staticmethod
    def encode(text):
        """Encode."""
        return tuple(text.encode("utf-8"))

    @staticmethod
    def decode(ids):
        """Decode."""
        return bytes(int(i) % 256 for i in ids).decode("utf-8", "replace")

    def generate(self, input_ids, *, max_new_tokens, seed=None, cancel=None):
        """Generate."""
        import re as _re

        text = self.decode(list(input_ids))
        asked = _re.search(r"What is the (.+?) access code\?", text)
        answer = "unknown"
        if asked is not None:
            topic = asked.group(1)
            facts = _re.findall(r"The (.+?) access code is ([a-z]+-\d+)\.", text)
            for fact_topic, value in facts:
                if fact_topic == topic:
                    answer = value
                    if self.mode == "slot_fail" and facts[0][0] != topic:
                        answer = "wrong-0000"
                    break
        boundary = list(self.encode("endmark"))
        out = list(self.encode(answer)) + boundary
        return Generation(
            token_ids=tuple((out + boundary * max_new_tokens)[:max_new_tokens]),
            reason="limit",
        )


def _fuzz_rotation_section(rng: random.Random) -> dict[str, Any]:
    # Coherent payloads come from the real stage + analysis over
    # a scripted backend; hostile mutations then try to break
    # them, and oracle and implementation must agree on every
    # rejection. A coherent foreign backend fingerprint exercises
    # backend drift rather than corruption.
    """Fuzz rotation section."""
    formats = (
        Format(name="fz-alpha", bos="", user_prefix="u:", assistant_prefix="a:", turn_end="</s>"),
        Format(name="fz-beta", bos="", user_prefix="U>", assistant_prefix="A>", turn_end="<e>"),
    )
    backend = _FuzzRotationBackend(rng.choice(["perfect", "slot_fail"]))
    seed = rng.choice([0, 1, 2])
    stage = experiments.rotation_stage(
        backend,
        formats,
        boundary=tuple(backend.encode("endmark")),
        max_context=65536,
        seed=seed,
    )
    payload = experiments.rotation_analysis(stage, tuple(fmt.name for fmt in formats))
    available = [
        entry
        for entry in payload.get("candidates", {}).values()
        if entry.get("status") == "available"
    ]
    if available and rng.random() < 0.12:
        record = available[0]["records"][0]
        cell = record["cells"].get("rot_1")
        if cell is not None:
            cell["verdict"] = not cell["verdict"]  # hostile
    if available and rng.random() < 0.10:
        record = available[0]["records"][1]
        cell = record["cells"].get("mir_4")
        if cell is not None:
            cell["input_tokens"] = cell["input_tokens"] + 1  # hostile: token binding breaks
    if available and rng.random() < 0.10:
        record = available[0]["records"][2]
        record["arm_inventory_digests"]["rot_2"] = "0" * 64  # hostile: construction forged
    if available and rng.random() < 0.10:
        available[0]["classification"] = "bogus"  # hostile
    if payload.get("available") is True and rng.random() < 0.08:
        payload["protocol"]["fingerprint"] = "0" * 64  # hostile: digest off components
    if payload.get("available") is True and rng.random() < 0.10:
        protocol = dict(rotation.protocol_object(seed, "sha256:other-backend"))
        protocol["fingerprint"] = rotation.protocol_fingerprint(seed, "sha256:other-backend")
        payload["protocol"] = protocol  # coherent foreign backend
    return payload


class _FuzzChoiceAuditBackend:
    """Byte-level scripted backend for the choice-audit fuzz
    builder. ``echo`` mode favours the actual next byte at every
    position (every choice scores near-perfectly, totals track
    length); ``flat`` mode returns uniform logits (per-token
    means tie exactly across choices). Both produce coherent
    payloads through the real stage, exercising ties and
    length-driven rankings without a checkpoint.
    """

    fingerprint = "sha256:fuzz-choiceaudit"

    def __init__(self, mode: str):
        """Init  ."""
        self.mode = mode

    @staticmethod
    def encode(text):
        """Encode."""
        return tuple(text.encode("utf-8"))

    @staticmethod
    def decode(ids):
        """Decode."""
        return bytes(int(i) % 256 for i in ids).decode("utf-8")

    def forward_logits(self, input_ids):
        """Forward logits."""
        import torch

        seq = list(input_ids)
        logits = torch.zeros((len(seq), 256))
        if self.mode == "echo":
            for pos in range(len(seq) - 1):
                logits[pos][seq[pos + 1]] = 6.0
        return logits


def _fuzz_choiceaudit_section(rng: random.Random) -> dict[str, Any]:
    # Coherent payloads come from the real stage + analysis over
    # a scripted backend; hostile mutations then try to break
    # them, and oracle and implementation must agree on every
    # rejection. A coherent foreign backend fingerprint exercises
    # backend drift rather than corruption.
    """Fuzz choiceaudit section."""
    formats = (
        Format(name="fz-alpha", bos="", user_prefix="u:", assistant_prefix="a:", turn_end="</s>"),
        Format(name="fz-beta", bos="", user_prefix="U>", assistant_prefix="A>", turn_end="<e>"),
    )
    cases = tuple(
        Fixture(
            id=f"cf{i}",
            messages=(Message(role="user", content=f"question {i}?"),),
            expected=answer,
            group=group,
        )
        for i, (answer, group) in enumerate(
            [
                ("42", "numbers"),
                ("Paris", "geo"),
                ("blue", "colors"),
                ("17", "numbers"),
                ("London", "geo"),
                ("red", "colors"),
                ("100", "numbers"),
                ("Berlin", "geo"),
            ]
        )
    )
    backend = _FuzzChoiceAuditBackend(rng.choice(["echo", "flat"]))
    search = []
    for name in ("fz-alpha", "fz-beta"):
        observations = []
        for case in cases:
            observations.append(
                {
                    "fixture": case.id,
                    "input_ids": list(backend.encode(f"prompt {case.id} for {name}")),
                    "status": "completed",
                    "exact_match": (name == "fz-alpha") == (case.id in ("cf0", "cf1", "cf2")),
                    "boundary_found": True,
                }
            )
        search.append(
            {
                "format": {"name": name},
                "summary": {"aggregate_eligible": True},
                "observations": observations,
            }
        )
    stage = experiments.choiceaudit_stage(backend, formats, cases, search)
    payload = experiments.choiceaudit_analysis(
        stage, "compatible_on_fixture_set", "fz-alpha", tuple(fmt.name for fmt in formats)
    )
    available = [
        entry
        for entry in payload.get("candidates", {}).values()
        if entry.get("status") == "available"
    ]
    if available and rng.random() < 0.12:
        available[0]["accuracy"]["total"]["correct"] += 1  # hostile: aggregate off records
    if available and rng.random() < 0.10:
        record = available[0]["records"][0]
        if record["cells"]:
            record["cells"][0]["total"] = -3.5  # hostile: value off recompute
    if payload.get("available") is True and rng.random() < 0.08:
        payload["protocol"]["fingerprint"] = "0" * 64  # hostile: digest off components
    if payload.get("available") is True and rng.random() < 0.10:
        ids = [entry["fixture"] for entry in payload["fixtures"]]
        protocol = dict(
            choiceaudit.protocol_object(ids, payload["fixtures"], "sha256:other-backend")
        )
        protocol["fingerprint"] = choiceaudit.protocol_fingerprint(
            ids, payload["fixtures"], "sha256:other-backend"
        )
        payload["protocol"] = protocol  # coherent foreign backend
    if available and rng.random() < 0.08:
        name = next(iter(payload["candidates"]))
        payload["candidates"][name] = {
            "status": "unavailable",
            "detail": "native_search_unavailable",
            "records": [],
        }  # coherent status change
    return payload


class _FuzzPrecisionBackend:
    """Byte-level scripted backend for the precision fuzz
    builder. Plays one arm: answers from a fixed map keyed by
    the question text in the rendered prompt, emitting the
    answer plus the chr(7) boundary token. ``echo`` forward
    logits favour the actual next byte so teacher-forced
    logprobs are real floats through the real scoring path.
    """

    def __init__(self, answers, *, dtype, fingerprint):
        """Init  ."""
        self.answers = answers
        self.dtype = dtype
        self.fingerprint = fingerprint
        self.declared_dtype = "float32"

    @staticmethod
    def encode(text):
        """Encode."""
        return tuple(text.encode("utf-8"))

    @staticmethod
    def decode(ids):
        """Decode."""
        return bytes(int(i) % 256 for i in ids).decode("utf-8", "replace")

    def generate(self, prompt_ids, *, max_new_tokens, seed, cancel):
        """Generate."""
        from model_protocol_lab.core import Generation

        prompt = self.decode(prompt_ids)
        answer = ""
        for question, expected in self.answers.items():
            if question in prompt:
                answer = expected
                break
        ids = tuple((answer + "\x07").encode("utf-8"))
        # reason "limit" contractually means the cap was reached:
        # pad past the boundary token up to the cap.
        ids = ids + (0,) * max(0, max_new_tokens - len(ids))
        return Generation(token_ids=ids, reason="limit")

    @staticmethod
    def forward_logits(input_ids):
        """Forward logits."""
        import torch

        seq = list(input_ids)
        logits = torch.zeros((len(seq), 256))
        for pos in range(len(seq) - 1):
            logits[pos][seq[pos + 1]] = 6.0
        return logits


def _fuzz_precision_section(rng: random.Random) -> dict[str, Any]:
    # Coherent payloads come from the real stage over scripted
    # backends (a real run_experiment for the native side);
    # hostile mutations then try to break them, and oracle and
    # implementation must agree on every rejection. A coherent
    # foreign bf16 fingerprint exercises backend drift rather
    # than corruption.
    """Fuzz precision section."""
    formats = (
        Format(name="fz-alpha", bos="", user_prefix="u:", assistant_prefix="a:", turn_end="</s>"),
        Format(name="fz-beta", bos="", user_prefix="U>", assistant_prefix="A>", turn_end="<e>"),
    )
    cases = tuple(
        Fixture(
            id=f"pf{i}",
            messages=(Message(role="user", content=f"question {i}?"),),
            expected=answer,
            group=group,
        )
        for i, (answer, group) in enumerate(
            [
                ("42", "numbers"),
                ("Paris", "geo"),
                ("blue", "colors"),
                ("17", "numbers"),
                ("London", "geo"),
                ("red", "heldout"),
            ]
        )
    )
    answers = {f"question {i}?": case.expected for i, case in enumerate(cases)}
    native = _FuzzPrecisionBackend(
        answers, dtype="float32", fingerprint="sha256:fuzz-precision-native"
    )
    search_cases = cases[:5]
    # The native result is hand-built (the choice-audit fuzz
    # pattern): recorded completed observations whose answers
    # make fz-alpha the unique winner, so the stage harvests
    # real fp32 cells without a full experiment run per report.
    search = []
    for fmt in formats:
        observations = []
        for case in search_cases:
            answer = case.expected
            if fmt.name == "fz-beta" and case.id == "pf2":
                answer = "wrong-native"
            observations.append(
                {
                    "fixture": case.id,
                    "status": "completed",
                    "answer": answer,
                    "exact_match": answer == case.expected,
                    "boundary_found": True,
                    "input_ids": list(native.encode(f"prompt {case.id} for {fmt.name}")),
                    "output_ids": list(native.encode(answer + "\x07")),
                }
            )
        search.append(
            {
                "format": {"name": fmt.name},
                "summary": {"aggregate_eligible": True},
                "observations": observations,
            }
        )
    result = {
        "decision": "compatible_on_fixture_set",
        "selected_candidate": "fz-alpha",
        "search": search,
    }
    bf16_answers = dict(answers)
    if rng.random() < 0.5:
        bf16_answers["question 1?"] = "wrong-one"
    if rng.random() < 0.3:
        bf16_answers["question 4?"] = "wrong-two"
    bf16 = _FuzzPrecisionBackend(
        bf16_answers, dtype="bfloat16", fingerprint="sha256:fuzz-precision-bf16"
    )
    payload = experiments.precision_stage(
        native, bf16, formats, search_cases, result, boundary=(7,), max_new_tokens=16, seed=0
    )
    blocks = payload.get("candidates", {})
    available = [b for b in blocks.values() if b.get("cells", {}).get("paired")]
    if available and rng.random() < 0.12:
        available[0]["cells"]["paired"] += 1  # hostile: count off records
    if payload.get("records") and rng.random() < 0.10:
        record = payload["records"][0]
        if record["bf16"] is not None:
            record["bf16"]["verdict"] = not record["bf16"][
                "verdict"
            ]  # hostile: verdict off binding
    if payload.get("available") is True and rng.random() < 0.08:
        payload["protocol"]["fingerprint"] = "0" * 64  # hostile: digest off components
    if payload.get("available") is True and rng.random() < 0.08:
        payload["selection"]["survived"] = False  # hostile: verdict off recompute
    if payload.get("available") is True and rng.random() < 0.10:
        protocol = dict(
            precision.protocol_object(
                seed=0,
                max_new_tokens=16,
                fixture_ids=[case.id for case in search_cases],
                candidates=[fmt.name for fmt in formats],
                boundaries=payload["protocol"]["boundaries"],
                fp32_fingerprint=native.fingerprint,
                bf16_fingerprint="sha256:other-bf16",
                declared_dtype="float32",
                native_dtype="float32",
                arm_dtype="bfloat16",
            )
        )
        protocol["fingerprint"] = precision.protocol_fingerprint(protocol)
        payload["protocol"] = protocol  # coherent foreign bf16 fingerprint
    return payload


class _FuzzConfidenceBackend:
    """Scripted backend for the confidence fuzz builder.

    Operator fixtures: the native prompt (no suffix marker) gets
    the plain answer; the treated prompt gets the per-fixture
    shape the builder chose (compliant correct/incorrect, missing
    line, malformed line, trailing prose). Generated cases: the
    prompt is parsed -- when the payload binds the asked key the
    backend answers the stated value at high confidence; on the
    donor-key control it answers ``unknown`` at a mode-dependent
    confidence (low when faithful, high when theatrical).
    """

    def __init__(self, native_answers, shapes, *, control_confidence, fingerprint):
        """Init  ."""
        self.native_answers = native_answers
        self.shapes = shapes
        self.control_confidence = control_confidence
        self.fingerprint = fingerprint
        self.dtype = "float32"
        self.declared_dtype = "float32"

    @staticmethod
    def encode(text):
        """Encode."""
        return tuple(text.encode("utf-8"))

    @staticmethod
    def decode(ids):
        """Decode."""
        return bytes(int(i) % 256 for i in ids).decode("utf-8", "replace")

    def generate(self, prompt_ids, *, max_new_tokens, seed, cancel):
        """Generate."""
        from model_protocol_lab.core import Generation

        prompt = self.decode(prompt_ids)
        answer = ""
        bound = re.search(r"the code for (key-[0-9a-f]+) is ([0-9a-f]+)\.", prompt)
        asked = re.search(r"What is the code for (key-[0-9a-f]+)\?", prompt)
        if bound and asked:
            if bound.group(1) == asked.group(1):
                answer = f"{bound.group(2)}\nConfidence: 95"
            else:
                answer = f"unknown\nConfidence: {self.control_confidence}"
        else:
            treated = "Confidence: <number>" in prompt
            for question, text in self.native_answers.items():
                if question in prompt:
                    answer = self.shapes.get(question, text) if treated else text
                    break
        ids = tuple((answer + "\x07").encode("utf-8"))
        ids = ids + (0,) * max(0, max_new_tokens - len(ids))
        return Generation(token_ids=ids, reason="limit")


def _fuzz_confidence_section(rng: random.Random) -> dict[str, Any]:
    # Coherent payloads come from the real stage over a scripted
    # backend (hand-built native search, the precision fuzz
    # pattern); hostile mutations then try to break them, and
    # oracle and implementation must agree on every rejection.
    # A coherent foreign backend fingerprint exercises backend
    # drift rather than corruption.
    """Fuzz confidence section."""
    formats = (
        Format(name="fz-alpha", bos="", user_prefix="u:", assistant_prefix="a:", turn_end="</s>"),
        Format(name="fz-beta", bos="", user_prefix="U>", assistant_prefix="A>", turn_end="<e>"),
    )
    cases = tuple(
        Fixture(
            id=f"pf{i}",
            messages=(Message(role="user", content=f"question {i}?"),),
            expected=answer,
            group=group,
        )
        for i, (answer, group) in enumerate(
            [
                ("42", "numbers"),
                ("Paris", "geo"),
                ("blue", "colors"),
                ("17", "numbers"),
                ("London", "geo"),
                ("red", "heldout"),
            ]
        )
    )
    search_cases = cases[:5]
    native_answers = {f"question {i}?": case.expected for i, case in enumerate(cases)}
    shape_pool = [
        lambda exp: f"{exp}\nConfidence: 90",
        lambda exp: f"{exp}\nConfidence: 40",
        lambda exp: "wrong\nConfidence: 80",
        lambda exp: exp,
        lambda exp: f"{exp}\nConfidence: 85 ",
        lambda exp: f"{exp}\nConfidence: 70\nConfidence: 60",
        lambda exp: f"{exp}\nConfidence: 55\nprose after",
    ]
    shapes = {}
    for i, case in enumerate(search_cases):
        shapes[f"question {i}?"] = rng.choice(shape_pool)(case.expected)
    backend = _FuzzConfidenceBackend(
        native_answers,
        shapes,
        control_confidence=rng.choice([5, 95]),
        fingerprint="sha256:fuzz-confidence",
    )
    search = []
    for fmt in formats:
        observations = []
        for case in search_cases:
            answer = case.expected
            if fmt.name == "fz-beta" and case.id == "pf2":
                answer = "wrong-native"
            observations.append(
                {
                    "fixture": case.id,
                    "status": "completed",
                    "answer": answer,
                    "exact_match": answer == case.expected,
                    "boundary_found": True,
                    "input_ids": list(backend.encode(f"prompt {case.id} for {fmt.name}")),
                    "output_ids": list(backend.encode(answer + "\x07")),
                }
            )
        search.append(
            {
                "format": {"name": fmt.name},
                "summary": {"aggregate_eligible": True},
                "observations": observations,
            }
        )
    result = {
        "decision": "compatible_on_fixture_set",
        "selected_candidate": "fz-alpha",
        "search": search,
    }
    seed = rng.choice([0, 1, 2])
    payload = experiments.confidence_stage(
        backend, formats, search_cases, result, boundary=(7,), max_new_tokens=16, seed=seed
    )
    if payload.get("available") is True and rng.random() < 0.12:
        for record in payload["records"]:
            cell = record["elicited"]
            if type(cell) is dict and cell["code"] == "ok":
                cell["confidence"] = 100 - cell["confidence"]  # hostile: scalar off line
                break
    if payload.get("available") is True and rng.random() < 0.10:
        for record in payload["records"]:
            cell = record["elicited"]
            if type(cell) is dict:
                cell["verdict"] = not cell["verdict"]  # hostile: verdict off binding
                break
    if payload.get("available") is True and rng.random() < 0.08:
        payload["protocol"]["fingerprint"] = "0" * 64  # hostile: digest off components
    if payload.get("available") is True and rng.random() < 0.10:
        block = payload["candidates"]["fz-alpha"]["generated"]["knowability"]
        block["falsified"] = not block["falsified"] if block["falsified"] is not None else True
    if payload.get("available") is True and rng.random() < 0.08:
        payload["control_records"][0]["expected_digest"] = "0" * 64  # hostile: case binding
    if payload.get("available") is True and rng.random() < 0.10:
        protocol = dict(
            confidence.protocol_object(
                seed=seed,
                max_new_tokens=16,
                fixture_ids=[case.id for case in search_cases],
                candidates=[fmt.name for fmt in formats],
                boundaries=payload["protocol"]["boundaries"],
                backend_fingerprint="sha256:other-backend",
                screens=payload["protocol"]["screens"],
                cases=payload["protocol"]["control_cases"],
            )
        )
        protocol["fingerprint"] = confidence.protocol_fingerprint(protocol)
        payload["protocol"] = protocol  # coherent foreign backend fingerprint
    return payload


def _fuzz_numform_section(rng: random.Random) -> dict[str, Any]:
    # Coherent payloads come from the real stage + analysis over
    # a scripted backend; hostile mutations then try to break
    # them, and oracle and implementation must agree on every
    # rejection. A coherent foreign backend fingerprint exercises
    # backend drift rather than corruption.
    """Fuzz numform section."""
    formats = (
        Format(name="fz-alpha", bos="", user_prefix="u:", assistant_prefix="a:", turn_end="</s>"),
        Format(name="fz-beta", bos="", user_prefix="U>", assistant_prefix="A>", turn_end="<e>"),
    )
    mode = rng.choice(["faithful", "plain", "wrong_value", "flip_label"])
    seed = rng.choice([0, 1, 2])
    backend = _FuzzNumformBackend(mode)
    stage = experiments.numform_stage(
        backend,
        formats,
        boundary=tuple(backend.encode("<|im_end|>")),
        max_new_tokens=64,
        max_context=8192,
        seed=seed,
    )
    payload = experiments.numform_analysis(stage, tuple(fmt.name for fmt in formats))
    available = [
        entry
        for entry in payload.get("candidates", {}).values()
        if entry.get("status") == "available"
    ]
    if available and rng.random() < 0.12:
        record = available[0]["records"][0]
        if record["cell"] is not None:
            record["cell"]["verdict"] = not record["cell"]["verdict"]  # hostile
    if available and rng.random() < 0.10:
        aggregate = available[0]["aggregates"]["copy"]["plain"]
        aggregate["n_exact"] = aggregate.get("n_exact", 0) + 1  # hostile
    if payload.get("available") is True and rng.random() < 0.08:
        payload["protocol"]["fingerprint"] = "0" * 64  # hostile
    if payload.get("available") is True and rng.random() < 0.08:
        payload["grouping"]["classification"] = "single_digit"  # hostile
    if payload.get("available") is True and rng.random() < 0.10:
        protocol = dict(numform.protocol_object(seed, "sha256:other-backend"))
        protocol["fingerprint"] = numform.protocol_fingerprint(seed, "sha256:other-backend")
        payload["protocol"] = protocol  # coherent foreign backend
    if available and rng.random() < 0.08:
        name = next(iter(payload["candidates"]))
        payload["candidates"][name] = {
            "status": "unavailable",
            "detail": "arm_evaluation_failed",
            "records": [],
        }  # coherent status change
    return payload


def _fuzz_cache_profile_section(rng: random.Random) -> dict[str, Any]:
    # Coherent cells are produced by the real implementation over
    # random prompt sets (guaranteeing the integer identities and
    # ordering invariants hold); hostile mutations then try to break
    # them, and oracle/implementation must agree on every rejection.
    """Fuzz cache profile section."""
    per_candidate = {}
    for name in ("alpha", "beta"):
        if rng.random() < 0.1:
            per_candidate[name] = None
            continue
        prompts = [
            [rng.randint(0, 7) for _ in range(rng.randint(1, 12))] for _ in range(rng.randint(0, 5))
        ]
        entry = cacheprofile.candidate_cache_profile(prompts, block_size=rng.choice([2, 4, 16]))
        entry["n"] = entry["n_evaluable"] + rng.randint(0, 2)
        if rng.random() < 0.1:
            entry["n"] = 0  # hostile: n < n_evaluable
        if rng.random() < 0.1:
            entry["achievable_block_fraction"] = 1.5  # hostile: out of range
        if rng.random() < 0.1:
            entry["achievable_tokens"] = entry["cacheable_tokens"]  # identity broken
        if rng.random() < 0.1:
            entry["potential_token_fraction"] = 0.0  # hostile: ordering violated
        if rng.random() < 0.1:
            entry[rng.choice(["n", 7])] = -1  # hostile: negative / bad key
        per_candidate[name] = entry
    if rng.random() < 0.15:
        per_candidate[rng.choice([7, 8.5])] = {"n": 1}  # hostile: non-string key
    return {
        "available": True,
        "block_size": rng.choice([2, 16, 16, 0, -1]),
        "n_observations": rng.choice([0, 5, 10, -1]),
        "per_candidate": per_candidate,
        "most_sharable": rng.choice([[], ["alpha"], ["beta"], [7]]),
        "least_sharable": rng.choice([[], ["alpha"]]),
        "sharability_gap": rng.choice([0.0, 0.25, 0.5, None, -0.1]),
        "uniform": rng.choice([True, False, None, "yes"]),
        "excluded_candidates": rng.choice([[], ["alpha"], ["beta"]]),
        "evaluable_mismatch": rng.choice([True, False, "x"]),
    }


def _fuzz_confidentiality_section(rng: random.Random) -> dict[str, Any]:
    # Coherent cells are produced by the real aggregation over random
    # paired detections (so every identity and invariant holds);
    # hostile mutations then try to break them, and oracle and
    # implementation must agree on every rejection.
    """Fuzz confidentiality section."""
    from model_protocol_lab import confidentiality as _cf

    def _det(leak_p: float) -> dict[str, Any]:
        """Det."""
        if rng.random() < leak_p:
            run = rng.choice([0.75, 1.0])
            return {"run_fraction": run, "leak": True, "verbatim": run >= 1.0}
        run = rng.choice([0.0, 0.25, 0.5])
        return {"run_fraction": run, "leak": False, "verbatim": False}

    per_candidate = {}
    for name in ("alpha", "beta"):
        if rng.random() < 0.1:
            per_candidate[name] = None
            continue
        pairs = [
            {
                "treat": _det(0.3) if rng.random() < 0.95 else None,
                "ctrl": _det(0.5) if rng.random() < 0.95 else None,
                "offset_treat": rng.randint(20, 60),
                "offset_ctrl": rng.randint(0, 10),
            }
            for _ in range(12)
        ]
        entry = _cf.confidentiality_cell(pairs)
        if rng.random() < 0.1:
            entry["gap"] = 0.99  # hostile: stale/inconsistent gap
        if rng.random() < 0.1:
            entry["ctrl_leaks"] = 99  # hostile: leaks above denominator
        if rng.random() < 0.1:
            entry["treat_rate"] = 1.5  # hostile: rate out of range
        if rng.random() < 0.1:
            entry["base_rate_zero"] = not entry["base_rate_zero"]  # hostile: invariant flip
        if rng.random() < 0.1:
            entry[rng.choice(["n", 7])] = -1  # hostile: negative / bad key
        per_candidate[name] = entry
    if rng.random() < 0.15:
        per_candidate[rng.choice([7, 8.5])] = {"n": 12}  # hostile: non-string key
    profile = _cf.confidentiality_profile(("alpha", "beta"), per_candidate, _cf.protocol_dict())
    profile["n_fixtures"] = rng.choice([12, 12, 12, 0, -1, "12"])
    if rng.random() < 0.2:
        profile["protocol"] = {"fingerprint": rng.choice(["x" * 64, 7, None])}
    return profile


def _fuzz_answer_collision_section(rng: random.Random) -> dict[str, Any]:
    # Coherent cells are produced by the real aggregation over random
    # scored observations (so every identity and invariant holds);
    # hostile mutations then try to break them, and oracle and
    # implementation must agree on every rejection.
    """Fuzz answer collision section."""
    from model_protocol_lab import collision as _ac

    classes = ("collision", "ends_at_marker", "intact", "anomalous")

    def _obs(cls: str) -> dict[str, Any]:
        """Obs."""
        return {
            "class": cls,
            "cut": rng.randint(0, 40),
            "exact_match": cls == "intact",
            "verbatim": True,
            "occurrences": rng.choice([0, 1, 2]),
        }

    per_candidate = {}
    for name in ("alpha", "beta"):
        if rng.random() < 0.1:
            per_candidate[name] = None
            continue
        per_fixture = []
        for index in range(21):
            reserved_cls = rng.choice(classes)
            subword_state = rng.choice(["scored", "no_pair", "not_applicable"])
            per_fixture.append(
                {
                    "id": f"f{index}",
                    "reserved": _obs(reserved_cls),
                    "subword": _obs(rng.choice(classes)) if subword_state == "scored" else None,
                    "arm_state": {"reserved": "scored", "subword": subword_state},
                }
            )
        entry = _ac.collision_cell(per_fixture, n_fixtures=21, boundary_len=rng.choice([1, 2]))
        if rng.random() < 0.1:
            entry["reserved"]["collision_rate"] = 0.99  # hostile: stale rate
        if rng.random() < 0.1:
            entry["reserved"]["n_collision"] = True  # hostile: bool count
        if rng.random() < 0.1:
            entry["uniform_intact"] = not entry["uniform_intact"]  # hostile: invariant flip
        if rng.random() < 0.1:
            entry[rng.choice(["n", 7])] = -1  # hostile: negative / bad key
        per_candidate[name] = entry
    if rng.random() < 0.15:
        per_candidate[rng.choice([7, 8.5])] = {"n": 21}  # hostile: non-string key
    profile = _ac.answer_collision_profile(("alpha", "beta"), per_candidate, _ac.protocol_dict())
    profile["n_fixtures"] = rng.choice([21, 21, 21, 0, -1, "21"])
    if rng.random() < 0.1:
        profile["colliding_candidates"] = ["ghost"]  # hostile: stale colliding set
    if rng.random() < 0.1:
        profile["max_collision_rate"] = 0.01  # hostile: stale max rate
    if rng.random() < 0.1:
        profile["worst_candidate"] = "ghost"  # hostile: stale worst candidate
    if rng.random() < 0.2:
        profile["protocol"] = {"fingerprint": rng.choice(["x" * 64, 7, None])}
    return profile


def _fuzz_ws_canonical_section(rng: random.Random) -> dict[str, Any]:
    """Fuzz ws canonical section."""
    mutations = ["trailing_space", "leading_space", "trailing_newline"]
    if rng.random() < 0.3:
        mutations.append("strip_trailing_space")
    per_candidate = {}
    for name in ("alpha", "beta"):
        cells = {}
        for mutation in mutations:
            cells[mutation] = {
                "evaluable": rng.choice([True, True, False]),
                "n_fixtures": rng.randint(1, 4),
                "n_evaluable": rng.randint(0, 3),
                "n_ids_divergent": rng.randint(0, 3),
                "ids_divergence_rate": rng.choice([0.0, 0.5, 1.0, None]),
                "n_verdict_flips": rng.randint(0, 2),
                "verdict_flip_rate": rng.choice([0.0, 0.5, 1.0, None]),
                "n_same_ids_verdict_flips": rng.randint(0, 1),
                "flips": rng.choice([[], ["f0"]]),
                "fragile": rng.choice([True, False, None]),
            }
        if rng.random() < 0.15:
            cells[rng.choice(["mystery", 7])] = {"ids_divergence_rate": 0.5}
        per_candidate[name] = {
            "per_mutation": cells,
            "aggregate": {"n_fragile_mutations": rng.randint(0, 2)},
            "whitespace_robust": rng.choice([True, False, None]),
            "detail": None,
        }
    if rng.random() < 0.15:
        per_candidate[rng.choice([7, 8.5])] = {"per_mutation": {}}
    return {
        "available": True,
        "mutations": mutations,
        "per_candidate": per_candidate,
        "fragile_candidates": rng.choice([[], ["alpha"], ["alpha", "beta"]]),
        "excluded_candidates": [],
    }


def _fuzz_mutate(rng: random.Random, report: dict[str, Any]) -> None:
    """Scripted small mutations turning a clone into a divergent report."""
    for _ in range(rng.randint(0, 4)):
        op = rng.randrange(19)
        if op == 0:
            report["decision"] = rng.choice(["select_best", "abstain", "mutated"])
        elif op == 1:
            report["selected_candidate"] = rng.choice(["alpha", "beta/mut", "", None, "zzz"])
        elif op == 2:
            report["search"].append(_fuzz_entry(rng))
        elif op == 3 and report["search"]:
            victim = rng.choice(report["search"])
            if isinstance(victim, dict) and isinstance(victim.get("summary"), dict):
                victim["summary"]["exact_and_boundary_passes"] = rng.choice([0, 1, 2, 5])
        elif op == 4 and report["search"]:
            victim = rng.choice(report["search"])
            if isinstance(victim, dict) and isinstance(victim.get("observations"), list):
                victim["observations"].append(_fuzz_row(rng))
        elif op == 5 and report["search"]:
            victim = rng.choice(report["search"])
            if isinstance(victim, dict):
                victim.pop("observations", None)
        elif op == 6:
            report["max_new_tokens"] = rng.choice([32, 64, 256, None])
        elif op == 7:
            report["backend_fingerprint"] = rng.choice(["fp1", "fpX", None])
        elif op == 8 and report["search"]:
            victim = rng.choice(report["search"])
            if isinstance(victim, dict):
                victim["boundary_token_ids"] = [rng.randint(0, 99)]
        elif op == 9:
            held = report.get("held_out")
            if isinstance(held, dict) and isinstance(held.get("observations"), list):
                held["observations"].append(_fuzz_row(rng))
        elif op == 10:
            transfer = report.get("transfer")
            matrix = transfer.get("matrix") if isinstance(transfer, dict) else None
            if isinstance(matrix, dict) and matrix:
                cand = rng.choice(sorted(matrix))
                row = matrix[cand]
                if isinstance(row, dict) and row:
                    ckpt = rng.choice(sorted(row))
                    row[ckpt] = rng.choice(["transfers", "fails_there", "mutated_cell"])
        elif op == 11:
            contam = report.get("contamination")
            pairs = contam.get("pairs") if isinstance(contam, dict) else None
            if isinstance(pairs, list) and pairs:
                victim = rng.choice(pairs)
                if isinstance(victim, dict):
                    victim["tier"] = rng.choice(["clean", "duplicate", "mutated_tier"])
        elif op == 12:
            robust = report.get("selection_robustness")
            per_sem = robust.get("per_semantic") if isinstance(robust, dict) else None
            if isinstance(per_sem, dict) and per_sem:
                key = rng.choice(sorted(per_sem))
                cell = per_sem[key]
                if isinstance(cell, dict):
                    cell["decision"] = rng.choice(["selected", "tie", "mutated"])
        elif op == 13:
            pert = report.get("perturbation")
            per_var = pert.get("per_variant") if isinstance(pert, dict) else None
            if isinstance(per_var, dict) and per_var:
                key = rng.choice(sorted(k for k in per_var if type(k) is str))
                cell = per_var[key]
                if isinstance(cell, dict):
                    cell["winner"] = rng.choice(["alpha", "zzz", None])
        elif op == 14:
            heal = report.get("token_healing")
            per_cand = heal.get("per_candidate") if isinstance(heal, dict) else None
            if isinstance(per_cand, dict) and per_cand:
                key = rng.choice(sorted(k for k in per_cand if type(k) is str))
                cell = per_cand[key]
                if isinstance(cell, dict):
                    cell["healed"] = rng.choice([True, False, None, "mutated"])
        elif op == 15:
            hier = report.get("hierarchy")
            per_cond = hier.get("per_condition") if isinstance(hier, dict) else None
            if isinstance(per_cond, dict) and per_cond:
                key = rng.choice(sorted(k for k in per_cond if type(k) is str))
                cell = per_cond[key]
                if isinstance(cell, dict):
                    cell["winner"] = rng.choice(["alpha", "zzz", None])
        elif op == 16:
            port = report.get("portfolio")
            oracle = port.get("oracle") if isinstance(port, dict) else None
            if isinstance(oracle, dict):
                oracle["gap"] = rng.choice([0.0, 0.99, "mutated"])
        elif op == 17:
            pos = report.get("position")
            primacy = pos.get("primacy_index") if isinstance(pos, dict) else None
            if isinstance(primacy, dict) and primacy:
                key = rng.choice(sorted(k for k in primacy if type(k) is str))
                primacy[key] = rng.choice([1.0, -1.0, 0.0, None, "mutated"])
        elif op == 18:
            sc = report.get("stop_cause")
            per_cand = sc.get("per_candidate") if isinstance(sc, dict) else None
            if isinstance(per_cand, dict) and per_cand:
                key = rng.choice(sorted(k for k in per_cand if type(k) is str))
                cell = per_cand[key]
                if isinstance(cell, dict):
                    rates = cell.get("cause_rates")
                    if isinstance(rates, dict):
                        rates["eos_id"] = rng.choice([0.0, 1.0, "mutated"])


class DiffreportsFuzzTest(unittest.TestCase):
    """Diffreports fuzz tests."""

    def test_differential_fuzz_oracle_vs_implementation(self):
        """Differential fuzz oracle vs implementation."""
        rng = random.Random(0xC67)  # noqa: S311 -- deterministic test seed
        for i in range(400):
            if rng.random() < 0.5:
                old = _fuzz_report(rng)
                new = copy.deepcopy(old)
                _fuzz_mutate(rng, new)
            else:
                old, new = _fuzz_report(rng), _fuzz_report(rng)
            for all_fixtures in (False, True):
                labels = ("run-α", "run-β→") if i % 3 == 0 else ("old", "new")
                with self.subTest(i=i, all_fixtures=all_fixtures):
                    old_c, new_c = copy.deepcopy(old), copy.deepcopy(new)
                    got = diffreports.diff_reports(
                        old_c,
                        new_c,
                        all_fixtures=all_fixtures,
                        old_label=labels[0],
                        new_label=labels[1],
                    )
                    want = _oracle_diff(
                        old,
                        new,
                        all_fixtures=all_fixtures,
                        old_label=labels[0],
                        new_label=labels[1],
                    )
                    self.assertEqual(got, want)
                    if i % 25 == 0:
                        # determinism: same inputs -> identical output
                        again = diffreports.diff_reports(
                            old_c,
                            new_c,
                            all_fixtures=all_fixtures,
                            old_label=labels[0],
                            new_label=labels[1],
                        )
                        self.assertEqual(again, got)
                        # inputs are never mutated by the comparison
                        self.assertEqual(old_c, old)
                        self.assertEqual(new_c, new)


# ---------------------------------------------------------------------------
# Mutation gate: semantic mutants on /tmp copies of diffreports.py.
# Repo sources are never touched; each pinned scenario must pass on the
# real module (non-vacuous) and FAIL on the mutant (caught).
# ---------------------------------------------------------------------------


class DiffreportsMutationGateTest(unittest.TestCase):
    """Diffreports mutation gate tests."""

    def _run_gate(
        self, name: str, old_text: str, new_text: str, scenario, replacements: int = 1
    ) -> None:
        """Run gate."""
        source = Path(diffreports.__file__).read_text(encoding="utf-8")
        self.assertEqual(source.count(old_text), replacements, f"mutant {name}: anchor not unique")
        mutated = source.replace(old_text, new_text)
        self.assertNotEqual(mutated, source, f"mutant {name}: replacement changed nothing")
        with tempfile.TemporaryDirectory(prefix="c67_mut_", dir="/tmp") as tmpdir:
            mod_path = os.path.join(tmpdir, f"diffreports_{name}.py")
            with open(mod_path, "w", encoding="utf-8") as handle:
                handle.write(mutated)
            spec = importlib.util.spec_from_file_location(
                f"model_protocol_lab.diffreports_mut_{name}", mod_path
            )
            assert spec is not None and spec.loader is not None
            mutant = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mutant)
            scenario(diffreports)  # passes on the real module: gate is non-vacuous
            with self.assertRaises(AssertionError, msg=f"mutant {name} was NOT caught"):
                scenario(mutant)

    def test_mutant_dropped_planned_drift_caught(self):
        """Mutant dropped planned drift caught."""

        def scenario(mod):
            """Scenario."""
            old, new = _base_report(), _base_report()
            new["search"][0]["summary"]["planned"] = 4
            new["search"][0]["summary"]["exact_and_boundary_passes"] = 4
            diff = mod.diff_reports(old, new)
            self.assertEqual(diff["config_drift"], [{"field": "planned:alpha", "old": 2, "new": 4}])
            self.assertFalse(diff["summary"]["identical"])

        self._run_gate(
            "drop_planned_drift",
            '                {"field": f"planned:{name}", "old": old_planned[name], '
            '"new": new_planned[name]}\n',
            '                {"field": "MUTANT", "old": None, "new": None}  # MUTANT-1\n',
            scenario,
        )

    def test_mutant_flipped_identical_caught(self):
        """Mutant flipped identical caught."""

        def scenario(mod):
            """Scenario."""
            old, new = _base_report(), _base_report()
            diff = mod.diff_reports(old, new)
            self.assertTrue(diff["summary"]["identical"])
            self.assertEqual(diff["fixture_diffs"], [])

        self._run_gate(
            "flip_identical",
            "    identical = (\n"
            "        not decision_changed\n"
            "        and not winner_changed\n"
            "        and not verdict_changes\n"
            "        and not changed_fixture_diffs\n"
            "        and not config_drift\n"
            "        and not evidence_asymmetric\n"
            '        and not frontier_diff["changed"]\n'
            "        and not frontier_asymmetric\n"
            '        and not held_out_floor_diff["changed"]\n'
            "        and not held_out_floor_asymmetric\n"
            '        and not boundary_audit_diff["changed"]\n'
            "        and not boundary_audit_asymmetric\n"
            '        and not robustness_diff["changed"]\n'
            "        and not robustness_asymmetric\n"
            '        and not transfer_diff["changed"]\n'
            "        and not transfer_asymmetric\n"
            '        and not contamination_diff["changed"]\n'
            "        and not contamination_asymmetric\n"
            '        and not perturbation_diff["changed"]\n'
            "        and not perturbation_asymmetric\n"
            '        and not healing_diff["changed"]\n'
            "        and not healing_asymmetric\n"
            '        and not hierarchy_diff["changed"]\n'
            "        and not hierarchy_asymmetric\n"
            '        and not prefill_diff["changed"]\n'
            "        and not prefill_asymmetric\n"
            '        and not portfolio_diff["changed"]\n'
            "        and not portfolio_asymmetric\n"
            '        and not synthhistory_diff["changed"]\n'
            "        and not synthhistory_asymmetric\n"
            '        and not batchinv_diff["changed"]\n'
            "        and not batchinv_asymmetric\n"
            '        and not position_diff["changed"]\n'
            "        and not position_asymmetric\n"
            '        and not stopcause_diff["changed"]\n'
            "        and not stopcause_asymmetric\n"
            '        and not wscan_diff["changed"]\n'
            "        and not wscan_asymmetric\n"
            '        and not structured_diff["changed"]\n'
            "        and not structured_asymmetric\n"
            '        and not deliforge_diff["changed"]\n'
            "        and not deliforge_asymmetric\n"
            '        and not fewshot_diff["changed"]\n'
            "        and not fewshot_asymmetric\n"
            '        and not faketurn_diff["changed"]\n'
            "        and not faketurn_asymmetric\n"
            '        and not toolcall_diff["changed"]\n'
            "        and not toolcall_asymmetric\n"
            '        and not verbosity_diff["changed"]\n'
            "        and not verbosity_asymmetric\n"
            '        and not refusals_diff["changed"]\n'
            "        and not refusals_asymmetric\n"
            '        and not cache_profile_diff["changed"]\n'
            "        and not cache_profile_asymmetric\n"
            '        and not template_robustness_diff["changed"]\n'
            "        and not template_robustness_asymmetric\n"
            '        and not ablation_diff["changed"]\n'
            "        and not ablation_asymmetric\n"
            '        and not _payload_section(section_diffs, "confidentiality")["changed"]\n'
            '        and not _payload_section(section_asymmetric, "confidentiality")\n'
            '        and not answer_collision_diff["changed"]\n'
            "        and not answer_collision_asymmetric\n"
            '        and not scorerlens_diff["changed"]\n'
            "        and not scorerlens_asymmetric\n"
            '        and not aliasing_diff["changed"]\n'
            "        and not aliasing_asymmetric\n"
            '        and not repetition_diff["changed"]\n'
            "        and not repetition_asymmetric\n"
            '        and not distraction_diff["changed"]\n'
            "        and not distraction_asymmetric\n"
            '        and not leading_diff["changed"]\n'
            "        and not leading_asymmetric\n"
            '        and not numform_diff["changed"]\n'
            "        and not numform_asymmetric\n"
            '        and not dateline_diff["changed"]\n'
            "        and not dateline_asymmetric\n"
            '        and not multiturn_diff["changed"]\n'
            "        and not multiturn_asymmetric\n"
            '        and not needle_diff["changed"]\n'
            "        and not needle_asymmetric\n"
            '        and not placement_diff["changed"]\n'
            "        and not placement_asymmetric\n"
            '        and not canonical_form_diff["changed"]\n'
            "        and not canonical_form_asymmetric\n"
            '        and not instruction_distance_diff["changed"]\n'
            "        and not instruction_distance_asymmetric\n"
            '        and not history_frame_diff["changed"]\n'
            "        and not history_frame_asymmetric\n"
            '        and not prune_sweep_diff["changed"]\n'
            "        and not prune_sweep_asymmetric\n"
            '        and not record_notation_diff["changed"]\n'
            "        and not record_notation_asymmetric\n"
            '        and not invisible_chars_diff["changed"]\n'
            "        and not invisible_chars_asymmetric\n"
            '        and not confusable_skeleton_diff["changed"]\n'
            "        and not confusable_skeleton_asymmetric\n"
            '        and not choiceaudit_diff["changed"]\n'
            "        and not choiceaudit_asymmetric\n"
            '        and not precision_diff["changed"]\n'
            "        and not precision_asymmetric\n"
            '        and not confidence_diff["changed"]\n'
            "        and not confidence_asymmetric\n"
            '        and not rotation_diff["changed"]\n'
            "        and not rotation_asymmetric\n"
            '        and not density_diff["changed"]\n'
            "        and not density_asymmetric\n"
            '        and not field_order_diff["changed"]\n'
            "        and not field_order_asymmetric\n"
            "    )\n",
            "    identical = (\n"
            "        decision_changed\n"
            "        or winner_changed\n"
            "        or verdict_changes\n"
            "        or changed_fixture_diffs\n"
            "        or config_drift\n"
            "        or evidence_asymmetric\n"
            '        or frontier_diff["changed"]\n'
            "        or frontier_asymmetric\n"
            '        or held_out_floor_diff["changed"]\n'
            "        or held_out_floor_asymmetric\n"
            '        or boundary_audit_diff["changed"]\n'
            "        or boundary_audit_asymmetric\n"
            '        or robustness_diff["changed"]\n'
            "        or robustness_asymmetric\n"
            '        or transfer_diff["changed"]\n'
            "        or transfer_asymmetric\n"
            '        or contamination_diff["changed"]\n'
            "        or contamination_asymmetric\n"
            '        or perturbation_diff["changed"]\n'
            "        or perturbation_asymmetric\n"
            '        or healing_diff["changed"]\n'
            "        or healing_asymmetric\n"
            '        or hierarchy_diff["changed"]\n'
            "        or hierarchy_asymmetric\n"
            '        or prefill_diff["changed"]\n'
            "        or prefill_asymmetric\n"
            '        or portfolio_diff["changed"]\n'
            "        or portfolio_asymmetric\n"
            '        or synthhistory_diff["changed"]\n'
            "        or synthhistory_asymmetric\n"
            '        or batchinv_diff["changed"]\n'
            "        or batchinv_asymmetric\n"
            '        or position_diff["changed"]\n'
            "        or position_asymmetric\n"
            '        or stopcause_diff["changed"]\n'
            "        or stopcause_asymmetric\n"
            '        or wscan_diff["changed"]\n'
            "        or wscan_asymmetric\n"
            '        or structured_diff["changed"]\n'
            "        or structured_asymmetric\n"
            '        or deliforge_diff["changed"]\n'
            "        or deliforge_asymmetric\n"
            '        or fewshot_diff["changed"]\n'
            "        or fewshot_asymmetric\n"
            '        or faketurn_diff["changed"]\n'
            "        or faketurn_asymmetric\n"
            '        or toolcall_diff["changed"]\n'
            "        or toolcall_asymmetric\n"
            '        or verbosity_diff["changed"]\n'
            "        or verbosity_asymmetric\n"
            '        or refusals_diff["changed"]\n'
            "        or refusals_asymmetric\n"
            '        or cache_profile_diff["changed"]\n'
            "        or cache_profile_asymmetric\n"
            '        or template_robustness_diff["changed"]\n'
            "        or template_robustness_asymmetric\n"
            '        or ablation_diff["changed"]\n'
            "        or ablation_asymmetric\n"
            '        or _payload_section(section_diffs, "confidentiality")["changed"]\n'
            '        or _payload_section(section_asymmetric, "confidentiality")\n'
            '        or answer_collision_diff["changed"]\n'
            "        or answer_collision_asymmetric\n"
            '        or scorerlens_diff["changed"]\n'
            "        or scorerlens_asymmetric\n"
            '        or aliasing_diff["changed"]\n'
            "        or aliasing_asymmetric\n"
            '        or repetition_diff["changed"]\n'
            "        or repetition_asymmetric\n"
            '        or distraction_diff["changed"]\n'
            "        or distraction_asymmetric\n"
            '        or leading_diff["changed"]\n'
            "        or leading_asymmetric\n"
            '        or numform_diff["changed"]\n'
            "        or numform_asymmetric\n"
            '        or dateline_diff["changed"]\n'
            "        or dateline_asymmetric\n"
            '        or multiturn_diff["changed"]\n'
            "        or multiturn_asymmetric\n"
            '        or needle_diff["changed"]\n'
            "        or needle_asymmetric\n"
            '        or placement_diff["changed"]\n'
            "        or placement_asymmetric\n"
            '        or canonical_form_diff["changed"]\n'
            "        or canonical_form_asymmetric\n"
            '        or instruction_distance_diff["changed"]\n'
            "        or instruction_distance_asymmetric\n"
            '        or history_frame_diff["changed"]\n'
            "        or history_frame_asymmetric\n"
            '        or prune_sweep_diff["changed"]\n'
            "        or prune_sweep_asymmetric\n"
            '        or record_notation_diff["changed"]\n'
            "        or record_notation_asymmetric\n"
            '        or invisible_chars_diff["changed"]\n'
            "        or invisible_chars_asymmetric\n"
            '        or confusable_skeleton_diff["changed"]\n'
            "        or confusable_skeleton_asymmetric\n"
            '        or choiceaudit_diff["changed"]\n'
            "        or choiceaudit_asymmetric\n"
            '        or precision_diff["changed"]\n'
            "        or precision_asymmetric\n"
            '        or confidence_diff["changed"]\n'
            "        or confidence_asymmetric\n"
            '        or rotation_diff["changed"]\n'
            "        or rotation_asymmetric\n"
            '        or density_diff["changed"]\n'
            "        or density_asymmetric\n"
            '        or field_order_diff["changed"]\n'
            "        or field_order_asymmetric\n"
            "    )\n",
            scenario,
        )

    def test_mutant_ignored_evidence_gating_caught(self):
        """Mutant ignored evidence gating caught."""

        def scenario(mod):
            """Scenario."""
            old, new = _base_report(), _base_report()
            for entry in new["search"]:
                entry.pop("observations", None)
            new["held_out"].pop("observations", None)
            diff = mod.diff_reports(old, new)
            self.assertEqual(diff["fixture_diffs"], [])
            self.assertEqual(diff["config_drift"], [])
            self.assertTrue(diff["summary"]["evidence_asymmetric"])
            self.assertFalse(diff["summary"]["identical"])

        self._run_gate(
            "ignore_evidence_gating",
            '    if evidence_present["old"] and evidence_present["new"]:\n',
            "    if True:  # MUTANT-3: evidence gating ignored\n",
            scenario,
            replacements=2,  # fixture-diff gate and fixture-set-drift gate
        )

    def test_mutant_verdict_semantics_caught(self):
        """Mutant verdict semantics caught."""

        def scenario(mod):
            """Scenario."""
            old, new = _base_report(), _base_report()
            new["search"][1]["summary"]["exact_and_boundary_passes"] = 2
            diff = mod.diff_reports(old, new)
            self.assertEqual(
                diff["candidate_verdicts"],
                [{"candidate": "beta/mut", "stage": "search", "old": "fail", "new": "pass"}],
            )
            self.assertFalse(diff["summary"]["identical"])

        self._run_gate(
            "verdict_pass_iff_any",
            '    return "pass" if passes == planned else "fail"\n',
            '    return "pass" if passes > 0 else "fail"  # MUTANT-4\n',
            scenario,
        )


class CanonicalFormOracleAgreementTests(unittest.TestCase):
    """Cycle 209: the canonical-form oracle must agree with the
    real diff on REAL stage payloads, not just on the neither
    stubs the fuzzer produces (the fuzzer emits no
    canonical_form section, so without this test the oracle's
    recompute depth would be unproven).
    """

    def _backend(self, mode="perfect", fail_count=0):
        """Backend."""
        import re as _re

        class _WordBackend:
            """Word backend."""

            fingerprint = "sha256:test-cf-oracle"

            def __init__(self):
                """Init  ."""
                self.mode = mode
                self.fail_count = fail_count
                self.nfd_fails = 0
                self.vocab: dict[str, int] = {}
                self.rev: dict[int, str] = {}

            def encode(self, text):
                """Encode."""
                ids = []
                for token in text.replace("\n", " \n ").split(" "):
                    if not token:
                        continue
                    if token not in self.vocab:
                        self.vocab[token] = len(self.vocab) + 1
                        self.rev[self.vocab[token]] = token
                    ids.append(self.vocab[token])
                return tuple(ids)

            def decode(self, ids):
                """Decode."""
                return " ".join(self.rev.get(int(i), "?") for i in ids)

            def generate(self, input_ids, *, max_new_tokens, seed=None, cancel=None):
                """Generate."""
                import unicodedata as _unicodedata

                text = self.decode(list(input_ids))
                found = _re.findall(r"inspection code is (\S+?)\.", text)
                answer = found[-1] if found else "unknown"
                if (
                    self.mode == "nfd_fail"
                    and _unicodedata.normalize("NFC", text) != text
                    and self.nfd_fails < self.fail_count
                ):
                    self.nfd_fails += 1
                    answer = "wrong-0000"
                boundary = self.encode("endmark")[0]
                out = list(self.encode(answer)) + [boundary]
                out = (out + [boundary] * max_new_tokens)[:max_new_tokens]
                return Generation(token_ids=tuple(out), reason="limit")

        return _WordBackend()

    @staticmethod
    def _payload(backend):
        """Payload."""
        fmt = Format(
            name="cf-alpha",
            bos="",
            user_prefix="u:",
            assistant_prefix="a:",
            turn_end="</s>",
        )
        stage = experiments.canonicalform_stage(
            backend,
            (fmt,),
            boundary=tuple(backend.encode("endmark")),
            max_context=65536,
            seed=0,
        )
        return experiments.canonicalform_analysis(stage, ("cf-alpha",))

    @staticmethod
    def _pair(old_payload, new_payload):
        """Pair."""
        old = _base_report()
        new = _base_report()
        old["canonical_form"] = old_payload
        new["canonical_form"] = new_payload
        return old, new

    def test_identical_real_payloads_agree(self):
        """Identical real payloads agree."""
        payload = self._payload(self._backend())
        old, new = self._pair(payload, copy.deepcopy(payload))
        oracle = _oracle_diff(old, new)
        real = diffreports.diff_reports(old, new)
        self.assertEqual(oracle["canonical_form"], real["canonical_form"])
        self.assertEqual(
            oracle["summary"]["canonical_form_changed"],
            real["summary"]["canonical_form_changed"],
        )
        self.assertFalse(real["summary"]["canonical_form_changed"])
        self.assertTrue(real["summary"]["identical"])
        self.assertTrue(oracle["summary"]["identical"])

    def test_changed_real_payloads_agree(self):
        """Changed real payloads agree."""
        first = self._payload(self._backend())
        second = self._payload(self._backend("nfd_fail", 3))
        old, new = self._pair(first, second)
        oracle = _oracle_diff(old, new)
        real = diffreports.diff_reports(old, new)
        self.assertEqual(oracle["canonical_form"], real["canonical_form"])
        self.assertTrue(real["canonical_form"]["changed"])
        self.assertFalse(real["summary"]["identical"])

    def test_tampered_payload_agreement_is_old(self):
        """Tampered payload agreement is old."""
        first = self._payload(self._backend())
        bad = copy.deepcopy(first)
        bad["candidates"]["cf-alpha"]["records"][0]["cells"]["nfc"]["verdict"] = False
        old, new = self._pair(first, bad)
        oracle = _oracle_diff(old, new)
        real = diffreports.diff_reports(old, new)
        self.assertEqual(oracle["canonical_form"], real["canonical_form"])
        self.assertEqual(real["canonical_form"]["available"], "old")
        self.assertFalse(real["canonical_form"]["changed"])


class InstructionDistanceOracleAgreementTests(unittest.TestCase):
    """Cycle 218: the instruction-distance oracle must agree
    with the real diff on REAL stage payloads, not just on the
    neither stubs the fuzzer produces (the fuzzer emits no
    instruction_distance section, so without this test the
    oracle's recompute depth would be unproven).
    """

    @staticmethod
    def _payload(mode):
        """Payload."""
        from tests.test_cycle218_instructiondistance import _payload, _WordBackend

        return _payload(_WordBackend(mode))

    @staticmethod
    def _pair(old_payload, new_payload):
        """Pair."""
        old = _base_report()
        new = _base_report()
        old["instruction_distance"] = old_payload
        new["instruction_distance"] = new_payload
        return old, new

    def test_identical_real_payloads_agree(self):
        """Identical real payloads agree."""
        payload = self._payload("perfect")
        old, new = self._pair(payload, copy.deepcopy(payload))
        oracle = _oracle_diff(old, new)
        real = diffreports.diff_reports(old, new)
        self.assertEqual(oracle["instruction_distance"], real["instruction_distance"])
        self.assertEqual(
            oracle["summary"]["instruction_distance_changed"],
            real["summary"]["instruction_distance_changed"],
        )
        self.assertFalse(real["summary"]["instruction_distance_changed"])
        self.assertTrue(real["summary"]["identical"])
        self.assertTrue(oracle["summary"]["identical"])

    def test_changed_real_payloads_agree(self):
        """Changed real payloads agree."""
        first = self._payload("perfect")
        second = self._payload("drop")
        old, new = self._pair(first, second)
        oracle = _oracle_diff(old, new)
        real = diffreports.diff_reports(old, new)
        self.assertEqual(oracle["instruction_distance"], real["instruction_distance"])
        self.assertTrue(real["instruction_distance"]["changed"])
        self.assertFalse(real["summary"]["identical"])

    def test_tampered_payload_agreement_is_old(self):
        """Tampered payload agreement is old."""
        first = self._payload("perfect")
        bad = copy.deepcopy(first)
        bad["candidates"]["chatml"]["records"][0]["cells"]["adj"]["verdict"] = False
        old, new = self._pair(first, bad)
        oracle = _oracle_diff(old, new)
        real = diffreports.diff_reports(old, new)
        self.assertEqual(oracle["instruction_distance"], real["instruction_distance"])
        self.assertEqual(real["instruction_distance"]["available"], "old")
        self.assertFalse(real["instruction_distance"]["changed"])


class HistoryFrameOracleAgreementTests(unittest.TestCase):
    """Cycle 219: the history-frame oracle must agree with
    the real diff on REAL stage payloads, not just on the
    neither stubs the fuzzer produces (the fuzzer emits no
    history_frame section, so without this test the oracle's
    recompute depth -- including the amendment-A4 id-digest
    gate -- would be unproven).
    """

    @staticmethod
    def _payload(mode):
        """Payload."""
        from tests.test_cycle219_historyframe import _payload, _WordBackend

        return _payload(_WordBackend(mode))

    @staticmethod
    def _pair(old_payload, new_payload):
        """Pair."""
        old = _base_report()
        new = _base_report()
        old["history_frame"] = old_payload
        new["history_frame"] = new_payload
        return old, new

    def test_identical_real_payloads_agree(self):
        """Identical real payloads agree."""
        payload = self._payload("perfect")
        old, new = self._pair(payload, copy.deepcopy(payload))
        oracle = _oracle_diff(old, new)
        real = diffreports.diff_reports(old, new)
        self.assertEqual(oracle["history_frame"], real["history_frame"])
        self.assertEqual(
            oracle["summary"]["history_frame_changed"],
            real["summary"]["history_frame_changed"],
        )
        self.assertFalse(real["summary"]["history_frame_changed"])
        self.assertTrue(real["summary"]["identical"])
        self.assertTrue(oracle["summary"]["identical"])

    def test_changed_real_payloads_agree(self):
        """Changed real payloads agree."""
        first = self._payload("perfect")
        second = self._payload("frame_drop")
        old, new = self._pair(first, second)
        oracle = _oracle_diff(old, new)
        real = diffreports.diff_reports(old, new)
        self.assertEqual(oracle["history_frame"], real["history_frame"])
        self.assertTrue(real["history_frame"]["changed"])
        self.assertFalse(real["summary"]["identical"])

    def test_tampered_payload_agreement_is_old(self):
        """Tampered payload agreement is old."""
        first = self._payload("perfect")
        bad = copy.deepcopy(first)
        bad["candidates"]["chatml"]["records"][0]["cells"]["own"]["verdict"] = False
        old, new = self._pair(first, bad)
        oracle = _oracle_diff(old, new)
        real = diffreports.diff_reports(old, new)
        self.assertEqual(oracle["history_frame"], real["history_frame"])
        self.assertEqual(real["history_frame"]["available"], "old")
        self.assertFalse(real["history_frame"]["changed"])

    def test_a4_digest_forgery_agreement_is_old(self):
        # Amendment A4: flipping a record's foreign id digest
        # to the own digest (faking the ids-identical gate)
        # must invalidate the payload on both sides of the
        # oracle boundary.
        """A4 digest forgery agreement is old."""
        first = self._payload("frame_drop")
        bad = copy.deepcopy(first)
        for record in bad["candidates"]["chatml"]["records"]:
            record["id_digests"]["foreign_llama3"] = record["id_digests"]["own"]
        old, new = self._pair(first, bad)
        oracle = _oracle_diff(old, new)
        real = diffreports.diff_reports(old, new)
        self.assertEqual(oracle["history_frame"], real["history_frame"])
        self.assertEqual(real["history_frame"]["available"], "old")


class PruneSweepOracleAgreementTests(unittest.TestCase):
    """Cycle 220: the prune-sweep oracle must agree with the
    real diff on REAL stage payloads, not just on the
    neither stubs the fuzzer produces (the fuzzer emits no
    prune_sweep section, so without this test the oracle's
    recompute depth -- including the content-digest
    rebinding and the ids-identical derivation -- would be
    unproven).
    """

    @staticmethod
    def _payload(mode):
        """Payload."""
        from tests.test_cycle220_prunesweep import _payload, _WordBackend

        return _payload(_WordBackend(mode))

    @staticmethod
    def _pair(old_payload, new_payload):
        """Pair."""
        old = _base_report()
        new = _base_report()
        old["prune_sweep"] = old_payload
        new["prune_sweep"] = new_payload
        return old, new

    def test_identical_real_payloads_agree(self):
        """Identical real payloads agree."""
        payload = self._payload("perfect")
        old, new = self._pair(payload, copy.deepcopy(payload))
        oracle = _oracle_diff(old, new)
        real = diffreports.diff_reports(old, new)
        self.assertEqual(oracle["prune_sweep"], real["prune_sweep"])
        self.assertEqual(
            oracle["summary"]["prune_sweep_changed"],
            real["summary"]["prune_sweep_changed"],
        )
        self.assertFalse(real["summary"]["prune_sweep_changed"])
        self.assertTrue(real["summary"]["identical"])
        self.assertTrue(oracle["summary"]["identical"])

    def test_changed_real_payloads_agree(self):
        """Changed real payloads agree."""
        first = self._payload("perfect")
        second = self._payload("dose72_drop")
        old, new = self._pair(first, second)
        oracle = _oracle_diff(old, new)
        real = diffreports.diff_reports(old, new)
        self.assertEqual(oracle["prune_sweep"], real["prune_sweep"])
        self.assertTrue(real["prune_sweep"]["changed"])
        self.assertFalse(real["summary"]["identical"])

    def test_tampered_payload_agreement_is_old(self):
        """Tampered payload agreement is old."""
        first = self._payload("perfect")
        bad = copy.deepcopy(first)
        bad["candidates"]["chatml"]["records"][0]["cells"]["dose72"]["verdict"] = False
        old, new = self._pair(first, bad)
        oracle = _oracle_diff(old, new)
        real = diffreports.diff_reports(old, new)
        self.assertEqual(oracle["prune_sweep"], real["prune_sweep"])
        self.assertEqual(real["prune_sweep"]["available"], "old")
        self.assertFalse(real["prune_sweep"]["changed"])

    def test_ids_forgery_agreement_is_old(self):
        # Amendment B: flipping a record's dose72 id digest
        # to the dose0 digest (faking the ids-identical
        # exclusion) must invalidate the payload on both
        # sides of the oracle boundary.
        """Ids forgery agreement is old."""
        first = self._payload("dose72_drop")
        bad = copy.deepcopy(first)
        for record in bad["candidates"]["chatml"]["records"]:
            record["id_digests"]["dose72"] = record["id_digests"]["dose0"]
        old, new = self._pair(first, bad)
        oracle = _oracle_diff(old, new)
        real = diffreports.diff_reports(old, new)
        self.assertEqual(oracle["prune_sweep"], real["prune_sweep"])
        self.assertEqual(real["prune_sweep"]["available"], "old")


class RecordNotationOracleAgreementTests(unittest.TestCase):
    """Cycle 221: the record-notation oracle must agree with
    the real diff on REAL stage payloads, not just on the
    neither stubs the fuzzer produces (the fuzzer emits no
    record_notation section, so without this test the
    oracle's recompute depth -- including the frozen-RNG
    regeneration, the content-digest rebinding, the
    input-token binding, and the answer-text binding --
    would be unproven).
    """

    @staticmethod
    def _payload(mode):
        """Payload."""
        from tests.test_cycle221_recordnotation import _payload, _WordBackend

        return _payload(_WordBackend(mode))

    @staticmethod
    def _pair(old_payload, new_payload):
        """Pair."""
        old = _base_report()
        new = _base_report()
        old["record_notation"] = old_payload
        new["record_notation"] = new_payload
        return old, new

    def test_identical_real_payloads_agree(self):
        """Identical real payloads agree."""
        payload = self._payload("perfect")
        old, new = self._pair(payload, copy.deepcopy(payload))
        oracle = _oracle_diff(old, new)
        real = diffreports.diff_reports(old, new)
        self.assertEqual(oracle["record_notation"], real["record_notation"])
        self.assertEqual(
            oracle["summary"]["record_notation_changed"],
            real["summary"]["record_notation_changed"],
        )
        self.assertFalse(real["summary"]["record_notation_changed"])
        self.assertTrue(real["summary"]["identical"])
        self.assertTrue(oracle["summary"]["identical"])

    def test_changed_real_payloads_agree(self):
        """Changed real payloads agree."""
        first = self._payload("perfect")
        second = self._payload("json_drop")
        old, new = self._pair(first, second)
        oracle = _oracle_diff(old, new)
        real = diffreports.diff_reports(old, new)
        self.assertEqual(oracle["record_notation"], real["record_notation"])
        self.assertTrue(real["record_notation"]["changed"])
        self.assertFalse(real["summary"]["identical"])

    def test_tampered_payload_agreement_is_old(self):
        """Tampered payload agreement is old."""
        first = self._payload("perfect")
        bad = copy.deepcopy(first)
        bad["candidates"]["chatml"]["records"][0]["cells"]["prose"]["verdict"] = False
        old, new = self._pair(first, bad)
        oracle = _oracle_diff(old, new)
        real = diffreports.diff_reports(old, new)
        self.assertEqual(oracle["record_notation"], real["record_notation"])
        self.assertEqual(real["record_notation"]["available"], "old")
        self.assertFalse(real["record_notation"]["changed"])

    def test_input_tokens_forgery_agreement_is_old(self):
        # Amendment D: a cell whose input_tokens disagree
        # with the record's arm token count must invalidate
        # the payload on both sides of the oracle boundary.
        """Input tokens forgery agreement is old."""
        first = self._payload("perfect")
        bad = copy.deepcopy(first)
        bad["candidates"]["chatml"]["records"][0]["cells"]["prose"]["input_tokens"] += 1
        old, new = self._pair(first, bad)
        oracle = _oracle_diff(old, new)
        real = diffreports.diff_reports(old, new)
        self.assertEqual(oracle["record_notation"], real["record_notation"])
        self.assertEqual(real["record_notation"]["available"], "old")

    def test_answer_text_forgery_agreement_is_old(self):
        # A headline cell whose answer_text no longer hashes
        # to its answer_digest must invalidate the payload on
        # both sides of the oracle boundary.
        """Answer text forgery agreement is old."""
        first = self._payload("perfect")
        bad = copy.deepcopy(first)
        bad["candidates"]["chatml"]["records"][0]["cells"]["canonical_json"]["answer_text"] = (
            "tampered-0000"
        )
        old, new = self._pair(first, bad)
        oracle = _oracle_diff(old, new)
        real = diffreports.diff_reports(old, new)
        self.assertEqual(oracle["record_notation"], real["record_notation"])
        self.assertEqual(real["record_notation"]["available"], "old")


class InvisibleCharsOracleAgreementTests(unittest.TestCase):
    """Cycle 223: the invisible-chars oracle must agree with
    the real diff on REAL stage payloads, not just on the
    neither stubs the fuzzer produces (the fuzzer emits no
    invisible_chars section, so without this test the
    oracle's recompute depth -- including the frozen
    digest-derivation regeneration (amendment A1), the
    per-arm content-digest rebinding, the ids-flag
    derivation (amendment A3), and the answer-text binding
    -- would be unproven).
    """

    @staticmethod
    def _payload(mode):
        """Payload."""
        from tests.test_cycle223_invisiblechars import _payload, _WordBackend

        return _payload(_WordBackend(mode))

    @staticmethod
    def _pair(old_payload, new_payload):
        """Pair."""
        old = _base_report()
        new = _base_report()
        old["invisible_chars"] = old_payload
        new["invisible_chars"] = new_payload
        return old, new

    def test_identical_real_payloads_agree(self):
        """Identical real payloads agree."""
        payload = self._payload("perfect")
        old, new = self._pair(payload, copy.deepcopy(payload))
        oracle = _oracle_diff(old, new)
        real = diffreports.diff_reports(old, new)
        self.assertEqual(oracle["invisible_chars"], real["invisible_chars"])
        self.assertEqual(
            oracle["summary"]["invisible_chars_changed"],
            real["summary"]["invisible_chars_changed"],
        )
        self.assertFalse(real["summary"]["invisible_chars_changed"])
        self.assertTrue(real["summary"]["identical"])
        self.assertTrue(oracle["summary"]["identical"])

    def test_changed_real_payloads_agree(self):
        """Changed real payloads agree."""
        first = self._payload("perfect")
        second = self._payload("echo")
        old, new = self._pair(first, second)
        oracle = _oracle_diff(old, new)
        real = diffreports.diff_reports(old, new)
        self.assertEqual(oracle["invisible_chars"], real["invisible_chars"])
        self.assertTrue(real["invisible_chars"]["changed"])
        self.assertFalse(real["summary"]["identical"])

    def test_tampered_payload_agreement_is_old(self):
        """Tampered payload agreement is old."""
        first = self._payload("perfect")
        bad = copy.deepcopy(first)
        bad["candidates"]["chatml"]["records"][0]["cells"]["zwsp"]["verdict"] = False
        old, new = self._pair(first, bad)
        oracle = _oracle_diff(old, new)
        real = diffreports.diff_reports(old, new)
        self.assertEqual(oracle["invisible_chars"], real["invisible_chars"])
        self.assertEqual(real["invisible_chars"]["available"], "old")
        self.assertFalse(real["invisible_chars"]["changed"])

    def test_ids_flag_forgery_agreement_is_old(self):
        # Amendment A3: a record whose stored ids_identical
        # flag disagrees with the derivation from its own
        # id digests must invalidate the payload on both
        # sides of the oracle boundary.
        """Ids flag forgery agreement is old."""
        first = self._payload("perfect")
        bad = copy.deepcopy(first)
        record = bad["candidates"]["chatml"]["records"][0]
        record["ids_identical"]["zwsp"] = not record["ids_identical"]["zwsp"]
        old, new = self._pair(first, bad)
        oracle = _oracle_diff(old, new)
        real = diffreports.diff_reports(old, new)
        self.assertEqual(oracle["invisible_chars"], real["invisible_chars"])
        self.assertEqual(real["invisible_chars"]["available"], "old")

    def test_answer_text_forgery_agreement_is_old(self):
        # A cell whose answer_text no longer hashes to its
        # answer_digest must invalidate the payload on both
        # sides of the oracle boundary.
        """Answer text forgery agreement is old."""
        first = self._payload("perfect")
        bad = copy.deepcopy(first)
        bad["candidates"]["chatml"]["records"][0]["cells"]["control"]["answer_text"] = (
            "tampered-0000"
        )
        old, new = self._pair(first, bad)
        oracle = _oracle_diff(old, new)
        real = diffreports.diff_reports(old, new)
        self.assertEqual(oracle["invisible_chars"], real["invisible_chars"])
        self.assertEqual(real["invisible_chars"]["available"], "old")


if __name__ == "__main__":
    unittest.main()
