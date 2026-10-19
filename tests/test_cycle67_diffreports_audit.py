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
import importlib.util
import os
import random
import tempfile
import unittest
from pathlib import Path
from typing import Any

from model_protocol_lab import diffreports

# ---------------------------------------------------------------------------
# Builders: canonical reports for the adversarial pairs.
# ---------------------------------------------------------------------------


def _obs(fixture: str, exact: bool = True, boundary: bool = True) -> dict[str, Any]:
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
        "template_robustness": {"available": "neither", "changed": False},
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
            "template_robustness_changed": False,
            "template_robustness_asymmetric": False,
            "identical": True,
        },
    }


# ---------------------------------------------------------------------------
# Independent oracle, written from the COMPARISON RULES above.
# Structure and control flow deliberately differ from the implementation
# (profiles-first, comprehensions instead of nested accumulation loops).
# ---------------------------------------------------------------------------


def _o_verdict(summary: Any) -> str:
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
    if type(entry) is not dict:
        return None
    fmt = entry.get("format")
    name = fmt.get("name") if type(fmt) is dict else None
    return name if type(name) is str and name else None


def _o_base_family(name: str) -> str:
    return name.split("/", 1)[0]


def _o_outcomes(observations: Any) -> tuple[dict[str, str], int]:
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
    winner = result.get("selected_candidate")
    if type(winner) is not str or not winner:
        return None, "absent"
    held = result.get("held_out")
    summary = held.get("summary") if type(held) is dict else None
    return winner, _o_verdict(summary)


def _o_evidence(result: dict[str, Any], stage: str) -> tuple[dict[str, dict[str, str]], int, bool]:
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
    o_trob, o_trob_changed, o_trob_asymmetric = _o_template_robustness(
        old.get("template_robustness"), new.get("template_robustness")
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
        and not o_trob_changed
        and not o_trob_asymmetric
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
        "template_robustness": o_trob,
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
            "template_robustness_changed": o_trob_changed,
            "template_robustness_asymmetric": o_trob_asymmetric,
            "identical": identical,
        },
    }


def _o_tr_valid(value: Any) -> bool:
    """Oracle template-robustness validator: mirrors _valid_template_robustness."""

    def _ok_rate(v: Any) -> bool:
        return v is None or (type(v) in (int, float) and 0.0 <= v <= 1.0)

    def _ok_report(rep: Any) -> bool:
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


# ---------------------------------------------------------------------------
# Adversarial pairs with exact expected outputs.
# ---------------------------------------------------------------------------


class DiffreportsAuditTest(unittest.TestCase):
    maxDiff = None

    def _pair(self, mutate=None):
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
        old, new = self._pair()
        self.assertEqual(
            diffreports.diff_reports(old, new), _identical_expected(all_fixtures=False)
        )

    def test_identical_reports_all_fixtures_lists_every_cell(self):
        old, new = self._pair()
        self.assertEqual(
            diffreports.diff_reports(old, new, all_fixtures=True),
            _identical_expected(all_fixtures=True),
        )

    def test_identical_reports_no_evidence_either_side(self):
        def strip(report):
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
        old, new = self._pair()
        new["search"] = list(reversed(new["search"]))
        for entry in new["search"]:
            entry["observations"] = list(reversed(entry["observations"]))
        new["held_out"]["observations"] = list(reversed(new["held_out"]["observations"]))
        self.assertEqual(
            diffreports.diff_reports(old, new), _identical_expected(all_fixtures=False)
        )

    def test_extra_unknown_top_level_keys_ignored(self):
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
        def empty(report):
            for entry in report["search"]:
                entry["observations"] = []
            report["held_out"]["observations"] = []

        def missing(report):
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
                "template_robustness_changed": False,
                "template_robustness_asymmetric": False,
                "identical": False,
            },
        )

    def test_minimal_pair_winner_flip_cascades_to_held_out(self):
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
        def flip(report):
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
        def flip(report):
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
        def flip(report):
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
        def change(report):
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
        old, new = self._pair(lambda r: r.update(backend_fingerprint="fp-2"))
        diff = diffreports.diff_reports(old, new)
        self.assertEqual(
            diff["config_drift"],
            [{"field": "backend_fingerprint", "old": "fp-1", "new": "fp-2"}],
        )
        self.assertFalse(diff["summary"]["identical"])

    def test_minimal_pair_candidate_rename_family_unchanged(self):
        def rename(report):
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
        def change(report):
            summary = report["search"][0]["summary"]
            summary["planned"] = 4
            summary["exact_and_boundary_passes"] = 4

        old, new = self._pair(change)
        diff = diffreports.diff_reports(old, new)
        self.assertEqual(diff["config_drift"], [{"field": "planned:alpha", "old": 2, "new": 4}])
        self.assertEqual(diff["candidate_verdicts"], [])  # still pass on both sides
        self.assertFalse(diff["summary"]["identical"])

    def test_minimal_pair_fixture_set_drift(self):
        def add(report):
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
        def change(report):
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
        def change(report):
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
        def strip(report):
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
        def strip(report):
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
        self.assertEqual(
            diffreports._verdict(
                {"planned": True, "exact_and_boundary_passes": True, "aggregate_eligible": True}
            ),
            "unknown",
        )

    def test_verdict_edges(self):
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
        def dup(report):
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
        def change(report):
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
        def change(report):
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
        def change(report):
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
        def change(report):
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

        old, new = self._pair(change)
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
        old, new = self._pair()
        text = diffreports.render_summary(diffreports.diff_reports(old, new))
        self.assertIn(
            "No differences: the two runs are identical on every compared axis "
            "(decision, winner, candidate verdicts, configuration, per-fixture evidence).",
            text,
        )

    def test_render_summary_identical_axes_without_evidence(self):
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
    return report


def _fuzz_position_section(rng: random.Random) -> dict[str, Any]:
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


def _fuzz_ws_canonical_section(rng: random.Random) -> dict[str, Any]:
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
    def test_differential_fuzz_oracle_vs_implementation(self):
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
    def _run_gate(
        self, name: str, old_text: str, new_text: str, scenario, replacements: int = 1
    ) -> None:
        source = Path(diffreports.__file__).read_text(encoding="utf-8")
        self.assertEqual(source.count(old_text), replacements, f"mutant {name}: anchor not unique")
        mutated = source.replace(old_text, new_text)
        self.assertNotEqual(mutated, source, f"mutant {name}: replacement changed nothing")
        with tempfile.TemporaryDirectory(prefix="c67_mut_", dir="/tmp") as tmpdir:
            mod_path = os.path.join(tmpdir, f"diffreports_{name}.py")
            with open(mod_path, "w", encoding="utf-8") as handle:
                handle.write(mutated)
            spec = importlib.util.spec_from_file_location(f"diffreports_mut_{name}", mod_path)
            assert spec is not None and spec.loader is not None
            mutant = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mutant)
            scenario(diffreports)  # passes on the real module: gate is non-vacuous
            with self.assertRaises(AssertionError, msg=f"mutant {name} was NOT caught"):
                scenario(mutant)

    def test_mutant_dropped_planned_drift_caught(self):
        def scenario(mod):
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
        def scenario(mod):
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
            '        and not template_robustness_diff["changed"]\n'
            "        and not template_robustness_asymmetric\n"
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
            "    )\n",
            scenario,
        )

    def test_mutant_ignored_evidence_gating_caught(self):
        def scenario(mod):
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
        def scenario(mod):
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


if __name__ == "__main__":
    unittest.main()
