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
            "identical": identical,
        },
    }


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
    return report


def _fuzz_mutate(rng: random.Random, report: dict[str, Any]) -> None:
    """Scripted small mutations turning a clone into a divergent report."""
    for _ in range(rng.randint(0, 4)):
        op = rng.randrange(10)
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
