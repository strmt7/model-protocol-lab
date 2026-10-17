"""Cycle 27: differential verification of the diff-reports drift logic.

This file is an INDEPENDENT reimplementation of the drift-detection
semantics in ``model_protocol_lab/diffreports.py``, derived from that
module's docstrings (not its code), plus a seeded fuzz harness that
compares the oracle's output against the real implementation over
randomized report pairs.

Scope: ``diff_reports`` (decision/winner flips, candidate-verdict flips,
per-fixture outcome diffs, config drift entries, summary counts) and the
helpers it composes: ``_verdict``, ``_search_verdicts``, ``_held_verdict``,
``_fixture_outcomes``, ``_stage_evidence``, ``_config_snapshot``,
``_drift``, ``_fixture_set_drift``. ``render_summary``/CLI plumbing is
covered by ``tests/test_cycle13_diffreports.py``.

DOCSTRING DISCREPANCIES RECORDED (code is taken as authoritative where
noted; see report):
1. (CYCLE 27 NOTE, FIXED IN CYCLE 28.) The ``_fixture_outcomes``
   docstring used to claim duplicate candidate names are first-wins in
   BOTH ``_search_verdicts`` and ``_stage_evidence``. The code disagrees:
   ``_search_verdicts`` is first-wins, but ``_stage_evidence``'s search
   branch has no repeat-name guard, so evidence for a duplicated
   candidate name is LAST-wins (and a later duplicate with no usable
   evidence silently keeps the earlier one). Cycle 28 corrected the
   docstring; the oracle below replicates the code.
   (``test_duplicate_candidate_name_evidence_last_wins`` pins this.)
2. (CYCLE 27 NOTE, FIXED IN CYCLE 28.) The ``_fixture_set_drift``
   docstring said "only when both sides carry evidence", which was true
   GLOBALLY but not per-stage: evidence gating in ``diff_reports`` is a
   single any-stage flag per side, so a run with search evidence vs a
   run with held-out evidence still emits per-stage
   ``fixtures:search``/``fixtures:held_out`` drift entries and "absent"
   fixture diffs on the empty stages. Cycle 28 corrected the docstring.
   The oracle replicates this; it is behavior the operators should be
   aware of, not a bug introduced here.

Deliberately NOT covered: ``load_results`` hostile-input handling and
``render_summary`` text (existing suite covers those).
"""

from __future__ import annotations

import copy
import json
import random
import unittest
from pathlib import Path
from typing import Any

from model_protocol_lab import diffreports as impl

# ---------------------------------------------------------------------------
# Independent oracle, written from the docstrings of diffreports.py.
# Naming and control flow deliberately differ from the implementation so a
# shared helper bug cannot hide. Any deliberate code-over-doc choice is
# marked with "CODE-DOC DIVERGENCE" below.
# ---------------------------------------------------------------------------


def _or_verdict(summary: Any) -> str:
    """Oracle for ``_verdict``: "pass" iff every planned fixture passed.

    "fail" covers partial passes and ineligible/incomplete runs
    (``aggregate_eligible`` False, ``planned`` <= 0); malformed summaries
    degrade to "unknown". Booleans are NOT ints (``type(x) is int``).
    """
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


def _or_candidate_name(entry: Any) -> str | None:
    if type(entry) is not dict:
        return None
    fmt = entry.get("format")
    if type(fmt) is not dict:
        return None
    name = fmt.get("name")
    return name if type(name) is str and name else None


def _or_search_verdicts(result: dict[str, Any]) -> tuple[dict[str, str], dict[str, int | None]]:
    """Oracle for ``_search_verdicts``: duplicate names are first-wins.

    ``planned`` is None when the summary is malformed/missing (it compares
    as its own distinct value so it still shows up in drift).
    """
    verdicts: dict[str, str] = {}
    planned_counts: dict[str, int | None] = {}
    for entry in result.get("search") or []:
        name = _or_candidate_name(entry)
        if name is None or name in verdicts:
            continue
        summary = entry.get("summary") if type(entry) is dict else None
        verdicts[name] = _or_verdict(summary)
        planned = summary.get("planned") if type(summary) is dict else None
        planned_counts[name] = planned if type(planned) is int else None
    return verdicts, planned_counts


def _or_held_verdict(result: dict[str, Any]) -> tuple[str | None, str]:
    """Oracle for ``_held_verdict``."""
    winner = result.get("selected_candidate")
    if type(winner) is not str or not winner:
        return None, "absent"
    held = result.get("held_out")
    summary = held.get("summary") if type(held) is dict else None
    return winner, _or_verdict(summary)


def _or_fixture_outcomes(observations: Any) -> tuple[dict[str, str], int]:
    """Oracle for ``_fixture_outcomes``: pass requires exact AND boundary.

    Malformed observations are skipped and counted. Duplicate fixture ids
    within one candidate are LAST-wins.
    """
    outcomes: dict[str, str] = {}
    skipped = 0
    if type(observations) is not list:
        return outcomes, 0
    for obs in observations:
        if (
            type(obs) is not dict
            or type(obs.get("fixture")) is not str
            or not obs["fixture"]
            or type(obs.get("exact_match")) is not bool
            or type(obs.get("boundary_found")) is not bool
        ):
            skipped += 1
            continue
        outcomes[obs["fixture"]] = (
            "pass" if (obs["exact_match"] and obs["boundary_found"]) else "fail"
        )
    return outcomes, skipped


def _or_stage_evidence(
    result: dict[str, Any], stage: str
) -> tuple[dict[str, dict[str, str]], int, bool]:
    """Oracle for ``_stage_evidence``.

    CODE-DOC DIVERGENCE (see module docstring): the ``_fixture_outcomes``
    docstring claims duplicate candidate names in search are first-wins;
    the code has no repeat-name guard in this function, so the oracle
    replicates the CODE: last entry with usable evidence wins for a
    duplicated name.
    """
    per_candidate: dict[str, dict[str, str]] = {}
    skipped = 0
    if stage == "search":
        for entry in result.get("search") or []:
            name = _or_candidate_name(entry)
            if name is None:
                continue
            observations = entry.get("observations") if type(entry) is dict else None
            outcomes, skipped_here = _or_fixture_outcomes(observations)
            skipped += skipped_here
            if type(observations) is list and outcomes:
                per_candidate[name] = outcomes
    else:
        winner = result.get("selected_candidate")
        held = result.get("held_out")
        observations = held.get("observations") if type(held) is dict else None
        outcomes, skipped = _or_fixture_outcomes(observations)
        if type(winner) is str and winner and type(observations) is list and outcomes:
            per_candidate[winner] = outcomes
    return per_candidate, skipped, bool(per_candidate)


def _or_base_family(name: str) -> str:
    return name.split("/", 1)[0]


def _or_config_snapshot(result: dict[str, Any]) -> dict[str, Any]:
    """Oracle for ``_config_snapshot``: families/candidates/boundaries.

    Names dedup first-occurrence; boundary lists must be all-int;
    ``max_new_tokens``/``backend_fingerprint`` normalize to None unless
    int/str respectively (bool/float are NOT int).
    """
    names: list[str] = []
    seen: set[str] = set()
    boundaries: dict[str, list[int]] = {}
    for entry in result.get("search") or []:
        name = _or_candidate_name(entry)
        if name is not None and name not in seen:
            seen.add(name)
            names.append(name)
        ids = entry.get("boundary_token_ids") if type(entry) is dict else None
        if name is not None and type(ids) is list and all(type(token) is int for token in ids):
            boundaries[name] = list(ids)
    max_new_tokens = result.get("max_new_tokens")
    fingerprint = result.get("backend_fingerprint")
    return {
        "families": sorted({_or_base_family(name) for name in names}),
        "candidates": sorted(names),
        "max_new_tokens": max_new_tokens if type(max_new_tokens) is int else None,
        "boundaries": {name: boundaries[name] for name in sorted(boundaries)},
        "backend_fingerprint": fingerprint if type(fingerprint) is str else None,
    }


def _or_drift(old: dict[str, Any], new: dict[str, Any]) -> list[dict[str, Any]]:
    """Oracle for ``_drift``: exact entries in fixed field order."""
    old_snap, new_snap = _or_config_snapshot(old), _or_config_snapshot(new)
    entries: list[dict[str, Any]] = []
    for field in (
        "families",
        "candidates",
        "max_new_tokens",
        "boundaries",
        "backend_fingerprint",
    ):
        if old_snap[field] != new_snap[field]:
            entries.append({"field": field, "old": old_snap[field], "new": new_snap[field]})
    return entries


def _or_fixture_set_drift(
    old_evidence: dict[str, dict[str, dict[str, str]]],
    new_evidence: dict[str, dict[str, dict[str, str]]],
) -> list[dict[str, Any]]:
    """Oracle for ``_fixture_set_drift``: per-stage sorted fixture-id sets."""
    entries: list[dict[str, Any]] = []
    for stage in ("search", "held_out"):
        old_ids = sorted({fixture for cand in old_evidence[stage].values() for fixture in cand})
        new_ids = sorted({fixture for cand in new_evidence[stage].values() for fixture in cand})
        if old_ids != new_ids:
            entries.append({"field": f"fixtures:{stage}", "old": old_ids, "new": new_ids})
    return entries


def oracle_diff_reports(
    old: dict[str, Any],
    new: dict[str, Any],
    *,
    all_fixtures: bool = False,
    old_label: str = "old",
    new_label: str = "new",
) -> dict[str, Any]:
    """Independent oracle for ``diff_reports``, derived from docstrings.

    Same inputs always produce the same output dict (fixed iteration
    order); compares deterministically against the implementation via
    canonical JSON.
    """
    old_decision = old["decision"]
    new_decision = new["decision"]
    old_winner = old.get("selected_candidate")
    new_winner = new.get("selected_candidate")
    old_winner = old_winner if type(old_winner) is str and old_winner else None
    new_winner = new_winner if type(new_winner) is str and new_winner else None

    old_search, old_planned = _or_search_verdicts(old)
    new_search, new_planned = _or_search_verdicts(new)
    old_held_name, old_held = _or_held_verdict(old)
    new_held_name, new_held = _or_held_verdict(new)

    verdict_changes: list[dict[str, str]] = []
    for name in sorted(set(old_search) | set(new_search)):
        old_v = old_search.get(name, "absent")
        new_v = new_search.get(name, "absent")
        if old_v != new_v:
            verdict_changes.append(
                {"candidate": name, "stage": "search", "old": old_v, "new": new_v}
            )
    held_names = sorted({n for n in (old_held_name, new_held_name) if n is not None})
    for name in held_names:
        old_v = old_held if name == old_held_name else "absent"
        new_v = new_held if name == new_held_name else "absent"
        if old_v != new_v:
            verdict_changes.append(
                {"candidate": name, "stage": "held_out", "old": old_v, "new": new_v}
            )
    changed_candidates = sorted({change["candidate"] for change in verdict_changes})

    old_evidence: dict[str, dict[str, dict[str, str]]] = {}
    new_evidence: dict[str, dict[str, dict[str, str]]] = {}
    skipped: dict[str, int] = {"old": 0, "new": 0}
    evidence_present = {"old": False, "new": False}
    for stage in ("search", "held_out"):
        old_map, old_skip, old_ok = _or_stage_evidence(old, stage)
        new_map, new_skip, new_ok = _or_stage_evidence(new, stage)
        old_evidence[stage] = old_map
        new_evidence[stage] = new_map
        skipped["old"] += old_skip
        skipped["new"] += new_skip
        evidence_present["old"] = evidence_present["old"] or old_ok
        evidence_present["new"] = evidence_present["new"] or new_ok

    # CODE-DOC DIVERGENCE: gating is global per side, not per stage.
    both_present = evidence_present["old"] and evidence_present["new"]
    fixture_diffs: list[dict[str, str]] = []
    changed_fixture_diffs: list[dict[str, str]] = []
    if both_present:
        for stage in ("search", "held_out"):
            for name in sorted(set(old_evidence[stage]) | set(new_evidence[stage])):
                old_out = old_evidence[stage].get(name, {})
                new_out = new_evidence[stage].get(name, {})
                for fixture in sorted(set(old_out) | set(new_out)):
                    old_o = old_out.get(fixture, "absent")
                    new_o = new_out.get(fixture, "absent")
                    entry = {
                        "stage": stage,
                        "candidate": name,
                        "fixture": fixture,
                        "old": old_o,
                        "new": new_o,
                    }
                    if old_o != new_o:
                        changed_fixture_diffs.append(entry)
                    if all_fixtures or old_o != new_o:
                        fixture_diffs.append(entry)
    fixture_diffs.sort(key=lambda e: (e["stage"], e["candidate"], e["fixture"]))
    changed_fixture_diffs.sort(key=lambda e: (e["stage"], e["candidate"], e["fixture"]))
    flips = [
        entry
        for entry in changed_fixture_diffs
        if entry["old"] in ("pass", "fail") and entry["new"] in ("pass", "fail")
    ]

    config_drift = _or_drift(old, new)
    for name in sorted(set(old_planned) & set(new_planned)):
        if old_planned[name] != new_planned[name]:
            config_drift.append(
                {
                    "field": f"planned:{name}",
                    "old": old_planned[name],
                    "new": new_planned[name],
                }
            )
    if both_present:
        config_drift.extend(_or_fixture_set_drift(old_evidence, new_evidence))

    decision_changed = old_decision != new_decision
    winner_changed = old_winner != new_winner
    evidence_asymmetric = evidence_present["old"] != evidence_present["new"]

    # Frontier axis: availability computed from the rules, mirroring the
    # implementation. Span-level comparison (decision/winner/leader flips)
    # is covered by the dedicated cycle-125 audit with hand-derived spans
    # and its own mutation gate; the report pairs exercised here never
    # carry budget_frontier payloads, so the comparison below agrees with
    # the implementation on every input this oracle sees.
    def _or_frontier_valid(value: Any) -> bool:
        if type(value) is not dict:
            return False
        candidates = value.get("candidates")
        if type(candidates) is not dict or not candidates:
            return False
        max_tokens = value.get("max_new_tokens")
        if type(max_tokens) is not int or max_tokens <= 0:
            return False
        return any(
            type(data) is dict
            and len(data.get("fixture_ids") or []) > 0
            and any(
                type(point) is list
                and len(point) == 2
                and type(point[0]) is int
                and type(point[1]) is int
                and 1 <= point[0] <= max_tokens
                for point in data.get("pass_steps") or []
            )
            for data in candidates.values()
        )

    _old_ok = _or_frontier_valid(old.get("budget_frontier"))
    _new_ok = _or_frontier_valid(new.get("budget_frontier"))
    if _old_ok and _new_ok:
        _o_available = "both"
    elif _old_ok:
        _o_available = "old"
    elif _new_ok:
        _o_available = "new"
    else:
        _o_available = "neither"
    _o_frontier_asymmetric = _o_available not in ("both", "neither")

    # Held-out floor axis: availability from the rules, mirroring the
    # implementation. The report pairs exercised here never carry
    # held_out_budget_floor payloads, so the comparison below agrees with
    # the implementation on every input this oracle sees.
    def _or_hof_valid(value: Any) -> bool:
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

    _hof_old_ok = _or_hof_valid(old.get("held_out_budget_floor"))
    _hof_new_ok = _or_hof_valid(new.get("held_out_budget_floor"))
    if _hof_old_ok and _hof_new_ok:
        _o_hof_available = "both"
    elif _hof_old_ok:
        _o_hof_available = "old"
    elif _hof_new_ok:
        _o_hof_available = "new"
    else:
        _o_hof_available = "neither"
    _o_hof_asymmetric = _o_hof_available not in ("both", "neither")
    identical = (
        not decision_changed
        and not winner_changed
        and not verdict_changes
        and not changed_fixture_diffs
        and not config_drift
        and not evidence_asymmetric
        and not _o_frontier_asymmetric
        and not _o_hof_asymmetric
    )
    summary = {
        "decision_changed": decision_changed,
        "winner_changed": winner_changed,
        "candidates_changed_status": len(changed_candidates),
        "fixtures_flipped": len(flips),
        "config_drift_warnings": len(config_drift),
        "evidence_asymmetric": evidence_asymmetric,
        "frontier_changed": False,
        "frontier_asymmetric": _o_frontier_asymmetric,
        "held_out_floor_changed": False,
        "held_out_floor_asymmetric": _o_hof_asymmetric,
        "identical": identical,
    }
    return {
        "tool": "diff-reports",
        "old": {"source": old_label},
        "new": {"source": new_label},
        "decision": {"old": old_decision, "new": new_decision, "changed": decision_changed},
        "winner": {"old": old_winner, "new": new_winner, "changed": winner_changed},
        "frontier": {"available": _o_available, "changed": False},
        "held_out_floor": {"available": _o_hof_available, "changed": False},
        "config_drift": config_drift,
        "candidate_verdicts": verdict_changes,
        "evidence": {
            "old": evidence_present["old"],
            "new": evidence_present["new"],
            "skipped_observations": skipped,
        },
        "fixture_diffs": fixture_diffs,
        "fixture_diffs_include_unchanged": all_fixtures,
        "summary": summary,
    }


def canonical(diff: dict[str, Any]) -> str:
    """Canonical serialization used for oracle-vs-implementation comparison."""
    return json.dumps(diff, ensure_ascii=True, sort_keys=True)


def compare_pair(
    implementation: Any, old: dict[str, Any], new: dict[str, Any], **kwargs: Any
) -> None:
    """Assert one report pair produces identical oracle/implementation output.

    ``implementation`` is the diffreports-like module under test (the real
    one, or a mutated copy for sensitivity checks). Inputs are deep-copied
    so neither side can mutate the caller's dicts.
    """
    expected = oracle_diff_reports(copy.deepcopy(old), copy.deepcopy(new), **kwargs)
    actual = implementation.diff_reports(copy.deepcopy(old), copy.deepcopy(new), **kwargs)
    assert canonical(expected) == canonical(actual), (
        "oracle/implementation divergence:\n"
        f"expected: {canonical(expected)[:2000]}\n"
        f"actual:   {canonical(actual)[:2000]}"
    )


# ---------------------------------------------------------------------------
# Seeded synthetic report builders (deterministic; no network, no models).
# ---------------------------------------------------------------------------

_FAMILIES = ["llama3", "gemma", "qwen", "mistral", "phi"]
_DECISIONS = ["select", "abstain", "inconclusive"]
_BOUNDARY_IDS = [1, 7, 42, 128, 32000, 128000]


def _rand_name(rng: random.Random) -> str:
    family = rng.choice(_FAMILIES)
    mutation = rng.choice(["", "/a", "/b", "/long-mutation-name"])
    return family + mutation


def _rand_summary(rng: random.Random) -> Any:
    """Randomized summaries: mostly well-formed, sometimes malformed."""
    style = rng.random()
    planned = rng.randint(1, 8)
    if style < 0.62:
        passes = rng.randint(0, planned)
        return {
            "planned": planned,
            "exact_and_boundary_passes": passes,
            "aggregate_eligible": rng.random() < 0.9,
        }
    if style < 0.70:
        # pass-everything summary
        return {
            "planned": planned,
            "exact_and_boundary_passes": planned,
            "aggregate_eligible": True,
        }
    if style < 0.78:
        # float counts (malformed: type() is int fails) -> "unknown"
        return {
            "planned": float(planned),
            "exact_and_boundary_passes": float(planned),
            "aggregate_eligible": True,
        }
    if style < 0.84:
        # bool counts (type(True) is not int) -> "unknown"
        return {
            "planned": True,
            "exact_and_boundary_passes": True,
            "aggregate_eligible": True,
        }
    if style < 0.90:
        # missing keys / zero planned / ineligible
        return {
            "planned": rng.choice([0, -1, planned]),
            "exact_and_boundary_passes": rng.randint(0, planned),
            "aggregate_eligible": rng.choice([False, "yes", None]),
        }
    # malformed container
    return rng.choice([None, "summary", [planned], {"planned": planned}])


def _rand_observation(rng: random.Random, fixture: str) -> dict[str, Any]:
    style = rng.random()
    if style < 0.80:
        return {
            "fixture": fixture,
            "exact_match": rng.random() < 0.6,
            "boundary_found": rng.random() < 0.7,
        }
    # malformed rows: skipped and counted
    malformed = rng.choice(
        [
            {"fixture": fixture},  # missing booleans
            {"fixture": fixture, "exact_match": 1, "boundary_found": 0},
            {"fixture": fixture, "exact_match": "yes", "boundary_found": True},
            {"fixture": "", "exact_match": True, "boundary_found": True},
            {"fixture": None, "exact_match": True, "boundary_found": True},
            {"exact_match": True, "boundary_found": True},  # no fixture key
            "not-a-dict",
            None,
        ]
    )
    return malformed  # type: ignore[return-value]


def _rand_observations(rng: random.Random, n_fixtures: int) -> Any:
    """Randomized observations: sometimes absent/None/malformed container."""
    style = rng.random()
    if style < 0.70:
        fixtures = [f"f{i}" for i in rng.sample(range(20), min(n_fixtures, 20))]
        obs = [_rand_observation(rng, fx) for fx in fixtures]
        rng.shuffle(obs)
        return obs
    if style < 0.78:
        return []  # present but empty -> no usable evidence
    if style < 0.84:
        return None
    if style < 0.90:
        return {"not": "a list"}
    # observations key omitted by the caller
    return ...  # sentinel meaning "omit the key"


def _rand_boundary_ids(rng: random.Random) -> Any:
    style = rng.random()
    if style < 0.60:
        return rng.sample(_BOUNDARY_IDS, rng.randint(1, 4))
    if style < 0.72:
        return []  # empty list is still all-int
    if style < 0.82:
        ids = rng.sample(_BOUNDARY_IDS, 2)
        ids.append(rng.choice([1.5, "7", None, True]))  # non-int token poisons the list
        return ids
    if style < 0.90:
        return None
    return ...  # omit


def _rand_entry(rng: random.Random, name: str, n_fixtures: int) -> dict[str, Any]:
    entry: dict[str, Any] = {"format": {"name": name}}
    if rng.random() < 0.85:
        entry["summary"] = _rand_summary(rng)
    observations = _rand_observations(rng, n_fixtures)
    if observations is not ...:
        entry["observations"] = observations
    bids = _rand_boundary_ids(rng)
    if bids is not ...:
        entry["boundary_token_ids"] = bids
    return entry


def make_report(
    rng: random.Random,
    *,
    n_candidates: int | None = None,
    n_fixtures: int | None = None,
) -> dict[str, Any]:
    """Build a shape-valid synthetic results.json dict."""
    n_candidates = n_candidates if n_candidates is not None else rng.randint(1, 4)
    n_fixtures = n_fixtures if n_fixtures is not None else rng.randint(1, 8)
    names = [_rand_name(rng) for _ in range(n_candidates)]
    report: dict[str, Any] = {
        "decision": rng.choice(_DECISIONS),
        "search": [_rand_entry(rng, name, n_fixtures) for name in names],
    }
    # selected_candidate: str winner, None, missing, or empty/garbage
    sel_style = rng.random()
    if sel_style < 0.55:
        report["selected_candidate"] = rng.choice(names)
    elif sel_style < 0.75:
        report["selected_candidate"] = None
    elif sel_style < 0.85:
        pass  # missing key
    elif sel_style < 0.93:
        report["selected_candidate"] = ""
    else:
        report["selected_candidate"] = 42
    # held_out: None, missing, or dict with summary/observations
    held_style = rng.random()
    if held_style < 0.50:
        held: dict[str, Any] = {}
        if rng.random() < 0.8:
            held["summary"] = _rand_summary(rng)
        observations = _rand_observations(rng, n_fixtures)
        if observations is not ...:
            held["observations"] = observations
        report["held_out"] = held
    elif held_style < 0.65:
        report["held_out"] = None
    # else: missing key
    # config values: int/float/None/missing/bool/str variants
    tok_style = rng.random()
    if tok_style < 0.55:
        report["max_new_tokens"] = rng.choice([64, 256, 512, 1024])
    elif tok_style < 0.65:
        report["max_new_tokens"] = float(rng.choice([64, 256]))
    elif tok_style < 0.72:
        report["max_new_tokens"] = None
    elif tok_style < 0.78:
        report["max_new_tokens"] = rng.choice([True, "512"])
    fp_style = rng.random()
    if fp_style < 0.55:
        report["backend_fingerprint"] = f"fp-{rng.randint(1, 5)}"
    elif fp_style < 0.65:
        report["backend_fingerprint"] = None
    elif fp_style < 0.72:
        report["backend_fingerprint"] = rng.choice([123, ["fp"], b"fp"])
    return report


# ---------------------------------------------------------------------------
# Scenario pair generators: each takes an RNG and returns (old, new).
# ---------------------------------------------------------------------------


def _flip_one_observation(observations: list[Any]) -> bool:
    """Flip exact_match/boundary_found of one well-formed observation."""
    wellformed = [
        i
        for i, ob in enumerate(observations)
        if type(ob) is dict
        and type(ob.get("fixture")) is str
        and ob.get("fixture")
        and type(ob.get("exact_match")) is bool
        and type(ob.get("boundary_found")) is bool
    ]
    if not wellformed:
        return False
    ob = observations[wellformed[0]]
    ob["exact_match"] = not ob["exact_match"]
    return True


def scenario_identical(rng: random.Random) -> tuple[dict, dict]:
    base = make_report(rng)
    return base, copy.deepcopy(base)


def scenario_flip_fixture(rng: random.Random) -> tuple[dict, dict]:
    base = make_report(rng)
    new = copy.deepcopy(base)
    # find a candidate entry with flippable evidence; retry budget
    for _ in range(50):
        entries = [e for e in new["search"] if type(e.get("observations")) is list]
        if not entries:
            new = copy.deepcopy(base)
            base = make_report(rng)
            new = copy.deepcopy(base)
            continue
        if _flip_one_observation(rng.choice(entries)["observations"]):
            return base, new
    return base, copy.deepcopy(base)  # degenerate: no usable evidence found


def scenario_add_remove_fixture(rng: random.Random) -> tuple[dict, dict]:
    base = make_report(rng)
    new = copy.deepcopy(base)
    for entry in new["search"]:
        observations = entry.get("observations")
        if type(observations) is list and observations:
            if rng.random() < 0.5:
                observations.append(
                    {
                        "fixture": f"f-extra-{rng.randint(100, 999)}",
                        "exact_match": True,
                        "boundary_found": True,
                    }
                )
            else:
                observations.pop(rng.randrange(len(observations)))
    return base, new


def scenario_config_changed(rng: random.Random) -> tuple[dict, dict]:
    base = make_report(rng)
    new = copy.deepcopy(base)
    which = rng.choice(
        ["max_new_tokens", "fingerprint", "boundaries", "add_candidate", "drop_candidate"]
    )
    if which == "max_new_tokens":
        current = base.get("max_new_tokens")
        new["max_new_tokens"] = current + 128 if type(current) is int else 999
    elif which == "fingerprint":
        new["backend_fingerprint"] = f"fp-changed-{rng.randint(100, 999)}"
    elif which == "boundaries":
        for entry in new["search"]:
            if type(entry.get("boundary_token_ids")) is list:
                entry["boundary_token_ids"] = entry["boundary_token_ids"] + [777]
                break
        else:
            new["search"].append(
                {
                    "format": {"name": "zeta"},
                    "boundary_token_ids": [9, 9, 9],
                }
            )
    elif which == "add_candidate":
        new["search"].append(_rand_entry(rng, "newfam/extra", 3))
    else:
        if len(new["search"]) > 1:
            new["search"].pop(rng.randrange(len(new["search"])))
    return base, new


def scenario_nested_field_changed(rng: random.Random) -> tuple[dict, dict]:
    base = make_report(rng)
    new = copy.deepcopy(base)
    entry = rng.choice(new["search"])
    summary = entry.get("summary")
    if type(summary) is dict and type(summary.get("planned")) is int:
        summary["planned"] = summary["planned"] + rng.choice([1, 2, 5])
    elif type(summary) is dict:
        summary["planned"] = rng.randint(1, 6)
        summary["exact_and_boundary_passes"] = 0
        summary["aggregate_eligible"] = True
    else:
        entry["summary"] = {
            "planned": 4,
            "exact_and_boundary_passes": 4,
            "aggregate_eligible": True,
        }
    return base, new


def scenario_evidence_stripped(rng: random.Random) -> tuple[dict, dict]:
    base = make_report(rng)
    # ensure at least one side has evidence before stripping the other
    new = copy.deepcopy(base)
    for entry in new["search"]:
        entry.pop("observations", None)
    if type(new.get("held_out")) is dict:
        new["held_out"].pop("observations", None)
    return base, new


def scenario_decision_winner(rng: random.Random) -> tuple[dict, dict]:
    base = make_report(rng)
    new = copy.deepcopy(base)
    decisions = [d for d in _DECISIONS if d != base["decision"]]
    new["decision"] = rng.choice(decisions)
    names = [
        e["format"]["name"]
        for e in new["search"]
        if type(e.get("format")) is dict and type(e["format"].get("name")) is str
    ]
    new["selected_candidate"] = rng.choice(names) if names else None
    return base, new


_SCENARIOS = {
    "identical": scenario_identical,
    "flip_fixture": scenario_flip_fixture,
    "add_remove_fixture": scenario_add_remove_fixture,
    "config_changed": scenario_config_changed,
    "nested_field_changed": scenario_nested_field_changed,
    "evidence_stripped": scenario_evidence_stripped,
    "decision_winner": scenario_decision_winner,
}

# Fixed seed budget: deterministic across runs and machines.
_FUZZ_SEEDS = list(range(60))

# ---------------------------------------------------------------------------
# Test cases (unittest-style, deterministic seeds only).
# ---------------------------------------------------------------------------


class FuzzOracleAgreement(unittest.TestCase):
    """Seeded differential fuzz: oracle vs implementation on every pair."""

    def test_scenarios_oracle_agreement(self) -> None:
        for scenario_name, generator in _SCENARIOS.items():
            for seed in _FUZZ_SEEDS:
                with self.subTest(scenario=scenario_name, seed=seed):
                    rng = random.Random(str(scenario_name) + ":" + str(seed))  # noqa: S311 -- deterministic test seed
                    old, new = generator(rng)
                    compare_pair(impl, old, new)

    def test_scenarios_oracle_agreement_all_fixtures_flag(self) -> None:
        for scenario_name, generator in _SCENARIOS.items():
            for seed in _FUZZ_SEEDS:
                with self.subTest(scenario=scenario_name, seed=seed, all_fixtures=True):
                    rng = random.Random("allflag" + ":" + str(scenario_name) + ":" + str(seed))  # noqa: S311 -- deterministic test seed
                    old, new = generator(rng)
                    compare_pair(impl, old, new, all_fixtures=True)

    def test_identical_pairs_are_identical(self) -> None:
        """Identical report pairs must be drift-free with identical=True."""
        for seed in _FUZZ_SEEDS:
            with self.subTest(seed=seed):
                rng = random.Random("identical-check" + ":" + str(seed))  # noqa: S311 -- deterministic test seed
                old, new = scenario_identical(rng)
                diff = impl.diff_reports(old, new)
                self.assertEqual(
                    canonical(diff),
                    canonical(oracle_diff_reports(old, new)),
                )
                self.assertTrue(diff["summary"]["identical"])
                self.assertEqual(diff["fixture_diffs"], [])
                self.assertEqual(diff["config_drift"], [])
                self.assertEqual(diff["candidate_verdicts"], [])
                self.assertFalse(diff["summary"]["decision_changed"])
                self.assertFalse(diff["summary"]["winner_changed"])
                self.assertFalse(diff["summary"]["evidence_asymmetric"])

    def test_key_reorder_is_identical(self) -> None:
        """Key insertion order (and a JSON round-trip) must not affect output."""

        def reorder(value: Any) -> Any:
            if type(value) is dict:
                items = list(value.items())
                return {k: reorder(v) for k, v in reversed(items)}
            if type(value) is list:
                return [reorder(v) for v in value]
            return value

        for seed in _FUZZ_SEEDS:
            with self.subTest(seed=seed):
                rng = random.Random("reorder" + ":" + str(seed))  # noqa: S311 -- deterministic test seed
                old, _ = scenario_identical(rng)
                new = reorder(old)
                reloaded = json.loads(json.dumps(new))  # round-trip
                self.assertEqual(
                    canonical(impl.diff_reports(old, reloaded)),
                    canonical(impl.diff_reports(old, old)),
                )
                self.assertEqual(
                    canonical(oracle_diff_reports(old, reloaded)),
                    canonical(impl.diff_reports(old, old)),
                )

    def test_determinism_byte_identical(self) -> None:
        """Two runs over deep copies must serialize byte-identically."""
        for scenario_name, generator in _SCENARIOS.items():
            for seed in range(10):
                with self.subTest(scenario=scenario_name, seed=seed):
                    rng = random.Random("determinism" + ":" + str(scenario_name) + ":" + str(seed))  # noqa: S311 -- deterministic test seed
                    old, new = generator(rng)
                    first = canonical(impl.diff_reports(copy.deepcopy(old), copy.deepcopy(new)))
                    second = canonical(impl.diff_reports(copy.deepcopy(old), copy.deepcopy(new)))
                    self.assertEqual(first, second)


class NumericAndMissingEdgeCases(unittest.TestCase):
    """float-vs-int, None-vs-missing, and other normalization edges."""

    def test_float_int_normalization(self) -> None:
        """float/bool max_new_tokens normalize to None (like missing/None).

        int-vs-float is then a REAL drift (512 vs None), while
        missing-vs-float is not.
        """
        rng = random.Random("floatint:1")  # noqa: S311 -- deterministic test seed
        old, _ = scenario_identical(rng)
        old.pop("max_new_tokens", None)
        new = copy.deepcopy(old)
        new["max_new_tokens"] = 512.0
        compare_pair(impl, old, new)
        diff = impl.diff_reports(old, new)
        self.assertFalse([e for e in diff["config_drift"] if e["field"] == "max_new_tokens"])
        for other in (512.0, True, "512"):
            with self.subTest(other=other):
                old2 = copy.deepcopy(old)
                new2 = copy.deepcopy(old)
                old2["max_new_tokens"] = 512
                new2["max_new_tokens"] = other
                compare_pair(impl, old2, new2)
                drifted = [
                    e
                    for e in impl.diff_reports(old2, new2)["config_drift"]
                    if e["field"] == "max_new_tokens"
                ]
                self.assertEqual(len(drifted), 1)
                self.assertEqual((drifted[0]["old"], drifted[0]["new"]), (512, None))

    def test_float_planned_is_unknown_not_int(self) -> None:
        """planned: 3 vs 3.0 -> 'pass' vs 'unknown' verdict flip + planned drift."""
        entry = {
            "format": {"name": "gemma"},
            "summary": {
                "planned": 3,
                "exact_and_boundary_passes": 3,
                "aggregate_eligible": True,
            },
            "observations": [{"fixture": "f1", "exact_match": True, "boundary_found": True}],
        }
        old = {
            "decision": "select",
            "search": [entry],
            "selected_candidate": None,
            "held_out": None,
        }
        new = copy.deepcopy(old)
        new["search"][0]["summary"]["planned"] = 3.0
        new["search"][0]["summary"]["exact_and_boundary_passes"] = 3.0
        compare_pair(impl, old, new)
        diff = impl.diff_reports(old, new)
        verdicts = diff["candidate_verdicts"]
        self.assertEqual(len(verdicts), 1)
        self.assertEqual((verdicts[0]["old"], verdicts[0]["new"]), ("pass", "unknown"))
        planned_drift = [e for e in diff["config_drift"] if e["field"] == "planned:gemma"]
        self.assertEqual(len(planned_drift), 1)
        self.assertEqual((planned_drift[0]["old"], planned_drift[0]["new"]), (3, None))

    def test_none_vs_missing_keys_identical(self) -> None:
        """None and absent keys normalize identically everywhere."""
        rng = random.Random("nonemissing:2")  # noqa: S311 -- deterministic test seed
        old, _ = scenario_identical(rng)
        new = copy.deepcopy(old)
        new["selected_candidate"] = None
        new["held_out"] = None
        new["max_new_tokens"] = None
        new["backend_fingerprint"] = None
        old.pop("selected_candidate", None)
        old.pop("held_out", None)
        old.pop("max_new_tokens", None)
        old.pop("backend_fingerprint", None)
        compare_pair(impl, old, new)
        diff = impl.diff_reports(old, new)
        self.assertTrue(diff["summary"]["identical"])

    def test_boundary_token_float_poisons_list(self) -> None:
        """A single non-int token id drops the candidate from boundaries."""
        old = {
            "decision": "select",
            "search": [
                {
                    "format": {"name": "qwen"},
                    "summary": {
                        "planned": 1,
                        "exact_and_boundary_passes": 1,
                        "aggregate_eligible": True,
                    },
                    "boundary_token_ids": [1, 7],
                }
            ],
            "selected_candidate": None,
            "held_out": None,
        }
        new = copy.deepcopy(old)
        new["search"][0]["boundary_token_ids"] = [1, 7.0]
        compare_pair(impl, old, new)
        diff = impl.diff_reports(old, new)
        drift = [e for e in diff["config_drift"] if e["field"] == "boundaries"]
        self.assertEqual(len(drift), 1)
        self.assertEqual(drift[0]["old"], {"qwen": [1, 7]})
        self.assertEqual(drift[0]["new"], {})


class FixtureOutcomeSemantics(unittest.TestCase):
    """Per-fixture outcome rules: flips vs absent, last-wins, skip counting."""

    def test_flips_count_only_pass_fail_transitions(self) -> None:
        """pass<->absent is a change but not a 'flip'; only pass<->fail flips."""
        old = {
            "decision": "select",
            "search": [
                {
                    "format": {"name": "m"},
                    "summary": {
                        "planned": 3,
                        "exact_and_boundary_passes": 3,
                        "aggregate_eligible": True,
                    },
                    "observations": [
                        {"fixture": "a", "exact_match": True, "boundary_found": True},
                        {"fixture": "b", "exact_match": True, "boundary_found": True},
                        {"fixture": "c", "exact_match": False, "boundary_found": True},
                    ],
                }
            ],
            "selected_candidate": None,
            "held_out": None,
        }
        new = copy.deepcopy(old)
        # a: pass -> fail (flip); b: pass -> removed (absent, not a flip);
        # d: absent -> pass (not a flip)
        new["search"][0]["observations"][0]["exact_match"] = False
        new["search"][0]["observations"].pop(1)
        new["search"][0]["observations"].append(
            {"fixture": "d", "exact_match": True, "boundary_found": True}
        )
        compare_pair(impl, old, new)
        diff = impl.diff_reports(old, new)
        self.assertEqual(diff["summary"]["fixtures_flipped"], 1)
        changed = {d["fixture"]: (d["old"], d["new"]) for d in diff["fixture_diffs"]}
        self.assertEqual(
            changed,
            {
                "a": ("pass", "fail"),
                "b": ("pass", "absent"),
                "d": ("absent", "pass"),
            },
        )

    def test_duplicate_fixture_id_last_wins(self) -> None:
        """Duplicate fixture ids inside one candidate: last row wins."""
        entry = {
            "format": {"name": "m"},
            "summary": {
                "planned": 1,
                "exact_and_boundary_passes": 1,
                "aggregate_eligible": True,
            },
            "observations": [
                {"fixture": "x", "exact_match": True, "boundary_found": True},
                {"fixture": "x", "exact_match": False, "boundary_found": True},
            ],
        }
        old = {"decision": "s", "search": [entry], "selected_candidate": None, "held_out": None}
        new = copy.deepcopy(old)
        compare_pair(impl, old, new)
        diff = impl.diff_reports(old, new)
        # last row is a fail on both sides -> no fixture diff
        self.assertEqual(diff["fixture_diffs"], [])
        self.assertTrue(diff["summary"]["identical"])

    def test_duplicate_candidate_name_evidence_last_wins(self) -> None:
        """Duplicate candidate names: verdicts first-wins, evidence LAST-wins.

        (Historical note: the ``_fixture_outcomes`` docstring used to claim
        first-wins for both; cycle 28 corrected it to match this code
        behavior, pinned by tests/test_cycle28_docstring_contract.py.)
        """
        entry1 = {
            "format": {"name": "dup"},
            "summary": {
                "planned": 1,
                "exact_and_boundary_passes": 1,
                "aggregate_eligible": True,
            },
            "observations": [{"fixture": "f1", "exact_match": True, "boundary_found": True}],
        }
        entry2 = {
            "format": {"name": "dup"},
            "summary": {
                "planned": 1,
                "exact_and_boundary_passes": 0,
                "aggregate_eligible": True,
            },
            "observations": [
                {"fixture": "f1", "exact_match": False, "boundary_found": True},
                {"fixture": "f2", "exact_match": True, "boundary_found": True},
            ],
        }
        old = {
            "decision": "s",
            "search": [entry1, entry2],
            "selected_candidate": None,
            "held_out": None,
        }
        new = copy.deepcopy(old)
        compare_pair(impl, old, new)
        diff = impl.diff_reports(old, new)
        # verdict is first-wins (pass); evidence is last-wins (f1 fail, f2 pass)
        self.assertEqual(diff["candidate_verdicts"], [])
        ev_old, ev_new = diff["evidence"]["old"], diff["evidence"]["new"]
        self.assertTrue(ev_old and ev_new)
        fixtures = {d["fixture"]: d["old"] for d in diff["fixture_diffs"]}
        # fixture_diffs is empty here (identical pair); check evidence directly
        self.assertEqual(fixtures, {})
        staged = impl._stage_evidence(old, "search")[0]
        self.assertEqual(staged, {"dup": {"f1": "fail", "f2": "pass"}})

    def test_malformed_observations_skipped_and_counted(self) -> None:
        """Malformed rows are skipped (not silent) and counted per side."""
        old = {
            "decision": "s",
            "search": [
                {
                    "format": {"name": "m"},
                    "summary": {
                        "planned": 2,
                        "exact_and_boundary_passes": 2,
                        "aggregate_eligible": True,
                    },
                    "observations": [
                        {"fixture": "ok", "exact_match": True, "boundary_found": True},
                        {"fixture": "bad"},  # malformed
                        {"fixture": "", "exact_match": True, "boundary_found": True},
                        "junk",
                    ],
                }
            ],
            "selected_candidate": None,
            "held_out": None,
        }
        new = copy.deepcopy(old)
        new["search"][0]["observations"] = [
            {"fixture": "ok", "exact_match": True, "boundary_found": True}
        ]
        compare_pair(impl, old, new)
        diff = impl.diff_reports(old, new)
        self.assertEqual(diff["evidence"]["skipped_observations"], {"old": 3, "new": 0})
        # identical usable outcomes -> no fixture diffs; identical must hold
        # (evidence present on both sides, nothing drifted)
        self.assertEqual(diff["fixture_diffs"], [])
        self.assertTrue(diff["summary"]["identical"])

    def test_evidence_asymmetry_blocks_identical(self) -> None:
        """One-sided evidence means a whole axis was uncompared: not identical."""
        rng = random.Random("asym:3")  # noqa: S311 -- deterministic test seed
        old, _ = scenario_identical(rng)
        new = copy.deepcopy(old)
        for entry in new["search"]:
            entry.pop("observations", None)
        if type(new.get("held_out")) is dict:
            new["held_out"].pop("observations", None)
        compare_pair(impl, old, new)
        diff = impl.diff_reports(old, new)
        self.assertTrue(diff["summary"]["evidence_asymmetric"])
        self.assertFalse(diff["summary"]["identical"])
        self.assertEqual(diff["fixture_diffs"], [])


class SummaryAndDriftShape(unittest.TestCase):
    """Hand-built pairs asserting exact summary counts and drift ordering."""

    def test_full_drift_shape_hand_built(self) -> None:
        old = {
            "decision": "abstain",
            "search": [
                {
                    "format": {"name": "llama3/a"},
                    "summary": {
                        "planned": 2,
                        "exact_and_boundary_passes": 2,
                        "aggregate_eligible": True,
                    },
                    "observations": [
                        {"fixture": "f1", "exact_match": True, "boundary_found": True},
                        {"fixture": "f2", "exact_match": True, "boundary_found": False},
                    ],
                    "boundary_token_ids": [1, 2],
                },
                {
                    "format": {"name": "gemma"},
                    "summary": {
                        "planned": 2,
                        "exact_and_boundary_passes": 1,
                        "aggregate_eligible": True,
                    },
                    "observations": [
                        {"fixture": "f1", "exact_match": False, "boundary_found": True}
                    ],
                },
            ],
            "selected_candidate": "llama3/a",
            "held_out": {
                "summary": {
                    "planned": 1,
                    "exact_and_boundary_passes": 1,
                    "aggregate_eligible": True,
                },
                "observations": [{"fixture": "h1", "exact_match": True, "boundary_found": True}],
            },
            "max_new_tokens": 256,
            "backend_fingerprint": "fp-1",
        }
        new = {
            "decision": "select",
            "search": [
                {
                    "format": {"name": "llama3/a"},
                    "summary": {
                        "planned": 3,  # planned drift
                        "exact_and_boundary_passes": 2,
                        "aggregate_eligible": True,
                    },
                    "observations": [
                        {"fixture": "f1", "exact_match": False, "boundary_found": True},
                        {"fixture": "f2", "exact_match": True, "boundary_found": False},
                    ],
                    "boundary_token_ids": [1, 3],  # boundary drift
                },
                {
                    "format": {"name": "qwen"},  # candidate swap: gemma -> qwen
                    "summary": {
                        "planned": 1,
                        "exact_and_boundary_passes": 1,
                        "aggregate_eligible": True,
                    },
                    "observations": [
                        {"fixture": "f9", "exact_match": True, "boundary_found": True}
                    ],
                },
            ],
            "selected_candidate": "qwen",
            "held_out": {
                "summary": {
                    "planned": 1,
                    "exact_and_boundary_passes": 0,
                    "aggregate_eligible": True,
                },
                "observations": [{"fixture": "h1", "exact_match": False, "boundary_found": True}],
            },
            "max_new_tokens": 512,
            "backend_fingerprint": "fp-1",
        }
        compare_pair(impl, old, new)
        diff = impl.diff_reports(old, new)
        oracle = oracle_diff_reports(old, new)
        self.assertEqual(canonical(diff), canonical(oracle))

        summary = diff["summary"]
        self.assertTrue(summary["decision_changed"])
        self.assertTrue(summary["winner_changed"])
        # verdict changes: llama3/a pass->fail (search), gemma fail->absent,
        # qwen absent->pass, held llama3/a pass->absent, qwen absent->fail
        self.assertEqual(summary["candidates_changed_status"], 3)
        # flips: only search/llama3/a f1 pass->fail; the held h1 transitions
        # are pass->absent and absent->fail (changes, not flips)
        self.assertEqual(summary["fixtures_flipped"], 1)
        self.assertFalse(summary["evidence_asymmetric"])
        self.assertFalse(summary["identical"])
        self.assertEqual(summary["config_drift_warnings"], len(diff["config_drift"]))

        fields = [e["field"] for e in diff["config_drift"]]
        # fixed field order, then planned drift (sorted), then fixture-set drift
        self.assertEqual(
            fields,
            [
                "families",
                "candidates",
                "max_new_tokens",
                "boundaries",
                "planned:llama3/a",
                "fixtures:search",
            ],
        )
        planned = next(e for e in diff["config_drift"] if e["field"] == "planned:llama3/a")
        self.assertEqual((planned["old"], planned["new"]), (2, 3))
        # no fixtures:held_out entry: held fixture ids identical ({h1})
        self.assertNotIn("fixtures:held_out", fields)

        verdicts = {
            (v["candidate"], v["stage"]): (v["old"], v["new"]) for v in diff["candidate_verdicts"]
        }
        self.assertEqual(
            verdicts,
            {
                ("gemma", "search"): ("fail", "absent"),
                ("llama3/a", "search"): ("pass", "fail"),
                ("qwen", "search"): ("absent", "pass"),
                ("llama3/a", "held_out"): ("pass", "absent"),
                ("qwen", "held_out"): ("absent", "fail"),
            },
        )

    def test_all_fixtures_flag_lists_unchanged(self) -> None:
        rng = random.Random("allflag-shape:4")  # noqa: S311 -- deterministic test seed
        old, new = scenario_flip_fixture(rng)
        changed_only = impl.diff_reports(old, new)
        all_cells = impl.diff_reports(old, new, all_fixtures=True)
        self.assertEqual(
            canonical(all_cells), canonical(oracle_diff_reports(old, new, all_fixtures=True))
        )
        self.assertTrue(all_cells["fixture_diffs_include_unchanged"])
        self.assertFalse(changed_only["fixture_diffs_include_unchanged"])
        self.assertGreaterEqual(len(all_cells["fixture_diffs"]), len(changed_only["fixture_diffs"]))
        # the changed-only list is exactly the changed subset of the full list
        changed_in_all = [d for d in all_cells["fixture_diffs"] if d["old"] != d["new"]]
        self.assertEqual(changed_in_all, changed_only["fixture_diffs"])

    def test_load_results_roundtrip_agrees(self) -> None:
        """Full path: JSON on disk -> load_results -> diff_reports."""
        import tempfile

        rng = random.Random("roundtrip:5")  # noqa: S311 -- deterministic test seed
        old, new = scenario_config_changed(rng)
        with tempfile.TemporaryDirectory() as tmp:
            old_p = Path(tmp) / "old.json"
            new_p = Path(tmp) / "new.json"
            old_p.write_text(json.dumps(old), encoding="utf-8")
            new_p.write_text(json.dumps(new), encoding="utf-8")
            loaded_old = impl.load_results(old_p)
            loaded_new = impl.load_results(new_p)
        compare_pair(impl, loaded_old, loaded_new)


if __name__ == "__main__":
    unittest.main()
