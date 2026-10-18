"""Deterministic comparison of two ``mpl run`` result directories.

``diff-reports`` is the operator's iterate-and-compare tool: tweak a
template or fixture, re-run, and see exactly what changed instead of
eyeballing two HTML reports. The comparison is exact (no fuzzy matching),
offline, and deterministic: the same two inputs always produce
byte-identical output.

Inputs are ``results.json`` files (or the report directories containing
them) as written by :func:`model_protocol_lab.reporting.save_report`.
Both inputs go through a bounded JSON reader (fixed error codes, no
tracebacks): oversized, malformed, or hostile inputs cannot hang the tool
or leak file content into error text.

Per-fixture outcome comparison needs per-fixture evidence
(``observations``), which ``results.json`` only carries when the run used
``--include-evidence``. When either side lacks it, the tool says so
plainly instead of guessing.
"""

import argparse
import json
import os
import re
import stat
import sys
from bisect import bisect_right
from collections.abc import Callable
from pathlib import Path
from typing import Any

# results.json with --include-evidence carries per-fixture token-ID arrays;
# 32 MiB comfortably covers the largest runs this lab produces (100 search +
# 100 held-out fixtures at the 4096-token cap) while keeping a hostile file
# from exhausting memory.
MAX_RESULTS_BYTES = 32 * 1024 * 1024

_CONTROL_CHARS = re.compile(
    r"[\x00-\x1f\x7f-\x9f\ud800-\udfff\u200e\u200f\u2028\u2029\u202a-\u202e\u2066-\u2069]"
)
# Lone surrogates are stripped too: they crash strict stream writes with
# UnicodeEncodeError and are never valid text on their own. The \u200e-\u2069
# ranges strip Unicode bidi controls: a hostile results.json entry like
# "safe-\u202e-evil" renders right-to-left (Trojan Source spoofing,
# CVE-2021-42574), and U+2028/U+2029 break line discipline. Other invisible
# format characters (BOM, zero-width space, soft hyphen, ...) are deliberately
# preserved: they are diagnostically valuable in error messages and cannot
# reorder displayed text. Mirrors cli._CONTROL_CHARS.


def _display_id(value: Any) -> str:
    """Operator-visible names/IDs with control characters stripped.

    Fixture IDs and candidate names arrive from the compared files; a
    hostile results.json must not inject terminal escape sequences into
    stderr output. (Mirrors ``cli._display_id``; kept local so this module
    does not import the CLI.) None renders as "(none)". Non-string
    values (a hand-edited or version-skewed file may carry a dict winner
    or a list tier) render via str() rather than crashing the summary --
    the module's degrade-gracefully contract covers malformed values as
    well as malformed structure (wave-2 finding C3).
    """
    if value is None:
        return "(none)"
    if type(value) is not str:
        value = str(value)
    return _CONTROL_CHARS.sub("", value)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("results_duplicate_field")
        result[key] = value
    return result


def _reject_constant(_value: Any) -> Any:
    raise ValueError("results_nonfinite_number")


def load_results(path: str | Path) -> dict[str, Any]:
    """Read a ``results.json`` (or a report directory holding one).

    The read is bounded and side-channel free: the path must resolve to a
    regular file (a directory resolves to the ``results.json`` inside it;
    FIFOs, devices and sockets are refused), at
    most ``MAX_RESULTS_BYTES`` are read, and every failure mode maps to a
    fixed error code raised as ``ValueError`` -- never a traceback and never
    file content echoed into the message.
    """
    target = Path(path)
    if not target.exists():
        raise ValueError("results_path_missing")
    resolved = target / "results.json" if target.is_dir() else target
    # O_NONBLOCK so that a named FIFO (checked below anyway) with no writer
    # cannot block; the fstat then refuses anything that is not a regular
    # file. Checking the opened fd (not the path) closes the stat->open
    # TOCTOU window. Mirrors datasets.load_dataset.
    try:
        fd = os.open(resolved, os.O_RDONLY | os.O_NONBLOCK)
    except OSError:
        raise ValueError("results_unreadable") from None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("results_not_regular_file")
        with os.fdopen(fd, "rb") as handle:
            fd = -1  # ownership moved to the file object
            content = handle.read(MAX_RESULTS_BYTES + 1)
    finally:
        if fd != -1:
            os.close(fd)
    if len(content) > MAX_RESULTS_BYTES:
        raise ValueError("results_too_large")
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("results_not_utf8") from None
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except RecursionError:
        # Deeply nested input within the byte cap exhausts the parser stack.
        raise ValueError("results_nesting_too_deep") from None
    except json.JSONDecodeError:
        raise ValueError("results_invalid_json") from None
    if type(value) is not dict:
        raise ValueError("results_not_an_object")
    _require_shape(value)
    return value


def _require_shape(value: dict[str, Any]) -> None:
    """The minimal ``results.json`` contract this tool compares.

    Only the compared axes are enforced strictly (decision, per-candidate
    search entries, held-out); everything else degrades to "unknown" or is
    skipped, so a hand-edited or version-skewed file still diffs instead
    of failing.
    """
    if type(value.get("decision")) is not str:
        raise ValueError("results_wrong_shape")
    search = value.get("search")
    if type(search) is not list:
        raise ValueError("results_wrong_shape")
    selected = value.get("selected_candidate")
    if selected is not None and type(selected) is not str:
        raise ValueError("results_wrong_shape")
    held = value.get("held_out")
    if held is not None and type(held) is not dict:
        raise ValueError("results_wrong_shape")


def _candidate_name(entry: Any) -> str | None:
    if type(entry) is not dict:
        return None
    fmt = entry.get("format")
    name = fmt.get("name") if type(fmt) is dict else None
    return name if type(name) is str and name else None


def _base_family(name: str) -> str:
    # Mutated candidates are "family/mutation"; the family is the base name.
    # results.json does not store the --family flag, so this is the exact
    # family signal available.
    return name.split("/", 1)[0]


def _verdict(summary: Any) -> str:
    """A candidate's pass/fail verdict from its stored summary.

    "pass" iff every planned fixture passed; "fail" covers both partial
    passes and runs that never completed (aggregate_eligible=False); the
    comparison degrades to "unknown" for malformed summaries.
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


def _search_verdicts(result: dict[str, Any]) -> tuple[dict[str, str], dict[str, int | None]]:
    """Per-candidate search verdicts plus per-candidate planned counts.

    Both come from the candidate's first "search" entry (duplicate names
    are first-wins). The planned count matters because a "pass" verdict
    means "all *planned* fixtures passed" -- planned 2 vs 50 with no
    evidence on either side must not compare as silently identical.
    ``None`` marks a malformed/missing summary; it compares as its own
    distinct value so it still shows up in drift.
    """
    verdicts: dict[str, str] = {}
    planned_counts: dict[str, int | None] = {}
    for entry in result.get("search") or []:
        name = _candidate_name(entry)
        if name is None or name in verdicts:
            continue
        summary = entry.get("summary") if type(entry) is dict else None
        verdicts[name] = _verdict(summary)
        planned = summary.get("planned") if type(summary) is dict else None
        planned_counts[name] = planned if type(planned) is int else None
    return verdicts, planned_counts


def _held_verdict(result: dict[str, Any]) -> tuple[str | None, str]:
    winner = result.get("selected_candidate")
    if type(winner) is not str or not winner:
        return None, "absent"
    held = result.get("held_out")
    summary = held.get("summary") if type(held) is dict else None
    return winner, _verdict(summary)


def _fixture_outcomes(observations: Any) -> tuple[dict[str, str], int]:
    """Map fixture id -> "pass"/"fail" from stored observations.

    pass requires both the exact answer and the boundary token match (the
    same criterion ``mpl run`` uses). Malformed observations are skipped
    and counted so the skip is visible, never silent.

    NOTE (documented inconsistency, behavior deliberately unchanged): a
    duplicate fixture id inside one candidate's observations is last-wins
    (each row overwrites the previous one), while a duplicate candidate
    name in "search" is first-wins for *verdicts* (``_search_verdicts``
    skips repeat names) but last-wins for *evidence* (``_stage_evidence``
    has no repeat-name guard, so a later entry with usable observations
    overwrites the earlier entry's outcomes; a later entry with none
    silently leaves the earlier outcomes in place). The two conventions
    were written at different times against different assumptions; ``mpl run`` never emits either
    duplicate, so reconciling them would change only hostile-input
    behavior. Left as-is rather than inventing a tie-break rule a future
    reader might mistake for a guarantee.
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


def _stage_evidence(
    result: dict[str, Any], stage: str
) -> tuple[dict[str, dict[str, str]], int, bool]:
    """Per-candidate fixture outcomes for one stage.

    Returns (candidate -> fixture -> outcome, skipped_count, present) where
    present is False when the side carries no usable evidence at all.
    """
    per_candidate: dict[str, dict[str, str]] = {}
    skipped = 0
    if stage == "search":
        entries = result.get("search") or []
        for entry in entries:
            name = _candidate_name(entry)
            if name is None:
                continue
            observations = entry.get("observations") if type(entry) is dict else None
            outcomes, skipped_here = _fixture_outcomes(observations)
            skipped += skipped_here
            if type(observations) is list and outcomes:
                per_candidate[name] = outcomes
    else:
        winner = result.get("selected_candidate")
        held = result.get("held_out")
        observations = held.get("observations") if type(held) is dict else None
        outcomes, skipped = _fixture_outcomes(observations)
        if type(winner) is str and winner and type(observations) is list and outcomes:
            per_candidate[winner] = outcomes
    return per_candidate, skipped, bool(per_candidate)


def _config_snapshot(result: dict[str, Any]) -> dict[str, Any]:
    """The comparability-relevant configuration of one run."""
    # Set-based dedup: the old `name not in names` list scan was O(n^2) and
    # hung ~1.5h on 800k candidate names. First-occurrence order is kept for
    # stability of the sorted() output contract.
    names: list[str] = []
    seen: set[str] = set()
    for entry in result.get("search") or []:
        name = _candidate_name(entry)
        if name is not None and name not in seen:
            seen.add(name)
            names.append(name)
    boundaries: dict[str, list[int]] = {}
    for entry in result.get("search") or []:
        name = _candidate_name(entry)
        ids = entry.get("boundary_token_ids") if type(entry) is dict else None
        if name is not None and type(ids) is list and all(type(token) is int for token in ids):
            boundaries[name] = list(ids)
    max_new_tokens = result.get("max_new_tokens")
    fingerprint = result.get("backend_fingerprint")
    return {
        "families": sorted({_base_family(name) for name in names}),
        "candidates": sorted(names),
        "max_new_tokens": max_new_tokens if type(max_new_tokens) is int else None,
        "boundaries": {name: boundaries[name] for name in sorted(boundaries)},
        "backend_fingerprint": fingerprint if type(fingerprint) is str else None,
    }


def _drift(old: dict[str, Any], new: dict[str, Any]) -> list[dict[str, Any]]:
    """Exact config-drift entries, in a fixed field order."""
    old_snap, new_snap = _config_snapshot(old), _config_snapshot(new)
    entries: list[dict[str, Any]] = []
    for field in ("families", "candidates", "max_new_tokens", "boundaries", "backend_fingerprint"):
        if old_snap[field] != new_snap[field]:
            entries.append({"field": field, "old": old_snap[field], "new": new_snap[field]})
    return entries


def _fixture_set_drift(
    old_evidence: dict[str, dict[str, dict[str, str]]],
    new_evidence: dict[str, dict[str, dict[str, str]]],
) -> list[dict[str, Any]]:
    """Fixture-set drift, gated on evidence presence on both sides.

    Comparing runs measured on different fixture files is the same class
    of operator mistake as config drift, so it is flagged here rather than
    silently unioning the sets.

    NOTE: the gate lives in ``diff_reports`` as one any-stage flag per side
    (search or held-out), not per-stage. When one side carries search
    evidence and the other only held-out evidence, the drift entries and
    "absent" fixture diffs for the empty stages are still emitted with an
    empty id list on that side.
    """
    entries: list[dict[str, Any]] = []
    for stage in ("search", "held_out"):
        old_ids = sorted({fixture for cand in old_evidence[stage].values() for fixture in cand})
        new_ids = sorted({fixture for cand in new_evidence[stage].values() for fixture in cand})
        if old_ids != new_ids:
            entries.append(
                {
                    "field": f"fixtures:{stage}",
                    "old": old_ids,
                    "new": new_ids,
                }
            )
    return entries


def diff_reports(
    old: dict[str, Any],
    new: dict[str, Any],
    *,
    all_fixtures: bool = False,
    old_label: str = "old",
    new_label: str = "new",
) -> dict[str, Any]:
    """Compare two loaded ``results.json`` dicts; deterministic.

    Same inputs always produce the same output dict (iteration order is
    fixed by sorting; serialization with ``sort_keys=True`` is
    byte-identical). No model, no network.
    """
    old_decision = old["decision"]
    new_decision = new["decision"]
    old_winner = old.get("selected_candidate")
    new_winner = new.get("selected_candidate")
    # Normalize "" to None, matching _held_verdict (which treats a falsy
    # winner as absent): otherwise two winnerless runs disagree across axes.
    old_winner = old_winner if type(old_winner) is str and old_winner else None
    new_winner = new_winner if type(new_winner) is str and new_winner else None

    # Candidate verdict flips on search and held-out.
    old_search, old_planned = _search_verdicts(old)
    new_search, new_planned = _search_verdicts(new)
    old_held_name, old_held = _held_verdict(old)
    new_held_name, new_held = _held_verdict(new)
    verdict_changes: list[dict[str, str]] = []
    for name in sorted(set(old_search) | set(new_search)):
        old_v = old_search.get(name, "absent")
        new_v = new_search.get(name, "absent")
        if old_v != new_v:
            verdict_changes.append(
                {"candidate": name, "stage": "search", "old": old_v, "new": new_v}
            )
    for name in sorted({n for n in (old_held_name, new_held_name) if n is not None}):
        old_v = old_held if name == old_held_name else "absent"
        new_v = new_held if name == new_held_name else "absent"
        if old_v != new_v:
            verdict_changes.append(
                {"candidate": name, "stage": "held_out", "old": old_v, "new": new_v}
            )
    changed_candidates = sorted({change["candidate"] for change in verdict_changes})

    # Per-fixture outcome diffs (evidence-gated).
    old_evidence: dict[str, dict[str, dict[str, str]]] = {}
    new_evidence: dict[str, dict[str, dict[str, str]]] = {}
    skipped: dict[str, int] = {"old": 0, "new": 0}
    evidence_present = {"old": False, "new": False}
    for stage in ("search", "held_out"):
        old_map, old_skip, old_ok = _stage_evidence(old, stage)
        new_map, new_skip, new_ok = _stage_evidence(new, stage)
        old_evidence[stage], new_evidence[stage] = old_map, new_map
        skipped["old"] += old_skip
        skipped["new"] += new_skip
        evidence_present["old"] = evidence_present["old"] or old_ok
        evidence_present["new"] = evidence_present["new"] or new_ok

    fixture_diffs: list[dict[str, str]] = []
    changed_fixture_diffs: list[dict[str, str]] = []
    if evidence_present["old"] and evidence_present["new"]:
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
    # Fixed ordering: (stage, candidate, fixture) is a total order over the
    # compared cells, so the output list order is deterministic regardless
    # of dict/set iteration.
    fixture_diffs.sort(key=lambda e: (e["stage"], e["candidate"], e["fixture"]))
    changed_fixture_diffs.sort(key=lambda e: (e["stage"], e["candidate"], e["fixture"]))
    flips = [
        entry
        for entry in changed_fixture_diffs
        if entry["old"] in ("pass", "fail") and entry["new"] in ("pass", "fail")
    ]

    config_drift = _drift(old, new)
    # Per-candidate planned-fixture counts: a "pass" verdict means "all
    # *planned* fixtures passed", so planned 2 vs 50 silently comparing
    # identical would be a lie of omission. Candidates present on only one
    # side already produce a verdict flip ("absent"), so this covers only
    # the both-sides intersection. Field order is fixed: after the
    # _drift() fields, sorted by candidate name.
    for name in sorted(set(old_planned) & set(new_planned)):
        if old_planned[name] != new_planned[name]:
            config_drift.append(
                {"field": f"planned:{name}", "old": old_planned[name], "new": new_planned[name]}
            )
    if evidence_present["old"] and evidence_present["new"]:
        config_drift.extend(_fixture_set_drift(old_evidence, new_evidence))

    decision_changed = old_decision != new_decision
    winner_changed = old_winner != new_winner
    # Evidence-presence asymmetry feeds the identical verdict: when only one
    # side carries --include-evidence observations there is a whole axis
    # (per-fixture outcomes) that could not be compared, so the runs are
    # not "identical on every compared axis" -- and the summary line below
    # now names the axes that were actually compared instead of asserting a
    # totality that may not hold.
    evidence_asymmetric = evidence_present["old"] != evidence_present["new"]
    frontier_diff = diff_frontiers(old.get("budget_frontier"), new.get("budget_frontier"))
    frontier_asymmetric = frontier_diff["available"] not in ("both", "neither")
    held_out_floor_diff = diff_held_out_floor(
        old.get("held_out_budget_floor"), new.get("held_out_budget_floor")
    )
    held_out_floor_asymmetric = held_out_floor_diff["available"] not in (
        "both",
        "neither",
    )
    boundary_audit_diff = diff_boundary_audit(old.get("boundary_audit"), new.get("boundary_audit"))
    boundary_audit_asymmetric = boundary_audit_diff["available"] not in (
        "both",
        "neither",
    )
    robustness_diff = diff_selection_robustness(
        old.get("selection_robustness"), new.get("selection_robustness")
    )
    robustness_asymmetric = robustness_diff["available"] not in (
        "both",
        "neither",
    )
    transfer_diff = diff_transfer(old.get("transfer"), new.get("transfer"))
    transfer_asymmetric = transfer_diff["available"] not in (
        "both",
        "neither",
    )
    contamination_diff = diff_contamination(old.get("contamination"), new.get("contamination"))
    contamination_asymmetric = contamination_diff["available"] not in (
        "both",
        "neither",
    )
    perturbation_diff = diff_perturbation(old.get("perturbation"), new.get("perturbation"))
    perturbation_asymmetric = perturbation_diff["available"] not in (
        "both",
        "neither",
    )
    healing_diff = diff_token_healing(old.get("token_healing"), new.get("token_healing"))
    healing_asymmetric = healing_diff["available"] not in (
        "both",
        "neither",
    )
    hierarchy_diff = diff_hierarchy(old.get("hierarchy"), new.get("hierarchy"))
    hierarchy_asymmetric = hierarchy_diff["available"] not in (
        "both",
        "neither",
    )
    prefill_diff = diff_prefill(old.get("prefill"), new.get("prefill"))
    prefill_asymmetric = prefill_diff["available"] not in (
        "both",
        "neither",
    )
    portfolio_diff = diff_portfolio(old.get("portfolio"), new.get("portfolio"))
    portfolio_asymmetric = portfolio_diff["available"] not in (
        "both",
        "neither",
    )
    synthhistory_diff = diff_synthhistory(old.get("synthhistory"), new.get("synthhistory"))
    synthhistory_asymmetric = synthhistory_diff["available"] not in (
        "both",
        "neither",
    )
    batchinv_diff = diff_batchinv(old.get("batch_invariance"), new.get("batch_invariance"))
    batchinv_asymmetric = batchinv_diff["available"] not in (
        "both",
        "neither",
    )
    position_diff = diff_position(old.get("position"), new.get("position"))
    position_asymmetric = position_diff["available"] not in (
        "both",
        "neither",
    )
    stopcause_diff = diff_stop_cause(old.get("stop_cause"), new.get("stop_cause"))
    stopcause_asymmetric = stopcause_diff["available"] not in (
        "both",
        "neither",
    )
    identical = (
        not decision_changed
        and not winner_changed
        and not verdict_changes
        and not changed_fixture_diffs
        and not config_drift
        and not evidence_asymmetric
        and not frontier_diff["changed"]
        and not frontier_asymmetric
        and not held_out_floor_diff["changed"]
        and not held_out_floor_asymmetric
        and not boundary_audit_diff["changed"]
        and not boundary_audit_asymmetric
        and not robustness_diff["changed"]
        and not robustness_asymmetric
        and not transfer_diff["changed"]
        and not transfer_asymmetric
        and not contamination_diff["changed"]
        and not contamination_asymmetric
        and not perturbation_diff["changed"]
        and not perturbation_asymmetric
        and not healing_diff["changed"]
        and not healing_asymmetric
        and not hierarchy_diff["changed"]
        and not hierarchy_asymmetric
        and not prefill_diff["changed"]
        and not prefill_asymmetric
        and not portfolio_diff["changed"]
        and not portfolio_asymmetric
        and not synthhistory_diff["changed"]
        and not synthhistory_asymmetric
        and not batchinv_diff["changed"]
        and not batchinv_asymmetric
        and not position_diff["changed"]
        and not position_asymmetric
        and not stopcause_diff["changed"]
        and not stopcause_asymmetric
    )
    summary = {
        "decision_changed": decision_changed,
        "winner_changed": winner_changed,
        "candidates_changed_status": len(changed_candidates),
        "fixtures_flipped": len(flips),
        "config_drift_warnings": len(config_drift),
        "evidence_asymmetric": evidence_asymmetric,
        "frontier_changed": frontier_diff["changed"],
        "frontier_asymmetric": frontier_asymmetric,
        "held_out_floor_changed": held_out_floor_diff["changed"],
        "held_out_floor_asymmetric": held_out_floor_asymmetric,
        "boundary_audit_changed": boundary_audit_diff["changed"],
        "boundary_audit_asymmetric": boundary_audit_asymmetric,
        "selection_robustness_changed": robustness_diff["changed"],
        "selection_robustness_asymmetric": robustness_asymmetric,
        "transfer_changed": transfer_diff["changed"],
        "transfer_asymmetric": transfer_asymmetric,
        "contamination_changed": contamination_diff["changed"],
        "contamination_asymmetric": contamination_asymmetric,
        "perturbation_changed": perturbation_diff["changed"],
        "perturbation_asymmetric": perturbation_asymmetric,
        "healing_changed": healing_diff["changed"],
        "healing_asymmetric": healing_asymmetric,
        "hierarchy_changed": hierarchy_diff["changed"],
        "hierarchy_asymmetric": hierarchy_asymmetric,
        "prefill_changed": prefill_diff["changed"],
        "prefill_asymmetric": prefill_asymmetric,
        "portfolio_changed": portfolio_diff["changed"],
        "portfolio_asymmetric": portfolio_asymmetric,
        "synthhistory_changed": synthhistory_diff["changed"],
        "synthhistory_asymmetric": synthhistory_asymmetric,
        "batch_invariance_changed": batchinv_diff["changed"],
        "batch_invariance_asymmetric": batchinv_asymmetric,
        "position_changed": position_diff["changed"],
        "position_asymmetric": position_asymmetric,
        "stop_cause_changed": stopcause_diff["changed"],
        "stop_cause_asymmetric": stopcause_asymmetric,
        "identical": identical,
    }
    return {
        "tool": "diff-reports",
        "old": {"source": old_label},
        "new": {"source": new_label},
        "decision": {"old": old_decision, "new": new_decision, "changed": decision_changed},
        "winner": {"old": old_winner, "new": new_winner, "changed": winner_changed},
        "frontier": frontier_diff,
        "held_out_floor": held_out_floor_diff,
        "boundary_audit": boundary_audit_diff,
        "selection_robustness": robustness_diff,
        "transfer": transfer_diff,
        "contamination": contamination_diff,
        "perturbation": perturbation_diff,
        "token_healing": healing_diff,
        "hierarchy": hierarchy_diff,
        "prefill": prefill_diff,
        "portfolio": portfolio_diff,
        "synthhistory": synthhistory_diff,
        "batch_invariance": batchinv_diff,
        "position": position_diff,
        "stop_cause": stopcause_diff,
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
    found = False
    total_points = 0
    for data in candidates.values():
        if type(data) is not dict:
            continue
        if len(data.get("fixture_ids") or []) <= 0:
            continue
        for point in data.get("pass_steps") or []:
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


def _normalize_attribution(value: Any) -> list[dict[str, Any]] | None:
    """Canonicalize a ``flip_attribution`` field for comparison.

    Returns ``[{"span": [lo, hi], "crossings": [...]}]`` with crossings
    sorted by (min_budget, candidate, fixture), or None when the field is
    absent or not a list. Malformed entries are skipped; a present-but-
    empty field normalizes to ``[]``, which is distinct from None (an
    older payload that never recorded attribution).
    """
    if type(value) is not list:
        return None
    normalized: list[dict[str, Any]] = []
    for entry in value:
        if type(entry) is not dict:
            continue
        span = entry.get("span")
        if (
            type(span) is not list
            or len(span) != 2
            or type(span[0]) is not int
            or type(span[1]) is not int
        ):
            continue
        crossings: list[dict[str, Any]] = []
        raw_crossings = entry.get("crossings")
        if type(raw_crossings) is list:
            for crossing in raw_crossings:
                if (
                    type(crossing) is dict
                    and type(crossing.get("candidate")) is str
                    and type(crossing.get("fixture")) is str
                    and type(crossing.get("min_budget")) is int
                ):
                    crossings.append(
                        {
                            "candidate": crossing["candidate"],
                            "fixture": crossing["fixture"],
                            "min_budget": crossing["min_budget"],
                        }
                    )
        crossings.sort(key=lambda c: (c["min_budget"], c["candidate"], c["fixture"]))
        normalized.append({"span": [span[0], span[1]], "crossings": crossings})
    return normalized


def _step_function(points: list[tuple[int, Any]]) -> Callable[[int], Any]:
    """Step function over sorted (budget, value) change points.

    The value at ``budget`` is the last change point at or below it; the
    value before the first point is the ``("unknown", None)`` sentinel --
    callers only compare budgets inside the compared range, so unknown
    never silently equals a real value. Sorted on the budget only (stable):
    duplicate budgets keep first-occurrence order, so a foreign payload
    can never crash the sort by comparing incomparable values (perf
    finding 1B). Lookups are O(log n) via bisect.
    """
    ordered = sorted(points, key=lambda point: point[0])
    keys = [budget for budget, _ in ordered]

    def value_at(budget: int) -> Any:
        index = bisect_right(keys, budget) - 1
        return ordered[index][1] if index >= 0 else ("unknown", None)

    return value_at


def _diff_spans(
    old_at: Callable[[int], Any],
    new_at: Callable[[int], Any],
    breakpoints: list[int],
    hi: int,
) -> list[list[Any]]:
    """Merged [lo, hi, old, new] spans where two step functions differ.

    Both functions are stepwise constant with breakpoints drawn from the
    union of their change budgets, so comparing at the breakpoints is
    exact; adjacent differing spans with identical values merge.
    """
    points = sorted({b for b in breakpoints if 1 <= b <= hi})
    spans: list[list[Any]] = []
    for index, budget in enumerate(points):
        end = points[index + 1] - 1 if index + 1 < len(points) else hi
        old_value = old_at(budget)
        new_value = new_at(budget)
        if old_value == new_value:
            continue
        if (
            spans
            and spans[-1][2] == old_value
            and spans[-1][3] == new_value
            and spans[-1][1] == budget - 1
        ):
            spans[-1][1] = end
        else:
            spans.append([budget, end, old_value, new_value])
    return spans


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
        points = []
        seen_budgets: set[int] = set()
        for entry in frontier.get("decision_steps") or []:
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
    floor = value.get("safe_budget_floor")
    return not (floor is not None and type(floor) is not int)


def _valid_boundary_audit(value: Any) -> bool:
    """Light validation: is this a comparable ``boundary_audit`` payload?"""
    if type(value) is not dict or value.get("available", True) is False:
        return False
    if type(value.get("per_family")) is not dict:
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


def _stability_triplet(stability: Any) -> tuple[Any, Any, Any]:
    """(winner_retained, decision_retained, n_evaluable) for drift comparison."""
    if type(stability) is not dict:
        return (None, None, None)
    return (
        stability.get("winner_retained"),
        stability.get("decision_retained"),
        stability.get("n_evaluable"),
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


def _valid_position(payload: Any) -> bool:
    """Structural check for a ``position`` payload.

    Enforces the wave-2 C3 lesson at the value level: the compared
    fields must be bools/strings/None/numbers so the renderers never hit
    a TypeError on a hostile payload, and map keys must be strings.
    """
    if type(payload) is not dict or payload.get("available") is not True:
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
    primacy = payload.get("primacy_index")
    if type(primacy) is not dict or not all(type(k) is str for k in primacy):
        return False
    for value in primacy.values():
        if value is not None and type(value) not in (int, float):
            return False
    effect = payload.get("condition_effect")
    if effect is not None:
        if type(effect) is not dict:
            return False
        mean = effect.get("mean_index")
        if mean is not None and type(mean) not in (int, float):
            return False
        for key in ("n_candidates", "n_positive", "n_negative"):
            if type(effect.get(key)) is not int:
                return False
    return type(payload.get("flips")) is list


def _position_condition_pair(payload: Any, condition: str) -> tuple[Any, Any]:
    """(decision, winner) for one position condition, for drift comparison."""
    per_condition = payload.get("per_condition") or {}
    entry = per_condition.get(condition) or {}
    return (entry.get("decision"), entry.get("winner"))


def diff_position(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``position`` payloads; deterministic.

    Answers "did the position-sensitivity story change between runs?":
    the per-condition (decision, winner) pairs, the winner-flip pattern
    against the primary run, per-candidate correct-rate drift, and the
    per-candidate primacy-index drift. A primacy-index shift after a
    template or fixture change is the position-bias regression signal.
    Candidates present on only one side are reported as added/removed.
    A side without a recorded position probe is reported plainly. No
    model, no network.
    """
    old_ok = _valid_position(old)
    new_ok = _valid_position(new)
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
        old_pair = _position_condition_pair(old, condition)
        new_pair = _position_condition_pair(new, condition)
        changed = old_pair != new_pair
        condition_changed[condition] = {
            "old_decision": old_pair[0],
            "old_winner": old_pair[1],
            "new_decision": new_pair[0],
            "new_winner": new_pair[1],
            "changed": changed,
        }

    def _flip_key(flips: list[Any]) -> list[tuple[Any, ...]]:
        # ``flips`` is a list here: _valid_position rejects anything else
        # before this point, so no defensive branch is needed.
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

    # NOTE: deliberately phrased to stay textually distinct from the
    # cycle-147 flips gate anchor ("flips_changed = _flip_key(old...)
    # != _flip_key(new...)"): new diff sections must not duplicate gate
    # anchors (see the lesson in AGENTS.md).
    flips_changed = _flip_key(new.get("flips")) != _flip_key(old.get("flips"))

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
    candidate_drift: list[dict[str, Any]] = []
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
    index_drift: list[dict[str, Any]] = []
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

    mean_index_changed = _mean_index(old) != _mean_index(new)

    result["conditions_added"] = conditions_added
    result["conditions_removed"] = conditions_removed
    result["condition_changed"] = condition_changed
    result["flips_changed"] = flips_changed
    result["candidate_drift"] = candidate_drift
    result["candidates_added"] = candidates_added
    result["candidates_removed"] = candidates_removed
    result["index_drift"] = index_drift
    result["mean_index"] = {
        "old": _mean_index(old),
        "new": _mean_index(new),
        "changed": mean_index_changed,
    }
    # NOTE: position-specific disjunct names keep these lines textually
    # distinct from every other diff section's ``changed`` anchor.
    result["changed"] = bool(
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
    return result


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
        oracle = payload.get("oracle") or {}
        return set(oracle.get("solvable_fixture_ids") or [])

    old_solvable = _solvable(old)
    new_solvable = _solvable(new)
    solvable_added = sorted(new_solvable - old_solvable)
    solvable_removed = sorted(old_solvable - new_solvable)
    result["solvable_added"] = solvable_added
    result["solvable_removed"] = solvable_removed

    def _members(payload: Any) -> list[str]:
        cover = payload.get("portfolio") or {}
        return list(cover.get("members") or [])

    old_members = _members(old)
    new_members = _members(new)
    members_added = [member for member in new_members if member not in old_members]
    members_removed = [member for member in old_members if member not in new_members]
    result["members_added"] = members_added
    result["members_removed"] = members_removed

    def _fixture_winners(payload: Any) -> dict[str, list[str]]:
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
        return f"{value:.1%}" if type(value) in (int, float) else "(none)"

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


def _render_position_summary(position_diff: dict[str, Any] | None) -> list[str]:
    """Position-sensitivity comparison lines for the stderr summary."""
    if type(position_diff) is not dict:
        return []
    available = position_diff.get("available")
    if available == "neither":
        return []
    if available != "both":
        missing = "new" if available == "old" else "old"
        return [
            "Position-sensitivity comparison unavailable: "
            f"{missing} run did not record the position probe."
        ]
    lines: list[str] = []

    def _verdict(decision: Any, winner: Any) -> str:
        return f"{_display_id(decision)}/{_display_id('(none)' if winner is None else winner)}"

    conditions = position_diff.get("condition_changed") or {}
    for condition in sorted(conditions):
        entry = conditions[condition]
        if entry.get("changed"):
            lines.append(
                f"Position ({_display_id(condition)}): "
                f"{_verdict(entry.get('old_decision'), entry.get('old_winner'))} -> "
                f"{_verdict(entry.get('new_decision'), entry.get('new_winner'))}"
            )
    added = position_diff.get("conditions_added") or []
    removed = position_diff.get("conditions_removed") or []
    if added:
        lines.append(f"Position conditions added: {', '.join(_display_id(c) for c in added)}")
    if removed:
        lines.append(f"Position conditions removed: {', '.join(_display_id(c) for c in removed)}")
    if position_diff.get("flips_changed"):
        lines.append("Position flip pattern changed between runs.")

    def _fmt_index(value: Any) -> str:
        return f"{value:+.2f}" if type(value) in (int, float) else "(none)"

    drift = position_diff.get("index_drift") or []
    if drift:
        lines.append(f"Primacy-index drift ({len(drift)}):")
        for entry in drift[:5]:
            lines.append(
                f"  {_display_id(entry['candidate'])}: "
                f"{_fmt_index(entry.get('old_primacy_index'))} -> "
                f"{_fmt_index(entry.get('new_primacy_index'))}"
            )
        if len(drift) > 5:
            lines.append(f"  ... and {len(drift) - 5} more")
    mean = position_diff.get("mean_index") or {}
    if mean.get("changed"):
        lines.append(
            f"Mean primacy index: {_fmt_index(mean.get('old'))} -> {_fmt_index(mean.get('new'))}"
        )
    added_c = position_diff.get("candidates_added") or []
    removed_c = position_diff.get("candidates_removed") or []
    if added_c:
        lines.append(f"Position candidates added: {', '.join(_display_id(c) for c in added_c)}")
    if removed_c:
        lines.append(f"Position candidates removed: {', '.join(_display_id(c) for c in removed_c)}")
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
        return f"{value:.0%}" if type(value) in (int, float) else "(none)"

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
        return f"{value:.1%}" if type(value) in (int, float) else "(none)"

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
        return f"{value:.1%}" if type(value) in (int, float) else "(none)"

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
        return f"{rate:.1%}" if type(rate) in (int, float) else "(none)"

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


def _compact(value: Any) -> str:
    """One-line JSON for a drift value in the stderr summary."""
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


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


def render_summary(diff: dict[str, Any]) -> str:
    """Human-readable summary for stderr; stdout stays parseable JSON."""
    lines = [
        # Source labels arrive from the compared files' caller (CLI paths
        # by default) and decisions arrive from the compared files
        # themselves; both are sanitized like every other
        # operator-visible identifier.
        f"diff-reports: {_display_id(diff['old']['source'])} vs "
        f"{_display_id(diff['new']['source'])}",
    ]
    drift = diff["config_drift"]
    if drift:
        lines.append("WARNING: run configurations differ -- results may not be comparable:")
        for entry in drift:
            # The planned-drift field is built as f"planned:{name}" from a
            # raw candidate name in the compared files, so the field is
            # sanitized like every other operator-visible identifier. Fixed
            # fields (families, fixtures:search, ...) are unaffected by the
            # strip, so sanitizing at render covers every drift field type.
            lines.append(
                f"  - {_display_id(entry['field'])}: {_compact(entry['old'])} -> "
                f"{_compact(entry['new'])}"
            )
    decision = diff["decision"]
    lines.append(f"Decision: {_display_id(decision['old'])} -> {_display_id(decision['new'])}")
    winner = diff["winner"]
    old_w = winner["old"] if winner["old"] is not None else "(none)"
    new_w = winner["new"] if winner["new"] is not None else "(none)"
    lines.append(f"Winner: {_display_id(old_w)} -> {_display_id(new_w)}")
    lines.extend(_render_frontier_summary(diff["frontier"]))
    lines.extend(_render_held_out_floor_summary(diff["held_out_floor"]))
    lines.extend(_render_boundary_audit_summary(diff["boundary_audit"]))
    lines.extend(_render_robustness_summary(diff.get("selection_robustness")))
    lines.extend(_render_transfer_summary(diff.get("transfer")))
    lines.extend(_render_contamination_summary(diff.get("contamination")))
    lines.extend(_render_perturbation_summary(diff.get("perturbation")))
    lines.extend(_render_healing_summary(diff.get("token_healing")))
    lines.extend(_render_hierarchy_summary(diff.get("hierarchy")))
    lines.extend(_render_prefill_summary(diff.get("prefill")))
    lines.extend(_render_synthhistory_summary(diff.get("synthhistory")))
    lines.extend(_render_batchinv_summary(diff.get("batch_invariance")))
    lines.extend(_render_position_summary(diff.get("position")))
    lines.extend(_render_stop_cause_summary(diff.get("stop_cause")))
    verdicts = diff["candidate_verdicts"]
    if verdicts:
        lines.append(f"Candidate verdict changes ({len(verdicts)}):")
        for change in verdicts:
            lines.append(
                f"  {change['stage']}/{_display_id(change['candidate'])}: "
                f"{change['old']} -> {change['new']}"
            )
    evidence = diff["evidence"]
    summary = diff["summary"]
    if evidence["old"] and evidence["new"]:
        diffs = diff["fixture_diffs"]
        changed = [d for d in diffs if d["old"] != d["new"]]
        lines.append(f"Fixture outcome diffs ({len(changed)} changed):")
        for entry in changed:
            lines.append(
                f"  {entry['stage']}/{_display_id(entry['candidate'])}/"
                f"{_display_id(entry['fixture'])}: {entry['old']} -> {entry['new']}"
            )
        if not changed:
            lines.append("  (none)")
    else:
        missing = [side for side in ("old", "new") if not evidence[side]]
        lines.append(
            "Per-fixture comparison unavailable: "
            + ", ".join(f"{side} run has no --include-evidence observations" for side in missing)
        )
    if summary["identical"]:
        axes = ["decision", "winner", "candidate verdicts", "configuration"]
        if evidence["old"] and evidence["new"]:
            axes.append("per-fixture evidence")
        if diff["frontier"]["available"] == "both":
            axes.append("token-budget frontier")
        if diff["held_out_floor"]["available"] == "both":
            axes.append("held-out budget floor")
        if diff["perturbation"]["available"] == "both":
            axes.append("fixture-perturbation robustness")
        if diff["token_healing"]["available"] == "both":
            axes.append("token-healing sensitivity")
        if diff["hierarchy"]["available"] == "both":
            axes.append("instruction-hierarchy adherence")
        if diff["prefill"]["available"] == "both":
            axes.append("assistant-prefill sensitivity")
        if diff["portfolio"]["available"] == "both":
            axes.append("protocol-portfolio gap")
        lines.append(
            "No differences: the two runs are identical on every compared axis "
            f"({', '.join(axes)})."
        )
    else:
        lines.append(
            "Summary: "
            f"decision {'CHANGED' if summary['decision_changed'] else 'unchanged'}, "
            f"winner {'CHANGED' if summary['winner_changed'] else 'unchanged'}, "
            f"frontier {'CHANGED' if summary['frontier_changed'] else 'unchanged'}, "
            f"{summary['candidates_changed_status']} candidate(s) changed verdict, "
            f"{summary['fixtures_flipped']} fixture(s) flipped, "
            f"{summary['config_drift_warnings']} config drift warning(s)."
        )
    return "\n".join(lines)


def run_diff_reports(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Implement ``mpl diff-reports``; returns the process exit code.

    JSON goes to stdout, the human-readable summary to stderr (mirroring
    ``diff-backends``). A successful comparison exits 0 even when it finds
    differences -- differences are the tool's useful output, not a failure.
    """
    loaded: dict[str, dict[str, Any]] = {}
    for side, path in (("old", args.old_results), ("new", args.new_results)):
        try:
            loaded[side] = load_results(path)
        except ValueError as exc:
            # Fixed error vocabulary from load_results; safe to show and
            # never a traceback.
            print(f"diff-reports: {exc} ({side} input)", file=sys.stderr)
            return 1
    diff = diff_reports(
        loaded["old"],
        loaded["new"],
        all_fixtures=args.all,
        old_label=str(args.old_results),
        new_label=str(args.new_results),
    )
    print(json.dumps(diff, indent=2, ensure_ascii=True, sort_keys=True))
    print(render_summary(diff), file=sys.stderr)
    return 0
