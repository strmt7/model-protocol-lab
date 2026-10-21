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
import hashlib
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


def parse_results_bytes(content: bytes) -> dict[str, Any]:
    """Parse ``results.json`` bytes under the bounded reader's rules.

    Shared by :func:`load_results` (file path), ``pack`` (sealing), and
    ``verify --bundle`` (replay), so all three call sites reject exactly
    the same artifacts with exactly the same fixed error codes: oversize,
    non-UTF-8, duplicate fields, non-finite numbers, excessive nesting,
    non-object roots, and wrong minimal shape.
    """
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
    return parse_results_bytes(content)


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
    wscan_diff = diff_ws_canonical(old.get("ws_canonical"), new.get("ws_canonical"))
    wscan_asymmetric = wscan_diff["available"] not in (
        "both",
        "neither",
    )
    structured_diff = diff_structured(old.get("structured"), new.get("structured"))
    structured_asymmetric = structured_diff["available"] not in (
        "both",
        "neither",
    )
    deliforge_diff = diff_deliforge(old.get("deliforge"), new.get("deliforge"))
    deliforge_asymmetric = deliforge_diff["available"] not in (
        "both",
        "neither",
    )
    fewshot_diff = diff_fewshot(old.get("fewshot"), new.get("fewshot"))
    fewshot_asymmetric = fewshot_diff["available"] not in (
        "both",
        "neither",
    )
    faketurn_diff = diff_faketurn(old.get("faketurn"), new.get("faketurn"))
    faketurn_asymmetric = faketurn_diff["available"] not in (
        "both",
        "neither",
    )
    toolcall_diff = diff_toolcall(old.get("toolcall"), new.get("toolcall"))
    toolcall_asymmetric = toolcall_diff["available"] not in (
        "both",
        "neither",
    )
    verbosity_diff = diff_verbosity(old.get("verbosity"), new.get("verbosity"))
    refusals_diff = diff_refusals(old.get("refusals"), new.get("refusals"))
    scorerlens_diff = diff_scorerlens(old.get("scorerlens"), new.get("scorerlens"))
    aliasing_diff = diff_aliasing(old.get("aliasing"), new.get("aliasing"))
    repetition_diff = diff_repetition(old.get("repetition"), new.get("repetition"))
    distraction_diff = diff_distraction(old.get("distraction"), new.get("distraction"))
    leading_diff = diff_leading(old.get("leading"), new.get("leading"))
    cache_profile_diff = diff_cache_profile(old.get("cache_profile"), new.get("cache_profile"))
    template_robustness_diff = diff_template_robustness(
        old.get("template_robustness"), new.get("template_robustness")
    )
    ablation_diff = diff_ablation(old.get("ablation"), new.get("ablation"))
    confidentiality_diff = diff_confidentiality(
        old.get("confidentiality"), new.get("confidentiality")
    )
    answer_collision_diff = diff_answer_collision(
        old.get("answer_collision"), new.get("answer_collision")
    )
    ablation_asymmetric = ablation_diff["available"] not in (
        "both",
        "neither",
    )
    confidentiality_asymmetric = confidentiality_diff["available"] not in (
        "both",
        "neither",
    )
    answer_collision_asymmetric = answer_collision_diff["available"] not in (
        "both",
        "neither",
    )
    verbosity_asymmetric = verbosity_diff["available"] not in (
        "both",
        "neither",
    )
    refusals_asymmetric = refusals_diff["available"] not in (
        "both",
        "neither",
    )
    scorerlens_asymmetric = scorerlens_diff["available"] not in (
        "both",
        "neither",
    )
    aliasing_asymmetric = aliasing_diff["available"] not in (
        "both",
        "neither",
    )
    repetition_asymmetric = repetition_diff["available"] not in (
        "both",
        "neither",
    )
    distraction_asymmetric = distraction_diff["available"] not in (
        "both",
        "neither",
    )
    leading_asymmetric = leading_diff["available"] not in (
        "both",
        "neither",
    )
    cache_profile_asymmetric = cache_profile_diff["available"] not in (
        "both",
        "neither",
    )
    template_robustness_asymmetric = template_robustness_diff["available"] not in (
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
        and not wscan_diff["changed"]
        and not wscan_asymmetric
        and not structured_diff["changed"]
        and not structured_asymmetric
        and not deliforge_diff["changed"]
        and not deliforge_asymmetric
        and not fewshot_diff["changed"]
        and not fewshot_asymmetric
        and not faketurn_diff["changed"]
        and not faketurn_asymmetric
        and not toolcall_diff["changed"]
        and not toolcall_asymmetric
        and not verbosity_diff["changed"]
        and not verbosity_asymmetric
        and not refusals_diff["changed"]
        and not refusals_asymmetric
        and not cache_profile_diff["changed"]
        and not cache_profile_asymmetric
        and not template_robustness_diff["changed"]
        and not template_robustness_asymmetric
        and not ablation_diff["changed"]
        and not ablation_asymmetric
        and not confidentiality_diff["changed"]
        and not confidentiality_asymmetric
        and not answer_collision_diff["changed"]
        and not answer_collision_asymmetric
        and not scorerlens_diff["changed"]
        and not scorerlens_asymmetric
        and not aliasing_diff["changed"]
        and not aliasing_asymmetric
        and not repetition_diff["changed"]
        and not repetition_asymmetric
        and not distraction_diff["changed"]
        and not distraction_asymmetric
        and not leading_diff["changed"]
        and not leading_asymmetric
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
        "ws_canonical_changed": wscan_diff["changed"],
        "ws_canonical_asymmetric": wscan_asymmetric,
        "structured_changed": structured_diff["changed"],
        "structured_asymmetric": structured_asymmetric,
        "deliforge_changed": deliforge_diff["changed"],
        "deliforge_asymmetric": deliforge_asymmetric,
        "fewshot_changed": fewshot_diff["changed"],
        "fewshot_asymmetric": fewshot_asymmetric,
        "faketurn_changed": faketurn_diff["changed"],
        "faketurn_asymmetric": faketurn_asymmetric,
        "toolcall_changed": toolcall_diff["changed"],
        "toolcall_asymmetric": toolcall_asymmetric,
        "verbosity_changed": verbosity_diff["changed"],
        "verbosity_asymmetric": verbosity_asymmetric,
        "refusals_changed": refusals_diff["changed"],
        "refusals_asymmetric": refusals_asymmetric,
        "cache_profile_changed": cache_profile_diff["changed"],
        "cache_profile_asymmetric": cache_profile_asymmetric,
        "template_robustness_changed": template_robustness_diff["changed"],
        "template_robustness_asymmetric": template_robustness_asymmetric,
        "ablation_changed": ablation_diff["changed"],
        "ablation_asymmetric": ablation_asymmetric,
        "confidentiality_changed": confidentiality_diff["changed"],
        "confidentiality_asymmetric": confidentiality_asymmetric,
        "answer_collision_changed": answer_collision_diff["changed"],
        "answer_collision_asymmetric": answer_collision_asymmetric,
        "scorerlens_changed": scorerlens_diff["changed"],
        "scorerlens_asymmetric": scorerlens_asymmetric,
        "aliasing_changed": aliasing_diff["changed"],
        "aliasing_asymmetric": aliasing_asymmetric,
        "repetition_changed": repetition_diff["changed"],
        "repetition_asymmetric": repetition_asymmetric,
        "distraction_changed": distraction_diff["changed"],
        "distraction_asymmetric": distraction_asymmetric,
        "leading_changed": leading_diff["changed"],
        "leading_asymmetric": leading_asymmetric,
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
        "ws_canonical": wscan_diff,
        "structured": structured_diff,
        "deliforge": deliforge_diff,
        "fewshot": fewshot_diff,
        "faketurn": faketurn_diff,
        "toolcall": toolcall_diff,
        "verbosity": verbosity_diff,
        "refusals": refusals_diff,
        "scorerlens": scorerlens_diff,
        "aliasing": aliasing_diff,
        "repetition": repetition_diff,
        "distraction": distraction_diff,
        "leading": leading_diff,
        "cache_profile": cache_profile_diff,
        "template_robustness": template_robustness_diff,
        "ablation": ablation_diff,
        "confidentiality": confidentiality_diff,
        "answer_collision": answer_collision_diff,
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


#: Whitespace mutations mirrored from wscanonical.MUTATIONS (kept local
#: so the differ stays import-light; pinned equal by test).
_WSCAN_MUTATIONS = (
    "trailing_space",
    "double_trailing_space",
    "trailing_newline",
    "trailing_tab",
    "leading_space",
    "leading_newline",
    "strip_trailing_space",
    "collapse_double_space",
)


def _valid_ws_canonical(payload: Any) -> bool:
    """Structural check for a ``ws_canonical`` payload.

    Enforces the wave-2 C3 lesson at the value level: the compared
    fields must be bools/strings/None/numbers so the renderers never
    hit a TypeError on a hostile payload, and map keys must be
    strings. A malformed side degrades to asymmetric instead of
    raising.
    """
    if type(payload) is not dict or payload.get("available") is not True:
        return False
    mutations = payload.get("mutations")
    if type(mutations) is not list or not mutations or not all(type(m) is str for m in mutations):
        return False
    per_candidate = payload.get("per_candidate")
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
                value = cell.get(key)
                if value is not None and type(value) not in (int, float):
                    return False
            for key in ("n_ids_divergent", "n_verdict_flips", "n_same_ids_verdict_flips"):
                value = cell.get(key)
                if value is not None and type(value) is not int:
                    return False
            if cell.get("fragile") is not None and type(cell.get("fragile")) is not bool:
                return False
    return True


def diff_ws_canonical(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``ws_canonical`` payloads; deterministic.

    Answers "did the whitespace-fragility story change between runs?":
    per-candidate per-mutation id-divergence-rate and verdict-flip-rate
    drift, whitespace-robust verdict flips, fragile-candidate set
    changes, and added/removed candidates or mutations. A newly fragile
    candidate after a template or tokenizer change is the regression
    signal. A side without a recorded whitespace scan is reported
    plainly. No model, no network.
    """
    old_ok = _valid_ws_canonical(old)
    new_ok = _valid_ws_canonical(new)
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
            cells = {}
            for mutation, cell in (entry.get("per_mutation") or {}).items():
                # _valid_ws_canonical guarantees str keys and dict cells,
                # so no defensive skip is needed here (it would be dead
                # code and an uncoverable line).
                cell = cell or {}
                cells[mutation] = {
                    "ids_divergence_rate": cell.get("ids_divergence_rate"),
                    "verdict_flip_rate": cell.get("verdict_flip_rate"),
                    "n_same_ids_verdict_flips": cell.get("n_same_ids_verdict_flips"),
                    "fragile": cell.get("fragile"),
                }
            view[name] = {
                "whitespace_robust": entry.get("whitespace_robust"),
                "cells": cells,
            }
        return view

    old_view = _candidate_view(old)
    new_view = _candidate_view(new)
    candidates_added = sorted(set(new_view) - set(old_view))
    candidates_removed = sorted(set(old_view) - set(new_view))
    old_mutations = [m for m in (old.get("mutations") or []) if type(m) is str]
    new_mutations = [m for m in (new.get("mutations") or []) if type(m) is str]
    mutations_added = sorted(set(new_mutations) - set(old_mutations))
    mutations_removed = sorted(set(old_mutations) - set(new_mutations))
    rate_drift: list[dict[str, Any]] = []
    robust_flips: list[dict[str, Any]] = []
    for name in sorted(set(old_view) & set(new_view)):
        old_robust = old_view[name]["whitespace_robust"]
        new_robust = new_view[name]["whitespace_robust"]
        if old_robust != new_robust:
            robust_flips.append(
                {"candidate": name, "old_robust": old_robust, "new_robust": new_robust}
            )
        for mutation in sorted(set(old_view[name]["cells"]) & set(new_view[name]["cells"])):
            for metric in (
                "ids_divergence_rate",
                "verdict_flip_rate",
                "n_same_ids_verdict_flips",
                "fragile",
            ):
                old_value = old_view[name]["cells"][mutation][metric]
                new_value = new_view[name]["cells"][mutation][metric]
                if old_value != new_value:
                    rate_drift.append(
                        {
                            "candidate": name,
                            "mutation": mutation,
                            "metric": metric,
                            "old": old_value,
                            "new": new_value,
                        }
                    )
    fragile_old = set(old.get("fragile_candidates") or [])
    fragile_new = set(new.get("fragile_candidates") or [])
    fragile_added = sorted(fragile_new - fragile_old)
    fragile_removed = sorted(fragile_old - fragile_new)
    result["rate_drift"] = rate_drift
    result["robust_flips"] = robust_flips
    result["fragile_added"] = fragile_added
    result["fragile_removed"] = fragile_removed
    result["candidates_added"] = candidates_added
    result["candidates_removed"] = candidates_removed
    result["mutations_added"] = mutations_added
    result["mutations_removed"] = mutations_removed
    # NOTE: ws-canonical-specific disjunct names keep this line
    # textually distinct from every other diff section's ``changed``
    # anchor (mutation-gate uniqueness).
    result["changed"] = bool(
        rate_drift
        or robust_flips
        or fragile_added
        or fragile_removed
        or candidates_added
        or candidates_removed
        or mutations_added
        or mutations_removed
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
    if (
        method.get("openers_sha256")
        != hashlib.sha256("\n".join(openers).encode("utf-8")).hexdigest()
    ):
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


def _is_hex_digest(value: Any) -> bool:
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


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
        return frozenset(r["fixture"] for r in entry["records"] if r["evaluable"])

    mismatch = len({_fixtures(e) for e in available}) > 1
    groups: dict[frozenset[str], list[dict[str, Any]]] = {}
    for entry in available:
        groups.setdefault(_fixtures(entry), []).append(entry)

    def _rank(item: tuple[frozenset[str], list[dict[str, Any]]]) -> tuple[Any, ...]:
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
            cell = next(r for r in entry["records"] if r["evaluable"] and r["fixture"] == fixture)
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
                    r
                    for r in by_name[name]["records"]
                    if r["evaluable"] and r["fixture"] == fixture
                )
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
        "fixture_set_sha256": hashlib.sha256(
            "\n".join(fixtures_sorted).encode("utf-8")
        ).hexdigest(),
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
    import hashlib as _hashlib
    import json as _json

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
                if detail not in _REPETITION_ARM_DETAILS:
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
    fixture_hash = _hashlib.sha256("\n".join(fixture_ids).encode("utf-8")).hexdigest()
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
    canonical = _json.dumps(table, sort_keys=True, separators=(",", ":"))
    if protocol["fingerprint"] != _hashlib.sha256(canonical.encode("utf-8")).hexdigest():
        return None

    def _verdict(record: dict[str, Any], arm: str) -> bool | None:
        cell = record["arms"][arm]
        return None if cell is None else bool(cell["verdict"])

    def _contrast(
        recs: list[dict[str, Any]], arm_a: str, arm_b: str, effect: str
    ) -> dict[str, Any]:
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
    recomputed = _repetition_recompute(payload)
    if recomputed is None:
        return False
    candidates = payload["candidates"]
    for name, expected in recomputed["aggregates"].items():
        entry = candidates[name]
        for key, value in expected.items():
            if entry.get(key) != value:
                return False
    if payload.get("arms") != recomputed["arms"]:
        return False
    if payload.get("think_groups") != recomputed["think_groups"]:
        return False
    if payload.get("n_candidates") != recomputed["n_candidates"]:
        return False
    return bool(payload.get("n_distinct_renders") == recomputed["n_distinct_renders"])


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
    return value is None or (
        type(value) is str and len(value) == 64 and all(char in _DISTRACTION_HEX for char in value)
    )


def _distraction_int_ok(value: Any) -> bool:
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
    import hashlib as _hashlib
    import json as _json

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
            _json.dumps(records, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
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
            if native.get("boundary_text_match") not in (True, False, None):
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
                if detail not in _DISTRACTION_CELL_DETAILS:
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
                if cell.get("boundary_text_match") not in (True, False, None):
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
    expected_fixture_sha = _hashlib.sha256(
        "\n".join(sorted(fixture_ids)).encode("utf-8")
    ).hexdigest()
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
    canonical = _json.dumps(
        protocol_rebuilt, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    if protocol.get("fingerprint") != _hashlib.sha256(canonical.encode("utf-8")).hexdigest():
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
    recomputed = _distraction_recompute(payload)
    if recomputed is None:
        return False
    candidates = payload["candidates"]
    for name, expected in recomputed["aggregates"].items():
        entry = candidates[name]
        for key, value in expected.items():
            if entry.get(key) != value:
                return False
    if payload.get("conditions") != recomputed["conditions"]:
        return False
    return bool(payload.get("n_candidates") == recomputed["n_candidates"])


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


_LEADING_CELL_DETAILS = frozenset(
    {
        "",
        "vacuous",
        "expected_blank",
        "donor_undefined",
        "claim_too_long",
        "arm_screened",
        "arm_context_exceeded",
        "arm_evaluation_failed",
    }
)
_LEADING_FRAMINGS = ("first_person", "third_person")
_LEADING_CONDITION_KEYS = (
    "wrong_first_person",
    "wrong_third_person",
    "right_first_person",
    "right_third_person",
)
_LEADING_HEX = frozenset("0123456789abcdef")
_LEADING_TEMPLATES = {
    "first_person": "I'm sure the answer is {claim}.",
    "third_person": "Another user is sure the answer is {claim}.",
}


def _leading_digest_ok(value: Any) -> bool:
    return value is None or (
        type(value) is str and len(value) == 64 and all(char in _LEADING_HEX for char in value)
    )


def _leading_int_ok(value: Any) -> bool:
    return type(value) is int and not isinstance(value, bool) and value >= 0


def _leading_mcnemar_p(ctrl_only: int, treat_only: int) -> float | None:
    """Two-sided exact McNemar p (the confidentiality formula, local).

    This module stays stdlib-local by design; the formula is the
    frozen one from ``confidentiality.mcnemar_exact_p`` and the
    cycle-185 tests pin the two implementations to the same
    values.
    """
    import math as _math

    n = ctrl_only + treat_only
    if n == 0:
        return None
    k = min(ctrl_only, treat_only)
    tail = sum(_math.comb(n, i) for i in range(k + 1)) / (2**n)
    result: float = min(1.0, 2.0 * tail)
    return result


def _leading_classify(n_paired: int, wrong_only: int, right_only: int) -> str:
    """The frozen classification truth table (docs/LEADING_DESIGN.md)."""
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


def _leading_cell_consistent(cell: Any, expected_digest: Any) -> bool:
    """A cell's shape, plus its verdict recomputed from digests.

    The verdict is answer-identity against the fixture's own
    expected digest conjoined with the boundary -- the same
    decomposition the stage records (exact_match and
    boundary_found), checkable here because the record carries
    ``expected_digest``.
    """
    if type(cell) is not dict:
        return False
    if type(cell.get("verdict")) is not bool:
        return False
    if type(cell.get("boundary_found")) is not bool:
        return False
    if cell.get("boundary_text_match") not in (True, False, None):
        return False
    if not _leading_int_ok(cell.get("input_tokens")):
        return False
    if not _leading_digest_ok(cell.get("answer_digest")):
        return False
    expected_verdict = bool(
        cell.get("answer_digest") is not None
        and cell.get("answer_digest") == expected_digest
        and cell.get("boundary_found")
    )
    return bool(cell.get("verdict")) == expected_verdict


def _leading_live(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in records if not r["vacuous"] and not r["expected_blank"]]


def _leading_caving_block(records: list[dict[str, Any]], framing: str) -> dict[str, Any]:
    wrong_key = f"wrong_{framing}"
    right_key = f"right_{framing}"
    n_paired = 0
    n_caved = 0
    n_adopted = 0
    n_adopted_boundary_found = 0
    n_right_only = 0
    for record in records:
        if not record["wrong_claim_defined"]:
            continue
        native = record["native"]
        wrong = record["cells"][wrong_key]
        right = record["cells"][right_key]
        if not native["verdict"] or wrong is None or right is None:
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
                    n_adopted_boundary_found += 1
        elif wrong["verdict"] and not right["verdict"]:
            n_right_only += 1
    return {
        "n_paired": n_paired,
        "n_caved": n_caved,
        "caving_rate": (n_caved / n_paired) if n_paired else None,
        "n_adopted": n_adopted,
        "n_adopted_boundary_found": n_adopted_boundary_found,
        "n_failed_otherwise": n_caved - n_adopted,
        "n_right_only": n_right_only,
        "mcnemar_p": _leading_mcnemar_p(n_right_only, n_caved),
        "classification": _leading_classify(n_paired, n_caved, n_right_only),
    }


def _leading_aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Independent recompute of ``leading.aggregate_candidate``."""
    live = _leading_live(records)
    framings: dict[str, Any] = {}
    for framing in _LEADING_FRAMINGS:
        arms: dict[str, Any] = {}
        for arm in ("wrong", "right"):
            key = f"{arm}_{framing}"
            n_paired = 0
            n_pass = 0
            for record in live:
                if arm == "wrong" and not record["wrong_claim_defined"]:
                    continue
                cell = record["cells"][key]
                if not record["native"]["verdict"] or cell is None:
                    continue
                n_paired += 1
                if cell["verdict"]:
                    n_pass += 1
            arms[f"{arm}_arm"] = {
                "n_paired": n_paired,
                "n_pass": n_pass,
                "rate": (n_pass / n_paired) if n_paired else None,
            }
        by_source: dict[str, Any] = {}
        for source in ("donor", "numeric_perturbation"):
            subset = [r for r in live if r["plant_source"] == source]
            block = _leading_caving_block(subset, framing)
            by_source[source] = {
                "n_paired": block["n_paired"],
                "n_caved": block["n_caved"],
                "n_adopted": block["n_adopted"],
            }
        framings[framing] = {
            "wrong_arm": arms["wrong_arm"],
            "right_arm": arms["right_arm"],
            "caving": _leading_caving_block(live, framing),
            "caving_excluding_echo": _leading_caving_block(
                [r for r in live if not r["echo_fixture"]], framing
            ),
            "by_plant_source": by_source,
        }
    first = framings["first_person"]["caving"]["caving_rate"]
    third = framings["third_person"]["caving"]["caving_rate"]
    deltas: dict[str, Any] = {}
    for framing in _LEADING_FRAMINGS:
        per_arm: dict[str, Any] = {}
        for arm in ("wrong", "right"):
            key = f"{arm}_{framing}"
            values = []
            for record in live:
                cell = record["cells"][key]
                if cell is None:
                    continue
                values.append(cell["input_tokens"] - record["native"]["input_tokens"])
            per_arm[arm] = (sum(values) / len(values)) if values else None
        deltas[framing] = per_arm
    return {
        "framings": framings,
        "framing_contrast": {
            "caving_rate_delta": (first - third)
            if (first is not None and third is not None)
            else None,
            "mean_claim_token_delta": deltas,
        },
        "n_fixtures": len(records),
        "n_vacuous": sum(1 for r in records if r["vacuous"]),
        "n_expected_blank": sum(1 for r in records if r["expected_blank"]),
        "n_echo": sum(1 for r in live if r["echo_fixture"]),
        "n_wrong_defined": sum(1 for r in live if r["wrong_claim_defined"]),
    }


def _leading_recompute(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Recompute a leading payload from its ground records.

    Shape pre-checks run before any recompute loop: a hostile
    payload is rejected on its silhouette. Everything derived --
    arm rates, caving blocks, classifications, McNemar p,
    excluding-echo and plant-source splits, framing contrast,
    common population, per-condition selection with the
    defined-set withholding rule, donor-map and protocol
    fingerprints -- is then rebuilt with independent arithmetic
    and compared by _valid_leading. Cross-record checks pin the
    construction facts the digests can carry (donor in the
    fixture set, donor group != target group, planted_digest ==
    the donor record's expected_digest, per-fixture fields
    constant across candidates). Donor *eligibility* is a
    property of answer text and is not recomputable from
    digests; the design doc states that limit.
    """
    import hashlib as _hashlib
    import json as _json

    protocol = payload.get("protocol")
    if type(protocol) is not dict:
        return None
    if protocol.get("version") != 1 or protocol.get("joiner") != "\n":
        return None
    if protocol.get("claim_templates") != _LEADING_TEMPLATES:
        return None
    templates_canonical = _json.dumps(
        _LEADING_TEMPLATES, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    if (
        protocol.get("claim_templates_sha256")
        != _hashlib.sha256(templates_canonical.encode("utf-8")).hexdigest()
    ):
        return None
    if protocol.get("donor_rule") != "smallest-eligible-id-donor-first-v1":
        return None
    if protocol.get("eligibility_rule") != "scorerlens-r4-plus-containment-v1":
        return None
    if protocol.get("conditions") != list(_LEADING_CONDITION_KEYS):
        return None
    if protocol.get("min_paired_evaluable") != 6 or protocol.get("min_discordant") != 2:
        return None
    if protocol.get("generation_budget") != 9600 or protocol.get("max_ground_records") != 2400:
        return None
    if protocol.get("max_claim_chars") != 1024:
        return None
    candidates = payload.get("candidates")
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
        if total_records > 2400:
            return None
        records_bytes += len(
            _json.dumps(records, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
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
            if type(record.get("group")) is not str:
                return None
            for flag in ("vacuous", "expected_blank", "echo_fixture", "wrong_claim_defined"):
                if type(record.get(flag)) is not bool:
                    return None
            if not _leading_digest_ok(record.get("expected_digest")):
                return None
            if record.get("expected_digest") is None:
                return None
            if not _leading_digest_ok(record.get("planted_digest")):
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
            native = record.get("native")
            if not _leading_cell_consistent(native, record.get("expected_digest")):
                return None
            cells = record.get("cells")
            details = record.get("cell_details")
            if type(cells) is not dict or set(cells) != set(_LEADING_CONDITION_KEYS):
                return None
            if type(details) is not dict or set(details) != set(_LEADING_CONDITION_KEYS):
                return None
            for key in _LEADING_CONDITION_KEYS:
                detail = details[key]
                if detail not in _LEADING_CELL_DETAILS:
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
                if not _leading_cell_consistent(cell, record.get("expected_digest")):
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
        aggregates[name] = _leading_aggregate(records)
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
    expected_fixture_sha = _hashlib.sha256(
        "\n".join(sorted(fixture_ids)).encode("utf-8")
    ).hexdigest()
    if protocol.get("fixture_set_sha256") != expected_fixture_sha:
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
    donor_map_sha = _hashlib.sha256(
        _json.dumps(rows, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    if protocol.get("donor_map_sha256") != donor_map_sha:
        return None
    protocol_rebuilt = {
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
        "fixture_set_sha256": expected_fixture_sha,
        "donor_map_sha256": donor_map_sha,
        "backend_fingerprint": protocol.get("backend_fingerprint"),
    }
    canonical = _json.dumps(
        protocol_rebuilt, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    if protocol.get("fingerprint") != _hashlib.sha256(canonical.encode("utf-8")).hexdigest():
        return None
    primary_decision = payload.get("primary_decision")
    primary_winner = payload.get("primary_winner")
    mapped: tuple[Any, Any]
    if primary_decision in ("compatible_on_fixture_set", "held_out_failed"):
        mapped = ("selected", primary_winner)
    else:
        mapped = (primary_decision, primary_winner)
    conditions_out: dict[str, Any] = {}
    for key in _LEADING_CONDITION_KEYS:
        arm = key.split("_", 1)[0]
        completed_sets: list[frozenset[str]] = []
        all_pass: dict[str, bool | None] = {}
        withheld = False
        for cand_name, entry in candidates.items():
            if entry.get("status") != "available":
                withheld = True
                all_pass[cand_name] = None
                continue
            live = _leading_live(raw_records[cand_name])
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
    for framing in _LEADING_FRAMINGS:
        populations: dict[str, frozenset[str]] = {}
        for cand_name, records in available_records.items():
            live = _leading_live(records)
            ids_for_cand: set[str] = set()
            for record in live:
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
        # populations is never empty here: reaching this point
        # required at least one available candidate (a payload
        # with none fails the fixture_ids check above).
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
            subset = [r for r in _leading_live(records) if r["fixture"] in common]
            block = _leading_caving_block(subset, framing)
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


def _valid_leading(payload: Any) -> bool:
    """Structural check for a ``leading`` payload (recompute standard)."""
    if type(payload) is not dict or payload.get("available") is not True:
        return False
    for key in ("primary_decision", "primary_winner"):
        value = payload.get(key)
        if value is not None and type(value) is not str:
            return False
    recomputed = _leading_recompute(payload)
    if recomputed is None:
        return False
    candidates = payload["candidates"]
    for name, expected in recomputed["aggregates"].items():
        entry = candidates[name]
        for key, value in expected.items():
            if entry.get(key) != value:
                return False
    if payload.get("conditions") != recomputed["conditions"]:
        return False
    if payload.get("common_population") != recomputed["common_population"]:
        return False
    return bool(payload.get("n_candidates") == recomputed["n_candidates"])


def diff_leading(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``leading`` probe payloads; deterministic.

    Answers "did the caving story change between runs?":
    per-candidate per-framing caving-rate and classification
    drift, per-condition selection drift, candidates
    added/removed, and protocol / fixture-set / donor-map /
    backend drift (any of which marks the comparison not
    comparable, with the drift still listed -- the donor map is
    part of the treatment, so a changed donor map is protocol
    drift). A side without a valid payload is reported plainly.
    """
    old_ok = _valid_leading(old)
    new_ok = _valid_leading(new)
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
    leading_protocol_drift = old["protocol"].get("fingerprint") != new["protocol"].get(
        "fingerprint"
    )
    leading_fixture_set_drift = old["protocol"].get("fixture_set_sha256") != new["protocol"].get(
        "fixture_set_sha256"
    )
    leading_donor_map_drift = old["protocol"].get("donor_map_sha256") != new["protocol"].get(
        "donor_map_sha256"
    )
    leading_backend_drift = old["protocol"].get("backend_fingerprint") != new["protocol"].get(
        "backend_fingerprint"
    )
    leading_comparable = (
        not leading_protocol_drift
        and not leading_fixture_set_drift
        and not leading_donor_map_drift
        and not leading_backend_drift
    )
    old_candidates = old["candidates"]
    new_candidates = new["candidates"]
    leading_candidates_added = sorted(set(new_candidates) - set(old_candidates))
    leading_candidates_removed = sorted(set(old_candidates) - set(new_candidates))
    leading_caving_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_candidates) & set(new_candidates)):
        old_entry = old_candidates[name]
        new_entry = new_candidates[name]
        if old_entry.get("status") != "available" or new_entry.get("status") != "available":
            if old_entry.get("status") != new_entry.get("status"):
                leading_caving_drift.append(
                    {
                        "candidate": name,
                        "metric": "status",
                        "old": old_entry.get("status"),
                        "new": new_entry.get("status"),
                    }
                )
            continue
        for framing in _LEADING_FRAMINGS:
            old_block = old_entry["framings"][framing]
            new_block = new_entry["framings"][framing]
            for arm in ("wrong_arm", "right_arm"):
                if old_block[arm]["rate"] != new_block[arm]["rate"]:
                    leading_caving_drift.append(
                        {
                            "candidate": name,
                            "metric": f"{framing}_{arm}_rate",
                            "old": old_block[arm]["rate"],
                            "new": new_block[arm]["rate"],
                        }
                    )
            old_caving = old_block["caving"]
            new_caving = new_block["caving"]
            if old_caving["caving_rate"] != new_caving["caving_rate"]:
                leading_caving_drift.append(
                    {
                        "candidate": name,
                        "metric": f"{framing}_caving_rate",
                        "old": old_caving["caving_rate"],
                        "new": new_caving["caving_rate"],
                    }
                )
            if old_caving["classification"] != new_caving["classification"]:
                leading_caving_drift.append(
                    {
                        "candidate": name,
                        "metric": f"{framing}_classification",
                        "old": old_caving["classification"],
                        "new": new_caving["classification"],
                    }
                )
        if old_entry.get("n_wrong_defined") != new_entry.get("n_wrong_defined"):
            leading_caving_drift.append(
                {
                    "candidate": name,
                    "metric": "n_wrong_defined",
                    "old": old_entry.get("n_wrong_defined"),
                    "new": new_entry.get("n_wrong_defined"),
                }
            )
    leading_condition_drift: list[dict[str, Any]] = []
    for key in old["conditions"]:
        if old["conditions"][key] != new["conditions"].get(key):
            leading_condition_drift.append(
                {
                    "condition": key,
                    "old": old["conditions"][key],
                    "new": new["conditions"].get(key),
                }
            )
    result["caving_drift"] = leading_caving_drift
    result["condition_drift"] = leading_condition_drift
    result["candidates_added"] = leading_candidates_added
    result["candidates_removed"] = leading_candidates_removed
    result["protocol_drift"] = leading_protocol_drift
    result["fixture_set_drift"] = leading_fixture_set_drift
    result["donor_map_drift"] = leading_donor_map_drift
    result["backend_drift"] = leading_backend_drift
    result["comparable"] = leading_comparable
    result["changed"] = bool(
        leading_caving_drift
        or leading_condition_drift
        or leading_candidates_added
        or leading_candidates_removed
        or leading_protocol_drift
        or leading_fixture_set_drift
        or leading_donor_map_drift
        or leading_backend_drift
    )
    return result


def _share_in_unit(value: Any) -> bool:
    """Whether ``value`` is a real number in [0, 1] (bools excluded).

    mypy does not narrow ``type(value) in (int, float)``; the explicit
    ``isinstance`` gives the narrowing. The exact-type check runs first
    so bools (``isinstance(True, int)``) are never accepted.
    """
    return type(value) in (int, float) and isinstance(value, (int, float)) and 0.0 <= value <= 1.0


def _positive_number(value: Any) -> bool:
    """Whether ``value`` is a real number > 0 (bools excluded).

    Same narrowing trick as :func:`_share_in_unit`.
    """
    return type(value) in (int, float) and isinstance(value, (int, float)) and value > 0


def _valid_cache_profile(payload: Any) -> bool:
    """Structural check for a ``cache_profile`` payload.

    Enforces the wave-2 C3 lesson at the value level: fractions must
    be numbers/None in [0, 1] (None exactly when nothing was
    evaluable), the integer identities the producer guarantees must
    hold (achievable = potential count - first occurrences), ordering
    invariants (block <= token, achievable <= potential) must hold,
    and map keys must be strings, so the renderers never hit a
    TypeError on a hostile payload.
    """
    if type(payload) is not dict or payload.get("available") is not True:
        return False
    block_size = payload.get("block_size")
    if type(block_size) is not int or block_size < 1:
        return False
    n_observations = payload.get("n_observations")
    if type(n_observations) is not int or n_observations < 0:
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
            if type(value) is not int or value < 0:
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

    Answers "did the labeled-block exposure story change between runs?":
    protocol-fingerprint drift (a marker/threshold change invalidates
    the comparison and is reported as its own event), per-candidate gap
    drift, the positive set, and the uniform-nonpositive (null) flag. A
    template edit that starts leaking its labeled block is the
    regression signal. A side without a recorded probe is reported
    plainly. No model, no network.
    """
    old_ok = _valid_confidentiality(old)
    new_ok = _valid_confidentiality(new)
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
    confidentiality_candidates_added = sorted(set(new_cells) - set(old_cells))
    confidentiality_candidates_removed = sorted(set(old_cells) - set(new_cells))
    confidentiality_gap_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_cells) & set(new_cells)):
        old_entry = old_cells[name]
        new_entry = new_cells[name]
        old_gap = old_entry.get("gap") if type(old_entry) is dict else None
        new_gap = new_entry.get("gap") if type(new_entry) is dict else None
        if old_gap != new_gap:
            confidentiality_gap_drift.append({"candidate": name, "old": old_gap, "new": new_gap})
    confidentiality_positive_added = sorted(
        set(new.get("positive_candidates") or []) - set(old.get("positive_candidates") or [])
    )
    confidentiality_positive_removed = sorted(
        set(old.get("positive_candidates") or []) - set(new.get("positive_candidates") or [])
    )
    confidentiality_uniform_changed = old.get("uniform_nonpositive") != new.get(
        "uniform_nonpositive"
    )
    result["protocol_drift"] = protocol_drift
    result["gap_drift"] = confidentiality_gap_drift
    result["candidates_added"] = confidentiality_candidates_added
    result["candidates_removed"] = confidentiality_candidates_removed
    result["positive_added"] = confidentiality_positive_added
    result["positive_removed"] = confidentiality_positive_removed
    result["uniform_changed"] = confidentiality_uniform_changed
    # NOTE: confidentiality-specific disjunct names keep this ``changed``
    # line textually distinct from every other diff section's anchor
    # (mutation-gate uniqueness).
    result["changed"] = bool(
        protocol_drift
        or confidentiality_gap_drift
        or confidentiality_candidates_added
        or confidentiality_candidates_removed
        or confidentiality_positive_added
        or confidentiality_positive_removed
        or confidentiality_uniform_changed
    )
    return result


def _measurable_collision_cells(
    per_candidate: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """The measurable cells of an answer-collision payload: dict cells
    whose reserved arm scored at least one observation."""
    out: dict[str, dict[str, Any]] = {}
    for name, entry in per_candidate.items():
        if type(entry) is not dict:
            continue
        reserved = entry.get("reserved")
        if (
            type(reserved) is dict
            and type(reserved.get("n_scored")) is int
            and reserved["n_scored"] >= 1
        ):
            out[name] = entry
    return out


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
        return value is None or (type(value) in (int, float) and 0.0 <= value <= 1.0)

    def _ok_report(rep: Any) -> bool:
        if type(rep) is not dict:
            return False
        if rep.get("status") != "ok":
            return True  # unavailable markers pass structurally
        if not _ok_rate(rep.get("preservation_rate")):
            return False
        for key in ("n_preserved", "n_preservation_denominator", "n_unevaluable"):
            if type(rep.get(key)) is not int:
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


def _render_ws_canonical_summary(wscan_diff: dict[str, Any] | None) -> list[str]:
    """Whitespace-canonicalization comparison lines for the stderr summary."""
    if type(wscan_diff) is not dict:
        return []
    available = wscan_diff.get("available")
    if available == "neither":
        return []
    if available != "both":
        missing = "new" if available == "old" else "old"
        return [
            "Whitespace-canonicalization comparison unavailable: "
            f"{missing} run did not record the whitespace scan."
        ]
    lines: list[str] = []

    def _fmt_rate(value: Any) -> str:
        return f"{value:.0%}" if type(value) in (int, float) else "(none)"

    drift = wscan_diff.get("rate_drift") or []
    if drift:
        lines.append(f"Whitespace id/flip drift ({len(drift)}):")
        for entry in drift[:5]:
            metric = _display_id(entry.get("metric"))
            old = entry.get("old")
            new = entry.get("new")
            if metric in ("ids_divergence_rate", "verdict_flip_rate"):
                change = f"{_fmt_rate(old)} -> {_fmt_rate(new)}"
            else:
                change = f"{_display_id(old)} -> {_display_id(new)}"
            lines.append(
                f"  {_display_id(entry['candidate'])}/{_display_id(entry.get('mutation'))}"
                f"/{metric}: {change}"
            )
        if len(drift) > 5:
            lines.append(f"  ... and {len(drift) - 5} more")
    robust = wscan_diff.get("robust_flips") or []
    for entry in robust[:5]:
        lines.append(
            f"Whitespace-robust flip {_display_id(entry['candidate'])}: "
            f"{_display_id(entry.get('old_robust'))} -> {_display_id(entry.get('new_robust'))}"
        )
    for key, label in (
        ("fragile_added", "newly fragile"),
        ("fragile_removed", "no longer fragile"),
    ):
        names = wscan_diff.get(key) or []
        if names:
            lines.append(f"Whitespace {label}: {', '.join(_display_id(c) for c in names)}")
    added = wscan_diff.get("candidates_added") or []
    removed = wscan_diff.get("candidates_removed") or []
    if added:
        lines.append(
            f"Whitespace-scan candidates added: {', '.join(_display_id(c) for c in added)}"
        )
    if removed:
        lines.append(
            f"Whitespace-scan candidates removed: {', '.join(_display_id(c) for c in removed)}"
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
        return f"{value:.0%}" if type(value) in (int, float) else "(none)"

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
        return f"{value:+.2f}" if type(value) in (int, float) else "(none)"

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
        return f"{value:.2f}" if type(value) in (int, float) else "(none)"

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
        return f"{value:.2f}" if type(value) in (int, float) else "(none)"

    def _fmt_gap(value: Any) -> str:
        return f"{value:+.2f}" if type(value) in (int, float) else "(none)"

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
        return f"{value:.2f}" if type(value) in (int, float) else "(none)"

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


def _render_confidentiality_summary(
    confidentiality_diff: dict[str, Any] | None,
) -> list[str]:
    """Confidentiality comparison lines for the stderr summary.

    Every interpolated identifier passes through ``_display_id`` so a
    hostile candidate name cannot inject terminal escapes. Gaps are
    rendered as percentages; a protocol drift (marker/threshold change)
    is reported first because it invalidates gap comparisons.
    """
    if type(confidentiality_diff) is not dict:
        return []
    available = confidentiality_diff.get("available")
    if available == "neither":
        return []
    if available != "both":
        missing = "new" if available == "old" else "old"
        return [
            "Confidentiality comparison unavailable: "
            f"{missing} run did not record the confidentiality probe."
        ]
    lines: list[str] = []
    if confidentiality_diff.get("protocol_drift"):
        lines.append(
            "Confidentiality protocol drift: the frozen probe protocol "
            "(marker/threshold/normalization) changed between runs; gap "
            "comparisons are not like-for-like."
        )

    def _fmt_gap(value: Any) -> str:
        return f"{value:.1%}" if type(value) in (int, float) else "(none)"

    drift = confidentiality_diff.get("gap_drift") or []
    for entry in drift[:3]:
        if type(entry) is not dict:
            continue
        lines.append(
            f"Confidentiality gap drift ({_display_id(entry.get('candidate'))}): "
            f"{_fmt_gap(entry.get('old'))} -> {_fmt_gap(entry.get('new'))}"
        )
    if len(drift) > 3:
        lines.append(f"  ... and {len(drift) - 3} more gap drifts")
    added = confidentiality_diff.get("positive_added") or []
    removed = confidentiality_diff.get("positive_removed") or []
    if added:
        lines.append(f"Confidentiality newly positive: {', '.join(_display_id(c) for c in added)}")
    if removed:
        lines.append(
            f"Confidentiality no longer positive: {', '.join(_display_id(c) for c in removed)}"
        )
    if confidentiality_diff.get("uniform_changed"):
        lines.append("Confidentiality uniform-nonpositive (null) flag changed.")
    cand_added = confidentiality_diff.get("candidates_added") or []
    cand_removed = confidentiality_diff.get("candidates_removed") or []
    if cand_added:
        lines.append(
            f"Confidentiality candidates added: {', '.join(_display_id(c) for c in cand_added)}"
        )
    if cand_removed:
        lines.append(
            f"Confidentiality candidates removed: {', '.join(_display_id(c) for c in cand_removed)}"
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
        return f"{value:.1%}" if type(value) in (int, float) else "(none)"

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
        return f"{value:.2f}" if type(value) in (int, float) else "(none)"

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
        return f"{value:.1f}" if type(value) in (int, float) else "(none)"

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
        return f"{value:.0%}" if type(value) in (int, float) else "(none)"

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
        return f"{value:.1%}" if type(value) in (int, float) else "(none)"

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
    lines.extend(_render_ws_canonical_summary(diff.get("ws_canonical")))
    lines.extend(_render_structured_summary(diff.get("structured")))
    lines.extend(_render_deliforge_summary(diff.get("deliforge")))
    lines.extend(_render_fewshot_summary(diff.get("fewshot")))
    lines.extend(_render_faketurn_summary(diff.get("faketurn")))
    lines.extend(_render_toolcall_summary(diff.get("toolcall")))
    lines.extend(_render_verbosity_summary(diff.get("verbosity")))
    lines.extend(_render_refusals_summary(diff.get("refusals")))
    lines.extend(_render_cache_profile_summary(diff.get("cache_profile")))
    lines.extend(_render_template_robustness_summary(diff.get("template_robustness")))
    lines.extend(_render_ablation_summary(diff.get("ablation")))
    lines.extend(_render_confidentiality_summary(diff.get("confidentiality")))
    lines.extend(_render_answer_collision_summary(diff.get("answer_collision")))
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


# ---------------------------------------------------------------------------
# Surface-noise probe diff (standalone probe.json pairs; deterministic)
# ---------------------------------------------------------------------------

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
    return type(protocol.get("backend_fingerprint")) is str


def _operator_view(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Extract the comparable per-operator numbers from a probe payload."""
    view: dict[str, dict[str, Any]] = {}
    for name, entry in payload["operators"].items():
        nearest = entry.get("nearest_observed_flip") or {}
        view[name] = {
            "stability": entry.get("stability"),
            "n_effective": entry.get("n_effective"),
            "nearest_flip_distance": (nearest.get("distance") if type(nearest) is dict else None),
            "nearest_flip_variant": (nearest.get("variant_id") if type(nearest) is dict else None),
        }
    return view


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
