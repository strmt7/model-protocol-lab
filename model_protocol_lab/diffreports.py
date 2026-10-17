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


def _display_id(value: str) -> str:
    """Operator-visible names/IDs with control characters stripped.

    Fixture IDs and candidate names arrive from the compared files; a
    hostile results.json must not inject terminal escape sequences into
    stderr output. (Mirrors ``cli._display_id``; kept local so this module
    does not import the CLI.)
    """
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
