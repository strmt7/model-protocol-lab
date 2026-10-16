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
from pathlib import Path
from typing import Any

# results.json with --include-evidence carries per-fixture token-ID arrays;
# 32 MiB comfortably covers the largest runs this lab produces (100 search +
# 100 held-out fixtures at the 4096-token cap) while keeping a hostile file
# from exhausting memory.
MAX_RESULTS_BYTES = 32 * 1024 * 1024

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f]")


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
    identical = (
        not decision_changed
        and not winner_changed
        and not verdict_changes
        and not changed_fixture_diffs
        and not config_drift
        and not evidence_asymmetric
    )
    summary = {
        "decision_changed": decision_changed,
        "winner_changed": winner_changed,
        "candidates_changed_status": len(changed_candidates),
        "fixtures_flipped": len(flips),
        "config_drift_warnings": len(config_drift),
        "evidence_asymmetric": evidence_asymmetric,
        "identical": identical,
    }
    return {
        "tool": "diff-reports",
        "old": {"source": old_label},
        "new": {"source": new_label},
        "decision": {"old": old_decision, "new": new_decision, "changed": decision_changed},
        "winner": {"old": old_winner, "new": new_winner, "changed": winner_changed},
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


def _compact(value: Any) -> str:
    """One-line JSON for a drift value in the stderr summary."""
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


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
        lines.append(
            "No differences: the two runs are identical on every compared axis "
            f"({', '.join(axes)})."
        )
    else:
        lines.append(
            "Summary: "
            f"decision {'CHANGED' if summary['decision_changed'] else 'unchanged'}, "
            f"winner {'CHANGED' if summary['winner_changed'] else 'unchanged'}, "
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
