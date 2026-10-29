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
import math
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
    cleaned = _CONTROL_CHARS.sub("", value)
    if len(cleaned) > 200:
        # The R4-7 200-character preview cap (oversight security
        # S-9): display only — never a comparison input.
        return cleaned[:200] + "..."
    return cleaned


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("results_duplicate_field")
        result[key] = value
    return result


#: The run-level decision vocabulary a probe section's
#: ``primary_decision`` can legitimately carry (the values
#: run_experiment's assembly produces; reporting._DECISION_CLASS
#: renders exactly these). The first two are the winner-bearing
#: class: the section mapping rewrites them to "selected".
_PRIMARY_DECISIONS = frozenset(
    {
        "compatible_on_fixture_set",
        "held_out_failed",
        "ambiguous",
        "no_candidate",
        "cancelled",
        "failed",
    }
)
_PRIMARY_WINNER_DECISIONS = frozenset({"compatible_on_fixture_set", "held_out_failed"})


def _valid_primary_pair(payload: dict[str, Any]) -> bool:
    """Bind a section's ``primary_decision``/``primary_winner`` pair.

    The recompute validators map the pair into per-arm selections
    and compare those, but the pair itself used to be only
    type-checked: any string in the same equality class as the arm
    outcomes validated, so the label could contradict its own
    section (oversight edge-case F3). The binding is: the decision
    is in the run-decision vocabulary; the winner is one of the
    payload's own candidate names exactly when the decision is in
    the winner-bearing class, and None otherwise. What this does
    NOT do: recompute the parent run's decision — the section does
    not carry the parent run's fixture outcomes, so within the
    vocabulary the decision remains stage-attested (documented in
    docs/DATELINE_DESIGN.md). The decision is type-checked before
    the vocabulary membership test, so an unhashable hostile value
    (list/dict) fails closed instead of raising — the verify.py
    A-F1 pattern, whose latent sibling this was until cycle 216
    (the precision and confidence recompute paths reach this
    function without the pre-guard the other four callers carry)."""
    decision = payload.get("primary_decision")
    winner = payload.get("primary_winner")
    if type(decision) is not str or decision not in _PRIMARY_DECISIONS:
        return False
    candidates = payload.get("candidates")
    if decision in _PRIMARY_WINNER_DECISIONS:
        return type(winner) is str and type(candidates) is dict and winner in candidates
    return winner is None


def _recomputed_equal(stored: Any, expected: Any) -> bool:
    """Type-strict equality for validator recompute comparisons.

    Bare ``==`` lets a JSON ``true`` validate as the integer 1 (bool
    is an int subclass), so an equality-compared count or version in
    a hostile payload could slip past a recompute check in the wrong
    type (oversight edge-case F2 — behaviourally inert, since the
    value is only ever compared, but a strictness wart in validators
    whose whole contract is exactness). Ints and bools must match in
    type AND value; dicts and lists recurse; everything else uses
    plain equality."""
    if type(expected) is bool:
        return type(stored) is bool and stored == expected
    if type(expected) is int:
        return type(stored) is int and stored == expected
    if type(expected) is dict:
        return (
            type(stored) is dict
            and stored.keys() == expected.keys()
            and all(_recomputed_equal(stored[key], value) for key, value in expected.items())
        )
    if type(expected) is list:
        return (
            type(stored) is list
            and len(stored) == len(expected)
            and all(
                _recomputed_equal(item, value) for item, value in zip(stored, expected, strict=True)
            )
        )
    return bool(stored == expected)


def _reject_constant(_value: Any) -> Any:
    raise ValueError("results_nonfinite_number")


def _reject_nonfinite_float(text: str) -> float:
    """Reject float literals that overflow to infinity (``1e999``).

    ``parse_constant`` only fires for the ``NaN``/``Infinity``
    literals; ``float()`` conversion of a huge exponent reaches
    ``inf`` without ever calling it. Before cycle 202 the results
    loader had no ``parse_float`` hook (``toolcall`` had one since its
    own audit), so ``1e999`` entered a payload as ``inf`` and
    re-serialised as ``Infinity`` — output this same loader rejects.
    """
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("results_nonfinite_number")
    return value


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
            parse_float=_reject_nonfinite_float,
        )
    except RecursionError:
        # Deeply nested input within the byte cap exhausts the parser stack.
        raise ValueError("results_nesting_too_deep") from None
    except json.JSONDecodeError:
        raise ValueError("results_invalid_json") from None
    except ValueError as exc:
        # The hooks above raise this module's own fixed codes
        # (results_duplicate_field, results_nonfinite_number); those
        # pass through untouched. The one remaining ValueError the
        # parser raises is the interpreter's integer-length limit on
        # >4300-digit literals, which is a plain ValueError rather
        # than a JSONDecodeError — it is a syntax-class rejection and
        # maps to the same code (cycle 202; the pack manifest boundary
        # already converted this class).
        if str(exc) in ("results_duplicate_field", "results_nonfinite_number"):
            raise
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
        fd = os.open(resolved, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
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
    numform_diff = diff_numform(old.get("numform"), new.get("numform"))
    dateline_diff = diff_dateline(old.get("dateline"), new.get("dateline"))
    multiturn_diff = diff_multiturn(old.get("multiturn"), new.get("multiturn"))
    needle_diff = diff_needle(old.get("needle"), new.get("needle"))
    placement_diff = diff_placement(old.get("placement"), new.get("placement"))
    canonical_form_diff = diff_canonicalform(old.get("canonical_form"), new.get("canonical_form"))
    instruction_distance_diff = diff_instruction_distance(
        old.get("instruction_distance"), new.get("instruction_distance")
    )
    history_frame_diff = diff_history_frame(old.get("history_frame"), new.get("history_frame"))
    choiceaudit_diff = diff_choiceaudit(old.get("choice_audit"), new.get("choice_audit"))
    precision_diff = diff_precision(old.get("precision"), new.get("precision"))
    confidence_diff = diff_confidence(old.get("confidence"), new.get("confidence"))
    rotation_diff = diff_rotation(old.get("rotation"), new.get("rotation"))
    density_diff = diff_density(old.get("density"), new.get("density"))
    field_order_diff = diff_field_order(old.get("field_order"), new.get("field_order"))
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
    numform_asymmetric = numform_diff["available"] not in (
        "both",
        "neither",
    )
    dateline_asymmetric = dateline_diff["available"] not in (
        "both",
        "neither",
    )
    multiturn_asymmetric = multiturn_diff["available"] not in (
        "both",
        "neither",
    )
    needle_asymmetric = needle_diff["available"] not in (
        "both",
        "neither",
    )
    placement_asymmetric = placement_diff["available"] not in (
        "both",
        "neither",
    )
    canonical_form_asymmetric = canonical_form_diff["available"] not in (
        "both",
        "neither",
    )
    instruction_distance_asymmetric = instruction_distance_diff["available"] not in (
        "both",
        "neither",
    )
    history_frame_asymmetric = history_frame_diff["available"] not in (
        "both",
        "neither",
    )
    choiceaudit_asymmetric = choiceaudit_diff["available"] not in (
        "both",
        "neither",
    )
    precision_asymmetric = precision_diff["available"] not in (
        "both",
        "neither",
    )
    confidence_asymmetric = confidence_diff["available"] not in (
        "both",
        "neither",
    )
    rotation_asymmetric = rotation_diff["available"] not in (
        "both",
        "neither",
    )
    density_asymmetric = density_diff["available"] not in (
        "both",
        "neither",
    )
    field_order_asymmetric = field_order_diff["available"] not in (
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
        and not numform_diff["changed"]
        and not numform_asymmetric
        and not dateline_diff["changed"]
        and not dateline_asymmetric
        and not multiturn_diff["changed"]
        and not multiturn_asymmetric
        and not needle_diff["changed"]
        and not needle_asymmetric
        and not placement_diff["changed"]
        and not placement_asymmetric
        and not canonical_form_diff["changed"]
        and not canonical_form_asymmetric
        and not instruction_distance_diff["changed"]
        and not instruction_distance_asymmetric
        and not history_frame_diff["changed"]
        and not history_frame_asymmetric
        and not choiceaudit_diff["changed"]
        and not choiceaudit_asymmetric
        and not precision_diff["changed"]
        and not precision_asymmetric
        and not confidence_diff["changed"]
        and not confidence_asymmetric
        and not rotation_diff["changed"]
        and not rotation_asymmetric
        and not density_diff["changed"]
        and not density_asymmetric
        and not field_order_diff["changed"]
        and not field_order_asymmetric
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
        "numform_changed": numform_diff["changed"],
        "numform_asymmetric": numform_asymmetric,
        "dateline_changed": dateline_diff["changed"],
        "dateline_asymmetric": dateline_asymmetric,
        "multiturn_changed": multiturn_diff["changed"],
        "multiturn_asymmetric": multiturn_asymmetric,
        "needle_changed": needle_diff["changed"],
        "needle_asymmetric": needle_asymmetric,
        "placement_changed": placement_diff["changed"],
        "placement_asymmetric": placement_asymmetric,
        "canonical_form_changed": canonical_form_diff["changed"],
        "canonical_form_asymmetric": canonical_form_asymmetric,
        "instruction_distance_changed": instruction_distance_diff["changed"],
        "instruction_distance_asymmetric": instruction_distance_asymmetric,
        "history_frame_changed": history_frame_diff["changed"],
        "history_frame_asymmetric": history_frame_asymmetric,
        "choice_audit_changed": choiceaudit_diff["changed"],
        "choice_audit_asymmetric": choiceaudit_asymmetric,
        "precision_changed": precision_diff["changed"],
        "precision_asymmetric": precision_asymmetric,
        "confidence_changed": confidence_diff["changed"],
        "confidence_asymmetric": confidence_asymmetric,
        "rotation_changed": rotation_diff["changed"],
        "rotation_asymmetric": rotation_asymmetric,
        "density_changed": density_diff["changed"],
        "density_asymmetric": density_asymmetric,
        "field_order_changed": field_order_diff["changed"],
        "field_order_asymmetric": field_order_asymmetric,
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
        "numform": numform_diff,
        "dateline": dateline_diff,
        "multiturn": multiturn_diff,
        "needle": needle_diff,
        "placement": placement_diff,
        "canonical_form": canonical_form_diff,
        "instruction_distance": instruction_distance_diff,
        "history_frame": history_frame_diff,
        "choice_audit": choiceaudit_diff,
        "precision": precision_diff,
        "confidence": confidence_diff,
        "rotation": rotation_diff,
        "density": density_diff,
        "field_order": field_order_diff,
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
    if not _recomputed_equal(protocol.get("version"), 1) or protocol.get("joiner") != "\n":
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
    if not _valid_primary_pair(payload):
        return False
    recomputed = _leading_recompute(payload)
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
    if not _recomputed_equal(payload.get("common_population"), recomputed["common_population"]):
        return False
    return _recomputed_equal(payload.get("n_candidates"), recomputed["n_candidates"])


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


_NUMFORM_TASKS = ("copy", "compare", "add")
_NUMFORM_FORMS = ("plain", "comma", "space", "scientific")
_NUMFORM_BANDS = ((1, 3), (4, 6), (7, 9), (10, 12))
_NUMFORM_SPACE = "\u202f"
_NUMFORM_TEMPLATES = {
    "copy": "Copy this number exactly, character for character:\n{operand}",
    "compare": (
        "Which of these two numbers is larger? Reply with only the word first or second.\n{a}\n{b}"
    ),
    "add": (
        "Add these two numbers and reply with only the sum, "
        "written in the same format as the numbers:\n{a}\n{b}"
    ),
}
_NUMFORM_DETAILS = (
    "",
    "arm_context_exceeded",
    "answer_budget_exceeded",
    "form_not_in_render",
    "arm_evaluation_failed",
)
_NUMFORM_RECORD_KEYS = {
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
_NUMFORM_CELL_KEYS = {
    "verdict",
    "boundary_found",
    "boundary_text_match",
    "input_tokens",
    "answer_digest",
}


def _numform_grouped(digits: str, separator: str) -> str:
    parts: list[str] = []
    while digits:
        parts.append(digits[-3:])
        digits = digits[:-3]
    return separator.join(reversed(parts))


def _numform_render(value: int, form: str) -> str:
    """Independent reimplementation of the numform renderers."""
    digits = str(value)
    if form == "plain":
        return digits
    if form == "comma":
        return _numform_grouped(digits, ",")
    if form == "space":
        return _numform_grouped(digits, _NUMFORM_SPACE)
    if value == 0:
        return "0e0"
    exponent = len(digits) - 1
    mantissa = digits.rstrip("0")
    if len(mantissa) > 1:
        return f"{mantissa[0]}.{mantissa[1:]}e{exponent}"
    return f"{mantissa}e{exponent}"


_NUMFORM_PLAIN_RE = re.compile(r"[0-9]+")
_NUMFORM_COMMA_RE = re.compile(r"[0-9]{1,3}(?:,[0-9]{3})*")
_NUMFORM_SPACE_RE = re.compile(r"[0-9]{1,3}(?:\u202f[0-9]{3})*")
_NUMFORM_SCI_RE = re.compile(r"([0-9]+)(?:\.([0-9]+))?[eE]\+?([0-9]{1,3})")
_NUMFORM_MAX_PARSE_DIGITS = 20


def _numform_parse_value(text: Any) -> int | None:
    """Independent mirror of numform.parse_value (cycle 190).

    The grammars are deliberately [0-9], never \\d: int() and
    \\d both accept non-ASCII decimal digits, the producer
    accepts neither. Caps are digit counts checked before
    conversion, and the scientific branch keeps the
    producer's shift arithmetic exactly (a negative shift
    must divide the mantissa, and results are uncapped).
    The differential battery in tests/test_cycle190_numform_f3.py
    pins this mirror against the producer over the grammar
    alphabet; the pattern texts are pinned equal by test.
    """
    if type(text) is not str or not text:
        return None
    if _NUMFORM_PLAIN_RE.fullmatch(text):
        if len(text) > _NUMFORM_MAX_PARSE_DIGITS:
            return None
        return int(text)
    if _NUMFORM_COMMA_RE.fullmatch(text):
        digits = text.replace(",", "")
        if len(digits) > _NUMFORM_MAX_PARSE_DIGITS:
            return None
        return int(digits)
    if _NUMFORM_SPACE_RE.fullmatch(text):
        digits = text.replace(_NUMFORM_SPACE, "")
        if len(digits) > _NUMFORM_MAX_PARSE_DIGITS:
            return None
        return int(digits)
    match = _NUMFORM_SCI_RE.fullmatch(text)
    if match is not None:
        head, frac, exponent_text = match.group(1), match.group(2) or "", match.group(3)
        if len(head) + len(frac) > _NUMFORM_MAX_PARSE_DIGITS:
            return None
        exponent = int(exponent_text)
        mantissa = int(head + frac)
        shift = exponent - len(frac)
        if shift >= 0:
            result: int = mantissa * 10**shift
            return result
        divisor = 10 ** (-shift)
        if mantissa % divisor:
            return None
        quotient: int = mantissa // divisor
        return quotient
    return None


def _numform_text_digest(text: str) -> str | None:
    """SHA-256 of the UTF-8 encoding, or None when the text
    cannot be encoded (a lone-surrogate answer_text is a
    validation failure, never an exception — cycle 190 A3)."""
    try:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    except UnicodeEncodeError:
        return None


def _numform_carry(a: int, b: int) -> str:
    digits_a = str(a)
    digits_b = str(b)
    width = max(len(digits_a), len(digits_b))
    carry_in = 0
    carries: list[bool] = []
    for pos in range(width):
        digit_a = int(digits_a[-1 - pos]) if pos < len(digits_a) else 0
        digit_b = int(digits_b[-1 - pos]) if pos < len(digits_b) else 0
        carry_in = 1 if digit_a + digit_b + carry_in >= 10 else 0
        carries.append(bool(carry_in))
    if not any(carries):
        return "none"
    if all(carries):
        return "full_chain"
    return "some"


def _numform_margin(a: int, b: int) -> str:
    digits_a = str(a)
    digits_b = str(b)
    if len(digits_a) != len(digits_b):
        return "different_length"
    for index in range(len(digits_a)):
        if digits_a[index] != digits_b[index]:
            if index == len(digits_a) - 1:
                return "same_length_last_digit"
            if index * 2 < len(digits_a):
                return "same_length_early"
            return "same_length_middle"
    return "equal"


def _numform_is_round(value: int) -> bool:
    if value < 10:
        return False
    digits = str(value)
    result: bool = value == 10 ** (len(digits) - 1) or value == 10 ** len(digits) - 1
    return result


# ---------------------------------------------------------------------------
# The generator, mirrored. The validator regenerates the value
# set from the protocol's seed with its own copy of the frozen
# derivation (never an import from numform.py -- independence is
# the point of this module) and compares every record's
# construction against the regenerated set. A payload whose
# records are not the seed's set is not this probe's payload,
# however self-consistent its aggregates are.
# ---------------------------------------------------------------------------


def _numform_derive(
    seed: int, task: str, band_index: int, slot: int, counter: int, stream: int
) -> int:
    import hashlib as _hashlib

    material = f"numform-v1|{seed}|{task}|{band_index}|{slot}|{counter}|{stream}"
    return int.from_bytes(_hashlib.sha256(material.encode("ascii")).digest()[:8], "big")


def _numform_value_in_band(derived: int, lo: int, hi: int, *, min_length: int = 1) -> int:
    low = max(lo, min_length)
    length = low + derived % (hi - low + 1)
    rest = derived >> 16
    if length == 1:
        return rest % 10
    value: int = 10 ** (length - 1) + rest % (9 * 10 ** (length - 1))
    return value


def _numform_round_value(band_index: int, slot: int) -> int:
    _lo, hi = _NUMFORM_BANDS[band_index]
    if slot == 0:
        power: int = 10 ** (hi - 1)
        return power
    nines: int = 10**hi - 1
    return nines


def _numform_band_of(value: int) -> int:
    length = len(str(value))
    for index, (lo, hi) in enumerate(_NUMFORM_BANDS):
        if lo <= length <= hi:
            return index
    raise ValueError("value outside the frozen bands")


def _numform_compare_partner(a: int, target: str, derived: int) -> int:
    digits = str(a)
    length = len(digits)
    if target == "different_length":
        new_length = length + 1 if derived % 2 == 0 else length - 1
        if new_length > 12:
            new_length = length - 1
        if new_length < 1:
            new_length = length + 1
        return _numform_value_in_band(derived >> 8, new_length, new_length)
    if target == "same_length_last_digit":
        last = int(digits[-1])
        replacement = (last + 1 + (derived % 9)) % 10
        return int(digits[:-1] + str(replacement)) if length > 1 else replacement
    position = derived % max(1, length // 2)
    current = int(digits[position])
    replacement = (current + 1 + (derived >> 4) % 9) % 10
    if position == 0 and replacement == 0:
        replacement = 1
    return int(digits[:position] + str(replacement) + digits[position + 1 :])


def _numform_add_partner(a: int, target: str, derived: int) -> int:
    digits = [int(ch) for ch in reversed(str(a))]
    partner_digits: list[int] = []
    if target == "full_chain" and digits[0] == 0:
        digits[0] = 1
    stream = derived
    if target == "none":
        for digit in digits:
            partner_digits.append(stream % (10 - digit) if digit < 9 else 0)
            stream //= 10
    elif target == "full_chain":
        carry_in = 0
        for digit in digits:
            minimum = 10 - digit - carry_in
            partner_digits.append(minimum + stream % (10 - minimum))
            carry_in = 1
            stream //= 10
    else:  # some: force exactly one carry position, none below it
        pivot_candidates = [pos for pos, digit in enumerate(digits) if digit >= 1]
        if not pivot_candidates:
            # a == 0: no digit can carry; propose zero and let
            # the caller's classifier check reject the pair
            # (mirrors the producer's cycle-187 fix exactly).
            return 0
        pivot = pivot_candidates[stream % len(pivot_candidates)]
        carry_in = 0
        for pos, digit in enumerate(digits):
            if pos < pivot:
                partner_digits.append(stream % (10 - digit) if digit < 9 else 0)
            elif pos == pivot:
                minimum = 10 - digit - carry_in
                partner_digits.append(minimum + stream % (10 - minimum))
                carry_in = 1
            elif digit + carry_in <= 9:
                partner_digits.append(stream % (10 - digit - carry_in))
                carry_in = 0
            else:
                partner_digits.append(stream % 10)
                carry_in = 1
            stream //= 10
    partner = int("".join(str(d) for d in reversed(partner_digits)))
    if partner == 0:
        partner = 1 if digits[0] <= 8 or target != "none" else 10
    return partner


def _numform_gen_copy(seed: int) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    index = 0
    for band_index, (lo, hi) in enumerate(_NUMFORM_BANDS):
        for slot in range(4):
            is_round = slot < 2
            if is_round:
                value = _numform_round_value(band_index, slot)
            else:
                value = _numform_value_in_band(
                    _numform_derive(seed, "copy", band_index, slot, 0, 0), lo, hi
                )
                counter = 0
                while _numform_is_round(value):
                    counter += 1
                    value = _numform_value_in_band(
                        _numform_derive(seed, "copy", band_index, slot, counter, 0), lo, hi
                    )
            entries.append(
                {
                    "value_id": f"nf-copy-{index:02d}",
                    "task": "copy",
                    "band": band_index,
                    "stratum": "round" if is_round else "random",
                    "operands": [value],
                    "expected_value": value,
                }
            )
            index += 1
    return entries


def _numform_gen_pairs(seed: int, task: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    index = 0
    targets = (
        ("different_length", "same_length_early", "same_length_last_digit")
        if task == "compare"
        else ("none", "some", "full_chain")
    )
    classifier = _numform_margin if task == "compare" else _numform_carry
    for band_index, (lo, hi) in enumerate(_NUMFORM_BANDS):
        for slot in range(4):
            target = targets[slot % len(targets)]
            is_round = slot < 2
            pair: tuple[int, int] | None = None
            for counter in range(1000):
                if is_round and counter == 0:
                    a = _numform_round_value(band_index, slot)
                else:
                    a = _numform_value_in_band(
                        _numform_derive(seed, task, band_index, slot, counter, 0),
                        lo,
                        hi,
                        min_length=2 if target == "same_length_early" else 1,
                    )
                derived = _numform_derive(seed, task, band_index, slot, counter, 1)
                b = (
                    _numform_compare_partner(a, target, derived)
                    if task == "compare"
                    else _numform_add_partner(a, target, derived)
                )
                if b < 0 or len(str(b)) > 12:
                    continue
                if task == "add" and target == "full_chain" and str(a)[-1] == "0":
                    a = a + 1
                if classifier(a, b) == target and a != b:
                    pair = (a, b)
                    break
            if pair is None:  # pragma: no cover - construction is verified below
                raise ValueError(f"numform generation failed for {task} slot {slot}")
            a, b = pair
            entry: dict[str, Any] = {
                "value_id": f"nf-{task}-{index:02d}",
                "task": task,
                "band": _numform_band_of(max(a, b)),
                "stratum": target,
                "operands": [a, b],
                "expected_value": max(a, b) if task == "compare" else a + b,
            }
            if task == "compare":
                larger_first = slot % 2 == 0
                first, second = (max(a, b), min(a, b)) if larger_first else (min(a, b), max(a, b))
                entry["operands"] = [first, second]
                entry["expected_label"] = "first" if larger_first else "second"
            entries.append(entry)
            index += 1
    return entries


def _numform_generated(
    seed: int,
) -> tuple[dict[tuple[str, str, str], tuple[Any, ...]], list[list[Any]]]:
    """Regenerate the seed's set; return the expected
    construction map (same tuple shape the record check builds)
    and the canonical generated rows the protocol hash binds."""
    value_set = {
        "copy": _numform_gen_copy(seed),
        "compare": _numform_gen_pairs(seed, "compare"),
        "add": _numform_gen_pairs(seed, "add"),
    }
    construction: dict[tuple[str, str, str], tuple[Any, ...]] = {}
    rows: list[list[Any]] = []
    for task in _NUMFORM_TASKS:
        for entry in value_set[task]:
            operands = [int(value) for value in entry["operands"]]
            answers = [
                str(entry["expected_label"])
                if task == "compare"
                else _numform_render(int(entry["expected_value"]), form)
                for form in _NUMFORM_FORMS
            ]
            for form, expected_answer in zip(_NUMFORM_FORMS, answers, strict=True):
                form_identical = all(
                    _numform_render(value, form) == _numform_render(value, "plain")
                    for value in operands
                )
                construction[(task, str(entry["value_id"]), form)] = (
                    task,
                    entry["value_id"],
                    form,
                    entry["band"],
                    entry["stratum"],
                    tuple(operands),
                    entry["expected_value"],
                    expected_answer,
                    form_identical,
                )
            rows.append(
                [
                    task,
                    entry["value_id"],
                    operands,
                    entry["expected_value"],
                    answers,
                ]
            )
    return construction, rows


def _numform_probe_strings() -> list[str]:
    probes: list[str] = []
    sequential = "123456789012"
    ones = "1" * 12
    zero_heavy = "1" + "0" * 11
    for length in range(1, 13):
        probes.append(sequential[:length])
        probes.append(ones[:length])
        probes.append(zero_heavy[:length])
    for form in _NUMFORM_FORMS:
        probes.append(_numform_render(123456789, form))
    return probes


def _numform_classify_grouping(evidence: Any) -> str:
    """Consistency re-run of the frozen grouping decision rule
    over stored probe evidence (the validator has no tokenizer;
    this checks the asserted class against the evidence, it
    does not re-measure the tokenizer)."""
    if type(evidence) is not dict or not evidence:
        return "unknown"
    digit_probes = [
        probe for probe in _numform_probe_strings() if probe.isascii() and probe.isdigit()
    ]
    if any(probe not in evidence for probe in digit_probes):
        return "unknown"
    counts: list[tuple[int, Any]] = []
    for probe in digit_probes:
        count = evidence[probe]
        if type(count) is not int or isinstance(count, bool) or count < 1:
            return "unknown"
        counts.append((len(probe), count))
    if all(count == length for length, count in counts):
        return "single_digit"
    if all(count == (length + 2) // 3 for length, count in counts):
        return "chunked_up_to_3"
    if all((length + 2) // 3 <= count <= length for length, count in counts):
        return "irregular_or_mixed"
    return "unknown"


def _numform_decomposition(record: dict[str, Any]) -> str:
    import hashlib as _hashlib

    cell = record["cell"]
    if cell["verdict"]:
        return "exact_pass"
    if record["task"] == "compare":
        other = "second" if record["expected_answer"] == "first" else "first"
        other_digest = _hashlib.sha256(other.encode("utf-8")).hexdigest()
        return "wrong_label" if cell["answer_digest"] == other_digest else "non_label"
    if record["answer_value"] is not None and record["answer_value"] == record["expected_value"]:
        return "value_pass_form_fail"
    return "value_fail"


def _numform_aggregate(records: list[dict[str, Any]], task: str, form: str) -> dict[str, Any]:
    selected = [r for r in records if r["task"] == task and r["form"] == form]
    completed = [r for r in selected if r["cell"] is not None]
    n_exact = sum(1 for r in completed if r["cell"]["verdict"])
    decomposition: dict[str, int] = {}
    for record in completed:
        cls = _numform_decomposition(record)
        decomposition[cls] = decomposition.get(cls, 0) + 1
    if task != "compare":
        decomposition["value_pass_form_fail_boundary_found"] = sum(
            1
            for r in completed
            if _numform_decomposition(r) == "value_pass_form_fail" and r["cell"]["boundary_found"]
        )
    by_band: dict[str, dict[str, Any]] = {}
    for band_index in range(len(_NUMFORM_BANDS)):
        band_records = [r for r in selected if r["band"] == band_index]
        band_completed = [r for r in band_records if r["cell"] is not None]
        by_band[str(band_index)] = {
            "n_values": len(band_records),
            "n_completed": len(band_completed),
            "n_exact": sum(1 for r in band_completed if r["cell"]["verdict"]),
            "n_form_identical": sum(1 for r in band_records if r["form_identical"]),
        }
    input_tokens = [int(r["cell"]["input_tokens"]) for r in completed]
    answer_tokens = [int(r["expected_answer_tokens"]) for r in completed]
    return {
        "n_values": len(selected),
        "n_completed": len(completed),
        "n_exact": n_exact,
        "exact_rate": (n_exact / len(completed)) if completed else None,
        "decomposition": decomposition,
        "by_band": by_band,
        "mean_input_tokens": (sum(input_tokens) / len(input_tokens)) if input_tokens else None,
        "mean_expected_answer_tokens": (
            (sum(answer_tokens) / len(answer_tokens)) if answer_tokens else None
        ),
    }


def _numform_classify_contrast(n_paired: int, n_a_only: int, n_b_only: int) -> str:
    if n_paired < 6:
        return "insufficient"
    if n_a_only == 0 and n_b_only == 0:
        return "no_difference_observed"
    if n_a_only >= 2 and n_a_only > n_b_only:
        return "a_advantage"
    if n_b_only >= 2 and n_b_only > n_a_only:
        return "b_advantage"
    if n_a_only == n_b_only and n_a_only >= 2:
        return "mixed_discordance"
    return "below_threshold"


def _numform_contrast(
    records: list[dict[str, Any]], task: str, form_a: str, form_b: str
) -> dict[str, Any]:
    by_value: dict[str, dict[str, Any]] = {}
    for record in records:
        if record["task"] != task or record["form"] not in (form_a, form_b):
            continue
        by_value.setdefault(str(record["value_id"]), {})[str(record["form"])] = record
    n_paired = n_a_only = n_b_only = n_both_pass = n_both_fail = 0
    for pair in by_value.values():
        rec_a = pair.get(form_a)
        rec_b = pair.get(form_b)
        if rec_a is None or rec_b is None:
            continue
        if rec_a["cell"] is None or rec_b["cell"] is None:
            continue
        n_paired += 1
        pass_a = bool(rec_a["cell"]["verdict"])
        pass_b = bool(rec_b["cell"]["verdict"])
        if pass_a and pass_b:
            n_both_pass += 1
        elif pass_a:
            n_a_only += 1
        elif pass_b:
            n_b_only += 1
        else:
            n_both_fail += 1
    label = _numform_classify_contrast(n_paired, n_a_only, n_b_only)
    if label == "a_advantage":
        label = f"{form_a}_advantage"
    elif label == "b_advantage":
        label = f"{form_b}_advantage"
    return {
        "n_paired": n_paired,
        "n_pass_a_only": n_a_only,
        "n_pass_b_only": n_b_only,
        "n_both_pass": n_both_pass,
        "n_both_fail": n_both_fail,
        "mcnemar_p": _leading_mcnemar_p(n_b_only, n_a_only),
        "classification": label,
    }


def _numform_record_construction(
    record: Any,
) -> tuple[Any, ...] | None:
    """Validate one ground record; return its construction
    tuple (the fields that must be constant across candidates)
    or None when the record is not an honest numform record."""
    import hashlib as _hashlib

    if type(record) is not dict or set(record) != _NUMFORM_RECORD_KEYS:
        return None
    task = record["task"]
    form = record["form"]
    if task not in _NUMFORM_TASKS or form not in _NUMFORM_FORMS:
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
    if any(
        type(value) is not int or isinstance(value, bool) or value < 0 or value >= 10**12
        for value in operands
    ):
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
        label = "first" if operands[0] > operands[1] else "second"
        if expected_answer != label:
            return None
    elif expected_answer != _numform_render(expected_value, form):
        return None
    band = record["band"]
    max_length = len(str(max(operands)))
    band_ok = (
        type(band) is int
        and not isinstance(band, bool)
        and 0 <= band < len(_NUMFORM_BANDS)
        and _NUMFORM_BANDS[band][0] <= max_length <= _NUMFORM_BANDS[band][1]
    )
    if not band_ok:
        return None
    stratum = record["stratum"]
    if task == "copy":
        if stratum != ("round" if _numform_is_round(operands[0]) else "random"):
            return None
    elif task == "compare":
        if stratum != _numform_margin(operands[0], operands[1]):
            return None
        if stratum in ("same_length_middle", "equal"):
            return None
    elif stratum != _numform_carry(operands[0], operands[1]):
        return None
    form_identical = record["form_identical"]
    if type(form_identical) is not bool:
        return None
    if form_identical != all(
        _numform_render(value, form) == _numform_render(value, "plain") for value in operands
    ):
        return None
    if not _leading_int_ok(record["expected_answer_tokens"]):
        return None
    answer_value = record["answer_value"]
    if answer_value is not None and (
        type(answer_value) is not int or isinstance(answer_value, bool) or answer_value < 0
    ):
        return None
    if task == "compare" and answer_value is not None:
        return None
    detail = record["detail"]
    if detail not in _NUMFORM_DETAILS:
        return None
    cell = record["cell"]
    if (cell is None) != (detail != ""):
        return None
    if cell is not None:
        if type(cell) is not dict or set(cell) != _NUMFORM_CELL_KEYS:
            return None
        if type(cell["verdict"]) is not bool or type(cell["boundary_found"]) is not bool:
            return None
        if cell["boundary_text_match"] not in (True, False, None):
            return None
        if not _leading_int_ok(cell["input_tokens"]):
            return None
        digest = cell["answer_digest"]
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(char not in _LEADING_HEX for char in digest)
        ):
            return None
        expected_digest = _hashlib.sha256(expected_answer.encode("utf-8")).hexdigest()
        if cell["verdict"] != bool(digest == expected_digest and cell["boundary_found"]):
            return None
        if (
            task != "compare"
            and digest == expected_digest
            and answer_value is not None
            and answer_value != expected_value
        ):
            return None
        # Cycle 190 (F3): the stored parse is no longer
        # trusted. The answer text must digest to the cell's
        # recorded digest, and re-parsing it under the
        # mirrored grammars must reproduce answer_value
        # exactly. Compare answers are words, not numbers:
        # they are bound by the digest alone (A2).
        answer_text = record["answer_text"]
        if type(answer_text) is not str:
            return None
        if _numform_text_digest(answer_text) != digest:
            return None
        if task != "compare" and _numform_parse_value(answer_text) != answer_value:
            return None
    elif record["answer_text"] is not None or answer_value is not None:
        # A cell that never completed carries neither an
        # answer nor a parse of one (A4: explicit, not a
        # byproduct of the aggregate filters).
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


def _numform_recompute(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Recompute a numform payload from its ground records.

    The validator carries its own renderers, classifiers,
    aggregators, and generator (no imports from numform.py):
    every record's construction facts are re-derived (expected
    value, expected answer string, band, stratum class, form
    identity), the generated set is regenerated from the
    protocol's seed and every record's construction is compared
    against it, the protocol is rebuilt from frozen scalars
    plus the payload's seed and backend fingerprint, and every
    aggregate and paired contrast is recomputed. The grouping
    classification is re-run over the stored probe evidence --
    a consistency check, since the validator has no tokenizer
    -- and the stored token means/deltas must satisfy the
    producer's arithmetic relation (each delta is its form's
    mean minus the plain mean, up to float summation order).
    The answer's parsed value is NOT trusted stored data:
    each record's answer_text is bound to its cell digest
    and re-parsed under this module's mirrored grammars,
    and the decomposition recomputes from the re-derived
    value (cycle 190); where the answer digest pins the
    answer to the expected string, the value is checked
    against the expected value exactly.
    """
    import hashlib as _hashlib
    import json as _json

    protocol = payload.get("protocol")
    if type(protocol) is not dict:
        return None
    seed = protocol.get("seed")
    if type(seed) is not int or isinstance(seed, bool):
        return None
    backend_fingerprint = protocol.get("backend_fingerprint")
    if type(backend_fingerprint) is not str or not backend_fingerprint:
        return None
    candidates = payload.get("candidates")
    if type(candidates) is not dict or not candidates:
        return None
    if payload.get("n_candidates") != len(candidates):
        return None
    construction: dict[tuple[str, str, str], tuple[Any, ...]] = {}
    answer_tokens: dict[tuple[str, str, str], int] = {}
    aggregates: dict[str, dict[str, Any]] = {}
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
            built = _numform_record_construction(record)
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
            # The expected answer's token count is measured
            # once per run (one backend), so it is part of the
            # construction every candidate must agree on, even
            # though it is not part of the regenerated set.
            tokens = record["expected_answer_tokens"]
            prior_tokens = answer_tokens.get(key)
            if prior_tokens is None:
                answer_tokens[key] = tokens
            elif prior_tokens != tokens:
                return None
        expected_keys = {
            (task, f"nf-{task}-{index:02d}", form)
            for task in _NUMFORM_TASKS
            for index in range(16)
            for form in _NUMFORM_FORMS
        }
        if seen != expected_keys:
            return None
    if not available_names or total_records > 4608:
        return None
    if (
        len(_json.dumps(all_records, separators=(",", ":"), ensure_ascii=True).encode("utf-8"))
        > 12 * 1024 * 1024
    ):
        return None
    # The generated set, regenerated from the protocol's seed:
    # the records must BE the seed's set, not merely a
    # self-consistent grid whose hash was recomputed from
    # itself. Regeneration is fail-closed -- a seed the mirror
    # cannot generate (the producer could not have generated
    # it either) invalidates the payload instead of raising.
    try:
        expected_construction, rows = _numform_generated(seed)
    except Exception:  # noqa: BLE001 -- hostile seeds fail closed
        return None
    if construction != expected_construction:
        return None
    generator_sha = _hashlib.sha256(
        _json.dumps(rows, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    templates_canonical = _json.dumps(
        _NUMFORM_TEMPLATES, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    expected_protocol: dict[str, Any] = {
        "version": 1,
        "tasks": list(_NUMFORM_TASKS),
        "forms": list(_NUMFORM_FORMS),
        "bands": [list(band) for band in _NUMFORM_BANDS],
        "values_per_task": 16,
        "min_paired_evaluable": 6,
        "min_discordant": 2,
        "generation_budget": 4608,
        "max_ground_records": 4608,
        "max_records_bytes": 12 * 1024 * 1024,
        "space_codepoint": "U+202F",
        "scientific_grammar": "mantissa-[1,10)-lowercase-e-v1",
        "renderer_spec": "value-only-renderers-v1",
        "generator_rule": "sha256-derived-paired-v1",
        "prompt_templates": dict(_NUMFORM_TEMPLATES),
        "prompt_templates_sha256": _hashlib.sha256(templates_canonical.encode("utf-8")).hexdigest(),
        "seed": seed,
        "generator_sha256": generator_sha,
        "backend_fingerprint": backend_fingerprint,
    }
    expected_with_fingerprint = {**expected_protocol, "fingerprint": protocol.get("fingerprint")}
    if not _recomputed_equal(protocol, expected_with_fingerprint):
        return None
    fingerprint = _hashlib.sha256(
        _json.dumps(
            expected_protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()
    if protocol.get("fingerprint") != fingerprint:
        return None
    grouping = payload.get("grouping")
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
    # Empty evidence is the honest stage output when the
    # backend's encoder raised during measurement (the class is
    # then "unknown", which the decision-rule re-run confirms).
    if evidence and set(evidence) != set(_numform_probe_strings()):
        return None
    if any(
        type(count) is not int or isinstance(count, bool) or count < 0
        for count in evidence.values()
    ):
        return None
    if grouping["classification"] != _numform_classify_grouping(evidence):
        return None
    means = grouping["form_token_means"]
    deltas = grouping["token_deltas_vs_plain"]
    if type(means) is not dict or set(means) != set(_NUMFORM_FORMS):
        return None
    if type(deltas) is not dict or set(deltas) != set(_NUMFORM_FORMS):
        return None
    for form in _NUMFORM_FORMS:
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
        for form in _NUMFORM_FORMS:
            mean = means[form]
            delta = deltas[form]
            if mean is None or delta is None:
                continue
            # The producer's delta is the mean of the paired
            # per-operand differences over the same operand
            # multiset as the means, so it equals mean - plain
            # mean in exact arithmetic; float summation order
            # can move the last bits, hence the tight relative
            # tolerance rather than an equality check (an exact
            # check would false-reject honest payloads).
            expected_delta = mean - plain_mean
            tolerance = 1e-9 * max(1.0, abs(expected_delta), abs(delta))
            if abs(delta - expected_delta) > tolerance:
                return None
    for name in available_names:
        records = all_records[name]
        aggregates[name] = {
            "aggregates": {
                task: {form: _numform_aggregate(records, task, form) for form in _NUMFORM_FORMS}
                for task in _NUMFORM_TASKS
            },
            "contrasts": {
                task: {
                    f"{form_a}__{form_b}": _numform_contrast(records, task, form_a, form_b)
                    for form_a, form_b in (
                        ("plain", "comma"),
                        ("plain", "space"),
                        ("plain", "scientific"),
                        ("comma", "space"),
                        ("comma", "scientific"),
                        ("space", "scientific"),
                    )
                }
                for task in _NUMFORM_TASKS
            },
        }
    return {"aggregates": aggregates, "n_candidates": len(candidates)}


def _valid_numform(payload: Any) -> bool:
    """Structural check for a ``numform`` payload (recompute standard)."""
    if type(payload) is not dict or payload.get("available") is not True:
        return False
    if set(payload) != {
        "available",
        "detail",
        "protocol",
        "grouping",
        "candidates",
        "n_candidates",
    }:
        # The exact top-level shape is the structural form of
        # the no-pooling rule: a cross-task total has no key to
        # live under, so a payload carrying one is not this
        # probe's payload.
        return False
    recomputed = _numform_recompute(payload)
    if recomputed is None:
        return False
    candidates = payload["candidates"]
    for name, expected in recomputed["aggregates"].items():
        entry = candidates[name]
        if entry.get("aggregates") != expected["aggregates"]:
            return False
        if entry.get("contrasts") != expected["contrasts"]:
            return False
    return bool(payload.get("n_candidates") == recomputed["n_candidates"])


def diff_numform(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``numform`` probe payloads; deterministic.

    Answers "did the surface-form story change between runs?":
    per-candidate per-task per-form exact-rate drift, paired
    contrast classification drift, candidates added/removed,
    and protocol / generator / backend drift (any of which
    marks the comparison not comparable, with the drift still
    listed -- the generated set is the treatment, so a changed
    generator is protocol drift). Tokenizer grouping drift is
    reported and flags the form-contrast comparison
    (``grouping_changed``) without by itself marking the
    sections incomparable: the grouping is a label on the run,
    and the rates do not depend on it.
    """
    old_ok = _valid_numform(old)
    new_ok = _valid_numform(new)
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
    numform_protocol_drift = old["protocol"].get("fingerprint") != new["protocol"].get(
        "fingerprint"
    )
    numform_generator_drift = old["protocol"].get("generator_sha256") != new["protocol"].get(
        "generator_sha256"
    )
    numform_backend_drift = old["protocol"].get("backend_fingerprint") != new["protocol"].get(
        "backend_fingerprint"
    )
    numform_comparable = (
        not numform_protocol_drift and not numform_generator_drift and not numform_backend_drift
    )
    numform_grouping_changed = old.get("grouping", {}).get("classification") != new.get(
        "grouping", {}
    ).get("classification") or old.get("grouping", {}).get("evidence") != new.get(
        "grouping", {}
    ).get("evidence")
    old_candidates = old["candidates"]
    new_candidates = new["candidates"]
    numform_candidates_added = sorted(set(new_candidates) - set(old_candidates))
    numform_candidates_removed = sorted(set(old_candidates) - set(new_candidates))
    numform_rate_drift: list[dict[str, Any]] = []
    numform_contrast_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_candidates) & set(new_candidates)):
        old_entry = old_candidates[name]
        new_entry = new_candidates[name]
        if old_entry.get("status") != "available" or new_entry.get("status") != "available":
            if old_entry.get("status") != new_entry.get("status"):
                numform_rate_drift.append(
                    {
                        "candidate": name,
                        "metric": "status",
                        "old": old_entry.get("status"),
                        "new": new_entry.get("status"),
                    }
                )
            continue
        for task in _NUMFORM_TASKS:
            for form in _NUMFORM_FORMS:
                old_rate = old_entry["aggregates"][task][form]["exact_rate"]
                new_rate = new_entry["aggregates"][task][form]["exact_rate"]
                if old_rate != new_rate:
                    numform_rate_drift.append(
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
                    numform_contrast_drift.append(
                        {
                            "candidate": name,
                            "metric": f"{task}_{pair_key}_classification",
                            "old": old_label,
                            "new": new_label,
                        }
                    )
    result["rate_drift"] = numform_rate_drift
    result["contrast_drift"] = numform_contrast_drift
    result["candidates_added"] = numform_candidates_added
    result["candidates_removed"] = numform_candidates_removed
    result["protocol_drift"] = numform_protocol_drift
    result["generator_drift"] = numform_generator_drift
    result["backend_drift"] = numform_backend_drift
    result["grouping_changed"] = numform_grouping_changed
    result["comparable"] = numform_comparable
    result["changed"] = bool(
        numform_rate_drift
        or numform_contrast_drift
        or numform_candidates_added
        or numform_candidates_removed
        or numform_protocol_drift
        or numform_generator_drift
        or numform_backend_drift
        or numform_grouping_changed
    )
    return result


# ---------------------------------------------------------------------
# Dateline (cycle 188): the validator carries its own literal copy
# of the frozen protocol -- every arm string, the block template,
# the neutral unit -- and never imports dateline.py. Independence
# is the point: a drifted producer constant must fail validation,
# not be re-derived from the drifted source.
# ---------------------------------------------------------------------

_DATELINE_VALUES: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("anchor", (2026, 10, 21)),
    ("minus_day", (2026, 10, 20)),
    ("plus_day", (2026, 10, 22)),
    ("minus_year", (2025, 10, 21)),
    ("plus_year", (2027, 10, 21)),
    ("leap", (2028, 2, 29)),
    ("boundary_eve", (2026, 12, 31)),
    ("boundary_day", (2027, 1, 1)),
)
_DATELINE_VALUE_KEYS: tuple[str, ...] = tuple(key for key, _ in _DATELINE_VALUES)
_DATELINE_FORMATS: tuple[str, ...] = ("iso", "long", "weekday_iso")
_DATELINE_MONTHS: tuple[str, ...] = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
_DATELINE_WEEKDAYS: tuple[str, ...] = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)
_DATELINE_DATE_ARMS: tuple[str, ...] = tuple(
    f"{value}_{fmt}" for value in _DATELINE_VALUE_KEYS for fmt in _DATELINE_FORMATS
)
_DATELINE_CONTROL_KEYS: tuple[str, ...] = tuple(f"nodate_{fmt}" for fmt in _DATELINE_FORMATS)
_DATELINE_TREATED: tuple[str, ...] = _DATELINE_DATE_ARMS + _DATELINE_CONTROL_KEYS + ("lie",)
_DATELINE_BLOCK_TEMPLATE = "[Context] Today's date is {DATE}. [/Context]"
_DATELINE_CONTROL_TEMPLATE = "[Context] {NEUTRAL}[/Context]"
_DATELINE_JOINER = "\n\n"
_DATELINE_NEUTRAL_UNIT = "The reference panel is mounted on the north wall of the room. "
_DATELINE_CELL_DETAILS = frozenset(
    (
        "",
        "vacuous",
        "expected_blank",
        "screened",
        "message_date_bearing",
        "arm_screened",
        "arm_context_exceeded",
        "arm_evaluation_failed",
        "control_unmatchable",
        "control_screened",
    )
)


def _dateline_format(value_key: str, fmt: str) -> str:
    """The validator's own rendering of one frozen date value."""
    import datetime as _datetime

    year, month, day = dict(_DATELINE_VALUES)[value_key]
    if fmt == "iso":
        return f"{year:04d}-{month:02d}-{day:02d}"
    if fmt == "long":
        return f"{day} {_DATELINE_MONTHS[month - 1]} {year}"
    weekday = _DATELINE_WEEKDAYS[_datetime.date(year, month, day).weekday()]
    return f"{weekday}, {year:04d}-{month:02d}-{day:02d}"


def _dateline_lie_string() -> str:
    """The lie literal, rebuilt from the validator's own tables.

    The anchor's calendar weekday is recomputed (locale-free
    integer arithmetic through the validator's own weekday
    table) and shifted +1 -- the check below then requires the
    payload's lie string to equal this literal AND to differ
    from the honest weekday-ISO rendering, so a validator-side
    derivation can never silently correct the lie into an
    honest arm.
    """
    import datetime as _datetime

    year, month, day = dict(_DATELINE_VALUES)["anchor"]
    weekday = _DATELINE_WEEKDAYS[(_datetime.date(year, month, day).weekday() + 1) % 7]
    return f"{weekday}, {year:04d}-{month:02d}-{day:02d}"


def _dateline_arm_table() -> dict[str, str]:
    table = {
        f"{value}_{fmt}": _dateline_format(value, fmt)
        for value in _DATELINE_VALUE_KEYS
        for fmt in _DATELINE_FORMATS
    }
    table["lie"] = _dateline_lie_string()
    return table


def _dateline_mcnemar_p(a_only: int, b_only: int) -> float | None:
    """Two-sided exact McNemar p (the frozen formula, local copy)."""
    import math as _math

    n = a_only + b_only
    if n == 0:
        return None
    k = min(a_only, b_only)
    tail = sum(_math.comb(n, i) for i in range(k + 1)) / (2**n)
    result: float = min(1.0, 2.0 * tail)
    return result


def _dateline_classify(n_paired: int, a_only: int, b_only: int) -> str:
    """The frozen pair-classification truth table."""
    if n_paired < 6:
        return "insufficient"
    if a_only == 0 and b_only == 0:
        return "no_discordance_observed"
    if a_only == b_only and a_only >= 2:
        return "discordant_balanced"
    if max(a_only, b_only) >= 2:
        return "discordant_asymmetric"
    return "below_threshold"


def _dateline_live(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        r
        for r in records
        if not r["vacuous"]
        and not r["expected_blank"]
        and not r["screened"]
        and not r["message_date_bearing"]
    ]


def _dateline_pair_block(records: list[dict[str, Any]], key_a: str, key_b: str) -> dict[str, Any]:
    n_paired = 0
    n_a_only = 0
    n_b_only = 0
    n_both = 0
    n_pass_a = 0
    n_pass_b = 0
    for record in records:
        cell_a = record["cells"][key_a]
        cell_b = record["cells"][key_b]
        if cell_a is None or cell_b is None:
            continue
        n_paired += 1
        if cell_a["verdict"]:
            n_pass_a += 1
        if cell_b["verdict"]:
            n_pass_b += 1
        if cell_a["verdict"] and cell_b["verdict"]:
            n_both += 1
        elif cell_a["verdict"]:
            n_a_only += 1
        elif cell_b["verdict"]:
            n_b_only += 1
    return {
        "n_paired": n_paired,
        "n_a_only": n_a_only,
        "n_b_only": n_b_only,
        "n_both": n_both,
        "n_neither": n_paired - n_a_only - n_b_only - n_both,
        "rate_a": (n_pass_a / n_paired) if n_paired else None,
        "rate_b": (n_pass_b / n_paired) if n_paired else None,
        "mcnemar_p": _dateline_mcnemar_p(n_a_only, n_b_only),
        "classification": _dateline_classify(n_paired, n_a_only, n_b_only),
    }


def _dateline_loss_block(live: list[dict[str, Any]], value: str) -> dict[str, Any]:
    """Independent recompute of ``dateline._loss_block``."""
    evaluable: set[str] = set()
    lost: set[str] = set()
    counterpart: set[str] = set()
    per_format: dict[str, Any] = {}
    for fmt in _DATELINE_FORMATS:
        control_key = f"nodate_{fmt}"
        value_key = f"{value}_{fmt}"
        n_evaluable = 0
        n_loss = 0
        for record in live:
            control = record["cells"][control_key]
            cell = record["cells"][value_key]
            if control is None or cell is None:
                continue
            n_evaluable += 1
            evaluable.add(record["fixture"])
            if control["verdict"] and not cell["verdict"]:
                n_loss += 1
                lost.add(record["fixture"])
            elif cell["verdict"] and not control["verdict"]:
                counterpart.add(record["fixture"])
        per_format[fmt] = {"n_evaluable": n_evaluable, "n_loss": n_loss}
    block: dict[str, Any] = {
        "n_evaluable": len(evaluable),
        "n_loss": len(lost),
        "loss_rate": (len(lost) / len(evaluable)) if evaluable else None,
        "n_counterpart": len(counterpart),
        "per_format": per_format,
    }
    if value == "anchor":
        n_three = 0
        n_any_text = 0
        n_content_specific = 0
        for record in live:
            control = record["cells"]["nodate_weekday_iso"]
            cell = record["cells"]["anchor_weekday_iso"]
            lie = record["cells"]["lie"]
            if control is None or cell is None or lie is None:
                continue
            n_three += 1
            if control["verdict"] and not cell["verdict"]:
                if not lie["verdict"]:
                    n_any_text += 1
                else:
                    n_content_specific += 1
        block["lie_split"] = {
            "n_evaluable": n_three,
            "n_loss_any_text": n_any_text,
            "n_loss_content_specific": n_content_specific,
        }
    else:
        block["lie_split"] = None
    return block


def _dateline_aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Independent recompute of ``dateline.aggregate_candidate``."""
    from itertools import combinations as _combinations

    live = _dateline_live(records)
    arms: dict[str, Any] = {}
    for key in _DATELINE_TREATED:
        n_completed = 0
        n_pass = 0
        for record in live:
            cell = record["cells"][key]
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
    pairs_by_format: dict[str, Any] = {}
    total_pure_date_discordant = 0
    for fmt in _DATELINE_FORMATS:
        table: dict[str, Any] = {}
        for value_a, value_b in _combinations(_DATELINE_VALUE_KEYS, 2):
            block = _dateline_pair_block(live, f"{value_a}_{fmt}", f"{value_b}_{fmt}")
            total_pure_date_discordant += block["n_a_only"] + block["n_b_only"]
            table[f"{value_a}|{value_b}"] = block
        pairs_by_format[fmt] = table
    format_pairs: dict[str, Any] = {}
    for value in _DATELINE_VALUE_KEYS:
        table = {}
        for fmt_a, fmt_b in _combinations(_DATELINE_FORMATS, 2):
            table[f"{fmt_a}|{fmt_b}"] = _dateline_pair_block(
                live, f"{value}_{fmt_a}", f"{value}_{fmt_b}"
            )
        format_pairs[value] = table
    date_loss = {value: _dateline_loss_block(live, value) for value in _DATELINE_VALUE_KEYS}
    control_evaluable: dict[str, Any] = {}
    for fmt in _DATELINE_FORMATS:
        flags = [record["control_evaluable"][fmt] for record in live]
        n_evaluable = sum(1 for flag in flags if flag)
        control_evaluable[fmt] = {
            "n_evaluable": n_evaluable,
            "n_live": len(flags),
            "share": (n_evaluable / len(flags)) if flags else None,
        }
    return {
        "arms": arms,
        "pairs_by_format": pairs_by_format,
        "format_pairs": format_pairs,
        "date_loss": date_loss,
        "control_evaluable": control_evaluable,
        "date_content_falsified": bool(
            total_pure_date_discordant == 0
            and all(block["n_loss"] == 0 for block in date_loss.values())
        ),
        "n_fixtures": len(records),
        "n_vacuous": sum(1 for r in records if r["vacuous"]),
        "n_expected_blank": sum(1 for r in records if r["expected_blank"]),
        "n_echo": sum(1 for r in live if r["echo_fixture"]),
        "n_screened": sum(1 for r in records if r["screened"]),
        "n_message_date_bearing": sum(1 for r in records if r["message_date_bearing"]),
    }


def _dateline_recompute(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Recompute a dateline payload from its ground records.

    The protocol is rebuilt from this module's own literal arm
    table and frozen scalars (never imported from dateline.py)
    plus the payload's fixture-set digest and backend
    fingerprint, and the fingerprint is compared. Weekday names
    in the arm table are recomputed here from the calendar, so
    an arm rendered with the anchor's weekday throughout -- the
    templating bug that would convert five arms into extra lie
    arms -- fails validation, and the lie arm must equal the
    frozen lie literal while differing from its honest rendering.
    Control matching is verified from per-cell input_tokens
    wherever the anchor cell completed (a completed control cell
    must carry exactly the anchor cell's token count); where the
    anchor cell did not complete, the sizing is stage-attested
    and the docstring says so. The eligibility screen is a
    property of expected *text* and is not recomputable from
    digests; the validator checks the screened and
    message-date-bearing flags' cross-candidate constancy as
    reference fields, and the design doc states that limit.
    """
    import hashlib as _hashlib
    import json as _json

    protocol = payload.get("protocol")
    if type(protocol) is not dict:
        return None
    if set(protocol) != {
        "version",
        "block_template",
        "control_template",
        "joiner",
        "neutral_unit",
        "sizing_rule",
        "screen_rule",
        "arm_dates",
        "treated_arms",
        "min_paired_evaluable",
        "min_discordant",
        "generation_budget",
        "max_ground_records",
        "max_records_bytes",
        "fixture_set_sha256",
        "backend_fingerprint",
        "fingerprint",
    }:
        return None
    if not _recomputed_equal(protocol.get("version"), 1):
        return None
    if protocol.get("block_template") != _DATELINE_BLOCK_TEMPLATE:
        return None
    if protocol.get("control_template") != _DATELINE_CONTROL_TEMPLATE:
        return None
    if protocol.get("joiner") != _DATELINE_JOINER:
        return None
    if protocol.get("neutral_unit") != _DATELINE_NEUTRAL_UNIT:
        return None
    if protocol.get("sizing_rule") != "whole-units-plus-one-truncated-suffix-v1":
        return None
    if protocol.get("screen_rule") != "dateline-components-r4-v1":
        return None
    arm_table = _dateline_arm_table()
    if protocol.get("arm_dates") != arm_table:
        return None
    if arm_table["lie"] == _dateline_format("anchor", "weekday_iso"):
        return None  # the lie must differ from the honest rendering
    if protocol.get("treated_arms") != list(_DATELINE_TREATED):
        return None
    if protocol.get("min_paired_evaluable") != 6 or protocol.get("min_discordant") != 2:
        return None
    if protocol.get("generation_budget") != 67200 or protocol.get("max_ground_records") != 2400:
        return None
    if protocol.get("max_records_bytes") != 24 * 1024 * 1024:
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
        if records_bytes > 24 * 1024 * 1024:
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
            if not _leading_int_ok(record.get("n_user_messages")) or record["n_user_messages"] < 1:
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
            if type(seam) is not list or any(not _leading_int_ok(token) for token in seam):
                return None
            if not _leading_digest_ok(record.get("expected_digest")):
                return None
            if record.get("expected_digest") is None:
                return None
            control_evaluable = record.get("control_evaluable")
            if type(control_evaluable) is not dict or set(control_evaluable) != set(
                _DATELINE_FORMATS
            ):
                return None
            if any(type(flag) is not bool for flag in control_evaluable.values()):
                return None
            native = record.get("native")
            if not _leading_cell_consistent(native, record.get("expected_digest")):
                return None
            cells = record.get("cells")
            details = record.get("cell_details")
            if type(cells) is not dict or set(cells) != set(_DATELINE_TREATED):
                return None
            if type(details) is not dict or set(details) != set(_DATELINE_TREATED):
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
            for key in _DATELINE_TREATED:
                detail = details[key]
                if detail not in _DATELINE_CELL_DETAILS:
                    return None
                cell = cells[key]
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
                if key in _DATELINE_CONTROL_KEYS:
                    fmt = key.removeprefix("nodate_")
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
                if not _leading_cell_consistent(cell, record.get("expected_digest")):
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
        aggregates[name] = _dateline_aggregate(records)
    if fixture_ids is None:
        return None
    expected_fixture_sha = _hashlib.sha256(
        "\n".join(sorted(fixture_ids)).encode("utf-8")
    ).hexdigest()
    if protocol.get("fixture_set_sha256") != expected_fixture_sha:
        return None
    protocol_rebuilt = {
        "version": 1,
        "block_template": _DATELINE_BLOCK_TEMPLATE,
        "control_template": _DATELINE_CONTROL_TEMPLATE,
        "joiner": _DATELINE_JOINER,
        "neutral_unit": _DATELINE_NEUTRAL_UNIT,
        "sizing_rule": "whole-units-plus-one-truncated-suffix-v1",
        "screen_rule": "dateline-components-r4-v1",
        "arm_dates": arm_table,
        "treated_arms": list(_DATELINE_TREATED),
        "min_paired_evaluable": 6,
        "min_discordant": 2,
        "generation_budget": 67200,
        "max_ground_records": 2400,
        "max_records_bytes": 24 * 1024 * 1024,
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
    arms_out: dict[str, Any] = {}
    for key in _DATELINE_TREATED:
        completed_sets: list[frozenset[str]] = []
        all_pass: dict[str, bool | None] = {}
        withheld = False
        for cand_name, entry in candidates.items():
            if entry.get("status") != "available":
                withheld = True
                all_pass[cand_name] = None
                continue
            live = _dateline_live(raw_records[cand_name])
            if key in _DATELINE_CONTROL_KEYS:
                control_format = key.removeprefix("nodate_")
                defined = frozenset(
                    r["fixture"] for r in live if r["control_evaluable"][control_format]
                )
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
            arms_out[key] = {
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
        arms_out[key] = {
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
    populations: dict[str, frozenset[str]] = {}
    for cand_name, records in available_records.items():
        ids_for_cand = {
            r["fixture"]
            for r in _dateline_live(records)
            if all(r["cells"][key] is not None for key in _DATELINE_TREATED)
        }
        populations[cand_name] = frozenset(ids_for_cand)
    common = frozenset.intersection(*populations.values()) if populations else frozenset()
    if not common:
        common_out: dict[str, Any] = {
            "n_common": 0,
            "detail": "common_population_empty",
            "per_candidate": {},
        }
    else:
        from itertools import combinations as _combinations

        per_candidate: dict[str, Any] = {}
        for cand_name, records in available_records.items():
            subset = [r for r in _dateline_live(records) if r["fixture"] in common]
            discordant = 0
            for fmt in _DATELINE_FORMATS:
                for value_a, value_b in _combinations(_DATELINE_VALUE_KEYS, 2):
                    block = _dateline_pair_block(subset, f"{value_a}_{fmt}", f"{value_b}_{fmt}")
                    discordant += block["n_a_only"] + block["n_b_only"]
            loss = sum(
                _dateline_loss_block(subset, value)["n_loss"] for value in _DATELINE_VALUE_KEYS
            )
            per_candidate[cand_name] = {
                "n_pure_date_discordant": discordant,
                "n_loss_total": loss,
            }
        common_out = {
            "n_common": len(common),
            "detail": "" if len(common) >= 6 else "common_population_below_floor",
            "per_candidate": per_candidate,
        }
    n_multi_turn = sum(1 for fields in reference.values() if fields["n_user_messages"] > 1)
    return {
        "aggregates": aggregates,
        "arms": arms_out,
        "common_population": common_out,
        "n_candidates": len(candidates),
        "n_search_fixtures": len(fixture_ids),
        "n_multi_turn_fixtures": n_multi_turn,
    }


def _valid_dateline(payload: Any) -> bool:
    """Structural check for a ``dateline`` payload (recompute standard)."""
    if type(payload) is not dict or payload.get("available") is not True:
        return False
    for key in ("primary_decision", "primary_winner"):
        value = payload.get(key)
        if value is not None and type(value) is not str:
            return False
    if not _valid_primary_pair(payload):
        return False
    recomputed = _dateline_recompute(payload)
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
    if not _recomputed_equal(payload.get("common_population"), recomputed["common_population"]):
        return False
    if not _recomputed_equal(payload.get("n_candidates"), recomputed["n_candidates"]):
        return False
    if not _recomputed_equal(payload.get("n_search_fixtures"), recomputed["n_search_fixtures"]):
        return False
    return _recomputed_equal(
        payload.get("n_multi_turn_fixtures"), recomputed["n_multi_turn_fixtures"]
    )


def diff_dateline(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``dateline`` probe payloads; deterministic.

    Answers "did the date-line story change between runs?":
    per-candidate date-loss and falsification drift, pure-date
    pair classification drift, per-arm selection drift,
    candidates added/removed, and protocol / fixture-set /
    backend drift (any of which marks the comparison not
    comparable, with the drift still listed -- the frozen arm
    table is part of the protocol, so a changed table is
    protocol drift). A side without a valid payload is reported
    plainly.
    """
    old_ok = _valid_dateline(old)
    new_ok = _valid_dateline(new)
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
    dateline_protocol_drift = old["protocol"].get("fingerprint") != new["protocol"].get(
        "fingerprint"
    )
    dateline_fixture_set_drift = old["protocol"].get("fixture_set_sha256") != new["protocol"].get(
        "fixture_set_sha256"
    )
    dateline_backend_drift = old["protocol"].get("backend_fingerprint") != new["protocol"].get(
        "backend_fingerprint"
    )
    dateline_comparable = (
        not dateline_protocol_drift
        and not dateline_fixture_set_drift
        and not dateline_backend_drift
    )
    old_candidates = old["candidates"]
    new_candidates = new["candidates"]
    dateline_candidates_added = sorted(set(new_candidates) - set(old_candidates))
    dateline_candidates_removed = sorted(set(old_candidates) - set(new_candidates))
    dateline_loss_drift: list[dict[str, Any]] = []
    dateline_pair_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_candidates) & set(new_candidates)):
        old_entry = old_candidates[name]
        new_entry = new_candidates[name]
        if old_entry.get("status") != "available" or new_entry.get("status") != "available":
            if old_entry.get("status") != new_entry.get("status"):
                dateline_loss_drift.append(
                    {
                        "candidate": name,
                        "metric": "status",
                        "old": old_entry.get("status"),
                        "new": new_entry.get("status"),
                    }
                )
            continue
        if old_entry.get("date_content_falsified") != new_entry.get("date_content_falsified"):
            dateline_loss_drift.append(
                {
                    "candidate": name,
                    "metric": "date_content_falsified",
                    "old": old_entry.get("date_content_falsified"),
                    "new": new_entry.get("date_content_falsified"),
                }
            )
        for value in _DATELINE_VALUE_KEYS:
            old_block = old_entry["date_loss"][value]
            new_block = new_entry["date_loss"][value]
            if old_block["loss_rate"] != new_block["loss_rate"]:
                dateline_loss_drift.append(
                    {
                        "candidate": name,
                        "metric": f"{value}_loss_rate",
                        "old": old_block["loss_rate"],
                        "new": new_block["loss_rate"],
                    }
                )
        for fmt in _DATELINE_FORMATS:
            for pair, old_block in old_entry["pairs_by_format"][fmt].items():
                new_block = new_entry["pairs_by_format"][fmt][pair]
                if old_block["classification"] != new_block["classification"]:
                    dateline_pair_drift.append(
                        {
                            "candidate": name,
                            "metric": f"{fmt}_{pair}_classification",
                            "old": old_block["classification"],
                            "new": new_block["classification"],
                        }
                    )
    dateline_arm_drift: list[dict[str, Any]] = []
    for key in old["arms"]:
        if old["arms"][key] != new["arms"].get(key):
            dateline_arm_drift.append(
                {
                    "arm": key,
                    "old": old["arms"][key],
                    "new": new["arms"].get(key),
                }
            )
    result["loss_drift"] = dateline_loss_drift
    result["pair_drift"] = dateline_pair_drift
    result["arm_drift"] = dateline_arm_drift
    result["candidates_added"] = dateline_candidates_added
    result["candidates_removed"] = dateline_candidates_removed
    result["protocol_drift"] = dateline_protocol_drift
    result["fixture_set_drift"] = dateline_fixture_set_drift
    result["backend_drift"] = dateline_backend_drift
    result["comparable"] = dateline_comparable
    result["changed"] = bool(
        dateline_loss_drift
        or dateline_pair_drift
        or dateline_arm_drift
        or dateline_candidates_added
        or dateline_candidates_removed
        or dateline_protocol_drift
        or dateline_fixture_set_drift
        or dateline_backend_drift
    )
    return result


# ---------------------------------------------------------------------------
# multiturn mirror (cycle 189): independent recompute -- never imports
# multiturn.py. The generator, the aggregate, and the classification
# are re-implemented from the frozen design literals, so a payload is
# accepted only when two independent implementations agree. The one
# stage-attested link is the correspondence between the recorded
# intermediate digests and the history actually rendered (disclosed
# in docs/MULTITURN_DESIGN.md); every identity among recorded fields
# is checked here.
# ---------------------------------------------------------------------------

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
    import hashlib as _hashlib

    material = f"multiturn-v1|{seed}|{family}|{depth}|{index}|{slot}|{counter}"
    return int.from_bytes(_hashlib.sha256(material.encode("ascii")).digest()[:8], "big")


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
                    expected = _MULTITURN_LABELS[
                        max(range(len(values)), key=lambda slot: values[slot])
                    ]
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
    import hashlib as _hashlib

    return _hashlib.sha256(text.encode("utf-8")).hexdigest()


def _multiturn_template_digest() -> str:
    import hashlib as _hashlib
    import json as _json

    canonical = _json.dumps(
        [list(_MULTITURN_TEMPLATES), _MULTITURN_JOINER, _MULTITURN_SCRIPTED],
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return _hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _multiturn_ground_sha(seed: int) -> str:
    import hashlib as _hashlib
    import json as _json

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
    canonical = _json.dumps(rows, separators=(",", ":"), ensure_ascii=True)
    return _hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _multiturn_protocol(seed: int, backend_fingerprint: str) -> dict[str, Any]:
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
    import math as _math

    n = a_only + b_only
    if n == 0:
        return None
    k = min(a_only, b_only)
    tail = sum(_math.comb(n, i) for i in range(k + 1)) / (2**n)
    result: float = min(1.0, 2.0 * tail)
    return result


def _multiturn_cell(record: dict[str, Any], arm: str) -> dict[str, Any] | None:
    cells = record.get("cells")
    if type(cells) is not dict:
        return None
    cell = cells.get(arm)
    return cell if type(cell) is dict else None


def _multiturn_verdict(record: dict[str, Any], arm: str) -> bool:
    cell = _multiturn_cell(record, arm)
    return bool(cell["verdict"]) if cell is not None else False


def _multiturn_pair(records: list[dict[str, Any]], arm_a: str, arm_b: str) -> dict[str, int]:
    counts = {"n_paired": 0, "a_only": 0, "b_only": 0, "both_pass": 0, "both_fail": 0}  # nosec B105 -- 'pass' keys are verdict counts in the frozen payload schema, not credentials
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
        self_history = {
            "n_full_pass": 0,  # nosec B105 -- 'pass' keys are verdict counts in the frozen payload schema, not credentials
            "sh_only": 0,
            "sc_only": 0,
            "both_pass": 0,  # nosec B105 -- 'pass' keys are verdict counts in the frozen payload schema, not credentials
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
    return (
        type(value) is str
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _multiturn_cell_ok(cell: Any, arm: str, depth: int, expected: str) -> bool:
    """One cell's shape and identities, including the verdict
    binding ``verdict == (answer_digest == sha256(expected))
    and boundary_found`` and, for the sharded arm, the turns
    identities."""
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
    import hashlib as _hashlib
    import json as _json

    protocol = payload.get("protocol")
    if type(protocol) is not dict:
        return None
    seed = protocol.get("seed")
    backend_fingerprint = protocol.get("backend_fingerprint")
    if type(seed) is not int or type(backend_fingerprint) is not str:
        return None
    expected_protocol = _multiturn_protocol(seed, backend_fingerprint)
    fingerprint = _hashlib.sha256(
        _json.dumps(
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
            _json.dumps(records, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
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


# ---------------------------------------------------------------------------
# needle mirror (cycle 191): independent recompute -- never imports
# needle.py. The generator, the case material, the aggregates, the
# headline readouts, and the selections are re-implemented from the
# frozen design literals, so a payload is accepted only when two
# independent implementations agree. The stage-attested links are
# the needle offsets (checked against the frozen targets; the
# validator has no renders) and the absolute level of input_tokens
# (every identity checked here is a difference) -- both disclosed
# in docs/NEEDLE_DESIGN.md.
# ---------------------------------------------------------------------------

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
    return f"dose_{dose}_depth_{depth}"


def _needle_key_absent(dose: int) -> str:
    return f"absent_{dose}"


def _needle_key_duplicate(dose: int) -> str:
    return f"duplicate_{dose}"


_NEEDLE_CELL_KEYS = (
    ("dose_0",)
    + tuple(_needle_key_treated(d, p) for d in _NEEDLE_DOSES for p in _NEEDLE_DEPTHS)
    + tuple(_needle_key_absent(d) for d in _NEEDLE_DOSES)
    + tuple(_needle_key_duplicate(d) for d in _NEEDLE_DOSES)
)


def _needle_digest(text: str) -> str:
    import hashlib as _hashlib

    return _hashlib.sha256(text.encode("utf-8")).hexdigest()


def _needle_digest_ok(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _needle_derive(seed: int, index: int, slot: int, counter: int) -> int:
    import hashlib as _hashlib

    material = f"{seed}|needle|{index}|{slot}|{counter}"
    return int.from_bytes(_hashlib.sha256(material.encode("ascii")).digest()[:8], "big")


def _needle_generate(seed: int) -> list[dict[str, Any]]:
    """Mirror of the frozen case generation: slot 0 is the code
    word, slot 1 the number; a value colliding with an earlier
    case redraws both slots at the next counter."""
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
    import hashlib as _hashlib
    import json as _json

    canonical = _json.dumps(
        [
            _NEEDLE_FACT_TEMPLATE,
            _NEEDLE_QUESTION_TEMPLATE,
            list(_NEEDLE_TOPICS),
            list(_NEEDLE_WORDS),
        ],
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return _hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _needle_ground_sha(seed: int) -> str:
    import hashlib as _hashlib
    import json as _json

    rows = [[case["fixture_id"], case["topic"], case["value"]] for case in _needle_generate(seed)]
    canonical = _json.dumps(rows, separators=(",", ":"), ensure_ascii=True)
    return _hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _needle_protocol(seed: int, backend_fingerprint: str) -> dict[str, Any]:
    import hashlib as _hashlib
    import json as _json

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
    fingerprint = _hashlib.sha256(
        _json.dumps(protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
    ).hexdigest()
    protocol["fingerprint"] = fingerprint
    return protocol


def _needle_cell(record: dict[str, Any], key: str) -> dict[str, Any] | None:
    cells = record.get("cells")
    if type(cells) is not dict:
        return None
    cell = cells.get(key)
    return cell if type(cell) is dict else None


def _needle_verdict(record: dict[str, Any], key: str) -> bool:
    cell = _needle_cell(record, key)
    return bool(cell["verdict"]) if cell is not None else False


def _needle_cell_ok(cell: Any, expected_digest: str) -> bool:
    """One cell's shape and the verdict binding
    ``verdict == (answer_digest == expected_digest) and
    boundary_found``. The expected digest is the record's,
    itself checked against the mirror-regenerated case."""
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
    paired-restricted flip baseline."""
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
    import json as _json

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
            _json.dumps(records, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
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
_PLACEMENT_JOINER = "\n\n"
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
    placement.py anywhere in this mirror)."""
    import hashlib as _hashlib

    return _hashlib.sha256(text.encode("utf-8")).hexdigest()


def _placement_digest_ok(value: Any) -> bool:
    """Whether a stored answer digest is None or a well-formed
    SHA-256 hex digest (fails closed on lone surrogates, the
    cycle-190 discipline)."""
    if value is None:
        return True
    if type(value) is not str or len(value) != 64:
        return False
    return all(ch in "0123456789abcdef" for ch in value)


def _placement_derive(seed: int, index: int, slot: int, counter: int) -> int:
    """Mirror of placement._derive (frozen material string)."""
    import hashlib as _hashlib

    material = f"{seed}|placement|{index}|{slot}|{counter}"
    return int.from_bytes(_hashlib.sha256(material.encode("ascii")).digest()[:8], "big")


def _placement_generate(seed: int) -> list[dict[str, Any]]:
    """Mirror of placement.generate_cases: the frozen case set
    for a seed, regenerated from the protocol's own seed so a
    fabricated set is rejected (the numform F2 lesson)."""
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
    import json as _json

    canonical = _json.dumps(
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
    import json as _json

    rows = [
        [case["fixture_id"], case["topic"], case["value"], list(case["sentence_indices"])]
        for case in _placement_generate(seed)
    ]
    canonical = _json.dumps(rows, separators=(",", ":"), ensure_ascii=True)
    return _placement_digest(canonical)


def _placement_protocol(seed: int, backend_fingerprint: str) -> dict[str, Any]:
    """The placement protocol rebuilt from this module's own
    literals plus the payload's seed and backend fingerprint,
    fingerprint included (compared whole)."""
    import json as _json

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
    canonical = _json.dumps(protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    protocol["fingerprint"] = _placement_digest(canonical)
    return protocol


def _placement_mcnemar_p(a_only: int, b_only: int) -> float | None:
    """Two-sided exact McNemar p (the frozen formula, local copy)."""
    import math as _math

    n = a_only + b_only
    if n == 0:
        return None
    k = min(a_only, b_only)
    tail = sum(_math.comb(n, i) for i in range(k + 1)) / (2**n)
    result: float = min(1.0, 2.0 * tail)
    return result


def _placement_cell_get(record: dict[str, Any], key: str) -> dict[str, Any] | None:
    cells = record.get("cells")
    if not isinstance(cells, dict):
        return None
    cell = cells.get(key)
    return cell if isinstance(cell, dict) else None


def _placement_verdict(record: dict[str, Any], key: str) -> bool:
    cell = _placement_cell_get(record, key)
    return bool(cell is not None and cell["verdict"])


def _placement_screened(record: dict[str, Any]) -> bool:
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
    itself checked against the mirror-regenerated case."""
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
    import json as _json

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
            _json.dumps(records, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
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


_CANONICALFORM_ARM_KEYS = ("nfc", "nfd")
_CANONICALFORM_BASELINE_ARM = "nfc"
_CANONICALFORM_CASE_COUNT = 24
_CANONICALFORM_FACT_TEMPLATE = "The {topic} inspection code is {value}."
_CANONICALFORM_QUESTION_TEMPLATE = "What is the {topic} inspection code?"
_CANONICALFORM_TOPICS = (
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
_CANONICALFORM_WORDS = (
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
_CANONICALFORM_NEUTRAL_POOL = (
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
_CANONICALFORM_MIN_PAIRED_EVALUABLE = 6
_CANONICALFORM_MIN_DISCORDANT = 2
_CANONICALFORM_MAX_GROUND_RECORDS = 24 * _CANONICALFORM_CASE_COUNT
_CANONICALFORM_GENERATION_BUDGET = len(_CANONICALFORM_ARM_KEYS) * _CANONICALFORM_MAX_GROUND_RECORDS
_CANONICALFORM_MAX_RECORDS_BYTES = 12 * 1024 * 1024
_CANONICALFORM_CELL_FIELDS = {
    "verdict",
    "boundary_found",
    "boundary_text_match",
    "input_tokens",
    "answer_digest",
}
_CANONICALFORM_DETAILS = ("arm_context_exceeded", "arm_evaluation_failed")
_CANONICALFORM_AGGREGATE_KEYS = {
    "arms",
    "ids",
    "headline",
    "invariance_falsified",
    "n_cases",
    "n_incomplete",
}


def _canonicalform_digest(text: str) -> str:
    """SHA-256 hex digest (local copy; no imports from
    canonicalform.py anywhere in this mirror)."""
    import hashlib as _hashlib

    return _hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonicalform_digest_ok(value: Any) -> bool:
    """Whether a stored digest is None or a well-formed SHA-256
    hex digest (fails closed on lone surrogates, the cycle-190
    discipline)."""
    if value is None:
        return True
    if type(value) is not str or len(value) != 64:
        return False
    return all(ch in "0123456789abcdef" for ch in value)


def _canonicalform_derive(seed: int, index: int, slot: int, counter: int) -> int:
    """Mirror of canonicalform._derive (frozen material string)."""
    import hashlib as _hashlib

    material = f"{seed}|canonicalform|{index}|{slot}|{counter}"
    return int.from_bytes(_hashlib.sha256(material.encode("ascii")).digest()[:8], "big")


def _canonicalform_generate(seed: int) -> list[dict[str, Any]]:
    """Mirror of canonicalform.generate_cases: the frozen case
    set for a seed, regenerated from the protocol's own seed so
    a fabricated set is rejected (the numform F2 lesson). The
    authored content is NFC; the NFD form is derived from it,
    exactly as the producer derives it."""
    import unicodedata as _unicodedata

    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index in range(_CANONICALFORM_CASE_COUNT):
        topic = _CANONICALFORM_TOPICS[index]
        counter = 0
        while True:
            word = _CANONICALFORM_WORDS[
                _canonicalform_derive(seed, index, 0, counter) % len(_CANONICALFORM_WORDS)
            ]
            number = 1000 + _canonicalform_derive(seed, index, 1, counter) % 9000
            value = f"{word}-{number}"
            if value not in seen:
                break
            counter += 1
        seen.add(value)
        first = _canonicalform_derive(seed, index, 2, 0) % len(_CANONICALFORM_NEUTRAL_POOL)
        second = _canonicalform_derive(seed, index, 3, 0) % len(_CANONICALFORM_NEUTRAL_POOL)
        if second == first:
            second = (first + 1) % len(_CANONICALFORM_NEUTRAL_POOL)
        fact = _CANONICALFORM_FACT_TEMPLATE.format(topic=topic, value=value)
        payload = (
            f"{_CANONICALFORM_NEUTRAL_POOL[first]} {fact} {_CANONICALFORM_NEUTRAL_POOL[second]}"
        )
        content = f"{payload} {_CANONICALFORM_QUESTION_TEMPLATE.format(topic=topic)}"
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


def _canonicalform_case_material_sha256() -> str:
    """Mirror of canonicalform.case_material_sha256."""
    import json as _json

    canonical = _json.dumps(
        [
            _CANONICALFORM_FACT_TEMPLATE,
            _CANONICALFORM_QUESTION_TEMPLATE,
            list(_CANONICALFORM_TOPICS),
            list(_CANONICALFORM_WORDS),
            list(_CANONICALFORM_NEUTRAL_POOL),
        ],
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return _canonicalform_digest(canonical)


def _canonicalform_ground_sha256(seed: int) -> str:
    """Mirror of canonicalform.ground_sha256, including both
    arms' content digests so the ground digest binds the
    treatment."""
    import json as _json

    rows = [
        [
            case["fixture_id"],
            case["topic"],
            case["value"],
            list(case["sentence_indices"]),
            _canonicalform_digest(case["nfc_content"]),
            _canonicalform_digest(case["nfd_content"]),
        ]
        for case in _canonicalform_generate(seed)
    ]
    canonical = _json.dumps(rows, separators=(",", ":"), ensure_ascii=True)
    return _canonicalform_digest(canonical)


def _canonicalform_protocol(seed: int, backend_fingerprint: str) -> dict[str, Any]:
    """The canonical-form protocol rebuilt from this module's
    own literals plus the payload's seed and backend
    fingerprint, fingerprint included (compared whole)."""
    import json as _json

    protocol: dict[str, Any] = {
        "version": 1,
        "arm_keys": list(_CANONICALFORM_ARM_KEYS),
        "baseline_arm": _CANONICALFORM_BASELINE_ARM,
        "case_count": _CANONICALFORM_CASE_COUNT,
        "case_material_sha256": _canonicalform_case_material_sha256(),
        "ground_sha256": _canonicalform_ground_sha256(seed),
        "min_paired_evaluable": _CANONICALFORM_MIN_PAIRED_EVALUABLE,
        "min_discordant": _CANONICALFORM_MIN_DISCORDANT,
        "generation_budget": _CANONICALFORM_GENERATION_BUDGET,
        "max_ground_records": _CANONICALFORM_MAX_GROUND_RECORDS,
        "max_records_bytes": _CANONICALFORM_MAX_RECORDS_BYTES,
        "seed": seed,
        "backend_fingerprint": backend_fingerprint,
    }
    canonical = _json.dumps(protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    protocol["fingerprint"] = _canonicalform_digest(canonical)
    return protocol


def _canonicalform_mcnemar_p(a_only: int, b_only: int) -> float | None:
    """Two-sided exact McNemar p (the frozen formula, local copy)."""
    import math as _math

    n = a_only + b_only
    if n == 0:
        return None
    k = min(a_only, b_only)
    tail = sum(_math.comb(n, i) for i in range(k + 1)) / (2**n)
    result: float = min(1.0, 2.0 * tail)
    return result


def _canonicalform_cell_get(record: dict[str, Any], key: str) -> dict[str, Any] | None:
    cells = record.get("cells")
    if not isinstance(cells, dict):
        return None
    cell = cells.get(key)
    return cell if isinstance(cell, dict) else None


def _canonicalform_verdict(record: dict[str, Any], key: str) -> bool:
    cell = _canonicalform_cell_get(record, key)
    return bool(cell is not None and cell["verdict"])


def _canonicalform_pair_block_divergent(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Mirror of canonicalform._pair_block_divergent: the
    headline population is both-cells-completed AND
    ids-divergent; ids-identical records never enter."""
    n_paired = 0
    n_nfc_only = 0
    n_nfd_only = 0
    n_both = 0
    n_pass_nfc = 0
    n_pass_nfd = 0
    for record in records:
        if record.get("ids_identical") is not False:
            continue
        cell_nfc = _canonicalform_cell_get(record, "nfc")
        cell_nfd = _canonicalform_cell_get(record, "nfd")
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
        "mcnemar_p": _canonicalform_mcnemar_p(n_nfc_only, n_nfd_only),
    }


def _canonicalform_classify(n_paired: int, nfc_only: int, nfd_only: int) -> str:
    """Mirror of canonicalform.classify_canonical_form."""
    d = nfc_only + nfd_only
    if n_paired < _CANONICALFORM_MIN_PAIRED_EVALUABLE:
        return "insufficient"
    if d == 0:
        return "no_discordance_observed"
    if nfc_only == nfd_only and nfc_only >= _CANONICALFORM_MIN_DISCORDANT:
        return "discordant_balanced"
    if max(nfc_only, nfd_only) >= _CANONICALFORM_MIN_DISCORDANT:
        return "discordant_asymmetric"
    return "below_threshold"


def _canonicalform_falsified(classification: str) -> bool:
    """Mirror of canonicalform.invariance_falsified."""
    return classification in ("discordant_balanced", "discordant_asymmetric")


def _canonicalform_aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Mirror of canonicalform.aggregate_candidate."""
    arms: dict[str, Any] = {}
    for key in _CANONICALFORM_ARM_KEYS:
        n_completed = 0
        n_pass = 0
        for record in records:
            cell = _canonicalform_cell_get(record, key)
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
        cell_nfc = _canonicalform_cell_get(record, "nfc")
        cell_nfd = _canonicalform_cell_get(record, "nfd")
        if cell_nfc is None or cell_nfd is None:
            continue
        if bool(cell_nfc["verdict"]) != bool(cell_nfd["verdict"]):
            n_same_ids_flips += 1
    headline_raw = _canonicalform_pair_block_divergent(records)
    classification = _canonicalform_classify(
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
        "invariance_falsified": _canonicalform_falsified(classification),
        "n_cases": len(records),
        "n_incomplete": sum(
            1
            for record in records
            if _canonicalform_cell_get(record, "nfc") is None
            or _canonicalform_cell_get(record, "nfd") is None
        ),
    }


def _canonicalform_shared_fixtures(
    records_by_candidate: dict[str, list[dict[str, Any]]],
) -> frozenset[str]:
    populations: list[frozenset[str]] = []
    for records in records_by_candidate.values():
        populations.append(frozenset(str(r.get("fixture")) for r in records))
    if not populations:
        return frozenset()
    return frozenset.intersection(*populations)


def _canonicalform_selection_over(
    records_by_candidate: dict[str, list[dict[str, Any]]],
    key: str,
    fixtures: frozenset[str],
) -> dict[str, Any] | None:
    """Mirror of canonicalform._selection_over."""
    completed_sets: list[frozenset[str]] = []
    all_pass: dict[str, bool | None] = {}
    for name, records in records_by_candidate.items():
        completed = frozenset(
            str(record["fixture"])
            for record in records
            if record["fixture"] in fixtures and _canonicalform_cell_get(record, key) is not None
        )
        completed_sets.append(completed)
        if not completed:
            return None
        all_pass[name] = all(
            _canonicalform_verdict(record, key)
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


def _canonicalform_arm_selection(
    records_by_candidate: dict[str, list[dict[str, Any]]],
    arm: str,
) -> dict[str, Any]:
    """Mirror of canonicalform.arm_selection."""
    shared = _canonicalform_shared_fixtures(records_by_candidate)
    own = _canonicalform_selection_over(records_by_candidate, arm, shared)
    if own is None:
        return {
            "decision": "unavailable",
            "detail": "selection_withheld_mismatched_sets",
            "winner": None,
            "withheld": True,
            "flip": None,
        }
    if arm == _CANONICALFORM_BASELINE_ARM:
        flip: bool | None = False
    else:
        completed = frozenset(
            str(record["fixture"])
            for records in records_by_candidate.values()
            for record in records
            if record["fixture"] in shared and _canonicalform_cell_get(record, arm) is not None
        )
        baseline = _canonicalform_selection_over(
            records_by_candidate, _CANONICALFORM_BASELINE_ARM, completed
        )
        flip = None if baseline is None else own != baseline
    return {
        "decision": own["decision"],
        "detail": "",
        "winner": own["winner"],
        "withheld": False,
        "flip": flip,
    }


def _canonicalform_winner_headline(conditions: dict[str, Any]) -> dict[str, Any]:
    """Mirror of canonicalform.winner_headline."""
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


def _canonicalform_cell_ok(cell: Any, expected_digest: str) -> bool:
    """One cell's shape and the verdict binding
    ``verdict == (answer_digest == expected_digest) and
    boundary_found``. The expected digest is the record's,
    itself checked against the mirror-regenerated case."""
    if type(cell) is not dict or set(cell) != _CANONICALFORM_CELL_FIELDS:
        return False
    if type(cell["verdict"]) is not bool or type(cell["boundary_found"]) is not bool:
        return False
    if cell["boundary_text_match"] is not None and type(cell["boundary_text_match"]) is not bool:
        return False
    if type(cell["input_tokens"]) is not int or cell["input_tokens"] < 0:
        return False
    if not _canonicalform_digest_ok(cell["answer_digest"]):
        return False
    verdict = (cell["answer_digest"] == expected_digest) and cell["boundary_found"]
    return bool(cell["verdict"] == verdict)


def _canonicalform_recompute(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Recompute a canonical-form payload from its ground records.

    The validator carries its own generator mirror, templates,
    pools, and aggregators (no imports from canonicalform.py):
    the protocol is rebuilt from frozen scalars plus the
    payload's seed and backend fingerprint and compared whole;
    the case set is regenerated from the seed and every record's
    construction is compared against it -- including, unlike
    placement, the TREATMENT itself: each record's NFC/NFD
    content digests must equal the digests of the mirror's own
    NFC/NFD derivations, the two digests must differ, and the
    id-divergence flags must be consistent with the arm token
    counts (ids_identical implies equal counts; the delta is
    the count difference). Every cell's verdict is re-derived
    from its digest identity; every completed cell's
    ``input_tokens`` must equal the record's arm token count
    for that arm; the aggregates, per-arm selections, and
    winner headline are recomputed and compared.
    """
    import json as _json

    protocol = payload.get("protocol")
    if type(protocol) is not dict:
        return None
    seed = protocol.get("seed")
    backend_fingerprint = protocol.get("backend_fingerprint")
    if type(seed) is not int or isinstance(seed, bool) or type(backend_fingerprint) is not str:
        return None
    if not _recomputed_equal(protocol, _canonicalform_protocol(seed, backend_fingerprint)):
        return None
    candidates = payload.get("candidates")
    if type(candidates) is not dict or not candidates:
        return None
    generated = _canonicalform_generate(seed)
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
        if set(entry) != _CANONICALFORM_AGGREGATE_KEYS | {"status", "detail", "records"}:
            return None
        records = entry["records"]
        if type(records) is not list or len(records) != len(generated):
            return None
        total_records += len(records)
        records_bytes += len(
            _json.dumps(records, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        )
        if records_bytes > _CANONICALFORM_MAX_RECORDS_BYTES:
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
            expected_digest = _canonicalform_digest(str(case["value"]))
            if record["expected_digest"] != expected_digest:
                return None
            if record["nfc_content_digest"] != _canonicalform_digest(case["nfc_content"]):
                return None
            if record["nfd_content_digest"] != _canonicalform_digest(case["nfd_content"]):
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
            if type(cells) is not dict or set(cells) != set(_CANONICALFORM_ARM_KEYS):
                return None
            if type(details) is not dict or set(details) != set(_CANONICALFORM_ARM_KEYS):
                return None
            for arm in _CANONICALFORM_ARM_KEYS:
                cell = cells[arm]
                detail = details[arm]
                if cell is None:
                    if detail not in _CANONICALFORM_DETAILS:
                        return None
                else:
                    if detail != "":
                        return None
                    if not _canonicalform_cell_ok(cell, expected_digest):
                        return None
                    arm_tokens = nfc_tokens if arm == "nfc" else nfd_tokens
                    if cell["input_tokens"] != arm_tokens:
                        return None
        raw_records[name] = records
        aggregates[name] = _canonicalform_aggregate(records)
    if total_records > _CANONICALFORM_MAX_GROUND_RECORDS:
        return None
    if not raw_records:
        return None
    conditions = {
        arm: _canonicalform_arm_selection(raw_records, arm) for arm in _CANONICALFORM_ARM_KEYS
    }
    return {
        "aggregates": aggregates,
        "conditions": conditions,
        "selection": {
            "nfc": conditions["nfc"],
            "nfd": conditions["nfd"],
            "winner": _canonicalform_winner_headline(conditions),
        },
        "n_candidates": len(candidates),
        "n_cases": len(generated),
    }


def _valid_canonicalform(payload: Any) -> bool:
    """Structural check for a ``canonical_form`` payload (recompute standard)."""
    if type(payload) is not dict or payload.get("available") is not True:
        return False
    if set(payload) != {
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
    if payload.get("detail") != "":
        return False
    recomputed = _canonicalform_recompute(payload)
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
    if not _recomputed_equal(payload.get("selection"), recomputed["selection"]):
        return False
    if not _recomputed_equal(payload.get("n_candidates"), recomputed["n_candidates"]):
        return False
    return _recomputed_equal(payload.get("n_cases"), recomputed["n_cases"])


def diff_canonicalform(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``canonical_form`` probe payloads; deterministic.

    Answers "did the canonical-form story change between runs?":
    per-candidate classification and falsification drift, the
    directional discordant counts, the ids-identical share and
    same-ids flip drift, the winner-headline drift, per-arm
    selection drift, candidates added/removed, and protocol /
    generator / backend drift (any of which marks the comparison
    not comparable, with the drift still listed -- the generated
    set is the treatment, so a changed generator is protocol
    drift). A side without a valid payload is reported plainly.
    """
    old_ok = _valid_canonicalform(old)
    new_ok = _valid_canonicalform(new)
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
    cf_protocol_drift = old["protocol"].get("fingerprint") != new["protocol"].get("fingerprint")
    cf_generator_drift = old["protocol"].get("ground_sha256") != new["protocol"].get(
        "ground_sha256"
    )
    cf_backend_drift = old["protocol"].get("backend_fingerprint") != new["protocol"].get(
        "backend_fingerprint"
    )
    cf_comparable = not cf_protocol_drift and not cf_generator_drift and not cf_backend_drift
    old_candidates = old["candidates"]
    new_candidates = new["candidates"]
    cf_candidates_added = sorted(set(new_candidates) - set(old_candidates))
    cf_candidates_removed = sorted(set(old_candidates) - set(new_candidates))
    cf_candidate_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_candidates) & set(new_candidates)):
        old_entry = old_candidates[name]
        new_entry = new_candidates[name]
        if old_entry.get("status") != "available" or new_entry.get("status") != "available":
            if old_entry.get("status") != new_entry.get("status"):
                cf_candidate_drift.append(
                    {
                        "candidate": name,
                        "metric": "status",
                        "old": old_entry.get("status"),
                        "new": new_entry.get("status"),
                    }
                )
            continue
        cf_metrics: list[tuple[str, Any, Any]] = [
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
        for metric, old_value, new_value in cf_metrics:
            if old_value != new_value:
                cf_candidate_drift.append(
                    {
                        "candidate": name,
                        "metric": metric,
                        "old": old_value,
                        "new": new_value,
                    }
                )
    cf_condition_drift: list[dict[str, Any]] = []
    for arm in _CANONICALFORM_ARM_KEYS:
        if old["conditions"][arm] != new["conditions"][arm]:
            cf_condition_drift.append(
                {
                    "condition": arm,
                    "old": old["conditions"][arm],
                    "new": new["conditions"][arm],
                }
            )
    cf_selection_drift: list[dict[str, Any]] = []
    if old["selection"] != new["selection"]:
        cf_selection_drift.append(
            {
                "selection": "winner_headline",
                "old": old["selection"]["winner"],
                "new": new["selection"]["winner"],
            }
        )
    result["candidate_drift"] = cf_candidate_drift
    result["condition_drift"] = cf_condition_drift
    result["selection_drift"] = cf_selection_drift
    result["candidates_added"] = cf_candidates_added
    result["candidates_removed"] = cf_candidates_removed
    result["protocol_drift"] = cf_protocol_drift
    result["generator_drift"] = cf_generator_drift
    result["backend_drift"] = cf_backend_drift
    result["comparable"] = cf_comparable
    result["changed"] = bool(
        cf_candidate_drift
        or cf_condition_drift
        or cf_selection_drift
        or cf_candidates_added
        or cf_candidates_removed
        or cf_protocol_drift
        or cf_generator_drift
        or cf_backend_drift
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
    instructiondistance.py anywhere in this mirror)."""
    import hashlib as _hashlib

    return _hashlib.sha256(text.encode("utf-8")).hexdigest()


def _instructiondistance_digest_ok(value: Any) -> bool:
    """Whether a stored digest is a well-formed SHA-256 hex
    digest (this probe's answer digests are never None: a
    completed cell always carries a decoded span)."""
    if type(value) is not str or len(value) != 64:
        return False
    return all(ch in "0123456789abcdef" for ch in value)


def _instructiondistance_run_digest_ok(value: Any) -> bool:
    """The first-digit-run digest field: "" (no ASCII digit
    run, amendment B3) or a well-formed digest."""
    if value == "":
        return True
    return _instructiondistance_digest_ok(value)


def _instructiondistance_derive(seed: int, index: int, slot: int, counter: int) -> int:
    """Mirror of instructiondistance._derive (frozen material
    string)."""
    import hashlib as _hashlib

    material = f"{seed}|instructiondistance|{index}|{slot}|{counter}"
    return int.from_bytes(_hashlib.sha256(material.encode("ascii")).digest()[:8], "big")


def _instructiondistance_generate(seed: int) -> list[dict[str, Any]]:
    """Mirror of instructiondistance.generate_cases: the
    frozen case set for a seed, regenerated from the
    protocol's own seed so a fabricated set is rejected (the
    numform F2 lesson)."""
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
    import json as _json

    canonical = _json.dumps(
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
    import json as _json

    rows = [
        [case["fixture_id"], case["topic"], case["value"]]
        for case in _instructiondistance_generate(seed)
    ]
    canonical = _json.dumps(rows, separators=(",", ":"), ensure_ascii=True)
    return _instructiondistance_digest(canonical)


def _instructiondistance_protocol(seed: int, backend_fingerprint: str) -> dict[str, Any]:
    """The instruction-distance protocol rebuilt from this
    module's own literals plus the payload's seed and backend
    fingerprint, fingerprint included (compared whole)."""
    import json as _json

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
    canonical = _json.dumps(protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    protocol["fingerprint"] = _instructiondistance_digest(canonical)
    return protocol


def _instructiondistance_cell_get(record: dict[str, Any], key: str) -> dict[str, Any] | None:
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
    forged digit_count cannot move a class)."""
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
    trusted as verdict inputs)."""
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
    import json as _json

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
            _json.dumps(records, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
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
    not recomputable from records."""
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
_HISTORYFRAME_CLASSIFICATIONS = (
    "insufficient",
    "no_discordance_observed",
    "discordant_balanced",
    "discordant_asymmetric",
    "below_threshold",
)
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
    historyframe.py anywhere in this mirror)."""
    import hashlib as _hashlib

    return _hashlib.sha256(text.encode("utf-8")).hexdigest()


def _historyframe_digest_ok(value: Any) -> bool:
    """Whether a stored digest is a well-formed SHA-256 hex
    digest."""
    if type(value) is not str or len(value) != 64:
        return False
    return all(ch in "0123456789abcdef" for ch in value)


def _historyframe_mcnemar_p(a: int, b: int) -> float | None:
    """Mirror of confidentiality.mcnemar_exact_p: two-sided
    exact McNemar p; None when there are no discordant pairs."""
    import math as _math

    n = a + b
    if n == 0:
        return None
    k = min(a, b)
    tail: float = sum(_math.comb(n, i) for i in range(k + 1)) / (2**n)
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
    split under independent fair arms)."""
    import math as _math

    if type(n) is not int or n < _HISTORYFRAME_MIN_N:
        return 0.0
    total = 0.0
    for d in range(_HISTORYFRAME_MIN_D, n + 1):
        p_d = _math.comb(n, d) * (0.5**n)
        for a in range(d + 1):
            b = d - a
            if a == b:
                continue
            p = _historyframe_mcnemar_p(a, b)
            if p is not None and p < 0.05:
                total += p_d * _math.comb(d, a) * (0.5**d)
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
    the frozen transcript."""
    import json as _json

    canonical = _json.dumps(
        [[role, content] for role, content in _HISTORYFRAME_TURNS],
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return _historyframe_digest(canonical)


def _historyframe_protocol(
    seed: int, backend_fingerprint: str, fixture_ids: list[str]
) -> dict[str, Any]:
    """Mirror of historyframe.protocol_object, fingerprint
    included (compared whole)."""
    import json as _json

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
    canonical = _json.dumps(protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    protocol["fingerprint"] = _historyframe_digest(canonical)
    return protocol


def _historyframe_cell_get(record: dict[str, Any], key: str) -> dict[str, Any] | None:
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
    carried flag or delta is never trusted."""
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
    inapplicable."""
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
    None (no answer string was decoded)."""
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
    import json as _json

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
            _json.dumps(records, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
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
    block -- they are shape-checked only."""
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
    not recomputable from records."""
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


_ROTATION_FACT_TEMPLATE = "The {topic} access code is {value}."
_ROTATION_QUESTION_TEMPLATE = "What is the {topic} access code?"
_ROTATION_TOPICS = (
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
    "greenhouse wing",
    "ferry terminal",
    "signal cabin",
    "customs shed",
    "observatory dome",
    "print workshop",
    "harbor station",
    "tunnel control",
    "solar array",
    "wind turbine",
    "clock tower",
    "grain silo",
    "kiln workshop",
    "sorting office",
    "parcel depot",
    "marble quarry",
)
_ROTATION_WORDS = (
    "ochre",
    "velvet",
    "cobalt",
    "saffron",
    "tundra",
    "zephyr",
    "onyx",
    "pepper",
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
    "zenith",
    "almond",
    "birch",
    "copper",
    "drizzle",
    "glacier",
    "mauve",
    "ivory",
    "juniper",
    "linden",
    "meteor",
    "nickel",
    "opal",
    "pyrite",
    "quill",
    "russet",
    "sorrel",
    "topaz",
    "flint",
    "jasper",
    "nutmeg",
    "rowan",
    "timber",
    "umber",
    "dune",
)
_ROTATION_ROTATION_ARMS = tuple(f"rot_{r}" for r in range(5))
_ROTATION_MIRROR_ARMS = ("mir_1", "mir_2", "mir_3", "mir_4")
_ROTATION_SPAN_ARMS = _ROTATION_ROTATION_ARMS + _ROTATION_MIRROR_ARMS
_ROTATION_ARM_KEYS = _ROTATION_SPAN_ARMS + ("absent", "closed_book")
_ROTATION_CONDITION_ARMS = _ROTATION_ROTATION_ARMS + ("mir_4",)
_ROTATION_BASELINE_ARM = "rot_0"
_ROTATION_CONTROL_ARMS = ("rot_0",) + _ROTATION_MIRROR_ARMS
_ROTATION_CASE_COUNT = 20
_ROTATION_MIN_HEADLINE_POPULATION = 6
_ROTATION_MIN_CONTROL_POPULATION = 6
_ROTATION_MAX_GROUND_RECORDS = 24 * _ROTATION_CASE_COUNT
_ROTATION_GENERATION_BUDGET = len(_ROTATION_ARM_KEYS) * _ROTATION_MAX_GROUND_RECORDS
_ROTATION_MAX_RECORDS_BYTES = 12 * 1024 * 1024
_ROTATION_CELL_FIELDS = {
    "verdict",
    "boundary_found",
    "boundary_text_match",
    "input_tokens",
    "answer_digest",
}
_ROTATION_DETAILS = ("arm_context_exceeded", "arm_evaluation_failed", "arm_screened")
_ROTATION_AGGREGATE_KEYS = {
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
}
_ROTATION_CONSTRUCTION_RULE_TEXT = (
    "rotation-construction-v1: a span arm's inventory digest is the "
    "SHA-256 of the comma-joined decimal token ids of the arm's full "
    "render, sorted ascending; an arm is id_exact iff its inventory "
    "digest and its token count both equal rot_0's, and rot_0 is "
    "id_exact by definition. The classification is a claim about the "
    "token inventory only; segmentation at the seams is not verified."
)
_ROTATION_SCREENING_RULE_TEXT = (
    "rotation-screen-v1: stage-level, the frozen texts (templates "
    "with slots emptied, topics, words) are audited for special-token "
    "literals against every candidate format, and every candidate's "
    "decoded boundary text is checked for containment in the frozen "
    "texts; a hit fails the whole stage closed before any generation. "
    "Per case and candidate, if any block text of the case contains "
    "the candidate's decoded boundary text, every cell of the case is "
    "arm_screened (the record is boundary-screened and enters no "
    "population). The closed-book screen is derived, never stored: a "
    "record whose closed-book cell passed is excluded from its "
    "candidate's rotation populations."
)


def _rotation_digest(text: str) -> str:
    """SHA-256 hex digest (local copy; no imports from
    rotation.py anywhere in this mirror)."""
    import hashlib as _hashlib

    return _hashlib.sha256(text.encode("utf-8")).hexdigest()


def _rotation_digest_ok(value: Any) -> bool:
    """Whether a stored digest is None or a well-formed SHA-256
    hex digest (fails closed on lone surrogates, the cycle-190
    discipline)."""
    if value is None:
        return True
    if type(value) is not str or len(value) != 64:
        return False
    return all(ch in "0123456789abcdef" for ch in value)


def _rotation_derive(seed: int, index: int, slot: int, counter: int) -> int:
    """Mirror of rotation._derive (frozen material string)."""
    import hashlib as _hashlib

    material = f"{seed}|rotation|{index}|{slot}|{counter}"
    return int.from_bytes(_hashlib.sha256(material.encode("ascii")).digest()[:8], "big")


def _rotation_generate(seed: int) -> list[dict[str, Any]]:
    """Mirror of rotation.generate_cases: the frozen case set
    for a seed, regenerated from the protocol's own seed so a
    fabricated set is rejected (the numform F2 lesson)."""
    cases: list[dict[str, Any]] = []
    used_asked: set[int] = set()
    seen_asked_values: set[str] = set()
    for index in range(_ROTATION_CASE_COUNT):
        counter = 0
        while True:
            asked_index = _rotation_derive(seed, index, 20, counter) % len(_ROTATION_TOPICS)
            if asked_index not in used_asked:
                break
            counter += 1
        used_asked.add(asked_index)
        topics = [_ROTATION_TOPICS[asked_index]]
        chosen = {_ROTATION_TOPICS[asked_index]}
        for slot in (21, 22, 23, 24, 25):
            counter = 0
            while True:
                topic = _ROTATION_TOPICS[
                    _rotation_derive(seed, index, slot, counter) % len(_ROTATION_TOPICS)
                ]
                if topic not in chosen:
                    break
                counter += 1
            chosen.add(topic)
            topics.append(topic)
        counters = [0] * 6
        while True:
            values = []
            for block in range(6):
                word = _ROTATION_WORDS[
                    _rotation_derive(seed, index, 2 * block, counters[block]) % len(_ROTATION_WORDS)
                ]
                number = 1000 + _rotation_derive(seed, index, 2 * block + 1, counters[block]) % 9000
                values.append(f"{word}-{number}")
            if len(set(values)) == 6 and values[0] not in seen_asked_values:
                break
            counters = [c + 1 for c in counters]
        seen_asked_values.add(values[0])
        cases.append(
            {
                "fixture_id": f"rotation-{index:02d}",
                "topic": topics[0],
                "topics": topics,
                "values": values,
            }
        )
    return cases


def _rotation_case_material_sha256() -> str:
    """Mirror of rotation.case_material_sha256."""
    import json as _json

    canonical = _json.dumps(
        [
            _ROTATION_FACT_TEMPLATE,
            _ROTATION_QUESTION_TEMPLATE,
            list(_ROTATION_TOPICS),
            list(_ROTATION_WORDS),
        ],
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return _rotation_digest(canonical)


def _rotation_ground_sha256(seed: int) -> str:
    """Mirror of rotation.ground_sha256."""
    import json as _json

    rows = [
        [case["fixture_id"], list(case["topics"]), list(case["values"])]
        for case in _rotation_generate(seed)
    ]
    canonical = _json.dumps(rows, separators=(",", ":"), ensure_ascii=True)
    return _rotation_digest(canonical)


def _rotation_protocol(seed: int, backend_fingerprint: str) -> dict[str, Any]:
    """The rotation protocol rebuilt from this module's own
    literals plus the payload's seed and backend fingerprint,
    fingerprint included (compared whole)."""
    import json as _json

    protocol: dict[str, Any] = {
        "version": 1,
        "arm_keys": list(_ROTATION_ARM_KEYS),
        "span_arms": list(_ROTATION_SPAN_ARMS),
        "condition_arms": list(_ROTATION_CONDITION_ARMS),
        "baseline_arm": _ROTATION_BASELINE_ARM,
        "case_count": _ROTATION_CASE_COUNT,
        "case_material_sha256": _rotation_case_material_sha256(),
        "ground_sha256": _rotation_ground_sha256(seed),
        "construction_rule_sha256": _rotation_digest(_ROTATION_CONSTRUCTION_RULE_TEXT),
        "screening_rule_sha256": _rotation_digest(_ROTATION_SCREENING_RULE_TEXT),
        "min_headline_population": _ROTATION_MIN_HEADLINE_POPULATION,
        "min_control_population": _ROTATION_MIN_CONTROL_POPULATION,
        "generation_budget": _ROTATION_GENERATION_BUDGET,
        "max_ground_records": _ROTATION_MAX_GROUND_RECORDS,
        "max_records_bytes": _ROTATION_MAX_RECORDS_BYTES,
        "seed": seed,
        "backend_fingerprint": backend_fingerprint,
    }
    canonical = _json.dumps(protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    protocol["fingerprint"] = _rotation_digest(canonical)
    return protocol


def _rotation_cell_get(record: dict[str, Any], key: str) -> dict[str, Any] | None:
    cells = record.get("cells")
    if not isinstance(cells, dict):
        return None
    cell = cells.get(key)
    return cell if isinstance(cell, dict) else None


def _rotation_verdict(record: dict[str, Any], key: str) -> bool:
    cell = _rotation_cell_get(record, key)
    return bool(cell is not None and cell["verdict"])


def _rotation_id_exact(record: dict[str, Any], arm: str) -> bool:
    """Mirror of rotation.arm_id_exact (derived, never stored)."""
    if arm == "rot_0":
        return True
    digests = record.get("arm_inventory_digests")
    tokens = record.get("arm_tokens")
    if not isinstance(digests, dict) or not isinstance(tokens, dict):
        return False
    return digests.get(arm) == digests.get("rot_0") and tokens.get(arm) == tokens.get("rot_0")


def _rotation_screened(record: dict[str, Any]) -> bool:
    cell = _rotation_cell_get(record, "closed_book")
    return cell is not None and bool(cell["verdict"])


def _rotation_boundary_screened(record: dict[str, Any]) -> bool:
    details = record.get("cell_details")
    if not isinstance(details, dict):
        return False
    return all(details.get(arm) == "arm_screened" for arm in _ROTATION_ARM_KEYS)


def _rotation_classify(
    n_headline: int,
    n_flip_any: int,
    n_all_pass: int,
    n_all_fail: int,
    n_mirror_paired: int,
    n_flips_paired: int,
    n_flips_mirror: int,
) -> str:
    """Mirror of rotation.classify_rotation."""
    if n_headline < _ROTATION_MIN_HEADLINE_POPULATION:
        return "insufficient"
    if n_flip_any == 0 and n_all_fail > n_all_pass:
        return "joint_collapse_dominant"
    if n_flip_any == 0:
        return "invariant_observed"
    if n_mirror_paired < _ROTATION_MIN_CONTROL_POPULATION:
        return "control_set_too_small"
    if n_flips_paired <= n_flips_mirror:
        return "within_mirror_control"
    return "rotation_sensitive"


def _rotation_falsified(n_headline: int, n_flip_any: int) -> bool:
    """Mirror of rotation.invariance_falsified."""
    return n_headline >= _ROTATION_MIN_HEADLINE_POPULATION and n_flip_any >= 1


def _rotation_aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Mirror of rotation.aggregate_candidate."""

    def rate(n_pass: int, n: int) -> float | None:
        return (n_pass / n) if n else None

    live = [r for r in records if not _rotation_screened(r) and not _rotation_boundary_screened(r)]
    arms: dict[str, Any] = {}
    for key in _ROTATION_SPAN_ARMS:
        n_completed = 0
        n_pass = 0
        for record in live:
            cell = _rotation_cell_get(record, key)
            if cell is None:
                continue
            n_completed += 1
            if cell["verdict"]:
                n_pass += 1
        arms[key] = {
            "n_completed": n_completed,
            "n_pass": n_pass,
            "rate": rate(n_pass, n_completed),
        }
    cb_completed = 0
    cb_pass = 0
    for record in records:
        cell = _rotation_cell_get(record, "closed_book")
        if cell is None:
            continue
        cb_completed += 1
        if cell["verdict"]:
            cb_pass += 1
    absent_evaluable = 0
    absent_false = 0
    for record in live:
        cell = _rotation_cell_get(record, "absent")
        if cell is None:
            continue
        absent_evaluable += 1
        if cell["verdict"]:
            absent_false += 1
    headline_records = [
        record
        for record in live
        if all(_rotation_cell_get(record, arm) is not None for arm in _ROTATION_ROTATION_ARMS)
        and all(_rotation_id_exact(record, arm) for arm in _ROTATION_ROTATION_ARMS)
    ]
    n_headline = len(headline_records)
    n_all_pass = 0
    n_all_fail = 0
    n_r0_pass = 0
    n_flip_from_pass = 0
    slot_pass: dict[str, int] = {str(s): 0 for s in range(5)}
    for record in headline_records:
        verdicts = [_rotation_verdict(record, arm) for arm in _ROTATION_ROTATION_ARMS]
        if all(verdicts):
            n_all_pass += 1
        elif not any(verdicts):
            n_all_fail += 1
        if verdicts[0]:
            n_r0_pass += 1
            if not all(verdicts):
                n_flip_from_pass += 1
        for r, verdict in enumerate(verdicts):
            if verdict:
                slot_pass[str((5 - r) % 5)] += 1
    n_invariant = n_all_pass + n_all_fail
    n_flip_any = n_headline - n_invariant
    rotation_present = 0
    rotation_exact = 0
    mirror_present = 0
    mirror_exact = 0
    for record in live:
        for arm in _ROTATION_ROTATION_ARMS:
            if _rotation_cell_get(record, arm) is not None:
                rotation_present += 1
                if _rotation_id_exact(record, arm):
                    rotation_exact += 1
        for arm in _ROTATION_MIRROR_ARMS:
            if _rotation_cell_get(record, arm) is not None:
                mirror_present += 1
                if _rotation_id_exact(record, arm):
                    mirror_exact += 1
    mirror_records = [
        record
        for record in headline_records
        if all(_rotation_cell_get(record, arm) is not None for arm in _ROTATION_MIRROR_ARMS)
        and all(_rotation_id_exact(record, arm) for arm in _ROTATION_MIRROR_ARMS)
    ]
    n_mirror_paired = len(mirror_records)
    n_flips_paired = 0
    n_flips_mirror = 0
    for record in mirror_records:
        rotation_verdicts = [_rotation_verdict(record, arm) for arm in _ROTATION_ROTATION_ARMS]
        if len(set(rotation_verdicts)) > 1:
            n_flips_paired += 1
        control_verdicts = [_rotation_verdict(record, arm) for arm in _ROTATION_CONTROL_ARMS]
        if len(set(control_verdicts)) > 1:
            n_flips_mirror += 1
    reversal_records = [
        record
        for record in headline_records
        if _rotation_cell_get(record, "mir_4") is not None and _rotation_id_exact(record, "mir_4")
    ]
    n_reversal_paired = len(reversal_records)
    n_discordant = sum(
        1
        for record in reversal_records
        if _rotation_verdict(record, "mir_4") != _rotation_verdict(record, "rot_0")
    )
    n_live = len(live)
    return {
        "arms": arms,
        "closed_book": {
            "n_completed": cb_completed,
            "n_pass": cb_pass,
            "pass_rate": rate(cb_pass, cb_completed),
        },
        "absent": {
            "n_evaluable": absent_evaluable,
            "n_false_pass": absent_false,
            "rate": rate(absent_false, absent_evaluable),
        },
        "headline": {
            "n_headline": n_headline,
            "n_all_pass": n_all_pass,
            "n_all_fail": n_all_fail,
            "n_invariant": n_invariant,
            "invariance_rate": rate(n_invariant, n_headline),
            "n_flip_any": n_flip_any,
            "n_r0_pass": n_r0_pass,
            "n_flip_from_pass": n_flip_from_pass,
            "flip_rate_from_pass": rate(n_flip_from_pass, n_r0_pass),
            "evaluable_share": rate(n_headline, n_live),
            "id_exact_share": rate(rotation_exact, rotation_present),
        },
        "slot_curve": {
            "slots": slot_pass,
            "n_headline": n_headline,
        },
        "mirror": {
            "n_paired": n_mirror_paired,
            "n_flips_paired": n_flips_paired,
            "n_flips_mirror": n_flips_mirror,
            "id_exact_share": rate(mirror_exact, mirror_present),
        },
        "reversal": {
            "n_paired": n_reversal_paired,
            "n_discordant": n_discordant,
            "discordance_rate": rate(n_discordant, n_reversal_paired),
        },
        "classification": _rotation_classify(
            n_headline,
            n_flip_any,
            n_all_pass,
            n_all_fail,
            n_mirror_paired,
            n_flips_paired,
            n_flips_mirror,
        ),
        "invariance_falsified": _rotation_falsified(n_headline, n_flip_any),
        "n_fixtures": len(records),
        "n_screened": sum(1 for record in records if _rotation_screened(record)),
        "n_boundary_screened": sum(1 for record in records if _rotation_boundary_screened(record)),
        "n_closed_book_incomplete": sum(
            1
            for record in records
            if not _rotation_boundary_screened(record)
            and _rotation_cell_get(record, "closed_book") is None
        ),
        "n_live": n_live,
    }


def _rotation_selection_over(
    records_by_candidate: dict[str, list[dict[str, Any]]],
    key: str,
    fixtures: frozenset[str],
) -> dict[str, Any] | None:
    """Mirror of rotation._selection_over."""
    completed_sets: list[frozenset[str]] = []
    all_pass: dict[str, bool | None] = {}
    for name, records in records_by_candidate.items():
        completed = frozenset(
            str(record["fixture"])
            for record in records
            if record["fixture"] in fixtures and _rotation_cell_get(record, key) is not None
        )
        completed_sets.append(completed)
        if not completed:
            return None
        all_pass[name] = all(
            _rotation_verdict(record, key) for record in records if record["fixture"] in completed
        )
    if len(set(completed_sets)) > 1:
        return None
    passers = [name for name, ok in all_pass.items() if ok]
    if len(passers) == 1:
        return {"decision": "selected", "winner": passers[0]}
    if len(passers) > 1:
        return {"decision": "ambiguous", "winner": None}
    return {"decision": "no_candidate", "winner": None}


def _rotation_arm_selection(
    records_by_candidate: dict[str, list[dict[str, Any]]],
    arm: str,
) -> dict[str, Any]:
    """Mirror of rotation.arm_selection."""
    populations: list[frozenset[str]] = []
    for records in records_by_candidate.values():
        populations.append(
            frozenset(
                str(r.get("fixture"))
                for r in records
                if not _rotation_screened(r) and not _rotation_boundary_screened(r)
            )
        )
    shared: frozenset[str] = frozenset.intersection(*populations) if populations else frozenset()
    by_fixture: dict[str, dict[str, dict[str, Any]]] = {
        name: {str(record.get("fixture")): record for record in records}
        for name, records in records_by_candidate.items()
    }
    gated = frozenset(
        fixture
        for fixture in shared
        if all(_rotation_id_exact(by_fixture[name][fixture], arm) for name in by_fixture)
    )
    own = _rotation_selection_over(records_by_candidate, arm, gated)
    if own is None:
        return {
            "decision": "unavailable",
            "detail": "selection_withheld_mismatched_sets",
            "winner": None,
            "withheld": True,
            "flip": None,
        }
    if arm == _ROTATION_BASELINE_ARM:
        flip: bool | None = False
    else:
        completed = frozenset(
            str(record["fixture"])
            for records in records_by_candidate.values()
            for record in records
            if record["fixture"] in gated and _rotation_cell_get(record, arm) is not None
        )
        baseline = _rotation_selection_over(records_by_candidate, _ROTATION_BASELINE_ARM, completed)
        flip = None if baseline is None else own != baseline
    return {
        "decision": own["decision"],
        "detail": "",
        "winner": own["winner"],
        "withheld": False,
        "flip": flip,
    }


def _rotation_cell_ok(cell: Any, expected_digest: str) -> bool:
    """One cell's shape and the verdict binding
    ``verdict == (answer_digest == expected_digest) and
    boundary_found``. The expected digest is the record's,
    itself checked against the mirror-regenerated case."""
    if type(cell) is not dict or set(cell) != _ROTATION_CELL_FIELDS:
        return False
    if type(cell["verdict"]) is not bool or type(cell["boundary_found"]) is not bool:
        return False
    if cell["boundary_text_match"] is not None and type(cell["boundary_text_match"]) is not bool:
        return False
    if type(cell["input_tokens"]) is not int or cell["input_tokens"] < 0:
        return False
    if not _rotation_digest_ok(cell["answer_digest"]):
        return False
    verdict = (cell["answer_digest"] == expected_digest) and cell["boundary_found"]
    return bool(cell["verdict"] == verdict)


def _rotation_recompute(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Recompute a rotation payload from its ground records.

    The validator carries its own generator mirror, templates,
    pools, and aggregators (no imports from rotation.py): the
    protocol is rebuilt from frozen scalars plus the payload's
    seed and backend fingerprint and compared whole; the case
    set is regenerated from the seed and every record's
    construction is compared against it in canonical order;
    every cell's verdict is re-derived from its digest
    identity; every completed cell's ``input_tokens`` must
    equal the record's ``arm_tokens`` for that arm; the
    construction classification is derived from the recorded
    inventory digests, never read from the payload; the
    aggregates and the per-arm selections are recomputed and
    compared.
    """
    import json as _json

    protocol = payload.get("protocol")
    if type(protocol) is not dict:
        return None
    seed = protocol.get("seed")
    backend_fingerprint = protocol.get("backend_fingerprint")
    if type(seed) is not int or isinstance(seed, bool) or type(backend_fingerprint) is not str:
        return None
    if not _recomputed_equal(protocol, _rotation_protocol(seed, backend_fingerprint)):
        return None
    candidates = payload.get("candidates")
    if type(candidates) is not dict or not candidates:
        return None
    generated = _rotation_generate(seed)
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
        if set(entry) != _ROTATION_AGGREGATE_KEYS | {"status", "detail", "records"}:
            return None
        records = entry["records"]
        if type(records) is not list or len(records) != len(generated):
            return None
        total_records += len(records)
        records_bytes += len(
            _json.dumps(records, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        )
        if records_bytes > _ROTATION_MAX_RECORDS_BYTES:
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
            expected_digest = _rotation_digest(str(case["values"][0]))
            if record["expected_digest"] != expected_digest:
                return None
            arm_tokens = record["arm_tokens"]
            if type(arm_tokens) is not dict or set(arm_tokens) != set(_ROTATION_ARM_KEYS):
                return None
            if any(type(v) is not int or v < 0 for v in arm_tokens.values()):
                return None
            digests = record["arm_inventory_digests"]
            if type(digests) is not dict or set(digests) != set(_ROTATION_SPAN_ARMS):
                return None
            if any(
                type(v) is not str or not _rotation_digest_ok(v) or v is None
                for v in digests.values()
            ):
                return None
            cells = record["cells"]
            details = record["cell_details"]
            if type(cells) is not dict or set(cells) != set(_ROTATION_ARM_KEYS):
                return None
            if type(details) is not dict or set(details) != set(_ROTATION_ARM_KEYS):
                return None
            for arm in _ROTATION_ARM_KEYS:
                cell = cells[arm]
                detail = details[arm]
                if cell is None:
                    if detail not in _ROTATION_DETAILS:
                        return None
                else:
                    if detail != "":
                        return None
                    if not _rotation_cell_ok(cell, expected_digest):
                        return None
                    if cell["input_tokens"] != arm_tokens[arm]:
                        return None
        raw_records[name] = records
        aggregates[name] = _rotation_aggregate(records)
    if total_records > _ROTATION_MAX_GROUND_RECORDS:
        return None
    if not raw_records:
        return None
    return {
        "aggregates": aggregates,
        "conditions": {
            arm: _rotation_arm_selection(raw_records, arm) for arm in _ROTATION_CONDITION_ARMS
        },
        "n_candidates": len(candidates),
        "n_cases": len(generated),
    }


def _valid_rotation(payload: Any) -> bool:
    """Structural check for a ``rotation`` payload (recompute standard)."""
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
    recomputed = _rotation_recompute(payload)
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


def diff_rotation(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``rotation`` probe payloads; deterministic.

    Answers "did the rotation story change between runs?":
    per-candidate classification and falsification drift, the
    flip and mirror-control counts, the reversal discordance,
    screening drift, per-arm selection drift, candidates
    added/removed, and protocol / generator / backend drift
    (any of which marks the comparison not comparable, with
    the drift still listed -- the generated set is the
    treatment, so a changed generator is protocol drift). A
    side without a valid payload is reported plainly.
    """
    old_ok = _valid_rotation(old)
    new_ok = _valid_rotation(new)
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
    rotation_protocol_drift = old["protocol"].get("fingerprint") != new["protocol"].get(
        "fingerprint"
    )
    rotation_generator_drift = old["protocol"].get("ground_sha256") != new["protocol"].get(
        "ground_sha256"
    )
    rotation_backend_drift = old["protocol"].get("backend_fingerprint") != new["protocol"].get(
        "backend_fingerprint"
    )
    rotation_comparable = (
        not rotation_protocol_drift and not rotation_generator_drift and not rotation_backend_drift
    )
    old_candidates = old["candidates"]
    new_candidates = new["candidates"]
    rotation_candidates_added = sorted(set(new_candidates) - set(old_candidates))
    rotation_candidates_removed = sorted(set(old_candidates) - set(new_candidates))
    rotation_candidate_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_candidates) & set(new_candidates)):
        old_entry = old_candidates[name]
        new_entry = new_candidates[name]
        if old_entry.get("status") != "available" or new_entry.get("status") != "available":
            if old_entry.get("status") != new_entry.get("status"):
                rotation_candidate_drift.append(
                    {
                        "candidate": name,
                        "metric": "status",
                        "old": old_entry.get("status"),
                        "new": new_entry.get("status"),
                    }
                )
            continue
        rotation_metrics: list[tuple[str, Any, Any]] = [
            (
                "classification",
                old_entry["classification"],
                new_entry["classification"],
            ),
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
            (
                "n_screened",
                old_entry["n_screened"],
                new_entry["n_screened"],
            ),
            (
                "n_boundary_screened",
                old_entry["n_boundary_screened"],
                new_entry["n_boundary_screened"],
            ),
        ]
        for metric, old_value, new_value in rotation_metrics:
            if old_value != new_value:
                rotation_candidate_drift.append(
                    {
                        "candidate": name,
                        "metric": metric,
                        "old": old_value,
                        "new": new_value,
                    }
                )
    rotation_condition_drift: list[dict[str, Any]] = []
    for arm in _ROTATION_CONDITION_ARMS:
        if old["conditions"][arm] != new["conditions"][arm]:
            rotation_condition_drift.append(
                {
                    "condition": arm,
                    "old": old["conditions"][arm],
                    "new": new["conditions"][arm],
                }
            )
    result["candidate_drift"] = rotation_candidate_drift
    result["condition_drift"] = rotation_condition_drift
    result["candidates_added"] = rotation_candidates_added
    result["candidates_removed"] = rotation_candidates_removed
    result["protocol_drift"] = rotation_protocol_drift
    result["generator_drift"] = rotation_generator_drift
    result["backend_drift"] = rotation_backend_drift
    result["comparable"] = rotation_comparable
    result["changed"] = bool(
        rotation_candidate_drift
        or rotation_condition_drift
        or rotation_candidates_added
        or rotation_candidates_removed
        or rotation_protocol_drift
        or rotation_generator_drift
        or rotation_backend_drift
    )
    return result


_DENSITY_FACT_TEMPLATE = "The {topic} signal word is {value}."
_DENSITY_QUESTION_TEMPLATE = "What is the {topic} signal word?"
_DENSITY_CONSTRAINT_TEMPLATES = {
    "include": "Include the word {target} in your answer.",
    "start": "Start your answer with the word {target}.",
    "end": "End your answer with the word {target}.",
    "count": "Use at most {limit} words in your answer.",
}
_DENSITY_LINE_TEMPLATE = "Instruction {position}: {text}"
_DENSITY_TOPICS = (
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
    "greenhouse wing",
    "ferry terminal",
    "signal cabin",
    "customs shed",
    "observatory dome",
    "print workshop",
    "harbor station",
    "tunnel control",
    "solar array",
    "wind turbine",
    "clock tower",
    "grain silo",
    "kiln workshop",
    "sorting office",
    "parcel depot",
    "marble quarry",
)
_DENSITY_WORDS = (
    "ochre",
    "velvet",
    "cobalt",
    "saffron",
    "tundra",
    "zephyr",
    "onyx",
    "pepper",
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
    "zenith",
)
_DENSITY_TARGET_WORDS = (
    "basket",
    "breeze",
    "candle",
    "carpet",
    "delta",
    "emerald",
    "feather",
    "garden",
    "compass",
    "island",
    "jacket",
    "kettle",
    "ladder",
    "meadow",
    "needle",
    "orchard",
    "pencil",
    "quilt",
    "ribbon",
    "bronze",
    "shadow",
    "timber",
    "copper",
    "window",
    "yellow",
    "drizzle",
    "bridge",
    "cloud",
    "drum",
    "eagle",
    "flame",
    "glove",
    "hammer",
    "ivory",
    "jungle",
    "lemon",
    "mirror",
    "night",
    "ocean",
    "palace",
    "quest",
    "river",
    "stone",
    "tiger",
    "urban",
    "valley",
    "winter",
    "xenon",
    "glacier",
    "lantern",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
)
_DENSITY_NEUTRALS = (
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
_DENSITY_CASE_COUNT = 20
_DENSITY_CONSTRAINT_COUNT = 16
_DENSITY_DENSITIES = (1, 2, 4, 8, 16)
_DENSITY_KINDS = ("include", "start", "end", "count")
_DENSITY_COUNT_PARAMS = (8, 12, 16, 24)
_DENSITY_SOLO_ARMS = tuple(f"solo_{j}" for j in range(1, _DENSITY_CONSTRAINT_COUNT + 1))
_DENSITY_ARM_KEYS = _DENSITY_SOLO_ARMS + ("k2", "k4", "k8", "k16") + ("sham",)
_DENSITY_MAX_NEW_TOKENS = 64
_DENSITY_MAX_RECORDS = 21 * _DENSITY_CASE_COUNT * 24
_DENSITY_MAX_PAYLOAD_BYTES = 52 * 1024 * 1024
_DENSITY_MIN_DISCORDANT_CASES = 6
_DENSITY_ALPHA = 0.05
_DENSITY_NEUTRALS_PER_CASE = 15
_DENSITY_RECORD_DETAILS = ("", "arm_evaluation_failed", "arm_context_exceeded")
_DENSITY_RECORD_KEYS = {
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
}
_DENSITY_AGGREGATE_KEYS = {
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
}
_DENSITY_GENERATION_RULE_TEXT = (
    "Cases are derived from the run seed by SHA-256 draws "
    "material=f'{seed}|density|{index}|{slot}|{counter}'. Per case, "
    "slots are drawn in this order: 0 task topic, 1 task value, "
    "10..24 the case's fifteen neutral statements (distinct, in "
    "drawn order), then constraint parameters -- include targets "
    "at slots 30..33 in position order, the shared start target "
    "at slot 34, the shared end target at slot 35, count limits "
    "at slots 40..43 in position order, drawn without "
    "replacement from {8, 12, 16, 24}. A "
    "case's sixteen constraint kinds are fixed by the kind cycle "
    "(include, start, end, count) rotated by the case index. A "
    "word target redraws at its own slot's next counter while it "
    "equals the task value, equals another target group of the "
    "case, or -- for position 1's target only -- occurs in a "
    "neutral statement of the case; occurrence means normalized-"
    "word equality. All start constraints of a case share the "
    "start target and all end constraints share the end target, "
    "so a case has at most seven distinct word targets (four "
    "include, one start, one end, the task value), below the "
    "smallest count limit."
)
_DENSITY_RENDERING_RULE_TEXT = (
    "A case's user turn is the task fact, the task question, "
    "and the arm's instruction block, joined by a blank line. "
    "A constraint renders through its kind template and is "
    "always labelled with its own position: 'Instruction "
    "{position}: {text}' -- in solo arms too, so a constraint's "
    "position label is identical in every arm that carries it. "
    "A density arm's block is the first K rendered constraints "
    "in position order. A solo arm's block is the single "
    "rendered constraint. The sham arm's block is constraint "
    "1's rendered line followed by bare neutral statements (no "
    "label), appended in drawn order, wrapping to the start of "
    "the drawn list, until the block's word count first reaches "
    "or exceeds the k16 block's word count; block word count is "
    "the normalized-word count of the block text."
)
_DENSITY_CHECKING_RULE_TEXT = (
    "A word is a maximal run of alphanumeric characters, "
    "case-folded. The commitment of an answer is the ordered "
    "list of SHA-256 hashes of its words; the list digest is "
    "SHA-256 of the compact-JSON list. Satisfaction is derived "
    "from the list: include = the target's hash occurs in the "
    "list; start = the list's first hash is the target's; end "
    "= the list's last hash is the target's; count = the "
    "list's length is at most the limit. The task value is "
    "present iff its hash occurs in the list. Aggregation "
    "excludes zero-word live records from count-kind checked "
    "denominators. The falsification decision aggregates "
    "solo-vs-k16 discordance per case and applies the exact "
    "two-sided sign test over discordant cases."
)


def _density_digest(text: str) -> str:
    """SHA-256 hex digest (local copy; no imports from
    density.py anywhere in this mirror)."""
    import hashlib as _hashlib

    return _hashlib.sha256(text.encode("utf-8")).hexdigest()


def _density_digest_ok(value: Any) -> bool:
    """Whether a stored digest is a well-formed SHA-256 hex
    digest (fails closed on lone surrogates, the cycle-190
    discipline)."""
    if type(value) is not str or len(value) != 64:
        return False
    return all(ch in "0123456789abcdef" for ch in value)


def _density_derive(seed: int, index: int, slot: int, counter: int) -> int:
    """Mirror of density._derive (frozen material string)."""
    import hashlib as _hashlib

    material = f"{seed}|density|{index}|{slot}|{counter}"
    return int.from_bytes(_hashlib.sha256(material.encode("ascii")).digest()[:8], "big")


def _density_normalize_words(text: str) -> list[str]:
    """Mirror of density.normalize_words."""
    words: list[str] = []
    current: list[str] = []
    for char in text:
        if char.isalnum():
            current.append(char)
        elif current:
            words.append("".join(current).casefold())
            current = []
    if current:
        words.append("".join(current).casefold())
    return words


def _density_kind_at(case_index: int, position: int) -> str:
    """Mirror of density.kind_at."""
    return _DENSITY_KINDS[(position - 1 + case_index) % len(_DENSITY_KINDS)]


def _density_draw_word(seed: int, index: int, slot: int, forbidden: set[str]) -> str:
    """Mirror of density._draw_word."""
    counter = 0
    while True:
        word = _DENSITY_TARGET_WORDS[
            _density_derive(seed, index, slot, counter) % len(_DENSITY_TARGET_WORDS)
        ]
        if word not in forbidden:
            return word
        counter += 1


def _density_generate(seed: int) -> list[dict[str, Any]]:
    """Mirror of density.generate_cases: the frozen case set
    for a seed, regenerated from the protocol's own seed so a
    payload cannot present a different instrument under the
    same protocol."""
    cases: list[dict[str, Any]] = []
    for index in range(_DENSITY_CASE_COUNT):
        topic = _DENSITY_TOPICS[_density_derive(seed, index, 0, 0) % len(_DENSITY_TOPICS)]
        value = _DENSITY_WORDS[_density_derive(seed, index, 1, 0) % len(_DENSITY_WORDS)]
        neutrals: list[str] = []
        chosen: set[int] = set()
        for k in range(_DENSITY_NEUTRALS_PER_CASE):
            counter = 0
            while True:
                pick = _density_derive(seed, index, 10 + k, counter) % len(_DENSITY_NEUTRALS)
                if pick not in chosen:
                    break
                counter += 1
            chosen.add(pick)
            neutrals.append(_DENSITY_NEUTRALS[pick])
        neutral_words: set[str] = set()
        for statement in neutrals:
            neutral_words.update(_density_normalize_words(statement))
        include_targets: list[str] = []
        forbidden: set[str] = {value}
        for slot in (30, 31, 32, 33):
            word = _density_draw_word(seed, index, slot, forbidden)
            include_targets.append(word)
            forbidden.add(word)
        start_target = _density_draw_word(seed, index, 34, forbidden)
        forbidden.add(start_target)
        end_target = _density_draw_word(seed, index, 35, forbidden)
        forbidden.add(end_target)
        first_kind = _density_kind_at(index, 1)
        if first_kind in ("include", "start", "end"):
            current = (
                include_targets[0]
                if first_kind == "include"
                else start_target
                if first_kind == "start"
                else end_target
            )
            if current in neutral_words:
                slot = 30 if first_kind == "include" else 34 if first_kind == "start" else 35
                fresh = _density_draw_word(seed, index, slot, forbidden | neutral_words)
                if first_kind == "include":
                    include_targets[0] = fresh
                elif first_kind == "start":
                    start_target = fresh
                else:
                    end_target = fresh
                forbidden.discard(current)
                forbidden.add(fresh)
        limit_pool = list(_DENSITY_COUNT_PARAMS)
        count_limits: list[int] = []
        for slot in (40, 41, 42, 43):
            pick = _density_derive(seed, index, slot, 0) % len(limit_pool)
            count_limits.append(limit_pool.pop(pick))
        constraints: list[dict[str, Any]] = []
        pools = {
            "include": iter(include_targets),
            "start": iter([start_target] * 4),
            "end": iter([end_target] * 4),
            "count": iter(count_limits),
        }
        for position in range(1, _DENSITY_CONSTRAINT_COUNT + 1):
            kind = _density_kind_at(index, position)
            if kind == "count":
                constraints.append(
                    {"position": position, "kind": kind, "limit": next(pools["count"])}
                )
            elif kind == "include":
                constraints.append(
                    {"position": position, "kind": kind, "target": next(pools["include"])}
                )
            else:
                constraints.append(
                    {"position": position, "kind": kind, "target": next(pools[kind])}
                )
        cases.append(
            {
                "fixture_id": f"density-{index:02d}",
                "topic": topic,
                "value": value,
                "fact": _DENSITY_FACT_TEMPLATE.format(topic=topic, value=value),
                "question": _DENSITY_QUESTION_TEMPLATE.format(topic=topic),
                "constraints": constraints,
                "neutrals": neutrals,
                "expected_word_digest": _density_digest(value),
            }
        )
    return cases


def _density_render_constraint(constraint: dict[str, Any]) -> str:
    """Mirror of density._render_constraint."""
    kind = str(constraint["kind"])
    if kind == "count":
        text = _DENSITY_CONSTRAINT_TEMPLATES["count"].format(limit=constraint["limit"])
    else:
        text = _DENSITY_CONSTRAINT_TEMPLATES[kind].format(target=constraint["target"])
    return _DENSITY_LINE_TEMPLATE.format(position=constraint["position"], text=text)


def _density_arm_block(case: dict[str, Any], arm: str) -> str:
    """Mirror of density._arm_block."""
    constraints = case["constraints"]
    if arm.startswith("solo_"):
        j = int(arm.split("_")[1])
        return "\n".join([_density_render_constraint(constraints[j - 1])])
    if arm.startswith("k"):
        k = int(arm[1:])
        return "\n".join(_density_render_constraint(c) for c in constraints[:k])
    if arm == "sham":
        first = _density_render_constraint(constraints[0])
        target = len(_density_normalize_words(_density_arm_block(case, "k16")))
        lines = [first]
        words = len(_density_normalize_words(first))
        pool = list(case["neutrals"])
        index = 0
        while words < target:
            statement = pool[index % len(pool)]
            lines.append(statement)
            words += len(_density_normalize_words(statement))
            index += 1
        return "\n".join(lines)
    raise ValueError(f"unknown density arm: {arm!r}")


def _density_arm_block_words(case: dict[str, Any], arm: str) -> int:
    """Mirror of density.arm_block_words."""
    return len(_density_normalize_words(_density_arm_block(case, arm)))


def _density_arm_content(case: dict[str, Any], arm: str) -> str:
    """Mirror of density.arm_content."""
    return "\n\n".join([str(case["fact"]), str(case["question"]), _density_arm_block(case, arm)])


def _density_case_material_sha256() -> str:
    """Mirror of density.case_material_sha256."""
    import json as _json

    canonical = _json.dumps(
        [
            _DENSITY_FACT_TEMPLATE,
            _DENSITY_QUESTION_TEMPLATE,
            _DENSITY_CONSTRAINT_TEMPLATES,
            _DENSITY_LINE_TEMPLATE,
            list(_DENSITY_TOPICS),
            list(_DENSITY_WORDS),
            list(_DENSITY_TARGET_WORDS),
            list(_DENSITY_NEUTRALS),
        ],
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return _density_digest(canonical)


def _density_ground_sha256(seed: int) -> str:
    """Mirror of density.ground_sha256."""
    import json as _json

    rows: list[list[Any]] = []
    for case in _density_generate(seed):
        rows.append(
            [
                case["fixture_id"],
                case["topic"],
                case["value"],
                [
                    [c["position"], c["kind"], c.get("target"), c.get("limit")]
                    for c in case["constraints"]
                ],
                list(case["neutrals"]),
            ]
        )
    canonical = _json.dumps(rows, separators=(",", ":"), ensure_ascii=True)
    return _density_digest(canonical)


def _density_protocol(seed: int, backend_fingerprint: str) -> dict[str, Any]:
    """The density protocol rebuilt from this module's own
    literals plus the payload's seed and backend fingerprint,
    fingerprint included (compared whole)."""
    import json as _json

    protocol: dict[str, Any] = {
        "version": 1,
        "arm_keys": list(_DENSITY_ARM_KEYS),
        "case_count": _DENSITY_CASE_COUNT,
        "constraint_count": _DENSITY_CONSTRAINT_COUNT,
        "densities": list(_DENSITY_DENSITIES),
        "count_params": list(_DENSITY_COUNT_PARAMS),
        "case_material_sha256": _density_case_material_sha256(),
        "ground_sha256": _density_ground_sha256(seed),
        "generation_rule_sha256": _density_digest(_DENSITY_GENERATION_RULE_TEXT),
        "rendering_rule_sha256": _density_digest(_DENSITY_RENDERING_RULE_TEXT),
        "checking_rule_sha256": _density_digest(_DENSITY_CHECKING_RULE_TEXT),
        "min_discordant_cases": _DENSITY_MIN_DISCORDANT_CASES,
        "alpha": _DENSITY_ALPHA,
        "max_new_tokens": _DENSITY_MAX_NEW_TOKENS,
        "max_density_records": _DENSITY_MAX_RECORDS,
        "max_density_payload_bytes": _DENSITY_MAX_PAYLOAD_BYTES,
        "seed": seed,
        "backend_fingerprint": backend_fingerprint,
    }
    canonical = _json.dumps(protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    protocol["fingerprint"] = _density_digest(canonical)
    return protocol


def _density_commitment_digest(hashes: list[str]) -> str:
    """Mirror of density.commitment_digest."""
    import json as _json

    canonical = _json.dumps(list(hashes), separators=(",", ":"), ensure_ascii=True)
    return _density_digest(canonical)


def _density_satisfied(constraint: dict[str, Any], hashes: list[str]) -> bool:
    """Mirror of density.satisfied."""
    kind = str(constraint["kind"])
    if kind == "count":
        return len(hashes) <= int(constraint["limit"])
    target = _density_digest(str(constraint["target"]))
    if kind == "include":
        return target in hashes
    if kind == "start":
        return bool(hashes) and hashes[0] == target
    if kind == "end":
        return bool(hashes) and hashes[-1] == target
    raise ValueError(f"unknown constraint kind: {kind!r}")


def _density_task_present(case: dict[str, Any], hashes: list[str]) -> bool:
    """Mirror of density.task_present."""
    return str(case["expected_word_digest"]) in hashes


def _density_sign_test_p(n_pos: int, n_neg: int) -> float:
    """Mirror of density.sign_test_p."""
    import math as _math

    n = n_pos + n_neg
    if n == 0:
        return 1.0
    k = min(n_pos, n_neg)
    tail: float = sum(_math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def _density_live(record: dict[str, Any]) -> bool:
    """Mirror of density._live."""
    return (
        record.get("detail") == ""
        and not record.get("case_screened")
        and bool(record.get("boundary_found"))
    )


def _density_rate(numerator: int, denominator: int) -> float | None:
    """Mirror of density._rate."""
    return numerator / denominator if denominator else None


def _density_eligible(constraint: dict[str, Any], hashes: list[str]) -> bool:
    """Mirror of density._eligible."""
    return not (constraint["kind"] == "count" and not hashes)


def _density_aggregate(
    records: list[dict[str, Any]], cases: list[dict[str, Any]]
) -> dict[str, Any]:
    """Mirror of density.aggregate_candidate."""
    by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for record in records:
        by_key[(int(record["case_index"]), str(record["arm"]))] = record
    n_generated = len(records)
    n_boundary_screened = sum(1 for r in records if r.get("case_screened"))
    rest = [r for r in records if not r.get("case_screened")]
    n_detail_screened = sum(1 for r in rest if r.get("detail") != "")
    rest = [r for r in rest if r.get("detail") == ""]
    n_no_boundary = sum(1 for r in rest if not r.get("boundary_found"))
    live = [r for r in rest if r.get("boundary_found")]
    n_zero_word = sum(1 for r in live if len(r.get("word_hashes") or []) == 0)
    block: dict[str, Any] = {
        "available": bool(live),
        "detail": "" if live else "density_unavailable",
        "n_generated": n_generated,
        "n_boundary_screened": n_boundary_screened,
        "n_detail_screened": n_detail_screened,
        "n_no_boundary": n_no_boundary,
        "n_zero_word": n_zero_word,
        "n_live": len(live),
    }

    def hashes_of(record: dict[str, Any]) -> list[str]:
        return list(record.get("word_hashes") or [])

    def case_of(record: dict[str, Any]) -> dict[str, Any]:
        return cases[int(record["case_index"])]

    density_rows: list[dict[str, Any]] = []
    for k in _DENSITY_DENSITIES:
        arm = "solo_1" if k == 1 else f"k{k}"
        sat = checked = 0
        solo_sat = solo_checked = 0
        for record in live:
            if record["arm"] != arm:
                continue
            case = case_of(record)
            hashes = hashes_of(record)
            for constraint in case["constraints"][:k]:
                if not _density_eligible(constraint, hashes):
                    continue
                checked += 1
                sat += _density_satisfied(constraint, hashes)
        for record in live:
            if not str(record["arm"]).startswith("solo_"):
                continue
            j = int(str(record["arm"]).split("_")[1])
            if j > k:
                continue
            case = case_of(record)
            constraint = case["constraints"][j - 1]
            if not _density_eligible(constraint, hashes_of(record)):
                continue
            solo_checked += 1
            solo_sat += _density_satisfied(constraint, hashes_of(record))
        density_rows.append(
            {
                "density": k,
                "satisfied": sat,
                "checked": checked,
                "rate": _density_rate(sat, checked),
                "solo_satisfied": solo_sat,
                "solo_checked": solo_checked,
                "solo_rate": _density_rate(solo_sat, solo_checked),
            }
        )
    block["by_density"] = density_rows

    presence_rows: list[dict[str, Any]] = []
    for arm in _DENSITY_ARM_KEYS:
        present = total = 0
        for record in live:
            if record["arm"] != arm:
                continue
            total += 1
            present += _density_task_present(case_of(record), hashes_of(record))
        presence_rows.append(
            {
                "arm": arm,
                "present": present,
                "live": total,
                "rate": _density_rate(present, total),
            }
        )
    block["task_presence"] = presence_rows

    k16_live = [r for r in live if r["arm"] == "k16"]
    position_rows: list[dict[str, Any]] = []
    for position in range(1, _DENSITY_CONSTRAINT_COUNT + 1):
        sat = total = 0
        for record in k16_live:
            constraint = case_of(record)["constraints"][position - 1]
            if not _density_eligible(constraint, hashes_of(record)):
                continue
            total += 1
            sat += _density_satisfied(constraint, hashes_of(record))
        position_rows.append(
            {
                "position": position,
                "satisfied": sat,
                "checked": total,
                "rate": _density_rate(sat, total),
            }
        )
    block["position_k16"] = position_rows
    bands: list[dict[str, Any]] = []
    for label, lo, hi in (("early", 1, 5), ("mid", 6, 11), ("late", 12, 16)):
        sat = sum(r["satisfied"] for r in position_rows[lo - 1 : hi])
        total = sum(r["checked"] for r in position_rows[lo - 1 : hi])
        bands.append(
            {"band": label, "satisfied": sat, "checked": total, "rate": _density_rate(sat, total)}
        )
    block["position_bands"] = bands

    kind_rows: list[dict[str, Any]] = []
    for kind in _DENSITY_KINDS:
        row: dict[str, Any] = {"kind": kind}
        for label, arms in (("solo", _DENSITY_SOLO_ARMS), ("k16", ("k16",))):
            sat = checked = 0
            for record in live:
                if record["arm"] not in arms:
                    continue
                case = case_of(record)
                hashes = hashes_of(record)
                if label == "solo":
                    j = int(str(record["arm"]).split("_")[1])
                    constraint = case["constraints"][j - 1]
                    if constraint["kind"] != kind:
                        continue
                    if not _density_eligible(constraint, hashes):
                        continue
                    checked += 1
                    sat += _density_satisfied(constraint, hashes)
                else:
                    for constraint in case["constraints"]:
                        if constraint["kind"] != kind:
                            continue
                        if not _density_eligible(constraint, hashes):
                            continue
                        checked += 1
                        sat += _density_satisfied(constraint, hashes)
            row[f"{label}_satisfied"] = sat
            row[f"{label}_checked"] = checked
            row[f"{label}_rate"] = _density_rate(sat, checked)
        kind_rows.append(row)
    block["by_kind"] = kind_rows

    joint_sat = 0
    for record in k16_live:
        case = case_of(record)
        hashes = hashes_of(record)
        if all(_density_satisfied(c, hashes) for c in case["constraints"]):
            joint_sat += 1
    block["joint_k16"] = {
        "satisfied": joint_sat,
        "live": len(k16_live),
        "rate": _density_rate(joint_sat, len(k16_live)),
    }

    sham_cases = [
        case_index
        for case_index in range(len(cases))
        if all(
            (case_index, arm) in by_key and _density_live(by_key[(case_index, arm)])
            for arm in ("solo_1", "sham", "k16")
        )
    ]
    sham_row: dict[str, Any] = {"cases": len(sham_cases)}
    for arm in ("solo_1", "sham", "k16"):
        sat = 0
        for case_index in sham_cases:
            record = by_key[(case_index, arm)]
            constraint = cases[case_index]["constraints"][0]
            if not _density_eligible(constraint, hashes_of(record)):
                continue
            sat += _density_satisfied(constraint, hashes_of(record))
        sham_row[f"{arm}_satisfied"] = sat
        sham_row[f"{arm}_rate"] = _density_rate(sat, len(sham_cases))
    if sham_cases:
        sham_row["sham_block_words_mean"] = sum(
            _density_arm_block_words(cases[i], "sham") for i in sham_cases
        ) / len(sham_cases)
        sham_row["k16_block_words_mean"] = sum(
            _density_arm_block_words(cases[i], "k16") for i in sham_cases
        ) / len(sham_cases)
    else:
        sham_row["sham_block_words_mean"] = None
        sham_row["k16_block_words_mean"] = None
    block["sham"] = sham_row

    per_position: list[dict[str, Any]] = [
        {"position": position, "f": 0, "p": 0}
        for position in range(1, _DENSITY_CONSTRAINT_COUNT + 1)
    ]
    total_f = total_p = 0
    case_discordance: list[int] = []
    for case_index, case in enumerate(cases):
        d_c = 0
        for position in range(1, _DENSITY_CONSTRAINT_COUNT + 1):
            solo = by_key.get((case_index, f"solo_{position}"))
            full = by_key.get((case_index, "k16"))
            if solo is None or full is None or not _density_live(solo) or not _density_live(full):
                continue
            constraint = case["constraints"][position - 1]
            if not _density_eligible(constraint, hashes_of(solo)) or not _density_eligible(
                constraint, hashes_of(full)
            ):
                continue
            solo_ok = _density_satisfied(constraint, hashes_of(solo))
            full_ok = _density_satisfied(constraint, hashes_of(full))
            row = per_position[position - 1]
            if solo_ok and not full_ok:
                row["f"] += 1
                total_f += 1
                d_c += 1
            elif full_ok and not solo_ok:
                row["p"] += 1
                total_p += 1
                d_c -= 1
        case_discordance.append(d_c)
    block["discordance_by_position"] = per_position
    discordant_cases = [d for d in case_discordance if d != 0]
    n_pos = sum(1 for d in discordant_cases if d > 0)
    n_neg = sum(1 for d in discordant_cases if d < 0)
    p_value = _density_sign_test_p(n_pos, n_neg)
    decision: dict[str, Any] = {
        "f": total_f,
        "p": total_p,
        "discordant_cases": len(discordant_cases),
        "cases_positive": n_pos,
        "cases_negative": n_neg,
        "sign_p": p_value,
    }
    if len(discordant_cases) < _DENSITY_MIN_DISCORDANT_CASES:
        decision["verdict"] = "insufficient"
        decision["falsified"] = None
    else:
        falsified = total_f > total_p and p_value <= _DENSITY_ALPHA
        decision["verdict"] = "falsified" if falsified else "not_falsified"
        decision["falsified"] = falsified
    block["decision"] = decision
    return block


def _density_recompute(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Recompute a density payload from its ground records.

    The validator carries its own generator mirror, templates,
    pools, and aggregator (no imports from density.py): the
    protocol is rebuilt from frozen scalars plus the payload's
    seed and backend fingerprint and compared whole; the case
    set is regenerated from the seed and every record's
    content digest is compared against the re-rendered arm;
    every record's commitment is verified (list digest and
    count) and every satisfaction verdict, readout, and the
    falsification decision is re-derived from the commitments
    and compared.
    """
    import json as _json

    protocol = payload.get("protocol")
    if type(protocol) is not dict:
        return None
    seed = protocol.get("seed")
    backend_fingerprint = protocol.get("backend_fingerprint")
    if type(seed) is not int or isinstance(seed, bool) or type(backend_fingerprint) is not str:
        return None
    if not _recomputed_equal(protocol, _density_protocol(seed, backend_fingerprint)):
        return None
    candidates = payload.get("candidates")
    if type(candidates) is not dict or not candidates:
        return None
    generated = _density_generate(seed)
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
        if status != "available":
            return None
        if set(entry) != _DENSITY_AGGREGATE_KEYS | {"status", "records"}:
            return None
        records = entry["records"]
        if type(records) is not list or len(records) != len(_DENSITY_ARM_KEYS) * len(generated):
            return None
        total_records += len(records)
        records_bytes += len(
            _json.dumps(records, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        )
        if records_bytes > _DENSITY_MAX_PAYLOAD_BYTES:
            return None
        expected_order = [
            (case_index, arm) for case_index in range(len(generated)) for arm in _DENSITY_ARM_KEYS
        ]
        for record, (case_index, arm) in zip(records, expected_order, strict=True):
            if type(record) is not dict or set(record) != _DENSITY_RECORD_KEYS:
                return None
            if record["candidate"] != name:
                return None
            if record["case_index"] != case_index or record["arm"] != arm:
                return None
            if record["detail"] not in _DENSITY_RECORD_DETAILS:
                return None
            if type(record["case_screened"]) is not bool:
                return None
            if type(record["boundary_found"]) is not bool:
                return None
            hashes = record["word_hashes"]
            if type(hashes) is not list or any(not _density_digest_ok(h) for h in hashes):
                return None
            if record["word_count"] != len(hashes):
                return None
            if record["word_list_digest"] != _density_commitment_digest(list(hashes)):
                return None
            if type(record["input_tokens"]) is not int or record["input_tokens"] < 0:
                return None
            if record["answer_digest"] is not None and not _density_digest_ok(
                record["answer_digest"]
            ):
                return None
            case = generated[case_index]
            if record["content_digest"] != _density_digest(_density_arm_content(case, arm)):
                return None
            if record["detail"] != "" or record["case_screened"]:
                # A screened or degraded record carries no
                # generation: the empty commitment, no answer
                # digest, no boundary.
                if hashes or record["answer_digest"] is not None or record["boundary_found"]:
                    return None
            else:
                # A completed record carries the generation's
                # answer digest (its boundary may or may not
                # have been found; that is a measurement).
                if record["answer_digest"] is None:
                    return None
        aggregates[name] = _density_aggregate(records, generated)
    if total_records > _DENSITY_MAX_RECORDS:
        return None
    if not aggregates:
        return None
    return {
        "aggregates": aggregates,
        "n_candidates": len(candidates),
        "n_cases": len(generated),
    }


def _valid_density(payload: Any) -> bool:
    """Structural check for a ``density`` payload (recompute standard)."""
    if type(payload) is not dict or payload.get("available") is not True:
        return False
    if set(payload) != {
        "available",
        "detail",
        "protocol",
        "candidates",
        "n_candidates",
        "n_cases",
    }:
        return False
    if payload.get("detail") != "":
        return False
    recomputed = _density_recompute(payload)
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


def diff_density(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``density`` probe payloads; deterministic.

    Answers "did the density story change between runs?":
    per-candidate endpoint verdict and falsification drift,
    the discordance counts, joint satisfaction at full
    density, the screening accounting, candidates
    added/removed, and protocol / generator / backend drift
    (any of which marks the comparison not comparable, with
    the drift still listed -- the generated set is the
    treatment, so a changed generator is protocol drift). A
    side without a valid payload is reported plainly.
    """
    old_ok = _valid_density(old)
    new_ok = _valid_density(new)
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
    density_protocol_drift = old["protocol"].get("fingerprint") != new["protocol"].get(
        "fingerprint"
    )
    density_generator_drift = old["protocol"].get("ground_sha256") != new["protocol"].get(
        "ground_sha256"
    )
    density_backend_drift = old["protocol"].get("backend_fingerprint") != new["protocol"].get(
        "backend_fingerprint"
    )
    density_comparable = (
        not density_protocol_drift and not density_generator_drift and not density_backend_drift
    )
    old_candidates = old["candidates"]
    new_candidates = new["candidates"]
    density_candidates_added = sorted(set(new_candidates) - set(old_candidates))
    density_candidates_removed = sorted(set(old_candidates) - set(new_candidates))
    density_candidate_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_candidates) & set(new_candidates)):
        old_entry = old_candidates[name]
        new_entry = new_candidates[name]
        if old_entry.get("status") != "available" or new_entry.get("status") != "available":
            if old_entry.get("status") != new_entry.get("status"):
                density_candidate_drift.append(
                    {
                        "candidate": name,
                        "metric": "status",
                        "old": old_entry.get("status"),
                        "new": new_entry.get("status"),
                    }
                )
            continue
        density_metrics: list[tuple[str, Any, Any]] = [
            (
                "verdict",
                old_entry["decision"]["verdict"],
                new_entry["decision"]["verdict"],
            ),
            (
                "falsified",
                old_entry["decision"]["falsified"],
                new_entry["decision"]["falsified"],
            ),
            ("f", old_entry["decision"]["f"], new_entry["decision"]["f"]),
            ("p", old_entry["decision"]["p"], new_entry["decision"]["p"]),
            (
                "discordant_cases",
                old_entry["decision"]["discordant_cases"],
                new_entry["decision"]["discordant_cases"],
            ),
            (
                "sign_p",
                old_entry["decision"]["sign_p"],
                new_entry["decision"]["sign_p"],
            ),
            (
                "joint_satisfied",
                old_entry["joint_k16"]["satisfied"],
                new_entry["joint_k16"]["satisfied"],
            ),
            (
                "joint_live",
                old_entry["joint_k16"]["live"],
                new_entry["joint_k16"]["live"],
            ),
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
        for metric, old_value, new_value in density_metrics:
            if old_value != new_value:
                density_candidate_drift.append(
                    {
                        "candidate": name,
                        "metric": metric,
                        "old": old_value,
                        "new": new_value,
                    }
                )
    result["candidate_drift"] = density_candidate_drift
    result["candidates_added"] = density_candidates_added
    result["candidates_removed"] = density_candidates_removed
    result["protocol_drift"] = density_protocol_drift
    result["generator_drift"] = density_generator_drift
    result["backend_drift"] = density_backend_drift
    result["comparable"] = density_comparable
    result["changed"] = bool(
        density_candidate_drift
        or density_candidates_added
        or density_candidates_removed
        or density_protocol_drift
        or density_generator_drift
        or density_backend_drift
    )
    return result


_FIELD_ORDER_SPECS: tuple[dict[str, Any], ...] = (
    {
        "id": "extract-person",
        "shape": "extraction",
        "question": "Extract the person's name and age.",
        "context": "Ada Lovelace was 36 when she wrote the first algorithm.",
        "schema": {
            "type": "object",
            "required": ["name", "age"],
            "properties": {"name": {"type": "string"}, "age": {"type": "integer", "minimum": 0}},
            "additionalProperties": False,
        },
        "answer": {"name": "Ada Lovelace", "age": 36},
        "answer_field": "name",
        "paired": True,
        "source": "structured",
    },
    {
        "id": "extract-city",
        "shape": "extraction",
        "question": "Extract the city and its population.",
        "context": "Lisbon has about 500000 residents.",
        "schema": {
            "type": "object",
            "required": ["city", "population"],
            "properties": {
                "city": {"type": "string"},
                "population": {"type": "integer", "minimum": 0},
            },
            "additionalProperties": False,
        },
        "answer": {"city": "Lisbon", "population": 500000},
        "answer_field": "city",
        "paired": True,
        "source": "structured",
    },
    {
        "id": "extract-book",
        "shape": "extraction",
        "question": "Extract the book title and its publication year.",
        "context": "The novel 'Dune' was first published in 1965.",
        "schema": {
            "type": "object",
            "required": ["title", "year"],
            "properties": {
                "title": {"type": "string"},
                "year": {"type": "integer", "minimum": 0, "maximum": 2100},
            },
            "additionalProperties": False,
        },
        "answer": {"title": "Dune", "year": 1965},
        "answer_field": "title",
        "paired": True,
        "source": "structured",
    },
    {
        "id": "classify-sentiment",
        "shape": "classification",
        "question": "Classify the sentiment of the review.",
        "context": "Review: 'The service was dreadful and painfully slow.'",
        "schema": {
            "type": "object",
            "required": ["label"],
            "properties": {"label": {"type": "string", "enum": ["positive", "negative"]}},
            "additionalProperties": False,
        },
        "answer": {"label": "negative"},
        "answer_field": "label",
        "paired": False,
        "source": "structured",
    },
    {
        "id": "classify-urgency",
        "shape": "classification",
        "question": "Classify the urgency of the ticket.",
        "context": "Ticket: 'The server room is on fire.'",
        "schema": {
            "type": "object",
            "required": ["level"],
            "properties": {"level": {"type": "string", "enum": ["low", "high"]}},
            "additionalProperties": False,
        },
        "answer": {"level": "high"},
        "answer_field": "level",
        "paired": False,
        "source": "structured",
    },
    {
        "id": "classify-language",
        "shape": "classification",
        "question": "Classify the language of the sentence.",
        "context": "Sentence: 'Guten Morgen, wie geht es dir?'",
        "schema": {
            "type": "object",
            "required": ["language"],
            "properties": {"language": {"type": "string", "enum": ["english", "german", "french"]}},
            "additionalProperties": False,
        },
        "answer": {"language": "german"},
        "answer_field": "language",
        "paired": False,
        "source": "structured",
    },
    {
        "id": "route-ticket",
        "shape": "routing",
        "question": "Route the support ticket and set its priority.",
        "context": "Ticket: 'My invoice shows a charge I do not recognize.'",
        "schema": {
            "type": "object",
            "required": ["department", "priority"],
            "properties": {
                "department": {"type": "string", "enum": ["sales", "support", "billing"]},
                "priority": {"type": "integer", "minimum": 1, "maximum": 5},
            },
            "additionalProperties": False,
        },
        "answer": {"department": "billing", "priority": 3},
        "answer_field": "department",
        "paired": True,
        "source": "structured",
    },
    {
        "id": "route-alert",
        "shape": "routing",
        "question": "Route the alert and say whether it needs acknowledgement.",
        "context": "Alert: 'Disk usage on db-1 reached 97%.'",
        "schema": {
            "type": "object",
            "required": ["channel", "ack_required"],
            "properties": {
                "channel": {"type": "string", "enum": ["email", "sms", "pager"]},
                "ack_required": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
        "answer": {"channel": "pager", "ack_required": True},
        "answer_field": "channel",
        "paired": False,
        "source": "structured",
    },
    {
        "id": "route-order",
        "shape": "routing",
        "question": "Route the order to a warehouse and flag express shipping.",
        "context": "Order: 'Customer in Porto needs the parcel by Friday.'",
        "schema": {
            "type": "object",
            "required": ["warehouse", "express"],
            "properties": {
                "warehouse": {"type": "string", "enum": ["north", "south"]},
                "express": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
        "answer": {"warehouse": "south", "express": True},
        "answer_field": "warehouse",
        "paired": False,
        "source": "structured",
    },
    {
        "id": "extract-product",
        "shape": "extraction",
        "question": "Extract the product's brand, model, price and stock status.",
        "context": "The catalogue lists it as: brand Acme, model Rotor X2. "
        "It costs 349 dollars and is currently in stock.",
        "schema": {
            "type": "object",
            "required": ["brand", "model", "price", "in_stock"],
            "properties": {
                "brand": {"type": "string"},
                "model": {"type": "string"},
                "price": {"type": "integer", "minimum": 0},
                "in_stock": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
        "answer": {"brand": "Acme", "model": "Rotor X2", "price": 349, "in_stock": True},
        "answer_field": "model",
        "paired": True,
        "source": "fieldorder",
    },
    {
        "id": "extract-event",
        "shape": "extraction",
        "question": "Extract the event venue, day and attendance.",
        "context": "The workshop meets at Harbor Hall on Friday with 40 attendees.",
        "schema": {
            "type": "object",
            "required": ["venue", "day", "attendance"],
            "properties": {
                "venue": {"type": "string"},
                "day": {
                    "type": "string",
                    "enum": [
                        "monday",
                        "tuesday",
                        "wednesday",
                        "thursday",
                        "friday",
                        "saturday",
                        "sunday",
                    ],
                },
                "attendance": {"type": "integer", "minimum": 0},
            },
            "additionalProperties": False,
        },
        "answer": {"venue": "Harbor Hall", "day": "friday", "attendance": 40},
        "answer_field": "venue",
        "paired": True,
        "source": "fieldorder",
    },
    {
        "id": "route-complaint",
        "shape": "routing",
        "question": "Route the complaint and record whether a defect was "
        "confirmed and whether data was lost.",
        "context": "Complaint: 'The update deleted my saved files.' "
        "Front-line support confirmed the update itself is defective.",
        "schema": {
            "type": "object",
            "required": ["team", "defect_confirmed", "data_lost"],
            "properties": {
                "team": {"type": "string", "enum": ["sales", "support", "technical"]},
                "defect_confirmed": {"type": "boolean"},
                "data_lost": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
        "answer": {"team": "technical", "defect_confirmed": True, "data_lost": True},
        "answer_field": "team",
        "paired": True,
        "source": "fieldorder",
    },
    {
        "id": "classify-document",
        "shape": "classification",
        "question": "Classify the document's type, language and confidentiality.",
        "context": "Document: a German-language internal memo marked confidential.",
        "schema": {
            "type": "object",
            "required": ["doc_type", "language", "confidential"],
            "properties": {
                "doc_type": {"type": "string", "enum": ["memo", "report", "invoice"]},
                "language": {"type": "string", "enum": ["english", "german", "french"]},
                "confidential": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
        "answer": {"doc_type": "memo", "language": "german", "confidential": True},
        "answer_field": "doc_type",
        "paired": True,
        "source": "fieldorder",
    },
    {
        "id": "extract-vehicle",
        "shape": "extraction",
        "question": "Extract the vehicle make, year, fuel and seat count.",
        "context": "The 2019 Volvo wagon runs on diesel and seats five.",
        "schema": {
            "type": "object",
            "required": ["make", "year", "fuel", "seats"],
            "properties": {
                "make": {"type": "string"},
                "year": {"type": "integer", "minimum": 1900, "maximum": 2100},
                "fuel": {"type": "string", "enum": ["petrol", "diesel", "electric"]},
                "seats": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": False,
        },
        "answer": {"make": "Volvo", "year": 2019, "fuel": "diesel", "seats": 5},
        "answer_field": "make",
        "paired": True,
        "source": "fieldorder",
    },
    {
        "id": "route-delivery",
        "shape": "routing",
        "question": "Route the delivery and flag its requirements.",
        "context": "Delivery: vaccines to a clinic in the south, sent by "
        "express courier for tomorrow morning; a signature is required "
        "and the box must stay refrigerated.",
        "schema": {
            "type": "object",
            "required": ["region", "speed", "signature_required", "refrigerated"],
            "properties": {
                "region": {"type": "string", "enum": ["north", "south", "east", "west"]},
                "speed": {"type": "string", "enum": ["standard", "express"]},
                "signature_required": {"type": "boolean"},
                "refrigerated": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
        "answer": {
            "region": "south",
            "speed": "express",
            "signature_required": True,
            "refrigerated": True,
        },
        "answer_field": "region",
        "paired": True,
        "source": "fieldorder",
    },
)


_FO_GENERATION_RULE_TEXT = (
    "The spec set is the frozen FIELD_ORDER_SPECS table: the nine "
    "structured specs (source 'structured', reused unchanged) plus "
    "six fieldorder specs. Each spec designates one answer field "
    "and a paired flag; a spec may be paired only if a reader "
    "given question, context and schema (not the answer table) "
    "can derive the answer field's value uniquely. Canonical "
    "order is alphabetical by field name. Arms: canonical; "
    "answer_first (answer field first, the rest in canonical "
    "relative order); answer_last (answer field last, the rest "
    "in canonical relative order); reversed (canonical order "
    "reversed). Generation is greedy via core.evaluate at a "
    "fixed 64 new tokens; exactly 60 records per candidate in "
    "(spec, arm) canonical order. Record budget 1440 (24 "
    "candidates); payload budget 8388608 bytes."
)

_FO_RENDERING_RULE_TEXT = (
    "The user content is the structured prompt template with the "
    "schema description rendered in the arm's order: question, "
    "then 'Context: ', then 'Respond with ONLY a JSON object "
    "matching this schema: ' plus the ordered description, then "
    "'Do not include any other text.' and the ref marker "
    "(structured/<id> for shipped specs, fieldorder/<id> for new "
    "specs). The ordered description lists each property in the "
    'arm order as "name": type with enum and bound annotations '
    "in the structured format, then the required list in the arm "
    "order and the additional-properties clause. The canonical "
    "arm's prompt is byte-identical to structured.render_prompt."
)

_FO_CHECKING_RULE_TEXT = (
    "Canonical JSON is structured._canonical_json: json.dumps "
    "with sort_keys=True, separators=(', ', ': '), "
    "ensure_ascii=True, allow_nan=False; the document "
    "{'name': 'Ada Lovelace', 'age': 36} canonicalises to "
    '\'{"age": 36, "name": "Ada Lovelace"}\'. The answer is '
    "parsed once with a duplicate-detecting object_pairs_hook "
    "(last occurrence wins). document_kind is 'object', "
    "'non_object', or 'unparseable' (JSON parse failure only). "
    "field_values maps each produced key to its canonical JSON "
    "string, or to None when the value has no canonical form (a "
    "non-finite number, or a value beyond the int/str digit "
    "limit); a produced JSON null is the string 'null', never "
    "None. document_digest is non-None iff kind is 'object' and "
    "no field value is None, and then equals the SHA-256 of the "
    "document's canonical form. A field is correct iff its key "
    "is present with a string whose digest equals the expected "
    "value's canonical digest. The schema verdict is recomputed "
    "from the reconstructed document with the flat schema check; "
    "a None value fails its key's check before reconstruction. "
    "Any exception while parsing or canonicalising a stored "
    "string makes the payload section invalid. parse_ok iff kind "
    "is 'object' or 'non_object'; adherent iff parse_ok, the "
    "schema verdict is ok, and duplicate_keys is false. Pairing: "
    "paired flag set, endpoint orders differ, both endpoint "
    "records adherent, neither with duplicate keys. Decision: "
    "exact two-sided sign test over discordant paired specs; "
    "falsified iff discordant >= 6 and p <= 0.05; insufficient "
    "iff discordant < 6; else not_falsified. Per-record field "
    "values are capped at 4096 summed chars; a live record over "
    "the cap is stored with detail 'arm_evaluation_failed' and "
    "empty field values."
)

_FO_ARM_KEYS: tuple[str, ...] = ("canonical", "answer_first", "answer_last", "reversed")
_FO_MAX_RECORDS = 60 * 24
_FO_MAX_PAYLOAD_BYTES = 8 * 1024 * 1024
_FO_FIELD_VALUES_CAP = 4096
_FO_MIN_DISCORDANT_SPECS = 6
_FO_SIGN_ALPHA = 0.05
_FO_RECORD_KEYS = {
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
_FO_RECORD_DETAILS = {"", "arm_evaluation_failed", "arm_context_exceeded"}
_FO_AGGREGATE_KEYS = {
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
}


def _fo_digest(text: str) -> str:
    """Mirror of fieldorder.digest_text (SHA-256 hex)."""
    import hashlib as _hashlib

    return _hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fo_canonical_json(document: Any) -> str:
    """Mirror of structured._canonical_json (the pinned form)."""
    import json as _json

    return _json.dumps(document, sort_keys=True, separators=(", ", ": "), allow_nan=False)


def _fo_digest_ok(value: Any) -> bool:
    if type(value) is not str or len(value) != 64:
        return False
    return all(char in "0123456789abcdef" for char in value)


def _fo_canonical_order(spec: dict[str, Any]) -> list[str]:
    return sorted(spec["schema"]["properties"])


def _fo_arm_order(spec: dict[str, Any], arm: str) -> list[str]:
    order = _fo_canonical_order(spec)
    if arm == "canonical":
        return order
    if arm == "reversed":
        return order[::-1]
    field = spec["answer_field"]
    rest = [name for name in order if name != field]
    if arm == "answer_first":
        return [field, *rest]
    if arm == "answer_last":
        return [*rest, field]
    raise ValueError(f"unknown field-order arm: {arm!r}")


def _fo_describe_ordered(schema: dict[str, Any], order: list[str]) -> str:
    properties = schema["properties"]
    parts: list[str] = []
    for name in order:
        field = properties[name]
        piece = f'"{name}": {field["type"]}'
        enum = field.get("enum")
        if enum is not None:
            piece += " one of " + _fo_canonical_json(enum)
        minimum = field.get("minimum")
        maximum = field.get("maximum")
        if minimum is not None or maximum is not None:
            lo = f">= {minimum}" if minimum is not None else ""
            hi = f"<= {maximum}" if maximum is not None else ""
            piece += f" ({' and '.join(part for part in (lo, hi) if part)})"
        parts.append(piece)
    required_names = set(schema["required"])
    required = [name for name in order if name in required_names]
    extras = (
        "no other keys allowed" if schema["additionalProperties"] is False else "other keys allowed"
    )
    return "{" + "; ".join(parts) + "} (required: " + ", ".join(required) + "; " + extras + ")"


def _fo_arm_prompt(spec: dict[str, Any], arm: str) -> str:
    return (
        f"{spec['question']}\n"
        f"Context: {spec['context']}\n"
        "Respond with ONLY a JSON object matching this schema: "
        f"{_fo_describe_ordered(spec['schema'], _fo_arm_order(spec, arm))}.\n"
        "Do not include any other text.\n"
        f"(ref: {spec['source']}/{spec['id']})"
    )


def _fo_spec_by_id(spec_id: str) -> dict[str, Any]:
    for spec in _FIELD_ORDER_SPECS:
        if spec["id"] == spec_id:
            return spec
    raise ValueError(f"unknown field-order spec: {spec_id!r}")


def _fo_expected_value_digest(spec: dict[str, Any], field: str) -> str:
    return _fo_digest(_fo_canonical_json(spec["answer"][field]))


def _fo_field_correct(spec: dict[str, Any], field: str, field_values: dict[str, Any]) -> bool:
    value = field_values.get(field)
    if type(value) is not str:
        return False
    return _fo_digest(value) == _fo_expected_value_digest(spec, field)


def _fo_is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return value == value and abs(value) != float("inf")


def _fo_field_value_ok(value: object, field: dict[str, Any]) -> bool:
    """Mirror of structured._field_value_ok."""
    field_type = field["type"]
    if field_type == "string":
        ok = type(value) is str
    elif field_type == "integer":
        ok = type(value) is int and not isinstance(value, bool)
    elif field_type == "number":
        ok = (
            type(value) in (int, float)
            and not isinstance(value, bool)
            and _fo_is_finite_number(value)
        )
    elif field_type == "boolean":
        ok = type(value) is bool
    elif field_type == "null":
        ok = value is None
    else:
        return False
    if not ok:
        return False
    enum = field.get("enum")
    if enum is not None and value not in enum:
        return False
    minimum = field.get("minimum")
    if minimum is not None and isinstance(minimum, (int, float)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        if value < minimum:
            return False
    maximum = field.get("maximum")
    if maximum is not None and isinstance(maximum, (int, float)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        return value <= maximum
    return True


def _fo_check_schema(document: object, schema: dict[str, Any]) -> str:
    """Mirror of structured.check_schema over a normalized schema."""
    if type(document) is not dict:
        return "schema_failed"
    required = schema["required"]
    properties = schema["properties"]
    for name in required:
        if name not in document:
            return "schema_failed"
    for name, value in document.items():
        field = properties.get(name)
        if field is None:
            if schema["additionalProperties"] is False:
                return "schema_failed"
            continue
        if not _fo_field_value_ok(value, field):
            return "schema_failed"
    return "schema_ok"


def _fo_parse_stored(value: Any) -> tuple[bool, Any]:
    """Parse a stored canonical string; (False, None) on any failure.

    Total (design F10): ValueError (including the int/str digit
    limit), TypeError, and RecursionError all yield False.
    """
    import json as _json

    if type(value) is not str:
        return False, None
    try:
        return True, _json.loads(value)
    except (ValueError, TypeError, RecursionError):
        return False, None


def _fo_schema_ok(spec: dict[str, Any], kind: str, field_values: dict[str, Any]) -> bool:
    if kind != "object":
        return False
    if any(value is None for value in field_values.values()):
        return False
    document: dict[str, Any] = {}
    for key, value in field_values.items():
        ok, parsed = _fo_parse_stored(value)
        if not ok:
            return False
        document[key] = parsed
    return _fo_check_schema(document, spec["schema"]) == "schema_ok"


def _fo_record_derived(spec: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    kind = record["document_kind"]
    field_values = record["field_values"]
    parse_ok = kind in ("object", "non_object")
    ok = _fo_schema_ok(spec, kind, field_values)
    return {
        "parse_ok": parse_ok,
        "schema_ok": ok,
        "adherent": parse_ok and ok and not record["duplicate_keys"],
        "fields": {
            name: _fo_field_correct(spec, name, field_values)
            for name in spec["schema"]["properties"]
        },
        "answer_correct": _fo_field_correct(spec, spec["answer_field"], field_values),
        "whole_doc_correct": ok
        and all(
            _fo_field_correct(spec, name, field_values) for name in spec["schema"]["properties"]
        )
        and not any(key not in spec["schema"]["properties"] for key in field_values),
    }


def _fo_sign_test_p(n_pos: int, n_neg: int) -> float:
    import math as _math

    n = n_pos + n_neg
    if n == 0:
        return 1.0
    k = min(n_pos, n_neg)
    tail = sum(_math.comb(n, i) for i in range(k + 1)) / (2**n)
    return float(min(1.0, 2.0 * tail))


def _fo_rate(correct: int, n: int) -> float | None:
    return float(correct / n) if n else None


def _fo_decision(n_first_only: int, n_last_only: int) -> dict[str, Any]:
    discordant = n_first_only + n_last_only
    p_value = _fo_sign_test_p(n_first_only, n_last_only)
    if discordant < _FO_MIN_DISCORDANT_SPECS:
        verdict = "insufficient"
    elif p_value <= _FO_SIGN_ALPHA:
        verdict = "falsified"
    else:
        verdict = "not_falsified"
    if n_first_only > n_last_only:
        direction = "first"
    elif n_last_only > n_first_only:
        direction = "last"
    else:
        direction = "none"
    return {
        "verdict": verdict,
        "n_discordant": discordant,
        "n_first_only": n_first_only,
        "n_last_only": n_last_only,
        "sign_p": p_value,
        "direction": direction,
    }


def _fo_aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Mirror of fieldorder.aggregate_candidate."""
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        by_key[(record["spec_id"], record["arm"])] = record
    n_generated = len(records)
    n_distinct_prompts = sum(
        len({tuple(_fo_arm_order(spec, arm)) for arm in _FO_ARM_KEYS})
        for spec in _FIELD_ORDER_SPECS
    )
    n_boundary_screened = sum(1 for r in records if r["case_screened"])
    n_detail_screened = sum(1 for r in records if not r["case_screened"] and r["detail"] != "")
    n_no_boundary = sum(
        1
        for r in records
        if not r["case_screened"] and r["detail"] == "" and not r["boundary_found"]
    )
    live = [
        r for r in records if not r["case_screened"] and r["detail"] == "" and r["boundary_found"]
    ]
    specs = {spec["id"]: spec for spec in _FIELD_ORDER_SPECS}
    derived = {(r["spec_id"], r["arm"]): _fo_record_derived(specs[r["spec_id"]], r) for r in live}

    by_arm: dict[str, Any] = {}
    for arm in _FO_ARM_KEYS:
        rows = [r for r in live if r["arm"] == arm]
        n = len(rows)
        parse_ok = sum(1 for r in rows if derived[(r["spec_id"], arm)]["parse_ok"])
        schema_ok_n = sum(1 for r in rows if derived[(r["spec_id"], arm)]["schema_ok"])
        adherent = sum(1 for r in rows if derived[(r["spec_id"], arm)]["adherent"])
        by_arm[arm] = {
            "n": n,
            "parse_ok": parse_ok,
            "schema_ok": schema_ok_n,
            "adherent": adherent,
            "parse_rate": _fo_rate(parse_ok, n),
            "schema_rate": _fo_rate(schema_ok_n, n),
            "adherent_rate": _fo_rate(adherent, n),
        }

    per_field: dict[str, Any] = {}
    for spec in _FIELD_ORDER_SPECS:
        fields: dict[str, Any] = {}
        for name in spec["schema"]["properties"]:
            per_arm: dict[str, Any] = {}
            for arm in _FO_ARM_KEYS:
                rows = [r for r in live if r["spec_id"] == spec["id"] and r["arm"] == arm]
                n = len(rows)
                correct = sum(1 for r in rows if derived[(spec["id"], arm)]["fields"][name])
                per_arm[arm] = {"correct": correct, "n": n, "rate": _fo_rate(correct, n)}
            fields[name] = per_arm
        per_field[spec["id"]] = fields

    answer_by_arm: dict[str, Any] = {}
    whole_doc_by_arm: dict[str, Any] = {}
    for arm in _FO_ARM_KEYS:
        rows = [r for r in live if r["arm"] == arm]
        n = len(rows)
        correct = sum(1 for r in rows if derived[(r["spec_id"], arm)]["answer_correct"])
        whole = sum(1 for r in rows if derived[(r["spec_id"], arm)]["whole_doc_correct"])
        answer_by_arm[arm] = {"correct": correct, "n": n, "rate": _fo_rate(correct, n)}
        whole_doc_by_arm[arm] = {"correct": whole, "n": n, "rate": _fo_rate(whole, n)}

    n_duplicates = sum(1 for r in live if r["duplicate_keys"])

    paired_first: dict[str, bool] = {}
    paired_last: dict[str, bool] = {}
    n_flagged = 0
    n_order_same = 0
    n_dup_excluded = 0
    n_not_adherent = 0
    ungated_first_only = 0
    ungated_last_only = 0
    ungated_specs = 0
    for spec in _FIELD_ORDER_SPECS:
        if not spec["paired"]:
            continue
        n_flagged += 1
        if _fo_arm_order(spec, "answer_first") == _fo_arm_order(spec, "answer_last"):
            n_order_same += 1
            continue
        first = by_key.get((spec["id"], "answer_first"))
        last = by_key.get((spec["id"], "answer_last"))
        if first is None or last is None:
            continue
        first_live = (spec["id"], "answer_first") in derived
        last_live = (spec["id"], "answer_last") in derived
        if first_live and last_live:
            ungated_specs += 1
            f_ok = derived[(spec["id"], "answer_first")]["answer_correct"]
            l_ok = derived[(spec["id"], "answer_last")]["answer_correct"]
            if f_ok and not l_ok:
                ungated_first_only += 1
            elif l_ok and not f_ok:
                ungated_last_only += 1
        if first["duplicate_keys"] or last["duplicate_keys"]:
            n_dup_excluded += 1
            continue
        if not (
            first_live
            and last_live
            and derived[(spec["id"], "answer_first")]["adherent"]
            and derived[(spec["id"], "answer_last")]["adherent"]
        ):
            n_not_adherent += 1
            continue
        paired_first[spec["id"]] = derived[(spec["id"], "answer_first")]["answer_correct"]
        paired_last[spec["id"]] = derived[(spec["id"], "answer_last")]["answer_correct"]

    n_paired = len(paired_first)
    n_first_only = sum(1 for s in paired_first if paired_first[s] and not paired_last[s])
    n_last_only = sum(1 for s in paired_first if paired_last[s] and not paired_first[s])
    first_correct = sum(1 for s in paired_first if paired_first[s])
    last_correct = sum(1 for s in paired_first if paired_last[s])
    decision = _fo_decision(n_first_only, n_last_only)
    decision["n_paired"] = n_paired

    extraction_ids = [
        s["id"] for s in _FIELD_ORDER_SPECS if s["paired"] and s["shape"] == "extraction"
    ]
    ex_first_only = sum(
        1 for s in extraction_ids if s in paired_first and paired_first[s] and not paired_last[s]
    )
    ex_last_only = sum(
        1 for s in extraction_ids if s in paired_first and paired_last[s] and not paired_first[s]
    )
    extraction_decision = _fo_decision(ex_first_only, ex_last_only)
    extraction_decision["n_paired"] = sum(1 for s in extraction_ids if s in paired_first)

    def _field_count_split(min_fields: int, max_fields: int | None) -> dict[str, Any]:
        ids = [
            s["id"]
            for s in _FIELD_ORDER_SPECS
            if s["id"] in paired_first
            and len(s["schema"]["properties"]) >= min_fields
            and (max_fields is None or len(s["schema"]["properties"]) <= max_fields)
        ]
        return {
            "n_paired": len(ids),
            "n_first_only": sum(1 for s in ids if paired_first[s] and not paired_last[s]),
            "n_last_only": sum(1 for s in ids if paired_last[s] and not paired_first[s]),
        }

    return {
        "n_generated": n_generated,
        "n_distinct_prompts": n_distinct_prompts,
        "n_boundary_screened": n_boundary_screened,
        "n_detail_screened": n_detail_screened,
        "n_no_boundary": n_no_boundary,
        "n_live": len(live),
        "n_duplicate_keys": n_duplicates,
        "by_arm": by_arm,
        "per_field": per_field,
        "answer_by_arm": answer_by_arm,
        "whole_doc_by_arm": whole_doc_by_arm,
        "pairing": {
            "n_flagged": n_flagged,
            "n_order_same": n_order_same,
            "n_duplicate_excluded": n_dup_excluded,
            "n_not_adherent": n_not_adherent,
            "n_paired": n_paired,
            "first_correct": first_correct,
            "last_correct": last_correct,
            "first_rate": _fo_rate(first_correct, n_paired),
            "last_rate": _fo_rate(last_correct, n_paired),
            "delta": (first_correct / n_paired - last_correct / n_paired if n_paired else None),
            "ungated": {
                "n_specs": ungated_specs,
                "n_first_only": ungated_first_only,
                "n_last_only": ungated_last_only,
                "n_discordant": ungated_first_only + ungated_last_only,
            },
        },
        "decision": decision,
        "extraction_decision": extraction_decision,
        "field_count_split": {
            "two": _field_count_split(2, 2),
            "three_plus": _field_count_split(3, None),
        },
    }


def _fo_protocol(seed: int, backend_fingerprint: str) -> dict[str, Any]:
    """Mirror of fieldorder.protocol_object plus its fingerprint."""
    protocol: dict[str, Any] = {
        "seed": seed,
        "backend_fingerprint": backend_fingerprint,
        "spec_count": 15,
        "arm_keys": list(_FO_ARM_KEYS),
        "inventory_digest": _fo_digest(_fo_canonical_json(list(_FIELD_ORDER_SPECS))),
        "generation_rule_digest": _fo_digest(_FO_GENERATION_RULE_TEXT),
        "rendering_rule_digest": _fo_digest(_FO_RENDERING_RULE_TEXT),
        "checking_rule_digest": _fo_digest(_FO_CHECKING_RULE_TEXT),
        "max_records": _FO_MAX_RECORDS,
        "max_payload_bytes": _FO_MAX_PAYLOAD_BYTES,
        "max_new_tokens": 64,
        "field_values_cap": _FO_FIELD_VALUES_CAP,
        "min_discordant_specs": _FO_MIN_DISCORDANT_SPECS,
    }
    protocol["fingerprint"] = _fo_digest(_fo_canonical_json(protocol))
    return protocol


def _fo_recompute(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Recompute a field-order payload from its ground records.

    The validator carries its own spec table, rendering, schema
    check, and aggregator (no imports from fieldorder.py): the
    protocol is rebuilt from the payload's seed and backend
    fingerprint and compared whole; every record's content
    digest is compared against the re-rendered arm prompt;
    every stored field value must be exactly the canonical
    form of its parse, the document digest must be consistent,
    and every readout and the endpoint decision is re-derived
    from the commitments and compared.
    """
    import json as _json

    protocol = payload.get("protocol")
    if type(protocol) is not dict:
        return None
    seed = protocol.get("seed")
    backend_fingerprint = protocol.get("backend_fingerprint")
    if type(seed) is not int or isinstance(seed, bool) or type(backend_fingerprint) is not str:
        return None
    if not _recomputed_equal(protocol, _fo_protocol(seed, backend_fingerprint)):
        return None
    candidates = payload.get("candidates")
    if type(candidates) is not dict or not candidates:
        return None
    total_records = 0
    records_bytes = 0
    aggregates: dict[str, Any] = {}
    expected_order = [(spec["id"], arm) for spec in _FIELD_ORDER_SPECS for arm in _FO_ARM_KEYS]
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
        if set(entry) != _FO_AGGREGATE_KEYS | {"status", "records"}:
            return None
        records = entry["records"]
        if type(records) is not list or len(records) != len(expected_order):
            return None
        total_records += len(records)
        records_bytes += len(
            _json.dumps(records, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        )
        if records_bytes > _FO_MAX_PAYLOAD_BYTES:
            return None
        for record, (spec_id, arm) in zip(records, expected_order, strict=True):
            if type(record) is not dict or set(record) != _FO_RECORD_KEYS:
                return None
            if record["candidate"] != name:
                return None
            if record["spec_id"] != spec_id or record["arm"] != arm:
                return None
            if record["detail"] not in _FO_RECORD_DETAILS:
                return None
            if type(record["case_screened"]) is not bool:
                return None
            if type(record["boundary_found"]) is not bool:
                return None
            if type(record["duplicate_keys"]) is not bool:
                return None
            if type(record["input_tokens"]) is not int or record["input_tokens"] < 0:
                return None
            if record["answer_digest"] is not None and not _fo_digest_ok(record["answer_digest"]):
                return None
            if not _fo_digest_ok(record["content_digest"]):
                return None
            spec = _fo_spec_by_id(spec_id)
            if record["content_digest"] != _fo_digest(_fo_arm_prompt(spec, arm)):
                return None
            kind = record["document_kind"]
            if kind not in ("", "object", "non_object", "unparseable"):
                return None
            field_values = record["field_values"]
            if type(field_values) is not dict:
                return None
            if record["detail"] != "" or record["case_screened"]:
                # A screened or degraded record carries no
                # generation: no document, no answer digest, no
                # boundary, no duplicates.
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
            # A completed record carries the generation's answer
            # digest (its boundary may or may not have been
            # found; that is a measurement).
            if record["answer_digest"] is None:
                return None
            if (
                sum(len(value) for value in field_values.values() if type(value) is str)
                > _FO_FIELD_VALUES_CAP
            ):
                return None
            for value in field_values.values():
                if value is None:
                    continue
                ok, parsed = _fo_parse_stored(value)
                if not ok:
                    return None
                try:
                    if _fo_canonical_json(parsed) != value:
                        return None
                except (ValueError, TypeError, RecursionError):
                    return None
            if kind != "object":
                if field_values or record["document_digest"] is not None:
                    return None
            elif any(value is None for value in field_values.values()):
                if record["document_digest"] is not None:
                    return None
            else:
                document: dict[str, Any] = {}
                for key, value in field_values.items():
                    ok, parsed = _fo_parse_stored(value)
                    if not ok:
                        return None
                    document[key] = parsed
                try:
                    expected_digest = _fo_digest(_fo_canonical_json(document))
                except (ValueError, TypeError, RecursionError):
                    return None
                if record["document_digest"] != expected_digest:
                    return None
        aggregates[name] = _fo_aggregate(records)
    if total_records > _FO_MAX_RECORDS:
        return None
    if not aggregates:
        return None
    return {
        "aggregates": aggregates,
        "n_candidates": len(candidates),
        "n_specs": len(_FIELD_ORDER_SPECS),
    }


def _valid_field_order(payload: Any) -> bool:
    """Structural check for a ``field_order`` payload (recompute standard)."""
    if type(payload) is not dict or payload.get("available") is not True:
        return False
    if set(payload) != {
        "available",
        "detail",
        "protocol",
        "candidates",
        "n_candidates",
        "n_specs",
    }:
        return False
    if payload.get("detail") != "":
        return False
    recomputed = _fo_recompute(payload)
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
    return _recomputed_equal(payload.get("n_specs"), recomputed["n_specs"])


def diff_field_order(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``field_order`` probe payloads; deterministic.

    Answers "did the field-order story change between runs?":
    per-candidate endpoint verdict and direction drift, the
    discordance counts and paired rates, answer-field accuracy
    by arm, the screening and duplicate accounting, candidates
    added/removed, and protocol / inventory / backend drift
    (any of which marks the comparison not comparable, with
    the drift still listed -- the spec set is the treatment, so
    a changed inventory is protocol drift). A side without a
    valid payload is reported plainly.
    """
    old_ok = _valid_field_order(old)
    new_ok = _valid_field_order(new)
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
    fo_protocol_drift = old["protocol"].get("fingerprint") != new["protocol"].get("fingerprint")
    fo_generator_drift = old["protocol"].get("inventory_digest") != new["protocol"].get(
        "inventory_digest"
    )
    fo_backend_drift = old["protocol"].get("backend_fingerprint") != new["protocol"].get(
        "backend_fingerprint"
    )
    fo_comparable = not fo_protocol_drift and not fo_generator_drift and not fo_backend_drift
    old_candidates = old["candidates"]
    new_candidates = new["candidates"]
    fo_candidates_added = sorted(set(new_candidates) - set(old_candidates))
    fo_candidates_removed = sorted(set(old_candidates) - set(new_candidates))
    fo_candidate_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_candidates) & set(new_candidates)):
        old_entry = old_candidates[name]
        new_entry = new_candidates[name]
        if old_entry.get("status") != "available" or new_entry.get("status") != "available":
            if old_entry.get("status") != new_entry.get("status"):
                fo_candidate_drift.append(
                    {
                        "candidate": name,
                        "metric": "status",
                        "old": old_entry.get("status"),
                        "new": new_entry.get("status"),
                    }
                )
            continue
        fo_metrics: list[tuple[str, Any, Any]] = [
            ("verdict", old_entry["decision"]["verdict"], new_entry["decision"]["verdict"]),
            ("direction", old_entry["decision"]["direction"], new_entry["decision"]["direction"]),
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
            ("last_rate", old_entry["pairing"]["last_rate"], new_entry["pairing"]["last_rate"]),
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
        for arm in _FO_ARM_KEYS:
            fo_metrics.append(
                (
                    f"answer_rate_{arm}",
                    old_entry["answer_by_arm"][arm]["rate"],
                    new_entry["answer_by_arm"][arm]["rate"],
                )
            )
        for metric, old_value, new_value in fo_metrics:
            if old_value != new_value:
                fo_candidate_drift.append(
                    {
                        "candidate": name,
                        "metric": metric,
                        "old": old_value,
                        "new": new_value,
                    }
                )
    result["candidate_drift"] = fo_candidate_drift
    result["candidates_added"] = fo_candidates_added
    result["candidates_removed"] = fo_candidates_removed
    result["protocol_drift"] = fo_protocol_drift
    result["generator_drift"] = fo_generator_drift
    result["backend_drift"] = fo_backend_drift
    result["comparable"] = fo_comparable
    result["changed"] = bool(
        fo_candidate_drift
        or fo_candidates_added
        or fo_candidates_removed
        or fo_protocol_drift
        or fo_generator_drift
        or fo_backend_drift
    )
    return result


_CA_K_DONORS = 3
_CA_RANKINGS = ("total", "byte", "mean")
_CA_MIN_COMMON_EVALUABLE = 6
_CA_MAX_RECORDS = 2400
_CA_MAX_RECORDS_BYTES = 12 * 1024 * 1024
_CA_DONOR_RULE = "smallest-eligible-id-first-three-v1"
_CA_SCREEN = "scorerlens-r4-plus-containment-v1"
_CA_EXCLUSIONS = (None, "blank_expected")
_CA_AGGREGATE_KEYS = {
    "n_fixtures",
    "n_constructible",
    "n_evaluable",
    "n_choice_unavailable",
    "n_prompt_failures",
    "unavailable_reasons",
    "n_generation_missing",
    "accuracy",
    "discordance",
    "strata",
}
_CA_METHOD_KEYS = {
    "n_common",
    "floor",
    "withheld",
    "detail",
    "common_accuracy",
    "choice_top_candidate",
    "choice_top_tied",
    "primary_decision",
    "primary_winner",
    "agreement",
}


def _ca_f32(value: float) -> float:
    """Mirror of scoring._f32 (struct-based float32 rounding)."""
    import math as _math
    import struct as _struct

    if _math.isfinite(value) and abs(value) > float.fromhex("0x1.fffffep+127"):
        return _math.copysign(_math.inf, value)
    rounded: float = _struct.unpack("<f", _struct.pack("<f", value))[0]
    return rounded


def _ca_donor_map_sha256(block: list[dict[str, Any]]) -> str:
    canonical = json.dumps(block, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _ca_fixture_set_sha256(fixture_ids: list[str]) -> str | None:
    """SHA-256 over the sorted fixture id set, or None when an id
    cannot be UTF-8 encoded (a lone surrogate): the expected
    protocol then mismatches any recorded payload, so the recompute
    fails closed instead of raising (cycle-190 discipline)."""
    try:
        return hashlib.sha256("\n".join(sorted(fixture_ids)).encode("utf-8")).hexdigest()
    except UnicodeEncodeError:
        return None


def _ca_protocol_object(
    fixture_ids: list[str], block: list[dict[str, Any]], backend_fingerprint: str
) -> dict[str, Any]:
    return {
        "version": 1,
        "donor_rule": _CA_DONOR_RULE,
        "screen": _CA_SCREEN,
        "k_donors": _CA_K_DONORS,
        "rankings": list(_CA_RANKINGS),
        "min_common_evaluable": _CA_MIN_COMMON_EVALUABLE,
        "max_records": _CA_MAX_RECORDS,
        "max_records_bytes": _CA_MAX_RECORDS_BYTES,
        "fixture_set_sha256": _ca_fixture_set_sha256(fixture_ids),
        "donor_map_sha256": _ca_donor_map_sha256(block),
        "backend_fingerprint": backend_fingerprint,
    }


def _ca_generation_verdict(status: Any, exact_match: Any, boundary_found: Any) -> str:
    if status != "completed":
        return "missing"
    return "pass" if (exact_match is True and boundary_found is True) else "fail"


def _ca_ranking_values(cells: list[dict[str, Any]], ranking: str) -> list[float]:
    if ranking == "total":
        return [float(cell["total"]) for cell in cells]
    if ranking == "mean":
        return [float(cell["mean"]) for cell in cells]
    return [float(cell["total"]) / int(cell["bytes"]) for cell in cells]


def _ca_rank_verdict(values: list[float]) -> str:
    top = max(values)
    if values[0] < top:
        return "incorrect"
    return "tied" if sum(1 for value in values if value == top) > 1 else "correct"


def _ca_record_evaluable(record: dict[str, Any], fixture_entry: dict[str, Any]) -> bool:
    import math as _math

    if not fixture_entry["constructible"]:
        return False
    cells = record["cells"]
    if len(cells) != 1 + _CA_K_DONORS:
        return False
    return all(
        cell["status"] == "ok"
        and type(cell["total"]) is float
        and _math.isfinite(cell["total"])
        and type(cell["mean"]) is float
        and _math.isfinite(cell["mean"])
        for cell in cells
    )


def _ca_accuracy_block(verdicts: list[str]) -> dict[str, Any]:
    correct = sum(1 for verdict in verdicts if verdict == "correct")
    incorrect = sum(1 for verdict in verdicts if verdict == "incorrect")
    tied = sum(1 for verdict in verdicts if verdict == "tied")
    n = len(verdicts)
    return {
        "correct": correct,
        "incorrect": incorrect,
        "tied": tied,
        "n": n,
        "accuracy": (correct / n) if n else None,
        "detail": "" if n else "no_evaluable_fixtures",
    }


def _ca_aggregate(
    records: list[dict[str, Any]], fixtures: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    """Mirror of choiceaudit.aggregate_candidate (public keys + verdicts)."""
    by_id = {entry["fixture"]: entry for entry in fixtures}
    verdicts: dict[str, list[str]] = {ranking: [] for ranking in _CA_RANKINGS}
    verdict_by_fixture: dict[str, dict[str, str]] = {}
    unavailable_reasons: dict[str, int] = {}
    n_constructible = 0
    n_evaluable = 0
    n_prompt_failures = 0
    n_generation_missing = 0
    paired = {
        "gen_pass_choice_correct": 0,  # nosec B105 -- verdict-count schema key, not a credential
        "gen_pass_choice_incorrect": 0,  # nosec B105 -- verdict-count schema key, not a credential
        "gen_fail_choice_correct": 0,
        "gen_fail_choice_incorrect": 0,
    }
    n_choice_tied = 0
    gen_fail_boundary_found = 0
    gen_fail_boundary_not_found = 0
    strata_verdicts: dict[str, list[str]] = {
        "numeric": [],
        "non_numeric": [],
        "length_matched": [],
    }
    agreement: dict[str, dict[str, int]] = {
        "byte_vs_total": {"agree": 0, "disagree": 0},
        "mean_vs_total": {"agree": 0, "disagree": 0},
    }
    for record in records:
        entry = by_id[record["fixture"]]
        gen = _ca_generation_verdict(
            record["obs_status"], record["exact_match"], record["boundary_found"]
        )
        if gen == "missing":
            n_generation_missing += 1
        if not entry["constructible"]:
            continue
        n_constructible += 1
        if record["prompt_reason"] is not None:
            n_prompt_failures += 1
            reason = str(record["prompt_reason"])
            unavailable_reasons[reason] = unavailable_reasons.get(reason, 0) + 1
        if not _ca_record_evaluable(record, entry):
            for cell in record["cells"]:
                if cell["status"] == "unavailable":
                    reason = str(cell["reason"])
                    unavailable_reasons[reason] = unavailable_reasons.get(reason, 0) + 1
                elif cell["status"] != "ok":
                    unavailable_reasons[cell["status"]] = (
                        unavailable_reasons.get(cell["status"], 0) + 1
                    )
            continue
        n_evaluable += 1
        cells = record["cells"]
        per_ranking = {
            ranking: _ca_rank_verdict(_ca_ranking_values(cells, ranking))
            for ranking in _CA_RANKINGS
        }
        verdict_by_fixture[record["fixture"]] = per_ranking
        for ranking in _CA_RANKINGS:
            verdicts[ranking].append(per_ranking[ranking])
        strata_verdicts["numeric" if entry["numeric"] else "non_numeric"].append(
            per_ranking["total"]
        )
        true_tokens = int(cells[0]["tokens"])
        if all(abs(int(cell["tokens"]) - true_tokens) <= 1 for cell in cells[1:]):
            strata_verdicts["length_matched"].append(per_ranking["total"])
        for key, ranking in (("byte_vs_total", "byte"), ("mean_vs_total", "mean")):
            same = per_ranking[ranking] == per_ranking["total"]
            agreement[key]["agree" if same else "disagree"] += 1
        if gen in ("pass", "fail"):
            total_verdict = per_ranking["total"]
            if total_verdict == "tied":
                n_choice_tied += 1
            else:
                paired[f"gen_{gen}_choice_{total_verdict}"] += 1
                if gen == "fail":
                    if record["boundary_found"] is True:
                        gen_fail_boundary_found += 1
                    else:
                        gen_fail_boundary_not_found += 1
    n_paired = sum(paired.values())
    n_discordant = paired["gen_pass_choice_incorrect"] + paired["gen_fail_choice_correct"]
    aggregates = {
        "n_fixtures": len(fixtures),
        "n_constructible": n_constructible,
        "n_evaluable": n_evaluable,
        "n_choice_unavailable": n_constructible - n_evaluable,
        "n_prompt_failures": n_prompt_failures,
        "unavailable_reasons": unavailable_reasons,
        "n_generation_missing": n_generation_missing,
        "accuracy": {ranking: _ca_accuracy_block(verdicts[ranking]) for ranking in _CA_RANKINGS},
        "discordance": {
            "n_paired": n_paired,
            **paired,
            "n_discordant": n_discordant,
            "n_choice_tied": n_choice_tied,
            "gen_fail_boundary_found": gen_fail_boundary_found,
            "gen_fail_boundary_not_found": gen_fail_boundary_not_found,
        },
        "strata": {
            "numeric": _ca_accuracy_block(strata_verdicts["numeric"]),
            "non_numeric": _ca_accuracy_block(strata_verdicts["non_numeric"]),
            "length_matched": _ca_accuracy_block(strata_verdicts["length_matched"]),
            "ranking_agreement": agreement,
        },
    }
    return aggregates, verdict_by_fixture


def _ca_method_comparison(
    verdicts_by_candidate: dict[str, dict[str, dict[str, str]]],
    candidate_names: list[str],
    primary_decision: Any,
    primary_winner: Any,
) -> dict[str, Any]:
    """Mirror of choiceaudit.method_comparison over recomputed verdicts."""
    evaluable_sets = [set(verdicts_by_candidate.get(name, {})) for name in candidate_names]
    common = set.intersection(*evaluable_sets) if evaluable_sets else set()
    n_common = len(common)
    common_accuracy: dict[str, Any] = {}
    for name in candidate_names:
        per_fixture = verdicts_by_candidate.get(name, {})
        correct = sum(
            1 for fixture in common if per_fixture.get(fixture, {}).get("total") == "correct"
        )
        common_accuracy[name] = {
            "correct": correct,
            "n": n_common,
            "accuracy": (correct / n_common) if n_common else None,
        }
    if primary_decision in ("compatible_on_fixture_set", "held_out_failed"):
        mapped_decision: Any = "selected"
        mapped_winner: Any = primary_winner
    else:
        mapped_decision = primary_decision
        mapped_winner = primary_winner
    withheld = n_common < _CA_MIN_COMMON_EVALUABLE
    top_candidate: str | None = None
    top_tied: list[str] = []
    if not withheld and n_common:
        best = max(common_accuracy[name]["correct"] for name in candidate_names)
        leaders = sorted(
            name for name in candidate_names if common_accuracy[name]["correct"] == best
        )
        if len(leaders) == 1:
            top_candidate = leaders[0]
        else:
            top_tied = leaders
    if withheld:
        agreement = "withheld"
    elif mapped_decision != "selected" or type(mapped_winner) is not str:
        agreement = "native_abstain"
    elif top_tied:
        agreement = "choice_tied"
    elif top_candidate == mapped_winner:
        agreement = "agree"
    else:
        agreement = "disagree"
    return {
        "n_common": n_common,
        "floor": _CA_MIN_COMMON_EVALUABLE,
        "withheld": withheld,
        "detail": "choiceaudit_common_population_below_floor" if withheld else "",
        "common_accuracy": common_accuracy,
        "choice_top_candidate": top_candidate,
        "choice_top_tied": top_tied,
        "primary_decision": mapped_decision,
        "primary_winner": mapped_winner,
        "agreement": agreement,
    }


def _choiceaudit_recompute(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Recompute a choice-audit payload from its ground records.

    The validator carries its own mirrors (no imports from
    choiceaudit.py): the protocol is rebuilt from the stored
    construction block and compared whole, the donor-map digest
    is recomputed over the stored block, every stored float is
    asserted float32-representable, and every derived value --
    per-ranking verdicts and ties, accuracies, the discordance
    table and its boundary split, the strata, the common
    population, the method-level agreement -- is recomputed
    from the per-cell reported values. What this cannot verify
    is stated in docs/CHOICE_AUDIT_DESIGN.md: the equivalence
    screen and donor provenance against the fixture texts
    (records carry digests, not texts).
    """
    import math as _math

    protocol = payload.get("protocol")
    block = payload.get("fixtures")
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
        if entry["excluded"] not in _CA_EXCLUSIONS:
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
        if entry["constructible"] != (len(choices) == 1 + _CA_K_DONORS):
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
    backend_fingerprint = protocol.get("backend_fingerprint")
    if type(backend_fingerprint) is not str:
        return None
    expected_protocol = _ca_protocol_object(fixture_ids, block, backend_fingerprint)
    expected_protocol["fingerprint"] = hashlib.sha256(
        json.dumps(
            expected_protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()
    if not _recomputed_equal(protocol, expected_protocol):
        return None
    candidates = payload.get("candidates")
    if type(candidates) is not dict or not candidates:
        return None
    total_records = 0
    records_bytes = 0
    aggregates: dict[str, Any] = {}
    verdicts_by_candidate: dict[str, dict[str, dict[str, str]]] = {}
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
        if set(entry) != _CA_AGGREGATE_KEYS | {"status", "detail", "records"}:
            return None
        records = entry["records"]
        if type(records) is not list or len(records) != len(block):
            return None
        total_records += len(records)
        records_bytes += len(
            json.dumps(records, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        )
        if records_bytes > _CA_MAX_RECORDS_BYTES:
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
            if len(cells) != 1 + _CA_K_DONORS:
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
                    for value in (cell["total"], cell["mean"]):
                        if (
                            type(value) is not float
                            or not _math.isfinite(value)
                            or _ca_f32(value) != value
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
        candidate_aggregates, candidate_verdicts = _ca_aggregate(records, block)
        aggregates[name] = candidate_aggregates
        verdicts_by_candidate[name] = candidate_verdicts
    if total_records > _CA_MAX_RECORDS:
        return None
    if not aggregates:
        return None
    names = sorted(candidates)
    method = payload.get("method_comparison")
    if type(method) is not dict or set(method) != _CA_METHOD_KEYS:
        return None
    # The stored method block carries the *mapped* primary
    # values; the mirror's mapping is the identity on "selected"
    # and on abstention-shaped decisions, so feeding the stored
    # values back reproduces the block exactly when the producer
    # mapped them honestly. A stored decision of
    # compatible_on_fixture_set / held_out_failed (an unmapped
    # passthrough) recomputes to "selected" and mismatches --
    # rejected, as forged.
    recomputed_method = _ca_method_comparison(
        verdicts_by_candidate,
        names,
        method["primary_decision"],
        method["primary_winner"],
    )
    n_cells = sum(entry["discordance"]["n_paired"] for entry in aggregates.values())
    n_discordant = sum(entry["discordance"]["n_discordant"] for entry in aggregates.values())
    return {
        "aggregates": aggregates,
        "method_comparison": recomputed_method,
        "discordance": {
            "n_cells": n_cells,
            "n_discordant": n_discordant,
            "falsified": (n_discordant == 0) if n_cells else None,
        },
        "n_candidates": len(candidates),
        "n_fixtures": len(block),
    }


def _valid_choiceaudit(payload: Any) -> bool:
    """Structural check for a ``choice_audit`` payload (recompute standard)."""
    if type(payload) is not dict or payload.get("available") is not True:
        return False
    if set(payload) != {
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
    if payload.get("detail") != "":
        return False
    recomputed = _choiceaudit_recompute(payload)
    if recomputed is None:
        return False
    candidates = payload["candidates"]
    for name, expected in recomputed["aggregates"].items():
        entry = candidates[name]
        for key, value in expected.items():
            if not _recomputed_equal(entry.get(key), value):
                return False
    if not _recomputed_equal(payload.get("method_comparison"), recomputed["method_comparison"]):
        return False
    if not _recomputed_equal(payload.get("discordance"), recomputed["discordance"]):
        return False
    if not _recomputed_equal(payload.get("n_candidates"), recomputed["n_candidates"]):
        return False
    return _recomputed_equal(payload.get("n_fixtures"), recomputed["n_fixtures"])


def diff_choiceaudit(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``choice_audit`` payloads; deterministic.

    Answers "did the second opinion change between runs?":
    per-candidate choice accuracy and discordance drift, the
    method-level agreement, candidates added/removed, and
    protocol / donor-map / backend drift (any of which marks
    the comparison not comparable, with the drift still listed
    -- the choice sets are the treatment, so a changed donor
    map is protocol drift). A side without a valid payload is
    reported plainly.
    """
    old_ok = _valid_choiceaudit(old)
    new_ok = _valid_choiceaudit(new)
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
    choiceaudit_protocol_drift = old["protocol"].get("fingerprint") != new["protocol"].get(
        "fingerprint"
    )
    choiceaudit_donor_drift = old["protocol"].get("donor_map_sha256") != new["protocol"].get(
        "donor_map_sha256"
    )
    choiceaudit_backend_drift = old["protocol"].get("backend_fingerprint") != new["protocol"].get(
        "backend_fingerprint"
    )
    choiceaudit_comparable = (
        not choiceaudit_protocol_drift
        and not choiceaudit_donor_drift
        and not choiceaudit_backend_drift
    )
    old_candidates = old["candidates"]
    new_candidates = new["candidates"]
    choiceaudit_candidates_added = sorted(set(new_candidates) - set(old_candidates))
    choiceaudit_candidates_removed = sorted(set(old_candidates) - set(new_candidates))
    choiceaudit_candidate_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_candidates) & set(new_candidates)):
        old_entry = old_candidates[name]
        new_entry = new_candidates[name]
        if old_entry.get("status") != "available" or new_entry.get("status") != "available":
            if old_entry.get("status") != new_entry.get("status"):
                choiceaudit_candidate_drift.append(
                    {
                        "candidate": name,
                        "metric": "status",
                        "old": old_entry.get("status"),
                        "new": new_entry.get("status"),
                    }
                )
            continue
        choiceaudit_metrics: list[tuple[str, Any, Any]] = [
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
            (
                "n_evaluable",
                old_entry["n_evaluable"],
                new_entry["n_evaluable"],
            ),
        ]
        for metric, old_value, new_value in choiceaudit_metrics:
            if old_value != new_value:
                choiceaudit_candidate_drift.append(
                    {
                        "candidate": name,
                        "metric": metric,
                        "old": old_value,
                        "new": new_value,
                    }
                )
    choiceaudit_method_drift: list[dict[str, Any]] = []
    if old["method_comparison"] != new["method_comparison"]:
        choiceaudit_method_drift.append(
            {"old": old["method_comparison"], "new": new["method_comparison"]}
        )
    if old["discordance"] != new["discordance"]:
        choiceaudit_method_drift.append({"old": old["discordance"], "new": new["discordance"]})
    result["candidate_drift"] = choiceaudit_candidate_drift
    result["method_drift"] = choiceaudit_method_drift
    result["candidates_added"] = choiceaudit_candidates_added
    result["candidates_removed"] = choiceaudit_candidates_removed
    result["protocol_drift"] = choiceaudit_protocol_drift
    result["donor_drift"] = choiceaudit_donor_drift
    result["backend_drift"] = choiceaudit_backend_drift
    result["comparable"] = choiceaudit_comparable
    result["changed"] = bool(
        choiceaudit_candidate_drift
        or choiceaudit_method_drift
        or choiceaudit_candidates_added
        or choiceaudit_candidates_removed
        or choiceaudit_protocol_drift
        or choiceaudit_donor_drift
        or choiceaudit_backend_drift
    )
    return result


_PRECISION_CELL_FIELDS = {"answer_digest", "boundary_found", "logprob", "output_digest", "verdict"}
_PRECISION_RECORD_FIELDS = {
    "bf16",
    "candidate",
    "diverged",
    "expected_digest",
    "first_divergence",
    "fixture",
    "fp32",
}
_PRECISION_PROTOCOL_FIELDS = {
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
}
_PRECISION_WITHHELD = {
    None,
    "bf16_unavailable",
    "mismatched_sets",
    "no_paired_fixtures",
    "primary_not_winner_bearing",
}


def _precision_digest_ok(value: Any) -> bool:
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _precision_cell_ok(cell: Any, expected_digest: str) -> bool:
    """One arm cell's shape and verdict binding (placement-cell standard)."""
    if type(cell) is not dict or set(cell) != _PRECISION_CELL_FIELDS:
        return False
    if not _precision_digest_ok(cell["answer_digest"]):
        return False
    if not _precision_digest_ok(cell["output_digest"]):
        return False
    if type(cell["boundary_found"]) is not bool or type(cell["verdict"]) is not bool:
        return False
    logprob = cell["logprob"]
    if logprob is not None and (
        type(logprob) not in (int, float)
        or isinstance(logprob, bool)
        or (type(logprob) is float and logprob != logprob)
        or (type(logprob) is float and logprob in (float("inf"), float("-inf")))
    ):
        return False
    return cell["verdict"] == (cell["answer_digest"] == expected_digest and cell["boundary_found"])


def _precision_selection_over(
    records: list[dict[str, Any]],
    candidates: list[str],
    arm: str,
    fixtures: frozenset[str],
) -> dict[str, Any] | None:
    """Mirror of precision.selection_over (verdicts read from the cells)."""
    completed_sets: list[frozenset[str]] = []
    all_pass: dict[str, bool] = {}
    for name in candidates:
        cells: dict[str, dict[str, Any]] = {}
        for record in records:
            if record["candidate"] != name:
                continue
            cell = record[arm]
            if type(cell) is dict:
                cells[str(record["fixture"])] = cell
        completed = frozenset(fid for fid in cells if fid in fixtures)
        completed_sets.append(completed)
        if not completed:
            return None
        all_pass[name] = all(cells[fid]["verdict"] for fid in completed)
    if len(set(completed_sets)) > 1:
        return None
    passers = [name for name, ok in all_pass.items() if ok]
    if len(passers) == 1:
        return {"decision": "selected", "winner": passers[0]}
    if len(passers) > 1:
        return {"decision": "ambiguous", "winner": None}
    return {"decision": "no_candidate", "winner": None}


def _precision_recompute(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Recompute every derived precision field from the ground records.

    Independent mirror of precision.compute_derived. Returns None on
    any structural violation. The per-arm all-pass verdicts are
    stage-attested (eligibility is a property of the arm's evaluation
    report, not of the cells) and are consistency-checked here:
    a True/False verdict requires the arm's cells to cover the whole
    fixture universe visible in the records, with the matching
    pass/fail pattern; None requires an incomplete cell set. The
    fixture universe itself is attested by the protocol's
    fixture-set digest -- fixtures completed by no candidate in
    either arm appear in no record and cannot be re-derived (the
    same attestation residue _valid_primary_pair documents).
    """
    protocol = payload.get("protocol")
    if type(protocol) is not dict or set(protocol) != _PRECISION_PROTOCOL_FIELDS:
        return None
    if protocol["version"] != 1 or type(protocol["version"]) is not int:
        return None
    if type(protocol["seed"]) is not int or protocol["seed"] < 0:
        return None
    if type(protocol["max_new_tokens"]) is not int or not 1 <= protocol["max_new_tokens"] <= 4096:
        return None
    if protocol["max_records"] != 2400 or protocol["max_records_bytes"] != 12 * 1024 * 1024:
        return None
    if not _precision_digest_ok(protocol["fixture_set_sha256"]):
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
        if (
            type(ids) is not list
            or not ids
            or any(type(t) is not int or isinstance(t, bool) or t < 0 for t in ids)
        ):
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
    body = {k: v for k, v in protocol.items() if k != "fingerprint"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    if protocol["fingerprint"] != hashlib.sha256(canonical.encode("utf-8")).hexdigest():
        return None
    if not _valid_primary_pair(payload):
        return None
    records = payload.get("records")
    if type(records) is not list or len(records) > protocol["max_records"]:
        return None
    records_blob = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    if len(records_blob.encode("utf-8")) > protocol["max_records_bytes"]:
        return None
    seen: set[tuple[str, str]] = set()
    universe: set[str] = set()
    record_candidates: set[str] = set()
    for record in records:
        if type(record) is not dict or set(record) != _PRECISION_RECORD_FIELDS:
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
        if not _precision_digest_ok(record["expected_digest"]):
            return None
        for arm in ("fp32", "bf16"):
            cell = record[arm]
            if cell is not None and not _precision_cell_ok(cell, str(record["expected_digest"])):
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
    unavailable = payload.get("unavailable_candidates")
    if (
        type(unavailable) is not list
        or unavailable != sorted(unavailable)
        or any(type(n) is not str for n in unavailable)
        or set(unavailable) & record_candidates
        or set(unavailable) | record_candidates != set(candidates)
    ):
        return None
    blocks_in = payload.get("candidates")
    if type(blocks_in) is not dict or set(blocks_in) != set(candidates):
        return None
    # Per-candidate blocks, recomputed; arm_passed consistency first.
    ordered = sorted(records, key=lambda r: (str(r["candidate"]), str(r["fixture"])))
    blocks: dict[str, Any] = {}
    flips_all: list[dict[str, Any]] = []
    total_paired = 0
    total_cells = {"pass_pass": 0, "pass_fail": 0, "fail_pass": 0, "fail_fail": 0}  # nosec B105 -- 'pass' keys are verdict counts in the frozen payload schema, not credentials
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
        cells = {"pass_pass": 0, "pass_fail": 0, "fail_pass": 0, "fail_fail": 0}  # nosec B105 -- 'pass' keys are verdict counts in the frozen payload schema, not credentials
        flips: list[dict[str, Any]] = []
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
    unpaired: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda r: (str(r["candidate"]), str(r["fixture"]))):
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
    falsified: bool | None = None if total_paired == 0 else len(flips_all) == 0
    shared: frozenset[str] | None = None
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
    primary_decision = payload.get("primary_decision")
    selection: dict[str, Any]
    if primary_decision not in ("compatible_on_fixture_set", "held_out_failed"):
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
        fp32_sel = _precision_selection_over(records, list(candidates), "fp32", shared)
        bf16_sel = _precision_selection_over(records, list(candidates), "bf16", shared)
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


def _valid_precision(payload: Any) -> bool:
    """Full-recompute check for a ``precision`` payload.

    Every derived value -- flip matrix, unpaired list, divergence
    aggregates, logprob summaries, both paired-population
    selections, survival, falsification -- is recomputed from the
    digest-bound ground records and compared type-strictly. The
    protocol fingerprint is recomputed, the two arm fingerprints
    must differ, and the primary pair is bound by
    _valid_primary_pair.
    """
    if type(payload) is not dict or payload.get("available") is not True:
        return False
    if set(payload) != {
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
    if payload.get("detail") is not None:
        return False
    recomputed = _precision_recompute(payload)
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
        if not _recomputed_equal(payload.get(key), recomputed[key]):
            return False
    return True


def diff_precision(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``precision`` payloads; deterministic.

    Answers "did the load-precision story change between runs?":
    per-candidate flip/divergence/logprob drift, the survival and
    falsification readouts, candidates added/removed, and protocol
    or backend-fingerprint drift (either arm's fingerprint -- a
    drifted arm makes the comparison not comparable, with the
    drift still listed, the placement pattern). A side without a
    valid payload is reported plainly.
    """
    old_ok = _valid_precision(old)
    new_ok = _valid_precision(new)
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
    precision_protocol_drift = old["protocol"].get("fingerprint") != new["protocol"].get(
        "fingerprint"
    )
    precision_backend_drift = old["protocol"].get("fp32_fingerprint") != new["protocol"].get(
        "fp32_fingerprint"
    ) or old["protocol"].get("bf16_fingerprint") != new["protocol"].get("bf16_fingerprint")
    precision_comparable = not precision_protocol_drift and not precision_backend_drift
    old_candidates = old["candidates"]
    new_candidates = new["candidates"]
    precision_candidates_added = sorted(set(new_candidates) - set(old_candidates))
    precision_candidates_removed = sorted(set(old_candidates) - set(new_candidates))
    precision_candidate_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_candidates) & set(new_candidates)):
        old_entry = old_candidates[name]
        new_entry = new_candidates[name]
        precision_metrics: list[tuple[str, Any, Any]] = [
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
        for metric, old_value, new_value in precision_metrics:
            if old_value != new_value:
                precision_candidate_drift.append(
                    {"candidate": name, "metric": metric, "old": old_value, "new": new_value}
                )
    precision_selection_drift: list[dict[str, Any]] = []
    if old["selection"] != new["selection"]:
        precision_selection_drift.append({"old": old["selection"], "new": new["selection"]})
    if old["precision_effect_falsified"] != new["precision_effect_falsified"]:
        precision_selection_drift.append(
            {
                "metric": "precision_effect_falsified",
                "old": old["precision_effect_falsified"],
                "new": new["precision_effect_falsified"],
            }
        )
    result["candidate_drift"] = precision_candidate_drift
    result["selection_drift"] = precision_selection_drift
    result["candidates_added"] = precision_candidates_added
    result["candidates_removed"] = precision_candidates_removed
    result["protocol_drift"] = precision_protocol_drift
    result["backend_drift"] = precision_backend_drift
    result["comparable"] = precision_comparable
    # NOTE: precision-specific names keep this line textually distinct
    # from every other section's changed anchor (mutation-gate
    # uniqueness).
    result["changed"] = bool(
        precision_candidate_drift
        or precision_selection_drift
        or precision_candidates_added
        or precision_candidates_removed
        or precision_protocol_drift
        or precision_backend_drift
    )
    return result


_CONFIDENCE_CELL_FIELDS = {
    "answer_digest",
    "boundary_found",
    "code",
    "confidence",
    "final_line",
    "final_line_length",
    "input_tokens",
    "span_digest",
    "verdict",
}
_CONFIDENCE_RECORD_FIELDS = {
    "candidate",
    "elicited",
    "expected_digest",
    "fixture",
    "native_verdict",
    "screen",
}
_CONFIDENCE_CONTROL_RECORD_FIELDS = {
    "candidate",
    "case",
    "control",
    "expected_digest",
    "knowable",
}
_CONFIDENCE_PROTOCOL_FIELDS = {
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
}
_CONFIDENCE_CASE_FIELDS = {
    "control_payload_digest",
    "expected_digest",
    "id",
    "key_digest",
    "knowable_payload_digest",
}
_CONFIDENCE_CODES = {
    "ok",
    "missing_line",
    "malformed_line",
    "multiple_lines",
    "confidence_not_final",
}
_CONFIDENCE_SCREEN_CLASSES = {
    "expected_blank",
    "expected_confidence_shaped",
    "vacuous_last_user",
    "no_user_turn",
}
_CONFIDENCE_GRAMMAR_RE = re.compile(r"^Confidence: (0|[1-9][0-9]?|100)$")
_CONFIDENCE_BIN_EDGES = [[0, 24], [25, 49], [50, 74], [75, 100]]


def _confidence_digest_ok(value: Any) -> bool:
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _confidence_cell_ok(cell: Any, expected_digest: str) -> bool:
    """One elicited cell's shape, carried-line binding, and verdict binding.

    The carried line is what moves the confidence scalar from
    stage-attested to validator-derived: an ok cell's line must be
    a grammar line whose value *is* the recorded scalar; a
    malformed cell's 32-character prefix plus full length must
    re-verify the class (prefix-shaped, and -- when the whole line
    is carried -- not itself a grammar line). Span/answer digest
    relations are code-dependent and unconditional: equal for
    every non-ok code, different for every ok cell, with no
    single-line exception (sha256("") != sha256("Confidence: 50")).
    """
    if type(cell) is not dict or set(cell) != _CONFIDENCE_CELL_FIELDS:
        return False
    if not _confidence_digest_ok(cell["answer_digest"]):
        return False
    if not _confidence_digest_ok(cell["span_digest"]):
        return False
    if type(cell["boundary_found"]) is not bool or type(cell["verdict"]) is not bool:
        return False
    if cell["verdict"] != (cell["span_digest"] == expected_digest and cell["boundary_found"]):
        return False
    if type(cell["input_tokens"]) is not int or cell["input_tokens"] < 1:
        return False
    code = cell["code"]
    if code not in _CONFIDENCE_CODES:
        return False
    confidence = cell["confidence"]
    line = cell["final_line"]
    line_length = cell["final_line_length"]
    if code == "ok":
        if cell["span_digest"] == cell["answer_digest"]:
            return False
        if type(confidence) is not int or not 0 <= confidence <= 100:
            return False
        if type(line) is not str or _CONFIDENCE_GRAMMAR_RE.match(line) is None:
            return False
        match = _CONFIDENCE_GRAMMAR_RE.match(line)
        if match is None or int(match.group(1)) != confidence:
            return False
        return bool(line_length == len(line))
    if cell["span_digest"] != cell["answer_digest"]:
        return False
    if code == "malformed_line":
        if confidence is not None:
            return False
        if type(line) is not str or not 1 <= len(line) <= 32:
            return False
        if type(line_length) is not int or line_length < len(line) or line_length < 11:
            return False
        if line_length > len(line) and len(line) != 32:
            return False
        if not (line.startswith("Confidence:") or line.lower().startswith("confidence:")):
            return False
        return not (line_length == len(line) and _CONFIDENCE_GRAMMAR_RE.match(line) is not None)
    return confidence is None and line is None and line_length is None


def _confidence_case_mirror(seed: int) -> list[dict[str, Any]]:
    """Regenerate the generated case set's protocol data from the seed.

    Independent mirror of confidence.case_protocol: the same seed
    derivation, distinctness repair, message template, and donor
    pairing, re-implemented here so a payload whose control cases
    do not regenerate from its own protocol seed is invalid.
    """

    def _derive(tag: str, index: int, attempt: int = 0) -> str:
        material = f"{seed}\0confidence-{tag}\0{index}\0{attempt}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def _text_digest(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()

    pairs: list[tuple[str, str]] = []
    used_keys: set[str] = set()
    used_values: set[str] = set()
    for index in range(24):
        attempt = 0
        key = "key-" + _derive("key", index)[:8]
        while key in used_keys:
            attempt += 1
            key = "key-" + _derive("key", index, attempt)[:8]
        used_keys.add(key)
        attempt = 0
        value = _derive("value", index)[:6]
        while value in used_values:
            attempt += 1
            value = _derive("value", index, attempt)[:6]
        used_values.add(value)
        pairs.append((key, value))
    out: list[dict[str, Any]] = []
    for index, (key, value) in enumerate(pairs):
        donor_key, donor_value = pairs[(index + 1) % 24]
        knowable_text = (
            f"Reference record: the code for {key} is {value}."
            f"\n\nWhat is the code for {key}? Answer with the code only."
        )
        control_text = (
            f"Reference record: the code for {donor_key} is {donor_value}."
            f"\n\nWhat is the code for {key}? Answer with the code only."
        )
        out.append(
            {
                "id": f"confidence-case-{index:02d}",
                "key_digest": _text_digest(key),
                "expected_digest": _text_digest(value),
                "knowable_payload_digest": _text_digest(knowable_text),
                "control_payload_digest": _text_digest(control_text),
            }
        )
    return out


def _confidence_block_mirror(cells: list[dict[str, Any]], n_population: int) -> dict[str, Any]:
    """Independent mirror of confidence._population_block."""
    compliant = [cell for cell in cells if cell["code"] == "ok"]
    correct = [cell for cell in compliant if cell["verdict"]]
    incorrect = [cell for cell in compliant if not cell["verdict"]]

    def _mean(values: list[int]) -> float | None:
        return (sum(values) / len(values)) if values else None

    mean_correct = _mean([int(cell["confidence"]) for cell in correct])
    mean_incorrect = _mean([int(cell["confidence"]) for cell in incorrect])
    if not compliant:
        gap: float | None = None
        gap_detail = "no_compliant_cells"
    elif mean_correct is None:
        gap = None
        gap_detail = "no_correct_compliant"
    elif mean_incorrect is None:
        gap = None
        gap_detail = "no_incorrect_compliant"
    else:
        gap = mean_correct - mean_incorrect
        gap_detail = ""
    bins: list[dict[str, Any]] = []
    for lo, hi in ((0, 24), (25, 49), (50, 74), (75, 100)):
        members = [cell for cell in compliant if lo <= int(cell["confidence"]) <= hi]
        n_correct = sum(1 for cell in members if cell["verdict"])
        bins.append(
            {
                "lo": lo,
                "hi": hi,
                "n": len(members),
                "n_correct": n_correct,
                "accuracy": (n_correct / len(members)) if members else None,
            }
        )
    return {
        "n_population": n_population,
        "n_completed": len(cells),
        "n_compliant": len(compliant),
        "compliance_rate": (len(compliant) / len(cells)) if cells else None,
        "mean_confidence": _mean([int(cell["confidence"]) for cell in compliant]),
        "mean_correct": mean_correct,
        "mean_incorrect": mean_incorrect,
        "gap": gap,
        "gap_detail": gap_detail,
        "joint": {
            "compliant_correct": len(correct),
            "compliant_incorrect": len(incorrect),
            "noncompliant": len(cells) - len(compliant),
        },
        "bins": bins,
    }


def _confidence_recompute(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Recompute every derived confidence field from the ground records.

    Independent mirror of confidence.compute_derived. Returns None
    on any structural violation. The protocol's generated case set
    is regenerated from the seed and must match exactly (ids, key
    digests, expected digests, both payload digests), so the
    control arm's construction is bound, not attested. Screens are
    fixture properties the validator cannot recompute from digests
    (the fixture texts are not in the payload); they are checked
    for consistency -- protocol screen map against every record,
    screened fixtures carrying no cell -- the attestation residue
    the design doc names. Per-cell parse/finality/multiplicity is
    stage-attested the same way; the confidence *scalar* is not:
    it is derived here from the carried line.
    """
    protocol = payload.get("protocol")
    if type(protocol) is not dict or set(protocol) != _CONFIDENCE_PROTOCOL_FIELDS:
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
    if protocol["bin_edges"] != _CONFIDENCE_BIN_EDGES:
        return None
    if not _confidence_digest_ok(protocol["suffix_sha256"]):
        return None
    if not _confidence_digest_ok(protocol["fixture_set_sha256"]):
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
        if (
            type(ids) is not list
            or not ids
            or any(type(t) is not int or isinstance(t, bool) or t < 0 for t in ids)
        ):
            return None
    if type(protocol["backend_fingerprint"]) is not str or not protocol["backend_fingerprint"]:
        return None
    screens = protocol["screens"]
    if type(screens) is not dict:
        return None
    for fixture_id, screen in screens.items():
        if type(fixture_id) is not str or screen not in _CONFIDENCE_SCREEN_CLASSES:
            return None
    cases_in = protocol["control_cases"]
    if type(cases_in) is not list or len(cases_in) != 24:
        return None
    for entry in cases_in:
        if type(entry) is not dict or set(entry) != _CONFIDENCE_CASE_FIELDS:
            return None
        for key in (
            "key_digest",
            "expected_digest",
            "knowable_payload_digest",
            "control_payload_digest",
        ):
            if not _confidence_digest_ok(entry[key]):
                return None
    if not _recomputed_equal(cases_in, _confidence_case_mirror(int(protocol["seed"]))):
        return None
    body = {k: v for k, v in protocol.items() if k != "fingerprint"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    if protocol["fingerprint"] != hashlib.sha256(canonical.encode("utf-8")).hexdigest():
        return None
    if not _valid_primary_pair(payload):
        return None
    records = payload.get("records")
    control_records = payload.get("control_records")
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
    case_expected = {entry["id"]: entry["expected_digest"] for entry in cases_in}
    seen: set[tuple[str, str]] = set()
    universe: set[str] = set()
    fixtures_by_candidate: dict[str, set[str]] = {name: set() for name in candidates}
    zero_cell: dict[str, bool] = dict.fromkeys(candidates, True)
    for record in records:
        if type(record) is not dict or set(record) != _CONFIDENCE_RECORD_FIELDS:
            return None
        name = record["candidate"]
        fixture = record["fixture"]
        if type(name) is not str or name not in candidates or type(fixture) is not str:
            return None
        if (name, fixture) in seen:
            return None
        seen.add((name, fixture))
        universe.add(fixture)
        fixtures_by_candidate[name].add(fixture)
        if record["screen"] != screens.get(fixture, ""):
            return None
        if not _confidence_digest_ok(record["expected_digest"]):
            return None
        if record["native_verdict"] not in (True, False, None):
            return None
        cell = record["elicited"]
        if record["screen"] != "" and cell is not None:
            return None
        if cell is not None:
            if not _confidence_cell_ok(cell, str(record["expected_digest"])):
                return None
            zero_cell[name] = False
    for name in candidates:
        if fixtures_by_candidate[name] != universe:
            return None
    seen_control: set[tuple[str, str]] = set()
    cases_by_candidate: dict[str, set[str]] = {name: set() for name in candidates}
    for record in control_records:
        if type(record) is not dict or set(record) != _CONFIDENCE_CONTROL_RECORD_FIELDS:
            return None
        name = record["candidate"]
        case_id = record["case"]
        if type(name) is not str or name not in candidates or case_id not in case_expected:
            return None
        if (name, case_id) in seen_control:
            return None
        seen_control.add((name, case_id))
        cases_by_candidate[name].add(case_id)
        if record["expected_digest"] != case_expected[case_id]:
            return None
        for arm in ("knowable", "control"):
            cell = record[arm]
            if cell is not None:
                if not _confidence_cell_ok(cell, str(record["expected_digest"])):
                    return None
                zero_cell[name] = False
    for name in candidates:
        if cases_by_candidate[name] != set(case_expected):
            return None
    unavailable = payload.get("unavailable_candidates")
    expected_unavailable = sorted(name for name in candidates if zero_cell[name])
    if unavailable != expected_unavailable:
        return None
    n_population = sum(1 for fixture in universe if screens.get(fixture, "") == "")
    blocks_in = payload.get("candidates")
    if type(blocks_in) is not dict or set(blocks_in) != set(candidates):
        return None
    ordered = sorted(records, key=lambda r: (str(r["candidate"]), str(r["fixture"])))
    ordered_control = sorted(control_records, key=lambda r: (str(r["candidate"]), str(r["case"])))
    blocks: dict[str, Any] = {}
    for name in candidates:
        own = [r for r in ordered if r["candidate"] == name]
        elicited = [r["elicited"] for r in own if type(r["elicited"]) is dict]
        operator = _confidence_block_mirror(elicited, n_population)
        flips = {"n_compared": 0, "pass_to_fail": 0, "fail_to_pass": 0}  # nosec B105 -- 'pass' keys are verdict-flip counts in the frozen payload schema, not credentials
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
        own_control = [r for r in ordered_control if r["candidate"] == name]
        knowable_cells = [r["knowable"] for r in own_control if type(r["knowable"]) is dict]
        control_cells = [r["control"] for r in own_control if type(r["control"]) is dict]
        knowable = _confidence_block_mirror(knowable_cells, 24)
        control_block = _confidence_block_mirror(control_cells, 24)
        paired = [
            r
            for r in own_control
            if type(r["knowable"]) is dict
            and type(r["control"]) is dict
            and r["knowable"]["code"] == "ok"
            and r["control"]["code"] == "ok"
        ]
        if len(paired) >= 6:
            knowable_values = [int(r["knowable"]["confidence"]) for r in paired]
            control_values = [int(r["control"]["confidence"]) for r in paired]
            knowable_mean = sum(knowable_values) / len(knowable_values)
            control_mean = sum(control_values) / len(control_values)
            knowability: dict[str, Any] = {
                "n_paired_compliant": len(paired),
                "knowable_mean": knowable_mean,
                "control_mean": control_mean,
                "difference": knowable_mean - control_mean,
                "falsified": control_mean >= knowable_mean,
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
        "totals": {
            "n_records": len(records),
            "n_control_records": len(control_records),
        },
    }


def _valid_confidence(payload: Any) -> bool:
    """Full-recompute check for a ``confidence`` payload.

    Every derived value -- compliance, joint tables, means, the
    gap, the bins, flips, the knowability readout -- is recomputed
    from the ground records and compared type-strictly. The
    confidence scalar itself is derived from each ok cell's
    carried line, not trusted; the generated case set is
    regenerated from the protocol seed; the primary pair is bound
    by _valid_primary_pair.
    """
    if type(payload) is not dict or payload.get("available") is not True:
        return False
    if set(payload) != {
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
    if payload.get("detail") is not None:
        return False
    recomputed = _confidence_recompute(payload)
    if recomputed is None:
        return False
    for key in ("candidates", "totals"):
        if not _recomputed_equal(payload.get(key), recomputed[key]):
            return False
    return True


def diff_confidence(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``confidence`` payloads; deterministic.

    Answers "did the stated-confidence story change between
    runs?": per-candidate compliance/gap/flip drift, knowable and
    control mean drift, knowability falsification drift, candidates
    added/removed, and protocol or backend-fingerprint drift (a
    drifted protocol or backend makes the comparison not
    comparable, with the drift still listed, the placement
    pattern). A side without a valid payload is reported plainly.
    """
    old_ok = _valid_confidence(old)
    new_ok = _valid_confidence(new)
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
    confidence_protocol_drift = old["protocol"].get("fingerprint") != new["protocol"].get(
        "fingerprint"
    )
    confidence_backend_drift = old["protocol"].get("backend_fingerprint") != new["protocol"].get(
        "backend_fingerprint"
    )
    confidence_comparable = not confidence_protocol_drift and not confidence_backend_drift
    old_candidates = old["candidates"]
    new_candidates = new["candidates"]
    confidence_candidates_added = sorted(set(new_candidates) - set(old_candidates))
    confidence_candidates_removed = sorted(set(old_candidates) - set(new_candidates))
    confidence_candidate_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_candidates) & set(new_candidates)):
        old_entry = old_candidates[name]
        new_entry = new_candidates[name]
        confidence_metrics: list[tuple[str, Any, Any]] = [
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
        for metric, old_value, new_value in confidence_metrics:
            if old_value != new_value:
                confidence_candidate_drift.append(
                    {"candidate": name, "metric": metric, "old": old_value, "new": new_value}
                )
    result["candidate_drift"] = confidence_candidate_drift
    result["candidates_added"] = confidence_candidates_added
    result["candidates_removed"] = confidence_candidates_removed
    result["protocol_drift"] = confidence_protocol_drift
    result["backend_drift"] = confidence_backend_drift
    result["comparable"] = confidence_comparable
    # NOTE: confidence-specific names keep this line textually distinct
    # from every other section's changed anchor (mutation-gate
    # uniqueness).
    result["changed"] = bool(
        confidence_candidate_drift
        or confidence_candidates_added
        or confidence_candidates_removed
        or confidence_protocol_drift
        or confidence_backend_drift
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

    Output contract (cycle 217, CodeQL logging adjudication): both sinks
    print the DERIVED diff dict only, never the loaded input payloads, so
    no input-only field can reach either sink. The ``confidentiality``
    section's name refers to the behavioral labeled-block exposure probe
    (cycle 169), not to secret material: the section carries only an
    availability enum, booleans, numeric gaps, and candidate labels from
    structurally validated payloads -- no prompt, completion, or
    credential material exists in it to leak.
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
