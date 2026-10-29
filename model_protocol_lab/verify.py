"""Independent results-integrity replay (``mpl verify``).

Re-derives an experiment's recorded decision, winner, and post-hoc
diagnostics from the stored ``results.json`` artifact alone, through code
paths independent of the ones that produced them (N-version cross-check):

- the selection rule is re-implemented as :func:`oracle_select_winner`
  instead of re-calling ``experiments._select_winner``;
- the token-budget frontier and the held-out budget floor are recomputed
  from the recorded observations and compared for exact equality with the
  stored sections;
- the search/held-out fixture split is checked for overlap, and every
  candidate is checked to have run the same search fixtures.

What verify can and cannot prove is stated on every check. Cancelled runs
do not record the cancel signal, so a stored ``"cancelled"`` decision is
reported ``"unavailable"`` rather than re-derived. Runs written without
``--include-evidence`` strip the raw observations, so
observation-dependent checks report ``"unavailable"`` instead of
pretending. A check is ``"failed"`` only when the artifact positively
contradicts what the pipeline would have written.

Plain verification loads no model and uses no network, and is
deterministic. The one exception is the opt-in ``--determinism``
audit: it loads the operator's local checkpoint through the same
backend ``run`` uses and re-executes the recorded search observations
(see ``determinism.py`` and ``docs/DETERMINISM_AUDIT_DESIGN.md``).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# Mirrors diffreports._CONTROL_CHARS (kept local so this module does not
# import the CLI or the diff engine): strips ASCII controls, DEL, C1
# controls, lone surrogates, and Unicode bidi controls. A hostile
# results.json must not inject terminal escape sequences or Trojan Source
# reordering into the stderr summary. BOM, zero-width space and soft
# hyphen are deliberately preserved: they are diagnostically valuable and
# cannot reorder displayed text.
_CONTROL_CHARS = re.compile(
    r"[\x00-\x1f\x7f-\x9f\ud800-\udfff\u200e\u200f\u2028\u2029\u202a-\u202e\u2066-\u2069]"
)


def _display_id(value: str) -> str:
    """Operator-visible names/IDs with control characters stripped
    and the R4-7 200-character preview cap applied: a hostile name
    must not be able to flood a report detail or a terminal
    (oversight security S-9b — one 60,000-character entry name
    produced a 60,013-character seal detail before the cap).
    """
    cleaned = _CONTROL_CHARS.sub("", value)
    if len(cleaned) > 200:
        return cleaned[:200] + "..."
    return cleaned


_PASSED = "passed"
_FAILED = "failed"
_UNAVAILABLE = "unavailable"

#: Decisions a saved artifact can legitimately carry. "not_started" is the
#: in-memory initial value and must never appear in a saved report.
_KNOWN_DECISIONS = frozenset(
    {
        "compatible_on_fixture_set",
        "held_out_failed",
        "ambiguous",
        "no_candidate",
        "cancelled",
        "failed",
    }
)

#: Decisions that imply a selected search candidate was carried forward.
_SELECTED_DECISIONS = frozenset({"compatible_on_fixture_set", "held_out_failed"})

#: RunStatus values a per-candidate summary can carry on the abort path.
_ABORT_STATUSES = frozenset({"cancelled", "failed"})


def oracle_select_winner(winner_names: list[str]) -> tuple[str, str | None]:
    """Independent reimplementation of the search-phase selection decision.

    N-version oracle for ``experiments._select_winner``: exactly one
    winner selects it; zero winners abstain as ``"no_candidate"``; more
    than one abstain as ``"ambiguous"``. Cancellation is deliberately not
    part of this oracle -- the cancel signal is not recorded in
    ``results.json``, so verify treats a stored ``"cancelled"`` decision
    as unverifiable rather than re-deriving it.

    A Hypothesis cross-test
    (``tests/test_cycle131_verify_audit.py``) pins agreement with
    ``experiments._select_winner`` on generated winner lists.
    """
    if len(winner_names) == 1:
        return "selected", winner_names[0]
    if winner_names:
        return "ambiguous", None
    return "no_candidate", None


def _record(checks: list[dict[str, str]], check_id: str, status: str, detail: str = "") -> None:
    """Record."""
    checks.append({"id": check_id, "status": status, "detail": detail})


def _json_equal(first: Any, second: Any) -> bool:
    """Exact structural equality via canonical JSON encoding."""
    return json.dumps(first, sort_keys=True, ensure_ascii=True) == json.dumps(
        second, sort_keys=True, ensure_ascii=True
    )


def _valid_token_ids(value: Any) -> bool:
    """Return whether token ids is valid."""
    return type(value) is list and bool(value) and all(type(v) is int and v >= 0 for v in value)


def _entry_name(entry: Any) -> str | None:
    """Return the entry name."""
    if type(entry) is not dict:
        return None
    fmt = entry.get("format")
    name = fmt.get("name") if type(fmt) is dict else None
    return name if type(name) is str and name else None


def _entry_boundary(entry: dict[str, Any], top_level: list[int]) -> list[int] | None:
    """The boundary a search entry was measured with (per-candidate first).

    Mirrors ``experiments.budget_frontier``'s rule: an entry's own
    ``boundary_token_ids`` wins; older artifacts without the field fall
    back to the run-level boundary.
    """
    entry_boundary = entry.get("boundary_token_ids")
    if entry_boundary is None:
        return top_level
    return entry_boundary if _valid_token_ids(entry_boundary) else None


def _check_schema(data: Any, checks: list[dict[str, str]]) -> dict[str, Any] | None:
    """Minimal artifact contract. Returns the data on success, else None."""
    if type(data) is not dict:
        _record(checks, "schema", _FAILED, "results.json is not a JSON object")
        return None
    problems: list[str] = []
    # Type-guard before the frozenset membership test: an unhashable
    # decision (dict/list from parsed JSON) raised TypeError here
    # instead of recording a failed schema check (cycle 207,
    # wave-10 A-F1). Every sibling field is already type-guarded
    # first; decision was the only one relying on hashability.
    if type(data.get("decision")) is not str or data.get("decision") not in _KNOWN_DECISIONS:
        problems.append("decision not in known vocabulary")
    if type(data.get("search")) is not list:
        problems.append("search is not a list")
    selected = data.get("selected_candidate")
    if selected is not None and type(selected) is not str:
        problems.append("selected_candidate is not a string or null")
    max_new_tokens = data.get("max_new_tokens")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 4096:
        problems.append("max_new_tokens is not an integer in 1..4096")
    if not _valid_token_ids(data.get("boundary_token_ids")):
        problems.append("boundary_token_ids is not a nonempty ID list")
    if type(data.get("failure_taxonomy")) is not dict:
        problems.append("failure_taxonomy missing")
    if problems:
        _record(checks, "schema", _FAILED, "; ".join(problems))
        return None
    _record(checks, "schema", _PASSED)
    return data


def _check_candidates_unique(data: dict[str, Any], checks: list[dict[str, str]]) -> None:
    """Check candidates unique."""
    names: list[str] = []
    for entry in data["search"]:
        name = _entry_name(entry)
        if name is None:
            _record(
                checks,
                "candidates_unique",
                _FAILED,
                "a search entry has no usable format.name",
            )
            return
        names.append(name)
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        _record(
            checks,
            "candidates_unique",
            _FAILED,
            "duplicate candidate names: " + ", ".join(_display_id(n) for n in duplicates),
        )
        return
    selected = data.get("selected_candidate")
    if selected is not None and selected not in names:
        _record(
            checks,
            "candidates_unique",
            _FAILED,
            "selected_candidate not among search candidates: " + _display_id(selected),
        )
        return
    _record(checks, "candidates_unique", _PASSED)


def _check_summaries_consistent(data: dict[str, Any], checks: list[dict[str, str]]) -> None:
    """Check summaries consistent."""
    for entry in data["search"]:
        summary = entry.get("summary") if type(entry) is dict else None
        if type(summary) is not dict:
            _record(
                checks,
                "summaries_consistent",
                _FAILED,
                "a search entry has no summary dict",
            )
            return
        planned = summary.get("planned")
        passes = summary.get("exact_and_boundary_passes")
        if type(planned) is not int or planned < 0:
            _record(
                checks,
                "summaries_consistent",
                _FAILED,
                "summary.planned is not a non-negative integer",
            )
            return
        if type(passes) is not int or not 0 <= passes <= planned:
            _record(
                checks,
                "summaries_consistent",
                _FAILED,
                "summary passes not in 0..planned",
            )
            return
    _record(checks, "summaries_consistent", _PASSED)


# skipcq: PY-R1000 - Gate-pinned frozen function; splitting breaks its verified contract.
def _replay_selection(data: dict[str, Any]) -> tuple[str, str | None, str]:
    """Re-derive (decision, selected_candidate, note) from stored summaries.

    The note explains the derivation path for the check detail; it is
    "cancel_unverifiable" when the stored decision is "cancelled" with no
    aborted search summary to explain it -- the cancel signal is not
    recorded in the artifact.
    """
    # Search-phase abort: the pipeline stops at the first candidate whose
    # report is not aggregate-eligible and records its status verbatim.
    # Non-dict entries (or entries without a dict summary) are fail-closed
    # as "malformed": verify never crashes on hostile input, it reports.
    for entry in data["search"]:
        if type(entry) is not dict or type(entry.get("summary")) is not dict:
            return "malformed", None, "search entry malformed"
        summary = entry["summary"]
        if not summary.get("aggregate_eligible"):
            status = summary.get("status")
            expected = status if type(status) is str else "failed"
            return expected, None, "search aborted at " + str(_entry_name(entry))
    winner_names = [
        name
        for entry in data["search"]
        if (name := _entry_name(entry)) is not None
        and entry["summary"].get("aggregate_eligible")
        and entry["summary"].get("exact_and_boundary_passes") == entry["summary"].get("planned")
    ]
    if data.get("decision") == "cancelled":
        # All summaries are aggregate-eligible, so the cancel happened
        # after the search completed; the signal itself is not recorded.
        return "cancelled", None, "cancel_unverifiable"
    search_decision, winner = oracle_select_winner(winner_names)
    if search_decision != "selected" or winner is None:
        return search_decision, None, "search decision replayed"
    held = data.get("held_out")
    if type(held) is not dict:
        return "incomplete", winner, "held_out missing for selected candidate"
    held_summary = held.get("summary")
    if type(held_summary) is not dict:
        return "incomplete", winner, "held_out summary missing"
    if not held_summary.get("aggregate_eligible"):
        status = held_summary.get("status")
        expected = status if type(status) is str else "failed"
        return expected, winner, "held-out aborted"
    passes = held_summary.get("exact_and_boundary_passes")
    planned = held_summary.get("planned")
    if type(passes) is int and type(planned) is int and planned >= 0 and passes == planned:
        return "compatible_on_fixture_set", winner, "held-out passed"
    return "held_out_failed", winner, "held-out verdict replayed"


def _check_selection_decision(data: dict[str, Any], checks: list[dict[str, str]]) -> None:
    """Check selection decision."""
    expected_decision, expected_winner, note = _replay_selection(data)
    if note == "cancel_unverifiable":
        _record(
            checks,
            "selection_decision",
            _UNAVAILABLE,
            "cancel signal not recorded in results.json",
        )
        return
    stored_decision = data.get("decision")
    stored_winner = data.get("selected_candidate")
    if stored_decision == expected_decision and stored_winner == expected_winner:
        _record(checks, "selection_decision", _PASSED, note)
        return
    _record(
        checks,
        "selection_decision",
        _FAILED,
        "stored decision "
        + repr(stored_decision)
        + "/winner "
        + repr(stored_winner)
        + " but replay derives "
        + repr(expected_decision)
        + "/"
        + repr(expected_winner)
        + " ("
        + note
        + ")",
    )


def _check_selected_boundary(data: dict[str, Any], checks: list[dict[str, str]]) -> None:
    """Check selected boundary."""
    key = "selected_boundary_token_ids"
    selected = data.get("selected_candidate")
    stored = data.get(key)
    if selected is None:
        if stored is not None:
            _record(
                checks,
                "selected_boundary",
                _FAILED,
                key + " set without a selected candidate",
            )
            return
        _record(checks, "selected_boundary", _PASSED)
        return
    entry = next((e for e in data["search"] if _entry_name(e) == selected), None)
    if entry is None:
        _record(
            checks,
            "selected_boundary",
            _FAILED,
            "selected candidate has no search entry",
        )
        return
    expected = _entry_boundary(entry, data["boundary_token_ids"])
    if (
        expected is None
        or type(stored) is not list
        or not _valid_token_ids(stored)
        or list(stored) != list(expected)
    ):
        _record(
            checks,
            "selected_boundary",
            _FAILED,
            key + " does not match the selected candidate's boundary",
        )
        return
    _record(checks, "selected_boundary", _PASSED)


def _observations_present(entry: Any) -> bool:
    """Return the observations present."""
    observations = entry.get("observations") if type(entry) is dict else None
    return type(observations) is list and bool(observations)


def _check_budget_frontier_replay(data: dict[str, Any], checks: list[dict[str, str]]) -> None:
    """Check budget frontier replay."""
    stored = data.get("budget_frontier")
    if stored is None:
        _record(
            checks,
            "budget_frontier_replay",
            _UNAVAILABLE,
            "artifact predates the token-budget frontier",
        )
        return
    if type(stored) is not dict:
        _record(checks, "budget_frontier_replay", _FAILED, "budget_frontier malformed")
        return
    if stored.get("available") is False:
        # The CLI's degrade-to-unavailable marker (successful payloads
        # carry no "available" key at all).
        detail = stored.get("detail")
        if type(detail) is str and detail:
            _record(checks, "budget_frontier_replay", _PASSED, "unavailable: " + detail)
        else:
            _record(checks, "budget_frontier_replay", _FAILED, "unavailable without a reason")
        return
    if any(not _observations_present(entry) for entry in data["search"]):
        _record(
            checks,
            "budget_frontier_replay",
            _UNAVAILABLE,
            "observations stripped; rerun with --include-evidence",
        )
        return
    try:
        from .experiments import budget_frontier

        recomputed = budget_frontier(
            data["search"],
            boundary=tuple(data["boundary_token_ids"]),
            max_new_tokens=data["max_new_tokens"],
            native_decision=data["decision"],
            native_winner=data.get("selected_candidate"),
        )
    except ValueError as exc:
        _record(checks, "budget_frontier_replay", _FAILED, "recompute failed: " + str(exc))
        return
    if _json_equal(recomputed, stored):
        _record(checks, "budget_frontier_replay", _PASSED, "exact replay match")
    else:
        _record(
            checks,
            "budget_frontier_replay",
            _FAILED,
            "recomputed frontier differs from the stored section",
        )


def _check_held_out_floor_replay(data: dict[str, Any], checks: list[dict[str, str]]) -> None:
    """Check held out floor replay."""
    stored = data.get("held_out_budget_floor")
    if stored is None:
        _record(
            checks,
            "held_out_floor_replay",
            _UNAVAILABLE,
            "artifact predates the held-out budget floor",
        )
        return
    if type(stored) is not dict:
        _record(checks, "held_out_floor_replay", _FAILED, "held_out_budget_floor malformed")
        return
    if stored.get("available") is False:
        # The CLI's degrade-to-unavailable marker (successful payloads
        # carry no "available" key at all).
        detail = stored.get("detail")
        if type(detail) is str and detail:
            _record(checks, "held_out_floor_replay", _PASSED, "unavailable: " + detail)
        else:
            _record(checks, "held_out_floor_replay", _FAILED, "unavailable without a reason")
        return
    held = data.get("held_out")
    observations = held.get("observations") if type(held) is dict else None
    if type(observations) is not list or not observations:
        _record(
            checks,
            "held_out_floor_replay",
            _UNAVAILABLE,
            "held-out observations stripped; rerun with --include-evidence",
        )
        return
    held_boundary = data.get("selected_boundary_token_ids") or data["boundary_token_ids"]
    if not _valid_token_ids(held_boundary):
        _record(checks, "held_out_floor_replay", _FAILED, "held-out boundary malformed")
        return
    try:
        from .experiments import held_out_budget_floor

        recomputed = held_out_budget_floor(
            observations,
            boundary=tuple(held_boundary),
            max_new_tokens=data["max_new_tokens"],
            native_verdict=data["decision"],
        )
    except ValueError as exc:
        _record(checks, "held_out_floor_replay", _FAILED, "recompute failed: " + str(exc))
        return
    if _json_equal(recomputed, stored):
        _record(checks, "held_out_floor_replay", _PASSED, "exact replay match")
    else:
        _record(
            checks,
            "held_out_floor_replay",
            _FAILED,
            "recomputed floor differs from the stored section",
        )


def _check_split_separation(data: dict[str, Any], checks: list[dict[str, str]]) -> None:
    """Check split separation."""
    if any(not _observations_present(entry) for entry in data["search"]):
        _record(
            checks,
            "split_separation",
            _UNAVAILABLE,
            "observations stripped; rerun with --include-evidence",
        )
        return
    # _observations_present already guaranteed a nonempty observations
    # list on a dict entry for every search entry (non-dict entries took
    # the unavailable branch above); observations themselves are still
    # fail-closed per element -- never AttributeError on hostile input.
    search_sets = []
    for entry in data["search"]:
        fixtures = set()
        for obs in entry["observations"]:
            fixtures.add(str(obs.get("fixture")) if type(obs) is dict else "")
        search_sets.append(fixtures)
    first = search_sets[0] if search_sets else set()
    if any(s != first for s in search_sets):
        _record(
            checks,
            "split_separation",
            _FAILED,
            "candidates were not evaluated on the same search fixtures",
        )
        return
    held = data.get("held_out")
    if type(held) is dict and _observations_present(held):
        held_ids = set()
        for obs in held["observations"]:
            held_ids.add(str(obs.get("fixture")) if type(obs) is dict else "")
        overlap = sorted(first & held_ids)
        if overlap:
            _record(
                checks,
                "split_separation",
                _FAILED,
                "held-out fixtures overlap search: "
                + ", ".join(_display_id(f) for f in overlap[:5]),
            )
            return
    _record(checks, "split_separation", _PASSED)


def verify_results(data: Any) -> dict[str, Any]:
    """Independently re-derive an experiment's recorded outcome.

    Takes the parsed ``results.json`` object and returns a report dict::

        {"verdict": "ok" | "failed",
         "checks": [{"id": ..., "status": "passed"|"failed"|"unavailable",
                     "detail": ...}, ...]}

    The verdict is ``"ok"`` when no check failed; ``"unavailable"``
    checks (cancelled runs, stripped observations, pre-diagnostic
    artifacts) do not fail the verification. Pure: no I/O, no model.
    """
    checks: list[dict[str, str]] = []
    shaped = _check_schema(data, checks)
    if shaped is None:
        for check_id in (
            "candidates_unique",
            "summaries_consistent",
            "selection_decision",
            "selected_boundary",
            "budget_frontier_replay",
            "held_out_floor_replay",
            "split_separation",
        ):
            _record(checks, check_id, _UNAVAILABLE, "schema check failed")
        return {"verdict": "failed", "checks": checks}
    _check_candidates_unique(shaped, checks)
    _check_summaries_consistent(shaped, checks)
    _check_selection_decision(shaped, checks)
    _check_selected_boundary(shaped, checks)
    _check_budget_frontier_replay(shaped, checks)
    _check_held_out_floor_replay(shaped, checks)
    _check_split_separation(shaped, checks)
    verdict = "ok" if all(c["status"] != _FAILED for c in checks) else "failed"
    return {"verdict": verdict, "checks": checks}


def _render_summary(report: dict[str, Any], label: str) -> str:
    """Render summary."""
    checks = report["checks"]
    passed = sum(1 for c in checks if c["status"] == _PASSED)
    failed = [c for c in checks if c["status"] == _FAILED]
    unavailable = [c for c in checks if c["status"] == _UNAVAILABLE]
    lines = [
        "verify: "
        + _display_id(label)
        + ": "
        + str(passed)
        + "/"
        + str(len(checks))
        + " checks passed"
        + (", " + str(len(unavailable)) + " unavailable" if unavailable else "")
        + ", verdict "
        + report["verdict"]
    ]
    for check in failed:
        detail = " (" + check["detail"] + ")" if check["detail"] else ""
        lines.append("FAILED: " + check["id"] + detail)
    for check in unavailable:
        detail = " (" + check["detail"] + ")" if check["detail"] else ""
        lines.append("unavailable: " + check["id"] + detail)
    return "\n".join(lines)


def _append_determinism_check(report: dict[str, Any], audit: dict[str, Any]) -> None:
    """Map a determinism audit onto verify's check vocabulary.

    ``diverged`` is ``failed``: the recorded outputs positively
    contradict the re-execution, which is verify's standing doctrine
    for a failed check. Every other non-deterministic outcome is
    ``unavailable`` -- the audit could not establish the claim either
    way; the JSON ``determinism`` section carries the full result.
    """
    checks = report["checks"]
    if not audit["available"]:
        _record(checks, "determinism_replay", _UNAVAILABLE, audit["detail"])
    elif audit["verdict"] == "deterministic":
        _record(
            checks,
            "determinism_replay",
            _PASSED,
            str(audit["n_reproduced"]) + " re-executed search cells reproduced token-identically",
        )
    elif audit["verdict"] == "diverged":
        _record(
            checks,
            "determinism_replay",
            _FAILED,
            str(audit["n_diverged"])
            + " of "
            + str(audit["n_attempted"])
            + " re-executed search cells diverged (backend-determinism "
            "finding; see the determinism section)",
        )
    else:
        detail = (
            "determinism not established: "
            + str(audit["n_error"])
            + " error cells, "
            + str(audit["n_skipped"])
            + " skipped cells"
        )
        if audit["search_incomplete"]:
            detail += ", search incomplete"
        _record(checks, "determinism_replay", _UNAVAILABLE, detail)
    report["determinism"] = audit
    report["verdict"] = "ok" if all(c["status"] != _FAILED for c in checks) else "failed"


# skipcq: PY-R1000 - Gate-pinned frozen function; splitting breaks its verified contract.
def run_verify(args: argparse.Namespace, _parser: argparse.ArgumentParser) -> int:
    """Implement ``mpl verify``; returns the process exit code.

    JSON goes to stdout, the human-readable summary to stderr (mirroring
    ``diff-reports``). Exit 0 when every applicable check passes, 2 when
    any check fails, 1 for unreadable/invalid input. With ``--bundle`` the
    bundle path is verified by ``pack.run_verify_bundle`` instead. With
    ``--determinism`` the recorded search observations are additionally
    re-executed against the ``--model`` checkpoint (see
    ``determinism.py``); the audit's check is appended before the
    verdict is recomputed, and backend construction is skipped entirely
    when the artifact precheck fails.
    """
    from .diffreports import load_results

    if args.bundle is not None and args.results is not None:
        print("verify: exactly one of results and --bundle", file=sys.stderr)
        return 1
    if args.determinism and args.bundle is not None:
        print("verify: --determinism cannot be combined with --bundle", file=sys.stderr)
        return 1
    if args.determinism and args.expect_digest is not None:
        print(
            "verify: --determinism cannot be combined with --expect-digest",
            file=sys.stderr,
        )
        return 1
    if args.model is not None and not args.determinism:
        print("verify: --model requires --determinism", file=sys.stderr)
        return 1
    if args.bundle is not None:
        from .pack import run_verify_bundle

        return run_verify_bundle(args)
    if args.expect_digest is not None:
        print("verify: --expect-digest requires --bundle", file=sys.stderr)
        return 1
    if args.determinism and args.results is None:
        print("verify: --determinism requires a results path", file=sys.stderr)
        return 1
    if args.determinism and args.model is None:
        print("verify: --determinism requires --model", file=sys.stderr)
        return 1
    if args.results is None:
        print("verify: exactly one of results and --bundle", file=sys.stderr)
        return 1
    try:
        data = load_results(args.results)
    except ValueError as exc:
        # Fixed error vocabulary from load_results; safe to show and never
        # a traceback.
        print(f"verify: {exc}", file=sys.stderr)
        return 1
    report = verify_results(data)
    if args.determinism:
        from .determinism import audit_determinism, precheck

        if precheck(data) is not None:
            # audit_determinism returns the unavailable report without
            # touching the backend when planning fails, so no
            # checkpoint is constructed (or loaded) for it.
            audit = audit_determinism(data, None)
        else:
            from .backends import LocalHFBackend

            try:
                backend = LocalHFBackend(
                    args.model, device=args.device, max_context=args.max_context
                )
            except Exception:  # noqa: BLE001 -- fixed message, never a traceback
                print(
                    "verify: determinism audit unavailable: local backend could not be initialized",
                    file=sys.stderr,
                )
                return 1
            audit = audit_determinism(
                data,
                backend,
                replay={"device": args.device, "max_context": args.max_context},
            )
        _append_determinism_check(report, audit)
    print(json.dumps(report, indent=2, ensure_ascii=True, sort_keys=True))
    print(_render_summary(report, str(args.results)), file=sys.stderr)
    return 0 if report["verdict"] == "ok" else 2


def _expect_digest_type(value: str) -> str:
    """Expect digest type."""
    if re.fullmatch(r"[0-9a-f]{64}", value or "") is None:
        raise argparse.ArgumentTypeError("expect-digest must be 64 lowercase hex characters")
    return value


def build_verify_parser(sub: Any) -> None:
    """Register the ``verify`` subcommand (called from cli.build_parser)."""
    verify_cmd = sub.add_parser(
        "verify",
        help="Independently re-derive an experiment's recorded decision and diagnostics. "
        "Monday morning, a user asks 'is this result self-consistent and was "
        "it edited since packing' — verify answers the first, and "
        "--bundle answers the second",
        allow_abbrev=False,
    )
    verify_cmd.add_argument(
        "results",
        type=Path,
        nargs="?",
        default=None,
        help="Report directory (containing results.json) or results.json path",
    )
    verify_cmd.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help="Sealed bundle to verify (exactly one of results and --bundle)",
    )
    verify_cmd.add_argument(
        "--expect-digest",
        type=_expect_digest_type,
        default=None,
        help="64-char hex bundle digest obtained through an independent "
        "channel (only with --bundle)",
    )
    verify_cmd.add_argument(
        "--determinism",
        action="store_true",
        help="Re-execute the recorded search observations against a local "
        "checkpoint and audit token-exact reproduction. Monday morning, a "
        "reviewer asks 'prove the backend actually generated those tokens "
        "deterministically' — this answers with a certificate or a "
        "per-cell divergence table (requires a results path and --model)",
    )
    verify_cmd.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Local checkpoint for --determinism (requires --determinism)",
    )
    verify_cmd.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Replay device for --determinism (default cpu)",
    )
    verify_cmd.add_argument(
        "--max-context",
        type=int,
        default=2048,
        help="Replay context limit for --determinism (default 2048)",
    )
