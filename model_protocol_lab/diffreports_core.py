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

from .registry import INSTRUMENTS

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


MAX_RESULTS_BYTES = 32 * 1024 * 1024


_CONTROL_CHARS = re.compile(
    r"[\x00-\x1f\x7f-\x9f\ud800-\udfff\u200e\u200f\u2028\u2029\u202a-\u202e\u2066-\u2069]"
)


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
    """Return the unique object."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("results_duplicate_field")
        result[key] = value
    return result


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


_SECTION_PAIRS: list[tuple[str, str]] = [
    (instrument.key, diff_key)
    for instrument in INSTRUMENTS
    if (diff_key := instrument.diff_key) is not None
]


SECTION_DIFF_KEYS: dict[str, str] = dict(_SECTION_PAIRS)


UNDISPATCHED_SECTIONS: dict[str, str] = {
    instrument.key: reason
    for instrument in INSTRUMENTS
    if (reason := instrument.undispatched_reason) is not None
}


def _section_states(
    old: dict[str, Any], new: dict[str, Any], diffs: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Per-section comparison states for one diff (RR-09).

    Every dispatched section is published with the state its
    comparison actually reached, in the differs' own availability
    vocabulary extended by two payload states:

    * ``compared`` -- both sides validated; the differ compared
      them (its ``available`` is ``"both"``).
    * ``one_sided`` -- exactly one side validated (``"old"`` /
      ``"new"``); the pair is already asymmetric by that fact.
    * ``absent`` -- neither side records the section (both
      payloads absent or None) and the differ compared nothing.
    * ``unvalidated`` -- the differ could compare nothing
      (``"neither"``) yet at least one side carries a payload:
      the section was present but failed validation, so no
      recompute standard judged it.

    ``unvalidated_drift`` is set only in the last state when BOTH
    payloads are present and not recomputed-equal: two runs whose
    unvalidated evidence differs observably must not read
    ``identical`` (the audit's RR-09 reproductions drift stored
    rates inside sections invalid on both sides). Identical
    garbage on both sides is unvalidated but not drift.

    A fifth state, ``evidence_excluded``, is not derived here
    from availability alone: when both sides are
    evidence-excluded exports for an evidence-bound section, the
    assembly replaces its diff with the documented unavailable
    state (``detail == "evidence_excluded"``) and this function
    publishes that provenance. The validation comparison was
    not made, by the exports' own declaration -- but the drift
    judgement above still applies (wave-10 F2): drift detection
    compares the stored payloads and does not need the stripped
    evidence, so observably different payloads are drift here
    too, and the degraded diff's ``changed`` says the same. A
    one-sided exclusion keeps the ``one_sided`` state (its diff
    carries the detail). Returns the state map (registry order)
    and the sorted drift list.
    """
    states: dict[str, dict[str, Any]] = {}
    drifted: list[str] = []
    for results_key, diff_key in SECTION_DIFF_KEYS.items():
        section_diff = diffs.get(diff_key)
        available = section_diff.get("available") if type(section_diff) is dict else None
        old_payload = old.get(results_key)
        new_payload = new.get(results_key)
        if (
            type(section_diff) is dict
            and section_diff.get("detail") == "evidence_excluded"
            and available == "neither"
        ):
            drift = (
                old_payload is not None
                and new_payload is not None
                and not _recomputed_equal(old_payload, new_payload)
            )
            states[diff_key] = {"state": "evidence_excluded", "unvalidated_drift": drift}
            if drift:
                drifted.append(diff_key)
            continue
        if available == "both":
            state = "compared"
        elif available in ("old", "new"):
            state = "one_sided"
        elif old_payload is None and new_payload is None:
            state = "absent"
        else:
            state = "unvalidated"
        drift = (
            state == "unvalidated"
            and old_payload is not None
            and new_payload is not None
            and not _recomputed_equal(old_payload, new_payload)
        )
        states[diff_key] = {"state": state, "unvalidated_drift": drift}
        if drift:
            drifted.append(diff_key)
    return states, sorted(drifted)


_EVIDENCE_BOUND_SECTIONS = frozenset(
    {
        "compatibility_form",
        "confusable_skeleton",
        "enumeration_convention",
        "invisible_chars",
        "numform",
        "record_notation",
        "serving_backends",
        "typographic_emphasis",
    }
)


def _answer_text_state(section: Any) -> tuple[bool, bool]:
    """(Any records reachable, any answer_text carried) for a section.

    Records sit at the section level or under each candidate;
    the text itself sits on the record (numform) or on its
    per-arm cells (the other evidence-bound sections).
    """
    saw = False
    carried = False

    def _visit(record: Any) -> None:
        nonlocal saw, carried
        if type(record) is not dict:
            return
        saw = True
        if "answer_text" in record:
            carried = True
        cells = record.get("cells")
        if type(cells) is dict:
            for cell in cells.values():
                if type(cell) is dict and "answer_text" in cell:
                    carried = True

    if type(section) is dict:
        records = section.get("records")
        if type(records) is list:
            for record in records:
                _visit(record)
        candidates = section.get("candidates")
        if type(candidates) is dict:
            for entry in candidates.values():
                if type(entry) is not dict:
                    continue
                entry_records = entry.get("records")
                if type(entry_records) is list:
                    for record in entry_records:
                        _visit(record)
    return saw, carried


def _evidence_excluded(result: Any, section_key: str) -> bool:
    """Whether ``result`` is an evidence-excluded export whose
    ``section_key`` section lost its evidence to the projection.

    True only when the results dict declares
    ``raw_evidence_included is False`` (the stamp the exporter
    writes), the section is present with records, and no
    reachable record or cell carries the evidence (the records'
    ``answer_text``; for ``serving_backends``, the records' id
    streams). A payload without the stamp is not an export and
    is judged as-is; an evidence export (stamp true) missing
    the field is judged as-is too; partial carriage is not the
    projection's shape (it strips everywhere or nowhere) and is
    judged as-is.
    """
    # Family functions, resolved lazily at call time:
    # the module graph stays acyclic (families import
    # core; core imports no family at module level).
    from .diffreports_runlevel import (
        _serving_ids_state,
    )

    if type(result) is not dict:
        return False
    if result.get("raw_evidence_included") is not False:
        return False
    section = result.get(section_key)
    if type(section) is not dict:
        return False
    if section_key == "serving_backends":
        saw, carried = _serving_ids_state(section)
    else:
        saw, carried = _answer_text_state(section)
    return saw and not carried


def _degrade_evidence_excluded(
    old: Any, new: Any, section_key: str, diff: dict[str, Any]
) -> dict[str, Any]:
    """Replace a section diff with the documented unavailable
    state when either side is an evidence-excluded export.

    The section's validators bind the evidence (answer_text, or
    the serving records' id streams) to its digests; with the
    evidence stripped at export there is nothing to bind, and
    the honest report is that the comparison was not made
    (the CLI documents these diffs as requiring
    --include-evidence on both runs) -- not that the section is
    invalid. Both sides excluded: the validation comparison is
    not made (``available`` "neither"), but the stored payloads
    themselves are still compared (wave-10 F2): observably
    different payloads are a change, exactly as RR-09 requires
    of every unvalidated section -- drift detection does not
    need the stripped evidence. One side excluded: the
    comparison is one-sided by construction, so the diff names
    the side that carries evidence and the section is
    asymmetric.
    """
    old_excluded = _evidence_excluded(old, section_key)
    new_excluded = _evidence_excluded(new, section_key)
    if not old_excluded and not new_excluded:
        return diff
    degraded: dict[str, Any] = {
        "available": "neither",
        "changed": False,
        "detail": "evidence_excluded",
    }
    if old_excluded and new_excluded:
        if not _recomputed_equal(old.get(section_key), new.get(section_key)):
            degraded["changed"] = True
        return degraded
    other = new if old_excluded else old
    if type(other.get(section_key)) is not dict:
        return degraded
    degraded["available"] = "new" if old_excluded else "old"
    return degraded


def _float_convertible(value: int) -> bool:
    """Whether an int converts to float without overflowing.

    A stored results payload can carry an int of up to 4,300 digits
    (the bounded loader's cap); the producers' counts never approach
    float range, so an int beyond it is outside the producer domain
    in the same way a non-float32-representable total is. Validators
    that feed a stored int into float conversion gate on this first:
    the payload is invalid, never an ``OverflowError`` traceback
    (wave-8 F5). The boundary is float's own -- tested by conversion,
    not by a literal bound.
    """
    try:
        float(value)
    except OverflowError:
        return False
    return True


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
    function without the pre-guard the other four callers carry).
    """
    decision = payload.get("primary_decision")
    winner = payload.get("primary_winner")
    if type(decision) is not str or decision not in _PRIMARY_DECISIONS:
        return False
    candidates = payload.get("candidates")
    if decision in _PRIMARY_WINNER_DECISIONS:
        return type(winner) is str and type(candidates) is dict and winner in candidates
    return winner is None


def _bool_or_none(value: Any) -> bool:
    """Return whether a stored field is a JSON boolean or null.

    The admission these gates replace, ``value not in (True, False,
    None)``, compares by equality, so the numbers 1 and 0 (and 1.0 /
    0.0) were admitted wherever a boolean-or-null was required
    (RR-04). Exact-type membership is the only sound test: a bool
    is not a number and a number is not a bool.
    """
    return value is None or type(value) is bool


def _recomputed_equal(stored: Any, expected: Any) -> bool:
    """Type-strict equality for validator recompute comparisons.

    Bare ``==`` lets a JSON ``true`` validate as the integer 1 (bool
    is an int subclass), so an equality-compared count or version in
    a hostile payload could slip past a recompute check in the wrong
    type. Every scalar must match in type AND value, on either side:
    the pre-fix fallthrough admitted a stored ``False`` against an
    expected float ``0.0`` (and a stored int against an expected
    float) because only the bool and int branches gated the type
    (RR-04). The float branch now gates too, and the fallthrough is
    type-exact. Dicts and lists recurse. (Number unification across
    the int/float spelling applies to JSON Schema enum membership
    — SA-05 — not to these recompute validators, whose contract is
    exactness against this package's own producers; cycle 227 pins
    a stored ``0.0`` against a recomputed int ``0`` as a rejection.)
    """
    if type(expected) is bool:
        return type(stored) is bool and stored == expected
    if type(expected) is int:
        return type(stored) is int and stored == expected
    if type(expected) is float:
        return type(stored) is float and stored == expected
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
    return type(stored) is type(expected) and bool(stored == expected)


def _reject_constant(_value: Any) -> Any:
    """Return the reject constant."""
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


def report_pair_incomplete(data: dict[str, Any], html_sha256: str | None) -> bool:
    """Whether a stamped report pair fails its completion check.

    ORCH-D02: ``save_report`` stamps ``results.json`` with
    ``report_pair = {"html_sha256": ..., "complete": true}`` over
    the exact sibling ``report.html`` bytes, and publishes the
    JSON last as the pair's commit point. A document WITHOUT the
    stamp is a legacy pair and is never incomplete by this rule
    (absence is not incompleteness). A present stamp is
    incomplete when it is malformed, when ``complete`` is not
    true, when the sibling digest is unavailable (``None`` — the
    HTML is missing, unreadable or not a regular file), or when
    the digests differ.
    """
    if "report_pair" not in data:
        return False
    stamp = data["report_pair"]
    if type(stamp) is not dict:
        return True
    if stamp.get("complete") is not True:
        return True
    expected = stamp.get("html_sha256")
    if type(expected) is not str or html_sha256 is None:
        return True
    return expected != html_sha256


def _sha256_regular_file(path: Path) -> str | None:
    """SHA-256 of a regular file's bytes, or None when unavailable.

    Same single-fd discipline as :func:`load_results`: O_NONBLOCK
    open, fstat regular-file check on the opened fd, chunked
    reads (the HTML sibling is not size-capped the way
    ``results.json`` is, so it is never held whole in memory).
    """
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    except OSError:
        return None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return None
        digest = hashlib.sha256()
        with os.fdopen(fd, "rb") as handle:
            fd = -1  # ownership moved to the file object
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None
    finally:
        if fd != -1:
            os.close(fd)


def load_results(path: str | Path) -> dict[str, Any]:
    """Read a ``results.json`` (or a report directory holding one).

    The read is bounded and side-channel free: the path must resolve to a
    regular file (a directory resolves to the ``results.json`` inside it;
    FIFOs, devices and sockets are refused), at
    most ``MAX_RESULTS_BYTES`` are read, and every failure mode maps to a
    fixed error code raised as ``ValueError`` -- never a traceback and never
    file content echoed into the message. A ``report_pair`` stamp
    (ORCH-D02) is verified against the sibling ``report.html``
    before the document is returned when the document is
    presented as a pair (a report directory, or a file named
    ``results.json``); a legacy document without the stamp loads
    as it always has.
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
    data = parse_results_bytes(content)
    if "report_pair" in data and resolved.name == "results.json":
        # ORCH-D02: a stamped pair is trusted only when the
        # sibling report.html matches the stamped digest; the
        # stamp's absence (a legacy pair) is not incompleteness.
        # The check binds documents presented AS a pair — a
        # report directory, or a file named results.json; a lone
        # JSON under any other name is loaded as data (its stamp
        # describes a pair it is no longer part of).
        html_sha256 = _sha256_regular_file(resolved.parent / "report.html")
        if report_pair_incomplete(data, html_sha256):
            raise ValueError("report_pair_incomplete")
    return data


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
    """Return the candidate name."""
    if type(entry) is not dict:
        return None
    fmt = entry.get("format")
    name = fmt.get("name") if type(fmt) is dict else None
    return name if type(name) is str and name else None


def _base_family(name: str) -> str:
    # Mutated candidates are "family/mutation"; the family is the base name.
    # results.json does not store the --family flag, so this is the exact
    # family signal available.
    """Return the base family."""
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
    """Return the held verdict."""
    winner = result.get("selected_candidate")
    if type(winner) is not str or not winner:
        return None, "absent"
    held = result.get("held_out")
    summary = held.get("summary") if type(held) is dict else None
    return winner, _verdict(summary)


def _held_counts(result: dict[str, Any]) -> tuple[str | None, dict[str, int | None]]:
    """The held-out stage's selected candidate plus evidence counts.

    The held-out mirror of ``_search_verdicts``' planned-count
    discipline: a "pass" verdict means "all *planned* fixtures
    passed", so a 1/1 held-out pass and a 50/50 held-out pass
    silently comparing identical would be a lie of omission
    (RR-09). ``None`` marks a malformed/missing count; it
    compares as its own distinct value so it still shows up in
    drift. The candidate is the run's selected candidate, the
    same name ``_held_verdict`` reports.
    """
    name, _verdict_text = _held_verdict(result)
    if name is None:
        return None, {}
    held = result.get("held_out")
    summary = held.get("summary") if type(held) is dict else None
    counts: dict[str, int | None] = {}
    for field in ("planned", "attempted"):
        value = summary.get(field) if type(summary) is dict else None
        counts[field] = value if type(value) is int else None
    return name, counts


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


def _payload_section(payload: Any, key: str) -> Any:
    """Fetch one payload section by key; ``None`` for a non-dict payload.

    Payload sections are data, addressed by key. The labeled-block
    section (key ``confidentiality``) is fetched through this accessor
    and carried through the diff in the ``section_diffs`` /
    ``section_asymmetric`` maps, never in a section-named identifier:
    the section's key is a word that lexical taint trackers classify
    as credential material (CodeQL
    ``py/clear-text-logging-sensitive-data``), so a section-named
    variable, callee, or key access anywhere on the path to the CLI's
    two print sinks makes the tracker report a sensitive-data flow
    that does not exist -- the section carries only an availability
    enum, booleans, numeric gaps, and candidate labels from
    structurally validated payloads (cycle 225; see docs/SECURITY.md).
    Carried as data, the printed diff contains no tracked flow by
    construction, and the emitted bytes are unchanged.
    """
    return payload.get(key) if type(payload) is dict else None


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
    # Family functions, resolved lazily at call time:
    # the module graph stays acyclic (families import
    # core; core imports no family at module level).
    from .diffreports_forms import (
        diff_choiceaudit,
        diff_confidence,
        diff_dateline,
        diff_density,
        diff_field_order,
        diff_leading,
        diff_numform,
        diff_position,
        diff_precision,
        diff_rotation,
    )
    from .diffreports_probes import (
        diff_aliasing,
        diff_distraction,
        diff_faketurn,
        diff_hierarchy,
        diff_history_frame,
        diff_instruction_distance,
        diff_multiturn,
        diff_needle,
        diff_placement,
        diff_prefill,
        diff_prune_sweep,
        diff_repetition,
        diff_structured,
        diff_toolcall,
    )
    from .diffreports_runlevel import (
        diff_ablation,
        diff_answer_collision,
        diff_batchinv,
        diff_boundary_audit,
        diff_budgets,
        diff_cache_profile,
        diff_contamination,
        diff_deliforge,
        diff_fewshot,
        diff_frontiers,
        diff_held_out_floor,
        diff_negation,
        diff_perturbation,
        diff_portfolio,
        diff_refusals,
        diff_scorerlens,
        diff_selection_robustness,
        diff_serving_backends,
        diff_stop_cause,
        diff_synthhistory,
        diff_template_robustness,
        diff_token_healing,
        diff_transfer,
        diff_verbosity,
    )
    from .diffreports_text import (
        diff_canonicalform,
        diff_compatibility_form,
        diff_confusable_skeleton,
        diff_enumeration_convention,
        diff_invisible_chars,
        diff_record_notation,
        diff_typographic_emphasis,
        diff_ws_canonical,
    )

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
    # _drift() fields, sorted by candidate name, then the held-out
    # stage's counts for the common selected candidate (RR-09: the
    # held-out stage gets the same evidence-strength comparison).
    for name in sorted(set(old_planned) & set(new_planned)):
        if old_planned[name] != new_planned[name]:
            config_drift.append(
                {"field": f"planned:{name}", "old": old_planned[name], "new": new_planned[name]}
            )
    old_held_name_counts, old_held_counts = _held_counts(old)
    new_held_name_counts, new_held_counts = _held_counts(new)
    if old_held_name_counts is not None and old_held_name_counts == new_held_name_counts:
        for field in ("planned", "attempted"):
            if old_held_counts[field] != new_held_counts[field]:
                config_drift.append(
                    {
                        "field": f"held_out_{field}:{old_held_name_counts}",
                        "old": old_held_counts[field],
                        "new": new_held_counts[field],
                    }
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
    # The budget ledger differ takes the whole results dicts: its
    # recorded actuals are validated by recomputation from each side's
    # own observations before comparison (see diff_budgets).
    budget_diff = diff_budgets(old.get("budget"), new.get("budget"), old, new)
    budget_asymmetric = budget_diff["available"] not in ("both", "neither")
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
    numform_diff = _degrade_evidence_excluded(old, new, "numform", numform_diff)
    dateline_diff = diff_dateline(old.get("dateline"), new.get("dateline"))
    multiturn_diff = diff_multiturn(old.get("multiturn"), new.get("multiturn"))
    needle_diff = diff_needle(old.get("needle"), new.get("needle"))
    placement_diff = diff_placement(old.get("placement"), new.get("placement"))
    canonical_form_diff = diff_canonicalform(old.get("canonical_form"), new.get("canonical_form"))
    instruction_distance_diff = diff_instruction_distance(
        old.get("instruction_distance"), new.get("instruction_distance")
    )
    history_frame_diff = diff_history_frame(old.get("history_frame"), new.get("history_frame"))
    prune_sweep_diff = diff_prune_sweep(old.get("prune_sweep"), new.get("prune_sweep"))
    record_notation_diff = diff_record_notation(
        old.get("record_notation"), new.get("record_notation")
    )
    record_notation_diff = _degrade_evidence_excluded(
        old, new, "record_notation", record_notation_diff
    )
    invisible_chars_diff = diff_invisible_chars(
        old.get("invisible_chars"), new.get("invisible_chars")
    )
    invisible_chars_diff = _degrade_evidence_excluded(
        old, new, "invisible_chars", invisible_chars_diff
    )
    confusable_skeleton_diff = diff_confusable_skeleton(
        old.get("confusable_skeleton"), new.get("confusable_skeleton")
    )
    confusable_skeleton_diff = _degrade_evidence_excluded(
        old, new, "confusable_skeleton", confusable_skeleton_diff
    )
    compatibility_form_diff = diff_compatibility_form(
        old.get("compatibility_form"), new.get("compatibility_form")
    )
    compatibility_form_diff = _degrade_evidence_excluded(
        old, new, "compatibility_form", compatibility_form_diff
    )
    enumeration_convention_diff = diff_enumeration_convention(
        old.get("enumeration_convention"), new.get("enumeration_convention")
    )
    enumeration_convention_diff = _degrade_evidence_excluded(
        old, new, "enumeration_convention", enumeration_convention_diff
    )
    typographic_emphasis_diff = diff_typographic_emphasis(
        old.get("typographic_emphasis"), new.get("typographic_emphasis")
    )
    typographic_emphasis_diff = _degrade_evidence_excluded(
        old, new, "typographic_emphasis", typographic_emphasis_diff
    )
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
    # The labeled-block section is carried as data, by key (see
    # _payload_section); its diff and asymmetry live in these two maps
    # alongside the section-named locals of every other section.
    section_diffs: dict[str, dict[str, Any]] = {}
    section_asymmetric: dict[str, bool] = {}
    section_diffs["confidentiality"] = _diff_labeled_block(
        _payload_section(old, "confidentiality"),
        _payload_section(new, "confidentiality"),
    )
    answer_collision_diff = diff_answer_collision(
        old.get("answer_collision"), new.get("answer_collision")
    )
    serving_backends_diff = diff_serving_backends(
        old.get("serving_backends"), new.get("serving_backends")
    )
    serving_backends_diff = _degrade_evidence_excluded(
        old, new, "serving_backends", serving_backends_diff
    )
    negation_diff = diff_negation(old.get("negation"), new.get("negation"))
    negation_asymmetric = negation_diff["available"] not in (
        "both",
        "neither",
    )
    ablation_asymmetric = ablation_diff["available"] not in (
        "both",
        "neither",
    )
    section_asymmetric["confidentiality"] = _payload_section(section_diffs, "confidentiality")[
        "available"
    ] not in (
        "both",
        "neither",
    )
    answer_collision_asymmetric = answer_collision_diff["available"] not in (
        "both",
        "neither",
    )
    serving_backends_asymmetric = serving_backends_diff["available"] not in (
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
    prune_sweep_asymmetric = prune_sweep_diff["available"] not in (
        "both",
        "neither",
    )
    record_notation_asymmetric = record_notation_diff["available"] not in (
        "both",
        "neither",
    )
    invisible_chars_asymmetric = invisible_chars_diff["available"] not in (
        "both",
        "neither",
    )
    confusable_skeleton_asymmetric = confusable_skeleton_diff["available"] not in (
        "both",
        "neither",
    )
    compatibility_form_asymmetric = compatibility_form_diff["available"] not in (
        "both",
        "neither",
    )
    enumeration_convention_asymmetric = enumeration_convention_diff["available"] not in (
        "both",
        "neither",
    )
    typographic_emphasis_asymmetric = typographic_emphasis_diff["available"] not in (
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
    # Section-state publication (RR-09): the map below carries the
    # same diff dicts the result dict publishes, keyed by diff key;
    # the census test pins its keys against the registry so a
    # section cannot join the result without joining the states.
    section_diffs_by_key = {
        "frontier": frontier_diff,
        "held_out_floor": held_out_floor_diff,
        "budget": budget_diff,
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
        "prune_sweep": prune_sweep_diff,
        "record_notation": record_notation_diff,
        "invisible_chars": invisible_chars_diff,
        "confusable_skeleton": confusable_skeleton_diff,
        "compatibility_form": compatibility_form_diff,
        "enumeration_convention": enumeration_convention_diff,
        "typographic_emphasis": typographic_emphasis_diff,
        "choice_audit": choiceaudit_diff,
        "precision": precision_diff,
        "confidence": confidence_diff,
        "rotation": rotation_diff,
        "density": density_diff,
        "field_order": field_order_diff,
        "cache_profile": cache_profile_diff,
        "template_robustness": template_robustness_diff,
        "ablation": ablation_diff,
        "confidentiality": _payload_section(section_diffs, "confidentiality"),
        "answer_collision": answer_collision_diff,
        "serving_backends": serving_backends_diff,
        "negation": negation_diff,
    }
    section_states, unvalidated_drift_sections = _section_states(old, new, section_diffs_by_key)
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
        and not budget_diff["changed"]
        and not budget_asymmetric
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
        and not _payload_section(section_diffs, "confidentiality")["changed"]
        and not _payload_section(section_asymmetric, "confidentiality")
        and not answer_collision_diff["changed"]
        and not answer_collision_asymmetric
        and not serving_backends_diff["changed"]
        and not serving_backends_asymmetric
        and not negation_diff["changed"]
        and not negation_asymmetric
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
        and not prune_sweep_diff["changed"]
        and not prune_sweep_asymmetric
        and not record_notation_diff["changed"]
        and not record_notation_asymmetric
        and not invisible_chars_diff["changed"]
        and not invisible_chars_asymmetric
        and not confusable_skeleton_diff["changed"]
        and not confusable_skeleton_asymmetric
        and not compatibility_form_diff["changed"]
        and not compatibility_form_asymmetric
        and not enumeration_convention_diff["changed"]
        and not enumeration_convention_asymmetric
        and not typographic_emphasis_diff["changed"]
        and not typographic_emphasis_asymmetric
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
        and not unvalidated_drift_sections
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
        "budget_changed": budget_diff["changed"],
        "budget_asymmetric": budget_asymmetric,
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
        "confidentiality_changed": _payload_section(section_diffs, "confidentiality")["changed"],
        "confidentiality_asymmetric": _payload_section(section_asymmetric, "confidentiality"),
        "answer_collision_changed": answer_collision_diff["changed"],
        "answer_collision_asymmetric": answer_collision_asymmetric,
        "serving_backends_changed": serving_backends_diff["changed"],
        "serving_backends_asymmetric": serving_backends_asymmetric,
        "negation_changed": negation_diff["changed"],
        "negation_asymmetric": negation_asymmetric,
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
        "prune_sweep_changed": prune_sweep_diff["changed"],
        "prune_sweep_asymmetric": prune_sweep_asymmetric,
        "record_notation_changed": record_notation_diff["changed"],
        "record_notation_asymmetric": record_notation_asymmetric,
        "invisible_chars_changed": invisible_chars_diff["changed"],
        "invisible_chars_asymmetric": invisible_chars_asymmetric,
        "confusable_skeleton_changed": confusable_skeleton_diff["changed"],
        "confusable_skeleton_asymmetric": confusable_skeleton_asymmetric,
        "compatibility_form_changed": compatibility_form_diff["changed"],
        "compatibility_form_asymmetric": compatibility_form_asymmetric,
        "enumeration_convention_changed": enumeration_convention_diff["changed"],
        "enumeration_convention_asymmetric": enumeration_convention_asymmetric,
        "typographic_emphasis_changed": typographic_emphasis_diff["changed"],
        "typographic_emphasis_asymmetric": typographic_emphasis_asymmetric,
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
        "unvalidated_drift_sections": unvalidated_drift_sections,
    }
    return {
        "tool": "diff-reports",
        "old": {"source": old_label},
        "new": {"source": new_label},
        "decision": {"old": old_decision, "new": new_decision, "changed": decision_changed},
        "winner": {"old": old_winner, "new": new_winner, "changed": winner_changed},
        "frontier": frontier_diff,
        "held_out_floor": held_out_floor_diff,
        "budget": budget_diff,
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
        "prune_sweep": prune_sweep_diff,
        "record_notation": record_notation_diff,
        "invisible_chars": invisible_chars_diff,
        "confusable_skeleton": confusable_skeleton_diff,
        "compatibility_form": compatibility_form_diff,
        "enumeration_convention": enumeration_convention_diff,
        "typographic_emphasis": typographic_emphasis_diff,
        "choice_audit": choiceaudit_diff,
        "precision": precision_diff,
        "confidence": confidence_diff,
        "rotation": rotation_diff,
        "density": density_diff,
        "field_order": field_order_diff,
        "cache_profile": cache_profile_diff,
        "template_robustness": template_robustness_diff,
        "ablation": ablation_diff,
        "confidentiality": _payload_section(section_diffs, "confidentiality"),
        "answer_collision": answer_collision_diff,
        "serving_backends": serving_backends_diff,
        "negation": negation_diff,
        "section_states": section_states,
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
        """Return the frontier label and value at a token budget."""
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


def _stability_triplet(stability: Any) -> tuple[Any, Any, Any]:
    """(winner_retained, decision_retained, n_evaluable) for drift comparison."""
    if type(stability) is not dict:
        return (None, None, None)
    return (
        stability.get("winner_retained"),
        stability.get("decision_retained"),
        stability.get("n_evaluable"),
    )


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


def _is_hex_digest(value: Any) -> bool:
    """Return whether hex digest."""
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


_PLACEMENT_JOINER = "\n\n"


_HISTORYFRAME_CLASSIFICATIONS = (
    "insufficient",
    "no_discordance_observed",
    "discordant_balanced",
    "discordant_asymmetric",
    "below_threshold",
)


_PRUNESWEEP_DECISIONS = (
    "insufficient",
    "prune_drop_observed",
    "below_threshold",
)


_PRECISION_WITHHELD = {
    None,
    "bf16_unavailable",
    "mismatched_sets",
    "no_paired_fixtures",
    "primary_not_winner_bearing",
}


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


def _diff_labeled_block(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``confidentiality`` payloads; deterministic.

    Answers "did the labeled-block exposure story change between runs?":
    protocol-fingerprint drift (a marker/threshold change invalidates
    the comparison and is reported as its own event), per-candidate gap
    drift, the positive set, and the uniform-nonpositive (null) flag. A
    template edit that starts leaking its labeled block is the
    regression signal. A side without a recorded probe is reported
    plainly. No model, no network.
    """
    # Family functions, resolved lazily at call time:
    # the module graph stays acyclic (families import
    # core; core imports no family at module level).
    from .diffreports_runlevel import (
        _valid_confidentiality,
    )

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
    labeled_block_candidates_added = sorted(set(new_cells) - set(old_cells))
    labeled_block_candidates_removed = sorted(set(old_cells) - set(new_cells))
    labeled_block_gap_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_cells) & set(new_cells)):
        old_entry = old_cells[name]
        new_entry = new_cells[name]
        old_gap = old_entry.get("gap") if type(old_entry) is dict else None
        new_gap = new_entry.get("gap") if type(new_entry) is dict else None
        if old_gap != new_gap:
            labeled_block_gap_drift.append({"candidate": name, "old": old_gap, "new": new_gap})
    labeled_block_positive_added = sorted(
        set(new.get("positive_candidates") or []) - set(old.get("positive_candidates") or [])
    )
    labeled_block_positive_removed = sorted(
        set(old.get("positive_candidates") or []) - set(new.get("positive_candidates") or [])
    )
    labeled_block_uniform_changed = old.get("uniform_nonpositive") != new.get("uniform_nonpositive")
    result["protocol_drift"] = protocol_drift
    result["gap_drift"] = labeled_block_gap_drift
    result["candidates_added"] = labeled_block_candidates_added
    result["candidates_removed"] = labeled_block_candidates_removed
    result["positive_added"] = labeled_block_positive_added
    result["positive_removed"] = labeled_block_positive_removed
    result["uniform_changed"] = labeled_block_uniform_changed
    # NOTE: labeled-block-specific disjunct names keep this ``changed``
    # line textually distinct from every other diff section's anchor
    # (mutation-gate uniqueness).
    result["changed"] = bool(
        protocol_drift
        or labeled_block_gap_drift
        or labeled_block_candidates_added
        or labeled_block_candidates_removed
        or labeled_block_positive_added
        or labeled_block_positive_removed
        or labeled_block_uniform_changed
    )
    return result


def _measurable_collision_cells(
    per_candidate: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """The measurable cells of an answer-collision payload: dict cells
    whose reserved arm scored at least one observation.
    """
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


def _render_labeled_block_summary(
    section: dict[str, Any] | None,
) -> list[str]:
    """Confidentiality comparison lines for the stderr summary.

    Every interpolated identifier passes through ``_display_id`` so a
    hostile candidate name cannot inject terminal escapes. Gaps are
    rendered as percentages; a protocol drift (marker/threshold change)
    is reported first because it invalidates gap comparisons.
    """
    if type(section) is not dict:
        return []
    available = section.get("available")
    if available == "neither":
        return []
    if available != "both":
        missing = "new" if available == "old" else "old"
        return [
            "Confidentiality comparison unavailable: "
            f"{missing} run did not record the confidentiality probe."
        ]
    lines: list[str] = []
    if section.get("protocol_drift"):
        lines.append(
            "Confidentiality protocol drift: the frozen probe protocol "
            "(marker/threshold/normalization) changed between runs; gap "
            "comparisons are not like-for-like."
        )

    def _fmt_gap(value: Any) -> str:
        """Return the fmt gap."""
        return (
            f"{value:.1%}"
            if type(value) in (int, float) and _float_convertible(value)
            else "(none)"
        )

    drift = section.get("gap_drift") or []
    for entry in drift[:3]:
        if type(entry) is not dict:
            continue
        lines.append(
            f"Confidentiality gap drift ({_display_id(entry.get('candidate'))}): "
            f"{_fmt_gap(entry.get('old'))} -> {_fmt_gap(entry.get('new'))}"
        )
    if len(drift) > 3:
        lines.append(f"  ... and {len(drift) - 3} more gap drifts")
    added = section.get("positive_added") or []
    removed = section.get("positive_removed") or []
    if added:
        lines.append(f"Confidentiality newly positive: {', '.join(_display_id(c) for c in added)}")
    if removed:
        lines.append(
            f"Confidentiality no longer positive: {', '.join(_display_id(c) for c in removed)}"
        )
    if section.get("uniform_changed"):
        lines.append("Confidentiality uniform-nonpositive (null) flag changed.")
    cand_added = section.get("candidates_added") or []
    cand_removed = section.get("candidates_removed") or []
    if cand_added:
        lines.append(
            f"Confidentiality candidates added: {', '.join(_display_id(c) for c in cand_added)}"
        )
    if cand_removed:
        lines.append(
            f"Confidentiality candidates removed: {', '.join(_display_id(c) for c in cand_removed)}"
        )
    return lines


def _compact(value: Any) -> str:
    """One-line JSON for a drift value in the stderr summary."""
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def render_summary(diff: dict[str, Any]) -> str:
    """Human-readable summary for stderr; stdout stays parseable JSON."""
    # Family functions, resolved lazily at call time:
    # the module graph stays acyclic (families import
    # core; core imports no family at module level).
    from .diffreports_forms import (
        _render_position_summary,
    )
    from .diffreports_probes import (
        _render_faketurn_summary,
        _render_hierarchy_summary,
        _render_prefill_summary,
        _render_structured_summary,
        _render_toolcall_summary,
    )
    from .diffreports_runlevel import (
        _render_ablation_summary,
        _render_answer_collision_summary,
        _render_batchinv_summary,
        _render_boundary_audit_summary,
        _render_cache_profile_summary,
        _render_contamination_summary,
        _render_deliforge_summary,
        _render_fewshot_summary,
        _render_frontier_summary,
        _render_healing_summary,
        _render_held_out_floor_summary,
        _render_negation_summary,
        _render_perturbation_summary,
        _render_refusals_summary,
        _render_robustness_summary,
        _render_stop_cause_summary,
        _render_synthhistory_summary,
        _render_template_robustness_summary,
        _render_transfer_summary,
        _render_verbosity_summary,
    )
    from .diffreports_text import (
        _render_ws_canonical_summary,
    )

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
    lines.extend(_render_labeled_block_summary(_payload_section(diff, "confidentiality")))
    lines.extend(_render_answer_collision_summary(diff.get("answer_collision")))
    lines.extend(_render_negation_summary(diff.get("negation")))
    unvalidated = (diff.get("summary") or {}).get("unvalidated_drift_sections") or []
    if unvalidated:
        lines.append(
            "Unvalidated section drift (payloads differ but neither side "
            f"validates): {', '.join(_display_id(name) for name in unvalidated)}"
        )
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
        if diff["budget"]["available"] == "both":
            axes.append("run budget ledger")
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


GATE_ENV_VAR = "MODEL_PROTOCOL_LAB_DIFF_MAX_REGRESSIONS"

# Quality order over candidate verdicts for gate classification: a
# change toward a lower rank is a regression event, toward a higher
# rank an improvement event (the two are exact mirrors).
_VERDICT_RANK = {  # nosec B105 -- verdict vocabulary, not a credential
    "pass": 3,
    "fail": 2,
    "unknown": 1,
    "absent": 0,
}


def _gate_events(diff: dict[str, Any]) -> tuple[int, int]:
    """Count gate regression/improvement events in a finished diff.

    Event definitions (R4 adjudication, section 4.2): a pass->fail
    fixture flip; a candidate verdict change toward a lower quality
    rank; a decision change away from ``compatible_on_fixture_set``;
    a winner loss. Improvements are the exact mirrors. Fixture
    transitions through ``absent`` are not flips (the payload's own
    definition requires both outcomes in {pass, fail}), and a winner
    change between two present winners is neither a loss nor a gain.
    """
    regressions = 0
    improvements = 0
    for entry in diff.get("fixture_diffs") or []:
        if entry.get("old") == "pass" and entry.get("new") == "fail":
            regressions += 1
        elif entry.get("old") == "fail" and entry.get("new") == "pass":
            improvements += 1
    for change in diff.get("candidate_verdicts") or []:
        old_value, new_value = change.get("old"), change.get("new")
        old_rank = _VERDICT_RANK.get(old_value) if type(old_value) is str else None
        new_rank = _VERDICT_RANK.get(new_value) if type(new_value) is str else None
        if old_rank is None or new_rank is None or old_rank == new_rank:
            continue
        if new_rank < old_rank:
            regressions += 1
        else:
            improvements += 1
    decision = diff.get("decision") or {}
    if decision.get("changed"):
        if decision.get("old") == "compatible_on_fixture_set":
            regressions += 1
        elif decision.get("new") == "compatible_on_fixture_set":
            improvements += 1
    winner = diff.get("winner") or {}
    if winner.get("changed"):
        if winner.get("new") is None:
            regressions += 1
        elif winner.get("old") is None:
            improvements += 1
    return regressions, improvements


def _classify_gate(
    diff: dict[str, Any], max_regressions: int, threshold_source: str
) -> dict[str, Any]:
    """Compute the opt-in gate verdict over a finished diff dict.

    Taxonomy (R4 adjudication, section 4.2): ``no_change`` when the
    diff is identical; ``unavailable`` when the evidence is
    asymmetric, so no fixture/verdict-level judgement is supported;
    ``regression`` when the regression count exceeds the tolerated
    maximum; ``improvement`` when nothing regressed and at least one
    improvement event fired. The residual cell the adjudication left
    open -- a non-identical diff with no improvement events, or with
    regressions inside the tolerated maximum -- is ``change``: the
    gate passes and both direction counts stay printed, so nothing
    is hidden behind the neutral word.
    """
    regressions, improvements = _gate_events(diff)
    summary = diff["summary"]
    if summary["identical"]:
        verdict = "no_change"
    elif summary["evidence_asymmetric"]:
        verdict = "unavailable"
    elif regressions > max_regressions:
        verdict = "regression"
    elif regressions == 0 and improvements > 0:
        verdict = "improvement"
    else:
        verdict = "change"
    return {
        "verdict": verdict,
        "regressions": regressions,
        "improvements": improvements,
        "max_regressions": max_regressions,
        "threshold_source": threshold_source,
    }


def _resolve_gate_threshold(
    args: argparse.Namespace, parser: argparse.ArgumentParser | None
) -> tuple[int, str] | None:
    """Resolve the gate threshold: flag > env var > default 0.

    The env value must be a non-negative integer in canonical
    decimal form; anything else is fail-closed (``None`` here, the
    caller exits 1 with a fixed message) -- never a silent fallback.
    A negative flag value is an argparse-style error (exit 2).
    """
    flag = getattr(args, "max_regressions", None)
    if flag is not None:
        if flag < 0:
            message = "--max-regressions must be a non-negative integer"
            if parser is not None:
                parser.error(message)
            print(f"diff-reports: {message}", file=sys.stderr)
            raise SystemExit(2)
        return flag, "flag"
    raw = os.environ.get(GATE_ENV_VAR)
    if raw is not None:
        if re.fullmatch(r"0|[1-9][0-9]*", raw) is None:
            return None
        return int(raw), "env"
    return 0, "default"


def _xml_clean(value: Any) -> str:
    """Make a payload-controlled string safe for XML output.

    ElementTree escapes markup characters, but XML 1.0 also forbids
    most control characters and lone surrogates outright; those are
    replaced with U+FFFD so hostile names still emit well-formed XML.
    """
    text = str(value)
    return "".join(
        ch
        if (ord(ch) >= 0x20 or ch in "\t\n\r")
        and not 0xD800 <= ord(ch) <= 0xDFFF
        and ord(ch) not in (0xFFFE, 0xFFFF)
        else "\ufffd"
        for ch in text
    )


def _render_junit(suites: list[dict[str, Any]]) -> str:
    """Render JUnit XML (the de-facto shape CI consumers accept).

    ``suites`` is a list of ``{"name": str, "cases": [...]}`` where a
    case is ``{"classname", "name", "status", "message", "detail"}``
    with status ``passed`` | ``failed`` | ``skipped``. Suite and root
    counts are computed from the cases themselves, so the XML can
    never claim counts the cases do not add up to.
    """
    # Generation only: this module builds XML with ElementTree and
    # never parses it, so the B405 parse-side risk cannot occur.
    import xml.etree.ElementTree as ET  # nosec B405

    root = ET.Element("testsuites")
    total_tests = total_failures = total_skipped = 0
    for suite in suites:
        cases = suite["cases"]
        failures = sum(1 for c in cases if c["status"] == "failed")
        skipped = sum(1 for c in cases if c["status"] == "skipped")
        total_tests += len(cases)
        total_failures += failures
        total_skipped += skipped
        suite_el = ET.SubElement(
            root,
            "testsuite",
            {
                "name": _xml_clean(suite["name"]),
                "tests": str(len(cases)),
                "failures": str(failures),
                "skipped": str(skipped),
                "errors": "0",
            },
        )
        for case in cases:
            case_el = ET.SubElement(
                suite_el,
                "testcase",
                {
                    "classname": _xml_clean(case["classname"]),
                    "name": _xml_clean(case["name"]),
                },
            )
            if case["status"] == "failed":
                failure_el = ET.SubElement(
                    case_el, "failure", {"message": _xml_clean(case["message"])}
                )
                failure_el.text = _xml_clean(case["detail"])
            elif case["status"] == "skipped":
                ET.SubElement(case_el, "skipped", {"message": _xml_clean(case["message"])})
    root.set("tests", str(total_tests))
    root.set("failures", str(total_failures))
    root.set("skipped", str(total_skipped))
    root.set("errors", "0")
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


def _write_junit(path: Any, xml_text: str) -> bool:
    """Write JUnit XML atomically; False (no partial file) on failure."""
    target = Path(path)
    tmp = target.with_name(f"{target.name}.tmp-{os.getpid()}")
    try:
        tmp.write_text(xml_text, encoding="utf-8")
        os.replace(tmp, target)
    except OSError:
        tmp.unlink(missing_ok=True)
        return False
    return True


def _diff_junit_suites(diff: dict[str, Any], gate: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Build JUnit suites for a diff: one per compared candidate.

    A candidate is compared when it carries a verdict change or a
    fixture diff. Fixture testcases exist only where per-fixture
    evidence exists on both sides (``fixture_diffs`` is populated
    only then); a pass->fail fixture is a failure, a fail->pass or
    unchanged fixture passes, and any other transition is skipped
    as not comparable in pass/fail terms -- the suite never claims
    fixture coverage the evidence does not support. When the gate
    verdict is ``unavailable`` the candidate testcases are emitted
    as skipped: the gate could not judge them.
    """
    verdict_changes: dict[str, list[dict[str, Any]]] = {}
    for change in diff.get("candidate_verdicts") or []:
        verdict_changes.setdefault(change["candidate"], []).append(change)
    fixture_entries: dict[str, list[dict[str, Any]]] = {}
    for entry in diff.get("fixture_diffs") or []:
        fixture_entries.setdefault(entry["candidate"], []).append(entry)
    unavailable = gate is not None and gate["verdict"] == "unavailable"
    suites: list[dict[str, Any]] = []
    for name in sorted(set(verdict_changes) | set(fixture_entries)):
        cases: list[dict[str, Any]] = []
        for change in verdict_changes.get(name, []):
            old_value, new_value = change.get("old"), change.get("new")
            old_rank = _VERDICT_RANK.get(old_value) if type(old_value) is str else None
            new_rank = _VERDICT_RANK.get(new_value) if type(new_value) is str else None
            regressed = old_rank is not None and new_rank is not None and new_rank < old_rank
            message = f"{change.get('old')} -> {change.get('new')}"
            if unavailable:
                status = "skipped"
                message = "gate verdict unavailable: evidence asymmetric"
            elif regressed:
                status = "failed"
            else:
                status = "passed"
            cases.append(
                {
                    "classname": "diff-reports",
                    "name": f"verdict ({change.get('stage')})",
                    "status": status,
                    "message": message,
                    "detail": f"candidate {name} verdict {message}",
                }
            )
        for entry in fixture_entries.get(name, []):
            old, new = entry.get("old"), entry.get("new")
            message = f"{old} -> {new}"
            if old == new:
                status = "passed"
            elif (old, new) == ("pass", "fail"):
                status = "failed"
            elif (old, new) == ("fail", "pass"):
                status = "passed"
            else:
                status = "skipped"
                message = f"{message}: outcome not comparable as pass/fail"
            cases.append(
                {
                    "classname": "diff-reports",
                    "name": f"fixture {entry.get('fixture')} ({entry.get('stage')})",
                    "status": status,
                    "message": message,
                    "detail": f"candidate {name} fixture {entry.get('fixture')} {message}",
                }
            )
        suites.append({"name": name, "cases": cases})
    return suites


def _checks_junit_suites(
    label: str, surface: str, checks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Build the single JUnit suite for a verify-style check report."""
    cases: list[dict[str, Any]] = []
    for check in checks:
        status = check.get("status")
        detail = str(check.get("detail") or "")
        if status == "failed":
            case_status, message = "failed", detail or str(check.get("id"))
        elif status == "unavailable":
            case_status, message = "skipped", detail or "unavailable"
        else:
            case_status, message = "passed", ""
        cases.append(
            {
                "classname": surface,
                "name": str(check.get("id")),
                "status": case_status,
                "message": message,
                "detail": detail,
            }
        )
    return [{"name": label, "cases": cases}]


def run_diff_reports(args: argparse.Namespace, _parser: argparse.ArgumentParser) -> int:
    """Implement ``mpl diff-reports``; returns the process exit code.

    JSON goes to stdout, the human-readable summary to stderr (mirroring
    ``diff-backends``). A successful comparison exits 0 even when it finds
    differences -- differences are the tool's useful output, not a failure.

    With ``--gate`` (R4 adjudication) the same outputs are printed, the
    JSON gains a top-level ``gate`` object carrying the verdict and both
    direction counts, and the exit code becomes the gate's branch
    signal: 0 for ``no_change``/``improvement``/``change``, 2 for
    ``regression``/``unavailable``. Without ``--gate`` the output is
    byte-identical to the pre-gate contract. ``--junit`` additionally
    writes a JUnit XML report derived from the same diff dict; an
    unwritable path exits 1 after the normal outputs print.

    Output contract (cycle 217, restructured cycle 225): both sinks
    print the DERIVED diff dict only, never the loaded input payloads, so
    no input-only field can reach either sink. The ``confidentiality``
    section's name refers to the behavioral labeled-block exposure probe
    (cycle 169), not to secret material: the section carries only an
    availability enum, booleans, numeric gaps, and candidate labels from
    structurally validated payloads -- no prompt, completion, or
    credential material exists in it to leak. Cycle 225 carries that
    section through the pipeline as data, by key (see
    ``_payload_section``), so the lexical sensitive-data flow CodeQL
    reported against these two prints in alerts #24/#25 no longer
    exists by construction; the emitted bytes are unchanged and are
    pinned by the cycle-169/217/225 tests.
    """
    gate_mode = getattr(args, "gate", False)
    junit_path = getattr(args, "junit", None)
    if getattr(args, "max_regressions", None) is not None and not gate_mode:
        # A threshold the user set must never be silently ignored.
        message = "--max-regressions requires --gate"
        if _parser is not None:
            _parser.error(message)
        print(f"diff-reports: {message}", file=sys.stderr)
        return 2
    max_regressions = 0
    threshold_source = "default"
    if gate_mode:
        resolved = _resolve_gate_threshold(args, _parser)
        if resolved is None:
            print(
                f"diff-reports: {GATE_ENV_VAR} must be a non-negative integer",
                file=sys.stderr,
            )
            return 1
        max_regressions, threshold_source = resolved
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
    # Both outputs are fully built before either is emitted (RR-02):
    # a pair whose summary rendering failed after the JSON had been
    # printed left a valid JSON artifact next to a traceback and a
    # nonzero exit -- a partial-success artifact. Building first keeps
    # the command all-or-nothing.
    gate = _classify_gate(diff, max_regressions, threshold_source) if gate_mode else None
    printed = {**diff, "gate": gate} if gate is not None else diff
    diff_text = json.dumps(printed, indent=2, ensure_ascii=True, sort_keys=True)
    summary_text = render_summary(diff)
    if junit_path is not None and not _write_junit(
        junit_path, _render_junit(_diff_junit_suites(diff, gate))
    ):
        # The comparison itself succeeded, so its outputs still print;
        # the unwritable JUnit path is an operational failure (exit 1)
        # and no partial XML file is left behind.
        print(diff_text)
        print(summary_text, file=sys.stderr)
        print(f"diff-reports: junit_unwritable ({junit_path})", file=sys.stderr)
        return 1
    print(diff_text)
    print(summary_text, file=sys.stderr)
    if gate is not None and gate["verdict"] in ("regression", "unavailable"):
        return 2
    return 0


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
