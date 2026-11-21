"""experiments text family: split from experiments.py (W7-2).

Facades keep the public surface: import from model_protocol_lab.experiments, not from this module.
"""

from threading import Event
from typing import Any

from .core import Fixture, Message, evaluate
from .experiments_core import (
    _answer_digest,
    _backend_fingerprint,
    _resolve_boundaries,
)
from .wscanonical import aggregate_mutation_records, compare_id_sequences


def ws_canonicalization(
    native_outcomes: dict[str, dict[str, Any]],
    native_input_ids: dict[str, dict[str, list[int]] | None],
    ws_result: dict[str, Any],
    candidates: tuple[str, ...],
) -> dict[str, Any]:
    """Compare whitespace-variant runs against the native search outcomes.

    For every candidate and mutation: count id-sequence divergences
    (variant input ids vs the ids the native run actually sent) and
    verdict flips (variant exact-and-boundary verdict vs the native
    per-fixture verdict). A flipped verdict on *identical* ids is
    recorded separately as a determinism anomaly
    (``n_same_ids_verdict_flips``) -- it cannot be whitespace
    fragility, and counting it as such would be dishonest.

    Fail-closed: a candidate with no native outcomes, no native input
    ids, or a None variant cell degrades to ``whitespace_robust: None``
    with a ``detail`` code, never a phantom robust/​fragile verdict.
    Never raises on malformed data: returns ``{"available": False}``
    with a ``detail`` code.

    Returns ``{"available": True, "mutations": [...],
    "per_candidate": {name: {"per_mutation": ..., "aggregate": ...,
    "whitespace_robust": bool | None, "detail": str | None}},
    "fragile_candidates": [...], "excluded_candidates": [...]}``.
    """
    if type(native_outcomes) is not dict or type(native_input_ids) is not dict:
        return {"available": False, "detail": "native_data_malformed"}
    if type(ws_result) is not dict:
        return {"available": False, "detail": "ws_result_malformed"}
    mutations = ws_result.get("mutations")
    per_mutation = ws_result.get("per_mutation")
    if (
        type(mutations) is not list
        or not mutations
        or not all(type(m) is str for m in mutations)
        or type(per_mutation) is not dict
    ):
        return {"available": False, "detail": "ws_result_malformed"}
    per_candidate: dict[str, dict[str, Any]] = {}
    excluded: list[str] = []
    fragile: list[str] = []
    for cand in candidates:
        outcome = native_outcomes.get(cand)
        base_ids = native_input_ids.get(cand)
        if (
            type(outcome) is not dict
            or type(base_ids) is not dict
            or type(outcome.get("per_fixture")) is not dict
        ):
            excluded.append(cand)
            continue
        base_verdicts = outcome["per_fixture"]
        cell_records: list[dict[str, object]] = []
        mutation_cells: dict[str, dict[str, Any]] = {}
        detail: str | None = None
        for mutation in mutations:
            section = per_mutation.get(mutation)
            cell = section.get("per_candidate", {}).get(cand) if type(section) is dict else None
            if type(cell) is not dict:
                detail = "variant_cell_unavailable"
                mutation_cells[mutation] = {
                    "evaluable": False,
                    "detail": "variant_cell_unavailable",
                }
                continue
            variant_verdicts = cell.get("verdicts")
            variant_ids = cell.get("input_ids")
            if type(variant_verdicts) is not dict or type(variant_ids) is not dict:
                detail = "variant_cell_malformed"
                mutation_cells[mutation] = {
                    "evaluable": False,
                    "detail": "variant_cell_malformed",
                }
                continue
            fixture_ids = [fid for fid in base_verdicts if type(fid) is str]
            n_ids_divergent = 0
            n_verdict_flips = 0
            n_same_ids_flips = 0
            n_evaluable = 0
            flips: list[str] = []
            for fid in fixture_ids:
                old_ids = base_ids.get(fid)
                new_ids = variant_ids.get(fid)
                if type(old_ids) is not list or type(new_ids) is not list:
                    continue
                # Both sides validated as nonneg-int lists above; the
                # pure comparison keeps one id-equality semantic.
                ids_diverged = not compare_id_sequences(tuple(old_ids), tuple(new_ids))
                if ids_diverged:
                    n_ids_divergent += 1
                new_verdict = variant_verdicts.get(fid)
                old_verdict = base_verdicts.get(fid)
                if type(new_verdict) is not bool or type(old_verdict) is not bool:
                    continue
                n_evaluable += 1
                if new_verdict != old_verdict:
                    n_verdict_flips += 1
                    flips.append(fid)
                    if not ids_diverged:
                        # Same ids, different verdict: the backend is
                        # nondeterministic across runs -- a determinism
                        # finding, not whitespace fragility.
                        n_same_ids_flips += 1
            cell_records.append(
                {
                    "mutation": mutation,
                    "n_fixtures": len(fixture_ids),
                    "n_evaluable": n_evaluable,
                    "n_ids_divergent": n_ids_divergent,
                    "n_verdict_flips": n_verdict_flips,
                    "n_same_ids_verdict_flips": n_same_ids_flips,
                }
            )
            mutation_cells[mutation] = {
                "evaluable": True,
                "n_fixtures": len(fixture_ids),
                "n_evaluable": n_evaluable,
                "n_ids_divergent": n_ids_divergent,
                "ids_divergence_rate": (n_ids_divergent / len(fixture_ids)) if fixture_ids else 0.0,
                "n_verdict_flips": n_verdict_flips,
                "verdict_flip_rate": (n_verdict_flips / n_evaluable) if n_evaluable else 0.0,
                "n_same_ids_verdict_flips": n_same_ids_flips,
                "n_fragile_flips": n_verdict_flips - n_same_ids_flips,
                "flips": sorted(flips),
                # Fragile means a verdict flipped on *divergent* ids.
                # Same-ids flips are a backend-determinism finding and
                # must never count as whitespace fragility (see the
                # docstring's honest split).
                "fragile": (n_verdict_flips - n_same_ids_flips) > 0,
            }
        try:
            aggregate = aggregate_mutation_records(cell_records)
        except ValueError:
            return {"available": False, "detail": "aggregation_failed"}
        if detail is not None:
            per_candidate[cand] = {
                "per_mutation": mutation_cells,
                "aggregate": aggregate,
                "whitespace_robust": None,
                "detail": detail,
            }
            excluded.append(cand)
            continue
        robust = bool(aggregate["whitespace_robust"])
        per_candidate[cand] = {
            "per_mutation": mutation_cells,
            "aggregate": aggregate,
            "whitespace_robust": robust,
            "detail": None,
        }
        # Fragile means a verdict flipped on divergent ids --
        # divergent ids with stable verdicts, and same-ids flips
        # (a determinism finding), are reported per mutation but are
        # not fragility.
        if aggregate["n_fragile_mutations"]:
            fragile.append(cand)
    if not per_candidate:
        return {"available": False, "detail": "no_evaluable_candidates"}
    return {
        "available": True,
        "mutations": list(mutations),
        "per_candidate": per_candidate,
        "fragile_candidates": sorted(fragile),
        "excluded_candidates": sorted(excluded),
    }


class _CanonicalFormCancelled(Exception):
    """Internal: the run's cancel event fired during the
    canonical-form stage; the stage unwinds to its fail-closed
    payload rather than mislabelling the interruption.
    """


def _canonical_form_cell(
    backend: Any,
    fmt: Any,
    fixture: Any,
    ids: tuple[int, ...],
    *,
    boundary_ids: tuple[int, ...],
    max_new_tokens: int,
    max_context: int,
    seed: int,
    cancel: Event | None,
) -> tuple[str | None, dict[str, Any] | None]:
    """One canonical-form generation: (detail, cell).

    Exactly one of the pair is set. ``ids`` are the arm
    render's ids, measured by the stage at construction for
    the record's arm token counts; the cell's ``input_tokens``
    come from the evaluation's own observation, so a backend
    whose encode disagreed with itself surfaces as a validator
    rejection downstream, never a silently wrong record. The
    cancel event is checked before and after ``evaluate`` (the
    multiturn rule: a cancel observed during a generation is
    never mislabelled an evaluation failure).
    """
    if len(ids) + max_new_tokens > max_context:
        return "arm_context_exceeded", None
    if cancel is not None and cancel.is_set():
        raise _CanonicalFormCancelled
    report = evaluate(
        backend,
        fmt,
        (fixture,),
        boundary=boundary_ids,
        max_new_tokens=max_new_tokens,
        seed=seed,
        cancel=cancel,
    )
    if cancel is not None and cancel.is_set():
        raise _CanonicalFormCancelled
    if report.aggregate_eligible and report.observations:
        obs = report.observations[0]
        return None, {
            "verdict": bool(obs.exact_match and obs.boundary_found),
            "boundary_found": bool(obs.boundary_found),
            "boundary_text_match": obs.boundary_text_match,
            "input_tokens": len(obs.input_ids),
            "answer_digest": _answer_digest(obs.answer),
        }
    return "arm_evaluation_failed", None


def canonicalform_stage(
    backend: Any,
    formats: tuple[Any, ...],
    *,
    boundary: tuple[int, ...],
    boundary_map: dict[str, tuple[int, ...]] | None = None,
    max_new_tokens: int = 64,
    max_context: int,
    seed: int = 0,
    cancel: Event | None = None,
) -> dict[str, Any]:
    """Canonical-form stage (cycle 209): generated single-fact
    cases rendered under NFC and NFD per candidate.

    The cases are generated from the run seed inside the stage;
    there is no native arm and no dependence on the user's
    fixtures or the primary run. Each case contributes two
    cells (NFC, NFD) whose contents differ only in canonical
    form, plus the id-divergence measurement: the two arms'
    full render id sequences are compared at encode time, so
    the headline population (id-divergent paired cases) is
    decided before any generation is spent, and ids-identical
    cases are reported as the instrument-inapplicable share.
    Design: docs/CANONICALFORM_DESIGN.md.
    """
    from . import canonicalform as _cf
    from .fixtures import audit_special_token_literals

    if type(formats) is not tuple or not formats:
        raise ValueError("formats must be a nonempty tuple")
    names = [fmt.name for fmt in formats]
    if len(set(names)) != len(names):
        # Candidate records are keyed by name; two formats
        # sharing a name would silently collapse into one
        # candidate's records (red-team F7, cycle 193). The
        # precondition is stated at this stage's own boundary.
        raise ValueError("formats must have unique names")
    protocol: dict[str, Any] = dict(_cf.protocol_object(seed, _backend_fingerprint(backend)))
    protocol["fingerprint"] = _cf.protocol_fingerprint(seed, _backend_fingerprint(backend))
    cases = _cf.generate_cases(seed)
    if len(formats) * len(cases) > _cf.MAX_GROUND_RECORDS:
        return {
            "available": False,
            "detail": "canonical_form_record_budget_exceeded",
            "protocol": protocol,
            "candidates": {},
        }
    generated_text = (
        _cf.FACT_TEMPLATE.replace("{topic}", "").replace("{value}", "")
        + _cf.QUESTION_TEMPLATE.replace("{topic}", "")
        + "".join(_cf.TOPICS)
        + "".join(_cf.WORDS)
        + "".join(_cf.NEUTRAL_POOL)
    )
    if audit_special_token_literals(generated_text, formats):
        # A special-token literal in the frozen texts would
        # forge turn structure in every cell of every run; the
        # stage fails closed before any generation.
        return {
            "available": False,
            "detail": "canonical_form_screen_failed",
            "protocol": protocol,
            "candidates": {},
        }
    boundaries = _resolve_boundaries(formats, boundary, boundary_map)
    boundary_text: dict[str, str] = {}
    for fmt in formats:
        try:
            boundary_text[fmt.name] = backend.decode(boundaries[fmt.name])
        except Exception:  # noqa: BLE001 -- an undecodable boundary screens nothing
            boundary_text[fmt.name] = ""
    for fmt in formats:
        if boundary_text[fmt.name] and boundary_text[fmt.name] in generated_text:
            return {
                "available": False,
                "detail": "canonical_form_screen_failed",
                "protocol": protocol,
                "candidates": {},
            }

    def _cancelled() -> bool:
        """Return the cancelled."""
        return cancel is not None and cancel.is_set()

    candidates: dict[str, Any] = {}
    for fmt in formats:
        name = fmt.name
        records: list[dict[str, Any]] = []
        try:
            for case in cases:
                if _cancelled():
                    return {
                        "available": False,
                        "detail": "cancelled",
                        "protocol": protocol,
                        "candidates": {},
                    }
                case_id = str(case["fixture_id"])
                value = str(case["expected"])
                # Both arms' renders are encoded BEFORE any
                # generation: id divergence is an encode-time
                # measurement, and the headline population is
                # decided here, not inferred from verdicts.
                arm_ids: dict[str, tuple[int, ...]] = {}
                arm_contents: dict[str, str] = {}
                for arm in _cf.ARM_KEYS:
                    content = _cf.form_content(case, arm)
                    arm_contents[arm] = content
                    fixture = Fixture(
                        id=f"{case_id}-{arm}",
                        messages=(Message("user", content),),
                        expected=value,
                        group=_cf.GROUP,
                    )
                    arm_ids[arm] = tuple(backend.encode(fmt.render(fixture.messages)))
                ids_identical = arm_ids["nfc"] == arm_ids["nfd"]
                record: dict[str, Any] = {
                    "fixture": case_id,
                    "topic": str(case["topic"]),
                    "expected_digest": _cf.digest_text(value),
                    "nfc_tokens": len(arm_ids["nfc"]),
                    "nfd_tokens": len(arm_ids["nfd"]),
                    "ids_identical": ids_identical,
                    "id_length_delta": len(arm_ids["nfc"]) - len(arm_ids["nfd"]),
                    "nfc_content_digest": _cf.digest_text(arm_contents["nfc"]),
                    "nfd_content_digest": _cf.digest_text(arm_contents["nfd"]),
                    "cells": dict.fromkeys(_cf.ARM_KEYS),
                    "cell_details": dict.fromkeys(_cf.ARM_KEYS, ""),
                }
                records.append(record)
                for arm in _cf.ARM_KEYS:
                    if _cancelled():
                        raise _CanonicalFormCancelled
                    fixture = Fixture(
                        id=f"{case_id}-{arm}",
                        messages=(Message("user", arm_contents[arm]),),
                        expected=value,
                        group=_cf.GROUP,
                    )
                    detail, cell = _canonical_form_cell(
                        backend,
                        fmt,
                        fixture,
                        arm_ids[arm],
                        boundary_ids=boundaries[name],
                        max_new_tokens=max_new_tokens,
                        max_context=max_context,
                        seed=seed,
                        cancel=cancel,
                    )
                    if detail is not None:
                        record["cell_details"][arm] = detail
                    elif cell is not None:
                        record["cells"][arm] = cell
        except _CanonicalFormCancelled:
            return {
                "available": False,
                "detail": "cancelled",
                "protocol": protocol,
                "candidates": {},
            }
        candidates[name] = {"status": "available", "detail": "", "records": records}
    return {"available": True, "detail": "", "protocol": protocol, "candidates": candidates}


def canonicalform_analysis(
    stage: dict[str, Any],
    candidate_names: tuple[str, ...],
) -> dict[str, Any]:
    """Aggregate a canonical-form stage into the stored payload
    (cycle 209).

    Per-candidate readouts come from
    canonicalform.aggregate_candidate; the per-arm selections
    come from canonicalform.arm_selection over the shared
    population; the winner headline compares the two arms'
    selections. Design: docs/CANONICALFORM_DESIGN.md.
    """
    from . import canonicalform as _cf

    if not stage.get("available"):
        return {
            "available": False,
            "detail": str(stage.get("detail", "canonical_form_unavailable")),
            "protocol": stage.get("protocol"),
        }
    stage_candidates = stage["candidates"]
    candidates_out: dict[str, Any] = {}
    records_by_candidate: dict[str, list[dict[str, Any]]] = {}
    for name in candidate_names:
        entry = stage_candidates[name]
        if entry["status"] != "available":
            candidates_out[name] = {
                "status": "unavailable",
                "detail": entry["detail"],
                "records": [],
            }
            continue
        aggregates = _cf.aggregate_candidate(entry["records"])
        aggregates.update({"status": "available", "detail": "", "records": entry["records"]})
        candidates_out[name] = aggregates
        records_by_candidate[name] = entry["records"]
    conditions = {arm: _cf.arm_selection(records_by_candidate, arm) for arm in _cf.ARM_KEYS}
    return {
        "available": True,
        "detail": "",
        "protocol": stage["protocol"],
        "candidates": candidates_out,
        "conditions": conditions,
        "selection": {
            "nfc": conditions["nfc"],
            "nfd": conditions["nfd"],
            "winner": _cf.winner_headline(conditions),
        },
        "n_candidates": len(candidate_names),
        "n_cases": len(_cf.generate_cases(int(stage["protocol"]["seed"]))),
    }


class _RecordNotationCancelled(Exception):
    """Internal: the run's cancel event fired during the
    record-notation stage; the stage unwinds to its fail-closed
    payload rather than mislabelling the interruption.
    """


def _record_notation_cell(
    backend: Any,
    fmt: Any,
    fixture: Any,
    ids: tuple[int, ...],
    *,
    boundary_ids: tuple[int, ...],
    max_new_tokens: int,
    max_context: int,
    seed: int,
    cancel: Event | None,
    carry_text: bool,
) -> tuple[str | None, dict[str, Any] | None]:
    """One record-notation generation: (detail, cell).

    Exactly one of the pair is set. ``ids`` are the arm
    render's ids, measured by the stage at construction for
    the record's arm token counts; the cell's ``input_tokens``
    come from the evaluation's own observation, so a backend
    whose encode disagreed with itself surfaces as a validator
    rejection downstream, never a silently wrong record (the
    mirror binds the two, amendment D). Headline-arm cells
    carry ``answer_text`` -- the decoded pre-boundary answer,
    verbatim, never normalised -- bound by the cell's
    ``answer_digest`` (the cycle-190 pattern); screen-arm
    cells omit it, and no text-derived quantity is reported
    for the screen arm. A cell is built only from an
    aggregate-eligible report, whose invariants force the
    recorded observation's status to "completed", so the
    section 4.1 verdict recomputes from the digest and the
    boundary flag exactly (amendment D). The cancel event is
    checked before and after ``evaluate`` (the multiturn rule:
    a cancel observed during a generation is never mislabelled
    an evaluation failure).
    """
    if len(ids) + max_new_tokens > max_context:
        return "arm_context_exceeded", None
    if cancel is not None and cancel.is_set():
        raise _RecordNotationCancelled
    report = evaluate(
        backend,
        fmt,
        (fixture,),
        boundary=boundary_ids,
        max_new_tokens=max_new_tokens,
        seed=seed,
        cancel=cancel,
    )
    if cancel is not None and cancel.is_set():
        raise _RecordNotationCancelled
    if report.aggregate_eligible and report.observations:
        obs = report.observations[0]
        cell: dict[str, Any] = {
            "verdict": bool(obs.exact_match and obs.boundary_found),
            "boundary_found": bool(obs.boundary_found),
            "boundary_text_match": obs.boundary_text_match,
            "input_tokens": len(obs.input_ids),
            "answer_digest": _answer_digest(obs.answer),
        }
        if carry_text:
            cell["answer_text"] = obs.answer
        return None, cell
    return "arm_evaluation_failed", None


def record_notation_stage(
    backend: Any,
    formats: tuple[Any, ...],
    *,
    boundary: tuple[int, ...],
    boundary_map: dict[str, tuple[int, ...]] | None = None,
    max_new_tokens: int = 64,
    max_context: int,
    seed: int = 0,
    cancel: Event | None = None,
) -> dict[str, Any]:
    """Record-notation stage (cycle 221): a generated 24-case
    panel of code tables, each rendered in two frozen notations
    (prose sentences and canonical JSON) inside the user
    message, plus the closed-book screen arm, per candidate.

    The cases are generated from the run seed inside the stage;
    there is no dependence on the user's fixtures or the primary
    run. Surgery is at the fixture level: every arm of a case is
    a Fixture with the same id, group, and expected, so request
    seeds are identical across arms by construction
    (``request_seed_for`` is id-derived) and ``core.evaluate``
    is used unchanged. The pre-generation screens (the per-case
    invariant battery, the special-token-literal audit over
    every arm's full message content for every family, and
    boundary-text containment) fail the whole stage closed with
    ``record_notation_screen_failed``; their counts travel in
    the payload's ``audit`` block, never silently dropped. The
    serialised-records byte cap degrades the whole stage to
    ``record_notation_record_budget_exceeded``. Design:
    docs/PAYLOADSERIALIZATION_DESIGN.md.
    """
    from . import recordnotation as _rn
    from .fixtures import audit_special_token_literals

    if type(formats) is not tuple or not formats:
        raise ValueError("formats must be a nonempty tuple")
    names = [fmt.name for fmt in formats]
    if len(set(names)) != len(names):
        # Candidate records are keyed by name; two formats
        # sharing a name would silently collapse into one
        # candidate's records (red-team F7, cycle 193). The
        # precondition is stated at this stage's own boundary.
        raise ValueError("formats must have unique names")
    protocol: dict[str, Any] = dict(_rn.protocol_object(seed, _backend_fingerprint(backend)))
    protocol["fingerprint"] = _rn.protocol_fingerprint(seed, _backend_fingerprint(backend))
    cases = _rn.generate_cases(seed)
    audit: dict[str, Any] = {
        "n_audit_excluded": 0,
        "by_reason": {},
        # nosec B105 -- integer audit counter field, not a credential
        "special_token_findings": 0,
    }
    if len(formats) * len(cases) > _rn.MAX_GROUND_RECORDS:
        return {
            "available": False,
            "detail": "record_notation_record_budget_exceeded",
            "protocol": protocol,
            "audit": audit,
            "candidates": {},
        }
    invalid = sum(1 for case in cases if not _rn.validate_case(case, seed))
    if invalid:
        audit = {
            "n_audit_excluded": 0,
            "by_reason": {"case_invariant": invalid},
            # nosec B105 -- integer audit counter field, not a credential
            "special_token_findings": 0,
        }
        return {
            "available": False,
            "detail": "record_notation_screen_failed",
            "protocol": protocol,
            "audit": audit,
            "candidates": {},
        }
    contents = tuple(_rn.arm_content(case, arm) for case in cases for arm in _rn.ARM_KEYS)
    findings = audit_special_token_literals(contents, formats)
    if findings:
        # A special-token literal in an arm's content would
        # forge turn structure in every cell of every run; the
        # stage fails closed before any generation, with the
        # count in the audit block (never silently dropped).
        audit = {
            "n_audit_excluded": 0,
            "by_reason": {"special_token_literal": len(findings)},
            "special_token_findings": len(findings),
        }
        return {
            "available": False,
            "detail": "record_notation_screen_failed",
            "protocol": protocol,
            "audit": audit,
            "candidates": {},
        }
    boundaries = _resolve_boundaries(formats, boundary, boundary_map)
    for fmt in formats:
        try:
            boundary_text = backend.decode(boundaries[fmt.name])
        except Exception:  # noqa: BLE001 -- an undecodable boundary screens nothing
            boundary_text = ""
        if boundary_text and any(boundary_text in content for content in contents):
            return {
                "available": False,
                "detail": "record_notation_screen_failed",
                "protocol": protocol,
                "audit": audit,
                "candidates": {},
            }

    def _cancelled() -> bool:
        """Return the cancelled."""
        return cancel is not None and cancel.is_set()

    candidates: dict[str, Any] = {}
    for fmt in formats:
        name = fmt.name
        records: list[dict[str, Any]] = []
        try:
            for case in cases:
                if _cancelled():
                    raise _RecordNotationCancelled
                case_id = str(case["fixture_id"])
                n_records, fields = _rn.table_shape(int(case_id.rsplit("-", 1)[1]))
                record: dict[str, Any] = {
                    "fixture": case_id,
                    "n_records": n_records,
                    "n_fields": len(fields),
                    "expected_digest": _rn.digest_text(str(case["expected"])),
                    "content_digests": dict.fromkeys(_rn.ARM_KEYS),
                    "arm_tokens": dict.fromkeys(_rn.ARM_KEYS),
                    "id_digests": dict.fromkeys(_rn.ARM_KEYS),
                    "ids_identical": False,
                    "id_length_delta": 0,
                    "cells": dict.fromkeys(_rn.ARM_KEYS),
                    "cell_details": dict.fromkeys(_rn.ARM_KEYS, ""),
                }
                for arm in _rn.ARM_KEYS:
                    content = _rn.arm_content(case, arm)
                    fixture = Fixture(
                        id=case_id,
                        messages=(Message("user", content),),
                        expected=str(case["expected"]),
                        group=_rn.GROUP,
                    )
                    ids = tuple(backend.encode(fmt.render(fixture.messages)))
                    record["arm_tokens"][arm] = len(ids)
                    record["id_digests"][arm] = _rn.digest_ids(ids)
                    record["content_digests"][arm] = _rn.digest_text(content)
                    detail, cell = _record_notation_cell(
                        backend,
                        fmt,
                        fixture,
                        ids,
                        boundary_ids=boundaries[name],
                        max_new_tokens=max_new_tokens,
                        max_context=max_context,
                        seed=seed,
                        cancel=cancel,
                        carry_text=arm in _rn.HEADLINE_ARMS,
                    )
                    if detail is not None:
                        record["cell_details"][arm] = detail
                    elif cell is not None:
                        record["cells"][arm] = cell
                record["ids_identical"] = (
                    record["id_digests"]["prose"] == record["id_digests"]["canonical_json"]
                )
                record["id_length_delta"] = int(record["arm_tokens"]["canonical_json"]) - int(
                    record["arm_tokens"]["prose"]
                )
                records.append(record)
        except _RecordNotationCancelled:
            return {
                "available": False,
                "detail": "cancelled",
                "protocol": protocol,
                "audit": audit,
                "candidates": {},
            }
        candidates[name] = {"status": "available", "detail": "", "records": records}
    if _rn.records_bytes({name: entry["records"] for name, entry in candidates.items()}) > (
        _rn.MAX_RECORDS_BYTES
    ):
        # The byte cap is a hard bound on what a results file
        # may carry (a non-ASCII-stressed panel of answer texts
        # reaches it; design Part I section 8): the whole stage
        # degrades to the frozen budget detail, never a partial
        # panel.
        return {
            "available": False,
            "detail": "record_notation_record_budget_exceeded",
            "protocol": protocol,
            "audit": audit,
            "candidates": {},
        }
    return {
        "available": True,
        "detail": "",
        "protocol": protocol,
        "audit": audit,
        "candidates": candidates,
    }


def record_notation_analysis(
    stage: dict[str, Any],
    candidate_names: tuple[str, ...],
) -> dict[str, Any]:
    """Aggregate a record-notation stage into the stored
    payload (cycle 221).

    Per-candidate readouts come from
    recordnotation.aggregate_candidate over the candidate's
    ground records and the regenerated panel: the audit counts,
    per-arm rates with the section 4.4 diagnostics, the
    headline paired discordance with its decomposition and
    resolution, the id report, and the sensitivities. The
    instrument is descriptive: there is no classification, no
    selection, and no winner readout (design Part I sections 6.1
    and 6.3). Design: docs/PAYLOADSERIALIZATION_DESIGN.md.
    """
    from . import recordnotation as _rn

    if not stage.get("available"):
        return {
            "available": False,
            "detail": str(stage.get("detail", "record_notation_unavailable")),
            "protocol": stage.get("protocol"),
            "audit": stage.get("audit"),
        }
    stage_candidates = stage["candidates"]
    cases = _rn.generate_cases(int(stage["protocol"]["seed"]))
    candidates_out: dict[str, Any] = {}
    for name in candidate_names:
        entry = stage_candidates[name]
        if entry["status"] != "available":
            candidates_out[name] = {
                "status": "unavailable",
                "detail": entry["detail"],
                "records": [],
            }
            continue
        aggregates = _rn.aggregate_candidate(entry["records"], cases)
        aggregates.update({"status": "available", "detail": "", "records": entry["records"]})
        candidates_out[name] = aggregates
    return {
        "available": True,
        "detail": "",
        "protocol": stage["protocol"],
        "audit": stage.get("audit"),
        "candidates": candidates_out,
        "n_candidates": len(candidate_names),
        "n_cases": len(cases),
    }


class _InvisibleCharsCancelled(Exception):
    """Internal: the run's cancel event fired during the
    invisible-chars stage; the stage unwinds to its fail-closed
    payload rather than mislabelling the interruption.
    """


def _invisible_chars_cell(
    backend: Any,
    fmt: Any,
    fixture: Any,
    ids: tuple[int, ...],
    *,
    boundary_ids: tuple[int, ...],
    max_new_tokens: int,
    max_context: int,
    seed: int,
    cancel: Event | None,
) -> tuple[str | None, dict[str, Any] | None]:
    """One invisible-chars generation: (detail, cell).

    Exactly one of the pair is set. ``ids`` are the arm
    render's ids, measured by the stage at construction for
    the record's arm token counts and ids-identical
    derivations; the cell's ``input_tokens`` come from the
    evaluation's own observation, so a backend whose encode
    disagreed with itself surfaces as a validator rejection
    downstream, never a silently wrong record (the mirror
    binds the two). Every cell carries ``answer_text`` -- the
    decoded pre-boundary answer, verbatim, never normalised
    -- bound by the cell's ``answer_digest`` (the cycle-190
    pattern), because the decomposition classes are functions
    of the answer text. A cell is built only from an
    aggregate-eligible report, whose invariants force the
    recorded observation's status to "completed", so the
    verdict recomputes from the digest and the boundary flag
    exactly. The cancel event is checked before and after
    ``evaluate`` (the multiturn rule: a cancel observed during
    a generation is never mislabelled an evaluation failure).
    """
    if len(ids) + max_new_tokens > max_context:
        return "arm_context_exceeded", None
    if cancel is not None and cancel.is_set():
        raise _InvisibleCharsCancelled
    report = evaluate(
        backend,
        fmt,
        (fixture,),
        boundary=boundary_ids,
        max_new_tokens=max_new_tokens,
        seed=seed,
        cancel=cancel,
    )
    if cancel is not None and cancel.is_set():
        raise _InvisibleCharsCancelled
    if report.aggregate_eligible and report.observations:
        obs = report.observations[0]
        return None, {
            "verdict": bool(obs.exact_match and obs.boundary_found),
            "boundary_found": bool(obs.boundary_found),
            "boundary_text_match": obs.boundary_text_match,
            "input_tokens": len(obs.input_ids),
            "answer_digest": _answer_digest(obs.answer),
            "answer_text": obs.answer,
        }
    return "arm_evaluation_failed", None


def invisible_chars_stage(
    backend: Any,
    formats: tuple[Any, ...],
    *,
    boundary: tuple[int, ...],
    boundary_map: dict[str, tuple[int, ...]] | None = None,
    max_new_tokens: int = 64,
    max_context: int,
    seed: int = 0,
    cancel: Event | None = None,
) -> dict[str, Any]:
    """Invisible-chars stage (cycle 223): a generated 24-case
    fact-recall panel, each case rendered verbatim and with
    one frozen invisible code point (or the matched visible
    hyphen) inserted at two frozen sites in its fact sentence,
    per candidate.

    The cases are generated from the run seed inside the stage;
    there is no dependence on the user's fixtures or the primary
    run. Surgery is at the fixture level: every arm of a case is
    a Fixture with the same id, group, and expected, so request
    seeds are identical across arms by construction
    (``request_seed_for`` is id-derived) and ``core.evaluate``
    is used unchanged. The pre-generation screens (the per-case
    invariant battery, the arm-population audit, the
    special-token-literal audit over every arm's full message
    content for every family, and boundary-text containment)
    fail the whole stage closed with
    ``invisible_chars_screen_failed``; their counts travel in
    the payload's ``audit`` block, never silently dropped. The
    serialised-records byte cap degrades the whole stage to
    ``invisible_chars_record_budget_exceeded``. Design:
    docs/INVISIBLECHARS_DESIGN.md.
    """
    from . import invisiblechars as _ic
    from .fixtures import audit_special_token_literals

    if type(formats) is not tuple or not formats:
        raise ValueError("formats must be a nonempty tuple")
    names = [fmt.name for fmt in formats]
    if len(set(names)) != len(names):
        # Candidate records are keyed by name; two formats
        # sharing a name would silently collapse into one
        # candidate's records (red-team F7, cycle 193). The
        # precondition is stated at this stage's own boundary.
        raise ValueError("formats must have unique names")
    try:
        protocol: dict[str, Any] = dict(_ic.protocol_object(seed, _backend_fingerprint(backend)))
        protocol["fingerprint"] = _ic.protocol_fingerprint(seed, _backend_fingerprint(backend))
    except _ic.RedrawExhausted:
        # The protocol's ground hash itself regenerates the
        # panel, so exhaustion can surface here first; it is
        # the same fail-closed screen outcome, with no
        # protocol to report.
        return {
            "available": False,
            "detail": "invisible_chars_screen_failed",
            "protocol": {},
            "audit": {
                "n_audit_excluded": 0,
                "by_reason": {"redraw_exhausted": 1},
                # nosec B105 -- integer audit counter field, not a credential
                "special_token_findings": 0,
            },
            "candidates": {},
        }
    audit: dict[str, Any] = {
        "n_audit_excluded": 0,
        "by_reason": {},
        # nosec B105 -- integer audit counter field, not a credential
        "special_token_findings": 0,
    }
    if len(formats) * _ic.CASE_COUNT > _ic.MAX_GROUND_RECORDS:
        return {
            "available": False,
            "detail": "invisible_chars_record_budget_exceeded",
            "protocol": protocol,
            "audit": audit,
            "candidates": {},
        }
    try:
        cases = _ic.generate_cases(seed)
    except _ic.RedrawExhausted:
        return {
            "available": False,
            "detail": "invisible_chars_screen_failed",
            "protocol": protocol,
            "audit": {
                "n_audit_excluded": 0,
                "by_reason": {"redraw_exhausted": 1},
                # nosec B105 -- integer audit counter field, not a credential
                "special_token_findings": 0,
            },
            "candidates": {},
        }
    invalid = sum(1 for case in cases if not _ic.validate_case(case, seed))
    if invalid or not _ic.POPULATION_AUDIT["ok"]:
        by_reason: dict[str, int] = {}
        if invalid:
            by_reason["case_invariant"] = invalid
        if not _ic.POPULATION_AUDIT["ok"]:
            by_reason["arm_population"] = 1
        return {
            "available": False,
            "detail": "invisible_chars_screen_failed",
            "protocol": protocol,
            "audit": {
                "n_audit_excluded": 0,
                "by_reason": by_reason,
                # nosec B105 -- integer audit counter field, not a credential
                "special_token_findings": 0,
            },
            "candidates": {},
        }
    contents = tuple(_ic.arm_content(case, arm) for case in cases for arm in _ic.ARM_KEYS)
    findings = audit_special_token_literals(contents, formats)
    if findings:
        # A special-token literal in an arm's content would
        # forge turn structure in every cell of every run; the
        # stage fails closed before any generation, with the
        # count in the audit block (never silently dropped).
        audit = {
            "n_audit_excluded": 0,
            "by_reason": {"special_token_literal": len(findings)},
            "special_token_findings": len(findings),
        }
        return {
            "available": False,
            "detail": "invisible_chars_screen_failed",
            "protocol": protocol,
            "audit": audit,
            "candidates": {},
        }
    boundaries = _resolve_boundaries(formats, boundary, boundary_map)
    for fmt in formats:
        try:
            boundary_text = backend.decode(boundaries[fmt.name])
        except Exception:  # noqa: BLE001 -- an undecodable boundary screens nothing
            boundary_text = ""
        if boundary_text and any(boundary_text in content for content in contents):
            return {
                "available": False,
                "detail": "invisible_chars_screen_failed",
                "protocol": protocol,
                "audit": audit,
                "candidates": {},
            }

    def _cancelled() -> bool:
        """Return the cancelled."""
        return cancel is not None and cancel.is_set()

    candidates: dict[str, Any] = {}
    for fmt in formats:
        name = fmt.name
        records: list[dict[str, Any]] = []
        try:
            for case in cases:
                if _cancelled():
                    raise _InvisibleCharsCancelled
                case_id = str(case["fixture_id"])
                record: dict[str, Any] = {
                    "fixture": case_id,
                    "topic": str(case["topic"]),
                    "expected_digest": _ic.digest_text(str(case["expected"])),
                    "content_digests": dict.fromkeys(_ic.ARM_KEYS),
                    "arm_tokens": dict.fromkeys(_ic.ARM_KEYS),
                    "id_digests": dict.fromkeys(_ic.ARM_KEYS),
                    "ids_identical": dict.fromkeys(_ic.TREATED_ARMS),
                    "id_length_delta": dict.fromkeys(_ic.TREATED_ARMS),
                    "cells": dict.fromkeys(_ic.ARM_KEYS),
                    "cell_details": dict.fromkeys(_ic.ARM_KEYS, ""),
                }
                for arm in _ic.ARM_KEYS:
                    content = _ic.arm_content(case, arm)
                    fixture = Fixture(
                        id=case_id,
                        messages=(Message("user", content),),
                        expected=str(case["expected"]),
                        group=_ic.GROUP,
                    )
                    ids = tuple(backend.encode(fmt.render(fixture.messages)))
                    record["arm_tokens"][arm] = len(ids)
                    record["id_digests"][arm] = _ic.digest_ids(ids)
                    record["content_digests"][arm] = _ic.digest_text(content)
                    detail, cell = _invisible_chars_cell(
                        backend,
                        fmt,
                        fixture,
                        ids,
                        boundary_ids=boundaries[name],
                        max_new_tokens=max_new_tokens,
                        max_context=max_context,
                        seed=seed,
                        cancel=cancel,
                    )
                    if detail is not None:
                        record["cell_details"][arm] = detail
                    elif cell is not None:
                        record["cells"][arm] = cell
                for arm in _ic.TREATED_ARMS:
                    record["ids_identical"][arm] = (
                        record["id_digests"][arm] == record["id_digests"]["control"]
                    )
                    record["id_length_delta"][arm] = int(record["arm_tokens"][arm]) - int(
                        record["arm_tokens"]["control"]
                    )
                records.append(record)
        except _InvisibleCharsCancelled:
            return {
                "available": False,
                "detail": "cancelled",
                "protocol": protocol,
                "audit": audit,
                "candidates": {},
            }
        candidates[name] = {"status": "available", "detail": "", "records": records}
    if _ic.records_bytes({name: entry["records"] for name, entry in candidates.items()}) > (
        _ic.MAX_RECORDS_BYTES
    ):
        # The byte cap is a hard bound on what a results file
        # may carry (a non-ASCII-stressed panel of answer texts
        # reaches it; design section 6 budgets): the whole stage
        # degrades to the frozen budget detail, never a partial
        # panel.
        return {
            "available": False,
            "detail": "invisible_chars_record_budget_exceeded",
            "protocol": protocol,
            "audit": audit,
            "candidates": {},
        }
    return {
        "available": True,
        "detail": "",
        "protocol": protocol,
        "audit": audit,
        "candidates": candidates,
    }


def invisible_chars_analysis(
    stage: dict[str, Any],
    candidate_names: tuple[str, ...],
) -> dict[str, Any]:
    """Aggregate an invisible-chars stage into the stored
    payload (cycle 223).

    Per-candidate readouts come from
    invisiblechars.aggregate_candidate over the candidate's
    ground records and the regenerated panel: per-arm rates
    with the six-class decomposition, the per-arm id report
    (the regime gate), and per treated arm the guarded
    headline over the id-divergent paired population, with
    its label, counts, and decomposition. The stage never
    affects selection. Design: docs/INVISIBLECHARS_DESIGN.md.
    """
    from . import invisiblechars as _ic

    if not stage.get("available"):
        return {
            "available": False,
            "detail": str(stage.get("detail", "invisible_chars_unavailable")),
            "protocol": stage.get("protocol"),
            "audit": stage.get("audit"),
        }
    stage_candidates = stage["candidates"]
    cases = _ic.generate_cases(int(stage["protocol"]["seed"]))
    candidates_out: dict[str, Any] = {}
    for name in candidate_names:
        entry = stage_candidates[name]
        if entry["status"] != "available":
            candidates_out[name] = {
                "status": "unavailable",
                "detail": entry["detail"],
                "records": [],
            }
            continue
        aggregates = _ic.aggregate_candidate(entry["records"], cases)
        aggregates.update({"status": "available", "detail": "", "records": entry["records"]})
        candidates_out[name] = aggregates
    return {
        "available": True,
        "detail": "",
        "protocol": stage["protocol"],
        "audit": stage.get("audit"),
        "candidates": candidates_out,
        "n_candidates": len(candidate_names),
        "n_cases": len(cases),
    }


class _ConfusableSkeletonCancelled(Exception):
    """Internal: the run's cancel event fired during the
    confusable-skeleton stage; the stage unwinds to its
    fail-closed payload rather than mislabelling the
    interruption.
    """


def _confusable_skeleton_cell(
    backend: Any,
    fmt: Any,
    fixture: Any,
    ids: tuple[int, ...],
    *,
    boundary_ids: tuple[int, ...],
    max_new_tokens: int,
    max_context: int,
    seed: int,
    cancel: Event | None,
) -> tuple[str | None, dict[str, Any] | None]:
    """One confusable-skeleton generation: (detail, cell).

    Exactly one of the pair is set. ``ids`` are the arm
    render's ids, measured by the stage at construction for
    the record's arm token counts and ids-identical
    derivations; the cell's ``input_tokens`` come from the
    evaluation's own observation, so a backend whose encode
    disagreed with itself surfaces as a validator rejection
    downstream, never a silently wrong record (the mirror
    binds the two). Every cell carries ``answer_text`` -- the
    decoded pre-boundary answer, verbatim, never normalised
    -- bound by the cell's ``answer_digest`` (the cycle-190
    pattern), because the decomposition classes are functions
    of the answer text. A cell is built only from an
    aggregate-eligible report, whose invariants force the
    recorded observation's status to "completed", so the
    verdict recomputes from the digest and the boundary flag
    exactly. The cancel event is checked before and after
    ``evaluate`` (the multiturn rule: a cancel observed during
    a generation is never mislabelled an evaluation failure).
    """
    if len(ids) + max_new_tokens > max_context:
        return "arm_context_exceeded", None
    if cancel is not None and cancel.is_set():
        raise _ConfusableSkeletonCancelled
    report = evaluate(
        backend,
        fmt,
        (fixture,),
        boundary=boundary_ids,
        max_new_tokens=max_new_tokens,
        seed=seed,
        cancel=cancel,
    )
    if cancel is not None and cancel.is_set():
        raise _ConfusableSkeletonCancelled
    if report.aggregate_eligible and report.observations:
        obs = report.observations[0]
        return None, {
            "verdict": bool(obs.exact_match and obs.boundary_found),
            "boundary_found": bool(obs.boundary_found),
            "boundary_text_match": obs.boundary_text_match,
            "input_tokens": len(obs.input_ids),
            "answer_digest": _answer_digest(obs.answer),
            "answer_text": obs.answer,
        }
    return "arm_evaluation_failed", None


def confusable_skeleton_stage(
    backend: Any,
    formats: tuple[Any, ...],
    *,
    boundary: tuple[int, ...],
    boundary_map: dict[str, tuple[int, ...]] | None = None,
    max_new_tokens: int = 64,
    max_context: int,
    seed: int = 0,
    cancel: Event | None = None,
) -> dict[str, Any]:
    """Confusable-skeleton stage (cycle 224): a generated
    24-case fact-recall panel, each case rendered verbatim and
    with frozen UTS #39 skeleton-equivalent Cyrillic/Greek
    letters substituted into its fact sentence (single-site,
    whole-word, and a same-script rarity control at the
    single-site sites), per candidate.

    The cases are generated from the run seed inside the stage;
    there is no dependence on the user's fixtures or the primary
    run. Surgery is at the fixture level: every arm of a case is
    a Fixture with the same id, group, and expected, so request
    seeds are identical across arms by construction
    (``request_seed_for`` is id-derived) and ``core.evaluate``
    is used unchanged. The pre-generation screens (the per-case
    invariant battery, the population audit, the
    special-token-literal audit over every arm's full message
    content for every family, and boundary-text containment)
    fail the whole stage closed with
    ``confusable_skeleton_screen_failed``; their counts travel
    in the payload's ``audit`` block, never silently dropped.
    The serialised-records byte cap degrades the whole stage to
    ``confusable_skeleton_record_budget_exceeded``. Design:
    docs/CONFUSABLESKELETON_DESIGN.md.
    """
    from . import confusableskeleton as _cs
    from .fixtures import audit_special_token_literals

    if type(formats) is not tuple or not formats:
        raise ValueError("formats must be a nonempty tuple")
    names = [fmt.name for fmt in formats]
    if len(set(names)) != len(names):
        # Candidate records are keyed by name; two formats
        # sharing a name would silently collapse into one
        # candidate's records (red-team F7, cycle 193). The
        # precondition is stated at this stage's own boundary.
        raise ValueError("formats must have unique names")
    try:
        protocol: dict[str, Any] = dict(_cs.protocol_object(seed, _backend_fingerprint(backend)))
        protocol["fingerprint"] = _cs.protocol_fingerprint(seed, _backend_fingerprint(backend))
    except _cs.RedrawExhausted:
        # The protocol's ground hash itself regenerates the
        # panel, so exhaustion can surface here first; it is
        # the same fail-closed screen outcome, with no
        # protocol to report.
        return {
            "available": False,
            "detail": "confusable_skeleton_screen_failed",
            "protocol": {},
            "audit": {
                "n_audit_excluded": 0,
                "by_reason": {"redraw_exhausted": 1},
                # nosec B105 -- integer audit counter field, not a credential
                "special_token_findings": 0,
            },
            "candidates": {},
        }
    audit: dict[str, Any] = {
        "n_audit_excluded": 0,
        "by_reason": {},
        # nosec B105 -- integer audit counter field, not a credential
        "special_token_findings": 0,
    }
    if len(formats) * _cs.CASE_COUNT > _cs.MAX_GROUND_RECORDS:
        return {
            "available": False,
            "detail": "confusable_skeleton_record_budget_exceeded",
            "protocol": protocol,
            "audit": audit,
            "candidates": {},
        }
    try:
        cases = _cs.generate_cases(seed)
    except _cs.RedrawExhausted:
        return {
            "available": False,
            "detail": "confusable_skeleton_screen_failed",
            "protocol": protocol,
            "audit": {
                "n_audit_excluded": 0,
                "by_reason": {"redraw_exhausted": 1},
                # nosec B105 -- integer audit counter field, not a credential
                "special_token_findings": 0,
            },
            "candidates": {},
        }
    invalid = sum(1 for case in cases if not _cs.validate_case(case, seed))
    if invalid or not _cs.POPULATION_AUDIT["ok"]:
        by_reason: dict[str, int] = {}
        if invalid:
            by_reason["case_invariant"] = invalid
        if not _cs.POPULATION_AUDIT["ok"]:
            by_reason["arm_population"] = 1
        return {
            "available": False,
            "detail": "confusable_skeleton_screen_failed",
            "protocol": protocol,
            "audit": {
                "n_audit_excluded": 0,
                "by_reason": by_reason,
                # nosec B105 -- integer audit counter field, not a credential
                "special_token_findings": 0,
            },
            "candidates": {},
        }
    contents = tuple(_cs.arm_content(case, arm) for case in cases for arm in _cs.ARM_KEYS)
    findings = audit_special_token_literals(contents, formats)
    if findings:
        # A special-token literal in an arm's content would
        # forge turn structure in every cell of every run; the
        # stage fails closed before any generation, with the
        # count in the audit block (never silently dropped).
        audit = {
            "n_audit_excluded": 0,
            "by_reason": {"special_token_literal": len(findings)},
            "special_token_findings": len(findings),
        }
        return {
            "available": False,
            "detail": "confusable_skeleton_screen_failed",
            "protocol": protocol,
            "audit": audit,
            "candidates": {},
        }
    boundaries = _resolve_boundaries(formats, boundary, boundary_map)
    for fmt in formats:
        try:
            boundary_text = backend.decode(boundaries[fmt.name])
        except Exception:  # noqa: BLE001 -- an undecodable boundary screens nothing
            boundary_text = ""
        if boundary_text and any(boundary_text in content for content in contents):
            return {
                "available": False,
                "detail": "confusable_skeleton_screen_failed",
                "protocol": protocol,
                "audit": audit,
                "candidates": {},
            }

    def _cancelled() -> bool:
        """Return the cancelled."""
        return cancel is not None and cancel.is_set()

    candidates: dict[str, Any] = {}
    for fmt in formats:
        name = fmt.name
        records: list[dict[str, Any]] = []
        try:
            for case in cases:
                if _cancelled():
                    raise _ConfusableSkeletonCancelled
                case_id = str(case["fixture_id"])
                record: dict[str, Any] = {
                    "fixture": case_id,
                    "topic": str(case["topic"]),
                    "expected_digest": _cs.digest_text(str(case["expected"])),
                    "content_digests": dict.fromkeys(_cs.ARM_KEYS),
                    "arm_tokens": dict.fromkeys(_cs.ARM_KEYS),
                    "id_digests": dict.fromkeys(_cs.ARM_KEYS),
                    "ids_identical": dict.fromkeys(_cs.TREATED_ARMS),
                    "id_length_delta": dict.fromkeys(_cs.TREATED_ARMS),
                    "table_pin": {
                        "version": _cs.CONFUSABLES_VERSION,
                        "table_sha256": _cs.table_sha256(),
                    },
                    "cells": dict.fromkeys(_cs.ARM_KEYS),
                    "cell_details": dict.fromkeys(_cs.ARM_KEYS, ""),
                }
                for arm in _cs.ARM_KEYS:
                    content = _cs.arm_content(case, arm)
                    fixture = Fixture(
                        id=case_id,
                        messages=(Message("user", content),),
                        expected=str(case["expected"]),
                        group=_cs.GROUP,
                    )
                    ids = tuple(backend.encode(fmt.render(fixture.messages)))
                    record["arm_tokens"][arm] = len(ids)
                    record["id_digests"][arm] = _cs.digest_ids(ids)
                    record["content_digests"][arm] = _cs.digest_text(content)
                    detail, cell = _confusable_skeleton_cell(
                        backend,
                        fmt,
                        fixture,
                        ids,
                        boundary_ids=boundaries[name],
                        max_new_tokens=max_new_tokens,
                        max_context=max_context,
                        seed=seed,
                        cancel=cancel,
                    )
                    if detail is not None:
                        record["cell_details"][arm] = detail
                    elif cell is not None:
                        record["cells"][arm] = cell
                for arm in _cs.TREATED_ARMS:
                    record["ids_identical"][arm] = (
                        record["id_digests"][arm] == record["id_digests"]["control"]
                    )
                    record["id_length_delta"][arm] = int(record["arm_tokens"][arm]) - int(
                        record["arm_tokens"]["control"]
                    )
                records.append(record)
        except _ConfusableSkeletonCancelled:
            return {
                "available": False,
                "detail": "cancelled",
                "protocol": protocol,
                "audit": audit,
                "candidates": {},
            }
        candidates[name] = {"status": "available", "detail": "", "records": records}
    if _cs.records_bytes({name: entry["records"] for name, entry in candidates.items()}) > (
        _cs.MAX_RECORDS_BYTES
    ):
        # The byte cap is a hard bound on what a results file
        # may carry (a non-ASCII-stressed panel of answer texts
        # reaches it; design section 6 budgets): the whole stage
        # degrades to the frozen budget detail, never a partial
        # panel.
        return {
            "available": False,
            "detail": "confusable_skeleton_record_budget_exceeded",
            "protocol": protocol,
            "audit": audit,
            "candidates": {},
        }
    return {
        "available": True,
        "detail": "",
        "protocol": protocol,
        "audit": audit,
        "candidates": candidates,
    }


def confusable_skeleton_analysis(
    stage: dict[str, Any],
    candidate_names: tuple[str, ...],
) -> dict[str, Any]:
    """Aggregate a confusable-skeleton stage into the stored
    payload (cycle 224).

    Per-candidate readouts come from
    confusableskeleton.aggregate_candidate over the candidate's
    ground records and the regenerated panel: per-arm rates
    with the seven-class decomposition, the per-arm id report
    (the regime gate), the whole-word dose, and per treated
    arm the guarded headline over the id-divergent paired
    population, with its label, counts, and decomposition.
    The stage never affects selection. Design:
    docs/CONFUSABLESKELETON_DESIGN.md.
    """
    from . import confusableskeleton as _cs

    if not stage.get("available"):
        return {
            "available": False,
            "detail": str(stage.get("detail", "confusable_skeleton_unavailable")),
            "protocol": stage.get("protocol"),
            "audit": stage.get("audit"),
        }
    stage_candidates = stage["candidates"]
    cases = _cs.generate_cases(int(stage["protocol"]["seed"]))
    candidates_out: dict[str, Any] = {}
    for name in candidate_names:
        entry = stage_candidates[name]
        if entry["status"] != "available":
            candidates_out[name] = {
                "status": "unavailable",
                "detail": entry["detail"],
                "records": [],
            }
            continue
        aggregates = _cs.aggregate_candidate(entry["records"], cases)
        aggregates.update({"status": "available", "detail": "", "records": entry["records"]})
        candidates_out[name] = aggregates
    return {
        "available": True,
        "detail": "",
        "protocol": stage["protocol"],
        "audit": stage.get("audit"),
        "candidates": candidates_out,
        "n_candidates": len(candidate_names),
        "n_cases": len(cases),
    }


class _CompatibilityFormCancelled(Exception):
    """Internal: the run's cancel event fired during the
    compatibility-form stage; the stage unwinds to its
    fail-closed payload rather than mislabelling the
    interruption.
    """


def _compatibility_form_cell(
    backend: Any,
    fmt: Any,
    fixture: Any,
    ids: tuple[int, ...],
    *,
    boundary_ids: tuple[int, ...],
    max_new_tokens: int,
    max_context: int,
    seed: int,
    cancel: Event | None,
) -> tuple[str | None, dict[str, Any] | None]:
    """One compatibility-form generation: (detail, cell).

    Exactly one of the pair is set. ``ids`` are the arm
    render's ids, measured by the stage at construction for
    the record's arm token counts and ids-identical
    derivations; the cell's ``input_tokens`` come from the
    evaluation's own observation, so a backend whose encode
    disagreed with itself surfaces as a validator rejection
    downstream, never a silently wrong record (the mirror
    binds the two). Every cell carries ``answer_text`` -- the
    decoded pre-boundary answer, verbatim, never normalised
    -- bound by the cell's ``answer_digest`` (the cycle-190
    pattern), because the decomposition classes are functions
    of the answer text. A cell is built only from an
    aggregate-eligible report, whose invariants force the
    recorded observation's status to "completed", so the
    verdict recomputes from the digest and the boundary flag
    exactly. The cancel event is checked before and after
    ``evaluate`` (the multiturn rule: a cancel observed during
    a generation is never mislabelled an evaluation failure).
    """
    if len(ids) + max_new_tokens > max_context:
        return "arm_context_exceeded", None
    if cancel is not None and cancel.is_set():
        raise _CompatibilityFormCancelled
    report = evaluate(
        backend,
        fmt,
        (fixture,),
        boundary=boundary_ids,
        max_new_tokens=max_new_tokens,
        seed=seed,
        cancel=cancel,
    )
    if cancel is not None and cancel.is_set():
        raise _CompatibilityFormCancelled
    if report.aggregate_eligible and report.observations:
        obs = report.observations[0]
        return None, {
            "verdict": bool(obs.exact_match and obs.boundary_found),
            "boundary_found": bool(obs.boundary_found),
            "boundary_text_match": obs.boundary_text_match,
            "input_tokens": len(obs.input_ids),
            "answer_digest": _answer_digest(obs.answer),
            "answer_text": obs.answer,
        }
    return "arm_evaluation_failed", None


def compatibility_form_stage(
    backend: Any,
    formats: tuple[Any, ...],
    *,
    boundary: tuple[int, ...],
    boundary_map: dict[str, tuple[int, ...]] | None = None,
    max_new_tokens: int = 64,
    max_context: int,
    seed: int = 0,
    cancel: Event | None = None,
) -> dict[str, Any]:
    """Compatibility-form stage (cycle 231): a generated
    24-case fact-recall panel whose frozen marker sentence
    hosts one field per compatibility class; each treated
    arm replaces exactly its own field's plain-text reading
    with the frozen compatibility character (117 members in
    six classes from UnicodeData.txt 15.0.0), per candidate.

    The cases are generated from the run seed inside the stage;
    there is no dependence on the user's fixtures or the primary
    run. Surgery is at the fixture level: every arm of a case is
    a Fixture with the same id, group, and expected, so request
    seeds are identical across arms by construction
    (``request_seed_for`` is id-derived) and ``core.evaluate``
    is used unchanged. The pre-generation screens (the per-case
    invariant battery, the population audit, the
    special-token-literal audit over every arm's full message
    content for every family, and boundary-text containment)
    fail the whole stage closed with
    ``compatibility_form_screen_failed``; their counts travel
    in the payload's ``audit`` block, never silently dropped.
    The serialised-records byte cap degrades the whole stage to
    ``compatibility_form_record_budget_exceeded``. Design:
    docs/COMPATIBILITYFORM_DESIGN.md.
    """
    from . import compatibilityform as _cf
    from .fixtures import audit_special_token_literals

    if type(formats) is not tuple or not formats:
        raise ValueError("formats must be a nonempty tuple")
    names = [fmt.name for fmt in formats]
    if len(set(names)) != len(names):
        # Candidate records are keyed by name; two formats
        # sharing a name would silently collapse into one
        # candidate's records (red-team F7, cycle 193). The
        # precondition is stated at this stage's own boundary.
        raise ValueError("formats must have unique names")
    try:
        protocol: dict[str, Any] = dict(_cf.protocol_object(seed, _backend_fingerprint(backend)))
        protocol["fingerprint"] = _cf.protocol_fingerprint(seed, _backend_fingerprint(backend))
    except _cf.RedrawExhausted:
        # The protocol's ground hash itself regenerates the
        # panel, so exhaustion can surface here first; it is
        # the same fail-closed screen outcome, with no
        # protocol to report.
        return {
            "available": False,
            "detail": "compatibility_form_screen_failed",
            "protocol": {},
            "audit": {
                "n_audit_excluded": 0,
                "by_reason": {"redraw_exhausted": 1},
                # nosec B105 -- integer audit counter field, not a credential
                "special_token_findings": 0,
            },
            "candidates": {},
        }
    audit: dict[str, Any] = {
        "n_audit_excluded": 0,
        "by_reason": {},
        # nosec B105 -- integer audit counter field, not a credential
        "special_token_findings": 0,
    }
    if len(formats) * _cf.CASE_COUNT > _cf.MAX_GROUND_RECORDS:
        return {
            "available": False,
            "detail": "compatibility_form_record_budget_exceeded",
            "protocol": protocol,
            "audit": audit,
            "candidates": {},
        }
    try:
        cases = _cf.generate_cases(seed)
    except _cf.RedrawExhausted:
        return {
            "available": False,
            "detail": "compatibility_form_screen_failed",
            "protocol": protocol,
            "audit": {
                "n_audit_excluded": 0,
                "by_reason": {"redraw_exhausted": 1},
                # nosec B105 -- integer audit counter field, not a credential
                "special_token_findings": 0,
            },
            "candidates": {},
        }
    invalid = sum(1 for case in cases if not _cf.validate_case(case))
    if invalid or not _cf.POPULATION_AUDIT["ok"]:
        by_reason: dict[str, int] = {}
        if invalid:
            by_reason["case_invariant"] = invalid
        if not _cf.POPULATION_AUDIT["ok"]:
            by_reason["arm_population"] = 1
        return {
            "available": False,
            "detail": "compatibility_form_screen_failed",
            "protocol": protocol,
            "audit": {
                "n_audit_excluded": 0,
                "by_reason": by_reason,
                # nosec B105 -- integer audit counter field, not a credential
                "special_token_findings": 0,
            },
            "candidates": {},
        }
    contents = tuple(_cf.arm_content(case, arm) for case in cases for arm in _cf.ARM_KEYS)
    findings = audit_special_token_literals(contents, formats)
    if findings:
        # A special-token literal in an arm's content would
        # forge turn structure in every cell of every run; the
        # stage fails closed before any generation, with the
        # count in the audit block (never silently dropped).
        audit = {
            "n_audit_excluded": 0,
            "by_reason": {"special_token_literal": len(findings)},
            "special_token_findings": len(findings),
        }
        return {
            "available": False,
            "detail": "compatibility_form_screen_failed",
            "protocol": protocol,
            "audit": audit,
            "candidates": {},
        }
    boundaries = _resolve_boundaries(formats, boundary, boundary_map)
    for fmt in formats:
        try:
            boundary_text = backend.decode(boundaries[fmt.name])
        except Exception:  # noqa: BLE001 -- an undecodable boundary screens nothing
            boundary_text = ""
        if boundary_text and any(boundary_text in content for content in contents):
            return {
                "available": False,
                "detail": "compatibility_form_screen_failed",
                "protocol": protocol,
                "audit": audit,
                "candidates": {},
            }

    def _cancelled() -> bool:
        """Return the cancelled."""
        return cancel is not None and cancel.is_set()

    candidates: dict[str, Any] = {}
    for fmt in formats:
        name = fmt.name
        records: list[dict[str, Any]] = []
        try:
            for case in cases:
                if _cancelled():
                    raise _CompatibilityFormCancelled
                case_id = str(case["fixture_id"])
                record: dict[str, Any] = {
                    "fixture": case_id,
                    "topic": str(case["topic"]),
                    "expected_digest": _cf.digest_text(str(case["expected"])),
                    "content_digests": dict.fromkeys(_cf.ARM_KEYS),
                    "arm_tokens": dict.fromkeys(_cf.ARM_KEYS),
                    "id_digests": dict.fromkeys(_cf.ARM_KEYS),
                    "ids_identical": dict.fromkeys(_cf.TREATED_ARMS),
                    "id_length_delta": dict.fromkeys(_cf.TREATED_ARMS),
                    "table_pin": {
                        "version": _cf.UNICODE_VERSION,
                        "unicodedata_sha256": _cf.UNICODEDATA_SHA256,
                        "table_sha256": _cf.table_sha256(),
                    },
                    "members": {key: int(value) for key, value in case["members"].items()},
                    "cells": dict.fromkeys(_cf.ARM_KEYS),
                    "cell_details": dict.fromkeys(_cf.ARM_KEYS, ""),
                }
                for arm in _cf.ARM_KEYS:
                    content = _cf.arm_content(case, arm)
                    fixture = Fixture(
                        id=case_id,
                        messages=(Message("user", content),),
                        expected=str(case["expected"]),
                        group=_cf.GROUP,
                    )
                    ids = tuple(backend.encode(fmt.render(fixture.messages)))
                    record["arm_tokens"][arm] = len(ids)
                    record["id_digests"][arm] = _cf.digest_ids(ids)
                    record["content_digests"][arm] = _cf.digest_text(content)
                    detail, cell = _compatibility_form_cell(
                        backend,
                        fmt,
                        fixture,
                        ids,
                        boundary_ids=boundaries[name],
                        max_new_tokens=max_new_tokens,
                        max_context=max_context,
                        seed=seed,
                        cancel=cancel,
                    )
                    if detail is not None:
                        record["cell_details"][arm] = detail
                    elif cell is not None:
                        record["cells"][arm] = cell
                for arm in _cf.TREATED_ARMS:
                    record["ids_identical"][arm] = (
                        record["id_digests"][arm] == record["id_digests"]["control"]
                    )
                    record["id_length_delta"][arm] = int(record["arm_tokens"][arm]) - int(
                        record["arm_tokens"]["control"]
                    )
                records.append(record)
        except _CompatibilityFormCancelled:
            return {
                "available": False,
                "detail": "cancelled",
                "protocol": protocol,
                "audit": audit,
                "candidates": {},
            }
        candidates[name] = {"status": "available", "detail": "", "records": records}
    if sum(_cf.records_bytes(entry["records"]) for entry in candidates.values()) > (
        _cf.MAX_RECORDS_BYTES
    ):
        # The byte cap is a hard bound on what a results file
        # may carry (a non-ASCII-stressed panel of answer texts
        # reaches it; design section 6 budgets): the whole stage
        # degrades to the frozen budget detail, never a partial
        # panel.
        return {
            "available": False,
            "detail": "compatibility_form_record_budget_exceeded",
            "protocol": protocol,
            "audit": audit,
            "candidates": {},
        }
    return {
        "available": True,
        "detail": "",
        "protocol": protocol,
        "audit": audit,
        "candidates": candidates,
    }


def compatibility_form_analysis(
    stage: dict[str, Any],
    candidate_names: tuple[str, ...],
) -> dict[str, Any]:
    """Aggregate a compatibility-form stage into the stored
    payload (cycle 231).

    Per-candidate readouts come from
    compatibilityform.aggregate_candidate over the candidate's
    ground records and the regenerated panel: per-arm rates
    with the six-class decomposition, the per-arm id report
    (the regime gate), the per-arm dose deltas, and per treated
    arm the guarded headline over the id-divergent paired
    population, with its label, counts, divergent-domain size
    M, and decomposition. The stage never affects selection.
    Design: docs/COMPATIBILITYFORM_DESIGN.md.
    """
    from . import compatibilityform as _cf

    if not stage.get("available"):
        return {
            "available": False,
            "detail": str(stage.get("detail", "compatibility_form_unavailable")),
            "protocol": stage.get("protocol"),
            "audit": stage.get("audit"),
        }
    stage_candidates = stage["candidates"]
    cases = _cf.generate_cases(int(stage["protocol"]["seed"]))
    candidates_out: dict[str, Any] = {}
    for name in candidate_names:
        entry = stage_candidates[name]
        if entry["status"] != "available":
            candidates_out[name] = {
                "status": "unavailable",
                "detail": entry["detail"],
                "records": [],
            }
            continue
        aggregates = _cf.aggregate_candidate(entry["records"], cases)
        aggregates.update({"status": "available", "detail": "", "records": entry["records"]})
        candidates_out[name] = aggregates
    return {
        "available": True,
        "detail": "",
        "protocol": stage["protocol"],
        "audit": stage.get("audit"),
        "candidates": candidates_out,
        "n_candidates": len(candidate_names),
        "n_cases": len(cases),
    }


class _EnumerationConventionCancelled(Exception):
    """Internal: the run's cancel event fired during the
    enumeration-convention stage; the stage unwinds to its
    fail-closed payload rather than mislabelling the
    interruption.
    """


def _enumeration_convention_cell(
    backend: Any,
    fmt: Any,
    fixture: Any,
    ids: tuple[int, ...],
    *,
    boundary_ids: tuple[int, ...],
    max_new_tokens: int,
    max_context: int,
    seed: int,
    cancel: Event | None,
) -> tuple[str | None, dict[str, Any] | None]:
    """One enumeration-convention generation: (detail, cell).

    Exactly one of the pair is set. ``ids`` are the arm
    render's ids, measured by the stage at construction for
    the record's arm token counts and ids-identical
    derivations; the cell's ``input_tokens`` come from the
    evaluation's own observation, so a backend whose encode
    disagreed with itself surfaces as a validator rejection
    downstream, never a silently wrong record (the mirror
    binds the two). Every cell carries ``answer_text`` -- the
    decoded pre-boundary answer, verbatim, never normalised
    -- bound by the cell's ``answer_digest`` (the cycle-190
    pattern), because the decomposition classes are functions
    of the answer text. A cell is built only from an
    aggregate-eligible report, whose invariants force the
    recorded observation's status to "completed", so the
    verdict recomputes from the digest and the boundary flag
    exactly. The cancel event is checked before and after
    ``evaluate`` (the multiturn rule: a cancel observed during
    a generation is never mislabelled an evaluation failure).
    """
    if len(ids) + max_new_tokens > max_context:
        return "arm_context_exceeded", None
    if cancel is not None and cancel.is_set():
        raise _EnumerationConventionCancelled
    report = evaluate(
        backend,
        fmt,
        (fixture,),
        boundary=boundary_ids,
        max_new_tokens=max_new_tokens,
        seed=seed,
        cancel=cancel,
    )
    if cancel is not None and cancel.is_set():
        raise _EnumerationConventionCancelled
    if report.aggregate_eligible and report.observations:
        obs = report.observations[0]
        return None, {
            "verdict": bool(obs.exact_match and obs.boundary_found),
            "boundary_found": bool(obs.boundary_found),
            "boundary_text_match": obs.boundary_text_match,
            "input_tokens": len(obs.input_ids),
            "answer_digest": _answer_digest(obs.answer),
            "answer_text": obs.answer,
        }
    return "arm_evaluation_failed", None


def enumeration_convention_stage(
    backend: Any,
    formats: tuple[Any, ...],
    *,
    boundary: tuple[int, ...],
    boundary_map: dict[str, tuple[int, ...]] | None = None,
    max_new_tokens: int = 64,
    max_context: int,
    seed: int = 0,
    cancel: Event | None = None,
) -> dict[str, Any]:
    """Enumeration-convention stage (cycle 233): a generated
    24-case fact-recall panel whose frozen list gives each
    position a nine-item enumeration under one of six marker
    systems (arabic control, upper/lower alpha, Roman,
    parenthesised wrapper, bullet); the count is never stated
    anywhere in the content, so the answer can only be found
    by enumerating the list, per candidate.

    The cases are generated from the run seed inside the stage;
    there is no dependence on the user's fixtures or the primary
    run. Surgery is at the fixture level: every arm of a case is
    a Fixture with the same id, group, and expected, so request
    seeds are identical across arms by construction
    (``request_seed_for`` is id-derived) and ``core.evaluate``
    is used unchanged. The pre-generation screens (the per-case
    invariant battery, the panel-global value distinctness, the
    special-token-literal audit over every arm's full message
    content for every family, and boundary-text containment)
    fail the whole stage closed with
    ``enumeration_convention_screen_failed``; their counts
    travel in the payload's ``audit`` block, never silently
    dropped. The serialised-records byte cap degrades the whole
    stage to ``enumeration_convention_record_budget_exceeded``.
    Design: docs/ENUMERATION_DESIGN.md.
    """
    from . import enumerationconvention as _ec
    from .fixtures import audit_special_token_literals

    if type(formats) is not tuple or not formats:
        raise ValueError("formats must be a nonempty tuple")
    names = [fmt.name for fmt in formats]
    if len(set(names)) != len(names):
        # Candidate records are keyed by name; two formats
        # sharing a name would silently collapse into one
        # candidate's records (red-team F7, cycle 193). The
        # precondition is stated at this stage's own boundary.
        raise ValueError("formats must have unique names")
    try:
        protocol: dict[str, Any] = dict(_ec.protocol_object(seed, _backend_fingerprint(backend)))
        protocol["fingerprint"] = _ec.protocol_fingerprint(seed, _backend_fingerprint(backend))
    except _ec.RedrawExhausted:
        # The protocol's ground hash itself regenerates the
        # panel, so exhaustion can surface here first; it is
        # the same fail-closed screen outcome, with no
        # protocol to report.
        return {
            "available": False,
            "detail": "enumeration_convention_screen_failed",
            "protocol": {},
            "audit": {
                "n_audit_excluded": 0,
                "by_reason": {"redraw_exhausted": 1},
                # nosec B105 -- integer audit counter field, not a credential
                "special_token_findings": 0,
            },
            "candidates": {},
        }
    audit: dict[str, Any] = {
        "n_audit_excluded": 0,
        "by_reason": {},
        # nosec B105 -- integer audit counter field, not a credential
        "special_token_findings": 0,
    }
    if len(formats) * _ec.CASE_COUNT > _ec.MAX_GROUND_RECORDS:
        return {
            "available": False,
            "detail": "enumeration_convention_record_budget_exceeded",
            "protocol": protocol,
            "audit": audit,
            "candidates": {},
        }
    try:
        cases = _ec.generate_cases(seed)
    except _ec.RedrawExhausted:
        return {
            "available": False,
            "detail": "enumeration_convention_screen_failed",
            "protocol": protocol,
            "audit": {
                "n_audit_excluded": 0,
                "by_reason": {"redraw_exhausted": 1},
                # nosec B105 -- integer audit counter field, not a credential
                "special_token_findings": 0,
            },
            "candidates": {},
        }
    invalid = sum(1 for case in cases if not _ec.validate_case(case))
    panel_values = [value for case in cases for value in case["values"]]
    panel_distinct = len(set(panel_values)) == len(panel_values)
    if invalid or not panel_distinct:
        by_reason: dict[str, int] = {}
        if invalid:
            by_reason["case_invariant"] = invalid
        if not panel_distinct:
            by_reason["panel_distinctness"] = 1
        return {
            "available": False,
            "detail": "enumeration_convention_screen_failed",
            "protocol": protocol,
            "audit": {
                "n_audit_excluded": 0,
                "by_reason": by_reason,
                # nosec B105 -- integer audit counter field, not a credential
                "special_token_findings": 0,
            },
            "candidates": {},
        }
    contents = tuple(_ec.arm_content(case, arm) for case in cases for arm in _ec.ARM_KEYS)
    findings = audit_special_token_literals(contents, formats)
    if findings:
        # A special-token literal in an arm's content would
        # forge turn structure in every cell of every run; the
        # stage fails closed before any generation, with the
        # count in the audit block (never silently dropped).
        audit = {
            "n_audit_excluded": 0,
            "by_reason": {"special_token_literal": len(findings)},
            "special_token_findings": len(findings),
        }
        return {
            "available": False,
            "detail": "enumeration_convention_screen_failed",
            "protocol": protocol,
            "audit": audit,
            "candidates": {},
        }
    boundaries = _resolve_boundaries(formats, boundary, boundary_map)
    for fmt in formats:
        try:
            boundary_text = backend.decode(boundaries[fmt.name])
        except Exception:  # noqa: BLE001 -- an undecodable boundary screens nothing
            boundary_text = ""
        if boundary_text and any(boundary_text in content for content in contents):
            return {
                "available": False,
                "detail": "enumeration_convention_screen_failed",
                "protocol": protocol,
                "audit": audit,
                "candidates": {},
            }

    def _cancelled() -> bool:
        """Return the cancelled."""
        return cancel is not None and cancel.is_set()

    candidates: dict[str, Any] = {}
    for fmt in formats:
        name = fmt.name
        records: list[dict[str, Any]] = []
        try:
            for case in cases:
                if _cancelled():
                    raise _EnumerationConventionCancelled
                case_id = str(case["fixture_id"])
                record: dict[str, Any] = {
                    "fixture": case_id,
                    "topic": str(case["topic"]),
                    "target_position": int(case["position"]),
                    "expected_digest": _ec.digest_text(str(case["expected"])),
                    "content_digests": dict.fromkeys(_ec.ARM_KEYS),
                    "arm_tokens": dict.fromkeys(_ec.ARM_KEYS),
                    "id_digests": dict.fromkeys(_ec.ARM_KEYS),
                    "ids_identical": dict.fromkeys(_ec.TREATED_ARMS),
                    "id_length_delta": dict.fromkeys(_ec.TREATED_ARMS),
                    "cells": dict.fromkeys(_ec.ARM_KEYS),
                    "cell_details": dict.fromkeys(_ec.ARM_KEYS, ""),
                }
                for arm in _ec.ARM_KEYS:
                    content = _ec.arm_content(case, arm)
                    fixture = Fixture(
                        id=case_id,
                        messages=(Message("user", content),),
                        expected=str(case["expected"]),
                        group=_ec.GROUP,
                    )
                    ids = tuple(backend.encode(fmt.render(fixture.messages)))
                    record["arm_tokens"][arm] = len(ids)
                    record["id_digests"][arm] = _ec.digest_ids(ids)
                    record["content_digests"][arm] = _ec.digest_text(content)
                    detail, cell = _enumeration_convention_cell(
                        backend,
                        fmt,
                        fixture,
                        ids,
                        boundary_ids=boundaries[name],
                        max_new_tokens=max_new_tokens,
                        max_context=max_context,
                        seed=seed,
                        cancel=cancel,
                    )
                    if detail is not None:
                        record["cell_details"][arm] = detail
                    elif cell is not None:
                        record["cells"][arm] = cell
                for arm in _ec.TREATED_ARMS:
                    record["ids_identical"][arm] = (
                        record["id_digests"][arm] == record["id_digests"]["control_arabic"]
                    )
                    record["id_length_delta"][arm] = int(record["arm_tokens"][arm]) - int(
                        record["arm_tokens"]["control_arabic"]
                    )
                records.append(record)
        except _EnumerationConventionCancelled:
            return {
                "available": False,
                "detail": "cancelled",
                "protocol": protocol,
                "audit": audit,
                "candidates": {},
            }
        candidates[name] = {"status": "available", "detail": "", "records": records}
    if sum(_ec.records_bytes(entry["records"]) for entry in candidates.values()) > (
        _ec.MAX_RECORDS_BYTES
    ):
        # The byte cap is a hard bound on what a results file
        # may carry (a non-ASCII-stressed panel of answer texts
        # reaches it; design section 6 budgets): the whole stage
        # degrades to the frozen budget detail, never a partial
        # panel.
        return {
            "available": False,
            "detail": "enumeration_convention_record_budget_exceeded",
            "protocol": protocol,
            "audit": audit,
            "candidates": {},
        }
    return {
        "available": True,
        "detail": "",
        "protocol": protocol,
        "audit": audit,
        "candidates": candidates,
    }


def enumeration_convention_analysis(
    stage: dict[str, Any],
    candidate_names: tuple[str, ...],
) -> dict[str, Any]:
    """Aggregate an enumeration-convention stage into the stored
    payload (cycle 233).

    Per-candidate readouts come from
    enumerationconvention.aggregate_candidate over the
    candidate's ground records and the regenerated panel:
    per-arm rates with the six-class decomposition, the per-arm
    id report (the regime gate), the per-position length join
    (amendment A5), and per treated arm the guarded headline
    over the id-divergent paired population, with its label,
    counts, divergent-domain size M, and decomposition. The
    stage never affects selection. Design:
    docs/ENUMERATION_DESIGN.md.
    """
    from . import enumerationconvention as _ec

    if not stage.get("available"):
        return {
            "available": False,
            "detail": str(stage.get("detail", "enumeration_convention_unavailable")),
            "protocol": stage.get("protocol"),
            "audit": stage.get("audit"),
        }
    stage_candidates = stage["candidates"]
    cases = _ec.generate_cases(int(stage["protocol"]["seed"]))
    candidates_out: dict[str, Any] = {}
    for name in candidate_names:
        entry = stage_candidates[name]
        if entry["status"] != "available":
            candidates_out[name] = {
                "status": "unavailable",
                "detail": entry["detail"],
                "records": [],
            }
            continue
        aggregates = _ec.aggregate_candidate(entry["records"], cases)
        aggregates.update({"status": "available", "detail": "", "records": entry["records"]})
        candidates_out[name] = aggregates
    return {
        "available": True,
        "detail": "",
        "protocol": stage["protocol"],
        "audit": stage.get("audit"),
        "candidates": candidates_out,
        "n_candidates": len(candidate_names),
        "n_cases": len(cases),
    }


class _TypographicEmphasisCancelled(Exception):
    """Internal: the run's cancel event fired during the
    typographic-emphasis stage; the stage unwinds to
    its fail-closed payload rather than mislabelling
    the interruption.
    """


def _typographic_emphasis_cell(
    backend: Any,
    fmt: Any,
    fixture: Any,
    ids: tuple[int, ...],
    *,
    boundary_ids: tuple[int, ...],
    max_new_tokens: int,
    max_context: int,
    seed: int,
    cancel: Event | None,
) -> tuple[str | None, dict[str, Any] | None]:
    """One typographic-emphasis generation: (detail, cell).

    Exactly one of the pair is set. ``ids`` are the arm
    render's ids, measured by the stage at construction for
    the record's arm token counts and ids-identical
    derivations; the cell's ``input_tokens`` come from the
    evaluation's own observation, so a backend whose encode
    disagreed with itself surfaces as a validator rejection
    downstream, never a silently wrong record (the mirror
    binds the two). Every cell carries ``answer_text`` -- the
    decoded pre-boundary answer, verbatim, never normalised
    -- bound by the cell's ``answer_digest`` (the cycle-190
    pattern), because the decomposition classes are functions
    of the answer text. A cell is built only from an
    aggregate-eligible report, whose invariants force the
    recorded observation's status to "completed", so the
    verdict recomputes from the digest and the boundary flag
    exactly. The cancel event is checked before and after
    ``evaluate`` (the multiturn rule: a cancel observed during
    a generation is never mislabelled an evaluation failure).
    """
    if len(ids) + max_new_tokens > max_context:
        return "arm_context_exceeded", None
    if cancel is not None and cancel.is_set():
        raise _TypographicEmphasisCancelled
    report = evaluate(
        backend,
        fmt,
        (fixture,),
        boundary=boundary_ids,
        max_new_tokens=max_new_tokens,
        seed=seed,
        cancel=cancel,
    )
    if cancel is not None and cancel.is_set():
        raise _TypographicEmphasisCancelled
    if report.aggregate_eligible and report.observations:
        obs = report.observations[0]
        return None, {
            "verdict": bool(obs.exact_match and obs.boundary_found),
            "boundary_found": bool(obs.boundary_found),
            "boundary_text_match": obs.boundary_text_match,
            "input_tokens": len(obs.input_ids),
            "answer_digest": _answer_digest(obs.answer),
            "answer_text": obs.answer,
        }
    return "arm_evaluation_failed", None


def typographic_emphasis_stage(
    backend: Any,
    formats: tuple[Any, ...],
    *,
    boundary: tuple[int, ...],
    boundary_map: dict[str, tuple[int, ...]] | None = None,
    max_new_tokens: int = 64,
    max_context: int,
    seed: int = 0,
    cancel: Event | None = None,
) -> dict[str, Any]:
    """Typographic-emphasis stage (cycle 237): a generated
    24-case fact-recall panel whose fact sentence's topic
    span is rendered verbatim (control) or under one
    typographic treatment (bold, italic, monospace, caps),
    with a bold-detection positive-control panel and a
    both-sites PC-B arm, per candidate.

    The cases are generated from the run seed inside the stage;
    there is no dependence on the user's fixtures or the primary
    run. Surgery is at the fixture level: every arm of a case is
    a Fixture with the same id, group, and expected, so request
    seeds are identical across arms by construction
    (``request_seed_for`` is id-derived) and ``core.evaluate``
    is used unchanged. The pre-generation screens (the per-case
    invariant battery, the panel-global value distinctness, the
    special-token-literal audit over every arm's full message
    content for every family, and boundary-text containment)
    fail the whole stage closed with
    ``typographic_emphasis_screen_failed``; their counts
    travel in the payload's ``audit`` block, never silently
    dropped. The serialised-records byte cap degrades the whole
    stage to ``typographic_emphasis_record_budget_exceeded``.
    Design: docs/TYPOGRAPHIC_EMPHASIS_DESIGN.md.
    """
    from . import typographicemphasis as _te
    from .fixtures import audit_special_token_literals

    if type(formats) is not tuple or not formats:
        raise ValueError("formats must be a nonempty tuple")
    names = [fmt.name for fmt in formats]
    if len(set(names)) != len(names):
        # Candidate records are keyed by name; two formats
        # sharing a name would silently collapse into one
        # candidate's records (red-team F7, cycle 193). The
        # precondition is stated at this stage's own boundary.
        raise ValueError("formats must have unique names")
    try:
        protocol: dict[str, Any] = dict(_te.protocol_object(seed, _backend_fingerprint(backend)))
        protocol["fingerprint"] = _te.protocol_fingerprint(seed, _backend_fingerprint(backend))
    except _te.RedrawExhausted:
        # The protocol's ground hash itself regenerates the
        # panel, so exhaustion can surface here first; it is
        # the same fail-closed screen outcome, with no
        # protocol to report.
        return {
            "available": False,
            "detail": "typographic_emphasis_screen_failed",
            "protocol": {},
            "audit": {
                "n_audit_excluded": 0,
                "by_reason": {"redraw_exhausted": 1},
                # nosec B105 -- integer audit counter field, not a credential
                "special_token_findings": 0,
            },
            "candidates": {},
        }
    audit: dict[str, Any] = {
        "n_audit_excluded": 0,
        "by_reason": {},
        # nosec B105 -- integer audit counter field, not a credential
        "special_token_findings": 0,
    }
    if len(formats) * _te.CASE_COUNT > _te.MAX_GROUND_RECORDS:
        return {
            "available": False,
            "detail": "typographic_emphasis_record_budget_exceeded",
            "protocol": protocol,
            "audit": audit,
            "candidates": {},
        }
    try:
        cases = _te.generate_cases(seed)
    except _te.RedrawExhausted:
        return {
            "available": False,
            "detail": "typographic_emphasis_screen_failed",
            "protocol": protocol,
            "audit": {
                "n_audit_excluded": 0,
                "by_reason": {"redraw_exhausted": 1},
                # nosec B105 -- integer audit counter field, not a credential
                "special_token_findings": 0,
            },
            "candidates": {},
        }
    invalid = sum(1 for case in cases if not _te.validate_case(case))
    panel_values = [str(case["expected"]) for case in cases]
    panel_distinct = len(set(panel_values)) == len(panel_values)
    if invalid or not panel_distinct:
        by_reason: dict[str, int] = {}
        if invalid:
            by_reason["case_invariant"] = invalid
        if not panel_distinct:
            by_reason["panel_distinctness"] = 1
        return {
            "available": False,
            "detail": "typographic_emphasis_screen_failed",
            "protocol": protocol,
            "audit": {
                "n_audit_excluded": 0,
                "by_reason": by_reason,
                # nosec B105 -- integer audit counter field, not a credential
                "special_token_findings": 0,
            },
            "candidates": {},
        }
    contents = tuple(_te.content_for(case, arm) for case in cases for arm in _te.RECORD_ARMS)
    findings = audit_special_token_literals(contents, formats)
    if findings:
        # A special-token literal in an arm's content would
        # forge turn structure in every cell of every run; the
        # stage fails closed before any generation, with the
        # count in the audit block (never silently dropped).
        audit = {
            "n_audit_excluded": 0,
            "by_reason": {"special_token_literal": len(findings)},
            "special_token_findings": len(findings),
        }
        return {
            "available": False,
            "detail": "typographic_emphasis_screen_failed",
            "protocol": protocol,
            "audit": audit,
            "candidates": {},
        }
    boundaries = _resolve_boundaries(formats, boundary, boundary_map)
    for fmt in formats:
        try:
            boundary_text = backend.decode(boundaries[fmt.name])
        except Exception:  # noqa: BLE001 -- an undecodable boundary screens nothing
            boundary_text = ""
        if boundary_text and any(boundary_text in content for content in contents):
            return {
                "available": False,
                "detail": "typographic_emphasis_screen_failed",
                "protocol": protocol,
                "audit": audit,
                "candidates": {},
            }

    def _cancelled() -> bool:
        """Return the cancelled."""
        return cancel is not None and cancel.is_set()

    candidates: dict[str, Any] = {}
    for fmt in formats:
        name = fmt.name
        records: list[dict[str, Any]] = []
        try:
            for case in cases:
                if _cancelled():
                    raise _TypographicEmphasisCancelled
                case_id = str(case["fixture_id"])
                record: dict[str, Any] = {
                    "fixture": case_id,
                    "topic": str(case["topic"]),
                    "expected_digest": _te.digest_text(str(case["expected"])),
                    "content_digests": dict.fromkeys(_te.RECORD_ARMS),
                    "arm_tokens": dict.fromkeys(_te.RECORD_ARMS),
                    "id_digests": dict.fromkeys(_te.RECORD_ARMS),
                    "ids_identical": dict.fromkeys(_te.TREATED_ARMS),
                    "id_length_delta": dict.fromkeys(_te.TREATED_ARMS),
                    "cells": dict.fromkeys(_te.RECORD_ARMS),
                    "cell_details": dict.fromkeys(_te.RECORD_ARMS, ""),
                }
                for arm in _te.RECORD_ARMS:
                    content = _te.content_for(case, arm)
                    fixture = Fixture(
                        id=case_id,
                        messages=(Message("user", content),),
                        expected=str(case["expected"]),
                        group=_te.GROUP,
                    )
                    ids = tuple(backend.encode(fmt.render(fixture.messages)))
                    record["arm_tokens"][arm] = len(ids)
                    record["id_digests"][arm] = _te.digest_ids(ids)
                    record["content_digests"][arm] = _te.digest_text(content)
                    detail, cell = _typographic_emphasis_cell(
                        backend,
                        fmt,
                        fixture,
                        ids,
                        boundary_ids=boundaries[name],
                        max_new_tokens=max_new_tokens,
                        max_context=max_context,
                        seed=seed,
                        cancel=cancel,
                    )
                    if detail is not None:
                        record["cell_details"][arm] = detail
                    elif cell is not None:
                        record["cells"][arm] = cell
                for arm in _te.TREATED_ARMS:
                    record["ids_identical"][arm] = (
                        record["id_digests"][arm] == record["id_digests"]["control"]
                    )
                    record["id_length_delta"][arm] = int(record["arm_tokens"][arm]) - int(
                        record["arm_tokens"]["control"]
                    )
                records.append(record)
        except _TypographicEmphasisCancelled:
            return {
                "available": False,
                "detail": "cancelled",
                "protocol": protocol,
                "audit": audit,
                "candidates": {},
            }
        candidates[name] = {"status": "available", "detail": "", "records": records}
    if sum(_te.records_bytes(entry["records"]) for entry in candidates.values()) > (
        _te.MAX_RECORDS_BYTES
    ):
        # The byte cap is a hard bound on what a results file
        # may carry (a non-ASCII-stressed panel of answer texts
        # reaches it; design section 6 budgets): the whole stage
        # degrades to the frozen budget detail, never a partial
        # panel.
        return {
            "available": False,
            "detail": "typographic_emphasis_record_budget_exceeded",
            "protocol": protocol,
            "audit": audit,
            "candidates": {},
        }
    return {
        "available": True,
        "detail": "",
        "protocol": protocol,
        "audit": audit,
        "candidates": candidates,
    }


def typographic_emphasis_analysis(
    stage: dict[str, Any],
    candidate_names: tuple[str, ...],
) -> dict[str, Any]:
    """Aggregate a typographic-emphasis stage into the stored
    payload (cycle 237).

    Per-candidate readouts come from
    typographicemphasis.aggregate_candidate over the
    candidate's ground records and the regenerated panel:
    per-arm rates with the completed-cell decomposition,
    the per-arm dose deltas, the per-arm id report (the
    regime gate), the positive-control block with its
    gate verdict, and per treated arm the combined
    guarded headline over the id-divergent paired
    population, with its label, counts, divergent-domain
    size M, certification companions, and the gates that
    blocked a certificate. The stage never affects
    selection. Design: docs/TYPOGRAPHIC_EMPHASIS_DESIGN.md.
    """
    from . import typographicemphasis as _te

    if not stage.get("available"):
        return {
            "available": False,
            "detail": str(stage.get("detail", "typographic_emphasis_unavailable")),
            "protocol": stage.get("protocol"),
            "audit": stage.get("audit"),
        }
    stage_candidates = stage["candidates"]
    cases = _te.generate_cases(int(stage["protocol"]["seed"]))
    candidates_out: dict[str, Any] = {}
    for name in candidate_names:
        entry = stage_candidates[name]
        if entry["status"] != "available":
            candidates_out[name] = {
                "status": "unavailable",
                "detail": entry["detail"],
                "records": [],
            }
            continue
        aggregates = _te.aggregate_candidate(entry["records"], cases)
        aggregates.update({"status": "available", "detail": "", "records": entry["records"]})
        candidates_out[name] = aggregates
    return {
        "available": True,
        "detail": "",
        "protocol": stage["protocol"],
        "audit": stage.get("audit"),
        "candidates": candidates_out,
        "n_candidates": len(candidate_names),
        "n_cases": len(cases),
    }
