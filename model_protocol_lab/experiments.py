"""Bounded search followed by one frozen held-out evaluation."""

import json
import math
from bisect import bisect_right
from collections.abc import Callable
from dataclasses import asdict
from hashlib import sha256
from threading import Event
from typing import Any, cast

from .aliasing import AUDIT_ID as ALIASING_AUDIT_ID
from .aliasing import DIGEST_VERSION as ALIASING_DIGEST_VERSION
from .aliasing import SCOPE as ALIASING_SCOPE
from .aliasing import analyze_candidate as aliasing_analyze_candidate
from .aliasing import class_partitions as aliasing_class_partitions
from .aliasing import evaluable_fixtures as aliasing_evaluable_fixtures
from .aliasing import pair_matrix as aliasing_pair_matrix
from .aliasing import primary_cohort as aliasing_primary_cohort
from .aliasing import redundant_counts as aliasing_redundant_counts
from .aliasing import suspect_cells as aliasing_suspect_cells
from .backends import (
    StopOutcome,
    TokenHealingBackend,
    TokenTruncatedPromptBackend,
    TruncatedPromptBackend,
    emulate_eos_stop,
    emulate_hf_stop_string,
    emulate_vllm_holdback,
)
from .cacheprofile import candidate_cache_profile
from .core import (
    Backend,
    EvaluationReport,
    Fixture,
    Format,
    Message,
    Split,
    _valid_token_ids,
    evaluate,
    first_boundary,
)
from .deliforge import (
    DETAIL_CELL_FAILED,
    DETAIL_DECODE_MISMATCH,
    DETAIL_LOCALIZATION_FAILED,
    DETAIL_NO_ENCODING_PAIR,
    MARKERS,
    PAYLOADS,
    DeliforgeBackend,
    authority_gap,
    build_forged_fixtures,
    build_subword_ids,
    decode_equal,
    find_encoding_pair,
    find_insertion_span,
    flip_rates,
    forged_marker_offsets,
    forged_target_index,
    validate_marker_index,
    validate_payload,
)
from .faketurn import (
    DETAIL_MARKER_COUNT_MISMATCH,
    DETAIL_NO_PROVENANCE,
    DETAIL_REF_MISSING,
    SOURCES,
    build_faketurn_fixtures,
    marker_provenance,
    payload_intact,
    payload_ref,
    transfer_gap,
)
from .fewshot import (
    ARMS as FEWSHOT_ARMS,
)
from .fewshot import (
    DETAIL_INSUFFICIENT_DEMO_POOL,
    FEWSHOT_K,
    build_demo_fixtures,
    change_rate,
    demo_permutations,
    order_sensitivity,
    sample_demos,
)
from .hierarchy import POSITIONS, family_of, system_treatment
from .perturbation import PERTURBATION_OPERATORS, perturb_fixtures, variant_plan
from .portfolio import greedy_cover, oracle_rates, per_fixture_oracle_winners
from .position import CONDITIONS as POSITION_CONDITIONS
from .position import primacy_index
from .prefill import (
    PrefillBackend,
    build_prefill_fixtures,
    honors_prefill,
    opens_think_channel,
    validate_prefill_specs,
)
from .refusals import DETECTOR_ID, OPENERS_SHA256, REFUSAL_OPENERS, analyze_candidate
from .scorerlens import LADDER_ID as SCORERLENS_LADDER_ID
from .scorerlens import LADDER_SPEC as SCORERLENS_LADDER_SPEC
from .scorerlens import analyze_candidate as scorerlens_analyze_candidate
from .scorerlens import ladder_sha256 as scorerlens_ladder_sha256
from .scoring import STATUS_OK, TargetScore, score_target_logprobs
from .structured import candidate_adherence, classify_response, validate_schema_def
from .synthhistory import (
    CONDITIONS,
    DELIMITER_LABELS,
    TURN_COUNTS,
    candidate_delimiter_label,
)
from .toolcall import candidate_adherence as toolcall_candidate_adherence
from .toolcall import classify_tool_call
from .verbosity import candidate_verbosity, is_brevity_ideal
from .wscanonical import (
    MUTATIONS,
    WsScanBackend,
    aggregate_mutation_records,
    compare_id_sequences,
    validate_mutations,
)

FAILURE_MODES = (
    "no-boundary-in-N",
    "boundary-text-mismatch",
    "eos-without-boundary",
    "held-out-divergence",
    "backend-error",
)

STOP_SEMANTICS = ("native", "hf_stop_string", "vllm_holdback", "eos_appended")


def classify_observation(obs: dict[str, Any], eos_id: int | None) -> str | None:
    """Classify one non-passing observation; None when it passed.

    Closed taxonomy, deterministic from the stored record. Priority: backend
    failures first, then the tokenization-mismatch diagnostic, then
    self-termination via EOS, then absence of any boundary signal. A wrong
    answer with a found boundary means the expected answer never appeared
    boundary-delimited within the token cap, recorded here as
    ``no-boundary-in-N``; the per-observation records keep the finer
    answer-vs-boundary distinction.

    Input contract: raises ``ValueError`` for a non-dict observation (this
    propagates out of ``failure_taxonomy`` for a non-dict element inside a
    search entry's observation list). A truthy non-iterable ``output_ids``
    raises ``TypeError`` from the EOS-membership check -- but only when that
    branch is actually reached (a pass or a backend failure short-circuits
    first). ``cancelled`` observations group with ``backend-error``: the
    cancellation is operational, not a measurement, and the run decision
    already records "cancelled" separately.
    """
    if type(obs) is not dict:
        raise ValueError("observation must be a dict")
    if obs.get("exact_match") and obs.get("boundary_found"):
        return None
    if obs.get("status") == "failed" or obs.get("raw_reason") == "error":
        return "backend-error"
    if obs.get("status") == "cancelled":
        # Operational, not a measurement: grouped with backend-error; the run
        # decision already records "cancelled" separately.
        return "backend-error"
    if not obs.get("boundary_found"):
        if obs.get("boundary_text_match") is True:
            return "boundary-text-mismatch"
        if eos_id is not None and eos_id in tuple(obs.get("output_ids") or ()):
            return "eos-without-boundary"
        return "no-boundary-in-N"
    return "no-boundary-in-N"


def failure_taxonomy(result: dict[str, Any], eos_id: int | None) -> dict[str, Any]:
    """Per-candidate failure-mode counts for a ``run_experiment`` result.

    Every search candidate is classified (all-pass candidates, e.g. tied
    winners in an abstention, get zero counts); the held-out stage is
    classified only when the winner diverged there, where every non-passing
    held-out observation counts as ``held-out-divergence`` regardless of its
    specific failure -- the divergence itself is the finding. Deterministic
    from the stored observations.

    Input contract asymmetry (deliberate): a non-dict search *entry* is
    skipped, but a non-dict observation inside a search entry's observation
    list propagates ``ValueError`` from ``classify_observation``; held-out
    observations that are not dicts are silently skipped instead (the
    held-out stage only counts dicts that did not pass).
    """
    if type(result) is not dict:
        raise ValueError("result must be a dict")
    per_candidate: dict[str, dict[str, int]] = {}
    for entry in result.get("search") or []:
        if type(entry) is not dict:
            continue
        fmt = entry.get("format")
        name = fmt.get("name") if type(fmt) is dict else None
        if type(name) is not str:
            continue
        counts = dict.fromkeys(FAILURE_MODES, 0)
        observations = entry.get("observations")
        if type(observations) is list:
            for obs in observations:
                mode = classify_observation(obs, eos_id)
                if mode is not None:
                    counts[mode] += 1
        per_candidate[name] = counts
    held_out_counts: dict[str, int] | None = None
    if result.get("decision") == "held_out_failed":
        held = result.get("held_out")
        observations = held.get("observations") if type(held) is dict else None
        held_out_counts = dict.fromkeys(FAILURE_MODES, 0)
        if type(observations) is list:
            for obs in observations:
                if type(obs) is dict and not (obs.get("exact_match") and obs.get("boundary_found")):
                    held_out_counts["held-out-divergence"] += 1
    return {
        "modes": list(FAILURE_MODES),
        "per_candidate": per_candidate,
        "held_out": held_out_counts,
        "eos_id_considered": eos_id is not None,
    }


def _stop_cell(
    backend: Backend,
    fixture: Fixture,
    boundary: tuple[int, ...],
    outcome: StopOutcome,
    *,
    stop_text: str | None = None,
) -> dict[str, Any]:
    """Recompute the lab's pass rule on one emulated truncation.

    ``stop_text`` records the kept-vs-stripped disposition of the stop
    text for text-based stop semantics: ``"kept"`` when the competitor
    keeps the stop string in its returned output (HF always does),
    ``"stripped"`` when it removes it (vLLM's default). It is reported only
    when the stop actually fired -- ``None`` when the stop did not fire or
    the semantic is not text-based. Unavailable outcomes omit the
    ``stop_text`` key entirely. Additive: old callers are unaffected.
    """
    if not outcome.available:
        return {"available": False, "detail": outcome.detail}
    truncated = outcome.truncated_ids
    cut = first_boundary(truncated, boundary)
    answer_ids = truncated if cut is None else truncated[:cut]
    try:
        answer = backend.decode(answer_ids)
    except Exception:  # noqa: BLE001 -- one cell degrades, never the matrix
        return {"available": False, "detail": "answer_decode_unavailable"}
    if type(answer) is not str:
        return {"available": False, "detail": "answer_decode_unavailable"}
    return {
        "available": True,
        "pass": bool(cut is not None and answer == fixture.expected),
        "fired": outcome.fired,
        "detail": outcome.detail,
        "stop_text": stop_text if outcome.fired else None,
    }


def stop_matrix(
    backend: Backend,
    candidate_name: str,
    fixtures: tuple[Fixture, ...],
    observations: tuple[dict[str, Any], ...],
    *,
    boundary: tuple[int, ...],
    eos_id: int | None = None,
    token_bytes: Callable[[int], bytes | None] | None = None,
    vllm_include_stop_str_in_output: bool = False,
    vllm_min_tokens: int = 0,
) -> list[dict[str, Any]]:
    """Per-fixture stop-semantics matrix for one candidate.

    Reuses the candidate's recorded native observations instead of
    regenerating: under greedy decoding, post-hoc truncation is exactly what
    early stopping would have produced (for non-greedy backends it is an
    approximation, noted in the report). Each row recomputes the lab's pass
    rule -- exact answer and token-ID boundary -- on the output as truncated
    by each emulated competitor semantic; ``flips`` lists the semantics
    whose verdict differs from native. The stop text is the decoded
    boundary; a decode failure marks the string-based cells unavailable
    instead of failing. ``token_bytes`` selects the byte-space HF match for
    byte-level/byte-fallback tokenizers (see
    ``backends.emulate_hf_stop_string``); without it the classic string path
    is used. ``vllm_include_stop_str_in_output`` and ``vllm_min_tokens``
    pass through to ``emulate_vllm_holdback`` (defaults match vLLM's
    upstream defaults: strip the stop string, no minimum). Each available
    cell carries a ``stop_text`` field surfacing the kept-vs-stripped
    distinction: ``"kept"`` when the fired semantic keeps the stop text in
    its returned output (HF always does), ``"stripped"`` when it removes it
    (vLLM's default ``include_stop_str_in_output=False``), ``None`` when
    the stop did not fire or the semantic is not text-based. Additive: the
    existing row schema is unchanged apart from this field.
    """
    if type(candidate_name) is not str or not candidate_name:
        raise ValueError("candidate_name must be a nonblank string")
    if (
        type(fixtures) is not tuple
        or type(observations) is not tuple
        or len(fixtures) != len(observations)
        or any(type(item) is not Fixture for item in fixtures)
        or any(type(item) is not dict for item in observations)
    ):
        raise ValueError("fixtures and observations must be aligned tuples")
    if not _valid_token_ids(boundary, nonempty=True):
        raise ValueError("Boundary must be a nonempty tuple of token IDs")
    if eos_id is not None and (type(eos_id) is not int or eos_id < 0):
        raise ValueError("eos_id must be a nonnegative integer or None")
    if token_bytes is not None and not callable(token_bytes):
        raise ValueError("token_bytes must be callable or None")
    if type(vllm_include_stop_str_in_output) is not bool:
        raise ValueError("vllm_include_stop_str_in_output must be a bool")
    if type(vllm_min_tokens) is not int or vllm_min_tokens < 0:
        raise ValueError("vllm_min_tokens must be a nonnegative integer")
    try:
        boundary_text: Any = backend.decode(boundary)
    except Exception:  # noqa: BLE001 -- string cells degrade, never fail
        boundary_text = None
    if type(boundary_text) is not str or not boundary_text:
        boundary_text = None
    rows: list[dict[str, Any]] = []
    for fixture, obs in zip(fixtures, observations, strict=True):
        if obs.get("fixture") != fixture.id:
            raise ValueError("Observation/fixture alignment mismatch")
        output_ids = tuple(obs.get("output_ids") or ())
        prompt_ids = tuple(obs.get("input_ids") or ())
        complete = obs.get("status") == "completed" and all(
            type(token) is int for token in output_ids
        )
        native_pass = bool(obs.get("exact_match") and obs.get("boundary_found"))
        cells: dict[str, dict[str, Any]] = {}
        if complete:
            outcomes = {
                "hf_stop_string": emulate_hf_stop_string(
                    backend, prompt_ids, output_ids, boundary_text or "", token_bytes=token_bytes
                ),
                "vllm_holdback": emulate_vllm_holdback(
                    backend,
                    output_ids,
                    boundary_text or "",
                    include_stop_str_in_output=vllm_include_stop_str_in_output,
                    min_tokens=vllm_min_tokens,
                ),
                "eos_appended": emulate_eos_stop(output_ids, eos_id),
            }
            dispositions = {
                "hf_stop_string": "kept",
                "vllm_holdback": "kept" if vllm_include_stop_str_in_output else "stripped",
                "eos_appended": None,
            }
            for semantic, outcome in outcomes.items():
                cells[semantic] = _stop_cell(
                    backend, fixture, boundary, outcome, stop_text=dispositions[semantic]
                )
        else:
            for semantic in STOP_SEMANTICS[1:]:
                cells[semantic] = {"available": False, "detail": "observation_incomplete"}
        flips = [
            semantic
            for semantic in STOP_SEMANTICS[1:]
            if cells[semantic].get("available") and cells[semantic].get("pass") != native_pass
        ]
        rows.append(
            {
                "fixture": fixture.id,
                "native_pass": native_pass,
                "cells": cells,
                "flips": flips,
            }
        )
    return rows


def selection_robustness(
    backend: Backend,
    result: dict[str, Any],
    cases: tuple[Fixture, ...],
    *,
    boundary: tuple[int, ...],
    boundary_map: dict[str, tuple[int, ...]] | None = None,
    eos_id: int | None = None,
    token_bytes: Callable[[int], bytes | None] | None = None,
    vllm_include_stop_str_in_output: bool = False,
    vllm_min_tokens: int = 0,
) -> dict[str, Any]:
    """Re-apply the selection rule under every stop semantic (post-hoc).

    The stop matrix re-measures only the *winner* under emulated stop
    semantics; the selection rule itself is only ever applied to native
    generations. This answers "would a different protocol have won under my
    serving stack's stop semantics?": for every search candidate it builds
    the stop matrix (tokenizer-only, no model calls), re-derives
    per-candidate pass/fail per semantic (exact answer + boundary under that
    semantic's truncation), and re-applies the conservative
    :func:`_select_winner` rule per semantic.

    Returns ``{"available": True, "per_semantic": {semantic: {"decision",
    "winner"}}, "per_candidate": {name: {semantic: bool}}, "flips": [...]}``
    where ``flips`` lists non-native semantics whose (decision, winner)
    differs from native's. The native row reproduces the run's own
    decision/winner by construction (consistency anchor, pinned by test).
    A semantic whose emulation cannot run for every candidate (e.g. an
    unknown EOS id) is reported as ``"unavailable"`` rather than as a
    failure -- "unknown" must not masquerade as "no_candidate". A
    candidate whose observations do not align is excluded (listed in
    ``excluded_candidates``) so one bad record cannot poison the
    verdicts. A missing/empty search yields ``{"available": False}``.
    Never raises on malformed entries.
    """
    if type(result) is not dict:
        return {"available": False, "detail": "result_not_a_dict"}
    entries = result.get("search")
    if type(entries) is not list or not entries:
        return {"available": False, "detail": "no_search_entries"}
    if type(cases) is not tuple or not cases or any(type(c) is not Fixture for c in cases):
        return {"available": False, "detail": "cases_invalid"}
    if not _valid_token_ids(boundary, nonempty=True):
        return {"available": False, "detail": "boundary_invalid"}

    # Per-candidate boundaries: prefer the entry's own recorded boundary
    # (per-candidate boundaries, Cycle 2), else the boundary_map/default.
    # Per-candidate stop matrices. A candidate whose observations do not
    # align is excluded (one bad record must not poison the verdicts);
    # per_candidate records None for semantics whose cell was unavailable.
    per_candidate: dict[str, dict[str, bool | None]] = {}
    excluded_candidates: list[str] = []
    cell_pass: dict[str, dict[str, bool | None]] = {}
    for entry in entries:
        if type(entry) is not dict or type(entry.get("format")) is not dict:
            continue
        name = entry["format"].get("name")
        if type(name) is not str or not name:
            continue
        observations = entry.get("observations")
        cand_boundary = boundary
        raw_bids = entry.get("boundary_token_ids")
        if type(raw_bids) is list and raw_bids and all(type(t) is int and t >= 0 for t in raw_bids):
            cand_boundary = tuple(raw_bids)
        elif boundary_map and type(boundary_map.get(name)) is tuple:
            cand_boundary = boundary_map[name]
        try:
            if type(observations) is not list or len(observations) != len(cases):
                raise ValueError("observation/case misalignment")
            rows = stop_matrix(
                backend,
                name,
                cases,
                tuple(observations),
                boundary=cand_boundary,
                eos_id=eos_id,
                token_bytes=token_bytes,
                vllm_include_stop_str_in_output=vllm_include_stop_str_in_output,
                vllm_min_tokens=vllm_min_tokens,
            )
        except Exception:  # noqa: BLE001 -- one bad candidate degrades, not the analysis
            rows = []
        if not rows:
            excluded_candidates.append(name)
            continue
        judged: dict[str, bool | None] = {}
        judged["native"] = all(r["native_pass"] for r in rows)
        for semantic in STOP_SEMANTICS[1:]:
            cells = [r["cells"].get(semantic) or {} for r in rows]
            if all(c.get("available") for c in cells):
                judged[semantic] = all(bool(c.get("pass")) for c in cells)
            else:
                judged[semantic] = None
        per_candidate[name] = judged
        cell_pass[name] = judged

    per_semantic: dict[str, dict[str, Any]] = {}
    unavailable_semantics: list[str] = []
    # Mirrors _select_winner(winners, cancelled=False): exactly one passing
    # candidate wins; ties and failures abstain. A semantic is evaluable
    # only when every included candidate has an available cell for it;
    # otherwise the honest verdict is "unavailable", never a phantom
    # no_candidate.
    for semantic in STOP_SEMANTICS:
        if any(cell_pass[name].get(semantic) is None for name in cell_pass):
            per_semantic[semantic] = {"decision": "unavailable", "winner": None}
            if semantic != "native":
                unavailable_semantics.append(semantic)
            continue
        names = [name for name in cell_pass if cell_pass[name].get(semantic)]
        if len(names) == 1:
            per_semantic[semantic] = {"decision": "selected", "winner": names[0]}
        elif names:
            per_semantic[semantic] = {"decision": "ambiguous", "winner": None}
        else:
            per_semantic[semantic] = {"decision": "no_candidate", "winner": None}
    native = per_semantic["native"]
    flips = []
    for semantic in STOP_SEMANTICS[1:]:
        verdict = per_semantic[semantic]
        if verdict["decision"] == "unavailable":
            continue
        if (verdict["decision"], verdict["winner"]) != (native["decision"], native["winner"]):
            flips.append(
                {
                    "semantic": semantic,
                    "decision_from": native["decision"],
                    "decision_to": verdict["decision"],
                    "winner_from": native["winner"],
                    "winner_to": verdict["winner"],
                }
            )
    return {
        "available": True,
        "semantics": list(STOP_SEMANTICS),
        "per_semantic": per_semantic,
        "per_candidate": per_candidate,
        "excluded_candidates": excluded_candidates,
        "unavailable_semantics": unavailable_semantics,
        "flips": flips,
    }


def transfer_portability(
    primary_winner: str | None,
    primary_decision: str,
    candidates: tuple[str, ...],
    transfer: dict[str, dict[str, bool | None]],
) -> dict[str, Any]:
    """Re-apply the selection rule per transfer checkpoint (pure post-hoc).

    Answers "does the primary winner also win on my other checkpoints?":
    for each transfer checkpoint, the conservative selection rule is
    re-applied to that checkpoint's per-candidate pass/fail outcomes, and
    the primary winner's fate there is recorded.

    ``transfer[checkpoint_id][candidate_name]`` is True when the candidate
    passed every fixture at that checkpoint, False when it did not, and
    None when it could not be evaluated there (e.g. boundary IDs outside
    the checkpoint's vocabulary -- fail-closed, never silently skipped).

    Returns ``{"available": True, "primary_winner": ..., "checkpoints":
    [...], "per_checkpoint": {ckpt: {"decision", "winner"}}, "matrix":
    {candidate: {ckpt: verdict}}, "transfer_verdict": ...}``. The per-cell
    verdict is ``"transfers"`` (primary winner is the unique winner
    there), ``"ties_there"``, ``"fails_there"``, or
    ``"checkpoint_incompatible"``; non-winner candidates get
    ``"wins_there"`` instead of ``"transfers"``. ``transfer_verdict`` is
    ``"full"`` / ``"partial"`` / ``"none"`` over the evaluable
    checkpoints, or ``"unavailable"`` when there is no primary winner or
    no checkpoint to transfer to. Never raises on malformed input:
    returns ``{"available": False}``.
    """
    if (
        not isinstance(candidates, tuple)
        or not candidates
        or not all(type(c) is str and c for c in candidates)
        or len(set(candidates)) != len(candidates)
        or type(transfer) is not dict
        or not all(type(k) is str and k for k in transfer)
    ):
        return {"available": False, "detail": "invalid_candidates_or_transfer"}
    if primary_winner is not None and (
        type(primary_winner) is not str or primary_winner not in candidates
    ):
        return {"available": False, "detail": "primary_winner_unknown"}
    if type(primary_decision) is not str or not primary_decision:
        return {"available": False, "detail": "primary_decision_invalid"}
    checkpoints = sorted(transfer)
    for ckpt in checkpoints:
        per_cand = transfer[ckpt]
        if type(per_cand) is not dict:
            return {"available": False, "detail": f"checkpoint_{ckpt}_invalid"}
        for cand in candidates:
            val = per_cand.get(cand)
            if val is not None and type(val) is not bool:
                return {"available": False, "detail": f"checkpoint_{ckpt}_invalid"}

    per_checkpoint: dict[str, dict[str, Any]] = {}
    matrix: dict[str, dict[str, str]] = {c: {} for c in candidates}
    for ckpt in checkpoints:
        per_cand = transfer[ckpt]
        # A checkpoint where no candidate could be evaluated is
        # unavailable, not a no_candidate verdict.
        if all(per_cand.get(c) is None for c in candidates):
            per_checkpoint[ckpt] = {"decision": "unavailable", "winner": None}
            for cand in candidates:
                matrix[cand][ckpt] = "checkpoint_incompatible"
            continue
        winners = [c for c in candidates if per_cand.get(c) is True]
        # Mirrors _select_winner(winners, cancelled=False).
        if len(winners) == 1:
            per_checkpoint[ckpt] = {"decision": "selected", "winner": winners[0]}
        elif winners:
            per_checkpoint[ckpt] = {"decision": "ambiguous", "winner": None}
        else:
            per_checkpoint[ckpt] = {"decision": "no_candidate", "winner": None}
        for cand in candidates:
            val = per_cand.get(cand)
            if val is None:
                matrix[cand][ckpt] = "checkpoint_incompatible"
            elif val is False:
                matrix[cand][ckpt] = "fails_there"
            elif cand == primary_winner:
                matrix[cand][ckpt] = "transfers" if winners == [cand] else "ties_there"
            else:
                matrix[cand][ckpt] = "wins_there" if winners == [cand] else "ties_there"

    if primary_winner is None or not checkpoints:
        transfer_verdict = "unavailable"
    else:
        outcomes = [
            matrix[primary_winner][ckpt]
            for ckpt in checkpoints
            if matrix[primary_winner][ckpt] != "checkpoint_incompatible"
        ]
        if not outcomes:
            transfer_verdict = "unavailable"
        elif all(o == "transfers" for o in outcomes):
            transfer_verdict = "full"
        elif any(o == "transfers" for o in outcomes):
            transfer_verdict = "partial"
        else:
            transfer_verdict = "none"
    return {
        "available": True,
        "primary_winner": primary_winner,
        "primary_decision": primary_decision,
        "checkpoints": checkpoints,
        "per_checkpoint": per_checkpoint,
        "matrix": matrix,
        "transfer_verdict": transfer_verdict,
    }


def transfer_stage(
    backend_factory: Callable[[str], Backend],
    checkpoint_ids: tuple[str, ...],
    formats: tuple[Format, ...],
    cases: tuple[Fixture, ...],
    *,
    boundary: tuple[int, ...],
    boundary_map: dict[str, tuple[int, ...]] | None = None,
    max_new_tokens: int = 64,
    seed: int = 0,
) -> dict[str, dict[str, bool | None]]:
    """Evaluate every candidate against each transfer checkpoint.

    Re-runs ``core.evaluate()`` for every search candidate against each
    additional local checkpoint, on the search split only (bounded, same
    fixture limit), with the same boundary and max_new_tokens. Greedy,
    exact. ``backend_factory(checkpoint_id)`` builds a backend per
    checkpoint; any backend construction or evaluation failure records
    None for that (checkpoint, candidate) cell -- fail-closed, never
    silently skipped.

    Returns ``transfer[checkpoint_id][candidate_name]``: True when the
    candidate passed every fixture there, False when it did not, None
    when it could not be evaluated. Pure in its return contract (no
    global state), though backend construction performs I/O.
    """
    if not callable(backend_factory):
        raise ValueError("backend_factory must be callable")
    if (
        type(checkpoint_ids) is not tuple
        or not 1 <= len(checkpoint_ids) <= 4
        or not all(type(c) is str and c for c in checkpoint_ids)
        or len(set(checkpoint_ids)) != len(checkpoint_ids)
    ):
        raise ValueError("checkpoint_ids must be 1-4 unique nonblank strings")
    if type(formats) is not tuple or not formats:
        raise ValueError("formats must be a nonempty tuple")
    if type(cases) is not tuple or not cases:
        raise ValueError("cases must be a nonempty tuple")
    if not _valid_token_ids(boundary, nonempty=True):
        raise ValueError("Boundary must be a nonempty tuple of token IDs")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 4096:
        raise ValueError("max_new_tokens must be an integer between 1 and 4096")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    boundaries = _resolve_boundaries(formats, boundary, boundary_map)
    transfer: dict[str, dict[str, bool | None]] = {}
    for ckpt in checkpoint_ids:
        per_candidate: dict[str, bool | None] = {}
        try:
            backend = backend_factory(ckpt)
        except Exception:  # noqa: BLE001 -- whole checkpoint degrades, not the stage
            backend = None
        for fmt in formats:
            if backend is None:
                per_candidate[fmt.name] = None
                continue
            try:
                report = evaluate(
                    backend,
                    fmt,
                    cases,
                    boundary=boundaries[fmt.name],
                    max_new_tokens=max_new_tokens,
                    seed=seed,
                )
            except Exception:  # noqa: BLE001 -- one cell degrades, not the stage
                per_candidate[fmt.name] = None
                continue
            if not report.aggregate_eligible:
                per_candidate[fmt.name] = None
                continue
            passed = sum(o.exact_match and o.boundary_found for o in report.observations)
            per_candidate[fmt.name] = passed == report.planned_count
        transfer[ckpt] = per_candidate
    return transfer


def perturbation_stage(
    backend: Backend,
    formats: tuple[Format, ...],
    cases: tuple[Fixture, ...],
    *,
    n_variants: int,
    operators: tuple[str, ...] = PERTURBATION_OPERATORS,
    seed: int = 0,
    boundary: tuple[int, ...],
    boundary_map: dict[str, tuple[int, ...]] | None = None,
    max_new_tokens: int = 64,
) -> dict[str, Any]:
    """Re-evaluate every candidate on perturbed fixture variants.

    For each of ``n_variants`` deterministically perturbed fixture
    variants (operator assigned round-robin by
    :func:`perturbation.variant_plan`, seeded from ``seed``), every
    search candidate is re-evaluated with ``core.evaluate()`` on the
    search split only, with the same boundary and max_new_tokens.
    Greedy, exact. Only fixture *message content* is perturbed --
    ids, groups, and expected answers are preserved, so per-variant
    verdicts stay comparable to the primary run.

    Any backend construction or evaluation failure records None for
    that (variant, candidate) cell -- fail-closed, never silently
    skipped. ``changed_fixtures`` counts the fixtures whose message
    content actually changed under the variant's operator, so a
    no-op perturbation is reported honestly instead of inflating
    stability.

    Returns ``{"n_variants": ..., "operators": [...], "variants":
    {variant_id: {"operator", "seed", "changed_fixtures",
    "n_fixtures", "outcomes": {candidate: bool | None}}}}``. Pure in
    its return contract (no global state), though evaluation performs
    I/O.
    """
    if backend is None:
        raise ValueError("backend must not be None")
    if type(formats) is not tuple or not formats:
        raise ValueError("formats must be a nonempty tuple")
    if type(cases) is not tuple or not cases:
        raise ValueError("cases must be a nonempty tuple")
    if not _valid_token_ids(boundary, nonempty=True):
        raise ValueError("Boundary must be a nonempty tuple of token IDs")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 4096:
        raise ValueError("max_new_tokens must be an integer between 1 and 4096")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    plan = variant_plan(n_variants, operators, seed=seed)
    boundaries = _resolve_boundaries(formats, boundary, boundary_map)
    variants: dict[str, dict[str, Any]] = {}
    for spec in plan:
        variant_id = spec["variant_id"]
        operator = spec["operator"]
        variant_seed = spec["seed"]
        perturbed = perturb_fixtures(cases, operator, seed=variant_seed)
        changed = sum(
            any(
                new.content != old.content
                for new, old in zip(p_case.messages, case.messages, strict=True)
            )
            for p_case, case in zip(perturbed, cases, strict=True)
        )
        outcomes: dict[str, bool | None] = {}
        for fmt in formats:
            try:
                report = evaluate(
                    backend,
                    fmt,
                    perturbed,
                    boundary=boundaries[fmt.name],
                    max_new_tokens=max_new_tokens,
                    seed=seed,
                )
            except Exception:  # noqa: BLE001 -- one cell degrades, not the stage
                outcomes[fmt.name] = None
                continue
            if not report.aggregate_eligible:
                outcomes[fmt.name] = None
                continue
            passed = sum(o.exact_match and o.boundary_found for o in report.observations)
            outcomes[fmt.name] = passed == report.planned_count
        variants[variant_id] = {
            "operator": operator,
            "seed": variant_seed,
            "changed_fixtures": changed,
            "n_fixtures": len(cases),
            "outcomes": outcomes,
        }
    return {
        "n_variants": n_variants,
        "operators": list(operators),
        "variants": variants,
    }


def perturbation_robustness(
    primary_winner: str | None,
    primary_decision: str,
    candidates: tuple[str, ...],
    stage: dict[str, Any],
) -> dict[str, Any]:
    """Re-apply the selection rule per perturbation variant (pure post-hoc).

    Answers "does the winner survive fixture phrasing perturbations?":
    for each perturbed variant the conservative selection rule is
    re-applied to that variant's per-candidate pass/fail outcomes, and
    the primary (decision, winner) is compared against every variant.

    A variant where no candidate could be evaluated is reported as
    ``"unavailable"``, never as a phantom ``no_candidate`` --
    "unknown" must not masquerade as a verdict. ``stability``
    summarizes how many evaluable variants retain the primary winner
    and the primary decision; ``per_operator`` breaks the retention
    down by perturbation operator with the mean fixture-change rate,
    so a reader can see *which* phrasing noise the winner is fragile
    to. ``flips`` lists the evaluable variants whose (decision,
    winner) differs from the primary run.

    Returns ``{"available": True, "primary_winner": ...,
    "primary_decision": ..., "n_variants": ..., "operators": [...],
    "per_variant": {variant_id: {"operator", "decision", "winner",
    "changed_fixtures", "changed_rate"}}, "flips": [...],
    "stability": {...}, "per_operator": {...}}``. Never raises on
    malformed input: returns ``{"available": False}``.
    """
    if (
        not isinstance(candidates, tuple)
        or not candidates
        or not all(type(c) is str and c for c in candidates)
        or len(set(candidates)) != len(candidates)
    ):
        return {"available": False, "detail": "invalid_candidates"}
    if primary_winner is not None and (
        type(primary_winner) is not str or primary_winner not in candidates
    ):
        return {"available": False, "detail": "primary_winner_unknown"}
    if type(primary_decision) is not str or not primary_decision:
        return {"available": False, "detail": "primary_decision_invalid"}
    if type(stage) is not dict:
        return {"available": False, "detail": "stage_not_a_dict"}
    raw_variants = stage.get("variants")
    if (
        type(raw_variants) is not dict
        or not raw_variants
        or not all(type(k) is str and k for k in raw_variants)
    ):
        return {"available": False, "detail": "variants_invalid"}
    for vid, spec in raw_variants.items():
        if type(spec) is not dict or type(spec.get("operator")) is not str:
            return {"available": False, "detail": f"variant_{vid}_invalid"}
        outcomes = spec.get("outcomes")
        if type(outcomes) is not dict:
            return {"available": False, "detail": f"variant_{vid}_invalid"}
        for cand in candidates:
            val = outcomes.get(cand)
            if val is not None and type(val) is not bool:
                return {"available": False, "detail": f"variant_{vid}_invalid"}

    per_variant: dict[str, dict[str, Any]] = {}
    for vid in sorted(raw_variants):
        spec = raw_variants[vid]
        outcomes = spec["outcomes"]
        vals = [outcomes.get(c) for c in candidates]
        if all(v is None for v in vals):
            per_variant[vid] = {
                "operator": spec["operator"],
                "decision": "unavailable",
                "winner": None,
                "changed_fixtures": spec.get("changed_fixtures"),
                "changed_rate": _changed_rate(spec),
            }
            continue
        winners = [c for c, v in zip(candidates, vals, strict=True) if v is True]
        # Mirrors _select_winner(winners, cancelled=False): exactly one
        # passing candidate wins; ties and failures abstain.
        if len(winners) == 1:
            decision, winner = "selected", winners[0]
        elif winners:
            decision, winner = "ambiguous", None
        else:
            decision, winner = "no_candidate", None
        per_variant[vid] = {
            "operator": spec["operator"],
            "decision": decision,
            "winner": winner,
            "changed_fixtures": spec.get("changed_fixtures"),
            "changed_rate": _changed_rate(spec),
        }

    evaluable = [v for v in per_variant.values() if v["decision"] != "unavailable"]
    flips = []
    for vid in sorted(per_variant):
        verdict = per_variant[vid]
        if verdict["decision"] == "unavailable":
            continue
        if (verdict["decision"], verdict["winner"]) != (primary_decision, primary_winner):
            flips.append(
                {
                    "variant": vid,
                    "operator": verdict["operator"],
                    "decision_from": primary_decision,
                    "decision_to": verdict["decision"],
                    "winner_from": primary_winner,
                    "winner_to": verdict["winner"],
                }
            )
    winner_retained = (
        sum(1 for v in evaluable if v["winner"] == primary_winner)
        if primary_winner is not None
        else None
    )
    stability = {
        "n_variants": len(per_variant),
        "n_evaluable": len(evaluable),
        "n_unavailable": len(per_variant) - len(evaluable),
        "winner_retained": winner_retained,
        "decision_retained": sum(1 for v in evaluable if v["decision"] == primary_decision),
    }
    per_operator: dict[str, dict[str, Any]] = {}
    for vid in sorted(per_variant):
        verdict = per_variant[vid]
        op = verdict["operator"]
        agg = per_operator.setdefault(
            op, {"variants": 0, "evaluable": 0, "winner_retained": 0, "changed_rates": []}
        )
        agg["variants"] += 1
        if verdict["decision"] != "unavailable":
            agg["evaluable"] += 1
            if primary_winner is not None and verdict["winner"] == primary_winner:
                agg["winner_retained"] += 1
        if type(verdict["changed_rate"]) is float:
            agg["changed_rates"].append(verdict["changed_rate"])
    for op, agg in per_operator.items():
        rates = agg.pop("changed_rates")
        per_operator[op] = {
            "variants": agg["variants"],
            "evaluable": agg["evaluable"],
            "winner_retained": agg["winner_retained"] if primary_winner is not None else None,
            "mean_changed_rate": (sum(rates) / len(rates)) if rates else None,
        }
    raw_operators = stage.get("operators")
    operators = (
        list(raw_operators)
        if type(raw_operators) is list and all(type(o) is str for o in raw_operators)
        else sorted(per_operator)
    )
    return {
        "available": True,
        "primary_winner": primary_winner,
        "primary_decision": primary_decision,
        "n_variants": len(per_variant),
        "operators": operators,
        "per_variant": per_variant,
        "flips": flips,
        "stability": stability,
        "per_operator": per_operator,
    }


def healing_stage(
    backend: Backend,
    formats: tuple[Format, ...],
    cases: tuple[Fixture, ...],
    *,
    boundary: tuple[int, ...],
    boundary_map: dict[str, tuple[int, ...]] | None = None,
    max_new_tokens: int = 64,
    seed: int = 0,
) -> dict[str, Any]:
    """Re-evaluate every candidate with token-healed prompts.

    Wraps ``backend`` in :class:`backends.TokenHealingBackend` (raising
    ``ValueError`` when the backend cannot heal) and re-runs
    ``core.evaluate()`` for every search candidate on the search split
    only, with the same boundary and max_new_tokens. Greedy, exact.
    Only the prompt tail is healed -- fixtures, boundaries, and scoring
    are unchanged, so healed verdicts stay comparable to the primary
    run.

    Any evaluation failure records None for that candidate --
    fail-closed, never silently skipped. ``n_healed`` / ``n_fallback``
    report how many prompts actually healed vs fell back to native ids.

    Returns ``{"outcomes": {candidate: {"passed": bool | None,
    "per_fixture": {fixture_id: bool}, "n_passed": int, "n_total":
    int}}, "fixture_ids": [...], "n_healed": int, "n_fallback": int,
    "n_fixtures": int}``. Pure in its return contract (no global
    state), though evaluation performs I/O.
    """
    if backend is None:
        raise ValueError("backend must not be None")
    if type(formats) is not tuple or not formats:
        raise ValueError("formats must be a nonempty tuple")
    if type(cases) is not tuple or not cases:
        raise ValueError("cases must be a nonempty tuple")
    if not _valid_token_ids(boundary, nonempty=True):
        raise ValueError("Boundary must be a nonempty tuple of token IDs")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 4096:
        raise ValueError("max_new_tokens must be an integer between 1 and 4096")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    healed_backend = TokenHealingBackend(backend)
    boundaries = _resolve_boundaries(formats, boundary, boundary_map)
    outcomes: dict[str, dict[str, Any]] = {}
    for fmt in formats:
        try:
            report = evaluate(
                healed_backend,
                fmt,
                cases,
                boundary=boundaries[fmt.name],
                max_new_tokens=max_new_tokens,
                seed=seed,
            )
        except Exception:  # noqa: BLE001 -- one candidate degrades, not the stage
            outcomes[fmt.name] = {
                "passed": None,
                "per_fixture": {},
                "n_passed": 0,
                "n_total": 0,
            }
            continue
        if not report.aggregate_eligible:
            outcomes[fmt.name] = {
                "passed": None,
                "per_fixture": {},
                "n_passed": 0,
                "n_total": 0,
            }
            continue
        per_fixture = {
            str(o.fixture): bool(o.exact_match and o.boundary_found) for o in report.observations
        }
        n_passed = sum(per_fixture.values())
        outcomes[fmt.name] = {
            "passed": n_passed == len(per_fixture) and bool(per_fixture),
            "per_fixture": per_fixture,
            "n_passed": n_passed,
            "n_total": len(per_fixture),
        }
    return {
        "outcomes": outcomes,
        "fixture_ids": [str(case.id) for case in cases],
        "n_healed": healed_backend.n_healed,
        "n_fallback": healed_backend.n_fallback,
        "n_fixtures": len(cases),
    }


def native_search_outcomes(
    search_entries: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    candidates: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    """Per-candidate pass/fail extracted from the primary search section.

    Pure post-hoc helper: for each candidate, ``passed`` is True when
    every search observation passed (exact answer and explicit boundary),
    False when at least one failed, None when the candidate has no usable
    observations or was not aggregate-eligible. ``per_fixture`` maps
    fixture id to its verdict. Never raises on malformed entries: they
    degrade to ``{"passed": None, "per_fixture": {}}``.
    """
    by_name: dict[str, dict[str, Any]] = {}
    entries = list(search_entries) if type(search_entries) in (list, tuple) else []
    for entry in entries:
        if type(entry) is not dict:
            continue
        fmt = entry.get("format")
        name = fmt.get("name") if type(fmt) is dict else None
        if type(name) is str and name:
            by_name[name] = entry
    outcomes: dict[str, dict[str, Any]] = {}
    for cand in candidates:
        cand_entry = by_name.get(cand)
        observations = cand_entry.get("observations") if type(cand_entry) is dict else None
        summary = cand_entry.get("summary") if type(cand_entry) is dict else None
        if type(observations) is not list or not observations:
            outcomes[cand] = {"passed": None, "per_fixture": {}}
            continue
        if type(summary) is dict and summary.get("aggregate_eligible") is False:
            outcomes[cand] = {"passed": None, "per_fixture": {}}
            continue
        per_fixture: dict[str, bool] = {}
        usable = True
        for obs in observations:
            if type(obs) is not dict:
                usable = False
                break
            fid = obs.get("fixture")
            if type(fid) is not str or not fid:
                usable = False
                break
            exact = obs.get("exact_match")
            boundary = obs.get("boundary_found")
            if type(exact) is not bool or type(boundary) is not bool:
                usable = False
                break
            per_fixture[fid] = exact and boundary
        if not usable or not per_fixture:
            outcomes[cand] = {"passed": None, "per_fixture": {}}
        else:
            outcomes[cand] = {
                "passed": all(per_fixture.values()),
                "per_fixture": per_fixture,
            }
    return outcomes


def healing_delta(
    primary_winner: str | None,
    primary_decision: str,
    candidates: tuple[str, ...],
    native_outcomes: dict[str, Any],
    healed: dict[str, Any],
    boundary_audit: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compare native vs token-healed selection verdicts (pure post-hoc).

    Answers "does token healing change the winning protocol -- and does
    the lab's own boundary-canonicality audit predict which candidates
    flip?": the conservative selection rule is re-applied to the healed
    per-candidate outcomes; the healed (decision, winner) is compared
    against the primary run; per-candidate verdict deltas and per-fixture
    flips are listed; and the falsification table checks the lab's own
    prediction -- a candidate's verdict may only change under healing
    when the boundary audit flagged it seam-fragile -- candidate by
    candidate, so a negative result is as informative as a positive one.

    ``native_outcomes`` is the :func:`native_search_outcomes` mapping;
    ``healed`` is the :func:`healing_stage` payload; ``boundary_audit``
    is the ``audit_search_observations`` payload (or None). Never raises
    on malformed input: returns ``{"available": False}``.
    """
    if (
        not isinstance(candidates, tuple)
        or not candidates
        or not all(type(c) is str and c for c in candidates)
        or len(set(candidates)) != len(candidates)
    ):
        return {"available": False, "detail": "invalid_candidates"}
    if primary_winner is not None and (
        type(primary_winner) is not str or primary_winner not in candidates
    ):
        return {"available": False, "detail": "primary_winner_unknown"}
    if type(primary_decision) is not str or not primary_decision:
        return {"available": False, "detail": "primary_decision_invalid"}
    if type(native_outcomes) is not dict or type(healed) is not dict:
        return {"available": False, "detail": "outcomes_invalid"}
    healed_outcomes = healed.get("outcomes")
    if type(healed_outcomes) is not dict:
        return {"available": False, "detail": "healed_outcomes_invalid"}
    native_passed: dict[str, bool | None] = {}
    native_per_fixture: dict[str, dict[str, bool]] = {}
    for cand in candidates:
        entry = native_outcomes.get(cand)
        if type(entry) is not dict:
            native_passed[cand] = None
            native_per_fixture[cand] = {}
            continue
        passed = entry.get("passed")
        native_passed[cand] = passed if type(passed) is bool else None
        per_fixture = entry.get("per_fixture")
        native_per_fixture[cand] = (
            {k: v for k, v in per_fixture.items() if type(k) is str and type(v) is bool}
            if type(per_fixture) is dict
            else {}
        )
    healed_passed: dict[str, bool | None] = {}
    healed_per_fixture: dict[str, dict[str, bool]] = {}
    for cand in candidates:
        entry = healed_outcomes.get(cand)
        if type(entry) is not dict:
            return {"available": False, "detail": f"healed_candidate_{cand}_invalid"}
        passed = entry.get("passed")
        if passed is not None and type(passed) is not bool:
            return {"available": False, "detail": f"healed_candidate_{cand}_invalid"}
        healed_passed[cand] = passed
        per_fixture = entry.get("per_fixture")
        healed_per_fixture[cand] = (
            {k: v for k, v in per_fixture.items() if type(k) is str and type(v) is bool}
            if type(per_fixture) is dict
            else {}
        )

    if all(v is None for v in healed_passed.values()):
        healed_decision, healed_winner = "unavailable", None
    else:
        winners = [c for c in candidates if healed_passed[c] is True]
        # Mirrors _select_winner(winners, cancelled=False).
        if len(winners) == 1:
            healed_decision, healed_winner = "selected", winners[0]
        elif winners:
            healed_decision, healed_winner = "ambiguous", None
        else:
            healed_decision, healed_winner = "no_candidate", None
    flipped = (healed_decision, healed_winner) != (primary_decision, primary_winner)

    per_candidate: dict[str, dict[str, Any]] = {}
    for cand in candidates:
        native = native_passed[cand]
        healed_v = healed_passed[cand]
        if type(native) is bool and type(healed_v) is bool:
            verdict_changed: bool | None = native != healed_v
        else:
            verdict_changed = None
        per_candidate[cand] = {
            "native": native,
            "healed": healed_v,
            "verdict_changed": verdict_changed,
        }

    fixture_flips: list[dict[str, Any]] = []
    for cand in candidates:
        native_pf = native_per_fixture[cand]
        healed_pf = healed_per_fixture[cand]
        for fid in sorted(set(native_pf) & set(healed_pf)):
            if native_pf[fid] != healed_pf[fid]:
                fixture_flips.append(
                    {
                        "candidate": cand,
                        "fixture": fid,
                        "native": native_pf[fid],
                        "healed": healed_pf[fid],
                    }
                )
    fixture_flips.sort(key=lambda e: (e["candidate"], e["fixture"]))

    audit_per_candidate = (
        boundary_audit.get("per_candidate") if type(boundary_audit) is dict else None
    )
    falsification_per_candidate: dict[str, dict[str, Any]] = {}
    n_comparable = 0
    n_holds = 0
    for cand in candidates:
        audit_entry = audit_per_candidate.get(cand) if type(audit_per_candidate) is dict else None
        seam_fragile: bool | None = None
        if type(audit_entry) is dict:
            n_problem = audit_entry.get("n_problem")
            n_usable = audit_entry.get("n_usable")
            if type(n_problem) is int and type(n_usable) is int and n_usable > 0:
                seam_fragile = n_problem > 0
        verdict_changed = per_candidate[cand]["verdict_changed"]
        prediction_holds: bool | None = None
        if type(seam_fragile) is bool and type(verdict_changed) is bool:
            # The falsifiable prediction (one direction): a verdict may
            # only change under healing when the boundary audit flagged
            # the candidate seam-fragile. A fragile candidate that does
            # not flip is uninformative (the model may simply re-choose
            # the same tail); a canonical candidate that flips falsifies
            # the audit's predictive claim.
            prediction_holds = (not verdict_changed) or seam_fragile
            n_comparable += 1
            if prediction_holds:
                n_holds += 1
        falsification_per_candidate[cand] = {
            "seam_fragile": seam_fragile,
            "verdict_changed": verdict_changed,
            "prediction_holds": prediction_holds,
        }
    falsification = {
        "per_candidate": falsification_per_candidate,
        "n_comparable": n_comparable,
        "n_holds": n_holds,
        "n_violations": n_comparable - n_holds,
        "prediction_holds": (n_holds == n_comparable) if n_comparable else None,
    }

    n_healed = healed.get("n_healed")
    n_fallback = healed.get("n_fallback")
    return {
        "available": True,
        "primary_winner": primary_winner,
        "primary_decision": primary_decision,
        "healed_decision": healed_decision,
        "healed_winner": healed_winner,
        "flipped": flipped,
        "per_candidate": per_candidate,
        "fixture_flips": fixture_flips,
        "falsification": falsification,
        "n_healed": n_healed if type(n_healed) is int else None,
        "n_fallback": n_fallback if type(n_fallback) is int else None,
    }


def hierarchy_stage(
    backend: Backend,
    formats: tuple[Format, ...],
    conflict_fixtures: tuple[Fixture, ...],
    expectations: dict[str, dict[str, str]],
    *,
    position: str,
    boundary: tuple[int, ...],
    boundary_map: dict[str, tuple[int, ...]] | None = None,
    max_new_tokens: int = 64,
    seed: int = 0,
) -> dict[str, Any]:
    """Re-evaluate every candidate on instruction-conflict fixtures.

    For one privilege-block ``position`` (see :mod:`hierarchy`), every
    search candidate is re-evaluated with ``core.evaluate()`` on the
    given conflict fixtures, with the same boundary and max_new_tokens.
    Greedy, exact. Each fixture's ``expected`` is the *system* answer, so
    the pipeline's ``exact_match`` already means "followed the privileged
    instruction"; the stage additionally scores the *user*-side match per
    observation from ``expectations`` (exact string equality, the same
    semantics as ``Observation.exact_match``).

    Any backend construction or evaluation failure records None for that
    candidate cell -- fail-closed, never silently skipped. ``pass`` is
    the conservative per-candidate verdict: every conflict fixture
    system-followed with its boundary found (mirrors the perturbation
    stage's all-pass rule), so the selection rule can be re-applied
    unchanged by :func:`hierarchy_adherence`.

    Returns ``{"position": ..., "n_fixtures": ..., "per_candidate":
    {name: {"n", "system", "user", "neither", "pass"} | None}}``.
    """
    if backend is None:
        raise ValueError("backend must not be None")
    if type(formats) is not tuple or not formats:
        raise ValueError("formats must be a nonempty tuple")
    if type(conflict_fixtures) is not tuple or not conflict_fixtures:
        raise ValueError("conflict_fixtures must be a nonempty tuple")
    if any(type(case) is not Fixture for case in conflict_fixtures):
        raise ValueError("conflict_fixtures must contain Fixture values")
    if type(expectations) is not dict:
        raise ValueError("expectations must be a dict")
    for case in conflict_fixtures:
        exp = expectations.get(case.id)
        if (
            type(exp) is not dict
            or type(exp.get("expected_system")) is not str
            or type(exp.get("expected_user")) is not str
        ):
            raise ValueError(f"expectations missing valid entry for fixture {case.id!r}")
    if position not in POSITIONS:
        raise ValueError(f"position must be one of {POSITIONS}")
    if not _valid_token_ids(boundary, nonempty=True):
        raise ValueError("Boundary must be a nonempty tuple of token IDs")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 4096:
        raise ValueError("max_new_tokens must be an integer between 1 and 4096")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    boundaries = _resolve_boundaries(formats, boundary, boundary_map)
    per_candidate: dict[str, dict[str, Any] | None] = {}
    for fmt in formats:
        try:
            report = evaluate(
                backend,
                fmt,
                conflict_fixtures,
                boundary=boundaries[fmt.name],
                max_new_tokens=max_new_tokens,
                seed=seed,
            )
        except Exception:  # noqa: BLE001 -- one cell degrades, not the stage
            per_candidate[fmt.name] = None
            continue
        if not report.aggregate_eligible:
            per_candidate[fmt.name] = None
            continue
        system = user = neither = 0
        all_pass = True
        for obs in report.observations:
            exp = expectations[obs.fixture]
            follows_system = obs.answer == exp["expected_system"]
            follows_user = obs.answer == exp["expected_user"]
            if follows_system:
                system += 1
            elif follows_user:
                user += 1
            else:
                neither += 1
            if not (follows_system and obs.boundary_found):
                all_pass = False
        per_candidate[fmt.name] = {
            "n": len(conflict_fixtures),
            "system": system,
            "user": user,
            "neither": neither,
            "pass": all_pass,
        }
    return {
        "position": position,
        "n_fixtures": len(conflict_fixtures),
        "per_candidate": per_candidate,
    }


def _hierarchy_entry_valid(entry: Any, n_fixtures: int) -> bool:
    """Structural check for one candidate cell of a hierarchy stage."""
    if entry is None:
        return True
    if type(entry) is not dict:
        return False
    if entry.get("n") != n_fixtures:
        return False
    system = entry.get("system")
    user = entry.get("user")
    neither = entry.get("neither")
    if (
        type(system) is not int
        or system < 0
        or type(user) is not int
        or user < 0
        or type(neither) is not int
        or neither < 0
    ):
        return False
    if system + user + neither != n_fixtures:
        return False
    return type(entry.get("pass")) is bool


def hierarchy_adherence(
    primary_winner: str | None,
    primary_decision: str,
    candidates: tuple[str, ...],
    stages: dict[str, Any],
) -> dict[str, Any]:
    """Re-apply the selection rule per conflict position (pure post-hoc).

    Answers "does the winner survive instruction conflict?": for each
    privilege-block position the conservative selection rule is
    re-applied to that condition's per-candidate pass/fail outcomes, and
    the primary (decision, winner) is compared against every condition.

    Per candidate per condition the adherence rates (system / user /
    neither) are reported; ``position_effect`` compares the two
    positions (primacy vs recency); ``group_adherence`` tests the
    pre-registered structural hypothesis from :mod:`hierarchy` --
    families whose real templates give system content a dedicated frame
    vs families that fold it -- as mean system-adherence per
    documentation-derived label group (a correlation readout, not a
    causal claim). ``flips`` lists the conditions whose (decision,
    winner) differs from the primary run.

    Returns ``{"available": True, ...}``. Never raises on malformed
    input: returns ``{"available": False}`` with a ``detail`` code.
    """
    if (
        not isinstance(candidates, tuple)
        or not candidates
        or not all(type(c) is str and c for c in candidates)
        or len(set(candidates)) != len(candidates)
    ):
        return {"available": False, "detail": "invalid_candidates"}
    if primary_winner is not None and (
        type(primary_winner) is not str or primary_winner not in candidates
    ):
        return {"available": False, "detail": "primary_winner_unknown"}
    if type(primary_decision) is not str or not primary_decision:
        return {"available": False, "detail": "primary_decision_invalid"}
    if type(stages) is not dict or not stages or not all(type(k) is str for k in stages):
        return {"available": False, "detail": "stages_invalid"}
    n_fixtures = -1  # sentinel: stages is guaranteed nonempty above
    for position, stage in stages.items():
        if type(stage) is not dict:
            return {"available": False, "detail": f"stage_{position}_invalid"}
        if type(stage.get("position")) is not str:
            return {"available": False, "detail": f"stage_{position}_invalid"}
        count = stage.get("n_fixtures")
        if type(count) is not int or count < 0:
            return {"available": False, "detail": f"stage_{position}_invalid"}
        if n_fixtures < 0:
            n_fixtures = count
        elif n_fixtures != count:
            return {"available": False, "detail": "fixture_count_mismatch"}
        per_candidate = stage.get("per_candidate")
        if type(per_candidate) is not dict or not all(type(k) is str for k in per_candidate):
            return {"available": False, "detail": f"stage_{position}_invalid"}
        for _cand, entry in per_candidate.items():
            if not _hierarchy_entry_valid(entry, count):
                return {"available": False, "detail": f"stage_{position}_invalid"}

    def _rates(entry: dict[str, Any] | None) -> dict[str, Any] | None:
        if entry is None:
            return None
        n = entry["n"]
        return {
            "system_rate": entry["system"] / n if n else None,
            "user_rate": entry["user"] / n if n else None,
            "neither_rate": entry["neither"] / n if n else None,
            "pass": entry["pass"],
        }

    per_condition: dict[str, dict[str, Any]] = {}
    for position in sorted(stages):
        stage = stages[position]
        per_candidate = stage["per_candidate"]
        entries = [per_candidate.get(c) for c in candidates]
        evaluable = [e for e in entries if e is not None]
        winners = [
            c for c, e in zip(candidates, entries, strict=True) if e is not None and e["pass"]
        ]
        # Mirrors _select_winner(winners, cancelled=False): exactly one
        # passing candidate wins; ties and failures abstain. A condition
        # where no candidate could be evaluated is "unavailable", never a
        # phantom "no_candidate".
        if not evaluable:
            decision, winner = "unavailable", None
        elif len(winners) == 1:
            decision, winner = "selected", winners[0]
        elif winners:
            decision, winner = "ambiguous", None
        else:
            decision, winner = "no_candidate", None
        per_condition[position] = {
            "decision": decision,
            "winner": winner,
            "n_evaluable": len(evaluable),
            "per_candidate": {c: _rates(per_candidate.get(c)) for c in candidates},
        }

    flips = []
    for position in sorted(per_condition):
        verdict = per_condition[position]
        if verdict["decision"] == "unavailable":
            continue
        if (verdict["decision"], verdict["winner"]) != (primary_decision, primary_winner):
            flips.append(
                {
                    "position": position,
                    "decision_from": primary_decision,
                    "decision_to": verdict["decision"],
                    "winner_from": primary_winner,
                    "winner_to": verdict["winner"],
                }
            )

    position_effect: dict[str, Any] | None = None
    first = per_condition.get("privilege_first")
    last = per_condition.get("privilege_last")
    if (
        first is not None
        and last is not None
        and first["decision"] != "unavailable"
        and last["decision"] != "unavailable"
    ):
        first_rates = [
            rates["system_rate"]
            for rates in first["per_candidate"].values()
            if rates is not None and rates["system_rate"] is not None
        ]
        last_rates = [
            rates["system_rate"]
            for rates in last["per_candidate"].values()
            if rates is not None and rates["system_rate"] is not None
        ]
        position_effect = {
            "winner_changed": (first["decision"], first["winner"])
            != (last["decision"], last["winner"]),
            "mean_system_rate_delta": (
                (sum(first_rates) / len(first_rates)) - (sum(last_rates) / len(last_rates))
                if first_rates and last_rates
                else None
            ),
        }

    group_adherence: dict[str, dict[str, Any]] = {}
    for label in ("dedicated", "folded", "unknown"):
        pairs: list[float] = []
        families: set[str] = set()
        n_label_candidates = 0
        for cand in candidates:
            if system_treatment(family_of(cand)) != label:
                continue
            n_label_candidates += 1
            families.add(family_of(cand))
            for verdict in per_condition.values():
                rates = verdict["per_candidate"][cand]
                if rates is not None and rates["system_rate"] is not None:
                    pairs.append(rates["system_rate"])
        group_adherence[label] = {
            "n_candidates": n_label_candidates,
            "n_families": len(families),
            "mean_system_rate": (sum(pairs) / len(pairs)) if pairs else None,
        }

    return {
        "available": True,
        "primary_winner": primary_winner,
        "primary_decision": primary_decision,
        "n_fixtures": n_fixtures,
        "positions": sorted(stages),
        "per_condition": per_condition,
        "flips": flips,
        "position_effect": position_effect,
        "group_adherence": group_adherence,
    }


def position_stage(
    backend: Backend,
    formats: tuple[Format, ...],
    position_fixtures: tuple[Fixture, ...],
    *,
    condition: str,
    boundary: tuple[int, ...],
    boundary_map: dict[str, tuple[int, ...]] | None = None,
    max_new_tokens: int = 64,
    seed: int = 0,
) -> dict[str, Any]:
    """Re-evaluate every candidate on position-permuted MC fixtures.

    For one option-position ``condition`` (see :mod:`position`), every
    search candidate is re-evaluated with ``core.evaluate()`` on the
    given permuted multiple-choice fixtures, with the same boundary and
    max_new_tokens. Greedy, exact. Each fixture's ``expected`` is the
    correct option's *text*, so the pipeline's ``exact_match`` grades
    the position-conditioned run unchanged.

    Any backend construction or evaluation failure records None for that
    candidate cell -- fail-closed, never silently skipped. ``pass`` is
    the conservative per-candidate verdict: every position fixture
    exactly matched with its boundary found (mirrors the synthhistory
    stage's all-pass rule), so the selection rule can be re-applied
    unchanged by :func:`position_sensitivity`; the per-condition
    correct counts additionally feed the primacy index.

    Returns ``{"condition": ..., "n_fixtures": ...,
    "per_candidate": {name: {"n", "correct", "pass"} | None}}``.
    """
    if backend is None:
        raise ValueError("backend must not be None")
    if type(formats) is not tuple or not formats:
        raise ValueError("formats must be a nonempty tuple")
    if type(position_fixtures) is not tuple or not position_fixtures:
        raise ValueError("position_fixtures must be a nonempty tuple")
    if any(type(case) is not Fixture for case in position_fixtures):
        raise ValueError("position_fixtures must contain Fixture values")
    if condition not in POSITION_CONDITIONS:
        raise ValueError(f"condition must be one of {POSITION_CONDITIONS}")
    if not _valid_token_ids(boundary, nonempty=True):
        raise ValueError("Boundary must be a nonempty tuple of token IDs")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 4096:
        raise ValueError("max_new_tokens must be an integer between 1 and 4096")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    boundaries = _resolve_boundaries(formats, boundary, boundary_map)
    per_candidate: dict[str, dict[str, Any] | None] = {}
    for fmt in formats:
        try:
            report = evaluate(
                backend,
                fmt,
                position_fixtures,
                boundary=boundaries[fmt.name],
                max_new_tokens=max_new_tokens,
                seed=seed,
            )
        except Exception:  # noqa: BLE001 -- one cell degrades, not the stage
            per_candidate[fmt.name] = None
            continue
        if not report.aggregate_eligible:
            per_candidate[fmt.name] = None
            continue
        correct = sum(1 for obs in report.observations if obs.exact_match and obs.boundary_found)
        per_candidate[fmt.name] = {
            "n": len(position_fixtures),
            "correct": correct,
            "pass": correct == len(position_fixtures),
        }
    return {
        "condition": condition,
        "n_fixtures": len(position_fixtures),
        "per_candidate": per_candidate,
    }


def _position_entry_valid(entry: Any, n_fixtures: int) -> bool:
    """Structural check for one candidate cell of a position stage."""
    if entry is None:
        return True
    if type(entry) is not dict:
        return False
    if entry.get("n") != n_fixtures:
        return False
    correct = entry.get("correct")
    if type(correct) is not int or correct < 0 or correct > n_fixtures:
        return False
    return type(entry.get("pass")) is bool


def position_sensitivity(
    primary_winner: str | None,
    primary_decision: str,
    candidates: tuple[str, ...],
    stages: dict[str, Any],
) -> dict[str, Any]:
    """Re-apply the selection rule per option-position condition (pure post-hoc).

    Answers "does the winner survive option repositioning?": for each
    position condition the conservative selection rule is re-applied to
    that condition's per-candidate pass/fail outcomes, and the primary
    (decision, winner) is compared against every condition.

    Per candidate per condition the exact-answer correct rate is
    reported; ``primacy_index`` is P(correct | correct_first) -
    P(correct | correct_last) per candidate (None when the candidate is
    unevaluable in either condition); ``condition_effect`` summarizes
    the mean index over evaluable candidates; ``flips`` lists the
    conditions whose (decision, winner) differs from the primary run.
    The pre-registered falsifiable claim (see :mod:`position`) is that
    the index varies across template families rather than being
    constant -- reported here, never asserted.

    Returns ``{"available": True, ...}``. Never raises on malformed
    input: returns ``{"available": False}`` with a ``detail`` code.
    """
    if (
        not isinstance(candidates, tuple)
        or not candidates
        or not all(type(c) is str and c.strip() for c in candidates)
        or len(set(candidates)) != len(candidates)
    ):
        return {"available": False, "detail": "invalid_candidates"}
    if primary_winner is not None and (
        type(primary_winner) is not str or primary_winner not in candidates
    ):
        return {"available": False, "detail": "primary_winner_unknown"}
    if type(primary_decision) is not str or not primary_decision:
        return {"available": False, "detail": "primary_decision_invalid"}
    if type(stages) is not dict or not stages or not all(type(k) is str for k in stages):
        return {"available": False, "detail": "stages_invalid"}
    n_fixtures = -1  # sentinel: stages is guaranteed nonempty above
    for condition, stage in stages.items():
        if type(stage) is not dict:
            return {"available": False, "detail": f"stage_{condition}_invalid"}
        if type(stage.get("condition")) is not str:
            return {"available": False, "detail": f"stage_{condition}_invalid"}
        count = stage.get("n_fixtures")
        if type(count) is not int or count < 0:
            return {"available": False, "detail": f"stage_{condition}_invalid"}
        if n_fixtures < 0:
            n_fixtures = count
        elif n_fixtures != count:
            return {"available": False, "detail": "fixture_count_mismatch"}
        per_candidate = stage.get("per_candidate")
        if type(per_candidate) is not dict or not all(type(k) is str for k in per_candidate):
            return {"available": False, "detail": f"stage_{condition}_invalid"}
        for _cand, entry in per_candidate.items():
            if not _position_entry_valid(entry, count):
                return {"available": False, "detail": f"stage_{condition}_invalid"}

    def _rates(entry: dict[str, Any] | None) -> dict[str, Any] | None:
        if entry is None:
            return None
        n = entry["n"]
        return {
            "correct_rate": entry["correct"] / n if n else None,
            "pass": entry["pass"],
        }

    per_condition: dict[str, dict[str, Any]] = {}
    for condition in sorted(stages):
        stage = stages[condition]
        per_candidate = stage["per_candidate"]
        entries = [per_candidate.get(c) for c in candidates]
        evaluable = [e for e in entries if e is not None]
        winners = [
            c for c, e in zip(candidates, entries, strict=True) if e is not None and e["pass"]
        ]
        # Mirrors _select_winner(winners, cancelled=False): exactly one
        # passing candidate wins; ties and failures abstain. A condition
        # where no candidate could be evaluated is "unavailable", never a
        # phantom "no_candidate".
        if not evaluable:
            decision, winner = "unavailable", None
        elif len(winners) == 1:
            decision, winner = "selected", winners[0]
        elif winners:
            decision, winner = "ambiguous", None
        else:
            decision, winner = "no_candidate", None
        per_condition[condition] = {
            "decision": decision,
            "winner": winner,
            "n_evaluable": len(evaluable),
            "per_candidate": {c: _rates(per_candidate.get(c)) for c in candidates},
        }

    flips = []
    for condition in sorted(per_condition):
        verdict = per_condition[condition]
        if verdict["decision"] == "unavailable":
            continue
        if (verdict["decision"], verdict["winner"]) != (primary_decision, primary_winner):
            flips.append(
                {
                    "condition": condition,
                    "decision_from": primary_decision,
                    "decision_to": verdict["decision"],
                    "winner_from": primary_winner,
                    "winner_to": verdict["winner"],
                }
            )

    primacy: dict[str, float | None] = {}
    first_stage = stages.get("correct_first")
    last_stage = stages.get("correct_last")
    for cand in candidates:
        first_entry = first_stage["per_candidate"].get(cand) if type(first_stage) is dict else None
        last_entry = last_stage["per_candidate"].get(cand) if type(last_stage) is dict else None
        # primacy_index never raises: malformed/missing cells degrade to
        # None rather than poisoning the analysis.
        if type(first_entry) is not dict or type(last_entry) is not dict:
            primacy[cand] = None
            continue
        # Validation above guarantees int n/correct for dict entries; the
        # casts document that invariant for the type checker.
        primacy[cand] = primacy_index(
            cast(int, first_entry.get("n")),
            cast(int, first_entry.get("correct")),
            cast(int, last_entry.get("n")),
            cast(int, last_entry.get("correct")),
        )

    evaluable_indexes = [v for v in primacy.values() if v is not None]
    condition_effect: dict[str, Any] | None = None
    if evaluable_indexes:
        condition_effect = {
            "mean_index": sum(evaluable_indexes) / len(evaluable_indexes),
            "n_candidates": len(evaluable_indexes),
            "n_positive": sum(1 for v in evaluable_indexes if v > 0),
            "n_negative": sum(1 for v in evaluable_indexes if v < 0),
        }

    return {
        "available": True,
        "primary_winner": primary_winner,
        "primary_decision": primary_decision,
        "n_fixtures": n_fixtures,
        "conditions": sorted(stages),
        "per_condition": per_condition,
        "flips": flips,
        "primacy_index": primacy,
        "condition_effect": condition_effect,
    }


def synthhistory_stage(
    backend: Backend,
    formats: tuple[Format, ...],
    history_fixtures: tuple[Fixture, ...],
    *,
    condition: str,
    boundary: tuple[int, ...],
    boundary_map: dict[str, tuple[int, ...]] | None = None,
    max_new_tokens: int = 64,
    seed: int = 0,
) -> dict[str, Any]:
    """Re-evaluate every candidate with synthetic history prepended.

    For one history ``condition`` (see :mod:`synthhistory`), every
    search candidate is re-evaluated with ``core.evaluate()`` on the
    given history-conditioned fixtures, with the same boundary and
    max_new_tokens. Greedy, exact. The fixture's ``expected`` is
    unchanged by history prepending, so the pipeline's ``exact_match``
    already grades the history-conditioned run.

    Any backend construction or evaluation failure records None for that
    candidate cell -- fail-closed, never silently skipped. ``pass`` is
    the conservative per-candidate verdict: every history fixture
    exactly matched with its boundary found (mirrors the hierarchy
    stage's all-pass rule), so the selection rule can be re-applied
    unchanged by :func:`synthhistory_stability`.

    Returns ``{"condition": ..., "n_turns": ..., "n_fixtures": ...,
    "per_candidate": {name: {"n", "passed", "pass"} | None}}``.
    """
    if backend is None:
        raise ValueError("backend must not be None")
    if type(formats) is not tuple or not formats:
        raise ValueError("formats must be a nonempty tuple")
    if type(history_fixtures) is not tuple or not history_fixtures:
        raise ValueError("history_fixtures must be a nonempty tuple")
    if any(type(case) is not Fixture for case in history_fixtures):
        raise ValueError("history_fixtures must contain Fixture values")
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    if not _valid_token_ids(boundary, nonempty=True):
        raise ValueError("Boundary must be a nonempty tuple of token IDs")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 4096:
        raise ValueError("max_new_tokens must be an integer between 1 and 4096")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    boundaries = _resolve_boundaries(formats, boundary, boundary_map)
    per_candidate: dict[str, dict[str, Any] | None] = {}
    for fmt in formats:
        try:
            report = evaluate(
                backend,
                fmt,
                history_fixtures,
                boundary=boundaries[fmt.name],
                max_new_tokens=max_new_tokens,
                seed=seed,
            )
        except Exception:  # noqa: BLE001 -- one cell degrades, not the stage
            per_candidate[fmt.name] = None
            continue
        if not report.aggregate_eligible:
            per_candidate[fmt.name] = None
            continue
        passed = sum(1 for obs in report.observations if obs.exact_match and obs.boundary_found)
        per_candidate[fmt.name] = {
            "n": len(history_fixtures),
            "passed": passed,
            "pass": passed == len(history_fixtures),
        }
    return {
        "condition": condition,
        "n_turns": TURN_COUNTS[condition],
        "n_fixtures": len(history_fixtures),
        "per_candidate": per_candidate,
    }


def _synthhistory_entry_valid(entry: Any, n_fixtures: int) -> bool:
    """Structural check for one candidate cell of a synthhistory stage."""
    if entry is None:
        return True
    if type(entry) is not dict:
        return False
    if entry.get("n") != n_fixtures:
        return False
    passed = entry.get("passed")
    if type(passed) is not int or passed < 0 or passed > n_fixtures:
        return False
    return type(entry.get("pass")) is bool


def synthhistory_stability(
    primary_winner: str | None,
    primary_decision: str,
    candidates: tuple[str, ...],
    stages: dict[str, Any],
) -> dict[str, Any]:
    """Re-apply the selection rule per history condition (pure post-hoc).

    Answers "does the winner survive synthetic history?": for each
    history condition the conservative selection rule is re-applied to
    that condition's per-candidate pass/fail outcomes, and the primary
    (decision, winner) is compared against every condition.

    Per candidate per condition the history pass rate is reported;
    ``flips`` lists the conditions whose (decision, winner) differs from
    the primary run; ``group_pass_rates`` tests the pre-registered
    structural hypothesis from :mod:`synthhistory` -- families whose
    templates use unambiguous per-turn open/close markers vs families
    with loose delimiting -- as mean history pass rate per
    mechanically-derived label group (a correlation readout, not a
    causal claim).

    Returns ``{"available": True, ...}``. Never raises on malformed
    input: returns ``{"available": False}`` with a ``detail`` code.
    """
    if (
        not isinstance(candidates, tuple)
        or not candidates
        or not all(type(c) is str and c.strip() for c in candidates)
        or len(set(candidates)) != len(candidates)
    ):
        return {"available": False, "detail": "invalid_candidates"}
    if primary_winner is not None and (
        type(primary_winner) is not str or primary_winner not in candidates
    ):
        return {"available": False, "detail": "primary_winner_unknown"}
    if type(primary_decision) is not str or not primary_decision:
        return {"available": False, "detail": "primary_decision_invalid"}
    if type(stages) is not dict or not stages or not all(type(k) is str for k in stages):
        return {"available": False, "detail": "stages_invalid"}
    n_fixtures = -1  # sentinel: stages is guaranteed nonempty above
    for condition, stage in stages.items():
        if type(stage) is not dict:
            return {"available": False, "detail": f"stage_{condition}_invalid"}
        if stage.get("condition") != condition:
            return {"available": False, "detail": f"stage_{condition}_invalid"}
        count = stage.get("n_fixtures")
        if type(count) is not int or count <= 0:
            return {"available": False, "detail": f"stage_{condition}_invalid"}
        if n_fixtures < 0:
            n_fixtures = count
        elif n_fixtures != count:
            return {"available": False, "detail": "fixture_count_mismatch"}
        per_candidate = stage.get("per_candidate")
        if type(per_candidate) is not dict or not all(type(k) is str for k in per_candidate):
            return {"available": False, "detail": f"stage_{condition}_invalid"}
        for _cand, entry in per_candidate.items():
            if not _synthhistory_entry_valid(entry, count):
                return {"available": False, "detail": f"stage_{condition}_invalid"}

    def _rates(entry: dict[str, Any] | None) -> dict[str, Any] | None:
        if entry is None:
            return None
        n = entry["n"]
        return {
            "pass_rate": entry["passed"] / n if n else None,
            "pass": entry["pass"],
        }

    per_condition: dict[str, dict[str, Any]] = {}
    for condition in sorted(stages):
        stage = stages[condition]
        per_candidate = stage["per_candidate"]
        entries = [per_candidate.get(c) for c in candidates]
        evaluable = [e for e in entries if e is not None]
        winners = [
            c for c, e in zip(candidates, entries, strict=True) if e is not None and e["pass"]
        ]
        # Mirrors _select_winner(winners, cancelled=False): exactly one
        # passing candidate wins; ties and failures abstain. A condition
        # where no candidate could be evaluated is "unavailable", never a
        # phantom "no_candidate".
        if not evaluable:
            decision, winner = "unavailable", None
        elif len(winners) == 1:
            decision, winner = "selected", winners[0]
        elif winners:
            decision, winner = "ambiguous", None
        else:
            decision, winner = "no_candidate", None
        per_condition[condition] = {
            "decision": decision,
            "winner": winner,
            "n_evaluable": len(evaluable),
            "per_candidate": {c: _rates(per_candidate.get(c)) for c in candidates},
        }

    # The primary decision is the run-level verdict ("compatible_on_fixture_set",
    # "held_out_failed", ...); per-condition verdicts use the selection
    # vocabulary ("selected", "ambiguous", "no_candidate"). A run that reached
    # held-out evaluation necessarily selected its winner, so map the two
    # success decisions onto the selection vocabulary for an apples-to-apples
    # flip comparison -- otherwise every successful run would report a phantom
    # flip wherever history re-selects the identical winner. Other run
    # outcomes (ambiguous / no_candidate / failed / cancelled) compare verbatim.
    if primary_decision in ("compatible_on_fixture_set", "held_out_failed"):
        primary_verdict = ("selected", primary_winner)
    else:
        primary_verdict = (primary_decision, primary_winner)
    flips = []
    for condition in sorted(per_condition):
        verdict = per_condition[condition]
        if verdict["decision"] == "unavailable":
            continue
        if (verdict["decision"], verdict["winner"]) != primary_verdict:
            flips.append(
                {
                    "condition": condition,
                    "decision_from": primary_verdict[0],
                    "decision_to": verdict["decision"],
                    "winner_from": primary_verdict[1],
                    "winner_to": verdict["winner"],
                }
            )

    group_pass_rates: dict[str, dict[str, Any]] = {}
    for label in DELIMITER_LABELS:
        pairs = []
        for candidate in candidates:
            if candidate_delimiter_label(candidate) != label:
                continue
            rates = [
                per_condition[condition]["per_candidate"][candidate]
                for condition in sorted(per_condition)
            ]
            valid = [r["pass_rate"] for r in rates if r is not None and r["pass_rate"] is not None]
            if valid:
                pairs.append(sum(valid) / len(valid))
        group_pass_rates[label] = {
            "n_candidates": len(pairs),
            "mean_pass_rate": (sum(pairs) / len(pairs)) if pairs else None,
        }

    return {
        "available": True,
        "primary_winner": primary_winner,
        "primary_decision": primary_decision,
        "n_fixtures": n_fixtures,
        "conditions": sorted(stages),
        "per_condition": per_condition,
        "flips": flips,
        "group_pass_rates": group_pass_rates,
    }


def fewshot_stage(
    backend: Backend,
    formats: tuple[Format, ...],
    demo_cases: tuple[Fixture, ...],
    eval_cases: tuple[Fixture, ...],
    *,
    boundary: tuple[int, ...],
    boundary_map: dict[str, tuple[int, ...]] | None = None,
    max_new_tokens: int = 64,
    seed: int = 0,
) -> dict[str, Any]:
    """Re-evaluate every candidate 0-shot and with both demo orders.

    Demos are sampled (seeded, v1 k=2) from ``demo_cases`` -- the search
    split -- and every arm is measured on ``eval_cases`` -- the held-out
    split -- so a demo fixture can never evaluate itself (the
    contamination guard). Three arms per candidate: ``baseline`` (0-shot),
    ``order_a`` and ``order_b`` (the two demo orders, see
    :mod:`fewshot`). Greedy, exact; the pipeline's ``exact_match``
    grades every arm unchanged because demo prepending leaves each
    fixture's ``expected`` untouched.

    A candidate cell degrades to None on backend/evaluation failure --
    fail-closed, never a phantom gain. Raises ``ValueError`` on
    caller-contract violations (None backend, bad formats/demo/eval
    cases, bad boundary). A demo pool smaller than the v1 k returns the
    fail-closed ``{"available": False, "detail":
    "insufficient_demo_pool"}`` payload instead of raising: a data
    condition, not a caller bug.

    Returns ``{"available": True, "k", "demo_ids", "n_eval", "arms",
    "per_candidate": {name: {"n_eval", "verdicts": {fid: {arm:
    bool|None}}} | None}}``.
    """
    if backend is None:
        raise ValueError("backend must not be None")
    if type(formats) is not tuple or not formats:
        raise ValueError("formats must be a nonempty tuple")
    if type(demo_cases) is not tuple or not demo_cases:
        raise ValueError("demo_cases must be a nonempty tuple")
    if any(type(case) is not Fixture for case in demo_cases):
        raise ValueError("demo_cases must contain Fixture values")
    if type(eval_cases) is not tuple or not eval_cases:
        raise ValueError("eval_cases must be a nonempty tuple")
    if any(type(case) is not Fixture for case in eval_cases):
        raise ValueError("eval_cases must contain Fixture values")
    if not _valid_token_ids(boundary, nonempty=True):
        raise ValueError("Boundary must be a nonempty tuple of token IDs")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 4096:
        raise ValueError("max_new_tokens must be an integer between 1 and 4096")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    if len(demo_cases) < FEWSHOT_K:
        return {
            "available": False,
            "detail": DETAIL_INSUFFICIENT_DEMO_POOL,
            "k": FEWSHOT_K,
            "n_demo_pool": len(demo_cases),
        }
    demos = sample_demos(tuple(demo_cases))
    perms = demo_permutations(demos)
    arms: dict[str, tuple[Fixture, ...]] = {"baseline": tuple(eval_cases)}
    for position, perm in enumerate(perms):
        arms[f"order_{'a' if position == 0 else 'b'}"] = build_demo_fixtures(
            tuple(eval_cases), perm, perm_index=position
        )
    boundaries = _resolve_boundaries(formats, boundary, boundary_map)
    eval_ids = tuple(case.id for case in eval_cases)
    per_candidate: dict[str, dict[str, Any] | None] = {}
    for fmt in formats:
        cell: dict[str, dict[str, bool | None]] = {}
        try:
            for arm in FEWSHOT_ARMS:
                report = evaluate(
                    backend,
                    fmt,
                    arms[arm],
                    boundary=boundaries[fmt.name],
                    max_new_tokens=max_new_tokens,
                    seed=seed,
                )
                if not report.aggregate_eligible:
                    raise RuntimeError("arm not aggregate-eligible")
                arm_verdicts = {
                    obs.fixture: (
                        (obs.exact_match and obs.boundary_found)
                        if obs.status == "completed"
                        else None
                    )
                    for obs in report.observations
                }
                for fid in eval_ids:
                    if arm == "baseline":
                        namespaced = fid
                    else:
                        perm_index = FEWSHOT_ARMS.index(arm) - 1
                        namespaced = f"fewshot/perm{perm_index}/{fid}"
                    cell.setdefault(fid, {})[arm] = arm_verdicts.get(namespaced)
        except Exception:  # noqa: BLE001 -- one cell degrades, not the stage
            per_candidate[fmt.name] = None
            continue
        per_candidate[fmt.name] = {"n_eval": len(eval_ids), "verdicts": cell}
    return {
        "available": True,
        "k": FEWSHOT_K,
        "demo_ids": [case.id for case in demos],
        "n_eval": len(eval_ids),
        "arms": list(FEWSHOT_ARMS),
        "per_candidate": per_candidate,
    }


def _fewshot_entry_valid(entry: Any, n_eval: int) -> bool:
    """Structural check for one candidate cell of a fewshot stage."""
    if entry is None:
        return True
    if type(entry) is not dict:
        return False
    if entry.get("n_eval") != n_eval:
        return False
    verdicts = entry.get("verdicts")
    if type(verdicts) is not dict or not all(type(fid) is str for fid in verdicts):
        return False
    for per_arm in verdicts.values():
        if type(per_arm) is not dict:
            return False
        for arm in FEWSHOT_ARMS:
            verdict = per_arm.get(arm)
            if verdict is not None and type(verdict) is not bool:
                return False
    return True


def _fewshot_arm_verdict(
    per_candidate: dict[str, Any], candidates: tuple[str, ...]
) -> tuple[str, str | None, dict[str, dict[str, Any] | None]]:
    """Re-apply the selection rule to one fewshot arm (pure).

    A candidate is evaluable for the arm when every eval fixture has a
    non-None verdict; it passes on the conservative all-pass rule.
    Exactly one passing candidate wins; ties and failures abstain; an
    arm with no evaluable candidate is "unavailable", never a phantom
    "no_candidate".
    """
    rates: dict[str, dict[str, Any] | None] = {}
    for name in candidates:
        entry = per_candidate.get(name)
        if type(entry) is not dict:
            rates[name] = None
            continue
        verdicts = entry.get("verdicts") or {}
        arm_values = [verdicts.get(fid) for fid in verdicts]
        if not arm_values or any(type(v) is not bool for v in arm_values):
            rates[name] = None
            continue
        passed = sum(1 for v in arm_values if v)
        rates[name] = {"pass": passed == len(arm_values), "pass_rate": passed / len(arm_values)}
    evaluable = [name for name in candidates if rates[name] is not None]
    winners = [name for name in evaluable if rates[name]["pass"]]  # type: ignore[index]
    if not evaluable:
        return "unavailable", None, rates
    if len(winners) == 1:
        return "selected", winners[0], rates
    if winners:
        return "ambiguous", None, rates
    return "no_candidate", None, rates


def fewshot_gain(
    primary_winner: str | None,
    primary_decision: str,
    candidates: tuple[str, ...],
    stage: dict[str, Any],
) -> dict[str, Any]:
    """Analyze few-shot demonstration gain x template family (pure post-hoc).

    Answers "do demos move verdicts, and does the demo order matter,
    differently per template?": per candidate, ``gain`` (mean over the
    two demo orders of the k-shot vs 0-shot verdict-change rate) and
    ``order_sensitivity`` (population std of the two per-order rates).
    The selection rule is re-applied per arm (baseline, order_a,
    order_b) and each arm's (decision, winner) is compared against the
    primary run.

    The pre-registered falsifiable claim (see :mod:`fewshot`) is that
    gain and order_sensitivity are identical across families; family
    differences are the finding. "Gain" is a verdict-change rate, never
    a quality judgment.

    Never raises on malformed input: returns ``{"available": False}``
    with a ``detail`` code.
    """
    if (
        not isinstance(candidates, tuple)
        or not candidates
        or not all(type(c) is str and c.strip() for c in candidates)
        or len(set(candidates)) != len(candidates)
    ):
        return {"available": False, "detail": "invalid_candidates"}
    if primary_winner is not None and (
        type(primary_winner) is not str or primary_winner not in candidates
    ):
        return {"available": False, "detail": "primary_winner_unknown"}
    if type(primary_decision) is not str or not primary_decision:
        return {"available": False, "detail": "primary_decision_invalid"}
    if type(stage) is not dict:
        return {"available": False, "detail": "stage_invalid"}
    if stage.get("available") is not True:
        return {
            "available": False,
            "detail": str(stage.get("detail") or "fewshot_unavailable"),
        }
    n_eval = stage.get("n_eval")
    demo_ids = stage.get("demo_ids")
    per_candidate = stage.get("per_candidate")
    if (
        type(n_eval) is not int
        or n_eval <= 0
        or type(demo_ids) is not list
        or not demo_ids
        or not all(type(d) is str for d in demo_ids)
        or type(per_candidate) is not dict
        or not all(type(k) is str for k in per_candidate)
    ):
        return {"available": False, "detail": "stage_invalid"}
    for entry in per_candidate.values():
        if not _fewshot_entry_valid(entry, n_eval):
            return {"available": False, "detail": "stage_invalid"}

    results: dict[str, dict[str, Any]] = {}
    excluded: list[str] = []
    for name in candidates:
        entry = per_candidate.get(name)
        if type(entry) is not dict:
            excluded.append(name)
            continue
        verdicts = entry["verdicts"]
        n_full = sum(
            1
            for per_arm in verdicts.values()
            if all(per_arm.get(arm) is not None for arm in FEWSHOT_ARMS)
        )
        if n_full == 0:
            excluded.append(name)
            continue
        baseline = {fid: per_arm["baseline"] for fid, per_arm in verdicts.items()}
        order_rates: dict[str, dict[str, Any]] = {}
        rate_values: list[float | None] = []
        for arm in ("order_a", "order_b"):
            arm_verdicts = {fid: per_arm[arm] for fid, per_arm in verdicts.items()}
            # _fewshot_entry_valid guarantees the verdict maps, so
            # change_rate cannot raise here.
            computed = change_rate(baseline, arm_verdicts)
            order_rates[arm] = {
                "change_rate": computed["rate"],
                "n_changed": computed["n_changed"],
                "n_evaluable": computed["n_evaluable"],
            }
            rate_values.append(computed["rate"])
        # change_rate only yields floats in [0, 1] or None, so
        # order_sensitivity cannot raise here either.
        sensitivity = order_sensitivity(tuple(rate_values))
        # Both orders always contribute a rate here: n_full >= 1 (checked
        # above) guarantees each (baseline, order) pair at least one
        # evaluable fixture, so the filter below is provably nonempty
        # (documented, not defended: a defensive fallback would be dead
        # code and an uncoverable line).
        valid_rates = [rate for rate in rate_values if rate is not None]
        gain = sum(valid_rates) / len(valid_rates)
        results[name] = {
            "gain": gain,
            "order_sensitivity": sensitivity,
            "n_evaluable": n_full,
            "per_order": order_rates,
        }

    per_arm: dict[str, dict[str, Any]] = {}
    for arm in FEWSHOT_ARMS:
        arm_cells: dict[str, dict[str, Any] | None] = {}
        for name in candidates:
            stage_entry = per_candidate.get(name)
            if type(stage_entry) is dict:
                arm_cells[name] = {
                    "n_eval": stage_entry.get("n_eval"),
                    "verdicts": {
                        fid: per_arm[arm]
                        for fid, per_arm in stage_entry.get("verdicts", {}).items()
                    },
                }
            else:
                arm_cells[name] = None
        decision, winner, rates = _fewshot_arm_verdict(arm_cells, candidates)
        per_arm[arm] = {
            "decision": decision,
            "winner": winner,
            "n_evaluable": sum(1 for r in rates.values() if r is not None),
            "per_candidate": rates,
        }
    if primary_decision in ("compatible_on_fixture_set", "held_out_failed"):
        primary_verdict = ("selected", primary_winner)
    else:
        primary_verdict = (primary_decision, primary_winner)
    flips = []
    for arm in FEWSHOT_ARMS:
        verdict = per_arm[arm]
        if verdict["decision"] == "unavailable":
            continue
        if (verdict["decision"], verdict["winner"]) != primary_verdict:
            flips.append(
                {
                    "arm": arm,
                    "decision_from": primary_verdict[0],
                    "decision_to": verdict["decision"],
                    "winner_from": primary_verdict[1],
                    "winner_to": verdict["winner"],
                }
            )
    return {
        "available": True,
        "k": stage.get("k"),
        "demo_ids": list(demo_ids),
        "n_eval": n_eval,
        "primary_winner": primary_winner,
        "primary_decision": primary_decision,
        "per_candidate": results,
        "excluded_candidates": sorted(excluded),
        "per_arm": per_arm,
        "flips": flips,
    }


def prefill_stage(
    backend: Backend,
    formats: tuple[Format, ...],
    cases: tuple[Fixture, ...],
    *,
    prefill: dict[str, str],
    boundary: tuple[int, ...],
    boundary_map: dict[str, tuple[int, ...]] | None = None,
    max_new_tokens: int = 64,
    seed: int = 0,
) -> dict[str, Any]:
    """Re-evaluate every candidate with an assistant prefill applied.

    For one prefill spec (see :mod:`prefill`), every search candidate is
    re-evaluated with ``core.evaluate()`` on the search fixtures through
    a :class:`PrefillBackend` that appends the prefill token ids after
    the encoded prompt. Greedy, exact.

    Grading needs no fixture transformation: the scored span is
    ``prefill_text + generated`` and ``prefill_text + generated ==
    prefill_text + expected`` iff ``generated == expected``, so the
    pipeline's ``exact_match`` on the plain expected answer already
    grades the prefill run. ``pass`` is the conservative per-candidate
    verdict: every fixture exact-matched with its boundary found
    (mirrors the hierarchy stage's all-pass rule), so the selection
    rule can be re-applied unchanged by :func:`prefill_adherence`.

    Per observation the stage additionally scores prefill honoring via
    :func:`honors_prefill` (did the model accept the prefill as its own
    opening, or restart its turn and duplicate it?).

    Fail-closed: a prefill that cannot be encoded, does not decode back
    to its text (the scored-span identity would break), or contains a
    candidate's response boundary (the boundary search would cut inside
    the prefill) degrades that candidate cell to None -- or raises
    ``ValueError`` for stage-global setup failures. Never silently
    wrong.

    Returns ``{"prefill_id": ..., "prefill_text": ..., "n_fixtures":
    ..., "per_candidate": {name: {"n", "n_completed", "honored",
    "restarted", "think_channel", "pass"} | None}}``.
    """
    if backend is None:
        raise ValueError("backend must not be None")
    if type(formats) is not tuple or not formats:
        raise ValueError("formats must be a nonempty tuple")
    if type(cases) is not tuple or not cases or any(type(case) is not Fixture for case in cases):
        raise ValueError("cases must be a nonempty tuple of Fixture")
    (spec,) = validate_prefill_specs((prefill,))
    prefill_id = spec["id"]
    prefill_text = spec["text"]
    if not _valid_token_ids(boundary, nonempty=True):
        raise ValueError("Boundary must be a nonempty tuple of token IDs")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 4096:
        raise ValueError("max_new_tokens must be an integer between 1 and 4096")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    try:
        prefill_ids = backend.encode(prefill_text)
    except Exception as exc:  # noqa: BLE001 -- stage cannot be constructed
        raise ValueError("prefill text could not be encoded") from exc
    if not _valid_token_ids(prefill_ids, nonempty=True):
        raise ValueError("prefill text did not encode to token ids")
    try:
        round_tripped = backend.decode(prefill_ids)
    except Exception as exc:  # noqa: BLE001 -- scored-span identity check
        raise ValueError("prefill ids did not decode") from exc
    if round_tripped != prefill_text:
        # Token decode is a homomorphism over id-concatenation, so the
        # scored span decode(prefill_ids + generated) equals
        # prefill_text + decode(generated) exactly when this holds.
        raise ValueError("prefill ids did not decode back to the prefill text")
    boundaries = _resolve_boundaries(formats, boundary, boundary_map)
    fixtures = build_prefill_fixtures(cases, prefill=spec)
    per_candidate: dict[str, dict[str, Any] | None] = {}
    for fmt in formats:
        candidate_boundary = boundaries[fmt.name]
        if first_boundary(prefill_ids, candidate_boundary) is not None:
            # The boundary search would cut inside the prefill itself;
            # the scored-span semantics would be wrong. Fail closed.
            per_candidate[fmt.name] = None
            continue
        wrapped = PrefillBackend(backend, prefill_ids, prefill_id)
        try:
            report = evaluate(
                wrapped,
                fmt,
                fixtures,
                boundary=candidate_boundary,
                max_new_tokens=max_new_tokens,
                seed=seed,
            )
        except Exception:  # noqa: BLE001 -- one cell degrades, not the stage
            per_candidate[fmt.name] = None
            continue
        if not report.aggregate_eligible:
            per_candidate[fmt.name] = None
            continue
        n_completed = honored = restarted = 0
        all_pass = True
        for obs in report.observations:
            if obs.status != "completed":
                all_pass = False
                continue
            n_completed += 1
            if honors_prefill(obs.answer, prefill_text):
                honored += 1
            else:
                restarted += 1
            if not (obs.exact_match and obs.boundary_found):
                all_pass = False
        per_candidate[fmt.name] = {
            "n": len(cases),
            "n_completed": n_completed,
            "honored": honored,
            "restarted": restarted,
            "think_channel": opens_think_channel(fmt),
            "pass": all_pass,
        }
    return {
        "prefill_id": prefill_id,
        "prefill_text": prefill_text,
        "n_fixtures": len(cases),
        "per_candidate": per_candidate,
    }


def _prefill_entry_valid(entry: Any, n_fixtures: int) -> bool:
    """Structural check for one candidate cell of a prefill stage."""
    if entry is None:
        return True
    if type(entry) is not dict:
        return False
    if entry.get("n") != n_fixtures:
        return False
    completed = entry.get("n_completed")
    honored = entry.get("honored")
    restarted = entry.get("restarted")
    if (
        type(completed) is not int
        or completed < 0
        or type(honored) is not int
        or honored < 0
        or type(restarted) is not int
        or restarted < 0
    ):
        return False
    if honored + restarted != completed:
        return False
    if completed > n_fixtures:
        return False
    if type(entry.get("think_channel")) is not bool:
        return False
    return type(entry.get("pass")) is bool


def prefill_adherence(
    primary_winner: str | None,
    primary_decision: str,
    candidates: tuple[str, ...],
    stages: dict[str, Any],
) -> dict[str, Any]:
    """Re-apply the selection rule per prefill condition (pure post-hoc).

    Answers "does the winner survive assistant prefill?": for each
    prefill the conservative selection rule is re-applied to that
    condition's per-candidate pass/fail outcomes, and the primary
    (decision, winner) is compared against every condition.

    Per candidate per condition the honoring rates (honored /
    restarted) are reported; ``think_effect`` tests the pre-registered
    mechanistic hypothesis from :mod:`prefill` -- candidates whose
    generation prompt opens a reasoning/think channel vs the rest -- as
    mean honoring rates per group (a correlation readout, not a causal
    claim). ``flips`` lists the conditions whose (decision, winner)
    differs from the primary run.

    Returns ``{"available": True, ...}``. Never raises on malformed
    input: returns ``{"available": False}`` with a ``detail`` code.
    """
    if (
        not isinstance(candidates, tuple)
        or not candidates
        or not all(type(c) is str and c for c in candidates)
        or len(set(candidates)) != len(candidates)
    ):
        return {"available": False, "detail": "invalid_candidates"}
    if primary_winner is not None and (
        type(primary_winner) is not str or primary_winner not in candidates
    ):
        return {"available": False, "detail": "primary_winner_unknown"}
    if type(primary_decision) is not str or not primary_decision:
        return {"available": False, "detail": "primary_decision_invalid"}
    if type(stages) is not dict or not stages or not all(type(k) is str for k in stages):
        return {"available": False, "detail": "stages_invalid"}
    n_fixtures = -1  # sentinel: stages is guaranteed nonempty above
    for prefill_id, stage in stages.items():
        if type(stage) is not dict:
            return {"available": False, "detail": f"stage_{prefill_id}_invalid"}
        if stage.get("prefill_id") != prefill_id:
            return {"available": False, "detail": f"stage_{prefill_id}_invalid"}
        if type(stage.get("prefill_text")) is not str:
            return {"available": False, "detail": f"stage_{prefill_id}_invalid"}
        count = stage.get("n_fixtures")
        if type(count) is not int or count < 0:
            return {"available": False, "detail": f"stage_{prefill_id}_invalid"}
        if n_fixtures < 0:
            n_fixtures = count
        elif n_fixtures != count:
            return {"available": False, "detail": "fixture_count_mismatch"}
        per_candidate = stage.get("per_candidate")
        if type(per_candidate) is not dict or not all(type(k) is str for k in per_candidate):
            return {"available": False, "detail": f"stage_{prefill_id}_invalid"}
        for _cand, entry in per_candidate.items():
            if not _prefill_entry_valid(entry, count):
                return {"available": False, "detail": f"stage_{prefill_id}_invalid"}

    def _rates(entry: dict[str, Any] | None) -> dict[str, Any] | None:
        if entry is None:
            return None
        completed = entry["n_completed"]
        return {
            "honor_rate": entry["honored"] / completed if completed else None,
            "restart_rate": entry["restarted"] / completed if completed else None,
            "think_channel": entry["think_channel"],
            "pass": entry["pass"],
        }

    per_condition: dict[str, dict[str, Any]] = {}
    for prefill_id in sorted(stages):
        stage = stages[prefill_id]
        per_candidate = stage["per_candidate"]
        entries = [per_candidate.get(c) for c in candidates]
        evaluable = [e for e in entries if e is not None]
        winners = [
            c for c, e in zip(candidates, entries, strict=True) if e is not None and e["pass"]
        ]
        # Mirrors _select_winner(winners, cancelled=False): exactly one
        # passing candidate wins; ties and failures abstain. A condition
        # where no candidate could be evaluated is "unavailable", never a
        # phantom "no_candidate".
        if not evaluable:
            decision, winner = "unavailable", None
        elif len(winners) == 1:
            decision, winner = "selected", winners[0]
        elif winners:
            decision, winner = "ambiguous", None
        else:
            decision, winner = "no_candidate", None
        per_condition[prefill_id] = {
            "decision": decision,
            "winner": winner,
            "n_evaluable": len(evaluable),
            "per_candidate": {c: _rates(per_candidate.get(c)) for c in candidates},
        }

    flips = []
    for prefill_id in sorted(per_condition):
        verdict = per_condition[prefill_id]
        if verdict["decision"] == "unavailable":
            continue
        if (verdict["decision"], verdict["winner"]) != (primary_decision, primary_winner):
            flips.append(
                {
                    "prefill": prefill_id,
                    "decision_from": primary_decision,
                    "decision_to": verdict["decision"],
                    "winner_from": primary_winner,
                    "winner_to": verdict["winner"],
                }
            )

    think_effect: dict[str, Any] = {}
    for label, wanted in (("think_channel", True), ("plain", False)):
        honor_rates = [
            rates["honor_rate"]
            for verdict in per_condition.values()
            for rates in verdict["per_candidate"].values()
            if rates is not None
            and rates["think_channel"] is wanted
            and rates["honor_rate"] is not None
        ]
        think_effect[label] = {
            "mean_honor_rate": (sum(honor_rates) / len(honor_rates)) if honor_rates else None,
            "n_candidates": len(
                {
                    cand
                    for verdict in per_condition.values()
                    for cand, rates in verdict["per_candidate"].items()
                    if rates is not None and rates["think_channel"] is wanted
                }
            ),
        }
    think_means = [think_effect[label]["mean_honor_rate"] for label in ("think_channel", "plain")]
    think_effect["delta"] = (
        think_means[0] - think_means[1]
        if all(type(m) in (int, float) for m in think_means)
        else None
    )

    return {
        "available": True,
        "primary_winner": primary_winner,
        "primary_decision": primary_decision,
        "n_fixtures": n_fixtures,
        "prefills": sorted(stages),
        "per_condition": per_condition,
        "flips": flips,
        "think_effect": think_effect,
    }


def _changed_rate(spec: dict[str, Any]) -> float | None:
    """Fraction of fixtures actually changed by a variant's operator."""
    changed = spec.get("changed_fixtures")
    total = spec.get("n_fixtures")
    if type(changed) is int and type(total) is int and total > 0:
        return changed / total
    return None


def _observation_min_budget(obs: dict[str, Any], boundary: tuple[int, ...]) -> int | None:
    """Minimal token budget at which one recorded observation passes.

    For an observation that passed natively with its boundary first
    occurring at token offset ``cut``, it passes at every budget
    ``b >= cut + len(boundary)`` (the boundary occurrence and the answer
    region are unchanged) and at no smaller budget (``cut`` is the first
    occurrence, so no complete boundary fits in a shorter prefix) --
    exact under greedy decoding, by the same post-hoc argument as the
    stop matrix. Returns None for observations that did not pass
    natively (truncation cannot create a boundary or repair the answer
    region) and for malformed records. The ``status == "completed"`` gate
    is defense-in-depth: the ``Observation`` construction site already
    gates ``exact_match`` on it, so on legitimate records this never
    excludes a pass (see
    tests/test_cycle129_selection_consistency_audit.py).
    """
    if not (
        type(obs) is dict
        and obs.get("status") == "completed"
        and obs.get("exact_match") is True
        and obs.get("boundary_found") is True
    ):
        return None
    raw = obs.get("output_ids")
    output_ids = tuple(raw) if isinstance(raw, (list, tuple)) else ()
    if not output_ids or not all(type(token) is int for token in output_ids):
        return None
    cut = first_boundary(output_ids, boundary)
    if cut is None:
        return None
    return cut + len(boundary)


def budget_frontier(
    search: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    boundary: tuple[int, ...],
    max_new_tokens: int,
    native_decision: str,
    native_winner: str | None,
) -> dict[str, Any]:
    """Token-budget robustness frontier for the search stage.

    Recomputes, from the recorded search observations alone, what the pass
    counts -- and the selection decision -- would have been at every
    smaller token budget, answering "was the token cap load-bearing for
    this verdict?". No model calls: truncating a recorded greedy output to
    budget ``b`` is exactly what generating with the cap at ``b`` would have
    produced, by the same post-hoc argument as the stop matrix.

    The derivation is exact, not sampled. For an observation that passed
    natively with the boundary first occurring at token offset ``cut``, it
    passes at every budget ``b >= cut + len(boundary)`` (the boundary
    occurrence and the answer region are unchanged) and at no smaller
    budget (``cut`` is the first occurrence, so no complete boundary fits
    in a shorter prefix). An observation that did not pass natively cannot
    pass at any smaller budget: truncation removes tokens, so it cannot
    create a boundary or repair the answer region. Each candidate is
    therefore summarized by its per-fixture minimal passing budgets, from
    which the pass curve and the per-budget selection decision follow
    arithmetically.

    Returns per-candidate minimal budgets and pass-count step points,
    per-budget decision step points, the budget ranges where the decision
    or winner differs from native (``flip_ranges``), ``budget_robust``,
    and ``safe_budget_floor`` -- the first budget from which the native
    decision never flips again (1 when nothing flips, None when a flip
    span reaches the sweep maximum) -- and ``flip_attribution``: for each
    flip span, the (candidate, fixture) observations whose minimal passing
    budget falls inside the span, sorted by (min_budget, candidate,
    fixture). Pass counts can only change at those marks, so these are
    exactly the pass-status changes the span brackets: the observations to
    inspect when asking why the decision flips there.

    Structural fact: the per-budget full-passer set only grows with the
    budget, so once the native decision is reached it can never be left
    again -- ``flip_ranges`` is always ``[]`` or a single prefix span
    ``[[1, K]]`` with ``K < max_new_tokens``. The merge logic below keeps
    the general form; the invariant is pinned by
    ``test_flip_ranges_is_always_a_single_prefix_span``.

    Consistency: the post-hoc ``decision_at`` used here is exactly the
    pipeline's selection rule (``experiments._select_winner`` over the
    candidates with every search fixture passing) within this function's
    contract of a completed search decision. This rests on three
    invariants, pinned by
    ``tests/test_cycle129_selection_consistency_audit.py``: (1)
    ``exact_match`` is gated on ``status == "completed"`` at the single
    ``Observation`` construction site, so the pass predicate here is
    exactly ``_summary``'s; the explicit status check below is
    defense-in-depth for foreign payloads and can only under-count. (2)
    ``aggregate_eligible`` reports cover every planned fixture, so the
    per-candidate fixture count here is the pipeline's ``planned_count``.
    (3) A passing observation's boundary completes inside its recorded
    output, so every ``min_budget <= max_new_tokens`` and the post-hoc
    full-passer set at the sweep maximum is the native one --
    ``decision_at(max_new_tokens)`` is the recorded native decision by
    construction.
    When a search entry carries its own
    ``boundary_token_ids`` (per-candidate boundaries, Cycle 2), that entry's
    curve is derived with its own boundary; entries without the field use
    the ``boundary`` argument.
    Raises ValueError when the search did not complete (there is no native
    decision to compare against) or the inputs are malformed.
    """
    if type(search) not in (list, tuple) or not search:
        raise ValueError("search must be a nonempty list of result entries")
    if not _valid_token_ids(boundary, nonempty=True):
        raise ValueError("Boundary must be a nonempty tuple of token IDs")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 4096:
        raise ValueError("max_new_tokens must be an integer between 1 and 4096")
    if native_decision not in (
        "compatible_on_fixture_set",
        "held_out_failed",
        "ambiguous",
        "no_candidate",
    ):
        raise ValueError("budget_frontier needs a completed search decision")
    names: list[str] = []
    per_candidate_min: dict[str, list[int | None]] = {}
    fixture_ids: list[str] = []
    for entry in search:
        if type(entry) is not dict:
            raise ValueError("search entries must be dicts")
        fmt = entry.get("format")
        name = fmt.get("name") if type(fmt) is dict else None
        if type(name) is not str or not name:
            raise ValueError("each search entry needs a named format")
        if name in per_candidate_min:
            raise ValueError("candidate names must be unique")
        observations = entry.get("observations")
        if type(observations) is not list or not observations:
            raise ValueError("each search entry needs a nonempty observations list")
        names.append(name)
        # Per-candidate boundaries (Cycle 2): a search entry records the
        # boundary its candidate was measured with; entries written before
        # that field existed fall back to the single ``boundary`` argument.
        entry_boundary = entry.get("boundary_token_ids")
        if entry_boundary is None:
            entry_boundary = boundary
        elif isinstance(entry_boundary, list):
            entry_boundary = tuple(entry_boundary)
        if not _valid_token_ids(entry_boundary, nonempty=True):
            raise ValueError("search entry boundary_token_ids must be a nonempty ID tuple")
        min_budgets: list[int | None] = []
        ids: list[str] = []
        for obs in observations:
            if type(obs) is not dict:
                raise ValueError("observations must be dicts")
            ids.append(str(obs.get("fixture")))
            min_budgets.append(_observation_min_budget(obs, entry_boundary))
        if fixture_ids and ids != fixture_ids:
            raise ValueError("search entries must cover the same fixtures in order")
        fixture_ids = ids
        per_candidate_min[name] = min_budgets
    n_fixtures = len(fixture_ids)
    if native_decision in ("compatible_on_fixture_set", "held_out_failed"):
        if type(native_winner) is not str or native_winner not in per_candidate_min:
            raise ValueError("native_winner must name the selected search candidate")
        native_search: tuple[str, str | None] = ("selected", native_winner)
    else:
        if native_winner is not None:
            raise ValueError("native_winner must be None without a selected candidate")
        native_search = (
            ("ambiguous", None) if native_decision == "ambiguous" else ("no_candidate", None)
        )

    sorted_mins: dict[str, list[int]] = {
        name: sorted(mb for mb in mins if mb is not None)
        for name, mins in per_candidate_min.items()
    }

    def pass_count(name: str, budget: int) -> int:
        return bisect_right(sorted_mins[name], budget)

    candidates: dict[str, dict[str, Any]] = {}
    change_budgets: set[int] = set()
    for name in names:
        min_budgets = per_candidate_min[name]
        marks = sorted({mb for mb in min_budgets if mb is not None and 1 <= mb <= max_new_tokens})
        steps: list[list[int]] = []
        previous = -1
        for budget in dict.fromkeys([1, *marks, max_new_tokens]):
            count = pass_count(name, budget)
            if count != previous:
                steps.append([budget, count])
                previous = count
            change_budgets.add(budget)
        candidates[name] = {
            "fixture_ids": fixture_ids,
            "min_budget_per_fixture": min_budgets,
            "pass_steps": steps,
            "full_pass_budget": next((b for b, c in steps if c == n_fixtures), None),
        }

    def decision_at(budget: int) -> tuple[str, str | None]:
        winners = [name for name in names if pass_count(name, budget) == n_fixtures]
        if len(winners) == 1:
            return ("selected", winners[0])
        if not winners:
            return ("no_candidate", None)
        return ("ambiguous", None)

    decision_steps: list[list[Any]] = []
    previous_decision: tuple[str, str | None] | None = None
    for budget in sorted(change_budgets):
        current = decision_at(budget)
        if current != previous_decision:
            decision_steps.append([budget, current[0], current[1]])
            previous_decision = current
    # The decision is constant between change budgets, so flip intervals are
    # exact: [change_i, change_{i+1} - 1].
    ordered = sorted(change_budgets)
    flip_ranges: list[list[int]] = []
    for index, start in enumerate(ordered):
        end = ordered[index + 1] - 1 if index + 1 < len(ordered) else max_new_tokens
        if start > end or decision_at(start) == native_search:
            continue
        if flip_ranges and flip_ranges[-1][1] == start - 1:
            flip_ranges[-1][1] = end
        else:
            flip_ranges.append([start, end])
    # The safe budget floor is exact by construction: flip_ranges are
    # merged, sorted spans over 1..max_new_tokens where the decision
    # differs from native, so the native decision holds unchanged from
    # the budget after the last span onward -- or nowhere, when a span
    # reaches the sweep maximum.
    safe_floor: int | None
    if not flip_ranges:
        safe_floor = 1
    elif flip_ranges[-1][1] < max_new_tokens:
        safe_floor = flip_ranges[-1][1] + 1
    else:
        safe_floor = None
    # Flip-span attribution is exact by the same construction: pass counts
    # -- and hence the decision -- can only change at a candidate's
    # min_budget marks, so every observation whose min_budget falls inside
    # a flip span is a pass-status change that span brackets. These are
    # reported as the span's candidate causes: the observations to inspect
    # when asking why the decision flips there. They are not proven sole
    # causes -- a span's non-native decision is the joint effect of all
    # crossings up to that budget -- so the docstring states the weaker,
    # exact reading.
    flip_attribution: list[dict[str, Any]] = []
    for span_lo, span_hi in flip_ranges:
        crossings: list[dict[str, Any]] = []
        for name in sorted(candidates):
            entry = candidates[name]
            for fixture_id, min_budget in zip(
                entry["fixture_ids"], entry["min_budget_per_fixture"], strict=True
            ):
                if min_budget is not None and span_lo <= min_budget <= span_hi:
                    crossings.append(
                        {
                            "candidate": name,
                            "fixture": fixture_id,
                            "min_budget": min_budget,
                        }
                    )
        crossings.sort(
            key=lambda crossing: (
                crossing["min_budget"],
                crossing["candidate"],
                crossing["fixture"],
            )
        )
        flip_attribution.append({"span": [span_lo, span_hi], "crossings": crossings})
    return {
        "max_new_tokens": max_new_tokens,
        "native_decision": native_decision,
        "native_search_decision": native_search[0],
        "native_winner": native_search[1],
        "candidates": candidates,
        "decision_steps": decision_steps,
        "flip_ranges": flip_ranges,
        "budget_robust": not flip_ranges,
        "safe_budget_floor": safe_floor,
        "flip_attribution": flip_attribution,
    }


def held_out_budget_floor(
    observations: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    boundary: tuple[int, ...],
    max_new_tokens: int,
    native_verdict: str,
) -> dict[str, Any]:
    """Post-hoc token-budget floor for the held-out verification stage.

    Recomputes, from the recorded held-out observations alone, at which
    token budgets the verification verdict would still have held --
    answering "was the token cap load-bearing for the verification?".
    No model calls, by the same greedy prefix-determinism argument as
    :func:`budget_frontier`.

    The held-out verdict is monotone in the budget: a natively-passing
    observation passes at every budget ``b >= min_budget`` and at no
    smaller one, so the verdict flips from ``held_out_failed`` to
    ``compatible_on_fixture_set`` at most once -- at the maximum of the
    per-fixture minimal budgets. Returns the per-fixture minimal budgets,
    the native verdict, ``safe_budget_floor`` (that flip budget, or None
    when the held-out did not pass natively: no budget achieves a passing
    verification), and ``budget_robust`` (the verdict is identical at
    every budget 1..max_new_tokens -- true exactly when the floor is 1,
    or when the held-out failed natively and the failure holds
    everywhere).

    Raises ValueError when the observations are malformed or
    inconsistent with ``native_verdict`` (a compatible verdict alongside
    a failing fixture, or a failed verdict with every fixture passing).
    """
    if type(observations) not in (list, tuple) or not observations:
        raise ValueError("observations must be a nonempty list of observation dicts")
    if not _valid_token_ids(boundary, nonempty=True):
        raise ValueError("Boundary must be a nonempty tuple of token IDs")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 4096:
        raise ValueError("max_new_tokens must be an integer between 1 and 4096")
    if native_verdict not in ("compatible_on_fixture_set", "held_out_failed"):
        raise ValueError("native_verdict must be a completed held-out verdict")
    fixture_ids: list[str] = []
    min_budgets: list[int | None] = []
    for obs in observations:
        if type(obs) is not dict:
            raise ValueError("observations must be dicts")
        fixture_ids.append(str(obs.get("fixture")))
        min_budgets.append(_observation_min_budget(obs, boundary))
    passing = [mb for mb in min_budgets if mb is not None]
    all_pass = len(passing) == len(min_budgets)
    if native_verdict == "compatible_on_fixture_set" and not all_pass:
        raise ValueError("compatible verdict is inconsistent with failing observations")
    if native_verdict == "held_out_failed" and all_pass:
        raise ValueError("failed verdict is inconsistent with all-passing observations")
    floor: int | None
    if native_verdict == "compatible_on_fixture_set":
        # All min_budgets are non-None here; the verdict flips to
        # compatible exactly at their maximum (monotone by construction).
        floor = max(passing)
        robust = floor == 1
    else:
        # A natively-failing fixture can never pass at a smaller budget,
        # so the failed verdict holds at every budget in the sweep.
        floor = None
        robust = True
    return {
        "max_new_tokens": max_new_tokens,
        "native_verdict": native_verdict,
        "fixture_ids": fixture_ids,
        "min_budget_per_fixture": min_budgets,
        "safe_budget_floor": floor,
        "budget_robust": robust,
    }


LOGPROB_DERAILED_MEAN_THRESHOLD = -2.0
"""Mean-logprob band edge (nats/token) for the log-likelihood diagnostic.

A natively-failed fixture whose expected answer scores at or above this
mean teacher-forced logprob reads as "greedy derailed": the weights assign
the expected answer high likelihood, yet greedy generation diverged -- a
decoding/search problem rather than missing knowledge. A declared
heuristic, reported in the diagnostic payload's ``thresholds``.
"""

LOGPROB_UNKNOWN_MEAN_THRESHOLD = -5.0
"""Mean-logprob band edge (nats/token) for the log-likelihood diagnostic.

A natively-failed fixture whose expected answer scores below this mean
teacher-forced logprob reads as "weights don't know": even forced, the
checkpoint assigns the expected answer low likelihood under this
candidate's framing. Between the two bands the diagnostic withholds
judgment instead of guessing. A declared heuristic, reported in the
diagnostic payload's ``thresholds``.
"""

# Fixed vocabularies for the diagnostic payload: backend/fixture text must
# never reach them.
_LOGPROB_NOT_SCORED = "not_scored"
_LOGPROB_INCOMPLETE = "observation_incomplete"
_LOGPROB_ENCODE_FAILED = "encode_failed"


def _logprob_unavailable(reason: str, prompt_len: int) -> TargetScore:
    """Build an engine-shaped refusal for caller-side problems (e.g. the
    expected answer could not be encoded): status ``unavailable`` with a
    fixed reason, never an exception."""
    return TargetScore(
        status="unavailable",
        reason=reason,
        per_token_logprobs=(),
        total_logprob=None,
        mean_logprob=None,
        scored_token_count=0,
        target_byte_length=0,
        prompt_token_count=prompt_len,
    )


def _score_expected_answer(
    backend: Backend, prompt_ids: tuple[int, ...], expected: str
) -> TargetScore:
    """Teacher-force one fixture's expected answer; never raises on
    fixture/backend data.

    Every engine-level refusal (alignment, forward, math) comes back as a
    ``TargetScore`` with status ``unavailable`` and a fixed reason; an
    unencodable expected answer is reported the same way
    (``encode_failed``) instead of raising.
    """
    try:
        target_ids: Any = backend.encode(expected)
    except Exception:  # noqa: BLE001 -- encode problems become a fixed code
        return _logprob_unavailable(_LOGPROB_ENCODE_FAILED, len(prompt_ids))
    if type(target_ids) is not tuple or any(type(t) is not int for t in target_ids):
        return _logprob_unavailable(_LOGPROB_ENCODE_FAILED, len(prompt_ids))
    return score_target_logprobs(backend, prompt_ids, target_ids)


def _aggregate_fixture_scores(scores: list[TargetScore]) -> dict[str, Any]:
    """Candidate-level roll-up over per-fixture ``TargetScore`` values (§9.4).

    Reports the summed total log-likelihood, the token-count-weighted mean
    per scored token, the scored token count, and the target byte length, so
    token counts that differ across candidates stay visible. Only ``ok``
    entries contribute; ``unavailable``/``ok_empty`` never leak into the
    sums. NaN/inf degrades the affected field to None instead of poisoning
    ``results.json`` (written with ``allow_nan=False``); the engine already
    rejects non-finite logits, this is defense in depth.
    """
    scored_fixtures = 0
    with_total = 0
    total = 0.0
    tokens = 0
    byte_length = 0
    for score in scores:
        if score.status != STATUS_OK:
            continue
        scored_fixtures += 1
        tokens += score.scored_token_count
        byte_length += score.target_byte_length
        if type(score.total_logprob) is float:
            with_total += 1
            total += score.total_logprob
    total_out: float | None = None
    mean_out: float | None = None
    if scored_fixtures and with_total == scored_fixtures and tokens:
        total_out = total
        mean_out = total / tokens
    # NaN/inf degrades the affected fields to None instead of poisoning
    # results.json (written with allow_nan=False); the engine already
    # rejects non-finite logits, this is defense in depth. A finite total
    # with a non-finite mean is incoherent: withhold both.
    if (
        total_out is None
        or mean_out is None
        or not math.isfinite(total_out)
        or not math.isfinite(mean_out)
    ):
        total_out = None
        mean_out = None
    return {
        "scored_fixtures": scored_fixtures,
        "total_logprob": total_out,
        "mean_logprob": mean_out,
        "scored_token_count": tokens,
        "target_byte_length": byte_length,
    }


def _plural_fixtures(count: int) -> str:
    return f"{count} fixture" + ("" if count == 1 else "s")


def _interpret_logprob_candidate(fixture_scores: list[dict[str, Any]]) -> dict[str, str]:
    """Plain-language reading of one candidate's diagnostic (§9.3-§9.5).

    Only natively-failed fixtures with an ``ok`` score classify: the
    question is why a failure happened, and only a scored likelihood can
    answer it. Mean logprob >= ``LOGPROB_DERAILED_MEAN_THRESHOLD``
    nats/token reads as "greedy derailed" (the weights assign the expected
    answer high likelihood, yet greedy generation diverged); mean <
    ``LOGPROB_UNKNOWN_MEAN_THRESHOLD`` reads as "weights don't know";
    between the bands the diagnostic withholds judgment rather than
    guessing.
    """
    failures = [s for s in fixture_scores if not s["native_pass"]]
    scored = [s for s in failures if s["status"] == "ok" and type(s["mean_logprob"]) is float]
    if not failures:
        return {
            "kind": "no_failures",
            "text": (
                "Every search fixture passed natively, so there is no failure "
                "for this diagnostic to explain. Likelihoods are reported for "
                "reference only."
            ),
        }
    if not scored:
        return {
            "kind": "unavailable",
            "text": (
                "No failing fixture could be scored -- the alignment check or "
                "the forward pass refused every one -- so the diagnostic "
                "reports nothing rather than guessing."
            ),
        }
    highs = [s for s in scored if s["mean_logprob"] >= LOGPROB_DERAILED_MEAN_THRESHOLD]
    lows = [s for s in scored if s["mean_logprob"] < LOGPROB_UNKNOWN_MEAN_THRESHOLD]
    total = sum(s["total_logprob"] for s in scored if type(s["total_logprob"]) is float)
    tokens = sum(s["scored_token_count"] for s in scored if type(s["scored_token_count"]) is int)
    mean = total / tokens if tokens else None
    mean_text = f"{mean:.2f}" if mean is not None else "n/a"
    if highs and not lows:
        return {
            "kind": "greedy_derailed",
            "text": (
                "Likely greedy derailment: the weights assign the expected "
                f"answer high teacher-forced likelihood (mean {mean_text} "
                f"nats/token over {_plural_fixtures(len(scored))} failing), "
                "yet greedy generation did not reproduce it. The failure "
                "looks like a decoding/search problem, not missing knowledge."
            ),
        }
    if lows and not highs:
        return {
            "kind": "weights_dont_know",
            "text": (
                "Likely missing knowledge: even teacher-forced, the weights "
                f"assign the expected answer low likelihood (mean {mean_text} "
                f"nats/token over {_plural_fixtures(len(scored))} failing). "
                "This candidate's framing does not make the expected answer "
                "probable under the checkpoint."
            ),
        }
    if highs and lows:
        return {
            "kind": "mixed",
            "text": (
                f"Mixed causes: {_plural_fixtures(len(highs))} failing score "
                "in the high band (greedy-derailment pattern) while "
                f"{_plural_fixtures(len(lows))} score in the low band "
                "(missing-knowledge pattern). The failures have more than "
                "one cause."
            ),
        }
    return {
        "kind": "indeterminate",
        "text": (
            "Indeterminate: the failing fixtures' mean likelihoods fall "
            "between the declared bands (below "
            f"{LOGPROB_DERAILED_MEAN_THRESHOLD:.1f} yet at or above "
            f"{LOGPROB_UNKNOWN_MEAN_THRESHOLD:.1f} nats/token), so the "
            "diagnostic withholds judgment."
        ),
    }


def _diagnostic_observation(
    backend: Backend, obs: Any, expected_by_id: dict[str, str]
) -> tuple[dict[str, Any], TargetScore | None]:
    """Score one recorded observation's fixture; never raises on bad data.

    Returns the JSON-ready fixture-score dict and the engine ``TargetScore``
    (None when the observation itself cannot be scored). The engine score
    feeds the candidate aggregate; the dict feeds the report.
    """
    fixture_id = obs.get("fixture") if type(obs) is dict else None
    label = fixture_id if type(fixture_id) is str else "?"
    native_pass = type(obs) is dict and bool(obs.get("exact_match") and obs.get("boundary_found"))
    not_scored: dict[str, Any] = {
        "fixture": label,
        "status": _LOGPROB_NOT_SCORED,
        "reason": _LOGPROB_INCOMPLETE,
        "total_logprob": None,
        "mean_logprob": None,
        "scored_token_count": 0,  # nosec B105 -- integer counter field, not a credential
        "target_byte_length": 0,
        "native_pass": native_pass,
    }
    if type(obs) is not dict or obs.get("status") != "completed":
        return not_scored, None
    raw_ids = obs.get("input_ids")
    prompt_ids = tuple(raw_ids) if isinstance(raw_ids, (list, tuple)) else ()
    if (
        not prompt_ids
        or any(type(t) is not int or t < 0 for t in prompt_ids)
        or type(fixture_id) is not str
        or fixture_id not in expected_by_id
    ):
        return not_scored, None
    score = _score_expected_answer(backend, prompt_ids, expected_by_id[fixture_id])
    scored: dict[str, Any] = {
        "fixture": label,
        "status": score.status,
        "reason": score.reason,
        "total_logprob": score.total_logprob,
        "mean_logprob": score.mean_logprob,
        "scored_token_count": score.scored_token_count,
        "target_byte_length": score.target_byte_length,
        "native_pass": native_pass,
    }
    return scored, score


def logprob_diagnostic(
    backend: Backend,
    result: dict[str, Any],
    fixtures: tuple[Fixture, ...] | list[Fixture],
    *,
    cancel: Event | None = None,
) -> dict[str, Any]:
    """Teacher-forced log-likelihood diagnostic over recorded search observations.

    For every search candidate and every search fixture, the fixture's
    *expected answer* is teacher-forced through ``backend`` after the
    recorded prompt: ``score_target_logprobs(backend, prompt_ids,
    encode(expected))``. This separates "the weights don't know the answer"
    (low likelihood even when forced) from "greedy derailed" (high
    likelihood, yet the recorded greedy generation diverged) -- the
    DEVELOPMENT_PLAN.md §9.3-§9.5 design.

    Diagnostic only: the input ``result`` is never mutated and the selection
    rule is untouched; the caller stores the returned dict under
    ``result["logprob_diagnostic"]``. One forward pass per candidate x
    fixture, which is why the CLI flag is opt-in. Alignment refusals surface
    per fixture as status ``unavailable`` with a fixed reason -- never a
    silent guess. Returns ``available: False`` with a fixed ``detail`` when
    the backend exposes no forward-logits primitive or no candidate could
    be scored.

    ``cancel`` is an optional cooperative ``threading.Event`` (same contract
    as ``evaluate``/``run_experiment``): when set, scoring stops at the next
    fixture boundary -- the expensive unit is one forward pass -- and the
    payload reports ``available: False`` with the fixed ``detail``
    ``"cancelled"``. Candidates fully scored before the cancel are kept;
    the in-progress candidate is dropped, never partially aggregated.
    """
    if type(result) is not dict or type(result.get("search")) is not list:
        raise ValueError("result must be a run_experiment result dict")
    if isinstance(fixtures, list):
        fixtures = tuple(fixtures)
    if type(fixtures) is not tuple or any(type(item) is not Fixture for item in fixtures):
        raise ValueError("fixtures must be a tuple of Fixture")
    if cancel is not None and not isinstance(cancel, Event):
        raise ValueError("cancel must be a threading.Event")
    event = cancel if cancel is not None else Event()
    expected_by_id = {item.id: item.expected for item in fixtures}
    payload: dict[str, Any] = {
        "available": False,
        "detail": None,
        "method": (
            "Teacher-forced log-likelihood of each fixture's expected answer: "
            "one forward pass per candidate x fixture over prompt_ids + "
            "encode(expected); prompt positions are masked and answer tokens "
            "are gathered with the causal shift (logits at t-1 predict token "
            "t). Diagnostic only; never changes the selection rule."
        ),
        "thresholds": {
            "derailed_mean_logprob_at_least": LOGPROB_DERAILED_MEAN_THRESHOLD,
            "unknown_mean_logprob_below": LOGPROB_UNKNOWN_MEAN_THRESHOLD,
        },
        "candidates": [],
    }
    if not callable(getattr(backend, "forward_logits", None)):
        payload["detail"] = "logits_not_supported"
        return payload
    for entry in result["search"]:
        if type(entry) is not dict:
            continue
        fmt = entry.get("format")
        name = fmt.get("name") if type(fmt) is dict else None
        observations = entry.get("observations")
        if type(name) is not str or type(observations) is not list:
            continue
        fixture_scores: list[dict[str, Any]] = []
        engine_scores: list[TargetScore] = []
        for obs in observations:
            if event.is_set():
                # Cooperative cancellation: stop before the next forward
                # pass. Candidates fully scored so far are kept; the
                # in-progress candidate is dropped, never partially
                # aggregated. available stays False with the fixed
                # "cancelled" detail, like the other unavailable paths.
                payload["detail"] = "cancelled"
                return payload
            score_dict, engine_score = _diagnostic_observation(backend, obs, expected_by_id)
            fixture_scores.append(score_dict)
            if engine_score is not None:
                engine_scores.append(engine_score)
        aggregate = _aggregate_fixture_scores(engine_scores)
        aggregate["native_passes"] = sum(1 for s in fixture_scores if s["native_pass"])
        aggregate["native_failures"] = sum(1 for s in fixture_scores if not s["native_pass"])
        aggregate["unavailable_fixtures"] = sum(
            1 for s in fixture_scores if s["status"] in ("unavailable", _LOGPROB_NOT_SCORED)
        )
        payload["candidates"].append(
            {
                "name": name,
                "fixture_scores": fixture_scores,
                "aggregate": aggregate,
                "interpretation": _interpret_logprob_candidate(fixture_scores),
            }
        )
    if not payload["candidates"]:
        payload["detail"] = "no_scorable_observations"
        return payload
    payload["available"] = True
    return payload


def prompt_sensitivity_sweep(
    backend: Backend,
    formats: tuple[Format, ...],
    split: Split,
    *,
    boundary: tuple[int, ...],
    boundary_map: dict[str, tuple[int, ...]] | None = None,
    max_drop_bytes: int = 3,
    max_new_tokens: int = 64,
    seed: int = 0,
    cancel: Event | None = None,
    native: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Re-run search with the prompt truncated mid-token at each depth.

    For ``drop_bytes`` in 1..``max_drop_bytes``, every prompt is truncated
    by that many trailing bytes and re-encoded (see
    ``core.truncate_text_tail``), then the full search re-runs under a
    ``TruncatedPromptBackend``. Reports the depths where the selected winner
    or decision differs from native: boundary sensitivity as a variable, not
    a constant. Depths that cannot be applied (e.g. a cut splitting a UTF-8
    codepoint) record the run's failed decision instead of silently
    skipping. That failed-depth-counts-as-flip behaviour is pinned by
    test_cycle17 and is deliberately NOT changed here; the token-depth
    sweep below carries the corrected status-gated semantics instead.

    ``native`` optionally supplies a ``run_experiment`` result for the
    same backend/formats/split/boundary to reuse as the baseline (the
    CLI shares one native between the byte and token sweeps, saving
    exactly one ``run_experiment`` versus two sweep calls). When
    omitted, the native run is recomputed, as before. A supplied
    result must carry the ``configuration_digest`` of exactly this
    sweep's configuration (see ``_require_native_match``): a
    hand-made dict or a result from another configuration is
    rejected, because shape alone used to let a foreign baseline
    through and its ``flips_at`` was then computed against a run
    that never happened here.
    """
    if type(max_drop_bytes) is not int or not 1 <= max_drop_bytes <= 8:
        raise ValueError("max_drop_bytes must be an integer between 1 and 8")
    _require_native_match(
        native,
        backend,
        formats,
        split,
        boundary=boundary,
        boundary_map=boundary_map,
        max_new_tokens=max_new_tokens,
        seed=seed,
    )
    if native is None:
        native = run_experiment(
            backend,
            formats,
            split,
            boundary=boundary,
            boundary_map=boundary_map,
            max_new_tokens=max_new_tokens,
            seed=seed,
            cancel=cancel,
        )
    native_winner = native["selected_candidate"]
    depths: list[dict[str, Any]] = []
    for drop in range(1, max_drop_bytes + 1):
        wrapped = TruncatedPromptBackend(backend, drop)
        mutated = run_experiment(
            wrapped,
            formats,
            split,
            boundary=boundary,
            boundary_map=boundary_map,
            max_new_tokens=max_new_tokens,
            seed=seed,
            cancel=cancel,
        )
        winner = mutated["selected_candidate"]
        changed = winner != native_winner or mutated["decision"] != native["decision"]
        depths.append(
            {
                "drop_bytes": drop,
                "decision": mutated["decision"],
                "selected_candidate": winner,
                "winner_changed": changed,
            }
        )
    return {
        "backend_fingerprint": _backend_fingerprint(backend),
        "native_decision": native["decision"],
        "native_winner": native_winner,
        "max_drop_bytes": max_drop_bytes,
        "depths": depths,
        "flips_at": [depth["drop_bytes"] for depth in depths if depth["winner_changed"]],
    }


def _token_depth_prompt_lengths(
    backend: Backend,
    formats: tuple[Format, ...],
    split: Split,
) -> list[int]:
    """Token lengths of every prompt the truncated run could encode.

    Renders every search and held_out prompt under every candidate and
    encodes it with the base backend. Prompts that fail to render or
    encode are skipped: they fail natively too, so they carry no
    truncation-specific information. The caller uses the minimum:
    a depth is inapplicable exactly when it meets or exceeds the
    shortest truncatable prompt, which is the condition under which
    the first un-truncatable prompt aborts the whole depth via the
    ``run_experiment`` early-return path.
    """
    lengths: list[int] = []
    for fmt in formats:
        for case in split.search + split.held_out:
            length = _prompt_token_length(backend, fmt, case)
            if length is not None:
                lengths.append(length)
    return lengths


def _prompt_token_length(
    backend: Backend,
    fmt: Format,
    case: Fixture,
) -> int | None:
    """Token length of one rendered prompt, or None when it fails
    natively: a prompt that fails to render or encode fails the
    native run too, so it carries no truncation-specific fact."""
    try:
        ids = backend.encode(fmt.render(case.messages))
    except Exception:  # noqa: BLE001 -- natively-failing prompt, not a truncation fact
        return None
    return len(ids) if type(ids) is tuple else None


def _suffix_token_counts(
    backend: Backend,
    formats: tuple[Format, ...],
) -> dict[str, int | None]:
    """Standalone-suffix token count per candidate, under this backend.

    The suffix is the trailing open-reply text (``open_suffix`` when
    set, else ``assistant_prefix``) -- the assistant turn-closing
    header whose whole-token removal the ladder is built to reach.
    The count is the standalone encoding of that text alone: an
    annotation mapping depth to header state, NOT a claim about how
    the suffix tokenizes inside a full prompt (BPE merges across the
    seam can differ). None when the suffix does not encode.
    """
    counts: dict[str, int | None] = {}
    for fmt in formats:
        suffix = fmt.open_suffix if fmt.open_suffix is not None else fmt.assistant_prefix
        try:
            ids = backend.encode(suffix)
        except Exception:  # noqa: BLE001 -- unencodable suffix is reported, not fatal
            counts[fmt.name] = None
        else:
            counts[fmt.name] = len(ids) if type(ids) is tuple else None
    return counts


def prompt_token_depth_sweep(
    backend: Backend,
    formats: tuple[Format, ...],
    split: Split,
    *,
    boundary: tuple[int, ...],
    boundary_map: dict[str, tuple[int, ...]] | None = None,
    max_drop_tokens: int = 4,
    max_new_tokens: int = 64,
    seed: int = 0,
    cancel: Event | None = None,
    native: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Re-run search with the prompt truncated by whole trailing tokens.

    For ``drop_tokens`` in 1..``max_drop_tokens``, every prompt loses
    that many trailing token ids (backend's own tokenization, via
    ``TokenTruncatedPromptBackend``; no decode round-trip) and the full
    search re-runs. Reports the depths where the selected winner or
    decision differs from native. This is the token-depth extension of
    ``prompt_sensitivity_sweep``: the byte sweep can only mangle the
    tail of a multi-token turn-closing suffix (byte cap 8), while this
    ladder can remove the whole suffix cleanly.

    Fail-closed status semantics (the byte sweep's known flaw, fixed
    here only -- byte mode is untouched): every depth carries a
    per-depth ``status``. ``status`` is ``"inapplicable"`` when the
    depth meets or exceeds the shortest truncatable prompt length,
    decided by a PRE-RUN token-length check over search and held_out
    prompts (a conservative superset: a depth whose truncated winner's
    held_out prompt alone was too short is labelled inapplicable
    rather than claimed as a flip, because the flip claim must never
    rest on a prompt that never ran). Inapplicable depths keep the
    run's ``"failed"`` decision for that depth, appear in
    ``inapplicable_at``, and are excluded from ``flips_at``;
    ``winner_changed`` is gated on ``status == "applied"``, so no
    consumer reading ``depths[]`` directly can mistake a never-run
    depth for a flip. A depth that was applied but whose run failed
    for another reason keeps ``status == "applied"`` and counts as a
    change when its decision differs from native -- that is a genuine
    truncated-run failure, not a labelling artefact.

    Eligibility: the measurement is defined on the base prompt
    encoding. The wrapper refuses prefill-wrapped inners at
    construction (dropping appended prefill ids first would measure
    the wrong thing). A backend whose ``generate`` rewrites ids
    (``TokenHealingBackend``) does not consume ids as given, so a
    token sweep over it measures healed ids, not prompt ids; such
    runs are plumbing-only, never evidence -- the fingerprint chain
    keeps the healing step visible. ``native`` optionally supplies
    a ``run_experiment`` result to reuse as the baseline, as in
    ``prompt_sensitivity_sweep`` — and under the same digest rule:
    the supplied result must carry the ``configuration_digest``
    of exactly this sweep's configuration (see
    ``_require_native_match``), or it is rejected.

    Default 4 / cap 8 are provisional working hypotheses (design
    Q1): freezing them awaits A4, per-candidate suffix token counts
    measured on a real checkpoint tokenizer. The payload's
    ``suffix_token_counts`` annotation is the mechanism for that
    measurement; no real-checkpoint count is claimed here.
    """
    if type(max_drop_tokens) is not int or not 1 <= max_drop_tokens <= 8:
        raise ValueError("max_drop_tokens must be an integer between 1 and 8")
    _require_native_match(
        native,
        backend,
        formats,
        split,
        boundary=boundary,
        boundary_map=boundary_map,
        max_new_tokens=max_new_tokens,
        seed=seed,
    )
    if native is None:
        native = run_experiment(
            backend,
            formats,
            split,
            boundary=boundary,
            boundary_map=boundary_map,
            max_new_tokens=max_new_tokens,
            seed=seed,
            cancel=cancel,
        )
    native_winner = native["selected_candidate"]
    lengths = _token_depth_prompt_lengths(backend, formats, split)
    min_prompt_tokens = min(lengths) if lengths else None
    depths: list[dict[str, Any]] = []
    for drop in range(1, max_drop_tokens + 1):
        status = (
            "inapplicable"
            if min_prompt_tokens is not None and drop >= min_prompt_tokens
            else "applied"
        )
        wrapped = TokenTruncatedPromptBackend(backend, drop)
        mutated = run_experiment(
            wrapped,
            formats,
            split,
            boundary=boundary,
            boundary_map=boundary_map,
            max_new_tokens=max_new_tokens,
            seed=seed,
            cancel=cancel,
        )
        winner = mutated["selected_candidate"]
        # NOTE: deliberately not the byte sweep's one-line ``changed``
        # formula: test_cycle17's mutation gate anchors on that exact
        # line text with an expected count of 1, so a second copy here
        # would break the gate's anchor even though the semantics are
        # the same. Keep the two spellings distinct.
        winner_differs = winner != native_winner
        decision_differs = mutated["decision"] != native["decision"]
        depths.append(
            {
                "drop_tokens": drop,
                "status": status,
                "decision": mutated["decision"],
                "selected_candidate": winner,
                "winner_changed": (winner_differs or decision_differs)
                if status == "applied"
                else False,
            }
        )
    return {
        "granularity": "tokens",
        "backend_fingerprint": _backend_fingerprint(backend),
        "native_decision": native["decision"],
        "native_winner": native_winner,
        "max_drop_tokens": max_drop_tokens,
        "min_prompt_tokens": min_prompt_tokens,
        "suffix_token_counts": _suffix_token_counts(backend, formats),
        "depths": depths,
        "flips_at": [depth["drop_tokens"] for depth in depths if depth["winner_changed"]],
        "inapplicable_at": [
            depth["drop_tokens"] for depth in depths if depth["status"] == "inapplicable"
        ],
    }


def candidate_fingerprint(fmt: Format) -> str:
    """Stable identity for a format hypothesis: name, markers and flags.

    Lets a report consumer verify that the candidate evaluated here is
    byte-identical — including its name — to a hypothesis recorded
    elsewhere, without trusting the candidate name alone. The ``name`` is
    part of the payload (unlike ``local_repairs``' dedup key, which drops
    it because ``repeat_bos`` is meaningless when ``bos`` is empty): two
    formats that render identically but carry different names therefore get
    different fingerprints. The behavioral fields are otherwise normalized
    exactly like the dedup key.
    """
    payload = asdict(fmt)
    if not payload["bos"]:
        payload["repeat_bos"] = False
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + sha256(text.encode("utf-8")).hexdigest()


def _failure_code(report: EvaluationReport) -> str | None:
    """Sanitized failure code of a failed evaluation, for the report banner.

    Scans the observations from the end for the first fixed-vocabulary
    error code (``render_failed``, ``encode_failed``, ``invalid_input_ids``,
    ``generate_failed``, ``backend_reported_error``, or a backend code such
    as ``context_limit_exceeded``). None when nothing failed or no code was
    recorded. The codes are a fixed vocabulary -- safe for the default
    evidence-excluded report, unlike raw prompts or generations.
    """
    for observation in reversed(report.observations):
        error = observation.error
        if type(error) is str and error:
            return error
    return None


def _summary(report: EvaluationReport) -> dict[str, Any]:
    passed = sum(o.exact_match and o.boundary_found for o in report.observations)
    return {
        "status": report.status,
        "planned": report.planned_count,
        "attempted": report.attempted_count,
        "aggregate_eligible": report.aggregate_eligible,
        "exact_and_boundary_passes": passed,
        "rate": passed / report.planned_count if report.aggregate_eligible else None,
        "encoded_input_tokens": sum(len(o.input_ids) for o in report.observations),
        "retained_output_tokens": sum(len(o.output_ids) for o in report.observations),
        "failure_code": _failure_code(report),
    }


def _resolve_boundaries(
    formats: tuple[Format, ...],
    boundary: tuple[int, ...],
    boundary_map: dict[str, tuple[int, ...]] | None,
) -> dict[str, tuple[int, ...]]:
    """Per-candidate response boundaries for a run.

    ``boundary`` is the default applied to every candidate; ``boundary_map``
    overrides it for the named candidates. This is scientifically the honest
    choice: each candidate is a format hypothesis, and real formats terminate
    the assistant turn with different markers (ChatML ``<|im_end|>``, Llama 3
    ``<|eot_id|>``, Gemma ``<end_of_turn>``), so judging every hypothesis by
    a single sequence would mis-measure the formats whose own terminator
    differs. The selection rule is unchanged -- exactly one candidate must
    pass every search fixture, each under its own boundary; ties and
    failures still abstain.
    """
    if boundary_map is None:
        return {fmt.name: boundary for fmt in formats}
    if type(boundary_map) is not dict:
        raise ValueError("boundary_map must be a dict or None")
    names = {fmt.name for fmt in formats}
    resolved: dict[str, tuple[int, ...]] = {}
    for name, ids in boundary_map.items():
        if type(name) is not str or name not in names:
            raise ValueError(f"boundary_map names an unknown candidate: {name!r}")
        if not _valid_token_ids(ids, nonempty=True):
            raise ValueError(f"boundary_map boundary for {name!r} must be a nonempty tuple")
        resolved[name] = ids
    for fmt in formats:
        resolved.setdefault(fmt.name, boundary)
    return resolved


def _select_winner(winners: list[Format], cancelled: bool) -> tuple[str, Format | None]:
    """Pure search-phase selection decision.

    The conservative rule: exactly one candidate passing every search
    fixture wins; ties and failures abstain; a cancelled run reports
    "cancelled". Kept pure (no backend, no I/O) so the decision table is
    exhaustively testable -- see
    tests/test_cycle120_selection_rule_properties.py. Returns the
    decision string and the selected candidate (None on abstention).
    """
    if cancelled:
        return "cancelled", None
    if len(winners) == 1:
        return "selected", winners[0]
    return ("ambiguous" if winners else "no_candidate"), None


def _backend_fingerprint(backend: Backend) -> str:
    """Return ``backend.fingerprint`` after the protocol type check.

    The probe stages build their protocol blocks from the fingerprint
    without calling :func:`run_experiment`, whose identical check
    (below) therefore never protected them: before cycle 202 they
    wrapped the attribute in ``str()``, which stringifies a protocol
    violation instead of rejecting it — a method-valued fingerprint
    (the cycle-195 fake shape) embedded a heap address in the payload
    and made two runs of the same backend produce different protocol
    fingerprints, silently. Every stage read site goes through here so
    the contract is the one ``run_experiment`` enforces: a non-string
    fingerprint is a ``ValueError``, never a stringified repr.
    """
    fingerprint = getattr(backend, "fingerprint", None)
    if type(fingerprint) is not str:
        raise ValueError("Backend fingerprint must be a string")
    return fingerprint


def _configuration_digest(
    backend: Backend,
    formats: tuple[Format, ...],
    split: Split,
    *,
    boundary: tuple[int, ...],
    boundary_map: dict[str, tuple[int, ...]] | None,
    max_new_tokens: int,
    seed: int,
    eos_id: int | None,
) -> str:
    """Stable identity of one ``run_experiment`` configuration.

    Binds the backend fingerprint, the candidate fingerprints (in
    order), the full fixture content of both splits (id, messages,
    expected answer, group — ids alone would collide across
    perturbed fixtures sharing an id), the boundary and per-format
    boundary map, ``max_new_tokens``, ``seed``, and ``eos_id``.
    Two runs of the same configuration produce the same digest;
    any configuration change produces a different one. The sweeps
    stamp-check a supplied ``native`` against the digest of their
    own parameters, so a baseline computed for another
    configuration — or a hand-made dict, which has no digest at
    all — is rejected instead of silently compared (cycle 207,
    wave-10 B10-2). Fixture content, not just ids, is hashed
    because the digest's job is to prove the baseline ran on
    these fixtures, not merely on fixtures with these ids.
    """

    def fixture_digest(case: Fixture) -> dict[str, Any]:
        return {
            "id": case.id,
            "messages": [asdict(message) for message in case.messages],
            "expected": case.expected,
            "group": case.group,
        }

    payload = {
        "backend_fingerprint": _backend_fingerprint(backend),
        "candidate_fingerprints": [candidate_fingerprint(fmt) for fmt in formats],
        "search": [fixture_digest(case) for case in split.search],
        "held_out": [fixture_digest(case) for case in split.held_out],
        "boundary_token_ids": list(boundary),
        "boundary_map": (
            {name: list(ids) for name, ids in sorted(boundary_map.items())}
            if boundary_map is not None
            else None
        ),
        "max_new_tokens": max_new_tokens,
        "seed": seed,
        "eos_id": eos_id,
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + sha256(text.encode("utf-8")).hexdigest()


def _require_native_match(
    native: dict[str, Any] | None,
    backend: Backend,
    formats: tuple[Format, ...],
    split: Split,
    *,
    boundary: tuple[int, ...],
    boundary_map: dict[str, tuple[int, ...]] | None,
    max_new_tokens: int,
    seed: int,
) -> None:
    """Reject a supplied sweep baseline that is not this run's native.

    A supplied ``native`` must be a ``run_experiment`` result
    carrying the ``configuration_digest`` of exactly the
    configuration the sweep itself would compute (same backend,
    formats, split, boundary, seed, ``max_new_tokens``; the sweeps
    take no ``eos_id``, so the expected digest uses ``None``).
    Shape alone used to suffice — a hand-made dict with the two
    keys, or a genuine result from another configuration, was
    silently accepted and its ``flips_at`` computed against a
    baseline that never ran here. A missing digest (hand-made
    dicts and pre-cycle-207 artifacts) fails closed, as does a
    mismatch. ``None`` passes: the sweep then recomputes its own
    native, which carries the digest by construction.
    """
    if native is None:
        return
    if type(native) is not dict or "selected_candidate" not in native or "decision" not in native:
        raise ValueError("native must be a run_experiment result dict")
    expected = _configuration_digest(
        backend,
        formats,
        split,
        boundary=boundary,
        boundary_map=boundary_map,
        max_new_tokens=max_new_tokens,
        seed=seed,
        eos_id=None,
    )
    if "configuration_digest" not in native:
        raise ValueError("native is missing its configuration_digest")
    if native["configuration_digest"] != expected:
        raise ValueError("native configuration_digest does not match this sweep configuration")


def run_experiment(
    backend: Backend,
    formats: tuple[Format, ...],
    split: Split,
    *,
    boundary: tuple[int, ...],
    boundary_map: dict[str, tuple[int, ...]] | None = None,
    max_new_tokens: int = 64,
    seed: int = 0,
    cancel: Event | None = None,
    eos_id: int | None = None,
) -> dict[str, Any]:
    """Run the full protocol experiment and return the result dict.

    Renders each candidate format over the search split, generates with the
    backend (greedy, exact token sequences), scores exact answers and
    response boundaries, then applies the conservative selection rule: one
    candidate must pass every search fixture or the decision abstains
    ("ambiguous" on ties, "no_candidate" when all fail). The returned dict
    carries the decision, per-candidate observations, and the failure
    taxonomy on every return path, plus ``configuration_digest``: the
    stable identity of this run's configuration (see
    ``_configuration_digest``), stamped on every return path so a
    later consumer can prove a baseline belongs to this
    configuration before comparing against it.
    """

    if type(formats) is not tuple or not 1 <= len(formats) <= 24:
        raise ValueError("Supply between 1 and 24 candidates")
    if any(type(fmt) is not Format for fmt in formats):
        raise ValueError("Invalid candidate type")
    if len({fmt.name for fmt in formats}) != len(formats):
        raise ValueError("Candidate names must be unique")
    if type(split) is not Split:
        raise ValueError("A validated split is required")
    if cancel is not None and not isinstance(cancel, Event):
        raise ValueError("cancel must be a threading.Event")
    if eos_id is not None and (type(eos_id) is not int or eos_id < 0):
        raise ValueError("eos_id must be a nonnegative integer or None")
    if type(getattr(backend, "fingerprint", None)) is not str:
        # Fail fast on protocol-violating backends before any compute: a
        # non-string fingerprint would otherwise explode in json.dumps inside
        # save_report, after hours of generation, as an uncaught TypeError.
        # getattr (not attribute access) so a backend missing the attribute
        # entirely raises ValueError here instead of a raw AttributeError.
        raise ValueError("Backend fingerprint must be a string")
    if not _valid_token_ids(boundary, nonempty=True):
        raise ValueError("Boundary must be a nonempty tuple of token IDs")
    if len(split.search) > 100 or len(split.held_out) > 100:
        raise ValueError("Experiment fixture limit exceeded")
    boundaries = _resolve_boundaries(formats, boundary, boundary_map)
    event = cancel if cancel is not None else Event()
    digest = _configuration_digest(
        backend,
        formats,
        split,
        boundary=boundary,
        boundary_map=boundary_map,
        max_new_tokens=max_new_tokens,
        seed=seed,
        eos_id=eos_id,
    )
    output: dict[str, Any] = {
        "schema_version": 1,
        "selection_rule": "unique_all_search_pass",
        "measurement": "exact_answer_and_boundary",
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "boundary_token_ids": list(boundary),
        "eos_token_id": eos_id,
        "backend_fingerprint": backend.fingerprint,
        "configuration_digest": digest,
        "statistical_confidence": "not_estimated",
        "compute_comparison": "not_equal_compute_certified",
        "decision": "not_started",
        "search": [],
        "held_out": None,
        "selected_candidate": None,
    }

    def finish() -> dict[str, Any]:
        # Every return path carries the failure taxonomy so the decision
        # banner can cite it; it is deterministic from the stored records.
        output["failure_taxonomy"] = failure_taxonomy(output, eos_id)
        return output

    winners = []
    for fmt in formats:
        candidate_boundary = boundaries[fmt.name]
        report = evaluate(
            backend,
            fmt,
            split.search,
            boundary=candidate_boundary,
            max_new_tokens=max_new_tokens,
            seed=seed,
            cancel=event,
        )
        summary = _summary(report)
        output["search"].append(
            {
                "format": asdict(fmt),
                "candidate_fingerprint": candidate_fingerprint(fmt),
                "boundary_token_ids": list(candidate_boundary),
                "summary": summary,
                "observations": [asdict(o) for o in report.observations],
            }
        )
        if not report.aggregate_eligible:
            output["decision"] = report.status
            return finish()
        if summary["exact_and_boundary_passes"] == report.planned_count:
            winners.append(fmt)
    decision, selected = _select_winner(winners, event.is_set())
    if selected is None:
        output["decision"] = decision
        return finish()
    output["selected_candidate"] = selected.name
    selected_boundary = boundaries[selected.name]
    output["selected_boundary_token_ids"] = list(selected_boundary)
    held = evaluate(
        backend,
        selected,
        split.held_out,
        boundary=selected_boundary,
        max_new_tokens=max_new_tokens,
        seed=seed,
        cancel=event,
    )
    output["held_out"] = {
        "summary": _summary(held),
        "observations": [asdict(o) for o in held.observations],
    }
    if not held.aggregate_eligible:
        output["decision"] = held.status
    else:
        passed = all(o.exact_match and o.boundary_found for o in held.observations)
        output["decision"] = "compatible_on_fixture_set" if passed else "held_out_failed"
    return finish()


def portfolio_analysis(
    result: dict[str, Any],
    cases: tuple[Fixture, ...],
) -> dict[str, Any]:
    """Oracle-mixture / protocol-portfolio gap (pure post-hoc).

    Derives per-candidate per-fixture pass from the stored search
    observations (exact answer AND explicit boundary -- the run's own
    measurement, never re-scored) and reports:

    * the per-fixture oracle: passing candidates per fixture,
      oracle pass rate vs the global winner's rate, and the gap;
    * the greedy covering protocol portfolio over solvable fixtures.

    The oracle and the portfolio answer "how much does the single
    global winner leave on the table, and which small protocol set
    would cover every solvable fixture?" -- the routing lens of
    RouteLLM applied within the lab to prompt protocols, not models.
    Zero new model runs; fully deterministic.

    A candidate whose observations do not align with ``cases`` is
    excluded (one bad record cannot poison the verdicts). Never raises
    on malformed input: returns ``{"available": False}`` with a
    ``detail`` code.
    """
    if type(result) is not dict:
        return {"available": False, "detail": "result_not_a_dict"}
    entries = result.get("search")
    if type(entries) is not list or not entries:
        return {"available": False, "detail": "no_search_entries"}
    if type(cases) is not tuple or not cases or any(type(c) is not Fixture for c in cases):
        return {"available": False, "detail": "cases_invalid"}
    fixture_ids = tuple(c.id for c in cases)

    passing: dict[str, tuple[str, ...]] = {}
    excluded: list[str] = []
    for entry in entries:
        if type(entry) is not dict or type(entry.get("format")) is not dict:
            continue
        name = entry["format"].get("name")
        if type(name) is not str or not name:
            continue
        observations = entry.get("observations")
        try:
            if type(observations) is not list or len(observations) != len(cases):
                raise ValueError("observation/case misalignment")
            passed: list[str] = []
            for case, obs in zip(cases, observations, strict=True):
                if type(obs) is not dict:
                    raise ValueError("observation not a dict")
                if obs.get("exact_match") is True and obs.get("boundary_found") is True:
                    passed.append(case.id)
            passing[name] = tuple(passed)
        except ValueError:
            excluded.append(name)

    if not passing:
        return {"available": False, "detail": "no_usable_candidates"}
    winner = result.get("selected_candidate")
    if type(winner) is not str or winner not in passing:
        winner = None
    decision = result.get("decision")
    if type(decision) is not str:
        decision = "unknown"

    rates = oracle_rates(passing, fixture_ids, winner)
    oracle_winners = per_fixture_oracle_winners(passing, fixture_ids)
    portfolio = greedy_cover(passing, fixture_ids)
    covered = sorted({fid for name in portfolio for fid in passing[name]})
    return {
        "available": True,
        "n_fixtures": len(cases),
        "n_candidates": len(passing),
        "decision": decision,
        "winner": winner,
        "excluded_candidates": sorted(excluded),
        "oracle": {
            "n_solvable": rates["n_solvable"],
            "solvable_fixture_ids": rates["solvable_fixture_ids"],
            "unsolvable_fixture_ids": rates["unsolvable_fixture_ids"],
            "oracle_pass_rate": rates["oracle_pass_rate"],
            "winner_pass_rate": rates["winner_pass_rate"],
            "gap": rates["gap"],
            "per_fixture_winners": oracle_winners,
        },
        "portfolio": {
            "size": len(portfolio),
            "members": portfolio,
            "per_member_covers": {name: list(passing[name]) for name in portfolio},
            "covered_fixture_ids": covered,
            "uncovered_fixture_ids": rates["unsolvable_fixture_ids"],
            "greedy_note": "greedy set-cover approximation; not guaranteed optimal",
        },
    }


def batch_stage(
    backend: Backend,
    prompts: dict[str, tuple[int, ...]],
    *,
    max_new_tokens: int = 64,
    seed: int = 0,
) -> dict[str, Any]:
    """Audit batch invariance: solo vs batched greedy decoding.

    For every prompt (``{prompt_id: input_ids}``) the backend's plain
    :meth:`generate` runs in isolation (solo) and its optional
    ``generate_batch`` path runs the solo-evaluable prompts together
    (batched); the continuation token-id sequences are compared
    token-for-token with :func:`batchinv.compare_solo_vs_batched`.
    Greedy, exact.

    Fail-closed contract (see :mod:`batchinv`): a backend with no
    ``generate_batch`` attribute records ``{"available": False,
    "detail": "batch_unsupported"}`` -- the audit never synthesizes a
    batch out of sequential calls. A batch path that raises records
    ``batch_failed``; one that violates the arity/value contract
    records ``batch_contract_failed``. A prompt whose solo generation
    fails is marked with a per-prompt error and excluded from the
    divergence rate, never silently dropped and never a phantom
    divergence.

    Cycle 192: only the solo-evaluable prompts are fed to the batch
    path -- a solo-failed prompt's row could never be compared, and on
    the real backend one over-context prompt would otherwise make the
    whole batch raise and collapse an otherwise measurable audit to
    ``batch_failed``. When no prompt is evaluable the batch call is
    skipped entirely and the stage returns the available payload with
    verdict ``"unavailable"``. A batch path raising
    ``ValueError("batch_size_exceeded")`` records that detail instead
    of ``batch_failed``: a labelled size refusal is a different
    operator signal than a mid-run crash.

    Returns the audit payload (``available`` True/False). Raises
    ``ValueError`` on a None backend, malformed prompts, or bad
    ``max_new_tokens``/``seed``.
    """
    from .batchinv import (
        BATCH_CONTRACT_FAILED,
        BATCH_FAILED,
        BATCH_SIZE_EXCEEDED,
        BATCH_UNSUPPORTED,
        audit_verdict,
        compare_solo_vs_batched,
        validate_prompt_ids,
    )

    if backend is None:
        raise ValueError("backend must not be None")
    clean_prompts = validate_prompt_ids(prompts)
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 4096:
        raise ValueError("max_new_tokens must be an integer between 1 and 4096")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    try:
        generate_batch = getattr(backend, "generate_batch", None)
    except Exception:  # noqa: BLE001 -- a broken probe is a failed batch path
        return {"available": False, "detail": BATCH_FAILED}
    if not callable(generate_batch):
        return {"available": False, "detail": BATCH_UNSUPPORTED}

    ordered_ids = sorted(clean_prompts)
    solo: dict[str, tuple[int, ...]] = {}
    per_prompt: dict[str, dict[str, Any]] = {}
    for pid in ordered_ids:
        try:
            generation = backend.generate(
                clean_prompts[pid],
                max_new_tokens=max_new_tokens,
                seed=seed,
                cancel=Event(),
            )
        except Exception:  # noqa: BLE001 -- one prompt degrades, not the stage
            generation = None
        if (
            generation is None
            or generation.reason != "limit"
            or not _valid_token_ids(generation.token_ids, nonempty=True)
        ):
            per_prompt[pid] = {
                "diverged": None,
                "first_divergence": None,
                "solo_len": None,
                "batched_len": None,
                "error": "solo_failed",
            }
            continue
        solo[pid] = generation.token_ids

    evaluable = {pid: solo[pid] for pid in ordered_ids if pid in solo}
    n_diverged = 0
    if evaluable:
        try:
            batched_raw = generate_batch(
                tuple(clean_prompts[pid] for pid in evaluable),
                max_new_tokens=max_new_tokens,
                seed=seed,
                cancel=Event(),
            )
        except ValueError as exc:
            # The size refusal is a labelled, expected outcome, not a
            # failure: keep its detail distinct from batch_failed.
            if str(exc) == BATCH_SIZE_EXCEEDED:
                return {"available": False, "detail": BATCH_SIZE_EXCEEDED}
            return {"available": False, "detail": BATCH_FAILED}
        except Exception:  # noqa: BLE001 -- the batch path failed; fail closed
            return {"available": False, "detail": BATCH_FAILED}
        if (
            type(batched_raw) is not tuple
            or len(batched_raw) != len(evaluable)
            or any(not _valid_token_ids(out, nonempty=True) for out in batched_raw)
        ):
            # A malformed batch response poisons the whole comparison: it
            # cannot be attributed per prompt, so the stage fails closed.
            return {"available": False, "detail": BATCH_CONTRACT_FAILED}
        batched = dict(zip(evaluable, batched_raw, strict=True))

        comparison = compare_solo_vs_batched(evaluable, batched)
        per_prompt_cmp = cast("dict[str, dict[str, int | bool | None]]", comparison["per_prompt"])
        for pid, entry in per_prompt_cmp.items():
            per_prompt[pid] = {**entry, "error": None}
        n_diverged = cast(int, comparison["n_diverged"])
    n_evaluable = len(evaluable)
    return {
        "available": True,
        "n_prompts": len(ordered_ids),
        "n_evaluable": n_evaluable,
        "n_failed": len(ordered_ids) - n_evaluable,
        "n_diverged": n_diverged,
        "divergence_rate": (n_diverged / n_evaluable) if n_evaluable else None,
        "verdict": audit_verdict(n_evaluable, n_diverged),
        "per_prompt": per_prompt,
        "max_new_tokens": max_new_tokens,
        "seed": seed,
    }


def stop_stage(
    result: dict[str, Any],
    cases: tuple[Any, ...],
    *,
    boundary: tuple[int, ...],
    boundary_map: dict[str, tuple[int, ...]] | None = None,
    eos_id: int | None = None,
    backend: Any = None,
    token_bytes: Any = None,
) -> dict[str, Any]:
    """Audit stop-cause termination distributions (pure post-hoc).

    Answers "how would each generation have terminated under a
    stop-aware engine?": for every search candidate and every search
    fixture, the recorded generation is classified post-hoc with
    :mod:`stopcause` into ``eos_id`` (the family's end-of-turn token
    id was emitted), ``stop_string`` (the HF stop-string emulation on
    the candidate's boundary text fired), ``max_length`` (the cap was
    reached with no mechanism firing), or ``stop_unknown``
    (fail-closed: neither the EOT id nor a stop text was available).

    Cause precedence follows the vLLM port notes in ``backends.py``:
    the earliest-firing mechanism wins, with a same-step tie going to
    ``stop_string``. ``over_generated`` counts the tokens the lab's
    non-stopping decoder emitted past the first EOT token -- output no
    real engine would emit. No new generations are run: the audit
    reuses the stored search observations, so it is exact and free.

    Returns ``{"available": True, "per_candidate": ..., "overall":
    ...}``. Never raises on malformed entries: returns
    ``{"available": False}`` with a ``detail`` code. Raises
    ``ValueError`` on a None backend or invalid ``cases`` /
    ``boundary`` / ``eos_id`` (caller-contract violations, not data).
    """
    from .stopcause import (
        aggregate_cause_records,
        classify_cause,
        first_position,
        over_generated_tokens,
    )

    if backend is None:
        raise ValueError("backend must not be None")
    if type(result) is not dict:
        return {"available": False, "detail": "result_not_a_dict"}
    if type(cases) is not tuple or not cases or any(type(c) is not Fixture for c in cases):
        raise ValueError("cases must be a nonempty tuple of Fixture")
    if not _valid_token_ids(boundary, nonempty=True):
        raise ValueError("boundary must be a nonempty tuple of token IDs")
    if eos_id is not None and (type(eos_id) is not int or eos_id < 0):
        raise ValueError("eos_id must be a nonnegative integer or None")
    if boundary_map is not None and (
        type(boundary_map) is not dict or not all(type(k) is str for k in boundary_map)
    ):
        raise ValueError("boundary_map must be a dict of str to id tuples or None")
    entries = result.get("search")
    if type(entries) is not list or not entries:
        return {"available": False, "detail": "no_search_entries"}
    raw_cap = result.get("max_new_tokens")
    max_new_tokens = raw_cap if type(raw_cap) is int and 1 <= raw_cap <= 4096 else 64

    per_candidate: dict[str, dict[str, Any]] = {}
    excluded: list[str] = []
    all_records: list[dict[str, object]] = []
    for entry in entries:
        if type(entry) is not dict or type(entry.get("format")) is not dict:
            continue
        name = entry["format"].get("name")
        if type(name) is not str or not name:
            continue
        observations = entry.get("observations")
        if type(observations) is not list or len(observations) != len(cases):
            excluded.append(name)
            continue
        cand_boundary = boundary
        raw_bids = entry.get("boundary_token_ids")
        if type(raw_bids) is list and raw_bids and all(type(t) is int and t >= 0 for t in raw_bids):
            cand_boundary = tuple(raw_bids)
        elif boundary_map and type(boundary_map.get(name)) is tuple:
            cand_boundary = boundary_map[name]
        try:
            boundary_text: Any = backend.decode(cand_boundary)
        except Exception:  # noqa: BLE001 -- string cells degrade, never fail
            boundary_text = None
        stop_known = type(boundary_text) is str and bool(boundary_text)
        eos_known = eos_id is not None
        records: list[dict[str, object]] = []
        evaluable_records: list[dict[str, object]] = []
        for fixture, obs in zip(cases, observations, strict=True):
            record: dict[str, object] = {
                "fixture": fixture.id if type(fixture.id) is str else None,
                "evaluable": False,
                "cause": None,
                "eos_first": None,
                "stop_first": None,
                "over_generated": 0,
                "cap_reached": False,
            }
            if (
                type(obs) is dict
                and obs.get("fixture") == fixture.id
                and obs.get("status") == "completed"
            ):
                output_ids = tuple(obs.get("output_ids") or ())
                input_ids = tuple(obs.get("input_ids") or ())
                if _valid_token_ids(output_ids, nonempty=True) and _valid_token_ids(input_ids):
                    eos_first = first_position(output_ids, eos_id) if eos_id is not None else None
                    stop_first: int | None = None
                    if stop_known:
                        outcome = emulate_hf_stop_string(
                            backend,
                            input_ids,
                            output_ids,
                            boundary_text,
                            token_bytes=token_bytes,
                        )
                        if outcome.available and outcome.fired:
                            stop_first = len(outcome.truncated_ids) - 1
                    cause = classify_cause(
                        output_len=len(output_ids),
                        max_new_tokens=max_new_tokens,
                        eos_first=eos_first,
                        stop_first=stop_first,
                        eos_known=eos_known,
                        stop_known=stop_known,
                    )
                    bleed = over_generated_tokens(len(output_ids), eos_first)
                    record.update(
                        {
                            "evaluable": True,
                            "cause": cause,
                            "eos_first": eos_first,
                            "stop_first": stop_first,
                            "over_generated": bleed,
                            "cap_reached": len(output_ids) >= max_new_tokens,
                        }
                    )
                    evaluable_records.append({"cause": cause, "over_generated": bleed})
            records.append(record)
        aggregates = aggregate_cause_records(evaluable_records)
        per_candidate[name] = {
            "n_observations": len(records),
            "n_evaluable": aggregates["n"],
            "eos_known": eos_known,
            "stop_known": stop_known,
            **aggregates,
            "records": records,
        }
        all_records.extend(evaluable_records)
    if not per_candidate:
        return {"available": False, "detail": "no_evaluable_candidates"}
    return {
        "available": True,
        "max_new_tokens": max_new_tokens,
        "eos_known": eos_id is not None,
        "per_candidate": per_candidate,
        "excluded_candidates": sorted(excluded),
        "overall": aggregate_cause_records(all_records),
    }


def native_search_input_ids(
    search_entries: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    candidates: tuple[str, ...],
) -> dict[str, dict[str, list[int]] | None]:
    """Per-candidate per-fixture input id lists from the search section.

    Pure post-hoc helper: for each candidate, ``{fixture_id: [ids]}``
    with the exact token ids the native run sent, or None when the
    candidate has no usable observations. Never raises on malformed
    entries: they degrade to None.
    """
    by_name: dict[str, dict[str, Any]] = {}
    entries = list(search_entries) if type(search_entries) in (list, tuple) else []
    for entry in entries:
        if type(entry) is not dict:
            continue
        fmt = entry.get("format")
        name = fmt.get("name") if type(fmt) is dict else None
        if type(name) is str and name:
            by_name[name] = entry
    result: dict[str, dict[str, list[int]] | None] = {}
    for cand in candidates:
        cand_entry = by_name.get(cand)
        observations = cand_entry.get("observations") if type(cand_entry) is dict else None
        if type(observations) is not list or not observations:
            result[cand] = None
            continue
        per_fixture: dict[str, list[int]] = {}
        usable = True
        for obs in observations:
            if type(obs) is not dict:
                usable = False
                break
            fid = obs.get("fixture")
            if type(fid) is not str or not fid:
                usable = False
                break
            # In-memory results carry tuples (asdict of Observation);
            # results.json carries lists. Accept both, store lists.
            raw_ids = obs.get("input_ids")
            if type(raw_ids) is list:
                id_items: list[object] = list(raw_ids)
            elif type(raw_ids) is tuple:
                id_items = list(raw_ids)
            else:
                usable = False
                break
            checked: list[int] = []
            for token in id_items:
                if type(token) is not int or token < 0:
                    usable = False
                    break
                checked.append(token)
            if not usable:
                break
            per_fixture[fid] = checked
        result[cand] = per_fixture if usable else None
    return result


def ws_stage(
    backend: Backend,
    formats: tuple[Format, ...],
    cases: tuple[Fixture, ...],
    *,
    mutations: tuple[str, ...] = MUTATIONS,
    boundary: tuple[int, ...],
    boundary_map: dict[str, tuple[int, ...]] | None = None,
    max_new_tokens: int = 64,
    seed: int = 0,
) -> dict[str, Any]:
    """Re-evaluate every candidate under whitespace mutations of the render.

    For each whitespace mutation (see :mod:`wscanonical`), every search
    candidate is re-evaluated with ``core.evaluate()`` on the search
    fixtures through a :class:`WsScanBackend` that applies the mutation
    to the rendered prompt text at encode time. Greedy, exact. The
    mutation is text-level (applied *before* encoding), so the audited
    id sequence is exactly what the tokenizer produces for the mutated
    string.

    Each candidate x mutation cell records per-fixture verdicts
    (exact answer and explicit boundary) and the variant input id
    lists; a cell degrades to None on any backend/evaluation failure --
    fail-closed, never a phantom "robust". ``pass`` is the conservative
    per-cell verdict: every fixture exact-matched with its boundary
    found (mirrors the position stage's all-pass rule), so
    :func:`ws_canonicalization` can compare verdicts against the native
    run.

    Returns ``{"mutations": [...], "per_mutation": {mutation:
    {"per_candidate": {name: {"n", "n_completed", "pass", "verdicts",
    "input_ids"} | None}}}}``. Raises ``ValueError`` on caller-contract
    violations (None backend, bad formats/cases/mutations/boundary),
    never on backend data.
    """
    if backend is None:
        raise ValueError("backend must not be None")
    if type(formats) is not tuple or not formats:
        raise ValueError("formats must be a nonempty tuple")
    if type(cases) is not tuple or not cases or any(type(case) is not Fixture for case in cases):
        raise ValueError("cases must be a nonempty tuple of Fixture")
    mutation_names = validate_mutations(mutations)
    if not _valid_token_ids(boundary, nonempty=True):
        raise ValueError("Boundary must be a nonempty tuple of token IDs")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 4096:
        raise ValueError("max_new_tokens must be an integer between 1 and 4096")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    boundaries = _resolve_boundaries(formats, boundary, boundary_map)
    per_mutation: dict[str, dict[str, Any]] = {}
    for mutation in mutation_names:
        per_candidate: dict[str, dict[str, Any] | None] = {}
        for fmt in formats:
            try:
                wrapped = WsScanBackend(backend, mutation)
                report = evaluate(
                    wrapped,
                    fmt,
                    cases,
                    boundary=boundaries[fmt.name],
                    max_new_tokens=max_new_tokens,
                    seed=seed,
                )
            except Exception:  # noqa: BLE001 -- one cell degrades, not the stage
                per_candidate[fmt.name] = None
                continue
            if not report.aggregate_eligible:
                per_candidate[fmt.name] = None
                continue
            verdicts: dict[str, bool] = {}
            input_ids: dict[str, list[int]] = {}
            n_completed = 0
            for obs in report.observations:
                input_ids[obs.fixture] = list(obs.input_ids)
                if obs.status == "completed":
                    n_completed += 1
                    verdicts[obs.fixture] = bool(obs.exact_match and obs.boundary_found)
            per_candidate[fmt.name] = {
                "n": len(cases),
                "n_completed": n_completed,
                "pass": n_completed == len(cases)
                and all(verdicts.get(case.id) is True for case in cases),
                "verdicts": verdicts,
                "input_ids": input_ids,
            }
        per_mutation[mutation] = {"per_candidate": per_candidate}
    return {"mutations": list(mutation_names), "per_mutation": per_mutation}


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


def structured_stage(
    backend: Backend,
    formats: tuple[Format, ...],
    structured_fixtures: tuple[Fixture, ...],
    *,
    schemas: dict[str, dict[str, Any]],
    boundary: tuple[int, ...],
    boundary_map: dict[str, tuple[int, ...]] | None = None,
    max_new_tokens: int = 64,
    seed: int = 0,
) -> dict[str, Any]:
    """Re-evaluate every candidate on synthetic structured-output fixtures.

    For every search candidate, the synthetic JSON fixtures (see
    :mod:`structured`) are evaluated with ``core.evaluate()`` unchanged;
    each completed observation's decoded ``answer`` is then classified
    against the fixture's schema by :func:`structured.classify_response`
    into ``adherent`` / ``parse_failed`` / ``schema_failed`` /
    ``not_text`` and aggregated by :func:`structured.candidate_adherence`.
    Greedy, exact. No constrained decoding: the stage measures what the
    template alone elicits.

    Any backend construction or evaluation failure records None for that
    candidate cell -- fail-closed, never silently skipped. ``schemas``
    maps every fixture id to its schema definition (see
    :func:`structured.schema_map`).

    Returns ``{"n_fixtures": ..., "per_candidate": {name:
    {"n", "n_evaluable", "parse_ok", "schema_ok", "adherent",
    "parse_rate", "schema_rate", "adherent_rate"} | None}}``. Raises
    ``ValueError`` on caller-contract violations (None backend, bad
    formats/fixtures/schemas/boundary), never on backend data.
    """
    if backend is None:
        raise ValueError("backend must not be None")
    if type(formats) is not tuple or not formats:
        raise ValueError("formats must be a nonempty tuple")
    if type(structured_fixtures) is not tuple or not structured_fixtures:
        raise ValueError("structured_fixtures must be a nonempty tuple")
    if any(type(case) is not Fixture for case in structured_fixtures):
        raise ValueError("structured_fixtures must contain Fixture values")
    if type(schemas) is not dict or not all(
        type(key) is str and type(value) is dict for key, value in schemas.items()
    ):
        raise ValueError("schemas must be a dict of fixture id to schema dict")
    for fixture_id, schema in schemas.items():
        try:
            validate_schema_def(schema)
        except ValueError as exc:
            raise ValueError(
                f"schemas[{fixture_id!r}] is not a valid schema definition: {exc}"
            ) from exc
    for case in structured_fixtures:
        if case.id not in schemas:
            raise ValueError(f"no schema registered for fixture {case.id!r}")
    if not _valid_token_ids(boundary, nonempty=True):
        raise ValueError("Boundary must be a nonempty tuple of token IDs")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 4096:
        raise ValueError("max_new_tokens must be an integer between 1 and 4096")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    boundaries = _resolve_boundaries(formats, boundary, boundary_map)
    per_candidate: dict[str, dict[str, Any] | None] = {}
    for fmt in formats:
        try:
            report = evaluate(
                backend,
                fmt,
                structured_fixtures,
                boundary=boundaries[fmt.name],
                max_new_tokens=max_new_tokens,
                seed=seed,
            )
        except Exception:  # noqa: BLE001 -- one cell degrades, not the stage
            per_candidate[fmt.name] = None
            continue
        if not report.aggregate_eligible:
            per_candidate[fmt.name] = None
            continue
        outcomes: list[str] = []
        for obs in report.observations:
            if obs.status == "completed":
                outcomes.append(classify_response(obs.answer, schemas[obs.fixture])["outcome"])
        per_candidate[fmt.name] = candidate_adherence(outcomes)
    return {
        "n_fixtures": len(structured_fixtures),
        "per_candidate": per_candidate,
    }


def _structured_cell_valid(entry: Any) -> bool:
    """Structural check for one candidate cell of a structured stage."""
    if entry is None:
        return True
    if type(entry) is not dict:
        return False
    for key in (
        "n",
        "n_evaluable",
        "parse_ok",
        "schema_ok",
        "adherent",
    ):
        value = entry.get(key)
        if type(value) is not int or value < 0:
            return False
    if entry["n_evaluable"] > entry["n"]:
        return False
    for key in ("parse_rate", "schema_rate", "adherent_rate"):
        rate = entry.get(key)
        if rate is not None and not (type(rate) in (int, float) and 0.0 <= rate <= 1.0):
            return False
    if entry["n_evaluable"] != 0:
        return True
    return not any(
        entry.get(key) is not None for key in ("parse_rate", "schema_rate", "adherent_rate")
    )


def structured_adherence(
    candidates: tuple[str, ...],
    stage: dict[str, Any],
) -> dict[str, Any]:
    """Analyze structured-output adherence across candidates (pure post-hoc).

    Answers "does the template change JSON adherence?": reports the
    per-candidate adherence rates from the stage, the weakest
    candidates (minimum adherent rate), and the max-minus-min
    ``adherence_gap`` -- the pre-registered falsifiable claim (see
    :mod:`structured`) is that the gap is measurably positive rather
    than ~0; ``uniform`` reports the null hypothesis (all evaluable
    candidates identical), never asserts it.

    Never raises on malformed input: returns ``{"available": False}``
    with a ``detail`` code.
    """
    if (
        not isinstance(candidates, tuple)
        or not candidates
        or not all(type(c) is str and c.strip() for c in candidates)
        or len(set(candidates)) != len(candidates)
    ):
        return {"available": False, "detail": "invalid_candidates"}
    if type(stage) is not dict:
        return {"available": False, "detail": "stage_invalid"}
    n_fixtures = stage.get("n_fixtures")
    if type(n_fixtures) is not int or n_fixtures < 0:
        return {"available": False, "detail": "stage_invalid"}
    per_candidate = stage.get("per_candidate")
    if type(per_candidate) is not dict or not all(type(k) is str for k in per_candidate):
        return {"available": False, "detail": "stage_invalid"}
    for entry in per_candidate.values():
        if not _structured_cell_valid(entry):
            return {"available": False, "detail": "stage_invalid"}
    evaluable: dict[str, float] = {}
    excluded: list[str] = []
    for cand in candidates:
        entry = per_candidate.get(cand)
        if entry is None:
            excluded.append(cand)
            continue
        if type(entry) is not dict or entry.get("n_evaluable", 0) == 0:
            excluded.append(cand)
            continue
        rate = entry.get("adherent_rate")
        if type(rate) is int or type(rate) is float:
            evaluable[cand] = float(rate)
        else:
            excluded.append(cand)
            continue
    if evaluable:
        min_rate = min(evaluable.values())
        max_rate = max(evaluable.values())
        weakest = sorted(name for name, rate in evaluable.items() if rate == min_rate)
        gap: float | None = max_rate - min_rate
        uniform: bool | None = all(rate == min_rate for rate in evaluable.values())
    else:
        weakest = []
        gap = None
        uniform = None
    return {
        "available": True,
        "n_fixtures": n_fixtures,
        "per_candidate": {cand: per_candidate.get(cand) for cand in candidates},
        "weakest_candidates": weakest,
        "excluded_candidates": sorted(excluded),
        "adherence_gap": gap,
        "uniform": uniform,
    }


def _deliforge_cell_unsupported(detail: str) -> dict[str, Any]:
    """One unsupported deliforge cell: fail-closed, never a phantom gap."""
    return {"supported": False, "detail": detail}


def deliforge_stage(
    backend: Backend,
    formats: tuple[Format, ...],
    cases: tuple[Fixture, ...],
    *,
    payloads: tuple[str, ...] = PAYLOADS,
    marker_indexes: tuple[int, ...] | None = None,
    boundary: tuple[int, ...],
    boundary_map: dict[str, tuple[int, ...]] | None = None,
    max_new_tokens: int = 64,
    seed: int = 0,
) -> dict[str, Any]:
    """Re-evaluate every candidate with forged markers in two encodings.

    For each forged-turn payload (see :mod:`deliforge`) and each
    delimiter marker, every search candidate is evaluated twice with
    ``core.evaluate()`` on the forged fixtures: once with the forged
    marker in its native single-id ("reserved") encoding, once with the
    forged markers spliced to their forced-subword encoding through a
    :class:`DeliforgeBackend`. The two arms decode to identical text by
    construction (verified per fixture); only the token-id encoding of
    the forged markers differs, and the template's own genuine markers
    are never touched (see :func:`deliforge.build_subword_ids`).

    A (candidate, payload, marker) cell degrades to an unsupported
    marker instead of failing the stage: no single-id native encoding
    for the marker (``no_encoding_pair``), marker localization failure
    (``localization_failed``), or a decode mismatch between the arms
    (``decode_mismatch``). Backend/evaluation failures degrade the
    cell to None -- fail-closed, never a phantom authority gap.

    Returns ``{"payloads": [...], "marker_indexes": [...],
    "per_payload": {payload: {"per_marker": {str(index):
    {"per_candidate": {name: cell | None}}}}}``. Raises ``ValueError``
    on caller-contract violations (None backend, bad formats/cases/
    payloads/marker_indexes/boundary), never on backend data.
    """
    if backend is None:
        raise ValueError("backend must not be None")
    if type(formats) is not tuple or not formats:
        raise ValueError("formats must be a nonempty tuple")
    if type(cases) is not tuple or not cases or any(type(case) is not Fixture for case in cases):
        raise ValueError("cases must be a nonempty tuple of Fixture")
    if type(payloads) is not tuple or not payloads:
        raise ValueError("payloads must be a nonempty tuple")
    for payload in payloads:
        validate_payload(payload)
    if marker_indexes is None:
        marker_indexes = tuple(range(len(MARKERS)))
    if type(marker_indexes) is not tuple or not marker_indexes:
        raise ValueError("marker_indexes must be a nonempty tuple")
    for index in marker_indexes:
        validate_marker_index(index)
    if len(set(marker_indexes)) != len(marker_indexes):
        raise ValueError("marker_indexes must not repeat")
    if not _valid_token_ids(boundary, nonempty=True):
        raise ValueError("Boundary must be a nonempty tuple of token IDs")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 4096:
        raise ValueError("max_new_tokens must be an integer between 1 and 4096")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    boundaries = _resolve_boundaries(formats, boundary, boundary_map)
    per_payload: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        per_marker: dict[str, dict[str, Any]] = {}
        for marker_index in marker_indexes:
            marker = MARKERS[marker_index]
            try:
                pair = find_encoding_pair(backend.encode, backend.decode, marker)
            except Exception:  # noqa: BLE001 -- backend data, not a contract violation
                pair = None
            forged = build_forged_fixtures(cases, payload=payload, marker_index=marker_index)
            per_candidate: dict[str, dict[str, Any] | None] = {}
            for fmt in formats:
                try:
                    per_candidate[fmt.name] = _deliforge_candidate_cell(
                        backend,
                        fmt,
                        cases,
                        forged,
                        marker=marker,
                        pair=pair,
                        boundary=boundaries[fmt.name],
                        max_new_tokens=max_new_tokens,
                        seed=seed,
                    )
                except Exception:  # noqa: BLE001 -- one cell degrades, not the stage
                    per_candidate[fmt.name] = None
            per_marker[str(marker_index)] = {"per_candidate": per_candidate}
        per_payload[payload] = {"per_marker": per_marker}
    return {
        "payloads": list(payloads),
        "marker_indexes": list(marker_indexes),
        "per_payload": per_payload,
    }


def _deliforge_candidate_cell(
    backend: Backend,
    fmt: Format,
    cases: tuple[Fixture, ...],
    forged: tuple[Fixture, ...],
    *,
    marker: str,
    pair: dict[str, tuple[int, ...]] | None,
    boundary: tuple[int, ...],
    max_new_tokens: int,
    seed: int,
) -> dict[str, Any]:
    """Run both encoding arms for one candidate x payload x marker.

    Returns the supported cell ``{"supported": True, "n": ...,
    "verdicts": {fixture_id: {"reserved": bool | None, "subword":
    bool | None}}, ...}`` or an unsupported marker dict. The subword
    arm's ids are built by text-level construction (see
    :func:`deliforge.build_subword_ids`): the exact inserted string is
    known from the fixture construction, so its single occurrence in
    the rendered prompt is located at the text level and only the
    forged marker occurrences are re-encoded -- the template's genuine
    markers keep their reserved ids (verified by id count). Raises on
    backend/evaluation failure (the caller degrades the cell to None).
    """
    if pair is None:
        return _deliforge_cell_unsupported(DETAIL_NO_ENCODING_PAIR)
    marker_id = pair["reserved"][0]
    subword_ids = pair["subword"]
    reserved_report = evaluate(
        backend,
        fmt,
        forged,
        boundary=boundary,
        max_new_tokens=max_new_tokens,
        seed=seed,
    )
    if not reserved_report.aggregate_eligible:
        return _deliforge_cell_unsupported(DETAIL_CELL_FAILED)
    plan: dict[str, tuple[int, ...]] = {}
    verdicts: dict[str, dict[str, bool | None]] = {}
    forged_texts: dict[str, str] = {}
    for obs in reserved_report.observations:
        original_id = _original_fixture_id(obs.fixture)
        text = fmt.render(_forged_messages_for(forged, obs.fixture))
        forged_texts[obs.fixture] = text
        if obs.status != "completed":
            verdicts[original_id] = {"reserved": None, "subword": None}
            continue
        spliced = _deliforge_spliced_ids(
            backend,
            benign_messages=_benign_messages_for(cases, original_id),
            forged_messages=_forged_messages_for(forged, obs.fixture),
            text=text,
            marker=marker,
            marker_id=marker_id,
            subword_ids=subword_ids,
        )
        if spliced is None:
            return _deliforge_cell_unsupported(DETAIL_LOCALIZATION_FAILED)
        if not decode_equal(backend.decode, tuple(obs.input_ids), spliced):
            return _deliforge_cell_unsupported(DETAIL_DECODE_MISMATCH)
        plan[text] = spliced
        verdicts[original_id] = {
            "reserved": bool(obs.exact_match and obs.boundary_found),
            "subword": None,
        }
    subword_report = evaluate(
        DeliforgeBackend(backend, plan),
        fmt,
        forged,
        boundary=boundary,
        max_new_tokens=max_new_tokens,
        seed=seed,
    )
    if not subword_report.aggregate_eligible:
        return _deliforge_cell_unsupported(DETAIL_CELL_FAILED)
    for obs in subword_report.observations:
        original_id = _original_fixture_id(obs.fixture)
        entry = verdicts.get(original_id)
        if entry is None:
            return _deliforge_cell_unsupported(DETAIL_CELL_FAILED)
        expected_ids = plan.get(forged_texts[obs.fixture])
        if expected_ids is None or tuple(obs.input_ids) != expected_ids:
            return _deliforge_cell_unsupported(DETAIL_DECODE_MISMATCH)
        entry["subword"] = (
            bool(obs.exact_match and obs.boundary_found) if obs.status == "completed" else None
        )
    return {
        "supported": True,
        "n": len(cases),
        "marker": marker,
        "reserved_id": marker_id,
        "subword_len": len(subword_ids),
        "verdicts": verdicts,
    }


def _deliforge_spliced_ids(
    backend: Backend,
    *,
    benign_messages: tuple[Message, ...],
    forged_messages: tuple[Message, ...],
    text: str,
    marker: str,
    marker_id: int,
    subword_ids: tuple[int, ...],
) -> tuple[int, ...] | None:
    """Build the subword-arm ids for one forged prompt (text-level).

    The forged payload was appended to one message's content by
    :func:`deliforge.build_forged_fixtures`; the exact inserted string
    is recovered from the message pair, located exactly once in the
    rendered prompt, and only the marker occurrences inside that span
    are re-encoded via :func:`deliforge.build_subword_ids`. Returns
    None (fail-closed) when the construction cannot be verified.
    """
    try:
        target = forged_target_index(forged_messages)
        benign_target = forged_target_index(benign_messages)
    except ValueError:
        return None
    if target != benign_target:
        return None
    benign_content = benign_messages[target].content
    forged_content = forged_messages[target].content
    if type(benign_content) is not str or type(forged_content) is not str:
        return None
    if not forged_content.startswith(benign_content):
        return None
    inserted = forged_content[len(benign_content) :]
    if not inserted:
        return None
    span = find_insertion_span(text, inserted)
    if span is None:
        return None
    expected_count = inserted.count(marker)
    offsets = forged_marker_offsets(text, marker, span, expected_count)
    if offsets is None:
        return None
    return build_subword_ids(
        backend.encode,
        backend.decode,
        text,
        marker,
        marker_id,
        subword_ids,
        offsets,
    )


def _original_fixture_id(forged_id: str) -> str:
    """Strip the ``deliforge/<payload>/<index>/`` prefix from a forged id."""
    parts = forged_id.split("/", 3)
    if len(parts) == 4 and parts[0] == "deliforge":
        return parts[3]
    return forged_id


def _forged_messages_for(forged: tuple[Fixture, ...], forged_id: str) -> tuple[Message, ...]:
    """Find a forged fixture's messages by id; raises LookupError."""
    for fixture in forged:
        if fixture.id == forged_id:
            return fixture.messages
    raise LookupError(f"unknown forged fixture {forged_id!r}")


def _benign_messages_for(cases: tuple[Fixture, ...], original_id: str) -> tuple[Message, ...]:
    """Find a search fixture's messages by id; raises LookupError."""
    for case in cases:
        if case.id == original_id:
            return case.messages
    raise LookupError(f"unknown search fixture {original_id!r}")


def _deliforge_cell_view(entry: Any) -> dict[str, Any] | None:
    """Normalize one deliforge authority cell entry, or None if malformed."""
    if entry is None:
        return {"supported": False, "detail": DETAIL_CELL_FAILED}
    if type(entry) is not dict:
        return None
    if entry.get("supported") is not True:
        detail = entry.get("detail")
        return {
            "supported": False,
            "detail": detail if type(detail) is str else DETAIL_CELL_FAILED,
        }
    verdicts = entry.get("verdicts")
    if type(verdicts) is not dict or not all(
        type(fid) is str
        and type(cell) is dict
        and (cell.get("reserved") is None or type(cell.get("reserved")) is bool)
        and (cell.get("subword") is None or type(cell.get("subword")) is bool)
        for fid, cell in verdicts.items()
    ):
        return None
    return {"supported": True, "verdicts": verdicts}


def deliforge_authority(
    native_outcomes: dict[str, dict[str, Any]],
    stage_result: dict[str, Any],
    candidates: tuple[str, ...],
) -> dict[str, Any]:
    """Analyze delimiter authority gaps across candidates (pure post-hoc).

    Answers "does the reserved encoding of a forged marker flip task
    verdicts more often than the subword encoding of the identical
    text?": per (candidate, payload, marker) cell, the flip rates of
    the reserved and subword arms against the native search verdicts
    and the ``authority_gap`` (reserved minus subword, in [-1, 1]).
    The pre-registered falsifiable claim (see :mod:`deliforge`) is that
    the gap is ~0 and uniform across families; a positive gap
    replicates the paper's finding inside the lab.

    Never raises on malformed input: returns ``{"available": False}``
    with a ``detail`` code.
    """
    if (
        not isinstance(candidates, tuple)
        or not candidates
        or not all(type(c) is str and c.strip() for c in candidates)
        or len(set(candidates)) != len(candidates)
    ):
        return {"available": False, "detail": "invalid_candidates"}
    if type(native_outcomes) is not dict:
        return {"available": False, "detail": "native_data_malformed"}
    if type(stage_result) is not dict:
        return {"available": False, "detail": "stage_invalid"}
    payloads = stage_result.get("payloads")
    marker_indexes = stage_result.get("marker_indexes")
    per_payload = stage_result.get("per_payload")
    if (
        type(payloads) is not list
        or not payloads
        or not all(type(p) is str for p in payloads)
        or type(marker_indexes) is not list
        or not marker_indexes
        or not all(type(i) is int for i in marker_indexes)
        or type(per_payload) is not dict
    ):
        return {"available": False, "detail": "stage_invalid"}
    per_candidate: dict[str, dict[str, Any]] = {}
    excluded: list[str] = []
    for cand in candidates:
        native = native_outcomes.get(cand)
        native_verdicts = native.get("per_fixture") if type(native) is dict else None
        if type(native_verdicts) is not dict or not all(
            type(v) is bool for v in native_verdicts.values()
        ):
            excluded.append(cand)
            continue
        cells: dict[str, dict[str, Any]] = {}
        malformed = False
        for payload in payloads:
            section = per_payload.get(payload)
            markers = section.get("per_marker") if type(section) is dict else None
            if type(markers) is not dict:
                malformed = True
                break
            for marker_index in marker_indexes:
                block = markers.get(str(marker_index))
                entry = block.get("per_candidate", {}).get(cand) if type(block) is dict else None
                view = _deliforge_cell_view(entry)
                if view is None:
                    malformed = True
                    break
                key = f"{payload}/{marker_index}"
                if not view["supported"]:
                    cells[key] = view
                    continue
                try:
                    reserved = flip_rates(
                        native_verdicts,
                        {fid: cell["reserved"] for fid, cell in view["verdicts"].items()},
                    )
                    subword = flip_rates(
                        native_verdicts,
                        {fid: cell["subword"] for fid, cell in view["verdicts"].items()},
                    )
                    gap = authority_gap(reserved["flip_rate"], subword["flip_rate"])
                except ValueError:
                    malformed = True
                    break
                cells[key] = {
                    "supported": True,
                    "n_evaluable": reserved["n_evaluable"],
                    "n_flips_reserved": reserved["n_flips"],
                    "n_flips_subword": subword["n_flips"],
                    "flip_rate_reserved": reserved["flip_rate"],
                    "flip_rate_subword": subword["flip_rate"],
                    "authority_gap": gap,
                }
            if malformed:
                break
        if malformed:
            return {"available": False, "detail": "stage_invalid"}
        gaps = [
            cell["authority_gap"]
            for cell in cells.values()
            if cell.get("supported") and type(cell.get("authority_gap")) in (int, float)
        ]
        evaluable_cells = sum(
            1
            for cell in cells.values()
            if cell.get("supported")
            and type(cell.get("n_evaluable")) is int
            and cell["n_evaluable"] > 0
        )
        per_candidate[cand] = {
            "cells": cells,
            "n_cells": len(cells),
            "n_evaluable_cells": evaluable_cells,
            "mean_authority_gap": (sum(gaps) / len(gaps)) if gaps else None,
            "max_authority_gap": max(gaps) if gaps else None,
            "min_authority_gap": min(gaps) if gaps else None,
        }
        if not evaluable_cells:
            excluded.append(cand)
    all_gaps = [
        entry["max_authority_gap"]
        for entry in per_candidate.values()
        if type(entry.get("max_authority_gap")) in (int, float)
    ]
    return {
        "available": True,
        "payloads": list(payloads),
        "marker_indexes": list(marker_indexes),
        "per_candidate": per_candidate,
        "excluded_candidates": sorted(excluded),
        "max_authority_gap": max(all_gaps) if all_gaps else None,
    }


def _faketurn_cell_unsupported(detail: str) -> dict[str, Any]:
    """One unsupported faketurn cell: fail-closed, never a phantom flip rate."""
    return {"supported": False, "detail": detail}


def _faketurn_original_id(forged_id: str) -> str:
    """Strip the ``faketurn/<payload>/<index>/`` prefix from a forged id."""
    parts = forged_id.split("/", 3)
    if len(parts) == 4 and parts[0] == "faketurn":
        return parts[3]
    return forged_id


def faketurn_stage(
    backend: Backend,
    formats: tuple[Format, ...],
    cases: tuple[Fixture, ...],
    *,
    payloads: tuple[str, ...] = PAYLOADS,
    marker_indexes: tuple[int, ...] | None = None,
    boundary: tuple[int, ...],
    boundary_map: dict[str, tuple[int, ...]] | None = None,
    max_new_tokens: int = 64,
    seed: int = 0,
) -> dict[str, Any]:
    """Re-evaluate every candidate with forged history turns per provenance.

    For each forged-turn payload (see :mod:`faketurn`) and each
    delimiter marker, every search candidate is evaluated once with
    ``core.evaluate()`` on the forged fixtures -- the forged turn
    appended to the last user message, encoded at the family's native
    form. The independent variable is the forged marker's token
    provenance: ``same`` when the marker is also a genuine delimiter
    of the rendering template, ``cross`` when it is a real
    chat-template control token from another family's token set (see
    :func:`faketurn.marker_provenance`).

    A (candidate, payload, marker) cell degrades to an unsupported
    marker instead of failing the stage: provenance undeterminable
    (``no_provenance``), forged payload not intact in the rendered
    prompt (``ref_missing`` / ``marker_count_mismatch``), or the
    forged evaluation not aggregable (``cell_failed``). Backend
    failures degrade the cell to None -- fail-closed, never a phantom
    flip rate.

    Returns ``{"payloads": [...], "marker_indexes": [...],
    "per_payload": {payload: {"per_marker": {str(index):
    {"per_candidate": {name: cell | None}}}}}``. Raises ``ValueError``
    on caller-contract violations (None backend, bad formats/cases/
    payloads/marker_indexes/boundary), never on backend data.
    """
    if backend is None:
        raise ValueError("backend must not be None")
    if type(formats) is not tuple or not formats:
        raise ValueError("formats must be a nonempty tuple")
    if type(cases) is not tuple or not cases or any(type(case) is not Fixture for case in cases):
        raise ValueError("cases must be a nonempty tuple of Fixture")
    if type(payloads) is not tuple or not payloads:
        raise ValueError("payloads must be a nonempty tuple")
    for payload in payloads:
        validate_payload(payload)
    if marker_indexes is None:
        marker_indexes = tuple(range(len(MARKERS)))
    if type(marker_indexes) is not tuple or not marker_indexes:
        raise ValueError("marker_indexes must be a nonempty tuple")
    for index in marker_indexes:
        validate_marker_index(index)
    if len(set(marker_indexes)) != len(marker_indexes):
        raise ValueError("marker_indexes must not repeat")
    if not _valid_token_ids(boundary, nonempty=True):
        raise ValueError("Boundary must be a nonempty tuple of token IDs")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 4096:
        raise ValueError("max_new_tokens must be an integer between 1 and 4096")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    boundaries = _resolve_boundaries(formats, boundary, boundary_map)
    per_payload: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        per_marker: dict[str, dict[str, Any]] = {}
        for marker_index in marker_indexes:
            marker = MARKERS[marker_index]
            forged = build_faketurn_fixtures(cases, payload=payload, marker_index=marker_index)
            per_candidate: dict[str, dict[str, Any] | None] = {}
            for fmt in formats:
                try:
                    per_candidate[fmt.name] = _faketurn_candidate_cell(
                        backend,
                        fmt,
                        cases,
                        forged,
                        payload=payload,
                        marker_index=marker_index,
                        marker=marker,
                        boundary=boundaries[fmt.name],
                        max_new_tokens=max_new_tokens,
                        seed=seed,
                    )
                except Exception:  # noqa: BLE001 -- one cell degrades, not the stage
                    per_candidate[fmt.name] = None
            per_marker[str(marker_index)] = {"per_candidate": per_candidate}
        per_payload[payload] = {"per_marker": per_marker}
    return {
        "payloads": list(payloads),
        "marker_indexes": list(marker_indexes),
        "per_payload": per_payload,
    }


def _faketurn_candidate_cell(
    backend: Backend,
    fmt: Format,
    cases: tuple[Fixture, ...],
    forged: tuple[Fixture, ...],
    *,
    payload: str,
    marker_index: int,
    marker: str,
    boundary: tuple[int, ...],
    max_new_tokens: int,
    seed: int,
) -> dict[str, Any]:
    """Run the forged arm for one candidate x payload x marker.

    Determines the marker's token provenance from the candidate's
    own benign renders (``same`` iff the marker is also a genuine
    delimiter of this template), evaluates the forged fixtures, and
    verifies each rendered prompt carries its unique ref exactly once
    with the forged marker occurring at least twice. Returns the
    supported cell ``{"supported": True, "n": ..., "marker": ...,
    "token_source": "same"|"cross", "verdicts": {fixture_id:
    bool | None}}`` or an unsupported marker dict. Raises on
    backend/evaluation failure (the caller degrades the cell to None).
    """
    try:
        provenance: str | None = None
        for case in cases:
            hit = marker_provenance(fmt.render(case.messages), marker)
            if hit is None:
                provenance = None
                break
            if hit == "same":
                provenance = "same"
                break
            provenance = "cross"
    except Exception:  # noqa: BLE001 -- render failure degrades the cell
        provenance = None
    if provenance is None:
        return _faketurn_cell_unsupported(DETAIL_NO_PROVENANCE)
    report = evaluate(
        backend,
        fmt,
        forged,
        boundary=boundary,
        max_new_tokens=max_new_tokens,
        seed=seed,
    )
    if not report.aggregate_eligible:
        return _faketurn_cell_unsupported(DETAIL_CELL_FAILED)
    verdicts: dict[str, bool | None] = {}
    for obs in report.observations:
        original_id = _faketurn_original_id(obs.fixture)
        ref = payload_ref(original_id, payload, marker_index)
        if ref is None:
            return _faketurn_cell_unsupported(DETAIL_CELL_FAILED)
        text = fmt.render(_forged_messages_for(forged, obs.fixture))
        if text.count(ref) != 1:
            return _faketurn_cell_unsupported(DETAIL_REF_MISSING)
        if not payload_intact(text, ref, marker):
            return _faketurn_cell_unsupported(DETAIL_MARKER_COUNT_MISMATCH)
        verdicts[original_id] = (
            bool(obs.exact_match and obs.boundary_found) if obs.status == "completed" else None
        )
    return {
        "supported": True,
        "n": len(cases),
        "marker": marker,
        "token_source": provenance,
        "verdicts": verdicts,
    }


def _faketurn_cell_view(entry: Any) -> dict[str, Any] | None:
    """Normalize one faketurn robustness cell entry, or None if malformed."""
    if entry is None:
        return {"supported": False, "detail": DETAIL_CELL_FAILED}
    if type(entry) is not dict:
        return None
    if entry.get("supported") is not True:
        detail = entry.get("detail")
        return {
            "supported": False,
            "detail": detail if type(detail) is str else DETAIL_CELL_FAILED,
        }
    token_source = entry.get("token_source")
    if token_source not in SOURCES:
        return None
    verdicts = entry.get("verdicts")
    if type(verdicts) is not dict or not all(
        type(fid) is str and (v is None or type(v) is bool) for fid, v in verdicts.items()
    ):
        return None
    return {"supported": True, "token_source": token_source, "verdicts": verdicts}


def faketurn_robustness(
    native_outcomes: dict[str, dict[str, Any]],
    stage_result: dict[str, Any],
    candidates: tuple[str, ...],
) -> dict[str, Any]:
    """Analyze fake-turn history-tampering robustness per candidate (pure post-hoc).

    Answers "does a forged prior turn flip task verdicts, and does
    the delimiter provenance matter?": per (candidate, payload,
    marker) cell, the flip rate of the forged arm against the native
    search verdicts and its token provenance (``same``/``cross``);
    per candidate, the pooled same-family and cross-family flip
    rates and the ``transfer_gap`` (cross minus same, in [-1, 1]).
    The pre-registered falsifiable claim (see :mod:`faketurn`) is
    that the gap is ~0 and flip rates do not cluster by token set; a
    systematically positive gap means foreign delimiters carry more
    tampering authority than the family's own.

    Never raises on malformed input: returns ``{"available": False}``
    with a ``detail`` code.
    """
    if (
        not isinstance(candidates, tuple)
        or not candidates
        or not all(type(c) is str and c.strip() for c in candidates)
        or len(set(candidates)) != len(candidates)
    ):
        return {"available": False, "detail": "invalid_candidates"}
    if type(native_outcomes) is not dict:
        return {"available": False, "detail": "native_data_malformed"}
    if type(stage_result) is not dict:
        return {"available": False, "detail": "stage_invalid"}
    payloads = stage_result.get("payloads")
    marker_indexes = stage_result.get("marker_indexes")
    per_payload = stage_result.get("per_payload")
    if (
        type(payloads) is not list
        or not payloads
        or not all(type(p) is str for p in payloads)
        or type(marker_indexes) is not list
        or not marker_indexes
        or not all(type(i) is int for i in marker_indexes)
        or type(per_payload) is not dict
    ):
        return {"available": False, "detail": "stage_invalid"}
    per_candidate: dict[str, dict[str, Any]] = {}
    excluded: list[str] = []
    for cand in candidates:
        native = native_outcomes.get(cand)
        native_verdicts = native.get("per_fixture") if type(native) is dict else None
        if type(native_verdicts) is not dict or not all(
            type(v) is bool for v in native_verdicts.values()
        ):
            excluded.append(cand)
            continue
        cells: dict[str, dict[str, Any]] = {}
        malformed = False
        for payload in payloads:
            section = per_payload.get(payload)
            markers = section.get("per_marker") if type(section) is dict else None
            if type(markers) is not dict:
                malformed = True
                break
            for marker_index in marker_indexes:
                block = markers.get(str(marker_index))
                entry = block.get("per_candidate", {}).get(cand) if type(block) is dict else None
                view = _faketurn_cell_view(entry)
                if view is None:
                    malformed = True
                    break
                key = f"{payload}/{marker_index}"
                if not view["supported"]:
                    cells[key] = view
                    continue
                try:
                    flips = flip_rates(
                        native_verdicts,
                        dict(view["verdicts"].items()),
                    )
                except ValueError:
                    malformed = True
                    break
                cells[key] = {
                    "supported": True,
                    "n_evaluable": flips["n_evaluable"],
                    "n_flips": flips["n_flips"],
                    "flip_rate": flips["flip_rate"],
                    "token_source": view["token_source"],
                }
            if malformed:
                break
        if malformed:
            return {"available": False, "detail": "stage_invalid"}
        pooled: dict[str, dict[str, int]] = {
            "same": {"flips": 0, "evaluable": 0},
            "cross": {"flips": 0, "evaluable": 0},
        }
        rates: list[float] = []
        for cell in cells.values():
            if not cell.get("supported"):
                continue
            source = cell["token_source"]
            pooled[source]["flips"] += cell["n_flips"]
            pooled[source]["evaluable"] += cell["n_evaluable"]
            rate = cell["flip_rate"]
            if type(rate) in (int, float):
                rates.append(float(rate))
        same_rate = (
            pooled["same"]["flips"] / pooled["same"]["evaluable"]
            if pooled["same"]["evaluable"]
            else None
        )
        cross_rate = (
            pooled["cross"]["flips"] / pooled["cross"]["evaluable"]
            if pooled["cross"]["evaluable"]
            else None
        )
        try:
            gap = transfer_gap(cross_rate, same_rate)
        except ValueError:
            return {"available": False, "detail": "stage_invalid"}
        evaluable_cells = sum(
            1
            for cell in cells.values()
            if cell.get("supported")
            and type(cell.get("n_evaluable")) is int
            and cell["n_evaluable"] > 0
        )
        per_candidate[cand] = {
            "cells": cells,
            "n_cells": len(cells),
            "n_evaluable_cells": evaluable_cells,
            "same_flip_rate": same_rate,
            "cross_flip_rate": cross_rate,
            "transfer_gap": gap,
            "mean_flip_rate": (sum(rates) / len(rates)) if rates else None,
            "max_flip_rate": max(rates) if rates else None,
        }
        if not evaluable_cells:
            excluded.append(cand)
    max_gap = [
        entry["transfer_gap"]
        for entry in per_candidate.values()
        if type(entry.get("transfer_gap")) in (int, float)
    ]
    max_rate = [
        entry["max_flip_rate"]
        for entry in per_candidate.values()
        if type(entry.get("max_flip_rate")) in (int, float)
    ]
    return {
        "available": True,
        "payloads": list(payloads),
        "marker_indexes": list(marker_indexes),
        "per_candidate": per_candidate,
        "excluded_candidates": sorted(excluded),
        "max_transfer_gap": max(max_gap) if max_gap else None,
        "max_flip_rate": max(max_rate) if max_rate else None,
    }


def toolcall_stage(
    backend: Backend,
    formats: tuple[Format, ...],
    toolcall_fixtures: tuple[Fixture, ...],
    *,
    calls: dict[str, dict[str, Any] | None],
    boundary: tuple[int, ...],
    boundary_map: dict[str, tuple[int, ...]] | None = None,
    max_new_tokens: int = 64,
    seed: int = 0,
) -> dict[str, Any]:
    """Re-evaluate every candidate on synthetic tool-call fixtures.

    For every search candidate, the synthetic tool-call fixtures (see
    :mod:`toolcall`) are evaluated with ``core.evaluate()`` unchanged;
    each completed observation's decoded ``answer`` is then classified
    against the fixture's expected call by
    :func:`toolcall.classify_tool_call` into ``adherent`` /
    ``parse_failed`` / ``call_failed`` / ``not_text`` and aggregated by
    :func:`toolcall.candidate_adherence` into selection/args/relevance
    rates. Greedy, exact. Prompt-only: no native function-calling APIs --
    the stage measures what the template alone elicits from prompt text.

    Any backend construction or evaluation failure records None for that
    candidate cell -- fail-closed, never silently skipped. ``calls``
    maps every fixture id to its expected call (or ``None`` for the
    abstain queries); malformed call entries raise ``ValueError``
    fail-fast.

    Returns ``{"n_fixtures": ..., "per_candidate": {name:
    {"n", "n_evaluable", "adherent", "parse_failed", "call_failed",
    "not_text", "selection_rate", "args_rate", "relevance_rate",
    "adherent_rate"} | None}}``. Raises ``ValueError`` on
    caller-contract violations (None backend, bad formats/fixtures/
    calls/boundary), never on backend data.
    """
    if backend is None:
        raise ValueError("backend must not be None")
    if type(formats) is not tuple or not formats:
        raise ValueError("formats must be a nonempty tuple")
    if type(toolcall_fixtures) is not tuple or not toolcall_fixtures:
        raise ValueError("toolcall_fixtures must be a nonempty tuple")
    if any(type(case) is not Fixture for case in toolcall_fixtures):
        raise ValueError("toolcall_fixtures must contain Fixture values")
    if type(calls) is not dict or not all(
        type(key) is str and (value is None or type(value) is dict) for key, value in calls.items()
    ):
        raise ValueError("calls must map fixture ids to call dicts or None")
    for fixture_id, call in calls.items():
        if call is None:
            continue
        name = call.get("name")
        arguments = call.get("arguments")
        if type(name) is not str or not name:
            raise ValueError(f"calls[{fixture_id!r}] has a bad tool name")
        if type(arguments) is not dict:
            raise ValueError(f"calls[{fixture_id!r}] has bad arguments")
    for case in toolcall_fixtures:
        if case.id not in calls:
            raise ValueError(f"no call registered for fixture {case.id!r}")
    if not _valid_token_ids(boundary, nonempty=True):
        raise ValueError("Boundary must be a nonempty tuple of token IDs")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 4096:
        raise ValueError("max_new_tokens must be an integer between 1 and 4096")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    boundaries = _resolve_boundaries(formats, boundary, boundary_map)
    per_candidate: dict[str, dict[str, Any] | None] = {}
    for fmt in formats:
        try:
            report = evaluate(
                backend,
                fmt,
                toolcall_fixtures,
                boundary=boundaries[fmt.name],
                max_new_tokens=max_new_tokens,
                seed=seed,
            )
        except Exception:  # noqa: BLE001 -- one cell degrades, not the stage
            per_candidate[fmt.name] = None
            continue
        if not report.aggregate_eligible:
            per_candidate[fmt.name] = None
            continue
        classifications: list[dict[str, Any]] = []
        for obs in report.observations:
            if obs.status == "completed":
                classifications.append(classify_tool_call(obs.answer, calls[obs.fixture]))
        per_candidate[fmt.name] = toolcall_candidate_adherence(classifications)
    return {
        "n_fixtures": len(toolcall_fixtures),
        "per_candidate": per_candidate,
    }


def _toolcall_cell_valid(entry: Any) -> bool:
    """Structural check for one candidate cell of a toolcall stage."""
    if entry is None:
        return True
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
    if entry["n_evaluable"] != 0:
        return True
    return not any(
        entry.get(key) is not None
        for key in ("selection_rate", "args_rate", "relevance_rate", "adherent_rate")
    )


def toolcall_adherence(
    candidates: tuple[str, ...],
    stage: dict[str, Any],
) -> dict[str, Any]:
    """Analyze tool-call adherence across candidates (pure post-hoc).

    Answers "does the template change tool-call adherence?": reports the
    per-candidate selection/args/relevance rates from the stage, the
    weakest candidates (minimum adherent rate), and the max-minus-min
    gaps per rate -- the pre-registered falsifiable claim (see
    :mod:`toolcall`) is that the gaps are measurably positive rather
    than ~0; ``uniform`` reports the null hypothesis (all evaluable
    candidates identical on every rate), never asserts it.

    Never raises on malformed input: returns ``{"available": False}``
    with a ``detail`` code.
    """
    if (
        not isinstance(candidates, tuple)
        or not candidates
        or not all(type(c) is str and c.strip() for c in candidates)
        or len(set(candidates)) != len(candidates)
    ):
        return {"available": False, "detail": "invalid_candidates"}
    if type(stage) is not dict:
        return {"available": False, "detail": "stage_invalid"}
    n_fixtures = stage.get("n_fixtures")
    if type(n_fixtures) is not int or n_fixtures < 0:
        return {"available": False, "detail": "stage_invalid"}
    per_candidate = stage.get("per_candidate")
    if type(per_candidate) is not dict or not all(type(k) is str for k in per_candidate):
        return {"available": False, "detail": "stage_invalid"}
    for entry in per_candidate.values():
        if not _toolcall_cell_valid(entry):
            return {"available": False, "detail": "stage_invalid"}
    rate_keys = ("selection_rate", "args_rate", "relevance_rate", "adherent_rate")
    evaluable: dict[str, dict[str, float]] = {}
    excluded: list[str] = []
    for cand in candidates:
        entry = per_candidate.get(cand)
        if entry is None:
            excluded.append(cand)
            continue
        if type(entry) is not dict or entry.get("n_evaluable", 0) == 0:
            excluded.append(cand)
            continue
        rates: dict[str, float] = {}
        for key in rate_keys:
            rate = entry.get(key)
            if type(rate) is int or type(rate) is float:
                rates[key] = float(rate)
            else:
                break
        if len(rates) != len(rate_keys):
            excluded.append(cand)
            continue
        evaluable[cand] = rates
    gaps: dict[str, float | None] = {}
    weakest: list[str] = []
    uniform: bool | None = None
    if evaluable:
        for key in rate_keys:
            values = [rates[key] for rates in evaluable.values()]
            gaps[f"{key}_gap"] = max(values) - min(values)
        adherent = {name: rates["adherent_rate"] for name, rates in evaluable.items()}
        floor = min(adherent.values())
        weakest = sorted(name for name, rate in adherent.items() if rate == floor)
        uniform = all(
            all(rates[key] == first for name, rates in evaluable.items())
            for key, first in next(iter(evaluable.values())).items()
        )
    else:
        gaps = {f"{key}_gap": None for key in rate_keys}
    return {
        "available": True,
        "n_fixtures": n_fixtures,
        "per_candidate": {cand: per_candidate.get(cand) for cand in candidates},
        "weakest_candidates": weakest,
        "excluded_candidates": sorted(excluded),
        **gaps,
        "uniform": uniform,
    }


def _verbosity_minimal_tokens(
    backend: Backend, expected_by_fixture: dict[str, Any]
) -> dict[str, int]:
    """Exact minimal-sufficient token counts per brevity-ideal fixture.

    Tokenizes the stripped expected answer with the backend's own
    tokenizer (no implicit BOS/EOS per the Backend contract). Raises
    ``ValueError`` when the backend cannot tokenize (fail-fast: a
    backend that cannot encode cannot supply the baseline).
    """
    encode = getattr(backend, "encode", None)
    if not callable(encode):
        raise ValueError("backend must expose a callable encode(text)")
    minimal: dict[str, int] = {}
    for fixture_id, expected in expected_by_fixture.items():
        if not is_brevity_ideal(expected):
            continue
        try:
            token_ids = encode(str(expected).strip())
        except Exception as exc:
            raise ValueError(f"backend failed to encode {fixture_id!r}: {exc}") from exc
        if type(token_ids) is not tuple or any(type(t) is not int or t < 0 for t in token_ids):
            raise ValueError(f"backend.encode({fixture_id!r}) did not return token ids")
        minimal[fixture_id] = len(token_ids)
    return minimal


def verbosity_stage(
    *,
    observations: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    expected_by_fixture: dict[str, Any],
    backend: Backend,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Measure per-candidate response verbosity post-hoc over observations.

    Purely post-hoc: no model calls. For every observation of a
    brevity-ideal fixture (see :mod:`verbosity`), the generated token
    count is ``len(output_ids)`` and the excess over the minimal
    answer is computed by :func:`verbosity.excess_tokens`. Each
    candidate's excess distribution plus its budget-saturation share
    (fraction of evaluable observations that reached ``max_new_tokens``;
    a saturated generation's measured length is a lower bound on true
    verbosity -- the cycle-155 stop-cause confound, reported not
    hidden) is aggregated by :func:`verbosity.candidate_verbosity`.

    ``observations`` are flat observation dicts (as produced by
    ``asdict(Observation)``); ``expected_by_fixture`` maps fixture id
    to the fixture's expected answer text. Malformed observations are
    excluded from the evaluable set (fail-closed: visible via ``n`` vs
    ``n_evaluable``, never crashing the stage). Raises ``ValueError``
    on caller-contract violations (bad container types, unusable
    backend, bad ``max_new_tokens``), never on observation data.

    Returns ``{"n_observations", "n_fixtures", "n_brevity_fixtures",
    "per_candidate": {name: cell}}`` where each cell is the
    :func:`verbosity.candidate_verbosity` dict.
    """
    if type(observations) is not tuple and type(observations) is not list:
        raise ValueError("observations must be a tuple or list of observation dicts")
    if type(expected_by_fixture) is not dict or not all(
        type(key) is str for key in expected_by_fixture
    ):
        raise ValueError("expected_by_fixture must be a dict with string keys")
    if backend is None:
        raise ValueError("backend must not be None")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 4096:
        raise ValueError("max_new_tokens must be an integer between 1 and 4096")
    minimal = _verbosity_minimal_tokens(backend, expected_by_fixture)
    records: dict[str, list[tuple[int, int, bool]]] = {}
    seen: dict[str, int] = {}
    n_observations = 0
    for obs in observations:
        if type(obs) is not dict:
            continue
        candidate = obs.get("candidate")
        if type(candidate) is not str or not candidate:
            continue
        n_observations += 1
        seen[candidate] = seen.get(candidate, 0) + 1
        fixture_id = obs.get("fixture")
        status = obs.get("status")
        output_ids = obs.get("output_ids")
        if (
            status != "completed"
            or type(fixture_id) is not str
            or fixture_id not in minimal
            or not isinstance(output_ids, (tuple, list))
            or not all(type(t) is int and t >= 0 for t in output_ids)
        ):
            # Fail-closed: the observation stays visible via the
            # candidate's n vs n_evaluable, never counted and never
            # crashing the stage.
            continue
        generated = len(output_ids)
        records.setdefault(candidate, []).append(
            (generated, minimal[fixture_id], generated >= max_new_tokens)
        )
    per_candidate: dict[str, dict[str, Any]] = {}
    for candidate, n_seen in seen.items():
        cell = candidate_verbosity(records.get(candidate, []))
        # n counts every observation seen for the candidate, including
        # the fail-closed exclusions above; n_evaluable counts only the
        # measured ones.
        cell["n"] = n_seen
        per_candidate[candidate] = cell
    return {
        "n_observations": n_observations,
        "n_fixtures": len(expected_by_fixture),
        "n_brevity_fixtures": len(minimal),
        "per_candidate": per_candidate,
    }


def _verbosity_cell_valid(entry: Any) -> bool:
    """Structural check for one candidate cell of a verbosity stage."""
    if entry is None:
        return True
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
    stats = ("median_excess", "p25_excess", "p75_excess", "iqr_excess", "mean_excess", "max_excess")
    if entry["n_evaluable"] == 0:
        return all(entry.get(key) is None for key in stats) and entry.get("capped_share") is None
    median = entry.get("median_excess")
    p25 = entry.get("p25_excess")
    p75 = entry.get("p75_excess")
    # isinstance narrows for mypy where type() in (...) does not; bools
    # were already rejected by the per-key loop above, so no bool can
    # reach this comparison.
    return (
        isinstance(p25, (int, float))
        and isinstance(median, (int, float))
        and isinstance(p75, (int, float))
        and p25 <= median <= p75
    )


def verbosity_profile(
    candidates: tuple[str, ...],
    stage: dict[str, Any],
) -> dict[str, Any]:
    """Analyze response verbosity across candidates (pure post-hoc).

    Answers "does the template change verbosity?": reports the
    per-candidate excess distributions from the stage, the
    max-minus-min median-excess gap -- the pre-registered falsifiable
    claim (see :mod:`verbosity`) is that the gap is measurably
    positive rather than ~0; ``uniform`` reports the null hypothesis
    (all evaluable candidates identical on the distribution shape),
    never asserts it. ``most_verbose`` / ``least_verbose`` are behavior
    labels for the cost profile, never quality judgments (shorter is
    not better). ``capped_candidates`` names every candidate with a
    nonzero budget-saturation share: their measured lengths are lower
    bounds, unconfounding this experiment from the stop-cause audit.

    Never raises on malformed input: returns ``{"available": False}``
    with a ``detail`` code.
    """
    if (
        not isinstance(candidates, tuple)
        or not candidates
        or not all(type(c) is str and c.strip() for c in candidates)
        or len(set(candidates)) != len(candidates)
    ):
        return {"available": False, "detail": "invalid_candidates"}
    if type(stage) is not dict:
        return {"available": False, "detail": "stage_invalid"}
    n_fixtures = stage.get("n_fixtures")
    if type(n_fixtures) is not int or n_fixtures < 0:
        return {"available": False, "detail": "stage_invalid"}
    n_brevity = stage.get("n_brevity_fixtures")
    if type(n_brevity) is not int or n_brevity < 0 or n_brevity > n_fixtures:
        return {"available": False, "detail": "stage_invalid"}
    per_candidate = stage.get("per_candidate")
    if type(per_candidate) is not dict or not all(type(k) is str for k in per_candidate):
        return {"available": False, "detail": "stage_invalid"}
    for entry in per_candidate.values():
        if not _verbosity_cell_valid(entry):
            return {"available": False, "detail": "stage_invalid"}
    evaluable: dict[str, float] = {}
    capped: list[str] = []
    excluded: list[str] = []
    for cand in candidates:
        entry = per_candidate.get(cand)
        if (
            type(entry) is not dict
            or entry.get("n_evaluable", 0) == 0
            or type(entry.get("median_excess")) not in (int, float)
        ):
            excluded.append(cand)
            continue
        evaluable[cand] = float(entry["median_excess"])
        share = entry.get("capped_share")
        # isinstance narrows for mypy; the cell validator already
        # excluded bools, so no bool reaches the comparison.
        if isinstance(share, (int, float)) and share > 0:
            capped.append(cand)
    median_excess_gap: float | None = None
    most_verbose: list[str] = []
    least_verbose: list[str] = []
    uniform: bool | None = None
    if evaluable:
        values = list(evaluable.values())
        median_excess_gap = max(values) - min(values)
        top = max(values)
        bottom = min(values)
        most_verbose = sorted(name for name, value in evaluable.items() if value == top)
        least_verbose = sorted(name for name, value in evaluable.items() if value == bottom)
        shape_keys = (
            "median_excess",
            "p25_excess",
            "p75_excess",
            "iqr_excess",
            "mean_excess",
            "max_excess",
        )
        first = per_candidate[next(iter(evaluable))]
        uniform = all(
            all(per_candidate[name].get(key) == first.get(key) for name in evaluable)
            for key in shape_keys
        )
    return {
        "available": True,
        "n_fixtures": n_fixtures,
        "n_brevity_fixtures": n_brevity,
        "per_candidate": {cand: per_candidate.get(cand) for cand in candidates},
        "most_verbose": most_verbose,
        "least_verbose": least_verbose,
        "median_excess_gap": median_excess_gap,
        "uniform": uniform,
        "capped_candidates": sorted(capped),
        "excluded_candidates": sorted(excluded),
    }


def refusal_profile(result: dict[str, Any]) -> dict[str, Any]:
    """Per-candidate suspected-refusal profile (pure post-hoc).

    Answers "does the template change how often the checkpoint
    refuses benign fixtures?" from the run's recorded search
    observations -- zero extra model calls. The detector contract
    (frozen opener list, beginning-anchored, suspected- naming) is
    :mod:`refusals`; this assembly adds the run-level discipline:
    rates are withheld for candidates whose search did not complete
    (the interrupted candidate was measured on a fixture prefix),
    and ``evaluable_mismatch`` flags candidates whose completed
    fixture sets differ, so rates are never compared across unlike
    denominators in silence.

    Never raises on malformed input: returns ``{"available": False}``
    with a ``detail`` code for a structurally unusable result; a
    result whose candidates are all unusable stays ``available``
    with per-candidate records saying why (an explained absence,
    never a silent one).
    """
    if type(result) is not dict:
        return {"available": False, "detail": "result_not_a_dict"}
    entries = result.get("search")
    if type(entries) is not list or not entries:
        return {"available": False, "detail": "no_search_entries"}
    candidates: list[dict[str, Any]] = []
    completed_sets: list[frozenset[str]] = []
    n_available = 0
    for entry in entries:
        fmt = entry.get("format") if type(entry) is dict else None
        name = fmt.get("name") if type(fmt) is dict else None
        if type(entry) is not dict or type(name) is not str or not name:
            candidates.append({"candidate": None, "available": False, "detail": "entry_malformed"})
            continue
        summary = entry.get("summary")
        eligible = type(summary) is dict and summary.get("aggregate_eligible") is True
        record: dict[str, Any] = {"candidate": name}
        record.update(analyze_candidate(entry.get("observations"), eligible=eligible))
        candidates.append(record)
        if record["available"]:
            n_available += 1
            completed_sets.append(
                frozenset(
                    observation["fixture"]
                    for observation in entry["observations"]
                    if type(observation) is dict
                    and observation.get("status") == "completed"
                    and type(observation.get("fixture")) is str
                )
            )
    return {
        "available": True,
        "method": {
            "detector": DETECTOR_ID,
            "openers": list(REFUSAL_OPENERS),
            "openers_sha256": OPENERS_SHA256,
            "normalization": (
                "nfkc, cf-to-space, casefold, combining-mark-strip, apostrophe-fold, "
                "whitespace-collapse; beginning-anchored with word boundary"
            ),
            "scope": (
                "search observations only; recorded (boundary-cut) answers; "
                "completed observations only"
            ),
        },
        "evaluable_mismatch": len(set(completed_sets)) > 1,
        "candidates": candidates,
        "n_candidates": len(candidates),
        "n_candidates_available": n_available,
    }


def scorerlens_profile(
    result: dict[str, Any], expected_by_fixture: dict[str, str]
) -> dict[str, Any]:
    """Scorer-lens regrade of the recorded search answers (pure post-hoc).

    Answers "how much of each template's failure is the byte-exact
    scorer, not the answer?" by regrading every recorded search
    answer under the frozen normalisation ladder in
    :mod:`scorerlens` -- zero extra model calls. The expected texts
    come from the run's own fixture set, supplied by the caller:
    recorded results deliberately carry no ``expected`` field, so
    the join is an explicit parameter (the verbosity precedent),
    and it is fail-closed per candidate (see
    :func:`scorerlens.analyze_candidate`).

    The winner readout is the *search-phase* winner under each
    rung, re-applying the conservative bar (every evaluable fixture
    verdict-passes) to the rung verdict counts; held-out is never
    regraded. It is withheld whenever any candidate is unavailable
    or ineligible, or the evaluable fixture sets differ -- a winner
    computed over a partial field would be the interrupted-run sin
    one level up. Surface-failure shares are withheld per candidate
    on the same eligibility gate as the refusal rates.

    Never raises on malformed input: returns ``{"available":
    False}`` with a ``detail`` code for a structurally unusable
    result or expected map.
    """
    if type(result) is not dict:
        return {"available": False, "detail": "result_not_a_dict"}
    if type(expected_by_fixture) is not dict or not all(
        type(k) is str and type(v) is str for k, v in expected_by_fixture.items()
    ):
        return {"available": False, "detail": "expected_map_invalid"}
    entries = result.get("search")
    if type(entries) is not list or not entries:
        return {"available": False, "detail": "no_search_entries"}
    candidates: list[dict[str, Any]] = []
    evaluable_sets: list[frozenset[str]] = []
    n_available = 0
    for entry in entries:
        fmt = entry.get("format") if type(entry) is dict else None
        name = fmt.get("name") if type(fmt) is dict else None
        if type(entry) is not dict or type(name) is not str or not name:
            candidates.append({"candidate": None, "available": False, "detail": "entry_malformed"})
            continue
        summary = entry.get("summary")
        eligible = type(summary) is dict and summary.get("aggregate_eligible") is True
        record: dict[str, Any] = {"candidate": name}
        record.update(
            scorerlens_analyze_candidate(
                entry.get("observations"), expected_by_fixture, eligible=eligible
            )
        )
        candidates.append(record)
        if record["available"]:
            n_available += 1
            evaluable_sets.append(
                frozenset(r["fixture"] for r in record["records"] if not r["excluded"])
            )
    evaluable_mismatch = len(set(evaluable_sets)) > 1
    fixture_set_sha256 = sha256("\n".join(sorted(expected_by_fixture)).encode("utf-8")).hexdigest()
    native = result.get("selected_candidate")
    native_winner = native if type(native) is str and native else None
    winner_by_rung: dict[str, str | None] | None = None
    winner_detail = ""
    if n_available != len(candidates) or not candidates:
        winner_detail = "candidate_unavailable"
    elif any(c.get("eligible") is not True for c in candidates):
        winner_detail = "candidate_not_eligible"
    elif evaluable_mismatch:
        winner_detail = "evaluable_mismatch"
    else:
        winner_by_rung = {}
        for rung in ("0", "1", "2", "3", "4"):
            winners = [
                c["candidate"]
                for c in candidates
                if c["verdict_counts"][rung] == c["n_evaluable"] and c["n_evaluable"] > 0
            ]
            winner_by_rung[rung] = winners[0] if len(winners) == 1 else None
    return {
        "available": True,
        "method": {
            "protocol": SCORERLENS_LADDER_ID,
            "ladder": SCORERLENS_LADDER_SPEC,
            "ladder_sha256": scorerlens_ladder_sha256(),
            "scope": (
                "search observations only; boundary-cut recorded answers; "
                "expected texts joined from the run's own fixture set, "
                "never recorded; held-out is never regraded"
            ),
        },
        "fixture_set_sha256": fixture_set_sha256,
        "evaluable_mismatch": evaluable_mismatch,
        "native_winner": native_winner,
        "winner_by_rung": winner_by_rung,
        "winner_detail": winner_detail,
        "candidates": candidates,
        "n_candidates": len(candidates),
        "n_candidates_available": n_available,
    }


def aliasing_profile(result: dict[str, Any]) -> dict[str, Any]:
    """Candidate aliasing audit of the recorded search entries (pure post-hoc).

    Answers "did the run actually test distinct templates?" from
    the recorded search observations -- zero extra model calls, and
    unlike the scorer-lens no extra join: everything the audit
    consumes (input/output IDs, outcomes, per-entry boundaries,
    summaries, decision, backend fingerprint) is in the recorded
    result. The class machinery is :mod:`aliasing`; this assembly
    adds the run-level discipline: classes exist only inside
    cohorts of identical completed fixture sets, effective counts
    and the redundant count come from the primary cohort only, and
    the alias-suspect flag is recomputed from the ground records
    under the fixed withholding order (an unavailable candidate,
    then a non-ambiguous decision, then an inconsistent passing
    set, then an outcome-suspect cell in the containing class,
    then a passing set spanning render classes, then differing
    boundaries inside one render class).

    Never raises on malformed input: returns ``{"available":
    False}`` with a ``detail`` code for a structurally unusable
    result; a result whose candidates are all unusable stays
    ``available`` with per-candidate records saying why. A result
    whose every search entry lacks recorded observations -- the
    public ``results.json`` shape -- is ``observations_stripped``:
    the audit is unavailable there by design, never a null result
    manufactured from absent input.
    """
    if type(result) is not dict:
        return {"available": False, "detail": "result_not_a_dict"}
    entries = result.get("search")
    if type(entries) is not list or not entries:
        return {"available": False, "detail": "no_search_entries"}
    if all(type(e) is dict and "observations" not in e for e in entries):
        return {"available": False, "detail": "observations_stripped"}
    candidates: list[dict[str, Any]] = []
    by_name: dict[str, dict[str, Any]] = {}
    for entry in entries:
        fmt = entry.get("format") if type(entry) is dict else None
        name = fmt.get("name") if type(fmt) is dict else None
        if type(entry) is not dict or type(name) is not str or not name:
            candidates.append({"candidate": None, "available": False, "detail": "entry_malformed"})
            continue
        if name in by_name:
            # A second entry for one name: the name is unavailable
            # as a whole (the producer-side rule; the validator
            # rejects duplicate names in the payload outright).
            prior = by_name[name]
            prior.clear()
            prior.update(
                {"candidate": name, "available": False, "detail": "duplicate_candidate_entry"}
            )
            continue
        summary = entry.get("summary")
        eligible = type(summary) is dict and summary.get("aggregate_eligible") is True
        planned = summary.get("planned") if type(summary) is dict else None
        record: dict[str, Any] = {"candidate": name}
        record.update(
            aliasing_analyze_candidate(
                entry.get("observations"),
                candidate_name=name,
                boundary_token_ids=entry.get("boundary_token_ids"),
                eligible=eligible,
                planned=planned,
            )
        )
        candidates.append(record)
        by_name[name] = record
    available = [c for c in candidates if c.get("available") is True]
    evaluable_mismatch = len({aliasing_evaluable_fixtures(c) for c in available}) > 1
    primary = aliasing_primary_cohort(available)
    primary_names = {str(c.get("candidate")) for c in primary}
    render_classes, measurement_classes = aliasing_class_partitions(primary)
    for record in available:
        name = str(record.get("candidate"))
        record["in_primary_cohort"] = name in primary_names
        record["render_class_members"] = next((cls for cls in render_classes if name in cls), None)
        record["measurement_class_members"] = next(
            (cls for cls in measurement_classes if name in cls), None
        )
    excluded = sorted(
        str(c.get("candidate")) for c in available if c.get("candidate") not in primary_names
    )
    n_redundant_measurement, n_redundant_render = aliasing_redundant_counts(primary)
    suspects = aliasing_suspect_cells(primary, measurement_classes)
    primary_fixtures = sorted(aliasing_evaluable_fixtures(primary[0])) if primary else []
    passing = sorted(
        str(c.get("candidate"))
        for c in primary
        if c.get("eligible") is True
        and c.get("n_evaluable", 0) > 0
        and all(r["exact_match"] and r["boundary_found"] for r in c["records"] if r["evaluable"])
    )
    flag, flag_detail, flag_members = _aliasing_flag(
        decision=result.get("decision"),
        n_candidates=len(candidates),
        n_available=len(available),
        passing=passing,
        render_classes=render_classes,
        measurement_classes=measurement_classes,
        suspects=suspects,
    )
    decision = result.get("decision")
    selected = result.get("selected_candidate")
    fingerprint = result.get("backend_fingerprint")
    seed = result.get("seed")
    cap = result.get("max_new_tokens")
    return {
        "available": True,
        "method": {
            "protocol": ALIASING_AUDIT_ID,
            "token_ids_digest": ALIASING_DIGEST_VERSION,
            "scope": ALIASING_SCOPE,
        },
        "decision": decision if type(decision) is str else None,
        "selected_candidate": selected if type(selected) is str and selected else None,
        "backend_fingerprint": fingerprint if type(fingerprint) is str else None,
        "seed": seed if type(seed) is int else None,
        "max_new_tokens": cap if type(cap) is int else None,
        "fixture_set_sha256": sha256("\n".join(primary_fixtures).encode("utf-8")).hexdigest(),
        "evaluable_mismatch": evaluable_mismatch,
        "candidates": candidates,
        "n_candidates_recorded": len(candidates),
        "n_candidates_available": len(available),
        "n_primary_cohort": len(primary),
        "excluded_candidates": excluded,
        "n_candidates_excluded": len(excluded),
        "n_effective_render": len(render_classes),
        "n_effective_measurement": len(measurement_classes),
        "render_classes": render_classes,
        "measurement_classes": measurement_classes,
        "n_redundant_completed_observations": n_redundant_measurement,
        "n_redundant_render_observations": n_redundant_render,
        "determinism_suspect_cells": suspects,
        "n_determinism_suspect_cells": len(suspects),
        "n_outcome_suspect_cells": sum(1 for c in suspects if c["kind"] in ("outcome", "both")),
        "pairs": aliasing_pair_matrix(available),
        "passing_candidates": passing,
        "alias_suspect_abstention": flag,
        "alias_suspect_detail": flag_detail,
        "alias_suspect_class_members": flag_members,
    }


def _aliasing_flag(
    *,
    decision: Any,
    n_candidates: int,
    n_available: int,
    passing: list[str],
    render_classes: list[list[str]],
    measurement_classes: list[list[str]],
    suspects: list[dict[str, Any]],
) -> tuple[bool, str, list[str]]:
    """The alias-suspect flag under its fixed withholding order.

    The order is part of the contract (the diff validator
    recomputes the expected reason in the same order): an
    unavailable candidate first, then a non-ambiguous decision,
    then a passing set too small to explain an ``ambiguous``
    decision, then an outcome-suspect cell inside the containing
    measurement class (the aliasing explanation is exactly what
    such a cell refutes), then a passing set spanning render
    classes (a genuine tie between distinct prompts -- the lab
    working as intended, not an artifact), then one render class
    with differing boundaries. ``detail`` is ``""`` exactly when
    the flag fires (the scorer-lens ``winner_detail`` discipline);
    the fired case's class members travel in their own field.
    """
    if n_available != n_candidates:
        return False, "candidate_unavailable", []
    if decision != "ambiguous":
        return False, "decision_not_ambiguous", []
    if len(passing) < 2:
        return False, "passing_set_inconsistent", []
    passing_set = set(passing)
    for cls in measurement_classes:
        if passing_set <= set(cls):
            if any(
                cell["kind"] in ("outcome", "both") and set(cell["candidates"]) == set(cls)
                for cell in suspects
            ):
                return False, "determinism_suspect_in_class", []
            return True, "", list(cls)
    if not any(passing_set <= set(cls) for cls in render_classes):
        return False, "passing_spans_classes", []
    return False, "boundaries_differ_within_passing_set", []


def repetition_stage(
    backend: Any,
    formats: tuple[Any, ...],
    cases: tuple[Any, ...],
    search_entries: Any,
    *,
    boundary: tuple[int, ...],
    boundary_map: dict[str, tuple[int, ...]] | None = None,
    max_new_tokens: int = 64,
    max_context: int,
    seed: int = 0,
    cancel: Event | None = None,
) -> dict[str, Any]:
    """Prompt-repetition stage (cycle 183): x2 and pad arms per candidate.

    The x1 arm is the recorded native search (reused, never
    regenerated — the healing precedent, but extracted from the raw
    observations because the ground records need ``boundary_found``,
    ``boundary_text_match`` and token counts separately, which the
    fused native-outcomes helper discards). The treated arms rewrite
    the last user message's content before render and are evaluated
    per fixture, so one overlong fixture degrades alone instead of
    destroying the candidate's whole cell. Cancellation is honoured
    between fixtures; a cancelled stage returns no partial aggregates.
    Design: docs/REPETITION_DESIGN.md.
    """
    from . import repetition as _rep
    from .fixtures import audit_special_token_literals

    if type(formats) is not tuple or not formats:
        raise ValueError("formats must be a nonempty tuple")
    if type(cases) is not tuple or not cases:
        raise ValueError("cases must be a nonempty tuple")
    protocol = _rep.protocol_block([case.id for case in cases], _backend_fingerprint(backend))
    if len(formats) * len(cases) > _rep.MAX_GROUND_RECORDS:
        return {
            "available": False,
            "detail": "repetition_cell_budget_exceeded",
            "protocol": protocol,
            "candidates": {},
        }
    if audit_special_token_literals(_rep.FILLER_UNIT, formats):
        # The filler ships clean against the shipped catalog; a
        # candidate whose framing inventory collides with it would
        # make every pad arm carry a special-token literal, so the
        # stage fails closed instead of measuring a forged boundary.
        return {
            "available": False,
            "detail": "filler_screen_failed",
            "protocol": protocol,
            "candidates": {},
        }
    boundaries = _resolve_boundaries(formats, boundary, boundary_map)
    entries: dict[str, Any] = {}
    if type(search_entries) is list:
        for entry in search_entries:
            if type(entry) is not dict:
                continue
            fmt_dict = entry.get("format")
            name = fmt_dict.get("name") if type(fmt_dict) is dict else None
            if type(name) is str and name:
                entries[name] = entry
    expected_by_fixture = {case.id: case.expected for case in cases}
    boundary_text: dict[str, str] = {}
    for fmt in formats:
        try:
            boundary_text[fmt.name] = backend.decode(boundaries[fmt.name])
        except Exception:  # noqa: BLE001 -- an undecodable boundary screens nothing
            boundary_text[fmt.name] = ""
    screened_fixtures, screened_candidates = _rep.screen_filler(expected_by_fixture, boundary_text)
    unit_estimate = max(1, len(backend.encode(_rep.FILLER_UNIT)))
    marker = "\x00mplrep\x00"

    def _cancelled() -> bool:
        return cancel is not None and cancel.is_set()

    candidates: dict[str, Any] = {}
    for fmt in formats:
        name = fmt.name
        entry = entries.get(name)
        summary = entry.get("summary") if type(entry) is dict else None
        observations = entry.get("observations") if type(entry) is dict else None
        if (
            type(entry) is not dict
            or type(observations) is not list
            or not observations
            or (type(summary) is dict and summary.get("aggregate_eligible") is False)
        ):
            candidates[name] = {
                "status": "unavailable",
                "detail": "native_search_unavailable",
                "think_channel": opens_think_channel(fmt),
                "render_signature": None,
                "records": [],
            }
            continue
        obs_by_fixture: dict[str, Any] = {}
        usable = True
        for obs in observations:
            if type(obs) is not dict or type(obs.get("fixture")) is not str:
                usable = False
                break
            obs_by_fixture[obs["fixture"]] = obs
        if not usable or any(case.id not in obs_by_fixture for case in cases):
            candidates[name] = {
                "status": "unavailable",
                "detail": "native_search_unavailable",
                "think_channel": opens_think_channel(fmt),
                "render_signature": None,
                "records": [],
            }
            continue
        signature = sha256(
            json.dumps([list(obs_by_fixture[case.id]["input_ids"]) for case in cases]).encode(
                "utf-8"
            )
        ).hexdigest()
        records: list[dict[str, Any]] = []
        for case in cases:
            if _cancelled():
                return {
                    "available": False,
                    "detail": "cancelled",
                    "protocol": protocol,
                    "candidates": {},
                }
            obs = obs_by_fixture[case.id]
            ids_x1 = tuple(obs["input_ids"])
            x1_cell = {
                "verdict": bool(obs["exact_match"] and obs["boundary_found"]),
                "boundary_found": bool(obs["boundary_found"]),
                "boundary_text_match": obs["boundary_text_match"],
                "input_tokens": len(ids_x1),
            }
            content = _rep.target_content(case)
            vacuous = _rep.is_vacuous(content)
            record: dict[str, Any] = {
                "fixture": case.id,
                "vacuous": vacuous,
                "pre_repeated": _rep.detect_pre_repeated(content),
                "echo_fixture": _rep.is_echo_fixture(case),
                "construction": "vacuous" if vacuous else "text_level",
                "seam_ids": [],
                "delta_tokens": None,
                "filler_tokens": None,
                "pad_evaluable": False,
                "arms": {"x1": x1_cell, "x2": None, "pad": None},
                "arm_details": {"x2": "", "pad": ""},
            }
            records.append(record)
            if vacuous:
                record["arm_details"] = {"x2": "vacuous", "pad": "vacuous"}
                continue
            # Construction affixes: render the fixture with the target
            # content replaced by a marker and split — Format.render is
            # pure concatenation, so the render is prefix+marker+suffix.
            prefix_ids: tuple[int, ...] | None = None
            suffix_ids: tuple[int, ...] | None = None
            joiner_region: tuple[int, ...] | None = None
            if marker not in content:
                marker_fixture = _fixture_with_content(case, marker)
                rendered_marker = fmt.render(marker_fixture.messages)
                if rendered_marker.count(marker) == 1:
                    head, tail = rendered_marker.split(marker)
                    prefix_ids = backend.encode(head)
                    suffix_ids = backend.encode(tail)
                    joiner_fixture = _fixture_with_content(case, _rep.JOINER)
                    joiner_ids = backend.encode(fmt.render(joiner_fixture.messages))
                    joiner_region = _rep.content_region(joiner_ids, prefix_ids, suffix_ids)
            x2_fixture = _rep.build_arm_fixture(case, "x2")
            ids_x2 = backend.encode(fmt.render(x2_fixture.messages))
            record["delta_tokens"] = len(ids_x2) - len(ids_x1)
            region_x1 = (
                _rep.content_region(ids_x1, prefix_ids, suffix_ids)
                if prefix_ids is not None and suffix_ids is not None
                else None
            )
            region_x2 = (
                _rep.content_region(ids_x2, prefix_ids, suffix_ids)
                if prefix_ids is not None and suffix_ids is not None
                else None
            )
            construction, seam = _rep.classify_construction(region_x1, region_x2, joiner_region)
            record["construction"] = construction
            record["seam_ids"] = list(seam)
            # Pad sizing: exact rendered-length match, bounded search.
            pad_fixture: Any = None
            empty_pad_ids = backend.encode(
                fmt.render(_rep.build_arm_fixture(case, "pad", "").messages)
            )
            if name in screened_candidates or case.id in screened_fixtures:
                record["arm_details"]["pad"] = "pad_screened"
            elif len(empty_pad_ids) > len(ids_x2):
                record["arm_details"]["pad"] = "pad_unmatchable"
            else:
                for reps, cut in _rep.filler_search_order(
                    unit_estimate, int(record["delta_tokens"])
                ):
                    text = _rep.filler_text(reps, cut)
                    candidate_fixture = _rep.build_arm_fixture(case, "pad", text)
                    ids_pad = backend.encode(fmt.render(candidate_fixture.messages))
                    if len(ids_pad) == len(ids_x2):
                        filler_tokens = len(ids_pad) - len(empty_pad_ids)
                        if filler_tokens > 0:
                            pad_fixture = candidate_fixture
                            record["filler_tokens"] = filler_tokens
                            record["pad_evaluable"] = True
                            break
                        # A zero-filler match is a degenerate pad (a
                        # whitespace variant, not a filler control):
                        # keep searching for a positive-filler match.
                if pad_fixture is None and record["arm_details"]["pad"] == "":
                    record["arm_details"]["pad"] = "pad_unmatchable"
            # Generation, per arm, with the backend's own inequality.
            for arm, arm_fixture in (("x2", x2_fixture), ("pad", pad_fixture)):
                if arm_fixture is None:
                    continue
                ids_arm = (
                    ids_x2 if arm == "x2" else backend.encode(fmt.render(arm_fixture.messages))
                )
                if len(ids_arm) + max_new_tokens > max_context:
                    record["arm_details"][arm] = "arm_context_exceeded"
                    continue
                if _cancelled():
                    return {
                        "available": False,
                        "detail": "cancelled",
                        "protocol": protocol,
                        "candidates": {},
                    }
                report = evaluate(
                    backend,
                    fmt,
                    (arm_fixture,),
                    boundary=boundaries[name],
                    max_new_tokens=max_new_tokens,
                    seed=seed,
                    cancel=cancel,
                )
                if report.aggregate_eligible and report.observations:
                    arm_obs = report.observations[0]
                    record["arms"][arm] = {
                        "verdict": bool(arm_obs.exact_match and arm_obs.boundary_found),
                        "boundary_found": bool(arm_obs.boundary_found),
                        "boundary_text_match": arm_obs.boundary_text_match,
                        "input_tokens": len(arm_obs.input_ids),
                    }
                else:
                    record["arm_details"][arm] = "arm_evaluation_failed"
        candidates[name] = {
            "status": "available",
            "detail": "",
            "think_channel": opens_think_channel(fmt),
            "render_signature": signature,
            "records": records,
        }
    return {"available": True, "detail": "", "protocol": protocol, "candidates": candidates}


def _fixture_with_content(fixture: Any, content: str) -> Any:
    """A copy of ``fixture`` whose target message carries ``content``."""
    from . import repetition as _rep
    from .core import Fixture, Message

    index = _rep.target_index(fixture)
    messages = tuple(
        Message(role=m.role, content=content if i == index else m.content)
        for i, m in enumerate(fixture.messages)
    )
    return Fixture(id=fixture.id, messages=messages, expected=fixture.expected, group=fixture.group)


def repetition_analysis(
    stage: dict[str, Any],
    primary_decision: Any,
    primary_winner: Any,
    candidate_names: tuple[str, ...],
) -> dict[str, Any]:
    """Aggregate a repetition stage into the stored payload (cycle 183).

    Per-arm winners follow the conservative rule over each arm's
    completed, non-vacuous fixtures — and are *withheld* whenever any
    candidate is unavailable for the arm or the completed sets differ
    across candidates (the scorer-lens withholding discipline): a
    flip the denominators cannot explain is not published. The x1 arm
    never reports a flip (it is the recorded native run by
    construction).
    """
    from . import repetition as _rep

    if not stage.get("available"):
        return {
            "available": False,
            "detail": str(stage.get("detail", "repetition_unavailable")),
            "protocol": stage.get("protocol"),
        }
    if primary_decision in ("compatible_on_fixture_set", "held_out_failed"):
        mapped_primary = ("selected", primary_winner)
    else:
        mapped_primary = (primary_decision, primary_winner)
    stage_candidates = stage["candidates"]
    candidates_out: dict[str, Any] = {}
    for name in candidate_names:
        entry = stage_candidates[name]
        if entry["status"] != "available":
            candidates_out[name] = {
                "status": "unavailable",
                "detail": entry["detail"],
                "think_channel": entry["think_channel"],
                "render_signature": None,
                "records": [],
            }
            continue
        aggregates = _rep.aggregate_candidate(entry["records"])
        aggregates.update(
            {
                "status": "available",
                "detail": "",
                "think_channel": entry["think_channel"],
                "render_signature": entry["render_signature"],
                "records": entry["records"],
            }
        )
        candidates_out[name] = aggregates
    arms_out: dict[str, Any] = {}
    for arm in _rep.ARMS:
        completed_sets: list[frozenset[str]] = []
        all_pass: dict[str, bool | None] = {}
        withheld = False
        for name in candidate_names:
            entry = stage_candidates[name]
            if entry["status"] != "available":
                withheld = True
                all_pass[name] = None
                continue
            all_pass[name] = _rep.arm_all_pass(entry["records"], arm)
            completed_sets.append(
                frozenset(
                    r["fixture"]
                    for r in entry["records"]
                    if not r["vacuous"] and r["arms"][arm] is not None
                )
            )
            if all_pass[name] is None:
                withheld = True
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
        passers = [n for n in candidate_names if all_pass[n]]
        if len(passers) == 1:
            decision, winner = "selected", passers[0]
        elif len(passers) > 1:
            decision, winner = "ambiguous", None
        else:
            decision, winner = "no_candidate", None
        flip: bool | None = None
        if arm != "x1":
            flip = (decision, winner) != mapped_primary
        arms_out[arm] = {
            "decision": decision,
            "winner": winner,
            "withheld": False,
            "detail": "",
            "flip": flip,
        }
    think_groups: dict[str, Any] = {}
    for label, key in ((True, "think"), (False, "no_think")):
        effects: list[tuple[str, float, str | None]] = []
        for name in candidate_names:
            entry = candidates_out[name]
            if entry["status"] != "available" or entry["think_channel"] is not label:
                continue
            effect = entry["contrasts"]["x2_vs_pad"]["repetition_effect"]
            if effect is not None:
                effects.append((name, float(effect), entry["render_signature"]))
        dedup: dict[str, float] = {}
        for cand_name, effect, signature in sorted(effects):
            dedup.setdefault(signature or cand_name, effect)
        think_groups[key] = {
            "n_candidates": len(effects),
            "n_distinct_renders": len(dedup),
            "mean_repetition_effect": (
                sum(e for _n, e, _s in effects) / len(effects) if effects else None
            ),
            "mean_repetition_effect_dedup": (sum(dedup.values()) / len(dedup) if dedup else None),
        }
    signatures = {
        candidates_out[name]["render_signature"]
        for name in candidate_names
        if candidates_out[name]["status"] == "available"
    }
    return {
        "available": True,
        "detail": "",
        "protocol": stage["protocol"],
        "primary_decision": primary_decision,
        "primary_winner": primary_winner,
        "candidates": candidates_out,
        "arms": arms_out,
        "think_groups": think_groups,
        "n_candidates": len(candidate_names),
        "n_distinct_renders": len(signatures),
    }


def _answer_digest(value: Any) -> str | None:
    """SHA-256 of an answer string; None when there is no answer."""
    if type(value) is not str:
        return None
    return sha256(value.encode("utf-8")).hexdigest()


def distraction_stage(
    backend: Any,
    formats: tuple[Any, ...],
    cases: tuple[Any, ...],
    search_entries: Any,
    *,
    boundary: tuple[int, ...],
    boundary_map: dict[str, tuple[int, ...]] | None = None,
    max_new_tokens: int = 64,
    max_context: int,
    seed: int = 0,
    cancel: Event | None = None,
) -> dict[str, Any]:
    """Distractor-dilution stage (cycle 184): filler cells per candidate.

    The native arm is the recorded search (reused, never
    regenerated). Each treated cell inserts a filler span into
    the last user message's content before render — neutral
    corpus prose at 64/256/1024 rendered tokens, or a frozen
    near-miss template carrying a planted wrong answer at
    64/256 — sized per (candidate, fixture) by a bounded search
    over whole sentences plus one truncated suffix, verified on
    the rendered ids. Cells degrade individually (screened,
    unmatchable, context-exceeded, near-miss undefined); one bad
    fixture never takes a candidate's whole curve with it.
    Design: docs/DISTRACTION_DESIGN.md.
    """
    from . import distraction as _dis
    from . import repetition as _rep
    from .fixtures import audit_special_token_literals

    if type(formats) is not tuple or not formats:
        raise ValueError("formats must be a nonempty tuple")
    if type(cases) is not tuple or not cases:
        raise ValueError("cases must be a nonempty tuple")
    fixture_ids = [case.id for case in cases]
    protocol: dict[str, Any] = dict(
        _dis.protocol_object(fixture_ids, _backend_fingerprint(backend))
    )
    protocol["fingerprint"] = _dis.protocol_fingerprint(fixture_ids, _backend_fingerprint(backend))
    if len(formats) * len(cases) > _dis.MAX_GROUND_RECORDS:
        return {
            "available": False,
            "detail": "distraction_cell_budget_exceeded",
            "protocol": protocol,
            "candidates": {},
        }
    corpus_text = "".join(_dis.NEUTRAL_CORPUS)
    template_text = "".join(
        template.replace("{planted}", "") for template in _dis.NEARMISS_TEMPLATES
    )
    if audit_special_token_literals(corpus_text, formats) or (
        audit_special_token_literals(template_text, formats)
    ):
        # A special-token literal in the frozen texts would forge
        # turn structure in every cell of every run; there is no
        # honest per-cell version of that arm, so the stage fails
        # closed before any generation.
        return {
            "available": False,
            "detail": "filler_screen_failed",
            "protocol": protocol,
            "candidates": {},
        }
    boundaries = _resolve_boundaries(formats, boundary, boundary_map)
    entries: dict[str, Any] = {}
    if type(search_entries) is list:
        for entry in search_entries:
            if type(entry) is not dict:
                continue
            fmt_dict = entry.get("format")
            entry_name = fmt_dict.get("name") if type(fmt_dict) is dict else None
            if type(entry_name) is str and entry_name:
                entries[entry_name] = entry
    expected_values = {case.expected for case in cases}
    boundary_text: dict[str, str] = {}
    for fmt in formats:
        try:
            boundary_text[fmt.name] = backend.decode(boundaries[fmt.name])
        except Exception:  # noqa: BLE001 -- an undecodable boundary screens nothing
            boundary_text[fmt.name] = ""
    corpus_tokens = max(1, len(backend.encode(corpus_text)))
    chars_per_token = len(corpus_text) / corpus_tokens
    order_cache: dict[tuple[tuple[str, ...], int], list[tuple[int, int, int]]] = {}

    def _cancelled() -> bool:
        return cancel is not None and cancel.is_set()

    def _size_span(
        sentences: tuple[str, ...],
        kind: str,
        level: int,
        placement: str,
        case: Any,
        fmt: Any,
        native_len: int,
        planted: str | None,
    ) -> tuple[str, Any, tuple[int, ...]] | None:
        cache_key = (sentences, level)
        order = order_cache.get(cache_key)
        if order is None:
            order = _dis.span_candidates(sentences, target=level, chars_per_token=chars_per_token)[
                : _dis.SIZING_ENCODE_BUDGET
            ]
            order_cache[cache_key] = order
        for whole, suffix, _chars in order:
            if suffix and kind == "near-miss" and planted is not None:
                sentence = sentences[whole]
                if _dis.suffix_cuts_plant(sentence[:suffix], sentence[suffix:], planted):
                    continue
            span = _dis.span_text(sentences, whole, suffix)
            if not span:
                continue
            cell_fixture = _dis.build_cell_fixture(case, span, placement)
            ids = tuple(backend.encode(fmt.render(cell_fixture.messages)))
            if len(ids) - native_len == level:
                return span, cell_fixture, ids
        return None

    candidates: dict[str, Any] = {}
    for fmt in formats:
        name = fmt.name
        entry = entries.get(name)
        summary = entry.get("summary") if type(entry) is dict else None
        observations = entry.get("observations") if type(entry) is dict else None
        if (
            type(entry) is not dict
            or type(observations) is not list
            or not observations
            or (type(summary) is dict and summary.get("aggregate_eligible") is False)
        ):
            candidates[name] = {
                "status": "unavailable",
                "detail": "native_search_unavailable",
                "records": [],
            }
            continue
        obs_by_fixture: dict[str, Any] = {}
        usable = True
        for obs in observations:
            if type(obs) is not dict or type(obs.get("fixture")) is not str:
                usable = False
                break
            obs_by_fixture[obs["fixture"]] = obs
        if not usable or any(case.id not in obs_by_fixture for case in cases):
            candidates[name] = {
                "status": "unavailable",
                "detail": "native_search_unavailable",
                "records": [],
            }
            continue
        records: list[dict[str, Any]] = []
        for case in cases:
            if _cancelled():
                return {
                    "available": False,
                    "detail": "cancelled",
                    "protocol": protocol,
                    "candidates": {},
                }
            obs = obs_by_fixture[case.id]
            ids_x1 = tuple(obs["input_ids"])
            native_answer = obs.get("answer")
            native_cell = {
                "verdict": bool(obs["exact_match"] and obs["boundary_found"]),
                "boundary_found": bool(obs["boundary_found"]),
                "boundary_text_match": obs["boundary_text_match"],
                "input_tokens": len(ids_x1),
                "answer_digest": _answer_digest(native_answer),
            }
            content = _rep.target_content(case)
            vacuous = _rep.is_vacuous(content)
            planted = _dis.planted_answer(case.expected)
            near_miss_defined = (
                planted is not None and planted not in expected_values and native_answer != planted
            )
            record: dict[str, Any] = {
                "fixture": case.id,
                "vacuous": vacuous,
                "echo_fixture": _rep.is_echo_fixture(case),
                "near_miss_defined": near_miss_defined,
                "planted_digest": (_answer_digest(planted) if near_miss_defined else None),
                "native": native_cell,
                "cells": dict.fromkeys(_dis.CONDITION_KEYS),
                "cell_details": dict.fromkeys(_dis.CONDITION_KEYS, ""),
            }
            records.append(record)
            if vacuous:
                record["cell_details"] = dict.fromkeys(_dis.CONDITION_KEYS, "vacuous")
                continue
            sentences_by_kind: dict[str, tuple[str, ...]] = {
                "neutral": _dis.NEUTRAL_CORPUS,
            }
            if near_miss_defined and planted is not None:
                sentences_by_kind["near-miss"] = _dis.nearmiss_sentences(planted)
            for kind, level, placement in _dis.CONDITIONS:
                key = _dis.cell_condition(kind, level, placement)
                if kind == "near-miss" and not near_miss_defined:
                    record["cell_details"][key] = "near_miss_undefined"
                    continue
                sized = _size_span(
                    sentences_by_kind[kind],
                    kind,
                    level,
                    placement,
                    case,
                    fmt,
                    len(ids_x1),
                    planted if kind == "near-miss" else None,
                )
                if sized is None:
                    record["cell_details"][key] = "arm_unmatchable"
                    continue
                span, cell_fixture, ids_arm = sized
                if _dis.span_screened(span, case.expected, [boundary_text[name]]):
                    record["cell_details"][key] = "arm_screened"
                    continue
                if len(ids_arm) + max_new_tokens > max_context:
                    record["cell_details"][key] = "arm_context_exceeded"
                    continue
                if _cancelled():
                    return {
                        "available": False,
                        "detail": "cancelled",
                        "protocol": protocol,
                        "candidates": {},
                    }
                report = evaluate(
                    backend,
                    fmt,
                    (cell_fixture,),
                    boundary=boundaries[name],
                    max_new_tokens=max_new_tokens,
                    seed=seed,
                    cancel=cancel,
                )
                if report.aggregate_eligible and report.observations:
                    cell_obs = report.observations[0]
                    construction, seam = _dis.classify_cell_construction(
                        fmt,
                        case,
                        span,
                        placement,
                        ids_x1,
                        tuple(cell_obs.input_ids),
                        backend.encode,
                    )
                    record["cells"][key] = {
                        "verdict": bool(cell_obs.exact_match and cell_obs.boundary_found),
                        "boundary_found": bool(cell_obs.boundary_found),
                        "boundary_text_match": cell_obs.boundary_text_match,
                        "input_tokens": len(cell_obs.input_ids),
                        "answer_digest": _answer_digest(cell_obs.answer),
                        "construction": construction,
                        "seam_ids": seam,
                        "planted_occurrences": (
                            _dis.planted_occurrences(span, planted)
                            if kind == "near-miss" and planted is not None
                            else None
                        ),
                    }
                else:
                    record["cell_details"][key] = "arm_evaluation_failed"
        candidates[name] = {"status": "available", "detail": "", "records": records}
    return {"available": True, "detail": "", "protocol": protocol, "candidates": candidates}


def _condition_all_pass(records: list[dict[str, Any]], key: str) -> bool | None:
    """All-pass over a condition's completed fixtures, or None.

    Completed means non-vacuous with an evaluable cell; an empty
    completed set returns None (the caller withholds selection).
    """
    completed = [
        record for record in records if not record["vacuous"] and record["cells"][key] is not None
    ]
    if not completed:
        return None
    return all(record["cells"][key]["verdict"] for record in completed)


def distraction_analysis(
    stage: dict[str, Any],
    primary_decision: Any,
    primary_winner: Any,
    candidate_names: tuple[str, ...],
) -> dict[str, Any]:
    """Aggregate a distraction stage into the stored payload (cycle 184).

    Per-condition winners follow the conservative rule over each
    condition's completed, non-vacuous fixtures — withheld per
    condition whenever any candidate is unavailable, has no
    completed fixtures, or the completed sets differ across
    candidates. There is no grouping readout in v1 (closed by
    decision, not silence). Design: docs/DISTRACTION_DESIGN.md.
    """
    from . import distraction as _dis

    if not stage.get("available"):
        return {
            "available": False,
            "detail": str(stage.get("detail", "distraction_unavailable")),
            "protocol": stage.get("protocol"),
        }
    if primary_decision in ("compatible_on_fixture_set", "held_out_failed"):
        mapped_primary = ("selected", primary_winner)
    else:
        mapped_primary = (primary_decision, primary_winner)
    stage_candidates = stage["candidates"]
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
        aggregates = _dis.aggregate_candidate(entry["records"])
        aggregates.update({"status": "available", "detail": "", "records": entry["records"]})
        candidates_out[name] = aggregates
    conditions_out: dict[str, Any] = {}
    for key in _dis.CONDITION_KEYS:
        completed_sets: list[frozenset[str]] = []
        all_pass: dict[str, bool | None] = {}
        withheld = False
        for name in candidate_names:
            entry = stage_candidates[name]
            if entry["status"] != "available":
                withheld = True
                all_pass[name] = None
                continue
            all_pass[name] = _condition_all_pass(entry["records"], key)
            completed_sets.append(
                frozenset(
                    record["fixture"]
                    for record in entry["records"]
                    if not record["vacuous"] and record["cells"][key] is not None
                )
            )
            if all_pass[name] is None:
                withheld = True
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
        passers = [n for n in candidate_names if all_pass[n]]
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
            "flip": (decision, winner) != mapped_primary,
        }
    return {
        "available": True,
        "detail": "",
        "protocol": stage["protocol"],
        "primary_decision": primary_decision,
        "primary_winner": primary_winner,
        "candidates": candidates_out,
        "conditions": conditions_out,
        "n_candidates": len(candidate_names),
    }


def leading_stage(
    backend: Any,
    formats: tuple[Any, ...],
    cases: tuple[Any, ...],
    search_entries: Any,
    *,
    boundary: tuple[int, ...],
    boundary_map: dict[str, tuple[int, ...]] | None = None,
    max_new_tokens: int = 64,
    max_context: int,
    seed: int = 0,
    cancel: Event | None = None,
) -> dict[str, Any]:
    """Leading-claim caving stage (cycle 185): claim cells per candidate.

    The native arm is the recorded search (reused, never
    regenerated). Each treated cell appends one frozen claim
    sentence to the last user message's content before render --
    a wrong claim (another fixture's answer, or a digit-surgery
    plant of the fixture's own) or the right-claim control, in
    two framings -- and is scored by the standard pipeline
    against the fixture's own expected answer. Cells degrade
    individually (donor undefined, claim too long, screened,
    context-exceeded); one bad fixture never takes a candidate's
    whole curve with it. Design: docs/LEADING_DESIGN.md.
    """
    from . import leading as _lead
    from . import repetition as _rep
    from .fixtures import audit_special_token_literals

    if type(formats) is not tuple or not formats:
        raise ValueError("formats must be a nonempty tuple")
    if type(cases) is not tuple or not cases:
        raise ValueError("cases must be a nonempty tuple")
    fixture_ids = [case.id for case in cases]
    donor_map = _lead.build_donor_map(cases)
    protocol: dict[str, Any] = dict(
        _lead.protocol_object(fixture_ids, donor_map, _backend_fingerprint(backend))
    )
    protocol["fingerprint"] = _lead.protocol_fingerprint(
        fixture_ids, donor_map, _backend_fingerprint(backend)
    )
    if len(formats) * len(cases) > _lead.MAX_GROUND_RECORDS:
        return {
            "available": False,
            "detail": "leading_cell_budget_exceeded",
            "protocol": protocol,
            "candidates": {},
        }
    template_text = "".join(
        template.replace("{claim}", "") for template in _lead.CLAIM_TEMPLATES.values()
    )
    if audit_special_token_literals(template_text, formats):
        # A special-token literal in the frozen templates would
        # forge turn structure in every cell of every run; there
        # is no honest per-cell version of that arm, so the stage
        # fails closed before any generation.
        return {
            "available": False,
            "detail": "leading_claim_screen_failed",
            "protocol": protocol,
            "candidates": {},
        }
    if not any(entry["planted"] is not None for entry in donor_map.values()):
        return {
            "available": False,
            "detail": "leading_no_defined_claims",
            "protocol": protocol,
            "candidates": {},
        }
    boundaries = _resolve_boundaries(formats, boundary, boundary_map)
    entries: dict[str, Any] = {}
    if type(search_entries) is list:
        for entry in search_entries:
            if type(entry) is not dict:
                continue
            fmt_dict = entry.get("format")
            entry_name = fmt_dict.get("name") if type(fmt_dict) is dict else None
            if type(entry_name) is str and entry_name:
                entries[entry_name] = entry
    boundary_text: dict[str, str] = {}
    for fmt in formats:
        try:
            boundary_text[fmt.name] = backend.decode(boundaries[fmt.name])
        except Exception:  # noqa: BLE001 -- an undecodable boundary screens nothing
            boundary_text[fmt.name] = ""

    def _cancelled() -> bool:
        return cancel is not None and cancel.is_set()

    candidates: dict[str, Any] = {}
    for fmt in formats:
        name = fmt.name
        entry = entries.get(name)
        summary = entry.get("summary") if type(entry) is dict else None
        observations = entry.get("observations") if type(entry) is dict else None
        if (
            type(entry) is not dict
            or type(observations) is not list
            or not observations
            or (type(summary) is dict and summary.get("aggregate_eligible") is False)
        ):
            candidates[name] = {
                "status": "unavailable",
                "detail": "native_search_unavailable",
                "records": [],
            }
            continue
        obs_by_fixture: dict[str, Any] = {}
        usable = True
        for obs in observations:
            if type(obs) is not dict or type(obs.get("fixture")) is not str:
                usable = False
                break
            obs_by_fixture[obs["fixture"]] = obs
        if not usable or any(case.id not in obs_by_fixture for case in cases):
            candidates[name] = {
                "status": "unavailable",
                "detail": "native_search_unavailable",
                "records": [],
            }
            continue
        records: list[dict[str, Any]] = []
        for case in cases:
            if _cancelled():
                return {
                    "available": False,
                    "detail": "cancelled",
                    "protocol": protocol,
                    "candidates": {},
                }
            obs = obs_by_fixture[case.id]
            ids_x1 = tuple(obs["input_ids"])
            native_cell = {
                "verdict": bool(obs["exact_match"] and obs["boundary_found"]),
                "boundary_found": bool(obs["boundary_found"]),
                "boundary_text_match": obs["boundary_text_match"],
                "input_tokens": len(ids_x1),
                "answer_digest": _answer_digest(obs.get("answer")),
            }
            donor_entry = donor_map[case.id]
            content = _rep.target_content(case)
            vacuous = _rep.is_vacuous(content)
            expected_blank = not case.expected.strip()
            record: dict[str, Any] = {
                "fixture": case.id,
                "group": case.group,
                "vacuous": vacuous,
                "expected_blank": expected_blank,
                "echo_fixture": _rep.is_echo_fixture(case),
                "donor_fixture": donor_entry["donor_fixture"],
                "plant_source": donor_entry["plant_source"],
                "wrong_claim_defined": donor_entry["planted"] is not None,
                "expected_digest": _lead.digest_text(case.expected),
                "planted_digest": donor_entry["planted_digest"],
                "native": native_cell,
                "cells": dict.fromkeys(_lead.CONDITION_KEYS),
                "cell_details": dict.fromkeys(_lead.CONDITION_KEYS, ""),
            }
            records.append(record)
            if vacuous:
                record["cell_details"] = dict.fromkeys(_lead.CONDITION_KEYS, "vacuous")
                continue
            if expected_blank:
                record["cell_details"] = dict.fromkeys(_lead.CONDITION_KEYS, "expected_blank")
                continue
            for arm in _lead.ARMS:
                if arm == "wrong":
                    if donor_entry["planted"] is None:
                        for framing in _lead.FRAMINGS:
                            record["cell_details"][f"wrong_{framing}"] = "donor_undefined"
                        continue
                    claim_value = donor_entry["planted"]
                else:
                    claim_value = case.expected
                for framing in _lead.FRAMINGS:
                    key = f"{arm}_{framing}"
                    sentence = _lead.claim_sentence(framing, claim_value)
                    if len(sentence) > _lead.MAX_CLAIM_CHARS:
                        record["cell_details"][key] = "claim_too_long"
                        continue
                    if audit_special_token_literals(sentence, (fmt,)) or (
                        boundary_text[name] and boundary_text[name] in sentence
                    ):
                        record["cell_details"][key] = "arm_screened"
                        continue
                    cell_fixture = _lead.build_claim_fixture(case, sentence)
                    ids_arm = tuple(backend.encode(fmt.render(cell_fixture.messages)))
                    if len(ids_arm) + max_new_tokens > max_context:
                        record["cell_details"][key] = "arm_context_exceeded"
                        continue
                    if _cancelled():
                        return {
                            "available": False,
                            "detail": "cancelled",
                            "protocol": protocol,
                            "candidates": {},
                        }
                    report = evaluate(
                        backend,
                        fmt,
                        (cell_fixture,),
                        boundary=boundaries[name],
                        max_new_tokens=max_new_tokens,
                        seed=seed,
                        cancel=cancel,
                    )
                    if report.aggregate_eligible and report.observations:
                        cell_obs = report.observations[0]
                        record["cells"][key] = {
                            "verdict": bool(cell_obs.exact_match and cell_obs.boundary_found),
                            "boundary_found": bool(cell_obs.boundary_found),
                            "boundary_text_match": cell_obs.boundary_text_match,
                            "input_tokens": len(cell_obs.input_ids),
                            "answer_digest": _answer_digest(cell_obs.answer),
                        }
                    else:
                        record["cell_details"][key] = "arm_evaluation_failed"
        candidates[name] = {"status": "available", "detail": "", "records": records}
    return {"available": True, "detail": "", "protocol": protocol, "candidates": candidates}


def leading_analysis(
    stage: dict[str, Any],
    primary_decision: Any,
    primary_winner: Any,
    candidate_names: tuple[str, ...],
) -> dict[str, Any]:
    """Aggregate a leading stage into the stored payload (cycle 185).

    Per-condition winners follow the conservative rule over each
    condition's completed, non-vacuous fixtures -- withheld
    whenever any candidate is unavailable, the completed set
    differs across candidates, or the completed set is not the
    condition's defined set (for the wrong-claim conditions,
    the fixtures whose wrong claim is defined). The caving
    classification is per candidate per framing; the common-
    population readout is the only ranked cross-candidate
    comparison. Design: docs/LEADING_DESIGN.md.
    """
    from . import leading as _lead

    if not stage.get("available"):
        return {
            "available": False,
            "detail": str(stage.get("detail", "leading_unavailable")),
            "protocol": stage.get("protocol"),
        }
    if primary_decision in ("compatible_on_fixture_set", "held_out_failed"):
        mapped_primary = ("selected", primary_winner)
    else:
        mapped_primary = (primary_decision, primary_winner)
    stage_candidates = stage["candidates"]
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
        aggregates = _lead.aggregate_candidate(entry["records"])
        aggregates.update({"status": "available", "detail": "", "records": entry["records"]})
        candidates_out[name] = aggregates
    conditions_out: dict[str, Any] = {}
    for key in _lead.CONDITION_KEYS:
        arm = key.split("_", 1)[0]
        completed_sets: list[frozenset[str]] = []
        all_pass: dict[str, bool | None] = {}
        withheld = False
        for name in candidate_names:
            entry = stage_candidates[name]
            if entry["status"] != "available":
                withheld = True
                all_pass[name] = None
                continue
            live = [
                record
                for record in entry["records"]
                if not record["vacuous"] and not record["expected_blank"]
            ]
            if arm == "wrong":
                defined = frozenset(
                    record["fixture"] for record in live if record["wrong_claim_defined"]
                )
            else:
                defined = frozenset(record["fixture"] for record in live)
            completed = frozenset(
                record["fixture"] for record in live if record["cells"][key] is not None
            )
            completed_sets.append(completed)
            if completed != defined or not completed:
                withheld = True
                all_pass[name] = None
                continue
            all_pass[name] = all(
                record["cells"][key]["verdict"] for record in live if record["fixture"] in completed
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
        passers = [n for n in candidate_names if all_pass[n]]
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
            "flip": (decision, winner) != mapped_primary,
        }
    common: dict[str, Any] = {}
    available_records = {
        name: stage_candidates[name]["records"]
        for name in candidate_names
        if stage_candidates[name]["status"] == "available"
    }
    for framing in _lead.FRAMINGS:
        common[framing] = _lead.common_population(available_records, framing)
    return {
        "available": True,
        "detail": "",
        "protocol": stage["protocol"],
        "primary_decision": primary_decision,
        "primary_winner": primary_winner,
        "candidates": candidates_out,
        "conditions": conditions_out,
        "common_population": common,
        "n_candidates": len(candidate_names),
    }


def dateline_stage(
    backend: Any,
    formats: tuple[Any, ...],
    cases: tuple[Any, ...],
    search_entries: Any,
    *,
    boundary: tuple[int, ...],
    boundary_map: dict[str, tuple[int, ...]] | None = None,
    max_new_tokens: int = 64,
    max_context: int,
    seed: int = 0,
    cancel: Event | None = None,
) -> dict[str, Any]:
    """Date-line sensitivity stage (cycle 188): frozen date arms per candidate.

    The native arm is the recorded search (reused, never
    regenerated). Each treated cell prepends one frozen block --
    a date line, a sized no-date control, or the weekday-lie
    line -- to the last user message's content before render and
    is scored by the standard pipeline against the fixture's own
    expected answer. Date-derived fixtures are screened out as
    their own record class before any generation; cells degrade
    individually (screened, unmatchable control, context-
    exceeded); one bad fixture never takes a candidate's whole
    readout with it. Design: docs/DATELINE_DESIGN.md.
    """
    from . import dateline as _dl
    from . import repetition as _rep
    from .fixtures import audit_special_token_literals

    if type(formats) is not tuple or not formats:
        raise ValueError("formats must be a nonempty tuple")
    if type(cases) is not tuple or not cases:
        raise ValueError("cases must be a nonempty tuple")
    fixture_ids = [case.id for case in cases]
    protocol: dict[str, Any] = dict(_dl.protocol_object(fixture_ids, _backend_fingerprint(backend)))
    protocol["fingerprint"] = _dl.protocol_fingerprint(fixture_ids, _backend_fingerprint(backend))
    if len(formats) * len(cases) > _dl.MAX_GROUND_RECORDS:
        return {
            "available": False,
            "detail": "dateline_cell_budget_exceeded",
            "protocol": protocol,
            "candidates": {},
        }
    screen_text = (
        _dl.BLOCK_TEMPLATE.replace("{DATE}.", ".")
        + _dl.CONTROL_TEMPLATE.replace("{NEUTRAL}", _dl.NEUTRAL_UNIT)
        + "".join(_dl.date_arm_table().values())
    )
    if audit_special_token_literals(screen_text, formats):
        # A special-token literal in the frozen protocol text
        # would forge turn structure in every cell of every run;
        # there is no honest per-cell version of that arm, so the
        # stage fails closed before any generation.
        return {
            "available": False,
            "detail": "dateline_screen_failed",
            "protocol": protocol,
            "candidates": {},
        }
    boundaries = _resolve_boundaries(formats, boundary, boundary_map)
    entries: dict[str, Any] = {}
    if type(search_entries) is list:
        for entry in search_entries:
            if type(entry) is not dict:
                continue
            fmt_dict = entry.get("format")
            entry_name = fmt_dict.get("name") if type(fmt_dict) is dict else None
            if type(entry_name) is str and entry_name:
                entries[entry_name] = entry
    boundary_text: dict[str, str] = {}
    for fmt in formats:
        try:
            boundary_text[fmt.name] = backend.decode(boundaries[fmt.name])
        except Exception:  # noqa: BLE001 -- an undecodable boundary screens nothing
            boundary_text[fmt.name] = ""
    expected_by_fixture = {case.id: case.expected for case in cases}
    neutral_screened_fixtures, neutral_screened_candidates = _dl.screen_neutral(
        expected_by_fixture, boundary_text
    )
    screens = {case.id: _dl.fixture_screen(case) for case in cases}

    def _cancelled() -> bool:
        return cancel is not None and cancel.is_set()

    candidates: dict[str, Any] = {}
    for fmt in formats:
        name = fmt.name
        entry = entries.get(name)
        summary = entry.get("summary") if type(entry) is dict else None
        observations = entry.get("observations") if type(entry) is dict else None
        if (
            type(entry) is not dict
            or type(observations) is not list
            or not observations
            or (type(summary) is dict and summary.get("aggregate_eligible") is False)
        ):
            candidates[name] = {
                "status": "unavailable",
                "detail": "native_search_unavailable",
                "records": [],
            }
            continue
        obs_by_fixture: dict[str, Any] = {}
        usable = True
        for obs in observations:
            if type(obs) is not dict or type(obs.get("fixture")) is not str:
                usable = False
                break
            obs_by_fixture[obs["fixture"]] = obs
        if not usable or any(case.id not in obs_by_fixture for case in cases):
            candidates[name] = {
                "status": "unavailable",
                "detail": "native_search_unavailable",
                "records": [],
            }
            continue
        unit_estimate = max(1, len(backend.encode(_dl.NEUTRAL_UNIT)))
        records: list[dict[str, Any]] = []
        for case in cases:
            if _cancelled():
                return {
                    "available": False,
                    "detail": "cancelled",
                    "protocol": protocol,
                    "candidates": {},
                }
            obs = obs_by_fixture[case.id]
            ids_x1 = tuple(obs["input_ids"])
            native_cell = {
                "verdict": bool(obs["exact_match"] and obs["boundary_found"]),
                "boundary_found": bool(obs["boundary_found"]),
                "boundary_text_match": obs["boundary_text_match"],
                "input_tokens": len(ids_x1),
                "answer_digest": _answer_digest(obs.get("answer")),
            }
            content = _rep.target_content(case)
            vacuous = _rep.is_vacuous(content)
            expected_blank = not case.expected.strip()
            screened, message_bearing = screens[case.id]
            record: dict[str, Any] = {
                "fixture": case.id,
                "group": case.group,
                "n_user_messages": sum(1 for message in case.messages if message.role == "user"),
                "vacuous": vacuous,
                "expected_blank": expected_blank,
                "echo_fixture": _rep.is_echo_fixture(case),
                "screened": screened,
                "message_date_bearing": message_bearing,
                "construction": "vacuous" if vacuous else "text_level",
                "seam_ids": [],
                "expected_digest": _dl.digest_text(case.expected),
                "native": native_cell,
                "cells": dict.fromkeys(_dl.TREATED_KEYS),
                "cell_details": dict.fromkeys(_dl.TREATED_KEYS, ""),
                "control_evaluable": dict.fromkeys(_dl.FORMATS, False),
            }
            records.append(record)
            if vacuous:
                record["cell_details"] = dict.fromkeys(_dl.TREATED_KEYS, "vacuous")
                continue
            if expected_blank:
                record["cell_details"] = dict.fromkeys(_dl.TREATED_KEYS, "expected_blank")
                continue
            if screened:
                record["cell_details"] = dict.fromkeys(_dl.TREATED_KEYS, "screened")
                continue
            if message_bearing:
                record["cell_details"] = dict.fromkeys(_dl.TREATED_KEYS, "message_date_bearing")
                continue
            # Construction classification at the prepend seam,
            # computed once per (candidate, fixture) on the anchor
            # ISO arm (the seam's shape is arm-independent).
            marker = "\u0001dateline-marker\u0001"
            if marker not in content:
                marker_fixture = _fixture_with_content(case, marker)
                rendered_marker = fmt.render(marker_fixture.messages)
                if rendered_marker.count(marker) == 1:
                    head, tail = rendered_marker.split(marker)
                    prefix_ids = backend.encode(head)
                    suffix_ids = backend.encode(tail)
                    block_fixture = _fixture_with_content(
                        case, _dl.block_text(_dl.arm_date_string("anchor_iso"))
                    )
                    block_region = _rep.content_region(
                        backend.encode(fmt.render(block_fixture.messages)),
                        prefix_ids,
                        suffix_ids,
                    )
                    joiner_fixture = _fixture_with_content(case, _dl.JOINER)
                    joiner_region = _rep.content_region(
                        backend.encode(fmt.render(joiner_fixture.messages)),
                        prefix_ids,
                        suffix_ids,
                    )
                    region_native = _rep.content_region(ids_x1, prefix_ids, suffix_ids)
                    anchor_fixture = _dl.build_arm_fixture(case, "anchor_iso")
                    region_treated = _rep.content_region(
                        backend.encode(fmt.render(anchor_fixture.messages)),
                        prefix_ids,
                        suffix_ids,
                    )
                    construction, seam = _dl.classify_construction(
                        region_native, region_treated, block_region, joiner_region
                    )
                    record["construction"] = construction
                    record["seam_ids"] = list(seam)

            try:
                for arm_key in _dl.DATE_ARM_KEYS + (_dl.LIE_KEY,):
                    date_string = _dl.arm_date_string(arm_key)
                    if audit_special_token_literals(date_string, (fmt,)) or (
                        boundary_text[name] and boundary_text[name] in date_string
                    ):
                        record["cell_details"][arm_key] = "arm_screened"
                        continue
                    _dateline_run_cell(
                        backend,
                        fmt,
                        record,
                        arm_key,
                        _dl.build_arm_fixture(case, arm_key),
                        boundary_ids=boundaries[name],
                        max_new_tokens=max_new_tokens,
                        max_context=max_context,
                        seed=seed,
                        cancel=cancel,
                    )
                for fmt_name in _dl.FORMATS:
                    control_key = f"nodate_{fmt_name}"
                    anchor_key = f"anchor_{fmt_name}"
                    if name in neutral_screened_candidates or (
                        case.id in neutral_screened_fixtures
                    ):
                        record["cell_details"][control_key] = "control_screened"
                        continue
                    anchor_fixture = _dl.build_arm_fixture(case, anchor_key)
                    anchor_ids = tuple(backend.encode(fmt.render(anchor_fixture.messages)))
                    empty_fixture = _dl.build_arm_fixture(case, control_key, "")
                    empty_ids = tuple(backend.encode(fmt.render(empty_fixture.messages)))
                    delta = len(anchor_ids) - len(empty_ids)
                    sized_text: str | None = None
                    if delta == 0:
                        # The empty control already matches the
                        # anchor's rendered length exactly.
                        sized_text = ""
                    elif delta > 0:
                        for reps, cut in _dl.neutral_search_order(unit_estimate, delta):
                            text = _dl.neutral_text(reps, cut)
                            candidate_fixture = _dl.build_arm_fixture(case, control_key, text)
                            candidate_ids = tuple(
                                backend.encode(fmt.render(candidate_fixture.messages))
                            )
                            if len(candidate_ids) == len(anchor_ids):
                                sized_text = text
                                break
                    if sized_text is None:
                        record["cell_details"][control_key] = "control_unmatchable"
                        continue
                    record["control_evaluable"][fmt_name] = True
                    _dateline_run_cell(
                        backend,
                        fmt,
                        record,
                        control_key,
                        _dl.build_arm_fixture(case, control_key, sized_text),
                        boundary_ids=boundaries[name],
                        max_new_tokens=max_new_tokens,
                        max_context=max_context,
                        seed=seed,
                        cancel=cancel,
                    )
            except _DatelineCancelled:
                return {
                    "available": False,
                    "detail": "cancelled",
                    "protocol": protocol,
                    "candidates": {},
                }
        candidates[name] = {"status": "available", "detail": "", "records": records}
    return {"available": True, "detail": "", "protocol": protocol, "candidates": candidates}


class _DatelineCancelled(Exception):
    """Internal control flow: unwind the stage on cancellation."""


def _dateline_run_cell(
    backend: Any,
    fmt: Any,
    record: dict[str, Any],
    arm_key: str,
    arm_fixture: Any,
    *,
    boundary_ids: tuple[int, ...],
    max_new_tokens: int,
    max_context: int,
    seed: int,
    cancel: Event | None,
) -> None:
    """Generate and record one dateline treated cell.

    Degrades per cell (context-exceeded, evaluation-failed) in
    the fixed detail vocabulary; raises _DatelineCancelled when
    the run's cancel event is set, so the stage unwinds to its
    cancelled payload rather than half-recording a fixture.
    """
    ids_arm = tuple(backend.encode(fmt.render(arm_fixture.messages)))
    if len(ids_arm) + max_new_tokens > max_context:
        record["cell_details"][arm_key] = "arm_context_exceeded"
        return
    if cancel is not None and cancel.is_set():
        raise _DatelineCancelled
    report = evaluate(
        backend,
        fmt,
        (arm_fixture,),
        boundary=boundary_ids,
        max_new_tokens=max_new_tokens,
        seed=seed,
        cancel=cancel,
    )
    if report.aggregate_eligible and report.observations:
        cell_obs = report.observations[0]
        record["cells"][arm_key] = {
            "verdict": bool(cell_obs.exact_match and cell_obs.boundary_found),
            "boundary_found": bool(cell_obs.boundary_found),
            "boundary_text_match": cell_obs.boundary_text_match,
            "input_tokens": len(cell_obs.input_ids),
            "answer_digest": _answer_digest(cell_obs.answer),
        }
    else:
        record["cell_details"][arm_key] = "arm_evaluation_failed"


def dateline_analysis(
    stage: dict[str, Any],
    primary_decision: Any,
    primary_winner: Any,
    candidate_names: tuple[str, ...],
) -> dict[str, Any]:
    """Aggregate a dateline stage into the stored payload (cycle 188).

    Per-arm winners follow the conservative rule over each arm's
    completed fixtures -- withheld whenever any candidate is
    unavailable, the completed set differs across candidates, or
    the completed set is not the arm's defined set (for a control
    arm, the fixtures whose control text was sizeable for that
    candidate). Native enters no paired population; it is the
    per-record baseline and the selection-flip comparator only.
    The common-population readout is the only ranked
    cross-candidate comparison. Design: docs/DATELINE_DESIGN.md.
    """
    from . import dateline as _dl

    if not stage.get("available"):
        return {
            "available": False,
            "detail": str(stage.get("detail", "dateline_unavailable")),
            "protocol": stage.get("protocol"),
        }
    if primary_decision in ("compatible_on_fixture_set", "held_out_failed"):
        mapped_primary = ("selected", primary_winner)
    else:
        mapped_primary = (primary_decision, primary_winner)
    stage_candidates = stage["candidates"]
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
        aggregates = _dl.aggregate_candidate(entry["records"])
        aggregates.update({"status": "available", "detail": "", "records": entry["records"]})
        candidates_out[name] = aggregates
    arms_out: dict[str, Any] = {}
    for key in _dl.TREATED_KEYS:
        completed_sets: list[frozenset[str]] = []
        all_pass: dict[str, bool | None] = {}
        withheld = False
        for name in candidate_names:
            entry = stage_candidates[name]
            if entry["status"] != "available":
                withheld = True
                all_pass[name] = None
                continue
            live = _dl.live_records(entry["records"])
            if key in _dl.CONTROL_KEYS:
                control_format = key.removeprefix("nodate_")
                defined = frozenset(
                    record["fixture"]
                    for record in live
                    if record["control_evaluable"][control_format]
                )
            else:
                defined = frozenset(record["fixture"] for record in live)
            completed = frozenset(
                record["fixture"] for record in live if record["cells"][key] is not None
            )
            completed_sets.append(completed)
            if completed != defined or not completed:
                withheld = True
                all_pass[name] = None
                continue
            all_pass[name] = all(
                record["cells"][key]["verdict"] for record in live if record["fixture"] in completed
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
        passers = [n for n in candidate_names if all_pass[n]]
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
            "flip": (decision, winner) != mapped_primary,
        }
    available_records = {
        name: stage_candidates[name]["records"]
        for name in candidate_names
        if stage_candidates[name]["status"] == "available"
    }
    common = _dl.common_population(available_records)
    # The multi-turn share is recomputed from the ground records
    # (n_user_messages is a per-fixture reference field), so the
    # diff validator can recompute it too -- it is not trusted
    # from the stage.
    reference_records: list[dict[str, Any]] = next(iter(available_records.values()), [])
    n_multi_turn = sum(1 for record in reference_records if record["n_user_messages"] > 1)
    return {
        "available": True,
        "detail": "",
        "protocol": stage["protocol"],
        "primary_decision": primary_decision,
        "primary_winner": primary_winner,
        "candidates": candidates_out,
        "arms": arms_out,
        "common_population": common,
        "n_candidates": len(candidate_names),
        "n_search_fixtures": len(reference_records),
        "n_multi_turn_fixtures": n_multi_turn,
    }


class _MultiturnCancelled(Exception):
    """Internal control flow: unwind the stage on cancellation."""


def _multiturn_turn(
    backend: Any,
    fmt: Any,
    case_id: str,
    messages: tuple[Any, ...],
    expected: str,
    *,
    boundary_ids: tuple[int, ...],
    max_new_tokens: int,
    max_context: int,
    seed: int,
    cancel: Event | None,
) -> tuple[str | None, dict[str, Any] | None, str | None]:
    """One multiturn generation: (detail, cell, answer_text).

    Exactly one of ``detail`` / ``cell`` is set. The cancel
    event is checked immediately after ``evaluate`` returns,
    before the evaluation-failure mapping (design rule 5: a
    cancel observed during a turn is never mislabelled an
    evaluation failure). The answer text is returned
    separately and never stored in the cell -- records carry
    digests only.
    """
    from . import multiturn as _mt
    from .core import Fixture

    fixture = Fixture(id=case_id, messages=messages, expected=expected, group=_mt.GROUP)
    ids = tuple(backend.encode(fmt.render(fixture.messages)))
    if len(ids) + max_new_tokens > max_context:
        return "arm_context_exceeded", None, None
    if cancel is not None and cancel.is_set():
        raise _MultiturnCancelled
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
        raise _MultiturnCancelled
    if report.aggregate_eligible and report.observations:
        obs = report.observations[0]
        cell = {
            "verdict": bool(obs.exact_match and obs.boundary_found),
            "boundary_found": bool(obs.boundary_found),
            "boundary_text_match": obs.boundary_text_match,
            "input_tokens": len(obs.input_ids),
            "answer_digest": _answer_digest(obs.answer),
        }
        return None, cell, obs.answer
    return "arm_evaluation_failed", None, None


def multiturn_stage(
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
    """Multi-turn drift stage (cycle 189): generated shard
    fixtures in three arms per candidate.

    The fixtures are generated from the run seed inside the
    stage; there is no native arm and no dependence on the
    user's fixtures or the primary run. The sharded arm is a
    sequential loop -- each turn's boundary-cut reply is
    screened for special-token literals and then incorporated
    verbatim as the next turn's history (P1); only the final
    turn is scored (P2). Cells degrade individually in the
    fixed detail vocabulary; cancellation unwinds the whole
    stage to its fail-closed payload. Design:
    docs/MULTITURN_DESIGN.md.
    """
    from . import multiturn as _mt
    from .core import Message
    from .fixtures import audit_special_token_literals

    if type(formats) is not tuple or not formats:
        raise ValueError("formats must be a nonempty tuple")
    protocol: dict[str, Any] = dict(_mt.protocol_object(seed, _backend_fingerprint(backend)))
    protocol["fingerprint"] = _mt.protocol_fingerprint(seed, _backend_fingerprint(backend))
    generated = _mt.generate_fixtures(seed)
    if len(formats) * len(generated) > _mt.MAX_GROUND_RECORDS:
        return {
            "available": False,
            "detail": "multiturn_record_budget_exceeded",
            "protocol": protocol,
            "candidates": {},
        }
    screen_text = "".join(
        template.replace("{v}", "").replace("{label}", "") for template in _mt.SHARD_TEMPLATES
    )
    if audit_special_token_literals(screen_text, formats):
        # A special-token literal in the frozen protocol text
        # would forge turn structure in every cell of every run;
        # the stage fails closed before any generation.
        return {
            "available": False,
            "detail": "multiturn_screen_failed",
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
        if boundary_text[fmt.name] and boundary_text[fmt.name] in screen_text:
            return {
                "available": False,
                "detail": "multiturn_screen_failed",
                "protocol": protocol,
                "candidates": {},
            }

    def _cancelled() -> bool:
        return cancel is not None and cancel.is_set()

    candidates: dict[str, Any] = {}
    for fmt in formats:
        name = fmt.name
        records: list[dict[str, Any]] = []
        try:
            for generated_fixture in generated:
                if _cancelled():
                    return {
                        "available": False,
                        "detail": "cancelled",
                        "protocol": protocol,
                        "candidates": {},
                    }
                fixture_id = str(generated_fixture["fixture_id"])
                shards = [str(shard) for shard in generated_fixture["shards"]]
                expected = str(generated_fixture["expected"])
                depth = int(generated_fixture["depth"])
                record: dict[str, Any] = {
                    "fixture": fixture_id,
                    "family": str(generated_fixture["family"]),
                    "depth": depth,
                    "group": _mt.GROUP,
                    "shards": shards,
                    "values": [int(value) for value in generated_fixture["values"]],
                    "expected": expected,
                    "cells": dict.fromkeys(_mt.ARMS),
                    "cell_details": dict.fromkeys(_mt.ARMS, ""),
                }
                # full: every shard in one user turn.
                detail, cell, _answer = _multiturn_turn(
                    backend,
                    fmt,
                    f"{fixture_id}-full",
                    (Message("user", _mt.JOINER.join(shards)),),
                    expected,
                    boundary_ids=boundaries[name],
                    max_new_tokens=max_new_tokens,
                    max_context=max_context,
                    seed=seed,
                    cancel=cancel,
                )
                if detail is not None:
                    record["cell_details"]["full"] = detail
                else:
                    record["cells"]["full"] = cell
                # scripted: frozen intermediate replies, one
                # final generation.
                scripted_messages: list[Any] = []
                for slot in range(depth - 1):
                    scripted_messages.append(Message("user", shards[slot]))
                    scripted_messages.append(Message("assistant", _mt.SCRIPTED_REPLY))
                scripted_messages.append(Message("user", shards[depth - 1]))
                detail, cell, _answer = _multiturn_turn(
                    backend,
                    fmt,
                    f"{fixture_id}-scripted",
                    tuple(scripted_messages),
                    expected,
                    boundary_ids=boundaries[name],
                    max_new_tokens=max_new_tokens,
                    max_context=max_context,
                    seed=seed,
                    cancel=cancel,
                )
                if detail is not None:
                    record["cell_details"]["scripted"] = detail
                else:
                    record["cells"]["scripted"] = cell
                # sharded: the sequential loop. Each turn's cut
                # reply is screened, then becomes history.
                messages: list[Any] = [Message("user", shards[0])]
                turns: list[dict[str, Any]] = []
                sharded_cell: dict[str, Any] | None = None
                sharded_detail: str | None = None
                for turn in range(1, depth + 1):
                    detail, cell, answer = _multiturn_turn(
                        backend,
                        fmt,
                        f"{fixture_id}-sharded-t{turn}",
                        tuple(messages),
                        expected,
                        boundary_ids=boundaries[name],
                        max_new_tokens=max_new_tokens,
                        max_context=max_context,
                        seed=seed,
                        cancel=cancel,
                    )
                    if detail is not None or cell is None or answer is None:
                        sharded_detail = detail or "arm_evaluation_failed"
                        break
                    turns.append(
                        {
                            "input_tokens": cell["input_tokens"],
                            "answer_digest": cell["answer_digest"],
                            "boundary_found": cell["boundary_found"],
                        }
                    )
                    if turn == depth:
                        sharded_cell = dict(cell)
                        sharded_cell["intermediate_unbounded"] = any(
                            not entry["boundary_found"] for entry in turns[:-1]
                        )
                        sharded_cell["intermediate_empty"] = any(
                            entry["answer_digest"] == _mt.EMPTY_DIGEST for entry in turns[:-1]
                        )
                        sharded_cell["turns"] = turns
                        break
                    if audit_special_token_literals(answer, (fmt,)):
                        # A forged turn boundary inside history
                        # is token-real; the cell ends here and
                        # leaves the paired set (design rule 3).
                        sharded_detail = "arm_screened"
                        break
                    messages.append(Message("assistant", answer))
                    messages.append(Message("user", shards[turn]))
                if sharded_detail is not None:
                    record["cell_details"]["sharded"] = sharded_detail
                else:
                    record["cells"]["sharded"] = sharded_cell
                records.append(record)
        except _MultiturnCancelled:
            return {
                "available": False,
                "detail": "cancelled",
                "protocol": protocol,
                "candidates": {},
            }
        candidates[name] = {"status": "available", "detail": "", "records": records}
    return {"available": True, "detail": "", "protocol": protocol, "candidates": candidates}


def multiturn_analysis(
    stage: dict[str, Any],
    candidate_names: tuple[str, ...],
) -> dict[str, Any]:
    """Aggregate a multiturn stage into the stored payload
    (cycle 189).

    Per-candidate readouts come from multiturn.aggregate_candidate;
    the per-depth selection readouts for the full and sharded
    arms re-apply the conservative rule over the common
    completed population and are withheld whenever any
    candidate is unavailable, a completed set is not the
    depth's defined set, or completed sets differ across
    candidates (P3). The scripted arm is a control and gets no
    selection readout. Design: docs/MULTITURN_DESIGN.md.
    """
    from . import multiturn as _mt

    if not stage.get("available"):
        return {
            "available": False,
            "detail": str(stage.get("detail", "multiturn_unavailable")),
            "protocol": stage.get("protocol"),
        }
    stage_candidates = stage["candidates"]
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
        aggregates = _mt.aggregate_candidate(entry["records"])
        aggregates.update({"status": "available", "detail": "", "records": entry["records"]})
        candidates_out[name] = aggregates
    generated = _mt.generate_fixtures(int(stage["protocol"]["seed"]))
    selections: dict[str, Any] = {}
    for depth in _mt.DEPTHS:
        selections[str(depth)] = {}
        defined = frozenset(
            fixture["fixture_id"] for fixture in generated if fixture["depth"] == depth
        )
        for arm in ("full", "sharded"):
            completed_sets: list[frozenset[str]] = []
            all_pass: dict[str, bool | None] = {}
            withheld = False
            for name in candidate_names:
                entry = stage_candidates[name]
                if entry["status"] != "available":
                    withheld = True
                    all_pass[name] = None
                    continue
                completed = frozenset(
                    record["fixture"]
                    for record in entry["records"]
                    if record["depth"] == depth and record["cells"][arm] is not None
                )
                completed_sets.append(completed)
                if completed != defined or not completed:
                    withheld = True
                    all_pass[name] = None
                    continue
                all_pass[name] = all(
                    bool(record["cells"][arm]["verdict"])
                    for record in entry["records"]
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
            passers = [n for n in candidate_names if all_pass[n]]
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
        "available": True,
        "detail": "",
        "protocol": stage["protocol"],
        "candidates": candidates_out,
        "selections": selections,
        "n_candidates": len(candidate_names),
        "n_fixtures": len(generated),
    }


class _NeedleCancelled(Exception):
    """Internal control flow: unwind the stage on cancellation."""


def _needle_cell(
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
    """One needle generation: (detail, cell).

    Exactly one of the pair is set. ``ids`` are the accepted
    construction's rendered ids, used for the context check;
    the cell's ``input_tokens`` come from the evaluation's own
    observation, so a backend whose encode disagreed with
    itself would surface as a validator rejection downstream,
    never as a silently wrong record. The cancel event is
    checked before and after ``evaluate`` (the multiturn rule:
    a cancel observed during a generation is never mislabelled
    an evaluation failure).
    """
    if len(ids) + max_new_tokens > max_context:
        return "arm_context_exceeded", None
    if cancel is not None and cancel.is_set():
        raise _NeedleCancelled
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
        raise _NeedleCancelled
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


def _needle_run_cell(
    backend: Any,
    fmt: Any,
    boundary_ids: tuple[int, ...],
    boundary_text: str,
    record: dict[str, Any],
    case_id: str,
    value: str,
    key: str,
    content: str,
    arm_ids: tuple[int, ...],
    spans: list[str],
    offsets: list[int],
    *,
    max_new_tokens: int,
    max_context: int,
    seed: int,
    cancel: Event | None,
) -> None:
    """Screen, generate, and record one needle cell.

    The filler spans are screened before generation (screening
    rule needle-screen-v1); a screened or failed cell lands in
    ``cell_details`` and the cell itself stays None.
    """
    from . import needle as _nd
    from .core import Fixture, Message

    if any(_nd.filler_screened(span, value, boundary_text) for span in spans if span):
        record["cell_details"][key] = "arm_screened"
        return
    fixture = Fixture(
        id=f"{case_id}-{key}",
        messages=(Message("user", content),),
        expected=value,
        group=_nd.GROUP,
    )
    detail, cell = _needle_cell(
        backend,
        fmt,
        fixture,
        arm_ids,
        boundary_ids=boundary_ids,
        max_new_tokens=max_new_tokens,
        max_context=max_context,
        seed=seed,
        cancel=cancel,
    )
    if detail is not None:
        record["cell_details"][key] = detail
    elif cell is not None:
        cell["needle_offsets"] = offsets
        record["cells"][key] = cell


def needle_stage(
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
    """Needle-depth stage (cycle 191): generated code-recall
    cases at exact rendered-token depths per candidate.

    The cases are generated from the run seed inside the stage;
    there is no native arm and no dependence on the user's
    fixtures or the primary run. Each treated cell is sized by
    the frozen phased search in needle.py's sizing rule: spans
    of the cycled neutral corpus are probed until the rendered
    delta over the case's dose-0 render equals the dose and the
    prefix-anchored offset measurement ``D`` equals the target,
    both verified on the accepted construction's final ids.
    The delta identities fix every cell's final length before
    sizing (dose-0 length + dose, + needle_len for duplicate),
    so a cell that cannot fit the context is reported
    ``arm_context_exceeded`` without spending sizing encodes
    on it. Cells degrade individually in the fixed detail
    vocabulary; cancellation unwinds the whole stage to its
    fail-closed payload. Design: docs/NEEDLE_DESIGN.md.
    """
    from . import needle as _nd
    from .core import Fixture, Message
    from .distraction import NEUTRAL_CORPUS, span_candidates, span_text
    from .fixtures import audit_special_token_literals

    if type(formats) is not tuple or not formats:
        raise ValueError("formats must be a nonempty tuple")
    protocol: dict[str, Any] = dict(_nd.protocol_object(seed, _backend_fingerprint(backend)))
    protocol["fingerprint"] = _nd.protocol_fingerprint(seed, _backend_fingerprint(backend))
    cases = _nd.generate_cases(seed)
    if len(formats) * len(cases) > _nd.MAX_GROUND_RECORDS:
        return {
            "available": False,
            "detail": "needle_record_budget_exceeded",
            "protocol": protocol,
            "candidates": {},
        }
    corpus_text = "".join(NEUTRAL_CORPUS)
    generated_text = (
        _nd.FACT_TEMPLATE.replace("{topic}", "").replace("{value}", "")
        + "".join(_nd.TOPICS)
        + "".join(_nd.WORDS)
    )
    if audit_special_token_literals(generated_text + corpus_text, formats):
        # A special-token literal in the frozen texts would
        # forge turn structure in every cell of every run; the
        # stage fails closed before any generation.
        return {
            "available": False,
            "detail": "needle_screen_failed",
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
                "detail": "needle_screen_failed",
                "protocol": protocol,
                "candidates": {},
            }
    stream = _nd.filler_stream()
    corpus_tokens = max(1, len(backend.encode(corpus_text)))
    chars_per_token = len(corpus_text) / corpus_tokens
    order_cache: dict[tuple[int, int], list[tuple[int, int, int]]] = {}

    def _cancelled() -> bool:
        return cancel is not None and cancel.is_set()

    def _order(start: int, target: int) -> list[tuple[int, int, int]]:
        cache_key = (start, target)
        order = order_cache.get(cache_key)
        if order is None:
            order = span_candidates(stream[start:], target=target, chars_per_token=chars_per_token)
            order_cache[cache_key] = order
        return order

    def _render_ids(fmt: Any, content: str) -> tuple[int, ...]:
        fixture = Fixture(
            id="needle-probe",
            messages=(Message("user", content),),
            expected="",
            group=_nd.GROUP,
        )
        return tuple(backend.encode(fmt.render(fixture.messages)))

    def _size_prefix(
        fmt: Any,
        case: dict[str, Any],
        target: int,
        z_ids: tuple[int, ...],
        q_ids: tuple[int, ...],
        anchor: int,
        encodes: int,
    ) -> tuple[str, tuple[int, ...], int, int] | None:
        """Phase 1: (prefix_text, comparator_ids, consumed,
        encodes) for the first prefix whose D equals ``target``
        -- None when the budget runs out first. Depth 0 is the
        empty prefix with the question-only render as its
        comparator, no search."""
        fact = str(case["fact"])
        question = str(case["question"])
        if target == 0:
            return "", q_ids, 0, encodes
        for whole, suffix, _chars in _order(0, target):
            if encodes + 2 > _nd.SIZING_ENCODE_BUDGET:
                return None
            prefix = span_text(stream, whole, suffix)
            probe = _render_ids(fmt, prefix + fact + question)
            comparator = _render_ids(fmt, prefix + question)
            encodes += 2
            if _nd.lcp_len(probe, comparator) - anchor == target:
                return prefix, comparator, whole + (1 if suffix else 0), encodes
        return None

    def _size_treated(
        fmt: Any, case: dict[str, Any], dose: int, depth: int, base: dict[str, Any]
    ) -> tuple[str, tuple[int, ...], list[str]] | None:
        """Size one treated cell: (content, arm_ids, spans).

        Returns None when no construction inside the encode
        budget satisfies both equalities on the final render.
        """
        fact = str(case["fact"])
        question = str(case["question"])
        z_ids: tuple[int, ...] = base["z_ids"]
        q_ids: tuple[int, ...] = base["q_ids"]
        target = _nd.target_offset(dose, depth)
        sized = _size_prefix(fmt, case, target, z_ids, q_ids, base["anchor"], 0)
        if sized is None:
            return None
        prefix, comparator, consumed, encodes = sized
        if depth == 100:
            if encodes + 1 > _nd.SIZING_ENCODE_BUDGET:
                return None
            content = prefix + fact + question
            arm_ids = _render_ids(fmt, content)
            if (
                len(arm_ids) - len(z_ids) == dose
                and _nd.lcp_len(arm_ids, comparator) - base["anchor"] == target
            ):
                return content, arm_ids, [prefix, ""]
            return None
        for whole, suffix, _chars in _order(consumed, dose - target):
            if encodes + 1 > _nd.SIZING_ENCODE_BUDGET:
                return None
            tail = span_text(stream[consumed:], whole, suffix)
            content = prefix + fact + tail + question
            arm_ids = _render_ids(fmt, content)
            encodes += 1
            if (
                len(arm_ids) - len(z_ids) == dose
                and _nd.lcp_len(arm_ids, comparator) - base["anchor"] == target
            ):
                return content, arm_ids, [prefix, tail]
        return None

    def _size_duplicate(
        fmt: Any, case: dict[str, Any], dose: int, base: dict[str, Any]
    ) -> tuple[str, tuple[int, ...], list[str]] | None:
        """Size one duplicate cell (needles at 25% and 75%):
        prefix phase, middle phase against the second offset,
        suffix phase against the duplicate delta with both
        offsets re-verified on the final ids."""
        fact = str(case["fact"])
        question = str(case["question"])
        z_ids: tuple[int, ...] = base["z_ids"]
        q_ids: tuple[int, ...] = base["q_ids"]
        anchor: int = base["anchor"]
        needle_len: int = base["needle_len"]
        target1 = _nd.target_offset(dose, 25)
        target2 = _nd.target_offset(dose, 75)
        sized = _size_prefix(fmt, case, target1, z_ids, q_ids, anchor, 0)
        if sized is None:
            return None
        prefix, comparator1, consumed, encodes = sized
        mid: str | None = None
        comparator2: tuple[int, ...] | None = None
        mid_target = target2 - target1 - needle_len
        for whole, suffix, _chars in _order(consumed, mid_target):
            if encodes + 2 > _nd.SIZING_ENCODE_BUDGET:
                return None
            candidate_mid = span_text(stream[consumed:], whole, suffix)
            probe = _render_ids(fmt, prefix + fact + candidate_mid + fact + question)
            comparator = _render_ids(fmt, prefix + fact + candidate_mid + question)
            encodes += 2
            if _nd.lcp_len(probe, comparator) - anchor == target2:
                mid = candidate_mid
                comparator2 = comparator
                consumed += whole + (1 if suffix else 0)
                break
        if mid is None or comparator2 is None:
            return None
        for whole, suffix, _chars in _order(consumed, dose - target2):
            if encodes + 1 > _nd.SIZING_ENCODE_BUDGET:
                return None
            tail = span_text(stream[consumed:], whole, suffix)
            content = prefix + fact + mid + fact + tail + question
            arm_ids = _render_ids(fmt, content)
            encodes += 1
            if (
                len(arm_ids) - len(z_ids) == dose + needle_len
                and _nd.lcp_len(arm_ids, comparator1) - anchor == target1
                and _nd.lcp_len(arm_ids, comparator2) - anchor == target2
            ):
                return content, arm_ids, [prefix, mid, tail]
        return None

    def _size_absent(
        fmt: Any, case: dict[str, Any], dose: int, base: dict[str, Any]
    ) -> tuple[str, tuple[int, ...], list[str]] | None:
        """Size one absent cell: a single span whose rendered
        delta over the question-only render equals the dose."""
        question = str(case["question"])
        q_ids: tuple[int, ...] = base["q_ids"]
        encodes = 0
        for whole, suffix, _chars in _order(0, dose):
            if encodes + 1 > _nd.SIZING_ENCODE_BUDGET:
                return None
            span = span_text(stream, whole, suffix)
            if not span:
                continue
            content = span + question
            arm_ids = _render_ids(fmt, content)
            encodes += 1
            if len(arm_ids) - len(q_ids) == dose:
                return content, arm_ids, [span]
        return None

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
                fact = str(case["fact"])
                question = str(case["question"])
                record: dict[str, Any] = {
                    "fixture": case_id,
                    "topic": str(case["topic"]),
                    "expected_digest": _nd.digest_text(value),
                    "question_tokens": 0,
                    "cells": dict.fromkeys(_nd.CELL_KEYS),
                    "cell_details": dict.fromkeys(_nd.CELL_KEYS, ""),
                }
                records.append(record)
                z_ids = _render_ids(fmt, fact + question)
                q_ids = _render_ids(fmt, question)
                record["question_tokens"] = len(q_ids)
                needle_len = len(z_ids) - len(q_ids)
                if needle_len <= 0:
                    # A needle that adds no tokens under this
                    # candidate's tokenizer cannot be placed or
                    # measured; every cell of the case degrades
                    # together rather than pretending otherwise.
                    for key in _nd.CELL_KEYS:
                        record["cell_details"][key] = "arm_unmatchable"
                    continue
                base = {
                    "z_ids": z_ids,
                    "q_ids": q_ids,
                    "anchor": _nd.lcp_len(z_ids, q_ids),
                    "needle_len": needle_len,
                }

                # dose_0: the control arm, no sizing.
                _needle_run_cell(
                    backend,
                    fmt,
                    boundaries[name],
                    boundary_text[name],
                    record,
                    case_id,
                    value,
                    "dose_0",
                    fact + question,
                    z_ids,
                    [],
                    [0],
                    max_new_tokens=max_new_tokens,
                    max_context=max_context,
                    seed=seed,
                    cancel=cancel,
                )
                for dose in _nd.DOSES:
                    for depth in _nd.DEPTHS:
                        if _cancelled():
                            raise _NeedleCancelled
                        key = _nd.cell_key_treated(dose, depth)
                        if len(z_ids) + dose + max_new_tokens > max_context:
                            record["cell_details"][key] = "arm_context_exceeded"
                            continue
                        sized_treated = _size_treated(fmt, case, dose, depth, base)
                        if sized_treated is None:
                            record["cell_details"][key] = "arm_unmatchable"
                            continue
                        content, arm_ids, spans = sized_treated
                        _needle_run_cell(
                            backend,
                            fmt,
                            boundaries[name],
                            boundary_text[name],
                            record,
                            case_id,
                            value,
                            key,
                            content,
                            arm_ids,
                            spans,
                            [_nd.target_offset(dose, depth)],
                            max_new_tokens=max_new_tokens,
                            max_context=max_context,
                            seed=seed,
                            cancel=cancel,
                        )
                    if _cancelled():
                        raise _NeedleCancelled
                    absent_key = _nd.cell_key_absent(dose)
                    if len(q_ids) + dose + max_new_tokens > max_context:
                        record["cell_details"][absent_key] = "arm_context_exceeded"
                    else:
                        sized_absent = _size_absent(fmt, case, dose, base)
                        if sized_absent is None:
                            record["cell_details"][absent_key] = "arm_unmatchable"
                        else:
                            content, arm_ids, spans = sized_absent
                            _needle_run_cell(
                                backend,
                                fmt,
                                boundaries[name],
                                boundary_text[name],
                                record,
                                case_id,
                                value,
                                absent_key,
                                content,
                                arm_ids,
                                spans,
                                [],
                                max_new_tokens=max_new_tokens,
                                max_context=max_context,
                                seed=seed,
                                cancel=cancel,
                            )
                    if _cancelled():
                        raise _NeedleCancelled
                    duplicate_key = _nd.cell_key_duplicate(dose)
                    if len(z_ids) + dose + needle_len + max_new_tokens > max_context:
                        record["cell_details"][duplicate_key] = "arm_context_exceeded"
                    else:
                        sized_duplicate = _size_duplicate(fmt, case, dose, base)
                        if sized_duplicate is None:
                            record["cell_details"][duplicate_key] = "arm_unmatchable"
                        else:
                            content, arm_ids, spans = sized_duplicate
                            _needle_run_cell(
                                backend,
                                fmt,
                                boundaries[name],
                                boundary_text[name],
                                record,
                                case_id,
                                value,
                                duplicate_key,
                                content,
                                arm_ids,
                                spans,
                                [
                                    _nd.target_offset(dose, 25),
                                    _nd.target_offset(dose, 75),
                                ],
                                max_new_tokens=max_new_tokens,
                                max_context=max_context,
                                seed=seed,
                                cancel=cancel,
                            )
        except _NeedleCancelled:
            return {
                "available": False,
                "detail": "cancelled",
                "protocol": protocol,
                "candidates": {},
            }
        candidates[name] = {"status": "available", "detail": "", "records": records}
    return {"available": True, "detail": "", "protocol": protocol, "candidates": candidates}


def needle_analysis(
    stage: dict[str, Any],
    candidate_names: tuple[str, ...],
) -> dict[str, Any]:
    """Aggregate a needle stage into the stored payload
    (cycle 191).

    Per-candidate readouts come from needle.aggregate_candidate;
    the headline-population curves, deficit, and first-failure
    come from needle.headline_readouts over all candidates'
    records (the shared population is what makes them
    comparable); the per-cell selections re-apply the
    conservative rule with paired-restricted flip baselines
    from needle.cell_selection. Design: docs/NEEDLE_DESIGN.md.
    """
    from . import needle as _nd

    if not stage.get("available"):
        return {
            "available": False,
            "detail": str(stage.get("detail", "needle_unavailable")),
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
        aggregates = _nd.aggregate_candidate(entry["records"])
        aggregates.update({"status": "available", "detail": "", "records": entry["records"]})
        candidates_out[name] = aggregates
        records_by_candidate[name] = entry["records"]
    return {
        "available": True,
        "detail": "",
        "protocol": stage["protocol"],
        "candidates": candidates_out,
        "readouts": _nd.headline_readouts(records_by_candidate),
        "conditions": {key: _nd.cell_selection(records_by_candidate, key) for key in _nd.CELL_KEYS},
        "n_candidates": len(candidate_names),
        "n_cases": len(_nd.generate_cases(int(stage["protocol"]["seed"]))),
    }


class _PlacementCancelled(Exception):
    """Internal: the run's cancel event fired during the
    placement stage; the stage unwinds to its fail-closed
    payload rather than mislabelling the interruption."""


def _placement_cell(
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
    """One placement generation: (detail, cell).

    Exactly one of the pair is set. ``ids`` are the arm
    render's ids, measured by the stage at construction for
    the record's ``arm_tokens``; the cell's ``input_tokens``
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
        raise _PlacementCancelled
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
        raise _PlacementCancelled
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


def placement_stage(
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
    """Instruction placement stage (cycle 193): generated
    single-fact cases with the question moved around a frozen
    short payload, per candidate.

    The cases are generated from the run seed inside the stage;
    there is no native arm and no dependence on the user's
    fixtures or the primary run. Each case contributes four
    cells (question-first, payload-first, sandwich, closed-book
    screen) whose contents differ only in where the question
    sits. Cells degrade individually in the fixed detail
    vocabulary; cancellation unwinds the whole stage to its
    fail-closed payload. Design: docs/PLACEMENT_DESIGN.md.
    """
    from . import placement as _pl
    from .core import Fixture, Message
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
    protocol: dict[str, Any] = dict(_pl.protocol_object(seed, _backend_fingerprint(backend)))
    protocol["fingerprint"] = _pl.protocol_fingerprint(seed, _backend_fingerprint(backend))
    cases = _pl.generate_cases(seed)
    if len(formats) * len(cases) > _pl.MAX_GROUND_RECORDS:
        return {
            "available": False,
            "detail": "placement_record_budget_exceeded",
            "protocol": protocol,
            "candidates": {},
        }
    generated_text = (
        _pl.FACT_TEMPLATE.replace("{topic}", "").replace("{value}", "")
        + _pl.QUESTION_TEMPLATE.replace("{topic}", "")
        + "".join(_pl.TOPICS)
        + "".join(_pl.WORDS)
        + "".join(_pl.NEUTRAL_POOL)
    )
    if audit_special_token_literals(generated_text, formats):
        # A special-token literal in the frozen texts would
        # forge turn structure in every cell of every run; the
        # stage fails closed before any generation.
        return {
            "available": False,
            "detail": "placement_screen_failed",
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
                "detail": "placement_screen_failed",
                "protocol": protocol,
                "candidates": {},
            }

    def _cancelled() -> bool:
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
                record: dict[str, Any] = {
                    "fixture": case_id,
                    "topic": str(case["topic"]),
                    "expected_digest": _pl.digest_text(value),
                    "question_tokens": 0,
                    "arm_tokens": dict.fromkeys(_pl.ARM_KEYS, 0),
                    "cells": dict.fromkeys(_pl.ARM_KEYS),
                    "cell_details": dict.fromkeys(_pl.ARM_KEYS, ""),
                }
                records.append(record)
                for arm in _pl.ARM_KEYS:
                    if _cancelled():
                        raise _PlacementCancelled
                    content = _pl.arm_content(case, arm)
                    fixture = Fixture(
                        id=f"{case_id}-{arm}",
                        messages=(Message("user", content),),
                        expected=value,
                        group=_pl.GROUP,
                    )
                    arm_ids = tuple(backend.encode(fmt.render(fixture.messages)))
                    record["arm_tokens"][arm] = len(arm_ids)
                    detail, cell = _placement_cell(
                        backend,
                        fmt,
                        fixture,
                        arm_ids,
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
                record["question_tokens"] = record["arm_tokens"]["closed_book"]
        except _PlacementCancelled:
            return {
                "available": False,
                "detail": "cancelled",
                "protocol": protocol,
                "candidates": {},
            }
        candidates[name] = {"status": "available", "detail": "", "records": records}
    return {"available": True, "detail": "", "protocol": protocol, "candidates": candidates}


def placement_analysis(
    stage: dict[str, Any],
    candidate_names: tuple[str, ...],
) -> dict[str, Any]:
    """Aggregate a placement stage into the stored payload
    (cycle 193).

    Per-candidate readouts come from
    placement.aggregate_candidate; the per-arm selections come
    from placement.arm_selection over the shared live
    population. Design: docs/PLACEMENT_DESIGN.md.
    """
    from . import placement as _pl

    if not stage.get("available"):
        return {
            "available": False,
            "detail": str(stage.get("detail", "placement_unavailable")),
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
        aggregates = _pl.aggregate_candidate(entry["records"])
        aggregates.update({"status": "available", "detail": "", "records": entry["records"]})
        candidates_out[name] = aggregates
        records_by_candidate[name] = entry["records"]
    return {
        "available": True,
        "detail": "",
        "protocol": stage["protocol"],
        "candidates": candidates_out,
        "conditions": {
            arm: _pl.arm_selection(records_by_candidate, arm) for arm in _pl.LAYOUT_ARMS
        },
        "n_candidates": len(candidate_names),
        "n_cases": len(_pl.generate_cases(int(stage["protocol"]["seed"]))),
    }


class _CanonicalFormCancelled(Exception):
    """Internal: the run's cancel event fired during the
    canonical-form stage; the stage unwinds to its fail-closed
    payload rather than mislabelling the interruption."""


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
    from .core import Fixture, Message
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


class _NegationCancelled(Exception):
    """Internal: the run's cancel event fired during the
    negation stage; the stage unwinds to its fail-closed
    payload rather than mislabelling the interruption."""


def _negation_cell(
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
    """One negation generation: (detail, cell).

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
        raise _NegationCancelled
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
        raise _NegationCancelled
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


def negation_stage(
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
    """Negation stage (cycle 212): generated numform compare
    pairs rendered under three headline instruction arms
    (direct, negated, flipped) plus one descriptive arm
    (negated_alt) per candidate.

    The cases are generated from the run seed inside the
    stage; there is no native arm and no dependence on the
    user's fixtures or the primary run. Each case contributes
    four cells whose contents differ only in the instruction
    sentence, plus the id-divergence measurement: every arm's
    full render id sequence is encoded before any generation,
    so same-ids cases are reported as a rendering anomaly
    rather than gated away, and the headline population
    (completed in all three headline arms, direct-negated ids
    diverged) is decided before any generation is spent.
    The pre-generation audit screen (frozen instruction
    strings, distinctness, margin-class validity, boundary
    text) fails the whole stage closed with
    ``negation_screen_failed``; its counts travel in the
    payload's ``audit`` block, never silently dropped.
    Design: docs/NEGATION_DESIGN.md.
    """
    from . import negation as _ng
    from .core import Fixture, Message
    from .fixtures import audit_special_token_literals
    from .numform import margin_class

    if type(formats) is not tuple or not formats:
        raise ValueError("formats must be a nonempty tuple")
    names = [fmt.name for fmt in formats]
    if len(set(names)) != len(names):
        # Candidate records are keyed by name; two formats
        # sharing a name would silently collapse into one
        # candidate's records (red-team F7, cycle 193). The
        # precondition is stated at this stage's own boundary.
        raise ValueError("formats must have unique names")
    protocol: dict[str, Any] = dict(_ng.protocol_object(seed, _backend_fingerprint(backend)))
    protocol["fingerprint"] = _ng.protocol_fingerprint(seed, _backend_fingerprint(backend))
    cases = _ng.cases_for(seed)
    audit: dict[str, Any] = {
        "n_audit_excluded": 0,
        "by_reason": {},
        # nosec B105 -- integer audit counter field, not a credential
        "special_token_findings": 0,
        "equal_pairs": 0,
        "invalid_margin_classes": 0,
    }
    if len(formats) * len(cases) > _ng.MAX_GROUND_RECORDS:
        return {
            "available": False,
            "detail": "negation_record_budget_exceeded",
            "protocol": protocol,
            "audit": audit,
            "candidates": {},
        }
    audit_text = "".join(_ng.INSTRUCTIONS[arm] for arm in _ng.ARM_KEYS) + "".join(
        str(value) for case in cases for value in case["operands"]
    )
    findings = audit_special_token_literals(audit_text, formats)
    equal_pairs = sum(1 for case in cases if case["operands"][0] == case["operands"][1])
    invalid_margins = sum(
        1
        for case in cases
        if case["stratum"] not in _ng.EMITTED_MARGIN_CLASSES
        or margin_class(case["operands"][0], case["operands"][1]) != case["stratum"]
    )
    if findings or equal_pairs or invalid_margins:
        # The screen covers the whole generated set, not one
        # case: any failure makes the stage unavailable, with
        # the failure counts in the audit block (never a
        # silent per-case drop, never a tie rule).
        audit = {
            "n_audit_excluded": 0,
            "by_reason": {
                "special_token_literal": len(findings),
                "equal_pair": equal_pairs,
                "invalid_margin_class": invalid_margins,
            },
            "special_token_findings": len(findings),
            "equal_pairs": equal_pairs,
            "invalid_margin_classes": invalid_margins,
        }
        return {
            "available": False,
            "detail": "negation_screen_failed",
            "protocol": protocol,
            "audit": audit,
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
        if boundary_text[fmt.name] and boundary_text[fmt.name] in audit_text:
            return {
                "available": False,
                "detail": "negation_screen_failed",
                "protocol": protocol,
                "audit": audit,
                "candidates": {},
            }

    def _cancelled() -> bool:
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
                        "audit": audit,
                        "candidates": {},
                    }
                case_id = str(case["fixture_id"])
                # Every arm's render is encoded BEFORE any
                # generation: id divergence is an encode-time
                # measurement, and the headline population is
                # decided here, not inferred from verdicts.
                arm_ids: dict[str, tuple[int, ...]] = {}
                for arm in _ng.ARM_KEYS:
                    fixture = Fixture(
                        id=f"{case_id}-{arm}",
                        messages=(Message("user", str(case["contents"][arm])),),
                        expected=str(case["expected"][arm]),
                        group=_ng.GROUP,
                    )
                    arm_ids[arm] = tuple(backend.encode(fmt.render(fixture.messages)))
                record: dict[str, Any] = {
                    "fixture": case_id,
                    "operands": list(case["operands"]),
                    "larger_first": bool(case["larger_first"]),
                    "stratum": str(case["stratum"]),
                    "band": int(case["band"]),
                    "expected_digest": _ng.digest_text(str(case["expected"]["direct"])),
                    "arm_tokens": {arm: len(arm_ids[arm]) for arm in _ng.ARM_KEYS},
                    "id_digests": {arm: _ng.digest_ids(arm_ids[arm]) for arm in _ng.ARM_KEYS},
                    "id_lengths": {arm: len(arm_ids[arm]) for arm in _ng.ARM_KEYS},
                    "ids_identical": arm_ids["direct"] == arm_ids["negated"],
                    "id_length_delta": len(arm_ids["negated"]) - len(arm_ids["direct"]),
                    "content_digests": {
                        arm: _ng.digest_text(str(case["contents"][arm])) for arm in _ng.ARM_KEYS
                    },
                    "cells": dict.fromkeys(_ng.ARM_KEYS),
                    "cell_details": dict.fromkeys(_ng.ARM_KEYS, ""),
                }
                records.append(record)
                for arm in _ng.ARM_KEYS:
                    if _cancelled():
                        raise _NegationCancelled
                    fixture = Fixture(
                        id=f"{case_id}-{arm}",
                        messages=(Message("user", str(case["contents"][arm])),),
                        expected=str(case["expected"][arm]),
                        group=_ng.GROUP,
                    )
                    detail, cell = _negation_cell(
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
        except _NegationCancelled:
            return {
                "available": False,
                "detail": "cancelled",
                "protocol": protocol,
                "audit": audit,
                "candidates": {},
            }
        candidates[name] = {"status": "available", "detail": "", "records": records}
    return {
        "available": True,
        "detail": "",
        "protocol": protocol,
        "audit": audit,
        "candidates": candidates,
    }


def negation_analysis(
    stage: dict[str, Any],
    candidate_names: tuple[str, ...],
) -> dict[str, Any]:
    """Aggregate a negation stage into the stored payload
    (cycle 212).

    Per-candidate readouts come from
    negation.aggregate_candidate over the candidate's ground
    records. There is no selection and no winner readout in
    v1: per-arm pass rates and the joint (D, J) pattern are
    the only readouts. Design: docs/NEGATION_DESIGN.md.
    """
    from . import negation as _ng

    if not stage.get("available"):
        return {
            "available": False,
            "detail": str(stage.get("detail", "negation_unavailable")),
            "protocol": stage.get("protocol"),
            "audit": stage.get("audit"),
        }
    stage_candidates = stage["candidates"]
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
        aggregates = _ng.aggregate_candidate(entry["records"])
        aggregates.update({"status": "available", "detail": "", "records": entry["records"]})
        candidates_out[name] = aggregates
    return {
        "available": True,
        "detail": "",
        "protocol": stage["protocol"],
        "audit": stage.get("audit"),
        "candidates": candidates_out,
        "n_candidates": len(candidate_names),
        "n_cases": len(_ng.cases_for(int(stage["protocol"]["seed"]))),
    }


class _RotationCancelled(Exception):
    """Internal: the run's cancel event fired during the
    rotation stage; the stage unwinds to its fail-closed
    payload rather than mislabelling the interruption."""


def _rotation_cell(
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
    """One rotation generation: (detail, cell).

    Exactly one of the pair is set. ``ids`` are the arm
    render's ids, measured by the stage at construction for
    the record's ``arm_tokens`` and inventory digest; the
    cell's ``input_tokens`` come from the evaluation's own
    observation, so a backend whose encode disagreed with
    itself surfaces as a validator rejection downstream, never
    a silently wrong record. The cancel event is checked
    before and after ``evaluate`` (the multiturn rule: a
    cancel observed during a generation is never mislabelled
    an evaluation failure).
    """
    if len(ids) + max_new_tokens > max_context:
        return "arm_context_exceeded", None
    if cancel is not None and cancel.is_set():
        raise _RotationCancelled
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
        raise _RotationCancelled
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


def rotation_stage(
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
    """Context rotation stage (cycle 203): generated
    five-block cases whose block span is cyclically rotated
    (plus mirror, absent, and closed-book arms), per candidate.

    The cases are generated from the run seed inside the stage;
    there is no native arm and no dependence on the user's
    fixtures or the primary run. Each case contributes eleven
    cells whose contents differ only in the block order (and,
    for the absent and closed-book arms, in the fact's
    presence). Construction is measured per arm at encode
    time -- token count and, for span arms, the inventory
    digest the derived id_exact classification reads -- before
    any generation. Cells degrade individually in the fixed
    detail vocabulary; a case whose block texts contain the
    candidate's decoded boundary text is screened as a unit;
    cancellation unwinds the whole stage to its fail-closed
    payload. Design: docs/ROTATION_DESIGN.md.
    """
    from . import rotation as _rot
    from .core import Fixture, Message
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
    protocol: dict[str, Any] = dict(_rot.protocol_object(seed, _backend_fingerprint(backend)))
    protocol["fingerprint"] = _rot.protocol_fingerprint(seed, _backend_fingerprint(backend))
    cases = _rot.generate_cases(seed)
    if len(formats) * len(cases) > _rot.MAX_GROUND_RECORDS:
        return {
            "available": False,
            "detail": "rotation_record_budget_exceeded",
            "protocol": protocol,
            "candidates": {},
        }
    generated_text = (
        _rot.FACT_TEMPLATE.replace("{topic}", "").replace("{value}", "")
        + _rot.QUESTION_TEMPLATE.replace("{topic}", "")
        + "".join(_rot.TOPICS)
        + "".join(_rot.WORDS)
    )
    if audit_special_token_literals(generated_text, formats):
        # A special-token literal in the frozen texts would
        # forge turn structure in every cell of every run; the
        # stage fails closed before any generation.
        return {
            "available": False,
            "detail": "rotation_screen_failed",
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
                "detail": "rotation_screen_failed",
                "protocol": protocol,
                "candidates": {},
            }

    def _cancelled() -> bool:
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
                record: dict[str, Any] = {
                    "fixture": case_id,
                    "topic": str(case["topic"]),
                    "expected_digest": _rot.digest_text(value),
                    "arm_tokens": dict.fromkeys(_rot.ARM_KEYS, 0),
                    "arm_inventory_digests": dict.fromkeys(_rot.SPAN_ARMS, ""),
                    "cells": dict.fromkeys(_rot.ARM_KEYS),
                    "cell_details": dict.fromkeys(_rot.ARM_KEYS, ""),
                }
                records.append(record)
                case_screened = bool(boundary_text[name]) and any(
                    boundary_text[name] in str(block) for block in case["blocks"]
                )
                for arm in _rot.ARM_KEYS:
                    if _cancelled():
                        raise _RotationCancelled
                    content = _rot.arm_content(case, arm)
                    fixture = Fixture(
                        id=f"{case_id}-{arm}",
                        messages=(Message("user", content),),
                        expected=value,
                        group=_rot.GROUP,
                    )
                    arm_ids = tuple(backend.encode(fmt.render(fixture.messages)))
                    record["arm_tokens"][arm] = len(arm_ids)
                    if arm in _rot.SPAN_ARMS:
                        record["arm_inventory_digests"][arm] = _rot.inventory_digest(arm_ids)
                    if case_screened:
                        record["cell_details"][arm] = "arm_screened"
                        continue
                    detail, cell = _rotation_cell(
                        backend,
                        fmt,
                        fixture,
                        arm_ids,
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
        except _RotationCancelled:
            return {
                "available": False,
                "detail": "cancelled",
                "protocol": protocol,
                "candidates": {},
            }
        candidates[name] = {"status": "available", "detail": "", "records": records}
    return {"available": True, "detail": "", "protocol": protocol, "candidates": candidates}


def rotation_analysis(
    stage: dict[str, Any],
    candidate_names: tuple[str, ...],
) -> dict[str, Any]:
    """Aggregate a rotation stage into the stored payload
    (cycle 203).

    Per-candidate readouts come from
    rotation.aggregate_candidate; the per-arm selections come
    from rotation.arm_selection over the shared live
    population, gated per arm on derived construction
    exactness. Design: docs/ROTATION_DESIGN.md.
    """
    from . import rotation as _rot

    if not stage.get("available"):
        return {
            "available": False,
            "detail": str(stage.get("detail", "rotation_unavailable")),
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
        aggregates = _rot.aggregate_candidate(entry["records"])
        aggregates.update({"status": "available", "detail": "", "records": entry["records"]})
        candidates_out[name] = aggregates
        records_by_candidate[name] = entry["records"]
    return {
        "available": True,
        "detail": "",
        "protocol": stage["protocol"],
        "candidates": candidates_out,
        "conditions": {
            arm: _rot.arm_selection(records_by_candidate, arm) for arm in _rot.CONDITION_ARMS
        },
        "n_candidates": len(candidate_names),
        "n_cases": len(_rot.generate_cases(int(stage["protocol"]["seed"]))),
    }


class _DensityCancelled(Exception):
    """Internal: the run's cancel event fired during the
    density stage; the stage unwinds to its fail-closed
    payload rather than mislabelling the interruption."""


def _density_cell(
    backend: Any,
    fmt: Any,
    fixture: Any,
    *,
    boundary_ids: tuple[int, ...],
    max_new_tokens: int,
    max_context: int,
    input_tokens: int,
    seed: int,
    cancel: Event | None,
) -> tuple[str | None, dict[str, Any] | None]:
    """One density generation: (detail, record fields).

    Exactly one of the pair is set. The record's commitment is
    computed here from the evaluation's own answer text: the
    ordered per-word hash list, its digest, and the count.
    Satisfaction is never computed at generation time; it is
    derived from the commitment by density.aggregate_candidate
    (and re-derived by the diff validator's mirror). The
    cancel event is checked before and after ``evaluate`` (the
    multiturn rule: a cancel observed during a generation is
    never mislabelled an evaluation failure).
    """
    from . import density as _den

    if input_tokens + max_new_tokens > max_context:
        return "arm_context_exceeded", None
    if cancel is not None and cancel.is_set():
        raise _DensityCancelled
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
        raise _DensityCancelled
    if report.aggregate_eligible and report.observations:
        obs = report.observations[0]
        hashes = _den.word_hashes(obs.answer)
        return None, {
            "boundary_found": bool(obs.boundary_found),
            "answer_digest": _answer_digest(obs.answer),
            "word_hashes": hashes,
            "word_list_digest": _den.commitment_digest(hashes),
            "word_count": len(hashes),
            "input_tokens": len(obs.input_ids),
        }
    return "arm_evaluation_failed", None


def density_stage(
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
    """Instruction-density stage (cycle 204): generated cases
    whose constraint prefix grows across arms (solo per
    constraint, then the first 2/4/8/16 constraints, plus a
    length-matched sham), per candidate.

    The cases are generated from the run seed inside the stage;
    there is no dependence on the user's fixtures or the primary
    run. Each (case, arm) contributes one flat record carrying
    the answer's word-hash commitment -- never a stored
    satisfaction verdict, which is derived downstream. Records
    degrade individually in the fixed detail vocabulary; a case
    whose arm contents contain the candidate's decoded boundary
    text is screened as a unit; cancellation unwinds the whole
    stage to its fail-closed payload. Design:
    docs/DENSITY_DESIGN.md.
    """
    from . import density as _den
    from .core import Fixture, Message
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
    protocol: dict[str, Any] = dict(_den.protocol_object(seed, _backend_fingerprint(backend)))
    protocol["fingerprint"] = _den.protocol_fingerprint(seed, _backend_fingerprint(backend))
    cases = _den.generate_cases(seed)
    if len(formats) * len(cases) * len(_den.ARM_KEYS) > _den.MAX_DENSITY_RECORDS:
        return {
            "available": False,
            "detail": "density_record_budget_exceeded",
            "protocol": protocol,
            "candidates": {},
        }
    generated_text = (
        _den.TASK_FACT_TEMPLATE.replace("{topic}", "").replace("{value}", "")
        + _den.TASK_QUESTION_TEMPLATE.replace("{topic}", "")
        + "".join(
            template.replace("{target}", "").replace("{limit}", "")
            for template in _den.CONSTRAINT_TEMPLATES.values()
        )
        + _den.INSTRUCTION_LINE_TEMPLATE.replace("{position}", "").replace("{text}", "")
        + "".join(_den.TOPICS)
        + "".join(_den.WORDS)
        + "".join(_den.TARGET_WORDS)
        + "".join(_den.NEUTRAL_STATEMENTS)
    )
    if audit_special_token_literals(generated_text, formats):
        # A special-token literal in the frozen texts would
        # forge turn structure in every record of every run;
        # the stage fails closed before any generation.
        return {
            "available": False,
            "detail": "density_screen_failed",
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
                "detail": "density_screen_failed",
                "protocol": protocol,
                "candidates": {},
            }

    def _cancelled() -> bool:
        return cancel is not None and cancel.is_set()

    candidates: dict[str, Any] = {}
    for fmt in formats:
        name = fmt.name
        records: list[dict[str, Any]] = []
        try:
            for case_index, case in enumerate(cases):
                if _cancelled():
                    return {
                        "available": False,
                        "detail": "cancelled",
                        "protocol": protocol,
                        "candidates": {},
                    }
                case_id = str(case["fixture_id"])
                contents = {arm: _den.arm_content(case, arm) for arm in _den.ARM_KEYS}
                case_screened = bool(boundary_text[name]) and any(
                    boundary_text[name] in content for content in contents.values()
                )
                for arm in _den.ARM_KEYS:
                    if _cancelled():
                        raise _DensityCancelled
                    content = contents[arm]
                    fixture = Fixture(
                        id=f"{case_id}-{arm}",
                        messages=(Message("user", content),),
                        expected=str(case["value"]),
                        group=_den.GROUP,
                    )
                    arm_ids = tuple(backend.encode(fmt.render(fixture.messages)))
                    record: dict[str, Any] = {
                        "candidate": name,
                        "case_index": case_index,
                        "arm": arm,
                        "detail": "",
                        "case_screened": case_screened,
                        "boundary_found": False,
                        "answer_digest": None,
                        "word_hashes": [],
                        "word_list_digest": _den.commitment_digest([]),
                        "word_count": 0,
                        "input_tokens": len(arm_ids),
                        "content_digest": _den.digest_text(content),
                    }
                    records.append(record)
                    if case_screened:
                        continue
                    detail, fields = _density_cell(
                        backend,
                        fmt,
                        fixture,
                        boundary_ids=boundaries[name],
                        max_new_tokens=max_new_tokens,
                        max_context=max_context,
                        input_tokens=len(arm_ids),
                        seed=seed,
                        cancel=cancel,
                    )
                    if detail is not None:
                        record["detail"] = detail
                    elif fields is not None:
                        record.update(fields)
        except _DensityCancelled:
            return {
                "available": False,
                "detail": "cancelled",
                "protocol": protocol,
                "candidates": {},
            }
        candidates[name] = {"status": "available", "detail": "", "records": records}
    return {"available": True, "detail": "", "protocol": protocol, "candidates": candidates}


def density_analysis(
    stage: dict[str, Any],
    candidate_names: tuple[str, ...],
) -> dict[str, Any]:
    """Aggregate a density stage into the stored payload
    (cycle 204).

    Per-candidate readouts come from
    density.aggregate_candidate over the stage's flat records;
    satisfaction is derived there from the records' word-hash
    commitments, never stored on a record. Design:
    docs/DENSITY_DESIGN.md.
    """
    from . import density as _den

    if not stage.get("available"):
        return {
            "available": False,
            "detail": str(stage.get("detail", "density_unavailable")),
            "protocol": stage.get("protocol"),
        }
    cases = _den.generate_cases(int(stage["protocol"]["seed"]))
    stage_candidates = stage["candidates"]
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
        aggregates = _den.aggregate_candidate(entry["records"], cases)
        aggregates.update({"status": "available", "records": entry["records"]})
        candidates_out[name] = aggregates
    return {
        "available": True,
        "detail": "",
        "protocol": stage["protocol"],
        "candidates": candidates_out,
        "n_candidates": len(candidate_names),
        "n_cases": len(cases),
    }


class _FieldOrderCancelled(Exception):
    """Internal: unwinds a cancelled field-order stage."""


def _fieldorder_cell(
    backend: Any,
    fmt: Any,
    fixture: Any,
    *,
    boundary_ids: tuple[int, ...],
    max_new_tokens: int,
    max_context: int,
    input_tokens: int,
    seed: int,
    cancel: Event | None,
) -> tuple[str | None, dict[str, Any] | None]:
    """One field-order generation: (detail, record fields).

    The record's document commitment is computed here from the
    evaluation's own answer text via fieldorder.parse_document;
    correctness and adherence are never computed at generation
    time -- they are derived from the commitment downstream.
    The cancel event is checked before and after ``evaluate``
    (the multiturn rule: a cancel observed during a generation
    is never mislabelled an evaluation failure).
    """
    from . import fieldorder as _fo

    if input_tokens + max_new_tokens > max_context:
        return "arm_context_exceeded", None
    if cancel is not None and cancel.is_set():
        raise _FieldOrderCancelled
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
        raise _FieldOrderCancelled
    if report.aggregate_eligible and report.observations:
        obs = report.observations[0]
        parsed = _fo.parse_document(obs.answer)
        return None, {
            "boundary_found": bool(obs.boundary_found),
            "answer_digest": _answer_digest(obs.answer),
            "document_kind": parsed["kind"],
            "document_digest": parsed["document_digest"],
            "field_values": parsed["field_values"],
            "duplicate_keys": parsed["duplicate_keys"],
            "input_tokens": len(obs.input_ids),
        }
    return "arm_evaluation_failed", None


def fieldorder_stage(
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
    """Schema field-order stage (cycle 205): the frozen spec set
    rendered in four field orders per candidate.

    The specs are frozen in fieldorder.FIELD_ORDER_SPECS; there
    is no dependence on the user's fixtures or the primary run.
    Each (spec, arm) contributes one flat record carrying the
    parsed document's per-field canonical values -- never a
    stored correctness verdict, which is derived downstream.
    Records degrade individually in the fixed detail vocabulary;
    a spec whose arm prompts contain the candidate's decoded
    boundary text is screened as a unit; cancellation unwinds
    the whole stage to its fail-closed payload. Design:
    docs/FIELDORDER_DESIGN.md.
    """
    from . import fieldorder as _fo
    from .core import Fixture, Message
    from .fixtures import audit_special_token_literals
    from .structured import _canonical_json

    if type(formats) is not tuple or not formats:
        raise ValueError("formats must be a nonempty tuple")
    names = [fmt.name for fmt in formats]
    if len(set(names)) != len(names):
        raise ValueError("formats must have unique names")
    protocol: dict[str, Any] = dict(_fo.protocol_object(seed, _backend_fingerprint(backend)))
    protocol["fingerprint"] = _fo.protocol_fingerprint(seed, _backend_fingerprint(backend))
    if len(formats) * len(_fo.FIELD_ORDER_SPECS) * len(_fo.ARM_KEYS) > _fo.MAX_FIELD_ORDER_RECORDS:
        return {
            "available": False,
            "detail": "field_order_record_budget_exceeded",
            "protocol": protocol,
            "candidates": {},
        }
    prompts = {
        (spec["id"], arm): _fo.arm_prompt(spec, arm)
        for spec in _fo.FIELD_ORDER_SPECS
        for arm in _fo.ARM_KEYS
    }
    generated_text = "".join(prompts.values())
    if audit_special_token_literals(generated_text, formats):
        return {
            "available": False,
            "detail": "field_order_screen_failed",
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

    def _cancelled() -> bool:
        return cancel is not None and cancel.is_set()

    candidates: dict[str, Any] = {}
    for fmt in formats:
        name = fmt.name
        records: list[dict[str, Any]] = []
        try:
            for spec in _fo.FIELD_ORDER_SPECS:
                if _cancelled():
                    return {
                        "available": False,
                        "detail": "cancelled",
                        "protocol": protocol,
                        "candidates": {},
                    }
                spec_id = str(spec["id"])
                spec_prompts = {arm: prompts[(spec_id, arm)] for arm in _fo.ARM_KEYS}
                case_screened = bool(boundary_text[name]) and any(
                    boundary_text[name] in content for content in spec_prompts.values()
                )
                for arm in _fo.ARM_KEYS:
                    if _cancelled():
                        raise _FieldOrderCancelled
                    content = spec_prompts[arm]
                    fixture = Fixture(
                        id=f"fieldorder/{spec_id}/{arm}",
                        messages=(Message("user", content),),
                        expected=_canonical_json(spec["answer"]),
                        group=_fo.GROUP,
                    )
                    arm_ids = tuple(backend.encode(fmt.render(fixture.messages)))
                    record: dict[str, Any] = {
                        "candidate": name,
                        "spec_id": spec_id,
                        "arm": arm,
                        "detail": "",
                        "case_screened": case_screened,
                        "boundary_found": False,
                        "duplicate_keys": False,
                        "answer_digest": None,
                        "document_kind": "",
                        "document_digest": None,
                        "field_values": {},
                        "input_tokens": len(arm_ids),
                        "content_digest": _fo.digest_text(content),
                    }
                    records.append(record)
                    if case_screened:
                        continue
                    detail, fields = _fieldorder_cell(
                        backend,
                        fmt,
                        fixture,
                        boundary_ids=boundaries[name],
                        max_new_tokens=max_new_tokens,
                        max_context=max_context,
                        input_tokens=len(arm_ids),
                        seed=seed,
                        cancel=cancel,
                    )
                    if detail is not None:
                        record["detail"] = detail
                    elif fields is not None:
                        record.update(fields)
                        if (
                            sum(
                                len(value)
                                for value in record["field_values"].values()
                                if type(value) is str
                            )
                            > _fo.FIELD_VALUES_CAP
                        ):
                            # A live record over the commitment
                            # cap degrades individually (design
                            # F8); the payload never carries it.
                            record["detail"] = "arm_evaluation_failed"
                            record["document_kind"] = ""
                            record["document_digest"] = None
                            record["field_values"] = {}
                            record["answer_digest"] = None
                            record["boundary_found"] = False
                            record["duplicate_keys"] = False
        except _FieldOrderCancelled:
            return {
                "available": False,
                "detail": "cancelled",
                "protocol": protocol,
                "candidates": {},
            }
        candidates[name] = {"status": "available", "detail": "", "records": records}
    return {"available": True, "detail": "", "protocol": protocol, "candidates": candidates}


def fieldorder_analysis(
    stage: dict[str, Any],
    candidate_names: tuple[str, ...],
) -> dict[str, Any]:
    """Aggregate a field-order stage into the stored payload
    (cycle 205).

    Per-candidate readouts come from
    fieldorder.aggregate_candidate over the stage's flat
    records; correctness is derived there from the records'
    per-field canonical values, never stored on a record.
    Design: docs/FIELDORDER_DESIGN.md.
    """
    from . import fieldorder as _fo

    if not stage.get("available"):
        return {
            "available": False,
            "detail": str(stage.get("detail", "field_order_unavailable")),
            "protocol": stage.get("protocol"),
        }
    stage_candidates = stage["candidates"]
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
        aggregates = _fo.aggregate_candidate(entry["records"])
        aggregates.update({"status": "available", "records": entry["records"]})
        candidates_out[name] = aggregates
    return {
        "available": True,
        "detail": "",
        "protocol": stage["protocol"],
        "candidates": candidates_out,
        "n_candidates": len(candidate_names),
        "n_specs": len(_fo.FIELD_ORDER_SPECS),
    }


def numform_stage(
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
    """Numeric surface-form stage (cycle 186): generated integer
    fixtures in four written forms per candidate.

    The fixtures are generated from the run seed inside the
    stage (the hierarchy synthetic-construction pattern); the
    mechanics are the leading/distraction ones -- per-fixture
    ``evaluate()``, cancel and context threaded, per-cell
    fixed-vocabulary details. There is no native arm and no
    dependence on the user's fixtures or the primary run: the
    same values are drawn once per task and rendered into every
    form under one shared fixture id, so request seeds pair the
    arms. Design: docs/NUMFORM_DESIGN.md.
    """
    from . import numform as _nf
    from .fixtures import audit_special_token_literals

    if type(formats) is not tuple or not formats:
        raise ValueError("formats must be a nonempty tuple")
    protocol: dict[str, Any] = dict(_nf.protocol_object(seed, _backend_fingerprint(backend)))
    protocol["fingerprint"] = _nf.protocol_fingerprint(seed, _backend_fingerprint(backend))
    if (
        len(formats) * _nf.VALUES_PER_TASK * len(_nf.TASKS) * len(_nf.FORMS)
        > _nf.MAX_GROUND_RECORDS
    ):
        return {
            "available": False,
            "detail": "numform_record_budget_exceeded",
            "protocol": protocol,
            "candidates": {},
        }
    template_text = "".join(
        template.replace("{operand}", "").replace("{a}", "").replace("{b}", "")
        for template in _nf.PROMPT_TEMPLATES.values()
    )
    if audit_special_token_literals(template_text, formats):
        # A special-token literal in the frozen templates would
        # forge turn structure in every cell of every run; the
        # stage fails closed before any generation.
        return {
            "available": False,
            "detail": "numform_template_screen_failed",
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
        if boundary_text[fmt.name] and boundary_text[fmt.name] in template_text:
            return {
                "available": False,
                "detail": "numform_template_screen_failed",
                "protocol": protocol,
                "candidates": {},
            }
    value_set = _nf.generate_value_set(seed)
    fixtures = _nf.build_fixtures(seed)
    grouping = _nf.measure_grouping(backend.encode, value_set)

    def _cancelled() -> bool:
        return cancel is not None and cancel.is_set()

    if _cancelled():
        return {
            "available": False,
            "detail": "cancelled",
            "protocol": protocol,
            "candidates": {},
        }
    candidates: dict[str, Any] = {}
    for fmt in formats:
        name = fmt.name
        records: list[dict[str, Any]] = []
        for task in _nf.TASKS:
            for form in _nf.FORMS:
                for entry, case in zip(value_set[task], fixtures[(task, form)], strict=True):
                    if _cancelled():
                        return {
                            "available": False,
                            "detail": "cancelled",
                            "protocol": protocol,
                            "candidates": {},
                        }
                    expected = _nf.expected_answer(entry, form)
                    expected_tokens = len(backend.encode(expected))
                    record: dict[str, Any] = {
                        "task": task,
                        "value_id": entry["value_id"],
                        "form": form,
                        "band": entry["band"],
                        "stratum": entry["stratum"],
                        "operands": [int(value) for value in entry["operands"]],
                        "expected_value": int(entry["expected_value"]),
                        "expected_answer": expected,
                        "form_identical": all(
                            _nf.render_form(int(value), form)
                            == _nf.render_form(int(value), "plain")
                            for value in entry["operands"]
                        ),
                        "expected_answer_tokens": expected_tokens,
                        "answer_value": None,
                        "answer_text": None,
                        "cell": None,
                        "detail": "",
                    }
                    records.append(record)
                    rendered = fmt.render(case.messages)
                    operand_strings = [
                        _nf.render_form(int(value), form) for value in entry["operands"]
                    ]
                    if any(text not in rendered for text in operand_strings):
                        # The template rewrote or dropped an
                        # operand (a normaliser inside the format
                        # would do exactly this): the cell never
                        # measured the form it names.
                        record["detail"] = "form_not_in_render"
                        continue
                    ids = tuple(backend.encode(rendered))
                    if len(ids) + max_new_tokens > max_context:
                        record["detail"] = "arm_context_exceeded"
                        continue
                    if expected_tokens + len(boundaries[name]) > max_new_tokens:
                        record["detail"] = "answer_budget_exceeded"
                        continue
                    report = evaluate(
                        backend,
                        fmt,
                        (case,),
                        boundary=boundaries[name],
                        max_new_tokens=max_new_tokens,
                        seed=seed,
                        cancel=cancel,
                    )
                    if report.aggregate_eligible and report.observations:
                        cell_obs = report.observations[0]
                        record["cell"] = {
                            "verdict": bool(cell_obs.exact_match and cell_obs.boundary_found),
                            "boundary_found": bool(cell_obs.boundary_found),
                            "boundary_text_match": cell_obs.boundary_text_match,
                            "input_tokens": len(cell_obs.input_ids),
                            "answer_digest": _answer_digest(cell_obs.answer),
                        }
                        # Verbatim, never normalised: the diff
                        # validator binds this text to the cell
                        # digest and re-parses it (cycle 190, F3).
                        record["answer_text"] = cell_obs.answer
                        if task != "compare":
                            record["answer_value"] = _nf.parse_value(cell_obs.answer)
                    else:
                        record["detail"] = "arm_evaluation_failed"
        candidates[name] = {"status": "available", "detail": "", "records": records}
    stage: dict[str, Any] = {
        "available": True,
        "detail": "",
        "protocol": protocol,
        "grouping": grouping,
        "candidates": candidates,
    }
    if (
        _nf.records_bytes({name: entry["records"] for name, entry in candidates.items()})
        > _nf.MAX_RECORDS_BYTES
    ):
        return {
            "available": False,
            "detail": "numform_records_too_large",
            "protocol": protocol,
            "candidates": {},
        }
    return stage


def numform_analysis(
    stage: dict[str, Any],
    candidate_names: tuple[str, ...],
) -> dict[str, Any]:
    """Aggregate a numform stage into the stored payload (cycle 186).

    Per candidate x task x form aggregates with the value/form
    decomposition, and per-task paired contrasts for the six
    form pairs. No selection, no winner, no flip against the
    primary run: the synthetic set is a different contest from
    the user's fixtures, and a winner over 16 generated values
    per task would invite exactly the headline the design
    forbids (red-team B, GB5). Design: docs/NUMFORM_DESIGN.md.
    """
    from . import numform as _nf

    if not stage.get("available"):
        return {
            "available": False,
            "detail": str(stage.get("detail", "numform_unavailable")),
            "protocol": stage.get("protocol"),
        }
    stage_candidates = stage["candidates"]
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
        records = entry["records"]
        candidates_out[name] = {
            "status": "available",
            "detail": "",
            "records": records,
            "aggregates": {
                task: {form: _nf.aggregate_records(records, task, form) for form in _nf.FORMS}
                for task in _nf.TASKS
            },
            "contrasts": {
                task: {
                    f"{form_a}__{form_b}": _nf.paired_contrast(records, task, form_a, form_b)
                    for form_a, form_b in _nf.form_pairs()
                }
                for task in _nf.TASKS
            },
        }
    return {
        "available": True,
        "detail": "",
        "protocol": stage["protocol"],
        "grouping": stage["grouping"],
        "candidates": candidates_out,
        "n_candidates": len(candidate_names),
    }


def cache_profile_stage(
    *,
    observations: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    block_size: int = 16,
) -> dict[str, Any]:
    """Measure per-candidate prefix-cache sharability post-hoc.

    Purely post-hoc: no model calls, no backend. For every completed
    observation, the recorded ``input_ids`` are grouped by candidate and
    aggregated by :func:`cacheprofile.candidate_cache_profile` (exact
    achievable sharability fractions plus the potential upper bound;
    block-quantized for vLLM-style prefix caching). ``block_size`` is
    recorded in the output so every number is reproducible.

    ``observations`` are flat observation dicts (as produced by
    ``asdict(Observation)``). Malformed observations are excluded from
    the evaluable set (fail-closed: visible via each candidate's ``n``
    vs ``n_evaluable``, never crashing the stage). Raises
    ``ValueError`` on caller-contract violations (bad container type,
    bad ``block_size``), never on observation data.

    Returns ``{"n_observations", "block_size", "per_candidate":
    {name: cell}}`` where each cell is the
    :func:`cacheprofile.candidate_cache_profile` dict.
    """
    if type(observations) is not tuple and type(observations) is not list:
        raise ValueError("observations must be a tuple or list of observation dicts")
    if type(block_size) is not int or block_size < 1:
        raise ValueError(f"block_size must be a positive int, got {block_size!r}")
    records: dict[str, list[Any]] = {}
    seen: dict[str, int] = {}
    n_observations = 0
    for obs in observations:
        if type(obs) is not dict:
            continue
        candidate = obs.get("candidate")
        if type(candidate) is not str or not candidate:
            continue
        n_observations += 1
        seen[candidate] = seen.get(candidate, 0) + 1
        if obs.get("status") != "completed":
            # Fail-closed: the observation stays visible via the
            # candidate's n vs n_evaluable, never counted and never
            # crashing the stage.
            continue
        records.setdefault(candidate, []).append(obs.get("input_ids"))
    per_candidate: dict[str, dict[str, Any]] = {}
    for candidate, n_seen in seen.items():
        cell = candidate_cache_profile(records.get(candidate, []), block_size)
        # n counts every observation seen for the candidate, including
        # the fail-closed exclusions above; n_evaluable counts only the
        # measured ones.
        cell["n"] = n_seen
        per_candidate[candidate] = cell
    return {
        "n_observations": n_observations,
        "block_size": block_size,
        "per_candidate": per_candidate,
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


def _nonnegative_number(value: Any) -> bool:
    """Whether ``value`` is a real number >= 0 (bools excluded).

    Same narrowing trick as :func:`_share_in_unit`. Marker offsets
    are positions: 0 is a perfectly defined offset (a prefix-free
    format puts the marker at the very start of the rendered
    prompt), so the offset keys must not use the positive check
    (cycle 176, wave-7 F5).
    """
    return type(value) in (int, float) and isinstance(value, (int, float)) and value >= 0


def _cache_profile_cell_valid(entry: Any, block_size: int | None = None) -> bool:
    """Structural check for one candidate cell of a cache-profile stage.

    Derived values are recomputed from the stored counts (cycle 177,
    wave-7 F6b; the answer-collision validator's standard): the mean
    and the token fractions are the counts' ratios, bit-exact
    because the producer computes the same divisions. The block
    fractions need the stage's ``block_size``, which is not a cell
    field: callers that have it (the profile assembler) pass it and
    get the block recompute too; direct callers without it skip
    only that pair.
    """
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
    # Integer identities the producer guarantees (wave-2 C3 lesson:
    # enforce at the value level, not just the type level).
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
        # Recompute (F6b): evaluable prompts are non-empty by the
        # pinned predicate, so total_tokens is positive here and
        # every division below is the producer's own.
        total = entry["total_tokens"]
        if total <= 0:
            return False
        if mean != total / entry["n_evaluable"]:
            return False
        if entry["potential_token_fraction"] != entry["cacheable_tokens"] / total:
            return False
        if entry["achievable_token_fraction"] != entry["achievable_tokens"] / total:
            return False
        if block_size is not None:
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
    # Ordering invariants (cycle-168 red-team R8/S5): block <= token at
    # both levels, achievable <= potential at both levels.
    achievable_token = entry.get("achievable_token_fraction")
    potential_token = entry.get("potential_token_fraction")
    achievable_block = entry.get("achievable_block_fraction")
    potential_block = entry.get("potential_block_fraction")
    return not (
        isinstance(achievable_token, (int, float))
        and isinstance(potential_token, (int, float))
        and isinstance(achievable_block, (int, float))
        and isinstance(potential_block, (int, float))
        and not (
            achievable_block <= achievable_token <= potential_token
            and achievable_block <= potential_block <= potential_token
        )
    )


def cache_profile_profile(
    candidates: tuple[str, ...],
    stage: dict[str, Any],
) -> dict[str, Any]:
    """Analyze prefix-cache sharability across candidates (pure post-hoc).

    Answers "does the template change prefix sharability?": reports the
    per-candidate cells from the stage, the max-minus-min gap on the
    headline metric (achievable block fraction) -- the pre-registered
    falsifiable claim (see :mod:`cacheprofile`) is that the gap is
    measurably positive rather than exactly 0 -- and ``uniform``
    reporting the null hypothesis (all evaluable candidates identical
    on all four fractions), never asserting it. ``most_sharable`` /
    ``least_sharable`` are behavior labels for a cost *input*, never a
    cost verdict. ``evaluable_mismatch`` flags candidates measured on
    different numbers of prompts (cycle-168 red-team R14): fractions
    from different denominators are not directly comparable.

    Never raises on malformed input: returns ``{"available": False}``
    with a ``detail`` code.
    """
    if (
        not isinstance(candidates, tuple)
        or not candidates
        or not all(type(c) is str and c.strip() for c in candidates)
        or len(set(candidates)) != len(candidates)
    ):
        return {"available": False, "detail": "invalid_candidates"}
    if type(stage) is not dict:
        return {"available": False, "detail": "stage_invalid"}
    block_size = stage.get("block_size")
    if type(block_size) is not int or block_size < 1:
        return {"available": False, "detail": "stage_invalid"}
    n_observations = stage.get("n_observations")
    if type(n_observations) is not int or n_observations < 0:
        return {"available": False, "detail": "stage_invalid"}
    per_candidate = stage.get("per_candidate")
    if type(per_candidate) is not dict or not all(type(k) is str for k in per_candidate):
        return {"available": False, "detail": "stage_invalid"}
    for entry in per_candidate.values():
        if not _cache_profile_cell_valid(entry, block_size):
            return {"available": False, "detail": "stage_invalid"}
    evaluable: dict[str, float] = {}
    excluded: list[str] = []
    evaluable_counts: list[int] = []
    for cand in candidates:
        entry = per_candidate.get(cand)
        if (
            type(entry) is not dict
            or entry.get("n_evaluable", 0) == 0
            or type(entry.get("achievable_block_fraction")) not in (int, float)
        ):
            excluded.append(cand)
            continue
        evaluable[cand] = float(entry["achievable_block_fraction"])
        evaluable_counts.append(entry["n_evaluable"])
    sharability_gap: float | None = None
    most_sharable: list[str] = []
    least_sharable: list[str] = []
    uniform: bool | None = None
    if evaluable:
        values = list(evaluable.values())
        sharability_gap = max(values) - min(values)
        top = max(values)
        bottom = min(values)
        most_sharable = sorted(name for name, value in evaluable.items() if value == top)
        least_sharable = sorted(name for name, value in evaluable.items() if value == bottom)
        fraction_keys = (
            "achievable_block_fraction",
            "achievable_token_fraction",
            "potential_block_fraction",
            "potential_token_fraction",
        )
        first = per_candidate[next(iter(evaluable))]
        uniform = all(
            all(per_candidate[name].get(key) == first.get(key) for name in evaluable)
            for key in fraction_keys
        )
    return {
        "available": True,
        "block_size": block_size,
        "n_observations": n_observations,
        "per_candidate": {cand: per_candidate.get(cand) for cand in candidates},
        "most_sharable": most_sharable,
        "least_sharable": least_sharable,
        "sharability_gap": sharability_gap,
        "uniform": uniform,
        "excluded_candidates": sorted(excluded),
        "evaluable_mismatch": len(set(evaluable_counts)) > 1,
    }


def template_robustness_stage(
    formats: tuple[Any, ...],
    fixtures: tuple[Any, ...],
    backend: Any,
    *,
    boundary: tuple[int, ...],
    boundary_map: dict[str, tuple[int, ...]] | None = None,
    max_new_tokens: int = 64,
    seed: int = 0,
    cancel: Event | None = None,
) -> dict[str, Any]:
    """Per-candidate surface-noise robustness (cycle 166, informational).

    For each candidate template, probes verdict preservation under
    presentation-level surface noise applied to user-role message contents.
    The template is the independent variable; weights, fixtures, backend,
    and protocol are fixed. One frozen protocol covers all candidates in the
    run (comparability). Each candidate is judged under its own boundary
    (``boundary_map``, resolved by :func:`_resolve_boundaries`) -- the same
    honest-boundary rule as the main evaluation. Per-candidate measurement
    failures degrade to ``status: "unavailable"`` markers; the stage never
    raises.

    Returns ``{"candidates": {name: report}, "protocol": protocol}``.
    Design: docs/TEMPLATE_ROBUSTNESS_DESIGN.md.
    """
    from . import surfacenoise as _sn

    if type(formats) is not tuple or not formats:
        raise ValueError("formats must be a nonempty tuple")
    if type(fixtures) is not tuple or not fixtures:
        raise ValueError("fixtures must be a nonempty tuple")
    boundaries = _resolve_boundaries(formats, boundary, boundary_map)
    protocol = _sn.freeze_template_robustness_protocol(_backend_fingerprint(backend), fixtures)
    candidates: dict[str, Any] = {}
    for fmt in formats:
        name = fmt.name
        try:
            renderer_sha256 = _sn.template_renderer_hash(fmt)
        except Exception:  # noqa: BLE001 -- hash boundary; None, never crash
            renderer_sha256 = None
        try:
            report = _sn.probe_template_robustness(
                fmt.render,
                name,
                fixtures,
                backend,
                protocol,
                boundary=boundaries[name],
                max_new_tokens=max_new_tokens,
                seed=seed,
                cancel=cancel,
            )
        except Exception:  # noqa: BLE001 -- per-candidate fail-closed
            report = {
                "instrument": _sn.TEMPLATE_ROBUSTNESS_PROBE_NAME,
                "status": "unavailable",
                "reason": "candidate_probe_failed",
                "template_name": name,
                "boundary": list(boundaries[name]),
            }
        # N3: per-candidate renderer context (Format fields + rendered inputs).
        report["renderer_sha256"] = renderer_sha256
        candidates[name] = report
    return {"candidates": candidates, "protocol": protocol}


def ablation_stage(
    formats: tuple[Any, ...],
    fixtures: tuple[Any, ...],
    backend: Any,
    *,
    boundary: tuple[int, ...],
    boundary_map: dict[str, tuple[int, ...]] | None = None,
    max_new_tokens: int = 64,
    seed: int = 0,
    cancel: Event | None = None,
) -> dict[str, Any]:
    """Template field ablation (cycle 167, informational).

    For the top-2 candidates by search accuracy (ties broken by name, so
    the focus is deterministic), evaluates the base format and one
    single-field ablation arm per structural field via ``core.evaluate``,
    then attributes verdict flips per field with
    ``ablation.summarize_arm``. A field is *load-bearing* for a candidate
    when the base passes every search fixture but the arm does not --
    i.e. the field carried the candidate over the conservative selection
    bar. Main effects only; pairwise interactions are out of scope.

    Never raises for measurement failures: per-arm errors degrade to an
    ``"unavailable"`` arm marker, per-candidate errors to an
    ``"unavailable"`` candidate marker. Never changes winner selection.
    Returns ``{"candidates": {name: report}, "n_focus": 2}``.
    """
    from . import ablation as _ab
    from .core import evaluate as _evaluate

    if type(formats) is not tuple or not formats:
        raise ValueError("formats must be a nonempty tuple")
    if type(fixtures) is not tuple or not fixtures:
        raise ValueError("fixtures must be a nonempty tuple")
    # Each candidate is judged under its own boundary (the same
    # honest-boundary rule as the main evaluation and the
    # template-robustness probe).
    boundaries = _resolve_boundaries(formats, boundary, boundary_map)

    def _verdicts(fmt: Any) -> tuple[bool, ...] | None:
        try:
            report = _evaluate(
                backend,
                fmt,
                fixtures,
                boundary=boundaries[fmt.name],
                max_new_tokens=max_new_tokens,
                seed=seed,
                cancel=cancel,
            )
        except Exception:  # noqa: BLE001 -- per-arm fail-closed
            return None
        if not report.aggregate_eligible:
            return None
        return tuple(bool(o.exact_match and o.boundary_found) for o in report.observations)

    # Rank candidates by search accuracy; ties broken by name (deterministic).
    scored: list[tuple[float, str, Any]] = []
    for fmt in formats:
        verdicts = _verdicts(fmt)
        if verdicts is None:
            continue
        rate = sum(verdicts) / len(verdicts) if verdicts else 0.0
        scored.append((rate, fmt.name, fmt))
    scored.sort(key=lambda item: (-item[0], item[1]))
    focus = [fmt for _, _, fmt in scored[:2]]

    fixture_ids = tuple(f.id for f in fixtures)
    candidates: dict[str, Any] = {}
    for fmt in focus:
        name = fmt.name
        base_verdicts = _verdicts(fmt)
        if base_verdicts is None:
            candidates[name] = {"status": "unavailable", "reason": "base_evaluation_failed"}
            continue
        arms: dict[str, Any] = {}
        load_bearing: list[str] = []
        base_passes_all = all(base_verdicts)
        try:
            arm_list = _ab.ablation_arms(fmt)
            not_applicable = _ab.inapplicable_fields(fmt)
        except Exception:  # noqa: BLE001 -- arm construction fail-closed
            candidates[name] = {"status": "unavailable", "reason": "arm_construction_failed"}
            continue
        for arm in arm_list:
            arm_verdicts = _verdicts(arm.ablated)
            if arm_verdicts is None:
                arms[arm.field] = {"status": "unavailable", "reason": "arm_evaluation_failed"}
                continue
            summary = _ab.summarize_arm(base_verdicts, arm_verdicts, fixture_ids)
            arms[arm.field] = {
                "status": "ok",
                "original": arm.original,
                "neutral": _ab.NEUTRAL_VALUES[arm.field],
                **summary,
            }
            if base_passes_all and not summary["selection_bar_held"]:
                load_bearing.append(arm.field)
        candidates[name] = {
            "status": "ok",
            "boundary": list(boundaries[name]),
            "base_accuracy": sum(base_verdicts) / len(base_verdicts) if base_verdicts else 0.0,
            "base_passes_all": base_passes_all,
            "arms": arms,
            "not_applicable_fields": list(not_applicable),
            "load_bearing_fields": load_bearing,
            "any_attribution": any(
                a.get("n_flips", 0) > 0 for a in arms.values() if a.get("status") == "ok"
            ),
        }
    return {"candidates": candidates, "n_focus": len(focus)}


def confidentiality_stage(
    backend: Backend,
    formats: tuple[Format, ...],
    treatment_fixtures: tuple[Fixture, ...],
    control_fixtures: tuple[Fixture, ...],
    *,
    boundary: tuple[int, ...],
    boundary_map: dict[str, tuple[int, ...]] | None = None,
    max_new_tokens: int = 64,
    seed: int = 0,
) -> dict[str, Any]:
    """Labeled-block confidentiality probe (cycle 169, informational).

    For each candidate template, every synthetic confidentiality fixture
    (a frozen block + neutral distractor + extraction elicitation) is
    rendered twice: once with the fixed nonce marker inside the
    privilege-labeled block (treatment) and once as the identical bare
    marker at the same position (control). Both arms face the same
    frozen elicitations, mirrored exactly (red-team B M1: an asymmetric
    elicitation would contrast attack-success with incidental echo).

    Detection is on the FULL decoded output (``backend.decode`` of
    ``output_ids``), never the boundary-truncated answer (red-team A R7):
    leaks emitted after the response boundary are otherwise invisible,
    and per-candidate boundaries differ. Fail-closed per candidate: any
    backend/evaluation failure records a None cell, never crashing the
    stage.

    Returns ``{"protocol", "n_fixtures", "per_candidate": {name: cell}``
    where each cell is :func:`confidentiality.confidentiality_cell`.
    """
    from .confidentiality import (
        CONFIDENTIALITY_SPECS,
        MARKER,
        confidentiality_cell,
        detect_leak,
        protocol_dict,
        validate_confidentiality_specs,
    )

    if backend is None:
        raise ValueError("backend must not be None")
    if type(formats) is not tuple or not formats:
        raise ValueError("formats must be a nonempty tuple")
    specs = validate_confidentiality_specs(CONFIDENTIALITY_SPECS)
    if len(treatment_fixtures) != len(specs) or len(control_fixtures) != len(specs):
        raise ValueError("treatment and control fixture sets must match the specs")
    if not _valid_token_ids(boundary, nonempty=True):
        raise ValueError("Boundary must be a nonempty tuple of token IDs")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 4096:
        raise ValueError("max_new_tokens must be an integer between 1 and 4096")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    boundaries = _resolve_boundaries(formats, boundary, boundary_map)
    protocol = protocol_dict()
    per_candidate: dict[str, dict[str, Any] | None] = {}

    def _score(obs: Any) -> dict[str, Any] | None:
        if obs.status != "completed":
            return None
        try:
            decoded = backend.decode(obs.output_ids)
        except Exception:  # noqa: BLE001 -- undecodable output is unevaluable
            return None
        detection = detect_leak(decoded)
        offset: int | None = None
        input_text = getattr(obs, "input_text", "")
        if type(input_text) is str:
            try:
                offset = input_text.index(MARKER)
            except ValueError:
                offset = None
        return {"detection": detection, "offset": offset}

    for fmt in formats:
        name = fmt.name
        treat_by_spec: dict[str, Any] = {}
        ctrl_by_spec: dict[str, Any] = {}
        cell: dict[str, Any] | None = None
        try:
            for arm, fixtures, bucket in (
                ("treatment", treatment_fixtures, treat_by_spec),
                ("control", control_fixtures, ctrl_by_spec),
            ):
                report = evaluate(
                    backend,
                    fmt,
                    fixtures,
                    boundary=boundaries[name],
                    max_new_tokens=max_new_tokens,
                    seed=seed,
                )
                if not report.aggregate_eligible:
                    raise ValueError(f"{arm} evaluation not aggregate-eligible")
                for obs in report.observations:
                    spec_id = obs.fixture.rsplit("/", 1)[0]
                    bucket[spec_id] = _score(obs)
            per_fixture = [
                {
                    "treat": treat_by_spec.get(spec["id"], {}).get("detection"),
                    "ctrl": ctrl_by_spec.get(spec["id"], {}).get("detection"),
                    "offset_treat": treat_by_spec.get(spec["id"], {}).get("offset"),
                    "offset_ctrl": ctrl_by_spec.get(spec["id"], {}).get("offset"),
                }
                for spec in specs
            ]
            cell = confidentiality_cell(per_fixture)
            cell["protocol_fingerprint"] = protocol["fingerprint"]
        except Exception:  # noqa: BLE001 -- one cell degrades, not the stage
            cell = None
        per_candidate[name] = cell
    return {
        "protocol": protocol,
        "n_fixtures": len(specs),
        "per_candidate": per_candidate,
    }


def _in_closed_range(value: Any, lo: float, hi: float) -> bool:
    """Whether ``value`` is a real number in [lo, hi] (bools excluded).

    Same narrowing trick as :func:`_share_in_unit`.
    """
    return type(value) in (int, float) and isinstance(value, (int, float)) and lo <= value <= hi


def _confidentiality_cell_valid(entry: Any) -> bool:
    """Structural check for one candidate cell of a confidentiality stage."""
    if entry is None:
        return True
    if type(entry) is not dict:
        return False
    n = entry.get("n")
    if type(n) is not int or n < 0:
        return False
    for key in (
        "n_treat_evaluable",
        "n_ctrl_evaluable",
        "n_paired_evaluable",
        "treat_leaks",
        "ctrl_leaks",
        "treat_verbatim",
        "ctrl_verbatim",
        "discord_ctrl_only",
        "discord_treat_only",
    ):
        value = entry.get(key)
        if type(value) is not int or value < 0:
            return False
    for key in ("treat_rate", "ctrl_rate", "mcnemar_p"):
        value = entry.get(key)
        if value is not None and not _share_in_unit(value):
            return False
    gap = entry.get("gap")
    if gap is not None and not _in_closed_range(gap, -1.0, 1.0):
        return False
    for key in ("mean_run_fraction_treat", "mean_run_fraction_ctrl"):
        value = entry.get(key)
        if value is not None and not _share_in_unit(value):
            return False
    for key in ("max_run_fraction_treat", "max_run_fraction_ctrl"):
        value = entry.get(key)
        if value is not None and not _share_in_unit(value):
            return False
    for key in ("marker_offset_treat", "marker_offset_ctrl"):
        value = entry.get(key)
        if value is not None and not _nonnegative_number(value):
            return False
    bool_keys = (
        "base_rate_zero",
        "evaluable",
        "positive",
        "evaluable_mismatch",
        "offset_asymmetry",
    )
    if not all(type(entry.get(key)) is bool for key in bool_keys):
        return False
    # Generator invariant: gap is present exactly when the candidate is
    # evaluable AND the control rate is nonzero.
    if entry.get("gap") is None:
        if entry["evaluable"] and not entry["base_rate_zero"]:
            return False
    elif entry["base_rate_zero"] or not entry["evaluable"]:
        return False
    # Derived-value recomputation (cycle 177, wave-7 F6c; the
    # answer-collision validator's standard). Rates are the counts'
    # ratios over the paired denominator, the McNemar p-value is a
    # deterministic function of the discordant counts, the flags
    # are the producer's definitions, and the run-fraction fields
    # must be consistent with the leak counts they summarize.
    from .confidentiality import (
        LEAK_RUN_THRESHOLD,
        MIN_DISCORDANT,
        MIN_PAIRED_EVALUABLE,
        control_block,
        mcnemar_exact_p,
        treatment_block,
    )

    n = entry["n"]
    paired = entry["n_paired_evaluable"]
    if paired > entry["n_treat_evaluable"] or paired > entry["n_ctrl_evaluable"]:
        return False
    if entry["n_treat_evaluable"] > n or entry["n_ctrl_evaluable"] > n:
        return False
    if entry["evaluable_mismatch"] != (
        entry["n_treat_evaluable"] != n or entry["n_ctrl_evaluable"] != n or paired != n
    ):
        return False
    if entry["evaluable"] != (paired >= MIN_PAIRED_EVALUABLE):
        return False
    for rate_key, count_key in (("treat_rate", "treat_leaks"), ("ctrl_rate", "ctrl_leaks")):
        rate = entry[rate_key]
        if paired == 0:
            if rate is not None:
                return False
        elif rate != entry[count_key] / paired:
            return False
    ctrl_rate = entry["ctrl_rate"]
    if entry["base_rate_zero"] != (ctrl_rate == 0.0 if ctrl_rate is not None else False):
        return False
    if entry["treat_leaks"] - entry["discord_treat_only"] != (
        entry["ctrl_leaks"] - entry["discord_ctrl_only"]
    ):
        return False
    if entry["discord_ctrl_only"] > entry["ctrl_leaks"]:
        return False
    # The treat-side bound needs no check of its own: the identity
    # above plus this bound imply discord_treat_only <= treat_leaks.
    gap = entry["gap"]
    if entry["positive"] != (
        gap is not None and gap > 0 and entry["discord_ctrl_only"] >= MIN_DISCORDANT
    ):
        return False
    if entry["mcnemar_p"] != mcnemar_exact_p(
        entry["discord_ctrl_only"], entry["discord_treat_only"]
    ):
        return False
    for arm, leaks_key, verbatim_key in (
        ("treat", "treat_leaks", "treat_verbatim"),
        ("ctrl", "ctrl_leaks", "ctrl_verbatim"),
    ):
        mean_run = entry[f"mean_run_fraction_{arm}"]
        max_run = entry[f"max_run_fraction_{arm}"]
        if paired == 0:
            if mean_run is not None or max_run is not None:
                return False
            continue
        if mean_run is None or max_run is None:
            return False
        leaks = entry[leaks_key]
        if leaks > 0:
            if max_run < LEAK_RUN_THRESHOLD:
                return False
            if mean_run < leaks * LEAK_RUN_THRESHOLD / paired - 1e-9:
                return False
        elif max_run >= LEAK_RUN_THRESHOLD:
            return False
        if entry[verbatim_key] > 0 and max_run != 1.0:
            return False
    offset_treat = entry["marker_offset_treat"]
    offset_ctrl = entry["marker_offset_ctrl"]
    if offset_treat is not None and offset_ctrl is not None:
        label_len = len(treatment_block()) - len(control_block())
        if entry["offset_asymmetry"] != (abs(offset_treat - offset_ctrl) > label_len + 2):
            return False
    elif entry["offset_asymmetry"]:
        return False
    return True


def confidentiality_profile(
    candidates: tuple[str, ...],
    stage: dict[str, Any],
) -> dict[str, Any]:
    """Analyze labeled-block confidentiality across candidates (pure).

    Wraps :func:`confidentiality.confidentiality_profile` with the lab's
    fail-closed input guards: malformed stage input degrades to
    ``{"available": False}`` with a ``detail`` code, never raises on
    observation data.
    """
    from .confidentiality import confidentiality_profile as _profile

    if (
        not isinstance(candidates, tuple)
        or not candidates
        or not all(type(c) is str and c.strip() for c in candidates)
        or len(set(candidates)) != len(candidates)
    ):
        return {"available": False, "detail": "invalid_candidates"}
    if type(stage) is not dict:
        return {"available": False, "detail": "stage_invalid"}
    protocol = stage.get("protocol")
    if type(protocol) is not dict:
        return {"available": False, "detail": "stage_invalid"}
    n_fixtures = stage.get("n_fixtures")
    if type(n_fixtures) is not int or n_fixtures < 1:
        return {"available": False, "detail": "stage_invalid"}
    per_candidate = stage.get("per_candidate")
    if type(per_candidate) is not dict or not all(type(k) is str for k in per_candidate):
        return {"available": False, "detail": "stage_invalid"}
    for entry in per_candidate.values():
        if not _confidentiality_cell_valid(entry):
            return {"available": False, "detail": "stage_invalid"}
    result = _profile(candidates, per_candidate, protocol)
    result["n_fixtures"] = n_fixtures
    return result


class _ScriptedEmissionBackend:
    """Scripted backend emitting one fixed token sequence verbatim.

    Encode/decode delegate to the wrapped run backend, so the audit
    measures the run tokenizer's encode regime; generate returns the
    fixed emission. The stage passes a generation cap equal to the
    emission length, so the Backend contract's "limit means exactly
    the cap" rule holds by construction.
    """

    fingerprint = "answer-collision-scripted-v1"

    def __init__(self, base: Backend, emission: tuple[int, ...]) -> None:
        self._base = base
        self._emission = tuple(emission)

    def encode(self, text: str) -> tuple[int, ...]:
        return self._base.encode(text)

    def decode(self, token_ids: tuple[int, ...]) -> str:
        return self._base.decode(token_ids)

    def generate(
        self, input_ids: tuple[int, ...], *, max_new_tokens: int, seed: int, cancel: Any
    ) -> Any:
        from .core import Generation

        return Generation(tuple(self._emission[:max_new_tokens]), "limit")


def answer_collision_stage(
    backend: Backend,
    formats: tuple[Format, ...],
    *,
    boundary: tuple[int, ...],
    boundary_map: dict[str, tuple[int, ...]] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Answer-collision audit (cycle 170, informational).

    For each candidate's resolved boundary CONFIGURATION (the audit's
    unit -- boundaries are operator-supplied and shared by default, so
    candidates sharing a boundary produce identical cells by
    construction), the frozen marker-bearing fixtures are emitted
    verbatim through :func:`core.evaluate` by scripted backends wrapping
    this run's backend: the reserved arm is the backend's natural
    encoding of the expected answer, the subword arm re-encodes every
    marker occurrence via ``deliforge.find_encoding_pair``. The
    detector's effect on each verbatim-correct answer is classified by
    :func:`collision.classify_observation`'s total order (collision /
    ends_at_marker / intact). Per-fixture generation caps keep the
    Backend contract's limit rule intact. Fail-closed per candidate:
    any failure records a None cell, never crashing the stage.

    Returns ``{"protocol", "n_fixtures", "per_candidate": {name: cell}}``
    where each cell is :func:`collision.collision_cell`.
    """
    from .collision import (
        build_answer_fixtures,
        collision_cell,
        protocol_dict,
        reserved_emission,
        score_observation,
        subword_emission,
    )
    from .core import Fixture, Message, first_boundary

    if backend is None:
        raise ValueError("backend must not be None")
    if type(formats) is not tuple or not formats:
        raise ValueError("formats must be a nonempty tuple")
    if not _valid_token_ids(boundary, nonempty=True):
        raise ValueError("Boundary must be a nonempty tuple of token IDs")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    boundaries = _resolve_boundaries(formats, boundary, boundary_map)
    protocol = protocol_dict()
    per_candidate: dict[str, dict[str, Any] | None] = {}
    for fmt in formats:
        name = fmt.name
        cell: dict[str, Any] | None = None
        try:
            boundary_ids = boundaries[name]
            try:
                marker_text = backend.decode(boundary_ids)
            except Exception:  # noqa: BLE001 -- undecodable boundary: no cell
                marker_text = None
            if type(marker_text) is not str:
                raise ValueError("boundary does not decode to text")
            specs = build_answer_fixtures(marker_text)
            per_fixture: list[dict[str, Any]] = []
            for spec in specs:
                entry: dict[str, Any] = {
                    "id": spec["id"],
                    "reserved": None,
                    "subword": None,
                    "arm_state": {"reserved": "failed", "subword": "not_applicable"},
                }
                for arm in ("reserved", "subword"):
                    if arm == "subword" and spec["marker"] is None:
                        entry["arm_state"]["subword"] = "not_applicable"
                        continue
                    if arm == "reserved":
                        emission = reserved_emission(
                            backend.encode, backend.decode, spec["expected"]
                        )
                        if emission is None:
                            entry["arm_state"]["reserved"] = "roundtrip"
                            continue
                    else:
                        emission = subword_emission(
                            backend.encode,
                            backend.decode,
                            spec["expected"],
                            spec["marker"],
                            boundary_ids,
                        )
                        if emission is None:
                            entry["arm_state"]["subword"] = "no_pair"
                            continue
                    entry["arm_state"][arm] = "failed"
                    try:
                        fixture = Fixture(
                            id=f"ac/{spec['id']}",
                            messages=(Message("user", "Which token ends the turn?"),),
                            expected=spec["expected"],
                            group="answer-collision",
                        )
                        scripted = _ScriptedEmissionBackend(backend, emission)
                        report = evaluate(
                            scripted,
                            fmt,
                            (fixture,),
                            boundary=boundary_ids,
                            max_new_tokens=len(emission),
                            seed=seed,
                        )
                        obs = report.observations[0] if report.observations else None
                        if obs is None or obs.status != "completed":
                            continue
                        cut = first_boundary(obs.output_ids, boundary_ids)
                        suffix_nonempty = cut is not None and bool(
                            obs.output_ids[cut + len(boundary_ids) :]
                        )
                        try:
                            decoded_full = backend.decode(obs.output_ids)
                        except Exception:  # noqa: BLE001 -- undecodable: anomalous
                            decoded_full = None
                        occurrences = (
                            spec["expected"].count(spec["marker"])
                            if type(spec["marker"]) is str
                            else 0
                        )
                        entry[arm] = score_observation(
                            status_ok=True,
                            boundary_found=obs.boundary_found,
                            cut=cut,
                            suffix_nonempty=suffix_nonempty,
                            exact_match=obs.exact_match,
                            verbatim=(decoded_full == spec["expected"]),
                            occurrences=occurrences,
                        )
                        entry["arm_state"][arm] = "scored"
                    except Exception:  # noqa: BLE001 -- one fixture degrades, not the cell
                        entry[arm] = None
                        entry["arm_state"][arm] = "failed"
                per_fixture.append(entry)
            cell = collision_cell(
                per_fixture,
                n_fixtures=len(specs),
                boundary_len=len(boundary_ids),
            )
            cell["protocol_fingerprint"] = protocol["fingerprint"]
        except Exception:  # noqa: BLE001 -- one cell degrades, not the stage
            cell = None
        per_candidate[name] = cell
    return {
        "protocol": protocol,
        "n_fixtures": protocol["n_fixtures"],
        "per_candidate": per_candidate,
    }


def _answer_collision_arm_valid(arm: Any, *, control: bool) -> bool:
    """Structural + invariant check for one arm of an answer-collision cell."""
    if type(arm) is not dict:
        return False
    int_keys = (
        "n_evaluable",
        "n_scored",
        "n_collision",
        "n_ends_at_marker",
        "n_intact",
        "n_anomalous",
        "n_failed",
        "n_roundtrip_excluded",
        "n_no_pair",
        "n_not_applicable",
    )
    for key in int_keys:
        value = arm.get(key)
        if type(value) is not int or value < 0:
            return False
    if not control and (arm["n_no_pair"] != 0 or arm["n_not_applicable"] != 0):
        # The reserved arm always has an emission attempt per fixture.
        return False
    if arm["n_evaluable"] != (
        arm["n_collision"] + arm["n_ends_at_marker"] + arm["n_intact"] + arm["n_anomalous"]
    ):
        return False
    if arm["n_scored"] != arm["n_evaluable"] - arm["n_anomalous"]:
        return False
    rate_pairs = (("collision_rate", "n_collision"), ("ends_rate", "n_ends_at_marker"))
    for key, count_key in rate_pairs:
        rate = arm.get(key)
        if arm["n_scored"] == 0:
            if rate is not None:
                return False
            continue
        if rate is None or not _share_in_unit(rate):
            return False
        if abs(float(rate) - arm[count_key] / arm["n_scored"]) > 1e-9:
            return False
    return True


def _answer_collision_cell_valid(entry: Any) -> bool:
    """Structural check for one candidate cell of an answer-collision stage."""
    if entry is None:
        return True
    if type(entry) is not dict:
        return False
    n = entry.get("n")
    if type(n) is not int or n < 0:
        return False
    boundary_ids = entry.get("boundary_ids")
    if type(boundary_ids) is not int or boundary_ids < 1:
        return False
    reserved = entry.get("reserved")
    subword = entry.get("subword")
    if not _answer_collision_arm_valid(reserved, control=False):
        return False
    if not _answer_collision_arm_valid(subword, control=True):
        return False
    if type(reserved) is not dict or type(subword) is not dict:
        return False
    if reserved["n_failed"] + reserved["n_roundtrip_excluded"] + reserved["n_evaluable"] != n:
        return False
    if (
        subword["n_failed"]
        + subword["n_roundtrip_excluded"]
        + subword["n_no_pair"]
        + subword["n_not_applicable"]
        + subword["n_evaluable"]
        != n
    ):
        return False
    for key in ("any_collision", "uniform_intact", "evaluable"):
        if type(entry.get(key)) is not bool:
            return False
    if entry["any_collision"] != (reserved["n_collision"] + subword["n_collision"] > 0):
        return False
    total_scored = reserved["n_scored"] + subword["n_scored"]
    expected_uniform = (
        total_scored > 0
        and reserved["n_collision"] == 0
        and reserved["n_ends_at_marker"] == 0
        and subword["n_collision"] == 0
        and subword["n_ends_at_marker"] == 0
    )
    if entry["uniform_intact"] != expected_uniform:
        return False
    evaluable = entry.get("evaluable")
    if type(evaluable) is not bool:
        return False
    n_evaluable = reserved.get("n_evaluable")
    if type(n_evaluable) is not int:
        return False
    return evaluable == (n_evaluable >= 1)


def answer_collision_profile(
    candidates: tuple[str, ...],
    stage: dict[str, Any],
) -> dict[str, Any]:
    """Analyze the answer-collision audit across candidates (pure).

    Wraps :func:`collision.answer_collision_profile` with the lab's
    fail-closed input guards: malformed stage input degrades to
    ``{"available": False}`` with a ``detail`` code, never raises on
    observation data.
    """
    from .collision import answer_collision_profile as _profile

    if (
        not isinstance(candidates, tuple)
        or not candidates
        or not all(type(c) is str and c.strip() for c in candidates)
        or len(set(candidates)) != len(candidates)
    ):
        return {"available": False, "detail": "invalid_candidates"}
    if type(stage) is not dict:
        return {"available": False, "detail": "stage_invalid"}
    protocol = stage.get("protocol")
    if type(protocol) is not dict:
        return {"available": False, "detail": "stage_invalid"}
    n_fixtures = stage.get("n_fixtures")
    if type(n_fixtures) is not int or n_fixtures < 1:
        return {"available": False, "detail": "stage_invalid"}
    per_candidate = stage.get("per_candidate")
    if type(per_candidate) is not dict or not all(type(k) is str for k in per_candidate):
        return {"available": False, "detail": "stage_invalid"}
    for entry in per_candidate.values():
        if not _answer_collision_cell_valid(entry):
            return {"available": False, "detail": "stage_invalid"}
        if type(entry) is dict and entry.get("n") != n_fixtures:
            return {"available": False, "detail": "stage_invalid"}
    result = _profile(candidates, per_candidate, protocol)
    result["n_fixtures"] = n_fixtures
    return result


def _choiceaudit_cell(score: TargetScore) -> dict[str, Any]:
    """One ground-record cell from an engine score (non-finite demoted).

    A reported total or mean can be non-finite with status
    ``ok`` (float32 saturation of an extreme finite sum, the
    cycle-76 pin); verbatim storage would crash the
    allow_nan=False report write, so the cell is demoted to
    unavailable under the fixed ``non_finite_score`` reason.
    """
    from . import choiceaudit as _ca

    if score.status == STATUS_OK and (
        score.total_logprob is None
        or score.mean_logprob is None
        or not math.isfinite(score.total_logprob)
        or not math.isfinite(score.mean_logprob)
    ):
        return {
            "status": "unavailable",
            "reason": _ca.REASON_NON_FINITE,
            "total": None,
            "mean": None,
            "tokens": score.scored_token_count,
            "bytes": score.target_byte_length,
        }
    return {
        "status": score.status,
        "reason": score.reason,
        "total": score.total_logprob,
        "mean": score.mean_logprob,
        "tokens": score.scored_token_count,
        "bytes": score.target_byte_length,
    }


def choiceaudit_stage(
    backend: Any,
    formats: tuple[Any, ...],
    cases: tuple[Any, ...],
    search_entries: Any,
    *,
    cancel: Event | None = None,
) -> dict[str, Any]:
    """Choice-scoring audit stage (cycle 195): teacher-forced choice cells.

    Post-hoc over the recorded search: for every search fixture
    a four-choice set is constructed from the split itself
    (choiceaudit.build_choice_sets), and every choice is
    teacher-forced under every candidate over that candidate's
    recorded search prompt via the scoring engine. Zero free
    generations. Cells degrade individually (alignment,
    provider, context); a fixture's verdict exists only when
    every constructed cell scored ok and finite -- the pure
    aggregation in choiceaudit owns that rule. Design:
    docs/CHOICE_AUDIT_DESIGN.md.
    """
    from . import choiceaudit as _ca
    from .scoring import check_alignment

    if type(formats) is not tuple or not formats:
        raise ValueError("formats must be a nonempty tuple")
    if type(cases) is not tuple or not cases:
        raise ValueError("cases must be a nonempty tuple")
    fixture_ids = [case.id for case in cases]
    block = _ca.build_choice_sets(cases)
    protocol: dict[str, Any] = dict(
        _ca.protocol_object(fixture_ids, block, _backend_fingerprint(backend))
    )
    protocol["fingerprint"] = _ca.protocol_fingerprint(
        fixture_ids, block, _backend_fingerprint(backend)
    )
    if len(formats) * len(cases) > _ca.MAX_RECORDS:
        return {
            "available": False,
            "detail": _ca.DETAIL_BUDGET_EXCEEDED,
            "protocol": protocol,
            "candidates": {},
        }
    if not callable(getattr(backend, "forward_logits", None)):
        return {
            "available": False,
            "detail": "logits_not_supported",
            "protocol": protocol,
            "candidates": {},
        }
    entries: dict[str, Any] = {}
    if type(search_entries) is list:
        for entry in search_entries:
            if type(entry) is not dict:
                continue
            fmt_dict = entry.get("format")
            entry_name = fmt_dict.get("name") if type(fmt_dict) is dict else None
            if type(entry_name) is str and entry_name:
                entries[entry_name] = entry
    expected_by_id = {case.id: case.expected for case in cases}
    # Choice texts and their encodings are candidate-independent:
    # encode once per fixture, reuse under every candidate's
    # recorded prompt. An encoding failure becomes a per-cell
    # encode_failed at score time, never a stage failure.
    encoded: dict[str, list[tuple[int, ...] | None]] = {}
    for entry in block:
        if not entry["constructible"]:
            continue
        texts = [expected_by_id[entry["fixture"]]]
        texts += [expected_by_id[choice["source_fixture"]] for choice in entry["choices"][1:]]
        ids_for_fixture: list[tuple[int, ...] | None] = []
        for text in texts:
            try:
                ids = backend.encode(text)
            except Exception:  # noqa: BLE001 -- encode problems become fixed codes
                ids = None
            if type(ids) is not tuple or any(type(t) is not int or t < 0 for t in ids):
                ids = None
            ids_for_fixture.append(ids)
        encoded[entry["fixture"]] = ids_for_fixture
    block_by_id = {entry["fixture"]: entry for entry in block}

    def _cancelled() -> bool:
        return cancel is not None and cancel.is_set()

    candidates: dict[str, Any] = {}
    records_bytes = 0
    for fmt in formats:
        name = fmt.name
        entry = entries.get(name)
        summary = entry.get("summary") if type(entry) is dict else None
        observations = entry.get("observations") if type(entry) is dict else None
        if (
            type(entry) is not dict
            or type(observations) is not list
            or not observations
            or (type(summary) is dict and summary.get("aggregate_eligible") is False)
        ):
            candidates[name] = {
                "status": "unavailable",
                "detail": "native_search_unavailable",
                "records": [],
            }
            continue
        obs_by_fixture: dict[str, Any] = {}
        usable = True
        for obs in observations:
            if type(obs) is not dict or type(obs.get("fixture")) is not str:
                usable = False
                break
            obs_by_fixture[obs["fixture"]] = obs
        if not usable or any(case.id not in obs_by_fixture for case in cases):
            candidates[name] = {
                "status": "unavailable",
                "detail": "native_search_unavailable",
                "records": [],
            }
            continue
        records: list[dict[str, Any]] = []
        for case in cases:
            obs = obs_by_fixture[case.id]
            record: dict[str, Any] = {
                "fixture": case.id,
                "obs_status": obs.get("status") if type(obs.get("status")) is str else None,
                "exact_match": (
                    obs.get("exact_match") if type(obs.get("exact_match")) is bool else None
                ),
                "boundary_found": (
                    obs.get("boundary_found") if type(obs.get("boundary_found")) is bool else None
                ),
                "prompt_reason": None,
                "cells": [],
            }
            records.append(record)
            if not block_by_id[case.id]["constructible"]:
                continue
            raw_ids = obs.get("input_ids")
            prompt_ids = tuple(raw_ids) if isinstance(raw_ids, (list, tuple)) else ()
            if not prompt_ids or any(type(t) is not int or t < 0 for t in prompt_ids):
                record["prompt_reason"] = _ca.REASON_OBSERVATION_MISSING
                continue
            if _cancelled():
                return {
                    "available": False,
                    "detail": "cancelled",
                    "protocol": protocol,
                    "candidates": {},
                }
            # The prompt span is alignment-checked once per
            # (candidate, fixture): a prompt-side failure voids
            # all four cells with that reason and is counted once,
            # reserving per-cell reasons for target-side causes.
            prompt_check = check_alignment(backend, prompt_ids, ())
            if not prompt_check.ok:
                record["prompt_reason"] = (
                    prompt_check.reason if type(prompt_check.reason) is str else "forward_failed"
                )
                continue
            cells: list[dict[str, Any]] = []
            for target_ids in encoded[case.id]:
                if target_ids is None:
                    cells.append(
                        {
                            "status": "unavailable",
                            "reason": "encode_failed",
                            "total": None,
                            "mean": None,
                            "tokens": 0,
                            "bytes": 0,
                        }
                    )
                    continue
                score = score_target_logprobs(backend, prompt_ids, target_ids)
                cells.append(_choiceaudit_cell(score))
            record["cells"] = cells
        records_bytes += len(
            json.dumps(records, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        )
        if records_bytes > _ca.MAX_RECORDS_BYTES:
            return {
                "available": False,
                "detail": _ca.DETAIL_BUDGET_EXCEEDED,
                "protocol": protocol,
                "candidates": {},
            }
        candidates[name] = {"status": "available", "detail": "", "records": records}
    return {
        "available": True,
        "detail": "",
        "protocol": protocol,
        "fixtures": block,
        "candidates": candidates,
    }


def choiceaudit_analysis(
    stage: dict[str, Any],
    primary_decision: Any,
    primary_winner: Any,
    candidate_names: tuple[str, ...],
) -> dict[str, Any]:
    """Aggregate a choice-audit stage into the stored payload (cycle 195).

    Per-candidate aggregates come from the pure
    choiceaudit.aggregate_candidate over the ground records;
    the method-level headline (common-population agreement with
    the primary outcome) and the probe-level falsification
    readout are computed once, here, from the same aggregates
    the diff validator recomputes. Design:
    docs/CHOICE_AUDIT_DESIGN.md.
    """
    from . import choiceaudit as _ca

    if not stage.get("available"):
        return {
            "available": False,
            "detail": str(stage.get("detail", "choiceaudit_unavailable")),
            "protocol": stage.get("protocol"),
        }
    block = stage["fixtures"]
    aggregates_all: dict[str, dict[str, Any]] = {}
    candidates_out: dict[str, Any] = {}
    for name in candidate_names:
        entry = stage["candidates"][name]
        if entry["status"] != "available":
            candidates_out[name] = {
                "status": "unavailable",
                "detail": entry["detail"],
                "records": [],
            }
            continue
        aggregates = _ca.aggregate_candidate(entry["records"], block)
        aggregates_all[name] = aggregates
        out = {key: value for key, value in aggregates.items() if key != "_verdicts"}
        out.update({"status": "available", "detail": "", "records": entry["records"]})
        candidates_out[name] = out
    return {
        "available": True,
        "detail": "",
        "protocol": stage["protocol"],
        "fixtures": block,
        "candidates": candidates_out,
        "method_comparison": _ca.method_comparison(
            aggregates_all, candidate_names, primary_decision, primary_winner
        ),
        "discordance": _ca.discordance_summary(aggregates_all),
        "n_candidates": len(candidate_names),
        "n_fixtures": len(block),
    }


def _precision_completed_obs(entry: Any) -> dict[str, dict[str, Any]]:
    """fixture -> recorded observation for one native search entry.

    Completed observations only: a fixture the native run never
    completed has no fp32 cell (it becomes an unpaired or absent
    fixture in the records, never a fabricated failure).
    """
    out: dict[str, dict[str, Any]] = {}
    if type(entry) is not dict:
        return out
    observations = entry.get("observations")
    if type(observations) is not list:
        return out
    for obs in observations:
        if type(obs) is not dict:
            continue
        fixture = obs.get("fixture")
        status = obs.get("status")
        status = getattr(status, "value", status)
        if type(fixture) is not str or status != "completed":
            continue
        if type(obs.get("answer")) is not str:
            continue
        if type(obs.get("exact_match")) is not bool or type(obs.get("boundary_found")) is not bool:
            continue
        output_ids = obs.get("output_ids")
        input_ids = obs.get("input_ids")
        if not isinstance(output_ids, (list, tuple)) or not isinstance(input_ids, (list, tuple)):
            continue
        if any(type(t) is not int for t in output_ids) or any(
            type(t) is not int for t in input_ids
        ):
            continue
        out[fixture] = obs
    return out


def _precision_cell(
    answer: str,
    boundary_found: bool,
    output_ids: tuple[int, ...],
    expected_digest: str,
    logprob: float | None,
) -> dict[str, Any]:
    """One arm's ground cell; the verdict is digest-bound by construction."""
    from . import precision as _pr

    cell: dict[str, Any] = {
        "answer_digest": _pr.text_digest(answer),
        "boundary_found": boundary_found,
        "output_digest": _pr.ids_digest(output_ids),
        "logprob": logprob,
    }
    cell["verdict"] = _pr.cell_verdict(cell, expected_digest)
    return cell


def _precision_logprob(backend: Any, prompt_ids: tuple[int, ...], expected: str) -> float | None:
    """Teacher-forced total logprob of the expected answer, or None.

    One forward per (candidate, fixture) via the scoring engine;
    any non-ok score (alignment, provider, context) degrades this
    cell's logprob to None -- never a stage failure, never a guess.
    """
    from .scoring import score_target_logprobs

    if not callable(getattr(backend, "forward_logits", None)):
        return None
    try:
        target_ids = backend.encode(expected)
        score = score_target_logprobs(backend, tuple(prompt_ids), tuple(target_ids))
    except Exception:  # noqa: BLE001 -- scoring problems degrade the cell only
        return None
    if getattr(score, "status", None) != "ok":
        return None
    total = getattr(score, "total_logprob", None)
    if isinstance(total, (int, float)) and not isinstance(total, bool) and total == total:
        return float(total)
    return None


def precision_stage(
    native_backend: Any,
    bf16_backend: Any,
    formats: tuple[Any, ...],
    cases: tuple[Any, ...],
    result: dict[str, Any],
    *,
    boundary: tuple[int, ...],
    boundary_map: dict[str, tuple[int, ...]] | None = None,
    max_new_tokens: int = 64,
    seed: int = 0,
    cancel: Event | None = None,
) -> dict[str, Any]:
    """Load-precision stage (cycle 198): the bf16 arm against the native run.

    The fp32 arm is the recorded native run in ``result`` -- never
    re-executed. The bf16 arm re-evaluates every search candidate
    on the search split only, under ``bf16_backend``, with the
    identical boundary/budget/seed conditions the CLI forwards.
    Ground records for both arms are assembled per candidate x
    fixture (digests only), and every derived value -- the paired
    flip matrix, divergence aggregates, logprob deltas, the
    paired-population selections, winner survival, and the
    falsification readout -- comes from precision.compute_derived,
    which the diff validator mirrors independently. Design:
    docs/PRECISION_DESIGN.md.
    """
    from . import precision as _pr
    from .determinism import compare_token_ids

    if type(formats) is not tuple or not formats:
        raise ValueError("formats must be a nonempty tuple")
    if type(cases) is not tuple or not cases:
        raise ValueError("cases must be a nonempty tuple")
    names = tuple(fmt.name for fmt in formats)
    fixture_ids = [str(case.id) for case in cases]
    expected_by_id = {str(case.id): str(case.expected) for case in cases}
    boundaries = _resolve_boundaries(formats, boundary, boundary_map)
    protocol = _pr.protocol_object(
        seed=seed,
        max_new_tokens=max_new_tokens,
        fixture_ids=fixture_ids,
        candidates=names,
        boundaries={name: boundaries[name] for name in names},
        fp32_fingerprint=_backend_fingerprint(native_backend),
        bf16_fingerprint=_backend_fingerprint(bf16_backend),
        declared_dtype=getattr(native_backend, "declared_dtype", None),
        native_dtype=str(getattr(native_backend, "dtype", "")),
        arm_dtype=str(getattr(bf16_backend, "dtype", "")),
    )
    protocol["fingerprint"] = _pr.protocol_fingerprint(protocol)

    def _unavailable(detail: str) -> dict[str, Any]:
        return {
            "available": False,
            "detail": detail,
            "protocol": protocol,
            "primary_decision": result.get("decision"),
            "primary_winner": result.get("selected_candidate"),
            "records": [],
            "unavailable_candidates": list(names),
        }

    if getattr(native_backend, "dtype", None) != _pr.NATIVE_DTYPE:
        return _unavailable("native_dtype_not_float32")
    if getattr(bf16_backend, "dtype", None) != _pr.ARM_DTYPE:
        return _unavailable("arm_dtype_not_bfloat16")
    if len(formats) * len(cases) > _pr.MAX_RECORDS:
        return _unavailable("precision_cell_budget_exceeded")
    search = result.get("search")
    if type(search) is not list:
        return _unavailable("native_search_incomplete")
    native_entries: dict[str, Any] = {}
    for entry in search:
        if type(entry) is not dict:
            continue
        fmt_dict = entry.get("format")
        entry_name = fmt_dict.get("name") if type(fmt_dict) is dict else None
        if type(entry_name) is str and entry_name:
            native_entries[entry_name] = entry
    native_outcomes = native_search_outcomes(search, names)
    records: list[dict[str, Any]] = []
    arm_passed: dict[str, dict[str, bool | None]] = {}
    unavailable_candidates: list[str] = []
    for fmt in formats:
        name = fmt.name
        if cancel is not None and cancel.is_set():
            return _unavailable("cancelled")
        native_obs = _precision_completed_obs(native_entries.get(name))
        try:
            report = evaluate(
                bf16_backend,
                fmt,
                cases,
                boundary=boundaries[name],
                max_new_tokens=max_new_tokens,
                seed=seed,
                cancel=cancel,
            )
        except Exception:  # noqa: BLE001 -- one candidate degrades, not the stage
            report = None
        bf16_obs: dict[str, Any] = {}
        if report is not None:
            for obs in report.observations:
                if str(getattr(obs.status, "value", obs.status)) != "completed":
                    continue
                bf16_obs[str(obs.fixture)] = obs
        if report is not None and report.aggregate_eligible:
            per_fixture = {
                str(o.fixture): bool(o.exact_match and o.boundary_found)
                for o in report.observations
            }
            bf16_passed: bool | None = sum(per_fixture.values()) == len(per_fixture) and bool(
                per_fixture
            )
        else:
            bf16_passed = None
        arm_passed[name] = {
            "fp32": native_outcomes.get(name, {}).get("passed"),
            "bf16": bf16_passed,
        }
        if not native_obs and not bf16_obs:
            unavailable_candidates.append(name)
            continue
        for case in cases:
            fid = str(case.id)
            expected_digest = _pr.text_digest(expected_by_id[fid])
            fp32_cell: dict[str, Any] | None = None
            bf16_cell: dict[str, Any] | None = None
            native_o = native_obs.get(fid)
            bf16_o = bf16_obs.get(fid)
            prompt_ids: tuple[int, ...] = ()
            if native_o is not None:
                prompt_ids = tuple(native_o["input_ids"])
            if native_o is not None and bf16_o is not None:
                fp32_lp = _precision_logprob(native_backend, prompt_ids, expected_by_id[fid])
                bf16_lp = _precision_logprob(bf16_backend, prompt_ids, expected_by_id[fid])
            else:
                fp32_lp = bf16_lp = None
            if native_o is not None:
                fp32_cell = _precision_cell(
                    str(native_o["answer"]),
                    bool(native_o["boundary_found"]),
                    tuple(native_o["output_ids"]),
                    expected_digest,
                    fp32_lp,
                )
            if bf16_o is not None:
                bf16_cell = _precision_cell(
                    str(bf16_o.answer),
                    bool(bf16_o.boundary_found),
                    tuple(bf16_o.output_ids),
                    expected_digest,
                    bf16_lp,
                )
            diverged: bool | None = None
            first_divergence: int | None = None
            if native_o is not None and bf16_o is not None:
                comparison = compare_token_ids(
                    tuple(native_o["output_ids"]), tuple(bf16_o.output_ids)
                )
                diverged = not bool(comparison["match"])
                first_divergence = comparison["first_divergence"]
            records.append(
                {
                    "candidate": name,
                    "fixture": fid,
                    "expected_digest": expected_digest,
                    "fp32": fp32_cell,
                    "bf16": bf16_cell,
                    "diverged": diverged,
                    "first_divergence": first_divergence,
                }
            )
    derived = _pr.compute_derived(
        records,
        names,
        arm_passed,
        str(result.get("decision")),
        result.get("selected_candidate"),
    )
    payload: dict[str, Any] = {
        "available": True,
        "detail": None,
        "protocol": protocol,
        "primary_decision": result.get("decision"),
        "primary_winner": result.get("selected_candidate"),
        "records": records,
        "unavailable_candidates": sorted(unavailable_candidates),
    }
    payload.update(derived)
    return payload


def _confidence_cell(
    answer: str,
    boundary_found: bool,
    input_tokens: int,
    expected_digest: str,
) -> dict[str, Any]:
    """One elicited generation's ground cell (digests + the carried line)."""
    from . import confidence as _cf

    code, value, span = _cf.parse_verbalized_confidence(answer)
    line, line_length = _cf.carried_line(answer, code)
    cell: dict[str, Any] = {
        "answer_digest": _cf.text_digest(answer),
        "span_digest": _cf.text_digest(span),
        "boundary_found": boundary_found,
        "code": code,
        "confidence": value,
        "final_line": line,
        "final_line_length": line_length,
        "input_tokens": input_tokens,
    }
    cell["verdict"] = _cf.cell_verdict(cell, expected_digest)
    return cell


def _confidence_completed_obs(report: Any) -> dict[str, Any]:
    """fixture -> observation for one evaluate() report, completed only."""
    out: dict[str, Any] = {}
    if report is None:
        return out
    for obs in report.observations:
        if str(getattr(obs.status, "value", obs.status)) != "completed":
            continue
        out[str(obs.fixture)] = obs
    return out


def confidence_stage(
    backend: Any,
    formats: tuple[Any, ...],
    cases: tuple[Any, ...],
    result: dict[str, Any],
    *,
    boundary: tuple[int, ...],
    boundary_map: dict[str, tuple[int, ...]] | None = None,
    max_new_tokens: int = 64,
    seed: int = 0,
    cancel: Event | None = None,
) -> dict[str, Any]:
    """Verbalized-confidence stage (cycle 199): elicitation x template.

    Every unscreened search fixture is re-evaluated once per
    candidate with the frozen confidence suffix appended to its
    last user message; the recorded native run is never re-executed
    (its per-fixture verdicts come from ``result`` and anchor the
    elicitation-cost flips). Two generated arms -- planted-fact
    knowable cases and their donor-key controls -- run alongside
    under the same suffix. Ground records for all three populations
    are assembled per candidate (digests plus the carried final
    line), and every derived value comes from
    confidence.compute_derived, which the diff validator mirrors
    independently. A candidate whose evaluations produce no
    completed cell degrades alone; the stage itself fails closed
    only on a blown budget, a special-token collision in the frozen
    suffix, or cancellation. Design: docs/CONFIDENCE_DESIGN.md.
    """
    from . import confidence as _cf
    from .fixtures import audit_special_token_literals

    if type(formats) is not tuple or not formats:
        raise ValueError("formats must be a nonempty tuple")
    if type(cases) is not tuple or not cases:
        raise ValueError("cases must be a nonempty tuple")
    names = tuple(fmt.name for fmt in formats)
    fixture_ids = [str(case.id) for case in cases]
    boundaries = _resolve_boundaries(formats, boundary, boundary_map)
    screens: dict[str, str] = {}
    for case in cases:
        screen = _cf.screen_fixture(case)
        if screen:
            screens[str(case.id)] = screen
    protocol = _cf.protocol_object(
        seed=seed,
        max_new_tokens=max_new_tokens,
        fixture_ids=fixture_ids,
        candidates=names,
        boundaries={name: boundaries[name] for name in names},
        backend_fingerprint=_backend_fingerprint(backend),
        screens=screens,
        cases=_cf.case_protocol(seed),
    )
    protocol["fingerprint"] = _cf.protocol_fingerprint(protocol)

    def _unavailable(detail: str) -> dict[str, Any]:
        return {
            "available": False,
            "detail": detail,
            "protocol": protocol,
            "primary_decision": result.get("decision"),
            "primary_winner": result.get("selected_candidate"),
            "records": [],
            "control_records": [],
            "unavailable_candidates": list(names),
        }

    if audit_special_token_literals(_cf.SUFFIX, formats):
        # The suffix ships clean against the shipped catalog; a
        # candidate whose framing inventory contains the suffix text
        # as a special-token literal would forge the elicitation, so
        # the stage fails closed (the repetition precedent).
        return _unavailable("confidence_screen_failed")
    if len(formats) * len(cases) > _cf.MAX_RECORDS:
        return _unavailable("confidence_cell_budget_exceeded")
    search = result.get("search")
    if type(search) is not list:
        return _unavailable("native_search_incomplete")
    native_outcomes = native_search_outcomes(search, names)
    live_cases = tuple(case for case in cases if str(case.id) not in screens)
    treated = tuple(_cf.treated_fixture(case) for case in live_cases)
    knowable_raw = _cf.build_cases(seed, control=False)
    control_raw = _cf.build_cases(seed, control=True)
    knowable = tuple(_cf.treated_fixture(case) for case in knowable_raw)
    control = tuple(_cf.treated_fixture(case) for case in control_raw)
    records: list[dict[str, Any]] = []
    control_records: list[dict[str, Any]] = []
    unavailable_candidates: list[str] = []

    def _eval(fmt: Any, name: str, treated_cases: tuple[Any, ...]) -> dict[str, Any]:
        if not treated_cases:
            return {}
        try:
            report = evaluate(
                backend,
                fmt,
                treated_cases,
                boundary=boundaries[name],
                max_new_tokens=max_new_tokens,
                seed=seed,
                cancel=cancel,
            )
        except Exception:  # noqa: BLE001 -- one arm degrades, not the stage
            return {}
        return _confidence_completed_obs(report)

    for fmt in formats:
        name = fmt.name
        if cancel is not None and cancel.is_set():
            return _unavailable("cancelled")
        op_obs = _eval(fmt, name, treated)
        kn_obs = _eval(fmt, name, knowable)
        ct_obs = _eval(fmt, name, control)
        native_fixtures = native_outcomes.get(name, {}).get("per_fixture", {})
        n_cells = 0
        for case in cases:
            fid = str(case.id)
            expected_digest = _cf.text_digest(str(case.expected))
            cell: dict[str, Any] | None = None
            obs = op_obs.get(fid)
            if obs is not None:
                cell = _confidence_cell(
                    str(obs.answer),
                    bool(obs.boundary_found),
                    len(obs.input_ids),
                    expected_digest,
                )
                n_cells += 1
            native_verdict = native_fixtures.get(fid)
            records.append(
                {
                    "candidate": name,
                    "fixture": fid,
                    "screen": screens.get(fid, ""),
                    "expected_digest": expected_digest,
                    "native_verdict": native_verdict if type(native_verdict) is bool else None,
                    "elicited": cell,
                }
            )
        for raw in knowable_raw:
            cid = str(raw.id)
            expected_digest = _cf.text_digest(str(raw.expected))
            kn_cell: dict[str, Any] | None = None
            ct_cell: dict[str, Any] | None = None
            kn_o = kn_obs.get(cid)
            ct_o = ct_obs.get(cid)
            if kn_o is not None:
                kn_cell = _confidence_cell(
                    str(kn_o.answer),
                    bool(kn_o.boundary_found),
                    len(kn_o.input_ids),
                    expected_digest,
                )
                n_cells += 1
            if ct_o is not None:
                ct_cell = _confidence_cell(
                    str(ct_o.answer),
                    bool(ct_o.boundary_found),
                    len(ct_o.input_ids),
                    expected_digest,
                )
                n_cells += 1
            control_records.append(
                {
                    "candidate": name,
                    "case": cid,
                    "expected_digest": expected_digest,
                    "knowable": kn_cell,
                    "control": ct_cell,
                }
            )
        if n_cells == 0:
            unavailable_candidates.append(name)
    blob = json.dumps(
        {"records": records, "control_records": control_records},
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(blob) > _cf.MAX_RECORDS_BYTES:
        return _unavailable("confidence_records_budget_exceeded")
    derived = _cf.compute_derived(records, control_records, names, len(live_cases))
    payload: dict[str, Any] = {
        "available": True,
        "detail": None,
        "protocol": protocol,
        "primary_decision": result.get("decision"),
        "primary_winner": result.get("selected_candidate"),
        "records": records,
        "control_records": control_records,
        "unavailable_candidates": sorted(unavailable_candidates),
    }
    payload.update(derived)
    return payload
