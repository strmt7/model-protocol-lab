"""Bounded search followed by one frozen held-out evaluation."""

import json
from bisect import bisect_right
from collections.abc import Callable
from dataclasses import asdict
from hashlib import sha256
from threading import Event
from typing import Any

from .backends import (
    StopOutcome,
    TruncatedPromptBackend,
    emulate_eos_stop,
    emulate_hf_stop_string,
    emulate_vllm_holdback,
)
from .core import Backend, Fixture, Format, Split, _valid_token_ids, evaluate, first_boundary

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
    the semantic is not text-based. Additive: old callers are unaffected.
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
    observations: tuple[dict, ...],
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


def budget_frontier(
    search: list | tuple,
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
    or winner differs from native (``flip_ranges``), and ``budget_robust``.
    When a search entry carries its own ``boundary_token_ids`` (per-candidate
    boundaries, Cycle 2), that entry's curve is derived with its own
    boundary; entries without the field use the ``boundary`` argument.
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
            min_budget: int | None = None
            if (
                obs.get("status") == "completed"
                and obs.get("exact_match") is True
                and obs.get("boundary_found") is True
            ):
                raw = obs.get("output_ids")
                output_ids = tuple(raw) if isinstance(raw, (list, tuple)) else ()
                if output_ids and all(type(token) is int for token in output_ids):
                    cut = first_boundary(output_ids, entry_boundary)
                    if cut is not None:
                        min_budget = cut + len(entry_boundary)
            min_budgets.append(min_budget)
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
    return {
        "max_new_tokens": max_new_tokens,
        "native_decision": native_decision,
        "native_search_decision": native_search[0],
        "native_winner": native_search[1],
        "candidates": candidates,
        "decision_steps": decision_steps,
        "flip_ranges": flip_ranges,
        "budget_robust": not flip_ranges,
    }


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
) -> dict[str, Any]:
    """Re-run search with the prompt truncated mid-token at each depth.

    For ``drop_bytes`` in 1..``max_drop_bytes``, every prompt is truncated
    by that many trailing bytes and re-encoded (see
    ``core.truncate_text_tail``), then the full search re-runs under a
    ``TruncatedPromptBackend``. Reports the depths where the selected winner
    or decision differs from native: boundary sensitivity as a variable, not
    a constant. Depths that cannot be applied (e.g. a cut splitting a UTF-8
    codepoint) record the run's failed decision instead of silently
    skipping.
    """
    if type(max_drop_bytes) is not int or not 1 <= max_drop_bytes <= 8:
        raise ValueError("max_drop_bytes must be an integer between 1 and 8")
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
        "native_decision": native["decision"],
        "native_winner": native_winner,
        "max_drop_bytes": max_drop_bytes,
        "depths": depths,
        "flips_at": [depth["drop_bytes"] for depth in depths if depth["winner_changed"]],
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


def _failure_code(report) -> str | None:
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


def _summary(report):
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
) -> dict:
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
    output: dict[str, Any] = {
        "schema_version": 1,
        "selection_rule": "unique_all_search_pass",
        "measurement": "exact_answer_and_boundary",
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "boundary_token_ids": list(boundary),
        "eos_token_id": eos_id,
        "backend_fingerprint": backend.fingerprint,
        "statistical_confidence": "not_estimated",
        "compute_comparison": "not_equal_compute_certified",
        "decision": "not_started",
        "search": [],
        "held_out": None,
        "selected_candidate": None,
    }

    def finish() -> dict:
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
    if event.is_set():
        output["decision"] = "cancelled"
        return finish()
    if len(winners) != 1:
        output["decision"] = "ambiguous" if winners else "no_candidate"
        return finish()
    selected = winners[0]
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
