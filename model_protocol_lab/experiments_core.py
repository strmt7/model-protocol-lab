import json
import math
from collections.abc import Callable, Mapping
from dataclasses import asdict
from hashlib import sha256
from threading import Event
from typing import Any, cast

from .backends import (
    StopOutcome,
    TokenTruncatedPromptBackend,
    TruncatedPromptBackend,
    emulate_eos_stop,
    emulate_hf_stop_string,
    emulate_vllm_holdback,
)
from .core import (
    Backend,
    EvaluationReport,
    Fixture,
    Format,
    Message,
    Observation,
    Split,
    _valid_token_ids,
    evaluate,
    first_boundary,
)
from .refusals import DETECTOR_ID, OPENERS_SHA256, REFUSAL_OPENERS, analyze_candidate
from .scoring import STATUS_OK, TargetScore, score_target_logprobs
from .wscanonical import MUTATIONS, WsScanBackend, validate_mutations

"""Bounded search followed by one frozen held-out evaluation."""


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


def _changed_rate(spec: dict[str, Any]) -> float | None:
    """Fraction of fixtures actually changed by a variant's operator."""
    changed = spec.get("changed_fixtures")
    total = spec.get("n_fixtures")
    if type(changed) is int and type(total) is int and total > 0:
        return changed / total
    return None


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


_LOGPROB_NOT_SCORED = "not_scored"


_LOGPROB_INCOMPLETE = "observation_incomplete"


_LOGPROB_ENCODE_FAILED = "encode_failed"


def _logprob_unavailable(reason: str, prompt_len: int) -> TargetScore:
    """Build an engine-shaped refusal for caller-side problems (e.g. the
    expected answer could not be encoded): status ``unavailable`` with a
    fixed reason, never an exception.
    """
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
    """Return the plural fixtures."""
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
    native run too, so it carries no truncation-specific fact.
    """
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
    """Return the summary."""
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
        """Return the fixture digest."""
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


def _preflight_estimate(
    backend: Backend,
    formats: tuple[Format, ...],
    split: Split,
    *,
    max_new_tokens: int,
) -> int:
    """Upper bound on a run's token spend, from renders + encodes only.

    Mirrors :func:`model_protocol_lab.core.evaluate`'s per-cell input
    construction exactly (render the candidate's format over the
    fixture's messages, then encode). Search cells are exact: every
    candidate runs every search fixture. The held-out stage runs only
    the winner, unknown at preflight, so each held-out fixture
    contributes its maximum input over candidates. Generation is
    hard-capped per cell at ``max_new_tokens``, so the sum is a true
    upper bound on the run's actual spend. Any cell whose input
    cannot be constructed (render/encode failure, invalid token IDs)
    refuses the capped run with ``run_budget_preflight_failed``: a
    bound that cannot be certified is not a bound.
    """

    def input_len(fmt: Format, case: Fixture) -> int:
        """Input len."""
        try:
            encoded = backend.encode(fmt.render(case.messages))
        except Exception:  # noqa: BLE001 -- sanitization boundary, see pyproject
            raise ValueError("run_budget_preflight_failed") from None
        if not _valid_token_ids(encoded):
            raise ValueError("run_budget_preflight_failed")
        return len(encoded)

    total = 0
    for fmt in formats:
        for case in split.search:
            total += input_len(fmt, case) + max_new_tokens
    for case in split.held_out:
        total += max(input_len(fmt, case) for fmt in formats) + max_new_tokens
    return total


def configuration_digest(
    backend: Backend,
    formats: tuple[Format, ...],
    split: Split,
    *,
    boundary: tuple[int, ...],
    boundary_map: dict[str, tuple[int, ...]] | None = None,
    max_new_tokens: int = 64,
    seed: int = 0,
    eos_id: int | None = None,
) -> str:
    """The configuration identity a run's progress journal binds to.

    Public wrapper over :func:`_configuration_digest` with
    ``run_experiment``'s own defaults, so the CLI can stamp the
    journal header before the run starts and a later ``--resume``
    can prove it is continuing the same configuration (audit
    adjudication D-3: digest + progress schema + package version;
    the budget cap is deliberately NOT part of the identity — the
    resuming invocation's cap governs, enforced by the same
    full-run preflight bound).
    """
    return _configuration_digest(
        backend,
        formats,
        split,
        boundary=boundary,
        boundary_map=boundary_map,
        max_new_tokens=max_new_tokens,
        seed=seed,
        eos_id=eos_id,
    )


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
    max_total_tokens: int | None = None,
    on_observation: Callable[[str, str, Observation], None] | None = None,
    resume_observations: Mapping[tuple[str, str, str], Observation] | None = None,
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

    ``max_total_tokens`` optionally caps the run's total token spend
    (the run budget ledger, RR-14a). The cap is enforced at a single
    preflight refusal point: :func:`_preflight_estimate` computes a
    true upper bound on spend from renders + encodes alone, and a
    run whose bound exceeds the cap is refused with
    ``run_budget_exceeded_preflight`` before any generation — a run
    that starts can therefore never exceed its estimate. Every
    return path carries the ``budget`` ledger (cap, estimate —
    ``None`` when no cap was set, since accounting without
    enforcement does not pay for a preflight pass — actual spend,
    planned vs executed cells).

    ``on_observation`` fires as ``(stage, candidate, observation)``
    once per freshly produced observation (the progress journal's
    write hook). ``resume_observations`` maps ``(stage, candidate,
    fixture)`` to observations a previous, identically configured
    run already produced; those cells are not re-executed, and the
    result gains a ``resumed`` block splitting reused vs generated
    cells. Per-cell seeds are position-independent, so a resumed
    run's result equals the uninterrupted run's apart from that
    block.
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
    if max_total_tokens is not None and (type(max_total_tokens) is not int or max_total_tokens < 1):
        raise ValueError("max_total_tokens must be a positive integer or None")
    if on_observation is not None and not callable(on_observation):
        raise ValueError("on_observation must be callable")
    if resume_observations is not None:
        if not isinstance(resume_observations, Mapping):
            raise ValueError("resume_observations must be a mapping")
        search_ids = {case.id for case in split.search}
        held_out_ids = {case.id for case in split.held_out}
        candidate_names = {fmt.name for fmt in formats}
        for key, obs in resume_observations.items():
            if (
                type(key) is not tuple
                or len(key) != 3
                or key[0] not in ("search", "held_out")
                or key[1] not in candidate_names
                or type(obs) is not Observation
                or obs.candidate != key[1]
                or obs.fixture != key[2]
                or key[2] not in (search_ids if key[0] == "search" else held_out_ids)
            ):
                raise ValueError("resume observations do not match this run's plan")
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
    cells_planned = len(formats) * len(split.search) + len(split.held_out)
    estimated_total_tokens: int | None = None
    if max_total_tokens is not None:
        # The single budget refusal point (RR-14a): the preflight
        # bound is a true upper bound on spend, so a run that passes
        # here can never exceed it, and a run refused here has spent
        # nothing on generation.
        estimated_total_tokens = _preflight_estimate(
            backend, formats, split, max_new_tokens=max_new_tokens
        )
        if estimated_total_tokens > max_total_tokens:
            raise ValueError("run_budget_exceeded_preflight")
    fresh_cells = 0

    def _emit(stage: str, candidate_name: str, observation: Observation) -> None:
        """Count and forward one freshly produced observation."""
        nonlocal fresh_cells
        fresh_cells += 1
        if on_observation is not None:
            on_observation(stage, candidate_name, observation)

    def _priors_for(stage: str, candidate_name: str) -> dict[str, Observation] | None:
        """This stage+candidate's resume priors, keyed by fixture ID."""
        if resume_observations is None:
            return None
        return {
            fixture: obs
            for (prior_stage, prior_candidate, fixture), obs in resume_observations.items()
            if prior_stage == stage and prior_candidate == candidate_name
        }

    _tracking = on_observation is not None or resume_observations is not None

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
        """Return the finish."""
        output["failure_taxonomy"] = failure_taxonomy(output, eos_id)
        # The budget ledger rides the same every-return-path discipline:
        # actuals are summed from the stored observations, so a
        # cancelled or failed run still reports what it spent.
        actual_total_tokens = 0
        cells_executed = 0
        observation_lists: list[Any] = [entry["observations"] for entry in output["search"]]
        held = output["held_out"]
        if type(held) is dict:
            observation_lists.append(held["observations"])
        for observations in observation_lists:
            for obs in observations:
                actual_total_tokens += len(obs["input_ids"]) + len(obs["output_ids"])
                cells_executed += 1
        output["budget"] = {
            "max_total_tokens": max_total_tokens,
            "estimated_total_tokens": estimated_total_tokens,
            "actual_total_tokens": actual_total_tokens,
            "cells_planned": cells_planned,
            "cells_executed": cells_executed,
            "estimate_includes_held_out": True,
        }
        if resume_observations is not None:
            # Every stored observation was either reused from the
            # journal or freshly generated (and counted by _emit).
            output["resumed"] = {
                "cells_reused": cells_executed - fresh_cells,
                "cells_generated": fresh_cells,
            }
        return output

    winners = []
    for fmt in formats:
        candidate_boundary = boundaries[fmt.name]

        def _search_emit(observation: Observation, _name: str = fmt.name) -> None:
            """Forward one search observation to the journal hook."""
            _emit("search", _name, observation)

        report = evaluate(
            backend,
            fmt,
            split.search,
            boundary=candidate_boundary,
            max_new_tokens=max_new_tokens,
            seed=seed,
            cancel=event,
            on_observation=_search_emit if _tracking else None,
            prior_observations=_priors_for("search", fmt.name),
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
    if resume_observations is not None:
        # Held-out cells exist only for the candidate the search
        # stage actually selected; a journal naming any other
        # candidate (or a held-out stage with no winner) cannot
        # have come from this configuration's run.
        held_out_priors = _priors_for("held_out", selected.name) if selected else None
        any_held_out = any(key[0] == "held_out" for key in resume_observations)
        if any_held_out and not held_out_priors:
            raise ValueError("resume_stage_order_violation")
    if selected is None:
        output["decision"] = decision
        return finish()
    output["selected_candidate"] = selected.name
    selected_boundary = boundaries[selected.name]
    output["selected_boundary_token_ids"] = list(selected_boundary)

    def _held_out_emit(observation: Observation) -> None:
        """Forward one held-out observation to the journal hook."""
        _emit("held_out", selected.name, observation)

    held = evaluate(
        backend,
        selected,
        split.held_out,
        boundary=selected_boundary,
        max_new_tokens=max_new_tokens,
        seed=seed,
        cancel=event,
        on_observation=_held_out_emit if _tracking else None,
        prior_observations=_priors_for("held_out", selected.name),
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


def batch_stage(
    backend: Backend,
    prompts: dict[str, tuple[int, ...]],
    *,
    max_new_tokens: int = 64,
    seed: int = 0,
    cancel: Event | None = None,
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

    ``cancel`` is the run's shared cancellation event (ORCH-02): the
    stage threads that one event into every ``generate`` /
    ``generate_batch`` call -- never a private event -- and checks it
    before each solo generation and before the batch call. Once set,
    the stage records ``{"available": False, "detail": "cancelled"}``
    and starts no further work.

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

    event = cancel if cancel is not None else Event()
    ordered_ids = sorted(clean_prompts)
    solo: dict[str, tuple[int, ...]] = {}
    per_prompt: dict[str, dict[str, Any]] = {}
    for pid in ordered_ids:
        if event.is_set():
            return {"available": False, "detail": "cancelled"}
        try:
            generation = backend.generate(
                clean_prompts[pid],
                max_new_tokens=max_new_tokens,
                seed=seed,
                cancel=event,
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
        if event.is_set():
            return {"available": False, "detail": "cancelled"}
        try:
            batched_raw = generate_batch(
                tuple(clean_prompts[pid] for pid in evaluable),
                max_new_tokens=max_new_tokens,
                seed=seed,
                cancel=event,
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


def _fixture_with_content(fixture: Any, content: str) -> Any:
    """A copy of ``fixture`` whose target message carries ``content``."""
    from . import repetition as _rep

    index = _rep.target_index(fixture)
    messages = tuple(
        Message(role=m.role, content=content if i == index else m.content)
        for i, m in enumerate(fixture.messages)
    )
    return Fixture(id=fixture.id, messages=messages, expected=fixture.expected, group=fixture.group)


def _answer_digest(value: Any) -> str | None:
    """SHA-256 of an answer string; None when there is no answer."""
    if type(value) is not str:
        return None
    return sha256(value.encode("utf-8")).hexdigest()


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


def _in_closed_range(value: Any, lo: float, hi: float) -> bool:
    """Whether ``value`` is a real number in [lo, hi] (bools excluded).

    Same narrowing trick as :func:`_share_in_unit`.
    """
    return type(value) in (int, float) and isinstance(value, (int, float)) and lo <= value <= hi


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
        """Initialise the scripted-emission backend over ``base``."""
        self._base = base
        self._emission = tuple(emission)

    def encode(self, text: str) -> tuple[int, ...]:
        """Encode the complete rendered prompt into raw token IDs."""
        return self._base.encode(text)

    def decode(self, token_ids: tuple[int, ...]) -> str:
        """Decode token IDs without dropping special tokens."""
        return self._base.decode(token_ids)

    def generate(
        self, input_ids: tuple[int, ...], *, max_new_tokens: int, seed: int, cancel: Any
    ) -> Any:
        """Generate exactly the cap unless cancelled or failed."""
        from .core import Generation

        return Generation(tuple(self._emission[:max_new_tokens]), "limit")
