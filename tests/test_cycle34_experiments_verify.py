"""Differential verification of ``experiments.py`` internals (cycle 34).

Target: the ``model_protocol_lab.experiments`` internals that cycle 21 did
*not* cover (cycle 21 verified the ``run_experiment`` orchestration: search
loop sequencing, winner rule, held-out gating, decision strings, result
assembly, and the taxonomy call). This file verifies everything else:

- ``classify_observation`` / ``failure_taxonomy`` (failure taxonomy)
- ``budget_frontier`` (token-budget robustness derivation)
- ``_resolve_boundaries`` / ``candidate_fingerprint`` (candidate identity)
- ``_stop_cell`` / ``stop_matrix`` (stop-semantics matrix)
- ``_logprob_unavailable`` / ``_score_expected_answer`` /
  ``_aggregate_fixture_scores`` / ``_interpret_logprob_candidate`` /
  ``_diagnostic_observation`` / ``logprob_diagnostic`` (logprob diagnostic)
- ``prompt_sensitivity_sweep`` (mid-token prompt-truncation sweep)
- seed handling through the per-candidate loop (``run_experiment`` and the
  sweep pass one experiment seed through; ``evaluate`` derives per-fixture
  request seeds from it)
- ``_summary`` / ``_failure_code`` behavioral pins (no docstrings; pinned
  from the code and noted below)

Method: an INDEPENDENT oracle reimplements each function from its
docstring only -- different structure and naming, sharing no branching
helpers with the implementation. Logic-free leaves are shared exactly as
in cycle 21 (``first_boundary``, ``_valid_token_ids``, ``asdict``,
``sha256``, the ``emulate_*`` stop-semantic ports, ``score_target_logprobs``,
``FAILURE_MODES``, ``STOP_SEMANTICS``, the fixed diagnostic vocabulary).
The oracle is fuzz-compared against the implementation on many generated
inputs (``random.Random`` with string seeds), using FAKE backends only --
no model, no network.

DOCSTRING-VS-CODE DIVERGENCES (code behavior pinned, source untouched):

D1. (CYCLE 34 NOTE, FIXED IN CYCLE 34.) ``_stop_cell``'s docstring used
    to say ``stop_text`` is "``None`` when the stop did not fire or the
    semantic is not text-based" -- but when the outcome is *unavailable*
    the returned dict has NO ``stop_text`` key at all (the early
    ``{"available": False, "detail": ...}`` return). The docstring was
    corrected in cycle 34. The same holds
    for ``stop_matrix``'s ``observation_incomplete`` cells. The oracle
    mirrors the code (key absent), and ``TestStopMatrix.test_unavailable_cells_omit_stop_text``
    pins it.
D2. The pass criterion is checked with different strictness in two places:
    ``classify_observation`` uses truthiness (``obs.get("exact_match") and
    obs.get("boundary_found")``) while ``budget_frontier`` requires
    ``exact_match is True`` / ``boundary_found is True``. An observation
    with ``exact_match=1`` (truthy, not ``True``) classifies as a pass but
    earns no budget in the frontier. Pinned by
    ``TestRedTeam.test_pass_strictness_divergence``; the oracle mirrors
    each side exactly.
D3. ``failure_taxonomy`` silently overwrites duplicate candidate names
    (last entry wins) while ``budget_frontier`` raises ``ValueError`` on
    duplicates. The taxonomy docstring documents the non-dict
    entry/observation asymmetry but not the duplicate-name behavior.
    Pinned by ``TestRedTeam.test_taxonomy_duplicate_names_last_wins``.
D4. ``stop_matrix`` assumes recorded observations carry tuple/list
    ``input_ids``/``output_ids``: a truthy non-iterable (e.g. ``5``)
    raises ``TypeError`` from ``tuple(...)``. ``budget_frontier`` is
    lenient instead: non-list/tuple ``output_ids`` silently become ``()``
    (no budget, no error). Pinned by
    ``TestRedTeam.test_stop_matrix_noniterable_ids_raise`` and
    ``TestRedTeam.test_budget_frontier_nonsequence_ids_silent``.
D5. ``run_experiment`` itself has no docstring (noted in cycle 21); the
    "mutation application" and "seed handling" internals named in the brief
    map to ``TruncatedPromptBackend`` (sweep), ``_resolve_boundaries``
    (per-candidate boundaries), and the single experiment ``seed`` passed
    to every ``evaluate`` call, from which ``evaluate`` derives per-fixture
    request seeds via ``sha256(f"{seed}\\0{case.id}")``.

Sensitivity: the ``/tmp/mut34/run_mutants.py`` runner (kept out of the
repo) applies 15 source mutants to a throwaway copy of the package and
re-runs this file; every mutant must be caught (suite goes red).
"""

import json
import math
import random
import unittest
from bisect import bisect_right
from dataclasses import asdict
from hashlib import sha256

from model_protocol_lab import experiments as X
from model_protocol_lab.backends import StopOutcome
from model_protocol_lab.backends import emulate_eos_stop as _leaf_emulate_eos
from model_protocol_lab.backends import emulate_hf_stop_string as _leaf_emulate_hf
from model_protocol_lab.backends import emulate_vllm_holdback as _leaf_emulate_vllm
from model_protocol_lab.core import (
    EvaluationReport,
    Fixture,
    Format,
    Generation,
    Message,
    Observation,
    Split,
    _valid_token_ids,
    first_boundary,
)
from model_protocol_lab.experiments import (
    FAILURE_MODES,
    LOGPROB_DERAILED_MEAN_THRESHOLD,
    LOGPROB_UNKNOWN_MEAN_THRESHOLD,
    STOP_SEMANTICS,
)
from model_protocol_lab.scoring import STATUS_OK, TargetScore

# ---------------------------------------------------------------------------
# Independent oracles (written from the docstrings, not the code)
# ---------------------------------------------------------------------------


def oracle_classify(obs, eos_id):
    """Reimplementation of classify_observation from its docstring.

    Priority: pass -> backend failure -> cancelled -> boundary-text-mismatch
    -> eos-without-boundary -> no-boundary-in-N. ValueError on non-dict.
    """
    if type(obs) is not dict:
        raise ValueError("observation must be a dict")
    if obs.get("exact_match") and obs.get("boundary_found"):
        return None
    if obs.get("status") == "failed" or obs.get("raw_reason") == "error":
        return "backend-error"
    if obs.get("status") == "cancelled":
        return "backend-error"
    if not obs.get("boundary_found"):
        if obs.get("boundary_text_match") is True:
            return "boundary-text-mismatch"
        if eos_id is not None and eos_id in tuple(obs.get("output_ids") or ()):
            return "eos-without-boundary"
        return "no-boundary-in-N"
    return "no-boundary-in-N"


def oracle_taxonomy(result, eos_id):
    """Reimplementation of failure_taxonomy from its docstring."""
    if type(result) is not dict:
        raise ValueError("result must be a dict")
    per_candidate = {}
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
                mode = oracle_classify(obs, eos_id)
                if mode is not None:
                    counts[mode] += 1
        per_candidate[name] = counts
    held_out_counts = None
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


def oracle_budget_frontier(search, *, boundary, max_new_tokens, native_decision, native_winner):
    """Reimplementation of budget_frontier from its docstring's derivation."""
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
    names = []
    per_candidate_min = {}
    fixture_ids = []
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
        entry_boundary = entry.get("boundary_token_ids")
        if entry_boundary is None:
            entry_boundary = boundary
        elif isinstance(entry_boundary, list):
            entry_boundary = tuple(entry_boundary)
        if not _valid_token_ids(entry_boundary, nonempty=True):
            raise ValueError("search entry boundary_token_ids must be a nonempty ID tuple")
        min_budgets = []
        ids = []
        for obs in observations:
            if type(obs) is not dict:
                raise ValueError("observations must be dicts")
            ids.append(str(obs.get("fixture")))
            min_budget = None
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
        native_search = ("selected", native_winner)
    else:
        if native_winner is not None:
            raise ValueError("native_winner must be None without a selected candidate")
        native_search = (
            ("ambiguous", None) if native_decision == "ambiguous" else ("no_candidate", None)
        )

    sorted_mins = {
        name: sorted(mb for mb in mins if mb is not None)
        for name, mins in per_candidate_min.items()
    }

    def pass_count(name, budget):
        return bisect_right(sorted_mins[name], budget)

    candidates = {}
    change_budgets = set()
    for name in names:
        min_budgets = per_candidate_min[name]
        marks = sorted({mb for mb in min_budgets if mb is not None and 1 <= mb <= max_new_tokens})
        steps = []
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

    def decision_at(budget):
        winners = [name for name in names if pass_count(name, budget) == n_fixtures]
        if len(winners) == 1:
            return ("selected", winners[0])
        if not winners:
            return ("no_candidate", None)
        return ("ambiguous", None)

    decision_steps = []
    previous_decision = None
    for budget in sorted(change_budgets):
        current = decision_at(budget)
        if current != previous_decision:
            decision_steps.append([budget, current[0], current[1]])
            previous_decision = current
    ordered = sorted(change_budgets)
    flip_ranges = []
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
        "safe_budget_floor": (
            1
            if not flip_ranges
            else (flip_ranges[-1][1] + 1 if flip_ranges[-1][1] < max_new_tokens else None)
        ),
    }


def oracle_resolve_boundaries(formats, boundary, boundary_map):
    """Reimplementation of _resolve_boundaries from its docstring."""
    if boundary_map is None:
        return {fmt.name: boundary for fmt in formats}
    if type(boundary_map) is not dict:
        raise ValueError("boundary_map must be a dict or None")
    names = {fmt.name for fmt in formats}
    resolved = {}
    for name, ids in boundary_map.items():
        if type(name) is not str or name not in names:
            raise ValueError(f"boundary_map names an unknown candidate: {name!r}")
        if not _valid_token_ids(ids, nonempty=True):
            raise ValueError(f"boundary_map boundary for {name!r} must be a nonempty tuple")
        resolved[name] = ids
    for fmt in formats:
        resolved.setdefault(fmt.name, boundary)
    return resolved


def oracle_fingerprint(fmt):
    """Reimplementation of candidate_fingerprint from its docstring."""
    payload = asdict(fmt)
    if not payload["bos"]:
        payload["repeat_bos"] = False
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + sha256(text.encode("utf-8")).hexdigest()


def oracle_stop_cell(backend, fixture, boundary, outcome, stop_text=None):
    """Reimplementation of _stop_cell from its docstring (mirrors D1)."""
    if not outcome.available:
        return {"available": False, "detail": outcome.detail}
    truncated = outcome.truncated_ids
    cut = first_boundary(truncated, boundary)
    answer_ids = truncated if cut is None else truncated[:cut]
    try:
        answer = backend.decode(answer_ids)
    except Exception:  # noqa: BLE001 -- oracle mirrors the implementation's degrade path
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


def oracle_stop_matrix(
    backend,
    candidate_name,
    fixtures,
    observations,
    *,
    boundary,
    eos_id=None,
    token_bytes=None,
    vllm_include_stop_str_in_output=False,
    vllm_min_tokens=0,
):
    """Reimplementation of stop_matrix from its docstring.

    The ``emulate_*`` ports are shared leaves (documented behavior verified
    elsewhere); everything else -- validation, alignment, completeness
    gating, disposition mapping, flips, row assembly -- is reimplemented.
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
        boundary_text = backend.decode(boundary)
    except Exception:  # noqa: BLE001 -- oracle mirrors the degrade path
        boundary_text = None
    if type(boundary_text) is not str or not boundary_text:
        boundary_text = None
    rows = []
    for fixture, obs in zip(fixtures, observations, strict=True):
        if obs.get("fixture") != fixture.id:
            raise ValueError("Observation/fixture alignment mismatch")
        output_ids = tuple(obs.get("output_ids") or ())
        prompt_ids = tuple(obs.get("input_ids") or ())
        complete = obs.get("status") == "completed" and all(
            type(token) is int for token in output_ids
        )
        native_pass = bool(obs.get("exact_match") and obs.get("boundary_found"))
        cells = {}
        if complete:
            outcomes = {
                "hf_stop_string": _leaf_emulate_hf(
                    backend, prompt_ids, output_ids, boundary_text or "", token_bytes=token_bytes
                ),
                "vllm_holdback": _leaf_emulate_vllm(
                    backend,
                    output_ids,
                    boundary_text or "",
                    include_stop_str_in_output=vllm_include_stop_str_in_output,
                    min_tokens=vllm_min_tokens,
                ),
                "eos_appended": _leaf_emulate_eos(output_ids, eos_id),
            }
            dispositions = {
                "hf_stop_string": "kept",
                "vllm_holdback": "kept" if vllm_include_stop_str_in_output else "stripped",
                "eos_appended": None,
            }
            for semantic, outcome in outcomes.items():
                cells[semantic] = oracle_stop_cell(
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


def oracle_aggregate(scores):
    """Reimplementation of _aggregate_fixture_scores from its docstring."""
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
    total_out = None
    mean_out = None
    if scored_fixtures and with_total == scored_fixtures and tokens:
        total_out = total
        mean_out = total / tokens
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


def oracle_interpret_kind(fixture_scores):
    """The interpretation *kind* from _interpret_logprob_candidate's docstring.

    Only the kind is derived from the docstring bands (the prose is pinned
    separately); the docstring fixes the bands and the selection of which
    fixtures classify.
    """
    failures = [s for s in fixture_scores if not s["native_pass"]]
    scored = [s for s in failures if s["status"] == "ok" and type(s["mean_logprob"]) is float]
    if not failures:
        return "no_failures"
    if not scored:
        return "unavailable"
    highs = [s for s in scored if s["mean_logprob"] >= LOGPROB_DERAILED_MEAN_THRESHOLD]
    lows = [s for s in scored if s["mean_logprob"] < LOGPROB_UNKNOWN_MEAN_THRESHOLD]
    if highs and not lows:
        return "greedy_derailed"
    if lows and not highs:
        return "weights_dont_know"
    if highs and lows:
        return "mixed"
    return "indeterminate"


# Fixed diagnostic vocabulary (logic-free constants, shared like FAILURE_MODES).
_NOT_SCORED = X._LOGPROB_NOT_SCORED
_INCOMPLETE = X._LOGPROB_INCOMPLETE
_ENCODE_FAILED = X._LOGPROB_ENCODE_FAILED


def oracle_score_expected(backend, prompt_ids, expected):
    """Reimplementation of _score_expected_answer from its docstring."""
    try:
        target_ids = backend.encode(expected)
    except Exception:  # noqa: BLE001 -- oracle mirrors the fixed-code path
        return X._logprob_unavailable(_ENCODE_FAILED, len(prompt_ids))
    if type(target_ids) is not tuple or any(type(t) is not int for t in target_ids):
        return X._logprob_unavailable(_ENCODE_FAILED, len(prompt_ids))
    return X.score_target_logprobs(backend, prompt_ids, target_ids)


def oracle_diagnostic_observation(backend, obs, expected_by_id):
    """Reimplementation of _diagnostic_observation from its docstring."""
    fixture_id = obs.get("fixture") if type(obs) is dict else None
    label = fixture_id if type(fixture_id) is str else "?"
    native_pass = type(obs) is dict and bool(obs.get("exact_match") and obs.get("boundary_found"))
    not_scored = {
        "fixture": label,
        "status": _NOT_SCORED,
        "reason": _INCOMPLETE,
        "total_logprob": None,
        "mean_logprob": None,
        "scored_token_count": 0,
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
    score = oracle_score_expected(backend, prompt_ids, expected_by_id[fixture_id])
    scored = {
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


def oracle_logprob_diagnostic(backend, result, fixtures):
    """Reimplementation of logprob_diagnostic from its docstring.

    The engine (``score_target_logprobs``) is a shared leaf; the candidate
    loop, the availability gating, the payload shape, the aggregates and the
    interpretation *kinds* are reimplemented. ``method`` prose and
    interpretation prose are pinned as behavioral facts, not oracle-derived.
    """
    if type(result) is not dict or type(result.get("search")) is not list:
        raise ValueError("result must be a run_experiment result dict")
    if isinstance(fixtures, list):
        fixtures = tuple(fixtures)
    if type(fixtures) is not tuple or any(type(item) is not Fixture for item in fixtures):
        raise ValueError("fixtures must be a tuple of Fixture")
    expected_by_id = {item.id: item.expected for item in fixtures}
    payload = {
        "available": False,
        "detail": None,
        # Method prose is pinned as a behavioral fact, not oracle-derived;
        # the test substitutes the implementation's string before comparing.
        "method": None,
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
        fixture_scores = []
        engine_scores = []
        for obs in observations:
            score_dict, engine_score = oracle_diagnostic_observation(backend, obs, expected_by_id)
            fixture_scores.append(score_dict)
            if engine_score is not None:
                engine_scores.append(engine_score)
        aggregate = oracle_aggregate(engine_scores)
        aggregate["native_passes"] = sum(1 for s in fixture_scores if s["native_pass"])
        aggregate["native_failures"] = sum(1 for s in fixture_scores if not s["native_pass"])
        aggregate["unavailable_fixtures"] = sum(
            1 for s in fixture_scores if s["status"] in ("unavailable", _NOT_SCORED)
        )
        payload["candidates"].append(
            {
                "name": name,
                "fixture_scores": fixture_scores,
                "aggregate": aggregate,
                "interpretation_kind": oracle_interpret_kind(fixture_scores),
            }
        )
    if not payload["candidates"]:
        payload["detail"] = "no_scorable_observations"
        return payload
    payload["available"] = True
    return payload


# ---------------------------------------------------------------------------
# Fake backends (no model, no network)
# ---------------------------------------------------------------------------


class ByteBackend:
    """Deterministic byte-level fake: encode = UTF-8 bytes, decode = reverse."""

    fingerprint = "fake-byte-backend"
    vocab_size = 256

    def encode(self, text):
        if type(text) is not str:
            raise ValueError("encode needs str")
        return tuple(text.encode("utf-8"))

    def decode(self, token_ids):
        return bytes(tuple(token_ids)).decode("utf-8")

    def generate(self, input_ids, *, max_new_tokens, seed, cancel):
        raise AssertionError("ByteBackend.generate must be overridden")


class ScriptedBackend(ByteBackend):
    """generate() answers from a per-test script; records request seeds."""

    fingerprint = "fake-scripted-backend"

    def __init__(self, script):
        # script: callable(input_text, seed) -> (ok: bool)
        self._script = script
        self.seeds = []
        self.answer_text = "YES"

    def generate(self, input_ids, *, max_new_tokens, seed, cancel):
        self.seeds.append(seed)
        text = self.decode(input_ids)
        ok = self._script(text, seed)
        ids = self.encode(self.answer_text) + (7, 8) if ok else self.encode("NOPE")
        ids = ids[:max_new_tokens]
        ids = ids + (0,) * (max_new_tokens - len(ids))
        return Generation(token_ids=ids, reason="limit", error=None)


class LogprobBackend(ByteBackend):
    """Byte backend with a scripted forward_logits for the diagnostic."""

    fingerprint = "fake-logprob-backend"

    def __init__(self, mode="boost"):
        self.mode = mode

    def forward_logits(self, input_ids):
        import torch

        seq = tuple(input_ids)
        rows, cols = len(seq), 256
        logits = torch.zeros((rows, cols), dtype=torch.float32)
        if self.mode == "boost":
            # Next-token prediction: row t-1 predicts token t with logit 6.
            for row in range(rows - 1):
                logits[row, seq[row + 1]] = 6.0
        return logits


class ExplodingDecodeBackend(ByteBackend):
    """decode raises (or returns non-str) to exercise degrade paths."""

    def __init__(self, mode="raise"):
        self.mode = mode

    def decode(self, token_ids):
        if self.mode == "raise":
            raise RuntimeError("decode exploded")
        return b"not-a-str"


# ---------------------------------------------------------------------------
# Fuzz helpers
# ---------------------------------------------------------------------------


def _rng(tag, index):
    return random.Random(f"cycle34:{tag}:{index}")  # noqa: S311 -- deterministic test seeding


def _rand_ids(rng, lo=0, hi=12, vmax=250):
    return tuple(rng.randrange(0, vmax) for _ in range(rng.randrange(lo, hi)))


def _rand_obs_dict(rng, fixture_id, boundary, eos_id, p_pass=0.35):
    """Random observation dict shaped like asdict(Observation)."""
    status = rng.choice(["completed"] * 6 + ["failed", "cancelled", "weird"])
    exact = rng.random() < p_pass
    bfound = exact or rng.random() < 0.3
    if status == "completed" and exact and bfound:
        cut = rng.randrange(0, 6)
        output = _rand_ids(rng, 0, 6)[:cut] + tuple(boundary) + _rand_ids(rng, 0, 3)
    else:
        output = _rand_ids(rng, 0, 9)
        if eos_id is not None and rng.random() < 0.4:
            pos = rng.randrange(0, len(output) + 1)
            output = output[:pos] + (eos_id,) + output[pos:]
    return {
        "candidate": "cand",
        "fixture": fixture_id,
        "input_ids": _rand_ids(rng, 1, 8),
        "output_ids": output,
        "answer": "x",
        "exact_match": rng.choice([exact, exact, 1 if exact else 0, None]),
        "boundary_found": bfound,
        "boundary_text_match": rng.choice([True, False, None, 1, "yes"]),
        "raw_reason": rng.choice([None, "limit", "error", "cancelled"]),
        "error": None,
        "status": status,
    }


def _assert_same_outcome(test, fn_impl, fn_oracle, *args, **kwargs):
    """Both callables must return equal values or raise the same exception type."""
    try:
        got = fn_impl(*args, **kwargs)
        impl_exc = None
    except Exception as exc:  # noqa: BLE001 -- comparing exception behavior is the point
        got, impl_exc = None, type(exc)
    try:
        want = fn_oracle(*args, **kwargs)
        oracle_exc = None
    except Exception as exc:  # noqa: BLE001 -- comparing exception behavior is the point
        want, oracle_exc = None, type(exc)
    test.assertEqual(impl_exc, oracle_exc, f"exception mismatch: {args!r} {kwargs!r}")
    if impl_exc is None:
        test.assertEqual(got, want, f"value mismatch: {args!r} {kwargs!r}")


# ---------------------------------------------------------------------------
# classify_observation / failure_taxonomy
# ---------------------------------------------------------------------------


class TestClassifyObservation(unittest.TestCase):
    def test_fuzz_vs_oracle(self):
        for i in range(400):
            rng = _rng("classify", i)
            boundary = tuple(rng.randrange(1, 250) for _ in range(rng.choice([1, 2])))
            eos_id = rng.choice([None, 3, 250])
            fid = f"f{rng.randrange(4)}"
            obs = _rand_obs_dict(rng, fid, boundary, eos_id)
            # Occasionally inject hostile shapes the docstring calls out.
            roll = rng.random()
            if roll < 0.05:
                obs["output_ids"] = 5  # truthy non-iterable -> TypeError path
                obs["boundary_found"] = False
                obs["boundary_text_match"] = False
                obs["status"] = "completed"
                obs["exact_match"] = False
                obs["raw_reason"] = "limit"
            elif roll < 0.08:
                obs["output_ids"] = "ab"  # truthy string: tuple() of chars
                obs["boundary_found"] = False
                obs["boundary_text_match"] = False
                obs["status"] = "completed"
                obs["exact_match"] = False
            with self.subTest(i=i, obs=obs, eos_id=eos_id):
                _assert_same_outcome(self, X.classify_observation, oracle_classify, obs, eos_id)

    def test_non_dict_raises_value_error(self):
        for bad in (None, 5, "x", [("a", 1)], (("a", 1),)):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    X.classify_observation(bad, None)
                with self.assertRaises(ValueError):
                    oracle_classify(bad, None)

    def test_documented_priority_order(self):
        # Backend failure beats the tokenization-mismatch diagnostic.
        obs = {
            "exact_match": False,
            "boundary_found": False,
            "status": "failed",
            "raw_reason": "limit",
            "boundary_text_match": True,
            "output_ids": (),
        }
        self.assertEqual(X.classify_observation(obs, None), "backend-error")
        # raw_reason == "error" alone is a backend failure too.
        obs2 = dict(obs, status="completed", raw_reason="error")
        self.assertEqual(X.classify_observation(obs2, None), "backend-error")
        # Cancelled groups with backend-error even with a boundary present...
        # but the pass check comes first: a "passing" cancelled observation
        # returns None (evaluate never produces one: exact_match requires
        # status == "completed" there).
        obs3 = {
            "exact_match": True,
            "boundary_found": False,
            "status": "cancelled",
            "output_ids": (),
        }
        self.assertEqual(X.classify_observation(obs3, None), "backend-error")
        self.assertIsNone(X.classify_observation(dict(obs3, boundary_found=True), None))
        # EOS membership beats the generic no-boundary bucket.
        obs4 = {
            "exact_match": False,
            "boundary_found": False,
            "status": "completed",
            "boundary_text_match": False,
            "output_ids": (9, 3, 11),
        }
        self.assertEqual(X.classify_observation(obs4, 3), "eos-without-boundary")
        self.assertEqual(X.classify_observation(obs4, None), "no-boundary-in-N")
        # boundary_text_match must be exactly True (D2-adjacent strictness).
        obs5 = dict(obs4, boundary_text_match=1, output_ids=(9, 11))
        self.assertEqual(X.classify_observation(obs5, 3), "no-boundary-in-N")
        # A pass classifies to None even when the backend also errored.
        obs6 = {
            "exact_match": True,
            "boundary_found": True,
            "status": "failed",
            "output_ids": (),
        }
        self.assertIsNone(X.classify_observation(obs6, None))

    def test_eos_branch_type_error_on_truthy_noniterable(self):
        # Docstring contract: a truthy non-iterable output_ids raises
        # TypeError from the EOS-membership check, but only when reached.
        obs = {
            "exact_match": False,
            "boundary_found": False,
            "status": "completed",
            "boundary_text_match": False,
            "output_ids": 5,
        }
        with self.assertRaises(TypeError):
            X.classify_observation(obs, 3)
        # Short-circuits first: pass and backend-failure never reach it.
        self.assertIsNone(
            X.classify_observation(dict(obs, exact_match=True, boundary_found=True), 3)
        )
        self.assertEqual(X.classify_observation(dict(obs, status="failed"), 3), "backend-error")


class TestFailureTaxonomy(unittest.TestCase):
    def _random_result(self, rng):
        boundary = (rng.randrange(1, 250),)
        eos_id = rng.choice([None, 7])
        n_candidates = rng.randrange(1, 4)
        search = []
        for c in range(n_candidates):
            roll = rng.random()
            if roll < 0.08:
                search.append(rng.choice(["junk", 42, None]))  # non-dict entry: skipped
                continue
            fmt: dict = {"name": f"cand{c}"}
            if rng.random() < 0.08:
                fmt = rng.choice([{}, {"name": 5}, {"name": ""}, "name-string"])
            observations = [
                _rand_obs_dict(rng, f"f{f}", boundary, eos_id) for f in range(rng.randrange(1, 4))
            ]
            if rng.random() < 0.06:
                observations.append("not-a-dict")  # propagates ValueError
            if rng.random() < 0.05:
                observations = "not-a-list"  # skipped silently
            search.append({"format": fmt, "observations": observations})
        decision = rng.choice(
            ["held_out_failed", "compatible_on_fixture_set", "ambiguous", "no_candidate"]
        )
        held_out = None
        if decision == "held_out_failed" or rng.random() < 0.3:
            held_obs = [
                _rand_obs_dict(rng, f"h{f}", boundary, eos_id) for f in range(rng.randrange(0, 3))
            ]
            if rng.random() < 0.1 and held_obs:
                held_obs[0] = 99  # non-dict held-out obs: silently skipped
            held_out = {"observations": held_obs}
            if rng.random() < 0.05:
                held_out = "not-a-dict"
        return {"search": search, "decision": decision, "held_out": held_out}, eos_id

    def test_fuzz_vs_oracle(self):
        for i in range(300):
            rng = _rng("taxonomy", i)
            result, eos_id = self._random_result(rng)
            with self.subTest(i=i):
                _assert_same_outcome(self, X.failure_taxonomy, oracle_taxonomy, result, eos_id)

    def test_result_must_be_dict(self):
        with self.assertRaises(ValueError):
            X.failure_taxonomy([], None)
        with self.assertRaises(ValueError):
            X.failure_taxonomy(None, None)

    def test_all_pass_candidate_gets_zero_counts(self):
        obs = {
            "exact_match": True,
            "boundary_found": True,
            "status": "completed",
            "output_ids": (1, 2),
        }
        result = {
            "search": [{"format": {"name": "a"}, "observations": [obs, obs]}],
            "decision": "compatible_on_fixture_set",
            "held_out": {"observations": [obs]},
        }
        got = X.failure_taxonomy(result, None)
        self.assertEqual(got["per_candidate"]["a"], dict.fromkeys(FAILURE_MODES, 0))
        self.assertIsNone(got["held_out"])
        self.assertEqual(got["modes"], list(FAILURE_MODES))
        self.assertFalse(got["eos_id_considered"])
        got2 = X.failure_taxonomy(result, 9)
        self.assertTrue(got2["eos_id_considered"])

    def test_held_out_only_counted_on_divergence(self):
        obs = {
            "exact_match": False,
            "boundary_found": False,
            "status": "completed",
            "output_ids": (1,),
        }
        base = {
            "search": [{"format": {"name": "a"}, "observations": [obs]}],
            "held_out": {"observations": [obs, obs]},
        }
        diverged = dict(base, decision="held_out_failed")
        got = X.failure_taxonomy(diverged, None)
        self.assertEqual(got["held_out"]["held-out-divergence"], 2)
        for mode in FAILURE_MODES:
            if mode != "held-out-divergence":
                self.assertEqual(got["held_out"][mode], 0)
        for decision in ("compatible_on_fixture_set", "ambiguous", "no_candidate"):
            calm = dict(base, decision=decision)
            self.assertIsNone(X.failure_taxonomy(calm, None)["held_out"])


# ---------------------------------------------------------------------------
# budget_frontier
# ---------------------------------------------------------------------------


class TestBudgetFrontier(unittest.TestCase):
    def _random_case(self, rng):
        n_candidates = rng.randrange(1, 4)
        n_fixtures = rng.randrange(1, 5)
        boundary = tuple(rng.randrange(1, 250) for _ in range(rng.choice([1, 2])))
        max_new_tokens = rng.randrange(8, 65)
        fixture_ids = [f"f{i}" for i in range(n_fixtures)]
        search = []
        for c in range(n_candidates):
            entry_boundary = boundary
            entry_extra = {}
            if rng.random() < 0.35:
                entry_boundary = tuple(rng.randrange(1, 250) for _ in range(rng.choice([1, 2])))
                entry_extra = {"boundary_token_ids": list(entry_boundary)}
            observations = []
            for fid in fixture_ids:
                passes = rng.random() < 0.45
                if passes:
                    cut = rng.randrange(0, 7)
                    output = _rand_ids(rng, 0, 6)[:cut] + entry_boundary + _rand_ids(rng, 0, 3)
                    status, exact, bfound = "completed", True, True
                else:
                    output = _rand_ids(rng, 0, 9)
                    status = rng.choice(["completed"] * 4 + ["failed"])
                    exact = status == "completed" and rng.random() < 0.3
                    bfound = False
                observations.append(
                    {
                        "fixture": fid,
                        "status": status,
                        "exact_match": exact,
                        "boundary_found": bfound,
                        "output_ids": rng.choice([list(output), tuple(output)]),
                    }
                )
            search.append(
                {"format": {"name": f"cand{c}"}, "observations": observations, **entry_extra}
            )
        native_decision = rng.choice(
            ["compatible_on_fixture_set", "held_out_failed", "ambiguous", "no_candidate"]
        )
        if native_decision in ("compatible_on_fixture_set", "held_out_failed"):
            native_winner = f"cand{rng.randrange(n_candidates)}"
        else:
            native_winner = None
        return search, {
            "boundary": boundary,
            "max_new_tokens": max_new_tokens,
            "native_decision": native_decision,
            "native_winner": native_winner,
        }

    def test_fuzz_vs_oracle(self):
        for i in range(300):
            rng = _rng("budget", i)
            search, kwargs = self._random_case(rng)
            with self.subTest(i=i):
                _assert_same_outcome(
                    self, X.budget_frontier, oracle_budget_frontier, search, **kwargs
                )

    def test_input_validation(self):
        search = [
            {
                "format": {"name": "a"},
                "observations": [
                    {
                        "fixture": "f0",
                        "status": "completed",
                        "exact_match": True,
                        "boundary_found": True,
                        "output_ids": [1, 2, 3],
                    }
                ],
            }
        ]
        good = {
            "boundary": (3,),
            "max_new_tokens": 16,
            "native_decision": "compatible_on_fixture_set",
            "native_winner": "a",
        }
        X.budget_frontier(search, **good)  # sanity: valid call works
        bad_kwarg_sets = [
            dict(good, boundary=()),
            dict(good, boundary=[3]),
            dict(good, boundary=(3, -1)),
            dict(good, max_new_tokens=0),
            dict(good, max_new_tokens=4097),
            dict(good, max_new_tokens="16"),
            dict(good, native_decision="bogus"),
            dict(good, native_winner="ghost"),
            dict(good, native_winner=None),
            dict(
                good,
                native_decision="ambiguous",
                native_winner="a",
            ),
        ]
        for i, kwargs in enumerate(bad_kwarg_sets):
            with self.subTest(i=i, kwargs=kwargs), self.assertRaises(ValueError):
                X.budget_frontier(search, **kwargs)
        bad_searches = [
            [],
            (),
            "nope",
            [{"format": {"name": "a"}, "observations": []}],
            [{"format": {"name": "a"}, "observations": "x"}],
            [{"format": {"name": "a"}, "observations": ["not-a-dict"]}],
            [{"format": {}, "observations": [{"fixture": "f"}]}],
            [{"format": {"name": ""}, "observations": [{"fixture": "f"}]}],
            [{"observations": [{"fixture": "f"}]}],
            ["not-a-dict"],
            [
                {
                    "format": {"name": "a"},
                    "observations": [{"fixture": "f0"}],
                    "boundary_token_ids": (),
                }
            ],
            [
                {
                    "format": {"name": "a"},
                    "observations": [{"fixture": "f0"}],
                    "boundary_token_ids": [1, True],
                }
            ],
        ]
        for i, bad in enumerate(bad_searches):
            with self.subTest(i=i), self.assertRaises(ValueError):
                X.budget_frontier(bad, **good)
        # Duplicate candidate names raise.
        dup = search + [
            {
                "format": {"name": "a"},
                "observations": [
                    {
                        "fixture": "f0",
                        "status": "completed",
                        "exact_match": True,
                        "boundary_found": True,
                        "output_ids": [1, 2, 3],
                    }
                ],
            }
        ]
        with self.assertRaises(ValueError):
            X.budget_frontier(dup, **good)
        # Fixture coverage must match in order.
        other = [
            {
                "format": {"name": "b"},
                "observations": [
                    {
                        "fixture": "f1",
                        "status": "completed",
                        "exact_match": True,
                        "boundary_found": True,
                        "output_ids": [1, 2, 3],
                    }
                ],
            }
        ]
        with self.assertRaises(ValueError):
            X.budget_frontier(search + other, **good)

    def test_tie_is_ambiguous_not_first_winner(self):
        # Two identical all-pass candidates: the conservative rule abstains.
        def entry(name):
            return {
                "format": {"name": name},
                "observations": [
                    {
                        "fixture": "f0",
                        "status": "completed",
                        "exact_match": True,
                        "boundary_found": True,
                        "output_ids": [9, 9, 5, 6],
                    }
                ],
            }

        got = X.budget_frontier(
            [entry("a"), entry("b")],
            boundary=(5, 6),
            max_new_tokens=16,
            native_decision="ambiguous",
            native_winner=None,
        )
        self.assertEqual(got["candidates"]["a"]["min_budget_per_fixture"], [4])
        self.assertEqual(got["candidates"]["a"]["full_pass_budget"], 4)
        # At budget 4 both pass -> ambiguous; decision_steps must say so.
        step_at_4 = [s for s in got["decision_steps"] if s[0] == 4]
        self.assertTrue(step_at_4)
        self.assertEqual(step_at_4[0][1], "ambiguous")
        # Below budget 4 nobody passes -> the frontier is not robust there.
        self.assertFalse(got["budget_robust"])
        self.assertEqual(got["flip_ranges"], [[1, 3]])

    def test_exact_derivation_example(self):
        # Hand-checkable: boundary (5,), outputs put the boundary at cuts
        # 0 and 2 -> min budgets 1 and 3.
        search = [
            {
                "format": {"name": "a"},
                "observations": [
                    {
                        "fixture": "f0",
                        "status": "completed",
                        "exact_match": True,
                        "boundary_found": True,
                        "output_ids": [5, 1, 1],
                    },
                    {
                        "fixture": "f1",
                        "status": "completed",
                        "exact_match": True,
                        "boundary_found": True,
                        "output_ids": [1, 1, 5],
                    },
                    {
                        "fixture": "f2",
                        "status": "completed",
                        "exact_match": False,
                        "boundary_found": True,
                        "output_ids": [1, 5],
                    },
                ],
            }
        ]
        got = X.budget_frontier(
            search,
            boundary=(5,),
            max_new_tokens=10,
            native_decision="no_candidate",
            native_winner=None,
        )
        cand = got["candidates"]["a"]
        self.assertEqual(cand["min_budget_per_fixture"], [1, 3, None])
        self.assertEqual(cand["pass_steps"], [[1, 1], [3, 2]])
        self.assertIsNone(cand["full_pass_budget"])
        # A non-completed observation never earns a budget even with a
        # boundary-shaped output.
        search[0]["observations"][0]["status"] = "failed"
        got2 = X.budget_frontier(
            search,
            boundary=(5,),
            max_new_tokens=10,
            native_decision="no_candidate",
            native_winner=None,
        )
        self.assertEqual(got2["candidates"]["a"]["min_budget_per_fixture"], [None, 3, None])


# ---------------------------------------------------------------------------
# _resolve_boundaries / candidate_fingerprint
# ---------------------------------------------------------------------------


class TestResolveBoundaries(unittest.TestCase):
    def _formats(self, rng, n):
        out = []
        for i in range(n):
            out.append(
                Format(
                    name=f"cand{i}",
                    bos=rng.choice(["", "<bos>"]),
                    user_prefix="U:",
                    assistant_prefix="A:",
                    turn_end="\n",
                )
            )
        return tuple(out)

    def test_fuzz_vs_oracle(self):
        for i in range(200):
            rng = _rng("resolve", i)
            n = rng.randrange(1, 5)
            formats = self._formats(rng, n)
            boundary = tuple(rng.randrange(0, 250) for _ in range(rng.randrange(1, 3)))
            roll = rng.random()
            if roll < 0.4:
                bmap = None
            elif roll < 0.7:
                bmap = {}
                for j in rng.sample(range(n), rng.randrange(0, n)):
                    bmap[f"cand{j}"] = tuple(
                        rng.randrange(0, 250) for _ in range(rng.randrange(1, 3))
                    )
            elif roll < 0.8:
                bmap = {"ghost": (1,)}  # unknown candidate
            elif roll < 0.9:
                bmap = {f"cand{rng.randrange(n)}": ()}  # empty boundary
            else:
                bmap = rng.choice(["nope", [(1,)], {5: (1,)}])
            with self.subTest(i=i, bmap=bmap):
                _assert_same_outcome(
                    self,
                    X._resolve_boundaries,
                    oracle_resolve_boundaries,
                    formats,
                    boundary,
                    bmap,
                )

    def test_default_applies_to_every_candidate(self):
        formats = self._formats(random.Random("cycle34:resolve:fixed"), 3)  # noqa: S311 -- deterministic test seeding
        got = X._resolve_boundaries(formats, (1, 2), None)
        self.assertEqual(got, {f.name: (1, 2) for f in formats})

    def test_map_overrides_only_named(self):
        formats = self._formats(random.Random("cycle34:resolve:fixed2"), 3)  # noqa: S311 -- deterministic test seeding
        got = X._resolve_boundaries(formats, (1, 2), {"cand1": (9,)})
        self.assertEqual(got["cand1"], (9,))
        self.assertEqual(got["cand0"], (1, 2))
        self.assertEqual(got["cand2"], (1, 2))


class TestCandidateFingerprint(unittest.TestCase):
    def _formats(self, rng):
        return Format(
            name=rng.choice(["a", "b"]),
            bos=rng.choice(["", "<s>"]),
            user_prefix="U:",
            assistant_prefix="A:",
            turn_end="\n",
            repeat_bos=rng.choice([True, False]),
            user_turn_end=rng.choice([None, "[/INST]"]),
            assistant_turn_end=rng.choice([None, ""]),
            open_suffix=rng.choice([None, "A:"]),
        )

    def test_fuzz_vs_oracle(self):
        for i in range(200):
            rng = _rng("fingerprint", i)
            fmt = self._formats(rng)
            with self.subTest(i=i):
                self.assertEqual(X.candidate_fingerprint(fmt), oracle_fingerprint(fmt))

    def test_name_is_part_of_identity(self):
        a = Format(name="alpha", bos="", user_prefix="U:", assistant_prefix="A:", turn_end="\n")
        b = Format(name="beta", bos="", user_prefix="U:", assistant_prefix="A:", turn_end="\n")
        self.assertNotEqual(X.candidate_fingerprint(a), X.candidate_fingerprint(b))
        self.assertTrue(X.candidate_fingerprint(a).startswith("sha256:"))

    def test_repeat_bos_normalized_when_bos_empty(self):
        # Docstring: with an empty bos, repeat_bos is meaningless, so the
        # fingerprint normalizes it away.
        a = Format(
            name="x",
            bos="",
            user_prefix="U:",
            assistant_prefix="A:",
            turn_end="\n",
            repeat_bos=True,
        )
        b = Format(
            name="x",
            bos="",
            user_prefix="U:",
            assistant_prefix="A:",
            turn_end="\n",
            repeat_bos=False,
        )
        self.assertEqual(X.candidate_fingerprint(a), X.candidate_fingerprint(b))
        c = Format(
            name="x",
            bos="<s>",
            user_prefix="U:",
            assistant_prefix="A:",
            turn_end="\n",
            repeat_bos=True,
        )
        d = Format(
            name="x",
            bos="<s>",
            user_prefix="U:",
            assistant_prefix="A:",
            turn_end="\n",
            repeat_bos=False,
        )
        self.assertNotEqual(X.candidate_fingerprint(c), X.candidate_fingerprint(d))


# ---------------------------------------------------------------------------
# _stop_cell / stop_matrix
# ---------------------------------------------------------------------------


class TestStopCell(unittest.TestCase):
    def _fixture(self, expected="AB"):
        return Fixture(
            id="f0",
            messages=(Message(role="user", content="q"),),
            expected=expected,
            group="g",
        )

    def test_fuzz_vs_oracle(self):
        backend = ByteBackend()
        for i in range(300):
            rng = _rng("stopcell", i)
            fixture = self._fixture(rng.choice(["AB", "A", "hello", ""]))
            boundary = tuple(rng.randrange(1, 250) for _ in range(rng.choice([1, 2])))
            available = rng.random() < 0.8
            fired = available and rng.random() < 0.6
            truncated = _rand_ids(rng, 0, 10)
            outcome = StopOutcome(
                available=available,
                fired=fired,
                truncated_ids=truncated,
                detail=rng.choice(["fired_at_step_2", "no_stop_within_cap", "x"]),
            )
            stop_text = rng.choice([None, "kept", "stripped"])
            if rng.random() < 0.06:
                be = ExplodingDecodeBackend(mode=rng.choice(["raise", "non-str"]))
                use_backend = be
            else:
                use_backend = backend
            with self.subTest(i=i):
                _assert_same_outcome(
                    self,
                    X._stop_cell,
                    oracle_stop_cell,
                    use_backend,
                    fixture,
                    boundary,
                    outcome,
                    stop_text=stop_text,
                )

    def test_pass_requires_boundary_and_answer(self):
        backend = ByteBackend()
        fixture = self._fixture("C")
        boundary = (65, 66)  # decodes to "AB"
        # Answer matches but no boundary present -> not a pass.
        outcome = StopOutcome(True, False, (67, 68), "no_stop_within_cap")
        got = X._stop_cell(backend, fixture, boundary, outcome)
        self.assertTrue(got["available"])
        self.assertFalse(got["pass"])
        self.assertIsNone(got["stop_text"])
        # Boundary present and answer matches -> pass; stop_text kept iff fired.
        outcome2 = StopOutcome(True, True, (67, 65, 66), "fired_at_step_3")
        got2 = X._stop_cell(backend, fixture, boundary, outcome2, stop_text="kept")
        self.assertTrue(got2["pass"])
        self.assertEqual(got2["stop_text"], "kept")
        got3 = X._stop_cell(backend, fixture, boundary, outcome2)
        self.assertIsNone(got3["stop_text"])

    def test_unavailable_omits_stop_text_key(self):
        # D1: the docstring suggests stop_text=None when the stop did not
        # fire; for an *unavailable* outcome the key is absent entirely.
        backend = ByteBackend()
        outcome = StopOutcome(False, False, (1,), "token_text_unavailable")
        got = X._stop_cell(backend, self._fixture(), (1,), outcome, stop_text="kept")
        self.assertEqual(got, {"available": False, "detail": "token_text_unavailable"})
        self.assertNotIn("stop_text", got)


class TestStopMatrix(unittest.TestCase):
    def _fixtures(self, n):
        return tuple(
            Fixture(
                id=f"f{i}",
                messages=(Message(role="user", content=f"q{i}"),),
                expected="AB",
                group="g",
            )
            for i in range(n)
        )

    def _random_obs(self, rng, fid, boundary, eos_id):
        status = rng.choice(["completed"] * 7 + ["failed", "cancelled"])
        output = _rand_ids(rng, 0, 10)
        if status == "completed" and rng.random() < 0.5:
            cut = rng.randrange(0, 6)
            output = _rand_ids(rng, 0, 6)[:cut] + tuple(boundary) + _rand_ids(rng, 0, 3)
            if eos_id is not None and rng.random() < 0.3:
                output = output + (eos_id,)
        exact = status == "completed" and rng.random() < 0.4
        bfound = exact or (status == "completed" and rng.random() < 0.3)
        return {
            "fixture": fid,
            "status": status,
            "input_ids": _rand_ids(rng, 1, 8),
            "output_ids": rng.choice([list(output), tuple(output)]),
            "exact_match": exact,
            "boundary_found": bfound,
        }

    def test_fuzz_vs_oracle(self):
        backend = ByteBackend()
        for i in range(150):
            rng = _rng("stopmatrix", i)
            n = rng.randrange(1, 4)
            fixtures = self._fixtures(n)
            boundary = tuple(rng.randrange(1, 250) for _ in range(rng.choice([1, 2])))
            eos_id = rng.choice([None, 250])
            observations = tuple(self._random_obs(rng, f"f{j}", boundary, eos_id) for j in range(n))
            kwargs = {
                "boundary": boundary,
                "eos_id": eos_id,
                "vllm_include_stop_str_in_output": rng.choice([True, False]),
                "vllm_min_tokens": rng.choice([0, 1, 3]),
            }
            if rng.random() < 0.25:
                kwargs["token_bytes"] = lambda tid: bytes([tid % 256])
            with self.subTest(i=i):
                _assert_same_outcome(
                    self,
                    X.stop_matrix,
                    oracle_stop_matrix,
                    backend,
                    "cand",
                    fixtures,
                    observations,
                    **kwargs,
                )

    def test_input_validation(self):
        backend = ByteBackend()
        fixtures = self._fixtures(1)
        obs = (
            {
                "fixture": "f0",
                "status": "completed",
                "input_ids": (1,),
                "output_ids": (65, 66),
                "exact_match": True,
                "boundary_found": True,
            },
        )
        X.stop_matrix(backend, "cand", fixtures, obs, boundary=(65, 66))  # valid
        cases = [
            {"candidate_name": ""},
            {"candidate_name": 5},
            {"candidate_name": "  "[:0]},
            {"fixtures": list(fixtures)},
            {"observations": list(obs)},
            {"fixtures": self._fixtures(2)},
            {"observations": obs + obs},
            {"boundary": ()},
            {"boundary": [65]},
            {"boundary": (65, -2)},
            {"eos_id": -1},
            {"eos_id": "3"},
            {"token_bytes": 42},
            {"vllm_include_stop_str_in_output": 1},
            {"vllm_min_tokens": -1},
            {"vllm_min_tokens": True},
        ]
        for i, kw in enumerate(cases):
            base = {
                "candidate_name": "cand",
                "fixtures": fixtures,
                "observations": obs,
                "boundary": (65, 66),
            }
            base.update(kw)
            with self.subTest(i=i, kw=kw), self.assertRaises(ValueError):
                X.stop_matrix(backend, **base)
        # Observation/fixture alignment mismatch.
        bad_obs = (
            {
                "fixture": "WRONG",
                "status": "completed",
                "input_ids": (1,),
                "output_ids": (65, 66),
                "exact_match": True,
                "boundary_found": True,
            },
        )
        with self.assertRaises(ValueError):
            X.stop_matrix(backend, "cand", fixtures, bad_obs, boundary=(65, 66))
        # Non-dict observation.
        with self.assertRaises(ValueError):
            X.stop_matrix(backend, "cand", fixtures, ("nope",), boundary=(65, 66))

    def test_incomplete_observations_degrade_cells(self):
        backend = ByteBackend()
        fixtures = self._fixtures(1)
        obs = (
            {
                "fixture": "f0",
                "status": "failed",
                "input_ids": (1,),
                "output_ids": (65, 66),
                "exact_match": False,
                "boundary_found": False,
            },
        )
        rows = X.stop_matrix(backend, "cand", fixtures, obs, boundary=(65, 66))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertFalse(row["native_pass"])
        self.assertEqual(row["flips"], [])
        for semantic in STOP_SEMANTICS[1:]:
            cell = row["cells"][semantic]
            self.assertEqual(cell, {"available": False, "detail": "observation_incomplete"})
            self.assertNotIn("stop_text", cell)  # D1

    def test_vllm_disposition_stripped_by_default(self):
        # Docstring: vLLM's upstream default strips the stop string; the
        # kept variant is opt-in.
        backend = ByteBackend()
        fixtures = self._fixtures(1)
        obs = (
            {
                "fixture": "f0",
                "status": "completed",
                "input_ids": (1,),
                "output_ids": (65, 66, 67),
                "exact_match": True,
                "boundary_found": True,
            },
        )
        rows = X.stop_matrix(backend, "cand", fixtures, obs, boundary=(65, 66))
        vllm = rows[0]["cells"]["vllm_holdback"]
        hf = rows[0]["cells"]["hf_stop_string"]
        self.assertEqual(vllm["stop_text"], "stripped")
        self.assertEqual(hf["stop_text"], "kept")
        rows2 = X.stop_matrix(
            backend,
            "cand",
            fixtures,
            obs,
            boundary=(65, 66),
            vllm_include_stop_str_in_output=True,
        )
        self.assertEqual(rows2[0]["cells"]["vllm_holdback"]["stop_text"], "kept")

    def test_boundary_decode_failure_degrades_string_cells(self):
        backend = ExplodingDecodeBackend(mode="raise")
        fixtures = self._fixtures(1)
        obs = (
            {
                "fixture": "f0",
                "status": "completed",
                "input_ids": (1,),
                "output_ids": (65, 66),
                "exact_match": True,
                "boundary_found": True,
            },
        )
        rows = X.stop_matrix(backend, "cand", fixtures, obs, boundary=(65, 66))
        # boundary_text degrades to None -> "" stop text -> emulation refuses.
        for semantic in ("hf_stop_string", "vllm_holdback"):
            self.assertFalse(rows[0]["cells"][semantic]["available"])
        # eos_appended needs no boundary-text decode; with eos_id=None the
        # emulation itself reports eos_id_unavailable.
        eos_cell = rows[0]["cells"]["eos_appended"]
        self.assertFalse(eos_cell["available"])
        self.assertEqual(eos_cell["detail"], "eos_id_unavailable")
        # With an eos_id the emulation runs, but _stop_cell still decodes
        # the answer span -> answer_decode_unavailable.
        rows2 = X.stop_matrix(backend, "cand", fixtures, obs, boundary=(65, 66), eos_id=250)
        eos_cell2 = rows2[0]["cells"]["eos_appended"]
        self.assertFalse(eos_cell2["available"])
        self.assertEqual(eos_cell2["detail"], "answer_decode_unavailable")


# ---------------------------------------------------------------------------
# logprob diagnostic internals
# ---------------------------------------------------------------------------


class TestLogprobScoring(unittest.TestCase):
    def _random_scores(self, rng, n):
        scores = []
        for _ in range(n):
            status = rng.choice(["ok"] * 5 + ["unavailable", "ok_empty"])
            total = rng.choice([rng.uniform(-20, 0), None, float("nan"), float("inf"), 0.0])
            scores.append(
                TargetScore(
                    status=status,
                    reason=None if status == "ok" else "x",
                    per_token_logprobs=(),
                    total_logprob=total,
                    mean_logprob=(
                        total / 3
                        if type(total) is float and total not in (float("inf"),)
                        else total
                    ),
                    scored_token_count=rng.randrange(0, 6),
                    target_byte_length=rng.randrange(0, 12),
                    prompt_token_count=rng.randrange(1, 9),
                )
            )
        return scores

    def test_aggregate_fuzz_vs_oracle(self):
        for i in range(300):
            rng = _rng("aggregate", i)
            scores = self._random_scores(rng, rng.randrange(0, 5))
            with self.subTest(i=i):
                self.assertEqual(X._aggregate_fixture_scores(scores), oracle_aggregate(scores))

    def test_aggregate_only_ok_contributes(self):
        scores = [
            TargetScore("ok", None, (), -3.0, -1.0, 3, 3, 5),
            TargetScore("unavailable", "x", (), None, None, 0, 0, 5),
            TargetScore("ok_empty", None, (), 0.0, None, 0, 0, 5),
        ]
        got = X._aggregate_fixture_scores(scores)
        self.assertEqual(got["scored_fixtures"], 1)
        self.assertEqual(got["total_logprob"], -3.0)
        self.assertEqual(got["mean_logprob"], -1.0)
        self.assertEqual(got["scored_token_count"], 3)

    def test_aggregate_nan_degrades_to_none(self):
        scores = [TargetScore("ok", None, (), float("nan"), float("nan"), 2, 2, 5)]
        got = X._aggregate_fixture_scores(scores)
        self.assertIsNone(got["total_logprob"])
        self.assertIsNone(got["mean_logprob"])
        self.assertEqual(got["scored_fixtures"], 1)  # counts stay visible

    def test_interpret_kind_fuzz_vs_oracle(self):
        for i in range(400):
            rng = _rng("interpret", i)
            fixture_scores = []
            for _ in range(rng.randrange(1, 5)):
                native_pass = rng.random() < 0.35
                status = rng.choice(["ok", "ok", "unavailable", "not_scored"])
                mean = rng.choice([rng.uniform(-9, 1), None, -2.0, -5.0])
                fixture_scores.append(
                    {
                        "native_pass": native_pass,
                        "status": status,
                        "mean_logprob": mean,
                        "total_logprob": (mean * 3 if type(mean) is float else None),
                        "scored_token_count": rng.choice([3, 0]),
                    }
                )
            with self.subTest(i=i):
                got = X._interpret_logprob_candidate(fixture_scores)
                self.assertEqual(got["kind"], oracle_interpret_kind(fixture_scores))
                self.assertIn("text", got)
                self.assertIsInstance(got["text"], str)

    def test_interpret_band_edges(self):
        def fs(mean, native_pass=False):
            return {
                "native_pass": native_pass,
                "status": "ok",
                "mean_logprob": mean,
                "total_logprob": mean * 2,
                "scored_token_count": 2,
            }

        self.assertEqual(X._interpret_logprob_candidate([fs(-2.0)])["kind"], "greedy_derailed")
        self.assertEqual(X._interpret_logprob_candidate([fs(-5.0)])["kind"], "indeterminate")
        self.assertEqual(X._interpret_logprob_candidate([fs(-5.01)])["kind"], "weights_dont_know")
        self.assertEqual(X._interpret_logprob_candidate([fs(-2.0), fs(-9.0)])["kind"], "mixed")
        self.assertEqual(
            X._interpret_logprob_candidate([fs(-1.0, native_pass=True)])["kind"],
            "no_failures",
        )
        bad = dict(fs(-1.0), status="unavailable", mean_logprob=None)
        self.assertEqual(X._interpret_logprob_candidate([bad])["kind"], "unavailable")
        # The indeterminate text cites the declared band edges.
        text = X._interpret_logprob_candidate([fs(-3.0)])["text"]
        self.assertIn(f"{LOGPROB_DERAILED_MEAN_THRESHOLD:.1f}", text)
        self.assertIn(f"{LOGPROB_UNKNOWN_MEAN_THRESHOLD:.1f}", text)

    def test_score_expected_answer_encode_failures(self):
        ByteBackend()

        # encode raising -> fixed encode_failed code, never an exception.
        class BoomEncode(ByteBackend):
            def encode(self, text):
                raise RuntimeError("tokenizer exploded")

        got = X._score_expected_answer(BoomEncode(), (1, 2), "AB")
        self.assertEqual(got.status, "unavailable")
        self.assertEqual(got.reason, "encode_failed")
        self.assertEqual(got.prompt_token_count, 2)

        # encode returning a non-tuple -> same fixed code.
        class WeirdEncode(ByteBackend):
            def encode(self, text):
                return [65, 66]

        got2 = X._score_expected_answer(WeirdEncode(), (1,), "AB")
        self.assertEqual(got2.reason, "encode_failed")
        # Success delegates to the engine (shared leaf): compare directly.
        logprob_backend = LogprobBackend()
        want = X.score_target_logprobs(logprob_backend, (10, 20), logprob_backend.encode("AB"))
        got3 = X._score_expected_answer(logprob_backend, (10, 20), "AB")
        self.assertEqual(got3, want)
        self.assertEqual(got3.status, "ok")

    def test_logprob_unavailable_shape(self):
        got = X._logprob_unavailable("encode_failed", 4)
        self.assertEqual(got.status, "unavailable")
        self.assertEqual(got.reason, "encode_failed")
        self.assertIsNone(got.total_logprob)
        self.assertEqual(got.prompt_token_count, 4)


class TestLogprobDiagnostic(unittest.TestCase):
    def _fixtures(self):
        return (
            Fixture(
                id="f0",
                messages=(Message(role="user", content="q0"),),
                expected="AB",
                group="g",
            ),
            Fixture(
                id="f1",
                messages=(Message(role="user", content="q1"),),
                expected="CD",
                group="g",
            ),
        )

    def _result(self, rng, statuses):
        observations_a = []
        observations_b = []
        for i, fid in enumerate(("f0", "f1")):
            status = statuses[i]
            base = {
                "fixture": fid,
                "status": status,
                "input_ids": list(ByteBackend().encode(f"prompt-{fid}")),
                "output_ids": [65, 66],
                "exact_match": status == "completed" and rng.random() < 0.5,
                "boundary_found": status == "completed" and rng.random() < 0.5,
            }
            observations_a.append(dict(base))
            observations_b.append(dict(base))
        return {
            "search": [
                {"format": {"name": "a"}, "observations": observations_a},
                {"format": {"name": "b"}, "observations": observations_b},
            ]
        }

    def _compare(self, backend, result, fixtures):
        got = X.logprob_diagnostic(backend, result, fixtures)
        want = oracle_logprob_diagnostic(backend, result, fixtures)
        # Method prose pinned as a behavioral fact (nonblank, stable).
        self.assertIsInstance(got["method"], str)
        self.assertTrue(got["method"].strip())
        want["method"] = got["method"]
        # Interpretation prose pinned via kind (oracle-derived).
        for cand_got, cand_want in zip(got["candidates"], want["candidates"], strict=False):
            self.assertEqual(
                cand_got["interpretation"]["kind"], cand_want.pop("interpretation_kind")
            )
            self.assertIsInstance(cand_got["interpretation"]["text"], str)
            cand_got.pop("interpretation")
        self.assertEqual(got, want)
        return got

    def test_fuzz_vs_oracle_boost(self):
        backend = LogprobBackend(mode="boost")
        for i in range(60):
            rng = _rng("diagboost", i)
            statuses = [
                rng.choice(["completed", "completed", "failed", "cancelled"]) for _ in range(2)
            ]
            result = self._result(rng, statuses)
            # Occasionally inject non-dict entries / bad observations.
            if rng.random() < 0.2:
                result["search"].append("junk-entry")
            if rng.random() < 0.2:
                result["search"][0]["observations"].append(42)
            with self.subTest(i=i, statuses=statuses):
                self._compare(backend, result, self._fixtures())

    def test_fuzz_vs_oracle_uniform(self):
        backend = LogprobBackend(mode="uniform")
        for i in range(60):
            rng = _rng("diaguniform", i)
            statuses = [rng.choice(["completed", "completed", "failed"]) for _ in range(2)]
            with self.subTest(i=i):
                self._compare(backend, self._result(rng, statuses), self._fixtures())

    def test_no_forward_logits_unavailable(self):
        got = X.logprob_diagnostic(
            ByteBackend(),
            self._result(random.Random("x"), ["completed"] * 2),  # noqa: S311
            self._fixtures(),
        )
        self.assertFalse(got["available"])
        self.assertEqual(got["detail"], "logits_not_supported")
        self.assertEqual(got["candidates"], [])
        self.assertEqual(
            got["thresholds"]["derailed_mean_logprob_at_least"],
            LOGPROB_DERAILED_MEAN_THRESHOLD,
        )
        self.assertEqual(
            got["thresholds"]["unknown_mean_logprob_below"],
            LOGPROB_UNKNOWN_MEAN_THRESHOLD,
        )

    def test_no_scorable_observations(self):
        backend = LogprobBackend()
        got = X.logprob_diagnostic(backend, {"search": []}, self._fixtures())
        self.assertFalse(got["available"])
        self.assertEqual(got["detail"], "no_scorable_observations")
        got2 = X.logprob_diagnostic(backend, {"search": ["junk", {"format": {}}]}, self._fixtures())
        self.assertFalse(got2["available"])
        self.assertEqual(got2["detail"], "no_scorable_observations")

    def test_input_validation(self):
        backend = LogprobBackend()
        fixtures = self._fixtures()
        with self.assertRaises(ValueError):
            X.logprob_diagnostic(backend, [], fixtures)
        with self.assertRaises(ValueError):
            X.logprob_diagnostic(backend, {"search": {}}, fixtures)
        with self.assertRaises(ValueError):
            X.logprob_diagnostic(backend, {"search": []}, list(fixtures)[:-1] + ["x"])
        # A list of fixtures is accepted (normalized to tuple).
        got = X.logprob_diagnostic(backend, {"search": []}, list(fixtures))
        self.assertFalse(got["available"])

    def test_boost_reads_as_derailed_uniform_as_unknown(self):
        # End-to-end meaning check: boosted logits on a *failed* fixture ->
        # greedy_derailed; uniform logits -> weights_dont_know.
        rng = random.Random("cycle34:diagkinds")  # noqa: S311 -- deterministic test seeding
        result = self._result(rng, ["completed", "completed"])
        for cand in result["search"]:
            for obs in cand["observations"]:
                obs["exact_match"] = False
                obs["boundary_found"] = False
        got_boost = X.logprob_diagnostic(LogprobBackend("boost"), result, self._fixtures())
        self.assertTrue(got_boost["available"])
        kinds = {c["interpretation"]["kind"] for c in got_boost["candidates"]}
        self.assertEqual(kinds, {"greedy_derailed"})
        got_uni = X.logprob_diagnostic(LogprobBackend("uniform"), result, self._fixtures())
        kinds2 = {c["interpretation"]["kind"] for c in got_uni["candidates"]}
        self.assertEqual(kinds2, {"weights_dont_know"})


# ---------------------------------------------------------------------------
# _summary / _failure_code pins (no docstrings; behavior pinned from code)
# ---------------------------------------------------------------------------


class TestSummaryPins(unittest.TestCase):
    def _obs(self, **kw):
        base = {
            "candidate": "c",
            "fixture": "f",
            "input_ids": (1, 2),
            "output_ids": (3,),
            "answer": "",
            "exact_match": False,
            "boundary_found": False,
            "boundary_text_match": None,
            "raw_reason": "limit",
            "error": None,
            "status": "completed",
        }
        base.update(kw)
        return Observation(**base)

    def _report(self, observations, planned, status):
        return EvaluationReport(
            observations=tuple(observations), planned_count=planned, status=status
        )

    def test_summary_completed(self):
        obs = [
            self._obs(exact_match=True, boundary_found=True, input_ids=(1,), output_ids=(2, 3)),
            self._obs(exact_match=True, boundary_found=False),
            self._obs(exact_match=False, boundary_found=True),
        ]
        got = X._summary(self._report(obs, 3, "completed"))
        self.assertEqual(got["status"], "completed")
        self.assertEqual(got["planned"], 3)
        self.assertEqual(got["attempted"], 3)
        self.assertTrue(got["aggregate_eligible"])
        self.assertEqual(got["exact_and_boundary_passes"], 1)
        self.assertEqual(got["rate"], 1 / 3)
        self.assertEqual(got["encoded_input_tokens"], 1 + 2 + 2)
        self.assertEqual(got["retained_output_tokens"], 2 + 1 + 1)
        self.assertIsNone(got["failure_code"])

    def test_summary_ineligible_rate_none(self):
        obs = [self._obs(status="failed", error="encode_failed", raw_reason=None)]
        got = X._summary(self._report(obs, 2, "failed"))
        self.assertFalse(got["aggregate_eligible"])
        self.assertIsNone(got["rate"])
        self.assertEqual(got["failure_code"], "encode_failed")

    def test_failure_code_scans_from_end(self):
        obs = [
            self._obs(status="completed", error="render_failed"),
            self._obs(status="failed", error="generate_failed", raw_reason=None),
        ]
        self.assertEqual(X._failure_code(self._report(obs, 2, "failed")), "generate_failed")
        # Non-string and empty errors are skipped.
        obs2 = [
            self._obs(status="completed", error=""),
            self._obs(status="failed", error=5, raw_reason=None),
        ]
        self.assertIsNone(X._failure_code(self._report(obs2, 2, "failed")))
        self.assertIsNone(X._failure_code(self._report([self._obs()], 1, "completed")))


# ---------------------------------------------------------------------------
# prompt_sensitivity_sweep + seed handling
# ---------------------------------------------------------------------------


def _sweep_formats(marker_tail):
    return (
        Format(
            name="winner",
            bos="",
            user_prefix="Q:",
            assistant_prefix="A:",
            turn_end="\n",
            open_suffix=f"A:{marker_tail}",
        ),
        Format(
            name="loser",
            bos="",
            user_prefix="Q:",
            assistant_prefix="A:",
            turn_end="\n",
        ),
    )


def _sweep_split():
    def fx(i, split_tag):
        return Fixture(
            id=f"{split_tag}{i}",
            # Held-out inputs must differ from search inputs (split hygiene).
            messages=(Message(role="user", content=f"{split_tag} question {i}"),),
            expected="YES",
            group=f"g{split_tag}",  # search/held-out groups must not overlap
        )

    return Split(
        search=(fx(0, "s"), fx(1, "s")),
        held_out=(fx(0, "h"),),
    )


class TestPromptSensitivitySweep(unittest.TestCase):
    def test_depths_flip_when_marker_truncated(self):
        backend = ScriptedBackend(lambda text, seed: "MARK" in text)
        split = _sweep_split()
        got = X.prompt_sensitivity_sweep(
            backend,
            _sweep_formats("MARK"),
            split,
            boundary=(7, 8),
            max_new_tokens=8,
            seed=11,
            max_drop_bytes=3,
        )
        self.assertEqual(got["native_decision"], "compatible_on_fixture_set")
        self.assertEqual(got["native_winner"], "winner")
        self.assertEqual(got["max_drop_bytes"], 3)
        self.assertEqual(len(got["depths"]), 3)
        for depth in got["depths"]:
            self.assertEqual(
                depth,
                {
                    "drop_bytes": depth["drop_bytes"],
                    "decision": "no_candidate",
                    "selected_candidate": None,
                    "winner_changed": True,
                },
            )
        self.assertEqual(got["flips_at"], [1, 2, 3])

    def test_no_flip_when_marker_survives(self):
        # Marker in the middle of the prompt: drops never remove it.
        backend = ScriptedBackend(lambda text, seed: "MARK" in text)
        formats = (
            Format(
                name="winner",
                bos="",
                user_prefix="Q:MARK\n",
                assistant_prefix="A:",
                turn_end="\n",
            ),
            Format(
                name="loser",
                bos="",
                user_prefix="Q:",
                assistant_prefix="A:",
                turn_end="\n",
            ),
        )
        got = X.prompt_sensitivity_sweep(
            backend,
            formats,
            _sweep_split(),
            boundary=(7, 8),
            max_new_tokens=8,
            seed=11,
            max_drop_bytes=2,
        )
        self.assertEqual(got["native_winner"], "winner")
        self.assertEqual(got["flips_at"], [])
        for depth in got["depths"]:
            self.assertFalse(depth["winner_changed"])
            self.assertEqual(depth["decision"], "compatible_on_fixture_set")
            self.assertEqual(depth["selected_candidate"], "winner")

    def test_codepoint_split_depth_records_failed_decision(self):
        # A cut splitting a UTF-8 codepoint records the run's failed
        # decision instead of silently skipping the depth.
        backend = ScriptedBackend(lambda text, seed: "MARK" in text)
        split = _sweep_split()
        got = X.prompt_sensitivity_sweep(
            backend,
            _sweep_formats("MARKé"),  # 2-byte tail char; drop=1 splits it
            split,
            boundary=(7, 8),
            max_new_tokens=8,
            seed=11,
            max_drop_bytes=1,
        )
        self.assertEqual(got["native_decision"], "compatible_on_fixture_set")
        self.assertEqual(len(got["depths"]), 1)
        depth = got["depths"][0]
        self.assertEqual(depth["drop_bytes"], 1)
        self.assertEqual(depth["decision"], "failed")
        self.assertIsNone(depth["selected_candidate"])
        self.assertTrue(depth["winner_changed"])
        self.assertEqual(got["flips_at"], [1])

    def test_max_drop_bytes_validation(self):
        backend = ScriptedBackend(lambda text, seed: True)
        for bad in (0, 9, -1, "3", 2.0):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                X.prompt_sensitivity_sweep(
                    backend,
                    _sweep_formats("MARK"),
                    _sweep_split(),
                    boundary=(7, 8),
                    max_drop_bytes=bad,
                )

    def test_seed_reaches_every_generate_call(self):
        # Seed handling: one experiment seed in, identical per-fixture
        # request seeds out -- across candidates, depths, and stages.
        backend = ScriptedBackend(lambda text, seed: "MARK" in text)
        seed = 1234567
        X.prompt_sensitivity_sweep(
            backend,
            _sweep_formats("MARK"),
            _sweep_split(),
            boundary=(7, 8),
            max_new_tokens=8,
            seed=seed,
            max_drop_bytes=2,
        )

        def expected(fid):
            material = f"{seed}\0{fid}".encode("utf-8", errors="surrogatepass")
            return int.from_bytes(sha256(material).digest()[:4], "big")

        # Native run: 2 search fixtures x 2 candidates + 1 held_out x 1 winner.
        # Depth runs (winner never selected -> no held_out): 2 x 2 each.
        want = []
        for _ in range(2):  # candidates
            want += [expected("s0"), expected("s1")]
        want += [expected("h0")]  # native held_out
        for _ in range(2):  # depths
            for _ in range(2):
                want += [expected("s0"), expected("s1")]
        self.assertEqual(sorted(backend.seeds), sorted(want))

    def test_different_experiment_seeds_differ(self):
        split = _sweep_split()
        seeds_a, seeds_b = [], []
        for store, seed in ((seeds_a, 1), (seeds_b, 2)):
            backend = ScriptedBackend(lambda text, seed: "MARK" in text)
            backend.seeds = store
            X.prompt_sensitivity_sweep(
                backend,
                _sweep_formats("MARK"),
                split,
                boundary=(7, 8),
                max_new_tokens=8,
                seed=seed,
                max_drop_bytes=1,
            )
        self.assertTrue(seeds_a)
        self.assertTrue(set(seeds_a).isdisjoint(set(seeds_b)))


class TestSeedHandling(unittest.TestCase):
    def test_same_seed_to_every_candidate(self):
        # The per-candidate loop passes one seed through; evaluate derives
        # per-fixture request seeds deterministically from (seed, fixture).
        backend = ScriptedBackend(lambda text, seed: True)
        formats = (
            Format(name="a", bos="", user_prefix="Q:", assistant_prefix="A:", turn_end="\n"),
            Format(name="b", bos="", user_prefix="Q:", assistant_prefix="A:", turn_end="\n"),
            Format(name="c", bos="", user_prefix="Q:", assistant_prefix="A:", turn_end="\n"),
        )
        split = _sweep_split()
        X.run_experiment(backend, formats, split, boundary=(7, 8), max_new_tokens=8, seed=99)
        # 3 candidates x 2 search fixtures; no held_out (3-way tie -> ambiguous).
        self.assertEqual(len(backend.seeds), 6)

        def expected(fid):
            material = f"99\0{fid}".encode("utf-8", errors="surrogatepass")
            return int.from_bytes(sha256(material).digest()[:4], "big")

        self.assertEqual(
            sorted(backend.seeds),
            sorted([expected("s0"), expected("s1")] * 3),
        )
        # The result records the experiment seed.
        # (run_experiment assembly itself is cycle-21 territory; the seed
        # field is pinned here as the loop's input.)
        backend2 = ScriptedBackend(lambda text, seed: True)
        out = X.run_experiment(
            backend2, formats[:1], split, boundary=(7, 8), max_new_tokens=8, seed=100
        )
        self.assertEqual(out["seed"], 100)
        self.assertTrue(set(backend2.seeds).isdisjoint(set(backend.seeds)))


# ---------------------------------------------------------------------------
# Red team: hidden assumptions and contract edges
# ---------------------------------------------------------------------------


class TestRedTeam(unittest.TestCase):
    def test_pass_strictness_divergence(self):
        # D2: classify_observation uses truthiness; budget_frontier needs
        # `is True`. A truthy-but-not-True exact_match passes classification
        # yet earns no budget.
        obs = {
            "exact_match": 1,
            "boundary_found": True,
            "status": "completed",
            "output_ids": (5, 1, 2),
        }
        self.assertIsNone(X.classify_observation(obs, None))
        search = [
            {
                "format": {"name": "a"},
                "observations": [dict(obs, fixture="f0")],
            }
        ]
        got = X.budget_frontier(
            search,
            boundary=(1, 2),
            max_new_tokens=8,
            native_decision="no_candidate",
            native_winner=None,
        )
        self.assertEqual(got["candidates"]["a"]["min_budget_per_fixture"], [None])

    def test_taxonomy_duplicate_names_last_wins(self):
        # D3: failure_taxonomy silently overwrites duplicate names while
        # budget_frontier raises on them.
        obs = {
            "exact_match": False,
            "boundary_found": False,
            "status": "completed",
            "output_ids": (1,),
        }
        result = {
            "search": [
                {"format": {"name": "dup"}, "observations": [obs]},
                {"format": {"name": "dup"}, "observations": []},
            ],
            "decision": "no_candidate",
        }
        got = X.failure_taxonomy(result, None)
        self.assertEqual(got["per_candidate"]["dup"], dict.fromkeys(FAILURE_MODES, 0))
        with self.assertRaises(ValueError):
            X.budget_frontier(
                result["search"],
                boundary=(1,),
                max_new_tokens=8,
                native_decision="no_candidate",
                native_winner=None,
            )

    def test_stop_matrix_noniterable_ids_raise(self):
        # D4a: stop_matrix assumes tuple/list ids in recorded observations.
        backend = ByteBackend()
        fixtures = (
            Fixture(
                id="f0", messages=(Message(role="user", content="q"),), expected="x", group="g"
            ),
        )
        obs = (
            {
                "fixture": "f0",
                "status": "completed",
                "input_ids": (1,),
                "output_ids": 5,
                "exact_match": False,
                "boundary_found": False,
            },
        )
        with self.assertRaises(TypeError):
            X.stop_matrix(backend, "c", fixtures, obs, boundary=(1,))

    def test_budget_frontier_nonsequence_ids_silent(self):
        # D4b: budget_frontier leniently treats non-list/tuple output_ids
        # as empty (no budget, no error).
        search = [
            {
                "format": {"name": "a"},
                "observations": [
                    {
                        "fixture": "f0",
                        "status": "completed",
                        "exact_match": True,
                        "boundary_found": True,
                        "output_ids": 5,
                    }
                ],
            }
        ]
        got = X.budget_frontier(
            search,
            boundary=(1,),
            max_new_tokens=8,
            native_decision="no_candidate",
            native_winner=None,
        )
        self.assertEqual(got["candidates"]["a"]["min_budget_per_fixture"], [None])

    def test_stop_matrix_strict_bool_rejection(self):
        # type() is X checks: bool is not int / not bool here.
        backend = ByteBackend()
        fixtures = (
            Fixture(
                id="f0", messages=(Message(role="user", content="q"),), expected="x", group="g"
            ),
        )
        obs = (
            {
                "fixture": "f0",
                "status": "completed",
                "input_ids": (1,),
                "output_ids": (65, 66),
                "exact_match": True,
                "boundary_found": True,
            },
        )
        with self.assertRaises(ValueError):
            X.stop_matrix(backend, "c", fixtures, obs, boundary=(65, 66), vllm_min_tokens=True)
        with self.assertRaises(ValueError):
            X.stop_matrix(
                backend,
                "c",
                fixtures,
                obs,
                boundary=(65, 66),
                vllm_include_stop_str_in_output=1,
            )
        # ... but a bool token inside output_ids just marks the observation
        # incomplete (type(token) is int fails) rather than raising.
        obs2 = (
            {
                "fixture": "f0",
                "status": "completed",
                "input_ids": (1,),
                "output_ids": (True, 66),
                "exact_match": False,
                "boundary_found": False,
            },
        )
        rows = X.stop_matrix(backend, "c", fixtures, obs2, boundary=(65, 66))
        self.assertEqual(rows[0]["cells"]["hf_stop_string"]["detail"], "observation_incomplete")

    def test_budget_frontier_bool_boundary_ids_rejected(self):
        search = [
            {
                "format": {"name": "a"},
                "observations": [
                    {
                        "fixture": "f0",
                        "status": "completed",
                        "exact_match": True,
                        "boundary_found": True,
                        "output_ids": [1, 2],
                    }
                ],
                "boundary_token_ids": [1, True],
            }
        ]
        with self.assertRaises(ValueError):
            X.budget_frontier(
                search,
                boundary=(1,),
                max_new_tokens=8,
                native_decision="no_candidate",
                native_winner=None,
            )

    def test_taxonomy_skips_malformed_entries(self):
        # Non-dict format values and missing names are skipped, not errors.
        result = {
            "search": [
                {"format": "just-a-string", "observations": []},
                {"format": {"name": 42}, "observations": []},
                {"format": {}, "observations": []},
                {"observations": []},
                "junk",
                None,
            ],
            "decision": "no_candidate",
        }
        got = X.failure_taxonomy(result, None)
        self.assertEqual(got["per_candidate"], {})

    def test_diagnostic_skips_malformed_entries(self):
        backend = LogprobBackend()
        fixtures = (
            Fixture(
                id="f0", messages=(Message(role="user", content="q"),), expected="AB", group="g"
            ),
        )
        result = {
            "search": [
                "junk",
                {"format": "s", "observations": []},
                {"format": {"name": 7}, "observations": []},
                {"format": {"name": "ok"}, "observations": "not-a-list"},
                {
                    "format": {"name": "good"},
                    "observations": [
                        {
                            "fixture": "f0",
                            "status": "completed",
                            "input_ids": list(backend.encode("prompt")),
                            "output_ids": [65, 66],
                            "exact_match": False,
                            "boundary_found": False,
                        }
                    ],
                },
            ]
        }
        got = X.logprob_diagnostic(backend, result, fixtures)
        self.assertTrue(got["available"])
        self.assertEqual([c["name"] for c in got["candidates"]], ["good"])

    def test_decision_steps_start_at_budget_one(self):
        # change_budgets always contains 1 and max_new_tokens.
        search = [
            {
                "format": {"name": "a"},
                "observations": [
                    {
                        "fixture": "f0",
                        "status": "completed",
                        "exact_match": True,
                        "boundary_found": True,
                        "output_ids": [5],
                    }
                ],
            }
        ]
        got = X.budget_frontier(
            search,
            boundary=(5,),
            max_new_tokens=16,
            native_decision="compatible_on_fixture_set",
            native_winner="a",
        )
        # decision_steps records only changes: one step at budget 1.
        self.assertEqual(got["decision_steps"], [[1, "selected", "a"]])
        self.assertEqual(got["candidates"]["a"]["pass_steps"], [[1, 1]])
        self.assertTrue(got["budget_robust"])
        self.assertEqual(got["flip_ranges"], [])

    def test_frontier_winner_must_be_str(self):
        search = [
            {
                "format": {"name": "a"},
                "observations": [
                    {
                        "fixture": "f0",
                        "status": "completed",
                        "exact_match": True,
                        "boundary_found": True,
                        "output_ids": [5],
                    }
                ],
            }
        ]
        with self.assertRaises(ValueError):
            X.budget_frontier(
                search,
                boundary=(5,),
                max_new_tokens=16,
                native_decision="compatible_on_fixture_set",
                native_winner=5,
            )


# ---------------------------------------------------------------------------
# Sensitivity-runner helper (used by /tmp/mut34/run_mutants.py)
# ---------------------------------------------------------------------------


def mutant_anchor_check():
    """Return the (path, [(old, new)]) mutant list for the /tmp runner.

    Kept here so the anchors live next to the tests that must catch them.
    """
    return [
        (
            "M1-classify-cancelled-branch",
            """    if obs.get("status") == "cancelled":
        # Operational, not a measurement: grouped with backend-error; the run
        # decision already records "cancelled" separately.
        return "backend-error"
""",
            "",
        ),
        (
            "M2-classify-eos-mode",
            '            return "eos-without-boundary"',
            '            return "no-boundary-in-N"',
        ),
        (
            "M3-taxonomy-heldout-gating",
            '    if result.get("decision") == "held_out_failed":',
            '    if result.get("decision") is not None:',
        ),
        (
            "M4-budget-min-budget-offbyone",
            "                        min_budget = cut + len(entry_boundary)",
            "                        min_budget = cut + len(entry_boundary) + 1",
        ),
        (
            "M5-budget-tie-breaks-to-first",
            '        if len(winners) == 1:\n            return ("selected", winners[0])',
            '        if len(winners) >= 1:\n            return ("selected", winners[0])',
        ),
        (
            "M6-stopcell-drops-boundary-requirement",
            '        "pass": bool(cut is not None and answer == fixture.expected),',
            '        "pass": bool(answer == fixture.expected),',
        ),
        (
            "M7-stopmatrix-flips-inverted",
            '            if cells[semantic].get("available") '
            'and cells[semantic].get("pass") != native_pass',
            '            if cells[semantic].get("available") '
            'and cells[semantic].get("pass") == native_pass',
        ),
        (
            "M8-aggregate-includes-non-ok",
            "    for score in scores:\n        if score.status != STATUS_OK:\n            continue",
            "    for score in scores:\n        if False:  # mutant: no skip\n            continue",
        ),
        (
            "M9-interpret-threshold-swap",
            "    highs = [s for s in scored "
            'if s["mean_logprob"] >= LOGPROB_DERAILED_MEAN_THRESHOLD]',
            "    highs = [s for s in scored "
            'if s["mean_logprob"] >= LOGPROB_UNKNOWN_MEAN_THRESHOLD]',
        ),
        (
            "M10-fingerprint-no-bos-normalization",
            '    payload = asdict(fmt)\n    if not payload["bos"]:\n        payload["repeat_bos"] = False\n',  # noqa: E501 -- mutant anchor data lines
            "    payload = asdict(fmt)\n",
        ),
        (
            "M11-resolve-drops-setdefault",
            "    for fmt in formats:\n        resolved.setdefault(fmt.name, boundary)\n",
            "",
        ),
        (
            "M12-sweep-flips_at-inverted",
            '    "flips_at": [depth["drop_bytes"] for depth in depths if depth["winner_changed"]],',
            '    "flips_at": [depth["drop_bytes"] for depth in depths '
            'if not depth["winner_changed"]],',
        ),
        (
            "M13-sweep-drops-last-depth",
            "    for drop in range(1, max_drop_bytes + 1):",
            "    for drop in range(1, max_drop_bytes):",
        ),
        (
            "M14-diagnostic-scores-failed-obs",
            '    if type(obs) is not dict or obs.get("status") != "completed":\n        return not_scored, None',  # noqa: E501 -- mutant anchor data lines
            "    if type(obs) is not dict:\n        return not_scored, None",
        ),
        (
            "M15-diagnostic-candidates-gate-inverted",
            '    if not payload["candidates"]:\n'
            '        payload["detail"] = "no_scorable_observations"\n        return payload',
            '    if payload["candidates"]:\n'
            '        payload["detail"] = "no_scorable_observations"\n        return payload',
        ),
    ]


if __name__ == "__main__":
    unittest.main()
