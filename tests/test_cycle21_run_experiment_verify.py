"""Differential verification of ``run_experiment`` orchestration (cycle 21).

Target: ``model_protocol_lab.experiments.run_experiment`` (~line 941).

Method: an INDEPENDENT oracle reimplements the orchestration from the
documented behavior only -- the search loop over candidates, the conservative
unique-winner selection rule, held-out gating, decision strings, the result
assembly, and the failure taxonomy -- and its output is compared field by
field against the real implementation on scripted, deterministic backends.

BRIEF-VS-DOCSTRING DISCREPANCIES (recorded per the task):
1. ``run_experiment`` has NO docstring (verified at experiments.py:941: the
   body starts immediately after the signature). The brief's instruction to
   "read its docstring" therefore cannot be followed; the oracle is grounded
   instead in the inline comments inside ``run_experiment`` plus the
   docstrings of ``_resolve_boundaries``, ``evaluate`` (core.py), ``_summary``
   (no docstring -- evident from its code), ``failure_taxonomy``,
   ``classify_observation``, ``_failure_code`` and the ``Backend`` protocol.
2. ``evaluate`` is defined in ``core.py`` (line 651), not ``experiments.py``
   -- the brief's warning is correct and was verified, not trusted.
3. The brief's fuzz range "0-3 held-out fixtures" is unreachable through the
   public API: ``Split.__post_init__`` requires BOTH splits to be nonempty
   tuples, and ``evaluate`` requires a nonempty cases tuple. Fuzzing uses
   1-3 held-out fixtures instead; an empty held-out split raises ValueError
   before any orchestration runs.
4. ``run_experiment`` allows 1-24 candidates; the brief's "2-5" range is a
   subset and is honored.

The oracle deliberately shares only logic-free leaf helpers with the
implementation (``first_boundary``, ``asdict``, ``sha256``, the
``Observation``/``Generation`` constructors, ``FAILURE_MODES``): everything
that branches -- the evaluate loop sequencing, the winner rule, held-out
gating, decision strings, assembly shape, taxonomy classification -- is
reimplemented from the documentation above.
"""

import json
import random
import re
import shutil
import subprocess
import sys
import unittest
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event

from model_protocol_lab.core import (
    Fixture,
    Format,
    Generation,
    Message,
    Observation,
    Split,
    first_boundary,
)
from model_protocol_lab.experiments import (
    FAILURE_MODES,
    candidate_fingerprint,
    run_experiment,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = Path(__file__).resolve().parent

DEFAULT_BOUNDARY = (2000000,)
PAD_ID = 32  # chr(32) == " ", never collides with the huge boundary ids

# Scripted per-fixture outcomes. "pass": exact answer + boundary present.
# "wrong_answer": boundary present, answer wrong. "no_boundary": answer text
# right (modulo padding) but no boundary emitted. "both": neither.
OUTCOMES = ("pass", "wrong_answer", "no_boundary", "both")


class Plan:
    """Scripted fixture: which candidates pass which fixtures."""

    def __init__(self):
        self.outcomes = {}  # (cand_idx, (tag, j)) -> outcome
        self.expected = {}  # (tag, j) -> expected answer text
        self.boundaries = {}  # cand_idx -> boundary tuple used when emitting
        self.fxkey = {}  # fixture id -> (tag, j) script key
        self.cand_index = {}  # candidate name -> cand_idx
        self.cancel_after = None  # set the cancel event after this many generate calls


def scripted_token_ids(backend, plan, cand_idx, fxkey, cand_boundary, max_new_tokens):
    """Pure function of the scripted outcome: the tokens generate() returns.

    Shared by the ScriptedBackend and the oracle: the backend IS the fixture,
    so both sides must agree on what a scripted outcome materializes as. The
    oracle's independence lies in the orchestration logic, not in this fixture.
    """
    tag, j = fxkey
    expected = plan.expected[fxkey]
    outcome = plan.outcomes[(cand_idx, fxkey)]
    answer_text = expected if outcome in ("pass", "no_boundary") else f"WRONG:{tag}{j}"
    ids = backend.encode(answer_text)
    if outcome in ("pass", "wrong_answer"):
        ids = ids + tuple(cand_boundary)
    if len(ids) > max_new_tokens:
        raise AssertionError("scripted fixture exceeds the token cap")
    return ids + (PAD_ID,) * (max_new_tokens - len(ids))


class ScriptedBackend:
    """Deterministic backend: generate() is a pure function of the prompt.

    encode/decode are a trivial chr-based codec (no implicit BOS/EOS). The
    prompt carries ``<cN>`` (candidate) and ``<sN>``/``<hN>`` (fixture) markers
    so the scripted outcome can be looked up per (candidate, fixture).
    """

    fingerprint = "scripted-test-backend"

    def __init__(self, plan):
        self.plan = plan
        self.calls = 0

    def encode(self, text):
        return tuple(ord(c) for c in text)

    def decode(self, token_ids):
        return "".join(chr(t) for t in token_ids)

    def generate(self, input_ids, *, max_new_tokens, seed, cancel):
        self.calls += 1
        text = self.decode(input_ids)
        cand_idx = int(re.search(r"<c(\d+)>", text).group(1))
        m = re.search(r"<([sh])(\d+)>", text)
        fxkey = (m.group(1), int(m.group(2)))
        if self.plan.cancel_after is not None and self.calls == self.plan.cancel_after:
            cancel.set()
        ids = scripted_token_ids(
            self, self.plan, cand_idx, fxkey, self.plan.boundaries[cand_idx], max_new_tokens
        )
        return Generation(token_ids=ids, reason="limit")


# ---------------------------------------------------------------------------
# Independent oracle, reimplemented from the documented behavior.
# ---------------------------------------------------------------------------


def oracle_resolve_boundaries(formats, boundary, boundary_map):
    """Per _resolve_boundaries docstring: default boundary for every candidate,
    boundary_map overrides for named candidates; unknown names, non-dict maps
    and empty/non-tuple boundaries raise ValueError."""
    if boundary_map is None:
        return {fmt.name: boundary for fmt in formats}
    if type(boundary_map) is not dict:
        raise ValueError("boundary_map must be a dict or None")
    names = {fmt.name for fmt in formats}
    resolved = {}
    for name, ids in boundary_map.items():
        if type(name) is not str or name not in names:
            raise ValueError(f"boundary_map names an unknown candidate: {name!r}")
        if not (type(ids) is tuple and bool(ids) and all(type(t) is int and t >= 0 for t in ids)):
            raise ValueError(f"boundary_map boundary for {name!r} must be a nonempty tuple")
        resolved[name] = ids
    for fmt in formats:
        resolved.setdefault(fmt.name, boundary)
    return resolved


def _oracle_obs(
    candidate,
    fixture,
    input_ids,
    output_ids,
    answer,
    cut,
    raw_reason,
    error,
    status,
    boundary_text_match=None,
):
    return Observation(
        candidate=candidate,
        fixture=fixture.id,  # evaluate records case.id, not the Fixture object
        input_ids=input_ids,
        output_ids=output_ids,
        answer=answer,
        exact_match=status == "completed" and answer == fixture.expected,
        boundary_found=cut is not None,
        boundary_text_match=boundary_text_match,
        raw_reason=raw_reason,
        error=error,
        status=status,
    )


def oracle_boundary_text_match(backend, output_ids, cut, boundary):
    """Per _boundary_text_match docstring: diagnostic only, never part of the
    pass criterion; any backend failure degrades to None."""
    try:
        boundary_text = backend.decode(boundary)
    except Exception:  # noqa: BLE001 -- mirrors _boundary_text_match: diagnostic degrades to None
        return None
    if type(boundary_text) is not str or not boundary_text:
        return None
    retained = output_ids if cut is None else output_ids[: cut + len(boundary)]
    try:
        retained_text = backend.decode(retained)
    except Exception:  # noqa: BLE001 -- mirrors _boundary_text_match: diagnostic degrades to None
        return None
    if type(retained_text) is not str:
        return None
    return retained_text.endswith(boundary_text)


def oracle_evaluate(
    backend, plan, fmt, cases, *, cand_boundary, max_new_tokens, seed, event, counter
):
    """Replicates evaluate's documented loop: per-fixture render -> encode ->
    generate -> boundary cut -> decode, with cancellation checks at each
    checkpoint, retaining incomplete-run evidence. Returns (records, status)."""
    records = []
    status = None
    for case in cases:
        if event.is_set():
            status = "cancelled"
            break
        prompt = fmt.render(case.messages)
        if event.is_set():
            records.append(_oracle_obs(fmt.name, case, (), (), "", None, None, None, "cancelled"))
            status = "cancelled"
            break
        input_ids = backend.encode(prompt)
        if event.is_set():
            records.append(
                _oracle_obs(fmt.name, case, input_ids, (), "", None, None, None, "cancelled")
            )
            status = "cancelled"
            break
        counter[0] += 1
        if plan.cancel_after is not None and counter[0] == plan.cancel_after:
            event.set()
        output_ids = scripted_token_ids(
            backend,
            plan,
            plan.cand_index[fmt.name],
            plan.fxkey[case.id],
            cand_boundary,
            max_new_tokens,
        )
        raw_reason = "limit"
        cut = first_boundary(output_ids, cand_boundary)
        if event.is_set():
            records.append(
                _oracle_obs(
                    fmt.name, case, input_ids, output_ids, "", cut, raw_reason, None, "cancelled"
                )
            )
            status = "cancelled"
            break
        answer_ids = output_ids if cut is None else output_ids[:cut]
        answer = backend.decode(answer_ids)
        if event.is_set():
            records.append(
                _oracle_obs(
                    fmt.name,
                    case,
                    input_ids,
                    output_ids,
                    answer,
                    cut,
                    raw_reason,
                    None,
                    "cancelled",
                )
            )
            status = "cancelled"
            break
        text_match = oracle_boundary_text_match(backend, output_ids, cut, cand_boundary)
        records.append(
            _oracle_obs(
                fmt.name,
                case,
                input_ids,
                output_ids,
                answer,
                cut,
                raw_reason,
                None,
                "completed",
                text_match,
            )
        )
    else:
        status = "cancelled" if event.is_set() else "completed"
    return records, status


def oracle_fingerprint(fmt):
    """Per candidate_fingerprint docstring: sha256 over the normalized asdict
    payload (repeat_bos normalized to False when bos is empty)."""
    payload = asdict(fmt)
    if not payload["bos"]:
        payload["repeat_bos"] = False
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + sha256(text.encode("utf-8")).hexdigest()


def oracle_failure_code(records):
    """Per _failure_code docstring: scan observations from the end for the
    first fixed-vocabulary error code; None when nothing failed."""
    for obs in reversed(records):
        error = obs.error
        if type(error) is str and error:
            return error
    return None


def oracle_summary(records, planned_count, status):
    """Per _summary's evident behavior."""
    eligible = status == "completed" and len(records) == planned_count
    passed = sum(o.exact_match and o.boundary_found for o in records)
    return {
        "status": status,
        "planned": planned_count,
        "attempted": len(records),
        "aggregate_eligible": eligible,
        "exact_and_boundary_passes": passed,
        "rate": passed / planned_count if eligible else None,
        "encoded_input_tokens": sum(len(o.input_ids) for o in records),
        "retained_output_tokens": sum(len(o.output_ids) for o in records),
        "failure_code": oracle_failure_code(records),
    }


def oracle_classify(obs, eos_id):
    """Per classify_observation docstring: closed taxonomy; backend failures
    first (cancelled groups with backend-error), then the tokenization-mismatch
    diagnostic, then EOS-without-boundary, then no-boundary-in-N. A wrong
    answer with a found boundary is no-boundary-in-N."""
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


def oracle_failure_taxonomy(result, eos_id):
    """Per failure_taxonomy docstring: per-candidate classification of search
    observations; held-out classified only when the decision is
    held_out_failed, counting every non-passing held-out observation as
    held-out-divergence."""
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


def oracle_run_experiment(
    backend, plan, formats, split, *, boundary, boundary_map, max_new_tokens, seed, eos_id
):
    """Independent reimplementation of run_experiment's orchestration, from the
    inline comments and helper docstrings: search every candidate, select iff
    exactly one candidate passes every search fixture (ties/failures abstain),
    run held-out for the winner, attach the failure taxonomy, assemble the
    result dict with config echoes."""
    boundaries = oracle_resolve_boundaries(formats, boundary, boundary_map)
    event = Event()
    counter = [0]
    output = {
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

    def finish():
        output["failure_taxonomy"] = oracle_failure_taxonomy(output, eos_id)
        return output

    winners = []
    for fmt in formats:
        cand_boundary = boundaries[fmt.name]
        records, status = oracle_evaluate(
            backend,
            plan,
            fmt,
            split.search,
            cand_boundary=cand_boundary,
            max_new_tokens=max_new_tokens,
            seed=seed,
            event=event,
            counter=counter,
        )
        eligible = status == "completed" and len(records) == len(split.search)
        summary = oracle_summary(records, len(split.search), status)
        output["search"].append(
            {
                "format": asdict(fmt),
                "candidate_fingerprint": oracle_fingerprint(fmt),
                "boundary_token_ids": list(cand_boundary),
                "summary": summary,
                "observations": [asdict(o) for o in records],
            }
        )
        if not eligible:
            output["decision"] = status
            return finish()
        if summary["exact_and_boundary_passes"] == len(split.search):
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
    held_records, held_status = oracle_evaluate(
        backend,
        plan,
        selected,
        split.held_out,
        cand_boundary=selected_boundary,
        max_new_tokens=max_new_tokens,
        seed=seed,
        event=event,
        counter=counter,
    )
    output["held_out"] = {
        "summary": oracle_summary(held_records, len(split.held_out), held_status),
        "observations": [asdict(o) for o in held_records],
    }
    held_eligible = held_status == "completed" and len(held_records) == len(split.held_out)
    if not held_eligible:
        output["decision"] = held_status
    else:
        passed = all(o.exact_match and o.boundary_found for o in held_records)
        output["decision"] = "compatible_on_fixture_set" if passed else "held_out_failed"
    return finish()


# ---------------------------------------------------------------------------
# Scenario builders.
# ---------------------------------------------------------------------------


def make_format(i):
    return Format(
        name=f"cand-{i}",
        bos="<bos>",
        user_prefix=f"<c{i}>U:",
        assistant_prefix=f"<c{i}>A:",
        turn_end="</t>",
    )


def make_fixture(tag, j):
    return Fixture(
        id=f"fx-{tag}{j}",
        messages=(Message("user", f"<{tag}{j}> question-{tag}{j}"),),
        expected=f"answer-{tag}{j}",
        group="search" if tag == "s" else "held",
    )


def full_matrix(n_cand, n_fx, tag, default, overrides=()):
    m = {(i, (tag, j)): default for i in range(n_cand) for j in range(n_fx)}
    m.update(overrides)
    return m


class Case:
    """Everything needed to run one differential comparison."""

    def __init__(
        self,
        *,
        n_cand,
        n_search,
        n_held,
        search_outcomes,
        held_outcomes,
        boundary_map=None,
        cancel_after=None,
        max_new_tokens=16,
        seed=0,
        eos_id=None,
    ):
        self.formats = tuple(make_format(i) for i in range(n_cand))
        search = tuple(make_fixture("s", j) for j in range(n_search))
        held = tuple(make_fixture("h", j) for j in range(n_held))
        self.split = Split(search=search, held_out=held)
        self.plan = Plan()
        self.plan.outcomes.update(search_outcomes)
        self.plan.outcomes.update(held_outcomes)
        for tag, n in (("s", n_search), ("h", n_held)):
            for j in range(n):
                self.plan.expected[(tag, j)] = f"answer-{tag}{j}"
                self.plan.fxkey[f"fx-{tag}{j}"] = (tag, j)
        self.plan.cand_index = {f"cand-{i}": i for i in range(n_cand)}
        for i in range(n_cand):
            name = f"cand-{i}"
            self.plan.boundaries[i] = (
                boundary_map[name] if boundary_map and name in boundary_map else DEFAULT_BOUNDARY
            )
        self.plan.cancel_after = cancel_after
        self.backend = ScriptedBackend(self.plan)
        self.boundary = DEFAULT_BOUNDARY
        self.boundary_map = boundary_map
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.eos_id = eos_id

    def run_actual(self):
        return run_experiment(
            self.backend,
            self.formats,
            self.split,
            boundary=self.boundary,
            boundary_map=self.boundary_map,
            max_new_tokens=self.max_new_tokens,
            seed=self.seed,
            eos_id=self.eos_id,
        )

    def run_expected(self):
        return oracle_run_experiment(
            self.backend,
            self.plan,
            self.formats,
            self.split,
            boundary=self.boundary,
            boundary_map=self.boundary_map,
            max_new_tokens=self.max_new_tokens,
            seed=self.seed,
            eos_id=self.eos_id,
        )


def diff_paths(actual, expected, path="result"):
    """Recursive field-by-field diff; [] means the dicts are identical."""
    diffs = []

    def rec(a, e, p):
        if isinstance(a, dict) and isinstance(e, dict):
            for key in sorted(set(a) | set(e), key=repr):
                if key not in a:
                    diffs.append((f"{p}[{key!r}]", "<missing>", e[key]))
                elif key not in e:
                    diffs.append((f"{p}[{key!r}]", a[key], "<missing>"))
                else:
                    rec(a[key], e[key], f"{p}[{key!r}]")
        elif isinstance(a, (list, tuple)) and isinstance(e, (list, tuple)):
            if len(a) != len(e):
                diffs.append((p, f"<len {len(a)}>", f"<len {len(e)}>"))
            else:
                for i, (x, y) in enumerate(zip(a, e, strict=True)):
                    rec(x, y, f"{p}[{i}]")
        else:
            if type(a) is not type(e) or a != e:
                diffs.append((p, a, e))

    rec(actual, expected, path)
    return diffs


def check_case(case):
    """Run the implementation and the oracle; return (mismatches, actual).

    The implementation runs exactly once: the ScriptedBackend counts generate
    calls (for cancel_after), so a second run would observe different state.
    """
    actual = case.run_actual()
    expected = case.run_expected()
    return diff_paths(actual, expected), actual


# ---------------------------------------------------------------------------
# Directed scenarios covering every mandated outcome shape.
# ---------------------------------------------------------------------------


class RunExperimentDirected(unittest.TestCase):
    def assert_oracle_agrees(self, case, label):
        diffs, actual = check_case(case)
        self.assertEqual(
            diffs,
            [],
            f"oracle/implementation mismatch in {label}:\n"
            + "\n".join(f"  {p}: actual={a!r} expected={e!r}" for p, a, e in diffs[:10]),
        )
        return actual

    def test_unique_winner_compatible(self):
        n_cand, n_search, n_held = 3, 4, 2
        search = full_matrix(
            n_cand,
            n_search,
            "s",
            "pass",
            {
                (1, ("s", 1)): "wrong_answer",
                (2, ("s", 0)): "no_boundary",
            },
        )
        held = full_matrix(n_cand, n_held, "h", "pass")
        case = Case(
            n_cand=n_cand,
            n_search=n_search,
            n_held=n_held,
            search_outcomes=search,
            held_outcomes=held,
            seed=11,
        )
        result = self.assert_oracle_agrees(case, "unique winner")
        self.assertEqual(result["decision"], "compatible_on_fixture_set")
        self.assertEqual(result["selected_candidate"], "cand-0")
        self.assertIsNotNone(result["held_out"])
        self.assertEqual(len(result["held_out"]["observations"]), n_held)

    def test_unique_winner_held_out_failed(self):
        n_cand, n_search, n_held = 3, 3, 2
        search = full_matrix(
            n_cand,
            n_search,
            "s",
            "pass",
            {
                (1, ("s", 2)): "both",
                (2, ("s", 0)): "wrong_answer",
            },
        )
        held = full_matrix(n_cand, n_held, "h", "pass", {(0, ("h", 1)): "no_boundary"})
        case = Case(
            n_cand=n_cand,
            n_search=n_search,
            n_held=n_held,
            search_outcomes=search,
            held_outcomes=held,
            seed=12,
        )
        result = self.assert_oracle_agrees(case, "held_out_failed")
        self.assertEqual(result["decision"], "held_out_failed")
        self.assertEqual(result["selected_candidate"], "cand-0")
        tax = result["failure_taxonomy"]
        self.assertEqual(tax["held_out"]["held-out-divergence"], 1)

    def test_two_way_tie_ambiguous(self):
        n_cand, n_search, n_held = 3, 3, 1
        search = full_matrix(n_cand, n_search, "s", "pass", {(2, ("s", 1)): "wrong_answer"})
        held = full_matrix(n_cand, n_held, "h", "pass")
        case = Case(
            n_cand=n_cand,
            n_search=n_search,
            n_held=n_held,
            search_outcomes=search,
            held_outcomes=held,
            seed=13,
        )
        result = self.assert_oracle_agrees(case, "two-way tie")
        self.assertEqual(result["decision"], "ambiguous")
        self.assertIsNone(result["selected_candidate"])
        self.assertIsNone(result["held_out"])

    def test_three_way_tie_ambiguous(self):
        n_cand, n_search, n_held = 4, 2, 1
        search = full_matrix(n_cand, n_search, "s", "pass", {(3, ("s", 0)): "both"})
        held = full_matrix(n_cand, n_held, "h", "pass")
        case = Case(
            n_cand=n_cand,
            n_search=n_search,
            n_held=n_held,
            search_outcomes=search,
            held_outcomes=held,
            seed=14,
        )
        result = self.assert_oracle_agrees(case, "three-way tie")
        self.assertEqual(result["decision"], "ambiguous")
        self.assertNotIn("selected_boundary_token_ids", result)

    def test_no_candidate_passes_all(self):
        n_cand, n_search, n_held = 3, 3, 1
        search = full_matrix(
            n_cand,
            n_search,
            "s",
            "pass",
            {
                (0, ("s", 0)): "wrong_answer",
                (1, ("s", 1)): "no_boundary",
                (2, ("s", 2)): "both",
            },
        )
        held = full_matrix(n_cand, n_held, "h", "pass")
        case = Case(
            n_cand=n_cand,
            n_search=n_search,
            n_held=n_held,
            search_outcomes=search,
            held_outcomes=held,
            seed=15,
        )
        result = self.assert_oracle_agrees(case, "no candidate")
        self.assertEqual(result["decision"], "no_candidate")
        self.assertIsNone(result["selected_candidate"])

    def test_single_search_fixture(self):
        n_cand, n_search, n_held = 2, 1, 1
        search = full_matrix(n_cand, n_search, "s", "pass", {(1, ("s", 0)): "no_boundary"})
        held = full_matrix(n_cand, n_held, "h", "pass")
        case = Case(
            n_cand=n_cand,
            n_search=n_search,
            n_held=n_held,
            search_outcomes=search,
            held_outcomes=held,
            seed=16,
        )
        result = self.assert_oracle_agrees(case, "single search fixture")
        self.assertEqual(result["decision"], "compatible_on_fixture_set")
        self.assertEqual(result["selected_candidate"], "cand-0")

    def test_boundary_map_per_candidate(self):
        n_cand, n_search, n_held = 3, 3, 2
        bmap = {"cand-0": (2000001,), "cand-1": (2000002, 2000003)}
        search = full_matrix(
            n_cand,
            n_search,
            "s",
            "pass",
            {
                (1, ("s", 0)): "wrong_answer",
                (2, ("s", 2)): "no_boundary",
            },
        )
        held = full_matrix(n_cand, n_held, "h", "pass")
        case = Case(
            n_cand=n_cand,
            n_search=n_search,
            n_held=n_held,
            search_outcomes=search,
            held_outcomes=held,
            boundary_map=bmap,
            seed=17,
        )
        result = self.assert_oracle_agrees(case, "boundary_map")
        self.assertEqual(result["decision"], "compatible_on_fixture_set")
        self.assertEqual(result["selected_candidate"], "cand-0")
        self.assertEqual(result["selected_boundary_token_ids"], [2000001])
        by_name = {e["format"]["name"]: e for e in result["search"]}
        self.assertEqual(by_name["cand-0"]["boundary_token_ids"], [2000001])
        self.assertEqual(by_name["cand-1"]["boundary_token_ids"], [2000002, 2000003])
        self.assertEqual(by_name["cand-2"]["boundary_token_ids"], [2000000])

    def test_cancel_mid_search(self):
        n_cand, n_search, n_held = 3, 4, 2
        search = full_matrix(n_cand, n_search, "s", "pass")
        held = full_matrix(n_cand, n_held, "h", "pass")
        case = Case(
            n_cand=n_cand,
            n_search=n_search,
            n_held=n_held,
            search_outcomes=search,
            held_outcomes=held,
            cancel_after=3,
            seed=18,
        )
        result = self.assert_oracle_agrees(case, "cancel mid-search")
        self.assertEqual(result["decision"], "cancelled")
        self.assertIsNone(result["selected_candidate"])
        # Cancel fired during cand-0's 3rd fixture: 2 completed + 1 cancelled.
        entry = result["search"][0]
        self.assertEqual(entry["summary"]["attempted"], 3)
        self.assertEqual(
            [o["status"] for o in entry["observations"]],
            ["completed", "completed", "cancelled"],
        )
        # The failure taxonomy still attaches on the cancelled path.
        self.assertIn("failure_taxonomy", result)

    def test_cancel_during_held_out(self):
        n_cand, n_search, n_held = 2, 2, 3
        search = full_matrix(n_cand, n_search, "s", "pass", {(1, ("s", 0)): "wrong_answer"})
        held = full_matrix(n_cand, n_held, "h", "pass")
        # 4 search generates, then cancel on the 2nd held-out fixture.
        case = Case(
            n_cand=n_cand,
            n_search=n_search,
            n_held=n_held,
            search_outcomes=search,
            held_outcomes=held,
            cancel_after=6,
            seed=19,
        )
        result = self.assert_oracle_agrees(case, "cancel during held-out")
        self.assertEqual(result["decision"], "cancelled")
        self.assertEqual(result["selected_candidate"], "cand-0")
        self.assertEqual(
            [o["status"] for o in result["held_out"]["observations"]],
            ["completed", "cancelled"],
        )

    def test_config_echoes(self):
        n_cand, n_search, n_held = 2, 2, 1
        search = full_matrix(n_cand, n_search, "s", "pass", {(1, ("s", 1)): "both"})
        held = full_matrix(n_cand, n_held, "h", "pass")
        case = Case(
            n_cand=n_cand,
            n_search=n_search,
            n_held=n_held,
            search_outcomes=search,
            held_outcomes=held,
            max_new_tokens=24,
            seed=12345,
            eos_id=7,
        )
        result = self.assert_oracle_agrees(case, "config echoes")
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["selection_rule"], "unique_all_search_pass")
        self.assertEqual(result["measurement"], "exact_answer_and_boundary")
        self.assertEqual(result["seed"], 12345)
        self.assertEqual(result["max_new_tokens"], 24)
        self.assertEqual(result["boundary_token_ids"], [2000000])
        self.assertEqual(result["eos_token_id"], 7)
        self.assertEqual(result["backend_fingerprint"], "scripted-test-backend")
        self.assertEqual(result["statistical_confidence"], "not_estimated")
        self.assertEqual(result["compute_comparison"], "not_equal_compute_certified")
        tax = result["failure_taxonomy"]
        self.assertTrue(tax["eos_id_considered"])
        self.assertEqual(tax["modes"], list(FAILURE_MODES))

    def test_oracle_fingerprint_matches_reference(self):
        """The oracle's docstring-derived fingerprint must equal the real one."""
        for i in range(4):
            fmt = make_format(i)
            self.assertEqual(oracle_fingerprint(fmt), candidate_fingerprint(fmt))


class RunExperimentValidation(unittest.TestCase):
    """Input-validation parity between the implementation and the oracle."""

    def test_boundary_map_unknown_name(self):
        case = Case(
            n_cand=2,
            n_search=1,
            n_held=1,
            search_outcomes=full_matrix(2, 1, "s", "pass"),
            held_outcomes=full_matrix(2, 1, "h", "pass"),
            boundary_map={"nope": (1,)},
        )
        with self.assertRaises(ValueError):
            case.run_actual()
        with self.assertRaises(ValueError):
            oracle_resolve_boundaries(case.formats, case.boundary, case.boundary_map)

    def test_boundary_map_not_a_dict(self):
        case = Case(
            n_cand=2,
            n_search=1,
            n_held=1,
            search_outcomes=full_matrix(2, 1, "s", "pass"),
            held_outcomes=full_matrix(2, 1, "h", "pass"),
            boundary_map=[(1,)],
        )
        with self.assertRaises(ValueError):
            case.run_actual()
        with self.assertRaises(ValueError):
            oracle_resolve_boundaries(case.formats, case.boundary, case.boundary_map)

    def test_boundary_map_empty_tuple(self):
        case = Case(
            n_cand=2,
            n_search=1,
            n_held=1,
            search_outcomes=full_matrix(2, 1, "s", "pass"),
            held_outcomes=full_matrix(2, 1, "h", "pass"),
            boundary_map={"cand-0": ()},
        )
        with self.assertRaises(ValueError):
            case.run_actual()
        with self.assertRaises(ValueError):
            oracle_resolve_boundaries(case.formats, case.boundary, case.boundary_map)

    def test_duplicate_candidate_names_rejected(self):
        formats = (make_format(0), make_format(0))
        split = Split(search=(make_fixture("s", 0),), held_out=(make_fixture("h", 0),))
        with self.assertRaises(ValueError):
            run_experiment(ScriptedBackend(Plan()), formats, split, boundary=DEFAULT_BOUNDARY)

    def test_empty_held_out_split_rejected(self):
        """Documents the brief discrepancy: 0 held-out fixtures cannot be
        constructed -- Split requires both splits nonempty."""
        with self.assertRaises(ValueError):
            Split(search=(make_fixture("s", 0),), held_out=())


# ---------------------------------------------------------------------------
# Seeded fuzz sweep.
# ---------------------------------------------------------------------------


class RunExperimentFuzz(unittest.TestCase):
    FUZZ_SEEDS = 48

    def test_fuzz_oracle_agreement(self):
        for fuzz_seed in range(self.FUZZ_SEEDS):
            rng = random.Random(20261015 + fuzz_seed)  # noqa: S311 -- deterministic seeded fuzz, not crypto
            n_cand = rng.randint(2, 5)
            n_search = rng.randint(1, 6)
            n_held = rng.randint(1, 3)
            search = {}
            for i in range(n_cand):
                for j in range(n_search):
                    r = rng.random()
                    search[(i, ("s", j))] = (
                        "pass"
                        if r < 0.55
                        else "wrong_answer"
                        if r < 0.70
                        else "no_boundary"
                        if r < 0.85
                        else "both"
                    )
            held = {}
            for i in range(n_cand):
                for j in range(n_held):
                    r = rng.random()
                    held[(i, ("h", j))] = "pass" if r < 0.7 else rng.choice(OUTCOMES[1:])
            # Bias some seeds toward structured shapes the pure-random draw
            # rarely hits: forced unique winner / forced tie.
            shape = rng.random()
            if shape < 0.25:
                for j in range(n_search):
                    search[(0, ("s", j))] = "pass"
                for j in range(n_search):
                    search[(1, ("s", j))] = rng.choice(OUTCOMES[1:])
            elif shape < 0.40:
                for j in range(n_search):
                    search[(0, ("s", j))] = "pass"
                    search[(1, ("s", j))] = "pass"
            boundary_map = None
            if rng.random() < 0.25:
                boundary_map = {}
                for i in rng.sample(range(n_cand), rng.randint(1, n_cand)):
                    boundary_map[f"cand-{i}"] = (
                        (2000000 + 10 * i + 1, 2000000 + 10 * i + 2)
                        if rng.random() < 0.5
                        else (2000000 + 10 * i + 1,)
                    )
            cancel_after = None
            if rng.random() < 0.15:
                cancel_after = rng.randint(1, n_cand * n_search + n_held)
            case = Case(
                n_cand=n_cand,
                n_search=n_search,
                n_held=n_held,
                search_outcomes=search,
                held_outcomes=held,
                boundary_map=boundary_map,
                cancel_after=cancel_after,
                max_new_tokens=rng.choice((16, 24, 32)),
                seed=5000 + fuzz_seed,
                eos_id=rng.choice((None, 7)),
            )
            with self.subTest(
                fuzz_seed=fuzz_seed,
                n_cand=n_cand,
                n_search=n_search,
                n_held=n_held,
                boundary_map=bool(boundary_map),
                cancel_after=cancel_after,
            ):
                diffs, _ = check_case(case)
                self.assertEqual(
                    diffs,
                    [],
                    "oracle/implementation mismatch:\n"
                    + "\n".join(f"  {p}: actual={a!r} expected={e!r}" for p, a, e in diffs[:8]),
                )


# ---------------------------------------------------------------------------
# Sensitivity controls: mutations of a COPY of the implementation must be
# caught by the oracle comparison. Each runs in a subprocess so the mutated
# package is imported fresh, without touching the real source tree.
# ---------------------------------------------------------------------------


def build_scenario(name):
    """Deterministic scenario builders shared with the mutation subprocess."""
    if name == "majority":
        # cand-0 passes all; cand-1 misses one. Unanimity selects cand-0;
        # a majority rule would see two "winners" -> ambiguous.
        n_cand, n_search, n_held = 2, 2, 1
        search = full_matrix(n_cand, n_search, "s", "pass", {(1, ("s", 1)): "wrong_answer"})
        held = full_matrix(n_cand, n_held, "h", "pass")
    elif name == "skip_held_out":
        n_cand, n_search, n_held = 2, 2, 2
        search = full_matrix(n_cand, n_search, "s", "pass", {(1, ("s", 0)): "both"})
        held = full_matrix(n_cand, n_held, "h", "pass")
    elif name == "decision_string":
        n_cand, n_search, n_held = 3, 2, 1
        search = full_matrix(n_cand, n_search, "s", "pass", {(2, ("s", 0)): "no_boundary"})
        held = full_matrix(n_cand, n_held, "h", "pass")
    else:
        raise ValueError(f"unknown scenario {name!r}")
    return Case(
        n_cand=n_cand,
        n_search=n_search,
        n_held=n_held,
        search_outcomes=search,
        held_outcomes=held,
        seed=99,
    )


def run_scenario(name):
    """Build a scenario and diff the (possibly mutated) run_experiment against
    the oracle. Returns the mismatch list; non-empty means caught."""
    diffs, _ = check_case(build_scenario(name))
    return diffs


MUTATIONS = {
    # Unanimity -> near-unanimity: a candidate missing one fixture counts.
    "majority_rule": (
        'if summary["exact_and_boundary_passes"] == report.planned_count:',
        'if summary["exact_and_boundary_passes"] >= report.planned_count - 1:',
        "majority",
    ),
    # Held-out stage results ignored: decision forced without evidence.
    "skip_held_out": (
        """    output["held_out"] = {
        "summary": _summary(held),
        "observations": [asdict(o) for o in held.observations],
    }
    if not held.aggregate_eligible:
        output["decision"] = held.status
    else:
        passed = all(o.exact_match and o.boundary_found for o in held.observations)
        output["decision"] = "compatible_on_fixture_set" if passed else "held_out_failed"
""",
        """    output["decision"] = "compatible_on_fixture_set"
""",
        "skip_held_out",
    ),
    # Misreported decision string on ties.
    "decision_string": (
        '"ambiguous" if winners else "no_candidate"',
        '"tie" if winners else "no_candidate"',
        "decision_string",
    ),
}

DRIVER = """\
import json
import sys

sys.path.insert(0, {mutdir!r})
sys.path.insert(0, {testsdir!r})

import test_cycle21_run_experiment_verify as t

print(json.dumps(t.run_scenario({scenario!r})))
"""


def run_mutated(mutation_name=None, scenario="majority"):
    """Copy the package to a temp dir, optionally apply one mutation, run the
    scenario driver in a subprocess, return the mismatch list."""
    with TemporaryDirectory(prefix="mpl_mut_") as tmp:
        mutdir = Path(tmp) / "mutpkg"
        shutil.copytree(REPO_ROOT / "model_protocol_lab", mutdir / "model_protocol_lab")
        if mutation_name is not None:
            old, new, _ = MUTATIONS[mutation_name]
            target = mutdir / "model_protocol_lab" / "experiments.py"
            text = target.read_text()
            if text.count(old) != 1:
                raise AssertionError(
                    f"mutation anchor not unique/found for {mutation_name}: count={text.count(old)}"
                )
            target.write_text(text.replace(old, new, 1))
            scenario = MUTATIONS[mutation_name][2]
        driver = Path(tmp) / "driver.py"
        driver.write_text(
            DRIVER.format(mutdir=str(mutdir), testsdir=str(TESTS_DIR), scenario=scenario)
        )
        proc = subprocess.run(
            [sys.executable, str(driver)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(REPO_ROOT),
        )
        if proc.returncode != 0:
            raise AssertionError(
                f"mutation driver failed for {mutation_name}:\n{proc.stderr[-2000:]}"
            )
        return json.loads(proc.stdout)


class RunExperimentMutationSensitivity(unittest.TestCase):
    def test_control_unmutated_copy_agrees(self):
        """The harness itself is sound: an unmutated copy shows zero diffs."""
        for scenario in ("majority", "skip_held_out", "decision_string"):
            with self.subTest(scenario=scenario):
                self.assertEqual(
                    run_mutated(None, scenario), [], f"control copy disagrees on {scenario}"
                )

    def test_mutation_majority_rule_caught(self):
        diffs = run_mutated("majority_rule")
        self.assertTrue(diffs, "majority-rule mutation was NOT caught by the oracle")
        paths = [p for p, _, _ in diffs]
        self.assertTrue(
            any("decision" in p for p in paths), f"expected a decision mismatch, got: {paths[:5]}"
        )

    def test_mutation_skip_held_out_caught(self):
        diffs = run_mutated("skip_held_out")
        self.assertTrue(diffs, "skip-held-out mutation was NOT caught by the oracle")
        paths = [p for p, _, _ in diffs]
        self.assertTrue(
            any("held_out" in p for p in paths), f"expected a held_out mismatch, got: {paths[:5]}"
        )

    def test_mutation_decision_string_caught(self):
        diffs = run_mutated("decision_string")
        self.assertTrue(diffs, "decision-string mutation was NOT caught by the oracle")
        self.assertIn(
            ["result['decision']", "tie", "ambiguous"],  # JSON round-trip: lists, not tuples
            diffs,
            f"expected the tie/ambiguous mismatch, got: {diffs[:5]}",
        )


if __name__ == "__main__":
    unittest.main()
