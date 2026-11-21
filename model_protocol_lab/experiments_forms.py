"""experiments forms family: split from experiments.py (W7-2).

Facades keep the public surface: import from model_protocol_lab.experiments, not from this module.
"""

import json
import math
from threading import Event
from typing import Any, cast

from .core import Backend, Fixture, Format, Message, _valid_token_ids, evaluate
from .experiments_core import (
    _answer_digest,
    _backend_fingerprint,
    _fixture_with_content,
    _resolve_boundaries,
    native_search_outcomes,
)
from .position import CONDITIONS as POSITION_CONDITIONS
from .position import primacy_index
from .scoring import STATUS_OK, TargetScore, score_target_logprobs


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
        """Return the rates."""
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
        """Return the cancelled."""
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
        """Return the cancelled."""
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


class _RotationCancelled(Exception):
    """Internal: the run's cancel event fired during the
    rotation stage; the stage unwinds to its fail-closed
    payload rather than mislabelling the interruption.
    """


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
    payload rather than mislabelling the interruption.
    """


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
        """Return the cancelled."""
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
        """Return the cancelled."""
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
        """Return the cancelled."""
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
        """Return the cancelled."""
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
    if isinstance(total, (int, float)) and not isinstance(total, bool) and not math.isnan(total):
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
        """Return the unavailable."""
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
        """Return the unavailable."""
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
        """Return the eval."""
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
