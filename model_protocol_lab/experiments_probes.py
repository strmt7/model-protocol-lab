"""experiments probes family: split from experiments.py (W7-2).

Facades keep the public surface: import from model_protocol_lab.experiments, not from this module.
"""

import json
from hashlib import sha256
from threading import Event
from typing import Any

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
from .core import Backend, Fixture, Format, Message, _valid_token_ids, evaluate, first_boundary
from .deliforge import (
    DETAIL_CELL_FAILED,
    MARKERS,
    PAYLOADS,
    validate_marker_index,
    validate_payload,
)
from .experiments_core import (
    _answer_digest,
    _backend_fingerprint,
    _condition_all_pass,
    _fixture_with_content,
    _forged_messages_for,
    _resolve_boundaries,
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
)
from .hierarchy import POSITIONS, family_of, system_treatment
from .prefill import (
    PrefillBackend,
    build_prefill_fixtures,
    honors_prefill,
    opens_think_channel,
    validate_prefill_specs,
)
from .structured import candidate_adherence, classify_response, validate_schema_def
from .toolcall import candidate_adherence as toolcall_candidate_adherence
from .toolcall import classify_tool_call


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
        """Return the rates."""
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
        """Return the rates."""
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
        """Return the cancelled."""
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
        """Return the size span."""
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
        """Return the cancelled."""
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
        """Return the cancelled."""
        return cancel is not None and cancel.is_set()

    def _order(start: int, target: int) -> list[tuple[int, int, int]]:
        """Return the order."""
        cache_key = (start, target)
        order = order_cache.get(cache_key)
        if order is None:
            order = span_candidates(stream[start:], target=target, chars_per_token=chars_per_token)
            order_cache[cache_key] = order
        return order

    def _render_ids(fmt: Any, content: str) -> tuple[int, ...]:
        """Render ids."""
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
        _z_ids: tuple[int, ...],
        q_ids: tuple[int, ...],
        anchor: int,
        encodes: int,
    ) -> tuple[str, tuple[int, ...], int, int] | None:
        """Phase 1: (prefix_text, comparator_ids, consumed,
        encodes) for the first prefix whose D equals ``target``
        -- None when the budget runs out first. Depth 0 is the
        empty prefix with the question-only render as its
        comparator, no search.
        """
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
        offsets re-verified on the final ids.
        """
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
        delta over the question-only render equals the dose.
        """
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
    payload rather than mislabelling the interruption.
    """


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


class _InstructionDistanceCancelled(Exception):
    """Internal: the run's cancel event fired during the
    instruction-distance stage; the stage unwinds to its
    fail-closed payload rather than mislabelling the
    interruption.
    """


def _instruction_distance_cell(
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
    """One instruction-distance generation: (detail, cell).

    Exactly one of the pair is set. The cell carries the
    span's digest-bound primitives (amendment C): the answer
    digest, the first-ASCII-digit-run digest ("" when the span
    has no run), and the non-digit-projection digest, from
    which the decomposition class is recomputed identically
    by the stage's aggregator and by the diff validator;
    ``answer_len`` / ``digit_count`` ride along as descriptive
    counts only. The cancel event is checked before and after
    ``evaluate`` (the multiturn rule: a cancel observed during
    a generation is never mislabelled an evaluation failure).
    """
    from . import instructiondistance as _id

    if len(ids) + max_new_tokens > max_context:
        return "arm_context_exceeded", None
    if cancel is not None and cancel.is_set():
        raise _InstructionDistanceCancelled
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
        raise _InstructionDistanceCancelled
    if report.aggregate_eligible and report.observations:
        obs = report.observations[0]
        span = obs.answer if type(obs.answer) is str else ""
        return None, {
            "verdict": bool(obs.exact_match and obs.boundary_found),
            "boundary_found": bool(obs.boundary_found),
            "boundary_text_match": obs.boundary_text_match,
            "input_tokens": len(obs.input_ids),
            "answer_digest": _answer_digest(span),
            "first_digit_run_digest": _id.first_digit_run_digest(span),
            "nondigit_projection_digest": _id.nondigit_projection_digest(span),
            "answer_len": len(span),
            "digit_count": _id.digit_count(span),
        }
    return "arm_evaluation_failed", None


def instruction_distance_stage(
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
    """Instruction-distance stage (cycle 218): generated
    fact+question cases governed by one frozen format
    directive, rendered in six arms per candidate -- adjacent,
    separated at doses 256/1024 (filler between the directive
    and the block), displaced at the same doses (filler after
    the block), and a descriptive order control.

    The cases are generated from the run seed inside the
    stage; there is no dependence on the user's fixtures or
    the primary run. Each sized arm's filler span is searched
    by the frozen single-phase sizing rule in
    instructiondistance.py: probes render the full arm content
    from the stream head in closeness order, one encode per
    probe inside the 512-encode per-cell budget, accepting the
    first construction whose rendered delta over the case's
    adjacent render equals the dose exactly on its final ids.
    A sized arm whose exact final length cannot fit the
    context is reported ``arm_context_exceeded`` before any
    sizing encode is spent (the needle discipline), and its
    record carries no arm token count -- an unconstructed arm
    has no render to measure. Constructed spans are screened
    per cell (``arm_screened``); the frozen texts are audited
    for special-token literals and boundary-text containment
    before any generation, failing the whole stage closed
    with ``instruction_distance_screen_failed``. Design:
    docs/INSTRUCTIONDISTANCE_DESIGN.md.
    """
    from . import instructiondistance as _id
    from .distraction import NEUTRAL_CORPUS, span_candidates, span_text
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
    protocol: dict[str, Any] = dict(_id.protocol_object(seed, _backend_fingerprint(backend)))
    protocol["fingerprint"] = _id.protocol_fingerprint(seed, _backend_fingerprint(backend))
    cases = _id.generate_cases(seed)
    audit: dict[str, Any] = {
        "n_audit_excluded": 0,
        "by_reason": {},
        # nosec B105 -- integer audit counter field, not a credential
        "special_token_findings": 0,
    }
    if len(formats) * len(cases) > _id.MAX_GROUND_RECORDS:
        return {
            "available": False,
            "detail": "instruction_distance_record_budget_exceeded",
            "protocol": protocol,
            "audit": audit,
            "candidates": {},
        }
    generated_text = (
        _id.CONSTRAINT
        + _id.FACT_TEMPLATE.replace("{topic}", "").replace("{value}", "")
        + _id.QUESTION_TEMPLATE.replace("{topic}", "")
        + "".join(_id.TOPICS)
    )
    corpus_text = "".join(NEUTRAL_CORPUS)
    findings = audit_special_token_literals(generated_text + corpus_text, formats)
    if findings:
        # A special-token literal in the frozen texts would
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
            "detail": "instruction_distance_screen_failed",
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
        if boundary_text[fmt.name] and boundary_text[fmt.name] in generated_text:
            return {
                "available": False,
                "detail": "instruction_distance_screen_failed",
                "protocol": protocol,
                "audit": audit,
                "candidates": {},
            }
    stream = _id.filler_stream()
    corpus_tokens = max(1, len(backend.encode(corpus_text)))
    chars_per_token = len(corpus_text) / corpus_tokens
    order_cache: dict[int, list[tuple[int, int, int]]] = {}

    def _cancelled() -> bool:
        """Return the cancelled."""
        return cancel is not None and cancel.is_set()

    def _order(dose: int) -> list[tuple[int, int, int]]:
        """Return the order."""
        order = order_cache.get(dose)
        if order is None:
            order = span_candidates(stream, target=dose, chars_per_token=chars_per_token)
            order_cache[dose] = order
        return order

    def _render_ids(fmt: Any, content: str) -> tuple[int, ...]:
        """Render ids."""
        fixture = Fixture(
            id="instructiondistance-probe",
            messages=(Message("user", content),),
            expected="",
            group=_id.GROUP,
        )
        return tuple(backend.encode(fmt.render(fixture.messages)))

    def _size_span(
        fmt: Any, case: dict[str, Any], arm: str, dose: int, adj_ids: tuple[int, ...]
    ) -> tuple[str, tuple[int, ...]] | None:
        """(span, arm_ids) for the first construction whose
        rendered delta over the adjacent render equals ``dose``
        exactly -- None when the encode budget runs out first
        (the cell is then arm_unmatchable, never approximated).
        """
        for probe_index, (whole, suffix, _chars) in enumerate(_order(dose)):
            if probe_index >= _id.SIZING_ENCODE_BUDGET:
                return None
            span = span_text(stream, whole, suffix)
            arm_ids = _render_ids(fmt, _id.arm_content(case, arm, span))
            if len(arm_ids) - len(adj_ids) == dose:
                return span, arm_ids
        return None

    def _run_cell(
        fmt: Any,
        boundary_ids: tuple[int, ...],
        record: dict[str, Any],
        case_id: str,
        value: str,
        arm: str,
        content: str,
        ids: tuple[int, ...],
    ) -> None:
        """Generate one constructed cell into its record."""
        if _cancelled():
            raise _InstructionDistanceCancelled
        fixture = Fixture(
            id=f"{case_id}-{arm}",
            messages=(Message("user", content),),
            expected=value,
            group=_id.GROUP,
        )
        detail, cell = _instruction_distance_cell(
            backend,
            fmt,
            fixture,
            ids,
            boundary_ids=boundary_ids,
            max_new_tokens=max_new_tokens,
            max_context=max_context,
            seed=seed,
            cancel=cancel,
        )
        if detail is not None:
            record["cell_details"][arm] = detail
        elif cell is not None:
            record["cells"][arm] = cell

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
                value = str(case["value"])
                record: dict[str, Any] = {
                    "fixture": case_id,
                    "topic": str(case["topic"]),
                    "expected_digest": _id.digest_text(str(case["expected"])),
                    "arm_tokens": dict.fromkeys(_id.ARM_KEYS),
                    "id_digests": dict.fromkeys(_id.ARM_KEYS),
                    "cells": dict.fromkeys(_id.ARM_KEYS),
                    "cell_details": dict.fromkeys(_id.ARM_KEYS, ""),
                }
                records.append(record)
                contents: dict[str, str] = {}
                arm_ids: dict[str, tuple[int, ...]] = {}
                for arm in ("adj", "adj_after"):
                    contents[arm] = _id.arm_content(case, arm)
                    arm_ids[arm] = _render_ids(fmt, contents[arm])
                    record["arm_tokens"][arm] = len(arm_ids[arm])
                    record["id_digests"][arm] = _id.digest_ids(arm_ids[arm])
                _run_cell(
                    fmt,
                    boundaries[name],
                    record,
                    case_id,
                    value,
                    "adj",
                    contents["adj"],
                    arm_ids["adj"],
                )
                adj_ids = arm_ids["adj"]
                for arm in _id.SIZED_ARMS:
                    if _cancelled():
                        raise _InstructionDistanceCancelled
                    dose = _id.arm_dose(arm)
                    if len(adj_ids) + dose + max_new_tokens > max_context:
                        # The delta identity fixes this cell's
                        # final length before sizing; a cell
                        # that cannot fit is reported without
                        # spending sizing encodes on it, and no
                        # construction exists to record.
                        record["cell_details"][arm] = "arm_context_exceeded"
                        continue
                    sized = _size_span(fmt, case, arm, dose, adj_ids)
                    if sized is None:
                        record["cell_details"][arm] = "arm_unmatchable"
                        continue
                    span, sized_ids = sized
                    contents[arm] = _id.arm_content(case, arm, span)
                    record["arm_tokens"][arm] = len(sized_ids)
                    record["id_digests"][arm] = _id.digest_ids(sized_ids)
                    if _id.filler_screened(span, value, boundary_text[name]):
                        record["cell_details"][arm] = "arm_screened"
                        continue
                    _run_cell(
                        fmt, boundaries[name], record, case_id, value, arm, contents[arm], sized_ids
                    )
                _run_cell(
                    fmt,
                    boundaries[name],
                    record,
                    case_id,
                    value,
                    "adj_after",
                    contents["adj_after"],
                    arm_ids["adj_after"],
                )
        except _InstructionDistanceCancelled:
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


def instruction_distance_analysis(
    stage: dict[str, Any],
    candidate_names: tuple[str, ...],
) -> dict[str, Any]:
    """Aggregate an instruction-distance stage into the stored
    payload (cycle 218).

    Per-candidate readouts come from
    instructiondistance.aggregate_candidate over the
    candidate's ground records: per-arm compliance and
    byte-exact rates with the decomposition, the per-dose
    (D, S, T) decisions, and the id-divergence report. There
    is no selection and no winner readout: per-candidate
    per-dose decisions are the only headline. Design:
    docs/INSTRUCTIONDISTANCE_DESIGN.md.
    """
    from . import instructiondistance as _id

    if not stage.get("available"):
        return {
            "available": False,
            "detail": str(stage.get("detail", "instruction_distance_unavailable")),
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
        aggregates = _id.aggregate_candidate(entry["records"])
        aggregates.update({"status": "available", "detail": "", "records": entry["records"]})
        candidates_out[name] = aggregates
    return {
        "available": True,
        "detail": "",
        "protocol": stage["protocol"],
        "audit": stage.get("audit"),
        "candidates": candidates_out,
        "n_candidates": len(candidate_names),
        "n_cases": len(_id.generate_cases(int(stage["protocol"]["seed"]))),
    }


class _HistoryFrameCancelled(Exception):
    """Internal: the run's cancel event fired during the
    history-frame stage; the stage unwinds to its fail-closed
    payload rather than mislabelling the interruption.
    """


def _history_frame_cell(
    backend: Any,
    carrier: Any,
    fixture: Any,
    ids: tuple[int, ...],
    *,
    boundary_ids: tuple[int, ...],
    max_new_tokens: int,
    max_context: int,
    seed: int,
    cancel: Event | None,
) -> tuple[str | None, dict[str, Any] | None]:
    """One history-frame generation: (detail, cell).

    Exactly one of the pair is set. ``ids`` are the arm hybrid
    text's ids, measured by the stage at construction for the
    record's arm token counts and id digests (amendment A4:
    the ids-identical gate is derived from those digests, so
    a backend whose encode disagreed with its evaluation
    surfaces downstream, never as a silently wrong gate). The
    cancel event is checked before and after ``evaluate``
    (the multiturn rule: a cancel observed during a generation
    is never mislabelled an evaluation failure).
    """
    if len(ids) + max_new_tokens > max_context:
        return "arm_context_exceeded", None
    if cancel is not None and cancel.is_set():
        raise _HistoryFrameCancelled
    report = evaluate(
        backend,
        carrier,
        (fixture,),
        boundary=boundary_ids,
        max_new_tokens=max_new_tokens,
        seed=seed,
        cancel=cancel,
    )
    if cancel is not None and cancel.is_set():
        raise _HistoryFrameCancelled
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


def history_frame_stage(
    backend: Any,
    formats: tuple[Any, ...],
    fixtures: tuple[Any, ...],
    *,
    boundary: tuple[int, ...],
    boundary_map: dict[str, tuple[int, ...]] | None = None,
    max_new_tokens: int = 64,
    max_context: int,
    seed: int = 0,
    cancel: Event | None = None,
) -> dict[str, Any]:
    """History-frame stage (cycle 219): the run's search
    fixtures preceded by the frozen synthetic history, with
    only the prepended segment reframed.

    Every search fixture is prepended with synthhistory's
    ``history_6`` turns (construction pins, amendment A5: the
    built fixtures' messages re-wrapped under each fixture's
    original id, original expected answer, and the module's
    frozen group). Per candidate, four arms render the
    prepended messages: ``own`` (the history segment in the
    candidate's own framing), ``foreign_chatml`` and
    ``foreign_llama3`` (the segment in the frozen donor
    framings), and the descriptive ``verbatim_chatml`` (the
    chatml segment under chatml's own leading bos). The tail,
    the prompt-initial bos, and the open suffix are always the
    candidate's own. Evaluation is one carrier ``evaluate``
    call per (candidate, arm, fixture) holding the hybrid
    text as a single user message, with the candidate's
    resolved boundary ids and the fixture's original id in
    every arm (request-seed identity across arms).

    All screens run before any generation: S1 (amendment
    A9's per-arm slice-swap identity against the candidate's
    native render, with the own hybrid byte-identical to it),
    S2 (special-token-literal audit over the frozen history
    texts and the donor and candidate framing literals; user
    fixture contents are never screened -- they are
    byte-identical across arms and cannot manufacture a frame
    contrast), and S3 (the record budget). A screen failure
    fails the whole stage closed. Design:
    docs/HISTORYFRAME_DESIGN.md.
    """
    from . import historyframe as _hf
    from .fixtures import audit_special_token_literals
    from .synthhistory import SYNTHETIC_TURNS

    if type(formats) is not tuple or not formats:
        raise ValueError("formats must be a nonempty tuple")
    names = [fmt.name for fmt in formats]
    if len(set(names)) != len(names):
        # Candidate records are keyed by name; two formats
        # sharing a name would silently collapse into one
        # candidate's records (red-team F7, cycle 193). The
        # precondition is stated at this stage's own boundary.
        raise ValueError("formats must have unique names")
    if type(fixtures) is not tuple or not fixtures:
        raise ValueError("fixtures must be a nonempty tuple")
    if any(type(case) is not Fixture for case in fixtures):
        raise ValueError("fixtures must contain Fixture values")
    fixture_ids = [case.id for case in fixtures]
    if len(set(fixture_ids)) != len(fixture_ids):
        raise ValueError("fixtures must have unique ids")
    protocol: dict[str, Any] = dict(
        _hf.protocol_object(seed, _backend_fingerprint(backend), fixture_ids)
    )
    protocol["fingerprint"] = _hf.protocol_fingerprint(
        seed, _backend_fingerprint(backend), fixture_ids
    )
    audit: dict[str, Any] = {
        "n_audit_excluded": 0,
        "by_reason": {},
        # nosec B105 -- integer audit counter field, not a credential
        "special_token_findings": 0,
    }
    if len(formats) * len(fixtures) > _hf.MAX_GROUND_RECORDS:
        return {
            "available": False,
            "detail": "historyframe_record_budget_exceeded",
            "protocol": protocol,
            "audit": audit,
            "candidates": {},
        }
    cases = _hf.stage_fixtures(fixtures)
    donors = {arm: _hf.donor_format(arm) for arm in _hf.CONTRASTS}
    # S2: the frozen history texts, audited against the
    # special-token inventory derived from the framing fields
    # of the donors and of every candidate family (the audit
    # derives its inventory from the formats it is given, so
    # the donor formats ride along -- a history text carrying
    # a donor's or a candidate's special token would forge
    # turn structure in the reframed segment).
    audit_texts: list[str] = [turn.content for turn in SYNTHETIC_TURNS]
    findings = audit_special_token_literals(tuple(audit_texts), (*formats, *donors.values()))
    if findings:
        audit = {
            "n_audit_excluded": 0,
            "by_reason": {"special_token_literal": len(findings)},
            "special_token_findings": len(findings),
        }
        return {
            "available": False,
            "detail": "historyframe_screen_failed",
            "protocol": protocol,
            "audit": audit,
            "candidates": {},
        }
    boundaries = _resolve_boundaries(formats, boundary, boundary_map)
    stratum = {arm: _hf.encoding_stratum(backend, donors[arm]) for arm in _hf.CONTRASTS}
    # S1 (strengthened, amendment A9): every arm's construction
    # for every fixture is built and identity-checked before
    # any generation is spent, for every candidate: the own
    # hybrid must equal the candidate's native render byte for
    # byte, each foreign hybrid must equal that render with
    # only the history-segment slice exchanged, and the
    # verbatim text must satisfy its own slice identity under
    # its defining leading-bos substitution.
    built: dict[str, list[dict[str, Any]]] = {}
    for fmt in formats:
        per_case: list[dict[str, Any]] = []
        for case in cases:
            native = fmt.render(case.messages)
            seg_own = _hf.history_segment(fmt, case.messages)
            texts: dict[str, str] = {"own": _hf.hybrid_text(fmt, fmt, case.messages)}
            ok = texts["own"] == native
            if ok:
                for arm in _hf.CONTRASTS:
                    texts[arm] = _hf.hybrid_text(donors[arm], fmt, case.messages)
                    ok = ok and _hf.slice_swap_identity(
                        native,
                        texts[arm],
                        seg_own,
                        _hf.history_segment(donors[arm], case.messages),
                        fmt.bos,
                    )
                texts[_hf.VERBATIM_ARM] = _hf.verbatim_text(
                    donors[_hf.VERBATIM_TWIN], fmt, case.messages
                )
                ok = ok and _hf.verbatim_slice_identity(
                    native,
                    texts[_hf.VERBATIM_ARM],
                    seg_own,
                    _hf.history_segment(donors[_hf.VERBATIM_TWIN], case.messages),
                    fmt.bos,
                    donors[_hf.VERBATIM_TWIN].bos,
                )
            if not ok:
                return {
                    "available": False,
                    "detail": "historyframe_screen_failed",
                    "protocol": protocol,
                    "audit": audit,
                    "candidates": {},
                }
            per_case.append({"case": case, "texts": texts})
        built[fmt.name] = per_case

    def _cancelled() -> bool:
        """Return the cancelled."""
        return cancel is not None and cancel.is_set()

    candidates: dict[str, Any] = {}
    for fmt in formats:
        name = fmt.name
        carrier = _hf.carrier_format(fmt)
        records: list[dict[str, Any]] = []
        try:
            for entry in built[name]:
                if _cancelled():
                    return {
                        "available": False,
                        "detail": "cancelled",
                        "protocol": protocol,
                        "audit": audit,
                        "candidates": {},
                    }
                case = entry["case"]
                texts = entry["texts"]
                arm_ids = {arm: tuple(backend.encode(texts[arm])) for arm in _hf.ARM_KEYS}
                record: dict[str, Any] = {
                    "fixture": case.id,
                    "expected_digest": _hf.digest_text(case.expected),
                    "own_bos_digest": _hf.digest_text(fmt.bos),
                    "donor_bos_digest": _hf.digest_text(donors[_hf.VERBATIM_TWIN].bos),
                    "segment_digests": {
                        "own": _hf.digest_text(_hf.history_segment(fmt, case.messages)),
                        "foreign_chatml": _hf.digest_text(
                            _hf.history_segment(donors["foreign_chatml"], case.messages)
                        ),
                        "foreign_llama3": _hf.digest_text(
                            _hf.history_segment(donors["foreign_llama3"], case.messages)
                        ),
                        "verbatim_chatml": _hf.digest_text(
                            _hf.history_segment(donors["foreign_chatml"], case.messages)
                        ),
                    },
                    "hybrid_digests": {arm: _hf.digest_text(texts[arm]) for arm in _hf.ARM_KEYS},
                    "arm_tokens": {arm: len(arm_ids[arm]) for arm in _hf.ARM_KEYS},
                    "id_digests": {arm: _hf.digest_ids(arm_ids[arm]) for arm in _hf.ARM_KEYS},
                    "cells": dict.fromkeys(_hf.ARM_KEYS),
                    "cell_details": dict.fromkeys(_hf.ARM_KEYS, ""),
                }
                records.append(record)
                for arm in _hf.ARM_KEYS:
                    if _cancelled():
                        raise _HistoryFrameCancelled
                    cell_fixture = Fixture(
                        id=case.id,
                        messages=(Message("user", texts[arm]),),
                        expected=case.expected,
                        group=_hf.GROUP,
                    )
                    detail, cell = _history_frame_cell(
                        backend,
                        carrier,
                        cell_fixture,
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
        except _HistoryFrameCancelled:
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
        "stratum": stratum,
        "candidates": candidates,
    }


def history_frame_analysis(
    stage: dict[str, Any],
    candidate_names: tuple[str, ...],
    primary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate a history-frame stage into the stored payload
    (cycle 219).

    Per-candidate readouts come from
    historyframe.aggregate_candidate over the candidate's
    ground records: per-arm rates and attrition, the two
    headline contrasts with their classifications and
    decompositions, and the descriptive verbatim readout.
    Selection is recomputed per headline arm on the common
    completed set (canonical-form discipline) and the winner
    headline is taken per contrast; flips against the primary
    run are computed with synthhistory's verdict mapping and
    are labelled as pooling history presence with frame. The
    family-wise block sums the exact at-n random-labeller
    fire rates over the emitted classifications (amendment
    A2 -- computed, never a frozen constant). Design:
    docs/HISTORYFRAME_DESIGN.md.
    """
    from . import historyframe as _hf

    if not stage.get("available"):
        return {
            "available": False,
            "detail": str(stage.get("detail", "historyframe_unavailable")),
            "protocol": stage.get("protocol"),
            "audit": stage.get("audit"),
            "stratum": stage.get("stratum"),
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
        aggregates = _hf.aggregate_candidate(entry["records"])
        aggregates.update({"status": "available", "detail": "", "records": entry["records"]})
        candidates_out[name] = aggregates
        records_by_candidate[name] = entry["records"]
    conditions = {arm: _hf.arm_selection(records_by_candidate, arm) for arm in _hf.HEADLINE_ARMS}
    selection = {
        contrast: _hf.winner_headline(conditions["own"], conditions[contrast])
        for contrast in _hf.CONTRASTS
    }
    rates: list[float] = []
    for entry in candidates_out.values():
        contrasts = entry.get("contrasts")
        if not isinstance(contrasts, dict):
            continue
        for contrast in _hf.CONTRASTS:
            block = contrasts.get(contrast)
            if isinstance(block, dict) and block.get("random_fire_rate") is not None:
                rates.append(float(block["random_fire_rate"]))
    protocol = stage["protocol"]
    return {
        "available": True,
        "detail": "",
        "protocol": protocol,
        "audit": stage.get("audit"),
        "stratum": stage.get("stratum"),
        "candidates": candidates_out,
        "conditions": conditions,
        "selection": selection,
        "primary": primary,
        "flips": _hf.primary_flips(conditions, primary),
        "familywise": _hf.familywise_expectation(rates),
        "n_candidates": len(candidate_names),
        "n_fixtures": len(protocol["fixture_ids"]),
    }


class _PruneSweepCancelled(Exception):
    """Internal: the run's cancel event fired during the
    prune-sweep stage; the stage unwinds to its fail-closed
    payload rather than mislabelling the interruption.
    """


def _prune_sweep_cell(
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
    """One prune-sweep generation: (detail, cell).

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
        raise _PruneSweepCancelled
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
        raise _PruneSweepCancelled
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


def prune_sweep_stage(
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
    """Prune-sweep stage (cycle 220): generated 24-case
    fact-recall panel whose passage context sentences lose
    seeded interior words at exact doses 0 / 18 / 36 / 72,
    plus the dose-matched tail72 displaced control and the
    closed_book screen arm, per candidate.

    The cases are generated from the run seed inside the
    stage; there is no dependence on the user's fixtures or
    the primary run. Surgery is at the fixture level: every
    arm of a case is a Fixture with the same id, group, and
    expected, so request seeds are identical across arms by
    construction (``request_seed_for`` is id-derived) and
    ``core.evaluate`` is used unchanged. The pre-generation
    screens (the per-case invariant battery, the
    special-token-literal audit over every arm's full
    message content for every family, and boundary-text
    containment) fail the whole stage closed with
    ``prune_screen_failed``; their counts travel in the
    payload's ``audit`` block, never silently dropped.
    Design: docs/PRUNESWEEP_DESIGN.md.
    """
    from . import prunesweep as _ps
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
    protocol: dict[str, Any] = dict(_ps.protocol_object(seed, _backend_fingerprint(backend)))
    protocol["fingerprint"] = _ps.protocol_fingerprint(seed, _backend_fingerprint(backend))
    cases = _ps.generate_cases(seed)
    audit: dict[str, Any] = {
        "n_audit_excluded": 0,
        "by_reason": {},
        # nosec B105 -- integer audit counter field, not a credential
        "special_token_findings": 0,
    }
    if len(formats) * len(cases) > _ps.MAX_GROUND_RECORDS:
        return {
            "available": False,
            "detail": "prune_record_budget_exceeded",
            "protocol": protocol,
            "audit": audit,
            "candidates": {},
        }
    invalid = sum(1 for case in cases if not _ps.validate_case(case, seed))
    if invalid:
        audit = {
            "n_audit_excluded": 0,
            "by_reason": {"case_invariant": invalid},
            # nosec B105 -- integer audit counter field, not a credential
            "special_token_findings": 0,
        }
        return {
            "available": False,
            "detail": "prune_screen_failed",
            "protocol": protocol,
            "audit": audit,
            "candidates": {},
        }
    contents = tuple(_ps.arm_content(case, arm, seed) for case in cases for arm in _ps.ARM_KEYS)
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
            "detail": "prune_screen_failed",
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
                "detail": "prune_screen_failed",
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
                    raise _PruneSweepCancelled
                case_id = str(case["fixture_id"])
                value = str(case["value"])
                record: dict[str, Any] = {
                    "fixture": case_id,
                    "topic": str(case["topic"]),
                    "expected_digest": _ps.digest_text(str(case["expected"])),
                    "arm_tokens": dict.fromkeys(_ps.ARM_KEYS),
                    "id_digests": dict.fromkeys(_ps.ARM_KEYS),
                    "ids_identical": False,
                    "content_digests": dict.fromkeys(_ps.ARM_KEYS),
                    "dropped_shares": _ps.dropped_set_shares(case, seed),
                    "cells": dict.fromkeys(_ps.ARM_KEYS),
                    "cell_details": dict.fromkeys(_ps.ARM_KEYS, ""),
                }
                for arm in _ps.ARM_KEYS:
                    content = _ps.arm_content(case, arm, seed)
                    fixture = Fixture(
                        id=case_id,
                        messages=(Message("user", content),),
                        expected=value,
                        group=_ps.GROUP,
                    )
                    ids = tuple(backend.encode(fmt.render(fixture.messages)))
                    record["arm_tokens"][arm] = len(ids)
                    record["id_digests"][arm] = _ps.digest_ids(ids)
                    record["content_digests"][arm] = _ps.digest_text(content)
                    detail, cell = _prune_sweep_cell(
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
                record["ids_identical"] = (
                    record["id_digests"]["dose0"] == record["id_digests"]["dose72"]
                )
                records.append(record)
        except _PruneSweepCancelled:
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


def prune_sweep_analysis(
    stage: dict[str, Any],
    candidate_names: tuple[str, ...],
) -> dict[str, Any]:
    """Aggregate a prune-sweep stage into the stored payload
    (cycle 220).

    Per-candidate readouts come from
    prunesweep.aggregate_candidate over the candidate's
    ground records: per-arm rates, attrition, and digest
    decomposition; the headline decision over the completed,
    unscreened, ids-diverged population; the descriptive
    shape readout; and the tail72 displaced-control
    contrast. There is no selection and no winner readout:
    the panel is generated, so the numform settlement
    forbids selection over it. Design:
    docs/PRUNESWEEP_DESIGN.md.
    """
    from . import prunesweep as _ps

    if not stage.get("available"):
        return {
            "available": False,
            "detail": str(stage.get("detail", "prune_unavailable")),
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
        aggregates = _ps.aggregate_candidate(entry["records"])
        aggregates.update({"status": "available", "detail": "", "records": entry["records"]})
        candidates_out[name] = aggregates
    return {
        "available": True,
        "detail": "",
        "protocol": stage["protocol"],
        "audit": stage.get("audit"),
        "candidates": candidates_out,
        "n_candidates": len(candidate_names),
        "n_cases": len(_ps.generate_cases(int(stage["protocol"]["seed"]))),
    }
