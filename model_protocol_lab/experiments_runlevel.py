"""experiments runlevel family: split from experiments.py (W7-2).

Facades keep the public surface: import from model_protocol_lab.experiments, not from this module.
"""

from bisect import bisect_right
from collections.abc import Callable
from contextlib import suppress
from hashlib import sha256
from threading import Event
from typing import Any

from .backends import TokenHealingBackend
from .cacheprofile import candidate_cache_profile
from .core import Backend, Fixture, Format, Message, _valid_token_ids, evaluate, first_boundary
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
from .experiments_core import (
    STOP_SEMANTICS,
    _answer_digest,
    _backend_fingerprint,
    _benign_messages_for,
    _changed_rate,
    _forged_messages_for,
    _in_closed_range,
    _nonnegative_number,
    _original_fixture_id,
    _positive_number,
    _resolve_boundaries,
    _ScriptedEmissionBackend,
    _share_in_unit,
    stop_matrix,
)
from .experiments_probes import (
    _faketurn_cell_view,
)
from .faketurn import transfer_gap
from .fewshot import ARMS as FEWSHOT_ARMS
from .fewshot import (
    DETAIL_INSUFFICIENT_DEMO_POOL,
    FEWSHOT_K,
    build_demo_fixtures,
    change_rate,
    demo_permutations,
    order_sensitivity,
    sample_demos,
)
from .perturbation import PERTURBATION_OPERATORS, perturb_fixtures, variant_plan
from .portfolio import greedy_cover, oracle_rates, per_fixture_oracle_winners
from .scorerlens import LADDER_ID as SCORERLENS_LADDER_ID
from .scorerlens import LADDER_SPEC as SCORERLENS_LADDER_SPEC
from .scorerlens import analyze_candidate as scorerlens_analyze_candidate
from .scorerlens import ladder_sha256 as scorerlens_ladder_sha256
from .synthhistory import CONDITIONS, DELIMITER_LABELS, TURN_COUNTS, candidate_delimiter_label
from .verbosity import candidate_verbosity, is_brevity_ideal


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
    cancel: Event | None = None,
) -> dict[str, dict[str, bool | None]]:
    """Evaluate every candidate against each transfer checkpoint.

    Re-runs ``core.evaluate()`` for every search candidate against each
    additional local checkpoint, on the search split only (bounded, same
    fixture limit), with the same boundary and max_new_tokens. Greedy,
    exact. ``backend_factory(checkpoint_id)`` builds a backend per
    checkpoint; any backend construction or evaluation failure records
    None for that (checkpoint, candidate) cell -- fail-closed, never
    silently skipped. Construction is lazy (ORCH-03): the factory is
    called only when the stage reaches the checkpoint, and a backend
    that offers ``close()`` is closed before the next checkpoint is
    constructed, so at most one secondary backend is ever live.

    ``cancel`` is the run's shared cancellation event (ORCH-02): it is
    checked before each checkpoint's backend is constructed and before
    each candidate's evaluation, and threaded into ``core.evaluate``.
    Once set, every unexecuted cell records None -- the stage's
    existing fail-closed cell value -- and no further backend is
    constructed, so a cancelled transfer is indistinguishable in shape
    from an incompatible-checkpoint transfer.

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
        backend = None
        if cancel is None or not cancel.is_set():
            try:
                backend = backend_factory(ckpt)
            except Exception:  # noqa: BLE001 -- whole checkpoint degrades, not the stage
                backend = None
        try:
            for fmt in formats:
                if backend is None or (cancel is not None and cancel.is_set()):
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
                        cancel=cancel,
                    )
                except Exception:  # noqa: BLE001 -- one cell degrades, not the stage
                    per_candidate[fmt.name] = None
                    continue
                if not report.aggregate_eligible:
                    per_candidate[fmt.name] = None
                    continue
                passed = sum(o.exact_match and o.boundary_found for o in report.observations)
                per_candidate[fmt.name] = passed == report.planned_count
        finally:
            # ORCH-03: a checkpoint's backend is released before the
            # next checkpoint is constructed, so at most one secondary
            # backend is ever live. Backends without close() are left
            # to the garbage collector; a failing close never fails
            # the stage.
            if backend is not None:
                close = getattr(backend, "close", None)
                if callable(close):
                    # Teardown is best-effort.
                    with suppress(Exception):
                        close()
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
        """Return the rates."""
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
    winners = [name for name in evaluable if (entry := rates[name]) is not None and entry["pass"]]
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
    ``order_sensitivity`` (population std of the two per-order rates),
    both computed on the frozen common population of fixtures with a
    verdict on all three arms — the population ``n_evaluable``
    counts (SCI-B-03). The ``per_order`` rates stay pairwise, each
    with its own denominator. The selection rule is re-applied per
    arm (baseline, order_a, order_b) and each arm's (decision,
    winner) is compared against the primary run.

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
        # The comparative headlines (gain, order_sensitivity) are
        # computed on the frozen common population — the fixtures
        # with a verdict on all three arms, exactly the population
        # n_evaluable counts (SCI-B-03). Computing each order's
        # headline rate on its own pairwise population mixes
        # cohorts: a fixture missing only order_b would move the
        # order_a rate and hence both headlines while contributing
        # nothing to the common case. The per-order rates below
        # stay pairwise, reported with their own denominators.
        common = [
            fid
            for fid, per_arm in verdicts.items()
            if all(per_arm[arm] is not None for arm in FEWSHOT_ARMS)
        ]
        common_baseline = {fid: baseline[fid] for fid in common}
        order_rates: dict[str, dict[str, Any]] = {}
        headline_rates: list[float | None] = []
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
            common_computed = change_rate(
                common_baseline, {fid: verdicts[fid][arm] for fid in common}
            )
            headline_rates.append(common_computed["rate"])
        # change_rate only yields floats in [0, 1] or None, so
        # order_sensitivity cannot raise here either.
        sensitivity = order_sensitivity(tuple(headline_rates))
        # Both orders always contribute a headline rate here:
        # n_full >= 1 (checked above) means the common population
        # is nonempty, so each common-population rate is a float
        # and the filter below is provably nonempty (documented,
        # not defended: a defensive fallback would be dead code
        # and an uncoverable line).
        valid_rates = [rate for rate in headline_rates if rate is not None]
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
        """Return the pass count."""
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
        """Return the selection decision at a token budget."""
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


class _NegationCancelled(Exception):
    """Internal: the run's cancel event fired during the
    negation stage; the stage unwinds to its fail-closed
    payload rather than mislabelling the interruption.
    """


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


def serving_backends_analysis(stage: dict[str, Any]) -> dict[str, Any]:
    """Aggregate an exec-backends stage into the stored section (cycle 243).

    The stage comes from
    :func:`model_protocol_lab.servingbackends.run_stage` (or is the
    fail-closed ``{"available": False, "detail": ...}`` shape a probe
    gate produced). The section carries the stack identity and
    capabilities, the execution records, and three readouts derived
    from the records: the serving-stack stop matrix (§9.1.1 of the
    design — the stop_matrix row shape with the measured stack as
    the semantic), the measured cache companion (§9.1.3: the §5.4
    positions as reported, with the sequence's verdict as its
    detail), and the fidelity aggregation (§9.3). Post-hoc and
    opt-in only: the stage is produced by the ``exec-backends``
    subcommand, never during ``run``, and never feeds selection.
    Design: docs/SERVING_BACKENDS_DESIGN.md.
    """
    from . import servingbackends as _sb

    protocol = {"version": _sb.PROTOCOL_VERSION}
    if not stage.get("available"):
        return {
            "available": False,
            "detail": str(stage.get("detail", "serving_backends_unavailable")),
            "protocol": protocol,
            "identity": None,
            "capabilities": None,
            "records": [],
            "stop_matrix": {"available": False, "rows": []},
            "cache": {"available": False, "detail": "not_attempted", "positions": {}},
            "fidelity": {
                "available": False,
                "detail": "not_attempted",
                "verdict": "fidelity_unavailable",
                "counts": {},
                "divergence_classes": {},
                "per_fixture": [],
            },
            "audit": stage.get("audit"),
        }
    records: list[dict[str, Any]] = stage["records"]
    rows: list[dict[str, Any]] = []
    for record in records:
        if record.get("cache_position") != "standalone":
            continue
        mirror = record["mirror"]
        winner = record.get("stop_winner")
        available = record.get("cell_detail") != "capability_absent"
        # An unavailable cell observed nothing (the stack does not
        # offer the capability): its fired state is unobservable
        # (None), never a measured negative (ORCH-04).
        fired = (winner is not None) if available else None
        returned = record.get("returned_ids")
        config = record.get("request_config") or {}
        if config.get("include_stop_str_in_output"):
            arm = "keep"
        elif mirror.get("hf_stream") is not None:
            arm = "strip"
        else:
            arm = "plain"
        cell = {
            "available": available,
            "fired": fired,
            "firing_step": len(returned) if fired and type(returned) is list else None,
            "disposition": mirror.get("observed_disposition"),
            "finish_class": record.get("finish_class"),
            "verdict": mirror.get("fidelity_verdict"),
            "detail": record.get("cell_detail", ""),
        }
        flips = (
            [mirror["divergence_class"]]
            if mirror.get("fidelity_verdict") == "diverge" and mirror.get("divergence_class")
            else []
        )
        # The cells key is the stage identity's stack. A stage
        # without an identity predates stack-keying (run_stage
        # cannot produce one — it raises without an identity), so
        # the historical llamacpp key is the only honest fallback.
        stage_identity = stage.get("identity")
        stack_key = stage_identity.get("stack") if type(stage_identity) is dict else None
        rows.append(
            {
                "fixture": record.get("fixture"),
                "candidate": record.get("candidate"),
                "arm": arm,
                "cells": {stack_key if type(stack_key) is str and stack_key else "llamacpp": cell},
                "flips": flips,
            }
        )
    sequence = stage.get("cache_sequence")
    if sequence is None:
        cache: dict[str, Any] = {
            "available": False,
            "detail": "not_attempted",
            "positions": {},
        }
    else:
        cache = {
            "available": bool(sequence.get("available")),
            "detail": str(sequence.get("detail", "")),
            "positions": sequence.get("positions") or {},
        }
    aggregate = _sb.aggregate_fidelity(records)
    fidelity_result = stage.get("fidelity")
    fidelity: dict[str, Any] = {
        "available": bool(aggregate["available"]),
        "detail": "" if aggregate["available"] else "not_attempted",
        "verdict": aggregate["verdict"],
        "counts": aggregate["counts"],
        "divergence_classes": aggregate["divergence_classes"],
        "per_fixture": fidelity_result["per_fixture"] if type(fidelity_result) is dict else [],
    }
    audit = dict(stage.get("audit") or {})
    audit.setdefault("n_records", len(records))
    return {
        "available": True,
        "detail": str(stage.get("detail", "")),
        "protocol": protocol,
        "identity": stage.get("identity"),
        "capabilities": stage.get("capabilities"),
        "records": records,
        "stop_matrix": {"available": bool(rows), "rows": rows},
        "cache": cache,
        "fidelity": fidelity,
        "audit": audit,
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
        """Return the verdicts."""
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
        """Return the score."""
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
