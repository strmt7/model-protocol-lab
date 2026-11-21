"""diffreports text family: split from diffreports.py (W7-2).

Facades keep the public surface: import from model_protocol_lab.diffreports, not from this module.
"""

import hashlib
import json
import math
import re
from typing import Any

from .diffreports_core import (
    _display_id,
    _float_convertible,
    _recomputed_equal,
    _reject_constant,
    _reject_nonfinite_float,
)


def _valid_ws_canonical(payload: Any) -> bool:
    """Structural check for a ``ws_canonical`` payload.

    Enforces the wave-2 C3 lesson at the value level: the compared
    fields must be bools/strings/None/numbers so the renderers never
    hit a TypeError on a hostile payload, and map keys must be
    strings. A malformed side degrades to asymmetric instead of
    raising.
    """
    if type(payload) is not dict or payload.get("available") is not True:
        return False
    mutations = payload.get("mutations")
    if type(mutations) is not list or not mutations or not all(type(m) is str for m in mutations):
        return False
    fragile_candidates = payload.get("fragile_candidates")
    if fragile_candidates is not None and (
        type(fragile_candidates) is not list
        or not all(type(name) is str for name in fragile_candidates)
    ):
        return False
    per_candidate = payload.get("per_candidate")
    if type(per_candidate) is not dict or not all(type(k) is str for k in per_candidate):
        return False
    for entry in per_candidate.values():
        if type(entry) is not dict:
            return False
        robust = entry.get("whitespace_robust")
        if robust is not None and type(robust) is not bool:
            return False
        cells = entry.get("per_mutation")
        if type(cells) is not dict:
            return False
        for mutation, cell in cells.items():
            if type(mutation) is not str or type(cell) is not dict:
                return False
            for key in ("ids_divergence_rate", "verdict_flip_rate"):
                value = cell.get(key)
                if value is not None and type(value) not in (int, float):
                    return False
            for key in ("n_ids_divergent", "n_verdict_flips", "n_same_ids_verdict_flips"):
                value = cell.get(key)
                if value is not None and type(value) is not int:
                    return False
            if cell.get("fragile") is not None and type(cell.get("fragile")) is not bool:
                return False
    return True


def diff_ws_canonical(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``ws_canonical`` payloads; deterministic.

    Answers "did the whitespace-fragility story change between runs?":
    per-candidate per-mutation id-divergence-rate and verdict-flip-rate
    drift, whitespace-robust verdict flips, fragile-candidate set
    changes, and added/removed candidates or mutations. A newly fragile
    candidate after a template or tokenizer change is the regression
    signal. A side without a recorded whitespace scan is reported
    plainly. No model, no network.
    """
    old_ok = _valid_ws_canonical(old)
    new_ok = _valid_ws_canonical(new)
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

    def _candidate_view(payload: Any) -> dict[str, dict[str, Any]]:
        """Return the candidate view."""
        view: dict[str, dict[str, Any]] = {}
        for name, entry in (payload.get("per_candidate") or {}).items():
            entry = entry or {}
            cells = {}
            for mutation, cell in (entry.get("per_mutation") or {}).items():
                # _valid_ws_canonical guarantees str keys and dict cells,
                # so no defensive skip is needed here (it would be dead
                # code and an uncoverable line).
                cell = cell or {}
                cells[mutation] = {
                    "ids_divergence_rate": cell.get("ids_divergence_rate"),
                    "verdict_flip_rate": cell.get("verdict_flip_rate"),
                    "n_same_ids_verdict_flips": cell.get("n_same_ids_verdict_flips"),
                    "fragile": cell.get("fragile"),
                }
            view[name] = {
                "whitespace_robust": entry.get("whitespace_robust"),
                "cells": cells,
            }
        return view

    old_view = _candidate_view(old)
    new_view = _candidate_view(new)
    candidates_added = sorted(set(new_view) - set(old_view))
    candidates_removed = sorted(set(old_view) - set(new_view))
    old_mutations = [m for m in (old.get("mutations") or []) if type(m) is str]
    new_mutations = [m for m in (new.get("mutations") or []) if type(m) is str]
    mutations_added = sorted(set(new_mutations) - set(old_mutations))
    mutations_removed = sorted(set(old_mutations) - set(new_mutations))
    rate_drift: list[dict[str, Any]] = []
    robust_flips: list[dict[str, Any]] = []
    for name in sorted(set(old_view) & set(new_view)):
        old_robust = old_view[name]["whitespace_robust"]
        new_robust = new_view[name]["whitespace_robust"]
        if old_robust != new_robust:
            robust_flips.append(
                {"candidate": name, "old_robust": old_robust, "new_robust": new_robust}
            )
        for mutation in sorted(set(old_view[name]["cells"]) & set(new_view[name]["cells"])):
            for metric in (
                "ids_divergence_rate",
                "verdict_flip_rate",
                "n_same_ids_verdict_flips",
                "fragile",
            ):
                old_value = old_view[name]["cells"][mutation][metric]
                new_value = new_view[name]["cells"][mutation][metric]
                if old_value != new_value:
                    rate_drift.append(
                        {
                            "candidate": name,
                            "mutation": mutation,
                            "metric": metric,
                            "old": old_value,
                            "new": new_value,
                        }
                    )
    fragile_old = set(old.get("fragile_candidates") or [])
    fragile_new = set(new.get("fragile_candidates") or [])
    fragile_added = sorted(fragile_new - fragile_old)
    fragile_removed = sorted(fragile_old - fragile_new)
    result["rate_drift"] = rate_drift
    result["robust_flips"] = robust_flips
    result["fragile_added"] = fragile_added
    result["fragile_removed"] = fragile_removed
    result["candidates_added"] = candidates_added
    result["candidates_removed"] = candidates_removed
    result["mutations_added"] = mutations_added
    result["mutations_removed"] = mutations_removed
    # NOTE: ws-canonical-specific disjunct names keep this line
    # textually distinct from every other diff section's ``changed``
    # anchor (mutation-gate uniqueness).
    result["changed"] = bool(
        rate_drift
        or robust_flips
        or fragile_added
        or fragile_removed
        or candidates_added
        or candidates_removed
        or mutations_added
        or mutations_removed
    )
    return result


_CANONICALFORM_ARM_KEYS = ("nfc", "nfd")


_CANONICALFORM_BASELINE_ARM = "nfc"


_CANONICALFORM_CASE_COUNT = 24


_CANONICALFORM_FACT_TEMPLATE = "The {topic} inspection code is {value}."


_CANONICALFORM_QUESTION_TEMPLATE = "What is the {topic} inspection code?"


_CANONICALFORM_TOPICS = (
    "café",
    "naïve",
    "Zürich",
    "crème",
    "señor",
    "façade",
    "résumé",
    "piñata",
    "jalapeño",
    "soufflé",
    "touché",
    "protégé",
    "fiancé",
    "blasé",
    "cliché",
    "exposé",
    "saké",
    "pâté",
    "brûlée",
    "décor",
    "élan",
    "über",
    "schön",
    "tête",
)


_CANONICALFORM_WORDS = (
    "alder",
    "basalt",
    "cedar",
    "drift",
    "elm",
    "fern",
    "glade",
    "heath",
    "iris",
    "jade",
    "kelp",
    "loom",
    "moss",
    "nova",
    "ochre",
    "pine",
    "quill",
    "reed",
    "sage",
    "tide",
    "urn",
    "vale",
    "wren",
    "yew",
)


_CANONICALFORM_NEUTRAL_POOL = (
    "The hôtel lobby smells faintly of coffee and old paper.",
    "A forêt trail winds past the lake before sunrise.",
    "The côte road curves above the harbour all morning.",
    "The fête banners hang above the market stalls at dusk.",
    "A silver locket with a fête ribbon rests on the wooden dresser.",
    "Letters from his mère fill the shallow drawer.",
    "The théâtre posters cover the wall beside the ticket desk.",
    "A matinée crowd gathers outside the cinéma doors.",
    "The musée catalogue lists every painting by number.",
    "Rain taps the lycée windows above the courtyard all day.",
    "The opéra programme rests on top of the piano.",
    "A séance of quiet reading fills the long afternoon.",
    "The Noël market smells of pine and roasted chestnuts.",
    "Dust gathers on the crèche figures in the attic box.",
)


_CANONICALFORM_MIN_PAIRED_EVALUABLE = 6


_CANONICALFORM_MIN_DISCORDANT = 2


_CANONICALFORM_MAX_GROUND_RECORDS = 24 * _CANONICALFORM_CASE_COUNT


_CANONICALFORM_GENERATION_BUDGET = len(_CANONICALFORM_ARM_KEYS) * _CANONICALFORM_MAX_GROUND_RECORDS


_CANONICALFORM_MAX_RECORDS_BYTES = 12 * 1024 * 1024


_CANONICALFORM_CELL_FIELDS = {
    "verdict",
    "boundary_found",
    "boundary_text_match",
    "input_tokens",
    "answer_digest",
}


_CANONICALFORM_DETAILS = ("arm_context_exceeded", "arm_evaluation_failed")


_CANONICALFORM_AGGREGATE_KEYS = {
    "arms",
    "ids",
    "headline",
    "invariance_falsified",
    "n_cases",
    "n_incomplete",
}


def _canonicalform_digest(text: str) -> str:
    """SHA-256 hex digest (local copy; no imports from
    canonicalform.py anywhere in this mirror).
    """

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonicalform_digest_ok(value: Any) -> bool:
    """Whether a stored digest is None or a well-formed SHA-256
    hex digest (fails closed on lone surrogates, the cycle-190
    discipline).
    """
    if value is None:
        return True
    if type(value) is not str or len(value) != 64:
        return False
    return all(ch in "0123456789abcdef" for ch in value)


def _canonicalform_derive(seed: int, index: int, slot: int, counter: int) -> int:
    """Mirror of canonicalform._derive (frozen material string)."""

    material = f"{seed}|canonicalform|{index}|{slot}|{counter}"
    return int.from_bytes(hashlib.sha256(material.encode("ascii")).digest()[:8], "big")


def _canonicalform_generate(seed: int) -> list[dict[str, Any]]:
    """Mirror of canonicalform.generate_cases: the frozen case
    set for a seed, regenerated from the protocol's own seed so
    a fabricated set is rejected (the numform F2 lesson). The
    authored content is NFC; the NFD form is derived from it,
    exactly as the producer derives it.
    """
    import unicodedata as _unicodedata

    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index in range(_CANONICALFORM_CASE_COUNT):
        topic = _CANONICALFORM_TOPICS[index]
        counter = 0
        while True:
            word = _CANONICALFORM_WORDS[
                _canonicalform_derive(seed, index, 0, counter) % len(_CANONICALFORM_WORDS)
            ]
            number = 1000 + _canonicalform_derive(seed, index, 1, counter) % 9000
            value = f"{word}-{number}"
            if value not in seen:
                break
            counter += 1
        seen.add(value)
        first = _canonicalform_derive(seed, index, 2, 0) % len(_CANONICALFORM_NEUTRAL_POOL)
        second = _canonicalform_derive(seed, index, 3, 0) % len(_CANONICALFORM_NEUTRAL_POOL)
        if second == first:
            second = (first + 1) % len(_CANONICALFORM_NEUTRAL_POOL)
        fact = _CANONICALFORM_FACT_TEMPLATE.format(topic=topic, value=value)
        payload = (
            f"{_CANONICALFORM_NEUTRAL_POOL[first]} {fact} {_CANONICALFORM_NEUTRAL_POOL[second]}"
        )
        content = f"{payload} {_CANONICALFORM_QUESTION_TEMPLATE.format(topic=topic)}"
        cases.append(
            {
                "fixture_id": f"canonicalform-{index:02d}",
                "topic": topic,
                "value": value,
                "sentence_indices": [first, second],
                "content": content,
                "nfc_content": _unicodedata.normalize("NFC", content),
                "nfd_content": _unicodedata.normalize("NFD", content),
            }
        )
    return cases


def _canonicalform_case_material_sha256() -> str:
    """Mirror of canonicalform.case_material_sha256."""

    canonical = json.dumps(
        [
            _CANONICALFORM_FACT_TEMPLATE,
            _CANONICALFORM_QUESTION_TEMPLATE,
            list(_CANONICALFORM_TOPICS),
            list(_CANONICALFORM_WORDS),
            list(_CANONICALFORM_NEUTRAL_POOL),
        ],
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return _canonicalform_digest(canonical)


def _canonicalform_ground_sha256(seed: int) -> str:
    """Mirror of canonicalform.ground_sha256, including both
    arms' content digests so the ground digest binds the
    treatment.
    """

    rows = [
        [
            case["fixture_id"],
            case["topic"],
            case["value"],
            list(case["sentence_indices"]),
            _canonicalform_digest(case["nfc_content"]),
            _canonicalform_digest(case["nfd_content"]),
        ]
        for case in _canonicalform_generate(seed)
    ]
    canonical = json.dumps(rows, separators=(",", ":"), ensure_ascii=True)
    return _canonicalform_digest(canonical)


def _canonicalform_protocol(seed: int, backend_fingerprint: str) -> dict[str, Any]:
    """The canonical-form protocol rebuilt from this module's
    own literals plus the payload's seed and backend
    fingerprint, fingerprint included (compared whole).
    """

    protocol: dict[str, Any] = {
        "version": 1,
        "arm_keys": list(_CANONICALFORM_ARM_KEYS),
        "baseline_arm": _CANONICALFORM_BASELINE_ARM,
        "case_count": _CANONICALFORM_CASE_COUNT,
        "case_material_sha256": _canonicalform_case_material_sha256(),
        "ground_sha256": _canonicalform_ground_sha256(seed),
        "min_paired_evaluable": _CANONICALFORM_MIN_PAIRED_EVALUABLE,
        "min_discordant": _CANONICALFORM_MIN_DISCORDANT,
        "generation_budget": _CANONICALFORM_GENERATION_BUDGET,
        "max_ground_records": _CANONICALFORM_MAX_GROUND_RECORDS,
        "max_records_bytes": _CANONICALFORM_MAX_RECORDS_BYTES,
        "seed": seed,
        "backend_fingerprint": backend_fingerprint,
    }
    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    protocol["fingerprint"] = _canonicalform_digest(canonical)
    return protocol


def _canonicalform_mcnemar_p(a_only: int, b_only: int) -> float | None:
    """Two-sided exact McNemar p (the frozen formula, local copy)."""

    n = a_only + b_only
    if n == 0:
        return None
    k = min(a_only, b_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    result: float = min(1.0, 2.0 * tail)
    return result


def _canonicalform_cell_get(record: dict[str, Any], key: str) -> dict[str, Any] | None:
    """Return the canonicalform cell get."""
    cells = record.get("cells")
    if not isinstance(cells, dict):
        return None
    cell = cells.get(key)
    return cell if isinstance(cell, dict) else None


def _canonicalform_verdict(record: dict[str, Any], key: str) -> bool:
    """Return the canonicalform verdict."""
    cell = _canonicalform_cell_get(record, key)
    return bool(cell is not None and cell["verdict"])


def _canonicalform_pair_block_divergent(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Mirror of canonicalform._pair_block_divergent: the
    headline population is both-cells-completed AND
    ids-divergent; ids-identical records never enter.
    """
    n_paired = 0
    n_nfc_only = 0
    n_nfd_only = 0
    n_both = 0
    n_pass_nfc = 0
    n_pass_nfd = 0
    for record in records:
        if record.get("ids_identical") is not False:
            continue
        cell_nfc = _canonicalform_cell_get(record, "nfc")
        cell_nfd = _canonicalform_cell_get(record, "nfd")
        if cell_nfc is None or cell_nfd is None:
            continue
        n_paired += 1
        verdict_nfc = bool(cell_nfc["verdict"])
        verdict_nfd = bool(cell_nfd["verdict"])
        if verdict_nfc:
            n_pass_nfc += 1
        if verdict_nfd:
            n_pass_nfd += 1
        if verdict_nfc and verdict_nfd:
            n_both += 1
        elif verdict_nfc:
            n_nfc_only += 1
        elif verdict_nfd:
            n_nfd_only += 1
    return {
        "n_paired": n_paired,
        "n_nfc_only": n_nfc_only,
        "n_nfd_only": n_nfd_only,
        "n_both": n_both,
        "n_neither": n_paired - n_nfc_only - n_nfd_only - n_both,
        "rate_nfc": (n_pass_nfc / n_paired) if n_paired else None,
        "rate_nfd": (n_pass_nfd / n_paired) if n_paired else None,
        "mcnemar_p": _canonicalform_mcnemar_p(n_nfc_only, n_nfd_only),
    }


def _canonicalform_classify(n_paired: int, nfc_only: int, nfd_only: int) -> str:
    """Mirror of canonicalform.classify_canonical_form."""
    d = nfc_only + nfd_only
    if n_paired < _CANONICALFORM_MIN_PAIRED_EVALUABLE:
        return "insufficient"
    if d == 0:
        return "no_discordance_observed"
    if nfc_only == nfd_only and nfc_only >= _CANONICALFORM_MIN_DISCORDANT:
        return "discordant_balanced"
    if max(nfc_only, nfd_only) >= _CANONICALFORM_MIN_DISCORDANT:
        return "discordant_asymmetric"
    return "below_threshold"


def _canonicalform_falsified(classification: str) -> bool:
    """Mirror of canonicalform.invariance_falsified."""
    return classification in ("discordant_balanced", "discordant_asymmetric")


def _canonicalform_aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Mirror of canonicalform.aggregate_candidate."""
    arms: dict[str, Any] = {}
    for key in _CANONICALFORM_ARM_KEYS:
        n_completed = 0
        n_pass = 0
        for record in records:
            cell = _canonicalform_cell_get(record, key)
            if cell is None:
                continue
            n_completed += 1
            if cell["verdict"]:
                n_pass += 1
        arms[key] = {
            "n_completed": n_completed,
            "n_pass": n_pass,
            "rate": (n_pass / n_completed) if n_completed else None,
        }
    n_ids_identical = 0
    n_ids_divergent = 0
    deltas: list[int] = []
    for record in records:
        if record.get("ids_identical") is True:
            n_ids_identical += 1
        elif record.get("ids_identical") is False:
            n_ids_divergent += 1
        delta = record.get("id_length_delta")
        deltas.append(delta if type(delta) is int else 0)
    n_same_ids_flips = 0
    for record in records:
        if record.get("ids_identical") is not True:
            continue
        cell_nfc = _canonicalform_cell_get(record, "nfc")
        cell_nfd = _canonicalform_cell_get(record, "nfd")
        if cell_nfc is None or cell_nfd is None:
            continue
        if bool(cell_nfc["verdict"]) != bool(cell_nfd["verdict"]):
            n_same_ids_flips += 1
    headline_raw = _canonicalform_pair_block_divergent(records)
    classification = _canonicalform_classify(
        headline_raw["n_paired"], headline_raw["n_nfc_only"], headline_raw["n_nfd_only"]
    )
    headline = {
        "n_paired": headline_raw["n_paired"],
        "n_nfc_only": headline_raw["n_nfc_only"],
        "n_nfd_only": headline_raw["n_nfd_only"],
        "n_both": headline_raw["n_both"],
        "n_neither": headline_raw["n_neither"],
        "rate_nfc": headline_raw["rate_nfc"],
        "rate_nfd": headline_raw["rate_nfd"],
        "mcnemar_p": headline_raw["mcnemar_p"],
        "classification": classification,
    }
    return {
        "arms": arms,
        "ids": {
            "n_ids_identical": n_ids_identical,
            "n_ids_divergent": n_ids_divergent,
            "n_same_ids_verdict_flips": n_same_ids_flips,
            "id_length_delta_by_case": deltas,
            "max_abs_id_delta": max((abs(d) for d in deltas), default=0),
        },
        "headline": headline,
        "invariance_falsified": _canonicalform_falsified(classification),
        "n_cases": len(records),
        "n_incomplete": sum(
            1
            for record in records
            if _canonicalform_cell_get(record, "nfc") is None
            or _canonicalform_cell_get(record, "nfd") is None
        ),
    }


def _canonicalform_shared_fixtures(
    records_by_candidate: dict[str, list[dict[str, Any]]],
) -> frozenset[str]:
    """Return the canonicalform shared fixtures."""
    populations: list[frozenset[str]] = []
    for records in records_by_candidate.values():
        populations.append(frozenset(str(r.get("fixture")) for r in records))
    if not populations:
        return frozenset()
    return frozenset.intersection(*populations)


def _canonicalform_selection_over(
    records_by_candidate: dict[str, list[dict[str, Any]]],
    key: str,
    fixtures: frozenset[str],
) -> dict[str, Any] | None:
    """Mirror of canonicalform._selection_over."""
    completed_sets: list[frozenset[str]] = []
    all_pass: dict[str, bool | None] = {}
    for name, records in records_by_candidate.items():
        completed = frozenset(
            str(record["fixture"])
            for record in records
            if record["fixture"] in fixtures and _canonicalform_cell_get(record, key) is not None
        )
        completed_sets.append(completed)
        if not completed:
            return None
        all_pass[name] = all(
            _canonicalform_verdict(record, key)
            for record in records
            if record["fixture"] in completed
        )
    if len(set(completed_sets)) > 1:
        return None
    passers = [name for name, ok in all_pass.items() if ok]
    if len(passers) == 1:
        return {"decision": "selected", "winner": passers[0]}
    if len(passers) > 1:
        return {"decision": "ambiguous", "winner": None}
    return {"decision": "no_candidate", "winner": None}


def _canonicalform_arm_selection(
    records_by_candidate: dict[str, list[dict[str, Any]]],
    arm: str,
) -> dict[str, Any]:
    """Mirror of canonicalform.arm_selection."""
    shared = _canonicalform_shared_fixtures(records_by_candidate)
    own = _canonicalform_selection_over(records_by_candidate, arm, shared)
    if own is None:
        return {
            "decision": "unavailable",
            "detail": "selection_withheld_mismatched_sets",
            "winner": None,
            "withheld": True,
            "flip": None,
        }
    if arm == _CANONICALFORM_BASELINE_ARM:
        flip: bool | None = False
    else:
        completed = frozenset(
            str(record["fixture"])
            for records in records_by_candidate.values()
            for record in records
            if record["fixture"] in shared and _canonicalform_cell_get(record, arm) is not None
        )
        baseline = _canonicalform_selection_over(
            records_by_candidate, _CANONICALFORM_BASELINE_ARM, completed
        )
        flip = None if baseline is None else own != baseline
    return {
        "decision": own["decision"],
        "detail": "",
        "winner": own["winner"],
        "withheld": False,
        "flip": flip,
    }


def _canonicalform_winner_headline(conditions: dict[str, Any]) -> dict[str, Any]:
    """Mirror of canonicalform.winner_headline."""
    nfc = conditions.get("nfc") if isinstance(conditions, dict) else None
    nfd = conditions.get("nfd") if isinstance(conditions, dict) else None
    if not isinstance(nfc, dict) or not isinstance(nfd, dict):
        return {
            "headline": "winner_undefined",
            "detail": "selection_missing",
            "winner_changed": None,
        }
    if nfc.get("withheld") is True or nfd.get("withheld") is True:
        return {
            "headline": "winner_undefined",
            "detail": "selection_withheld_mismatched_sets",
            "winner_changed": None,
        }
    nfc_selected = nfc.get("decision") == "selected"
    nfd_selected = nfd.get("decision") == "selected"
    if nfc_selected and nfd_selected:
        if nfc.get("winner") != nfd.get("winner"):
            return {"headline": "winner_changed", "detail": "", "winner_changed": True}
        return {"headline": "winner_stable", "detail": "", "winner_changed": False}
    if nfc_selected != nfd_selected:
        return {
            "headline": "winner_abstention_asymmetric",
            "detail": "",
            "winner_changed": False,
        }
    return {"headline": "winner_undefined", "detail": "", "winner_changed": False}


def _canonicalform_cell_ok(cell: Any, expected_digest: str) -> bool:
    """One cell's shape and the verdict binding
    ``verdict == (answer_digest == expected_digest) and
    boundary_found``. The expected digest is the record's,
    itself checked against the mirror-regenerated case.
    """
    if type(cell) is not dict or set(cell) != _CANONICALFORM_CELL_FIELDS:
        return False
    if type(cell["verdict"]) is not bool or type(cell["boundary_found"]) is not bool:
        return False
    if cell["boundary_text_match"] is not None and type(cell["boundary_text_match"]) is not bool:
        return False
    if type(cell["input_tokens"]) is not int or cell["input_tokens"] < 0:
        return False
    if not _canonicalform_digest_ok(cell["answer_digest"]):
        return False
    verdict = (cell["answer_digest"] == expected_digest) and cell["boundary_found"]
    return bool(cell["verdict"] == verdict)


def _canonicalform_recompute(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Recompute a canonical-form payload from its ground records.

    The validator carries its own generator mirror, templates,
    pools, and aggregators (no imports from canonicalform.py):
    the protocol is rebuilt from frozen scalars plus the
    payload's seed and backend fingerprint and compared whole;
    the case set is regenerated from the seed and every record's
    construction is compared against it -- including, unlike
    placement, the TREATMENT itself: each record's NFC/NFD
    content digests must equal the digests of the mirror's own
    NFC/NFD derivations, the two digests must differ, and the
    id-divergence flags must be consistent with the arm token
    counts (ids_identical implies equal counts; the delta is
    the count difference). Every cell's verdict is re-derived
    from its digest identity; every completed cell's
    ``input_tokens`` must equal the record's arm token count
    for that arm; the aggregates, per-arm selections, and
    winner headline are recomputed and compared.
    """

    protocol = payload.get("protocol")
    if type(protocol) is not dict:
        return None
    seed = protocol.get("seed")
    backend_fingerprint = protocol.get("backend_fingerprint")
    if type(seed) is not int or isinstance(seed, bool) or type(backend_fingerprint) is not str:
        return None
    if not _recomputed_equal(protocol, _canonicalform_protocol(seed, backend_fingerprint)):
        return None
    candidates = payload.get("candidates")
    if type(candidates) is not dict or not candidates:
        return None
    generated = _canonicalform_generate(seed)
    total_records = 0
    records_bytes = 0
    aggregates: dict[str, Any] = {}
    raw_records: dict[str, list[dict[str, Any]]] = {}
    for name, entry in candidates.items():
        if type(entry) is not dict:
            return None
        status = entry.get("status")
        if status == "unavailable":
            if set(entry) != {"status", "detail", "records"} or entry.get("records") != []:
                return None
            continue
        if status != "available" or entry.get("detail") != "":
            return None
        if set(entry) != _CANONICALFORM_AGGREGATE_KEYS | {"status", "detail", "records"}:
            return None
        records = entry["records"]
        if type(records) is not list or len(records) != len(generated):
            return None
        total_records += len(records)
        records_bytes += len(
            json.dumps(records, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        )
        if records_bytes > _CANONICALFORM_MAX_RECORDS_BYTES:
            return None
        for record, case in zip(records, generated, strict=True):
            if type(record) is not dict or set(record) != {
                "fixture",
                "topic",
                "expected_digest",
                "nfc_tokens",
                "nfd_tokens",
                "ids_identical",
                "id_length_delta",
                "nfc_content_digest",
                "nfd_content_digest",
                "cells",
                "cell_details",
            }:
                return None
            if record["fixture"] != case["fixture_id"] or record["topic"] != case["topic"]:
                return None
            expected_digest = _canonicalform_digest(str(case["value"]))
            if record["expected_digest"] != expected_digest:
                return None
            if record["nfc_content_digest"] != _canonicalform_digest(case["nfc_content"]):
                return None
            if record["nfd_content_digest"] != _canonicalform_digest(case["nfd_content"]):
                return None
            if record["nfc_content_digest"] == record["nfd_content_digest"]:
                return None
            nfc_tokens = record["nfc_tokens"]
            nfd_tokens = record["nfd_tokens"]
            if type(nfc_tokens) is not int or nfc_tokens < 0:
                return None
            if type(nfd_tokens) is not int or nfd_tokens < 0:
                return None
            if type(record["ids_identical"]) is not bool:
                return None
            if record["ids_identical"] and nfc_tokens != nfd_tokens:
                return None
            if record["id_length_delta"] != nfc_tokens - nfd_tokens:
                return None
            cells = record["cells"]
            details = record["cell_details"]
            if type(cells) is not dict or set(cells) != set(_CANONICALFORM_ARM_KEYS):
                return None
            if type(details) is not dict or set(details) != set(_CANONICALFORM_ARM_KEYS):
                return None
            for arm in _CANONICALFORM_ARM_KEYS:
                cell = cells[arm]
                detail = details[arm]
                if cell is None:
                    if detail not in _CANONICALFORM_DETAILS:
                        return None
                else:
                    if detail != "":
                        return None
                    if not _canonicalform_cell_ok(cell, expected_digest):
                        return None
                    arm_tokens = nfc_tokens if arm == "nfc" else nfd_tokens
                    if cell["input_tokens"] != arm_tokens:
                        return None
        raw_records[name] = records
        aggregates[name] = _canonicalform_aggregate(records)
    if total_records > _CANONICALFORM_MAX_GROUND_RECORDS:
        return None
    if not raw_records:
        return None
    conditions = {
        arm: _canonicalform_arm_selection(raw_records, arm) for arm in _CANONICALFORM_ARM_KEYS
    }
    return {
        "aggregates": aggregates,
        "conditions": conditions,
        "selection": {
            "nfc": conditions["nfc"],
            "nfd": conditions["nfd"],
            "winner": _canonicalform_winner_headline(conditions),
        },
        "n_candidates": len(candidates),
        "n_cases": len(generated),
    }


def _valid_canonicalform(payload: Any) -> bool:
    """Structural check for a ``canonical_form`` payload (recompute standard)."""
    if type(payload) is not dict or payload.get("available") is not True:
        return False
    if set(payload) != {
        "available",
        "detail",
        "protocol",
        "candidates",
        "conditions",
        "selection",
        "n_candidates",
        "n_cases",
    }:
        return False
    if payload.get("detail") != "":
        return False
    recomputed = _canonicalform_recompute(payload)
    if recomputed is None:
        return False
    candidates = payload["candidates"]
    for name, expected in recomputed["aggregates"].items():
        entry = candidates[name]
        for key, value in expected.items():
            if not _recomputed_equal(entry.get(key), value):
                return False
    if not _recomputed_equal(payload.get("conditions"), recomputed["conditions"]):
        return False
    if not _recomputed_equal(payload.get("selection"), recomputed["selection"]):
        return False
    if not _recomputed_equal(payload.get("n_candidates"), recomputed["n_candidates"]):
        return False
    return _recomputed_equal(payload.get("n_cases"), recomputed["n_cases"])


def diff_canonicalform(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``canonical_form`` probe payloads; deterministic.

    Answers "did the canonical-form story change between runs?":
    per-candidate classification and falsification drift, the
    directional discordant counts, the ids-identical share and
    same-ids flip drift, the winner-headline drift, per-arm
    selection drift, candidates added/removed, and protocol /
    generator / backend drift (any of which marks the comparison
    not comparable, with the drift still listed -- the generated
    set is the treatment, so a changed generator is protocol
    drift). A side without a valid payload is reported plainly.
    """
    old_ok = _valid_canonicalform(old)
    new_ok = _valid_canonicalform(new)
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
    cf_protocol_drift = old["protocol"].get("fingerprint") != new["protocol"].get("fingerprint")
    cf_generator_drift = old["protocol"].get("ground_sha256") != new["protocol"].get(
        "ground_sha256"
    )
    cf_backend_drift = old["protocol"].get("backend_fingerprint") != new["protocol"].get(
        "backend_fingerprint"
    )
    cf_comparable = not cf_protocol_drift and not cf_generator_drift and not cf_backend_drift
    old_candidates = old["candidates"]
    new_candidates = new["candidates"]
    cf_candidates_added = sorted(set(new_candidates) - set(old_candidates))
    cf_candidates_removed = sorted(set(old_candidates) - set(new_candidates))
    cf_candidate_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_candidates) & set(new_candidates)):
        old_entry = old_candidates[name]
        new_entry = new_candidates[name]
        if old_entry.get("status") != "available" or new_entry.get("status") != "available":
            if old_entry.get("status") != new_entry.get("status"):
                cf_candidate_drift.append(
                    {
                        "candidate": name,
                        "metric": "status",
                        "old": old_entry.get("status"),
                        "new": new_entry.get("status"),
                    }
                )
            continue
        cf_metrics: list[tuple[str, Any, Any]] = [
            (
                "classification",
                old_entry["headline"]["classification"],
                new_entry["headline"]["classification"],
            ),
            (
                "invariance_falsified",
                old_entry["invariance_falsified"],
                new_entry["invariance_falsified"],
            ),
            (
                "n_nfc_only",
                old_entry["headline"]["n_nfc_only"],
                new_entry["headline"]["n_nfc_only"],
            ),
            (
                "n_nfd_only",
                old_entry["headline"]["n_nfd_only"],
                new_entry["headline"]["n_nfd_only"],
            ),
            (
                "n_ids_identical",
                old_entry["ids"]["n_ids_identical"],
                new_entry["ids"]["n_ids_identical"],
            ),
            (
                "n_same_ids_verdict_flips",
                old_entry["ids"]["n_same_ids_verdict_flips"],
                new_entry["ids"]["n_same_ids_verdict_flips"],
            ),
        ]
        for metric, old_value, new_value in cf_metrics:
            if old_value != new_value:
                cf_candidate_drift.append(
                    {
                        "candidate": name,
                        "metric": metric,
                        "old": old_value,
                        "new": new_value,
                    }
                )
    cf_condition_drift: list[dict[str, Any]] = []
    for arm in _CANONICALFORM_ARM_KEYS:
        if old["conditions"][arm] != new["conditions"][arm]:
            cf_condition_drift.append(
                {
                    "condition": arm,
                    "old": old["conditions"][arm],
                    "new": new["conditions"][arm],
                }
            )
    cf_selection_drift: list[dict[str, Any]] = []
    if old["selection"] != new["selection"]:
        cf_selection_drift.append(
            {
                "selection": "winner_headline",
                "old": old["selection"]["winner"],
                "new": new["selection"]["winner"],
            }
        )
    result["candidate_drift"] = cf_candidate_drift
    result["condition_drift"] = cf_condition_drift
    result["selection_drift"] = cf_selection_drift
    result["candidates_added"] = cf_candidates_added
    result["candidates_removed"] = cf_candidates_removed
    result["protocol_drift"] = cf_protocol_drift
    result["generator_drift"] = cf_generator_drift
    result["backend_drift"] = cf_backend_drift
    result["comparable"] = cf_comparable
    result["changed"] = bool(
        cf_candidate_drift
        or cf_condition_drift
        or cf_selection_drift
        or cf_candidates_added
        or cf_candidates_removed
        or cf_protocol_drift
        or cf_generator_drift
        or cf_backend_drift
    )
    return result


_RECORDNOTATION_WORDS = (
    "amber",
    "birch",
    "cedar",
    "copper",
    "coral",
    "falcon",
    "garnet",
    "harbor",
    "indigo",
    "juniper",
    "lagoon",
    "magnet",
    "marble",
    "nickel",
    "onyx",
    "pepper",
    "quartz",
    "saddle",
    "saffron",
    "summit",
    "tundra",
    "velvet",
    "walnut",
    "willow",
)


_RECORDNOTATION_FIELDS_3 = ("code", "label", "zone")


_RECORDNOTATION_FIELDS_4 = ("code", "label", "mark", "zone")


_RECORDNOTATION_ANSWER_FIELD = "code"


_RECORDNOTATION_KEY_FIELD = "zone"


_RECORDNOTATION_QUESTION_TEMPLATE = "What is the code of the record whose zone is {key}?"


_RECORDNOTATION_CONTRACT = "Answer with ONLY the value."


_RECORDNOTATION_GENERATOR_SPEC = (
    "rng = random.Random(seed * 1000 + i); per value draw, in order: "
    "word = WORDS[rng.randrange(24)], then number = rng.randrange(1000, 10000); "
    "a duplicate value is consumed and discarded; the table fills row-major, "
    "fields in alphabetical order; the queried record index is the next draw, "
    "rng.randrange(n_records), after the table is complete"
)


_RECORDNOTATION_HEADLINE_ARMS = ("prose", "canonical_json")


_RECORDNOTATION_SCREEN_ARM = "closed_book"


_RECORDNOTATION_ARM_KEYS = ("prose", "canonical_json", "closed_book")


_RECORDNOTATION_CASE_COUNT = 24


_RECORDNOTATION_CODE_SPACE = 216000


_RECORDNOTATION_MAX_CANDIDATES = 24


_RECORDNOTATION_MAX_GROUND_RECORDS = _RECORDNOTATION_MAX_CANDIDATES * _RECORDNOTATION_CASE_COUNT


_RECORDNOTATION_GENERATION_BUDGET = (
    len(_RECORDNOTATION_ARM_KEYS) * _RECORDNOTATION_MAX_GROUND_RECORDS
)


_RECORDNOTATION_HEADLINE_GENERATIONS = (
    len(_RECORDNOTATION_HEADLINE_ARMS) * _RECORDNOTATION_MAX_GROUND_RECORDS
)


_RECORDNOTATION_SCREEN_GENERATIONS = _RECORDNOTATION_MAX_GROUND_RECORDS


_RECORDNOTATION_MAX_RECORDS_BYTES = 12 * 1024 * 1024


_RECORDNOTATION_HEADLINE_CELL_FIELDS = {
    "verdict",
    "boundary_found",
    "boundary_text_match",
    "input_tokens",
    "answer_digest",
    "answer_text",
}


_RECORDNOTATION_SCREEN_CELL_FIELDS = {
    "verdict",
    "boundary_found",
    "boundary_text_match",
    "input_tokens",
    "answer_digest",
}


_RECORDNOTATION_DETAILS = ("arm_context_exceeded", "arm_evaluation_failed")


_RECORDNOTATION_CELL_CLASSES = ("pass", "boundary_only", "other_failure")


_RECORDNOTATION_PAIR_CLASSES = ("exact_only", "boundary_only", "mixed")


_RECORDNOTATION_OPERATING_CHARACTERISTICS: dict[str, Any] = {
    "transplant_fire_competent_independent": 0.8897,
    "transplant_fire_competent_paired_phi05": 0.5088,
    "transplant_fire_delta_005": 0.2276,
    "transplant_familywise_expected": 21.35,
    "transplant_familywise_any": 1.0,
    "rule_a_fire_competent_independent": 0.65,
    "rule_a_fire_competent_paired_phi05": 0.1648,
    "rule_a_fire_delta_005": 0.0298,
    "rule_a_familywise_expected": 0.71,
    "rule_a_familywise_any": 0.516,
    "mcnemar_fire_competent_independent": 0.0061,
    "mcnemar_fire_competent_paired_phi05": 0.0005,
    "mcnemar_power_gap25_marginals_85_60": 0.3547,
    "mcnemar_power_gap20_marginals_90_70": 0.2598,
    "mcnemar_power_gap25_base_90": 0.4,
    "sign_bonferroni_power_gap25_marginals_85_60": 0.0542,
    "sign_bonferroni_power_gap20_marginals_90_70": 0.0242,
    "panel96_bonferroni_false_fire": 0.0008,
    "panel96_bonferroni_power_gap25_marginals_85_60": 0.7452,
    "panel96_bonferroni_power_gap25_base_90": 0.8329,
    "panel96_generations": 6912,
    # bandit B105: the three 'pass' keys below are operating-
    # characteristic probabilities in the frozen payload schema,
    # not credentials.
    "winner_pass_probability_95": 0.292,  # nosec B105
    "winner_pass_probability_90": 0.08,  # nosec B105
    "winner_pass_probability_80": 0.005,  # nosec B105
    "screen_guess_probability": 1 / 216000,
    "screen_expected_exclusions_per_candidate": 24 / 216000,
    "screen_expected_exclusions_per_panel": 576 / 216000,
}


_RECORDNOTATION_AGGREGATE_KEYS = {
    "arms",
    "audit",
    "headline",
    "ids",
    "sensitivities",
    "operating_characteristics",
    "n_cases",
    "n_incomplete",
}


def _recordnotation_digest(text: str) -> str:
    """SHA-256 hex digest (local copy; no imports from
    recordnotation.py anywhere in this mirror).
    """

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _recordnotation_text_digest(text: str) -> str | None:
    """SHA-256 of the UTF-8 encoding, or None when the
    text cannot be encoded (a lone-surrogate answer_text
    is a validation failure, never an exception — the
    cycle-190 A3 pattern, applied per cycle-231 audit F2
    sibling sweep).
    """
    try:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    except UnicodeEncodeError:
        return None


def _recordnotation_digest_ok(value: Any) -> bool:
    """Whether a stored digest is a well-formed SHA-256 hex
    digest.
    """
    if type(value) is not str or len(value) != 64:
        return False
    return all(ch in "0123456789abcdef" for ch in value)


def _recordnotation_table_shape(index: int) -> tuple[int, tuple[str, ...]]:
    """Mirror of recordnotation.table_shape."""
    return 4 + (index % 3), (_RECORDNOTATION_FIELDS_4 if index % 2 else _RECORDNOTATION_FIELDS_3)


def _recordnotation_generate(seed: int) -> list[dict[str, Any]]:
    """Mirror of recordnotation.generate_cases: the frozen
    case generator with its RNG call sequence frozen
    (amendment A) -- per value draw the word first, then the
    number; duplicates consumed; row-major fill; the queried
    index is the next draw.
    """
    import random as _random

    if type(seed) is not int:
        raise ValueError("record-notation seed must be an int")
    cases: list[dict[str, Any]] = []
    for index in range(_RECORDNOTATION_CASE_COUNT):
        n_records, fields = _recordnotation_table_shape(index)
        rng = _random.Random(seed * 1000 + index)  # noqa: S311 # nosec B311
        values: list[str] = []
        seen: set[str] = set()
        while len(values) < n_records * len(fields):
            word = _RECORDNOTATION_WORDS[rng.randrange(len(_RECORDNOTATION_WORDS))]
            number = rng.randrange(1000, 10000)
            value = f"{word}-{number}"
            if value in seen:
                continue
            seen.add(value)
            values.append(value)
        table = [
            {field: values[r * len(fields) + j] for j, field in enumerate(fields)}
            for r in range(n_records)
        ]
        query_index = rng.randrange(n_records)
        cases.append(
            {
                "fixture_id": f"record-{index:02d}",
                "table": table,
                "query_index": query_index,
                "key": table[query_index][_RECORDNOTATION_KEY_FIELD],
                "expected": table[query_index][_RECORDNOTATION_ANSWER_FIELD],
            }
        )
    return cases


def _recordnotation_render_json(table: list[dict[str, str]]) -> str:
    """Mirror of recordnotation.render_json (the canonical
    JSON parameters, applied to the record sequence).
    """

    return json.dumps(table, sort_keys=True, separators=(", ", ": "), allow_nan=False)


def _recordnotation_render_prose(table: list[dict[str, str]]) -> str:
    """Mirror of recordnotation.render_prose."""
    sentences: list[str] = []
    for record in table:
        parts = [f"the {field} is {record[field]}" for field in sorted(record)]
        if len(parts) == 1:
            sentence = f"The {parts[0][4:]}."
        elif len(parts) == 2:
            sentence = f"The {parts[0][4:]} and {parts[1]}."
        else:
            sentence = "The " + ", ".join([parts[0][4:], *parts[1:-1]]) + f", and {parts[-1]}."
        sentences.append(sentence)
    return " ".join(sentences)


def _recordnotation_arm_content(case: dict[str, Any], arm: str) -> str:
    """Mirror of recordnotation.arm_content."""
    question = _RECORDNOTATION_QUESTION_TEMPLATE.format(key=case["key"])
    if arm == _RECORDNOTATION_SCREEN_ARM:
        return f"{question} {_RECORDNOTATION_CONTRACT}"
    if arm == "prose":
        rendering = _recordnotation_render_prose(case["table"])
    elif arm == "canonical_json":
        rendering = _recordnotation_render_json(case["table"])
    else:
        raise ValueError(f"unknown record-notation arm: {arm}")
    return f"{rendering}\n\n{question} {_RECORDNOTATION_CONTRACT}"


def _recordnotation_embedding_audit() -> dict[str, Any]:
    """Mirror of recordnotation.audit_answer_embedding (the
    guard set stated exactly, amendment B).
    """

    tokens = set(_RECORDNOTATION_FIELDS_4)
    tokens.update(
        re.findall(r"[a-z]+", _RECORDNOTATION_QUESTION_TEMPLATE.replace("{key}", " ").lower())
    )
    tokens.update(re.findall(r"[a-z]+", _RECORDNOTATION_CONTRACT.lower()))
    violations = 0
    for token in tokens:
        for word in _RECORDNOTATION_WORDS:
            if token[:4] == word[:4] or word.startswith(token) or token.startswith(word):
                violations += 1
    return {
        "template_violations": violations,
        "key_word_prefix_cases": len(_RECORDNOTATION_WORDS),
        "key_word_prefix_of": len(_RECORDNOTATION_WORDS),
        "answer_key_same_word_rate": 8999 / 215999,
    }


def _recordnotation_case_material_sha256() -> str:
    """Mirror of recordnotation.case_material_sha256."""

    material = {
        "words": list(_RECORDNOTATION_WORDS),
        "fields_3": list(_RECORDNOTATION_FIELDS_3),
        "fields_4": list(_RECORDNOTATION_FIELDS_4),
        "answer_field": _RECORDNOTATION_ANSWER_FIELD,
        "key_field": _RECORDNOTATION_KEY_FIELD,
        "question_template": _RECORDNOTATION_QUESTION_TEMPLATE,
        "contract": _RECORDNOTATION_CONTRACT,
        "generator_spec": _RECORDNOTATION_GENERATOR_SPEC,
        "case_count": _RECORDNOTATION_CASE_COUNT,
        "arm_keys": list(_RECORDNOTATION_ARM_KEYS),
        "embedding_audit": _recordnotation_embedding_audit(),
    }
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return _recordnotation_digest(canonical)


def _recordnotation_ground_sha256(seed: int) -> str:
    """Mirror of recordnotation.ground_sha256."""

    rows = [
        [
            case["fixture_id"],
            [[[field, record[field]] for field in sorted(record)] for record in case["table"]],
            case["query_index"],
            case["key"],
            case["expected"],
        ]
        for case in _recordnotation_generate(seed)
    ]
    canonical = json.dumps(rows, separators=(",", ":"), ensure_ascii=True)
    return _recordnotation_digest(canonical)


def _recordnotation_protocol(seed: int, backend_fingerprint: str) -> dict[str, Any]:
    """The record-notation protocol rebuilt from this module's
    own literals, fingerprint included (compared whole).
    """

    protocol: dict[str, Any] = {
        "version": 1,
        "arm_keys": list(_RECORDNOTATION_ARM_KEYS),
        "headline_arms": list(_RECORDNOTATION_HEADLINE_ARMS),
        "screen_arm": _RECORDNOTATION_SCREEN_ARM,
        "answer_field": _RECORDNOTATION_ANSWER_FIELD,
        "key_field": _RECORDNOTATION_KEY_FIELD,
        "case_count": _RECORDNOTATION_CASE_COUNT,
        "code_space": _RECORDNOTATION_CODE_SPACE,
        "case_material_sha256": _recordnotation_case_material_sha256(),
        "ground_sha256": _recordnotation_ground_sha256(seed),
        "embedding_audit": _recordnotation_embedding_audit(),
        "generation_budget": _RECORDNOTATION_GENERATION_BUDGET,
        "headline_generations": _RECORDNOTATION_HEADLINE_GENERATIONS,
        "screen_generations": _RECORDNOTATION_SCREEN_GENERATIONS,
        "max_ground_records": _RECORDNOTATION_MAX_GROUND_RECORDS,
        "max_records_bytes": _RECORDNOTATION_MAX_RECORDS_BYTES,
        "max_candidates": _RECORDNOTATION_MAX_CANDIDATES,
        "seed": seed,
        "backend_fingerprint": backend_fingerprint,
    }
    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    protocol["fingerprint"] = _recordnotation_digest(canonical)
    return protocol


def _recordnotation_scorer_r1(text: str) -> str:
    """Literal mirror of scorerlens._norm_r1: trim, collapse
    space/tab runs, cap newline runs at two.
    """

    out = text.strip()
    out = out.replace("\r\n", "\n").replace("\r", "\n")
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out


def _recordnotation_scorer_fence(text: str) -> str:
    """Literal mirror of scorerlens._fence_strip: remove one
    pair of line-anchored fences, if present.
    """

    lines = text.split("\n")
    if len(lines) >= 3 and re.fullmatch(r"```[^`]*", lines[0]) and lines[-1] == "```":
        return "\n".join(lines[1:-1]).strip("\n")
    return text


def _recordnotation_cell_get(record: dict[str, Any], key: str) -> dict[str, Any] | None:
    """Return the recordnotation cell get."""
    cells = record.get("cells")
    if not isinstance(cells, dict):
        return None
    cell = cells.get(key)
    return cell if isinstance(cell, dict) else None


def _recordnotation_detail_get(record: dict[str, Any], key: str) -> str:
    """Return the recordnotation detail get."""
    details = record.get("cell_details")
    if not isinstance(details, dict):
        return ""
    detail = details.get(key, "")
    return detail if type(detail) is str else ""


def _recordnotation_verdict(record: dict[str, Any], key: str) -> bool:
    """Return the recordnotation verdict."""
    cell = _recordnotation_cell_get(record, key)
    return bool(cell is not None and cell["verdict"])


def _recordnotation_screened(record: dict[str, Any]) -> bool:
    """Return the recordnotation screened."""
    cell = _recordnotation_cell_get(record, _RECORDNOTATION_SCREEN_ARM)
    return bool(cell is not None and cell["verdict"])


def _recordnotation_ids_identical(record: dict[str, Any]) -> bool:
    """Mirror of recordnotation._ids_identical: derived from
    the recorded id digests alone.
    """
    digests = record.get("id_digests")
    if not isinstance(digests, dict):
        return False
    return digests.get("prose") is not None and digests.get("prose") == digests.get(
        "canonical_json"
    )


def _recordnotation_decompose_cell(cell: dict[str, Any], expected_digest: str) -> str:
    """Mirror of recordnotation.decompose_cell."""
    if cell.get("verdict") is True:
        return "pass"
    digest = cell.get("answer_digest")
    if digest is not None and digest == expected_digest:
        return "boundary_only"
    return "other_failure"


def _recordnotation_decompose_pair(prose_cell: dict[str, Any], json_cell: dict[str, Any]) -> str:
    """Mirror of recordnotation.decompose_pair."""
    if prose_cell.get("answer_digest") == json_cell.get("answer_digest"):
        return "boundary_only"
    if prose_cell.get("boundary_found") and json_cell.get("boundary_found"):
        return "exact_only"
    return "mixed"


def _recordnotation_parsed_hit(answer_text: str, expected: str) -> bool:
    """Mirror of recordnotation._parsed_hit."""

    try:
        parsed = json.loads(
            answer_text,
            parse_constant=_reject_constant,
            parse_float=_reject_nonfinite_float,
        )
    except (RecursionError, ValueError):
        return False
    return type(parsed) is dict and parsed.get(_RECORDNOTATION_ANSWER_FIELD) == expected


def _recordnotation_regrade_verdict(cell: dict[str, Any], expected: str, transform: Any) -> bool:
    """Mirror of recordnotation._regrade_verdict."""
    text = cell.get("answer_text")
    if type(text) is not str:
        return False
    return transform(text) == expected and bool(cell.get("boundary_found"))


def _recordnotation_aggregate(
    records: list[dict[str, Any]], cases: list[dict[str, Any]]
) -> dict[str, Any]:
    """Mirror of recordnotation.aggregate_candidate."""
    expected_by_fixture = {str(case["fixture_id"]): str(case["expected"]) for case in cases}
    arms: dict[str, Any] = {}
    for key in _RECORDNOTATION_ARM_KEYS:
        n_completed = 0
        n_pass = 0
        n_context = 0
        n_failed = 0
        decomposition = dict.fromkeys(_RECORDNOTATION_CELL_CLASSES, 0)
        n_failures = 0
        echo_hits = 0
        parsed_hits = 0
        for record in records:
            detail = _recordnotation_detail_get(record, key)
            if detail == "arm_context_exceeded":
                n_context += 1
            elif detail == "arm_evaluation_failed":
                n_failed += 1
            cell = _recordnotation_cell_get(record, key)
            if cell is None:
                continue
            n_completed += 1
            if cell["verdict"]:
                n_pass += 1
            decomposition[
                _recordnotation_decompose_cell(cell, str(record.get("expected_digest", "")))
            ] += 1
            if key in _RECORDNOTATION_HEADLINE_ARMS and not cell["verdict"]:
                n_failures += 1
                text = cell.get("answer_text")
                expected = expected_by_fixture.get(str(record.get("fixture", "")), "")
                if type(text) is str:
                    if expected in text:
                        echo_hits += 1
                    if _recordnotation_parsed_hit(text, expected):
                        parsed_hits += 1
        arms[key] = {
            "n_completed": n_completed,
            "n_pass": n_pass,
            "rate": (n_pass / n_completed) if n_completed else None,
            "n_context_exceeded": n_context,
            "n_evaluation_failed": n_failed,
            "decomposition": decomposition,
            "n_failures": n_failures,
            "echo_rate": ((echo_hits / n_failures) if n_failures else None)
            if key in _RECORDNOTATION_HEADLINE_ARMS
            else None,
            "parsed_rate": ((parsed_hits / n_failures) if n_failures else None)
            if key in _RECORDNOTATION_HEADLINE_ARMS
            else None,
        }
    both = [
        record
        for record in records
        if _recordnotation_cell_get(record, "prose") is not None
        and _recordnotation_cell_get(record, "canonical_json") is not None
    ]
    unscreened_pairs = [record for record in both if not _recordnotation_ids_identical(record)]
    live_pairs = [record for record in unscreened_pairs if not _recordnotation_screened(record)]

    def _discordance(pairs: list[dict[str, Any]]) -> dict[str, Any]:
        """Return the discordance."""
        prose_only = 0
        json_only = 0
        decomposition = dict.fromkeys(_RECORDNOTATION_PAIR_CLASSES, 0)
        for record in pairs:
            prose_cell = _recordnotation_cell_get(record, "prose")
            json_cell = _recordnotation_cell_get(record, "canonical_json")
            if prose_cell is None or json_cell is None:
                continue
            if bool(prose_cell["verdict"]) == bool(json_cell["verdict"]):
                continue
            if prose_cell["verdict"]:
                prose_only += 1
            else:
                json_only += 1
            decomposition[_recordnotation_decompose_pair(prose_cell, json_cell)] += 1
        return {
            "n_paired": len(pairs),
            "discordant": prose_only + json_only,
            "n_prose_only": prose_only,
            "n_canonical_json_only": json_only,
            "decomposition": decomposition,
        }

    headline_counts = _discordance(live_pairs)
    headline = {
        **headline_counts,
        "prose_rate": (
            (
                sum(1 for record in live_pairs if _recordnotation_verdict(record, "prose"))
                / len(live_pairs)
            )
            if live_pairs
            else None
        ),
        "canonical_json_rate": (
            (
                sum(1 for record in live_pairs if _recordnotation_verdict(record, "canonical_json"))
                / len(live_pairs)
            )
            if live_pairs
            else None
        ),
        "resolution_points": (100.0 / len(live_pairs)) if live_pairs else None,
    }
    deltas: list[int] = []
    n_same_ids_flips = 0
    for record in records:
        tokens = record.get("arm_tokens")
        tokens = tokens if isinstance(tokens, dict) else {}
        prose_tokens = tokens.get("prose")
        json_tokens = tokens.get("canonical_json")
        if type(prose_tokens) is int and type(json_tokens) is int:
            deltas.append(json_tokens - prose_tokens)
        else:
            deltas.append(0)
        if _recordnotation_ids_identical(record):
            prose_cell = _recordnotation_cell_get(record, "prose")
            json_cell = _recordnotation_cell_get(record, "canonical_json")
            if (
                prose_cell is not None
                and json_cell is not None
                and prose_cell["verdict"] != json_cell["verdict"]
            ):
                n_same_ids_flips += 1
    ids = {
        "n_ids_identical": sum(1 for record in records if _recordnotation_ids_identical(record)),
        "id_length_delta": deltas,
        "max_abs_id_delta": max((abs(delta) for delta in deltas), default=0),
        "n_same_ids_verdict_flips": n_same_ids_flips,
    }
    regrade: dict[str, int] = {}
    for label, transform in (
        ("r1", _recordnotation_scorer_r1),
        ("fence", _recordnotation_scorer_fence),
    ):
        count = 0
        for record in live_pairs:
            prose_cell = _recordnotation_cell_get(record, "prose")
            json_cell = _recordnotation_cell_get(record, "canonical_json")
            if prose_cell is None or json_cell is None:
                continue
            expected = expected_by_fixture.get(str(record.get("fixture", "")), "")
            if _recordnotation_regrade_verdict(
                prose_cell, expected, transform
            ) != _recordnotation_regrade_verdict(json_cell, expected, transform):
                count += 1
        regrade[label] = count
    sensitivities = {
        "unscreened": _discordance(unscreened_pairs),
        "regrade_r1_discordant": regrade["r1"],
        "regrade_fence_discordant": regrade["fence"],
    }
    return {
        "arms": arms,
        "audit": {
            "n_screened": sum(1 for record in records if _recordnotation_screened(record)),
            "n_closed_book_incomplete": sum(
                1
                for record in records
                if _recordnotation_cell_get(record, _RECORDNOTATION_SCREEN_ARM) is None
            ),
        },
        "headline": headline,
        "ids": ids,
        "sensitivities": sensitivities,
        "operating_characteristics": dict(_RECORDNOTATION_OPERATING_CHARACTERISTICS),
        "n_cases": len(records),
        "n_incomplete": sum(
            1
            for record in records
            if any(
                _recordnotation_cell_get(record, arm) is None for arm in _RECORDNOTATION_ARM_KEYS
            )
        ),
    }


def _recordnotation_cell_ok(cell: Any, expected_digest: str, headline: bool) -> bool:
    """One cell's shape and bindings: exact field set for its
    arm class; the verdict binding ``verdict ==
    (answer_digest == expected_digest) and boundary_found``;
    the input-token binding to the record's arm token count is
    checked by the caller (amendment D); a headline cell's
    carried ``answer_text`` must hash to its ``answer_digest``
    (the cycle-190 binding), and a screen cell carries no text
    at all.
    """
    fields = (
        _RECORDNOTATION_HEADLINE_CELL_FIELDS if headline else _RECORDNOTATION_SCREEN_CELL_FIELDS
    )
    if type(cell) is not dict or set(cell) != fields:
        return False
    if type(cell["verdict"]) is not bool or type(cell["boundary_found"]) is not bool:
        return False
    if cell["boundary_text_match"] is not None and type(cell["boundary_text_match"]) is not bool:
        return False
    if type(cell["input_tokens"]) is not int or cell["input_tokens"] < 0:
        return False
    digest = cell["answer_digest"]
    if digest is not None and not _recordnotation_digest_ok(digest):
        return False
    if headline:
        text = cell["answer_text"]
        if text is None:
            if digest is not None:
                return False
        elif type(text) is not str or _recordnotation_text_digest(text) != digest:
            return False
    verdict = (digest == expected_digest) and cell["boundary_found"]
    return bool(cell["verdict"] == verdict)


def _recordnotation_recompute(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Recompute a record-notation payload from its ground
    records.

    The validator carries its own generator mirror with the
    frozen RNG call sequence, both renderers, the embedding
    audit, and the aggregators (no imports from
    recordnotation.py): the protocol is rebuilt from frozen
    literals plus the payload's seed and backend fingerprint
    and compared whole -- the fingerprint binds the
    seed-to-table mapping, so a payload whose tables were
    generated under any other RNG reading fails regeneration
    (amendment A); the case set is regenerated from the seed
    and every record is checked against it in protocol order;
    every record's per-arm content digests are rebound to the
    regenerated arm contents; the record's ids_identical flag
    and id_length_delta must equal the derivations from its
    recorded digests and token counts; every headline cell's
    answer_text is verified against its answer_digest, its
    verdict is re-derived from its digest identity, and its
    input_tokens are bound to the record's arm token count
    (amendment D -- a payload with fabricated token fields
    fails here, as it does under every sibling mirror); the
    decompositions, diagnostics, headline discordance,
    sensitivities, and id report are recomputed from the
    records and the regenerated expected values; the
    aggregates are recomputed and compared. Cells exist only
    from aggregate-eligible reports, so a non-null cell
    certifies a completed evaluation and the verdict's status
    conjunct is discharged structurally (amendment D).
    """

    protocol = payload.get("protocol")
    if type(protocol) is not dict:
        return None
    seed = protocol.get("seed")
    backend_fingerprint = protocol.get("backend_fingerprint")
    if type(seed) is not int or isinstance(seed, bool) or type(backend_fingerprint) is not str:
        return None
    if not _recomputed_equal(protocol, _recordnotation_protocol(seed, backend_fingerprint)):
        return None
    candidates = payload.get("candidates")
    if type(candidates) is not dict or not candidates:
        return None
    if len(candidates) > _RECORDNOTATION_MAX_CANDIDATES:
        return None
    generated = _recordnotation_generate(seed)
    total_records = 0
    records_bytes = 0
    aggregates: dict[str, Any] = {}
    for name, entry in candidates.items():
        if type(entry) is not dict:
            return None
        status = entry.get("status")
        if status == "unavailable":
            if set(entry) != {"status", "detail", "records"} or entry.get("records") != []:
                return None
            continue
        if status != "available" or entry.get("detail") != "":
            return None
        if set(entry) != _RECORDNOTATION_AGGREGATE_KEYS | {"status", "detail", "records"}:
            return None
        records = entry["records"]
        if type(records) is not list or len(records) != len(generated):
            return None
        total_records += len(records)
        records_bytes += len(
            json.dumps(records, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        )
        if records_bytes > _RECORDNOTATION_MAX_RECORDS_BYTES:
            return None
        for record, case in zip(records, generated, strict=True):
            if type(record) is not dict or set(record) != {
                "fixture",
                "n_records",
                "n_fields",
                "expected_digest",
                "content_digests",
                "arm_tokens",
                "id_digests",
                "ids_identical",
                "id_length_delta",
                "cells",
                "cell_details",
            }:
                return None
            if record["fixture"] != case["fixture_id"]:
                return None
            if record["n_records"] != len(case["table"]) or record["n_fields"] != len(
                case["table"][0]
            ):
                return None
            expected_digest = _recordnotation_digest(str(case["expected"]))
            if record["expected_digest"] != expected_digest:
                return None
            arm_tokens = record["arm_tokens"]
            id_digests = record["id_digests"]
            content_digests = record["content_digests"]
            cells = record["cells"]
            details = record["cell_details"]
            for mapping in (arm_tokens, id_digests, content_digests, cells, details):
                if type(mapping) is not dict or set(mapping) != set(_RECORDNOTATION_ARM_KEYS):
                    return None
            if record["ids_identical"] is not (id_digests["prose"] == id_digests["canonical_json"]):
                return None
            for arm in _RECORDNOTATION_ARM_KEYS:
                tokens = arm_tokens[arm]
                if type(tokens) is not int or isinstance(tokens, bool) or tokens < 0:
                    return None
                if not _recordnotation_digest_ok(id_digests[arm]):
                    return None
                if content_digests[arm] != _recordnotation_digest(
                    _recordnotation_arm_content(case, arm)
                ):
                    return None
                cell = cells[arm]
                detail = details[arm]
                if detail not in ("", *_RECORDNOTATION_DETAILS):
                    return None
                if cell is None:
                    if detail == "":
                        return None
                    continue
                if detail != "":
                    return None
                if not _recordnotation_cell_ok(
                    cell, expected_digest, arm in _RECORDNOTATION_HEADLINE_ARMS
                ):
                    return None
                if cell["input_tokens"] != tokens:
                    return None
            if record["id_length_delta"] != arm_tokens["canonical_json"] - arm_tokens["prose"]:
                return None
        aggregates[name] = _recordnotation_aggregate(records, generated)
    if total_records > _RECORDNOTATION_MAX_GROUND_RECORDS:
        return None
    if not aggregates:
        return None
    return {
        "aggregates": aggregates,
        "n_candidates": len(candidates),
        "n_cases": len(generated),
    }


def _valid_record_notation(payload: Any) -> bool:
    """Structural check for a ``record_notation`` payload
    (recompute standard). The audit block is shape-checked
    only: its counts are stage observations (special-token
    findings against the run's own formats), not recomputable
    from records.
    """
    if type(payload) is not dict or payload.get("available") is not True:
        return False
    if set(payload) != {
        "available",
        "detail",
        "protocol",
        "audit",
        "candidates",
        "n_candidates",
        "n_cases",
    }:
        return False
    if payload.get("detail") != "":
        return False
    audit = payload.get("audit")
    if type(audit) is not dict or set(audit) != {
        "n_audit_excluded",
        "by_reason",
        "special_token_findings",
    }:
        return False
    if type(audit["n_audit_excluded"]) is not int or audit["n_audit_excluded"] < 0:
        return False
    if type(audit["special_token_findings"]) is not int or audit["special_token_findings"] < 0:
        return False
    by_reason = audit["by_reason"]
    if type(by_reason) is not dict:
        return False
    for key, count in by_reason.items():
        if type(key) is not str or type(count) is not int or count < 0:
            return False
    recomputed = _recordnotation_recompute(payload)
    if recomputed is None:
        return False
    candidates = payload["candidates"]
    for name, expected in recomputed["aggregates"].items():
        entry = candidates[name]
        for key, value in expected.items():
            if not _recomputed_equal(entry.get(key), value):
                return False
    if not _recomputed_equal(payload.get("n_candidates"), recomputed["n_candidates"]):
        return False
    return _recomputed_equal(payload.get("n_cases"), recomputed["n_cases"])


def diff_record_notation(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``record_notation`` probe payloads;
    deterministic.

    Answers "did the record-notation story change between
    runs?": per-candidate drift in the audit counts, the
    headline paired discordance with its direction and
    decomposition, the paired per-arm rates, the per-arm echo
    and parsed-value rates, the id report, and the
    sensitivities; candidates added/removed; and protocol /
    generator / backend drift (any of which marks the
    comparison not comparable, with the drift still listed --
    the generated panel is the treatment, so a changed
    generator is protocol drift). The instrument is
    descriptive: there is no decision whose drift could be
    reported. A side without a valid payload is reported
    plainly.
    """
    old_ok = _valid_record_notation(old)
    new_ok = _valid_record_notation(new)
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
    rn_protocol_drift = old["protocol"].get("fingerprint") != new["protocol"].get("fingerprint")
    rn_generator_drift = old["protocol"].get("ground_sha256") != new["protocol"].get(
        "ground_sha256"
    ) or old["protocol"].get("case_material_sha256") != new["protocol"].get("case_material_sha256")
    rn_backend_drift = old["protocol"].get("backend_fingerprint") != new["protocol"].get(
        "backend_fingerprint"
    )
    rn_comparable = not rn_protocol_drift and not rn_generator_drift and not rn_backend_drift
    old_candidates = old["candidates"]
    new_candidates = new["candidates"]
    rn_candidates_added = sorted(set(new_candidates) - set(old_candidates))
    rn_candidates_removed = sorted(set(old_candidates) - set(new_candidates))
    rn_candidate_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_candidates) & set(new_candidates)):
        old_entry = old_candidates[name]
        new_entry = new_candidates[name]
        if old_entry.get("status") != "available" or new_entry.get("status") != "available":
            if old_entry.get("status") != new_entry.get("status"):
                rn_candidate_drift.append(
                    {
                        "candidate": name,
                        "metric": "status",
                        "old": old_entry.get("status"),
                        "new": new_entry.get("status"),
                    }
                )
            continue
        rn_metrics: list[tuple[str, Any, Any]] = [
            (
                "n_screened",
                old_entry["audit"]["n_screened"],
                new_entry["audit"]["n_screened"],
            ),
            (
                "n_closed_book_incomplete",
                old_entry["audit"]["n_closed_book_incomplete"],
                new_entry["audit"]["n_closed_book_incomplete"],
            ),
            ("n_paired", old_entry["headline"]["n_paired"], new_entry["headline"]["n_paired"]),
            (
                "discordant",
                old_entry["headline"]["discordant"],
                new_entry["headline"]["discordant"],
            ),
            (
                "n_prose_only",
                old_entry["headline"]["n_prose_only"],
                new_entry["headline"]["n_prose_only"],
            ),
            (
                "n_canonical_json_only",
                old_entry["headline"]["n_canonical_json_only"],
                new_entry["headline"]["n_canonical_json_only"],
            ),
            (
                "prose_rate",
                old_entry["headline"]["prose_rate"],
                new_entry["headline"]["prose_rate"],
            ),
            (
                "canonical_json_rate",
                old_entry["headline"]["canonical_json_rate"],
                new_entry["headline"]["canonical_json_rate"],
            ),
            (
                "decomposition_exact_only",
                old_entry["headline"]["decomposition"]["exact_only"],
                new_entry["headline"]["decomposition"]["exact_only"],
            ),
            (
                "decomposition_boundary_only",
                old_entry["headline"]["decomposition"]["boundary_only"],
                new_entry["headline"]["decomposition"]["boundary_only"],
            ),
            (
                "decomposition_mixed",
                old_entry["headline"]["decomposition"]["mixed"],
                new_entry["headline"]["decomposition"]["mixed"],
            ),
            (
                "prose_echo_rate",
                old_entry["arms"]["prose"]["echo_rate"],
                new_entry["arms"]["prose"]["echo_rate"],
            ),
            (
                "prose_parsed_rate",
                old_entry["arms"]["prose"]["parsed_rate"],
                new_entry["arms"]["prose"]["parsed_rate"],
            ),
            (
                "canonical_json_echo_rate",
                old_entry["arms"]["canonical_json"]["echo_rate"],
                new_entry["arms"]["canonical_json"]["echo_rate"],
            ),
            (
                "canonical_json_parsed_rate",
                old_entry["arms"]["canonical_json"]["parsed_rate"],
                new_entry["arms"]["canonical_json"]["parsed_rate"],
            ),
            (
                "n_ids_identical",
                old_entry["ids"]["n_ids_identical"],
                new_entry["ids"]["n_ids_identical"],
            ),
            (
                "max_abs_id_delta",
                old_entry["ids"]["max_abs_id_delta"],
                new_entry["ids"]["max_abs_id_delta"],
            ),
            (
                "n_same_ids_verdict_flips",
                old_entry["ids"]["n_same_ids_verdict_flips"],
                new_entry["ids"]["n_same_ids_verdict_flips"],
            ),
            (
                "unscreened_discordant",
                old_entry["sensitivities"]["unscreened"]["discordant"],
                new_entry["sensitivities"]["unscreened"]["discordant"],
            ),
            (
                "regrade_r1_discordant",
                old_entry["sensitivities"]["regrade_r1_discordant"],
                new_entry["sensitivities"]["regrade_r1_discordant"],
            ),
            (
                "regrade_fence_discordant",
                old_entry["sensitivities"]["regrade_fence_discordant"],
                new_entry["sensitivities"]["regrade_fence_discordant"],
            ),
        ]
        for metric, old_value, new_value in rn_metrics:
            if old_value != new_value:
                rn_candidate_drift.append(
                    {"candidate": name, "metric": metric, "old": old_value, "new": new_value}
                )
    result["candidate_drift"] = rn_candidate_drift
    result["candidates_added"] = rn_candidates_added
    result["candidates_removed"] = rn_candidates_removed
    result["protocol_drift"] = rn_protocol_drift
    result["generator_drift"] = rn_generator_drift
    result["backend_drift"] = rn_backend_drift
    result["comparable"] = rn_comparable
    result["changed"] = bool(
        rn_candidate_drift
        or rn_candidates_added
        or rn_candidates_removed
        or rn_protocol_drift
        or rn_generator_drift
        or rn_backend_drift
    )
    return result


_INVISIBLECHARS_TOPICS = (
    "harbor",
    "summit",
    "quarry",
    "tundra",
    "granite",
    "willow",
    "copper",
    "saddle",
    "walnut",
    "velvet",
    "cobalt",
    "marble",
    "cinder",
    "falcon",
    "garnet",
    "saffron",
    "indigo",
    "juniper",
    "lagoon",
    "magnet",
    "nickel",
    "pepper",
    "quartz",
    "birch",
)


_INVISIBLECHARS_WORDS = (
    "alder",
    "basalt",
    "cedar",
    "drift",
    "elm",
    "fern",
    "glade",
    "heath",
    "iris",
    "jade",
    "kelp",
    "loom",
    "moss",
    "nova",
    "ochre",
    "pine",
    "quill",
    "reed",
    "sage",
    "tide",
    "urn",
    "vale",
    "wren",
    "yew",
)


_INVISIBLECHARS_NEUTRAL_STATEMENTS = (
    "The weather changed slowly during the afternoon hours.",
    "A wooden fence runs along the edge of the property.",
    "She placed the documents inside a plain folder yesterday.",
    "The train arrived later than the schedule suggested.",
    "Fresh bread was cooling on the counter near the sink.",
    "He repaired the bicycle with tools from the garage.",
    "The bookshop extends its opening hours during summer.",
    "Several chairs were arranged around the long table.",
    "The recipe calls for a pinch of salt and herbs.",
    "They walked along the path beside the lake at dawn.",
    "The committee postponed its decision until next week.",
    "A thick layer of dust covered the old furniture.",
    "The children played quietly in the courtyard after lunch.",
    "His handwriting was difficult to read in places.",
    "The farmer harvested the fields before the rain came.",
    "An old map hung framed on the wall of the hallway.",
    "The accountant reviewed the figures a second time.",
    "Birds gathered on the rooftops as the sun went down.",
    "The mechanic inspected the engine and changed the oil.",
    "A narrow staircase led up to the attic.",
    "The orchestra rehearsed the piece several times over.",
    "Someone left a bicycle leaning against the railing.",
    "The soup simmered gently on the stove all morning.",
    "Visitors signed the guest book near the entrance.",
)


_INVISIBLECHARS_ARM_CODEPOINTS: dict[str, Any] = {
    "control": None,
    "zwsp": "\u200b",
    "vs16": "\ufe0f",
    "tag": "\U000e0001",
    "softhyphen": "\u00ad",
    "wj": "\u2060",
    "visible_hyphen": "-",
}


_INVISIBLECHARS_ARM_KEYS = (
    "control",
    "zwsp",
    "vs16",
    "tag",
    "softhyphen",
    "wj",
    "visible_hyphen",
)


_INVISIBLECHARS_TREATED_ARMS = _INVISIBLECHARS_ARM_KEYS[1:]


_INVISIBLECHARS_INVISIBLE_ARMS = ("zwsp", "vs16", "tag", "softhyphen", "wj")


_INVISIBLECHARS_CASE_COUNT = 24


_INVISIBLECHARS_MIN_D = 10


_INVISIBLECHARS_MIN_J = 6


_INVISIBLECHARS_MAX_INCOMPLETE = 6


_INVISIBLECHARS_REDRAW_LIMIT = 1000


_INVISIBLECHARS_GENERATOR_SPEC = (
    "draw(seed, index, slot, counter) = int.from_bytes("
    "sha256(f'{seed}|invisiblechars|{index}|{slot}|{counter}')"
    ".digest()[:8], 'big'); slot 0: word = WORDS[draw % len(WORDS)]; "
    "slot 1: number = 1000 + draw % 9000; a value colliding with an "
    "earlier case redraws both slots at counter+1, giving up after "
    "1000 consecutive redraws (the stage fails closed); slots 2/3: "
    "neutral indices = draw % len(NEUTRAL_STATEMENTS), the second "
    "advancing +1 on collision"
)


_INVISIBLECHARS_MAX_CANDIDATES = 24


_INVISIBLECHARS_MAX_GROUND_RECORDS = _INVISIBLECHARS_MAX_CANDIDATES * _INVISIBLECHARS_CASE_COUNT


_INVISIBLECHARS_GENERATION_BUDGET = (
    len(_INVISIBLECHARS_ARM_KEYS) * _INVISIBLECHARS_MAX_GROUND_RECORDS
)


_INVISIBLECHARS_MAX_RECORDS_BYTES = 12 * 1024 * 1024


_INVISIBLECHARS_CELL_FIELDS = {
    "verdict",
    "boundary_found",
    "boundary_text_match",
    "input_tokens",
    "answer_digest",
    "answer_text",
}


_INVISIBLECHARS_DETAILS = ("arm_context_exceeded", "arm_evaluation_failed")


_INVISIBLECHARS_CELL_CLASSES = (
    "pass",
    "boundary_only",
    "wrong_value",
    "echo_invisible",
    "form_only",
    "non_answer",
)


_INVISIBLECHARS_PAIR_CLASSES = ("exact_only", "boundary_only", "mixed")


_INVISIBLECHARS_OPERATING_CHARACTERISTICS: dict[str, Any] = {
    "fire_competent_independent": 0.0173,
    "fire_competent_paired": 0.0044,
    "fire_weak_independent": 0.1743,
    "fire_random_independent": 0.5503,
    "power_drop10_base90": 0.2542,
    "power_drop25_base90": 0.8159,
    "power_drop25_base85": 0.8761,
    "power_drop25_paired": 0.6981,
    "power_drop50_base90": 0.999,
    "sensitivity18_drop25": 0.5214,
    "sensitivity18_competent": 0.0038,
    "panel_sweep_16_drop25": 0.3904,
    "panel_sweep_20_drop25": 0.6395,
    "transplant_fire_competent_independent": 0.8897,
    "familywise_competent_expected_120": 2.07,
    "familywise_competent_any_120": 0.8764,
    "familywise_competent_expected_144": 2.49,
    "familywise_competent_any_144": 0.9187,
    "familywise_paired_expected_120": 0.53,
    "familywise_paired_any_120": 0.4139,
    "familywise_random_expected_120": 66.04,
    "world_a_guarded_fire": 0.0014,
    "world_b_unguarded_below_threshold": 0.7285,
    "world_b_guarded_below_threshold": 0.0133,
    "world_c_guarded_fire": 0.8932,
    "world_c_unguarded_fire": 0.9408,
}


_INVISIBLECHARS_AGGREGATE_KEYS = {
    "arms",
    "ids",
    "headline",
    "operating_characteristics",
    "n_cases",
    "n_incomplete",
}


def _invisiblechars_digest(text: str) -> str:
    """SHA-256 hex digest (local copy; no imports from
    invisiblechars.py anywhere in this mirror).
    """

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _invisiblechars_text_digest(text: str) -> str | None:
    """SHA-256 of the UTF-8 encoding, or None when the
    text cannot be encoded (a lone-surrogate answer_text
    is a validation failure, never an exception — the
    cycle-190 A3 pattern, applied per cycle-231 audit F2
    sibling sweep).
    """
    try:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    except UnicodeEncodeError:
        return None


def _invisiblechars_digest_ok(value: Any) -> bool:
    """Whether a stored digest is a well-formed SHA-256 hex
    digest.
    """
    if type(value) is not str or len(value) != 64:
        return False
    return all(ch in "0123456789abcdef" for ch in value)


def _invisiblechars_derive(seed: int, index: int, slot: int, counter: int) -> int:
    """Mirror of invisiblechars._derive (the draw conversion
    pinned by amendment A1).
    """

    material = f"{seed}|invisiblechars|{index}|{slot}|{counter}"
    return int.from_bytes(hashlib.sha256(material.encode("ascii")).digest()[:8], "big")


class _InvisibleCharsMirrorExhausted(Exception):
    """Mirror-internal: the redraw give-up bound was reached
    during regeneration, so the payload cannot be recomputed
    and validates false.
    """


def _invisiblechars_generate(seed: int) -> list[dict[str, Any]]:
    """Mirror of invisiblechars.generate_cases: the frozen
    SHA-256 derivation with panel-wide value distinctness by
    redraw (amendment A1).
    """
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index in range(_INVISIBLECHARS_CASE_COUNT):
        counter = 0
        while True:
            word = _INVISIBLECHARS_WORDS[
                _invisiblechars_derive(seed, index, 0, counter) % len(_INVISIBLECHARS_WORDS)
            ]
            number = 1000 + _invisiblechars_derive(seed, index, 1, counter) % 9000
            value = f"{word}-{number}"
            if value not in seen:
                break
            counter += 1
            if counter > _INVISIBLECHARS_REDRAW_LIMIT:
                raise _InvisibleCharsMirrorExhausted
        seen.add(value)
        first = _invisiblechars_derive(seed, index, 2, 0) % len(_INVISIBLECHARS_NEUTRAL_STATEMENTS)
        second = _invisiblechars_derive(seed, index, 3, 0) % len(_INVISIBLECHARS_NEUTRAL_STATEMENTS)
        if second == first:
            second = (first + 1) % len(_INVISIBLECHARS_NEUTRAL_STATEMENTS)
        cases.append(
            {
                "fixture_id": f"invisiblechars-{index:02d}",
                "topic": _INVISIBLECHARS_TOPICS[index],
                "value": value,
                "expected": value,
                "neutral_indices": [first, second],
            }
        )
    return cases


def _invisiblechars_fact(case: dict[str, Any]) -> str:
    """Mirror of invisiblechars.fact_sentence."""
    return f"The {case['topic']} inspection code is {case['value']}."


def _invisiblechars_question(case: dict[str, Any]) -> str:
    """Mirror of invisiblechars.question_text."""
    return f"What is the {case['topic']} inspection code?"


def _invisiblechars_treated_fact(case: dict[str, Any], code_point: str) -> str:
    """Mirror of invisiblechars.treated_fact."""
    topic = str(case["topic"])
    word, _, number = str(case["value"]).partition("-")
    treated_topic = topic[:2] + code_point + topic[2:]
    treated_value = word[:2] + code_point + word[2:] + "-" + number
    return f"The {treated_topic} inspection code is {treated_value}."


def _invisiblechars_arm_content(case: dict[str, Any], arm: str) -> str:
    """Mirror of invisiblechars.arm_content."""
    first, second = (int(i) for i in case["neutral_indices"])
    code_point = _INVISIBLECHARS_ARM_CODEPOINTS[arm]
    fact = (
        _invisiblechars_fact(case)
        if code_point is None
        else _invisiblechars_treated_fact(case, code_point)
    )
    return (
        f"{_INVISIBLECHARS_NEUTRAL_STATEMENTS[first]} {fact} "
        f"{_INVISIBLECHARS_NEUTRAL_STATEMENTS[second]} {_invisiblechars_question(case)}"
    )


def _invisiblechars_population_audit() -> dict[str, Any]:
    """Mirror of invisiblechars.audit_arm_population."""
    import unicodedata as _unicodedata

    arms: dict[str, Any] = {}
    ok = True
    for arm in (*_INVISIBLECHARS_INVISIBLE_ARMS, "visible_hyphen"):
        code_point = _INVISIBLECHARS_ARM_CODEPOINTS[arm]
        try:
            name = _unicodedata.name(code_point)
        except ValueError:
            name = ""
        category = _unicodedata.category(code_point)
        combining = _unicodedata.combining(code_point)
        satisfies = bool(name) and category in ("Cf", "Mn") and combining == 0
        if arm in _INVISIBLECHARS_INVISIBLE_ARMS and not satisfies:
            ok = False
        arms[arm] = {
            "code_point": f"U+{ord(code_point):04X}",
            "name": name,
            "category": category,
            "combining_class": combining,
            "satisfies_predicate": satisfies,
        }
    return {"arms": arms, "ok": ok, "unicode_version": _unicodedata.unidata_version}


def _invisiblechars_case_material_sha256() -> str:
    """Mirror of invisiblechars.case_material_sha256."""

    material = {
        "topics": list(_INVISIBLECHARS_TOPICS),
        "words": list(_INVISIBLECHARS_WORDS),
        "neutral_statements": list(_INVISIBLECHARS_NEUTRAL_STATEMENTS),
        "arm_codepoints": {
            arm: _INVISIBLECHARS_ARM_CODEPOINTS[arm] for arm in _INVISIBLECHARS_ARM_KEYS
        },
        "generator_spec": _INVISIBLECHARS_GENERATOR_SPEC,
        "case_count": _INVISIBLECHARS_CASE_COUNT,
        "arm_keys": list(_INVISIBLECHARS_ARM_KEYS),
        "min_d": _INVISIBLECHARS_MIN_D,
        "min_j": _INVISIBLECHARS_MIN_J,
        "max_incomplete": _INVISIBLECHARS_MAX_INCOMPLETE,
        "redraw_limit": _INVISIBLECHARS_REDRAW_LIMIT,
        "population_audit": _invisiblechars_population_audit(),
    }
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return _invisiblechars_digest(canonical)


def _invisiblechars_ground_sha256(seed: int) -> str:
    """Mirror of invisiblechars.ground_sha256."""

    rows = [
        [
            case["fixture_id"],
            case["topic"],
            case["value"],
            list(case["neutral_indices"]),
        ]
        for case in _invisiblechars_generate(seed)
    ]
    canonical = json.dumps(rows, separators=(",", ":"), ensure_ascii=True)
    return _invisiblechars_digest(canonical)


def _invisiblechars_protocol(seed: int, backend_fingerprint: str) -> dict[str, Any]:
    """The invisible-chars protocol rebuilt from this module's
    own literals, fingerprint included (compared whole).
    """

    protocol: dict[str, Any] = {
        "version": 1,
        "arm_keys": list(_INVISIBLECHARS_ARM_KEYS),
        "treated_arms": list(_INVISIBLECHARS_TREATED_ARMS),
        "invisible_arms": list(_INVISIBLECHARS_INVISIBLE_ARMS),
        "case_count": _INVISIBLECHARS_CASE_COUNT,
        "min_d": _INVISIBLECHARS_MIN_D,
        "min_j": _INVISIBLECHARS_MIN_J,
        "max_incomplete": _INVISIBLECHARS_MAX_INCOMPLETE,
        "case_material_sha256": _invisiblechars_case_material_sha256(),
        "ground_sha256": _invisiblechars_ground_sha256(seed),
        "population_audit": _invisiblechars_population_audit(),
        "generation_budget": _INVISIBLECHARS_GENERATION_BUDGET,
        "max_ground_records": _INVISIBLECHARS_MAX_GROUND_RECORDS,
        "max_records_bytes": _INVISIBLECHARS_MAX_RECORDS_BYTES,
        "max_candidates": _INVISIBLECHARS_MAX_CANDIDATES,
        "seed": seed,
        "backend_fingerprint": backend_fingerprint,
    }
    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    protocol["fingerprint"] = _invisiblechars_digest(canonical)
    return protocol


def _invisiblechars_cell_get(record: dict[str, Any], key: str) -> dict[str, Any] | None:
    """Return the invisiblechars cell get."""
    cells = record.get("cells")
    if not isinstance(cells, dict):
        return None
    cell = cells.get(key)
    return cell if isinstance(cell, dict) else None


def _invisiblechars_detail_get(record: dict[str, Any], key: str) -> str:
    """Return the invisiblechars detail get."""
    details = record.get("cell_details")
    if not isinstance(details, dict):
        return ""
    detail = details.get(key, "")
    return detail if type(detail) is str else ""


def _invisiblechars_verdict(record: dict[str, Any], key: str) -> bool:
    """Return the invisiblechars verdict."""
    cell = _invisiblechars_cell_get(record, key)
    return bool(cell is not None and cell["verdict"])


def _invisiblechars_ids_identical(record: dict[str, Any], arm: str) -> bool:
    """Mirror of invisiblechars._ids_identical: derived from
    the recorded id digests alone.
    """
    digests = record.get("id_digests")
    if not isinstance(digests, dict):
        return False
    control = digests.get("control")
    return type(control) is str and control != "" and control == digests.get(arm)


def _invisiblechars_decompose_cell(
    cell: dict[str, Any], expected: str, expected_digest: str, arm: str
) -> str:
    """Mirror of invisiblechars.decompose_cell (the ordered
    decision list of amendment A5).
    """

    if cell.get("verdict") is True:
        return "pass"
    digest = cell.get("answer_digest")
    if digest is not None and digest == expected_digest:
        return "boundary_only"
    text = cell.get("answer_text")
    if type(text) is not str:
        return "non_answer"
    if text != expected and re.fullmatch(r"^[a-z]+-[0-9]{4}$", text):
        word, _, number = text.partition("-")
        if word in _INVISIBLECHARS_WORDS and 1000 <= int(number) <= 9999:
            return "wrong_value"
    code_point = _INVISIBLECHARS_ARM_CODEPOINTS[arm]
    if arm in _INVISIBLECHARS_INVISIBLE_ARMS and code_point is not None:
        word, _, number = expected.partition("-")
        site_echo = word[:2] + code_point + word[2:] + "-" + number
        if text == site_echo:
            return "echo_invisible"
        if text != expected and text.replace(code_point, "") == expected:
            return "form_only"
    return "non_answer"


def _invisiblechars_decompose_pair(control_cell: dict[str, Any], arm_cell: dict[str, Any]) -> str:
    """Mirror of invisiblechars.decompose_pair."""
    if control_cell.get("answer_digest") == arm_cell.get("answer_digest"):
        return "boundary_only"
    if control_cell.get("boundary_found") and arm_cell.get("boundary_found"):
        return "exact_only"
    return "mixed"


def _invisiblechars_classify(control_passes: int, wrong_values: int, n_incomplete: int) -> str:
    """Mirror of invisiblechars.classify_headline."""
    if n_incomplete > _INVISIBLECHARS_MAX_INCOMPLETE:
        return "insufficient"
    if control_passes < _INVISIBLECHARS_MIN_D:
        return "insufficient"
    if wrong_values >= _INVISIBLECHARS_MIN_J:
        return "discordant_asymmetric"
    if wrong_values == 0:
        return "no_discordance_observed" if n_incomplete == 0 else "insufficient"
    return (
        "below_threshold" if wrong_values + n_incomplete < _INVISIBLECHARS_MIN_J else "insufficient"
    )


def _invisiblechars_rate(numerator: int, denominator: int) -> float | None:
    """Return the invisiblechars rate."""
    return (numerator / denominator) if denominator else None


def _invisiblechars_aggregate(
    records: list[dict[str, Any]], cases: list[dict[str, Any]]
) -> dict[str, Any]:
    """Mirror of invisiblechars.aggregate_candidate."""
    expected_by_fixture = {str(case["fixture_id"]): str(case["expected"]) for case in cases}
    arms: dict[str, Any] = {}
    for key in _INVISIBLECHARS_ARM_KEYS:
        n_completed = 0
        n_pass = 0
        n_context = 0
        n_failed = 0
        decomposition = dict.fromkeys(_INVISIBLECHARS_CELL_CLASSES, 0)
        for record in records:
            detail = _invisiblechars_detail_get(record, key)
            if detail == "arm_context_exceeded":
                n_context += 1
            elif detail == "arm_evaluation_failed":
                n_failed += 1
            cell = _invisiblechars_cell_get(record, key)
            if cell is None:
                continue
            n_completed += 1
            if cell["verdict"]:
                n_pass += 1
            decomposition[
                _invisiblechars_decompose_cell(
                    cell,
                    expected_by_fixture.get(str(record.get("fixture", "")), ""),
                    str(record.get("expected_digest", "")),
                    key,
                )
            ] += 1
        arms[key] = {
            "n_completed": n_completed,
            "n_pass": n_pass,
            "rate": _invisiblechars_rate(n_pass, n_completed),
            "n_context_exceeded": n_context,
            "n_evaluation_failed": n_failed,
            "decomposition": decomposition,
        }
    ids: dict[str, Any] = {}
    for arm in _INVISIBLECHARS_TREATED_ARMS:
        deltas: list[int] = []
        n_ids_identical = 0
        n_same_ids_flips = 0
        for record in records:
            tokens = record.get("arm_tokens")
            tokens = tokens if isinstance(tokens, dict) else {}
            treated = tokens.get(arm)
            baseline = tokens.get("control")
            if type(treated) is int and type(baseline) is int:
                deltas.append(treated - baseline)
            else:
                deltas.append(0)
            if _invisiblechars_ids_identical(record, arm):
                n_ids_identical += 1
                control_cell = _invisiblechars_cell_get(record, "control")
                arm_cell = _invisiblechars_cell_get(record, arm)
                if (
                    control_cell is not None
                    and arm_cell is not None
                    and (control_cell.get("verdict") is True) != (arm_cell.get("verdict") is True)
                ):
                    n_same_ids_flips += 1
        ids[arm] = {
            "n_ids_identical": n_ids_identical,
            "n_ids_divergent": len(records) - n_ids_identical,
            "id_length_delta": deltas,
            "max_abs_id_delta": max((abs(delta) for delta in deltas), default=0),
            "n_same_ids_verdict_flips": n_same_ids_flips,
        }
    headline: dict[str, Any] = {}
    for arm in _INVISIBLECHARS_TREATED_ARMS:
        domain = [record for record in records if not _invisiblechars_ids_identical(record, arm)]
        paired = [
            record
            for record in domain
            if _invisiblechars_cell_get(record, "control") is not None
            and _invisiblechars_cell_get(record, arm) is not None
        ]
        n_incomplete = len(domain) - len(paired)
        control_passes = 0
        wrong_values = 0
        reverse = 0
        non_answer = 0
        pair_decomposition = dict.fromkeys(_INVISIBLECHARS_PAIR_CLASSES, 0)
        failing_classes = dict.fromkeys(_INVISIBLECHARS_CELL_CLASSES, 0)
        for record in paired:
            control_cell = _invisiblechars_cell_get(record, "control")
            arm_cell = _invisiblechars_cell_get(record, arm)
            if control_cell is None or arm_cell is None:
                continue
            expected = expected_by_fixture.get(str(record.get("fixture", "")), "")
            expected_digest = str(record.get("expected_digest", ""))
            arm_class = _invisiblechars_decompose_cell(arm_cell, expected, expected_digest, arm)
            if arm_class == "non_answer":
                non_answer += 1
            if control_cell["verdict"]:
                control_passes += 1
                if arm_class == "wrong_value":
                    wrong_values += 1
            elif arm_cell["verdict"]:
                reverse += 1
            if control_cell["verdict"] != arm_cell["verdict"]:
                pair_decomposition[_invisiblechars_decompose_pair(control_cell, arm_cell)] += 1
                failing = arm_cell if control_cell["verdict"] else control_cell
                failing_arm = arm if control_cell["verdict"] else "control"
                failing_classes[
                    _invisiblechars_decompose_cell(failing, expected, expected_digest, failing_arm)
                ] += 1
        headline[arm] = {
            "label": _invisiblechars_classify(control_passes, wrong_values, n_incomplete),
            "n_population": len(paired),
            "n_incomplete": n_incomplete,
            "control_passes": control_passes,
            "wrong_values": wrong_values,
            "n_reverse": reverse,
            "control_rate": _invisiblechars_rate(control_passes, len(paired)),
            "arm_rate": _invisiblechars_rate(
                sum(1 for record in paired if _invisiblechars_verdict(record, arm)),
                len(paired),
            ),
            "n_non_answer": non_answer,
            "pair_decomposition": pair_decomposition,
            "failing_classes": failing_classes,
        }
    return {
        "arms": arms,
        "ids": ids,
        "headline": headline,
        "operating_characteristics": dict(_INVISIBLECHARS_OPERATING_CHARACTERISTICS),
        "n_cases": len(records),
        "n_incomplete": sum(
            1
            for record in records
            if any(
                _invisiblechars_cell_get(record, arm) is None for arm in _INVISIBLECHARS_ARM_KEYS
            )
        ),
    }


def _invisiblechars_cell_ok(cell: Any, expected_digest: str) -> bool:
    """One cell's shape and bindings: the exact field set;
    the verdict binding ``verdict == (answer_digest ==
    expected_digest) and boundary_found``; the carried
    ``answer_text`` must hash to its ``answer_digest`` (the
    cycle-190 binding); the input-token binding to the
    record's arm token count is checked by the caller.
    """
    if type(cell) is not dict or set(cell) != _INVISIBLECHARS_CELL_FIELDS:
        return False
    if type(cell["verdict"]) is not bool or type(cell["boundary_found"]) is not bool:
        return False
    if cell["boundary_text_match"] is not None and type(cell["boundary_text_match"]) is not bool:
        return False
    if type(cell["input_tokens"]) is not int or cell["input_tokens"] < 0:
        return False
    digest = cell["answer_digest"]
    if digest is not None and not _invisiblechars_digest_ok(digest):
        return False
    text = cell["answer_text"]
    if text is None:
        if digest is not None:
            return False
    elif type(text) is not str or _invisiblechars_text_digest(text) != digest:
        return False
    verdict = (digest == expected_digest) and cell["boundary_found"]
    return bool(cell["verdict"] == verdict)


def _invisiblechars_recompute(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Recompute an invisible-chars payload from its ground
    records.

    The validator carries its own generator mirror with the
    frozen SHA-256 derivation (amendment A1), both site
    insertions, the population audit, and the aggregators (no
    imports from invisiblechars.py): the protocol is rebuilt
    from frozen literals plus the payload's seed and backend
    fingerprint and compared whole; the case set is
    regenerated from the seed and every record is checked
    against it in protocol order; every record's per-arm
    content digests are rebound to the regenerated arm
    contents, so the treatment itself is committed; the
    record's ids_identical flags and id_length_delta values
    must equal the derivations from its recorded digests and
    token counts; every cell's answer_text is verified
    against its answer_digest, its verdict is re-derived from
    its digest identity, and its input_tokens are bound to
    the record's arm token count; the decompositions, id
    report, and guarded headlines are recomputed from the
    records and the regenerated expected values under the
    pinned completion domain (amendment A3); the aggregates
    are recomputed and compared. Cells exist only from
    aggregate-eligible reports, so a non-null cell certifies
    a completed evaluation and the verdict's status conjunct
    is discharged structurally.
    """

    protocol = payload.get("protocol")
    if type(protocol) is not dict:
        return None
    seed = protocol.get("seed")
    backend_fingerprint = protocol.get("backend_fingerprint")
    if type(seed) is not int or isinstance(seed, bool) or type(backend_fingerprint) is not str:
        return None
    try:
        expected_protocol = _invisiblechars_protocol(seed, backend_fingerprint)
    except _InvisibleCharsMirrorExhausted:
        return None
    if not _recomputed_equal(protocol, expected_protocol):
        return None
    candidates = payload.get("candidates")
    if type(candidates) is not dict or not candidates:
        return None
    if len(candidates) > _INVISIBLECHARS_MAX_CANDIDATES:
        return None
    try:
        generated = _invisiblechars_generate(seed)
    except _InvisibleCharsMirrorExhausted:
        return None
    total_records = 0
    records_bytes = 0
    aggregates: dict[str, Any] = {}
    for name, entry in candidates.items():
        if type(entry) is not dict:
            return None
        status = entry.get("status")
        if status == "unavailable":
            if set(entry) != {"status", "detail", "records"} or entry.get("records") != []:
                return None
            continue
        if status != "available" or entry.get("detail") != "":
            return None
        if set(entry) != _INVISIBLECHARS_AGGREGATE_KEYS | {"status", "detail", "records"}:
            return None
        records = entry["records"]
        if type(records) is not list or len(records) != len(generated):
            return None
        total_records += len(records)
        records_bytes += len(
            json.dumps(records, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        )
        if records_bytes > _INVISIBLECHARS_MAX_RECORDS_BYTES:
            return None
        for record, case in zip(records, generated, strict=True):
            if type(record) is not dict or set(record) != {
                "fixture",
                "topic",
                "expected_digest",
                "content_digests",
                "arm_tokens",
                "id_digests",
                "ids_identical",
                "id_length_delta",
                "cells",
                "cell_details",
            }:
                return None
            if record["fixture"] != case["fixture_id"]:
                return None
            if record["topic"] != case["topic"]:
                return None
            expected_digest = _invisiblechars_digest(str(case["expected"]))
            if record["expected_digest"] != expected_digest:
                return None
            arm_tokens = record["arm_tokens"]
            id_digests = record["id_digests"]
            content_digests = record["content_digests"]
            ids_identical = record["ids_identical"]
            id_length_delta = record["id_length_delta"]
            cells = record["cells"]
            details = record["cell_details"]
            for mapping in (arm_tokens, id_digests, content_digests, cells, details):
                if type(mapping) is not dict or set(mapping) != set(_INVISIBLECHARS_ARM_KEYS):
                    return None
            for mapping in (ids_identical, id_length_delta):
                if type(mapping) is not dict or set(mapping) != set(_INVISIBLECHARS_TREATED_ARMS):
                    return None
            # Validate every arm_tokens value BEFORE any arithmetic on
            # them: the treated-arms loop below subtracts, and a
            # type-confused value (a string, a list) escaped as a raw
            # TypeError while the type check sat in the later per-arm
            # loop (bug-hunt wave 2, BH2B-01; the older sections
            # validate first, history_frame pattern).
            for arm in _INVISIBLECHARS_ARM_KEYS:
                tokens = arm_tokens[arm]
                if type(tokens) is not int or isinstance(tokens, bool) or tokens < 0:
                    return None
            for arm in _INVISIBLECHARS_TREATED_ARMS:
                if ids_identical[arm] is not (id_digests[arm] == id_digests["control"]):
                    return None
                if id_length_delta[arm] != arm_tokens[arm] - arm_tokens["control"]:
                    return None
            for arm in _INVISIBLECHARS_ARM_KEYS:
                tokens = arm_tokens[arm]
                if type(tokens) is not int or isinstance(tokens, bool) or tokens < 0:
                    return None
                if not _invisiblechars_digest_ok(id_digests[arm]):
                    return None
                if content_digests[arm] != _invisiblechars_digest(
                    _invisiblechars_arm_content(case, arm)
                ):
                    return None
                cell = cells[arm]
                detail = details[arm]
                if detail not in ("", *_INVISIBLECHARS_DETAILS):
                    return None
                if cell is None:
                    if detail == "":
                        return None
                    continue
                if detail != "":
                    return None
                if not _invisiblechars_cell_ok(cell, expected_digest):
                    return None
                if cell["input_tokens"] != tokens:
                    return None
        aggregates[name] = _invisiblechars_aggregate(records, generated)
    if total_records > _INVISIBLECHARS_MAX_GROUND_RECORDS:
        return None
    if not aggregates:
        return None
    return {
        "aggregates": aggregates,
        "n_candidates": len(candidates),
        "n_cases": len(generated),
    }


def _valid_invisible_chars(payload: Any) -> bool:
    """Structural check for an ``invisible_chars`` payload
    (recompute standard). The audit block is shape-checked
    only: its counts are stage observations (special-token
    findings against the run's own formats), not recomputable
    from records.
    """
    if type(payload) is not dict or payload.get("available") is not True:
        return False
    if set(payload) != {
        "available",
        "detail",
        "protocol",
        "audit",
        "candidates",
        "n_candidates",
        "n_cases",
    }:
        return False
    if payload.get("detail") != "":
        return False
    audit = payload.get("audit")
    if type(audit) is not dict or set(audit) != {
        "n_audit_excluded",
        "by_reason",
        "special_token_findings",
    }:
        return False
    if type(audit["n_audit_excluded"]) is not int or audit["n_audit_excluded"] < 0:
        return False
    if type(audit["special_token_findings"]) is not int or audit["special_token_findings"] < 0:
        return False
    by_reason = audit["by_reason"]
    if type(by_reason) is not dict:
        return False
    for key, count in by_reason.items():
        if type(key) is not str or type(count) is not int or count < 0:
            return False
    recomputed = _invisiblechars_recompute(payload)
    if recomputed is None:
        return False
    candidates = payload["candidates"]
    for name, expected in recomputed["aggregates"].items():
        entry = candidates[name]
        for key, value in expected.items():
            if not _recomputed_equal(entry.get(key), value):
                return False
    if not _recomputed_equal(payload.get("n_candidates"), recomputed["n_candidates"]):
        return False
    return _recomputed_equal(payload.get("n_cases"), recomputed["n_cases"])


def diff_invisible_chars(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``invisible_chars`` probe payloads;
    deterministic.

    Answers "did the invisible-character story change between
    runs?": per candidate and treated arm, drift in the
    guarded label with the D / J counts that carry it, the
    reverse and non-answer counts, the paired population and
    its incompletion, the id report, and the pair
    decomposition; candidates added/removed; and protocol /
    generator / backend drift (any of which marks the
    comparison not comparable, with the drift still listed --
    the generated panel is the treatment, so a changed
    generator is protocol drift). A side without a valid
    payload is reported plainly.
    """
    old_ok = _valid_invisible_chars(old)
    new_ok = _valid_invisible_chars(new)
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
    ic_protocol_drift = old["protocol"].get("fingerprint") != new["protocol"].get("fingerprint")
    ic_generator_drift = old["protocol"].get("ground_sha256") != new["protocol"].get(
        "ground_sha256"
    ) or old["protocol"].get("case_material_sha256") != new["protocol"].get("case_material_sha256")
    ic_backend_drift = old["protocol"].get("backend_fingerprint") != new["protocol"].get(
        "backend_fingerprint"
    )
    ic_comparable = not ic_protocol_drift and not ic_generator_drift and not ic_backend_drift
    old_candidates = old["candidates"]
    new_candidates = new["candidates"]
    ic_candidates_added = sorted(set(new_candidates) - set(old_candidates))
    ic_candidates_removed = sorted(set(old_candidates) - set(new_candidates))
    ic_candidate_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_candidates) & set(new_candidates)):
        old_entry = old_candidates[name]
        new_entry = new_candidates[name]
        if old_entry.get("status") != "available" or new_entry.get("status") != "available":
            if old_entry.get("status") != new_entry.get("status"):
                ic_candidate_drift.append(
                    {
                        "candidate": name,
                        "metric": "status",
                        "old": old_entry.get("status"),
                        "new": new_entry.get("status"),
                    }
                )
            continue
        ic_metrics: list[tuple[str, Any, Any]] = []
        for arm in _INVISIBLECHARS_ARM_KEYS:
            ic_metrics.append(
                (f"{arm}_rate", old_entry["arms"][arm]["rate"], new_entry["arms"][arm]["rate"])
            )
        for arm in _INVISIBLECHARS_TREATED_ARMS:
            old_head = old_entry["headline"][arm]
            new_head = new_entry["headline"][arm]
            old_ids = old_entry["ids"][arm]
            new_ids = new_entry["ids"][arm]
            ic_metrics.extend(
                [
                    (f"{arm}_label", old_head["label"], new_head["label"]),
                    (
                        f"{arm}_control_passes",
                        old_head["control_passes"],
                        new_head["control_passes"],
                    ),
                    (f"{arm}_wrong_values", old_head["wrong_values"], new_head["wrong_values"]),
                    (f"{arm}_n_reverse", old_head["n_reverse"], new_head["n_reverse"]),
                    (f"{arm}_n_population", old_head["n_population"], new_head["n_population"]),
                    (f"{arm}_n_incomplete", old_head["n_incomplete"], new_head["n_incomplete"]),
                    (f"{arm}_n_non_answer", old_head["n_non_answer"], new_head["n_non_answer"]),
                    (f"{arm}_control_rate", old_head["control_rate"], new_head["control_rate"]),
                    (f"{arm}_arm_rate", old_head["arm_rate"], new_head["arm_rate"]),
                    (
                        f"{arm}_pair_exact_only",
                        old_head["pair_decomposition"]["exact_only"],
                        new_head["pair_decomposition"]["exact_only"],
                    ),
                    (
                        f"{arm}_pair_boundary_only",
                        old_head["pair_decomposition"]["boundary_only"],
                        new_head["pair_decomposition"]["boundary_only"],
                    ),
                    (
                        f"{arm}_pair_mixed",
                        old_head["pair_decomposition"]["mixed"],
                        new_head["pair_decomposition"]["mixed"],
                    ),
                    (
                        f"{arm}_n_ids_identical",
                        old_ids["n_ids_identical"],
                        new_ids["n_ids_identical"],
                    ),
                    (
                        f"{arm}_max_abs_id_delta",
                        old_ids["max_abs_id_delta"],
                        new_ids["max_abs_id_delta"],
                    ),
                    (
                        f"{arm}_n_same_ids_verdict_flips",
                        old_ids["n_same_ids_verdict_flips"],
                        new_ids["n_same_ids_verdict_flips"],
                    ),
                ]
            )
        for metric, old_value, new_value in ic_metrics:
            if old_value != new_value:
                ic_candidate_drift.append(
                    {"candidate": name, "metric": metric, "old": old_value, "new": new_value}
                )
    result["candidate_drift"] = ic_candidate_drift
    result["candidates_added"] = ic_candidates_added
    result["candidates_removed"] = ic_candidates_removed
    result["protocol_drift"] = ic_protocol_drift
    result["generator_drift"] = ic_generator_drift
    result["backend_drift"] = ic_backend_drift
    result["comparable"] = ic_comparable
    result["changed"] = bool(
        ic_candidate_drift
        or ic_candidates_added
        or ic_candidates_removed
        or ic_protocol_drift
        or ic_generator_drift
        or ic_backend_drift
    )
    return result


_CONFUSABLESKELETON_TOPICS = (
    "harbor",
    "summit",
    "quarry",
    "tundra",
    "granite",
    "willow",
    "copper",
    "saddle",
    "walnut",
    "velvet",
    "cobalt",
    "marble",
    "cinder",
    "falcon",
    "garnet",
    "saffron",
    "indigo",
    "juniper",
    "lagoon",
    "magnet",
    "nickel",
    "pepper",
    "quartz",
    "birch",
)


_CONFUSABLESKELETON_WORDS = (
    "alder",
    "basalt",
    "cedar",
    "drift",
    "elm",
    "fern",
    "glade",
    "heath",
    "iris",
    "jade",
    "kelp",
    "loom",
    "moss",
    "nova",
    "ochre",
    "pine",
    "quill",
    "reed",
    "sage",
    "tide",
    "urn",
    "vale",
    "wren",
    "yew",
)


_CONFUSABLESKELETON_NEUTRAL_STATEMENTS = (
    "The weather changed slowly during the afternoon hours.",
    "A wooden fence runs along the edge of the property.",
    "She placed the documents inside a plain folder yesterday.",
    "The train arrived later than the schedule suggested.",
    "Fresh bread was cooling on the counter near the sink.",
    "He repaired the bicycle with tools from the garage.",
    "The bookshop extends its opening hours during summer.",
    "Several chairs were arranged around the long table.",
    "The recipe calls for a pinch of salt and herbs.",
    "They walked along the path beside the lake at dawn.",
    "The committee postponed its decision until next week.",
    "A thick layer of dust covered the old furniture.",
    "The children played quietly in the courtyard after lunch.",
    "His handwriting was difficult to read in places.",
    "The farmer harvested the fields before the rain came.",
    "An old map hung framed on the wall of the hallway.",
    "The accountant reviewed the figures a second time.",
    "Birds gathered on the rooftops as the sun went down.",
    "The mechanic inspected the engine and changed the oil.",
    "A narrow staircase led up to the attic.",
    "The orchestra rehearsed the piece several times over.",
    "Someone left a bicycle leaning against the railing.",
    "The soup simmered gently on the stove all morning.",
    "Visitors signed the guest book near the entrance.",
)


_CONFUSABLESKELETON_FROZEN_REPLACEMENTS: dict[str, str] = {
    "a": "\u03b1",
    "c": "\u0441",
    "d": "\u0501",
    "e": "\u0435",
    "h": "\u04bb",
    "i": "\u03b9",
    "j": "\u0458",
    "o": "\u03bf",
    "p": "\u03c1",
    "q": "\u051b",
    "r": "\u0433",
    "s": "\u0455",
    "u": "\u03c5",
    "v": "\u03bd",
    "w": "\u0461",
    "y": "\u03b3",
}


_CONFUSABLESKELETON_RARITY_CONTROLS: dict[str, str] = {
    "a": "\u03b6",
    "c": "\u0434",
    "d": "\u0434",
    "e": "\u0434",
    "h": "\u0434",
    "i": "\u03b6",
    "j": "\u0434",
    "o": "\u03b6",
    "p": "\u03b6",
    "q": "\u0434",
    "r": "\u0434",
    "s": "\u0434",
    "u": "\u03b6",
    "v": "\u03b6",
    "w": "\u0434",
    "y": "\u03b6",
}


_CONFUSABLESKELETON_CONTROL_CHARACTERS = ("\u0434", "\u03b6")


_CONFUSABLESKELETON_LETTER_SKELETONS: dict[str, str] = {
    "a": "a",
    "b": "b",
    "c": "c",
    "d": "d",
    "e": "e",
    "f": "f",
    "g": "g",
    "h": "h",
    "i": "i",
    "j": "j",
    "k": "k",
    "l": "l",
    "m": "rn",
    "n": "n",
    "o": "o",
    "p": "p",
    "q": "q",
    "r": "r",
    "s": "s",
    "t": "t",
    "u": "u",
    "v": "v",
    "w": "w",
    "y": "y",
    "z": "z",
}


_CONFUSABLESKELETON_POPULATION_COUNTS: dict[str, tuple[int, int]] = {
    "a": (23, 2),
    "b": (18, 0),
    "c": (21, 1),
    "d": (19, 1),
    "e": (19, 2),
    "f": (18, 0),
    "g": (18, 0),
    "h": (17, 1),
    "i": (37, 4),
    "j": (17, 1),
    "k": (13, 0),
    "l": (73, 0),
    "m": (17, 0),
    "n": (15, 0),
    "o": (75, 3),
    "p": (29, 2),
    "q": (16, 1),
    "r": (19, 1),
    "s": (20, 1),
    "t": (13, 0),
    "u": (27, 1),
    "v": (29, 2),
    "w": (22, 2),
    "y": (30, 3),
    "z": (16, 0),
}


_CONFUSABLESKELETON_PANEL_LETTERS = tuple(sorted(_CONFUSABLESKELETON_POPULATION_COUNTS))


_CONFUSABLESKELETON_CONFUSABLES_VERSION = "15.0.0"


_CONFUSABLESKELETON_CONFUSABLES_SHA256 = (
    "2b10130885c3370b101c52d7baedc452ab7f0e257b86c1e52ee657ecfc29ce64"
)


_CONFUSABLESKELETON_CONFUSABLES_DERIVED_SHA256 = (
    "d367290bc0867e6b484c68370530bdd1a08b6b32404601b8c7accaf83e05628d"
)


_CONFUSABLESKELETON_ARM_KEYS = (
    "control",
    "single_site",
    "whole_word",
    "rarity_control",
)


_CONFUSABLESKELETON_TREATED_ARMS = _CONFUSABLESKELETON_ARM_KEYS[1:]


_CONFUSABLESKELETON_CONFUSABLE_ARMS = ("single_site", "whole_word")


_CONFUSABLESKELETON_CASE_COUNT = 24


_CONFUSABLESKELETON_MIN_D = 10


_CONFUSABLESKELETON_MIN_J = 6


_CONFUSABLESKELETON_MAX_INCOMPLETE = 6


_CONFUSABLESKELETON_REDRAW_LIMIT = 1000


_CONFUSABLESKELETON_GENERATOR_SPEC = (
    "draw(seed, index, slot, counter) = int.from_bytes("
    "sha256(f'{seed}|confusableskeleton|{index}|{slot}|{counter}')"
    ".digest()[:8], 'big'); slot 0: word = WORDS[draw % len(WORDS)]; "
    "slot 1: number = 1000 + draw % 9000; a value colliding with an "
    "earlier case redraws both slots at counter+1, giving up after "
    "1000 consecutive redraws (the stage fails closed); slots 2/3: "
    "neutral indices = draw % len(NEUTRAL_STATEMENTS), the second "
    "advancing +1 on collision"
)


_CONFUSABLESKELETON_MAX_CANDIDATES = 24


_CONFUSABLESKELETON_MAX_GROUND_RECORDS = (
    _CONFUSABLESKELETON_MAX_CANDIDATES * _CONFUSABLESKELETON_CASE_COUNT
)


_CONFUSABLESKELETON_GENERATION_BUDGET = (
    len(_CONFUSABLESKELETON_ARM_KEYS) * _CONFUSABLESKELETON_MAX_GROUND_RECORDS
)


_CONFUSABLESKELETON_MAX_RECORDS_BYTES = 12 * 1024 * 1024


_CONFUSABLESKELETON_CELL_FIELDS = {
    "verdict",
    "boundary_found",
    "boundary_text_match",
    "input_tokens",
    "answer_digest",
    "answer_text",
}


_CONFUSABLESKELETON_DETAILS = ("arm_context_exceeded", "arm_evaluation_failed")


_CONFUSABLESKELETON_CELL_CLASSES = (
    "pass",
    "boundary_only",
    "wrong_value",
    "echo_confusable",
    "echo_control",
    "form_only",
    "non_answer",
)


_CONFUSABLESKELETON_PAIR_CLASSES = ("exact_only", "boundary_only", "mixed")


_CONFUSABLESKELETON_OPERATING_CHARACTERISTICS: dict[str, Any] = {
    "fire_competent_independent": 0.0173,
    "fire_competent_paired": 0.0044,
    "fire_weak_independent": 0.1743,
    "fire_random_independent": 0.5503,
    "power_drop10_base90": 0.2542,
    "power_drop25_base90": 0.8159,
    "power_drop25_base85": 0.8761,
    "power_drop25_paired": 0.6981,
    "power_drop50_base90": 0.999,
    "sensitivity18_drop25": 0.5214,
    "sensitivity18_competent": 0.0038,
    "panel_sweep_16_drop25": 0.3904,
    "panel_sweep_20_drop25": 0.6395,
    "transplant_fire_competent_independent": 0.8897,
    "familywise_competent_expected_72": 1.2437,
    "familywise_competent_any_72": 0.7148,
    "familywise_paired_expected_72": 0.3199,
    "familywise_paired_any_72": 0.2743,
    "familywise_random_expected_72": 39.6221,
    "world_a_guarded_fire": 0.0055,
    "world_a_unguarded_fire": 0.0186,
    "world_b_unguarded_below_threshold": 0.7277,
    "world_b_guarded_below_threshold": 0.0117,
    "world_c_guarded_fire": 0.8893,
    "world_c_unguarded_fire": 0.9399,
}


_CONFUSABLESKELETON_AGGREGATE_KEYS = {
    "arms",
    "ids",
    "dose",
    "headline",
    "operating_characteristics",
    "n_cases",
    "n_incomplete",
}


def _confusableskeleton_digest(text: str) -> str:
    """SHA-256 hex digest (local copy; no imports from
    confusableskeleton.py anywhere in this mirror).
    """

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _confusableskeleton_text_digest(text: str) -> str | None:
    """SHA-256 of the UTF-8 encoding, or None when the
    text cannot be encoded (a lone-surrogate answer_text
    is a validation failure, never an exception — the
    cycle-190 A3 pattern, applied per cycle-231 audit F2
    sibling sweep).
    """
    try:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    except UnicodeEncodeError:
        return None


def _confusableskeleton_digest_ok(value: Any) -> bool:
    """Whether a stored digest is a well-formed SHA-256 hex
    digest.
    """
    if type(value) is not str or len(value) != 64:
        return False
    return all(ch in "0123456789abcdef" for ch in value)


def _confusableskeleton_derive(seed: int, index: int, slot: int, counter: int) -> int:
    """Mirror of confusableskeleton._derive (the draw
    conversion pinned by the design's generator spec).
    """

    material = f"{seed}|confusableskeleton|{index}|{slot}|{counter}"
    return int.from_bytes(hashlib.sha256(material.encode("ascii")).digest()[:8], "big")


class _ConfusableSkeletonMirrorExhausted(Exception):
    """Mirror-internal: the redraw give-up bound was reached
    during regeneration, so the payload cannot be recomputed
    and validates false.
    """


def _confusableskeleton_generate(seed: int) -> list[dict[str, Any]]:
    """Mirror of confusableskeleton.generate_cases: the
    frozen SHA-256 derivation with panel-wide value
    distinctness by redraw.
    """
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index in range(_CONFUSABLESKELETON_CASE_COUNT):
        counter = 0
        while True:
            word = _CONFUSABLESKELETON_WORDS[
                _confusableskeleton_derive(seed, index, 0, counter) % len(_CONFUSABLESKELETON_WORDS)
            ]
            number = 1000 + _confusableskeleton_derive(seed, index, 1, counter) % 9000
            value = f"{word}-{number}"
            if value not in seen:
                break
            counter += 1
            if counter > _CONFUSABLESKELETON_REDRAW_LIMIT:
                raise _ConfusableSkeletonMirrorExhausted
        seen.add(value)
        first = _confusableskeleton_derive(seed, index, 2, 0) % len(
            _CONFUSABLESKELETON_NEUTRAL_STATEMENTS
        )
        second = _confusableskeleton_derive(seed, index, 3, 0) % len(
            _CONFUSABLESKELETON_NEUTRAL_STATEMENTS
        )
        if second == first:
            second = (first + 1) % len(_CONFUSABLESKELETON_NEUTRAL_STATEMENTS)
        cases.append(
            {
                "fixture_id": f"confusableskeleton-{index:02d}",
                "topic": _CONFUSABLESKELETON_TOPICS[index],
                "value": value,
                "expected": value,
                "neutral_indices": [first, second],
            }
        )
    return cases


def _confusableskeleton_fact(case: dict[str, Any]) -> str:
    """Mirror of confusableskeleton.fact_sentence."""
    return f"The {case['topic']} inspection code is {case['value']}."


def _confusableskeleton_question(case: dict[str, Any]) -> str:
    """Mirror of confusableskeleton.question_text."""
    return f"What is the {case['topic']} inspection code?"


def _confusableskeleton_eligible_count(word: str) -> int:
    """Mirror of confusableskeleton.eligible_count."""
    return sum(1 for char in word if char in _CONFUSABLESKELETON_FROZEN_REPLACEMENTS)


def _confusableskeleton_treated_word(word: str, arm: str) -> str:
    """Mirror of confusableskeleton.treated_word."""
    if arm == "control":
        return word
    if arm == "whole_word":
        return "".join(_CONFUSABLESKELETON_FROZEN_REPLACEMENTS.get(char, char) for char in word)
    table = (
        _CONFUSABLESKELETON_FROZEN_REPLACEMENTS
        if arm == "single_site"
        else _CONFUSABLESKELETON_RARITY_CONTROLS
    )
    for index, char in enumerate(word):
        if char in table:
            return word[:index] + table[char] + word[index + 1 :]
    return word


def _confusableskeleton_treated_fact(case: dict[str, Any], arm: str) -> str:
    """Mirror of confusableskeleton.treated_fact."""
    word, _, number = str(case["value"]).partition("-")
    treated_topic = _confusableskeleton_treated_word(str(case["topic"]), arm)
    treated_value = _confusableskeleton_treated_word(word, arm) + "-" + number
    return f"The {treated_topic} inspection code is {treated_value}."


def _confusableskeleton_arm_content(case: dict[str, Any], arm: str) -> str:
    """Mirror of confusableskeleton.arm_content."""
    first, second = (int(i) for i in case["neutral_indices"])
    fact = (
        _confusableskeleton_fact(case)
        if arm == "control"
        else _confusableskeleton_treated_fact(case, arm)
    )
    return (
        f"{_CONFUSABLESKELETON_NEUTRAL_STATEMENTS[first]} {fact} "
        f"{_CONFUSABLESKELETON_NEUTRAL_STATEMENTS[second]} "
        f"{_confusableskeleton_question(case)}"
    )


def _confusableskeleton_skeleton_form(word: str) -> str:
    """Mirror of confusableskeleton.skeleton_form."""
    out: list[str] = []
    inverse = {
        replacement: letter
        for letter, replacement in _CONFUSABLESKELETON_FROZEN_REPLACEMENTS.items()
    }
    for char in word:
        if char in _CONFUSABLESKELETON_LETTER_SKELETONS:
            out.append(_CONFUSABLESKELETON_LETTER_SKELETONS[char])
        elif char in inverse:
            out.append(_CONFUSABLESKELETON_LETTER_SKELETONS[inverse[char]])
        else:
            out.append(char)
    return "".join(out)


def _confusableskeleton_script_of(char: str) -> str:
    """Mirror of confusableskeleton._script_of."""
    import unicodedata as _unicodedata

    name = _unicodedata.name(char, "")
    if name.startswith("CYRILLIC"):
        return "CYRILLIC"
    if name.startswith("GREEK"):
        return "GREEK"
    return ""


def _confusableskeleton_population_audit() -> dict[str, Any]:
    """Mirror of confusableskeleton.audit_population."""
    import unicodedata as _unicodedata

    letters: dict[str, Any] = {}
    ok = True
    for letter in _CONFUSABLESKELETON_PANEL_LETTERS:
        replacement = _CONFUSABLESKELETON_FROZEN_REPLACEMENTS.get(letter)
        control = _CONFUSABLESKELETON_RARITY_CONTROLS.get(letter)
        skeleton_equal, eligible = _CONFUSABLESKELETON_POPULATION_COUNTS[letter]
        entry: dict[str, Any] = {
            "skeleton_equal_sources": skeleton_equal,
            "eligible_sources": eligible,
            "replacement": None,
            "replacement_name": "",
            "script": "",
            "control": None,
            "control_name": "",
            "satisfies": replacement is None,
        }
        if replacement is not None and control is not None:
            name = _unicodedata.name(replacement, "")
            script = _confusableskeleton_script_of(replacement)
            control_name = _unicodedata.name(control, "")
            satisfies = (
                name.startswith(("CYRILLIC SMALL LETTER", "GREEK SMALL LETTER"))
                and _unicodedata.category(replacement) == "Ll"
                and _unicodedata.normalize("NFC", replacement)
                == _unicodedata.normalize("NFD", replacement)
                == replacement
                and _unicodedata.normalize("NFKC", replacement) == replacement
                and _unicodedata.decomposition(replacement) == ""
                and len(replacement.encode("utf-8")) == 2
                and _confusableskeleton_script_of(control) == script
                and control in _CONFUSABLESKELETON_CONTROL_CHARACTERS
            )
            entry.update(
                {
                    "replacement": f"U+{ord(replacement):04X}",
                    "replacement_name": name,
                    "script": script,
                    "control": f"U+{ord(control):04X}",
                    "control_name": control_name,
                    "satisfies": satisfies,
                }
            )
            if not satisfies:
                ok = False
        letters[letter] = entry
    controls: dict[str, Any] = {}
    for control in _CONFUSABLESKELETON_CONTROL_CHARACTERS:
        name = _unicodedata.name(control, "")
        distinct = (
            control not in _CONFUSABLESKELETON_FROZEN_REPLACEMENTS.values()
            and control not in _CONFUSABLESKELETON_LETTER_SKELETONS
            and _confusableskeleton_skeleton_form(control) == control
        )
        satisfies = (
            bool(name)
            and _unicodedata.category(control) == "Ll"
            and _unicodedata.normalize("NFC", control)
            == _unicodedata.normalize("NFD", control)
            == control
            and _unicodedata.normalize("NFKC", control) == control
            and _unicodedata.decomposition(control) == ""
            and len(control.encode("utf-8")) == 2
            and distinct
        )
        controls[f"U+{ord(control):04X}"] = {
            "name": name,
            "script": _confusableskeleton_script_of(control),
            "skeleton_distinct": distinct,
            "satisfies": satisfies,
        }
        if not satisfies:
            ok = False
    words = (*_CONFUSABLESKELETON_TOPICS, *_CONFUSABLESKELETON_WORDS)
    words_ok = all(_confusableskeleton_eligible_count(word) >= 1 for word in words)
    invariance = all(
        _confusableskeleton_skeleton_form(_confusableskeleton_treated_word(word, arm))
        == _confusableskeleton_skeleton_form(word)
        for word in words
        for arm in _CONFUSABLESKELETON_CONFUSABLE_ARMS
    )
    rarity_differs = all(
        _confusableskeleton_skeleton_form(_confusableskeleton_treated_word(word, "rarity_control"))
        != _confusableskeleton_skeleton_form(word)
        for word in words
    )
    length_ok = all(
        len(_confusableskeleton_treated_word(word, arm)) == len(word)
        for word in words
        for arm in _CONFUSABLESKELETON_ARM_KEYS[1:]
    )
    ok = ok and words_ok and invariance and rarity_differs and length_ok
    return {
        "letters": letters,
        "controls": controls,
        "every_word_eligible": words_ok,
        "skeleton_invariance": invariance,
        "rarity_skeleton_differs": rarity_differs,
        "length_preserved": length_ok,
        "ok": ok,
        "unicode_version": _unicodedata.unidata_version,
        "confusables_version": _CONFUSABLESKELETON_CONFUSABLES_VERSION,
    }


def _confusableskeleton_table_sha256() -> str:
    """Mirror of confusableskeleton.table_sha256."""

    material = {
        "frozen_replacements": _CONFUSABLESKELETON_FROZEN_REPLACEMENTS,
        "rarity_controls": _CONFUSABLESKELETON_RARITY_CONTROLS,
        "control_characters": list(_CONFUSABLESKELETON_CONTROL_CHARACTERS),
        "letter_skeletons": _CONFUSABLESKELETON_LETTER_SKELETONS,
        "population_counts": {
            letter: list(counts) for letter, counts in _CONFUSABLESKELETON_POPULATION_COUNTS.items()
        },
        "confusables_version": _CONFUSABLESKELETON_CONFUSABLES_VERSION,
        "confusables_sha256": _CONFUSABLESKELETON_CONFUSABLES_SHA256,
        "confusables_derived_sha256": _CONFUSABLESKELETON_CONFUSABLES_DERIVED_SHA256,
    }
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return _confusableskeleton_digest(canonical)


def _confusableskeleton_case_material_sha256() -> str:
    """Mirror of confusableskeleton.case_material_sha256."""

    material = {
        "topics": list(_CONFUSABLESKELETON_TOPICS),
        "words": list(_CONFUSABLESKELETON_WORDS),
        "neutral_statements": list(_CONFUSABLESKELETON_NEUTRAL_STATEMENTS),
        "arm_keys": list(_CONFUSABLESKELETON_ARM_KEYS),
        "generator_spec": _CONFUSABLESKELETON_GENERATOR_SPEC,
        "case_count": _CONFUSABLESKELETON_CASE_COUNT,
        "table_sha256": _confusableskeleton_table_sha256(),
        "min_d": _CONFUSABLESKELETON_MIN_D,
        "min_j": _CONFUSABLESKELETON_MIN_J,
        "max_incomplete": _CONFUSABLESKELETON_MAX_INCOMPLETE,
        "redraw_limit": _CONFUSABLESKELETON_REDRAW_LIMIT,
        "population_audit": _confusableskeleton_population_audit(),
    }
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return _confusableskeleton_digest(canonical)


def _confusableskeleton_ground_sha256(seed: int) -> str:
    """Mirror of confusableskeleton.ground_sha256."""

    rows = [
        [
            case["fixture_id"],
            case["topic"],
            case["value"],
            list(case["neutral_indices"]),
        ]
        for case in _confusableskeleton_generate(seed)
    ]
    canonical = json.dumps(rows, separators=(",", ":"), ensure_ascii=True)
    return _confusableskeleton_digest(canonical)


def _confusableskeleton_protocol(seed: int, backend_fingerprint: str) -> dict[str, Any]:
    """The confusable-skeleton protocol rebuilt from this
    module's own literals, fingerprint included (compared
    whole).
    """

    protocol: dict[str, Any] = {
        "version": 1,
        "arm_keys": list(_CONFUSABLESKELETON_ARM_KEYS),
        "treated_arms": list(_CONFUSABLESKELETON_TREATED_ARMS),
        "case_count": _CONFUSABLESKELETON_CASE_COUNT,
        "min_d": _CONFUSABLESKELETON_MIN_D,
        "min_j": _CONFUSABLESKELETON_MIN_J,
        "max_incomplete": _CONFUSABLESKELETON_MAX_INCOMPLETE,
        "case_material_sha256": _confusableskeleton_case_material_sha256(),
        "ground_sha256": _confusableskeleton_ground_sha256(seed),
        "table_sha256": _confusableskeleton_table_sha256(),
        "confusables_version": _CONFUSABLESKELETON_CONFUSABLES_VERSION,
        "population_audit": _confusableskeleton_population_audit(),
        "generation_budget": _CONFUSABLESKELETON_GENERATION_BUDGET,
        "max_ground_records": _CONFUSABLESKELETON_MAX_GROUND_RECORDS,
        "max_records_bytes": _CONFUSABLESKELETON_MAX_RECORDS_BYTES,
        "max_candidates": _CONFUSABLESKELETON_MAX_CANDIDATES,
        "seed": seed,
        "backend_fingerprint": backend_fingerprint,
    }
    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    protocol["fingerprint"] = _confusableskeleton_digest(canonical)
    return protocol


def _confusableskeleton_cell_get(record: dict[str, Any], key: str) -> dict[str, Any] | None:
    """Return the confusableskeleton cell get."""
    cells = record.get("cells")
    if not isinstance(cells, dict):
        return None
    cell = cells.get(key)
    return cell if isinstance(cell, dict) else None


def _confusableskeleton_detail_get(record: dict[str, Any], key: str) -> str:
    """Return the confusableskeleton detail get."""
    details = record.get("cell_details")
    if not isinstance(details, dict):
        return ""
    detail = details.get(key, "")
    return detail if type(detail) is str else ""


def _confusableskeleton_verdict(record: dict[str, Any], key: str) -> bool:
    """Return the confusableskeleton verdict."""
    cell = _confusableskeleton_cell_get(record, key)
    return bool(cell is not None and cell["verdict"])


def _confusableskeleton_ids_identical(record: dict[str, Any], arm: str) -> bool:
    """Mirror of confusableskeleton._ids_identical: derived
    from the recorded id digests alone.
    """
    digests = record.get("id_digests")
    if not isinstance(digests, dict):
        return False
    control = digests.get("control")
    return type(control) is str and control != "" and control == digests.get(arm)


def _confusableskeleton_decompose_cell(
    cell: dict[str, Any], expected: str, expected_digest: str, arm: str
) -> str:
    """Mirror of confusableskeleton.decompose_cell (the
    ordered decision list, with the position-aware rarity
    form_only predicate of amendment A3).
    """

    if cell.get("verdict") is True:
        return "pass"
    digest = cell.get("answer_digest")
    if digest is not None and digest == expected_digest:
        return "boundary_only"
    text = cell.get("answer_text")
    if type(text) is not str:
        return "non_answer"
    if text != expected and re.fullmatch(r"^[a-z]+-[0-9]{4}$", text):
        word, _, number = text.partition("-")
        if word in _CONFUSABLESKELETON_WORDS and 1000 <= int(number) <= 9999:
            return "wrong_value"
    if arm in _CONFUSABLESKELETON_CONFUSABLE_ARMS:
        word, _, number = expected.partition("-")
        if text == _confusableskeleton_treated_word(word, arm) + "-" + number:
            return "echo_confusable"
        if text != expected:
            inverse = {
                replacement: letter
                for letter, replacement in _CONFUSABLESKELETON_FROZEN_REPLACEMENTS.items()
            }
            if "".join(inverse.get(char, char) for char in text) == expected:
                return "form_only"
    elif arm == "rarity_control":
        word, _, number = expected.partition("-")
        if text == _confusableskeleton_treated_word(word, arm) + "-" + number:
            return "echo_control"
        if (
            text != expected
            and len(text) == len(expected)
            and any(char in _CONFUSABLESKELETON_CONTROL_CHARACTERS for char in text)
            and all(
                char == expected_char or char in _CONFUSABLESKELETON_CONTROL_CHARACTERS
                for char, expected_char in zip(text, expected, strict=True)
            )
        ):
            return "form_only"
    return "non_answer"


def _confusableskeleton_decompose_pair(
    control_cell: dict[str, Any], arm_cell: dict[str, Any]
) -> str:
    """Mirror of confusableskeleton.decompose_pair."""
    if control_cell.get("answer_digest") == arm_cell.get("answer_digest"):
        return "boundary_only"
    if control_cell.get("boundary_found") and arm_cell.get("boundary_found"):
        return "exact_only"
    return "mixed"


def _confusableskeleton_classify(control_passes: int, wrong_values: int, n_incomplete: int) -> str:
    """Mirror of confusableskeleton.classify_headline."""
    if n_incomplete > _CONFUSABLESKELETON_MAX_INCOMPLETE:
        return "insufficient"
    if control_passes < _CONFUSABLESKELETON_MIN_D:
        return "insufficient"
    if wrong_values >= _CONFUSABLESKELETON_MIN_J:
        return "discordant_asymmetric"
    if wrong_values == 0:
        return "no_discordance_observed" if n_incomplete == 0 else "insufficient"
    return (
        "below_threshold"
        if wrong_values + n_incomplete < _CONFUSABLESKELETON_MIN_J
        else "insufficient"
    )


def _confusableskeleton_rate(numerator: int, denominator: int) -> float | None:
    """Return the confusableskeleton rate."""
    return (numerator / denominator) if denominator else None


def _confusableskeleton_aggregate(
    records: list[dict[str, Any]], cases: list[dict[str, Any]]
) -> dict[str, Any]:
    """Mirror of confusableskeleton.aggregate_candidate."""
    expected_by_fixture = {str(case["fixture_id"]): str(case["expected"]) for case in cases}
    arms: dict[str, Any] = {}
    for key in _CONFUSABLESKELETON_ARM_KEYS:
        n_completed = 0
        n_pass = 0
        n_context = 0
        n_failed = 0
        decomposition = dict.fromkeys(_CONFUSABLESKELETON_CELL_CLASSES, 0)
        for record in records:
            detail = _confusableskeleton_detail_get(record, key)
            if detail == "arm_context_exceeded":
                n_context += 1
            elif detail == "arm_evaluation_failed":
                n_failed += 1
            cell = _confusableskeleton_cell_get(record, key)
            if cell is None:
                continue
            n_completed += 1
            if cell["verdict"]:
                n_pass += 1
            decomposition[
                _confusableskeleton_decompose_cell(
                    cell,
                    expected_by_fixture.get(str(record.get("fixture", "")), ""),
                    str(record.get("expected_digest", "")),
                    key,
                )
            ] += 1
        arms[key] = {
            "n_completed": n_completed,
            "n_pass": n_pass,
            "rate": _confusableskeleton_rate(n_pass, n_completed),
            "n_context_exceeded": n_context,
            "n_evaluation_failed": n_failed,
            "decomposition": decomposition,
        }
    ids: dict[str, Any] = {}
    for arm in _CONFUSABLESKELETON_TREATED_ARMS:
        deltas: list[int] = []
        n_ids_identical = 0
        n_same_ids_flips = 0
        for record in records:
            tokens = record.get("arm_tokens")
            tokens = tokens if isinstance(tokens, dict) else {}
            treated = tokens.get(arm)
            baseline = tokens.get("control")
            if type(treated) is int and type(baseline) is int:
                deltas.append(treated - baseline)
            else:
                deltas.append(0)
            if _confusableskeleton_ids_identical(record, arm):
                n_ids_identical += 1
                control_cell = _confusableskeleton_cell_get(record, "control")
                arm_cell = _confusableskeleton_cell_get(record, arm)
                if (
                    control_cell is not None
                    and arm_cell is not None
                    and (control_cell.get("verdict") is True) != (arm_cell.get("verdict") is True)
                ):
                    n_same_ids_flips += 1
        ids[arm] = {
            "n_ids_identical": n_ids_identical,
            "n_ids_divergent": len(records) - n_ids_identical,
            "id_length_delta": deltas,
            "max_abs_id_delta": max((abs(delta) for delta in deltas), default=0),
            "n_same_ids_verdict_flips": n_same_ids_flips,
        }
    dose_eligible = 0
    dose_letters = 0
    for case in cases:
        word, _, _number = str(case["value"]).partition("-")
        for treated in (str(case["topic"]), word):
            dose_eligible += _confusableskeleton_eligible_count(treated)
            dose_letters += len(treated)
    dose = {
        "whole_word": {
            "eligible_letters": dose_eligible,
            "total_letters": dose_letters,
            "fraction": _confusableskeleton_rate(dose_eligible, dose_letters),
        }
    }
    headline: dict[str, Any] = {}
    for arm in _CONFUSABLESKELETON_TREATED_ARMS:
        domain = [
            record for record in records if not _confusableskeleton_ids_identical(record, arm)
        ]
        paired = [
            record
            for record in domain
            if _confusableskeleton_cell_get(record, "control") is not None
            and _confusableskeleton_cell_get(record, arm) is not None
        ]
        n_incomplete = len(domain) - len(paired)
        control_passes = 0
        wrong_values = 0
        reverse = 0
        non_answer = 0
        pair_decomposition = dict.fromkeys(_CONFUSABLESKELETON_PAIR_CLASSES, 0)
        failing_classes = dict.fromkeys(_CONFUSABLESKELETON_CELL_CLASSES, 0)
        for record in paired:
            control_cell = _confusableskeleton_cell_get(record, "control")
            arm_cell = _confusableskeleton_cell_get(record, arm)
            if control_cell is None or arm_cell is None:
                continue
            expected = expected_by_fixture.get(str(record.get("fixture", "")), "")
            expected_digest = str(record.get("expected_digest", ""))
            arm_class = _confusableskeleton_decompose_cell(arm_cell, expected, expected_digest, arm)
            if arm_class == "non_answer":
                non_answer += 1
            if control_cell["verdict"]:
                control_passes += 1
                if arm_class == "wrong_value":
                    wrong_values += 1
            elif arm_cell["verdict"]:
                reverse += 1
            if control_cell["verdict"] != arm_cell["verdict"]:
                pair_decomposition[_confusableskeleton_decompose_pair(control_cell, arm_cell)] += 1
                failing = arm_cell if control_cell["verdict"] else control_cell
                failing_arm = arm if control_cell["verdict"] else "control"
                failing_classes[
                    _confusableskeleton_decompose_cell(
                        failing, expected, expected_digest, failing_arm
                    )
                ] += 1
        headline[arm] = {
            "label": _confusableskeleton_classify(control_passes, wrong_values, n_incomplete),
            "n_population": len(paired),
            "n_incomplete": n_incomplete,
            "control_passes": control_passes,
            "wrong_values": wrong_values,
            "n_reverse": reverse,
            "control_rate": _confusableskeleton_rate(control_passes, len(paired)),
            "arm_rate": _confusableskeleton_rate(
                sum(1 for record in paired if _confusableskeleton_verdict(record, arm)),
                len(paired),
            ),
            "n_non_answer": non_answer,
            "pair_decomposition": pair_decomposition,
            "failing_classes": failing_classes,
        }
    return {
        "arms": arms,
        "ids": ids,
        "dose": dose,
        "headline": headline,
        "operating_characteristics": dict(_CONFUSABLESKELETON_OPERATING_CHARACTERISTICS),
        "n_cases": len(records),
        "n_incomplete": sum(
            1
            for record in records
            if any(
                _confusableskeleton_cell_get(record, arm) is None
                for arm in _CONFUSABLESKELETON_ARM_KEYS
            )
        ),
    }


def _confusableskeleton_cell_ok(cell: Any, expected_digest: str) -> bool:
    """One cell's shape and bindings: the exact field set;
    the verdict binding ``verdict == (answer_digest ==
    expected_digest) and boundary_found``; the carried
    ``answer_text`` must hash to its ``answer_digest`` (the
    cycle-190 binding); the input-token binding to the
    record's arm token count is checked by the caller.
    """
    if type(cell) is not dict or set(cell) != _CONFUSABLESKELETON_CELL_FIELDS:
        return False
    if type(cell["verdict"]) is not bool or type(cell["boundary_found"]) is not bool:
        return False
    if cell["boundary_text_match"] is not None and type(cell["boundary_text_match"]) is not bool:
        return False
    if type(cell["input_tokens"]) is not int or cell["input_tokens"] < 0:
        return False
    digest = cell["answer_digest"]
    if digest is not None and not _confusableskeleton_digest_ok(digest):
        return False
    text = cell["answer_text"]
    if text is None:
        if digest is not None:
            return False
    elif type(text) is not str or _confusableskeleton_text_digest(text) != digest:
        return False
    verdict = (digest == expected_digest) and cell["boundary_found"]
    return bool(cell["verdict"] == verdict)


def _confusableskeleton_recompute(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Recompute a confusable-skeleton payload from its
    ground records.

    The validator carries its own generator mirror with the
    frozen SHA-256 derivation, the frozen population tables,
    the population audit, and the aggregators (no imports from
    confusableskeleton.py): the protocol is rebuilt from
    frozen literals plus the payload's seed and backend
    fingerprint and compared whole; the case set is
    regenerated from the seed and every record is checked
    against it in protocol order; every record's per-arm
    content digests are rebound to the regenerated arm
    contents, so the treatment itself is committed; every
    record's frozen-table pin is rebound to the mirror's own
    table digest; the record's ids_identical flags and
    id_length_delta values must equal the derivations from its
    recorded digests and token counts; every cell's
    answer_text is verified against its answer_digest, its
    verdict is re-derived from its digest identity, and its
    input_tokens are bound to the record's arm token count;
    the decompositions, id report, dose, and guarded headlines
    are recomputed from the records and the regenerated
    expected values under the pinned completion domain; the
    aggregates are recomputed and compared. Cells exist only
    from aggregate-eligible reports, so a non-null cell
    certifies a completed evaluation and the verdict's status
    conjunct is discharged structurally.
    """

    protocol = payload.get("protocol")
    if type(protocol) is not dict:
        return None
    seed = protocol.get("seed")
    backend_fingerprint = protocol.get("backend_fingerprint")
    if type(seed) is not int or isinstance(seed, bool) or type(backend_fingerprint) is not str:
        return None
    try:
        expected_protocol = _confusableskeleton_protocol(seed, backend_fingerprint)
    except _ConfusableSkeletonMirrorExhausted:
        return None
    if not _recomputed_equal(protocol, expected_protocol):
        return None
    candidates = payload.get("candidates")
    if type(candidates) is not dict or not candidates:
        return None
    if len(candidates) > _CONFUSABLESKELETON_MAX_CANDIDATES:
        return None
    try:
        generated = _confusableskeleton_generate(seed)
    except _ConfusableSkeletonMirrorExhausted:
        return None
    total_records = 0
    records_bytes = 0
    aggregates: dict[str, Any] = {}
    for name, entry in candidates.items():
        if type(entry) is not dict:
            return None
        status = entry.get("status")
        if status == "unavailable":
            if set(entry) != {"status", "detail", "records"} or entry.get("records") != []:
                return None
            continue
        if status != "available" or entry.get("detail") != "":
            return None
        if set(entry) != _CONFUSABLESKELETON_AGGREGATE_KEYS | {"status", "detail", "records"}:
            return None
        records = entry["records"]
        if type(records) is not list or len(records) != len(generated):
            return None
        total_records += len(records)
        records_bytes += len(
            json.dumps(records, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        )
        if records_bytes > _CONFUSABLESKELETON_MAX_RECORDS_BYTES:
            return None
        for record, case in zip(records, generated, strict=True):
            if type(record) is not dict or set(record) != {
                "fixture",
                "topic",
                "expected_digest",
                "content_digests",
                "arm_tokens",
                "id_digests",
                "ids_identical",
                "id_length_delta",
                "table_pin",
                "cells",
                "cell_details",
            }:
                return None
            if record["fixture"] != case["fixture_id"]:
                return None
            if record["topic"] != case["topic"]:
                return None
            expected_digest = _confusableskeleton_digest(str(case["expected"]))
            if record["expected_digest"] != expected_digest:
                return None
            table_pin = record["table_pin"]
            if type(table_pin) is not dict or set(table_pin) != {"version", "table_sha256"}:
                return None
            if table_pin["version"] != _CONFUSABLESKELETON_CONFUSABLES_VERSION:
                return None
            if table_pin["table_sha256"] != _confusableskeleton_table_sha256():
                return None
            arm_tokens = record["arm_tokens"]
            id_digests = record["id_digests"]
            content_digests = record["content_digests"]
            ids_identical = record["ids_identical"]
            id_length_delta = record["id_length_delta"]
            cells = record["cells"]
            details = record["cell_details"]
            for mapping in (arm_tokens, id_digests, content_digests, cells, details):
                if type(mapping) is not dict or set(mapping) != set(_CONFUSABLESKELETON_ARM_KEYS):
                    return None
            for mapping in (ids_identical, id_length_delta):
                if type(mapping) is not dict or set(mapping) != set(
                    _CONFUSABLESKELETON_TREATED_ARMS
                ):
                    return None
            # Validate every arm_tokens value BEFORE any arithmetic on
            # them, exactly as in _invisiblechars_recompute: a
            # type-confused value escaped as a raw TypeError while the
            # type check sat in the later per-arm loop (bug-hunt
            # wave 2, BH2B-02).
            for arm in _CONFUSABLESKELETON_ARM_KEYS:
                tokens = arm_tokens[arm]
                if type(tokens) is not int or isinstance(tokens, bool) or tokens < 0:
                    return None
            for arm in _CONFUSABLESKELETON_TREATED_ARMS:
                if ids_identical[arm] is not (id_digests[arm] == id_digests["control"]):
                    return None
                if id_length_delta[arm] != arm_tokens[arm] - arm_tokens["control"]:
                    return None
            for arm in _CONFUSABLESKELETON_ARM_KEYS:
                tokens = arm_tokens[arm]
                if type(tokens) is not int or isinstance(tokens, bool) or tokens < 0:
                    return None
                if not _confusableskeleton_digest_ok(id_digests[arm]):
                    return None
                if content_digests[arm] != _confusableskeleton_digest(
                    _confusableskeleton_arm_content(case, arm)
                ):
                    return None
                cell = cells[arm]
                detail = details[arm]
                if detail not in ("", *_CONFUSABLESKELETON_DETAILS):
                    return None
                if cell is None:
                    if detail == "":
                        return None
                    continue
                if detail != "":
                    return None
                if not _confusableskeleton_cell_ok(cell, expected_digest):
                    return None
                if cell["input_tokens"] != tokens:
                    return None
        aggregates[name] = _confusableskeleton_aggregate(records, generated)
    if total_records > _CONFUSABLESKELETON_MAX_GROUND_RECORDS:
        return None
    if not aggregates:
        return None
    return {
        "aggregates": aggregates,
        "n_candidates": len(candidates),
        "n_cases": len(generated),
    }


def _valid_confusable_skeleton(payload: Any) -> bool:
    """Structural check for a ``confusable_skeleton`` payload
    (recompute standard). The audit block is shape-checked
    only: its counts are stage observations (special-token
    findings against the run's own formats), not recomputable
    from records.
    """
    if type(payload) is not dict or payload.get("available") is not True:
        return False
    if set(payload) != {
        "available",
        "detail",
        "protocol",
        "audit",
        "candidates",
        "n_candidates",
        "n_cases",
    }:
        return False
    if payload.get("detail") != "":
        return False
    audit = payload.get("audit")
    if type(audit) is not dict or set(audit) != {
        "n_audit_excluded",
        "by_reason",
        "special_token_findings",
    }:
        return False
    if type(audit["n_audit_excluded"]) is not int or audit["n_audit_excluded"] < 0:
        return False
    if type(audit["special_token_findings"]) is not int or audit["special_token_findings"] < 0:
        return False
    by_reason = audit["by_reason"]
    if type(by_reason) is not dict:
        return False
    for key, count in by_reason.items():
        if type(key) is not str or type(count) is not int or count < 0:
            return False
    recomputed = _confusableskeleton_recompute(payload)
    if recomputed is None:
        return False
    candidates = payload["candidates"]
    for name, expected in recomputed["aggregates"].items():
        entry = candidates[name]
        for key, value in expected.items():
            if not _recomputed_equal(entry.get(key), value):
                return False
    if not _recomputed_equal(payload.get("n_candidates"), recomputed["n_candidates"]):
        return False
    return _recomputed_equal(payload.get("n_cases"), recomputed["n_cases"])


def diff_confusable_skeleton(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``confusable_skeleton`` probe payloads;
    deterministic.

    Answers "did the confusable-skeleton story change between
    runs?": per candidate and treated arm, drift in the
    guarded label with the D / J counts that carry it, the
    reverse and non-answer counts, the paired population and
    its incompletion, the id report, the whole-word dose, and
    the pair decomposition; candidates added/removed; and
    protocol / generator / backend drift (any of which marks
    the comparison not comparable, with the drift still listed
    -- the generated panel is the treatment, so a changed
    generator is protocol drift). A side without a valid
    payload is reported plainly.
    """
    old_ok = _valid_confusable_skeleton(old)
    new_ok = _valid_confusable_skeleton(new)
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
    cs_protocol_drift = old["protocol"].get("fingerprint") != new["protocol"].get("fingerprint")
    cs_generator_drift = old["protocol"].get("ground_sha256") != new["protocol"].get(
        "ground_sha256"
    ) or old["protocol"].get("case_material_sha256") != new["protocol"].get("case_material_sha256")
    cs_backend_drift = old["protocol"].get("backend_fingerprint") != new["protocol"].get(
        "backend_fingerprint"
    )
    cs_comparable = not cs_protocol_drift and not cs_generator_drift and not cs_backend_drift
    old_candidates = old["candidates"]
    new_candidates = new["candidates"]
    cs_candidates_added = sorted(set(new_candidates) - set(old_candidates))
    cs_candidates_removed = sorted(set(old_candidates) - set(new_candidates))
    cs_candidate_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_candidates) & set(new_candidates)):
        old_entry = old_candidates[name]
        new_entry = new_candidates[name]
        if old_entry.get("status") != "available" or new_entry.get("status") != "available":
            if old_entry.get("status") != new_entry.get("status"):
                cs_candidate_drift.append(
                    {
                        "candidate": name,
                        "metric": "status",
                        "old": old_entry.get("status"),
                        "new": new_entry.get("status"),
                    }
                )
            continue
        cs_metrics: list[tuple[str, Any, Any]] = []
        for arm in _CONFUSABLESKELETON_ARM_KEYS:
            cs_metrics.append(
                (f"{arm}_rate", old_entry["arms"][arm]["rate"], new_entry["arms"][arm]["rate"])
            )
        cs_metrics.extend(
            [
                (
                    "whole_word_eligible_letters",
                    old_entry["dose"]["whole_word"]["eligible_letters"],
                    new_entry["dose"]["whole_word"]["eligible_letters"],
                ),
                (
                    "whole_word_total_letters",
                    old_entry["dose"]["whole_word"]["total_letters"],
                    new_entry["dose"]["whole_word"]["total_letters"],
                ),
                (
                    "whole_word_fraction",
                    old_entry["dose"]["whole_word"]["fraction"],
                    new_entry["dose"]["whole_word"]["fraction"],
                ),
            ]
        )
        for arm in _CONFUSABLESKELETON_TREATED_ARMS:
            old_head = old_entry["headline"][arm]
            new_head = new_entry["headline"][arm]
            old_ids = old_entry["ids"][arm]
            new_ids = new_entry["ids"][arm]
            cs_metrics.extend(
                [
                    (f"{arm}_label", old_head["label"], new_head["label"]),
                    (
                        f"{arm}_control_passes",
                        old_head["control_passes"],
                        new_head["control_passes"],
                    ),
                    (f"{arm}_wrong_values", old_head["wrong_values"], new_head["wrong_values"]),
                    (f"{arm}_n_reverse", old_head["n_reverse"], new_head["n_reverse"]),
                    (f"{arm}_n_population", old_head["n_population"], new_head["n_population"]),
                    (f"{arm}_n_incomplete", old_head["n_incomplete"], new_head["n_incomplete"]),
                    (f"{arm}_n_non_answer", old_head["n_non_answer"], new_head["n_non_answer"]),
                    (f"{arm}_control_rate", old_head["control_rate"], new_head["control_rate"]),
                    (f"{arm}_arm_rate", old_head["arm_rate"], new_head["arm_rate"]),
                    (
                        f"{arm}_pair_exact_only",
                        old_head["pair_decomposition"]["exact_only"],
                        new_head["pair_decomposition"]["exact_only"],
                    ),
                    (
                        f"{arm}_pair_boundary_only",
                        old_head["pair_decomposition"]["boundary_only"],
                        new_head["pair_decomposition"]["boundary_only"],
                    ),
                    (
                        f"{arm}_pair_mixed",
                        old_head["pair_decomposition"]["mixed"],
                        new_head["pair_decomposition"]["mixed"],
                    ),
                    (
                        f"{arm}_n_ids_identical",
                        old_ids["n_ids_identical"],
                        new_ids["n_ids_identical"],
                    ),
                    (
                        f"{arm}_max_abs_id_delta",
                        old_ids["max_abs_id_delta"],
                        new_ids["max_abs_id_delta"],
                    ),
                    (
                        f"{arm}_n_same_ids_verdict_flips",
                        old_ids["n_same_ids_verdict_flips"],
                        new_ids["n_same_ids_verdict_flips"],
                    ),
                ]
            )
        for metric, old_value, new_value in cs_metrics:
            if old_value != new_value:
                cs_candidate_drift.append(
                    {"candidate": name, "metric": metric, "old": old_value, "new": new_value}
                )
    result["candidate_drift"] = cs_candidate_drift
    result["candidates_added"] = cs_candidates_added
    result["candidates_removed"] = cs_candidates_removed
    result["protocol_drift"] = cs_protocol_drift
    result["generator_drift"] = cs_generator_drift
    result["backend_drift"] = cs_backend_drift
    result["comparable"] = cs_comparable
    result["changed"] = bool(
        cs_candidate_drift
        or cs_candidates_added
        or cs_candidates_removed
        or cs_protocol_drift
        or cs_generator_drift
        or cs_backend_drift
    )
    return result


_COMPATIBILITYFORM_TOPICS = (
    "harbor",
    "summit",
    "quarry",
    "tundra",
    "granite",
    "willow",
    "copper",
    "saddle",
    "walnut",
    "velvet",
    "cobalt",
    "marble",
    "cinder",
    "falcon",
    "garnet",
    "saffron",
    "indigo",
    "juniper",
    "lagoon",
    "magnet",
    "nickel",
    "pepper",
    "quartz",
    "birch",
)


_COMPATIBILITYFORM_WORDS = (
    "alder",
    "basalt",
    "cedar",
    "drift",
    "elm",
    "fern",
    "glade",
    "heath",
    "iris",
    "jade",
    "kelp",
    "loom",
    "moss",
    "nova",
    "ochre",
    "pine",
    "quill",
    "reed",
    "sage",
    "tide",
    "urn",
    "vale",
    "wren",
    "yew",
)


_COMPATIBILITYFORM_NEUTRAL_STATEMENTS = (
    "The weather changed slowly during the afternoon hours.",
    "A wooden fence runs along the edge of the property.",
    "She placed the documents inside a plain folder yesterday.",
    "The train arrived later than the schedule suggested.",
    "Fresh bread was cooling on the counter near the sink.",
    "He repaired the bicycle with tools from the garage.",
    "The bookshop extends its opening hours during summer.",
    "Several chairs were arranged around the long table.",
    "The recipe calls for a pinch of salt and herbs.",
    "They walked along the path beside the lake at dawn.",
    "The committee postponed its decision until next week.",
    "A thick layer of dust covered the old furniture.",
    "The children played quietly in the courtyard after lunch.",
    "His handwriting was difficult to read in places.",
    "The farmer harvested the fields before the rain came.",
    "An old map hung framed on the wall of the hallway.",
    "The accountant reviewed the figures a second time.",
    "Birds gathered on the rooftops as the sun went down.",
    "The mechanic inspected the engine and changed the oil.",
    "A narrow staircase led up to the attic.",
    "The orchestra rehearsed the piece several times over.",
    "Someone left a bicycle leaning against the railing.",
    "The soup simmered gently on the stove all morning.",
    "Visitors signed the guest book near the entrance.",
)


_COMPATIBILITYFORM_CLASS_TABLES: dict[str, tuple[tuple[str, str], ...]] = {
    "ligature": (
        ("ĳ", "ij"),
        ("ﬀ", "ff"),
        ("ﬁ", "fi"),
        ("ﬂ", "fl"),
        ("ﬃ", "ffi"),
        ("ﬄ", "ffl"),
        ("ﬆ", "st"),
    ),
    "enclosed": (
        ("①", "1"),
        ("②", "2"),
        ("③", "3"),
        ("④", "4"),
        ("⑤", "5"),
        ("⑥", "6"),
        ("⑦", "7"),
        ("⑧", "8"),
        ("⑨", "9"),
        ("⓪", "0"),
    ),
    "fraction": (
        ("¼", "1/4"),
        ("½", "1/2"),
        ("¾", "3/4"),
        ("⅐", "1/7"),
        ("⅑", "1/9"),
        ("⅒", "1/10"),
        ("⅓", "1/3"),
        ("⅔", "2/3"),
        ("⅕", "1/5"),
        ("⅖", "2/5"),
        ("⅗", "3/5"),
        ("⅘", "4/5"),
        ("⅙", "1/6"),
        ("⅚", "5/6"),
        ("⅛", "1/8"),
        ("⅜", "3/8"),
        ("⅝", "5/8"),
        ("⅞", "7/8"),
        ("↉", "0/3"),
    ),
    "superscript": (
        ("²", "2"),
        ("³", "3"),
        ("¹", "1"),
        ("⁰", "0"),
        ("⁴", "4"),
        ("⁵", "5"),
        ("⁶", "6"),
        ("⁷", "7"),
        ("⁸", "8"),
        ("⁹", "9"),
    ),
    "roman": (
        ("Ⅰ", "I"),
        ("Ⅱ", "II"),
        ("Ⅲ", "III"),
        ("Ⅳ", "IV"),
        ("Ⅴ", "V"),
        ("Ⅵ", "VI"),
        ("Ⅶ", "VII"),
        ("Ⅷ", "VIII"),
        ("Ⅸ", "IX"),
        ("Ⅹ", "X"),
        ("Ⅺ", "XI"),
        ("Ⅻ", "XII"),
        ("Ⅼ", "L"),
        ("Ⅽ", "C"),
        ("Ⅾ", "D"),
        ("Ⅿ", "M"),
    ),
    "katakana": (
        ("ｦ", "ヲ"),
        ("ｧ", "ァ"),
        ("ｨ", "ィ"),
        ("ｩ", "ゥ"),
        ("ｪ", "ェ"),
        ("ｫ", "ォ"),
        ("ｬ", "ャ"),
        ("ｭ", "ュ"),
        ("ｮ", "ョ"),
        ("ｯ", "ッ"),
        ("ｱ", "ア"),
        ("ｲ", "イ"),
        ("ｳ", "ウ"),
        ("ｴ", "エ"),
        ("ｵ", "オ"),
        ("ｶ", "カ"),
        ("ｷ", "キ"),
        ("ｸ", "ク"),
        ("ｹ", "ケ"),
        ("ｺ", "コ"),
        ("ｻ", "サ"),
        ("ｼ", "シ"),
        ("ｽ", "ス"),
        ("ｾ", "セ"),
        ("ｿ", "ソ"),
        ("ﾀ", "タ"),
        ("ﾁ", "チ"),
        ("ﾂ", "ツ"),
        ("ﾃ", "テ"),
        ("ﾄ", "ト"),
        ("ﾅ", "ナ"),
        ("ﾆ", "ニ"),
        ("ﾇ", "ヌ"),
        ("ﾈ", "ネ"),
        ("ﾉ", "ノ"),
        ("ﾊ", "ハ"),
        ("ﾋ", "ヒ"),
        ("ﾌ", "フ"),
        ("ﾍ", "ヘ"),
        ("ﾎ", "ホ"),
        ("ﾏ", "マ"),
        ("ﾐ", "ミ"),
        ("ﾑ", "ム"),
        ("ﾒ", "メ"),
        ("ﾓ", "モ"),
        ("ﾔ", "ヤ"),
        ("ﾕ", "ユ"),
        ("ﾖ", "ヨ"),
        ("ﾗ", "ラ"),
        ("ﾘ", "リ"),
        ("ﾙ", "ル"),
        ("ﾚ", "レ"),
        ("ﾛ", "ロ"),
        ("ﾜ", "ワ"),
        ("ﾝ", "ン"),
    ),
}


_COMPATIBILITYFORM_CLASS_ORDER = tuple(_COMPATIBILITYFORM_CLASS_TABLES)


_COMPATIBILITYFORM_ARM_KEYS = ("control", *_COMPATIBILITYFORM_CLASS_ORDER)


_COMPATIBILITYFORM_TREATED_ARMS = _COMPATIBILITYFORM_ARM_KEYS[1:]


_COMPATIBILITYFORM_READING_BY_CP: dict[str, dict[int, str]] = {
    name: {ord(char): reading for char, reading in table}
    for name, table in _COMPATIBILITYFORM_CLASS_TABLES.items()
}


_COMPATIBILITYFORM_CHAR_BY_CP: dict[str, dict[int, str]] = {
    name: {ord(char): char for char, _reading in table}
    for name, table in _COMPATIBILITYFORM_CLASS_TABLES.items()
}


_COMPATIBILITYFORM_CLASS_CHARS: dict[str, frozenset[str]] = {
    arm: frozenset(char for char, _reading in _COMPATIBILITYFORM_CLASS_TABLES[arm])
    if arm != "control"
    else frozenset()
    for arm in _COMPATIBILITYFORM_ARM_KEYS
}


_COMPATIBILITYFORM_UNICODE_VERSION = "15.0.0"


_COMPATIBILITYFORM_UNICODEDATA_SHA256 = (
    "806e9aed65037197f1ec85e12be6e8cd870fc5608b4de0fffd990f689f376a73"
)


_COMPATIBILITYFORM_MARKER_TEMPLATE = (
    "The marker line reads script {f0}, circle {f1}, share {f2}, "
    "power {f3}, regnal Henry {f4}, kana {f5}."
)


_COMPATIBILITYFORM_CASE_COUNT = 24


_COMPATIBILITYFORM_MIN_D = 10


_COMPATIBILITYFORM_MIN_J = 6


_COMPATIBILITYFORM_MAX_INCOMPLETE = 6


_COMPATIBILITYFORM_REDRAW_LIMIT = 1000


_COMPATIBILITYFORM_GENERATOR_SPEC = (
    "draw(seed, index, slot, counter) = int.from_bytes("
    "sha256(f'{seed}|compatfold|{index}|{slot}|{counter}')"
    ".digest()[:8], 'big'); slot 0: word = WORDS[draw % len(WORDS)]; "
    "slot 1: number = 1000 + draw % 9000; a value colliding with an "
    "earlier case redraws slots 0 and 1 only at counter+1 (slots 2-9 "
    "stay at counter 0), the counter incrementing once per colliding "
    "redraw and giving up when it exceeds 1000 (the stage fails "
    "closed); slots 2/3: neutral indices = draw % "
    "len(NEUTRAL_STATEMENTS), the second advancing +1 on collision; "
    "slot 4: ligature; slot 5: enclosed; slot 6: fraction; slot 7: "
    "superscript; slot 8: roman; slot 9: katakana, each draw % "
    "len(class table) over the codepoint-sorted frozen tables"
)


_COMPATIBILITYFORM_MAX_CANDIDATES = 24


_COMPATIBILITYFORM_MAX_GROUND_RECORDS = (
    _COMPATIBILITYFORM_MAX_CANDIDATES * _COMPATIBILITYFORM_CASE_COUNT
)


_COMPATIBILITYFORM_GENERATION_BUDGET = (
    len(_COMPATIBILITYFORM_ARM_KEYS) * _COMPATIBILITYFORM_MAX_GROUND_RECORDS
)


_COMPATIBILITYFORM_MAX_RECORDS_BYTES = 12 * 1024 * 1024


_COMPATIBILITYFORM_CELL_FIELDS = {
    "verdict",
    "boundary_found",
    "boundary_text_match",
    "input_tokens",
    "answer_digest",
    "answer_text",
}


_COMPATIBILITYFORM_DETAILS = ("arm_context_exceeded", "arm_evaluation_failed")


_COMPATIBILITYFORM_CELL_CLASSES = (
    "pass",
    "boundary_only",
    "wrong_value",
    "echo_compat",
    "form_only",
    "non_answer",
)


_COMPATIBILITYFORM_PAIR_CLASSES = ("exact_only", "boundary_only", "mixed")


_COMPATIBILITYFORM_OPERATING_CHARACTERISTICS: dict[str, Any] = {
    "fire_competent_independent": 0.0173,
    "fire_competent_paired": 0.0044,
    "fire_weak_independent": 0.1743,
    "fire_random_independent": 0.5503,
    "power_drop10_base90": 0.2542,
    "power_drop25_base90": 0.8159,
    "power_drop25_base85": 0.8761,
    "power_drop25_paired": 0.6981,
    "power_drop50_base90": 0.999,
    "sensitivity18_drop25": 0.5214,
    "sensitivity18_competent": 0.0038,
    "panel_sweep_16_drop25": 0.3904,
    "panel_sweep_20_drop25": 0.6395,
    "transplant_fire_competent_independent": 0.8897,
    "familywise_competent_expected_144": 2.4875,
    "familywise_competent_any_144": 0.9187,
    "familywise_paired_expected_144": 0.6397,
    "familywise_paired_any_144": 0.4733,
    "familywise_random_expected_144": 79.2441,
    "world_a_guarded_fire": 0.0037,
    "world_a_unguarded_fire": 0.0069,
    "world_a_guarded_insufficient": 0.9765,
    "world_b_unguarded_below_threshold": 0.7231,
    "world_b_guarded_below_threshold": 0.0763,
    "world_b_guarded_fire": 0.1046,
    "world_c_guarded_fire": 0.7719,
    "world_c_unguarded_fire": 0.7824,
    "m20_drop25": 0.6395,
    "m20_drop10": 0.1356,
    "m20_competent": 0.0068,
    "m18_drop25": 0.5214,
    "m18_drop10": 0.0889,
    "m18_competent": 0.0038,
    "m16_drop25": 0.3904,
    "m16_drop10": 0.0527,
    "m16_competent": 0.0019,
    "m12_drop25": 0.1379,
    "m12_drop10": 0.0113,
    "m12_competent": 0.0003,
    "m10_drop25": 0.0331,
    "m10_drop10": 0.0022,
    "m10_competent": 0.0001,
}


_COMPATIBILITYFORM_AGGREGATE_KEYS = {
    "arms",
    "ids",
    "dose",
    "headline",
    "operating_characteristics",
    "n_cases",
    "n_incomplete",
}


def _compatibilityform_digest(text: str) -> str:
    """SHA-256 hex digest (local copy; no imports from
    compatibilityform.py anywhere in this mirror).
    """

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _compatibilityform_text_digest(text: str) -> str | None:
    """SHA-256 of the UTF-8 encoding, or None when the
    text cannot be encoded (a lone-surrogate answer_text
    is a validation failure, never an exception — the
    cycle-190 A3 pattern, applied per audit F2).
    """
    try:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    except UnicodeEncodeError:
        return None


def _compatibilityform_digest_ok(value: Any) -> bool:
    """Whether a stored digest is a well-formed SHA-256 hex
    digest.
    """
    if type(value) is not str or len(value) != 64:
        return False
    return all(ch in "0123456789abcdef" for ch in value)


def _compatibilityform_derive(seed: int, index: int, slot: int, counter: int) -> int:
    """Mirror of compatibilityform._derive (the draw
    conversion pinned by the design's generator spec).
    """

    material = f"{seed}|compatfold|{index}|{slot}|{counter}"
    return int.from_bytes(hashlib.sha256(material.encode("ascii")).digest()[:8], "big")


class _CompatibilityFormMirrorExhausted(Exception):
    """Mirror-internal: the redraw give-up bound was reached
    during regeneration, so the payload cannot be recomputed
    and validates false.
    """


def _compatibilityform_generate(seed: int) -> list[dict[str, Any]]:
    """Mirror of compatibilityform.generate_cases: the
    frozen SHA-256 derivation with panel-wide value
    distinctness by redraw.
    """
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index in range(_COMPATIBILITYFORM_CASE_COUNT):
        counter = 0
        while True:
            word = _COMPATIBILITYFORM_WORDS[
                _compatibilityform_derive(seed, index, 0, counter) % len(_COMPATIBILITYFORM_WORDS)
            ]
            number = 1000 + _compatibilityform_derive(seed, index, 1, counter) % 9000
            value = f"{word}-{number}"
            if value not in seen:
                break
            counter += 1
            if counter > _COMPATIBILITYFORM_REDRAW_LIMIT:
                raise _CompatibilityFormMirrorExhausted
        seen.add(value)
        first = _compatibilityform_derive(seed, index, 2, 0) % len(
            _COMPATIBILITYFORM_NEUTRAL_STATEMENTS
        )
        second = _compatibilityform_derive(seed, index, 3, 0) % len(
            _COMPATIBILITYFORM_NEUTRAL_STATEMENTS
        )
        if second == first:
            second = (first + 1) % len(_COMPATIBILITYFORM_NEUTRAL_STATEMENTS)
        members: dict[str, int] = {}
        for offset, name in enumerate(_COMPATIBILITYFORM_CLASS_ORDER):
            table = _COMPATIBILITYFORM_CLASS_TABLES[name]
            drawn = table[_compatibilityform_derive(seed, index, 4 + offset, 0) % len(table)]
            members[name] = ord(drawn[0])
        cases.append(
            {
                "fixture_id": f"compatibilityform-{index:02d}",
                "topic": _COMPATIBILITYFORM_TOPICS[index],
                "value": value,
                "expected": value,
                "neutral_indices": [first, second],
                "members": members,
            }
        )
    return cases


def _compatibilityform_fact(case: dict[str, Any]) -> str:
    """Mirror of compatibilityform.fact_sentence."""
    return f"The {case['topic']} inspection code is {case['value']}."


def _compatibilityform_question(case: dict[str, Any]) -> str:
    """Mirror of compatibilityform.question_text."""
    return f"What is the {case['topic']} inspection code?"


def _compatibilityform_marker_fields(case: dict[str, Any], arm: str) -> list[str]:
    """Mirror of compatibilityform.marker_fields."""
    members = case["members"]
    fields: list[str] = []
    for name in _COMPATIBILITYFORM_CLASS_ORDER:
        codepoint = int(members[name])
        reading = _COMPATIBILITYFORM_READING_BY_CP[name][codepoint]
        if arm == name:
            form = _COMPATIBILITYFORM_CHAR_BY_CP[name][codepoint]
            fields.append(f"x{form}" if name == "superscript" else form)
        else:
            fields.append(f"x{reading}" if name == "superscript" else reading)
    return fields


def _compatibilityform_marker_sentence(case: dict[str, Any], arm: str) -> str:
    """Mirror of compatibilityform.marker_sentence."""
    fields = _compatibilityform_marker_fields(case, arm)
    return _COMPATIBILITYFORM_MARKER_TEMPLATE.format(
        **{f"f{position}": field for position, field in enumerate(fields)}
    )


def _compatibilityform_arm_content(case: dict[str, Any], arm: str) -> str:
    """Mirror of compatibilityform.arm_content."""
    if arm not in _COMPATIBILITYFORM_ARM_KEYS:
        raise ValueError(f"unknown compatibility-form arm: {arm!r}")
    first, second = (int(i) for i in case["neutral_indices"])
    return (
        f"{_COMPATIBILITYFORM_NEUTRAL_STATEMENTS[first]} "
        f"{_compatibilityform_fact(case)} "
        f"{_compatibilityform_marker_sentence(case, arm)} "
        f"{_COMPATIBILITYFORM_NEUTRAL_STATEMENTS[second]} "
        f"{_compatibilityform_question(case)}"
    )


def _compatibilityform_member_satisfies(name: str, char: str, reading: str) -> bool:
    """Mirror of compatibilityform._member_satisfies."""
    import unicodedata as _unicodedata

    decomposition = _unicodedata.decomposition(char)
    parts = decomposition.split()
    if not parts or not parts[0].startswith("<") or parts[0] == "<wide>":
        return False
    if _unicodedata.normalize("NFC", char) != char or _unicodedata.normalize("NFD", char) != char:
        return False
    folded = _unicodedata.normalize("NFKC", char)
    if folded == char:
        return False
    member_name = _unicodedata.name(char, "")
    if name == "ligature":
        targets = "".join(chr(int(token, 16)) for token in parts[1:])
        return (
            member_name.startswith("LATIN SMALL LIGATURE")
            and targets.isascii()
            and reading == folded == targets
        )
    if name == "enclosed":
        return member_name.startswith("CIRCLED DIGIT") and reading == folded
    if name == "fraction":
        return member_name.startswith("VULGAR FRACTION") and reading == folded.replace(
            "\u2044", "/"
        )
    if name == "superscript":
        return (
            parts[0] == "<super>"
            and len(parts) == 2
            and reading == folded
            and reading in "0123456789"
        )
    if name == "roman":
        return member_name.startswith("ROMAN NUMERAL") and reading == folded
    if name == "katakana":
        return member_name.startswith("HALFWIDTH KATAKANA LETTER") and reading == folded
    return False


def _compatibilityform_population_audit() -> dict[str, Any]:
    """Mirror of compatibilityform.audit_population."""
    import unicodedata as _unicodedata

    classes: dict[str, Any] = {}
    seen: set[str] = set()
    ok = True
    for name in _COMPATIBILITYFORM_CLASS_ORDER:
        table = _COMPATIBILITYFORM_CLASS_TABLES[name]
        codepoints = [ord(char) for char, _reading in table]
        members_ok = all(
            _compatibilityform_member_satisfies(name, char, reading) for char, reading in table
        )
        disjoint = not any(char in seen for char, _reading in table)
        seen.update(char for char, _reading in table)
        entry_ok = (
            members_ok
            and disjoint
            and codepoints == sorted(codepoints)
            and all("-" not in reading for _char, reading in table)
        )
        if name != "katakana":
            entry_ok = entry_ok and all(reading.isascii() for _char, reading in table)
        else:
            entry_ok = entry_ok and all(
                len(reading) == 1 and not reading.isascii() for _char, reading in table
            )
        ok = ok and entry_ok
        classes[name] = {
            "count": len(table),
            "members": [f"U+{codepoint:04X}" for codepoint in codepoints],
            "satisfies": entry_ok,
        }
    return {
        "classes": classes,
        "total": sum(
            len(_COMPATIBILITYFORM_CLASS_TABLES[name]) for name in _COMPATIBILITYFORM_CLASS_ORDER
        ),
        "pairwise_disjoint": len(seen)
        == sum(
            len(_COMPATIBILITYFORM_CLASS_TABLES[name]) for name in _COMPATIBILITYFORM_CLASS_ORDER
        ),
        "ok": ok,
        "unicode_version": _unicodedata.unidata_version,
        "unicodedata_version": _COMPATIBILITYFORM_UNICODE_VERSION,
        "unicodedata_sha256": _COMPATIBILITYFORM_UNICODEDATA_SHA256,
    }


def _compatibilityform_table_sha256() -> str:
    """Mirror of compatibilityform.table_sha256."""

    material = {
        "classes": {
            name: [[char, reading] for char, reading in _COMPATIBILITYFORM_CLASS_TABLES[name]]
            for name in _COMPATIBILITYFORM_CLASS_ORDER
        },
        "unicode_version": _COMPATIBILITYFORM_UNICODE_VERSION,
        "unicodedata_sha256": _COMPATIBILITYFORM_UNICODEDATA_SHA256,
    }
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return _compatibilityform_digest(canonical)


def _compatibilityform_case_material_sha256() -> str:
    """Mirror of compatibilityform.case_material_sha256."""

    material = {
        "topics": list(_COMPATIBILITYFORM_TOPICS),
        "words": list(_COMPATIBILITYFORM_WORDS),
        "neutral_statements": list(_COMPATIBILITYFORM_NEUTRAL_STATEMENTS),
        "arm_keys": list(_COMPATIBILITYFORM_ARM_KEYS),
        "marker_template": _COMPATIBILITYFORM_MARKER_TEMPLATE,
        "generator_spec": _COMPATIBILITYFORM_GENERATOR_SPEC,
        "case_count": _COMPATIBILITYFORM_CASE_COUNT,
        "table_sha256": _compatibilityform_table_sha256(),
        "min_d": _COMPATIBILITYFORM_MIN_D,
        "min_j": _COMPATIBILITYFORM_MIN_J,
        "max_incomplete": _COMPATIBILITYFORM_MAX_INCOMPLETE,
        "redraw_limit": _COMPATIBILITYFORM_REDRAW_LIMIT,
        "population_audit": _compatibilityform_population_audit(),
    }
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return _compatibilityform_digest(canonical)


def _compatibilityform_ground_sha256(seed: int) -> str:
    """Mirror of compatibilityform.ground_sha256."""

    rows = [
        [
            case["fixture_id"],
            case["topic"],
            case["value"],
            list(case["neutral_indices"]),
            [int(case["members"][name]) for name in _COMPATIBILITYFORM_CLASS_ORDER],
        ]
        for case in _compatibilityform_generate(seed)
    ]
    canonical = json.dumps(rows, separators=(",", ":"), ensure_ascii=True)
    return _compatibilityform_digest(canonical)


def _compatibilityform_protocol(seed: int, backend_fingerprint: str) -> dict[str, Any]:
    """The compatibility-form protocol rebuilt from this
    module's own literals, fingerprint included (compared
    whole).
    """

    protocol: dict[str, Any] = {
        "version": 1,
        "arm_keys": list(_COMPATIBILITYFORM_ARM_KEYS),
        "treated_arms": list(_COMPATIBILITYFORM_TREATED_ARMS),
        "case_count": _COMPATIBILITYFORM_CASE_COUNT,
        "min_d": _COMPATIBILITYFORM_MIN_D,
        "min_j": _COMPATIBILITYFORM_MIN_J,
        "max_incomplete": _COMPATIBILITYFORM_MAX_INCOMPLETE,
        "case_material_sha256": _compatibilityform_case_material_sha256(),
        "ground_sha256": _compatibilityform_ground_sha256(seed),
        "table_sha256": _compatibilityform_table_sha256(),
        "unicode_version": _COMPATIBILITYFORM_UNICODE_VERSION,
        "unicodedata_sha256": _COMPATIBILITYFORM_UNICODEDATA_SHA256,
        "population_audit": _compatibilityform_population_audit(),
        "generation_budget": _COMPATIBILITYFORM_GENERATION_BUDGET,
        "max_ground_records": _COMPATIBILITYFORM_MAX_GROUND_RECORDS,
        "max_records_bytes": _COMPATIBILITYFORM_MAX_RECORDS_BYTES,
        "max_candidates": _COMPATIBILITYFORM_MAX_CANDIDATES,
        "seed": seed,
        "backend_fingerprint": backend_fingerprint,
    }
    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    protocol["fingerprint"] = _compatibilityform_digest(canonical)
    return protocol


def _compatibilityform_cell_get(record: dict[str, Any], key: str) -> dict[str, Any] | None:
    """Return the compatibilityform cell get."""
    cells = record.get("cells")
    if not isinstance(cells, dict):
        return None
    cell = cells.get(key)
    return cell if isinstance(cell, dict) else None


def _compatibilityform_detail_get(record: dict[str, Any], key: str) -> str:
    """Return the compatibilityform detail get."""
    details = record.get("cell_details")
    if not isinstance(details, dict):
        return ""
    detail = details.get(key, "")
    return detail if type(detail) is str else ""


def _compatibilityform_verdict(record: dict[str, Any], key: str) -> bool:
    """Return the compatibilityform verdict."""
    cell = _compatibilityform_cell_get(record, key)
    return bool(cell is not None and cell.get("verdict") is True)


def _compatibilityform_ids_identical(record: dict[str, Any], arm: str) -> bool:
    """Mirror of compatibilityform._ids_identical: derived
    from the recorded id digests alone.
    """
    digests = record.get("id_digests")
    if not isinstance(digests, dict):
        return False
    control = digests.get("control")
    return type(control) is str and control != "" and control == digests.get(arm)


def _compatibilityform_decompose_cell(
    cell: dict[str, Any], expected: str, expected_digest: str, arm: str
) -> str:
    """Mirror of compatibilityform.decompose_cell (the
    ordered decision list; the control arm's class-4 table
    is the empty set, amendment A3).
    """
    import unicodedata as _unicodedata

    if cell.get("verdict") is True:
        return "pass"
    digest = cell.get("answer_digest")
    if digest is not None and digest == expected_digest:
        return "boundary_only"
    text = cell.get("answer_text")
    if type(text) is not str:
        return "non_answer"
    if text != expected and re.fullmatch(r"^[a-z]+-[0-9]{4}$", text):
        word, _, number = text.partition("-")
        if word in _COMPATIBILITYFORM_WORDS and 1000 <= int(number) <= 9999:
            return "wrong_value"
    if text != expected and _unicodedata.normalize("NFKC", text) == expected:
        if any(char in _COMPATIBILITYFORM_CLASS_CHARS.get(arm, frozenset()) for char in text):
            return "echo_compat"
        return "form_only"
    return "non_answer"


def _compatibilityform_decompose_pair(
    control_cell: dict[str, Any], arm_cell: dict[str, Any]
) -> str:
    """Mirror of compatibilityform.decompose_pair."""
    if control_cell.get("answer_digest") == arm_cell.get("answer_digest"):
        return "boundary_only"
    if control_cell.get("boundary_found") and arm_cell.get("boundary_found"):
        return "exact_only"
    return "mixed"


def _compatibilityform_classify(control_passes: int, wrong_values: int, n_incomplete: int) -> str:
    """Mirror of compatibilityform.classify_headline."""
    if n_incomplete > _COMPATIBILITYFORM_MAX_INCOMPLETE:
        return "insufficient"
    if control_passes < _COMPATIBILITYFORM_MIN_D:
        return "insufficient"
    if wrong_values >= _COMPATIBILITYFORM_MIN_J:
        return "discordant_asymmetric"
    if wrong_values == 0:
        return "no_discordance_observed" if n_incomplete == 0 else "insufficient"
    return (
        "below_threshold"
        if wrong_values + n_incomplete < _COMPATIBILITYFORM_MIN_J
        else "insufficient"
    )


def _compatibilityform_rate(numerator: int, denominator: int) -> float | None:
    """Return the compatibilityform rate."""
    return (numerator / denominator) if denominator else None


def _compatibilityform_aggregate(
    records: list[dict[str, Any]], cases: list[dict[str, Any]]
) -> dict[str, Any]:
    """Mirror of compatibilityform.aggregate_candidate."""
    expected_by_fixture = {str(case["fixture_id"]): str(case["expected"]) for case in cases}
    arms: dict[str, Any] = {}
    for key in _COMPATIBILITYFORM_ARM_KEYS:
        n_completed = 0
        n_pass = 0
        n_context = 0
        n_failed = 0
        decomposition = dict.fromkeys(_COMPATIBILITYFORM_CELL_CLASSES, 0)
        for record in records:
            detail = _compatibilityform_detail_get(record, key)
            if detail == "arm_context_exceeded":
                n_context += 1
            elif detail == "arm_evaluation_failed":
                n_failed += 1
            cell = _compatibilityform_cell_get(record, key)
            if cell is None:
                continue
            n_completed += 1
            if cell.get("verdict") is True:
                n_pass += 1
            decomposition[
                _compatibilityform_decompose_cell(
                    cell,
                    expected_by_fixture.get(str(record.get("fixture", "")), ""),
                    str(record.get("expected_digest", "")),
                    key,
                )
            ] += 1
        arms[key] = {
            "n_completed": n_completed,
            "n_pass": n_pass,
            "rate": _compatibilityform_rate(n_pass, n_completed),
            "n_context_exceeded": n_context,
            "n_evaluation_failed": n_failed,
            "decomposition": decomposition,
        }
    ids: dict[str, Any] = {}
    for arm in _COMPATIBILITYFORM_TREATED_ARMS:
        deltas: list[int] = []
        n_ids_identical = 0
        n_same_ids_flips = 0
        for record in records:
            tokens = record.get("arm_tokens")
            tokens = tokens if isinstance(tokens, dict) else {}
            treated = tokens.get(arm)
            baseline = tokens.get("control")
            if type(treated) is int and type(baseline) is int:
                deltas.append(treated - baseline)
            else:
                deltas.append(0)
            if _compatibilityform_ids_identical(record, arm):
                n_ids_identical += 1
                control_cell = _compatibilityform_cell_get(record, "control")
                arm_cell = _compatibilityform_cell_get(record, arm)
                if (
                    control_cell is not None
                    and arm_cell is not None
                    and (control_cell.get("verdict") is True) != (arm_cell.get("verdict") is True)
                ):
                    n_same_ids_flips += 1
        ids[arm] = {
            "n_ids_identical": n_ids_identical,
            "n_ids_divergent": len(records) - n_ids_identical,
            "id_length_delta": deltas,
            "max_abs_id_delta": max((abs(delta) for delta in deltas), default=0),
            "n_same_ids_verdict_flips": n_same_ids_flips,
        }
    dose: dict[str, Any] = {}
    for arm in _COMPATIBILITYFORM_TREATED_ARMS:
        codepoint_deltas: list[int] = []
        byte_deltas: list[int] = []
        for case in cases:
            treated = _compatibilityform_arm_content(case, arm)
            control = _compatibilityform_arm_content(case, "control")
            codepoint_deltas.append(len(treated) - len(control))
            byte_deltas.append(len(treated.encode("utf-8")) - len(control.encode("utf-8")))
        dose[arm] = {
            "codepoint_delta_min": min(codepoint_deltas, default=0),
            "codepoint_delta_max": max(codepoint_deltas, default=0),
            "byte_delta_min": min(byte_deltas, default=0),
            "byte_delta_max": max(byte_deltas, default=0),
        }
    headline: dict[str, Any] = {}
    for arm in _COMPATIBILITYFORM_TREATED_ARMS:
        domain = [record for record in records if not _compatibilityform_ids_identical(record, arm)]
        paired = [
            record
            for record in domain
            if _compatibilityform_cell_get(record, "control") is not None
            and _compatibilityform_cell_get(record, arm) is not None
        ]
        n_incomplete = len(domain) - len(paired)
        control_passes = 0
        wrong_values = 0
        reverse = 0
        non_answer = 0
        pair_decomposition = dict.fromkeys(_COMPATIBILITYFORM_PAIR_CLASSES, 0)
        failing_classes = dict.fromkeys(_COMPATIBILITYFORM_CELL_CLASSES, 0)
        for record in paired:
            control_cell = _compatibilityform_cell_get(record, "control")
            arm_cell = _compatibilityform_cell_get(record, arm)
            if control_cell is None or arm_cell is None:
                continue
            expected = expected_by_fixture.get(str(record.get("fixture", "")), "")
            expected_digest = str(record.get("expected_digest", ""))
            arm_class = _compatibilityform_decompose_cell(arm_cell, expected, expected_digest, arm)
            if arm_class == "non_answer":
                non_answer += 1
            control_verdict = control_cell.get("verdict") is True
            arm_verdict = arm_cell.get("verdict") is True
            if control_verdict:
                control_passes += 1
                if arm_class == "wrong_value":
                    wrong_values += 1
            elif arm_verdict:
                reverse += 1
            if control_verdict != arm_verdict:
                pair_decomposition[_compatibilityform_decompose_pair(control_cell, arm_cell)] += 1
                failing = arm_cell if control_verdict else control_cell
                failing_arm = arm if control_verdict else "control"
                failing_classes[
                    _compatibilityform_decompose_cell(
                        failing, expected, expected_digest, failing_arm
                    )
                ] += 1
        headline[arm] = {
            "label": _compatibilityform_classify(control_passes, wrong_values, n_incomplete),
            "n_population": len(paired),
            "n_incomplete": n_incomplete,
            "n_ids_divergent": len(domain),
            "control_passes": control_passes,
            "wrong_values": wrong_values,
            "n_reverse": reverse,
            "control_rate": _compatibilityform_rate(control_passes, len(paired)),
            "arm_rate": _compatibilityform_rate(
                sum(1 for record in paired if _compatibilityform_verdict(record, arm)),
                len(paired),
            ),
            "n_non_answer": non_answer,
            "pair_decomposition": pair_decomposition,
            "failing_classes": failing_classes,
        }
    return {
        "arms": arms,
        "ids": ids,
        "dose": dose,
        "headline": headline,
        "operating_characteristics": dict(_COMPATIBILITYFORM_OPERATING_CHARACTERISTICS),
        "n_cases": len(records),
        "n_incomplete": sum(
            1
            for record in records
            if any(
                _compatibilityform_cell_get(record, arm) is None
                for arm in _COMPATIBILITYFORM_ARM_KEYS
            )
        ),
    }


def _compatibilityform_cell_ok(cell: Any, expected_digest: str) -> bool:
    """One cell's shape and bindings: the exact field set;
    the verdict binding ``verdict == (answer_digest ==
    expected_digest) and boundary_found``; the carried
    ``answer_text`` must hash to its ``answer_digest`` (the
    cycle-190 binding); the input-token binding to the
    record's arm token count is checked by the caller.
    """
    if type(cell) is not dict or set(cell) != _COMPATIBILITYFORM_CELL_FIELDS:
        return False
    if type(cell["verdict"]) is not bool or type(cell["boundary_found"]) is not bool:
        return False
    if cell["boundary_text_match"] is not None and type(cell["boundary_text_match"]) is not bool:
        return False
    if type(cell["input_tokens"]) is not int or cell["input_tokens"] < 0:
        return False
    digest = cell["answer_digest"]
    if digest is not None and not _compatibilityform_digest_ok(digest):
        return False
    text = cell["answer_text"]
    if text is None:
        if digest is not None:
            return False
    elif type(text) is not str or _compatibilityform_text_digest(text) != digest:
        return False
    verdict = (digest == expected_digest) and cell["boundary_found"]
    return bool(cell["verdict"] == verdict)


def _compatibilityform_recompute(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Recompute a compatibility-form payload from its
    ground records.

    The validator carries its own generator mirror with the
    frozen SHA-256 derivation, the frozen class tables, the
    population audit, and the aggregators (no imports from
    compatibilityform.py): the protocol is rebuilt from
    frozen literals plus the payload's seed and backend
    fingerprint and compared whole; the case set is
    regenerated from the seed and every record is checked
    against it in protocol order; every record's per-arm
    content digests are rebound to the regenerated arm
    contents, so the treatment itself is committed; every
    record's drawn members are rebound to the regenerated
    case's members; every record's frozen-table pin is
    rebound to the mirror's own table digest and data-file
    digest; the record's ids_identical flags and
    id_length_delta values must equal the derivations from
    its recorded digests and token counts; every cell's
    answer_text is verified against its answer_digest, its
    verdict is re-derived from its digest identity, and its
    input_tokens are bound to the record's arm token count;
    the decompositions, id report, dose, and guarded
    headlines are recomputed from the records and the
    regenerated expected values under the pinned completion
    domain; the aggregates are recomputed and compared.
    Cells exist only from aggregate-eligible reports, so a
    non-null cell certifies a completed evaluation and the
    verdict's status conjunct is discharged structurally.
    """

    protocol = payload.get("protocol")
    if type(protocol) is not dict:
        return None
    seed = protocol.get("seed")
    backend_fingerprint = protocol.get("backend_fingerprint")
    if type(seed) is not int or isinstance(seed, bool) or type(backend_fingerprint) is not str:
        return None
    try:
        expected_protocol = _compatibilityform_protocol(seed, backend_fingerprint)
    except _CompatibilityFormMirrorExhausted:
        return None
    if not _recomputed_equal(protocol, expected_protocol):
        return None
    candidates = payload.get("candidates")
    if type(candidates) is not dict or not candidates:
        return None
    if len(candidates) > _COMPATIBILITYFORM_MAX_CANDIDATES:
        return None
    try:
        generated = _compatibilityform_generate(seed)
    except _CompatibilityFormMirrorExhausted:
        return None
    total_records = 0
    records_bytes = 0
    aggregates: dict[str, Any] = {}
    for name, entry in candidates.items():
        if type(entry) is not dict:
            return None
        status = entry.get("status")
        if status == "unavailable":
            if set(entry) != {"status", "detail", "records"} or entry.get("records") != []:
                return None
            continue
        if status != "available" or entry.get("detail") != "":
            return None
        if set(entry) != _COMPATIBILITYFORM_AGGREGATE_KEYS | {"status", "detail", "records"}:
            return None
        records = entry["records"]
        if type(records) is not list or len(records) != len(generated):
            return None
        total_records += len(records)
        records_bytes += len(
            json.dumps(records, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        )
        if records_bytes > _COMPATIBILITYFORM_MAX_RECORDS_BYTES:
            return None
        for record, case in zip(records, generated, strict=True):
            if type(record) is not dict or set(record) != {
                "fixture",
                "topic",
                "expected_digest",
                "content_digests",
                "arm_tokens",
                "id_digests",
                "ids_identical",
                "id_length_delta",
                "table_pin",
                "members",
                "cells",
                "cell_details",
            }:
                return None
            if record["fixture"] != case["fixture_id"]:
                return None
            if record["topic"] != case["topic"]:
                return None
            expected_digest = _compatibilityform_digest(str(case["expected"]))
            if record["expected_digest"] != expected_digest:
                return None
            if record["members"] != case["members"]:
                return None
            table_pin = record["table_pin"]
            if type(table_pin) is not dict or set(table_pin) != {
                "version",
                "unicodedata_sha256",
                "table_sha256",
            }:
                return None
            if table_pin["version"] != _COMPATIBILITYFORM_UNICODE_VERSION:
                return None
            if table_pin["unicodedata_sha256"] != _COMPATIBILITYFORM_UNICODEDATA_SHA256:
                return None
            if table_pin["table_sha256"] != _compatibilityform_table_sha256():
                return None
            arm_tokens = record["arm_tokens"]
            id_digests = record["id_digests"]
            content_digests = record["content_digests"]
            ids_identical = record["ids_identical"]
            id_length_delta = record["id_length_delta"]
            cells = record["cells"]
            details = record["cell_details"]
            for mapping in (arm_tokens, id_digests, content_digests, cells, details):
                if type(mapping) is not dict or set(mapping) != set(_COMPATIBILITYFORM_ARM_KEYS):
                    return None
            for mapping in (ids_identical, id_length_delta):
                if type(mapping) is not dict or set(mapping) != set(
                    _COMPATIBILITYFORM_TREATED_ARMS
                ):
                    return None
            # Validate every arm_tokens value BEFORE any arithmetic on
            # them, exactly as in _invisiblechars_recompute: a
            # type-confused value escaped as a raw TypeError while the
            # type check sat in the later per-arm loop (bug-hunt
            # wave 2, BH2B-02).
            for arm in _COMPATIBILITYFORM_ARM_KEYS:
                tokens = arm_tokens[arm]
                if type(tokens) is not int or isinstance(tokens, bool) or tokens < 0:
                    return None
            for arm in _COMPATIBILITYFORM_TREATED_ARMS:
                if ids_identical[arm] is not (id_digests[arm] == id_digests["control"]):
                    return None
                if id_length_delta[arm] != arm_tokens[arm] - arm_tokens["control"]:
                    return None
            for arm in _COMPATIBILITYFORM_ARM_KEYS:
                tokens = arm_tokens[arm]
                if type(tokens) is not int or isinstance(tokens, bool) or tokens < 0:
                    return None
                if not _compatibilityform_digest_ok(id_digests[arm]):
                    return None
                if content_digests[arm] != _compatibilityform_digest(
                    _compatibilityform_arm_content(case, arm)
                ):
                    return None
                cell = cells[arm]
                detail = details[arm]
                if detail not in ("", *_COMPATIBILITYFORM_DETAILS):
                    return None
                if cell is None:
                    if detail == "":
                        return None
                    continue
                if detail != "":
                    return None
                if not _compatibilityform_cell_ok(cell, expected_digest):
                    return None
                if cell["input_tokens"] != tokens:
                    return None
        aggregates[name] = _compatibilityform_aggregate(records, generated)
    if total_records > _COMPATIBILITYFORM_MAX_GROUND_RECORDS:
        return None
    if not aggregates:
        return None
    return {
        "aggregates": aggregates,
        "n_candidates": len(candidates),
        "n_cases": len(generated),
    }


def _valid_compatibility_form(payload: Any) -> bool:
    """Structural check for a ``compatibility_form`` payload
    (recompute standard). The audit block is shape-checked
    only: its counts are stage observations (special-token
    findings against the run's own formats), not recomputable
    from records.
    """
    if type(payload) is not dict or payload.get("available") is not True:
        return False
    if set(payload) != {
        "available",
        "detail",
        "protocol",
        "audit",
        "candidates",
        "n_candidates",
        "n_cases",
    }:
        return False
    if payload.get("detail") != "":
        return False
    audit = payload.get("audit")
    if type(audit) is not dict or set(audit) != {
        "n_audit_excluded",
        "by_reason",
        "special_token_findings",
    }:
        return False
    if type(audit["n_audit_excluded"]) is not int or audit["n_audit_excluded"] < 0:
        return False
    if type(audit["special_token_findings"]) is not int or audit["special_token_findings"] < 0:
        return False
    by_reason = audit["by_reason"]
    if type(by_reason) is not dict:
        return False
    for key, count in by_reason.items():
        if type(key) is not str or type(count) is not int or count < 0:
            return False
    recomputed = _compatibilityform_recompute(payload)
    if recomputed is None:
        return False
    candidates = payload["candidates"]
    for name, expected in recomputed["aggregates"].items():
        entry = candidates[name]
        for key, value in expected.items():
            if not _recomputed_equal(entry.get(key), value):
                return False
    if not _recomputed_equal(payload.get("n_candidates"), recomputed["n_candidates"]):
        return False
    return _recomputed_equal(payload.get("n_cases"), recomputed["n_cases"])


def diff_compatibility_form(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``compatibility_form`` probe payloads;
    deterministic.

    Answers "did the compatibility-form story change between
    runs?": per candidate and treated arm, drift in the
    guarded label with the D / J counts that carry it, the
    divergent-domain size M, the reverse and non-answer
    counts, the paired population and its incompletion, the
    id report, the per-arm dose deltas, and the pair
    decomposition; candidates added/removed; and protocol /
    generator / backend drift (any of which marks the
    comparison not comparable, with the drift still listed
    -- the generated panel is the treatment, so a changed
    generator is protocol drift). A side without a valid
    payload is reported plainly.
    """
    old_ok = _valid_compatibility_form(old)
    new_ok = _valid_compatibility_form(new)
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
    cf_protocol_drift = old["protocol"].get("fingerprint") != new["protocol"].get("fingerprint")
    cf_generator_drift = old["protocol"].get("ground_sha256") != new["protocol"].get(
        "ground_sha256"
    ) or old["protocol"].get("case_material_sha256") != new["protocol"].get("case_material_sha256")
    cf_backend_drift = old["protocol"].get("backend_fingerprint") != new["protocol"].get(
        "backend_fingerprint"
    )
    cf_comparable = not cf_protocol_drift and not cf_generator_drift and not cf_backend_drift
    old_candidates = old["candidates"]
    new_candidates = new["candidates"]
    cf_candidates_added = sorted(set(new_candidates) - set(old_candidates))
    cf_candidates_removed = sorted(set(old_candidates) - set(new_candidates))
    cf_candidate_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_candidates) & set(new_candidates)):
        old_entry = old_candidates[name]
        new_entry = new_candidates[name]
        if old_entry.get("status") != "available" or new_entry.get("status") != "available":
            if old_entry.get("status") != new_entry.get("status"):
                cf_candidate_drift.append(
                    {
                        "candidate": name,
                        "metric": "status",
                        "old": old_entry.get("status"),
                        "new": new_entry.get("status"),
                    }
                )
            continue
        cf_metrics: list[tuple[str, Any, Any]] = []
        for arm in _COMPATIBILITYFORM_ARM_KEYS:
            cf_metrics.append(
                (f"{arm}_rate", old_entry["arms"][arm]["rate"], new_entry["arms"][arm]["rate"])
            )
        for arm in _COMPATIBILITYFORM_TREATED_ARMS:
            cf_metrics.extend(
                [
                    (
                        f"{arm}_dose_codepoint_min",
                        old_entry["dose"][arm]["codepoint_delta_min"],
                        new_entry["dose"][arm]["codepoint_delta_min"],
                    ),
                    (
                        f"{arm}_dose_codepoint_max",
                        old_entry["dose"][arm]["codepoint_delta_max"],
                        new_entry["dose"][arm]["codepoint_delta_max"],
                    ),
                    (
                        f"{arm}_dose_byte_min",
                        old_entry["dose"][arm]["byte_delta_min"],
                        new_entry["dose"][arm]["byte_delta_min"],
                    ),
                    (
                        f"{arm}_dose_byte_max",
                        old_entry["dose"][arm]["byte_delta_max"],
                        new_entry["dose"][arm]["byte_delta_max"],
                    ),
                ]
            )
        for arm in _COMPATIBILITYFORM_TREATED_ARMS:
            old_head = old_entry["headline"][arm]
            new_head = new_entry["headline"][arm]
            old_ids = old_entry["ids"][arm]
            new_ids = new_entry["ids"][arm]
            cf_metrics.extend(
                [
                    (f"{arm}_label", old_head["label"], new_head["label"]),
                    (
                        f"{arm}_control_passes",
                        old_head["control_passes"],
                        new_head["control_passes"],
                    ),
                    (f"{arm}_wrong_values", old_head["wrong_values"], new_head["wrong_values"]),
                    (f"{arm}_n_reverse", old_head["n_reverse"], new_head["n_reverse"]),
                    (f"{arm}_n_population", old_head["n_population"], new_head["n_population"]),
                    (f"{arm}_n_incomplete", old_head["n_incomplete"], new_head["n_incomplete"]),
                    (
                        f"{arm}_n_ids_divergent",
                        old_head["n_ids_divergent"],
                        new_head["n_ids_divergent"],
                    ),
                    (f"{arm}_n_non_answer", old_head["n_non_answer"], new_head["n_non_answer"]),
                    (f"{arm}_control_rate", old_head["control_rate"], new_head["control_rate"]),
                    (f"{arm}_arm_rate", old_head["arm_rate"], new_head["arm_rate"]),
                    (
                        f"{arm}_pair_exact_only",
                        old_head["pair_decomposition"]["exact_only"],
                        new_head["pair_decomposition"]["exact_only"],
                    ),
                    (
                        f"{arm}_pair_boundary_only",
                        old_head["pair_decomposition"]["boundary_only"],
                        new_head["pair_decomposition"]["boundary_only"],
                    ),
                    (
                        f"{arm}_pair_mixed",
                        old_head["pair_decomposition"]["mixed"],
                        new_head["pair_decomposition"]["mixed"],
                    ),
                    (
                        f"{arm}_n_ids_identical",
                        old_ids["n_ids_identical"],
                        new_ids["n_ids_identical"],
                    ),
                    (
                        f"{arm}_max_abs_id_delta",
                        old_ids["max_abs_id_delta"],
                        new_ids["max_abs_id_delta"],
                    ),
                    (
                        f"{arm}_n_same_ids_verdict_flips",
                        old_ids["n_same_ids_verdict_flips"],
                        new_ids["n_same_ids_verdict_flips"],
                    ),
                ]
            )
        for metric, old_value, new_value in cf_metrics:
            if old_value != new_value:
                cf_candidate_drift.append(
                    {"candidate": name, "metric": metric, "old": old_value, "new": new_value}
                )
    result["candidate_drift"] = cf_candidate_drift
    result["candidates_added"] = cf_candidates_added
    result["candidates_removed"] = cf_candidates_removed
    result["protocol_drift"] = cf_protocol_drift
    result["generator_drift"] = cf_generator_drift
    result["backend_drift"] = cf_backend_drift
    result["comparable"] = cf_comparable
    result["changed"] = bool(
        cf_candidate_drift
        or cf_candidates_added
        or cf_candidates_removed
        or cf_protocol_drift
        or cf_generator_drift
        or cf_backend_drift
    )
    return result


_ENUMERATIONCONVENTION_TOPICS = (
    "harbor",
    "summit",
    "quarry",
    "tundra",
    "granite",
    "willow",
    "copper",
    "saddle",
    "walnut",
    "velvet",
    "cobalt",
    "marble",
    "cinder",
    "falcon",
    "garnet",
    "saffron",
    "indigo",
    "juniper",
    "lagoon",
    "magnet",
    "nickel",
    "pepper",
    "quartz",
    "birch",
)


_ENUMERATIONCONVENTION_WORDS = (
    "alder",
    "basalt",
    "cedar",
    "drift",
    "elm",
    "fern",
    "glade",
    "heath",
    "iris",
    "jade",
    "kelp",
    "loom",
    "moss",
    "nova",
    "ochre",
    "pine",
    "quill",
    "reed",
    "sage",
    "tide",
    "urn",
    "vale",
    "wren",
    "yew",
)


_ENUMERATIONCONVENTION_NEUTRAL_STATEMENTS = (
    "The weather changed slowly during the afternoon hours.",
    "A wooden fence runs along the edge of the property.",
    "She placed the documents inside a plain folder yesterday.",
    "The train arrived later than the schedule suggested.",
    "Fresh bread was cooling on the counter near the sink.",
    "He repaired the bicycle with tools from the garage.",
    "The bookshop extends its opening hours during summer.",
    "Several chairs were arranged around the long table.",
    "The recipe calls for a pinch of salt and herbs.",
    "They walked along the path beside the lake at dawn.",
    "The committee postponed its decision until next week.",
    "A thick layer of dust covered the old furniture.",
    "The children played quietly in the courtyard after lunch.",
    "His handwriting was difficult to read in places.",
    "The farmer harvested the fields before the rain came.",
    "An old map hung framed on the wall of the hallway.",
    "The accountant reviewed the figures a second time.",
    "Birds gathered on the rooftops as the sun went down.",
    "The mechanic inspected the engine and changed the oil.",
    "A narrow staircase led up to the attic.",
    "The orchestra rehearsed the piece several times over.",
    "Someone left a bicycle leaning against the railing.",
    "The soup simmered gently on the stove all morning.",
    "Visitors signed the guest book near the entrance.",
)


_ENUMERATIONCONVENTION_MARKER_TABLE: dict[str, tuple[str, ...]] = {
    "control_arabic": ("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9."),
    "alpha_upper": ("A.", "B.", "C.", "D.", "E.", "F.", "G.", "H.", "I."),
    "alpha_lower": ("a.", "b.", "c.", "d.", "e.", "f.", "g.", "h.", "i."),
    "roman": ("I.", "II.", "III.", "IV.", "V.", "VI.", "VII.", "VIII.", "IX."),
    "wrapper": ("(1)", "(2)", "(3)", "(4)", "(5)", "(6)", "(7)", "(8)", "(9)"),
    "bullet": ("•", "•", "•", "•", "•", "•", "•", "•", "•"),
}


_ENUMERATIONCONVENTION_ARM_KEYS = (
    "control_arabic",
    "alpha_upper",
    "alpha_lower",
    "roman",
    "wrapper",
    "bullet",
)


_ENUMERATIONCONVENTION_TREATED_ARMS = _ENUMERATIONCONVENTION_ARM_KEYS[1:]


_ENUMERATIONCONVENTION_INDICES = tuple(range(1, 10))


_ENUMERATIONCONVENTION_STRIDE = 5


_ENUMERATIONCONVENTION_BULLET = "\u2022"


_ENUMERATIONCONVENTION_INTRO = "Inspection codes:"


_ENUMERATIONCONVENTION_CASE_COUNT = 24


_ENUMERATIONCONVENTION_MIN_D = 10


_ENUMERATIONCONVENTION_MIN_J = 6


_ENUMERATIONCONVENTION_MAX_INCOMPLETE = 6


_ENUMERATIONCONVENTION_REDRAW_LIMIT = 1000


_ENUMERATIONCONVENTION_GENERATOR_SPEC = (
    "draw(seed, index, slot, counter) = "
    "int.from_bytes(sha256(f'{seed}|enumerationconvention|{index}|{slot}|{counter}').digest()[:8], "
    "'big'); the material index is the 0-based case number; slots 2 and 3 at counter 0: "
    "neutral indices = draw % len(NEUTRAL_STATEMENTS) (slot 2 is the first neutral, slot 3 "
    "the second; the second advances +1 mod len on collision); slot 20 at counter 0: target "
    "position = 1 + draw % len(INDICES); list topics are the stride-5 base sequence B_j = "
    "TOPICS[(index + 5*j) % len(TOPICS)] for j = 0..8 with the target B_0 placed at the drawn "
    "position and B_1..B_8 in order at the remaining positions; item values are drawn per "
    "position t = 1..9 in order (word slot 30 + (t-1): WORDS[draw % len(WORDS)]; number slot "
    "40 + (t-1): 1000 + draw % 9000) with panel traversal in case-then-position order; a "
    "value colliding with any earlier value in the panel redraws that item's two slots "
    "together -- the item's word and number slots share one redraw counter -- and give-up "
    "raises when the counter exceeds 1000 (the stage fails closed)"
)


_ENUMERATIONCONVENTION_MAX_CANDIDATES = 24


_ENUMERATIONCONVENTION_MAX_GROUND_RECORDS = (
    _ENUMERATIONCONVENTION_MAX_CANDIDATES * _ENUMERATIONCONVENTION_CASE_COUNT
)


_ENUMERATIONCONVENTION_GENERATION_BUDGET = (
    len(_ENUMERATIONCONVENTION_ARM_KEYS) * _ENUMERATIONCONVENTION_MAX_GROUND_RECORDS
)


_ENUMERATIONCONVENTION_MAX_RECORDS_BYTES = 12 * 1024 * 1024


_ENUMERATIONCONVENTION_CELL_FIELDS = {
    "verdict",
    "boundary_found",
    "boundary_text_match",
    "input_tokens",
    "answer_digest",
    "answer_text",
}


_ENUMERATIONCONVENTION_DETAILS = ("arm_context_exceeded", "arm_evaluation_failed")


_ENUMERATIONCONVENTION_CELL_CLASSES = (
    "pass",
    "boundary_only",
    "wrong_value",
    "echo",
    "form_only",
    "non_answer",
)


_ENUMERATIONCONVENTION_PAIR_CLASSES = ("exact_only", "boundary_only", "mixed")


_ENUMERATIONCONVENTION_OPERATING_CHARACTERISTICS: dict[str, Any] = {
    "fire_competent_independent": 0.0173,
    "fire_competent_paired": 0.0044,
    "fire_weak_independent": 0.1743,
    "fire_random_independent": 0.5503,
    "power_drop5_base90": 0.0947,
    "power_drop10_base90": 0.2542,
    "power_drop25_base90": 0.8159,
    "power_drop25_base85": 0.8761,
    "power_drop25_paired": 0.6981,
    "power_drop50_base90": 0.999,
    "sensitivity18_drop25": 0.5214,
    "sensitivity18_competent": 0.0038,
    "sensitivity12_drop25": 0.1379,
    "sensitivity12_competent": 0.0003,
    "transplant_fire_competent_independent": 0.8897,
    "familywise_competent_expected_120": 2.0729,
    "familywise_competent_any_120": 0.8764,
    "familywise_paired_expected_120": 0.5331,
    "familywise_paired_any_120": 0.4139,
    "familywise_random_expected_120": 66.0368,
    "familywise_random_any_120": 1.0,
    "world_a1_guarded_fire": 0.0055,
    "world_a1_unguarded_fire": 0.0186,
    "world_a1_guarded_insufficient": 0.9923,
    "world_a2_guarded_fire": 0.0173,
    "world_a2_unguarded_fire": 0.0173,
    "world_a2_guarded_insufficient": 0.2897,
    "world_b_unguarded_fire": 0.2551,
    "world_b_unguarded_below_threshold": 0.7277,
    "world_b_guarded_fire": 0.1731,
    "world_b_guarded_below_threshold": 0.0117,
    "world_b_guarded_insufficient": 0.8152,
    "world_b777_guarded_below_threshold": 0.011,
    "world_c_guarded_fire": 0.8893,
    "world_c_unguarded_fire": 0.9399,
    "world_c_guarded_below_threshold": 0.0016,
    "required_n_smallest": 117,
    "required_n_min_d": 0,
    "required_n_min_j": 18,
    "required_n_null": 0.0172,
    "required_n_power10": 0.8023,
    "required_n_grid120_power10": 0.7657,
    "required_n_grid126_power10": 0.8336,
    "required_n_grid132_power10": 0.8325,
    "required_n_cost": 16848,
}


_ENUMERATIONCONVENTION_AGGREGATE_KEYS = {
    "arms",
    "ids",
    "positions",
    "headline",
    "operating_characteristics",
    "n_cases",
    "n_incomplete",
}


_ENUMERATIONCONVENTION_ROMAN_RE = re.compile(
    r"M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})"
)


_ENUMERATIONCONVENTION_ROMAN_VAL = {
    "I": 1,
    "V": 5,
    "X": 10,
    "L": 50,
    "C": 100,
    "D": 500,
    "M": 1000,
}


_ENUMERATIONCONVENTION_ROMAN_TABLE = {
    1: "I",
    2: "II",
    3: "III",
    4: "IV",
    5: "V",
    6: "VI",
    7: "VII",
    8: "VIII",
    9: "IX",
}


def _enumerationconvention_digest(text: str) -> str:
    """SHA-256 hex digest (local copy; no imports from
    enumerationconvention.py anywhere in this mirror).
    """

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _enumerationconvention_text_digest(text: str) -> str | None:
    """SHA-256 of the UTF-8 encoding, or None when the
    text cannot be encoded (a lone-surrogate answer_text
    is a validation failure, never an exception — the
    cycle-190 A3 pattern, applied per audit F2).
    """
    try:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    except UnicodeEncodeError:
        return None


def _enumerationconvention_digest_ok(value: Any) -> bool:
    """Whether a stored digest is a well-formed SHA-256 hex
    digest.
    """
    if type(value) is not str or len(value) != 64:
        return False
    return all(ch in "0123456789abcdef" for ch in value)


def _enumerationconvention_pool_digests() -> dict[str, str]:
    """The pool digests recomputed from this mirror's own
    literal pools.
    """
    return {
        "topics": _enumerationconvention_digest("\n".join(_ENUMERATIONCONVENTION_TOPICS)),
        "words": _enumerationconvention_digest("\n".join(_ENUMERATIONCONVENTION_WORDS)),
        "neutral_statements": _enumerationconvention_digest(
            "\n".join(_ENUMERATIONCONVENTION_NEUTRAL_STATEMENTS)
        ),
    }


def _enumerationconvention_derive(seed: int, index: int, slot: int, counter: int) -> int:
    """Mirror of enumerationconvention._derive (the draw
    conversion pinned by the design's generator spec).
    """

    material = f"{seed}|enumerationconvention|{index}|{slot}|{counter}"
    return int.from_bytes(hashlib.sha256(material.encode("ascii")).digest()[:8], "big")


class _EnumerationConventionMirrorExhausted(Exception):
    """Mirror-internal: the redraw give-up bound was reached
    during regeneration, so the payload cannot be recomputed
    and validates false.
    """


def _enumerationconvention_generate(seed: int) -> list[dict[str, Any]]:
    """Mirror of enumerationconvention.generate_cases: the
    frozen SHA-256 derivation with panel-wide value
    distinctness by redraw.
    """
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index in range(_ENUMERATIONCONVENTION_CASE_COUNT):
        first = _enumerationconvention_derive(seed, index, 2, 0) % len(
            _ENUMERATIONCONVENTION_NEUTRAL_STATEMENTS
        )
        second = _enumerationconvention_derive(seed, index, 3, 0) % len(
            _ENUMERATIONCONVENTION_NEUTRAL_STATEMENTS
        )
        if second == first:
            second = (first + 1) % len(_ENUMERATIONCONVENTION_NEUTRAL_STATEMENTS)
        position = 1 + _enumerationconvention_derive(seed, index, 20, 0) % len(
            _ENUMERATIONCONVENTION_INDICES
        )
        base = [
            _ENUMERATIONCONVENTION_TOPICS[
                (index + _ENUMERATIONCONVENTION_STRIDE * j) % len(_ENUMERATIONCONVENTION_TOPICS)
            ]
            for j in range(len(_ENUMERATIONCONVENTION_INDICES))
        ]
        topics: list[str] = []
        rest = iter(base[1:])
        for t in _ENUMERATIONCONVENTION_INDICES:
            topics.append(base[0] if t == position else next(rest))
        values: list[str] = []
        for t in _ENUMERATIONCONVENTION_INDICES:
            counter = 0
            while True:
                word = _ENUMERATIONCONVENTION_WORDS[
                    _enumerationconvention_derive(seed, index, 30 + (t - 1), counter)
                    % len(_ENUMERATIONCONVENTION_WORDS)
                ]
                number = (
                    1000 + _enumerationconvention_derive(seed, index, 40 + (t - 1), counter) % 9000
                )
                value = f"{word}-{number}"
                if value not in seen:
                    break
                counter += 1
                if counter > _ENUMERATIONCONVENTION_REDRAW_LIMIT:
                    raise _EnumerationConventionMirrorExhausted
            seen.add(value)
            values.append(value)
        cases.append(
            {
                "fixture_id": f"enumerationconvention-{index:02d}",
                "index": index,
                "topic": _ENUMERATIONCONVENTION_TOPICS[index],
                "position": position,
                "topics": topics,
                "values": values,
                "expected": values[position - 1],
                "question": (
                    f"What is the {_ENUMERATIONCONVENTION_TOPICS[index]} inspection code?"
                ),
                "neutral_indices": [first, second],
            }
        )
    return cases


def _enumerationconvention_item_sentence(topic: str, value: str) -> str:
    """Mirror of enumerationconvention.item_sentence."""
    return f"The {topic} inspection code is {value}."


def _enumerationconvention_question(case: dict[str, Any]) -> str:
    """Mirror of enumerationconvention.question_text."""
    return f"What is the {case['topic']} inspection code?"


def _enumerationconvention_list_block(case: dict[str, Any], arm: str) -> str:
    """Mirror of enumerationconvention.list_block."""
    if arm not in _ENUMERATIONCONVENTION_MARKER_TABLE:
        raise ValueError(f"unknown enumeration-convention arm: {arm!r}")
    lines = [
        f"{_ENUMERATIONCONVENTION_MARKER_TABLE[arm][t - 1]} "
        f"{_enumerationconvention_item_sentence(case['topics'][t - 1], case['values'][t - 1])}"
        for t in _ENUMERATIONCONVENTION_INDICES
    ]
    return _ENUMERATIONCONVENTION_INTRO + "\n" + "\n".join(lines)


def _enumerationconvention_arm_content(case: dict[str, Any], arm: str) -> str:
    """Mirror of enumerationconvention.arm_content."""
    if arm not in _ENUMERATIONCONVENTION_MARKER_TABLE:
        raise ValueError(f"unknown enumeration-convention arm: {arm!r}")
    first, second = (int(i) for i in case["neutral_indices"])
    return (
        f"{_ENUMERATIONCONVENTION_NEUTRAL_STATEMENTS[first]} "
        f"{_enumerationconvention_list_block(case, arm)} "
        f"{_ENUMERATIONCONVENTION_NEUTRAL_STATEMENTS[second]} "
        f"{_enumerationconvention_question(case)}"
    )


def _enumerationconvention_case_material_sha256() -> str:
    """Mirror of enumerationconvention.case_material_sha256."""

    material = {
        "topics": list(_ENUMERATIONCONVENTION_TOPICS),
        "words": list(_ENUMERATIONCONVENTION_WORDS),
        "neutral_statements": list(_ENUMERATIONCONVENTION_NEUTRAL_STATEMENTS),
        "arm_keys": list(_ENUMERATIONCONVENTION_ARM_KEYS),
        "marker_table": {
            arm: list(_ENUMERATIONCONVENTION_MARKER_TABLE[arm])
            for arm in _ENUMERATIONCONVENTION_ARM_KEYS
        },
        "generator_spec": _ENUMERATIONCONVENTION_GENERATOR_SPEC,
        "case_count": _ENUMERATIONCONVENTION_CASE_COUNT,
        "indices": list(_ENUMERATIONCONVENTION_INDICES),
        "stride": _ENUMERATIONCONVENTION_STRIDE,
        "pool_digests": _enumerationconvention_pool_digests(),
        "min_d": _ENUMERATIONCONVENTION_MIN_D,
        "min_j": _ENUMERATIONCONVENTION_MIN_J,
        "max_incomplete": _ENUMERATIONCONVENTION_MAX_INCOMPLETE,
        "redraw_limit": _ENUMERATIONCONVENTION_REDRAW_LIMIT,
    }
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return _enumerationconvention_digest(canonical)


def _enumerationconvention_ground_sha256(seed: int) -> str:
    """Mirror of enumerationconvention.ground_sha256."""

    rows = [
        [
            case["fixture_id"],
            case["topic"],
            case["position"],
            list(case["topics"]),
            list(case["values"]),
            list(case["neutral_indices"]),
        ]
        for case in _enumerationconvention_generate(seed)
    ]
    canonical = json.dumps(rows, separators=(",", ":"), ensure_ascii=True)
    return _enumerationconvention_digest(canonical)


def _enumerationconvention_protocol(seed: int, backend_fingerprint: str) -> dict[str, Any]:
    """The enumeration-convention protocol rebuilt from
    this module's own literals, fingerprint included
    (compared whole).
    """

    protocol: dict[str, Any] = {
        "version": 1,
        "arm_keys": list(_ENUMERATIONCONVENTION_ARM_KEYS),
        "treated_arms": list(_ENUMERATIONCONVENTION_TREATED_ARMS),
        "case_count": _ENUMERATIONCONVENTION_CASE_COUNT,
        "min_d": _ENUMERATIONCONVENTION_MIN_D,
        "min_j": _ENUMERATIONCONVENTION_MIN_J,
        "max_incomplete": _ENUMERATIONCONVENTION_MAX_INCOMPLETE,
        "case_material_sha256": _enumerationconvention_case_material_sha256(),
        "ground_sha256": _enumerationconvention_ground_sha256(seed),
        "pool_digests": _enumerationconvention_pool_digests(),
        "generation_budget": _ENUMERATIONCONVENTION_GENERATION_BUDGET,
        "max_ground_records": _ENUMERATIONCONVENTION_MAX_GROUND_RECORDS,
        "max_records_bytes": _ENUMERATIONCONVENTION_MAX_RECORDS_BYTES,
        "max_candidates": _ENUMERATIONCONVENTION_MAX_CANDIDATES,
        "seed": seed,
        "backend_fingerprint": backend_fingerprint,
    }
    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    protocol["fingerprint"] = _enumerationconvention_digest(canonical)
    return protocol


def _enumerationconvention_cell_get(record: dict[str, Any], key: str) -> dict[str, Any] | None:
    """Return the enumerationconvention cell get."""
    cells = record.get("cells")
    if not isinstance(cells, dict):
        return None
    cell = cells.get(key)
    return cell if isinstance(cell, dict) else None


def _enumerationconvention_detail_get(record: dict[str, Any], key: str) -> str:
    """Return the enumerationconvention detail get."""
    details = record.get("cell_details")
    if not isinstance(details, dict):
        return ""
    detail = details.get(key, "")
    return detail if type(detail) is str else ""


def _enumerationconvention_verdict(record: dict[str, Any], key: str) -> bool:
    """Return the enumerationconvention verdict."""
    cell = _enumerationconvention_cell_get(record, key)
    return bool(cell is not None and cell.get("verdict") is True)


def _enumerationconvention_ids_identical(record: dict[str, Any], arm: str) -> bool:
    """Mirror of enumerationconvention._ids_identical:
    derived from the recorded id digests alone.
    """
    digests = record.get("id_digests")
    if not isinstance(digests, dict):
        return False
    control = digests.get("control_arabic")
    return type(control) is str and control != "" and control == digests.get(arm)


def _enumerationconvention_value_ok(value: Any) -> bool:
    """The frozen value grammar, fullmatch semantics
    (mirror of enumerationconvention._value_grammar_ok).
    """
    if type(value) is not str or not re.fullmatch(r"[a-z]+-[0-9]{4}", value):
        return False
    word, _, number = value.partition("-")
    return word in _ENUMERATIONCONVENTION_WORDS and 1000 <= int(number) <= 9999


def _enumerationconvention_is_ascii_digits(text: str) -> bool:
    """Whether every character is an ASCII digit (never
    Unicode-lenient: amendment A2).
    """
    return bool(text) and all("0" <= char <= "9" for char in text)


def _enumerationconvention_is_ascii_letters(text: str) -> bool:
    """Whether every character is an ASCII letter."""
    return bool(text) and all("A" <= char <= "Z" or "a" <= char <= "z" for char in text)


def _enumerationconvention_parse_roman(glyph: str) -> int | None:
    """Mirror of enumerationconvention.parse_roman:
    subtractive parse of a canonical Roman numeral, or
    None. ASCII-pinned (amendment A2).
    """
    if not glyph or not _enumerationconvention_is_ascii_letters(glyph):
        return None
    if glyph.upper() != glyph and glyph.lower() != glyph:
        return None  # mixed case is not a canonical rendering
    upper = glyph.upper()
    if any(char not in _ENUMERATIONCONVENTION_ROMAN_VAL for char in upper):
        return None
    if not _ENUMERATIONCONVENTION_ROMAN_RE.fullmatch(upper):
        return None
    total, prev = 0, 0
    for char in reversed(upper):
        value = _ENUMERATIONCONVENTION_ROMAN_VAL[char]
        total += -value if value < prev else value
        prev = max(prev, value)
    return total


def _enumerationconvention_glyph_ok(glyph: str) -> bool:
    """Mirror of enumerationconvention.glyph_ok."""
    if _enumerationconvention_is_ascii_digits(glyph):
        return len(glyph) <= 2
    if not _enumerationconvention_is_ascii_letters(glyph):
        return False
    if len(glyph) == 1:
        return True
    if glyph.upper() != glyph and glyph.lower() != glyph:
        return False  # mixed case is not a canonical rendering
    return _enumerationconvention_parse_roman(glyph) is not None


def _enumerationconvention_split_token(head: str) -> tuple[str, str] | None:
    """Mirror of enumerationconvention.split_token.
    Digit tests are ASCII-only (amendment A2).
    """
    if head == _ENUMERATIONCONVENTION_BULLET:
        return ("bullet", _ENUMERATIONCONVENTION_BULLET)
    match = re.fullmatch(r"\(([0-9]{1,2})\)", head)
    if match:
        return ("wrapper-parens", match.group(1))
    match = re.fullmatch(r"\[([^\]]{1,6})\]", head)
    if match and _enumerationconvention_glyph_ok(match.group(1)):
        return ("bracket", match.group(1))
    match = re.fullmatch(r"(.{1,6})([.)])", head)
    if match and _enumerationconvention_glyph_ok(match.group(1)):
        return ("close-paren" if match.group(2) == ")" else "dotted", match.group(1))
    return None


def _enumerationconvention_is_echo(head: str, arm: str, target: int) -> bool:
    """Mirror of enumerationconvention.is_echo: string
    identity with the arm's own marker (amendment A3).
    """
    if arm not in _ENUMERATIONCONVENTION_MARKER_TABLE or (
        target not in _ENUMERATIONCONVENTION_INDICES
    ):
        return False
    marker = _ENUMERATIONCONVENTION_MARKER_TABLE[arm][target - 1]
    if arm == "roman":
        return head in (marker, _ENUMERATIONCONVENTION_ROMAN_TABLE[target].lower() + ".")
    return head == marker


def _enumerationconvention_strip_one(text: str, where: str) -> str | None:
    """Mirror of enumerationconvention.strip_one."""
    if where == "lead":
        if " " not in text:
            return None
        head, tail = text.split(" ", 1)
        return tail if _enumerationconvention_split_token(head) is not None else None
    if " " not in text:
        return None
    body, tail = text.rsplit(" ", 1)
    return body if _enumerationconvention_split_token(tail) is not None else None


def _enumerationconvention_decompose_cell(
    cell: dict[str, Any], expected: str, expected_digest: str, arm: str, target: int
) -> str:
    """Mirror of enumerationconvention.decompose_cell (the
    ordered decision list, amendments A1-A3 included).
    """
    if cell.get("verdict") is True:
        return "pass"
    digest = cell.get("answer_digest")
    if digest is not None and digest == expected_digest:
        return "boundary_only"
    text = cell.get("answer_text")
    if type(text) is not str:
        return "non_answer"
    if text != expected:
        if _enumerationconvention_value_ok(text):
            return "wrong_value"
        for where in ("lead", "trail"):
            remainder = _enumerationconvention_strip_one(text, where)
            if (
                remainder is not None
                and remainder != expected
                and _enumerationconvention_value_ok(remainder)
            ):
                return "wrong_value"
    lead = _enumerationconvention_strip_one(text, "lead")
    if lead is not None and lead == expected:
        head = text.split(" ", 1)[0]
        return "echo" if _enumerationconvention_is_echo(head, arm, target) else "form_only"
    if _enumerationconvention_strip_one(text, "trail") == expected:
        return "form_only"  # trailing placement is never the arm marker
    return "non_answer"


def _enumerationconvention_decompose_pair(
    control_cell: dict[str, Any], arm_cell: dict[str, Any]
) -> str:
    """Mirror of enumerationconvention.decompose_pair."""
    if control_cell.get("answer_digest") == arm_cell.get("answer_digest"):
        return "boundary_only"
    if control_cell.get("boundary_found") and arm_cell.get("boundary_found"):
        return "exact_only"
    return "mixed"


def _enumerationconvention_classify(
    control_passes: int, wrong_values: int, n_incomplete: int
) -> str:
    """Mirror of enumerationconvention.classify_headline."""
    if n_incomplete > _ENUMERATIONCONVENTION_MAX_INCOMPLETE:
        return "insufficient"
    if control_passes < _ENUMERATIONCONVENTION_MIN_D:
        return "insufficient"
    if wrong_values >= _ENUMERATIONCONVENTION_MIN_J:
        return "discordant_asymmetric"
    if wrong_values == 0:
        return "no_discordance_observed" if n_incomplete == 0 else "insufficient"
    return (
        "below_threshold"
        if wrong_values + n_incomplete < _ENUMERATIONCONVENTION_MIN_J
        else "insufficient"
    )


def _enumerationconvention_rate(numerator: int, denominator: int) -> float | None:
    """Return the enumerationconvention rate."""
    return (numerator / denominator) if denominator else None


def _enumerationconvention_aggregate(
    records: list[dict[str, Any]], cases: list[dict[str, Any]]
) -> dict[str, Any]:
    """Mirror of enumerationconvention.aggregate_candidate."""
    case_by_fixture = {str(case["fixture_id"]): case for case in cases}
    expected_by_fixture = {
        fixture: str(case["expected"]) for fixture, case in case_by_fixture.items()
    }
    arms: dict[str, Any] = {}
    for key in _ENUMERATIONCONVENTION_ARM_KEYS:
        n_completed = 0
        n_pass = 0
        n_context = 0
        n_failed = 0
        decomposition = dict.fromkeys(_ENUMERATIONCONVENTION_CELL_CLASSES, 0)
        for record in records:
            detail = _enumerationconvention_detail_get(record, key)
            if detail == "arm_context_exceeded":
                n_context += 1
            elif detail == "arm_evaluation_failed":
                n_failed += 1
            cell = _enumerationconvention_cell_get(record, key)
            if cell is None:
                continue
            n_completed += 1
            if cell.get("verdict") is True:
                n_pass += 1
            case = case_by_fixture.get(str(record.get("fixture", "")))
            target = int(case["position"]) if case is not None else 0
            decomposition[
                _enumerationconvention_decompose_cell(
                    cell,
                    expected_by_fixture.get(str(record.get("fixture", "")), ""),
                    str(record.get("expected_digest", "")),
                    key,
                    target,
                )
            ] += 1
        arms[key] = {
            "n_completed": n_completed,
            "n_pass": n_pass,
            "rate": _enumerationconvention_rate(n_pass, n_completed),
            "n_context_exceeded": n_context,
            "n_evaluation_failed": n_failed,
            "decomposition": decomposition,
        }
    ids: dict[str, Any] = {}
    for arm in _ENUMERATIONCONVENTION_TREATED_ARMS:
        deltas: list[int] = []
        n_ids_identical = 0
        n_same_ids_flips = 0
        for record in records:
            tokens = record.get("arm_tokens")
            tokens = tokens if isinstance(tokens, dict) else {}
            treated = tokens.get(arm)
            baseline = tokens.get("control_arabic")
            if type(treated) is int and type(baseline) is int:
                deltas.append(treated - baseline)
            else:
                deltas.append(0)
            if _enumerationconvention_ids_identical(record, arm):
                n_ids_identical += 1
                control_cell = _enumerationconvention_cell_get(record, "control_arabic")
                arm_cell = _enumerationconvention_cell_get(record, arm)
                if (
                    control_cell is not None
                    and arm_cell is not None
                    and (control_cell.get("verdict") is True) != (arm_cell.get("verdict") is True)
                ):
                    n_same_ids_flips += 1
        ids[arm] = {
            "n_ids_identical": n_ids_identical,
            "n_ids_divergent": len(records) - n_ids_identical,
            "id_length_delta": deltas,
            "max_abs_id_delta": max((abs(delta) for delta in deltas), default=0),
            "n_same_ids_verdict_flips": n_same_ids_flips,
        }
    headline: dict[str, Any] = {}
    positions: dict[str, Any] = {}
    for arm in _ENUMERATIONCONVENTION_TREATED_ARMS:
        domain = [
            record for record in records if not _enumerationconvention_ids_identical(record, arm)
        ]
        paired = [
            record
            for record in domain
            if _enumerationconvention_cell_get(record, "control_arabic") is not None
            and _enumerationconvention_cell_get(record, arm) is not None
        ]
        n_incomplete = len(domain) - len(paired)
        control_passes = 0
        wrong_values = 0
        reverse = 0
        non_answer = 0
        pair_decomposition = dict.fromkeys(_ENUMERATIONCONVENTION_PAIR_CLASSES, 0)
        failing_classes = dict.fromkeys(_ENUMERATIONCONVENTION_CELL_CLASSES, 0)
        by_position: dict[int, dict[str, Any]] = {
            t: {"n_population": 0, "control_passes": 0, "wrong_values": 0, "deltas": []}
            for t in _ENUMERATIONCONVENTION_INDICES
        }
        for record in paired:
            control_cell = _enumerationconvention_cell_get(record, "control_arabic")
            arm_cell = _enumerationconvention_cell_get(record, arm)
            if control_cell is None or arm_cell is None:
                continue
            case = case_by_fixture.get(str(record.get("fixture", "")))
            target = int(case["position"]) if case is not None else 0
            expected = expected_by_fixture.get(str(record.get("fixture", "")), "")
            expected_digest = str(record.get("expected_digest", ""))
            arm_class = _enumerationconvention_decompose_cell(
                arm_cell, expected, expected_digest, arm, target
            )
            if arm_class == "non_answer":
                non_answer += 1
            control_verdict = control_cell.get("verdict") is True
            arm_verdict = arm_cell.get("verdict") is True
            slot = by_position.get(target)
            if slot is not None:
                slot["n_population"] += 1
                tokens = record.get("arm_tokens")
                tokens = tokens if isinstance(tokens, dict) else {}
                treated_tokens = tokens.get(arm)
                baseline_tokens = tokens.get("control_arabic")
                if type(treated_tokens) is int and type(baseline_tokens) is int:
                    slot["deltas"].append(treated_tokens - baseline_tokens)
                if control_verdict:
                    slot["control_passes"] += 1
                    if arm_class == "wrong_value":
                        slot["wrong_values"] += 1
            if control_verdict:
                control_passes += 1
                if arm_class == "wrong_value":
                    wrong_values += 1
            elif arm_verdict:
                reverse += 1
            if control_verdict != arm_verdict:
                pair_decomposition[
                    _enumerationconvention_decompose_pair(control_cell, arm_cell)
                ] += 1
                failing = arm_cell if control_verdict else control_cell
                failing_arm = arm if control_verdict else "control_arabic"
                failing_classes[
                    _enumerationconvention_decompose_cell(
                        failing, expected, expected_digest, failing_arm, target
                    )
                ] += 1
        headline[arm] = {
            "label": _enumerationconvention_classify(control_passes, wrong_values, n_incomplete),
            "n_population": len(paired),
            "n_incomplete": n_incomplete,
            "n_ids_divergent": len(domain),
            "control_passes": control_passes,
            "wrong_values": wrong_values,
            "n_reverse": reverse,
            "control_rate": _enumerationconvention_rate(control_passes, len(paired)),
            "arm_rate": _enumerationconvention_rate(
                sum(1 for record in paired if _enumerationconvention_verdict(record, arm)),
                len(paired),
            ),
            "n_non_answer": non_answer,
            "pair_decomposition": pair_decomposition,
            "failing_classes": failing_classes,
        }
        positions[arm] = [
            {
                "position": t,
                "n_population": by_position[t]["n_population"],
                "control_passes": by_position[t]["control_passes"],
                "wrong_values": by_position[t]["wrong_values"],
                "mean_id_length_delta": _enumerationconvention_rate(
                    sum(by_position[t]["deltas"]), len(by_position[t]["deltas"])
                ),
            }
            for t in _ENUMERATIONCONVENTION_INDICES
        ]
    return {
        "arms": arms,
        "ids": ids,
        "positions": positions,
        "headline": headline,
        "operating_characteristics": dict(_ENUMERATIONCONVENTION_OPERATING_CHARACTERISTICS),
        "n_cases": len(records),
        "n_incomplete": sum(
            1
            for record in records
            if any(
                _enumerationconvention_cell_get(record, arm) is None
                for arm in _ENUMERATIONCONVENTION_ARM_KEYS
            )
        ),
    }


def _enumerationconvention_cell_ok(cell: Any, expected_digest: str) -> bool:
    """One cell's shape and bindings: the exact field set;
    the verdict binding ``verdict == (answer_digest ==
    expected_digest) and boundary_found``; the carried
    ``answer_text`` must hash to its ``answer_digest`` (the
    cycle-190 binding); the input-token binding to the
    record's arm token count is checked by the caller.
    """
    if type(cell) is not dict or set(cell) != _ENUMERATIONCONVENTION_CELL_FIELDS:
        return False
    if type(cell["verdict"]) is not bool or type(cell["boundary_found"]) is not bool:
        return False
    if cell["boundary_text_match"] is not None and type(cell["boundary_text_match"]) is not bool:
        return False
    if type(cell["input_tokens"]) is not int or cell["input_tokens"] < 0:
        return False
    digest = cell["answer_digest"]
    if digest is not None and not _enumerationconvention_digest_ok(digest):
        return False
    text = cell["answer_text"]
    if text is None:
        if digest is not None:
            return False
    elif type(text) is not str or _enumerationconvention_text_digest(text) != digest:
        return False
    verdict = (digest == expected_digest) and cell["boundary_found"]
    return bool(cell["verdict"] == verdict)


def _enumerationconvention_recompute(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Recompute an enumeration-convention payload from its
    ground records.

    The validator carries its own generator mirror with the
    frozen SHA-256 derivation, the frozen marker table, and
    the aggregators (no imports from
    enumerationconvention.py): the protocol is rebuilt from
    frozen literals plus the payload's seed and backend
    fingerprint and compared whole; the case set is
    regenerated from the seed and every record is checked
    against it in protocol order; every record's per-arm
    content digests are rebound to the regenerated arm
    contents, so the treatment itself is committed; every
    record's target position is rebound to the regenerated
    case's drawn position; the record's ids_identical flags
    and id_length_delta values must equal the derivations
    from its recorded digests and token counts; every cell's
    answer_text is verified against its answer_digest, its
    verdict is re-derived from its digest identity, and its
    input_tokens are bound to the record's arm token count;
    the decompositions, id report, per-position join, and
    guarded headlines are recomputed from the records and
    the regenerated expected values under the pinned
    completion domain; the aggregates are recomputed and
    compared. Cells exist only from aggregate-eligible
    reports, so a non-null cell certifies a completed
    evaluation and the verdict's status conjunct is
    discharged structurally.
    """

    protocol = payload.get("protocol")
    if type(protocol) is not dict:
        return None
    seed = protocol.get("seed")
    backend_fingerprint = protocol.get("backend_fingerprint")
    if type(seed) is not int or isinstance(seed, bool) or type(backend_fingerprint) is not str:
        return None
    try:
        expected_protocol = _enumerationconvention_protocol(seed, backend_fingerprint)
    except _EnumerationConventionMirrorExhausted:
        return None
    if not _recomputed_equal(protocol, expected_protocol):
        return None
    candidates = payload.get("candidates")
    if type(candidates) is not dict or not candidates:
        return None
    if len(candidates) > _ENUMERATIONCONVENTION_MAX_CANDIDATES:
        return None
    try:
        generated = _enumerationconvention_generate(seed)
    except _EnumerationConventionMirrorExhausted:
        return None
    total_records = 0
    records_bytes = 0
    aggregates: dict[str, Any] = {}
    for name, entry in candidates.items():
        if type(entry) is not dict:
            return None
        status = entry.get("status")
        if status == "unavailable":
            if set(entry) != {"status", "detail", "records"} or entry.get("records") != []:
                return None
            continue
        if status != "available" or entry.get("detail") != "":
            return None
        if set(entry) != _ENUMERATIONCONVENTION_AGGREGATE_KEYS | {"status", "detail", "records"}:
            return None
        records = entry["records"]
        if type(records) is not list or len(records) != len(generated):
            return None
        total_records += len(records)
        records_bytes += len(
            json.dumps(records, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        )
        if records_bytes > _ENUMERATIONCONVENTION_MAX_RECORDS_BYTES:
            return None
        for record, case in zip(records, generated, strict=True):
            if type(record) is not dict or set(record) != {
                "fixture",
                "topic",
                "target_position",
                "expected_digest",
                "content_digests",
                "arm_tokens",
                "id_digests",
                "ids_identical",
                "id_length_delta",
                "cells",
                "cell_details",
            }:
                return None
            if record["fixture"] != case["fixture_id"]:
                return None
            if record["topic"] != case["topic"]:
                return None
            if record["target_position"] != case["position"]:
                return None
            expected_digest = _enumerationconvention_digest(str(case["expected"]))
            if record["expected_digest"] != expected_digest:
                return None
            arm_tokens = record["arm_tokens"]
            id_digests = record["id_digests"]
            content_digests = record["content_digests"]
            ids_identical = record["ids_identical"]
            id_length_delta = record["id_length_delta"]
            cells = record["cells"]
            details = record["cell_details"]
            for mapping in (arm_tokens, id_digests, content_digests, cells, details):
                if type(mapping) is not dict or set(mapping) != set(
                    _ENUMERATIONCONVENTION_ARM_KEYS
                ):
                    return None
            for mapping in (ids_identical, id_length_delta):
                if type(mapping) is not dict or set(mapping) != set(
                    _ENUMERATIONCONVENTION_TREATED_ARMS
                ):
                    return None
            # Validate every arm_tokens value BEFORE any arithmetic on
            # them, exactly as in _invisiblechars_recompute: a
            # type-confused value escaped as a raw TypeError while the
            # type check sat in the later per-arm loop (bug-hunt
            # wave 2, BH2B-02).
            for arm in _ENUMERATIONCONVENTION_ARM_KEYS:
                tokens = arm_tokens[arm]
                if type(tokens) is not int or isinstance(tokens, bool) or tokens < 0:
                    return None
            for arm in _ENUMERATIONCONVENTION_TREATED_ARMS:
                if ids_identical[arm] is not (id_digests[arm] == id_digests["control_arabic"]):
                    return None
                if id_length_delta[arm] != arm_tokens[arm] - arm_tokens["control_arabic"]:
                    return None
            for arm in _ENUMERATIONCONVENTION_ARM_KEYS:
                tokens = arm_tokens[arm]
                if type(tokens) is not int or isinstance(tokens, bool) or tokens < 0:
                    return None
                if not _enumerationconvention_digest_ok(id_digests[arm]):
                    return None
                if content_digests[arm] != _enumerationconvention_digest(
                    _enumerationconvention_arm_content(case, arm)
                ):
                    return None
                cell = cells[arm]
                detail = details[arm]
                if detail not in ("", *_ENUMERATIONCONVENTION_DETAILS):
                    return None
                if cell is None:
                    if detail == "":
                        return None
                    continue
                if detail != "":
                    return None
                if not _enumerationconvention_cell_ok(cell, expected_digest):
                    return None
                if cell["input_tokens"] != tokens:
                    return None
        aggregates[name] = _enumerationconvention_aggregate(records, generated)
    if total_records > _ENUMERATIONCONVENTION_MAX_GROUND_RECORDS:
        return None
    if not aggregates:
        return None
    return {
        "aggregates": aggregates,
        "n_candidates": len(candidates),
        "n_cases": len(generated),
    }


def _valid_enumeration_convention(payload: Any) -> bool:
    """Structural check for an ``enumeration_convention``
    payload (recompute standard). The audit block is
    shape-checked only: its counts are stage observations
    (special-token findings against the run's own formats),
    not recomputable from records.
    """
    if type(payload) is not dict or payload.get("available") is not True:
        return False
    if set(payload) != {
        "available",
        "detail",
        "protocol",
        "audit",
        "candidates",
        "n_candidates",
        "n_cases",
    }:
        return False
    if payload.get("detail") != "":
        return False
    audit = payload.get("audit")
    if type(audit) is not dict or set(audit) != {
        "n_audit_excluded",
        "by_reason",
        "special_token_findings",
    }:
        return False
    if type(audit["n_audit_excluded"]) is not int or audit["n_audit_excluded"] < 0:
        return False
    if type(audit["special_token_findings"]) is not int or audit["special_token_findings"] < 0:
        return False
    by_reason = audit["by_reason"]
    if type(by_reason) is not dict:
        return False
    for key, count in by_reason.items():
        if type(key) is not str or type(count) is not int or count < 0:
            return False
    recomputed = _enumerationconvention_recompute(payload)
    if recomputed is None:
        return False
    candidates = payload["candidates"]
    for name, expected in recomputed["aggregates"].items():
        entry = candidates[name]
        for key, value in expected.items():
            if not _recomputed_equal(entry.get(key), value):
                return False
    if not _recomputed_equal(payload.get("n_candidates"), recomputed["n_candidates"]):
        return False
    return _recomputed_equal(payload.get("n_cases"), recomputed["n_cases"])


def diff_enumeration_convention(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``enumeration_convention`` probe payloads;
    deterministic.

    Answers "did the enumeration-convention story change
    between runs?": per candidate and treated arm, drift in
    the guarded label with the D / J counts that carry it,
    the divergent-domain size M, the reverse and non-answer
    counts, the paired population and its incompletion, the
    id report, the per-position counts and mean length
    deltas, and the pair decomposition; candidates
    added/removed; and protocol / generator / backend drift
    (any of which marks the comparison not comparable, with
    the drift still listed -- the generated panel is the
    treatment, so a changed generator is protocol drift). A
    side without a valid payload is reported plainly.
    """
    old_ok = _valid_enumeration_convention(old)
    new_ok = _valid_enumeration_convention(new)
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
    ec_protocol_drift = old["protocol"].get("fingerprint") != new["protocol"].get("fingerprint")
    ec_generator_drift = old["protocol"].get("ground_sha256") != new["protocol"].get(
        "ground_sha256"
    ) or old["protocol"].get("case_material_sha256") != new["protocol"].get("case_material_sha256")
    ec_backend_drift = old["protocol"].get("backend_fingerprint") != new["protocol"].get(
        "backend_fingerprint"
    )
    ec_comparable = not ec_protocol_drift and not ec_generator_drift and not ec_backend_drift
    old_candidates = old["candidates"]
    new_candidates = new["candidates"]
    ec_candidates_added = sorted(set(new_candidates) - set(old_candidates))
    ec_candidates_removed = sorted(set(old_candidates) - set(new_candidates))
    ec_candidate_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_candidates) & set(new_candidates)):
        old_entry = old_candidates[name]
        new_entry = new_candidates[name]
        if old_entry.get("status") != "available" or new_entry.get("status") != "available":
            if old_entry.get("status") != new_entry.get("status"):
                ec_candidate_drift.append(
                    {
                        "candidate": name,
                        "metric": "status",
                        "old": old_entry.get("status"),
                        "new": new_entry.get("status"),
                    }
                )
            continue
        ec_metrics: list[tuple[str, Any, Any]] = []
        for arm in _ENUMERATIONCONVENTION_ARM_KEYS:
            ec_metrics.append(
                (f"{arm}_rate", old_entry["arms"][arm]["rate"], new_entry["arms"][arm]["rate"])
            )
        for arm in _ENUMERATIONCONVENTION_TREATED_ARMS:
            old_head = old_entry["headline"][arm]
            new_head = new_entry["headline"][arm]
            old_ids = old_entry["ids"][arm]
            new_ids = new_entry["ids"][arm]
            ec_metrics.extend(
                [
                    (f"{arm}_label", old_head["label"], new_head["label"]),
                    (
                        f"{arm}_control_passes",
                        old_head["control_passes"],
                        new_head["control_passes"],
                    ),
                    (f"{arm}_wrong_values", old_head["wrong_values"], new_head["wrong_values"]),
                    (f"{arm}_n_reverse", old_head["n_reverse"], new_head["n_reverse"]),
                    (f"{arm}_n_population", old_head["n_population"], new_head["n_population"]),
                    (f"{arm}_n_incomplete", old_head["n_incomplete"], new_head["n_incomplete"]),
                    (
                        f"{arm}_n_ids_divergent",
                        old_head["n_ids_divergent"],
                        new_head["n_ids_divergent"],
                    ),
                    (f"{arm}_n_non_answer", old_head["n_non_answer"], new_head["n_non_answer"]),
                    (f"{arm}_control_rate", old_head["control_rate"], new_head["control_rate"]),
                    (f"{arm}_arm_rate", old_head["arm_rate"], new_head["arm_rate"]),
                    (
                        f"{arm}_pair_exact_only",
                        old_head["pair_decomposition"]["exact_only"],
                        new_head["pair_decomposition"]["exact_only"],
                    ),
                    (
                        f"{arm}_pair_boundary_only",
                        old_head["pair_decomposition"]["boundary_only"],
                        new_head["pair_decomposition"]["boundary_only"],
                    ),
                    (
                        f"{arm}_pair_mixed",
                        old_head["pair_decomposition"]["mixed"],
                        new_head["pair_decomposition"]["mixed"],
                    ),
                    (
                        f"{arm}_n_ids_identical",
                        old_ids["n_ids_identical"],
                        new_ids["n_ids_identical"],
                    ),
                    (
                        f"{arm}_max_abs_id_delta",
                        old_ids["max_abs_id_delta"],
                        new_ids["max_abs_id_delta"],
                    ),
                    (
                        f"{arm}_n_same_ids_verdict_flips",
                        old_ids["n_same_ids_verdict_flips"],
                        new_ids["n_same_ids_verdict_flips"],
                    ),
                ]
            )
            old_positions = {row["position"]: row for row in old_entry["positions"][arm]}
            new_positions = {row["position"]: row for row in new_entry["positions"][arm]}
            for t in _ENUMERATIONCONVENTION_INDICES:
                ec_metrics.extend(
                    [
                        (
                            f"{arm}_pos{t}_control_passes",
                            old_positions[t]["control_passes"],
                            new_positions[t]["control_passes"],
                        ),
                        (
                            f"{arm}_pos{t}_wrong_values",
                            old_positions[t]["wrong_values"],
                            new_positions[t]["wrong_values"],
                        ),
                        (
                            f"{arm}_pos{t}_mean_delta",
                            old_positions[t]["mean_id_length_delta"],
                            new_positions[t]["mean_id_length_delta"],
                        ),
                    ]
                )
        for metric, old_value, new_value in ec_metrics:
            if old_value != new_value:
                ec_candidate_drift.append(
                    {"candidate": name, "metric": metric, "old": old_value, "new": new_value}
                )
    result["candidate_drift"] = ec_candidate_drift
    result["candidates_added"] = ec_candidates_added
    result["candidates_removed"] = ec_candidates_removed
    result["protocol_drift"] = ec_protocol_drift
    result["generator_drift"] = ec_generator_drift
    result["backend_drift"] = ec_backend_drift
    result["comparable"] = ec_comparable
    result["changed"] = bool(
        ec_candidate_drift
        or ec_candidates_added
        or ec_candidates_removed
        or ec_protocol_drift
        or ec_generator_drift
        or ec_backend_drift
    )
    return result


_TYPOGRAPHICEMPHASIS_TOPICS = (
    "harbor",
    "summit",
    "quarry",
    "tundra",
    "granite",
    "willow",
    "copper",
    "saddle",
    "walnut",
    "velvet",
    "cobalt",
    "marble",
    "cinder",
    "falcon",
    "garnet",
    "saffron",
    "indigo",
    "juniper",
    "lagoon",
    "magnet",
    "nickel",
    "pepper",
    "quartz",
    "birch",
)


_TYPOGRAPHICEMPHASIS_WORDS = (
    "alder",
    "basalt",
    "cedar",
    "drift",
    "elm",
    "fern",
    "glade",
    "heath",
    "iris",
    "jade",
    "kelp",
    "loom",
    "moss",
    "nova",
    "ochre",
    "pine",
    "quill",
    "reed",
    "sage",
    "tide",
    "urn",
    "vale",
    "wren",
    "yew",
)


_TYPOGRAPHICEMPHASIS_NEUTRAL_STATEMENTS = (
    "The weather changed slowly during the afternoon hours.",
    "A wooden fence runs along the edge of the property.",
    "She placed the documents inside a plain folder yesterday.",
    "The train arrived later than the schedule suggested.",
    "Fresh bread was cooling on the counter near the sink.",
    "He repaired the bicycle with tools from the garage.",
    "The bookshop extends its opening hours during summer.",
    "Several chairs were arranged around the long table.",
    "The recipe calls for a pinch of salt and herbs.",
    "They walked along the path beside the lake at dawn.",
    "The committee postponed its decision until next week.",
    "A thick layer of dust covered the old furniture.",
    "The children played quietly in the courtyard after lunch.",
    "His handwriting was difficult to read in places.",
    "The farmer harvested the fields before the rain came.",
    "An old map hung framed on the wall of the hallway.",
    "The accountant reviewed the figures a second time.",
    "Birds gathered on the rooftops as the sun went down.",
    "The mechanic inspected the engine and changed the oil.",
    "A narrow staircase led up to the attic.",
    "The orchestra rehearsed the piece several times over.",
    "Someone left a bicycle leaning against the railing.",
    "The soup simmered gently on the stove all morning.",
    "Visitors signed the guest book near the entrance.",
)


_TYPOGRAPHICEMPHASIS_ARM_KEYS = ("control", "bold", "italic", "monospace", "caps")


_TYPOGRAPHICEMPHASIS_TREATED_ARMS = _TYPOGRAPHICEMPHASIS_ARM_KEYS[1:]


_TYPOGRAPHICEMPHASIS_PC_ARM_KEYS = ("pc_plain", "pc_info")


_TYPOGRAPHICEMPHASIS_PCB_ARM = "pc_b"


_TYPOGRAPHICEMPHASIS_RECORD_ARMS = (
    _TYPOGRAPHICEMPHASIS_ARM_KEYS
    + _TYPOGRAPHICEMPHASIS_PC_ARM_KEYS
    + (_TYPOGRAPHICEMPHASIS_PCB_ARM,)
)


_TYPOGRAPHICEMPHASIS_PC_QUESTION = "What is the inspection code of the topic written in bold?"


_TYPOGRAPHICEMPHASIS_CASE_COUNT = 24


_TYPOGRAPHICEMPHASIS_MIN_D = 10


_TYPOGRAPHICEMPHASIS_MIN_J = 6


_TYPOGRAPHICEMPHASIS_MAX_INCOMPLETE = 6


_TYPOGRAPHICEMPHASIS_REDRAW_LIMIT = 1000


_TYPOGRAPHICEMPHASIS_DELTA = 0.34


_TYPOGRAPHICEMPHASIS_ALPHA = 0.02


_TYPOGRAPHICEMPHASIS_PC_MIN_INFO_PASSES = 10


_TYPOGRAPHICEMPHASIS_PC_MIN_MARGIN = 8


_TYPOGRAPHICEMPHASIS_GENERATOR_SPEC = (
    "per case index i in 0..23 with run seed s: topic = TOPICS[i]; "
    "value drawn at slots 0 (word) and 1 (number) of the frozen draw "
    "int.from_bytes(sha256(f'{s}|typographicemphasis|{i}|{slot}|{counter}')"
    ".digest()[:8], 'big'), word = WORDS[draw % 24], number = 1000 + "
    "draw % 9000, redrawn at counter+1 while the value collides with an "
    "earlier case's value (give up past 1000: the stage fails closed); "
    "neutrals at slots 2 and 3 (second index +1 mod 24 on collision); "
    "positive-control distractor topic at slot 4 (draw % 24, +1 mod 24 "
    "when equal to i), distractor value at slots 5 and 6 (redrawn at "
    "counter+1 while equal to the case's own value, give up past 1000), "
    "position bit at slot 7 (0 = the target fact is printed first)"
)


_TYPOGRAPHICEMPHASIS_MAX_CANDIDATES = 24


_TYPOGRAPHICEMPHASIS_MAX_GROUND_RECORDS = (
    _TYPOGRAPHICEMPHASIS_MAX_CANDIDATES * _TYPOGRAPHICEMPHASIS_CASE_COUNT
)


_TYPOGRAPHICEMPHASIS_GENERATION_BUDGET = (
    _TYPOGRAPHICEMPHASIS_MAX_CANDIDATES
    * _TYPOGRAPHICEMPHASIS_CASE_COUNT
    * len(_TYPOGRAPHICEMPHASIS_RECORD_ARMS)
)


_TYPOGRAPHICEMPHASIS_MAX_RECORDS_BYTES = 12 * 1024 * 1024


_TYPOGRAPHICEMPHASIS_CELL_FIELDS = {
    "verdict",
    "boundary_found",
    "boundary_text_match",
    "input_tokens",
    "answer_digest",
    "answer_text",
}


_TYPOGRAPHICEMPHASIS_DETAILS = ("arm_context_exceeded", "arm_evaluation_failed")


_TYPOGRAPHICEMPHASIS_CELL_CLASSES = (
    "pass",
    "boundary_only",
    "wrong_value",
    "echo_bold",
    "echo_italic",
    "echo_monospace",
    "echo_caps",
    "form_only",
    "non_answer",
)


_TYPOGRAPHICEMPHASIS_PAIR_CLASSES = ("boundary_only", "exact_only", "mixed")


_TYPOGRAPHICEMPHASIS_OPERATING_CHARACTERISTICS: dict[str, Any] = {
    "delta": 0.34,
    "alpha": 0.02,
    "certification_bound_completed_per_comparison": 0.013826,
    "certification_bound_restricted_vertex_delta_034": 0.012253,
    "certification_bound_completed_m23": 0.017934,
    "certification_bound_completed_m22": 0.023163,
    "familywise_certification_expected_96": 1.3273,
    "familywise_certification_any_96": 0.7373,
    "certify_anchored_delta_034_pc_090": 0.0034,
    "certification_bound_uniform_homogeneous_delta_034": 0.005414,
    "rtb_counterexample_certification_probability": 0.022662,
    "fire_competent_independent": 0.017274,
    "fire_competent_paired": 0.004442,
    "fire_weak_independent": 0.1743,
    "fire_random_independent": 0.550306,
    "power_drop10_base90": 0.2542,
    "power_drop25_base90": 0.8159,
    "power_drop25_base85": 0.8761,
    "power_drop25_paired": 0.698078,
    "power_drop50_base90": 0.9990,
    "certify_paired_rho_05_delta_000": 0.6139,
    "certify_paired_rho_05_delta_028": 0.0044,
    "familywise_fire_expected_96": 1.6583,
    "familywise_fire_any_96": 0.8123,
    "certify_anchored_delta_000_pc_090": 0.5974,
    "pc_gate_deaf_x_025_independence": 0.0147,
    "pc_gate_deaf_x_030": 0.0235,
    "pc_gate_deaf_x_035": 0.0332,
    "pc_gate_deaf_x_050": 0.0758,
    "world_a_guarded_certification": 0.0016,
    "world_a_guarded_fire": 0.0047,
    "world_b_delta_unguarded_certification": 0.2709,
    "world_b_delta_guarded_certification": 0.0127,
    "world_b_pasta_guarded_certification": 0.0006,
    "world_c_guarded_fire": 0.6318,
}


_TYPOGRAPHICEMPHASIS_AGGREGATE_KEYS = {
    "arms",
    "dose",
    "ids",
    "positive_control",
    "headline",
    "operating_characteristics",
    "n_cases",
    "n_incomplete",
}


def _typographicemphasis_digest(text: str) -> str:
    """SHA-256 hex digest (local copy; no imports from
    typographicemphasis.py anywhere in this mirror).
    """

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _typographicemphasis_text_digest(text: str) -> str | None:
    """SHA-256 of the UTF-8 encoding, or None when the
    text cannot be encoded (a lone-surrogate answer_text
    is a validation failure, never an exception -- the
    cycle-190 A3 pattern, applied per audit F2).
    """
    try:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    except UnicodeEncodeError:
        return None


def _typographicemphasis_digest_ok(value: Any) -> bool:
    """Whether a stored digest is a well-formed SHA-256 hex
    digest.
    """
    if type(value) is not str or len(value) != 64:
        return False
    return all(ch in "0123456789abcdef" for ch in value)


def _typographicemphasis_derive(seed: int, index: int, slot: int, counter: int) -> int:
    """Mirror of typographicemphasis._derive (the draw
    conversion pinned by the design's generator spec).
    """

    material = f"{seed}|typographicemphasis|{index}|{slot}|{counter}"
    return int.from_bytes(hashlib.sha256(material.encode("ascii")).digest()[:8], "big")


class _TypographicEmphasisMirrorExhausted(Exception):
    """Mirror-internal: the redraw give-up bound was reached
    during regeneration, so the payload cannot be recomputed
    and validates false.
    """


def _typographicemphasis_generate(seed: int) -> list[dict[str, Any]]:
    """Mirror of typographicemphasis.generate_cases: the
    frozen SHA-256 derivation with panel-wide value
    distinctness by redraw, the neutral pair, and the
    positive-control draws (distractor topic, distractor
    value, position bit).
    """
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index in range(_TYPOGRAPHICEMPHASIS_CASE_COUNT):
        counter = 0
        while True:
            word = _TYPOGRAPHICEMPHASIS_WORDS[
                _typographicemphasis_derive(seed, index, 0, counter)
                % len(_TYPOGRAPHICEMPHASIS_WORDS)
            ]
            number = 1000 + _typographicemphasis_derive(seed, index, 1, counter) % 9000
            value = f"{word}-{number}"
            if value not in seen:
                break
            counter += 1
            if counter > _TYPOGRAPHICEMPHASIS_REDRAW_LIMIT:
                raise _TypographicEmphasisMirrorExhausted
        seen.add(value)
        first = _typographicemphasis_derive(seed, index, 2, 0) % len(
            _TYPOGRAPHICEMPHASIS_NEUTRAL_STATEMENTS
        )
        second = _typographicemphasis_derive(seed, index, 3, 0) % len(
            _TYPOGRAPHICEMPHASIS_NEUTRAL_STATEMENTS
        )
        if second == first:
            second = (first + 1) % len(_TYPOGRAPHICEMPHASIS_NEUTRAL_STATEMENTS)
        distractor_index = _typographicemphasis_derive(seed, index, 4, 0) % len(
            _TYPOGRAPHICEMPHASIS_TOPICS
        )
        if distractor_index == index:
            distractor_index = (index + 1) % len(_TYPOGRAPHICEMPHASIS_TOPICS)
        counter = 0
        while True:
            d_word = _TYPOGRAPHICEMPHASIS_WORDS[
                _typographicemphasis_derive(seed, index, 5, counter)
                % len(_TYPOGRAPHICEMPHASIS_WORDS)
            ]
            d_number = 1000 + _typographicemphasis_derive(seed, index, 6, counter) % 9000
            distractor_value = f"{d_word}-{d_number}"
            if distractor_value != value:
                break
            counter += 1
            if counter > _TYPOGRAPHICEMPHASIS_REDRAW_LIMIT:
                raise _TypographicEmphasisMirrorExhausted
        cases.append(
            {
                "fixture_id": f"typographicemphasis-{index:02d}",
                "index": index,
                "topic": _TYPOGRAPHICEMPHASIS_TOPICS[index],
                "expected": value,
                "neutral_indices": [first, second],
                "distractor_topic": _TYPOGRAPHICEMPHASIS_TOPICS[distractor_index],
                "distractor_value": distractor_value,
                "target_first": _typographicemphasis_derive(seed, index, 7, 0) % 2 == 0,
            }
        )
    return cases


def _typographicemphasis_wrap(span: str, arm: str) -> str:
    """Mirror of typographicemphasis.wrap_span."""
    if arm == "bold":
        return f"**{span}**"
    if arm == "italic":
        return f"*{span}*"
    if arm == "monospace":
        return f"`{span}`"
    if arm == "caps":
        return span.upper()
    return span


def _typographicemphasis_fact(case: dict[str, Any], arm: str) -> str:
    """Mirror of typographicemphasis.fact_sentence."""
    return (
        f"The {_typographicemphasis_wrap(str(case['topic']), arm)} "
        f"inspection code is {case['expected']}."
    )


def _typographicemphasis_question(case: dict[str, Any]) -> str:
    """Mirror of typographicemphasis.question_text."""
    return f"What is the {case['topic']} inspection code?"


def _typographicemphasis_arm_content(case: dict[str, Any], arm: str) -> str:
    """Mirror of typographicemphasis.arm_content."""
    first, second = (int(i) for i in case["neutral_indices"])
    return (
        f"{_TYPOGRAPHICEMPHASIS_NEUTRAL_STATEMENTS[first]} "
        f"{_typographicemphasis_fact(case, arm)} "
        f"{_TYPOGRAPHICEMPHASIS_NEUTRAL_STATEMENTS[second]} "
        f"{_typographicemphasis_question(case)}"
    )


def _typographicemphasis_pc_content(case: dict[str, Any], arm: str) -> str:
    """Mirror of typographicemphasis.pc_content."""
    topic = str(case["topic"])
    wrapped = f"**{topic}**" if arm == "pc_info" else topic
    target = f"The {wrapped} inspection code is {case['expected']}."
    distractor = f"The {case['distractor_topic']} inspection code is {case['distractor_value']}."
    if case["target_first"]:
        fact_a, fact_b = target, distractor
    else:
        fact_a, fact_b = distractor, target
    first, second = (int(i) for i in case["neutral_indices"])
    return (
        f"{_TYPOGRAPHICEMPHASIS_NEUTRAL_STATEMENTS[first]} {fact_a} {fact_b} "
        f"{_TYPOGRAPHICEMPHASIS_NEUTRAL_STATEMENTS[second]} "
        f"{_TYPOGRAPHICEMPHASIS_PC_QUESTION}"
    )


def _typographicemphasis_pcb_content(case: dict[str, Any]) -> str:
    """Mirror of typographicemphasis.pcb_content."""
    first, second = (int(i) for i in case["neutral_indices"])
    topic = str(case["topic"])
    fact = f"The ***{topic}*** inspection code is {case['expected']}."
    question = f"What is the ***{topic}*** inspection code?"
    return (
        f"{_TYPOGRAPHICEMPHASIS_NEUTRAL_STATEMENTS[first]} {fact} "
        f"{_TYPOGRAPHICEMPHASIS_NEUTRAL_STATEMENTS[second]} {question}"
    )


def _typographicemphasis_content_for(case: dict[str, Any], arm: str) -> str:
    """Mirror of typographicemphasis.content_for."""
    if arm in _TYPOGRAPHICEMPHASIS_PC_ARM_KEYS:
        return _typographicemphasis_pc_content(case, arm)
    if arm == _TYPOGRAPHICEMPHASIS_PCB_ARM:
        return _typographicemphasis_pcb_content(case)
    return _typographicemphasis_arm_content(case, arm)


def _typographicemphasis_pool_digests() -> dict[str, str]:
    """Mirror of typographicemphasis.POOL_DIGESTS."""

    def _pool_digest(pool: tuple[str, ...]) -> str:
        """Pool digest."""
        canonical = json.dumps(list(pool), separators=(",", ":"), ensure_ascii=True)
        return _typographicemphasis_digest(canonical)

    return {
        "topics": _pool_digest(_TYPOGRAPHICEMPHASIS_TOPICS),
        "words": _pool_digest(_TYPOGRAPHICEMPHASIS_WORDS),
        "neutral_statements": _pool_digest(_TYPOGRAPHICEMPHASIS_NEUTRAL_STATEMENTS),
    }


def _typographicemphasis_case_material_sha256() -> str:
    """Mirror of typographicemphasis.case_material_sha256."""

    material = {
        "topics": list(_TYPOGRAPHICEMPHASIS_TOPICS),
        "words": list(_TYPOGRAPHICEMPHASIS_WORDS),
        "neutral_statements": list(_TYPOGRAPHICEMPHASIS_NEUTRAL_STATEMENTS),
        "pool_digests": _typographicemphasis_pool_digests(),
        "generator": _TYPOGRAPHICEMPHASIS_GENERATOR_SPEC,
    }
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return _typographicemphasis_digest(canonical)


def _typographicemphasis_ground_sha256(seed: int) -> str:
    """Mirror of typographicemphasis.ground_sha256."""

    canonical = json.dumps(
        _typographicemphasis_generate(seed),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return _typographicemphasis_digest(canonical)


def _typographicemphasis_protocol(seed: int, backend_fingerprint: str) -> dict[str, Any]:
    """The typographic-emphasis protocol rebuilt from this
    module's own literals, fingerprint included (compared
    whole).
    """

    protocol: dict[str, Any] = {
        "version": 1,
        "arm_keys": list(_TYPOGRAPHICEMPHASIS_ARM_KEYS),
        "treated_arms": list(_TYPOGRAPHICEMPHASIS_TREATED_ARMS),
        "pc_arm_keys": list(_TYPOGRAPHICEMPHASIS_PC_ARM_KEYS),
        "pcb_arm": _TYPOGRAPHICEMPHASIS_PCB_ARM,
        "record_arms": list(_TYPOGRAPHICEMPHASIS_RECORD_ARMS),
        "case_count": _TYPOGRAPHICEMPHASIS_CASE_COUNT,
        "min_d": _TYPOGRAPHICEMPHASIS_MIN_D,
        "min_j": _TYPOGRAPHICEMPHASIS_MIN_J,
        "max_incomplete": _TYPOGRAPHICEMPHASIS_MAX_INCOMPLETE,
        "delta": _TYPOGRAPHICEMPHASIS_DELTA,
        "alpha": _TYPOGRAPHICEMPHASIS_ALPHA,
        "pc_min_info_passes": _TYPOGRAPHICEMPHASIS_PC_MIN_INFO_PASSES,
        "pc_min_margin": _TYPOGRAPHICEMPHASIS_PC_MIN_MARGIN,
        "generator_spec": _TYPOGRAPHICEMPHASIS_GENERATOR_SPEC,
        "pool_digests": _typographicemphasis_pool_digests(),
        "case_material_sha256": _typographicemphasis_case_material_sha256(),
        "ground_sha256": _typographicemphasis_ground_sha256(seed),
        "generation_budget": _TYPOGRAPHICEMPHASIS_GENERATION_BUDGET,
        "max_ground_records": _TYPOGRAPHICEMPHASIS_MAX_GROUND_RECORDS,
        "max_records_bytes": _TYPOGRAPHICEMPHASIS_MAX_RECORDS_BYTES,
        "max_candidates": _TYPOGRAPHICEMPHASIS_MAX_CANDIDATES,
        "seed": seed,
        "backend_fingerprint": backend_fingerprint,
    }
    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    protocol["fingerprint"] = _typographicemphasis_digest(canonical)
    return protocol


def _typographicemphasis_cell_get(record: dict[str, Any], key: str) -> dict[str, Any] | None:
    """Return the typographicemphasis cell get."""
    cells = record.get("cells")
    if not isinstance(cells, dict):
        return None
    cell = cells.get(key)
    return cell if isinstance(cell, dict) else None


def _typographicemphasis_detail_get(record: dict[str, Any], key: str) -> str:
    """Return the typographicemphasis detail get."""
    details = record.get("cell_details")
    if not isinstance(details, dict):
        return ""
    detail = details.get(key, "")
    return detail if type(detail) is str else ""


def _typographicemphasis_verdict(record: dict[str, Any], key: str) -> bool:
    """Return the typographicemphasis verdict."""
    cell = _typographicemphasis_cell_get(record, key)
    return bool(cell is not None and cell.get("verdict") is True)


def _typographicemphasis_ids_identical(record: dict[str, Any], arm: str) -> bool:
    """Mirror of typographicemphasis._ids_divergent_records'
    complement: derived from the recorded id digests alone.
    """
    digests = record.get("id_digests")
    if not isinstance(digests, dict):
        return False
    control = digests.get("control")
    return type(control) is str and control != "" and control == digests.get(arm)


def _typographicemphasis_grammar_ok(text: Any) -> bool:
    """Mirror of typographicemphasis._value_grammar_ok
    (fullmatch semantics only).
    """
    if type(text) is not str or re.fullmatch(r"[a-z]+-[0-9]{4}", text) is None:
        return False
    word, _, number = text.partition("-")
    return word in _TYPOGRAPHICEMPHASIS_WORDS and 1000 <= int(number) <= 9999


def _typographicemphasis_echo_set(expected: str, arm: str) -> set[str]:
    """Mirror of typographicemphasis.echo_set."""
    word, _, number = expected.partition("-")
    if arm == "bold":
        return {f"**{expected}**", f"**{word}**-{number}"}
    if arm == "italic":
        return {f"*{expected}*", f"*{word}*-{number}"}
    if arm == "monospace":
        return {f"`{expected}`", f"`{word}`-{number}"}
    if arm == "caps":
        return {expected.upper(), f"{word.upper()}-{number}"}
    return set()


def _typographicemphasis_strip(text: str, arm: str) -> str:
    """Mirror of typographicemphasis.strip_marks."""
    if arm in ("bold", "italic"):
        return text.replace("*", "")
    if arm == "monospace":
        return text.replace("`", "")
    if arm == "caps":
        return text.lower()
    return text


def _typographicemphasis_decompose_cell(
    cell: dict[str, Any], expected: str, expected_digest: str, arm: str
) -> str:
    """Mirror of typographicemphasis.decompose_cell (the
    ordered decision list; boundary-only requires the
    boundary to have failed, exactly as in the producer).
    """
    if cell.get("verdict") is True:
        return "pass"
    digest = cell.get("answer_digest")
    if digest is not None and digest == expected_digest and cell.get("boundary_found") is not True:
        return "boundary_only"
    text = cell.get("answer_text")
    if type(text) is not str:
        return "non_answer"
    if text != expected and _typographicemphasis_grammar_ok(text):
        return "wrong_value"
    if text in _typographicemphasis_echo_set(expected, arm):
        return f"echo_{arm}"
    if text != expected and _typographicemphasis_strip(text, arm) == expected:
        return "form_only"
    return "non_answer"


def _typographicemphasis_decompose_pair(
    control_cell: dict[str, Any], arm_cell: dict[str, Any]
) -> str:
    """Mirror of typographicemphasis.decompose_pair."""
    if control_cell.get("answer_digest") == arm_cell.get("answer_digest"):
        return "boundary_only"
    if control_cell.get("boundary_found") and arm_cell.get("boundary_found"):
        return "exact_only"
    return "mixed"


def _typographicemphasis_pc_gate(
    info_passes: int, margin_base: int, margin_harm: int, n_incomplete: int
) -> bool:
    """Mirror of typographicemphasis.pc_gate_passed (the
    A5 completion guard included).
    """
    if info_passes < _TYPOGRAPHICEMPHASIS_PC_MIN_INFO_PASSES:
        return False
    if margin_base - margin_harm < _TYPOGRAPHICEMPHASIS_PC_MIN_MARGIN:
        return False
    if n_incomplete > _TYPOGRAPHICEMPHASIS_MAX_INCOMPLETE:
        return False
    return margin_base - (margin_harm + n_incomplete) >= _TYPOGRAPHICEMPHASIS_PC_MIN_MARGIN


def _typographicemphasis_blockers(
    control_passes: int,
    harm: int,
    reverse: int,
    n_incomplete: int,
    domain_size: int,
    pc_passed: bool,
) -> list[str]:
    """Mirror of typographicemphasis.certification_blockers."""
    blockers: list[str] = []
    if not (control_passes >= _TYPOGRAPHICEMPHASIS_MIN_D and harm <= 5 and harm - reverse <= 0):
        blockers.append("statistic_event")
    if domain_size != _TYPOGRAPHICEMPHASIS_CASE_COUNT:
        blockers.append("domain_size")
    if not (harm + n_incomplete <= 5 and harm - reverse + n_incomplete <= 0):
        blockers.append("completion_guard")
    if not pc_passed:
        blockers.append("positive_control")
    return blockers


def _typographicemphasis_classify(
    control_passes: int,
    wrong_values: int,
    harm: int,
    reverse: int,
    n_incomplete: int,
    domain_size: int,
    pc_passed: bool,
) -> str:
    """Mirror of typographicemphasis.classify_headline
    (the combined guarded label).
    """
    if n_incomplete > _TYPOGRAPHICEMPHASIS_MAX_INCOMPLETE:
        return "insufficient"
    if control_passes < _TYPOGRAPHICEMPHASIS_MIN_D:
        return "insufficient"
    if wrong_values >= _TYPOGRAPHICEMPHASIS_MIN_J:
        return "discordant_asymmetric"
    if not _typographicemphasis_blockers(
        control_passes, harm, reverse, n_incomplete, domain_size, pc_passed
    ):
        return "equivalence_certified"
    if wrong_values == 0:
        return "no_discordance_observed" if n_incomplete == 0 else "insufficient"
    return (
        "below_threshold"
        if wrong_values + n_incomplete < _TYPOGRAPHICEMPHASIS_MIN_J
        else "insufficient"
    )


def _typographicemphasis_rate(numerator: int, denominator: int) -> float | None:
    """Return the typographicemphasis rate."""
    return (numerator / denominator) if denominator else None


def _typographicemphasis_aggregate(
    records: list[dict[str, Any]], cases: list[dict[str, Any]]
) -> dict[str, Any]:
    """Mirror of typographicemphasis.aggregate_candidate."""
    expected_by_fixture = {str(case["fixture_id"]): str(case["expected"]) for case in cases}
    arms: dict[str, Any] = {}
    for key in _TYPOGRAPHICEMPHASIS_RECORD_ARMS:
        n_completed = 0
        n_pass = 0
        n_context = 0
        n_failed = 0
        decomposition = dict.fromkeys(_TYPOGRAPHICEMPHASIS_CELL_CLASSES, 0)
        for record in records:
            detail = _typographicemphasis_detail_get(record, key)
            if detail == "arm_context_exceeded":
                n_context += 1
            elif detail == "arm_evaluation_failed":
                n_failed += 1
            cell = _typographicemphasis_cell_get(record, key)
            if cell is None:
                continue
            n_completed += 1
            if cell.get("verdict") is True:
                n_pass += 1
            if key in _TYPOGRAPHICEMPHASIS_ARM_KEYS:
                decomposition[
                    _typographicemphasis_decompose_cell(
                        cell,
                        expected_by_fixture.get(str(record.get("fixture", "")), ""),
                        str(record.get("expected_digest", "")),
                        key,
                    )
                ] += 1
        entry: dict[str, Any] = {
            "rate": _typographicemphasis_rate(n_pass, n_completed),
            "n_pass": n_pass,
            "n_completed": n_completed,
            "n_context_exceeded": n_context,
            "n_evaluation_failed": n_failed,
        }
        if key in _TYPOGRAPHICEMPHASIS_ARM_KEYS:
            entry["decomposition"] = decomposition
        arms[key] = entry
    dose: dict[str, Any] = {}
    for arm in _TYPOGRAPHICEMPHASIS_TREATED_ARMS:
        codepoint_deltas: list[int] = []
        byte_deltas: list[int] = []
        for case in cases:
            treated = _typographicemphasis_arm_content(case, arm)
            control = _typographicemphasis_arm_content(case, "control")
            codepoint_deltas.append(len(treated) - len(control))
            byte_deltas.append(len(treated.encode("utf-8")) - len(control.encode("utf-8")))
        dose[arm] = {
            "codepoint_delta_min": min(codepoint_deltas, default=0),
            "codepoint_delta_max": max(codepoint_deltas, default=0),
            "byte_delta_min": min(byte_deltas, default=0),
            "byte_delta_max": max(byte_deltas, default=0),
        }
    ids: dict[str, Any] = {}
    for arm in _TYPOGRAPHICEMPHASIS_TREATED_ARMS:
        deltas: list[int] = []
        n_ids_identical = 0
        n_same_ids_flips = 0
        for record in records:
            tokens = record.get("arm_tokens")
            tokens = tokens if isinstance(tokens, dict) else {}
            treated_count = tokens.get(arm)
            baseline_count = tokens.get("control")
            if type(treated_count) is int and type(baseline_count) is int:
                deltas.append(treated_count - baseline_count)
            else:
                deltas.append(0)
            if _typographicemphasis_ids_identical(record, arm):
                n_ids_identical += 1
                control_cell = _typographicemphasis_cell_get(record, "control")
                arm_cell = _typographicemphasis_cell_get(record, arm)
                if (
                    control_cell is not None
                    and arm_cell is not None
                    and (control_cell.get("verdict") is True) != (arm_cell.get("verdict") is True)
                ):
                    n_same_ids_flips += 1
        ids[arm] = {
            "n_ids_identical": n_ids_identical,
            "n_ids_divergent": len(records) - n_ids_identical,
            "id_length_delta": deltas,
            "max_abs_id_delta": max((abs(delta) for delta in deltas), default=0),
            "n_same_ids_verdict_flips": n_same_ids_flips,
        }
    n_pc_pairs = 0
    pc_info_passes = 0
    pc_b = 0
    pc_h = 0
    for record in records:
        plain = _typographicemphasis_cell_get(record, "pc_plain")
        info = _typographicemphasis_cell_get(record, "pc_info")
        if plain is not None and info is not None:
            n_pc_pairs += 1
            plain_verdict = plain.get("verdict") is True
            info_verdict = info.get("verdict") is True
            if info_verdict:
                pc_info_passes += 1
            if not plain_verdict and info_verdict:
                pc_b += 1
            if plain_verdict and not info_verdict:
                pc_h += 1
    pc_incomplete = _TYPOGRAPHICEMPHASIS_CASE_COUNT - n_pc_pairs
    positive_control = {
        "n_pairs": n_pc_pairs,
        "n_incomplete": pc_incomplete,
        "info_passes": pc_info_passes,
        "plain_fail_info_pass": pc_b,
        "plain_pass_info_fail": pc_h,
        "margin": pc_b - pc_h,
        "passed": _typographicemphasis_pc_gate(pc_info_passes, pc_b, pc_h, pc_incomplete),
    }
    headline: dict[str, Any] = {}
    for arm in _TYPOGRAPHICEMPHASIS_TREATED_ARMS:
        domain = [
            record for record in records if not _typographicemphasis_ids_identical(record, arm)
        ]
        paired = [
            record
            for record in domain
            if _typographicemphasis_cell_get(record, "control") is not None
            and _typographicemphasis_cell_get(record, arm) is not None
        ]
        n_incomplete = len(domain) - len(paired)
        control_passes = 0
        wrong_values = 0
        harm = 0
        reverse = 0
        arm_passes = 0
        non_answer = 0
        pair_decomposition = dict.fromkeys(_TYPOGRAPHICEMPHASIS_PAIR_CLASSES, 0)
        failing_classes = dict.fromkeys(_TYPOGRAPHICEMPHASIS_CELL_CLASSES, 0)
        for record in paired:
            control_cell = _typographicemphasis_cell_get(record, "control")
            arm_cell = _typographicemphasis_cell_get(record, arm)
            if control_cell is None or arm_cell is None:
                continue
            expected = expected_by_fixture.get(str(record.get("fixture", "")), "")
            expected_digest = str(record.get("expected_digest", ""))
            arm_class = _typographicemphasis_decompose_cell(
                arm_cell, expected, expected_digest, arm
            )
            if arm_class == "non_answer":
                non_answer += 1
            control_verdict = control_cell.get("verdict") is True
            arm_verdict = arm_cell.get("verdict") is True
            if control_verdict:
                control_passes += 1
                if arm_class == "wrong_value":
                    wrong_values += 1
                if not arm_verdict:
                    harm += 1
            elif arm_verdict:
                reverse += 1
            if arm_verdict:
                arm_passes += 1
            if control_verdict != arm_verdict:
                pair_decomposition[_typographicemphasis_decompose_pair(control_cell, arm_cell)] += 1
                failing = arm_cell if control_verdict else control_cell
                failing_arm = arm if control_verdict else "control"
                failing_classes[
                    _typographicemphasis_decompose_cell(
                        failing, expected, expected_digest, failing_arm
                    )
                ] += 1
        headline[arm] = {
            "label": _typographicemphasis_classify(
                control_passes,
                wrong_values,
                harm,
                reverse,
                n_incomplete,
                len(domain),
                bool(positive_control["passed"]),
            ),
            "n_population": len(paired),
            "n_incomplete": n_incomplete,
            "n_ids_divergent": len(domain),
            "control_passes": control_passes,
            "wrong_values": wrong_values,
            "harm": harm,
            "n_reverse": reverse,
            "net_harm": harm - reverse,
            "control_rate": _typographicemphasis_rate(control_passes, len(paired)),
            "arm_rate": _typographicemphasis_rate(arm_passes, len(paired)),
            "n_non_answer": non_answer,
            "pair_decomposition": pair_decomposition,
            "failing_classes": failing_classes,
            "delta": _TYPOGRAPHICEMPHASIS_DELTA,
            "alpha": _TYPOGRAPHICEMPHASIS_ALPHA,
            "pc_gate_passed": positive_control["passed"],
            "certification_blocked_by": _typographicemphasis_blockers(
                control_passes,
                harm,
                reverse,
                n_incomplete,
                len(domain),
                bool(positive_control["passed"]),
            ),
        }
    return {
        "arms": arms,
        "dose": dose,
        "ids": ids,
        "positive_control": positive_control,
        "headline": headline,
        "operating_characteristics": dict(_TYPOGRAPHICEMPHASIS_OPERATING_CHARACTERISTICS),
        "n_cases": len(records),
        "n_incomplete": sum(
            1
            for record in records
            if any(
                _typographicemphasis_cell_get(record, arm) is None
                for arm in _TYPOGRAPHICEMPHASIS_RECORD_ARMS
            )
        ),
    }


def _typographicemphasis_cell_ok(cell: Any, expected_digest: str) -> bool:
    """One cell's shape and bindings: the exact field set;
    the verdict binding ``verdict == (answer_digest ==
    expected_digest) and boundary_found``; the carried
    ``answer_text`` must hash to its ``answer_digest`` (the
    cycle-190 binding); the input-token binding to the
    record's arm token count is checked by the caller.
    """
    if type(cell) is not dict or set(cell) != _TYPOGRAPHICEMPHASIS_CELL_FIELDS:
        return False
    if type(cell["verdict"]) is not bool or type(cell["boundary_found"]) is not bool:
        return False
    if cell["boundary_text_match"] is not None and type(cell["boundary_text_match"]) is not bool:
        return False
    if type(cell["input_tokens"]) is not int or cell["input_tokens"] < 0:
        return False
    digest = cell["answer_digest"]
    if digest is not None and not _typographicemphasis_digest_ok(digest):
        return False
    text = cell["answer_text"]
    if text is None:
        if digest is not None:
            return False
    elif type(text) is not str or _typographicemphasis_text_digest(text) != digest:
        return False
    verdict = (digest == expected_digest) and cell["boundary_found"]
    return bool(cell["verdict"] == verdict)


def _typographicemphasis_recompute(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Recompute a typographic-emphasis payload from its
    ground records.

    The validator carries its own generator mirror with the
    frozen SHA-256 derivation, the frozen pools, and the
    aggregators (no imports from typographicemphasis.py):
    the protocol is rebuilt from frozen literals plus the
    payload's seed and backend fingerprint and compared
    whole; the case set is regenerated from the seed and
    every record is checked against it in protocol order;
    every record's per-arm content digests are rebound to
    the regenerated arm contents (all eight record arms,
    so the treatment itself -- the main arms' wraps, the
    positive control's bold wrap, the PC-B double wrap --
    is committed); the record's ids_identical flags and
    id_length_delta values must equal the derivations from
    its recorded digests and token counts; every cell's
    answer_text is verified against its answer_digest, its
    verdict is re-derived from its digest identity, and
    its input_tokens are bound to the record's arm token
    count; the decompositions, id report, dose, positive
    control, and combined headlines -- including every
    certification label and its blocking gates -- are
    recomputed from the records and the regenerated
    expected values; the aggregates are recomputed and
    compared. A certificate therefore cannot be asserted
    by a record whose committed counts do not earn it.
    Cells exist only from aggregate-eligible reports, so a
    non-null cell certifies a completed evaluation and the
    verdict's status conjunct is discharged structurally.
    """

    protocol = payload.get("protocol")
    if type(protocol) is not dict:
        return None
    seed = protocol.get("seed")
    backend_fingerprint = protocol.get("backend_fingerprint")
    if type(seed) is not int or isinstance(seed, bool) or type(backend_fingerprint) is not str:
        return None
    try:
        expected_protocol = _typographicemphasis_protocol(seed, backend_fingerprint)
    except _TypographicEmphasisMirrorExhausted:
        return None
    if not _recomputed_equal(protocol, expected_protocol):
        return None
    candidates = payload.get("candidates")
    if type(candidates) is not dict or not candidates:
        return None
    if len(candidates) > _TYPOGRAPHICEMPHASIS_MAX_CANDIDATES:
        return None
    try:
        generated = _typographicemphasis_generate(seed)
    except _TypographicEmphasisMirrorExhausted:
        return None
    total_records = 0
    records_bytes = 0
    aggregates: dict[str, Any] = {}
    for name, entry in candidates.items():
        if type(entry) is not dict:
            return None
        status = entry.get("status")
        if status == "unavailable":
            if set(entry) != {"status", "detail", "records"} or entry.get("records") != []:
                return None
            continue
        if status != "available" or entry.get("detail") != "":
            return None
        if set(entry) != _TYPOGRAPHICEMPHASIS_AGGREGATE_KEYS | {"status", "detail", "records"}:
            return None
        records = entry["records"]
        if type(records) is not list or len(records) != len(generated):
            return None
        total_records += len(records)
        records_bytes += len(
            json.dumps(records, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        )
        if records_bytes > _TYPOGRAPHICEMPHASIS_MAX_RECORDS_BYTES:
            return None
        for record, case in zip(records, generated, strict=True):
            if type(record) is not dict or set(record) != {
                "fixture",
                "topic",
                "expected_digest",
                "content_digests",
                "arm_tokens",
                "id_digests",
                "ids_identical",
                "id_length_delta",
                "cells",
                "cell_details",
            }:
                return None
            if record["fixture"] != case["fixture_id"]:
                return None
            if record["topic"] != case["topic"]:
                return None
            expected_digest = _typographicemphasis_digest(str(case["expected"]))
            if record["expected_digest"] != expected_digest:
                return None
            arm_tokens = record["arm_tokens"]
            id_digests = record["id_digests"]
            content_digests = record["content_digests"]
            ids_identical = record["ids_identical"]
            id_length_delta = record["id_length_delta"]
            cells = record["cells"]
            details = record["cell_details"]
            for mapping in (arm_tokens, id_digests, content_digests, cells, details):
                if type(mapping) is not dict or set(mapping) != set(
                    _TYPOGRAPHICEMPHASIS_RECORD_ARMS
                ):
                    return None
            for mapping in (ids_identical, id_length_delta):
                if type(mapping) is not dict or set(mapping) != set(
                    _TYPOGRAPHICEMPHASIS_TREATED_ARMS
                ):
                    return None
            # Validate every arm_tokens value BEFORE any arithmetic on
            # them, exactly as in _invisiblechars_recompute: a
            # type-confused value escaped as a raw TypeError while the
            # type check sat in the later per-arm loop (bug-hunt
            # wave 2, BH2B-02).
            for arm in _TYPOGRAPHICEMPHASIS_RECORD_ARMS:
                tokens = arm_tokens[arm]
                if type(tokens) is not int or isinstance(tokens, bool) or tokens < 0:
                    return None
            for arm in _TYPOGRAPHICEMPHASIS_TREATED_ARMS:
                if ids_identical[arm] is not (id_digests[arm] == id_digests["control"]):
                    return None
                if id_length_delta[arm] != arm_tokens[arm] - arm_tokens["control"]:
                    return None
            for arm in _TYPOGRAPHICEMPHASIS_RECORD_ARMS:
                tokens = arm_tokens[arm]
                if type(tokens) is not int or isinstance(tokens, bool) or tokens < 0:
                    return None
                if not _typographicemphasis_digest_ok(id_digests[arm]):
                    return None
                if content_digests[arm] != _typographicemphasis_digest(
                    _typographicemphasis_content_for(case, arm)
                ):
                    return None
                cell = cells[arm]
                detail = details[arm]
                if detail not in ("", *_TYPOGRAPHICEMPHASIS_DETAILS):
                    return None
                if cell is None:
                    if detail == "":
                        return None
                    continue
                if detail != "":
                    return None
                if not _typographicemphasis_cell_ok(cell, expected_digest):
                    return None
                if cell["input_tokens"] != tokens:
                    return None
        aggregates[name] = _typographicemphasis_aggregate(records, generated)
    if total_records > _TYPOGRAPHICEMPHASIS_MAX_GROUND_RECORDS:
        return None
    if not aggregates:
        return None
    return {
        "aggregates": aggregates,
        "n_candidates": len(candidates),
        "n_cases": len(generated),
    }


def _valid_typographic_emphasis(payload: Any) -> bool:
    """Structural check for a ``typographic_emphasis`` payload
    (recompute standard). The audit block is shape-checked
    only: its counts are stage observations (special-token
    findings against the run's own formats), not recomputable
    from records.
    """
    if type(payload) is not dict or payload.get("available") is not True:
        return False
    if set(payload) != {
        "available",
        "detail",
        "protocol",
        "audit",
        "candidates",
        "n_candidates",
        "n_cases",
    }:
        return False
    if payload.get("detail") != "":
        return False
    audit = payload.get("audit")
    if type(audit) is not dict or set(audit) != {
        "n_audit_excluded",
        "by_reason",
        "special_token_findings",
    }:
        return False
    if type(audit["n_audit_excluded"]) is not int or audit["n_audit_excluded"] < 0:
        return False
    if type(audit["special_token_findings"]) is not int or audit["special_token_findings"] < 0:
        return False
    by_reason = audit["by_reason"]
    if type(by_reason) is not dict:
        return False
    for key, count in by_reason.items():
        if type(key) is not str or type(count) is not int or count < 0:
            return False
    recomputed = _typographicemphasis_recompute(payload)
    if recomputed is None:
        return False
    candidates = payload["candidates"]
    for name, expected in recomputed["aggregates"].items():
        entry = candidates[name]
        for key, value in expected.items():
            if not _recomputed_equal(entry.get(key), value):
                return False
    if not _recomputed_equal(payload.get("n_candidates"), recomputed["n_candidates"]):
        return False
    return _recomputed_equal(payload.get("n_cases"), recomputed["n_cases"])


def diff_typographic_emphasis(old: Any, new: Any) -> dict[str, Any]:
    """Compare two ``typographic_emphasis`` probe payloads;
    deterministic.

    Answers "did the typographic-emphasis story change
    between runs?": per candidate and treated arm, drift in
    the combined label with the D / J / H / R counts that
    carry it, the divergent-domain size M, the reverse and
    non-answer counts, the paired population and its
    incompletion, the certification blocking gates, the
    positive-control counts and gate verdict, the id
    report, the per-arm dose deltas, and the pair
    decomposition; candidates added/removed; and protocol /
    generator / backend drift (any of which marks the
    comparison not comparable, with the drift still listed
    -- the generated panel is the treatment, so a changed
    generator is protocol drift). A side without a valid
    payload is reported plainly.
    """
    old_ok = _valid_typographic_emphasis(old)
    new_ok = _valid_typographic_emphasis(new)
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
    te_protocol_drift = old["protocol"].get("fingerprint") != new["protocol"].get("fingerprint")
    te_generator_drift = old["protocol"].get("ground_sha256") != new["protocol"].get(
        "ground_sha256"
    ) or old["protocol"].get("case_material_sha256") != new["protocol"].get("case_material_sha256")
    te_backend_drift = old["protocol"].get("backend_fingerprint") != new["protocol"].get(
        "backend_fingerprint"
    )
    te_comparable = not te_protocol_drift and not te_generator_drift and not te_backend_drift
    old_candidates = old["candidates"]
    new_candidates = new["candidates"]
    te_candidates_added = sorted(set(new_candidates) - set(old_candidates))
    te_candidates_removed = sorted(set(old_candidates) - set(new_candidates))
    te_candidate_drift: list[dict[str, Any]] = []
    for name in sorted(set(old_candidates) & set(new_candidates)):
        old_entry = old_candidates[name]
        new_entry = new_candidates[name]
        if old_entry.get("status") != "available" or new_entry.get("status") != "available":
            if old_entry.get("status") != new_entry.get("status"):
                te_candidate_drift.append(
                    {
                        "candidate": name,
                        "metric": "status",
                        "old": old_entry.get("status"),
                        "new": new_entry.get("status"),
                    }
                )
            continue
        te_metrics: list[tuple[str, Any, Any]] = []
        for arm in _TYPOGRAPHICEMPHASIS_RECORD_ARMS:
            te_metrics.append(
                (f"{arm}_rate", old_entry["arms"][arm]["rate"], new_entry["arms"][arm]["rate"])
            )
        old_pc = old_entry["positive_control"]
        new_pc = new_entry["positive_control"]
        te_metrics.extend(
            [
                ("pc_passed", old_pc["passed"], new_pc["passed"]),
                ("pc_info_passes", old_pc["info_passes"], new_pc["info_passes"]),
                (
                    "pc_plain_fail_info_pass",
                    old_pc["plain_fail_info_pass"],
                    new_pc["plain_fail_info_pass"],
                ),
                (
                    "pc_plain_pass_info_fail",
                    old_pc["plain_pass_info_fail"],
                    new_pc["plain_pass_info_fail"],
                ),
                ("pc_margin", old_pc["margin"], new_pc["margin"]),
                ("pc_n_incomplete", old_pc["n_incomplete"], new_pc["n_incomplete"]),
            ]
        )
        for arm in _TYPOGRAPHICEMPHASIS_TREATED_ARMS:
            te_metrics.extend(
                [
                    (
                        f"{arm}_dose_codepoint_min",
                        old_entry["dose"][arm]["codepoint_delta_min"],
                        new_entry["dose"][arm]["codepoint_delta_min"],
                    ),
                    (
                        f"{arm}_dose_codepoint_max",
                        old_entry["dose"][arm]["codepoint_delta_max"],
                        new_entry["dose"][arm]["codepoint_delta_max"],
                    ),
                    (
                        f"{arm}_dose_byte_min",
                        old_entry["dose"][arm]["byte_delta_min"],
                        new_entry["dose"][arm]["byte_delta_min"],
                    ),
                    (
                        f"{arm}_dose_byte_max",
                        old_entry["dose"][arm]["byte_delta_max"],
                        new_entry["dose"][arm]["byte_delta_max"],
                    ),
                ]
            )
        for arm in _TYPOGRAPHICEMPHASIS_TREATED_ARMS:
            old_head = old_entry["headline"][arm]
            new_head = new_entry["headline"][arm]
            old_ids = old_entry["ids"][arm]
            new_ids = new_entry["ids"][arm]
            te_metrics.extend(
                [
                    (f"{arm}_label", old_head["label"], new_head["label"]),
                    (
                        f"{arm}_control_passes",
                        old_head["control_passes"],
                        new_head["control_passes"],
                    ),
                    (f"{arm}_wrong_values", old_head["wrong_values"], new_head["wrong_values"]),
                    (f"{arm}_harm", old_head["harm"], new_head["harm"]),
                    (f"{arm}_n_reverse", old_head["n_reverse"], new_head["n_reverse"]),
                    (f"{arm}_net_harm", old_head["net_harm"], new_head["net_harm"]),
                    (f"{arm}_n_population", old_head["n_population"], new_head["n_population"]),
                    (f"{arm}_n_incomplete", old_head["n_incomplete"], new_head["n_incomplete"]),
                    (
                        f"{arm}_n_ids_divergent",
                        old_head["n_ids_divergent"],
                        new_head["n_ids_divergent"],
                    ),
                    (f"{arm}_n_non_answer", old_head["n_non_answer"], new_head["n_non_answer"]),
                    (f"{arm}_control_rate", old_head["control_rate"], new_head["control_rate"]),
                    (f"{arm}_arm_rate", old_head["arm_rate"], new_head["arm_rate"]),
                    (
                        f"{arm}_pc_gate_passed",
                        old_head["pc_gate_passed"],
                        new_head["pc_gate_passed"],
                    ),
                    (
                        f"{arm}_certification_blocked_by",
                        old_head["certification_blocked_by"],
                        new_head["certification_blocked_by"],
                    ),
                    (
                        f"{arm}_pair_exact_only",
                        old_head["pair_decomposition"]["exact_only"],
                        new_head["pair_decomposition"]["exact_only"],
                    ),
                    (
                        f"{arm}_pair_boundary_only",
                        old_head["pair_decomposition"]["boundary_only"],
                        new_head["pair_decomposition"]["boundary_only"],
                    ),
                    (
                        f"{arm}_pair_mixed",
                        old_head["pair_decomposition"]["mixed"],
                        new_head["pair_decomposition"]["mixed"],
                    ),
                    (
                        f"{arm}_n_ids_identical",
                        old_ids["n_ids_identical"],
                        new_ids["n_ids_identical"],
                    ),
                    (
                        f"{arm}_max_abs_id_delta",
                        old_ids["max_abs_id_delta"],
                        new_ids["max_abs_id_delta"],
                    ),
                    (
                        f"{arm}_n_same_ids_verdict_flips",
                        old_ids["n_same_ids_verdict_flips"],
                        new_ids["n_same_ids_verdict_flips"],
                    ),
                ]
            )
        for metric, old_value, new_value in te_metrics:
            if old_value != new_value:
                te_candidate_drift.append(
                    {"candidate": name, "metric": metric, "old": old_value, "new": new_value}
                )
    result["candidate_drift"] = te_candidate_drift
    result["candidates_added"] = te_candidates_added
    result["candidates_removed"] = te_candidates_removed
    result["protocol_drift"] = te_protocol_drift
    result["generator_drift"] = te_generator_drift
    result["backend_drift"] = te_backend_drift
    result["comparable"] = te_comparable
    result["changed"] = bool(
        te_candidate_drift
        or te_candidates_added
        or te_candidates_removed
        or te_protocol_drift
        or te_generator_drift
        or te_backend_drift
    )
    return result


def _render_ws_canonical_summary(wscan_diff: dict[str, Any] | None) -> list[str]:
    """Whitespace-canonicalization comparison lines for the stderr summary."""
    if type(wscan_diff) is not dict:
        return []
    available = wscan_diff.get("available")
    if available == "neither":
        return []
    if available != "both":
        missing = "new" if available == "old" else "old"
        return [
            "Whitespace-canonicalization comparison unavailable: "
            f"{missing} run did not record the whitespace scan."
        ]
    lines: list[str] = []

    def _fmt_rate(value: Any) -> str:
        """Return the fmt rate."""
        return (
            f"{value:.0%}"
            if type(value) in (int, float) and _float_convertible(value)
            else "(none)"
        )

    drift = wscan_diff.get("rate_drift") or []
    if drift:
        lines.append(f"Whitespace id/flip drift ({len(drift)}):")
        for entry in drift[:5]:
            metric = _display_id(entry.get("metric"))
            old = entry.get("old")
            new = entry.get("new")
            if metric in ("ids_divergence_rate", "verdict_flip_rate"):
                change = f"{_fmt_rate(old)} -> {_fmt_rate(new)}"
            else:
                change = f"{_display_id(old)} -> {_display_id(new)}"
            lines.append(
                f"  {_display_id(entry['candidate'])}/{_display_id(entry.get('mutation'))}"
                f"/{metric}: {change}"
            )
        if len(drift) > 5:
            lines.append(f"  ... and {len(drift) - 5} more")
    robust = wscan_diff.get("robust_flips") or []
    for entry in robust[:5]:
        lines.append(
            f"Whitespace-robust flip {_display_id(entry['candidate'])}: "
            f"{_display_id(entry.get('old_robust'))} -> {_display_id(entry.get('new_robust'))}"
        )
    for key, label in (
        ("fragile_added", "newly fragile"),
        ("fragile_removed", "no longer fragile"),
    ):
        names = wscan_diff.get(key) or []
        if names:
            lines.append(f"Whitespace {label}: {', '.join(_display_id(c) for c in names)}")
    added = wscan_diff.get("candidates_added") or []
    removed = wscan_diff.get("candidates_removed") or []
    if added:
        lines.append(
            f"Whitespace-scan candidates added: {', '.join(_display_id(c) for c in added)}"
        )
    if removed:
        lines.append(
            f"Whitespace-scan candidates removed: {', '.join(_display_id(c) for c in removed)}"
        )
    return lines
