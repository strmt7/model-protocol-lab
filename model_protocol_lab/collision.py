"""Answer-collision audit (research wave-6 idea 4).

The question: when a correct answer legitimately contains marker-like
text, does the lab's own token-ID boundary detector truncate it -- and
manufacture an exact-match failure out of a correct answer?

Architectural grounding (read before extending this module): the lab's
response boundary is a post-hoc token-ID detector. ``core.first_boundary``
finds the FIRST occurrence of the configured boundary id tuple in the
emitted token ids, and ``core.evaluate`` keeps only
``output_ids[:cut]`` as the answer. The boundary configuration is
operator-supplied (``--boundary-ids`` / ``--boundary-map``) and shared
across candidates by default; NOTHING in the tree derives a boundary
from a candidate's template rendering. So this audit's unit is the
boundary CONFIGURATION, not the template: each candidate's cell is
computed under its resolved boundary, and candidates sharing a boundary
produce identical cells by construction.

What is measured (and what is not):
  * Synthetic fixtures whose expected answers embed marker-like text are
    emitted VERBATIM by scripted backends (per-fixture generation cap,
    so the backend contract's ``len(ids) == cap`` rule holds) through
    ``core.evaluate``. What the detector does to a verbatim-correct
    answer is then classified by a total order over emission shape:
    ``collision`` (the cut fires and tokens follow the marker -- real
    content was discarded), ``ends_at_marker`` (the cut fires and
    nothing follows -- the scripted emission simply ended there; its
    own cell, never conflated with a collision), ``intact`` (no cut).
    The phrase "model stopped early" is deliberately absent: with
    scripted emission there is no model, only fixture bookkeeping.
  * Two emission arms per fixture. The RESERVED arm is the backend's
    natural encoding of the expected answer (the measurement: under a
    specials-parsing tokenizer the marker text becomes the boundary's
    reserved ids). The SUBWORD arm rebuilds the same expected text with
    every marker occurrence replaced by a subword re-encoding
    (via :func:`deliforge.find_encoding_pair`; decode-identical,
    reserved id absent) -- a control arm: a collision there means the
    boundary matches non-reserved encodings, a different detector
    property. Where no encoding pair exists the arm records
    ``no_encoding_pair``, never a silent skip.
  * "Answer collision" has no established meaning elsewhere; the
    one-sentence definition above is the whole definition.
  * Informational only: the probe never touches candidate selection.

Scope (honest framing, repeated in CLI help and the report copy): the
configured boundaries only, this tokenizer's encode/decode, the frozen
fixture set. Scripted verbatim emission measures the DETECTOR, not any
model's behaviour; nothing here says how often real answers contain
marker text, or what a real model would do. A fixture-authoring scanner
will flag this module's own fixtures (they contain hostile marker
strings by design); that is a documented exemption, not a finding.

Pre-registered falsifiable claims (see docs/RESEARCH_LOG.md cycle 170):
  F1: for a single-id reserved boundary (e.g. ChatML ``<|im_end|>``
      under a specials-parsing tokenizer), the reserved arm produces at
      least one interior collision. Falsified if every reserved-arm
      fixture is intact or ends_at_marker across configurations -- the
      audit then stands as a passing regression guard.
  F2: the subword arm never collides when the boundary equals the
      marker's reserved encoding. Falsified by any such collision.
  F3: the clean fixture never collides in either arm. Falsified by any
      clean-fixture boundary firing (a detector false positive on
      marker-free text).

Everything here is pure: fixture construction, emission building,
classification, cell aggregation, and profile aggregation need no
model. The stage (experiments.answer_collision_stage) supplies the
scripted backends.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from typing import Any

from .control_tokens import HOSTILE_MARKER_STRINGS

PROTOCOL_VERSION = "answer-collision/1"

#: Frozen answer templates; ``{marker}`` is the config's own marker text,
#: ``{hostile}`` one HOSTILE_MARKER_STRINGS entry. Positions are pinned:
#: start / interior / end / repeat describe where the marker sits in the
#: expected answer, which is what the total-order classifier needs.
_OWN_TEMPLATES: tuple[tuple[str, str], ...] = (
    ("start", "{marker} ends the turn in this format."),
    ("interior", "The turn ends with {marker} mid-sentence."),
    ("end", "The turn ends with {marker}"),
    ("repeat", "{marker} appears, then {marker} appears again."),
)
_HOSTILE_TEMPLATE = "Another format ends turns with {hostile} here."
_CLEAN_TEXT = "The turn ends with no marker at all."

#: Classification vocabulary. "collision": cut fired and tokens follow
#: the marker (content discarded). "ends_at_marker": cut fired, nothing
#: follows (scripted end-of-answer). "intact": no cut. "anomalous": the
#: observation is internally inconsistent (e.g. no cut but the answer
#: mismatches a verbatim emission) -- counted, excluded from rates.
CLASSES: tuple[str, ...] = ("collision", "ends_at_marker", "intact", "anomalous")

ARMS: tuple[str, str] = ("reserved", "subword")


def n_fixtures_per_config() -> int:
    """Fixtures built per boundary configuration (constant by design)."""
    return len(_OWN_TEMPLATES) + len(HOSTILE_MARKER_STRINGS) + 1


def _fixture_specs(marker_text: str) -> list[dict[str, Any]]:
    """Raw fixture specs for one boundary configuration's marker text."""
    specs: list[dict[str, Any]] = []
    for position, template in _OWN_TEMPLATES:
        specs.append(
            {
                "id": f"own/{position}",
                "marker": marker_text,
                "position": position,
                "expected": template.format(marker=marker_text),
            }
        )
    for index, hostile in enumerate(HOSTILE_MARKER_STRINGS):
        specs.append(
            {
                "id": f"hostile/{index:02d}",
                "marker": hostile,
                "position": "hostile",
                "expected": _HOSTILE_TEMPLATE.format(hostile=hostile),
            }
        )
    specs.append(
        {
            "id": "clean",
            "marker": None,
            "position": "clean",
            "expected": _CLEAN_TEXT,
        }
    )
    return specs


def validate_answer_fixtures(
    specs: Sequence[dict[str, Any]], marker_text: str
) -> tuple[dict[str, Any], ...]:
    """Validate fixture specs for one boundary configuration.

    Raises ``ValueError`` on any malformed spec: non-dict, missing/blank
    fields, duplicate ids, an expected answer that does not place its
    marker at the pinned position, or a clean fixture that contains the
    config's marker text or any hostile string. Fail-closed: the stage
    degrades the candidate's cell to None rather than scoring a broken
    fixture set.
    """
    if type(marker_text) is not str or not marker_text:
        raise ValueError("marker_text must be a nonblank string")
    if not isinstance(specs, (list, tuple)) or not specs:
        raise ValueError("specs must be a nonempty sequence")
    seen: set[str] = set()
    for spec in specs:
        if type(spec) is not dict:
            raise ValueError("each spec must be a dict")
        spec_id = spec.get("id")
        position = spec.get("position")
        marker = spec.get("marker")
        expected = spec.get("expected")
        if type(spec_id) is not str or not spec_id.strip():
            raise ValueError("spec id must be a nonblank string")
        if spec_id in seen:
            raise ValueError(f"duplicate spec id {spec_id!r}")
        seen.add(spec_id)
        if type(expected) is not str or not expected.strip():
            raise ValueError(f"spec {spec_id!r}: expected must be a nonblank string")
        if position == "clean":
            if marker is not None:
                raise ValueError(f"spec {spec_id!r}: clean fixture must have no marker")
            if marker_text in expected:
                raise ValueError(f"spec {spec_id!r}: clean fixture contains the marker text")
            for hostile in HOSTILE_MARKER_STRINGS:
                if hostile in expected:
                    raise ValueError(f"spec {spec_id!r}: clean fixture contains hostile string")
        else:
            if type(marker) is not str or not marker:
                raise ValueError(f"spec {spec_id!r}: marker must be a nonblank string")
            count = expected.count(marker)
            if position == "start":
                if not expected.startswith(marker) or count != 1:
                    raise ValueError(f"spec {spec_id!r}: marker must start the answer exactly once")
            elif position == "interior":
                if count != 1 or expected.startswith(marker) or expected.endswith(marker):
                    raise ValueError(
                        f"spec {spec_id!r}: marker must sit strictly inside the answer"
                    )
            elif position == "end":
                if not expected.endswith(marker) or count != 1:
                    raise ValueError(f"spec {spec_id!r}: marker must end the answer exactly once")
            elif position == "repeat":
                if count != 2:
                    raise ValueError(f"spec {spec_id!r}: marker must appear exactly twice")
            elif position == "hostile":
                if count < 1:
                    raise ValueError(f"spec {spec_id!r}: hostile marker must appear in the answer")
            else:
                raise ValueError(f"spec {spec_id!r}: unknown position {position!r}")
    return tuple(dict(spec) for spec in specs)


def build_answer_fixtures(marker_text: str) -> tuple[dict[str, Any], ...]:
    """Build and validate the frozen fixture set for one marker text."""
    return validate_answer_fixtures(_fixture_specs(marker_text), marker_text)


def classify_observation(boundary_found: bool, suffix_nonempty: bool) -> str:
    """Total-order classifier over the detector's emission shape.

    ``collision``: the cut fired and tokens follow the marker -- the
    detector discarded content. ``ends_at_marker``: the cut fired and
    nothing follows -- the scripted emission ended at the marker, its
    own cell. ``intact``: the detector did not fire. The order is total:
    every completed observation lands in exactly one cell.
    """
    if not boundary_found:
        return "intact"
    if suffix_nonempty:
        return "collision"
    return "ends_at_marker"


def reserved_emission(
    encode: Callable[[str], Sequence[int]],
    decode: Callable[[tuple[int, ...]], str],
    expected: str,
) -> tuple[int, ...] | None:
    """The reserved arm's emission: the natural encoding of the answer.

    Returns None (fail-closed) when the backend's encode/decode does not
    round-trip the expected text exactly -- such an emission is not
    evidence about the detector (red-team A R7: the exclusion axis is
    the encode/decode round-trip). Never raises.
    """
    try:
        ids = tuple(int(i) for i in encode(expected))
    except Exception:  # noqa: BLE001 -- fail-closed: no emission, no evidence
        return None
    if any(type(i) is not int or i < 0 for i in ids) or not ids:
        return None
    try:
        if decode(ids) != expected:
            return None
    except Exception:  # noqa: BLE001 -- fail-closed: undecodable is unevaluable
        return None
    return ids


def subword_emission(
    encode: Callable[[str], Sequence[int]],
    decode: Callable[[tuple[int, ...]], str],
    expected: str,
    marker: str,
    boundary: tuple[int, ...],
) -> tuple[int, ...] | None:
    """The subword arm's emission: every marker occurrence re-encoded.

    Splits the expected text on the marker and encodes the chunks
    separately (text-level construction, the deliforge pattern), joining
    them with the marker's subword re-encoding so the emission decodes
    to exactly the expected text while containing none of the marker's
    reserved ids. Returns None when no encoding pair exists
    (``no_encoding_pair``), on decode mismatch, or when the reserved id
    would still appear in a single-id-boundary emission (a construction
    failure, never scored as a collision). Never raises.
    """
    from .deliforge import find_encoding_pair

    if type(marker) is not str or not marker:
        return None
    try:
        pair = find_encoding_pair(encode, decode, marker)
    except Exception:  # noqa: BLE001 -- fail-closed: no pair, no arm
        return None
    if pair is None:
        return None
    subword = pair["subword"]
    reserved = pair["reserved"]
    parts = expected.split(marker)
    try:
        chunks: list[int] = []
        for index, part in enumerate(parts):
            if part:
                chunks.extend(int(i) for i in encode(part))
            if index < len(parts) - 1:
                chunks.extend(subword)
        ids = tuple(chunks)
    except Exception:  # noqa: BLE001 -- fail-closed: chunk encoding failed
        return None
    if not ids or any(type(i) is not int or i < 0 for i in ids):
        return None
    if len(boundary) == 1 and boundary == reserved and reserved[0] in ids:
        # The control arm must not contain the reserved id it controls
        # for; a construction failure is not a measurement.
        return None
    try:
        if decode(ids) != expected:
            return None
    except Exception:  # noqa: BLE001 -- fail-closed: undecodable is unevaluable
        return None
    return ids


def score_observation(
    *,
    status_ok: bool,
    boundary_found: bool,
    cut: int | None,
    suffix_nonempty: bool,
    exact_match: bool,
    verbatim: bool,
    occurrences: int,
) -> dict[str, Any] | None:
    """Score one scripted observation into a class record.

    Returns None when the observation is unevaluable (the generation
    did not complete): the fixture still counts in the denominator
    bookkeeping, but in no rate. ``anomalous`` covers internally
    inconsistent outcomes (a verbatim emission that does not round-trip,
    or a completed run whose answer mismatches with no cut) -- counted
    and excluded from rates, never silently dropped.
    """
    if not status_ok:
        return None
    if not verbatim:
        return {
            "class": "anomalous",
            "cut": cut,
            "exact_match": exact_match,
            "verbatim": False,
            "occurrences": occurrences,
        }
    cls = classify_observation(boundary_found, suffix_nonempty)
    if cls == "intact" and not exact_match:
        cls = "anomalous"
    return {
        "class": cls,
        "cut": cut,
        "exact_match": exact_match,
        "verbatim": True,
        "occurrences": occurrences,
    }


def collision_cell(
    per_fixture: Sequence[dict[str, Any]],
    *,
    n_fixtures: int,
    boundary_len: int,
) -> dict[str, Any]:
    """Aggregate one boundary configuration's scored observations.

    ``per_fixture`` entries: ``{"id", "reserved": scored|None,
    "subword": scored|None, "arm_state": {"reserved": ..., "subword":
    ...}}`` where each ``arm_state`` is one of "scored" (a scored
    record), "failed" (generation did not complete), "roundtrip"
    (encode/decode did not round-trip the expected answer -- the R7
    exclusion axis), "no_pair" (no subword encoding pair exists), or
    "not_applicable" (the clean fixture has no subword arm). Rates are
    over scored, non-anomalous observations only; every other outcome
    is counted in its own bucket, never silent.
    """
    if type(n_fixtures) is not int or n_fixtures < 1:
        raise ValueError("n_fixtures must be a positive integer")
    if type(boundary_len) is not int or boundary_len < 1:
        raise ValueError("boundary_len must be a positive integer")
    arms: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        scored = [
            f[arm]
            for f in per_fixture
            if f.get("arm_state", {}).get(arm) == "scored" and type(f.get(arm)) is dict
        ]
        counts = {cls: sum(1 for s in scored if s.get("class") == cls) for cls in CLASSES}
        n_anomalous = counts["anomalous"]
        n_scored = len(scored) - n_anomalous
        n_failed = sum(1 for f in per_fixture if f.get("arm_state", {}).get(arm) == "failed")
        n_roundtrip = sum(1 for f in per_fixture if f.get("arm_state", {}).get(arm) == "roundtrip")
        n_no_pair = sum(1 for f in per_fixture if f.get("arm_state", {}).get(arm) == "no_pair")
        n_na = sum(1 for f in per_fixture if f.get("arm_state", {}).get(arm) == "not_applicable")
        arms[arm] = {
            "n_evaluable": len(scored),
            "n_scored": n_scored,
            "n_collision": counts["collision"],
            "n_ends_at_marker": counts["ends_at_marker"],
            "n_intact": counts["intact"],
            "n_anomalous": n_anomalous,
            "n_failed": n_failed,
            "n_roundtrip_excluded": n_roundtrip,
            "n_no_pair": n_no_pair,
            "n_not_applicable": n_na,
            "collision_rate": (counts["collision"] / n_scored) if n_scored else None,
            "ends_rate": (counts["ends_at_marker"] / n_scored) if n_scored else None,
        }
    reserved = arms["reserved"]
    subword = arms["subword"]
    total_scored = reserved["n_scored"] + subword["n_scored"]
    any_collision = reserved["n_collision"] > 0 or subword["n_collision"] > 0
    uniform_intact = (
        total_scored > 0
        and reserved["n_collision"] == 0
        and reserved["n_ends_at_marker"] == 0
        and subword["n_collision"] == 0
        and subword["n_ends_at_marker"] == 0
    )
    return {
        "n": n_fixtures,
        "boundary_ids": boundary_len,
        "reserved": reserved,
        "subword": subword,
        "any_collision": any_collision,
        "uniform_intact": uniform_intact,
        "evaluable": reserved["n_evaluable"] >= 1,
    }


def protocol_fingerprint() -> str:
    """sha256 over the canonical frozen protocol (templates, positions,
    hostile set, classification rule, arms).
    """
    payload = {
        "version": PROTOCOL_VERSION,
        "own_templates": [list(t) for t in _OWN_TEMPLATES],
        "hostile_template": _HOSTILE_TEMPLATE,
        "clean_text": _CLEAN_TEXT,
        "hostile_strings": list(HOSTILE_MARKER_STRINGS),
        "classes": list(CLASSES),
        "arms": list(ARMS),
        "rule": (
            "collision=cut and suffix nonempty; ends_at_marker=cut and suffix empty; intact=no cut"
        ),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def protocol_dict() -> dict[str, Any]:
    """The serializable protocol block stored in the stage payload."""
    return {
        "version": PROTOCOL_VERSION,
        "fingerprint": protocol_fingerprint(),
        "n_fixtures": n_fixtures_per_config(),
        "arms": list(ARMS),
        "classes": list(CLASSES),
    }


def answer_collision_profile(
    candidates: tuple[str, ...],
    per_candidate: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate boundary-configuration cells into the run-level profile.

    Available only when at least one candidate's cell is measurable (a
    reserved-arm scored observation exists); otherwise the audit is
    uninformative and reports ``{"available": False, "detail":
    "answer_collision_unavailable"}``. ``uniform_intact`` is the
    publishable negative: every measurable configuration's detector left
    every marker-bearing correct answer untouched -- the regression-guard
    pass. ``colliding_candidates`` names the configurations whose
    detector provably truncated a verbatim-correct answer.
    """
    cells: dict[str, dict[str, Any]] = {}
    for name in candidates:
        entry = per_candidate.get(name)
        if type(entry) is dict:
            cells[name] = entry
    measurable = {
        name: cell
        for name, cell in cells.items()
        if type(cell.get("reserved")) is dict
        and type(cell["reserved"].get("n_scored")) is int
        and cell["reserved"]["n_scored"] >= 1
    }
    base: dict[str, Any] = {
        "n_candidates": len(candidates),
        "n_cells": len(cells),
        "n_measurable": len(measurable),
        "protocol": protocol,
        "per_candidate": {name: per_candidate.get(name) for name in candidates},
    }
    if not measurable:
        return {
            "available": False,
            "detail": "answer_collision_unavailable",
            **base,
            "any_collision": False,
            "uniform_intact": False,
            "colliding_candidates": [],
            "max_collision_rate": None,
            "worst_candidate": None,
        }
    rates = {
        name: cell["reserved"]["collision_rate"]
        for name, cell in measurable.items()
        if type(cell["reserved"].get("collision_rate")) in (int, float)
    }
    worst = max(rates, key=lambda name: (rates[name], name)) if rates else None
    colliding = sorted(
        name for name, cell in measurable.items() if cell.get("any_collision") is True
    )
    return {
        "available": True,
        **base,
        "any_collision": any(cell.get("any_collision") is True for cell in measurable.values()),
        "uniform_intact": all(cell.get("uniform_intact") is True for cell in measurable.values()),
        "colliding_candidates": colliding,
        "max_collision_rate": max(rates.values()) if rates else None,
        "worst_candidate": worst,
    }
