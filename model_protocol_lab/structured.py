"""Structured-output (JSON mode) adherence per template (research wave-3 idea 5).

The question: prompt-only JSON adherence -- parse-valid rate and
schema-valid rate, with no constrained decoder -- is uniform across
template families for a fixed schema set? JSONSchemaBench (arXiv:2501.10868v2,
ICML 2025) varies frameworks and models; askpert.dev ("Does JSON Mode Make
Your Model Worse at Reasoning?", Sep 2026) names the prompt/template as an
explicit confound. Nobody varies the *chat template* as the independent
variable while holding model, schema, and decoder fixed. The lab already
has the exact machinery (template families, exact token sequences,
fixture split, conservative selection) to run the missing experiment --
and prompt-only adherence is the honest baseline every
constrained-decoding paper omits.

Design: synthetic fixtures, one per schema spec. Each spec has a benign
question, a context, a fixed flat object schema (extraction /
classification / routing shapes, no nested-schema pathologies), and a
known-correct answer document. The fixture's ``expected`` is the
canonical JSON rendering of that answer, so the pipeline's
``exact_match`` still grades strictly; adherence (``parse_ok`` /
``schema_ok``) is measured independently on the decoded answer text by
:func:`classify_response`, with a deterministic hand-rolled schema
validator -- the lab has zero runtime dependencies, so there is no
``jsonschema`` package to lean on and none is needed for the flat
dialect used here.

No constrained decoding in v1: that keeps the backend contract identical
to every other experiment (greedy text generation) and keeps the claim
honest -- what the template alone elicits.

Pre-registered claims (falsifiable):

1. Adherence is not uniform across candidates: the max-minus-min
   ``adherence_gap`` over evaluable candidates is either ~0 (uniform,
   the null hypothesis -- reported, never asserted) or measurably
   positive.
2. Parse-validity and schema-validity decompose: a response can parse
   yet violate the schema (``schema_failed``), and the two rates are
   reported separately rather than conflated.
3. Classification is total and fail-closed: non-string responses become
   ``not_text`` (excluded from rates, never silently dropped), and a
   malformed schema definition can never produce a phantom
   ``adherent`` verdict.

Everything here is pure: spec validation, the schema dialect,
classification, and rate aggregation need no model.
"""

from __future__ import annotations

import json
from typing import Any

from .core import Fixture, Message

#: Fixture shapes, in canonical order. Each shape exercises a different
#: failure mode: extraction (wrong types), classification (wrong enum
#: member), routing (range/boolean violations).
SHAPES = ("extraction", "classification", "routing")

#: Field types supported by the flat schema dialect.
FIELD_TYPES = ("string", "integer", "number", "boolean", "null")

#: Response outcome vocabulary. ``not_text`` marks a non-string response
#: (fail-closed: excluded from rates); ``schema_invalid`` marks a
#: malformed schema definition (never a phantom adherent).
OUTCOMES = ("adherent", "parse_failed", "schema_failed", "not_text")

#: Synthetic structured-output fixture specifications. Each spec is a
#: benign, fully deterministic task: the context contains the answer,
#: the schema is flat (no nesting), and the answer document is
#: unambiguous. The prompt asks for ONLY the JSON object so adherence
#: measures what the template elicits without further instruction.
STRUCTURED_SPECS: tuple[dict[str, object], ...] = (
    {
        "id": "extract-person",
        "shape": "extraction",
        "question": "Extract the person's name and age.",
        "context": "Ada Lovelace was 36 when she wrote the first algorithm.",
        "schema": {
            "type": "object",
            "required": ["name", "age"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0},
            },
            "additionalProperties": False,
        },
        "answer": {"name": "Ada Lovelace", "age": 36},
    },
    {
        "id": "extract-city",
        "shape": "extraction",
        "question": "Extract the city and its population.",
        "context": "Lisbon has about 500000 residents.",
        "schema": {
            "type": "object",
            "required": ["city", "population"],
            "properties": {
                "city": {"type": "string"},
                "population": {"type": "integer", "minimum": 0},
            },
            "additionalProperties": False,
        },
        "answer": {"city": "Lisbon", "population": 500000},
    },
    {
        "id": "extract-book",
        "shape": "extraction",
        "question": "Extract the book title and its publication year.",
        "context": "The novel 'Dune' was first published in 1965.",
        "schema": {
            "type": "object",
            "required": ["title", "year"],
            "properties": {
                "title": {"type": "string"},
                "year": {"type": "integer", "minimum": 0, "maximum": 2100},
            },
            "additionalProperties": False,
        },
        "answer": {"title": "Dune", "year": 1965},
    },
    {
        "id": "classify-sentiment",
        "shape": "classification",
        "question": "Classify the sentiment of the review.",
        "context": "Review: 'The service was dreadful and painfully slow.'",
        "schema": {
            "type": "object",
            "required": ["label"],
            "properties": {
                "label": {"type": "string", "enum": ["positive", "negative"]},
            },
            "additionalProperties": False,
        },
        "answer": {"label": "negative"},
    },
    {
        "id": "classify-urgency",
        "shape": "classification",
        "question": "Classify the urgency of the ticket.",
        "context": "Ticket: 'The server room is on fire.'",
        "schema": {
            "type": "object",
            "required": ["level"],
            "properties": {
                "level": {"type": "string", "enum": ["low", "high"]},
            },
            "additionalProperties": False,
        },
        "answer": {"level": "high"},
    },
    {
        "id": "classify-language",
        "shape": "classification",
        "question": "Classify the language of the sentence.",
        "context": "Sentence: 'Guten Morgen, wie geht es dir?'",
        "schema": {
            "type": "object",
            "required": ["language"],
            "properties": {
                "language": {
                    "type": "string",
                    "enum": ["english", "german", "french"],
                },
            },
            "additionalProperties": False,
        },
        "answer": {"language": "german"},
    },
    {
        "id": "route-ticket",
        "shape": "routing",
        "question": "Route the support ticket and set its priority.",
        "context": "Ticket: 'My invoice shows a charge I do not recognize.'",
        "schema": {
            "type": "object",
            "required": ["department", "priority"],
            "properties": {
                "department": {
                    "type": "string",
                    "enum": ["sales", "support", "billing"],
                },
                "priority": {"type": "integer", "minimum": 1, "maximum": 5},
            },
            "additionalProperties": False,
        },
        "answer": {"department": "billing", "priority": 3},
    },
    {
        "id": "route-alert",
        "shape": "routing",
        "question": "Route the alert and say whether it needs acknowledgement.",
        "context": "Alert: 'Disk usage on db-1 reached 97%.'",
        "schema": {
            "type": "object",
            "required": ["channel", "ack_required"],
            "properties": {
                "channel": {"type": "string", "enum": ["email", "sms", "pager"]},
                "ack_required": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
        "answer": {"channel": "pager", "ack_required": True},
    },
    {
        "id": "route-order",
        "shape": "routing",
        "question": "Route the order to a warehouse and flag express shipping.",
        "context": "Order: 'Customer in Porto needs the parcel by Friday.'",
        "schema": {
            "type": "object",
            "required": ["warehouse", "express"],
            "properties": {
                "warehouse": {"type": "string", "enum": ["north", "south"]},
                "express": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
        "answer": {"warehouse": "south", "express": True},
    },
)

_SPEC_KEYS = ("id", "question", "context", "shape", "schema", "answer")


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return value == value and abs(value) != float("inf")


def validate_field_def(field: object, *, where: str) -> dict[str, Any]:
    """Validate one flat field definition; returns the normalized form.

    A field definition is ``{"type": <one of FIELD_TYPES>}`` with
    optional ``"enum"`` (a nonempty list of JSON scalars), and optional
    ``"minimum"``/``"maximum"`` (finite numbers, integer fields require
    integral bounds) for ``integer``/``number`` types. No nesting: a
    field whose type is ``"object"`` or ``"array"`` is rejected, keeping
    the dialect flat by construction. Raises ``ValueError`` on any
    violation; the returned dict carries only the applicable keys.
    """
    if type(field) is not dict:
        raise ValueError(f"{where}: field definition must be a dict")
    field_type = field.get("type")
    if field_type not in FIELD_TYPES:
        raise ValueError(f"{where}: unknown field type {field_type!r}")
    for key in field:
        if key not in ("type", "enum", "minimum", "maximum"):
            raise ValueError(f"{where}: unknown field key {key!r}")
    normalized: dict[str, Any] = {"type": field_type}
    enum = field.get("enum")
    if enum is not None:
        if type(enum) is not list or not enum:
            raise ValueError(f"{where}: enum must be a nonempty list")
        for member in enum:
            if type(member) not in (str, int, float, bool) and member is not None:
                raise ValueError(f"{where}: enum members must be JSON scalars")
        normalized["enum"] = list(enum)
    for bound in ("minimum", "maximum"):
        value = field.get(bound)
        if value is not None:
            if not _is_finite_number(value):
                raise ValueError(f"{where}: {bound} must be a finite number")
            if field_type == "integer" and int(value) != value:
                raise ValueError(f"{where}: {bound} must be integral for integer fields")
            normalized[bound] = value
    minimum = normalized.get("minimum")
    maximum = normalized.get("maximum")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError(f"{where}: minimum exceeds maximum")
    return normalized


def validate_schema_def(schema: object) -> dict[str, Any]:
    """Validate a flat object schema definition; returns the normalized form.

    The dialect: ``{"type": "object", "required": [names...],
    "properties": {name: field_def}, "additionalProperties": bool}``.
    Every required name must be a declared property; every property gets
    a flat field definition (see :func:`validate_field_def`). Raises
    ``ValueError`` on any violation. The returned dict is a normalized
    copy safe to thread through the rest of the pipeline without
    re-validation.
    """
    if type(schema) is not dict:
        raise ValueError("schema must be a dict")
    if schema.get("type") != "object":
        raise ValueError("schema type must be 'object'")
    required = schema.get("required")
    if (
        type(required) is not list
        or not required
        or not all(type(name) is str for name in required)
    ):
        raise ValueError("schema required must be a nonempty list of strings")
    properties = schema.get("properties")
    if (
        type(properties) is not dict
        or not properties
        or not all(type(name) is str for name in properties)
    ):
        raise ValueError("schema properties must be a nonempty dict of string names")
    additional = schema.get("additionalProperties")
    if type(additional) is not bool:
        raise ValueError("schema additionalProperties must be a bool")
    for key in schema:
        if key not in ("type", "required", "properties", "additionalProperties"):
            raise ValueError(f"schema has unknown key {key!r}")
    normalized_properties: dict[str, dict[str, Any]] = {}
    for name in required:
        if name not in properties:
            raise ValueError(f"required field {name!r} is not a declared property")
        normalized_properties[name] = validate_field_def(
            properties[name], where=f"property {name!r}"
        )
    for name, field in properties.items():
        if name not in normalized_properties:
            normalized_properties[name] = validate_field_def(field, where=f"property {name!r}")
    return {
        "type": "object",
        "required": list(required),
        "properties": normalized_properties,
        "additionalProperties": additional,
    }


def _canonical_json(document: object) -> str:
    """Canonical JSON rendering: sorted keys, fixed separators."""
    return json.dumps(document, sort_keys=True, separators=(", ", ": "))


def validate_structured_specs(specs: object) -> tuple[dict[str, object], ...]:
    """Validate synthetic structured-output fixture specifications.

    Returns the specs as a tuple of plain dicts. Raises ``ValueError``
    on any malformed spec: not a dict, missing/blank ``id``/``question``/
    ``context``, duplicate ids, unknown ``shape``, malformed ``schema``
    (see :func:`validate_schema_def`), or an ``answer`` that is not a
    dict passing its own schema (fail-closed: the answer is the
    canonical truth the pipeline grades against, so it must be valid).
    """
    if type(specs) is not tuple or not specs:
        raise ValueError("specs must be a nonempty tuple")
    seen: set[str] = set()
    validated: list[dict[str, object]] = []
    for spec in specs:
        if type(spec) is not dict:
            raise ValueError("each spec must be a dict")
        for key in spec:
            if key not in _SPEC_KEYS:
                raise ValueError(f"spec has unknown key {key!r}")
        spec_id = spec.get("id")
        if type(spec_id) is not str or not spec_id.strip():
            raise ValueError("spec id must be a nonblank string")
        if spec_id in seen:
            raise ValueError(f"duplicate spec id {spec_id!r}")
        seen.add(spec_id)
        for key in ("question", "context"):
            value = spec.get(key)
            if type(value) is not str or not value.strip():
                raise ValueError(f"spec {spec_id!r}: {key} must be a nonblank string")
        shape = spec.get("shape")
        if shape not in SHAPES:
            raise ValueError(f"spec {spec_id!r}: unknown shape {shape!r}")
        schema = spec.get("schema")
        try:
            schema = validate_schema_def(schema)
        except ValueError as exc:
            raise ValueError(f"spec {spec_id!r}: invalid schema: {exc}") from exc
        answer = spec.get("answer")
        if type(answer) is not dict:
            raise ValueError(f"spec {spec_id!r}: answer must be a dict")
        verdict = check_schema(answer, schema)  # schema just validated: total here
        if verdict != "schema_ok":
            raise ValueError(f"spec {spec_id!r}: answer does not satisfy its own schema")
        validated.append(
            {
                "id": spec_id,
                "shape": shape,
                "question": spec["question"],
                "context": spec["context"],
                "schema": schema,
                "answer": answer,
            }
        )
    return tuple(validated)


def describe_schema(schema: object) -> str:
    """Render a deterministic one-line description of a flat schema.

    The description names every property with its type, enum, and
    bounds, then the required set and the additional-properties rule --
    enough for a model to reconstruct the exact JSON shape, and stable
    across runs (properties sorted by name). The schema is validated
    first (raising ``ValueError`` on hostile input); validated specs
    already carry the normalized form, so this is a no-op for them.
    """
    normalized = validate_schema_def(schema)
    properties = normalized["properties"]
    parts: list[str] = []
    for name in sorted(properties):
        field = properties[name]
        piece = f'"{name}": {field["type"]}'
        enum = field.get("enum")
        if enum is not None:
            piece += " one of " + _canonical_json(enum)
        minimum = field.get("minimum")
        maximum = field.get("maximum")
        if minimum is not None or maximum is not None:
            lo = f">= {minimum}" if minimum is not None else ""
            hi = f"<= {maximum}" if maximum is not None else ""
            piece += f" ({' and '.join(part for part in (lo, hi) if part)})"
        parts.append(piece)
    required = normalized["required"]
    if normalized["additionalProperties"] is False:
        extras = "no other keys allowed"
    else:
        extras = "other keys allowed"
    return (
        "{"
        + "; ".join(parts)
        + "} (required: "
        + ", ".join(sorted(str(name) for name in required))
        + "; "
        + extras
        + ")"
    )


def render_prompt(spec: dict[str, object]) -> str:
    """Render the fixture prompt for one validated spec.

    The prompt carries the task, the context, and the schema
    description, and ends with a fixture-identity marker
    ``(ref: structured/<id>)`` so scripted backends can key their
    policy per fixture deterministically. The marker is inert for real
    models.
    """
    schema = spec["schema"]
    normalized = validate_schema_def(schema)  # raises ValueError on hostile input
    return (
        f"{spec['question']}\n"
        f"Context: {spec['context']}\n"
        "Respond with ONLY a JSON object matching this schema: "
        f"{describe_schema(normalized)}.\n"
        "Do not include any other text.\n"
        f"(ref: structured/{spec['id']})"
    )


def build_structured_fixtures(
    specs: tuple[dict[str, object], ...],
) -> tuple[Fixture, ...]:
    """Build lab ``Fixture`` objects from validated specs.

    Each fixture's ``expected`` is the canonical JSON rendering of the
    spec's answer, so the pipeline's ``exact_match`` grades strictly
    (a model must emit the canonical form byte-for-byte to pass); the
    adherence analysis in :func:`classify_response` measures
    parse/schema validity independently of that strict verdict.
    """
    fixtures: list[Fixture] = []
    for spec in specs:
        answer = spec["answer"]
        fixtures.append(
            Fixture(
                id=f"structured/{spec['id']}",
                messages=(Message(role="user", content=render_prompt(spec)),),
                expected=_canonical_json(answer),
                group="structured",
            )
        )
    return tuple(fixtures)


def schema_map(specs: tuple[dict[str, object], ...]) -> dict[str, dict[str, object]]:
    """Map fixture id -> normalized schema dict for validated specs."""
    result: dict[str, dict[str, object]] = {}
    for spec in specs:
        schema = validate_schema_def(spec["schema"])  # raises ValueError on hostile input
        result[f"structured/{spec['id']}"] = schema
    return result


def check_schema(document: object, schema: object) -> str:
    """Check a parsed JSON document against a flat schema definition.

    Returns ``"schema_ok"``, ``"schema_failed"``, or
    ``"schema_invalid"`` (the schema definition itself is malformed --
    fail-closed, never a phantom ok). Total: never raises.
    """
    try:
        normalized = validate_schema_def(schema)
    except ValueError:
        return "schema_invalid"
    if type(document) is not dict:
        return "schema_failed"
    required = normalized["required"]
    properties = normalized["properties"]
    for name in required:
        if name not in document:
            return "schema_failed"
    for name, value in document.items():
        field = properties.get(name)
        if field is None:
            if normalized["additionalProperties"] is False:
                return "schema_failed"
            continue
        if not _field_value_ok(value, field):
            return "schema_failed"
    return "schema_ok"


def _field_value_ok(value: object, field: dict[str, Any]) -> bool:
    """Check one JSON value against one flat field definition."""
    field_type = field["type"]
    if field_type == "string":
        ok = type(value) is str
    elif field_type == "integer":
        ok = type(value) is int and not isinstance(value, bool)
    elif field_type == "number":
        ok = type(value) in (int, float) and not isinstance(value, bool)
    elif field_type == "boolean":
        ok = type(value) is bool
    elif field_type == "null":
        ok = value is None
    else:  # unreachable: validated upstream; defensive for hostile callers
        return False
    if not ok:
        return False
    enum = field.get("enum")
    if enum is not None and value not in enum:
        return False
    minimum = field.get("minimum")
    if minimum is not None and isinstance(minimum, (int, float)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        if value < minimum:
            return False
    maximum = field.get("maximum")
    if maximum is not None and isinstance(maximum, (int, float)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        return value <= maximum
    return True


def classify_response(text: object, schema: object) -> dict[str, str]:
    """Classify one decoded response against a schema definition.

    Returns ``{"parse": ..., "schema": ..., "outcome": ...}`` where
    ``parse`` is ``parse_ok`` / ``parse_failed`` / ``not_text``,
    ``schema`` is ``schema_ok`` / ``schema_failed`` / ``not_evaluated``
    / ``schema_invalid``, and ``outcome`` is one of :data:`OUTCOMES` --
    ``adherent`` exactly when the response parsed *and* satisfied the
    schema. Total: never raises on hostile input.
    """
    if type(text) is not str:
        return {"parse": "not_text", "schema": "not_evaluated", "outcome": "not_text"}
    try:
        document = json.loads(text)
    except (ValueError, TypeError, RecursionError):
        return {"parse": "parse_failed", "schema": "not_evaluated", "outcome": "parse_failed"}
    schema_verdict = check_schema(document, schema)
    if schema_verdict == "schema_invalid":
        return {"parse": "parse_ok", "schema": "schema_invalid", "outcome": "not_text"}
    if schema_verdict == "schema_ok":
        return {"parse": "parse_ok", "schema": "schema_ok", "outcome": "adherent"}
    return {"parse": "parse_ok", "schema": "schema_failed", "outcome": "schema_failed"}


def candidate_adherence(outcomes: list[str]) -> dict[str, object]:
    """Aggregate per-response outcomes into per-candidate adherence rates.

    ``outcomes`` holds one :data:`OUTCOMES` value per response.
    ``not_text`` responses are fail-closed exclusions (counted in ``n``
    but not in ``n_evaluable``); every rate is in [0, 1] or None when
    there is nothing evaluable. ``adherent`` implies parse-valid, so
    ``adherent_rate`` <= ``parse_rate`` always.
    """
    for outcome in outcomes:
        if outcome not in OUTCOMES:
            raise ValueError(f"unknown outcome {outcome!r}")
    evaluable = [outcome for outcome in outcomes if outcome != "not_text"]
    n_evaluable = len(evaluable)
    parse_ok = sum(1 for outcome in evaluable if outcome in ("adherent", "schema_failed"))
    schema_ok = sum(1 for outcome in evaluable if outcome == "adherent")
    return {
        "n": len(outcomes),
        "n_evaluable": n_evaluable,
        "parse_ok": parse_ok,
        "schema_ok": schema_ok,
        "adherent": schema_ok,
        "parse_rate": parse_ok / n_evaluable if n_evaluable else None,
        "schema_rate": schema_ok / n_evaluable if n_evaluable else None,
        "adherent_rate": schema_ok / n_evaluable if n_evaluable else None,
    }
