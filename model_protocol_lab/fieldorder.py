"""Schema field-order probe: does the answer field's position change its value?

The question (docs/FIELDORDER_DESIGN.md, cycle 205): for a fixed
candidate (chat template) and a fixed set of flat JSON
extraction/classification/routing specs, is the value produced
for a spec's designated answer field invariant under the field's
position in the presented schema? Tam et al. (arXiv:2408.02442)
establish that format restrictions degrade reasoning, and their
section 4.1 reports emitted key order as an important factor
under JSON mode; whether the order in which a prompt *presents*
a schema's fields changes the values produced for them, crossed
with chat templates and scored per field, is the unmeasured
combination this probe measures.

Design: the shipped ``structured`` spec set (9 specs) is reused
unchanged and extended with six 3-4-field specs frozen in this
module (the shipped schemas alone collapse the four arms to at
most two distinct orders). Each spec designates one answer
field and a frozen ``paired`` flag; two shipped specs are
excluded from pairing because their frozen answer labels are
not entailed by their contexts (route-alert's ``pager``,
route-order's ``south``) -- the reader-entailment rule and the
full adjudication are in the design doc. Arms per spec:
``canonical`` (alphabetical, the shipped presentation),
``answer_first``, ``answer_last``, ``reversed``; the four orders
are all distinct only when the answer field sits in a middle
position, and the aggregation prints the distinct-prompt count
(37 of 60) rather than hiding the degeneracy.

Records commit to the parsed document as per-key canonical
JSON strings (``structured._canonical_json`` is the pinned
canonical form), with None as the sentinel for a value that has
no canonical form (a non-finite number, or a value beyond the
int<->str digit limit). Per-field correctness, the schema
verdict, and adherence are derived from that commitment by this
module's aggregator and by the diff validator's mirror alike;
satisfaction is never stored. Stated residue: the link between
``answer_digest`` and the record's document fields is
stage-attested (origin only), as is ``duplicate_keys`` -- every
consequence drawn from them is recomputed.

Pre-registered claim (falsifiable): per-field value accuracy
is invariant under schema field order. Per candidate it is
falsified iff, over specs paired by the frozen rule (paired
flag set, endpoint orders differ, both endpoint records
adherent, neither with duplicate keys), the answer field's
discordant specs number at least MIN_DISCORDANT_SPECS and the
exact two-sided sign test over them gives p <= 0.05; below the
discordant floor the verdict is ``insufficient`` -- the boolean
is withheld, never defaulted to acquittal. The decision's
operating characteristics are printed with it: at 6-8 discordant
specs only a unanimous split falsifies, and a true 0.90-vs-0.60
endpoint gap falsifies only about 15% of the time.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from .structured import _canonical_json

#: Arms in canonical order.
ARM_KEYS: tuple[str, ...] = ("canonical", "answer_first", "answer_last", "reversed")

#: Fixture group for the probe's synthetic fixtures.
GROUP = "fieldorder"

#: Generation budget per (spec, arm) cell, fixed by the protocol:
#: the record caps below are derived from it (design NB4).
MAX_NEW_TOKENS = 64

#: Records per candidate: 15 specs x 4 arms.
SPEC_COUNT = 15

#: Sentinel for an integer literal beyond the int<->str digit
#: limit: it has no canonical JSON form, so it canonicalises to
#: None exactly like a non-finite number (design section 4).
_OVERSIZE_INT = object()
RECORDS_PER_CANDIDATE = SPEC_COUNT * len(ARM_KEYS)
MAX_FIELD_ORDER_RECORDS = RECORDS_PER_CANDIDATE * 24

#: Payload cap, derived from the enforced per-record cap:
#: 1,440 records x (4,096 chars of field values + ~1.4 KB of
#: other fields and JSON overhead) ~= 7.9 MB.
MAX_FIELD_ORDER_PAYLOAD_BYTES = 8 * 1024 * 1024

#: Per-record cap on the summed length of ``field_values``
#: strings, validator-enforced. Derivation: at most 32 decoded
#: chars per generated token for the tokenizers this lab
#: targets, so a 64-token answer is at most 2,048 chars; the
#: cap is 2x that.
FIELD_VALUES_CAP = 4096

#: Minimum discordant specs for a falsified / not_falsified
#: verdict; below it the verdict is ``insufficient``.
MIN_DISCORDANT_SPECS = 6

#: Per-candidate level for the sign-test decision.
SIGN_ALPHA = 0.05

GENERATION_RULE_TEXT = (
    "The spec set is the frozen FIELD_ORDER_SPECS table: the nine "
    "structured specs (source 'structured', reused unchanged) plus "
    "six fieldorder specs. Each spec designates one answer field "
    "and a paired flag; a spec may be paired only if a reader "
    "given question, context and schema (not the answer table) "
    "can derive the answer field's value uniquely. Canonical "
    "order is alphabetical by field name. Arms: canonical; "
    "answer_first (answer field first, the rest in canonical "
    "relative order); answer_last (answer field last, the rest "
    "in canonical relative order); reversed (canonical order "
    "reversed). Generation is greedy via core.evaluate at a "
    "fixed 64 new tokens; exactly 60 records per candidate in "
    "(spec, arm) canonical order. Record budget 1440 (24 "
    "candidates); payload budget 8388608 bytes."
)

RENDERING_RULE_TEXT = (
    "The user content is the structured prompt template with the "
    "schema description rendered in the arm's order: question, "
    "then 'Context: ', then 'Respond with ONLY a JSON object "
    "matching this schema: ' plus the ordered description, then "
    "'Do not include any other text.' and the ref marker "
    "(structured/<id> for shipped specs, fieldorder/<id> for new "
    "specs). The ordered description lists each property in the "
    'arm order as "name": type with enum and bound annotations '
    "in the structured format, then the required list in the arm "
    "order and the additional-properties clause. The canonical "
    "arm's prompt is byte-identical to structured.render_prompt."
)

CHECKING_RULE_TEXT = (
    "Canonical JSON is structured._canonical_json: json.dumps "
    "with sort_keys=True, separators=(', ', ': '), "
    "ensure_ascii=True, allow_nan=False; the document "
    "{'name': 'Ada Lovelace', 'age': 36} canonicalises to "
    '\'{"age": 36, "name": "Ada Lovelace"}\'. The answer is '
    "parsed once with a duplicate-detecting object_pairs_hook "
    "(last occurrence wins). document_kind is 'object', "
    "'non_object', or 'unparseable' (JSON parse failure only). "
    "field_values maps each produced key to its canonical JSON "
    "string, or to None when the value has no canonical form (a "
    "non-finite number, or a value beyond the int/str digit "
    "limit); a produced JSON null is the string 'null', never "
    "None. document_digest is non-None iff kind is 'object' and "
    "no field value is None, and then equals the SHA-256 of the "
    "document's canonical form. A field is correct iff its key "
    "is present with a string whose digest equals the expected "
    "value's canonical digest. The schema verdict is recomputed "
    "from the reconstructed document with the flat schema check; "
    "a None value fails its key's check before reconstruction. "
    "Any exception while parsing or canonicalising a stored "
    "string makes the payload section invalid. parse_ok iff kind "
    "is 'object' or 'non_object'; adherent iff parse_ok, the "
    "schema verdict is ok, and duplicate_keys is false. Pairing: "
    "paired flag set, endpoint orders differ, both endpoint "
    "records adherent, neither with duplicate keys. Decision: "
    "exact two-sided sign test over discordant paired specs; "
    "falsified iff discordant >= 6 and p <= 0.05; insufficient "
    "iff discordant < 6; else not_falsified. Per-record field "
    "values are capped at 4096 summed chars; a live record over "
    "the cap is stored with detail 'arm_evaluation_failed' and "
    "empty field values."
)


FIELD_ORDER_SPECS: tuple[dict[str, Any], ...] = (
    {
        "id": "extract-person",
        "shape": "extraction",
        "question": "Extract the person's name and age.",
        "context": "Ada Lovelace was 36 when she wrote the first algorithm.",
        "schema": {
            "type": "object",
            "required": ["name", "age"],
            "properties": {"name": {"type": "string"}, "age": {"type": "integer", "minimum": 0}},
            "additionalProperties": False,
        },
        "answer": {"name": "Ada Lovelace", "age": 36},
        "answer_field": "name",
        "paired": True,
        "source": "structured",
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
        "answer_field": "city",
        "paired": True,
        "source": "structured",
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
        "answer_field": "title",
        "paired": True,
        "source": "structured",
    },
    {
        "id": "classify-sentiment",
        "shape": "classification",
        "question": "Classify the sentiment of the review.",
        "context": "Review: 'The service was dreadful and painfully slow.'",
        "schema": {
            "type": "object",
            "required": ["label"],
            "properties": {"label": {"type": "string", "enum": ["positive", "negative"]}},
            "additionalProperties": False,
        },
        "answer": {"label": "negative"},
        "answer_field": "label",
        "paired": False,
        "source": "structured",
    },
    {
        "id": "classify-urgency",
        "shape": "classification",
        "question": "Classify the urgency of the ticket.",
        "context": "Ticket: 'The server room is on fire.'",
        "schema": {
            "type": "object",
            "required": ["level"],
            "properties": {"level": {"type": "string", "enum": ["low", "high"]}},
            "additionalProperties": False,
        },
        "answer": {"level": "high"},
        "answer_field": "level",
        "paired": False,
        "source": "structured",
    },
    {
        "id": "classify-language",
        "shape": "classification",
        "question": "Classify the language of the sentence.",
        "context": "Sentence: 'Guten Morgen, wie geht es dir?'",
        "schema": {
            "type": "object",
            "required": ["language"],
            "properties": {"language": {"type": "string", "enum": ["english", "german", "french"]}},
            "additionalProperties": False,
        },
        "answer": {"language": "german"},
        "answer_field": "language",
        "paired": False,
        "source": "structured",
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
                "department": {"type": "string", "enum": ["sales", "support", "billing"]},
                "priority": {"type": "integer", "minimum": 1, "maximum": 5},
            },
            "additionalProperties": False,
        },
        "answer": {"department": "billing", "priority": 3},
        "answer_field": "department",
        "paired": True,
        "source": "structured",
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
        "answer_field": "channel",
        "paired": False,
        "source": "structured",
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
        "answer_field": "warehouse",
        "paired": False,
        "source": "structured",
    },
    {
        "id": "extract-product",
        "shape": "extraction",
        "question": "Extract the product's brand, model, price and stock status.",
        "context": "The catalogue lists it as: brand Acme, model Rotor X2. "
        "It costs 349 dollars and is currently in stock.",
        "schema": {
            "type": "object",
            "required": ["brand", "model", "price", "in_stock"],
            "properties": {
                "brand": {"type": "string"},
                "model": {"type": "string"},
                "price": {"type": "integer", "minimum": 0},
                "in_stock": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
        "answer": {"brand": "Acme", "model": "Rotor X2", "price": 349, "in_stock": True},
        "answer_field": "model",
        "paired": True,
        "source": "fieldorder",
    },
    {
        "id": "extract-event",
        "shape": "extraction",
        "question": "Extract the event venue, day and attendance.",
        "context": "The workshop meets at Harbor Hall on Friday with 40 attendees.",
        "schema": {
            "type": "object",
            "required": ["venue", "day", "attendance"],
            "properties": {
                "venue": {"type": "string"},
                "day": {
                    "type": "string",
                    "enum": [
                        "monday",
                        "tuesday",
                        "wednesday",
                        "thursday",
                        "friday",
                        "saturday",
                        "sunday",
                    ],
                },
                "attendance": {"type": "integer", "minimum": 0},
            },
            "additionalProperties": False,
        },
        "answer": {"venue": "Harbor Hall", "day": "friday", "attendance": 40},
        "answer_field": "venue",
        "paired": True,
        "source": "fieldorder",
    },
    {
        "id": "route-complaint",
        "shape": "routing",
        "question": "Route the complaint and record whether a defect was "
        "confirmed and whether data was lost.",
        "context": "Complaint: 'The update deleted my saved files.' "
        "Front-line support confirmed the update itself is defective.",
        "schema": {
            "type": "object",
            "required": ["team", "defect_confirmed", "data_lost"],
            "properties": {
                "team": {"type": "string", "enum": ["sales", "support", "technical"]},
                "defect_confirmed": {"type": "boolean"},
                "data_lost": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
        "answer": {"team": "technical", "defect_confirmed": True, "data_lost": True},
        "answer_field": "team",
        "paired": True,
        "source": "fieldorder",
    },
    {
        "id": "classify-document",
        "shape": "classification",
        "question": "Classify the document's type, language and confidentiality.",
        "context": "Document: a German-language internal memo marked confidential.",
        "schema": {
            "type": "object",
            "required": ["doc_type", "language", "confidential"],
            "properties": {
                "doc_type": {"type": "string", "enum": ["memo", "report", "invoice"]},
                "language": {"type": "string", "enum": ["english", "german", "french"]},
                "confidential": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
        "answer": {"doc_type": "memo", "language": "german", "confidential": True},
        "answer_field": "doc_type",
        "paired": True,
        "source": "fieldorder",
    },
    {
        "id": "extract-vehicle",
        "shape": "extraction",
        "question": "Extract the vehicle make, year, fuel and seat count.",
        "context": "The 2019 Volvo wagon runs on diesel and seats five.",
        "schema": {
            "type": "object",
            "required": ["make", "year", "fuel", "seats"],
            "properties": {
                "make": {"type": "string"},
                "year": {"type": "integer", "minimum": 1900, "maximum": 2100},
                "fuel": {"type": "string", "enum": ["petrol", "diesel", "electric"]},
                "seats": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": False,
        },
        "answer": {"make": "Volvo", "year": 2019, "fuel": "diesel", "seats": 5},
        "answer_field": "make",
        "paired": True,
        "source": "fieldorder",
    },
    {
        "id": "route-delivery",
        "shape": "routing",
        "question": "Route the delivery and flag its requirements.",
        "context": "Delivery: vaccines to a clinic in the south, sent by "
        "express courier for tomorrow morning; a signature is required "
        "and the box must stay refrigerated.",
        "schema": {
            "type": "object",
            "required": ["region", "speed", "signature_required", "refrigerated"],
            "properties": {
                "region": {"type": "string", "enum": ["north", "south", "east", "west"]},
                "speed": {"type": "string", "enum": ["standard", "express"]},
                "signature_required": {"type": "boolean"},
                "refrigerated": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
        "answer": {
            "region": "south",
            "speed": "express",
            "signature_required": True,
            "refrigerated": True,
        },
        "answer_field": "region",
        "paired": True,
        "source": "fieldorder",
    },
)


def digest_text(text: str) -> str:
    """SHA-256 hex digest of a text (the lab's digest primitive)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_order(spec: dict[str, Any]) -> list[str]:
    """The canonical (alphabetical) field order of a spec's schema."""
    return sorted(spec["schema"]["properties"])


def arm_order(spec: dict[str, Any], arm: str) -> list[str]:
    """The field enumeration order of one arm of a spec."""
    order = canonical_order(spec)
    if arm == "canonical":
        return order
    if arm == "reversed":
        return order[::-1]
    field = spec["answer_field"]
    rest = [name for name in order if name != field]
    if arm == "answer_first":
        return [field, *rest]
    if arm == "answer_last":
        return [*rest, field]
    raise ValueError(f"unknown field-order arm: {arm!r}")


def describe_ordered(schema: dict[str, Any], order: list[str]) -> str:
    """``structured.describe_schema`` with an explicit field order.

    Byte-identical to ``structured.describe_schema(schema)`` when
    ``order`` is the canonical (alphabetical) order.
    """
    properties = schema["properties"]
    parts: list[str] = []
    for name in order:
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
    required_names = set(schema["required"])
    required = [name for name in order if name in required_names]
    if schema["additionalProperties"] is False:
        extras = "no other keys allowed"
    else:
        extras = "other keys allowed"
    return "{" + "; ".join(parts) + "} (required: " + ", ".join(required) + "; " + extras + ")"


def arm_prompt(spec: dict[str, Any], arm: str) -> str:
    """The rendered user content for one (spec, arm)."""
    return (
        f"{spec['question']}\n"
        f"Context: {spec['context']}\n"
        "Respond with ONLY a JSON object matching this schema: "
        f"{describe_ordered(spec['schema'], arm_order(spec, arm))}.\n"
        "Do not include any other text.\n"
        f"(ref: {spec['source']}/{spec['id']})"
    )


def spec_by_id(spec_id: str) -> dict[str, Any]:
    """Look up a frozen spec by id; raises ValueError if unknown."""
    for spec in FIELD_ORDER_SPECS:
        if spec["id"] == spec_id:
            return spec
    raise ValueError(f"unknown field-order spec: {spec_id!r}")


def expected_value_digest(spec: dict[str, Any], field: str) -> str:
    """Digest of the canonical form of a spec's expected field value."""
    return digest_text(_canonical_json(spec["answer"][field]))


def _safe_canonical(value: object) -> str | None:
    """Canonical JSON of a value, or None when it has none.

    Total: a non-finite float, an integer beyond the int/str
    digit limit, or a structure too deep to serialise all yield
    None rather than raising.
    """
    try:
        return _canonical_json(value)
    except (ValueError, TypeError, RecursionError):
        return None


def parse_document(text: object) -> dict[str, Any]:
    """Parse a decoded answer into the record's document fields.

    Returns {"kind", "field_values", "duplicate_keys",
    "document_digest"}. Total: hostile text yields
    ``unparseable``, never an exception.
    """
    if type(text) is not str:
        return {
            "kind": "unparseable",
            "field_values": {},
            "duplicate_keys": False,
            "document_digest": None,
        }
    duplicates = False

    def _hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        """Return the hook."""
        nonlocal duplicates
        seen: dict[str, Any] = {}
        for key, value in pairs:
            if key in seen:
                duplicates = True
            seen[key] = value
        return seen

    def _parse_int(raw: str) -> Any:
        # An integer literal beyond the int<->str digit limit has
        # no canonical form: keep the document parseable and let
        # the sentinel canonicalise to None (design section 4).
        """Parse int."""
        try:
            return int(raw)
        except ValueError:
            return _OVERSIZE_INT

    try:
        document = json.loads(text, object_pairs_hook=_hook, parse_int=_parse_int)
    except (ValueError, TypeError, RecursionError):
        return {
            "kind": "unparseable",
            "field_values": {},
            "duplicate_keys": False,
            "document_digest": None,
        }
    if type(document) is not dict:
        return {
            "kind": "non_object",
            "field_values": {},
            "duplicate_keys": duplicates,
            "document_digest": None,
        }
    field_values: dict[str, str | None] = {}
    for key, value in document.items():
        field_values[key] = _safe_canonical(value)
    document_digest = None
    if all(value is not None for value in field_values.values()):
        document_digest = digest_text(_canonical_json(document))
    return {
        "kind": "object",
        "field_values": field_values,
        "duplicate_keys": duplicates,
        "document_digest": document_digest,
    }


def field_correct(spec: dict[str, Any], field: str, field_values: dict[str, Any]) -> bool:
    """Whether a produced field value equals the expected value.

    Derived from the commitment: the key must be present with a
    canonical string whose digest equals the expected value's
    canonical digest. A None (no canonical form) is incorrect
    for every field.
    """
    value = field_values.get(field)
    if type(value) is not str:
        return False
    return digest_text(value) == expected_value_digest(spec, field)


def schema_ok(spec: dict[str, Any], kind: str, field_values: dict[str, Any]) -> bool:
    """The recomputed schema verdict for one parsed document.

    The verdict is the flat schema check over the reconstructed
    document and nothing else (the frozen CHECKING rule):
    duplicate keys are an adherence condition, not a schema
    input. A None value fails its key's check before
    reconstruction, so the verdict is False whenever any value
    has no canonical form.
    """
    from .structured import check_schema

    if kind != "object":
        return False
    if any(value is None for value in field_values.values()):
        return False
    document: dict[str, Any] = {}
    for key, value in field_values.items():
        try:
            document[key] = json.loads(value)
        except (ValueError, TypeError, RecursionError):
            return False
    return check_schema(document, spec["schema"]) == "schema_ok"


def record_derived(spec: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    """All scored quantities of one record, derived from its fields."""
    kind = record["document_kind"]
    field_values = record["field_values"]
    parse_ok = kind in ("object", "non_object")
    ok = schema_ok(spec, kind, field_values)
    return {
        "parse_ok": parse_ok,
        "schema_ok": ok,
        "adherent": parse_ok and ok and not record["duplicate_keys"],
        "fields": {
            name: field_correct(spec, name, field_values) for name in spec["schema"]["properties"]
        },
        "answer_correct": field_correct(spec, spec["answer_field"], field_values),
        "whole_doc_correct": ok
        and all(field_correct(spec, name, field_values) for name in spec["schema"]["properties"])
        and not any(key not in spec["schema"]["properties"] for key in field_values),
    }


def sign_test_p(n_pos: int, n_neg: int) -> float:
    """Exact two-sided sign-test p-value for discordant counts."""
    n = n_pos + n_neg
    if n == 0:
        return 1.0
    k = min(n_pos, n_neg)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return float(min(1.0, 2.0 * tail))


def _rate(correct: int, n: int) -> float | None:
    """Return the rate."""
    return correct / n if n else None


def _decision(n_first_only: int, n_last_only: int) -> dict[str, Any]:
    """The endpoint decision over discordant paired specs."""
    discordant = n_first_only + n_last_only
    p_value = sign_test_p(n_first_only, n_last_only)
    if discordant < MIN_DISCORDANT_SPECS:
        verdict = "insufficient"
    elif p_value <= SIGN_ALPHA:
        verdict = "falsified"
    else:
        verdict = "not_falsified"
    if n_first_only > n_last_only:
        direction = "first"
    elif n_last_only > n_first_only:
        direction = "last"
    else:
        direction = "none"
    return {
        "verdict": verdict,
        "n_discordant": discordant,
        "n_first_only": n_first_only,
        "n_last_only": n_last_only,
        "sign_p": p_value,
        "direction": direction,
    }


# skipcq: PY-R1000 - branches are frozen payload-contract cases pinned by tests
def aggregate_candidate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one candidate's 60 records into the readout block.

    Populations are disjoint by construction order: boundary
    screened, then detail screened, then no-boundary, then live.
    Every rate is None iff its denominator is 0, and every
    denominator is printed beside its rate.
    """
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        by_key[(record["spec_id"], record["arm"])] = record
    n_generated = len(records)
    # Distinct prompts actually rendered: an arm's prompt is a
    # function of its field order, so orders that coincide are
    # the same prompt generated twice (design F3: 37 of 60).
    n_distinct_prompts = sum(
        len({tuple(arm_order(spec, arm)) for arm in ARM_KEYS}) for spec in FIELD_ORDER_SPECS
    )
    n_boundary_screened = sum(1 for r in records if r["case_screened"])
    n_detail_screened = sum(1 for r in records if not r["case_screened"] and r["detail"] != "")
    n_no_boundary = sum(
        1
        for r in records
        if not r["case_screened"] and r["detail"] == "" and not r["boundary_found"]
    )
    live = [
        r for r in records if not r["case_screened"] and r["detail"] == "" and r["boundary_found"]
    ]
    specs = {spec["id"]: spec for spec in FIELD_ORDER_SPECS}
    derived = {(r["spec_id"], r["arm"]): record_derived(specs[r["spec_id"]], r) for r in live}

    by_arm: dict[str, Any] = {}
    for arm in ARM_KEYS:
        rows = [r for r in live if r["arm"] == arm]
        n = len(rows)
        parse_ok = sum(1 for r in rows if derived[(r["spec_id"], arm)]["parse_ok"])
        schema_ok_n = sum(1 for r in rows if derived[(r["spec_id"], arm)]["schema_ok"])
        adherent = sum(1 for r in rows if derived[(r["spec_id"], arm)]["adherent"])
        by_arm[arm] = {
            "n": n,
            "parse_ok": parse_ok,
            "schema_ok": schema_ok_n,
            "adherent": adherent,
            "parse_rate": _rate(parse_ok, n),
            "schema_rate": _rate(schema_ok_n, n),
            "adherent_rate": _rate(adherent, n),
        }

    per_field: dict[str, Any] = {}
    for spec in FIELD_ORDER_SPECS:
        fields: dict[str, Any] = {}
        for name in spec["schema"]["properties"]:
            per_arm: dict[str, Any] = {}
            for arm in ARM_KEYS:
                rows = [r for r in live if r["spec_id"] == spec["id"] and r["arm"] == arm]
                n = len(rows)
                correct = sum(1 for r in rows if derived[(spec["id"], arm)]["fields"][name])
                per_arm[arm] = {"correct": correct, "n": n, "rate": _rate(correct, n)}
            fields[name] = per_arm
        per_field[spec["id"]] = fields

    answer_by_arm: dict[str, Any] = {}
    whole_doc_by_arm: dict[str, Any] = {}
    for arm in ARM_KEYS:
        rows = [r for r in live if r["arm"] == arm]
        n = len(rows)
        correct = sum(1 for r in rows if derived[(r["spec_id"], arm)]["answer_correct"])
        whole = sum(1 for r in rows if derived[(r["spec_id"], arm)]["whole_doc_correct"])
        answer_by_arm[arm] = {"correct": correct, "n": n, "rate": _rate(correct, n)}
        whole_doc_by_arm[arm] = {"correct": whole, "n": n, "rate": _rate(whole, n)}

    n_duplicates = sum(1 for r in live if r["duplicate_keys"])

    # Pairing (derived from the frozen flags + the derived
    # adherence): attrition counted in a stated order -- order
    # rule, then duplicate keys, then adherence.
    paired_first: dict[str, bool] = {}
    paired_last: dict[str, bool] = {}
    n_flagged = 0
    n_order_same = 0
    n_dup_excluded = 0
    n_not_adherent = 0
    ungated_first_only = 0
    ungated_last_only = 0
    ungated_specs = 0
    for spec in FIELD_ORDER_SPECS:
        if not spec["paired"]:
            continue
        n_flagged += 1
        if arm_order(spec, "answer_first") == arm_order(spec, "answer_last"):
            n_order_same += 1
            continue
        first = by_key.get((spec["id"], "answer_first"))
        last = by_key.get((spec["id"], "answer_last"))
        if first is None or last is None:
            continue
        first_live = (spec["id"], "answer_first") in derived
        last_live = (spec["id"], "answer_last") in derived
        if first_live and last_live:
            ungated_specs += 1
            f_ok = derived[(spec["id"], "answer_first")]["answer_correct"]
            l_ok = derived[(spec["id"], "answer_last")]["answer_correct"]
            if f_ok and not l_ok:
                ungated_first_only += 1
            elif l_ok and not f_ok:
                ungated_last_only += 1
        if first["duplicate_keys"] or last["duplicate_keys"]:
            n_dup_excluded += 1
            continue
        if not (
            first_live
            and last_live
            and derived[(spec["id"], "answer_first")]["adherent"]
            and derived[(spec["id"], "answer_last")]["adherent"]
        ):
            n_not_adherent += 1
            continue
        paired_first[spec["id"]] = derived[(spec["id"], "answer_first")]["answer_correct"]
        paired_last[spec["id"]] = derived[(spec["id"], "answer_last")]["answer_correct"]

    n_paired = len(paired_first)
    n_first_only = sum(1 for s in paired_first if paired_first[s] and not paired_last[s])
    n_last_only = sum(1 for s in paired_first if paired_last[s] and not paired_first[s])
    first_correct = sum(1 for s in paired_first if paired_first[s])
    last_correct = sum(1 for s in paired_first if paired_last[s])
    decision = _decision(n_first_only, n_last_only)
    decision["n_paired"] = n_paired

    extraction_ids = [
        s["id"] for s in FIELD_ORDER_SPECS if s["paired"] and s["shape"] == "extraction"
    ]
    ex_first_only = sum(
        1 for s in extraction_ids if s in paired_first and paired_first[s] and not paired_last[s]
    )
    ex_last_only = sum(
        1 for s in extraction_ids if s in paired_first and paired_last[s] and not paired_first[s]
    )
    extraction_decision = _decision(ex_first_only, ex_last_only)
    extraction_decision["n_paired"] = sum(1 for s in extraction_ids if s in paired_first)

    def _field_count_split(min_fields: int, max_fields: int | None) -> dict[str, Any]:
        """Return the field count split."""
        ids = [
            s["id"]
            for s in FIELD_ORDER_SPECS
            if s["id"] in paired_first
            and len(s["schema"]["properties"]) >= min_fields
            and (max_fields is None or len(s["schema"]["properties"]) <= max_fields)
        ]
        return {
            "n_paired": len(ids),
            "n_first_only": sum(1 for s in ids if paired_first[s] and not paired_last[s]),
            "n_last_only": sum(1 for s in ids if paired_last[s] and not paired_first[s]),
        }

    return {
        "n_generated": n_generated,
        "n_distinct_prompts": n_distinct_prompts,
        "n_boundary_screened": n_boundary_screened,
        "n_detail_screened": n_detail_screened,
        "n_no_boundary": n_no_boundary,
        "n_live": len(live),
        "n_duplicate_keys": n_duplicates,
        "by_arm": by_arm,
        "per_field": per_field,
        "answer_by_arm": answer_by_arm,
        "whole_doc_by_arm": whole_doc_by_arm,
        "pairing": {
            "n_flagged": n_flagged,
            "n_order_same": n_order_same,
            "n_duplicate_excluded": n_dup_excluded,
            "n_not_adherent": n_not_adherent,
            "n_paired": n_paired,
            "first_correct": first_correct,
            "last_correct": last_correct,
            "first_rate": _rate(first_correct, n_paired),
            "last_rate": _rate(last_correct, n_paired),
            "delta": (first_correct / n_paired - last_correct / n_paired if n_paired else None),
            "ungated": {
                "n_specs": ungated_specs,
                "n_first_only": ungated_first_only,
                "n_last_only": ungated_last_only,
                "n_discordant": ungated_first_only + ungated_last_only,
            },
        },
        "decision": decision,
        "extraction_decision": extraction_decision,
        "field_count_split": {
            "two": _field_count_split(2, 2),
            "three_plus": _field_count_split(3, None),
        },
    }


def protocol_object(seed: int, backend_fingerprint: str) -> dict[str, Any]:
    """The field-order protocol object (fingerprint added by the stage)."""
    if type(seed) is not int or isinstance(seed, bool):
        raise ValueError("seed must be an int")
    return {
        "seed": seed,
        "backend_fingerprint": backend_fingerprint,
        "spec_count": SPEC_COUNT,
        "arm_keys": list(ARM_KEYS),
        "inventory_digest": digest_text(_canonical_json(list(FIELD_ORDER_SPECS))),
        "generation_rule_digest": digest_text(GENERATION_RULE_TEXT),
        "rendering_rule_digest": digest_text(RENDERING_RULE_TEXT),
        "checking_rule_digest": digest_text(CHECKING_RULE_TEXT),
        "max_records": MAX_FIELD_ORDER_RECORDS,
        "max_payload_bytes": MAX_FIELD_ORDER_PAYLOAD_BYTES,
        "max_new_tokens": MAX_NEW_TOKENS,
        "field_values_cap": FIELD_VALUES_CAP,
        "min_discordant_specs": MIN_DISCORDANT_SPECS,
    }


def protocol_fingerprint(seed: int, backend_fingerprint: str) -> str:
    """Digest binding the whole protocol object."""
    return digest_text(_canonical_json(protocol_object(seed, backend_fingerprint)))
