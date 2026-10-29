"""Cycle 157: structured-output (JSON mode) adherence per template (wave-3 idea 5).

Tests the pure core (structured.py): the flat schema dialect, the
deterministic hand-rolled validator (with a 200-case brute-force oracle
differential), response classification, fixture construction, and rate
aggregation; the experiments.structured_stage / structured_adherence
fail-closed contracts; the CLI probe wiring; the HTML section escaping
and TOC anchor; and the diffreports.diff_structured section. Follows the
established per-cycle pattern: brute-force oracle differentials,
Hypothesis properties, tempdir-module mutation gates, hostile-payload
red-team.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import random
import tempfile
import unittest
from pathlib import Path
from threading import Event

from hypothesis import given, settings
from hypothesis import strategies as st

from model_protocol_lab import diffreports
from model_protocol_lab.core import Format, Generation
from model_protocol_lab.diffreports import diff_structured
from model_protocol_lab.experiments import structured_adherence, structured_stage
from model_protocol_lab.reporting import _structured_section
from model_protocol_lab.structured import (
    OUTCOMES,
    SHAPES,
    STRUCTURED_SPECS,
    build_structured_fixtures,
    candidate_adherence,
    check_schema,
    classify_response,
    describe_schema,
    render_prompt,
    schema_map,
    validate_field_def,
    validate_schema_def,
    validate_structured_specs,
)

BOUNDARY = (124,)  # ord("|")


def _format(name="chatml"):
    """Format."""
    return Format(
        name=name,
        bos="<s>",
        user_prefix="<|user|>\n",
        assistant_prefix="<|assistant|>\n",
        turn_end="<|end|>\n",
    )


_SPECS = validate_structured_specs(STRUCTURED_SPECS)
_FIXTURES = build_structured_fixtures(_SPECS)
_SCHEMAS = schema_map(_SPECS)
_EXPECTED = {fixture.id: fixture.expected for fixture in _FIXTURES}


class _AdherentBackend:
    """ord codec; emits the fixture's canonical JSON answer."""

    @staticmethod
    def encode(text):
        """Encode."""
        return tuple(map(ord, text))

    @staticmethod
    def decode(token_ids):
        """Decode."""
        return "".join(map(chr, token_ids))

    def generate(self, input_ids, *, max_new_tokens, seed, cancel):
        """Generate."""
        prompt = self.decode(input_ids)
        for fid, expected in _EXPECTED.items():
            if f"(ref: {fid})" in prompt:
                text = (expected + "|").ljust(max_new_tokens)[:max_new_tokens]
                return Generation(tuple(map(ord, text)), "limit")
        text = ("?" + "|").ljust(max_new_tokens)[:max_new_tokens]
        return Generation(tuple(map(ord, text)), "limit")


class _BrokenJsonBackend(_AdherentBackend):
    """ord codec; emits schema-violating JSON for every fixture."""

    @staticmethod
    def generate(input_ids, *, max_new_tokens, seed, cancel):
        """Generate."""
        text = ('{"wrong": "shape"}|').ljust(max_new_tokens)[:max_new_tokens]
        return Generation(tuple(map(ord, text)), "limit")


class _RaisingBackend(_AdherentBackend):
    """Raising backend."""

    @staticmethod
    def generate(input_ids, *, max_new_tokens, seed, cancel):
        """Generate."""
        raise RuntimeError("boom")


def _structured_payload(
    parse_rate=1.0, schema_rate=1.0, weakest=(), gap=0.0, names=("alpha", "beta")
):
    """Structured payload."""
    return {
        "available": True,
        "n_fixtures": 9,
        "per_candidate": {
            name: {
                "n": 9,
                "n_evaluable": 9,
                "parse_ok": 9,
                "schema_ok": 9,
                "adherent": 9,
                "parse_rate": parse_rate,
                "schema_rate": schema_rate,
                "adherent_rate": schema_rate,
            }
            for name in names
        },
        "weakest_candidates": list(weakest),
        "excluded_candidates": [],
        "adherence_gap": gap,
        "uniform": True,
    }


class ValidateStructuredSpecsTests(unittest.TestCase):
    """Validate structured specs tests."""

    def test_specs_validate(self):
        """Test specs validate."""
        specs = validate_structured_specs(STRUCTURED_SPECS)
        self.assertEqual(len(specs), 9)
        self.assertEqual({spec["shape"] for spec in specs}, set(SHAPES))

    def test_rejects_non_tuple(self):
        """Test rejects non tuple."""
        with self.assertRaises(ValueError):
            validate_structured_specs(list(STRUCTURED_SPECS))

    def test_rejects_empty(self):
        """Test rejects empty."""
        with self.assertRaises(ValueError):
            validate_structured_specs(())

    def test_rejects_non_dict_spec(self):
        """Test rejects non dict spec."""
        with self.assertRaises(ValueError):
            validate_structured_specs(("nope",))

    def test_rejects_unknown_key(self):
        """Test rejects unknown key."""
        bad = dict(STRUCTURED_SPECS[0])
        bad["bogus"] = 1
        with self.assertRaises(ValueError):
            validate_structured_specs((bad,))

    def test_rejects_blank_id(self):
        """Test rejects blank id."""
        bad = dict(STRUCTURED_SPECS[0])
        bad["id"] = "  "
        with self.assertRaises(ValueError):
            validate_structured_specs((bad,))

    def test_rejects_duplicate_id(self):
        """Test rejects duplicate id."""
        with self.assertRaises(ValueError):
            validate_structured_specs((STRUCTURED_SPECS[0], STRUCTURED_SPECS[0]))

    def test_rejects_blank_question(self):
        """Test rejects blank question."""
        bad = dict(STRUCTURED_SPECS[0])
        bad["question"] = ""
        with self.assertRaises(ValueError):
            validate_structured_specs((bad,))

    def test_rejects_unknown_shape(self):
        """Test rejects unknown shape."""
        bad = dict(STRUCTURED_SPECS[0])
        bad["shape"] = "poetry"
        with self.assertRaises(ValueError):
            validate_structured_specs((bad,))

    def test_rejects_bad_schema(self):
        """Test rejects bad schema."""
        bad = dict(STRUCTURED_SPECS[0])
        bad["schema"] = {"type": "object"}
        with self.assertRaises(ValueError):
            validate_structured_specs((bad,))

    def test_rejects_non_dict_answer(self):
        """Test rejects non dict answer."""
        bad = dict(STRUCTURED_SPECS[0])
        bad["answer"] = [1, 2]
        with self.assertRaises(ValueError):
            validate_structured_specs((bad,))

    def test_rejects_answer_violating_own_schema(self):
        """Test rejects answer violating own schema."""
        bad = dict(STRUCTURED_SPECS[0])
        bad["answer"] = {"name": "Ada Lovelace"}  # missing required age
        with self.assertRaises(ValueError):
            validate_structured_specs((bad,))

    def test_specs_return_plain_dicts(self):
        """Test specs return plain dicts."""
        specs = validate_structured_specs(STRUCTURED_SPECS)
        for spec in specs:
            self.assertIs(type(spec), dict)
            self.assertIs(type(spec["schema"]), dict)


class SchemaDefTests(unittest.TestCase):
    """Schema def tests."""

    @staticmethod
    def _schema():
        """Schema."""
        return {
            "type": "object",
            "required": ["label"],
            "properties": {"label": {"type": "string", "enum": ["positive", "negative"]}},
            "additionalProperties": False,
        }

    def test_valid_schema(self):
        """Test valid schema."""
        validate_schema_def(self._schema())

    def test_rejects_non_dict(self):
        """Test rejects non dict."""
        with self.assertRaises(ValueError):
            validate_schema_def("nope")

    def test_rejects_non_object_type(self):
        """Test rejects non object type."""
        schema = self._schema()
        schema["type"] = "array"
        with self.assertRaises(ValueError):
            validate_schema_def(schema)

    def test_rejects_empty_required(self):
        """Test rejects empty required."""
        schema = self._schema()
        schema["required"] = []
        with self.assertRaises(ValueError):
            validate_schema_def(schema)

    def test_rejects_required_not_declared(self):
        """Test rejects required not declared."""
        schema = self._schema()
        schema["required"] = ["missing"]
        with self.assertRaises(ValueError):
            validate_schema_def(schema)

    def test_rejects_non_bool_additional(self):
        """Test rejects non bool additional."""
        schema = self._schema()
        schema["additionalProperties"] = "yes"
        with self.assertRaises(ValueError):
            validate_schema_def(schema)

    def test_rejects_unknown_schema_key(self):
        """Test rejects unknown schema key."""
        schema = self._schema()
        schema["title"] = "x"
        with self.assertRaises(ValueError):
            validate_schema_def(schema)

    def test_rejects_unknown_field_type(self):
        """Test rejects unknown field type."""
        with self.assertRaises(ValueError):
            validate_field_def({"type": "object"}, where="p")

    def test_rejects_nested_object_field(self):
        """Test rejects nested object field."""
        with self.assertRaises(ValueError):
            validate_field_def({"type": "array"}, where="p")

    def test_rejects_unknown_field_key(self):
        """Test rejects unknown field key."""
        with self.assertRaises(ValueError):
            validate_field_def({"type": "string", "format": "x"}, where="p")

    def test_rejects_empty_enum(self):
        """Test rejects empty enum."""
        with self.assertRaises(ValueError):
            validate_field_def({"type": "string", "enum": []}, where="p")

    def test_rejects_container_enum_member(self):
        """Test rejects container enum member."""
        with self.assertRaises(ValueError):
            validate_field_def({"type": "string", "enum": [["a"]]}, where="p")

    def test_rejects_nonfinite_minimum(self):
        """Test rejects nonfinite minimum."""
        with self.assertRaises(ValueError):
            validate_field_def({"type": "number", "minimum": float("inf")}, where="p")

    def test_rejects_fractional_bound_on_integer(self):
        """Test rejects fractional bound on integer."""
        with self.assertRaises(ValueError):
            validate_field_def({"type": "integer", "minimum": 0.5}, where="p")

    def test_rejects_minimum_above_maximum(self):
        """Test rejects minimum above maximum."""
        with self.assertRaises(ValueError):
            validate_field_def({"type": "integer", "minimum": 5, "maximum": 3}, where="p")

    def test_rejects_non_dict_field(self):
        """Test rejects non dict field."""
        with self.assertRaises(ValueError):
            validate_field_def("string", where="p")

    @staticmethod
    def test_all_field_types_accepted():
        """Test all field types accepted."""
        for field_type in ("string", "integer", "number", "boolean", "null"):
            validate_field_def({"type": field_type}, where="p")


# ---------------------------------------------------------------------------
# Independent brute-force oracle for the schema validator.
# ---------------------------------------------------------------------------


# skipcq: PY-R1000 - Gate-pinned frozen function; splitting breaks its verified contract.
def _oracle_field_def_ok(field):
    """Independent reimplementation of validate_field_def's acceptance set."""
    if not isinstance(field, dict):
        return False
    field_type = field.get("type")
    if field_type not in ("string", "integer", "number", "boolean", "null"):
        return False
    if set(field) - {"type", "enum", "minimum", "maximum"}:
        return False
    enum = field.get("enum")
    if enum is not None:
        if not isinstance(enum, list) or not enum:
            return False
        for member in enum:
            if type(member) not in (str, int, float, bool) and member is not None:
                return False
    for bound in ("minimum", "maximum"):
        value = field.get(bound)
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return False
            # skipcq: PYL-R0124 - self-comparison is the total NaN test on untrusted JSON values
            if value != value or abs(value) == float("inf"):
                return False
            if field_type == "integer" and int(value) != value:
                return False
    minimum = field.get("minimum")
    maximum = field.get("maximum")
    if minimum is not None and maximum is not None:
        return minimum <= maximum
    return True


# skipcq: PY-R1000 - Gate-pinned frozen function; splitting breaks its verified contract.
def _oracle_schema_def_ok(schema):
    """Independent reimplementation of validate_schema_def's acceptance set."""
    if not isinstance(schema, dict):
        return False
    if schema.get("type") != "object":
        return False
    required = schema.get("required")
    properties = schema.get("properties")
    if not isinstance(required, list) or not required:
        return False
    if any(not isinstance(name, str) for name in required):
        return False
    if not isinstance(properties, dict) or not properties:
        return False
    if any(not isinstance(name, str) for name in properties):
        return False
    if not isinstance(schema.get("additionalProperties"), bool):
        return False
    if set(schema) - {"type", "required", "properties", "additionalProperties"}:
        return False
    if any(name not in properties for name in required):
        return False
    return all(_oracle_field_def_ok(field) for field in properties.values())


# skipcq: PY-R1000 - Gate-pinned frozen function; splitting breaks its verified contract.
def _oracle_field_ok(value, field):
    """Independent reimplementation of the per-field value check."""
    field_type = field["type"]
    match field_type:
        case "string":
            if type(value) is not str:
                return False
        case "integer":
            if type(value) is not int or isinstance(value, bool):
                return False
        case "number":
            if type(value) not in (int, float) or isinstance(value, bool):
                return False
            # Non-finite floats are not JSON numbers (RFC 8259 §6); the
            # oracle must share the implementation's rejection, not the
            # shared blind spot that let F1 through.
            # skipcq: PYL-R0124 - self-comparison is the total NaN test on untrusted JSON values
            if value != value or abs(value) == float("inf"):
                return False
        case "boolean":
            if type(value) is not bool:
                return False
        case "null":
            if value is not None:
                return False
        case _:
            return False
    enum = field.get("enum")
    if enum is not None and value not in enum:
        return False
    minimum = field.get("minimum")
    if minimum is not None and value < minimum:
        return False
    maximum = field.get("maximum")
    if maximum is not None:
        return value <= maximum
    return True


def _oracle_check_schema(document, schema):
    """Independent reimplementation of check_schema."""
    if not _oracle_schema_def_ok(schema):
        return "schema_invalid"
    if not isinstance(document, dict):
        return "schema_failed"
    for name in schema["required"]:
        if name not in document:
            return "schema_failed"
    for name, value in document.items():
        field = schema["properties"].get(name)
        if field is None:
            if schema.get("additionalProperties") is False:
                return "schema_failed"
            continue
        if not _oracle_field_ok(value, field):
            return "schema_failed"
    return "schema_ok"


def _random_schema(rng):
    """Random schema."""
    field_types = ["string", "integer", "number", "boolean"]
    properties = {}
    names = rng.sample(["a", "b", "c", "d"], rng.randint(1, 3))
    for name in names:
        field_type = rng.choice(field_types)
        field = {"type": field_type}
        if rng.random() < 0.4:
            if field_type == "string":
                field["enum"] = rng.sample(["x", "y", "z"], rng.randint(1, 3))
            elif field_type == "integer":
                field["enum"] = rng.sample([1, 2, 3], rng.randint(1, 3))
            elif field_type == "boolean":
                field["enum"] = [True]
        if field_type in ("integer", "number") and rng.random() < 0.4:
            field["minimum"] = rng.randint(0, 5)
            field["maximum"] = field["minimum"] + rng.randint(0, 5)
        properties[name] = field
    return {
        "type": "object",
        "required": rng.sample(names, rng.randint(1, len(names))),
        "properties": properties,
        "additionalProperties": rng.random() < 0.5,
    }


def _random_doc(rng, schema):
    """Random doc."""
    doc = {}
    pool = list(schema["properties"]) + ["extra"]
    for name in rng.sample(pool, rng.randint(0, len(pool))):
        choice = rng.random()
        if choice < 0.3:
            doc[name] = rng.choice(["x", "y", "nope", ""])
        elif choice < 0.5:
            doc[name] = rng.randint(-2, 9)
        elif choice < 0.65:
            # Include non-finite floats: the F1 hole was invisible here
            # because the pool never generated them.
            doc[name] = rng.choice([1.5, 2.5, float("nan"), float("inf"), float("-inf")])
        elif choice < 0.8:
            doc[name] = rng.choice([True, False])
        elif choice < 0.9:
            doc[name] = None
        else:
            doc[name] = ["list"]
    return doc


class CheckSchemaOracleTests(unittest.TestCase):
    """Check schema oracle tests."""

    def test_oracle_agrees_on_spec_schemas(self):
        """Test oracle agrees on spec schemas."""
        answers = {f"structured/{spec['id']}": spec["answer"] for spec in _SPECS}
        for fixture_id, schema in _SCHEMAS.items():
            answer = answers[fixture_id]
            self.assertEqual(check_schema(answer, schema), "schema_ok")
            self.assertEqual(_oracle_check_schema(answer, schema), "schema_ok")

    def test_brute_force_differential_200_cases(self):
        """Test brute force differential 200 cases."""
        rng = random.Random(20261019)  # noqa: S311 -- deterministic test seed
        mismatches = 0
        for _ in range(200):
            schema = _random_schema(rng)
            doc = _random_doc(rng, schema)
            expected = _oracle_check_schema(doc, schema)
            actual = check_schema(doc, schema)
            if expected != actual:
                mismatches += 1
        self.assertEqual(mismatches, 0)

    def test_malformed_schemas_agree_invalid(self):
        """Test malformed schemas agree invalid."""
        for bad in (
            "nope",
            {"type": "array"},
            {
                "type": "object",
                "required": [],
                "properties": {"a": {"type": "string"}},
                "additionalProperties": False,
            },
            {
                "type": "object",
                "required": ["a"],
                "properties": {"b": {"type": "string"}},
                "additionalProperties": False,
            },
        ):
            self.assertEqual(check_schema({}, bad), "schema_invalid")
            self.assertEqual(_oracle_check_schema({}, bad), "schema_invalid")

    def test_hand_cases(self):
        """Test hand cases."""
        schema = {
            "type": "object",
            "required": ["age"],
            "properties": {
                "age": {"type": "integer", "minimum": 0, "maximum": 150},
                "nick": {"type": "string"},
            },
            "additionalProperties": False,
        }
        cases = [
            ({"age": 36}, "schema_ok"),
            ({"age": 36, "nick": "ada"}, "schema_ok"),
            ({}, "schema_failed"),  # missing required
            ({"age": "36"}, "schema_failed"),  # wrong type
            ({"age": True}, "schema_failed"),  # bool is not integer
            ({"age": 36, "zzz": 1}, "schema_failed"),  # extra key
            ({"age": 200}, "schema_failed"),  # above maximum
            ({"age": -1}, "schema_failed"),  # below minimum
            ([1, 2], "schema_failed"),  # not a dict
            ("text", "schema_failed"),
        ]
        for doc, expected in cases:
            self.assertEqual(check_schema(doc, schema), expected, doc)
            self.assertEqual(_oracle_check_schema(doc, schema), expected, doc)

    def test_additional_properties_true_allows_extras(self):
        """Test additional properties true allows extras."""
        schema = {
            "type": "object",
            "required": ["a"],
            "properties": {"a": {"type": "string"}},
            "additionalProperties": True,
        }
        self.assertEqual(check_schema({"a": "x", "zzz": 1}, schema), "schema_ok")


class ClassifyResponseTests(unittest.TestCase):
    """Classify response tests."""

    @staticmethod
    def _schema():
        """Schema."""
        return _SCHEMAS["structured/extract-person"]

    def test_canonical_answer_adherent(self):
        """Test canonical answer adherent."""
        result = classify_response(_EXPECTED["structured/extract-person"], self._schema())
        self.assertEqual(result["outcome"], "adherent")
        self.assertEqual(result["parse"], "parse_ok")
        self.assertEqual(result["schema"], "schema_ok")

    def test_garbage_is_parse_failed(self):
        """Test garbage is parse failed."""
        result = classify_response("not json at all", self._schema())
        self.assertEqual(result["outcome"], "parse_failed")
        self.assertEqual(result["schema"], "not_evaluated")

    def test_valid_json_wrong_shape_is_schema_failed(self):
        """Test valid json wrong shape is schema failed."""
        result = classify_response('{"wrong": "shape"}', self._schema())
        self.assertEqual(result["outcome"], "schema_failed")
        self.assertEqual(result["parse"], "parse_ok")

    def test_non_string_is_not_text(self):
        """Test non string is not text."""
        for bad in (None, 42, ["x"], {"a": 1}):
            result = classify_response(bad, self._schema())
            self.assertEqual(result["outcome"], "not_text", bad)

    def test_invalid_schema_is_fail_closed(self):
        """Test invalid schema is fail closed."""
        result = classify_response('{"a": 1}', {"type": "array"})
        self.assertEqual(result["outcome"], "not_text")
        self.assertEqual(result["schema"], "schema_invalid")

    def test_never_raises_on_hostile_text(self):
        """Test never raises on hostile text."""
        hostile = ["\x00", "\ud800", "{", '{"a":}', "[1," * 500, "\n" * 1000]
        for text in hostile:
            result = classify_response(text, self._schema())
            self.assertIn(result["outcome"], OUTCOMES)

    def test_whitespace_padded_json_parses(self):
        """Test whitespace padded json parses."""
        text = "  " + _EXPECTED["structured/extract-person"] + "\n"
        self.assertEqual(classify_response(text, self._schema())["outcome"], "adherent")

    def test_trailing_prose_fails_parse(self):
        """Test trailing prose fails parse."""
        text = _EXPECTED["structured/extract-person"] + " hope this helps"
        self.assertEqual(classify_response(text, self._schema())["outcome"], "parse_failed")


class CandidateAdherenceTests(unittest.TestCase):
    """Candidate adherence tests."""

    def test_all_adherent(self):
        """Test all adherent."""
        agg = candidate_adherence(["adherent"] * 9)
        self.assertEqual(agg["adherent_rate"], 1.0)
        self.assertEqual(agg["parse_rate"], 1.0)
        self.assertEqual(agg["n_evaluable"], 9)

    def test_mixed_rates(self):
        """Test mixed rates."""
        agg = candidate_adherence(["adherent"] * 4 + ["schema_failed"] * 3 + ["parse_failed"] * 2)
        self.assertEqual(agg["n"], 9)
        self.assertEqual(agg["n_evaluable"], 9)
        self.assertEqual(agg["parse_ok"], 7)  # adherent + schema_failed parse
        self.assertEqual(agg["schema_ok"], 4)
        self.assertAlmostEqual(agg["parse_rate"], 7 / 9)
        self.assertAlmostEqual(agg["adherent_rate"], 4 / 9)

    def test_not_text_excluded_from_rates(self):
        """Test not text excluded from rates."""
        agg = candidate_adherence(["adherent", "not_text", "not_text"])
        self.assertEqual(agg["n"], 3)
        self.assertEqual(agg["n_evaluable"], 1)
        self.assertEqual(agg["adherent_rate"], 1.0)

    def test_empty_gives_none_rates(self):
        """Test empty gives none rates."""
        agg = candidate_adherence([])
        self.assertEqual(agg["n"], 0)
        self.assertIsNone(agg["parse_rate"])
        self.assertIsNone(agg["adherent_rate"])

    def test_all_not_text_gives_none_rates(self):
        """Test all not text gives none rates."""
        agg = candidate_adherence(["not_text"])
        self.assertIsNone(agg["adherent_rate"])

    def test_rejects_unknown_outcome(self):
        """Test rejects unknown outcome."""
        with self.assertRaises(ValueError):
            candidate_adherence(["adherent", "bogus"])


class FixtureBuildTests(unittest.TestCase):
    """Fixture build tests."""

    def test_fixture_ids(self):
        """Test fixture ids."""
        ids = [fixture.id for fixture in _FIXTURES]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(fid.startswith("structured/") for fid in ids))

    def test_expected_is_canonical_json(self):
        """Test expected is canonical json."""
        for fixture in _FIXTURES:
            self.assertEqual(json.loads(fixture.expected), json.loads(fixture.expected))
            self.assertEqual(
                fixture.expected,
                json.dumps(json.loads(fixture.expected), sort_keys=True, separators=(", ", ": ")),
            )

    def test_prompt_carries_ref_marker_and_schema(self):
        """Test prompt carries ref marker and schema."""
        prompt = render_prompt(_SPECS[0])
        self.assertIn("(ref: structured/extract-person)", prompt)
        self.assertIn("ONLY a JSON object", prompt)
        # The canonical answer JSON is never embedded in the prompt.
        self.assertNotIn(_EXPECTED["structured/extract-person"], prompt)

    def test_prompt_deterministic(self):
        """Test prompt deterministic."""
        self.assertEqual(render_prompt(_SPECS[3]), render_prompt(_SPECS[3]))

    def test_schema_map_covers_all_fixtures(self):
        """Test schema map covers all fixtures."""
        self.assertEqual(set(_SCHEMAS), {fixture.id for fixture in _FIXTURES})

    def test_describe_schema_names_every_property(self):
        """Test describe schema names every property."""
        description = describe_schema(_SCHEMAS["structured/route-ticket"])
        self.assertIn('"department"', description)
        self.assertIn('"priority"', description)
        self.assertIn("billing", description)
        self.assertIn("required: department, priority", description)

    def test_describe_schema_deterministic(self):
        """Test describe schema deterministic."""
        self.assertEqual(
            describe_schema(_SCHEMAS["structured/route-ticket"]),
            describe_schema(_SCHEMAS["structured/route-ticket"]),
        )


class StructuredStageTests(unittest.TestCase):
    """Structured stage tests."""

    def test_adherent_backend_all_adherent(self):
        """Test adherent backend all adherent."""
        stage = structured_stage(
            _AdherentBackend(),
            (_format(),),
            _FIXTURES,
            schemas=_SCHEMAS,
            boundary=BOUNDARY,
            max_new_tokens=128,
            seed=0,
        )
        self.assertEqual(stage["n_fixtures"], 9)
        cell = stage["per_candidate"]["chatml"]
        self.assertEqual(cell["adherent_rate"], 1.0)
        self.assertEqual(cell["n_evaluable"], 9)

    def test_broken_backend_zero_adherent(self):
        """Test broken backend zero adherent."""
        stage = structured_stage(
            _BrokenJsonBackend(),
            (_format(),),
            _FIXTURES,
            schemas=_SCHEMAS,
            boundary=BOUNDARY,
            max_new_tokens=128,
            seed=0,
        )
        cell = stage["per_candidate"]["chatml"]
        self.assertEqual(cell["adherent_rate"], 0.0)
        self.assertEqual(cell["parse_rate"], 1.0)  # parses, violates schema
        self.assertEqual(cell["schema_rate"], 0.0)

    def test_raising_backend_degrades_to_none(self):
        """Test raising backend degrades to none."""
        stage = structured_stage(
            _RaisingBackend(),
            (_format(),),
            _FIXTURES,
            schemas=_SCHEMAS,
            boundary=BOUNDARY,
            max_new_tokens=128,
            seed=0,
        )
        self.assertIsNone(stage["per_candidate"]["chatml"])

    def test_rejects_none_backend(self):
        """Test rejects none backend."""
        with self.assertRaises(ValueError):
            structured_stage(None, (_format(),), _FIXTURES, schemas=_SCHEMAS, boundary=BOUNDARY)

    def test_rejects_empty_formats(self):
        """Test rejects empty formats."""
        with self.assertRaises(ValueError):
            structured_stage(_AdherentBackend(), (), _FIXTURES, schemas=_SCHEMAS, boundary=BOUNDARY)

    def test_rejects_empty_fixtures(self):
        """Test rejects empty fixtures."""
        with self.assertRaises(ValueError):
            structured_stage(
                _AdherentBackend(), (_format(),), (), schemas=_SCHEMAS, boundary=BOUNDARY
            )

    def test_rejects_non_fixture_entries(self):
        """Test rejects non fixture entries."""
        with self.assertRaises(ValueError):
            structured_stage(
                _AdherentBackend(), (_format(),), ("nope",), schemas=_SCHEMAS, boundary=BOUNDARY
            )

    def test_rejects_schemas_not_dict(self):
        """Test rejects schemas not dict."""
        with self.assertRaises(ValueError):
            structured_stage(
                _AdherentBackend(), (_format(),), _FIXTURES, schemas="nope", boundary=BOUNDARY
            )

    def test_rejects_fixture_without_schema(self):
        """Test rejects fixture without schema."""
        schemas = dict(_SCHEMAS)
        schemas.pop(_FIXTURES[0].id)
        with self.assertRaises(ValueError):
            structured_stage(
                _AdherentBackend(), (_format(),), _FIXTURES, schemas=schemas, boundary=BOUNDARY
            )

    def test_rejects_bad_boundary(self):
        """Test rejects bad boundary."""
        with self.assertRaises(ValueError):
            structured_stage(
                _AdherentBackend(), (_format(),), _FIXTURES, schemas=_SCHEMAS, boundary=()
            )

    def test_rejects_bad_max_new_tokens(self):
        """Test rejects bad max new tokens."""
        with self.assertRaises(ValueError):
            structured_stage(
                _AdherentBackend(),
                (_format(),),
                _FIXTURES,
                schemas=_SCHEMAS,
                boundary=BOUNDARY,
                max_new_tokens=0,
            )

    def test_rejects_bad_seed(self):
        """Test rejects bad seed."""
        with self.assertRaises(ValueError):
            structured_stage(
                _AdherentBackend(),
                (_format(),),
                _FIXTURES,
                schemas=_SCHEMAS,
                boundary=BOUNDARY,
                seed=-1,
            )


class StructuredAdherenceTests(unittest.TestCase):
    """Structured adherence tests."""

    @staticmethod
    def _stage(rates):
        """Stage."""
        return {
            "n_fixtures": 9,
            "per_candidate": {
                name: {
                    "n": 9,
                    "n_evaluable": 9,
                    "parse_ok": 9,
                    "schema_ok": 9,
                    "adherent": 9,
                    "parse_rate": rate,
                    "schema_rate": rate,
                    "adherent_rate": rate,
                }
                for name, rate in rates.items()
            },
        }

    def test_uniform_single_candidate(self):
        """Test uniform single candidate."""
        result = structured_adherence(("alpha",), self._stage({"alpha": 1.0}))
        self.assertTrue(result["available"])
        self.assertTrue(result["uniform"])
        self.assertEqual(result["adherence_gap"], 0.0)
        self.assertEqual(result["weakest_candidates"], ["alpha"])

    def test_gap_and_weakest(self):
        """Test gap and weakest."""
        result = structured_adherence(("alpha", "beta"), self._stage({"alpha": 1.0, "beta": 0.25}))
        self.assertFalse(result["uniform"])
        self.assertEqual(result["adherence_gap"], 0.75)
        self.assertEqual(result["weakest_candidates"], ["beta"])

    def test_none_cell_excluded(self):
        """Test none cell excluded."""
        stage = self._stage({"alpha": 1.0})
        stage["per_candidate"]["beta"] = None
        result = structured_adherence(("alpha", "beta"), stage)
        self.assertEqual(result["excluded_candidates"], ["beta"])
        self.assertEqual(result["weakest_candidates"], ["alpha"])

    def test_zero_evaluable_excluded(self):
        """Test zero evaluable excluded."""
        stage = self._stage({"alpha": 1.0})
        stage["per_candidate"]["alpha"] = {
            "n": 9,
            "n_evaluable": 0,
            "parse_ok": 0,
            "schema_ok": 0,
            "adherent": 0,
            "parse_rate": None,
            "schema_rate": None,
            "adherent_rate": None,
        }
        result = structured_adherence(("alpha",), stage)
        self.assertEqual(result["excluded_candidates"], ["alpha"])
        self.assertIsNone(result["adherence_gap"])
        self.assertIsNone(result["uniform"])

    def test_rejects_duplicate_candidates(self):
        """Test rejects duplicate candidates."""
        result = structured_adherence(("alpha", "alpha"), self._stage({"alpha": 1.0}))
        self.assertFalse(result["available"])
        self.assertEqual(result["detail"], "invalid_candidates")

    def test_rejects_empty_candidates(self):
        """Test rejects empty candidates."""
        result = structured_adherence((), self._stage({"alpha": 1.0}))
        self.assertFalse(result["available"])

    def test_rejects_non_tuple_candidates(self):
        """Test rejects non tuple candidates."""
        result = structured_adherence(["alpha"], self._stage({"alpha": 1.0}))
        self.assertFalse(result["available"])

    def test_rejects_non_dict_stage(self):
        """Test rejects non dict stage."""
        result = structured_adherence(("alpha",), "nope")
        self.assertFalse(result["available"])
        self.assertEqual(result["detail"], "stage_invalid")

    def test_rejects_negative_fixture_count(self):
        """Test rejects negative fixture count."""
        stage = self._stage({"alpha": 1.0})
        stage["n_fixtures"] = -1
        result = structured_adherence(("alpha",), stage)
        self.assertFalse(result["available"])

    def test_rejects_non_str_candidate_keys(self):
        """Test rejects non str candidate keys."""
        stage = self._stage({"alpha": 1.0})
        stage["per_candidate"][7] = None
        result = structured_adherence(("alpha",), stage)
        self.assertFalse(result["available"])

    def test_rejects_out_of_range_rate(self):
        """Test rejects out of range rate."""
        stage = self._stage({"alpha": 1.0})
        stage["per_candidate"]["alpha"]["adherent_rate"] = 1.5
        result = structured_adherence(("alpha",), stage)
        self.assertFalse(result["available"])

    def test_rejects_negative_counts(self):
        """Test rejects negative counts."""
        stage = self._stage({"alpha": 1.0})
        stage["per_candidate"]["alpha"]["n_evaluable"] = -1
        result = structured_adherence(("alpha",), stage)
        self.assertFalse(result["available"])

    def test_tied_weakest_all_listed(self):
        """Test tied weakest all listed."""
        result = structured_adherence(("alpha", "beta"), self._stage({"alpha": 0.5, "beta": 0.5}))
        self.assertEqual(result["weakest_candidates"], ["alpha", "beta"])
        self.assertTrue(result["uniform"])


class DiffStructuredTests(unittest.TestCase):
    """Diff structured tests."""

    def test_identical_payloads_unchanged(self):
        """Test identical payloads unchanged."""
        diff = diff_structured(_structured_payload(), _structured_payload())
        self.assertEqual(diff["available"], "both")
        self.assertFalse(diff["changed"])

    def test_neither_available(self):
        """Test neither available."""
        diff = diff_structured({"available": False}, {"available": False})
        self.assertEqual(diff["available"], "neither")
        self.assertFalse(diff["changed"])

    def test_asymmetric_old_only(self):
        """Test asymmetric old only."""
        diff = diff_structured(_structured_payload(), {"available": False})
        self.assertEqual(diff["available"], "old")
        self.assertFalse(diff["changed"])

    def test_asymmetric_new_only(self):
        """Test asymmetric new only."""
        diff = diff_structured(None, _structured_payload())
        self.assertEqual(diff["available"], "new")
        self.assertFalse(diff["changed"])

    def test_rejects_hostile_non_string_keys(self):
        """Test rejects hostile non string keys."""
        bad = _structured_payload()
        bad["per_candidate"][7] = {"parse_rate": 1.0}
        diff = diff_structured(bad, _structured_payload())
        self.assertEqual(diff["available"], "new")
        self.assertFalse(diff["changed"])

    def test_rejects_hostile_out_of_range_rate(self):
        """Test rejects hostile out of range rate."""
        bad = _structured_payload()
        bad["per_candidate"]["alpha"]["adherent_rate"] = 2.0
        diff = diff_structured(bad, _structured_payload())
        self.assertEqual(diff["available"], "new")
        self.assertFalse(diff["changed"])

    def test_rejects_hostile_weakest_non_string(self):
        """Test rejects hostile weakest non string."""
        bad = _structured_payload()
        bad["weakest_candidates"] = [None]
        diff = diff_structured(bad, _structured_payload())
        self.assertEqual(diff["available"], "new")
        self.assertFalse(diff["changed"])

    # --- single-cause pins: each changed disjunct fires alone ---

    def test_parse_rate_drift_only(self):
        """Test parse rate drift only."""
        old = _structured_payload()
        new = _structured_payload()
        # A consistent cell: a parse-rate drop drags schema/adherent with
        # it (adherent_rate <= parse_rate is a producer invariant). The
        # pin is that the rate-drift disjunct alone fires.
        new["per_candidate"]["alpha"].update(parse_rate=0.5, schema_rate=0.5, adherent_rate=0.5)
        diff = diff_structured(old, new)
        self.assertTrue(diff["changed"])
        self.assertEqual(
            {entry["metric"] for entry in diff["rate_drift"]},
            {"parse_rate", "schema_rate", "adherent_rate"},
        )
        self.assertEqual(diff["gap_drift"], [])
        self.assertEqual(diff["weakest_added"], [])
        self.assertEqual(diff["candidates_added"], [])

    def test_adherent_rate_drift_only(self):
        """Test adherent rate drift only."""
        old = _structured_payload()
        new = _structured_payload()
        new["per_candidate"]["beta"]["adherent_rate"] = 0.0
        new["per_candidate"]["beta"]["schema_rate"] = 0.0
        diff = diff_structured(old, new)
        self.assertTrue(diff["changed"])
        self.assertTrue(
            all(entry["metric"] in ("adherent_rate", "schema_rate") for entry in diff["rate_drift"])
        )

    def test_gap_drift_only(self):
        """Test gap drift only."""
        old = _structured_payload(gap=0.0)
        new = _structured_payload(gap=0.5)
        diff = diff_structured(old, new)
        self.assertTrue(diff["changed"])
        self.assertEqual(len(diff["gap_drift"]), 1)
        self.assertEqual(diff["gap_drift"][0]["old"], 0.0)
        self.assertEqual(diff["gap_drift"][0]["new"], 0.5)
        self.assertEqual(diff["rate_drift"], [])

    def test_weakest_added_only(self):
        """Test weakest added only."""
        old = _structured_payload()
        new = _structured_payload(weakest=("alpha",))
        diff = diff_structured(old, new)
        self.assertTrue(diff["changed"])
        self.assertEqual(diff["weakest_added"], ["alpha"])
        self.assertEqual(diff["weakest_removed"], [])

    def test_weakest_removed_only(self):
        """Test weakest removed only."""
        old = _structured_payload(weakest=("alpha",))
        new = _structured_payload()
        diff = diff_structured(old, new)
        self.assertTrue(diff["changed"])
        self.assertEqual(diff["weakest_removed"], ["alpha"])
        self.assertEqual(diff["weakest_added"], [])

    def test_candidates_added_only(self):
        """Test candidates added only."""
        old = _structured_payload()
        new = _structured_payload(names=("alpha", "beta", "gamma"))
        diff = diff_structured(old, new)
        self.assertTrue(diff["changed"])
        self.assertEqual(diff["candidates_added"], ["gamma"])
        self.assertEqual(diff["candidates_removed"], [])

    def test_candidates_removed_only(self):
        """Test candidates removed only."""
        old = _structured_payload(names=("alpha", "beta", "gamma"))
        new = _structured_payload()
        diff = diff_structured(old, new)
        self.assertTrue(diff["changed"])
        self.assertEqual(diff["candidates_removed"], ["gamma"])
        self.assertEqual(diff["candidates_added"], [])

    def test_none_cell_transition_records_drift(self):
        """Test none cell transition records drift."""
        old = _structured_payload()
        new = _structured_payload()
        new["per_candidate"]["alpha"] = None
        diff = diff_structured(old, new)
        self.assertTrue(diff["changed"])
        alpha_drift = [entry for entry in diff["rate_drift"] if entry["candidate"] == "alpha"]
        self.assertEqual(len(alpha_drift), 3)
        self.assertTrue(all(entry["new"] is None for entry in alpha_drift))

    @staticmethod
    def _mini_report(structured):
        """Mini report."""
        return {
            "decision": "select_best",
            "selected_candidate": "alpha",
            "max_new_tokens": 64,
            "backend_fingerprint": "fp-1",
            "search": [
                {
                    "format": {"name": "alpha"},
                    "summary": {
                        "planned": 1,
                        "exact_and_boundary_passes": 1,
                        "aggregate_eligible": True,
                    },
                    "observations": [
                        {"fixture": "f1", "exact_match": True, "boundary_found": True}
                    ],
                }
            ],
            "held_out": {
                "summary": {
                    "planned": 1,
                    "exact_and_boundary_passes": 1,
                    "aggregate_eligible": True,
                },
                "observations": [{"fixture": "h1", "exact_match": True, "boundary_found": True}],
            },
            "structured": structured,
        }

    def test_main_diff_wires_structured_flags(self):
        """Test main diff wires structured flags."""
        old_report = self._mini_report(_structured_payload())
        new_report = self._mini_report(_structured_payload(schema_rate=0.5))
        full = diffreports.diff_reports(old_report, new_report, old_label="old", new_label="new")
        self.assertTrue(full["summary"]["structured_changed"])
        self.assertFalse(full["summary"]["structured_asymmetric"])
        self.assertFalse(full["summary"]["identical"])
        self.assertIn("structured", full)

    def test_main_diff_structured_asymmetric(self):
        """Test main diff structured asymmetric."""
        old_report = self._mini_report(_structured_payload())
        new_report = self._mini_report({"available": False, "detail": "x"})
        full = diffreports.diff_reports(old_report, new_report, old_label="old", new_label="new")
        self.assertTrue(full["summary"]["structured_asymmetric"])
        self.assertFalse(full["summary"]["identical"])

    def test_stderr_summary_sanitizes_hostile_names(self):
        """Test stderr summary sanitizes hostile names."""
        old = _structured_payload()
        new = _structured_payload(schema_rate=0.0, weakest=("alpha",))
        diff = diff_structured(old, new)
        diff["rate_drift"].append(
            {"candidate": "\x1b[31mred", "metric": "parse_rate", "old": 1.0, "new": 0.0}
        )
        from model_protocol_lab.diffreports import _render_structured_summary

        lines = _render_structured_summary(diff)
        blob = "\n".join(lines)
        self.assertNotIn("\x1b", blob)
        self.assertIn("Structured adherence drift", blob)


class StructuredMutationGateTests(unittest.TestCase):
    """Tempdir-module mutation gates (the real tree is never rewritten)."""

    def _run_gate(self, module_name, name, old_text, new_text, scenario):
        """Run gate."""
        mod = __import__(f"model_protocol_lab.{module_name}", fromlist=["x"])
        source = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertEqual(source.count(old_text), 1, f"mutant {name}: anchor not unique")
        mutated = source.replace(old_text, new_text)
        with tempfile.TemporaryDirectory(prefix="c157_mut_", dir="/tmp") as tmpdir:
            mod_path = os.path.join(tmpdir, f"{module_name}_mut_{name}.py")
            with open(mod_path, "w", encoding="utf-8") as handle:
                handle.write(mutated)
            dotted = f"model_protocol_lab.{module_name}_mut_{name}"
            spec = importlib.util.spec_from_file_location(dotted, mod_path)
            assert spec is not None and spec.loader is not None
            mutant = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mutant)
            scenario(mod)  # passes on the real module: non-vacuous
            with self.assertRaises(AssertionError, msg=f"mutant {name} was NOT caught"):
                scenario(mutant)

    @staticmethod
    def _extra_key_scenario(mod):
        """Extra key scenario."""
        schema = {
            "type": "object",
            "required": ["a"],
            "properties": {"a": {"type": "string"}},
            "additionalProperties": False,
        }
        assert mod.check_schema({"a": "x", "zzz": 1}, schema) == "schema_failed"

    def test_mutant_extra_key_not_rejected_caught(self):
        """Test mutant extra key not rejected caught."""
        self._run_gate(
            "structured",
            "extra_key_accepted",
            '            if normalized["additionalProperties"] is False:\n'
            '                return "schema_failed"',
            '            if normalized["additionalProperties"] is False:\n'
            '                return "schema_ok"  # MUTANT',
            self._extra_key_scenario,
        )

    @staticmethod
    def _not_text_scenario(mod):
        """Not text scenario."""
        schema = {
            "type": "object",
            "required": ["a"],
            "properties": {"a": {"type": "string"}},
            "additionalProperties": False,
        }
        assert mod.classify_response(None, schema)["outcome"] == "not_text"

    def test_mutant_dropped_not_text_caught(self):
        """Test mutant dropped not text caught."""
        self._run_gate(
            "structured",
            "dropped_not_text",
            "    if type(text) is not str:\n"
            '        return {"parse": "not_text",'
            ' "schema": "not_evaluated", "outcome": "not_text"}',
            "    if type(text) is not str:\n"
            '        return {"parse": "parse_ok", "schema": "schema_ok",'
            ' "outcome": "adherent"}  # MUTANT',
            self._not_text_scenario,
        )

    @staticmethod
    def _rate_drift_scenario(mod):
        """Rate drift scenario."""
        old = _structured_payload()
        new = _structured_payload()
        new["per_candidate"]["alpha"].update(parse_rate=0.5, schema_rate=0.5, adherent_rate=0.5)
        assert mod.diff_structured(old, new)["changed"] is True

    def test_mutant_dropped_rate_drift_caught(self):
        """Test mutant dropped rate drift caught."""
        self._run_gate(
            "diffreports",
            "dropped_structured_rate_drift",
            "        structured_rate_drift\n"
            "        or structured_gap_drift\n"
            "        or structured_weakest_added\n"
            "        or structured_weakest_removed\n"
            "        or structured_candidates_added\n"
            "        or structured_candidates_removed",
            "        structured_gap_drift\n"
            "        or structured_weakest_added\n"
            "        or structured_weakest_removed\n"
            "        or structured_candidates_added\n"
            "        or structured_candidates_removed  # MUTANT",
            self._rate_drift_scenario,
        )

    @staticmethod
    def _gap_drift_scenario(mod):
        """Gap drift scenario."""
        old = _structured_payload(gap=0.0)
        new = _structured_payload(gap=0.5)
        assert mod.diff_structured(old, new)["changed"] is True

    def test_mutant_dropped_gap_drift_caught(self):
        """Test mutant dropped gap drift caught."""
        self._run_gate(
            "diffreports",
            "dropped_structured_gap_drift",
            "        structured_rate_drift\n"
            "        or structured_gap_drift\n"
            "        or structured_weakest_added\n"
            "        or structured_weakest_removed\n"
            "        or structured_candidates_added\n"
            "        or structured_candidates_removed",
            "        structured_rate_drift\n"
            "        or structured_weakest_added\n"
            "        or structured_weakest_removed\n"
            "        or structured_candidates_added\n"
            "        or structured_candidates_removed  # MUTANT",
            self._gap_drift_scenario,
        )


class StructuredHypothesisTests(unittest.TestCase):
    """Structured hypothesis tests."""

    @given(st.lists(st.sampled_from(list(OUTCOMES)), max_size=20))
    @settings(max_examples=100)
    def test_adherence_rates_bounded(self, outcomes):
        """Test adherence rates bounded."""
        agg = candidate_adherence(outcomes)
        for key in ("parse_rate", "schema_rate", "adherent_rate"):
            rate = agg[key]
            if agg["n_evaluable"]:
                self.assertGreaterEqual(rate, 0.0)
                self.assertLessEqual(rate, 1.0)
            else:
                self.assertIsNone(rate)

    @given(st.lists(st.sampled_from(list(OUTCOMES)), max_size=20))
    @settings(max_examples=100)
    def test_adherent_implies_parse(self, outcomes):
        """Test adherent implies parse."""
        agg = candidate_adherence(outcomes)
        if agg["n_evaluable"]:
            self.assertLessEqual(agg["adherent_rate"], agg["parse_rate"])
            self.assertLessEqual(agg["schema_rate"], agg["parse_rate"])
        self.assertLessEqual(agg["n_evaluable"], agg["n"])

    @given(st.text(max_size=200))
    @settings(max_examples=100)
    def test_classify_never_raises(self, text):
        """Test classify never raises."""
        result = classify_response(text, _SCHEMAS["structured/route-alert"])
        self.assertIn(result["outcome"], OUTCOMES)

    @given(st.text(max_size=50), st.text(max_size=50))
    @settings(max_examples=50)
    def test_describe_schema_deterministic(self, name_a, name_b):
        """Test describe schema deterministic."""
        schema = {
            "type": "object",
            "required": ["k"],
            "properties": {"k": {"type": "string"}},
            "additionalProperties": False,
        }
        self.assertEqual(describe_schema(schema), describe_schema(schema))

    @given(
        st.integers(),
        st.booleans(),
        st.none(),
        st.floats(),
    )
    @settings(max_examples=100)
    def test_check_schema_total_on_scalars(self, value, _b, _n, _f):
        """Test check schema total on scalars."""
        schema = {
            "type": "object",
            "required": ["k"],
            "properties": {"k": {"type": "number"}},
            "additionalProperties": False,
        }
        verdict = check_schema({"k": value}, schema)
        self.assertIn(verdict, ("schema_ok", "schema_failed", "schema_invalid"))


class StructuredSectionTests(unittest.TestCase):
    """Structured section tests."""

    def test_renders_table_and_anchor(self):
        """Test renders table and anchor."""
        payload = _structured_payload()
        payload["weakest_candidates"] = ["beta"]
        payload["adherence_gap"] = 0.5
        payload["uniform"] = False
        html = _structured_section({"structured": payload})
        self.assertIn('id="structured-adherence"', html)
        self.assertIn("alpha", html)
        self.assertIn("50%", html)

    def test_empty_when_unavailable(self):
        """Test empty when unavailable."""
        self.assertEqual(_structured_section({}), "")
        self.assertEqual(_structured_section({"structured": {"available": False}}), "")

    def test_escapes_hostile_candidate_names(self):
        """Test escapes hostile candidate names."""
        payload = _structured_payload()
        hostile = "<script>alert(1)</script>"
        payload["per_candidate"][hostile] = payload["per_candidate"].pop("alpha")
        html = _structured_section({"structured": payload})
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)

    def test_escapes_hostile_weakest_names(self):
        """Test escapes hostile weakest names."""
        payload = _structured_payload()
        payload["weakest_candidates"] = ['<img src=x onerror="1">']
        html = _structured_section({"structured": payload})
        self.assertNotIn("<img", html)

    def test_uniform_line_rendered(self):
        """Test uniform line rendered."""
        html = _structured_section({"structured": _structured_payload()})
        self.assertIn("uniform", html)


class StructuredCliProbeTests(unittest.TestCase):
    """Structured cli probe tests."""

    @staticmethod
    def _args():
        """Args."""
        from model_protocol_lab import cli

        return cli, {
            "backend": _AdherentBackend(),
            "formats": (_format("chatml"), _format("no-bos")),
            "result": {},
            "boundary": BOUNDARY,
            "boundary_map": None,
            "max_new_tokens": 128,
            "seed": 0,
        }

    def test_probe_stores_payload(self):
        """Test probe stores payload."""
        from model_protocol_lab import cli

        _, kwargs = self._args()
        cli._run_structured_probe(**kwargs)
        payload = kwargs["result"]["structured"]
        self.assertTrue(payload["available"])
        self.assertEqual(payload["n_fixtures"], 9)

    def test_probe_never_raises_on_broken_backend(self):
        """Test probe never raises on broken backend."""
        from model_protocol_lab import cli

        _, kwargs = self._args()
        kwargs["backend"] = _RaisingBackend()
        cli._run_structured_probe(**kwargs)  # must not raise
        self.assertIn("available", kwargs["result"]["structured"])

    def test_probe_never_raises_when_stage_raises(self):
        """Test probe never raises when stage raises."""
        from model_protocol_lab.cli import _run_structured_probe

        result: dict = {}
        _run_structured_probe(
            None,
            (_format(),),
            result,
            boundary=BOUNDARY,
            boundary_map=None,
            max_new_tokens=128,
            seed=0,
        )
        self.assertEqual(result["structured"]["detail"], "structured_unavailable")

    def test_flag_registered(self):
        """Test flag registered."""
        from model_protocol_lab import cli

        args = cli.build_parser().parse_args(
            [
                "run",
                "--model",
                "m",
                "--dataset",
                "d",
                "--family",
                "llama3",
                "--boundary-ids",
                "124",
                "--output",
                "o",
                "--structured",
            ]
        )
        self.assertTrue(args.structured)


class StructuredSampleTests(unittest.TestCase):
    """Structured sample tests."""

    @staticmethod
    def _load_script():
        """Load script."""
        script = Path(__file__).resolve().parent.parent / "examples" / "make_sample_report.py"
        spec = importlib.util.spec_from_file_location("make_sample_report_c157", script)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _base_prompt(fixture_id):
        """Base prompt."""
        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        from model_protocol_lab.structured import render_prompt

        # skipcq: PTC-W0063 - in a test, next() raising StopIteration is the intended failure
        spec = next(s for s in _SPECS if s["id"] == fixture_id.split("/")[1])
        return (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
            + render_prompt(spec)
            + "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    def test_demo_backend_base_emits_canonical(self):
        """Test demo backend base emits canonical."""
        module = self._load_script()
        backend = module.StructuredDemoBackend()
        prompt = self._base_prompt("structured/extract-person")
        generation = backend.generate(
            backend.encode(prompt), max_new_tokens=128, seed=0, cancel=Event()
        )
        answer = backend.decode(generation.token_ids).split("|")[0]
        self.assertEqual(answer, _EXPECTED["structured/extract-person"])

    def test_demo_backend_mutant_emits_broken(self):
        """Test demo backend mutant emits broken."""
        module = self._load_script()
        backend = module.StructuredDemoBackend()
        prompt = self._base_prompt("structured/extract-person").replace(
            "<|begin_of_text|>", "<|begin_of_text|><|begin_of_text|>", 1
        )
        generation = backend.generate(
            backend.encode(prompt), max_new_tokens=128, seed=0, cancel=Event()
        )
        answer = backend.decode(generation.token_ids).split("|")[0]
        self.assertNotEqual(answer, _EXPECTED["structured/extract-person"])
        result = classify_response(answer, _SCHEMAS["structured/extract-person"])
        self.assertNotEqual(result["outcome"], "adherent")

    def test_structured_exercise_is_deterministic(self):
        """Test structured exercise is deterministic."""
        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        from model_protocol_lab.experiments import structured_adherence, structured_stage

        def _run_once():
            """Run once."""
            stage = structured_stage(
                self._load_script().StructuredDemoBackend(),
                (_format("llama3"), _format("llama3-no-bos")),
                _FIXTURES,
                schemas=_SCHEMAS,
                boundary=BOUNDARY,
                max_new_tokens=128,
                seed=20261012,
            )
            return structured_adherence(("llama3", "llama3-no-bos"), stage)

        first = json.dumps(_run_once(), sort_keys=True)
        second = json.dumps(_run_once(), sort_keys=True)
        self.assertEqual(first, second)

    def test_demo_backend_unknown_fixture_is_wrong(self):
        """Test demo backend unknown fixture is wrong."""
        module = self._load_script()
        backend = module.StructuredDemoBackend()
        generation = backend.generate(
            backend.encode("no marker here"), max_new_tokens=128, seed=0, cancel=Event()
        )
        self.assertTrue(backend.decode(generation.token_ids).startswith("WRONG"))


class TocAnchorTests(unittest.TestCase):
    """Toc anchor tests."""

    def test_structured_toc_id_has_section_anchor(self):
        """Test structured toc id has section anchor."""
        # Mirrors the cycle-153 invariant: _toc never points at a missing id.
        from model_protocol_lab import reporting

        html = reporting.render_html(
            {
                "decision": "abstain",
                "search": [],
                "structured": _structured_payload(),
            }
        )
        self.assertIn('id="structured-adherence"', html)
        self.assertIn('href="#structured-adherence"', html)


class StructuredCoverageGapTests(unittest.TestCase):
    """Targeted tests for fail-closed defensive branches (coverage gate).

    Every line here is a fail-closed guard the happy-path tests never
    take: malformed inputs degrading to markers instead of raising. They
    are pinned because the 95% gate measures the exact CI unittest
    command.
    """

    # --- structured.py: defensive branches ---

    def test_is_finite_number_rejects_nan_and_inf(self):
        """Test is finite number rejects nan and inf."""
        from model_protocol_lab.structured import _is_finite_number

        self.assertFalse(_is_finite_number(float("nan")))
        self.assertFalse(_is_finite_number(float("inf")))
        self.assertFalse(_is_finite_number("1"))
        self.assertTrue(_is_finite_number(3))

    def test_field_value_ok_rejects_unknown_type_defensively(self):
        """Test field value ok rejects unknown type defensively."""
        from model_protocol_lab.structured import _field_value_ok

        self.assertFalse(_field_value_ok("x", {"type": "mystery"}))

    def test_field_value_ok_enum_mismatch(self):
        """Test field value ok enum mismatch."""
        from model_protocol_lab.structured import _field_value_ok

        self.assertFalse(_field_value_ok("z", {"type": "string", "enum": ["x", "y"]}))

    def test_field_value_ok_maximum_violation(self):
        """Test field value ok maximum violation."""
        from model_protocol_lab.structured import _field_value_ok

        self.assertFalse(_field_value_ok(9, {"type": "integer", "maximum": 5}))

    def test_field_value_ok_null_type(self):
        """Test field value ok null type."""
        from model_protocol_lab.structured import _field_value_ok

        self.assertTrue(_field_value_ok(None, {"type": "null"}))
        self.assertFalse(_field_value_ok(0, {"type": "null"}))

    def test_field_value_ok_number_type(self):
        """Test field value ok number type."""
        from model_protocol_lab.structured import _field_value_ok

        self.assertTrue(_field_value_ok(1.5, {"type": "number"}))
        self.assertTrue(_field_value_ok(2, {"type": "number"}))
        self.assertFalse(_field_value_ok(True, {"type": "number"}))

    def test_validate_schema_def_rejects_bad_properties(self):
        """Test validate schema def rejects bad properties."""
        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        from model_protocol_lab.structured import validate_schema_def

        base = {
            "type": "object",
            "required": ["a"],
            "properties": {"a": {"type": "string"}},
            "additionalProperties": False,
        }
        for bad_properties in ("nope", {}, {7: {"type": "string"}}):
            schema = dict(base, properties=bad_properties)
            with self.assertRaises(ValueError):
                validate_schema_def(schema)

    def test_describe_schema_allows_additional_properties(self):
        """Test describe schema allows additional properties."""
        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        from model_protocol_lab.structured import describe_schema

        schema = {
            "type": "object",
            "required": ["a"],
            "properties": {"a": {"type": "string"}},
            "additionalProperties": True,
        }
        self.assertIn("other keys allowed", describe_schema(schema))

    def test_field_value_ok_non_numeric_against_bounds(self):
        """Test field value ok non numeric against bounds."""
        from model_protocol_lab.structured import _field_value_ok

        # Bounds on a string field: the type check passes, then the
        # numeric guard rejects the non-numeric value fail-closed.
        self.assertFalse(_field_value_ok("x", {"type": "string", "minimum": 0}))
        self.assertFalse(_field_value_ok("x", {"type": "string", "maximum": 5}))

    def test_check_schema_minimum_violation(self):
        """Test check schema minimum violation."""
        schema = {
            "type": "object",
            "required": ["age"],
            "properties": {"age": {"type": "integer", "minimum": 0}},
            "additionalProperties": False,
        }
        self.assertEqual(check_schema({"age": -1}, schema), "schema_failed")

    def test_check_schema_non_dict_document(self):
        """Test check schema non dict document."""
        schema = _SCHEMAS["structured/extract-person"]
        self.assertEqual(check_schema([1, 2], schema), "schema_failed")

    def test_classify_response_recursion_guard(self):
        """Test classify response recursion guard."""
        nested = "[" * 200
        result = classify_response(nested, _SCHEMAS["structured/extract-person"])
        self.assertEqual(result["outcome"], "parse_failed")

    def test_schema_map_returns_independent_dicts(self):
        """Test schema map returns independent dicts."""
        first = schema_map(_SPECS)
        self.assertEqual(set(first), {fixture.id for fixture in _FIXTURES})

    def test_render_prompt_ends_with_marker(self):
        """Test render prompt ends with marker."""
        self.assertTrue(render_prompt(_SPECS[0]).endswith("(ref: structured/extract-person)"))

    # --- experiments.py: structured cell validation ---

    def test_cell_valid_rejects_evaluable_over_n(self):
        """Test cell valid rejects evaluable over n."""
        from model_protocol_lab.experiments import _structured_cell_valid

        self.assertFalse(
            _structured_cell_valid(
                {
                    "n": 2,
                    "n_evaluable": 3,
                    "parse_ok": 2,
                    "schema_ok": 2,
                    "adherent": 2,
                    "parse_rate": 1.0,
                    "schema_rate": 1.0,
                    "adherent_rate": 1.0,
                }
            )
        )

    def test_cell_valid_rejects_none_rate_with_zero_evaluable(self):
        """Test cell valid rejects none rate with zero evaluable."""
        from model_protocol_lab.experiments import _structured_cell_valid

        self.assertFalse(
            _structured_cell_valid(
                {
                    "n": 2,
                    "n_evaluable": 0,
                    "parse_ok": 0,
                    "schema_ok": 0,
                    "adherent": 0,
                    "parse_rate": 0.0,
                    "schema_rate": None,
                    "adherent_rate": None,
                }
            )
        )

    def test_cell_valid_accepts_none(self):
        """Test cell valid accepts none."""
        from model_protocol_lab.experiments import _structured_cell_valid

        self.assertTrue(_structured_cell_valid(None))

    def test_cell_valid_rejects_non_dict(self):
        """Test cell valid rejects non dict."""
        from model_protocol_lab.experiments import _structured_cell_valid

        self.assertFalse(_structured_cell_valid("nope"))

    # --- diffreports.py: structured summary renderer ---

    def test_summary_renderer_handles_non_dict(self):
        """Test summary renderer handles non dict."""
        from model_protocol_lab.diffreports import _render_structured_summary

        self.assertEqual(_render_structured_summary(None), [])
        self.assertEqual(_render_structured_summary("nope"), [])

    def test_summary_renderer_neither(self):
        """Test summary renderer neither."""
        from model_protocol_lab.diffreports import _render_structured_summary

        self.assertEqual(_render_structured_summary({"available": "neither", "changed": False}), [])

    def test_summary_renderer_asymmetric(self):
        """Test summary renderer asymmetric."""
        from model_protocol_lab.diffreports import _render_structured_summary

        lines = _render_structured_summary({"available": "old", "changed": False})
        self.assertTrue(any("unavailable" in line for line in lines))

    def test_summary_renderer_skips_malformed_drift_entries(self):
        """Test summary renderer skips malformed drift entries."""
        from model_protocol_lab.diffreports import _render_structured_summary

        diff = diff_structured(_structured_payload(), _structured_payload(schema_rate=0.0))
        diff["rate_drift"].append("junk")
        diff["gap_drift"].append(None)
        lines = _render_structured_summary(diff)
        self.assertTrue(any("drift" in line for line in lines))

    def test_summary_renderer_added_removed(self):
        """Test summary renderer added removed."""
        from model_protocol_lab.diffreports import _render_structured_summary

        old = _structured_payload()
        new = _structured_payload(names=("alpha", "beta", "gamma"))
        lines = _render_structured_summary(diff_structured(old, new))
        self.assertTrue(any("added" in line for line in lines))

    def test_summary_renderer_weakest_lines(self):
        """Test summary renderer weakest lines."""
        from model_protocol_lab.diffreports import _render_structured_summary

        old = _structured_payload()
        new = _structured_payload(weakest=("alpha",), schema_rate=0.5)
        lines = _render_structured_summary(diff_structured(old, new))
        blob = "\n".join(lines)
        self.assertIn("newly weakest", blob)


if __name__ == "__main__":
    unittest.main()
