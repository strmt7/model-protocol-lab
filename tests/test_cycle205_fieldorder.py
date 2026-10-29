"""Cycle 205: schema field-order probe (`run --field-order`).

The instrument (docs/FIELDORDER_DESIGN.md v2, adjudicated in
the cycle-205 coordinator adjudication): fifteen frozen
extraction/classification/routing specs -- the nine shipped
``structured`` specs reused unchanged plus six fieldorder
specs -- each rendered in four field orders (canonical,
answer-first, answer-last, reversed); the answer field's
value accuracy is compared between the endpoint arms over
the specs the frozen reader-entailment rule pairs. These
tests pin the adjudication's load-bearing points: the paired
pool is exactly ten specs (F1), the None sentinel means "no
canonical form" and never a produced JSON null (F2), the
distinct-prompt count is 37 of 60 and is carried by the
aggregation itself (F3), duplicate keys are an adherence
condition and never an input to the schema verdict (F7 --
the verdict is the flat schema check over the reconstructed
document and nothing else), the parser and validator are
total over hostile stored strings (F10), and the diff
validator recomputes the whole section from the commitment.

The stage is exercised with a scripted word-token backend.
Its generate() honours the core.evaluate contract: a
``limit`` result carries exactly max_new_tokens tokens.
"""

from __future__ import annotations

import copy
import re
import threading
import unittest

from model_protocol_lab import cli, experiments, fieldorder, reporting, structured
from model_protocol_lab.core import Format, Generation
from model_protocol_lab.diffreports import _valid_field_order, diff_field_order

FORMATS = (
    Format(name="fo-a", bos="", user_prefix="u:", assistant_prefix="a:", turn_end="</s>"),
    Format(name="fo-b", bos="", user_prefix="U>", assistant_prefix="A>", turn_end="<e>"),
)

_REF_RE = re.compile(r"\(ref: (?:structured|fieldorder)/([a-z0-9-]+)\)")

PAIRED_IDS = {
    "extract-person",
    "extract-city",
    "extract-book",
    "extract-event",
    "extract-product",
    "extract-vehicle",
    "classify-document",
    "route-ticket",
    "route-delivery",
    "route-complaint",
}


def _wrong_value(spec, field):
    """A schema-valid but incorrect value for a spec field."""
    schema_field = spec["schema"]["properties"][field]
    expected = spec["answer"][field]
    enum = schema_field.get("enum")
    if enum is not None:
        # skipcq: PTC-W0063 - in a test, next() raising StopIteration is the intended failure
        return next(value for value in enum if value != expected)
    if schema_field["type"] == "integer":
        for candidate in (expected + 1, expected - 1):
            if schema_field.get("minimum") is not None and candidate < schema_field["minimum"]:
                continue
            if schema_field.get("maximum") is not None and candidate > schema_field["maximum"]:
                continue
            return candidate
        raise AssertionError("no in-range wrong integer")
    if schema_field["type"] == "boolean":
        return not expected
    return "ZZZ wrong"


def _presented_order(text, spec):
    block = text.split("matching this schema: {", 1)[1].split("} (required:", 1)[0]
    return [part.split(":", 1)[0].strip().strip('"') for part in block.split("; ")]


class _FieldOrderBackend:
    """Word-token scripted backend with switchable behaviours.

    Modes: ``perfect`` (emit the frozen answer as canonical
    JSON), ``last_wrong`` (when the presented order is the
    answer_last order and not also the answer_first order,
    emit a schema-valid wrong value for the answer field --
    the endpoint discordance the decision exists to catch),
    ``wrong`` (emit non-JSON text), ``dup`` (emit the answer
    with the answer field duplicated, wrong value first so
    the last-wins parse is the correct document),
    ``noboundary`` (never emit the boundary token), ``nan``
    (emit one integer field as the bare NaN literal),
    ``verbose`` (emit one 5,000-char string field, over the
    per-record commitment cap), ``explode`` (generate raises).
    ``boundary_word`` is the word whose token terminates a
    generation (the stage's boundary must be its encoding).
    """

    fingerprint = "sha256:test-fieldorder"

    def __init__(self, mode: str = "perfect", boundary_word: str = "endmark"):
        self.mode = mode
        self.boundary_word = boundary_word
        self.calls = 0
        self.vocab: dict[str, int] = {}
        self.rev: dict[int, str] = {}

    def encode(self, text):
        ids = []
        for token in text.replace("\n", " \n ").split(" "):
            if not token:
                continue
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab) + 1
                self.rev[self.vocab[token]] = token
            ids.append(self.vocab[token])
        return tuple(ids)

    def decode(self, ids):
        return " ".join(self.rev.get(int(i), "?") for i in ids)

    def generate(self, input_ids, *, max_new_tokens, seed=None, cancel=None):
        self.calls += 1
        if self.mode == "explode":
            raise RuntimeError("boom")
        text = self.decode(list(input_ids))
        match = _REF_RE.search(text)
        spec = fieldorder.spec_by_id(match.group(1)) if match else None
        if spec is None or self.mode == "wrong":
            answer_text = "no idea"
        elif self.mode == "verbose":
            answer_text = '{"text": "' + "x" * 5000 + '"}'
        else:
            answer = dict(spec["answer"])
            field = spec["answer_field"]
            if self.mode == "last_wrong":
                presented = _presented_order(text, spec)
                if presented == fieldorder.arm_order(
                    spec, "answer_last"
                ) and presented != fieldorder.arm_order(spec, "answer_first"):
                    answer[field] = _wrong_value(spec, field)
            answer_text = structured._canonical_json(answer)
            if self.mode == "dup":
                wrong = structured._canonical_json(_wrong_value(spec, field))
                answer_text = "{" + f'"{field}": {wrong}, ' + answer_text[1:]
            if self.mode == "nan":
                for key, value in answer.items():
                    if type(value) is int:
                        answer_text = answer_text.replace(f'"{key}": {value}', f'"{key}": NaN', 1)
                        break
        boundary = self.encode(self.boundary_word)[0]
        out = list(self.encode(answer_text))
        if self.mode == "noboundary":
            filler = self.encode("padding")[0]
            out = (out + [filler] * max_new_tokens)[:max_new_tokens]
            return Generation(token_ids=tuple(out), reason="limit")
        out = (out + [boundary] * max_new_tokens)[:max_new_tokens]
        return Generation(token_ids=tuple(out), reason="limit")


def _payload(backend=None, formats=FORMATS, seed=0, **stage_kwargs):
    backend = backend if backend is not None else _FieldOrderBackend()
    stage_kwargs.setdefault("max_context", 8192)
    stage = experiments.fieldorder_stage(
        backend,
        formats,
        boundary=tuple(backend.encode("endmark")),
        seed=seed,
        **stage_kwargs,
    )
    return experiments.fieldorder_analysis(stage, tuple(f.name for f in formats))


def _record(spec, arm, text, **overrides):
    """A synthetic record committing to the document in ``text``."""
    if text is None:
        parsed = {"kind": "", "field_values": {}, "duplicate_keys": False, "document_digest": None}
    else:
        parsed = fieldorder.parse_document(text)
    record = {
        "candidate": "cand",
        "spec_id": spec["id"],
        "arm": arm,
        "detail": "",
        "case_screened": False,
        "boundary_found": True,
        "duplicate_keys": parsed["duplicate_keys"],
        "answer_digest": fieldorder.digest_text(text) if text is not None else None,
        "document_kind": parsed["kind"],
        "document_digest": parsed["document_digest"],
        "field_values": parsed["field_values"],
        "input_tokens": 10,
        "content_digest": fieldorder.digest_text(fieldorder.arm_prompt(spec, arm)),
    }
    record.update(overrides)
    return record


def _good(spec):
    return structured._canonical_json(spec["answer"])


def _corrupted(spec, arm):
    document = dict(spec["answer"])
    field = spec["answer_field"]
    document[field] = _wrong_value(spec, field)
    return structured._canonical_json(document)


class TestSpecTable(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(
            fieldorder.ARM_KEYS, ("canonical", "answer_first", "answer_last", "reversed")
        )
        self.assertEqual(fieldorder.SPEC_COUNT, 15)
        self.assertEqual(fieldorder.RECORDS_PER_CANDIDATE, 60)
        self.assertEqual(fieldorder.MAX_FIELD_ORDER_RECORDS, 1440)
        self.assertEqual(fieldorder.MAX_FIELD_ORDER_PAYLOAD_BYTES, 8 * 1024 * 1024)
        self.assertEqual(fieldorder.FIELD_VALUES_CAP, 4096)
        self.assertEqual(fieldorder.MAX_NEW_TOKENS, 64)
        self.assertEqual(fieldorder.MIN_DISCORDANT_SPECS, 6)
        self.assertEqual(fieldorder.SIGN_ALPHA, 0.05)

    def test_fifteen_specs_ten_paired(self):
        # F1: pairing eligibility is the frozen per-spec flag,
        # set only where a reader given question, context and
        # schema can derive the answer field's value uniquely.
        self.assertEqual(len(fieldorder.FIELD_ORDER_SPECS), 15)
        self.assertEqual({s["id"] for s in fieldorder.FIELD_ORDER_SPECS if s["paired"]}, PAIRED_IDS)
        self.assertEqual(
            {s["id"] for s in fieldorder.FIELD_ORDER_SPECS if not s["paired"]},
            {
                "classify-sentiment",
                "classify-urgency",
                "classify-language",
                "route-alert",
                "route-order",
            },
        )

    def test_spec_table_shape(self):
        specs = fieldorder.FIELD_ORDER_SPECS
        self.assertEqual(sum(1 for s in specs if s["source"] == "structured"), 9)
        self.assertEqual(sum(1 for s in specs if s["source"] == "fieldorder"), 6)
        self.assertEqual(len({s["id"] for s in specs}), 15)
        for spec in specs:
            # Every frozen answer passes its own schema, and
            # every numeric field is an integer (F4/F11: no
            # float equality is ever load-bearing).
            self.assertEqual(
                structured.check_schema(spec["answer"], spec["schema"]), "schema_ok", spec["id"]
            )
            self.assertIn(spec["answer_field"], spec["schema"]["properties"])
            self.assertEqual(set(spec["schema"]["required"]), set(spec["schema"]["properties"]))
            for field in spec["schema"]["properties"].values():
                self.assertNotEqual(field["type"], "number")

    def test_canonical_answer_positions(self):
        positions = {"first": 0, "last": 0, "middle": 0}
        for spec in fieldorder.FIELD_ORDER_SPECS:
            if not spec["paired"]:
                continue
            order = fieldorder.canonical_order(spec)
            index = order.index(spec["answer_field"])
            if index == 0:
                positions["first"] += 1
            elif index == len(order) - 1:
                positions["last"] += 1
            else:
                positions["middle"] += 1
        self.assertEqual(positions, {"first": 3, "last": 3, "middle": 4})

    def test_distinct_prompts_and_endpoint_pairs(self):
        # F3: 37 distinct prompts across the 60 (spec, arm)
        # cells; the endpoint orders differ on 12 of 15 specs
        # (they coincide exactly on the three 1-field specs).
        prompts = {
            fieldorder.arm_prompt(spec, arm)
            for spec in fieldorder.FIELD_ORDER_SPECS
            for arm in fieldorder.ARM_KEYS
        }
        self.assertEqual(len(prompts), 37)
        endpoint_distinct = sum(
            1
            for spec in fieldorder.FIELD_ORDER_SPECS
            if fieldorder.arm_order(spec, "answer_first")
            != fieldorder.arm_order(spec, "answer_last")
        )
        self.assertEqual(endpoint_distinct, 12)

    def test_distinct_orders_follow_the_position_rule(self):
        total = 0
        for spec in fieldorder.FIELD_ORDER_SPECS:
            orders = {tuple(fieldorder.arm_order(spec, arm)) for arm in fieldorder.ARM_KEYS}
            n_fields = len(spec["schema"]["properties"])
            position = fieldorder.canonical_order(spec).index(spec["answer_field"])
            if n_fields == 1:
                expected = 1
            elif n_fields == 2:
                expected = 2
            elif position in (0, n_fields - 1):
                expected = 3
            else:
                expected = 4
            self.assertEqual(len(orders), expected, spec["id"])
            total += len(orders)
        self.assertEqual(total, 37)

    def test_canonical_coincides_with_an_endpoint_on_six_paired(self):
        count = 0
        for spec in fieldorder.FIELD_ORDER_SPECS:
            if not spec["paired"]:
                continue
            canonical = fieldorder.arm_order(spec, "canonical")
            if canonical in (
                fieldorder.arm_order(spec, "answer_first"),
                fieldorder.arm_order(spec, "answer_last"),
            ):
                count += 1
        self.assertEqual(count, 6)

    def test_spec_by_id_unknown_raises(self):
        with self.assertRaises(ValueError):
            fieldorder.spec_by_id("nope")


class TestRendering(unittest.TestCase):
    def test_describe_byte_identity_for_all_specs(self):
        for spec in fieldorder.FIELD_ORDER_SPECS:
            self.assertEqual(
                fieldorder.describe_ordered(spec["schema"], fieldorder.canonical_order(spec)),
                structured.describe_schema(spec["schema"]),
                spec["id"],
            )

    def test_canonical_prompt_byte_identity_for_shipped_specs(self):
        shipped = {spec["id"]: spec for spec in structured.STRUCTURED_SPECS}
        self.assertEqual(len(shipped), 9)
        for spec in fieldorder.FIELD_ORDER_SPECS:
            if spec["id"] in shipped:
                self.assertEqual(
                    fieldorder.arm_prompt(spec, "canonical"),
                    structured.render_prompt(shipped[spec["id"]]),
                    spec["id"],
                )

    def test_arm_orders(self):
        spec = fieldorder.spec_by_id("extract-product")
        self.assertEqual(fieldorder.canonical_order(spec), ["brand", "in_stock", "model", "price"])
        self.assertEqual(
            fieldorder.arm_order(spec, "answer_first"), ["model", "brand", "in_stock", "price"]
        )
        self.assertEqual(
            fieldorder.arm_order(spec, "answer_last"), ["brand", "in_stock", "price", "model"]
        )
        self.assertEqual(
            fieldorder.arm_order(spec, "reversed"), ["price", "model", "in_stock", "brand"]
        )
        described = fieldorder.describe_ordered(
            spec["schema"], fieldorder.arm_order(spec, "answer_first")
        )
        self.assertIn("required: model, brand, in_stock, price", described)
        with self.assertRaises(ValueError):
            fieldorder.arm_order(spec, "sideways")

    def test_new_spec_prompt_carries_fieldorder_ref(self):
        prompt = fieldorder.arm_prompt(fieldorder.spec_by_id("extract-product"), "canonical")
        self.assertIn("(ref: fieldorder/extract-product)", prompt)
        shipped = fieldorder.arm_prompt(fieldorder.spec_by_id("extract-person"), "canonical")
        self.assertIn("(ref: structured/extract-person)", shipped)


class TestProtocol(unittest.TestCase):
    def test_protocol_object_and_fingerprint(self):
        proto = fieldorder.protocol_object(7, "sha256:x")
        self.assertEqual(proto["spec_count"], 15)
        self.assertEqual(proto["max_records"], 1440)
        self.assertEqual(proto["max_new_tokens"], 64)
        self.assertEqual(proto["field_values_cap"], 4096)
        self.assertEqual(proto["min_discordant_specs"], 6)
        finger = fieldorder.protocol_fingerprint(7, "sha256:x")
        self.assertEqual(finger, fieldorder.protocol_fingerprint(7, "sha256:x"))
        self.assertNotEqual(finger, fieldorder.protocol_fingerprint(8, "sha256:x"))
        self.assertNotEqual(finger, fieldorder.protocol_fingerprint(7, "sha256:y"))
        with self.assertRaises(ValueError):
            fieldorder.protocol_object(True, "sha256:x")
        with self.assertRaises(ValueError):
            fieldorder.protocol_object("0", "sha256:x")

    def test_inventory_digest_binds_the_spec_table(self):
        proto = fieldorder.protocol_object(0, "sha256:x")
        expect = fieldorder.digest_text(
            structured._canonical_json(list(fieldorder.FIELD_ORDER_SPECS))
        )
        self.assertEqual(proto["inventory_digest"], expect)
        self.assertEqual(
            proto["inventory_digest"],
            fieldorder.protocol_object(9, "sha256:other")["inventory_digest"],
        )


class TestParsing(unittest.TestCase):
    def test_object_document(self):
        parsed = fieldorder.parse_document('{"name": "Ada Lovelace", "age": 36}')
        self.assertEqual(parsed["kind"], "object")
        self.assertEqual(parsed["field_values"], {"age": "36", "name": '"Ada Lovelace"'})
        self.assertEqual(
            parsed["document_digest"],
            fieldorder.digest_text('{"age": 36, "name": "Ada Lovelace"}'),
        )
        self.assertFalse(parsed["duplicate_keys"])

    def test_duplicate_keys_flagged_last_wins(self):
        parsed = fieldorder.parse_document('{"a": 1, "a": 2}')
        self.assertTrue(parsed["duplicate_keys"])
        self.assertEqual(parsed["field_values"], {"a": "2"})
        self.assertEqual(parsed["document_digest"], fieldorder.digest_text('{"a": 2}'))

    def test_nonfinite_numbers_become_none(self):
        # F2: a non-finite number has no canonical form; its
        # value is the None sentinel and the document digest
        # is withheld.
        for text in ('{"a": NaN}', '{"a": Infinity}', '{"a": -Infinity}', '{"a": 1e999}'):
            parsed = fieldorder.parse_document(text)
            self.assertEqual(parsed["kind"], "object")
            self.assertIsNone(parsed["field_values"]["a"])
            self.assertIsNone(parsed["document_digest"])

    def test_null_is_the_string_null(self):
        # F2: a produced JSON null is the canonical string
        # "null", never the None sentinel.
        parsed = fieldorder.parse_document('{"a": null}')
        self.assertEqual(parsed["field_values"]["a"], "null")
        self.assertIsNotNone(parsed["document_digest"])

    def test_non_object_and_unparseable(self):
        self.assertEqual(fieldorder.parse_document("[1, 2]")["kind"], "non_object")
        self.assertEqual(fieldorder.parse_document("42")["kind"], "non_object")
        self.assertEqual(fieldorder.parse_document("no json")["kind"], "unparseable")
        self.assertEqual(fieldorder.parse_document("")["kind"], "unparseable")
        self.assertEqual(fieldorder.parse_document(None)["kind"], "unparseable")

    def test_huge_integer_has_no_canonical_form(self):
        parsed = fieldorder.parse_document('{"a": ' + "9" * 5000 + "}")
        self.assertEqual(parsed["kind"], "object")
        self.assertIsNone(parsed["field_values"]["a"])
        self.assertIsNone(parsed["document_digest"])

    def test_deep_nesting_never_crashes(self):
        # F10: hostile text yields a kind, never an exception.
        parsed = fieldorder.parse_document("[" * 4000 + "]" * 4000)
        self.assertIn(parsed["kind"], ("unparseable", "non_object"))


class TestDerived(unittest.TestCase):
    def test_field_correct_and_expected_digest(self):
        spec = fieldorder.spec_by_id("extract-person")
        digest = fieldorder.expected_value_digest(spec, "name")
        self.assertEqual(digest, fieldorder.digest_text(structured._canonical_json("Ada Lovelace")))
        derived = fieldorder.record_derived(spec, _record(spec, "canonical", _good(spec)))
        self.assertTrue(derived["adherent"])
        self.assertTrue(derived["answer_correct"])
        self.assertTrue(derived["fields"]["name"])
        self.assertTrue(derived["fields"]["age"])

    def test_schema_verdict_ignores_duplicate_keys(self):
        # F7 + the frozen CHECKING rule: the schema verdict
        # is the flat schema check over the reconstructed
        # document and nothing else. A duplicate-keys record
        # whose last-wins document satisfies the schema is
        # schema_ok; duplication bites through adherence
        # only, and the last-wins values are what is scored.
        spec = fieldorder.spec_by_id("extract-person")
        dup = _record(spec, "canonical", '{"name": "A", "name": "Ada Lovelace", "age": 36}')
        self.assertTrue(dup["duplicate_keys"])
        derived = fieldorder.record_derived(spec, dup)
        self.assertTrue(derived["schema_ok"])
        self.assertFalse(derived["adherent"])
        self.assertTrue(derived["fields"]["name"])

    def test_schema_ok_variants(self):
        spec = fieldorder.spec_by_id("extract-person")
        self.assertFalse(fieldorder.schema_ok(spec, "object", {"name": '"X"'}))  # missing field
        self.assertFalse(fieldorder.schema_ok(spec, "non_object", {}))
        record = _record(spec, "canonical", _good(spec))
        record["field_values"]["extra"] = "1"
        self.assertFalse(fieldorder.record_derived(spec, record)["schema_ok"])
        record = _record(spec, "canonical", '{"name": "Ada Lovelace", "age": true}')
        derived = fieldorder.record_derived(spec, record)
        self.assertFalse(derived["schema_ok"])  # bool is not an integer
        self.assertTrue(derived["fields"]["name"])

    def test_none_value_fails_schema_before_reconstruction(self):
        spec = fieldorder.spec_by_id("extract-product")
        record = _record(spec, "canonical", _good(spec))
        field = spec["answer_field"]
        record["field_values"][field] = None
        record["document_digest"] = None
        derived = fieldorder.record_derived(spec, record)
        self.assertFalse(derived["schema_ok"])
        self.assertFalse(derived["fields"][field])

    def test_whole_document_accuracy_rejects_extra_keys(self):
        # F15: whole-document accuracy is every spec field
        # correct AND no extra keys in field_values.
        spec = fieldorder.spec_by_id("extract-product")
        document = dict(spec["answer"])
        document["extra"] = 1
        record = _record(spec, "canonical", structured._canonical_json(document))
        self.assertFalse(fieldorder.record_derived(spec, record)["whole_doc_correct"])
        clean = fieldorder.record_derived(spec, _record(spec, "canonical", _good(spec)))
        self.assertTrue(clean["whole_doc_correct"])


class TestAggregate(unittest.TestCase):
    @staticmethod
    def _records(behaviour):
        return [
            behaviour(spec, arm)
            for spec in fieldorder.FIELD_ORDER_SPECS
            for arm in fieldorder.ARM_KEYS
        ]

    def test_perfect_model(self):
        aggregate = fieldorder.aggregate_candidate(
            self._records(lambda s, a: _record(s, a, _good(s)))
        )
        self.assertEqual(aggregate["n_generated"], 60)
        self.assertEqual(aggregate["n_live"], 60)
        # F3: the aggregation itself carries the count the
        # reporting layer prints.
        self.assertEqual(aggregate["n_distinct_prompts"], 37)
        self.assertEqual(aggregate["decision"]["n_paired"], 10)
        self.assertEqual(aggregate["decision"]["verdict"], "insufficient")
        self.assertEqual(aggregate["pairing"]["first_rate"], 1.0)
        self.assertEqual(aggregate["pairing"]["delta"], 0.0)
        self.assertEqual(aggregate["answer_by_arm"]["canonical"]["rate"], 1.0)
        self.assertEqual(aggregate["whole_doc_by_arm"]["reversed"]["rate"], 1.0)

    def test_first_only_model_is_falsified(self):
        def behaviour(spec, arm):
            if arm == "answer_first":
                return _record(spec, arm, _good(spec))
            return _record(spec, arm, _corrupted(spec, arm))

        aggregate = fieldorder.aggregate_candidate(self._records(behaviour))
        decision = aggregate["decision"]
        self.assertEqual(decision["verdict"], "falsified")
        self.assertEqual(decision["direction"], "first")
        self.assertEqual(decision["n_first_only"], 10)
        self.assertEqual(decision["sign_p"], 0.001953125)
        self.assertEqual(aggregate["extraction_decision"]["n_paired"], 6)

    def test_sign_test_values(self):
        # The design's printed values, recomputed.
        self.assertEqual(fieldorder.sign_test_p(6, 0), 0.03125)
        self.assertEqual(fieldorder.sign_test_p(8, 1), 0.0390625)
        self.assertEqual(fieldorder.sign_test_p(9, 1), 0.021484375)
        self.assertEqual(fieldorder.sign_test_p(5, 1), 0.21875)
        self.assertEqual(fieldorder.sign_test_p(10, 0), 0.001953125)
        self.assertEqual(fieldorder.sign_test_p(8, 2), 0.109375)
        self.assertEqual(fieldorder.sign_test_p(0, 0), 1.0)

    def test_five_unanimous_is_insufficient(self):
        # Below six discordant specs the verdict is withheld
        # as insufficient -- never a defaulted acquittal.
        five = {
            "extract-person",
            "extract-city",
            "extract-book",
            "extract-event",
            "extract-product",
        }

        def behaviour(spec, arm):
            if arm == "answer_last" and spec["id"] in five:
                return _record(spec, arm, _corrupted(spec, arm))
            return _record(spec, arm, _good(spec))

        aggregate = fieldorder.aggregate_candidate(self._records(behaviour))
        self.assertEqual(aggregate["decision"]["n_discordant"], 5)
        self.assertEqual(aggregate["decision"]["verdict"], "insufficient")

    def test_duplicate_records_are_excluded_from_pairing(self):
        # F7: pairing excludes any spec with a duplicate-keys
        # record in either endpoint arm, counted as its own
        # attrition step; schema rates are untouched.
        targets = {"extract-person", "route-complaint", "extract-vehicle"}
        records = self._records(lambda s, a: _record(s, a, _good(s)))
        for record in records:
            if record["spec_id"] in targets and record["arm"] == "answer_last":
                record["duplicate_keys"] = True
        aggregate = fieldorder.aggregate_candidate(records)
        self.assertEqual(aggregate["pairing"]["n_duplicate_excluded"], 3)
        self.assertEqual(aggregate["pairing"]["n_paired"], 7)
        self.assertEqual(aggregate["n_duplicate_keys"], 3)
        self.assertEqual(aggregate["by_arm"]["answer_last"]["schema_rate"], 1.0)
        self.assertLess(aggregate["by_arm"]["answer_last"]["adherent_rate"], 1.0)

    def test_zero_live_records_withhold_everything(self):
        # NB6 shape at the aggregate level: no live records ->
        # rates None, decision insufficient, n_paired 0, and
        # every readout key still present beside the detail.
        records = self._records(lambda s, a: _record(s, a, _good(s)))
        for record in records:
            record["detail"] = "arm_evaluation_failed"
            record["boundary_found"] = False
            record["answer_digest"] = None
            record["document_kind"] = ""
            record["document_digest"] = None
            record["field_values"] = {}
        aggregate = fieldorder.aggregate_candidate(records)
        self.assertEqual(aggregate["n_live"], 0)
        self.assertEqual(aggregate["n_detail_screened"], 60)
        self.assertIsNone(aggregate["answer_by_arm"]["canonical"]["rate"])
        self.assertEqual(aggregate["decision"]["verdict"], "insufficient")
        self.assertEqual(aggregate["decision"]["n_paired"], 0)
        self.assertIsNone(aggregate["pairing"]["first_rate"])
        for key in (
            "by_arm",
            "per_field",
            "answer_by_arm",
            "whole_doc_by_arm",
            "pairing",
            "decision",
            "extraction_decision",
            "field_count_split",
        ):
            self.assertIn(key, aggregate)

    def test_sensitivity_blocks_present(self):
        # F5/F6/F13: the decision never travels alone -- the
        # extraction-only sensitivity, the ungated
        # discordance, and the field-count split always ship.
        aggregate = fieldorder.aggregate_candidate(
            self._records(lambda s, a: _record(s, a, _good(s)))
        )
        self.assertIn("sign_p", aggregate["extraction_decision"])
        self.assertEqual(aggregate["pairing"]["ungated"]["n_specs"], 10)
        split = aggregate["field_count_split"]
        self.assertEqual(split["two"]["n_paired"] + split["three_plus"]["n_paired"], 10)


class TestStage(unittest.TestCase):
    def test_perfect_backend_validates(self):
        payload = _payload()
        self.assertTrue(payload["available"])
        self.assertEqual(payload["n_candidates"], 2)
        self.assertEqual(payload["n_specs"], 15)
        entry = payload["candidates"]["fo-a"]
        self.assertEqual(len(entry["records"]), 60)
        self.assertEqual(entry["n_live"], 60)
        self.assertEqual(entry["n_distinct_prompts"], 37)
        self.assertEqual(entry["decision"]["n_paired"], 10)
        self.assertEqual(entry["decision"]["verdict"], "insufficient")
        self.assertTrue(_valid_field_order(payload))

    def test_last_wrong_backend_is_falsified(self):
        payload = _payload(_FieldOrderBackend("last_wrong"))
        self.assertTrue(_valid_field_order(payload))
        for entry in payload["candidates"].values():
            self.assertEqual(entry["decision"]["verdict"], "falsified")
            self.assertEqual(entry["decision"]["direction"], "first")
            self.assertEqual(entry["decision"]["n_first_only"], 10)

    def test_wrong_backend(self):
        payload = _payload(_FieldOrderBackend("wrong"))
        self.assertTrue(_valid_field_order(payload))
        for entry in payload["candidates"].values():
            self.assertEqual(entry["n_live"], 60)
            self.assertEqual(entry["by_arm"]["canonical"]["parse_rate"], 0.0)
            self.assertEqual(entry["answer_by_arm"]["canonical"]["rate"], 0.0)

    def test_dup_backend_schema_ok_but_not_adherent(self):
        # F7 end to end: every record carries duplicate keys,
        # every last-wins document is the correct answer --
        # schema rates stay 1.0, adherence is 0, and pairing
        # excludes everything as duplicate attrition.
        payload = _payload(_FieldOrderBackend("dup"))
        self.assertTrue(_valid_field_order(payload))
        for entry in payload["candidates"].values():
            self.assertEqual(entry["n_duplicate_keys"], 60)
            self.assertEqual(entry["by_arm"]["canonical"]["schema_rate"], 1.0)
            self.assertEqual(entry["by_arm"]["canonical"]["adherent_rate"], 0.0)
            self.assertEqual(entry["pairing"]["n_duplicate_excluded"], 10)
            self.assertEqual(entry["pairing"]["n_paired"], 0)

    def test_nan_backend_yields_none_values(self):
        payload = _payload(_FieldOrderBackend("nan"))
        self.assertTrue(_valid_field_order(payload))
        entry = payload["candidates"]["fo-a"]
        none_records = [
            r
            for r in entry["records"]
            if any(value is None for value in r["field_values"].values())
        ]
        self.assertTrue(none_records)
        for record in none_records:
            self.assertIsNone(record["document_digest"])
            self.assertFalse(
                fieldorder.record_derived(fieldorder.spec_by_id(record["spec_id"]), record)[
                    "schema_ok"
                ]
            )

    def test_no_boundary_backend(self):
        payload = _payload(_FieldOrderBackend("noboundary"))
        self.assertTrue(_valid_field_order(payload))
        for entry in payload["candidates"].values():
            self.assertEqual(entry["n_no_boundary"], 60)
            self.assertEqual(entry["n_live"], 0)

    def test_explode_backend_degrades_and_validates(self):
        payload = _payload(_FieldOrderBackend("explode"))
        self.assertTrue(payload["available"])
        self.assertTrue(_valid_field_order(payload))
        for entry in payload["candidates"].values():
            self.assertEqual(entry["n_detail_screened"], 60)
            self.assertEqual(entry["n_live"], 0)

    def test_field_values_cap_degrades(self):
        # F8: a live record whose committed field values sum
        # over the 4,096-char cap degrades individually to
        # arm_evaluation_failed and is counted as detail
        # screened; the payload never carries it.
        payload = _payload(_FieldOrderBackend("verbose"))
        self.assertTrue(_valid_field_order(payload))
        for entry in payload["candidates"].values():
            self.assertEqual(entry["n_live"], 0)
            self.assertEqual(entry["n_detail_screened"], 60)

    def test_cancel_unwinds_stage(self):
        backend = _FieldOrderBackend()
        event = threading.Event()
        event.set()
        stage = experiments.fieldorder_stage(
            backend,
            FORMATS,
            boundary=tuple(backend.encode("endmark")),
            max_context=8192,
            seed=0,
            cancel=event,
        )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "cancelled")

    def test_record_budget(self):
        backend = _FieldOrderBackend()
        formats = tuple(
            Format(name=f"fo-{i}", bos="", user_prefix="u:", assistant_prefix="a:", turn_end="</s>")
            for i in range(25)
        )
        stage = experiments.fieldorder_stage(
            backend,
            formats,
            boundary=tuple(backend.encode("endmark")),
            max_context=8192,
            seed=0,
        )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "field_order_record_budget_exceeded")

    def test_boundary_in_prompts_screens_per_spec_not_stage(self):
        # Design section 6: boundary-text containment screens the
        # affected spec's records; it does not fail the stage (the
        # stage-level screen is the special-token audit only).
        backend = _FieldOrderBackend()
        backend.vocab.update({"the": 1, "endmark": 1})
        backend.rev.update({1: "the"})
        stage = experiments.fieldorder_stage(
            backend,
            FORMATS[:1],
            boundary=tuple(backend.encode("the")),
            max_context=8192,
            seed=0,
        )
        self.assertTrue(stage["available"])
        payload = experiments.fieldorder_analysis(stage, ("fo-a",))
        entry = payload["candidates"]["fo-a"]
        expected_screened = 4 * sum(
            1
            for spec in fieldorder.FIELD_ORDER_SPECS
            if any("the" in fieldorder.arm_prompt(spec, arm) for arm in fieldorder.ARM_KEYS)
        )
        self.assertGreater(expected_screened, 0)
        self.assertEqual(entry["n_boundary_screened"], expected_screened)
        self.assertTrue(_valid_field_order(payload))

    def test_per_spec_screen(self):
        backend = _FieldOrderBackend()
        # The candidate's boundary token decodes to "Rotor" (the
        # endmark token shares its id), so the boundary text
        # contaminates exactly the extract-product prompts while
        # generations still terminate at that token.
        backend.vocab.update({"Rotor": 1, "endmark": 1})
        backend.rev.update({1: "Rotor"})
        stage = experiments.fieldorder_stage(
            backend,
            FORMATS[:1],
            boundary=tuple(backend.encode("Rotor")),
            max_context=8192,
            seed=0,
        )
        payload = experiments.fieldorder_analysis(stage, ("fo-a",))
        entry = payload["candidates"]["fo-a"]
        self.assertEqual(entry["n_boundary_screened"], 4)
        self.assertEqual(entry["n_live"], 56)
        self.assertTrue(_valid_field_order(payload))

    def test_context_overflow_skeleton(self):
        backend = _FieldOrderBackend()
        stage = experiments.fieldorder_stage(
            backend,
            FORMATS[:1],
            boundary=tuple(backend.encode("endmark")),
            max_context=8,
            seed=0,
        )
        payload = experiments.fieldorder_analysis(stage, ("fo-a",))
        entry = payload["candidates"]["fo-a"]
        self.assertEqual(entry["n_live"], 0)
        self.assertEqual(entry["n_detail_screened"], 60)
        self.assertTrue(_valid_field_order(payload))

    def test_duplicate_format_names_rejected(self):
        backend = _FieldOrderBackend()
        with self.assertRaises(ValueError):
            experiments.fieldorder_stage(
                backend,
                (FORMATS[0], FORMATS[0]),
                boundary=tuple(backend.encode("endmark")),
                max_context=8192,
                seed=0,
            )


class TestDiff(unittest.TestCase):
    def test_self_diff_unchanged(self):
        payload = _payload()
        result = diff_field_order(payload, payload)
        self.assertEqual(result["available"], "both")
        self.assertFalse(result["changed"])
        self.assertEqual(result["candidate_drift"], [])

    def test_verdict_drift_detected(self):
        old = _payload()
        new = _payload(_FieldOrderBackend("last_wrong"))
        result = diff_field_order(old, new)
        self.assertTrue(result["changed"])
        metrics = {(d["candidate"], d["metric"]) for d in result["candidate_drift"]}
        self.assertIn(("fo-a", "verdict"), metrics)
        self.assertIn(("fo-a", "n_first_only"), metrics)
        self.assertIn(("fo-a", "answer_rate_answer_last"), metrics)

    def test_one_sided(self):
        payload = _payload()
        self.assertEqual(diff_field_order(None, payload)["available"], "new")
        self.assertEqual(diff_field_order(payload, None)["available"], "old")
        self.assertEqual(diff_field_order(None, None)["available"], "neither")
        self.assertFalse(diff_field_order(None, payload)["changed"])

    def test_seed_drift_not_comparable(self):
        old = _payload(seed=0)
        new = _payload(seed=1)
        result = diff_field_order(old, new)
        self.assertTrue(result["protocol_drift"])
        self.assertFalse(result["comparable"])

    def test_tamper_battery(self):
        payload = _payload()

        def tampered(mutator):
            clone = copy.deepcopy(payload)
            mutator(clone)
            return clone

        def agg_rate(p):
            p["candidates"]["fo-a"]["answer_by_arm"]["canonical"]["rate"] = 0.5

        def verdict(p):
            p["candidates"]["fo-a"]["decision"]["verdict"] = "falsified"

        def field_value(p):
            p["candidates"]["fo-a"]["records"][0]["field_values"]["age"] = "99"

        def document_kind(p):
            p["candidates"]["fo-a"]["records"][0]["document_kind"] = "non_object"

        def dup_flag(p):
            p["candidates"]["fo-a"]["records"][0]["duplicate_keys"] = True

        def doc_digest(p):
            p["candidates"]["fo-a"]["records"][0]["document_digest"] = "0" * 64

        def content_digest(p):
            p["candidates"]["fo-a"]["records"][0]["content_digest"] = "0" * 64

        def noncanonical_value(p):
            p["candidates"]["fo-a"]["records"][0]["field_values"]["age"] = " 36"

        def seed_type(p):
            p["protocol"]["seed"] = True

        def extra_record_key(p):
            p["candidates"]["fo-a"]["records"][0]["extra"] = 1

        def deep_nesting(p):
            p["candidates"]["fo-a"]["records"][0]["field_values"]["age"] = "[" * 20000

        def nan_string(p):
            p["candidates"]["fo-a"]["records"][0]["field_values"]["age"] = "NaN"

        def record_swap(p):
            records = p["candidates"]["fo-a"]["records"]
            records[0], records[1] = records[1], records[0]

        def distinct_count(p):
            p["candidates"]["fo-a"]["n_distinct_prompts"] = 60

        for mutator in (
            agg_rate,
            verdict,
            field_value,
            document_kind,
            dup_flag,
            doc_digest,
            content_digest,
            noncanonical_value,
            seed_type,
            extra_record_key,
            deep_nesting,
            nan_string,
            record_swap,
            distinct_count,
        ):
            with self.subTest(mutator=mutator.__name__):
                self.assertFalse(_valid_field_order(tampered(mutator)))

    def test_unavailable_candidate_entry(self):
        payload = _payload()
        payload["candidates"]["fo-b"] = {
            "status": "unavailable",
            "detail": "cancelled",
            "records": [],
        }
        self.assertTrue(_valid_field_order(payload))
        payload["candidates"]["fo-b"]["records"] = [{}]
        self.assertFalse(_valid_field_order(payload))


class TestReportingAndCli(unittest.TestCase):
    def test_section_renders_decision_and_limits(self):
        payload = _payload(formats=FORMATS[:1])
        html = reporting.render_html({"field_order": payload})
        self.assertIn("Schema field order", html)
        self.assertIn("insufficient", html)
        self.assertIn("Minimum discordant specs: 6", html)
        self.assertIn("capped at 4096 characters in total", html)
        self.assertIn("37 distinct prompts", html)

    def test_section_renders_falsified(self):
        payload = _payload(_FieldOrderBackend("last_wrong"), formats=FORMATS[:1])
        html = reporting.render_html({"field_order": payload})
        self.assertIn("field-order invariance falsified", html)

    def test_section_absent_without_payload(self):
        self.assertNotIn('id="field_order"', reporting.render_html({}))
        self.assertNotIn(
            'id="field_order"', reporting.render_html({"field_order": {"available": False}})
        )

    def test_section_escapes_candidate_names(self):
        payload = _payload(formats=FORMATS[:1])
        payload["candidates"]["<script>x</script>"] = payload["candidates"].pop("fo-a")
        payload["n_candidates"] = 1
        html = reporting.render_html({"field_order": payload})
        self.assertNotIn("<script>x</script>", html)

    def test_section_na_rates(self):
        payload = _payload()
        sparse = copy.deepcopy(payload)
        entry = sparse["candidates"]["fo-a"]
        entry["by_arm"] = {}
        entry["per_field"] = {}
        entry["answer_by_arm"] = {}
        entry["whole_doc_by_arm"] = {}
        entry["pairing"] = {}
        entry["decision"] = {}
        entry["extraction_decision"] = {}
        entry["field_count_split"] = {}
        html = reporting.render_html({"field_order": sparse})
        self.assertIn("Schema field order", html)

    def test_cli_helper_attaches_payload(self):
        backend = _FieldOrderBackend()
        result: dict = {}
        cli._run_fieldorder_probe(
            backend,
            FORMATS,
            result,
            boundary=tuple(backend.encode("endmark")),
            boundary_map=None,
            max_context=8192,
            seed=0,
            cancel=None,
        )
        self.assertTrue(result["field_order"]["available"])
        self.assertEqual(result["field_order"]["protocol"]["max_new_tokens"], 64)
        self.assertTrue(_valid_field_order(result["field_order"]))

    def test_cli_helper_never_raises(self):
        class _ExplodingFingerprint(_FieldOrderBackend):
            @property
            def fingerprint(self):  # noqa: ANN202 -- test fake
                raise RuntimeError("no fingerprint")

        result: dict = {}
        cli._run_fieldorder_probe(
            _ExplodingFingerprint(),
            FORMATS,
            result,
            boundary=(1,),
            boundary_map=None,
            max_context=8192,
            seed=0,
            cancel=None,
        )
        self.assertFalse(result["field_order"]["available"])
        self.assertEqual(result["field_order"]["detail"], "field_order_unavailable")

        def boom(*args, **kwargs):
            raise RuntimeError("stage exploded")

        original = experiments.fieldorder_stage
        experiments.fieldorder_stage = boom
        try:
            result = {}
            cli._run_fieldorder_probe(
                _FieldOrderBackend(),
                FORMATS,
                result,
                boundary=(1,),
                boundary_map=None,
                max_context=8192,
                seed=0,
                cancel=None,
            )
        finally:
            experiments.fieldorder_stage = original
        self.assertFalse(result["field_order"]["available"])
        self.assertEqual(result["field_order"]["detail"], "field_order_unavailable")


if __name__ == "__main__":
    unittest.main()
