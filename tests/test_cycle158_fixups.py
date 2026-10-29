"""Cycle 158: wave-4 red-team fix-ups -- regression tests and mutation gates.

The wave-4 adversarial round (redteam_a.md / redteam_b.md) found six
surviving findings in the cycle-157 tree. This module pins the fixes:

- B1 [MODERATE]: same-ids verdict flips were counted as whitespace
  fragility, contradicting the documented honest split. The fix excludes
  them from every fragility surface (``fragile``, ``n_fragile_mutations``,
  ``fragile_candidates``, ``whitespace_robust``), keeping
  ``n_same_ids_verdict_flips`` as its own determinism anomaly counter.
- F1 [MODERATE]: NaN/±Infinity classified ``adherent`` for ``number``
  fields (and accepted in specs/fixtures). The fix rejects non-finite
  floats in ``_field_value_ok``'s number branch, in enum members, in
  ``validate_structured_specs`` (via check_schema), and in
  ``_canonical_json`` (``allow_nan=False``).
- F2 [LOW]: ``structured_stage`` absorbed malformed schema dicts instead
  of raising (its docstring promised ValueError). The fix validates
  every schema with ``validate_schema_def`` up front.
- F3 [LOW]: ``layout_options`` silently shortened layouts on duplicate
  options. The fix rejects duplicates fail-closed.
- F4 [LOW]: ``minimum``/``maximum`` were accepted (and carried) on
  non-numeric field types, making unsatisfiable schemas. The fix
  rejects bounds on non-integer/number fields.
- F5 [LOW]: ``_valid_structured`` accepted self-contradictory cells
  (adherent_rate > parse_rate). The fix enforces the producer
  invariants documented by ``candidate_adherence``.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

from model_protocol_lab.core import Fixture, Format, Generation, Message
from model_protocol_lab.diffreports import _valid_structured
from model_protocol_lab.experiments import (
    native_search_input_ids,
    native_search_outcomes,
    structured_stage,
    ws_canonicalization,
    ws_stage,
)
from model_protocol_lab.position import layout_options
from model_protocol_lab.structured import (
    _canonical_json,
    classify_response,
    validate_field_def,
    validate_structured_specs,
)
from model_protocol_lab.wscanonical import aggregate_mutation_records

BOUNDARY = (124,)  # ord("|")


def _format(name="chatml"):
    return Format(
        name=name,
        bos="<s>",
        user_prefix="<|user|>\n",
        assistant_prefix="<|assistant|>\n",
        turn_end="<|end|>\n",
    )


def _fixture(fid, expected="WORD"):
    return Fixture(
        id=fid,
        messages=(Message(role="user", content="Say the word."),),
        expected=expected,
        group="g",
    )


def _number_schema(**field_kwargs):
    field = {"type": "number"}
    field.update(field_kwargs)
    return {
        "type": "object",
        "required": ["v"],
        "properties": {"v": field},
        "additionalProperties": False,
    }


# ---------------------------------------------------------------------------
# B1: same-ids verdict flips never count as fragility
# ---------------------------------------------------------------------------


class SameIdsFragilityExclusionTests(unittest.TestCase):
    @staticmethod
    def _record(mutation, *, n=9, divergent=0, flips=0, same=0):
        return {
            "mutation": mutation,
            "n_fixtures": n,
            "n_evaluable": n,
            "n_ids_divergent": divergent,
            "n_verdict_flips": flips,
            "n_same_ids_verdict_flips": same,
        }

    def test_same_ids_only_flips_not_fragile(self):
        # B's exact reproduction: flips on identical ids must not be
        # fragility.
        agg = aggregate_mutation_records([self._record("trailing_space", flips=3, same=3)])
        cell = agg["per_mutation"]["trailing_space"]
        self.assertFalse(cell["fragile"])
        self.assertEqual(cell["n_fragile_flips"], 0)
        self.assertEqual(cell["n_verdict_flips"], 3)  # total still recorded
        self.assertEqual(cell["n_same_ids_verdict_flips"], 3)
        self.assertEqual(agg["n_fragile_mutations"], 0)
        self.assertTrue(agg["whitespace_robust"])

    def test_mixed_flips_fragile_only_on_divergent_ids(self):
        agg = aggregate_mutation_records([self._record("trailing_space", flips=3, same=2)])
        cell = agg["per_mutation"]["trailing_space"]
        self.assertTrue(cell["fragile"])
        self.assertEqual(cell["n_fragile_flips"], 1)
        self.assertEqual(agg["n_fragile_mutations"], 1)
        self.assertFalse(agg["whitespace_robust"])

    def test_divergent_ids_without_flips_not_fragile(self):
        agg = aggregate_mutation_records([self._record("trailing_space", divergent=4)])
        cell = agg["per_mutation"]["trailing_space"]
        self.assertFalse(cell["fragile"])
        self.assertEqual(agg["n_fragile_mutations"], 0)
        # Ids diverged: not whitespace-robust even with zero flips.
        self.assertFalse(agg["whitespace_robust"])

    def test_same_ids_exceeding_total_rejected(self):
        # n_same_ids_verdict_flips is a subset of n_verdict_flips; a
        # record violating that is malformed (fail-closed, like the
        # other cross-field checks).
        with self.assertRaises(ValueError):
            aggregate_mutation_records([self._record("trailing_space", flips=2, same=3)])


class FlipFlopBackend:
    """ord codec; alternates answers across calls: same ids, flipped verdict."""

    fingerprint = "scripted-ws-flipflop-c158"

    calls = 0

    @staticmethod
    def encode(text):
        return tuple(map(ord, text))

    @staticmethod
    def decode(token_ids):
        return "".join(map(chr, token_ids))

    def generate(self, input_ids, *, max_new_tokens, seed, cancel):
        type(self).calls += 1
        answer = "WORD" if type(self).calls % 2 == 0 else "NOPE"
        out = tuple(map(ord, (answer + "|").ljust(max_new_tokens, ".")[:max_new_tokens]))
        return Generation(token_ids=out, reason="limit")


class WsCanonicalizationSameIdsEndToEndTests(unittest.TestCase):
    @staticmethod
    def _run():
        from model_protocol_lab.core import evaluate

        names = ("chatml",)
        formats = tuple(_format(name) for name in names)
        cases = (_fixture("f0"),)
        reports = {
            name: evaluate(
                FlipFlopBackend(), fmt, cases, boundary=BOUNDARY, max_new_tokens=16, seed=7
            )
            for name, fmt in zip(names, formats, strict=True)
        }
        entries = []
        for name, report in reports.items():
            entries.append(
                {
                    "format": {"name": name},
                    "observations": [
                        {
                            "fixture": obs.fixture,
                            "status": obs.status,
                            "exact_match": obs.exact_match,
                            "boundary_found": obs.boundary_found,
                            "input_ids": list(obs.input_ids),
                        }
                        for obs in report.observations
                    ],
                }
            )
        FlipFlopBackend.calls = 0
        stage = ws_stage(
            FlipFlopBackend(), formats, cases, boundary=BOUNDARY, max_new_tokens=16, seed=7
        )
        return ws_canonicalization(
            native_search_outcomes(entries, names),
            native_search_input_ids(entries, names),
            stage,
            names,
        )

    def test_end_to_end_same_ids_flips_excluded_from_fragility(self):
        payload = self._run()
        self.assertTrue(payload["available"])
        cand = payload["per_candidate"]["chatml"]
        for mutation, cell in cand["per_mutation"].items():
            if type(cell) is not dict or not cell.get("evaluable"):
                continue
            # The honest-split invariant, per mutation, end to end:
            self.assertEqual(
                cell["fragile"],
                (cell["n_verdict_flips"] - cell["n_same_ids_verdict_flips"]) > 0,
                f"mutation {mutation}",
            )
        # If every recorded flip is a same-ids flip, the candidate is
        # not fragile and not whitespace-fragile.
        agg = cand["aggregate"]
        per = agg["per_mutation"]
        if all(
            entry["n_verdict_flips"] == entry["n_same_ids_verdict_flips"] for entry in per.values()
        ):
            self.assertEqual(agg["n_fragile_mutations"], 0)
            self.assertNotIn("chatml", payload["fragile_candidates"])


# ---------------------------------------------------------------------------
# F1: non-finite floats are never valid JSON numbers
# ---------------------------------------------------------------------------


class NonFiniteNumberTests(unittest.TestCase):
    def test_nan_not_adherent_minimum_only(self):
        result = classify_response('{"v": NaN}', _number_schema(minimum=0))
        self.assertEqual(result["parse"], "parse_ok")
        self.assertEqual(result["outcome"], "schema_failed")

    def test_infinity_not_adherent_unbounded(self):
        result = classify_response('{"v": Infinity}', _number_schema())
        self.assertEqual(result["outcome"], "schema_failed")

    def test_negative_infinity_not_adherent(self):
        result = classify_response('{"v": -Infinity}', _number_schema(maximum=10))
        self.assertEqual(result["outcome"], "schema_failed")

    def test_nan_not_adherent_with_enum(self):
        schema = _number_schema()
        schema["properties"]["v"]["enum"] = [1.5, 2.5]
        result = classify_response('{"v": NaN}', schema)
        self.assertEqual(result["outcome"], "schema_failed")

    def test_finite_float_still_adherent(self):
        result = classify_response('{"v": 1.5}', _number_schema(minimum=0))
        self.assertEqual(result["outcome"], "adherent")

    def test_finite_int_still_adherent(self):
        result = classify_response('{"v": 3}', _number_schema(minimum=0, maximum=5))
        self.assertEqual(result["outcome"], "adherent")

    def test_spec_with_nan_answer_rejected(self):
        spec = {
            "id": "nan-answer",
            "question": "Pick a number.",
            "context": "The answer is not a number.",
            "shape": "extraction",
            "schema": _number_schema(),
            "answer": {"v": float("nan")},
        }
        with self.assertRaises(ValueError):
            validate_structured_specs((spec,))

    def test_spec_with_infinity_answer_rejected(self):
        spec = {
            "id": "inf-answer",
            "question": "Pick a number.",
            "context": "The answer is infinite.",
            "shape": "extraction",
            "schema": _number_schema(),
            "answer": {"v": float("inf")},
        }
        with self.assertRaises(ValueError):
            validate_structured_specs((spec,))

    def test_canonical_json_rejects_non_finite(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                _canonical_json({"v": bad})

    def test_enum_nan_member_rejected(self):
        with self.assertRaises(ValueError):
            validate_field_def({"type": "number", "enum": [1.0, float("nan")]}, where="p")

    def test_enum_infinity_member_rejected(self):
        with self.assertRaises(ValueError):
            validate_field_def({"type": "number", "enum": [float("inf")]}, where="p")


# ---------------------------------------------------------------------------
# F2: structured_stage validates schemas up front
# ---------------------------------------------------------------------------


class StructuredStageSchemaValidationTests(unittest.TestCase):
    @staticmethod
    def _schemas():
        return {
            "s/x": {
                "type": "object",
                "required": ["a"],
                "properties": {"a": {"type": "string"}},
                "additionalProperties": False,
            }
        }

    def test_malformed_schema_dict_raises(self):
        fmt = _format()
        fixture = _fixture("s/x", expected='{"a": "x"}')
        bad = dict(self._schemas())
        bad["s/x"] = {"type": "bogus"}
        with self.assertRaises(ValueError):
            structured_stage(
                FlipFlopBackend(),
                (fmt,),
                (fixture,),
                schemas=bad,
                boundary=BOUNDARY,
            )

    def test_malformed_field_def_raises(self):
        fmt = _format()
        fixture = _fixture("s/x", expected='{"a": "x"}')
        bad = dict(self._schemas())
        bad["s/x"] = {
            "type": "object",
            "required": ["a"],
            "properties": {"a": {"type": "bogus-type"}},
            "additionalProperties": False,
        }
        with self.assertRaises(ValueError):
            structured_stage(
                FlipFlopBackend(),
                (fmt,),
                (fixture,),
                schemas=bad,
                boundary=BOUNDARY,
            )

    def test_valid_schemas_do_not_raise(self):
        fmt = _format()
        fixture = _fixture("s/x", expected='{"a": "x"}')
        # Must not raise; the cell degrades or classifies, never explodes.
        structured_stage(
            FlipFlopBackend(),
            (fmt,),
            (fixture,),
            schemas=self._schemas(),
            boundary=BOUNDARY,
        )


# ---------------------------------------------------------------------------
# F3: layout_options rejects duplicate options
# ---------------------------------------------------------------------------


class LayoutOptionsDuplicateTests(unittest.TestCase):
    def test_duplicate_options_raise(self):
        with self.assertRaises(ValueError):
            layout_options(("a", "a", "b"), "a", condition="correct_first", seed=0)

    def test_all_identical_options_raise(self):
        with self.assertRaises(ValueError):
            layout_options(("a", "a"), "a", condition="correct_last", seed=0)

    def test_distinct_options_still_work(self):
        layout = layout_options(("a", "b", "c"), "b", condition="correct_first", seed=0)
        self.assertEqual(len(layout), 3)
        self.assertEqual(layout[0], "b")
        self.assertEqual(sorted(layout), ["a", "b", "c"])


# ---------------------------------------------------------------------------
# F4: bounds only on integer/number fields
# ---------------------------------------------------------------------------


class FieldDefBoundsTests(unittest.TestCase):
    def test_minimum_on_string_rejected(self):
        with self.assertRaises(ValueError):
            validate_field_def({"type": "string", "minimum": 3}, where="p")

    def test_maximum_on_boolean_rejected(self):
        with self.assertRaises(ValueError):
            validate_field_def({"type": "boolean", "maximum": 1}, where="p")

    def test_bounds_on_null_rejected(self):
        with self.assertRaises(ValueError):
            validate_field_def({"type": "null", "minimum": 0, "maximum": 1}, where="p")

    def test_bounds_on_number_accepted(self):
        normalized = validate_field_def(
            {"type": "number", "minimum": 0.5, "maximum": 2.5}, where="p"
        )
        self.assertEqual(normalized["minimum"], 0.5)
        self.assertEqual(normalized["maximum"], 2.5)

    def test_bounds_on_integer_accepted(self):
        normalized = validate_field_def({"type": "integer", "minimum": 1, "maximum": 5}, where="p")
        self.assertEqual(normalized["minimum"], 1)

    def test_bounds_not_carried_on_string(self):
        # The docstring promises the normalized dict carries only
        # applicable keys; a rejected bound cannot leak through.
        with self.assertRaises(ValueError):
            validate_field_def({"type": "string", "maximum": 9}, where="p")


# ---------------------------------------------------------------------------
# F5: _valid_structured enforces producer invariants
# ---------------------------------------------------------------------------


def _structured_cell(parse_rate, schema_rate, adherent_rate):
    return {
        "available": True,
        "n_fixtures": 9,
        "per_candidate": {
            "x": {
                "parse_rate": parse_rate,
                "schema_rate": schema_rate,
                "adherent_rate": adherent_rate,
            }
        },
        "weakest_candidates": ["x"],
        "adherence_gap": 0.0,
    }


class ValidStructuredCrossFieldTests(unittest.TestCase):
    def test_contradictory_adherent_rate_rejected(self):
        self.assertFalse(_valid_structured(_structured_cell(0.0, 1.0, 1.0)))

    def test_contradictory_schema_rate_rejected(self):
        self.assertFalse(_valid_structured(_structured_cell(0.5, 0.9, 0.1)))

    def test_consistent_cell_accepted(self):
        self.assertTrue(_valid_structured(_structured_cell(1.0, 0.5, 0.5)))

    def test_none_rates_accepted(self):
        self.assertTrue(_valid_structured(_structured_cell(None, None, None)))

    def test_mixed_none_and_consistent_accepted(self):
        self.assertTrue(_valid_structured(_structured_cell(0.8, None, 0.2)))


# ---------------------------------------------------------------------------
# Mutation gates (tempdir-module; the real tree is never rewritten)
# ---------------------------------------------------------------------------


class FixupMutationGateTests(unittest.TestCase):
    """Each gate reverts one fix; the regression scenario must catch it."""

    def _run_gate(self, module_name, name, old_text, new_text, scenario):
        mod = __import__(f"model_protocol_lab.{module_name}", fromlist=["x"])
        source = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertEqual(source.count(old_text), 1, f"mutant {name}: anchor not unique")
        mutated = source.replace(old_text, new_text)
        with tempfile.TemporaryDirectory(prefix="c158_mut_", dir="/tmp") as tmpdir:
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

    # -- B1 gates ----------------------------------------------------

    @staticmethod
    def _b1_scenario(mod):
        record = {
            "mutation": "trailing_space",
            "n_fixtures": 9,
            "n_evaluable": 9,
            "n_ids_divergent": 0,
            "n_verdict_flips": 3,
            "n_same_ids_verdict_flips": 3,
        }
        agg = mod.aggregate_mutation_records([record])
        assert agg["per_mutation"]["trailing_space"]["fragile"] is False
        assert agg["n_fragile_mutations"] == 0
        assert agg["whitespace_robust"] is True

    def test_mutant_b1_same_ids_counted_as_fragile_caught(self):
        self._run_gate(
            "wscanonical",
            "b1_fragile_counts_same_ids",
            '"fragile": n_fragile_flips > 0,',
            '"fragile": counts["n_verdict_flips"] > 0,  # MUTANT',
            self._b1_scenario,
        )

    def test_mutant_b1_dropped_subset_validation_caught(self):
        self._run_gate(
            "wscanonical",
            "b1_dropped_subset_check",
            '            raise ValueError("n_same_ids_verdict_flips '
            'cannot exceed n_verdict_flips")\n',
            "            pass  # MUTANT\n",
            self._b1_subset_scenario,
        )

    @staticmethod
    def _b1_subset_scenario(mod):
        record = {
            "mutation": "trailing_space",
            "n_fixtures": 9,
            "n_evaluable": 9,
            "n_ids_divergent": 0,
            "n_verdict_flips": 2,
            "n_same_ids_verdict_flips": 3,
        }
        try:
            mod.aggregate_mutation_records([record])
        except ValueError:
            return
        raise AssertionError("hostile same-ids > flips record was accepted")

    # -- F1 gate -----------------------------------------------------

    @staticmethod
    def _f1_scenario(mod):
        result = mod.classify_response('{"v": NaN}', _number_schema(minimum=0))
        assert result["outcome"] == "schema_failed", result

    def test_mutant_f1_nan_accepted_as_number_caught(self):
        self._run_gate(
            "structured",
            "f1_nan_adherent",
            "            type(value) in (int, float) "
            "and not isinstance(value, bool) and _is_finite_number(value)\n",
            "            type(value) in (int, float) "
            "and not isinstance(value, bool) and True  # MUTANT\n",
            self._f1_scenario,
        )

    # -- F2 gate -----------------------------------------------------

    @staticmethod
    def _f2_scenario(mod):
        fmt = _format()
        fixture = _fixture("s/x", expected='{"a": "x"}')
        try:
            mod.structured_stage(
                FlipFlopBackend(),
                (fmt,),
                (fixture,),
                schemas={"s/x": {"type": "bogus"}},
                boundary=BOUNDARY,
            )
        except ValueError:
            return
        raise AssertionError("malformed schema dict was absorbed by the stage")

    def test_mutant_f2_stage_skips_schema_validation_caught(self):
        self._run_gate(
            "experiments",
            "f2_skipped_schema_validation",
            "            validate_schema_def(schema)\n",
            "            None  # MUTANT\n",
            self._f2_scenario,
        )

    # -- F3 gate -----------------------------------------------------

    @staticmethod
    def _f3_scenario(mod):
        try:
            mod.layout_options(("a", "a", "b"), "a", condition="correct_first", seed=0)
        except ValueError:
            return
        raise AssertionError("duplicate options were silently shortened")

    def test_mutant_f3_duplicates_accepted_caught(self):
        self._run_gate(
            "position",
            "f3_duplicates_accepted",
            "    if len(set(options)) != len(options):\n"
            '        raise ValueError("options must be distinct strings")\n',
            "    pass  # MUTANT\n",
            self._f3_scenario,
        )

    # -- F4 gate -----------------------------------------------------

    @staticmethod
    def _f4_scenario(mod):
        try:
            mod.validate_field_def({"type": "string", "minimum": 3}, where="p")
        except ValueError:
            return
        raise AssertionError("bound on a string field was accepted")

    def test_mutant_f4_bounds_on_string_accepted_caught(self):
        self._run_gate(
            "structured",
            "f4_bounds_on_string",
            '            if field_type not in ("integer", "number"):\n'
            '                raise ValueError(f"{where}: {bound} is only '
            'meaningful for integer/number fields")\n',
            "            pass  # MUTANT\n",
            self._f4_scenario,
        )

    # -- F5 gate -----------------------------------------------------

    @staticmethod
    def _f5_scenario(mod):
        payload = _structured_cell(0.0, 1.0, 1.0)
        assert mod._valid_structured(payload) is False

    def test_mutant_f5_contradictory_cell_accepted_caught(self):
        self._run_gate(
            "diffreports",
            "f5_contradictory_cell",
            "                if other is not None and other > parse_rate:\n"
            "                    return False\n",
            "                pass  # MUTANT\n",
            self._f5_scenario,
        )


if __name__ == "__main__":
    unittest.main()
