"""W2 Push 2 — exact-type admission (audit RR-04, SA-04, SA-05).

A JSON bool is never a number and a number is never a bool, at any
nesting depth, in any validator that recomputes a stored value
(RR-04); a finite integer is finite without float conversion
(SA-04); enum membership follows the JSON value model of JSON
Schema section 4.2.2 — bool matches only bool, numbers match
numerically across int/float, everything else matches its own type
(SA-05). Valid controls pin the equivalences that must survive:
int/float numeric equality (1 == 1.0) everywhere.
"""

from __future__ import annotations

import copy
import sys
import unittest
from typing import Any

sys.path.insert(0, "tests")

from model_protocol_lab import diffreports, structured


def _wrap(key: str, payload: Any) -> dict[str, Any]:
    return {
        "decision": "no_candidate",
        "selected_candidate": None,
        "search": [],
        "held_out": None,
        key: payload,
    }


class RecomputedEqualTests(unittest.TestCase):
    """RR-04 at the helper every recompute consumer shares."""

    def test_bool_never_equals_a_number(self) -> None:
        """Pre-fix the fallthrough returned bool(stored == expected):
        False validated as the recomputed rate 0.0."""
        eq = diffreports._recomputed_equal
        self.assertFalse(eq(False, 0.0))
        self.assertFalse(eq(True, 1.0))
        self.assertFalse(eq(False, 0))
        self.assertFalse(eq(0, False))
        self.assertFalse(eq(1, True))

    def test_type_strictness_is_symmetric(self) -> None:
        """Same-type equality holds; cross-spelling is rejected both ways.

        The recompute validators' contract is exactness against this
        package's own producers (cycle 227 pins a stored 0.0 against
        a recomputed int 0 as a rejection); the pre-fix fallthrough
        admitted the stored-int/expected-float direction only.
        """
        eq = diffreports._recomputed_equal
        self.assertTrue(eq(1, 1))
        self.assertTrue(eq(1.0, 1.0))
        self.assertTrue(eq(2.5, 2.5))
        self.assertFalse(eq(1, 1.0))
        self.assertFalse(eq(0.0, 0))
        self.assertTrue(eq(True, True))
        self.assertFalse(eq(True, False))

    def test_nested_bool_conflation_rejected(self) -> None:
        """The gate applies at every nesting depth."""
        eq = diffreports._recomputed_equal
        self.assertFalse(eq({"a": False}, {"a": 0.0}))
        self.assertFalse(eq([False], [0.0]))
        self.assertFalse(eq({"a": [{"b": True}]}, {"a": [{"b": 1.0}]}))
        self.assertTrue(eq({"a": [1, {"b": 2.0}]}, {"a": [1, {"b": 2.0}]}))


class PrecisionExactTypeTests(unittest.TestCase):
    """RR-04 end to end on the precision section (cycle-198 payload)."""

    def _diff(self, mutate: Any) -> dict[str, Any]:
        from test_cycle198_precision import _valid_payload

        valid = _valid_payload()
        mutated = copy.deepcopy(valid)
        mutate(mutated)
        return diffreports.diff_reports(_wrap("precision", valid), _wrap("precision", mutated))

    def test_divergence_rate_bool_is_not_zero(self) -> None:
        """divergence_rate=False: pre-fix the pair compared identical
        (False == the recomputed 0.0)."""
        diff = self._diff(lambda p: p["candidates"]["beta"].__setitem__("divergence_rate", False))
        self.assertNotEqual(diff["precision"]["available"], "both")
        self.assertFalse(diff["summary"]["identical"])

    def test_logprob_mean_delta_bool_is_not_zero(self) -> None:
        """mean_delta=False inside the logprob block: same conflation,
        one nesting level down."""

        def mutate(p: dict[str, Any]) -> None:
            p["candidates"]["beta"]["logprob"]["mean_delta"] = False

        diff = self._diff(mutate)
        self.assertNotEqual(diff["precision"]["available"], "both")

    def test_diverged_string_rejected(self) -> None:
        """diverged="yes": the identity probes alone admitted it when
        the both-arms flag was False."""

        def mutate(p: dict[str, Any]) -> None:
            p["records"][0]["diverged"] = "yes"

        diff = self._diff(mutate)
        self.assertNotEqual(diff["precision"]["available"], "both")

    def test_diverged_list_rejected(self) -> None:
        """diverged=[True] likewise."""

        def mutate(p: dict[str, Any]) -> None:
            p["records"][0]["diverged"] = [True]

        diff = self._diff(mutate)
        self.assertNotEqual(diff["precision"]["available"], "both")

    def test_pass_flag_int_rejected_at_recompute(self) -> None:
        """fp32_passed=1: `1 in (True, False, None)` admitted it.
        The recompute itself must refuse the payload."""

        from test_cycle198_precision import _valid_payload

        mutated = copy.deepcopy(_valid_payload())
        name = sorted(mutated["candidates"])[0]
        mutated["candidates"][name]["fp32_passed"] = 1
        self.assertIsNone(diffreports._precision_recompute(mutated))
        diff = diffreports.diff_reports(
            _wrap("precision", _valid_payload()), _wrap("precision", mutated)
        )
        self.assertNotEqual(diff["precision"]["available"], "both")

    def test_valid_pair_still_identical(self) -> None:
        """Control: the genuine payload pair is untouched."""
        from test_cycle198_precision import _valid_payload

        valid = _valid_payload()
        diff = diffreports.diff_reports(
            _wrap("precision", valid), _wrap("precision", copy.deepcopy(valid))
        )
        self.assertEqual(diff["precision"]["available"], "both")
        self.assertTrue(diff["summary"]["identical"])


class LeadingCellExactTypeTests(unittest.TestCase):
    """RR-04 sibling gate: _leading_cell_consistent's boundary_text_match
    used the same loose membership as the precision pass flags."""

    _DIGEST = "a" * 64

    def _cell(self, match: Any) -> dict[str, Any]:
        return {
            "verdict": True,
            "boundary_found": True,
            "boundary_text_match": match,
            "input_tokens": 5,
            "answer_digest": self._DIGEST,
        }

    def test_int_match_rejected(self) -> None:
        """boundary_text_match=1: pre-fix admitted (1 in (True, ...))."""
        self.assertFalse(diffreports._leading_cell_consistent(self._cell(1), self._DIGEST))

    def test_genuine_matches_still_consistent(self) -> None:
        """Controls: True and None are the field's genuine values."""
        self.assertTrue(diffreports._leading_cell_consistent(self._cell(True), self._DIGEST))
        self.assertTrue(diffreports._leading_cell_consistent(self._cell(None), self._DIGEST))


class StructuredFiniteNumberTests(unittest.TestCase):
    """SA-04: a 401-digit integer is a finite number; asking must not
    raise OverflowError through a float conversion."""

    def test_huge_int_is_finite(self) -> None:
        """Pre-fix: OverflowError from math.isnan(10**400)."""
        self.assertTrue(structured._is_finite_number(10**400))
        self.assertTrue(structured._is_finite_number(-(10**400)))

    def test_huge_int_field_value_ok(self) -> None:
        """The per-field check over a number field accepts it."""
        self.assertTrue(structured._field_value_ok(10**400, {"type": "number"}))

    def test_nonfinite_floats_still_rejected(self) -> None:
        """Controls: NaN and the infinities are not JSON numbers."""
        self.assertFalse(structured._is_finite_number(float("nan")))
        self.assertFalse(structured._is_finite_number(float("inf")))
        self.assertFalse(structured._field_value_ok(float("inf"), {"type": "number"}))

    def test_small_values_unchanged(self) -> None:
        """Controls: ordinary numbers behave as before."""
        self.assertTrue(structured._is_finite_number(5))
        self.assertTrue(structured._is_finite_number(2.5))
        self.assertFalse(structured._is_finite_number(True))
        self.assertFalse(structured._is_finite_number("5"))


class StructuredEnumValueModelTests(unittest.TestCase):
    """SA-05: enum membership under the JSON value model."""

    def test_int_field_enum_true_rejects_one(self) -> None:
        """Pre-fix: 1 in [True] was True under Python equality."""
        field = {"type": "integer", "enum": [True]}
        self.assertFalse(structured._field_value_ok(1, field))

    def test_bool_field_enum_one_rejects_true(self) -> None:
        """Pre-fix: True in [1] was True."""
        field = {"type": "boolean", "enum": [1]}
        self.assertFalse(structured._field_value_ok(True, field))

    def test_number_field_enum_int_accepts_float_spelling(self) -> None:
        """Control: 1.0 is the number 1 (JSON Schema 4.2.2:
        mathematical equality across int/float)."""
        field = {"type": "number", "enum": [1]}
        self.assertTrue(structured._field_value_ok(1.0, field))

    def test_int_field_enum_int(self) -> None:
        """Controls: same-type membership is unchanged."""
        field = {"type": "integer", "enum": [1]}
        self.assertTrue(structured._field_value_ok(1, field))
        self.assertFalse(structured._field_value_ok(2, field))

    def test_bool_field_enum_bool(self) -> None:
        """Control: a bool matches a bool member."""
        field = {"type": "boolean", "enum": [True]}
        self.assertTrue(structured._field_value_ok(True, field))

    def test_string_and_null_enums(self) -> None:
        """Controls: strings and null match only their own kind."""
        self.assertTrue(structured._field_value_ok("a", {"type": "string", "enum": ["a"]}))
        self.assertFalse(structured._field_value_ok("a", {"type": "string", "enum": [1]}))
        self.assertTrue(structured._field_value_ok(None, {"type": "null", "enum": [None]}))
        self.assertFalse(structured._field_value_ok(None, {"type": "null", "enum": [False]}))


if __name__ == "__main__":
    unittest.main()
