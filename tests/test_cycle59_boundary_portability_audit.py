"""Cycle 59 adversarial audit: boundary portability checkers.

TARGET
------
``model_protocol_lab.core.check_boundary_portability`` (core.py:413) -- the
probe that decides whether a boundary marker's token IDs port from a primary
tokenizer to a secondary one -- plus its generalization
``model_protocol_lab.core.template_tokenizer_compatibility`` (core.py:462).

METHOD
------
1. The documented/observed contract of each function is written down from the
   source (see CONTRACT below) and hand-pinned with exact-dict assertions.
2. An INDEPENDENT oracle for ``check_boundary_portability`` is hand-rolled
   from the docstring -- it does not reuse the implementation or its helpers
   (not even ``_valid_token_ids``) -- and differential-fuzzed against the
   implementation: 266 randomized cases over boundary texts, per-backend ID
   maps and encode behaviours (identical / divergent / empty / malformed /
   raising), plus 60 randomized invalid-argument cases. Every randomized case
   asserts full output equality (or a matching ValueError) between oracle and
   implementation.
3. Non-tautology: the oracle is derived from the docstring, not the code, and
   every hand-pinned test asserts the exact observable contract (full dict
   equality, not derived quantities). The mutation gate (step 4) proves each
   behaviour class discriminates -- every mutant is caught by at least one
   failing test.
4. MUTATION GATE: the suite runs against an unmutated copy in /tmp, then
   against copies carrying 6 semantic mutants (flipped comparison, inverted
   validity guard, dropped invalid-output guard, swapped detail strings,
   fragments misclassification, degraded-membership narrowing). All 6 are
   caught. Copies are removed afterwards; repo sources were never edited.

FINDINGS
--------
CLEAN: no genuine bugs found in either function. ``check_boundary_portability``
is a total function over its documented domain: every input class maps to
exactly one of the four fixed detail codes, guards reject every malformed
argument with ValueError, and the encode probe degrades to a report instead
of raising.

One contract gray zone (pinned, not a bug): a secondary encode returning an
EMPTY tuple is classified "divergent_ids" (secondary_ids=[]) rather than
"secondary_encode_invalid". The sibling probe
``template_tokenizer_compatibility`` treats an empty encode as "encode_invalid".
Both behaviours are now pinned by tests so any future change is deliberate.

CONTRACT (check_boundary_portability)
-------------------------------------
Inputs: boundary_text (type-exact nonempty str), primary_ids (type-exact
nonempty tuple of non-negative ints; bools rejected), secondary_encode
(callable).
- Guard failure -> ValueError.
- secondary_encode(boundary_text) raises Exception -> portable False,
  secondary_ids None, detail "secondary_encode_failed".
- Result not a tuple of non-negative ints (empty tuple allowed) ->
  portable False, secondary_ids None, detail "secondary_encode_invalid".
- Else portable = (secondary_ids == primary_ids) (order-sensitive tuple
  equality); detail "identical" / "divergent_ids"; both id lists echoed as
  fresh lists.
"""

import random
import unittest

from model_protocol_lab.core import (
    check_boundary_portability,
    template_tokenizer_compatibility,
)

# ---------------------------------------------------------------------------
# Independent oracle for check_boundary_portability.
# Hand-rolled from the docstring; shares no code with the implementation.
# ---------------------------------------------------------------------------

_VALID_DETAILS = frozenset(
    {"identical", "divergent_ids", "secondary_encode_failed", "secondary_encode_invalid"}
)


def _oracle_check_boundary_portability(boundary_text, primary_ids, secondary_encode):
    """Independent reimplementation of the documented contract."""
    if type(boundary_text) is not str or len(boundary_text) == 0:
        raise ValueError("boundary_text must be a nonempty string")
    if type(primary_ids) is not tuple or len(primary_ids) == 0:
        raise ValueError("primary_ids must be a nonempty tuple of token IDs")
    for token in primary_ids:
        if type(token) is not int or token < 0:
            raise ValueError("primary_ids must be a nonempty tuple of token IDs")
    if not callable(secondary_encode):
        raise ValueError("secondary_encode must be callable")
    try:
        secondary_ids = secondary_encode(boundary_text)
    except Exception:  # noqa: BLE001 -- oracle mirrors the probe's failure boundary
        return {
            "portable": False,
            "primary_ids": list(primary_ids),
            "secondary_ids": None,
            "detail": "secondary_encode_failed",
        }
    valid = type(secondary_ids) is tuple and all(
        type(token) is int and token >= 0 for token in secondary_ids
    )
    if not valid:
        return {
            "portable": False,
            "primary_ids": list(primary_ids),
            "secondary_ids": None,
            "detail": "secondary_encode_invalid",
        }
    portable = secondary_ids == primary_ids
    return {
        "portable": portable,
        "primary_ids": list(primary_ids),
        "secondary_ids": list(secondary_ids),
        "detail": "identical" if portable else "divergent_ids",
    }


# ---------------------------------------------------------------------------
# Encode-behaviour factories for the differential fuzz.
# Each takes (primary_ids, rng) and returns a secondary_encode callable.
# ---------------------------------------------------------------------------


def _beh_identical(primary, rng):
    return lambda text: primary


def _beh_divergent_values(primary, rng):
    shifted = tuple(token + 1 + rng.randrange(3) for token in primary)
    return lambda text: shifted


def _beh_divergent_length(primary, rng):
    if len(primary) > 1 and rng.random() < 0.5:
        out = primary[:-1]
    else:
        out = primary + (rng.choice((0, 1, 999)),)
    return lambda text: out


def _beh_divergent_order(primary, rng):
    out = primary[::-1]
    return lambda text: out


def _beh_empty_tuple(primary, rng):
    return lambda text: ()


def _beh_list_copy(primary, rng):
    return lambda text: list(primary)


def _beh_none(primary, rng):
    return lambda text: None


def _beh_string(primary, rng):
    return lambda text: "not-a-tuple"


def _beh_float_ids(primary, rng):
    return lambda text: tuple(float(token) for token in primary)


def _beh_bool_ids(primary, rng):
    return lambda text: (True,) + tuple(primary[1:])


def _beh_negative_ids(primary, rng):
    return lambda text: (-1,) + tuple(primary[1:])


def _beh_nested_tuple(primary, rng):
    return lambda text: (tuple(primary),)


def _beh_generator(primary, rng):
    return lambda text: (token for token in primary)


def _beh_range(primary, rng):
    return lambda text: range(len(primary))


def _beh_raises_runtime(primary, rng):
    def encode(text):
        raise RuntimeError("synthetic tokenizer blowup")

    return encode


def _beh_raises_value(primary, rng):
    def encode(text):
        raise ValueError("synthetic bad input")

    return encode


def _beh_raises_key(primary, rng):
    def encode(text):
        raise KeyError("synthetic missing vocab")

    return encode


def _beh_huge_ids(primary, rng):
    return lambda text: (2**62, 10**30, 0)


def _beh_single_zero(primary, rng):
    return lambda text: (0,)


_BEHAVIOURS = (
    ("identical", _beh_identical),
    ("divergent_values", _beh_divergent_values),
    ("divergent_length", _beh_divergent_length),
    ("divergent_order", _beh_divergent_order),
    ("empty_tuple", _beh_empty_tuple),
    ("list_copy", _beh_list_copy),
    ("none", _beh_none),
    ("string", _beh_string),
    ("float_ids", _beh_float_ids),
    ("bool_ids", _beh_bool_ids),
    ("negative_ids", _beh_negative_ids),
    ("nested_tuple", _beh_nested_tuple),
    ("generator", _beh_generator),
    ("range", _beh_range),
    ("raises_runtime", _beh_raises_runtime),
    ("raises_value", _beh_raises_value),
    ("raises_key", _beh_raises_key),
    ("huge_ids", _beh_huge_ids),
    ("single_zero", _beh_single_zero),
)

_BOUNDARY_TEXTS = (
    "a",
    "ab",
    "hello world",
    "<|eot_id|>",
    "<|im_start|>user",
    "x" * 512,
    "héllo→wörld",
    "emoji 🎉 boundary 🔚",
    "\t\n leading/trailing \r\n",
    " ",
    "Ünïcodé boundary: \u00e9\u4e2d\u6587",
    "a" * 3 + "\U0001f600" + "b",
)

_ID_POOL = (0, 1, 2, 5, 42, 100, 32000, 99999, 2**31 - 1)


def _random_primary_ids(rng):
    return tuple(rng.choice(_ID_POOL) for _ in range(rng.randint(1, 6)))


class CheckBoundaryPortabilityContractTests(unittest.TestCase):
    """Hand-pinned exact-contract tests: every branch, no derived quantities."""

    def test_identical_encodings_are_portable_exact_dict(self):
        outcome = check_boundary_portability("ab", (1, 2), lambda text: (1, 2))
        self.assertEqual(
            outcome,
            {
                "portable": True,
                "primary_ids": [1, 2],
                "secondary_ids": [1, 2],
                "detail": "identical",
            },
        )

    def test_divergent_encodings_flagged_exact_dict(self):
        outcome = check_boundary_portability("ab", (1, 2), lambda text: (7,))
        self.assertEqual(
            outcome,
            {
                "portable": False,
                "primary_ids": [1, 2],
                "secondary_ids": [7],
                "detail": "divergent_ids",
            },
        )

    def test_order_sensitive_comparison(self):
        # Same multiset, different order -> divergent (tuple equality).
        outcome = check_boundary_portability("ab", (1, 2), lambda text: (2, 1))
        self.assertFalse(outcome["portable"])
        self.assertEqual(outcome["detail"], "divergent_ids")
        self.assertEqual(outcome["secondary_ids"], [2, 1])

    def test_single_zero_id_is_portable(self):
        outcome = check_boundary_portability("x", (0,), lambda text: (0,))
        self.assertTrue(outcome["portable"])
        self.assertEqual(outcome["detail"], "identical")

    def test_encode_failure_variants_reported_not_raised(self):
        def custom(text):
            raise _CustomEncodeError("synthetic")

        raisers = [
            lambda text: (_ for _ in ()).throw(RuntimeError("boom")),
            lambda text: (_ for _ in ()).throw(ValueError("bad")),
            lambda text: (_ for _ in ()).throw(KeyError("missing")),
            custom,
        ]
        for i, raiser in enumerate(raisers):
            with self.subTest(i=i):
                outcome = check_boundary_portability("ab", (1, 2), raiser)
                self.assertEqual(
                    outcome,
                    {
                        "portable": False,
                        "primary_ids": [1, 2],
                        "secondary_ids": None,
                        "detail": "secondary_encode_failed",
                    },
                )

    def test_invalid_secondary_shapes_reported(self):
        invalid_outputs = [
            [1, 2],  # list, not tuple
            None,
            "12",
            b"\x01\x02",
            (1.0, 2.0),  # floats would compare equal to (1, 2) -- validity wins
            (True, 2),  # bools would compare equal to (1, 2) -- validity wins
            (False,),
            (-1, 2),
            ("1", "2"),
            ((1, 2),),  # nested
            (x for x in (1, 2)),  # generator
            range(2),
            {"ids": (1, 2)},
            (None,),
        ]
        for i, bad in enumerate(invalid_outputs):
            with self.subTest(i=i, bad=repr(bad)[:40]):
                outcome = check_boundary_portability("ab", (1, 2), lambda text, _bad=bad: _bad)
                self.assertFalse(outcome["portable"])
                self.assertIsNone(outcome["secondary_ids"])
                self.assertEqual(outcome["detail"], "secondary_encode_invalid")
                self.assertEqual(outcome["primary_ids"], [1, 2])

    def test_validity_gate_precedes_equality(self):
        # (True, 2) == (1, 2) and (1.0, 2.0) == (1, 2) in Python; the validity
        # gate must reject them before the comparison can claim "identical".
        for bad in ((True, 2), (1.0, 2.0)):
            with self.subTest(bad=bad):
                outcome = check_boundary_portability("ab", (1, 2), lambda text, _bad=bad: _bad)
                self.assertFalse(outcome["portable"])
                self.assertEqual(outcome["detail"], "secondary_encode_invalid")

    def test_empty_secondary_tuple_is_divergent_not_invalid(self):
        # Contract gray zone, pinned: an empty (but type-valid) secondary
        # encoding is "divergent_ids" with secondary_ids=[], not
        # "secondary_encode_invalid".
        outcome = check_boundary_portability("ab", (1, 2), lambda text: ())
        self.assertEqual(
            outcome,
            {
                "portable": False,
                "primary_ids": [1, 2],
                "secondary_ids": [],
                "detail": "divergent_ids",
            },
        )

    def test_boundary_text_forwarded_verbatim(self):
        seen = []

        def encode(text):
            seen.append(text)
            return (9,)

        for boundary in ("ab", "héllo→🎉", "x" * 1000, "\t "):
            with self.subTest(boundary=boundary[:12]):
                seen.clear()
                check_boundary_portability(boundary, (9,), encode)
                self.assertEqual(seen, [boundary])

    def test_output_lists_are_fresh_copies(self):
        primary = (1, 2)
        secondary = (1, 2)
        outcome = check_boundary_portability("ab", primary, lambda text: secondary)
        self.assertIsNot(outcome["primary_ids"], primary)
        self.assertEqual(outcome["primary_ids"], [1, 2])
        outcome["primary_ids"].append(999)
        outcome["secondary_ids"].append(999)
        # Inputs untouched; a second call is unaffected by the mutation.
        self.assertEqual(primary, (1, 2))
        self.assertEqual(secondary, (1, 2))
        again = check_boundary_portability("ab", primary, lambda text: secondary)
        self.assertEqual(again["primary_ids"], [1, 2])
        self.assertEqual(again["secondary_ids"], [1, 2])

    def test_detail_vocabulary_is_fixed(self):
        cases = [
            ("ab", (1,), lambda text: (1,), "identical"),
            ("ab", (1,), lambda text: (2,), "divergent_ids"),
            ("ab", (1,), lambda text: (), "divergent_ids"),
            ("ab", (1,), lambda text: [1], "secondary_encode_invalid"),
        ]

        def boom(text):
            raise RuntimeError("x")

        cases.append(("ab", (1,), boom, "secondary_encode_failed"))
        for i, (boundary, primary, encode, detail) in enumerate(cases):
            with self.subTest(i=i):
                outcome = check_boundary_portability(boundary, primary, encode)
                self.assertIn(outcome["detail"], _VALID_DETAILS)
                self.assertEqual(outcome["detail"], detail)
                self.assertEqual(
                    set(outcome), {"portable", "primary_ids", "secondary_ids", "detail"}
                )
                self.assertIs(type(outcome["portable"]), bool)

    def test_guard_rejects_malformed_boundary_text(self):
        bad_texts = ["", b"ab", None, 123, 4.5, ["ab"], ("ab",), bytearray(b"ab"), _MyStr("ab")]
        for i, bad in enumerate(bad_texts):
            with self.subTest(i=i, bad=repr(bad)[:30]):
                with self.assertRaises(ValueError):
                    check_boundary_portability(bad, (1,), lambda text: (1,))
                with self.assertRaises(ValueError):
                    _oracle_check_boundary_portability(bad, (1,), lambda text: (1,))

    def test_guard_rejects_malformed_primary_ids(self):
        bad_primaries = [
            (),
            [],
            [1, 2],
            "12",
            None,
            7,
            (1, -1),
            (-5,),
            (1.0, 2),
            ("1",),
            (True,),
            (False,),
            (None,),
            ((1,),),
            _MyTuple((1, 2)),
            (1, _MyInt(2)),
        ]
        for i, bad in enumerate(bad_primaries):
            with self.subTest(i=i, bad=repr(bad)[:30]):
                with self.assertRaises(ValueError):
                    check_boundary_portability("ab", bad, lambda text: (1,))
                with self.assertRaises(ValueError):
                    _oracle_check_boundary_portability("ab", bad, lambda text: (1,))

    def test_guard_rejects_non_callable_encode(self):
        for bad in (None, "encode", 42, [1], (1,)):
            with self.subTest(bad=repr(bad)):
                with self.assertRaises(ValueError):
                    check_boundary_portability("ab", (1,), bad)
                with self.assertRaises(ValueError):
                    _oracle_check_boundary_portability("ab", (1,), bad)

    def test_exact_type_semantics_not_isinstance(self):
        # type(x) is T, not isinstance: subclasses are rejected everywhere.
        with self.assertRaises(ValueError):
            check_boundary_portability(_MyStr("ab"), (1,), lambda text: (1,))
        with self.assertRaises(ValueError):
            check_boundary_portability("ab", _MyTuple((1,)), lambda text: (1,))
        with self.assertRaises(ValueError):
            check_boundary_portability("ab", (_MyInt(1),), lambda text: (1,))
        # ... but the values themselves are fine as plain types.
        outcome = check_boundary_portability("ab", (1,), lambda text: (1,))
        self.assertTrue(outcome["portable"])


class _CustomEncodeError(Exception):
    pass


class _MyStr(str):
    pass


class _MyTuple(tuple):
    pass


class _MyInt(int):
    pass


class CheckBoundaryPortabilityDifferentialFuzzTests(unittest.TestCase):
    """Differential fuzz: independent oracle vs implementation, 240 cases."""

    def test_oracle_matches_implementation_randomized(self):
        rng = random.Random(0xC5959)  # noqa: S311 -- deterministic test seed
        case = 0
        for name, factory in _BEHAVIOURS:
            for _ in range(14):
                boundary = rng.choice(_BOUNDARY_TEXTS)
                primary = _random_primary_ids(rng)
                encode = factory(primary, rng)
                with self.subTest(case=case, behaviour=name):
                    expected = _oracle_check_boundary_portability(boundary, primary, encode)
                    actual = check_boundary_portability(boundary, primary, encode)
                    self.assertEqual(actual, expected)
                    self.assertIn(actual["detail"], _VALID_DETAILS)
                    self.assertEqual(
                        set(actual), {"portable", "primary_ids", "secondary_ids", "detail"}
                    )
                case += 1
        self.assertEqual(case, 19 * 14)  # 266 randomized valid-input cases

    def test_oracle_matches_implementation_invalid_arguments(self):
        rng = random.Random(0xBEEF59)  # noqa: S311 -- deterministic test seed
        bad_texts = ["", b"ab", None, 123, ["ab"], _MyStr("ok")]
        bad_primaries = [(), [1], (1, -1), (1.5,), (True,), "12", None, _MyTuple((1,))]
        bad_encodes = [None, "encode", 42, [1]]
        for case in range(60):
            slot = rng.randrange(3)
            boundary = rng.choice(bad_texts) if slot == 0 else rng.choice(_BOUNDARY_TEXTS)
            primary = rng.choice(bad_primaries) if slot == 1 else _random_primary_ids(rng)
            encode = rng.choice(bad_encodes) if slot == 2 else (lambda text: (1,))
            with self.subTest(case=case, slot=slot):
                with self.assertRaises(ValueError):
                    _oracle_check_boundary_portability(boundary, primary, encode)
                with self.assertRaises(ValueError):
                    check_boundary_portability(boundary, primary, encode)

    def test_oracle_agrees_with_hand_pinned_golden_cases(self):
        # The oracle itself must reproduce the hand-pinned contract points;
        # otherwise the fuzz above would be vacuous.
        golden = [
            (("ab", (1, 2), lambda text: (1, 2)), {"portable": True, "detail": "identical"}),
            (("ab", (1, 2), lambda text: (2, 1)), {"portable": False, "detail": "divergent_ids"}),
            (("ab", (1, 2), lambda text: ()), {"portable": False, "detail": "divergent_ids"}),
            (
                ("ab", (1, 2), lambda text: [1]),
                {"portable": False, "detail": "secondary_encode_invalid"},
            ),
            (
                ("ab", (1, 2), lambda text: (True, 2)),
                {"portable": False, "detail": "secondary_encode_invalid"},
            ),
        ]

        def boom(text):
            raise RuntimeError("x")

        golden.append(
            (("ab", (1, 2), boom), {"portable": False, "detail": "secondary_encode_failed"})
        )
        for i, ((boundary, primary, encode), wanted) in enumerate(golden):
            with self.subTest(i=i):
                for fn in (_oracle_check_boundary_portability, check_boundary_portability):
                    outcome = fn(boundary, primary, encode)
                    self.assertEqual(outcome["portable"], wanted["portable"])
                    self.assertEqual(outcome["detail"], wanted["detail"])


class TemplateTokenizerCompatibilityEdgeTests(unittest.TestCase):
    """Adversarial edges for the generalization (core.py:462)."""

    def test_compatible_all_ok_exact_rows(self):
        result = template_tokenizer_compatibility(
            "llama3",
            ("<|eot_id|>", "<|begin_of_text|>"),
            {"<|eot_id|>": 128009, "<|begin_of_text|>": 128000},
            lambda text: {"<|eot_id|>": (128009,), "<|begin_of_text|>": (128000,)}[text],
        )
        self.assertEqual(result["format"], "llama3")
        self.assertEqual(result["verdict"], "compatible")
        self.assertEqual(result["reason"], "")
        self.assertTrue(
            result["detail"].startswith(
                "2/2 inventory tokens registered, atomic, and id-consistent"
            )
        )
        self.assertIn("predicts nothing", result["detail"])
        self.assertEqual(len(result["tokens"]), 2)
        row = result["tokens"][0]
        self.assertEqual(
            row,
            {
                "token": "<|eot_id|>",
                "registered": True,
                "registered_id": 128009,
                "single_token": True,
                "encoded_id": 128009,
                "id_match": True,
                "status": "ok",
            },
        )

    def test_unregistered_token_row_schema(self):
        result = template_tokenizer_compatibility("fmt", ("<|eot_id|>",), {}, lambda text: (9,))
        self.assertEqual(result["verdict"], "incompatible")
        self.assertEqual(result["reason"], "unregistered")
        self.assertIn("'<|eot_id|>'", result["detail"])
        row = result["tokens"][0]
        self.assertEqual(row["registered"], False)
        self.assertIsNone(row["registered_id"])
        self.assertTrue(row["single_token"])
        self.assertEqual(row["encoded_id"], 9)
        self.assertIsNone(row["id_match"])  # no registered id to compare against
        self.assertEqual(row["status"], "unregistered")

    def test_fragments_row_schema(self):
        result = template_tokenizer_compatibility("fmt", ("tok",), {"tok": 5}, lambda text: (5, 6))
        self.assertEqual(result["verdict"], "incompatible")
        self.assertEqual(result["reason"], "fragments")
        row = result["tokens"][0]
        self.assertFalse(row["single_token"])
        self.assertEqual(row["encoded_id"], 5)
        self.assertIsNone(row["id_match"])
        self.assertEqual(row["status"], "fragments")

    def test_id_mismatch_row_schema(self):
        result = template_tokenizer_compatibility("fmt", ("tok",), {"tok": 5}, lambda text: (7,))
        self.assertEqual(result["verdict"], "incompatible")
        self.assertEqual(result["reason"], "id_mismatch")
        row = result["tokens"][0]
        self.assertTrue(row["single_token"])
        self.assertEqual(row["encoded_id"], 7)
        self.assertFalse(row["id_match"])
        self.assertEqual(row["status"], "id_mismatch")

    def test_within_token_precedence_registered_then_atomic_then_id(self):
        # Registered but fragmenting, and the first fragment happens to equal
        # the registered id: still "fragments", never "ok".
        result = template_tokenizer_compatibility("fmt", ("tok",), {"tok": 5}, lambda text: (5, 6))
        self.assertEqual(result["tokens"][0]["status"], "fragments")
        # Unregistered and fragmenting: "unregistered" wins over "fragments".
        result = template_tokenizer_compatibility("fmt", ("tok",), {}, lambda text: (5, 6))
        self.assertEqual(result["tokens"][0]["status"], "unregistered")

    def test_first_failure_in_inventory_order(self):
        def encode(text):
            return {"bad1": (1, 2), "bad2": (9,), "good": (3,)}[text]

        result = template_tokenizer_compatibility(
            "fmt", ("bad1", "bad2", "good"), {"bad1": 1, "bad2": 5, "good": 3}, encode
        )
        self.assertEqual(result["verdict"], "incompatible")
        self.assertEqual(result["reason"], "fragments")  # bad1 first
        self.assertIn("'bad1'", result["detail"])
        result = template_tokenizer_compatibility(
            "fmt", ("bad2", "bad1", "good"), {"bad1": 1, "bad2": 5, "good": 3}, encode
        )
        self.assertEqual(result["reason"], "id_mismatch")  # bad2 first now
        self.assertIn("'bad2'", result["detail"])

    def test_unknown_empty_inventory_never_compatible(self):
        result = template_tokenizer_compatibility("fmt", (), {}, lambda text: (1,))
        self.assertEqual(result["verdict"], "unknown")
        self.assertEqual(result["reason"], "empty_inventory")
        self.assertIn("never", result["detail"])
        self.assertEqual(result["tokens"], [])

    def test_unknown_on_encode_failure_row_schema(self):
        def boom(text):
            raise RuntimeError("synthetic")

        result = template_tokenizer_compatibility("fmt", ("tok",), {"tok": 5}, boom)
        self.assertEqual(result["verdict"], "unknown")
        self.assertEqual(result["reason"], "encode_failed")
        self.assertIn("'tok'", result["detail"])
        row = result["tokens"][0]
        self.assertEqual(row["status"], "encode_failed")
        self.assertIsNone(row["single_token"])
        self.assertIsNone(row["encoded_id"])
        self.assertIsNone(row["id_match"])
        self.assertTrue(row["registered"])
        self.assertEqual(row["registered_id"], 5)

    def test_unknown_on_encode_invalid_variants(self):
        for i, bad in enumerate([[1], None, (), (1.5,), (-1,), (True,), "x"]):
            with self.subTest(i=i, bad=repr(bad)[:20]):
                result = template_tokenizer_compatibility(
                    "fmt", ("tok",), {"tok": 5}, lambda text, _bad=bad: _bad
                )
                self.assertEqual(result["verdict"], "unknown")
                self.assertEqual(result["reason"], "encode_invalid")
                self.assertEqual(result["tokens"][0]["status"], "encode_invalid")

    def test_degraded_takes_precedence_over_incompatible(self):
        # bad1 is unregistered (incompatible-worthy); bad2's encode raises.
        # The probe degraded -> "unknown", not "incompatible".
        def encode(text):
            if text == "bad2":
                raise RuntimeError("synthetic")
            return (1, 2)

        result = template_tokenizer_compatibility("fmt", ("bad1", "bad2"), {"bad2": 1}, encode)
        self.assertEqual(result["verdict"], "unknown")
        self.assertEqual(result["reason"], "encode_failed")
        # Reversed order: first degraded row in inventory order names the reason.
        result = template_tokenizer_compatibility("fmt", ("bad2", "bad1"), {"bad2": 1}, encode)
        self.assertEqual(result["verdict"], "unknown")
        self.assertEqual(result["reason"], "encode_failed")
        self.assertIn("'bad2'", result["detail"])

    def test_duplicate_inventory_tokens_get_duplicate_rows(self):
        result = template_tokenizer_compatibility(
            "fmt", ("tok", "tok"), {"tok": 5}, lambda text: (5,)
        )
        self.assertEqual(len(result["tokens"]), 2)
        self.assertEqual(result["tokens"][0]["status"], "ok")
        self.assertEqual(result["tokens"][1]["status"], "ok")
        self.assertEqual(result["verdict"], "compatible")
        self.assertTrue(result["detail"].startswith("2/2"))

    def test_rows_preserve_inventory_order(self):
        result = template_tokenizer_compatibility(
            "fmt",
            ("c", "a", "b"),
            {"a": 1, "b": 2, "c": 3},
            lambda text: {"a": (1,), "b": (2,), "c": (3,)}[text],
        )
        self.assertEqual([row["token"] for row in result["tokens"]], ["c", "a", "b"])
        self.assertEqual(result["verdict"], "compatible")

    def test_verdict_keys_and_types(self):
        result = template_tokenizer_compatibility("fmt", ("t",), {"t": 1}, lambda text: (1,))
        self.assertEqual(set(result), {"format", "verdict", "reason", "detail", "tokens"})
        self.assertIn(result["verdict"], ("compatible", "incompatible", "unknown"))
        self.assertIs(type(result["reason"]), str)
        self.assertIs(type(result["detail"]), str)

    def test_guards_reject_malformed_arguments(self):
        good_encode = lambda text: (1,)  # noqa: E731 -- test stub
        with self.assertRaises(ValueError):
            template_tokenizer_compatibility("", ("t",), {"t": 1}, good_encode)
        with self.assertRaises(ValueError):
            template_tokenizer_compatibility(None, ("t",), {"t": 1}, good_encode)
        with self.assertRaises(ValueError):
            template_tokenizer_compatibility("fmt", "t", {"t": 1}, good_encode)
        with self.assertRaises(ValueError):
            template_tokenizer_compatibility("fmt", ("",), {"t": 1}, good_encode)
        with self.assertRaises(ValueError):
            template_tokenizer_compatibility("fmt", (123,), {"t": 1}, good_encode)
        with self.assertRaises(ValueError):
            template_tokenizer_compatibility("fmt", ("t",), [("t", 1)], good_encode)
        with self.assertRaises(ValueError):
            template_tokenizer_compatibility("fmt", ("t",), {"t": -1}, good_encode)
        with self.assertRaises(ValueError):
            template_tokenizer_compatibility("fmt", ("t",), {"t": True}, good_encode)
        with self.assertRaises(ValueError):
            template_tokenizer_compatibility("fmt", ("t",), {1: 2}, good_encode)
        with self.assertRaises(ValueError):
            template_tokenizer_compatibility("fmt", ("t",), {"t": 1}, None)
        with self.assertRaises(ValueError):
            template_tokenizer_compatibility("fmt", ("t",), {"t": 1}, "encode")

    def test_single_token_boundary_zero_and_large_ids(self):
        result = template_tokenizer_compatibility(
            "fmt",
            ("z", "big"),
            {"z": 0, "big": 2**31 - 1},
            lambda text: {"z": (0,), "big": (2**31 - 1,)}[text],
        )
        self.assertEqual(result["verdict"], "compatible")
        self.assertTrue(result["detail"].startswith("2/2"))


if __name__ == "__main__":
    unittest.main()
