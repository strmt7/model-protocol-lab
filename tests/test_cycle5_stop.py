"""Cycle 5: stop-matrix hardening -- vLLM strip semantics and min_tokens gating.

- vLLM strips the stop string from the returned text by default
  (``include_stop_str_in_output=False``, the upstream default) while HF
  always keeps it; the emulation exposes the competitor's text as
  ``StopOutcome.returned_text``.
- vLLM's ``min_tokens`` gates all stops (vLLM PR #22014): the stop check is
  skipped while ``num_output_tokens <= min_tokens``; at the first ungated
  step the find window only covers the tail region, so a stop fully
  completed during the gate is *not* refound (only a stop completing
  at/after the gate fires) -- verified against vLLM ``main``
  ``v1/engine/detokenizer.py``.
- ``stop_matrix`` passes the new vLLM knobs through and surfaces the
  kept-vs-stripped distinction per cell (``stop_text`` field), additive to
  the existing row schema.
"""

import unittest
from threading import Event

from model_protocol_lab.backends import (
    StopOutcome,
    emulate_hf_stop_string,
    emulate_vllm_holdback,
)
from model_protocol_lab.core import Fixture, Generation, Message
from model_protocol_lab.experiments import stop_matrix


class TextBackend:
    """Backend decoding token ids through a fixed per-token text table.

    Whole-sequence decodes concatenate the per-token texts, so the vLLM
    emulation's per-prefix decodes and the HF emulation's per-token decodes
    agree here -- these tests pin the stop-string disposition, not decode
    quirks (those live in test_round3's QuirkBackend).
    """

    fingerprint = "cycle5-text-backend-not-a-model"

    def __init__(self, texts: dict[int, str]) -> None:
        self._texts = dict(texts)

    @staticmethod
    def encode(text: str) -> tuple[int, ...]:
        raise AssertionError("encode unused in stop tests")

    def decode(self, token_ids: tuple[int, ...]) -> str:
        return "".join(self._texts.get(token, chr(token)) for token in token_ids)

    @staticmethod
    def generate(
        input_ids: tuple[int, ...],
        *,
        max_new_tokens: int,
        seed: int,
        cancel: Event,
    ) -> Generation:
        raise AssertionError("generate unused in stop tests")


class VllmStripSemanticsTests(unittest.TestCase):
    def setUp(self) -> None:
        # 1->"a", 2->"b", 3->"STOP", 4->"c": stop "STOP" completes at step 3.
        self.backend = TextBackend({1: "a", 2: "b", 3: "STOP", 4: "c"})
        self.output_ids = (1, 2, 3, 4)

    def test_default_strips_stop_text(self) -> None:
        outcome = emulate_vllm_holdback(self.backend, self.output_ids, "STOP")
        self.assertTrue(outcome.available)
        self.assertTrue(outcome.fired)
        self.assertEqual(outcome.detail, "fired_at_step_3")
        # The completing token stays in token_ids (vLLM's token_ids/text
        # overshoot), but the returned text is truncated to the stop start.
        self.assertEqual(outcome.truncated_ids, (1, 2, 3))
        self.assertEqual(outcome.returned_text, "ab")

    def test_include_stop_str_in_output_keeps_it(self) -> None:
        outcome = emulate_vllm_holdback(
            self.backend, self.output_ids, "STOP", include_stop_str_in_output=True
        )
        self.assertTrue(outcome.available)
        self.assertTrue(outcome.fired)
        self.assertEqual(outcome.detail, "fired_at_step_3")
        self.assertEqual(outcome.truncated_ids, (1, 2, 3))
        self.assertEqual(outcome.returned_text, "abSTOP")

    def test_no_fire_leaves_returned_text_none(self) -> None:
        outcome = emulate_vllm_holdback(self.backend, self.output_ids, "ZZZ")
        self.assertTrue(outcome.available)
        self.assertFalse(outcome.fired)
        self.assertEqual(outcome.detail, "no_stop_within_cap")
        self.assertIsNone(outcome.returned_text)

    def test_returned_text_defaults_to_none(self) -> None:
        # Old-style positional construction stays valid and unaffected.
        outcome = StopOutcome(True, False, (1,), "no_stop_within_cap")
        self.assertIsNone(outcome.returned_text)
        self.assertEqual(outcome, StopOutcome(True, False, (1,), "no_stop_within_cap", None))


class VllmMinTokensTests(unittest.TestCase):
    def setUp(self) -> None:
        # Stop "STOP" completes at output step 2.
        self.backend = TextBackend({1: "x", 2: "STOP", 3: "y"})

    def test_gated_stop_does_not_fire(self) -> None:
        outcome = emulate_vllm_holdback(self.backend, (1, 2, 3), "STOP", min_tokens=3)
        self.assertTrue(outcome.available)
        self.assertFalse(outcome.fired)
        self.assertEqual(outcome.detail, "gated_by_min_tokens")
        self.assertEqual(outcome.truncated_ids, (1, 2, 3))
        self.assertIsNone(outcome.returned_text)

    def test_empty_output_with_zero_min_tokens_is_not_labeled_gated(self) -> None:
        # Vacuous case: no steps existed, so nothing was gated; the accurate
        # detail is no_stop_within_cap, not gated_by_min_tokens.
        outcome = emulate_vllm_holdback(self.backend, (), "STOP", min_tokens=0)
        self.assertTrue(outcome.available)
        self.assertFalse(outcome.fired)
        self.assertEqual(outcome.detail, "no_stop_within_cap")

    def test_stop_past_min_tokens_fires(self) -> None:
        outcome = emulate_vllm_holdback(self.backend, (1, 2, 3), "STOP", min_tokens=1)
        self.assertTrue(outcome.available)
        self.assertTrue(outcome.fired)
        self.assertEqual(outcome.detail, "fired_at_step_2")
        self.assertEqual(outcome.truncated_ids, (1, 2))
        self.assertEqual(outcome.returned_text, "x")

    def test_stop_spanning_gate_boundary_fires(self) -> None:
        # The stop completes at the first ungated step (its text spans the
        # gate boundary): the check runs and the tail-region find window
        # catches it, exactly like upstream's straddling-update path.
        backend = TextBackend({1: "aST", 2: "O", 3: "Pb", 4: "c"})
        outcome = emulate_vllm_holdback(backend, (1, 2, 3, 4), "STOP", min_tokens=2)
        self.assertTrue(outcome.available)
        self.assertTrue(outcome.fired)
        self.assertEqual(outcome.detail, "fired_at_step_3")
        self.assertEqual(outcome.truncated_ids, (1, 2, 3))
        self.assertEqual(outcome.returned_text, "a")

    def test_stop_completed_during_gate_is_not_refound(self) -> None:
        # The stop completed entirely inside the gated region. Upstream's
        # find window (negative start counts from the end of the text) only
        # covers the tail where a newly completing stop could begin, so the
        # old stop is never refound at later steps: checks ran, nothing found.
        backend = TextBackend({1: "x", 2: "STOP", 3: "y", 4: "z"})
        outcome = emulate_vllm_holdback(backend, (1, 2, 3, 4), "STOP", min_tokens=3)
        self.assertTrue(outcome.available)
        self.assertFalse(outcome.fired)
        self.assertEqual(outcome.detail, "no_stop_within_cap")
        self.assertEqual(outcome.truncated_ids, (1, 2, 3, 4))
        self.assertIsNone(outcome.returned_text)

    def test_min_tokens_zero_matches_default(self) -> None:
        explicit = emulate_vllm_holdback(self.backend, (1, 2, 3), "STOP", min_tokens=0)
        default = emulate_vllm_holdback(self.backend, (1, 2, 3), "STOP")
        self.assertEqual(explicit, default)

    def test_invalid_param_types(self) -> None:
        for kwargs in (
            {"include_stop_str_in_output": "yes"},
            {"include_stop_str_in_output": 1},
            {"include_stop_str_in_output": None},
            {"min_tokens": -1},
            {"min_tokens": True},
            {"min_tokens": "3"},
            {"min_tokens": 1.5},
        ):
            with self.subTest(kwargs=kwargs):
                outcome = emulate_vllm_holdback(
                    self.backend,
                    (1, 2, 3),
                    "STOP",
                    **kwargs,  # type: ignore[arg-type]
                )
                self.assertFalse(outcome.available)
                self.assertFalse(outcome.fired)
                self.assertEqual(outcome.detail, "invalid_emulation_input")
                self.assertEqual(outcome.truncated_ids, (1, 2, 3))
                self.assertIsNone(outcome.returned_text)


class HfKeepSemanticsTests(unittest.TestCase):
    def test_hf_keeps_stop_text(self) -> None:
        # HF has no include/exclude option upstream: the token completing
        # the stop string stays in the returned sequence, stop text and all.
        backend = TextBackend({1: "a", 2: "STOP"})
        outcome = emulate_hf_stop_string(backend, (), (1, 2), "STOP")
        self.assertTrue(outcome.available)
        self.assertTrue(outcome.fired)
        self.assertEqual(outcome.detail, "fired_at_step_2")
        self.assertEqual(outcome.truncated_ids, (1, 2))
        self.assertIn("STOP", backend.decode(outcome.truncated_ids))
        # Nothing is stripped, so returned_text stays None: the kept text is
        # exactly the decode of truncated_ids.
        self.assertIsNone(outcome.returned_text)

    def test_hf_suffix_match_still_fires(self) -> None:
        # The port's suffix-only match rule is unchanged: a stop string that
        # is a suffix of the decoded text fires (and is kept).
        backend = TextBackend({1: "ab", 2: "cd"})
        outcome = emulate_hf_stop_string(backend, (), (1, 2), "cd")
        self.assertTrue(outcome.fired)
        self.assertEqual(outcome.truncated_ids, (1, 2))
        self.assertIsNone(outcome.returned_text)


def _cycle5_fixture() -> Fixture:
    return Fixture("f1", (Message("user", "q"),), "ab", "g1")


def _cycle5_observation(**overrides: object) -> dict:
    obs = {
        "candidate": "c1",
        "fixture": "f1",
        "input_ids": (9,),
        "output_ids": (1, 2, 3, 4),
        "status": "completed",
        "exact_match": True,
        "boundary_found": True,
    }
    obs.update(overrides)
    return obs


class StopMatrixStopTextTests(unittest.TestCase):
    """stop_matrix surfaces the kept-vs-stripped distinction per cell."""

    def setUp(self) -> None:
        # Boundary token 3 decodes to "END"; both semantics fire at step 3.
        self.backend = TextBackend({1: "a", 2: "b", 3: "END", 4: "c", 9: "q"})

    def _rows(self, **kwargs: object) -> list:
        return stop_matrix(
            self.backend,
            "c1",
            (_cycle5_fixture(),),
            (_cycle5_observation(),),
            boundary=(3,),
            **kwargs,  # type: ignore[arg-type]
        )

    def test_stop_text_field_distinguishes_hf_and_vllm(self) -> None:
        # skipcq: PYL-W0632 - the one-element unpack is the assertion of exactly one row
        (row,) = self._rows()
        hf = row["cells"]["hf_stop_string"]
        vllm = row["cells"]["vllm_holdback"]
        self.assertTrue(hf["available"] and hf["fired"])
        self.assertTrue(vllm["available"] and vllm["fired"])
        self.assertEqual(hf["stop_text"], "kept")
        self.assertEqual(vllm["stop_text"], "stripped")
        # The EOS cell is not text-based and unavailable here (no eos_id):
        # no disposition key on the unavailable branch.
        self.assertNotIn("stop_text", row["cells"]["eos_appended"])

    def test_include_stop_str_flips_vllm_disposition(self) -> None:
        # skipcq: PYL-W0632 - the one-element unpack is the assertion of exactly one row
        (row,) = self._rows(vllm_include_stop_str_in_output=True)
        self.assertEqual(row["cells"]["vllm_holdback"]["stop_text"], "kept")
        self.assertEqual(row["cells"]["hf_stop_string"]["stop_text"], "kept")

    def test_min_tokens_gates_vllm_cell(self) -> None:
        # skipcq: PYL-W0632 - the one-element unpack is the assertion of exactly one row
        (row,) = self._rows(vllm_min_tokens=9)
        vllm = row["cells"]["vllm_holdback"]
        self.assertTrue(vllm["available"])
        self.assertFalse(vllm["fired"])
        self.assertEqual(vllm["detail"], "gated_by_min_tokens")
        # A stop that did not fire carries no disposition.
        self.assertIsNone(vllm["stop_text"])
        # The HF cell is unaffected by the vLLM-only knob.
        self.assertTrue(row["cells"]["hf_stop_string"]["fired"])

    def test_invalid_vllm_knobs_raise(self) -> None:
        with self.assertRaises(ValueError):
            self._rows(vllm_include_stop_str_in_output=1)
        with self.assertRaises(ValueError):
            self._rows(vllm_min_tokens=-1)
        with self.assertRaises(ValueError):
            self._rows(vllm_min_tokens=True)

    def test_row_schema_is_additive(self) -> None:
        # skipcq: PYL-W0632 - the one-element unpack is the assertion of exactly one row
        (row,) = self._rows()
        self.assertEqual(set(row.keys()), {"fixture", "native_pass", "cells", "flips"})
        for semantic, cell in row["cells"].items():
            self.assertIn("detail", cell, semantic)
            self.assertIn("available", cell, semantic)
            if cell["available"]:
                for key in ("pass", "fired", "stop_text"):
                    self.assertIn(key, cell, semantic)


class StopDeterminismTests(unittest.TestCase):
    def test_emulations_are_deterministic(self) -> None:
        backend = TextBackend({1: "a", 2: "b", 3: "STOP", 4: "c", 9: "q"})
        calls = [
            lambda: emulate_vllm_holdback(backend, (1, 2, 3, 4), "STOP"),
            lambda: emulate_vllm_holdback(
                backend, (1, 2, 3, 4), "STOP", include_stop_str_in_output=True
            ),
            lambda: emulate_vllm_holdback(backend, (1, 2, 3, 4), "STOP", min_tokens=5),
            lambda: emulate_hf_stop_string(backend, (9,), (1, 2, 3, 4), "STOP"),
        ]
        for call in calls:
            first, second = call(), call()
            self.assertEqual(first, second)
            self.assertEqual(hash(first), hash(second))


if __name__ == "__main__":
    unittest.main()
