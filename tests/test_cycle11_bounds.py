"""Cycle 11: observation-count bound on ``divergence_section_data``.

Pins the documented residual of the H1/M1/L1 red-team fix cycle: each
recorded observation's ids/stop-texts are individually bounded, but the
*count* of observations fed by the ``run`` diagnosis path was unbounded --
a hostile report with millions of observations would burn open-ended
emulation time (each observation costs one full ``diff_backends`` run).

``MAX_DIVERGENCE_OBSERVATIONS`` bounds the count. Over-count input
degrades by omission, never by exception: the first N observations (in
input order) are processed and the output records the fixed-vocabulary
``observations_truncated`` / ``observations_total`` /
``observations_processed`` keys. Legitimate runs are never affected:
experiments.py caps the fixture split at 100 search + 100 held-out
fixtures, and only the winner's observations feed the section, so a real
run hands over at most 200 -- well under the 512 bound.

All tests are stub-only (no torch, no network, no model weights) and
deterministic.
"""

import json
import time
import unittest
from typing import Any

from model_protocol_lab import diffbackends
from model_protocol_lab.diffbackends import (
    MAX_DIVERGENCE_OBSERVATIONS,
    divergence_section_data,
)


class LinearStubBackend:
    """O(k) decode stub: whole-sequence decodes concatenate per-token texts."""

    fingerprint = "cycle11-bounds-linear-stub-not-a-model"

    def __init__(self, size: int) -> None:
        self._texts = dict.fromkeys(range(size), "x")

    def decode(self, token_ids: tuple[int, ...]) -> str:
        return "".join(self._texts[token] for token in token_ids)


def _observation(index: int, *, ids: list[int] | None = None) -> dict[str, Any]:
    return {
        "fixture": f"f{index}",
        "input_ids": [],
        "output_ids": ids if ids is not None else [1, 2, 3],
    }


def _section(observations: list[Any], backend: LinearStubBackend) -> dict[str, Any] | None:
    return divergence_section_data(
        observations=observations,
        stop_texts=("STOP",),
        eos_id=None,
        min_tokens=0,
        backend=backend,
    )


class ObservationCountBoundTests(unittest.TestCase):
    def test_bound_is_named_constant(self) -> None:
        # No magic numbers; the rationale lives on the constant's comment.
        self.assertEqual(MAX_DIVERGENCE_OBSERVATIONS, 512)
        self.assertIs(diffbackends.MAX_DIVERGENCE_OBSERVATIONS, MAX_DIVERGENCE_OBSERVATIONS)

    def test_at_bound_everything_processed_not_truncated(self) -> None:
        # A legitimate-size input (a real run tops out at 200) is fully
        # processed with the truncated flag clear.
        backend = LinearStubBackend(16)
        observations = [_observation(i) for i in range(MAX_DIVERGENCE_OBSERVATIONS)]
        section = _section(observations, backend)
        assert section is not None
        self.assertFalse(section["observations_truncated"])
        self.assertEqual(section["observations_total"], MAX_DIVERGENCE_OBSERVATIONS)
        self.assertEqual(section["observations_processed"], MAX_DIVERGENCE_OBSERVATIONS)
        self.assertEqual(len(section["per_fixture"]), MAX_DIVERGENCE_OBSERVATIONS)
        self.assertEqual(section["per_fixture"][0]["fixture"], "f0")
        self.assertEqual(
            section["per_fixture"][-1]["fixture"], f"f{MAX_DIVERGENCE_OBSERVATIONS - 1}"
        )

    def test_boundary_n_plus_1_is_truncated(self) -> None:
        backend = LinearStubBackend(16)
        observations = [_observation(i) for i in range(MAX_DIVERGENCE_OBSERVATIONS + 1)]
        section = _section(observations, backend)
        assert section is not None
        self.assertTrue(section["observations_truncated"])
        self.assertEqual(section["observations_total"], MAX_DIVERGENCE_OBSERVATIONS + 1)
        self.assertEqual(section["observations_processed"], MAX_DIVERGENCE_OBSERVATIONS)
        self.assertEqual(len(section["per_fixture"]), MAX_DIVERGENCE_OBSERVATIONS)
        # First N kept, in input order: the (N+1)-th observation is dropped.
        self.assertEqual(
            section["per_fixture"][-1]["fixture"], f"f{MAX_DIVERGENCE_OBSERVATIONS - 1}"
        )

    def test_over_count_degrades_by_omission_never_raises(self) -> None:
        # Hostile-scale input: thousands of observations finish quickly
        # (fixed work, not open-ended) and the run never raises.
        backend = LinearStubBackend(16)
        over = MAX_DIVERGENCE_OBSERVATIONS * 5
        observations = [_observation(i) for i in range(over)]
        start = time.perf_counter()
        section = _section(observations, backend)
        elapsed = time.perf_counter() - start
        assert section is not None
        self.assertTrue(section["observations_truncated"])
        self.assertEqual(section["observations_total"], over)
        self.assertEqual(section["observations_processed"], MAX_DIVERGENCE_OBSERVATIONS)
        self.assertEqual(len(section["per_fixture"]), MAX_DIVERGENCE_OBSERVATIONS)
        # 5x the bound of tiny observations must stay cheap: the bound
        # turns a millions-of-observations DoS into fixed work.
        self.assertLess(elapsed, 120.0)

    def test_truncation_keeps_per_observation_skip_contract(self) -> None:
        # Invalid observations inside the first N are still skipped by
        # omission; valid ones beyond N are never reached.
        backend = LinearStubBackend(16)
        observations: list[Any] = []
        for i in range(MAX_DIVERGENCE_OBSERVATIONS):
            observations.append(_observation(i) if i % 2 == 0 else {"fixture": f"bad{i}"})
        observations.append(_observation(999_999))  # valid, but beyond the bound
        section = _section(observations, backend)
        assert section is not None
        self.assertTrue(section["observations_truncated"])
        self.assertEqual(section["observations_total"], MAX_DIVERGENCE_OBSERVATIONS + 1)
        self.assertEqual(len(section["per_fixture"]), MAX_DIVERGENCE_OBSERVATIONS // 2)
        self.assertNotIn("f999_999", [entry["fixture"] for entry in section["per_fixture"]])

    def test_truncation_is_deterministic(self) -> None:
        # Same input -> identical output, truncation included.
        backend = LinearStubBackend(16)
        observations = [_observation(i) for i in range(MAX_DIVERGENCE_OBSERVATIONS + 7)]
        first = _section(observations, backend)
        second = _section(observations, backend)
        assert first is not None and second is not None
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))

    def test_truncation_with_no_usable_observation_still_degrades(self) -> None:
        # Over-count input whose first N are all unusable still degrades to
        # an omitted section (None), never an exception.
        backend = LinearStubBackend(16)
        observations: list[Any] = [{"not": "an observation"}] * (MAX_DIVERGENCE_OBSERVATIONS + 3)
        section = _section(observations, backend)
        self.assertIsNone(section)

    def test_tuple_input_also_truncated(self) -> None:
        backend = LinearStubBackend(16)
        observations = tuple(_observation(i) for i in range(MAX_DIVERGENCE_OBSERVATIONS + 2))
        section = divergence_section_data(
            observations=observations,
            stop_texts=("STOP",),
            eos_id=None,
            min_tokens=0,
            backend=backend,
        )
        assert section is not None
        self.assertTrue(section["observations_truncated"])
        self.assertEqual(section["observations_total"], MAX_DIVERGENCE_OBSERVATIONS + 2)
        self.assertEqual(section["observations_processed"], MAX_DIVERGENCE_OBSERVATIONS)


if __name__ == "__main__":
    unittest.main()
