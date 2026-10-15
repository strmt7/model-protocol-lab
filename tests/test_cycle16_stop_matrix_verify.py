"""Cycle 16: independent differential verification of stop_matrix/_stop_cell.

Target: ``stop_matrix`` and ``_stop_cell`` in
``model_protocol_lab/experiments.py``. The docstring claims an EXACT post-hoc
derivation: for each recorded observation (token-id output) and each
stop-string candidate, the matrix recomputes what the backend stop semantics
WOULD have produced (which stop fired, at which token offset, what the
truncated output and kept/stripped text would be) without any model calls, by
the same post-hoc argument used in ``budget_frontier``.

Oracle method (from the docstring definition only): for each
(observation, stop-string) pair, SIMULATE generation step by step -- replay
the recorded ``output_ids`` one token at a time through the already-verified
``emulate_*`` stop functions in ``model_protocol_lab/backends.py`` (cycles
9-11 ground truth) and record the first step where each semantic fires. The
oracle never does closed-form offset arithmetic and never parses the
implementation's ``detail`` strings: the firing step is discovered by the
prefix scan, ``truncated_ids`` come from the firing prefix's own outcome, and
the pass verdict is recomputed from those simulated ids with
``first_boundary`` + ``backend.decode``. The implementation's cell dicts
(``fired``, the ``fired_at_step_N`` detail, ``pass``, the kept/stripped
``stop_text`` disposition) and the matrix aggregation (``flips``) are then
compared field by field.

``stop_matrix`` derives ONE stop string per call (the decoded boundary), so
the "1-3 stop strings per matrix" fuzz dimension runs the matrix once per
boundary candidate: overlapping stops, prefix-related stops, multi-token
stops, and empty-after-decode boundaries.
"""

import importlib
import os
import random
import shutil
import sys
import tempfile
import unittest
from collections.abc import Callable
from typing import Any

import model_protocol_lab
from model_protocol_lab.backends import (
    emulate_eos_stop,
    emulate_hf_stop_string,
    emulate_vllm_holdback,
)
from model_protocol_lab.core import Fixture, Message, first_boundary
from model_protocol_lab.experiments import stop_matrix

_SEMANTICS = ("hf_stop_string", "vllm_holdback", "eos_appended")

_TOKEN_TEXTS: dict[int, str] = {
    0: "<EOS>",
    1: "a",
    2: "b",
    3: "c",
    4: "ab",
    5: "bc",
    6: "STOP",
    7: "ST",
    8: "OP",
    9: "",  # decodes to no text: exercises vLLM's new_char_count skip
}

_SINGLE_TOKENS = (1, 2, 3, 4, 5, 6, 7, 8)

# Boundary token tuples and the stop text each decodes to:
#   (6,)   -> "STOP"                 single-token stop
#   (7, 8) -> "STOP"                 multi-token stop, same text, other split
#   (7,)   -> "ST"                   prefix of "STOP"
#   (8,)   -> "OP"                   suffix overlapping "STOP"
#   (4,)   -> "ab"                   overlapping with "a"+"b"
#   (1, 2) -> "ab"                   same text via two tokens
#   (3,)   -> "c"
#   (9,)   -> ""                     empty after decode: string cells unavailable
_BOUNDARY_POOL: tuple[tuple[int, ...], ...] = (
    (6,),
    (7, 8),
    (7,),
    (8,),
    (4,),
    (1, 2),
    (3,),
    (9,),
)


class TextBackend:
    """Backend decoding token ids through a fixed per-token text table.

    Whole-sequence decodes concatenate the per-token texts, so incremental
    prefix decodes (vLLM path) and per-token decodes (HF path) agree here --
    these tests pin the stop-matrix derivation, not decode quirks.
    """

    fingerprint = "cycle16-stop-matrix-oracle-not-a-model"

    def __init__(self, texts: dict[int, str]) -> None:
        self._texts = dict(texts)

    def decode(self, token_ids: tuple[int, ...]) -> str:
        return "".join(self._texts[token] for token in token_ids)


def _oracle_outcome(
    backend: TextBackend,
    semantic: str,
    prompt_ids: tuple[int, ...],
    output_ids: tuple[int, ...],
    stop_text: str | None,
    *,
    vllm_include_stop_str_in_output: bool,
    vllm_min_tokens: int,
    eos_id: int | None,
) -> tuple[bool, bool, str, tuple[int, ...]]:
    """Incremental-simulation oracle: (available, fired, detail, truncated_ids).

    Replays the recorded output one token at a time through the verified
    ``emulate_*`` functions and reports the first prefix that fires. The
    firing step is discovered by simulation -- never parsed from the
    implementation's detail strings and never computed by closed-form
    offset arithmetic.
    """
    ids = tuple(output_ids)
    if semantic in ("hf_stop_string", "vllm_holdback") and not stop_text:
        # stop_matrix passes "" to the emulators when the boundary decodes
        # to no text; both emulators reject it as invalid input.
        return False, False, "invalid_emulation_input", ids
    if semantic == "hf_stop_string":
        assert stop_text is not None
        for step in range(1, len(ids) + 1):
            outcome = emulate_hf_stop_string(backend, prompt_ids, ids[:step], stop_text)
            if not outcome.available:
                return False, False, outcome.detail, ids
            if outcome.fired:
                return True, True, f"fired_at_step_{step}", outcome.truncated_ids
        return True, False, "no_stop_within_cap", ids
    if semantic == "vllm_holdback":
        assert stop_text is not None
        for step in range(1, len(ids) + 1):
            outcome = emulate_vllm_holdback(
                backend,
                ids[:step],
                stop_text,
                include_stop_str_in_output=vllm_include_stop_str_in_output,
                min_tokens=vllm_min_tokens,
            )
            if not outcome.available:
                return False, False, outcome.detail, ids
            if outcome.fired:
                return True, True, f"fired_at_step_{step}", outcome.truncated_ids
        # No step fired: the docstring's terminal condition decides the detail.
        if vllm_min_tokens >= len(ids) and vllm_min_tokens > 0:
            return True, False, "gated_by_min_tokens", ids
        return True, False, "no_stop_within_cap", ids
    if semantic == "eos_appended":
        if eos_id is None:
            return False, False, "eos_id_unavailable", ids
        for step in range(1, len(ids) + 1):
            outcome = emulate_eos_stop(ids[:step], eos_id)
            if not outcome.available:
                return False, False, outcome.detail, ids
            if outcome.fired:
                return True, True, f"fired_at_step_{step}", outcome.truncated_ids
        return True, False, "no_stop_within_cap", ids
    raise AssertionError(f"unknown semantic {semantic!r}")


def _expected_cell(
    backend: TextBackend,
    fixture: Fixture,
    boundary: tuple[int, ...],
    available: bool,
    fired: bool,
    detail: str,
    truncated_ids: tuple[int, ...],
    stop_text_disposition: str | None,
) -> dict[str, Any]:
    """Recompute the lab's pass rule on the oracle's simulated truncation."""
    if not available:
        return {"available": False, "detail": detail}
    truncated = tuple(truncated_ids)
    cut = first_boundary(truncated, tuple(boundary))
    answer_ids = truncated if cut is None else truncated[:cut]
    answer = backend.decode(answer_ids)
    return {
        "available": True,
        "pass": bool(cut is not None and answer == fixture.expected),
        "fired": fired,
        "detail": detail,
        "stop_text": stop_text_disposition if fired else None,
    }


def check_matrix_against_oracle(
    stop_matrix_fn: Callable[..., list[dict[str, Any]]],
    backend: TextBackend,
    fixtures: tuple[Fixture, ...],
    observations: tuple[dict[str, Any], ...],
    *,
    boundary: tuple[int, ...],
    eos_id: int | None,
    vllm_include_stop_str_in_output: bool,
    vllm_min_tokens: int,
) -> None:
    """Assert every row, cell, and flip matches the incremental oracle.

    Raises AssertionError on any mismatch, so sensitivity tests can assert
    the harness catches implementation mutations.
    """
    rows = stop_matrix_fn(
        backend,
        "cycle16-oracle-candidate",
        fixtures,
        observations,
        boundary=boundary,
        eos_id=eos_id,
        vllm_include_stop_str_in_output=vllm_include_stop_str_in_output,
        vllm_min_tokens=vllm_min_tokens,
    )
    assert len(rows) == len(fixtures), (len(rows), len(fixtures))
    try:
        boundary_text: Any = backend.decode(boundary)
    except Exception:  # noqa: BLE001 -- mirrors stop_matrix's degrade path
        boundary_text = None
    if type(boundary_text) is not str or not boundary_text:
        boundary_text = None
    for row, fixture, obs in zip(rows, fixtures, observations, strict=True):
        assert row["fixture"] == fixture.id, (row["fixture"], fixture.id)
        native_pass = bool(obs.get("exact_match") and obs.get("boundary_found"))
        assert row["native_pass"] == native_pass, (row["native_pass"], native_pass)
        prompt_ids = tuple(obs.get("input_ids") or ())
        output_ids = tuple(obs.get("output_ids") or ())
        complete = obs.get("status") == "completed" and all(
            type(token) is int for token in output_ids
        )
        expected_cells: dict[str, dict[str, Any]] = {}
        for semantic in _SEMANTICS:
            if not complete:
                expected_cells[semantic] = {
                    "available": False,
                    "detail": "observation_incomplete",
                }
                continue
            if semantic == "hf_stop_string":
                disposition: str | None = "kept"
            elif semantic == "vllm_holdback":
                disposition = "kept" if vllm_include_stop_str_in_output else "stripped"
            else:
                disposition = None
            available, fired, detail, truncated = _oracle_outcome(
                backend,
                semantic,
                prompt_ids,
                output_ids,
                boundary_text,
                vllm_include_stop_str_in_output=vllm_include_stop_str_in_output,
                vllm_min_tokens=vllm_min_tokens,
                eos_id=eos_id,
            )
            expected_cells[semantic] = _expected_cell(
                backend,
                fixture,
                boundary,
                available,
                fired,
                detail,
                truncated,
                disposition,
            )
        assert row["cells"] == expected_cells, (semantic, row["cells"], expected_cells)
        expected_flips = [
            semantic
            for semantic in _SEMANTICS
            if expected_cells[semantic].get("available")
            and expected_cells[semantic].get("pass") != native_pass
        ]
        assert row["flips"] == expected_flips, (row["flips"], expected_flips)


def _make_fixture(fixture_id: str, expected: str) -> Fixture:
    return Fixture(
        id=fixture_id,
        messages=(Message(role="user", content="cycle16 probe"),),
        expected=expected,
        group="cycle16",
    )


def _gen_case(rng: random.Random) -> dict[str, Any]:
    """One deterministic fuzz case: outputs, prompt, and per-case knobs."""
    # Build the output from chunks so stop occurrences land at varied
    # offsets (0, middle, final tokens), or are absent / partial.
    chunks: list[tuple[int, ...]] = []
    for _ in range(rng.randrange(0, 5)):
        roll = rng.random()
        if roll < 0.45:
            chunks.append(rng.choice(_BOUNDARY_POOL))
        elif roll < 0.85:
            chunks.append((rng.choice(_SINGLE_TOKENS),))
        elif roll < 0.95:
            chunks.append((0,))  # EOS token
        else:
            chunks.append((9,))  # empty-text token
    output_ids = tuple(token for chunk in chunks for token in chunk)[:8]
    prompt_ids = tuple(rng.choice((1, 2, 7)) for _ in range(rng.randrange(0, 3)))
    n_boundaries = rng.choice((1, 1, 2, 2, 3))
    return {
        "output_ids": output_ids,
        "prompt_ids": prompt_ids,
        "boundaries": tuple(rng.sample(_BOUNDARY_POOL, k=n_boundaries)),
        "vllm_include_stop_str_in_output": rng.random() < 0.5,
        "vllm_min_tokens": rng.randrange(0, 4),
        "eos_id": rng.choice((None, 0)),
    }


def _native_fields(
    backend: TextBackend,
    output_ids: tuple[int, ...],
    boundary: tuple[int, ...],
    rng: random.Random,
) -> tuple[bool, bool, str]:
    """Honest (exact_match, boundary_found, expected) for a fuzz observation."""
    cut = first_boundary(tuple(output_ids), tuple(boundary))
    if cut is None:
        return False, False, "unmatched∤"
    answer = backend.decode(tuple(output_ids[:cut]))
    if rng.random() < 0.7:
        return True, True, answer
    return False, True, answer + "∤"


class StopMatrixOracleFuzzTests(unittest.TestCase):
    N_CASES = 250

    def test_fuzz_against_incremental_oracle(self) -> None:
        backend = TextBackend(_TOKEN_TEXTS)
        rng = random.Random(0xC16C)  # noqa: S311 -- deterministic test seed
        for case_index in range(self.N_CASES):
            case = _gen_case(rng)
            for boundary in case["boundaries"]:
                with self.subTest(case=case_index, boundary=boundary):
                    exact_match, boundary_found, expected = _native_fields(
                        backend, case["output_ids"], boundary, rng
                    )
                    fixture = _make_fixture(f"fuzz-{case_index}", expected)
                    obs = {
                        "fixture": fixture.id,
                        "input_ids": case["prompt_ids"],
                        "output_ids": case["output_ids"],
                        "status": "completed",
                        "exact_match": exact_match,
                        "boundary_found": boundary_found,
                    }
                    check_matrix_against_oracle(
                        stop_matrix,
                        backend,
                        (fixture,),
                        (obs,),
                        boundary=boundary,
                        eos_id=case["eos_id"],
                        vllm_include_stop_str_in_output=case["vllm_include_stop_str_in_output"],
                        vllm_min_tokens=case["vllm_min_tokens"],
                    )


class StopMatrixTargetedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = TextBackend(_TOKEN_TEXTS)

    def _run(
        self,
        output_ids: tuple[int, ...],
        boundary: tuple[int, ...],
        expected: str,
        *,
        prompt_ids: tuple[int, ...] = (),
        status: str = "completed",
        exact_match: bool | None = None,
        boundary_found: bool | None = None,
        eos_id: int | None = None,
        vllm_include_stop_str_in_output: bool = False,
        vllm_min_tokens: int = 0,
        fixture_id: str = "fx",
    ) -> list[dict[str, Any]]:
        cut = first_boundary(tuple(output_ids), tuple(boundary))
        if boundary_found is None:
            boundary_found = cut is not None
        if exact_match is None:
            answer = self.backend.decode(tuple(output_ids[:cut])) if cut is not None else None
            exact_match = bool(boundary_found and answer == expected)
        fixture = _make_fixture(fixture_id, expected)
        obs = {
            "fixture": fixture_id,
            "input_ids": prompt_ids,
            "output_ids": output_ids,
            "status": status,
            "exact_match": exact_match,
            "boundary_found": boundary_found,
        }
        check_matrix_against_oracle(
            stop_matrix,
            self.backend,
            (fixture,),
            (obs,),
            boundary=boundary,
            eos_id=eos_id,
            vllm_include_stop_str_in_output=vllm_include_stop_str_in_output,
            vllm_min_tokens=vllm_min_tokens,
        )
        return stop_matrix(
            self.backend,
            "targeted",
            (fixture,),
            (obs,),
            boundary=boundary,
            eos_id=eos_id,
            vllm_include_stop_str_in_output=vllm_include_stop_str_in_output,
            vllm_min_tokens=vllm_min_tokens,
        )

    def test_stop_at_offset_zero_fires_at_step_1(self) -> None:
        rows = self._run((6, 1, 2), (6,), "")
        cells = rows[0]["cells"]
        self.assertEqual(cells["hf_stop_string"]["detail"], "fired_at_step_1")
        self.assertEqual(cells["vllm_holdback"]["detail"], "fired_at_step_1")
        self.assertTrue(cells["hf_stop_string"]["pass"])

    def test_first_occurrence_wins(self) -> None:
        rows = self._run((6, 1, 6), (6,), "")
        self.assertEqual(rows[0]["cells"]["hf_stop_string"]["detail"], "fired_at_step_1")
        self.assertEqual(rows[0]["cells"]["vllm_holdback"]["detail"], "fired_at_step_1")

    def test_prompt_overhang_fires_hf_but_not_vllm(self) -> None:
        # Prompt ends with "ST", output starts with "OP": HF sees the
        # prompt+output overhang and fires at step 1; vLLM only searches the
        # decoded output text, which never contains "STOP".
        rows = self._run((8, 1), (6,), "", prompt_ids=(7,))
        cells = rows[0]["cells"]
        self.assertEqual(cells["hf_stop_string"]["detail"], "fired_at_step_1")
        self.assertFalse(cells["vllm_holdback"]["fired"])
        self.assertEqual(cells["vllm_holdback"]["detail"], "no_stop_within_cap")

    def test_empty_output_fires_nothing(self) -> None:
        rows = self._run((), (6,), "zzz", eos_id=0)
        cells = rows[0]["cells"]
        for semantic in _SEMANTICS:
            self.assertFalse(cells[semantic]["fired"])
        self.assertEqual(cells["vllm_holdback"]["detail"], "no_stop_within_cap")
        self.assertEqual(rows[0]["flips"], [])

    def test_empty_boundary_text_marks_string_cells_unavailable(self) -> None:
        rows = self._run((1, 2), (9,), "zzz", eos_id=0)
        cells = rows[0]["cells"]
        for semantic in ("hf_stop_string", "vllm_holdback"):
            self.assertEqual(
                cells[semantic], {"available": False, "detail": "invalid_emulation_input"}
            )
        # The EOS cell does not depend on the stop text.
        self.assertTrue(cells["eos_appended"]["available"])
        self.assertFalse(cells["eos_appended"]["fired"])

    def test_eos_before_boundary_flips_verdict(self) -> None:
        # Token 0 decodes to "<EOS>" text, so the native answer is "ab<EOS>".
        rows = self._run((1, 2, 0, 6), (6,), "ab<EOS>", eos_id=0)
        row = rows[0]
        self.assertTrue(row["native_pass"])
        eos_cell = row["cells"]["eos_appended"]
        self.assertTrue(eos_cell["fired"])
        self.assertEqual(eos_cell["detail"], "fired_at_step_3")
        self.assertFalse(eos_cell["pass"])
        self.assertIsNone(eos_cell["stop_text"])
        self.assertEqual(row["flips"], ["eos_appended"])

    def test_min_tokens_gate_suppresses_completed_stop(self) -> None:
        # "STOP" completes at step 2 but min_tokens=2 gates it; the later
        # find window does not refind it, so vLLM never fires.
        rows = self._run((1, 6, 2, 3), (6,), "ab", vllm_min_tokens=2)
        vllm_cell = rows[0]["cells"]["vllm_holdback"]
        self.assertFalse(vllm_cell["fired"])
        self.assertEqual(vllm_cell["detail"], "no_stop_within_cap")
        # HF has no min_tokens gate: it fires at step 2.
        self.assertEqual(rows[0]["cells"]["hf_stop_string"]["detail"], "fired_at_step_2")

    def test_fully_gated_output_reports_gated_detail(self) -> None:
        rows = self._run((1, 6), (6,), "ab", vllm_min_tokens=2)
        vllm_cell = rows[0]["cells"]["vllm_holdback"]
        self.assertFalse(vllm_cell["fired"])
        self.assertEqual(vllm_cell["detail"], "gated_by_min_tokens")

    def test_incomplete_observation_marks_cells_unavailable(self) -> None:
        rows = self._run(
            (1, 2, 6), (6,), "ab", status="failed", exact_match=False, boundary_found=True
        )
        row = rows[0]
        for semantic in _SEMANTICS:
            self.assertEqual(
                row["cells"][semantic],
                {"available": False, "detail": "observation_incomplete"},
            )
        self.assertEqual(row["flips"], [])

    def test_stop_spanning_final_tokens(self) -> None:
        # The "STOP" text ends at the final token; the token-id boundary is
        # present, so the truncated output still passes.
        rows = self._run((1, 6), (6,), "a")
        cells = rows[0]["cells"]
        self.assertEqual(cells["hf_stop_string"]["detail"], "fired_at_step_2")
        self.assertEqual(cells["vllm_holdback"]["detail"], "fired_at_step_2")
        self.assertTrue(cells["hf_stop_string"]["pass"])

    def test_multi_token_stop_boundary(self) -> None:
        # Boundary (7, 8) decodes to "STOP" across two tokens; the output
        # carries the same split, so the token-id boundary is found and the
        # string stop fires on the completed text at step 3.
        rows = self._run((1, 7, 8), (7, 8), "a")
        cells = rows[0]["cells"]
        self.assertEqual(cells["hf_stop_string"]["detail"], "fired_at_step_3")
        self.assertEqual(cells["vllm_holdback"]["detail"], "fired_at_step_3")
        self.assertTrue(cells["hf_stop_string"]["pass"])

    def test_prefix_related_stop_strings(self) -> None:
        # Same output, two matrices: "ST" fires at step 1, "STOP" never fires.
        rows_st = self._run((7, 1), (7,), "", fixture_id="fx-st")
        self.assertEqual(rows_st[0]["cells"]["hf_stop_string"]["detail"], "fired_at_step_1")
        rows_stop = self._run((7, 1), (6,), "", fixture_id="fx-stop")
        self.assertFalse(rows_stop[0]["cells"]["hf_stop_string"]["fired"])
        self.assertFalse(rows_stop[0]["cells"]["vllm_holdback"]["fired"])

    def test_overlapping_stop_occurrence(self) -> None:
        # "abab": the "ab" stop completes at step 1 (first occurrence wins).
        rows = self._run((4, 4), (4,), "")
        self.assertEqual(rows[0]["cells"]["hf_stop_string"]["detail"], "fired_at_step_1")
        self.assertTrue(rows[0]["cells"]["hf_stop_string"]["pass"])

    def test_vllm_keep_vs_strip_disposition(self) -> None:
        rows_stripped = self._run((1, 6), (6,), "a")
        self.assertEqual(rows_stripped[0]["cells"]["vllm_holdback"]["stop_text"], "stripped")
        self.assertEqual(rows_stripped[0]["cells"]["hf_stop_string"]["stop_text"], "kept")
        rows_kept = self._run((1, 6), (6,), "a", vllm_include_stop_str_in_output=True)
        self.assertEqual(rows_kept[0]["cells"]["vllm_holdback"]["stop_text"], "kept")

    def test_multi_fixture_matrix(self) -> None:
        backend = self.backend
        fixtures = (
            _make_fixture("m1", "a"),  # passes natively
            _make_fixture("m2", "zzz"),  # fails natively
            _make_fixture("m3", "a"),  # incomplete observation
        )
        observations = (
            {
                "fixture": "m1",
                "input_ids": (),
                "output_ids": (1, 6),
                "status": "completed",
                "exact_match": True,
                "boundary_found": True,
            },
            {
                "fixture": "m2",
                "input_ids": (),
                "output_ids": (1, 2),
                "status": "completed",
                "exact_match": False,
                "boundary_found": False,
            },
            {
                "fixture": "m3",
                "input_ids": (),
                "output_ids": (1, 6),
                "status": "cancelled",
                "exact_match": False,
                "boundary_found": False,
            },
        )
        check_matrix_against_oracle(
            stop_matrix,
            backend,
            fixtures,
            observations,
            boundary=(6,),
            eos_id=None,
            vllm_include_stop_str_in_output=False,
            vllm_min_tokens=0,
        )
        rows = stop_matrix(backend, "multi", fixtures, observations, boundary=(6,), eos_id=None)
        self.assertEqual([row["fixture"] for row in rows], ["m1", "m2", "m3"])
        self.assertEqual([row["native_pass"] for row in rows], [True, False, False])


class StopMatrixValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = TextBackend(_TOKEN_TEXTS)
        self.fixture = _make_fixture("v", "a")
        self.obs = {
            "fixture": "v",
            "input_ids": (),
            "output_ids": (1, 6),
            "status": "completed",
            "exact_match": True,
            "boundary_found": True,
        }

    def _call(self, **kwargs: Any) -> list[dict[str, Any]]:
        args: dict[str, Any] = {
            "candidate_name": "c",
            "fixtures": (self.fixture,),
            "observations": (self.obs,),
            "boundary": (6,),
        }
        args.update(kwargs)
        return stop_matrix(self.backend, **args)  # type: ignore[arg-type]

    def test_mismatched_fixture_id_raises(self) -> None:
        obs = dict(self.obs, fixture="other")
        with self.assertRaises(ValueError):
            self._call(observations=(obs,))

    def test_blank_candidate_name_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._call(candidate_name="")

    def test_misaligned_lengths_raise(self) -> None:
        with self.assertRaises(ValueError):
            self._call(fixtures=(self.fixture, self.fixture))

    def test_empty_boundary_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._call(boundary=())

    def test_negative_min_tokens_raises(self) -> None:
        with self.assertRaises(ValueError):
            stop_matrix(
                self.backend,
                "c",
                (self.fixture,),
                (self.obs,),
                boundary=(6,),
                vllm_min_tokens=-1,
            )


_BUMP_STEP_HELPER = (
    "def _c16_bump_step(detail):\n"
    "    import re\n"
    "\n"
    '    match = re.match(r"fired_at_step_(\\d+)$", detail or "")\n'
    "    if match:\n"
    '        return f"fired_at_step_{int(match.group(1)) + 1}"\n'
    "    return detail\n"
    "\n\n"
)


class StopMatrixSensitivityTests(unittest.TestCase):
    """Prove the oracle harness catches mutations of the implementation.

    Each test copies the whole ``model_protocol_lab`` package to a temp dir
    under a unique name (all intra-package imports are relative, so the copy
    imports cleanly), applies a textual mutation to the copy's
    ``experiments.py`` -- with an exact-occurrence guard so a drifted source
    fails loudly instead of silently not mutating -- and asserts the oracle
    checker raises AssertionError on the mutated matrix.
    """

    def _mutated(self, tag: str, patches: list[tuple[str, str, int]]) -> tuple[Any, Any]:
        src_dir = os.path.dirname(os.path.abspath(model_protocol_lab.__file__))
        tmp = tempfile.mkdtemp(prefix="c16sens_")
        self.addCleanup(shutil.rmtree, tmp, True)
        pkg_name = f"c16sens_{tag}"
        shutil.copytree(src_dir, os.path.join(tmp, pkg_name))
        exp_path = os.path.join(tmp, pkg_name, "experiments.py")
        with open(exp_path, encoding="utf-8") as handle:
            src = handle.read()
        for old, new, expected_count in patches:
            actual = src.count(old)
            assert actual == expected_count, (tag, old, actual)
            src = src.replace(old, new)
        with open(exp_path, "w", encoding="utf-8") as handle:
            handle.write(src)
        sys.path.insert(0, tmp)
        self.addCleanup(sys.path.remove, tmp)
        experiments_mod = importlib.import_module(f"{pkg_name}.experiments")
        core_mod = importlib.import_module(f"{pkg_name}.core")
        return experiments_mod, core_mod

    def _sensitivity_inputs(
        self, core_mod: Any
    ) -> tuple[TextBackend, tuple[Any, ...], tuple[dict[str, Any], ...]]:
        backend = TextBackend(_TOKEN_TEXTS)
        fixture = core_mod.Fixture(
            id="sens",
            messages=(core_mod.Message(role="user", content="q"),),
            expected="ab",
            group="cycle16",
        )
        obs = {
            "fixture": "sens",
            "input_ids": (),
            "output_ids": (1, 2, 6, 3),
            "status": "completed",
            "exact_match": True,
            "boundary_found": True,
        }
        return backend, (fixture,), (obs,)

    def _assert_mutation_caught(self, tag: str, patches: list[tuple[str, str, int]]) -> None:
        experiments_mod, core_mod = self._mutated(tag, patches)
        backend, fixtures, observations = self._sensitivity_inputs(core_mod)
        with self.assertRaises(AssertionError):
            check_matrix_against_oracle(
                experiments_mod.stop_matrix,
                backend,
                fixtures,
                observations,
                boundary=(6,),
                eos_id=None,
                vllm_include_stop_str_in_output=False,
                vllm_min_tokens=0,
            )

    def test_unmutated_copy_passes_oracle(self) -> None:
        # Control: the harness itself is sound -- an exact copy passes.
        experiments_mod, core_mod = self._mutated("control", [])
        backend, fixtures, observations = self._sensitivity_inputs(core_mod)
        check_matrix_against_oracle(
            experiments_mod.stop_matrix,
            backend,
            fixtures,
            observations,
            boundary=(6,),
            eos_id=None,
            vllm_include_stop_str_in_output=False,
            vllm_min_tokens=0,
        )

    def test_shifted_firing_step_is_caught(self) -> None:
        self._assert_mutation_caught(
            "step",
            [
                ("def _stop_cell(\n", _BUMP_STEP_HELPER + "def _stop_cell(\n", 1),
                (
                    '        "fired": outcome.fired,\n        "detail": outcome.detail,\n',
                    '        "fired": outcome.fired,\n'
                    '        "detail": _c16_bump_step(outcome.detail),\n',
                    1,
                ),
            ],
        )

    def test_swapped_kept_stripped_is_caught(self) -> None:
        self._assert_mutation_caught(
            "disposition",
            [
                (
                    '            "hf_stop_string": "kept",\n',
                    '            "hf_stop_string": "stripped",\n',
                    1,
                ),
                (
                    '            "vllm_holdback": "kept" if vllm_include_stop_str_in_output'
                    ' else "stripped",\n',
                    '            "vllm_holdback": "stripped" if vllm_include_stop_str_in_output'
                    ' else "kept",\n',
                    1,
                ),
            ],
        )

    def test_inverted_pass_verdict_is_caught(self) -> None:
        self._assert_mutation_caught(
            "pass",
            [
                (
                    '        "pass": bool(cut is not None and answer == fixture.expected),\n',
                    '        "pass": not bool(cut is not None and answer == fixture.expected),\n',
                    1,
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
