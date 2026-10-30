"""Cycle-77 adversarial audit: observation classification and failure taxonomy.

Target: ``model_protocol_lab.experiments.classify_observation``,
``failure_taxonomy``, ``_stop_cell`` and ``stop_matrix`` -- the analytical
core that turns raw backend observations into the classifications and
matrices shown in reports. Cycle 69 audited the run diagnostics; the
classification layer itself had no dedicated adversarial audit until now.

Method:
  1. Independent oracles re-derived from the docstrings (structurally
     different from the source), checked against the implementation over
     the full cartesian product of field values.
  2. Minimal pairs: exactly one field flips the class, pinning the
     priority order of the closed taxonomy.
  3. Adversarial edges: empty id tuples, contradictory fields, unicode,
     huge observations, missing keys, wrong types (fail-closed with fixed
     codes, never tracebacks).
  4. Hand-computed taxonomy aggregations, empty and all-failed inputs.
  5. One fully hand-derived end-to-end stop-matrix case plus a
     differential wiring test (trusted emulation + oracle cell).

Notable design property verified here: the classification layer has NO
"ambiguous"/"unclassified" escape hatch -- every non-passing observation
maps to exactly one of ``FAILURE_MODES``, and the docstring contract says
so ("Closed taxonomy"). The words "ambiguous"/"unclassified" do not appear
in this layer.

All backends/tokenizers here are synthetic and scripted; no model
weights, no network, no downloads.
"""

import itertools
import unittest
from typing import Any, cast

from model_protocol_lab.backends import (
    StopOutcome,
    emulate_eos_stop,
    emulate_hf_stop_string,
    emulate_vllm_holdback,
)
from model_protocol_lab.core import Fixture, Message
from model_protocol_lab.experiments import (
    FAILURE_MODES,
    STOP_SEMANTICS,
    _stop_cell,
    classify_observation,
    failure_taxonomy,
    stop_matrix,
)

ALL_MODES = frozenset(FAILURE_MODES)


def oracle_classify_observation(obs: Any, eos_id: int | None) -> str | None:
    """Independent re-derivation of ``classify_observation`` from its docstring.

    Structurally different from the source: the guards are ordered as an
    explicit if/elif chain, the EOS-membership test is a manual loop rather
    than ``in tuple(...)``, and the falsy-id guard is written out.
    """
    if not isinstance(obs, dict):
        raise ValueError("observation must be a dict")
    exact = obs.get("exact_match")
    found = obs.get("boundary_found")
    if exact and found:
        return None
    status = obs.get("status")
    raw_reason = obs.get("raw_reason")
    if status == "failed" or raw_reason == "error" or status == "cancelled":
        return "backend-error"
    if not found:
        if obs.get("boundary_text_match") is True:
            return "boundary-text-mismatch"
        if eos_id is not None:
            # The id sequence is only touched when the EOS branch is
            # actually reached: a truthy non-iterable output_ids raises
            # TypeError here, but never when eos_id is None.
            raw_ids = obs.get("output_ids")
            ids: list[Any] = list(raw_ids) if raw_ids else []
            for token in ids:  # manual membership instead of ``in tuple(...)``
                if token == eos_id:
                    return "eos-without-boundary"
        return "no-boundary-in-N"
    # Boundary found but the answer was wrong: the expected answer never
    # appeared boundary-delimited within the token cap.
    return "no-boundary-in-N"


def oracle_first_boundary(tokens: tuple[int, ...], stop: tuple[int, ...]) -> int | None:
    """Own contiguous-subsequence search (while-loop, no slicing)."""
    n = len(stop)
    i = 0
    while i + n <= len(tokens):
        match = True
        j = 0
        while j < n:
            if tokens[i + j] != stop[j]:
                match = False
                break
            j += 1
        if match:
            return i
        i += 1
    return None


def oracle_stop_cell(
    backend: Any,
    fixture: Fixture,
    boundary: tuple[int, ...],
    outcome: StopOutcome,
    stop_text: str | None = None,
) -> dict[str, Any]:
    """Independent re-derivation of ``_stop_cell`` from its docstring.

    Never raises on decode in the fuzz harness (the scripted backend is
    total there); the raising-decode and non-str-decode cases are pinned
    separately as fixed cases.
    """
    if not outcome.available:
        return {"available": False, "detail": outcome.detail}
    truncated = tuple(outcome.truncated_ids)
    cut = oracle_first_boundary(truncated, boundary)
    answer_ids = truncated if cut is None else truncated[:cut]
    answer = backend.decode(answer_ids)
    if type(answer) is not str:
        return {"available": False, "detail": "answer_decode_unavailable"}
    return {
        "available": True,
        "pass": bool(cut is not None and answer == fixture.expected),
        "fired": outcome.fired,
        "detail": outcome.detail,
        "stop_text": stop_text if outcome.fired else None,
    }


def _run_both(obs: dict[str, Any], eos_id: int | None) -> tuple[tuple[str, Any], tuple[str, Any]]:
    """Run oracle and implementation; normalize to ('value', v) or ('raised', T)."""
    try:
        expected = oracle_classify_observation(obs, eos_id)
        expected_outcome: tuple[str, Any] = ("value", expected)
    except Exception as exc:  # noqa: BLE001 -- differential harness records the type
        expected_outcome = ("raised", type(exc).__name__)
    try:
        actual = classify_observation(obs, eos_id)
        actual_outcome: tuple[str, Any] = ("value", actual)
    except Exception as exc:  # noqa: BLE001 -- differential harness records the type
        actual_outcome = ("raised", type(exc).__name__)
    return expected_outcome, actual_outcome


class ClassifyDifferentialTests(unittest.TestCase):
    """Oracle-vs-implementation over the full field cartesian product."""

    def test_differential_over_cartesian_product(self):
        """Test differential over cartesian product."""
        exacts = [True, False, None, 1]
        founds = [True, False, None]
        mismatches = [True, False, None, 1, "yes"]
        statuses = ["completed", "failed", "cancelled", None]
        raw_reasons = ["error", "limit", None]
        id_seqs: list[Any] = [(), (1, 2), (99,), None, 123, [99], "99"]
        eos_ids: list[int | None] = [None, 99]
        total = 0
        for combo in itertools.product(
            exacts, founds, mismatches, statuses, raw_reasons, id_seqs, eos_ids
        ):
            exact, found, mismatch, status, raw_reason, ids, eos_id = combo
            obs = {
                "exact_match": exact,
                "boundary_found": found,
                "boundary_text_match": mismatch,
                "status": status,
                "raw_reason": raw_reason,
                "output_ids": ids,
            }
            total += 1
            with self.subTest(combo=combo):
                expected, actual = _run_both(obs, eos_id)
                self.assertEqual(actual, expected)
                if actual[0] == "value":
                    # The taxonomy is closed: no ambiguous/unclassified hatch.
                    self.assertIn(actual[1], ALL_MODES | {None})
        self.assertEqual(total, 4 * 3 * 5 * 4 * 3 * 7 * 2)


class ClassifyMinimalPairTests(unittest.TestCase):
    """Exactly one field flips the class: pins the priority order."""

    EOS = 99

    @staticmethod
    def _obs(**overrides: Any) -> dict[str, Any]:
        """Obs."""
        obs: dict[str, Any] = {
            "exact_match": False,
            "boundary_found": False,
            "boundary_text_match": False,
            "status": "completed",
            "raw_reason": "limit",
            "output_ids": (1, 2),
        }
        obs.update(overrides)
        return obs

    def test_failed_status_beats_eos_signal(self):
        """Test failed status beats eos signal."""
        base = self._obs(output_ids=(1, self.EOS, 2))
        self.assertEqual(classify_observation(base, self.EOS), "eos-without-boundary")
        flipped = self._obs(output_ids=(1, self.EOS, 2), status="failed")
        self.assertEqual(classify_observation(flipped, self.EOS), "backend-error")

    def test_cancelled_groups_with_backend_error(self):
        """Test cancelled groups with backend error."""
        base = self._obs(output_ids=(1, self.EOS, 2))
        self.assertEqual(classify_observation(base, self.EOS), "eos-without-boundary")
        flipped = self._obs(output_ids=(1, self.EOS, 2), status="cancelled")
        self.assertEqual(classify_observation(flipped, self.EOS), "backend-error")

    def test_raw_reason_error_alone_is_backend_error(self):
        """Test raw reason error alone is backend error."""
        base = self._obs(output_ids=(1, self.EOS, 2))
        self.assertEqual(classify_observation(base, self.EOS), "eos-without-boundary")
        flipped = self._obs(output_ids=(1, self.EOS, 2), raw_reason="error")
        self.assertEqual(classify_observation(flipped, self.EOS), "backend-error")

    def test_failed_status_beats_text_mismatch(self):
        """Test failed status beats text mismatch."""
        base = self._obs(boundary_text_match=True)
        self.assertEqual(classify_observation(base, self.EOS), "boundary-text-mismatch")
        flipped = self._obs(boundary_text_match=True, status="failed")
        self.assertEqual(classify_observation(flipped, self.EOS), "backend-error")

    def test_cancelled_beats_text_mismatch(self):
        """Test cancelled beats text mismatch."""
        flipped = self._obs(boundary_text_match=True, status="cancelled")
        self.assertEqual(classify_observation(flipped, self.EOS), "backend-error")

    def test_text_mismatch_beats_eos_signal(self):
        """Test text mismatch beats eos signal."""
        base = self._obs(output_ids=(1, self.EOS, 2))
        self.assertEqual(classify_observation(base, self.EOS), "eos-without-boundary")
        flipped = self._obs(output_ids=(1, self.EOS, 2), boundary_text_match=True)
        self.assertEqual(classify_observation(flipped, self.EOS), "boundary-text-mismatch")

    def test_eos_id_none_downgrades_to_no_boundary(self):
        """Test eos id none downgrades to no boundary."""
        base = self._obs(output_ids=(1, self.EOS, 2))
        self.assertEqual(classify_observation(base, self.EOS), "eos-without-boundary")
        self.assertEqual(classify_observation(base, None), "no-boundary-in-N")

    def test_eos_absent_from_ids_is_no_boundary(self):
        """Test eos absent from ids is no boundary."""
        base = self._obs(output_ids=(1, 2))
        self.assertEqual(classify_observation(base, self.EOS), "no-boundary-in-N")

    def test_pass_requires_both_flags(self):
        """Test pass requires both flags."""
        # boundary found but answer wrong -> not a pass, and not a mismatch:
        # the fallthrough records no-boundary-in-N.
        obs = self._obs(boundary_found=True)
        self.assertEqual(classify_observation(obs, self.EOS), "no-boundary-in-N")
        passing = self._obs(boundary_found=True, exact_match=True)
        self.assertIsNone(classify_observation(passing, self.EOS))

    def test_pass_beats_failed_status(self):
        """Test pass beats failed status."""
        # Contradictory record (claims pass AND failed status): the docstring
        # contract is "None when it passed".
        obs = self._obs(exact_match=True, boundary_found=True, status="failed")
        self.assertIsNone(classify_observation(obs, self.EOS))

    def test_truthy_flags_count_as_pass(self):
        """Test truthy flags count as pass."""
        obs = self._obs(exact_match=1, boundary_found=1)
        self.assertIsNone(classify_observation(obs, self.EOS))

    def test_boundary_text_match_must_be_exactly_true(self):
        """Test boundary text match must be exactly true."""
        # `is True`, not truthiness: a truthy non-True flag does not diagnose
        # a tokenization mismatch.
        for flag in (1, "yes", [True]):
            with self.subTest(flag=flag):
                obs = self._obs(boundary_text_match=flag)
                self.assertEqual(classify_observation(obs, self.EOS), "no-boundary-in-N")


class ClassifyAdversarialTests(unittest.TestCase):
    """Empty / contradictory / huge / missing / wrong-typed observations."""

    EOS = 99

    def test_empty_observation(self):
        """Test empty observation."""
        self.assertEqual(classify_observation({}, None), "no-boundary-in-N")

    def test_empty_observation_with_eos_id(self):
        """Test empty observation with eos id."""
        self.assertEqual(classify_observation({}, self.EOS), "no-boundary-in-N")

    def test_empty_output_ids(self):
        """Test empty output ids."""
        obs = {"boundary_found": False, "status": "completed", "output_ids": ()}
        self.assertEqual(classify_observation(obs, self.EOS), "no-boundary-in-N")

    def test_falsy_output_id_containers(self):
        """Test falsy output id containers."""
        for ids in (None, (), [], "", 0, False):
            with self.subTest(ids=ids):
                obs = {"boundary_found": False, "status": "completed", "output_ids": ids}
                self.assertEqual(classify_observation(obs, self.EOS), "no-boundary-in-N")

    def test_contradictory_stop_text_claim_without_stop_ids(self):
        """Test contradictory stop text claim without stop ids."""
        # The classifier trusts the stored boundary_text_match flag even when
        # no ids are present: the flag is the measurement, ids are context.
        obs = {
            "boundary_found": False,
            "status": "completed",
            "boundary_text_match": True,
            "output_ids": (),
        }
        self.assertEqual(classify_observation(obs, self.EOS), "boundary-text-mismatch")

    def test_contradictory_boundary_claim_with_passing_flags(self):
        """Test contradictory boundary claim with passing flags."""
        # exact+boundary true but no boundary ids at all: still a pass --
        # classification reads the record, it does not re-derive it.
        obs = {"exact_match": True, "boundary_found": True, "output_ids": ()}
        self.assertIsNone(classify_observation(obs, self.EOS))

    def test_unicode_fields_never_leak_into_codes(self):
        """Test unicode fields never leak into codes."""
        obs = {
            "exact_match": False,
            "boundary_found": False,
            "status": "failed",
            "error": "weird unicode ❌ 日本語🔥",
            "answer": "日本語の答え",
            "output_ids": (1, 2),
        }
        self.assertEqual(classify_observation(obs, self.EOS), "backend-error")
        self.assertIn(classify_observation(obs, self.EOS), ALL_MODES)

    def test_huge_observation_with_late_eos(self):
        """Test huge observation with late eos."""
        ids = tuple(range(100_000)) + (self.EOS,)
        obs = {"boundary_found": False, "status": "completed", "output_ids": ids}
        self.assertEqual(classify_observation(obs, self.EOS), "eos-without-boundary")

    def test_huge_observation_without_eos(self):
        """Test huge observation without eos."""
        ids = tuple(i for i in range(100_000) if i != self.EOS)
        obs = {"boundary_found": False, "status": "completed", "output_ids": ids}
        self.assertEqual(classify_observation(obs, self.EOS), "no-boundary-in-N")

    def test_non_dict_raises_valueerror(self):
        """Test non dict raises valueerror."""
        for bad in ("nope", None, 123, [("exact_match", True)], b"bytes"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                classify_observation(cast(Any, bad), None)

    def test_truthy_non_iterable_output_ids_raises_typeerror(self):
        """Test truthy non iterable output ids raises typeerror."""
        # Documented contract: a truthy non-iterable output_ids raises
        # TypeError from the EOS-membership check.
        obs = {"boundary_found": False, "status": "completed", "output_ids": 123}
        with self.assertRaises(TypeError):
            classify_observation(obs, self.EOS)

    def test_truthy_non_iterable_short_circuits_on_pass(self):
        """Test truthy non iterable short circuits on pass."""
        obs = {"exact_match": True, "boundary_found": True, "output_ids": 123}
        self.assertIsNone(classify_observation(obs, self.EOS))

    def test_truthy_non_iterable_short_circuits_on_backend_failure(self):
        """Test truthy non iterable short circuits on backend failure."""
        obs = {"status": "failed", "output_ids": 123}
        self.assertEqual(classify_observation(obs, self.EOS), "backend-error")

    def test_extra_keys_ignored(self):
        """Test extra keys ignored."""
        obs = {
            "exact_match": False,
            "boundary_found": False,
            "status": "completed",
            "output_ids": (1, 2),
            "surprise": object(),
            "nested": {"deep": ["junk"]},
        }
        self.assertEqual(classify_observation(obs, self.EOS), "no-boundary-in-N")


def _obs_pass() -> dict[str, Any]:
    """Obs pass."""
    return {
        "exact_match": True,
        "boundary_found": True,
        "status": "completed",
        "output_ids": (1, 2),
    }


def _obs_failed() -> dict[str, Any]:
    """Obs failed."""
    return {"exact_match": False, "boundary_found": False, "status": "failed", "output_ids": ()}


def _obs_mismatch() -> dict[str, Any]:
    """Obs mismatch."""
    return {
        "exact_match": False,
        "boundary_found": False,
        "status": "completed",
        "boundary_text_match": True,
        "output_ids": (1, 2),
    }


def _obs_eos(eos: int = 99) -> dict[str, Any]:
    """Obs eos."""
    return {
        "exact_match": False,
        "boundary_found": False,
        "status": "completed",
        "boundary_text_match": False,
        "output_ids": (1, eos),
    }


def _obs_no_boundary() -> dict[str, Any]:
    """Obs no boundary."""
    return {
        "exact_match": False,
        "boundary_found": False,
        "status": "completed",
        "boundary_text_match": False,
        "output_ids": (1, 2),
    }


def _entry(name: Any, observations: Any) -> dict[str, Any]:
    """Entry."""
    return {"format": {"name": name}, "observations": observations}


class FailureTaxonomyTests(unittest.TestCase):
    """Hand-computed aggregations: every count is derived by hand below."""

    EOS = 99

    def test_hand_computed_two_candidates(self):
        """Test hand computed two candidates."""
        result = {
            "search": [
                _entry(
                    "A",
                    [
                        _obs_pass(),
                        _obs_failed(),
                        _obs_mismatch(),
                        _obs_eos(self.EOS),
                        _obs_no_boundary(),
                    ],
                ),
                _entry("B", [_obs_pass(), _obs_pass()]),
            ],
            "decision": "no_candidate",
        }
        tax = failure_taxonomy(result, self.EOS)
        self.assertEqual(tax["modes"], list(FAILURE_MODES))
        self.assertEqual(
            tax["per_candidate"]["A"],
            {
                "no-boundary-in-N": 1,
                "boundary-text-mismatch": 1,
                "eos-without-boundary": 1,
                "held-out-divergence": 0,
                "backend-error": 1,
            },
        )
        self.assertEqual(sum(tax["per_candidate"]["A"].values()), 4)
        self.assertEqual(tax["per_candidate"]["B"], dict.fromkeys(FAILURE_MODES, 0))
        self.assertIsNone(tax["held_out"])
        self.assertTrue(tax["eos_id_considered"])

    def test_eos_id_none_reclassifies_eos_obs(self):
        """Test eos id none reclassifies eos obs."""
        # Same observations, eos_id=None: the eos observation drops into
        # no-boundary-in-N and the flag reports the eos id was not considered.
        result = {
            "search": [_entry("A", [_obs_eos(self.EOS), _obs_pass()])],
            "decision": "no_candidate",
        }
        tax = failure_taxonomy(result, None)
        self.assertEqual(tax["per_candidate"]["A"]["eos-without-boundary"], 0)
        self.assertEqual(tax["per_candidate"]["A"]["no-boundary-in-N"], 1)
        self.assertFalse(tax["eos_id_considered"])

    def test_empty_input(self):
        """Test empty input."""
        tax = failure_taxonomy({"search": []}, None)
        self.assertEqual(tax["modes"], list(FAILURE_MODES))
        self.assertEqual(tax["per_candidate"], {})
        self.assertIsNone(tax["held_out"])
        self.assertFalse(tax["eos_id_considered"])

    def test_missing_search_key(self):
        """Test missing search key."""
        tax = failure_taxonomy({"decision": "no_candidate"}, self.EOS)
        self.assertEqual(tax["per_candidate"], {})
        self.assertTrue(tax["eos_id_considered"])

    def test_all_failed_input(self):
        """Test all failed input."""
        result = {
            "search": [_entry("A", [_obs_failed(), _obs_failed(), _obs_failed()])],
            "decision": "failed",
        }
        tax = failure_taxonomy(result, self.EOS)
        counts = tax["per_candidate"]["A"]
        self.assertEqual(counts["backend-error"], 3)
        self.assertEqual(sum(counts.values()), 3)

    def test_all_pass_candidate_gets_zero_counts(self):
        """Test all pass candidate gets zero counts."""
        result = {"search": [_entry("A", [_obs_pass(), _obs_pass()])], "decision": "ambiguous"}
        tax = failure_taxonomy(result, self.EOS)
        self.assertEqual(tax["per_candidate"]["A"], dict.fromkeys(FAILURE_MODES, 0))

    def test_non_dict_search_entry_skipped(self):
        """Test non dict search entry skipped."""
        result = {
            "search": ["junk", None, 42, _entry("A", [_obs_failed()])],
            "decision": "no_candidate",
        }
        tax = failure_taxonomy(result, self.EOS)
        self.assertEqual(list(tax["per_candidate"]), ["A"])

    def test_entry_without_str_name_skipped(self):
        """Test entry without str name skipped."""
        result = {
            "search": [
                {"format": {"name": None}, "observations": [_obs_failed()]},
                {"format": {"name": 42}, "observations": [_obs_failed()]},
                {"format": "not-a-dict", "observations": [_obs_failed()]},
                {"observations": [_obs_failed()]},
                _entry("A", [_obs_failed()]),
            ],
            "decision": "no_candidate",
        }
        tax = failure_taxonomy(result, self.EOS)
        self.assertEqual(list(tax["per_candidate"]), ["A"])

    def test_non_list_observations_gives_zero_counts(self):
        """Test non list observations gives zero counts."""
        for bad_obs in (None, "junk", {"not": "a-list"}, 42):
            with self.subTest(bad_obs=bad_obs):
                result = {"search": [_entry("A", bad_obs)], "decision": "no_candidate"}
                tax = failure_taxonomy(result, self.EOS)
                self.assertEqual(tax["per_candidate"]["A"], dict.fromkeys(FAILURE_MODES, 0))

    def test_non_dict_observation_propagates_valueerror(self):
        """Test non dict observation propagates valueerror."""
        result = {"search": [_entry("A", [_obs_pass(), "junk"])], "decision": "no_candidate"}
        with self.assertRaises(ValueError):
            failure_taxonomy(result, self.EOS)

    def test_result_non_dict_raises(self):
        """Test result non dict raises."""
        for bad in ("nope", None, [1, 2]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                failure_taxonomy(cast(Any, bad), None)

    def test_held_out_divergence_counts_every_non_passing_dict(self):
        """Test held out divergence counts every non passing dict."""
        result = {
            "search": [_entry("A", [_obs_pass()])],
            "decision": "held_out_failed",
            "held_out": {
                "observations": [
                    _obs_pass(),  # passes: not counted
                    _obs_failed(),  # non-passing dict: counted
                    _obs_no_boundary(),  # non-passing dict: counted
                    "junk",  # non-dict: silently skipped
                    None,  # non-dict: silently skipped
                ]
            },
        }
        tax = failure_taxonomy(result, self.EOS)
        held = tax["held_out"]
        assert held is not None
        self.assertEqual(held["held-out-divergence"], 2)
        self.assertEqual(set(held), ALL_MODES)
        # Search-stage classification is untouched by the held-out stage.
        self.assertEqual(tax["per_candidate"]["A"], dict.fromkeys(FAILURE_MODES, 0))

    def test_held_out_absent_without_divergence_decision(self):
        """Test held out absent without divergence decision."""
        for decision in ("compatible_on_fixture_set", "ambiguous", "no_candidate", "failed"):
            with self.subTest(decision=decision):
                result = {
                    "search": [_entry("A", [_obs_pass()])],
                    "decision": decision,
                    "held_out": {"observations": [_obs_failed()]},
                }
                self.assertIsNone(failure_taxonomy(result, self.EOS)["held_out"])

    def test_held_out_failed_without_held_dict_gives_zero_counts(self):
        """Test held out failed without held dict gives zero counts."""
        # Deliberate asymmetry: held-out counts exist (zeroed) whenever the
        # decision says the stage diverged, even with no held-out records.
        result = {"search": [_entry("A", [_obs_pass()])], "decision": "held_out_failed"}
        held = failure_taxonomy(result, self.EOS)["held_out"]
        assert held is not None
        self.assertEqual(held, dict.fromkeys(FAILURE_MODES, 0))

    def test_held_out_failed_with_non_dict_held(self):
        """Test held out failed with non dict held."""
        result = {
            "search": [_entry("A", [_obs_pass()])],
            "decision": "held_out_failed",
            "held_out": "junk",
        }
        held = failure_taxonomy(result, self.EOS)["held_out"]
        assert held is not None
        self.assertEqual(held, dict.fromkeys(FAILURE_MODES, 0))

    def test_counts_keys_cover_all_modes(self):
        """Test counts keys cover all modes."""
        result = {"search": [_entry("A", [_obs_pass()])], "decision": "no_candidate"}
        tax = failure_taxonomy(result, self.EOS)
        self.assertEqual(set(tax["per_candidate"]["A"]), ALL_MODES)


class ScriptBackend:
    """Total deterministic decode: per-token texts joined, never raises."""

    fingerprint = "script-backend-not-a-model"

    def __init__(self, texts: dict[int, str] | None = None):
        self._texts = dict(texts) if texts else {}

    def decode(self, token_ids: Any) -> str:
        """Decode."""
        return "".join(self._texts.get(t, f"<{t}>") for t in tuple(token_ids))

    @staticmethod
    def encode(text: str) -> tuple[int, ...]:
        """Encode."""
        raise AssertionError("encode unused in classification-audit tests")

    @staticmethod
    def generate(*args: Any, **kwargs: Any) -> Any:
        """Generate."""
        raise AssertionError("generate unused in classification-audit tests")


def _audit_fixture(fid: str = "f1", expected: str = "ab") -> Fixture:
    """Audit fixture."""
    return Fixture(fid, (Message("user", "q-" + fid),), expected, "group-" + fid)


class StopCellDifferentialTests(unittest.TestCase):
    """Oracle-vs-implementation over token sequences x boundaries x outcomes."""

    def test_differential_over_sequences(self):
        """Test differential over sequences."""
        backend = ScriptBackend({1: "a", 2: "b", 3: "c"})
        fixture = _audit_fixture()
        alphabet = (1, 2, 3)
        seqs: list[tuple[int, ...]] = [()]
        for length in (1, 2, 3):
            seqs.extend(itertools.product(alphabet, repeat=length))
        boundaries = [(2,), (3,), (1, 2), (3, 2)]
        stop_texts: list[str | None] = [None, "kept", "🔥kept"]
        total = 0
        for seq, boundary, fired, available, stop_text in itertools.product(
            seqs, boundaries, (False, True), (False, True), stop_texts
        ):
            outcome = StopOutcome(available, fired, seq, "synthetic_detail")
            total += 1
            with self.subTest(
                seq=seq, boundary=boundary, fired=fired, available=available, stop_text=stop_text
            ):
                expected = oracle_stop_cell(
                    backend, fixture, boundary, outcome, stop_text=stop_text
                )
                actual = _stop_cell(backend, fixture, boundary, outcome, stop_text=stop_text)
                self.assertEqual(actual, expected)
                if not available:
                    self.assertEqual(set(actual), {"available", "detail"})
                else:
                    self.assertIn("stop_text", actual)
        self.assertEqual(total, len(seqs) * len(boundaries) * 2 * 2 * len(stop_texts))


class StopCellFixedTests(unittest.TestCase):
    """Decode degradation, stop-text disposition, pass-rule sharp edges."""

    def test_unavailable_outcome_passes_through(self):
        """Test unavailable outcome passes through."""
        outcome = StopOutcome(False, False, (1, 2), "decode_unavailable")
        cell = _stop_cell(ScriptBackend(), _audit_fixture(), (2,), outcome, stop_text="kept")
        self.assertEqual(cell, {"available": False, "detail": "decode_unavailable"})

    def test_decode_exception_degrades_cell(self):
        """Test decode exception degrades cell."""

        class RaisingDecode(ScriptBackend):
            """Raising decode."""

            @staticmethod
            def decode(token_ids: Any) -> str:
                """Decode."""
                raise RuntimeError("synthetic decode failure")

        outcome = StopOutcome(True, True, (1, 2), "fired_at_step_2")
        cell = _stop_cell(RaisingDecode(), _audit_fixture(), (2,), outcome)
        self.assertEqual(cell, {"available": False, "detail": "answer_decode_unavailable"})

    def test_decode_non_string_degrades_cell(self):
        """Test decode non string degrades cell."""

        class NonStringDecode(ScriptBackend):
            """Non string decode."""

            @staticmethod
            def decode(token_ids: Any) -> Any:
                """Decode."""
                return 12345

        outcome = StopOutcome(True, True, (1, 2), "fired_at_step_2")
        cell = _stop_cell(NonStringDecode(), _audit_fixture(), (2,), outcome)
        self.assertEqual(cell, {"available": False, "detail": "answer_decode_unavailable"})

    def test_stop_text_none_when_not_fired(self):
        """Test stop text none when not fired."""
        outcome = StopOutcome(True, False, (1, 2), "no_stop_within_cap")
        cell = _stop_cell(
            ScriptBackend({1: "a", 2: "b"}), _audit_fixture(), (3,), outcome, stop_text="kept"
        )
        self.assertTrue(cell["available"])
        self.assertIsNone(cell["stop_text"])
        # The pass rule still applies: boundary (3,) never occurs here.
        self.assertFalse(cell["pass"])

    def test_stop_text_passes_through_when_fired(self):
        """Test stop text passes through when fired."""
        outcome = StopOutcome(True, True, (1, 2), "fired_at_step_2")
        cell = _stop_cell(
            ScriptBackend({1: "a", 2: "b"}),
            _audit_fixture(expected="a"),
            (2,),
            outcome,
            stop_text="stripped",
        )
        self.assertEqual(cell["stop_text"], "stripped")
        self.assertTrue(cell["pass"])

    def test_unicode_stop_text_preserved(self):
        """Test unicode stop text preserved."""
        outcome = StopOutcome(True, True, (1, 2), "fired_at_step_2")
        cell = _stop_cell(
            ScriptBackend({1: "a", 2: "b"}), _audit_fixture(), (2,), outcome, stop_text="🔥kept"
        )
        self.assertEqual(cell["stop_text"], "🔥kept")

    def test_pass_requires_boundary_even_when_answer_matches(self):
        """Test pass requires boundary even when answer matches."""
        # answer (1,) decodes to "a" == expected "a", but no boundary in the
        # truncated ids: the pass rule is boundary AND answer, not answer alone.
        outcome = StopOutcome(True, False, (1,), "no_stop_within_cap")
        cell = _stop_cell(ScriptBackend({1: "a"}), _audit_fixture(expected="a"), (2,), outcome)
        self.assertTrue(cell["available"])
        self.assertFalse(cell["pass"])

    def test_boundary_at_offset_zero(self):
        """Test boundary at offset zero."""
        # The boundary at offset 0 leaves an empty answer region; with an
        # empty expected answer the cell still passes.
        outcome = StopOutcome(True, True, (2, 1), "fired_at_step_2")
        cell = _stop_cell(
            ScriptBackend({1: "a", 2: "b"}), _audit_fixture(expected=""), (2,), outcome
        )
        self.assertTrue(cell["pass"])

    def test_first_boundary_occurrence_wins(self):
        """Test first boundary occurrence wins."""
        # (2,) occurs at offsets 1 and 3; cutting at the FIRST occurrence
        # leaves answer (1,) -> "a". Cutting at the last would give "aba".
        outcome = StopOutcome(True, True, (1, 2, 1, 2), "fired_at_step_4")
        cell = _stop_cell(
            ScriptBackend({1: "a", 2: "b"}), _audit_fixture(expected="a"), (2,), outcome
        )
        self.assertTrue(cell["pass"])


class TinyBackend(ScriptBackend):
    """Character vocabulary backend for stop-matrix audit cases.

    65->"A", 66->"B", 33->"!", 8364->"€"; anything else decodes as "<id>".
    """

    def __init__(self) -> None:
        super().__init__({65: "A", 66: "B", 33: "!", 8364: "€"})


def _matrix_observation(**overrides: Any) -> dict[str, Any]:
    """Matrix observation."""
    obs: dict[str, Any] = {
        "fixture": "f1",
        "input_ids": (),
        "output_ids": (65, 66, 33, 67),
        "exact_match": True,
        "boundary_found": True,
        "status": "completed",
    }
    obs.update(overrides)
    return obs


def _matrix_fixture() -> Fixture:
    """Matrix fixture."""
    # TinyBackend decodes (65, 66) as "AB".
    return _audit_fixture(expected="AB")


class StopMatrixHandDerivedTests(unittest.TestCase):
    """One fully hand-derived end-to-end case, pinned with literal values.

    Backend: 65->"A", 66->"B", 33->"!", boundary (33,) decodes to "!",
    fixture expects "AB", eos id 99. The emulation outcomes were confirmed
    against the documented emulation rules before pinning (HF fires when the
    stop string overlaps the final token's text; vLLM searches decoded
    prefixes; EOS truncates before the first EOS token).
    """

    def test_simple_case_all_agree_with_native(self):
        """Test simple case all agree with native."""
        rows = stop_matrix(
            TinyBackend(),
            "tiny",
            (_matrix_fixture(),),
            (_matrix_observation(),),
            boundary=(33,),
            eos_id=99,
        )
        (row,) = rows
        self.assertEqual(
            row,
            {
                "fixture": "f1",
                "native_pass": True,
                "cells": {
                    "hf_stop_string": {
                        "available": True,
                        "pass": True,
                        "fired": True,
                        "detail": "fired_at_step_3",
                        "stop_text": "kept",
                    },
                    "vllm_holdback": {
                        "available": True,
                        "pass": True,
                        "fired": True,
                        "detail": "fired_at_step_3",
                        "stop_text": "stripped",
                    },
                    "eos_appended": {
                        "available": True,
                        "pass": True,
                        "fired": False,
                        "detail": "no_stop_within_cap",
                        "stop_text": None,
                    },
                },
                "flips": [],
            },
        )

    def test_eos_and_stop_fire_flip_cells(self):
        """Test eos and stop fire flip cells."""
        # Output (65,66,99,33): the EOS at index 2 kills the eos cell; the
        # trailing "!" fires HF/vLLM AFTER the EOS token, so their answers
        # decode to "AB<99>" != "AB".
        rows = stop_matrix(
            TinyBackend(),
            "tiny",
            (_matrix_fixture(),),
            (_matrix_observation(output_ids=(65, 66, 99, 33)),),
            boundary=(33,),
            eos_id=99,
        )
        (row,) = rows
        self.assertTrue(row["native_pass"])
        cells = row["cells"]
        self.assertEqual(cells["eos_appended"]["detail"], "fired_at_step_3")
        self.assertTrue(cells["eos_appended"]["fired"])
        self.assertFalse(cells["eos_appended"]["pass"])
        self.assertIsNone(cells["eos_appended"]["stop_text"])
        self.assertTrue(cells["hf_stop_string"]["fired"])
        self.assertFalse(cells["hf_stop_string"]["pass"])
        self.assertEqual(cells["hf_stop_string"]["stop_text"], "kept")
        self.assertTrue(cells["vllm_holdback"]["fired"])
        self.assertFalse(cells["vllm_holdback"]["pass"])
        self.assertEqual(cells["vllm_holdback"]["stop_text"], "stripped")
        self.assertEqual(row["flips"], ["hf_stop_string", "vllm_holdback", "eos_appended"])

    def test_vllm_keep_flag_surfaces_kept_disposition(self):
        """Test vllm keep flag surfaces kept disposition."""
        rows = stop_matrix(
            TinyBackend(),
            "tiny",
            (_matrix_fixture(),),
            (_matrix_observation(),),
            boundary=(33,),
            eos_id=99,
            vllm_include_stop_str_in_output=True,
        )
        (row,) = rows
        self.assertEqual(row["cells"]["vllm_holdback"]["stop_text"], "kept")
        self.assertTrue(row["cells"]["vllm_holdback"]["fired"])
        self.assertEqual(row["flips"], [])

    def test_unicode_boundary_text(self):
        """Test unicode boundary text."""
        # Boundary id 8364 decodes to "€": the string cells must work in
        # non-ASCII text, not degrade.
        fixture = _audit_fixture(expected="A")
        rows = stop_matrix(
            TinyBackend(),
            "tiny",
            (fixture,),
            (
                _matrix_observation(
                    output_ids=(65, 8364, 67), exact_match=True, boundary_found=True
                ),
            ),
            boundary=(8364,),
            eos_id=99,
        )
        (row,) = rows
        hf = row["cells"]["hf_stop_string"]
        vllm = row["cells"]["vllm_holdback"]
        for cell in (hf, vllm):
            self.assertTrue(cell["available"])
            self.assertTrue(cell["fired"])
            self.assertTrue(cell["pass"])
            self.assertEqual(cell["detail"], "fired_at_step_2")
        self.assertEqual(hf["stop_text"], "kept")
        self.assertEqual(vllm["stop_text"], "stripped")
        self.assertEqual(row["flips"], [])

    def test_incomplete_observation_disables_string_cells(self):
        """Test incomplete observation disables string cells."""
        rows = stop_matrix(
            TinyBackend(),
            "tiny",
            (_matrix_fixture(),),
            (_matrix_observation(status="failed", exact_match=False, boundary_found=False),),
            boundary=(33,),
            eos_id=99,
        )
        (row,) = rows
        self.assertFalse(row["native_pass"])
        for semantic in STOP_SEMANTICS[1:]:
            with self.subTest(semantic=semantic):
                cell = row["cells"][semantic]
                self.assertEqual(cell, {"available": False, "detail": "observation_incomplete"})
        self.assertEqual(set(row["cells"]), set(STOP_SEMANTICS[1:]))
        self.assertEqual(row["flips"], [])

    def test_non_int_output_ids_are_incomplete(self):
        """Test non int output ids are incomplete."""
        rows = stop_matrix(
            TinyBackend(),
            "tiny",
            (_matrix_fixture(),),
            (_matrix_observation(output_ids=("a",), exact_match=False, boundary_found=False),),
            boundary=(33,),
        )
        (row,) = rows
        for semantic in STOP_SEMANTICS[1:]:
            with self.subTest(semantic=semantic):
                self.assertFalse(row["cells"][semantic]["available"])

    def test_native_is_not_a_cell(self):
        """Test native is not a cell."""
        # STOP_SEMANTICS[0] == "native" is a label, not a competitor cell.
        rows = stop_matrix(
            TinyBackend(),
            "tiny",
            (_matrix_fixture(),),
            (_matrix_observation(),),
            boundary=(33,),
        )
        self.assertNotIn("native", rows[0]["cells"])
        self.assertEqual(set(rows[0]["cells"]), set(STOP_SEMANTICS[1:]))


class StopMatrixWiringTests(unittest.TestCase):
    """Differential wiring: trusted emulation outcomes + oracle cells.

    The emulation functions are NOT the audit target; they supply the
    StopOutcome ground truth. The oracle then predicts each cell and the
    flip list, and stop_matrix must match -- this pins the wiring
    (dispositions, stop_text gating, flip definition) independently of the
    source's own composition.
    """

    @staticmethod
    def _predict(
        backend: Any,
        fixture: Fixture,
        boundary: tuple[int, ...],
        obs: dict[str, Any],
        eos_id: int | None,
        keep_vllm: bool = False,
    ) -> dict[str, Any]:
        """Predict."""
        output_ids = tuple(obs.get("output_ids") or ())
        prompt_ids = tuple(obs.get("input_ids") or ())
        complete = obs.get("status") == "completed" and all(type(t) is int for t in output_ids)
        native_pass = bool(obs.get("exact_match") and obs.get("boundary_found"))
        boundary_text = backend.decode(boundary)
        cells: dict[str, dict[str, Any]] = {}
        if complete:
            outcomes = {
                "hf_stop_string": emulate_hf_stop_string(
                    backend, prompt_ids, output_ids, boundary_text
                ),
                "vllm_holdback": emulate_vllm_holdback(
                    backend, output_ids, boundary_text, include_stop_str_in_output=keep_vllm
                ),
                "eos_appended": emulate_eos_stop(output_ids, eos_id),
            }
            dispositions = {
                "hf_stop_string": "kept",
                "vllm_holdback": "kept" if keep_vllm else "stripped",
                "eos_appended": None,
            }
            for semantic, outcome in outcomes.items():
                cells[semantic] = oracle_stop_cell(
                    backend, fixture, boundary, outcome, stop_text=dispositions[semantic]
                )
        else:
            for semantic in STOP_SEMANTICS[1:]:
                cells[semantic] = {"available": False, "detail": "observation_incomplete"}
        flips = [
            s
            for s in STOP_SEMANTICS[1:]
            if cells[s].get("available") and cells[s].get("pass") != native_pass
        ]
        return {"fixture": fixture.id, "native_pass": native_pass, "cells": cells, "flips": flips}

    def test_wiring_over_observation_shapes(self):
        """Test wiring over observation shapes."""
        backend = TinyBackend()
        fixture = _matrix_fixture()
        shapes = [
            _matrix_observation(),
            _matrix_observation(output_ids=(65, 66, 33)),
            _matrix_observation(output_ids=(65, 66, 99, 33)),
            _matrix_observation(
                output_ids=(65, 66, 33, 67), exact_match=False, boundary_found=True
            ),
            _matrix_observation(output_ids=(65, 66), exact_match=False, boundary_found=False),
            _matrix_observation(output_ids=(), exact_match=False, boundary_found=False),
            _matrix_observation(status="failed", exact_match=False, boundary_found=False),
            _matrix_observation(status="cancelled", exact_match=False, boundary_found=False),
            _matrix_observation(output_ids=(33, 65, 66)),
            _matrix_observation(output_ids=(65, 33, 99)),
        ]
        for index, obs in enumerate(shapes):
            for eos_id, keep in itertools.product((None, 99), (False, True)):
                with self.subTest(index=index, eos_id=eos_id, keep=keep):
                    expected = self._predict(backend, fixture, (33,), obs, eos_id, keep_vllm=keep)
                    (actual,) = stop_matrix(
                        backend,
                        "tiny",
                        (fixture,),
                        (obs,),
                        boundary=(33,),
                        eos_id=eos_id,
                        vllm_include_stop_str_in_output=keep,
                    )
                    self.assertEqual(actual, expected)


class StopMatrixValidationTests(unittest.TestCase):
    """Every documented input contract raises ValueError, never a traceback."""

    @staticmethod
    def _good_args() -> dict[str, Any]:
        """Good args."""
        return {
            "backend": TinyBackend(),
            "candidate_name": "tiny",
            "fixtures": (_matrix_fixture(),),
            "observations": (_matrix_observation(),),
            "boundary": (33,),
        }

    def _run(self, **overrides: Any) -> Any:
        """Run."""
        args = self._good_args()
        args.update(overrides)
        backend = args.pop("backend")
        candidate_name = args.pop("candidate_name")
        fixtures = args.pop("fixtures")
        observations = args.pop("observations")
        boundary = args.pop("boundary")
        return stop_matrix(
            backend, candidate_name, fixtures, observations, boundary=boundary, **args
        )

    def test_rejects_bad_inputs(self):
        """Test rejects bad inputs."""
        cases: list[tuple[str, dict[str, Any]]] = [
            ("blank name", {"candidate_name": ""}),
            ("non-str name", {"candidate_name": 42}),
            ("fixtures not tuple", {"fixtures": [_audit_fixture()]}),
            ("observations not tuple", {"observations": [_matrix_observation()]}),
            ("length mismatch", {"observations": (_matrix_observation(), _matrix_observation())}),
            ("non-fixture", {"fixtures": ({"id": "f1"},)}),
            ("non-dict observation", {"observations": ("junk",)}),
            ("empty boundary", {"boundary": ()}),
            ("boundary not tuple", {"boundary": [33]}),
            ("negative boundary id", {"boundary": (-1,)}),
            ("non-int boundary id", {"boundary": ("33",)}),
            ("negative eos", {"eos_id": -1}),
            ("non-int eos", {"eos_id": "99"}),
            ("non-callable token_bytes", {"token_bytes": 42}),
            ("non-bool vllm flag", {"vllm_include_stop_str_in_output": 1}),
            ("negative min tokens", {"vllm_min_tokens": -1}),
            ("non-int min tokens", {"vllm_min_tokens": 1.5}),
        ]
        for label, overrides in cases:
            with self.subTest(label=label), self.assertRaises(ValueError):
                self._run(**overrides)

    def test_fixture_observation_alignment_mismatch(self):
        """Test fixture observation alignment mismatch."""
        with self.assertRaises(ValueError):
            self._run(observations=(_matrix_observation(fixture="other"),))

    def test_undecodable_boundary_disables_string_cells(self):
        """Test undecodable boundary disables string cells."""

        class NoBoundaryText(TinyBackend):
            """No boundary text."""

            def decode(self, token_ids: Any) -> Any:
                """Decode."""
                if tuple(token_ids) == (33,):
                    return ""
                return super().decode(token_ids)

        rows = self._run(backend=NoBoundaryText(), eos_id=99)
        (row,) = rows
        self.assertFalse(row["cells"]["hf_stop_string"]["available"])
        self.assertFalse(row["cells"]["vllm_holdback"]["available"])
        # The token-id cell does not need the boundary text.
        self.assertTrue(row["cells"]["eos_appended"]["available"])


if __name__ == "__main__":
    unittest.main()
