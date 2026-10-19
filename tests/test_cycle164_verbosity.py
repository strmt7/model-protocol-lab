"""Cycle 164: response verbosity profile per template (wave-4 idea 5).

Tests the pure ``verbosity`` core (brevity-ideal filter, excess
arithmetic, pinned type-7 quantiles, per-candidate aggregation), the
post-hoc stage (fail-closed observation handling, fail-fast contract
violations), the profile analysis (gaps, most/least verbose behavior
labels, capped-share unconfounding vs the stop-cause audit), the
200-case brute-force oracle differential (Fraction-exact independent
reimplementation), Hypothesis properties, 4/4 tempdir mutation gates,
single-cause ``changed`` pins for every diff disjunct, hostile-input
sanitization (``<script>`` candidate names, terminal escapes in the
stderr summary), the CLI probe, and the TOC anchor -- plus extensions
to both diff oracles (cycle-27 availability mirror, cycle-67 full
oracle + fuzz).
"""

import importlib.util
import os
import random
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from threading import Event
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from model_protocol_lab import diffreports
from model_protocol_lab.core import Format, Generation
from model_protocol_lab.experiments import verbosity_profile, verbosity_stage
from model_protocol_lab.verbosity import (
    BREVITY_IDEAL_MAX_CHARS,
    candidate_verbosity,
    excess_tokens,
    is_brevity_ideal,
    length_distribution,
)


class VerbosityScriptedBackend:
    """Ord-codec backend; generate() is overridden per test via policy."""

    fingerprint = "scripted-verbosity-c164"

    def __init__(self, lengths: dict[str, int] | None = None):
        # maps candidate name -> emitted token count
        self.lengths = lengths or {}

    @staticmethod
    def encode(text: str) -> tuple[int, ...]:
        return tuple(map(ord, text))

    @staticmethod
    def decode(token_ids: tuple[int, ...]) -> str:
        return "".join(map(chr, token_ids))

    def generate(
        self,
        input_ids: tuple[int, ...],
        *,
        max_new_tokens: int,
        seed: int,
        cancel: Event,
    ) -> Generation:
        prompt = self.decode(tuple(input_ids))
        # candidate name travels in the prompt via the format marker
        name = prompt.split("\x00")[-1] if "\x00" in prompt else "?"
        n = self.lengths.get(name, max_new_tokens)
        out = tuple(map(ord, "A".ljust(n)[:n]))
        return Generation(token_ids=out, reason="limit")


def _format(name: str) -> Format:
    return Format(
        name=name,
        bos="",
        user_prefix="[user]\n",
        assistant_prefix="[asst]\n\x00" + name,
        turn_end="",
    )


def _obs(candidate: str, fixture: str, n_tokens: int, status: str = "completed") -> dict[str, Any]:
    return {
        "candidate": candidate,
        "fixture": fixture,
        "output_ids": tuple(range(n_tokens)),
        "status": status,
        "answer": "x",
    }


def _expected() -> dict[str, str]:
    return {"f-blue": "BLUE", "f-green": "GREEN", "f-long": "x" * 100}


class BrevityIdealTests(unittest.TestCase):
    def test_short_answers_qualify(self):
        self.assertTrue(is_brevity_ideal("BLUE"))
        self.assertTrue(is_brevity_ideal("7"))
        self.assertTrue(is_brevity_ideal("  SOUTH  "))
        self.assertTrue(is_brevity_ideal("x" * BREVITY_IDEAL_MAX_CHARS))

    def test_long_answer_rejected(self):
        self.assertFalse(is_brevity_ideal("x" * (BREVITY_IDEAL_MAX_CHARS + 1)))

    def test_empty_and_whitespace_rejected(self):
        self.assertFalse(is_brevity_ideal(""))
        self.assertFalse(is_brevity_ideal("   "))

    def test_multiline_rejected(self):
        self.assertFalse(is_brevity_ideal("a\nb"))
        self.assertFalse(is_brevity_ideal("a\rb"))

    def test_non_string_rejected(self):
        self.assertFalse(is_brevity_ideal(None))
        self.assertFalse(is_brevity_ideal(123))
        self.assertFalse(is_brevity_ideal(b"BLUE"))


class ExcessTokensTests(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(excess_tokens(16, 4), 12)
        self.assertEqual(excess_tokens(4, 4), 0)

    def test_clamped_at_zero(self):
        # Under-generation is a different failure mode (graded by
        # exact_match); verbosity never reports negative excess.
        self.assertEqual(excess_tokens(0, 4), 0)
        self.assertEqual(excess_tokens(3, 10), 0)

    def test_rejects_bad_input(self):
        for bad in (-1, 1.5, "4", True, None):
            with self.assertRaises(ValueError, msg=f"generated={bad!r}"):
                excess_tokens(bad, 4)
            with self.assertRaises(ValueError, msg=f"minimal={bad!r}"):
                excess_tokens(4, bad)


class LengthDistributionTests(unittest.TestCase):
    def test_empty(self):
        dist = length_distribution([])
        self.assertEqual(dist["n"], 0)
        for key in (
            "median_excess",
            "p25_excess",
            "p75_excess",
            "iqr_excess",
            "mean_excess",
            "max_excess",
        ):
            self.assertIsNone(dist[key])

    def test_single(self):
        dist = length_distribution([7])
        self.assertEqual(dist["n"], 1)
        self.assertEqual(dist["median_excess"], 7.0)
        self.assertEqual(dist["p25_excess"], 7.0)
        self.assertEqual(dist["p75_excess"], 7.0)
        self.assertEqual(dist["iqr_excess"], 0.0)
        self.assertEqual(dist["mean_excess"], 7.0)
        self.assertEqual(dist["max_excess"], 7)

    def test_odd_count(self):
        # [0, 10, 20]: type-7 quartiles -> p25=5, median=10, p75=15
        dist = length_distribution([20, 0, 10])
        self.assertEqual(dist["median_excess"], 10.0)
        self.assertEqual(dist["p25_excess"], 5.0)
        self.assertEqual(dist["p75_excess"], 15.0)
        self.assertEqual(dist["iqr_excess"], 10.0)
        self.assertEqual(dist["mean_excess"], 10.0)
        self.assertEqual(dist["max_excess"], 20)

    def test_even_count_interpolates(self):
        # [0, 10]: median=(0+10)/2=5, p25=2.5, p75=7.5
        dist = length_distribution([10, 0])
        self.assertEqual(dist["median_excess"], 5.0)
        self.assertEqual(dist["p25_excess"], 2.5)
        self.assertEqual(dist["p75_excess"], 7.5)
        self.assertEqual(dist["iqr_excess"], 5.0)

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            length_distribution("not a list")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            length_distribution([1, -2])
        with self.assertRaises(ValueError):
            length_distribution([1, 2.5])


class CandidateVerbosityTests(unittest.TestCase):
    def test_aggregation(self):
        cell = candidate_verbosity([(16, 4, True), (10, 5, False), (4, 4, False)])
        self.assertEqual(cell["n"], 3)
        self.assertEqual(cell["n_evaluable"], 3)
        # excesses: 12, 5, 0 -> median 5.0
        self.assertEqual(cell["median_excess"], 5.0)
        self.assertEqual(cell["max_excess"], 12)
        self.assertAlmostEqual(cell["capped_share"], 1 / 3)

    def test_malformed_records_excluded_fail_closed(self):
        cell = candidate_verbosity(
            [
                (16, 4, True),
                ("x", 4, False),  # bad generated
                (10, -1, False),  # negative minimal
                (10, 5, "yes"),  # non-bool capped
                (10,),  # wrong arity
                "nope",  # not a tuple
            ]
        )
        self.assertEqual(cell["n"], 6)
        self.assertEqual(cell["n_evaluable"], 1)
        self.assertEqual(cell["median_excess"], 12.0)

    def test_empty(self):
        cell = candidate_verbosity([])
        self.assertEqual(cell["n"], 0)
        self.assertEqual(cell["n_evaluable"], 0)
        self.assertIsNone(cell["median_excess"])
        self.assertIsNone(cell["capped_share"])

    def test_rejects_non_list(self):
        with self.assertRaises(ValueError):
            candidate_verbosity("nope")  # type: ignore[arg-type]


class VerbosityStageTests(unittest.TestCase):
    def test_posthoc_measurement(self):
        backend = VerbosityScriptedBackend()
        stage = verbosity_stage(
            observations=[
                _obs("a", "f-blue", 4),
                _obs("a", "f-green", 9),
                _obs("b", "f-blue", 16),
                _obs("b", "f-green", 16),
            ],
            expected_by_fixture=_expected(),
            backend=backend,
            max_new_tokens=16,
        )
        self.assertEqual(stage["n_observations"], 4)
        self.assertEqual(stage["n_fixtures"], 3)
        self.assertEqual(stage["n_brevity_fixtures"], 2)  # f-long excluded
        a = stage["per_candidate"]["a"]
        # excesses: 4-4=0, 9-5=4 -> median 2.0
        self.assertEqual(a["median_excess"], 2.0)
        self.assertEqual(a["n_evaluable"], 2)
        self.assertEqual(a["capped_share"], 0.0)
        b = stage["per_candidate"]["b"]
        self.assertEqual(b["median_excess"], 11.5)
        self.assertEqual(b["capped_share"], 1.0)

    def test_non_brevity_fixtures_excluded(self):
        backend = VerbosityScriptedBackend()
        stage = verbosity_stage(
            observations=[_obs("a", "f-long", 16)],
            expected_by_fixture=_expected(),
            backend=backend,
            max_new_tokens=16,
        )
        cell = stage["per_candidate"]["a"]
        self.assertEqual(cell["n"], 1)
        self.assertEqual(cell["n_evaluable"], 0)
        self.assertIsNone(cell["median_excess"])

    def test_failed_status_excluded(self):
        backend = VerbosityScriptedBackend()
        stage = verbosity_stage(
            observations=[_obs("a", "f-blue", 4, status="failed")],
            expected_by_fixture=_expected(),
            backend=backend,
            max_new_tokens=16,
        )
        cell = stage["per_candidate"]["a"]
        self.assertEqual(cell["n"], 1)
        self.assertEqual(cell["n_evaluable"], 0)

    def test_malformed_observations_fail_closed(self):
        backend = VerbosityScriptedBackend()
        stage = verbosity_stage(
            observations=[
                _obs("a", "f-blue", 4),
                {
                    "candidate": "a",
                    "fixture": "f-blue",
                    "output_ids": "xx",
                    "status": "completed",
                },
                {
                    "candidate": "a",
                    "fixture": "f-blue",
                    "output_ids": (1, -2),
                    "status": "completed",
                },
                {
                    "candidate": "a",
                    "fixture": "f-blue",
                    "output_ids": (True,),
                    "status": "completed",
                },
                {
                    "candidate": "",
                    "fixture": "f-blue",
                    "output_ids": (1,),
                    "status": "completed",
                },
                {
                    "candidate": "a",
                    "fixture": "f-missing",
                    "output_ids": (1,),
                    "status": "completed",
                },
                "not a dict",
            ],
            expected_by_fixture=_expected(),
            backend=backend,
            max_new_tokens=16,
        )
        cell = stage["per_candidate"]["a"]
        # 5 observations attributed to "a" (the empty-candidate and
        # non-dict ones are unattributable); only the first is
        # evaluable.
        self.assertEqual(cell["n"], 5)
        self.assertEqual(cell["n_evaluable"], 1)

    def test_contract_violations_raise(self):
        backend = VerbosityScriptedBackend()
        good_obs = [_obs("a", "f-blue", 4)]
        good_exp = _expected()
        with self.assertRaises(ValueError):
            verbosity_stage(
                observations="nope",  # type: ignore[arg-type]
                expected_by_fixture=good_exp,
                backend=backend,
                max_new_tokens=16,
            )
        with self.assertRaises(ValueError):
            verbosity_stage(
                observations=good_obs,
                expected_by_fixture={1: "x"},  # type: ignore[dict-item] -- keys must be str
                backend=backend,
                max_new_tokens=16,
            )
        with self.assertRaises(ValueError):
            verbosity_stage(
                observations=good_obs,
                expected_by_fixture=good_exp,
                backend=None,  # type: ignore[arg-type]
                max_new_tokens=16,
            )
        with self.assertRaises(ValueError):
            verbosity_stage(
                observations=good_obs,
                expected_by_fixture=good_exp,
                backend=backend,
                max_new_tokens=0,
            )

    def test_backend_without_encode_raises(self):
        class NoEncode:
            pass

        with self.assertRaises(ValueError):
            verbosity_stage(
                observations=[_obs("a", "f-blue", 4)],
                expected_by_fixture=_expected(),
                backend=NoEncode(),  # type: ignore[arg-type]
                max_new_tokens=16,
            )

    def test_backend_bad_encode_raises(self):
        class BadEncode(VerbosityScriptedBackend):
            @staticmethod
            def encode(text: str):
                return "not ids"  # type: ignore[return-value]

        with self.assertRaises(ValueError):
            verbosity_stage(
                observations=[_obs("a", "f-blue", 4)],
                expected_by_fixture=_expected(),
                backend=BadEncode(),
                max_new_tokens=16,
            )


def _stage_two_candidates() -> dict[str, Any]:
    backend = VerbosityScriptedBackend()
    return verbosity_stage(
        observations=[
            _obs("a", "f-blue", 4),
            _obs("a", "f-green", 9),
            _obs("b", "f-blue", 16),
            _obs("b", "f-green", 16),
        ],
        expected_by_fixture=_expected(),
        backend=backend,
        max_new_tokens=16,
    )


class VerbosityProfileTests(unittest.TestCase):
    def test_gap_and_extremes(self):
        profile = verbosity_profile(("a", "b"), _stage_two_candidates())
        self.assertTrue(profile["available"])
        self.assertEqual(profile["n_brevity_fixtures"], 2)
        self.assertEqual(profile["median_excess_gap"], 9.5)
        self.assertEqual(profile["most_verbose"], ["b"])
        self.assertEqual(profile["least_verbose"], ["a"])
        self.assertFalse(profile["uniform"])
        self.assertEqual(profile["capped_candidates"], ["b"])
        self.assertEqual(profile["excluded_candidates"], [])

    def test_uniform(self):
        backend = VerbosityScriptedBackend()
        stage = verbosity_stage(
            observations=[_obs("a", "f-blue", 8), _obs("b", "f-blue", 8)],
            expected_by_fixture=_expected(),
            backend=backend,
            max_new_tokens=16,
        )
        profile = verbosity_profile(("a", "b"), stage)
        self.assertTrue(profile["uniform"])
        self.assertEqual(profile["median_excess_gap"], 0.0)
        # tied extremes name every tied candidate: behavior labels,
        # never a quality judgment
        self.assertEqual(profile["most_verbose"], ["a", "b"])
        self.assertEqual(profile["least_verbose"], ["a", "b"])

    def test_excluded_candidates(self):
        backend = VerbosityScriptedBackend()
        stage = verbosity_stage(
            observations=[_obs("a", "f-blue", 8)],
            expected_by_fixture=_expected(),
            backend=backend,
            max_new_tokens=16,
        )
        profile = verbosity_profile(("a", "b"), stage)
        self.assertEqual(profile["excluded_candidates"], ["b"])
        self.assertIsNone(profile["per_candidate"]["b"])
        self.assertEqual(profile["median_excess_gap"], 0.0)  # single evaluable candidate

    def test_invalid_input_unavailable(self):
        stage = _stage_two_candidates()
        self.assertFalse(verbosity_profile((), stage)["available"])
        self.assertFalse(verbosity_profile(("a", "a"), stage)["available"])
        self.assertFalse(verbosity_profile("ab", stage)["available"])  # type: ignore[arg-type]
        self.assertFalse(verbosity_profile(("a",), "nope")["available"])  # type: ignore[arg-type]
        bad = dict(stage)
        bad["per_candidate"] = {"a": {"n": "x"}}
        self.assertEqual(verbosity_profile(("a",), bad)["detail"], "stage_invalid")


# ---------------------------------------------------------------------------
# Brute-force oracle differential (200 cases, Fraction-exact)
# ---------------------------------------------------------------------------


def _oracle_distribution(excesses: list[int]) -> dict[str, Any]:
    """Independent reimplementation with Fraction-exact arithmetic.

    Sorts via a different path (keyed sort on negated values, then
    reversed) and computes type-7 quantiles with Fraction positions,
    so float rounding in the implementation cannot hide.
    """
    ordered = sorted((-v, v) for v in excesses)
    ordered = [v for _, v in reversed(ordered)]
    n = len(ordered)
    if n == 0:
        return {
            "n": 0,
            "median_excess": None,
            "p25_excess": None,
            "p75_excess": None,
            "iqr_excess": None,
            "mean_excess": None,
            "max_excess": None,
        }

    def q(frac: Fraction) -> float:
        pos = frac * (n - 1)
        low = int(pos)  # floor for non-negative
        high = low + (1 if pos != low else 0)
        if high == low:
            return float(ordered[low])
        weight = pos - low
        return float(ordered[low] + weight * (ordered[high] - ordered[low]))

    p25 = q(Fraction(1, 4))
    median = q(Fraction(1, 2))
    p75 = q(Fraction(3, 4))
    return {
        "n": n,
        "median_excess": median,
        "p25_excess": p25,
        "p75_excess": p75,
        "iqr_excess": p75 - p25,
        "mean_excess": float(sum(Fraction(v) for v in ordered) / n),
        "max_excess": ordered[-1],
    }


class VerbosityOracleTests(unittest.TestCase):
    def test_200_case_differential(self):
        rng = random.Random(20261019)  # noqa: S311 -- deterministic test seed
        for trial in range(200):
            n = rng.randint(0, 25)
            excesses = [rng.randint(0, 200) for _ in range(n)]
            got = length_distribution(excesses)
            want = _oracle_distribution(excesses)
            self.assertEqual(got["n"], want["n"], f"trial {trial}")
            for key in (
                "median_excess",
                "p25_excess",
                "p75_excess",
                "iqr_excess",
                "mean_excess",
                "max_excess",
            ):
                if want[key] is None:
                    self.assertIsNone(got[key], f"trial {trial} {key}")
                else:
                    self.assertAlmostEqual(
                        got[key], want[key], places=9, msg=f"trial {trial} {key}"
                    )

    def test_excess_nonnegative_oracle(self):
        rng = random.Random(77)  # noqa: S311 -- deterministic test seed
        for _ in range(200):
            generated = rng.randint(0, 500)
            minimal = rng.randint(0, 500)
            self.assertGreaterEqual(excess_tokens(generated, minimal), 0)
            self.assertEqual(excess_tokens(generated, minimal), max(0, generated - minimal))


# ---------------------------------------------------------------------------
# Hypothesis properties
# ---------------------------------------------------------------------------


class VerbosityPropertyTests(unittest.TestCase):
    @given(st.lists(st.integers(min_value=0, max_value=10**6), max_size=50))
    @settings(max_examples=100)
    def test_distribution_properties(self, excesses):
        dist = length_distribution(excesses)
        self.assertEqual(dist["n"], len(excesses))
        if not excesses:
            return
        lo, hi = min(excesses), max(excesses)
        for key in ("median_excess", "p25_excess", "p75_excess", "mean_excess"):
            self.assertGreaterEqual(dist[key], lo)
            self.assertLessEqual(dist[key], hi)
        self.assertLessEqual(dist["p25_excess"], dist["median_excess"])
        self.assertLessEqual(dist["median_excess"], dist["p75_excess"])
        self.assertGreaterEqual(dist["iqr_excess"], 0)
        self.assertEqual(dist["max_excess"], hi)

    @given(st.integers(min_value=0, max_value=10**6), st.integers(min_value=0, max_value=10**6))
    @settings(max_examples=100)
    def test_excess_clamped(self, generated, minimal):
        self.assertGreaterEqual(excess_tokens(generated, minimal), 0)

    @given(st.text())
    @settings(max_examples=200)
    def test_brevity_ideal_bounded(self, text):
        if is_brevity_ideal(text):
            stripped = text.strip()
            self.assertTrue(stripped)
            self.assertLessEqual(len(stripped), BREVITY_IDEAL_MAX_CHARS)
            self.assertNotIn("\n", stripped)


# ---------------------------------------------------------------------------
# Tempdir mutation gates (4/4)
# ---------------------------------------------------------------------------


class VerbosityMutationGateTests(unittest.TestCase):
    def _run_gate(self, module_name, name, old_text, new_text, scenario):
        mod = __import__(f"model_protocol_lab.{module_name}", fromlist=["x"])
        source = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertEqual(source.count(old_text), 1, f"mutant {name}: anchor not unique")
        mutated = source.replace(old_text, new_text)
        with tempfile.TemporaryDirectory(prefix="c164_mut_", dir="/tmp") as tmpdir:
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

    def _clamp_scenario(self, mod):
        # Under-generation must clamp to zero, never go negative.
        result = mod.excess_tokens(3, 10)
        assert result == 0, result

    def test_mutant_clamp_removed_caught(self):
        self._run_gate(
            "verbosity",
            "clamp_removed",
            "    return generated - minimal if generated > minimal else 0",
            "    return generated - minimal  # MUTANT",
            self._clamp_scenario,
        )

    def _quantile_scenario(self, mod):
        # Even-n median must interpolate, not snap to the lower value.
        dist = mod.length_distribution([0, 10])
        assert dist["median_excess"] == 5.0, dist

    def test_mutant_quantile_no_interpolation_caught(self):
        self._run_gate(
            "verbosity",
            "quantile_no_interpolation",
            "    return sorted_values[low] + frac * (sorted_values[high] - sorted_values[low])",
            "    return float(sorted_values[low])  # MUTANT",
            self._quantile_scenario,
        )

    def _capped_boundary_scenario(self, mod):
        backend = VerbosityScriptedBackend()
        stage = mod.verbosity_stage(
            observations=[_obs("a", "f-blue", 16)],
            expected_by_fixture=_expected(),
            backend=backend,
            max_new_tokens=16,
        )
        cell = stage["per_candidate"]["a"]
        # generated == max_new_tokens saturates the budget.
        assert cell["capped_share"] == 1.0, cell

    def test_mutant_capped_strict_caught(self):
        self._run_gate(
            "experiments",
            "capped_strict",
            "            (generated, minimal[fixture_id], generated >= max_new_tokens)",
            "            (generated, minimal[fixture_id], generated > max_new_tokens)  # MUTANT",
            self._capped_boundary_scenario,
        )

    def _iqr_drift_scenario(self, mod):
        old = _vb_diff_payload()
        new = _vb_diff_payload()
        # IQR widens within valid quartile ordering: only iqr_drift fires.
        new["per_candidate"]["c"]["p25_excess"] = 0.5
        new["per_candidate"]["c"]["p75_excess"] = 4.0
        new["per_candidate"]["c"]["iqr_excess"] = 3.5
        diff = mod.diff_verbosity(old, new)
        assert diff["available"] == "both", diff
        assert diff["changed"] is True, diff
        assert diff["iqr_drift"], diff
        assert not diff["median_drift"], diff
        assert not diff["capped_share_drift"], diff

    def test_mutant_diff_iqr_drift_ignored_caught(self):
        self._run_gate(
            "diffreports",
            "iqr_drift_ignored",
            "        or verbosity_iqr_drift\n",
            "        # or verbosity_iqr_drift  # MUTANT\n",
            self._iqr_drift_scenario,
        )


def _vb_entry(**overrides):
    entry = {
        "n": 4,
        "n_evaluable": 4,
        "median_excess": 2.0,
        "p25_excess": 1.0,
        "p75_excess": 3.0,
        "iqr_excess": 2.0,
        "mean_excess": 2.0,
        "max_excess": 5,
        "capped_share": 0.0,
    }
    entry.update(overrides)
    return entry


def _vb_diff_payload():
    return {
        "available": True,
        "n_fixtures": 3,
        "n_brevity_fixtures": 2,
        "per_candidate": {"c": _vb_entry(), "d": _vb_entry()},
        "most_verbose": [],
        "least_verbose": [],
        "capped_candidates": [],
        "excluded_candidates": [],
    }


class DiffVerbosityTests(unittest.TestCase):
    def test_identical_payloads_unchanged(self):
        diff = diffreports.diff_verbosity(_vb_diff_payload(), _vb_diff_payload())
        self.assertEqual(diff["available"], "both")
        self.assertFalse(diff["changed"])
        self.assertEqual(diff["median_drift"], [])
        self.assertEqual(diff["candidates_added"], [])

    def test_median_drift_single_cause(self):
        old, new = _vb_diff_payload(), _vb_diff_payload()
        # Median moves within the pinned quartiles (2.0 -> 2.5).
        new["per_candidate"]["c"]["median_excess"] = 2.5
        diff = diffreports.diff_verbosity(old, new)
        self.assertTrue(diff["changed"])
        self.assertEqual(len(diff["median_drift"]), 1)
        self.assertEqual(diff["median_drift"][0]["candidate"], "c")
        self.assertEqual(diff["median_drift"][0]["old"], 2.0)
        self.assertEqual(diff["median_drift"][0]["new"], 2.5)
        self.assertFalse(diff["iqr_drift"])
        self.assertFalse(diff["capped_share_drift"])

    def test_iqr_drift_single_cause(self):
        old, new = _vb_diff_payload(), _vb_diff_payload()
        new["per_candidate"]["d"]["iqr_excess"] = 7.5
        diff = diffreports.diff_verbosity(old, new)
        self.assertTrue(diff["changed"])
        self.assertEqual(len(diff["iqr_drift"]), 1)
        self.assertFalse(diff["median_drift"])
        self.assertFalse(diff["capped_share_drift"])

    def test_capped_share_drift_single_cause(self):
        old, new = _vb_diff_payload(), _vb_diff_payload()
        new["per_candidate"]["c"]["capped_share"] = 1.0
        diff = diffreports.diff_verbosity(old, new)
        self.assertTrue(diff["changed"])
        self.assertEqual(len(diff["capped_share_drift"]), 1)
        self.assertFalse(diff["median_drift"])
        self.assertFalse(diff["iqr_drift"])

    def test_candidates_added_removed(self):
        old, new = _vb_diff_payload(), _vb_diff_payload()
        new["per_candidate"]["e"] = _vb_entry()
        diff = diffreports.diff_verbosity(old, new)
        self.assertTrue(diff["changed"])
        self.assertEqual(diff["candidates_added"], ["e"])
        old2, new2 = _vb_diff_payload(), _vb_diff_payload()
        del new2["per_candidate"]["d"]
        diff2 = diffreports.diff_verbosity(old2, new2)
        self.assertTrue(diff2["changed"])
        self.assertEqual(diff2["candidates_removed"], ["d"])

    def test_excluded_set_changes(self):
        old, new = _vb_diff_payload(), _vb_diff_payload()
        new["excluded_candidates"] = ["c"]
        diff = diffreports.diff_verbosity(old, new)
        self.assertTrue(diff["changed"])
        self.assertEqual(diff["excluded_added"], ["c"])
        self.assertEqual(diff["excluded_removed"], [])

    def test_asymmetric_availability(self):
        diff = diffreports.diff_verbosity(_vb_diff_payload(), {"available": False})
        self.assertEqual(diff["available"], "old")
        self.assertFalse(diff["changed"])
        neither = diffreports.diff_verbosity({"available": False}, {"available": False})
        self.assertEqual(neither["available"], "neither")

    def test_hostile_payload_rejected(self):
        hostile = _vb_diff_payload()
        hostile["per_candidate"]["c"]["median_excess"] = "huge"
        diff = diffreports.diff_verbosity(hostile, _vb_diff_payload())
        self.assertEqual(diff["available"], "new")  # hostile old side degrades
        self.assertFalse(diff["changed"])

    def test_none_cell_supported(self):
        old, new = _vb_diff_payload(), _vb_diff_payload()
        old["per_candidate"]["c"] = None
        new["per_candidate"]["c"] = _vb_entry()
        diff = diffreports.diff_verbosity(old, new)
        self.assertTrue(diff["changed"])
        self.assertEqual(len(diff["median_drift"]), 1)
        self.assertIsNone(diff["median_drift"][0]["old"])

    def test_quartile_ordering_rejected(self):
        hostile = _vb_diff_payload()
        hostile["per_candidate"]["c"]["p25_excess"] = 9.0  # > median
        diff = diffreports.diff_verbosity(hostile, _vb_diff_payload())
        self.assertEqual(diff["available"], "new")
        self.assertFalse(diff["changed"])


class VerbositySanitizationTests(unittest.TestCase):
    def _result_with_evil_name(self):
        backend = VerbosityScriptedBackend()
        stage = verbosity_stage(
            observations=[_obs("evil<script>", "f-blue", 16)],
            expected_by_fixture=_expected(),
            backend=backend,
            max_new_tokens=16,
        )
        return {"verbosity": verbosity_profile(("evil<script>",), stage)}

    def test_script_bearing_candidate_escaped_in_html(self):
        from model_protocol_lab import reporting

        html = reporting._verbosity_section(self._result_with_evil_name())
        self.assertIn('id="verbosity"', html)
        self.assertNotIn("evil<script>", html)
        self.assertIn("evil&lt;script&gt;", html)

    def test_terminal_escapes_stripped_from_stderr_summary(self):
        old, new = _vb_diff_payload(), _vb_diff_payload()
        evil_name = "cand\x1b[31mRED"
        old["per_candidate"][evil_name] = _vb_entry()
        new["per_candidate"][evil_name] = _vb_entry(median_excess=2.5)
        diff = diffreports.diff_verbosity(old, new)
        lines = diffreports._render_verbosity_summary(diff)
        joined = "\n".join(lines)
        self.assertIn("median-excess", joined)
        self.assertNotIn("\x1b[31m", joined)

    def test_unavailable_summary_line(self):
        diff = diffreports.diff_verbosity({"available": False}, _vb_diff_payload())
        lines = diffreports._render_verbosity_summary(diff)
        self.assertEqual(len(lines), 1)
        self.assertIn("Verbosity comparison unavailable", lines[0])
        self.assertEqual(diffreports._render_verbosity_summary({"available": "neither"}), [])
        self.assertEqual(diffreports._render_verbosity_summary(None), [])


class VerbosityCliTests(unittest.TestCase):
    def _search_entries(self, names=("a",)):
        return [
            {
                "format": {"name": name},
                "observations": [_obs(name, "f-blue", 8)],
            }
            for name in names
        ]

    def test_probe_records_available(self):
        from model_protocol_lab import cli
        from model_protocol_lab.core import Fixture, Message

        fixtures = (
            Fixture(
                id="f-blue",
                messages=(Message(role="user", content="x"),),
                expected="BLUE",
                group="g",
            ),
        )
        result = {"search": self._search_entries()}
        cli._run_verbosity_probe(
            result,
            (_format("a"),),
            backend=VerbosityScriptedBackend(),
            search_fixtures=fixtures,
            max_new_tokens=16,
        )
        self.assertTrue(result["verbosity"]["available"])
        self.assertEqual(result["verbosity"]["per_candidate"]["a"]["median_excess"], 4.0)

    def test_probe_never_raises_on_hostile_result(self):
        from model_protocol_lab import cli

        for bad_result in ({}, {"search": "nope"}, None):
            result = bad_result if type(bad_result) is dict else {}
            cli._run_verbosity_probe(
                result,
                (_format("a"),),
                backend=VerbosityScriptedBackend(),
                search_fixtures=(),
                max_new_tokens=16,
            )
            self.assertFalse(result["verbosity"]["available"])
            self.assertEqual(result["verbosity"]["detail"], "verbosity_unavailable")

    def test_probe_empty_observations_reports_excluded(self):
        # No observations is not a failure: the probe ran, there is
        # just nothing to profile -- the candidate is excluded, honestly.
        from model_protocol_lab import cli

        result: dict = {"search": [{"observations": "nope"}]}
        cli._run_verbosity_probe(
            result,
            (_format("a"),),
            backend=VerbosityScriptedBackend(),
            search_fixtures=(),
            max_new_tokens=16,
        )
        self.assertTrue(result["verbosity"]["available"])
        self.assertEqual(result["verbosity"]["excluded_candidates"], ["a"])

    def test_probe_never_raises_on_backend_failure(self):
        from model_protocol_lab import cli
        from model_protocol_lab.core import Fixture, Message

        class BoomBackend(VerbosityScriptedBackend):
            @staticmethod
            def encode(text: str):
                raise RuntimeError("boom")

        fixtures = (
            Fixture(
                id="f-blue",
                messages=(Message(role="user", content="x"),),
                expected="BLUE",
                group="g",
            ),
        )
        result = {"search": self._search_entries()}
        cli._run_verbosity_probe(
            result,
            (_format("a"),),
            backend=BoomBackend(),
            search_fixtures=fixtures,
            max_new_tokens=16,
        )
        self.assertFalse(result["verbosity"]["available"])

    def test_flag_registered(self):
        from model_protocol_lab import cli as cli_module

        parser = cli_module.build_parser()
        args = parser.parse_args(
            [
                "run",
                "--verbosity",
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
            ]
        )
        self.assertTrue(args.verbosity)


class VerbosityReportIntegrationTests(unittest.TestCase):
    def test_toc_anchor_present(self):
        from model_protocol_lab import reporting

        result = {"verbosity": verbosity_profile(("a", "b"), _stage_two_candidates())}
        html = reporting._verbosity_section(result)
        self.assertIn('id="verbosity"', html)
        self.assertIn("Response verbosity profile", html)
        # honest framing: behavior profile, never a quality judgment
        self.assertIn("not a", html)
        self.assertIn("quality judgment", html)

    def test_section_empty_when_unavailable(self):
        from model_protocol_lab import reporting

        self.assertEqual(reporting._verbosity_section({}), "")
        self.assertEqual(reporting._verbosity_section({"verbosity": {"available": False}}), "")

    def test_full_diff_reports_verbosity_section(self):
        def report():
            return {
                "decision": "select_best",
                "selected_candidate": "c",
                "search": [],
                "verbosity": _vb_diff_payload(),
            }

        diff = diffreports.diff_reports(report(), report())
        self.assertIn("verbosity", diff)
        self.assertEqual(diff["verbosity"]["available"], "both")
        self.assertFalse(diff["summary"]["verbosity_changed"])
        self.assertFalse(diff["summary"]["verbosity_asymmetric"])
        self.assertTrue(diff["summary"]["identical"])

    def test_full_diff_reports_verbosity_changed_breaks_identical(self):
        def report(payload):
            return {
                "decision": "select_best",
                "selected_candidate": "c",
                "search": [],
                "verbosity": payload,
            }

        new_payload = _vb_diff_payload()
        new_payload["per_candidate"]["c"]["median_excess"] = 2.5
        diff = diffreports.diff_reports(report(_vb_diff_payload()), report(new_payload))
        self.assertTrue(diff["summary"]["verbosity_changed"])
        self.assertFalse(diff["summary"]["identical"])
        stderr_lines = diffreports.render_summary(diff)
        self.assertIn("Verbosity", stderr_lines)


class VerbosityCoverageGapTests(unittest.TestCase):
    """Defensive fail-closed branches in the verbosity wiring.

    The main test classes exercise the happy paths and the differential
    oracles; these pin the validator/profile/section rejection branches
    that keep hostile payloads from reaching the renderers.
    """

    def test_cell_valid_none_and_non_dict(self):
        from model_protocol_lab.experiments import _verbosity_cell_valid

        self.assertTrue(_verbosity_cell_valid(None))
        for bad in ("x", 42, ["n"]):
            self.assertFalse(_verbosity_cell_valid(bad))

    def test_cell_valid_bad_counts(self):
        from model_protocol_lab.experiments import _verbosity_cell_valid

        for bad in (-1, True, "4", 2.5):
            self.assertFalse(_verbosity_cell_valid(_vb_entry(n=bad)), bad)
            self.assertFalse(_verbosity_cell_valid(_vb_entry(n_evaluable=bad)), bad)
        # Valid ints but n_evaluable > n is structurally impossible.
        self.assertFalse(_verbosity_cell_valid(_vb_entry(n=2, n_evaluable=3)))

    def test_cell_valid_bad_stat_value(self):
        from model_protocol_lab.experiments import _verbosity_cell_valid

        self.assertFalse(_verbosity_cell_valid(_vb_entry(median_excess="x")))
        self.assertFalse(_verbosity_cell_valid(_vb_entry(p25_excess=-1)))

    def test_cell_valid_bad_max_excess_and_share(self):
        from model_protocol_lab.experiments import _verbosity_cell_valid

        self.assertFalse(_verbosity_cell_valid(_vb_entry(max_excess=-1)))
        self.assertFalse(_verbosity_cell_valid(_vb_entry(max_excess="5")))
        for bad in ("0", -0.1, 1.5, True):
            self.assertFalse(_verbosity_cell_valid(_vb_entry(capped_share=bad)), bad)

    def test_cell_valid_zero_evaluable(self):
        from model_protocol_lab.experiments import _verbosity_cell_valid

        empty = dict.fromkeys(
            (
                "median_excess",
                "p25_excess",
                "p75_excess",
                "iqr_excess",
                "mean_excess",
                "max_excess",
                "capped_share",
            )
        )
        empty.update(n=2, n_evaluable=0)
        self.assertTrue(_verbosity_cell_valid(empty))
        partial = dict(empty)
        partial["median_excess"] = 1.0
        self.assertFalse(_verbosity_cell_valid(partial))

    def test_profile_stage_invalid_branches(self):
        from model_protocol_lab.experiments import verbosity_profile

        base = {
            "available": True,
            "n_fixtures": 3,
            "n_brevity_fixtures": 2,
            "per_candidate": {"c": _vb_entry()},
        }
        for key, bad in [
            ("n_fixtures", "3"),
            ("n_fixtures", -1),
            ("n_brevity_fixtures", -1),
            ("n_brevity_fixtures", 4),
            ("n_brevity_fixtures", "2"),
            ("per_candidate", ["c"]),
            ("per_candidate", {"c": _vb_entry(), 42: _vb_entry()}),
        ]:
            stage = dict(base)
            stage[key] = bad
            prof = verbosity_profile(("c",), stage)
            self.assertFalse(prof["available"], (key, bad))
            self.assertEqual(prof["detail"], "stage_invalid")

    def test_profile_non_dict_entry_is_stage_invalid(self):
        from model_protocol_lab.experiments import verbosity_profile

        stage = {
            "available": True,
            "n_fixtures": 3,
            "n_brevity_fixtures": 2,
            "per_candidate": {"c": "nope"},
        }
        prof = verbosity_profile(("c",), stage)
        self.assertFalse(prof["available"])
        self.assertEqual(prof["detail"], "stage_invalid")

    def test_diff_valid_verbosity_rejections(self):
        from model_protocol_lab import diffreports

        base = _vb_diff_payload()
        p = dict(base)
        p["n_fixtures"] = "3"
        self.assertFalse(diffreports._valid_verbosity(p))
        p = dict(base)
        p["n_brevity_fixtures"] = -1
        self.assertFalse(diffreports._valid_verbosity(p))
        p = dict(base)
        p["per_candidate"] = {"c": "nope"}
        self.assertFalse(diffreports._valid_verbosity(p))
        for overrides in (
            {"n": -1},
            {"n": "4"},
            {"n_evaluable": 5},
            {"max_excess": -1},
            {"max_excess": "5"},
            {"capped_share": 2.0},
        ):
            p = dict(base)
            p["per_candidate"] = {"c": _vb_entry(**overrides)}
            self.assertFalse(diffreports._valid_verbosity(p), overrides)
        # None entry is skipped, not a rejection.
        p = dict(base)
        p["per_candidate"] = {"c": None}
        self.assertTrue(diffreports._valid_verbosity(p))

    def test_cli_probe_skips_non_dict_search_entry(self):
        from model_protocol_lab import cli

        result: dict = {"search": ["nope", {"observations": []}]}
        cli._run_verbosity_probe(
            result,
            (_format("a"),),
            backend=VerbosityScriptedBackend(),
            search_fixtures=(),
            max_new_tokens=16,
        )
        self.assertTrue(result["verbosity"]["available"])
        self.assertEqual(result["verbosity"]["excluded_candidates"], ["a"])

    def test_reporting_section_degraded_paths(self):
        from model_protocol_lab import reporting

        self.assertEqual(reporting._verbosity_section({}), "")
        self.assertEqual(
            reporting._verbosity_section({"verbosity": {"available": False}}),
            "",
        )
        self.assertEqual(
            reporting._verbosity_section({"verbosity": "nope"}),
            "",
        )
        self.assertEqual(
            reporting._verbosity_section(
                {"verbosity": {"available": True, "per_candidate": "nope"}}
            ),
            "",
        )
        # n/a rendering for non-numeric stats and skipped non-dict entry.
        payload = _vb_diff_payload()
        payload["per_candidate"]["c"]["median_excess"] = "x"
        payload["per_candidate"]["c"]["capped_share"] = "y"
        payload["per_candidate"]["d"] = "nope"
        payload["median_excess_gap"] = "x"
        html = reporting._verbosity_section({"verbosity": payload})
        self.assertIn("n/a", html)
        self.assertIn('id="verbosity"', html)


if __name__ == "__main__":
    unittest.main()
