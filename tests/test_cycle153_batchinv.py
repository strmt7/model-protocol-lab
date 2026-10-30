"""Cycle 153: batch-invariance audit (research wave-3 idea 2).

Tests the pure comparison core (batchinv.py), the experiments.batch_stage
fail-closed contract, the CLI probe wiring, the HTML section escaping,
and the diffreports section. Follows the established per-cycle pattern:
brute-force oracle differentials, Hypothesis properties, tempdir-module
mutation gates, hostile-payload red-team.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import random
import re
import tempfile
import unittest
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from model_protocol_lab import core, diffreports, reporting
from model_protocol_lab.batchinv import (
    audit_verdict,
    compare_outputs,
    compare_solo_vs_batched,
    validate_prompt_ids,
)
from model_protocol_lab.experiments import batch_stage


def _ref_compare_outputs(solo_ids, batched_ids):
    """Independent reference implementation of compare_outputs."""
    assert type(solo_ids) is tuple and type(batched_ids) is tuple
    assert all(type(t) is int and t >= 0 for t in solo_ids + batched_ids)
    first = None
    for i, (a, b) in enumerate(zip(solo_ids, batched_ids, strict=False)):
        if a != b:
            first = i
            break
    else:
        if len(solo_ids) != len(batched_ids):
            first = min(len(solo_ids), len(batched_ids))
    return {
        "diverged": first is not None,
        "first_divergence": first,
        "solo_len": len(solo_ids),
        "batched_len": len(batched_ids),
    }


class ValidatePromptIdsTests(unittest.TestCase):
    """Validate prompt ids tests."""

    def test_valid_returns_dict_copy(self):
        """Test valid returns dict copy."""
        prompts = {"a": (1, 2), "b": (3,)}
        clean = validate_prompt_ids(prompts)
        self.assertEqual(clean, prompts)
        self.assertIsNot(prompts, clean)

    def test_rejects_non_dict(self):
        """Test rejects non dict."""
        for bad in (None, [], [("a", (1,))], "a"):
            with self.assertRaises(ValueError, msg=repr(bad)):
                validate_prompt_ids(bad)

    def test_rejects_empty(self):
        """Test rejects empty."""
        with self.assertRaises(ValueError):
            validate_prompt_ids({})

    def test_rejects_bad_keys(self):
        """Test rejects bad keys."""
        for bad_key in (7, None, "", "   "):
            with self.assertRaises(ValueError, msg=repr(bad_key)):
                validate_prompt_ids({bad_key: (1, 2)})

    def test_rejects_bad_ids(self):
        """Test rejects bad ids."""
        for bad_ids in (None, [], [1, 2], (), (1, -2), (1, True), ("x",)):
            with self.assertRaises(ValueError, msg=repr(bad_ids)):
                validate_prompt_ids({"a": bad_ids})


class CompareOutputsTests(unittest.TestCase):
    """Compare outputs tests."""

    def test_identical(self):
        """Test identical."""
        out = compare_outputs((1, 2, 3), (1, 2, 3))
        self.assertEqual(
            out,
            {"diverged": False, "first_divergence": None, "solo_len": 3, "batched_len": 3},
        )

    def test_empty_identical(self):
        """Test empty identical."""
        out = compare_outputs((), ())
        self.assertFalse(out["diverged"])
        self.assertIsNone(out["first_divergence"])

    def test_single_token_difference(self):
        """Test single token difference."""
        out = compare_outputs((1, 2, 3), (1, 9, 3))
        self.assertTrue(out["diverged"])
        self.assertEqual(out["first_divergence"], 1)

    def test_first_token_difference(self):
        """Test first token difference."""
        out = compare_outputs((1, 2), (7, 2))
        self.assertEqual(out["first_divergence"], 0)

    def test_strict_prefix_solo_shorter(self):
        """Test strict prefix solo shorter."""
        out = compare_outputs((1, 2), (1, 2, 3))
        self.assertTrue(out["diverged"])
        self.assertEqual(out["first_divergence"], 2)

    def test_strict_prefix_batched_shorter(self):
        """Test strict prefix batched shorter."""
        out = compare_outputs((1, 2, 3), (1, 2))
        self.assertTrue(out["diverged"])
        self.assertEqual(out["first_divergence"], 2)

    def test_malformed_raises(self):
        """Test malformed raises."""
        for bad in (None, [1, 2], (1, -1), (1, True), "ab"):
            with self.assertRaises(ValueError, msg=repr(bad)):
                compare_outputs(bad, (1, 2))
            with self.assertRaises(ValueError, msg=repr(bad)):
                compare_outputs((1, 2), bad)

    def test_brute_force_oracle_200_cases(self):
        """Test brute force oracle 200 cases."""
        rng = random.Random(20261018)  # noqa: S311 -- deterministic test seed
        for _ in range(200):
            n = rng.randint(0, 8)
            m = rng.randint(0, 8)
            solo = tuple(rng.randint(0, 500) for _ in range(n))
            batched = solo if rng.random() < 0.5 else tuple(rng.randint(0, 500) for _ in range(m))
            self.assertEqual(
                compare_outputs(solo, batched),
                _ref_compare_outputs(solo, batched),
                msg=f"solo={solo} batched={batched}",
            )


class CompareMapsTests(unittest.TestCase):
    """Compare maps tests."""

    def test_key_mismatch_raises(self):
        """Test key mismatch raises."""
        with self.assertRaises(ValueError):
            compare_solo_vs_batched({"a": (1,)}, {"a": (1,), "b": (2,)})
        with self.assertRaises(ValueError):
            compare_solo_vs_batched({"a": (1,)}, {"b": (1,)})

    def test_non_mapping_raises(self):
        """Test non mapping raises."""
        with self.assertRaises(ValueError):
            compare_solo_vs_batched([("a", (1,))], {"a": (1,)})

    def test_aggregates(self):
        """Test aggregates."""
        result = compare_solo_vs_batched(
            {"a": (1, 2), "b": (3,), "c": (4, 5)},
            {"a": (1, 2), "b": (9,), "c": (4, 5)},
        )
        self.assertEqual(result["n_prompts"], 3)
        self.assertEqual(result["n_diverged"], 1)
        self.assertAlmostEqual(result["divergence_rate"], 1 / 3)
        self.assertFalse(result["per_prompt"]["a"]["diverged"])
        self.assertTrue(result["per_prompt"]["b"]["diverged"])
        self.assertEqual(result["per_prompt"]["b"]["first_divergence"], 0)

    def test_brute_force_oracle_200_cases(self):
        """Test brute force oracle 200 cases."""
        rng = random.Random(20261019)  # noqa: S311 -- deterministic test seed
        for _ in range(200):
            n = rng.randint(1, 5)
            pids = [f"p{i}" for i in range(n)]
            solo, batched = {}, {}
            for pid in pids:
                length = rng.randint(0, 6)
                seq = tuple(rng.randint(0, 200) for _ in range(length))
                solo[pid] = seq
                if rng.random() < 0.6:
                    batched[pid] = seq
                else:
                    other = list(seq)
                    if other and rng.random() < 0.7:
                        other[rng.randrange(len(other))] ^= 1
                    else:
                        other.append(rng.randint(0, 200))
                    batched[pid] = tuple(other)
            result = compare_solo_vs_batched(solo, batched)
            expected_diverged = sum(
                1 for pid in pids if _ref_compare_outputs(solo[pid], batched[pid])["diverged"]
            )
            self.assertEqual(result["n_diverged"], expected_diverged)
            self.assertAlmostEqual(result["divergence_rate"], expected_diverged / n)
            for pid in pids:
                self.assertEqual(
                    result["per_prompt"][pid],
                    _ref_compare_outputs(solo[pid], batched[pid]),
                )


class AuditVerdictTests(unittest.TestCase):
    """Audit verdict tests."""

    def test_invariant(self):
        """Test invariant."""
        self.assertEqual(audit_verdict(4, 0), "batch_invariant")

    def test_variant(self):
        """Test variant."""
        self.assertEqual(audit_verdict(4, 1), "batch_variant")
        self.assertEqual(audit_verdict(4, 4), "batch_variant")

    def test_unavailable(self):
        """Test unavailable."""
        self.assertEqual(audit_verdict(0, 0), "unavailable")

    def test_bad_counts_raise(self):
        """Test bad counts raise."""
        for bad in ((-1, 0), (2, 3), (2, -1), ("2", 0), (2.0, 0)):
            with self.assertRaises(ValueError, msg=repr(bad)):
                audit_verdict(*bad)


class BatchinvPropertyTests(unittest.TestCase):
    """Batchinv property tests."""

    @given(
        st.lists(st.integers(min_value=0, max_value=1000), max_size=10),
        st.lists(st.integers(min_value=0, max_value=1000), max_size=10),
    )
    @settings(max_examples=150)
    def test_first_divergence_consistent(self, solo, batched):
        """Test first divergence consistent."""
        solo_t, batched_t = tuple(solo), tuple(batched)
        out = compare_outputs(solo_t, batched_t)
        if out["diverged"]:
            self.assertIsNotNone(out["first_divergence"])
            idx = out["first_divergence"]
            self.assertLessEqual(idx, min(len(solo_t), len(batched_t)))
            if idx < len(solo_t) and idx < len(batched_t):
                self.assertNotEqual(solo_t[idx], batched_t[idx])
        else:
            self.assertIsNone(out["first_divergence"])
            self.assertEqual(solo_t, batched_t)

    @given(
        st.dictionaries(
            st.text(min_size=1, max_size=8),
            st.tuples(
                st.integers(min_value=0, max_value=50), st.integers(min_value=0, max_value=50)
            ),
            min_size=1,
            max_size=6,
        )
    )
    @settings(max_examples=100)
    def test_identical_maps_zero_divergence(self, mapping):
        """Test identical maps zero divergence."""
        as_tuples = {k: tuple(v) for k, v in mapping.items()}
        result = compare_solo_vs_batched(as_tuples, dict(as_tuples))
        self.assertEqual(result["n_diverged"], 0)
        self.assertEqual(result["divergence_rate"], 0.0)


class BatchinvMutationGateTests(unittest.TestCase):
    """Tempdir-module mutation gates (the real tree is never rewritten)."""

    def _run_gate(self, module_name, name, old_text, new_text, scenario):
        """Run gate."""
        mod = __import__(f"model_protocol_lab.{module_name}", fromlist=["x"])
        source = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertEqual(source.count(old_text), 1, f"mutant {name}: anchor not unique")
        mutated = source.replace(old_text, new_text)
        with tempfile.TemporaryDirectory(prefix="c153_mut_", dir="/tmp") as tmpdir:
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
    def _diverged_predicate_scenario(mod):
        """Diverged predicate scenario."""
        assert mod.compare_outputs((1, 2), (1, 9))["diverged"] is True
        assert mod.compare_outputs((1, 2), (1, 2))["diverged"] is False

    def test_mutant_flipped_diverged_predicate_caught(self):
        """Test mutant flipped diverged predicate caught."""
        self._run_gate(
            "batchinv",
            "flipped_diverged",
            '        "diverged": first is not None,',
            '        "diverged": first is None,  # MUTANT',
            self._diverged_predicate_scenario,
        )

    @staticmethod
    def _key_set_scenario(mod):
        """Key set scenario."""
        try:
            mod.compare_solo_vs_batched({"a": (1,)}, {"b": (1,)})
        except ValueError:
            return  # specified behavior: key mismatch rejected
        except Exception as exc:  # noqa: BLE001 -- any other failure is also a detection
            raise AssertionError(
                f"mutant raised {type(exc).__name__} instead of ValueError"
            ) from exc
        raise AssertionError("key mismatch not rejected")

    def test_mutant_dropped_key_set_check_caught(self):
        """Test mutant dropped key set check caught."""
        self._run_gate(
            "batchinv",
            "dropped_keyset",
            "    if set(solo) != set(batched):\n"
            '        raise ValueError("solo and batched must cover the same prompt ids")',
            "    if False:  # MUTANT\n"
            '        raise ValueError("solo and batched must cover the same prompt ids")',
            self._key_set_scenario,
        )

    @staticmethod
    def _verdict_scenario(mod):
        """Verdict scenario."""
        assert mod.audit_verdict(3, 0) == "batch_invariant"
        assert mod.audit_verdict(3, 2) == "batch_variant"

    def test_mutant_flipped_verdict_caught(self):
        """Test mutant flipped verdict caught."""
        self._run_gate(
            "batchinv",
            "flipped_verdict",
            '    return "batch_variant" if n_diverged else "batch_invariant"',
            '    return "batch_invariant" if n_diverged else "batch_variant"  # MUTANT',
            self._verdict_scenario,
        )

    @staticmethod
    def _nonempty_scenario(mod):
        """Nonempty scenario."""
        try:
            mod.validate_prompt_ids({"a": ()})
        except ValueError:
            return
        raise AssertionError("empty prompt ids not rejected")

    def test_mutant_dropped_nonempty_check_caught(self):
        """Test mutant dropped nonempty check caught."""
        self._run_gate(
            "batchinv",
            "dropped_nonempty",
            "        if not _valid_token_ids(ids, nonempty=True):",
            "        if not _valid_token_ids(ids):  # MUTANT",
            self._nonempty_scenario,
        )


class _SoloBackend:
    """Minimal backend without a batch path."""

    fingerprint = "fake-solo"

    @staticmethod
    def encode(text):
        """Encode."""
        return tuple(map(ord, text))

    @staticmethod
    def decode(ids):
        """Decode."""
        return "".join(map(chr, ids))

    @staticmethod
    def generate(input_ids, *, max_new_tokens, seed, cancel):
        """Generate."""
        return core.Generation(tuple(range(50, 50 + max_new_tokens)), "limit")


class _BatchBackend(_SoloBackend):
    """Backend whose batch path echoes the solo outputs exactly."""

    fingerprint = "fake-batch-identical"

    def generate_batch(self, prompts, *, max_new_tokens, seed, cancel):
        """Generate batch."""
        return tuple(
            self.generate(ids, max_new_tokens=max_new_tokens, seed=seed, cancel=cancel).token_ids
            for ids in prompts
        )


class _DivergentBatchBackend(_SoloBackend):
    """Backend whose batch path flips one token on the second prompt."""

    fingerprint = "fake-batch-divergent"

    def generate_batch(self, prompts, *, max_new_tokens, seed, cancel):
        """Generate batch."""
        outputs = []
        for index, ids in enumerate(prompts):
            tokens = self.generate(
                ids, max_new_tokens=max_new_tokens, seed=seed, cancel=cancel
            ).token_ids
            if index == 1:
                tokens = (tokens[0] ^ 1,) + tokens[1:]
            outputs.append(tokens)
        return tuple(outputs)


class _ShortBatchBackend(_SoloBackend):
    """Short batch backend."""

    @staticmethod
    def generate_batch(prompts, *, max_new_tokens, seed, cancel):
        """Generate batch."""
        return ()


class _GarbageBatchBackend(_SoloBackend):
    """Garbage batch backend."""

    @staticmethod
    def generate_batch(prompts, *, max_new_tokens, seed, cancel):
        """Generate batch."""
        return tuple("nope" for _ in prompts)


class _RaisingBatchBackend(_SoloBackend):
    """Raising batch backend."""

    @staticmethod
    def generate_batch(prompts, *, max_new_tokens, seed, cancel):
        """Generate batch."""
        raise RuntimeError("boom")


class _ExplodingProbeBackend(_SoloBackend):
    """Exploding probe backend."""

    @property
    def generate_batch(self):
        raise RuntimeError("probe boom")


class _FlakySoloBackend(_SoloBackend):
    """Flaky solo backend."""

    def generate(self, input_ids, *, max_new_tokens, seed, cancel):
        """Generate."""
        if input_ids[0] == ord("b"):
            raise RuntimeError("solo boom")
        return super().generate(input_ids, max_new_tokens=max_new_tokens, seed=seed, cancel=cancel)

    @staticmethod
    def generate_batch(prompts, *, max_new_tokens, seed, cancel):
        """Generate batch."""
        return tuple(tuple(range(50, 50 + max_new_tokens)) for _ in prompts)


def _prompts(*names):
    """Prompts."""
    return {name: tuple(map(ord, name * 4)) for name in names}


class BatchStageTests(unittest.TestCase):
    """Batch stage tests."""

    def test_no_batch_path_records_unsupported(self):
        """Test no batch path records unsupported."""
        result = batch_stage(_SoloBackend(), _prompts("a", "b"))
        self.assertEqual(result, {"available": False, "detail": "batch_unsupported"})

    def test_identical_batch_is_invariant(self):
        """Test identical batch is invariant."""
        result = batch_stage(_BatchBackend(), _prompts("a", "b", "c"))
        self.assertTrue(result["available"])
        self.assertEqual(result["n_prompts"], 3)
        self.assertEqual(result["n_evaluable"], 3)
        self.assertEqual(result["n_diverged"], 0)
        self.assertEqual(result["divergence_rate"], 0.0)
        self.assertEqual(result["verdict"], "batch_invariant")
        self.assertEqual(
            result["per_prompt"]["a"]["diverged"],
            False,
        )

    def test_divergent_batch_detected(self):
        """Test divergent batch detected."""
        result = batch_stage(_DivergentBatchBackend(), _prompts("a", "b", "c"))
        self.assertTrue(result["available"])
        self.assertEqual(result["n_diverged"], 1)
        self.assertAlmostEqual(result["divergence_rate"], 1 / 3)
        self.assertEqual(result["verdict"], "batch_variant")
        entry = result["per_prompt"]["b"]
        self.assertTrue(entry["diverged"])
        self.assertEqual(entry["first_divergence"], 0)
        self.assertIsNone(entry["error"])
        self.assertFalse(result["per_prompt"]["a"]["diverged"])

    def test_short_batch_is_contract_failed(self):
        """Test short batch is contract failed."""
        result = batch_stage(_ShortBatchBackend(), _prompts("a"))
        self.assertEqual(result, {"available": False, "detail": "batch_contract_failed"})

    def test_garbage_batch_is_contract_failed(self):
        """Test garbage batch is contract failed."""
        result = batch_stage(_GarbageBatchBackend(), _prompts("a", "b"))
        self.assertEqual(result, {"available": False, "detail": "batch_contract_failed"})

    def test_raising_batch_is_failed_not_crash(self):
        """Test raising batch is failed not crash."""
        result = batch_stage(_RaisingBatchBackend(), _prompts("a"))
        self.assertEqual(result, {"available": False, "detail": "batch_failed"})

    def test_exploding_probe_is_failed_not_crash(self):
        """Test exploding probe is failed not crash."""
        result = batch_stage(_ExplodingProbeBackend(), _prompts("a"))
        self.assertEqual(result, {"available": False, "detail": "batch_failed"})

    def test_solo_failure_degrades_per_prompt(self):
        """Test solo failure degrades per prompt."""
        result = batch_stage(_FlakySoloBackend(), _prompts("a", "b", "c"))
        self.assertTrue(result["available"])
        self.assertEqual(result["n_prompts"], 3)
        self.assertEqual(result["n_evaluable"], 2)
        self.assertEqual(result["n_failed"], 1)
        self.assertEqual(result["n_diverged"], 0)
        self.assertEqual(result["divergence_rate"], 0.0)
        bad = result["per_prompt"]["b"]
        self.assertIsNone(bad["diverged"])
        self.assertEqual(bad["error"], "solo_failed")

    def test_non_callable_batch_path_is_unsupported(self):
        """Test non callable batch path is unsupported."""
        backend = _SoloBackend()
        backend.generate_batch = "not-callable"
        result = batch_stage(backend, _prompts("a"))
        self.assertEqual(result, {"available": False, "detail": "batch_unsupported"})

    def test_validation(self):
        """Test validation."""
        with self.assertRaises(ValueError):
            batch_stage(None, _prompts("a"))
        with self.assertRaises(ValueError):
            batch_stage(_SoloBackend(), {})
        with self.assertRaises(ValueError):
            batch_stage(_SoloBackend(), {"a": ()})
        with self.assertRaises(ValueError):
            batch_stage(_SoloBackend(), _prompts("a"), max_new_tokens=0)
        with self.assertRaises(ValueError):
            batch_stage(_SoloBackend(), _prompts("a"), seed=-1)


class CliProbeTests(unittest.TestCase):
    """Cli probe tests."""

    @staticmethod
    def _formats_and_cases():
        """Formats and cases."""
        fmt = core.Format(
            name="t",
            bos="",
            user_prefix="U:",
            assistant_prefix="A:",
            turn_end="\n",
        )
        cases = (
            core.Fixture(
                id="f1",
                messages=(core.Message("user", "hi"),),
                expected="x",
                group="g",
            ),
        )
        return (fmt,), cases

    def test_probe_records_unsupported_and_never_raises(self):
        """Test probe records unsupported and never raises."""
        from model_protocol_lab.cli import _run_batch_probe

        formats, cases = self._formats_and_cases()
        result: dict = {}
        _run_batch_probe(_SoloBackend(), formats, result, cases, max_new_tokens=8, seed=1)
        self.assertEqual(
            result["batch_invariance"],
            {"available": False, "detail": "batch_unsupported"},
        )

    def test_probe_records_available_payload(self):
        """Test probe records available payload."""
        from model_protocol_lab.cli import _run_batch_probe

        formats, cases = self._formats_and_cases()
        result: dict = {}
        _run_batch_probe(_BatchBackend(), formats, result, cases, max_new_tokens=8, seed=1)
        payload = result["batch_invariance"]
        self.assertTrue(payload["available"])
        self.assertEqual(payload["verdict"], "batch_invariant")
        self.assertEqual(payload["n_prompts"], 1)

    def test_probe_survives_broken_backend(self):
        """Test probe survives broken backend."""
        from model_protocol_lab.cli import _run_batch_probe

        class Broken:
            """Broken."""

            @staticmethod
            def encode(text):
                """Encode."""
                raise RuntimeError("nope")

        formats, cases = self._formats_and_cases()
        result: dict = {}
        _run_batch_probe(Broken(), formats, result, cases, max_new_tokens=8, seed=1)
        self.assertFalse(result["batch_invariance"]["available"])

    def test_flag_registered(self):
        """Test flag registered."""
        from model_protocol_lab.cli import build_parser

        base = [
            "run",
            "--model",
            "m",
            "--dataset",
            "d",
            "--family",
            "llama3",
            "--boundary-ids",
            "1",
            "--output",
            "o",
        ]
        args = build_parser().parse_args(base + ["--batch-invariance"])
        self.assertTrue(args.batch_invariance)
        args = build_parser().parse_args(base)
        self.assertFalse(args.batch_invariance)


class HtmlSectionTests(unittest.TestCase):
    """Html section tests."""

    @staticmethod
    def _payload():
        """Payload."""
        return {
            "available": True,
            "n_prompts": 2,
            "n_evaluable": 2,
            "n_failed": 0,
            "n_diverged": 1,
            "divergence_rate": 0.5,
            "verdict": "batch_variant",
            "per_prompt": {
                "p1": {
                    "diverged": False,
                    "first_divergence": None,
                    "solo_len": 8,
                    "batched_len": 8,
                    "error": None,
                },
                "p2": {
                    "diverged": True,
                    "first_divergence": 3,
                    "solo_len": 8,
                    "batched_len": 8,
                    "error": None,
                },
            },
        }

    def test_renders_available_payload(self):
        """Test renders available payload."""
        html = reporting._batchinv_section({"batch_invariance": self._payload()})
        self.assertIn("Batch-invariance audit", html)
        self.assertIn("batch-variant", html)
        self.assertIn("p2", html)
        self.assertIn("diverged", html)

    def test_unavailable_renders_nothing(self):
        """Test unavailable renders nothing."""
        self.assertEqual(reporting._batchinv_section({}), "")
        self.assertEqual(
            reporting._batchinv_section(
                {"batch_invariance": {"available": False, "detail": "batch_unsupported"}}
            ),
            "",
        )
        self.assertEqual(reporting._batchinv_section({"batch_invariance": None}), "")

    def test_escapes_hostile_values(self):
        """Test escapes hostile values."""
        payload = self._payload()
        payload["per_prompt"]["<script>alert(1)</script>"] = {
            "diverged": True,
            "first_divergence": 0,
            "solo_len": 1,
            "batched_len": 1,
            "error": "\x1b[31mred",
        }
        html = reporting._batchinv_section({"batch_invariance": payload})
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)
        # Terminal escapes are inert in HTML (escape() is the control
        # there); the stderr path strips them via _display_id, tested
        # in StderrSummaryTests.


def _good_batchinv_payload():
    """Good batchinv payload."""
    return {
        "available": True,
        "n_prompts": 2,
        "n_evaluable": 2,
        "n_failed": 0,
        "n_diverged": 1,
        "divergence_rate": 0.5,
        "verdict": "batch_variant",
        "per_prompt": {
            "p1": {
                "diverged": False,
                "first_divergence": None,
                "solo_len": 8,
                "batched_len": 8,
                "error": None,
            },
            "p2": {
                "diverged": True,
                "first_divergence": 3,
                "solo_len": 8,
                "batched_len": 8,
                "error": None,
            },
        },
    }


class DiffValidatorTests(unittest.TestCase):
    """Diff validator tests."""

    def test_valid_payload_accepted(self):
        """Test valid payload accepted."""
        self.assertTrue(diffreports._valid_batchinv(_good_batchinv_payload()))

    def test_hostile_payloads_rejected(self):
        """Test hostile payloads rejected."""
        good = _good_batchinv_payload()
        mutants = [
            ("nope",),
            ({"available": False},),
            ({**good, "available": True, "n_prompts": "2"},),
            ({**good, "n_diverged": 3},),
            ({**good, "n_evaluable": 1, "n_failed": 0},),
            ({**good, "divergence_rate": True},),
            ({**good, "n_evaluable": 0, "divergence_rate": 0.0},),
            ({**good, "verdict": "maybe"},),
            ({**good, "per_prompt": {"p1": "x"}},),
            ({**good, "per_prompt": {7: good["per_prompt"]["p1"]}},),
        ]
        for (mutant,) in mutants:
            self.assertFalse(diffreports._valid_batchinv(mutant), msg=f"{mutant!r} accepted")
        entry = dict(good["per_prompt"]["p1"])
        for bad_entry in (
            {**entry, "diverged": "yes"},
            {**entry, "first_divergence": -1},
            {**entry, "solo_len": "8"},
            {**entry, "error": 7},
        ):
            bad = dict(good)
            bad["per_prompt"] = {"p1": bad_entry}
            self.assertFalse(diffreports._valid_batchinv(bad), msg=f"{bad_entry!r}")

    def test_asymmetric_on_malformed_side(self):
        """Test asymmetric on malformed side."""
        good = _good_batchinv_payload()
        diff = diffreports.diff_batchinv({"available": False}, good)
        self.assertEqual(diff["available"], "new")
        self.assertFalse(diff["changed"])
        diff = diffreports.diff_batchinv(good, {"available": True, "n_prompts": "x"})
        self.assertEqual(diff["available"], "old")
        diff = diffreports.diff_batchinv(None, None)
        self.assertEqual(diff["available"], "neither")


class DiffChangedPinTests(unittest.TestCase):
    """Each changed disjunct pinned by a single-cause scenario."""

    def test_verdict_only(self):
        """Test verdict only."""
        old = _good_batchinv_payload()
        new = _good_batchinv_payload()
        # Only the verdict string differs (the validator checks types,
        # not cross-field consistency -- the disjunct under test is the
        # verdict comparison itself).
        new["verdict"] = "batch_invariant"
        diff = diffreports.diff_batchinv(old, new)
        self.assertTrue(diff["changed"])
        self.assertTrue(diff["verdict_changed"])
        self.assertFalse(diff["divergence_rate_changed"])
        self.assertEqual(diff["diverged_added"], [])
        self.assertEqual(diff["diverged_removed"], [])
        self.assertEqual(diff["prompts_added"], [])
        self.assertEqual(diff["prompts_removed"], [])

    def test_rate_only(self):
        """Test rate only."""
        old = _good_batchinv_payload()
        new = _good_batchinv_payload()
        new["divergence_rate"] = 0.5001
        diff = diffreports.diff_batchinv(old, new)
        self.assertTrue(diff["changed"])
        self.assertTrue(diff["divergence_rate_changed"])
        self.assertFalse(diff["verdict_changed"])
        self.assertEqual(diff["diverged_added"], [])
        self.assertEqual(diff["diverged_removed"], [])
        self.assertEqual(diff["prompts_added"], [])
        self.assertEqual(diff["prompts_removed"], [])

    def test_diverged_added_only(self):
        """Test diverged added only."""
        old = _good_batchinv_payload()
        new = _good_batchinv_payload()
        # Move the single divergence from p2 to p1: same counts, same
        # rate, same verdict -- only the diverged set changes.
        new["per_prompt"] = {
            "p1": {**old["per_prompt"]["p1"], "diverged": True, "first_divergence": 0},
            "p2": {**old["per_prompt"]["p2"], "diverged": False, "first_divergence": None},
        }
        diff = diffreports.diff_batchinv(old, new)
        self.assertTrue(diff["changed"])
        self.assertEqual(diff["diverged_added"], ["p1"])
        self.assertEqual(diff["diverged_removed"], ["p2"])
        self.assertFalse(diff["verdict_changed"])
        self.assertFalse(diff["divergence_rate_changed"])
        self.assertEqual(diff["prompts_added"], [])
        self.assertEqual(diff["prompts_removed"], [])

    def test_prompts_added_only(self):
        """Test prompts added only."""
        old = _good_batchinv_payload()
        new = _good_batchinv_payload()
        # Add a failed prompt (excluded from the rate): same counts
        # that matter, same rate, same verdict.
        new["n_prompts"] = 3
        new["n_failed"] = 1
        new["per_prompt"] = {
            **old["per_prompt"],
            "p3": {
                "diverged": None,
                "first_divergence": None,
                "solo_len": None,
                "batched_len": None,
                "error": "solo_failed",
            },
        }
        diff = diffreports.diff_batchinv(old, new)
        self.assertTrue(diff["changed"])
        self.assertEqual(diff["prompts_added"], ["p3"])
        self.assertEqual(diff["prompts_removed"], [])
        self.assertFalse(diff["verdict_changed"])
        self.assertFalse(diff["divergence_rate_changed"])
        self.assertEqual(diff["diverged_added"], [])
        self.assertEqual(diff["diverged_removed"], [])

    def test_identical_payloads_unchanged(self):
        """Test identical payloads unchanged."""
        diff = diffreports.diff_batchinv(_good_batchinv_payload(), _good_batchinv_payload())
        self.assertEqual(diff["available"], "both")
        self.assertFalse(diff["changed"])


class StderrSummaryTests(unittest.TestCase):
    """Stderr summary tests."""

    def test_lines_and_display_id_sanitization(self):
        """Test lines and display id sanitization."""
        old = _good_batchinv_payload()
        new = _good_batchinv_payload()
        new["verdict"] = "batch_invariant"
        new["n_diverged"] = 0
        new["divergence_rate"] = 0.0
        new["per_prompt"] = {
            pid: {**e, "diverged": False, "first_divergence": None}
            for pid, e in old["per_prompt"].items()
        }
        diff = diffreports.diff_batchinv(old, new)
        lines = diffreports._render_batchinv_summary(diff)
        self.assertTrue(any("verdict changed" in line for line in lines))

    def test_hostile_prompt_ids_stripped(self):
        """Test hostile prompt ids stripped."""
        old = _good_batchinv_payload()
        new = _good_batchinv_payload()
        evil = "\x1b[31mred\x1b[0m"
        new["per_prompt"] = {
            **old["per_prompt"],
            evil: {
                "diverged": True,
                "first_divergence": 0,
                "solo_len": 4,
                "batched_len": 4,
                "error": None,
            },
        }
        new["n_prompts"] = 3
        new["n_evaluable"] = 3
        new["n_diverged"] = 2
        new["divergence_rate"] = 2 / 3
        diff = diffreports.diff_batchinv(old, new)
        lines = diffreports._render_batchinv_summary(diff)
        blob = "\n".join(lines)
        self.assertNotIn("\x1b", blob)

    def test_unavailable_sides(self):
        """Test unavailable sides."""
        self.assertEqual(diffreports._render_batchinv_summary({"available": "neither"}), [])
        lines = diffreports._render_batchinv_summary({"available": "old"})
        self.assertTrue(any("new run did not record" in line for line in lines))
        self.assertEqual(diffreports._render_batchinv_summary(None), [])
        self.assertEqual(diffreports._render_batchinv_summary("x"), [])


class SampleDeterminismTests(unittest.TestCase):
    """Sample determinism tests."""

    def test_sample_batchinv_deterministic(self):
        """Test sample batchinv deterministic."""
        import importlib.util

        script = Path(__file__).resolve().parent.parent / "examples" / "make_sample_report.py"
        spec = importlib.util.spec_from_file_location("make_sample_report_c153", script)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory(prefix="c153_sample_") as first:
            result_a = module.build_sample_report(Path(first) / "regen")
        with tempfile.TemporaryDirectory(prefix="c153_sample_") as second:
            result_b = module.build_sample_report(Path(second) / "regen")
        self.assertEqual(result_a["batch_invariance"], result_b["batch_invariance"])
        payload = result_a["batch_invariance"]
        self.assertTrue(payload["available"])
        # The scripted backend diverges on exactly one fixture.
        self.assertEqual(payload["n_diverged"], 1)
        self.assertEqual(payload["verdict"], "batch_variant")


class TestTocAnchorsResolve(unittest.TestCase):
    """The report TOC must never point at a missing id (_toc docstring)."""

    def test_every_toc_href_has_matching_section_id(self):
        """Test every toc href has matching section id."""
        sample = Path(__file__).resolve().parent.parent / "examples" / "sample-report"
        results = json.loads((sample / "results.json").read_text(encoding="utf-8"))
        html = reporting.render_html(results)
        hrefs = set(re.findall(r'href="#([A-Za-z][\w-]*)"', html))
        ids = set(re.findall(r'id="([A-Za-z][\w-]*)"', html))
        missing = sorted(hrefs - ids)
        self.assertEqual(missing, [], f"TOC links with no matching section id: {missing}")
        # The card sections fixed in cycle 153 must resolve wherever rendered
        # (token-healing is absent here because the sample does not run that
        # probe -- the TOC skips unrendered sections by design).
        for anchor in (
            "selection-robustness",
            "transfer",
            "perturbation-robustness",
            "instruction-hierarchy",
            "synthetic-history",
            "batch-invariance",
            "prefill-sensitivity",
        ):
            self.assertIn(anchor, hrefs & ids, anchor)


if __name__ == "__main__":
    unittest.main()
