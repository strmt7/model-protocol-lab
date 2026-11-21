"""Wave-14 repair regressions (BH14-1..BH14-8).

Each test cites the contract its expected values come from — the
module's own stated invariant, a sibling validator's rule, or the
mirror's recompute standard — never the buggy output. The hostile
scenarios are the bug-hunt wave-14 batteries
(~/workspace/agents_out/bughunt14/repro/) re-built on the shipped
fake-vllm harness from tests.test_cycle243_serving_backends.
"""

from __future__ import annotations

import copy
import importlib.metadata
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from model_protocol_lab import boundary_audit, diffreports, experiments, reporting
from model_protocol_lab import servingbackends as sb
from tests.test_cycle243_serving_backends import (
    _fake_metadata_version,
    _fake_vllm_module,
    _FakeCompletion,
    _FakeLLM,
    _FakeRequestOutput,
    _restore_modules,
    _VllmFakeTokenizer,
)


def _decode(ids: tuple[int, ...]) -> str:
    return _VllmFakeTokenizer().decode(list(ids))


class _VllmCase(unittest.TestCase):
    """Checkpoint dir + env/module hygiene (the shipped case's
    fixture, replicated so this module imports no TestCase)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.model_dir = Path(self._tmp.name) / "ckpt"
        self.model_dir.mkdir()
        (self.model_dir / "config.json").write_text(
            json.dumps({"eos_token_id": 0, "max_position_embeddings": 512}),
            encoding="utf-8",
        )
        (self.model_dir / "generation_config.json").write_text(
            json.dumps({"eos_token_id": 0}), encoding="utf-8"
        )
        (self.model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
        (self.model_dir / "model.safetensors").write_bytes(b"weights")
        self._saved_env = {
            key: os.environ.get(key) for key in list(sb._VLLM_ENV_PINS) + ["VLLM_CPU_KVCACHE_SPACE"]
        }
        self.addCleanup(self._restore_env)
        sb._VLLM_ENV_DIGEST = None
        self.addCleanup(setattr, sb, "_VLLM_ENV_DIGEST", None)

    def _restore_env(self) -> None:
        for key, value in self._saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def install_fake(self, script: dict[str, Any] | None = None) -> None:
        saved = _fake_vllm_module(script)
        self.addCleanup(_restore_modules, saved)
        patcher = mock.patch.object(importlib.metadata, "version", _fake_metadata_version)
        patcher.start()
        self.addCleanup(patcher.stop)

    def make_backend(self, **overrides: Any) -> sb.VllmOfflineBackend:
        kwargs: dict[str, Any] = {
            "tokenizer": _VllmFakeTokenizer(),
            "kv_cache_memory_bytes": 512 * 1024 * 1024,
            "sleep": lambda _seconds: None,
        }
        kwargs.update(overrides)
        return sb.VllmOfflineBackend(self.model_dir, **kwargs)


# ---------------------------------------------------------------------------
# BH14-1 — execute() must never raise for a stack-side failure: a
# tokenizer whose all_special_ids holds non-integers makes the set
# unreadable as ids, so the declared-EOS fallback applies.
# ---------------------------------------------------------------------------


class _StringSpecialsTokenizer(_VllmFakeTokenizer):
    all_special_ids = ["<|endoftext|>"]  # type: ignore[assignment]


class _NoneSpecialsTokenizer(_VllmFakeTokenizer):
    all_special_ids = [None]  # type: ignore[assignment]


class _MixedSpecialsTokenizer(_VllmFakeTokenizer):
    all_special_ids = [0, "x"]  # type: ignore[assignment]


class VllmSpecialIdsTests(_VllmCase):
    def _execute_with(self, engine_tokenizer: Any, completion: _FakeCompletion) -> Any:
        self.install_fake({"engine_tokenizer": engine_tokenizer})
        backend = self.make_backend()
        backend.probe()
        llm = _FakeLLM.instances[-1]
        llm.generate_handler = lambda prompt_ids, params: [
            _FakeRequestOutput("fake-req", prompt_ids, completion)
        ]
        request = sb.StackRequest(prompt_ids=(1, 2), seed=7, max_new_tokens=4, eos_id=0)
        return backend.execute(request)

    def test_string_special_ids_do_not_raise(self) -> None:
        obs = self._execute_with(_StringSpecialsTokenizer(), _FakeCompletion([8], "a", "length"))
        self.assertEqual(obs.finish_class, "length")
        self.assertEqual(obs.detail, "")

    def test_none_special_id_does_not_raise(self) -> None:
        obs = self._execute_with(_NoneSpecialsTokenizer(), _FakeCompletion([8], "a", "length"))
        self.assertEqual(obs.finish_class, "length")

    def test_mixed_special_ids_fall_back_to_declared_eos(self) -> None:
        # The set cannot be read as ids, so the declared EOS set
        # applies: the EOS id in the stream still triggers the
        # paired no-special arm.
        obs = self._execute_with(
            _MixedSpecialsTokenizer(),
            _FakeCompletion([8, 0], "a<|endoftext|>", "stop", 0),
        )
        self.assertIsNotNone(obs.returned_text_digest_no_special)

    def test_genuine_special_set_still_pairs(self) -> None:
        # Control: a readable special set containing the EOS id
        # pairs exactly as before the fix.
        obs = self._execute_with(
            _VllmFakeTokenizer(),
            _FakeCompletion([8, 0], "a<|endoftext|>", "stop", 0),
        )
        self.assertIsNotNone(obs.returned_text_digest_no_special)


# ---------------------------------------------------------------------------
# BH14-2 — the vLLM predictor's stripped digest names the text
# before the FIRING occurrence of the winning stop (the detokenizer
# port's own truncation), not the first occurrence in the stream.
# ---------------------------------------------------------------------------


class PredictorStrippedDigestTests(_VllmCase):
    IDS = (8, 9, 1, 2, 3, 8, 9)  # "ab" + "The capital of" + "ab"

    def test_gated_then_fired_stop_digests_the_firing_occurrence(self) -> None:
        prediction = sb.predict_stop_outcome(
            decode=_decode,
            returned_ids=self.IDS,
            stop_texts=("ab",),
            eos_id=0,
            min_tokens=5,
            stop_model="vllm",
        )
        self.assertEqual(prediction["predicted_firing_step"], 7)
        self.assertEqual(prediction["predicted_winner"], {"kind": "stop_text", "index": 0})
        self.assertEqual(prediction["stripped_text_digest"], sb._text_digest("abThe capital of"))

    def test_generic_model_keeps_first_occurrence(self) -> None:
        # Control: the generic model fires at the first completion,
        # whose occurrence IS the first in the text — unchanged.
        prediction = sb.predict_stop_outcome(
            decode=_decode,
            returned_ids=self.IDS,
            stop_texts=("ab",),
            eos_id=0,
            stop_model="generic",
        )
        self.assertEqual(prediction["predicted_firing_step"], 2)
        self.assertEqual(prediction["stripped_text_digest"], sb._text_digest(""))

    def test_gated_then_fired_stop_agrees_end_to_end(self) -> None:
        self.install_fake()
        backend = self.make_backend()
        backend.probe()
        llm = _FakeLLM.instances[-1]
        llm.generate_handler = lambda prompt_ids, params: [
            _FakeRequestOutput(
                "fake-req",
                prompt_ids,
                _FakeCompletion(list(self.IDS), "abThe capital of", "stop", "ab"),
            )
        ]
        request = sb.StackRequest(
            prompt_ids=(1, 2),
            seed=7,
            max_new_tokens=16,
            stop_texts=("ab",),
            eos_id=0,
            min_tokens=5,
        )
        record = sb.execute_cell(
            backend, request=request, decode=_decode, candidate="c", fixture="f"
        )
        mirror = record["mirror"]
        self.assertEqual(mirror["fidelity_verdict"], "agree")
        self.assertEqual(mirror["observed_disposition"], "stripped")
        self.assertEqual(mirror["divergence_detail"], "")


# ---------------------------------------------------------------------------
# BH14-3 / BH14-4 — boundary_audit: recorded ids are exact ints or
# the check is unusable (the sibling validators' rule), and the
# per-candidate rollup conserves under duplicate format names
# (SA-02's principle one level down).
# ---------------------------------------------------------------------------


def _encode(text: str) -> tuple[int, ...]:
    return tuple(text.encode("ascii"))


def _decode_bytes(ids: tuple[int, ...]) -> str:
    return bytes(ids).decode("ascii")


class BoundaryAuditExactnessTests(unittest.TestCase):
    def test_exact_int_control_is_usable(self) -> None:
        check = boundary_audit.check_boundary("ab", [97, 98], [99], _encode, _decode_bytes)
        self.assertTrue(check.usable)
        self.assertFalse(check.has_problem)

    def test_infinite_ids_are_unusable_never_a_crash(self) -> None:
        for prompt_ids, completion_ids in (
            ([float("inf"), 98], [99]),
            ([97], [float("-inf")]),
        ):
            check = boundary_audit.check_boundary(
                "ab", prompt_ids, completion_ids, _encode, _decode_bytes
            )
            self.assertFalse(check.usable)
            self.assertFalse(check.has_problem)

    def test_string_float_and_bool_ids_are_unusable_not_coerced(self) -> None:
        for prompt_ids, completion_ids in (
            (["97", "98"], [99]),
            ([97, 98], [99.9]),
            ([True, 98], [99]),
        ):
            check = boundary_audit.check_boundary(
                "ab", prompt_ids, completion_ids, _encode, _decode_bytes
            )
            self.assertFalse(check.usable, (prompt_ids, completion_ids))

    def test_non_sequence_ids_are_unusable_tuples_stay_usable(self) -> None:
        # A bare string is a Sequence of characters, not a recorded
        # id sequence; a mapping is not a sequence at all. Both are
        # unusable, never iterated into ids. Tuples are the other
        # exact container and stay usable (control).
        check = boundary_audit.check_boundary("ab", "9798", [99], _encode, _decode_bytes)
        self.assertFalse(check.usable)
        self.assertFalse(check.has_problem)
        check = boundary_audit.check_boundary("ab", [97, 98], {99: None}, _encode, _decode_bytes)
        self.assertFalse(check.usable)
        check = boundary_audit.check_boundary("ab", (97, 98), (99,), _encode, _decode_bytes)
        self.assertTrue(check.usable)
        self.assertFalse(check.has_problem)

    def test_encode_output_must_be_exact_ints(self) -> None:
        def lying_encode(text: str) -> Any:
            return tuple(float(b) for b in text.encode("ascii"))

        check = boundary_audit.check_boundary("ab", [97, 98], [99], lying_encode, _decode_bytes)
        self.assertFalse(check.usable)

    def test_audit_counts_infinite_observation_unusable(self) -> None:
        entries = [
            {
                "format": {"name": "chatml/base"},
                "observations": [
                    {"input_text": "hello", "input_ids": [float("inf"), 2], "output_ids": [1]},
                ],
            }
        ]
        out = boundary_audit.audit_search_observations(entries, _encode, _decode_bytes)
        self.assertEqual(out["n_observations"], 1)
        self.assertEqual(out["n_usable"], 0)
        self.assertEqual(out["n_unusable"], 1)

    def test_audit_does_not_coerce_mistyped_ids(self) -> None:
        entries = [
            {
                "format": {"name": "chatml/base"},
                "observations": [
                    {"input_text": "hello", "input_ids": ["1", "2"], "output_ids": [1.9]},
                ],
            }
        ]
        out = boundary_audit.audit_search_observations(entries, _encode, _decode_bytes)
        self.assertEqual(out["n_usable"], 0)
        self.assertEqual(out["n_unusable"], 1)

    def test_duplicate_candidate_names_conserve(self) -> None:
        def obs(text: str) -> dict[str, Any]:
            return {
                "input_text": text,
                "input_ids": list(text.encode("ascii")),
                "output_ids": [98],
            }

        entries = [
            {"format": {"name": "chatml/base"}, "observations": [obs("aa"), obs("bb")]},
            {"format": {"name": "chatml/base"}, "observations": [obs("cc")]},
        ]
        out = boundary_audit.audit_search_observations(entries, _encode, _decode_bytes)
        self.assertEqual(out["n_observations"], 3)
        per_candidate = out["per_candidate"]
        self.assertEqual(set(per_candidate), {"chatml/base"})
        self.assertEqual(per_candidate["chatml/base"]["n_observations"], 3)
        self.assertEqual(
            sum(c["n_observations"] for c in per_candidate.values()), out["n_observations"]
        )
        self.assertEqual(sum(c["n_usable"] for c in per_candidate.values()), out["n_usable"])
        self.assertEqual(out["n_usable"], 3)


# ---------------------------------------------------------------------------
# Shared genuine-section builder (the battery-F shape on the shipped
# harness): one standalone plan cell plus the full cache companion.
# ---------------------------------------------------------------------------

_CACHE_TABLE: dict[tuple[int, ...], list[tuple[int, int]]] = {
    (1, 2, 3, 4): [(0, 512), (384, 128)],
    (1, 2, 9, 9): [(128, 256)],
    (13, 10): [(0, 384)],
}

_CACHE_PROMPTS = {
    "cold_ids": (1, 2, 3, 4),
    "shared_prefix_ids": (1, 2, 9, 9),
    "divergent_control_ids": (13, 10),
}

_PLAN_CELL = {
    "candidate": "t",
    "fixture": "t0",
    "cache_position": "standalone",
}


class _SectionCase(_VllmCase):
    def build_stage(
        self,
        *,
        table: dict[tuple[int, ...], list[tuple[int, int]]] | None = None,
        on_record: Any = None,
        with_plan: bool = True,
    ) -> dict[str, Any]:
        self.install_fake()
        backend = self.make_backend()
        backend.probe()
        llm = _FakeLLM.instances[-1]
        state = {
            "table": {k: list(v) for k, v in (table if table is not None else _CACHE_TABLE).items()}
        }

        def handler(prompt_ids: Any, params: Any) -> list[_FakeRequestOutput]:
            key = tuple(prompt_ids)
            if key in state["table"] and state["table"][key]:
                cached, creation = state["table"][key].pop(0)
                return [
                    _FakeRequestOutput(
                        "r",
                        prompt_ids,
                        _FakeCompletion([8], "a", "length"),
                        num_cached_tokens=cached,
                        num_cache_creation_tokens=creation,
                    )
                ]
            return [_FakeRequestOutput("r", prompt_ids, _FakeCompletion([8], "a", "length"))]

        llm.generate_handler = handler
        plan_cells = []
        if with_plan:
            plan_cells.append(
                {
                    **_PLAN_CELL,
                    "request": sb.StackRequest(
                        prompt_ids=(1, 2, 3), seed=7, max_new_tokens=4, eos_id=0
                    ),
                }
            )
        return sb.run_stage(
            backend,
            plan_cells=plan_cells,
            fidelity_input=None,
            cache_prompts=_CACHE_PROMPTS,
            seed=0,
            decode=_decode,
            capabilities=backend.capabilities,
            on_record=on_record,
        )

    def build_section(self, **kwargs: Any) -> dict[str, Any]:
        return experiments.serving_backends_analysis(self.build_stage(**kwargs))


# ---------------------------------------------------------------------------
# BH14-5 — abort_probe coherence: the producer sets the three
# terminal fields atomically (a terminal whose stream length
# cannot be read was not observed), and the mirror enforces the
# conjunction (observed <=> abort terminal + a stream length the
# 256-token cell can produce).
# ---------------------------------------------------------------------------


class AbortProbeCoherenceTests(_SectionCase):
    def test_terminal_with_unreadable_token_ids_records_unobserved(self) -> None:
        self.install_fake()
        backend = self.make_backend()

        def wait_unsized(llm: Any, use_tqdm: bool = True) -> list[_FakeRequestOutput]:
            rid, prompt_ids = llm._pending
            return [_FakeRequestOutput(rid, prompt_ids, _FakeCompletion(None, "abc", "abort"))]

        with mock.patch.object(_FakeLLM, "wait_for_completion", wait_unsized):
            identity = backend.probe()
        probe = identity.config["abort_probe"]
        self.assertIs(probe["terminal_observed"], False)
        self.assertIsNone(probe["finish_reason"])
        self.assertIsNone(probe["returned_tokens"])

    def _with_abort(self, section: dict[str, Any], **patch: Any) -> dict[str, Any]:
        mutated = copy.deepcopy(section)
        mutated["identity"]["config"]["abort_probe"].update(patch)
        mutated["identity"]["config_fingerprint"] = sb.sha256_hex(
            sb.canonical_json(mutated["identity"]["config"]).encode("utf-8")
        )
        return mutated

    def test_mirror_rejects_incoherent_abort_records(self) -> None:
        section = self.build_section()
        self.assertTrue(diffreports._valid_serving_backends(section))
        for patch in (
            {"terminal_observed": True, "finish_reason": None, "returned_tokens": None},
            {"terminal_observed": True, "finish_reason": "abort", "returned_tokens": None},
            {"terminal_observed": False, "finish_reason": "abort", "returned_tokens": 12},
            {"terminal_observed": False, "finish_reason": None, "returned_tokens": 12},
            {"terminal_observed": True, "finish_reason": "abort", "returned_tokens": 999999},
            {"terminal_observed": True, "finish_reason": "abort", "returned_tokens": 257},
        ):
            self.assertFalse(
                diffreports._valid_serving_backends(self._with_abort(section, **patch)),
                patch,
            )

    def test_mirror_accepts_coherent_abort_records(self) -> None:
        section = self.build_section()
        for patch in (
            {"terminal_observed": False, "finish_reason": None, "returned_tokens": None},
            {"terminal_observed": True, "finish_reason": "abort", "returned_tokens": 0},
            {"terminal_observed": True, "finish_reason": "abort", "returned_tokens": 256},
        ):
            self.assertTrue(
                diffreports._valid_serving_backends(self._with_abort(section, **patch)),
                patch,
            )


# ---------------------------------------------------------------------------
# BH14-6 — the mirror rejects "agree" records its own stored
# fields disprove: agreement with no prediction, and agreement on
# a stop-token winner the record's own request_config rules out.
# ---------------------------------------------------------------------------


class MirrorAgreeCoherenceTests(_SectionCase):
    def _with_record(self, section: dict[str, Any], mutate: Any) -> dict[str, Any]:
        mutated = copy.deepcopy(section)
        record = next(r for r in mutated["records"] if r["cache_position"] == "standalone")
        mutate(record)
        mutated["stop_matrix"] = {
            "available": True,
            "rows": diffreports._serving_recompute_rows(mutated["records"], "vllm"),
        }
        agg = sb.aggregate_fidelity(mutated["records"])
        mutated["fidelity"]["available"] = agg["available"]
        mutated["fidelity"]["verdict"] = agg["verdict"]
        mutated["fidelity"]["counts"] = agg["counts"]
        mutated["fidelity"]["divergence_classes"] = agg["divergence_classes"]
        mutated["fidelity"]["detail"] = "" if agg["available"] else "not_attempted"
        return mutated

    def test_genuine_section_validates(self) -> None:
        self.assertTrue(diffreports._valid_serving_backends(self.build_section()))

    def test_agree_without_prediction_is_invalid(self) -> None:
        def mutate(record: dict[str, Any]) -> None:
            record["stop_winner"] = {"kind": "stop_token", "id": 8}
            record["mirror"]["predicted_winner"] = None
            record["mirror"]["predicted_firing_step"] = None
            record["mirror"]["fidelity_verdict"] = "agree"
            record["mirror"]["divergence_class"] = None
            record["mirror"]["divergence_detail"] = ""

        section = self._with_record(self.build_section(), mutate)
        self.assertFalse(diffreports._valid_serving_backends(section))

    def test_agree_with_prediction_but_no_winner_is_invalid(self) -> None:
        # The producer diverges MISSING_STOP_AT_CAP when a stop is
        # predicted but nothing fired; an agree record whose own
        # prediction stands while its observed winner is absent is
        # disproved by its fields (BH14-6). The genuine record is a
        # no-fire agreement (no winner, no prediction), so the
        # disproved shape is constructed on it.
        def mutate(record: dict[str, Any]) -> None:
            record["stop_winner"] = None
            record["mirror"]["predicted_winner"] = {"kind": "eos", "id": 0}
            record["mirror"]["predicted_firing_step"] = len(record["returned_ids"])
            record["mirror"]["fidelity_verdict"] = "agree"
            record["mirror"]["divergence_class"] = None
            record["mirror"]["divergence_detail"] = ""

        section = self._with_record(self.build_section(), mutate)
        self.assertFalse(diffreports._valid_serving_backends(section))

    def test_agree_with_disproven_stop_token_winner_is_invalid(self) -> None:
        def mutate(record: dict[str, Any]) -> None:
            # Winner id 77: not in the record's stop_token_ids
            # (empty) and not a declared EOS (0).
            record["stop_winner"] = {"kind": "stop_token", "id": 77}
            record["mirror"]["predicted_winner"] = {"kind": "stop_token", "id": 77}
            record["mirror"]["predicted_firing_step"] = len(record["returned_ids"])
            record["mirror"]["fidelity_verdict"] = "agree"
            record["mirror"]["divergence_class"] = None
            record["mirror"]["divergence_detail"] = ""

        section = self._with_record(self.build_section(), mutate)
        self.assertFalse(diffreports._valid_serving_backends(section))

    def test_agree_with_listed_stop_token_winner_stays_valid(self) -> None:
        # Control: the same shape with the winner id the request
        # actually listed — a legitimate agreement.
        def mutate(record: dict[str, Any]) -> None:
            record["stop_winner"] = {"kind": "stop_token", "id": 8}
            record["request_config"]["stop_token_ids"] = [8]
            record["request_digest"] = diffreports._serving_sha_text(
                diffreports._serving_canonical(
                    {
                        "config": record["request_config"],
                        "prompt_id_digest": record["prompt_id_digest"],
                    }
                )
            )
            record["mirror"]["predicted_winner"] = {"kind": "stop_token", "id": 8}
            record["mirror"]["predicted_firing_step"] = len(record["returned_ids"])
            record["mirror"]["fidelity_verdict"] = "agree"
            record["mirror"]["divergence_class"] = None
            record["mirror"]["divergence_detail"] = ""

        section = self._with_record(self.build_section(), mutate)
        self.assertTrue(diffreports._valid_serving_backends(section))


# ---------------------------------------------------------------------------
# BH14-7 — a records-budget trip inside the cache sequence reports
# the partial cache measurement with the budget detail (the
# positions the executed arms produced), so the section stays
# mirror-valid like the sibling plan-cell degradation.
# ---------------------------------------------------------------------------


def _trip_on(call_number: int) -> Any:
    state = {"n": 0}

    def on_record(record: dict[str, Any]) -> bool:
        state["n"] += 1
        return state["n"] < call_number

    return on_record


class CacheBudgetTripTests(_SectionCase):
    def test_no_trip_control_validates(self) -> None:
        section = self.build_section()
        self.assertTrue(diffreports._valid_serving_backends(section))
        self.assertEqual(section["cache"]["detail"], "")

    def test_trip_on_first_cache_arm_reports_partial_positions(self) -> None:
        section = self.build_section(on_record=_trip_on(2))
        self.assertTrue(diffreports._valid_serving_backends(section))
        self.assertEqual(set(section["cache"]["positions"]), {"cold"})
        self.assertEqual(section["cache"]["detail"], "serving_backends_record_budget_exceeded")
        self.assertIs(section["cache"]["available"], True)

    def test_trip_on_second_cache_arm_reports_partial_positions(self) -> None:
        section = self.build_section(on_record=_trip_on(3))
        self.assertTrue(diffreports._valid_serving_backends(section))
        self.assertEqual(set(section["cache"]["positions"]), {"cold", "warm_repeat"})
        self.assertEqual(section["cache"]["detail"], "serving_backends_record_budget_exceeded")

    def test_trip_on_plan_cell_keeps_pre_sequence_degradation(self) -> None:
        # Control: the sibling path (budget spent before the
        # sequence starts) is unchanged — absent records, no
        # positions, still valid.
        section = self.build_section(on_record=_trip_on(1))
        self.assertTrue(diffreports._valid_serving_backends(section))
        self.assertEqual(section["cache"]["positions"], {})
        self.assertIs(section["cache"]["available"], False)

    def test_contaminated_partial_sequence_reports_contamination(self) -> None:
        # A cold arm that already breached the quantum is the
        # cache story even when the budget then trips: the detail
        # follows the completed-sequence precedence.
        table = {**_CACHE_TABLE, (1, 2, 3, 4): [(128, 512), (384, 128)]}
        section = self.build_section(table=table, on_record=_trip_on(2))
        self.assertTrue(diffreports._valid_serving_backends(section))
        self.assertEqual(section["cache"]["detail"], "cache_protocol_contaminated")
        self.assertEqual(set(section["cache"]["positions"]), {"cold"})


class CacheDerivationRuleTests(unittest.TestCase):
    """The partial-record derivations behind BH14-7, pinned at the
    helper: positions skip exactly the records the mirror skips
    (standalone, capability-absent, budget-omitted), and the
    contamination judgement refuses to invent a verdict when the
    engine block size was never read back."""

    @staticmethod
    def _identity(stack: str, config: dict[str, Any]) -> Any:
        return sb.StackIdentity(
            stack=stack,
            stack_version="0.33.0",
            checkpoint_digest="0" * 64,
            checkpoint_bytes=1,
            tokenizer_fingerprint="sha256:" + "0" * 64,
            hardware_class="cpu",
            declared_eos_ids=(0,),
            config=config,
            config_fingerprint="0" * 64,
        )

    @staticmethod
    def _record(position: str, detail: str = "", **reported: Any) -> dict[str, Any]:
        record: dict[str, Any] = {
            "cache_position": position,
            "cell_detail": detail,
            "cache_n_reported": None,
            "tokens_cached_reported": None,
            "tokens_evaluated_reported": None,
            "cache_creation_tokens_reported": None,
        }
        record.update(reported)
        return record

    def test_positions_skip_absent_and_budget_records(self) -> None:
        identity = self._identity("vllm", {"engine": {"block_size": 128}})
        records = [
            self._record("standalone", tokens_cached_reported=0),
            self._record("cold", "capability_absent"),
            self._record("warm_repeat", "serving_backends_record_budget_exceeded"),
            self._record("shared_prefix", tokens_cached_reported=64),
        ]
        positions = sb._cache_positions_from_records(identity, records)
        self.assertEqual(set(positions), {"shared_prefix"})
        self.assertEqual(
            positions["shared_prefix"],
            {"tokens_cached": 64, "cache_creation_tokens": None},
        )

    def test_contamination_needs_a_read_back_block_size(self) -> None:
        records = [self._record("cold", tokens_cached_reported=128)]
        no_block = self._identity("vllm", {"engine": {}})
        self.assertFalse(sb._cache_contamination_over(no_block, records))
        with_block = self._identity("vllm", {"engine": {"block_size": 128}})
        self.assertTrue(sb._cache_contamination_over(with_block, records))
        below = [self._record("cold", tokens_cached_reported=127)]
        self.assertFalse(sb._cache_contamination_over(with_block, below))
        llama = self._identity("llamacpp", {})
        self.assertTrue(
            sb._cache_contamination_over(llama, [self._record("cold", cache_n_reported=2)])
        )
        self.assertFalse(
            sb._cache_contamination_over(llama, [self._record("cold", cache_n_reported=1)])
        )


# ---------------------------------------------------------------------------
# BH14-8 — the HTML renderer guards the identity.config chain at
# the level it dereferences: a type-confused config renders the
# section degraded instead of raising AttributeError.
# ---------------------------------------------------------------------------


class RendererConfigGuardTests(_SectionCase):
    def _render(self, payload: dict[str, Any]) -> str:
        return reporting._serving_backends_section({"serving_backends": payload})

    def test_genuine_section_renders(self) -> None:
        html = self._render(self.build_section())
        self.assertGreater(len(html), 1000)

    def test_config_as_string_renders_degraded(self) -> None:
        section = self.build_section()
        section["identity"] = dict(section["identity"])
        section["identity"]["config"] = "not-a-dict"
        html = self._render(section)
        self.assertIsInstance(html, str)
        self.assertGreater(len(html), 1000)

    def test_config_engine_as_string_renders(self) -> None:
        # Control: the next level down was already guarded.
        section = self.build_section()
        section["identity"] = dict(section["identity"])
        section["identity"]["config"] = {"engine": "not-a-dict"}
        html = self._render(section)
        self.assertIsInstance(html, str)
