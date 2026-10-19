"""Cycle 163: prompt-only tool-call syntax adherence per template (wave-4 idea 4).

Tests the pure ``toolcall`` core (validation, canonical rendering,
deterministic classification, aggregation), the scripted-backend stage,
the 200-case brute-force oracle differential, Hypothesis properties,
4/4 tempdir mutation gates, single-cause ``changed`` pins for every
diff disjunct, hostile-input sanitization (``<script>``-bearing
arguments, terminal escapes in candidate names), the CLI probe, and
the TOC anchor -- plus extensions to both diff oracles (cycle-27
availability mirror, cycle-67 full oracle + fuzz).
"""

import importlib.util
import json
import os
import random
import re
import tempfile
import unittest
from pathlib import Path
from threading import Event
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from model_protocol_lab import diffreports
from model_protocol_lab.core import Format, Generation
from model_protocol_lab.experiments import toolcall_adherence, toolcall_stage
from model_protocol_lab.toolcall import (
    OUTCOMES,
    TOOLCALL_QUERIES,
    TOOLS,
    build_toolcall_fixtures,
    candidate_adherence,
    canonical_call,
    classify_tool_call,
    query_calls,
    render_prompt,
    validate_tool_def,
    validate_toolcall_queries,
)


class ToolcallScriptedBackend:
    """Policy backend keyed on the ``(ref: toolcall/<id>)`` marker.

    Ord codec. The ``policy`` maps fixture id -> emitted text; ids not
    in the map get a garbage line (parse_failed). Deterministic.
    """

    fingerprint = "scripted-toolcall-c163"

    def __init__(self, policy: dict[str, str]):
        self.policy = policy

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
        match = re.search(r"\(ref: toolcall/([a-z-]+)\)", prompt)
        answer = self.policy.get(match.group(1), "not a call at all") if match else "x"
        # reason="limit" requires exactly max_new_tokens ids; pad with
        # spaces (classify_tool_call strips, so padding is inert).
        out = tuple(map(ord, answer.ljust(max_new_tokens)[:max_new_tokens]))
        return Generation(token_ids=out, reason="limit")


def _format(name: str) -> Format:
    return Format(
        name=name,
        bos="",
        user_prefix="[user]\n",
        assistant_prefix="[asst]\n",
        turn_end="",
    )


def _queries():
    return validate_toolcall_queries(TOOLCALL_QUERIES)


def _correct_policy() -> dict[str, str]:
    queries = _queries()
    return {q["id"]: canonical_call(q["call"]) for q in queries}


class ToolcallCoreTests(unittest.TestCase):
    def test_tool_defs_validate(self):
        for tool in TOOLS:
            normalized = validate_tool_def(tool)
            self.assertEqual(normalized["name"], tool["name"])
            self.assertEqual(set(normalized["required"]), set(tool["required"]))

    def test_tool_def_rejects_bad_name(self):
        bad = dict(TOOLS[0])
        bad["name"] = "not a name!"
        with self.assertRaises(ValueError):
            validate_tool_def(bad)

    def test_tool_def_rejects_unknown_param_type(self):
        bad = dict(TOOLS[0])
        params = {k: dict(v) for k, v in bad["params"].items()}
        params["city"] = {"type": "object"}
        bad["params"] = params
        with self.assertRaises(ValueError):
            validate_tool_def(bad)

    def test_tool_def_rejects_bad_enum(self):
        bad = dict(TOOLS[0])
        params = {k: dict(v) for k, v in bad["params"].items()}
        params["unit"] = {"type": "string", "enum": "CF"}
        bad["params"] = params
        with self.assertRaises(ValueError):
            validate_tool_def(bad)

    def test_tool_def_rejects_required_outside_params(self):
        bad = dict(TOOLS[0])
        bad["required"] = ("city", "nope")
        with self.assertRaises(ValueError):
            validate_tool_def(bad)

    def test_tool_def_rejects_non_dict(self):
        with self.assertRaises(ValueError):
            validate_tool_def(["get_weather"])

    def test_tool_def_rejects_empty_params(self):
        bad = dict(TOOLS[0])
        bad["params"] = {}
        with self.assertRaises(ValueError):
            validate_tool_def(bad)

    def test_tool_def_rejects_non_string_param_name(self):
        bad = dict(TOOLS[0])
        bad["params"] = {123: {"type": "string"}}
        with self.assertRaises(ValueError):
            validate_tool_def(bad)

    def test_tool_def_rejects_non_dict_param(self):
        bad = dict(TOOLS[0])
        bad["params"] = {"city": "string"}
        with self.assertRaises(ValueError):
            validate_tool_def(bad)

    def test_tool_def_rejects_bad_enum_variants(self):
        for bad_enum in ([], ["C", 1], "CF", None.__class__):
            if bad_enum is None.__class__:
                continue
            bad = dict(TOOLS[0])
            params = {k: dict(v) for k, v in bad["params"].items()}
            # get_weather has unit with enum ["celsius", "fahrenheit"]
            params["unit"] = {"type": "string", "enum": bad_enum}
            bad["params"] = params
            with self.assertRaises(ValueError):
                validate_tool_def(bad)

    def test_tool_def_rejects_non_numeric_bound(self):
        bad = dict(TOOLS[0])
        params = {k: dict(v) for k, v in bad["params"].items()}
        # convert_currency amount has minimum 0
        params["amount"] = {"type": "number", "minimum": "zero"}
        bad["params"] = params
        with self.assertRaises(ValueError):
            validate_tool_def(bad)
        # bool is not a valid bound
        params["amount"] = {"type": "number", "maximum": True}
        bad["params"] = params
        with self.assertRaises(ValueError):
            validate_tool_def(bad)

    def test_tool_def_rejects_bad_required(self):
        bad = dict(TOOLS[0])
        bad["required"] = ["city"]  # list, not tuple
        with self.assertRaises(ValueError):
            validate_tool_def(bad)
        bad = dict(TOOLS[0])
        bad["required"] = ()  # empty
        with self.assertRaises(ValueError):
            validate_tool_def(bad)

    def test_tool_def_rejects_bad_description(self):
        bad = dict(TOOLS[0])
        bad["description"] = 123
        with self.assertRaises(ValueError):
            validate_tool_def(bad)

    def test_queries_validate_and_normalize(self):
        queries = _queries()
        self.assertEqual(len(queries), 9)
        self.assertEqual([q["id"] for q in queries][:2], ["weather-paris", "weather-berlin"])
        abstains = [q for q in queries if q["call"] is None]
        self.assertEqual(len(abstains), 2)

    def test_queries_reject_duplicate_id(self):
        dup_list = [dict(q) for q in TOOLCALL_QUERIES]
        dup_list[1] = dict(dup_list[1])
        dup_list[1]["id"] = dup_list[0]["id"]
        with self.assertRaises(ValueError):
            validate_toolcall_queries(tuple(dup_list))

    def test_queries_reject_unknown_tool(self):
        bad_list = [dict(q) for q in TOOLCALL_QUERIES]
        bad_list[0] = dict(bad_list[0])
        bad_list[0]["call"] = {"name": "nope", "arguments": {}}
        with self.assertRaises(ValueError):
            validate_toolcall_queries(tuple(bad_list))

    def test_queries_reject_bad_enum_arg(self):
        bad_list = [dict(q) for q in TOOLCALL_QUERIES]
        bad_list[0] = dict(bad_list[0])
        bad_list[0]["call"] = {
            "name": "get_weather",
            "arguments": {"city": "Paris", "unit": "K"},
        }
        with self.assertRaises(ValueError):
            validate_toolcall_queries(tuple(bad_list))

    def test_queries_reject_out_of_bounds_arg(self):
        bad_list = [dict(q) for q in TOOLCALL_QUERIES]
        bad_list[5] = dict(bad_list[5])
        bad_list[5]["call"] = {
            "name": "search_docs",
            "arguments": {"query": "x", "top_k": 99},
        }
        with self.assertRaises(ValueError):
            validate_toolcall_queries(tuple(bad_list))

    def test_queries_reject_missing_required(self):
        bad_list = [dict(q) for q in TOOLCALL_QUERIES]
        bad_list[0] = dict(bad_list[0])
        bad_list[0]["call"] = {"name": "get_weather", "arguments": {"city": "Paris"}}
        with self.assertRaises(ValueError):
            validate_toolcall_queries(tuple(bad_list))

    def test_queries_reject_non_tuple(self):
        with self.assertRaises(ValueError):
            validate_toolcall_queries([dict(q) for q in TOOLCALL_QUERIES])
        with self.assertRaises(ValueError):
            validate_toolcall_queries(())

    def test_queries_reject_non_dict_query(self):
        bad_list = [dict(q) for q in TOOLCALL_QUERIES]
        bad_list[0] = "not a dict"
        with self.assertRaises(ValueError):
            validate_toolcall_queries(tuple(bad_list))

    def test_queries_reject_bad_id(self):
        for bad_id in (123, "", None):
            bad_list = [dict(q) for q in TOOLCALL_QUERIES]
            bad_list[0] = dict(bad_list[0])
            bad_list[0]["id"] = bad_id
            with self.assertRaises(ValueError):
                validate_toolcall_queries(tuple(bad_list))

    def test_queries_reject_bad_question(self):
        for bad_q in (123, "", None):
            bad_list = [dict(q) for q in TOOLCALL_QUERIES]
            bad_list[0] = dict(bad_list[0])
            bad_list[0]["question"] = bad_q
            with self.assertRaises(ValueError):
                validate_toolcall_queries(tuple(bad_list))

    def test_queries_reject_non_dict_call(self):
        bad_list = [dict(q) for q in TOOLCALL_QUERIES]
        bad_list[0] = dict(bad_list[0])
        bad_list[0]["call"] = "CALL get_weather"
        with self.assertRaises(ValueError):
            validate_toolcall_queries(tuple(bad_list))

    def test_queries_reject_non_dict_arguments(self):
        bad_list = [dict(q) for q in TOOLCALL_QUERIES]
        bad_list[0] = dict(bad_list[0])
        bad_list[0]["call"] = {"name": "get_weather", "arguments": "city=Paris"}
        with self.assertRaises(ValueError):
            validate_toolcall_queries(tuple(bad_list))

    def test_queries_reject_unknown_params(self):
        bad_list = [dict(q) for q in TOOLCALL_QUERIES]
        bad_list[0] = dict(bad_list[0])
        bad_list[0]["call"] = {
            "name": "get_weather",
            "arguments": {"city": "Paris", "unit": "celsius", "extra": "x"},
        }
        with self.assertRaises(ValueError):
            validate_toolcall_queries(tuple(bad_list))

    def test_check_arg_value_boolean_and_bounds(self):
        from model_protocol_lab.toolcall import _check_arg_value

        # Boolean param
        _check_arg_value(True, {"type": "boolean"}, where="t")
        with self.assertRaises(ValueError):
            _check_arg_value("yes", {"type": "boolean"}, where="t")
        with self.assertRaises(ValueError):
            _check_arg_value(1, {"type": "boolean"}, where="t")
        # Wrong type for string param
        with self.assertRaises(ValueError):
            _check_arg_value(123, {"type": "string"}, where="t")
        # Below minimum
        with self.assertRaises(ValueError):
            _check_arg_value(-1, {"type": "number", "minimum": 0}, where="t")
        # Above maximum
        with self.assertRaises(ValueError):
            _check_arg_value(101, {"type": "number", "maximum": 100}, where="t")

    def test_canonical_call_exact(self):
        self.assertEqual(
            canonical_call({"name": "get_weather", "arguments": {"unit": "C", "city": "Paris"}}),
            'CALL get_weather {"city": "Paris", "unit": "C"}',
        )
        self.assertEqual(canonical_call(None), "NO_CALL")

    def test_render_prompt_carries_tools_and_ref(self):
        queries = _queries()
        prompt = render_prompt(queries[0])
        self.assertIn("get_weather", prompt)
        self.assertIn("NO_CALL", prompt)
        self.assertIn("(ref: toolcall/weather-paris)", prompt)
        self.assertIn("What is the weather in Paris", prompt)

    def test_build_fixtures_ids_and_expected(self):
        fixtures = build_toolcall_fixtures(_queries())
        self.assertEqual(len(fixtures), 9)
        by_id = {f.id: f for f in fixtures}
        self.assertEqual(
            by_id["toolcall/weather-paris"].expected,
            'CALL get_weather {"city": "Paris", "unit": "C"}',
        )
        self.assertEqual(by_id["toolcall/abstain-capital"].expected, "NO_CALL")
        self.assertTrue(all(f.group == "toolcall" for f in fixtures))

    def test_query_calls_mapping(self):
        calls = query_calls(_queries())
        self.assertEqual(len(calls), 9)
        self.assertIsNone(calls["toolcall/abstain-capital"])
        self.assertEqual(calls["toolcall/weather-paris"]["name"], "get_weather")


class ClassifyToolCallTests(unittest.TestCase):
    def setUp(self):
        self.calls = query_calls(_queries())

    def test_correct_call_adherent(self):
        result = classify_tool_call(
            'CALL get_weather {"city": "Paris", "unit": "C"}',
            self.calls["toolcall/weather-paris"],
        )
        self.assertEqual(result["outcome"], "adherent")
        self.assertTrue(result["selection_ok"] and result["args_ok"] and result["relevance_ok"])

    def test_correct_call_whitespace_tolerant(self):
        result = classify_tool_call(
            '  CALL get_weather {"unit": "C", "city": "Paris"}  \n',
            self.calls["toolcall/weather-paris"],
        )
        self.assertEqual(result["outcome"], "adherent")

    def test_wrong_tool_call_failed(self):
        result = classify_tool_call(
            'CALL search_docs {"query": "Paris weather", "top_k": 1}',
            self.calls["toolcall/weather-paris"],
        )
        self.assertEqual(result["outcome"], "call_failed")
        self.assertFalse(result["selection_ok"])
        self.assertFalse(result["args_ok"])
        self.assertTrue(result["relevance_ok"])

    def test_unknown_tool_never_correct_selection(self):
        result = classify_tool_call(
            'CALL get_weather_v2 {"city": "Paris", "unit": "C"}',
            self.calls["toolcall/weather-paris"],
        )
        self.assertEqual(result["outcome"], "call_failed")
        self.assertFalse(result["selection_ok"])

    def test_wrong_arg_value_call_failed(self):
        result = classify_tool_call(
            'CALL get_weather {"city": "Paris", "unit": "F"}',
            self.calls["toolcall/weather-paris"],
        )
        self.assertEqual(result["outcome"], "call_failed")
        self.assertTrue(result["selection_ok"])
        self.assertFalse(result["args_ok"])

    def test_wrong_arg_type_call_failed(self):
        result = classify_tool_call(
            'CALL search_docs {"query": "x", "top_k": "3"}',
            self.calls["toolcall/docs-healing"],
        )
        self.assertFalse(result["args_ok"])

    def test_extra_arg_call_failed(self):
        result = classify_tool_call(
            'CALL get_weather {"city": "Paris", "unit": "C", "extra": 1}',
            self.calls["toolcall/weather-paris"],
        )
        self.assertEqual(result["outcome"], "call_failed")
        self.assertFalse(result["args_ok"])

    def test_missing_arg_call_failed(self):
        result = classify_tool_call(
            'CALL get_weather {"city": "Paris"}',
            self.calls["toolcall/weather-paris"],
        )
        self.assertFalse(result["args_ok"])

    def test_number_equality_across_int_float(self):
        result = classify_tool_call(
            'CALL convert_currency {"amount": 100.0, "from_currency": "USD", "to_currency": "EUR"}',
            self.calls["toolcall/currency-usd-eur"],
        )
        self.assertEqual(result["outcome"], "adherent")

    def test_bool_is_not_integer(self):
        result = classify_tool_call(
            'CALL search_docs {"query": "x", "top_k": true}',
            self.calls["toolcall/docs-healing"],
        )
        self.assertFalse(result["args_ok"])

    def test_correct_abstention_adherent(self):
        result = classify_tool_call("NO_CALL", self.calls["toolcall/abstain-capital"])
        self.assertEqual(result["outcome"], "adherent")
        self.assertTrue(result["selection_ok"] and result["args_ok"] and result["relevance_ok"])

    def test_call_when_should_abstain_fails_all(self):
        result = classify_tool_call(
            'CALL get_weather {"city": "Paris", "unit": "C"}',
            self.calls["toolcall/abstain-capital"],
        )
        self.assertEqual(result["outcome"], "call_failed")
        self.assertFalse(result["selection_ok"])
        self.assertFalse(result["args_ok"])
        self.assertFalse(result["relevance_ok"])

    def test_abstain_when_tool_needed_fails_all(self):
        result = classify_tool_call("NO_CALL", self.calls["toolcall/weather-paris"])
        self.assertEqual(result["outcome"], "call_failed")
        self.assertFalse(result["relevance_ok"])

    def test_garbage_parse_failed(self):
        for text in (
            "Sure, here you go!",
            "CALL",
            "CALL get_weather",
            "CALL x [1,2]",
            "CALL get_weather [1,2]",
            "CALL get_weather {bad json}",
        ):
            with self.subTest(text=text):
                result = classify_tool_call(text, self.calls["toolcall/weather-paris"])
                self.assertEqual(result["outcome"], "parse_failed")
                self.assertFalse(result["selection_ok"])
                self.assertFalse(result["args_ok"])
                self.assertFalse(result["relevance_ok"])

    def test_digit_led_name_parses_as_unknown_tool(self):
        # "123" is alnum: parses as a call to an unknown tool -> call_failed,
        # never a phantom parse failure or a correct selection.
        result = classify_tool_call("CALL 123 {}", self.calls["toolcall/weather-paris"])
        self.assertEqual(result["outcome"], "call_failed")
        self.assertFalse(result["selection_ok"])

    def test_non_finite_constant_rejected(self):
        result = classify_tool_call(
            'CALL convert_currency {"amount": NaN, "from_currency": "USD", "to_currency": "EUR"}',
            self.calls["toolcall/currency-usd-eur"],
        )
        self.assertEqual(result["outcome"], "parse_failed")

    def test_non_string_not_text(self):
        for bad in (None, 123, b"CALL x {}"):
            with self.subTest(bad=bad):
                result = classify_tool_call(bad, self.calls["toolcall/weather-paris"])
                self.assertEqual(result["outcome"], "not_text")

    def test_outcome_vocabulary(self):
        self.assertEqual(OUTCOMES, ("adherent", "parse_failed", "call_failed", "not_text"))


class CandidateAdherenceTests(unittest.TestCase):
    def _classify(self, text, expected):
        return classify_tool_call(text, expected)

    def test_perfect_model_scores_one_everywhere(self):
        calls = query_calls(_queries())
        by_id = {q["id"]: q for q in _queries()}
        classifications = [
            classify_tool_call(canonical_call(by_id[qid]["call"]), calls[f"toolcall/{qid}"])
            for qid in [q["id"] for q in _queries()]
        ]
        agg = candidate_adherence(classifications)
        self.assertEqual(agg["n"], 9)
        self.assertEqual(agg["n_evaluable"], 9)
        self.assertEqual(agg["adherent"], 9)
        for key in ("selection_rate", "args_rate", "relevance_rate", "adherent_rate"):
            self.assertEqual(agg[key], 1.0)

    def test_not_text_excluded_not_counted(self):
        calls = query_calls(_queries())
        classifications = [
            classify_tool_call(None, calls["toolcall/weather-paris"]),
            classify_tool_call(
                'CALL get_weather {"city": "Paris", "unit": "C"}',
                calls["toolcall/weather-paris"],
            ),
        ]
        agg = candidate_adherence(classifications)
        self.assertEqual(agg["n"], 2)
        self.assertEqual(agg["n_evaluable"], 1)
        self.assertEqual(agg["not_text"], 1)
        self.assertEqual(agg["adherent_rate"], 1.0)

    def test_empty_evaluable_rates_none(self):
        agg = candidate_adherence([])
        self.assertEqual(agg["n"], 0)
        self.assertIsNone(agg["selection_rate"])
        self.assertIsNone(agg["adherent_rate"])

    def test_decomposition_selection_vs_args(self):
        calls = query_calls(_queries())
        classifications = [
            # right tool, wrong args on both weather queries
            classify_tool_call(
                'CALL get_weather {"city": "Paris", "unit": "F"}',
                calls["toolcall/weather-paris"],
            ),
            classify_tool_call(
                'CALL get_weather {"city": "Berlin", "unit": "C"}',
                calls["toolcall/weather-berlin"],
            ),
        ]
        agg = candidate_adherence(classifications)
        self.assertEqual(agg["selection_rate"], 1.0)
        self.assertEqual(agg["args_rate"], 0.0)
        self.assertEqual(agg["adherent"], 0)


class ToolcallStageTests(unittest.TestCase):
    def _run(self, policy, n_formats=2):
        queries = _queries()
        fixtures = build_toolcall_fixtures(queries)
        calls = query_calls(queries)
        formats = tuple(_format(f"cand{i}") for i in range(n_formats))
        stage = toolcall_stage(
            ToolcallScriptedBackend(policy),
            formats,
            fixtures,
            calls=calls,
            boundary=(124,),
            max_new_tokens=128,
            seed=7,
        )
        return toolcall_adherence(tuple(f.name for f in formats), stage), stage

    def test_perfect_policy_all_adherent(self):
        adherence, stage = self._run(_correct_policy())
        self.assertTrue(adherence["available"])
        for cand, entry in adherence["per_candidate"].items():
            self.assertEqual(entry["adherent_rate"], 1.0, cand)
        self.assertEqual(adherence["adherent_rate_gap"], 0.0)
        self.assertTrue(adherence["uniform"])
        self.assertEqual(adherence["excluded_candidates"], [])

    def test_wrong_tool_policy_selection_zero(self):
        policy = _correct_policy()
        for qid in ("weather-paris", "weather-berlin"):
            policy[qid] = 'CALL search_docs {"query": "x", "top_k": 1}'
        adherence, _ = self._run(policy, n_formats=1)
        entry = adherence["per_candidate"]["cand0"]
        # 2 of 9 wrong selection; relevance still 1.0 (calls were emitted)
        self.assertAlmostEqual(entry["selection_rate"], 7 / 9)
        self.assertEqual(entry["relevance_rate"], 1.0)
        self.assertAlmostEqual(entry["adherent_rate"], 7 / 9)

    def test_always_abstain_policy(self):
        policy = dict.fromkeys(_correct_policy(), "NO_CALL")
        adherence, _ = self._run(policy, n_formats=1)
        entry = adherence["per_candidate"]["cand0"]
        # only the 2 abstain queries are correct
        self.assertAlmostEqual(entry["relevance_rate"], 2 / 9)
        self.assertAlmostEqual(entry["selection_rate"], 2 / 9)
        self.assertAlmostEqual(entry["adherent_rate"], 2 / 9)

    def test_gap_and_weakest_across_candidates(self):
        good = _correct_policy()
        bad = dict.fromkeys(good, "definitely not a call")
        queries = _queries()
        fixtures = build_toolcall_fixtures(queries)
        calls = query_calls(queries)

        def _fmt(name: str, prefix: str) -> Format:
            return Format(
                name=name,
                bos="",
                user_prefix=prefix,
                assistant_prefix="[asst]\n",
                turn_end="",
            )

        class TwoPolicyBackend(ToolcallScriptedBackend):
            def generate(self, input_ids, *, max_new_tokens, seed, cancel):
                prompt = self.decode(tuple(input_ids))
                self.policy = good if "[user0]" in prompt else bad
                return super().generate(
                    input_ids, max_new_tokens=max_new_tokens, seed=seed, cancel=cancel
                )

        formats = (_fmt("cand0", "[user0]\n"), _fmt("cand1", "[user1]\n"))
        stage = toolcall_stage(
            TwoPolicyBackend({}),
            formats,
            fixtures,
            calls=calls,
            boundary=(124,),
            max_new_tokens=128,
            seed=7,
        )
        adherence = toolcall_adherence(("cand0", "cand1"), stage)
        self.assertEqual(adherence["weakest_candidates"], ["cand1"])
        self.assertEqual(adherence["adherent_rate_gap"], 1.0)
        self.assertFalse(adherence["uniform"])

    def test_stage_rejects_bad_contract(self):
        queries = _queries()
        fixtures = build_toolcall_fixtures(queries)
        calls = query_calls(queries)
        backend = ToolcallScriptedBackend({})
        with self.assertRaises(ValueError):
            toolcall_stage(None, (_format("a"),), fixtures, calls=calls, boundary=(124,))
        with self.assertRaises(ValueError):
            toolcall_stage(backend, (), fixtures, calls=calls, boundary=(124,))
        with self.assertRaises(ValueError):
            toolcall_stage(
                backend, (_format("a"),), fixtures, calls={"nope": None}, boundary=(124,)
            )
        with self.assertRaises(ValueError):
            toolcall_stage(backend, (_format("a"),), fixtures, calls=calls, boundary=())

    def test_backend_failure_cell_degrades_to_none(self):
        class BoomBackend(ToolcallScriptedBackend):
            def generate(self, input_ids, *, max_new_tokens, seed, cancel):
                raise RuntimeError("boom")

        queries = _queries()
        stage = toolcall_stage(
            BoomBackend({}),
            (_format("cand0"),),
            build_toolcall_fixtures(queries),
            calls=query_calls(queries),
            boundary=(124,),
            max_new_tokens=128,
            seed=7,
        )
        self.assertIsNone(stage["per_candidate"]["cand0"])
        adherence = toolcall_adherence(("cand0",), stage)
        self.assertEqual(adherence["excluded_candidates"], ["cand0"])

    def test_adherence_invalid_input_unavailable(self):
        self.assertEqual(toolcall_adherence(("a",), {"available": True})["detail"], "stage_invalid")
        self.assertEqual(toolcall_adherence([], {})["detail"], "invalid_candidates")


# ---------------------------------------------------------------------------
# 200-case brute-force oracle differential (independent reimplementation)
# ---------------------------------------------------------------------------


def _oracle_classify(text: object, expected: dict[str, Any] | None) -> dict[str, Any]:
    """Independent oracle for classify_tool_call, written from the docstring."""
    if type(text) is not str:
        return {
            "outcome": "not_text",
            "selection_ok": False,
            "args_ok": False,
            "relevance_ok": False,
        }
    stripped = text.strip()
    should_abstain = expected is None
    if stripped == "NO_CALL":
        relevance_ok = should_abstain
        ok = relevance_ok
        return {
            "outcome": "adherent" if ok else "call_failed",
            "selection_ok": ok,
            "args_ok": ok,
            "relevance_ok": relevance_ok,
        }
    if not stripped.startswith("CALL "):
        return {
            "outcome": "parse_failed",
            "selection_ok": False,
            "args_ok": False,
            "relevance_ok": False,
        }
    rest = stripped[len("CALL ") :]
    if " " not in rest:
        return {
            "outcome": "parse_failed",
            "selection_ok": False,
            "args_ok": False,
            "relevance_ok": False,
        }
    name, args_text = rest.split(" ", 1)
    if not name or not name.replace("_", "").isalnum():
        return {
            "outcome": "parse_failed",
            "selection_ok": False,
            "args_ok": False,
            "relevance_ok": False,
        }
    try:
        arguments = json.loads(args_text)
    except json.JSONDecodeError:
        return {
            "outcome": "parse_failed",
            "selection_ok": False,
            "args_ok": False,
            "relevance_ok": False,
        }
    if type(arguments) is not dict:
        return {
            "outcome": "parse_failed",
            "selection_ok": False,
            "args_ok": False,
            "relevance_ok": False,
        }
    # NaN/Infinity check: json round-trip would differ
    try:
        json.dumps(arguments, allow_nan=False)
    except ValueError:
        return {
            "outcome": "parse_failed",
            "selection_ok": False,
            "args_ok": False,
            "relevance_ok": False,
        }
    relevance_ok = not should_abstain  # a call was emitted
    if should_abstain:
        return {
            "outcome": "call_failed",
            "selection_ok": False,
            "args_ok": False,
            "relevance_ok": False,
        }
    assert expected is not None
    selection_ok = name == expected["name"]
    args_ok = False
    if selection_ok:
        want = expected["arguments"]
        args_ok = set(arguments) == set(want) and all(
            _oracle_value_equal(arguments[k], v) for k, v in want.items()
        )
    ok = selection_ok and args_ok and relevance_ok
    return {
        "outcome": "adherent" if ok else "call_failed",
        "selection_ok": selection_ok,
        "args_ok": args_ok,
        "relevance_ok": relevance_ok,
    }


def _oracle_value_equal(got: object, want: object) -> bool:
    if type(want) is bool or type(got) is bool:
        return type(want) is type(got) and want == got
    if isinstance(want, (int, float)) and isinstance(got, (int, float)):
        return got == want
    return type(want) is type(got) and want == got


def _rand_text(rng: random.Random, queries) -> tuple[str, dict[str, Any] | None]:
    """Random (text, expected) pair exercising every outcome class."""
    q = rng.choice(queries)
    expected = q["call"]
    roll = rng.random()
    if roll < 0.30:
        return canonical_call(expected), expected
    if roll < 0.40:
        return "NO_CALL", expected
    if roll < 0.50 and expected is not None:
        wrong = dict(expected["arguments"])
        key = rng.choice(sorted(wrong))
        wrong[key] = "WRONG_VALUE"
        return canonical_call({"name": expected["name"], "arguments": wrong}), expected
    if roll < 0.60 and expected is not None:
        return (
            canonical_call({"name": "search_docs", "arguments": {"query": "x", "top_k": 1}})
            if expected["name"] != "search_docs"
            else canonical_call({"name": "get_weather", "arguments": {"city": "x", "unit": "C"}}),
            expected,
        )
    if roll < 0.70:
        return rng.choice(
            [
                "garbage",
                "CALL",
                "CALL x",
                "CALL x [1]",
                "CALL x {oops",
                "call get_weather {}",
                "CALL get_weather null",
            ]
        ), expected
    if roll < 0.80 and expected is not None:
        extra = dict(expected["arguments"])
        extra["injected"] = 1
        return canonical_call({"name": expected["name"], "arguments": extra}), expected
    return "  " + canonical_call(expected) + "  " if expected is not None else "NO_CALL", expected


class ToolcallOracleDifferentialTests(unittest.TestCase):
    def test_200_case_oracle_agreement(self):
        rng = random.Random(163)  # noqa: S311 -- deterministic test seed
        queries = _queries()
        for case in range(200):
            text, expected = _rand_text(rng, queries)
            got = classify_tool_call(text, expected)
            want = _oracle_classify(text, expected)
            self.assertEqual(
                {k: got[k] for k in ("outcome", "selection_ok", "args_ok", "relevance_ok")},
                want,
                f"case {case}: text={text!r} expected={expected!r}",
            )

    def test_oracle_aggregation_agreement(self):
        rng = random.Random(164)  # noqa: S311 -- deterministic test seed
        queries = _queries()
        for case in range(50):
            classifications = [
                _oracle_classify(*_rand_text(rng, queries)) for _ in range(rng.randint(1, 9))
            ]
            # oracle aggregation: not_text excluded, rates over evaluable
            evaluable = [c for c in classifications if c["outcome"] != "not_text"]
            n = len(evaluable)
            got = candidate_adherence(
                [dict(c, not_text=(c["outcome"] == "not_text")) for c in classifications]
            )
            self.assertEqual(got["n_evaluable"], n)
            if n:
                for key, rkey in (
                    ("selection_ok", "selection_rate"),
                    ("args_ok", "args_rate"),
                    ("relevance_ok", "relevance_rate"),
                ):
                    self.assertAlmostEqual(
                        got[rkey],
                        sum(1 for c in evaluable if c[key]) / n,
                        msg=f"case {case} {rkey}",
                    )


class ToolcallHypothesisTests(unittest.TestCase):
    @given(st.text())
    @settings(max_examples=200)
    def test_classify_never_raises(self, text):
        queries = _queries()
        calls = query_calls(queries)
        for fid in ("toolcall/weather-paris", "toolcall/abstain-capital"):
            result = classify_tool_call(text, calls[fid])
            self.assertIn(result["outcome"], OUTCOMES)
            for key in ("selection_ok", "args_ok", "relevance_ok"):
                self.assertIs(type(result[key]), bool)

    @given(st.text(min_size=1, max_size=40))
    @settings(max_examples=100)
    def test_canonical_call_round_trips(self, city):
        call = {"name": "get_weather", "arguments": {"city": city, "unit": "C"}}
        text = canonical_call(call)
        name, arguments, error = None, None, None
        from model_protocol_lab.toolcall import _parse_emitted

        name, arguments, error = _parse_emitted(text)
        self.assertIsNone(error)
        self.assertEqual(name, "get_weather")
        self.assertEqual(arguments, {"city": city, "unit": "C"})

    @given(st.integers(), st.integers())
    @settings(max_examples=100)
    def test_int_float_arg_equality(self, a, b):
        from model_protocol_lab.toolcall import _args_match

        self.assertEqual(_args_match({"v": a}, {"v": b}), a == b)
        self.assertEqual(_args_match({"v": float(a)}, {"v": b}), float(a) == b)

    def test_args_match_numeric_mismatch(self):
        from model_protocol_lab.toolcall import _args_match

        # Numeric branch with unequal values -> False
        self.assertFalse(_args_match({"v": 1}, {"v": 2}))
        self.assertFalse(_args_match({"v": 1.5}, {"v": 2.5}))
        self.assertFalse(_args_match({"v": 1}, {"v": 1.5}))
        # Numeric branch with equal values -> True (int/float equality)
        self.assertTrue(_args_match({"v": 100}, {"v": 100.0}))
        # Boolean branch
        self.assertTrue(_args_match({"v": True}, {"v": True}))
        self.assertFalse(_args_match({"v": True}, {"v": False}))
        self.assertFalse(_args_match({"v": True}, {"v": 1}))


# ---------------------------------------------------------------------------
# Tempdir mutation gates (4/4)
# ---------------------------------------------------------------------------


class ToolcallMutationGateTests(unittest.TestCase):
    def _run_gate(self, module_name, name, old_text, new_text, scenario):
        mod = __import__(f"model_protocol_lab.{module_name}", fromlist=["x"])
        source = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertEqual(source.count(old_text), 1, f"mutant {name}: anchor not unique")
        mutated = source.replace(old_text, new_text)
        with tempfile.TemporaryDirectory(prefix="c163_mut_", dir="/tmp") as tmpdir:
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

    def _selection_scenario(self, mod):
        expected = {"name": "get_weather", "arguments": {"city": "Paris", "unit": "C"}}
        result = mod.classify_tool_call('CALL search_docs {"query": "x", "top_k": 1}', expected)
        assert result["outcome"] == "call_failed", result
        assert result["selection_ok"] is False, result

    def test_mutant_selection_inverted_caught(self):
        self._run_gate(
            "toolcall",
            "selection_inverted",
            '    selection_ok = name == expected_call["name"]',
            '    selection_ok = name != expected_call["name"]  # MUTANT',
            self._selection_scenario,
        )

    def _args_strict_scenario(self, mod):
        expected = {"name": "get_weather", "arguments": {"city": "Paris", "unit": "C"}}
        result = mod.classify_tool_call(
            'CALL get_weather {"city": "Paris", "unit": "C", "extra": 1}', expected
        )
        assert result["outcome"] == "call_failed", result
        assert result["args_ok"] is False, result

    def test_mutant_args_lax_caught(self):
        self._run_gate(
            "toolcall",
            "args_lax",
            "    if set(arguments) != set(expected):\n        return False",
            "    if not set(expected) <= set(arguments):  # MUTANT\n        return False",
            self._args_strict_scenario,
        )

    def _relevance_scenario(self, mod):
        result = mod.classify_tool_call("NO_CALL", None)
        assert result["outcome"] == "adherent", result
        assert result["relevance_ok"] is True, result

    def test_mutant_relevance_inverted_caught(self):
        self._run_gate(
            "toolcall",
            "relevance_inverted",
            "    relevance_ok = abstained == should_abstain",
            "    relevance_ok = abstained != should_abstain  # MUTANT",
            self._relevance_scenario,
        )

    def _args_drift_scenario(self, mod):
        old = _tc_diff_payload()
        new = _tc_diff_payload()
        new["per_candidate"]["c"]["args_rate"] = 0.25
        diff = mod.diff_toolcall(old, new)
        assert diff["available"] == "both", diff
        assert diff["changed"] is True, diff
        # Single cause among the rate drifts: only args_drift fires.
        assert diff["args_drift"], diff
        assert not diff["selection_drift"], diff
        assert not diff["relevance_drift"], diff

    def test_mutant_diff_args_drift_ignored_caught(self):
        self._run_gate(
            "diffreports",
            "args_drift_ignored",
            "        or toolcall_args_drift\n",
            "        # or toolcall_args_drift  # MUTANT\n",
            self._args_drift_scenario,
        )


def _tc_entry(**overrides):
    entry = {
        "n": 9,
        "n_evaluable": 9,
        "adherent": 9,
        "parse_failed": 0,
        "call_failed": 0,
        "not_text": 0,
        "selection_rate": 1.0,
        "args_rate": 1.0,
        "relevance_rate": 1.0,
        "adherent_rate": 1.0,
    }
    entry.update(overrides)
    return entry


def _tc_diff_payload():
    return {
        "available": True,
        "n_fixtures": 9,
        "per_candidate": {"c": _tc_entry(), "d": _tc_entry()},
        "weakest_candidates": [],
        "excluded_candidates": [],
    }


class DiffToolcallTests(unittest.TestCase):
    def test_identical_payloads_unchanged(self):
        diff = diffreports.diff_toolcall(_tc_diff_payload(), _tc_diff_payload())
        self.assertEqual(diff["available"], "both")
        self.assertFalse(diff["changed"])
        self.assertEqual(diff["selection_drift"], [])
        self.assertEqual(diff["candidates_added"], [])

    def test_selection_drift_single_cause(self):
        old, new = _tc_diff_payload(), _tc_diff_payload()
        new["per_candidate"]["c"]["selection_rate"] = 0.5
        new["per_candidate"]["c"]["adherent_rate"] = 0.5
        diff = diffreports.diff_toolcall(old, new)
        self.assertTrue(diff["changed"])
        self.assertEqual(len(diff["selection_drift"]), 1)
        self.assertEqual(diff["selection_drift"][0]["candidate"], "c")
        self.assertFalse(diff["args_drift"])
        self.assertFalse(diff["relevance_drift"])

    def test_relevance_drift_single_cause(self):
        old, new = _tc_diff_payload(), _tc_diff_payload()
        new["per_candidate"]["d"]["relevance_rate"] = 0.0
        diff = diffreports.diff_toolcall(old, new)
        self.assertTrue(diff["changed"])
        self.assertEqual(len(diff["relevance_drift"]), 1)
        self.assertFalse(diff["selection_drift"])
        self.assertFalse(diff["args_drift"])

    def test_candidates_added_removed(self):
        old, new = _tc_diff_payload(), _tc_diff_payload()
        new["per_candidate"]["e"] = _tc_entry()
        diff = diffreports.diff_toolcall(old, new)
        self.assertTrue(diff["changed"])
        self.assertEqual(diff["candidates_added"], ["e"])
        old2, new2 = _tc_diff_payload(), _tc_diff_payload()
        del new2["per_candidate"]["d"]
        diff2 = diffreports.diff_toolcall(old2, new2)
        self.assertTrue(diff2["changed"])
        self.assertEqual(diff2["candidates_removed"], ["d"])

    def test_excluded_set_changes(self):
        old, new = _tc_diff_payload(), _tc_diff_payload()
        new["excluded_candidates"] = ["c"]
        diff = diffreports.diff_toolcall(old, new)
        self.assertTrue(diff["changed"])
        self.assertEqual(diff["excluded_added"], ["c"])
        self.assertEqual(diff["excluded_removed"], [])

    def test_asymmetric_availability(self):
        diff = diffreports.diff_toolcall(_tc_diff_payload(), {"available": False})
        self.assertEqual(diff["available"], "old")
        self.assertFalse(diff["changed"])
        neither = diffreports.diff_toolcall({"available": False}, {"available": False})
        self.assertEqual(neither["available"], "neither")

    def test_hostile_payload_rejected(self):
        hostile = _tc_diff_payload()
        hostile["per_candidate"]["c"]["selection_rate"] = "high"
        diff = diffreports.diff_toolcall(hostile, _tc_diff_payload())
        self.assertEqual(diff["available"], "new")  # hostile old side degrades
        self.assertFalse(diff["changed"])

    def test_none_cell_supported(self):
        old, new = _tc_diff_payload(), _tc_diff_payload()
        old["per_candidate"]["c"] = None
        new["per_candidate"]["c"] = _tc_entry()
        diff = diffreports.diff_toolcall(old, new)
        self.assertTrue(diff["changed"])
        self.assertEqual(len(diff["selection_drift"]), 1)
        self.assertIsNone(diff["selection_drift"][0]["old"])


class ToolcallSanitizationTests(unittest.TestCase):
    def test_script_bearing_argument_escaped_in_html(self):
        from model_protocol_lab import reporting

        queries = _queries()
        fixtures = build_toolcall_fixtures(queries)
        calls = query_calls(queries)
        evil = 'CALL search_docs {"query": "<script>alert(1)</script>", "top_k": 1}'
        backend_policy = _correct_policy()
        backend_policy["docs-healing"] = evil
        backend = ToolcallScriptedBackend(backend_policy)
        stage = toolcall_stage(
            backend,
            (_format("evil<script>"),),
            fixtures,
            calls=calls,
            boundary=(124,),
            max_new_tokens=128,
            seed=7,
        )
        result = {"toolcall": toolcall_adherence(("evil<script>",), stage)}
        html = reporting._toolcall_section(result)
        self.assertIn("tool-calls", html)
        self.assertNotIn("<script>alert(1)</script>", html)
        # the candidate name itself is escaped too
        self.assertNotIn("evil<script>", html)
        self.assertIn("evil&lt;script&gt;", html)

    def test_terminal_escapes_stripped_from_stderr_summary(self):
        old, new = _tc_diff_payload(), _tc_diff_payload()
        evil_name = "cand\x1b[31mRED"
        old["per_candidate"][evil_name] = _tc_entry()
        new["per_candidate"][evil_name] = _tc_entry(selection_rate=0.0)
        diff = diffreports.diff_toolcall(old, new)
        lines = diffreports._render_toolcall_summary(diff)
        joined = "\n".join(lines)
        self.assertIn("selection-rate", joined)
        self.assertNotIn("\x1b[31m", joined)

    def test_unavailable_summary_line(self):
        diff = diffreports.diff_toolcall({"available": False}, _tc_diff_payload())
        lines = diffreports._render_toolcall_summary(diff)
        self.assertEqual(len(lines), 1)
        self.assertIn("Tool-call comparison unavailable", lines[0])
        self.assertEqual(diffreports._render_toolcall_summary({"available": "neither"}), [])
        self.assertEqual(diffreports._render_toolcall_summary(None), [])


class ToolcallCliTests(unittest.TestCase):
    def test_probe_never_raises_on_backend_failure(self):
        from model_protocol_lab import cli

        class BoomBackend(ToolcallScriptedBackend):
            def generate(self, input_ids, *, max_new_tokens, seed, cancel):
                raise RuntimeError("boom")

        result: dict = {}
        cli._run_toolcall_probe(
            BoomBackend({}),
            (_format("a"),),
            result,
            boundary=(124,),
            boundary_map=None,
            max_new_tokens=128,
            seed=1,
        )
        # The stage degrades each cell to None fail-closed (never raises);
        # the analysis honestly reports the candidate as excluded.
        self.assertTrue(result["toolcall"]["available"])
        self.assertIsNone(result["toolcall"]["per_candidate"]["a"])
        self.assertEqual(result["toolcall"]["excluded_candidates"], ["a"])

    def test_probe_records_available(self):
        from model_protocol_lab import cli

        result: dict = {}
        cli._run_toolcall_probe(
            ToolcallScriptedBackend(_correct_policy()),
            (_format("a"),),
            result,
            boundary=(124,),
            boundary_map=None,
            max_new_tokens=128,
            seed=1,
        )
        self.assertTrue(result["toolcall"]["available"])
        self.assertEqual(result["toolcall"]["per_candidate"]["a"]["adherent_rate"], 1.0)

    def test_flag_registered(self):
        from model_protocol_lab import cli as cli_module

        parser = cli_module.build_parser()
        args = parser.parse_args(
            [
                "run",
                "--tool-call",
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
        self.assertTrue(args.tool_call)


class ToolcallReportIntegrationTests(unittest.TestCase):
    def test_toc_anchor_present(self):
        from model_protocol_lab import reporting

        queries = _queries()
        stage = toolcall_stage(
            ToolcallScriptedBackend(_correct_policy()),
            (_format("a"),),
            build_toolcall_fixtures(queries),
            calls=query_calls(queries),
            boundary=(124,),
            max_new_tokens=128,
            seed=7,
        )
        result = {"toolcall": toolcall_adherence(("a",), stage)}
        html = reporting._toolcall_section(result)
        self.assertIn('id="tool-calls"', html)
        self.assertIn("Prompt-only tool-call adherence", html)

    def test_section_empty_when_unavailable(self):
        from model_protocol_lab import reporting

        self.assertEqual(reporting._toolcall_section({}), "")
        self.assertEqual(reporting._toolcall_section({"toolcall": {"available": False}}), "")

    def test_full_diff_reports_toolcall_section(self):
        def report():
            return {
                "decision": "select_best",
                "selected_candidate": "c",
                "search": [],
                "toolcall": _tc_diff_payload(),
            }

        diff = diffreports.diff_reports(report(), report())
        self.assertIn("toolcall", diff)
        self.assertEqual(diff["toolcall"]["available"], "both")
        self.assertFalse(diff["summary"]["toolcall_changed"])
        self.assertFalse(diff["summary"]["toolcall_asymmetric"])
        self.assertTrue(diff["summary"]["identical"])

    def test_full_diff_reports_toolcall_changed_breaks_identical(self):
        def report(payload):
            return {
                "decision": "select_best",
                "selected_candidate": "c",
                "search": [],
                "toolcall": payload,
            }

        new_payload = _tc_diff_payload()
        new_payload["per_candidate"]["c"]["selection_rate"] = 0.0
        diff = diffreports.diff_reports(report(_tc_diff_payload()), report(new_payload))
        self.assertTrue(diff["summary"]["toolcall_changed"])
        self.assertFalse(diff["summary"]["identical"])
        stderr_lines = diffreports.render_summary(diff)
        self.assertIn("Tool-call", stderr_lines)


if __name__ == "__main__":
    unittest.main()
