"""W2 Push 1 — input-boundary totality (audit RR-02, RTI-03/04/06,
SA-01/03, wave-9 S3-1/S3-2/S4-1/S4-2).

Every defect case below was reproduced against the pre-fix tree (the
audit re-verification lane's reproducers, re-run by the W2
coordinator): a malformed section, artifact, server response or RPC
frame escaped as a raw TypeError / OverflowError / KeyError /
UnicodeDecodeError, or was silently admitted, instead of degrading
through the module's own fixed vocabulary. Valid controls pin that
well-formed neighbours keep their exact pre-fix behaviour.

Spec bases: validators gate shape before consumption (the wave-8 F4
discipline); JSON-RPC 2.0 section 5.1 (-32600 Invalid Request for a
frame that is not a valid Request object, -32602 Invalid params);
JSON values compare under the JSON value model, never Python
coercion.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any
from unittest import mock

from model_protocol_lab import diffreports, mcp_server, servingbackends, surfacenoise, toolcall
from model_protocol_lab.core import Generation

# ---------------------------------------------------------------------------
# RR-02: diff consumers are total over malformed sections
# ---------------------------------------------------------------------------

_FRONTIER = {
    "max_new_tokens": 64,
    "budget_robust": True,
    "candidates": {
        "alpha": {"fixture_ids": ["f1", "f2"], "pass_steps": [[1, 0], [32, 1], [64, 2]]}
    },
    "decision_steps": [[32, "no_candidate", None], [64, "compatible_on_fixture_set", "alpha"]],
    "flip_ranges": [[1, 31]],
    "safe_budget_floor": 32,
}

_FLOOR = {
    "available": True,
    "native_verdict": "compatible_on_fixture_set",
    "max_new_tokens": 64,
    "safe_budget_floor": 32,
    "budget_robust": True,
}


def _wrap(key: str, payload: Any) -> dict[str, Any]:
    return {
        "decision": "no_candidate",
        "selected_candidate": None,
        "search": [],
        "held_out": None,
        key: payload,
    }


class FrontierTotalityTests(unittest.TestCase):
    """RR-02 frontier shapes: scalar fields must not raise."""

    def test_scalar_decision_steps_skips_decision_axes(self) -> None:
        """decision_steps=1: the decision/winner axes skip (the
        differ's documented reading of malformed steps); the rest of
        the frontier still compares and nothing raises."""
        bad = copy.deepcopy(_FRONTIER)
        bad["decision_steps"] = 1
        diff = diffreports.diff_reports(
            _wrap("budget_frontier", _FRONTIER), _wrap("budget_frontier", bad)
        )
        section = diff["frontier"]
        self.assertEqual(section["available"], "both")
        self.assertEqual(section["decision_flips"], [])
        self.assertEqual(section["winner_flips"], [])

    def test_scalar_flip_ranges_without_recorded_floor_degrades(self) -> None:
        """flip_ranges=5 on a pre-field payload (no recorded floor):
        the section degrades instead of raising in the derivation."""
        old = copy.deepcopy(_FRONTIER)
        del old["safe_budget_floor"]
        bad = copy.deepcopy(old)
        bad["flip_ranges"] = 5
        diff = diffreports.diff_reports(
            _wrap("budget_frontier", old), _wrap("budget_frontier", bad)
        )
        self.assertNotEqual(diff["frontier"]["available"], "both")

    def test_scalar_pass_steps_in_one_candidate_does_not_raise(self) -> None:
        """pass_steps=7 in one candidate: the trajectory derivation
        skips the malformed series instead of raising."""
        bad = copy.deepcopy(_FRONTIER)
        bad["candidates"]["alpha"]["pass_steps"] = 7
        bad["candidates"]["beta"] = {"fixture_ids": ["f1"], "pass_steps": [[1, 1]]}
        diff = diffreports.diff_reports(
            _wrap("budget_frontier", _FRONTIER), _wrap("budget_frontier", bad)
        )
        self.assertIn(diff["frontier"]["available"], ("both", "old", "new", "neither"))

    def test_list_budget_robust_degrades_and_renders(self) -> None:
        """budget_robust=[]: the section is invalid (a list is not a
        bool) and the summary renders — pre-fix the renderer raised
        TypeError: unhashable type: 'list' on the yes/no lookup."""
        bad = copy.deepcopy(_FRONTIER)
        bad["budget_robust"] = []
        diff = diffreports.diff_reports(
            _wrap("budget_frontier", _FRONTIER), _wrap("budget_frontier", bad)
        )
        self.assertNotEqual(diff["frontier"]["available"], "both")
        summary = diffreports.render_summary(diff)
        self.assertIn("frontier", summary.lower())

    def test_valid_frontier_pair_still_identical(self) -> None:
        """Control: the well-formed pair compares identical."""
        diff = diffreports.diff_reports(
            _wrap("budget_frontier", _FRONTIER),
            _wrap("budget_frontier", copy.deepcopy(_FRONTIER)),
        )
        self.assertEqual(diff["frontier"]["available"], "both")
        self.assertFalse(diff["frontier"]["changed"])


class HeldOutFloorTotalityTests(unittest.TestCase):
    """RR-02 held-out floor: the partial-success artifact case."""

    def test_list_budget_robust_degrades_and_renders(self) -> None:
        """budget_robust=[]: diff completes AND the summary renders.
        Pre-fix the differ emitted a valid diff (valid stdout JSON
        through the CLI) and render_summary then raised TypeError."""
        bad = copy.deepcopy(_FLOOR)
        bad["budget_robust"] = []
        diff = diffreports.diff_reports(
            _wrap("held_out_budget_floor", _FLOOR), _wrap("held_out_budget_floor", bad)
        )
        self.assertNotEqual(diff["held_out_floor"]["available"], "both")
        self.assertTrue(diffreports.render_summary(diff))

    def test_valid_bool_change_still_compared(self) -> None:
        """Control: budget_robust True -> False is a real change."""
        good = copy.deepcopy(_FLOOR)
        good["budget_robust"] = False
        diff = diffreports.diff_reports(
            _wrap("held_out_budget_floor", _FLOOR), _wrap("held_out_budget_floor", good)
        )
        self.assertEqual(diff["held_out_floor"]["available"], "both")
        self.assertTrue(diff["held_out_floor"]["changed"])


class PrecisionSortTotalityTests(unittest.TestCase):
    """RR-02 precision: element types gate before any sorting."""

    def _diff_with_unavailable(self, unavailable: Any) -> dict[str, Any]:
        from test_cycle198_precision import _valid_payload

        valid = _valid_payload()
        mutated = copy.deepcopy(valid)
        mutated["unavailable_candidates"] = unavailable
        return diffreports.diff_reports(_wrap("precision", valid), _wrap("precision", mutated))

    def test_mixed_type_unavailable_does_not_raise(self) -> None:
        """[1, "zzz"] and ["zzz", 1]: pre-fix sorted() raised TypeError
        before the element-type check could reject the payload."""
        for bad in ([1, "zzz"], ["zzz", 1]):
            diff = self._diff_with_unavailable(bad)
            self.assertNotEqual(diff["precision"]["available"], "both")

    def test_valid_unavailable_still_validates(self) -> None:
        """Control: the genuine payload's sorted string list is fine."""
        from test_cycle198_precision import _valid_payload

        valid = _valid_payload()
        diff = diffreports.diff_reports(
            _wrap("precision", valid), _wrap("precision", copy.deepcopy(valid))
        )
        self.assertEqual(diff["precision"]["available"], "both")
        self.assertTrue(diff["summary"]["identical"])


class DiffReportsOutputAtomicityTests(unittest.TestCase):
    """RR-02: both outputs are fully built before either is emitted,
    so a pair can never yield a valid JSON artifact and then fail."""

    def _write_pair(self, tmp: Path, new_floor: Any) -> tuple[Path, Path]:
        old_doc = _wrap("held_out_budget_floor", _FLOOR)
        new_doc = _wrap("held_out_budget_floor", new_floor)
        old_path = tmp / "old.json"
        new_path = tmp / "new.json"
        old_path.write_text(json.dumps(old_doc), encoding="utf-8")
        new_path.write_text(json.dumps(new_doc), encoding="utf-8")
        return old_path, new_path

    def test_cli_pair_with_list_budget_robust_exits_zero(self) -> None:
        """The audit's CLI case end to end: exit 0, parseable JSON on
        stdout, a summary on stderr, no traceback anywhere."""
        import tempfile

        bad = copy.deepcopy(_FLOOR)
        bad["budget_robust"] = []
        with tempfile.TemporaryDirectory() as td:
            old_path, new_path = self._write_pair(Path(td), bad)
            args = argparse.Namespace(old_results=old_path, new_results=new_path, all=False)
            out, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = diffreports.run_diff_reports(args, argparse.ArgumentParser())
        self.assertEqual(code, 0)
        parsed = json.loads(out.getvalue())
        self.assertIn("summary", parsed)
        self.assertNotIn("Traceback", err.getvalue())
        self.assertTrue(err.getvalue().strip())


# ---------------------------------------------------------------------------
# SA-01: the template-case evaluator honours the generation contract
# ---------------------------------------------------------------------------


class _StubBackend:
    """Minimal Backend-shaped stub returning a canned generation."""

    def __init__(self, gen: Any) -> None:
        self._gen = gen

    def encode(self, text: str) -> tuple[int, ...]:
        return (1, 2, 3)

    def generate(self, ids: Any, *, max_new_tokens: int, seed: int, cancel: Any) -> Any:
        return self._gen

    def decode(self, ids: Any) -> str:
        return "BLUE"


class TemplateCaseGenerationGateTests(unittest.TestCase):
    """SA-01: _evaluate_template_case must apply core._valid_generation
    exactly where _evaluate_one does."""

    def _eval(self, gen: Any) -> str:
        return surfacenoise._evaluate_template_case(
            lambda messages: "prompt",
            (),
            "BLUE",
            "case1",
            _StubBackend(gen),
            (9,),
            4,
            0,
            threading.Event(),
        )

    def test_none_generation_abstains(self) -> None:
        """Pre-fix: AttributeError on gen.reason."""
        self.assertEqual(self._eval(None), "abstain")

    def test_short_limit_generation_abstains(self) -> None:
        """A limit-stopped generation short of the cap proves nothing
        (core._valid_generation requires len == cap for 'limit');
        pre-fix it was credited as a pass."""
        gen = Generation(token_ids=(5, 6), reason="limit")
        self.assertEqual(self._eval(gen), "abstain")

    def test_full_limit_generation_still_passes(self) -> None:
        """Control: a cap-length limit generation scoring the expected
        answer still passes."""
        gen = Generation(token_ids=(5, 6, 9, 7), reason="limit")
        self.assertEqual(self._eval(gen), "pass")


# ---------------------------------------------------------------------------
# SA-03: duplicate JSON members in an emitted tool call are ambiguous
# ---------------------------------------------------------------------------


class ToolcallDuplicateMemberTests(unittest.TestCase):
    """SA-03: a duplicated argument member must not collapse to
    whichever value json.loads happens to keep; the emission is a
    parse failure, like every other loader in the tree."""

    _EXPECTED = {"name": "get_weather", "arguments": {"city": "Lisbon"}}

    def test_duplicate_member_null_first_is_parse_failed(self) -> None:
        """Pre-fix this graded 'adherent' (the null was discarded)."""
        outcome = toolcall.classify_tool_call(
            'CALL get_weather {"city": null, "city": "Lisbon"}', self._EXPECTED
        )
        self.assertEqual(outcome["outcome"], "parse_failed")

    def test_duplicate_member_value_first_is_parse_failed(self) -> None:
        """Pre-fix this graded 'call_failed' — the same emission graded
        two ways depending on member order."""
        outcome = toolcall.classify_tool_call(
            'CALL get_weather {"city": "Lisbon", "city": null}', self._EXPECTED
        )
        self.assertEqual(outcome["outcome"], "parse_failed")

    def test_nested_duplicate_member_is_parse_failed(self) -> None:
        """Duplicates at any nesting level are ambiguous."""
        name, arguments, error = toolcall._parse_emitted(
            'CALL get_weather {"where": {"city": "Lisbon", "city": "Porto"}}'
        )
        self.assertIsNone(name)
        self.assertIsNone(arguments)
        self.assertEqual(error, "bad_json")

    def test_clean_call_still_adherent(self) -> None:
        """Control: the unduplicated call grades adherent."""
        outcome = toolcall.classify_tool_call('CALL get_weather {"city": "Lisbon"}', self._EXPECTED)
        self.assertEqual(outcome["outcome"], "adherent")


# ---------------------------------------------------------------------------
# RTI-06: MCP envelope validation before dispatch
# ---------------------------------------------------------------------------


class McpEnvelopeTests(unittest.TestCase):
    """RTI-06: JSON-RPC 2.0 section 5.1 — a frame that is not a valid
    Request object is an Invalid Request (-32600) and never executes;
    params-shape problems stay -32602."""

    def test_wrong_jsonrpc_version_rejected(self) -> None:
        """Pre-fix a 1.0 frame with an object id was answered."""
        response = mcp_server.handle_request({"jsonrpc": "1.0", "id": 1, "method": "ping"})
        self.assertEqual(response["error"]["code"], -32600)

    def test_object_id_rejected_with_null_id(self) -> None:
        """An id that cannot be echoed is reported with a null id."""
        response = mcp_server.handle_request({"jsonrpc": "2.0", "id": {"x": 1}, "method": "ping"})
        self.assertEqual(response["error"]["code"], -32600)
        self.assertIsNone(response["id"])

    def test_list_arguments_rejected_not_executed(self) -> None:
        """Pre-fix arguments=[] was coerced to {} and the tool ran."""
        response = mcp_server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 7,
                "method": "tools/call",
                "params": {"name": "surface_noise_protocol_info", "arguments": []},
            }
        )
        self.assertEqual(response["error"]["code"], -32602)

    def test_false_arguments_rejected_not_executed(self) -> None:
        """Pre-fix arguments=false was coerced to {} and the tool ran."""
        response = mcp_server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 8,
                "method": "tools/call",
                "params": {"name": "surface_noise_protocol_info", "arguments": False},
            }
        )
        self.assertEqual(response["error"]["code"], -32602)

    def test_valid_ping_still_answered(self) -> None:
        """Control: a well-formed ping returns an empty result."""
        response = mcp_server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "ping"})
        self.assertEqual(response["result"], {})

    def test_null_id_request_still_dispatches(self) -> None:
        """Control: an explicit null id is a request (S-13), not a
        notification, and dispatches normally."""
        response = mcp_server.handle_request({"jsonrpc": "2.0", "id": None, "method": "ping"})
        self.assertEqual(response["result"], {})

    def test_notification_still_never_executed(self) -> None:
        """Control: no id member -> notification -> never answered,
        never executed, whatever the envelope."""
        response = mcp_server.handle_request(
            {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": "surface_noise_protocol_info", "arguments": {}},
            }
        )
        self.assertIsNone(response)

    def test_unknown_method_still_method_not_found(self) -> None:
        """Control: a valid envelope with an unknown method is -32601."""
        response = mcp_server.handle_request({"jsonrpc": "2.0", "id": 3, "method": "nope"})
        self.assertEqual(response["error"]["code"], -32601)


# ---------------------------------------------------------------------------
# RTI-03 (MCP half): cheap params validate before backend construction
# ---------------------------------------------------------------------------


class McpValidationOrderTests(unittest.TestCase):
    """RTI-03: tool_probe must reject an out-of-range cap/seed before
    _build_backend runs (construction can load a checkpoint)."""

    _PARAMS = {
        "prompt": "x",
        "fixtures": [
            {
                "id": "f1",
                "messages": [{"role": "user", "content": "q"}],
                "expected": "BLUE",
            }
        ],
        "boundary": [9],
        "backend": {"kind": "selftest"},
    }

    def test_cap_rejected_before_construction(self) -> None:
        """Pre-fix the spy recorded one construction before the -32602."""
        with (
            mock.patch.object(mcp_server, "_build_backend", wraps=mcp_server._build_backend) as spy,
            self.assertRaises(mcp_server.MCPError) as caught,
        ):
            mcp_server.tool_probe({**self._PARAMS, "max_new_tokens": 0})
        self.assertEqual(caught.exception.code, -32602)
        self.assertEqual(spy.call_count, 0)

    def test_seed_rejected_before_construction(self) -> None:
        """Same order for the seed range check."""
        with (
            mock.patch.object(mcp_server, "_build_backend", wraps=mcp_server._build_backend) as spy,
            self.assertRaises(mcp_server.MCPError) as caught,
        ):
            mcp_server.tool_probe({**self._PARAMS, "seed": -1})
        self.assertEqual(caught.exception.code, -32602)
        self.assertEqual(spy.call_count, 0)


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Serving-backend fixtures (cycle-243 fakes: GGUF + hostile HTTP server)
# ---------------------------------------------------------------------------

import sys  # noqa: E402

sys.path.insert(0, "tests")

from test_cycle243_serving_backends import (  # noqa: E402
    EOS,
    ServingBackendsTestBase,
    _FakeServer,
)

from model_protocol_lab import doctor  # noqa: E402
from model_protocol_lab.servingbackends import (  # noqa: E402
    StackRequest,
    tier_a_sampler_params,
)


class ExecBackendsValidationOrderTests(ServingBackendsTestBase):
    """RTI-03 (CLI half): every CLI-local validation in run_exec_backends
    runs before the tokenizer load and the backend probe — an invalid
    invocation must never pay for (or be masked by) expensive setup.
    Pre-fix a tokenizer-load failure answered first in all three
    defect cases below."""

    def _run_with_spies(self, **overrides: Any) -> tuple[Any, mock.MagicMock]:
        import tempfile

        out = Path(tempfile.mkdtemp()) / "records.jsonl"
        args = self._args(None, out, **overrides)
        tokenizer_spy = mock.MagicMock(side_effect=AssertionError("tokenizer loaded"))
        backend_spy = mock.MagicMock(side_effect=AssertionError("backend built"))
        err = io.StringIO()
        result: Any = None
        with (
            mock.patch.object(doctor, "_build_primary_tokenizer", tokenizer_spy),
            mock.patch.object(servingbackends, "LlamacppHttpBackend", backend_spy),
            redirect_stderr(err),
        ):
            try:
                result = servingbackends.run_exec_backends(args, argparse.ArgumentParser())
            except SystemExit as exc:  # parser.error path
                result = exc
        return result, tokenizer_spy

    def test_empty_stop_part_rejected_before_setup(self) -> None:
        """--stops with an empty member: parser exit 2, no setup."""
        result, spy = self._run_with_spies(stops="STOP,,END")
        self.assertIsInstance(result, SystemExit)
        self.assertEqual(result.code, 2)
        self.assertEqual(spy.call_count, 0)

    def test_overlong_stop_text_rejected_before_setup(self) -> None:
        """A stop text past the budget: invalid_stop_texts exit 2."""
        result, spy = self._run_with_spies(stops="x" * 1025)
        self.assertEqual(result, 2)
        self.assertEqual(spy.call_count, 0)

    def test_missing_results_file_rejected_before_setup(self) -> None:
        """An unreadable --results file: exit 2, no setup."""
        import tempfile

        missing = Path(tempfile.mkdtemp()) / "nope.json"
        result, spy = self._run_with_spies(stops="STOP", results=missing)
        self.assertEqual(result, 2)
        self.assertEqual(spy.call_count, 0)

    def test_zero_cap_rejected_before_probe(self) -> None:
        """max_new_tokens=0: the request's own validation answers
        (exit 2) before any backend construction or probe."""
        result, spy = self._run_with_spies(max_new_tokens=0)
        self.assertEqual(result, 2)
        self.assertEqual(spy.call_count, 0)

    def test_valid_invocation_still_reaches_setup(self) -> None:
        """Control: with everything valid the tokenizer load is the
        first expensive step — its failure still answers exit 1."""

        def _no_tokenizer(*_a: Any, **_k: Any) -> Any:
            raise ValueError("tokenizer_files_missing")

        import tempfile

        out = Path(tempfile.mkdtemp()) / "records.jsonl"
        args = self._args(None, out)
        with (
            mock.patch.object(doctor, "_build_primary_tokenizer", _no_tokenizer),
            redirect_stderr(io.StringIO()),
        ):
            code = servingbackends.run_exec_backends(args, argparse.ArgumentParser())
        self.assertEqual(code, 1)


class RecordsArtifactReaderTests(unittest.TestCase):
    """RTI-04: load_records_artifact is a bounded, strict, total
    reader — the cross-run reader must fail with its own fixed code,
    never leak codec/type errors (wave-9 A8-1/A8-2). The artifact is
    JSONL envelope lines: {"header": {..., "plan": [...]}}, then
    {"record": {...}} lines, then {"footer": {"complete": ...}}."""

    def _record(self, **overrides: Any) -> dict[str, Any]:
        record: dict[str, Any] = {
            "candidate": "cand-a",
            "fixture": "fx-1",
            "cache_position": "standalone",
            "cell_detail": "",
            "requested": {"seed": 7},
            "mirror": {"fidelity_verdict": "agree"},
        }
        record.update(overrides)
        return record

    def _header(self, candidates: Any) -> dict[str, Any]:
        return {
            "header": {
                "plan": [
                    {
                        "candidate": c,
                        "fixture": "fx-1",
                        "cache_position": "standalone",
                    }
                    for c in candidates
                ]
            }
        }

    def _write(self, tmp: Path, lines: list[str]) -> Path:
        path = tmp / "records.jsonl"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def _lines(self, record: dict[str, Any], candidates: Any = ("cand-a",)) -> list[str]:
        return [
            json.dumps(self._header(list(candidates))),
            json.dumps({"record": record}),
            json.dumps({"footer": {"complete": True}}),
        ]

    def test_genuine_artifact_loads_with_missing_accounting(self) -> None:
        """Control: one executed cell of a two-cell plan loads; the
        other is reported missing, never silently dropped."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            path = self._write(Path(td), self._lines(self._record(), ("cand-a", "cand-b")))
            result = servingbackends.load_records_artifact(path)
        self.assertEqual(len(result["records"]), 1)
        self.assertEqual(len(result["missing"]), 1)
        self.assertEqual(result["missing"][0]["cell_detail"], "cell_not_executed")

    def test_non_utf8_artifact_is_the_fixed_code(self) -> None:
        """Pre-fix: UnicodeDecodeError escaped."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "records.jsonl"
            path.write_bytes(b'{"header": {}}\n\xff\xfe\n')
            with self.assertRaises(ValueError) as caught:
                servingbackends.load_records_artifact(path)
        self.assertEqual(str(caught.exception), "serving_backends_artifact_invalid")

    def test_non_finite_literal_is_the_fixed_code(self) -> None:
        """A bare NaN is not strict JSON; pre-fix json.loads accepted
        it and the record loaded."""
        import tempfile

        lines = self._lines(self._record())
        lines[1] = lines[1].replace('"seed": 7', '"seed": NaN')
        with tempfile.TemporaryDirectory() as td:
            path = self._write(Path(td), lines)
            with self.assertRaises(ValueError) as caught:
                servingbackends.load_records_artifact(path)
        self.assertEqual(str(caught.exception), "serving_backends_artifact_invalid")

    def test_oversized_artifact_is_the_fixed_code(self) -> None:
        """The reader is bounded: an artifact past the record sink's
        own budget (+ its bounded finalize tail) is refused unread.
        Pre-fix the whole file was read and parsed."""
        import tempfile

        record = self._record(cell_detail="x" * (servingbackends.MAX_RECORD_BYTES + 70_000))
        with tempfile.TemporaryDirectory() as td:
            path = self._write(Path(td), self._lines(record))
            with self.assertRaises(ValueError) as caught:
                servingbackends.load_records_artifact(path)
        self.assertEqual(str(caught.exception), "serving_backends_artifact_invalid")

    def test_list_candidate_in_record_is_accounted_not_raised(self) -> None:
        """A8-1: a record whose candidate is a list cannot key a plan
        cell; it stays in the records list, the genuine cell reports
        missing, and nothing raises. Pre-fix: TypeError."""
        import tempfile

        record = self._record(candidate=["x"])
        with tempfile.TemporaryDirectory() as td:
            path = self._write(Path(td), self._lines(record))
            result = servingbackends.load_records_artifact(path)
        self.assertEqual(len(result["records"]), 1)
        self.assertEqual(len(result["missing"]), 1)

    def test_list_candidate_in_plan_is_accounted_not_raised(self) -> None:
        """A plan descriptor with a non-string candidate can never
        match a record; it reports missing instead of raising."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            path = self._write(Path(td), self._lines(self._record(), (["x"],)))
            result = servingbackends.load_records_artifact(path)
        self.assertEqual(len(result["missing"]), 1)

    def test_finalized_document_form_still_loads(self) -> None:
        """Control: the single-line finalized payload form loads."""
        import tempfile

        doc = {"serving_backends": {"records": [self._record()]}}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "results.json"
            path.write_text(json.dumps(doc), encoding="utf-8")
            result = servingbackends.load_records_artifact(path)
        self.assertTrue(result["finalized"])
        self.assertEqual(len(result["records"]), 1)


# ---------------------------------------------------------------------------
# Wave-9 S3-1 / S3-2 / S4-1 / S4-2
# ---------------------------------------------------------------------------


class EchoComparisonTotalityTests(ServingBackendsTestBase):
    """S3-1: comparing a server-echoed sampler field never converts
    through float() — a huge echoed integer is a mismatch, not an
    OverflowError."""

    def test_huge_echoed_ints_are_mismatches(self) -> None:
        """Pre-fix: OverflowError from float(10**400)."""
        for field in ("temperature", "top_p", "min_p", "repeat_penalty", "dynatemp_range"):
            self.assertFalse(
                servingbackends._echo_matches({field: -(10**400)}, {field: 0.0}),
                field,
            )

    def test_genuine_echo_still_matches(self) -> None:
        """Control: exact echo equality is unchanged (0 == 0.0 across
        the int/float spellings both name the same JSON number)."""
        self.assertTrue(servingbackends._echo_matches({"temperature": 0}, {"temperature": 0.0}))
        self.assertTrue(servingbackends._echo_matches({"temperature": 0.5}, {"temperature": 0.5}))
        self.assertFalse(servingbackends._echo_matches({"temperature": 0.5}, {"temperature": 0.6}))

    def test_execute_cell_banks_mismatch_not_crash(self) -> None:
        """End to end through the fake server: the poisoned echo banks
        a config_echo_mismatch record (pre-fix: OverflowError)."""
        with _FakeServer(self.gguf) as server:
            backend = self.probed_backend(server)
            server.state.echo_override = {"temperature": -(10**400)}
            record = servingbackends.execute_cell(
                backend,
                request=StackRequest(prompt_ids=self.sentinel(), seed=7, max_new_tokens=4),
                decode=backend.decode_ids,
                candidate="t",
                fixture="t1",
            )
        self.assertEqual(record["cell_detail"], "config_echo_mismatch")


class OovTokenClassificationTests(ServingBackendsTestBase):
    """S3-2: a returned token id the local tokenizer cannot decode is
    the stack_response_invalid observation, never a raw KeyError out
    of execute (the predictor assumes returned ids decode)."""

    def _payload(self, tokens: Any) -> dict[str, Any]:
        return {
            "tokens": tokens,
            "content": "abc",
            "stop": True,
            "stop_type": "limit",
            "generation_settings": tier_a_sampler_params(7),
            "timings": {"cache_n": 0},
            "tokens_cached": 0,
            "tokens_evaluated": 3,
        }

    def _classify(self, server: _FakeServer, tokens: Any) -> Any:
        backend = self.probed_backend(server)
        request = StackRequest(
            prompt_ids=(1, 2, 3), seed=7, max_new_tokens=4, stop_texts=("STOP",), eos_id=EOS
        )
        return backend._classify(request, self._payload(tokens))

    def test_oov_token_is_invalid_observation(self) -> None:
        """999999 is outside the fake vocab: pre-fix the record path
        raised KeyError when the predictor decoded the winner."""
        with _FakeServer(self.gguf) as server:
            obs = self._classify(server, [6, 999999])
        self.assertEqual(obs.detail, "stack_response_invalid")
        self.assertIsNone(obs.returned_ids)

    def test_negative_and_huge_tokens_are_invalid_observations(self) -> None:
        """Negative and astronomically large ids likewise."""
        with _FakeServer(self.gguf) as server:
            for tokens in ([-1], [10**400]):
                obs = self._classify(server, tokens)
                self.assertEqual(obs.detail, "stack_response_invalid", tokens)

    def test_in_vocab_stream_still_classifies(self) -> None:
        """Control: a decodable stream classifies as before."""
        with _FakeServer(self.gguf) as server:
            obs = self._classify(server, [6])
        self.assertEqual(obs.detail, "")
        self.assertEqual(obs.returned_ids, (6,))


class ExportedApiTotalityTests(unittest.TestCase):
    """S4-1 / S4-2: the exported aggregation/prediction helpers are
    total over malformed mirrors and stop lists."""

    def test_unhashable_verdict_counts_unobservable(self) -> None:
        """Pre-fix: TypeError: unhashable type: 'list'."""
        observations = [{"mirror": {"fidelity_verdict": []}}]
        result = servingbackends.aggregate_fidelity(observations)
        self.assertEqual(result["verdict"], "fidelity_unavailable")
        self.assertEqual(result["counts"]["unobservable"], 1)

    def test_dict_verdict_counts_unobservable(self) -> None:
        """A dict verdict is likewise not a verdict string."""
        observations = [{"mirror": {"fidelity_verdict": {"x": 1}}}]
        result = servingbackends.aggregate_fidelity(observations)
        self.assertEqual(result["counts"]["unobservable"], 1)

    def test_genuine_verdicts_still_count(self) -> None:
        """Control: string verdicts aggregate as before (a None verdict
        was already counted unobservable)."""
        observations = [
            {"mirror": {"fidelity_verdict": "agree"}},
            {"mirror": {"fidelity_verdict": None}},
        ]
        result = servingbackends.aggregate_fidelity(observations)
        # Partial coverage resolves toward unavailable (the function's
        # own stated rule); a fully agreeing set is fidelity_agree.
        self.assertEqual(result["verdict"], "fidelity_unavailable")
        self.assertEqual(result["counts"]["agree"], 1)
        self.assertEqual(result["counts"]["unobservable"], 1)
        pure = servingbackends.aggregate_fidelity([{"mirror": {"fidelity_verdict": "agree"}}])
        self.assertEqual(pure["verdict"], "fidelity_agree")

    def test_bytes_stop_text_is_the_modules_own_error(self) -> None:
        """Pre-fix: TypeError from `text in prefix` — a bytes member
        is invalid_stop_texts, the same code the CLI validator uses."""
        with self.assertRaises(ValueError) as caught:
            servingbackends.predict_stop_outcome(
                decode=lambda ids: "hello STOP",
                returned_ids=(1,),
                stop_texts=(b"STOP",),
                eos_id=None,
            )
        self.assertEqual(str(caught.exception), "invalid_stop_texts")

    def test_genuine_prediction_unchanged(self) -> None:
        """Control: a firing word stop predicts its first token."""
        prediction = servingbackends.predict_stop_outcome(
            decode=lambda ids: "hello STOP",
            returned_ids=(1,),
            stop_texts=("STOP",),
            eos_id=None,
        )
        self.assertEqual(prediction["predicted_firing_step"], 1)
