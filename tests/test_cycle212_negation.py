"""Cycle 212: antonym-negated definite-description probe
(``run --negation``).

Generated numform compare pairs rendered under direct, negated,
flipped, and descriptive negated_alt instruction arms. The
load-bearing pins:

* Frozen instruction strings byte-for-byte; the negated arm
  differs from direct in two lexical moves, so the claim is
  antonym-negated definite-description discordance, never
  negation simpliciter.
* The (8, 5) decision rule is a total function of D and J; no
  borrowed floor, no winner readout, no re-derivation.
* ``negation_echo`` is a frozen two-form set, a subclass of
  non-label: it never pools into ``wrong_label`` and never
  passes, so echoes cannot fire the joint rule.
* The stage fails closed (screen, budget, cancel, context,
  evaluation failure) and the analysis never invents a
  verdict word; model parse failure is scored wrong, never a
  detail.
* Token-ID divergence is reported, not gated: same-ids cases
  leave the headline population and count as a rendering
  anomaly; a same-ids verdict flip is a determinism finding.
"""

import re
import threading
import unittest
from unittest import mock

from model_protocol_lab import experiments
from model_protocol_lab import negation as ng
from model_protocol_lab import reporting as reporting_mod
from model_protocol_lab.backends import Generation
from model_protocol_lab.core import Format
from model_protocol_lab.core import evaluate as _real_evaluate  # noqa: F401 -- documents the scorer pin

_CHATML = Format(
    name="chatml",
    bos="",
    user_prefix="u:",
    assistant_prefix="a:",
    turn_end="</s>",
)


class _WordBackend:
    """Word-token scripted backend with switchable behaviours.

    Modes: ``perfect`` (answer the larger label on direct /
    negated / negated_alt, the smaller on flipped), ``negation_drop``
    (negated answers the smaller label), ``larger_default``
    (every arm answers the larger label), ``echo`` (negated and
    negated_alt answer "not " + the smaller label), ``explode``
    (generate raises). Arm detection is by instruction
    substring; operands are the digit tokens in prompt order.
    """

    fingerprint = "sha256:test-negation"

    def __init__(self, mode: str = "perfect") -> None:
        self.mode = mode
        self.calls = 0
        self.vocab: dict[str, int] = {}
        self.rev: dict[int, str] = {}

    def encode(self, text):
        ids = []
        for token in text.replace("\n", " \n ").split(" "):
            if not token:
                continue
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab) + 1
                self.rev[self.vocab[token]] = token
            ids.append(self.vocab[token])
        return tuple(ids)

    def decode(self, ids):
        return " ".join(self.rev.get(int(i), "?") for i in ids)

    def generate(self, input_ids, *, max_new_tokens, seed=None, cancel=None):
        self.calls += 1
        if self.mode == "explode":
            raise RuntimeError("boom")
        text = self.decode(list(input_ids))
        # Regex, not token parsing: the render glues the turn
        # end and assistant prefix onto the last operand
        # ("47</s>a:"), so the second operand is never a clean
        # digit token. The instruction carries no digits, so
        # the first two digit runs are the operands, in order.
        numbers = [int(tok) for tok in re.findall(r"\d+", text)]
        first, second = numbers[0], numbers[1]
        larger_label = "first" if first > second else "second"
        smaller_label = "second" if first > second else "first"
        if "not smaller than the other" in text:
            arm = "negated_alt"
        elif "not the smaller" in text:
            arm = "negated"
        elif "is smaller" in text:
            arm = "flipped"
        elif "is larger" in text:
            arm = "direct"
        else:
            arm = "unknown"
        if self.mode == "negation_drop" and arm == "negated":
            answer = smaller_label
        elif self.mode == "larger_default":
            answer = larger_label
        elif self.mode == "echo" and arm in ("negated", "negated_alt"):
            answer = "not " + smaller_label
        elif arm == "flipped":
            answer = smaller_label
        else:
            answer = larger_label
        boundary = self.encode("endmark")[0]
        out = list(self.encode(answer)) + [boundary]
        out = (out + [boundary] * max_new_tokens)[:max_new_tokens]
        return Generation(token_ids=tuple(out), reason="limit")


def _payload(backend=None, formats=(_CHATML,), seed=0, **stage_kwargs):
    backend = backend if backend is not None else _WordBackend()
    stage_kwargs.setdefault("max_context", 4096)
    stage = experiments.negation_stage(
        backend,
        formats,
        boundary=tuple(backend.encode("endmark")),
        seed=seed,
        **stage_kwargs,
    )
    return experiments.negation_analysis(stage, tuple(f.name for f in formats))


def _cell(verdict, answer, boundary=True):
    return {
        "verdict": verdict,
        "boundary_found": boundary,
        "boundary_text_match": None,
        "input_tokens": 3,
        "answer_digest": ng.digest_text(answer) if answer is not None else None,
    }


def _record(larger_first, stratum, cells, details=None, ids_identical=False, delta=2):
    return {
        "fixture": "nf-compare-00",
        "operands": [100, 47] if larger_first else [47, 100],
        "larger_first": larger_first,
        "stratum": stratum,
        "band": 0,
        "cells": cells,
        "cell_details": details or dict.fromkeys(ng.ARM_KEYS, ""),
        "ids_identical": ids_identical,
        "id_length_delta": delta,
    }


def _triple_record(larger_first, stratum, direct, negated, flipped, alt="first"):
    # direct/negated/flipped/alt are answer strings; the
    # verdict follows from the arm's expected label.
    larger = "first" if larger_first else "second"
    smaller = "second" if larger_first else "first"
    expected = {"direct": larger, "negated": larger, "flipped": smaller, "negated_alt": larger}
    answers = {"direct": direct, "negated": negated, "flipped": flipped, "negated_alt": alt}
    cells = {arm: _cell(answers[arm] == expected[arm], answers[arm]) for arm in ng.ARM_KEYS}
    return _record(larger_first, stratum, cells)


class FrozenMaterialTests(unittest.TestCase):
    def test_instructions_byte_exact(self):
        self.assertEqual(
            ng.INSTRUCTIONS["direct"],
            "Which of these two numbers is larger? Reply with only the word first or second.",
        )
        self.assertEqual(
            ng.INSTRUCTIONS["negated"],
            "Which of these two numbers is not the smaller? "
            "Reply with only the word first or second.",
        )
        self.assertEqual(
            ng.INSTRUCTIONS["flipped"],
            "Which of these two numbers is smaller? Reply with only the word first or second.",
        )
        self.assertEqual(
            ng.INSTRUCTIONS["negated_alt"],
            "Which of these two numbers is not smaller than the other? "
            "Reply with only the word first or second.",
        )

    def test_counts_and_floors(self):
        self.assertEqual(ng.CASE_COUNT, 16)
        self.assertEqual(ng.MAX_GROUND_RECORDS, 384)
        self.assertEqual(ng.HEADLINE_GENERATIONS, 1152)
        self.assertEqual(ng.GENERATION_BUDGET, 1536)
        self.assertEqual(ng.MIN_DIRECT_PASSES, 8)
        self.assertEqual(ng.MIN_JOINT, 5)
        self.assertEqual(ng.HEADLINE_ARMS, ("direct", "negated", "flipped"))
        self.assertEqual(ng.DESCRIPTIVE_ARM, "negated_alt")

    def test_expected_labels(self):
        self.assertEqual(ng.expected_label("direct", True), "first")
        self.assertEqual(ng.expected_label("negated", True), "first")
        self.assertEqual(ng.expected_label("negated_alt", True), "first")
        self.assertEqual(ng.expected_label("flipped", True), "second")
        self.assertEqual(ng.expected_label("direct", False), "second")
        self.assertEqual(ng.expected_label("flipped", False), "first")
        with self.assertRaises(ValueError):
            ng.expected_label("negated_flipped", True)

    def test_echo_set_exact_and_frozen(self):
        self.assertEqual(ng.echo_answers("first"), ("not second", "not the second"))
        self.assertEqual(ng.echo_answers("second"), ("not first", "not the first"))
        # Near-miss negated forms are non-label (other), never echo.
        for answer in ("not the second one", "not second.", "NOT second", "notsecond"):
            cell = _cell(False, answer)
            self.assertEqual(ng.decompose_cell(cell, "first"), "non_label", answer)


class ClassifyTests(unittest.TestCase):
    def test_decision_boundaries(self):
        self.assertEqual(ng.classify_negation(0, 0), "insufficient")
        self.assertEqual(ng.classify_negation(7, 7), "insufficient")
        self.assertEqual(ng.classify_negation(8, 0), "below_threshold")
        self.assertEqual(ng.classify_negation(8, 4), "below_threshold")
        self.assertEqual(ng.classify_negation(8, 5), "negation_drop_observed")
        self.assertEqual(ng.classify_negation(16, 5), "negation_drop_observed")
        self.assertEqual(ng.classify_negation(16, 16), "negation_drop_observed")


class DecomposeTests(unittest.TestCase):
    def test_all_classes(self):
        self.assertEqual(ng.decompose_cell(_cell(True, "first"), "first"), "exact_pass")
        self.assertEqual(ng.decompose_cell(_cell(False, "second"), "first"), "wrong_label")
        self.assertEqual(ng.decompose_cell(_cell(False, "not second"), "first"), "negation_echo")
        self.assertEqual(
            ng.decompose_cell(_cell(False, "not the second"), "first"), "negation_echo"
        )
        self.assertEqual(ng.decompose_cell(_cell(False, "not first"), "second"), "negation_echo")
        self.assertEqual(ng.decompose_cell(_cell(False, "gibberish"), "first"), "non_label")
        # Answer right, boundary missed: a boundary effect,
        # never a negation effect, never a J contributor.
        self.assertEqual(
            ng.decompose_cell(_cell(False, "first", boundary=False), "first"),
            "boundary_only",
        )
        with self.assertRaises(ValueError):
            ng.decompose_cell(None, "first")  # type: ignore[arg-type]


class CasesTests(unittest.TestCase):
    def test_seed_zero_matches_design_worked_examples(self):
        cases = ng.cases_for(0)
        self.assertEqual(len(cases), 16)
        first, second = cases[0], cases[1]
        self.assertEqual(first["fixture_id"], "nf-compare-00")
        self.assertEqual(first["operands"], [100, 47])
        self.assertTrue(first["larger_first"])
        self.assertEqual(first["expected"]["direct"], "first")
        self.assertEqual(first["expected"]["negated"], "first")
        self.assertEqual(first["expected"]["flipped"], "second")
        self.assertEqual(second["operands"], [199, 999])
        self.assertFalse(second["larger_first"])
        self.assertEqual(second["expected"]["direct"], "second")
        self.assertEqual(second["expected"]["flipped"], "first")

    def test_population_shape(self):
        cases = ng.cases_for(0)
        self.assertEqual(cases, ng.cases_for(0))
        for case in cases:
            self.assertNotEqual(case["operands"][0], case["operands"][1])
            self.assertIn(case["stratum"], ng.EMITTED_MARGIN_CLASSES)
        from collections import Counter

        self.assertEqual(
            Counter(c["stratum"] for c in cases),
            {"different_length": 8, "same_length_early": 4, "same_length_last_digit": 4},
        )
        self.assertEqual(Counter(c["larger_first"] for c in cases), {True: 8, False: 8})

    def test_ground_digest_binds_treatment(self):
        self.assertEqual(ng.ground_sha256(0), ng.ground_sha256(0))
        self.assertNotEqual(ng.ground_sha256(0), ng.ground_sha256(1))
        self.assertNotEqual(ng.case_material_sha256(), ng.ground_sha256(0))
        protocol = ng.protocol_object(0, "sha256:test")
        self.assertEqual(protocol["min_direct_passes"], 8)
        self.assertEqual(protocol["min_joint"], 5)
        self.assertEqual(protocol["headline_generations"], 1152)


class AggregateTests(unittest.TestCase):
    def _joint_records(self, n_direct_pass, n_joint, larger_first_flags):
        records = []
        joint_left = n_joint
        for index in range(16):
            larger_first = larger_first_flags[index % len(larger_first_flags)]
            larger = "first" if larger_first else "second"
            smaller = "second" if larger_first else "first"
            if index < n_direct_pass:
                negated = smaller if joint_left > 0 else larger
                if joint_left > 0:
                    joint_left -= 1
                records.append(
                    _triple_record(
                        larger_first,
                        "different_length",
                        larger,
                        negated,
                        smaller,
                    )
                )
            else:
                records.append(
                    _triple_record(
                        larger_first,
                        "different_length",
                        smaller,
                        larger,
                        smaller,
                    )
                )
        return records

    def test_negation_drop_observed(self):
        agg = ng.aggregate_candidate(self._joint_records(16, 16, [True, False]))
        headline = agg["headline"]
        self.assertEqual(headline["direct_passes"], 16)
        self.assertEqual(headline["joint"], 16)
        self.assertEqual(headline["decision"], "negation_drop_observed")
        self.assertEqual(headline["j_by_order"], {"larger_first": 8, "larger_second": 8})
        self.assertEqual(headline["j_by_stratum"], {"different_length": 16})
        self.assertFalse(headline["gate_weak"])
        self.assertFalse(headline["echo_dominant"])
        self.assertFalse(headline["order_concentrated"])

    def test_insufficient_withholds(self):
        agg = ng.aggregate_candidate(self._joint_records(7, 7, [True]))
        self.assertEqual(agg["headline"]["decision"], "insufficient")

    def test_below_threshold(self):
        agg = ng.aggregate_candidate(self._joint_records(10, 4, [True, False]))
        self.assertEqual(agg["headline"]["decision"], "below_threshold")

    def test_larger_default_is_gate_weak_not_acquittal(self):
        records = [
            _triple_record(True, "different_length", "first", "first", "first") for _ in range(16)
        ]
        agg = ng.aggregate_candidate(records)
        headline = agg["headline"]
        self.assertEqual(headline["direct_passes"], 16)
        self.assertEqual(headline["joint"], 0)
        self.assertEqual(headline["flipped_rate_on_d"], 0.0)
        self.assertTrue(headline["gate_weak"])
        self.assertEqual(headline["decision"], "below_threshold")

    def test_echo_dominant_not_drop(self):
        records = [
            _triple_record(True, "different_length", "first", "not second", "second")
            for _ in range(16)
        ]
        agg = ng.aggregate_candidate(records)
        headline = agg["headline"]
        self.assertEqual(headline["joint"], 0)
        self.assertTrue(headline["echo_dominant"])
        self.assertEqual(agg["arms"]["negated"]["decomposition"]["negation_echo"], 16)
        self.assertEqual(agg["arms"]["negated"]["decomposition"]["wrong_label"], 0)

    def test_order_concentrated_flag(self):
        records = self._joint_records(8, 8, [True])
        agg = ng.aggregate_candidate(records)
        self.assertTrue(agg["headline"]["order_concentrated"])
        self.assertEqual(agg["headline"]["j_by_order"]["larger_second"], 0)

    def test_same_ids_excluded_and_counted(self):
        records = self._joint_records(16, 16, [True, False])
        records.append(_triple_record(True, "different_length", "first", "second", "second"))
        records[-1]["ids_identical"] = True
        # The appended record would be a joint case if counted;
        # same-ids exclusion keeps J at 16 and counts the anomaly.
        # Its direct/negated verdicts differ, so it also counts
        # as a same-ids verdict flip (a determinism finding).
        agg = ng.aggregate_candidate(records)
        self.assertEqual(agg["headline"]["joint"], 16)
        self.assertEqual(agg["ids"]["n_ids_identical"], 1)
        self.assertEqual(agg["ids"]["n_same_ids_verdict_flips"], 1)
        # A same-ids verdict flip is a determinism finding.
        flipped = _triple_record(True, "different_length", "first", "first", "second")
        flipped["ids_identical"] = True
        flipped["cells"]["negated"] = _cell(False, "first")
        agg2 = ng.aggregate_candidate([flipped])
        self.assertEqual(agg2["ids"]["n_same_ids_verdict_flips"], 1)
        self.assertEqual(agg2["headline"]["direct_passes"], 0)

    def test_discordance_decomposition(self):
        direct_boundary = _cell(False, "first", boundary=False)
        negated_pass = _cell(True, "first")
        record = _record(
            True,
            "different_length",
            {
                "direct": direct_boundary,
                "negated": negated_pass,
                "flipped": _cell(True, "second"),
                "negated_alt": _cell(True, "first"),
            },
        )
        direct_pass = _cell(True, "first")
        negated_wrong = _cell(False, "second")
        record2 = _record(
            True,
            "different_length",
            {
                "direct": direct_pass,
                "negated": negated_wrong,
                "flipped": _cell(True, "second"),
                "negated_alt": _cell(True, "first"),
            },
        )
        agg = ng.aggregate_candidate([record, record2])
        self.assertEqual(agg["discordance"]["boundary_only"], 1)
        self.assertEqual(agg["discordance"]["exact_only"], 1)
        self.assertEqual(agg["discordance"]["mixed"], 0)

    def test_descriptive_alt_never_decides(self):
        records = self._joint_records(16, 0, [True, False])
        for record in records:
            smaller = "second" if record["larger_first"] else "first"
            record["cells"]["negated_alt"] = _cell(False, smaller)
        agg = ng.aggregate_candidate(records)
        self.assertEqual(agg["headline"]["joint"], 0)
        self.assertEqual(agg["descriptive"]["negated_alt_wrong_label_on_d"], 16)
        self.assertEqual(agg["descriptive"]["negated_alt_joint"], 16)

    def test_attrition_and_incomplete(self):
        record = _record(
            True,
            "different_length",
            {
                "direct": _cell(True, "first"),
                "negated": None,
                "flipped": _cell(True, "second"),
                "negated_alt": _cell(True, "first"),
            },
            details={
                "direct": "",
                "negated": "arm_context_exceeded",
                "flipped": "",
                "negated_alt": "",
            },
        )
        agg = ng.aggregate_candidate([record])
        self.assertEqual(agg["n_incomplete"], 1)
        self.assertEqual(agg["arms"]["negated"]["n_context_exceeded"], 1)
        self.assertEqual(agg["arms"]["negated"]["n_completed"], 0)
        self.assertIsNone(agg["arms"]["negated"]["rate"])


class StageTests(unittest.TestCase):
    def test_perfect_backend_below_threshold(self):
        payload = _payload()
        self.assertTrue(payload["available"])
        self.assertEqual(payload["n_cases"], 16)
        entry = payload["candidates"]["chatml"]
        self.assertEqual(entry["headline"]["direct_passes"], 16)
        self.assertEqual(entry["headline"]["joint"], 0)
        self.assertEqual(entry["headline"]["decision"], "below_threshold")
        self.assertEqual(entry["arms"]["direct"]["rate"], 1.0)
        self.assertEqual(len(entry["records"]), 16)
        self.assertTrue(all(d > 0 for d in entry["ids"]["id_length_delta_by_case"]))
        self.assertEqual(entry["ids"]["n_ids_identical"], 0)
        # Content digests bind the frozen instructions.
        record = entry["records"][0]
        self.assertEqual(
            record["content_digests"]["negated"],
            ng.digest_text(ng.arm_content("negated", 100, 47)),
        )

    def test_negation_drop_backend_observed(self):
        payload = _payload(_WordBackend("negation_drop"))
        entry = payload["candidates"]["chatml"]
        self.assertEqual(entry["headline"]["joint"], 16)
        self.assertEqual(entry["headline"]["decision"], "negation_drop_observed")

    def test_larger_default_backend_gate_weak(self):
        payload = _payload(_WordBackend("larger_default"))
        entry = payload["candidates"]["chatml"]
        self.assertTrue(entry["headline"]["gate_weak"])
        self.assertEqual(entry["headline"]["decision"], "below_threshold")

    def test_echo_backend_echo_dominant(self):
        payload = _payload(_WordBackend("echo"))
        entry = payload["candidates"]["chatml"]
        self.assertTrue(entry["headline"]["echo_dominant"])
        self.assertEqual(entry["headline"]["joint"], 0)

    def test_explode_backend_scores_wrong_never_a_detail(self):
        payload = _payload(_WordBackend("explode"))
        entry = payload["candidates"]["chatml"]
        self.assertEqual(entry["arms"]["direct"]["n_evaluation_failed"], 16)
        self.assertEqual(entry["arms"]["direct"]["n_completed"], 0)
        self.assertEqual(entry["headline"]["decision"], "insufficient")
        details = {d for r in entry["records"] for d in r["cell_details"].values()}
        self.assertEqual(details, {"arm_evaluation_failed"})

    def test_cancel_unwinds_to_fail_closed_payload(self):
        event = threading.Event()
        event.set()
        backend = _WordBackend()
        stage = experiments.negation_stage(
            backend,
            (_CHATML,),
            boundary=tuple(backend.encode("endmark")),
            max_context=4096,
            seed=0,
            cancel=event,
        )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "cancelled")
        analysis = experiments.negation_analysis(stage, ("chatml",))
        self.assertFalse(analysis["available"])
        self.assertEqual(analysis["detail"], "cancelled")

    def test_duplicate_format_names_rejected(self):
        backend = _WordBackend()
        with self.assertRaises(ValueError):
            experiments.negation_stage(
                backend,
                (_CHATML, _CHATML),
                boundary=tuple(backend.encode("endmark")),
                max_context=4096,
                seed=0,
            )

    def test_record_budget_exceeded(self):
        backend = _WordBackend()
        formats = tuple(
            Format(
                name=f"ng{i}",
                bos="",
                user_prefix=f"u{i}:",
                assistant_prefix="a:",
                turn_end="</s>",
            )
            for i in range(25)
        )
        stage = experiments.negation_stage(
            backend,
            formats,
            boundary=tuple(backend.encode("endmark")),
            max_context=4096,
            seed=0,
        )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "negation_record_budget_exceeded")

    def test_special_token_screen_wiring_proven_by_stub(self):
        from model_protocol_lab import fixtures as fixtures_mod
        from model_protocol_lab.fixtures import AuditFinding

        backend = _WordBackend()
        finding = AuditFinding(
            fixture_id="generated",
            field="text",
            literal="<|im_start|>",
            format_name="screen",
            offset=0,
        )
        with mock.patch.object(
            fixtures_mod,
            "audit_special_token_literals",
            return_value=(finding,),
        ):
            stage = experiments.negation_stage(
                backend,
                (_CHATML,),
                boundary=tuple(backend.encode("endmark")),
                max_context=4096,
                seed=0,
            )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "negation_screen_failed")

    def test_boundary_text_screen(self):
        backend = _WordBackend()
        stage = experiments.negation_stage(
            backend,
            (_CHATML,),
            boundary=tuple(backend.encode("larger")),
            max_context=4096,
            seed=0,
        )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], "negation_screen_failed")

    def test_context_exceeded_cells(self):
        payload = _payload(max_context=1, max_new_tokens=64)
        entry = payload["candidates"]["chatml"]
        self.assertEqual(entry["arms"]["direct"]["n_context_exceeded"], 16)
        self.assertEqual(entry["headline"]["decision"], "insufficient")

    def test_unavailable_analysis_passthrough(self):
        stage = {
            "available": False,
            "detail": "negation_screen_failed",
            "protocol": {},
            "audit": {},
        }
        analysis = experiments.negation_analysis(stage, ("chatml",))
        self.assertFalse(analysis["available"])
        self.assertEqual(analysis["detail"], "negation_screen_failed")


class ReportingTests(unittest.TestCase):
    def test_section_renders_decision_and_bundle(self):
        payload = _payload(_WordBackend("negation_drop"))
        html = reporting_mod._negation_section({"negation": payload})
        self.assertIn('id="negation"', html)
        self.assertIn("negation_drop_observed", html)
        self.assertIn("antonym-negated definite-description discordance", html)
        self.assertIn("two lexical moves", html)
        self.assertIn("does not establish invariance", html)
        self.assertNotIn("winner", html)
        self.assertNotIn("negation-invariance", html)
        self.assertNotIn("negation sensitivity", html)

    def test_section_echo_share_and_gate_weak(self):
        echo_html = reporting_mod._negation_section({"negation": _payload(_WordBackend("echo"))})
        self.assertIn("Echo-dominant", echo_html)
        self.assertIn("Echo share among non-pass", echo_html)
        weak_html = reporting_mod._negation_section(
            {"negation": _payload(_WordBackend("larger_default"))}
        )
        self.assertIn("Gate-weak", weak_html)
        self.assertIn("negation-drop undiagnosed, not absent", weak_html)

    def test_section_in_render_html_toc_and_body(self):
        payload = _payload(_WordBackend("negation_drop"))
        html = reporting_mod.render_html(
            {"decision": "compatible_on_fixture_set", "negation": payload}
        )
        self.assertIn('href="#negation"', html)
        self.assertIn('id="negation"', html)

    def test_section_empty_when_unavailable(self):
        self.assertEqual(reporting_mod._negation_section({}), "")
        self.assertEqual(
            reporting_mod._negation_section(
                {"negation": {"available": False, "detail": "negation_unavailable"}}
            ),
            "",
        )


class CliTests(unittest.TestCase):
    def test_negation_flag_parses(self):
        from model_protocol_lab.cli import build_parser

        # The run subcommand's required args (cycle 179's
        # _REQUIRED); the flag itself must set its dest.
        required = [
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
        parser = build_parser()
        args = parser.parse_args(["run", "--negation", *required])
        self.assertTrue(args.negation)
        args = parser.parse_args(["run", *required])
        self.assertFalse(args.negation)

    def test_probe_wrapper_never_raises(self):
        from model_protocol_lab import cli as cli_module

        class _EncodeExplodeBackend(_WordBackend):
            def encode(self, text):
                raise RuntimeError("no encode")

            def decode(self, ids):
                # A decode of "?" would trip the boundary-text
                # screen (every instruction ends in "?"); the
                # contract here is a terminator outside the
                # audit text, so the encode failure is what the
                # wrapper must survive.
                return "ENDMARK"

        result: dict = {}
        cli_module._run_negation_probe(
            _EncodeExplodeBackend(),
            (_CHATML,),
            result,
            boundary=(1,),
            boundary_map=None,
            max_new_tokens=64,
            max_context=4096,
            seed=0,
            cancel=None,
        )
        self.assertEqual(
            result["negation"],
            {"available": False, "detail": "negation_unavailable"},
        )


if __name__ == "__main__":
    unittest.main()
