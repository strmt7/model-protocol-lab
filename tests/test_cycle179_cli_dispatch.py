"""Parametrized CLI dispatch e2e for the run-probe flags (Cycle 179).

Before this cycle, each probe's CLI wiring was tested only by its own
cycle's tests, in whatever shape that cycle chose, and five flags
(--transfer-model, --perturb, --token-healing, --logprob-diagnostic,
--sensitivity-sweep) dispatch inline in the run flow with no per-flag
dispatch test at all. A renamed dest, a
dropped `if args.<flag>` branch, or a helper that stops attaching its
result key would have failed no test for most flags.

This file pins the dispatch layer for every stage flag, following
the cycle-174 pattern (flag parses -> probe attaches its profile ->
probe never raises):

- TestFlagParsing: every stage flag in STAGE_FLAGS sets its
  argparse dest when present and leaves it falsy when absent.
- TestHelperDispatch: the `_run_*_probe` helpers in cli.py (30
  after cycle 195 added choice-audit). With
  the helper's experiments functions stubbed to a sentinel, the
  helper attaches a non-marker payload under its result key; with
  the first experiments function stubbed to raise, the helper
  attaches exactly {"available": False, "detail": "<key>_unavailable"}
  and does not raise. The contrast between the two halves is the
  dispatch proof: the success payload can only appear if the flag's
  helper really called through to the stage/profile layer.
- TestInlineDispatch: the five inline flags driven through the real
  run flow (cycle-31 harness pattern: fake backend, fake
  run_experiment, stubbed stage functions), asserting the stage key
  lands on the result document.

The same file adds the tempdir mutation gates for ablation.py
(cycle-169 gate pattern), the last pure module whose exactness-
critical lines had no gate -- VALIDATION's "mutation gates for
every module" claim now holds for it.
"""

import contextlib
import inspect
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from model_protocol_lab import backends, experiments
from model_protocol_lab import cli as cli_module
from model_protocol_lab import doctor as doctor_mod
from model_protocol_lab import reporting as reporting_mod
from model_protocol_lab.core import Fixture, Format, Message

REPO_ROOT = Path(__file__).resolve().parent.parent

_FORMAT = Format(
    name="chatml",
    bos="",
    user_prefix="u:",
    assistant_prefix="a:",
    turn_end="</s>",
)
_FIXTURE = Fixture(
    id="f1",
    messages=(Message(role="user", content="hi"),),
    expected="yo",
    group="g1",
)
_ENTRY = {
    "candidate": "chatml",
    "name": "chatml",
    "observations": [
        {
            "fixture_id": "f1",
            "answer": "I'm sorry, no.",
            "exact_match": False,
            "boundary_found": True,
            "input_ids": [1, 2, 3],
            "output_ids": [4],
            "status": "completed",
        }
    ],
}

# (flag, dest, value-or-None) for every stage flag on `run`.
STAGE_FLAGS = [
    ("--transfer-model", "transfer_model", "ckpt2"),
    ("--perturb", "perturb", "2"),
    ("--token-healing", "token_healing", None),
    ("--hierarchy", "hierarchy", None),
    ("--prefill", "prefill", None),
    ("--synth-history", "synth_history", None),
    ("--few-shot", "few_shot", None),
    ("--batch-invariance", "batch_invariance", None),
    ("--position", "position", None),
    ("--stop-cause", "stop_cause", None),
    ("--ws-scan", "ws_scan", None),
    ("--structured", "structured", None),
    ("--deliforge", "deliforge", None),
    ("--fake-turn", "fake_turn", None),
    ("--tool-call", "tool_call", None),
    ("--verbosity", "verbosity", None),
    ("--refusals", "refusals", None),
    ("--scorer-lens", "scorer_lens", None),
    ("--aliasing", "aliasing", None),
    ("--repeat", "repeat", None),
    ("--distract", "distract", None),
    ("--leading", "leading", None),
    ("--numform", "numform", None),
    ("--dateline", "dateline", None),
    ("--multiturn", "multiturn", None),
    ("--needle", "needle", None),
    ("--placement", "placement", None),
    ("--rotate", "rotate", None),
    ("--choice-audit", "choice_audit", None),
    ("--precision", "precision", None),
    ("--confidence", "confidence", None),
    ("--cache-profile", "cache_profile", None),
    ("--confidentiality", "confidentiality", None),
    ("--answer-collision", "answer_collision", None),
    ("--template-robustness", "template_robustness", None),
    ("--ablate-fields", "ablate_fields", None),
    ("--logprob-diagnostic", "logprob_diagnostic", None),
    ("--sensitivity-sweep", "sensitivity_sweep", None),
]

_REQUIRED = [
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


class _Sentinel(dict):
    """Stage/profile stand-in: any key lookup yields the sentinel."""

    def __missing__(self, key):  # noqa: ANN001, ANN202
        return self


class _FakeBackend:
    """Scripted backend for helper dispatch: encodes anything."""

    def encode(self, text):  # noqa: ANN001, ANN202
        return (7, 8)


def _call_helper(helper, result):  # noqa: ANN001, ANN202
    """Call a probe helper with the shared input pool, filtered to
    the parameters its signature actually declares."""
    pool = {
        "backend": _FakeBackend(),
        "formats": (_FORMAT,),
        "result": result,
        "cases": (_FIXTURE,),
        "demo_cases": (_FIXTURE,),
        "eval_cases": (_FIXTURE,),
        "search_fixtures": (_FIXTURE,),
        "boundary": (198,),
        "boundary_map": None,
        "bf16_backend": _FakeBackend(),
        "eos_id": 2,
        "token_bytes": None,
        "max_new_tokens": 8,
        "max_context": 4096,
        "seed": 0,
        "block_size": 16,
        "cancel": None,
    }
    params = inspect.signature(helper).parameters
    kwargs = {name: pool[name] for name in params if name in pool}
    return helper(**kwargs)


# (helper attribute, result key, unavailable detail, experiments names
# the helper imports -- stubbed for both dispatch halves)
HELPER_DISPATCH = [
    (
        "_run_hierarchy_probe",
        "hierarchy",
        "hierarchy_unavailable",
        ["hierarchy_stage", "hierarchy_adherence"],
    ),
    (
        "_run_position_probe",
        "position",
        "position_unavailable",
        ["position_stage", "position_sensitivity"],
    ),
    ("_run_stop_probe", "stop_cause", "stop_unavailable", ["stop_stage"]),
    (
        "_run_ws_probe",
        "ws_canonical",
        "ws_scan_unavailable",
        ["ws_stage", "ws_canonicalization", "native_search_input_ids", "native_search_outcomes"],
    ),
    (
        "_run_structured_probe",
        "structured",
        "structured_unavailable",
        ["structured_stage", "structured_adherence"],
    ),
    (
        "_run_deliforge_probe",
        "deliforge",
        "deliforge_unavailable",
        ["deliforge_stage", "deliforge_authority", "native_search_outcomes"],
    ),
    (
        "_run_faketurn_probe",
        "faketurn",
        "faketurn_unavailable",
        ["faketurn_stage", "faketurn_robustness", "native_search_outcomes"],
    ),
    (
        "_run_toolcall_probe",
        "toolcall",
        "toolcall_unavailable",
        ["toolcall_stage", "toolcall_adherence"],
    ),
    (
        "_run_verbosity_probe",
        "verbosity",
        "verbosity_unavailable",
        ["verbosity_stage", "verbosity_profile"],
    ),
    ("_run_refusals_probe", "refusals", "refusals_unavailable", ["refusal_profile"]),
    (
        "_run_scorerlens_probe",
        "scorerlens",
        "scorerlens_unavailable",
        ["scorerlens_profile"],
    ),
    (
        "_run_aliasing_probe",
        "aliasing",
        "aliasing_unavailable",
        ["aliasing_profile"],
    ),
    (
        "_run_repeat_probe",
        "repetition",
        "repetition_unavailable",
        ["repetition_stage", "repetition_analysis"],
    ),
    (
        "_run_distract_probe",
        "distraction",
        "distraction_unavailable",
        ["distraction_stage", "distraction_analysis"],
    ),
    (
        "_run_leading_probe",
        "leading",
        "leading_unavailable",
        ["leading_stage", "leading_analysis"],
    ),
    (
        "_run_numform_probe",
        "numform",
        "numform_unavailable",
        ["numform_stage", "numform_analysis"],
    ),
    (
        "_run_dateline_probe",
        "dateline",
        "dateline_unavailable",
        ["dateline_stage", "dateline_analysis"],
    ),
    (
        "_run_multiturn_probe",
        "multiturn",
        "multiturn_unavailable",
        ["multiturn_stage", "multiturn_analysis"],
    ),
    (
        "_run_needle_probe",
        "needle",
        "needle_unavailable",
        ["needle_stage", "needle_analysis"],
    ),
    (
        "_run_placement_probe",
        "placement",
        "placement_unavailable",
        ["placement_stage", "placement_analysis"],
    ),
    (
        "_run_rotation_probe",
        "rotation",
        "rotation_unavailable",
        ["rotation_stage", "rotation_analysis"],
    ),
    (
        "_run_choiceaudit_probe",
        "choice_audit",
        "choiceaudit_unavailable",
        ["choiceaudit_stage", "choiceaudit_analysis"],
    ),
    (
        "_run_precision_probe",
        "precision",
        "precision_unavailable",
        ["precision_stage"],
    ),
    (
        "_run_confidence_probe",
        "confidence",
        "confidence_unavailable",
        ["confidence_stage"],
    ),
    (
        "_run_cache_profile_probe",
        "cache_profile",
        "cache_profile_unavailable",
        ["cache_profile_stage", "cache_profile_profile"],
    ),
    (
        "_run_confidentiality_probe",
        "confidentiality",
        "confidentiality_unavailable",
        ["confidentiality_stage", "confidentiality_profile"],
    ),
    (
        "_run_answer_collision_probe",
        "answer_collision",
        "answer_collision_unavailable",
        ["answer_collision_stage", "answer_collision_profile"],
    ),
    (
        "_run_template_robustness_probe",
        "template_robustness",
        "template_robustness_unavailable",
        ["template_robustness_stage"],
    ),
    ("_run_ablation_probe", "ablation", "ablation_unavailable", ["ablation_stage"]),
    (
        "_run_prefill_probe",
        "prefill",
        "prefill_unavailable",
        ["prefill_stage", "prefill_adherence"],
    ),
    (
        "_run_synthhistory_probe",
        "synthhistory",
        "synthhistory_unavailable",
        ["synthhistory_stage", "synthhistory_stability"],
    ),
    ("_run_fewshot_probe", "fewshot", "fewshot_unavailable", ["fewshot_stage", "fewshot_gain"]),
    ("_run_batch_probe", "batch_invariance", "batch_unavailable", ["batch_stage"]),
]


class TestFlagParsing(unittest.TestCase):
    def test_every_stage_flag_sets_its_dest(self):
        parser = cli_module.build_parser()
        for flag, dest, value in STAGE_FLAGS:
            with self.subTest(flag=flag):
                argv = ["run", flag, *([value] if value else []), *_REQUIRED]
                args = parser.parse_args(argv)
                self.assertTrue(getattr(args, dest), f"{flag} did not set {dest}")
                args = parser.parse_args(["run", *_REQUIRED])
                self.assertFalse(getattr(args, dest), f"{dest} truthy without {flag}")


class TestHelperDispatch(unittest.TestCase):
    def test_helper_attaches_payload(self):
        for helper_name, key, detail, exp_names in HELPER_DISPATCH:
            with self.subTest(helper=helper_name):
                helper = getattr(cli_module, helper_name)
                sentinel = _Sentinel()
                with contextlib.ExitStack() as stack:
                    for name in exp_names:
                        stack.enter_context(
                            mock.patch.object(experiments, name, return_value=sentinel)
                        )
                    result: dict = {
                        "search": [dict(_ENTRY)],
                        "decision": "no_candidate",
                        "selected_candidate": None,
                    }
                    _call_helper(helper, result)
                self.assertIn(key, result)
                self.assertNotEqual(result[key], {"available": False, "detail": detail})

    def test_helper_never_raises(self):
        for helper_name, key, detail, exp_names in HELPER_DISPATCH:
            with self.subTest(helper=helper_name):
                helper = getattr(cli_module, helper_name)
                with mock.patch.object(
                    experiments,
                    exp_names[0],
                    side_effect=RuntimeError("boom"),
                ):
                    result: dict = {
                        "search": [dict(_ENTRY)],
                        "decision": "no_candidate",
                        "selected_candidate": None,
                    }
                    _call_helper(helper, result)
                self.assertEqual(result[key], {"available": False, "detail": detail})


class _FakeHFBackend:
    """Stand-in for LocalHFBackend in the run flow: no weights."""

    def __init__(self, model, device=None, max_context=None, dtype=None):
        self.model = model
        self.device = device
        self.max_context = max_context
        self.dtype = dtype

    eos_token_id = None


def _run_cli(argv):
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli_module.main(argv)
    except SystemExit as exc:
        return ("exit", exc.code, out.getvalue(), err.getvalue())
    return ("return", code, out.getvalue(), err.getvalue())


def _write_fixture_file(directory):
    path = Path(directory) / "fixture.json"
    path.write_text(
        json.dumps(
            {
                "search": [
                    {
                        "id": "s1",
                        "group": "g1",
                        "messages": [{"role": "user", "content": "hi"}],
                        "expected": "yo",
                    }
                ],
                "held_out": [
                    {
                        "id": "h1",
                        "group": "g2",
                        "messages": [{"role": "user", "content": "hey"}],
                        "expected": "yo",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


class TestInlineDispatch(unittest.TestCase):
    """The five inline flags, driven through the real run flow with
    the stage layer stubbed (cycle-31 harness pattern)."""

    def _run_with(self, argv_extra, exp_patches):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _write_fixture_file(tmp)
            outdir = Path(tmp) / "report"
            result_doc = {
                "decision": "compatible_on_fixture_set",
                "selected_candidate": None,
                "search": [],
                "held_out": {},
            }
            sentinel = _Sentinel()
            with contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch.object(backends, "LocalHFBackend", _FakeHFBackend))
                stack.enter_context(
                    mock.patch.object(cli_module, "run_experiment", return_value=result_doc)
                )
                stack.enter_context(
                    mock.patch.object(
                        cli_module,
                        "save_report",
                        return_value=(
                            str(outdir / "results.json"),
                            str(outdir / "report.md"),
                        ),
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        doctor_mod,
                        "template_conformance",
                        return_value={"ok": True},
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        experiments,
                        "budget_frontier",
                        return_value={"available": False},
                    )
                )
                for name in exp_patches:
                    stack.enter_context(mock.patch.object(experiments, name, return_value=sentinel))

                def fake_prepare(path):
                    Path(path).mkdir(parents=True, exist_ok=False)
                    return Path(path)

                stack.enter_context(
                    mock.patch.object(reporting_mod, "prepare_report_dir", fake_prepare)
                )
                argv = [
                    "run",
                    "--model",
                    "ckpt-dir",
                    "--dataset",
                    str(fixture),
                    "--family",
                    "chatml",
                    "--boundary-ids",
                    "198,199",
                    "--output",
                    str(outdir),
                    "--seed",
                    "7",
                    *argv_extra,
                ]
                outcome = _run_cli(argv)
            return outcome, result_doc, sentinel

    def test_perturb_dispatches(self):
        (kind, code, _out, _err), result_doc, sentinel = self._run_with(
            ["--perturb", "2"],
            ["perturbation_stage", "perturbation_robustness"],
        )
        self.assertEqual((kind, code), ("return", 0))
        self.assertIs(result_doc["perturbation"], sentinel)

    def test_token_healing_dispatches(self):
        (kind, code, _out, _err), result_doc, sentinel = self._run_with(
            ["--token-healing"],
            ["healing_stage", "native_search_outcomes", "healing_delta"],
        )
        self.assertEqual((kind, code), ("return", 0))
        self.assertIs(result_doc["token_healing"], sentinel)

    def test_transfer_dispatches(self):
        (kind, code, _out, _err), result_doc, sentinel = self._run_with(
            ["--transfer-model", "ckpt2"],
            ["transfer_stage", "transfer_portability"],
        )
        self.assertEqual((kind, code), ("return", 0))
        self.assertIs(result_doc["transfer"], sentinel)

    def test_logprob_dispatches(self):
        (kind, code, _out, _err), result_doc, sentinel = self._run_with(
            ["--logprob-diagnostic"],
            ["logprob_diagnostic"],
        )
        self.assertEqual((kind, code), ("return", 0))
        self.assertIs(result_doc["logprob_diagnostic"], sentinel)

    def test_sensitivity_sweep_dispatches(self):
        (kind, code, _out, _err), result_doc, sentinel = self._run_with(
            ["--sensitivity-sweep"],
            ["prompt_sensitivity_sweep"],
        )
        self.assertEqual((kind, code), ("return", 0))
        self.assertIs(result_doc["sensitivity_sweep"], sentinel)


class AblationMutationGateTests(unittest.TestCase):
    """Tempdir mutation gates on ablation.py's exactness-critical
    lines (cycle-169 pattern: the scenario passes on the real module
    and must raise AssertionError on the mutant)."""

    def _run_gate(self, name, old_text, new_text, scenario):
        import importlib.util

        mod = __import__("model_protocol_lab.ablation", fromlist=["x"])
        source = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertEqual(source.count(old_text), 1, f"mutant {name}: anchor not unique")
        mutated = source.replace(old_text, new_text)
        with tempfile.TemporaryDirectory(prefix="c179_mut_") as tmpdir:
            mod_path = os.path.join(tmpdir, f"ablation_mut_{name}.py")
            with open(mod_path, "w", encoding="utf-8") as handle:
                handle.write(mutated)
            dotted = f"model_protocol_lab.ablation_mut_{name}"
            spec = importlib.util.spec_from_file_location(dotted, mod_path)
            assert spec is not None and spec.loader is not None
            mutant = importlib.util.module_from_spec(spec)
            # ablation.py defines a dataclass at module level; the
            # dataclass machinery resolves the defining module through
            # sys.modules, so the mutant must be registered first.
            sys.modules[dotted] = mutant
            try:
                spec.loader.exec_module(mutant)
            finally:
                sys.modules.pop(dotted, None)
            scenario(mod)  # passes on the real module: non-vacuous
            with self.assertRaises(AssertionError, msg=f"mutant {name} was NOT caught"):
                scenario(mutant)

    def _swap_scenario(self, mod):
        base = Format(
            name="x",
            bos="<s>",
            user_prefix="u:",
            assistant_prefix="a:",
            turn_end="</s>",
        )
        ablated = mod.ablate_format(base, "bos")
        assert ablated.bos == ""
        assert ablated.user_prefix == "u:"  # only the named field moves

    def test_mutant_swap_dropped_caught(self):
        self._run_gate(
            "swap_dropped",
            "return dataclasses.replace(base, **{field: neutral})",
            "return base  # MUTANT",
            self._swap_scenario,
        )

    def _noop_arm_scenario(self, mod):
        base = Format(
            name="x",
            bos="",
            user_prefix="u:",
            assistant_prefix="a:",
            turn_end="</s>",
        )
        fields = [arm.field for arm in mod.ablation_arms(base)]
        assert "bos" not in fields  # already neutral: no arm
        assert "user_prefix" in fields

    def test_mutant_noop_arm_included_caught(self):
        self._run_gate(
            "noop_arm",
            "if original != NEUTRAL_VALUES[field]:",
            "if True:  # MUTANT",
            self._noop_arm_scenario,
        )

    def _flip_scenario(self, mod):
        summary = mod.summarize_arm((True,), (False,), ("f1",))
        assert summary["pass_to_fail"] == 1
        assert summary["fail_to_pass"] == 0
        assert summary["flipped_fixture_ids"] == ("f1",)

    def test_mutant_flip_direction_swapped_caught(self):
        self._run_gate(
            "flip_swapped",
            '"pass_to_fail": pass_to_fail,',
            '"pass_to_fail": fail_to_pass,  # MUTANT',
            self._flip_scenario,
        )

    def _bar_scenario(self, mod):
        summary = mod.summarize_arm((True, True), (True, False), ("f1", "f2"))
        assert summary["selection_bar_held"] is False

    def test_mutant_selection_bar_weakened_caught(self):
        self._run_gate(
            "bar_any",
            '"selection_bar_held": all(arm_verdicts),',
            '"selection_bar_held": any(arm_verdicts),  # MUTANT',
            self._bar_scenario,
        )

    def _inapplicable_scenario(self, mod):
        base = Format(
            name="x",
            bos="",
            user_prefix="u:",
            assistant_prefix="a:",
            turn_end="</s>",
        )
        assert "bos" in mod.inapplicable_fields(base)
        assert "user_prefix" not in mod.inapplicable_fields(base)

    def test_mutant_inapplicable_inverted_caught(self):
        self._run_gate(
            "inapplicable_inverted",
            "getattr(base, field) == NEUTRAL_VALUES[field]",
            "getattr(base, field) != NEUTRAL_VALUES[field]  # MUTANT",
            self._inapplicable_scenario,
        )


if __name__ == "__main__":
    unittest.main()
