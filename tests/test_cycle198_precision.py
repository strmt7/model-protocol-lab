"""Cycle 198: load precision (`run --precision`, fp32 vs bf16).

The primary measurement always loads in float32. The flag adds a
second load of the same checkpoint in bfloat16 and re-runs the
search fixtures through the unchanged evaluation pipeline,
comparing the two arms fixture by fixture: pass/fail flips,
output divergence, teacher-forced expected-answer logprob
deltas, and whether the primary selection (decision, winner)
survives the precision change on the shared paired population.
This file pins the backend dtype contract, the stage's pairing
and withholding rules, the diff validator's full-recompute
binding, the HTML section, and the CLI wrapper's fail-closed
behaviour.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from threading import Event

from model_protocol_lab import cli, diffreports, experiments, precision, reporting
from model_protocol_lab.core import Fixture, Format, Generation, Message

_FORMATS = (
    Format(name="alpha", bos="", user_prefix="u:", assistant_prefix="a:", turn_end="</s>"),
    Format(name="beta", bos="", user_prefix="U>", assistant_prefix="A>", turn_end="<e>"),
)


def _fx(fid, expected, group):
    """Fx."""
    return Fixture(
        id=fid,
        messages=(Message(role="user", content=f"question {fid}?"),),
        expected=expected,
        group=group,
    )


def _cases():
    """Cases."""
    return (
        _fx("s0", "42", "numbers"),
        _fx("s1", "Paris", "geo"),
        _fx("s2", "blue", "colors"),
        _fx("s3", "17", "numbers"),
    )


class _ScriptedBackend:
    """Answers per (format name, fixture id); the rendered prompt
    carries the format prefix and the fixture question, so the
    fake keys on (prefix, question). Emits the answer plus the
    chr(7) boundary token, padded to the generation cap (the
    ``limit`` reason contract). Echo forward logits make
    teacher-forced logprobs real floats through the real
    scoring path.
    """

    def __init__(self, table, *, dtype="float32", fingerprint="sha256:native-fp"):
        self.table = table
        self.dtype = dtype
        self.fingerprint = fingerprint
        self.declared_dtype = "float32"

    @staticmethod
    def encode(text):
        """Encode."""
        return tuple(text.encode("utf-8"))

    @staticmethod
    def decode(ids):
        """Decode."""
        return bytes(int(i) % 256 for i in ids).decode("utf-8", "replace")

    def generate(self, input_ids, *, max_new_tokens, seed, cancel):
        """Generate."""
        prompt = self.decode(input_ids)
        fmt = "alpha" if prompt.startswith("u:") else "beta"
        answer = ""
        for (name, fid), text in self.table.items():
            if name == fmt and f"question {fid}?" in prompt:
                answer = text
                break
        ids = tuple((answer + "\x07").encode("utf-8"))
        ids = ids + (0,) * max(0, max_new_tokens - len(ids))
        return Generation(token_ids=ids, reason="limit")

    @staticmethod
    def forward_logits(input_ids):
        """Forward logits."""
        import torch

        seq = list(input_ids)
        logits = torch.zeros((len(seq), 256))
        for pos in range(len(seq) - 1):
            logits[pos][seq[pos + 1]] = 6.0
        return logits


def _full_table(fail=()):
    """Full table."""
    # Base shape: alpha passes everything, beta fails s3 -- the
    # native run's unique winner is alpha (a both-pass table
    # would make the primary decision ambiguous, and the
    # survival readout would be withheld before it starts).
    # ``fail`` adds further failing cells on top of the base.
    fails = {("beta", "s3")} | set(fail)
    table = {}
    for fmt in ("alpha", "beta"):
        for case in _cases():
            table[(fmt, case.id)] = "wrong" if (fmt, case.id) in fails else case.expected
        table[(fmt, "h0")] = "held"
    return table


def _native_result(native, cases):
    """Native result."""
    return experiments.run_experiment(
        native,
        _FORMATS,
        experiments.Split(search=cases, held_out=(_fx("h0", "held", "heldout"),)),
        boundary=(7,),
        max_new_tokens=64,
        seed=0,
    )


def _run_stage(native=None, bf16=None, *, cases=None, result=None):
    """Run stage."""
    cases = _cases() if cases is None else cases
    native = _ScriptedBackend(_full_table()) if native is None else native
    bf16 = (
        _ScriptedBackend(_full_table(), dtype="bfloat16", fingerprint="sha256:bf16-fp")
        if bf16 is None
        else bf16
    )
    result = _native_result(native, cases) if result is None else result
    payload = experiments.precision_stage(
        native, bf16, _FORMATS, cases, result, boundary=(7,), max_new_tokens=64, seed=0
    )
    return result, payload


def _valid_payload():
    """Valid payload."""
    _result, payload = _run_stage(
        bf16=_ScriptedBackend(
            _full_table(fail={("alpha", "s2")}), dtype="bfloat16", fingerprint="sha256:bf16-fp"
        )
    )
    assert payload["available"] is True, payload
    return json.loads(json.dumps(payload))


class TestPrecisionStage(unittest.TestCase):
    """Test precision stage."""

    def test_identical_arms_survive_and_falsify(self):
        """Test identical arms survive and falsify."""
        _result, payload = _run_stage()
        self.assertTrue(payload["available"])
        self.assertIsNone(payload["detail"])
        selection = payload["selection"]
        self.assertIsNone(selection["withheld"])
        self.assertTrue(selection["survived"])
        self.assertEqual(selection["fp32"], {"decision": "selected", "winner": "alpha"})
        self.assertEqual(selection["bf16"], {"decision": "selected", "winner": "alpha"})
        self.assertEqual(selection["paired_population"], ["s0", "s1", "s2", "s3"])
        self.assertIs(payload["precision_effect_falsified"], True)
        self.assertEqual(payload["totals"]["n_paired"], 8)
        self.assertEqual(payload["totals"]["n_flips"], 0)
        self.assertEqual(payload["totals"]["cells"]["pass_pass"], 7)
        self.assertEqual(payload["totals"]["cells"]["fail_fail"], 1)
        self.assertEqual(payload["unpaired"], [])
        self.assertEqual(payload["flips"], [])
        block = payload["candidates"]["alpha"]
        self.assertEqual(block["n_paired"], 4)
        self.assertEqual(block["cells"]["pass_pass"], 4)
        self.assertIs(block["fp32_passed"], True)
        self.assertIs(block["bf16_passed"], True)
        self.assertEqual(block["logprob"]["n"], 4)
        self.assertEqual(block["logprob"]["mean_delta"], 0.0)

    def test_flip_breaks_survival(self):
        """Test flip breaks survival."""
        _result, payload = _run_stage(
            bf16=_ScriptedBackend(
                _full_table(fail={("alpha", "s2")}),
                dtype="bfloat16",
                fingerprint="sha256:bf16-fp",
            )
        )
        self.assertFalse(payload["selection"]["survived"])
        self.assertEqual(payload["selection"]["bf16"], {"decision": "no_candidate", "winner": None})
        self.assertIs(payload["precision_effect_falsified"], False)
        self.assertEqual(
            payload["flips"],
            [{"candidate": "alpha", "fixture": "s2", "direction": "pass_to_fail"}],
        )
        block = payload["candidates"]["alpha"]
        self.assertEqual(block["cells"]["pass_fail"], 1)
        self.assertEqual(block["flips"], [{"fixture": "s2", "direction": "pass_to_fail"}])
        self.assertIs(block["bf16_passed"], False)
        self.assertIs(block["fp32_passed"], True)
        self.assertEqual(payload["totals"]["n_flips"], 1)

    def test_bf16_only_pass_flips_without_changing_selection(self):
        """Test bf16 only pass flips without changing selection."""
        native = _ScriptedBackend(_full_table(fail={("beta", "s1")}))
        result = _native_result(native, _cases())
        _r, payload = _run_stage(native=native, result=result)
        self.assertEqual(payload["totals"]["n_flips"], 1)
        self.assertEqual(payload["candidates"]["beta"]["cells"]["fail_pass"], 1)
        # beta gains s1 in bf16 but still fails s3 in both arms:
        # both arms select alpha alone, so the selection survives
        # even though a cell flipped (and the falsification
        # readout still falls).
        self.assertEqual(payload["selection"]["fp32"], {"decision": "selected", "winner": "alpha"})
        self.assertEqual(payload["selection"]["bf16"], {"decision": "selected", "winner": "alpha"})
        self.assertTrue(payload["selection"]["survived"])
        self.assertIs(payload["precision_effect_falsified"], False)

    def test_withheld_when_primary_not_winner_bearing(self):
        """Test withheld when primary not winner bearing."""
        native = _ScriptedBackend(_full_table(fail={("alpha", "s0"), ("beta", "s1")}))
        result = _native_result(native, _cases())
        self.assertIn(result["decision"], ("ambiguous", "no_candidate"))
        _r, payload = _run_stage(native=native, result=result)
        selection = payload["selection"]
        self.assertEqual(selection["withheld"], "primary_not_winner_bearing")
        self.assertIsNone(selection["survived"])
        self.assertIsNone(selection["fp32"])
        self.assertIsNone(selection["bf16"])
        self.assertEqual(selection["paired_population"], ["s0", "s1", "s2", "s3"])

    def test_withheld_on_mismatched_sets(self):
        """Test withheld on mismatched sets."""

        class _DropBackend(_ScriptedBackend):
            """Drop backend."""

            def generate(self, input_ids, *, max_new_tokens, seed, cancel):
                """Generate."""
                if "question s3?" in self.decode(input_ids):
                    raise RuntimeError("boom")
                return super().generate(
                    input_ids, max_new_tokens=max_new_tokens, seed=seed, cancel=cancel
                )

        bf16 = _DropBackend(_full_table(), dtype="bfloat16", fingerprint="sha256:bf16-fp")
        _r, payload = _run_stage(bf16=bf16)
        selection = payload["selection"]
        self.assertEqual(selection["withheld"], "mismatched_sets")
        self.assertIsNone(selection["survived"])
        self.assertEqual(
            payload["unpaired"],
            [
                {"candidate": "alpha", "fixture": "s3", "arm": "fp32"},
                {"candidate": "beta", "fixture": "s3", "arm": "fp32"},
            ],
        )
        self.assertIs(payload["candidates"]["alpha"]["bf16_passed"], None)
        self.assertEqual(payload["candidates"]["alpha"]["n_fp32_only"], 1)

    def test_stage_native_search_not_a_list(self):
        """Test stage native search not a list."""
        result = self._result()
        result["search"] = "nope"
        payload = experiments.precision_stage(
            _ScriptedBackend(_full_table()),
            _ScriptedBackend(_full_table(), dtype="bfloat16", fingerprint="sha256:bf16-fp"),
            _FORMATS,
            _cases(),
            result,
            boundary=(7,),
        )
        self.assertFalse(payload["available"])
        self.assertEqual(payload["detail"], "native_search_incomplete")
        self.assertEqual(payload["records"], [])

    @staticmethod
    def _result():
        """Result."""
        return _native_result(_ScriptedBackend(_full_table()), _cases())

    def test_native_dtype_gate(self):
        """Test native dtype gate."""
        native = _ScriptedBackend(_full_table(), dtype="bfloat16")
        result = _native_result(native, _cases())
        payload = experiments.precision_stage(
            native,
            _ScriptedBackend(_full_table(), dtype="bfloat16", fingerprint="sha256:x"),
            _FORMATS,
            _cases(),
            result,
            boundary=(7,),
        )
        self.assertFalse(payload["available"])
        self.assertEqual(payload["detail"], "native_dtype_not_float32")

    def test_arm_dtype_gate(self):
        """Test arm dtype gate."""
        payload = experiments.precision_stage(
            _ScriptedBackend(_full_table()),
            _ScriptedBackend(_full_table(), dtype="float32", fingerprint="sha256:x"),
            _FORMATS,
            _cases(),
            self._result(),
            boundary=(7,),
        )
        self.assertFalse(payload["available"])
        self.assertEqual(payload["detail"], "arm_dtype_not_bfloat16")

    def test_cancelled(self):
        """Test cancelled."""
        event = Event()
        event.set()
        payload = experiments.precision_stage(
            _ScriptedBackend(_full_table()),
            _ScriptedBackend(_full_table(), dtype="bfloat16", fingerprint="sha256:x"),
            _FORMATS,
            _cases(),
            self._result(),
            boundary=(7,),
            cancel=event,
        )
        self.assertFalse(payload["available"])
        self.assertEqual(payload["detail"], "cancelled")

    def test_records_carry_no_answer_text(self):
        """Test records carry no answer text."""
        _r, payload = _run_stage()
        blob = json.dumps(payload)
        self.assertNotIn('"answer":', blob)
        self.assertNotIn("output_ids", blob)
        record = payload["records"][0]
        self.assertEqual(len(record["fp32"]["answer_digest"]), 64)
        self.assertEqual(len(record["expected_digest"]), 64)
        self.assertIn("verdict", record["fp32"])

    def test_protocol_binds_both_fingerprints(self):
        """Test protocol binds both fingerprints."""
        _r, payload = _run_stage()
        protocol = payload["protocol"]
        self.assertEqual(protocol["fp32_fingerprint"], "sha256:native-fp")
        self.assertEqual(protocol["bf16_fingerprint"], "sha256:bf16-fp")
        self.assertEqual(protocol["native_dtype"], "float32")
        self.assertEqual(protocol["arm_dtype"], "bfloat16")
        self.assertEqual(protocol["fingerprint"], precision.protocol_fingerprint(protocol))
        self.assertEqual(protocol["candidates"], ["alpha", "beta"])

    def test_primary_result_not_mutated(self):
        """Test primary result not mutated."""
        result, payload = _run_stage()
        self.assertNotIn("precision", result)
        self.assertEqual(result["decision"], "compatible_on_fixture_set")
        self.assertTrue(payload["available"])


class TestPrecisionValidator(unittest.TestCase):
    """Test precision validator."""

    def setUp(self):
        """Set up the test fixture."""
        self.payload = _valid_payload()

    @staticmethod
    def _result():
        """Result."""
        return _native_result(_ScriptedBackend(_full_table()), _cases())

    def _tampered(self, mutate):
        """Tampered."""
        payload = copy.deepcopy(self.payload)
        mutate(payload)
        return payload

    def test_valid_payload_accepted(self):
        """Test valid payload accepted."""
        self.assertTrue(diffreports._valid_precision(self.payload))

    def test_self_diff_unchanged(self):
        """Test self diff unchanged."""
        diff = diffreports.diff_precision(self.payload, copy.deepcopy(self.payload))
        self.assertEqual(diff["available"], "both")
        self.assertFalse(diff["changed"])
        self.assertTrue(diff["comparable"])

    def test_tamper_battery(self):
        """Test tamper battery."""
        tampers = {
            "extra_top_key": lambda p: p.update({"zzz": 1}),
            "detail_set": lambda p: p.update({"detail": "x"}),
            "fingerprint_zeroed": lambda p: p["protocol"].update({"fingerprint": "0" * 64}),
            "fingerprints_equal": lambda p: p["protocol"].update(
                {"bf16_fingerprint": p["protocol"]["fp32_fingerprint"]}
            ),
            "seed_changed": lambda p: p["protocol"].update({"seed": 9}),
            "native_dtype_literal": lambda p: p["protocol"].update({"native_dtype": "bfloat16"}),
            "boundary_emptied": lambda p: p["protocol"]["boundaries"].update({"alpha": []}),
            "winner_unknown": lambda p: p.update({"primary_winner": "gamma"}),
            "winner_on_ambiguous": lambda p: p.update(
                {"primary_decision": "ambiguous", "primary_winner": "alpha"}
            ),
            "n_paired_off": lambda p: p["candidates"]["alpha"].update({"n_paired": 99}),
            "divergence_rate_off": lambda p: p["candidates"]["alpha"].update(
                {"divergence_rate": 0.5}
            ),
            "totals_off": lambda p: p["totals"].update({"n_flips": 7}),
            "survived_forged": lambda p: p["selection"].update({"survived": True}),
            "withheld_forged": lambda p: p["selection"].update({"withheld": "mismatched_sets"}),
            "paired_population_forged": lambda p: p["selection"].update({"paired_population": []}),
            "falsified_forged": lambda p: p.update({"precision_effect_falsified": True}),
            "flips_emptied": lambda p: p.update({"flips": []}),
            "flip_direction_forged": lambda p: p["flips"][0].update({"direction": "fail_to_pass"}),
            "unpaired_forged": lambda p: p.update(
                {"unpaired": [{"candidate": "alpha", "fixture": "s0", "arm": "fp32"}]}
            ),
            "cell_verdict_flipped": lambda p: p["records"][0]["fp32"].update({"verdict": False}),
            "record_duplicated": lambda p: p["records"].append(copy.deepcopy(p["records"][0])),
            "diverged_inconsistent": lambda p: p["records"][0].update({"diverged": None}),
            "arm_passed_forged_true": lambda p: p["candidates"]["alpha"].update(
                {"bf16_passed": True}
            ),
            "logprob_nan": lambda p: p["records"][0]["fp32"].update({"logprob": float("nan")}),
            "unavailable_overlaps_records": lambda p: p.update(
                {"unavailable_candidates": ["alpha"]}
            ),
        }
        for name, mutate in tampers.items():
            with self.subTest(tamper=name):
                self.assertFalse(
                    diffreports._valid_precision(self._tampered(mutate)),
                    f"tamper {name} accepted",
                )

    def test_arm_passed_none_with_complete_cells_rejected(self):
        """Test arm passed none with complete cells rejected."""
        _r, payload = _run_stage()
        payload = json.loads(json.dumps(payload))
        payload["candidates"]["alpha"]["fp32_passed"] = None
        self.assertFalse(diffreports._valid_precision(payload))

    def test_diff_backend_drift(self):
        """Test diff backend drift."""
        other = copy.deepcopy(self.payload)
        other["protocol"]["bf16_fingerprint"] = "sha256:foreign"
        other["protocol"]["fingerprint"] = precision.protocol_fingerprint(other["protocol"])
        diff = diffreports.diff_precision(self.payload, other)
        self.assertTrue(diff["backend_drift"])
        self.assertFalse(diff["comparable"])
        self.assertTrue(diff["changed"])

    def test_diff_one_sided(self):
        """Test diff one sided."""
        diff = diffreports.diff_precision(self.payload, None)
        self.assertEqual(diff["available"], "old")
        self.assertFalse(diff["changed"])
        old_report = self._result()
        old_report["precision"] = self.payload
        new_report = self._result()
        new_report["precision"] = {"available": False, "detail": "precision_unavailable"}
        diff = diffreports.diff_reports(old_report, new_report)
        self.assertTrue(diff["summary"]["precision_asymmetric"])
        self.assertFalse(diff["summary"]["identical"])


class TestPrecisionReporting(unittest.TestCase):
    """Test precision reporting."""

    def test_section_renders_without_answer_text(self):
        """Test section renders without answer text."""
        payload = _valid_payload()
        html = reporting.render_html({"precision": payload})
        self.assertIn("Load precision (fp32 vs bf16)", html)
        self.assertIn("does not survive the bf16 load on this evidence", html)
        self.assertIn("do not serve this winner in bf16 on this evidence", html)
        self.assertIn("pass to fail", html)
        self.assertNotIn("Paris", html)
        self.assertNotIn('"42"', html)

    def test_section_absent_without_payload(self):
        """Test section absent without payload."""
        self.assertNotIn("Load precision", reporting.render_html({}))
        html = reporting.render_html(
            {"precision": {"available": False, "detail": "precision_unavailable"}}
        )
        self.assertNotIn("Load precision", html)


class TestPrecisionCliWrapper(unittest.TestCase):
    """Test precision cli wrapper."""

    @staticmethod
    def _result():
        """Result."""
        return _native_result(_ScriptedBackend(_full_table()), _cases())

    @staticmethod
    def _probe(result, bf16_backend):
        """Probe."""
        cli._run_precision_probe(
            _ScriptedBackend(_full_table()),
            _FORMATS,
            result,
            _cases(),
            bf16_backend=bf16_backend,
            boundary=(7,),
            boundary_map=None,
            max_new_tokens=64,
            seed=0,
            cancel=None,
        )

    def test_gate_on_incomplete_native(self):
        """Test gate on incomplete native."""
        result = self._result()
        result["search"] = []
        self._probe(
            result,
            _ScriptedBackend(_full_table(), dtype="bfloat16", fingerprint="sha256:b"),
        )
        self.assertEqual(
            result["precision"],
            {"available": False, "detail": "native_search_incomplete"},
        )

    def test_gate_on_failed_decision(self):
        """Test gate on failed decision."""
        result = self._result()
        result["decision"] = "failed"
        self._probe(
            result,
            _ScriptedBackend(_full_table(), dtype="bfloat16", fingerprint="sha256:b"),
        )
        self.assertEqual(
            result["precision"],
            {"available": False, "detail": "native_search_incomplete"},
        )

    def test_backend_construction_failure(self):
        """Test backend construction failure."""
        result = self._result()
        self._probe(result, None)
        self.assertEqual(
            result["precision"],
            {"available": False, "detail": "precision_unavailable"},
        )

    def test_happy_path_attaches_payload_and_preserves_primary(self):
        """Test happy path attaches payload and preserves primary."""
        result = self._result()
        before = (result["decision"], result.get("selected_candidate"))
        self._probe(
            result,
            _ScriptedBackend(_full_table(), dtype="bfloat16", fingerprint="sha256:b"),
        )
        self.assertTrue(result["precision"]["available"])
        self.assertEqual((result["decision"], result.get("selected_candidate")), before)

    def test_flag_default_and_set(self):
        """Test flag default and set."""
        parser = cli.build_parser()
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
        self.assertFalse(parser.parse_args(base).precision)
        self.assertTrue(parser.parse_args(["run", "--precision"] + base[1:]).precision)


class TestPrecisionPureModule(unittest.TestCase):
    """Test precision pure module."""

    def test_fingerprint_stable_and_bound(self):
        """Test fingerprint stable and bound."""
        protocol = precision.protocol_object(
            seed=0,
            max_new_tokens=8,
            fixture_ids=["a", "b"],
            candidates=["alpha"],
            boundaries={"alpha": [7]},
            fp32_fingerprint="sha256:x",
            bf16_fingerprint="sha256:y",
            declared_dtype=None,
            native_dtype="float32",
            arm_dtype="bfloat16",
        )
        fp = precision.protocol_fingerprint(protocol)
        self.assertEqual(len(fp), 64)
        protocol["fingerprint"] = fp
        self.assertEqual(precision.protocol_fingerprint(protocol), fp)
        changed = dict(protocol, seed=1)
        self.assertNotEqual(precision.protocol_fingerprint(changed), fp)

    def test_cell_verdict_binds_boundary_and_digest(self):
        """Test cell verdict binds boundary and digest."""
        cell = {"answer_digest": "a" * 64, "boundary_found": True}
        self.assertTrue(precision.cell_verdict(cell, "a" * 64))
        self.assertFalse(precision.cell_verdict(cell, "b" * 64))
        cell = {"answer_digest": "a" * 64, "boundary_found": False}
        self.assertFalse(precision.cell_verdict(cell, "a" * 64))

    def test_derived_zero_paired_vacuates(self):
        """Test derived zero paired vacuates."""
        derived = precision.compute_derived(
            [],
            ["alpha"],
            {"alpha": {"fp32": None, "bf16": None}},
            "no_candidate",
            None,
        )
        self.assertIsNone(derived["precision_effect_falsified"])
        self.assertEqual(derived["totals"]["n_paired"], 0)
        self.assertEqual(derived["selection"]["withheld"], "primary_not_winner_bearing")
        self.assertIsNone(derived["selection"]["survived"])


class TestBackendDtype(unittest.TestCase):
    """Real tiny-checkpoint dtype contract (checkpoints build in
    seconds; the loads dominate and are shared per class).
    """

    @classmethod
    def setUpClass(cls):
        """Set up the test fixture."""
        if importlib.util.find_spec("torch") is None:  # pragma: no cover
            raise unittest.SkipTest("torch unavailable")
        from tests.test_adapter_integration import build_tiny_gpt2

        cls._tmp = tempfile.TemporaryDirectory()
        cls.model_dir = Path(cls._tmp.name) / "tiny"
        build_tiny_gpt2(cls.model_dir)

    @classmethod
    def tearDownClass(cls):
        """Tear down the test fixture."""
        cls._tmp.cleanup()

    def test_default_is_float32_and_fingerprint_stable(self):
        """Test default is float32 and fingerprint stable."""
        from model_protocol_lab.backends import LocalHFBackend

        default = LocalHFBackend(self.model_dir, max_context=64)
        explicit = LocalHFBackend(self.model_dir, max_context=64, dtype="float32")
        self.assertEqual(default.fingerprint, explicit.fingerprint)
        self.assertEqual(default.dtype, "float32")

    def test_bf16_fingerprint_differs_and_params_are_bf16(self):
        """Test bf16 fingerprint differs and params are bf16."""
        import torch

        from model_protocol_lab.backends import LocalHFBackend

        bf16 = LocalHFBackend(self.model_dir, max_context=64, dtype="bfloat16")
        fp32 = LocalHFBackend(self.model_dir, max_context=64)
        self.assertNotEqual(bf16.fingerprint, fp32.fingerprint)
        self.assertEqual(bf16.dtype, "bfloat16")
        self.assertEqual({p.dtype for p in bf16._model.parameters()}, {torch.bfloat16})
        gen = bf16.generate(bf16.encode("Hello"), max_new_tokens=4, seed=0, cancel=_event())
        self.assertEqual(gen.reason, "limit")
        self.assertEqual(len(gen.token_ids), 4)

    def test_declared_dtype_recorded(self):
        """Test declared dtype recorded."""
        from model_protocol_lab.backends import LocalHFBackend

        backend = LocalHFBackend(self.model_dir, max_context=64)
        self.assertEqual(backend.declared_dtype, "float32")

    def test_invalid_dtype_rejected(self):
        """Test invalid dtype rejected."""
        from model_protocol_lab.backends import LocalHFBackend

        for bad in ("float16", "int8", "FP32", ""):
            with self.subTest(dtype=bad):
                with self.assertRaises(RuntimeError) as ctx:
                    LocalHFBackend(self.model_dir, max_context=64, dtype=bad)
                self.assertEqual(str(ctx.exception), "local_backend_initialization_failed")

    def test_nonfinite_logits_surface_as_error_generation(self):
        """Test nonfinite logits surface as error generation."""
        from model_protocol_lab.backends import LocalHFBackend

        backend = LocalHFBackend(self.model_dir, max_context=64, dtype="bfloat16")
        orig = backend._model.forward

        def bad_forward(*args, **kwargs):
            """Bad forward."""
            out = orig(*args, **kwargs)
            out.logits[..., :] = float("inf")
            return out

        backend._model.forward = bad_forward
        try:
            gen = backend.generate(
                backend.encode("Hello"), max_new_tokens=4, seed=0, cancel=_event()
            )
        finally:
            backend._model.forward = orig
        # The existing generation path converts the non-finite
        # step into an error Generation, which evaluate() records
        # as a non-completed observation: a bf16-only numerical
        # failure surfaces as missing arm cells, never as a pass.
        self.assertEqual(gen.reason, "error")
        self.assertEqual(gen.token_ids, ())


def _event():
    """Event."""
    # skipcq: PYL-W0404 - test-local import isolates the module under audit
    from threading import Event

    return Event()


class TestPrecisionEndToEndTiny(unittest.TestCase):
    """Test precision end to end tiny."""

    def test_real_checkpoint_arms_and_primary_invariant(self):
        """Test real checkpoint arms and primary invariant."""
        if importlib.util.find_spec("torch") is None:  # pragma: no cover
            raise unittest.SkipTest("torch unavailable")
        from model_protocol_lab.backends import LocalHFBackend
        from model_protocol_lab.catalog import CATALOG
        from tests.test_adapter_integration import build_tiny_gpt2

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "tiny"
            build_tiny_gpt2(model_dir)
            native = LocalHFBackend(model_dir, max_context=64)
            bf16 = LocalHFBackend(model_dir, max_context=64, dtype="bfloat16")
            fmt = CATALOG["chatml"]
            cases = (
                Fixture(
                    id="real-one",
                    group="g",
                    expected="x",
                    messages=(Message("user", "Say x."),),
                ),
            )
            held = (
                Fixture(
                    id="real-held",
                    group="h",
                    expected="y",
                    messages=(Message("user", "Say y."),),
                ),
            )
            result = experiments.run_experiment(
                native,
                (fmt,),
                experiments.Split(search=cases, held_out=held),
                boundary=(999,),
                max_new_tokens=6,
                seed=5,
            )
            payload = experiments.precision_stage(
                native,
                bf16,
                (fmt,),
                cases,
                result,
                boundary=(999,),
                max_new_tokens=6,
                seed=5,
            )
            self.assertTrue(payload["available"], payload)
            self.assertEqual(payload["protocol"]["fp32_fingerprint"], native.fingerprint)
            self.assertEqual(payload["protocol"]["bf16_fingerprint"], bf16.fingerprint)
            self.assertEqual(payload["totals"]["n_paired"], 1)
            self.assertTrue(diffreports._valid_precision(payload))
            # The primary measurement is untouched by the probe.
            self.assertEqual(result["backend_fingerprint"], native.fingerprint)
            self.assertNotIn("precision", result)


if __name__ == "__main__":
    unittest.main()
