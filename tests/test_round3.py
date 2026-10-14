"""Round-3 feature tests: template conformance, failure taxonomy, portability,
stop-behavior matrix, and prompt-boundary sensitivity sweep.

All backends/tokenizers here are synthetic and scripted; no model weights,
no network, no downloads.
"""

import importlib.util
import json
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event

from model_protocol_lab.backends import (
    StopOutcome,
    TruncatedPromptBackend,
    emulate_eos_stop,
    emulate_hf_stop_string,
    emulate_vllm_holdback,
)
from model_protocol_lab.catalog import CATALOG
from model_protocol_lab.cli import _backend_eos_id, _matrix_stage_rows, build_parser
from model_protocol_lab.core import (
    Fixture,
    Format,
    Generation,
    Message,
    Split,
    check_boundary_portability,
    template_diff,
    truncate_text_tail,
)
from model_protocol_lab.doctor import (
    _portability_check,
    doctor,
    examine_model,
    template_conformance,
)
from model_protocol_lab.experiments import (
    classify_observation,
    failure_taxonomy,
    prompt_sensitivity_sweep,
    run_experiment,
    stop_matrix,
)
from model_protocol_lab.reporting import render_html
from tests.test_core import BASE, FakeBackend, case, ids
from tests.test_doctor import write_model_dir

BOUNDARY = ids("|")  # (124,)
EOS_ID = 99

# Round-3 diagnostics shell out to the optional HF stack (transformers /
# tokenizers / torch) for tokenizer loading and template rendering; without
# the 'hf' extra those test classes skip instead of erroring, keeping the
# README [dev]-only quick-start path green.
HAS_HF = all(
    importlib.util.find_spec(name) is not None for name in ("transformers", "tokenizers", "torch")
)


def split():
    return Split((case("s1"), case("s2")), (case("h1"), case("h2")))


class TemplateDiffTests(unittest.TestCase):
    def test_identical_sequences_yield_empty_diff(self):
        self.assertEqual(template_diff((1, 2, 3), (1, 2, 3)), ())
        self.assertEqual(template_diff((), ()), ())

    def test_single_divergent_region_reports_exact_spans(self):
        segments = template_diff((1, 2, 3, 4), (1, 9, 4))
        self.assertEqual(len(segments), 1)
        (segment,) = segments
        self.assertEqual(segment["hand_span"], [1, 3])
        self.assertEqual(segment["template_span"], [1, 2])
        self.assertEqual(segment["hand_ids"], [2, 3])
        self.assertEqual(segment["template_ids"], [9])

    def test_multiple_divergent_regions(self):
        segments = template_diff((1, 2, 3, 4, 5), (1, 8, 3, 9, 5))
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0]["hand_ids"], [2])
        self.assertEqual(segments[0]["template_ids"], [8])
        self.assertEqual(segments[1]["hand_ids"], [4])
        self.assertEqual(segments[1]["template_ids"], [9])

    def test_insertion_and_deletion(self):
        inserted = template_diff((1, 2), (1, 7, 8, 2))
        self.assertEqual(len(inserted), 1)
        self.assertEqual(inserted[0]["hand_ids"], [])
        self.assertEqual(inserted[0]["template_ids"], [7, 8])
        deleted = template_diff((1, 7, 8, 2), (1, 2))
        self.assertEqual(len(deleted), 1)
        self.assertEqual(deleted[0]["hand_ids"], [7, 8])
        self.assertEqual(deleted[0]["template_ids"], [])

    def test_segments_reconstruct_template_ids_exactly(self):
        hand = (10, 11, 12, 13, 14, 15, 16)
        template = (10, 99, 13, 14, 88, 16)
        segments = template_diff(hand, template)
        rebuilt = list(hand)
        for segment in reversed(segments):
            start, end = segment["hand_span"]
            rebuilt[start:end] = segment["template_ids"]
        self.assertEqual(tuple(rebuilt), template)

    def test_invalid_inputs_raise(self):
        for bad in ("12", [1, 2], (1, -2), (1, True), (1, 2.0), None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                template_diff(bad, (1, 2))
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                template_diff((1, 2), bad)

    def test_pathological_fragmentation_degrades_instead_of_hanging(self):
        # Alternating equal/divergent tokens make difflib super-quadratic
        # (~45 s at n=2000, unbounded above); the diff must degrade to a
        # fixed ValueError inside its budget instead of hanging the caller.
        hand = tuple([1, 2] * 600)
        template = tuple([1, 3] * 600)
        with self.assertRaises(ValueError) as ctx:
            template_diff(hand, template, _deadline_s=1.0)
        self.assertEqual(str(ctx.exception), "template_diff_too_fragmented")

    def test_ordinary_large_diff_unaffected_by_budget(self):
        # Localized differences in long sequences stay fast and exact.
        hand = tuple(range(5000))
        template = list(hand)
        template[1000] = 999999
        template[3000] = 888888
        segments = template_diff(tuple(template), hand, _deadline_s=1.0)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0]["hand_span"], [1000, 1001])


LLAMA3_TEMPLATE = (
    "{% set loop_messages = messages %}"
    "{% for message in loop_messages %}"
    "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'"
    " + message['content'] | trim + '<|eot_id|>' %}"
    "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}"
    "{{ content }}"
    "{% endfor %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
)


def write_template_model_dir(root: Path, *, template: str | None) -> Path:
    """Minimal checkpoint dir with a real BPE tokenizer.json for conformance."""
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel

    vocab = {
        "<|begin_of_text|>": 0,
        "<|start_header_id|>": 1,
        "<|end_header_id|>": 2,
        "<|eot_id|>": 3,
        "u": 4,
        "s": 5,
        "e": 6,
        "r": 7,
        "h": 8,
        "i": 9,
        " ": 10,
        "<unk>": 11,
        "us": 12,
        "use": 13,
        "user": 14,
        "hi": 15,
    }
    merges = [("u", "s"), ("us", "e"), ("use", "r"), ("h", "i")]
    tokenizer = Tokenizer(BPE(vocab=vocab, merges=merges, unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.add_special_tokens(
        ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]
    )
    tokenizer.save(str(root / "tokenizer.json"))
    config = {"bos_token": "<|begin_of_text|>"}
    if template is not None:
        config["chat_template"] = template
    (root / "tokenizer_config.json").write_text(json.dumps(config), encoding="utf-8")
    return root


def llama_case(content: str, name: str = "c1") -> Fixture:
    return Fixture(name, (Message("user", content),), "ok", "group-" + name)


@unittest.skipUnless(HAS_HF, "the 'hf' extra is not installed")
class TemplateConformanceTests(unittest.TestCase):
    def test_llama3_hypothesis_byte_identical_to_real_template(self):
        with TemporaryDirectory() as tmp:
            write_template_model_dir(Path(tmp), template=LLAMA3_TEMPLATE)
            outcome = template_conformance(tmp, (CATALOG["llama3"],), (llama_case("hi"),))
        self.assertEqual(outcome["status"], "ok")
        self.assertEqual(len(outcome["diffs"]), 1)
        entry = outcome["diffs"][0]
        self.assertEqual(entry["status"], "ok")
        self.assertEqual(entry["diff"], [])
        self.assertEqual(entry["hand_tokens"], entry["template_tokens"])

    def test_trimmed_content_reports_exact_divergent_subsequences(self):
        from transformers import PreTrainedTokenizerFast

        with TemporaryDirectory() as tmp:
            root = write_template_model_dir(Path(tmp), template=LLAMA3_TEMPLATE)
            outcome = template_conformance(
                tmp, (CATALOG["llama3"],), (llama_case("  hi  ", "padded"),)
            )
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=str(root / "tokenizer.json"),
                clean_up_tokenization_spaces=False,
            )
            tokenizer.chat_template = LLAMA3_TEMPLATE
            tokenizer.bos_token = "<|begin_of_text|>"
            fmt = CATALOG["llama3"]
            fixture = llama_case("  hi  ", "padded")
            hand_ids = tuple(
                tokenizer.encode(fmt.render(fixture.messages), add_special_tokens=False)
            )
            rendered = tokenizer.apply_chat_template(
                [{"role": m.role, "content": m.content} for m in fixture.messages],
                tokenize=False,
                add_generation_prompt=True,
            )
            template_ids = tuple(tokenizer.encode(rendered, add_special_tokens=False))
        self.assertEqual(outcome["status"], "ok")
        (entry,) = outcome["diffs"]
        self.assertEqual(entry["status"], "ok")
        # The only textual difference is the trimmed padding: the diff must
        # be non-empty and reconstruct the template IDs exactly.
        self.assertTrue(entry["diff"])
        rebuilt = list(hand_ids)
        for segment in reversed(entry["diff"]):
            start, end = segment["hand_span"]
            rebuilt[start:end] = segment["template_ids"]
        self.assertEqual(tuple(rebuilt), template_ids)
        self.assertNotEqual(list(hand_ids), list(template_ids))

    def test_missing_config_reports_no_template_declared(self):
        with TemporaryDirectory() as tmp:
            outcome = template_conformance(tmp, (CATALOG["llama3"],), (llama_case("hi"),))
        self.assertEqual(outcome["status"], "no_template_declared")
        self.assertEqual(outcome["detail"], "tokenizer_config_unreadable")
        self.assertEqual(outcome["diffs"], [])

    def test_config_without_template_reports_no_template_declared(self):
        with TemporaryDirectory() as tmp:
            write_template_model_dir(Path(tmp), template=None)
            outcome = template_conformance(tmp, (CATALOG["llama3"],), (llama_case("hi"),))
        self.assertEqual(outcome["status"], "no_template_declared")
        self.assertEqual(outcome["detail"], "no_chat_template_in_config")

    def test_blank_template_reports_no_template_declared(self):
        with TemporaryDirectory() as tmp:
            write_template_model_dir(Path(tmp), template="   ")
            outcome = template_conformance(tmp, (CATALOG["llama3"],), (llama_case("hi"),))
        self.assertEqual(outcome["status"], "no_template_declared")

    def test_missing_model_dir_is_unavailable_not_a_crash(self):
        with TemporaryDirectory() as tmp:
            outcome = template_conformance(
                str(Path(tmp) / "nope"), (CATALOG["llama3"],), (llama_case("hi"),)
            )
        self.assertEqual(outcome["status"], "unavailable")

    def test_unrenderable_pair_is_unavailable_not_a_crash(self):
        bad = Fixture(
            "bad",
            (Message("user", "q"), Message("assistant", "a")),
            "ok",
            "group-bad",
        )
        with TemporaryDirectory() as tmp:
            write_template_model_dir(Path(tmp), template=LLAMA3_TEMPLATE)
            outcome = template_conformance(tmp, (CATALOG["llama3"],), (bad,))
        self.assertEqual(outcome["status"], "ok")
        self.assertEqual(outcome["diffs"][0]["status"], "unavailable")

    def test_invalid_arguments_raise(self):
        with self.assertRaises(ValueError):
            template_conformance("unused-path", (), (llama_case("hi"),))
        with self.assertRaises(ValueError):
            template_conformance("unused-path", (CATALOG["llama3"],), ())
        with self.assertRaises(ValueError):
            template_conformance("unused-path", ("nope",), (llama_case("hi"),))

    def test_oversized_tokenizer_json_never_loaded(self):
        # template_conformance re-reads tokenizer.json for the HF loader; the
        # 128 MB pre-load cap must apply here too, degrading to
        # "unavailable" without parsing the file.
        from unittest.mock import patch

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_template_model_dir(root, template=LLAMA3_TEMPLATE)
            with open(root / "tokenizer.json", "wb") as handle:
                handle.truncate(200 * 1024 * 1024)  # sparse: size without disk use
            with patch("transformers.PreTrainedTokenizerFast") as loader:
                outcome = template_conformance(tmp, (CATALOG["llama3"],), (llama_case("hi"),))
            loader.assert_not_called()
        self.assertEqual(outcome["status"], "unavailable")
        self.assertEqual(outcome["detail"], "tokenizer_load_failed")

    def test_oversized_template_source_is_unavailable(self):
        # A megabyte-plus chat template is never handed to Jinja.
        with TemporaryDirectory() as tmp:
            write_template_model_dir(Path(tmp), template="x" * 1_000_001)
            outcome = template_conformance(tmp, (CATALOG["llama3"],), (llama_case("hi"),))
        self.assertEqual(outcome["status"], "unavailable")
        self.assertEqual(outcome["detail"], "chat_template_too_large")

    def test_render_bomb_degrades_pair_not_run(self):
        # A hostile template rendering megabytes of text degrades its pair
        # instead of feeding a memory bomb to the encoder.
        with TemporaryDirectory() as tmp:
            write_template_model_dir(Path(tmp), template="{{ 'x' * 2000000 }}")
            outcome = template_conformance(tmp, (CATALOG["llama3"],), (llama_case("hi"),))
        self.assertEqual(outcome["status"], "ok")
        self.assertEqual(len(outcome["diffs"]), 1)
        self.assertEqual(outcome["diffs"][0]["status"], "unavailable")


class BoundaryPortabilityTests(unittest.TestCase):
    def test_identical_encodings_are_portable(self):
        outcome = check_boundary_portability("ab", (1, 2), lambda text: (1, 2))
        self.assertEqual(
            outcome,
            {
                "portable": True,
                "primary_ids": [1, 2],
                "secondary_ids": [1, 2],
                "detail": "identical",
            },
        )

    def test_divergent_encoding_is_flagged_with_both_id_lists(self):
        outcome = check_boundary_portability("ab", (1, 2), lambda text: (7,))
        self.assertFalse(outcome["portable"])
        self.assertEqual(outcome["primary_ids"], [1, 2])
        self.assertEqual(outcome["secondary_ids"], [7])
        self.assertEqual(outcome["detail"], "divergent_ids")

    def test_secondary_encode_failure_is_reported_not_raised(self):
        def boom(text):
            raise RuntimeError("synthetic tokenizer failure")

        outcome = check_boundary_portability("ab", (1, 2), boom)
        self.assertFalse(outcome["portable"])
        self.assertIsNone(outcome["secondary_ids"])
        self.assertEqual(outcome["detail"], "secondary_encode_failed")

    def test_invalid_secondary_output_is_reported(self):
        outcome = check_boundary_portability("ab", (1, 2), lambda text: [1, 2])
        self.assertFalse(outcome["portable"])
        self.assertEqual(outcome["detail"], "secondary_encode_invalid")

    def test_invalid_arguments_raise(self):
        with self.assertRaises(ValueError):
            check_boundary_portability("", (1,), lambda text: (1,))
        with self.assertRaises(ValueError):
            check_boundary_portability("ab", (), lambda text: ())
        with self.assertRaises(ValueError):
            check_boundary_portability("ab", (1, -1), lambda text: (1,))
        with self.assertRaises(ValueError):
            check_boundary_portability("ab", (1,), "not-callable")


def write_wordlevel_model_dir(root: Path, words: list[str]) -> Path:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    root.mkdir(parents=True, exist_ok=True)
    vocab = {word: index for index, word in enumerate(words)}
    vocab["[UNK]"] = len(words)
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.add_special_tokens(["[UNK]"])
    write_model_dir(root)
    tokenizer.save(str(root / "tokenizer.json"))
    return root


@unittest.skipUnless(HAS_HF, "the 'hf' extra is not installed")
class DoctorPortabilityTests(unittest.TestCase):
    def test_portable_boundary_reports_ok(self):
        with TemporaryDirectory() as tmp:
            primary = write_wordlevel_model_dir(
                Path(tmp) / "primary", [f"tok{i}" for i in range(15)]
            )
            reference = write_wordlevel_model_dir(
                Path(tmp) / "reference", [f"tok{i}" for i in range(15)]
            )
            report = doctor(primary, boundary_ids=(5,), reference_model_path=reference)
        check = report["model"]["checks"]["boundary_portability"]
        self.assertEqual(check["status"], "ok")
        self.assertEqual(report["verdict"], "ok")

    def test_non_portable_boundary_is_mismatch_not_failure(self):
        with TemporaryDirectory() as tmp:
            primary = write_wordlevel_model_dir(
                Path(tmp) / "primary", [f"tok{i}" for i in range(15)]
            )
            reference = write_wordlevel_model_dir(
                Path(tmp) / "reference", [f"w{i}" for i in range(15)]
            )
            report = doctor(primary, boundary_ids=(5,), reference_model_path=reference)
        check = report["model"]["checks"]["boundary_portability"]
        self.assertEqual(check["status"], "mismatch")
        self.assertIn("primary=[5]", check["detail"])
        self.assertIn("reference=[15]", check["detail"])
        # Informational only: a tokenizer-specific boundary is expected.
        self.assertEqual(report["verdict"], "ok")

    def test_no_reference_model_skips_check(self):
        with TemporaryDirectory() as tmp:
            primary = write_wordlevel_model_dir(
                Path(tmp) / "primary", [f"tok{i}" for i in range(15)]
            )
            report = examine_model(primary, boundary_ids=(5,))
        self.assertNotIn("boundary_portability", report["model"]["checks"])

    def test_reference_without_boundary_ids_skips(self):
        with TemporaryDirectory() as tmp:
            primary = write_wordlevel_model_dir(
                Path(tmp) / "primary", [f"tok{i}" for i in range(15)]
            )
            reference = write_wordlevel_model_dir(
                Path(tmp) / "reference", [f"tok{i}" for i in range(15)]
            )
            report = examine_model(primary, reference_model_path=reference)
        self.assertEqual(report["model"]["checks"]["boundary_portability"]["status"], "skipped")

    def test_unreadable_reference_skips_gracefully(self):
        with TemporaryDirectory() as tmp:
            primary = write_wordlevel_model_dir(
                Path(tmp) / "primary", [f"tok{i}" for i in range(15)]
            )
            report = examine_model(
                primary,
                boundary_ids=(5,),
                reference_model_path=Path(tmp) / "missing",
            )
        check = report["model"]["checks"]["boundary_portability"]
        self.assertEqual(check["status"], "skipped")
        self.assertEqual(report["verdict"], "ok")

    def test_portability_check_unit_cases(self):
        self.assertEqual(_portability_check("/nonexistent", (5,), None)["status"], "skipped")
        self.assertEqual(_portability_check("/nonexistent", None, object())["status"], "skipped")
        self.assertEqual(_portability_check("/nonexistent", (5, -1), object())["status"], "skipped")

        class ExplodingTokenizer:
            def decode(self, ids):
                raise RuntimeError("synthetic decode failure")

        self.assertEqual(
            _portability_check("/nonexistent", (5,), ExplodingTokenizer())["status"],
            "skipped",
        )

    def test_oversized_reference_tokenizer_never_loaded(self):
        # The reference-model path must enforce the same 128 MB pre-load cap
        # as the backend: a swapped-in oversized tokenizer.json must degrade
        # to "skipped" without transformers ever parsing it.
        from unittest.mock import patch

        class StubPrimary:
            def decode(self, ids):
                return "tok5"

        with TemporaryDirectory() as tmp:
            reference = Path(tmp) / "reference"
            write_wordlevel_model_dir(reference, [f"tok{i}" for i in range(15)])
            with open(reference / "tokenizer.json", "wb") as handle:
                handle.truncate(200 * 1024 * 1024)  # sparse: size without disk use
            with patch("transformers.PreTrainedTokenizerFast") as loader:
                check = _portability_check(str(reference), (5,), StubPrimary())
            loader.assert_not_called()
        self.assertEqual(check["status"], "skipped")
        self.assertEqual(check["detail"], "reference_tokenizer_unavailable")


class FailureTaxonomyTests(unittest.TestCase):
    def run_taxonomy(self, backend, *, eos_id=None, max_new_tokens=5):
        result = run_experiment(
            backend,
            (BASE,),
            split(),
            boundary=BOUNDARY,
            max_new_tokens=max_new_tokens,
            eos_id=eos_id,
        )
        return result["decision"], result["failure_taxonomy"]

    def test_no_boundary_in_n(self):
        backend = FakeBackend(generation=Generation(ids("WRONG"), "limit"))
        decision, taxonomy = self.run_taxonomy(backend)
        self.assertEqual(decision, "no_candidate")
        counts = taxonomy["per_candidate"]["base"]
        self.assertEqual(counts["no-boundary-in-N"], 2)
        self.assertEqual(sum(counts.values()), 2)

    def test_boundary_text_mismatch(self):
        backend = FakeBackend(
            generation=Generation((10, 20), "limit"),
            decode_script=["ok|", "|", "ok|", "ok|", "|", "ok|"],
        )
        decision, taxonomy = self.run_taxonomy(backend, max_new_tokens=2)
        counts = taxonomy["per_candidate"]["base"]
        self.assertEqual(counts["boundary-text-mismatch"], 2)

    def test_eos_without_boundary(self):
        backend = FakeBackend(generation=Generation((7, EOS_ID, 8), "limit"))
        decision, taxonomy = self.run_taxonomy(backend, eos_id=EOS_ID, max_new_tokens=3)
        counts = taxonomy["per_candidate"]["base"]
        self.assertEqual(counts["eos-without-boundary"], 2)

    def test_eos_ignored_without_eos_id(self):
        backend = FakeBackend(generation=Generation((7, EOS_ID, 8), "limit"))
        decision, taxonomy = self.run_taxonomy(backend, eos_id=None, max_new_tokens=3)
        counts = taxonomy["per_candidate"]["base"]
        self.assertEqual(counts["no-boundary-in-N"], 2)
        self.assertFalse(taxonomy["eos_id_considered"])

    def test_held_out_divergence(self):
        bad = Generation(ids("BAD"), "limit")
        backend = FakeBackend(
            generation=Generation(ids("ok|"), "limit"),
            generate_script=[
                Generation(ids("ok|"), "limit"),
                Generation(ids("ok|"), "limit"),
                bad,
                bad,
            ],
        )
        decision, taxonomy = self.run_taxonomy(backend, max_new_tokens=3)
        self.assertEqual(decision, "held_out_failed")
        search_counts = taxonomy["per_candidate"]["base"]
        self.assertEqual(sum(search_counts.values()), 0)
        self.assertEqual(taxonomy["held_out"]["held-out-divergence"], 2)

    def test_backend_error(self):
        backend = FakeBackend(generate_script=[Generation((), "error")])
        decision, taxonomy = self.run_taxonomy(backend)
        self.assertEqual(decision, "failed")
        counts = taxonomy["per_candidate"]["base"]
        self.assertEqual(counts["backend-error"], 1)

    def test_cancelled_observation_is_backend_error(self):
        backend = FakeBackend(generate_script=[Generation((1, 2), "cancelled")])
        decision, taxonomy = self.run_taxonomy(backend)
        self.assertEqual(decision, "cancelled")
        counts = taxonomy["per_candidate"]["base"]
        self.assertEqual(counts["backend-error"], 1)

    def test_wrong_answer_with_boundary_is_no_boundary_in_n(self):
        backend = FakeBackend(generation=Generation(ids("WRONG|"), "limit"))
        decision, taxonomy = self.run_taxonomy(backend, max_new_tokens=6)
        counts = taxonomy["per_candidate"]["base"]
        # Documented reading: the expected answer never appeared
        # boundary-delimited within the cap.
        self.assertEqual(counts["no-boundary-in-N"], 2)

    def test_tied_winners_get_zero_counts(self):
        other = Format("other", "<B>", "<U>", "<A>", "<E>")
        result = run_experiment(
            FakeBackend(), (BASE, other), split(), boundary=BOUNDARY, max_new_tokens=7
        )
        self.assertEqual(result["decision"], "ambiguous")
        taxonomy = result["failure_taxonomy"]
        for counts in taxonomy["per_candidate"].values():
            self.assertEqual(sum(counts.values()), 0)

    def test_every_mode_present_in_taxonomy_shape(self):
        backend = FakeBackend(generation=Generation(ids("WRONG"), "limit"))
        _, taxonomy = self.run_taxonomy(backend)
        self.assertEqual(
            taxonomy["modes"],
            [
                "no-boundary-in-N",
                "boundary-text-mismatch",
                "eos-without-boundary",
                "held-out-divergence",
                "backend-error",
            ],
        )
        self.assertIsNone(taxonomy["held_out"])

    def test_invalid_eos_id_rejected(self):
        with self.assertRaises(ValueError):
            run_experiment(FakeBackend(), (BASE,), split(), boundary=BOUNDARY, eos_id=-1)
        with self.assertRaises(ValueError):
            run_experiment(FakeBackend(), (BASE,), split(), boundary=BOUNDARY, eos_id="99")

    def test_classify_observation_rejects_non_dict(self):
        with self.assertRaises(ValueError):
            classify_observation("nope", None)
        with self.assertRaises(ValueError):
            failure_taxonomy("nope", None)

    def test_decision_banner_cites_taxonomy(self):
        backend = FakeBackend(generation=Generation(ids("WRONG"), "limit"))
        result = run_experiment(backend, (BASE,), split(), boundary=BOUNDARY, max_new_tokens=5)
        html = render_html(result)
        self.assertIn("Failure-mode taxonomy:", html)
        self.assertIn("no-boundary-in-N", html)

    def test_clean_pass_has_no_taxonomy_note(self):
        result = run_experiment(
            FakeBackend(), (BASE,), split(), boundary=BOUNDARY, max_new_tokens=7
        )
        self.assertEqual(result["decision"], "compatible_on_fixture_set")
        html = render_html(result)
        self.assertNotIn("Failure-mode taxonomy:", html)


class HFOverhangTests(unittest.TestCase):
    """Unit tests for the StopStringCriteria port against the HF docstring."""

    def test_hf_docstring_positive_examples(self):
        from model_protocol_lab.backends import _hf_overhang_match

        positives = [
            (["st", "op"], "stop"),
            (["stop"], "stop"),
            (["st", "opera"], "stop"),
            (["sto", "pper"], "stop"),
            (["las", "topper"], "stop"),
            (["s", "to", "pped"], "stop"),
        ]
        for tokens, stop in positives:
            with self.subTest(tokens=tokens):
                self.assertTrue(_hf_overhang_match(tokens, stop))

    def test_hf_docstring_negative_examples(self):
        from model_protocol_lab.backends import _hf_overhang_match

        negatives = [
            (["stop", "at"], "stop"),
            (["st", "op", "at"], "stop"),
            (["st", "opera", "tion"], "stop"),
        ]
        for tokens, stop in negatives:
            with self.subTest(tokens=tokens):
                self.assertFalse(_hf_overhang_match(tokens, stop))

    def test_empty_inputs_never_match(self):
        from model_protocol_lab.backends import _hf_overhang_match

        self.assertFalse(_hf_overhang_match([], "stop"))
        self.assertFalse(_hf_overhang_match(["stop"], ""))
        self.assertFalse(_hf_overhang_match(["x"], "stop"))

    def test_multiple_end_overlaps_all_explored(self):
        from model_protocol_lab.backends import _hf_overhang_match

        # "aaa" overlaps "aa" at lengths 1 and 2; both must be tried.
        self.assertTrue(_hf_overhang_match(["xaa", "aab"], "aab"))
        self.assertTrue(_hf_overhang_match(["aaa"], "aa"))

    def test_mid_token_occurrence_matches_real_hf(self):
        # The tensor implementation records an end-overlap whenever the
        # stop string *ends inside* the final token, not only at its end or
        # as a prefix overhang: the docstring's "impossible to construct"
        # claim does not hold for the implementation. The port used to miss
        # this whole class, so the stop matrix under-reported HF stops.
        from model_protocol_lab.backends import _hf_overhang_match

        self.assertTrue(_hf_overhang_match([" world"], "or"))
        self.assertTrue(_hf_overhang_match(["hello ", "xstop"], "stop"))
        self.assertTrue(_hf_overhang_match(["xstopper"], "stop"))
        self.assertTrue(_hf_overhang_match(["stop"], "to"))
        self.assertTrue(_hf_overhang_match(["las"], "as"))
        # Near-misses still rejected: no occurrence, no overhang.
        self.assertFalse(_hf_overhang_match(["hello ", "xsto"], "stop"))
        self.assertFalse(_hf_overhang_match([" wold"], "or"))
        self.assertFalse(_hf_overhang_match(["stap"], "top"))


def _oracle_bpe_tokenizer(path):
    """Tiny BPE tokenizer whose token strings concatenate cleanly, so HF's
    own StopStringCriteria can serve as a test oracle."""
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel

    chains = {
        "st": [("s", "t")],
        "op": [("o", "p")],
        "stop": [("s", "t"), ("st", "o"), ("sto", "p")],
        "opera": [("o", "p"), ("op", "e"), ("ope", "r"), ("oper", "a")],
        "sto": [("s", "t"), ("st", "o")],
        "pper": [("p", "p"), ("pp", "e"), ("ppe", "r")],
        "las": [("l", "a"), ("la", "s")],
        "topper": [("t", "o"), ("to", "p"), ("top", "p"), ("topp", "e"), ("toppe", "r")],
        "to": [("t", "o")],
        "pped": [("p", "p"), ("pp", "e"), ("ppe", "d")],
        "at": [("a", "t")],
        "tion": [("t", "i"), ("ti", "o"), ("tio", "n")],
    }
    vocab = {}
    for char in "abcdefhilnoprstx":
        vocab[char] = len(vocab)
    merges = []
    for word, chain in chains.items():
        for first, second in chain:
            product = first + second
            if product not in vocab:
                vocab[product] = len(vocab)
            merges.append((first, second))
        if word not in vocab:
            vocab[word] = len(vocab)
    vocab["<unk>"] = len(vocab)
    tokenizer = Tokenizer(BPE(vocab=vocab, merges=merges, unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.save(str(path))
    return vocab


@unittest.skipUnless(HAS_HF, "the 'hf' extra is not installed")
class HFOracleTests(unittest.TestCase):
    """The port agrees with the real transformers StopStringCriteria."""

    def test_matches_oracle_on_docstring_cases(self):
        import torch
        from transformers import PreTrainedTokenizerFast
        from transformers.generation.stopping_criteria import StopStringCriteria

        from model_protocol_lab.backends import _hf_overhang_match

        with TemporaryDirectory() as tmp:
            vocab = _oracle_bpe_tokenizer(Path(tmp) / "tokenizer.json")
            hf_tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=str(Path(tmp) / "tokenizer.json"),
                clean_up_tokenization_spaces=False,
            )
            criterion = StopStringCriteria(hf_tokenizer, "stop")
            cases = [
                (["st", "op"], True),
                (["stop"], True),
                (["st", "opera"], True),
                (["sto", "pper"], True),
                (["las", "topper"], True),
                (["s", "to", "pped"], True),
                (["stop", "at"], False),
                (["st", "op", "at"], False),
                (["st", "opera", "tion"], False),
            ]
            for tokens, expected in cases:
                with self.subTest(tokens=tokens):
                    ids = torch.tensor([[vocab[token] for token in tokens]])
                    oracle = bool(criterion(ids, None).item())
                    self.assertEqual(oracle, expected)
                    self.assertEqual(_hf_overhang_match(tokens, "stop"), oracle)

    def test_mid_token_occurrence_agrees_with_oracle(self):
        # The implementation fires when the stop string ends inside the
        # final token (e.g. "to" inside "stop"); the port used to miss
        # this class entirely.
        import torch
        from transformers import PreTrainedTokenizerFast
        from transformers.generation.stopping_criteria import StopStringCriteria

        from model_protocol_lab.backends import _hf_overhang_match

        with TemporaryDirectory() as tmp:
            vocab = _oracle_bpe_tokenizer(Path(tmp) / "tokenizer.json")
            hf_tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=str(Path(tmp) / "tokenizer.json"),
                clean_up_tokenization_spaces=False,
            )
            cases = [
                ("to", ["stop"], True),
                ("as", ["las"], True),
                ("to", ["st", "op"], True),
                ("to", ["st", "st"], False),
                ("to", ["st", "pper"], False),
            ]
            for stop, tokens, expected in cases:
                with self.subTest(stop=stop, tokens=tokens):
                    criterion = StopStringCriteria(hf_tokenizer, stop)
                    ids = torch.tensor([[vocab[token] for token in tokens]])
                    oracle = bool(criterion(ids, None).item())
                    self.assertEqual(oracle, expected)
                    self.assertEqual(_hf_overhang_match(tokens, stop), oracle)


class StopEmulationTests(unittest.TestCase):
    def test_hf_emulation_fires_at_completing_token(self):
        texts = {1: "a", 2: "b", 3: "stop"}
        backend = FakeBackend(decode_script=[texts[i] for i in (1, 2, 3)])
        outcome = emulate_hf_stop_string(backend, (1,), (2, 3), "stop")
        self.assertTrue(outcome.available)
        self.assertTrue(outcome.fired)
        self.assertEqual(outcome.truncated_ids, (2, 3))
        self.assertEqual(outcome.detail, "fired_at_step_2")

    def test_hf_emulation_no_fire(self):
        backend = FakeBackend()
        outcome = emulate_hf_stop_string(backend, (1,), (2, 3), "stop")
        self.assertTrue(outcome.available)
        self.assertFalse(outcome.fired)
        self.assertEqual(outcome.truncated_ids, (2, 3))
        self.assertEqual(outcome.detail, "no_stop_within_cap")

    def test_hf_emulation_degrades_on_decode_failure(self):
        class BadDecode:
            fingerprint = "bad"

            def decode(self, ids):
                raise RuntimeError("synthetic")

        outcome = emulate_hf_stop_string(BadDecode(), (1,), (2,), "stop")
        self.assertFalse(outcome.available)
        self.assertEqual(outcome.detail, "token_text_unavailable")

    def test_hf_emulation_rejects_bad_input(self):
        backend = FakeBackend()
        outcome = emulate_hf_stop_string(backend, (1,), (2,), "")
        self.assertFalse(outcome.available)
        self.assertEqual(outcome.detail, "invalid_emulation_input")
        outcome = emulate_hf_stop_string(backend, (1,), [2], "stop")
        self.assertFalse(outcome.available)

    def test_vllm_emulation_truncates_before_stop_string(self):
        backend = FakeBackend()
        # decode of each prefix: "answer", "answer<stop>", "answer<stop>tail"
        outcome = emulate_vllm_holdback(backend, ids("answer<stop>tail"), "<stop>")
        self.assertTrue(outcome.available)
        self.assertTrue(outcome.fired)
        self.assertEqual(backend.decode(outcome.truncated_ids), "answer<stop>")
        self.assertEqual(outcome.detail, "fired_at_step_12")

    def test_vllm_emulation_fires_when_stop_completes(self):
        backend = FakeBackend()
        outcome = emulate_vllm_holdback(backend, ids("xab"), "ab")
        self.assertTrue(outcome.available)
        self.assertTrue(outcome.fired)
        self.assertEqual(outcome.detail, "fired_at_step_3")
        self.assertEqual(outcome.truncated_ids, ids("xab"))

    def test_vllm_emulation_no_stop_string(self):
        backend = FakeBackend()
        outcome = emulate_vllm_holdback(backend, (1, 2), "ab")
        self.assertTrue(outcome.available)
        self.assertFalse(outcome.fired)
        self.assertEqual(outcome.detail, "no_stop_within_cap")

    def test_vllm_emulation_degrades_on_decode_failure(self):
        class BadDecode:
            fingerprint = "bad"

            def decode(self, ids):
                raise RuntimeError("synthetic")

        outcome = emulate_vllm_holdback(BadDecode(), (1, 2), "stop")
        self.assertFalse(outcome.available)
        self.assertEqual(outcome.detail, "decode_unavailable")

    def test_vllm_emulation_rejects_bad_input(self):
        backend = FakeBackend()
        outcome = emulate_vllm_holdback(backend, (1,), "")
        self.assertFalse(outcome.available)

    def test_eos_emulation_truncates_before_first_eos(self):
        outcome = emulate_eos_stop((1, 99, 2, 99), 99)
        self.assertTrue(outcome.available)
        self.assertTrue(outcome.fired)
        self.assertEqual(outcome.truncated_ids, (1,))
        self.assertEqual(outcome.detail, "fired_at_step_2")

    def test_eos_emulation_no_eos(self):
        outcome = emulate_eos_stop((1, 2), 99)
        self.assertTrue(outcome.available)
        self.assertFalse(outcome.fired)
        self.assertEqual(outcome.truncated_ids, (1, 2))

    def test_eos_emulation_unavailable_without_id(self):
        outcome = emulate_eos_stop((1, 99), None)
        self.assertFalse(outcome.available)
        self.assertEqual(outcome.detail, "eos_id_unavailable")

    def test_eos_emulation_rejects_bad_id(self):
        outcome = emulate_eos_stop((1,), -1)
        self.assertFalse(outcome.available)
        self.assertEqual(outcome.detail, "invalid_eos_id")

    def test_stop_outcome_is_immutable(self):
        outcome = StopOutcome(True, False, (1,), "no_stop_within_cap")
        with self.assertRaises(FrozenInstanceError):
            outcome.fired = True


class QuirkBackend:
    """Synthetic backend where per-token texts and whole-sequence decode
    disagree -- the realistic gap between HF's vocab-text criterion and
    vLLM's incremental decode stream.

    Token texts: 10->"pre<", 11->"stop>post", 20->"REAL", 5->"<stop>",
    12->"tail". Whole-sequence decode inserts a space after "pre<" whenever
    tokens 10 and 11 co-occur, so the overhang "<stop>" spanning the 10/11
    boundary is visible to HF's token-text walk but invisible to vLLM's
    text search.
    """

    fingerprint = "quirk-backend-not-a-model"
    TEXTS = {10: "pre<", 11: "stop>post", 20: "REAL", 5: "<stop>", 12: "tail"}

    def encode(self, text):
        raise AssertionError("encode unused in stop-matrix tests")

    def decode(self, token_ids):
        ids_tuple = tuple(token_ids)
        if len(ids_tuple) == 1:
            return self.TEXTS.get(ids_tuple[0], chr(ids_tuple[0]))
        parts = [self.TEXTS.get(i, chr(i)) for i in ids_tuple]
        if len(ids_tuple) >= 2 and ids_tuple[0] == 10 and ids_tuple[1] == 11:
            return "pre< " + "".join(parts[1:])
        return "".join(parts)

    def generate(self, input_ids, *, max_new_tokens, seed, cancel):
        raise AssertionError("generate unused in stop-matrix tests")


def quirk_observation(**overrides):
    obs = {
        "candidate": "quirk",
        "fixture": "f1",
        "input_ids": (1, 2),
        "output_ids": (10, 11, 20, 5, 12),
        "answer": "pre< stop>postREAL",
        "exact_match": True,
        "boundary_found": True,
        "boundary_text_match": None,
        "raw_reason": "limit",
        "error": None,
        "status": "completed",
    }
    obs.update(overrides)
    return obs


def quirk_fixture(fid="f1", expected="pre< stop>postREAL"):
    return Fixture(fid, (Message("user", "q-" + fid),), expected, "group-" + fid)


class StopMatrixTests(unittest.TestCase):
    def test_overhang_fires_under_hf_but_not_token_id_boundary(self):
        rows = stop_matrix(
            QuirkBackend(),
            "quirk",
            (quirk_fixture(),),
            (quirk_observation(),),
            boundary=(5,),
            eos_id=99,
        )
        (row,) = rows
        self.assertTrue(row["native_pass"])
        # HF's overhang walk fires at step 2 ("pre<" + "stop>post" covers
        # "<stop>"), truncating before the real boundary token: flip.
        hf = row["cells"]["hf_stop_string"]
        self.assertTrue(hf["available"])
        self.assertTrue(hf["fired"])
        self.assertFalse(hf["pass"])
        # vLLM's text search cannot see the overhang (the decode quirk
        # breaks the literal string), so it fires at the true boundary:
        # no flip.
        vllm = row["cells"]["vllm_holdback"]
        self.assertTrue(vllm["available"])
        self.assertTrue(vllm["fired"])
        self.assertTrue(vllm["pass"])
        # No EOS present: agrees with native.
        eos = row["cells"]["eos_appended"]
        self.assertTrue(eos["available"])
        self.assertFalse(eos["fired"])
        self.assertTrue(eos["pass"])
        self.assertEqual(row["flips"], ["hf_stop_string"])

    def test_second_fixture_agrees_everywhere(self):
        fixture = quirk_fixture("f2")
        rows = stop_matrix(
            QuirkBackend(),
            "quirk",
            (quirk_fixture(), fixture),
            (quirk_observation(), quirk_observation(fixture="f2", input_ids=(1, 2))),
            boundary=(5,),
            eos_id=99,
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["flips"], ["hf_stop_string"])
        # Native verdict is per-fixture; both pass natively here.
        self.assertTrue(all(r["native_pass"] for r in rows))

    def test_eos_firing_flips_cell(self):
        rows = stop_matrix(
            QuirkBackend(),
            "quirk",
            (quirk_fixture(),),
            (
                quirk_observation(
                    output_ids=(10, 99, 5), exact_match=False, boundary_found=False, answer="xx"
                ),
            ),
            boundary=(5,),
            eos_id=99,
        )
        (row,) = rows
        eos = row["cells"]["eos_appended"]
        self.assertTrue(eos["fired"])
        self.assertEqual(eos["detail"], "fired_at_step_2")

    def test_boundary_decode_failure_disables_string_cells(self):
        class NoBoundaryText(QuirkBackend):
            def decode(self, token_ids):
                if tuple(token_ids) == (5,):
                    raise RuntimeError("synthetic")
                return super().decode(token_ids)

        rows = stop_matrix(
            NoBoundaryText(),
            "quirk",
            (quirk_fixture(),),
            (quirk_observation(),),
            boundary=(5,),
            eos_id=99,
        )
        (row,) = rows
        self.assertFalse(row["cells"]["hf_stop_string"]["available"])
        self.assertFalse(row["cells"]["vllm_holdback"]["available"])
        self.assertTrue(row["cells"]["eos_appended"]["available"])
        self.assertEqual(row["flips"], [])

    def test_incomplete_observation_disables_cells(self):
        rows = stop_matrix(
            QuirkBackend(),
            "quirk",
            (quirk_fixture(),),
            (quirk_observation(status="failed", exact_match=False, boundary_found=False),),
            boundary=(5,),
        )
        (row,) = rows
        self.assertFalse(row["native_pass"])
        for semantic in ("hf_stop_string", "vllm_holdback", "eos_appended"):
            self.assertFalse(row["cells"][semantic]["available"])
        self.assertEqual(row["flips"], [])

    def test_alignment_mismatch_raises(self):
        with self.assertRaises(ValueError):
            stop_matrix(
                QuirkBackend(),
                "quirk",
                (quirk_fixture("f1"), quirk_fixture("f2")),
                (quirk_observation(),),
                boundary=(5,),
            )
        with self.assertRaises(ValueError):
            stop_matrix(
                QuirkBackend(),
                "quirk",
                (quirk_fixture("f2"),),
                (quirk_observation(fixture="f1"),),
                boundary=(5,),
            )

    def test_invalid_arguments_raise(self):
        with self.assertRaises(ValueError):
            stop_matrix(
                QuirkBackend(), "", (quirk_fixture(),), (quirk_observation(),), boundary=(5,)
            )
        with self.assertRaises(ValueError):
            stop_matrix(
                QuirkBackend(), "quirk", (quirk_fixture(),), (quirk_observation(),), boundary=()
            )
        with self.assertRaises(ValueError):
            stop_matrix(
                QuirkBackend(),
                "quirk",
                (quirk_fixture(),),
                (quirk_observation(),),
                boundary=(5,),
                eos_id=-1,
            )

    def test_matrix_renders_in_html(self):
        rows = stop_matrix(
            QuirkBackend(),
            "quirk",
            (quirk_fixture(),),
            (quirk_observation(),),
            boundary=(5,),
            eos_id=99,
        )
        result = {
            "decision": "compatible_on_fixture_set",
            "stop_matrix": {
                "candidate": "quirk",
                "semantics": [
                    {"id": "native", "label": "Native (lab)", "description": "d"},
                    {"id": "hf_stop_string", "label": "HF", "description": "d"},
                    {"id": "vllm_holdback", "label": "vLLM", "description": "d"},
                    {"id": "eos_appended", "label": "EOS", "description": "d"},
                ],
                "search": rows,
                "held_out": None,
            },
        }
        html = render_html(result)
        self.assertIn("Stop-behavior isolation matrix", html)
        self.assertIn("hf_stop_string", html)

    def test_html_without_matrix_has_no_matrix_section(self):
        html = render_html({"decision": "no_candidate"})
        self.assertNotIn("Stop-behavior isolation matrix", html)


class TruncateTextTailTests(unittest.TestCase):
    def test_ascii_truncation(self):
        self.assertEqual(truncate_text_tail("hello", 1), "hell")
        self.assertEqual(truncate_text_tail("hello", 4), "h")

    def test_codepoint_split_returns_none(self):
        # "é" is two bytes in UTF-8; dropping 4 of "héllo"'s 6 bytes lands
        # mid-codepoint, dropping 5 removes it whole.
        self.assertIsNone(truncate_text_tail("héllo", 4))
        self.assertEqual(truncate_text_tail("héllo", 5), "h")

    def test_out_of_range_raises(self):
        for text, drop in (("", 1), ("hi", 0), ("hi", 2), ("hi", 5), ("hi", -1)):
            with self.subTest(text=text, drop=drop), self.assertRaises(ValueError):
                truncate_text_tail(text, drop)

    def test_bad_types_raise(self):
        with self.assertRaises(ValueError):
            truncate_text_tail(b"hi", 1)
        with self.assertRaises(ValueError):
            truncate_text_tail("hi", "1")


class MarkerBackend:
    """Synthetic backend whose correctness hinges on a prompt-final marker.

    encode maps a trailing "<END>" marker to the single token 50; any
    mid-token truncation of the prompt re-encodes differently and the
    backend answers wrong.
    """

    fingerprint = "marker-backend-not-a-model"

    def encode(self, text):
        if text.endswith("<END>"):
            return tuple(map(ord, text[:-5])) + (50,)
        return tuple(map(ord, text))

    def decode(self, token_ids):
        return "".join("<END>" if t == 50 else chr(t) for t in token_ids)

    def generate(self, input_ids, *, max_new_tokens, seed, cancel):
        if input_ids and input_ids[-1] == 50:
            return Generation(ids("ok|"), "limit")
        return Generation(ids("BAD"), "limit")


MARKER_FORMAT = Format("marker", "", "U:", "A:<END>", "\n")


def marker_split():
    search = (Fixture("s1", (Message("user", "q1"),), "ok", "g1"),)
    held_out = (Fixture("h1", (Message("user", "q2"),), "ok", "g2"),)
    return Split(search, held_out)


class TruncatedPromptBackendTests(unittest.TestCase):
    def test_encode_truncates_and_reencodes(self):
        backend = TruncatedPromptBackend(MarkerBackend(), 1)
        self.assertEqual(backend.encode("ab<END>"), tuple(map(ord, "ab<END")))
        self.assertIn("prompt-trunc-1b", backend.fingerprint)

    def test_decode_and_generate_pass_through(self):
        inner = MarkerBackend()
        backend = TruncatedPromptBackend(inner, 2)
        self.assertEqual(backend.decode((65,)), "A")
        result = backend.generate((50,), max_new_tokens=4, seed=0, cancel=Event())
        self.assertEqual(result.token_ids, ids("ok|"))

    def test_codepoint_split_surfaces_as_backend_error(self):
        backend = TruncatedPromptBackend(MarkerBackend(), 1)
        with self.assertRaises(ValueError):
            backend.encode("hé")

    def test_invalid_arguments_raise(self):
        with self.assertRaises(ValueError):
            TruncatedPromptBackend(MarkerBackend(), 0)
        with self.assertRaises(ValueError):
            TruncatedPromptBackend(MarkerBackend(), -2)

        class NoFingerprint:
            pass

        with self.assertRaises(ValueError):
            TruncatedPromptBackend(NoFingerprint(), 1)


class SensitivitySweepTests(unittest.TestCase):
    def test_sweep_detects_mid_token_flip(self):
        sweep = prompt_sensitivity_sweep(
            MarkerBackend(),
            (MARKER_FORMAT,),
            marker_split(),
            boundary=BOUNDARY,
            max_drop_bytes=2,
            max_new_tokens=3,
            seed=0,
        )
        self.assertEqual(sweep["native_decision"], "compatible_on_fixture_set")
        self.assertEqual(sweep["native_winner"], "marker")
        self.assertEqual(sweep["flips_at"], [1, 2])
        for depth in sweep["depths"]:
            self.assertTrue(depth["winner_changed"])
            self.assertEqual(depth["decision"], "no_candidate")
            self.assertIsNone(depth["selected_candidate"])

    def test_sweep_validates_drop_range(self):
        with self.assertRaises(ValueError):
            prompt_sensitivity_sweep(
                MarkerBackend(),
                (MARKER_FORMAT,),
                marker_split(),
                boundary=BOUNDARY,
                max_drop_bytes=0,
            )
        with self.assertRaises(ValueError):
            prompt_sensitivity_sweep(
                MarkerBackend(),
                (MARKER_FORMAT,),
                marker_split(),
                boundary=BOUNDARY,
                max_drop_bytes=9,
            )

    def test_sweep_renders_in_html(self):
        sweep = prompt_sensitivity_sweep(
            MarkerBackend(),
            (MARKER_FORMAT,),
            marker_split(),
            boundary=BOUNDARY,
            max_drop_bytes=1,
            max_new_tokens=3,
        )
        html = render_html({"decision": "no_candidate", "sensitivity_sweep": sweep})
        self.assertIn("Prompt-boundary sensitivity sweep", html)


@unittest.skipUnless(HAS_HF, "the 'hf' extra is not installed")
class ConformanceReportTests(unittest.TestCase):
    def test_conformance_renders_in_html(self):
        with TemporaryDirectory() as tmp:
            write_template_model_dir(Path(tmp), template=LLAMA3_TEMPLATE)
            conformance = template_conformance(tmp, (CATALOG["llama3"],), (llama_case("hi"),))
        html = render_html(
            {"decision": "compatible_on_fixture_set", "template_conformance": conformance}
        )
        self.assertIn("Template conformance", html)
        self.assertIn("identical", html)

    def test_no_template_declared_renders_gracefully(self):
        html = render_html(
            {
                "decision": "no_candidate",
                "template_conformance": {
                    "status": "no_template_declared",
                    "detail": "no_chat_template_in_config",
                    "diffs": [],
                },
            }
        )
        self.assertIn("Template conformance", html)
        self.assertIn("No chat template is declared", html)

    def test_unavailable_conformance_renders_gracefully(self):
        html = render_html(
            {
                "decision": "no_candidate",
                "template_conformance": {
                    "status": "unavailable",
                    "detail": "tokenizer_load_failed",
                    "diffs": [],
                },
            }
        )
        self.assertIn("tokenizer_load_failed", html)

    def test_divergent_conformance_renders_segments(self):
        with TemporaryDirectory() as tmp:
            write_template_model_dir(Path(tmp), template=LLAMA3_TEMPLATE)
            conformance = template_conformance(
                tmp, (CATALOG["llama3"],), (llama_case("  hi  ", "padded"),)
            )
        html = render_html({"decision": "no_candidate", "template_conformance": conformance})
        self.assertIn("divergent segment", html)


class CLIFeatureTests(unittest.TestCase):
    def test_parser_accepts_new_flags(self):
        parser = build_parser()
        args = parser.parse_args(["doctor", "--model", "m", "--reference-model", "r"])
        self.assertEqual(str(args.reference_model), "r")
        args = parser.parse_args(
            [
                "run",
                "--model",
                "m",
                "--dataset",
                "d",
                "--family",
                "chatml",
                "--boundary-ids",
                "1",
                "--output",
                "o",
                "--sensitivity-sweep",
                "--max-drop-bytes",
                "5",
            ]
        )
        self.assertTrue(args.sensitivity_sweep)
        self.assertEqual(args.max_drop_bytes, 5)

    def test_backend_eos_id_extraction(self):
        class WithEos:
            eos_token_id = 42

        class WithoutEos:
            pass

        class BadEos:
            eos_token_id = -3

        self.assertEqual(_backend_eos_id(WithEos()), 42)
        self.assertIsNone(_backend_eos_id(WithoutEos()))
        self.assertIsNone(_backend_eos_id(BadEos()))

    def test_matrix_stage_rows(self):
        backend = QuirkBackend()
        result = {
            "selected_candidate": "quirk",
            "search": [
                {
                    "format": {"name": "quirk"},
                    "observations": [quirk_observation()],
                }
            ],
            "held_out": {"observations": [quirk_observation(fixture="h1")]},
        }
        rows = _matrix_stage_rows(result, "search", "quirk", (quirk_fixture(),), backend, (5,), 99)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["flips"], ["hf_stop_string"])
        held = _matrix_stage_rows(
            result, "held_out", "quirk", (quirk_fixture("h1"),), backend, (5,), 99
        )
        self.assertEqual(len(held), 1)
        # Length mismatch -> None rather than a crash.
        self.assertIsNone(_matrix_stage_rows(result, "search", "quirk", (), backend, (5,), 99))
        # Missing winner entry -> None.
        self.assertIsNone(
            _matrix_stage_rows(
                {"selected_candidate": "quirk", "search": []},
                "search",
                "quirk",
                (quirk_fixture(),),
                backend,
                (5,),
                99,
            )
        )
        # Missing held_out -> None.
        self.assertIsNone(
            _matrix_stage_rows(
                {"selected_candidate": "quirk", "held_out": None},
                "held_out",
                "quirk",
                (quirk_fixture("h1"),),
                backend,
                (5,),
                99,
            )
        )


class FakeRunBackend:
    """Scripted backend standing in for LocalHFBackend in CLI e2e tests."""

    fingerprint = "fake-run-backend-not-a-model"
    eos_token_id = 999

    @staticmethod
    def _codec_ids(text):
        return tuple(map(ord, text))

    def __init__(self, model_path, *, device="cpu", max_context=2048):
        self.model_path = model_path

    def encode(self, text):
        return self._codec_ids(text)

    def decode(self, token_ids):
        return "".join(map(chr, token_ids))

    def generate(self, input_ids, *, max_new_tokens, seed, cancel):
        prompt = self.decode(input_ids)
        if "qq1" in prompt:
            answer = "A1"
        elif "qq2" in prompt:
            answer = "A2"
        else:
            answer = "WRONG"
        return Generation(self._codec_ids(answer + "|"), "limit")


def write_e2e_dataset(path: Path) -> None:
    def item(name, question, expected):
        return {
            "id": name,
            "group": "group-" + name,
            "messages": [{"role": "user", "content": question}],
            "expected": expected,
        }

    path.write_text(
        json.dumps(
            {
                "search": [item("s1", "question qq1", "A1")],
                "held_out": [item("h1", "question qq2", "A2")],
            }
        ),
        encoding="utf-8",
    )


@unittest.skipUnless(HAS_HF, "the 'hf' extra is not installed")
class RunCommandE2ETests(unittest.TestCase):
    def run_cli(self, tmp, argv, backend_cls=FakeRunBackend):
        from unittest.mock import patch

        from model_protocol_lab.cli import main

        model_dir = Path(tmp) / "model"
        model_dir.mkdir()
        write_template_model_dir(model_dir, template=LLAMA3_TEMPLATE)
        dataset = Path(tmp) / "fixtures.json"
        write_e2e_dataset(dataset)
        output = Path(tmp) / "report"
        full_argv = [
            "run",
            "--model",
            str(model_dir),
            "--dataset",
            str(dataset),
            "--family",
            "chatml",
            "--boundary-ids",
            "124",
            "--max-new-tokens",
            "3",
            "--output",
            str(output),
        ] + argv
        with patch("model_protocol_lab.backends.LocalHFBackend", backend_cls):
            code = main(full_argv)
        results = json.loads((output / "results.json").read_text(encoding="utf-8"))
        return code, results

    def test_run_attaches_diagnostics(self):
        with TemporaryDirectory() as tmp:
            code, results = self.run_cli(tmp, [])
        self.assertEqual(code, 0)
        self.assertEqual(results["decision"], "compatible_on_fixture_set")
        conformance = results["template_conformance"]
        self.assertEqual(conformance["status"], "ok")
        self.assertTrue(conformance["diffs"])
        matrix = results["stop_matrix"]
        self.assertEqual(matrix["candidate"], "chatml")
        self.assertEqual(len(matrix["search"]), 1)
        self.assertEqual(len(matrix["held_out"]), 1)
        self.assertEqual(matrix["search"][0]["flips"], [])
        taxonomy = results["failure_taxonomy"]
        self.assertEqual(sum(taxonomy["per_candidate"]["chatml"].values()), 0)
        self.assertNotIn("diagnostics_unavailable", results)

    def test_run_with_sensitivity_sweep(self):
        with TemporaryDirectory() as tmp:
            code, results = self.run_cli(tmp, ["--sensitivity-sweep", "--max-drop-bytes", "2"])
        self.assertEqual(code, 0)
        sweep = results["sensitivity_sweep"]
        self.assertEqual(sweep["native_winner"], "chatml")
        self.assertEqual(sweep["flips_at"], [])

    def test_run_survives_diagnostics_failure(self):
        from unittest.mock import patch

        def boom(*args, **kwargs):
            raise RuntimeError("synthetic diagnostics failure")

        with (
            TemporaryDirectory() as tmp,
            patch("model_protocol_lab.doctor.template_conformance", boom),
        ):
            code, results = self.run_cli(tmp, [])
        self.assertEqual(code, 0)
        self.assertTrue(results["diagnostics_unavailable"])
        self.assertEqual(results["decision"], "compatible_on_fixture_set")

    def test_run_rejects_bad_max_drop_bytes(self):
        from model_protocol_lab.cli import main

        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "report"
            with self.assertRaises(SystemExit) as ctx:
                main(
                    [
                        "run",
                        "--model",
                        "m",
                        "--dataset",
                        "d",
                        "--family",
                        "chatml",
                        "--boundary-ids",
                        "1",
                        "--output",
                        str(output),
                        "--max-drop-bytes",
                        "9",
                    ]
                )
        self.assertEqual(ctx.exception.code, 2)


class StopCellTests(unittest.TestCase):
    def test_unavailable_outcome(self):
        from model_protocol_lab.experiments import _stop_cell

        cell = _stop_cell(
            QuirkBackend(),
            quirk_fixture(),
            (5,),
            StopOutcome(False, False, (10,), "decode_unavailable"),
        )
        self.assertEqual(cell, {"available": False, "detail": "decode_unavailable"})

    def test_answer_decode_failure(self):
        from model_protocol_lab.experiments import _stop_cell

        class BadAnswerDecode(QuirkBackend):
            def decode(self, token_ids):
                if tuple(token_ids) == (10, 11, 20):
                    raise RuntimeError("synthetic")
                return super().decode(token_ids)

        cell = _stop_cell(
            BadAnswerDecode(),
            quirk_fixture(),
            (5,),
            StopOutcome(True, True, (10, 11, 20, 5), "fired_at_step_4"),
        )
        self.assertEqual(cell, {"available": False, "detail": "answer_decode_unavailable"})

    def test_answer_decode_non_string(self):
        from model_protocol_lab.experiments import _stop_cell

        class WeirdDecode(QuirkBackend):
            def decode(self, token_ids):
                if tuple(token_ids) == (10, 11, 20):
                    return 12345
                return super().decode(token_ids)

        cell = _stop_cell(
            WeirdDecode(),
            quirk_fixture(),
            (5,),
            StopOutcome(True, True, (10, 11, 20, 5), "fired_at_step_4"),
        )
        self.assertEqual(cell, {"available": False, "detail": "answer_decode_unavailable"})

    def test_passing_cell(self):
        from model_protocol_lab.experiments import _stop_cell

        cell = _stop_cell(
            QuirkBackend(),
            quirk_fixture(),
            (5,),
            StopOutcome(True, True, (10, 11, 20, 5), "fired_at_step_4"),
        )
        self.assertTrue(cell["available"])
        self.assertTrue(cell["pass"])
        self.assertTrue(cell["fired"])


class EmulationBranchTests(unittest.TestCase):
    def test_hf_match_rejects_non_string_last_token(self):
        from model_protocol_lab.backends import _hf_overhang_match

        self.assertFalse(_hf_overhang_match(["a", 123], "a"))

    def test_hf_match_breaks_on_non_string_token(self):
        from model_protocol_lab.backends import _hf_overhang_match

        self.assertFalse(_hf_overhang_match(["x", 123, "b"], "ab"))

    def test_hf_match_breaks_on_suffix_mismatch(self):
        from model_protocol_lab.backends import _hf_overhang_match

        self.assertFalse(_hf_overhang_match(["q", "b"], "ab"))

    def test_hf_emulation_rejects_non_string_token_text(self):
        backend = FakeBackend(decode_script=["a", 123])
        outcome = emulate_hf_stop_string(backend, (1,), (2,), "a")
        self.assertFalse(outcome.available)
        self.assertEqual(outcome.detail, "token_text_unavailable")

    def test_vllm_emulation_rejects_non_string_decode(self):
        backend = FakeBackend(decode_script=[123])
        outcome = emulate_vllm_holdback(backend, (1,), "a")
        self.assertFalse(outcome.available)
        self.assertEqual(outcome.detail, "decode_unavailable")

    def test_matrix_with_empty_boundary_text(self):
        class EmptyBoundaryText(QuirkBackend):
            def decode(self, token_ids):
                if tuple(token_ids) == (5,):
                    return ""
                return super().decode(token_ids)

        rows = stop_matrix(
            EmptyBoundaryText(),
            "quirk",
            (quirk_fixture(),),
            (quirk_observation(),),
            boundary=(5,),
            eos_id=99,
        )
        (row,) = rows
        self.assertFalse(row["cells"]["hf_stop_string"]["available"])
        self.assertFalse(row["cells"]["vllm_holdback"]["available"])

    def test_matrix_with_non_int_output_ids(self):
        rows = stop_matrix(
            QuirkBackend(),
            "quirk",
            (quirk_fixture(),),
            (quirk_observation(output_ids=("a",), exact_match=False, boundary_found=False),),
            boundary=(5,),
        )
        (row,) = rows
        for semantic in ("hf_stop_string", "vllm_holdback", "eos_appended"):
            cell = row["cells"][semantic]
            self.assertFalse(cell["available"])
            self.assertEqual(cell["detail"], "observation_incomplete")


@unittest.skipUnless(HAS_HF, "the 'hf' extra is not installed")
class DoctorBranchTests(unittest.TestCase):
    def test_portability_check_rejects_non_iterable_ids(self):
        check = _portability_check("/nonexistent", 123, object())
        self.assertEqual(check["status"], "skipped")
        self.assertEqual(check["detail"], "invalid boundary ids")

    def test_portability_check_rejects_empty_boundary_text(self):
        class EmptyDecode:
            def decode(self, ids):
                return ""

        check = _portability_check("/nonexistent", (5,), EmptyDecode())
        self.assertEqual(check["status"], "skipped")
        self.assertEqual(check["detail"], "boundary_decode_failed")

    def test_conformance_accepts_dict_bos_token(self):
        with TemporaryDirectory() as tmp:
            root = write_template_model_dir(Path(tmp), template=LLAMA3_TEMPLATE)
            config = json.loads((root / "tokenizer_config.json").read_text(encoding="utf-8"))
            config["bos_token"] = {"content": "<|begin_of_text|>"}
            (root / "tokenizer_config.json").write_text(json.dumps(config), encoding="utf-8")
            outcome = template_conformance(tmp, (CATALOG["llama3"],), (llama_case("hi"),))
        self.assertEqual(outcome["status"], "ok")
        self.assertEqual(outcome["diffs"][0]["diff"], [])

    def test_conformance_without_bos_token_config(self):
        with TemporaryDirectory() as tmp:
            root = write_template_model_dir(Path(tmp), template=LLAMA3_TEMPLATE)
            config = json.loads((root / "tokenizer_config.json").read_text(encoding="utf-8"))
            del config["bos_token"]
            (root / "tokenizer_config.json").write_text(json.dumps(config), encoding="utf-8")
            outcome = template_conformance(tmp, (CATALOG["llama3"],), (llama_case("hi"),))
        # bos_token is unset so the template renders without it; the check
        # still runs to completion instead of crashing.
        self.assertEqual(outcome["status"], "ok")
        self.assertEqual(len(outcome["diffs"]), 1)


class BoundaryTextMatchBranchTests(unittest.TestCase):
    def run_evaluate(self, decode_script):
        backend = FakeBackend(
            generation=Generation(ids("ok|"), "limit"), decode_script=decode_script
        )
        report = backend_evaluate(backend)
        return report.observations[0].boundary_text_match

    def test_non_string_boundary_text_is_none(self):
        # answer decode, boundary decode (non-str), retained decode
        self.assertIsNone(self.run_evaluate(["ok|", 123, "ok|"]))

    def test_retained_decode_failure_is_none(self):
        self.assertIsNone(self.run_evaluate(["ok|", "|", RuntimeError("synthetic")]))

    def test_retained_decode_non_string_is_none(self):
        self.assertIsNone(self.run_evaluate(["ok|", "|", 456]))


def backend_evaluate(backend):
    from model_protocol_lab.core import evaluate

    return evaluate(backend, BASE, (case(),), boundary=BOUNDARY, max_new_tokens=3, seed=0)


class ReportingBranchTests(unittest.TestCase):
    MODES = [
        "no-boundary-in-N",
        "boundary-text-mismatch",
        "eos-without-boundary",
        "held-out-divergence",
        "backend-error",
    ]

    def taxonomy_result(self, decision, per_candidate, held_out=None):
        return {
            "decision": decision,
            "failure_taxonomy": {
                "modes": list(self.MODES),
                "per_candidate": per_candidate,
                "held_out": held_out,
                "eos_id_considered": False,
            },
        }

    def zero_counts(self):
        return {
            "no-boundary-in-N": 0,
            "boundary-text-mismatch": 0,
            "eos-without-boundary": 0,
            "held-out-divergence": 0,
            "backend-error": 0,
        }

    def test_cancelled_note_when_no_observations(self):
        html = render_html(self.taxonomy_result("cancelled", {}))
        self.assertIn("did not complete", html)

    def test_tie_note_when_no_failures(self):
        counts = {"a": self.zero_counts(), "b": self.zero_counts()}
        html = render_html(self.taxonomy_result("ambiguous", counts))
        self.assertIn("no failing observations recorded", html)

    def test_held_out_counts_cited(self):
        counts = {"a": self.zero_counts()}
        held_out = dict(self.zero_counts(), **{"held-out-divergence": 2})
        html = render_html(self.taxonomy_result("held_out_failed", counts, held_out))
        self.assertIn("held-out-divergence \u00d72", html)

    def test_junk_taxonomy_shapes_ignored(self):
        result = self.taxonomy_result("no_candidate", {"a": "junk"})
        result["failure_taxonomy"]["per_candidate"] = "junk"
        html = render_html(result)
        self.assertIn("no failing observations recorded", html)

    def test_non_dict_counts_ignored(self):
        html = render_html(self.taxonomy_result("no_candidate", {"a": "junk"}))
        self.assertIn("no failing observations recorded", html)

    def test_hostile_names_neutralized_in_new_sections(self):
        # Hostile candidate/fixture names must render escaped in every new
        # report section (conformance, stop matrix, sensitivity sweep).
        evil = "<script>alert(1)</script><img src=x onerror=alert(2)>"
        result = {
            "decision": "no_candidate",
            "search": [{"format": {"name": evil}, "summary": {"rate": 0.0}}],
            "failure_taxonomy": {
                "modes": list(self.MODES),
                "per_candidate": {evil: dict(self.zero_counts(), **{"no-boundary-in-N": 1})},
                "held_out": None,
                "eos_id_considered": False,
            },
            "template_conformance": {
                "status": "ok",
                "diffs": [
                    {
                        "format": evil,
                        "fixture": evil,
                        "status": "ok",
                        "diff": [
                            {
                                "hand_span": [0, 1],
                                "template_span": [0, 1],
                                "hand_ids": [evil],
                                "template_ids": [1],
                            }
                        ],
                        "hand_tokens": 1,
                        "template_tokens": 1,
                    }
                ],
            },
            "stop_matrix": {
                "candidate": evil,
                "semantics": [
                    {"id": "native", "label": "Native (lab)", "description": "d"},
                    {"id": "hf_stop_string", "label": evil, "description": evil},
                ],
                "search": [
                    {
                        "fixture": evil,
                        "native_pass": False,
                        "cells": {
                            "hf_stop_string": {
                                "available": True,
                                "pass": False,
                                "fired": True,
                                "detail": evil,
                            }
                        },
                        "flips": ["hf_stop_string"],
                    }
                ],
                "held_out": [],
            },
            "sensitivity_sweep": {
                "native_winner": evil,
                "native_decision": "no_candidate",
                "max_drop_bytes": 1,
                "depths": [
                    {
                        "drop_bytes": 1,
                        "decision": "no_candidate",
                        "selected_candidate": evil,
                        "winner_changed": True,
                    }
                ],
                "flips_at": [1],
            },
        }
        html = render_html(result)
        self.assertNotIn("<script>", html)
        self.assertNotIn("<img", html)
        self.assertNotIn(evil, html)
        # The escaped form is present: neutralization, not deletion.
        self.assertIn("&lt;script&gt;", html)


class TaxonomyBranchTests(unittest.TestCase):
    def test_junk_result_shapes_ignored(self):
        taxonomy = failure_taxonomy(
            {
                "decision": "held_out_failed",
                "search": [
                    "junk",
                    {"format": "junk"},
                    {"format": {"name": 123}},
                    {"format": {"name": "x"}},
                    {"format": {"name": "y"}, "observations": "junk"},
                ],
                "held_out": {"observations": "junk"},
            },
            None,
        )
        self.assertEqual(set(taxonomy["per_candidate"]), {"x", "y"})
        self.assertEqual(sum(taxonomy["per_candidate"]["x"].values()), 0)
        self.assertEqual(sum(taxonomy["held_out"].values()), 0)


class BackendReadyTests(unittest.TestCase):
    def test_uninitialized_backend_fails_closed(self):
        from model_protocol_lab.backends import LocalHFBackend

        backend = LocalHFBackend.__new__(LocalHFBackend)
        backend._tokenizer = None
        backend._model = None
        with self.assertRaises(RuntimeError):
            backend.encode("x")


@unittest.skipUnless(HAS_HF, "the 'hf' extra is not installed")
class PortabilityWiringTests(unittest.TestCase):
    def test_secondary_encode_failure_skips_gracefully(self):
        from unittest.mock import patch

        with TemporaryDirectory() as tmp:
            primary = write_wordlevel_model_dir(
                Path(tmp) / "primary", [f"tok{i}" for i in range(15)]
            )
            reference = write_wordlevel_model_dir(
                Path(tmp) / "reference", [f"tok{i}" for i in range(15)]
            )
            with patch(
                "model_protocol_lab.doctor.check_boundary_portability",
                return_value={
                    "portable": False,
                    "primary_ids": [5],
                    "secondary_ids": None,
                    "detail": "secondary_encode_failed",
                },
            ):
                report = examine_model(primary, boundary_ids=(5,), reference_model_path=reference)
        check = report["model"]["checks"]["boundary_portability"]
        self.assertEqual(check["status"], "skipped")
        self.assertEqual(report["verdict"], "ok")


class CLIRuntimeErrorTests(unittest.TestCase):
    def test_generic_runtime_error_message(self):
        from unittest.mock import patch

        from model_protocol_lab.cli import main

        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "report"
            with patch(
                "model_protocol_lab.reporting.prepare_report_dir",
                side_effect=RuntimeError("synthetic"),
            ):
                code = main(
                    [
                        "run",
                        "--model",
                        "m",
                        "--dataset",
                        "d",
                        "--family",
                        "chatml",
                        "--boundary-ids",
                        "1",
                        "--output",
                        str(output),
                    ]
                )
        self.assertEqual(code, 1)

    def test_conformance_junk_entries_ignored(self):
        html = render_html(
            {
                "decision": "no_candidate",
                "template_conformance": {
                    "status": "ok",
                    "diffs": [
                        None,
                        {"status": "unavailable", "format": "f", "fixture": "x"},
                        {
                            "status": "ok",
                            "format": "f",
                            "fixture": "y",
                            "hand_tokens": 1,
                            "template_tokens": 2,
                            "diff": [None],
                        },
                    ],
                },
            }
        )
        self.assertIn("Template conformance", html)
        self.assertIn("unavailable", html)

    def test_conformance_no_pairs_checked(self):
        html = render_html(
            {
                "decision": "no_candidate",
                "template_conformance": {"status": "ok", "diffs": []},
            }
        )
        self.assertIn("No format/fixture pairs were checked.", html)

    def test_matrix_without_semantics_renders_nothing(self):
        html = render_html({"decision": "no_candidate", "stop_matrix": {"semantics": []}})
        self.assertNotIn("Stop-behavior isolation matrix", html)
        html = render_html({"decision": "no_candidate", "stop_matrix": {"semantics": "junk"}})
        self.assertNotIn("Stop-behavior isolation matrix", html)

    def test_matrix_junk_rows_and_cells(self):
        rows = stop_matrix(
            QuirkBackend(),
            "quirk",
            (quirk_fixture(),),
            (quirk_observation(),),
            boundary=(5,),
            eos_id=99,
        )
        rows = [None, *rows]
        rows[1]["cells"]["hf_stop_string"] = None
        html = render_html(
            {
                "decision": "no_candidate",
                "stop_matrix": {
                    "candidate": "quirk",
                    "semantics": [
                        {"id": "native", "label": "Native (lab)", "description": "d"},
                        {"id": "hf_stop_string", "label": "HF", "description": "d"},
                        {"id": "vllm_holdback", "label": "vLLM", "description": "d"},
                        {"id": "eos_appended", "label": "EOS", "description": "d"},
                    ],
                    "search": rows,
                    "held_out": "junk",
                },
            }
        )
        self.assertIn("Stop-behavior isolation matrix", html)
        self.assertIn("n/a", html)
        self.assertIn("No rows for this stage.", html)

    def test_sweep_junk_depths(self):
        html = render_html(
            {
                "decision": "no_candidate",
                "sensitivity_sweep": {
                    "native_winner": "w",
                    "native_decision": "d",
                    "flips_at": [],
                    "depths": [None],
                },
            }
        )
        self.assertIn("No depths were evaluated.", html)


if __name__ == "__main__":
    unittest.main()
