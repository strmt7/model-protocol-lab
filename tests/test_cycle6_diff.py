"""Cycle 6: `mpl diff-backends` -- differential testing of stop semantics.

Covers ``model_protocol_lab/diffbackends.py`` and the CLI wiring:

- kept-vs-stripped divergence detected (HF always keeps the stop string;
  vLLM's default ``include_stop_str_in_output=False`` strips it);
- firing-step agreement across all semantics -> verdict ``"agree"``;
- ``min_tokens`` gating divergence (vLLM PR #22014);
- EOS precedence divergence (the SGLang #28802 class);
- harness-normalization reliance (cf. lm-eval's
  ``postprocess_generated_text`` re-strip);
- strict input validation;
- determinism (repeat runs identical) and JSON-serializability;
- CLI wiring: parser flags, fast-fail validation, a full run against a
  hand-written tiny ``tokenizer.json`` (no model weights, no network).

All backends are fakes decoding through fixed per-token text tables; no
torch, no network, no model weights.
"""

import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from model_protocol_lab.cli import build_parser, main
from model_protocol_lab.diffbackends import (
    EOS_APPENDED,
    HF_STOP_STRING,
    VLLM_HOLDBACK_KEEP,
    VLLM_HOLDBACK_STRIP,
    diff_backends,
    run_diff_backends,
)


class TableBackend:
    """Fake backend decoding token ids through a fixed per-token text table.

    Whole-sequence decodes concatenate the per-token texts, so the vLLM
    emulation's per-prefix decodes and the HF emulation's per-token decodes
    agree here -- these tests pin the stop-semantics diff, not decode
    quirks (those live in test_round3's QuirkBackend).
    """

    fingerprint = "cycle6-table-backend-not-a-model"

    def __init__(self, texts: dict[int, str]) -> None:
        self._texts = dict(texts)

    def decode(self, token_ids: tuple[int, ...]) -> str:
        """Decode."""
        return "".join(self._texts[token] for token in token_ids)


class DiffBackendsTests(unittest.TestCase):
    """Diff backends tests."""

    def setUp(self) -> None:
        """Set up the test fixture."""
        # 1->"a", 2->"b", 3->"STOP", 4->"c": stop "STOP" completes at step 3.
        self.backend = TableBackend({1: "a", 2: "b", 3: "STOP", 4: "c"})
        self.output_ids = (1, 2, 3, 4)

    def _run(self, **kwargs: Any) -> dict[str, Any]:
        """Run."""
        base: dict[str, Any] = {
            "prompt_ids": (),
            "output_ids": self.output_ids,
            "stop_texts": ("STOP",),
            "backend": self.backend,
        }
        base.update(kwargs)
        return diff_backends(**base)

    def test_kept_vs_stripped_divergence_detected(self) -> None:
        """Test kept vs stripped divergence detected."""
        result = self._run()
        classes = [finding["class"] for finding in result["divergences"]]
        self.assertIn("KEPT_VS_STRIPPED", classes)
        self.assertEqual(result["verdict"], "diverge")
        record = result["cells"][0]
        self.assertEqual(record["stop_text"], "STOP")
        hf = record["semantics"][HF_STOP_STRING]
        self.assertTrue(hf["available"])
        self.assertTrue(hf["fired"])
        # Firing step is parsed from the emulation's fixed detail string.
        self.assertEqual(hf["firing_step"], 3)
        self.assertEqual(hf["disposition"], "kept")
        self.assertIsNone(hf["returned_text"])
        self.assertEqual(hf["truncated_ids"], [1, 2, 3])
        strip = record["semantics"][VLLM_HOLDBACK_STRIP]
        self.assertEqual(strip["firing_step"], 3)
        self.assertEqual(strip["disposition"], "stripped")
        self.assertEqual(strip["truncated_ids"], [1, 2, 3])
        # The completing token stays in token_ids (vLLM's token/text
        # overshoot) but the returned text drops the stop string.
        self.assertEqual(strip["returned_text"], "ab")
        keep = record["semantics"][VLLM_HOLDBACK_KEEP]
        self.assertEqual(keep["firing_step"], 3)
        self.assertEqual(keep["disposition"], "kept")
        self.assertEqual(keep["returned_text"], "abSTOP")
        # Same firing step everywhere: no firing-step disagreement.
        self.assertNotIn("FIRING_STEP_DIFFERS", classes)

    def test_firing_step_agreement_verdict_agree(self) -> None:
        """Test firing step agreement verdict agree."""
        result = self._run(stop_texts=("ZZZ",))
        self.assertEqual(result["divergences"], [])
        self.assertEqual(result["verdict"], "agree")
        for name, cell in result["cells"][0]["semantics"].items():
            if name == EOS_APPENDED:
                continue
            self.assertTrue(cell["available"])
            self.assertFalse(cell["fired"])
            self.assertIsNone(cell["firing_step"])
            self.assertEqual(cell["truncated_ids"], [1, 2, 3, 4])

    def test_min_tokens_gating_divergence(self) -> None:
        """Test min tokens gating divergence."""
        backend = TableBackend({1: "x", 2: "STOP", 3: "y"})
        result = diff_backends(
            prompt_ids=(),
            output_ids=(1, 2, 3),
            stop_texts=("STOP",),
            min_tokens=3,
            backend=backend,
        )
        classes = [finding["class"] for finding in result["divergences"]]
        self.assertIn("MIN_TOKENS_GATED", classes)
        # The gated vLLM cell disagrees with HF about the firing step too.
        self.assertIn("FIRING_STEP_DIFFERS", classes)
        self.assertEqual(result["verdict"], "diverge")
        # skipcq: PTC-W0063 - in a test, next() raising StopIteration is the intended failure
        gated = next(
            finding for finding in result["divergences"] if finding["class"] == "MIN_TOKENS_GATED"
        )
        self.assertIn("step 2", gated["detail"])
        self.assertEqual(gated["stop_text"], "STOP")
        strip = result["cells"][0]["semantics"][VLLM_HOLDBACK_STRIP]
        self.assertTrue(strip["available"])
        self.assertFalse(strip["fired"])
        self.assertIsNone(strip["firing_step"])
        self.assertEqual(strip["truncated_ids"], [1, 2, 3])

    def test_min_tokens_zero_is_not_gating(self) -> None:
        """Test min tokens zero is not gating."""
        result = self._run(min_tokens=0)
        classes = [finding["class"] for finding in result["divergences"]]
        self.assertNotIn("MIN_TOKENS_GATED", classes)

    def test_eos_precedence_divergence(self) -> None:
        """Test eos precedence divergence."""
        # 2 is EOS: the eos_appended cell fires at step 2, before the
        # stop-string cells fire at step 3 (the SGLang #28802 class).
        backend = TableBackend({1: "a", 2: "<eos>", 3: "STOP"})
        result = diff_backends(
            prompt_ids=(),
            output_ids=(1, 2, 3),
            stop_texts=("STOP",),
            eos_id=2,
            backend=backend,
        )
        classes = [finding["class"] for finding in result["divergences"]]
        self.assertIn("EOS_PRECEDENCE", classes)
        self.assertEqual(result["verdict"], "diverge")
        eos = result["cells"][0]["semantics"][EOS_APPENDED]
        self.assertTrue(eos["available"])
        self.assertTrue(eos["fired"])
        self.assertEqual(eos["firing_step"], 2)
        self.assertEqual(eos["truncated_ids"], [1])
        self.assertIsNone(eos["disposition"])
        self.assertIsNone(eos["returned_text"])

    def test_eos_without_eos_id_is_unavailable(self) -> None:
        """Test eos without eos id is unavailable."""
        result = self._run()
        eos = result["cells"][0]["semantics"][EOS_APPENDED]
        self.assertFalse(eos["available"])
        self.assertFalse(eos["fired"])
        classes = [finding["class"] for finding in result["divergences"]]
        self.assertNotIn("EOS_PRECEDENCE", classes)

    def test_harness_normalization_relied(self) -> None:
        """Test harness normalization relied."""
        # Kept-disposition cells keep the stop string by definition and are
        # not flagged; the stripped cell's returned text excludes it, so no
        # failed strip is reported in the normal case.
        result = self._run()
        relied = [
            finding
            for finding in result["divergences"]
            if finding["class"] == "HARNESS_NORMALIZATION_RELIED"
        ]
        self.assertEqual(relied, [])

    def test_harness_normalization_relied_on_failed_strip(self) -> None:
        """Test harness normalization relied on failed strip."""
        # A stop that completes during the min_tokens gate is invisible to
        # the check; when a second stop fires after the gate, the stripped
        # cell's returned text still contains the earlier (gated) stop
        # string -- the strip failed to remove every occurrence, so correct
        # scoring would depend on harness-side re-stripping.
        backend = TableBackend({1: "STOP", 2: "x", 3: "STOP", 4: "y"})
        result = diff_backends(
            prompt_ids=(),
            output_ids=(1, 2, 3, 4),
            stop_texts=("STOP",),
            min_tokens=2,
            backend=backend,
        )
        relied = [
            finding
            for finding in result["divergences"]
            if finding["class"] == "HARNESS_NORMALIZATION_RELIED"
        ]
        self.assertEqual(len(relied), 1)
        self.assertTrue(relied[0]["detail"].startswith(VLLM_HOLDBACK_STRIP))

    def test_duplicate_stop_texts_deduped(self) -> None:
        """Test duplicate stop texts deduped."""
        result = self._run(stop_texts=("STOP", "STOP", "ZZZ"))
        self.assertEqual(result["stop_texts"], ["STOP", "ZZZ"])
        self.assertEqual(len(result["cells"]), 2)

    def test_oversize_token_sequences_rejected(self) -> None:
        """Test oversize token sequences rejected."""
        with self.assertRaises(ValueError) as ctx:
            self._run(output_ids=tuple([1] * 131073))
        self.assertEqual(str(ctx.exception), "output_ids_too_long")

    def test_invalid_inputs_rejected(self) -> None:
        """Test invalid inputs rejected."""
        good: dict[str, Any] = {
            "prompt_ids": (),
            "output_ids": (1, 2, 3),
            "stop_texts": ("STOP",),
            "backend": self.backend,
        }
        bad_cases: list[dict[str, Any]] = [
            {"prompt_ids": [1, 2]},  # list, not tuple
            {"prompt_ids": (1, -2)},
            {"prompt_ids": (1, True)},  # bool is not an int here
            {"prompt_ids": (1, 2**31)},
            {"output_ids": "123"},
            {"output_ids": (1, 2.0)},
            {"stop_texts": ()},
            {"stop_texts": ("STOP", "")},
            {"stop_texts": ("STOP", 7)},
            {"stop_texts": "STOP"},
            {"eos_id": -1},
            {"eos_id": True},
            {"min_tokens": -1},
            {"min_tokens": "3"},
            {"backend": object()},
        ]

        class NoDecode:
            """No decode."""

            fingerprint = "no-decode"

        bad_cases.append({"backend": NoDecode()})
        for bad in bad_cases:
            with self.subTest(bad=bad):
                kwargs = dict(good)
                kwargs.update(bad)
                with self.assertRaises(ValueError):
                    diff_backends(**kwargs)

    def test_determinism_and_json_round_trip(self) -> None:
        """Test determinism and json round trip."""
        kwargs: dict[str, Any] = {
            "prompt_ids": (1,),
            "output_ids": self.output_ids,
            "stop_texts": ("STOP", "ZZZ"),
            "eos_id": 99,
            "min_tokens": 1,
            "backend": self.backend,
        }
        first = diff_backends(**kwargs)
        second = diff_backends(**kwargs)
        self.assertEqual(first, second)
        # The result is JSON-serializable by contract.
        self.assertEqual(json.loads(json.dumps(first)), first)
        self.assertEqual(first["prompt_tokens"], 1)
        self.assertEqual(first["output_tokens"], 4)
        self.assertEqual(first["min_tokens"], 1)
        self.assertEqual(first["eos_id"], 99)


class DiffBackendsCliTests(unittest.TestCase):
    """Diff backends cli tests."""

    @staticmethod
    def _args(extra: list[str]) -> Any:
        """Args."""
        parser = build_parser()
        return parser, parser.parse_args(
            [
                "diff-backends",
                "--model",
                "m",
                "--prompt-text",
                "p",
                "--output-text",
                "o",
                "--stops",
                "a,b",
                *extra,
            ]
        )

    def test_parser_registers_diff_backends(self) -> None:
        """Test parser registers diff backends."""
        _parser, args = self._args([])
        self.assertEqual(args.command, "diff-backends")
        self.assertEqual(str(args.model), "m")
        self.assertEqual(args.prompt_text, "p")
        self.assertEqual(args.output_text, "o")
        self.assertEqual(args.stops, "a,b")
        self.assertIsNone(args.eos_id)
        self.assertEqual(args.min_tokens, 0)

    def test_negative_min_tokens_exits_two(self) -> None:
        """Test negative min tokens exits two."""
        parser, args = self._args(["--min-tokens", "-1"])
        with self.assertRaises(SystemExit) as ctx:
            run_diff_backends(args, parser)
        self.assertEqual(ctx.exception.code, 2)

    def test_negative_eos_id_exits_two(self) -> None:
        """Test negative eos id exits two."""
        parser, args = self._args(["--eos-id", "-2"])
        with self.assertRaises(SystemExit) as ctx:
            run_diff_backends(args, parser)
        self.assertEqual(ctx.exception.code, 2)

    def test_empty_stop_part_exits_two(self) -> None:
        """Test empty stop part exits two."""
        parser = build_parser()
        args = parser.parse_args(
            [
                "diff-backends",
                "--model",
                "m",
                "--prompt-text",
                "p",
                "--output-text",
                "o",
                "--stops",
                "a,,b",
            ]
        )
        with self.assertRaises(SystemExit) as ctx:
            run_diff_backends(args, parser)
        self.assertEqual(ctx.exception.code, 2)

    def test_missing_checkpoint_returns_one(self) -> None:
        """Test missing checkpoint returns one."""
        # Never loads weights: a missing directory fails closed with a
        # fixed message and exit 1.
        with TemporaryDirectory() as tmp:
            code = main(
                [
                    "diff-backends",
                    "--model",
                    str(Path(tmp) / "missing"),
                    "--prompt-text",
                    "p",
                    "--output-text",
                    "o",
                    "--stops",
                    "STOP",
                ]
            )
        self.assertEqual(code, 1)

    def test_full_run_against_tiny_tokenizer(self) -> None:
        """Test full run against tiny tokenizer."""
        # End-to-end through main() with a hand-written tokenizer.json:
        # tokenizer only, no weights, no network.
        tokenizer_doc: dict[str, Any] = {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": [],
            "normalizer": None,
            "pre_tokenizer": {"type": "WhitespaceSplit"},
            "post_processor": None,
            "decoder": None,
            "model": {
                "type": "WordLevel",
                "vocab": {"a": 0, "b": 1, "STOP": 2, "c": 3, "[UNK]": 4},
                "unk_token": "[UNK]",
            },
        }
        with TemporaryDirectory() as tmp:
            (Path(tmp) / "tokenizer.json").write_text(json.dumps(tokenizer_doc), encoding="utf-8")
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = main(
                    [
                        "diff-backends",
                        "--model",
                        tmp,
                        "--prompt-text",
                        "a b",
                        "--output-text",
                        "a b STOP c",
                        "--stops",
                        "STOP",
                    ]
                )
        self.assertEqual(code, 0)
        result = json.loads(buffer.getvalue())
        self.assertEqual(result["verdict"], "diverge")
        classes = [finding["class"] for finding in result["divergences"]]
        self.assertIn("KEPT_VS_STRIPPED", classes)
        self.assertEqual(result["prompt_tokens"], 2)
        self.assertEqual(result["output_tokens"], 4)
        self.assertEqual(result["stop_texts"], ["STOP"])
        record = result["cells"][0]
        self.assertEqual(record["semantics"][VLLM_HOLDBACK_STRIP]["returned_text"], "a b ")


if __name__ == "__main__":
    unittest.main()
