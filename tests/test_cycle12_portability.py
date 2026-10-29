"""Tests for the static template<->tokenizer portability matrix (Cycle 12).

Covers ``core.template_tokenizer_compatibility`` (the per-format probe),
``doctor.portability_matrix`` (the catalog-wide matrix wiring),
``doctor.render_portability_table``, hostile second-checkpoint handling,
determinism, and the ``doctor --portability-matrix`` CLI flag.

Research basis: docs/RESEARCH_PORTABILITY_2026-10-15.md. The load-bearing
property under test is the asymmetry: "incompatible" is a static exclusion
answer ("the format cannot be this checkpoint's native format"), while
"compatible" predicts NOTHING about experimental success -- the output must
never claim a transfer of the selection verdict and must never emit a
portability score.
"""

import importlib.util
import io
import json
import os
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from model_protocol_lab.core import template_tokenizer_compatibility
from model_protocol_lab.doctor import (
    _PORTABILITY_ASYMMETRY,
    _secondary_registered,
    doctor,
    portability_matrix,
    render_portability_table,
)

HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None
HAS_TOKENIZERS = importlib.util.find_spec("tokenizers") is not None

CHATML_TOKENS = ("<|im_start|>", "<|im_end|>")

#: Fixed vocabulary the probe may report; anything else is a bug.
REASONS = frozenset(
    {
        "",
        "unregistered",
        "fragments",
        "id_mismatch",
        "encode_failed",
        "encode_invalid",
        "empty_inventory",
    }
)
VERDICTS = frozenset({"compatible", "incompatible", "unknown"})


def write_tokenizer_checkpoint(root: Path, added: tuple[str, ...] = ()) -> Path:
    """Minimal second-checkpoint dir: a tokenizer.json with added tokens."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    root.mkdir(parents=True, exist_ok=True)
    vocab = {"a": 0, "b": 1, "[UNK]": 2}
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    if added:
        tokenizer.add_special_tokens(list(added))
    tokenizer.save(str(root / "tokenizer.json"))
    return root


def write_primary_model(tmp: str) -> Path:
    """Healthy primary model dir for doctor(), via the test_doctor helper."""
    from tests.test_doctor import write_model_dir

    primary = Path(tmp) / "primary"
    primary.mkdir()
    return write_model_dir(primary)


class CompatibilityProbeTests(unittest.TestCase):
    """core.template_tokenizer_compatibility with stub encode callables."""

    def test_compatible_when_registered_atomic_id_consistent(self):
        """Test compatible when registered atomic id consistent."""
        result = template_tokenizer_compatibility(
            "chatml",
            CHATML_TOKENS,
            {"<|im_start|>": 3, "<|im_end|>": 4},
            lambda text: {"<|im_start|>": (3,), "<|im_end|>": (4,)}[text],
        )
        self.assertEqual(result["format"], "chatml")
        self.assertEqual(result["verdict"], "compatible")
        self.assertEqual(result["reason"], "")
        self.assertTrue(all(row["status"] == "ok" for row in result["tokens"]))
        self.assertTrue(all(row["id_match"] for row in result["tokens"]))
        self.assertIn("predicts nothing", result["detail"])

    def test_incompatible_when_token_missing(self):
        """Test incompatible when token missing."""
        result = template_tokenizer_compatibility("llama3", ("<|eot_id|>",), {}, lambda text: (9,))
        self.assertEqual(result["verdict"], "incompatible")
        self.assertEqual(result["reason"], "unregistered")
        self.assertIn("<|eot_id|>", result["detail"])
        row = result["tokens"][0]
        self.assertFalse(row["registered"])
        self.assertIsNone(row["registered_id"])
        self.assertTrue(row["single_token"])

    def test_incompatible_when_token_fragments(self):
        """Test incompatible when token fragments."""
        result = template_tokenizer_compatibility(
            "chatml", ("<|im_start|>",), {"<|im_start|>": 3}, lambda text: (1, 2)
        )
        self.assertEqual(result["verdict"], "incompatible")
        self.assertEqual(result["reason"], "fragments")
        row = result["tokens"][0]
        self.assertTrue(row["registered"])
        self.assertFalse(row["single_token"])
        self.assertIsNone(row["id_match"])

    def test_incompatible_on_id_mismatch(self):
        """Test incompatible on id mismatch."""
        result = template_tokenizer_compatibility(
            "chatml", ("<|im_start|>",), {"<|im_start|>": 3}, lambda text: (7,)
        )
        self.assertEqual(result["verdict"], "incompatible")
        self.assertEqual(result["reason"], "id_mismatch")
        row = result["tokens"][0]
        self.assertTrue(row["registered"])
        self.assertTrue(row["single_token"])
        self.assertFalse(row["id_match"])
        self.assertEqual(row["encoded_id"], 7)
        self.assertEqual(row["registered_id"], 3)

    def test_unknown_on_empty_inventory(self):
        """Test unknown on empty inventory."""
        # A format with no token-shaped control literals cannot be judged
        # -- "unknown", never "compatible".
        result = template_tokenizer_compatibility("plain", (), {}, lambda text: (1,))
        self.assertEqual(result["verdict"], "unknown")
        self.assertEqual(result["reason"], "empty_inventory")
        self.assertEqual(result["tokens"], [])

    def test_unknown_when_encode_raises(self):
        """Test unknown when encode raises."""

        def hostile(text):
            """Hostile."""
            raise RuntimeError("boom")

        result = template_tokenizer_compatibility(
            "chatml", CHATML_TOKENS, {"<|im_start|>": 3, "<|im_end|>": 4}, hostile
        )
        self.assertEqual(result["verdict"], "unknown")
        self.assertEqual(result["reason"], "encode_failed")
        self.assertIn("cannot be judged", result["detail"])

    def test_unknown_when_encode_returns_garbage(self):
        """Test unknown when encode returns garbage."""
        for bad in (["not-a-tuple"], (1, -2), ()):
            with self.subTest(bad=bad):
                # skipcq: PYL-W0102 - bad=bad binds the loop value at def time by design.
                def encode(text, bad=bad):
                    """Encode."""
                    return bad

                result = template_tokenizer_compatibility(
                    "chatml", ("<|im_start|>",), {"<|im_start|>": 3}, encode
                )
                self.assertEqual(result["verdict"], "unknown")
                self.assertEqual(result["reason"], "encode_invalid")

    def test_first_failure_in_inventory_order_wins(self):
        """Test first failure in inventory order wins."""
        result = template_tokenizer_compatibility(
            "f",
            ("<a>", "<b>"),
            {"<a>": 5},
            lambda text: (1, 2) if text == "<a>" else (9,),
        )
        # "<a>" fragments, "<b>" is unregistered: inventory order decides.
        self.assertEqual(result["verdict"], "incompatible")
        self.assertEqual(result["reason"], "fragments")
        self.assertIn("<a>", result["detail"])

    def test_degraded_probe_beats_incompatible(self):
        """Test degraded probe beats incompatible."""

        def flaky(text):
            """Flaky."""
            if text == "<a>":
                raise RuntimeError("boom")
            return (9,)

        result = template_tokenizer_compatibility("f", ("<a>", "<b>"), {}, flaky)
        # A degraded probe cannot answer: "unknown", not "incompatible".
        self.assertEqual(result["verdict"], "unknown")
        self.assertEqual(result["reason"], "encode_failed")

    def test_token_row_schema(self):
        """Test token row schema."""
        result = template_tokenizer_compatibility("f", ("<t>",), {"<t>": 5}, lambda text: (5,))
        self.assertEqual(
            set(result),
            {"format", "verdict", "reason", "detail", "tokens"},
        )
        self.assertEqual(
            set(result["tokens"][0]),
            {
                "token",
                "registered",
                "registered_id",
                "single_token",
                "encoded_id",
                "id_match",
                "status",
            },
        )

    def test_rejects_bad_inputs(self):
        """Test rejects bad inputs."""

        def encode(text):
            """Encode."""
            return (1,)

        with self.assertRaises(ValueError):
            template_tokenizer_compatibility("", ("<a>",), {}, encode)
        with self.assertRaises(ValueError):
            template_tokenizer_compatibility("f", ["<a>"], {}, encode)
        with self.assertRaises(ValueError):
            template_tokenizer_compatibility("f", ("",), {}, encode)
        with self.assertRaises(ValueError):
            template_tokenizer_compatibility("f", ("<a>",), {"<a>": -1}, encode)
        with self.assertRaises(ValueError):
            template_tokenizer_compatibility("f", ("<a>",), {"<a>": True}, encode)
        with self.assertRaises(ValueError):
            template_tokenizer_compatibility("f", ("<a>",), [("<a>", 1)], encode)
        with self.assertRaises(ValueError):
            template_tokenizer_compatibility("f", ("<a>",), {}, None)


@unittest.skipUnless(HAS_TRANSFORMERS and HAS_TOKENIZERS, "hf extra not installed")
class PortabilityMatrixTests(unittest.TestCase):
    """Portability matrix tests."""

    @staticmethod
    def _chatml_section(tmp: str) -> dict:
        """Chatml section."""
        second = write_tokenizer_checkpoint(Path(tmp) / "second", CHATML_TOKENS)
        return portability_matrix(second)

    def test_matrix_covers_full_mutate_candidate_space(self):
        """Test matrix covers full mutate candidate space."""
        from model_protocol_lab.catalog import CATALOG, candidates

        expected = [
            fmt.name for family in sorted(CATALOG) for fmt in candidates(family, mutate=True)
        ]
        with TemporaryDirectory() as tmp:
            section = self._chatml_section(tmp)
            self.assertEqual(section["status"], "ok")
            self.assertEqual(
                [row["format"] for row in section["rows"]],
                expected,
            )

    def test_chatml_family_compatible_on_chatml_tokens(self):
        """Test chatml family compatible on chatml tokens."""
        with TemporaryDirectory() as tmp:
            section = self._chatml_section(tmp)
            by_name = {row["format"]: row for row in section["rows"]}
            for name in ("chatml", "chatml/prefix-newline", "chatml/prefix-trim"):
                with self.subTest(name=name):
                    self.assertEqual(by_name[name]["verdict"], "compatible")

    def test_foreign_tokens_yield_incompatible(self):
        """Test foreign tokens yield incompatible."""
        with TemporaryDirectory() as tmp:
            section = self._chatml_section(tmp)
            by_name = {row["format"]: row for row in section["rows"]}
            row = by_name["llama3"]
            self.assertEqual(row["verdict"], "incompatible")
            self.assertEqual(row["reason"], "unregistered")
            self.assertFalse(row["bos_skipped"])

    def test_bos_less_formats_skip_the_bos_requirement(self):
        """Test bos less formats skip the bos requirement."""
        from model_protocol_lab.catalog import CATALOG, candidates

        expected_skip = {
            fmt.name: fmt.bos == ""
            for family in sorted(CATALOG)
            for fmt in candidates(family, mutate=True)
        }
        # Sanity: the catalog really does contain BOS-less families, and
        # at least one format with a BOS, so the flag is exercised both ways.
        self.assertTrue(any(expected_skip.values()))
        self.assertTrue(any(not skip for skip in expected_skip.values()))
        with TemporaryDirectory() as tmp:
            section = self._chatml_section(tmp)
            for row in section["rows"]:
                with self.subTest(format=row["format"]):
                    self.assertEqual(row["bos_skipped"], expected_skip[row["format"]])
                    self.assertIn(row["reason"], REASONS)
                    self.assertNotIn("bos", row["reason"])
            by_name = {row["format"]: row for row in section["rows"]}
            # The skip is documentary, not a free pass: llama3/no-bos skips
            # the BOS token but still fails on its other missing tokens.
            row = by_name["llama3/no-bos"]
            self.assertTrue(row["bos_skipped"])
            self.assertEqual(row["verdict"], "incompatible")
            self.assertEqual(row["reason"], "unregistered")

    def test_token_less_checkpoint_is_all_incompatible(self):
        """Test token less checkpoint is all incompatible."""
        with TemporaryDirectory() as tmp:
            second = write_tokenizer_checkpoint(Path(tmp) / "second")
            section = portability_matrix(second)
            self.assertEqual(section["status"], "ok")
            verdicts = {row["verdict"] for row in section["rows"]}
            self.assertEqual(verdicts, {"incompatible"})

    def test_compatible_makes_no_transfer_claim(self):
        """Test compatible makes no transfer claim."""
        with TemporaryDirectory() as tmp:
            section = self._chatml_section(tmp)
            self.assertIn("predicts NOTHING", section["asymmetry"])
            self.assertIn("never emits a portability score", section["asymmetry"])
            forbidden = ("score", "prediction", "transfer", "will_succeed", "success")

            def check_keys(node, path="section"):
                """Check keys."""
                if isinstance(node, dict):
                    for key, value in node.items():
                        self.assertFalse(
                            any(word in key.lower() for word in forbidden),
                            f"forbidden key fragment in {path}.{key}",
                        )
                        check_keys(value, f"{path}.{key}")
                elif isinstance(node, list):
                    for index, value in enumerate(node):
                        check_keys(value, f"{path}[{index}]")

            check_keys(section)
            for row in section["rows"]:
                with self.subTest(format=row["format"]):
                    self.assertIn(row["verdict"], VERDICTS)
                    self.assertIn(row["reason"], REASONS)
            compatible_rows = [row for row in section["rows"] if row["verdict"] == "compatible"]
            self.assertTrue(compatible_rows)
            for row in compatible_rows:
                self.assertIn("predicts nothing", row["detail"])

    def test_section_and_row_schema(self):
        """Test section and row schema."""
        with TemporaryDirectory() as tmp:
            section = self._chatml_section(tmp)
            self.assertEqual(
                set(section),
                {"status", "detail", "second_checkpoint", "asymmetry", "rows"},
            )
            self.assertEqual(section["asymmetry"], _PORTABILITY_ASYMMETRY)
            for row in section["rows"]:
                with self.subTest(format=row["format"]):
                    self.assertEqual(
                        set(row),
                        {"format", "verdict", "reason", "detail", "tokens", "bos_skipped"},
                    )
                    self.assertTrue(row["tokens"])
                    for token_row in row["tokens"]:
                        self.assertEqual(
                            set(token_row),
                            {
                                "token",
                                "registered",
                                "registered_id",
                                "single_token",
                                "encoded_id",
                                "id_match",
                                "status",
                            },
                        )

    def test_matrix_is_deterministic(self):
        """Test matrix is deterministic."""
        with TemporaryDirectory() as tmp:
            second = write_tokenizer_checkpoint(Path(tmp) / "second", CHATML_TOKENS)
            first = json.dumps(portability_matrix(second), sort_keys=True)
            again = json.dumps(portability_matrix(second), sort_keys=True)
            self.assertEqual(first, again)

    def test_matrix_never_fails_the_doctor_verdict(self):
        """Test matrix never fails the doctor verdict."""
        with TemporaryDirectory() as tmp:
            primary = write_primary_model(tmp)
            second = write_tokenizer_checkpoint(Path(tmp) / "second")
            report = doctor(primary, portability_matrix_path=second)
            self.assertEqual(report["verdict"], "ok")
            section = report["portability_matrix"]
            self.assertEqual(section["status"], "ok")
            verdicts = {row["verdict"] for row in section["rows"]}
            self.assertEqual(verdicts, {"incompatible"})

    def test_doctor_omits_section_without_flag(self):
        """Test doctor omits section without flag."""
        with TemporaryDirectory() as tmp:
            primary = write_primary_model(tmp)
            report = doctor(primary)
            self.assertNotIn("portability_matrix", report)

    def test_unloadable_tokenizer_doc_is_sanitized(self):
        """Test unloadable tokenizer doc is sanitized."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "second"
            root.mkdir()
            (root / "tokenizer.json").write_text(
                json.dumps({"model": {"type": "BPE"}}), encoding="utf-8"
            )
            section = portability_matrix(root)
            self.assertEqual(section["status"], "unavailable")
            self.assertEqual(section["detail"], "tokenizer_load_failed")
            # The section carries no checkpoint-controlled bytes.
            self.assertNotIn("BPE", json.dumps(section["rows"]))


class HostileSecondCheckpointTests(unittest.TestCase):
    """Fixed-vocabulary, fail-closed handling of hostile second checkpoints."""

    def test_missing_directory_is_unavailable(self):
        """Test missing directory is unavailable."""
        section = portability_matrix("/definitely/not/here-12345")
        self.assertEqual(section["status"], "unavailable")
        self.assertEqual(section["detail"], "not_a_readable_directory")
        self.assertEqual(section["rows"], [])

    def test_file_path_is_unavailable(self):
        """Test file path is unavailable."""
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "file"
            target.write_text("x", encoding="utf-8")
            section = portability_matrix(target)
            self.assertEqual(section["detail"], "not_a_readable_directory")

    def test_oversized_tokenizer_is_rejected_without_reading(self):
        """Test oversized tokenizer is rejected without reading."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "second"
            root.mkdir()
            target = root / "tokenizer.json"
            target.write_bytes(b"{}")
            # Sparse: the 128 MB cap trips on stat, before any read -- a
            # real 128 MB+ read would hang or OOM the check.
            os.truncate(target, 128 * 1024 * 1024 + 1)
            section = portability_matrix(root)
            self.assertEqual(section["status"], "unavailable")
            self.assertEqual(section["detail"], "oversized_tokenizer_file")

    def test_malformed_tokenizer_json_is_sanitized(self):
        """Test malformed tokenizer json is sanitized."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "second"
            root.mkdir()
            marker = "HOSTILE_MARKER_BYTES_<|im_start|>"
            (root / "tokenizer.json").write_bytes(b"\x00\x01 not json " + marker.encode())
            section = portability_matrix(root)
            self.assertEqual(section["status"], "unavailable")
            self.assertEqual(section["detail"], "tokenizer_load_failed")
            self.assertNotIn(marker, json.dumps(section))

    def test_missing_tokenizer_json_is_unavailable(self):
        """Test missing tokenizer json is unavailable."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "second"
            root.mkdir()
            section = portability_matrix(root)
            self.assertEqual(section["status"], "unavailable")
            self.assertEqual(section["detail"], "tokenizer_load_failed")

    def test_non_dict_tokenizer_json_is_unavailable(self):
        """Test non dict tokenizer json is unavailable."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "second"
            root.mkdir()
            (root / "tokenizer.json").write_text("[1, 2, 3]", encoding="utf-8")
            section = portability_matrix(root)
            self.assertEqual(section["detail"], "tokenizer_load_failed")

    def test_non_path_input_raises(self):
        """Test non path input raises."""
        with self.assertRaises(ValueError):
            portability_matrix(123)

    def test_doctor_survives_hostile_second_checkpoint(self):
        """Test doctor survives hostile second checkpoint."""
        with TemporaryDirectory() as tmp:
            primary = write_primary_model(tmp)
            report = doctor(primary, portability_matrix_path=Path(tmp) / "missing")
            self.assertEqual(report["verdict"], "ok")
            self.assertEqual(report["portability_matrix"]["status"], "unavailable")

    def test_secondary_registered_skips_malformed_entries(self):
        """Test secondary registered skips malformed entries."""

        class BareTokenizer:
            """Bare tokenizer."""

        doc = {
            "added_tokens": [
                {"content": "<t>", "id": 7},
                {"content": "<bad>", "id": -1},
                {"content": "", "id": 8},
                {"content": 9, "id": 10},
                "not-a-dict",
            ]
        }
        self.assertEqual(_secondary_registered(BareTokenizer(), doc), {"<t>": 7})

    def test_secondary_registered_degrades_on_hostile_tokenizer(self):
        """Test secondary registered degrades on hostile tokenizer."""

        class HostileTokenizer:
            """Hostile tokenizer."""

            bos_token = "<s>"

            @staticmethod
            def convert_tokens_to_ids(token):
                """Convert tokens to ids."""
                raise RuntimeError("boom")

        doc = {"added_tokens": [{"content": "<t>", "id": 7}]}
        self.assertEqual(_secondary_registered(HostileTokenizer(), doc), {"<t>": 7})


class RenderTableTests(unittest.TestCase):
    """Render table tests."""

    def test_table_renders_unavailable(self):
        """Test table renders unavailable."""
        table = render_portability_table(portability_matrix("/definitely/not/here-12345"))
        self.assertIn("unavailable", table)
        self.assertIn("not_a_readable_directory", table)
        self.assertIn("ASYMMETRY WARNING", table)

    def test_table_has_no_trailing_whitespace(self):
        """Test table has no trailing whitespace."""
        table = render_portability_table(portability_matrix("/definitely/not/here-12345"))
        for line in table.splitlines():
            self.assertEqual(line, line.rstrip(), f"trailing whitespace: {line!r}")

    @unittest.skipUnless(HAS_TRANSFORMERS and HAS_TOKENIZERS, "hf extra not installed")
    def test_table_lists_every_candidate_with_verdict(self):
        """Test table lists every candidate with verdict."""
        from model_protocol_lab.catalog import CATALOG, candidates

        expected = [
            fmt.name for family in sorted(CATALOG) for fmt in candidates(family, mutate=True)
        ]
        with TemporaryDirectory() as tmp:
            second = write_tokenizer_checkpoint(Path(tmp) / "second", CHATML_TOKENS)
            table = render_portability_table(portability_matrix(second))
            self.assertIn("portability matrix", table)
            self.assertIn("ASYMMETRY WARNING", table)
            for name in expected:
                self.assertIn(name, table)


@unittest.skipUnless(HAS_TRANSFORMERS and HAS_TOKENIZERS, "hf extra not installed")
class CliWiringTests(unittest.TestCase):
    """Cli wiring tests."""

    def test_cli_flag_adds_matrix_and_stderr_table(self):
        """Test cli flag adds matrix and stderr table."""
        from model_protocol_lab.cli import main

        with TemporaryDirectory() as tmp:
            primary = write_primary_model(tmp)
            second = write_tokenizer_checkpoint(Path(tmp) / "second", CHATML_TOKENS)
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = main(
                    [
                        "doctor",
                        "--model",
                        str(primary),
                        "--portability-matrix",
                        str(second),
                    ]
                )
            self.assertEqual(code, 0)
            # stdout stays a parseable JSON report...
            report = json.loads(stdout.getvalue())
            self.assertEqual(report["verdict"], "ok")
            section = report["portability_matrix"]
            self.assertEqual(section["status"], "ok")
            self.assertTrue(section["rows"])
            # ...while the human-readable table goes to stderr.
            table = stderr.getvalue()
            self.assertIn("portability matrix", table)
            self.assertIn("chatml", table)
            self.assertIn("compatible", table)
            self.assertIn("ASYMMETRY WARNING", table)

    def test_cli_without_flag_has_no_section(self):
        """Test cli without flag has no section."""
        from model_protocol_lab.cli import main

        with TemporaryDirectory() as tmp:
            primary = write_primary_model(tmp)
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(["doctor", "--model", str(primary)])
            self.assertEqual(code, 0)
            self.assertNotIn("portability_matrix", json.loads(stdout.getvalue()))

    def test_cli_unavailable_second_checkpoint_still_exits_zero(self):
        """Test cli unavailable second checkpoint still exits zero."""
        from model_protocol_lab.cli import main

        with TemporaryDirectory() as tmp:
            primary = write_primary_model(tmp)
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = main(
                    [
                        "doctor",
                        "--model",
                        str(primary),
                        "--portability-matrix",
                        str(Path(tmp) / "missing"),
                    ]
                )
            self.assertEqual(code, 0)
            report = json.loads(stdout.getvalue())
            self.assertEqual(report["portability_matrix"]["status"], "unavailable")
            self.assertIn("unavailable", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
