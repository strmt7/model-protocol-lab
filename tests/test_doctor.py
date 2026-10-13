"""Tests for the doctor pre-flight command and asset error vocabulary."""

import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from model_protocol_lab.backends import _directory
from model_protocol_lab.cli import main
from model_protocol_lab.doctor import examine_model
from model_protocol_lab.experiments import candidate_fingerprint
from tests.test_core import BASE
from tests.test_workflows import dataset_object

HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None
HAS_TOKENIZERS = importlib.util.find_spec("tokenizers") is not None


def write_model_dir(
    root: Path,
    *,
    model_type: str = "gpt2",
    extra_config: dict | None = None,
    tokenizer_model: dict | None = None,
    adapter: bool = False,
    weights: bool = True,
) -> Path:
    config = {"model_type": model_type}
    config.update(extra_config or {})
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if weights:
        (root / "model.safetensors").write_bytes(b"synthetic weights, not a model")
    if tokenizer_model is None:
        # Prefer a genuinely loadable tokenizer when the library is present;
        # otherwise fall back to a structurally valid document (deep checks
        # are skipped without transformers anyway).
        if importlib.util.find_spec("tokenizers") is not None:
            write_loadable_tokenizer(root)
            tokenizer_model = None
        else:
            tokenizer_model = {"type": "WordLevel", "vocab": {"a": 0}}
    if tokenizer_model is not None:
        (root / "tokenizer.json").write_text(
            json.dumps({"model": tokenizer_model}), encoding="utf-8"
        )
    if adapter:
        (root / "adapter_config.json").write_text("{}", encoding="utf-8")
    return root


def write_loadable_tokenizer(root: Path, vocab_size: int = 16) -> None:
    """Build a genuinely loadable tokenizer.json with the tokenizers library."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    vocab = {f"tok{i}": i for i in range(vocab_size - 1)}
    vocab["[UNK]"] = vocab_size - 1
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.add_special_tokens(["[UNK]", "[BOS]", "[EOS]"])
    tokenizer.save(str(root / "tokenizer.json"))


class DirectoryTests(unittest.TestCase):
    def test_missing_path_raises_value_error_not_os_error(self):
        with TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            _directory(Path(tmp) / "no-such-dir")

    def test_empty_and_file_paths_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "file").write_text("x", encoding="utf-8")
            for bad in ("", root / "file", root / "missing"):
                with self.subTest(bad=str(bad)), self.assertRaises(ValueError):
                    _directory(bad)

    def test_valid_directory_resolves(self):
        with TemporaryDirectory() as tmp:
            self.assertEqual(_directory(tmp), Path(tmp).resolve())


class DoctorTests(unittest.TestCase):
    def test_healthy_directory_passes(self):
        with TemporaryDirectory() as tmp:
            root = write_model_dir(Path(tmp))
            report = examine_model(root)
            self.assertEqual(report["verdict"], "ok")
            self.assertEqual(report["model"]["checks"]["directory"]["status"], "ok")
            self.assertEqual(report["model"]["checks"]["asset_layout"]["status"], "ok")
            self.assertEqual(report["model"]["checks"]["config"]["status"], "ok")
            self.assertEqual(len(report["model"]["assets"]), 2)

    def test_missing_directory_fails(self):
        report = examine_model("/definitely/not/here-12345")
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(report["model"]["checks"]["directory"]["status"], "failed")

    def test_adapter_directory_fails(self):
        with TemporaryDirectory() as tmp:
            root = write_model_dir(Path(tmp), adapter=True)
            report = examine_model(root)
            self.assertEqual(report["verdict"], "failed")
            self.assertEqual(report["model"]["checks"]["asset_layout"]["status"], "failed")

    def test_missing_weights_fails(self):
        with TemporaryDirectory() as tmp:
            root = write_model_dir(Path(tmp), weights=False)
            self.assertEqual(examine_model(root)["verdict"], "failed")

    def test_unsupported_model_type_fails(self):
        with TemporaryDirectory() as tmp:
            root = write_model_dir(Path(tmp), model_type="bert")
            report = examine_model(root)
            self.assertEqual(report["verdict"], "failed")
            self.assertIn("unsupported", report["model"]["checks"]["config"]["detail"])

    def test_rejected_config_keys_fail(self):
        for key in ("auto_map", "quantization_config", "is_encoder_decoder"):
            with self.subTest(key=key), TemporaryDirectory() as tmp:
                root = write_model_dir(Path(tmp), extra_config={key: {"x": 1}})
                report = examine_model(root)
                self.assertEqual(report["verdict"], "failed")

    def test_stochastic_tokenizer_fails(self):
        with TemporaryDirectory() as tmp:
            root = write_model_dir(Path(tmp), tokenizer_model={"type": "BPE", "dropout": 0.5})
            report = examine_model(root)
            self.assertEqual(report["verdict"], "failed")
            self.assertEqual(report["model"]["checks"]["tokenizer_file"]["status"], "failed")

    def test_manifest_records_hashes(self):
        with TemporaryDirectory() as tmp:
            root = write_model_dir(Path(tmp))
            report = examine_model(root)
            for record in report["model"]["assets"]:
                self.assertEqual(len(record["sha256"]), 64)
                self.assertGreater(record["bytes"], 0)

    @unittest.skipUnless(HAS_TRANSFORMERS and HAS_TOKENIZERS, "hf extra not installed")
    def test_deep_tokenizer_check_reports_vocab_and_specials(self):
        with TemporaryDirectory() as tmp:
            root = write_model_dir(Path(tmp))
            write_loadable_tokenizer(root, vocab_size=16)
            report = examine_model(root, boundary_ids=(0, 15))
            checks = report["model"]["checks"]
            self.assertEqual(report["verdict"], "ok")
            self.assertEqual(checks["tokenizer_deep"]["status"], "ok")
            self.assertIn("vocab_size=18", checks["tokenizer_deep"]["detail"])
            self.assertEqual(checks["boundary_ids"]["status"], "ok")
            added = report["model"]["added_tokens"]
            self.assertEqual(added["[UNK]"], 15)
            self.assertEqual(added["[BOS]"], 16)
            self.assertEqual(added["[EOS]"], 17)

    @unittest.skipUnless(HAS_TRANSFORMERS and HAS_TOKENIZERS, "hf extra not installed")
    def test_out_of_vocabulary_boundary_ids_fail(self):
        with TemporaryDirectory() as tmp:
            root = write_model_dir(Path(tmp))
            write_loadable_tokenizer(root, vocab_size=16)  # 18 ids after specials
            report = examine_model(root, boundary_ids=(18,))
            self.assertEqual(report["verdict"], "failed")
            self.assertEqual(report["model"]["checks"]["boundary_ids"]["status"], "failed")

    def test_doctor_cli_exit_codes(self):
        import io
        from contextlib import redirect_stdout

        with TemporaryDirectory() as tmp:
            root = write_model_dir(Path(tmp))
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["doctor", "--model", str(root)]), 0)
            parsed = json.loads(output.getvalue())
            self.assertEqual(parsed["verdict"], "ok")

            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["doctor", "--model", str(root / "missing")]), 1)

    def test_doctor_cli_checks_dataset(self):
        import io
        from contextlib import redirect_stdout

        with TemporaryDirectory() as tmp:
            root = write_model_dir(Path(tmp))
            fixtures = Path(tmp) / "f.json"
            fixtures.write_text(json.dumps(dataset_object()), encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(["doctor", "--model", str(root), "--dataset", str(fixtures)])
            self.assertEqual(code, 0)
            parsed = json.loads(output.getvalue())
            self.assertEqual(parsed["dataset"]["search_cases"], 2)

            bad = Path(tmp) / "bad.json"
            bad.write_text("{}", encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(["doctor", "--model", str(root), "--dataset", str(bad)])
            self.assertEqual(code, 1)


class FingerprintTests(unittest.TestCase):
    def test_fingerprint_is_stable_and_discriminating(self):
        first, second = candidate_fingerprint(BASE), candidate_fingerprint(BASE)
        self.assertEqual(first, second)
        self.assertTrue(first.startswith("sha256:"))
        self.assertEqual(len(first), len("sha256:") + 64)
        other = candidate_fingerprint(BASE.__class__("other", "<B>", "<U>", "<A>", "<E>"))
        self.assertNotEqual(first, other)

    def test_fingerprint_matches_manual_computation(self):
        import hashlib

        payload = json.dumps(
            {
                "assistant_prefix": "<A>",
                "bos": "<B>",
                "name": "base",
                "repeat_bos": False,
                "turn_end": "<E>",
                "user_prefix": "<U>",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        expected = "sha256:" + hashlib.sha256(payload).hexdigest()
        self.assertEqual(candidate_fingerprint(BASE), expected)


if __name__ == "__main__":
    unittest.main()


class DoctorBranchTests(unittest.TestCase):
    def test_manifest_failure_fails_closed(self):
        from unittest.mock import patch

        with TemporaryDirectory() as tmp:
            root = write_model_dir(Path(tmp))
            with patch("model_protocol_lab.doctor._manifest", side_effect=ValueError("boom")):
                report = examine_model(root)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(report["model"]["checks"]["asset_manifest"]["status"], "failed")

    def test_invalid_config_json_fails(self):
        with TemporaryDirectory() as tmp:
            root = write_model_dir(Path(tmp))
            (root / "config.json").write_text("not json", encoding="utf-8")
            report = examine_model(root)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(report["model"]["checks"]["config"]["status"], "failed")

    def test_invalid_tokenizer_json_fails(self):
        with TemporaryDirectory() as tmp:
            root = write_model_dir(Path(tmp))
            (root / "tokenizer.json").write_text("not json", encoding="utf-8")
            report = examine_model(root)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(report["model"]["checks"]["tokenizer_file"]["status"], "failed")

    def test_missing_tokenizer_model_section_fails(self):
        with TemporaryDirectory() as tmp:
            root = write_model_dir(Path(tmp))
            (root / "tokenizer.json").write_text("{}", encoding="utf-8")
            report = examine_model(root)
        self.assertEqual(report["verdict"], "failed")
        self.assertIn("missing", report["model"]["checks"]["tokenizer_file"]["detail"])

    def test_missing_transformers_skips_deep_checks(self):
        import sys
        from unittest.mock import patch

        with TemporaryDirectory() as tmp:
            root = write_model_dir(Path(tmp))
            with patch.dict(sys.modules, {"transformers": None}):
                report = examine_model(root, boundary_ids=(0,))
        checks = report["model"]["checks"]
        self.assertEqual(checks["tokenizer_deep"]["status"], "skipped")
        self.assertEqual(checks["boundary_ids"]["status"], "skipped")
        # Skips are not failures.
        self.assertEqual(report["verdict"], "ok")

    @unittest.skipUnless(HAS_TRANSFORMERS, "hf extra not installed")
    def test_unloadable_tokenizer_fails_deep_check(self):
        with TemporaryDirectory() as tmp:
            root = write_model_dir(Path(tmp), tokenizer_model={"type": "BPE"})
            report = examine_model(root)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(report["model"]["checks"]["tokenizer_deep"]["status"], "failed")

    def test_dataset_examination_reports_counts(self):
        import json as _json

        from model_protocol_lab.doctor import examine_dataset

        with TemporaryDirectory() as tmp:
            good = Path(tmp) / "good.json"
            good.write_text(_json.dumps(dataset_object()), encoding="utf-8")
            self.assertEqual(examine_dataset(good)["verdict"], "ok")
            bad = Path(tmp) / "bad.json"
            bad.write_text("{}", encoding="utf-8")
            report = examine_dataset(bad)
            self.assertEqual(report["verdict"], "failed")
            self.assertIn("detail", report["dataset"])


class AddedTokensTests(unittest.TestCase):
    def test_malformed_added_tokens_entries_ignored(self):
        with TemporaryDirectory() as tmp:
            root = write_model_dir(Path(tmp))
            doc = json.loads((root / "tokenizer.json").read_text(encoding="utf-8"))
            doc["added_tokens"] = [
                {"content": "<s>", "id": 1},
                {"content": 123, "id": 2},
                {"content": "<x>", "id": "3"},
                "not-a-dict",
                None,
            ]
            (root / "tokenizer.json").write_text(json.dumps(doc), encoding="utf-8")
            report = examine_model(root)
        self.assertEqual(report["model"]["added_tokens"], {"<s>": 1})
