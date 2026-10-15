"""Cycle 40: behavioral verification of doctor.py's examine_model internals.

Scope: everything in ``examine_model`` EXCEPT the asset filename validation
(verified in cycle 25) and the ``weight_headers`` byte check (cycle 26) --
i.e. the per-asset checks (config / tokenizer_file / tokenizer_deep /
boundary_ids / boundary_map / asset_manifest), report assembly
(``_model_result`` key contract, ``context_window`` extraction), verdict
logic, the cross-tokenizer portability machinery (``_portability_check``,
``portability_matrix``, ``render_portability_table``,
``_secondary_registered``), and the error paths of ``examine_dataset``,
``template_conformance`` and ``doctor``.

Method: synthetic checkpoints are built in tmp dirs only (hand-written
safetensors weight files, WordLevel tokenizer.json files via the
``tokenizers`` library, hostile JSON / symlink / oversized fixtures). Every
test pins the exact contract -- check names, status strings, fixed-vocabulary
details, report key sets, row ordering -- with adversarial inputs that must
degrade to a check status instead of raising or echoing raw bytes.

Doc/code discrepancies found while grounding the tests (no source behavior
changed; all are documentation precision notes, none are bugs):

* ``_portability_check`` docstring says "Never raises." -- verified to hold
  even for caller-type errors: a non-``str``/``Path``
  ``reference_model_path`` degrades to ``skipped /
  reference_tokenizer_unavailable`` instead of raising (the ``_directory``
  call sits inside the guarded block). No discrepancy; pinned by test.
* ``portability_matrix`` docstring says it "never raises for checkpoint
  problems" -- accurate -- but a non-``str``/``Path`` second checkpoint
  raises ``ValueError("second_checkpoint must be a path")`` by design; it is
  a caller-type error, not a checkpoint problem.
* ``template_conformance`` docstring says "Never raises" for checkpoint
  problems, but the *input* validation (non-tuple / empty / mistyped
  ``formats``/``cases``) raises ``ValueError`` by design; the no-raise
  promise starts after argument validation.
* ``examine_model``'s context-window rule is type-strict: ``bool`` is not
  accepted (``type(declared) is int``), so ``n_positions=True`` yields
  ``context_window=None`` rather than ``1`` -- the docstring's "missing or
  malformed value leaves it unknown" covers this, but the bool exclusion is
  easy to miss.
* ``portability_matrix``'s docstring promises "A format whose inventory is
  empty reports 'unknown', never 'compatible'." -- the current catalog has no
  empty-inventory format (all 114 candidates carry at least one token-shaped
  literal), so that branch is unreachable from the matrix today; the
  underlying ``core.template_tokenizer_compatibility`` contract still holds.
* ``doctor`` docstring calls the result a "combined pre-flight report" but
  the dataset half is optional -- ``doctor(model)`` returns no ``"dataset"``
  key at all, and ``"suggestion"`` is always present (possibly None).
"""

from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import struct
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

from model_protocol_lab import doctor as doctor_mod
from model_protocol_lab.catalog import CATALOG, candidates
from model_protocol_lab.core import Fixture, Format, Message

# ---------------------------------------------------------------------------
# Fixture builders (tmp dirs only; no network, no downloads)
# ---------------------------------------------------------------------------


def _write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj), encoding="utf-8")


def _write_valid_safetensors(path: Path) -> None:
    header = json.dumps(
        {"__metadata__": {}, "w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
    ).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\x00" * 8)


def _write_wordlevel_tokenizer(
    root: Path, n_words: int = 24, extra_specials: tuple[str, ...] = ()
) -> Path:
    """A genuinely loadable WordLevel tokenizer.json (tokenizers library)."""
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
    except ImportError:
        raise unittest.SkipTest("tokenizers library not installed") from None
    vocab = {f"w{i}": i for i in range(n_words)}
    vocab["[UNK]"] = n_words
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.add_special_tokens(["[UNK]", "[BOS]", "[EOS]", *extra_specials])
    tok.save(str(root / "tokenizer.json"))
    return root / "tokenizer.json"


def _write_checkpoint(
    root: Path,
    *,
    config: object = None,
    raw_files: dict[str, bytes] | None = None,
    extra_specials: tuple[str, ...] = (),
) -> Path:
    """Minimal valid single-file checkpoint; callers override pieces."""
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "config.json", {"model_type": "gpt2"} if config is None else config)
    _write_wordlevel_tokenizer(root, extra_specials=extra_specials)
    _write_valid_safetensors(root / "model.safetensors")
    for name, data in (raw_files or {}).items():
        (root / name).write_bytes(data)
    return root


def _write_dataset(path: Path, search_texts: list[str], heldout_texts: list[str]) -> None:
    def item(prefix: str, group: str, index: int, text: str) -> dict:
        return {
            "id": f"{prefix}{index}",
            "group": group,
            "messages": [{"role": "user", "content": text}],
            "expected": "e",
        }

    _write_json(
        path,
        {
            "search": [item("s", "gs", i, t) for i, t in enumerate(search_texts)],
            "held_out": [item("h", "gh", i, t) for i, t in enumerate(heldout_texts)],
        },
    )


class _TmpTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmpdir = Path(self._tmp.name)

    def ckpt(self, name: str = "ckpt", **kwargs) -> Path:
        return _write_checkpoint(self.tmpdir / name, **kwargs)


# ---------------------------------------------------------------------------
# Report assembly: _model_result key contract and early-failure shapes
# ---------------------------------------------------------------------------


class ModelResultAssemblyTests(_TmpTestCase):
    def test_check_returns_status_detail_shape(self) -> None:
        self.assertEqual(doctor_mod._check("ok", "x"), {"status": "ok", "detail": "x"})
        self.assertEqual(doctor_mod._check("failed"), {"status": "failed", "detail": ""})

    def test_bad_directory_report_carries_full_key_set(self) -> None:
        report = doctor_mod.examine_model(self.tmpdir / "does-not-exist")
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(set(report), {"verdict", "model"})
        model = report["model"]
        self.assertEqual(
            set(model), {"checks", "assets", "special_tokens", "added_tokens", "context_window"}
        )
        self.assertEqual(
            model["checks"],
            {"directory": {"status": "failed", "detail": "not a readable local directory"}},
        )
        self.assertEqual(model["assets"], [])
        self.assertEqual(model["special_tokens"], {})
        self.assertEqual(model["added_tokens"], {})
        self.assertIsNone(model["context_window"])

    def test_hostile_model_path_types_degrade_to_directory_failed(self) -> None:
        for bad in ("", 123, None, b"bytes-path"):
            with self.subTest(bad=repr(bad)):
                report = doctor_mod.examine_model(bad)  # type: ignore[arg-type]
                self.assertEqual(report["verdict"], "failed")
                self.assertEqual(
                    report["model"]["checks"]["directory"]["detail"],
                    "not a readable local directory",
                )
                self.assertEqual(report["model"]["assets"], [])

    def test_path_to_file_is_not_a_directory(self) -> None:
        target = self.tmpdir / "afile"
        target.write_text("x")
        report = doctor_mod.examine_model(target)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(
            report["model"]["checks"]["directory"]["detail"], "not a readable local directory"
        )

    def test_config_failure_returns_assets_but_empty_token_maps(self) -> None:
        root = self.ckpt(raw_files={"config.json": b"not json{"})
        report = doctor_mod.examine_model(root)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(report["model"]["checks"]["config"]["detail"], "invalid_asset_json")
        # Manifest ran before the config failure, so assets are recorded.
        self.assertTrue(report["model"]["assets"])
        self.assertEqual(report["model"]["special_tokens"], {})
        self.assertEqual(report["model"]["added_tokens"], {})
        self.assertIsNone(report["model"]["context_window"])
        self.assertNotIn("tokenizer_file", report["model"]["checks"])


# ---------------------------------------------------------------------------
# context_window extraction
# ---------------------------------------------------------------------------


class ContextWindowTests(_TmpTestCase):
    def _window(self, config: dict) -> object:
        root = self.ckpt(config=config)
        return doctor_mod.examine_model(root)["model"]["context_window"]

    def test_gpt2_uses_n_positions(self) -> None:
        self.assertEqual(self._window({"model_type": "gpt2", "n_positions": 1024}), 1024)

    def test_llama_uses_max_position_embeddings(self) -> None:
        self.assertEqual(
            self._window({"model_type": "llama", "max_position_embeddings": 8192}), 8192
        )

    def test_gpt2_ignores_max_position_embeddings(self) -> None:
        self.assertIsNone(self._window({"model_type": "gpt2", "max_position_embeddings": 4096}))

    def test_llama_ignores_n_positions(self) -> None:
        self.assertIsNone(self._window({"model_type": "llama", "n_positions": 4096}))

    def test_malformed_values_leave_window_unknown(self) -> None:
        for config in (
            {"model_type": "gpt2", "n_positions": 0},
            {"model_type": "gpt2", "n_positions": -5},
            {"model_type": "gpt2", "n_positions": "1024"},
            {"model_type": "gpt2", "n_positions": 1024.0},
            {"model_type": "gpt2", "n_positions": True},
            {"model_type": "gpt2"},
            {"model_type": "llama", "max_position_embeddings": None},
        ):
            with self.subTest(config=config):
                self.assertIsNone(self._window(config))


# ---------------------------------------------------------------------------
# Per-asset check: config
# ---------------------------------------------------------------------------


class ConfigCheckTests(_TmpTestCase):
    def _config_check(self, config: object, raw: bytes | None = None) -> dict:
        raw_files = {"config.json": raw} if raw is not None else None
        root = self.ckpt(config=config, raw_files=raw_files)
        return doctor_mod.examine_model(root)["model"]["checks"]["config"]

    def test_missing_config_json_fails_asset_layout_first(self) -> None:
        # asset_layout runs before the config check, so a missing config.json
        # surfaces as an asset-layout failure, not a config failure: this pins
        # the check ordering, not the layout rules themselves.
        root = self.ckpt()
        (root / "config.json").unlink()
        checks = doctor_mod.examine_model(root)["model"]["checks"]
        self.assertEqual(
            checks["asset_layout"], {"status": "failed", "detail": "missing_or_unsafe_asset"}
        )
        self.assertNotIn("config", checks)

    def test_config_symlink_fails_asset_layout_first(self) -> None:
        root = self.ckpt()
        real = root / "config.json.real"
        (root / "config.json").rename(real)
        (root / "config.json").symlink_to(real)
        checks = doctor_mod.examine_model(root)["model"]["checks"]
        self.assertEqual(
            checks["asset_layout"], {"status": "failed", "detail": "missing_or_unsafe_asset"}
        )
        self.assertNotIn("config", checks)

    def test_non_dict_config_json(self) -> None:
        check = self._config_check(None, raw=b"[1, 2, 3]")
        self.assertEqual(check, {"status": "failed", "detail": "invalid_asset_json"})

    def test_adversarial_config_bytes_never_echoed(self) -> None:
        hostile = b'{"model_type": "gpt2", "zzz": "' + b"A" * 500 + b'"} trailing {{{'
        check = self._config_check(None, raw=hostile)
        self.assertEqual(check["status"], "failed")
        self.assertEqual(check["detail"], "invalid_asset_json")
        self.assertNotIn("A" * 10, check["detail"])

    def test_unsupported_model_type_uses_repr(self) -> None:
        check = self._config_check({"model_type": "bert"})
        self.assertEqual(check, {"status": "failed", "detail": "unsupported model_type: 'bert'"})
        check = self._config_check({"model_type": None})
        self.assertEqual(check, {"status": "failed", "detail": "unsupported model_type: None"})

    def test_rejected_keys_only_when_truthy(self) -> None:
        check = self._config_check(
            {"model_type": "gpt2", "auto_map": {"x": 1}, "quantization_config": {"y": 2}}
        )
        self.assertEqual(
            check,
            {
                "status": "failed",
                "detail": "rejected config keys: auto_map, quantization_config",
            },
        )
        # Detail order follows _REJECTED_CONFIG_KEYS, not config order.
        check = self._config_check(
            {"model_type": "gpt2", "add_cross_attention": True, "auto_map": {"x": 1}}
        )
        self.assertEqual(check["detail"], "rejected config keys: auto_map, add_cross_attention")

    def test_falsy_rejected_keys_are_ignored(self) -> None:
        for config in (
            {"model_type": "gpt2", "is_encoder_decoder": False},
            {"model_type": "gpt2", "quantization_config": 0},
            {"model_type": "gpt2", "auto_map": None},
            {"model_type": "gpt2", "add_cross_attention": ""},
        ):
            with self.subTest(config=config):
                check = self._config_check(config)
                self.assertEqual(check["status"], "ok")

    def test_ok_config_detail(self) -> None:
        check = self._config_check({"model_type": "llama"})
        self.assertEqual(check, {"status": "ok", "detail": "model_type=llama"})


# ---------------------------------------------------------------------------
# Per-asset check: tokenizer_file
# ---------------------------------------------------------------------------


class TokenizerFileCheckTests(_TmpTestCase):
    def _patch_tokenizer_doc(self, patch: dict) -> dict:
        """Rewrite tokenizer.json's document with hostile content, keep it loadable."""
        root = self.ckpt()
        doc = json.loads((root / "tokenizer.json").read_text(encoding="utf-8"))
        doc.update(patch)
        (root / "tokenizer.json").write_text(json.dumps(doc), encoding="utf-8")
        return doctor_mod.examine_model(root)["model"]["checks"]["tokenizer_file"]

    def test_missing_model_section(self) -> None:
        check = self._patch_tokenizer_doc({"model": None})
        self.assertEqual(check, {"status": "failed", "detail": "tokenizer model section missing"})
        check = self._patch_tokenizer_doc({"model": "WordLevel"})
        self.assertEqual(check, {"status": "failed", "detail": "tokenizer model section missing"})

    def test_stochastic_dropout_rejected(self) -> None:
        for dropout in (0.5, 1, 0.1, True):
            with self.subTest(dropout=dropout):
                root = self.ckpt()
                doc = json.loads((root / "tokenizer.json").read_text(encoding="utf-8"))
                doc["model"]["dropout"] = dropout
                (root / "tokenizer.json").write_text(json.dumps(doc), encoding="utf-8")
                check = doctor_mod.examine_model(root)["model"]["checks"]["tokenizer_file"]
                self.assertEqual(
                    check, {"status": "failed", "detail": "stochastic tokenizer not supported"}
                )

    def test_zero_dropout_variants_accepted(self) -> None:
        for dropout in (0, 0.0, False, None):
            with self.subTest(dropout=repr(dropout)):
                check = self._patch_tokenizer_doc({"model": {"dropout": dropout}})
                # A bare {"dropout": ...} model dict is not loadable later, but
                # the *file* check only inspects the dropout field.
                self.assertEqual(check, {"status": "ok", "detail": "standalone tokenizer.json"})

    def test_ok_detail(self) -> None:
        root = self.ckpt()
        check = doctor_mod.examine_model(root)["model"]["checks"]["tokenizer_file"]
        self.assertEqual(check, {"status": "ok", "detail": "standalone tokenizer.json"})

    def test_oversized_tokenizer_json(self) -> None:
        # NOTE: examine_model reads tokenizer.json via _read_json (128 MB cap,
        # code "oversized_asset_json"), not via _tokenizer_file (whose own
        # 128 MB pre-load cap reports "oversized_tokenizer_file" and is only
        # exercised by the tokenizer-load paths: deep check, portability,
        # template conformance).
        root = self.ckpt()
        target = root / "tokenizer.json"
        target.unlink()
        # Sparse file: logical size over the 128 MB cap, no disk usage.
        with target.open("wb") as handle:
            handle.truncate(128 * 1024 * 1024 + 1)
        check = doctor_mod.examine_model(root)["model"]["checks"]["tokenizer_file"]
        self.assertEqual(check, {"status": "failed", "detail": "oversized_asset_json"})

    def test_tokenizer_json_as_directory(self) -> None:
        root = self.ckpt()
        (root / "tokenizer.json").unlink()
        (root / "tokenizer.json").mkdir()
        check = doctor_mod.examine_model(root)["model"]["checks"]["tokenizer_file"]
        self.assertEqual(check, {"status": "failed", "detail": "missing_or_unsafe_asset"})


# ---------------------------------------------------------------------------
# Per-asset check: tokenizer_deep + added_tokens / special_tokens maps
# ---------------------------------------------------------------------------


class TokenizerDeepCheckTests(_TmpTestCase):
    def test_deep_ok_reports_vocab_and_added_count(self) -> None:
        root = self.ckpt()
        report = doctor_mod.examine_model(root)
        check = report["model"]["checks"]["tokenizer_deep"]
        # 24 words + [UNK] + [BOS] + [EOS].
        self.assertEqual(check, {"status": "ok", "detail": "vocab_size=27, 3 added tokens"})
        self.assertEqual(report["model"]["added_tokens"], {"[UNK]": 24, "[BOS]": 25, "[EOS]": 26})
        # transformers does not surface special attrs from tokenizer_file alone.
        self.assertEqual(report["model"]["special_tokens"], {})

    def test_added_tokens_hostile_entries_filtered(self) -> None:
        root = self.ckpt()
        doc = json.loads((root / "tokenizer.json").read_text(encoding="utf-8"))
        doc["added_tokens"] = [
            {"content": "good", "id": 100},
            {"content": "good", "id": 101},  # duplicate: last wins
            {"content": "bool-id", "id": True},  # bool is not an int here
            {"content": 123, "id": 5},
            {"content": "no-id"},
            {"id": 6},
            "not-a-dict",
            None,
        ]
        (root / "tokenizer.json").write_text(json.dumps(doc), encoding="utf-8")
        report = doctor_mod.examine_model(root)
        # The tokenizer itself is now unloadable (added_tokens section is
        # consumed by the loader); the *document* parse still records "good".
        added = report["model"]["added_tokens"]
        self.assertEqual(added.get("good"), 101)
        self.assertNotIn("bool-id", added)

    def test_deep_failed_is_fail_closed_and_sanitized(self) -> None:
        root = self.ckpt()
        (root / "tokenizer.json").write_text(
            json.dumps({"model": {"type": "NopeXYZ"}, "added_tokens": [{"content": "x", "id": 1}]}),
            encoding="utf-8",
        )
        report = doctor_mod.examine_model(root)
        check = report["model"]["checks"]["tokenizer_deep"]
        self.assertEqual(check, {"status": "failed", "detail": "tokenizer_deep_check_failed"})
        self.assertEqual(report["verdict"], "failed")
        # The raw dependency exception text (which echoes hostile bytes) is
        # never interpolated into the report.
        self.assertNotIn("NopeXYZ", json.dumps(report))
        # added_tokens still parsed from the document even though deep failed.
        self.assertEqual(report["model"]["added_tokens"], {"x": 1})
        self.assertEqual(report["model"]["special_tokens"], {})

    def test_deep_skipped_without_transformers(self) -> None:
        root = self.ckpt()
        with mock.patch.object(
            doctor_mod, "_build_primary_tokenizer", side_effect=ImportError("nope")
        ):
            report = doctor_mod.examine_model(root, boundary_ids=(1,))
        self.assertEqual(
            report["model"]["checks"]["tokenizer_deep"],
            {
                "status": "skipped",
                "detail": "transformers not installed; install the 'hf' extra for deep checks",
            },
        )
        # Verdict stays ok: a skipped deep check is not a failure.
        self.assertEqual(report["verdict"], "ok")
        # Boundary validation degrades to skipped without a vocabulary.
        self.assertEqual(
            report["model"]["checks"]["boundary_ids"],
            {"status": "skipped", "detail": "vocabulary size unknown without the 'hf' extra"},
        )


# ---------------------------------------------------------------------------
# Per-asset check: asset_manifest structure
# ---------------------------------------------------------------------------


class AssetManifestTests(_TmpTestCase):
    def test_manifest_records_name_bytes_sha256(self) -> None:
        root = self.ckpt()
        report = doctor_mod.examine_model(root)
        assets = report["model"]["assets"]
        self.assertEqual([a["name"] for a in assets], ["config.json", "model.safetensors"])
        for record in assets:
            self.assertEqual(set(record), {"name", "bytes", "sha256"})
            data = (root / record["name"]).read_bytes()
            self.assertEqual(record["bytes"], len(data))
            self.assertEqual(record["sha256"], hashlib.sha256(data).hexdigest())


# ---------------------------------------------------------------------------
# Per-asset checks: boundary_ids / boundary_map
# ---------------------------------------------------------------------------


class BoundaryChecksTests(_TmpTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.root = self.ckpt()  # vocab: w0..w23 -> 0..23, [UNK]=24, [BOS]=25, [EOS]=26

    def _check(self, name: str, **kwargs) -> dict:
        return doctor_mod.examine_model(self.root, **kwargs)["model"]["checks"][name]

    def test_valid_boundary_ids(self) -> None:
        check = self._check("boundary_ids", boundary_ids=(0, 26))
        self.assertEqual(check, {"status": "ok", "detail": "2 ids within vocabulary"})

    def test_list_input_accepted(self) -> None:
        check = self._check("boundary_ids", boundary_ids=[1, 2])
        self.assertEqual(check["status"], "ok")

    def test_hostile_boundary_ids_fail(self) -> None:
        cases = {
            "empty": (),
            "out_of_range": (27,),
            "negative": (-1,),
            "bool": (True,),
            "float": (1.5,),
            "string": ("w0",),
            "non_iterable": 5,
            "none_element": (None,),
        }
        for label, bad in cases.items():
            with self.subTest(label=label):
                check = self._check("boundary_ids", boundary_ids=bad)
                self.assertEqual(
                    check, {"status": "failed", "detail": "invalid boundary ids"}, label
                )

    def test_no_boundary_ids_no_check_key(self) -> None:
        report = doctor_mod.examine_model(self.root)
        self.assertNotIn("boundary_ids", report["model"]["checks"])
        self.assertNotIn("boundary_map", report["model"]["checks"])

    def test_boundary_map_per_name_labels(self) -> None:
        report = doctor_mod.examine_model(
            self.root, boundary_map={"chatml": (0, 1), "llama3": (26,)}
        )
        checks = report["model"]["checks"]
        self.assertEqual(
            checks["boundary_ids:chatml"], {"status": "ok", "detail": "2 ids within vocabulary"}
        )
        self.assertEqual(
            checks["boundary_ids:llama3"], {"status": "ok", "detail": "1 ids within vocabulary"}
        )
        self.assertEqual(report["verdict"], "ok")

    def test_boundary_map_hostile_entries_fail_per_name(self) -> None:
        report = doctor_mod.examine_model(
            self.root,
            boundary_map={
                "good": (1,),
                "bad-ids": (999,),
                "empty": (),
                "": (1,),
                "non-iterable": 7,
                "bool": (False,),
            },
        )
        checks = report["model"]["checks"]
        self.assertEqual(checks["boundary_ids:good"]["status"], "ok")
        for name in ("bad-ids", "empty", "", "non-iterable", "bool"):
            with self.subTest(name=name):
                self.assertEqual(
                    checks[f"boundary_ids:{name}"],
                    {"status": "failed", "detail": "invalid boundary ids"},
                )
        self.assertEqual(report["verdict"], "failed")

    def test_boundary_map_non_string_name_fails(self) -> None:
        report = doctor_mod.examine_model(self.root, boundary_map={1: (0,)})  # type: ignore[dict-item]
        self.assertEqual(
            report["model"]["checks"]["boundary_ids:1"],
            {"status": "failed", "detail": "invalid boundary ids"},
        )

    def test_boundary_map_non_dict_fails(self) -> None:
        report = doctor_mod.examine_model(self.root, boundary_map=[("a", (1,))])  # type: ignore[arg-type]
        self.assertEqual(
            report["model"]["checks"]["boundary_map"],
            {"status": "failed", "detail": "invalid boundary map"},
        )
        self.assertEqual(report["verdict"], "failed")

    def test_boundary_map_skipped_without_vocab(self) -> None:
        with mock.patch.object(
            doctor_mod, "_build_primary_tokenizer", side_effect=ImportError("nope")
        ):
            report = doctor_mod.examine_model(self.root, boundary_map={"a": (1,), "b": (2,)})
        self.assertEqual(
            report["model"]["checks"]["boundary_ids:a"],
            {"status": "skipped", "detail": "vocabulary size unknown without the 'hf' extra"},
        )
        self.assertEqual(
            report["model"]["checks"]["boundary_ids:b"],
            {"status": "skipped", "detail": "vocabulary size unknown without the 'hf' extra"},
        )
        self.assertEqual(report["verdict"], "ok")


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------


class VerdictLogicTests(_TmpTestCase):
    def test_clean_checkpoint_verdict_ok(self) -> None:
        root = self.ckpt(config={"model_type": "llama", "max_position_embeddings": 512})
        report = doctor_mod.examine_model(root, boundary_ids=(0, 1))
        self.assertEqual(report["verdict"], "ok")
        self.assertTrue(all(c["status"] != "failed" for c in report["model"]["checks"].values()))

    def test_single_failed_check_fails_verdict(self) -> None:
        root = self.ckpt()
        report = doctor_mod.examine_model(root, boundary_ids=(10**9,))
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(report["model"]["checks"]["boundary_ids"]["status"], "failed")

    def test_skipped_checks_do_not_fail(self) -> None:
        root = self.ckpt()
        with mock.patch.object(
            doctor_mod, "_build_primary_tokenizer", side_effect=ImportError("nope")
        ):
            report = doctor_mod.examine_model(root, boundary_ids=(1,))
        statuses = {c["status"] for c in report["model"]["checks"].values()}
        self.assertIn("skipped", statuses)
        self.assertNotIn("failed", statuses)
        self.assertEqual(report["verdict"], "ok")

    def test_portability_mismatch_does_not_fail_verdict(self) -> None:
        primary = self.ckpt("primary")
        ref = self.tmpdir / "ref"
        ref.mkdir()
        # Reference maps w1 to a different id, so the boundary is not portable.
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel

        vocab = {f"w{i}": (i + 100) % 200 for i in range(24)}
        vocab["w1"] = 7
        vocab["[UNK]"] = 24
        tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
        tok.add_special_tokens(["[UNK]", "[BOS]", "[EOS]"])
        tok.save(str(ref / "tokenizer.json"))
        report = doctor_mod.examine_model(primary, boundary_ids=(1,), reference_model_path=ref)
        port = report["model"]["checks"]["boundary_portability"]
        self.assertEqual(port["status"], "mismatch")
        self.assertEqual(port["detail"], "primary=[1] reference=[7]")
        self.assertEqual(report["verdict"], "ok")


# ---------------------------------------------------------------------------
# _portability_check error paths (direct unit pins)
# ---------------------------------------------------------------------------


class PortabilityCheckUnitTests(_TmpTestCase):
    def test_primary_tokenizer_none(self) -> None:
        check = doctor_mod._portability_check(self.tmpdir, (1,), None)
        self.assertEqual(check, {"status": "skipped", "detail": "primary tokenizer unavailable"})

    def test_boundary_ids_none(self) -> None:
        check = doctor_mod._portability_check(self.tmpdir, None, object())
        self.assertEqual(check, {"status": "skipped", "detail": "needs --boundary-ids"})

    def test_invalid_boundary_ids(self) -> None:
        for bad in ((), (True,), (-1,), (1.5,), "ab", 5, (None,)):
            with self.subTest(bad=repr(bad)):
                check = doctor_mod._portability_check(self.tmpdir, bad, object())  # type: ignore[arg-type]
                self.assertEqual(check, {"status": "skipped", "detail": "invalid boundary ids"})

    def test_decode_failure_degrades(self) -> None:
        class ExplodingTokenizer:
            def decode(self, ids):  # noqa: ANN001, ANN202
                raise RuntimeError("boom")

        check = doctor_mod._portability_check(self.tmpdir, (1,), ExplodingTokenizer())
        self.assertEqual(check, {"status": "skipped", "detail": "boundary_decode_failed"})

    def test_non_string_or_empty_decode_degrades(self) -> None:
        for decoded in ("", None, 123):
            with self.subTest(decoded=repr(decoded)):
                fake = SimpleNamespace(decode=lambda ids, d=decoded: d)
                check = doctor_mod._portability_check(self.tmpdir, (1,), fake)
                self.assertEqual(check, {"status": "skipped", "detail": "boundary_decode_failed"})

    def test_reference_is_file(self) -> None:
        target = self.tmpdir / "afile"
        target.write_text("x")
        root = self.ckpt()
        report = doctor_mod.examine_model(root, boundary_ids=(1,), reference_model_path=target)
        self.assertEqual(
            report["model"]["checks"]["boundary_portability"],
            {"status": "skipped", "detail": "reference_tokenizer_unavailable"},
        )

    def test_reference_without_tokenizer_json(self) -> None:
        ref = self.tmpdir / "refdir"
        ref.mkdir()
        root = self.ckpt()
        report = doctor_mod.examine_model(root, boundary_ids=(1,), reference_model_path=ref)
        self.assertEqual(
            report["model"]["checks"]["boundary_portability"],
            {"status": "skipped", "detail": "reference_tokenizer_unavailable"},
        )

    def test_reference_non_path_type_degrades(self) -> None:
        root = self.ckpt()
        report = doctor_mod.examine_model(root, boundary_ids=(1,), reference_model_path=123)  # type: ignore[arg-type]
        self.assertEqual(
            report["model"]["checks"]["boundary_portability"],
            {"status": "skipped", "detail": "reference_tokenizer_unavailable"},
        )

    def test_secondary_encode_failure_maps_to_unavailable(self) -> None:
        ref = self.tmpdir / "refok"
        ref.mkdir()
        _write_wordlevel_tokenizer(ref)
        fake_primary = SimpleNamespace(decode=lambda ids: "w1")
        with mock.patch.object(
            doctor_mod,
            "check_boundary_portability",
            return_value={
                "detail": "secondary_encode_failed",
                "primary_ids": [1],
                "secondary_ids": None,
            },
        ):
            check = doctor_mod._portability_check(ref, (1,), fake_primary)
        self.assertEqual(check, {"status": "skipped", "detail": "reference_tokenizer_unavailable"})

    def test_identical_boundary_ok(self) -> None:
        root = self.ckpt()
        ref = self.tmpdir / "refsame"
        ref.mkdir()
        _write_wordlevel_tokenizer(ref)  # same w1 -> 1 mapping
        report = doctor_mod.examine_model(root, boundary_ids=(1,), reference_model_path=ref)
        self.assertEqual(
            report["model"]["checks"]["boundary_portability"],
            {"status": "ok", "detail": "boundary ids identical under reference tokenizer"},
        )


# ---------------------------------------------------------------------------
# _secondary_registered: hostile tokenizer.json documents
# ---------------------------------------------------------------------------


class SecondaryRegisteredTests(unittest.TestCase):
    def _registered(self, doc: dict, **attrs) -> dict:
        fake = SimpleNamespace(convert_tokens_to_ids=lambda tok: {"[BOS]": 25}.get(tok, 0), **attrs)
        return doctor_mod._secondary_registered(fake, doc)

    def test_hostile_entries_filtered_first_wins(self) -> None:
        doc = {
            "added_tokens": [
                {"content": "good", "id": 100},
                {"content": "good", "id": 101},  # duplicate: first wins
                {"content": "zero", "id": 0},  # id 0 is allowed
                {"content": "", "id": 5},  # empty literal skipped
                {"content": "neg", "id": -1},  # negative id skipped
                {"content": "bool-id", "id": True},  # bool is not an int here
                {"content": 7, "id": 8},  # non-str content skipped
                {"content": "no-id"},  # missing id skipped
                "not-a-dict",
                None,
            ]
        }
        registered = self._registered(doc, bos_token=None)
        self.assertEqual(registered, {"good": 100, "zero": 0})

    def test_non_list_added_tokens_ignored(self) -> None:
        self.assertEqual(self._registered({"added_tokens": {"a": 1}}, bos_token=None), {})

    def test_tokenizer_attrs_supplement_document(self) -> None:
        doc = {"added_tokens": [{"content": "doc-tok", "id": 42}]}
        registered = self._registered(doc, bos_token="[BOS]", eos_token=None)
        self.assertEqual(registered["doc-tok"], 42)
        self.assertEqual(registered["[BOS]"], 25)
        self.assertNotIn(None, registered)

    def test_document_wins_over_attr_on_collision(self) -> None:
        doc = {"added_tokens": [{"content": "[BOS]", "id": 99}]}
        registered = self._registered(doc, bos_token="[BOS]")
        self.assertEqual(registered["[BOS]"], 99)

    def test_hostile_convert_degrades_per_attribute(self) -> None:
        class HostileTokenizer:
            bos_token = "[BOS]"
            eos_token = "[EOS]"
            unk_token = None
            pad_token = None
            sep_token = None
            cls_token = None
            mask_token = None

            def convert_tokens_to_ids(self, token):  # noqa: ANN001, ANN202
                raise RuntimeError("hostile")

        registered = doctor_mod._secondary_registered(HostileTokenizer(), {"added_tokens": []})
        self.assertEqual(registered, {})

    def test_non_str_attr_ignored(self) -> None:
        registered = self._registered({"added_tokens": []}, bos_token=123, eos_token="")
        self.assertEqual(registered, {})


# ---------------------------------------------------------------------------
# _loader_failure_code
# ---------------------------------------------------------------------------


class LoaderFailureCodeTests(unittest.TestCase):
    def test_allowlisted_code_passes_through(self) -> None:
        self.assertEqual(
            doctor_mod._loader_failure_code(ValueError("oversized_tokenizer_file")),
            "oversized_tokenizer_file",
        )

    def test_everything_else_collapses(self) -> None:
        for exc in (
            ValueError("invalid_asset_json"),
            ValueError(" Oversized_tokenizer_file "),
            RuntimeError("boom"),
            Exception("data did not match any variant of untagged enum"),
        ):
            with self.subTest(exc=repr(exc)):
                self.assertEqual(doctor_mod._loader_failure_code(exc), "tokenizer_load_failed")


# ---------------------------------------------------------------------------
# portability_matrix
# ---------------------------------------------------------------------------


class PortabilityMatrixTests(_TmpTestCase):
    def _second_ckpt(self, name: str = "second", **kwargs) -> Path:
        root = self.tmpdir / name
        root.mkdir(parents=True, exist_ok=True)
        _write_wordlevel_tokenizer(root, **kwargs)
        return root

    def test_non_path_input_raises(self) -> None:
        with self.assertRaises(ValueError):
            doctor_mod.portability_matrix(123)  # type: ignore[arg-type]

    def test_unreadable_directory(self) -> None:
        section = doctor_mod.portability_matrix(self.tmpdir / "missing")
        self.assertEqual(section["status"], "unavailable")
        self.assertEqual(section["detail"], "not_a_readable_directory")
        self.assertEqual(section["rows"], [])
        self.assertEqual(section["second_checkpoint"], str(self.tmpdir / "missing"))
        self.assertIn("ASYMMETRY WARNING", section["asymmetry"])

    def test_file_path_is_unavailable(self) -> None:
        target = self.tmpdir / "afile"
        target.write_text("x")
        section = doctor_mod.portability_matrix(target)
        self.assertEqual(section["status"], "unavailable")
        self.assertEqual(section["detail"], "not_a_readable_directory")

    def test_invalid_tokenizer_json_sanitized(self) -> None:
        root = self._second_ckpt()
        (root / "tokenizer.json").write_bytes(b'{"broken": "json" trailing \xff\xfe')
        section = doctor_mod.portability_matrix(root)
        self.assertEqual(section["status"], "unavailable")
        self.assertEqual(section["detail"], "tokenizer_load_failed")
        self.assertNotIn("broken", json.dumps(section))

    def test_oversized_tokenizer_json(self) -> None:
        root = self._second_ckpt()
        target = root / "tokenizer.json"
        target.unlink()
        with target.open("wb") as handle:  # sparse: no disk usage
            handle.truncate(128 * 1024 * 1024 + 1)
        section = doctor_mod.portability_matrix(root)
        self.assertEqual(section["status"], "unavailable")
        self.assertEqual(section["detail"], "oversized_tokenizer_file")

    def test_unloadable_tokenizer_sanitized(self) -> None:
        root = self._second_ckpt()
        (root / "tokenizer.json").write_text(
            json.dumps({"model": {"type": "NopeXYZ"}, "added_tokens": []}), encoding="utf-8"
        )
        section = doctor_mod.portability_matrix(root)
        self.assertEqual(section["status"], "unavailable")
        self.assertEqual(section["detail"], "tokenizer_load_failed")
        self.assertNotIn("NopeXYZ", json.dumps(section))

    def test_ok_matrix_structure_and_order(self) -> None:
        root = self._second_ckpt(extra_specials=("<|im_start|>", "<|im_end|>"))
        section = doctor_mod.portability_matrix(root)
        self.assertEqual(section["status"], "ok")
        self.assertEqual(section["detail"], "114 candidates checked (tokenizer only, no weights)")
        self.assertEqual(section["second_checkpoint"], str(root.resolve()))
        self.assertEqual(section["asymmetry"], doctor_mod._PORTABILITY_ASYMMETRY)
        rows = section["rows"]
        self.assertEqual(len(rows), 114)
        expected_names = [
            fmt.name for family in sorted(CATALOG) for fmt in candidates(family, mutate=True)
        ]
        self.assertEqual([r["format"] for r in rows], expected_names)
        for row, family_fmt in zip(
            rows,
            [(f, fmt) for f in sorted(CATALOG) for fmt in candidates(f, mutate=True)],
            strict=True,
        ):
            family, fmt = family_fmt
            with self.subTest(format=row["format"]):
                self.assertIn(row["verdict"], {"compatible", "incompatible", "unknown"})
                self.assertIn("reason", row)
                self.assertEqual(row["bos_skipped"], fmt.bos == "")
        # Exactly the bos-less formats carry the skip flag.
        self.assertEqual(
            sum(1 for r in rows if r["bos_skipped"]),
            sum(1 for f in CATALOG for fmt in candidates(f, mutate=True) if fmt.bos == ""),
        )

    def test_compatible_row_when_inventory_registered(self) -> None:
        root = self._second_ckpt(extra_specials=("<|im_start|>", "<|im_end|>"))
        section = doctor_mod.portability_matrix(root)
        row = next(r for r in section["rows"] if r["format"] == "chatml")
        self.assertEqual(row["verdict"], "compatible")
        self.assertTrue(row["bos_skipped"])

    def test_incompatible_row_when_inventory_missing(self) -> None:
        root = self._second_ckpt()
        section = doctor_mod.portability_matrix(root)
        row = next(r for r in section["rows"] if r["format"] == "chatml")
        self.assertEqual(row["verdict"], "incompatible")

    def test_matrix_never_touches_weights(self) -> None:
        # A second "checkpoint" that is ONLY a tokenizer.json is enough.
        root = self._second_ckpt()
        section = doctor_mod.portability_matrix(root)
        self.assertEqual(section["status"], "ok")
        self.assertFalse((root / "model.safetensors").exists())


# ---------------------------------------------------------------------------
# render_portability_table
# ---------------------------------------------------------------------------


class RenderPortabilityTableTests(_TmpTestCase):
    def _ok_section(self) -> dict:
        root = self.tmpdir / "second"
        root.mkdir()
        _write_wordlevel_tokenizer(root, extra_specials=("<|im_start|>", "<|im_end|>"))
        return doctor_mod.portability_matrix(root)

    def test_ok_section_renders_table_and_asymmetry(self) -> None:
        text = doctor_mod.render_portability_table(self._ok_section())
        lines = text.split("\n")
        self.assertTrue(
            lines[0].startswith("portability matrix vs ")
            and lines[0].endswith(": 114 candidates (tokenizer-only; no weights loaded)")
        )
        header = next(line for line in lines if line.startswith("format"))
        self.assertIn("verdict", header)
        self.assertIn("reason", header)
        chatml_line = next(line for line in lines if line.startswith("chatml "))
        self.assertIn("compatible", chatml_line)
        self.assertTrue(chatml_line.rstrip().endswith("bos-skipped"))
        # A bos-carrying format has no bos-skipped suffix.
        deepseek_line = next(line for line in lines if line.startswith("deepseek "))
        self.assertFalse(deepseek_line.rstrip().endswith("bos-skipped"))
        self.assertIn(doctor_mod._PORTABILITY_ASYMMETRY, text)

    def test_unavailable_section(self) -> None:
        text = doctor_mod.render_portability_table(
            {
                "status": "unavailable",
                "detail": "not_a_readable_directory",
                "second_checkpoint": "/x",
                "asymmetry": doctor_mod._PORTABILITY_ASYMMETRY,
                "rows": [],
            }
        )
        self.assertIn("portability matrix vs /x", text)
        self.assertIn("status: unavailable (not_a_readable_directory)", text)
        self.assertIn(doctor_mod._PORTABILITY_ASYMMETRY, text)

    def test_hostile_section_degrades(self) -> None:
        text = doctor_mod.render_portability_table(
            {"status": "ok", "rows": "not-a-list", "asymmetry": doctor_mod._PORTABILITY_ASYMMETRY}
        )
        self.assertIn("portability matrix vs ?", text)
        self.assertIn("0 candidates", text)
        text = doctor_mod.render_portability_table(
            {
                "status": "ok",
                "second_checkpoint": "/x",
                "rows": ["junk", {"format": "f", "verdict": "v", "reason": "r"}],
                "asymmetry": doctor_mod._PORTABILITY_ASYMMETRY,
            }
        )
        self.assertIn("f", text)
        self.assertNotIn("junk", text.split(doctor_mod._PORTABILITY_ASYMMETRY)[0])


# ---------------------------------------------------------------------------
# examine_dataset error paths and advisory audit
# ---------------------------------------------------------------------------


class ExamineDatasetTests(_TmpTestCase):
    def test_missing_file_failed(self) -> None:
        report = doctor_mod.examine_dataset(self.tmpdir / "missing.json")
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(report["dataset"]["status"], "failed")
        # Fixed vocabulary: the raw OSError text never reaches the detail.
        self.assertEqual(report["dataset"]["detail"], "ValueError: Unreadable fixture file")

    def test_directory_path_failed(self) -> None:
        report = doctor_mod.examine_dataset(self.tmpdir)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(report["dataset"]["status"], "failed")

    def test_malformed_json_failed(self) -> None:
        target = self.tmpdir / "bad.json"
        target.write_bytes(b"{oops")
        report = doctor_mod.examine_dataset(target)
        self.assertEqual(report["verdict"], "failed")

    def test_clean_fixture_ok(self) -> None:
        target = self.tmpdir / "clean.json"
        _write_dataset(target, ["w0 w1 w2"], ["w3 w4"])
        report = doctor_mod.examine_dataset(target)
        self.assertEqual(report["verdict"], "ok")
        dataset = report["dataset"]
        self.assertEqual(dataset["status"], "ok")
        self.assertEqual(dataset["search_cases"], 1)
        self.assertEqual(dataset["held_out_cases"], 1)
        audit = dataset["special_token_audit"]
        self.assertEqual(audit["status"], "ok")
        self.assertEqual(audit["detail"], "no special-token literals in fixture text")
        self.assertEqual(audit["findings"], [])

    def test_special_token_literal_is_warning_not_failure(self) -> None:
        target = self.tmpdir / "dirty.json"
        _write_dataset(target, ["hello <|im_start|> world"], ["plain"])
        report = doctor_mod.examine_dataset(target)
        self.assertEqual(report["verdict"], "ok")
        audit = report["dataset"]["special_token_audit"]
        self.assertEqual(audit["status"], "warning")
        self.assertTrue(audit["findings"])
        finding = audit["findings"][0]
        self.assertEqual(set(finding), {"fixture_id", "field", "literal", "format_name", "offset"})
        self.assertEqual(finding["fixture_id"], "s0")
        self.assertEqual(finding["literal"], "<|im_start|>")
        self.assertEqual(finding["format_name"], "chatml")
        self.assertIn("GHSA-g5f9-3xfg-p9mf", audit["detail"])


# ---------------------------------------------------------------------------
# template_conformance error paths (+ one happy path)
# ---------------------------------------------------------------------------


class TemplateConformanceTests(_TmpTestCase):
    def _formats_cases(self) -> tuple[tuple[Format, ...], tuple[Fixture, ...]]:
        fmt = candidates("chatml", mutate=False)[0]
        case = Fixture(
            id="c0", messages=(Message(role="user", content="w0 w1"),), expected="e", group="g"
        )
        return (fmt,), (case,)

    def test_invalid_inputs_raise(self) -> None:
        formats, cases = self._formats_cases()
        bad_calls = [
            (list(formats), cases),
            ((), cases),
            (("not-a-format",), cases),
            (formats, list(cases)),
            (formats, ()),
            ((formats[0], "x"), cases),
            (formats, (cases[0], 42)),
        ]
        for bad_formats, bad_cases in bad_calls:
            with self.subTest(bad=(bad_formats, bad_cases)), self.assertRaises(ValueError):
                doctor_mod.template_conformance(  # type: ignore[arg-type]
                    self.tmpdir, bad_formats, bad_cases
                )

    def test_missing_tokenizer_config(self) -> None:
        root = self.ckpt()
        result = doctor_mod.template_conformance(root, *self._formats_cases())
        self.assertEqual(result["status"], "no_template_declared")
        self.assertEqual(result["detail"], "tokenizer_config_unreadable")
        self.assertEqual(result["diffs"], [])

    def test_no_chat_template_in_config(self) -> None:
        for config in ({}, {"chat_template": ""}, {"chat_template": "   "}, {"chat_template": 42}):
            with self.subTest(config=config):
                root = self.ckpt(raw_files={"tokenizer_config.json": json.dumps(config).encode()})
                result = doctor_mod.template_conformance(root, *self._formats_cases())
                self.assertEqual(result["status"], "no_template_declared")
                self.assertEqual(result["detail"], "no_chat_template_in_config")

    def test_oversized_template_unavailable(self) -> None:
        root = self.ckpt(
            raw_files={
                "tokenizer_config.json": json.dumps({"chat_template": "x" * 1_000_001}).encode()
            }
        )
        result = doctor_mod.template_conformance(root, *self._formats_cases())
        self.assertEqual(
            result, {"status": "unavailable", "detail": "chat_template_too_large", "diffs": []}
        )

    def test_happy_path_structure(self) -> None:
        template = "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
        root = self.ckpt(
            raw_files={"tokenizer_config.json": json.dumps({"chat_template": template}).encode()}
        )
        formats, cases = self._formats_cases()
        result = doctor_mod.template_conformance(root, formats, cases)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["diffs"]), 1)
        entry = result["diffs"][0]
        self.assertEqual(entry["format"], "chatml")
        self.assertEqual(entry["fixture"], "c0")
        self.assertEqual(entry["status"], "ok")
        self.assertIsInstance(entry["diff"], list)
        self.assertGreater(entry["hand_tokens"], 0)
        self.assertGreater(entry["template_tokens"], 0)


# ---------------------------------------------------------------------------
# doctor() integration: key assembly, dataset verdict, matrix triage
# ---------------------------------------------------------------------------


class DoctorIntegrationTests(_TmpTestCase):
    def test_suggestion_key_always_present(self) -> None:
        root = self.ckpt()
        report = doctor_mod.doctor(root)
        self.assertIn("suggestion", report)
        self.assertIsNone(report["suggestion"])
        self.assertNotIn("dataset", report)
        self.assertNotIn("portability_matrix", report)

    def test_dataset_failure_flips_verdict(self) -> None:
        root = self.ckpt()
        report = doctor_mod.doctor(root, dataset_path=self.tmpdir / "missing.json")
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(report["dataset"]["status"], "failed")

    def test_dataset_warning_keeps_verdict_ok(self) -> None:
        root = self.ckpt()
        ds = self.tmpdir / "dirty.json"
        _write_dataset(ds, ["hello <|im_start|> world"], ["plain"])
        report = doctor_mod.doctor(root, dataset_path=ds)
        self.assertEqual(report["verdict"], "ok")
        self.assertEqual(report["dataset"]["special_token_audit"]["status"], "warning")

    def test_matrix_exception_degrades_to_unavailable(self) -> None:
        root = self.ckpt()
        report = doctor_mod.doctor(root, portability_matrix_path=123)  # type: ignore[arg-type]
        section = report["portability_matrix"]
        self.assertEqual(section["status"], "unavailable")
        self.assertEqual(section["detail"], "matrix_failed")
        self.assertEqual(section["rows"], [])
        self.assertEqual(report["verdict"], "ok")

    def test_matrix_ok_section_attached(self) -> None:
        root = self.ckpt()
        second = self.tmpdir / "second"
        second.mkdir()
        _write_wordlevel_tokenizer(second)
        report = doctor_mod.doctor(root, portability_matrix_path=second)
        self.assertEqual(report["portability_matrix"]["status"], "ok")
        self.assertEqual(len(report["portability_matrix"]["rows"]), 114)

    def test_matrix_never_fails_verdict(self) -> None:
        root = self.ckpt()
        report = doctor_mod.doctor(root, portability_matrix_path=self.tmpdir / "missing")
        self.assertEqual(report["portability_matrix"]["status"], "unavailable")
        self.assertEqual(report["verdict"], "ok")

    def test_suggestion_present_for_clean_preflight(self) -> None:
        root = self.ckpt(config={"model_type": "gpt2", "n_positions": 4096})
        ds = self.tmpdir / "ds.json"
        _write_dataset(ds, ["w0 w1"], ["w2"])
        report = doctor_mod.doctor(root, dataset_path=ds, boundary_ids=(1, 2))
        suggestion = report["suggestion"]
        self.assertIsNotNone(suggestion)
        self.assertIn("python -m model_protocol_lab run", suggestion["next_command"])
        self.assertIn("--max-context 4096", suggestion["next_command"])
        self.assertIn("--boundary-ids 1,2", suggestion["next_command"])

    def test_no_suggestion_without_boundary_ids(self) -> None:
        root = self.ckpt(config={"model_type": "gpt2", "n_positions": 4096})
        ds = self.tmpdir / "ds.json"
        _write_dataset(ds, ["w0"], ["w1"])
        report = doctor_mod.doctor(root, dataset_path=ds)
        self.assertIsNone(report["suggestion"])


# ---------------------------------------------------------------------------
# Mutation-style sensitivity controls (mutate a /tmp copy, never the repo)
# ---------------------------------------------------------------------------

_MUTANT_PKG = "mplmut40"


def _make_mutant_package(tmpdir: str, mutations: list[tuple[str, str]]) -> Path:
    """Copy the package to a tmp dir under a fresh top-level name and mutate doctor.py."""
    src = Path(__file__).resolve().parent.parent / "model_protocol_lab"
    pkgdir = Path(tmpdir) / "mutpkg"
    shutil.copytree(src, pkgdir / _MUTANT_PKG, ignore=shutil.ignore_patterns("__pycache__"))
    target = pkgdir / _MUTANT_PKG / "doctor.py"
    text = target.read_text(encoding="utf-8")
    for old, new in mutations:
        count = text.count(old)
        if count != 1:
            raise AssertionError(f"mutation anchor not unique: {count} x {old[:60]!r}")
        text = text.replace(old, new)
    target.write_text(text, encoding="utf-8")
    return pkgdir


def _import_mutant_doctor(pkgdir: Path):
    sys.path.insert(0, str(pkgdir))
    try:
        return importlib.import_module(f"{_MUTANT_PKG}.doctor")
    finally:
        sys.path.remove(str(pkgdir))


def _cleanup_mutant_modules() -> None:
    for name in [n for n in sys.modules if n == _MUTANT_PKG or n.startswith(_MUTANT_PKG + ".")]:
        del sys.modules[name]


class MutationTests(_TmpTestCase):
    """Each mutant must be caught: the probe assertion must FAIL on the mutant."""

    def _assert_mutant_caught(self, mutations: list[tuple[str, str]], probe) -> None:
        pkgdir = _make_mutant_package(str(self.tmpdir), mutations)
        try:
            mutant = _import_mutant_doctor(pkgdir)
            with self.assertRaises(AssertionError):
                probe(mutant)
        finally:
            _cleanup_mutant_modules()
            shutil.rmtree(pkgdir, ignore_errors=True)

    def _good_ckpt(self, mutant, name: str) -> Path:
        root = self.tmpdir / name
        return _write_checkpoint(root)

    def test_mutant_verdict_inverted(self) -> None:
        def probe(m):
            report = m.examine_model(self._good_ckpt(m, "m1"), boundary_ids=(10**9,))
            assert report["verdict"] == "failed", "inverted verdict not caught"

        self._assert_mutant_caught(
            [
                (
                    'verdict = "failed" if any(',
                    'verdict = "ok" if any(',
                )
            ],
            probe,
        )

    def test_mutant_bool_boundary_ids_accepted(self) -> None:
        def probe(m):
            report = m.examine_model(self._good_ckpt(m, "m2"), boundary_ids=(True,))
            check = report["model"]["checks"]["boundary_ids"]
            assert check["status"] == "failed", "bool boundary id accepted"

        self._assert_mutant_caught(
            [
                (
                    "if ids and all(type(token) is int and 0 <= token "
                    "< vocab_size for token in ids):",
                    "if ids and all(isinstance(token, int) and 0 <= token "
                    "< vocab_size for token in ids):",
                )
            ],
            probe,
        )

    def test_mutant_mismatch_detail_swapped(self) -> None:
        def probe(m):
            ref = self.tmpdir / "m3ref"
            ref.mkdir(exist_ok=True)
            from tokenizers import Tokenizer
            from tokenizers.models import WordLevel

            vocab = {f"w{i}": i for i in range(24)}
            vocab["w1"] = 7
            vocab["[UNK]"] = 24
            tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
            tok.add_special_tokens(["[UNK]", "[BOS]", "[EOS]"])
            tok.save(str(ref / "tokenizer.json"))
            check = m._portability_check(ref, (1,), SimpleNamespace(decode=lambda ids: "w1"))
            assert check["detail"] == "primary=[1] reference=[7]", check["detail"]

        self._assert_mutant_caught(
            [
                (
                    "f\"primary={outcome['primary_ids']} reference={outcome['secondary_ids']}\"",
                    "f\"primary={outcome['secondary_ids']} reference={outcome['primary_ids']}\"",
                )
            ],
            probe,
        )

    def test_mutant_asymmetry_dropped(self) -> None:
        def probe(m):
            text = m.render_portability_table(
                {
                    "status": "unavailable",
                    "detail": "x",
                    "second_checkpoint": "/x",
                    "asymmetry": m._PORTABILITY_ASYMMETRY,
                    "rows": [],
                }
            )
            assert m._PORTABILITY_ASYMMETRY in text, "asymmetry dropped"

        self._assert_mutant_caught(
            [('    lines.extend(("", _PORTABILITY_ASYMMETRY))', '    lines.extend(("",))')],
            probe,
        )

    def test_mutant_bos_skipped_dropped(self) -> None:
        def probe(m):
            root = self.tmpdir / "m5second"
            root.mkdir(exist_ok=True)
            _write_wordlevel_tokenizer(root)
            section = m.portability_matrix(root)
            assert all("bos_skipped" in r for r in section["rows"]), "bos_skipped dropped"

        self._assert_mutant_caught(
            [
                (
                    '            row["bos_skipped"] = fmt.bos == ""\n            rows.append(row)',
                    "            rows.append(row)",
                )
            ],
            probe,
        )

    def test_mutant_bool_context_window(self) -> None:
        def probe(m):
            root = self.tmpdir / "m6"
            _write_checkpoint(root, config={"model_type": "gpt2", "n_positions": True})
            window = m.examine_model(root)["model"]["context_window"]
            assert window is None, f"bool window accepted: {window!r}"

        self._assert_mutant_caught(
            [
                (
                    "if type(declared) is int and declared > 0 else None",
                    "if isinstance(declared, int) and declared > 0 else None",
                )
            ],
            probe,
        )

    def test_mutant_portability_detail_reworded(self) -> None:
        def probe(m):
            check = m._portability_check(self.tmpdir, None, object())
            assert check == {"status": "skipped", "detail": "needs --boundary-ids"}, check

        self._assert_mutant_caught(
            [
                (
                    'return _check("skipped", "needs --boundary-ids")',
                    'return _check("skipped", "missing boundary ids")',
                )
            ],
            probe,
        )

    def test_mutant_secondary_registered_id_zero_rejected(self) -> None:
        def probe(m):
            fake = SimpleNamespace(bos_token=None)
            registered = m._secondary_registered(
                fake, {"added_tokens": [{"content": "z", "id": 0}]}
            )
            assert registered == {"z": 0}, registered

        self._assert_mutant_caught(
            [('and entry["id"] >= 0', 'and entry["id"] > 0')],
            probe,
        )

    def test_mutant_model_result_assets_default(self) -> None:
        def probe(m):
            report = m._model_result("failed", {})
            assert report["model"]["assets"] == [], report["model"]["assets"]

        self._assert_mutant_caught(
            [('"assets": assets if assets is not None else [],', '"assets": assets,')],
            probe,
        )

    def test_mutant_template_conformance_validation_removed(self) -> None:
        def probe(m):
            with self.assertRaises(ValueError):
                m.template_conformance(self.tmpdir, (), ())

        self._assert_mutant_caught(
            [
                (
                    '        raise ValueError("template_conformance needs '
                    'nonempty tuples of Format and Fixture")',
                    "        pass  # validation removed",
                )
            ],
            probe,
        )


if __name__ == "__main__":
    unittest.main()
