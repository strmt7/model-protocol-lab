"""Cycle 72 adversarial audit: ``examine_model`` / ``doctor()`` integration.

Independent-oracle differential tests for the end-to-end model-directory
examination flow in ``model_protocol_lab.doctor``:

* how ``examine_model`` assembles the safetensors / config / tokenizer
  inspection stages, in what order checks are reported, and which stages
  abort the run early vs. let later stages continue;
* how the verdict is computed from per-check statuses (verdict precedence
  when several stages fail, minimal-pair missing-vs-malformed cases);
* how ``doctor()`` orchestrates the model report, the dataset report, the
  portability-matrix triage and the run suggestion into one final report;
* adversarial edges: empty dirs, 0-byte files, garbage bytes, oversized
  configs, unicode paths, symlinked dirs, sharded weight layouts.

Every expectation below is hand-derived from the documented stage
semantics (an independent oracle), not from re-running the source logic.
Fixtures are synthetic model directories built in PID-scoped ``/tmp`` dirs
-- never the repo tree. No network, no weight execution: weight files get
byte-level header validation only, exactly like the audited code.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from model_protocol_lab.doctor import doctor, examine_model

HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None
HAS_TOKENIZERS = importlib.util.find_spec("tokenizers") is not None
NEEDS_HF = unittest.skipUnless(
    HAS_TRANSFORMERS and HAS_TOKENIZERS, "transformers/tokenizers not installed"
)

# 128 MiB bound enforced by backends._read_json, plus one byte.
_OVERSIZED = 128 * 1024 * 1024 + 1


def _valid_safetensors_bytes() -> bytes:
    """Minimal structurally-valid safetensors payload (header only matters)."""
    header = {
        "__metadata__": {},
        "weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
    }
    header_bytes = json.dumps(header).encode("utf-8")
    return struct.pack("<Q", len(header_bytes)) + header_bytes + b"\x00" * 4


def _write_loadable_tokenizer(root: Path, vocab_size: int = 16) -> int:
    """Genuinely loadable WordLevel tokenizer.json; returns vocab size."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    vocab = {f"tok{i}": i for i in range(vocab_size - 1)}
    vocab["[UNK]"] = vocab_size - 1
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.add_special_tokens(["[UNK]", "[BOS]", "[EOS]"])
    tokenizer.save(str(root / "tokenizer.json"))
    return len(tokenizer.get_vocab())


def _write_tokenizer_doc(root: Path, doc: dict) -> None:
    (root / "tokenizer.json").write_text(json.dumps(doc), encoding="utf-8")


def _write_config(root: Path, obj: object) -> None:
    (root / "config.json").write_text(json.dumps(obj), encoding="utf-8")


def _healthy(
    root: Path,
    *,
    model_type: str = "gpt2",
    config_extra: dict | None = None,
    weights: bool = True,
    tokenizer: bool = True,
) -> Path:
    """Fully valid synthetic model dir (config + weights + loadable tokenizer)."""
    config: dict = {"model_type": model_type}
    config.update(config_extra or {})
    _write_config(root, config)
    if weights:
        (root / "model.safetensors").write_bytes(_valid_safetensors_bytes())
    if tokenizer:
        _write_loadable_tokenizer(root)
    return root


def _write_sharded_layout(root: Path, shards: dict[str, bytes]) -> None:
    """Sharded (index.json) layout with caller-controlled shard payloads."""
    _write_config(root, {"model_type": "gpt2"})
    index = {
        "metadata": {},
        "weight_map": {f"tensor{i}": name for i, name in enumerate(shards)},
    }
    (root / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    for name, payload in shards.items():
        (root / name).write_bytes(payload)


def _write_dataset(path: Path, *, special_literal: bool = False) -> None:
    def item(name: str) -> dict:
        # Content is unique per fixture: identical inputs may not cross the
        # search/held_out split (anti-contamination rule in datasets).
        content = f"say <|im_start|> {name} out loud" if special_literal else f"question:{name}"
        return {
            "id": name,
            "group": name,
            "messages": [{"role": "user", "content": content}],
            "expected": "ok",
        }

    split = {
        "search": [item("s1"), item("s2")],
        "held_out": [item("h1"), item("h2")],
    }
    path.write_text(json.dumps(split), encoding="utf-8")


class _Base(unittest.TestCase):
    def mkroot(self) -> Path:
        tmp = Path(tempfile.mkdtemp(prefix=f"mpl-c72-{os.getpid()}-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        return tmp

    def healthy(self, **kwargs) -> Path:
        return _healthy(self.mkroot(), **kwargs)

    def checks(self, report: dict) -> dict:
        return report["model"]["checks"]


class ReportAssemblyTests(_Base):
    """Check ordering, result shape, determinism, fail-fast vs. continue."""

    @NEEDS_HF
    def test_full_pass_check_order_is_stable(self):
        root = self.healthy()
        report = examine_model(
            root,
            boundary_ids=(0, 1),
            boundary_map={"alpha": (0,), "beta": (1, 2)},
            reference_model_path=root,
        )
        self.assertEqual(report["verdict"], "ok")
        self.assertEqual(
            list(self.checks(report)),
            [
                "directory",
                "asset_layout",
                "asset_manifest",
                "weight_headers",
                "config",
                "tokenizer_file",
                "tokenizer_deep",
                "boundary_ids",
                "boundary_ids:alpha",
                "boundary_ids:beta",
                "boundary_portability",
            ],
        )

    def test_every_return_carries_the_same_result_keys(self):
        """All early-failure returns keep the full model-result shape."""
        variants = {
            "bad directory": lambda root: Path(root) / "missing",
            "no weights": lambda root: _healthy(root, weights=False),
            "corrupt weights": lambda root: (
                _healthy(root),
                (root / "model.safetensors").write_bytes(b"garbage" * 32),
            )[0],
            "bad config": lambda root: (
                _healthy(root),
                (root / "config.json").write_text("not json", encoding="utf-8"),
            )[0],
            "no tokenizer": lambda root: _healthy(root, tokenizer=False),
        }
        for label, build in variants.items():
            with self.subTest(label=label):
                root = build(self.mkroot())
                report = examine_model(root)
                self.assertEqual(report["verdict"], "failed")
                model = report["model"]
                self.assertEqual(
                    set(model),
                    {"checks", "assets", "special_tokens", "added_tokens", "context_window"},
                )
                self.assertIs(type(model["assets"]), list)
                self.assertIs(type(model["special_tokens"]), dict)
                self.assertIs(type(model["added_tokens"]), dict)

    def test_report_is_deterministic_and_json_serializable(self):
        root = self.healthy()
        first = json.dumps(examine_model(root, boundary_ids=(0,)), sort_keys=True)
        second = json.dumps(examine_model(root, boundary_ids=(0,)), sort_keys=True)
        self.assertEqual(first, second)
        # The CLI prints the report as JSON; it must always serialize.
        json.dumps(examine_model(root))

    def test_weight_header_failure_aborts_before_config_stage(self):
        """Fail-fast: a corrupt header means the config stage never runs."""
        root = self.healthy()
        (root / "model.safetensors").write_bytes(b"\x00" * 32)
        (root / "config.json").write_text("also broken", encoding="utf-8")
        report = examine_model(root)
        checks = self.checks(report)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(checks["weight_headers"]["status"], "failed")
        self.assertTrue(
            checks["weight_headers"]["detail"].startswith("invalid_safetensors_header: ")
        )
        self.assertNotIn("config", checks)
        self.assertNotIn("tokenizer_file", checks)

    def test_tokenizer_file_failure_does_not_abort_deep_stage(self):
        """Unlike weight/config failures, a bad tokenizer doc still runs deep checks."""
        root = self.healthy()
        (root / "tokenizer.json").write_text("{}", encoding="utf-8")
        report = examine_model(root)
        checks = self.checks(report)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(checks["tokenizer_file"]["status"], "failed")
        self.assertIn("missing", checks["tokenizer_file"]["detail"])
        # The deep stage still ran (failed with transformers, skipped without).
        self.assertIn(checks["tokenizer_deep"]["status"], ("failed", "skipped"))

    def test_manifest_failure_still_records_asset_layout_ok(self):
        """Regression: asset enumeration succeeded, so asset_layout must be ok.

        The manifest stage can fail after assets were enumerated (TOCTOU
        races, vanishing files); the completed asset_layout stage must still
        be reported instead of silently dropped.
        """
        root = self.healthy()
        with patch("model_protocol_lab.doctor._manifest", side_effect=ValueError("assets_changed")):
            report = examine_model(root)
        checks = self.checks(report)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(list(checks), ["directory", "asset_layout", "asset_manifest"])
        self.assertEqual(checks["asset_layout"], {"status": "ok", "detail": "2 assets"})
        self.assertEqual(checks["asset_manifest"]["status"], "failed")

    def test_assets_present_on_config_failure_but_absent_without_weights(self):
        """Manifest records exist once assets were enumerated, even if config fails."""
        root = self.healthy()
        (root / "config.json").write_text("broken", encoding="utf-8")
        report = examine_model(root)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(
            [a["name"] for a in report["model"]["assets"]],
            ["config.json", "model.safetensors"],
        )
        # Without weights, asset enumeration itself failed: no records at all.
        bare = _healthy(self.mkroot(), weights=False)
        bare_report = examine_model(bare)
        self.assertEqual(bare_report["model"]["assets"], [])
        self.assertNotIn("asset_manifest", self.checks(bare_report))


class VerdictPrecedenceTests(_Base):
    """Minimal pairs: which failure wins, and with what detail vocabulary."""

    def _shard_pair(self, corrupt_a: bool, corrupt_b: bool) -> Path:
        root = self.mkroot()
        _write_sharded_layout(
            root,
            {
                "shard-a.safetensors": b"nope" if corrupt_a else _valid_safetensors_bytes(),
                "shard-b.safetensors": b"nope" if corrupt_b else _valid_safetensors_bytes(),
            },
        )
        return root

    def test_one_corrupt_shard_vs_all_corrupt(self):
        """The first corrupt shard in sorted-name order is the one reported."""
        only_b = examine_model(self._shard_pair(False, True))
        self.assertEqual(only_b["verdict"], "failed")
        self.assertEqual(
            self.checks(only_b)["weight_headers"]["detail"],
            "invalid_safetensors_header: shard-b.safetensors",
        )
        both = examine_model(self._shard_pair(True, True))
        self.assertEqual(both["verdict"], "failed")
        self.assertEqual(
            self.checks(both)["weight_headers"]["detail"],
            "invalid_safetensors_header: shard-a.safetensors",
        )
        neither = examine_model(self._shard_pair(False, False))
        self.assertEqual(neither["verdict"], "failed")  # no tokenizer.json written
        self.assertEqual(self.checks(neither)["weight_headers"]["status"], "ok")

    def test_sharded_layout_full_pass_reports_sorted_assets(self):
        root = self.mkroot()
        _write_sharded_layout(
            root,
            {
                "shard-b.safetensors": _valid_safetensors_bytes(),
                "shard-a.safetensors": _valid_safetensors_bytes(),
            },
        )
        _write_loadable_tokenizer(root)
        report = examine_model(root)
        self.assertEqual(report["verdict"], "ok")
        self.assertEqual(
            [a["name"] for a in report["model"]["assets"]],
            [
                "config.json",
                "model.safetensors.index.json",
                "shard-a.safetensors",
                "shard-b.safetensors",
            ],
        )

    def test_missing_config_vs_malformed_config(self):
        """config.json is a required asset first, a config document second.

        A missing config.json breaks the asset stages (layout or manifest);
        a present-but-unparseable one breaks the config stage. The failing
        stage and the detail vocabulary differ accordingly.
        """
        missing_root = self.healthy()
        (missing_root / "config.json").unlink()
        missing = examine_model(missing_root)
        malformed_root = self.healthy()
        (malformed_root / "config.json").write_bytes(b"\xff\xfe broken")
        malformed = examine_model(malformed_root)
        for report in (missing, malformed):
            self.assertEqual(report["verdict"], "failed")
        missing_checks = self.checks(missing)
        self.assertNotIn("config", missing_checks)
        missing_detail = (
            missing_checks.get("asset_layout", {}).get("detail")
            or missing_checks["asset_manifest"]["detail"]
        )
        self.assertEqual(missing_detail, "missing_or_unsafe_asset")
        self.assertEqual(missing["model"]["assets"], [])
        malformed_checks = self.checks(malformed)
        self.assertEqual(malformed_checks["config"]["detail"], "invalid_asset_json")
        self.assertNotEqual(malformed["model"]["assets"], [])
        # A JSON non-object is malformed too, not "missing".
        other = self.healthy()
        (other / "config.json").write_text("[1, 2]", encoding="utf-8")
        self.assertEqual(
            self.checks(examine_model(other))["config"]["detail"], "invalid_asset_json"
        )

    def test_unsupported_model_type_takes_precedence_over_rejected_keys(self):
        root = self.healthy(model_type="bert", config_extra={"quantization_config": {"bits": 4}})
        report = examine_model(root)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(self.checks(report)["config"]["detail"], "unsupported model_type: 'bert'")

    def test_rejected_keys_listed_in_fixed_order(self):
        root = self.healthy(
            config_extra={"is_encoder_decoder": True, "quantization_config": {"bits": 4}}
        )
        report = examine_model(root)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(
            self.checks(report)["config"]["detail"],
            "rejected config keys: quantization_config, is_encoder_decoder",
        )

    def test_falsy_rejected_keys_do_not_fail(self):
        """The rejection gate is truthiness-based: empty/false values pass."""
        root = self.healthy(config_extra={"quantization_config": {}, "is_encoder_decoder": False})
        report = examine_model(root)
        self.assertEqual(self.checks(report)["config"]["status"], "ok")

    def test_missing_model_type_is_unsupported(self):
        root = self.mkroot()
        _write_config(root, {})
        (root / "model.safetensors").write_bytes(_valid_safetensors_bytes())
        report = examine_model(root)
        self.assertEqual(self.checks(report)["config"]["detail"], "unsupported model_type: None")

    def test_tokenizer_dropout_variants(self):
        """dropout>0 fails; 0/absent pass the tokenizer_file check."""
        for dropout, ok in ((0.5, False), (0, True), (0.0, True)):
            with self.subTest(dropout=dropout):
                root = self.mkroot()
                _healthy(root)
                doc = json.loads((root / "tokenizer.json").read_text(encoding="utf-8"))
                doc["model"]["dropout"] = dropout
                _write_tokenizer_doc(root, doc)
                checks = self.checks(examine_model(root))
                if ok:
                    self.assertEqual(checks["tokenizer_file"]["status"], "ok")
                else:
                    self.assertEqual(checks["tokenizer_file"]["status"], "failed")
                    self.assertEqual(
                        checks["tokenizer_file"]["detail"],
                        "stochastic tokenizer not supported",
                    )


class ContextWindowTests(_Base):
    """The declared window is informational: malformed values degrade to None."""

    def _ctx(self, **kwargs) -> object:
        root = self.healthy(tokenizer=False, **kwargs)
        return examine_model(root)["model"]["context_window"]

    def test_gpt2_uses_n_positions(self):
        self.assertEqual(self._ctx(config_extra={"n_positions": 1024}), 1024)

    def test_llama_uses_max_position_embeddings(self):
        self.assertEqual(
            self._ctx(model_type="llama", config_extra={"max_position_embeddings": 2048}),
            2048,
        )

    def test_malformed_window_degrades_to_none_without_failing(self):
        for extra in (
            {"n_positions": "1024"},
            {"n_positions": 0},
            {"n_positions": -5},
            {"n_positions": 10.5},
        ):
            with self.subTest(extra=extra):
                root = self.healthy(tokenizer=False, config_extra=extra)
                report = examine_model(root)
                self.assertIsNone(report["model"]["context_window"])
                self.assertEqual(self.checks(report)["config"]["status"], "ok")

    def test_window_survives_tokenizer_stage_failure(self):
        """context_window comes from config, so it is set even without a tokenizer."""
        root = self.healthy(tokenizer=False, config_extra={"n_positions": 512})
        report = examine_model(root)
        self.assertEqual(report["verdict"], "failed")  # tokenizer_file failed
        self.assertEqual(report["model"]["context_window"], 512)


class BoundaryIntegrationTests(_Base):
    """boundary_ids / boundary_map validation against the measured vocabulary."""

    @NEEDS_HF
    def test_boundary_ids_valid_ok(self):
        root = self.healthy()
        report = examine_model(root, boundary_ids=(0, 1))
        self.assertEqual(self.checks(report)["boundary_ids"]["status"], "ok")

    @NEEDS_HF
    def test_boundary_ids_invalid_variants(self):
        root = self.healthy()
        for label, ids in (
            ("empty tuple", ()),
            ("negative id", (0, -1)),
            ("bool element", (True, 0)),
            ("non-iterable", 5),
            ("string element", ("0",)),
        ):
            with self.subTest(label=label):
                report = examine_model(root, boundary_ids=ids)
                self.assertEqual(report["verdict"], "failed")
                self.assertEqual(
                    self.checks(report)["boundary_ids"]["detail"], "invalid boundary ids"
                )

    @NEEDS_HF
    def test_boundary_map_per_name_reporting(self):
        root = self.healthy()
        report = examine_model(
            root,
            boundary_map={"good": (0, 1), "": (0,), 123: (0,), "bad": (10**9,)},
        )
        checks = self.checks(report)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(checks["boundary_ids:good"]["status"], "ok")
        for name in ("boundary_ids:", "boundary_ids:123", "boundary_ids:bad"):
            with self.subTest(name=name):
                self.assertEqual(checks[name]["status"], "failed")
        # One bad mapping cannot hide behind the default boundary_ids check.
        self.assertNotIn("boundary_ids", checks)

    def test_boundary_map_non_dict_fails(self):
        root = self.healthy()
        report = examine_model(root, boundary_map="not-a-map")  # type: ignore[arg-type]
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(self.checks(report)["boundary_map"]["detail"], "invalid boundary map")

    def test_boundary_map_empty_dict_adds_no_checks(self):
        root = self.healthy()
        report = examine_model(root, boundary_map={})
        self.assertNotIn("boundary_map", self.checks(report))
        self.assertFalse([k for k in self.checks(report) if k.startswith("boundary_ids:")])

    def test_boundary_map_skipped_without_transformers(self):
        root = self.healthy()
        with patch.dict(sys.modules, {"transformers": None}):
            report = examine_model(root, boundary_map={"a": (0,), "b": (1,)})
        checks = self.checks(report)
        self.assertEqual(report["verdict"], "ok")
        self.assertEqual(checks["tokenizer_deep"]["status"], "skipped")
        self.assertEqual(checks["boundary_ids:a"]["status"], "skipped")
        self.assertEqual(checks["boundary_ids:b"]["status"], "skipped")

    @NEEDS_HF
    def test_reference_portability_never_fails_verdict(self):
        """Portability is informational: mismatch/skip leave the verdict alone."""
        root = self.healthy()
        same = examine_model(root, boundary_ids=(0, 1), reference_model_path=root)
        self.assertEqual(same["verdict"], "ok")
        self.assertIn(self.checks(same)["boundary_portability"]["status"], ("ok", "mismatch"))
        # A reference without a usable tokenizer degrades to skipped.
        weights_only = self.healthy(tokenizer=False)
        degraded = examine_model(root, boundary_ids=(0, 1), reference_model_path=weights_only)
        self.assertEqual(degraded["verdict"], "ok")
        self.assertEqual(
            self.checks(degraded)["boundary_portability"],
            {"status": "skipped", "detail": "reference_tokenizer_unavailable"},
        )
        # No boundary ids: the check explains it needs them.
        no_ids = examine_model(root, reference_model_path=root)
        self.assertEqual(
            self.checks(no_ids)["boundary_portability"]["detail"], "needs --boundary-ids"
        )
        self.assertEqual(no_ids["verdict"], "ok")


class AdversarialEdgeTests(_Base):
    """Empty dirs, 0-byte files, garbage, oversized configs, odd paths."""

    def test_empty_directory(self):
        root = self.mkroot()
        report = examine_model(root)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(
            self.checks(report)["asset_layout"]["detail"],
            "require_one_standard_safetensors_layout",
        )
        self.assertEqual(report["model"]["assets"], [])

    def test_zero_byte_files(self):
        expectations = {
            "model.safetensors": ("weight_headers", "invalid_safetensors_header"),
            "config.json": ("config", "invalid_asset_json"),
            "tokenizer.json": ("tokenizer_file", "invalid_asset_json"),
        }
        for name, (check, detail_prefix) in expectations.items():
            with self.subTest(file=name):
                root = self.healthy()
                (root / name).write_bytes(b"")
                report = examine_model(root)
                self.assertEqual(report["verdict"], "failed")
                self.assertTrue(
                    self.checks(report)[check]["detail"].startswith(detail_prefix),
                    self.checks(report)[check]["detail"],
                )

    def test_garbage_everywhere_fails_at_weights_first(self):
        root = self.healthy()
        for name in ("model.safetensors", "config.json", "tokenizer.json"):
            (root / name).write_bytes(os.urandom(64))
        report = examine_model(root)
        checks = self.checks(report)
        self.assertEqual(checks["weight_headers"]["status"], "failed")
        self.assertNotIn("config", checks)  # fail-fast: later stages never ran

    def test_header_variants(self):
        cases = {
            "json list header": struct.pack("<Q", 7) + b"[1,2,3]",
            "declared size exceeds file": struct.pack("<Q", 100) + b"short",
            "zero-length header": struct.pack("<Q", 0),
            "truncated prefix": b"\x01\x02\x03",
        }
        for label, payload in cases.items():
            with self.subTest(label=label):
                root = self.healthy()
                (root / "model.safetensors").write_bytes(payload)
                report = examine_model(root)
                self.assertEqual(report["verdict"], "failed")
                self.assertEqual(
                    self.checks(report)["weight_headers"]["detail"],
                    "invalid_safetensors_header: model.safetensors",
                )

    def test_oversized_config_rejected(self):
        root = self.healthy()
        # Sparse file: _read_json reads byte content (holes read as zeros),
        # so the oversized path triggers while consuming ~0 tmpfs blocks.
        # A real 128 MiB write flakes with ENOSPC on the 512 MiB /tmp tmpfs.
        with (root / "config.json").open("wb") as handle:
            handle.truncate(_OVERSIZED)
        report = examine_model(root)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(self.checks(report)["config"]["detail"], "oversized_asset_json")

    def test_unicode_model_path(self):
        root = self.mkroot() / "μοντέλο-🚀"
        root.mkdir()
        _healthy(root)
        report = examine_model(root)
        self.assertEqual(report["verdict"], "ok")
        self.assertEqual(self.checks(report)["directory"]["detail"], str(root.resolve()))

    def test_symlinked_model_dir_resolves(self):
        real = self.healthy()
        link = self.mkroot() / "link-to-model"
        link.symlink_to(real, target_is_directory=True)
        report = examine_model(link)
        self.assertEqual(report["verdict"], "ok")
        self.assertEqual(self.checks(report)["directory"]["detail"], str(real.resolve()))

    def test_symlink_loop_fails_closed(self):
        root = self.mkroot()
        (root / "a").symlink_to(root / "b")
        (root / "b").symlink_to(root / "a")
        report = examine_model(root / "a")
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(
            self.checks(report)["directory"]["detail"], "not a readable local directory"
        )

    def test_model_path_being_a_file_fails(self):
        root = self.mkroot()
        target = root / "file"
        target.write_text("x", encoding="utf-8")
        report = examine_model(target)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(
            self.checks(report)["directory"]["detail"], "not a readable local directory"
        )

    def test_single_and_index_layouts_together_rejected(self):
        root = self.healthy()
        (root / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
        report = examine_model(root)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(
            self.checks(report)["asset_layout"]["detail"],
            "require_one_standard_safetensors_layout",
        )

    def test_invalid_weight_index_variants(self):
        for label, weight_map in (
            ("missing weight_map", None),
            ("empty weight_map", {}),
            ("non-dict weight_map", ["shard.safetensors"]),
            ("unsafe shard name", {"w": "../evil.safetensors"}),
        ):
            with self.subTest(label=label):
                root = self.mkroot()
                _write_config(root, {"model_type": "gpt2"})
                index: dict = {}
                if weight_map is not None:
                    index["weight_map"] = weight_map
                (root / "model.safetensors.index.json").write_text(
                    json.dumps(index), encoding="utf-8"
                )
                report = examine_model(root)
                self.assertEqual(report["verdict"], "failed")
                self.assertEqual(
                    self.checks(report)["asset_layout"]["detail"], "invalid_weight_index"
                )

    def test_adapter_layout_rejected(self):
        root = self.healthy()
        (root / "adapter_config.json").write_text("{}", encoding="utf-8")
        report = examine_model(root)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(self.checks(report)["asset_layout"]["detail"], "adapters_not_supported")

    @unittest.skipIf(os.geteuid() == 0, "root bypasses file permission checks; nothing to deny")
    def test_unreadable_config_fails_closed(self):
        root = self.healthy()
        (root / "config.json").chmod(0o000)
        try:
            report = examine_model(root)
        finally:
            (root / "config.json").chmod(0o644)
        # The manifest stage hashes every asset before the config stage
        # reads it, so an unreadable config.json fails the manifest check
        # (fail-closed with the fixed vocabulary), not the config check.
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(self.checks(report)["asset_manifest"]["detail"], "missing_or_unsafe_asset")

    def _denying_open(self, name: str, skip_first: int = 0):
        """Path.open replacement raising PermissionError for one file name.

        ``skip_first`` allows the first N opens to succeed: the manifest
        stage hashes every asset before the config stage reads it, so
        reaching the config try-block requires config.json's manifest open
        to succeed (skip_first=1).
        """
        real_open = Path.open
        calls = 0

        def fake_open(self: Path, *args: object, **kwargs: object):  # noqa: ANN003
            nonlocal calls
            if self.name == name:
                calls += 1
                if calls > skip_first:
                    raise PermissionError(13, "Permission denied")
            return real_open(self, *args, **kwargs)

        return fake_open

    def test_unreadable_config_fails_closed_without_root(self):
        """Root-independent regression test for the config OSError hole.

        The chmod-based test above self-skips when uid 0 (root bypasses
        permission checks); this one mocks Path.open so it runs everywhere,
        including CI. The manifest open succeeds and the config read raises,
        exercising the config try-block's ``except OSError`` branch. Pre-fix,
        the PermissionError escaped examine_model instead of failing closed.
        """
        root = self.healthy()
        with patch.object(Path, "open", self._denying_open("config.json", skip_first=1)):
            report = examine_model(root)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(self.checks(report)["config"]["detail"], "missing_or_unsafe_asset")

    def test_unreadable_weight_index_fails_closed_without_root(self):
        """Same OSError hole class in the _model_assets try-block.

        In the sharded layout _model_assets reads the weight index via
        _read_json; an unreadable index must fail the asset_layout check
        with the fixed vocabulary instead of escaping.
        """
        root = self.mkroot()
        _write_sharded_layout(root, {"shard0.safetensors": _valid_safetensors_bytes()})
        with patch.object(Path, "open", self._denying_open("model.safetensors.index.json")):
            report = examine_model(root)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(self.checks(report)["asset_layout"]["detail"], "missing_or_unsafe_asset")

    def test_unreadable_tokenizer_fails_closed_without_root(self):
        """Same OSError hole class in the tokenizer.json try-block."""
        root = self.healthy()
        with patch.object(Path, "open", self._denying_open("tokenizer.json")):
            report = examine_model(root)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(self.checks(report)["tokenizer_file"]["detail"], "missing_or_unsafe_asset")

    def test_unreadable_weight_during_manifest_fails_closed_without_root(self):
        """Same OSError hole class in the _manifest try-block."""
        root = self.healthy()
        with patch.object(Path, "open", self._denying_open("model.safetensors")):
            report = examine_model(root)
        self.assertEqual(report["verdict"], "failed")
        checks = self.checks(report)
        self.assertEqual(checks["asset_layout"], {"status": "ok", "detail": "2 assets"})
        self.assertEqual(checks["asset_manifest"]["detail"], "missing_or_unsafe_asset")


class DoctorOrchestrationTests(_Base):
    """How doctor() combines model, dataset, matrix and suggestion sections."""

    def _dataset(self, **kwargs) -> Path:
        path = self.mkroot() / "ds.json"
        _write_dataset(path, **kwargs)
        return path

    def test_doctor_without_dataset_has_no_dataset_section(self):
        root = self.healthy()
        report = doctor(root)
        self.assertEqual(report["verdict"], "ok")
        self.assertNotIn("dataset", report)
        self.assertNotIn("portability_matrix", report)
        self.assertIsNone(report["suggestion"])

    def test_doctor_dataset_failure_forces_overall_failure(self):
        root = self.healthy(config_extra={"n_positions": 1024})
        bad = self.mkroot() / "bad.json"
        bad.write_text("{}", encoding="utf-8")
        report = doctor(root, bad, boundary_ids=(0, 1))
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(report["dataset"]["status"], "failed")
        # A failed pre-flight never suggests a run command.
        self.assertIsNone(report["suggestion"])

    def test_doctor_dataset_ok_reports_counts(self):
        root = self.healthy()
        report = doctor(root, self._dataset())
        self.assertEqual(report["verdict"], "ok")
        dataset = report["dataset"]
        self.assertEqual(dataset["status"], "ok")
        self.assertEqual(dataset["search_cases"], 2)
        self.assertEqual(dataset["held_out_cases"], 2)
        self.assertEqual(dataset["special_token_audit"]["status"], "ok")

    def test_doctor_dataset_warning_does_not_fail_verdict(self):
        """A special-token audit warning is advisory, like in fixtures build."""
        root = self.healthy()
        report = doctor(root, self._dataset(special_literal=True))
        self.assertEqual(report["verdict"], "ok")
        audit = report["dataset"]["special_token_audit"]
        self.assertEqual(audit["status"], "warning")
        self.assertTrue(audit["findings"])
        self.assertIn("<|im_start|>", {f["literal"] for f in audit["findings"]})

    def test_doctor_model_failure_with_good_dataset_stays_failed(self):
        root = self.healthy()
        (root / "model.safetensors").write_bytes(b"corrupt")
        report = doctor(root, self._dataset(), boundary_ids=(0, 1))
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(report["dataset"]["status"], "ok")
        self.assertIsNone(report["suggestion"])

    @NEEDS_HF
    def test_doctor_suggestion_when_preflight_clean(self):
        root = self.healthy(config_extra={"n_positions": 1024})
        report = doctor(root, self._dataset(), boundary_ids=(0, 1))
        self.assertEqual(report["verdict"], "ok")
        suggestion = report["suggestion"]
        self.assertIs(type(suggestion), dict)
        assert type(suggestion) is dict  # for mypy narrowing
        command = suggestion["next_command"]
        self.assertIn("--max-context 1024", command)
        self.assertIn("--boundary-ids 0,1", command)
        self.assertIn("--family FAMILY", command)
        self.assertIn("1024", suggestion["note"])

    @NEEDS_HF
    def test_doctor_suggestion_absent_unless_everything_known(self):
        good = self.healthy(config_extra={"n_positions": 1024})
        no_window = self.healthy()
        dataset = self._dataset()
        cases = {
            "failed model": (self.healthy(), dataset, (0, 1)),
            "no dataset": (good, None, (0, 1)),
            "no boundary ids": (good, dataset, None),
            "no context window": (no_window, dataset, (0, 1)),
        }
        for label, (model, ds, ids) in cases.items():
            with self.subTest(label=label):
                if label == "failed model":
                    (model / "model.safetensors").write_bytes(b"corrupt")
                report = doctor(model, ds, boundary_ids=ids)
                self.assertIsNone(report["suggestion"])

    def test_doctor_portability_matrix_failure_degrades_section(self):
        """A broken second checkpoint degrades the section, never the verdict."""
        root = self.healthy()
        empty = self.mkroot()
        report = doctor(root, portability_matrix_path=empty)
        self.assertEqual(report["verdict"], "ok")
        section = report["portability_matrix"]
        self.assertEqual(section["status"], "unavailable")
        self.assertTrue(section["rows"] == [])
        # A non-path matrix argument degrades the same way, without raising.
        report2 = doctor(root, portability_matrix_path=123)  # type: ignore[arg-type]
        self.assertEqual(report2["verdict"], "ok")
        self.assertEqual(report2["portability_matrix"]["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
