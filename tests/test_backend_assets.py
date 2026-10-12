"""Asset-validation tests independent of optional model dependencies."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from model_protocol_lab.backends import _ids_valid, _manifest, _member, _model_assets, _read_json


class AssetTests(unittest.TestCase):
    def test_ids_require_exact_integer_tuple_and_vocabulary_bounds(self):
        self.assertTrue(_ids_valid((0, 3), 4))
        for value in ([0], (True,), (-1,), (4,), (1.0,), None):
            with self.subTest(value=value):
                self.assertFalse(_ids_valid(value, 4))

    def test_member_rejects_paths_and_missing_assets(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("../config.json", "sub/config.json", "sub\\config.json", "/config.json", "missing.json"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    _member(root, name)

    def test_json_requires_object(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text("[]", encoding="utf-8")
            with self.assertRaises(ValueError):
                _read_json(path)

    def test_single_layout_and_manifest_are_path_independent(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text("{}", encoding="utf-8")
            (root / "model.safetensors").write_bytes(b"synthetic asset bytes")
            names = _model_assets(root)
            self.assertEqual(names, ("config.json", "model.safetensors"))
            records = _manifest(root, names)
            self.assertEqual([record["name"] for record in records], list(names))
            self.assertNotIn(tmp, json.dumps(records))
            self.assertEqual(len(records[0]["sha256"]), 64)
            self.assertEqual(records, _manifest(root, names))

    def test_conflicting_layouts_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model.safetensors").write_bytes(b"")
            (root / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                _model_assets(root)

    def test_adapter_directory_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "adapter_config.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                _model_assets(root)

    def test_sharded_layout_rejects_traversal(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text("{}", encoding="utf-8")
            index = root / "model.safetensors.index.json"
            for name in ("../weights.safetensors", "sub/weights.safetensors", "weights.bin"):
                index.write_text(json.dumps({"weight_map": {"weight": name}}), encoding="utf-8")
                with self.subTest(name=name), self.assertRaises(ValueError):
                    _model_assets(root)

    def test_sharded_layout_requires_every_shard(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text("{}", encoding="utf-8")
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"weight": "part.safetensors"}}), encoding="utf-8")
            with self.assertRaises(ValueError):
                _model_assets(root)
            (root / "part.safetensors").write_bytes(b"synthetic shard")
            self.assertIn("part.safetensors", _model_assets(root))
