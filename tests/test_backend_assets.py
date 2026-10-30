"""Asset-validation tests independent of optional model dependencies."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from model_protocol_lab.backends import _ids_valid, _manifest, _member, _model_assets, _read_json


class AssetTests(unittest.TestCase):
    """Asset tests."""

    def test_ids_require_exact_integer_tuple_and_vocabulary_bounds(self):
        """Test ids require exact integer tuple and vocabulary bounds."""
        self.assertTrue(_ids_valid((0, 3), 4))
        for value in ([0], (True,), (-1,), (4,), (1.0,), None):
            with self.subTest(value=value):
                self.assertFalse(_ids_valid(value, 4))

    def test_member_rejects_paths_and_missing_assets(self):
        """Test member rejects paths and missing assets."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in (
                "../config.json",
                "sub/config.json",
                "sub\\config.json",
                "/config.json",
                "missing.json",
            ):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    _member(root, name)

    def test_json_requires_object(self):
        """Test json requires object."""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text("[]", encoding="utf-8")
            with self.assertRaises(ValueError):
                _read_json(path)

    def test_single_layout_and_manifest_are_path_independent(self):
        """Test single layout and manifest are path independent."""
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
        """Test conflicting layouts rejected."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model.safetensors").write_bytes(b"")
            (root / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                _model_assets(root)

    def test_adapter_directory_rejected(self):
        """Test adapter directory rejected."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "adapter_config.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                _model_assets(root)

    def test_sharded_layout_rejects_traversal(self):
        """Test sharded layout rejects traversal."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text("{}", encoding="utf-8")
            index = root / "model.safetensors.index.json"
            for name in ("../weights.safetensors", "sub/weights.safetensors", "weights.bin"):
                index.write_text(json.dumps({"weight_map": {"weight": name}}), encoding="utf-8")
                with self.subTest(name=name), self.assertRaises(ValueError):
                    _model_assets(root)

    def test_sharded_layout_requires_every_shard(self):
        """Test sharded layout requires every shard."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text("{}", encoding="utf-8")
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"weight": "part.safetensors"}}), encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                _model_assets(root)
            (root / "part.safetensors").write_bytes(b"synthetic shard")
            self.assertIn("part.safetensors", _model_assets(root))


class WeightIndexEdgeTests(unittest.TestCase):
    """Weight index edge tests."""

    @staticmethod
    def _root_with_index(tmp, weight_map):
        """Root with index."""
        from pathlib import Path as _Path

        root = _Path(tmp)
        (root / "config.json").write_text("{}", encoding="utf-8")
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map}), encoding="utf-8"
        )
        return root

    def test_empty_and_nondict_weight_maps_rejected(self):
        """Test empty and nondict weight maps rejected."""
        from tempfile import TemporaryDirectory as _TD

        for mapping in ({}, [], "x", None, 0):
            with self.subTest(mapping=mapping), _TD() as tmp:
                root = self._root_with_index(tmp, mapping)
                with self.assertRaises(ValueError):
                    _model_assets(root)

    def test_oversized_asset_json_rejected(self):
        """Test oversized asset json rejected."""
        from pathlib import Path as _Path
        from tempfile import TemporaryDirectory as _TD

        with _TD() as tmp:
            path = _Path(tmp) / "big.json"
            # Sparse file: reports a huge size without consuming disk, so this
            # stays reliable on small tmpfs mounts.
            with path.open("wb") as handle:
                handle.write(b'{"k": "')
                handle.truncate(128 * 1024 * 1024 + 1)
            with self.assertRaises(ValueError):
                _read_json(path)
