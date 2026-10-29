"""Cycle 115: manifest fingerprint audit.

The backend used to sha256-hash every weight file twice per run (once before
and once after transformers loaded the model) to detect assets changing
during load. For multi-gigabyte checkpoints that is a full extra read of
every weight file. The backend now hashes once and re-verifies afterwards
with the cheap ``(st_size, st_mtime_ns, st_ino)`` stat fingerprint captured
while hashing.

These tests pin the contract:
* ``_manifest_with_fingerprints`` records are byte-identical to ``_manifest``
  records (the report schema is unchanged).
* ``_verify_manifest_fingerprints`` passes on untouched assets.
* It fails closed with ``ValueError("assets_changed")`` on in-place
  modification, file replacement (new inode), and deletion.
* The backend no longer calls the full ``_manifest`` itself (single hash),
  and its ``identity["assets"]`` still carries the sha256 records.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from model_protocol_lab.backends import (
    _manifest,
    _manifest_with_fingerprints,
    _verify_manifest_fingerprints,
)


def _write(path: Path, data: bytes) -> None:
    """Write."""
    path.write_bytes(data)


class TestManifestFingerprintEquivalence(unittest.TestCase):
    """Test manifest fingerprint equivalence."""

    def test_records_match_plain_manifest(self) -> None:
        """Test records match plain manifest."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "a.safetensors", b"\x00" * 100)
            _write(root / "b.safetensors", b"\x01" * 200)
            names = ("a.safetensors", "b.safetensors")
            records, prints = _manifest_with_fingerprints(root, names)
            self.assertEqual(records, _manifest(root, names))
            self.assertEqual(set(prints), set(names))
            for record in records:
                self.assertEqual(set(record), {"name", "bytes", "sha256"}, "record schema changed")

    def test_fingerprint_is_stat_triple(self) -> None:
        """Test fingerprint is stat triple."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "a.safetensors", b"data")
            _, prints = _manifest_with_fingerprints(root, ("a.safetensors",))
            stat = (root / "a.safetensors").stat()
            self.assertEqual(prints["a.safetensors"], (stat.st_size, stat.st_mtime_ns, stat.st_ino))


class TestVerifyManifestFingerprints(unittest.TestCase):
    """Test verify manifest fingerprints."""

    @staticmethod
    def _root(tmp: str) -> tuple[Path, tuple[str, ...], dict[str, tuple[int, int, int]]]:
        """Root."""
        root = Path(tmp)
        _write(root / "w.safetensors", b"\x02" * 64)
        names = ("w.safetensors",)
        _, prints = _manifest_with_fingerprints(root, names)
        return root, names, prints

    def test_unchanged_passes(self) -> None:
        """Test unchanged passes."""
        with tempfile.TemporaryDirectory() as tmp:
            root, names, prints = self._root(tmp)
            _verify_manifest_fingerprints(root, names, prints)  # must not raise

    def test_in_place_modification_detected(self) -> None:
        """Test in place modification detected."""
        with tempfile.TemporaryDirectory() as tmp:
            root, names, prints = self._root(tmp)
            _write(root / "w.safetensors", b"\x03" * 64)  # same size, new mtime/content
            with self.assertRaisesRegex(ValueError, "^assets_changed$"):
                _verify_manifest_fingerprints(root, names, prints)

    def test_size_change_detected(self) -> None:
        """Test size change detected."""
        with tempfile.TemporaryDirectory() as tmp:
            root, names, prints = self._root(tmp)
            _write(root / "w.safetensors", b"\x03" * 65)
            with self.assertRaisesRegex(ValueError, "^assets_changed$"):
                _verify_manifest_fingerprints(root, names, prints)

    def test_replacement_detected(self) -> None:
        """Test replacement detected."""
        with tempfile.TemporaryDirectory() as tmp:
            root, names, prints = self._root(tmp)
            # Replace via rename: new inode, then restore the original mtime
            # so only the inode differs.
            original = root / "w.safetensors"
            stat = original.stat()
            replacement = root / "w.safetensors.new"
            _write(replacement, b"\x02" * 64)
            os.utime(replacement, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            replacement.replace(original)
            with self.assertRaisesRegex(ValueError, "^assets_changed$"):
                _verify_manifest_fingerprints(root, names, prints)

    def test_deletion_detected(self) -> None:
        """Test deletion detected."""
        with tempfile.TemporaryDirectory() as tmp:
            root, names, prints = self._root(tmp)
            (root / "w.safetensors").unlink()
            # Fails closed with the fixed vocabulary (same code the old
            # second _manifest pass produced via _member).
            with self.assertRaisesRegex(ValueError, "^missing_or_unsafe_asset$"):
                _verify_manifest_fingerprints(root, names, prints)


class TestBackendSingleHash(unittest.TestCase):
    """Test backend single hash."""

    def test_backend_does_not_call_full_manifest(self) -> None:
        """The backend must hash weights once; the old double-_manifest call
        is gone. Patch _manifest to explode if the backend touches it, and
        record the fingerprint verification calls.
        """
        from tests.test_adapter_integration import build_tiny_gpt2

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "tiny"
            model_dir.mkdir()
            build_tiny_gpt2(model_dir)
            from model_protocol_lab import backends
            from model_protocol_lab.backends import LocalHFBackend

            real_manifest = _manifest
            manifest_calls: list[tuple[Path, tuple[str, ...]]] = []
            verify_calls: list[tuple[tuple[str, ...], dict[str, tuple[int, int, int]]]] = []
            real_verify = backends._verify_manifest_fingerprints

            def exploding_manifest(root: Path, names: tuple[str, ...]):
                """Exploding manifest."""
                # type: ignore[no-untyped-def]
                manifest_calls.append((root, names))
                return real_manifest(root, names)

            def recording_verify(  # type: ignore[no-untyped-def]
                root: Path, names: tuple[str, ...], fingerprints: dict[str, tuple[int, int, int]]
            ) -> None:
                """Recording verify."""
                verify_calls.append((names, fingerprints))
                real_verify(root, names, fingerprints)

            with (
                patch.object(backends, "_manifest", side_effect=exploding_manifest),
                patch.object(
                    backends, "_verify_manifest_fingerprints", side_effect=recording_verify
                ),
            ):
                backend = LocalHFBackend(model_dir, device="cpu", max_context=64)
            self.assertEqual(manifest_calls, [], "backend must not call _manifest (double hash)")
            # Both asset groups re-verified after load with the fingerprints
            # captured during the single hash pass.
            verified = {names for names, _ in verify_calls}
            self.assertIn(("tokenizer.json",), verified)
            self.assertTrue(
                any("model.safetensors" in names for names in verified),
                f"model weights not re-verified: {verified}",
            )
            self.assertTrue(backend.fingerprint.startswith("sha256:"))

    def test_backend_fails_closed_on_changed_assets(self) -> None:
        """If the post-load fingerprint check fires, init fails closed with
        the fixed vocabulary instead of returning a half-built backend.
        """
        from tests.test_adapter_integration import build_tiny_gpt2

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "tiny"
            model_dir.mkdir()
            build_tiny_gpt2(model_dir)
            from model_protocol_lab import backends
            from model_protocol_lab.backends import LocalHFBackend

            def failing_verify(  # type: ignore[no-untyped-def]
                root: Path, names: tuple[str, ...], fingerprints: dict[str, tuple[int, int, int]]
            ) -> None:
                """Failing verify."""
                raise ValueError("assets_changed")

            with (
                patch.object(backends, "_verify_manifest_fingerprints", side_effect=failing_verify),
                self.assertRaisesRegex(RuntimeError, "^local_backend_initialization_failed$"),
            ):
                LocalHFBackend(model_dir, device="cpu", max_context=64)


if __name__ == "__main__":
    unittest.main()
