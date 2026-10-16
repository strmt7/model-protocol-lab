"""Cycle 102: true ``except OSError`` branch coverage for examine_model.

Background: ``_open_regular_read`` converts ``os.open`` failures to
``ValueError("missing_or_unsafe_asset")``, so the ``except OSError``
branches in ``examine_model`` are *not* reachable via open-time
permission errors — the cycle-92 "without_root" tests (which mock
``os.open``) exercise the ``except ValueError`` path, and their
docstrings are corrected here to say so. The ``except OSError``
branches are defense-in-depth for I/O errors *after* a successful
open (disk errors during ``read()``, files vanishing between open and
stat). These tests inject ``OSError`` at the read stage — via a
tracked-fd ``os.fdopen`` replacement — and pin the fail-closed
behavior of each branch:

- ``_model_assets`` (weight-index read error) -> asset_layout failed,
  early return;
- ``_manifest`` (asset read error) -> asset_layout ok, asset_manifest
  failed with the fixed vocabulary;
- config ``_read_json`` (read error) -> config failed with the fixed
  vocabulary;
- tokenizer ``_read_json`` (read error) -> tokenizer check failed with
  the fixed vocabulary.

In every case the raw OSError must never escape ``examine_model`` and
the detail must be the fixed ``missing_or_unsafe_asset`` vocabulary.
"""

from __future__ import annotations

import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from model_protocol_lab import backends
from model_protocol_lab.doctor import examine_model


def _write_healthy(root: Path, *, sharded: bool = False) -> None:
    (root / "config.json").write_text(json.dumps({"model_type": "gpt2", "n_positions": 32}))
    (root / "tokenizer.json").write_text(json.dumps({"version": "1.0"}))
    header = json.dumps(
        {
            "__metadata__": {},
            "w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
        }
    ).encode()
    blob = struct.pack("<Q", len(header)) + header + b"\x00" * 8
    if sharded:
        index = {
            "metadata": {},
            "weight_map": {"model.safetensors": "shard0.safetensors"},
        }
        (root / "model.safetensors.index.json").write_text(json.dumps(index))
        (root / "shard0.safetensors").write_bytes(blob)
    else:
        (root / "model.safetensors").write_bytes(blob)


class _FailingReader:
    """Binary file stand-in whose read() raises OSError (disk I/O error).

    Closes the real fd on context exit so the injection leaks nothing.
    """

    def __init__(self, fd: int) -> None:
        self._fd = fd

    def __enter__(self) -> _FailingReader:
        return self

    def __exit__(self, *args: object) -> bool:
        os.close(self._fd)
        return False

    def read(self, *args: object, **kwargs: object) -> bytes:
        raise OSError(5, "I/O error")


class _ReadErrorInjector:
    """Replaces ``os.fdopen`` so reads of one named file raise OSError.

    Tracks fd -> path via a wrapped ``os.open`` (platform-independent,
    unlike /proc/self/fd). Only the named file's reads fail; every other
    file reads normally, so the error lands in a chosen pipeline stage.
    ``skip_first`` lets the first N reads of the target succeed — needed
    when the manifest stage hashes a file before a later stage reads it.
    """

    def __init__(self, target_name: str, *, skip_first: int = 0) -> None:
        self._target = target_name
        self._skip_first = skip_first
        self._real_open = os.open
        self._real_fdopen = os.fdopen
        self._fd_paths: dict[int, str] = {}
        self._hits = 0

    def _fake_open(self, path: object, flags: int, *args: object, **kwargs: object) -> int:
        fd = self._real_open(path, flags, *args, **kwargs)
        self._fd_paths[fd] = os.fspath(path)  # type: ignore[arg-type]
        return fd

    def _fake_fdopen(self, fd: int, *args: object, **kwargs: object) -> object:
        if os.path.basename(self._fd_paths.get(fd, "")) == self._target:
            self._hits += 1
            if self._hits > self._skip_first:
                return _FailingReader(fd)
        return self._real_fdopen(fd, *args, **kwargs)

    def __enter__(self) -> _ReadErrorInjector:
        self._open_patch = mock.patch.object(backends.os, "open", self._fake_open)
        self._fdopen_patch = mock.patch.object(backends.os, "fdopen", self._fake_fdopen)
        self._open_patch.start()
        self._fdopen_patch.start()
        return self

    def __exit__(self, *args: object) -> None:
        self._fdopen_patch.stop()
        self._open_patch.stop()


def _checks(report: dict) -> dict:
    return report["model"]["checks"]


class ExamineModelOSErrorBranchTests(unittest.TestCase):
    def test_index_read_error_fails_asset_layout(self):
        """OSError while _model_assets reads the weight index -> the
        ``except OSError`` around _model_assets fires: failed verdict,
        asset_layout failed with the fixed vocabulary, early return."""
        root = Path(tempfile.mkdtemp())
        _write_healthy(root, sharded=True)
        with _ReadErrorInjector("model.safetensors.index.json"):
            report = examine_model(root)
        self.assertEqual(report["verdict"], "failed")
        checks = _checks(report)
        self.assertEqual(checks["asset_layout"]["status"], "failed")
        self.assertEqual(checks["asset_layout"]["detail"], "missing_or_unsafe_asset")
        # Early return: no later-stage checks recorded.
        self.assertNotIn("asset_manifest", checks)

    def test_manifest_read_error_fails_manifest_check(self):
        """OSError while _manifest hashes an asset -> the ``except
        OSError`` around _manifest fires: asset_layout ok (enumeration
        succeeded), asset_manifest failed with the fixed vocabulary."""
        root = Path(tempfile.mkdtemp())
        _write_healthy(root)
        with _ReadErrorInjector("model.safetensors"):
            report = examine_model(root)
        self.assertEqual(report["verdict"], "failed")
        checks = _checks(report)
        self.assertEqual(checks["asset_layout"]["status"], "ok")
        self.assertEqual(checks["asset_manifest"]["status"], "failed")
        self.assertEqual(checks["asset_manifest"]["detail"], "missing_or_unsafe_asset")

    def test_config_read_error_fails_config_check(self):
        """OSError while _read_json reads config.json -> the ``except
        OSError`` around the config read fires: config failed with the
        fixed vocabulary, never a raw OSError."""
        root = Path(tempfile.mkdtemp())
        _write_healthy(root)
        # Manifest hashes config.json first; skip that read so the error
        # lands in the config stage's _read_json.
        with _ReadErrorInjector("config.json", skip_first=1):
            report = examine_model(root)
        self.assertEqual(report["verdict"], "failed")
        checks = _checks(report)
        self.assertEqual(checks["config"]["status"], "failed")
        self.assertEqual(checks["config"]["detail"], "missing_or_unsafe_asset")

    def test_tokenizer_read_error_fails_tokenizer_check(self):
        """OSError while _read_json reads tokenizer.json -> the
        ``except OSError`` around the tokenizer read fires."""
        root = Path(tempfile.mkdtemp())
        _write_healthy(root)
        # tokenizer.json is read only by its own stage (the manifest
        # hashes just the model assets), so no skip is needed.
        with _ReadErrorInjector("tokenizer.json"):
            report = examine_model(root)
        self.assertEqual(report["verdict"], "failed")
        checks = _checks(report)
        # Find the tokenizer check regardless of its exact key.
        details = {name: c["detail"] for name, c in checks.items()}
        self.assertIn("missing_or_unsafe_asset", set(details.values()))

    def test_no_raw_oserror_escapes(self):
        """Belt and braces: even with every read failing, examine_model
        returns a report — it never propagates OSError."""
        root = Path(tempfile.mkdtemp())
        _write_healthy(root, sharded=True)
        for target in (
            "model.safetensors.index.json",
            "model.safetensors",
            "config.json",
            "tokenizer.json",
            "shard0.safetensors",
        ):
            with _ReadErrorInjector(target):
                try:
                    report = examine_model(root)
                except OSError:
                    self.fail(f"raw OSError escaped for {target}")
                self.assertEqual(report["verdict"], "failed")


if __name__ == "__main__":
    unittest.main()
