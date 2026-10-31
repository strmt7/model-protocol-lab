"""Cycle 226: pack container vocabulary (B-06).

``verify_bundle_bytes`` normalized every container-level failure to
``ValueError("bundle_container_invalid")`` except one class: a
central directory declaring an extract version newer than the
running Python's zipfile supports raises ``NotImplementedError``
from ``ZipFile`` construction, which escaped both the function and
the ``verify --bundle`` CLI entry (whose handler catches
``ValueError``) as a raw traceback. The construction guard now names
that class, like every sibling container check in the module.

The patched bundle below is built with the module's own writer and
then has its central-directory "version needed to extract" field
rewritten to 222 (22.2) — the readiness audit's reproducer
construction. Reverting the guard widening re-breaks both patched
tests; the unpatched control pins that the widened guard did not
swallow healthy bundles.
"""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from model_protocol_lab.pack import (
    build_bundle_bytes,
    build_manifest,
    manifest_bytes,
    run_verify_bundle,
    verify_bundle_bytes,
)

RESULTS = json.dumps(
    {"decision": "keep", "search": [], "held_out": {}, "backend_fingerprint": "fp"}
).encode()


def _bundle() -> bytes:
    """Build a bundle with the module's own writer."""
    files = [
        {
            "path": "results.json",
            "role": "results",
            "sha256": hashlib.sha256(RESULTS).hexdigest(),
            "size": len(RESULTS),
        }
    ]
    manifest = build_manifest(
        files,
        {"backend_fingerprint": "fp", "catalog_sha256": "0" * 64, "package_version": "0.0.0"},
    )
    return build_bundle_bytes(manifest_bytes(manifest), [("results.json", RESULTS)])


def _future_version_bundle() -> bytes:
    """The valid bundle with its extract-version field set to 222."""
    data = bytearray(_bundle())
    offset = data.find(b"PK\x01\x02")
    assert offset > 0
    data[offset + 6 : offset + 8] = (222).to_bytes(2, "little")
    return bytes(data)


class BundleContainerVocabularyTests(unittest.TestCase):
    """B-06: the zip-version escape joins the fixed vocabulary."""

    def test_unsupported_extract_version_is_container_invalid(self):
        """Test unsupported extract version is container invalid."""
        with self.assertRaises(ValueError) as caught:
            verify_bundle_bytes(_future_version_bundle())
        self.assertEqual(str(caught.exception), "bundle_container_invalid")

    def test_verify_bundle_cli_returns_1_not_a_traceback(self):
        """Test verify bundle cli returns 1 not a traceback."""
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "bundle.zip"
            target.write_bytes(_future_version_bundle())
            args = types.SimpleNamespace(bundle=str(target), expect_digest=None)
            err = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(err):
                code = run_verify_bundle(args)
            self.assertEqual(code, 1)
            self.assertIn("bundle_container_invalid", err.getvalue())

    def test_unpatched_bundle_still_verifies(self):
        """Test unpatched bundle still verifies."""
        report = verify_bundle_bytes(_bundle())
        self.assertIn("verdict", report)
        self.assertIn("checks", report)


if __name__ == "__main__":
    unittest.main()
