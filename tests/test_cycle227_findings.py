"""Cycle 227: the deferred wave-2 correctness findings.

Bug-hunt wave 2 (coordinators #86/#89) and cybersecurity audit 2
(coordinator #87) confirmed sixteen defects against the cycle-225
head; cycle 226 absorbed two and defer-listed the rest, each
re-verified still open at its own final head. This file pins the
fix batch:

- F2 -- the sealed-bundle manifest parse had a ``parse_constant``
  hook but no ``parse_float`` hook, so a bundle-supplied ``1e999``
  became ``inf``, the verify verdict stayed ``ok``, and the report
  re-emitted the value as the bare token ``Infinity`` (invalid
  JSON). The manifest now carries the same hook pair the results
  parser has carried since cycle 202.
- C-02 -- ``structured.classify_response`` parsed model output with
  a plain ``json.loads``; NaN / ``1e999`` / duplicate keys in
  unchecked fields earned ``adherent``. Duplicate keys are now a
  parse failure (the shared discipline), and an undeclared field's
  subtree must be finite throughout -- the schema stage owns
  number finiteness in this module (cycle 158), so a non-finite
  number in an extra field is ``schema_failed``, the same verdict
  a checked field earns.
- C-03 -- the checkpoint JSON readers (``backends._read_json`` and
  ``doctor._check_weight_headers``) accepted duplicate keys,
  non-finite numbers, and lone surrogates; both now apply the same
  discipline and fail in their existing fixed vocabularies.

Each test below is red on the pre-fix tree for the class it names.
"""

from __future__ import annotations

import io
import json
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path

from model_protocol_lab import backends, doctor, pack, structured

_SCHEMA = {
    "type": "object",
    "required": ["answer"],
    "properties": {"answer": {"type": "string"}},
    "additionalProperties": True,
}


def _bundle_with_manifest(mutate: str) -> bytes:
    """Build a genuine bundle, then rewrite its manifest text."""
    src = Path(__file__).resolve().parent.parent / "examples" / "sample-report"
    bundle, _manifest, _digest = pack.pack_bundle(src, [])
    zin = zipfile.ZipFile(io.BytesIO(bundle))
    entries = [
        ("results.json", zin.read("results.json")),
        ("report.html", zin.read("report.html")),
    ]
    raw_manifest = zin.read("manifest.json").decode()
    if mutate == "context_1e999":
        mutated = raw_manifest.replace('"context":{', '"context":{"zz_probe":1e999,', 1)
    elif mutate == "toplevel_1e999":
        mutated = raw_manifest.rstrip()[:-1] + ',"zz_probe":1e999}'
    elif mutate == "context_infinity":
        mutated = raw_manifest.replace('"context":{', '"context":{"zz_probe":Infinity,', 1)
    elif mutate == "context_finite_float":
        mutated = raw_manifest.replace('"context":{', '"context":{"zz_probe":1.5,', 1)
    else:  # pragma: no cover - test-author error, not a code path
        raise AssertionError(f"unknown mutation {mutate}")
    return pack.build_bundle_bytes(mutated.encode(), entries)


def _schema_check(report: dict) -> dict:
    """Return the seal_manifest_schema check of a verify report."""
    return next(c for c in report["checks"] if c["id"] == "seal_manifest_schema")


class ManifestNonfiniteFloatTests(unittest.TestCase):
    """F2: an overflowing float literal is not manifest JSON."""

    def test_1e999_in_context_fails_manifest_schema(self):
        """Test 1e999 in context fails manifest schema."""
        report = pack.verify_bundle_bytes(_bundle_with_manifest("context_1e999"))
        check = _schema_check(report)
        self.assertEqual(check["status"], "failed")
        self.assertEqual(check["detail"], "manifest_unparseable")
        self.assertEqual(report["verdict"], "failed")

    def test_1e999_top_level_fails_manifest_schema(self):
        """Test 1e999 top level fails manifest schema."""
        report = pack.verify_bundle_bytes(_bundle_with_manifest("toplevel_1e999"))
        check = _schema_check(report)
        self.assertEqual(check["status"], "failed")
        self.assertEqual(check["detail"], "manifest_unparseable")
        self.assertEqual(report["verdict"], "failed")

    def test_literal_infinity_still_fails_manifest_schema(self):
        """Test literal Infinity still fails manifest schema."""
        report = pack.verify_bundle_bytes(_bundle_with_manifest("context_infinity"))
        check = _schema_check(report)
        self.assertEqual(check["status"], "failed")
        self.assertEqual(check["detail"], "manifest_unparseable")

    def test_finite_float_in_context_verdict_ok(self):
        """Test finite float in context verdict ok."""
        report = pack.verify_bundle_bytes(_bundle_with_manifest("context_finite_float"))
        self.assertEqual(report["verdict"], "ok")
        self.assertEqual((report.get("context") or {}).get("zz_probe"), 1.5)

    def test_genuine_bundle_verdict_ok_and_report_strict_json(self):
        """Test genuine bundle verdict ok and report strict json."""
        src = Path(__file__).resolve().parent.parent / "examples" / "sample-report"
        bundle, _manifest, _digest = pack.pack_bundle(src, [])
        report = pack.verify_bundle_bytes(bundle)
        self.assertEqual(report["verdict"], "ok")
        # The emitted report must parse under a strict JSON reader
        # (no bare Infinity/NaN tokens) -- the pre-fix report for a
        # 1e999 manifest did not.
        rendered = json.dumps(report, indent=2, ensure_ascii=True, sort_keys=True)

        def _reject(value: str) -> None:
            raise AssertionError(f"non-strict token in report: {value}")

        json.loads(rendered, parse_constant=_reject)


class ClassifyResponseDisciplineTests(unittest.TestCase):
    """C-02: classify_response applies the shared JSON discipline."""

    def test_nan_in_unchecked_field_is_schema_failed(self):
        """Test NaN in unchecked field is schema failed."""
        verdict = structured.classify_response('{"answer": "x", "note": NaN}', _SCHEMA)
        self.assertEqual(verdict["parse"], "parse_ok")
        self.assertEqual(verdict["schema"], "schema_failed")
        self.assertEqual(verdict["outcome"], "schema_failed")

    def test_overflowing_float_in_unchecked_field_is_schema_failed(self):
        """Test overflowing float in unchecked field is schema failed."""
        verdict = structured.classify_response('{"answer": "x", "note": 1e999}', _SCHEMA)
        self.assertEqual(verdict["parse"], "parse_ok")
        self.assertEqual(verdict["schema"], "schema_failed")
        self.assertEqual(verdict["outcome"], "schema_failed")

    def test_nonfinite_nested_in_unchecked_field_is_schema_failed(self):
        """Test nonfinite nested in unchecked field is schema failed."""
        verdict = structured.classify_response(
            '{"answer": "x", "note": {"deep": [1, Infinity]}}', _SCHEMA
        )
        self.assertEqual(verdict["outcome"], "schema_failed")

    def test_duplicate_keys_are_parse_failed(self):
        """Test duplicate keys are parse failed."""
        verdict = structured.classify_response('{"answer": "x", "answer": "y"}', _SCHEMA)
        self.assertEqual(verdict["parse"], "parse_failed")
        self.assertEqual(verdict["outcome"], "parse_failed")

    def test_finite_unchecked_field_still_adherent(self):
        """Test finite unchecked field still adherent."""
        verdict = structured.classify_response('{"answer": "x", "note": 1.5}', _SCHEMA)
        self.assertEqual(verdict["outcome"], "adherent")


class CheckpointJsonDisciplineTests(unittest.TestCase):
    """C-03: the checkpoint readers apply the shared discipline."""

    def _read(self, payload: str) -> dict:
        """Write payload as config.json in a temp dir and read it."""
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "config.json"
            target.write_text(payload, encoding="utf-8")
            return backends._read_json(target)

    def test_read_json_rejects_duplicate_keys(self):
        """Test read json rejects duplicate keys."""
        with self.assertRaises(ValueError) as ctx:
            self._read('{"a": 1, "a": 2}')
        self.assertEqual(str(ctx.exception), "invalid_asset_json")

    def test_read_json_rejects_nonfinite(self):
        """Test read json rejects nonfinite."""
        with self.assertRaises(ValueError) as ctx:
            self._read('{"a": 1e999}')
        self.assertEqual(str(ctx.exception), "invalid_asset_json")

    def test_read_json_rejects_literal_constant(self):
        """Test read json rejects literal constant."""
        with self.assertRaises(ValueError) as ctx:
            self._read('{"a": NaN}')
        self.assertEqual(str(ctx.exception), "invalid_asset_json")

    def test_read_json_rejects_lone_surrogate(self):
        """Test read json rejects lone surrogate."""
        with self.assertRaises(ValueError) as ctx:
            self._read('{"a": "\\ud800"}')
        self.assertEqual(str(ctx.exception), "invalid_asset_json")

    def test_read_json_wellformed_unchanged(self):
        """Test read json wellformed unchanged."""
        self.assertEqual(self._read('{"a": 1, "b": [1.5, "x"]}'), {"a": 1, "b": [1.5, "x"]})

    def _header_check(self, header_text: str) -> dict:
        """Run the weight-header check on a crafted shard."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            header = header_text.encode("utf-8")
            (root / "model.safetensors").write_bytes(
                struct.pack("<Q", len(header)) + header + b"\x00" * 16
            )
            return doctor._check_weight_headers(root, ("model.safetensors",))

    def test_weight_headers_reject_duplicate_keys(self):
        """Test weight headers reject duplicate keys."""
        result = self._header_check('{"__metadata__": {"a": "1", "a": "2"}}')
        self.assertEqual(result["status"], "failed")

    def test_weight_headers_reject_nonfinite(self):
        """Test weight headers reject nonfinite."""
        result = self._header_check('{"__metadata__": {"a": 1e999}}')
        self.assertEqual(result["status"], "failed")

    def test_weight_headers_reject_literal_constant(self):
        """Test weight headers reject literal constant."""
        result = self._header_check('{"__metadata__": {"a": Infinity}}')
        self.assertEqual(result["status"], "failed")

    def test_weight_headers_reject_lone_surrogate(self):
        """Test weight headers reject lone surrogate."""
        result = self._header_check('{"__metadata__": {"a": "\\ud800"}}')
        self.assertEqual(result["status"], "failed")

    def test_weight_headers_finite_float_ok(self):
        """Test weight headers finite float ok."""
        result = self._header_check('{"__metadata__": {"a": 1.5}}')
        self.assertEqual(result["status"], "ok")

    def test_weight_headers_wellformed_unchanged(self):
        """Test weight headers wellformed unchanged."""
        result = self._header_check('{"__metadata__": {"format": "pt"}}')
        self.assertEqual(result["status"], "ok")


if __name__ == "__main__":
    unittest.main()
