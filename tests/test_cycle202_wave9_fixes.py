"""Cycle 202 — wave-9 red-team correction round.

Specs: the wave-9 round briefs (research / red-team A / red-team B /
verifier, adjudicated by the coordinator; docs/RESEARCH_LOG.md cycle-202
entry). Every test here pins one confirmed finding: each was reproduced
by two independent executions before the fix, and each test fails on
the pre-fix tree for the stated reason. The contract under test is the
repo's fixed error vocabulary — hostile input on an untrusted surface
maps to a fixed code or a clean refusal, never a raw interpreter
exception, a heap address in a payload, or invalid JSON on a wire.
"""

from __future__ import annotations

import argparse
import io
import json
import struct
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from model_protocol_lab import backends, diffreports, doctor, reporting
from model_protocol_lab import pack as pk
from model_protocol_lab.experiments import _backend_fingerprint
from tests.test_cycle201_readiness import _mentioned

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_REPORT = REPO_ROOT / "examples" / "sample-report"
# The server ships as model_protocol_lab.mcp_server (cycle 208).
MCP_SERVER_CMD = [sys.executable, "-m", "model_protocol_lab.mcp_server"]


def _valid_bundle(tmpdir: Path) -> bytes:
    report = Path(tmpdir) / "report"
    report.mkdir()
    for name in ("results.json", "report.html"):
        (report / name).write_bytes((SAMPLE_REPORT / name).read_bytes())
    data, _manifest, _digest = pk.pack_bundle(report, [])
    return data


def _recompressed(bundle: bytes, target: str, codec: int) -> bytes:
    """Rebuild a bundle with one entry stored under a real codec."""
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        comment = archive.comment
        entries = [(info.filename, archive.read(info.filename)) for info in archive.infolist()]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, payload in entries:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = codec if name == target else zipfile.ZIP_STORED
            info.create_system = 0
            info.external_attr = (0o644 & 0xFFFF) << 16
            archive.writestr(info, payload)
        archive.comment = comment
    return buffer.getvalue()


def _garble_entry_stream(bundle: bytes, target: str) -> bytes:
    """Overwrite one entry's compressed stream with 0xFF bytes.

    The container stays structurally valid (same sizes, same comment),
    so verification reaches the entry read; the codec then fails on the
    stream itself — the path the STORED-only writer never exercises.
    """
    data = bytearray(bundle)
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        info = archive.getinfo(target)
        offset, size = info.header_offset, info.compress_size
    name_len, extra_len = struct.unpack_from("<HH", data, offset + 26)
    start = offset + 30 + name_len + extra_len
    for i in range(start, start + size):
        data[i] = 0xFF
    return bytes(data)


class TestPackCodecNormalisation(unittest.TestCase):
    """F1: corrupt compressed entries map to bundle_entry_read_failed."""

    def test_corrupt_codec_manifest_fails_closed_with_fixed_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = _valid_bundle(Path(tmp))
            for codec in (zipfile.ZIP_DEFLATED, zipfile.ZIP_LZMA, zipfile.ZIP_BZIP2):
                with self.subTest(codec=codec):
                    hostile = _garble_entry_stream(
                        _recompressed(bundle, pk.MANIFEST_NAME, codec), pk.MANIFEST_NAME
                    )
                    report = pk.verify_bundle_bytes(hostile)
                    self.assertEqual(report["verdict"], "failed")
                    details = [c["detail"] for c in report["checks"]]
                    self.assertIn("bundle_entry_read_failed", details)

    def test_corrupt_codec_bundle_cli_path_returns_failure_not_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = _valid_bundle(Path(tmp))
            hostile = _garble_entry_stream(
                _recompressed(bundle, pk.MANIFEST_NAME, zipfile.ZIP_DEFLATED),
                pk.MANIFEST_NAME,
            )
            path = Path(tmp) / "hostile.bundle.zip"
            path.write_bytes(hostile)
            args = argparse.Namespace(bundle=str(path), expect_digest=None)
            # Pre-fix this raised zlib.error out of run_verify_bundle.
            self.assertEqual(pk.run_verify_bundle(args), 2)


class TestPackSymlinkConsistency(unittest.TestCase):
    """F11: both pack readers refuse symlinks (O_NOFOLLOW)."""

    def test_load_bundle_bytes_refuses_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            real = Path(tmp) / "real.bundle.zip"
            real.write_bytes(b"bundle-bytes")
            link = Path(tmp) / "link.bundle.zip"
            link.symlink_to(real)
            self.assertEqual(pk.load_bundle_bytes(real), b"bundle-bytes")
            with self.assertRaises(ValueError) as ctx:
                pk.load_bundle_bytes(link)
            self.assertEqual(str(ctx.exception), "bundle_unreadable")


class TestFingerprintGuard(unittest.TestCase):
    """F2: stages enforce the Backend protocol's fingerprint type."""

    def test_helper_rejects_non_string_fingerprints(self) -> None:
        class MethodBackend:
            def fingerprint(self) -> dict:  # the cycle-195 fake shape
                return {"model": "fake"}

        class IntBackend:
            fingerprint = 7

        class MissingBackend:
            pass

        for backend in (MethodBackend(), IntBackend(), MissingBackend()):
            with self.subTest(backend=type(backend).__name__):
                with self.assertRaises(ValueError) as ctx:
                    _backend_fingerprint(backend)  # type: ignore[arg-type]
                self.assertEqual(str(ctx.exception), "Backend fingerprint must be a string")

    def test_helper_passes_conforming_fingerprint_through(self) -> None:
        class GoodBackend:
            fingerprint = "sha256:good"

        self.assertEqual(_backend_fingerprint(GoodBackend()), "sha256:good")  # type: ignore[arg-type]

    def test_stage_raises_on_method_fingerprint(self) -> None:
        from model_protocol_lab.experiments import distraction_stage
        from tests.test_cycle184_distraction import (
            BOUNDARY_WORD,
            CASES,
            FORMATS,
            _entries,
            _WordBackend,
        )

        class MethodFingerprintBackend(_WordBackend):
            def fingerprint(self) -> dict:  # type: ignore[override]
                return {"model": "fake"}

        backend = MethodFingerprintBackend()
        with self.assertRaises(ValueError) as ctx:
            distraction_stage(
                backend,
                FORMATS,
                CASES,
                _entries(backend, FORMATS, CASES),
                boundary=tuple(backend.encode(BOUNDARY_WORD)),
                max_new_tokens=8,
                max_context=8192,
            )
        self.assertEqual(str(ctx.exception), "Backend fingerprint must be a string")

    def test_stage_payload_carries_the_protocol_string(self) -> None:
        from model_protocol_lab import distraction
        from model_protocol_lab.experiments import distraction_stage
        from tests.test_cycle184_distraction import (
            BOUNDARY_WORD,
            CASES,
            FORMATS,
            _entries,
            _WordBackend,
        )

        backend = _WordBackend()
        stage = distraction_stage(
            backend,
            FORMATS,
            CASES,
            _entries(backend, FORMATS, CASES),
            boundary=tuple(backend.encode(BOUNDARY_WORD)),
            max_new_tokens=8,
            max_context=8192,
        )
        fixture_ids = [case.id for case in CASES]
        self.assertEqual(stage["protocol"]["backend_fingerprint"], "sha256:word")
        self.assertEqual(
            stage["protocol"]["fingerprint"],
            distraction.protocol_fingerprint(fixture_ids, "sha256:word"),
        )

    def test_no_stage_stringifies_the_fingerprint(self) -> None:
        # Source audit: the stringify patterns are gone, and every
        # stage that builds a protocol from a backend routes the read
        # through the guard (the enumeration the wave-9 briefs counted
        # three times: 21 call sites across 12 stage functions).
        source = (REPO_ROOT / "model_protocol_lab" / "experiments.py").read_text(encoding="utf-8")
        self.assertNotIn("str(backend.fingerprint)", source)
        self.assertNotIn('str(getattr(native_backend, "fingerprint"', source)
        self.assertNotIn('str(getattr(bf16_backend, "fingerprint"', source)
        self.assertNotIn('str(getattr(backend, "fingerprint"', source)
        stages = (
            "repetition_stage",
            "distraction_stage",
            "leading_stage",
            "dateline_stage",
            "multiturn_stage",
            "needle_stage",
            "placement_stage",
            "numform_stage",
            "template_robustness_stage",
            "choiceaudit_stage",
            "precision_stage",
            "confidence_stage",
        )
        for name in stages:
            with self.subTest(stage=name):
                start = source.index(f"def {name}(")
                nxt = source.find("\ndef ", start + 1)
                body = source[start:] if nxt == -1 else source[start:nxt]
                self.assertIn("_backend_fingerprint(", body)


class TestIntegerLimitVocabulary(unittest.TestCase):
    """F3: >4300-digit integer literals map to each site's fixed code."""

    BIG = b"9" * 5000

    def test_parse_results_bytes_maps_int_limit_to_results_invalid_json(self) -> None:
        payload = b'{"decision":"x","search":[],"held_out":null,"seed":' + self.BIG + b"}"
        with self.assertRaises(ValueError) as ctx:
            diffreports.parse_results_bytes(payload)
        self.assertEqual(str(ctx.exception), "results_invalid_json")

    def test_parse_results_bytes_keeps_its_own_fixed_codes(self) -> None:
        # The broadened ValueError clause must not swallow the codes
        # the module's own hooks raise.
        with self.assertRaises(ValueError) as ctx:
            diffreports.parse_results_bytes(b'{"a":1,"a":2}')
        self.assertEqual(str(ctx.exception), "results_duplicate_field")
        with self.assertRaises(ValueError) as ctx:
            diffreports.parse_results_bytes(b'{"seed":NaN}')
        self.assertEqual(str(ctx.exception), "results_nonfinite_number")

    def test_backends_read_json_maps_int_limit_to_invalid_asset_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "asset.json"
            path.write_bytes(b'{"a": ' + self.BIG + b"}")
            with self.assertRaises(ValueError) as ctx:
                backends._read_json(path)
            self.assertEqual(str(ctx.exception), "invalid_asset_json")

    def test_doctor_fails_the_check_instead_of_raising(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_bytes(b"{}")
            header = b'{"t":{"dtype":"F32","shape":[' + self.BIG + b'],"data_offsets":[0,0]}}'
            weights = struct.pack("<Q", len(header)) + header
            (root / "model.safetensors").write_bytes(weights)
            # Pre-fix this raised the raw int-limit ValueError.
            report = doctor.examine_model(root)
            found: list[dict] = []

            def walk(node: object) -> None:
                if type(node) is dict:
                    if "weight_headers" in node:
                        found.append(node["weight_headers"])
                    for value in node.values():
                        walk(value)

            walk(report)
            self.assertEqual(len(found), 1)
            self.assertEqual(found[0]["status"], "failed")
            self.assertIn("invalid_safetensors_header", found[0]["detail"])


class TestNonfiniteFloatRejection(unittest.TestCase):
    """F4: overflow float literals are rejected at the results loader."""

    def test_overflow_float_maps_to_results_nonfinite_number(self) -> None:
        payload = b'{"decision":"x","search":[],"held_out":null,"seed":1e999}'
        with self.assertRaises(ValueError) as ctx:
            diffreports.parse_results_bytes(payload)
        self.assertEqual(str(ctx.exception), "results_nonfinite_number")

    def test_finite_floats_still_parse(self) -> None:
        payload = b'{"decision":"x","search":[],"held_out":null,"seed":1.5e3}'
        parsed = diffreports.parse_results_bytes(payload)
        self.assertEqual(parsed["seed"], 1500.0)


class TestChangelogMentionBoundaries(unittest.TestCase):
    """F5: gate mentions are whole tokens, not substrings."""

    def test_flag_inside_longer_flag_is_not_a_mention(self) -> None:
        self.assertFalse(_mentioned("--include", "flags: `--include-evidence`"))
        self.assertFalse(_mentioned("--seed", "the --seeded-output option"))
        self.assertTrue(_mentioned("--include", "the `--include` option"))

    def test_cycle_inside_longer_number_is_not_a_mention(self) -> None:
        self.assertFalse(_mentioned("cycle 202", "the cycle 2025 retrospective"))
        self.assertTrue(_mentioned("cycle 202", "### Fixed (cycle 202 — wave-9)"))


class TestMcpFrameDiscipline(unittest.TestCase):
    """F6/F7: the MCP frame parser rejects non-finite numbers; a
    non-object backend is -32602 (invalid params), not -32603."""

    def _serve(self, lines: str) -> str:
        proc = subprocess.run(
            MCP_SERVER_CMD,
            input=lines,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return proc.stdout

    def test_nan_id_is_a_parse_error_and_never_echoed(self) -> None:
        out = self._serve('{"jsonrpc":"2.0","id":NaN,"method":"ping"}\n')
        self.assertNotIn("NaN", out)
        response = json.loads(out.strip().splitlines()[0])
        self.assertEqual(response["error"]["code"], -32700)
        self.assertIsNone(response["id"])

    def test_overflow_float_id_is_a_parse_error(self) -> None:
        out = self._serve('{"jsonrpc":"2.0","id":1e999,"method":"ping"}\n')
        self.assertNotIn("Infinity", out)
        response = json.loads(out.strip().splitlines()[0])
        self.assertEqual(response["error"]["code"], -32700)

    def test_non_object_backend_is_invalid_params(self) -> None:
        import importlib

        module = importlib.import_module("model_protocol_lab.mcp_server")
        response = module.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "probe_surface_noise_stability",
                    "arguments": {"prompt": "hello", "backend": "selftest"},
                },
            }
        )
        self.assertIsNotNone(response)
        self.assertEqual(response["error"]["code"], -32602)


class TestReleaseBuildRefusals(unittest.TestCase):
    """F8/F9 + leads L2/L5: release_build refuses cleanly."""

    def test_symlinks_are_found_and_refused(self) -> None:
        from tools import release_build

        with tempfile.TemporaryDirectory() as tmp:
            tree = Path(tmp) / "tree"
            tree.mkdir()
            (tree / "real.txt").write_text("content", encoding="utf-8")
            outside = Path(tmp) / "outside.txt"
            outside.write_text("outside", encoding="utf-8")
            (tree / "leak.txt").symlink_to(outside)
            (tree / "dangling.txt").symlink_to(Path(tmp) / "missing.txt")
            found = release_build._find_symlinks(tree)
            self.assertEqual([p.name for p in found], ["dangling.txt", "leak.txt"])
            out = Path(tmp) / "dist"
            code = release_build.main(
                ["--source", str(tree), "--output", str(out), "--epoch", "1700000000"]
            )
            self.assertEqual(code, 2)
            self.assertFalse(out.exists())

    def test_expect_version_mismatch_is_refused_before_building(self) -> None:
        from tools import release_build

        self.assertEqual(release_build._package_version(REPO_ROOT), "0.1.0")
        with tempfile.TemporaryDirectory() as tmp:
            code = release_build.main(
                [
                    "--source",
                    str(REPO_ROOT),
                    "--output",
                    str(Path(tmp) / "dist"),
                    "--epoch",
                    "1700000000",
                    "--expect-version",
                    "9.9.9",
                ]
            )
            self.assertEqual(code, 2)

    def test_absurd_epoch_is_a_clean_usage_error(self) -> None:
        from tools import release_build

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit) as ctx:
                release_build.main(
                    [
                        "--source",
                        str(REPO_ROOT),
                        "--output",
                        str(Path(tmp) / "dist"),
                        "--epoch",
                        "-1",
                    ]
                )
            self.assertEqual(ctx.exception.code, 2)

    def test_missing_artifact_is_a_named_failure(self) -> None:
        from tools import release_build

        with self.assertRaises(SystemExit) as ctx:
            release_build._first(iter([]), "sdist")
        self.assertIn("no sdist artifact", str(ctx.exception))

    def test_workflow_binds_tag_to_version(self) -> None:
        workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
        self.assertIn("--expect-version", workflow)
        self.assertIn("GITHUB_REF_NAME#v", workflow)


class TestReportingControlStrip(unittest.TestCase):
    """F10: conformance + transfer sections strip C0 controls."""

    def test_conformance_section_strips_controls(self) -> None:
        result = {
            "template_conformance": {
                "status": "ok",
                "diffs": [
                    {
                        "status": "ok",
                        "format": "fmt\x1b[31mRED\x1b[0m",
                        "fixture": "fix\x00\x07ture",
                        "hand_tokens": 3,
                        "template_tokens": 4,
                        "diff": [
                            {"hand_span": "a\x1bb", "template_span": "c\x07d"},
                        ],
                    }
                ],
            }
        }
        html = reporting._conformance_section(result)
        self.assertNotIn("\x1b", html)
        self.assertNotIn("\x00", html)
        self.assertNotIn("\x07", html)
        self.assertIn("fmt[31mRED[0m", html)

    def test_conformance_detail_path_strips_controls(self) -> None:
        result = {"template_conformance": {"status": "broken", "detail": "bad\x1bdetail"}}
        html = reporting._conformance_section(result)
        self.assertNotIn("\x1b", html)
        self.assertIn("baddetail", html)

    def test_transfer_section_strips_controls(self) -> None:
        transfer = {
            "available": True,
            "primary_winner": "alpha",
            "checkpoints": [
                {"id": "ck\x00pt", "path": "/models/a", "fingerprint": "fp\x1b1234567890abcdef"},
            ],
            "per_checkpoint": {"ck\x00pt": {"decision": "selected", "winner": "alpha"}},
            "matrix": {"alpha": {"ck\x00pt": "transfers"}},
            "transfer_verdict": "partial",
        }
        html = reporting._transfer_section({"transfer": transfer})
        self.assertNotIn("\x00", html)
        self.assertNotIn("\x1b", html)


if __name__ == "__main__":
    unittest.main()
