"""Tests for cycle 171: sealed result bundles (``pack`` / ``verify --bundle``).

Covers the format-1 container grammar, manifest schema validation, the
digest-over-stored-bytes property, golden round-trips (byte-identical
rebuilds), the tamper matrix, pack filesystem discipline, the
``verify --bundle`` CLI, and 4 tempdir mutation gates. See
``docs/SEALED_BUNDLE_DESIGN.md`` for the contract under test.
"""

import io
import json
import os
import sys
import tempfile
import unittest
import zipfile
from hashlib import sha256
from pathlib import Path

from model_protocol_lab import pack as pk
from model_protocol_lab.verify import _display_id

_REPO = Path(__file__).resolve().parent.parent
_SAMPLE = _REPO / "examples" / "sample-report"


def _sample_report_dir(tmpdir: Path) -> Path:
    report = Path(tmpdir) / "report"
    report.mkdir()
    for name in ("results.json", "report.html"):
        (report / name).write_bytes((_SAMPLE / name).read_bytes())
    return report


def _pack_to_bytes(report: Path, includes: list | None = None) -> tuple[bytes, dict, str]:
    return pk.pack_bundle(report, [Path(p) for p in (includes or [])])


def _hostile_zip(entries: list[tuple[str, bytes]], comment: bytes = b"") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in entries:
            archive.writestr(name, data)
        archive.comment = comment
    return buffer.getvalue()


def _manifest_entry(bundle: bytes) -> bytes:
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        return archive.read(pk.MANIFEST_NAME)


def _manifest_dict(bundle: bytes) -> dict:
    return json.loads(_manifest_entry(bundle).decode("utf-8"))


def _report_from_bundle(bundle: bytes, digest: str | None = None) -> dict:
    return pk.verify_bundle_bytes(bundle, expect_digest=digest)


class PathGrammarTests(unittest.TestCase):
    def test_accepts_canonical_paths(self):
        for path in ("results.json", "report.html", "inputs/dataset.json", "a/b/c.txt"):
            self.assertTrue(pk._valid_bundle_path(path), path)

    def test_rejects_non_canonical(self):
        for path in (
            "",
            ".",
            "..",
            "a/../b",
            "/abs",
            "C:foo",
            r"a\b",
            "a//b",
            "./x",
            "x/",
            "a\x00b",
            "a\x1bb",
        ):
            self.assertFalse(pk._valid_bundle_path(path), repr(path))

    def test_rejects_non_strings(self):
        for value in (None, 12, b"results.json", ["results.json"]):
            self.assertFalse(pk._valid_bundle_path(value))


class ManifestSchemaTests(unittest.TestCase):
    def _valid(self) -> dict:
        return {
            "context": {
                "backend_fingerprint": "fp",
                "catalog_sha256": "0" * 64,
                "package_version": "0.1.0",
            },
            "files": [
                {
                    "path": "results.json",
                    "role": "results",
                    "sha256": "a" * 64,
                    "size": 10,
                }
            ],
            "format": "mpl-bundle/1",
        }

    def test_valid_manifest(self):
        self.assertIsNone(pk._valid_manifest(self._valid()))

    def test_rejects_wrong_types(self):
        cases = [
            ({"format": 1}, "manifest_format_mismatch"),
            ({"files": "x"}, "manifest_files_invalid"),
            ({"files": []}, "manifest_files_invalid"),
            (
                {"files": [{"path": 1, "role": "results", "sha256": "a" * 64, "size": 1}]},
                "manifest_path_invalid",
            ),
            (
                {
                    "files": [
                        {
                            "path": "manifest.json",
                            "role": "report",
                            "sha256": "a" * 64,
                            "size": 1,
                        }
                    ]
                },
                "manifest_path_invalid",
            ),
            (
                {
                    "files": [
                        {
                            "path": "results.json",
                            "role": "results",
                            "sha256": "A" * 64,
                            "size": 1,
                        }
                    ]
                },
                "manifest_sha256_invalid",
            ),
            (
                {
                    "files": [
                        {
                            "path": "results.json",
                            "role": "results",
                            "sha256": "a" * 63,
                            "size": 1,
                        }
                    ]
                },
                "manifest_sha256_invalid",
            ),
            (
                {
                    "files": [
                        {
                            "path": "results.json",
                            "role": "results",
                            "sha256": "a" * 64,
                            "size": True,
                        }
                    ]
                },
                "manifest_size_invalid",
            ),
            (
                {
                    "files": [
                        {
                            "path": "results.json",
                            "role": "results",
                            "sha256": "a" * 64,
                            "size": 1.0,
                        }
                    ]
                },
                "manifest_size_invalid",
            ),
            (
                {"files": [{"path": "results.json", "role": 1, "sha256": "a" * 64, "size": 1}]},
                "manifest_role_invalid",
            ),
            (
                {
                    "files": [
                        {
                            "path": "report.html",
                            "role": "report",
                            "sha256": "a" * 64,
                            "size": 1,
                        }
                    ]
                },
                "manifest_results_missing",
            ),
        ]
        for patch, detail in cases:
            manifest = self._valid()
            manifest.update(patch)
            self.assertEqual(pk._valid_manifest(manifest), detail, patch)

    def test_rejects_duplicate_paths(self):
        manifest = self._valid()
        manifest["files"].append(dict(manifest["files"][0]))
        self.assertEqual(pk._valid_manifest(manifest), "manifest_path_duplicate")

    def test_rejects_bad_context(self):
        for key in ("package_version", "catalog_sha256", "backend_fingerprint"):
            manifest = self._valid()
            manifest["context"][key] = ""
            self.assertEqual(pk._valid_manifest(manifest), "manifest_context_invalid", key)

    def test_accepts_unknown_keys_and_roles_forward_compat(self):
        manifest = self._valid()
        manifest["future_extension"] = {"note": "ignored"}
        manifest["files"][0]["role"] = "results-v2"
        self.assertIsNone(pk._valid_manifest(manifest))

    def test_rejects_non_object(self):
        self.assertEqual(pk._valid_manifest([1, 2]), "manifest_not_an_object")


class DigestTests(unittest.TestCase):
    def test_digest_is_sha256_of_stored_bytes(self):
        manifest = {"b": 1, "a": [2, 3]}
        data = pk.manifest_bytes(manifest)
        self.assertEqual(data, b'{"a":[2,3],"b":1}')
        self.assertEqual(pk.bundle_digest(data), sha256(data).hexdigest())

    def test_digest_not_reserialisation(self):
        # A manifest whose stored bytes are canonical parses to the same
        # object, but the digest contract is explicitly over stored bytes.
        data = b'{"format":"x"}'
        self.assertEqual(pk.bundle_digest(data), sha256(data).hexdigest())
        reparsed = pk.manifest_bytes(json.loads(data))
        self.assertEqual(reparsed, data)

    def test_canonical_form_stable(self):
        self.assertEqual(
            pk.manifest_bytes({"z": 1, "a": "x"}), pk.manifest_bytes({"a": "x", "z": 1})
        )

    def test_catalog_digest_deterministic_and_contentful(self):
        first = pk.catalog_digest()
        second = pk.catalog_digest()
        self.assertEqual(first, second)
        self.assertRegex(first, r"^[0-9a-f]{64}$")


class GoldenRoundTripTests(unittest.TestCase):
    def test_pack_verify_ok_and_deterministic(self):
        with tempfile.TemporaryDirectory(prefix="c171_gold_") as tmpdir:
            report = _sample_report_dir(Path(tmpdir))
            bundle1, manifest, digest = _pack_to_bytes(report)
            bundle2, _, _ = _pack_to_bytes(report)
            self.assertEqual(bundle1, bundle2)
            self.assertRegex(digest, r"^[0-9a-f]{64}$")
            paths = sorted(entry["path"] for entry in manifest["files"])
            self.assertEqual(paths, ["report.html", "results.json"])
            roles = {entry["path"]: entry["role"] for entry in manifest["files"]}
            self.assertEqual(roles, {"report.html": "report", "results.json": "results"})
            report_out = _report_from_bundle(bundle1)
            self.assertEqual(report_out["verdict"], "ok")
            seal = {c["id"]: c["status"] for c in report_out["checks"]}
            for check_id in (
                "seal_container",
                "seal_manifest_schema",
                "seal_comment_digest",
                "seal_file_set",
                "seal_file_hashes",
            ):
                self.assertEqual(seal[check_id], "passed", check_id)
            self.assertEqual(report_out["bundle_digest"], digest)

    def test_include_files_sealed_under_inputs(self):
        with tempfile.TemporaryDirectory(prefix="c171_inc_") as tmpdir:
            tmpdir_path = Path(tmpdir)
            report = _sample_report_dir(tmpdir_path)
            dataset = tmpdir_path / "dataset.json"
            dataset.write_text('{"search": [], "held_out": []}', encoding="utf-8")
            bundle, manifest, _ = _pack_to_bytes(report, [dataset])
            paths = sorted(entry["path"] for entry in manifest["files"])
            self.assertIn("inputs/dataset.json", paths)
            report_out = _report_from_bundle(bundle)
            self.assertEqual(report_out["verdict"], "ok")

    def test_expect_digest_match_and_mismatch(self):
        with tempfile.TemporaryDirectory(prefix="c171_exp_") as tmpdir:
            report = _sample_report_dir(Path(tmpdir))
            bundle, _, digest = _pack_to_bytes(report)
            self.assertEqual(_report_from_bundle(bundle, digest)["verdict"], "ok")
            wrong = "0" * 64
            failed = _report_from_bundle(bundle, wrong)
            self.assertEqual(failed["verdict"], "failed")
            statuses = {c["id"]: c["status"] for c in failed["checks"]}
            self.assertEqual(statuses["seal_expected_digest"], "failed")


class TamperTests(unittest.TestCase):
    def _golden(self) -> bytes:
        with tempfile.TemporaryDirectory(prefix="c171_t_") as tmpdir:
            report = _sample_report_dir(Path(tmpdir))
            bundle, _, _ = _pack_to_bytes(report)
            return bundle

    def _verify_failed(self, bundle: bytes, check_id: str) -> None:
        report_out = _report_from_bundle(bundle)
        self.assertEqual(report_out["verdict"], "failed")
        statuses = {c["id"]: (c["status"], c["detail"]) for c in report_out["checks"]}
        self.assertEqual(statuses[check_id][0], "failed", check_id)

    def test_byte_flip_fails_hashes(self):
        bundle = self._golden()
        src = zipfile.ZipFile(io.BytesIO(bundle))
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w") as archive:
            for info in src.infolist():
                data = src.read(info.filename)
                if info.filename == "results.json":
                    data = data.replace(b'"decision"', b'"decisXon"', 1)
                archive.writestr(info, data)
            archive.comment = src.comment
        self._verify_failed(out.getvalue(), "seal_file_hashes")

    def test_comment_tamper_fails_digest(self):
        bundle = self._golden()
        src = zipfile.ZipFile(io.BytesIO(bundle))
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w") as archive:
            for info in src.infolist():
                archive.writestr(info, src.read(info.filename))
            archive.comment = b"mpl-bundle-sha256:" + b"0" * 64
        self._verify_failed(out.getvalue(), "seal_comment_digest")

    def test_comment_suffix_fails_digest(self):
        bundle = self._golden()
        src = zipfile.ZipFile(io.BytesIO(bundle))
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w") as archive:
            for info in src.infolist():
                archive.writestr(info, src.read(info.filename))
            archive.comment = src.comment + b" EXTRA"
        self._verify_failed(out.getvalue(), "seal_comment_digest")

    def test_extra_entry_fails_file_set(self):
        bundle = self._golden()
        src = zipfile.ZipFile(io.BytesIO(bundle))
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w") as archive:
            for info in src.infolist():
                archive.writestr(info, src.read(info.filename))
            archive.writestr(pk._zip_info("smuggled.txt"), b"x")
            archive.comment = src.comment
        self._verify_failed(out.getvalue(), "seal_file_set")

    def test_prepended_bytes_rejected(self):
        bundle = self._golden()
        with self.assertRaises(ValueError) as ctx:
            pk.verify_bundle_bytes(b"<html>" + bundle)
        self.assertEqual(str(ctx.exception), "bundle_container_invalid")

    def test_trailing_bytes_rejected(self):
        bundle = self._golden()
        with self.assertRaises(ValueError) as ctx:
            pk.verify_bundle_bytes(bundle + b"junk")
        self.assertEqual(str(ctx.exception), "bundle_container_invalid")

    def test_duplicate_entry_fails_container(self):
        bundle = self._golden()
        manifest_raw = _manifest_entry(bundle)
        digest_comment = b"mpl-bundle-sha256:" + pk.bundle_digest(manifest_raw).encode("ascii")
        out = _hostile_zip(
            [("manifest.json", manifest_raw), ("manifest.json", b"second")],
            comment=digest_comment,
        )
        report_out = _report_from_bundle(out)
        self.assertEqual(report_out["verdict"], "failed")
        statuses = {c["id"]: c["status"] for c in report_out["checks"]}
        self.assertEqual(statuses["seal_container"], "failed")

    def test_manifest_duplicate_key_rejected(self):
        manifest_raw = (
            b'{"format":"mpl-bundle/1","format":"mpl-bundle/1",'
            b'"context":{"package_version":"0.1.0","catalog_sha256":"'
            + b"0"
            * 64
            + b'","backend_fingerprint":"x"},'
            b'"files":[{"path":"results.json","role":"results","sha256":"'
            + b"a" * 64
            + b'","size":1}]}'
        )
        out = _hostile_zip(
            [("manifest.json", manifest_raw)],
            comment=b"mpl-bundle-sha256:" + pk.bundle_digest(manifest_raw).encode("ascii"),
        )
        self._verify_failed(out, "seal_manifest_schema")

    def test_results_duplicate_decision_key_is_unreadable(self):
        # The bundle path must agree with plain `verify`: a results.json
        # that load_results rejects is unreadable, never ok.
        with tempfile.TemporaryDirectory(prefix="c171_dupk_") as tmpdir:
            report = _sample_report_dir(Path(tmpdir))
            raw = (_SAMPLE / "results.json").read_text(encoding="utf-8")
            dup = raw.replace('"decision":', '"decision": "abstained", "decision":', 1)
            (report / "results.json").write_text(dup, encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                _pack_to_bytes(report)
            self.assertEqual(str(ctx.exception), "results_duplicate_field")

    def test_list_results_json_refused_at_pack(self):
        with tempfile.TemporaryDirectory(prefix="c171_lst_") as tmpdir:
            report = _sample_report_dir(Path(tmpdir))
            (report / "results.json").write_text("[1, 2]", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                _pack_to_bytes(report)
            self.assertEqual(str(ctx.exception), "results_not_an_object")

    def test_results_missing_fingerprint_refused(self):
        with tempfile.TemporaryDirectory(prefix="c171_fp_") as tmpdir:
            report = _sample_report_dir(Path(tmpdir))
            data = json.loads((_SAMPLE / "results.json").read_text(encoding="utf-8"))
            data.pop("backend_fingerprint", None)
            (report / "results.json").write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                _pack_to_bytes(report)
            self.assertEqual(str(ctx.exception), "pack_results_missing_fingerprint")


class PackDisciplineTests(unittest.TestCase):
    def test_symlink_entry_rejected(self):
        with tempfile.TemporaryDirectory(prefix="c171_sym_") as tmpdir:
            tmpdir_path = Path(tmpdir)
            report = _sample_report_dir(tmpdir_path)
            secret = tmpdir_path / "secret.txt"
            secret.write_text("top-secret", encoding="utf-8")
            os.symlink(secret, report / "leak.txt")
            with self.assertRaises(ValueError) as ctx:
                _pack_to_bytes(report)
            self.assertIn(str(ctx.exception), ("pack_entry_unreadable", "pack_entry_not_regular"))

    def test_subdirectory_rejected(self):
        with tempfile.TemporaryDirectory(prefix="c171_dir_") as tmpdir:
            report = _sample_report_dir(Path(tmpdir))
            (report / "nested").mkdir()
            with self.assertRaises(ValueError) as ctx:
                _pack_to_bytes(report)
            self.assertEqual(str(ctx.exception), "pack_subdirectory_not_allowed")

    def test_missing_results_refused(self):
        with tempfile.TemporaryDirectory(prefix="c171_nor_") as tmpdir:
            report = Path(tmpdir) / "report"
            report.mkdir()
            (report / "report.html").write_text("x", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                _pack_to_bytes(report)
            self.assertEqual(str(ctx.exception), "pack_results_missing")

    def test_include_duplicate_basename_rejected(self):
        with tempfile.TemporaryDirectory(prefix="c171_dbi_") as tmpdir:
            tmpdir_path = Path(tmpdir)
            report = _sample_report_dir(tmpdir_path)
            (tmpdir_path / "a").mkdir()
            (tmpdir_path / "b").mkdir()
            first = tmpdir_path / "a" / "same.json"
            second = tmpdir_path / "b" / "same.json"
            first.write_text("{}", encoding="utf-8")
            second.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                _pack_to_bytes(report, [first, second])
            self.assertEqual(str(ctx.exception), "pack_include_duplicate")

    def test_include_dotdot_basename_rejected(self):
        # Stored include names must be single canonical segments; "..",
        # empty, and separator-bearing names can never become one.
        self.assertFalse(pk._valid_segment(".."))
        self.assertFalse(pk._valid_segment(""))
        self.assertFalse(pk._valid_segment("a/b"))
        self.assertFalse(pk._valid_segment("a\\b"))


class CliTests(unittest.TestCase):
    def _run_main(self, argv: list[str]) -> tuple[int, str, str]:
        from model_protocol_lab.cli import main

        stdout, stderr = io.StringIO(), io.StringIO()
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = stdout, stderr
        try:
            code = main(argv)
        finally:
            sys.stdout, sys.stderr = old_out, old_err
        return code, stdout.getvalue(), stderr.getvalue()

    def test_pack_and_verify_round_trip_cli(self):
        with tempfile.TemporaryDirectory(prefix="c171_cli_") as tmpdir:
            report = _sample_report_dir(Path(tmpdir))
            bundle = Path(tmpdir) / "out.zip"
            code, _, stderr = self._run_main(["pack", str(report), "--output", str(bundle)])
            self.assertEqual(code, 0)
            self.assertIn("bundle digest", stderr)
            code, out, _ = self._run_main(["verify", "--bundle", str(bundle)])
            self.assertEqual(code, 0)
            self.assertIn('"verdict": "ok"', out)

    def test_verify_both_or_neither_is_exit_1(self):
        with tempfile.TemporaryDirectory(prefix="c171_cli2_") as tmpdir:
            report = _sample_report_dir(Path(tmpdir))
            code, _, err = self._run_main(["verify"])
            self.assertEqual(code, 1)
            self.assertIn("exactly one", err)
            code, _, err = self._run_main(["verify", str(report), "--bundle", str(report)])
            self.assertEqual(code, 1)

    def test_verify_bundle_missing_is_exit_1(self):
        code, _, err = self._run_main(["verify", "--bundle", "/nonexistent.zip"])
        self.assertEqual(code, 1)
        self.assertIn("verify: bundle_path_missing", err)

    def test_verify_expect_digest_requires_bundle(self):
        with tempfile.TemporaryDirectory(prefix="c171_cli3_") as tmpdir:
            report = _sample_report_dir(Path(tmpdir))
            code, _, err = self._run_main(["verify", str(report), "--expect-digest", "0" * 64])
            self.assertEqual(code, 1)
            self.assertIn("--expect-digest requires --bundle", err)

    def test_pack_refuses_output_inside_report_dir(self):
        with tempfile.TemporaryDirectory(prefix="c171_cli4_") as tmpdir:
            report = _sample_report_dir(Path(tmpdir))
            code, _, err = self._run_main(
                ["pack", str(report), "--output", str(report / "inner.zip")]
            )
            self.assertEqual(code, 1)
            self.assertIn("pack: output_inside_report_dir", err)

    def test_pack_refuses_existing_output(self):
        with tempfile.TemporaryDirectory(prefix="c171_cli5_") as tmpdir:
            report = _sample_report_dir(Path(tmpdir))
            bundle = Path(tmpdir) / "out.zip"
            bundle.write_bytes(b"existing")
            code, _, err = self._run_main(["pack", str(report), "--output", str(bundle)])
            self.assertEqual(code, 1)
            self.assertIn("pack: output_unwritable_or_exists", err)
            self.assertEqual(bundle.read_bytes(), b"existing")

    def test_hostile_manifest_path_sanitised(self):
        manifest = {
            "context": {
                "backend_fingerprint": "x",
                "catalog_sha256": "0" * 64,
                "package_version": "0.1.0",
            },
            "files": [],
            "format": "mpl-bundle/1",
        }
        hostile = pk._valid_manifest(manifest)
        self.assertEqual(hostile, "manifest_files_invalid")
        rendered = _display_id("\x1b[2J\u202eEvil")
        self.assertNotIn("\x1b", rendered)
        self.assertNotIn("\u202e", rendered)


# Tempdir mutation gates (4/4): each gate proves its scenario fails only
# because the guarded logic is present, never vacuously.
# ---------------------------------------------------------------------------


class PackMutationGateTests(unittest.TestCase):
    def _run_gate(self, name, old_text, new_text, scenario):
        source = Path(pk.__file__).read_text(encoding="utf-8")
        self.assertEqual(source.count(old_text), 1, f"mutant {name}: anchor not unique")
        mutated = source.replace(old_text, new_text)
        with tempfile.TemporaryDirectory(prefix="c171_mut_", dir="/tmp") as tmpdir:
            mod_path = os.path.join(tmpdir, f"pack_mut_{name}.py")
            with open(mod_path, "w", encoding="utf-8") as handle:
                handle.write(mutated)
            dotted = f"model_protocol_lab.pack_mut_{name}"
            import importlib.util

            spec = importlib.util.spec_from_file_location(dotted, mod_path)
            assert spec is not None and spec.loader is not None
            mutant = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mutant)
            scenario(pk)  # passes on the real module: non-vacuous
            with self.assertRaises(AssertionError, msg=f"mutant {name} was NOT caught"):
                scenario(mutant)

    def _tampered_comment_bundle(self) -> bytes:
        with tempfile.TemporaryDirectory(prefix="c171_g1_") as tmpdir:
            report = _sample_report_dir(Path(tmpdir))
            bundle, _, _ = _pack_to_bytes(report)
        src = zipfile.ZipFile(io.BytesIO(bundle))
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w") as archive:
            for info in src.infolist():
                archive.writestr(info, src.read(info.filename))
            archive.comment = b"mpl-bundle-sha256:" + b"0" * 64
        return out.getvalue()

    def _extra_entry_bundle(self) -> bytes:
        with tempfile.TemporaryDirectory(prefix="c171_g2_") as tmpdir:
            report = _sample_report_dir(Path(tmpdir))
            bundle, _, _ = _pack_to_bytes(report)
        src = zipfile.ZipFile(io.BytesIO(bundle))
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w") as archive:
            for info in src.infolist():
                archive.writestr(info, src.read(info.filename))
            archive.writestr(pk._zip_info("smuggled.txt"), b"x")
            archive.comment = src.comment
        return out.getvalue()

    def test_mutant_comment_digest_ignored_caught(self):
        def scenario(mod):
            report_out = mod.verify_bundle_bytes(self._tampered_comment_bundle())
            assert report_out["verdict"] == "failed", report_out
            statuses = {c["id"]: c["status"] for c in report_out["checks"]}
            assert statuses["seal_comment_digest"] == "failed", statuses

        self._run_gate(
            "comment_digest_ignored",
            'expected_comment = _COMMENT_PREFIX + digest_hex.encode("ascii")',
            "expected_comment = comment  # MUTANT: digest always matches",
            scenario,
        )

    def test_mutant_file_set_not_checked_caught(self):
        def scenario(mod):
            report_out = mod.verify_bundle_bytes(self._extra_entry_bundle())
            assert report_out["verdict"] == "failed", report_out
            statuses = {c["id"]: c["status"] for c in report_out["checks"]}
            assert statuses["seal_file_set"] == "failed", statuses

        self._run_gate(
            "file_set_not_checked",
            "if actual == expected_set:",
            "if True:  # MUTANT: any archive set passes",
            scenario,
        )

    def test_mutant_entry_cap_removed_caught(self):
        def scenario(mod):
            big = b"\x00" * (pk.MAX_BUNDLE_FILE_BYTES + 1)
            manifest = {
                "context": {
                    "backend_fingerprint": "x",
                    "catalog_sha256": "0" * 64,
                    "package_version": "0.1.0",
                },
                "files": [
                    {
                        "path": "big.bin",
                        "role": "report",
                        "sha256": sha256(big).hexdigest(),
                        "size": len(big),
                    },
                    {
                        "path": "results.json",
                        "role": "results",
                        "sha256": sha256(b"{}").hexdigest(),
                        "size": 2,
                    },
                ],
                "format": pk.FORMAT_ID,
            }
            raw = pk.manifest_bytes(manifest)
            comment = b"mpl-bundle-sha256:" + pk.bundle_digest(raw).encode("ascii")
            bundle = _hostile_zip([("manifest.json", raw), ("big.bin", big)], comment=comment)
            report_out = mod.verify_bundle_bytes(bundle)
            assert report_out["verdict"] == "failed", report_out
            schema = [c for c in report_out["checks"] if c["id"] == "seal_manifest_schema"][0]
            assert schema["status"] == "failed", schema
            assert schema["detail"] == "entry_too_large", schema

        self._run_gate(
            "entry_cap_removed",
            (
                "        if info.file_size > MAX_BUNDLE_FILE_BYTES:\n"
                '            caps_detail = "entry_too_large"'
            ),
            "        pass  # MUTANT: caps disabled",
            scenario,
        )

    def test_mutant_replay_layering_removed_caught(self):
        def scenario(mod):
            report_out = mod.verify_bundle_bytes(self._tampered_comment_bundle())
            assert report_out["verdict"] == "failed", report_out
            replay = [c for c in report_out["checks"] if not c["id"].startswith("seal_")]
            assert replay, report_out
            assert all(c["status"] == "unavailable" for c in replay), replay
            assert all(c["detail"] == "seal_failed" for c in replay), replay

        self._run_gate(
            "replay_layering_removed",
            '    seal_failed = any(c["status"] == "failed" for c in checks)',
            "    seal_failed = False  # MUTANT: replay runs on failed seals",
            scenario,
        )
