"""Cycle 171 gap tests: every guarded branch in pack.py exercised.

Written against the scoped-coverage report for ``pack.py`` -- each test
names the guard it closes. A test that passes because its setup failed
before the guarded line ran would be vacuous; these call the guarded
functions directly wherever possible so the guard cannot be skipped.
"""

import io
import json
import os
import struct
import tempfile
import unittest
import zipfile
from hashlib import sha256
from pathlib import Path
from unittest import mock

from model_protocol_lab import cli
from model_protocol_lab import pack as pk

SAMPLE = Path(__file__).resolve().parents[1] / "examples" / "sample-report" / "results.json"


def _sample_report_dir(base: Path) -> Path:
    report = base / "run"
    report.mkdir()
    (report / "results.json").write_bytes(SAMPLE.read_bytes())
    (report / "report.html").write_text("<html>demo</html>", encoding="utf-8")
    return report


def _sealed_bundle(report: Path) -> bytes:
    bundle, _, _ = pk.pack_bundle(report, [])
    return bundle


def _manifest_for(entries):
    manifest = {
        "context": {
            "backend_fingerprint": "x",
            "catalog_sha256": "0" * 64,
            "package_version": "0.1.0",
        },
        "files": [
            {
                "path": name,
                "role": "results" if name == "results.json" else "report",
                "sha256": sha256(data).hexdigest(),
                "size": len(data),
            }
            for name, data in entries
        ],
        "format": pk.FORMAT_ID,
    }
    return manifest


def _zip(entries, comment=b""):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries:
            archive.writestr(name, data)
    raw = bytearray(buffer.getvalue())
    if comment:
        eocd = raw.rfind(b"PK\x05\x06")
        struct.pack_into("<H", raw, eocd + 20, len(comment))
        raw.extend(comment)
    return bytes(raw)


def _sealed_shape(entries):
    """Zip whose comment carries the digest of its stored manifest."""
    manifest = _manifest_for(entries)
    raw_manifest = pk.manifest_bytes(manifest)
    comment = b"mpl-bundle-sha256:" + pk.bundle_digest(raw_manifest).encode("ascii")
    return _zip([("manifest.json", raw_manifest), *entries], comment=comment)


def _status(report, check_id):
    for check in report["checks"]:
        if check["id"] == check_id:
            return check["status"], check["detail"]
    raise AssertionError(f"no check {check_id}")


class PackReadGuardTests(unittest.TestCase):
    def test_fifo_is_not_regular(self):
        with tempfile.TemporaryDirectory(prefix="c171g_fifo_") as tmpdir:
            fifo = Path(tmpdir) / "pipe"
            os.mkfifo(fifo)
            with self.assertRaises(ValueError) as ctx:
                pk._read_regular_file(fifo, too_large_code="unused", cap=10)
            self.assertEqual(str(ctx.exception), "pack_entry_not_regular")

    def test_over_cap(self):
        with tempfile.TemporaryDirectory(prefix="c171g_cap_") as tmpdir:
            target = Path(tmpdir) / "f.bin"
            target.write_bytes(b"12345")
            with self.assertRaises(ValueError) as ctx:
                pk._read_regular_file(target, too_large_code="too_big_code", cap=4)
            self.assertEqual(str(ctx.exception), "too_big_code")

    def test_missing_file_is_unreadable_not_oserror(self):
        with tempfile.TemporaryDirectory(prefix="c171g_miss_") as tmpdir:
            with self.assertRaises(ValueError) as ctx:
                pk._read_regular_file(Path(tmpdir) / "nope", too_large_code="unused", cap=10)
            self.assertEqual(str(ctx.exception), "pack_entry_unreadable")

    def test_collect_missing_dir(self):
        with self.assertRaises(ValueError) as ctx:
            pk.collect_pack_files(Path("/nonexistent-dir-xyz"), [])
        self.assertEqual(str(ctx.exception), "pack_report_dir_missing")

    def test_collect_backslash_name_invalid(self):
        with tempfile.TemporaryDirectory(prefix="c171g_bs_") as tmpdir:
            report = Path(tmpdir) / "run"
            report.mkdir()
            (report / "results.json").write_bytes(SAMPLE.read_bytes())
            (report / "a\\b").write_bytes(b"x")
            with self.assertRaises(ValueError) as ctx:
                pk.collect_pack_files(report, [])
            self.assertEqual(str(ctx.exception), "pack_entry_name_invalid")

    def test_collect_include_duplicate(self):
        with tempfile.TemporaryDirectory(prefix="c171g_dup_") as tmpdir:
            report = _sample_report_dir(Path(tmpdir))
            extra = Path(tmpdir) / "notes.md"
            extra.write_text("n", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                pk.collect_pack_files(report, [extra, extra])
            self.assertEqual(str(ctx.exception), "pack_include_duplicate")

    def test_collect_include_missing(self):
        with tempfile.TemporaryDirectory(prefix="c171g_inc_") as tmpdir:
            report = _sample_report_dir(Path(tmpdir))
            with self.assertRaises(ValueError) as ctx:
                pk.collect_pack_files(report, [Path(tmpdir) / "ghost.md"])
            self.assertEqual(str(ctx.exception), "pack_entry_unreadable")


class LoadGuardTests(unittest.TestCase):
    def test_missing(self):
        with self.assertRaises(ValueError) as ctx:
            pk.load_bundle_bytes(Path("/nonexistent-bundle-xyz"))
        self.assertEqual(str(ctx.exception), "bundle_path_missing")

    def test_directory_not_regular(self):
        with tempfile.TemporaryDirectory(prefix="c171g_dir_") as tmpdir:
            with self.assertRaises(ValueError) as ctx:
                pk.load_bundle_bytes(Path(tmpdir))
            self.assertEqual(str(ctx.exception), "bundle_not_regular_file")

    def test_fifo_not_regular(self):
        with tempfile.TemporaryDirectory(prefix="c171g_lfifo_") as tmpdir:
            fifo = Path(tmpdir) / "pipe"
            os.mkfifo(fifo)
            with self.assertRaises(ValueError) as ctx:
                pk.load_bundle_bytes(fifo)
            self.assertEqual(str(ctx.exception), "bundle_not_regular_file")

    def test_oversize_sparse(self):
        with tempfile.TemporaryDirectory(prefix="c171g_big_") as tmpdir:
            target = Path(tmpdir) / "big.zip"
            with open(target, "wb") as handle:
                handle.truncate(pk.MAX_BUNDLE_ZIP_BYTES + 1)
            with self.assertRaises(ValueError) as ctx:
                pk.load_bundle_bytes(target)
            self.assertEqual(str(ctx.exception), "bundle_too_large")

    def test_growth_between_stat_and_read(self):
        # The post-read length guard is a race defence: simulate the file
        # growing after fstat by lying about st_size and shrinking the cap.
        with tempfile.TemporaryDirectory(prefix="c171g_race_") as tmpdir:
            target = Path(tmpdir) / "b.zip"
            target.write_bytes(b"0123456789")
            real_fstat = os.fstat

            def lying_fstat(fd, *args, **kwargs):
                result = real_fstat(fd, *args, **kwargs)
                return os.stat_result(
                    (
                        result.st_mode,
                        result.st_ino,
                        result.st_dev,
                        result.st_nlink,
                        result.st_uid,
                        result.st_gid,
                        0,
                        int(result.st_atime),
                        int(result.st_mtime),
                        int(result.st_ctime),
                    )
                )

            with (
                mock.patch.object(pk, "MAX_BUNDLE_ZIP_BYTES", 4),
                mock.patch("os.fstat", lying_fstat),
                self.assertRaises(ValueError) as ctx,
            ):
                pk.load_bundle_bytes(target)
            self.assertEqual(str(ctx.exception), "bundle_too_large")


class PreparseGuardTests(unittest.TestCase):
    def _patch_eocd(self, bundle: bytes, **fields) -> bytes:
        raw = bytearray(bundle)
        eocd = raw.rfind(b"PK\x05\x06")
        offsets = {
            "disk": eocd + 4,
            "cd_disk": eocd + 6,
            "entries_disk": eocd + 8,
            "entries": eocd + 10,
            "cd_size": eocd + 12,
            "cd_offset": eocd + 16,
        }
        for key, value in fields.items():
            if key in ("cd_size", "cd_offset"):
                struct.pack_into("<I", raw, offsets[key], value)
            else:
                struct.pack_into("<H", raw, offsets[key], value)
        return bytes(raw)

    def test_multidisk_rejected(self):
        with tempfile.TemporaryDirectory(prefix="c171g_md_") as tmpdir:
            bundle = _sealed_bundle(_sample_report_dir(Path(tmpdir)))
            with self.assertRaises(ValueError) as ctx:
                pk.verify_bundle_bytes(self._patch_eocd(bundle, disk=1))
            self.assertEqual(str(ctx.exception), "bundle_container_invalid")

    def test_zip64_sentinel_rejected(self):
        with tempfile.TemporaryDirectory(prefix="c171g_z64_") as tmpdir:
            bundle = _sealed_bundle(_sample_report_dir(Path(tmpdir)))
            with self.assertRaises(ValueError) as ctx:
                pk.verify_bundle_bytes(self._patch_eocd(bundle, cd_size=0xFFFFFFFF))
            self.assertEqual(str(ctx.exception), "bundle_container_invalid")

    def test_too_many_entries(self):
        with tempfile.TemporaryDirectory(prefix="c171g_many_") as tmpdir:
            bundle = _sealed_bundle(_sample_report_dir(Path(tmpdir)))
            patched = self._patch_eocd(bundle, entries=2000, entries_disk=2000)
            with self.assertRaises(ValueError) as ctx:
                pk.verify_bundle_bytes(patched)
            self.assertEqual(str(ctx.exception), "bundle_too_many_entries")

    def test_zero_entries(self):
        with tempfile.TemporaryDirectory(prefix="c171g_zero_") as tmpdir:
            bundle = _sealed_bundle(_sample_report_dir(Path(tmpdir)))
            patched = self._patch_eocd(bundle, entries=0, entries_disk=0)
            with self.assertRaises(ValueError) as ctx:
                pk.verify_bundle_bytes(patched)
            self.assertEqual(str(ctx.exception), "bundle_container_invalid")

    def test_bad_magic_start(self):
        with tempfile.TemporaryDirectory(prefix="c171g_magic_") as tmpdir:
            bundle = bytearray(_sealed_bundle(_sample_report_dir(Path(tmpdir))))
            bundle[0:4] = b"JUNK"
            with self.assertRaises(ValueError) as ctx:
                pk.verify_bundle_bytes(bytes(bundle))
            self.assertEqual(str(ctx.exception), "bundle_container_invalid")

    def test_corrupt_central_directory(self):
        with tempfile.TemporaryDirectory(prefix="c171g_cd_") as tmpdir:
            bundle = bytearray(_sealed_bundle(_sample_report_dir(Path(tmpdir))))
            eocd = bytes(bundle).rfind(b"PK\x05\x06")
            cd_offset = struct.unpack_from("<I", bundle, eocd + 16)[0]
            bundle[cd_offset + 3] = 0x09  # break the CD signature
            with self.assertRaises(ValueError) as ctx:
                pk.verify_bundle_bytes(bytes(bundle))
            self.assertEqual(str(ctx.exception), "bundle_container_invalid")


class ContainerAndCapsGapTests(unittest.TestCase):
    def test_entry_count_mismatch(self):
        with tempfile.TemporaryDirectory(prefix="c171g_cnt_") as tmpdir:
            bundle = _sealed_bundle(_sample_report_dir(Path(tmpdir)))
            raw = bytearray(bundle)
            eocd = raw.rfind(b"PK\x05\x06")
            struct.pack_into("<H", raw, eocd + 8, 4)
            struct.pack_into("<H", raw, eocd + 10, 4)
            report = pk.verify_bundle_bytes(bytes(raw))
            self.assertEqual(_status(report, "seal_container"), ("failed", "entry_count_mismatch"))

    def test_directory_entry(self):
        entries = [("results.json", SAMPLE.read_bytes())]
        bundle = _zip(
            [
                ("manifest.json", pk.manifest_bytes(_manifest_for(entries))),
                ("sub/", b""),
                *entries,
            ],
            comment=b"mpl-bundle-sha256:"
            + pk.bundle_digest(pk.manifest_bytes(_manifest_for(entries))).encode(),
        )
        report = pk.verify_bundle_bytes(bundle)
        self.assertEqual(_status(report, "seal_container"), ("failed", "directory_entry_present"))

    def test_entry_path_invalid(self):
        entries = [("results.json", SAMPLE.read_bytes())]
        raw_manifest = pk.manifest_bytes(_manifest_for(entries))
        bundle = _zip(
            [("manifest.json", raw_manifest), ("../evil", b"x"), *entries],
            comment=b"mpl-bundle-sha256:" + pk.bundle_digest(raw_manifest).encode(),
        )
        report = pk.verify_bundle_bytes(bundle)
        self.assertEqual(_status(report, "seal_container"), ("failed", "entry_path_invalid"))

    def test_total_too_large(self):
        filler = b"\x00" * (60 * 1024 * 1024)
        entries = [(f"f{i}.bin", filler) for i in range(5)]
        entries.append(("results.json", SAMPLE.read_bytes()))
        bundle = _sealed_shape(entries)
        report = pk.verify_bundle_bytes(bundle)
        self.assertEqual(_status(report, "seal_manifest_schema"), ("failed", "total_too_large"))


class ManifestSchemaGapTests(unittest.TestCase):
    def _report_for_manifest(self, manifest_obj=None, raw_manifest=None):
        entries = [("results.json", SAMPLE.read_bytes())]
        raw = raw_manifest if raw_manifest is not None else pk.manifest_bytes(manifest_obj)
        comment = b"mpl-bundle-sha256:" + pk.bundle_digest(raw).encode()
        bundle = _zip([("manifest.json", raw), *entries], comment=comment)
        return pk.verify_bundle_bytes(bundle)

    def test_file_entry_not_dict(self):
        manifest = _manifest_for([("results.json", SAMPLE.read_bytes())])
        manifest["files"] = [1]
        report = self._report_for_manifest(manifest)
        self.assertEqual(
            _status(report, "seal_manifest_schema"),
            ("failed", "manifest_file_entry_invalid"),
        )

    def test_context_version_missing(self):
        manifest = _manifest_for([("results.json", SAMPLE.read_bytes())])
        del manifest["context"]["package_version"]
        report = self._report_for_manifest(manifest)
        self.assertEqual(
            _status(report, "seal_manifest_schema"),
            ("failed", "manifest_context_invalid"),
        )

    def test_context_catalog_bad(self):
        manifest = _manifest_for([("results.json", SAMPLE.read_bytes())])
        manifest["context"]["catalog_sha256"] = "zz"
        report = self._report_for_manifest(manifest)
        self.assertEqual(
            _status(report, "seal_manifest_schema"),
            ("failed", "manifest_context_invalid"),
        )

    def test_context_fingerprint_bad(self):
        manifest = _manifest_for([("results.json", SAMPLE.read_bytes())])
        manifest["context"]["backend_fingerprint"] = ""
        report = self._report_for_manifest(manifest)
        self.assertEqual(
            _status(report, "seal_manifest_schema"),
            ("failed", "manifest_context_invalid"),
        )

    def test_manifest_not_object(self):
        report = self._report_for_manifest(raw_manifest=b"[1,2]")
        self.assertEqual(
            _status(report, "seal_manifest_schema"), ("failed", "manifest_not_an_object")
        )

    def test_manifest_unparseable(self):
        report = self._report_for_manifest(raw_manifest=b"\xff\xfe{")
        self.assertEqual(
            _status(report, "seal_manifest_schema"), ("failed", "manifest_unparseable")
        )

    def test_expected_digest_unavailable_when_manifest_broken(self):
        report = self._report_for_manifest(raw_manifest=b"\xff\xfe{")
        # expect_digest path is exercised through verify_bundle_bytes below
        entries = [("results.json", SAMPLE.read_bytes())]
        raw = b"\xff\xfe{"
        comment = b"mpl-bundle-sha256:" + pk.bundle_digest(raw).encode()
        bundle = _zip([("manifest.json", raw), *entries], comment=comment)
        report = pk.verify_bundle_bytes(bundle, expect_digest="0" * 64)
        self.assertEqual(
            _status(report, "seal_expected_digest"), ("unavailable", "manifest_unusable")
        )
        self.assertEqual(
            _status(report, "seal_comment_digest"), ("unavailable", "manifest_unusable")
        )
        self.assertEqual(_status(report, "seal_file_set"), ("unavailable", "manifest_unusable"))


class ReadErrorGapTests(unittest.TestCase):
    def test_corrupt_results_crc_surfaces_read_error(self):
        with tempfile.TemporaryDirectory(prefix="c171g_crc2_") as tmpdir:
            bundle = bytearray(_sealed_bundle(_sample_report_dir(Path(tmpdir))))
            eocd = bytes(bundle).rfind(b"PK\x05\x06")
            cd_offset = struct.unpack_from("<I", bundle, eocd + 16)[0]
            pos = cd_offset
            while bundle[pos : pos + 4] == b"PK\x01\x02":
                name_len, extra_len, comment_len = struct.unpack_from("<HHH", bundle, pos + 28)
                name = bytes(bundle[pos + 46 : pos + 46 + name_len]).decode()
                if name == "results.json":
                    struct.pack_into("<I", bundle, pos + 16, 0)  # CRC -> 0
                    break
                pos += 46 + name_len + extra_len + comment_len
            report = pk.verify_bundle_bytes(bytes(bundle))
            self.assertEqual(
                _status(report, "seal_manifest_schema"),
                ("failed", "bundle_entry_read_failed"),
            )

    def test_hash_entry_missing(self):
        entries = [("results.json", SAMPLE.read_bytes())]
        manifest = _manifest_for(entries)
        manifest["files"].append(
            {
                "path": "ghost.txt",
                "role": "report",
                "sha256": "0" * 64,
                "size": 3,
            }
        )
        raw = pk.manifest_bytes(manifest)
        bundle = _zip(
            [("manifest.json", raw), *entries],
            comment=b"mpl-bundle-sha256:" + pk.bundle_digest(raw).encode(),
        )
        report = pk.verify_bundle_bytes(bundle)
        self.assertEqual(_status(report, "seal_file_hashes"), ("failed", "entry_missing"))


class CliGapTests(unittest.TestCase):
    def _run_main(self, argv):
        import contextlib as _ctx

        out, err = io.StringIO(), io.StringIO()
        with _ctx.redirect_stdout(out), _ctx.redirect_stderr(err):
            code = cli.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_pack_missing_dir_exit_1(self):
        code, _, err = self._run_main(["pack", "/nonexistent-dir-xyz"])
        self.assertEqual(code, 1)
        self.assertIn("pack_report_dir_missing", err)

    def test_pack_default_output(self):
        with tempfile.TemporaryDirectory(prefix="c171g_def_") as tmpdir:
            report = _sample_report_dir(Path(tmpdir))
            code, _, err = self._run_main(["pack", str(report)])
            self.assertEqual(code, 0)
            default_out = Path(tmpdir) / "run.bundle.zip"
            self.assertTrue(default_out.is_file())
            code, _, _ = self._run_main(["verify", "--bundle", str(default_out)])
            self.assertEqual(code, 0)

    def test_pack_output_unwritable(self):
        with tempfile.TemporaryDirectory(prefix="c171g_unw_") as tmpdir:
            report = _sample_report_dir(Path(tmpdir))
            target = Path(tmpdir) / "no-such-dir" / "out.zip"
            code, _, err = self._run_main(["pack", str(report), "--output", str(target)])
            self.assertEqual(code, 1)
            self.assertIn("output_unwritable_or_exists", err)

    def test_pack_include_is_output(self):
        with tempfile.TemporaryDirectory(prefix="c171g_iio_") as tmpdir:
            report = _sample_report_dir(Path(tmpdir))
            target = Path(tmpdir) / "out.zip"
            code, _, err = self._run_main(
                ["pack", str(report), "--output", str(target), "--include", str(target)]
            )
            self.assertEqual(code, 1)
            self.assertIn("include_is_output", err)

    def test_verify_bundle_missing_file(self):
        code, _, err = self._run_main(["verify", "--bundle", "/nonexistent-bundle-xyz"])
        self.assertEqual(code, 1)
        self.assertIn("bundle_path_missing", err)

    def test_replay_failure_with_intact_seal(self):
        # Edit the decision BEFORE packing: the seal is honest, the
        # replay must fail, and the exit code must be 2 (not 1).
        with tempfile.TemporaryDirectory(prefix="c171g_replay_") as tmpdir:
            report_dir = Path(tmpdir) / "run"
            report_dir.mkdir()
            data = json.loads(SAMPLE.read_text(encoding="utf-8"))
            data["selected_candidate"] = "no-such-candidate"
            (report_dir / "results.json").write_text(json.dumps(data), encoding="utf-8")
            bundle = Path(tmpdir) / "out.zip"
            code, _, _ = self._run_main(["pack", str(report_dir), "--output", str(bundle)])
            self.assertEqual(code, 0)
            code, out, err = self._run_main(["verify", "--bundle", str(bundle)])
            self.assertEqual(code, 2)
            parsed = json.loads(out)
            statuses = {c["id"]: c["status"] for c in parsed["checks"]}
            self.assertEqual(statuses["seal_file_hashes"], "passed")
            self.assertEqual(statuses["selection_decision"], "failed")


class SecondGapTests(unittest.TestCase):
    def _run_main(self, argv):
        import contextlib as _ctx

        out, err = io.StringIO(), io.StringIO()
        with _ctx.redirect_stdout(out), _ctx.redirect_stderr(err):
            code = cli.main(argv)
        return code, out.getvalue(), err.getvalue()

    def _patch_local_and_cd_size(self, bundle: bytes, target: str, delta: int) -> bytes:
        raw = bytearray(bundle)
        # local header
        pos = 0
        while raw[pos : pos + 4] == b"PK\x03\x04":
            name_len, extra_len = struct.unpack_from("<HH", raw, pos + 26)
            name = bytes(raw[pos + 30 : pos + 30 + name_len]).decode()
            if name == target:
                actual = struct.unpack_from("<I", raw, pos + 22)[0]
                struct.pack_into("<I", raw, pos + 22, actual + delta)
            pos += 30 + name_len + extra_len
            # skip compressed payload via CD later; find next local header
            nxt = bytes(raw).find(b"PK\x03\x04", pos)
            if nxt == -1:
                break
            pos = nxt
        # central directory
        eocd = bytes(raw).rfind(b"PK\x05\x06")
        cd_offset = struct.unpack_from("<I", raw, eocd + 16)[0]
        pos = cd_offset
        while raw[pos : pos + 4] == b"PK\x01\x02":
            name_len, extra_len, comment_len = struct.unpack_from("<HHH", raw, pos + 28)
            name = bytes(raw[pos + 46 : pos + 46 + name_len]).decode()
            if name == target:
                actual = struct.unpack_from("<I", raw, pos + 24)[0]
                struct.pack_into("<I", raw, pos + 24, actual + delta)
                break
            pos += 46 + name_len + extra_len + comment_len
        return bytes(raw)

    def test_context_not_dict(self):
        manifest = _manifest_for([("results.json", SAMPLE.read_bytes())])
        manifest["context"] = "nope"
        report = ManifestSchemaGapTests()._report_for_manifest(manifest)
        self.assertEqual(
            _status(report, "seal_manifest_schema"),
            ("failed", "manifest_context_invalid"),
        )

    def test_manifest_missing_entry(self):
        bundle = _zip([("results.json", SAMPLE.read_bytes())])
        report = pk.verify_bundle_bytes(bundle)
        self.assertEqual(_status(report, "seal_manifest_schema"), ("failed", "manifest_missing"))
        self.assertEqual(_status(report, "seal_file_set"), ("unavailable", "manifest_unusable"))
        self.assertEqual(
            _status(report, "seal_comment_digest"), ("unavailable", "manifest_unusable")
        )

    def test_oversize_entry_fails_at_caps_layer(self):
        # Layering is deliberate: a cap failure means the manifest is
        # never trusted, so downstream checks report unavailable rather
        # than re-deriving their own verdict on an unvalidated manifest.
        big = b"\x00" * (pk.MAX_BUNDLE_FILE_BYTES + 1)
        entries = [("results.json", SAMPLE.read_bytes()), ("big.bin", big)]
        manifest = _manifest_for(entries)
        raw = pk.manifest_bytes(manifest)
        bundle = _zip(
            [("manifest.json", raw), *entries],
            comment=b"mpl-bundle-sha256:" + pk.bundle_digest(raw).encode(),
        )
        report = pk.verify_bundle_bytes(bundle)
        self.assertEqual(_status(report, "seal_manifest_schema"), ("failed", "entry_too_large"))
        self.assertEqual(_status(report, "seal_file_hashes"), ("unavailable", "manifest_unusable"))

    def test_hash_read_error_nonstored_entry(self):
        with tempfile.TemporaryDirectory(prefix="c171g_hre_") as tmpdir:
            report = _sample_report_dir(Path(tmpdir))
            bundle = bytearray(_sealed_bundle(report))
            eocd = bytes(bundle).rfind(b"PK\x05\x06")
            cd_offset = struct.unpack_from("<I", bundle, eocd + 16)[0]
            pos = cd_offset
            while bundle[pos : pos + 4] == b"PK\x01\x02":
                name_len, extra_len, comment_len = struct.unpack_from("<HHH", bundle, pos + 28)
                name = bytes(bundle[pos + 46 : pos + 46 + name_len]).decode()
                if name == "report.html":
                    struct.pack_into("<I", bundle, pos + 16, 0)  # CRC -> 0
                    break
                pos += 46 + name_len + extra_len + comment_len
            report_out = pk.verify_bundle_bytes(bytes(bundle))
            self.assertEqual(_status(report_out, "seal_manifest_schema")[0], "passed")
            self.assertEqual(
                _status(report_out, "seal_file_hashes"),
                ("failed", "bundle_entry_read_failed"),
            )

    def test_declared_size_lie(self):
        with tempfile.TemporaryDirectory(prefix="c171g_lie_") as tmpdir:
            report = _sample_report_dir(Path(tmpdir))
            bundle = self._patch_local_and_cd_size(_sealed_bundle(report), "report.html", 5)
            report_out = pk.verify_bundle_bytes(bundle)
            self.assertEqual(
                _status(report_out, "seal_file_hashes"),
                ("failed", "bundle_entry_read_failed"),
            )

    def test_declared_size_smaller_aborts_stream(self):
        # Declared size BELOW the real payload: the reader must abort
        # mid-stream (bounded memory), not buffer the whole lie first.
        with tempfile.TemporaryDirectory(prefix="c171g_lie2_") as tmpdir:
            report = _sample_report_dir(Path(tmpdir))
            bundle = self._patch_local_and_cd_size(_sealed_bundle(report), "report.html", -5)
            report_out = pk.verify_bundle_bytes(bundle)
            self.assertEqual(
                _status(report_out, "seal_file_hashes"),
                ("failed", "bundle_entry_read_failed"),
            )

    def test_load_unreadable_via_mocked_open(self):
        with tempfile.TemporaryDirectory(prefix="c171g_unr_") as tmpdir:
            target = Path(tmpdir) / "b.zip"
            target.write_bytes(b"PK")
            with (
                mock.patch("os.open", side_effect=OSError("denied")),
                self.assertRaises(ValueError) as ctx,
            ):
                pk.load_bundle_bytes(target)
            self.assertEqual(str(ctx.exception), "bundle_unreadable")

    def test_preparse_entries_disk_mismatch(self):
        with tempfile.TemporaryDirectory(prefix="c171g_edm_") as tmpdir:
            bundle = _sealed_bundle(_sample_report_dir(Path(tmpdir)))
            raw = bytearray(bundle)
            eocd = raw.rfind(b"PK\x05\x06")
            struct.pack_into("<H", raw, eocd + 8, 1)  # entries_disk != entries
            with self.assertRaises(ValueError) as ctx:
                pk.verify_bundle_bytes(bytes(raw))
            self.assertEqual(str(ctx.exception), "bundle_container_invalid")

    def test_preparse_cd_disk_nonzero(self):
        with tempfile.TemporaryDirectory(prefix="c171g_cdd_") as tmpdir:
            bundle = _sealed_bundle(_sample_report_dir(Path(tmpdir)))
            raw = bytearray(bundle)
            eocd = raw.rfind(b"PK\x05\x06")
            struct.pack_into("<H", raw, eocd + 6, 1)  # cd on disk 1
            with self.assertRaises(ValueError) as ctx:
                pk.verify_bundle_bytes(bytes(raw))
            self.assertEqual(str(ctx.exception), "bundle_container_invalid")

    def test_run_pack_resolve_oserror(self):
        with tempfile.TemporaryDirectory(prefix="c171g_res_") as tmpdir:
            report = _sample_report_dir(Path(tmpdir))
            with mock.patch.object(Path, "resolve", side_effect=OSError("nope")):
                code, _, err = self._run_main(["pack", str(report)])
            self.assertEqual(code, 1)
            self.assertIn("path_unresolvable", err)

    def test_run_pack_include_resolve_oserror(self):
        with tempfile.TemporaryDirectory(prefix="c171g_ires_") as tmpdir:
            report = _sample_report_dir(Path(tmpdir))
            include = Path(tmpdir) / "inc.md"
            include.write_text("x", encoding="utf-8")
            real_resolve = Path.resolve

            def picky(self, *args, **kwargs):
                if self.name == "inc.md":
                    raise OSError("nope")
                return real_resolve(self, *args, **kwargs)

            with mock.patch.object(Path, "resolve", picky):
                code, _, err = self._run_main(["pack", str(report), "--include", str(include)])
            self.assertEqual(code, 1)
            self.assertIn("path_unresolvable", err)

    def test_run_pack_include_cli_success(self):
        with tempfile.TemporaryDirectory(prefix="c171g_ics_") as tmpdir:
            report = _sample_report_dir(Path(tmpdir))
            include = Path(tmpdir) / "notes.md"
            include.write_text("context", encoding="utf-8")
            bundle = Path(tmpdir) / "out.zip"
            code, _, err = self._run_main(
                ["pack", str(report), "--output", str(bundle), "--include", str(include)]
            )
            self.assertEqual(code, 0)
            self.assertIn("sealed 3 files", err)

    def test_run_pack_write_failure_removes_partial(self):
        class _BadHandle:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def write(self, data):
                raise OSError("disk full")

        with tempfile.TemporaryDirectory(prefix="c171g_wf_") as tmpdir:
            report = _sample_report_dir(Path(tmpdir))
            bundle = Path(tmpdir) / "out.zip"
            real_fdopen = os.fdopen

            def fake_fdopen(fd, mode, *args, **kwargs):
                if mode == "wb":
                    return _BadHandle()
                return real_fdopen(fd, mode, *args, **kwargs)

            with mock.patch("os.fdopen", fake_fdopen):
                code, _, err = self._run_main(["pack", str(report), "--output", str(bundle)])
            self.assertEqual(code, 1)
            self.assertIn("output_write_failed", err)
            self.assertFalse(bundle.exists())

    def test_verify_bundle_not_zip(self):
        with tempfile.TemporaryDirectory(prefix="c171g_nz_") as tmpdir:
            target = Path(tmpdir) / "b.zip"
            target.write_bytes(b"this is not a zip at all")
            code, _, err = self._run_main(["verify", "--bundle", str(target)])
            self.assertEqual(code, 1)
            self.assertIn("bundle_container_invalid", err)

    def test_include_name_invalid_via_cli(self):
        with tempfile.TemporaryDirectory(prefix="c171g_ini_") as tmpdir:
            report = _sample_report_dir(Path(tmpdir))
            bad = Path(tmpdir) / "a\\b"
            bad.write_text("x", encoding="utf-8")
            bundle = Path(tmpdir) / "out.zip"
            code, _, err = self._run_main(
                ["pack", str(report), "--output", str(bundle), "--include", str(bad)]
            )
            self.assertEqual(code, 1)
            self.assertIn("pack_include_name_invalid", err)

    def test_preparse_entries_sentinel(self):
        with tempfile.TemporaryDirectory(prefix="c171g_es_") as tmpdir:
            bundle = _sealed_bundle(_sample_report_dir(Path(tmpdir)))
            raw = bytearray(bundle)
            eocd = raw.rfind(b"PK\x05\x06")
            struct.pack_into("<H", raw, eocd + 10, 0xFFFF)
            with self.assertRaises(ValueError) as ctx:
                pk.verify_bundle_bytes(bytes(raw))
            self.assertEqual(str(ctx.exception), "bundle_container_invalid")

    def test_preparse_decoy_eocd_signature_fails_closed(self):
        # The comment contains a fake EOCD signature near its end. Our
        # preparse skips it (length mismatch) and finds the real EOCD --
        # but zipfile's own scan is fooled into an empty infolist, and
        # the disagreement fails closed as entry_count_mismatch. (A
        # bundle from `pack` can never do this: its comment is hex-only.)
        with tempfile.TemporaryDirectory(prefix="c171g_decoy_") as tmpdir:
            bundle = bytearray(_sealed_bundle(_sample_report_dir(Path(tmpdir))))
            eocd = bundle.rfind(b"PK\x05\x06")
            comment_len = struct.unpack_from("<H", bundle, eocd + 20)[0]
            fake = b"PK\x05\x06" + b"\x00" * 16 + b"\x07\x00"  # 22 bytes, len 7
            assert comment_len >= len(fake)
            comment_start = eocd + 22
            bundle[comment_start + comment_len - len(fake) : comment_start + comment_len] = fake
            report = pk.verify_bundle_bytes(bytes(bundle))
            self.assertEqual(_status(report, "seal_container"), ("failed", "entry_count_mismatch"))
            self.assertEqual(report["verdict"], "failed")

    def test_preparse_cd_offset_sentinel(self):
        with tempfile.TemporaryDirectory(prefix="c171g_cos_") as tmpdir:
            bundle = _sealed_bundle(_sample_report_dir(Path(tmpdir)))
            raw = bytearray(bundle)
            eocd = raw.rfind(b"PK\x05\x06")
            struct.pack_into("<I", raw, eocd + 16, 0xFFFFFFFF)
            with self.assertRaises(ValueError) as ctx:
                pk.verify_bundle_bytes(bytes(raw))
            self.assertEqual(str(ctx.exception), "bundle_container_invalid")

    def test_preparse_truncated_trailing_signature(self):
        # A stray signature too close to EOF to hold a record must be
        # skipped, not crash the scan; the archive then fails strict
        # parsing (trailing bytes) as designed.
        with tempfile.TemporaryDirectory(prefix="c171g_trunc_") as tmpdir:
            bundle = _sealed_bundle(_sample_report_dir(Path(tmpdir)))
            with self.assertRaises(ValueError) as ctx:
                pk.verify_bundle_bytes(bundle + b"PK\x05\x06xyz")
            self.assertEqual(str(ctx.exception), "bundle_container_invalid")

    def test_read_entry_aborts_on_lying_stream(self):
        # The in-memory lie: the declared size is smaller than the
        # stream, so the reader must abort instead of buffering it.
        # Deterministic at the unit level: zipfile's own CRC check can
        # pre-empt this path on real archives, so the guard is driven
        # with a stubbed handle that outruns the declared size.
        from types import SimpleNamespace

        class _Handle:
            def __init__(self):
                self.n = 0

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self, size):
                self.n += 1
                return b"x" * 10 if self.n <= 2 else b""

        class _Archive:
            def open(self, info, mode):
                return _Handle()

        info = SimpleNamespace(file_size=5)
        with self.assertRaises(ValueError) as ctx:
            pk._read_entry(_Archive(), info)
        self.assertEqual(str(ctx.exception), "bundle_entry_read_failed")

    def test_run_pack_fdopen_failure_closes_fd(self):
        with tempfile.TemporaryDirectory(prefix="c171g_fo_") as tmpdir:
            report = _sample_report_dir(Path(tmpdir))
            bundle = Path(tmpdir) / "out.zip"
            real_fdopen = os.fdopen

            def fake_fdopen(fd, mode, *args, **kwargs):
                if mode == "wb":
                    raise OSError("no handle")
                return real_fdopen(fd, mode, *args, **kwargs)

            with mock.patch("os.fdopen", fake_fdopen):
                code, _, err = self._run_main(["pack", str(report), "--output", str(bundle)])
            self.assertEqual(code, 1)
            self.assertIn("output_write_failed", err)
            self.assertFalse(bundle.exists())


if __name__ == "__main__":
    unittest.main()
