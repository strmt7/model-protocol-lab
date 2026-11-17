"""W5-2 (RTI-02): the pack producer must not emit bundles its own
verifier rejects.

Pre-fix, ``pack_bundle`` capped each file but had no aggregate
ceilings: a report with 1,022 tiny includes packed successfully into
a 1,025-entry bundle (the manifest is the unreserved overhead) that
``verify_bundle_bytes`` then rejected with ``bundle_too_many_entries``.
The producer now enforces the verifier's limits before any bundle
bytes are built: the entry ceiling counting the manifest (same fixed
code the verifier raises), the aggregate uncompressed ceiling
(entries + manifest bytes), and the stored-archive ceiling, computed
exactly for the STORED layout ``build_bundle_bytes`` emits.

The boundary pair at the real constants kills the off-by-one mutant:
dropping the +1 manifest reservation makes the 1,025-entry arm pack
instead of refusing. The byte ceilings are proven with reduced
budgets (monkeypatched constants, tiny files), per the audit — no
large allocations.
"""

from __future__ import annotations

import argparse
import io
import json
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import model_protocol_lab
from model_protocol_lab import pack as pk

SAMPLE = Path(__file__).resolve().parents[1] / "examples" / "sample-report" / "results.json"


def _sample_report_dir(base: Path) -> Path:
    """A two-file report dir (the cycle-171 legacy-pair fixture)."""
    report = base / "run"
    report.mkdir()
    data = json.loads(SAMPLE.read_bytes())
    data.pop("report_pair", None)
    (report / "results.json").write_text(json.dumps(data), encoding="utf-8")
    (report / "report.html").write_text("<html>demo</html>", encoding="utf-8")
    return report


def _tiny_includes(base: Path, count: int) -> list[Path]:
    """``count`` tiny include files, each a few bytes."""
    includes = []
    for index in range(count):
        path = base / f"inc{index:04d}.txt"
        path.write_bytes(b"x")
        includes.append(path)
    return includes


class EntryCeilingTests(unittest.TestCase):
    """The real-constant boundary: 1,024 entries pass, 1,025 refuse."""

    def test_1024_entries_pack_and_verify(self) -> None:
        """Test 1024 entries pack and verify."""
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            report = _sample_report_dir(base)
            # 2 report files + 1,021 includes + the manifest = 1,024.
            includes = _tiny_includes(base, pk.MAX_BUNDLE_ENTRIES - 3)
            bundle, _manifest, _digest = pk.pack_bundle(report, includes)
            verdict = pk.verify_bundle_bytes(bundle)
            self.assertEqual(verdict["verdict"], "ok")

    def test_1025_entries_refused_at_pack_with_verifier_code(self) -> None:
        """Test 1025 entries refused at pack with verifier code."""
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            report = _sample_report_dir(base)
            # 2 report files + 1,022 includes + the manifest = 1,025.
            includes = _tiny_includes(base, pk.MAX_BUNDLE_ENTRIES - 2)
            with self.assertRaises(ValueError) as caught:
                pk.pack_bundle(report, includes)
            self.assertEqual(str(caught.exception), "bundle_too_many_entries")

    def test_manifest_reservation_is_exact(self) -> None:
        """Test manifest reservation is exact."""
        # Reduced budget: with the ceiling set to the entry count
        # alone (forgetting the manifest), the pack must refuse;
        # set to entries + 1, it must succeed. A producer that
        # forgets the +1 fails the first arm.
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            report = _sample_report_dir(base)
            includes = _tiny_includes(base, 3)  # 2 + 3 = 5 entries
            with mock.patch.object(pk, "MAX_BUNDLE_ENTRIES", 5):
                with self.assertRaises(ValueError) as caught:
                    pk.pack_bundle(report, includes)
                self.assertEqual(str(caught.exception), "bundle_too_many_entries")
            with mock.patch.object(pk, "MAX_BUNDLE_ENTRIES", 6):
                pk.pack_bundle(report, includes)

    def test_cli_refusal_creates_no_output(self) -> None:
        """Test CLI refusal creates no output."""
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            report = _sample_report_dir(base)
            includes = _tiny_includes(base, pk.MAX_BUNDLE_ENTRIES - 2)
            output = base / "out.bundle.zip"
            args = argparse.Namespace(
                report_dir=str(report),
                include=[str(path) for path in includes],
                output=str(output),
            )
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                code = pk.run_pack(args, None)
            self.assertEqual(code, 1)
            self.assertIn("bundle_too_many_entries", stderr.getvalue())
            self.assertFalse(output.exists())


class ByteCeilingTests(unittest.TestCase):
    """Reduced-budget proofs for the two aggregate byte ceilings."""

    def _bundle_total(self, report: Path) -> tuple[int, int]:
        """(uncompressed total incl. manifest, stored zip size)."""
        entries, _results = pk.collect_pack_files(report, [])
        files = [
            {
                "path": name,
                "role": pk._role_for(name),
                "sha256": pk._sha256_hex(data),
                "size": len(data),
            }
            for name, data in entries
        ]
        results = json.loads(dict(entries)["results.json"])
        manifest = pk.build_manifest(
            files,
            {
                "backend_fingerprint": results["backend_fingerprint"],
                "catalog_sha256": pk.catalog_digest(),
                "package_version": model_protocol_lab.__version__,
            },
        )
        manifest_data = pk.manifest_bytes(manifest)
        total = len(manifest_data) + sum(len(data) for _name, data in entries)
        return total, len(pk.build_bundle_bytes(manifest_data, entries))

    def test_total_ceiling_refuses_below_and_admits_at(self) -> None:
        """Test total ceiling refuses below and admits at."""
        with TemporaryDirectory() as tmp:
            report = _sample_report_dir(Path(tmp))
            total, _zip_size = self._bundle_total(report)
            with mock.patch.object(pk, "MAX_BUNDLE_TOTAL_BYTES", total - 1):
                with self.assertRaises(ValueError) as caught:
                    pk.pack_bundle(report, [])
                self.assertEqual(str(caught.exception), "pack_total_too_large")
            with mock.patch.object(pk, "MAX_BUNDLE_TOTAL_BYTES", total):
                pk.pack_bundle(report, [])

    def test_zip_ceiling_refuses_below_and_admits_at(self) -> None:
        """Test zip ceiling refuses below and admits at."""
        with TemporaryDirectory() as tmp:
            report = _sample_report_dir(Path(tmp))
            _total, zip_size = self._bundle_total(report)
            with mock.patch.object(pk, "MAX_BUNDLE_ZIP_BYTES", zip_size - 1):
                with self.assertRaises(ValueError) as caught:
                    pk.pack_bundle(report, [])
                self.assertEqual(str(caught.exception), "pack_zip_too_large")
            with mock.patch.object(pk, "MAX_BUNDLE_ZIP_BYTES", zip_size):
                pk.pack_bundle(report, [])


class StoredSizeArithmeticTests(unittest.TestCase):
    """The producer's stored-size computation is exact, not estimated."""

    def test_matches_built_bundle_for_varied_shapes(self) -> None:
        """Test matches built bundle for varied shapes."""
        shapes: list[list[tuple[str, bytes]]] = [
            [],
            [("a.txt", b"")],
            [("a.txt", b"x" * 1000), ("inputs/b.bin", bytes(range(256)))],
            [("résumé.txt", "données".encode())],
        ]
        for entries in shapes:
            manifest_data = b'{"context":{},"files":[],"format":"mpl-bundle/1"}'
            built = pk.build_bundle_bytes(manifest_data, entries)
            self.assertEqual(pk._stored_bundle_size(manifest_data, entries), len(built))


if __name__ == "__main__":
    unittest.main()
