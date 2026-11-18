"""W4 push 4: the report-pair completion boundary (ORCH-D02).

``save_report`` publishes ``results.json`` + ``report.html`` as a
staged pair: both files are written to ``.part`` staging files
(flush + fsync), renamed HTML first and JSON last — the final
``results.json`` is the commit point, stamped at construction
with ``report_pair.html_sha256`` over the exact HTML bytes and
``complete: true``. A hard crash between the renames therefore
leaves an orphan HTML (or only ``.part`` files), never a plausible
but wrong pair. Consumers verify a present stamp before trusting
a pair: ``diffreports.load_results`` (the shared loader, also
used by ``verify``) refuses a stamped pair whose sibling is
missing or whose digest mismatches with
``report_pair_incomplete``; ``pack`` refuses to seal such a pair
and reports the same code as a failed ``report_pair`` check when
verifying a bundle. Legacy pairs (no stamp) verify exactly as
they did before W4 — absence is not incompleteness.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from model_protocol_lab import core, pack, reporting
from model_protocol_lab.diffreports import (
    _sha256_regular_file,
    load_results,
    report_pair_incomplete,
)
from model_protocol_lab.experiments import run_experiment

BOUNDARY = (999,)
MAX_NEW_TOKENS = 40


class ScriptedBackend:
    """Scripted backend."""

    name = "scripted"
    fingerprint = "w4-pair-scripted"

    def __init__(self):
        """Init."""
        self.generate_calls = 0

    def encode(self, text):
        """Encode."""
        return tuple(map(ord, text))

    def decode(self, ids):
        """Decode."""
        return "".join(chr(i) for i in ids if 32 <= i < 127)

    def generate(self, ids, *, max_new_tokens, seed):
        """Generate."""
        self.generate_calls += 1
        text = "answer text GREEN"
        encoded = tuple(map(ord, text))
        assert len(encoded) <= max_new_tokens
        return core.Generation(encoded + (1,) * (max_new_tokens - len(encoded)), "limit")


def _result():
    """Result."""
    fixtures = tuple(
        core.Fixture(
            id=fid,
            messages=(core.Message(role="user", content=f"question-{fid}"),),
            expected="GREEN",
            group=f"group-{fid}",
        )
        for fid in ("S1", "H1")
    )
    split = core.Split(search=fixtures[:1], held_out=fixtures[1:])
    formats = (
        core.Format(
            name="cand-A",
            bos="",
            user_prefix="[cand-A]U:",
            assistant_prefix="A:",
            turn_end="\n",
        ),
    )
    return run_experiment(
        ScriptedBackend(),
        formats,
        split,
        boundary=BOUNDARY,
        max_new_tokens=MAX_NEW_TOKENS,
        seed=7,
    )


def _save(directory):
    """Save."""
    return reporting.save_report(_result(), Path(directory) / "report")


def _checks_by_id(out):
    """Checks by id."""
    return {c["id"]: c for c in out["checks"]}


class SavePairTests(unittest.TestCase):
    """Test save pair."""

    def test_save_stamps_pair_and_cleans_staging(self):
        """Test save stamps pair and cleans staging."""
        with TemporaryDirectory() as tmp:
            json_path, html_path = _save(tmp)
            data = json.loads(json_path.read_text(encoding="utf-8"))
            stamp = data["report_pair"]
            self.assertIs(stamp["complete"], True)
            self.assertEqual(
                stamp["html_sha256"],
                hashlib.sha256(html_path.read_bytes()).hexdigest(),
            )
            names = sorted(p.name for p in json_path.parent.iterdir())
            self.assertEqual(names, ["report.html", "results.json"])
            for path in (json_path, html_path):
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            loaded = load_results(json_path.parent)
            self.assertEqual(loaded["report_pair"], stamp)

    def test_save_fsyncs_files_and_directory(self):
        """Test save fsyncs files and directory."""
        if os.name != "posix":
            self.skipTest("directory fsync is POSIX-only")
        real_fsync = os.fsync
        calls = []

        def counting(fd):
            """Counting."""
            calls.append(fd)
            return real_fsync(fd)

        with TemporaryDirectory() as tmp, mock.patch.object(os, "fsync", counting):
            _save(tmp)
        self.assertGreaterEqual(len(calls), 3)

    def test_failed_second_rename_leaves_no_pair(self):
        """Test failed second rename leaves no pair."""
        real_replace = os.replace
        calls = []

        def flaky(src, dst):
            """Flaky."""
            calls.append(dst)
            if len(calls) == 2:
                raise OSError("simulated crash between renames")
            return real_replace(src, dst)

        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "report"
            with mock.patch.object(os, "replace", flaky), self.assertRaises(OSError):
                reporting.save_report(_result(), target)
            names = sorted(p.name for p in target.iterdir())
            self.assertEqual(names, [])


class CrashLeftoverResumeTests(unittest.TestCase):
    """BH12-2: a crash inside save_report's publication window
    leaves .part staging files or an orphan report.html beside the
    progress journal. The --resume re-save (same flags the CLI
    uses) must recover the directory -- the leftovers are the
    run's own unpublished staging -- while anything else beside
    the journal still refuses fail-closed."""

    def _claimed_dir(self, tmp, leftovers=(), journal=True):
        """Claimed dir."""
        target = Path(tmp) / "report"
        target.mkdir()
        if journal:
            (target / "progress.jsonl").write_text("{}\n", encoding="utf-8")
        for name in leftovers:
            (target / name).write_text("x", encoding="utf-8")
        return target

    def _resume_save(self, target):
        """Resume save."""
        return reporting.save_report(
            _result(),
            target,
            _allow_empty_existing=True,
            _allow_progress_journal=True,
        )

    def _assert_valid_pair(self, target):
        """Assert valid pair."""
        data = json.loads((target / "results.json").read_text(encoding="utf-8"))
        stamp = data["report_pair"]
        self.assertTrue(stamp["complete"])
        self.assertEqual(
            stamp["html_sha256"],
            hashlib.sha256((target / "report.html").read_bytes()).hexdigest(),
        )
        names = sorted(p.name for p in target.iterdir())
        self.assertEqual(names, ["progress.jsonl", "report.html", "results.json"])

    def test_resume_save_with_journal_only(self):
        """Test resume save with journal only (control)."""
        with TemporaryDirectory() as tmp:
            target = self._claimed_dir(tmp)
            self._resume_save(target)
            self._assert_valid_pair(target)

    def test_resume_save_recovers_from_part_leftovers(self):
        """Test resume save recovers from part leftovers (BH12-2)."""
        for leftover in ("results.json.part", "report.html.part"):
            with self.subTest(leftover=leftover), TemporaryDirectory() as tmp:
                target = self._claimed_dir(tmp, (leftover,))
                self._resume_save(target)
                self._assert_valid_pair(target)

    def test_resume_save_recovers_from_orphan_html(self):
        """Test resume save recovers from orphan html (BH12-2)."""
        with TemporaryDirectory() as tmp:
            target = self._claimed_dir(tmp, ("report.html",))
            self._resume_save(target)
            self._assert_valid_pair(target)

    def test_resume_save_still_refuses_foreign_file(self):
        """Test resume save still refuses foreign file."""
        with TemporaryDirectory() as tmp:
            target = self._claimed_dir(tmp, ("notes.txt",))
            with self.assertRaises(FileExistsError):
                self._resume_save(target)

    def test_resume_save_still_refuses_completed_pair(self):
        """Test resume save still refuses completed pair."""
        with TemporaryDirectory() as tmp:
            target = self._claimed_dir(tmp, ("report.html", "results.json"))
            with self.assertRaises(FileExistsError):
                self._resume_save(target)

    def test_resume_save_still_refuses_leftovers_without_journal(self):
        """Test resume save still refuses leftovers without journal."""
        with TemporaryDirectory() as tmp:
            target = self._claimed_dir(tmp, ("report.html.part",), journal=False)
            with self.assertRaises(FileExistsError):
                self._resume_save(target)

    def test_resume_save_still_refuses_symlink_leftover(self):
        """Test resume save still refuses symlink leftover."""
        with TemporaryDirectory() as tmp:
            target = self._claimed_dir(tmp)
            os.symlink(Path(tmp) / "elsewhere", target / "report.html.part")
            with self.assertRaises(FileExistsError):
                self._resume_save(target)


class LoadPairTests(unittest.TestCase):
    """Test load pair."""

    def test_tampered_html_refused(self):
        """Test tampered html refused."""
        with TemporaryDirectory() as tmp:
            _, html_path = _save(tmp)
            data = bytearray(html_path.read_bytes())
            data[10] ^= 0x01
            html_path.write_bytes(bytes(data))
            with self.assertRaises(ValueError) as caught:
                load_results(html_path.parent)
            self.assertEqual(str(caught.exception), "report_pair_incomplete")

    def test_missing_html_refused(self):
        """Test missing html refused."""
        with TemporaryDirectory() as tmp:
            json_path, html_path = _save(tmp)
            html_path.unlink()
            with self.assertRaises(ValueError) as caught:
                load_results(json_path.parent)
            self.assertEqual(str(caught.exception), "report_pair_incomplete")

    def test_incomplete_stamp_refused(self):
        """Test incomplete stamp refused."""
        with TemporaryDirectory() as tmp:
            json_path, _ = _save(tmp)
            data = json.loads(json_path.read_text(encoding="utf-8"))
            data["report_pair"]["complete"] = False
            json_path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                load_results(json_path.parent)
            self.assertEqual(str(caught.exception), "report_pair_incomplete")

    def test_malformed_stamp_refused(self):
        """Test malformed stamp refused."""
        with TemporaryDirectory() as tmp:
            json_path, _ = _save(tmp)
            data = json.loads(json_path.read_text(encoding="utf-8"))
            data["report_pair"]["html_sha256"] = 42
            json_path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                load_results(json_path.parent)
            self.assertEqual(str(caught.exception), "report_pair_incomplete")

    def test_legacy_pair_loads(self):
        """Test legacy pair loads."""
        with TemporaryDirectory() as tmp:
            json_path, _ = _save(tmp)
            data = json.loads(json_path.read_text(encoding="utf-8"))
            del data["report_pair"]
            json_path.write_text(json.dumps(data), encoding="utf-8")
            loaded = load_results(json_path.parent)
            self.assertNotIn("report_pair", loaded)

    def test_stamped_results_file_refuses_tampered_sibling(self):
        """Test stamped results file refuses tampered sibling."""
        with TemporaryDirectory() as tmp:
            json_path, html_path = _save(tmp)
            data = bytearray(html_path.read_bytes())
            data[10] ^= 0x01
            html_path.write_bytes(bytes(data))
            with self.assertRaises(ValueError) as caught:
                load_results(json_path)
            self.assertEqual(str(caught.exception), "report_pair_incomplete")

    def test_lone_stamped_json_loads_as_data(self):
        """Test lone stamped json loads as data."""
        with TemporaryDirectory() as tmp:
            json_path, _ = _save(tmp)
            lone = Path(tmp) / "perturbed.json"
            lone.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
            loaded = load_results(lone)
            self.assertIn("report_pair", loaded)

    def test_orphan_html_is_no_report(self):
        """Test orphan html is no report."""
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "report"
            target.mkdir()
            (target / "report.html").write_text("<html/>", encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                load_results(target)
            self.assertEqual(str(caught.exception), "results_unreadable")

    def test_staging_orphans_are_no_report(self):
        """Test staging orphans are no report."""
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "report"
            target.mkdir()
            (target / "report.html.part").write_text("<html/>", encoding="utf-8")
            (target / "results.json.part").write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                load_results(target)
            self.assertEqual(str(caught.exception), "results_unreadable")


class PackPairTests(unittest.TestCase):
    """Test pack pair."""

    def test_pack_seals_and_verifies_stamped_pair(self):
        """Test pack seals and verifies stamped pair."""
        with TemporaryDirectory() as tmp:
            _save(tmp)
            bundle, _, _ = pack.pack_bundle(Path(tmp) / "report", [])
            out = pack.verify_bundle_bytes(bundle)
            self.assertEqual(out["verdict"], "ok")
            self.assertEqual(_checks_by_id(out)["report_pair"]["status"], "passed")

    def test_pack_refuses_mismatched_pair(self):
        """Test pack refuses mismatched pair."""
        with TemporaryDirectory() as tmp:
            _, html_path = _save(tmp)
            data = bytearray(html_path.read_bytes())
            data[10] ^= 0x01
            html_path.write_bytes(bytes(data))
            with self.assertRaises(ValueError) as caught:
                pack.pack_bundle(Path(tmp) / "report", [])
            self.assertEqual(str(caught.exception), "report_pair_incomplete")

    def test_verify_bundle_mismatched_pair_fails_check(self):
        """Test verify bundle mismatched pair fails check."""
        with TemporaryDirectory() as tmp:
            json_path, html_path = _save(tmp)
            results_bytes = json_path.read_bytes()
            entries = [
                ("report.html", b"<html>forged</html>"),
                ("results.json", results_bytes),
            ]
            files = [
                {
                    "path": name,
                    "role": pack._role_for(name),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "size": len(data),
                }
                for name, data in entries
            ]
            results_data = json.loads(results_bytes)
            context = {
                "backend_fingerprint": results_data["backend_fingerprint"],
                "catalog_sha256": pack.catalog_digest(),
                "package_version": pack.__version__,
            }
            manifest = pack.build_manifest(files, context)
            manifest_data = pack.manifest_bytes(manifest)
            bundle = pack.build_bundle_bytes(manifest_data, entries)
            out = pack.verify_bundle_bytes(bundle)
            pair = _checks_by_id(out)["report_pair"]
            self.assertEqual(pair["status"], "failed")
            self.assertEqual(pair["detail"], "report_pair_incomplete")
            self.assertEqual(_checks_by_id(out)["seal_file_hashes"]["status"], "passed")

    def test_verify_bundle_legacy_pair_unstamped(self):
        """Test verify bundle legacy pair unstamped."""
        with TemporaryDirectory() as tmp:
            json_path, _ = _save(tmp)
            data = json.loads(json_path.read_text(encoding="utf-8"))
            del data["report_pair"]
            json_path.write_text(json.dumps(data), encoding="utf-8")
            bundle, _, _ = pack.pack_bundle(Path(tmp) / "report", [])
            out = pack.verify_bundle_bytes(bundle)
            pair = _checks_by_id(out)["report_pair"]
            self.assertEqual(pair["status"], "unavailable")
            self.assertEqual(pair["detail"], "legacy_unstamped")


class PairCoverageTests(unittest.TestCase):
    """Test pair coverage."""

    def test_predicate_legacy_and_malformed(self):
        """Test predicate legacy and malformed."""
        self.assertFalse(report_pair_incomplete({}, None))
        self.assertFalse(report_pair_incomplete({"other": 1}, "abc"))
        self.assertTrue(report_pair_incomplete({"report_pair": "x"}, "abc"))
        good = {"report_pair": {"complete": True, "html_sha256": "abc"}}
        self.assertFalse(report_pair_incomplete(good, "abc"))
        self.assertTrue(report_pair_incomplete(good, None))
        self.assertTrue(report_pair_incomplete(good, "abd"))

    def test_nondict_stamp_refused_on_load(self):
        """Test nondict stamp refused on load."""
        with TemporaryDirectory() as tmp:
            json_path, _ = _save(tmp)
            data = json.loads(json_path.read_text(encoding="utf-8"))
            data["report_pair"] = "stamped"
            json_path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                load_results(json_path.parent)
            self.assertEqual(str(caught.exception), "report_pair_incomplete")

    def test_html_as_directory_refused(self):
        """Test html as directory refused."""
        with TemporaryDirectory() as tmp:
            json_path, html_path = _save(tmp)
            html_path.unlink()
            html_path.mkdir()
            with self.assertRaises(ValueError) as caught:
                load_results(json_path.parent)
            self.assertEqual(str(caught.exception), "report_pair_incomplete")

    def test_hasher_read_failure_returns_none(self):
        """Test hasher read failure returns none."""
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "report.html"
            target.write_text("<html/>", encoding="utf-8")
            with mock.patch.object(os, "fdopen", side_effect=OSError("read failed")):
                self.assertIsNone(_sha256_regular_file(target))

    def test_staged_fdopen_failure_cleans_up(self):
        """Test staged fdopen failure cleans up."""
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "report"
            with (
                mock.patch.object(os, "fdopen", side_effect=OSError("fdopen failed")),
                self.assertRaises(OSError),
            ):
                reporting.save_report(_result(), target)
            self.assertEqual(sorted(p.name for p in target.iterdir()), [])

    def test_fsync_directory_non_posix_noop(self):
        """Test fsync directory non posix noop."""
        # The Path is built BEFORE os.name is patched: on
        # Python 3.11 pathlib dispatches Path() on os.name, so
        # constructing it under the "nt" patch tries to build a
        # WindowsPath on a POSIX host and raises
        # NotImplementedError (the GitHub 3.11 leg's error).
        # _fsync_directory reads os.name itself and returns
        # before touching the path on non-POSIX, so the patch
        # region only needs the call.
        with TemporaryDirectory() as tmp:
            path = Path(tmp)
            with mock.patch.object(os, "name", "nt"):
                self.assertIsNone(reporting._fsync_directory(path))

    def test_verify_bundle_seal_failed_records_pair_unavailable(self):
        """Test verify bundle seal failed records pair unavailable."""
        with TemporaryDirectory() as tmp:
            _save(tmp)
            bundle, _, _ = pack.pack_bundle(Path(tmp) / "report", [])
            buffer = io.BytesIO()
            with zipfile.ZipFile(io.BytesIO(bundle)) as src, zipfile.ZipFile(buffer, "w") as dst:
                for info in src.infolist():
                    data = src.read(info.filename)
                    if info.filename == "report.html":
                        data = data + b"tampered"
                    dst.writestr(info, data)
            out = pack.verify_bundle_bytes(buffer.getvalue())
            pair = _checks_by_id(out)["report_pair"]
            self.assertEqual(pair["status"], "unavailable")
            self.assertEqual(pair["detail"], "seal_failed")


if __name__ == "__main__":
    unittest.main()
