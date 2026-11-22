"""External index mirror and freshness contracts, with no package or network.

Adapted from strmt7/FlitzZip tools/test_cocoindex_agent_search.py (MIT).
"""

import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.cocoindex_agent_search import (
    CONFIG_DIGEST,
    PROJECT_SETTINGS,
    ROOT,
    SETTINGS,
    ccc_path,
    home,
    index,
    paths,
    prepare_mirror,
    search,
    source_digest,
    tracked_files,
)


class CocoIndexAgentSearchTests(unittest.TestCase):
    """Coco index agent search tests."""

    def test_successful_search_records_real_use_without_query_text(self):
        """Purpose: Prove routing receipt semantics.
        Inputs: fresh source/real wrapper/mock package.
        Outputs: deduplicated use record.
        """
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            source = base / "source.py"
            source.write_text("pass\n", encoding="utf-8")
            entries = [("source.py", source)]
            digest = source_digest(entries)
            mirror = paths(base, base)[1]
            marker = base / "active" / (mirror.name + ".json")
            marker.parent.mkdir()
            marker.write_text(
                json.dumps(
                    {
                        "source_digest": digest,
                        "config_digest": CONFIG_DIGEST,
                        "cocoindex_code": "0.2.42",
                    }
                ),
                encoding="utf-8",
            )
            hit = {
                "file_path": "source.py",
                "start_line": 1,
                "end_line": 1,
                "score": 0.9,
                "content": "pass",
            }
            result = subprocess.CompletedProcess(
                [], 0, stdout=json.dumps({"success": True, "results": [hit, hit]})
            )
            query = "private conceptual question"
            with (
                patch("tools.cocoindex_agent_search.tracked_files", return_value=entries),
                patch("tools.cocoindex_agent_search._run", return_value=result),
                contextlib.redirect_stdout(io.StringIO()) as output,
            ):
                search(base, query, 5, repo=base)
            self.assertIn("results=1", output.getvalue())
            receipt = marker.with_name(mirror.name + ".usage.json").read_text()
            self.assertNotIn(query, receipt)
            self.assertEqual(json.loads(receipt)["source_digest"], digest)
            self.assertEqual(json.loads(receipt)["hit_count"], 1)
            source.write_text("changed\n", encoding="utf-8")
            original = receipt
            with (
                patch("tools.cocoindex_agent_search.tracked_files", return_value=entries),
                self.assertRaisesRegex(RuntimeError, "index is stale"),
            ):
                search(base, query, 5, repo=base)
            self.assertEqual(marker.with_name(mirror.name + ".usage.json").read_text(), original)

    def test_source_changed_during_search_cannot_record_current_use(self):
        """Purpose: Close the search freshness race.
        Inputs: source mutation during the package call.
        Outputs: explicit failure without a new receipt.
        """
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            source = base / "source.py"
            source.write_text("pass\n", encoding="utf-8")
            entries = [("source.py", source)]
            mirror = paths(base, base)[1]
            marker = base / "active" / (mirror.name + ".json")
            marker.parent.mkdir()
            marker.write_text(
                json.dumps(
                    {
                        "source_digest": source_digest(entries),
                        "config_digest": CONFIG_DIGEST,
                        "cocoindex_code": "0.2.42",
                    }
                ),
                encoding="utf-8",
            )

            def mutate_source(*args, **kwargs):
                """Purpose: Inject source drift
                Inputs: package argv
                Outputs: successful stale search data.
                """
                source.write_text("changed\n", encoding="utf-8")
                return subprocess.CompletedProcess(
                    [], 0, stdout=json.dumps({"success": True, "results": []})
                )

            with (
                patch("tools.cocoindex_agent_search.tracked_files", return_value=entries),
                patch("tools.cocoindex_agent_search._run", side_effect=mutate_source),
                self.assertRaisesRegex(RuntimeError, "source changed during semantic search"),
            ):
                search(base, "conceptual question", 5, repo=base)
            self.assertFalse(marker.with_name(mirror.name + ".usage.json").exists())

    def test_home_requires_absolute_override(self):
        """Test home requires absolute override."""
        with (
            patch.dict("os.environ", {"AGENT_CODE_HOME": "relative"}),
            self.assertRaisesRegex(ValueError, "absolute"),
        ):
            home()

    def test_home_rejects_checkout_state(self):
        """Test home rejects checkout state."""
        with (
            patch.dict("os.environ", {"AGENT_CODE_HOME": str(ROOT / "index-state")}),
            self.assertRaisesRegex(ValueError, "outside the checkout"),
        ):
            home()

    def test_mirror_copies_exact_bytes_and_prunes_stale_generated_file(self):
        """Test mirror copies exact bytes and prunes stale generated file."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.py"
            source.write_bytes(b"print('safe')\n")
            mirror = root / "index-mirror"
            mirror.mkdir()
            stale = mirror / "stale.py"
            stale.write_text("old", encoding="utf-8")
            entries = [("nested/source.py", source)]
            prepare_mirror(mirror, entries)
            self.assertEqual((mirror / "nested/source.py").read_bytes(), source.read_bytes())
            self.assertFalse(stale.exists())
            first = source_digest(entries)
            source.write_bytes(b"print('changed')\n")
            self.assertNotEqual(first, source_digest(entries))
            prepare_mirror(mirror, entries)
            self.assertEqual((mirror / "nested/source.py").read_bytes(), source.read_bytes())

    def test_search_rejects_missing_index_without_starting_package(self):
        """Test search rejects missing index without starting package."""
        with (
            tempfile.TemporaryDirectory() as temp,
            self.assertRaisesRegex(RuntimeError, "no active index"),
        ):
            search(Path(temp), "training labels", 5, repo=Path(temp))

    def test_search_rejects_unreliable_wildcard_filter(self):
        """Test search rejects unreliable wildcard filter."""
        with (
            tempfile.TemporaryDirectory() as temp,
            self.assertRaisesRegex(ValueError, "wildcard path filters"),
        ):
            search(Path(temp), "training labels", 5, repo=Path(temp), path="model_protocol_lab/*")

    def test_search_rejects_stale_source_before_package_call(self):
        """Test search rejects stale source before package call."""
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            source = base / "source.py"
            source.write_text("pass\n", encoding="utf-8")
            mirror = paths(base, base)[1]
            marker = base / "active" / (mirror.name + ".json")
            marker.parent.mkdir()
            marker.write_text(
                json.dumps({"source_digest": "incorrect", "cocoindex_code": "0.2.42"}),
                encoding="utf-8",
            )
            with (
                patch(
                    "tools.cocoindex_agent_search.tracked_files",
                    return_value=[("source.py", source)],
                ),
                self.assertRaisesRegex(RuntimeError, "index is stale"),
            ):
                search(base, "training labels", 5, repo=base)

    def test_mirror_paths_are_per_repository(self):
        """Test mirror paths are per repository."""
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            self.assertNotEqual(paths(base, base / "a")[1], paths(base, base / "b")[1])

    def test_tracked_files_rejects_escape_and_real_env(self):
        """Test tracked files rejects escape and real env."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for unsafe in (b"../outside\0", b".env\0", b"secrets/key.txt\0", b"tls/private.pem\0"):
                with (
                    self.subTest(unsafe=unsafe),
                    patch("tools.cocoindex_agent_search.subprocess.run") as run,
                    self.assertRaises(ValueError),
                ):
                    run.return_value.stdout = unsafe
                    tracked_files(root)


class CocoIndexIndexSettingsTests(unittest.TestCase):
    """First-run bootstrap contracts for index() settings handling.

    Census section 5 adjudication: the project-settings guard refused
    the absent case (previous is None) as if the file had changed, so
    the documented install -> index flow was unreachable from a fresh
    state. These tests pin the resolved semantics: absent bootstraps
    from the pinned constants, identical indexes, changed refuses.
    """

    def _fresh_state(self, temp: str) -> tuple[Path, Path, Path, Path]:
        """Build a fresh base/repo pair with a fake installed ccc.

        Returns (base, global_settings, project_settings, marker).
        """
        base = Path(temp)
        source = base / "source.py"
        source.write_text("pass\n", encoding="utf-8")
        venv, mirror, config, _runtime = paths(base, base)
        command = ccc_path(venv)
        command.parent.mkdir(parents=True)
        command.write_text("#!/bin/sh\n", encoding="utf-8")
        global_settings = config / "global_settings.yml"
        project_settings = mirror / ".cocoindex_code" / "settings.yml"
        marker = base / "active" / (mirror.name + ".json")
        return base, global_settings, project_settings, marker

    def _index_with_mocks(self, base: Path, run) -> None:
        source = base / "source.py"
        entries = [("source.py", source)]
        with (
            patch("tools.cocoindex_agent_search.tracked_files", return_value=entries),
            patch("tools.cocoindex_agent_search._run", run),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            index(base, repo=base)

    @staticmethod
    def _ok_run(calls: list) -> object:
        def run(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, stdout="Indexing: done error: 0")

        return run

    def test_index_bootstraps_absent_settings_and_indexes(self):
        """Purpose: Pin the first-run fix (census section 5.2 case 1).
        Inputs: fresh state, neither settings file present.
        Outputs: both files written from the pinned constants, ccc
        index invoked once, marker bound to CONFIG_DIGEST.
        Mutant killer: reverting the guard to `previous !=
        PROJECT_SETTINGS` makes index() raise here.
        """
        with tempfile.TemporaryDirectory() as temp:
            base, global_settings, project_settings, marker = self._fresh_state(temp)
            calls: list = []
            self._index_with_mocks(base, self._ok_run(calls))
            self.assertEqual(project_settings.read_text(encoding="utf-8"), PROJECT_SETTINGS)
            self.assertEqual(global_settings.read_text(encoding="utf-8"), SETTINGS)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][1], "index")
            self.assertEqual(json.loads(marker.read_text())["config_digest"], CONFIG_DIGEST)

    def test_index_accepts_identical_project_settings(self):
        """Purpose: Regression-pin the pre-fix working path (case 2).
        Inputs: project settings pre-written byte-identical.
        Outputs: index succeeds; file content unchanged.
        """
        with tempfile.TemporaryDirectory() as temp:
            base, _global, project_settings, marker = self._fresh_state(temp)
            project_settings.parent.mkdir(parents=True)
            project_settings.write_text(PROJECT_SETTINGS, encoding="utf-8")
            calls: list = []
            self._index_with_mocks(base, self._ok_run(calls))
            self.assertEqual(project_settings.read_text(encoding="utf-8"), PROJECT_SETTINGS)
            self.assertEqual(len(calls), 1)
            self.assertTrue(marker.exists())

    def test_index_refuses_changed_project_settings(self):
        """Purpose: Pin the drift refusal the guard exists for (case 3).
        Inputs: project settings present with one extra byte.
        Outputs: RuntimeError, ccc never invoked, bytes unmodified,
        no marker written.
        Mutant killer: dropping the refusal makes index() proceed.
        """
        with tempfile.TemporaryDirectory() as temp:
            base, _global, project_settings, marker = self._fresh_state(temp)
            project_settings.parent.mkdir(parents=True)
            drifted = PROJECT_SETTINGS + "x"
            project_settings.write_text(drifted, encoding="utf-8")
            calls: list = []
            with self.assertRaisesRegex(RuntimeError, "project settings changed"):
                self._index_with_mocks(base, self._ok_run(calls))
            self.assertEqual(calls, [])
            self.assertEqual(project_settings.read_text(encoding="utf-8"), drifted)
            self.assertFalse(marker.exists())

    def test_index_refuses_changed_global_settings(self):
        """Purpose: Pin the sibling global guard's refusal (case 4).
        Inputs: global settings present with one extra byte.
        Outputs: RuntimeError, ccc never invoked.
        """
        with tempfile.TemporaryDirectory() as temp:
            base, global_settings, _project, _marker = self._fresh_state(temp)
            global_settings.parent.mkdir(parents=True)
            global_settings.write_text(SETTINGS + "x", encoding="utf-8")
            calls: list = []
            with self.assertRaisesRegex(RuntimeError, "CocoIndex settings changed"):
                self._index_with_mocks(base, self._ok_run(calls))
            self.assertEqual(calls, [])

    def test_index_bootstrapped_state_is_a_fixed_point(self):
        """Purpose: Pin idempotence of the bootstrapped state (case 5).
        Inputs: fresh state.
        Outputs: two consecutive index() runs both succeed.
        """
        with tempfile.TemporaryDirectory() as temp:
            base, _global, _project, marker = self._fresh_state(temp)
            calls: list = []
            self._index_with_mocks(base, self._ok_run(calls))
            self._index_with_mocks(base, self._ok_run(calls))
            self.assertEqual(len(calls), 2)
            self.assertTrue(marker.exists())


if __name__ == "__main__":
    unittest.main()
