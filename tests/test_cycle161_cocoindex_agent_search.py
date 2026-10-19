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
    ROOT,
    home,
    paths,
    prepare_mirror,
    search,
    source_digest,
    tracked_files,
)


class CocoIndexAgentSearchTests(unittest.TestCase):
    def test_successful_search_records_real_use_without_query_text(self):
        """Purpose: Prove routing receipt semantics.
        Inputs: fresh source/real wrapper/mock package.
        Outputs: deduplicated use record."""
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
                        "cocoindex_code": "0.2.41",
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
                        "cocoindex_code": "0.2.41",
                    }
                ),
                encoding="utf-8",
            )

            def mutate_source(*args, **kwargs):
                """Purpose: Inject source drift
                Inputs: package argv
                Outputs: successful stale search data."""
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
        with (
            patch.dict("os.environ", {"AGENT_CODE_HOME": "relative"}),
            self.assertRaisesRegex(ValueError, "absolute"),
        ):
            home()

    def test_home_rejects_checkout_state(self):
        with (
            patch.dict("os.environ", {"AGENT_CODE_HOME": str(ROOT / "index-state")}),
            self.assertRaisesRegex(ValueError, "outside the checkout"),
        ):
            home()

    def test_mirror_copies_exact_bytes_and_prunes_stale_generated_file(self):
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
        with (
            tempfile.TemporaryDirectory() as temp,
            self.assertRaisesRegex(RuntimeError, "no active index"),
        ):
            search(Path(temp), "training labels", 5, repo=Path(temp))

    def test_search_rejects_unreliable_wildcard_filter(self):
        with (
            tempfile.TemporaryDirectory() as temp,
            self.assertRaisesRegex(ValueError, "wildcard path filters"),
        ):
            search(Path(temp), "training labels", 5, repo=Path(temp), path="model_protocol_lab/*")

    def test_search_rejects_stale_source_before_package_call(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            source = base / "source.py"
            source.write_text("pass\n", encoding="utf-8")
            mirror = paths(base, base)[1]
            marker = base / "active" / (mirror.name + ".json")
            marker.parent.mkdir()
            marker.write_text(
                json.dumps({"source_digest": "incorrect", "cocoindex_code": "0.2.41"}),
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
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            self.assertNotEqual(paths(base, base / "a")[1], paths(base, base / "b")[1])

    def test_tracked_files_rejects_escape_and_real_env(self):
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


if __name__ == "__main__":
    unittest.main()
