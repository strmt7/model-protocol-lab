"""Golden test: the committed sample report regenerates byte-identically."""

import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "examples" / "make_sample_report.py"
COMMITTED = REPO_ROOT / "examples" / "sample-report"


def load_builder():
    spec = importlib.util.spec_from_file_location("make_sample_report", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_sample_report


class SampleReportTests(unittest.TestCase):
    def test_sample_report_regenerates_byte_identically(self):
        build_sample_report = load_builder()
        with TemporaryDirectory() as tmp:
            build_sample_report(Path(tmp) / "regen")
            for name in ("results.json", "report.html"):
                committed = (COMMITTED / name).read_bytes()
                regenerated = (Path(tmp) / "regen" / name).read_bytes()
                self.assertEqual(regenerated, committed, name)

    def test_committed_sample_report_is_complete(self):
        import json

        results = json.loads((COMMITTED / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(results["decision"], "compatible_on_fixture_set")
        self.assertEqual(results["selected_candidate"], "llama3")
        self.assertGreater(len(results["search"]), 1)
        html = (COMMITTED / "report.html").read_text(encoding="utf-8")
        self.assertIn("compatible_on_fixture_set", html)
        self.assertNotIn("http://", html)
        self.assertNotIn("https://", html)


if __name__ == "__main__":
    unittest.main()
