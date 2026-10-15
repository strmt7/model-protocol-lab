"""End-to-end audit of the sample report pipeline (Cycle 47, 2026-10-15).

Audits ``examples/make_sample_report.py`` + the committed artifacts in
``examples/sample-report/`` (results.json + report.html) as the repo's showcase
artifact. The pipeline uses the script's own scripted ``DemoBackend`` (no real
model, no network) -- the same code paths as the committed regeneration.

## Bugs found (2026-10-15)

NONE. Regeneration succeeds end-to-end via the scripted pipeline, the HTML is
fully self-contained, output is byte-deterministic across runs, and the
regenerated output is byte-identical to the committed example.

## Schema drift: COMMITTED -> REGENERATED (2026-10-15)

NONE. ``diff -r`` of a fresh regeneration against
``examples/sample-report/`` is empty: same top-level key set, same key order,
same ``schema_version`` (1), same decision/candidate/selection_rule values,
and byte-identical HTML.

## Deliberately NOT verified

- Visual rendering in a real browser (no live browser available in this
  environment); HTML self-containment is verified statically instead.
- The CSP ``script-src`` hash vs. the inline script body is not re-derived
  here; the committed HTML already carries an inline-only ``<script>`` with a
  ``default-src 'none'`` policy, and no external refs exist to execute.
- ``pip-audit`` / CI state: out of scope for this audit.

These tests only ever call ``build_sample_report()`` with a temp output dir;
they never invoke the script's ``__main__`` block and never write into
``examples/sample-report/``.
"""

import importlib.util
import json
import re
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "examples" / "make_sample_report.py"
COMMITTED = REPO_ROOT / "examples" / "sample-report"

# Pinned full top-level report schema (as committed in
# examples/sample-report/results.json). Asserted by exact key-set AND order.
EXPECTED_TOP_LEVEL_KEYS = [
    "schema_version",
    "selection_rule",
    "measurement",
    "seed",
    "max_new_tokens",
    "boundary_token_ids",
    "eos_token_id",
    "backend_fingerprint",
    "statistical_confidence",
    "compute_comparison",
    "decision",
    "search",
    "held_out",
    "selected_candidate",
    "selected_boundary_token_ids",
    "failure_taxonomy",
    "selection_margin",
    "token_overhead",
    "raw_evidence_included",
]

EXPECTED_SEARCH_ENTRY_KEYS = {"boundary_token_ids", "candidate_fingerprint", "format", "summary"}
EXPECTED_FORMAT_KEYS = {
    "assistant_prefix",
    "assistant_turn_end",
    "bos",
    "name",
    "open_suffix",
    "repeat_bos",
    "turn_end",
    "user_prefix",
    "user_turn_end",
}
EXPECTED_SUMMARY_KEYS = {
    "aggregate_eligible",
    "attempted",
    "encoded_input_tokens",
    "exact_and_boundary_passes",
    "failure_code",
    "planned",
    "rate",
    "retained_output_tokens",
    "status",
}
EXPECTED_SELECTION_MARGIN_KEYS = {"decisive_fixtures", "direction", "runner_up", "value"}
EXPECTED_TOKEN_OVERHEAD_KEYS = {"cheapest_candidate", "per_candidate", "winner_tax_vs_cheapest_pct"}
EXPECTED_PER_CANDIDATE_OVERHEAD_KEYS = {
    "max_prompt_tokens",
    "mean_output_tokens",
    "mean_prompt_tokens",
}
EXPECTED_FAILURE_TAXONOMY_KEYS = {"eos_id_considered", "held_out", "modes", "per_candidate"}
EXPECTED_CANDIDATES = (
    "llama3",
    "llama3/toggle-bos",
    "llama3/no-bos",
    "llama3/prefix-newline",
    "llama3/prefix-trim",
)

# Matches real external references; the literal "http-equiv" attribute is not one.
_EXTERNAL_REF = re.compile(
    r"""(?:src|href|action|formaction)\s*=\s*["']https?://|
        url\(\s*["']?https?://|
        @import\s+(?:url\(\s*)?["']?https?://|
        from\s+["']https?://|
        import\(\s*["']https?://""",
    re.IGNORECASE | re.VERBOSE,
)


def load_builder():
    spec = importlib.util.spec_from_file_location("make_sample_report", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_sample_report


class SampleReportAuditTests(unittest.TestCase):
    def test_regeneration_succeeds_end_to_end(self):
        build_sample_report = load_builder()
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "regen"
            result = build_sample_report(out)
            for name in ("results.json", "report.html"):
                path = out / name
                self.assertTrue(path.is_file(), name)
                self.assertGreater(path.stat().st_size, 0, name)
            self.assertEqual(result["decision"], "compatible_on_fixture_set")
            self.assertEqual(result["selected_candidate"], "llama3")
            # JSON on disk parses and matches the returned result document.
            on_disk = json.loads((out / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(on_disk["decision"], result["decision"])
            self.assertEqual(on_disk["schema_version"], 1)

    def test_html_is_self_contained(self):
        build_sample_report = load_builder()
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "regen"
            build_sample_report(out)
            html = (out / "report.html").read_text(encoding="utf-8")
            # No external URL references of any kind.
            hits = _EXTERNAL_REF.findall(html)
            self.assertEqual(hits, [], f"external refs: {hits[:3]}")
            self.assertNotIn("http://", html.replace("http-equiv", ""))
            self.assertNotIn("https://", html)
            # No remote-loading elements at all.
            for tag in ("link", "img", "iframe", "video", "audio", "source", "embed", "object"):
                self.assertNotRegex(html, rf"<{tag}\b", f"<{tag}> element present")
            # Exactly one inline script; styles are inline; CSP blocks remote loads.
            self.assertEqual(len(re.findall(r"<script\b", html)), 1)
            self.assertIn("Content-Security-Policy", html)
            self.assertIn("default-src 'none'", html)
            # Report content actually rendered.
            self.assertIn("compatible_on_fixture_set", html)
            self.assertIn("llama3", html)

    def test_json_report_schema_complete(self):
        build_sample_report = load_builder()
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "regen"
            build_sample_report(out)
            results = json.loads((out / "results.json").read_text(encoding="utf-8"))

            # Exact top-level schema: key set AND insertion order pinned.
            self.assertEqual(list(results.keys()), EXPECTED_TOP_LEVEL_KEYS)

            # Config echoes pinned by other tests (cycle45 audit pins these too).
            self.assertEqual(results["schema_version"], 1)
            self.assertEqual(results["selection_rule"], "unique_all_search_pass")
            self.assertEqual(results["measurement"], "exact_answer_and_boundary")
            self.assertEqual(results["seed"], 20261012)
            self.assertEqual(results["max_new_tokens"], 16)
            self.assertEqual(results["boundary_token_ids"], [124])
            self.assertIsNone(results["eos_token_id"])
            self.assertEqual(
                results["backend_fingerprint"], "synthetic-demo-backend-v1-not-a-model"
            )
            self.assertEqual(results["raw_evidence_included"], False)

            # Decision block.
            self.assertEqual(results["decision"], "compatible_on_fixture_set")
            self.assertEqual(results["selected_candidate"], "llama3")
            self.assertEqual(results["selected_boundary_token_ids"], [124])
            self.assertEqual(results["statistical_confidence"], "not_estimated")
            self.assertEqual(results["compute_comparison"], "not_equal_compute_certified")

            # Search section: 5 candidates, full nested schema.
            search = results["search"]
            self.assertEqual(len(search), 5)
            for entry in search:
                with self.subTest(candidate=entry.get("format", {}).get("name")):
                    self.assertEqual(set(entry.keys()), EXPECTED_SEARCH_ENTRY_KEYS)
                    self.assertEqual(set(entry["format"].keys()), EXPECTED_FORMAT_KEYS)
                    self.assertEqual(set(entry["summary"].keys()), EXPECTED_SUMMARY_KEYS)
            self.assertEqual([e["format"]["name"] for e in search], list(EXPECTED_CANDIDATES))

            # Held-out section.
            self.assertEqual(set(results["held_out"].keys()), {"summary"})
            self.assertEqual(set(results["held_out"]["summary"].keys()), EXPECTED_SUMMARY_KEYS)
            self.assertEqual(results["held_out"]["summary"]["status"], "completed")

            # Selection margin, token overhead, failure taxonomy.
            self.assertEqual(
                set(results["selection_margin"].keys()), EXPECTED_SELECTION_MARGIN_KEYS
            )
            self.assertEqual(set(results["token_overhead"].keys()), EXPECTED_TOKEN_OVERHEAD_KEYS)
            per_candidate = results["token_overhead"]["per_candidate"]
            self.assertEqual(set(per_candidate.keys()), set(EXPECTED_CANDIDATES))
            for name, stats in per_candidate.items():
                with self.subTest(candidate=name):
                    self.assertEqual(set(stats.keys()), EXPECTED_PER_CANDIDATE_OVERHEAD_KEYS)
            self.assertEqual(
                set(results["failure_taxonomy"].keys()), EXPECTED_FAILURE_TAXONOMY_KEYS
            )
            self.assertEqual(
                set(results["failure_taxonomy"]["per_candidate"].keys()),
                set(EXPECTED_CANDIDATES),
            )

    def test_regeneration_is_deterministic(self):
        build_sample_report = load_builder()
        with TemporaryDirectory() as tmp:
            first = Path(tmp) / "first"
            second = Path(tmp) / "second"
            build_sample_report(first)
            build_sample_report(second)
            for name in ("results.json", "report.html"):
                a = (first / name).read_bytes()
                b = (second / name).read_bytes()
                self.assertEqual(a, b, f"{name} differs between two runs")

    def test_regenerated_matches_committed_schema_and_version(self):
        build_sample_report = load_builder()
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "regen"
            build_sample_report(out)
            committed = json.loads((COMMITTED / "results.json").read_text(encoding="utf-8"))
            regenerated = json.loads((out / "results.json").read_text(encoding="utf-8"))
            # Same schema/version keys as the committed example.
            self.assertEqual(list(regenerated.keys()), list(committed.keys()))
            self.assertEqual(regenerated["schema_version"], committed["schema_version"])
            for key in (
                "decision",
                "selected_candidate",
                "selection_rule",
                "measurement",
                "seed",
                "backend_fingerprint",
            ):
                self.assertEqual(regenerated[key], committed[key], key)
            # No structural drift: byte-identical to the committed showcase.
            for name in ("results.json", "report.html"):
                self.assertEqual(
                    (out / name).read_bytes(),
                    (COMMITTED / name).read_bytes(),
                    f"{name}: regenerated != committed",
                )


if __name__ == "__main__":
    unittest.main()
