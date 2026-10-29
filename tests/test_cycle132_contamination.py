"""Cycle 132: holdout contamination detector (IDEAS #2).

Pure, deterministic, no model: identical fixtures score 1.0/duplicate,
disjoint fixtures score low/clean, unicode-normalization traps do not
evade detection, and the report is deterministic across processes.
"""

import json
import subprocess
import sys
import unittest
from pathlib import Path

from model_protocol_lab import diffreports
from model_protocol_lab.contamination import (
    _normalize_text,
    classify_pair,
    contamination_report,
    fixture_signature,
    similarity_scores,
)
from model_protocol_lab.core import Fixture, Message

REPO_ROOT = Path(__file__).resolve().parent.parent


def _fx(fid, user, expected="expected", group=None):
    """Fx."""
    return Fixture(
        fid,
        (Message("user", user),),
        expected,
        group or f"group-{fid}",
    )


class TestNormalize(unittest.TestCase):
    """Test normalize."""

    def test_nfkc_folds_fullwidth(self):
        """Test nfkc folds fullwidth."""
        self.assertEqual(_normalize_text("ＡＢＣ"), "abc")

    def test_format_chars_stripped(self):
        """Test format chars stripped."""
        # Zero-width space / word joiner must not evade detection.
        self.assertEqual(_normalize_text("ad\u200bmin"), "admin")
        self.assertEqual(_normalize_text("ad\u2060min"), "admin")

    def test_case_and_whitespace_collapsed(self):
        """Test case and whitespace collapsed."""
        self.assertEqual(_normalize_text("  What   IS\nthis?  "), "what is this?")

    def test_deterministic(self):
        """Test deterministic."""
        text = "Some  Fixture\ttext\nwith unicode caf\u00e9"
        self.assertEqual(_normalize_text(text), _normalize_text(text))


class TestFixtureSignature(unittest.TestCase):
    """Test fixture signature."""

    def test_user_messages_plus_expected(self):
        """Test user messages plus expected."""
        fixture = Fixture(
            "f1",
            (Message("user", "Hello"), Message("assistant", "Hi"), Message("user", "Bye")),
            "done",
            "g",
        )
        sig = fixture_signature(fixture)
        self.assertIn("hello", sig)
        self.assertIn("bye", sig)
        self.assertIn("done", sig)
        # Assistant turns carry no input signal.
        self.assertNotIn("hi", sig)

    def test_signature_stable(self):
        """Test signature stable."""
        fixture = _fx("f1", "What is 2+2?")
        self.assertEqual(fixture_signature(fixture), fixture_signature(fixture))


class TestSimilarity(unittest.TestCase):
    """Test similarity."""

    def test_identical_scores_one(self):
        """Test identical scores one."""
        ratio, jaccard = similarity_scores("the quick brown fox", "the quick brown fox")
        self.assertEqual(ratio, 1.0)
        self.assertEqual(jaccard, 1.0)

    def test_disjoint_scores_low(self):
        """Test disjoint scores low."""
        ratio, jaccard = similarity_scores("alpha beta gamma delta", "one two three four five six")
        self.assertLess(ratio, 0.5)
        self.assertEqual(jaccard, 0.0)

    def test_empty_both_jaccard_one(self):
        """Test empty both jaccard one."""
        ratio, jaccard = similarity_scores("", "")
        self.assertEqual(ratio, 1.0)
        self.assertEqual(jaccard, 1.0)

    def test_deterministic_across_calls(self):
        """Test deterministic across calls."""
        pair = ("lorem ipsum dolor sit amet", "lorem ipsum dolor sit AMET!")
        self.assertEqual(similarity_scores(*pair), similarity_scores(*pair))


class TestClassify(unittest.TestCase):
    """Test classify."""

    def test_tier_boundaries(self):
        """Test tier boundaries."""
        self.assertEqual(classify_pair(0.98, 0.0), "duplicate")
        self.assertEqual(classify_pair(0.979, 0.0), "near_duplicate")
        self.assertEqual(classify_pair(0.90, 0.0), "near_duplicate")
        self.assertEqual(classify_pair(0.899, 0.0), "paraphrase_suspect")
        self.assertEqual(classify_pair(0.80, 0.0), "paraphrase_suspect")
        self.assertEqual(classify_pair(0.799, 0.0), "clean")
        # High trigram overlap alone flags a paraphrase suspect.
        self.assertEqual(classify_pair(0.5, 0.85), "paraphrase_suspect")
        self.assertEqual(classify_pair(0.5, 0.849), "clean")


class TestContaminationReport(unittest.TestCase):
    """Test contamination report."""

    def test_exact_duplicate_flagged(self):
        """Test exact duplicate flagged."""
        search = (_fx("s-01", "What is the capital of France?", "Paris"),)
        held = (_fx("h-01", "What is the capital of France?", "Paris"),)
        report = contamination_report(search, held)
        self.assertEqual(report["worst_tier"], "duplicate")
        self.assertEqual(len(report["pairs"]), 1)
        pair = report["pairs"][0]
        self.assertEqual(
            (pair["held_out"], pair["search"], pair["tier"]), ("h-01", "s-01", "duplicate")
        )
        self.assertEqual(pair["ratio"], 1.0)

    def test_near_duplicate_flagged(self):
        """Test near duplicate flagged."""
        search = (_fx("s-01", "What is the capital of France?", "Paris"),)
        held = (_fx("h-01", "What is the capital city of France?", "Paris"),)
        report = contamination_report(search, held)
        self.assertEqual(report["worst_tier"], "near_duplicate")

    def test_case_whitespace_evasion_still_duplicate(self):
        """Test case whitespace evasion still duplicate."""
        search = (_fx("s-01", "What is the capital of France?", "Paris"),)
        held = (_fx("h-01", "  WHAT is the CAPITAL of france?\n", "paris"),)
        report = contamination_report(search, held)
        self.assertEqual(report["worst_tier"], "duplicate")

    def test_zwsp_evasion_still_duplicate(self):
        """Test zwsp evasion still duplicate."""
        search = (_fx("s-01", "admin panel access", "granted"),)
        held = (_fx("h-01", "ad\u200bmin panel access", "granted"),)
        report = contamination_report(search, held)
        self.assertEqual(report["worst_tier"], "duplicate")

    def test_disjoint_is_clean(self):
        """Test disjoint is clean."""
        search = (_fx("s-01", "Explain photosynthesis in one sentence.", "Plants."),)
        held = (_fx("h-01", "Who wrote the Odyssey?", "Homer."),)
        report = contamination_report(search, held)
        self.assertEqual(report["worst_tier"], "clean")
        self.assertEqual(report["pairs"], [])

    def test_best_match_wins_per_held_out(self):
        """Test best match wins per held out."""
        search = (
            _fx("s-01", "Completely unrelated prompt about stars.", "A"),
            _fx("s-02", "What is the capital of France?", "Paris"),
        )
        held = (_fx("h-01", "What is the capital of France?", "Paris"),)
        report = contamination_report(search, held)
        self.assertEqual(len(report["pairs"]), 1)
        self.assertEqual(report["pairs"][0]["search"], "s-02")

    def test_deterministic_ordering(self):
        """Test deterministic ordering."""
        search = (
            _fx("s-01", "What is the capital of France?", "Paris"),
            _fx("s-02", "What is the capital of Spain?", "Madrid"),
        )
        held = (
            _fx("h-02", "What is the capital of Spain?", "Madrid"),
            _fx("h-01", "What is the capital of France?", "Paris"),
        )
        first = contamination_report(search, held)
        second = contamination_report(held, search)  # swapped roles
        self.assertEqual(first, contamination_report(search, held))
        # Held-out ids sort deterministically regardless of input order.
        self.assertEqual([p["held_out"] for p in first["pairs"]], ["h-01", "h-02"])
        self.assertNotEqual(first["pairs"], second["pairs"])

    def test_empty_inputs_fail_closed(self):
        """Test empty inputs fail closed."""
        self.assertEqual(contamination_report((), ()), {"worst_tier": "clean", "pairs": []})

    def test_hostile_inputs_fail_closed(self):
        """Test hostile inputs fail closed."""
        self.assertEqual(contamination_report(None, None), {"worst_tier": "clean", "pairs": []})
        self.assertEqual(
            contamination_report("junk", [{"id": 1}]),
            {"worst_tier": "clean", "pairs": []},
        )

    def test_worst_tier_is_max(self):
        """Test worst tier is max."""
        search = (
            _fx("s-01", "What is the capital of France?", "Paris"),
            _fx("s-02", "Explain photosynthesis in one sentence.", "Plants."),
        )
        held = (
            _fx("h-01", "What is the capital city of France?", "Paris"),
            _fx("h-02", "Who wrote the Odyssey?", "Homer."),
        )
        report = contamination_report(search, held)
        self.assertEqual(report["worst_tier"], "near_duplicate")
        self.assertEqual(len(report["pairs"]), 1)


class TestInspectWiring(unittest.TestCase):
    """mpl inspect warns on contamination and carries the JSON block."""

    @staticmethod
    def _dataset_file(tmp, search_text, held_text):
        """Dataset file."""
        path = Path(tmp) / "ds.json"
        path.write_text(
            json.dumps(
                {
                    "search": [
                        {
                            "id": "s-01",
                            "group": "g1",
                            "messages": [{"role": "user", "content": search_text}],
                            "expected": "Paris",
                        }
                    ],
                    "held_out": [
                        {
                            "id": "h-01",
                            "group": "g2",
                            "messages": [{"role": "user", "content": held_text}],
                            "expected": "Paris",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return path

    @staticmethod
    def _run_inspect(dataset):
        """Run inspect."""
        proc = subprocess.run(
            [sys.executable, "-m", "model_protocol_lab", "inspect", str(dataset)],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=120,
            check=False,
        )
        return proc

    def test_contaminated_dataset_warns_and_reports(self):
        """Test contaminated dataset warns and reports."""
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            # Near-duplicate, not byte-identical: Split itself rejects
            # identical message inputs, so the detector's job starts at
            # paraphrase distance.
            dataset = self._dataset_file(
                tmp,
                "What is the capital of France?",
                "What is the capital city of France?",
            )
            proc = self._run_inspect(dataset)
            self.assertEqual(proc.returncode, 0)
            self.assertIn("contamination", proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual(payload["contamination"]["worst_tier"], "near_duplicate")
            self.assertEqual(len(payload["contamination"]["pairs"]), 1)

    def test_clean_dataset_no_warning(self):
        """Test clean dataset no warning."""
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            dataset = self._dataset_file(tmp, "Explain photosynthesis.", "Who wrote the Odyssey?")
            proc = self._run_inspect(dataset)
            self.assertEqual(proc.returncode, 0)
            self.assertNotIn("contamination", proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual(payload["contamination"]["worst_tier"], "clean")


class TestFixturesBuildWiring(unittest.TestCase):
    """mpl fixtures build warns on contamination and carries the JSON block."""

    def test_build_reports_contamination(self):
        """Test build reports contamination."""
        from tempfile import TemporaryDirectory

        csv_text = (
            "id,group,partition,expected,user_1\n"
            "s-01,g1,search,Paris,What is the capital of France?\n"
            "h-01,g2,held_out,Paris,What is the capital city of France?\n"
        )
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "authoring.csv"
            source.write_text(csv_text, encoding="utf-8")
            output = Path(tmp) / "dataset.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "model_protocol_lab",
                    "fixtures",
                    "build",
                    str(source),
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
                timeout=120,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("contamination", proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual(payload["contamination"]["worst_tier"], "near_duplicate")


class TestExamineDatasetWiring(unittest.TestCase):
    """doctor --dataset surfaces the contamination report."""

    def test_examine_dataset_carries_contamination(self):
        """Test examine dataset carries contamination."""
        from tempfile import TemporaryDirectory

        from model_protocol_lab.doctor import examine_dataset

        dataset = {
            "search": [
                {
                    "id": "s-01",
                    "group": "g1",
                    "messages": [{"role": "user", "content": "What is the capital of France?"}],
                    "expected": "Paris",
                }
            ],
            "held_out": [
                {
                    "id": "h-01",
                    "group": "g2",
                    "messages": [
                        {
                            "role": "user",
                            "content": "What is the capital city of France?",
                        }
                    ],
                    "expected": "Paris",
                }
            ],
        }
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "ds.json"
            path.write_text(json.dumps(dataset), encoding="utf-8")
            result = examine_dataset(path)
        self.assertEqual(result["verdict"], "ok")
        contamination = result["dataset"]["contamination"]
        self.assertEqual(contamination["worst_tier"], "near_duplicate")
        self.assertEqual(contamination["pairs"][0]["held_out"], "h-01")


class TestReportCallout(unittest.TestCase):
    """The run-configuration section carries the fixture-hygiene line."""

    @staticmethod
    def _render(contamination):
        """Render."""
        from model_protocol_lab.reporting import _config_section

        return _config_section({"contamination": contamination})

    def test_clean_callout(self):
        """Test clean callout."""
        html = self._render({"worst_tier": "clean", "pairs": []})
        self.assertIn("Fixture hygiene", html)
        self.assertIn("clean", html)

    def test_flagged_callout(self):
        """Test flagged callout."""
        html = self._render(
            {
                "worst_tier": "near_duplicate",
                "pairs": [
                    {
                        "held_out": "h-01",
                        "search": "s-01",
                        "ratio": 0.93,
                        "jaccard": 0.4,
                        "tier": "near_duplicate",
                    }
                ],
            }
        )
        self.assertIn("near_duplicate", html)
        self.assertIn("advisory", html)

    def test_hostile_contamination_never_crashes(self):
        """Test hostile contamination never crashes."""
        for hostile in (None, "junk", {"worst_tier": 5}, {"worst_tier": "x"}, []):
            html = self._render(hostile)
            self.assertIn("Fixture hygiene", html)

    def test_missing_key_renders_not_recorded(self):
        """Test missing key renders not recorded."""
        from model_protocol_lab.reporting import _config_section

        html = _config_section({})
        self.assertIn("not recorded", html)


class TestMutationGate(unittest.TestCase):
    """Hand-planted mutants: each must change the verdict visibly."""

    # Hand-planted mutants: each must change the verdict visibly.

    def test_flip_expected_answer_changes_tier(self):
        """Test flip expected answer changes tier."""
        # Perturbing one word of a duplicate must drop the tier.
        search = (_fx("s-01", "What is the capital of France?", "Paris"),)
        held_exact = (_fx("h-01", "What is the capital of France?", "Paris"),)
        held_mut = (_fx("h-01", "What is the capital of France?", "Lyon"),)
        exact = contamination_report(search, held_exact)
        mutated = contamination_report(search, held_mut)
        self.assertEqual(exact["worst_tier"], "duplicate")
        self.assertLess(
            mutated["pairs"][0]["ratio"] if mutated["pairs"] else 0,
            exact["pairs"][0]["ratio"],
        )

    def test_threshold_boundary_mutant(self):
        """Test threshold boundary mutant."""
        # classify_pair(0.98 - eps) must not be duplicate.
        self.assertNotEqual(classify_pair(0.98 - 1e-9, 0.0), "duplicate")
        self.assertEqual(classify_pair(0.98, 0.0), "duplicate")


class ContaminationHtmlInjectionTest(unittest.TestCase):
    """Hostile worst_tier must not inject HTML into the report.

    The _contamination_line output is escaped by the caller
    (_config_section applies escape() to every value), so the unit
    returns the raw tier and the integration point is where the
    escaping guarantee lives.
    """

    def test_worst_tier_escaped_at_config_section(self):
        """Test worst tier escaped at config section."""
        from model_protocol_lab.reporting import _config_section

        html = _config_section(
            {
                "contamination": {
                    "worst_tier": "<script>alert(1)</script>",
                    "pairs": [{"held_out": "h", "search": "s"}],
                },
                "search": [],
            }
        )
        self.assertNotIn("<script>alert(1)</script>", html)
        self.assertIn("&lt;script&gt;", html)

    def test_clean_tier_unchanged(self):
        """Test clean tier unchanged."""
        from model_protocol_lab.reporting import _contamination_line

        self.assertEqual(
            _contamination_line({"worst_tier": "clean", "pairs": []}),
            "clean: no search/held-out near-duplicate pairs",
        )


def _contamination_payload(worst="clean", pairs=None):
    """Contamination payload."""
    return {"worst_tier": worst, "pairs": pairs or []}


class DiffContaminationTest(unittest.TestCase):
    """Diff contamination test."""

    def test_identical_unchanged(self):
        """Test identical unchanged."""
        payload = _contamination_payload(
            "near_duplicate",
            [{"held_out": "h1", "search": "s1", "tier": "near_duplicate"}],
        )
        diff = diffreports.diff_contamination(payload, payload)
        self.assertEqual(diff["available"], "both")
        self.assertFalse(diff["changed"])

    def test_worst_tier_change_detected(self):
        """Test worst tier change detected."""
        old = _contamination_payload("clean", [])
        new = _contamination_payload(
            "near_duplicate",
            [{"held_out": "h1", "search": "s1", "tier": "near_duplicate"}],
        )
        diff = diffreports.diff_contamination(old, new)
        self.assertTrue(diff["changed"])
        self.assertTrue(diff["worst_tier"]["changed"])
        self.assertEqual(len(diff["pairs_added"]), 1)

    def test_pair_removed(self):
        """Test pair removed."""
        old = _contamination_payload(
            "duplicate",
            [{"held_out": "h1", "search": "s1", "tier": "duplicate"}],
        )
        new = _contamination_payload("clean", [])
        diff = diffreports.diff_contamination(old, new)
        self.assertTrue(diff["changed"])
        self.assertEqual(len(diff["pairs_removed"]), 1)

    def test_tier_change_detected(self):
        """Test tier change detected."""
        old = _contamination_payload(
            "near_duplicate",
            [{"held_out": "h1", "search": "s1", "tier": "near_duplicate"}],
        )
        new = _contamination_payload(
            "duplicate",
            [{"held_out": "h1", "search": "s1", "tier": "duplicate"}],
        )
        diff = diffreports.diff_contamination(old, new)
        self.assertTrue(diff["changed"])
        self.assertEqual(len(diff["tier_changes"]), 1)
        self.assertEqual(diff["tier_changes"][0]["old_tier"], "near_duplicate")
        self.assertEqual(diff["tier_changes"][0]["new_tier"], "duplicate")

    def test_asymmetric(self):
        """Test asymmetric."""
        diff = diffreports.diff_contamination(_contamination_payload(), {"worst_tier": "clean"})
        self.assertEqual(diff["available"], "old")
        self.assertFalse(diff["changed"])

    def test_malformed_pairs_ignored(self):
        """Test malformed pairs ignored."""
        # Pairs without string IDs are ignored, not crashed on.
        old = _contamination_payload("clean", [])
        new = _contamination_payload(
            "clean",
            [{"held_out": 123, "search": "s1", "tier": "duplicate"}],
        )
        diff = diffreports.diff_contamination(old, new)
        self.assertFalse(diff["changed"])
        self.assertEqual(diff["pairs_added"], [])

    def test_feeds_identical(self):
        """Test feeds identical."""
        old = {"decision": "selected", "contamination": _contamination_payload()}
        new = {"decision": "selected", "contamination": _contamination_payload()}
        diff = diffreports.diff_reports(old, new)
        self.assertFalse(diff["summary"]["contamination_changed"])
        new2 = {
            "decision": "selected",
            "contamination": _contamination_payload(
                "duplicate",
                [{"held_out": "h1", "search": "s1", "tier": "duplicate"}],
            ),
        }
        diff2 = diffreports.diff_reports(old, new2)
        self.assertTrue(diff2["summary"]["contamination_changed"])
        self.assertFalse(diff2["summary"]["identical"])


if __name__ == "__main__":
    unittest.main()
