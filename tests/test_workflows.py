"""Schema, selection, privacy and CLI contracts using scripted fixtures."""

import argparse
import io
import json
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event

from model_protocol_lab.catalog import CATALOG, candidates
from model_protocol_lab.cli import main, parse_boundary
from model_protocol_lab.core import Generation, Split
from model_protocol_lab.datasets import MAX_BYTES, load_dataset, parse_dataset
from model_protocol_lab.experiments import run_experiment
from model_protocol_lab.reporting import _SCRIPT, render_html, save_report
from tests.test_core import BASE, GOOD, RAW, FakeBackend, case, ids


def dataset_object():
    def item(name):
        return {
            "id": name,
            "group": name,
            "messages": [{"role": "user", "content": "question:" + name}],
            "expected": "ok",
        }

    return {"search": [item("s1"), item("s2")], "held_out": [item("h1"), item("h2")]}


def split():
    return Split((case("s1"), case("s2")), (case("h1"), case("h2")))


def experiment(backend=None, formats=(BASE,), cancel=None):
    return run_experiment(
        backend or FakeBackend(),
        formats,
        split(),
        boundary=ids("|"),
        max_new_tokens=len(RAW),
        cancel=cancel,
    )


class DatasetTests(unittest.TestCase):
    def test_valid_json_roundtrip(self):
        parsed = parse_dataset(json.dumps(dataset_object()))
        self.assertEqual([c.id for c in parsed.search], ["s1", "s2"])
        self.assertEqual([c.id for c in parsed.held_out], ["h1", "h2"])

    def test_extra_fields_and_duplicate_json_keys_rejected(self):
        value = dataset_object()
        value["unexpected"] = True
        with self.assertRaises(ValueError):
            parse_dataset(json.dumps(value))
        with self.assertRaises(ValueError):
            parse_dataset('{"search":[],"search":[],"held_out":[]}')

    def test_nonfinite_and_oversized_json_rejected(self):
        with self.assertRaises(ValueError):
            parse_dataset('{"search":NaN,"held_out":[]}')
        with self.assertRaises(ValueError):
            parse_dataset(" " * (MAX_BYTES + 1))

    def test_bad_conversation_and_unknown_message_keys_rejected(self):
        for edit in ("role", "final", "extra"):
            with self.subTest(edit=edit):
                value = dataset_object()
                messages = value["search"][0]["messages"]
                if edit == "role":
                    messages[0]["role"] = "system"
                elif edit == "final":
                    messages.append({"role": "assistant", "content": "a"})
                else:
                    messages[0]["tool_calls"] = []
                with self.assertRaises(ValueError):
                    parse_dataset(json.dumps(value))

    def test_file_loader_uses_bounded_utf8_json(self):
        with TemporaryDirectory() as tmp:
            file = Path(tmp) / "fixtures.json"
            file.write_text(json.dumps(dataset_object()), encoding="utf-8")
            self.assertEqual(len(load_dataset(file).search), 2)
            file.write_bytes(b"\xff")
            with self.assertRaisesRegex(ValueError, "not valid UTF-8"):
                load_dataset(file)

    def test_split_leakage_survives_json_loading(self):
        value = dataset_object()
        value["held_out"][0]["messages"] = value["search"][0]["messages"]
        with self.assertRaises(ValueError):
            parse_dataset(json.dumps(value))


class WorkflowTests(unittest.TestCase):
    def test_one_unique_search_candidate_receives_heldout_tests(self):
        backend = FakeBackend()
        result = experiment(backend)
        self.assertEqual(result["decision"], "compatible_on_fixture_set")
        self.assertEqual(result["selected_candidate"], BASE.name)
        self.assertEqual(len(backend.generated), 4)
        self.assertEqual(result["held_out"]["summary"]["planned"], 2)
        self.assertEqual(result["statistical_confidence"], "not_estimated")

    def test_ties_abstain_without_touching_heldout(self):
        backend = FakeBackend()
        result = experiment(backend, (BASE, replace(BASE, name="other")))
        self.assertEqual(result["decision"], "ambiguous")
        self.assertIsNone(result["held_out"])
        self.assertIsNone(result["selected_candidate"])
        self.assertEqual(len(backend.generated), 4)
        self.assertFalse(any("question:h" in prompt for prompt in backend.encoded))

    def test_no_search_success_abstains(self):
        backend = FakeBackend(Generation(ids("no|tail"), "limit"))
        result = experiment(backend)
        self.assertEqual(result["decision"], "no_candidate")
        self.assertIsNone(result["held_out"])
        self.assertEqual(len(backend.generated), 2)

    def test_heldout_failure_is_not_reported_as_compatibility(self):
        backend = FakeBackend(
            generate_script=[GOOD, GOOD, Generation(ids("no|tail"), "limit"), GOOD]
        )
        result = experiment(backend)
        self.assertEqual(result["decision"], "held_out_failed")
        self.assertEqual(result["held_out"]["summary"]["exact_and_boundary_passes"], 1)

    def test_cancelled_and_failed_search_never_selects_candidate(self):
        for reason in ("cancelled", "error"):
            with self.subTest(reason=reason):
                result = experiment(FakeBackend(Generation((), reason)))
                self.assertEqual(
                    result["decision"], "cancelled" if reason == "cancelled" else "failed"
                )
                self.assertIsNone(result["selected_candidate"])
                self.assertIsNone(result["held_out"])

    def test_duplicate_candidate_names_rejected(self):
        with self.assertRaises(ValueError):
            experiment(formats=(BASE, BASE))

    def test_candidate_limits_are_bounded(self):
        for formats in ((), [BASE], tuple(replace(BASE, name=str(i)) for i in range(25))):
            with self.subTest(count=len(formats)), self.assertRaises(ValueError):
                experiment(formats=formats)

    def test_pre_cancelled_experiment_preserves_incomplete_status(self):
        event = Event()
        event.set()
        backend = FakeBackend()
        result = experiment(backend, cancel=event)
        self.assertEqual(result["decision"], "cancelled")
        self.assertEqual(backend.generated, [])


class ReportTests(unittest.TestCase):
    def test_reports_exclude_raw_evidence_by_default(self):
        result = experiment()
        with TemporaryDirectory() as tmp:
            json_path, html_path = save_report(result, Path(tmp) / "new-report")
            saved = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertFalse(saved["raw_evidence_included"])
            self.assertNotIn("observations", saved["search"][0])
            self.assertNotIn("observations", saved["held_out"])
            self.assertNotIn("question:s1", html_path.read_text(encoding="utf-8"))
        self.assertIn("observations", result["search"][0])

    def test_evidence_export_requires_explicit_option(self):
        with TemporaryDirectory() as tmp:
            path, _ = save_report(experiment(), Path(tmp) / "evidence", include_evidence=True)
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertTrue(saved["raw_evidence_included"])
            self.assertIn("observations", saved["search"][0])

    def test_existing_directory_is_not_overwritten(self):
        with TemporaryDirectory() as tmp, self.assertRaises(FileExistsError):
            save_report(experiment(), tmp)

    def test_html_escapes_candidate_names_and_decisions(self):
        result = experiment()
        result["search"][0]["format"]["name"] = '<script>alert("x")</script>'
        result["decision"] = "<img src=x onerror=alert(1)>"
        result["search"][0]["summary"]["attempted"] = "<img src=x>"
        result["search"][0]["summary"]["planned"] = "<script>x</script>"
        html = render_html(result)
        # The only <script> permitted is the allowlisted inline table sorter.
        scrubbed = html.replace("<script>" + _SCRIPT + "</script>", "")
        self.assertNotIn("<script>", scrubbed)
        self.assertNotIn("<img", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("default-src 'none'", html)

    def test_serialization_failure_does_not_create_directory(self):
        result = experiment()
        result["unserializable"] = object()
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "new-report"
            with self.assertRaises(TypeError):
                save_report(result, target)
            self.assertFalse(target.exists())


class CLITests(unittest.TestCase):
    def test_catalog_does_not_require_model_dependencies(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["catalog"]), 0)
        self.assertEqual(len(json.loads(output.getvalue())), len(CATALOG))

    def test_inspect_reports_only_fixture_counts(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.json"
            path.write_text(json.dumps(dataset_object()), encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["inspect", str(path)]), 0)
            # The synthetic "question:s1"/"question:h1" fixtures differ by a
            # single character, so the contamination check (correctly) flags
            # them as near-duplicates; the values are deterministic.
            self.assertEqual(
                json.loads(output.getvalue()),
                {
                    "search_cases": 2,
                    "held_out_cases": 2,
                    "valid": True,
                    "contamination": {
                        "worst_tier": "near_duplicate",
                        "pairs": [
                            {
                                "held_out": "h1",
                                "search": "s1",
                                "ratio": 0.9286,
                                "jaccard": 0.0,
                                "tier": "near_duplicate",
                            },
                            {
                                "held_out": "h2",
                                "search": "s2",
                                "ratio": 0.9286,
                                "jaccard": 0.0,
                                "tier": "near_duplicate",
                            },
                        ],
                    },
                },
            )

    def test_boundary_parser(self):
        self.assertEqual(parse_boundary("1, 2,3"), (1, 2, 3))
        for value in ("", "-1", "1,", "a", "1.5", "1," * 16 + "1", str(2**31)):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                parse_boundary(value)

    def test_catalog_lookup_is_explicit(self):
        for name in CATALOG:
            self.assertEqual(candidates(name)[0], CATALOG[name])
            self.assertGreaterEqual(len(candidates(name, mutate=True)), 1)
        with self.assertRaises(ValueError):
            candidates("unknown")
