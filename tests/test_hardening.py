"""Hardening tests: edge cases, missed branches, determinism, and properties.

These tests guard the fixes and invariants added during the hardening pass.
They use scripted backends only; the optional HF adapter has its own
integration module (tests/test_adapter_integration.py).
"""

import argparse
import importlib.util
import io
import json
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from unittest.mock import patch

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

# Note on suppress_health_check=[HealthCheck.differing_executors] below: the
# check fires when a property test is re-run in the same process (programmatic
# double-invocation, IDE re-runs). Our tests are deterministic; it is a harness
# artifact here, not a flakiness signal. Suppressed with justification.
from model_protocol_lab.catalog import CATALOG, candidates
from model_protocol_lab.cli import main, parse_boundary
from model_protocol_lab.core import (
    Format,
    Generation,
    Message,
    Split,
    evaluate,
    first_boundary,
    local_repairs,
)
from model_protocol_lab.datasets import (
    MAX_BYTES,
    MAX_CASES_PER_SPLIT,
    MAX_MESSAGES,
    MAX_TEXT,
    load_dataset,
    parse_dataset,
)
from model_protocol_lab.experiments import run_experiment
from model_protocol_lab.reporting import _SCRIPT, render_html, save_report
from tests.test_core import BASE, GOOD, RAW, FakeBackend, case, ids
from tests.test_workflows import dataset_object, experiment, split

HAS_HF = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "transformers", "safetensors", "tokenizers")
)

SECRET = "synthetic-secret-must-never-leak"


def fixture_json(search_count=2, held_count=2):
    """Fixture json."""

    def item(prefix, index):
        """Item."""
        return {
            "id": f"{prefix}-{index}",
            "group": f"{prefix}-group-{index}",
            "messages": [{"role": "user", "content": f"q-{prefix}-{index}"}],
            "expected": "ok",
        }

    return {
        "search": [item("s", i) for i in range(search_count)],
        "held_out": [item("h", i) for i in range(held_count)],
    }


class DatasetEdgeTests(unittest.TestCase):
    """Dataset edge tests."""

    def test_blank_identifiers_rejected(self):
        """Test blank identifiers rejected."""
        for field in ("id", "group"):
            with self.subTest(field=field):
                value = dataset_object()
                value["search"][0][field] = "   "
                with self.assertRaises(ValueError):
                    parse_dataset(json.dumps(value))

    def test_oversized_text_field_rejected(self):
        """Test oversized text field rejected."""
        value = dataset_object()
        value["search"][0]["messages"][0]["content"] = "x" * (MAX_TEXT + 1)
        with self.assertRaises(ValueError):
            parse_dataset(json.dumps(value))
        value["search"][0]["messages"][0]["content"] = "x" * MAX_TEXT
        self.assertEqual(len(parse_dataset(json.dumps(value)).search), 2)

    def test_message_count_boundaries(self):
        """Test message count boundaries."""

        def messages(count):
            """Messages."""
            return [
                {"role": "user" if i % 2 == 0 else "assistant", "content": "m"}
                for i in range(count)
            ]

        value = dataset_object()
        value["search"][0]["messages"] = messages(MAX_MESSAGES)
        self.assertEqual(len(parse_dataset(json.dumps(value)).search[0].messages), MAX_MESSAGES)
        value["search"][0]["messages"] = messages(MAX_MESSAGES + 1)
        with self.assertRaises(ValueError):
            parse_dataset(json.dumps(value))

    def test_fixture_count_boundaries(self):
        """Test fixture count boundaries."""
        parsed = parse_dataset(json.dumps(fixture_json(MAX_CASES_PER_SPLIT, 1)))
        self.assertEqual(len(parsed.search), MAX_CASES_PER_SPLIT)
        with self.assertRaises(ValueError):
            parse_dataset(json.dumps(fixture_json(MAX_CASES_PER_SPLIT + 1, 1)))
        with self.assertRaises(ValueError):
            parse_dataset(json.dumps({"search": [], "held_out": fixture_json(1, 1)["held_out"]}))

    def test_non_list_partitions_rejected(self):
        """Test non list partitions rejected."""
        for partitions in (
            {"search": {}, "held_out": []},
            {"search": "x", "held_out": []},
            {"search": None, "held_out": []},
        ):
            with self.subTest(partitions=partitions), self.assertRaises(ValueError):
                parse_dataset(json.dumps(partitions))

    def test_duplicate_ids_across_partitions_rejected(self):
        """Test duplicate ids across partitions rejected."""
        value = dataset_object()
        value["held_out"][0]["id"] = value["search"][0]["id"]
        with self.assertRaises(ValueError):
            parse_dataset(json.dumps(value))

    def test_group_leakage_across_partitions_rejected(self):
        """Test group leakage across partitions rejected."""
        value = dataset_object()
        value["held_out"][0]["group"] = value["search"][1]["group"]
        with self.assertRaises(ValueError):
            parse_dataset(json.dumps(value))

    def test_load_dataset_rejects_oversized_file(self):
        """Test load dataset rejects oversized file."""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "big.json"
            path.write_bytes(b'{"search": [], "held_out": []}'.ljust(MAX_BYTES + 1, b" "))
            with self.assertRaises(ValueError):
                load_dataset(path)

    def test_expected_may_be_empty_string(self):
        """Test expected may be empty string."""
        value = dataset_object()
        value["search"][0]["expected"] = ""
        parsed = parse_dataset(json.dumps(value))
        self.assertEqual(parsed.search[0].expected, "")

    def test_fixture_with_wrong_item_type_rejected(self):
        """Test fixture with wrong item type rejected."""
        value = dataset_object()
        value["search"][0] = "not-a-fixture"
        with self.assertRaises(ValueError):
            parse_dataset(json.dumps(value))


class BoundaryEdgeTests(unittest.TestCase):
    """Boundary edge tests."""

    def test_overlapping_and_repeated_matches(self):
        """Test overlapping and repeated matches."""
        cases = [
            ((1, 1, 1), (1, 1), 0),
            ((1, 1, 2), (1, 2), 1),
            ((7, 7, 7, 7), (7, 7, 7), 0),
            ((5, 1, 2, 1, 2), (1, 2), 1),
            ((9,), (9,), 0),
            ((), (9,), None),
            ((1, 2), (1, 2, 3, 4), None),
            ((3, 3), (3, 3, 3), None),
        ]
        for tokens, stop, expected in cases:
            with self.subTest(tokens=tokens, stop=stop):
                self.assertEqual(first_boundary(tokens, stop), expected)

    def test_boundary_at_output_start_truncates_everything(self):
        """Test boundary at output start truncates everything."""
        backend = FakeBackend(Generation(ids("|tail"), "limit"))
        obs = evaluate(
            backend, BASE, (case(expected=""),), boundary=ids("|"), max_new_tokens=5
        ).observations[0]
        self.assertTrue(obs.boundary_found)
        self.assertEqual(obs.answer, "")
        self.assertTrue(obs.exact_match)

    def test_multi_token_boundary(self):
        """Test multi token boundary."""
        stop = (11, 22, 33)
        raw = ids("ok") + stop + ids("junk")
        backend = FakeBackend(Generation(raw, "limit"))
        obs = evaluate(
            backend, BASE, (case(),), boundary=stop, max_new_tokens=len(raw)
        ).observations[0]
        self.assertTrue(obs.exact_match and obs.boundary_found)

    @given(
        tokens=st.lists(st.integers(min_value=0, max_value=5), max_size=12),
        stop=st.lists(st.integers(min_value=0, max_value=5), min_size=1, max_size=4),
    )
    @settings(max_examples=300, suppress_health_check=[HealthCheck.differing_executors])
    def test_first_boundary_matches_naive_reference(self, tokens, stop):
        """Test first boundary matches naive reference."""
        tokens_t, stop_t = tuple(tokens), tuple(stop)
        expected = None
        for index in range(len(tokens_t) - len(stop_t) + 1):
            if list(tokens_t[index : index + len(stop_t)]) == list(stop_t):
                expected = index
                break
        self.assertEqual(first_boundary(tokens_t, stop_t), expected)


class RenderPropertyTests(unittest.TestCase):
    """Render property tests."""

    @given(
        bos=st.text(max_size=8),
        user_prefix=st.text(max_size=8),
        assistant_prefix=st.text(max_size=8),
        turn_end=st.text(max_size=8),
        contents=st.lists(st.text(max_size=20), min_size=1, max_size=5),
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.differing_executors])
    def test_render_preserves_content_verbatim(
        self, bos, user_prefix, assistant_prefix, turn_end, contents
    ):
        """Test render preserves content verbatim."""
        fmt = Format("prop", bos, user_prefix, assistant_prefix, turn_end)
        messages = tuple(
            Message("user" if i % 2 == 0 else "assistant", text) for i, text in enumerate(contents)
        )
        if messages[-1].role != "user":
            messages = messages[:-1]
            if not messages:
                return
        rendered = fmt.render(messages)
        self.assertTrue(rendered.startswith(bos))
        self.assertTrue(rendered.endswith(assistant_prefix))
        cursor = 0
        for message in messages:
            position = rendered.find(message.content, cursor)
            self.assertNotEqual(position, -1)
            cursor = position + len(message.content)

    @given(
        bos=st.text(max_size=6),
        user_prefix=st.text(max_size=6),
        assistant_prefix=st.text(max_size=6),
        turn_end=st.text(max_size=6),
        repeat_bos=st.booleans(),
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.differing_executors])
    def test_local_repairs_bounded_unique_and_stable(
        self, bos, user_prefix, assistant_prefix, turn_end, repeat_bos
    ):
        """Test local repairs bounded unique and stable."""
        base = Format("base", bos, user_prefix, assistant_prefix, turn_end, repeat_bos)
        before = replace(base)
        variants = local_repairs(base)
        self.assertEqual(base, before)
        self.assertLessEqual(len(variants), 5)
        self.assertGreaterEqual(len(variants), 1)
        self.assertIs(variants[0], base)
        names = [v.name for v in variants]
        self.assertEqual(len(set(names)), len(names))


class SelectionRuleTests(unittest.TestCase):
    """Selection rule tests."""

    def test_three_way_tie_abstains_without_held_out(self):
        """Test three way tie abstains without held out."""
        backend = FakeBackend()
        names = [BASE, replace(BASE, name="b"), replace(BASE, name="c")]
        result = run_experiment(
            backend, tuple(names), split(), boundary=ids("|"), max_new_tokens=len(RAW)
        )
        self.assertEqual(result["decision"], "ambiguous")
        self.assertIsNone(result["held_out"])
        self.assertIsNone(result["selected_candidate"])
        self.assertFalse(any("question:h" in prompt for prompt in backend.encoded))

    def test_search_failure_records_decision_and_selects_nothing(self):
        """Test search failure records decision and selects nothing."""
        backend = FakeBackend(Generation((), "error", "boom"))
        result = run_experiment(
            backend, (BASE,), split(), boundary=ids("|"), max_new_tokens=len(RAW)
        )
        self.assertEqual(result["decision"], "failed")
        self.assertIsNone(result["selected_candidate"])
        self.assertIsNone(result["held_out"])

    def test_cancellation_between_search_and_held_out_abstains(self):
        """Test cancellation between search and held out abstains."""
        # A counting event that trips only after the search phase completed:
        # evaluate() must have returned an eligible report, then the
        # post-search cancellation check in run_experiment fires.
        real_is_set = Event.is_set

        class CountingEvent(Event):
            """Counting event."""

            def __init__(self, trip_after):
                super().__init__()
                self.calls = 0
                self.trip_after = trip_after

            def is_set(self):
                """Is set."""
                self.calls += 1
                if self.calls > self.trip_after:
                    return True
                return real_is_set(self)

        # evaluate() performs exactly 11 event checks for 1 candidate x 2
        # fixtures (5 per fixture: loop-top, post-render, post-encode,
        # post-generate, post-decode, plus the final report-line check).
        # The 12th check is run_experiment's post-search cancellation guard.
        # This is intentionally white-box: adding/removing a checkpoint must
        # force a conscious update of this number.
        event = CountingEvent(trip_after=11)
        backend = FakeBackend()
        result = run_experiment(
            backend, (BASE,), split(), boundary=ids("|"), max_new_tokens=len(RAW), cancel=event
        )
        self.assertEqual(result["decision"], "cancelled")
        self.assertIsNone(result["selected_candidate"])
        self.assertIsNone(result["held_out"])
        # Held-out fixtures were never encoded.
        self.assertFalse(any("question:h" in prompt for prompt in backend.encoded))

    def test_winner_must_pass_every_search_fixture(self):
        """Test winner must pass every search fixture."""
        script = [GOOD, Generation(ids("wrong!|"), "limit")]  # 7 tokens == len(RAW)
        backend = FakeBackend(generate_script=script)
        result = run_experiment(
            backend, (BASE,), split(), boundary=ids("|"), max_new_tokens=len(RAW)
        )
        self.assertEqual(result["decision"], "no_candidate")
        self.assertIsNone(result["held_out"])

    def test_experiment_rejects_non_format_candidates_and_non_split(self):
        """Test experiment rejects non format candidates and non split."""
        backend = FakeBackend()
        with self.assertRaises(ValueError):
            run_experiment(backend, (object(),), split(), boundary=ids("|"))
        with self.assertRaises(ValueError):
            run_experiment(backend, (BASE,), object(), boundary=ids("|"))

    def test_experiment_rejects_oversized_partitions(self):
        """Test experiment rejects oversized partitions."""
        # Split itself has no fixture-count limit (that lives in datasets.py);
        # run_experiment enforces the experiment-level guard instead.
        small = Split((case("s", group="gs"),), (case("h", group="gh"),))
        oversized = replace(small, search=tuple(case(f"s{i}", group=f"g{i}") for i in range(101)))
        with self.assertRaises(ValueError):
            run_experiment(FakeBackend(), (BASE,), oversized, boundary=ids("|"))


class SanitizationTests(unittest.TestCase):
    """Sanitization tests."""

    def test_encode_exception_text_never_leaves_evaluate(self):
        """Test encode exception text never leaves evaluate."""
        backend = FakeBackend(encode_script=[RuntimeError(SECRET)])
        report = evaluate(backend, BASE, (case(),), boundary=ids("|"))
        self.assertEqual(report.observations[0].error, "encode_failed")
        self.assertNotIn(SECRET, repr(report))

    def test_render_exception_text_never_leaves_evaluate(self):
        """Test render exception text never leaves evaluate."""

        def bad_render(_messages):
            """Bad render."""
            raise RuntimeError(SECRET)

        with patch.object(Format, "render", side_effect=bad_render):
            report = evaluate(FakeBackend(), BASE, (case(),), boundary=ids("|"))
        self.assertEqual(report.observations[0].error, "render_failed")
        self.assertNotIn(SECRET, repr(report))

    def test_sanitized_failure_survives_json_serialization(self):
        """Test sanitized failure survives json serialization."""
        backend = FakeBackend(Generation(ids("ok|"), "error", SECRET))
        result = run_experiment(
            backend, (BASE,), split(), boundary=ids("|"), max_new_tokens=len(RAW)
        )
        with TemporaryDirectory() as tmp:
            json_path, _ = save_report(result, Path(tmp) / "r", include_evidence=True)
            saved = json_path.read_text(encoding="utf-8")
        self.assertNotIn(SECRET, saved)


class DeterminismTests(unittest.TestCase):
    """Determinism tests."""

    def test_repeated_runs_produce_byte_identical_reports(self):
        """Test repeated runs produce byte identical reports."""

        def build():
            """Build."""
            return run_experiment(
                FakeBackend(),
                (BASE,),
                split(),
                boundary=ids("|"),
                max_new_tokens=len(RAW),
                seed=123,
            )

        first, second = build(), build()
        with TemporaryDirectory() as tmp:
            first_json, first_html = save_report(first, Path(tmp) / "a")
            second_json, second_html = save_report(second, Path(tmp) / "b")
            self.assertEqual(first_json.read_bytes(), second_json.read_bytes())
            self.assertEqual(first_html.read_bytes(), second_html.read_bytes())

    def test_catalog_rendering_is_deterministic(self):
        """Test catalog rendering is deterministic."""
        for family in CATALOG:
            first = [f.render((Message("user", "q"),)) for f in candidates(family, mutate=True)]
            second = [f.render((Message("user", "q"),)) for f in candidates(family, mutate=True)]
            self.assertEqual(first, second)


class CLIEdgeTests(unittest.TestCase):
    """Cli edge tests."""

    def test_boundary_parser_rejects_superscript_digits(self):
        """Test boundary parser rejects superscript digits."""
        # Superscript two is not a decimal digit: rejected by the first check.
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_boundary("\u00b2")

    def test_boundary_parser_rejects_unparseable_decimal_digits(self):
        """Test boundary parser rejects unparseable decimal digits."""
        # 5000 decimal digits pass isdecimal() but int() refuses (digit limit):
        # the conversion failure must still fail closed as an argument error.
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_boundary("9" * 5000)

    def test_boundary_parser_rejects_unicode_digits(self):
        """Test boundary parser rejects unicode digits."""
        # Token IDs are ASCII by convention; Unicode digits are rejected
        # rather than silently coerced.
        for text in ("\u0663", "1\uff123", "\u00b2"):
            with self.subTest(text=text), self.assertRaises(argparse.ArgumentTypeError):
                parse_boundary(text)

    def test_no_command_is_an_argument_error(self):
        """Test no command is an argument error."""
        with self.assertRaises(SystemExit) as ctx:
            main([])
        self.assertEqual(ctx.exception.code, 2)

    def test_run_rejects_out_of_range_token_limits(self):
        """Test run rejects out of range token limits."""
        base = [
            "run",
            "--model",
            "m",
            "--dataset",
            "d",
            "--family",
            "chatml",
            "--boundary-ids",
            "1",
            "--output",
            "o",
        ]
        for flag, value in (
            ("--max-new-tokens", "0"),
            ("--max-new-tokens", "4097"),
            ("--max-context", "0"),
            ("--max-context", "32769"),
        ):
            with self.subTest(flag=flag, value=value), self.assertRaises(SystemExit) as ctx:
                main(base + [flag, value])
            self.assertEqual(ctx.exception.code, 2)

    def test_run_rejects_bad_seed(self):
        """Test run rejects bad seed."""
        with self.assertRaises(SystemExit) as ctx:
            main(
                [
                    "run",
                    "--model",
                    "m",
                    "--dataset",
                    "d",
                    "--family",
                    "chatml",
                    "--boundary-ids",
                    "1",
                    "--output",
                    "o",
                    "--seed",
                    str(2**64),
                ]
            )
        self.assertEqual(ctx.exception.code, 2)

    def test_run_rejects_existing_output_directory(self):
        """Test run rejects existing output directory."""
        with TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit) as ctx:
                main(
                    [
                        "run",
                        "--model",
                        "m",
                        "--dataset",
                        "d",
                        "--family",
                        "chatml",
                        "--boundary-ids",
                        "1",
                        "--output",
                        tmp,
                    ]
                )
            self.assertEqual(ctx.exception.code, 2)

    def test_run_rejects_unknown_family(self):
        """Test run rejects unknown family."""
        with self.assertRaises(SystemExit) as ctx:
            main(
                [
                    "run",
                    "--model",
                    "m",
                    "--dataset",
                    "d",
                    "--family",
                    "nope",
                    "--boundary-ids",
                    "1",
                    "--output",
                    "o",
                ]
            )
        self.assertEqual(ctx.exception.code, 2)

    def test_inspect_rejects_malformed_dataset(self):
        """Test inspect rejects malformed dataset."""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text('{"search": [], "held_out": []}', encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["inspect", str(path)]), 1)

    def test_catalog_lists_all_families(self):
        """Test catalog lists all families."""
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["catalog"]), 0)
        entries = json.loads(output.getvalue())
        self.assertEqual({entry["name"] for entry in entries}, set(CATALOG))


class ReportEdgeTests(unittest.TestCase):
    """Report edge tests."""

    def test_html_handles_empty_search_and_missing_held_out(self):
        """Test html handles empty search and missing held out."""
        result = {
            "decision": "not_started",
            "search": [],
            "held_out": None,
        }
        html = render_html(result)
        self.assertIn("Not evaluated", html)
        self.assertIn("default-src 'none'", html)

    def test_public_view_never_contains_nan_or_infinity(self):
        """Test public view never contains nan or infinity."""
        result = experiment()
        result["search"][0]["summary"]["rate"] = float("nan")
        with TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            save_report(result, Path(tmp) / "bad")

    def test_report_summary_rate_none_renders_incomplete(self):
        """Test report summary rate none renders incomplete."""
        result = experiment(FakeBackend(Generation((), "cancelled")))
        html = render_html(result)
        self.assertIn("incomplete", html)


if __name__ == "__main__":
    unittest.main()


class RunDecisionExitTests(unittest.TestCase):
    """Run decision exit tests."""

    @staticmethod
    def _dataset_file(tmp):
        """Dataset file."""
        path = Path(tmp) / "f.json"
        path.write_text(json.dumps(dataset_object()), encoding="utf-8")
        return path

    def test_run_exit_follows_returned_decision(self):
        """Test run exit follows returned decision."""
        from unittest.mock import MagicMock

        for decision, code in (
            ("cancelled", 130),
            ("failed", 1),
            ("compatible_on_fixture_set", 0),
            ("ambiguous", 2),
            ("no_candidate", 2),
            ("held_out_failed", 2),
        ):
            with self.subTest(decision=decision):
                backend = MagicMock()
                backend.fingerprint = "test-fingerprint"
                canned = {
                    "decision": decision,
                    "selected_candidate": None,
                    "search": [],
                    "held_out": None,
                }
                with TemporaryDirectory() as tmp:
                    output = Path(tmp) / "report"
                    with (
                        patch(
                            "model_protocol_lab.backends.LocalHFBackend",
                            return_value=backend,
                        ),
                        patch(
                            "model_protocol_lab.cli.run_experiment",
                            return_value=canned,
                        ),
                        redirect_stdout(io.StringIO()),
                    ):
                        self.assertEqual(
                            main(
                                [
                                    "run",
                                    "--model",
                                    "m",
                                    "--dataset",
                                    str(self._dataset_file(tmp)),
                                    "--family",
                                    "chatml",
                                    "--boundary-ids",
                                    "1",
                                    "--output",
                                    str(output),
                                ]
                            ),
                            code,
                        )
                    saved = json.loads((output / "results.json").read_text(encoding="utf-8"))
                    self.assertEqual(saved["decision"], decision)


class HeldOutFailureTests(unittest.TestCase):
    """Held out failure tests."""

    def test_search_winner_with_failing_held_out_records_failed(self):
        """Test search winner with failing held out records failed."""
        script = [GOOD, GOOD, Generation((), "error", "x")]
        backend = FakeBackend(generate_script=script)
        result = run_experiment(
            backend, (BASE,), split(), boundary=ids("|"), max_new_tokens=len(RAW)
        )
        self.assertEqual(result["decision"], "failed")
        self.assertEqual(result["selected_candidate"], BASE.name)
        self.assertFalse(result["held_out"]["summary"]["aggregate_eligible"])


class MainModuleTests(unittest.TestCase):
    """Main module tests."""

    def test_main_module_delegates_to_cli(self):
        """Test main module delegates to cli."""
        import runpy
        import sys

        with (
            patch.object(sys, "argv", ["model-protocol-lab", "catalog"]),
            redirect_stdout(io.StringIO()) as output,
            self.assertRaises(SystemExit) as ctx,
        ):
            runpy.run_module("model_protocol_lab", run_name="__main__")
        self.assertEqual(ctx.exception.code, 0)
        self.assertIn("chatml", output.getvalue())


class BoundaryTextMatchTests(unittest.TestCase):
    """Boundary text match tests."""

    @staticmethod
    def _case():
        """Case."""
        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        from tests.test_workflows import split

        return split().search[0]

    def test_found_boundary_matches_text(self):
        """Test found boundary matches text."""
        # Normal case: ID-level hit implies the text cross-check holds.
        backend = FakeBackend(generate_script=[GOOD])
        report = evaluate(
            backend, BASE, (self._case(),), boundary=ids("|"), max_new_tokens=len(RAW)
        )
        obs = report.observations[0]
        self.assertTrue(obs.boundary_found)
        self.assertTrue(obs.boundary_text_match)

    def test_tokenization_mismatch_flagged(self):
        """Test tokenization mismatch flagged."""
        # The model emits the boundary TEXT ("|") as token 2000 while the
        # configured boundary is token 1000: the ID-level match misses, but
        # the text cross-check catches it.
        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        from model_protocol_lab.core import Generation

        class MismatchBackend(FakeBackend):
            """Mismatch backend."""

            @staticmethod
            def decode(token_ids):
                """Decode."""
                if token_ids == (1000,):
                    return "|"
                return "".join("|" if t == 2000 else chr(t) for t in token_ids)

        emitted = Generation(ids("ok") + (2000,), "limit")
        backend = MismatchBackend(generate_script=[emitted])
        report = evaluate(backend, BASE, (self._case(),), boundary=(1000,), max_new_tokens=3)
        obs = report.observations[0]
        self.assertFalse(obs.boundary_found)
        self.assertTrue(obs.boundary_text_match)
        # The pass criterion is unchanged: the fixture still fails.
        self.assertFalse(obs.exact_match and obs.boundary_found)

    def test_genuine_miss_has_no_text_match(self):
        """Test genuine miss has no text match."""
        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        from model_protocol_lab.core import Generation

        backend = FakeBackend(generate_script=[Generation(ids("nope"), "limit")])
        report = evaluate(backend, BASE, (self._case(),), boundary=ids("|"), max_new_tokens=4)
        obs = report.observations[0]
        self.assertFalse(obs.boundary_found)
        self.assertFalse(obs.boundary_text_match)

    def test_decode_failure_degrades_to_none(self):
        """Test decode failure degrades to none."""

        class FlakyBackend(FakeBackend):
            """Flaky backend."""

            # skipcq: PYL-R0201 - zero-arg super() requires an instance method
            def decode(self, token_ids):
                """Decode."""
                if token_ids == (1000,):
                    raise RuntimeError("decode exploded")
                return super().decode(token_ids)

        backend = FlakyBackend(generate_script=[GOOD])
        report = evaluate(backend, BASE, (self._case(),), boundary=(1000,), max_new_tokens=len(RAW))
        obs = report.observations[0]
        self.assertIsNone(obs.boundary_text_match)
        self.assertEqual(obs.status, "completed")

    def test_terminated_observations_carry_none(self):
        """Test terminated observations carry none."""
        backend = FakeBackend(generate_script=[Generation((), "error", "x")])
        report = evaluate(
            backend, BASE, (self._case(),), boundary=ids("|"), max_new_tokens=len(RAW)
        )
        self.assertIsNone(report.observations[0].boundary_text_match)


class AdversarialLoaderTests(unittest.TestCase):
    """Adversarial loader tests."""

    @given(
        data=st.recursive(
            st.none() | st.booleans() | st.integers() | st.text(max_size=30),
            lambda children: (
                st.lists(children, max_size=4)
                | st.dictionaries(st.text(max_size=10), children, max_size=4)
            ),
            max_leaves=15,
        )
    )
    @settings(max_examples=500, suppress_health_check=[HealthCheck.differing_executors])
    def test_loader_never_crashes_on_adversarial_json(self, data):
        """Test loader never crashes on adversarial json."""
        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        import json as _json

        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        from model_protocol_lab.datasets import load_dataset

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "adv.json"
            try:
                text = _json.dumps(data, allow_nan=False)
            except (ValueError, TypeError):
                return  # not representable; nothing to test
            path.write_text(text, encoding="utf-8")
            try:
                split_result = load_dataset(path)
            except ValueError:
                return  # the only acceptable failure mode
            # Accepted: the result must satisfy every documented invariant.
            search_ids = [c.id for c in split_result.search]
            held_ids = [c.id for c in split_result.held_out]
            self.assertEqual(len(set(search_ids)), len(search_ids))
            self.assertEqual(len(set(held_ids)), len(held_ids))
            self.assertFalse(set(search_ids) & set(held_ids))
            for case in (*split_result.search, *split_result.held_out):
                self.assertTrue(case.id)
                self.assertTrue(case.messages)
                self.assertEqual(case.messages[0].role, "user")
                self.assertEqual(case.messages[-1].role, "user")

    @given(
        payload=st.sampled_from(
            [
                b"\xff\xfe\x00bad",
                b'{"search": [',
                b"",
                b"null",
                b"[]",
                b'{"search": [], "held_out": [], "extra": {}}',
            ]
        )
    )
    @settings(suppress_health_check=[HealthCheck.differing_executors])
    def test_loader_rejects_garbage_bytes(self, payload):
        """Test loader rejects garbage bytes."""
        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        from model_protocol_lab.datasets import load_dataset

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "garbage.json"
            path.write_bytes(payload)
            with self.assertRaises(ValueError):
                load_dataset(path)


class SecurityFixTests(unittest.TestCase):
    """Security fix tests."""

    def test_deeply_nested_json_raises_value_error(self):
        """Test deeply nested json raises value error."""
        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        from model_protocol_lab.datasets import parse_dataset

        payload = "[" * 200000 + "]" * 200000
        self.assertLess(len(payload.encode("utf-8")), 2_000_000)
        with self.assertRaisesRegex(ValueError, "nesting too deep"):
            parse_dataset(payload)

    def test_character_device_fixture_rejected(self):
        """Test character device fixture rejected."""
        import os

        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        from model_protocol_lab.datasets import load_dataset

        # /dev/null is a character device on POSIX; skip where it is absent.
        if not os.path.exists("/dev/null"):
            self.skipTest("/dev/null not available")
        with self.assertRaisesRegex(ValueError, "not a regular file"):
            load_dataset("/dev/null")

    def test_fifo_fixture_rejected_without_hanging(self):
        """Test fifo fixture rejected without hanging."""
        # A named FIFO with no writer must be refused, not block forever.
        # The test itself would hang (not fail) on a regression, so bound it
        # with a watchdog thread.
        import os
        import threading

        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        from model_protocol_lab.datasets import load_dataset

        with TemporaryDirectory() as tmp:
            fifo = Path(tmp) / "pipe.json"
            os.mkfifo(fifo)
            outcome = {}

            def attempt():
                """Attempt."""
                try:
                    load_dataset(fifo)
                except ValueError as exc:
                    outcome["error"] = str(exc)
                except Exception as exc:  # noqa: BLE001 - record, then fail
                    outcome["unexpected"] = repr(exc)

            worker = threading.Thread(target=attempt, daemon=True)
            worker.start()
            worker.join(timeout=10)
            self.assertFalse(worker.is_alive(), "load_dataset hung on a FIFO")
            self.assertIn("not a regular file", outcome.get("error", ""))
            self.assertNotIn("unexpected", outcome)

    def test_missing_fixture_file_is_value_error(self):
        """Test missing fixture file is value error."""
        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        from model_protocol_lab.datasets import load_dataset

        with (
            TemporaryDirectory() as tmp,
            self.assertRaisesRegex(ValueError, "Unreadable fixture file"),
        ):
            load_dataset(Path(tmp) / "does-not-exist.json")

    def test_report_files_have_restrictive_permissions(self):
        """Test report files have restrictive permissions."""
        import os as _os
        import stat as _stat

        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        from model_protocol_lab.reporting import save_report

        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        from tests.test_workflows import experiment

        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "report"
            json_path, html_path = save_report(experiment(), target)
            self.assertEqual(_stat.S_IMODE(_os.stat(target).st_mode), 0o700)
            self.assertEqual(_stat.S_IMODE(_os.stat(json_path).st_mode), 0o600)
            self.assertEqual(_stat.S_IMODE(_os.stat(html_path).st_mode), 0o600)

    def test_report_refuses_to_overwrite(self):
        """Test report refuses to overwrite."""
        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        from model_protocol_lab.reporting import save_report

        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        from tests.test_workflows import experiment

        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "report"
            save_report(experiment(), target)
            (target / "results.json").write_text("tampered", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                save_report(experiment(), target)


class InspectDiagnosticTests(unittest.TestCase):
    """Inspect diagnostic tests."""

    def test_inspect_reports_validation_reason(self):
        """Test inspect reports validation reason."""
        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        import json as _json

        with TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            bad.write_text(_json.dumps({"search": [], "held_out": []}), encoding="utf-8")
            with redirect_stdout(io.StringIO()), patch("sys.stderr", io.StringIO()) as err:
                code = main(["inspect", str(bad)])
            self.assertEqual(code, 1)
            self.assertIn("Invalid fixture count", err.getvalue())

    def test_inspect_missing_file_reports_clearly(self):
        """Test inspect missing file reports clearly."""
        with TemporaryDirectory() as tmp:
            with redirect_stdout(io.StringIO()), patch("sys.stderr", io.StringIO()) as err:
                code = main(["inspect", str(Path(tmp) / "missing.json")])
            self.assertEqual(code, 1)
            self.assertIn("inspect: invalid fixture file", err.getvalue())


class RenderRobustnessTests(unittest.TestCase):
    """Render robustness tests."""

    def test_none_search_renders(self):
        """Test none search renders."""
        html = render_html({"decision": "x", "search": None, "held_out": None})
        self.assertIn("No candidates were evaluated", html)

    def test_malformed_runs_render_safely(self):
        """Test malformed runs render safely."""
        html = render_html(
            {
                "decision": "x",
                "search": [
                    None,
                    {"format": {"name": 123}, "summary": None},
                    {"format": {}, "summary": {"rate": 0.5}},
                ],
                "held_out": {"summary": None},
            }
        )
        self.assertIn("123", html)  # int name coerced, escaped
        # No script element beyond the single allowlisted inline table sorter.
        scrubbed = html.replace("<script>" + _SCRIPT + "</script>", "")
        self.assertNotIn("<script>", scrubbed)

    def test_cli_shows_sanitized_error_code(self):
        """Test cli shows sanitized error code."""
        with TemporaryDirectory() as tmp:
            with patch("sys.stderr", io.StringIO()) as err:
                code = main(
                    [
                        "run",
                        "--model",
                        str(Path(tmp) / "missing"),
                        "--dataset",
                        "examples/fixtures.json",
                        "--family",
                        "chatml",
                        "--boundary-ids",
                        "1",
                        "--output",
                        str(Path(tmp) / "out"),
                    ]
                )
            self.assertEqual(code, 1)
            # The backend's sanitized wrapper code is shown, not internals.
            self.assertIn("local_backend_initialization_failed", err.getvalue())

    def test_output_dir_claimed_before_backend_init_and_released_on_failure(self):
        """Test output dir claimed before backend init and released on failure."""
        # M1: an unwritable output path must fail before model loading, and a
        # claimed-but-unused directory must not block a retry.
        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        import io

        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        from unittest.mock import patch

        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        from model_protocol_lab.cli import main

        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            with patch("sys.stderr", io.StringIO()):
                code = main(
                    [
                        "run",
                        "--model",
                        str(Path(tmp) / "missing-model"),
                        "--dataset",
                        "examples/fixtures.json",
                        "--family",
                        "chatml",
                        "--boundary-ids",
                        "1",
                        "--output",
                        str(out),
                    ]
                )
            self.assertEqual(code, 1)
            # Backend init failed -> the claimed dir was released.
            self.assertFalse(out.exists())

    def test_output_through_regular_file_fails_before_backend_init(self):
        """Test output through regular file fails before backend init."""
        # M1: --output beneath a regular file is a pre-detectable error.
        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        import io

        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        from unittest.mock import patch

        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        from model_protocol_lab.cli import main

        with TemporaryDirectory() as tmp:
            blocker = Path(tmp) / "afile"
            blocker.write_text("x")
            with patch("sys.stderr", io.StringIO()) as err:
                code = main(
                    [
                        "run",
                        "--model",
                        str(Path(tmp) / "missing-model"),
                        "--dataset",
                        "examples/fixtures.json",
                        "--family",
                        "chatml",
                        "--boundary-ids",
                        "1",
                        "--output",
                        str(blocker / "sub"),
                    ]
                )
            self.assertEqual(code, 1)
            self.assertIn("Operation failed", err.getvalue())

    def test_fingerprint_normalizes_null_bos_repeat(self):
        """Test fingerprint normalizes null bos repeat."""
        # L2: behaviorally identical hypotheses share a fingerprint, matching
        # local_repairs' dedup key.
        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        from model_protocol_lab.core import Format
        from model_protocol_lab.experiments import candidate_fingerprint

        base = Format("x", "", "u:", "a:", "\n", repeat_bos=False)
        variant = Format("x", "", "u:", "a:", "\n", repeat_bos=True)
        messages = (Message("user", "q"),)
        self.assertEqual(base.render(messages), variant.render(messages))
        self.assertEqual(candidate_fingerprint(base), candidate_fingerprint(variant))

    def test_report_intermediate_dirs_are_owner_only(self):
        """Test report intermediate dirs are owner only."""
        # L3: every created path component is 0o700, not just the leaf.
        import os
        import stat as _stat

        from model_protocol_lab.reporting import prepare_report_dir

        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "deep" / "a" / "b" / "c"
            prepare_report_dir(target)
            components = (target, target.parent, target.parent.parent)
            components += (target.parent.parent.parent,)
            for component in components:
                mode = _stat.S_IMODE(os.stat(component).st_mode)
                self.assertEqual(mode, 0o700, f"{component} has mode {oct(mode)}")

    def test_doctor_tokenizer_deep_failure_is_sanitized(self):
        """Test doctor tokenizer deep failure is sanitized."""
        # L4: a corrupt tokenizer.json yields a fixed code, not raw exception text.
        # Requires the hf stack: without transformers the deep check is skipped.
        if not HAS_HF:
            self.skipTest("the 'hf' extra is not installed")
        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        import json

        from model_protocol_lab.doctor import examine_model

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(
                json.dumps({"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"]})
            )
            from tests.test_doctor import write_minimal_safetensors

            write_minimal_safetensors(root / "model.safetensors")
            # Valid JSON shape but unusable as a tokenizer -> deep check fails.
            (root / "tokenizer.json").write_text(json.dumps({"version": "1.0"}))
            report = examine_model(root)
            detail = report["model"]["checks"]["tokenizer_deep"]["detail"]
            self.assertEqual(detail, "tokenizer_deep_check_failed")

    def test_inspect_warns_on_empty_expected(self):
        """Test inspect warns on empty expected."""
        # L1: empty expected answers are flagged as vacuously passing.
        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        import io

        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        import json

        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        from unittest.mock import patch

        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        from model_protocol_lab.cli import main

        payload = {
            "search": [
                {
                    "id": "s1",
                    "messages": [{"role": "user", "content": "search question"}],
                    "expected": "",
                    "group": "search-group",
                }
            ],
            "held_out": [
                {
                    "id": "h1",
                    "messages": [{"role": "user", "content": "heldout question"}],
                    "expected": "a",
                    "group": "heldout-group",
                }
            ],
        }
        with TemporaryDirectory() as tmp:
            fixture = Path(tmp) / "fx.json"
            fixture.write_text(json.dumps(payload))
            with patch("sys.stderr", io.StringIO()) as err, patch("sys.stdout", io.StringIO()):
                code = main(["inspect", str(fixture)])
            self.assertEqual(code, 0)
            self.assertIn("vacuously", err.getvalue())
            self.assertIn("s1", err.getvalue())

    def test_no_network_access_during_doctor_and_backend_init(self):
        """Test no network access during doctor and backend init."""
        # The HF adapter must never download: block all socket connections and
        # verify doctor + backend init + generation complete without any.
        # Requires the hf stack (builds a tiny local checkpoint).
        if not HAS_HF:
            self.skipTest("the 'hf' extra is not installed")
        import socket

        real_socket = socket.socket

        class BlockedSocket:
            """Blocked socket."""

            def __init__(self, *args, **kwargs):
                raise AssertionError("network access attempted")

        from model_protocol_lab.doctor import examine_model

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Minimal local GPT-2: reuse the adapter integration fixture builder.
            from tests.test_adapter_integration import build_tiny_gpt2

            build_tiny_gpt2(root)
            with patch("socket.socket", BlockedSocket):
                report = examine_model(root)
                self.assertEqual(report["verdict"], "ok")
                # skipcq: PYL-W0404 - test-local import isolates the module under audit
                from threading import Event

                from model_protocol_lab.backends import LocalHFBackend

                backend = LocalHFBackend(root, device="cpu", max_context=64)
                gen = backend.generate((1, 2, 3), max_new_tokens=4, seed=0, cancel=Event())
                self.assertEqual(gen.reason, "limit")
                self.assertEqual(len(gen.token_ids), 4)
        # Restore (patch context already handles it, but be explicit).
        self.assertIs(socket.socket, real_socket)

    def test_run_warns_on_empty_expected_before_backend_init(self):
        """Test run warns on empty expected before backend init."""
        # The run-path empty-expected warning fires even when backend init
        # later fails (warning precedes expensive work).
        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        import io

        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        import json

        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        from unittest.mock import patch

        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        from model_protocol_lab.cli import main

        payload = {
            "search": [
                {
                    "id": "se1",
                    "messages": [{"role": "user", "content": "search q"}],
                    "expected": "",
                    "group": "sg",
                }
            ],
            "held_out": [
                {
                    "id": "he1",
                    "messages": [{"role": "user", "content": "heldout q"}],
                    "expected": "a",
                    "group": "hg",
                }
            ],
        }
        with TemporaryDirectory() as tmp:
            fixture = Path(tmp) / "fx.json"
            fixture.write_text(json.dumps(payload))
            with patch("sys.stderr", io.StringIO()) as err:
                code = main(
                    [
                        "run",
                        "--model",
                        str(Path(tmp) / "missing-model"),
                        "--dataset",
                        str(fixture),
                        "--family",
                        "chatml",
                        "--boundary-ids",
                        "1",
                        "--output",
                        str(Path(tmp) / "out"),
                    ]
                )
            self.assertEqual(code, 1)
            self.assertIn("run: warning", err.getvalue())
            self.assertIn("se1", err.getvalue())

    def test_run_value_error_shows_sanitized_code(self):
        """Test run value error shows sanitized code."""
        # A fixture ValueError in run surfaces the fixed vocabulary (line 191).
        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        import io

        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        from unittest.mock import patch

        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        from model_protocol_lab.cli import main

        with TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            bad.write_text('{"search": [], "held_out": []}')
            with patch("sys.stderr", io.StringIO()) as err:
                code = main(
                    [
                        "run",
                        "--model",
                        str(Path(tmp) / "missing-model"),
                        "--dataset",
                        str(bad),
                        "--family",
                        "chatml",
                        "--boundary-ids",
                        "1",
                        "--output",
                        str(Path(tmp) / "out"),
                    ]
                )
            self.assertEqual(code, 1)
            self.assertIn("Operation failed [", err.getvalue())

    def test_deep_nested_output_chain_released_on_failure(self):
        """Test deep nested output chain released on failure."""
        # QA round 2, issue 1: a failed run with --output a/b/c removes the
        # whole created chain, not just the leaf.
        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        import io

        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        from unittest.mock import patch

        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        from model_protocol_lab.cli import main

        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "a" / "b" / "c"
            with patch("sys.stderr", io.StringIO()):
                code = main(
                    [
                        "run",
                        "--model",
                        str(Path(tmp) / "missing-model"),
                        "--dataset",
                        "examples/fixtures.json",
                        "--family",
                        "chatml",
                        "--boundary-ids",
                        "1",
                        "--output",
                        str(out),
                    ]
                )
            self.assertEqual(code, 1)
            self.assertFalse(out.exists())
            self.assertFalse((Path(tmp) / "a").exists())


class RoundTwoFixTests(unittest.TestCase):
    """Regression tests for the second hardening round (audit + red-team 2)."""

    def test_candidate_fingerprint_includes_name(self):
        """Test candidate fingerprint includes name."""
        # The docstring was corrected: the name IS part of the payload, so
        # identically-rendering formats with different names differ.
        from model_protocol_lab.experiments import candidate_fingerprint

        base = Format("base", "<B>", "<U>", "<A>", "<E>")
        alias = Format("alias", "<B>", "<U>", "<A>", "<E>")
        self.assertNotEqual(candidate_fingerprint(base), candidate_fingerprint(alias))
        self.assertEqual(candidate_fingerprint(base), candidate_fingerprint(base))

    def test_run_experiment_rejects_bad_boundary(self):
        """Test run experiment rejects bad boundary."""
        backend = FakeBackend()
        with self.assertRaisesRegex(ValueError, "Boundary must be a nonempty tuple"):
            run_experiment(backend, (BASE,), split(), boundary=999)
        with self.assertRaisesRegex(ValueError, "Boundary must be a nonempty tuple"):
            run_experiment(backend, (BASE,), split(), boundary=())

    def test_evaluate_and_run_experiment_reject_non_event_cancel(self):
        """Test evaluate and run experiment reject non event cancel."""
        backend = FakeBackend()
        with self.assertRaisesRegex(ValueError, "threading.Event"):
            evaluate(backend, BASE, (case("x"),), boundary=ids("|"), cancel="x")
        with self.assertRaisesRegex(ValueError, "threading.Event"):
            run_experiment(backend, (BASE,), split(), boundary=ids("|"), cancel=123)

    def test_run_experiment_rejects_non_string_fingerprint(self):
        """Test run experiment rejects non string fingerprint."""

        # A protocol-violating backend must fail fast, before any compute,
        # instead of exploding in json.dumps inside save_report.
        class BadFingerprintBackend(FakeBackend):
            """Bad fingerprint backend."""

            fingerprint = 123

        with self.assertRaisesRegex(ValueError, "fingerprint must be a string"):
            run_experiment(BadFingerprintBackend(), (BASE,), split(), boundary=ids("|"))

    def test_doctor_failure_reports_have_stable_shape(self):
        """Test doctor failure reports have stable shape."""
        from model_protocol_lab.doctor import examine_model

        report = examine_model("/nonexistent/path/xyz")
        self.assertEqual(report["verdict"], "failed")
        model = report["model"]
        self.assertEqual(model["assets"], [])
        self.assertEqual(model["special_tokens"], {})
        self.assertEqual(model["added_tokens"], {})

    def test_doctor_rejects_mistyped_boundary_ids(self):
        """Test doctor rejects mistyped boundary ids."""
        # Needs the hf stack: without transformers the boundary check is
        # skipped (vocab size unknown) rather than evaluated.
        if not HAS_HF:
            self.skipTest("the 'hf' extra is not installed")
        from model_protocol_lab.doctor import examine_model
        from tests.test_doctor import write_loadable_tokenizer, write_model_dir

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_model_dir(root)
            write_loadable_tokenizer(root, vocab_size=16)
            for bad in (("1",), 5, (), (None,), (True,)):
                report = examine_model(root, boundary_ids=bad)
                self.assertEqual(report["model"]["checks"]["boundary_ids"]["status"], "failed", bad)
            report = examine_model(root, boundary_ids=(0, 15))
            self.assertEqual(report["model"]["checks"]["boundary_ids"]["status"], "ok")

    def test_split_rejects_nfc_nfd_duplicate_ids(self):
        """Test split rejects nfc nfd duplicate ids."""
        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        from model_protocol_lab.core import Fixture, Message, Split

        def fx(fid, content):
            """Fx."""
            return Fixture(fid, (Message("user", content),), "a", "g1")

        with self.assertRaisesRegex(ValueError, "globally unique"):
            Split((fx("caf\u00e9", "q1"),), (fx("cafe\u0301", "q2"),))

    def test_split_rejects_nfc_nfd_cross_split_message_leak(self):
        """Test split rejects nfc nfd cross split message leak."""
        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        from model_protocol_lab.core import Fixture, Message, Split

        search = Fixture("s1", (Message("user", "caf\u00e9 au lait?"),), "a", "gs")
        held = Fixture("h1", (Message("user", "cafe\u0301 au lait?"),), "a", "gh")
        with self.assertRaisesRegex(ValueError, "Identical message inputs"):
            Split((search,), (held,))

    def test_split_rejects_nfc_nfd_group_overlap(self):
        """Test split rejects nfc nfd group overlap."""
        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        from model_protocol_lab.core import Fixture, Message, Split

        search = Fixture("s1", (Message("user", "q1"),), "a", "caf\u00e9")
        held = Fixture("h1", (Message("user", "q2"),), "a", "cafe\u0301")
        with self.assertRaisesRegex(ValueError, "cannot cross the split"):
            Split((search,), (held,))

    def test_inspect_warning_strips_terminal_escapes(self):
        """Test inspect warning strips terminal escapes."""
        # A hostile fixture ID must not inject terminal escape sequences
        # into the empty-expected stderr warning.
        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        import io

        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        from unittest.mock import patch

        payload = {
            "search": [
                {
                    "id": "s1\x1b]0;PWNED\x07",
                    "group": "g",
                    "messages": [{"role": "user", "content": "q"}],
                    "expected": "",
                }
            ],
            "held_out": [
                {
                    "id": "h1",
                    "group": "gh",
                    "messages": [{"role": "user", "content": "q2"}],
                    "expected": "a",
                }
            ],
        }
        with TemporaryDirectory() as tmp:
            fixture = Path(tmp) / "fx.json"
            fixture.write_text(json.dumps(payload))
            with patch("sys.stderr", io.StringIO()) as err, patch("sys.stdout", io.StringIO()):
                code = main(["inspect", str(fixture)])
            self.assertEqual(code, 0)
            self.assertNotIn("\x1b", err.getvalue())
            self.assertNotIn("\x07", err.getvalue())
            self.assertIn("vacuously", err.getvalue())

    def test_load_dataset_rejects_non_path_input(self):
        """Test load dataset rejects non path input."""
        with self.assertRaisesRegex(ValueError, "Invalid fixture path"):
            load_dataset(123)
        with self.assertRaisesRegex(ValueError, "Invalid fixture path"):
            load_dataset(None)


class RoundThreeFixTests(unittest.TestCase):
    """Regression tests for the third hardening round (fresh hostile audit)."""

    def test_doctor_config_failure_keeps_uniform_shape(self):
        """Test doctor config failure keeps uniform shape."""
        # The config-read failure path returned {"verdict", "model": {"checks"}}
        # without the assets/special_tokens/added_tokens keys that
        # _model_result guarantees on every other path.
        from model_protocol_lab.doctor import examine_model
        from tests.test_doctor import write_model_dir

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_model_dir(root)
            (root / "config.json").write_bytes(b"{invalid json")
            report = examine_model(root)
            self.assertEqual(report["verdict"], "failed")
            self.assertEqual(report["model"]["checks"]["config"]["status"], "failed")
            model = report["model"]
            self.assertIn("assets", model)
            self.assertIn("special_tokens", model)
            self.assertIn("added_tokens", model)
            self.assertEqual(model["special_tokens"], {})
            self.assertEqual(model["added_tokens"], {})
            # The manifest was already computed before config parsing, so the
            # failure report still carries the asset identity records.
            self.assertTrue(model["assets"])
            self.assertTrue(all("sha256" in asset for asset in model["assets"]))

    def test_doctor_config_detail_uses_fixed_vocabulary(self):
        """Test doctor config detail uses fixed vocabulary."""
        # Raw codec/parser text (UnicodeDecodeError, JSONDecodeError,
        # RecursionError) from the inspected config.json must not reach the
        # check detail; it collapses to the fixed "invalid_asset_json" code.
        from model_protocol_lab.doctor import examine_model
        from tests.test_doctor import write_model_dir

        def detail_for(payload: bytes) -> str:
            """Detail for."""
            with TemporaryDirectory() as tmp:
                root = Path(tmp)
                write_model_dir(root)
                (root / "config.json").write_bytes(payload)
                report = examine_model(root)
                self.assertEqual(report["model"]["checks"]["config"]["status"], "failed")
                return report["model"]["checks"]["config"]["detail"]

        self.assertEqual(detail_for(b"{invalid json"), "invalid_asset_json")
        self.assertEqual(detail_for(b'{"model_type": "gpt2", \xff}'), "invalid_asset_json")
        nested = ("[" * 20000 + "]" * 20000).encode("utf-8")
        self.assertEqual(detail_for(nested), "invalid_asset_json")

    def test_read_json_rejects_raw_codec_and_parser_failures(self):
        """Test read json rejects raw codec and parser failures."""
        # Direct unit coverage for the _read_json vocabulary normalization.
        from model_protocol_lab.backends import _read_json

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad_utf8 = root / "bad_utf8.json"
            bad_utf8.write_bytes(b"\xff\xfe{")
            with self.assertRaisesRegex(ValueError, "invalid_asset_json"):
                _read_json(bad_utf8)
            bad_json = root / "bad_json.json"
            bad_json.write_bytes(b'{"a": }')
            with self.assertRaisesRegex(ValueError, "invalid_asset_json"):
                _read_json(bad_json)
            nested = root / "nested.json"
            nested.write_bytes(("[" * 20000 + "]" * 20000).encode("utf-8"))
            with self.assertRaisesRegex(ValueError, "invalid_asset_json"):
                _read_json(nested)
            good = root / "good.json"
            good.write_bytes(b'{"a": 1}')
            self.assertEqual(_read_json(good), {"a": 1})

    def test_run_experiment_rejects_backend_missing_fingerprint(self):
        """Test run experiment rejects backend missing fingerprint."""

        # A backend with no fingerprint attribute at all raised a raw
        # AttributeError; it must fail fast with the documented ValueError.
        class NoFingerprint:
            """No fingerprint."""

            pass

        with self.assertRaisesRegex(ValueError, "fingerprint must be a string"):
            run_experiment(NoFingerprint(), (BASE,), split(), boundary=ids("|"))

    def test_evaluate_rejects_nfc_nfd_duplicate_ids(self):
        """Test evaluate rejects nfc nfd duplicate ids."""
        # evaluate's uniqueness check used raw string equality while Split
        # compares NFC-normalized keys (red-team round 2): canonically
        # equivalent IDs evaded evaluate's check.
        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        from model_protocol_lab.core import Fixture, Message

        fx1 = Fixture("caf\u00e9", (Message("user", "q"),), "a", "g")
        fx2 = Fixture("cafe\u0301", (Message("user", "q"),), "a", "g")
        with self.assertRaisesRegex(ValueError, "Evaluation fixture IDs must be unique"):
            evaluate(FakeBackend(), BASE, (fx1, fx2), boundary=ids("|"))
        # Genuinely distinct IDs still pass validation.
        fx3 = Fixture("other", (Message("user", "q"),), "a", "g")
        report = evaluate(
            FakeBackend(), BASE, (fx1, fx3), boundary=ids("|"), max_new_tokens=len(RAW)
        )
        self.assertEqual(report.status, "completed")

    def test_prepare_report_dir_never_chmods_through_symlink(self):
        """Test prepare report dir never chmods through symlink."""
        # os.chmod follows symlinks: claiming a report dir through a symlinked
        # leaf used to tighten permissions on the link target, a directory
        # outside the requested tree. A symlinked leaf is now rejected
        # outright (FileExistsError), even with _allow_empty_existing --
        # report files must never be written through a link into an
        # attacker-chosen directory.
        import os
        import stat as _stat

        from model_protocol_lab.reporting import prepare_report_dir

        with TemporaryDirectory() as tmp:
            real = Path(tmp) / "real"
            real.mkdir()
            # A non-700 baseline is required: TemporaryDirectory itself is
            # 0o700, which would mask the chmod-through-symlink bug.
            os.chmod(real, 0o755)  # noqa: S103 -- test baseline, not a product permission
            before = _stat.S_IMODE(os.stat(real).st_mode)
            link = Path(tmp) / "link"
            link.symlink_to(real, target_is_directory=True)
            with self.assertRaises(FileExistsError):
                prepare_report_dir(link, _allow_empty_existing=True)
            after = _stat.S_IMODE(os.stat(real).st_mode)
            self.assertEqual(
                after, before, f"link target mode changed {oct(before)} -> {oct(after)}"
            )
            self.assertEqual(list(real.iterdir()), [])

    def test_prepare_report_dir_still_secures_real_components(self):
        """Test prepare report dir still secures real components."""
        # The symlink guard must not weaken the normal guarantee: created
        # components are still owner-only.
        import os
        import stat as _stat

        from model_protocol_lab.reporting import prepare_report_dir

        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "a" / "b"
            prepare_report_dir(target)
            for component in (target, target.parent):
                mode = _stat.S_IMODE(os.stat(component).st_mode)
                self.assertEqual(mode, 0o700, f"{component} has mode {oct(mode)}")
