"""Cycle 3 tests: doctor suggestion window-fit and failed-run diagnosis.

Covers the two UX findings from the Cycle 3 first-run walk:
(1) the doctor's suggested `run` command sized `--max-new-tokens` without
regard for the checkpoint's declared window, so the suggested command could
fail with `context_limit_exceeded` on small-window checkpoints;
(2) a failed run's report banner said only "backend-error", hiding the
sanitized failure code that names the cause.
"""

import importlib.util
import json
import re
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from model_protocol_lab.core import Generation

REQUIRED = ("torch", "transformers", "safetensors", "tokenizers")
HAS_HF = all(importlib.util.find_spec(name) is not None for name in REQUIRED)

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = REPO_ROOT / "examples" / "fixtures.json"


def _suggested_new_tokens(command: str) -> int:
    """Suggested new tokens."""
    match = re.search(r"--max-new-tokens (\d+)", command)
    assert match is not None, f"no --max-new-tokens in suggestion: {command!r}"
    return int(match.group(1))


@unittest.skipUnless(HAS_HF, "the 'hf' extra is not installed")
class DoctorSuggestionSizingTests(unittest.TestCase):
    """Doctor suggestion sizing tests."""

    @classmethod
    def setUpClass(cls):
        """Set up the test fixture."""
        from tests.test_adapter_integration import build_tiny_gpt2

        cls._tmp = TemporaryDirectory()
        cls.addClassCleanup(cls._tmp.cleanup)
        cls.base = Path(cls._tmp.name) / "tiny"
        build_tiny_gpt2(cls.base)

    def _variant(self, n_positions):
        """Variant."""
        from tests.test_adapter_integration import build_tiny_gpt2  # noqa: F401

        root = Path(self._tmp.name) / f"tiny-{n_positions}"
        if not root.exists():
            shutil.copytree(self.base, root)
        config_path = root / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["n_positions"] = n_positions
        config_path.write_text(json.dumps(config), encoding="utf-8")
        return root

    @staticmethod
    def _doctor(model_path):
        """Doctor."""
        from model_protocol_lab.doctor import doctor

        return doctor(str(model_path), str(FIXTURES), boundary_ids=(62,))

    def test_suggestion_sizes_new_tokens_to_fit_declared_window(self):
        """Test suggestion sizes new tokens to fit declared window."""
        from model_protocol_lab.doctor import _build_primary_tokenizer, _longest_prompt_tokens

        # A window comfortably larger than the catalog's longest render, so
        # the fit (not warning) path is exercised regardless of catalog size.
        window = 512
        report = self._doctor(self._variant(window))
        self.assertEqual(report["verdict"], "ok")
        suggestion = report["suggestion"]
        self.assertIsNotNone(suggestion)
        new_tokens = _suggested_new_tokens(suggestion["next_command"])
        tokenizer = _build_primary_tokenizer(self._variant(window))
        longest = _longest_prompt_tokens(tokenizer, FIXTURES)
        self.assertIsNotNone(longest)
        # Pins the exact sizing formula: largest budget (up to 64) such that
        # prompt + budget still fits the declared window.
        self.assertEqual(new_tokens, max(1, min(64, window - longest)))
        # The backend refuses generation when prompt + new tokens exceed the
        # window; the suggestion must fit.
        self.assertLessEqual(longest + new_tokens, window)
        self.assertNotIn("WARNING", suggestion["note"])
        self.assertIn("sized so the longest fixture prompt", suggestion["note"])

    def test_suggestion_keeps_default_for_large_window(self):
        """Test suggestion keeps default for large window."""
        report = self._doctor(self._variant(2048))
        suggestion = report["suggestion"]
        self.assertIsNotNone(suggestion)
        self.assertEqual(_suggested_new_tokens(suggestion["next_command"]), 64)
        self.assertNotIn("WARNING", suggestion["note"])

    def test_suggestion_warns_when_prompt_fills_window(self):
        """Test suggestion warns when prompt fills window."""
        report = self._doctor(self._variant(8))
        self.assertEqual(report["verdict"], "ok")
        suggestion = report["suggestion"]
        self.assertIsNotNone(suggestion)
        self.assertEqual(_suggested_new_tokens(suggestion["next_command"]), 1)
        self.assertIn("WARNING", suggestion["note"])


class NextCommandDegradationTests(unittest.TestCase):
    """Next command degradation tests."""

    @staticmethod
    def _report():
        """Report."""
        return {"verdict": "ok", "model": {"context_window": 64}}

    def test_no_tokenizer_keeps_default_budget(self):
        """Test no tokenizer keeps default budget."""
        from model_protocol_lab.doctor import _next_command

        suggestion = _next_command("model", str(FIXTURES), (62,), self._report())
        self.assertIsNotNone(suggestion)
        self.assertEqual(_suggested_new_tokens(suggestion["next_command"]), 64)
        self.assertNotIn("sized so the longest fixture prompt", suggestion["note"])

    def test_longest_prompt_tokens_degrades_gracefully(self):
        """Test longest prompt tokens degrades gracefully."""
        from model_protocol_lab.doctor import _longest_prompt_tokens

        # No tokenizer, or an unreadable dataset: None, never an exception.
        self.assertIsNone(_longest_prompt_tokens(None, str(FIXTURES)))
        self.assertIsNone(_longest_prompt_tokens(None, "/nonexistent.json"))

    def test_sizing_degrades_when_measurement_fails(self):
        """Test sizing degrades when measurement fails."""
        from model_protocol_lab.doctor import _next_command, _suggestion_tokenizer

        report = {"verdict": "ok", "model": {"context_window": 64}}
        # A tokenizer is present but the dataset cannot be measured.
        suggestion = _next_command("m", "/nonexistent.json", (1,), report, tokenizer=object())
        self.assertIsNotNone(suggestion)
        self.assertEqual(_suggested_new_tokens(suggestion["next_command"]), 64)
        # The deep check passed but the model directory is unreadable.
        deep_ok = {
            "verdict": "ok",
            "model": {"checks": {"tokenizer_deep": {"status": "ok"}}},
        }
        self.assertIsNone(_suggestion_tokenizer(deep_ok, "/nonexistent"))
        # The deep check did not pass: no tokenizer is attempted.
        self.assertIsNone(_suggestion_tokenizer({"model": {"checks": {}}}, "m"))


class AuthoringBoundaryTests(unittest.TestCase):
    """Authoring boundary tests."""

    def test_exactly_at_size_limit_is_accepted(self):
        """Test exactly at size limit is accepted."""
        # Found by the Cycle 3 mutation-testing trial (cosmic-ray): the
        # mutant `len(content) > MAX_BYTES` -> `>=` survived, so no test
        # pinned the boundary. A file of exactly MAX_BYTES bytes is within
        # the documented limit and must be accepted.
        from model_protocol_lab.datasets import MAX_BYTES
        from model_protocol_lab.fixtures import read_authoring_file

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.csv"
            path.write_bytes(b"x" * MAX_BYTES)
            self.assertEqual(read_authoring_file(path), "x" * MAX_BYTES)

    def test_exactly_at_case_limit_is_accepted(self):
        """Test exactly at case limit is accepted."""
        # Same trial, same pattern: `len(cases) > MAX_CASES_PER_SPLIT` ->
        # `>=` survived. Exactly MAX_CASES_PER_SPLIT fixtures per split is
        # within the documented limit ("more than N" is the refusal).
        from model_protocol_lab.datasets import MAX_CASES_PER_SPLIT
        from model_protocol_lab.fixtures import parse_fixture_csv

        rows = ["id,group,partition,expected,user_1"]
        for index in range(MAX_CASES_PER_SPLIT):
            rows.append(f"s{index},g{index},search,ok,q{index}")
        rows.append("h0,gh,held_out,ok,qh")
        split, _ = parse_fixture_csv("\n".join(rows) + "\n")
        self.assertEqual(len(split.search), MAX_CASES_PER_SPLIT)
        self.assertEqual(len(split.held_out), 1)

    def test_all_blank_cell_rows_are_skipped_like_blank_lines(self):
        """Test all blank cell rows are skipped like blank lines."""
        # Same trial: `not record or all(...)` -> `and` survived. A row of
        # only empty cells is a blank line for authoring purposes and must
        # be skipped, not rejected downstream.
        from model_protocol_lab.fixtures import parse_fixture_csv

        split, _ = parse_fixture_csv(
            "id,group,partition,expected,user_1\n"
            "s1,g1,search,ok,question one\n"
            ",,,\n"
            "h1,g2,held_out,ok,question two\n"
        )
        self.assertEqual([c.id for c in split.search], ["s1"])
        self.assertEqual([c.id for c in split.held_out], ["h1"])


class FailureCodeTests(unittest.TestCase):
    """Failure code tests."""

    def test_allowlisted_backend_code_passes_through(self):
        """Test allowlisted backend code passes through."""
        from tests.test_core import FakeBackend, run_case

        backend = FakeBackend(Generation((), "error", "context_limit_exceeded"))
        report = run_case(backend)
        self.assertEqual(report.status, "failed")
        self.assertEqual(report.observations[0].error, "context_limit_exceeded")

    def test_arbitrary_backend_text_still_collapses(self):
        """Test arbitrary backend text still collapses."""
        from tests.test_core import FakeBackend, run_case

        backend = FakeBackend(Generation((), "error", "weird <script>alert(1)"))
        report = run_case(backend)
        self.assertEqual(report.observations[0].error, "backend_reported_error")

    def test_passing_report_has_no_failure_code(self):
        """Test passing report has no failure code."""
        from model_protocol_lab.experiments import _summary
        from tests.test_core import FakeBackend, run_case

        summary = _summary(run_case(FakeBackend()))
        self.assertIsNone(summary["failure_code"])

    def test_run_experiment_failure_code_reaches_banner(self):
        """Test run experiment failure code reaches banner."""
        from model_protocol_lab.experiments import run_experiment
        from model_protocol_lab.reporting import _failure_diagnosis, render_html
        from tests.test_core import FakeBackend
        from tests.test_workflows import BASE, ids, split

        backend = FakeBackend(generate_script=[Generation((), "error", "context_limit_exceeded")])
        result = run_experiment(
            backend,
            (BASE,),
            split(),
            boundary=ids("|"),
            max_new_tokens=8,
        )
        self.assertEqual(result["decision"], "failed")
        summary = result["search"][0]["summary"]
        self.assertEqual(summary["failure_code"], "context_limit_exceeded")
        diagnosis = _failure_diagnosis(result)
        self.assertIn("context_limit_exceeded", diagnosis)
        self.assertIn("declared context window", diagnosis)
        html = render_html(result)
        self.assertIn("Failure diagnosis", html)
        self.assertIn("context_limit_exceeded", html)

    def test_failure_diagnosis_empty_for_other_decisions(self):
        """Test failure diagnosis empty for other decisions."""
        from model_protocol_lab.reporting import _failure_diagnosis

        self.assertEqual(_failure_diagnosis({"decision": "no_candidate"}), "")
        self.assertEqual(
            _failure_diagnosis({"decision": "failed", "search": []}),
            "",
        )
        # Unknown codes still surface with generic guidance, never raw text.
        result = {
            "decision": "failed",
            "search": [{"summary": {"failure_code": "mystery_code"}}],
            "held_out": None,
        }
        diagnosis = _failure_diagnosis(result)
        self.assertIn("mystery_code", diagnosis)


if __name__ == "__main__":
    unittest.main()
