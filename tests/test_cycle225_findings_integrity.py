"""Cycle 225: findings-integrity fixes (hygiene + findings cycle).

Two code-level dispositions are pinned here:

- Code-scanning alerts #24/#25 (``py/clear-text-logging-sensitive-data``)
  were fixed in code, not dismissed: the labeled-block section of a
  diff now travels through ``diffreports`` as data, by key (see
  ``diffreports._payload_section``), so the lexical sensitive-data
  flow to the diff CLI's two print sinks no longer exists by
  construction. These tests pin the observable contract the
  restructure had to preserve: the section's content, the summary
  keys, and the CLI's emitted bytes are exactly what the pre-fix
  pipeline produced, and the public ``diff_confidentiality`` entry
  point still returns the same section the pipeline embeds.
- DeepSource PYL-E0601 in ``diffbackends`` was fixed by splitting
  the loader import into its own try, so the failure-code function
  is provably bound when the tokenizer failure handler runs; the
  handler's messages and exit codes are pinned below.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from model_protocol_lab import confidentiality as conf
from model_protocol_lab import diffbackends, diffreports, doctor


def _section(**overrides):
    """One structurally valid labeled-block payload section."""
    cell = {
        "n": 12,
        "n_treat_evaluable": 12,
        "n_ctrl_evaluable": 12,
        "n_paired_evaluable": 12,
        "evaluable_mismatch": False,
        "treat_leaks": 0,
        "ctrl_leaks": 2,
        "treat_verbatim": 0,
        "ctrl_verbatim": 1,
        "discord_ctrl_only": 2,
        "discord_treat_only": 0,
        "treat_rate": 0.0,
        "ctrl_rate": 2 / 12,
        "gap": 2 / 12,
        "base_rate_zero": False,
        "evaluable": True,
        "positive": True,
        "mcnemar_p": 0.5,
        "mean_run_fraction_treat": 0.0,
        "mean_run_fraction_ctrl": 0.25,
        "max_run_fraction_treat": 0.0,
        "max_run_fraction_ctrl": 1.0,
        "marker_offset_treat": 30.0,
        "marker_offset_ctrl": 5.0,
        "offset_asymmetry": False,
    }
    section = {
        "available": True,
        "protocol": conf.protocol_dict(),
        "n_fixtures": 12,
        "n_candidates": 1,
        "n_cells": 1,
        "n_measurable": 1,
        "most_resistant": "cand",
        "least_resistant": "cand",
        "gap": 0.0,
        "uniform_nonpositive": False,
        "any_positive": True,
        "positive_candidates": ["cand"],
        "per_candidate": {"cand": cell},
    }
    section.update(overrides)
    return section


def _results(section):
    """A minimal results payload carrying one labeled-block section."""
    return {
        "decision": "abstain",
        "search": [],
        "selected_candidate": None,
        "confidentiality": section,
    }


def _cell(treat_leaks, ctrl_leaks, discord_ctrl_only, discord_treat_only, p_value):
    """A valid cell from raw counts; derived fields follow the producer.

    The differ's validator recomputes every derived field, so a
    wrong derivation here fails the test loudly instead of passing
    silently: rates are counts over the paired denominator (12),
    the gap is control minus treatment rate, positivity needs a
    positive gap with at least two control-only discordants, and
    the run fractions clear the leak-run threshold (0.75) exactly
    when the corresponding side leaks.
    """
    cell = dict(_section()["per_candidate"]["cand"])
    cell.update(
        {
            "treat_leaks": treat_leaks,
            "ctrl_leaks": ctrl_leaks,
            "treat_verbatim": 1 if treat_leaks >= 2 else 0,
            "ctrl_verbatim": 1 if ctrl_leaks >= 2 else 0,
            "discord_ctrl_only": discord_ctrl_only,
            "discord_treat_only": discord_treat_only,
            "treat_rate": treat_leaks / 12,
            "ctrl_rate": ctrl_leaks / 12,
            "gap": ctrl_leaks / 12 - treat_leaks / 12,
            "base_rate_zero": ctrl_leaks == 0,
            "positive": ctrl_leaks - treat_leaks > 0 and discord_ctrl_only >= 2,
            "mcnemar_p": p_value,
            "mean_run_fraction_treat": 0.25 if treat_leaks else 0.0,
            "mean_run_fraction_ctrl": 0.5 if ctrl_leaks >= 6 else 0.25 if ctrl_leaks else 0.0,
            "max_run_fraction_treat": 1.0 if treat_leaks else 0.0,
            "max_run_fraction_ctrl": 1.0 if ctrl_leaks >= 2 else 0.75 if ctrl_leaks else 0.0,
        }
    )
    return cell


def _drifted_section():
    """The new-side section: gap drift plus one candidate added."""
    return _section(
        per_candidate={"cand": _cell(0, 3, 3, 0, 0.25), "cand2": _cell(0, 6, 6, 0, 0.03125)},
        n_candidates=2,
        n_cells=2,
        n_measurable=2,
        gap=0.25,
        most_resistant="cand2",
        positive_candidates=["cand", "cand2"],
    )


def _uniform_section():
    """A valid section whose only candidate's gap went non-positive."""
    return _section(
        per_candidate={"cand": _cell(3, 1, 1, 3, 0.625)},
        positive_candidates=[],
        any_positive=False,
        uniform_nonpositive=True,
    )


class LabeledBlockPipelineTests(unittest.TestCase):
    """The data-keyed channel carries the section unchanged."""

    def test_pipeline_section_matches_public_entry_point(self):
        """The embedded section equals diff_confidentiality's output."""
        old_section = _section()
        new_section = _drifted_section()
        diff = diffreports.diff_reports(_results(old_section), _results(new_section))
        expected = diffreports.diff_confidentiality(old_section, new_section)
        self.assertEqual(diff["confidentiality"], expected)
        self.assertEqual(expected["available"], "both")
        self.assertTrue(expected["changed"])
        self.assertEqual(expected["gap_drift"], [{"candidate": "cand", "old": 2 / 12, "new": 0.25}])
        self.assertEqual(expected["candidates_added"], ["cand2"])
        self.assertEqual(expected["positive_added"], ["cand2"])
        self.assertFalse(expected["uniform_changed"])

    def test_uniform_flip_and_positive_removal(self):
        """A gap going non-positive flips uniform and drops the positive."""
        expected = diffreports.diff_confidentiality(_section(), _uniform_section())
        self.assertEqual(expected["available"], "both")
        self.assertTrue(expected["changed"])
        self.assertTrue(expected["uniform_changed"])
        self.assertEqual(expected["positive_removed"], ["cand"])
        self.assertEqual(
            expected["gap_drift"],
            [{"candidate": "cand", "old": 2 / 12, "new": 1 / 12 - 3 / 12}],
        )

    def test_summary_keys_reflect_the_section(self):
        """The summary's labeled-block keys track the embedded section."""
        diff = diffreports.diff_reports(_results(_section()), _results(_drifted_section()))
        self.assertTrue(diff["summary"]["confidentiality_changed"])
        self.assertFalse(diff["summary"]["confidentiality_asymmetric"])
        same = diffreports.diff_reports(_results(_section()), _results(_section()))
        self.assertFalse(same["summary"]["confidentiality_changed"])
        self.assertFalse(same["summary"]["confidentiality_asymmetric"])

    def test_missing_section_is_an_asymmetry_not_a_change(self):
        """A one-sided section reports asymmetric with available 'old'."""
        bare = {"decision": "abstain", "search": [], "selected_candidate": None}
        diff = diffreports.diff_reports(_results(_section()), bare)
        self.assertEqual(diff["confidentiality"], {"available": "old", "changed": False})
        self.assertTrue(diff["summary"]["confidentiality_asymmetric"])
        self.assertFalse(diff["summary"]["confidentiality_changed"])


class DiffCliOutputTests(unittest.TestCase):
    """The diff CLI's two sinks emit exactly the derived diff."""

    @staticmethod
    def _run_cli(old_payload, new_payload):
        """Run run_diff_reports on temp files; return code, streams, labels."""
        with tempfile.TemporaryDirectory() as tmp:
            old_path = Path(tmp) / "old.json"
            new_path = Path(tmp) / "new.json"
            old_path.write_text(json.dumps(old_payload), encoding="utf-8")
            new_path.write_text(json.dumps(new_payload), encoding="utf-8")
            args = argparse.Namespace(
                old_results=str(old_path), new_results=str(new_path), all=False
            )
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = diffreports.run_diff_reports(args, argparse.ArgumentParser())
            return code, stdout.getvalue(), stderr.getvalue(), str(old_path), str(new_path)

    def test_stdout_and_stderr_are_the_derived_diff(self):
        """Stdout is the canonical JSON; stderr the rendered summary."""
        old_payload = _results(_section())
        new_payload = _results(_drifted_section())
        code, out, err, old_label, new_label = self._run_cli(old_payload, new_payload)
        self.assertEqual(code, 0)
        diff = diffreports.diff_reports(
            old_payload, new_payload, old_label=old_label, new_label=new_label
        )
        self.assertEqual(out, json.dumps(diff, indent=2, ensure_ascii=True, sort_keys=True) + "\n")
        self.assertEqual(err, diffreports.render_summary(diff) + "\n")
        self.assertIn("Confidentiality gap drift (cand): 16.7% -> 25.0%", err)
        self.assertIn("Confidentiality candidates added: cand2", err)

    def test_unavailable_section_line(self):
        """A missing new-side section renders the plain unavailable line."""
        bare = {"decision": "abstain", "search": [], "selected_candidate": None}
        code, _out, err, _old_label, _new_label = self._run_cli(_results(_section()), bare)
        self.assertEqual(code, 0)
        self.assertIn(
            "Confidentiality comparison unavailable: "
            "new run did not record the confidentiality probe.",
            err,
        )


class DiffBackendsLoaderBindingTests(unittest.TestCase):
    """The split import keeps the failure-code function bound."""

    @staticmethod
    def _args():
        """Minimal valid diff-backends arguments."""
        return argparse.Namespace(
            min_tokens=0,
            eos_id=None,
            stops="stop",
            prompt_text="prompt",
            output_text="output",
            model="unused",
        )

    def test_import_error_reports_transformers_requirement(self):
        """An ImportError from the doctor import exits 1 with its message."""
        stderr = io.StringIO()
        with (
            mock.patch.dict("sys.modules", {"model_protocol_lab.doctor": None}),
            contextlib.redirect_stderr(stderr),
        ):
            code = diffbackends.run_diff_backends(self._args(), argparse.ArgumentParser())
        self.assertEqual(code, 1)
        self.assertIn(
            "diff-backends: transformers is required for tokenizer loading",
            stderr.getvalue(),
        )

    def test_loader_failure_uses_the_bound_code_function(self):
        """A tokenizer build failure is reported through the fixed code."""
        stderr = io.StringIO()
        with (
            mock.patch.object(doctor, "_build_primary_tokenizer", side_effect=RuntimeError("boom")),
            mock.patch.object(doctor, "_loader_failure_code", return_value="loader_failed"),
            contextlib.redirect_stderr(stderr),
        ):
            code = diffbackends.run_diff_backends(self._args(), argparse.ArgumentParser())
        self.assertEqual(code, 1)
        self.assertEqual(stderr.getvalue(), "diff-backends: loader_failed\n")


if __name__ == "__main__":
    unittest.main()
