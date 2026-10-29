"""Cycle 217: pin the CodeQL py/clear-text-logging-sensitive-data
adjudication for the diff-reports sinks (alerts #24/#25).

The flagged flow is the ``confidentiality`` diff section reaching the
stdout/stderr prints. The section's name refers to the behavioral
labeled-block exposure probe (cycle 169), not to secret material:
these tests pin, by execution, that (a) the section's schema admits
only an availability enum, booleans, numeric gaps, and candidate
labels from structurally validated payloads, and (b) input-only
fields of a hostile results file never reach either sink object.
The sinks print the derived diff dict verbatim
(``print(json.dumps(diff, ...))`` / ``print(render_summary(diff))``),
so pinning those two objects pins the sinks. If a future change
starts emitting prompt, completion, or credential material through
this section, these tests fail instead of the finding returning
silently.
"""

from __future__ import annotations

import json
import unittest

from model_protocol_lab import confidentiality as conf
from model_protocol_lab import diffreports


def _payload(**overrides):
    """Build the confidentiality-section payload used by the logging adjudication."""
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
    payload = {
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
    payload.update(overrides)
    return payload


def _leaves(node, out):
    """Yield every leaf value in the nested payload."""
    if isinstance(node, dict):
        for value in node.values():
            _leaves(value, out)
    elif isinstance(node, list):
        for value in node:
            _leaves(value, out)
    else:
        out.append(node)


class LoggingAdjudicationTests(unittest.TestCase):
    """The clear-text-logging alerts are adjudicated false positives, with executed proof."""

    def test_confidentiality_section_schema_is_aggregates_only(self):
        """Section leaves are enum/bool/number/label only."""
        old, new = _payload(), _payload()
        new["per_candidate"] = {
            "cand": dict(
                new["per_candidate"]["cand"],
                gap=0.25,
                ctrl_rate=0.25,
                ctrl_leaks=3,
                discord_ctrl_only=3,
                mcnemar_p=0.25,
            )
        }
        diff = diffreports.diff_reports(
            {"decision": "abstain", "confidentiality": old},
            {"decision": "abstain", "confidentiality": new},
        )
        section = diff["confidentiality"]
        self.assertEqual(section["available"], "both")
        leaves = []
        _leaves(section, leaves)
        for leaf in leaves:
            self.assertIsInstance(leaf, (str, bool, float, int, type(None)))
        strings = [leaf for leaf in leaves if isinstance(leaf, str)]
        self.assertEqual(sorted(set(strings)), ["both", "cand"])

    def test_input_only_fields_never_reach_sink_objects(self):
        """Extra input fields (incl. secret-shaped ones) are not printed."""
        old = _payload()
        old["prompt_text"] = "TOP-SECRET operator prompt, do not print"
        old["api_key"] = "sk-live-should-never-appear"
        old["per_candidate"]["cand"]["hidden_note"] = "credential hunter"
        diff = diffreports.diff_reports(
            {"decision": "abstain", "confidentiality": old},
            {"decision": "abstain", "confidentiality": _payload()},
        )
        printed = json.dumps(diff, indent=2, ensure_ascii=True, sort_keys=True)
        summary = diffreports.render_summary(diff)
        for marker in (
            "TOP-SECRET",
            "sk-live-should-never-appear",
            "credential hunter",
            "hidden_note",
            "prompt_text",
            "api_key",
        ):
            self.assertNotIn(marker, printed)
            self.assertNotIn(marker, summary)


if __name__ == "__main__":
    unittest.main()
