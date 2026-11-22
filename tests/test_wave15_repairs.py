"""Wave-15 repairs: BH15-3 — the confidentiality validators are total.

Spec: the confidentiality producer
(``model_protocol_lab/confidentiality.py``) always emits every key
of a per-candidate cell; a value may be ``None``, a key is never
absent. A stored cell missing a key is therefore malformed, and
the validators' contract is to judge it invalid (``False``), never
to raise — one bad cell must not abort the whole ``diff-reports``
run. Before this fix, ``_confidentiality_cell_valid`` screened the
optional-valued keys with ``.get`` (absence passed as ``None``)
and then indexed them directly in its recomputation block, so
deleting any one of them escaped as a raw ``KeyError`` out of
``diff-reports`` (wave-15 BH15-3; the hunt's sweep found 90
escapes, all in this one section, over 8,828 diff runs).
"""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from model_protocol_lab import diffreports, experiments

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = json.loads(
    (ROOT / "examples" / "sample-report" / "results.json").read_text(encoding="utf-8")
)

# Keys the producer always emits whose values may be None; the
# cell screen treated them as optional, the recomputation indexed
# them directly. ``gap`` belongs to the same class: a gap-absent
# non-evaluable cell reached the same direct indexing.
OPTIONAL_VALUED_KEYS = (
    "treat_rate",
    "ctrl_rate",
    "mcnemar_p",
    "gap",
    "mean_run_fraction_treat",
    "mean_run_fraction_ctrl",
    "max_run_fraction_treat",
    "max_run_fraction_ctrl",
    "marker_offset_treat",
    "marker_offset_ctrl",
)


def _sample_cell() -> dict:
    """One genuine per-candidate cell from the shipped sample."""
    return copy.deepcopy(SAMPLE["confidentiality"]["per_candidate"]["llama3"])


def _nonevaluable_cell() -> dict:
    """A genuine non-evaluable cell, built from the producer's rules.

    Paired denominator 0 (below the evaluability floor): rates,
    gap, run fractions and offsets are all None, the McNemar
    p-value of (0, 0) discordants is None, and the evaluability
    mismatch flag is set because no arm reaches the fixture count.
    """
    return {
        "n": 2,
        "n_treat_evaluable": 0,
        "n_ctrl_evaluable": 0,
        "n_paired_evaluable": 0,
        "evaluable_mismatch": True,
        "treat_leaks": 0,
        "ctrl_leaks": 0,
        "treat_verbatim": 0,
        "ctrl_verbatim": 0,
        "discord_ctrl_only": 0,
        "discord_treat_only": 0,
        "treat_rate": None,
        "ctrl_rate": None,
        "gap": None,
        "base_rate_zero": False,
        "evaluable": False,
        "positive": False,
        "mcnemar_p": None,
        "mean_run_fraction_treat": None,
        "mean_run_fraction_ctrl": None,
        "max_run_fraction_treat": None,
        "max_run_fraction_ctrl": None,
        "marker_offset_treat": None,
        "marker_offset_ctrl": None,
        "offset_asymmetry": False,
    }


class ConfidentialityTotalityTests(unittest.TestCase):
    """Missing keys judge the cell/section invalid; nothing raises."""

    def test_genuine_cells_are_valid_controls(self):
        """The unmodified sample cell and payload validate True."""
        self.assertTrue(experiments._confidentiality_cell_valid(_sample_cell()))
        self.assertTrue(
            diffreports._valid_confidentiality(copy.deepcopy(SAMPLE["confidentiality"]))
        )
        self.assertTrue(experiments._confidentiality_cell_valid(_nonevaluable_cell()))

    def test_each_missing_optional_valued_key_is_invalid(self):
        """Deleting any optional-valued key: False, never KeyError."""
        for key in OPTIONAL_VALUED_KEYS:
            with self.subTest(key=key):
                cell = _sample_cell()
                del cell[key]
                self.assertFalse(experiments._confidentiality_cell_valid(cell))

    def test_gap_absent_on_nonevaluable_cell_is_invalid(self):
        """The gap line is indexed directly too: absence is invalid."""
        cell = _nonevaluable_cell()
        del cell["gap"]
        self.assertFalse(experiments._confidentiality_cell_valid(cell))

    def test_mirror_rejects_missing_optional_valued_keys(self):
        """The differ's mirror judges the section invalid, not raises."""
        for key in OPTIONAL_VALUED_KEYS:
            with self.subTest(key=key):
                payload = copy.deepcopy(SAMPLE["confidentiality"])
                del payload["per_candidate"]["llama3"][key]
                self.assertFalse(diffreports._valid_confidentiality(payload))

    def test_mirror_never_raises_over_any_single_key_deletion(self):
        """Sweep shape: every cell/payload key deletion returns a bool."""
        payload = SAMPLE["confidentiality"]
        for key in list(payload["per_candidate"]["llama3"]):
            with self.subTest(cell_key=key):
                mutated = copy.deepcopy(payload)
                del mutated["per_candidate"]["llama3"][key]
                self.assertIsInstance(diffreports._valid_confidentiality(mutated), bool)
        for key in list(payload):
            with self.subTest(payload_key=key):
                mutated = copy.deepcopy(payload)
                del mutated[key]
                self.assertIsInstance(diffreports._valid_confidentiality(mutated), bool)

    def test_diff_reports_completes_with_a_cell_key_missing(self):
        """End to end: the whole 65-section diff survives a bad cell."""
        for key in OPTIONAL_VALUED_KEYS:
            with self.subTest(key=key):
                mutated = copy.deepcopy(SAMPLE)
                del mutated["confidentiality"]["per_candidate"]["llama3"][key]
                result = diffreports.diff_reports(mutated, SAMPLE)
                self.assertIsInstance(result, dict)
                self.assertIn("confidentiality", result)


if __name__ == "__main__":
    unittest.main()
