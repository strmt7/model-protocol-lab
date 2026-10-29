"""Cycle 169 coverage-gap tests: fail-closed validator branches.

Each test here exists to execute a defensive branch the main suites
leave cold (verified via the full-run coverage.json): hostile payload
rejections, degenerate denominators, and the never-raised wrappers'
edge paths.
"""

from __future__ import annotations

import unittest

from model_protocol_lab import confidentiality as conf
from model_protocol_lab import diffreports, experiments, reporting


def _valid_cell(**overrides):
    base = {
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
    base.update(overrides)
    return base


def _valid_payload(cell=None, **overrides):
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
        "per_candidate": {"cand": cell if cell is not None else _valid_cell()},
    }
    payload.update(overrides)
    return payload


class ConfidentialityGapTests(unittest.TestCase):
    def test_build_arm_fixtures_bad_arm(self):
        with self.assertRaises(ValueError):
            conf.build_arm_fixtures("sideways", conf.CONFIDENTIALITY_SPECS)

    def test_normalize_tokens_non_string(self):
        with self.assertRaises(ValueError):
            conf.normalize_tokens(123)


class StageValidatorGapTests(unittest.TestCase):
    def test_negative_n_rejected(self):
        self.assertFalse(experiments._confidentiality_cell_valid(_valid_cell(n=-1)))

    def test_out_of_range_rate_rejected(self):
        self.assertFalse(experiments._confidentiality_cell_valid(_valid_cell(treat_rate=1.5)))

    def test_bool_rate_rejected(self):
        self.assertFalse(experiments._confidentiality_cell_valid(_valid_cell(treat_rate=True)))

    def test_gap_out_of_range_rejected(self):
        self.assertFalse(experiments._confidentiality_cell_valid(_valid_cell(gap=2.0)))

    def test_negative_mean_run_rejected(self):
        self.assertFalse(
            experiments._confidentiality_cell_valid(_valid_cell(mean_run_fraction_treat=-0.5))
        )

    def test_bool_max_run_rejected(self):
        self.assertFalse(
            experiments._confidentiality_cell_valid(_valid_cell(max_run_fraction_treat=True))
        )

    def test_non_number_offset_rejected(self):
        self.assertFalse(
            experiments._confidentiality_cell_valid(_valid_cell(marker_offset_treat="x"))
        )

    def test_non_bool_flag_rejected(self):
        self.assertFalse(experiments._confidentiality_cell_valid(_valid_cell(base_rate_zero=1)))

    def test_gap_none_but_evaluable_and_base_rate_rejected(self):
        cell = _valid_cell(gap=None, base_rate_zero=False, evaluable=True)
        self.assertFalse(experiments._confidentiality_cell_valid(cell))

    def test_profile_invalid_candidates(self):
        self.assertEqual(
            experiments.confidentiality_profile((), {})["detail"], "invalid_candidates"
        )

    def test_profile_stage_not_dict(self):
        self.assertEqual(
            experiments.confidentiality_profile(("a",), "junk")["detail"], "stage_invalid"
        )

    def test_profile_protocol_not_dict(self):
        self.assertEqual(
            experiments.confidentiality_profile(("a",), {"protocol": 7})["detail"],
            "stage_invalid",
        )

    def test_cell_not_dict(self):
        self.assertFalse(experiments._confidentiality_cell_valid("junk"))

    def test_profile_missing_n_fixtures(self):
        stage = {"protocol": {}, "per_candidate": {}}
        self.assertEqual(
            experiments.confidentiality_profile(("a",), stage)["detail"], "stage_invalid"
        )

    def test_profile_per_candidate_not_dict(self):
        stage = {"protocol": {}, "n_fixtures": 12, "per_candidate": "junk"}
        self.assertEqual(
            experiments.confidentiality_profile(("a",), stage)["detail"], "stage_invalid"
        )

    def test_profile_invalid_entry(self):
        stage = {"protocol": {}, "n_fixtures": 12, "per_candidate": {"a": "junk"}}
        self.assertEqual(
            experiments.confidentiality_profile(("a",), stage)["detail"], "stage_invalid"
        )


class DiffValidatorGapTests(unittest.TestCase):
    def test_positive_candidates_not_list_rejected(self):
        self.assertFalse(
            diffreports._valid_confidentiality(_valid_payload(positive_candidates="x"))
        )

    def test_most_resistant_not_str_rejected(self):
        self.assertFalse(diffreports._valid_confidentiality(_valid_payload(most_resistant=7)))

    def test_profile_gap_none_rejected(self):
        # Cycle 177 (F6c): the profile gap is recomputed as the
        # measurable cells' spread, so a None gap on a measurable
        # profile is an inconsistency, not a defensive shape. The
        # validator still must not crash on it -- it returns False.
        self.assertFalse(diffreports._valid_confidentiality(_valid_payload(gap=None)))


class TemplateRobustnessValidatorGapTests(unittest.TestCase):
    @staticmethod
    def _payload(old_cand, new_cand):
        def _side(cand):
            return {
                "available": True,
                "candidates": {"alpha": cand},
                "protocol": {"backend_fingerprint": "x"},
            }

        return _side(old_cand), _side(new_cand)

    def test_non_dict_report_rejected(self):
        old, new = self._payload("junk", {"status": "unavailable"})
        self.assertFalse(diffreports._valid_template_robustness(old))
        self.assertTrue(diffreports._valid_template_robustness(new))

    def test_bad_preservation_rate_rejected(self):
        old, _ = self._payload(
            {
                "status": "ok",
                "preservation_rate": 2.0,
                "n_preserved": 0,
                "n_preservation_denominator": 0,
                "n_unevaluable": 0,
            },
            None,
        )
        self.assertFalse(diffreports._valid_template_robustness(old))

    def test_non_int_counts_rejected(self):
        old, _ = self._payload(
            {
                "status": "ok",
                "preservation_rate": 0.5,
                "n_preserved": "0",
                "n_preservation_denominator": 0,
                "n_unevaluable": 0,
            },
            None,
        )
        self.assertFalse(diffreports._valid_template_robustness(old))

    def test_non_dict_candidates_rejected(self):
        payload = {"available": True, "candidates": "junk", "protocol": {}}
        self.assertFalse(diffreports._valid_template_robustness(payload))

    def test_diff_with_unavailable_marker_candidate(self):
        # Exercises the non-ok view branch in diff_template_robustness.
        old, new = self._payload(
            {"status": "unavailable", "reason": "candidate_probe_failed"},
            {"status": "unavailable", "reason": "candidate_probe_failed"},
        )
        diff = diffreports.diff_template_robustness(old, new)
        self.assertEqual(diff["available"], "both")
        self.assertFalse(diff["changed"])


class ReportingGapTests(unittest.TestCase):
    def test_section_per_candidate_not_dict(self):
        html = reporting._confidentiality_section(
            {"confidentiality": {"available": True, "per_candidate": 7}}
        )
        self.assertEqual(html, "")

    def test_section_none_mcnemar_p(self):
        payload = _valid_payload()
        payload["per_candidate"]["cand"]["mcnemar_p"] = None
        html = reporting._confidentiality_section({"confidentiality": payload})
        self.assertIn("n/a", html)

    def test_section_none_rates_render_na(self):
        payload = _valid_payload()
        cell = payload["per_candidate"]["cand"]
        cell.update(
            {
                "n_paired_evaluable": 0,
                "treat_leaks": 0,
                "ctrl_leaks": 0,
                "treat_rate": None,
                "ctrl_rate": None,
                "gap": None,
            }
        )
        html = reporting._confidentiality_section({"confidentiality": payload})
        self.assertIn("n/a", html)

    def test_section_uniform_nonpositive_line(self):
        payload = _valid_payload(uniform_nonpositive=True)
        html = reporting._confidentiality_section({"confidentiality": payload})
        self.assertIn("Null result", html)


if __name__ == "__main__":
    unittest.main()
