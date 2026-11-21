"""W7-0 facade-surface pin: the consumed import surface of the
three monoliths, measured by AST census at baseline 4cbcfc1d
(W7 re-census lane A, consumers.json).

Every name any file in the package, tests, tools or examples
imports from model_protocol_lab.diffreports / .experiments /
.reporting — including underscore-private names the suite
depends on — must resolve in that module's own namespace.
The W7-2 splits move code into family modules behind facades;
this pin is the compatibility contract those facades keep.
A name dropped from a facade fails here by name, before the
consumers that import it fail at collection time.
"""

from __future__ import annotations

import importlib
import unittest

SURFACE: dict[str, tuple[str, ...]] = {
    "diffreports": (
        "MAX_RESULTS_BYTES",
        "_config_snapshot",
        "_dateline_classify",
        "_dateline_mcnemar_p",
        "_diff_spans",
        "_leading_classify",
        "_leading_mcnemar_p",
        "_normalize_attribution",
        "_placement_generate",
        "_recordnotation_generate",
        "_recordnotation_parsed_hit",
        "_render_healing_summary",
        "_render_hierarchy_summary",
        "_render_structured_summary",
        "_render_synthhistory_summary",
        "_render_template_robustness_summary",
        "_render_ws_canonical_summary",
        "_sha256_regular_file",
        "_step_function",
        "_valid_aliasing",
        "_valid_confidence",
        "_valid_dateline",
        "_valid_density",
        "_valid_distraction",
        "_valid_field_order",
        "_valid_frontier",
        "_valid_leading",
        "_valid_multiturn",
        "_valid_needle",
        "_valid_numform",
        "_valid_placement",
        "_valid_precision",
        "_valid_primary_pair",
        "_valid_repetition",
        "_valid_rotation",
        "_valid_scorerlens",
        "_valid_structured",
        "_valid_template_robustness",
        "_valid_token_healing",
        "_valid_ws_canonical",
        "diff_aliasing",
        "diff_boundary_audit",
        "diff_confidence",
        "diff_dateline",
        "diff_density",
        "diff_distraction",
        "diff_field_order",
        "diff_frontiers",
        "diff_held_out_floor",
        "diff_hierarchy",
        "diff_leading",
        "diff_multiturn",
        "diff_needle",
        "diff_numform",
        "diff_placement",
        "diff_position",
        "diff_precision",
        "diff_prefill",
        "diff_reports",
        "diff_rotation",
        "diff_scorerlens",
        "diff_stop_cause",
        "diff_structured",
        "diff_surface_noise",
        "diff_synthhistory",
        "diff_template_robustness",
        "diff_token_healing",
        "diff_ws_canonical",
        "load_results",
        "parse_results_bytes",
        "render_summary",
        "report_pair_incomplete",
        "run_diff_reports",
    ),
    "experiments": (
        "FAILURE_MODES",
        "LOGPROB_DERAILED_MEAN_THRESHOLD",
        "LOGPROB_UNKNOWN_MEAN_THRESHOLD",
        "STOP_SEMANTICS",
        "_aggregate_fixture_scores",
        "_answer_collision_cell_valid",
        "_backend_fingerprint",
        "_confidentiality_cell_valid",
        "_diagnostic_observation",
        "_interpret_logprob_candidate",
        "_observation_min_budget",
        "_resolve_boundaries",
        "_score_expected_answer",
        "_select_winner",
        "_stop_cell",
        "_structured_cell_valid",
        "_summary",
        "_synthhistory_entry_valid",
        "_verbosity_cell_valid",
        "ablation_stage",
        "aliasing_profile",
        "answer_collision_profile",
        "answer_collision_stage",
        "batch_stage",
        "budget_frontier",
        "cache_profile_profile",
        "cache_profile_stage",
        "candidate_fingerprint",
        "canonicalform_analysis",
        "canonicalform_stage",
        "choiceaudit_analysis",
        "choiceaudit_stage",
        "classify_observation",
        "compatibility_form_analysis",
        "compatibility_form_stage",
        "confidence_stage",
        "confidentiality_profile",
        "confidentiality_stage",
        "configuration_digest",
        "confusable_skeleton_analysis",
        "confusable_skeleton_stage",
        "dateline_analysis",
        "dateline_stage",
        "deliforge_authority",
        "deliforge_stage",
        "density_analysis",
        "density_stage",
        "distraction_analysis",
        "distraction_stage",
        "enumeration_convention_analysis",
        "enumeration_convention_stage",
        "failure_taxonomy",
        "faketurn_robustness",
        "faketurn_stage",
        "fewshot_gain",
        "fewshot_stage",
        "fieldorder_analysis",
        "fieldorder_stage",
        "healing_delta",
        "healing_stage",
        "held_out_budget_floor",
        "hierarchy_adherence",
        "hierarchy_stage",
        "history_frame_analysis",
        "history_frame_stage",
        "instruction_distance_analysis",
        "instruction_distance_stage",
        "invisible_chars_analysis",
        "invisible_chars_stage",
        "leading_analysis",
        "leading_stage",
        "logprob_diagnostic",
        "multiturn_analysis",
        "multiturn_stage",
        "native_search_input_ids",
        "native_search_outcomes",
        "needle_analysis",
        "needle_stage",
        "negation_analysis",
        "negation_stage",
        "numform_analysis",
        "numform_stage",
        "perturbation_robustness",
        "perturbation_stage",
        "placement_analysis",
        "placement_stage",
        "portfolio_analysis",
        "position_sensitivity",
        "position_stage",
        "precision_stage",
        "prefill_adherence",
        "prefill_stage",
        "prompt_sensitivity_sweep",
        "prompt_token_depth_sweep",
        "prune_sweep_analysis",
        "prune_sweep_stage",
        "record_notation_analysis",
        "record_notation_stage",
        "refusal_profile",
        "repetition_analysis",
        "repetition_stage",
        "rotation_analysis",
        "rotation_stage",
        "run_experiment",
        "scorerlens_profile",
        "selection_robustness",
        "serving_backends_analysis",
        "stop_matrix",
        "stop_stage",
        "structured_adherence",
        "structured_stage",
        "synthhistory_stability",
        "synthhistory_stage",
        "template_robustness_stage",
        "toolcall_adherence",
        "toolcall_stage",
        "transfer_portability",
        "transfer_stage",
        "typographic_emphasis_analysis",
        "typographic_emphasis_stage",
        "verbosity_profile",
        "verbosity_stage",
        "ws_canonicalization",
        "ws_stage",
    ),
    "reporting": (
        "MAX_FRONTIER_STEP_POINTS",
        "_CSP",
        "_SCRIPT",
        "_STOPCAUSE_CAUSES",
        "_WITHHELD_DIAGNOSTIC_DETAIL",
        "_config_section",
        "_contamination_line",
        "_decisive_section",
        "_deliforge_section",
        "_failure_diagnosis",
        "_faketurn_section",
        "_fewshot_section",
        "_fixture_total",
        "_frontier_chart_svg",
        "_frontier_section",
        "_frontier_series",
        "_frontier_spark_table",
        "_held_out_floor_callout",
        "_held_out_section",
        "_hierarchy_section",
        "_margin_chart_svg",
        "_outcome_grid_section",
        "_perturbation_section",
        "_portfolio_section",
        "_position_section",
        "_prefill_section",
        "_public_view",
        "_search_rows",
        "_selection_robustness_section",
        "_spark_svg",
        "_sparkline",
        "_stopcause_section",
        "_structured_section",
        "_synthhistory_section",
        "_template_robustness_section",
        "_token_healing_section",
        "_transfer_section",
        "_wscanonical_section",
        "attach_diagnostics",
        "margin_trajectory",
        "prepare_report_dir",
        "render_html",
        "safe_budget_floor",
        "save_report",
        "selection_margin",
        "token_overhead",
    ),
}


def missing_names(module: object, names: tuple[str, ...]) -> list[str]:
    """Names from ``names`` absent from the module's own namespace."""
    namespace = vars(module)
    return [name for name in names if name not in namespace]


class FacadeSurfaceTests(unittest.TestCase):
    """Every consumed name resolves from its original module."""

    def test_diffreports_surface(self) -> None:
        module = importlib.import_module("model_protocol_lab.diffreports")
        self.assertEqual(missing_names(module, SURFACE["diffreports"]), [])

    def test_experiments_surface(self) -> None:
        module = importlib.import_module("model_protocol_lab.experiments")
        self.assertEqual(missing_names(module, SURFACE["experiments"]), [])

    def test_reporting_surface(self) -> None:
        module = importlib.import_module("model_protocol_lab.reporting")
        self.assertEqual(missing_names(module, SURFACE["reporting"]), [])

    def test_surface_totals(self) -> None:
        # 73 + 124 + 46 = 243 distinct consumed names at the
        # W7 baseline (lane A re-census, consumers.json).
        self.assertEqual(
            {name: len(names) for name, names in SURFACE.items()},
            {"diffreports": 73, "experiments": 124, "reporting": 46},
        )


class FacadeCheckerSelfTests(unittest.TestCase):
    """The checker itself can fail (the by-hand mutant check, mechanised).

    Dropping a name must be reported; a bogus name must be
    reported; a present name must not be.
    """

    def test_missing_name_is_reported(self) -> None:
        module = importlib.import_module("model_protocol_lab.diffreports")
        self.assertEqual(
            missing_names(module, ("definitely_not_exported_w7",)),
            ["definitely_not_exported_w7"],
        )

    def test_present_name_is_not_reported(self) -> None:
        module = importlib.import_module("model_protocol_lab.diffreports")
        self.assertEqual(missing_names(module, ("diff_reports",)), [])


if __name__ == "__main__":
    unittest.main()
