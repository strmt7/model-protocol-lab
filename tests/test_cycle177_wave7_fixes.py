"""Cycle 177: wave-7 F6b/F6c -- cache-profile and confidentiality
validators recompute their derived values.

Wave-7 red-team A showed both validators accepted internally
impossible payloads (fractions set to 1.0/0.99 with the counts
unchanged, a mean of 999, lying rates whose stored gap stayed
consistent, a McNemar p-value unrelated to the discordant counts,
profile extremes naming candidates that do not exist). The
recompute standard is the answer-collision validator's: every
value that is a deterministic function of stored counts is
recomputed and compared. Each test below breaks exactly one
identity in an otherwise genuine payload.
"""

from __future__ import annotations

import json
import unittest
from typing import Any

from model_protocol_lab import cacheprofile, diffreports, experiments
from model_protocol_lab import confidentiality as conf


def _cache_profile() -> dict[str, Any]:
    """Cache profile."""
    prompts_a = [(1, 2, 3, 4), (1, 2, 3, 5), (1, 2, 9, 9)]
    prompts_b = [(7, 7, 7, 7), (7, 7, 7, 8)]
    stage = {
        "n_observations": 5,
        "block_size": 2,
        "per_candidate": {
            "a": cacheprofile.candidate_cache_profile(prompts_a, 2),
            "b": cacheprofile.candidate_cache_profile(prompts_b, 2),
        },
    }
    stage["per_candidate"]["a"]["n"] = 3
    stage["per_candidate"]["b"]["n"] = 2
    return experiments.cache_profile_profile(("a", "b"), stage)


def _conf_cell(**overrides: Any) -> dict[str, Any]:
    """Conf cell."""
    cell: dict[str, Any] = {
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
    cell.update(overrides)
    return cell


def _conf_profile(**cell_overrides: Any) -> dict[str, Any]:
    """Conf profile."""
    return {
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
        "per_candidate": {"cand": _conf_cell(**cell_overrides)},
    }


class CacheRecomputeTests(unittest.TestCase):
    """Cache recompute tests."""

    def test_genuine_profile_valid(self) -> None:
        """Test genuine profile valid."""
        self.assertTrue(diffreports._valid_cache_profile(_cache_profile()))

    @staticmethod
    def _broken(mutate: Any) -> dict[str, Any]:
        """Broken."""
        payload = json.loads(json.dumps(_cache_profile()))
        mutate(payload)
        return payload

    def test_fraction_lie_rejected(self) -> None:
        """Test fraction lie rejected."""

        def mutate(p: dict[str, Any]) -> None:
            """Mutate."""
            p["per_candidate"]["a"]["potential_token_fraction"] = 0.99

        self.assertFalse(diffreports._valid_cache_profile(self._broken(mutate)))

    def test_mean_lie_rejected(self) -> None:
        """Test mean lie rejected."""

        def mutate(p: dict[str, Any]) -> None:
            """Mutate."""
            p["per_candidate"]["a"]["mean_prompt_tokens"] = 999.0

        self.assertFalse(diffreports._valid_cache_profile(self._broken(mutate)))

    def test_block_fraction_lie_rejected(self) -> None:
        """Test block fraction lie rejected."""

        def mutate(p: dict[str, Any]) -> None:
            """Mutate."""
            p["per_candidate"]["a"]["achievable_block_fraction"] = 0.99

        self.assertFalse(diffreports._valid_cache_profile(self._broken(mutate)))

    def test_gap_lie_rejected(self) -> None:
        """Test gap lie rejected."""

        def mutate(p: dict[str, Any]) -> None:
            """Mutate."""
            p["sharability_gap"] = 0.87

        self.assertFalse(diffreports._valid_cache_profile(self._broken(mutate)))

    def test_ghost_extreme_rejected(self) -> None:
        """Test ghost extreme rejected."""

        def mutate(p: dict[str, Any]) -> None:
            """Mutate."""
            p["most_sharable"] = ["ghost"]

        self.assertFalse(diffreports._valid_cache_profile(self._broken(mutate)))

    def test_excluded_lie_rejected(self) -> None:
        """Test excluded lie rejected."""

        def mutate(p: dict[str, Any]) -> None:
            """Mutate."""
            p["excluded_candidates"] = ["a"]

        self.assertFalse(diffreports._valid_cache_profile(self._broken(mutate)))

    def test_cell_validator_block_recompute_needs_block_size(self) -> None:
        """Test cell validator block recompute needs block size."""
        cell = json.loads(json.dumps(_cache_profile()))["per_candidate"]["a"]
        self.assertTrue(experiments._cache_profile_cell_valid(cell, 2))
        cell["achievable_block_fraction"] = 0.99
        # Without the stage's block size the block pair cannot be
        # recomputed (documented contract); with it, the lie fails.
        self.assertFalse(experiments._cache_profile_cell_valid(cell, 2))


class ConfidentialityCellRecomputeTests(unittest.TestCase):
    """Confidentiality cell recompute tests."""

    def test_genuine_cell_valid(self) -> None:
        """Test genuine cell valid."""
        self.assertTrue(experiments._confidentiality_cell_valid(_conf_cell()))

    @staticmethod
    def _broken(**overrides: Any) -> dict[str, Any]:
        """Broken."""
        return _conf_cell(**overrides)

    def test_rate_lie_rejected(self) -> None:
        """Test rate lie rejected."""
        # Counts 3/12 and 6/12, stored rates 0.0/0.25, gap kept
        # consistent with the lying rates (A's exact repro shape).
        # Every other identity is satisfied, so only the rate
        # recompute can reject this cell.
        cell = self._broken(
            treat_leaks=3,
            ctrl_leaks=6,
            discord_ctrl_only=3,
            treat_rate=0.0,
            ctrl_rate=0.25,
            gap=0.25,
            mcnemar_p=0.25,
            mean_run_fraction_treat=0.5,
            max_run_fraction_treat=1.0,
            mean_run_fraction_ctrl=0.5,
            treat_verbatim=1,
        )
        self.assertFalse(experiments._confidentiality_cell_valid(cell))

    def test_mcnemar_lie_rejected(self) -> None:
        """Test mcnemar lie rejected."""
        cell = self._broken(mcnemar_p=0.999)
        self.assertFalse(experiments._confidentiality_cell_valid(cell))

    def test_positive_with_zero_gap_rejected(self) -> None:
        """Test positive with zero gap rejected."""
        cell = self._broken(
            treat_leaks=2,
            discord_ctrl_only=0,
            treat_rate=2 / 12,
            gap=0.0,
            mcnemar_p=None,
            positive=True,
            mean_run_fraction_treat=0.25,
            max_run_fraction_treat=1.0,
            treat_verbatim=1,
        )
        self.assertFalse(experiments._confidentiality_cell_valid(cell))

    def test_run_fraction_lie_rejected(self) -> None:
        """Test run fraction lie rejected."""
        # Consistent everywhere except the run fractions: 3 treat
        # leaks cannot coexist with a max run fraction of 0.0.
        # Discordants (ctrl 1, treat 2) -> both-leak count 1;
        # McNemar (1, 2): n=3, k=1 -> min(1, 2*(1+3)/8) = 1.0.
        cell = self._broken(
            treat_leaks=3,
            ctrl_leaks=2,
            discord_ctrl_only=1,
            discord_treat_only=2,
            treat_rate=3 / 12,
            ctrl_rate=2 / 12,
            gap=2 / 12 - 3 / 12,
            positive=False,
            mcnemar_p=1.0,
            mean_run_fraction_treat=0.0,
            max_run_fraction_treat=0.0,
        )
        self.assertFalse(experiments._confidentiality_cell_valid(cell))


class ConfidentialityProfileRecomputeTests(unittest.TestCase):
    """Confidentiality profile recompute tests."""

    def test_genuine_profile_valid(self) -> None:
        """Test genuine profile valid."""
        self.assertTrue(diffreports._valid_confidentiality(_conf_profile()))

    @staticmethod
    def _broken(mutate: Any) -> dict[str, Any]:
        """Broken."""
        payload = json.loads(json.dumps(_conf_profile()))
        mutate(payload)
        return payload

    def test_uniform_lie_rejected(self) -> None:
        """Test uniform lie rejected."""

        def mutate(p: dict[str, Any]) -> None:
            """Mutate."""
            p["uniform_nonpositive"] = True

        self.assertFalse(diffreports._valid_confidentiality(self._broken(mutate)))

    def test_positive_set_lie_rejected(self) -> None:
        """Test positive set lie rejected."""

        def mutate(p: dict[str, Any]) -> None:
            """Mutate."""
            p["positive_candidates"] = []

        self.assertFalse(diffreports._valid_confidentiality(self._broken(mutate)))

    def test_ghost_extreme_rejected(self) -> None:
        """Test ghost extreme rejected."""

        def mutate(p: dict[str, Any]) -> None:
            """Mutate."""
            p["most_resistant"] = "ghost"

        self.assertFalse(diffreports._valid_confidentiality(self._broken(mutate)))

    def test_measurable_count_lie_rejected(self) -> None:
        """Test measurable count lie rejected."""

        def mutate(p: dict[str, Any]) -> None:
            """Mutate."""
            p["n_measurable"] = 0

        self.assertFalse(diffreports._valid_confidentiality(self._broken(mutate)))


class ConfidentialityCellGapBatteryTests(unittest.TestCase):
    """One broken identity per recompute branch (coverage-driven:
    each variant keeps every other identity consistent).
    """

    def test_paired_above_arm_evaluable_rejected(self) -> None:
        """Test paired above arm evaluable rejected."""
        cell = _conf_cell(n_treat_evaluable=11, evaluable_mismatch=True)
        self.assertFalse(experiments._confidentiality_cell_valid(cell))

    def test_arm_evaluable_above_n_rejected(self) -> None:
        """Test arm evaluable above n rejected."""
        cell = _conf_cell(n_treat_evaluable=13, evaluable_mismatch=True)
        self.assertFalse(experiments._confidentiality_cell_valid(cell))

    def test_mismatch_flag_lie_rejected(self) -> None:
        """Test mismatch flag lie rejected."""
        cell = _conf_cell(evaluable_mismatch=True)
        self.assertFalse(experiments._confidentiality_cell_valid(cell))

    def test_evaluable_flag_lie_rejected(self) -> None:
        """Test evaluable flag lie rejected."""
        cell = _conf_cell(gap=None, evaluable=False, positive=False)
        self.assertFalse(experiments._confidentiality_cell_valid(cell))

    def test_base_rate_zero_lie_rejected(self) -> None:
        """Test base rate zero lie rejected."""
        cell = _conf_cell(gap=None, evaluable=False, positive=False, base_rate_zero=True)
        self.assertFalse(experiments._confidentiality_cell_valid(cell))

    def test_zero_paired_with_rate_present_rejected(self) -> None:
        """Test zero paired with rate present rejected."""
        cell = _conf_cell(
            n_treat_evaluable=0,
            n_ctrl_evaluable=0,
            n_paired_evaluable=0,
            evaluable_mismatch=True,
            treat_leaks=0,
            ctrl_leaks=0,
            treat_verbatim=0,
            ctrl_verbatim=0,
            discord_ctrl_only=0,
            treat_rate=0.0,
            ctrl_rate=None,
            gap=None,
            evaluable=False,
            positive=False,
            mcnemar_p=None,
            mean_run_fraction_treat=None,
            mean_run_fraction_ctrl=None,
            max_run_fraction_treat=None,
            max_run_fraction_ctrl=None,
            marker_offset_treat=None,
            marker_offset_ctrl=None,
        )
        self.assertFalse(experiments._confidentiality_cell_valid(cell))

    def test_zero_paired_with_run_fraction_present_rejected(self) -> None:
        """Test zero paired with run fraction present rejected."""
        cell = _conf_cell(
            n_treat_evaluable=0,
            n_ctrl_evaluable=0,
            n_paired_evaluable=0,
            evaluable_mismatch=True,
            treat_leaks=0,
            ctrl_leaks=0,
            treat_verbatim=0,
            ctrl_verbatim=0,
            discord_ctrl_only=0,
            treat_rate=None,
            ctrl_rate=None,
            gap=None,
            evaluable=False,
            positive=False,
            mcnemar_p=None,
            mean_run_fraction_treat=0.0,
            mean_run_fraction_ctrl=None,
            max_run_fraction_treat=None,
            max_run_fraction_ctrl=None,
            marker_offset_treat=None,
            marker_offset_ctrl=None,
        )
        self.assertFalse(experiments._confidentiality_cell_valid(cell))

    def test_discord_identity_break_rejected(self) -> None:
        """Test discord identity break rejected."""
        # McNemar (1, 0) = 1.0; positive needs >= 2 ctrl discordants.
        cell = _conf_cell(discord_ctrl_only=1, mcnemar_p=1.0, positive=False)
        self.assertFalse(experiments._confidentiality_cell_valid(cell))

    def test_discord_above_leaks_rejected(self) -> None:
        """Test discord above leaks rejected."""
        # Identity holds (both-leak = 1); the ctrl bound breaks.
        # McNemar (3, 1) = 0.625.
        cell = _conf_cell(discord_ctrl_only=3, discord_treat_only=1, mcnemar_p=0.625)
        self.assertFalse(experiments._confidentiality_cell_valid(cell))

    def test_run_fraction_missing_when_paired_rejected(self) -> None:
        """Test run fraction missing when paired rejected."""
        cell = _conf_cell(max_run_fraction_ctrl=None)
        self.assertFalse(experiments._confidentiality_cell_valid(cell))

    def test_max_run_above_threshold_without_leaks_rejected(self) -> None:
        """Test max run above threshold without leaks rejected."""
        cell = _conf_cell(max_run_fraction_treat=0.75)
        self.assertFalse(experiments._confidentiality_cell_valid(cell))

    def test_mean_run_below_threshold_bound_rejected(self) -> None:
        """Test mean run below threshold bound rejected."""
        cell = _conf_cell(mean_run_fraction_ctrl=0.1)
        self.assertFalse(experiments._confidentiality_cell_valid(cell))

    def test_verbatim_without_full_max_rejected(self) -> None:
        """Test verbatim without full max rejected."""
        cell = _conf_cell(max_run_fraction_ctrl=0.75)
        self.assertFalse(experiments._confidentiality_cell_valid(cell))

    def test_offset_asymmetry_lie_rejected(self) -> None:
        """Test offset asymmetry lie rejected."""
        cell = _conf_cell(marker_offset_ctrl=29.0, offset_asymmetry=True)
        self.assertFalse(experiments._confidentiality_cell_valid(cell))

    def test_offset_asymmetry_without_both_offsets_rejected(self) -> None:
        """Test offset asymmetry without both offsets rejected."""
        cell = _conf_cell(marker_offset_ctrl=None, offset_asymmetry=True)
        self.assertFalse(experiments._confidentiality_cell_valid(cell))


class CacheGapBatteryTests(unittest.TestCase):
    """The experiments-side cell recompute branches, via the
    profile assembler (stage_invalid) and the direct validator.
    """

    @staticmethod
    def _cell() -> dict[str, Any]:
        """Cell."""
        profile = _cache_profile()
        return json.loads(json.dumps(profile))["per_candidate"]["a"]

    def test_token_fraction_lies_rejected(self) -> None:
        """Test token fraction lies rejected."""
        cell = self._cell()
        cell["potential_token_fraction"] = 0.99
        self.assertFalse(experiments._cache_profile_cell_valid(cell, 2))
        cell = self._cell()
        cell["achievable_token_fraction"] = 0.99
        self.assertFalse(experiments._cache_profile_cell_valid(cell, 2))

    def test_block_fraction_lies_rejected(self) -> None:
        """Test block fraction lies rejected."""
        cell = self._cell()
        cell["potential_block_fraction"] = 0.99
        self.assertFalse(experiments._cache_profile_cell_valid(cell, 2))
        cell = self._cell()
        cell["achievable_block_fraction"] = 0.99
        self.assertFalse(experiments._cache_profile_cell_valid(cell, 2))

    def test_zero_total_with_evaluable_rejected(self) -> None:
        """Test zero total with evaluable rejected."""
        cell = self._cell()
        cell["total_tokens"] = 0
        cell["cacheable_tokens"] = 0
        cell["achievable_tokens"] = 0
        cell["shared_prefixes"] = 0
        self.assertFalse(experiments._cache_profile_cell_valid(cell, 2))

    def test_stage_rejects_lying_cell(self) -> None:
        """Test stage rejects lying cell."""
        stage = {
            "n_observations": 3,
            "block_size": 2,
            "per_candidate": {"a": self._cell()},
        }
        stage["per_candidate"]["a"]["mean_prompt_tokens"] = 999.0
        profile = experiments.cache_profile_profile(("a",), stage)
        self.assertEqual(profile["detail"], "stage_invalid")


class DiffProfileGapBatteryTests(unittest.TestCase):
    """Profile-level branches in both diff validators."""

    def test_cache_achievable_token_lie_rejected(self) -> None:
        """Test cache achievable token lie rejected."""
        payload = json.loads(json.dumps(_cache_profile()))
        payload["per_candidate"]["a"]["achievable_token_fraction"] = 0.99
        self.assertFalse(diffreports._valid_cache_profile(payload))

    def test_cache_mismatch_lie_rejected(self) -> None:
        """Test cache mismatch lie rejected."""
        payload = json.loads(json.dumps(_cache_profile()))
        self.assertTrue(payload["evaluable_mismatch"])
        payload["evaluable_mismatch"] = False
        self.assertFalse(diffreports._valid_cache_profile(payload))

    def test_cache_uniform_lie_rejected(self) -> None:
        """Test cache uniform lie rejected."""
        payload = json.loads(json.dumps(_cache_profile()))
        payload["uniform"] = True
        self.assertFalse(diffreports._valid_cache_profile(payload))

    def test_conf_n_candidates_lie_rejected(self) -> None:
        """Test conf n candidates lie rejected."""
        payload = json.loads(json.dumps(_conf_profile()))
        payload["n_candidates"] = 2
        self.assertFalse(diffreports._valid_confidentiality(payload))

    def test_conf_n_cells_lie_rejected(self) -> None:
        """Test conf n cells lie rejected."""
        payload = json.loads(json.dumps(_conf_profile()))
        payload["n_cells"] = 0
        self.assertFalse(diffreports._valid_confidentiality(payload))

    def test_conf_least_ghost_rejected(self) -> None:
        """Test conf least ghost rejected."""
        payload = json.loads(json.dumps(_conf_profile()))
        payload["least_resistant"] = "ghost"
        self.assertFalse(diffreports._valid_confidentiality(payload))

    def test_conf_profile_gap_lie_rejected(self) -> None:
        """Test conf profile gap lie rejected."""
        payload = json.loads(json.dumps(_conf_profile()))
        payload["gap"] = 0.5
        self.assertFalse(diffreports._valid_confidentiality(payload))

    def test_conf_any_positive_lie_rejected(self) -> None:
        """Test conf any positive lie rejected."""
        payload = json.loads(json.dumps(_conf_profile()))
        payload["any_positive"] = False
        self.assertFalse(diffreports._valid_confidentiality(payload))


if __name__ == "__main__":
    unittest.main()
