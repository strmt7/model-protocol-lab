"""Cycle 168: prefix-cache sharability profile (research wave-6 idea 2).

Tests the pure ``cacheprofile`` module (trie prefix counts, exact
achievable sharability at token and block level, the fail-closed
evaluability predicate, brute-force simulation agreement over all
request orders), ``experiments.cache_profile_stage`` / ``cache_profile_profile``
(post-hoc aggregation, gap/uniform/mismatch reporting), the CLI probe
runner, the HTML report section (incl. a TOC-anchor regression test),
``diffreports.diff_cache_profile``, and tempdir mutation gates pinning
the exactness-critical lines.

Theory (red-team reviewed pre-code, briefs in
~/workspace/agents_out/cycle168/): for every shared prefix v with k_v
sharing prompts, every request order yields exactly k_v - 1 hits and
1 miss, so achievable = potential - #shared_prefixes order-free at
token level, and achievable_blocks = block_occurrences -
#distinct_shared_block_prefixes at block level. The
test_brute_force_agreement_* tests pin this proof in reality by
simulating an exact prefix cache over every permutation of random
prompt sets.
"""

import importlib.util
import itertools
import os
import random
import re
import tempfile
import unittest
from pathlib import Path

from model_protocol_lab.cacheprofile import (
    DEFAULT_BLOCK_SIZE,
    candidate_cache_profile,
    is_evaluable_prompt,
)


def _simulate_token_hits(prompts, order):
    """Exact token-prefix cache simulation (independent oracle code)."""
    cache = set()
    hits = 0
    for idx in order:
        prompt = prompts[idx]
        for end in range(1, len(prompt) + 1):
            if prompt[:end] in cache:
                hits += 1
        for end in range(1, len(prompt) + 1):
            cache.add(prompt[:end])
    return hits


def _simulate_block_hits(prompts, order, block_size):
    """Exact full-block prefix cache simulation (independent oracle code)."""
    cache = set()
    blocks = 0
    for idx in order:
        prompt = prompts[idx]
        length = len(prompt)
        b = 0
        while (b + 1) * block_size <= length:
            if prompt[: (b + 1) * block_size] in cache:
                blocks += 1
            b += 1
        b = 0
        while (b + 1) * block_size <= length:
            cache.add(prompt[: (b + 1) * block_size])
            b += 1
    return blocks


class CacheProfilePredicateTests(unittest.TestCase):
    """Cache profile predicate tests."""

    def test_accepts_lists_and_tuples(self):
        """Test accepts lists and tuples."""
        self.assertTrue(is_evaluable_prompt([1, 2, 3]))
        self.assertTrue(is_evaluable_prompt((0,)))

    def test_rejects_empty(self):
        """Test rejects empty."""
        self.assertFalse(is_evaluable_prompt([]))
        self.assertFalse(is_evaluable_prompt(()))

    def test_rejects_bools_floats_negatives(self):
        """Test rejects bools floats negatives."""
        # type(True) is int-adjacent: bools are not token ids (R13).
        self.assertFalse(is_evaluable_prompt([True, 2]))
        self.assertFalse(is_evaluable_prompt([1.0, 2]))
        self.assertFalse(is_evaluable_prompt([-1]))
        self.assertFalse(is_evaluable_prompt(["1"]))

    def test_rejects_non_sequences(self):
        """Test rejects non sequences."""
        self.assertFalse(is_evaluable_prompt(None))
        self.assertFalse(is_evaluable_prompt("123"))
        self.assertFalse(is_evaluable_prompt({"a": 1}))
        self.assertFalse(is_evaluable_prompt(42))


class CacheProfilePureTests(unittest.TestCase):
    """Cache profile pure tests."""

    def test_hand_verified_basic(self):
        """Test hand verified basic."""
        # Design-phase hand verification (5-case check, 2026-10-20):
        # [(1,2,3,4,5), (1,2,3,9,9), (7,8)]: cacheable 6/12, block(4) 0/12,
        # block(2) 4/12.
        cell = candidate_cache_profile([[1, 2, 3, 4, 5], [1, 2, 3, 9, 9], [7, 8]], block_size=4)
        self.assertEqual(cell["n"], 3)
        self.assertEqual(cell["n_evaluable"], 3)
        self.assertEqual(cell["total_tokens"], 12)
        self.assertEqual(cell["cacheable_tokens"], 6)
        self.assertEqual(cell["shared_prefixes"], 3)
        self.assertEqual(cell["achievable_tokens"], 3)
        self.assertEqual(cell["potential_token_fraction"], 6 / 12)
        self.assertEqual(cell["achievable_token_fraction"], 3 / 12)
        # No full 4-token block is shared (shared prefixes are
        # lengths 1..3): zero block-quantized reuse.
        self.assertEqual(cell["block_cacheable_blocks"], 0)
        self.assertEqual(cell["achievable_block_fraction"], 0.0)
        cell2 = candidate_cache_profile([[1, 2, 3, 4, 5], [1, 2, 3, 9, 9], [7, 8]], block_size=2)
        # Block prefixes (1,2) shared by 2 prompts: 2 occurrences,
        # 1 distinct identity -> 1 achievable block = 2 tokens.
        self.assertEqual(cell2["block_cacheable_blocks"], 2)
        self.assertEqual(cell2["distinct_block_prefixes"], 1)
        self.assertEqual(cell2["achievable_blocks"], 1)
        self.assertEqual(cell2["achievable_block_fraction"], 2 / 12)

    def test_r1_inversion_flips_under_achievable(self):
        """Test r1 inversion flips under achievable."""
        # Red-team A R1: potential provably misranks these two
        # templates against every request order; achievable ranks
        # them correctly (B > A). This is THE design-defining test.
        a = [[0] * 100 + [1] * 10, [0] * 100 + [2] * 10, [3] * 100 + [4] * 10, [3] * 100 + [5] * 10]
        b = [[7] * 80 + [100 + i] * 30 for i in range(1, 5)]
        ca = candidate_cache_profile(a)
        cb = candidate_cache_profile(b)
        self.assertEqual(ca["cacheable_tokens"], 400)
        self.assertEqual(cb["cacheable_tokens"], 320)
        self.assertGreater(ca["potential_token_fraction"], cb["potential_token_fraction"])
        # Every order: A yields 200 hits, B yields 240.
        self.assertEqual(ca["achievable_tokens"], 200)
        self.assertEqual(cb["achievable_tokens"], 240)
        self.assertLess(ca["achievable_token_fraction"], cb["achievable_token_fraction"])

    def test_r1_inversion_at_block_level(self):
        """Test r1 inversion at block level."""
        a = [[0] * 96 + [1] * 16, [0] * 96 + [2] * 16, [3] * 96 + [4] * 16, [3] * 96 + [5] * 16]
        b = [[7] * 80 + [100 + i] * 32 for i in range(1, 5)]
        ca = candidate_cache_profile(a, block_size=16)
        cb = candidate_cache_profile(b, block_size=16)
        self.assertGreater(ca["potential_block_fraction"], cb["potential_block_fraction"])
        self.assertEqual(ca["achievable_blocks"], 12)  # 24 - 12
        self.assertEqual(cb["achievable_blocks"], 15)  # 20 - 5
        self.assertLess(ca["achievable_block_fraction"], cb["achievable_block_fraction"])

    def test_identical_prompts_potential_vs_achievable(self):
        """Test identical prompts potential vs achievable."""
        # R7 framing guard: potential 1.0, realizable (n-1)/n = 0.5.
        cell = candidate_cache_profile([[1] * 20, [1] * 20])
        self.assertEqual(cell["potential_token_fraction"], 1.0)
        self.assertEqual(cell["achievable_token_fraction"], 0.5)
        self.assertEqual(cell["n_duplicate_prompts"], 2)

    def test_single_prompt_zero_sharing_not_none(self):
        """Test single prompt zero sharing not none."""
        # n_evaluable == 1 is "no sharing" (0.0), not "no data" (None).
        cell = candidate_cache_profile([[1, 2, 3]])
        self.assertEqual(cell["n_evaluable"], 1)
        self.assertEqual(cell["achievable_token_fraction"], 0.0)
        self.assertEqual(cell["potential_block_fraction"], 0.0)

    def test_prefix_of_another_prompt(self):
        """Test prefix of another prompt."""
        cell = candidate_cache_profile([[1, 2], [1, 2, 3, 4]])
        # Shared prefixes: (1,), (1,2); cacheable 2+2=4.
        self.assertEqual(cell["cacheable_tokens"], 4)
        self.assertEqual(cell["achievable_tokens"], 2)
        self.assertEqual(cell["achievable_token_fraction"], 2 / 6)
        cell2 = candidate_cache_profile([[1, 2], [1, 2, 3, 4]], block_size=2)
        self.assertEqual(cell2["achievable_blocks"], 1)
        self.assertEqual(cell2["achievable_block_fraction"], 2 / 6)

    def test_partial_trailing_blocks_never_count(self):
        """Test partial trailing blocks never count."""
        # vLLM caches full blocks only (S2): a shared 20-token prefix
        # with B=16 contributes exactly one block per prompt.
        cell = candidate_cache_profile([[1] * 20, [1] * 20], block_size=16)
        self.assertEqual(cell["block_cacheable_blocks"], 2)
        self.assertEqual(cell["achievable_block_fraction"], 16 / 40)

    def test_shorter_than_block_size(self):
        """Test shorter than block size."""
        cell = candidate_cache_profile([[1, 2, 3], [1, 2, 3]], block_size=16)
        self.assertEqual(cell["block_cacheable_blocks"], 0)
        self.assertEqual(cell["achievable_block_fraction"], 0.0)
        self.assertEqual(cell["achievable_token_fraction"], 0.5)

    def test_block_size_contract(self):
        """Test block size contract."""
        with self.assertRaises(ValueError):
            candidate_cache_profile([[1]], block_size=0)
        with self.assertRaises(ValueError):
            candidate_cache_profile([[1]], block_size=-2)
        with self.assertRaises(ValueError):
            candidate_cache_profile([[1]], block_size=True)  # bool is not int here
        with self.assertRaises(ValueError):
            candidate_cache_profile("not-a-list")

    def test_all_malformed_is_none_not_zero(self):
        """Test all malformed is none not zero."""
        # R5: n_evaluable == 0 -> fractions are None (never 0.0/nan).
        cell = candidate_cache_profile([[], "x", [True, 2], None, [-1], [1.5]])
        self.assertEqual(cell["n"], 6)
        self.assertEqual(cell["n_evaluable"], 0)
        self.assertEqual(cell["total_tokens"], 0)
        self.assertIsNone(cell["achievable_token_fraction"])
        self.assertIsNone(cell["potential_token_fraction"])
        self.assertIsNone(cell["achievable_block_fraction"])
        self.assertIsNone(cell["potential_block_fraction"])
        self.assertIsNone(cell["mean_prompt_tokens"])

    def test_mixed_malformed_excluded_fail_closed(self):
        """Test mixed malformed excluded fail closed."""
        cell = candidate_cache_profile([[1, 2, 3], "junk", [1, 2, 3]])
        self.assertEqual(cell["n"], 3)
        self.assertEqual(cell["n_evaluable"], 2)
        self.assertEqual(cell["n_duplicate_prompts"], 2)

    def test_empty_records_list(self):
        """Test empty records list."""
        cell = candidate_cache_profile([])
        self.assertEqual(cell["n"], 0)
        self.assertEqual(cell["n_evaluable"], 0)
        self.assertIsNone(cell["achievable_block_fraction"])

    def test_default_block_size_is_vllm_16(self):
        """Test default block size is vllm 16."""
        self.assertEqual(DEFAULT_BLOCK_SIZE, 16)
        explicit = candidate_cache_profile([[1, 2], [1, 2]])
        self.assertEqual(explicit["block_cacheable_blocks"], 0)

    def test_invariants_hold_on_random_sets(self):
        """Test invariants hold on random sets."""
        rng = random.Random(20261020)  # noqa: S311 -- deterministic test seed
        for _ in range(60):
            prompts = [
                [rng.randint(0, 5) for _ in range(rng.randint(1, 24))]
                for _ in range(rng.randint(1, 8))
            ]
            block_size = rng.choice([2, 4, 8, 16])
            cell = candidate_cache_profile(prompts, block_size=block_size)
            at = cell["achievable_token_fraction"]
            pt = cell["potential_token_fraction"]
            ab = cell["achievable_block_fraction"]
            pb = cell["potential_block_fraction"]
            # Red-team R8/S5/S11 ordering invariants.
            self.assertLessEqual(ab, at)
            self.assertLessEqual(at, pt)
            self.assertLessEqual(ab, pb)
            self.assertLessEqual(pb, pt)
            for key in (
                "achievable_token_fraction",
                "potential_token_fraction",
                "achievable_block_fraction",
                "potential_block_fraction",
            ):
                self.assertLessEqual(cell[key], 1.0)
                self.assertGreaterEqual(cell[key], 0.0)
            self.assertEqual(
                cell["achievable_tokens"],
                cell["cacheable_tokens"] - cell["shared_prefixes"],
            )
            self.assertEqual(
                cell["achievable_blocks"],
                cell["block_cacheable_blocks"] - cell["distinct_block_prefixes"],
            )
            self.assertLessEqual(cell["cacheable_tokens"], cell["total_tokens"])
            self.assertLessEqual(cell["n_duplicate_prompts"], cell["n_evaluable"])

    def test_brute_force_agreement_token_level(self):
        """Test brute force agreement token level."""
        # The R1 exactness proof, pinned in reality: for random prompt
        # sets, the achievable token count equals a simulated
        # exact-prefix cache over EVERY request order.
        rng = random.Random(7)  # noqa: S311 -- deterministic test seed
        checked = 0
        for _ in range(40):
            prompts = tuple(
                tuple(rng.randint(0, 2) for _ in range(rng.randint(1, 6)))
                for _ in range(rng.randint(1, 5))
            )
            cell = candidate_cache_profile([list(p) for p in prompts])
            simulated = {
                _simulate_token_hits(prompts, order)
                for order in itertools.permutations(range(len(prompts)))
            }
            self.assertEqual(len(simulated), 1, prompts)
            self.assertEqual(cell["achievable_tokens"], next(iter(simulated)))
            checked += 1
        self.assertGreater(checked, 0)

    def test_brute_force_agreement_block_level(self):
        """Test brute force agreement block level."""
        rng = random.Random(11)  # noqa: S311 -- deterministic test seed
        checked = 0
        for block_size in (2, 3, 4):
            for _ in range(25):
                prompts = tuple(
                    tuple(rng.randint(0, 2) for _ in range(rng.randint(1, 8)))
                    for _ in range(rng.randint(1, 5))
                )
                cell = candidate_cache_profile([list(p) for p in prompts], block_size=block_size)
                simulated = {
                    _simulate_block_hits(prompts, order, block_size)
                    for order in itertools.permutations(range(len(prompts)))
                }
                self.assertEqual(len(simulated), 1, prompts)
                self.assertEqual(cell["achievable_blocks"], next(iter(simulated)))
                checked += 1
        self.assertGreater(checked, 0)

    def test_deterministic_repeat(self):
        """Test deterministic repeat."""
        prompts = [[3, 1, 4, 1, 5], [3, 1, 4, 9], [2, 7]]
        first = candidate_cache_profile(prompts)
        second = candidate_cache_profile([list(p) for p in prompts])
        self.assertEqual(first, second)


class CacheProfileStageTests(unittest.TestCase):
    """Cache profile stage tests."""

    @staticmethod
    def _obs(candidate, fixture, status, input_ids):
        """Obs."""
        return {
            "candidate": candidate,
            "fixture": fixture,
            "status": status,
            "input_ids": input_ids,
        }

    def _stage(self, block_size=16):
        """Stage."""
        from model_protocol_lab import experiments

        return experiments.cache_profile_stage(
            observations=[
                self._obs("a", "f1", "completed", [1, 2, 3, 4]),
                self._obs("a", "f2", "completed", [1, 2, 3, 9]),
                self._obs("a", "f3", "failed", None),
                self._obs("a", "f4", "completed", "junk-ids"),
                self._obs("b", "f1", "completed", [7, 7, 7, 7]),
                self._obs("b", "f2", "completed", [7, 7, 8, 8]),
                {"not": "an observation"},
                None,
            ],
            block_size=block_size,
        )

    def test_stage_groups_and_excludes_fail_closed(self):
        """Test stage groups and excludes fail closed."""
        stage = self._stage()
        self.assertEqual(stage["n_observations"], 6)
        self.assertEqual(stage["block_size"], 16)
        cell_a = stage["per_candidate"]["a"]
        # 4 observations for "a" (one failed, one malformed-ids);
        # 2 evaluable.
        self.assertEqual(cell_a["n"], 4)
        self.assertEqual(cell_a["n_evaluable"], 2)
        self.assertEqual(cell_a["achievable_tokens"], 3)

    def test_stage_cells_pass_validator(self):
        """Test stage cells pass validator."""
        from model_protocol_lab import experiments

        stage = self._stage()
        for cell in stage["per_candidate"].values():
            self.assertTrue(experiments._cache_profile_cell_valid(cell), cell)

    def test_stage_block_size_recorded(self):
        """Test stage block size recorded."""
        stage = self._stage(block_size=8)
        self.assertEqual(stage["block_size"], 8)

    def test_stage_accepts_list_and_tuple(self):
        """Test stage accepts list and tuple."""
        from model_protocol_lab import experiments

        obs = [self._obs("a", "f1", "completed", (1, 2))]
        as_tuple = experiments.cache_profile_stage(observations=tuple(obs))
        as_list = experiments.cache_profile_stage(observations=list(obs))
        self.assertEqual(as_tuple["per_candidate"]["a"]["n_evaluable"], 1)
        self.assertEqual(as_list["per_candidate"]["a"]["n_evaluable"], 1)

    def test_stage_rejects_bad_inputs(self):
        """Test stage rejects bad inputs."""
        from model_protocol_lab import experiments

        with self.assertRaises(ValueError):
            experiments.cache_profile_stage(observations="junk")
        with self.assertRaises(ValueError):
            experiments.cache_profile_stage(observations=[], block_size=0)
        with self.assertRaises(ValueError):
            experiments.cache_profile_stage(observations=[], block_size=4.0)


class CacheProfileProfileTests(unittest.TestCase):
    """Cache profile profile tests."""

    @staticmethod
    def _obs_pair(a_prompts, b_prompts):
        """Obs pair."""
        from model_protocol_lab import experiments

        obs = []
        for i, ids in enumerate(a_prompts):
            obs.append(
                {
                    "candidate": "a",
                    "fixture": f"fa{i}",
                    "status": "completed",
                    "input_ids": ids,
                }
            )
        for i, ids in enumerate(b_prompts):
            obs.append(
                {
                    "candidate": "b",
                    "fixture": f"fb{i}",
                    "status": "completed",
                    "input_ids": ids,
                }
            )
        return experiments.cache_profile_stage(observations=obs)

    def test_gap_most_least(self):
        """Test gap most least."""
        from model_protocol_lab import experiments

        # Default block size 16: candidate a's prompts share a full
        # 16-token block (achievable block fraction 16/64 = 0.25);
        # candidate b's prompts share nothing.
        stage = self._obs_pair([[1] * 32, [1] * 31 + [2]], [[7] * 32, [8] * 32])
        profile = experiments.cache_profile_profile(("a", "b"), stage)
        self.assertTrue(profile["available"])
        self.assertEqual(profile["block_size"], 16)
        self.assertEqual(profile["n_observations"], 4)
        gap = profile["sharability_gap"]
        self.assertIsNotNone(gap)
        self.assertAlmostEqual(gap, 0.25)
        self.assertEqual(profile["most_sharable"], ["a"])
        self.assertEqual(profile["least_sharable"], ["b"])
        self.assertFalse(profile["uniform"])
        self.assertFalse(profile["evaluable_mismatch"])

    def test_uniform_exact_zero(self):
        """Test uniform exact zero."""
        # F1/D5: uniform is an exact 0.0 gap on identical input data.
        from model_protocol_lab import experiments

        prompts = [[1, 2, 3], [1, 2, 4]]
        stage = self._obs_pair(prompts, [list(p) for p in prompts])
        profile = experiments.cache_profile_profile(("a", "b"), stage)
        self.assertTrue(profile["available"])
        self.assertEqual(profile["sharability_gap"], 0.0)
        self.assertTrue(profile["uniform"])
        self.assertEqual(set(profile["most_sharable"]), {"a", "b"})
        self.assertEqual(set(profile["least_sharable"]), {"a", "b"})

    def test_excluded_candidates(self):
        """Test excluded candidates."""
        from model_protocol_lab import experiments

        stage = self._obs_pair([[1, 2], [1, 3]], [])
        profile = experiments.cache_profile_profile(("a", "b"), stage)
        self.assertTrue(profile["available"])
        self.assertEqual(profile["excluded_candidates"], ["b"])
        self.assertIsNone(profile["per_candidate"]["b"])
        self.assertEqual(profile["most_sharable"], ["a"])

    def test_all_excluded_gap_and_uniform_none(self):
        """Test all excluded gap and uniform none."""
        from model_protocol_lab import experiments

        stage = self._obs_pair([], [])
        profile = experiments.cache_profile_profile(("a", "b"), stage)
        self.assertTrue(profile["available"])
        self.assertIsNone(profile["sharability_gap"])
        self.assertIsNone(profile["uniform"])
        self.assertEqual(set(profile["excluded_candidates"]), {"a", "b"})

    def test_evaluable_mismatch_flagged(self):
        """Test evaluable mismatch flagged."""
        # R14: different denominators across candidates must warn.
        from model_protocol_lab import experiments

        stage = self._obs_pair([[1, 2], [1, 3], [1, 4]], [[1, 2], [1, 3]])
        profile = experiments.cache_profile_profile(("a", "b"), stage)
        self.assertTrue(profile["evaluable_mismatch"])

    def test_invalid_candidates(self):
        """Test invalid candidates."""
        from model_protocol_lab import experiments

        stage = self._obs_pair([[1, 2]], [[1, 3]])
        for bad in ((), ("a", "a"), ("",), "a"):
            profile = experiments.cache_profile_profile(bad, stage)
            self.assertFalse(profile["available"])
            self.assertEqual(profile["detail"], "invalid_candidates")

    def test_stage_invalid_branches(self):
        """Test stage invalid branches."""
        from model_protocol_lab import experiments

        stage = self._obs_pair([[1, 2]], [[1, 3]])
        self.assertEqual(
            experiments.cache_profile_profile(("a", "b"), None)["detail"],
            "stage_invalid",
        )
        broken = dict(stage)
        broken["block_size"] = 0
        self.assertEqual(
            experiments.cache_profile_profile(("a", "b"), broken)["detail"],
            "stage_invalid",
        )
        broken2 = dict(stage)
        broken2["n_observations"] = -1
        self.assertEqual(
            experiments.cache_profile_profile(("a", "b"), broken2)["detail"],
            "stage_invalid",
        )
        broken3 = dict(stage)
        broken3["per_candidate"] = [1, 2]
        self.assertEqual(
            experiments.cache_profile_profile(("a", "b"), broken3)["detail"],
            "stage_invalid",
        )

    def test_hostile_cell_rejected(self):
        """Test hostile cell rejected."""
        from model_protocol_lab import experiments

        stage = self._obs_pair([[1, 2], [1, 3]], [[7, 8], [7, 9]])
        cell = stage["per_candidate"]["a"]
        cell["achievable_tokens"] = 5  # break the achievable = cacheable - shared identity
        self.assertEqual(
            experiments.cache_profile_profile(("a", "b"), stage)["detail"],
            "stage_invalid",
        )

    def test_profile_non_dict_entry_is_stage_invalid(self):
        """Test profile non dict entry is stage invalid."""
        from model_protocol_lab import experiments

        stage = self._obs_pair([[1, 2]], [[1, 3]])
        stage["per_candidate"]["a"] = [1, 2, 3]
        self.assertEqual(
            experiments.cache_profile_profile(("a", "b"), stage)["detail"],
            "stage_invalid",
        )


class CacheProfileCellValidTests(unittest.TestCase):
    """Cache profile cell valid tests."""

    @staticmethod
    def _good_cell():
        """Good cell."""
        return candidate_cache_profile([[1, 2, 3], [1, 2, 4]])

    def test_valid_cell(self):
        """Test valid cell."""
        from model_protocol_lab import experiments

        self.assertTrue(experiments._cache_profile_cell_valid(self._good_cell()))

    def test_rejections(self):
        """Test rejections."""
        from model_protocol_lab import experiments

        valid = experiments._cache_profile_cell_valid
        base = self._good_cell()
        for key in ("n", "n_evaluable", "total_tokens", "cacheable_tokens"):
            broken = dict(base)
            broken[key] = -1
            self.assertFalse(valid(broken), key)
        broken = dict(base)
        broken["n_evaluable"] = broken["n"] + 1
        self.assertFalse(valid(broken))
        broken = dict(base)
        broken["achievable_token_fraction"] = 2.0
        self.assertFalse(valid(broken))
        broken = dict(base)
        broken["achievable_block_fraction"] = -0.1
        self.assertFalse(valid(broken))
        # Ordering violation: block above token.
        broken = dict(base)
        broken["achievable_block_fraction"] = broken["achievable_token_fraction"] + 0.5
        self.assertFalse(valid(broken))
        # Identity violation: needs a cell with actual block sharing
        # (the default good cell has none with block size 16).
        block_cell = candidate_cache_profile([[1] * 32, [1] * 31 + [2]], block_size=16)
        self.assertEqual(block_cell["achievable_blocks"], 1)
        broken = dict(block_cell)
        broken["distinct_block_prefixes"] = 0
        self.assertFalse(valid(broken))
        self.assertFalse(valid(None))
        self.assertFalse(valid([1]))

    def test_zero_evaluable_none_enforced(self):
        """Test zero evaluable none enforced."""
        from model_protocol_lab import experiments

        valid = experiments._cache_profile_cell_valid
        cell = candidate_cache_profile([])
        self.assertTrue(valid(cell))
        broken = dict(cell)
        broken["achievable_token_fraction"] = 0.0  # must be None, not 0.0 (R5)
        self.assertFalse(valid(broken))


class _Fmt:
    """Minimal stand-in for core.Format: the probe only reads .name."""

    def __init__(self, name):
        self.name = name


class CacheProfileCliTests(unittest.TestCase):
    """Cache profile cli tests."""

    @staticmethod
    def _result():
        """Result."""
        from dataclasses import asdict

        from model_protocol_lab.core import Observation

        def _observation(fixture, input_ids):
            """Observation."""
            return asdict(
                Observation(
                    candidate="t1",
                    fixture=fixture,
                    input_ids=input_ids,
                    output_ids=(9,),
                    answer="x",
                    exact_match=True,
                    boundary_found=True,
                    boundary_text_match=None,
                    raw_reason=None,
                    error=None,
                    status="completed",
                    input_text="",
                )
            )

        return {
            "search": [
                {
                    "format": {"name": "t1"},
                    "observations": [
                        _observation("f1", (1, 2, 3, 4, 5)),
                        _observation("f2", (1, 2, 3, 9, 9)),
                    ],
                },
            ]
        }

    def test_probe_runner_records_payload(self):
        """Test probe runner records payload."""
        from model_protocol_lab import cli

        result = self._result()
        cli._run_cache_profile_probe(result, (_Fmt("t1"),))
        payload = result["cache_profile"]
        self.assertTrue(payload["available"])
        self.assertIn("t1", payload["per_candidate"])
        self.assertEqual(payload["per_candidate"]["t1"]["n_evaluable"], 2)
        self.assertEqual(payload["block_size"], 16)
        # 5-token prompts share prefixes (1,), (1,2), (1,2,3):
        # achievable 3/10, and the probe records the exact fractions.
        self.assertAlmostEqual(payload["per_candidate"]["t1"]["achievable_token_fraction"], 3 / 10)

    def test_probe_runner_respects_block_size(self):
        """Test probe runner respects block size."""
        from model_protocol_lab import cli

        result = self._result()
        cli._run_cache_profile_probe(result, (_Fmt("t1"),), block_size=2)
        self.assertEqual(result["cache_profile"]["block_size"], 2)

    def test_probe_runner_never_raises(self):
        """Test probe runner never raises."""
        from model_protocol_lab import cli

        result = {"search": "not-a-list"}
        cli._run_cache_profile_probe(result, (_Fmt("t1"),))
        self.assertFalse(result["cache_profile"]["available"])
        self.assertEqual(result["cache_profile"]["detail"], "cache_profile_unavailable")

    def test_probe_runner_never_raises_on_bad_block_size(self):
        """Test probe runner never raises on bad block size."""
        from model_protocol_lab import cli

        result = self._result()
        cli._run_cache_profile_probe(result, (_Fmt("t1"),), block_size=0)
        self.assertFalse(result["cache_profile"]["available"])

    def test_probe_never_changes_selection(self):
        """Test probe never changes selection."""
        # R12: non-interference is mechanical -- the probe reads
        # recorded input_ids only and writes one namespaced key; the
        # run decision and selection keys are bit-identical afterwards.
        from model_protocol_lab import cli

        result = self._result()
        result["decision"] = "abstain"
        result["selected_candidate"] = None
        before = dict(result)
        cli._run_cache_profile_probe(result, (_Fmt("t1"),))
        for key in ("decision", "selected_candidate", "search"):
            self.assertEqual(result[key], before[key])
        self.assertIn("cache_profile", result)

    def test_probe_skips_non_dict_entries(self):
        """Test probe skips non dict entries."""
        from model_protocol_lab import cli

        result = {
            "search": [
                "junk",
                None,
                {"format": {"name": "t1"}},
            ]
        }
        cli._run_cache_profile_probe(result, (_Fmt("t1"),))
        self.assertTrue(result["cache_profile"]["available"])


class CacheProfileReportingTests(unittest.TestCase):
    """Cache profile reporting tests."""

    @staticmethod
    def _result(per_candidate, **overrides):
        """Result."""
        payload = {
            "available": True,
            "block_size": 16,
            "n_observations": 4,
            "per_candidate": per_candidate,
            "most_sharable": [],
            "least_sharable": [],
            "sharability_gap": None,
            "uniform": None,
            "excluded_candidates": [],
            "evaluable_mismatch": False,
        }
        payload.update(overrides)
        return {"cache_profile": payload}

    @staticmethod
    def _cell():
        """Cell."""
        return candidate_cache_profile([[1] * 32, [1] * 31 + [2]], block_size=16)

    def test_section_renders_sharability(self):
        """Test section renders sharability."""
        from model_protocol_lab import reporting

        result = self._result(
            {"t1": self._cell()},
            most_sharable=["t1"],
            sharability_gap=0.25,
        )
        html = reporting._cache_profile_section(result)
        self.assertIn('id="cache-profile"', html)
        self.assertIn("25.0%", html)  # achievable block fraction 16/64
        # Honest framing is present, not buried.
        self.assertIn("cost", html)
        self.assertIn("not a cost verdict", html)

    def test_section_escapes_hostile_names(self):
        """Test section escapes hostile names."""
        from model_protocol_lab import reporting

        evil = 't1"><script>alert(1)</script>'
        result = self._result({evil: self._cell()})
        html = reporting._cache_profile_section(result)
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)

    def test_section_empty_when_unavailable(self):
        """Test section empty when unavailable."""
        from model_protocol_lab import reporting

        self.assertEqual(reporting._cache_profile_section({}), "")
        self.assertEqual(
            reporting._cache_profile_section({"cache_profile": {"available": False}}),
            "",
        )

    def test_section_mismatch_warning(self):
        """Test section mismatch warning."""
        from model_protocol_lab import reporting

        result = self._result({"t1": self._cell()}, evaluable_mismatch=True)
        html = reporting._cache_profile_section(result)
        self.assertIn("different numbers of evaluable prompts", html)

    def test_toc_anchors_resolve_to_body_ids(self):
        """Test toc anchors resolve to body ids."""
        # Regression test (found during cycle 168): template_robustness
        # and ablation were registered in the TOC but never concatenated
        # into the HTML body, so their TOC links pointed at missing
        # anchors. Every TOC href must resolve to a body id.
        from model_protocol_lab import reporting

        result = {
            "decision": "abstain",
            "run_config": {},
            "search": [],
            "cache_profile": self._result({"t1": self._cell()})["cache_profile"],
            "template_robustness": {
                "available": True,
                "candidates": {
                    "tmpl-a": {
                        "status": "ok",
                        "preservation_rate": 0.75,
                        "n_preserved": 3,
                        "n_preservation_denominator": 4,
                        "operators": {
                            "case_flip": {"preservation_rate": 1.0},
                        },
                    },
                },
            },
            "ablation": {
                "available": True,
                "candidates": {
                    "t1": {
                        "status": "ok",
                        "base_accuracy": 1.0,
                        "base_passes_all": True,
                        "arms": {},
                        "not_applicable_fields": [],
                        "load_bearing_fields": [],
                        "any_attribution": False,
                    },
                },
            },
        }
        html = reporting.render_html(result)
        anchors = re.findall(r'href="#([a-z0-9-]+)"', html)
        self.assertIn("cache-profile", anchors)
        self.assertIn("template-robustness", anchors)
        self.assertIn("field-ablation", anchors)
        for anchor in set(anchors):
            self.assertIn(f'id="{anchor}"', html, f"TOC anchor missing: {anchor}")


def _cp_diff_payload(name="c", achievable_block=0.25, **overrides):
    """Cp diff payload."""
    # Producer-consistent stories (cycle 177, wave-7 F6b): the
    # strengthened validator recomputes every fraction from the
    # counts, so a different achievable-block share is a different
    # COUNT story (a larger prompt set with the same single
    # achievable block), never a bare fraction edit.
    if achievable_block == 0.0625:
        total, cacheable, shared, bcb, distinct, ablocks = 256, 128, 64, 8, 7, 1
    else:
        total, cacheable, shared, bcb, distinct, ablocks = 64, 32, 16, 2, 1, 1
    cell = {
        "n": 2,
        "n_evaluable": 2,
        "total_tokens": total,
        "mean_prompt_tokens": total / 2,
        "cacheable_tokens": cacheable,
        "shared_prefixes": shared,
        "achievable_tokens": cacheable - shared,
        "potential_token_fraction": cacheable / total,
        "achievable_token_fraction": (cacheable - shared) / total,
        "block_cacheable_blocks": bcb,
        "distinct_block_prefixes": distinct,
        "achievable_blocks": ablocks,
        "potential_block_fraction": 16 * bcb / total,
        "achievable_block_fraction": 16 * ablocks / total,
        "n_duplicate_prompts": 0,
    }
    cell.update(overrides)
    return {
        "available": True,
        "block_size": 16,
        "n_observations": 2,
        "per_candidate": {name: cell},
        "most_sharable": [name],
        "least_sharable": [name],
        "sharability_gap": 0.0,
        "uniform": True,
        "excluded_candidates": [],
        "evaluable_mismatch": False,
    }


class CacheProfileDiffTests(unittest.TestCase):
    """Cache profile diff tests."""

    def test_identical_payloads(self):
        """Test identical payloads."""
        from model_protocol_lab import diffreports

        old = _cp_diff_payload()
        diff = diffreports.diff_cache_profile(old, _cp_diff_payload())
        self.assertEqual(diff["available"], "both")
        self.assertFalse(diff["changed"])
        self.assertEqual(diff["block_drift"], [])
        self.assertEqual(diff["candidates_added"], [])

    def test_block_drift_detected(self):
        """Test block drift detected."""
        from model_protocol_lab import diffreports

        old = _cp_diff_payload(achievable_block=0.25)
        new = _cp_diff_payload(achievable_block=0.0625)
        diff = diffreports.diff_cache_profile(old, new)
        self.assertTrue(diff["changed"])
        self.assertEqual(len(diff["block_drift"]), 1)
        self.assertEqual(diff["block_drift"][0]["candidate"], "c")
        self.assertEqual(diff["block_drift"][0]["old"], 0.25)
        self.assertEqual(diff["block_drift"][0]["new"], 0.0625)
        self.assertEqual(diff["token_drift"], [])
        self.assertEqual(diff["potential_drift"], [])

    def test_token_and_potential_drift_detected(self):
        """Test token and potential drift detected."""
        from model_protocol_lab import diffreports

        old = _cp_diff_payload(achievable_block=0.25)
        # Count story for the new side (cycle 177, F6b): total 320,
        # achievable tokens 288 (0.9), cacheable 304 (potential
        # token 0.95), block-cacheable 18 (potential block 0.9),
        # achievable blocks 5 (achievable block stays 0.25).
        new = _cp_diff_payload(
            achievable_block=0.25,
            total_tokens=320,
            mean_prompt_tokens=160.0,
            cacheable_tokens=304,
            shared_prefixes=16,
            achievable_tokens=288,
            block_cacheable_blocks=18,
            distinct_block_prefixes=13,
            achievable_blocks=5,
            achievable_token_fraction=0.9,
            potential_block_fraction=0.9,
            potential_token_fraction=0.95,
        )
        diff = diffreports.diff_cache_profile(old, new)
        self.assertTrue(diff["changed"])
        self.assertEqual(diff["block_drift"], [])
        self.assertEqual(len(diff["token_drift"]), 1)
        self.assertEqual(len(diff["potential_drift"]), 1)

    def test_candidates_added_removed(self):
        """Test candidates added removed."""
        from model_protocol_lab import diffreports

        old = _cp_diff_payload(name="c")
        new = _cp_diff_payload(name="d")
        diff = diffreports.diff_cache_profile(old, new)
        self.assertTrue(diff["changed"])
        self.assertEqual(diff["candidates_added"], ["d"])
        self.assertEqual(diff["candidates_removed"], ["c"])

    def test_excluded_added_removed(self):
        """Test excluded added removed."""
        from model_protocol_lab import diffreports

        old = _cp_diff_payload(name="c")
        new = _cp_diff_payload(name="c")
        old["per_candidate"]["c"] = None
        old["excluded_candidates"] = ["c"]
        # The no-evaluable profile story (cycle 177, F6b): no
        # extremes, no gap, no uniformity verdict.
        old["most_sharable"] = []
        old["least_sharable"] = []
        old["sharability_gap"] = None
        old["uniform"] = None
        diff = diffreports.diff_cache_profile(old, new)
        self.assertTrue(diff["changed"])
        self.assertEqual(diff["excluded_added"], [])
        self.assertEqual(diff["excluded_removed"], ["c"])

    def test_asymmetric_and_neither(self):
        """Test asymmetric and neither."""
        from model_protocol_lab import diffreports

        old = _cp_diff_payload()
        only_old = diffreports.diff_cache_profile(old, None)
        self.assertEqual(only_old["available"], "old")
        self.assertFalse(only_old["changed"])
        only_new = diffreports.diff_cache_profile(None, old)
        self.assertEqual(only_new["available"], "new")
        neither = diffreports.diff_cache_profile(None, None)
        self.assertEqual(neither["available"], "neither")
        self.assertFalse(neither["changed"])

    def test_hostile_payload_rejected(self):
        """Test hostile payload rejected."""
        from model_protocol_lab import diffreports

        valid = _cp_diff_payload()
        self.assertTrue(diffreports._valid_cache_profile(valid))
        hostile = _cp_diff_payload()
        hostile["block_size"] = 0
        self.assertFalse(diffreports._valid_cache_profile(hostile))
        hostile2 = _cp_diff_payload()
        hostile2["per_candidate"]["c"]["achievable_token_fraction"] = 1.5
        self.assertFalse(diffreports._valid_cache_profile(hostile2))
        hostile3 = _cp_diff_payload()
        hostile3["per_candidate"]["c"]["achievable_tokens"] = 99
        self.assertFalse(diffreports._valid_cache_profile(hostile3))
        hostile4 = _cp_diff_payload()
        hostile4["per_candidate"]["c"] = None  # missing cell is fine
        hostile4["excluded_candidates"] = ["c"]
        hostile4["most_sharable"] = []
        hostile4["least_sharable"] = []
        hostile4["sharability_gap"] = None
        hostile4["uniform"] = None
        self.assertTrue(diffreports._valid_cache_profile(hostile4))
        hostile5 = _cp_diff_payload()
        hostile5["per_candidate"] = {"c": "not-a-dict"}
        self.assertFalse(diffreports._valid_cache_profile(hostile5))
        # None-vs-0.0 discipline (R5).
        hostile6 = _cp_diff_payload()
        hostile6["per_candidate"]["c"] = candidate_cache_profile([])
        hostile6["excluded_candidates"] = ["c"]
        hostile6["most_sharable"] = []
        hostile6["least_sharable"] = []
        hostile6["sharability_gap"] = None
        hostile6["uniform"] = None
        self.assertTrue(diffreports._valid_cache_profile(hostile6))
        hostile6["per_candidate"]["c"]["achievable_token_fraction"] = 0.0
        self.assertFalse(diffreports._valid_cache_profile(hostile6))

    def test_render_summary_lines(self):
        """Test render summary lines."""
        from model_protocol_lab import diffreports

        old = _cp_diff_payload(achievable_block=0.25)
        new = _cp_diff_payload(achievable_block=0.0625)
        diff = diffreports.diff_cache_profile(old, new)
        lines = diffreports._render_cache_profile_summary(diff)
        self.assertTrue(
            any("achievable-block-share" in line and "c" in line for line in lines),
            lines,
        )
        self.assertEqual(diffreports._render_cache_profile_summary({"available": "neither"}), [])
        self.assertEqual(diffreports._render_cache_profile_summary(None), [])
        one_sided = diffreports.diff_cache_profile(old, None)
        lines = diffreports._render_cache_profile_summary(one_sided)
        self.assertTrue(any("unavailable" in line for line in lines), lines)

    def test_render_summary_escapes_hostile_names(self):
        """Test render summary escapes hostile names."""
        from model_protocol_lab import diffreports

        evil = "c\rmalicious"
        old = _cp_diff_payload(name=evil, achievable_block=0.25)
        new = _cp_diff_payload(name=evil, achievable_block=0.0625)
        diff = diffreports.diff_cache_profile(old, new)
        lines = diffreports._render_cache_profile_summary(diff)
        joined = "\n".join(lines)
        self.assertNotIn("\r", joined)

    def test_diff_reports_integration(self):
        """Test diff reports integration."""
        from model_protocol_lab import diffreports

        base = diffreports.diff_reports({"decision": "abstain"}, {"decision": "abstain"})
        self.assertIn("cache_profile", base)
        self.assertIn("cache_profile_changed", base["summary"])
        self.assertFalse(base["summary"]["cache_profile_changed"])
        old = {
            "decision": "abstain",
            "cache_profile": _cp_diff_payload(achievable_block=0.25),
        }
        new = {
            "decision": "abstain",
            "cache_profile": _cp_diff_payload(achievable_block=0.0625),
        }
        diff = diffreports.diff_reports(old, new)
        self.assertTrue(diff["summary"]["cache_profile_changed"])
        self.assertFalse(diff["summary"]["identical"])
        self.assertFalse(diff["summary"]["cache_profile_asymmetric"])
        # Only the cache-profile section moved: the neighboring
        # sections changed flags stay False.
        self.assertFalse(diff["summary"]["verbosity_changed"])
        self.assertFalse(diff["summary"]["ablation_changed"])


class CacheProfileFailClosedTests(unittest.TestCase):
    """Cache profile fail closed tests."""

    def test_all_observations_malformed(self):
        """Test all observations malformed."""
        from model_protocol_lab import experiments

        stage = experiments.cache_profile_stage(
            observations=[
                {"candidate": "a", "status": "completed", "input_ids": None},
                {"candidate": "a", "status": "completed", "input_ids": []},
                {"candidate": "a", "status": "completed", "input_ids": [True]},
            ]
        )
        profile = experiments.cache_profile_profile(("a",), stage)
        self.assertTrue(profile["available"])
        self.assertEqual(profile["excluded_candidates"], ["a"])
        self.assertIsNone(profile["sharability_gap"])
        self.assertIsNone(profile["uniform"])

    def test_stage_never_raises_on_observation_data(self):
        """Test stage never raises on observation data."""
        from model_protocol_lab import experiments

        stage = experiments.cache_profile_stage(
            observations=[{"candidate": "a", "status": "completed", "input_ids": [2**70]}]
        )
        cell = stage["per_candidate"]["a"]
        self.assertEqual(cell["n_evaluable"], 1)

    def test_profile_invalidated_by_hostile_payload(self):
        """Test profile invalidated by hostile payload."""
        # _valid_cache_profile must reject a hostile payload that
        # looks structurally right: fractions inconsistent with counts.
        from model_protocol_lab import diffreports

        payload = _cp_diff_payload()
        payload["per_candidate"]["c"]["achievable_block_fraction"] = None
        self.assertFalse(diffreports._valid_cache_profile(payload))


class CacheProfileMutationGateTests(unittest.TestCase):
    """Tempdir mutation gates on the exactness-critical lines of cacheprofile."""

    def _run_gate(self, module_name, name, old_text, new_text, scenario):
        """Run gate."""
        mod = __import__(f"model_protocol_lab.{module_name}", fromlist=["x"])
        source = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertEqual(source.count(old_text), 1, f"mutant {name}: anchor not unique")
        mutated = source.replace(old_text, new_text)
        with tempfile.TemporaryDirectory(prefix="c168_mut_", dir="/tmp") as tmpdir:
            mod_path = os.path.join(tmpdir, f"{module_name}_mut_{name}.py")
            with open(mod_path, "w", encoding="utf-8") as handle:
                handle.write(mutated)
            dotted = f"model_protocol_lab.{module_name}_mut_{name}"
            spec = importlib.util.spec_from_file_location(dotted, mod_path)
            assert spec is not None and spec.loader is not None
            mutant = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mutant)
            scenario(mod)  # passes on the real module: non-vacuous
            with self.assertRaises(AssertionError, msg=f"mutant {name} was NOT caught"):
                scenario(mutant)

    @staticmethod
    def _r1_scenario(mod):
        """R1 scenario."""
        # The exactness core: achievable tokens must drop the one
        # unavoidable first miss per shared prefix (R1).
        a = [[0] * 100 + [1] * 10, [0] * 100 + [2] * 10, [3] * 100 + [4] * 10, [3] * 100 + [5] * 10]
        cell = mod.candidate_cache_profile(a)
        assert cell["achievable_tokens"] == 200, cell
        assert cell["achievable_token_fraction"] == 200 / 440, cell

    def test_mutant_achievable_drops_first_miss_caught(self):
        """Test mutant achievable drops first miss caught."""
        self._run_gate(
            "cacheprofile",
            "achievable_no_first_miss",
            "    achievable_tokens = cacheable_tokens - shared_prefixes",
            "    achievable_tokens = cacheable_tokens  # MUTANT",
            self._r1_scenario,
        )

    @staticmethod
    def _r1_block_scenario(mod):
        """R1 block scenario."""
        b = [[7] * 80 + [100 + i] * 32 for i in range(1, 5)]
        cell = mod.candidate_cache_profile(b, block_size=16)
        assert cell["achievable_blocks"] == 15, cell
        assert cell["achievable_block_fraction"] == 16 * 15 / 448, cell

    def test_mutant_achievable_block_drops_first_miss_caught(self):
        """Test mutant achievable block drops first miss caught."""
        self._run_gate(
            "cacheprofile",
            "achievable_block_no_first_miss",
            "    achievable_blocks = block_cacheable_blocks - distinct_block_prefixes",
            "    achievable_blocks = block_cacheable_blocks  # MUTANT",
            self._r1_block_scenario,
        )

    @staticmethod
    def _predicate_scenario(mod):
        """Predicate scenario."""
        # Empty prompts must be excluded (R5/D1), not counted.
        assert mod.is_evaluable_prompt([1]) is True
        assert mod.is_evaluable_prompt([]) is False
        cell = mod.candidate_cache_profile([[]])
        assert cell["n_evaluable"] == 0, cell

    def test_mutant_empty_prompt_accepted_caught(self):
        """Test mutant empty prompt accepted caught."""
        self._run_gate(
            "cacheprofile",
            "empty_prompt_accepted",
            "    sequence: Any = value\n"
            "    if type(sequence) not in (list, tuple) or len(sequence) == 0:\n"
            "        return False",
            "    sequence: Any = value\n"
            "    if type(sequence) not in (list, tuple) or len(sequence) < 0:\n"
            "        return False  # MUTANT",
            self._predicate_scenario,
        )

    @staticmethod
    def _partial_block_scenario(mod):
        """Partial block scenario."""
        # S2: partial trailing blocks never count, even when the full
        # shared prefix is longer.
        cell = mod.candidate_cache_profile([[1] * 20, [1] * 20], block_size=16)
        assert cell["block_cacheable_blocks"] == 2, cell
        assert cell["achievable_block_fraction"] == 16 / 40, cell

    def test_mutant_partial_block_counted_caught(self):
        """Test mutant partial block counted caught."""
        self._run_gate(
            "cacheprofile",
            "partial_block_counted",
            "        if sharers >= 2 and len(prefix) % block_size == 0:",
            "        if sharers >= 2:  # MUTANT: ignores block quantization",
            self._partial_block_scenario,
        )


if __name__ == "__main__":
    unittest.main()
