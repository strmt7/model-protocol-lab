"""Cycle 168: coverage-gap tests for the new fail-closed branches.

The validators reject hostile payloads fail-closed; these tests pin
every rejection branch of _cache_profile_cell_valid (experiments) and
_valid_cache_profile (diffreports) plus the reporting section's
degraded paths, keeping the coverage gate above its rounding edge.
"""

import unittest


def _valid_cell():
    from model_protocol_lab.cacheprofile import candidate_cache_profile

    return candidate_cache_profile([[1, 2, 3], [1, 2, 4]])


class CellValidatorCoverageTests(unittest.TestCase):
    def test_fail_closed_rejections(self):
        from model_protocol_lab import experiments

        valid = experiments._cache_profile_cell_valid
        self.assertTrue(valid(_valid_cell()))
        hostile_values = {
            # Block identity: valid cell has 0 == 0 - 0; 0 - 1 != 0.
            "distinct_block_prefixes": 1,
            # Cacheable tokens cannot exceed the token total.
            "cacheable_tokens": 99,
            # Duplicates cannot exceed the evaluable count.
            "n_duplicate_prompts": 99,
            # Mean must be positive when n_evaluable > 0.
            "mean_prompt_tokens": -1.0,
        }
        for key, value in hostile_values.items():
            broken = dict(_valid_cell())
            broken[key] = value
            self.assertFalse(valid(broken), key)
        broken = dict(_valid_cell())
        broken["mean_prompt_tokens"] = 0.0
        self.assertFalse(valid(broken))
        # achievable tokens exceeding the (shrunk) total also breaks the
        # cacheable <= total invariant via achievable_tokens identity.
        broken = dict(_valid_cell())
        broken["total_tokens"] = 0
        self.assertFalse(valid(broken))

    def test_zero_cell_mean_branches(self):
        from model_protocol_lab import experiments
        from model_protocol_lab.cacheprofile import candidate_cache_profile

        valid = experiments._cache_profile_cell_valid
        zero = candidate_cache_profile([])
        self.assertTrue(valid(zero))
        broken = dict(zero)
        broken["mean_prompt_tokens"] = 3.0
        self.assertFalse(valid(broken))


class DiffValidatorCoverageTests(unittest.TestCase):
    def _payload(self):
        from model_protocol_lab.cacheprofile import candidate_cache_profile

        return {
            "available": True,
            "block_size": 16,
            "n_observations": 2,
            "per_candidate": {
                "c": candidate_cache_profile([[1] * 32, [1] * 31 + [2]], block_size=16)
            },
            "most_sharable": ["c"],
            "least_sharable": ["c"],
            "sharability_gap": 0.0,
            "uniform": True,
            "excluded_candidates": [],
            "evaluable_mismatch": False,
        }

    def test_rejection_branches(self):
        from model_protocol_lab import diffreports
        from model_protocol_lab.cacheprofile import candidate_cache_profile

        valid = diffreports._valid_cache_profile
        self.assertTrue(valid(self._payload()))
        # achievable_blocks identity.
        p = self._payload()
        p["per_candidate"]["c"]["distinct_block_prefixes"] = 0
        self.assertFalse(valid(p))
        # cacheable > total.
        p = self._payload()
        p["per_candidate"]["c"]["cacheable_tokens"] = 10**9
        self.assertFalse(valid(p))
        # duplicates > n_evaluable.
        p = self._payload()
        p["per_candidate"]["c"]["n_duplicate_prompts"] = 99
        self.assertFalse(valid(p))
        # Zero-evaluable cell with a fraction set (None discipline).
        p = self._payload()
        p["per_candidate"]["c"] = candidate_cache_profile([])
        p["per_candidate"]["c"]["achievable_token_fraction"] = 0.0
        self.assertFalse(valid(p))
        # Zero-evaluable cell with a mean set.
        p = self._payload()
        p["per_candidate"]["c"] = candidate_cache_profile([])
        p["per_candidate"]["c"]["mean_prompt_tokens"] = 1.0
        self.assertFalse(valid(p))
        # Non-positive mean.
        p = self._payload()
        p["per_candidate"]["c"]["mean_prompt_tokens"] = 0.0
        self.assertFalse(valid(p))
        # Negative count (per-key fail-closed loop).
        p = self._payload()
        p["per_candidate"]["c"]["total_tokens"] = -1
        self.assertFalse(valid(p))
        # Non-integer count.
        p = self._payload()
        p["per_candidate"]["c"]["n_evaluable"] = "2"
        self.assertFalse(valid(p))
        # achievable_tokens identity at payload level.
        p = self._payload()
        p["per_candidate"]["c"]["achievable_tokens"] = 10**9
        self.assertFalse(valid(p))


class ReportingDegradedPathsTests(unittest.TestCase):
    def _section(self, payload):
        from model_protocol_lab import reporting

        return reporting._cache_profile_section({"cache_profile": payload})

    def _base_payload(self, per_candidate):
        return {
            "available": True,
            "block_size": 16,
            "n_observations": 2,
            "per_candidate": per_candidate,
            "most_sharable": [],
            "least_sharable": [],
            "sharability_gap": None,
            "uniform": None,
            "excluded_candidates": [],
            "evaluable_mismatch": False,
        }

    def test_non_dict_per_candidate(self):
        self.assertEqual(self._section(self._base_payload([1, 2])), "")

    def test_non_dict_entry_skipped(self):
        from model_protocol_lab.cacheprofile import candidate_cache_profile

        cell = candidate_cache_profile([[1, 2], [1, 3]])
        html = self._section(self._base_payload({"good": cell, "bad": [1, 2, 3]}))
        self.assertIn("good", html)
        self.assertNotIn("bad", html)

    def test_zero_evaluable_renders_na(self):
        from model_protocol_lab.cacheprofile import candidate_cache_profile

        html = self._section(self._base_payload({"c": candidate_cache_profile([])}))
        self.assertIn("n/a", html)

    def test_excluded_candidates_rendered(self):
        from model_protocol_lab.cacheprofile import candidate_cache_profile

        payload = self._base_payload({"c": candidate_cache_profile([])})
        payload["excluded_candidates"] = ["ghost"]
        html = self._section(payload)
        self.assertIn("ghost", html)


if __name__ == "__main__":
    unittest.main()
