"""W4 push 9 (SCI-B-04): the cache-profile prefix trie.

``cacheprofile._prefix_counts`` materialized every prefix of every
prompt as a tuple dict key: the stored key material is the sum over
distinct prefixes of their lengths -- quadratic. The readiness
lane's executed witness: prompts ``[range(200), range(200),
range(150)]`` (550 input tokens) produced 200 dict entries holding
20,100 stored key tokens. The replacement is a token trie (one node
per distinct prefix, O(1) tokens stored per node), with all four
consumers re-expressed over it and every output number unchanged:

- the golden example below was executed against the pre-swap
  implementation and is pinned byte-for-byte;
- a differential battery compares the public profile against a
  reference copy of the pre-swap algorithm (kept here as the
  oracle) over adversarial prompt sets;
- the witness's stored-node count is exactly 200, asserted via
  the trie's node counter (never via timing).
"""

import random
import unittest

from model_protocol_lab.cacheprofile import _prefix_trie, candidate_cache_profile

GOLDEN_PROMPTS = [(1, 2, 3, 4), (1, 2, 3, 9), (1, 2), (5,), (1, 2, 3, 4)]
GOLDEN = {
    "achievable_block_fraction": 0.5333333333333333,
    "achievable_blocks": 4,
    "achievable_token_fraction": 0.6,
    "achievable_tokens": 9,
    "block_cacheable_blocks": 6,
    "cacheable_tokens": 13,
    "distinct_block_prefixes": 2,
    "mean_prompt_tokens": 3.0,
    "n": 5,
    # SCI-B-02 correction: exact multiplicity gives 2 here (the two
    # occurrences of the one repeated prompt); the pre-audit prefix
    # count gave 3 by also counting (1, 2), a proper prefix of
    # longer prompts, as a duplicate.
    "n_duplicate_prompts": 2,
    "n_evaluable": 5,
    "potential_block_fraction": 0.8,
    "potential_token_fraction": 0.8666666666666667,
    "shared_prefixes": 4,
    "total_tokens": 15,
}


def _reference_profile(records, block_size):
    """The pre-swap dict-of-tuples algorithm, kept as the oracle.

    One deliberate deviation: ``n_duplicate_prompts`` follows the
    SCI-B-02-corrected exact-multiplicity rule, not the pre-swap
    prefix count — the external audit found the prefix count
    wrong (a proper prefix of a longer prompt is not a duplicate),
    and the oracle must not keep asserting the audited defect.
    Every other quantity is the pre-swap reference verbatim.
    """
    from model_protocol_lab.cacheprofile import is_evaluable_prompt

    n = len(records)
    prompts = [tuple(record) for record in records if is_evaluable_prompt(record)]
    n_evaluable = len(prompts)
    if n_evaluable == 0:
        return {
            "n": n,
            "n_evaluable": 0,
            "total_tokens": 0,
            "mean_prompt_tokens": None,
            "cacheable_tokens": 0,
            "shared_prefixes": 0,
            "achievable_tokens": 0,
            "potential_token_fraction": None,
            "achievable_token_fraction": None,
            "block_cacheable_blocks": 0,
            "distinct_block_prefixes": 0,
            "achievable_blocks": 0,
            "potential_block_fraction": None,
            "achievable_block_fraction": None,
            "n_duplicate_prompts": 0,
        }
    counts = {}
    for prompt in prompts:
        for end in range(1, len(prompt) + 1):
            prefix = prompt[:end]
            counts[prefix] = counts.get(prefix, 0) + 1
    exact: dict = {}
    for prompt in prompts:
        exact[prompt] = exact.get(prompt, 0) + 1
    total_tokens = sum(len(prompt) for prompt in prompts)
    cacheable_tokens = 0
    for prompt in prompts:
        for end in range(1, len(prompt) + 1):
            if counts[prompt[:end]] >= 2:
                cacheable_tokens += 1
    shared_prefixes = sum(1 for sharers in counts.values() if sharers >= 2)
    achievable_tokens = cacheable_tokens - shared_prefixes
    block_cacheable_blocks = 0
    distinct_block_prefixes = 0
    for prefix, sharers in counts.items():
        if sharers >= 2 and len(prefix) % block_size == 0:
            distinct_block_prefixes += 1
            block_cacheable_blocks += sharers
    return {
        "n": n,
        "n_evaluable": n_evaluable,
        "total_tokens": total_tokens,
        "mean_prompt_tokens": total_tokens / n_evaluable,
        "cacheable_tokens": cacheable_tokens,
        "shared_prefixes": shared_prefixes,
        "achievable_tokens": achievable_tokens,
        "potential_token_fraction": cacheable_tokens / total_tokens,
        "achievable_token_fraction": achievable_tokens / total_tokens,
        "block_cacheable_blocks": block_cacheable_blocks,
        "distinct_block_prefixes": distinct_block_prefixes,
        "achievable_blocks": block_cacheable_blocks - distinct_block_prefixes,
        "potential_block_fraction": block_size * block_cacheable_blocks / total_tokens,
        "achievable_block_fraction": block_size
        * (block_cacheable_blocks - distinct_block_prefixes)
        / total_tokens,
        "n_duplicate_prompts": sum(1 for prompt in prompts if exact[prompt] >= 2),
    }


class GoldenTests(unittest.TestCase):
    """Test golden."""

    def test_golden_example_byte_identical(self):
        """Test golden example byte identical."""
        self.assertEqual(candidate_cache_profile(list(GOLDEN_PROMPTS), block_size=2), GOLDEN)


class DifferentialTests(unittest.TestCase):
    """Test differential."""

    def _check(self, records, block_size):
        """Check."""
        self.assertEqual(
            candidate_cache_profile(list(records), block_size=block_size),
            _reference_profile(list(records), block_size),
        )

    def test_adversarial_sets(self):
        """Test adversarial sets."""
        sets = [
            [],
            [None, "x", [], [True, False]],
            [(7,)],
            [(3, 3), (3, 3), (3, 3)],
            [(1,), (1, 2), (1, 2, 3), (1, 2, 3, 4)],
            [tuple(range(16)), tuple(range(16)), tuple(range(32))],
            [tuple(range(17)), tuple(range(17))],
            [(0, 0, 0), (0, 0), (0,)],
            [[1, 2], [1, 2, 3]],
        ]
        for records in sets:
            for block_size in (1, 2, 16):
                with self.subTest(records=records, block_size=block_size):
                    self._check(records, block_size)

    def test_randomized_battery(self):
        """Test randomized battery."""
        rng = random.Random(20261116)  # noqa: S311 -- deterministic test seed
        for trial in range(60):
            n_prompts = rng.randrange(0, 9)
            records = []
            for _ in range(n_prompts):
                length = rng.randrange(1, 25)
                base = rng.choice([(1, 2, 3), (9,), (4, 5), ()])
                records.append(base + tuple(rng.randrange(0, 6) for _ in range(length)))
            block_size = rng.choice([1, 2, 3, 16])
            with self.subTest(trial=trial):
                self._check(records, block_size)


class TrieStorageTests(unittest.TestCase):
    """Test trie storage."""

    def test_witness_node_count_is_linear(self):
        """Test witness node count is linear."""
        prompts = [tuple(range(200)), tuple(range(200)), tuple(range(150))]
        _root, node_count = _prefix_trie(prompts)
        # 200 distinct prefixes; the pre-swap dict stored 20,100 key
        # tokens for the same input.
        self.assertEqual(node_count, 200)

    def test_node_count_equals_distinct_prefixes(self):
        """Test node count equals distinct prefixes."""
        prompts = [(1, 2, 3), (1, 2), (1, 9), (5,)]
        _root, node_count = _prefix_trie(prompts)
        distinct = {(1,), (1, 2), (1, 2, 3), (1, 9), (5,)}
        self.assertEqual(node_count, len(distinct))


if __name__ == "__main__":
    unittest.main()
