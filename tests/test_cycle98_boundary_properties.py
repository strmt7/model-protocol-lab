"""Cycle 98: property-based correctness audit of core token algorithms.

Example-based tests (97 cycles) pin specific behaviors. This module uses
Hypothesis to verify the *universal* properties of the core token-sequence
algorithms in ``model_protocol_lab.core`` — the exact functions the lab's
measurement depends on.
"""

from __future__ import annotations

import unittest

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from model_protocol_lab.core import first_boundary

_token_ids = st.lists(st.integers(min_value=0, max_value=50000), max_size=40).map(tuple)
_nonempty_stop = st.lists(st.integers(min_value=0, max_value=50000), min_size=1, max_size=8).map(
    tuple
)


class FirstBoundaryProperties(unittest.TestCase):
    @given(tokens=_token_ids, stop=_nonempty_stop)
    @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
    def test_found_index_is_exact_match(self, tokens, stop):
        """If an index is returned, the slice there equals stop exactly."""
        idx = first_boundary(tokens, stop)
        if idx is not None:
            self.assertEqual(tokens[idx : idx + len(stop)], stop)

    @given(tokens=_token_ids, stop=_nonempty_stop)
    @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
    def test_none_means_absent(self, tokens, stop):
        """None is returned iff stop never occurs as a contiguous slice."""
        idx = first_boundary(tokens, stop)
        occurs = any(tokens[i : i + len(stop)] == stop for i in range(len(tokens) - len(stop) + 1))
        self.assertEqual(idx is None, not occurs)

    @given(tokens=_token_ids, stop=_nonempty_stop)
    @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
    def test_result_is_first_occurrence(self, tokens, stop):
        """The returned index is the *first* occurrence, not just any."""
        idx = first_boundary(tokens, stop)
        if idx is not None:
            earlier = any(tokens[i : i + len(stop)] == stop for i in range(idx))
            self.assertFalse(earlier, f"earlier occurrence missed at {idx}")

    @given(tokens=_token_ids, stop=_nonempty_stop)
    @settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
    def test_index_in_range(self, tokens, stop):
        """A returned index is always a valid slice start."""
        idx = first_boundary(tokens, stop)
        if idx is not None:
            self.assertGreaterEqual(idx, 0)
            self.assertLessEqual(idx, len(tokens) - len(stop))

    @given(tokens=_token_ids)
    @settings(max_examples=200)
    def test_empty_stop_raises(self, tokens):
        with self.assertRaises(ValueError):
            first_boundary(tokens, ())

    @given(tokens=_token_ids, stop=_nonempty_stop)
    @settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
    def test_stop_longer_than_tokens_returns_none(self, tokens, stop):
        """A stop longer than the token sequence can never match."""
        if len(stop) > len(tokens):
            self.assertIsNone(first_boundary(tokens, stop))

    @given(
        prefix=_token_ids,
        stop=_nonempty_stop,
        suffix=_token_ids,
    )
    @settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
    def test_embedded_stop_found_at_prefix_end(self, prefix, stop, suffix):
        """A stop embedded at a known position is found there or earlier."""
        tokens = prefix + stop + suffix
        idx = first_boundary(tokens, stop)
        self.assertIsNotNone(idx)
        self.assertLessEqual(idx, len(prefix))
        # And the slice at the found index really is the stop.
        self.assertEqual(tokens[idx : idx + len(stop)], stop)


if __name__ == "__main__":
    unittest.main()
