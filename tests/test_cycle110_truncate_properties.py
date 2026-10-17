"""Cycle 110 -- Hypothesis property tests for truncate_text_tail.

Scope
-----
``model_protocol_lab.core.truncate_text_tail`` drops trailing bytes for
mid-token truncation in the boundary sensitivity sweep. It returns the
truncated string, or None when the cut would split a UTF-8 codepoint.
Example-based tests (test_cycle42, test_cycle56) pin specific behaviors;
these properties verify the UNIVERSAL invariants across arbitrary Unicode
text and cut positions.

Properties
----------
1. Prefix: a non-None result is always a strict prefix of the input.
2. Byte-exact: a non-None result encodes to exactly len(raw)-drop_bytes bytes.
3. None-means-split: a None result implies the cut bytes are not valid UTF-8.
4. Valid-cut-succeeds: cutting at a codepoint boundary always returns a string.
5. Invalid inputs (out-of-range drop_bytes, wrong types) raise ValueError.

Mutation gate: 3 mutants -- (1) return "" instead of None on split codepoint,
(2) drop drop_bytes+1 bytes, (3) accept drop_bytes == len(raw) -- each killed
by >=1 property test.
"""

import unittest
from typing import Any

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from model_protocol_lab.core import truncate_text_tail


def _encodable_text(min_size: int) -> Any:
    """Text strategy excluding lone surrogates (not UTF-8 encodable)."""
    # The exclude_categories tuple is a Literal["Cs"] per Hypothesis types;
    # the ignore documents the intentional str-tuple usage.
    return st.text(
        st.characters(exclude_categories=("Cs",)),  # type: ignore[arg-type]
        min_size=min_size,
    )


class TestTruncateTextTailProperties(unittest.TestCase):
    @given(data=st.data())
    @settings(max_examples=200)
    def test_result_is_prefix_or_none(self, data: Any) -> None:
        """A non-None result is always a strict prefix of the input text."""
        # min_size=2 ensures at least 2 bytes when UTF-8 encoded.
        text = data.draw(_encodable_text(min_size=2), label="text")
        raw = text.encode("utf-8")
        drop_bytes = data.draw(st.integers(min_value=1, max_value=len(raw) - 1), label="drop_bytes")
        result = truncate_text_tail(text, drop_bytes)
        if result is not None:
            self.assertTrue(
                text.startswith(result),
                f"result {result!r} is not a prefix of {text!r}",
            )
            self.assertLess(len(result), len(text))

    @given(data=st.data())
    @settings(max_examples=200)
    def test_result_byte_length_exact(self, data: Any) -> None:
        """A non-None result encodes to exactly len(raw) - drop_bytes bytes."""
        text = data.draw(_encodable_text(min_size=2), label="text")
        raw = text.encode("utf-8")
        drop_bytes = data.draw(st.integers(min_value=1, max_value=len(raw) - 1), label="drop_bytes")
        result = truncate_text_tail(text, drop_bytes)
        if result is not None:
            self.assertEqual(
                len(result.encode("utf-8")),
                len(raw) - drop_bytes,
            )

    @given(data=st.data())
    @settings(max_examples=200)
    def test_none_implies_split_codepoint(self, data: Any) -> None:
        """None is returned iff the cut splits a UTF-8 codepoint."""
        text = data.draw(_encodable_text(min_size=2), label="text")
        raw = text.encode("utf-8")
        drop_bytes = data.draw(st.integers(min_value=1, max_value=len(raw) - 1), label="drop_bytes")
        result = truncate_text_tail(text, drop_bytes)
        cut = raw[:-drop_bytes]
        try:
            cut.decode("utf-8")
            decodable = True
        except UnicodeDecodeError:
            decodable = False
        if result is None:
            self.assertFalse(
                decodable,
                "returned None but the cut bytes decode as valid UTF-8",
            )
        else:
            self.assertTrue(
                decodable,
                "returned a string but the cut bytes are not valid UTF-8",
            )

    @given(data=st.data())
    @settings(max_examples=200)
    def test_codepoint_boundary_cut_always_succeeds(self, data: Any) -> None:
        """Cutting exactly at a character boundary never returns None."""
        # Build text from whole characters, then cut at a char boundary.
        chars = data.draw(
            st.lists(
                st.characters(exclude_categories=("Cs",)),  # type: ignore[arg-type]
                min_size=2,
                max_size=20,
            ),
            label="chars",
        )
        text = "".join(chars)
        raw = text.encode("utf-8")
        # Choose a split point between characters 1..len(chars)-1.
        split_at = data.draw(st.integers(min_value=1, max_value=len(chars) - 1), label="split_at")
        keep_raw = "".join(chars[:split_at]).encode("utf-8")
        drop_bytes = len(raw) - len(keep_raw)
        result = truncate_text_tail(text, drop_bytes)
        self.assertIsNotNone(
            result,
            f"cut at char boundary {split_at} returned None for {text!r}",
        )
        assert result is not None  # narrow for mypy
        self.assertEqual(result, "".join(chars[:split_at]))

    @given(data=st.data())
    @settings(max_examples=100)
    def test_out_of_range_drop_bytes_raises(self, data: Any) -> None:
        """drop_bytes outside [1, len(raw)) raises ValueError."""
        text = data.draw(_encodable_text(min_size=1), label="text")
        raw = text.encode("utf-8")
        # Target the exact boundaries plus arbitrary out-of-range values.
        boundary = data.draw(
            st.sampled_from([0, -1, -100, len(raw), len(raw) + 1, len(raw) + 100]),
            label="boundary",
        )
        # Also try a random out-of-range integer.
        if data.draw(st.booleans(), label="use_random"):
            candidate = data.draw(st.integers(), label="random_int")
            assume(not (1 <= candidate < len(raw)))
            boundary = candidate
        with self.assertRaises(ValueError):
            truncate_text_tail(text, boundary)

    def test_wrong_types_raise(self) -> None:
        """Non-str text or non-int drop_bytes raises ValueError."""
        with self.assertRaises(ValueError):
            truncate_text_tail(123, 1)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            truncate_text_tail(b"bytes", 1)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            truncate_text_tail("hello", "1")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            truncate_text_tail("hello", 1.5)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
