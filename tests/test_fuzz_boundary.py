"""Property-based fuzz tests for boundary / stop-string matching.

Covers ``_hf_overhang_match`` and ``_first_firing_step`` from
``model_protocol_lab.backends`` (a port of HF transformers'
``StopStringCriteria``), the UTF-8 incremental-decoding buffering invariant,
and the NFC-normalization invariant documented in docs/RESEARCH_LOG.md
("Round 4 -- Cycle 1 research / R3": BPE merge fragments spanning stop
strings, mid-token UTF-8 splits, NFC shifting bytes, SentencePiece
byte-fallback).

A note on the overhang properties: the naive reading is "match fires iff the
stop string is a suffix of the concatenated text", but that "iff" is *false*
for this port -- intentionally. The port reproduces HF's overhang behaviour,
where a match fires even though the stop string never appears as a literal
suffix of the decoded text (e.g. ``["las", "topper"]`` matching ``"stop"``;
see the ``_hf_overhang_match`` docstring). The fuzz tests therefore assert
the true one-way invariant -- a genuine suffix completion is never missed --
and pin the documented overhang cases as regression tests instead of
asserting the false converse.
"""

import codecs
import unicodedata
import unittest

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from model_protocol_lab.backends import _first_firing_step, _hf_overhang_match

FUZZ = settings(
    derandomize=True,
    max_examples=40,
    # Same rationale as tests/test_hardening.py: differing_executors fires on
    # in-process re-runs (a harness artifact), not on flakiness; these tests
    # are deterministic, so the check is suppressed with justification.
    suppress_health_check=[HealthCheck.differing_executors],
)

ALPHA = "abcdef "
MIXED = ["a", "b", " ", "\u4e2d", "\u6587", "\u8a9e", "\u65e5"]


def _split_into_chunks(data, text):
    """Split ``text`` at strictly interior cut points: chunks stay nonempty."""
    if len(text) < 2:
        return [text]
    points = sorted(data.draw(st.lists(st.integers(1, len(text) - 1), max_size=5, unique=True)))
    chunks = [text[a:b] for a, b in zip([0, *points], [*points, len(text)], strict=True)]
    assert all(chunks)
    return chunks


def _naive_first_suffix_step(chunks, stop, base, total):
    """Reference: first 1-based step whose concatenated prefix ends with stop."""
    for step in range(1, total + 1):
        if "".join(chunks[: base + step]).endswith(stop):
            return step
    return None


class TestFuzzBoundary(unittest.TestCase):
    """Test fuzz boundary."""

    # -- _hf_overhang_match -------------------------------------------------

    @FUZZ
    @given(
        data=st.data(),
        prefix=st.text(alphabet=ALPHA, max_size=8),
        stop=st.text(alphabet=ALPHA, min_size=1, max_size=6),
    )
    def test_overhang_suffix_completeness(self, data, prefix, stop):
        """A genuine suffix completion is never missed.

        For arbitrary text ``prefix + stop`` split into nonempty chunks at
        arbitrary cut points, the stop string sits at the end of the
        concatenation, so the overhang matcher must fire.
        """
        chunks = _split_into_chunks(data, prefix + stop)
        self.assertTrue(_hf_overhang_match(chunks, stop))

    @FUZZ
    @given(chunks=st.lists(st.text(alphabet=ALPHA, max_size=5), max_size=5))
    def test_overhang_empty_inputs(self, chunks):
        """Guard rails: empty stop or no chunks never match."""
        self.assertFalse(_hf_overhang_match(chunks, ""))
        self.assertFalse(_hf_overhang_match([], "x"))

    def test_overhang_documented_hf_cases(self):
        """Pin HF's overhang behaviour: fires *without* a literal suffix.

        These are the token-boundary "overhang" cases from the HF docstring /
        tensor implementation that the port reproduces on purpose. They show
        why the naive "iff suffix" reading does not hold.
        """
        self.assertTrue(_hf_overhang_match(["las", "topper"], "stop"))
        self.assertFalse("lastopper".endswith("stop"))
        self.assertTrue(_hf_overhang_match([" hello", " world"], "or"))
        self.assertFalse(" hello world".endswith("or"))

    # -- _first_firing_step ---------------------------------------------------

    @FUZZ
    @given(
        data=st.data(),
        chunks=st.lists(st.text(alphabet=ALPHA, min_size=1, max_size=6), min_size=1, max_size=8),
        stop=st.text(alphabet=ALPHA, min_size=1, max_size=6),
    )
    def test_first_firing_step_range_consistency_minimality(self, data, chunks, stop):
        """Result is in range, self-consistent, and truly the *first* step."""
        base = data.draw(st.integers(0, len(chunks) - 1))
        total = data.draw(st.integers(1, len(chunks) - base))
        result = _first_firing_step(chunks, stop, base, total)
        self.assertTrue(result is None or 1 <= result <= total)
        if result is not None:
            self.assertTrue(_hf_overhang_match(chunks[: base + result], stop))
            for earlier in range(1, result):
                self.assertFalse(_hf_overhang_match(chunks[: base + earlier], stop))

    @FUZZ
    @given(
        data=st.data(),
        chunks=st.lists(st.text(alphabet=ALPHA, min_size=1, max_size=6), min_size=1, max_size=8),
        stop=st.text(alphabet=ALPHA, min_size=1, max_size=6),
    )
    def test_first_firing_step_never_misses_suffix(self, data, chunks, stop):
        """Never misses a genuine suffix completion (cross-checked vs naive).

        If plain ``str.endswith`` on concatenated prefixes first completes at
        step ``n``, the port must fire at step ``n`` or earlier (overhang /
        interior-occurrence matches may fire earlier -- see
        ``test_first_firing_step_overhang_fires_early``). No assertion is
        made when the naive reference never fires: the port can still fire
        without any literal suffix (e.g. chunks ``[" a"]``, stop ``" "``,
        where the stop merely *occurs* inside the final token, as in real
        HF).
        """
        base = data.draw(st.integers(0, len(chunks) - 1))
        total = data.draw(st.integers(1, len(chunks) - base))
        naive = _naive_first_suffix_step(chunks, stop, base, total)
        result = _first_firing_step(chunks, stop, base, total)
        if naive is not None:
            self.assertIsNotNone(result)
            self.assertLessEqual(result, naive)

    def test_first_firing_step_overhang_fires_early(self):
        """Overhang can fire where the naive suffix reference never does."""
        chunks = ["las", "topper"]
        self.assertEqual(_first_firing_step(chunks, "stop", 0, 2), 2)
        self.assertIsNone(_naive_first_suffix_step(chunks, "stop", 0, 2))

    # -- UTF-8 incremental decoding -------------------------------------------

    @FUZZ
    @given(text=st.text(max_size=30), cuts=st.lists(st.integers(0, 120), max_size=10))
    def test_utf8_incremental_decode(self, text, cuts):
        """Splitting UTF-8 bytes at arbitrary offsets and re-decoding
        incrementally (with buffering) always yields the original text --
        even when cuts land inside multi-byte sequences.
        """
        raw = text.encode("utf-8")
        points = sorted({c for c in cuts if 0 < c < len(raw)})
        pieces = [raw[a:b] for a, b in zip([0, *points], [*points, len(raw)], strict=True)]
        decoder = codecs.getincrementaldecoder("utf-8")()
        rebuilt = "".join(decoder.decode(piece) for piece in pieces)
        rebuilt += decoder.decode(b"", final=True)
        self.assertEqual(rebuilt, text)

    def test_utf8_split_inside_multibyte_sequence(self):
        """Pinned mid-codepoint split: "h\u00e9llo" cut inside "\u00e9"."""
        raw = "h\u00e9llo".encode("utf-8")  # b'h\xc3\xa9llo'
        decoder = codecs.getincrementaldecoder("utf-8")()
        out = decoder.decode(raw[:2]) + decoder.decode(raw[2:]) + decoder.decode(b"", final=True)
        self.assertEqual(out, "h\u00e9llo")

    # -- NFC normalization ------------------------------------------------------

    @FUZZ
    @given(text=st.text(max_size=30))
    def test_nfc_utf8_roundtrip(self, text):
        """NFC-normalized text round-trips through UTF-8 unchanged."""
        nfc = unicodedata.normalize("NFC", text)
        self.assertEqual(nfc.encode("utf-8").decode("utf-8"), nfc)

    @FUZZ
    @given(
        pairs=st.lists(
            st.tuples(
                st.sampled_from("aeiouAEIOU"),
                # grave/acute/circumflex: every pair composes to a 2-byte
                # precomposed vowel (verified), so the raw 3-byte sequence
                # always differs in length from its NFC form.
                st.characters(min_codepoint=0x300, max_codepoint=0x302),
            ),
            min_size=1,
            max_size=6,
        )
    )
    def test_nfc_decomposed_changes_byte_stream(self, pairs):
        """Decomposed (non-NFC) text has a different, longer byte stream.

        Every base-letter + combining-mark pair (U+0300..U+0306) composes, so
        the raw UTF-8 bytes can never serve as a byte-offset table for the
        NFC-normalized text -- this is why the lab normalizes fixtures to
        NFC *before* encoding.
        """
        text = "".join(a + b for a, b in pairs)
        nfc = unicodedata.normalize("NFC", text)
        self.assertNotEqual(nfc, text)
        raw, normalized = text.encode("utf-8"), nfc.encode("utf-8")
        self.assertNotEqual(raw, normalized)
        self.assertGreater(len(raw), len(normalized))

    def test_nfc_byte_offset_disagreement(self):
        """Pinned: a byte offset valid for raw bytes misleads on NFC text.

        In "e\\u0301x" the final "x" starts at byte 3 of the raw encoding
        but at byte 2 of the NFC encoding -- a naive byte-offset index
        built on raw bytes points at the wrong place after normalization.
        """
        text = "e\u0301x"
        raw, nfc = text.encode("utf-8"), unicodedata.normalize("NFC", text).encode("utf-8")
        self.assertEqual(raw, b"e\xcc\x81x")
        self.assertEqual(nfc, "\u00e9x".encode("utf-8"))
        raw_offset = len("e\u0301".encode("utf-8"))
        nfc_offset = len("\u00e9".encode("utf-8"))
        self.assertEqual(raw_offset, 3)
        self.assertEqual(nfc_offset, 2)
        self.assertNotEqual(raw_offset, nfc_offset)

    # -- SentencePiece-style byte fallback ---------------------------------------

    @FUZZ
    @given(text=st.text(alphabet=st.sampled_from(MIXED), min_size=1, max_size=8))
    def test_byte_fallback_single_byte_chunks(self, text):
        """Byte-fallback tokens (one UTF-8 byte per chunk): a stop string
        spanning those byte chunks still fires at the suffix.
        """
        stop = text[-1:]  # nonempty suffix of the text
        byte_chunks = [bytes([b]) for b in text.encode("utf-8")]
        self.assertTrue(_hf_overhang_match(byte_chunks, stop.encode("utf-8")))

    @FUZZ
    @given(
        data=st.data(),
        text=st.text(alphabet=st.sampled_from(MIXED), min_size=1, max_size=8),
    )
    def test_byte_fallback_arbitrary_byte_splits(self, data, text):
        """Byte stream split at arbitrary byte offsets (cuts may land inside
        a multi-byte sequence): a suffix stop string still fires.
        """
        raw = text.encode("utf-8")
        stop = text[-2:] if len(text) >= 2 else text
        if len(raw) >= 2:
            points = sorted(
                data.draw(st.lists(st.integers(1, len(raw) - 1), max_size=10, unique=True))
            )
        else:
            points = []
        chunks = [raw[a:b] for a, b in zip([0, *points], [*points, len(raw)], strict=True)]
        assert all(chunks)
        self.assertTrue(_hf_overhang_match(chunks, stop.encode("utf-8")))


if __name__ == "__main__":
    unittest.main()
