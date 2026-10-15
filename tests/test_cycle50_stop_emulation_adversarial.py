"""Cycle 50: adversarial differential re-verification of the HF stop-string emulation.

Compares ``backends.emulate_hf_stop_string`` against the real
``transformers.StopStringCriteria`` (transformers 5.19.0, torch 2.14.1+cpu)
as the reference oracle, on adversarial inputs plus seeded fuzz sweeps.

Cycle 14 verified the basic differential layer with minimal fake tokenizers
whose cleaned vocab coincides with per-token decode by construction. This
cycle is the adversarial layer: it drives the comparison with REAL
``tokenizers``-built ``PreTrainedTokenizerFast`` tokenizers on all three
upstream matching modes -- byte_level (ByteLevel decoder), byte_fallback
(ByteFallback decoder) and classic/None (Metaspace decoder: concatenative,
so the classic fuzz isolates the match algorithm; the WordPiece spacing
approximation is pinned separately) -- so the
reference exercises the genuine upstream table build
(``clean_tokenizer_vocab`` static-prefix cleaning, ``_token_to_bytes``,
the tensor walk), while the lab side runs the genuine
``emulate_hf_stop_string`` through the repo's own ``_hf_byte_token_mapper``
and a decode adapter issuing the exact
``decode(..., skip_special_tokens=False, clean_up_tokenization_spaces=False)``
call ``LocalHFBackend.decode`` uses. Identical ``(prompt_ids, output_ids,
stop_strings)`` feed both sides; the differential property is the first
firing step and, on fire, the kept ``truncated_ids`` with ``returned_text``
staying None (HF keeps the stop string).

Tie-break -- DOCUMENTED (repo) vs ACTUAL (transformers 5.19.0):
- DOCUMENTED (this repo: ``diffbackends`` module docstring, cycle-14
  ``HfMultiStopTests``): HF exposes no per-stop winner. ``__call__``
  returns a bare per-sequence any-match bool, so there is no
  first-listed-wins rule; the earliest-completing stop wins by
  construction of the per-token check, and same-step ties are
  outcome-identical (the same completing token is kept either way).
- ACTUAL (transformers 5.19.0 ``stopping_criteria.py``): ``__call__`` ends
  with ``return torch.any(string_matches, dim=-1)`` -- there is literally
  no selection or tie-break code path; which stop matched is unobservable
  from the API. Verified below (``TieBreakTests``): the joint
  first-firing step equals the min over per-stop first-firing steps for
  both stop orders, ties included -- with one documented exception, the
  multi-stop window divergence.

REAL DIVERGENCES found (EMULATION -> TRANSFORMERS, minimal reproducers;
source NOT fixed -- the coordinator decides):

1. Empty stop string ``""``: EMULATION ``available=False, fired=False``
   (``invalid_emulation_input``) -> TRANSFORMERS fires at step 1 for ANY
   input (the constructor accepts ``""``; ``target_lens=[0]`` is vacuously
   satisfied). Reproducer: ``emulate_hf_stop_string(backend, (), (7,), "")``
   vs ``StopStringCriteria(tokenizer, [""])`` stepped once.

2. Multi-stop window: EMULATION per-stop cells (each bounded to the last
   ``len(stop)`` representations) report never-fires -> TRANSFORMERS joint
   criteria fires at step 4. Upstream bounds the walk by the GLOBAL
   ``maximum_token_len = max(len(stop))``; empty-string token
   representations consume window budget without consuming stop-string
   units, so a short stop's walk can reach past its own per-stop window
   when a longer stop is present. Reproducer (byte path): tokens
   ``["a", "", "", "b"]``, stops ``["ab", "abcdef"]`` -> joint real step 4,
   per-stop lab ``{"ab": None, "abcdef": None}``. Single-stop behavior
   agrees (both silent). Requires ``""`` vocab entries -- exotic but
   representable in a real tokenizer.

3. Classic-path non-concatenative decoders (WordPiece space insertion):
   EMULATION matches on per-token isolated decodes (implicitly
   concatenated) while TRANSFORMERS matches on cleaned vocab strings (the
   static-prefix trick). Firing decisions diverge in both directions:
   a. EMULATION fires at step 2 -> TRANSFORMERS never fires: stop
      ``"ab"``, tokens ``["a", "b"]``; the true decoded text is ``"a b"``
      (the decoder inserts the inter-token space), so the real criterion
      is right and the emulation false-positives.
   b. EMULATION fires at step 2 -> TRANSFORMERS fires at step 1: stop
      ``" "``, tokens ``["aaa", " "]``; the cleaned ``" aaa"`` starts with
      a space (a static-prefix cleaning artifact), so the real criterion
      fires a step early -- arguably a real-HF false positive against its
      own decoded text ``"aaa"`` -- while the emulation fires at the true
      position. (The classic-path port docstring discloses "exotic spacing
      edge cases"; these are concrete instances.)

4. Non-contiguous vocab ids (degenerate): EMULATION's byte mapper bounds
   by ``len(vocab)`` and treats higher ids as walk-killing OOV ->
   TRANSFORMERS sizes its table by ``max(ids)`` and matches normally.
   Reproducer: vocab ids ``0..13`` plus ``"zzz"`` at id 50, stop ``"zzz"``,
   output ``(50,)`` -> real fires at step 1, lab never fires.

Related strictness note (outcome-agreement, not a firing divergence):
real ``StopStringCriteria`` raises ``ValueError`` at construction when no
vocab token can overlap the stop string; the emulation never raises and
returns available/no-fire. Both sides "never fire"; the lab is less
strict. Pinned by ``ConstructorStrictnessTests``.

Deliberately NOT verified:
- A live ``generate()`` loop (no model weights are available locally);
  the per-step ``__call__`` sweep replicates generate's per-token check
  exactly.
- OOV ids on the classic path: the repo's real backend rejects them in
  ``decode`` (the emulation degrades to ``token_text_unavailable``) while
  upstream clamps them to its dummy table row. Not differentially drivable
  through the repo's own backend validation, so documented as a boundary.
- The installed transformers ``.py`` source has one redacted line
  (``reversed_token=<redacted>``); the behavior verified here comes from
  the compiled ``.pyc`` actually imported (disassembly confirms the
  upstream ``token[::-1]``) and from empirical black-box agreement.
"""

from __future__ import annotations

import random
import unittest

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from transformers.generation.stopping_criteria import StopStringCriteria

from model_protocol_lab.backends import (
    _bytes_to_unicode,
    _hf_byte_token_mapper,
    emulate_hf_stop_string,
)

HAS_HF = True  # torch + transformers imported above; fail loudly otherwise.

_MERGES_ABCDEF = [
    ("a", "b"),
    ("ab", "c"),
    ("abc", "d"),
    ("abcd", "e"),
    ("abcde", "f"),
]

_LONE_SURROGATE = chr(0xD800)  # lone surrogate; not UTF-8-encodable.


def _build_tokenizer(
    pieces: list[str], merges: list[tuple[str, str]], decoder: str
) -> PreTrainedTokenizerFast:
    """Build a real PreTrainedTokenizerFast over an explicit BPE vocab."""
    if len(set(pieces)) != len(pieces):
        raise ValueError("duplicate piece in test vocab")
    vocab = {piece: index for index, piece in enumerate(pieces)}
    merges = [
        pair
        for pair in merges
        if pair[0] in vocab and pair[1] in vocab and pair[0] + pair[1] in vocab
    ]
    tok = Tokenizer(models.BPE(vocab=vocab, merges=merges, unk_token="[UNK]"))
    if decoder == "byte":
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
    elif decoder == "fallback":
        tok.decoder = decoders.ByteFallback()
    elif decoder == "wordpiece":
        tok.decoder = decoders.WordPiece(prefix="##")
    elif decoder == "metaspace":
        # Concatenative classic-path decoder (mode None): keeps the
        # differential fuzz on the match algorithm, free of the
        # WordPiece spacing approximation pinned separately below.
        tok.decoder = decoders.Metaspace()
    elif decoder == "none":
        pass
    else:
        raise ValueError("unknown decoder kind")
    return PreTrainedTokenizerFast(tokenizer_object=tok, clean_up_tokenization_spaces=False)


def _byte_pieces() -> list[str]:
    frag = _bytes_to_unicode()
    emoji_frags = [frag[byte] for byte in "😀".encode()]
    return [
        "[UNK]",
        "",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "ab",
        "abc",
        "abcd",
        "abcde",
        "abcdef",
        " ",
        "  ",
        "\n",
        "\t",
        "t",
        "h",
        "th",
        "he",
        "the",
        "s",
        "to",
        "st",
        "op",
        "stop",
        "p",
        "pp",
        "pped",
        "per",
        "topper",
        "las",
        "x",
        "zz",
        "or",
        "bab",
        "Ã",
        "¤",
        "©",
        "â",
        "é",
        "caf",
        "café",
        *emoji_frags,
        "😀",
        "€",
        "中",
        "é",
        "<0x41>",
        "<0x42>",
        "aa",
        "aaa",
        "aaaa",
        "A",
        "B",
        "AB",
        "q",
        "qq",
    ]


def _classic_pieces() -> list[str]:
    return [
        "[UNK]",
        "",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "ab",
        "abc",
        "abcd",
        "abcde",
        "abcdef",
        " ",
        "\n",
        "the",
        "##ing",
        "##ed",
        "s",
        "to",
        "st",
        "op",
        "stop",
        "x",
        "zz",
        "é",
        "😀",
        "aa",
        "aaa",
        "xstopper",
        "stopx",
        "las",
        "topper",
        "pped",
        "q",
    ]


def _fallback_pieces() -> list[str]:
    return [
        "[UNK]",
        "",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "ab",
        "abc",
        "abcd",
        "abcde",
        "abcdef",
        "<0x41>",
        "<0x42>",
        "x",
        "A",
        "B",
        " ",
        "\n",
    ]


_BYTE_TOK = _build_tokenizer(_byte_pieces(), _MERGES_ABCDEF, "byte")
_CLASSIC_TOK = _build_tokenizer(_classic_pieces(), _MERGES_ABCDEF, "metaspace")
_FALLBACK_TOK = _build_tokenizer(_fallback_pieces(), _MERGES_ABCDEF, "fallback")
_WORDPIECE_TOK = _build_tokenizer(_classic_pieces(), _MERGES_ABCDEF, "wordpiece")

_BYTE_VOCAB = _BYTE_TOK.get_vocab()
_CLASSIC_VOCAB = _CLASSIC_TOK.get_vocab()
_FALLBACK_VOCAB = _FALLBACK_TOK.get_vocab()
_WORDPIECE_VOCAB = _WORDPIECE_TOK.get_vocab()

_BYTE_MAPPER = _hf_byte_token_mapper(_BYTE_TOK)
_FALLBACK_MAPPER = _hf_byte_token_mapper(_FALLBACK_TOK)
assert _BYTE_MAPPER is not None and _FALLBACK_MAPPER is not None
assert _hf_byte_token_mapper(_CLASSIC_TOK) is None
assert StopStringCriteria._get_stop_string_matching_mode(_BYTE_TOK) == "byte_level"
assert StopStringCriteria._get_stop_string_matching_mode(_FALLBACK_TOK) == "byte_fallback"
assert StopStringCriteria._get_stop_string_matching_mode(_CLASSIC_TOK) is None


class _RealDecodeBackend:
    """decode adapter issuing LocalHFBackend.decode's exact decode call."""

    fingerprint = "cycle50-real-decode-adapter"

    def __init__(self, tokenizer: PreTrainedTokenizerFast, mapper) -> None:  # type: ignore[no-untyped-def]
        self._tokenizer = tokenizer
        self.hf_stop_token_bytes = mapper

    def decode(self, token_ids: tuple[int, ...]) -> str:
        return self._tokenizer.decode(
            list(token_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )


_BYTE_BACKEND = _RealDecodeBackend(_BYTE_TOK, _BYTE_MAPPER)
_CLASSIC_BACKEND = _RealDecodeBackend(_CLASSIC_TOK, None)
_FALLBACK_BACKEND = _RealDecodeBackend(_FALLBACK_TOK, _FALLBACK_MAPPER)
_WORDPIECE_BACKEND = _RealDecodeBackend(_WORDPIECE_TOK, None)

# Reference criteria cache: (id(tokenizer), stops) -> criterion or None
# (None = upstream ValueError at construction: no vocab token can overlap).
_CRITERIA_CACHE: dict[tuple[int, tuple[str, ...]], StopStringCriteria | None] = {}


def _reference_criterion(
    tokenizer: PreTrainedTokenizerFast, stops: tuple[str, ...]
) -> StopStringCriteria | None:
    key = (id(tokenizer), stops)
    if key not in _CRITERIA_CACHE:
        try:
            _CRITERIA_CACHE[key] = StopStringCriteria(tokenizer, list(stops))
        except ValueError:
            _CRITERIA_CACHE[key] = None
    return _CRITERIA_CACHE[key]


def _reference_first_step(
    tokenizer: PreTrainedTokenizerFast,
    stops: tuple[str, ...],
    prompt: tuple[int, ...],
    output: tuple[int, ...],
) -> int | None:
    """First output step where the real HF criteria fires, if any."""
    criterion = _reference_criterion(tokenizer, stops)
    if criterion is None:
        return None
    with torch.no_grad():
        for step in range(1, len(output) + 1):
            ids = torch.tensor([list(prompt) + list(output[:step])])
            if bool(criterion(ids, None)[0].item()):
                return step
    return None


def _lab_first_step(
    backend: _RealDecodeBackend,
    prompt: tuple[int, ...],
    output: tuple[int, ...],
    stop: str,
) -> int | None:
    """First firing step per the emulation; asserts the cell is available."""
    outcome = emulate_hf_stop_string(
        backend, prompt, output, stop, token_bytes=backend.hf_stop_token_bytes
    )
    assert outcome.available, outcome.detail
    assert outcome.returned_text is None  # HF always keeps the stop string
    if not outcome.fired:
        assert outcome.detail == "no_stop_within_cap"
        assert outcome.truncated_ids == tuple(output)
        return None
    step = int(outcome.detail.rsplit("_", 1)[1])
    assert outcome.detail == f"fired_at_step_{step}"
    assert outcome.truncated_ids == tuple(output[:step])
    return step


def _assert_agreement(
    test: unittest.TestCase,
    tokenizer: PreTrainedTokenizerFast,
    backend: _RealDecodeBackend,
    stop: str,
    prompt: tuple[int, ...],
    output: tuple[int, ...],
) -> None:
    """Assert emulation and real HF agree on (fired, step, truncated ids)."""
    expected = _reference_first_step(tokenizer, (stop,), prompt, output)
    actual = _lab_first_step(backend, prompt, output, stop)
    test.assertEqual(actual, expected, f"stop={stop!r} prompt={prompt} output={output}")


def _ids(vocab: dict[str, int], *pieces: str) -> tuple[int, ...]:
    return tuple(vocab[piece] for piece in pieces)


def backend_vocab(tokenizer: PreTrainedTokenizerFast) -> dict[str, int]:
    if tokenizer is _BYTE_TOK:
        return _BYTE_VOCAB
    if tokenizer is _CLASSIC_TOK:
        return _CLASSIC_VOCAB
    raise AssertionError("unexpected tokenizer")


def _assert_multi_agreement(
    test: unittest.TestCase,
    tokenizer: PreTrainedTokenizerFast,
    backend: _RealDecodeBackend,
    stops: tuple[str, ...],
    prompt: tuple[int, ...],
    output: tuple[int, ...],
) -> None:
    """Joint real criteria vs min-over-per-stop-emulation composition.

    Also asserts per-stop real-vs-lab agreement and that the winning-stop
    sets coincide (the observable part of "which stop fired").
    """
    per_real = {stop: _reference_first_step(tokenizer, (stop,), prompt, output) for stop in stops}
    for stop in stops:
        test.assertEqual(
            _lab_first_step(backend, prompt, output, stop),
            per_real[stop],
            f"per-stop disagreement for {stop!r}",
        )
    joint = _reference_first_step(tokenizer, stops, prompt, output)
    fired = {stop: step for stop, step in per_real.items() if step is not None}
    expected_joint = min(fired.values()) if fired else None
    test.assertEqual(joint, expected_joint, f"stops={stops!r}")
    if joint is not None:
        real_winners = {stop for stop, step in per_real.items() if step == joint}
        lab_winners = {
            stop for stop in stops if _lab_first_step(backend, prompt, output, stop) == joint
        }
        test.assertEqual(lab_winners, real_winners)
        # Same-step ties are outcome-identical: same completing token kept.
        first = next(iter(real_winners))
        for stop in real_winners:
            outcome = emulate_hf_stop_string(
                backend, prompt, output, stop, token_bytes=backend.hf_stop_token_bytes
            )
            first_outcome = emulate_hf_stop_string(
                backend, prompt, output, first, token_bytes=backend.hf_stop_token_bytes
            )
            test.assertEqual(outcome.truncated_ids, first_outcome.truncated_ids)


@unittest.skipUnless(HAS_HF, "the 'hf' extra is not installed")
class ByteBoundaryTests(unittest.TestCase):
    """UTF-8 fragments split across tokens (the transformers #46519 class)."""

    def test_ae_umlaut_split(self) -> None:
        # "ä" = U+00E4 = bytes C3 A4; pieces "Ã" (U+00C3) and "¤" (U+00A4).
        _assert_agreement(self, _BYTE_TOK, _BYTE_BACKEND, "ä", (), _ids(_BYTE_VOCAB, "Ã", "¤"))
        self.assertEqual(
            _reference_first_step(_BYTE_TOK, ("ä",), (), _ids(_BYTE_VOCAB, "Ã", "¤")), 2
        )

    def test_e_acute_split(self) -> None:
        # "é" = U+00E9 = bytes C3 A9; pieces "Ã" and "©" (U+00A9).
        _assert_agreement(self, _BYTE_TOK, _BYTE_BACKEND, "é", (), _ids(_BYTE_VOCAB, "Ã", "©"))

    def test_cafe_split_across_three_tokens(self) -> None:
        _assert_agreement(
            self, _BYTE_TOK, _BYTE_BACKEND, "café", (), _ids(_BYTE_VOCAB, "caf", "Ã", "©")
        )
        self.assertEqual(
            _reference_first_step(_BYTE_TOK, ("café",), (), _ids(_BYTE_VOCAB, "caf", "Ã", "©")),
            3,
        )

    def test_emoji_split_across_four_fragment_tokens(self) -> None:
        frag = _bytes_to_unicode()
        frags = tuple(frag[byte] for byte in "😀".encode())
        self.assertEqual(len(frags), 4)
        _assert_agreement(self, _BYTE_TOK, _BYTE_BACKEND, "😀", (), _ids(_BYTE_VOCAB, *frags))
        self.assertEqual(
            _reference_first_step(_BYTE_TOK, ("😀",), (), _ids(_BYTE_VOCAB, *frags)), 4
        )

    def test_fragment_overhang_off_prompt(self) -> None:
        # Prompt supplies the first fragment; the stop completes at step 1.
        _assert_agreement(
            self,
            _BYTE_TOK,
            _BYTE_BACKEND,
            "é",
            _ids(_BYTE_VOCAB, "Ã"),
            _ids(_BYTE_VOCAB, "©"),
        )
        self.assertEqual(
            _reference_first_step(
                _BYTE_TOK, ("é",), _ids(_BYTE_VOCAB, "Ã"), _ids(_BYTE_VOCAB, "©")
            ),
            1,
        )

    def test_whole_token_vs_fragments_agree(self) -> None:
        _assert_agreement(self, _BYTE_TOK, _BYTE_BACKEND, "é", (), _ids(_BYTE_VOCAB, "é"))
        _assert_agreement(self, _BYTE_TOK, _BYTE_BACKEND, "café", (), _ids(_BYTE_VOCAB, "café"))

    def test_lookalike_fragment_pair_does_not_fire(self) -> None:
        # "Ã¤" (U+00C3 U+00A4) is NOT "ä" (U+00E4): 4 bytes vs 2.
        _assert_agreement(self, _BYTE_TOK, _BYTE_BACKEND, "Ã¤", (), _ids(_BYTE_VOCAB, "Ã", "¤"))
        self.assertIsNone(
            _reference_first_step(_BYTE_TOK, ("Ã¤",), (), _ids(_BYTE_VOCAB, "Ã", "¤"))
        )

    def test_non_alphabet_token_fallback(self) -> None:
        # "€" (U+20AC) is outside the GPT-2 byte alphabet: both sides fall
        # back (upstream: cleaned string encoded; lab: decode encoded).
        _assert_agreement(self, _BYTE_TOK, _BYTE_BACKEND, "€", (), _ids(_BYTE_VOCAB, "€"))
        self.assertEqual(_reference_first_step(_BYTE_TOK, ("€",), (), _ids(_BYTE_VOCAB, "€")), 1)

    def test_cjk_token_fallback(self) -> None:
        _assert_agreement(self, _BYTE_TOK, _BYTE_BACKEND, "中", (), _ids(_BYTE_VOCAB, "中"))

    def test_mixed_fragment_and_whole(self) -> None:
        _assert_agreement(self, _BYTE_TOK, _BYTE_BACKEND, "é", (), _ids(_BYTE_VOCAB, "Ã", "é"))
        _assert_agreement(
            self, _BYTE_TOK, _BYTE_BACKEND, "café", (), _ids(_BYTE_VOCAB, "caf", "é", "x")
        )
        self.assertIsNone(
            _reference_first_step(_BYTE_TOK, ("café",), (), _ids(_BYTE_VOCAB, "caf", "é", "x"))
        )


@unittest.skipUnless(HAS_HF, "the 'hf' extra is not installed")
class UnicodeNormalizationTests(unittest.TestCase):
    NFC_CAFE = "café"
    NFD_CAFE = "café"

    def test_nfc_stop_matches_nfc_tokens(self) -> None:
        _assert_agreement(
            self,
            _BYTE_TOK,
            _BYTE_BACKEND,
            self.NFC_CAFE,
            (),
            _ids(_BYTE_VOCAB, "caf", "é"),
        )

    def test_nfd_stop_does_not_match_nfc_tokens(self) -> None:
        _assert_agreement(
            self,
            _BYTE_TOK,
            _BYTE_BACKEND,
            self.NFD_CAFE,
            (),
            _ids(_BYTE_VOCAB, "caf", "é"),
        )
        self.assertIsNone(
            _reference_first_step(_BYTE_TOK, (self.NFD_CAFE,), (), _ids(_BYTE_VOCAB, "caf", "é"))
        )

    def test_nfd_stop_matches_nfd_token_piece(self) -> None:
        # The "é" piece is NFD (e + U+0301), outside the byte alphabet, so
        # both sides take their cleaned-string/decode fallback paths.
        _assert_agreement(
            self,
            _BYTE_TOK,
            _BYTE_BACKEND,
            self.NFD_CAFE,
            (),
            _ids(_BYTE_VOCAB, "caf", "é"),
        )
        self.assertEqual(
            _reference_first_step(_BYTE_TOK, (self.NFD_CAFE,), (), _ids(_BYTE_VOCAB, "caf", "é")),
            2,
        )

    def test_zwj_emoji_stop_is_unconstructible(self) -> None:
        # No vocab token overlaps "👨\u200d👩\u200d👧": upstream raises
        # ValueError at construction; the lab runs and stays silent.
        stop = "👨\u200d👩\u200d👧"
        self.assertIsNone(_reference_first_step(_BYTE_TOK, (stop,), (), ()))
        self.assertIsNone(
            _lab_first_step(_BYTE_BACKEND, (), _ids(_BYTE_VOCAB, "the", "stop"), stop)
        )

    def test_lone_surrogate_stop_byte_path(self) -> None:
        # Neither side can UTF-8-encode a lone surrogate: the lab degrades
        # to token_text_unavailable, upstream raises UnicodeEncodeError in
        # the constructor (not the documented ValueError).
        outcome = emulate_hf_stop_string(
            _BYTE_BACKEND, (), _ids(_BYTE_VOCAB, "a"), _LONE_SURROGATE, token_bytes=_BYTE_MAPPER
        )
        self.assertFalse(outcome.available)
        self.assertEqual(outcome.detail, "token_text_unavailable")
        with self.assertRaises(UnicodeEncodeError):
            StopStringCriteria(_BYTE_TOK, [_LONE_SURROGATE])

    def test_lone_surrogate_stop_classic_path(self) -> None:
        # On the classic path a lone surrogate is a plain str unit: no
        # vocab token overlaps it, so upstream raises ValueError (mapped to
        # never-fires) and the lab runs silent.
        self.assertIsNone(_reference_first_step(_CLASSIC_TOK, (_LONE_SURROGATE,), (), ()))
        self.assertIsNone(
            _lab_first_step(
                _CLASSIC_BACKEND, (), _ids(_CLASSIC_VOCAB, "the", "stop"), _LONE_SURROGATE
            )
        )


@unittest.skipUnless(HAS_HF, "the 'hf' extra is not installed")
class MidTokenVsBoundaryTests(unittest.TestCase):
    def test_stop_mid_token_fires(self) -> None:
        # "b" occurs mid-token in "abc": the end-overlap with a dropped
        # trailing char still starts a match (the docstring's "impossible
        # to construct" class).
        _assert_agreement(self, _BYTE_TOK, _BYTE_BACKEND, "b", (), _ids(_BYTE_VOCAB, "abc"))
        self.assertEqual(_reference_first_step(_BYTE_TOK, ("b",), (), _ids(_BYTE_VOCAB, "abc")), 1)

    def test_stop_at_token_start_overhangs_end(self) -> None:
        # "ab" is a prefix of "abc": dropping the trailing "c" completes it.
        _assert_agreement(self, _BYTE_TOK, _BYTE_BACKEND, "ab", (), _ids(_BYTE_VOCAB, "abc"))

    def test_stop_spanning_two_tokens(self) -> None:
        _assert_agreement(self, _BYTE_TOK, _BYTE_BACKEND, "stop", (), _ids(_BYTE_VOCAB, "st", "op"))

    def test_repeated_stop_inside_one_token(self) -> None:
        _assert_agreement(self, _BYTE_TOK, _BYTE_BACKEND, "aa", (), _ids(_BYTE_VOCAB, "aaaa"))

    def test_stop_fires_at_first_completion_not_later(self) -> None:
        # "ab" completes at step 1; the later repeat must not move the step.
        step = _reference_first_step(_BYTE_TOK, ("ab",), (), _ids(_BYTE_VOCAB, "ab", "x", "ab"))
        self.assertEqual(step, 1)
        _assert_agreement(
            self, _BYTE_TOK, _BYTE_BACKEND, "ab", (), _ids(_BYTE_VOCAB, "ab", "x", "ab")
        )

    def test_classic_mid_token(self) -> None:
        _assert_agreement(
            self,
            _CLASSIC_TOK,
            _CLASSIC_BACKEND,
            "op",
            (),
            _ids(_CLASSIC_VOCAB, "stop"),
        )
        # "stop" inside the longer "xstopper".
        _assert_agreement(
            self,
            _CLASSIC_TOK,
            _CLASSIC_BACKEND,
            "stop",
            (),
            _ids(_CLASSIC_VOCAB, "xstopper"),
        )
        # "stop" as a prefix of "stopx": dropping "x" completes it.
        _assert_agreement(
            self,
            _CLASSIC_TOK,
            _CLASSIC_BACKEND,
            "stop",
            (),
            _ids(_CLASSIC_VOCAB, "stopx"),
        )


@unittest.skipUnless(HAS_HF, "the 'hf' extra is not installed")
class OverlappingMultiStopTests(unittest.TestCase):
    """One stop a prefix/suffix of another; joint vs per-stop composition."""

    def test_prefix_pair(self) -> None:
        _assert_multi_agreement(
            self,
            _BYTE_TOK,
            _BYTE_BACKEND,
            ("ab", "abc"),
            (),
            _ids(_BYTE_VOCAB, "x", "abc"),
        )

    def test_suffix_pair(self) -> None:
        _assert_multi_agreement(
            self,
            _BYTE_TOK,
            _BYTE_BACKEND,
            ("abc", "bc"),
            (),
            _ids(_BYTE_VOCAB, "x", "abc"),
        )

    def test_repeated_char_pair(self) -> None:
        _assert_multi_agreement(
            self,
            _BYTE_TOK,
            _BYTE_BACKEND,
            ("aa", "aaa"),
            (),
            _ids(_BYTE_VOCAB, "aaaa"),
        )

    def test_suffix_inside_longer(self) -> None:
        _assert_multi_agreement(
            self,
            _BYTE_TOK,
            _BYTE_BACKEND,
            ("stop", "top"),
            (),
            _ids(_BYTE_VOCAB, "las", "topper"),
        )

    def test_overlap_pair(self) -> None:
        _assert_multi_agreement(
            self,
            _BYTE_TOK,
            _BYTE_BACKEND,
            ("abab", "bab"),
            (),
            _ids(_BYTE_VOCAB, "a", "bab"),
        )

    def test_three_stops(self) -> None:
        _assert_multi_agreement(
            self,
            _BYTE_TOK,
            _BYTE_BACKEND,
            ("a", "ab", "abc"),
            (),
            _ids(_BYTE_VOCAB, "x", "abc"),
        )

    def test_mixed_matchable_and_unmatchable(self) -> None:
        # "\x00" is unconstructible alone; jointly it is simply inert.
        _assert_multi_agreement(
            self,
            _BYTE_TOK,
            _BYTE_BACKEND,
            ("ab", "\x00"),
            (),
            _ids(_BYTE_VOCAB, "x", "ab"),
        )

    def test_tied_stops_same_step(self) -> None:
        # "ab" and "b" both complete on the single token "ab".
        _assert_multi_agreement(
            self, _BYTE_TOK, _BYTE_BACKEND, ("ab", "b"), (), _ids(_BYTE_VOCAB, "ab")
        )
        joint = _reference_first_step(_BYTE_TOK, ("ab", "b"), (), _ids(_BYTE_VOCAB, "ab"))
        self.assertEqual(joint, 1)


@unittest.skipUnless(HAS_HF, "the 'hf' extra is not installed")
class WhitespaceStopTests(unittest.TestCase):
    STOPS = (" ", "  ", "\n", "\t", " \n", "a b", "ab ", " b", "the ")

    def test_whitespace_stops(self) -> None:
        seqs = [
            (),
            _ids(_BYTE_VOCAB, "the"),
            _ids(_BYTE_VOCAB, "a", " ", "b"),
            _ids(_BYTE_VOCAB, "  ", "x"),
            _ids(_BYTE_VOCAB, "a", "\n"),
            _ids(_BYTE_VOCAB, "\t", "\t"),
            _ids(_BYTE_VOCAB, " ", "\n"),
        ]
        for stop in self.STOPS:
            for output in seqs:
                with self.subTest(stop=stop, output=output):
                    _assert_agreement(self, _BYTE_TOK, _BYTE_BACKEND, stop, (), output)

    def test_newline_stop(self) -> None:
        step = _reference_first_step(_BYTE_TOK, ("\n",), (), _ids(_BYTE_VOCAB, "a", "\n", "b"))
        self.assertEqual(step, 2)
        _assert_agreement(
            self, _BYTE_TOK, _BYTE_BACKEND, "\n", (), _ids(_BYTE_VOCAB, "a", "\n", "b")
        )


@unittest.skipUnless(HAS_HF, "the 'hf' extra is not installed")
class EmptyTokenTests(unittest.TestCase):
    """Empty-string token representations on the byte path."""

    def test_empty_token_is_transparent(self) -> None:
        # ["a", "", "ab"] matches "aab": the walk passes through "".
        output = _ids(_BYTE_VOCAB, "a", "", "ab")
        self.assertEqual(_reference_first_step(_BYTE_TOK, ("aab",), (), output), 3)
        _assert_agreement(self, _BYTE_TOK, _BYTE_BACKEND, "aab", (), output)

    def test_empty_final_token_never_starts_match(self) -> None:
        # ["q", ""]: the empty final token cannot start a match.
        output = _ids(_BYTE_VOCAB, "q", "")
        _assert_agreement(self, _BYTE_TOK, _BYTE_BACKEND, "a", (), output)
        self.assertIsNone(_reference_first_step(_BYTE_TOK, ("a",), (), output))
        # ["a", ""]: fires at step 1 (the "a"), never at the empty token.
        output2 = _ids(_BYTE_VOCAB, "a", "")
        self.assertEqual(_reference_first_step(_BYTE_TOK, ("a",), (), output2), 1)
        _assert_agreement(self, _BYTE_TOK, _BYTE_BACKEND, "a", (), output2)

    def test_window_bound_single_stop(self) -> None:
        # Upstream truncates to the last len(stop) tokens: the "a" four
        # tokens back cannot participate through the empties.
        output = _ids(_BYTE_VOCAB, "a", "", "", "", "b")
        _assert_agreement(self, _BYTE_TOK, _BYTE_BACKEND, "ab", (), output)
        self.assertIsNone(_reference_first_step(_BYTE_TOK, ("ab",), (), output))

    def test_multi_stop_window_divergence(self) -> None:
        # REAL DIVERGENCE (docstring item 2): the joint criteria bounds the
        # walk by the GLOBAL max stop length, so the short stop's walk
        # reaches the "a" through the empties; each per-stop lab cell uses
        # its own shorter window and stays silent.
        output = _ids(_BYTE_VOCAB, "a", "", "", "b")
        joint = _reference_first_step(_BYTE_TOK, ("ab", "abcdef"), (), output)
        self.assertEqual(joint, 4)
        # Single-stop behavior still agrees on both sides.
        self.assertIsNone(_reference_first_step(_BYTE_TOK, ("ab",), (), output))
        self.assertIsNone(_lab_first_step(_BYTE_BACKEND, (), output, "ab"))
        self.assertIsNone(_reference_first_step(_BYTE_TOK, ("abcdef",), (), output))
        self.assertIsNone(_lab_first_step(_BYTE_BACKEND, (), output, "abcdef"))


@unittest.skipUnless(HAS_HF, "the 'hf' extra is not installed")
class EmptyStopTests(unittest.TestCase):
    def test_empty_stop_string_divergence(self) -> None:
        # REAL DIVERGENCE (docstring item 1): upstream accepts "" and fires
        # at step 1 for any input (vacuous target length); the emulation
        # rejects "" as invalid_emulation_input.
        for tokenizer in (_BYTE_TOK, _CLASSIC_TOK, _FALLBACK_TOK):
            with self.subTest(mode=tokenizer):
                self.assertEqual(_reference_first_step(tokenizer, ("",), (), (0,)), 1)
        outcome = emulate_hf_stop_string(_BYTE_BACKEND, (), (0,), "", token_bytes=_BYTE_MAPPER)
        self.assertFalse(outcome.available)
        self.assertFalse(outcome.fired)
        self.assertEqual(outcome.detail, "invalid_emulation_input")

    def test_empty_stop_list_raises_upstream(self) -> None:
        # No lab counterpart exists (single-stop API); upstream raises at
        # construction on max() of an empty sequence.
        with self.assertRaises(ValueError):
            StopStringCriteria(_BYTE_TOK, [])


@unittest.skipUnless(HAS_HF, "the 'hf' extra is not installed")
class ConstructorStrictnessTests(unittest.TestCase):
    def test_unmatchable_stop(self) -> None:
        # "\x00" overlaps no vocab token: upstream raises ValueError at
        # construction (mapped to never-fires); the lab runs and stays
        # silent. Outcome-agreement; strictness differs.
        for tokenizer, backend in (
            (_BYTE_TOK, _BYTE_BACKEND),
            (_CLASSIC_TOK, _CLASSIC_BACKEND),
        ):
            with self.subTest(tokenizer=tokenizer):
                self.assertIsNone(_reference_first_step(tokenizer, ("\x00",), (), ()))
                self.assertIsNone(
                    _lab_first_step(backend, (), _ids(backend_vocab(tokenizer), "a"), "\x00")
                )


@unittest.skipUnless(HAS_HF, "the 'hf' extra is not installed")
class TieBreakTests(unittest.TestCase):
    """DOCUMENTED (repo) vs ACTUAL (transformers 5.19.0) tie-break rule.

    DOCUMENTED: HF exposes no per-stop winner -- ``__call__`` returns a bare
    per-sequence any-match bool, so there is no first-listed-wins rule; the
    earliest-completing stop wins by construction of the per-token check,
    and same-step ties are outcome-identical (the same completing token is
    kept either way).
    ACTUAL: ``__call__`` ends with ``return torch.any(string_matches,
    dim=-1)`` -- no selection code exists; which stop matched is
    unobservable from the API.
    """

    def test_no_list_order_dependence(self) -> None:
        output = _ids(_BYTE_VOCAB, "ab")
        for stops in (("ab", "b"), ("b", "ab")):
            with self.subTest(stops=stops):
                self.assertEqual(_reference_first_step(_BYTE_TOK, stops, (), output), 1)

    def test_api_returns_bool_only(self) -> None:
        # The real criterion's return carries no stop identity: a single
        # per-sequence bool tensor.
        criterion = _reference_criterion(_BYTE_TOK, ("ab", "b"))
        assert criterion is not None
        with torch.no_grad():
            result = criterion(torch.tensor([list(_ids(_BYTE_VOCAB, "ab"))]), None)
        self.assertEqual(result.shape, torch.Size([1]))
        self.assertTrue(bool(result[0].item()))

    def test_tied_stops_outcome_identical(self) -> None:
        output = _ids(_BYTE_VOCAB, "ab")
        outcomes = [
            emulate_hf_stop_string(_BYTE_BACKEND, (), output, stop, token_bytes=_BYTE_MAPPER)
            for stop in ("ab", "b")
        ]
        for outcome in outcomes:
            self.assertTrue(outcome.available and outcome.fired)
            self.assertIsNone(outcome.returned_text)
        self.assertEqual(outcomes[0].truncated_ids, outcomes[1].truncated_ids)
        self.assertEqual(outcomes[0].truncated_ids, output)

    def test_earliest_completing_stop_wins(self) -> None:
        # "ab" completes at step 1, "bc" at step 2: the joint criterion
        # fires at 1 either way -- by construction, not by list order.
        output = _ids(_BYTE_VOCAB, "ab", "c")
        for stops in (("ab", "bc"), ("bc", "ab")):
            with self.subTest(stops=stops):
                _assert_multi_agreement(self, _BYTE_TOK, _BYTE_BACKEND, stops, (), output)
                self.assertEqual(_reference_first_step(_BYTE_TOK, stops, (), output), 1)

    def test_tie_across_different_strings(self) -> None:
        # "abc" and "bc" both complete on the token "abc" at step 2.
        output = _ids(_BYTE_VOCAB, "x", "abc")
        _assert_multi_agreement(self, _BYTE_TOK, _BYTE_BACKEND, ("abc", "bc"), (), output)
        self.assertEqual(_reference_first_step(_BYTE_TOK, ("abc", "bc"), (), output), 2)


@unittest.skipUnless(HAS_HF, "the 'hf' extra is not installed")
class LongInputTests(unittest.TestCase):
    def test_very_long_stop_fires(self) -> None:
        stop = "ab" * 150
        output = _ids(_BYTE_VOCAB, "ab") * 150
        self.assertEqual(_reference_first_step(_BYTE_TOK, (stop,), (), output), 150)
        _assert_agreement(self, _BYTE_TOK, _BYTE_BACKEND, stop, (), output)

    def test_very_long_stop_no_fire(self) -> None:
        stop = "ab" * 150 + "x"
        output = _ids(_BYTE_VOCAB, "ab") * 150
        self.assertIsNone(_reference_first_step(_BYTE_TOK, (stop,), (), output))
        _assert_agreement(self, _BYTE_TOK, _BYTE_BACKEND, stop, (), output)

    def test_very_long_stop_list(self) -> None:
        alphabet = "abcd thsop"
        stops = tuple(
            list(alphabet)
            + [c1 + c2 for c1 in "abcdef" for c2 in alphabet][:72]
            + [c1 + c2 + c3 for c1 in "ab" for c2 in "ab" for c3 in alphabet][:36]
        )
        self.assertGreater(len(stops), 100)  # 10 + 60 + 36
        output = _ids(_BYTE_VOCAB, "the", " ", "stop", "x")
        prompt = _ids(_BYTE_VOCAB, "a")
        _assert_multi_agreement(self, _BYTE_TOK, _BYTE_BACKEND, stops, prompt, output)


@unittest.skipUnless(HAS_HF, "the 'hf' extra is not installed")
class ClassicSpacingDivergenceTests(unittest.TestCase):
    """WordPiece-decoder spacing: cleaned vocab != isolated per-token decode.

    REAL DIVERGENCES (docstring item 3). The WordPiece decoder inserts an
    inter-token space in multi-token decode (``decode([a, b]) == "a b"``)
    but not in isolated single-token decode (``decode([a]) + decode([b]) ==
    "ab"``); upstream's static-prefix cleaning bakes the spaced form into
    the match table while the emulation walks the unspaced concatenation.
    """

    def test_emulation_false_positive_on_space(self) -> None:
        # True decoded text is "a b": no "ab" is ever emitted, so the real
        # criterion stays silent while the emulation fires at step 2.
        output = _ids(_WORDPIECE_VOCAB, "a", "b")
        self.assertEqual(
            _WORDPIECE_BACKEND.decode(output), "a b", "true decoded text has the space"
        )
        self.assertIsNone(_reference_first_step(_WORDPIECE_TOK, ("ab",), (), output))
        self.assertEqual(_lab_first_step(_WORDPIECE_BACKEND, (), output, "ab"), 2)

    def test_real_fires_early_on_cleaning_artifact(self) -> None:
        # The cleaned " aaa" starts with a space, so the real criterion
        # fires at step 1 -- before any space exists in the true decoded
        # text "aaa". The emulation fires at the true position, step 2.
        output = _ids(_WORDPIECE_VOCAB, "aaa", " ")
        self.assertEqual(_WORDPIECE_BACKEND.decode(output[:1]), "aaa")
        self.assertEqual(_reference_first_step(_WORDPIECE_TOK, (" ",), (), output), 1)
        self.assertEqual(_lab_first_step(_WORDPIECE_BACKEND, (), output, " "), 2)


@unittest.skipUnless(HAS_HF, "the 'hf' extra is not installed")
class NonContiguousVocabTests(unittest.TestCase):
    def test_high_id_treated_as_oov_by_lab(self) -> None:
        # REAL DIVERGENCE (docstring item 4): vocab ids {0..13} plus "zzz"
        # at id 50. The lab's byte mapper bounds by len(vocab) == 15 and
        # treats id 50 as walk-killing OOV; upstream sizes its table by
        # max(ids) and matches "zzz" normally.
        pieces = _fallback_pieces()
        vocab = {piece: index for index, piece in enumerate(pieces)}
        vocab["zzz"] = 50
        tok = Tokenizer(models.BPE(vocab=vocab, merges=_MERGES_ABCDEF, unk_token="[UNK]"))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tok, clean_up_tokenization_spaces=False
        )
        mapper = _hf_byte_token_mapper(tokenizer)
        assert mapper is not None
        backend = _RealDecodeBackend(tokenizer, mapper)
        self.assertEqual(_reference_first_step(tokenizer, ("zzz",), (), (50,)), 1)
        self.assertIsNone(_lab_first_step(backend, (), (50,), "zzz"))


@unittest.skipUnless(HAS_HF, "the 'hf' extra is not installed")
class FallbackDecoderTests(unittest.TestCase):
    """ByteFallback decoder: explicit <0xNN> tokens plus decode fallback."""

    def test_hex_token_fires(self) -> None:
        output = _ids(_FALLBACK_VOCAB, "<0x41>")
        self.assertEqual(_reference_first_step(_FALLBACK_TOK, ("A",), (), output), 1)
        _assert_agreement(self, _FALLBACK_TOK, _FALLBACK_BACKEND, "A", (), output)

    def test_hex_pair_spells_stop(self) -> None:
        output = _ids(_FALLBACK_VOCAB, "<0x41>", "<0x42>")
        _assert_agreement(self, _FALLBACK_TOK, _FALLBACK_BACKEND, "AB", (), output)
        self.assertEqual(_reference_first_step(_FALLBACK_TOK, ("AB",), (), output), 2)

    def test_non_hex_token_uses_decode_fallback(self) -> None:
        # "x" is not <0xNN>: the mapper returns None and both sides fall
        # back (upstream: cleaned string; lab: isolated decode).
        self.assertIsNone(_FALLBACK_MAPPER(_FALLBACK_VOCAB["x"]))
        output = _ids(_FALLBACK_VOCAB, "x")
        _assert_agreement(self, _FALLBACK_TOK, _FALLBACK_BACKEND, "x", (), output)

    def test_mixed_hex_and_plain(self) -> None:
        output = _ids(_FALLBACK_VOCAB, "<0x41>", "x")
        _assert_agreement(self, _FALLBACK_TOK, _FALLBACK_BACKEND, "Ax", (), output)
        self.assertEqual(_reference_first_step(_FALLBACK_TOK, ("Ax",), (), output), 2)


def _valid_stops(tokenizer: PreTrainedTokenizerFast, stops: list[str]) -> list[str]:
    """Stops the real criteria can construct (upstream ValueError otherwise)."""
    return [stop for stop in stops if _reference_criterion(tokenizer, (stop,)) is not None]


@unittest.skipUnless(HAS_HF, "the 'hf' extra is not installed")
class BytePathFuzzTests(unittest.TestCase):
    """Seeded differential fuzz on the byte-level path (168 cases)."""

    STOPS = [
        "ab",
        "abc",
        "a",
        "aa",
        "aaa",
        " ",
        "  ",
        "\n",
        "ä",
        "é",
        "café",
        "😀",
        "€",
        "中",
        "A",
        "AA",
        "stop",
        "op",
        "topper",
        "b",
        "©",
        "â",
        "caf",
        "\t",
        "ab ",
        "the",
        "st",
    ]
    NFD_CAFE = "cafe" + chr(0x301)

    def test_byte_path_fuzz(self) -> None:
        rng = random.Random(20261050)  # noqa: S311 -- deterministic test seed
        stops = _valid_stops(_BYTE_TOK, [*self.STOPS, self.NFD_CAFE])
        self.assertGreater(len(stops), 20)
        ids = sorted(_BYTE_VOCAB.values())
        oov_id = len(_BYTE_VOCAB)  # out-of-vocabulary: dummy row upstream
        trials = 0
        for stop in stops:
            for _ in range(6):
                prompt = tuple(rng.choice(ids) for _ in range(rng.randint(0, 3)))
                output = tuple(rng.choice(ids) for _ in range(rng.randint(0, 8)))
                if rng.random() < 0.15:
                    output = output + (oov_id,)
                with self.subTest(stop=stop, prompt=prompt, output=output):
                    _assert_agreement(self, _BYTE_TOK, _BYTE_BACKEND, stop, prompt, output)
                trials += 1
        self.assertEqual(trials, len(stops) * 6)


@unittest.skipUnless(HAS_HF, "the 'hf' extra is not installed")
class ClassicPathFuzzTests(unittest.TestCase):
    """Seeded differential fuzz on the classic path (144 cases).

    Uses the Metaspace-decoder tokenizer, whose decode is concatenative,
    so the fuzz isolates the match algorithm from the documented
    cleaned-vocab-vs-decode spacing approximation (pinned separately by
    ``ClassicSpacingDivergenceTests``).
    """

    STOPS = [
        "ab",
        "abc",
        "a",
        " ",
        "the",
        "stop",
        "op",
        "x",
        "zz",
        "aa",
        "aaa",
        "\n",
        "ed",
        "es",
        "he",
        "to",
        "st",
        "q",
        "  ",
        "b ",
        "top",
        "las",
        "pped",
        "c",
    ]

    def test_classic_path_fuzz(self) -> None:
        rng = random.Random(20261051)  # noqa: S311 -- deterministic test seed
        stops = _valid_stops(_CLASSIC_TOK, self.STOPS)
        self.assertGreater(len(stops), 15)
        ids = sorted(_CLASSIC_VOCAB.values())
        trials = 0
        for stop in stops:
            for _ in range(6):
                prompt = tuple(rng.choice(ids) for _ in range(rng.randint(0, 3)))
                output = tuple(rng.choice(ids) for _ in range(rng.randint(0, 8)))
                with self.subTest(stop=stop, prompt=prompt, output=output):
                    _assert_agreement(self, _CLASSIC_TOK, _CLASSIC_BACKEND, stop, prompt, output)
                trials += 1
        self.assertEqual(trials, len(stops) * 6)


@unittest.skipUnless(HAS_HF, "the 'hf' extra is not installed")
class FallbackPathFuzzTests(unittest.TestCase):
    """Seeded differential fuzz on the byte-fallback path (60 cases)."""

    STOPS = ["A", "B", "AB", "a", " ", "x", "AA", "\n", "ab", "Ax"]

    def test_fallback_path_fuzz(self) -> None:
        rng = random.Random(20261052)  # noqa: S311 -- deterministic test seed
        stops = _valid_stops(_FALLBACK_TOK, self.STOPS)
        self.assertGreater(len(stops), 5)
        ids = sorted(_FALLBACK_VOCAB.values())
        trials = 0
        for stop in stops:
            for _ in range(6):
                prompt = tuple(rng.choice(ids) for _ in range(rng.randint(0, 2)))
                output = tuple(rng.choice(ids) for _ in range(rng.randint(0, 7)))
                with self.subTest(stop=stop, prompt=prompt, output=output):
                    _assert_agreement(self, _FALLBACK_TOK, _FALLBACK_BACKEND, stop, prompt, output)
                trials += 1
        self.assertEqual(trials, len(stops) * 6)


@unittest.skipUnless(HAS_HF, "the 'hf' extra is not installed")
class MultiStopFuzzTests(unittest.TestCase):
    """Seeded multi-stop fuzz: joint real vs min-over-per-stop lab (55 cases)."""

    POOL = [
        "ab",
        "abc",
        "a",
        "aa",
        " ",
        "\n",
        "é",
        "café",
        "stop",
        "op",
        "topper",
        "b",
        "the",
        "st",
        "A",
        "AA",
        "aaa",
        "x",
    ]

    def test_multi_stop_fuzz(self) -> None:
        rng = random.Random(20261053)  # noqa: S311 -- deterministic test seed
        pool = _valid_stops(_BYTE_TOK, self.POOL)
        self.assertGreater(len(pool), 10)
        ids = sorted(_BYTE_VOCAB.values())
        trials = 0
        for _ in range(55):
            stops = tuple(rng.sample(pool, rng.randint(2, 4)))
            prompt = tuple(rng.choice(ids) for _ in range(rng.randint(0, 2)))
            output = tuple(rng.choice(ids) for _ in range(rng.randint(1, 7)))
            with self.subTest(stops=stops, prompt=prompt, output=output):
                _assert_multi_agreement(self, _BYTE_TOK, _BYTE_BACKEND, stops, prompt, output)
            trials += 1
        self.assertEqual(trials, 55)


if __name__ == "__main__":
    unittest.main()
