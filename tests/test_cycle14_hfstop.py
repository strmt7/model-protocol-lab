"""Cycle 14: differential verification of ``emulate_hf_stop_string``.

Compares ``backends.emulate_hf_stop_string`` against the real
``transformers.StopStringCriteria`` (transformers 5.19.0, the version
pinned in the repo venv) used as the reference oracle, over adversarial
inputs plus a seeded fuzz sweep -- on both the classic string path and
the byte-space path (byte-level decoders).

The fake tokenizers below are minimal ``PreTrainedTokenizerBase``
stand-ins: on the classic path the cleaned vocab strings are the tokens
themselves (``convert_tokens_to_string`` concatenates; the static
``"abcdef"`` prefix round-trips through single-char tokens), and on the
byte path a ``ByteLevel``-named decoder selects upstream's byte-space
table build. The fake backend decodes each id to its token string, so the
emulation's per-token decodes coincide exactly with the oracle's cleaned
vocab -- isolating the stop algorithm from the documented
cleaned-vocab-vs-decode approximation.

Verified properties (each pinned by at least one case below):

- byte-space matching is faithful: the stop is UTF-8-encoded and matched
  against raw token bytes, so a stop split across UTF-8-fragment tokens
  (the transformers #46519 class) fires in byte space while the classic
  path stays silent;
- stop strings are KEPT: ``truncated_ids`` ends with the completing
  token and ``returned_text`` stays None (vs vLLM/SGLang stripping);
- the prompt supplies overhang context (a stop spanning the
  prompt/output boundary fires), while a stop fully inside the prompt
  never fires (it must overlap the final generated token);
- multi-stop: upstream reports a per-sequence any-match bool -- there is
  no first-listed-wins rule in HF; the earliest-completing stop wins by
  construction of the per-token check, and ties are outcome-identical
  (both kept through the same completing token);
- edge cases: stop at position 0, empty output, unicode multi-byte,
  overlapping stops, stop longer than the output;
- empty token representations are transparent to the match walk
  (regression: the port used to stay silent while upstream fired at the
  completing token);
- the walk is bounded to the last ``len(stop)`` representations,
  mirroring upstream's ``input_ids[:, -maximum_token_len:]`` truncation
  (matters only with empty tokens consuming window budget);
- out-of-vocabulary ids on the byte path kill the walk, mirroring
  upstream clamping them to the dummy table row.
"""

from __future__ import annotations

import importlib.util
import random
import unittest
from threading import Event
from types import SimpleNamespace

from model_protocol_lab.backends import (
    _hf_byte_token_mapper,
    emulate_hf_stop_string,
)
from model_protocol_lab.core import Generation

HAS_HF = all(importlib.util.find_spec(name) is not None for name in ("transformers", "torch"))


class _FakeStringTokenizer:
    """Classic path: cleaned vocab strings are the tokens themselves."""

    def __init__(self, tokens: list[str]) -> None:
        self._tokens = list(tokens)
        self._vocab = {token: index for index, token in enumerate(self._tokens)}

    @property
    def backend_tokenizer(self):  # type: ignore[no-untyped-def]
        return None

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def _convert_id_to_token(self, index: int) -> str:
        return self._tokens[index]

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return "".join(tokens)

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        return {"input_ids": [self._vocab[char] for char in text]}


class _FakeByteTokenizer(_FakeStringTokenizer):
    """Byte-level path: a ``ByteLevel``-named decoder selects the byte table."""

    @property
    def backend_tokenizer(self):  # type: ignore[no-untyped-def]
        return SimpleNamespace(decoder=type("ByteLevel", (), {})())

    def convert_ids_to_tokens(self, index: int) -> str:
        return self._tokens[index]


class _TextBackend:
    """Backend decoding each id to its token string (concatenative)."""

    fingerprint = "cycle14-hfstop-not-a-model"

    def __init__(self, tokens: list[str]) -> None:
        self._texts = dict(enumerate(tokens))

    def encode(self, text: str) -> tuple[int, ...]:
        raise AssertionError("encode unused in stop tests")

    def decode(self, token_ids: tuple[int, ...]) -> str:
        return "".join(self._texts[token] for token in token_ids)

    def generate(
        self,
        input_ids: tuple[int, ...],
        *,
        max_new_tokens: int,
        seed: int,
        cancel: Event,
    ) -> Generation:
        raise AssertionError("generate unused in stop tests")


def _with_prefix_tokens(tokens: list[str]) -> list[str]:
    """Ensure the single-char a..f tokens HF's static prefix needs."""
    return list(dict.fromkeys(["a", "b", "c", "d", "e", "f", *tokens]))


def _reference_first_step(
    tokenizer: _FakeStringTokenizer,
    stops: list[str],
    prompt: tuple[int, ...],
    output: tuple[int, ...],
) -> int | None:
    """First output step where the real HF criteria fires, if any."""
    import torch
    from transformers.generation.stopping_criteria import StopStringCriteria

    try:
        criterion = StopStringCriteria(tokenizer, stops)
    except ValueError:
        # Upstream raises at construction when no vocab token can overlap
        # the stop string; generate() would never run, so the reference
        # outcome is "never fires".
        return None
    with torch.no_grad():
        for step in range(1, len(output) + 1):
            ids = torch.tensor([list(prompt) + list(output[:step])])
            if bool(criterion(ids, None)[0].item()):
                return step
    return None


def _check_agreement(
    test: unittest.TestCase,
    tokens: list[str],
    stop: str,
    prompt: tuple[int, ...],
    output: tuple[int, ...],
    *,
    byte_mode: bool = False,
) -> None:
    """Assert the emulation agrees with real HF on (fired, step, ids)."""
    full = _with_prefix_tokens(tokens)
    base = len(full)
    remap = {index: full.index(token) for index, token in enumerate(tokens)}
    # Ids at/above len(tokens) stay out-of-vocabulary.
    prompt = tuple(remap[i] if i < len(tokens) else base + (i - len(tokens)) for i in prompt)
    output = tuple(remap[i] if i < len(tokens) else base + (i - len(tokens)) for i in output)
    tokenizer = _FakeByteTokenizer(full) if byte_mode else _FakeStringTokenizer(full)
    backend = _TextBackend(full)
    token_bytes = _hf_byte_token_mapper(tokenizer) if byte_mode else None
    if byte_mode:
        test.assertIsNotNone(token_bytes)

    expected_step = _reference_first_step(tokenizer, [stop], prompt, output)
    outcome = emulate_hf_stop_string(backend, prompt, output, stop, token_bytes=token_bytes)
    test.assertTrue(outcome.available)
    test.assertEqual(outcome.fired, expected_step is not None)
    test.assertIsNone(outcome.returned_text)  # HF always keeps the stop string
    if expected_step is None:
        test.assertEqual(outcome.detail, "no_stop_within_cap")
        test.assertEqual(outcome.truncated_ids, tuple(output))
    else:
        test.assertEqual(outcome.detail, f"fired_at_step_{expected_step}")
        test.assertEqual(outcome.truncated_ids, tuple(output[:expected_step]))


@unittest.skipUnless(HAS_HF, "the 'hf' extra is not installed")
class HfDocstringCasesTests(unittest.TestCase):
    """The nine match/no-match examples from the StopStringCriteria docstring."""

    TOKENS = [
        "st",
        "op",
        "stop",
        "opera",
        "sto",
        "pper",
        "las",
        "topper",
        "at",
        "tion",
        "s",
        "to",
        "pped",
    ]

    def test_docstring_positive_cases_fire(self) -> None:
        for tokens in (
            ["st", "op"],
            ["stop"],
            ["st", "opera"],
            ["sto", "pper"],
            ["las", "topper"],
            ["s", "to", "pped"],
        ):
            with self.subTest(tokens=tokens):
                ids = tuple(self.TOKENS.index(t) for t in tokens)
                _check_agreement(self, self.TOKENS, "stop", (), ids)

    def test_docstring_negative_cases_stay_silent(self) -> None:
        # The stop string does not overlap the final token: upstream
        # deliberately does not fire.
        for tokens in (["stop", "at"], ["st", "op", "at"], ["st", "opera", "tion"]):
            with self.subTest(tokens=tokens):
                ids = tuple(self.TOKENS.index(t) for t in tokens)
                _check_agreement(self, self.TOKENS, "stop", (), ids)


@unittest.skipUnless(HAS_HF, "the 'hf' extra is not installed")
class HfStopDispositionTests(unittest.TestCase):
    """HF keeps the stop string; prompt/output boundary behavior."""

    def test_stop_kept_not_stripped(self) -> None:
        tokens = ["a", "b", "STOP", "c"]
        backend = _TextBackend(_with_prefix_tokens(tokens))
        full = _with_prefix_tokens(tokens)
        output = tuple(full.index(t) for t in tokens)
        outcome = emulate_hf_stop_string(backend, (), output, "STOP")
        self.assertTrue(outcome.available and outcome.fired)
        self.assertEqual(outcome.detail, "fired_at_step_3")
        # The completing token stays; nothing is stripped from the text.
        self.assertEqual(outcome.truncated_ids, output[:3])
        self.assertIsNone(outcome.returned_text)
        self.assertEqual(backend.decode(outcome.truncated_ids), "abSTOP")

    def test_stop_at_position_zero(self) -> None:
        _check_agreement(self, ["STOP", "x"], "STOP", (), (0, 1))

    def test_empty_output_never_fires(self) -> None:
        tokens = _with_prefix_tokens(["STOP"])
        backend = _TextBackend(tokens)
        outcome = emulate_hf_stop_string(backend, (), (), "STOP")
        self.assertTrue(outcome.available)
        self.assertFalse(outcome.fired)
        self.assertEqual(outcome.detail, "no_stop_within_cap")

    def test_stop_inside_prompt_does_not_fire(self) -> None:
        # The stop must overlap the final *generated* token.
        _check_agreement(self, ["stop", "x"], "stop", (0,), (1,))

    def test_stop_spanning_prompt_boundary_fires(self) -> None:
        # Prompt supplies overhang context, like HF's full-input_ids check.
        _check_agreement(self, ["st", "op"], "stop", (0,), (1,))

    def test_overhang_off_prompt_start(self) -> None:
        _check_agreement(self, ["s", "to", "pped"], "stop", (0, 1), (2,))


@unittest.skipUnless(HAS_HF, "the 'hf' extra is not installed")
class HfEdgeCaseTests(unittest.TestCase):
    def test_unicode_multibyte(self) -> None:
        _check_agreement(self, ["caf", "é", "x"], "café", (), (0, 1, 2))
        _check_agreement(self, ["😀", "x"], "😀", (), (0, 1))

    def test_overlapping_stops(self) -> None:
        _check_agreement(self, ["aaa"], "aa", (), (0,))
        _check_agreement(self, ["xaa", "aab"], "aab", (), (0, 1))

    def test_stop_longer_than_output(self) -> None:
        _check_agreement(self, ["stop"], "stopstop", (), (0,))
        _check_agreement(self, ["st", "op"], "stopstop!", (), (0, 1))

    def test_mid_token_occurrence_fires(self) -> None:
        # The stop need only occur in the final token ("or" in " world").
        _check_agreement(self, [" world"], "or", (), (0,))
        _check_agreement(self, ["xstopper"], "stop", (), (0,))


@unittest.skipUnless(HAS_HF, "the 'hf' extra is not installed")
class HfEmptyTokenRegressionTests(unittest.TestCase):
    """Empty token representations are transparent to the match walk.

    Regression: the port used to ``break`` the walk on an empty
    representation, staying silent where upstream fires (valid positions
    ``1..len(stop)-1``, zero length contribution).
    """

    def test_empty_token_is_transparent(self) -> None:
        # ["s", "", "topper"] matches "stop": upstream fires at step 3.
        _check_agreement(self, ["s", "", "topper"], "stop", (), (0, 1, 2))

    def test_window_bound_with_empty_tokens(self) -> None:
        # Upstream truncates to the last len(stop) tokens before matching,
        # so the "a" three empties back cannot participate: no fire.
        _check_agreement(self, ["a", "", "", "", "b"], "ab", (), (0, 1, 2, 3, 4))

    def test_empty_final_token_never_starts_a_match(self) -> None:
        _check_agreement(self, ["s", ""], "stop", (), (0, 1))


@unittest.skipUnless(HAS_HF, "the 'hf' extra is not installed")
class HfOOVByteTests(unittest.TestCase):
    """Out-of-vocabulary ids on the byte path kill the walk (dummy row)."""

    TOKENS = ["a", "b"]

    def test_oov_kills_walk(self) -> None:
        # ["a", OOV, "b"] with stop "ab": the walk from "b" dies at OOV.
        _check_agreement(self, self.TOKENS, "ab", (), (0, 2, 1), byte_mode=True)

    def test_oov_as_final_token_never_fires(self) -> None:
        _check_agreement(self, self.TOKENS, "ab", (), (0, 2), byte_mode=True)

    def test_oov_far_from_stop_is_inert(self) -> None:
        _check_agreement(self, self.TOKENS, "ab", (), (2, 0, 1), byte_mode=True)


@unittest.skipUnless(HAS_HF, "the 'hf' extra is not installed")
class HfMultiStopTests(unittest.TestCase):
    """Upstream has no first-listed-wins rule: any-match bool per sequence."""

    def test_earliest_completing_stop_wins(self) -> None:
        tokens = _with_prefix_tokens(["ab", "q", "b"])
        backend = _TextBackend(tokens)
        tokenizer = _FakeStringTokenizer(tokens)
        output = tuple(tokens.index(t) for t in ["ab", "q", "b"])
        # "ab" completes at step 1; "qb" only at step 3. Upstream checks all
        # stops every step and reports a bare per-sequence bool, so list
        # order cannot matter: the joint criteria fires at step 1 either
        # way -- there is no first-listed-wins rule in HF.
        for stops in (["ab", "qb"], ["qb", "ab"]):
            with self.subTest(stops=stops):
                self.assertEqual(_reference_first_step(tokenizer, stops, (), output), 1)
        first = emulate_hf_stop_string(backend, (), output, "ab")
        second = emulate_hf_stop_string(backend, (), output, "qb")
        self.assertEqual(first.detail, "fired_at_step_1")
        self.assertEqual(second.detail, "fired_at_step_3")
        # The lab's per-stop cells recover the joint outcome via the
        # minimum firing step -- no list-order rule needed.
        self.assertEqual(
            min(
                int(first.detail.rsplit("_", 1)[1]),
                int(second.detail.rsplit("_", 1)[1]),
            ),
            1,
        )

    def test_tied_stops_are_outcome_identical(self) -> None:
        tokens = _with_prefix_tokens(["ab"])
        backend = _TextBackend(tokens)
        tokenizer = _FakeStringTokenizer(tokens)
        output = (tokens.index("ab"),)
        joint = _reference_first_step(tokenizer, ["ab", "b"], (), output)
        self.assertEqual(joint, 1)
        for stop in ("ab", "b"):
            outcome = emulate_hf_stop_string(backend, (), output, stop)
            self.assertTrue(outcome.fired)
            self.assertEqual(outcome.truncated_ids, output)
            self.assertIsNone(outcome.returned_text)


@unittest.skipUnless(HAS_HF, "the 'hf' extra is not installed")
class HfByteSpaceTests(unittest.TestCase):
    """Byte-space matching: UTF-8 fragments fire where the string path is silent."""

    def test_fragment_stop_fires_in_byte_space(self) -> None:
        # "ä" = b"\xc3\xa4" split across "Ã" (U+00C3) and "¤" (U+00A4).
        tokens = ["Ã", "¤", "x"]
        full = _with_prefix_tokens(tokens)
        backend = _TextBackend(full)
        tokenizer = _FakeByteTokenizer(full)
        mapper = _hf_byte_token_mapper(tokenizer)
        self.assertIsNotNone(mapper)
        output = tuple(full.index(t) for t in tokens)
        classic = emulate_hf_stop_string(backend, (), output, "ä")
        self.assertTrue(classic.available)
        self.assertFalse(classic.fired)
        byte = emulate_hf_stop_string(backend, (), output, "ä", token_bytes=mapper)
        self.assertTrue(byte.available and byte.fired)
        self.assertEqual(byte.detail, "fired_at_step_2")
        self.assertEqual(_reference_first_step(tokenizer, ["ä"], (), output), 2)

    def test_byte_fallback_hex_tokens(self) -> None:
        import json

        tokens = ["<0x41>", "x"]
        full = _with_prefix_tokens(tokens)
        backend = _TextBackend(full)

        class _FallbackTokenizer(_FakeByteTokenizer):
            """ByteFallback decoder config via __getstate__ (not class name)."""

            @property
            def backend_tokenizer(self):  # type: ignore[no-untyped-def]
                decoder = SimpleNamespace()
                decoder.__getstate__ = lambda: json.dumps(
                    {"type": "Sequence", "decoders": [{"type": "ByteFallback"}]}
                )
                return SimpleNamespace(decoder=decoder)

        fb_tokenizer = _FallbackTokenizer(full)
        mapper = _hf_byte_token_mapper(fb_tokenizer)
        self.assertIsNotNone(mapper)
        assert mapper is not None
        self.assertEqual(mapper(full.index("<0x41>")), b"A")
        output = tuple(full.index(t) for t in tokens)
        outcome = emulate_hf_stop_string(backend, (), output, "A", token_bytes=mapper)
        self.assertTrue(outcome.available and outcome.fired)
        self.assertEqual(
            outcome.detail,
            f"fired_at_step_{_reference_first_step(fb_tokenizer, ['A'], (), output)}",
        )


@unittest.skipUnless(HAS_HF, "the 'hf' extra is not installed")
class HfDifferentialFuzzTests(unittest.TestCase):
    """Seeded differential fuzz vs the real HF criteria, both paths."""

    STRING_TOKENS = [
        "st",
        "op",
        "to",
        "s",
        "p",
        "x",
        "ab",
        "é",
        "😀",
        "caf",
        "stop",
        "las",
        "topper",
        "pped",
        "opera",
        "zz",
        "",
    ]
    STRING_STOPS = [
        "stop",
        "a",
        "ab",
        "abc",
        "é",
        "😀",
        "stopstop",
        "x",
        "abcdef",
        "zz",
        "p",
        "op",
    ]
    BYTE_TOKENS = [
        "b",
        "c",
        "f",
        " ",
        "!",
        "Ã",
        "©",
        "Ā",
        "cafÃ",
        "<0x41>",
        "x",
        "e",
        "é",
        "😀",
    ]
    BYTE_STOPS = ["café", "a", "ab", "😀", " ", "é", "caf", "AA", "A"]

    def test_string_path_fuzz(self) -> None:
        rng = random.Random(20261015)  # noqa: S311 -- deterministic test seed
        tokens = _with_prefix_tokens(self.STRING_TOKENS)
        tokenizer = _FakeStringTokenizer(tokens)
        backend = _TextBackend(tokens)
        trials = 0
        for stop in self.STRING_STOPS:
            for _ in range(25):
                prompt = tuple(rng.randrange(len(tokens)) for _ in range(rng.randint(0, 3)))
                output = tuple(rng.randrange(len(tokens)) for _ in range(rng.randint(0, 6)))
                expected = _reference_first_step(tokenizer, [stop], prompt, output)
                outcome = emulate_hf_stop_string(backend, prompt, output, stop)
                self.assertTrue(outcome.available)
                self.assertEqual(outcome.fired, expected is not None)
                self.assertIsNone(outcome.returned_text)
                if expected is None:
                    self.assertEqual(outcome.detail, "no_stop_within_cap")
                    self.assertEqual(outcome.truncated_ids, tuple(output))
                else:
                    self.assertEqual(outcome.detail, f"fired_at_step_{expected}")
                    self.assertEqual(outcome.truncated_ids, tuple(output[:expected]))
                trials += 1
        self.assertEqual(trials, len(self.STRING_STOPS) * 25)

    def test_byte_path_fuzz(self) -> None:
        rng = random.Random(20261016)  # noqa: S311 -- deterministic test seed
        tokens = _with_prefix_tokens(self.BYTE_TOKENS)
        tokenizer = _FakeByteTokenizer(tokens)
        backend = _TextBackend(tokens)
        mapper = _hf_byte_token_mapper(tokenizer)
        self.assertIsNotNone(mapper)
        trials = 0
        for stop in self.BYTE_STOPS:
            for _ in range(25):
                prompt = tuple(rng.randrange(len(tokens)) for _ in range(rng.randint(0, 3)))
                output = tuple(rng.randrange(len(tokens)) for _ in range(rng.randint(0, 6)))
                expected = _reference_first_step(tokenizer, [stop], prompt, output)
                outcome = emulate_hf_stop_string(backend, prompt, output, stop, token_bytes=mapper)
                self.assertTrue(outcome.available)
                self.assertEqual(outcome.fired, expected is not None)
                self.assertIsNone(outcome.returned_text)
                if expected is None:
                    self.assertEqual(outcome.detail, "no_stop_within_cap")
                    self.assertEqual(outcome.truncated_ids, tuple(output))
                else:
                    self.assertEqual(outcome.detail, f"fired_at_step_{expected}")
                    self.assertEqual(outcome.truncated_ids, tuple(output[:expected]))
                trials += 1
        self.assertEqual(trials, len(self.BYTE_STOPS) * 25)


if __name__ == "__main__":
    unittest.main()
