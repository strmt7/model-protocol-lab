"""Tests for Cycle 1 additions.

A. Byte-space HF stop-string matching (transformers issue #46519 / PR
   #46530): the emulation mirrors current transformers, which builds the
   StopStringCriteria match table in byte space for byte-level and
   byte-fallback decoders.
B. Token-budget robustness frontier (experiments.budget_frontier): the exact
   post-hoc pass/budget curve and the budgets where the selection decision
   would flip.
C. doctor next-command suggestion.
D. Frontier HTML report section.
"""

import importlib.util
import io
import json
import random
import shlex
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from model_protocol_lab.backends import (
    TruncatedPromptBackend,
    _bytes_to_unicode,
    _decoder_config_has_type,
    _hf_byte_token_mapper,
    _hf_overhang_match,
    emulate_hf_stop_string,
)
from model_protocol_lab.cli import _matrix_stage_rows, main
from model_protocol_lab.core import Fixture, Message, first_boundary
from model_protocol_lab.doctor import _next_command, doctor, examine_model
from model_protocol_lab.experiments import budget_frontier, stop_matrix
from model_protocol_lab.reporting import _SCRIPT, render_html
from tests.test_core import FakeBackend
from tests.test_doctor import write_model_dir

HAS_HF = all(
    importlib.util.find_spec(name) is not None for name in ("transformers", "tokenizers", "torch")
)

FIXTURES_JSON = Path(__file__).resolve().parent.parent / "examples" / "fixtures.json"


def _fixture(fid):
    """Fixture."""
    return Fixture(id=fid, messages=(Message("user", "q"),), expected="a", group="g")


# ---------------------------------------------------------------------------
# A. Byte-space matching


class OverhangBytesTests(unittest.TestCase):
    """Overhang bytes tests."""

    def test_bytes_agree_with_str_on_ascii(self):
        """Test bytes agree with str on ascii."""
        rng = random.Random(1234)  # noqa: S311 -- deterministic test seed
        alphabet = "abc "
        for _ in range(500):
            tokens = [
                "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 5)))
                for _ in range(rng.randint(1, 4))
            ]
            stop = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 4)))
            with self.subTest(tokens=tokens, stop=stop):
                self.assertEqual(
                    _hf_overhang_match(tokens, stop),
                    _hf_overhang_match([t.encode() for t in tokens], stop.encode()),
                )

    def test_bytes_mid_token_occurrence(self):
        """Test bytes mid token occurrence."""
        self.assertTrue(_hf_overhang_match([b" world"], b"or"))
        self.assertTrue(_hf_overhang_match([b"las", b"topper"], b"stop"))
        self.assertFalse(_hf_overhang_match([b" wold"], b"or"))
        self.assertFalse(_hf_overhang_match([b"hello ", b"xsto"], b"stop"))

    def test_mixed_types_rejected(self):
        """Test mixed types rejected."""
        self.assertFalse(_hf_overhang_match([b"ab"], "ab"))
        self.assertFalse(_hf_overhang_match(["ab"], b"ab"))


class _NamedDecoder:
    """Test double with a controllable class name for decoder detection."""

    def __init__(self, name, state):
        self._decoder_name = name
        self._state = state

    def __getstate__(self):
        return self._state


def _stub_tokenizer(decoder_name, decoder_state, vocab):
    """Minimal tokenizer duck-type for _hf_byte_token_mapper (no HF needed)."""
    ids = {token: index for index, token in enumerate(vocab)}

    class StubDecoder(_NamedDecoder):
        """Stub decoder."""

    StubDecoder.__name__ = decoder_name
    return SimpleNamespace(
        backend_tokenizer=SimpleNamespace(decoder=StubDecoder(decoder_name, decoder_state)),
        get_vocab=lambda: dict(zip(vocab, range(len(vocab)), strict=True)),
        convert_ids_to_tokens=lambda i: vocab[i],
        _ids=ids,
    )


class ByteMapperModeTests(unittest.TestCase):
    """Byte mapper mode tests."""

    def test_bytes_to_unicode_matches_spec(self):
        """Test bytes to unicode matches spec."""
        table = _bytes_to_unicode()
        self.assertEqual(len(table), 256)
        # Printable ASCII maps to itself; byte 0x00 is U+0100; 0xE4 is "ä".
        self.assertEqual(table[0x41], "A")
        self.assertEqual(table[0x00], "Ā")
        self.assertEqual(table[0xE4], "ä")
        self.assertEqual(sorted(table), list(range(256)))

    @unittest.skipUnless(
        importlib.util.find_spec("transformers") is not None, "transformers not installed"
    )
    def test_bytes_to_unicode_matches_transformers(self):
        """Test bytes to unicode matches transformers."""
        from transformers.convert_slow_tokenizer import bytes_to_unicode

        self.assertEqual(_bytes_to_unicode(), bytes_to_unicode())

    def test_no_decoder_returns_none(self):
        """Test no decoder returns none."""
        self.assertIsNone(_hf_byte_token_mapper(SimpleNamespace()))
        self.assertIsNone(_hf_byte_token_mapper(None))

    def test_non_byte_decoder_returns_none(self):
        """Test non byte decoder returns none."""
        tokenizer = _stub_tokenizer("WordLevel", "", ["a", "b"])
        self.assertIsNone(_hf_byte_token_mapper(tokenizer))

    def test_unparseable_decoder_state_returns_none(self):
        """Test unparseable decoder state returns none."""
        tokenizer = _stub_tokenizer("Sequence", "not-json{", ["a"])
        self.assertIsNone(_hf_byte_token_mapper(tokenizer))

    def test_byte_level_maps_raw_bytes(self):
        """Test byte level maps raw bytes."""
        # "€" (U+20AC) is outside the byte-level alphabet; "Ā" (U+0100) is
        # byte 0x00 in it, so it must not be used as the unmappable case.
        tokenizer = _stub_tokenizer("ByteLevel", "", ["Ã", "¤", "ab", "€"])
        mapper = _hf_byte_token_mapper(tokenizer)
        self.assertIsNotNone(mapper)
        assert mapper is not None
        self.assertEqual(mapper(0), b"\xc3")
        self.assertEqual(mapper(1), b"\xa4")
        self.assertEqual(mapper(2), b"ab")
        self.assertIsNone(mapper(3))

    def test_byte_fallback_maps_hex_tokens_only(self):
        """Test byte fallback maps hex tokens only."""
        state = json.dumps({"type": "Sequence", "decoders": [{"type": "ByteFallback"}]})
        tokenizer = _stub_tokenizer("Sequence", state, ["<0x41>", "<0xZZ>", "hello"])
        mapper = _hf_byte_token_mapper(tokenizer)
        self.assertIsNotNone(mapper)
        assert mapper is not None
        self.assertEqual(mapper(0), b"A")
        self.assertIsNone(mapper(1))
        self.assertIsNone(mapper(2))

    def test_byte_level_preferred_detector_falls_back_to_config(self):
        """Test byte level preferred detector falls back to config."""
        # A decoder that is not literally named ByteLevel but whose config
        # carries the marker still selects the byte path.
        state = json.dumps({"type": "ByteLevel"})
        tokenizer = _stub_tokenizer("Custom", state, ["a"])
        mapper = _hf_byte_token_mapper(tokenizer)
        self.assertIsNotNone(mapper)

    def test_str_decoder_state_is_parsed(self):
        """Test str decoder state is parsed."""
        # __getstate__ may return str instead of bytes; both are accepted.
        state = json.dumps({"type": "Sequence", "decoders": [{"type": "ByteFallback"}]})
        tokenizer = _stub_tokenizer("Sequence", state, ["<0x41>"])
        mapper = _hf_byte_token_mapper(tokenizer)
        self.assertIsNotNone(mapper)
        assert mapper is not None
        self.assertEqual(mapper(0), b"A")

    def test_bytes_decoder_state_is_parsed(self):
        """Test bytes decoder state is parsed."""
        state = json.dumps({"type": "ByteLevel"}).encode()
        tokenizer = _stub_tokenizer("Custom", state, ["a"])
        mapper = _hf_byte_token_mapper(tokenizer)
        self.assertIsNotNone(mapper)

    def test_mapper_construction_failure_degrades_to_none(self):
        """Test mapper construction failure degrades to none."""

        class BoomTokenizer(SimpleNamespace):
            """Boom tokenizer."""

            @property
            def backend_tokenizer(self):
                raise RuntimeError("synthetic")

            @staticmethod
            def get_vocab():
                """Get vocab."""
                # pragma: no cover -- never reached
                return {}

        self.assertIsNone(_hf_byte_token_mapper(BoomTokenizer()))

    def test_empty_vocab_degrades_to_none(self):
        """Test empty vocab degrades to none."""
        tokenizer = _stub_tokenizer("ByteLevel", "", [])
        self.assertIsNone(_hf_byte_token_mapper(tokenizer))

    def test_non_string_token_degrades_to_none(self):
        """Test non string token degrades to none."""
        tokenizer = _stub_tokenizer("ByteLevel", "", ["a"])
        tokenizer.convert_ids_to_tokens = lambda i: None
        mapper = _hf_byte_token_mapper(tokenizer)
        assert mapper is not None
        self.assertIsNone(mapper(0))
        tokenizer = _stub_tokenizer("ByteLevel", "", ["a"])
        mapper = _hf_byte_token_mapper(tokenizer)
        assert mapper is not None
        for bad in ("0", 1.5, -1, None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                mapper(bad)  # type: ignore[arg-type]
        # Out-of-vocabulary ids mirror HF's clamp-to-dummy: empty bytes that
        # can never match.
        self.assertEqual(mapper(999), b"")

    def test_decoder_config_has_type(self):
        """Test decoder config has type."""
        self.assertTrue(_decoder_config_has_type({"type": "ByteFallback"}, "ByteFallback"))
        self.assertTrue(
            _decoder_config_has_type(
                {"type": "Sequence", "decoders": [{"type": "ByteLevel"}]}, "ByteLevel"
            )
        )
        self.assertTrue(_decoder_config_has_type([{"type": "ByteFallback"}], "ByteFallback"))
        self.assertFalse(_decoder_config_has_type({"type": "WordLevel"}, "ByteFallback"))
        self.assertFalse(_decoder_config_has_type(None, "ByteFallback"))
        self.assertFalse(_decoder_config_has_type("ByteLevel", "ByteLevel"))


class EmulationBytePathTests(unittest.TestCase):
    """Emulation byte path tests."""

    def test_byte_path_fires_on_fragment_bytes(self):
        """Test byte path fires on fragment bytes."""
        # "ä" as raw bytes b"\xc3\xa4" split across two tokens: the decode
        # strings are useless here, the bytes match.
        backend = FakeBackend(decode_script=["�", "�"])
        mapper = {10: b"\xc3", 11: b"\xa4"}
        outcome = emulate_hf_stop_string(backend, (), (10, 11), "ä", token_bytes=mapper.get)
        self.assertTrue(outcome.available)
        self.assertTrue(outcome.fired)
        self.assertEqual(outcome.truncated_ids, (10, 11))
        self.assertEqual(outcome.detail, "fired_at_step_2")

    def test_byte_path_falls_back_to_decode_for_unmapped_tokens(self):
        """Test byte path falls back to decode for unmapped tokens."""
        backend = FakeBackend(decode_script=["a", "b"])
        outcome = emulate_hf_stop_string(backend, (), (1, 2), "ab", token_bytes=lambda i: None)
        self.assertTrue(outcome.available)
        self.assertTrue(outcome.fired)

    def test_byte_path_rejects_non_callable(self):
        """Test byte path rejects non callable."""
        outcome = emulate_hf_stop_string(FakeBackend(), (1,), (2,), "x", token_bytes="nope")
        self.assertFalse(outcome.available)
        self.assertEqual(outcome.detail, "invalid_emulation_input")

    def test_byte_path_hook_failure_degrades(self):
        """Test byte path hook failure degrades."""

        def bad(token_id):
            """Bad."""
            raise RuntimeError("synthetic")

        outcome = emulate_hf_stop_string(FakeBackend(), (), (1,), "x", token_bytes=bad)
        self.assertFalse(outcome.available)
        self.assertEqual(outcome.detail, "token_text_unavailable")

    def test_byte_path_hook_returning_non_bytes_degrades(self):
        """Test byte path hook returning non bytes degrades."""
        outcome = emulate_hf_stop_string(FakeBackend(), (), (1,), "x", token_bytes=lambda i: "s")
        self.assertFalse(outcome.available)
        self.assertEqual(outcome.detail, "token_text_unavailable")

    def test_byte_path_decode_non_string_degrades(self):
        """Test byte path decode non string degrades."""

        class BadDecode:
            """Bad decode."""

            fingerprint = "bad"

            @staticmethod
            def decode(token_ids):
                """Decode."""
                return 42

        outcome = emulate_hf_stop_string(BadDecode(), (), (1,), "x", token_bytes=lambda i: None)
        self.assertFalse(outcome.available)
        self.assertEqual(outcome.detail, "token_text_unavailable")

    def test_truncated_backend_forwards_byte_mapper(self):
        """Test truncated backend forwards byte mapper."""
        inner = FakeBackend()
        inner.hf_stop_token_bytes = lambda i: b"q"  # type: ignore[attr-defined]
        wrapped = TruncatedPromptBackend(inner, 1)
        self.assertIs(wrapped.hf_stop_token_bytes, inner.hf_stop_token_bytes)
        plain = TruncatedPromptBackend(FakeBackend(), 1)
        self.assertIsNone(plain.hf_stop_token_bytes)

    def test_stop_matrix_rejects_non_callable_token_bytes(self):
        """Test stop matrix rejects non callable token bytes."""
        obs = {
            "fixture": "f1",
            "status": "completed",
            "exact_match": True,
            "boundary_found": True,
            "input_ids": [1],
            "output_ids": [5],
        }
        with self.assertRaises(ValueError):
            stop_matrix(
                FakeBackend(), "A", (_fixture("f1"),), (obs,), boundary=(5,), token_bytes=42
            )

    def test_matrix_stage_rows_threads_token_bytes(self):
        """Test matrix stage rows threads token bytes."""
        obs = {
            "fixture": "f1",
            "status": "completed",
            "exact_match": True,
            "boundary_found": True,
            "input_ids": [1],
            "output_ids": [5],
        }
        result = {"search": [{"format": {"name": "A"}, "observations": [obs]}]}
        # Decode order: boundary text, prompt token, output token. The byte
        # path maps every token to None, so each falls back to its decode
        # encoded as UTF-8; the output byte then equals the stop bytes.
        backend = FakeBackend(decode_script=["S", "P", "S"])
        rows = _matrix_stage_rows(
            result,
            "search",
            "A",
            (_fixture("f1"),),
            backend,
            (5,),
            None,
            token_bytes=lambda i: None,
        )
        assert rows is not None
        cell = rows[0]["cells"]["hf_stop_string"]
        self.assertTrue(cell["available"])
        self.assertTrue(cell["fired"])


def _fragment_tokenizer_json(path):
    """Byte-level BPE where U+00E4 ('ä') fragments into two byte tokens.

    Mirrors transformers issue #46519: the character's UTF-8 bytes are kept
    as separate tokens, so per-token isolated decodes yield replacement
    characters while the byte stream matches the stop string.
    """
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel

    vocab = {"Ã": 0, "¤": 1, "a": 2, "b": 3, "<unk>": 4}
    tokenizer = Tokenizer(BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.save(str(path))
    return vocab


@unittest.skipUnless(HAS_HF, "the 'hf' extra is not installed")
class ByteSpaceOracleTests(unittest.TestCase):
    """The byte-space port agrees with the real StopStringCriteria."""

    def test_fragment_stop_matches_oracle(self):
        """Test fragment stop matches oracle."""
        import torch
        from transformers import PreTrainedTokenizerFast
        from transformers.generation.stopping_criteria import StopStringCriteria

        with TemporaryDirectory() as tmp:
            vocab = _fragment_tokenizer_json(Path(tmp) / "tokenizer.json")
            hf_tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=str(Path(tmp) / "tokenizer.json"),
                clean_up_tokenization_spaces=False,
            )
            ids = [vocab["Ã"], vocab["¤"]]
            oracle = bool(StopStringCriteria(hf_tokenizer, "ä")(torch.tensor([ids]), None).item())
            self.assertTrue(oracle)

            mapper = _hf_byte_token_mapper(hf_tokenizer)
            self.assertIsNotNone(mapper)
            assert mapper is not None
            self.assertEqual(mapper(vocab["Ã"]), "ä".encode()[:1])
            self.assertEqual(mapper(vocab["¤"]), "ä".encode()[1:])

            class DecodeBackend:
                """Decode backend."""

                fingerprint = "fragment-oracle-not-a-model"

                @staticmethod
                def decode(token_ids):
                    """Decode."""
                    return hf_tokenizer.decode(list(token_ids))

            backend = DecodeBackend()
            old = emulate_hf_stop_string(backend, (), tuple(ids), "ä")
            new = emulate_hf_stop_string(backend, (), tuple(ids), "ä", token_bytes=mapper)
            # The classic decode-string path stays silent on fragments (the
            # previously documented residual); the byte-space path fires with
            # the oracle.
            self.assertTrue(old.available)
            self.assertFalse(old.fired)
            self.assertTrue(new.available)
            self.assertTrue(new.fired)
            self.assertEqual(new.truncated_ids, tuple(ids))

    def test_byte_path_agrees_with_oracle_on_ascii(self):
        """Test byte path agrees with oracle on ascii."""
        import torch
        from transformers import PreTrainedTokenizerFast
        from transformers.generation.stopping_criteria import StopStringCriteria

        from tests.test_round3 import _oracle_bpe_tokenizer

        rng = random.Random(20261013)  # noqa: S311 -- deterministic test seed
        with TemporaryDirectory() as tmp:
            vocab = _oracle_bpe_tokenizer(Path(tmp) / "tokenizer.json")
            hf_tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=str(Path(tmp) / "tokenizer.json"),
                clean_up_tokenization_spaces=False,
            )
            tokens = sorted(vocab)
            stops = ["stop", "to", "as", "op"]
            mapper = _hf_byte_token_mapper(hf_tokenizer)
            self.assertIsNotNone(mapper)

            class DecodeBackend:
                """Decode backend."""

                fingerprint = "ascii-oracle-not-a-model"

                @staticmethod
                def decode(token_ids):
                    """Decode."""
                    return hf_tokenizer.decode(list(token_ids))

            backend = DecodeBackend()
            for trial in range(200):
                seq = [rng.choice(tokens) for _ in range(rng.randint(1, 4))]
                stop = rng.choice(stops)
                ids = tuple(vocab[token] for token in seq)
                criterion = StopStringCriteria(hf_tokenizer, stop)
                # generate() consults the criterion after every new token, so
                # the oracle fires when any prefix completes the stop.
                oracle = any(
                    bool(criterion(torch.tensor([list(ids[:k])]), None).item())
                    for k in range(1, len(ids) + 1)
                )
                with self.subTest(trial=trial, seq=seq, stop=stop):
                    classic = emulate_hf_stop_string(backend, (), ids, stop)
                    byte = emulate_hf_stop_string(backend, (), ids, stop, token_bytes=mapper)
                    self.assertTrue(classic.available)
                    self.assertTrue(byte.available)
                    self.assertEqual(classic.fired, oracle)
                    self.assertEqual(byte.fired, oracle)


# ---------------------------------------------------------------------------
# B. Token-budget frontier


def _search_entry(name, specs):
    """specs: list of (output_ids, passed) per fixture."""
    observations = []
    for index, (output_ids, passed) in enumerate(specs):
        observations.append(
            {
                "fixture": f"f{index}",
                "status": "completed",
                "exact_match": passed,
                "boundary_found": passed,
                "output_ids": list(output_ids),
            }
        )
    return {"format": {"name": name}, "observations": observations}


class BudgetFrontierTests(unittest.TestCase):
    """Budget frontier tests."""

    def test_min_budgets_curve_and_flips(self):
        """Test min budgets curve and flips."""
        boundary = (9,)
        search = [
            _search_entry("A", [([1, 2, 9], True), ([9], True)]),
            _search_entry("B", [([1, 2, 3, 9], True), ([1, 2, 3], False)]),
        ]
        frontier = budget_frontier(
            search,
            boundary=boundary,
            max_new_tokens=16,
            native_decision="compatible_on_fixture_set",
            native_winner="A",
        )
        candidate_a = frontier["candidates"]["A"]
        self.assertEqual(candidate_a["min_budget_per_fixture"], [3, 1])
        self.assertEqual(candidate_a["pass_steps"], [[1, 1], [3, 2]])
        self.assertEqual(candidate_a["full_pass_budget"], 3)
        candidate_b = frontier["candidates"]["B"]
        self.assertEqual(candidate_b["min_budget_per_fixture"], [4, None])
        self.assertEqual(candidate_b["pass_steps"], [[1, 0], [4, 1]])
        self.assertIsNone(candidate_b["full_pass_budget"])
        self.assertEqual(
            frontier["decision_steps"], [[1, "no_candidate", None], [3, "selected", "A"]]
        )
        self.assertEqual(frontier["flip_ranges"], [[1, 2]])
        self.assertFalse(frontier["budget_robust"])
        self.assertEqual(frontier["native_winner"], "A")

    def test_ambiguous_native_can_flip_to_unique_winner(self):
        """Test ambiguous native can flip to unique winner."""
        boundary = (9,)
        search = [
            _search_entry("A", [([1, 2, 9], True), ([4, 5, 9], True)]),
            _search_entry("B", [([1, 2, 3, 4, 9], True), ([6, 7, 8, 9, 9], True)]),
        ]
        frontier = budget_frontier(
            search,
            boundary=boundary,
            max_new_tokens=16,
            native_decision="ambiguous",
            native_winner=None,
        )
        self.assertEqual(
            frontier["decision_steps"],
            [
                [1, "no_candidate", None],
                [3, "selected", "A"],
                [5, "ambiguous", None],
            ],
        )
        self.assertEqual(frontier["flip_ranges"], [[1, 4]])
        self.assertFalse(frontier["budget_robust"])

    def test_stable_frontier(self):
        """Test stable frontier."""
        boundary = (9,)
        search = [
            _search_entry("A", [([9], True), ([9], True)]),
            _search_entry("B", [([1, 2], False), ([3], False)]),
        ]
        frontier = budget_frontier(
            search,
            boundary=boundary,
            max_new_tokens=16,
            native_decision="compatible_on_fixture_set",
            native_winner="A",
        )
        self.assertEqual(frontier["flip_ranges"], [])
        self.assertTrue(frontier["budget_robust"])
        self.assertEqual(frontier["decision_steps"], [[1, "selected", "A"]])

    def test_no_candidate_native(self):
        """Test no candidate native."""
        boundary = (9,)
        search = [
            _search_entry("A", [([1, 2, 9], True), ([1], False)]),
            _search_entry("B", [([2], False), ([3, 4, 9], True)]),
        ]
        frontier = budget_frontier(
            search,
            boundary=boundary,
            max_new_tokens=16,
            native_decision="no_candidate",
            native_winner=None,
        )
        self.assertTrue(frontier["budget_robust"])

    def test_truncation_lemma_bruteforce(self):
        """first_boundary on truncated outputs obeys the min-budget rule.

        This is the combinatorial claim the frontier's exactness rests on:
        for a native first occurrence at `cut`, a truncated prefix contains
        a complete boundary iff b >= cut + len(boundary).
        """
        rng = random.Random(77)  # noqa: S311 -- deterministic test seed
        boundary = (9, 9)
        for _ in range(300):
            output = tuple(rng.choice((1, 2, 9)) for _ in range(rng.randint(1, 10)))
            cut = first_boundary(output, boundary)
            for budget in range(1, len(output) + 1):
                found = first_boundary(output[:budget], boundary) is not None
                expected = cut is not None and budget >= cut + len(boundary)
                with self.subTest(output=output, budget=budget):
                    self.assertEqual(found, expected)

    def test_incomplete_observations_never_pass(self):
        """Test incomplete observations never pass."""
        boundary = (9,)
        search = [
            {
                "format": {"name": "A"},
                "observations": [
                    {
                        "fixture": "f0",
                        "status": "failed",
                        "exact_match": False,
                        "boundary_found": False,
                        "output_ids": [1, 2, 9],
                    }
                ],
            }
        ]
        frontier = budget_frontier(
            search,
            boundary=boundary,
            max_new_tokens=8,
            native_decision="no_candidate",
            native_winner=None,
        )
        self.assertEqual(frontier["candidates"]["A"]["min_budget_per_fixture"], [None])
        self.assertEqual(frontier["candidates"]["A"]["pass_steps"], [[1, 0]])

    def test_rejects_bad_input(self):
        """Test rejects bad input."""
        good = [_search_entry("A", [([9], True)])]
        kwargs = {
            "boundary": (9,),
            "max_new_tokens": 16,
            "native_decision": "compatible_on_fixture_set",
            "native_winner": "A",
        }
        with self.assertRaises(ValueError):
            budget_frontier([], **kwargs)
        with self.assertRaises(ValueError):
            budget_frontier("nope", **kwargs)
        with self.assertRaises(ValueError):
            budget_frontier(
                [_search_entry("A", [([9], True)]), _search_entry("A", [([9], True)])], **kwargs
            )
        with self.assertRaises(ValueError):
            budget_frontier(
                [
                    _search_entry("A", [([9], True)]),
                    {"format": {"name": "B"}, "observations": []},
                ],
                **kwargs,
            )
        with self.assertRaises(ValueError):
            budget_frontier(
                good,
                boundary=(),
                max_new_tokens=16,
                native_decision="compatible_on_fixture_set",
                native_winner="A",
            )
        with self.assertRaises(ValueError):
            budget_frontier(
                good,
                boundary=(9,),
                max_new_tokens=0,
                native_decision="compatible_on_fixture_set",
                native_winner="A",
            )
        with self.assertRaises(ValueError):
            budget_frontier(
                good, boundary=(9,), max_new_tokens=16, native_decision="failed", native_winner=None
            )
        with self.assertRaises(ValueError):
            budget_frontier(
                good,
                boundary=(9,),
                max_new_tokens=16,
                native_decision="compatible_on_fixture_set",
                native_winner="Z",
            )
        with self.assertRaises(ValueError):
            budget_frontier(
                good,
                boundary=(9,),
                max_new_tokens=16,
                native_decision="ambiguous",
                native_winner="A",
            )
        # Entries must cover the same fixtures in the same order.
        other = {
            "format": {"name": "B"},
            "observations": [
                {
                    "fixture": "g0",
                    "status": "completed",
                    "exact_match": True,
                    "boundary_found": True,
                    "output_ids": [9],
                }
            ],
        }
        with self.assertRaises(ValueError):
            budget_frontier([good[0], other], **kwargs)
        # Malformed entries and observations are rejected, not misread.
        with self.assertRaises(ValueError):
            budget_frontier([42], **kwargs)
        with self.assertRaises(ValueError):
            budget_frontier([{"format": {}}], **kwargs)
        with self.assertRaises(ValueError):
            budget_frontier([{"format": {"name": "A"}, "observations": [42]}], **kwargs)


# ---------------------------------------------------------------------------
# C. doctor next-command suggestion


def _write_model(tmp, **config_extra):
    """Write model."""
    root = Path(tmp) / "model"
    root.mkdir()
    config = {"model_type": "gpt2", "n_positions": 1024}
    config.update(config_extra)
    return write_model_dir(root, extra_config=config)


class DoctorSuggestionTests(unittest.TestCase):
    """Doctor suggestion tests."""

    def test_suggestion_on_clean_doctor(self):
        """Test suggestion on clean doctor."""
        with TemporaryDirectory() as tmp:
            model = _write_model(tmp)
            report = doctor(model, FIXTURES_JSON, boundary_ids=(1, 2))
            self.assertEqual(report["verdict"], "ok")
            suggestion = report["suggestion"]
            self.assertIsNotNone(suggestion)
            assert suggestion is not None
            command = suggestion["next_command"]
            self.assertIn("--max-context 1024", command)
            self.assertIn("--boundary-ids 1,2", command)
            self.assertIn("--family FAMILY", command)
            self.assertIn(shlex.quote(str(model)), command)
            self.assertIn(shlex.quote(str(FIXTURES_JSON)), command)
            self.assertIn("--output NEW_REPORT_DIRECTORY", command)
            for family in ("chatml", "llama3", "gemma"):
                self.assertIn(family, suggestion["note"])

    def test_no_suggestion_without_dataset(self):
        """Test no suggestion without dataset."""
        with TemporaryDirectory() as tmp:
            report = doctor(_write_model(tmp), boundary_ids=(1, 2))
            self.assertIsNone(report["suggestion"])

    def test_no_suggestion_without_boundary_ids(self):
        """Test no suggestion without boundary ids."""
        with TemporaryDirectory() as tmp:
            report = doctor(_write_model(tmp), FIXTURES_JSON)
            self.assertIsNone(report["suggestion"])

    def test_no_suggestion_on_failed_verdict(self):
        """Test no suggestion on failed verdict."""
        with TemporaryDirectory() as tmp:
            report = doctor(Path(tmp) / "missing", FIXTURES_JSON, boundary_ids=(1, 2))
            self.assertEqual(report["verdict"], "failed")
            self.assertIsNone(report["suggestion"])

    def test_no_suggestion_without_declared_window(self):
        """Test no suggestion without declared window."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "model"
            root.mkdir()
            model = write_model_dir(root, extra_config={"model_type": "gpt2"})
            report = doctor(model, FIXTURES_JSON, boundary_ids=(1, 2))
            self.assertEqual(report["verdict"], "ok")
            self.assertIsNone(report["suggestion"])

    def test_suggestion_quotes_paths_with_spaces(self):
        """Test suggestion quotes paths with spaces."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "my model"
            root.mkdir()
            model = write_model_dir(root, extra_config={"model_type": "gpt2", "n_positions": 512})
            report = doctor(model, FIXTURES_JSON, boundary_ids=(7,))
            suggestion = report["suggestion"]
            assert suggestion is not None
            self.assertIn(shlex.quote(str(model)), suggestion["next_command"])
            self.assertNotIn(" --model my model ", suggestion["next_command"])

    def test_examine_model_reports_context_window(self):
        """Test examine model reports context window."""
        with TemporaryDirectory() as tmp:
            gpt2 = _write_model(tmp)
            self.assertEqual(examine_model(gpt2)["model"]["context_window"], 1024)
            root = Path(tmp) / "llama"
            root.mkdir()
            llama = write_model_dir(
                root,
                extra_config={"model_type": "llama", "max_position_embeddings": 2048},
            )
            self.assertEqual(examine_model(llama)["model"]["context_window"], 2048)
            root = Path(tmp) / "nowindow"
            root.mkdir()
            plain = write_model_dir(root, extra_config={"model_type": "gpt2"})
            self.assertIsNone(examine_model(plain)["model"]["context_window"])

    def test_next_command_unit_validation(self):
        """Test next command unit validation."""
        base = {"verdict": "ok", "model": {"context_window": 8}}
        self.assertIsNone(_next_command("m", "d", None, base))
        self.assertIsNone(_next_command("m", "d", (), base))
        self.assertIsNone(_next_command("m", "d", (1, -2), base))
        self.assertIsNone(_next_command("m", "d", (1, "x"), base))
        self.assertIsNone(_next_command("m", "d", 5, base))  # not iterable
        self.assertIsNone(_next_command("m", None, (1,), base))
        self.assertIsNone(
            _next_command("m", "d", (1,), {"verdict": "failed", "model": {"context_window": 8}})
        )
        self.assertIsNone(_next_command("m", "d", (1,), {"verdict": "ok", "model": {}}))
        good = _next_command("m", "d", (1, 2), base)
        assert good is not None
        self.assertIn("--max-context 8", good["next_command"])
        self.assertIn("--boundary-ids 1,2", good["next_command"])

    def test_doctor_cli_includes_suggestion(self):
        """Test doctor cli includes suggestion."""
        with TemporaryDirectory() as tmp:
            model = _write_model(tmp)
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = main(
                    [
                        "doctor",
                        "--model",
                        str(model),
                        "--dataset",
                        str(FIXTURES_JSON),
                        "--boundary-ids",
                        "1,2",
                    ]
                )
            self.assertEqual(code, 0)
            payload = json.loads(buffer.getvalue())
            self.assertIn("--max-context 1024", payload["suggestion"]["next_command"])


# ---------------------------------------------------------------------------
# D. Frontier report section


def _frontier_result(**overrides):
    """Frontier result."""
    boundary = (9,)
    search = [
        _search_entry("A", [([1, 2, 9], True), ([9], True)]),
        _search_entry("B", [([1, 2, 3, 9], True), ([1, 2, 3], False)]),
    ]
    frontier = budget_frontier(
        search,
        boundary=boundary,
        max_new_tokens=16,
        native_decision="compatible_on_fixture_set",
        native_winner="A",
    )
    result = {
        "decision": "compatible_on_fixture_set",
        "selected_candidate": "A",
        "budget_frontier": frontier,
        "search": [],
    }
    result.update(overrides)
    return result


class FrontierReportTests(unittest.TestCase):
    """Frontier report tests."""

    def test_stable_section(self):
        """Test stable section."""
        boundary = (9,)
        search = [
            _search_entry("A", [([9], True), ([9], True)]),
            _search_entry("B", [([1], False), ([2], False)]),
        ]
        frontier = budget_frontier(
            search,
            boundary=boundary,
            max_new_tokens=16,
            native_decision="compatible_on_fixture_set",
            native_winner="A",
        )
        html = render_html(
            {
                "decision": "compatible_on_fixture_set",
                "selected_candidate": "A",
                "budget_frontier": frontier,
                "search": [],
            }
        )
        self.assertIn("Token-budget frontier", html)
        self.assertIn("stable", html)
        self.assertIn("Winner pass curve: A", html)

    def test_flips_section(self):
        """Test flips section."""
        html = render_html(_frontier_result())
        self.assertIn("Token-budget frontier", html)
        self.assertIn("budget-sensitive", html)
        # Flip range 1..2 renders with an HTML en-dash.
        self.assertIn("budgets 1&ndash;2", html)
        self.assertIn("no_candidate", html)

    def test_section_tolerates_malformed_points(self):
        """Test section tolerates malformed points."""
        boundary = (9,)
        search = [
            _search_entry("A", [([9], True)]),
            _search_entry("B", [([1], False)]),
        ]
        frontier = budget_frontier(
            search,
            boundary=boundary,
            max_new_tokens=16,
            native_decision="compatible_on_fixture_set",
            native_winner="A",
        )
        # Malformed step data degrades to skipped rows, never a crash.
        frontier["budget_robust"] = False
        frontier["flip_ranges"] = ["oops", [1, 2], None]
        frontier["decision_steps"] = [["bad"], [1, "no_candidate", None]]
        frontier["candidates"]["A"]["pass_steps"] = ["bad", [1, 1], None]
        html = render_html(
            {
                "decision": "compatible_on_fixture_set",
                "selected_candidate": "A",
                "budget_frontier": frontier,
                "search": [],
            }
        )
        self.assertIn("Token-budget frontier", html)

    def test_flip_label_names_winner(self):
        """Test flip label names winner."""
        # Ambiguous native where A alone passes early: the flip range names A.
        boundary = (9,)
        search = [
            _search_entry("A", [([9], True), ([9], True)]),
            _search_entry("B", [([1, 2, 3, 4, 9], True), ([5, 6, 7, 8, 9], True)]),
        ]
        frontier = budget_frontier(
            search,
            boundary=boundary,
            max_new_tokens=16,
            native_decision="ambiguous",
            native_winner=None,
        )
        self.assertEqual(frontier["flip_ranges"], [[1, 4]])
        html = render_html(
            {
                "decision": "ambiguous",
                "selected_candidate": None,
                "budget_frontier": frontier,
                "search": [],
            }
        )
        self.assertIn("winner: A", html)
        boundary = (9,)
        evil = "<script>alert(1)</script>"
        search = [
            _search_entry(evil, [([9], True)]),
            _search_entry("B", [([1], False)]),
        ]
        frontier = budget_frontier(
            search,
            boundary=boundary,
            max_new_tokens=16,
            native_decision="compatible_on_fixture_set",
            native_winner=evil,
        )
        html = render_html(
            {
                "decision": "compatible_on_fixture_set",
                "selected_candidate": evil,
                "budget_frontier": frontier,
                "search": [],
            }
        )
        self.assertIn("&lt;script&gt;", html)
        # The hostile name must not introduce any script element beyond the
        # single allowlisted inline table sorter.
        scrubbed = html.replace("<script>" + _SCRIPT + "</script>", "")
        self.assertNotIn("<script>", scrubbed)

    def test_section_absent_when_unavailable(self):
        """Test section absent when unavailable."""
        html = render_html({"decision": "failed", "search": []})
        self.assertNotIn("Token-budget frontier", html)
        html = render_html(
            {"decision": "failed", "search": [], "budget_frontier": {"available": False}}
        )
        self.assertNotIn("Token-budget frontier", html)


if __name__ == "__main__":
    unittest.main()
