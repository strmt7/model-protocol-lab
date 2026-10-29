"""Cycle 35: behavioral verification of ``model_protocol_lab/backends.py``.

Target: the backend layer -- the ``Backend`` protocol contract, the
asset-loading/validation helpers (``_directory``, ``_member``,
``_read_json``, ``_model_assets``, ``_manifest``, ``_ids_valid``), the
``LocalHFBackend`` adapter's fail-closed initialization and its
encode/decode/generate/forward_logits validation, the
``TruncatedPromptBackend`` wrapper, the HF/vLLM/SGLang/EOS stop-semantic
ports (``emulate_*``), the byte-space stop machinery
(``_bytes_to_unicode``, ``_decoder_config_has_type``, ``_OOVTokenBytes``,
``_hf_byte_token_mapper``), and the ``StopOutcome`` record.

Method: FAKE backends only (no model, no network). torch/transformers are
installed in the test venv, so ``LocalHFBackend`` initialization is
exercised through asset validation, the declared-context-window check,
and tokenizer parsing -- all of which must fail closed with the fixed
``RuntimeError`` vocabulary; the actual weight-loading path needs a real
checkpoint and is not reachable here. An INDEPENDENT oracle reimplements
each stop port from its docstring with a
different structure (suffix-of-concatenation view for the HF overhang
match, linear scan instead of ``bisect_left`` for the vLLM truncation,
direct ``str.find`` semantics for the gate window) and is fuzz-compared
against the implementation on seeded-random inputs. ``LocalHFBackend``
instances for the white-box tests are built with ``object.__new__`` so
the validation layers can be exercised without a checkpoint.

DOCSTRING-VS-CODE DIVERGENCES (code behavior pinned, source untouched):

D1. ``_directory`` documents rejection of empty local directories via
    ``not str(value)``, but ``str(Path(""))`` is ``"."`` (truthy), so
    ``_directory(Path(""))`` resolves to the *current working directory*
    instead of raising ``invalid_local_directory``. Pinned by
    ``TestRedTeam.test_directory_empty_path_object_resolves_cwd``.
D2. All four ``emulate_*`` functions funnel invalid inputs to a
    ``StopOutcome(..., "invalid_emulation_input")`` -- except the funnel
    itself calls ``tuple(output_ids)`` first, so a non-iterable
    ``output_ids`` (``int``, ``None``) raises ``TypeError`` instead of
    returning the fixed-vocabulary outcome. Pinned by
    ``TestRedTeam.test_emulations_noniterable_ids_raise_typeerror``.
D3. ``LocalHFBackend.encode`` raises its own ``ValueError`` with code
    ``invalid_encoded_ids`` *inside* the ``try`` whose ``except``
    converts everything to ``RuntimeError("local_encode_failed")`` -- the
    inner code is unobservable from outside. Pinned white-box by
    ``TestLocalHFBackendValidation.test_encode_invalid_ids_swallowed``.
D4. ``LocalHFBackend.decode`` is annotated ``-> str`` but never checks
    what the tokenizer returned; a non-str tokenizer result propagates to
    the caller. (The ``emulate_*`` functions defensively degrade on
    non-str decodes, consistent with the adapter offering no guarantee.)
    Pinned white-box by
    ``TestLocalHFBackendValidation.test_decode_does_not_enforce_str``.
D5. ``LocalHFBackend.__init__`` maps *every* inner ``ValueError`` code
    (``invalid_local_directory``, ``adapters_not_supported``,
    ``unsupported_model_type``, ...) to one of two ``RuntimeError``
    codes; the fine-grained vocabulary in the comments is unobservable
    from outside by design ("init must fail closed"). Pinned by
    ``TestLocalHFBackendInit``.
D6. ``TruncatedPromptBackend.encode`` documents the
    ``prompt_truncation_splits_codepoint`` error for mid-codepoint cuts,
    but a ``drop_bytes`` at/above the text's byte length raises
    ``ValueError("drop_bytes must lie within the text's byte length")``
    straight from ``truncate_text_tail``, unwrapped. Pinned by
    ``TestTruncatedPromptBackend.test_encode_drop_beyond_length_unwrapped``.
D7. The ``_hf_overhang_match`` docstring notes the walk's
    ``max(0, len - target)`` lower bound "is a no-op for non-empty
    representations" -- verified: each consumed representation adds >= 1
    to ``matched``, so the walk can never reach below the bound. This
    makes a window-bound mutant an *equivalent* mutant (see sensitivity
    notes); it is not counted as a miss.

Sensitivity: the ``/tmp/run_mutants35.py`` runner (kept out of the repo)
applies 20 source mutants to a throwaway copy of the package and re-runs
this file; every mutant must turn the suite red.
"""

from __future__ import annotations

import json
import random
import shutil
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from threading import Event, RLock
from types import SimpleNamespace

from model_protocol_lab.backends import (
    LocalHFBackend,
    StopOutcome,
    TruncatedPromptBackend,
    _bytes_to_unicode,
    _decoder_config_has_type,
    _directory,
    _first_firing_step,
    _hf_byte_token_mapper,
    _hf_overhang_match,
    _ids_valid,
    _manifest,
    _member,
    _model_assets,
    _OOVTokenBytes,
    _read_json,
    emulate_eos_stop,
    emulate_hf_stop_string,
    emulate_sglang_stop,
    emulate_vllm_holdback,
)
from model_protocol_lab.core import Generation

# ---------------------------------------------------------------------------
# Fake backends (no model, no network)
# ---------------------------------------------------------------------------


class ByteBackend:
    """Byte-level fake: encode = UTF-8 bytes, decode = best-effort UTF-8."""

    fingerprint = "fake-byte-backend-cycle35"
    vocab_size = 256

    @staticmethod
    def encode(text):
        """Encode."""
        if type(text) is not str:
            raise ValueError("invalid_text")
        return tuple(text.encode("utf-8"))

    @staticmethod
    def decode(token_ids):
        """Decode."""
        return bytes(b & 0xFF for b in tuple(token_ids)).decode("utf-8", errors="replace")

    @staticmethod
    def generate(input_ids, *, max_new_tokens, seed, cancel):
        """Generate."""
        raise AssertionError("ByteBackend.generate is not scripted for this test")


class SeededFakeBackend:
    """Deterministic seeded fake honoring the Backend protocol.

    ``generate`` uses a request-local RNG derived from ``(seed,
    input_ids)`` via SHA-256 (stable across interpreters), emits exactly
    ``max_new_tokens`` ids unless cancelled, and never stops early.
    """

    fingerprint = "fake-seeded-backend-cycle35"
    vocab_size = 256
    cap = 64

    @staticmethod
    def encode(text):
        """Encode."""
        if type(text) is not str:
            raise ValueError("invalid_text")
        return tuple(text.encode("utf-8"))

    @staticmethod
    def decode(token_ids):
        """Decode."""
        if type(token_ids) is not tuple:
            raise ValueError("invalid_decode_ids")
        return bytes(b & 0xFF for b in token_ids).decode("utf-8", errors="replace")

    def generate(self, input_ids, *, max_new_tokens, seed, cancel):
        """Generate."""
        if (
            type(input_ids) is not tuple
            or not input_ids
            or any(type(t) is not int or not 0 <= t < 256 for t in input_ids)
        ):
            return Generation((), "error", "invalid_or_empty_input")
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= self.cap:
            return Generation((), "error", "invalid_output_cap")
        if type(seed) is not int or not 0 <= seed < 2**64:
            return Generation((), "error", "invalid_seed")
        if not isinstance(cancel, Event):
            return Generation((), "error", "invalid_cancellation_event")
        if cancel.is_set():
            return Generation((), "cancelled")
        digest = sha256(f"{seed}\0{input_ids!r}".encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))  # noqa: S311 -- deterministic test seeding
        out = []
        for _ in range(max_new_tokens):
            if cancel.is_set():
                return Generation(tuple(out), "cancelled")
            out.append(rng.randrange(256))
        return Generation(tuple(out), "cancelled" if cancel.is_set() else "limit")


class ExplodingDecodeBackend(ByteBackend):
    """decode raises (or returns non-str) to exercise degrade paths."""

    fingerprint = "fake-exploding-backend-cycle35"

    def __init__(self, mode="raise"):
        self.mode = mode

    def decode(self, token_ids):
        """Decode."""
        if self.mode == "raise":
            raise RuntimeError("decode exploded")
        return b"not-a-str"


class ScriptedDecodeBackend(ByteBackend):
    """decode answers from a per-id script (may yield empty strings)."""

    fingerprint = "fake-scripted-decode-cycle35"

    def __init__(self, mapping):
        self._mapping = dict(mapping)

    def decode(self, token_ids):
        """Decode."""
        return "".join(self._mapping[t] for t in tuple(token_ids))


class FlippingEvent(Event):
    """Event whose is_set() turns True after a fixed number of calls."""

    def __init__(self, flip_after):
        super().__init__()
        self._calls = 0
        self._flip_after = flip_after

    def is_set(self):
        """Is set."""
        self._calls += 1
        return self._calls > self._flip_after


# ---------------------------------------------------------------------------
# Independent oracles (reimplemented from docstrings, different structure)
# ---------------------------------------------------------------------------


def oracle_valid_ids(value):
    """Oracle valid ids."""
    return type(value) is tuple and all(type(t) is int and t >= 0 for t in value)


# skipcq: PY-R1000 - Gate-pinned frozen function; splitting breaks its verified contract.
def oracle_hf_overhang_match(token_texts, stop):
    """Suffix-of-concatenation view of the HF overhang match rule.

    The stop fires iff some end-truncation of the final representation
    covers a suffix of the stop string AND the remaining stop prefix is a
    suffix of the concatenation of the non-empty, same-typed
    representations in the trailing window. A type-mismatched
    representation below the final one is an opaque wall (the walk
    breaks); an empty same-typed one is transparent. This is
    structurally different from the implementation's incremental
    end-overlap set + backwards walk.
    """
    if not stop or not token_texts:
        return False
    if type(token_texts[-1]) is not type(stop):
        return False
    target = len(stop)
    count = len(token_texts)
    earliest = max(0, count - target)
    last = token_texts[-1]
    empty = stop[:0]
    for dropped in range(len(last)):
        head = last[: len(last) - dropped]
        width = len(head)
        need = stop if width >= target else stop[target - width :]
        if not head.endswith(need):
            continue
        overlap = width if width < target else target
        if overlap >= target:
            return True
        rest = stop[: target - overlap]
        # Nearest type-mismatched representation below the final one walls
        # off everything beneath it.
        floor = earliest
        for pos in range(count - 2, earliest - 1, -1):
            if type(token_texts[pos]) is not type(stop):
                floor = pos + 1
                break
        for low in range(floor, count):
            parts = [
                piece
                for piece in token_texts[low : count - 1]
                if type(piece) is type(stop) and len(piece) > 0
            ]
            if empty.join(parts).endswith(rest):
                return True
    return False


def oracle_first_firing_step(token_texts, stop, base, total):
    """Oracle first firing step."""
    for step in range(1, total + 1):
        if oracle_hf_overhang_match(token_texts[: base + step], stop):
            return step
    return None


# skipcq: PY-R1000 - Gate-pinned frozen function; splitting breaks its verified contract.
def oracle_emulate_hf(backend, prompt_ids, output_ids, stop_text, token_bytes=None):
    """Oracle emulate hf."""
    if (
        not oracle_valid_ids(prompt_ids)
        or not oracle_valid_ids(output_ids)
        or type(stop_text) is not str
        or not stop_text
    ):
        return StopOutcome(False, False, tuple(output_ids), "invalid_emulation_input")
    if token_bytes is not None and not callable(token_bytes):
        return StopOutcome(False, False, tuple(output_ids), "invalid_emulation_input")
    base = len(prompt_ids)
    try:
        if token_bytes is None:
            texts = [backend.decode((token,)) for token in (*prompt_ids, *output_ids)]
            if any(type(text) is not str for text in texts):
                return StopOutcome(False, False, tuple(output_ids), "token_text_unavailable")
            step = oracle_first_firing_step(texts, stop_text, base, len(output_ids))
        else:
            byte_texts = []
            for token in (*prompt_ids, *output_ids):
                mapped = token_bytes(token)
                if mapped is None:
                    decoded = backend.decode((token,))
                    if type(decoded) is not str:
                        return StopOutcome(
                            False, False, tuple(output_ids), "token_text_unavailable"
                        )
                    mapped = decoded.encode("utf-8")
                if type(mapped) is not bytes and type(mapped) is not _OOVTokenBytes:
                    return StopOutcome(False, False, tuple(output_ids), "token_text_unavailable")
                byte_texts.append(mapped)
            step = oracle_first_firing_step(
                byte_texts, stop_text.encode("utf-8"), base, len(output_ids)
            )
    except Exception:  # noqa: BLE001 -- oracle mirrors the implementation's degrade path
        return StopOutcome(False, False, tuple(output_ids), "token_text_unavailable")
    if step is None:
        return StopOutcome(True, False, tuple(output_ids), "no_stop_within_cap")
    return StopOutcome(True, True, tuple(output_ids[:step]), f"fired_at_step_{step}")


# skipcq: PY-R1000 - Gate-pinned frozen function; splitting breaks its verified contract.
def oracle_emulate_vllm(
    backend, output_ids, stop_text, include_stop_str_in_output=False, min_tokens=0
):
    """Oracle emulate vllm."""
    if not oracle_valid_ids(output_ids) or type(stop_text) is not str or not stop_text:
        return StopOutcome(False, False, tuple(output_ids), "invalid_emulation_input")
    if (
        type(include_stop_str_in_output) is not bool
        or type(min_tokens) is not int
        or min_tokens < 0
    ):
        return StopOutcome(False, False, tuple(output_ids), "invalid_emulation_input")
    stop_len = len(stop_text)
    texts = []
    try:
        for step in range(1, len(output_ids) + 1):
            text = backend.decode(tuple(output_ids[:step]))
            if type(text) is not str:
                return StopOutcome(False, False, tuple(output_ids), "decode_unavailable")
            texts.append(text)
    except Exception:  # noqa: BLE001 -- oracle mirrors the implementation's degrade path
        return StopOutcome(False, False, tuple(output_ids), "decode_unavailable")
    for step, text in enumerate(texts, start=1):
        prev_len = len(texts[step - 2]) if step > 1 else 0
        new_char_count = len(text) - prev_len
        if new_char_count == 0:
            continue
        index = text.find(stop_text, 1 - new_char_count - stop_len)
        if index < 0:
            continue
        if step <= min_tokens:
            continue
        stop_end = index + stop_len
        # Linear scan for the first prefix offset >= stop_end (the
        # implementation uses bisect_left; same result on sorted offsets).
        kept = len(texts)
        for candidate in range(1, len(texts) + 1):
            if len(texts[candidate - 1]) >= stop_end:
                kept = candidate
                break
        returned = text[:stop_end] if include_stop_str_in_output else text[:index]
        return StopOutcome(True, True, tuple(output_ids[:kept]), f"fired_at_step_{step}", returned)
    if min_tokens >= len(output_ids) and min_tokens > 0:
        return StopOutcome(True, False, tuple(output_ids), "gated_by_min_tokens")
    return StopOutcome(True, False, tuple(output_ids), "no_stop_within_cap")


def oracle_emulate_sglang(backend, output_ids, stop_text, no_stop_trim=False):
    """Oracle emulate sglang."""
    if not oracle_valid_ids(output_ids) or type(stop_text) is not str or not stop_text:
        return StopOutcome(False, False, tuple(output_ids), "invalid_emulation_input")
    if type(no_stop_trim) is not bool:
        return StopOutcome(False, False, tuple(output_ids), "invalid_emulation_input")
    stop_len = len(stop_text)
    try:
        for step in range(1, len(output_ids) + 1):
            text = backend.decode(tuple(output_ids[:step]))
            if type(text) is not str:
                return StopOutcome(False, False, tuple(output_ids), "decode_unavailable")
            index = text.find(stop_text)
            if index < 0:
                continue
            stop_end = index + stop_len
            returned = text[:stop_end] if no_stop_trim else text[:index]
            return StopOutcome(
                True, True, tuple(output_ids[:step]), f"fired_at_step_{step}", returned
            )
    except Exception:  # noqa: BLE001 -- oracle mirrors the implementation's degrade path
        return StopOutcome(False, False, tuple(output_ids), "decode_unavailable")
    return StopOutcome(True, False, tuple(output_ids), "no_stop_within_cap")


def oracle_emulate_eos(output_ids, eos_id):
    """Oracle emulate eos."""
    ids = tuple(output_ids)
    if eos_id is None:
        return StopOutcome(False, False, ids, "eos_id_unavailable")
    if type(eos_id) is not int or eos_id < 0:
        return StopOutcome(False, False, ids, "invalid_eos_id")
    for index, token in enumerate(ids):
        if token == eos_id:
            return StopOutcome(True, True, ids[:index], f"fired_at_step_{index + 1}")
    return StopOutcome(True, False, ids, "no_stop_within_cap")


# ---------------------------------------------------------------------------
# Fuzz helpers
# ---------------------------------------------------------------------------


def fuzz_rng(name):
    """Fuzz rng."""
    return random.Random(f"cycle35-backends-{name}")  # noqa: S311 -- deterministic test seeding


STR_TOKENS = ["a", "b", "ab", "ba", "las", "topper", "stop", "or", "", " ", "xab", "q"]
STR_STOPS = ["stop", "or", "ab", "a", "abc", "topper", " ", "qab", "zzz"]


def random_texts(rng, alphabet, oov=False):
    """Random texts."""
    count = rng.randint(0, 6)
    out = [rng.choice(alphabet) for _ in range(count)]
    if oov and alphabet and isinstance(alphabet[0], bytes) and rng.random() < 0.4:
        out.insert(rng.randint(0, len(out)), _OOVTokenBytes())
    return out


# ---------------------------------------------------------------------------
# Backend contract: synthetic backend behavior
# ---------------------------------------------------------------------------


class TestSyntheticBackendContract(unittest.TestCase):
    """Pins the Backend protocol contract on a deterministic seeded fake."""

    def setUp(self):
        """Set up the test fixture."""
        self.backend = SeededFakeBackend()

    def test_fingerprint_is_str(self):
        """Test fingerprint is str."""
        self.assertIs(type(self.backend.fingerprint), str)

    def test_encode_returns_tuple_of_ints(self):
        """Test encode returns tuple of ints."""
        ids = self.backend.encode("hello world")
        self.assertIs(type(ids), tuple)
        self.assertTrue(all(type(t) is int and 0 <= t < 256 for t in ids))
        self.assertEqual(ids, tuple(b"hello world"))

    def test_encode_rejects_non_str(self):
        """Test encode rejects non str."""
        for bad in (None, 5, b"x", ["a"], True):
            with self.assertRaises(ValueError, msg=repr(bad)):
                self.backend.encode(bad)

    def test_decode_returns_str(self):
        """Test decode returns str."""
        self.assertIs(type(self.backend.decode((104, 105))), str)

    def test_generate_is_deterministic(self):
        """Test generate is deterministic."""
        args = {"max_new_tokens": 8, "seed": 7, "cancel": Event()}
        first = self.backend.generate((1, 2, 3), **args)
        second = self.backend.generate((1, 2, 3), **args)
        self.assertEqual(first.token_ids, second.token_ids)
        self.assertEqual(first.reason, second.reason)
        self.assertEqual(first.reason, "limit")
        self.assertIsNone(first.error)

    def test_generate_seed_changes_output(self):
        """Test generate seed changes output."""
        seen = {
            self.backend.generate((1, 2, 3), max_new_tokens=8, seed=s, cancel=Event()).token_ids
            for s in range(20)
        }
        self.assertGreater(len(seen), 1)

    def test_generate_input_changes_output(self):
        """Test generate input changes output."""
        seen = {
            self.backend.generate(ids, max_new_tokens=8, seed=7, cancel=Event()).token_ids
            for ids in ((1,), (2,), (1, 2), (2, 1), (255,))
        }
        self.assertGreater(len(seen), 1)

    def test_generate_emits_exact_cap(self):
        """Test generate emits exact cap."""
        for cap in (1, 5, 64):
            gen = self.backend.generate((9,), max_new_tokens=cap, seed=3, cancel=Event())
            self.assertEqual(len(gen.token_ids), cap)
            self.assertEqual(gen.reason, "limit")
            self.assertTrue(all(type(t) is int and 0 <= t < 256 for t in gen.token_ids))

    def test_generate_precancelled_event(self):
        """Test generate precancelled event."""
        cancel = Event()
        cancel.set()
        gen = self.backend.generate((1, 2), max_new_tokens=8, seed=1, cancel=cancel)
        self.assertEqual(gen.token_ids, ())
        self.assertEqual(gen.reason, "cancelled")

    def test_generate_mid_run_cancellation(self):
        """Test generate mid run cancellation."""
        # Flips after 3 is_set() calls: 1 pre-check + 2 loop checks, so two
        # tokens are emitted before the cancellation is observed.
        gen = self.backend.generate(
            (1, 2), max_new_tokens=8, seed=1, cancel=FlippingEvent(flip_after=3)
        )
        self.assertEqual(gen.reason, "cancelled")
        self.assertEqual(len(gen.token_ids), 2)

    def test_generate_error_vocabulary(self):
        """Test generate error vocabulary."""
        bad_calls = [
            {"input_ids": (), "max_new_tokens": 4, "seed": 1, "cancel": Event()},
            {"input_ids": [1, 2], "max_new_tokens": 4, "seed": 1, "cancel": Event()},
            {"input_ids": (True,), "max_new_tokens": 4, "seed": 1, "cancel": Event()},
            {"input_ids": (1,), "max_new_tokens": 0, "seed": 1, "cancel": Event()},
            {"input_ids": (1,), "max_new_tokens": 65, "seed": 1, "cancel": Event()},
            {"input_ids": (1,), "max_new_tokens": "4", "seed": 1, "cancel": Event()},
            {"input_ids": (1,), "max_new_tokens": 4, "seed": -1, "cancel": Event()},
            {"input_ids": (1,), "max_new_tokens": 4, "seed": 2**64, "cancel": Event()},
            {"input_ids": (1,), "max_new_tokens": 4, "seed": True, "cancel": Event()},
            {"input_ids": (1,), "max_new_tokens": 4, "seed": 1, "cancel": "nope"},
        ]
        for kwargs in bad_calls:
            gen = self.backend.generate(**kwargs)
            self.assertEqual(gen.reason, "error", msg=repr(kwargs))
            self.assertEqual(gen.token_ids, ())
            self.assertIs(type(gen.error), str)

    def test_generate_result_is_generation(self):
        """Test generate result is generation."""
        gen = self.backend.generate((1,), max_new_tokens=2, seed=0, cancel=Event())
        self.assertIs(type(gen), Generation)
        self.assertIn(gen.reason, ("limit", "cancelled", "error"))


# ---------------------------------------------------------------------------
# Asset loading / validation helpers
# ---------------------------------------------------------------------------


def _write_json(path, value):
    """Write json."""
    path.write_text(json.dumps(value), encoding="utf-8")


# skipcq: PTC-W0046 - shared base: subclasses carry the test methods; the base runs none by design
class AssetHelpers(unittest.TestCase):
    """Temp-dir scaffolding shared by the loading tests."""

    def setUp(self):
        """Set up the test fixture."""
        self.tmp = Path(tempfile.mkdtemp(prefix="cycle35-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def make_dir(self, name="model"):
        """Make dir."""
        root = self.tmp / name
        root.mkdir(parents=True, exist_ok=True)
        return root

    @staticmethod
    def write_layout(
        root,
        *,
        model_type="gpt2",
        dropout=None,
        single=True,
        index=None,
        adapter=False,
        generation_config=False,
        with_config=True,
    ):
        """Write layout."""
        if adapter:
            _write_json(root / "adapter_config.json", {"peft_type": "LORA"})
        if with_config:
            _write_json(root / "config.json", {"model_type": model_type})
        tok = {"model": {}}
        if dropout is not None:
            tok["model"]["dropout"] = dropout
        _write_json(root / "tokenizer.json", tok)
        if generation_config:
            _write_json(root / "generation_config.json", {})
        if single:
            (root / "model.safetensors").write_bytes(b"\x00" * 32)
        if index is not None:
            _write_json(root / "model.safetensors.index.json", {"weight_map": index})
            if isinstance(index, dict):
                for shard in index.values():
                    if isinstance(shard, str):
                        (root / shard).write_bytes(b"\x00" * 16)


class TestDirectory(AssetHelpers):
    """Test directory."""

    def test_valid_directory(self):
        """Test valid directory."""
        root = self.make_dir()
        self.assertEqual(_directory(root), root.resolve())

    def test_valid_directory_string(self):
        """Test valid directory string."""
        root = self.make_dir()
        self.assertEqual(_directory(str(root)), root.resolve())

    def test_rejects_file(self):
        """Test rejects file."""
        target = self.tmp / "f.bin"
        target.write_bytes(b"x")
        with self.assertRaises(ValueError) as ctx:
            _directory(target)
        self.assertEqual(str(ctx.exception), "invalid_local_directory")

    def test_rejects_missing(self):
        """Test rejects missing."""
        with self.assertRaises(ValueError) as ctx:
            _directory(self.tmp / "nope")
        self.assertEqual(str(ctx.exception), "invalid_local_directory")

    def test_rejects_empty_string(self):
        """Test rejects empty string."""
        with self.assertRaises(ValueError) as ctx:
            _directory("")
        self.assertEqual(str(ctx.exception), "invalid_local_directory")

    def test_rejects_non_path_types(self):
        """Test rejects non path types."""
        for bad in (None, 0, b"x", ["a"], True):
            with self.assertRaises(ValueError, msg=repr(bad)):
                _directory(bad)


class TestMember(AssetHelpers):
    """Test member."""

    def setUp(self):
        """Set up the test fixture."""
        super().setUp()
        self.root = self.make_dir()
        (self.root / "config.json").write_bytes(b"{}")

    def test_valid_member(self):
        """Test valid member."""
        self.assertEqual(_member(self.root, "config.json"), self.root / "config.json")

    def test_missing_asset(self):
        """Test missing asset."""
        with self.assertRaises(ValueError) as ctx:
            _member(self.root, "absent.json")
        self.assertEqual(str(ctx.exception), "missing_or_unsafe_asset")

    def test_symlink_rejected(self):
        """Test symlink rejected."""
        link = self.root / "link.json"
        try:
            link.symlink_to(self.root / "config.json")
        except OSError:
            self.skipTest("symlinks unsupported here")
        with self.assertRaises(ValueError) as ctx:
            _member(self.root, "link.json")
        self.assertEqual(str(ctx.exception), "missing_or_unsafe_asset")

    def test_directory_rejected(self):
        """Test directory rejected."""
        (self.root / "sub").mkdir()
        with self.assertRaises(ValueError) as ctx:
            _member(self.root, "sub")
        self.assertEqual(str(ctx.exception), "missing_or_unsafe_asset")

    def test_dotdot_rejected_as_missing(self):
        """Test dotdot rejected as missing."""
        # ".." passes the name regex but is not a file.
        with self.assertRaises(ValueError) as ctx:
            _member(self.root, "..")
        self.assertEqual(str(ctx.exception), "missing_or_unsafe_asset")

    def test_traversal_names_rejected(self):
        """Test traversal names rejected."""
        for bad in ("a/b", "", "a b", "a*b", "../x", "x\x00y"):
            with self.assertRaises(ValueError, msg=repr(bad)) as ctx:
                _member(self.root, bad)
            self.assertEqual(str(ctx.exception), "invalid_asset_name")


class TestReadJson(AssetHelpers):
    """Test read json."""

    def test_valid_dict(self):
        """Test valid dict."""
        target = self.tmp / "c.json"
        _write_json(target, {"a": 1})
        self.assertEqual(_read_json(target), {"a": 1})

    def test_non_dict_rejected(self):
        """Test non dict rejected."""
        for value in ([1, 2], "str", 5, None, True):
            target = self.tmp / "c.json"
            _write_json(target, value)
            with self.assertRaises(ValueError, msg=repr(value)) as ctx:
                _read_json(target)
            self.assertEqual(str(ctx.exception), "invalid_asset_json")

    def test_malformed_rejected(self):
        """Test malformed rejected."""
        target = self.tmp / "c.json"
        target.write_bytes(b"{oops")
        with self.assertRaises(ValueError) as ctx:
            _read_json(target)
        self.assertEqual(str(ctx.exception), "invalid_asset_json")

    def test_non_utf8_rejected(self):
        """Test non utf8 rejected."""
        target = self.tmp / "c.json"
        target.write_bytes(b'{"a": "\xff"}')
        with self.assertRaises(ValueError) as ctx:
            _read_json(target)
        self.assertEqual(str(ctx.exception), "invalid_asset_json")

    def test_deeply_nested_rejected(self):
        """Test deeply nested rejected."""
        target = self.tmp / "c.json"
        target.write_text("[" * 100000 + "]" * 100000, encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            _read_json(target)
        self.assertEqual(str(ctx.exception), "invalid_asset_json")

    def test_oversized_rejected(self):
        """Test oversized rejected."""
        target = self.tmp / "big.json"
        # Sparse file: same size on disk-metadata and on read (holes read as
        # zeros), but consumes ~no blocks. The rejection depends only on the
        # byte count, so this exercises the identical code path while staying
        # safe on small tmpfs mounts (a full 128 MiB write hit ENOSPC).
        with target.open("wb") as handle:
            handle.seek(128 * 1024 * 1024)
            handle.write(b"\x00")
        self.assertEqual(target.stat().st_size, 128 * 1024 * 1024 + 1)
        with self.assertRaises(ValueError) as ctx:
            _read_json(target)
        self.assertEqual(str(ctx.exception), "oversized_asset_json")

    def test_exactly_at_cap_parses(self):
        """Test exactly at cap parses."""
        # The bound is strict `>`: exactly 128 MiB is accepted.
        # This needs 128 MiB of real bytes (valid JSON); skip honestly when
        # the scratch filesystem cannot hold it instead of failing with
        # ENOSPC, which says nothing about the implementation.
        free = shutil.disk_usage(self.tmp).free
        if free < 256 * 1024 * 1024:
            self.skipTest(f"needs 256 MiB scratch space, have {free // (1024 * 1024)} MiB")
        target = self.tmp / "edge.json"
        with target.open("wb") as handle:
            handle.write(b'{"a":"')
            handle.write(b"x" * (128 * 1024 * 1024 - 10))
            handle.write(b'"}')
        value = _read_json(target)
        self.assertIs(type(value), dict)
        self.assertEqual(len(value["a"]), 128 * 1024 * 1024 - 10)


class TestModelAssets(AssetHelpers):
    """Test model assets."""

    def test_single_layout(self):
        """Test single layout."""
        root = self.make_dir()
        self.write_layout(root)
        self.assertEqual(_model_assets(root), ("config.json", "model.safetensors"))

    def test_sorted_names(self):
        """Test sorted names."""
        root = self.make_dir()
        self.write_layout(root, generation_config=True)
        self.assertEqual(
            _model_assets(root),
            ("config.json", "generation_config.json", "model.safetensors"),
        )

    def test_index_layout(self):
        """Test index layout."""
        root = self.make_dir()
        self.write_layout(root, single=False, index={"w1": "shard-a.safetensors"})
        self.assertEqual(
            _model_assets(root),
            ("config.json", "model.safetensors.index.json", "shard-a.safetensors"),
        )

    def test_adapter_rejected(self):
        """Test adapter rejected."""
        root = self.make_dir()
        self.write_layout(root, adapter=True)
        with self.assertRaises(ValueError) as ctx:
            _model_assets(root)
        self.assertEqual(str(ctx.exception), "adapters_not_supported")

    def test_both_layouts_rejected(self):
        """Test both layouts rejected."""
        root = self.make_dir()
        self.write_layout(root, single=True, index={"w1": "shard-a.safetensors"})
        with self.assertRaises(ValueError) as ctx:
            _model_assets(root)
        self.assertEqual(str(ctx.exception), "require_one_standard_safetensors_layout")

    def test_neither_layout_rejected(self):
        """Test neither layout rejected."""
        root = self.make_dir()
        self.write_layout(root, single=False)
        with self.assertRaises(ValueError) as ctx:
            _model_assets(root)
        self.assertEqual(str(ctx.exception), "require_one_standard_safetensors_layout")

    def test_missing_config_rejected(self):
        """Test missing config rejected."""
        root = self.make_dir()
        self.write_layout(root, with_config=False)
        with self.assertRaises(ValueError) as ctx:
            _model_assets(root)
        self.assertEqual(str(ctx.exception), "missing_or_unsafe_asset")

    def test_bad_weight_index(self):
        """Test bad weight index."""
        for index in (
            ["w1"],
            {},
            {"w1": 5},
            {"w1": "../evil.safetensors"},
            {"w1": "shard safetensors"},
            {"w1": "shard.bin"},
        ):
            root = self.make_dir()
            self.write_layout(root, single=False, index=index)
            with self.assertRaises(ValueError, msg=repr(index)) as ctx:
                _model_assets(root)
            self.assertEqual(str(ctx.exception), "invalid_weight_index")

    def test_missing_shard_rejected(self):
        """Test missing shard rejected."""
        root = self.make_dir()
        _write_json(root / "config.json", {"model_type": "gpt2"})
        _write_json(root / "tokenizer.json", {"model": {}})
        _write_json(
            root / "model.safetensors.index.json", {"weight_map": {"w": "gone.safetensors"}}
        )
        with self.assertRaises(ValueError) as ctx:
            _model_assets(root)
        self.assertEqual(str(ctx.exception), "missing_or_unsafe_asset")


class TestManifest(AssetHelpers):
    """Test manifest."""

    def test_records(self):
        """Test records."""
        root = self.make_dir()
        self.write_layout(root)
        records = _manifest(root, ("config.json", "model.safetensors"))
        self.assertEqual([r["name"] for r in records], ["config.json", "model.safetensors"])
        for record in records:
            data = (root / record["name"]).read_bytes()
            self.assertEqual(record["bytes"], len(data))
            self.assertEqual(record["sha256"], sha256(data).hexdigest())
            self.assertEqual(sorted(record.keys()), ["bytes", "name", "sha256"])

    def test_order_preserved(self):
        """Test order preserved."""
        root = self.make_dir()
        self.write_layout(root)
        records = _manifest(root, ("model.safetensors", "config.json"))
        self.assertEqual([r["name"] for r in records], ["model.safetensors", "config.json"])


class TestIdsValid(unittest.TestCase):
    """Test ids valid."""

    def test_valid(self):
        """Test valid."""
        self.assertTrue(_ids_valid((0, 1, 9), 10))
        self.assertTrue(_ids_valid((), 10))

    def test_invalid(self):
        """Test invalid."""
        self.assertFalse(_ids_valid([0, 1], 10))
        self.assertFalse(_ids_valid((True,), 10))
        self.assertFalse(_ids_valid((-1,), 10))
        self.assertFalse(_ids_valid((10,), 10))
        self.assertFalse(_ids_valid(("1",), 10))
        self.assertFalse(_ids_valid(None, 10))
        self.assertFalse(_ids_valid((1.0,), 10))


# ---------------------------------------------------------------------------
# LocalHFBackend: fail-closed initialization (no checkpoint here: every
# constructor path below must end in a fixed RuntimeError code)
# ---------------------------------------------------------------------------


class TestLocalHFBackendInit(AssetHelpers):
    """Test local hf backend init."""

    def assertInitFailed(self, *args, **kwargs):
        """AssertInitFailed."""
        with self.assertRaises(RuntimeError) as ctx:
            LocalHFBackend(*args, **kwargs)
        self.assertEqual(str(ctx.exception), "local_backend_initialization_failed")

    def test_missing_directory(self):
        """Test missing directory."""
        self.assertInitFailed(self.tmp / "nope")

    def test_file_not_directory(self):
        """Test file not directory."""
        target = self.tmp / "f.bin"
        target.write_bytes(b"x")
        self.assertInitFailed(target)

    def test_bad_context_limits(self):
        """Test bad context limits."""
        root = self.make_dir()
        self.write_layout(root)
        for bad in (0, -1, 2**20, "2048", 2.0, True, None):
            self.assertInitFailed(root, max_context=bad)

    def test_bad_device(self):
        """Test bad device."""
        root = self.make_dir()
        self.write_layout(root)
        for bad in ("xpu", "", "CPU", None, 0):
            self.assertInitFailed(root, device=bad)

    def test_empty_directory(self):
        """Test empty directory."""
        self.assertInitFailed(self.make_dir())

    def test_adapter_config_rejected(self):
        """Test adapter config rejected."""
        root = self.make_dir()
        self.write_layout(root, adapter=True)
        self.assertInitFailed(root)

    def test_unsupported_model_type(self):
        """Test unsupported model type."""
        root = self.make_dir()
        self.write_layout(root, model_type="bert")
        self.assertInitFailed(root)

    def test_stochastic_tokenizer_rejected(self):
        """Test stochastic tokenizer rejected."""
        root = self.make_dir()
        self.write_layout(root, dropout=0.5)
        self.assertInitFailed(root)

    def test_context_window_exceeded_distinct_code(self):
        """Test context window exceeded distinct code."""
        # A valid layout whose declared window (GPT-2 default n_positions =
        # 1024) is below the requested max_context takes the distinct
        # _ContextWindowError path, not the generic init failure.
        root = self.make_dir()
        self.write_layout(root)
        with self.assertRaises(RuntimeError) as ctx:
            LocalHFBackend(root)
        self.assertEqual(str(ctx.exception), "local_context_limit_exceeds_declared_window")

    def test_tokenizer_parse_failure_fails_closed(self):
        """Test tokenizer parse failure fails closed."""
        # Within the declared window, init proceeds to tokenizer parsing and
        # fails closed on the fake tokenizer.json with the generic code.
        root = self.make_dir()
        self.write_layout(root)
        with self.assertRaises(RuntimeError) as ctx:
            LocalHFBackend(root, max_context=512)
        self.assertEqual(str(ctx.exception), "local_backend_initialization_failed")
        self.assertNotIn("tokenizer", str(ctx.exception))

    def test_no_inner_vocabulary_leaks(self):
        """Test no inner vocabulary leaks."""
        # The fine-grained inner codes are unobservable (D5).
        root = self.make_dir()
        self.write_layout(root, model_type="bert")
        try:
            LocalHFBackend(root)
        except RuntimeError as exc:
            self.assertNotIn("unsupported_model_type", str(exc))
            self.assertNotIn("bert", str(exc))


# ---------------------------------------------------------------------------
# LocalHFBackend: validation layers via white-box instances (no checkpoint)
# ---------------------------------------------------------------------------


class StubTokenizer:
    """Tokenizers stand-in recording calls; scriptable encode/decode."""

    def __init__(self, enc=None, dec=None):
        self._enc = enc
        self._dec = dec
        self.calls = []

    def encode(self, text, **kwargs):
        """Encode."""
        self.calls.append(("encode", text, kwargs))
        if self._enc is None:
            return tuple(text.encode("utf-8"))
        result = self._enc(text)
        if isinstance(result, Exception):
            raise result
        return result

    def decode(self, ids, **kwargs):
        """Decode."""
        self.calls.append(("decode", tuple(ids), kwargs))
        if self._dec is None:
            return bytes(ids).decode("utf-8", errors="replace")
        result = self._dec(ids)
        if isinstance(result, Exception):
            raise result
        return result


def make_whitebox(tokenizer=None, vocab_size=16, tokenizer_ids=frozenset(range(8))):
    """Make whitebox."""
    obj = LocalHFBackend.__new__(LocalHFBackend)
    obj._lock = RLock()
    obj._model = object()
    obj._tokenizer = tokenizer if tokenizer is not None else StubTokenizer()
    obj._torch = None
    obj._device = None
    obj._vocab_size = vocab_size
    obj._tokenizer_ids = tokenizer_ids
    obj._context_limit = 32
    obj.fingerprint = "whitebox"
    obj.seed_used_for_selection = False
    obj.eos_token_id = None
    obj.hf_stop_token_bytes = None
    return obj


class TestLocalHFBackendValidation(unittest.TestCase):
    """Test local hf backend validation."""

    def test_ready_uninitialized(self):
        """Test ready uninitialized."""
        obj = LocalHFBackend.__new__(LocalHFBackend)
        obj._tokenizer = None
        obj._model = None
        with self.assertRaises(RuntimeError) as ctx:
            obj._ready()
        self.assertEqual(str(ctx.exception), "local_backend_not_initialized")

    def test_encode_ok(self):
        """Test encode ok."""
        backend = make_whitebox(vocab_size=256, tokenizer_ids=frozenset(range(256)))
        self.assertEqual(backend.encode("hi"), tuple(b"hi"))
        kind, text, kwargs = backend._tokenizer.calls[0]
        self.assertEqual(kind, "encode")
        self.assertEqual(text, "hi")
        self.assertEqual(
            kwargs, {"add_special_tokens": False, "truncation": False, "padding": False}
        )

    def test_encode_rejects_non_str(self):
        """Test encode rejects non str."""
        backend = make_whitebox()
        for bad in (None, 5, b"x", True, ["a"]):
            with self.assertRaises(ValueError, msg=repr(bad)) as ctx:
                backend.encode(bad)
            self.assertEqual(str(ctx.exception), "invalid_text")

    def test_encode_tokenizer_failure_maps_to_fixed_code(self):
        """Test encode tokenizer failure maps to fixed code."""
        backend = make_whitebox(StubTokenizer(enc=RuntimeError("boom")))
        with self.assertRaises(RuntimeError) as ctx:
            backend.encode("hi")
        self.assertEqual(str(ctx.exception), "local_encode_failed")

    def test_encode_invalid_ids_swallowed(self):
        """Test encode invalid ids swallowed."""
        # D3: the adapter's own invalid_encoded_ids is raised inside the
        # sanitization boundary and surfaces as local_encode_failed.
        backend = make_whitebox(StubTokenizer(enc=lambda text: (999,)))
        with self.assertRaises(RuntimeError) as ctx:
            backend.encode("hi")
        self.assertEqual(str(ctx.exception), "local_encode_failed")

    def test_encode_list_ids_coerced_to_tuple(self):
        """Test encode list ids coerced to tuple."""
        # The adapter wraps the tokenizer result in tuple(): a list return
        # is normalized, not rejected.
        backend = make_whitebox(StubTokenizer(enc=lambda text: [1, 2]))
        self.assertEqual(backend.encode("hi"), (1, 2))

    def test_encode_non_iterable_ids_swallowed(self):
        """Test encode non iterable ids swallowed."""
        backend = make_whitebox(StubTokenizer(enc=lambda text: 5))
        with self.assertRaises(RuntimeError) as ctx:
            backend.encode("hi")
        self.assertEqual(str(ctx.exception), "local_encode_failed")

    def test_decode_ok(self):
        """Test decode ok."""
        backend = make_whitebox(vocab_size=256, tokenizer_ids=frozenset(range(256)))
        self.assertEqual(backend.decode((104, 105)), "hi")
        kind, ids, kwargs = backend._tokenizer.calls[0]
        self.assertEqual(kind, "decode")
        self.assertEqual(ids, (104, 105))
        self.assertEqual(
            kwargs,
            {
                "skip_special_tokens": False,
                "clean_up_tokenization_spaces": False,
            },
        )

    def test_decode_empty_tuple_valid(self):
        """Test decode empty tuple valid."""
        backend = make_whitebox()
        self.assertEqual(backend.decode(()), "")

    def test_decode_rejects_bad_ids(self):
        """Test decode rejects bad ids."""
        backend = make_whitebox()
        for bad in ((16,), (8,), [1], (True,), (-1,), ("1",), None):
            with self.assertRaises(ValueError, msg=repr(bad)) as ctx:
                backend.decode(bad)
            self.assertEqual(str(ctx.exception), "invalid_decode_ids")

    def test_decode_tokenizer_failure_maps_to_fixed_code(self):
        """Test decode tokenizer failure maps to fixed code."""
        backend = make_whitebox(StubTokenizer(dec=RuntimeError("boom")))
        with self.assertRaises(RuntimeError) as ctx:
            backend.decode((1,))
        self.assertEqual(str(ctx.exception), "local_decode_failed")

    def test_decode_does_not_enforce_str(self):
        """Test decode does not enforce str."""
        # D4: the `-> str` annotation is unchecked; a non-str tokenizer
        # result propagates.
        backend = make_whitebox(StubTokenizer(dec=lambda ids: 42))
        self.assertEqual(backend.decode((1,)), 42)

    def test_vocab_size_property(self):
        """Test vocab size property."""
        self.assertEqual(make_whitebox(vocab_size=16).vocab_size, 16)

    def test_forward_logits_validation(self):
        """Test forward logits validation."""
        backend = make_whitebox()
        for bad in ((), "ab", [1], (True,), None):
            with self.assertRaises(ValueError, msg=repr(bad)) as ctx:
                backend.forward_logits(bad)
            self.assertEqual(str(ctx.exception), "invalid_input_ids")
        with self.assertRaises(ValueError) as ctx:
            backend.forward_logits((1,) * 33)
        self.assertEqual(str(ctx.exception), "context_limit_exceeded")

    def test_forward_logits_inference_failure_maps_to_fixed_code(self):
        """Test forward logits inference failure maps to fixed code."""
        backend = make_whitebox()
        with self.assertRaises(RuntimeError) as ctx:
            backend.forward_logits((1, 2))
        self.assertEqual(str(ctx.exception), "local_forward_failed")

    def test_generate_validation_table(self):
        """Test generate validation table."""
        backend = make_whitebox()
        event = Event()
        cases = [
            ((), 4, 1, event, "invalid_or_empty_input"),
            ([1, 2], 4, 1, event, "invalid_or_empty_input"),
            ((True,), 4, 1, event, "invalid_or_empty_input"),
            ((1, 2), 0, 1, event, "invalid_output_cap"),
            ((1, 2), 4097, 1, event, "invalid_output_cap"),
            ((1, 2), "4", 1, event, "invalid_output_cap"),
            ((1, 2), True, 1, event, "invalid_output_cap"),
            ((1, 2), 4, -1, event, "invalid_seed"),
            ((1, 2), 4, 2**64, event, "invalid_seed"),
            ((1, 2), 4, True, event, "invalid_seed"),
            ((1, 2), 4, 1, "nope", "invalid_cancellation_event"),
        ]
        for input_ids, cap, seed, cancel, code in cases:
            gen = backend.generate(input_ids, max_new_tokens=cap, seed=seed, cancel=cancel)
            self.assertEqual(gen.reason, "error", msg=repr((input_ids, cap, seed, code)))
            self.assertEqual(gen.error, code)
            self.assertEqual(gen.token_ids, ())

    def test_generate_precancelled(self):
        """Test generate precancelled."""
        backend = make_whitebox()
        cancel = Event()
        cancel.set()
        gen = backend.generate((1, 2), max_new_tokens=4, seed=1, cancel=cancel)
        self.assertEqual((gen.token_ids, gen.reason), ((), "cancelled"))

    def test_generate_context_limit(self):
        """Test generate context limit."""
        backend = make_whitebox()
        gen = backend.generate((1,) * 31, max_new_tokens=2, seed=1, cancel=Event())
        self.assertEqual((gen.reason, gen.error), ("error", "context_limit_exceeded"))

    def test_generate_inference_failure_maps_to_fixed_code(self):
        """Test generate inference failure maps to fixed code."""
        # torch is None here: the forward raises inside the try and the
        # adapter returns the fixed code with whatever partial output
        # exists (none yet).
        backend = make_whitebox()
        gen = backend.generate((1, 2), max_new_tokens=4, seed=1, cancel=Event())
        self.assertEqual(
            (gen.token_ids, gen.reason, gen.error), ((), "error", "local_inference_failed")
        )


# ---------------------------------------------------------------------------
# TruncatedPromptBackend
# ---------------------------------------------------------------------------


class RecordingInner(ByteBackend):
    """Recording inner."""

    fingerprint = "fake-inner-cycle35"

    def __init__(self):
        self.encoded = []
        self.generated = []

    def encode(self, text):
        """Encode."""
        self.encoded.append(text)
        return super().encode(text)

    def generate(self, input_ids, *, max_new_tokens, seed, cancel):
        """Generate."""
        self.generated.append((input_ids, max_new_tokens, seed, cancel))
        return Generation((9, 9), "limit", None)


class TestTruncatedPromptBackend(unittest.TestCase):
    """Test truncated prompt backend."""

    def test_constructor_validation(self):
        """Test constructor validation."""
        inner = RecordingInner()
        for bad in (0, -3, "3", 2.0, True, None):
            with self.assertRaises(ValueError, msg=repr(bad)):
                TruncatedPromptBackend(inner, bad)
        with self.assertRaises(ValueError):
            TruncatedPromptBackend(SimpleNamespace(fingerprint=123), 2)
        with self.assertRaises(ValueError):
            TruncatedPromptBackend(SimpleNamespace(), 2)

    def test_fingerprint_format(self):
        """Test fingerprint format."""
        backend = TruncatedPromptBackend(RecordingInner(), 3)
        self.assertEqual(backend.fingerprint, "fake-inner-cycle35+prompt-trunc-3b")

    def test_hf_stop_token_bytes_forwarded(self):
        """Test hf stop token bytes forwarded."""
        inner = RecordingInner()
        inner.hf_stop_token_bytes = lambda tid: b"x"
        backend = TruncatedPromptBackend(inner, 1)
        self.assertIs(backend.hf_stop_token_bytes, inner.hf_stop_token_bytes)

    def test_hf_stop_token_bytes_defaults_none(self):
        """Test hf stop token bytes defaults none."""
        backend = TruncatedPromptBackend(RecordingInner(), 1)
        self.assertIsNone(backend.hf_stop_token_bytes)

    def test_encode_truncates_mid_token(self):
        """Test encode truncates mid token."""
        inner = RecordingInner()
        backend = TruncatedPromptBackend(inner, 2)
        ids = backend.encode("hello world")
        self.assertEqual(inner.encoded, ["hello wor"])
        self.assertEqual(ids, tuple(b"hello wor"))

    def test_encode_multibyte_boundary(self):
        """Test encode multibyte boundary."""
        inner = RecordingInner()
        backend = TruncatedPromptBackend(inner, 1)
        backend.encode("héllo")
        self.assertEqual(inner.encoded, ["héll"])

    def test_encode_codepoint_split_rejected(self):
        """Test encode codepoint split rejected."""
        # "hé" is h + 2-byte é; dropping 1 byte splits the codepoint.
        backend = TruncatedPromptBackend(RecordingInner(), 1)
        with self.assertRaises(ValueError) as ctx:
            backend.encode("hé")
        self.assertEqual(str(ctx.exception), "prompt_truncation_splits_codepoint")

    def test_encode_drop_beyond_length_unwrapped(self):
        """Test encode drop beyond length unwrapped."""
        # D6: truncate_text_tail's own ValueError propagates unwrapped.
        backend = TruncatedPromptBackend(RecordingInner(), 5)
        with self.assertRaises(ValueError) as ctx:
            backend.encode("hi")
        self.assertEqual(str(ctx.exception), "drop_bytes must lie within the text's byte length")

    def test_decode_passthrough(self):
        """Test decode passthrough."""
        backend = TruncatedPromptBackend(RecordingInner(), 2)
        self.assertEqual(backend.decode((104, 105)), "hi")

    def test_generate_passthrough(self):
        """Test generate passthrough."""
        inner = RecordingInner()
        backend = TruncatedPromptBackend(inner, 2)
        cancel = Event()
        gen = backend.generate((1, 2), max_new_tokens=4, seed=9, cancel=cancel)
        self.assertEqual(gen, Generation((9, 9), "limit", None))
        self.assertEqual(inner.generated, [((1, 2), 4, 9, cancel)])


# ---------------------------------------------------------------------------
# StopOutcome
# ---------------------------------------------------------------------------


class TestStopOutcome(unittest.TestCase):
    """Test stop outcome."""

    def test_defaults(self):
        """Test defaults."""
        outcome = StopOutcome(True, False, (1, 2), "no_stop_within_cap")
        self.assertIsNone(outcome.returned_text)
        self.assertTrue(outcome.available)
        self.assertFalse(outcome.fired)

    def test_positional_construction(self):
        """Test positional construction."""
        outcome = StopOutcome(True, True, (1,), "fired_at_step_1", "kept")
        self.assertEqual(outcome.returned_text, "kept")

    def test_equality(self):
        """Test equality."""
        self.assertEqual(
            StopOutcome(True, True, (1,), "fired_at_step_1"),
            StopOutcome(True, True, (1,), "fired_at_step_1"),
        )
        self.assertNotEqual(
            StopOutcome(True, True, (1,), "fired_at_step_1"),
            StopOutcome(True, False, (1,), "no_stop_within_cap"),
        )

    def test_frozen(self):
        """Test frozen."""
        outcome = StopOutcome(True, False, (), "no_stop_within_cap")
        with self.assertRaises(AttributeError):
            outcome.fired = True  # noqa: B018 -- frozen check is the point


# ---------------------------------------------------------------------------
# Byte-space stop machinery
# ---------------------------------------------------------------------------


class TestBytesToUnicode(unittest.TestCase):
    """Test bytes to unicode."""

    def test_spec_shape(self):
        """Test spec shape."""
        mapping = _bytes_to_unicode()
        self.assertEqual(len(mapping), 256)
        self.assertEqual(set(mapping.keys()), set(range(256)))
        self.assertEqual(len(set(mapping.values())), 256)

    def test_printable_ascii_identity(self):
        """Test printable ascii identity."""
        mapping = _bytes_to_unicode()
        self.assertEqual(mapping[33], "!")
        self.assertEqual(mapping[65], "A")
        self.assertEqual(mapping[126], "~")

    def test_gpt2_space_mapping(self):
        """Test gpt2 space mapping."""
        # Byte 32 (space) is outside the printable ranges, so it maps to
        # chr(256 + 32), the "Ġ" of GPT-2 byte-level BPE.
        self.assertEqual(_bytes_to_unicode()[32], "Ġ")

    def test_control_bytes(self):
        """Test control bytes."""
        mapping = _bytes_to_unicode()
        self.assertEqual(mapping[0], "Ā")  # chr(256)
        self.assertEqual(mapping[173], chr(256 + 67))

    def test_roundtrip(self):
        """Test roundtrip."""
        mapping = _bytes_to_unicode()
        inverse = {char: byte for byte, char in mapping.items()}
        for byte in range(256):
            self.assertEqual(inverse[mapping[byte]], byte)

    def test_deterministic(self):
        """Test deterministic."""
        self.assertEqual(_bytes_to_unicode(), _bytes_to_unicode())


class TestDecoderConfigHasType(unittest.TestCase):
    """Test decoder config has type."""

    def test_direct(self):
        """Test direct."""
        self.assertTrue(_decoder_config_has_type({"type": "ByteFallback"}, "ByteFallback"))

    def test_nested(self):
        """Test nested."""
        config = {"decoders": [{"type": "Metaspace"}, {"type": "ByteLevel"}]}
        self.assertTrue(_decoder_config_has_type(config, "ByteLevel"))
        self.assertFalse(_decoder_config_has_type(config, "ByteFallback"))

    def test_deep_nesting(self):
        """Test deep nesting."""
        self.assertTrue(_decoder_config_has_type({"a": {"b": [{"type": "X"}]}}, "X"))

    def test_absent(self):
        """Test absent."""
        self.assertFalse(_decoder_config_has_type({"type": "Other"}, "ByteFallback"))
        self.assertFalse(_decoder_config_has_type({}, "ByteFallback"))
        self.assertFalse(_decoder_config_has_type([], "ByteFallback"))
        self.assertFalse(_decoder_config_has_type(None, "ByteFallback"))
        self.assertFalse(_decoder_config_has_type("ByteFallback", "ByteFallback"))
        self.assertFalse(_decoder_config_has_type(5, "ByteFallback"))


class TestOOVTokenBytes(unittest.TestCase):
    """Test oov token bytes."""

    def test_equals_empty_bytes(self):
        """Test equals empty bytes."""
        marker = _OOVTokenBytes()
        self.assertEqual(marker, b"")
        self.assertEqual(len(marker), 0)
        self.assertTrue(isinstance(marker, bytes))

    def test_type_distinct_from_bytes(self):
        """Test type distinct from bytes."""
        self.assertIsNot(type(_OOVTokenBytes()), bytes)
        self.assertIs(type(_OOVTokenBytes()), _OOVTokenBytes)


class FakeDecoder:
    """Stand-in for a tokenizers decoder exposing __getstate__."""

    def __init__(self, state=None, exc=None):
        self._state = state
        self._exc = exc

    def __getstate__(self):
        if self._exc is not None:
            raise self._exc
        return self._state


def make_fake_tokenizer(decoder, vocab, id_to_token):
    """Make fake tokenizer."""
    return SimpleNamespace(
        backend_tokenizer=SimpleNamespace(decoder=decoder),
        get_vocab=lambda: dict(vocab),
        convert_ids_to_tokens=lambda tid: id_to_token[tid],
    )


class TestHfByteTokenMapper(unittest.TestCase):
    """Test hf byte token mapper."""

    def test_no_backend_tokenizer(self):
        """Test no backend tokenizer."""
        self.assertIsNone(_hf_byte_token_mapper(SimpleNamespace()))

    def test_none_decoder(self):
        """Test none decoder."""
        tok = SimpleNamespace(backend_tokenizer=SimpleNamespace(decoder=None))
        self.assertIsNone(_hf_byte_token_mapper(tok))

    def test_byte_level_direct(self):
        """Test byte level direct."""
        decoder = type("ByteLevel", (), {})()
        tok = make_fake_tokenizer(decoder, {"a": 0, "b": 1}, {0: "a", 1: "b"})
        mapper = _hf_byte_token_mapper(tok)
        self.assertIsNotNone(mapper)
        self.assertEqual(mapper(0), b"a")
        self.assertEqual(mapper(1), b"b")

    def test_oov_maps_to_marker(self):
        """Test oov maps to marker."""
        decoder = type("ByteLevel", (), {})()
        tok = make_fake_tokenizer(decoder, {"a": 0}, {0: "a"})
        mapper = _hf_byte_token_mapper(tok)
        marker = mapper(5)
        self.assertIs(type(marker), _OOVTokenBytes)
        self.assertEqual(marker, b"")

    def test_map_rejects_bad_ids(self):
        """Test map rejects bad ids."""
        decoder = type("ByteLevel", (), {})()
        tok = make_fake_tokenizer(decoder, {"a": 0}, {0: "a"})
        mapper = _hf_byte_token_mapper(tok)
        for bad in (-1, "0", None, True):
            with self.assertRaises(ValueError, msg=repr(bad)):
                mapper(bad)

    def test_unmappable_char_returns_none(self):
        """Test unmappable char returns none."""
        # "€" is not in the GPT-2 byte alphabet.
        decoder = type("ByteLevel", (), {})()
        tok = make_fake_tokenizer(decoder, {"a": 0, "€": 1}, {0: "a", 1: "€"})
        mapper = _hf_byte_token_mapper(tok)
        self.assertEqual(mapper(0), b"a")
        self.assertIsNone(mapper(1))

    def test_byte_fallback_explicit_tokens(self):
        """Test byte fallback explicit tokens."""
        state = json.dumps({"decoders": [{"type": "ByteFallback"}]})
        tok = make_fake_tokenizer(
            FakeDecoder(state=state),
            {"<0xAB>": 0, "<0xab>": 1, "hello": 2, "<0xZZ>": 3, "<0xA>": 4},
            {0: "<0xAB>", 1: "<0xab>", 2: "hello", 3: "<0xZZ>", 4: "<0xA>"},
        )
        mapper = _hf_byte_token_mapper(tok)
        self.assertIsNotNone(mapper)
        self.assertEqual(mapper(0), b"\xab")
        self.assertEqual(mapper(1), b"\xab")
        self.assertIsNone(mapper(2))
        self.assertIsNone(mapper(3))
        self.assertIsNone(mapper(4))

    def test_byte_fallback_preferred_over_byte_level(self):
        """Test byte fallback preferred over byte level."""
        state = json.dumps({"decoders": [{"type": "ByteFallback"}, {"type": "ByteLevel"}]})
        tok = make_fake_tokenizer(
            FakeDecoder(state=state), {"<0xAB>": 0, "a": 1}, {0: "<0xAB>", 1: "a"}
        )
        mapper = _hf_byte_token_mapper(tok)
        self.assertEqual(mapper(0), b"\xab")
        # In byte_fallback mode plain tokens do not map directly.
        self.assertIsNone(mapper(1))

    def test_byte_level_via_config(self):
        """Test byte level via config."""
        state = json.dumps({"type": "ByteLevel"})
        tok = make_fake_tokenizer(FakeDecoder(state=state), {"a": 0}, {0: "a"})
        mapper = _hf_byte_token_mapper(tok)
        self.assertIsNotNone(mapper)
        self.assertEqual(mapper(0), b"a")

    def test_unrecognized_decoder_returns_none(self):
        """Test unrecognized decoder returns none."""
        state = json.dumps({"type": "Metaspace"})
        tok = make_fake_tokenizer(FakeDecoder(state=state), {"a": 0}, {0: "a"})
        self.assertIsNone(_hf_byte_token_mapper(tok))

    def test_broken_state_degrades_to_none(self):
        """Test broken state degrades to none."""
        tok = make_fake_tokenizer(FakeDecoder(exc=RuntimeError("boom")), {"a": 0}, {0: "a"})
        self.assertIsNone(_hf_byte_token_mapper(tok))

    def test_invalid_json_state_degrades_to_none(self):
        """Test invalid json state degrades to none."""
        tok = make_fake_tokenizer(FakeDecoder(state=b"not-json{"), {"a": 0}, {0: "a"})
        self.assertIsNone(_hf_byte_token_mapper(tok))

    def test_vocab_problems_degrade_to_none(self):
        """Test vocab problems degrade to none."""
        decoder = type("ByteLevel", (), {})()
        bad_vocab = SimpleNamespace(
            backend_tokenizer=SimpleNamespace(decoder=decoder),
            get_vocab=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        self.assertIsNone(_hf_byte_token_mapper(bad_vocab))
        empty_vocab = SimpleNamespace(
            backend_tokenizer=SimpleNamespace(decoder=decoder),
            get_vocab=lambda: {},
            convert_ids_to_tokens=lambda tid: "a",
        )
        self.assertIsNone(_hf_byte_token_mapper(empty_vocab))
        non_dict_vocab = SimpleNamespace(
            backend_tokenizer=SimpleNamespace(decoder=decoder),
            get_vocab=lambda: [("a", 0)],
            convert_ids_to_tokens=lambda tid: "a",
        )
        self.assertIsNone(_hf_byte_token_mapper(non_dict_vocab))

    def test_non_str_token_returns_none(self):
        """Test non str token returns none."""
        decoder = type("ByteLevel", (), {})()
        tok = make_fake_tokenizer(decoder, {"a": 0}, {0: 5})
        mapper = _hf_byte_token_mapper(tok)
        self.assertIsNone(mapper(0))


# ---------------------------------------------------------------------------
# HF overhang match: targeted pins + oracle fuzz
# ---------------------------------------------------------------------------


class TestHfOverhangMatch(unittest.TestCase):
    """Test hf overhang match."""

    def test_docstring_overhang_example(self):
        """Test docstring overhang example."""
        self.assertTrue(_hf_overhang_match(["las", "topper"], "stop"))

    def test_stop_inside_final_token(self):
        """Test stop inside final token."""
        self.assertTrue(_hf_overhang_match([" world"], "or"))

    def test_suffix_across_boundary(self):
        """Test suffix across boundary."""
        self.assertTrue(_hf_overhang_match(["ab", "c"], "abc"))

    def test_no_match(self):
        """Test no match."""
        self.assertFalse(_hf_overhang_match(["x"], "abc"))
        self.assertFalse(_hf_overhang_match(["ab", "x"], "abc"))

    def test_empty_inputs(self):
        """Test empty inputs."""
        self.assertFalse(_hf_overhang_match([], "a"))
        self.assertFalse(_hf_overhang_match(["a"], ""))

    def test_type_mismatch(self):
        """Test type mismatch."""
        self.assertFalse(_hf_overhang_match(["a"], b"a"))
        self.assertFalse(_hf_overhang_match([b"a"], "a"))

    def test_bytes_path(self):
        """Test bytes path."""
        self.assertTrue(_hf_overhang_match([b"las", b"topper"], b"stop"))
        self.assertFalse(_hf_overhang_match([b"las", b"x"], b"stop"))

    def test_oov_kills_walk_but_empty_passes_through(self):
        """Test oov kills walk but empty passes through."""
        # An OOV marker in the middle kills the walk (upstream's dummy-row
        # clamp); a genuinely empty representation is transparent to it --
        # though both still consume the trailing-window budget.
        self.assertFalse(_hf_overhang_match([b"a", _OOVTokenBytes(), b"bc"], b"abc"))
        self.assertTrue(_hf_overhang_match([b"a", b"", b"bc"], b"abc"))
        # The window budget still applies: with target=2 the leading "b" is
        # unreachable past the empty representation.
        self.assertFalse(_hf_overhang_match([b"b", b"", b"c"], b"bc"))

    def test_trailing_empty_token_kills_match(self):
        """Test trailing empty token kills match."""
        # No end-overlap can start from an empty final representation.
        self.assertFalse(_hf_overhang_match(["ab", ""], "ab"))

    def test_full_stop_inside_truncated_last_token(self):
        """Test full stop inside truncated last token."""
        self.assertTrue(_hf_overhang_match(["xabcd"], "abc"))

    def test_oracle_fuzz_str(self):
        """Test oracle fuzz str."""
        rng = fuzz_rng("overhang-str")
        for _ in range(400):
            texts = random_texts(rng, STR_TOKENS)
            stop = rng.choice(STR_STOPS + [""])
            self.assertEqual(
                _hf_overhang_match(texts, stop),
                oracle_hf_overhang_match(texts, stop),
                msg=f"texts={texts!r} stop={stop!r}",
            )

    def test_oracle_fuzz_bytes_with_oov(self):
        """Test oracle fuzz bytes with oov."""
        rng = fuzz_rng("overhang-bytes")
        byte_tokens = [t.encode("utf-8") for t in STR_TOKENS]
        byte_stops = [s.encode("utf-8") for s in STR_STOPS] + [b""]
        for _ in range(400):
            texts = random_texts(rng, byte_tokens, oov=True)
            stop = rng.choice(byte_stops)
            self.assertEqual(
                _hf_overhang_match(texts, stop),
                oracle_hf_overhang_match(texts, stop),
                msg=f"texts={texts!r} stop={stop!r}",
            )

    def test_oracle_fuzz_mixed_types(self):
        """Test oracle fuzz mixed types."""
        rng = fuzz_rng("overhang-mixed")
        for _ in range(200):
            texts = [rng.choice([t, t.encode("utf-8")]) for t in random_texts(rng, STR_TOKENS)]
            stop = rng.choice(STR_STOPS + [s.encode("utf-8") for s in STR_STOPS])
            self.assertEqual(
                _hf_overhang_match(texts, stop),
                oracle_hf_overhang_match(texts, stop),
                msg=f"texts={texts!r} stop={stop!r}",
            )

    def test_first_firing_step_matches_oracle(self):
        """Test first firing step matches oracle."""
        rng = fuzz_rng("first-step")
        for _ in range(200):
            prompt = random_texts(rng, STR_TOKENS)
            output = random_texts(rng, STR_TOKENS)
            texts = prompt + output
            stop = rng.choice(STR_STOPS)
            self.assertEqual(
                _first_firing_step(texts, stop, len(prompt), len(output)),
                oracle_first_firing_step(texts, stop, len(prompt), len(output)),
                msg=f"prompt={prompt!r} output={output!r} stop={stop!r}",
            )

    def test_first_firing_step_prompt_context(self):
        """Test first firing step prompt context."""
        # The prompt supplies overhang context: "ab" + "c" fires "abc" at
        # output step 1.
        self.assertEqual(_first_firing_step(["ab", "c"], "abc", 1, 1), 1)
        self.assertIsNone(_first_firing_step(["ab", "x"], "abc", 1, 1))


# ---------------------------------------------------------------------------
# Stop-semantic ports: targeted pins + oracle fuzz
# ---------------------------------------------------------------------------


def _ids_for(backend, text):
    """Ids for."""
    return backend.encode(text)


class TestEmulateHfStopString(unittest.TestCase):
    """Test emulate hf stop string."""

    def setUp(self):
        """Set up the test fixture."""
        self.backend = ByteBackend()

    def test_invalid_inputs(self):
        """Test invalid inputs."""
        good = (1, 2)
        bad_prompt = [1, 2]
        bad_output = "ab"
        for prompt_ids, output_ids, stop_text, token_bytes in [
            (bad_prompt, good, "x", None),
            (good, bad_output, "x", None),
            (good, good, "", None),
            (good, good, 5, None),
            (good, good, b"x", None),
            (good, good, "x", 5),
        ]:
            outcome = emulate_hf_stop_string(
                self.backend, prompt_ids, output_ids, stop_text, token_bytes=token_bytes
            )
            self.assertEqual(
                (outcome.available, outcome.fired, outcome.detail),
                (False, False, "invalid_emulation_input"),
                msg=repr((prompt_ids, output_ids, stop_text)),
            )
            self.assertEqual(outcome.truncated_ids, tuple(output_ids))

    def test_no_fire(self):
        """Test no fire."""
        output = _ids_for(self.backend, "hello")
        outcome = emulate_hf_stop_string(self.backend, (), output, "zzz")
        self.assertEqual(
            (outcome.available, outcome.fired, outcome.truncated_ids, outcome.detail),
            (True, False, output, "no_stop_within_cap"),
        )
        self.assertIsNone(outcome.returned_text)

    def test_fire_truncates_at_firing_token(self):
        """Test fire truncates at firing token."""
        output = _ids_for(self.backend, "say stop now")
        outcome = emulate_hf_stop_string(self.backend, (), output, "stop")
        self.assertTrue(outcome.available and outcome.fired)
        self.assertEqual(outcome.truncated_ids, _ids_for(self.backend, "say stop"))
        self.assertTrue(outcome.detail.startswith("fired_at_step_"))
        self.assertIsNone(outcome.returned_text)  # HF keeps the stop string

    def test_fire_step_number(self):
        """Test fire step number."""
        output = _ids_for(self.backend, "ab stop")
        outcome = emulate_hf_stop_string(self.backend, (), output, "stop")
        step = len(_ids_for(self.backend, "ab stop"))
        self.assertEqual(outcome.detail, f"fired_at_step_{step}")

    def test_empty_output(self):
        """Test empty output."""
        outcome = emulate_hf_stop_string(self.backend, (), (), "x")
        self.assertEqual((outcome.available, outcome.fired), (True, False))

    def test_decode_failure_degrades(self):
        """Test decode failure degrades."""
        outcome = emulate_hf_stop_string(ExplodingDecodeBackend("raise"), (), (1, 2), "x")
        self.assertEqual(outcome.detail, "token_text_unavailable")
        self.assertFalse(outcome.available)

    def test_non_str_decode_degrades(self):
        """Test non str decode degrades."""
        outcome = emulate_hf_stop_string(ExplodingDecodeBackend("non-str"), (), (1, 2), "x")
        self.assertEqual(outcome.detail, "token_text_unavailable")

    def test_byte_path_fire(self):
        """Test byte path fire."""
        token_bytes = lambda tid: bytes([tid & 0xFF])  # noqa: E731
        output = _ids_for(self.backend, "say stop")
        outcome = emulate_hf_stop_string(self.backend, (), output, "stop", token_bytes=token_bytes)
        self.assertTrue(outcome.fired)
        self.assertEqual(outcome.truncated_ids, output)

    def test_byte_path_fallback_to_decode(self):
        """Test byte path fallback to decode."""
        calls = []

        def token_bytes(tid):
            """Token bytes."""
            calls.append(tid)
            # Unmappable: implicit None return; the caller falls back to decode.

        output = _ids_for(self.backend, "say stop")
        outcome = emulate_hf_stop_string(self.backend, (), output, "stop", token_bytes=token_bytes)
        self.assertTrue(outcome.fired)
        self.assertTrue(calls)

    def test_byte_path_oov_marker_kills_walk(self):
        """Test byte path oov marker kills walk."""

        # stop=b"bc": [b"b", OOV, b"c"] must not fire; the OOV marker kills
        # the walk where a transparent empty would pass through.
        def token_bytes(tid):
            """Token bytes."""
            if tid >= 256:
                return _OOVTokenBytes()
            return bytes([tid])

        prompt = ()
        output = (ord("b"), 500, ord("c"))
        outcome = emulate_hf_stop_string(
            self.backend, prompt, output, "bc", token_bytes=token_bytes
        )
        self.assertFalse(outcome.fired)

    def test_byte_path_bad_mapping_degrades(self):
        """Test byte path bad mapping degrades."""
        outcome = emulate_hf_stop_string(
            self.backend,
            (),
            (1, 2),
            "x",
            token_bytes=lambda tid: "not-bytes",  # noqa: E731
        )
        self.assertEqual(outcome.detail, "token_text_unavailable")

    def test_byte_path_raising_mapper_degrades(self):
        """Test byte path raising mapper degrades."""

        def token_bytes(tid):
            """Token bytes."""
            raise RuntimeError("mapper exploded")

        outcome = emulate_hf_stop_string(self.backend, (), (1, 2), "x", token_bytes=token_bytes)
        self.assertEqual(outcome.detail, "token_text_unavailable")

    def test_oracle_fuzz_string_path(self):
        """Test oracle fuzz string path."""
        rng = fuzz_rng("hf-str")
        for _ in range(250):
            prompt = tuple(rng.randint(0, 120) for _ in range(rng.randint(0, 4)))
            output = tuple(rng.randint(0, 120) for _ in range(rng.randint(0, 10)))
            stop = rng.choice(STR_STOPS)
            self.assertEqual(
                emulate_hf_stop_string(self.backend, prompt, output, stop),
                oracle_emulate_hf(self.backend, prompt, output, stop),
                msg=f"prompt={prompt!r} output={output!r} stop={stop!r}",
            )

    def test_oracle_fuzz_byte_path(self):
        """Test oracle fuzz byte path."""
        rng = fuzz_rng("hf-bytes")
        for _ in range(250):
            prompt = tuple(rng.randint(0, 300) for _ in range(rng.randint(0, 4)))
            output = tuple(rng.randint(0, 300) for _ in range(rng.randint(0, 10)))
            stop = rng.choice(STR_STOPS)

            # Deterministic per (seed, token id): the implementation and
            # the oracle must see the identical mapping.
            seed = rng.randint(0, 2**32)

            def tb(tid, _seed=seed):
                """Tb."""
                if tid >= 256:
                    return _OOVTokenBytes()
                draw = int.from_bytes(sha256(f"{_seed}\0{tid}".encode()).digest()[:1], "big")
                if draw < 51:  # ~0.2: unmappable, fall back to decode
                    return None
                return bytes([tid & 0xFF])

            self.assertEqual(
                emulate_hf_stop_string(self.backend, prompt, output, stop, token_bytes=tb),
                oracle_emulate_hf(self.backend, prompt, output, stop, token_bytes=tb),
                msg=f"prompt={prompt!r} output={output!r} stop={stop!r}",
            )


class TestEmulateVllmHoldback(unittest.TestCase):
    """Test emulate vllm holdback."""

    def setUp(self):
        """Set up the test fixture."""
        self.backend = ByteBackend()

    def test_invalid_inputs(self):
        """Test invalid inputs."""
        for output_ids, stop_text, kwargs in [
            ("ab", "x", {}),
            ((1,), "", {}),
            ((1,), 5, {}),
            ((1,), "x", {"include_stop_str_in_output": "yes"}),
            ((1,), "x", {"min_tokens": -1}),
            ((1,), "x", {"min_tokens": True}),
            ((1,), "x", {"min_tokens": 1.5}),
        ]:
            outcome = emulate_vllm_holdback(self.backend, output_ids, stop_text, **kwargs)
            self.assertEqual(
                (outcome.available, outcome.fired, outcome.detail),
                (False, False, "invalid_emulation_input"),
                msg=repr((output_ids, stop_text, kwargs)),
            )

    def test_fire_strips_stop_by_default(self):
        """Test fire strips stop by default."""
        output = _ids_for(self.backend, "hello or")
        outcome = emulate_vllm_holdback(self.backend, output, "or")
        self.assertTrue(outcome.fired)
        self.assertEqual(outcome.returned_text, "hello ")
        self.assertEqual(outcome.truncated_ids, output)

    def test_include_stop_str_variant(self):
        """Test include stop str variant."""
        output = _ids_for(self.backend, "hello or")
        outcome = emulate_vllm_holdback(self.backend, output, "or", include_stop_str_in_output=True)
        self.assertTrue(outcome.fired)
        self.assertEqual(outcome.returned_text, "hello or")

    def test_fire_mid_text_truncates_ids(self):
        """Test fire mid text truncates ids."""
        output = _ids_for(self.backend, "aXb")
        outcome = emulate_vllm_holdback(self.backend, output, "X")
        self.assertTrue(outcome.fired)
        self.assertEqual(outcome.truncated_ids, _ids_for(self.backend, "aX"))
        self.assertEqual(outcome.returned_text, "a")

    def test_bisect_exact_offset_keeps_firing_prefix(self):
        """Test bisect exact offset keeps firing prefix."""
        # Stop "ab" completes at step 2 with stop_end=2 exactly equal to the
        # step-2 prefix offset: bisect_left(offsets, 2, lo=1) over [0,1,2,3]
        # keeps exactly the firing prefix (bisect_right would keep 3).
        output = _ids_for(self.backend, "abc")
        outcome = emulate_vllm_holdback(self.backend, output, "ab")
        self.assertTrue(outcome.fired)
        self.assertEqual(outcome.detail, "fired_at_step_2")
        self.assertEqual(outcome.truncated_ids, _ids_for(self.backend, "ab"))
        self.assertEqual(outcome.returned_text, "")

    def test_min_tokens_gates_all_stops(self):
        """Test min tokens gates all stops."""
        backend = ScriptedDecodeBackend({0: "a", 1: "X", 2: "b"})
        outcome = emulate_vllm_holdback(backend, (0, 1, 2), "X", min_tokens=1)
        self.assertTrue(outcome.fired)
        self.assertEqual(outcome.detail, "fired_at_step_2")

    def test_gate_completed_stop_not_refound(self):
        """Test gate completed stop not refound."""
        # Stop completes at step 1 (gated); the tail-only find window must
        # not refind it at step 2.
        backend = ScriptedDecodeBackend({0: "X", 1: "b"})
        outcome = emulate_vllm_holdback(backend, (0, 1), "X", min_tokens=1)
        self.assertEqual((outcome.available, outcome.fired), (True, False))
        self.assertEqual(outcome.detail, "no_stop_within_cap")

    def test_empty_text_step_skipped(self):
        """Test empty text step skipped."""
        # Without the new_char_count guard, the step-2 window start would be
        # 0 for the single-char stop and refind the gated stop.
        backend = ScriptedDecodeBackend({0: "X", 1: ""})
        outcome = emulate_vllm_holdback(backend, (0, 1), "X", min_tokens=1)
        self.assertFalse(outcome.fired)
        self.assertEqual(outcome.detail, "no_stop_within_cap")

    def test_fully_gated_output(self):
        """Test fully gated output."""
        output = _ids_for(self.backend, "hello")
        outcome = emulate_vllm_holdback(self.backend, output, "zzz", min_tokens=99)
        self.assertEqual((outcome.available, outcome.fired), (True, False))
        self.assertEqual(outcome.detail, "gated_by_min_tokens")

    def test_no_fire(self):
        """Test no fire."""
        output = _ids_for(self.backend, "hello")
        outcome = emulate_vllm_holdback(self.backend, output, "zzz")
        self.assertEqual(outcome.detail, "no_stop_within_cap")
        self.assertIsNone(outcome.returned_text)

    def test_decode_failure_degrades(self):
        """Test decode failure degrades."""
        outcome = emulate_vllm_holdback(ExplodingDecodeBackend("raise"), (1, 2), "x")
        self.assertEqual(outcome.detail, "decode_unavailable")
        outcome = emulate_vllm_holdback(ExplodingDecodeBackend("non-str"), (1, 2), "x")
        self.assertEqual(outcome.detail, "decode_unavailable")

    def test_oracle_fuzz(self):
        """Test oracle fuzz."""
        rng = fuzz_rng("vllm")
        for _ in range(300):
            output = tuple(rng.randint(0, 120) for _ in range(rng.randint(0, 10)))
            stop = rng.choice(STR_STOPS)
            include = rng.random() < 0.5
            min_tokens = rng.choice([0, 0, 1, 2, 5])
            self.assertEqual(
                emulate_vllm_holdback(
                    self.backend,
                    output,
                    stop,
                    include_stop_str_in_output=include,
                    min_tokens=min_tokens,
                ),
                oracle_emulate_vllm(
                    self.backend,
                    output,
                    stop,
                    include_stop_str_in_output=include,
                    min_tokens=min_tokens,
                ),
                msg=f"output={output!r} stop={stop!r} include={include} min={min_tokens}",
            )

    def test_oracle_fuzz_empty_decodes(self):
        """Test oracle fuzz empty decodes."""
        rng = fuzz_rng("vllm-empty")
        mapping = {i: rng.choice(["a", "X", ""]) for i in range(8)}
        backend = ScriptedDecodeBackend(mapping)
        for _ in range(200):
            output = tuple(rng.randint(0, 7) for _ in range(rng.randint(0, 8)))
            stop = rng.choice(["X", "a", "aX"])
            min_tokens = rng.choice([0, 1, 2])
            self.assertEqual(
                emulate_vllm_holdback(backend, output, stop, min_tokens=min_tokens),
                oracle_emulate_vllm(backend, output, stop, min_tokens=min_tokens),
                msg=f"output={output!r} stop={stop!r} min={min_tokens}",
            )


class TestEmulateSglangStop(unittest.TestCase):
    """Test emulate sglang stop."""

    def setUp(self):
        """Set up the test fixture."""
        self.backend = ByteBackend()

    def test_invalid_inputs(self):
        """Test invalid inputs."""
        for output_ids, stop_text, kwargs in [
            ("ab", "x", {}),
            ((1,), "", {}),
            ((1,), 5, {}),
            ((1,), "x", {"no_stop_trim": "yes"}),
        ]:
            outcome = emulate_sglang_stop(self.backend, output_ids, stop_text, **kwargs)
            self.assertEqual(
                (outcome.available, outcome.fired, outcome.detail),
                (False, False, "invalid_emulation_input"),
                msg=repr((output_ids, stop_text, kwargs)),
            )

    def test_fire_trims_at_leftmost(self):
        """Test fire trims at leftmost."""
        output = _ids_for(self.backend, "aXbX")
        outcome = emulate_sglang_stop(self.backend, output, "X")
        self.assertTrue(outcome.fired)
        self.assertEqual(outcome.detail, "fired_at_step_2")
        self.assertEqual(outcome.returned_text, "a")
        self.assertEqual(outcome.truncated_ids, _ids_for(self.backend, "aX"))

    def test_no_stop_trim_keeps_through_stop_end(self):
        """Test no stop trim keeps through stop end."""
        output = _ids_for(self.backend, "aXb")
        outcome = emulate_sglang_stop(self.backend, output, "X", no_stop_trim=True)
        self.assertEqual(outcome.returned_text, "aX")

    def test_no_min_tokens_gating(self):
        """Test no min tokens gating."""
        # Unlike vLLM, SGLang has no min_tokens equivalent: a stop at step
        # 1 fires.
        output = _ids_for(self.backend, "Xb")
        outcome = emulate_sglang_stop(self.backend, output, "X")
        self.assertTrue(outcome.fired)
        self.assertEqual(outcome.detail, "fired_at_step_1")

    def test_no_fire(self):
        """Test no fire."""
        output = _ids_for(self.backend, "hello")
        outcome = emulate_sglang_stop(self.backend, output, "zzz")
        self.assertEqual((outcome.available, outcome.fired), (True, False))
        self.assertEqual(outcome.detail, "no_stop_within_cap")
        self.assertIsNone(outcome.returned_text)

    def test_decode_failure_degrades(self):
        """Test decode failure degrades."""
        outcome = emulate_sglang_stop(ExplodingDecodeBackend("raise"), (1,), "x")
        self.assertEqual(outcome.detail, "decode_unavailable")
        outcome = emulate_sglang_stop(ExplodingDecodeBackend("non-str"), (1,), "x")
        self.assertEqual(outcome.detail, "decode_unavailable")

    def test_oracle_fuzz(self):
        """Test oracle fuzz."""
        rng = fuzz_rng("sglang")
        for _ in range(300):
            output = tuple(rng.randint(0, 120) for _ in range(rng.randint(0, 10)))
            stop = rng.choice(STR_STOPS)
            no_trim = rng.random() < 0.5
            self.assertEqual(
                emulate_sglang_stop(self.backend, output, stop, no_stop_trim=no_trim),
                oracle_emulate_sglang(self.backend, output, stop, no_stop_trim=no_trim),
                msg=f"output={output!r} stop={stop!r} no_trim={no_trim}",
            )


class TestEmulateEosStop(unittest.TestCase):
    """Test emulate eos stop."""

    def test_fire_truncates_before_eos(self):
        """Test fire truncates before eos."""
        outcome = emulate_eos_stop((1, 5, 2), 5)
        self.assertEqual(
            (outcome.available, outcome.fired, outcome.truncated_ids, outcome.detail),
            (True, True, (1,), "fired_at_step_2"),
        )
        self.assertIsNone(outcome.returned_text)

    def test_first_eos_wins(self):
        """Test first eos wins."""
        outcome = emulate_eos_stop((5, 1, 5), 5)
        self.assertEqual(outcome.truncated_ids, ())
        self.assertEqual(outcome.detail, "fired_at_step_1")

    def test_no_eos(self):
        """Test no eos."""
        outcome = emulate_eos_stop((1, 2, 3), 9)
        self.assertEqual((outcome.available, outcome.fired), (True, False))
        self.assertEqual(outcome.detail, "no_stop_within_cap")
        self.assertEqual(outcome.truncated_ids, (1, 2, 3))

    def test_empty_ids(self):
        """Test empty ids."""
        outcome = emulate_eos_stop((), 5)
        self.assertEqual(outcome.detail, "no_stop_within_cap")

    def test_eos_unavailable(self):
        """Test eos unavailable."""
        outcome = emulate_eos_stop((1, 2), None)
        self.assertEqual(
            (outcome.available, outcome.fired, outcome.detail),
            (False, False, "eos_id_unavailable"),
        )

    def test_invalid_eos_id(self):
        """Test invalid eos id."""
        for bad in (-1, "5", 2.0, True, (5,)):
            outcome = emulate_eos_stop((1, 2), bad)
            self.assertEqual(
                (outcome.available, outcome.fired, outcome.detail),
                (False, False, "invalid_eos_id"),
                msg=repr(bad),
            )

    def test_oracle_fuzz(self):
        """Test oracle fuzz."""
        rng = fuzz_rng("eos")
        for _ in range(300):
            output = tuple(rng.randint(0, 9) for _ in range(rng.randint(0, 8)))
            eos = rng.choice([None, 0, 3, 9, -1, "x", 99])
            self.assertEqual(
                emulate_eos_stop(output, eos),
                oracle_emulate_eos(output, eos),
                msg=f"output={output!r} eos={eos!r}",
            )


# ---------------------------------------------------------------------------
# Red team: hostile inputs, doc/code divergences, cross-port divergences
# ---------------------------------------------------------------------------


class TestRedTeam(unittest.TestCase):
    """Test red team."""

    def setUp(self):
        """Set up the test fixture."""
        self.backend = ByteBackend()

    def test_directory_empty_path_object_resolves_cwd(self):
        """Test directory empty path object resolves cwd."""
        # D1: str(Path("")) == "." is truthy, so the empty-path guard is
        # bypassed and the cwd is accepted as a model directory.
        self.assertEqual(_directory(Path("")), Path.cwd().resolve())

    def test_emulations_noniterable_ids_raise_typeerror(self):
        """Test emulations noniterable ids raise typeerror."""
        # D2: the invalid-input funnel calls tuple(output_ids) before
        # returning invalid_emulation_input.
        with self.assertRaises(TypeError):
            emulate_eos_stop(5, None)
        with self.assertRaises(TypeError):
            emulate_eos_stop(None, 1)
        with self.assertRaises(TypeError):
            emulate_hf_stop_string(self.backend, (), 5, "x")
        with self.assertRaises(TypeError):
            emulate_vllm_holdback(self.backend, 5, "x")
        with self.assertRaises(TypeError):
            emulate_sglang_stop(self.backend, None, "x")

    def test_emulation_ids_not_vocab_bounded(self):
        """Test emulation ids not vocab bounded."""
        # _valid_token_ids (core) has no vocab bound: huge ids are valid
        # emulation inputs; the byte backend masks them.
        output = (10**18, 10**18 + 1)
        outcome = emulate_hf_stop_string(self.backend, (), output, "zzz")
        self.assertEqual(outcome.detail, "no_stop_within_cap")
        outcome = emulate_vllm_holdback(self.backend, output, "zzz")
        self.assertEqual(outcome.detail, "no_stop_within_cap")

    def test_bool_ids_rejected(self):
        """Test bool ids rejected."""
        outcome = emulate_hf_stop_string(self.backend, (True,), (1,), "x")
        self.assertEqual(outcome.detail, "invalid_emulation_input")
        outcome = emulate_eos_stop((True, 1), 1)
        # emulate_eos_stop does not validate ids at all: True == 1 fires.
        self.assertTrue(outcome.fired)
        self.assertEqual(outcome.truncated_ids, ())

    def test_all_oov_byte_path_never_fires(self):
        """Test all oov byte path never fires."""
        token_bytes = lambda tid: _OOVTokenBytes()  # noqa: E731
        outcome = emulate_hf_stop_string(
            self.backend, (), (1, 2, 3), "abc", token_bytes=token_bytes
        )
        self.assertEqual((outcome.available, outcome.fired), (True, False))

    def test_vllm_stop_exactly_at_gate_end_reports_gated(self):
        """Test vllm stop exactly at gate end reports gated."""
        backend = ScriptedDecodeBackend({0: "a", 1: "X"})
        outcome = emulate_vllm_holdback(backend, (0, 1), "X", min_tokens=2)
        self.assertEqual(outcome.detail, "gated_by_min_tokens")
        self.assertFalse(outcome.fired)

    def test_vllm_stop_longer_than_text(self):
        """Test vllm stop longer than text."""
        output = _ids_for(self.backend, "ab")
        outcome = emulate_vllm_holdback(self.backend, output, "abcdefgh")
        self.assertEqual(outcome.detail, "no_stop_within_cap")

    def test_sglang_multichar_stop_leftmost_trim(self):
        """Test sglang multichar stop leftmost trim."""
        output = _ids_for(self.backend, "xxabYYab")
        outcome = emulate_sglang_stop(self.backend, output, "ab")
        self.assertTrue(outcome.fired)
        self.assertEqual(outcome.returned_text, "xx")

    def test_hf_vs_vllm_returned_text_divergence(self):
        """Test hf vs vllm returned text divergence."""
        # Hard divergence pinned by both docstrings: HF always keeps the
        # stop string (returned_text None), vLLM strips it by default.
        output = _ids_for(self.backend, "aXb")
        hf = emulate_hf_stop_string(self.backend, (), output, "X")
        vllm = emulate_vllm_holdback(self.backend, output, "X")
        sglang = emulate_sglang_stop(self.backend, output, "X")
        self.assertTrue(hf.fired and vllm.fired and sglang.fired)
        self.assertIsNone(hf.returned_text)
        self.assertEqual(vllm.returned_text, "a")
        self.assertEqual(sglang.returned_text, "a")
        self.assertEqual(hf.truncated_ids, vllm.truncated_ids)
        self.assertEqual(vllm.truncated_ids, sglang.truncated_ids)

    def test_overhang_longer_than_texts(self):
        """Test overhang longer than texts."""
        self.assertFalse(_hf_overhang_match(["a", "b"], "abcdef"))

    def test_member_long_name_ok(self):
        """Test member long name ok."""
        root = Path(tempfile.mkdtemp(prefix="cycle35-"))
        self.addCleanup(shutil.rmtree, root, True)
        name = "a" * 200 + ".json"
        (root / name).write_bytes(b"{}")
        self.assertEqual(_member(root, name), root / name)

    def test_large_output_perf_sanity(self):
        """Test large output perf sanity."""
        # No-fire scans must stay usable on long outputs (no blowup).
        output = tuple([97] * 1200)
        hf = emulate_hf_stop_string(self.backend, (), output, "zzz")
        self.assertEqual(hf.detail, "no_stop_within_cap")
        vllm = emulate_vllm_holdback(self.backend, output, "zzz")
        self.assertEqual(vllm.detail, "no_stop_within_cap")
        sglang = emulate_sglang_stop(self.backend, output, "zzz")
        self.assertEqual(sglang.detail, "no_stop_within_cap")
        eos = emulate_eos_stop(output, 98)
        self.assertEqual(eos.detail, "no_stop_within_cap")

    def test_emulations_never_raise_on_iterable_hostiles(self):
        """Test emulations never raise on iterable hostiles."""
        hostiles = [
            (),
            (0,),
            (10**18,),
            tuple(range(50)),
        ]
        for output in hostiles:
            emulate_hf_stop_string(self.backend, (), output, "Q")
            emulate_vllm_holdback(self.backend, output, "Q")
            emulate_sglang_stop(self.backend, output, "Q")
            emulate_eos_stop(output, 3)

    def test_truncated_backend_fingerprint_uniqueness(self):
        """Test truncated backend fingerprint uniqueness."""
        inner = RecordingInner()
        first = TruncatedPromptBackend(inner, 1)
        second = TruncatedPromptBackend(inner, 2)
        self.assertNotEqual(first.fingerprint, second.fingerprint)
        self.assertIn("+prompt-trunc-1b", first.fingerprint)


if __name__ == "__main__":
    unittest.main()
