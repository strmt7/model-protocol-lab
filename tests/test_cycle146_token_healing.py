"""Cycle 146: token-healing selection sensitivity experiment (wave-2 idea 3).

``healing.py`` ports HF transformers' ``heal_tokens`` mechanism into a pure,
backend-agnostic core (injected decode/re-encode/argmax). ``backends`` gains
``LocalHFBackend.heal_input_ids`` and the ``TokenHealingBackend`` wrapper.
``experiments.healing_stage`` re-evaluates every search candidate with healed
prompts; the pure ``experiments.healing_delta`` re-applies the selection
rule and tests the lab's own boundary-canonicality audit as a predictor
(the falsification table). Wired into ``mpl run --token-healing``,
results.json, the HTML "Token-healing selection sensitivity" section, and
``mpl diff-reports`` (``diff_token_healing`` with ``healing_changed`` /
``healing_asymmetric`` summary flags feeding ``identical``).
"""

import copy
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from threading import Event

from model_protocol_lab import backends, core, experiments
from model_protocol_lab.core import Fixture, Format, Message
from model_protocol_lab.diffreports import (
    _render_healing_summary,
    _valid_token_healing,
    diff_token_healing,
)
from model_protocol_lab.experiments import (
    healing_delta,
    healing_stage,
    native_search_outcomes,
)
from model_protocol_lab.healing import (
    HEAL_EXTENSION_BIAS,
    HEAL_ORIGINAL_BIAS,
    heal_prompt_ids,
    healing_biases,
    normalize_tail_text,
    tail_extensions,
)
from model_protocol_lab.reporting import _token_healing_section

CANDIDATES = ("alpha", "beta")


def _fixture(fid, content, expected="BLUE"):
    return Fixture(
        id=fid,
        messages=(Message(role="user", content=content),),
        expected=expected,
        group=f"g-{fid}",
    )


def _format(name):
    return Format(
        name=name,
        bos="",
        user_prefix=f"[{name}]",
        assistant_prefix="A:",
        turn_end="\n",
    )


class ScriptedHealBackend:
    """Deterministic backend with scriptable healing behavior.

    ``heal_mode``: "ok" (passthrough), "raise" (healing fails -> the
    wrapper must fall back to native ids), "rewrite" (returns a fixed
    marker prompt, exercising the healed path distinctly).
    ``behavior`` maps candidate name to "pass" / "fail" / "error".
    """

    fingerprint = "cycle146-healing-scripted"

    def __init__(self, behavior, heal_mode="ok"):
        self._behavior = dict(behavior)
        self._heal_mode = heal_mode

    @staticmethod
    def encode(text):
        return tuple(map(ord, text))

    @staticmethod
    def decode(ids):
        return "".join(map(chr, ids))

    def heal_input_ids(self, input_ids):
        if self._heal_mode == "raise":
            raise RuntimeError("heal boom")
        if self._heal_mode == "rewrite":
            return (ord("H"),)
        return input_ids

    def generate(self, input_ids, *, max_new_tokens, seed, cancel):
        prompt = self.decode(input_ids)
        candidate = next((n for n in CANDIDATES if f"[{n}]" in prompt), None)
        if candidate is None:
            return core.Generation((), "error", "unknown")
        mode = self._behavior.get(candidate, "pass")
        if mode == "error":
            return core.Generation((), "error", "boom")
        text = "BLUE" if mode == "pass" else "WRONG"
        ids = self.encode(text + chr(7))
        return core.Generation(ids + (1,) * (max_new_tokens - len(ids)), "limit")


def _stage(behavior, heal_mode="ok"):
    cases = (_fixture("S1", "Say BLUE please"), _fixture("S2", "Say BLUE now"))
    formats = tuple(_format(n) for n in CANDIDATES)
    return experiments.healing_stage(
        ScriptedHealBackend(behavior, heal_mode=heal_mode),
        formats,
        cases,
        boundary=(7,),
        max_new_tokens=32,
        seed=3,
    )


def _native(behavior):
    # Build native outcomes by hand (mirrors native_search_outcomes).
    return {
        name: {
            "passed": mode == "pass",
            "per_fixture": {"S1": mode == "pass", "S2": mode == "pass"},
        }
        for name, mode in behavior.items()
    }


def _healed_payload(behavior):
    stage = _stage(behavior)
    return stage


def _audit(fragile=()):
    return {
        "available": True,
        "per_candidate": {
            name: {
                "n_problem": 3 if name in fragile else 0,
                "n_usable": 5,
            }
            for name in CANDIDATES
        },
    }


def _delta(behavior_native, behavior_healed, fragile=(), primary=("selected", "alpha")):
    decision, winner = primary
    return healing_delta(
        winner,
        decision,
        CANDIDATES,
        _native(behavior_native),
        _healed_payload(behavior_healed),
        _audit(fragile),
    )


class HealingPureTests(unittest.TestCase):
    def test_constants_match_hf(self):
        self.assertEqual(HEAL_EXTENSION_BIAS, 10.0)
        self.assertEqual(HEAL_ORIGINAL_BIAS, 1.0)

    def test_normalize_tail_text(self):
        self.assertEqual(normalize_tail_text(" hello", "Ġ"), "Ġhello")
        self.assertEqual(normalize_tail_text("hello", "Ġ"), "hello")
        self.assertEqual(normalize_tail_text(" hello", None), " hello")
        with self.assertRaises(ValueError):
            normalize_tail_text(7, "Ġ")

    def test_tail_extensions_sorted_and_includes_tail(self):
        vocab = {"Par": 10, "Paris": 11, "Park": 12, "spar": 13}
        self.assertEqual(tail_extensions(vocab, "Par"), (10, 11, 12))
        self.assertEqual(tail_extensions(vocab, "zzz"), ())
        self.assertEqual(tail_extensions({}, "Par"), ())
        with self.assertRaises(ValueError):
            tail_extensions({"a": "not-an-int"}, "a")
        with self.assertRaises(ValueError):
            tail_extensions(vocab, 7)

    def test_healing_biases(self):
        biases = healing_biases((10, 11), 10)
        self.assertEqual(biases, {10: 11.0, 11: 10.0})
        # Tail id outside the extension set still gets its original bias.
        biases = healing_biases((11, 12), 10)
        self.assertEqual(biases[10], 1.0)
        with self.assertRaises(ValueError):
            healing_biases((), 10)

    def test_heal_prompt_ids_happy_path(self):
        vocab = {"Par": 10, "Paris": 11, "Park": 12}
        seen = {}

        def argmax(trimmed, biases):
            seen["trimmed"] = trimmed
            seen["biases"] = biases
            return 11

        out = heal_prompt_ids(
            (1, 2, 10),
            decode_prompt=lambda ids: "Say Par",
            decode_token=lambda t: "Par",
            reencode=lambda text: (1, 2, 10),
            vocab=vocab,
            space_token_text="Ġ",
            biased_argmax=argmax,
        )
        self.assertEqual(out, (1, 2, 11))
        self.assertEqual(seen["trimmed"], (1, 2))
        self.assertEqual(seen["biases"][11], 10.0)
        self.assertEqual(seen["biases"][10], 11.0)

    def test_heal_prompt_ids_no_extensions_returns_canonical(self):
        # Only the tail token itself extends the tail text: HF skips the
        # biased step but still returns the re-canonicalized prompt.
        out = heal_prompt_ids(
            (5, 6),
            decode_prompt=lambda ids: "ab",
            decode_token=lambda t: "b",
            reencode=lambda text: (5, 6, 7),  # re-canonicalization adds an id
            vocab={"b": 6},
            space_token_text=None,
            biased_argmax=lambda trimmed, biases: 99,
        )
        self.assertEqual(out, (5, 6, 7))

    def test_heal_prompt_ids_single_token_is_failure(self):
        with self.assertRaisesRegex(ValueError, "heal_empty_trimmed_prompt"):
            heal_prompt_ids(
                (10,),
                decode_prompt=lambda ids: "Par",
                decode_token=lambda t: "Par",
                reencode=lambda text: (10,),
                vocab={"Par": 10, "Paris": 11},
                space_token_text=None,
                biased_argmax=lambda trimmed, biases: 11,
            )

    def test_heal_prompt_ids_invalid_input(self):
        kw = {
            "decode_prompt": lambda ids: "x",
            "decode_token": lambda t: "x",
            "reencode": lambda text: (1,),
            "vocab": {"x": 1},
            "space_token_text": None,
            "biased_argmax": lambda trimmed, biases: 1,
        }
        for bad in ((), "nope", (1, -2), (1, "x")):
            with self.assertRaises(ValueError, msg=f"input={bad!r}"):
                heal_prompt_ids(bad, **kw)

    def test_heal_prompt_ids_injected_failures_fail_closed(self):
        base = {
            "decode_token": lambda t: "Par",
            "vocab": {"Par": 10, "Paris": 11},
            "space_token_text": None,
            "biased_argmax": lambda trimmed, biases: 11,
        }

        def boom(ids):
            raise RuntimeError("decode exploded")

        with self.assertRaisesRegex(ValueError, "heal_decode_failed"):
            heal_prompt_ids(
                (1, 10),
                decode_prompt=boom,
                reencode=lambda text: (1, 10),
                **base,
            )
        with self.assertRaisesRegex(ValueError, "heal_decode_failed"):
            heal_prompt_ids(
                (1, 10),
                decode_prompt=lambda ids: 42,  # non-str decode
                reencode=lambda text: (1, 10),
                **base,
            )
        with self.assertRaisesRegex(ValueError, "heal_reencode_failed"):
            heal_prompt_ids(
                (1, 10),
                decode_prompt=lambda ids: "Say Par",
                reencode=lambda text: [1, 10],  # not a tuple
                **base,
            )
        with self.assertRaisesRegex(ValueError, "heal_argmax_failed"):
            heal_prompt_ids(
                (1, 10),
                decode_prompt=lambda ids: "Say Par",
                decode_token=lambda t: "Par",
                reencode=lambda text: (1, 10),
                vocab={"Par": 10, "Paris": 11},
                space_token_text=None,
                biased_argmax=lambda trimmed, biases: -3,
            )

    def test_heal_prompt_ids_deterministic(self):
        kw = {
            "decode_prompt": lambda ids: "Say Par",
            "decode_token": lambda t: "Par",
            "reencode": lambda text: (1, 2, 10),
            "vocab": {"Par": 10, "Paris": 11},
            "space_token_text": None,
            "biased_argmax": lambda trimmed, biases: 10,
        }
        self.assertEqual(heal_prompt_ids((1, 2, 10), **kw), heal_prompt_ids((1, 2, 10), **kw))


class TokenHealingBackendTests(unittest.TestCase):
    def test_rejects_backend_without_heal(self):
        class Plain:
            fingerprint = "plain"

        with self.assertRaisesRegex(ValueError, "does not support token healing"):
            backends.TokenHealingBackend(Plain())

    def test_rejects_non_string_fingerprint(self):
        inner = ScriptedHealBackend({"alpha": "pass"})
        inner.fingerprint = 42
        with self.assertRaisesRegex(ValueError, "fingerprint must be a string"):
            backends.TokenHealingBackend(inner)

    def test_fingerprint_suffix_and_forwarding(self):
        inner = ScriptedHealBackend({"alpha": "pass"})
        inner.hf_stop_token_bytes = "sentinel"
        wrapper = backends.TokenHealingBackend(inner)
        self.assertEqual(wrapper.fingerprint, "cycle146-healing-scripted+token-healing")
        self.assertEqual(wrapper.hf_stop_token_bytes, "sentinel")
        self.assertEqual(wrapper.encode("ab"), inner.encode("ab"))

    def test_generate_heals_and_counts(self):
        inner = ScriptedHealBackend({"alpha": "pass"}, heal_mode="ok")
        wrapper = backends.TokenHealingBackend(inner)
        gen = wrapper.generate(
            tuple(map(ord, "[alpha] hi")), max_new_tokens=8, seed=0, cancel=Event()
        )
        self.assertEqual(gen.reason, "limit")
        self.assertEqual(wrapper.n_healed, 1)
        self.assertEqual(wrapper.n_fallback, 0)

    def test_generate_falls_back_on_contract_violation(self):
        # A healer returning a non-tuple degrades to native ids instead
        # of feeding malformed ids into generation.
        inner = ScriptedHealBackend({"alpha": "pass"}, heal_mode="ok")
        seen = {}

        def bad_heal(input_ids):
            return [1, 2, 3]

        def spy_generate(input_ids, *, max_new_tokens, seed, cancel):
            seen["input_ids"] = input_ids
            return core.Generation((1,), "limit")

        inner.heal_input_ids = bad_heal
        inner.generate = spy_generate
        wrapper = backends.TokenHealingBackend(inner)
        native = tuple(map(ord, "[alpha]"))
        wrapper.generate(native, max_new_tokens=8, seed=0, cancel=Event())
        self.assertEqual(seen["input_ids"], native)
        self.assertEqual(wrapper.n_fallback, 1)
        self.assertEqual(wrapper.n_healed, 0)

    def test_generate_falls_back_on_heal_failure(self):
        inner = ScriptedHealBackend({"alpha": "pass"}, heal_mode="raise")
        wrapper = backends.TokenHealingBackend(inner)
        seen = {}

        def spy_generate(input_ids, *, max_new_tokens, seed, cancel):
            seen["input_ids"] = input_ids
            return core.Generation((1, 2), "limit")

        inner.generate = spy_generate
        wrapper.generate((65, 66), max_new_tokens=8, seed=0, cancel=Event())
        # Native ids used, fallback counted.
        self.assertEqual(seen["input_ids"], (65, 66))
        self.assertEqual(wrapper.n_healed, 0)
        self.assertEqual(wrapper.n_fallback, 1)


class HealInputIdsAdapterTests(unittest.TestCase):
    """Wire-level test of LocalHFBackend.heal_input_ids with fakes.

    No model is downloaded in this repo, so the adapter's forward-pass
    wiring (tokenizer calls, bias application, argmax) is verified
    against a fake tokenizer and a fake torch/model pair. The
    algorithm itself is covered by HealingPureTests.
    """

    def _backend(self, logits):
        import contextlib
        from threading import RLock

        vocab = {"a": 0, "ab": 1, "abc": 2}

        class FakeTokenizer:
            @staticmethod
            def get_vocab():
                return dict(vocab)

            @staticmethod
            def convert_tokens_to_ids(text):
                return vocab.get(text)

            @staticmethod
            def convert_ids_to_tokens(ids):
                inv = {v: k for k, v in vocab.items()}
                return [inv[i] for i in ids]

            @staticmethod
            def decode(ids, **kwargs):
                inv = {v: k for k, v in vocab.items()}
                return "".join(inv[i] for i in ids)

            @staticmethod
            def encode(text, **kwargs):
                table = {"aab": (0, 1), "a": (0,)}
                if text not in table:
                    raise ValueError("unknown text")
                return table[text]

        class FakeLogits:
            def __init__(self, values):
                self._v = list(values)

            @property
            def ndim(self):
                return 1

            def numel(self):
                return len(self._v)

            def __setitem__(self, i, v):
                self._v[i] = v

            def __getitem__(self, i):
                return self._v[i]

        class FakeLogits3D:
            def __init__(self, values):
                self._values = values

            def __getitem__(self, key):
                assert key == (0, -1, slice(None, None, None))
                return FakeLogits(self._values)

        class FakeResult:
            def __init__(self, values):
                self.logits = FakeLogits3D(values)

        class FakeModel:
            def __call__(self, **kwargs):
                return FakeResult(list(logits))

        class FakeAll:
            def all(self):
                return self

            @staticmethod
            def item():
                return True

        class FakeScalar:
            def __init__(self, value):
                self._value = value

            def item(self):
                return self._value

        class FakeTorch:
            long = "long"

            @staticmethod
            def inference_mode():
                return contextlib.nullcontext()

            @staticmethod
            def tensor(data, dtype=None, device=None):
                return data

            @staticmethod
            def ones_like(seq):
                return seq

            @staticmethod
            def isfinite(logits):
                return FakeAll()

            @staticmethod
            def argmax(logits):
                best = max(range(logits.numel()), key=lambda i: logits[i])
                return FakeScalar(best)

        backend = backends.LocalHFBackend.__new__(backends.LocalHFBackend)
        backend._vocab_size = 3
        backend._context_limit = 100
        backend._device = "cpu"
        backend._torch = FakeTorch()
        backend._lock = RLock()
        backend._tokenizer = FakeTokenizer()
        backend._model = FakeModel()
        return backend

    def test_heal_input_ids_rewires_tail(self):
        # Prompt (0, 1) decodes to "aab"; tail "ab" has extensions
        # {ab, abc}; logits [0, 1, 5] + biases {1: 11, 2: 10} ->
        # argmax 2 ("abc"). Healed prompt: (0, 2).
        backend = self._backend([0.0, 1.0, 5.0])
        self.assertEqual(backend.heal_input_ids((0, 1)), (0, 2))

    def test_heal_input_ids_prefers_original_on_tie(self):
        # Logits tie between "ab" and "abc": the +1.0 original-token
        # bias breaks the tie toward the original tail.
        backend = self._backend([0.0, 5.0, 5.0])
        self.assertEqual(backend.heal_input_ids((0, 1)), (0, 1))

    def test_heal_input_ids_rejects_bad_input(self):
        backend = self._backend([0.0, 1.0, 5.0])
        with self.assertRaises(ValueError):
            backend.heal_input_ids(())
        with self.assertRaises(ValueError):
            backend.heal_input_ids((0, 99))


class HealingStageTests(unittest.TestCase):
    def test_outcomes_and_shape(self):
        out = _stage({"alpha": "pass", "beta": "fail"})
        self.assertEqual(out["n_fixtures"], 2)
        self.assertEqual(sorted(out["outcomes"]), ["alpha", "beta"])
        alpha = out["outcomes"]["alpha"]
        self.assertTrue(alpha["passed"])
        self.assertEqual(alpha["per_fixture"], {"S1": True, "S2": True})
        self.assertEqual(alpha["n_passed"], 2)
        self.assertEqual(alpha["n_total"], 2)
        self.assertFalse(out["outcomes"]["beta"]["passed"])
        # Every prompt healed (2 candidates x 2 fixtures).
        self.assertEqual(out["n_healed"], 4)
        self.assertEqual(out["n_fallback"], 0)
        self.assertEqual(out["fixture_ids"], ["S1", "S2"])

    def test_heal_failure_falls_back_and_still_evaluates(self):
        out = _stage({"alpha": "pass", "beta": "pass"}, heal_mode="raise")
        self.assertTrue(out["outcomes"]["alpha"]["passed"])
        self.assertEqual(out["n_healed"], 0)
        self.assertEqual(out["n_fallback"], 4)

    def test_backend_error_records_none(self):
        out = _stage({"alpha": "error", "beta": "pass"})
        self.assertIsNone(out["outcomes"]["alpha"]["passed"])
        self.assertTrue(out["outcomes"]["beta"]["passed"])

    def test_deterministic(self):
        self.assertEqual(
            _stage({"alpha": "pass", "beta": "fail"}),
            _stage({"alpha": "pass", "beta": "fail"}),
        )

    def test_validation(self):
        cases = (_fixture("S1", "Say BLUE please"),)
        formats = (_format("alpha"),)
        with self.assertRaises(ValueError):
            healing_stage(None, formats, cases, boundary=(7,))
        with self.assertRaises(ValueError):
            healing_stage(ScriptedHealBackend({"alpha": "pass"}), (), cases, boundary=(7,))
        with self.assertRaises(ValueError):
            healing_stage(
                ScriptedHealBackend({"alpha": "pass"}),
                formats,
                cases,
                boundary=(),
            )

        # A backend that cannot heal fails the stage loudly (the CLI
        # converts this into the unavailable marker, never a crash).
        class Plain:
            fingerprint = "plain"

        with self.assertRaises(ValueError):
            healing_stage(Plain(), formats, cases, boundary=(7,))


class NativeSearchOutcomesTests(unittest.TestCase):
    @staticmethod
    def _entries():
        def obs(passed):
            return {
                "fixture": "S1",
                "exact_match": passed,
                "boundary_found": passed,
            }

        return [
            {
                "format": {"name": "alpha"},
                "summary": {"aggregate_eligible": True},
                "observations": [obs(True), obs(True)],
            },
            {
                "format": {"name": "beta"},
                "summary": {"aggregate_eligible": True},
                "observations": [obs(True), obs(False)],
            },
        ]

    def test_extracts_pass_fail(self):
        out = native_search_outcomes(self._entries(), CANDIDATES)
        self.assertTrue(out["alpha"]["passed"])
        self.assertEqual(out["alpha"]["per_fixture"], {"S1": True})
        self.assertFalse(out["beta"]["passed"])

    def test_missing_candidate_is_none(self):
        out = native_search_outcomes(self._entries(), ("alpha", "gamma"))
        self.assertIsNone(out["gamma"]["passed"])
        self.assertEqual(out["gamma"]["per_fixture"], {})

    def test_ineligible_and_malformed_degrade(self):
        entries = self._entries()
        entries[0]["summary"]["aggregate_eligible"] = False
        entries[1]["observations"] = [{"fixture": "S1"}]  # missing verdict fields
        out = native_search_outcomes(entries, CANDIDATES)
        self.assertIsNone(out["alpha"]["passed"])
        self.assertIsNone(out["beta"]["passed"])
        # Non-list entries never raise.
        out = native_search_outcomes("junk", CANDIDATES)
        self.assertIsNone(out["alpha"]["passed"])


class HealingDeltaTests(unittest.TestCase):
    def test_stable_winner(self):
        d = _delta({"alpha": "pass", "beta": "fail"}, {"alpha": "pass", "beta": "fail"})
        self.assertTrue(d["available"])
        self.assertEqual(d["healed_decision"], "selected")
        self.assertEqual(d["healed_winner"], "alpha")
        self.assertFalse(d["flipped"])
        self.assertEqual(d["per_candidate"]["alpha"]["verdict_changed"], False)

    def test_flipped_winner(self):
        d = _delta({"alpha": "pass", "beta": "fail"}, {"alpha": "fail", "beta": "pass"})
        self.assertTrue(d["flipped"])
        self.assertEqual(d["healed_winner"], "beta")
        self.assertEqual(len(d["fixture_flips"]), 4)

    def test_healed_tie_is_ambiguous(self):
        d = _delta(
            {"alpha": "pass", "beta": "fail"},
            {"alpha": "pass", "beta": "pass"},
        )
        self.assertEqual(d["healed_decision"], "ambiguous")
        self.assertIsNone(d["healed_winner"])
        self.assertTrue(d["flipped"])

    def test_all_unevaluable_is_unavailable(self):
        d = _delta(
            {"alpha": "pass", "beta": "fail"},
            {"alpha": "error", "beta": "error"},
        )
        self.assertEqual(d["healed_decision"], "unavailable")
        self.assertIsNone(d["healed_winner"])

    def test_falsification_table(self):
        # beta is seam-fragile and flips; alpha is canonical and stable:
        # the prediction holds for both.
        d = _delta(
            {"alpha": "pass", "beta": "fail"},
            {"alpha": "pass", "beta": "pass"},
            fragile=("beta",),
        )
        fals = d["falsification"]
        self.assertEqual(
            fals["per_candidate"]["beta"],
            {"seam_fragile": True, "verdict_changed": True, "prediction_holds": True},
        )
        self.assertEqual(
            fals["per_candidate"]["alpha"],
            {"seam_fragile": False, "verdict_changed": False, "prediction_holds": True},
        )
        self.assertEqual((fals["n_comparable"], fals["n_holds"]), (2, 2))
        self.assertTrue(fals["prediction_holds"])

    def test_falsification_violation(self):
        # alpha is canonical but flips anyway: a genuine violation of the
        # prediction (the interesting, publishable outcome).
        d = _delta(
            {"alpha": "pass", "beta": "fail"},
            {"alpha": "fail", "beta": "fail"},
            fragile=("beta",),
        )
        fals = d["falsification"]
        self.assertFalse(fals["per_candidate"]["alpha"]["prediction_holds"])
        self.assertEqual(fals["n_violations"], 1)
        self.assertFalse(fals["prediction_holds"])

    def test_falsification_without_audit(self):
        d = healing_delta(
            "alpha",
            "selected",
            CANDIDATES,
            _native({"alpha": "pass", "beta": "fail"}),
            _healed_payload({"alpha": "pass", "beta": "fail"}),
            None,
        )
        fals = d["falsification"]
        self.assertIsNone(fals["per_candidate"]["alpha"]["seam_fragile"])
        self.assertIsNone(fals["per_candidate"]["alpha"]["prediction_holds"])
        self.assertEqual(fals["n_comparable"], 0)
        self.assertIsNone(fals["prediction_holds"])

    def test_malformed_input_never_raises(self):
        good_native = _native({"alpha": "pass", "beta": "fail"})
        good_healed = _healed_payload({"alpha": "pass", "beta": "fail"})
        bads = [
            # candidates not a tuple
            {
                "primary_winner": "alpha",
                "primary_decision": "selected",
                "candidates": ["alpha", "beta"],
                "native": good_native,
                "healed": good_healed,
                "audit": _audit(),
            },
            # unknown winner
            {
                "primary_winner": "zzz",
                "primary_decision": "selected",
                "candidates": CANDIDATES,
                "native": good_native,
                "healed": good_healed,
                "audit": _audit(),
            },
            # empty decision
            {
                "primary_winner": "alpha",
                "primary_decision": "",
                "candidates": CANDIDATES,
                "native": good_native,
                "healed": good_healed,
                "audit": _audit(),
            },
            # healed not a dict
            {
                "primary_winner": "alpha",
                "primary_decision": "selected",
                "candidates": CANDIDATES,
                "native": good_native,
                "healed": None,
                "audit": _audit(),
            },
            # healed outcomes not a dict
            {
                "primary_winner": "alpha",
                "primary_decision": "selected",
                "candidates": CANDIDATES,
                "native": good_native,
                "healed": {"outcomes": None},
                "audit": _audit(),
            },
            # non-bool passed value
            {
                "primary_winner": "alpha",
                "primary_decision": "selected",
                "candidates": CANDIDATES,
                "native": good_native,
                "healed": {"outcomes": {"alpha": {"passed": "yes"}, "beta": {"passed": True}}},
                "audit": _audit(),
            },
        ]
        for i, kw in enumerate(bads):
            with self.subTest(i=i):
                d = healing_delta(
                    kw["primary_winner"],
                    kw["primary_decision"],
                    kw["candidates"],
                    kw["native"],
                    kw["healed"],
                    kw["audit"],
                )
                self.assertFalse(d["available"])

    def test_oracle_differential(self):
        """Brute-force reimplementation vs healing_delta on random inputs."""
        import random

        rng = random.Random(0xC146)  # noqa: S311 -- deterministic test seed
        for i in range(300):
            cands = tuple(f"c{j}" for j in range(rng.randint(1, 4)))
            native = {}
            healed_outcomes = {}
            audit_cands = {}
            for c in cands:
                n = rng.choice([True, False, None])
                h = rng.choice([True, False, None])
                native[c] = {"passed": n, "per_fixture": {"f1": True} if n else {}}
                healed_outcomes[c] = {
                    "passed": h,
                    "per_fixture": {"f1": h is True},
                    "n_passed": 1 if h else 0,
                    "n_total": 1,
                }
                audit_cands[c] = {
                    "n_problem": rng.randint(0, 2),
                    "n_usable": rng.choice([0, 3]),
                }
            passed = [c for c in cands if native[c]["passed"] is True]
            primary_winner = rng.choice([None] + passed or [None])
            if primary_winner is None:
                primary_winner = rng.choice(list(cands) + [None])
            primary_decision = rng.choice(["selected", "compatible_on_fixture_set", "x"])
            healed = {"outcomes": healed_outcomes, "n_healed": 2, "n_fallback": 0}
            audit = {"available": True, "per_candidate": audit_cands}
            got = healing_delta(primary_winner, primary_decision, cands, native, healed, audit)
            want = _oracle_healing_delta(
                primary_winner, primary_decision, cands, native, healed, audit
            )
            with self.subTest(i=i):
                self.assertEqual(got, want)


def _oracle_healing_delta(primary_winner, primary_decision, candidates, native, healed, audit):
    """Independent reimplementation of healing_delta (different structure)."""
    if not (
        isinstance(candidates, tuple)
        and candidates
        and all(type(c) is str and c for c in candidates)
        and len(set(candidates)) == len(candidates)
    ):
        return {"available": False, "detail": "invalid_candidates"}
    if primary_winner is not None and (
        type(primary_winner) is not str or primary_winner not in candidates
    ):
        return {"available": False, "detail": "primary_winner_unknown"}
    if type(primary_decision) is not str or not primary_decision:
        return {"available": False, "detail": "primary_decision_invalid"}
    if type(native) is not dict or type(healed) is not dict:
        return {"available": False, "detail": "outcomes_invalid"}
    outcomes = healed.get("outcomes")
    if type(outcomes) is not dict:
        return {"available": False, "detail": "healed_outcomes_invalid"}
    hp = {}
    for c in candidates:
        e = outcomes.get(c)
        if type(e) is not dict:
            return {"available": False, "detail": f"healed_candidate_{c}_invalid"}
        p = e.get("passed")
        if p is not None and type(p) is not bool:
            return {"available": False, "detail": f"healed_candidate_{c}_invalid"}
        hp[c] = p
    if all(v is None for v in hp.values()):
        hd, hw = "unavailable", None
    else:
        w = [c for c in candidates if hp[c] is True]
        hd, hw = (
            ("selected", w[0])
            if len(w) == 1
            else ("ambiguous", None)
            if w
            else ("no_candidate", None)
        )
    flipped = (hd, hw) != (primary_decision, primary_winner)
    pc = {}
    for c in candidates:
        ne = native.get(c) if type(native.get(c)) is dict else {}
        n = ne.get("passed")
        n = n if type(n) is bool else None
        h = hp[c]
        pc[c] = {
            "native": n,
            "healed": h,
            "verdict_changed": (n != h) if type(n) is bool and type(h) is bool else None,
        }
    flips = sorted(
        [
            {"candidate": c, "fixture": f, "native": nv, "healed": hv}
            for c in candidates
            for f, nv in (
                native.get(c, {}).get("per_fixture", {}).items()
                if type(native.get(c, {}).get("per_fixture")) is dict
                else []
            )
            for hv in [
                healed["outcomes"][c].get("per_fixture", {}).get(f)
                if type(healed["outcomes"][c].get("per_fixture")) is dict
                else None
            ]
            if type(nv) is bool and type(hv) is bool and nv != hv
        ],
        key=lambda e: (e["candidate"], e["fixture"]),
    )
    audit_pc = audit.get("per_candidate") if type(audit) is dict else None
    fpc, comp, holds = {}, 0, 0
    for c in candidates:
        ae = audit_pc.get(c) if type(audit_pc) is dict else None
        sf = None
        if type(ae) is dict:
            np_, nu = ae.get("n_problem"), ae.get("n_usable")
            if type(np_) is int and type(nu) is int and nu > 0:
                sf = np_ > 0
        vc = pc[c]["verdict_changed"]
        ph = None
        if type(sf) is bool and type(vc) is bool:
            ph = (not vc) or sf
            comp += 1
            holds += bool(ph)
        fpc[c] = {"seam_fragile": sf, "verdict_changed": vc, "prediction_holds": ph}
    nh = healed.get("n_healed")
    nf = healed.get("n_fallback")
    return {
        "available": True,
        "primary_winner": primary_winner,
        "primary_decision": primary_decision,
        "healed_decision": hd,
        "healed_winner": hw,
        "flipped": flipped,
        "per_candidate": pc,
        "fixture_flips": flips,
        "falsification": {
            "per_candidate": fpc,
            "n_comparable": comp,
            "n_holds": holds,
            "n_violations": comp - holds,
            "prediction_holds": (holds == comp) if comp else None,
        },
        "n_healed": nh if type(nh) is int else None,
        "n_fallback": nf if type(nf) is int else None,
    }


class DiffTokenHealingTests(unittest.TestCase):
    @staticmethod
    def _payload(flipped=False, winner="alpha", extra=None):
        payload = {
            "available": True,
            "primary_winner": "alpha",
            "primary_decision": "selected",
            "healed_decision": "selected",
            "healed_winner": winner,
            "flipped": flipped,
            "per_candidate": {
                "alpha": {"native": True, "healed": True, "verdict_changed": False},
                "beta": {"native": False, "healed": False, "verdict_changed": False},
            },
            "fixture_flips": [],
            "falsification": {
                "per_candidate": {},
                "n_comparable": 2,
                "n_holds": 2,
                "n_violations": 0,
                "prediction_holds": True,
            },
            "n_healed": 4,
            "n_fallback": 0,
        }
        if extra:
            payload.update(extra)
        return payload

    def test_identical_payloads_unchanged(self):
        p = self._payload()
        diff = diff_token_healing(p, copy.deepcopy(p))
        self.assertEqual(diff["available"], "both")
        self.assertFalse(diff["changed"])

    def test_flip_only_change_pins_disjunct(self):
        old = self._payload(flipped=False)
        new = self._payload(flipped=True, winner="beta")
        diff = diff_token_healing(old, new)
        self.assertTrue(diff["changed"])
        self.assertTrue(diff["flip"]["changed"])

    def test_candidate_drift_only_change_pins_disjunct(self):
        old = self._payload()
        new = self._payload()
        new["per_candidate"]["beta"]["healed"] = True
        diff = diff_token_healing(old, new)
        self.assertTrue(diff["changed"])
        self.assertEqual(len(diff["candidate_drift"]), 1)
        self.assertEqual(diff["candidate_drift"][0]["candidate"], "beta")

    def test_falsification_only_change_pins_disjunct(self):
        old = self._payload()
        new = self._payload()
        new["falsification"]["n_holds"] = 1
        diff = diff_token_healing(old, new)
        self.assertTrue(diff["changed"])
        self.assertTrue(diff["falsification"]["changed"])

    def test_added_removed_candidates_pin_disjunct(self):
        old = self._payload()
        new = self._payload()
        new["per_candidate"]["gamma"] = {
            "native": True,
            "healed": False,
            "verdict_changed": True,
        }
        diff = diff_token_healing(old, new)
        self.assertTrue(diff["changed"])
        self.assertEqual(diff["candidates_added"], ["gamma"])
        old2 = self._payload()
        new2 = self._payload()
        del new2["per_candidate"]["beta"]
        diff2 = diff_token_healing(old2, new2)
        self.assertTrue(diff2["changed"])
        self.assertEqual(diff2["candidates_removed"], ["beta"])

    def test_availability_contract(self):
        p = self._payload()
        self.assertEqual(diff_token_healing(p, None)["available"], "old")
        self.assertEqual(diff_token_healing(None, p)["available"], "new")
        self.assertEqual(diff_token_healing(None, None)["available"], "neither")
        d = diff_token_healing(p, None)
        self.assertFalse(d["changed"])

    def test_hostile_values_degrade_not_crash(self):
        # Non-bool flipped / non-bool verdict values fail validation on
        # that side instead of crashing the comparison (wave-2 C3).
        p = self._payload()
        hostile = self._payload()
        hostile["flipped"] = "yes"
        self.assertEqual(diff_token_healing(p, hostile)["available"], "old")
        hostile2 = self._payload()
        hostile2["per_candidate"]["alpha"]["native"] = {"nested": "dict"}
        self.assertEqual(diff_token_healing(p, hostile2)["available"], "old")
        hostile3 = self._payload()
        hostile3["per_candidate"] = {7: {"native": True}}
        self.assertFalse(_valid_token_healing(hostile3))
        # ... and the summary renderer survives hostile values too.
        diff = diff_token_healing(p, p)
        lines = _render_healing_summary(diff)
        self.assertEqual(lines, [])
        evil = self._payload()
        evil["healed_winner"] = "\x1b[31mRED\x1b[0m"
        diff2 = diff_token_healing(self._payload(), evil)
        rendered = _render_healing_summary(diff2)
        self.assertTrue(rendered)
        self.assertFalse(any("\x1b" in line for line in rendered))

    def test_render_summary_lines(self):
        old = self._payload(flipped=False)
        new = self._payload(flipped=True, winner="beta")
        new["per_candidate"]["beta"]["healed"] = True
        new["falsification"]["n_holds"] = 1
        diff = diff_token_healing(old, new)
        lines = _render_healing_summary(diff)
        text = "\n".join(lines)
        self.assertIn("Token-healing flip: False -> True", text)
        self.assertIn("candidate drift (1)", text)
        self.assertIn("falsification accuracy", text)
        # One-sided comparison names the missing side.
        one = _render_healing_summary(diff_token_healing(old, None))
        self.assertTrue(any("new run did not record" in line for line in one))
        self.assertEqual(_render_healing_summary(None), [])
        self.assertEqual(_render_healing_summary({"available": "neither"}), [])


class TokenHealingSectionTests(unittest.TestCase):
    @staticmethod
    def _result():
        d = _delta(
            {"alpha": "pass", "beta": "fail"},
            {"alpha": "fail", "beta": "pass"},
            fragile=("beta",),
        )
        return {"token_healing": d}

    def test_renders_nothing_when_unavailable(self):
        self.assertEqual(_token_healing_section({}), "")
        self.assertEqual(_token_healing_section({"token_healing": {"available": False}}), "")
        self.assertEqual(_token_healing_section({"token_healing": {"available": True}}), "")

    def test_renders_verdict_and_tables(self):
        section_html = _token_healing_section(self._result())
        self.assertIn("Token-healing selection sensitivity", section_html)
        self.assertIn("flipped", section_html)
        self.assertIn("Seam-fragile (boundary audit)", section_html)
        self.assertIn("Boundary-audit prediction holds for", section_html)

    def test_escapes_hostile_candidate_names(self):
        import html as html_module

        result = self._result()
        evil = "<script>alert(1)</script>\x1b[31m"
        pc = result["token_healing"]["per_candidate"]
        pc[evil] = {"native": True, "healed": False, "verdict_changed": True}
        section_html = _token_healing_section(result)
        self.assertNotIn("<script>", section_html)
        self.assertIn(html_module.escape(evil), section_html)


class HealingMutationGateTests(unittest.TestCase):
    """Tempdir-module mutation gates (the real tree is never rewritten)."""

    def _run_gate(self, module_name, name, old_text, new_text, scenario):
        mod = __import__(f"model_protocol_lab.{module_name}", fromlist=["x"])
        source = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertEqual(source.count(old_text), 1, f"mutant {name}: anchor not unique")
        mutated = source.replace(old_text, new_text)
        with tempfile.TemporaryDirectory(prefix="c146_mut_", dir="/tmp") as tmpdir:
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

    def test_mutant_dropped_original_bias_caught(self):
        def scenario(mod):
            # Without the +1.0 original-token bias, the tail token loses
            # its slight healing advantage: the bias map is wrong.
            self.assertEqual(mod.healing_biases((10, 11), 10)[10], 11.0)

        self._run_gate(
            "healing",
            "dropped_original_bias",
            "    biases[tail_id] = biases.get(tail_id, 0.0) + HEAL_ORIGINAL_BIAS",
            "    biases[tail_id] = biases.get(tail_id, 0.0)  # MUTANT: bias dropped",
            scenario,
        )

    def test_mutant_endswith_prefix_search_caught(self):
        def scenario(mod):
            # A suffix search instead of a prefix search finds the wrong
            # extension set: "sPar" does not extend "Par".
            self.assertEqual(
                mod.tail_extensions({"Par": 10, "Paris": 11, "sPar": 12}, "Par"),
                (10, 11),
            )

        self._run_gate(
            "healing",
            "endswith_search",
            "        if text.startswith(prefix):",
            "        if text.endswith(prefix):  # MUTANT: suffix search",
            scenario,
        )

    def test_mutant_inverted_falsification_caught(self):
        def scenario(mod):
            # Weakening the prediction to require a flip for every fragile
            # candidate misclassifies the uninformative case (fragile but
            # stable): the scenario holds on the real module and must fail
            # on the mutant.
            d = mod.healing_delta(
                "alpha",
                "selected",
                ("alpha", "beta"),
                {
                    "alpha": {"passed": True, "per_fixture": {"f1": True}},
                    "beta": {"passed": False, "per_fixture": {"f1": False}},
                },
                {
                    "outcomes": {
                        "alpha": {"passed": True, "per_fixture": {"f1": True}},
                        "beta": {"passed": False, "per_fixture": {"f1": False}},
                    }
                },
                {
                    "per_candidate": {
                        "alpha": {"n_problem": 0, "n_usable": 2},
                        "beta": {"n_problem": 1, "n_usable": 2},
                    }
                },
            )
            self.assertTrue(d["falsification"]["prediction_holds"])

        self._run_gate(
            "experiments",
            "weakened_falsification",
            "            prediction_holds = (not verdict_changed) or seam_fragile",
            "            prediction_holds = verdict_changed and seam_fragile  # MUTANT",
            scenario,
        )

    def test_mutant_dropped_flip_disjunct_caught(self):
        def scenario(mod):
            old = {
                "available": True,
                "flipped": False,
                "healed_decision": "selected",
                "healed_winner": "alpha",
                "per_candidate": {
                    "alpha": {"native": True, "healed": True},
                },
                "falsification": {"n_holds": 1, "n_comparable": 1},
            }
            new = dict(old, flipped=True, healed_winner="beta")
            new = copy.deepcopy(new)
            self.assertTrue(mod.diff_token_healing(old, new)["changed"])

        self._run_gate(
            "diffreports",
            "dropped_flip_disjunct",
            '    result["changed"] = bool(\n'
            "        flip_changed\n"
            "        or candidate_drift\n"
            "        or candidates_added\n"
            "        or candidates_removed\n"
            "        or falsification_changed\n"
            "    )",
            '    result["changed"] = bool(\n'
            "        candidate_drift\n"
            "        or candidates_added\n"
            "        or candidates_removed\n"
            "        or falsification_changed\n"
            "    )  # MUTANT: flip disjunct dropped",
            scenario,
        )


if __name__ == "__main__":
    unittest.main()
