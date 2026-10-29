"""Cycle 207: wave-10 correction round (red-team B B10-1/B10-2/B10-3,
red-team A A-F1).

Each test cites the finding it pins and fails on the pre-fix code:
- B10-1 tests fail pre-fix because the nested composition constructs
  and drops prefill ids instead of raising.
- B10-2 tests fail pre-fix because a foreign/hand-made native is
  silently accepted (no digest exists to mismatch).
- B10-3 tests fail pre-fix because the payloads carry no
  backend_fingerprint key at all.
- A-F1 tests fail pre-fix because verify_results raises TypeError
  instead of returning a failed verdict.
"""

import unittest
from threading import Event

from model_protocol_lab.backends import TokenTruncatedPromptBackend, TruncatedPromptBackend
from model_protocol_lab.core import Fixture, Format, Generation, Message, Split
from model_protocol_lab.experiments import (
    prompt_sensitivity_sweep,
    prompt_token_depth_sweep,
    run_experiment,
)
from model_protocol_lab.prefill import PrefillBackend
from model_protocol_lab.verify import verify_results

_BOUNDARY = 0xE000
_CAP = 8


class FakeBase:
    """Char-codepoint backend; the base prompt encoding."""

    fingerprint = "cycle207-fake-base"

    @staticmethod
    def encode(text: str) -> tuple[int, ...]:
        return tuple(ord(char) for char in text)

    @staticmethod
    def decode(token_ids: tuple[int, ...]) -> str:
        return "".join(chr(token) for token in token_ids)

    @staticmethod
    def generate(
        input_ids: tuple[int, ...], *, max_new_tokens: int, seed: int, cancel: Event
    ) -> Generation:
        return Generation(token_ids=(ord("X"),), reason="limit")


class ScriptedBackend:
    """Char-token backend: generate looks the decoded prompt up in a script."""

    def __init__(self, script: dict[str, tuple[str, str]]) -> None:
        self._script = dict(script)
        self.fingerprint = "cycle207-scripted-backend"

    @staticmethod
    def encode(text: str) -> tuple[int, ...]:
        return tuple(ord(char) for char in text)

    @staticmethod
    def decode(token_ids: tuple[int, ...]) -> str:
        return "".join(chr(token) for token in token_ids)

    def generate(
        self, input_ids: tuple[int, ...], *, max_new_tokens: int, seed: int, cancel: Event
    ) -> Generation:
        prompt = "".join(chr(token) for token in input_ids)
        answer, marker = self._script.get(prompt, ("WRONG", ""))
        text = answer + marker
        ids = tuple(ord(char) for char in text)
        if len(ids) > max_new_tokens:
            raise ValueError("scripted output exceeds the token cap")
        padding = (ord("~"),) * (max_new_tokens - len(ids))
        return Generation(token_ids=ids + padding, reason="limit")


_FMT_A = Format(name="cand-a", bos="", user_prefix="u:", assistant_prefix="a:", turn_end="</s>")
_FMT_B = Format(name="cand-b", bos="", user_prefix="U>", assistant_prefix="A>", turn_end="<e>")
_SEARCH = (
    Fixture(
        id="s0",
        messages=(Message(role="user", content="hi"),),
        expected="yes",
        group="g1",
    ),
)
_HELD = (
    Fixture(
        id="h0",
        messages=(Message(role="user", content="hey"),),
        expected="yes",
        group="g2",
    ),
)
_SPLIT = Split(search=_SEARCH, held_out=_HELD)


def _script() -> dict[str, tuple[str, str]]:
    script: dict[str, tuple[str, str]] = {}
    for fmt in (_FMT_A, _FMT_B):
        full = fmt.render(_SEARCH[0].messages)
        held_full = fmt.render(_HELD[0].messages)
        script[full] = ("yes", chr(_BOUNDARY)) if fmt.name == "cand-a" else ("WRONG", "")
        script[held_full] = ("yes", chr(_BOUNDARY)) if fmt.name == "cand-a" else ("WRONG", "")
    return script


def _native(backend, **overrides) -> dict:
    kwargs = {
        "boundary": (_BOUNDARY,),
        "max_new_tokens": _CAP,
        "seed": 0,
    }
    kwargs.update(overrides)
    return run_experiment(backend, (_FMT_A, _FMT_B), _SPLIT, **kwargs)


class NestedPrefillRefusalTests(unittest.TestCase):
    """B10-1: the prefill refusal walks the whole wrapper chain."""

    def test_nested_prefill_behind_byte_wrapper_is_refused(self) -> None:
        # Pre-fix this constructs; encode then drops prefill ids first.
        with self.assertRaises(ValueError) as caught:
            TokenTruncatedPromptBackend(
                TruncatedPromptBackend(PrefillBackend(FakeBase(), (999, 998), "p1"), 1),
                2,
            )
        self.assertIn("not a prefill-wrapped encoding", str(caught.exception))

    def test_direct_prefill_inner_still_refused(self) -> None:
        with self.assertRaises(ValueError):
            TokenTruncatedPromptBackend(PrefillBackend(FakeBase(), (999,), "p1"), 1)

    def test_deeply_nested_prefill_is_refused(self) -> None:
        wrapped = TruncatedPromptBackend(
            TruncatedPromptBackend(PrefillBackend(FakeBase(), (7,), "p1"), 1), 1
        )
        with self.assertRaises(ValueError):
            TokenTruncatedPromptBackend(wrapped, 1)

    def test_chain_without_prefill_still_constructs(self) -> None:
        # The walk must not over-refuse: a byte wrapper over the base
        # backend is a legitimate inner for the token wrapper.
        # Byte drop removes "O", then token drop removes "L".
        wrapped = TokenTruncatedPromptBackend(TruncatedPromptBackend(FakeBase(), 1), 1)
        self.assertEqual(wrapped.encode("HELLO"), (72, 69, 76))

    def test_cyclic_inner_chain_terminates(self) -> None:
        # A pathological _inner loop must terminate construction
        # instead of hanging it; with no prefill anywhere it constructs.
        class Loopy:
            fingerprint = "cycle207-loopy"

            def __init__(self) -> None:
                self._inner = self

            @staticmethod
            def encode(text: str) -> tuple[int, ...]:
                return tuple(ord(char) for char in text)

        wrapped = TokenTruncatedPromptBackend(Loopy(), 1)
        self.assertEqual(wrapped.encode("HI"), (72,))


class ConfigurationDigestTests(unittest.TestCase):
    """B10-2: sweep baselines are bound to their configuration."""

    def test_run_experiment_stamps_a_stable_digest(self) -> None:
        backend = ScriptedBackend(_script())
        first = _native(backend)
        second = _native(backend)
        digest = first["configuration_digest"]
        self.assertIsInstance(digest, str)
        self.assertTrue(digest.startswith("sha256:"))
        self.assertEqual(digest, second["configuration_digest"])

    def test_digest_changes_with_configuration(self) -> None:
        backend = ScriptedBackend(_script())
        base = _native(backend)["configuration_digest"]
        self.assertNotEqual(base, _native(backend, seed=1)["configuration_digest"])
        changed_boundary = _native(backend, boundary=(_BOUNDARY + 1,))["configuration_digest"]
        self.assertNotEqual(base, changed_boundary)
        self.assertNotEqual(base, _native(backend, max_new_tokens=4)["configuration_digest"])

    def test_handmade_native_rejected_by_both_sweeps(self) -> None:
        # Pre-fix shape-only validation accepted this dict.
        backend = ScriptedBackend(_script())
        foreign = {"selected_candidate": "cand-b", "decision": "abstain", "unrelated": 12345}
        for sweep in (
            lambda native: prompt_token_depth_sweep(
                backend,
                (_FMT_A, _FMT_B),
                _SPLIT,
                boundary=(_BOUNDARY,),
                max_drop_tokens=1,
                max_new_tokens=_CAP,
                seed=0,
                native=native,
            ),
            lambda native: prompt_sensitivity_sweep(
                backend,
                (_FMT_A, _FMT_B),
                _SPLIT,
                boundary=(_BOUNDARY,),
                max_drop_bytes=1,
                max_new_tokens=_CAP,
                seed=0,
                native=native,
            ),
        ):
            with self.assertRaises(ValueError) as caught:
                sweep(foreign)
            self.assertIn("missing its configuration_digest", str(caught.exception))

    def test_stale_native_rejected_by_both_sweeps(self) -> None:
        # A genuine result from another configuration (different seed)
        # must not compare against this sweep's runs.
        backend = ScriptedBackend(_script())
        stale = _native(backend, seed=1)
        for sweep in (
            lambda native: prompt_token_depth_sweep(
                backend,
                (_FMT_A, _FMT_B),
                _SPLIT,
                boundary=(_BOUNDARY,),
                max_drop_tokens=1,
                max_new_tokens=_CAP,
                seed=0,
                native=native,
            ),
            lambda native: prompt_sensitivity_sweep(
                backend,
                (_FMT_A, _FMT_B),
                _SPLIT,
                boundary=(_BOUNDARY,),
                max_drop_bytes=1,
                max_new_tokens=_CAP,
                seed=0,
                native=native,
            ),
        ):
            with self.assertRaises(ValueError) as caught:
                sweep(stale)
            self.assertIn("does not match this sweep configuration", str(caught.exception))

    def test_same_configuration_native_honoured_by_both_sweeps(self) -> None:
        # The CLI's shared-native path: one genuine native, computed
        # for exactly these parameters, is honoured by both sweeps.
        backend = ScriptedBackend(_script())
        native = _native(backend)
        token_result = prompt_token_depth_sweep(
            backend,
            (_FMT_A, _FMT_B),
            _SPLIT,
            boundary=(_BOUNDARY,),
            max_drop_tokens=1,
            max_new_tokens=_CAP,
            seed=0,
            native=native,
        )
        byte_result = prompt_sensitivity_sweep(
            backend,
            (_FMT_A, _FMT_B),
            _SPLIT,
            boundary=(_BOUNDARY,),
            max_drop_bytes=1,
            max_new_tokens=_CAP,
            seed=0,
            native=native,
        )
        self.assertEqual(token_result["native_winner"], native["selected_candidate"])
        self.assertEqual(byte_result["native_winner"], native["selected_candidate"])
        self.assertEqual(token_result["native_decision"], native["decision"])
        self.assertEqual(byte_result["native_decision"], native["decision"])


class SweepFingerprintTests(unittest.TestCase):
    """B10-3: both sweep payloads carry the backend fingerprint."""

    def test_token_sweep_payload_carries_backend_fingerprint(self) -> None:
        backend = ScriptedBackend(_script())
        result = prompt_token_depth_sweep(
            backend,
            (_FMT_A, _FMT_B),
            _SPLIT,
            boundary=(_BOUNDARY,),
            max_drop_tokens=1,
            max_new_tokens=_CAP,
            seed=0,
        )
        self.assertEqual(result["backend_fingerprint"], backend.fingerprint)

    def test_byte_sweep_payload_carries_backend_fingerprint(self) -> None:
        backend = ScriptedBackend(_script())
        result = prompt_sensitivity_sweep(
            backend,
            (_FMT_A, _FMT_B),
            _SPLIT,
            boundary=(_BOUNDARY,),
            max_drop_bytes=1,
            max_new_tokens=_CAP,
            seed=0,
        )
        self.assertEqual(result["backend_fingerprint"], backend.fingerprint)


class UnhashableDecisionTests(unittest.TestCase):
    """A-F1: verify_results records a failed schema check, never a crash."""

    def _schema_check(self, result: dict) -> dict:
        schema = [c for c in result["checks"] if c["id"] == "schema"]
        self.assertEqual(len(schema), 1)
        return schema[0]

    def test_dict_decision_returns_failed_verdict(self) -> None:
        result = verify_results({"decision": {"x": 1}})
        self.assertEqual(result["verdict"], "failed")
        check = self._schema_check(result)
        self.assertEqual(check["status"], "failed")
        self.assertIn("decision not in known vocabulary", check["detail"])

    def test_list_decision_returns_failed_verdict(self) -> None:
        result = verify_results({"decision": ["failed"]})
        self.assertEqual(result["verdict"], "failed")
        check = self._schema_check(result)
        self.assertEqual(check["status"], "failed")
        self.assertIn("decision not in known vocabulary", check["detail"])

    def test_other_checks_are_unavailable_not_crashed(self) -> None:
        result = verify_results({"decision": {"x": 1}})
        others = [c for c in result["checks"] if c["id"] != "schema"]
        self.assertTrue(others)
        self.assertTrue(all(c["status"] == "unavailable" for c in others))


if __name__ == "__main__":
    unittest.main()
