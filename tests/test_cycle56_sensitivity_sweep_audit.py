"""Cycle 56: adversarial audit of the prompt-sensitivity sweep.

Audited module (the sweep's perturbation machinery):
  * ``model_protocol_lab.core.truncate_text_tail`` -- THE perturbation operator.
  * ``model_protocol_lab.backends.TruncatedPromptBackend`` -- applies the
    operator to every encoded prompt.
  * ``model_protocol_lab.experiments.prompt_sensitivity_sweep`` -- re-runs the
    search per truncation depth and aggregates stability
    (``winner_changed`` / ``flips_at``).

SCOPE NOTE: the assignment brief mentions "whitespace/case/unicode
perturbations". The implementation has exactly ONE perturbation operator:
dropping ``drop_bytes`` (1..8) trailing bytes from the rendered prompt at
whole-UTF-8-character granularity. Unicode enters only through that
granularity rule (``None`` when a cut would split a codepoint). There are no
case/whitespace operators to audit; the sweep measures boundary sensitivity
to prompt-final tokenization, per the docstring.

Findings -- NO REAL BUGS FOUND:
  * Edge (documented here, NOT classified as a bug): ``truncate_text_tail``
    on text containing lone surrogates (e.g. ``"a\\ud800"``) raises an
    uncaught ``UnicodeEncodeError`` instead of ``ValueError``/``None``.
    Downstream this is still safe: ``TruncatedPromptBackend.encode``
    propagates it, ``evaluate`` collapses it to ``encode_failed``, and the
    sweep records the depth's failed decision -- exactly the documented
    fallback ("record the run's failed decision instead of silently
    skipping"). A lone surrogate is not valid UTF-8-encodable text, so it
    sits outside the operator's documented input contract.
  * Documented non-properties (pinned by tests, by design):
    - the operator is NOT idempotent: applying drop ``d`` twice drops ``2d``
      bytes (``t(t(x,d),d) == t(x,2*d)``);
    - depths do NOT compose: ``t(t(x,a),b)`` can be undefined (``None``)
      where ``t(x,a+b)`` is defined -- the sweep therefore applies every
      depth fresh to the native prompt, never chained (pinned end-to-end);
    - the semantic outcome (winner/decision) is NOT invariant under
      perturbation by design -- ``flips_at`` records exactly where it flips.
      What IS invariant: the kept text is always a strict whole-character
      prefix of the original (content invariance), and every depth is
      independent of the others.

Coverage: exact output pins on adversarial inputs (empty, 1-byte, combining
marks, emoji, already-perturbed, very long, length caps), documented
invariants, operator algebra (idempotency/associativity), a seeded fuzz
against an independent UTF-8 boundary oracle, the wrapper contract, and the
sweep aggregation truth table including the decision-differs/winner-same
branch that a weakened stability metric would miss.
"""

import random
import unittest
from threading import Event

from model_protocol_lab import core
from model_protocol_lab.backends import TruncatedPromptBackend
from model_protocol_lab.core import truncate_text_tail
from model_protocol_lab.experiments import prompt_sensitivity_sweep


def _naive_cut_ok(raw: bytes, drop: int) -> bool:
    """Independent oracle: does ``raw[:-drop]`` end on a UTF-8 char boundary?

    Walks continuation bytes backwards from the cut and checks the lead
    byte's expected length -- no ``bytes.decode`` involved, so agreement
    with ``truncate_text_tail`` cross-validates the implementation's own
    decode-based check. ``raw`` must be canonical UTF-8 (it always is here:
    it comes from ``str.encode("utf-8")``), so a prefix is valid iff it
    does not end mid-codepoint.
    """
    data = raw[: len(raw) - drop]
    index = len(data) - 1
    continuations = 0
    while index >= 0 and 0x80 <= data[index] <= 0xBF:
        continuations += 1
        index -= 1
    if continuations == 0:
        # No trailing continuation bytes: the final byte must itself be a
        # complete character (ASCII). A bare lead byte (0xC2..0xF4) means the
        # codepoint was cut mid-way.
        return data[-1] < 0x80
    if index < 0:
        return False
    lead = data[index]
    if 0xC2 <= lead <= 0xDF:
        need = 1
    elif 0xE0 <= lead <= 0xEF:
        need = 2
    elif 0xF0 <= lead <= 0xF4:
        need = 3
    else:
        return False
    return continuations == need


class TruncateExactPins(unittest.TestCase):
    """Pin the operator's EXACT output on adversarial inputs."""

    def test_ascii_pins(self) -> None:
        """Test ascii pins."""
        self.assertEqual(truncate_text_tail("hello", 1), "hell")
        self.assertEqual(truncate_text_tail("hello", 4), "h")
        self.assertEqual(truncate_text_tail("ab", 1), "a")

    def test_multibyte_boundary_pins(self) -> None:
        """Test multibyte boundary pins."""
        # "héllo": h(1) é(2) l(1) l(1) o(1) = 6 bytes.
        self.assertIsNone(truncate_text_tail("héllo", 4))  # splits é
        self.assertEqual(truncate_text_tail("héllo", 5), "h")  # removes éllo
        # "xhé": x(1) h(1) é(2) = 4 bytes.
        self.assertIsNone(truncate_text_tail("xhé", 1))  # splits é
        self.assertEqual(truncate_text_tail("xhé", 2), "xh")
        self.assertEqual(truncate_text_tail("xhé", 3), "x")

    def test_combining_mark_pins(self) -> None:
        """Test combining mark pins."""
        # "e\u0301" is e(1) + COMBINING ACUTE ACCENT(2) = 3 bytes.
        text = "é"
        self.assertEqual(len(text.encode("utf-8")), 3)
        self.assertIsNone(truncate_text_tail(text, 1))  # splits the mark
        self.assertEqual(truncate_text_tail(text, 2), "e")

    def test_emoji_pins(self) -> None:
        """Test emoji pins."""
        # "a😀b": a(1) 😀(4) b(1) = 6 bytes.
        self.assertEqual(truncate_text_tail("a😀b", 1), "a😀")
        self.assertIsNone(truncate_text_tail("a😀b", 2))  # splits 😀
        self.assertEqual(truncate_text_tail("a😀b", 5), "a")
        self.assertIsNone(truncate_text_tail("😀", 1))
        self.assertIsNone(truncate_text_tail("😀", 2))
        self.assertIsNone(truncate_text_tail("😀", 3))

    def test_length_cap_pins(self) -> None:
        """Test length cap pins."""
        # drop == byte length, or larger, is out of contract: ValueError.
        with self.assertRaises(ValueError):
            truncate_text_tail("hello", 5)
        with self.assertRaises(ValueError):
            truncate_text_tail("hello", 6)
        with self.assertRaises(ValueError):
            truncate_text_tail("hello", 100)
        with self.assertRaises(ValueError):
            truncate_text_tail("", 1)
        with self.assertRaises(ValueError):
            truncate_text_tail("a", 1)
        # One byte short of the cap is fine.
        self.assertEqual(truncate_text_tail("abc", 2), "a")

    def test_type_pins(self) -> None:
        """Test type pins."""
        for bad_text in (b"hi", None, 123, ["h"]):
            with self.assertRaises(ValueError, msg=f"text={bad_text!r}"):
                truncate_text_tail(bad_text, 1)  # type: ignore[arg-type]
        # bool is not int here (strict type check): True/False rejected.
        for bad_drop in (True, False, 0, -1, 1.0, None, "1", (1,)):
            with self.assertRaises(ValueError, msg=f"drop={bad_drop!r}"):
                truncate_text_tail("hello", bad_drop)  # type: ignore[arg-type]

    def test_already_perturbed_pin(self) -> None:
        """Test already perturbed pin."""
        # Perturbing an already-perturbed text just drops further bytes.
        once = truncate_text_tail("hello world", 3)
        self.assertEqual(once, "hello wo")
        twice = truncate_text_tail(once, 3)
        self.assertEqual(twice, "hello")

    def test_very_long_pins(self) -> None:
        """Test very long pins."""
        big = "a" * 100_000
        out = truncate_text_tail(big, 8)
        self.assertEqual(out, "a" * 99_992)
        accented = "é" * 10_000  # 20_000 bytes
        self.assertIsNone(truncate_text_tail(accented, 1))  # splits last é
        self.assertEqual(truncate_text_tail(accented, 2), "é" * 9_999)
        mixed = "ab" * 25_000 + "😀"  # 50_000 chars, 50_004 bytes
        # Drop the 4-byte emoji plus one more byte: removes "b😀".
        self.assertEqual(truncate_text_tail(mixed, 5), "ab" * 24_999 + "a")


class TruncateInvariants(unittest.TestCase):
    """Verify the operator's documented invariants and algebra."""

    def test_byte_length_invariant(self) -> None:
        """Test byte length invariant."""
        cases = ["hello", "héllo", "a😀b", "日本語", "éx", "a" * 50]
        for text in cases:
            raw = text.encode("utf-8")
            for drop in range(1, min(8, len(raw))):
                out = truncate_text_tail(text, drop)
                if out is not None:
                    self.assertEqual(
                        len(out.encode("utf-8")),
                        len(raw) - drop,
                        f"{text!r} drop={drop}",
                    )

    def test_strict_whole_character_prefix(self) -> None:
        """Test strict whole character prefix."""
        # Content invariance: the kept text is always a strict prefix of the
        # original, and the removed tail encodes to exactly drop_bytes.
        cases = ["hello world", "héllo wörld 😀", "日本語テスト", "a" * 30 + "é"]
        for text in cases:
            raw = text.encode("utf-8")
            for drop in range(1, min(8, len(raw))):
                out = truncate_text_tail(text, drop)
                if out is not None:
                    self.assertTrue(text.startswith(out), f"{text!r} drop={drop}")
                    tail = text[len(out) :]
                    self.assertEqual(tail.encode("utf-8"), raw[-drop:])

    def test_none_iff_codepoint_split(self) -> None:
        """Test none iff codepoint split."""
        # The None contract, cross-checked by the independent oracle on a
        # sweep over every byte position of multibyte strings.
        for text in ["héllo", "a😀b", "日本語", "é", "😀😀", "x́ŷz̃"]:
            raw = text.encode("utf-8")
            for drop in range(1, len(raw)):
                out = truncate_text_tail(text, drop)
                oracle_ok = _naive_cut_ok(raw, drop)
                if oracle_ok:
                    self.assertIsInstance(out, str, f"{text!r} drop={drop}")
                    self.assertEqual(out.encode("utf-8"), raw[: len(raw) - drop])
                else:
                    self.assertIsNone(out, f"{text!r} drop={drop}")

    def test_never_fabricates_replacement_chars(self) -> None:
        """Test never fabricates replacement chars."""
        # Docstring: "rather than fabricating replacement characters".
        for text in ["héllo", "a😀b", "日本語", "é", "\U0001f600\U0001f601"]:
            raw = text.encode("utf-8")
            for drop in range(1, len(raw)):
                out = truncate_text_tail(text, drop)
                if out is not None:
                    self.assertNotIn("�", out)
                    # No surrogates leak into the output either.
                    out.encode("utf-8")

    def test_deterministic(self) -> None:
        """Test deterministic."""
        cases = ["hello", "héllo", "a😀b", "", "é"]
        for text in cases:
            for drop in range(1, 9):
                try:
                    first = truncate_text_tail(text, drop)
                except ValueError:
                    first = "RAISED"
                try:
                    second = truncate_text_tail(text, drop)
                except ValueError:
                    second = "RAISED"
                self.assertEqual(first, second, f"{text!r} drop={drop}")

    def test_not_idempotent(self) -> None:
        """Test not idempotent."""
        # Documented non-property: applying the operator twice drops 2d
        # bytes, it does not converge. t(t(x,d),d) == t(x,2d) != t(x,d).
        self.assertEqual(
            truncate_text_tail(truncate_text_tail("hello world", 3), 3),
            truncate_text_tail("hello world", 6),
        )
        self.assertEqual(truncate_text_tail("hello world", 6), "hello")
        self.assertNotEqual(
            truncate_text_tail("hello world", 3), truncate_text_tail("hello world", 6)
        )

    def test_depths_do_not_compose(self) -> None:
        """Test depths do not compose."""
        # Composition can be UNDEFINED where the summed single drop is
        # defined: t("xhé",1) is None, yet t("xhé",3) == "x". The sweep must
        # therefore apply each depth fresh -- never chained (pinned
        # end-to-end in TestSweepAggregation.test_depths_are_independent).
        self.assertIsNone(truncate_text_tail("xhé", 1))
        self.assertEqual(truncate_text_tail("xhé", 3), "x")
        # And where both orders are defined they agree with the summed drop.
        self.assertEqual(
            truncate_text_tail(truncate_text_tail("abcdef", 1), 2),
            truncate_text_tail("abcdef", 3),
        )

    def test_lone_surrogate_edge_documented(self) -> None:
        """Test lone surrogate edge documented."""
        # Edge OUTSIDE the documented contract: lone surrogates are not
        # UTF-8-encodable, so strict encoding raises before any truncation
        # logic runs. Documented here as an observed edge, NOT a bug: the
        # sweep still records such a depth as a failed decision
        # (encode_failed) rather than crashing or corrupting.
        with self.assertRaises(UnicodeEncodeError):
            truncate_text_tail("a\ud800", 1)


class TruncateFuzz(unittest.TestCase):
    """Seeded fuzz: random prompts x all depths vs the independent oracle."""

    def test_fuzz_against_oracle(self) -> None:
        """Test fuzz against oracle."""
        rng = random.Random(0xC0FFEE)  # noqa: S311 -- deterministic test seed
        alphabet = (
            "abcdefghijklmnopqrstuvwxyz \n\t"
            "éüñßøå"
            "日本語テスト한글"
            "😀🎉☕"
            "́̂̃"  # combining marks U+0301/0302/0303
            "‍​"  # ZWJ, zero-width space
            "§¶"
        )
        texts: list[str] = ["", "a", "ab", "é", "😀"]
        for _ in range(1200):
            roll = rng.random()
            if roll < 0.05:
                length = rng.choice((0, 1, 2, 3))
            elif roll < 0.10:
                length = rng.randint(500, 2000)
            else:
                length = rng.randint(0, 64)
            texts.append("".join(rng.choice(alphabet) for _ in range(length)))
        # A lone-surrogate case exercises the documented edge path.
        texts.append("ok\ud800tail")

        for text in texts:
            try:
                raw = text.encode("utf-8")
            except UnicodeEncodeError:
                # Documented edge: outside the operator's input contract.
                with self.assertRaises(UnicodeEncodeError):
                    truncate_text_tail(text, 1)
                continue
            for drop in range(1, 9):
                with self.subTest(text=text[:20], drop=drop):
                    if not 1 <= drop < len(raw):
                        with self.assertRaises(ValueError):
                            truncate_text_tail(text, drop)
                        continue
                    out = truncate_text_tail(text, drop)
                    if _naive_cut_ok(raw, drop):
                        self.assertIsInstance(out, str)
                        assert isinstance(out, str)
                        self.assertEqual(out.encode("utf-8"), raw[: len(raw) - drop])
                        self.assertTrue(text.startswith(out))
                        self.assertNotIn("�", out)
                    else:
                        self.assertIsNone(out)
                    # Determinism on every fuzz case.
                    self.assertEqual(truncate_text_tail(text, drop), out)


class _RecordingInner:
    """Minimal backend double: records encode inputs, ord-based codec."""

    def __init__(self) -> None:
        self.fingerprint = "cycle56-recording-inner"
        self.hf_stop_token_bytes = b"stop"
        self.encode_inputs: list[str] = []
        self.encode_calls = 0

    def encode(self, text: str) -> tuple[int, ...]:
        """Encode."""
        self.encode_calls += 1
        self.encode_inputs.append(text)
        return tuple(ord(char) for char in text)

    @staticmethod
    def decode(token_ids: tuple[int, ...]) -> str:
        """Decode."""
        return "".join(chr(token) for token in token_ids)

    @staticmethod
    def generate(
        input_ids: tuple[int, ...], *, max_new_tokens: int, seed: int, cancel: Event
    ) -> core.Generation:
        """Generate."""
        return core.Generation(token_ids=(ord("~"),) * max_new_tokens, reason="limit")


class TruncatedBackendContract(unittest.TestCase):
    """Pin TruncatedPromptBackend's contract with its inner backend."""

    def test_encode_applies_truncation(self) -> None:
        """Test encode applies truncation."""
        inner = _RecordingInner()
        backend = TruncatedPromptBackend(inner, 2)
        self.assertEqual(backend.encode("hello"), tuple(ord(c) for c in "hel"))
        self.assertEqual(inner.encode_inputs, ["hel"])

    def test_encode_raises_fixed_code_on_split(self) -> None:
        """Test encode raises fixed code on split."""
        inner = _RecordingInner()
        backend = TruncatedPromptBackend(inner, 1)
        with self.assertRaises(ValueError) as ctx:
            backend.encode("hé")  # 3 bytes; drop 1 splits é
        self.assertEqual(str(ctx.exception), "prompt_truncation_splits_codepoint")
        # The inner backend never sees the undecodable cut.
        self.assertEqual(inner.encode_calls, 0)

    def test_encode_propagates_length_violation(self) -> None:
        """Test encode propagates length violation."""
        # A prompt shorter than drop_bytes is "cannot be applied": the
        # ValueError propagates (evaluate records encode_failed; the sweep
        # records the failed decision). It must NOT be masked as None.
        inner = _RecordingInner()
        backend = TruncatedPromptBackend(inner, 5)
        with self.assertRaises(ValueError) as ctx:
            backend.encode("hi")
        self.assertEqual(str(ctx.exception), "drop_bytes must lie within the text's byte length")

    def test_decode_and_generate_pass_through(self) -> None:
        """Test decode and generate pass through."""
        inner = _RecordingInner()
        backend = TruncatedPromptBackend(inner, 3)
        ids = (65, 66)
        self.assertEqual(backend.decode(ids), inner.decode(ids))
        event = Event()
        gen_inner = inner.generate(ids, max_new_tokens=4, seed=1, cancel=event)
        gen_wrapped = backend.generate(ids, max_new_tokens=4, seed=1, cancel=event)
        # Pass-through: the wrapper forwards the call and returns the inner
        # result unchanged (equal value; the wrapper constructs no copy).
        self.assertEqual(gen_wrapped, gen_inner)

    def test_fingerprint_and_stop_bytes(self) -> None:
        """Test fingerprint and stop bytes."""
        inner = _RecordingInner()
        backend = TruncatedPromptBackend(inner, 4)
        self.assertEqual(backend.fingerprint, "cycle56-recording-inner+prompt-trunc-4b")
        self.assertIs(backend.hf_stop_token_bytes, inner.hf_stop_token_bytes)

    def test_constructor_validation(self) -> None:
        """Test constructor validation."""
        inner = _RecordingInner()
        for bad in (0, -1, -100, "1", 1.0, None, True):
            with self.assertRaises(ValueError, msg=f"drop_bytes={bad!r}"):
                TruncatedPromptBackend(inner, bad)  # type: ignore[arg-type]

        class NoFingerprint:
            """No fingerprint."""

            fingerprint = 123

        with self.assertRaises(ValueError):
            TruncatedPromptBackend(NoFingerprint(), 1)

        class MissingFingerprint:
            """Missing fingerprint."""

        with self.assertRaises(ValueError):
            TruncatedPromptBackend(MissingFingerprint(), 1)

    def test_construction_has_no_side_effects(self) -> None:
        """Test construction has no side effects."""
        inner = _RecordingInner()
        TruncatedPromptBackend(inner, 2)
        self.assertEqual(inner.encode_calls, 0)
        self.assertEqual(inner.encode_inputs, [])


_MARKER = "§¶"
_BOUNDARY = tuple(ord(c) for c in _MARKER)
_CAP = 16


class _ScriptedBackend:
    """Deterministic text-keyed backend: generate() is scripted per prompt.

    ``script`` maps a prompt string to ``(answer, marker)``; the output is
    answer+marker padded to the token cap. Unscripted prompts raise
    KeyError loudly (the test pre-asserts script coverage instead).
    """

    def __init__(self, script: dict[str, tuple[str, str]]) -> None:
        self._script = dict(script)
        self.fingerprint = "cycle56-scripted-backend"
        self.hf_stop_token_bytes = None
        self.encode_inputs: list[str] = []

    def encode(self, text: str) -> tuple[int, ...]:
        """Encode."""
        self.encode_inputs.append(text)
        return tuple(ord(char) for char in text)

    @staticmethod
    def decode(token_ids: tuple[int, ...]) -> str:
        """Decode."""
        return "".join(chr(token) for token in token_ids)

    def generate(
        self, input_ids: tuple[int, ...], *, max_new_tokens: int, seed: int, cancel: Event
    ) -> core.Generation:
        """Generate."""
        prompt = "".join(chr(token) for token in input_ids)
        answer, marker = self._script[prompt]
        body = tuple(ord(c) for c in answer + marker)
        assert len(body) <= max_new_tokens
        padding = (ord("~"),) * (max_new_tokens - len(body))
        return core.Generation(token_ids=body + padding, reason="limit")


def _make_format() -> core.Format:
    """Make format."""
    # Rendered prompt ends with "A:éx" (12 bytes: ..."A:" + é(2) + "x"):
    #   drop 1 -> "Q:what?A:é"  (valid)
    #   drop 2 -> splits é      (None -> encode raises)
    #   drop 3 -> "Q:what?A:"   (valid)
    # -- exercises pass/flip, codepoint-split, and search-fail depths.
    return core.Format(name="F1", bos="", user_prefix="Q:", assistant_prefix="A:éx", turn_end="")


def _make_split() -> core.Split:
    """Make split."""
    search = (
        core.Fixture(
            id="s1",
            messages=(core.Message(role="user", content="what?"),),
            expected="42",
            group="g1",
        ),
    )
    held_out = (
        core.Fixture(
            id="h1",
            messages=(core.Message(role="user", content="other?"),),
            expected="42",
            group="g2",
        ),
    )
    return core.Split(search=search, held_out=held_out)


def _flip_script() -> dict[str, tuple[str, str]]:
    """Script where depths 1..3 hit: decision-flip, codepoint-split, winner-flip."""
    return {
        # Native prompts: everything passes.
        "Q:what?A:éx": ("42", _MARKER),
        "Q:other?A:éx": ("42", _MARKER),
        # Depth 1 (drop 1 -> "Q:what?A:é"): search passes, held-out fails.
        "Q:what?A:é": ("42", _MARKER),
        "Q:other?A:é": ("WRONG", _MARKER),
        # Depth 2 (drop 2): truncate_text_tail returns None -> encode raises
        # before generate is ever called; no script entry needed or wanted.
        # Depth 3 (drop 3 -> "Q:what?A:"): search fails outright.
        "Q:what?A:": ("WRONG", _MARKER),
        "Q:other?A:": ("42", _MARKER),
    }


def _stable_script() -> dict[str, tuple[str, str]]:
    """Script where depth 1 changes nothing observable (no flip)."""
    return {
        "Q:what?A:éx": ("42", _MARKER),
        "Q:other?A:éx": ("42", _MARKER),
        "Q:what?A:é": ("42", _MARKER),
        "Q:other?A:é": ("42", _MARKER),
    }


class SweepAggregation(unittest.TestCase):
    """Pin the sweep's aggregation: winner_changed truth table + flips_at."""

    @staticmethod
    def _run(script: dict[str, tuple[str, str]], max_drop: int):
        """Run."""
        backend = _ScriptedBackend(script)
        result = prompt_sensitivity_sweep(
            backend,
            (_make_format(),),
            _make_split(),
            boundary=_BOUNDARY,
            max_drop_bytes=max_drop,
            max_new_tokens=_CAP,
            seed=7,
        )
        return backend, result

    def test_flip_scenario_aggregation(self) -> None:
        """Test flip scenario aggregation."""
        _, result = self._run(_flip_script(), 3)
        self.assertEqual(result["native_decision"], "compatible_on_fixture_set")
        self.assertEqual(result["native_winner"], "F1")
        self.assertEqual(result["max_drop_bytes"], 3)
        depths = result["depths"]
        self.assertEqual([d["drop_bytes"] for d in depths], [1, 2, 3])
        # Depth 1: winner SAME ("F1"), decision DIFFERS -> winner_changed True.
        # This branch kills a mutant that drops the decision comparison.
        self.assertEqual(depths[0]["decision"], "held_out_failed")
        self.assertEqual(depths[0]["selected_candidate"], "F1")
        self.assertTrue(depths[0]["winner_changed"])
        # Depth 2: codepoint split -> the run's failed decision, recorded not skipped.
        self.assertEqual(depths[1]["decision"], "failed")
        self.assertIsNone(depths[1]["selected_candidate"])
        self.assertTrue(depths[1]["winner_changed"])
        # Depth 3: winner DIFFERS (None vs "F1").
        self.assertEqual(depths[2]["decision"], "no_candidate")
        self.assertIsNone(depths[2]["selected_candidate"])
        self.assertTrue(depths[2]["winner_changed"])
        self.assertEqual(result["flips_at"], [1, 2, 3])

    def test_stable_scenario_no_flips(self) -> None:
        """Test stable scenario no flips."""
        # Winner same AND decision same -> winner_changed False, flips_at [].
        _, result = self._run(_stable_script(), 1)
        self.assertEqual(result["native_decision"], "compatible_on_fixture_set")
        self.assertEqual(result["native_winner"], "F1")
        depths = result["depths"]
        self.assertEqual(len(depths), 1)
        self.assertEqual(depths[0]["drop_bytes"], 1)
        self.assertEqual(depths[0]["decision"], "compatible_on_fixture_set")
        self.assertEqual(depths[0]["selected_candidate"], "F1")
        self.assertFalse(depths[0]["winner_changed"])
        self.assertEqual(result["flips_at"], [])

    def test_depths_are_independent_not_chained(self) -> None:
        """Test depths are independent not chained."""
        # Each depth truncates the NATIVE prompt fresh. If depths chained
        # (depth d applied to depth d-1's output), depth 2 would truncate
        # "Q:what?A:é" by 1 -> "Q:what?A:" (valid, no split) and the decision
        # would not be "failed". The recorded "failed" pins independence.
        backend, result = self._run(_flip_script(), 3)
        self.assertEqual(result["depths"][1]["decision"], "failed")
        # The inner backend only ever saw whole-character prefixes of native
        # prompts, each shortened by at most its own depth -- never a
        # double-truncated or mid-codepoint string.
        self.assertEqual(
            backend.encode_inputs,
            [
                "Q:what?A:éx",  # native search
                "Q:other?A:éx",  # native held-out
                "Q:what?A:é",  # depth 1
                "Q:other?A:é",
                # depth 2: encode raises before the inner backend is called
                "Q:what?A:",  # depth 3 search only: no winner -> no held-out run
            ],
        )

    def test_sweep_is_deterministic(self) -> None:
        """Test sweep is deterministic."""
        _, first = self._run(_flip_script(), 3)
        _, second = self._run(_flip_script(), 3)
        self.assertEqual(first, second)

    def test_max_drop_bytes_validation(self) -> None:
        """Test max drop bytes validation."""
        backend = _ScriptedBackend(_flip_script())
        for bad in (0, -1, 9, 100, "3", 2.5, None, True):
            with self.assertRaises(ValueError, msg=f"max_drop_bytes={bad!r}"):
                prompt_sensitivity_sweep(
                    backend,
                    (_make_format(),),
                    _make_split(),
                    boundary=_BOUNDARY,
                    max_drop_bytes=bad,  # type: ignore[arg-type]
                    max_new_tokens=_CAP,
                    seed=7,
                )

    def test_all_depths_failed_still_reports(self) -> None:
        """Test all depths failed still reports."""
        # Every depth splits a codepoint: all decisions "failed", winners
        # None, flips recorded -- nothing is silently skipped. A trailing
        # 4-byte emoji makes drops 1..3 all split it.
        fmt = core.Format(name="F1", bos="", user_prefix="Q:", assistant_prefix="A:😀", turn_end="")
        split = _make_split()
        script = {
            "Q:what?A:😀": ("42", _MARKER),
            "Q:other?A:😀": ("42", _MARKER),
        }
        backend = _ScriptedBackend(script)
        result = prompt_sensitivity_sweep(
            backend,
            (fmt,),
            split,
            boundary=_BOUNDARY,
            max_drop_bytes=3,
            max_new_tokens=_CAP,
            seed=7,
        )
        self.assertEqual(result["native_decision"], "compatible_on_fixture_set")
        self.assertEqual([d["decision"] for d in result["depths"]], ["failed", "failed", "failed"])
        self.assertTrue(all(d["winner_changed"] for d in result["depths"]))
        self.assertEqual(result["flips_at"], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
