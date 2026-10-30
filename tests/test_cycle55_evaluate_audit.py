"""Cycle 55: differential audit of ``core.evaluate()`` vs an independent oracle.

Scope: ``evaluate()`` and its immediate observation-pipeline helpers --
exact answer match, boundary detection (``first_boundary``), observation
assembly (``record``/``terminate``/``abort``), the diagnostic
``_boundary_text_match``, decode-failure handling, and the failure taxonomy
(``render_failed`` / ``encode_failed`` / ``invalid_input_ids`` /
``generate_failed`` / ``invalid_generation`` / ``decode_failed`` /
``backend_reported_error`` / allowlisted backend codes).

Method: ``oracle_evaluate`` below reimplements the documented contract from
scratch (control flow, validation, seed derivation, sanitization rules) using
only the dataclass constructors as shared vocabulary. A scriptable backend
drives both implementations through identical randomized scenarios; every
``Observation`` field is compared field-by-field, plus report status and the
request-seed derivation.

REAL -> ORACLE
--------------
No discrepancies found between ``core.evaluate()`` and the independent
oracle across 400 randomized differential trials plus 25 targeted edge-case
tests (run 2026-10-15). This section documents real bugs only; it stays
empty when the audit finds none.
"""

import random
import unicodedata
import unittest
from dataclasses import asdict, dataclass
from hashlib import sha256
from threading import Event

from model_protocol_lab.core import (
    EvaluationReport,
    Fixture,
    Format,
    Generation,
    Message,
    Observation,
    evaluate,
)

# ---------------------------------------------------------------------------
# Scriptable backend: deterministic, fully controllable encode/decode/generate.
# ---------------------------------------------------------------------------

_BASE_VOCAB = {t: chr(0x61 + (t % 26)) for t in range(64)}
_BASE_VOCAB.update(
    {
        1: "\u03b1",  # multi-byte boundary tokens
        2: "\u03b2",
        3: "\u03b3",
        810: "\u2192",
        811: "\U0001f389",
        812: "\u4e2d",
    }
)
# Same decoded text as tokens 1/2/3 but different IDs: exercises the
# tokenization-mismatch false-negative mode (boundary_found=False yet
# boundary_text_match=True).
_ALIAS_VOCAB = {901: "\u03b1", 902: "\u03b2", 903: "\u03b3"}
_FAIL_TOKEN = 500  # decode raises on any id list containing this token
_NONSTR_TOKEN = 600  # decode returns a non-str for any id list containing it
_EMPTY_TEXT_TOKEN = 700  # decodes to the empty string

_PROBE_SECRET = "s3cr3t-hunter2-leakprobe"

_ALLOWLIST_CODES = frozenset(
    {
        "invalid_or_empty_input",
        "invalid_output_cap",
        "invalid_seed",
        "invalid_cancellation_event",
        "context_limit_exceeded",
    }
)

# Every error value an Observation may legitimately carry.
_FIXED_ERROR_VOCAB = _ALLOWLIST_CODES | frozenset(
    {
        "render_failed",
        "encode_failed",
        "invalid_input_ids",
        "generate_failed",
        "invalid_generation",
        "decode_failed",
        "backend_reported_error",
    }
)


def scripted_decode(token_ids: tuple[int, ...]) -> str:
    """Pure, deterministic decode shared by backend and test builders."""
    parts: list[str] = []
    for token in token_ids:
        if token == _FAIL_TOKEN:
            raise RuntimeError(f"decode exploded: {_PROBE_SECRET}")
        if token == _NONSTR_TOKEN:
            return 12345  # type: ignore[return-value] -- deliberate non-str
        if token == _EMPTY_TEXT_TOKEN:
            parts.append("")
        elif token in _ALIAS_VOCAB:
            parts.append(_ALIAS_VOCAB[token])
        else:
            parts.append(_BASE_VOCAB.get(token, f"<{token}>"))
    return "".join(parts)


@dataclass
class CaseScript:
    """Per-fixture behaviour script for the backend."""

    encode_kind: str = "ok"  # ok | raise | invalid_list | invalid_neg | invalid_bool
    gen_kind: str = "limit"  # limit | cancelled | error_allow | error_text |
    # error_none | raise | bad_too_long | bad_limit_short | bad_reason |
    # bad_error_type | bad_error_on_limit | bad_not_generation
    tokens: tuple[int, ...] = ()
    set_event_at: str | None = None  # None | encode | generate | decode_answer | decode_diag


class ScriptBackend:
    """Deterministic backend driven by a per-fixture CaseScript list."""

    def __init__(
        self,
        candidate: Format,
        fixtures: list[Fixture],
        scripts: list[CaseScript],
        event: Event,
    ) -> None:
        # encode() is called at most once per case, in case order, so the
        # fixture index is tracked by call count -- never by prompt text
        # (fixtures may share identical messages).
        self._scripts = scripts
        self._event = event
        self._encode_calls = 0
        self.seeds: list[tuple[int, int]] = []
        self._decode_calls_since_generate = 0
        self._current_index = 0

    # -- Backend protocol -------------------------------------------------
    def encode(self, text: str) -> tuple[int, ...]:
        """Encode."""
        index = self._encode_calls
        self._encode_calls += 1
        script = self._scripts[index]
        if script.set_event_at == "encode":
            self._event.set()
        if script.encode_kind == "raise":
            raise RuntimeError(f"encode exploded: {_PROBE_SECRET}")
        if script.encode_kind == "invalid_list":
            return [1, 2]  # type: ignore[return-value]
        if script.encode_kind == "invalid_neg":
            return (1, -1)
        if script.encode_kind == "invalid_bool":
            return (1, True)  # type: ignore[return-value]
        return (1000 + index,)

    def decode(self, token_ids: tuple[int, ...]) -> str:
        """Decode."""
        self._decode_calls_since_generate += 1
        script = self._current_script()
        if script.set_event_at == "decode_answer" and self._decode_calls_since_generate == 1:
            self._event.set()
        if script.set_event_at == "decode_diag" and self._decode_calls_since_generate >= 2:
            self._event.set()
        return scripted_decode(token_ids)

    def generate(
        self, input_ids: tuple[int, ...], *, max_new_tokens: int, seed: int, cancel: Event
    ) -> Generation:
        """Generate."""
        assert input_ids and 1000 <= input_ids[0] < 1000 + len(self._scripts)
        index = input_ids[0] - 1000
        script = self._scripts[index]
        self._current_index = index
        self._decode_calls_since_generate = 0
        self.seeds.append((index, seed))
        if script.set_event_at == "generate":
            self._event.set()
        kind = script.gen_kind
        tokens = script.tokens
        if kind == "raise":
            raise RuntimeError(f"generate exploded: {_PROBE_SECRET}")
        if kind == "limit":
            assert len(tokens) == max_new_tokens
            return Generation(token_ids=tokens, reason="limit")
        if kind == "cancelled":
            return Generation(token_ids=tokens, reason="cancelled")
        if kind == "error_allow":
            return Generation(token_ids=tokens, reason="error", error="context_limit_exceeded")
        if kind == "error_text":
            return Generation(token_ids=tokens, reason="error", error=f"kaboom: {_PROBE_SECRET}")
        if kind == "error_none":
            return Generation(token_ids=tokens, reason="error", error=None)
        if kind == "bad_too_long":
            return Generation(token_ids=tokens + (9, 9, 9), reason="limit")
        if kind == "bad_limit_short":
            return Generation(token_ids=tokens[:-1], reason="limit")
        if kind == "bad_reason":
            return Generation(token_ids=tokens, reason="bogus")  # type: ignore[arg-type]
        if kind == "bad_error_type":
            return Generation(token_ids=tokens, reason="error", error=123)  # type: ignore[arg-type]
        if kind == "bad_error_on_limit":
            return Generation(token_ids=tokens, reason="limit", error="context_limit_exceeded")
        if kind == "bad_not_generation":
            return ("not", "a", "generation")  # type: ignore[return-value]
        raise AssertionError(f"unknown gen_kind {kind!r}")

    def _current_script(self) -> CaseScript:
        """Current script."""
        return self._scripts[self._current_index]


# ---------------------------------------------------------------------------
# Independent oracle: reimplementation of the evaluate() contract from the
# documented behaviour. No code is shared with core.py beyond the dataclass
# constructors (Fixture/Format/Message/Generation/Observation/
# EvaluationReport) and stdlib primitives.
# ---------------------------------------------------------------------------


def _o_valid_token_ids(value: object, *, nonempty: bool = False) -> bool:
    """O valid token ids."""
    return (
        type(value) is tuple
        and (not nonempty or bool(value))
        and all(type(token) is int and token >= 0 for token in value)
    )


def _o_valid_generation(result: object, cap: int) -> bool:
    """O valid generation."""
    if type(result) is not Generation or not _o_valid_token_ids(result.token_ids):
        return False
    if len(result.token_ids) > cap or type(result.reason) is not str:
        return False
    if result.reason not in ("limit", "cancelled", "error"):
        return False
    if result.error is not None and type(result.error) is not str:
        return False
    if result.reason != "error" and result.error is not None:
        return False
    if result.reason == "limit":
        return len(result.token_ids) == cap
    return True


def _o_first_boundary(tokens: tuple[int, ...], stop: tuple[int, ...]) -> int | None:
    """O first boundary."""
    if not stop:
        raise ValueError("A nonempty boundary is required")
    for index in range(len(tokens) - len(stop) + 1):
        if tokens[index : index + len(stop)] == stop:
            return index
    return None


def _o_boundary_text_match(
    backend: ScriptBackend,
    output_ids: tuple[int, ...],
    cut: int | None,
    boundary: tuple[int, ...],
) -> bool | None:
    """O boundary text match."""
    try:
        boundary_text = backend.decode(boundary)
    except Exception:  # noqa: BLE001 -- diagnostic must never fail the run
        return None
    if type(boundary_text) is not str or not boundary_text:
        return None
    retained = output_ids if cut is None else output_ids[: cut + len(boundary)]
    try:
        retained_text = backend.decode(retained)
    except Exception:  # noqa: BLE001 -- diagnostic must never fail the run
        return None
    if type(retained_text) is not str:
        return None
    return retained_text.endswith(boundary_text)


def _o_request_seed(seed: int, case_id: str) -> int:
    """O request seed."""
    material = f"{seed}\0{case_id}".encode("utf-8", errors="surrogatepass")
    return int.from_bytes(sha256(material).digest()[:4], "big")


def oracle_evaluate(
    backend: ScriptBackend,
    candidate: Format,
    cases: tuple[Fixture, ...],
    *,
    boundary: tuple[int, ...],
    max_new_tokens: int = 64,
    seed: int = 0,
    cancel: Event | None = None,
) -> EvaluationReport:
    """Independent reimplementation of the ``evaluate()`` contract."""
    if type(candidate) is not Format:
        raise ValueError("candidate must be a Format")
    if type(cases) is not tuple or not cases or any(type(c) is not Fixture for c in cases):
        raise ValueError("cases must be a nonempty tuple of Fixture")
    if len({unicodedata.normalize("NFC", c.id) for c in cases}) != len(cases):
        raise ValueError("Evaluation fixture IDs must be unique")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 4096:
        raise ValueError("Output cap must be an integer between 1 and 4096")
    if not _o_valid_token_ids(boundary, nonempty=True):
        raise ValueError("Boundary must be a nonempty tuple of token IDs")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    if cancel is not None and not isinstance(cancel, Event):
        raise ValueError("cancel must be a threading.Event")
    event = cancel if cancel is not None else Event()
    records: list[Observation] = []

    def emit(
        status: str,
        case: Fixture,
        input_ids: tuple[int, ...],
        output_ids: tuple[int, ...],
        answer: str,
        cut: int | None,
        raw_reason: str | None,
        error: str | None,
        boundary_text_match: bool | None = None,
        input_text: str = "",
    ) -> None:
        """Emit."""
        records.append(
            Observation(
                candidate=candidate.name,
                fixture=case.id,
                input_ids=input_ids,
                output_ids=output_ids,
                answer=answer,
                exact_match=status == "completed" and answer == case.expected,
                boundary_found=cut is not None,
                boundary_text_match=boundary_text_match,
                raw_reason=raw_reason,
                error=error,
                status=status,  # type: ignore[arg-type]
                input_text=input_text,
            )
        )

    def done(status: str) -> EvaluationReport:
        """Done."""
        return EvaluationReport(tuple(records), len(cases), status)  # type: ignore[arg-type]

    for case in cases:
        if event.is_set():
            return done("cancelled")
        input_ids: tuple[int, ...] = ()
        output_ids: tuple[int, ...] = ()
        answer = ""
        raw_reason: str | None = None
        cut: int | None = None
        prompt_text = ""

        try:
            prompt = candidate.render(case.messages)
        except Exception:  # noqa: BLE001 -- sanitization boundary
            emit("failed", case, (), (), "", None, None, "render_failed")
            return done("failed")
        prompt_text = prompt
        if event.is_set():
            emit("cancelled", case, (), (), "", None, None, None)
            return done("cancelled")
        try:
            encoded = backend.encode(prompt)
        except Exception:  # noqa: BLE001 -- sanitization boundary
            emit("failed", case, (), (), "", None, None, "encode_failed")
            return done("failed")
        if not _o_valid_token_ids(encoded):
            emit("failed", case, (), (), "", None, None, "invalid_input_ids")
            return done("failed")
        input_ids = encoded
        if event.is_set():
            # NOTE: terminate (not abort) here: the encoded input ids are
            # retained in the cancelled observation.
            emit("cancelled", case, input_ids, (), "", None, None, None, input_text=prompt_text)
            return done("cancelled")
        request_seed = _o_request_seed(seed, case.id)
        try:
            result = backend.generate(
                input_ids, max_new_tokens=max_new_tokens, seed=request_seed, cancel=event
            )
        except Exception:  # noqa: BLE001 -- sanitization boundary
            emit(
                "failed",
                case,
                input_ids,
                (),
                "",
                None,
                None,
                "generate_failed",
                input_text=prompt_text,
            )
            return done("failed")
        if not _o_valid_generation(result, max_new_tokens):
            emit(
                "failed",
                case,
                input_ids,
                (),
                "",
                None,
                None,
                "invalid_generation",
                input_text=prompt_text,
            )
            return done("failed")
        output_ids = result.token_ids
        raw_reason = result.reason
        cut = _o_first_boundary(output_ids, boundary)
        if result.reason == "error":
            code = (
                result.error
                if type(result.error) is str and result.error in _ALLOWLIST_CODES
                else "backend_reported_error"
            )
            emit(
                "failed",
                case,
                input_ids,
                output_ids,
                "",
                cut,
                raw_reason,
                code,
                input_text=prompt_text,
            )
            return done("failed")
        if result.reason == "cancelled" or event.is_set():
            emit(
                "cancelled",
                case,
                input_ids,
                output_ids,
                "",
                cut,
                raw_reason,
                None,
                input_text=prompt_text,
            )
            return done("cancelled")
        answer_ids = output_ids if cut is None else output_ids[:cut]
        try:
            decoded = backend.decode(answer_ids)
        except Exception:  # noqa: BLE001 -- sanitization boundary
            emit(
                "failed",
                case,
                input_ids,
                output_ids,
                "",
                cut,
                raw_reason,
                "decode_failed",
                input_text=prompt_text,
            )
            return done("failed")
        if type(decoded) is not str:
            emit(
                "failed",
                case,
                input_ids,
                output_ids,
                "",
                cut,
                raw_reason,
                "decode_failed",
                input_text=prompt_text,
            )
            return done("failed")
        answer = decoded
        if event.is_set():
            emit(
                "cancelled",
                case,
                input_ids,
                output_ids,
                answer,
                cut,
                raw_reason,
                None,
                input_text=prompt_text,
            )
            return done("cancelled")
        text_match = _o_boundary_text_match(backend, output_ids, cut, boundary)
        emit(
            "completed",
            case,
            input_ids,
            output_ids,
            answer,
            cut,
            raw_reason,
            None,
            boundary_text_match=text_match,
            input_text=prompt_text,
        )
    return done("cancelled" if event.is_set() else "completed")


# ---------------------------------------------------------------------------
# Randomized differential fuzzing.
# ---------------------------------------------------------------------------

_CANDIDATE = Format(
    name="audit-candidate",
    bos="<s>",
    user_prefix="[U]",
    assistant_prefix="[A]",
    turn_end="[/]",
)

_TEXT_POOL = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-.,!?'\n\té→中🎉\u0301"
)
_BOUNDARIES = [(1,), (2, 3), (1, 2, 3), (3, 3), (7,), (5, 6), (1,)]
_TOKEN_POOL = list(range(0, 31)) + [901, 902, 903]


def _rand_text(rng: random.Random, max_len: int = 10) -> str:
    """Rand text."""
    return "".join(rng.choice(_TEXT_POOL) for _ in range(rng.randint(0, max_len)))


def _rand_tokens(rng: random.Random, length: int) -> tuple[int, ...]:
    """Rand tokens."""
    return tuple(rng.choice(_TOKEN_POOL) for _ in range(length))


def _contains_boundary(tokens: tuple[int, ...], boundary: tuple[int, ...]) -> bool:
    """Contains boundary."""
    return any(
        tokens[i : i + len(boundary)] == boundary for i in range(len(tokens) - len(boundary) + 1)
    )


def _build_trial(
    rng: random.Random,
) -> tuple[list[Fixture], list[CaseScript], tuple[int, ...], int, int]:
    """Build one randomized (fixtures, scripts, boundary, cap, seed) trial."""
    boundary = rng.choice(_BOUNDARIES)
    cap = rng.randint(1, 6)
    seed = rng.randint(0, 2**64 - 1)
    n_cases = rng.randint(1, 3)
    fixtures: list[Fixture] = []
    scripts: list[CaseScript] = []
    for i in range(n_cases):
        if rng.random() < 0.05:
            # render_failed: messages that pass Fixture validation but break render().
            bad = rng.choice(["dup_user", "assistant_last"])
            if bad == "dup_user":
                messages = (
                    Message(role="user", content="q"),
                    Message(role="user", content="q2"),
                )
            else:
                messages = (
                    Message(role="user", content="q"),
                    Message(role="assistant", content="a"),
                )
            fixtures.append(
                Fixture(id=f"case-{i}-renderfail", messages=messages, expected="x", group="g")
            )
            scripts.append(CaseScript())
            continue
        messages = (Message(role="user", content=_rand_text(rng, 12)),)
        if rng.random() < 0.4:
            messages = messages + (Message(role="assistant", content=_rand_text(rng, 6)),)
            messages = messages + (Message(role="user", content=_rand_text(rng, 12)),)
        script = CaseScript()
        enc_roll = rng.random()
        if enc_roll < 0.05:
            script.encode_kind = "raise"
        elif enc_roll < 0.10:
            script.encode_kind = rng.choice(["invalid_list", "invalid_neg", "invalid_bool"])
        gen_roll = rng.random()
        if gen_roll < 0.50:
            script.gen_kind = "limit"
            tokens = _rand_tokens(rng, cap)
            if rng.random() < 0.55 and len(boundary) <= cap:
                pos = rng.randint(0, cap - len(boundary))
                tokens = tokens[:pos] + boundary + tokens[pos + len(boundary) :]
            script.tokens = tokens
        elif gen_roll < 0.62:
            script.gen_kind = "cancelled"
            script.tokens = _rand_tokens(rng, rng.randint(0, cap))
        elif gen_roll < 0.68:
            script.gen_kind = "error_allow"
            script.tokens = _rand_tokens(rng, rng.randint(0, cap))
        elif gen_roll < 0.74:
            script.gen_kind = "error_text"
            script.tokens = _rand_tokens(rng, rng.randint(0, cap))
        elif gen_roll < 0.77:
            script.gen_kind = "error_none"
            script.tokens = _rand_tokens(rng, rng.randint(0, cap))
        elif gen_roll < 0.81:
            script.gen_kind = "raise"
        elif gen_roll < 0.84:
            script.gen_kind = "bad_too_long"
            script.tokens = _rand_tokens(rng, cap)
        elif gen_roll < 0.87:
            script.gen_kind = "bad_limit_short"
            script.tokens = _rand_tokens(rng, cap)
        elif gen_roll < 0.90:
            script.gen_kind = "bad_reason"
            script.tokens = _rand_tokens(rng, rng.randint(0, cap))
        elif gen_roll < 0.92:
            script.gen_kind = "bad_error_type"
            script.tokens = _rand_tokens(rng, rng.randint(0, cap))
        elif gen_roll < 0.94:
            script.gen_kind = "bad_error_on_limit"
            script.tokens = _rand_tokens(rng, cap)
        elif gen_roll < 0.96:
            script.gen_kind = "bad_not_generation"
        else:
            script.gen_kind = "limit"
            script.tokens = _rand_tokens(rng, cap)
        # Decode-failure injection on paths that reach the answer decode.
        if script.gen_kind in ("limit", "cancelled") and rng.random() < 0.12 and script.tokens:
            tokens = list(script.tokens)
            tokens[rng.randrange(len(tokens))] = rng.choice([_FAIL_TOKEN, _NONSTR_TOKEN])
            script.tokens = tuple(tokens)
        # Cancellation hooks.
        if rng.random() < 0.06:
            script.set_event_at = rng.choice(["encode", "generate", "decode_answer", "decode_diag"])
        # Expected answer: exact-match, near-miss, or unrelated.
        exp_roll = rng.random()
        if exp_roll < 0.35 and script.gen_kind == "limit":
            attempt = _rand_tokens(rng, cap)
            for _ in range(20):
                if not _contains_boundary(attempt, boundary):
                    break
                attempt = _rand_tokens(rng, cap)
            if not _contains_boundary(attempt, boundary):
                script.tokens = attempt
                expected = scripted_decode(attempt)
            else:
                expected = _rand_text(rng)
        elif exp_roll < 0.45:
            expected = ""
        else:
            expected = _rand_text(rng)
        fixtures.append(Fixture(id=f"case-{i}", messages=messages, expected=expected, group="g"))
        scripts.append(script)
    return fixtures, scripts, boundary, cap, seed


def _run_pair(
    fixtures: list[Fixture],
    scripts: list[CaseScript],
    boundary: tuple[int, ...],
    cap: int,
    seed: int,
    preset_cancel: bool = False,
) -> tuple[EvaluationReport, EvaluationReport, ScriptBackend, ScriptBackend]:
    """Run pair."""
    event_a, event_b = Event(), Event()
    if preset_cancel:
        event_a.set()
        event_b.set()
    backend_a = ScriptBackend(_CANDIDATE, fixtures, scripts, event_a)
    backend_b = ScriptBackend(_CANDIDATE, fixtures, scripts, event_b)
    real = evaluate(
        backend_a,
        _CANDIDATE,
        tuple(fixtures),
        boundary=boundary,
        max_new_tokens=cap,
        seed=seed,
        cancel=event_a,
    )
    expected = oracle_evaluate(
        backend_b,
        _CANDIDATE,
        tuple(fixtures),
        boundary=boundary,
        max_new_tokens=cap,
        seed=seed,
        cancel=event_b,
    )
    return real, expected, backend_a, backend_b


def _assert_reports_equal(real: EvaluationReport, expected: EvaluationReport) -> None:
    """Assert reports equal."""
    assert [asdict(o) for o in real.observations] == [asdict(o) for o in expected.observations]
    assert real.status == expected.status
    assert real.planned_count == expected.planned_count
    assert real.attempted_count == expected.attempted_count
    # EvaluationReport.__post_init__ already validated invariants on construction.


class EvaluateAuditTests(unittest.TestCase):
    """unittest entry point; pytest runs TestCase classes natively."""

    @staticmethod
    def test_differential_fuzz_400_trials():
        """Test differential fuzz 400 trials."""
        rng = random.Random(20261015)  # noqa: S311 -- deterministic test seed
        for _ in range(400):
            fixtures, scripts, boundary, cap, seed = _build_trial(rng)
            real, expected, _, _ = _run_pair(fixtures, scripts, boundary, cap, seed)
            _assert_reports_equal(real, expected)

    @staticmethod
    def test_differential_fuzz_preset_cancel():
        """Test differential fuzz preset cancel."""
        rng = random.Random(99)  # noqa: S311 -- deterministic test seed
        for _ in range(25):
            fixtures, scripts, boundary, cap, seed = _build_trial(rng)
            real, expected, _, _ = _run_pair(
                fixtures, scripts, boundary, cap, seed, preset_cancel=True
            )
            _assert_reports_equal(real, expected)
            assert real.status == "cancelled"
            assert real.observations == ()

    @staticmethod
    def test_exact_match_with_boundary_strip():
        """Test exact match with boundary strip."""
        tokens = (10, 11, 1)
        expected = scripted_decode((10, 11))
        real, oracle = _one_case(tokens=tokens, boundary=(1,), expected=expected, cap=3)
        _assert_reports_equal(real, oracle)
        (obs,) = real.observations
        assert obs.status == "completed"
        assert obs.answer == expected
        assert obs.exact_match is True
        assert obs.boundary_found is True
        assert obs.boundary_text_match is True
        assert obs.output_ids == tokens
        assert obs.raw_reason == "limit"
        assert obs.error is None

    @staticmethod
    def test_case_difference_is_not_exact_match():
        """Test case difference is not exact match."""
        tokens = (10, 11)  # decodes to "kl"
        real, oracle = _one_case(tokens=tokens, boundary=(9,), expected="KL", cap=2)
        _assert_reports_equal(real, oracle)
        (obs,) = real.observations
        assert obs.answer == "kl"
        assert obs.exact_match is False
        assert obs.boundary_found is False

    @staticmethod
    def test_boundary_at_position_zero():
        """Test boundary at position zero."""
        real, oracle = _one_case(tokens=(1, 10), boundary=(1,), expected="", cap=2)
        _assert_reports_equal(real, oracle)
        (obs,) = real.observations
        assert obs.boundary_found is True
        assert obs.answer == ""
        assert obs.exact_match is True

    @staticmethod
    def test_boundary_longer_than_output():
        """Test boundary longer than output."""
        real, oracle = _one_case(tokens=(1, 2), boundary=(1, 2, 3), expected="x", cap=2)
        _assert_reports_equal(real, oracle)
        (obs,) = real.observations
        assert obs.boundary_found is False
        assert obs.boundary_text_match is False

    @staticmethod
    def test_overlapping_boundary():
        """Test overlapping boundary."""
        real, oracle = _one_case(tokens=(3, 3, 3), boundary=(3, 3), expected="", cap=3)
        _assert_reports_equal(real, oracle)
        (obs,) = real.observations
        assert obs.boundary_found is True
        assert obs.answer == ""  # first occurrence wins: cut == 0

    @staticmethod
    def test_tokenization_mismatch_flag():
        """Test tokenization mismatch flag."""
        real, oracle = _one_case(tokens=(7, 901), boundary=(1,), expected="x", cap=2)
        _assert_reports_equal(real, oracle)
        (obs,) = real.observations
        assert obs.boundary_found is False
        assert obs.boundary_text_match is True
        assert obs.answer == scripted_decode((7, 901)) == "hα"

    @staticmethod
    def test_empty_generation_cancelled():
        """Test empty generation cancelled."""
        real, oracle = _one_case(tokens=(), boundary=(1,), expected="", gen_kind="cancelled", cap=4)
        _assert_reports_equal(real, oracle)
        (obs,) = real.observations
        assert obs.status == "cancelled"
        assert obs.output_ids == ()
        assert obs.answer == ""
        assert obs.exact_match is False
        assert obs.boundary_found is False
        assert obs.raw_reason == "cancelled"

    @staticmethod
    def test_unicode_roundtrip():
        """Test unicode roundtrip."""
        tokens = (1, 2, 3, 810, 811, 812)
        expected = scripted_decode(tokens)
        assert expected == "αβγ→🎉中"
        real, oracle = _one_case(tokens=tokens, boundary=(9,), expected=expected, cap=6)
        _assert_reports_equal(real, oracle)
        (obs,) = real.observations
        assert obs.exact_match is True
        assert obs.answer == "αβγ→🎉中"

    @staticmethod
    def test_decode_failure_taxonomy():
        """Test decode failure taxonomy."""
        real, oracle = _one_case(tokens=(10, _FAIL_TOKEN, 11), boundary=(9,), expected="x", cap=3)
        _assert_reports_equal(real, oracle)
        (obs,) = real.observations
        assert obs.status == "failed"
        assert obs.error == "decode_failed"
        assert obs.answer == ""
        assert obs.raw_reason == "limit"
        assert obs.output_ids == (10, _FAIL_TOKEN, 11)

    @staticmethod
    def test_decode_non_str_taxonomy():
        """Test decode non str taxonomy."""
        real, oracle = _one_case(tokens=(_NONSTR_TOKEN,), boundary=(9,), expected="x", cap=1)
        _assert_reports_equal(real, oracle)
        (obs,) = real.observations
        assert obs.status == "failed"
        assert obs.error == "decode_failed"

    @staticmethod
    def test_diagnostic_decode_failure_gives_none():
        """Test diagnostic decode failure gives none."""
        real, oracle = _one_case(tokens=(10, 11), boundary=(_FAIL_TOKEN,), expected="kl", cap=2)
        _assert_reports_equal(real, oracle)
        (obs,) = real.observations
        assert obs.status == "completed"
        assert obs.exact_match is True
        assert obs.boundary_found is False
        assert obs.boundary_text_match is None

    @staticmethod
    def test_empty_boundary_text_gives_none():
        """Test empty boundary text gives none."""
        real, oracle = _one_case(tokens=(10,), boundary=(_EMPTY_TEXT_TOKEN,), expected="k", cap=1)
        _assert_reports_equal(real, oracle)
        (obs,) = real.observations
        assert obs.status == "completed"
        assert obs.boundary_found is False
        assert obs.boundary_text_match is None

    @staticmethod
    def test_error_allowlist_passthrough():
        """Test error allowlist passthrough."""
        real, oracle = _one_case(
            tokens=(5,), boundary=(9,), expected="x", gen_kind="error_allow", cap=4
        )
        _assert_reports_equal(real, oracle)
        (obs,) = real.observations
        assert obs.status == "failed"
        assert obs.error == "context_limit_exceeded"
        assert obs.raw_reason == "error"

    @staticmethod
    def test_error_text_sanitized():
        """Test error text sanitized."""
        real, oracle = _one_case(
            tokens=(5,), boundary=(9,), expected="x", gen_kind="error_text", cap=4
        )
        _assert_reports_equal(real, oracle)
        (obs,) = real.observations
        assert obs.status == "failed"
        assert obs.error == "backend_reported_error"
        assert _PROBE_SECRET not in repr(asdict(obs))

    @staticmethod
    def test_error_none_sanitized():
        """Test error none sanitized."""
        real, oracle = _one_case(
            tokens=(5,), boundary=(9,), expected="x", gen_kind="error_none", cap=4
        )
        _assert_reports_equal(real, oracle)
        (obs,) = real.observations
        assert obs.error == "backend_reported_error"

    def test_failure_taxonomy(self):
        """Test failure taxonomy."""
        for encode_kind, gen_kind, want_error in [
            ("raise", "limit", "encode_failed"),
            ("invalid_list", "limit", "invalid_input_ids"),
            ("invalid_neg", "limit", "invalid_input_ids"),
            ("invalid_bool", "limit", "invalid_input_ids"),
            ("ok", "raise", "generate_failed"),
            ("ok", "bad_too_long", "invalid_generation"),
            ("ok", "bad_limit_short", "invalid_generation"),
            ("ok", "bad_reason", "invalid_generation"),
            ("ok", "bad_error_type", "invalid_generation"),
            ("ok", "bad_error_on_limit", "invalid_generation"),
            ("ok", "bad_not_generation", "invalid_generation"),
        ]:
            with self.subTest(encode_kind=encode_kind, gen_kind=gen_kind, want_error=want_error):
                tokens = (10, 11, 12, 13)
                real, oracle = _one_case(
                    tokens=tokens,
                    boundary=(9,),
                    expected="x",
                    gen_kind=gen_kind,
                    cap=4,
                    encode_kind=encode_kind,
                )
                _assert_reports_equal(real, oracle)
                (obs,) = real.observations
                assert obs.status == "failed"
                assert obs.error == want_error
                assert obs.exact_match is False

    @staticmethod
    def test_render_failed():
        """Test render failed."""
        fixture = Fixture(
            id="renderfail",
            messages=(Message(role="user", content="q"), Message(role="user", content="q2")),
            expected="x",
            group="g",
        )
        real, oracle, _, _ = _run_pair([fixture], [CaseScript()], (1,), 4, 7)
        _assert_reports_equal(real, oracle)
        (obs,) = real.observations
        assert obs.status == "failed"
        assert obs.error == "render_failed"
        assert obs.input_ids == ()
        assert obs.output_ids == ()
        assert obs.raw_reason is None

    @staticmethod
    def test_generate_sets_event_mid_run():
        """Test generate sets event mid run."""
        real, oracle = _one_case(
            tokens=(10, 11), boundary=(9,), expected="kl", cap=2, set_event_at="generate"
        )
        _assert_reports_equal(real, oracle)
        (obs,) = real.observations
        assert obs.status == "cancelled"
        assert obs.raw_reason == "limit"
        assert obs.output_ids == (10, 11)
        assert obs.answer == ""
        assert obs.exact_match is False
        assert obs.error is None

    @staticmethod
    def test_encode_sets_event_keeps_input_ids():
        """Test encode sets event keeps input ids."""
        real, oracle = _one_case(
            tokens=(10, 11), boundary=(9,), expected="kl", cap=2, set_event_at="encode"
        )
        _assert_reports_equal(real, oracle)
        (obs,) = real.observations
        assert obs.status == "cancelled"
        assert obs.input_ids == (1000,)
        assert obs.output_ids == ()
        assert obs.answer == ""
        assert obs.raw_reason is None
        assert obs.error is None
        assert obs.boundary_found is False

    @staticmethod
    def test_decode_diag_event_then_loop_top_cancel():
        """Test decode diag event then loop top cancel."""
        fixtures = [
            Fixture(
                id="c0", messages=(Message(role="user", content="q"),), expected="kl", group="g"
            ),
            Fixture(
                id="c1", messages=(Message(role="user", content="q"),), expected="kl", group="g"
            ),
        ]
        scripts = [
            CaseScript(gen_kind="limit", tokens=(10, 11), set_event_at="decode_diag"),
            CaseScript(gen_kind="limit", tokens=(10, 11)),
        ]
        real, oracle, _, _ = _run_pair(fixtures, scripts, (9,), 2, 7)
        _assert_reports_equal(real, oracle)
        assert real.status == "cancelled"
        assert len(real.observations) == 1
        assert real.observations[0].status == "completed"
        assert real.observations[0].exact_match is True

    @staticmethod
    def test_request_seed_derivation():
        """Test request seed derivation."""
        fixtures = [
            Fixture(
                id=f"case-{i}☃",
                messages=(Message(role="user", content="q"),),
                expected="e",
                group="g",
            )
            for i in range(3)
        ]
        scripts = [CaseScript(gen_kind="limit", tokens=(10, 11)) for _ in range(3)]
        real, oracle, backend_a, _ = _run_pair(fixtures, scripts, (9,), 2, 12345)
        _assert_reports_equal(real, oracle)
        assert backend_a.seeds == [(i, _o_request_seed(12345, f"case-{i}☃")) for i in range(3)]
        assert real.status == "completed"

    @staticmethod
    def test_determinism():
        """Test determinism."""
        rng = random.Random(31337)  # noqa: S311 -- deterministic test seed
        fixtures, scripts, boundary, cap, seed = _build_trial(rng)
        first, _, _, _ = _run_pair(fixtures, scripts, boundary, cap, seed)
        second, _, _, _ = _run_pair(fixtures, scripts, boundary, cap, seed)
        assert [asdict(o) for o in first.observations] == [asdict(o) for o in second.observations]
        assert (first.status, first.planned_count) == (second.status, second.planned_count)

    @staticmethod
    def test_no_secret_leak():
        """Test no secret leak."""
        scenarios = [
            {"gen_kind": "error_text"},
            {"gen_kind": "raise"},
            {"encode_kind": "raise"},
            {"tokens": (10, _FAIL_TOKEN), "gen_kind": "limit"},
        ]
        for i, kwargs in enumerate(scenarios):
            tokens = kwargs.pop("tokens", (10, 11))
            real, oracle = _one_case(
                tokens=tokens, boundary=(9,), expected="x", cap=2, fixture_id=f"leak-{i}", **kwargs
            )
            _assert_reports_equal(real, oracle)
            for obs in real.observations:
                blob = repr(asdict(obs))
                assert _PROBE_SECRET not in blob
                assert obs.error in _FIXED_ERROR_VOCAB | {None}
                assert type(obs.input_ids) is tuple
                assert all(type(t) is int for t in obs.output_ids)

    @staticmethod
    def test_observation_field_structure():
        """Test observation field structure."""
        tokens = (10, 11, 1)
        real, oracle = _one_case(
            tokens=tokens, boundary=(1,), expected=scripted_decode((10, 11)), cap=3
        )
        _assert_reports_equal(real, oracle)
        (obs,) = real.observations
        assert obs.candidate == "audit-candidate"
        assert obs.fixture == "edge"
        assert obs.input_ids == (1000,)
        assert obs.output_ids == tokens
        assert type(obs.answer) is str
        assert type(obs.exact_match) is bool
        assert type(obs.boundary_found) is bool
        assert obs.boundary_text_match is True
        assert obs.raw_reason == "limit"
        assert obs.error is None
        assert obs.status == "completed"

    def test_validation_parity(self):
        """Test validation parity."""
        for breakage in [
            "candidate",
            "cases_empty",
            "cases_list",
            "cases_bad_member",
            "cases_dup_nfc_id",
            "cap_zero",
            "cap_too_big",
            "cap_str",
            "cap_bool",
            "boundary_empty",
            "boundary_list",
            "boundary_negative",
            "boundary_bool_token",
            "seed_negative",
            "seed_too_big",
            "seed_str",
            "cancel_str",
        ]:
            with self.subTest(breakage=breakage):
                kwargs = _valid_kwargs()
                if breakage == "candidate":
                    kwargs["candidate"] = "not-a-format"
                elif breakage == "cases_empty":
                    kwargs["cases"] = ()
                elif breakage == "cases_list":
                    kwargs["cases"] = list(kwargs["cases"])
                elif breakage == "cases_bad_member":
                    kwargs["cases"] = ("nope",)
                elif breakage == "cases_dup_nfc_id":
                    kwargs["cases"] = _nfc_dup_cases()
                elif breakage == "cap_zero":
                    kwargs["max_new_tokens"] = 0
                elif breakage == "cap_too_big":
                    kwargs["max_new_tokens"] = 4097
                elif breakage == "cap_str":
                    kwargs["max_new_tokens"] = "64"
                elif breakage == "cap_bool":
                    kwargs["max_new_tokens"] = True
                elif breakage == "boundary_empty":
                    kwargs["boundary"] = ()
                elif breakage == "boundary_list":
                    kwargs["boundary"] = [1]
                elif breakage == "boundary_negative":
                    kwargs["boundary"] = (1, -2)
                elif breakage == "boundary_bool_token":
                    kwargs["boundary"] = (1, True)
                elif breakage == "seed_negative":
                    kwargs["seed"] = -1
                elif breakage == "seed_too_big":
                    kwargs["seed"] = 2**64
                elif breakage == "seed_str":
                    kwargs["seed"] = "0"
                elif breakage == "cancel_str":
                    kwargs["cancel"] = "not-an-event"
                else:  # pragma: no cover -- parametrize list is exhaustive
                    raise AssertionError(breakage)
                backend = ScriptBackend(_CANDIDATE, [], [], Event())
                with self.assertRaises(ValueError):
                    evaluate(backend, **kwargs)
                with self.assertRaises(ValueError):
                    oracle_evaluate(backend, **kwargs)


# ---------------------------------------------------------------------------
# Targeted edge cases (also the deterministic mutation-killers).
# ---------------------------------------------------------------------------


def _one_case(
    *,
    tokens: tuple[int, ...],
    boundary: tuple[int, ...],
    expected: str,
    gen_kind: str = "limit",
    cap: int = 4,
    seed: int = 7,
    set_event_at: str | None = None,
    encode_kind: str = "ok",
    fixture_id: str = "edge",
) -> tuple[EvaluationReport, EvaluationReport]:
    """One case."""
    fixture = Fixture(
        id=fixture_id,
        messages=(Message(role="user", content="q"),),
        expected=expected,
        group="g",
    )
    script = CaseScript(
        encode_kind=encode_kind, gen_kind=gen_kind, tokens=tokens, set_event_at=set_event_at
    )
    real, expected_report, _, _ = _run_pair([fixture], [script], boundary, cap, seed)
    return real, expected_report


# ---------------------------------------------------------------------------
# Input-validation parity: both implementations must reject the same inputs.
# ---------------------------------------------------------------------------


def _valid_kwargs() -> dict:
    """Valid kwargs."""
    fixture = Fixture(
        id="v", messages=(Message(role="user", content="q"),), expected="e", group="g"
    )
    return {
        "candidate": _CANDIDATE,
        "cases": (fixture,),
        "boundary": (1,),
        "max_new_tokens": 4,
        "seed": 0,
        "cancel": None,
    }


def _nfc_dup_cases() -> tuple[Fixture, Fixture]:
    """Nfc dup cases."""

    def mk(fid: str) -> Fixture:
        """Mk."""
        return Fixture(
            id=fid, messages=(Message(role="user", content="q"),), expected="e", group="g"
        )

    return (mk("caf\u00e9"), mk("cafe\u0301"))


if __name__ == "__main__":
    unittest.main()
