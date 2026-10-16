"""Cycle-79 adversarial audit: ``core.evaluate`` per-candidate evaluation engine.

Target: ``model_protocol_lab.core.evaluate`` -- the per-fixture generation
loop that cycle 78's audited ``run_experiment`` drives: input validation
(NFC-normalized fixture-ID uniqueness, cap bounds, boundary non-empty, seed
range, cancel type), observation assembly (``exact_match``,
``boundary_found``, truncation/limit reasons), the ``aggregate_eligible``
computation, ``planned_count`` accounting, cancellation checks, and the
fail-closed backend-error mapping (no exception text leaks).

BRANCH MAP (hand-derived from the source, not from running it):

    validation (each raises ValueError):
        candidate is not a Format
        cases is not a nonempty tuple of Fixture
        fixture IDs collide after NFC normalization
        max_new_tokens is not an int in [1, 4096]
        boundary is not a nonempty tuple of valid token ids
        seed is not an unsigned 64-bit int
        cancel is neither None nor a threading.Event

    per-fixture loop (``records`` become the observations):
        event set before the fixture  -> report("cancelled"), no observation
        render raises                 -> abort failed, error "render_failed"
        event set after render        -> abort cancelled (race window; a fake
                                         backend cannot deterministically set
                                         the event during render, so this
                                         path is documented but untestable)
        encode raises                 -> abort failed, error "encode_failed"
        encode returns invalid ids    -> abort failed, error "invalid_input_ids"
        event set after encode        -> terminate cancelled (input_ids kept)
        generate raises               -> terminate failed "generate_failed"
        generate returns invalid      -> terminate failed "invalid_generation"
        reason == "error"             -> terminate failed; error is the
                                         allowlisted code or the sanitized
                                         "backend_reported_error"
        reason == "cancelled" or event set -> terminate cancelled
        decode raises / returns non-str    -> terminate failed "decode_failed"
        event set after decode        -> terminate cancelled (answer kept)
        otherwise                     -> record completed;
                                         exact_match = (answer == expected)
                                         boundary_found = (cut is not None)

    report: planned_count == len(cases) on every path; aggregate_eligible ==
            complete == (status == "completed" and attempted == planned).

Method:
  1. Scripted fake backends (chr-codec, deterministic, no models/network)
     control generated ids per fixture: exact answer, wrong answer, boundary
     present/absent, truncation at the cap, backend raising at the
     encode/generate/decode stages, invalid generations, and cancellation.
  2. Observation-field fidelity: every expected ``exact_match`` /
     ``boundary_found`` / stop-reason / error-code value is hand-derived from
     the script definition. The tests never re-run the source's logic to
     build expectations (``Format.render`` is used only to build the
     expected *input* ids; the chr-codec is the test's own).
  3. Validation branches: each ValueError case is hit, plus edge acceptances
     (cap 1/4096, seed 2**64-1, boundary (0,)).
  4. aggregate_eligible/planned_count semantics across completed, cancelled
     (pre-loop, mid-loop, post-decode) and failed reports.
  5. Minimal pairs: one-id flips moving exactly one of exact_match /
     boundary_found; empty generations; boundary at generation start/end;
     cancellation mid-loop; exception-text leak scans (raw backend text must
     not appear anywhere in the report).
  6. Mutation gate (run separately): 6 mutants in /tmp copies.

All backends/tokenizers here are synthetic and scripted; no model weights,
no network, no downloads.
"""

import re
import unittest
from threading import Event
from typing import Any, cast

from model_protocol_lab.core import (
    EvaluationReport,
    Fixture,
    Format,
    Generation,
    Message,
    evaluate,
)

BOUNDARY = (999,)  # chr(999) == "ϧ": the chr-codec never emits it from ASCII text
CAP = 24  # max_new_tokens for most tests
PAD = 32  # chr(32) == " ": padding never collides with the boundary id

# Fixed-vocabulary error codes the source passes through verbatim
# (hand-derived from _BACKEND_ERROR_CODES in core.py).
ALLOWLISTED_CODES = frozenset(
    {
        "invalid_or_empty_input",
        "invalid_output_cap",
        "invalid_seed",
        "invalid_cancellation_event",
        "context_limit_exceeded",
    }
)
# Every error string evaluate() itself may ever record.
FIXED_ERRORS = ALLOWLISTED_CODES | {
    "render_failed",
    "encode_failed",
    "invalid_input_ids",
    "generate_failed",
    "invalid_generation",
    "decode_failed",
    "backend_reported_error",
}


def _ids(text: str) -> tuple[int, ...]:
    return tuple(ord(c) for c in text)


def _emit_ids(answer_text: str, pos: str, cap: int) -> tuple[int, ...]:
    """Build a cap-length "limit" output: answer ids + boundary + space pad.

    pos: "after_answer" (boundary right after the answer, then pad),
         "end" (boundary is the final id; needs len(answer_text) == cap - 1),
         "start" (boundary first; answer decodes to answer_text),
         "none" (no boundary at all).
    """
    out = [ord(c) for c in answer_text]
    if pos == "start":
        out = [BOUNDARY[0]] + out
    elif pos in ("after_answer", "end"):
        out = out + [BOUNDARY[0]]
    if pos == "end":
        assert len(out) == cap, "end position needs answer length == cap - 1"
        return tuple(out)
    pad = cap - len(out)
    assert pad >= 0, "answer too long for cap"
    return tuple(out + [PAD] * pad)


def _gen(token_ids: tuple[int, ...], reason: Any, error: Any = None) -> Generation:
    # Generation is a frozen dataclass: construct it directly (the constructor
    # performs no validation, so invalid payloads can be built on purpose).
    return Generation(token_ids=token_ids, reason=reason, error=error)


def make_candidate(name: str = "cand") -> Format:
    return Format(
        name=name,
        bos="<bos>",
        user_prefix="U:",
        assistant_prefix="A:",
        turn_end=";",
    )


def make_fixture(fid: str, expected: str, content: str | None = None) -> Fixture:
    return Fixture(
        id=fid,
        messages=(Message(role="user", content=content or f"probe <fx:{fid}>"),),
        expected=expected,
        group=f"c79-{fid}",
    )


def make_render_failure_fixture(fid: str) -> Fixture:
    # render() requires a final user turn; this fixture fails it.
    return Fixture(
        id=fid,
        messages=(
            Message(role="user", content="q"),
            Message(role="assistant", content="a"),
        ),
        expected="whatever",
        group=f"c79-{fid}",
    )


class ScriptedBackend:
    """Deterministic fake backend; generate() is a pure function of the script.

    encode/decode are a chr-codec. The rendered prompt carries ``<fx:ID>``
    (from the fixture message), so every backend call routes to the scripted
    outcome for that fixture. Outcome vocabulary (values of the script dict):

      ("emit", (answer_text, pos)) -> reason="limit", cap-length ids
      ("raw", token_ids)           -> reason="limit", ids used verbatim
      ("error", value)             -> reason="error", token_ids=()
      ("cancelled", token_ids)     -> reason="cancelled"
      ("raise", stage, secret)     -> stage in {"encode","generate","decode"}
                                      raises RuntimeError(secret)
      ("encode_bad",)              -> encode returns ids with a negative entry
      ("encode_set_event",)        -> encode sets the shared event, then works
      ("decode_set_event", (answer_text, pos)) -> decode sets the shared event,
                                      then decodes normally (answer is kept)
      ("decode_non_str",)          -> decode returns a non-str
      ("gen_invalid", sub)         -> sub in {"too_long","bad_reason",
                                      "error_non_str","limit_with_error"}
      ("set_event", inner)         -> set the cancel event, then behave as inner
    """

    fingerprint: str = "cycle79-audit-scripted"

    def __init__(
        self,
        script: dict[str, tuple[Any, ...]],
        *,
        raise_on_boundary_decode: bool = False,
        event: Event | None = None,
    ) -> None:
        self.script = dict(script)
        self.raise_on_boundary_decode = raise_on_boundary_decode
        self.event = event
        self._current: str | None = None

    @staticmethod
    def _fixture_of_text(text: str) -> str | None:
        match = re.search(r"<fx:([^>]+)>", text)
        return match.group(1) if match else None

    def _outcome(self, fid: str | None) -> tuple[Any, ...]:
        return self.script.get(fid or "", ("emit", ("?", "after_answer")))

    def encode(self, text: str) -> tuple[int, ...]:
        outcome = self._outcome(self._fixture_of_text(text))
        if outcome[0] == "raise" and outcome[1] == "encode":
            raise RuntimeError(outcome[2])
        if outcome[0] == "encode_set_event":
            assert self.event is not None
            self.event.set()
        if outcome[0] == "encode_bad":
            return (0, -1)
        return _ids(text)

    def decode(self, token_ids: tuple[int, ...]) -> str:
        if self.raise_on_boundary_decode and BOUNDARY[0] in token_ids:
            raise RuntimeError("diag SECRET-BOUNDARY-DECODE")
        outcome = self._outcome(self._current)
        if outcome[0] == "raise" and outcome[1] == "decode":
            raise RuntimeError(outcome[2])
        if outcome[0] == "decode_set_event":
            assert self.event is not None
            self.event.set()
        if outcome[0] == "decode_non_str":
            return cast(str, 12345)
        return "".join(chr(i) for i in token_ids)

    def generate(
        self, input_ids: tuple[int, ...], *, max_new_tokens: int, seed: int, cancel: Event
    ) -> Generation:
        fid = self._fixture_of_text("".join(chr(i) for i in input_ids))
        self._current = fid
        outcome = self._outcome(fid)
        if outcome[0] == "set_event":
            cancel.set()
            outcome = outcome[1]
        kind = outcome[0]
        if kind == "emit":
            answer_text, pos = outcome[1]
            return _gen(_emit_ids(answer_text, pos, max_new_tokens), "limit")
        if kind == "decode_set_event":
            # generate succeeds; decode() sets the event mid-decode.
            answer_text, pos = outcome[1]
            return _gen(_emit_ids(answer_text, pos, max_new_tokens), "limit")
        if kind == "decode_non_str":
            # generate succeeds; decode() returns a non-str.
            return _gen(_emit_ids("?", "after_answer", max_new_tokens), "limit")
        if kind == "raw":
            return _gen(outcome[1], "limit")
        if kind == "error":
            return _gen((), "error", outcome[1])
        if kind == "cancelled":
            return _gen(outcome[1], "cancelled")
        if kind == "raise":
            if outcome[1] == "decode":
                # generate succeeds; the raise happens in decode().
                return _gen(_emit_ids("?", "after_answer", max_new_tokens), "limit")
            raise RuntimeError(outcome[2])
        if kind == "gen_invalid":
            sub = outcome[1]
            if sub == "too_long":
                return _gen(tuple(range(max_new_tokens + 3)), "limit")
            if sub == "bad_reason":
                return _gen((1,), "stop")
            if sub == "error_non_str":
                return _gen((), "error", 123)
            if sub == "limit_with_error":
                return _gen((5,) * max_new_tokens, "limit", "x")
            raise AssertionError(f"bad gen_invalid subkind: {sub}")
        raise AssertionError(f"bad outcome kind: {kind}")


class EvaluateHarness(unittest.TestCase):
    def setUp(self) -> None:
        self.candidate = make_candidate()

    def run_report(
        self,
        script: dict[str, tuple[Any, ...]],
        cases: tuple[Fixture, ...],
        *,
        cap: int = CAP,
        boundary: tuple[int, ...] = BOUNDARY,
        seed: int = 0,
        cancel: Event | None = None,
        backend_kw: dict[str, Any] | None = None,
    ) -> EvaluationReport:
        backend = ScriptedBackend(script, **(backend_kw or {}))
        return evaluate(
            backend,
            self.candidate,
            cases,
            boundary=boundary,
            max_new_tokens=cap,
            seed=seed,
            cancel=cancel,
        )

    def expected_input_ids(self, case: Fixture) -> tuple[int, ...]:
        return _ids(self.candidate.render(case.messages))


class EvaluateValidation(EvaluateHarness):
    """Every validation branch raises ValueError."""

    def test_candidate_must_be_format(self) -> None:
        cases = (make_fixture("v0", "A"),)
        bad_values: list[Any] = ["nope", None, 42, make_fixture("v1", "B")]
        for bad in bad_values:
            with self.subTest(candidate=bad), self.assertRaises(ValueError):
                evaluate(ScriptedBackend({}), bad, cases, boundary=BOUNDARY)

    def test_cases_must_be_nonempty_tuple_of_fixtures(self) -> None:
        good = make_fixture("v0", "A")
        bad_cases: list[Any] = [
            [good],
            (),
            None,
            "nope",
            (good, "nope"),
            (good, None),
        ]
        for bad in bad_cases:
            with self.subTest(cases=bad), self.assertRaises(ValueError):
                evaluate(ScriptedBackend({}), self.candidate, bad, boundary=BOUNDARY)

    def test_duplicate_fixture_ids_rejected(self) -> None:
        cases = (make_fixture("dup", "A"), make_fixture("dup", "B"))
        with self.assertRaises(ValueError):
            evaluate(ScriptedBackend({}), self.candidate, cases, boundary=BOUNDARY)

    def test_nfc_colliding_fixture_ids_rejected(self) -> None:
        # "caf\u00e9" (NFC) vs "cafe\u0301" (NFD): canonically equivalent.
        cases = (
            make_fixture("caf\u00e9", "A"),
            make_fixture("cafe\u0301", "B"),
        )
        with self.assertRaises(ValueError):
            evaluate(ScriptedBackend({}), self.candidate, cases, boundary=BOUNDARY)

    def test_cap_bounds(self) -> None:
        cases = (make_fixture("v0", "A"),)
        bad_values: list[Any] = [0, -1, 4097, 5000, "64", 64.0, True, None]
        for bad_cap in bad_values:
            with self.subTest(cap=bad_cap), self.assertRaises(ValueError):
                evaluate(
                    ScriptedBackend({}),
                    self.candidate,
                    cases,
                    boundary=BOUNDARY,
                    max_new_tokens=bad_cap,
                )

    def test_boundary_must_be_nonempty_valid_token_ids(self) -> None:
        cases = (make_fixture("v0", "A"),)
        bad_values: list[Any] = [(), [], (1.5,), (-1,), (True,), "999", (999, 1.5), None]
        for bad in bad_values:
            with self.subTest(boundary=bad), self.assertRaises(ValueError):
                evaluate(ScriptedBackend({}), self.candidate, cases, boundary=bad)

    def test_seed_range(self) -> None:
        cases = (make_fixture("v0", "A"),)
        bad_values: list[Any] = [-1, 2**64, "0", 2.0, None, True]
        for bad_seed in bad_values:
            with self.subTest(seed=bad_seed), self.assertRaises(ValueError):
                evaluate(
                    ScriptedBackend({}),
                    self.candidate,
                    cases,
                    boundary=BOUNDARY,
                    seed=bad_seed,
                )

    def test_cancel_type(self) -> None:
        cases = (make_fixture("v0", "A"),)
        bad_values: list[Any] = ["event", 123, object()]
        for bad in bad_values:
            with self.subTest(cancel=bad), self.assertRaises(ValueError):
                evaluate(
                    ScriptedBackend({}),
                    self.candidate,
                    cases,
                    boundary=BOUNDARY,
                    cancel=bad,
                )

    def test_edge_values_accepted(self) -> None:
        # cap=1: boundary alone fills the cap; answer is empty == expected.
        report = self.run_report({"e1": ("emit", ("", "start"))}, (make_fixture("e1", ""),), cap=1)
        self.assertEqual(report.status, "completed")
        obs = report.observations[0]
        self.assertTrue(obs.exact_match)
        self.assertTrue(obs.boundary_found)
        # cap=4096 and seed=2**64-1 are accepted and run.
        report = self.run_report(
            {"e2": ("emit", ("OK", "after_answer"))},
            (make_fixture("e2", "OK"),),
            cap=4096,
            seed=2**64 - 1,
        )
        self.assertEqual(report.status, "completed")
        self.assertTrue(report.observations[0].exact_match)
        # boundary=(0,) is a valid boundary id.
        report = self.run_report(
            {"e3": ("raw", (65, 0) + (PAD,) * (CAP - 2))},
            (make_fixture("e3", "A"),),
            boundary=(0,),
        )
        obs = report.observations[0]
        self.assertTrue(obs.exact_match)
        self.assertTrue(obs.boundary_found)


class EvaluateObservationFidelity(EvaluateHarness):
    """Hand-derived expected observation fields per scripted outcome."""

    def test_pass_full_fidelity(self) -> None:
        case = make_fixture("f0", "OK-1")
        report = self.run_report({"f0": ("emit", ("OK-1", "after_answer"))}, (case,))
        self.assertEqual(report.status, "completed")
        self.assertEqual(report.planned_count, 1)
        (obs,) = report.observations
        self.assertEqual(obs.candidate, "cand")
        self.assertEqual(obs.fixture, "f0")
        self.assertEqual(obs.input_ids, self.expected_input_ids(case))
        self.assertEqual(obs.output_ids, _ids("OK-1") + BOUNDARY + (PAD,) * 19)
        self.assertEqual(obs.answer, "OK-1")
        self.assertTrue(obs.exact_match)
        self.assertTrue(obs.boundary_found)
        self.assertTrue(obs.boundary_text_match)
        self.assertEqual(obs.raw_reason, "limit")
        self.assertIsNone(obs.error)
        self.assertEqual(obs.status, "completed")
        self.assertTrue(report.aggregate_eligible)

    def test_wrong_answer(self) -> None:
        case = make_fixture("f0", "OK-1")
        report = self.run_report({"f0": ("emit", ("NOPE", "after_answer"))}, (case,))
        (obs,) = report.observations
        self.assertEqual(obs.answer, "NOPE")
        self.assertFalse(obs.exact_match)
        self.assertTrue(obs.boundary_found)
        self.assertEqual(obs.status, "completed")
        self.assertEqual(obs.raw_reason, "limit")
        self.assertTrue(report.aggregate_eligible)

    def test_no_boundary_exact_match(self) -> None:
        # No boundary anywhere: cut is None, so the whole cap-length output is
        # the answer; the fixture's expected is exactly cap chars long.
        case = make_fixture("g0", "A" * CAP)
        report = self.run_report({"g0": ("emit", ("A" * CAP, "none"))}, (case,))
        (obs,) = report.observations
        self.assertEqual(obs.answer, "A" * CAP)
        self.assertTrue(obs.exact_match)
        self.assertFalse(obs.boundary_found)
        self.assertFalse(obs.boundary_text_match)
        self.assertEqual(obs.status, "completed")

    def test_wrong_answer_and_no_boundary(self) -> None:
        case = make_fixture("g0", "A" * CAP)
        report = self.run_report({"g0": ("emit", ("B" * CAP, "none"))}, (case,))
        (obs,) = report.observations
        self.assertFalse(obs.exact_match)
        self.assertFalse(obs.boundary_found)
        self.assertEqual(obs.status, "completed")

    def test_truncation_at_cap(self) -> None:
        # Expected answer is longer than the cap; the boundary lands on the
        # final id, so the decoded answer is a strict prefix of expected.
        case = make_fixture("t0", "ABCDEFGHIJ")
        report = self.run_report({"t0": ("emit", ("ABCDEFG", "end"))}, (case,), cap=8)
        (obs,) = report.observations
        self.assertEqual(len(obs.output_ids), 8)
        self.assertEqual(obs.answer, "ABCDEFG")
        self.assertFalse(obs.exact_match)
        self.assertTrue(obs.boundary_found)
        self.assertEqual(obs.raw_reason, "limit")

    def test_boundary_at_generation_start(self) -> None:
        # cut == 0: the answer is empty, which matches the empty expected.
        case = make_fixture("s0", "")
        report = self.run_report({"s0": ("emit", ("", "start"))}, (case,))
        (obs,) = report.observations
        self.assertEqual(obs.output_ids, BOUNDARY + (PAD,) * (CAP - 1))
        self.assertEqual(obs.answer, "")
        self.assertTrue(obs.exact_match)
        self.assertTrue(obs.boundary_found)
        self.assertTrue(obs.boundary_text_match)

    def test_boundary_at_generation_end(self) -> None:
        case = make_fixture("s1", "E" * (CAP - 1))
        report = self.run_report({"s1": ("emit", ("E" * (CAP - 1), "end"))}, (case,))
        (obs,) = report.observations
        self.assertEqual(obs.answer, "E" * (CAP - 1))
        self.assertTrue(obs.exact_match)
        self.assertTrue(obs.boundary_found)
        self.assertTrue(obs.boundary_text_match)

    def test_empty_cancelled_generation(self) -> None:
        case = make_fixture("c0", "X")
        report = self.run_report({"c0": ("cancelled", ())}, (case,))
        (obs,) = report.observations
        self.assertEqual(obs.status, "cancelled")
        self.assertEqual(obs.output_ids, ())
        self.assertEqual(obs.answer, "")
        self.assertFalse(obs.exact_match)
        self.assertFalse(obs.boundary_found)
        self.assertIsNone(obs.boundary_text_match)
        self.assertEqual(obs.raw_reason, "cancelled")
        self.assertIsNone(obs.error)
        self.assertEqual(report.status, "cancelled")
        self.assertFalse(report.aggregate_eligible)

    def test_error_allowlisted_code_passes_through(self) -> None:
        case = make_fixture("e0", "X")
        report = self.run_report({"e0": ("error", "context_limit_exceeded")}, (case,))
        (obs,) = report.observations
        self.assertEqual(obs.status, "failed")
        self.assertEqual(obs.output_ids, ())
        self.assertEqual(obs.raw_reason, "error")
        self.assertEqual(obs.error, "context_limit_exceeded")
        self.assertFalse(obs.exact_match)
        self.assertFalse(obs.boundary_found)
        self.assertEqual(report.status, "failed")
        self.assertFalse(report.aggregate_eligible)

    def test_error_arbitrary_text_sanitized(self) -> None:
        secret = "ARBITRARY-SECRET-leak-1"
        case = make_fixture("e0", "X")
        report = self.run_report({"e0": ("error", secret)}, (case,))
        (obs,) = report.observations
        self.assertEqual(obs.status, "failed")
        self.assertEqual(obs.error, "backend_reported_error")
        self.assertNotIn(secret, repr(report))

    def test_error_none_and_empty_string_sanitized(self) -> None:
        for value in (None, ""):
            with self.subTest(error=value):
                case = make_fixture("e0", "X")
                report = self.run_report({"e0": ("error", value)}, (case,))
                (obs,) = report.observations
                self.assertEqual(obs.error, "backend_reported_error")

    def test_raise_at_encode(self) -> None:
        secret = "ENC-SECRET-9"
        case = make_fixture("r0", "X")
        report = self.run_report({"r0": ("raise", "encode", secret)}, (case,))
        (obs,) = report.observations
        self.assertEqual(obs.status, "failed")
        self.assertEqual(obs.error, "encode_failed")
        self.assertEqual(obs.input_ids, ())
        self.assertEqual(obs.output_ids, ())
        self.assertEqual(obs.answer, "")
        self.assertIsNone(obs.raw_reason)
        self.assertFalse(obs.exact_match)
        self.assertNotIn(secret, repr(report))

    def test_raise_at_generate(self) -> None:
        secret = "GEN-SECRET-9"
        case = make_fixture("r0", "X")
        report = self.run_report({"r0": ("raise", "generate", secret)}, (case,))
        (obs,) = report.observations
        self.assertEqual(obs.status, "failed")
        self.assertEqual(obs.error, "generate_failed")
        # encode succeeded, so the prompt evidence is retained.
        self.assertEqual(obs.input_ids, self.expected_input_ids(case))
        self.assertEqual(obs.output_ids, ())
        self.assertIsNone(obs.raw_reason)
        self.assertNotIn(secret, repr(report))

    def test_raise_at_decode(self) -> None:
        secret = "DEC-SECRET-9"
        case = make_fixture("r0", "OK-1")
        report = self.run_report({"r0": ("raise", "decode", secret)}, (case,))
        (obs,) = report.observations
        self.assertEqual(obs.status, "failed")
        self.assertEqual(obs.error, "decode_failed")
        # generate() succeeded with the default payload before decode raised.
        self.assertEqual(obs.output_ids, _ids("?") + BOUNDARY + (PAD,) * (CAP - 2))
        self.assertEqual(obs.raw_reason, "limit")
        # cut was computed before decode ran: boundary_found is True even on
        # this failed observation; the answer was never decoded.
        self.assertTrue(obs.boundary_found)
        self.assertEqual(obs.answer, "")
        self.assertFalse(obs.exact_match)
        self.assertNotIn(secret, repr(report))

    def test_encode_bad_ids(self) -> None:
        case = make_fixture("b0", "X")
        report = self.run_report({"b0": ("encode_bad",)}, (case,))
        (obs,) = report.observations
        self.assertEqual(obs.status, "failed")
        self.assertEqual(obs.error, "invalid_input_ids")
        self.assertEqual(obs.input_ids, ())
        self.assertEqual(obs.output_ids, ())
        self.assertIsNone(obs.raw_reason)

    def test_invalid_generations(self) -> None:
        for sub in ("too_long", "bad_reason", "error_non_str", "limit_with_error"):
            with self.subTest(sub=sub):
                case = make_fixture("i0", "X")
                report = self.run_report({"i0": ("gen_invalid", sub)}, (case,))
                (obs,) = report.observations
                self.assertEqual(obs.status, "failed")
                self.assertEqual(obs.error, "invalid_generation")
                # Rejected before output_ids/raw_reason were assigned.
                self.assertEqual(obs.input_ids, self.expected_input_ids(case))
                self.assertEqual(obs.output_ids, ())
                self.assertIsNone(obs.raw_reason)

    def test_decode_non_str(self) -> None:
        case = make_fixture("d0", "X")
        report = self.run_report({"d0": ("decode_non_str",)}, (case,))
        (obs,) = report.observations
        self.assertEqual(obs.status, "failed")
        self.assertEqual(obs.error, "decode_failed")

    def test_render_failure(self) -> None:
        case = make_render_failure_fixture("rf0")
        report = self.run_report({}, (case,))
        (obs,) = report.observations
        self.assertEqual(obs.status, "failed")
        self.assertEqual(obs.error, "render_failed")
        self.assertEqual(obs.input_ids, ())
        self.assertEqual(obs.output_ids, ())
        self.assertEqual(obs.answer, "")
        self.assertIsNone(obs.raw_reason)
        self.assertFalse(obs.exact_match)

    def test_boundary_text_match_none_when_diagnostic_fails(self) -> None:
        # decode() raises on the boundary ids: the diagnostic degrades to
        # None instead of failing the run.
        case = make_fixture("n0", "ABC" + " " * (CAP - 3))
        report = self.run_report(
            {"n0": ("emit", ("ABC" + " " * (CAP - 3), "none"))},
            (case,),
            backend_kw={"raise_on_boundary_decode": True},
        )
        (obs,) = report.observations
        self.assertEqual(obs.status, "completed")
        self.assertTrue(obs.exact_match)
        self.assertFalse(obs.boundary_found)
        self.assertIsNone(obs.boundary_text_match)

    def test_error_observation_records_cut(self) -> None:
        # reason="error": cut is computed before the error branch, so
        # boundary_found follows cut even on a failed observation.
        case = make_fixture("e0", "X")
        report = self.run_report({"e0": ("error", "context_limit_exceeded")}, (case,))
        (obs,) = report.observations
        self.assertFalse(obs.boundary_found)
        self.assertIsNone(obs.boundary_text_match)


class EvaluateMinimalPairs(EvaluateHarness):
    """One-id flips move exactly one observation field."""

    def test_one_id_flip_moves_exact_match_only(self) -> None:
        case = make_fixture("p0", "AB")
        run1 = self.run_report({"p0": ("emit", ("AB", "after_answer"))}, (case,))
        run2 = self.run_report({"p0": ("emit", ("AC", "after_answer"))}, (case,))
        o1 = run1.observations[0]
        o2 = run2.observations[0]
        diffs = sum(a != b for a, b in zip(o1.output_ids, o2.output_ids, strict=True))
        self.assertEqual(diffs, 1)
        self.assertEqual(len(o1.output_ids), len(o2.output_ids))
        self.assertTrue(o1.exact_match)
        self.assertFalse(o2.exact_match)
        self.assertTrue(o1.boundary_found)
        self.assertTrue(o2.boundary_found)

    def test_one_id_flip_moves_boundary_found_only(self) -> None:
        case = make_fixture("q0", "AB")
        flipped = _ids("ZZ") + (PAD,) + (PAD,) * (CAP - 3)
        run1 = self.run_report({"q0": ("emit", ("ZZ", "after_answer"))}, (case,))
        run2 = self.run_report({"q0": ("raw", flipped)}, (case,))
        o1 = run1.observations[0]
        o2 = run2.observations[0]
        diffs = sum(a != b for a, b in zip(o1.output_ids, o2.output_ids, strict=True))
        self.assertEqual(diffs, 1)  # the single boundary id 999 -> 32
        self.assertEqual(len(o1.output_ids), len(o2.output_ids))
        self.assertFalse(o1.exact_match)
        self.assertFalse(o2.exact_match)
        self.assertTrue(o1.boundary_found)
        self.assertFalse(o2.boundary_found)

    def test_cancellation_preset_before_loop(self) -> None:
        cases = tuple(make_fixture(f"k{j}", "A") for j in range(3))
        event = Event()
        event.set()
        report = self.run_report({}, cases, cancel=event)
        self.assertEqual(report.status, "cancelled")
        self.assertEqual(report.observations, ())
        self.assertEqual(report.planned_count, 3)
        self.assertEqual(report.attempted_count, 0)
        self.assertFalse(report.aggregate_eligible)

    def test_cancellation_mid_loop(self) -> None:
        cases = tuple(make_fixture(f"m{j}", f"A{j}") for j in range(3))
        script = {
            "m0": ("emit", ("A0", "after_answer")),
            "m1": ("set_event", ("emit", ("A1", "after_answer"))),
            "m2": ("emit", ("A2", "after_answer")),
        }
        event = Event()
        report = self.run_report(script, cases, cancel=event)
        self.assertEqual(report.status, "cancelled")
        self.assertEqual(report.planned_count, 3)
        self.assertEqual(report.attempted_count, 2)
        first, second = report.observations
        self.assertEqual(first.status, "completed")
        self.assertTrue(first.exact_match)
        self.assertEqual(second.status, "cancelled")
        # The cancellation is noticed before decode runs, so the answer is
        # still empty; a cancelled observation never counts as an exact match.
        self.assertEqual(second.answer, "")
        self.assertFalse(second.exact_match)
        self.assertTrue(second.boundary_found)
        self.assertEqual(second.raw_reason, "limit")
        self.assertIsNone(second.error)
        self.assertIsNone(second.boundary_text_match)
        self.assertFalse(report.aggregate_eligible)

    def test_cancellation_set_during_final_generate(self) -> None:
        # All fixtures attempted, but the report is cancelled: not eligible
        # even though attempted_count == planned_count.
        cases = tuple(make_fixture(f"z{j}", f"A{j}") for j in range(2))
        script = {
            "z0": ("emit", ("A0", "after_answer")),
            "z1": ("set_event", ("emit", ("A1", "after_answer"))),
        }
        report = self.run_report(script, cases, cancel=Event())
        self.assertEqual(report.status, "cancelled")
        self.assertEqual(report.planned_count, 2)
        self.assertEqual(report.attempted_count, 2)
        self.assertFalse(report.complete)
        self.assertFalse(report.aggregate_eligible)
        self.assertEqual(report.observations[0].status, "completed")
        self.assertEqual(report.observations[1].status, "cancelled")
        self.assertEqual(report.observations[1].answer, "")

    def test_cancellation_during_decode_keeps_answer(self) -> None:
        # The event is set by decode() itself: the answer was already decoded,
        # so the cancelled observation keeps it -- but exact_match is still
        # False because the status is not "completed".
        case = make_fixture("w0", "A1")
        event = Event()
        report = self.run_report(
            {"w0": ("decode_set_event", ("A1", "after_answer"))},
            (case,),
            cancel=event,
            backend_kw={"event": event},
        )
        (obs,) = report.observations
        self.assertEqual(report.status, "cancelled")
        self.assertEqual(obs.status, "cancelled")
        self.assertEqual(obs.answer, "A1")
        self.assertFalse(obs.exact_match)
        self.assertTrue(obs.boundary_found)
        self.assertEqual(obs.raw_reason, "limit")
        self.assertIsNone(obs.error)
        self.assertIsNone(obs.boundary_text_match)
        self.assertFalse(report.aggregate_eligible)

    def test_cancellation_after_encode(self) -> None:
        # The backend sets the event from encode(); evaluate notices before
        # generate and terminates cancelled with the prompt evidence kept.
        case = make_fixture("ce0", "X")
        event = Event()
        report = self.run_report(
            {"ce0": ("encode_set_event",)},
            (case,),
            cancel=event,
            backend_kw={"event": event},
        )
        (obs,) = report.observations
        self.assertEqual(report.status, "cancelled")
        self.assertEqual(obs.status, "cancelled")
        self.assertEqual(obs.input_ids, self.expected_input_ids(case))
        self.assertEqual(obs.output_ids, ())
        self.assertEqual(obs.answer, "")
        self.assertIsNone(obs.raw_reason)
        self.assertIsNone(obs.error)
        self.assertFalse(obs.exact_match)


class EvaluateAggregateSemantics(EvaluateHarness):
    """Which report statuses are eligible; planned_count accounting."""

    def test_completed_all_pass_eligible(self) -> None:
        cases = tuple(make_fixture(f"a{j}", f"A{j}") for j in range(3))
        script = {f"a{j}": ("emit", (f"A{j}", "after_answer")) for j in range(3)}
        report = self.run_report(script, cases)
        self.assertEqual(report.status, "completed")
        self.assertEqual(report.planned_count, 3)
        self.assertEqual(report.attempted_count, 3)
        self.assertTrue(report.complete)
        self.assertTrue(report.aggregate_eligible)

    def test_failed_report_not_eligible(self) -> None:
        cases = tuple(make_fixture(f"b{j}", f"A{j}") for j in range(3))
        script = {
            "b0": ("emit", ("A0", "after_answer")),
            "b1": ("raise", "generate", "GEN-SECRET-X"),
            "b2": ("emit", ("A2", "after_answer")),
        }
        report = self.run_report(script, cases)
        self.assertEqual(report.status, "failed")
        self.assertEqual(report.planned_count, 3)
        self.assertEqual(report.attempted_count, 2)
        self.assertEqual(report.observations[0].status, "completed")
        self.assertEqual(report.observations[1].status, "failed")
        self.assertEqual(report.observations[1].error, "generate_failed")
        self.assertFalse(report.aggregate_eligible)

    def test_single_fixture_report(self) -> None:
        case = make_fixture("s0", "A")
        report = self.run_report({"s0": ("emit", ("A", "after_answer"))}, (case,))
        self.assertTrue(report.aggregate_eligible)
        self.assertEqual(report.planned_count, 1)

    def test_observations_follow_fixture_order(self) -> None:
        cases = tuple(make_fixture(f"o{j}", f"A{j}") for j in range(4))
        script = {f"o{j}": ("emit", (f"A{j}", "after_answer")) for j in range(4)}
        report = self.run_report(script, cases)
        self.assertEqual([o.fixture for o in report.observations], [f"o{j}" for j in range(4)])
        self.assertTrue(all(o.candidate == "cand" for o in report.observations))

    def test_deterministic_across_runs(self) -> None:
        cases = tuple(make_fixture(f"d{j}", f"A{j}") for j in range(2))
        script = {f"d{j}": ("emit", (f"A{j}", "after_answer")) for j in range(2)}
        first = self.run_report(script, cases, seed=12345)
        second = self.run_report(script, cases, seed=12345)
        self.assertEqual(repr(first), repr(second))


class EvaluateNoLeak(EvaluateHarness):
    """Raw backend text must never appear in a report."""

    def test_exception_text_never_leaks(self) -> None:
        secrets = {
            "r0": ("raise", "encode", "LEAK-ENC-111"),
            "r1": ("raise", "generate", "LEAK-GEN-222"),
            "r2": ("raise", "decode", "LEAK-DEC-333"),
            "r3": ("error", "LEAK-ERR-444"),
        }
        # One fixture per run: evaluate aborts at the first failure.
        for fid, outcome in secrets.items():
            with self.subTest(fixture=fid):
                report = self.run_report({fid: outcome}, (make_fixture(fid, "X"),))
                secret = outcome[2] if outcome[0] == "raise" else outcome[1]
                self.assertNotIn(secret, repr(report))

    def test_recorded_errors_are_fixed_vocabulary(self) -> None:
        script = {
            "v0": ("raise", "encode", "x"),
            "v1": ("raise", "generate", "x"),
            "v2": ("raise", "decode", "x"),
            "v3": ("error", "not-a-code"),
            "v4": ("error", "context_limit_exceeded"),
            "v5": ("gen_invalid", "too_long"),
            "v6": ("encode_bad",),
            "v7": ("decode_non_str",),
        }
        # Each failure aborts the run, so drive one fixture at a time.
        for fid, outcome in script.items():
            with self.subTest(fixture=fid):
                report = self.run_report({fid: outcome}, (make_fixture(fid, "X"),))
                (obs,) = report.observations
                self.assertIn(obs.error, FIXED_ERRORS)


if __name__ == "__main__":
    unittest.main()
