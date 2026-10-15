"""Differential verification of ``model_protocol_lab.core.evaluate`` (cycle 20).

Grounding notes (read before extending this file):

* LOCATION: the assignment brief names ``model_protocol_lab/experiments.py``,
  but ``evaluate`` is *defined* in ``model_protocol_lab/core.py`` (line 651)
  and only re-exported/used by ``experiments.py``. All tests target
  ``model_protocol_lab.core.evaluate``.
* BRIEF VS DOCSTRING: the brief describes a "pass/fail decision per fixture"
  and a "pass/fail rule". The docstring states the opposite: "Exact match and
  boundary presence are separate." ``evaluate`` applies NO composite
  pass/fail gate -- it records independent ``exact_match`` / ``boundary_found``
  fields and a per-attempt ``status``. Candidate selection lives elsewhere.
  ``TestEvaluateDocstringContracts`` pins this explicitly.
* The docstring is terse ("Measure a frozen candidate, retaining
  incomplete-run evidence. ..."). The oracle below is grounded on the
  docstring's stated guarantees (separate exact/boundary, "A limit result has
  exactly the requested cap", "No backend exception text is exported",
  aggregate eligibility) plus a triple-checked reading of the implementation
  for the undocumented-but-deterministic mechanics it pins: the per-case
  ``request_seed`` derivation (sha256 of ``f"{seed}\\0{case.id}"``), the exact
  placement of the six cancel checks, the fixed-vocabulary error-code
  sanitization, and the abort/terminate record shapes. Where behavior goes
  beyond the docstring, the test names say so.
* INDEPENDENCE: the oracle never calls ``Format.render`` (it re-implements the
  rendering from the documented field semantics), never calls
  ``core.first_boundary`` / ``core._valid_token_ids`` / ``core._valid_generation``
  (it re-implements them), and never drives the scripted backend -- it reads
  the same declarative ``Plan`` the backend is built from and simulates the
  documented step order purely. The only shared pieces are the data contracts
  (``Format``/``Fixture``/``Generation``/``Observation`` dataclasses) and the
  test backend's own encode mapping (``ord(c) + 1000``), which is test-side
  scaffolding, not the unit under test.
"""

import contextlib
import dataclasses
import importlib.util
import os
import pathlib
import random
import sys
import tempfile
import unittest
from dataclasses import dataclass, field
from hashlib import sha256
from threading import Event
from unittest import mock

from model_protocol_lab import core
from model_protocol_lab.core import (
    Fixture,
    Format,
    Generation,
    Message,
    Observation,
    evaluate,
)

# ---------------------------------------------------------------------------
# Test-backend codec: a bijective char mapping plus a small merge table.
# ---------------------------------------------------------------------------

_OFFSET = 1000
# Token IDs that decode to fixed text, letting one string have two
# tokenizations (the documented boundary-text false-negative mode):
# (70, 71) decodes to "XY", exactly like the canonical (1088, 1089).
_MERGE = {70: "X", 71: "Y", 72: ""}


def script_encode(text: str) -> tuple:
    """The scripted backend's encode: pure function of the prompt text."""
    return tuple(ord(char) + _OFFSET for char in text)


def pure_decode(token_ids: tuple) -> str:
    """The scripted backend's decode model, without any test hooks."""
    chars = []
    for token in token_ids:
        if token in _MERGE:
            chars.append(_MERGE[token])
        elif type(token) is int and token >= _OFFSET:
            chars.append(chr(token - _OFFSET))
        else:
            raise ValueError(f"undecodable token id {token!r}")
    return "".join(chars)


# Fixed-vocabulary backend error codes. This is an independent copy of the
# allowlist documented in core.py's sanitization-boundary comment (importing
# the private ``core._BACKEND_ERROR_CODES`` would couple the oracle to the
# implementation's internals).
ORACLE_ALLOWLIST = frozenset(
    {
        "invalid_or_empty_input",
        "invalid_output_cap",
        "invalid_seed",
        "invalid_cancellation_event",
        "context_limit_exceeded",
    }
)


# ---------------------------------------------------------------------------
# Independent re-implementations of the documented primitives.
# ---------------------------------------------------------------------------


def oracle_render(fmt: Format, messages: tuple) -> str:
    """Render a prompt from the Format fields only (never ``Format.render``).

    Follows the documented semantics: repeat_bos puts BOS before every
    message including the open reply; per-role turn ends fall back to
    turn_end; open_suffix falls back to assistant_prefix.
    """
    parts = [fmt.bos]
    for index, message in enumerate(messages):
        if index and fmt.repeat_bos:
            parts.append(fmt.bos)
        prefix = fmt.user_prefix if message.role == "user" else fmt.assistant_prefix
        per_role_end = fmt.user_turn_end if message.role == "user" else fmt.assistant_turn_end
        turn_end = per_role_end if per_role_end is not None else fmt.turn_end
        parts.extend((prefix, message.content, turn_end))
    if fmt.repeat_bos:
        parts.append(fmt.bos)
    suffix = fmt.open_suffix if fmt.open_suffix is not None else fmt.assistant_prefix
    parts.append(suffix)
    return "".join(parts)


def oracle_first_boundary(tokens: tuple, stop: tuple):
    """First index where ``stop`` occurs as a contiguous subsequence."""
    if not stop:
        raise ValueError("A nonempty boundary is required")
    for index in range(len(tokens) - len(stop) + 1):
        if tokens[index : index + len(stop)] == stop:
            return index
    return None


def oracle_valid_token_ids(value, *, nonempty: bool = False) -> bool:
    return (
        type(value) is tuple
        and (not nonempty or bool(value))
        and all(type(token) is int and token >= 0 for token in value)
    )


def oracle_valid_generation(result, cap: int) -> bool:
    if type(result) is not Generation or not oracle_valid_token_ids(result.token_ids):
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


def oracle_request_seed(seed: int, case_id: str) -> int:
    material = f"{seed}\0{case_id}".encode("utf-8", errors="surrogatepass")
    return int.from_bytes(sha256(material).digest()[:4], "big")


# ---------------------------------------------------------------------------
# Declarative test plan shared by the scripted backend and the oracle.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Behavior:
    """What ``generate`` does for one fixture's input IDs."""

    kind: str  # "limit" | "cancelled" | "error" | "raise" | "invalid"
    output_ids: tuple = ()
    error: object = None
    invalid_payload: object = None
    set_event_in_generate: bool = False


@dataclass(frozen=True)
class Plan:
    behaviors: dict = field(default_factory=dict)
    encode_mode: str = "ok"  # "ok" | "raise" | "invalid" | "empty"
    encode_sets_event: bool = False
    render_sets_event: bool = False
    render_raises: bool = False
    decode_raises: frozenset = frozenset()
    decode_non_str: frozenset = frozenset()
    decode_sets_event: frozenset = frozenset()


class ScriptedBackend:
    """Deterministic backend: generate() is a pure function of the prompt IDs."""

    fingerprint = "scripted-test-backend"

    def __init__(self, plan: Plan, event: Event):
        self.plan = plan
        self.event = event
        self.encode_log: list = []
        self.decode_log: list = []
        self.generate_log: list = []

    def encode(self, text: str) -> tuple:
        self.encode_log.append(text)
        if self.plan.encode_sets_event and self.event is not None:
            self.event.set()
        mode = self.plan.encode_mode
        if mode == "raise":
            raise RuntimeError("scripted encode explosion")
        if mode == "invalid":
            return (1, -2)
        if mode == "empty":
            return ()
        return script_encode(text)

    def decode(self, token_ids: tuple) -> str:
        self.decode_log.append(token_ids)
        if token_ids in self.plan.decode_sets_event and self.event is not None:
            self.event.set()
        if token_ids in self.plan.decode_raises:
            raise RuntimeError("scripted decode explosion")
        if token_ids in self.plan.decode_non_str:
            return 42
        return pure_decode(token_ids)

    def generate(
        self, input_ids: tuple, *, max_new_tokens: int, seed: int, cancel: Event
    ) -> Generation:
        self.generate_log.append((input_ids, max_new_tokens, seed, cancel))
        behavior = self.plan.behaviors.get(input_ids)
        if behavior is None:
            # No scripted behavior: the implementation's render drifted from
            # the oracle's, so fail loudly through the sanitized path.
            raise RuntimeError("no scripted behavior for these input_ids (render drift?)")
        if behavior.set_event_in_generate:
            cancel.set()
        if behavior.kind == "raise":
            raise RuntimeError("scripted generate explosion: backend detail must not leak")
        if behavior.kind == "invalid":
            return behavior.invalid_payload
        return Generation(token_ids=behavior.output_ids, reason=behavior.kind, error=behavior.error)


# ---------------------------------------------------------------------------
# The independent oracle: pure simulation of the documented step order.
# ---------------------------------------------------------------------------


def oracle_evaluate(plan, candidate, cases, *, boundary, max_new_tokens, seed, event):
    """Simulate ``evaluate`` from the docstring + triple-checked code reading.

    Never touches the backend or the real cancel event; the cancel state is
    tracked in ``vset``. Returns ``(records, status, encode_texts, gen_calls)``.
    """
    records = []
    vset = event.is_set()
    encode_texts = []
    gen_calls = []

    def make(status, case, input_ids, output_ids, answer, cut, raw_reason, error=None, btm=None):
        return Observation(
            candidate=candidate.name,
            fixture=case.id,
            input_ids=input_ids,
            output_ids=output_ids,
            answer=answer,
            exact_match=(status == "completed" and answer == case.expected),
            boundary_found=(cut is not None),
            boundary_text_match=btm,
            raw_reason=raw_reason,
            error=error,
            status=status,
        )

    def abort(status, case, error=None):
        records.append(make(status, case, (), (), "", None, None, error=error))

    report_status = None
    for case in cases:
        if vset:
            report_status = "cancelled"
            break
        prompt = oracle_render(candidate, case.messages)
        if plan.render_raises:
            abort("failed", case, error="render_failed")
            report_status = "failed"
            break
        if plan.render_sets_event:
            vset = True
        if vset:
            abort("cancelled", case)
            report_status = "cancelled"
            break
        encode_texts.append(prompt)
        if plan.encode_sets_event:
            vset = True
        if plan.encode_mode == "raise":
            abort("failed", case, error="encode_failed")
            report_status = "failed"
            break
        if plan.encode_mode == "invalid":
            encoded = (1, -2)
        elif plan.encode_mode == "empty":
            encoded = ()
        else:
            encoded = script_encode(prompt)
        if not oracle_valid_token_ids(encoded):
            abort("failed", case, error="invalid_input_ids")
            report_status = "failed"
            break
        input_ids = encoded
        if vset:
            records.append(make("cancelled", case, input_ids, (), "", None, None))
            report_status = "cancelled"
            break
        request_seed = oracle_request_seed(seed, case.id)
        gen_calls.append((input_ids, max_new_tokens, request_seed))
        behavior = plan.behaviors.get(input_ids)
        if behavior is None:
            records.append(
                make("failed", case, input_ids, (), "", None, None, error="generate_failed")
            )
            report_status = "failed"
            break
        if behavior.set_event_in_generate:
            vset = True
        if behavior.kind == "raise":
            records.append(
                make("failed", case, input_ids, (), "", None, None, error="generate_failed")
            )
            report_status = "failed"
            break
        if behavior.kind == "invalid":
            result = behavior.invalid_payload
        else:
            result = Generation(
                token_ids=behavior.output_ids, reason=behavior.kind, error=behavior.error
            )
        if not oracle_valid_generation(result, max_new_tokens):
            records.append(
                make("failed", case, input_ids, (), "", None, None, error="invalid_generation")
            )
            report_status = "failed"
            break
        output_ids = result.token_ids
        raw_reason = result.reason
        cut = oracle_first_boundary(output_ids, boundary)
        if result.reason == "error":
            code = (
                result.error
                if type(result.error) is str and result.error in ORACLE_ALLOWLIST
                else "backend_reported_error"
            )
            records.append(
                make("failed", case, input_ids, output_ids, "", cut, raw_reason, error=code)
            )
            report_status = "failed"
            break
        if result.reason == "cancelled" or vset:
            records.append(make("cancelled", case, input_ids, output_ids, "", cut, raw_reason))
            report_status = "cancelled"
            break
        answer_ids = output_ids if cut is None else output_ids[:cut]
        if answer_ids in plan.decode_sets_event:
            vset = True
        if answer_ids in plan.decode_raises or answer_ids in plan.decode_non_str:
            records.append(
                make(
                    "failed",
                    case,
                    input_ids,
                    output_ids,
                    "",
                    cut,
                    raw_reason,
                    error="decode_failed",
                )
            )
            report_status = "failed"
            break
        try:
            answer = pure_decode(answer_ids)
        except Exception:  # noqa: BLE001 -- oracle mirrors the decode_failed path
            records.append(
                make(
                    "failed",
                    case,
                    input_ids,
                    output_ids,
                    "",
                    cut,
                    raw_reason,
                    error="decode_failed",
                )
            )
            report_status = "failed"
            break
        if vset:
            records.append(make("cancelled", case, input_ids, output_ids, answer, cut, raw_reason))
            report_status = "cancelled"
            break
        # Diagnostic boundary-text cross-check (never part of any pass rule).
        if boundary in plan.decode_sets_event:
            vset = True
        try:
            boundary_text = pure_decode(boundary)
        except Exception:  # noqa: BLE001 -- diagnostic degrades to None
            boundary_text = None
        btm = None
        if type(boundary_text) is str and boundary_text:
            retained = output_ids if cut is None else output_ids[: cut + len(boundary)]
            if retained in plan.decode_sets_event:
                vset = True
            try:
                retained_text = pure_decode(retained)
            except Exception:  # noqa: BLE001 -- diagnostic degrades to None
                retained_text = None
            if type(retained_text) is str:
                btm = retained_text.endswith(boundary_text)
        records.append(
            make("completed", case, input_ids, output_ids, answer, cut, raw_reason, btm=btm)
        )
    if report_status is None:
        report_status = "cancelled" if vset else "completed"
    return tuple(records), report_status, tuple(encode_texts), tuple(gen_calls)


# ---------------------------------------------------------------------------
# Scenario runner and builders.
# ---------------------------------------------------------------------------


def plan_for(candidate, pairs, **kwargs):
    """Build a Plan whose behaviors are keyed by the oracle-rendered inputs."""
    behaviors = {}
    for case, behavior in pairs:
        behaviors[script_encode(oracle_render(candidate, case.messages))] = behavior
    return Plan(behaviors=behaviors, **kwargs)


def run_scenario(
    t: unittest.TestCase,
    *,
    candidate,
    cases,
    boundary,
    plan,
    max_new_tokens=64,
    seed=0,
    preset_cancel=False,
    evaluate_fn=None,
    event=None,
    render_wrapper=None,
):
    """Run oracle and implementation; compare every record field and call.

    Returns ``(report, backend)``. ``render_wrapper`` is an optional
    ``event -> context manager`` used to patch ``Format.render`` around the
    implementation call only (the oracle never calls it).
    """
    evaluate_fn = evaluate_fn or evaluate
    if event is None:
        event = Event()
    if preset_cancel:
        event.set()
    backend = ScriptedBackend(plan, event)
    exp_records, exp_status, exp_encode_texts, exp_gen_calls = oracle_evaluate(
        plan,
        candidate,
        cases,
        boundary=boundary,
        max_new_tokens=max_new_tokens,
        seed=seed,
        event=event,
    )
    enc_off = len(backend.encode_log)
    gen_off = len(backend.generate_log)
    ctx = render_wrapper(event) if render_wrapper is not None else contextlib.nullcontext()
    with ctx:
        report = evaluate_fn(
            backend,
            candidate,
            cases,
            boundary=boundary,
            max_new_tokens=max_new_tokens,
            seed=seed,
            cancel=event,
        )
    t.assertEqual(report.planned_count, len(cases))
    t.assertEqual(report.status, exp_status)
    t.assertEqual(len(report.observations), len(exp_records))
    for exp, act in zip(exp_records, report.observations, strict=True):
        for f in dataclasses.fields(Observation):
            t.assertEqual(
                getattr(act, f.name),
                getattr(exp, f.name),
                f"fixture {exp.fixture!r} field {f.name}: "
                f"expected {getattr(exp, f.name)!r}, got {getattr(act, f.name)!r}",
            )
    # Backend call contract: exact prompt texts, seeds, cap, and event identity.
    t.assertEqual(backend.encode_log[enc_off:], list(exp_encode_texts))
    new_gens = backend.generate_log[gen_off:]
    t.assertEqual(
        [(i, m, s) for (i, m, s, _c) in new_gens],
        list(exp_gen_calls),
    )
    for _i, _m, _s, c in new_gens:
        t.assertIs(c, event)
    return report, backend


def make_format(**overrides):
    params = {
        "name": "fmt",
        "bos": "<s>",
        "user_prefix": "[U]",
        "assistant_prefix": "[A]",
        "turn_end": "[/]",
        "repeat_bos": False,
        "user_turn_end": None,
        "assistant_turn_end": None,
        "open_suffix": None,
    }
    params.update(overrides)
    return Format(**params)


def make_fixture(fid, messages, expected, group="g"):
    return Fixture(id=fid, messages=messages, expected=expected, group=group)


def msgs(*contents):
    """Build alternating messages starting and ending with user (odd count)."""
    assert len(contents) % 2 == 1, "messages must end with a user turn"
    return tuple(
        Message(role="user" if i % 2 == 0 else "assistant", content=c)
        for i, c in enumerate(contents)
    )


def tid(text):
    return script_encode(text)


def pad_to(ids, cap):
    assert len(ids) <= cap
    return ids + tid("q") * (cap - len(ids))


# ---------------------------------------------------------------------------
# Deterministic differential scenarios: oracle vs implementation, field by field.
# ---------------------------------------------------------------------------


class EvaluateAgainstOracle(unittest.TestCase):
    BOUNDARY = (70, 71)  # decodes to "XY" via the merge table

    def test_clean_match_with_trailing_suffix(self):
        candidate = make_format()
        case = make_fixture("m1", msgs("hello"), "Paris")
        output = tid("Paris") + self.BOUNDARY + tid(" junk")
        plan = plan_for(candidate, [(case, Behavior("limit", output))])
        report, _backend = run_scenario(
            self,
            candidate=candidate,
            cases=(case,),
            boundary=self.BOUNDARY,
            plan=plan,
            max_new_tokens=len(output),
        )
        obs = report.observations[0]
        self.assertEqual(obs.answer, "Paris")  # suffix tokens stay out of the answer
        self.assertEqual(obs.output_ids, output)  # ...but remain in evidence
        self.assertTrue(obs.exact_match and obs.boundary_found)
        self.assertEqual(obs.raw_reason, "limit")
        self.assertTrue(obs.boundary_text_match)
        self.assertTrue(report.aggregate_eligible)

    def test_exact_answer_mismatch(self):
        candidate = make_format()
        case = make_fixture("m2", msgs("hello"), "Paris")
        output = tid("London") + self.BOUNDARY
        plan = plan_for(candidate, [(case, Behavior("limit", output))])
        report, _backend = run_scenario(
            self,
            candidate=candidate,
            cases=(case,),
            boundary=self.BOUNDARY,
            plan=plan,
            max_new_tokens=len(output),
        )
        obs = report.observations[0]
        self.assertEqual(obs.status, "completed")
        self.assertFalse(obs.exact_match)
        self.assertTrue(obs.boundary_found)

    def test_boundary_absent_answer_matches(self):
        candidate = make_format()
        case = make_fixture("m3", msgs("hello"), "Paris")
        output = tid("Paris")
        plan = plan_for(candidate, [(case, Behavior("limit", output))])
        report, _backend = run_scenario(
            self,
            candidate=candidate,
            cases=(case,),
            boundary=self.BOUNDARY,
            plan=plan,
            max_new_tokens=len(output),
        )
        obs = report.observations[0]
        self.assertEqual(obs.status, "completed")
        self.assertTrue(obs.exact_match)
        self.assertFalse(obs.boundary_found)
        self.assertFalse(obs.boundary_text_match)

    def test_boundary_text_match_without_token_id_match(self):
        # The documented false-negative mode: the boundary text is emitted
        # with a different tokenization than the configured IDs.
        candidate = make_format()
        case = make_fixture("m4", msgs("hello"), "say XY")
        output = tid("say XY")  # canonical X/Y ids, not (70, 71)
        plan = plan_for(candidate, [(case, Behavior("limit", output))])
        report, _backend = run_scenario(
            self,
            candidate=candidate,
            cases=(case,),
            boundary=self.BOUNDARY,
            plan=plan,
            max_new_tokens=len(output),
        )
        obs = report.observations[0]
        self.assertFalse(obs.boundary_found)
        self.assertTrue(obs.boundary_text_match)
        self.assertTrue(obs.exact_match)

    def test_boundary_text_match_none_when_boundary_text_empty(self):
        candidate = make_format()
        case = make_fixture("m5", msgs("hello"), "hi")
        output = pad_to(tid("hi"), 8)
        plan = plan_for(candidate, [(case, Behavior("limit", output))])
        report, _backend = run_scenario(
            self,
            candidate=candidate,
            cases=(case,),
            boundary=(72,),
            plan=plan,
            max_new_tokens=8,
        )
        self.assertIsNone(report.observations[0].boundary_text_match)

    def test_boundary_text_match_none_when_boundary_undecodable(self):
        candidate = make_format()
        case = make_fixture("m6", msgs("hello"), "hi")
        output = pad_to(tid("hi"), 8)
        plan = plan_for(candidate, [(case, Behavior("limit", output))])
        report, _backend = run_scenario(
            self,
            candidate=candidate,
            cases=(case,),
            boundary=(5,),
            plan=plan,
            max_new_tokens=8,
        )
        self.assertIsNone(report.observations[0].boundary_text_match)

    def test_error_with_allowlisted_code_passes_through(self):
        candidate = make_format()
        case = make_fixture("e1", msgs("hello"), "Paris")
        output = tid("oops") + self.BOUNDARY
        plan = plan_for(
            candidate,
            [(case, Behavior("error", output, error="context_limit_exceeded"))],
        )
        report, _backend = run_scenario(
            self,
            candidate=candidate,
            cases=(case,),
            boundary=self.BOUNDARY,
            plan=plan,
            max_new_tokens=len(output),
        )
        obs = report.observations[0]
        self.assertEqual(obs.status, "failed")
        self.assertEqual(obs.error, "context_limit_exceeded")
        self.assertEqual(obs.raw_reason, "error")
        self.assertTrue(obs.boundary_found)  # cut is computed before the error branch
        self.assertEqual(obs.answer, "")
        self.assertIsNone(obs.boundary_text_match)
        self.assertFalse(report.aggregate_eligible)

    def test_error_with_arbitrary_text_is_sanitized(self):
        candidate = make_format()
        case = make_fixture("e2", msgs("hello"), "Paris")
        output = tid("oops") + self.BOUNDARY
        plan = plan_for(
            candidate,
            [(case, Behavior("error", output, error="backend exploded: CUDA OOM"))],
        )
        report, _backend = run_scenario(
            self,
            candidate=candidate,
            cases=(case,),
            boundary=self.BOUNDARY,
            plan=plan,
            max_new_tokens=len(output),
        )
        obs = report.observations[0]
        self.assertEqual(obs.error, "backend_reported_error")
        self.assertNotIn("OOM", repr(report))

    def test_error_none_collapses_to_sanitized(self):
        candidate = make_format()
        case = make_fixture("e3", msgs("hello"), "Paris")
        plan = plan_for(candidate, [(case, Behavior("error", (), error=None))])
        report, _backend = run_scenario(
            self,
            candidate=candidate,
            cases=(case,),
            boundary=self.BOUNDARY,
            plan=plan,
            max_new_tokens=8,
        )
        self.assertEqual(report.observations[0].error, "backend_reported_error")

    def test_generate_raises_records_sanitized_failure(self):
        candidate = make_format()
        case = make_fixture("g1", msgs("hello"), "Paris")
        plan = plan_for(candidate, [(case, Behavior("raise"))])
        report, backend = run_scenario(
            self,
            candidate=candidate,
            cases=(case,),
            boundary=self.BOUNDARY,
            plan=plan,
            max_new_tokens=8,
        )
        obs = report.observations[0]
        self.assertEqual(obs.status, "failed")
        self.assertEqual(obs.error, "generate_failed")
        self.assertEqual(obs.output_ids, ())
        self.assertEqual(obs.answer, "")
        self.assertIsNotNone(obs.input_ids)  # encode succeeded before generate
        self.assertNotIn("explosion", repr(report))
        self.assertEqual(backend.generate_log[0][1], 8)

    def test_invalid_generation_too_long(self):
        self._check_invalid(Generation(tuple(range(9)), "limit"), 8)

    def test_invalid_generation_limit_short(self):
        self._check_invalid(Generation(tuple(range(7)), "limit"), 8)

    def test_invalid_generation_bad_reason(self):
        self._check_invalid(Generation(tuple(range(8)), "done"), 8)

    def test_invalid_generation_error_field_on_limit(self):
        self._check_invalid(Generation(tuple(range(8)), "limit", error="x"), 8)

    def test_invalid_generation_not_a_generation(self):
        self._check_invalid("not-a-generation", 8)

    def test_invalid_generation_negative_ids(self):
        self._check_invalid(Generation((1, -5), "cancelled"), 8)

    def _check_invalid(self, payload, cap):
        candidate = make_format()
        case = make_fixture("iv", msgs("hello"), "Paris")
        plan = plan_for(candidate, [(case, Behavior("invalid", invalid_payload=payload))])
        report, _backend = run_scenario(
            self,
            candidate=candidate,
            cases=(case,),
            boundary=self.BOUNDARY,
            plan=plan,
            max_new_tokens=cap,
        )
        obs = report.observations[0]
        self.assertEqual(obs.status, "failed")
        self.assertEqual(obs.error, "invalid_generation")

    def test_empty_output_on_cancelled_reason(self):
        candidate = make_format()
        case = make_fixture("c1", msgs("hello"), "Paris")
        plan = plan_for(candidate, [(case, Behavior("cancelled", ()))])
        report, _backend = run_scenario(
            self,
            candidate=candidate,
            cases=(case,),
            boundary=self.BOUNDARY,
            plan=plan,
            max_new_tokens=8,
        )
        obs = report.observations[0]
        self.assertEqual(obs.status, "cancelled")
        self.assertEqual(obs.output_ids, ())
        self.assertFalse(obs.boundary_found)
        self.assertEqual(obs.raw_reason, "cancelled")
        self.assertEqual(obs.answer, "")

    def test_output_shorter_than_boundary(self):
        candidate = make_format()
        case = make_fixture("c2", msgs("hello"), "Paris")
        plan = plan_for(candidate, [(case, Behavior("cancelled", (1088,)))])
        report, _backend = run_scenario(
            self,
            candidate=candidate,
            cases=(case,),
            boundary=self.BOUNDARY,
            plan=plan,
            max_new_tokens=8,
        )
        obs = report.observations[0]
        self.assertEqual(obs.status, "cancelled")
        self.assertFalse(obs.boundary_found)

    def test_generate_reports_cancelled_keeps_evidence(self):
        candidate = make_format()
        case = make_fixture("c3", msgs("hello"), "Paris")
        output = tid("partial") + self.BOUNDARY
        plan = plan_for(candidate, [(case, Behavior("cancelled", output))])
        report, _backend = run_scenario(
            self,
            candidate=candidate,
            cases=(case,),
            boundary=self.BOUNDARY,
            plan=plan,
            max_new_tokens=len(output),
        )
        obs = report.observations[0]
        self.assertEqual(obs.status, "cancelled")
        self.assertEqual(obs.output_ids, output)
        self.assertTrue(obs.boundary_found)
        self.assertEqual(obs.answer, "")  # decode never ran
        self.assertEqual(obs.raw_reason, "cancelled")

    def test_decode_raises(self):
        candidate = make_format()
        case = make_fixture("d1", msgs("hello"), "Paris")
        output = pad_to(tid("Paris") + self.BOUNDARY, 16)
        plan = plan_for(
            candidate,
            [(case, Behavior("limit", output))],
            decode_raises=frozenset({tid("Paris")}),
        )
        report, _backend = run_scenario(
            self,
            candidate=candidate,
            cases=(case,),
            boundary=self.BOUNDARY,
            plan=plan,
            max_new_tokens=16,
        )
        obs = report.observations[0]
        self.assertEqual(obs.status, "failed")
        self.assertEqual(obs.error, "decode_failed")
        self.assertEqual(obs.output_ids, output)

    def test_decode_non_str(self):
        candidate = make_format()
        case = make_fixture("d2", msgs("hello"), "Paris")
        output = pad_to(tid("Paris") + self.BOUNDARY, 16)
        plan = plan_for(
            candidate,
            [(case, Behavior("limit", output))],
            decode_non_str=frozenset({tid("Paris")}),
        )
        report, _backend = run_scenario(
            self,
            candidate=candidate,
            cases=(case,),
            boundary=self.BOUNDARY,
            plan=plan,
            max_new_tokens=16,
        )
        self.assertEqual(report.observations[0].error, "decode_failed")

    def test_cancel_preset_aborts_run(self):
        candidate = make_format()
        cases = (
            make_fixture("p1", msgs("one"), "Paris"),
            make_fixture("p2", msgs("two"), "London"),
        )
        plan = plan_for(
            candidate,
            [
                (cases[0], Behavior("limit", pad_to(tid("Paris"), 8))),
                (cases[1], Behavior("limit", pad_to(tid("London"), 8))),
            ],
        )
        report, backend = run_scenario(
            self,
            candidate=candidate,
            cases=cases,
            boundary=self.BOUNDARY,
            plan=plan,
            max_new_tokens=8,
            preset_cancel=True,
        )
        self.assertEqual(report.status, "cancelled")
        self.assertEqual(report.observations, ())
        self.assertEqual(report.planned_count, 2)
        self.assertEqual(backend.generate_log, [])
        self.assertFalse(report.aggregate_eligible)

    def test_cancel_during_encode(self):
        candidate = make_format()
        case = make_fixture("ce", msgs("hello"), "Paris")
        plan = plan_for(
            candidate,
            [(case, Behavior("limit", pad_to(tid("Paris"), 8)))],
            encode_sets_event=True,
        )
        report, _backend = run_scenario(
            self,
            candidate=candidate,
            cases=(case,),
            boundary=self.BOUNDARY,
            plan=plan,
            max_new_tokens=8,
        )
        obs = report.observations[0]
        self.assertEqual(obs.status, "cancelled")
        self.assertNotEqual(obs.input_ids, ())  # encode succeeded first
        self.assertEqual(obs.output_ids, ())
        self.assertIsNone(obs.error)
        self.assertIsNone(obs.raw_reason)

    def test_cancel_between_render_and_encode(self):
        candidate = make_format()
        case = make_fixture("cr", msgs("hello"), "Paris")
        plan = plan_for(
            candidate,
            [(case, Behavior("limit", pad_to(tid("Paris"), 8)))],
            render_sets_event=True,
        )

        def render_wrapper(event):
            real_render = Format.render

            def setting_render(self, messages):
                event.set()
                return real_render(self, messages)

            return mock.patch.object(Format, "render", setting_render)

        report, _backend = run_scenario(
            self,
            candidate=candidate,
            cases=(case,),
            boundary=self.BOUNDARY,
            plan=plan,
            max_new_tokens=8,
            render_wrapper=render_wrapper,
        )
        obs = report.observations[0]
        self.assertEqual(obs.status, "cancelled")
        self.assertEqual(obs.input_ids, ())  # aborted before encode
        self.assertIsNone(obs.error)

    def test_cancel_during_generate(self):
        candidate = make_format()
        case = make_fixture("cg", msgs("hello"), "Paris")
        output = tid("Paris") + self.BOUNDARY
        plan = plan_for(
            candidate,
            [(case, Behavior("limit", output, set_event_in_generate=True))],
        )
        report, _backend = run_scenario(
            self,
            candidate=candidate,
            cases=(case,),
            boundary=self.BOUNDARY,
            plan=plan,
            max_new_tokens=len(output),
        )
        obs = report.observations[0]
        self.assertEqual(obs.status, "cancelled")
        self.assertEqual(obs.output_ids, output)
        self.assertTrue(obs.boundary_found)
        self.assertEqual(obs.answer, "")
        self.assertEqual(obs.raw_reason, "limit")

    def test_cancel_between_fixtures_via_decode_hook(self):
        candidate = make_format()
        c1 = make_fixture("k1", msgs("one"), "Paris")
        c2 = make_fixture("k2", msgs("two"), "London")
        out1 = pad_to(tid("Paris") + self.BOUNDARY, 8)
        out2 = pad_to(tid("London") + self.BOUNDARY, 8)
        retained1 = out1[: 5 + len(self.BOUNDARY)]  # answer ids + boundary
        plan = plan_for(
            candidate,
            [(c1, Behavior("limit", out1)), (c2, Behavior("limit", out2))],
            decode_sets_event=frozenset({retained1}),
        )
        report, _backend = run_scenario(
            self,
            candidate=candidate,
            cases=(c1, c2),
            boundary=self.BOUNDARY,
            plan=plan,
            max_new_tokens=8,
        )
        self.assertEqual(report.status, "cancelled")
        self.assertEqual(len(report.observations), 1)
        self.assertEqual(report.observations[0].status, "completed")
        self.assertEqual(report.observations[0].fixture, "k1")
        self.assertFalse(report.aggregate_eligible)

    def test_render_failed(self):
        candidate = make_format()
        # Messages that fail the renderer's alternation check.
        case = make_fixture("rf", (Message("assistant", "x"),), "Paris")
        plan = plan_for(candidate, [(case, Behavior("raise"))], render_raises=True)
        report, backend = run_scenario(
            self,
            candidate=candidate,
            cases=(case,),
            boundary=self.BOUNDARY,
            plan=plan,
            max_new_tokens=8,
        )
        obs = report.observations[0]
        self.assertEqual(obs.status, "failed")
        self.assertEqual(obs.error, "render_failed")
        self.assertEqual(obs.input_ids, ())
        self.assertEqual(backend.encode_log, [])

    def test_encode_failed(self):
        candidate = make_format()
        case = make_fixture("ef", msgs("hello"), "Paris")
        plan = plan_for(candidate, [(case, Behavior("raise"))], encode_mode="raise")
        report, _backend = run_scenario(
            self,
            candidate=candidate,
            cases=(case,),
            boundary=self.BOUNDARY,
            plan=plan,
            max_new_tokens=8,
        )
        obs = report.observations[0]
        self.assertEqual(obs.status, "failed")
        self.assertEqual(obs.error, "encode_failed")

    def test_invalid_input_ids(self):
        candidate = make_format()
        case = make_fixture("ii", msgs("hello"), "Paris")
        plan = plan_for(candidate, [(case, Behavior("raise"))], encode_mode="invalid")
        report, _backend = run_scenario(
            self,
            candidate=candidate,
            cases=(case,),
            boundary=self.BOUNDARY,
            plan=plan,
            max_new_tokens=8,
        )
        obs = report.observations[0]
        self.assertEqual(obs.status, "failed")
        self.assertEqual(obs.error, "invalid_input_ids")
        self.assertEqual(obs.input_ids, ())

    def test_empty_input_ids_are_valid(self):
        candidate = make_format()
        case = make_fixture("ei", msgs("hello"), "Paris")
        output = tid("Paris") + self.BOUNDARY
        # The behavior is keyed by the empty input the backend returns.
        plan = Plan(behaviors={(): Behavior("limit", output)}, encode_mode="empty")
        report, backend = run_scenario(
            self,
            candidate=candidate,
            cases=(case,),
            boundary=self.BOUNDARY,
            plan=plan,
            max_new_tokens=len(output),
        )
        self.assertEqual(report.observations[0].status, "completed")
        self.assertEqual(backend.generate_log[0][0], ())

    def test_request_seed_and_call_contract(self):
        candidate = make_format()
        case = make_fixture("s1", msgs("hello"), "Paris")
        output = tid("Paris") + self.BOUNDARY
        plan = plan_for(candidate, [(case, Behavior("limit", output))])
        seed = 2**64 - 1
        report, backend = run_scenario(
            self,
            candidate=candidate,
            cases=(case,),
            boundary=self.BOUNDARY,
            plan=plan,
            max_new_tokens=len(output),
            seed=seed,
        )
        (input_ids, cap, logged_seed, cancel_event) = backend.generate_log[0]
        self.assertEqual(logged_seed, oracle_request_seed(seed, "s1"))
        self.assertEqual(cap, len(output))
        # Encode saw the exact oracle-rendered prompt (independent render check).
        self.assertEqual(backend.encode_log, [oracle_render(candidate, case.messages)])
        self.assertEqual(report.observations[0].input_ids, tid(backend.encode_log[0]))

    def test_two_fixtures_mixed(self):
        candidate = make_format()
        c1 = make_fixture("t1", msgs("one"), "Paris")
        c2 = make_fixture("t2", msgs("two"), "London")
        out1 = tid("Paris") + self.BOUNDARY
        out2 = tid("Madrid") + self.BOUNDARY
        cap = len(out2)
        plan = plan_for(
            candidate,
            [(c1, Behavior("limit", pad_to(out1, cap))), (c2, Behavior("limit", out2))],
        )
        report, _backend = run_scenario(
            self,
            candidate=candidate,
            cases=(c1, c2),
            boundary=self.BOUNDARY,
            plan=plan,
            max_new_tokens=cap,
        )
        self.assertEqual(report.status, "completed")
        self.assertEqual(len(report.observations), 2)
        self.assertTrue(report.observations[0].exact_match)
        self.assertFalse(report.observations[1].exact_match)
        self.assertTrue(report.aggregate_eligible)

    def test_repeat_bos_and_per_role_format(self):
        candidate = make_format(
            bos="<s>",
            repeat_bos=True,
            user_turn_end="<|uend|>",
            assistant_turn_end="<|aend|>",
            open_suffix="<open>",
        )
        case = make_fixture("rb", msgs("q1", "a1", "q2"), "Paris")
        output = tid("Paris") + self.BOUNDARY
        plan = plan_for(candidate, [(case, Behavior("limit", output))])
        report, backend = run_scenario(
            self,
            candidate=candidate,
            cases=(case,),
            boundary=self.BOUNDARY,
            plan=plan,
            max_new_tokens=len(output),
        )
        self.assertEqual(report.observations[0].status, "completed")
        # The implementation's render must equal the oracle's field-built prompt.
        self.assertEqual(backend.encode_log[0], oracle_render(candidate, case.messages))
        self.assertIn("<s>", backend.encode_log[0])
        self.assertIn("<|uend|>", backend.encode_log[0])
        self.assertTrue(backend.encode_log[0].endswith("<open>"))


# ---------------------------------------------------------------------------
# Docstring-level contracts, stated as the docstring states them.
# ---------------------------------------------------------------------------


class EvaluateDocstringContracts(unittest.TestCase):
    BOUNDARY = (70, 71)
    FIXED_ERRORS = {
        "render_failed",
        "encode_failed",
        "invalid_input_ids",
        "generate_failed",
        "invalid_generation",
        "decode_failed",
        "backend_reported_error",
    } | set(ORACLE_ALLOWLIST)

    def test_exact_and_boundary_are_separate_no_composite_gate(self):
        # The docstring: "Exact match and boundary presence are separate."
        # Neither combination may flip a fixture's status away from completed.
        candidate = make_format()
        c1 = make_fixture("s1", msgs("one"), "Parisqqq")  # exact hit, no boundary
        c2 = make_fixture("s2", msgs("two"), "London")  # boundary hit, exact miss
        out1 = tid("Parisqqq")
        out2 = tid("Madrid") + self.BOUNDARY
        assert len(out1) == len(out2)
        plan = plan_for(
            candidate,
            [(c1, Behavior("limit", out1)), (c2, Behavior("limit", out2))],
        )
        report, _backend = run_scenario(
            self,
            candidate=candidate,
            cases=(c1, c2),
            boundary=self.BOUNDARY,
            plan=plan,
            max_new_tokens=len(out2),
        )
        self.assertEqual(report.status, "completed")
        o1, o2 = report.observations
        self.assertEqual((o1.status, o1.exact_match, o1.boundary_found), ("completed", True, False))
        self.assertEqual((o2.status, o2.exact_match, o2.boundary_found), ("completed", False, True))

    def test_only_complete_reports_are_aggregate_eligible(self):
        candidate = make_format()
        case = make_fixture("a1", msgs("hello"), "Paris")
        out = tid("Paris") + self.BOUNDARY
        ok_plan = plan_for(candidate, [(case, Behavior("limit", out))])
        report, _b = run_scenario(
            self,
            candidate=candidate,
            cases=(case,),
            boundary=self.BOUNDARY,
            plan=ok_plan,
            max_new_tokens=len(out),
        )
        self.assertTrue(report.aggregate_eligible)

        cancelled, _b = run_scenario(
            self,
            candidate=candidate,
            cases=(case,),
            boundary=self.BOUNDARY,
            plan=ok_plan,
            max_new_tokens=len(out),
            preset_cancel=True,
        )
        self.assertFalse(cancelled.aggregate_eligible)

        failed, _b = run_scenario(
            self,
            candidate=candidate,
            cases=(case,),
            boundary=self.BOUNDARY,
            plan=plan_for(candidate, [(case, Behavior("raise"))]),
            max_new_tokens=len(out),
        )
        self.assertFalse(failed.aggregate_eligible)

    def test_no_backend_exception_text_exported(self):
        candidate = make_format()
        c1 = make_fixture("x1", msgs("one"), "Paris")
        c2 = make_fixture("x2", msgs("two"), "Paris")
        # c1's failure ends the run; check c1 alone and c2 alone for the vocab.
        for single, plan_single in (
            ((c1,), plan_for(candidate, [(c1, Behavior("raise"))])),
            (
                (c2,),
                plan_for(candidate, [(c2, Behavior("error", (), error="secret backend detail"))]),
            ),
        ):
            report, _b = run_scenario(
                self,
                candidate=candidate,
                cases=single,
                boundary=self.BOUNDARY,
                plan=plan_single,
                max_new_tokens=8,
            )
            for obs in report.observations:
                self.assertIn(obs.error, self.FIXED_ERRORS)
            text = repr(report)
            self.assertNotIn("explosion", text)
            self.assertNotIn("secret backend detail", text)

    def test_limit_result_has_exactly_the_requested_cap(self):
        candidate = make_format()
        case = make_fixture("l1", msgs("hello"), "Paris")
        plan = plan_for(
            candidate,
            [(case, Behavior("invalid", invalid_payload=Generation(tuple(range(7)), "limit")))],
        )
        report, _b = run_scenario(
            self,
            candidate=candidate,
            cases=(case,),
            boundary=self.BOUNDARY,
            plan=plan,
            max_new_tokens=8,
        )
        self.assertEqual(report.observations[0].error, "invalid_generation")

    def test_suffix_tokens_remain_in_evidence(self):
        # The docstring: "Boundary IDs must be verified in context; suffix
        # tokens remain in evidence." Tokens after the boundary stay in
        # output_ids but out of the decoded answer.
        candidate = make_format()
        case = make_fixture("sfx", msgs("hello"), "A")
        output = tid("A") + self.BOUNDARY + tid("zzz")
        plan = plan_for(candidate, [(case, Behavior("limit", output))])
        report, _b = run_scenario(
            self,
            candidate=candidate,
            cases=(case,),
            boundary=self.BOUNDARY,
            plan=plan,
            max_new_tokens=len(output),
        )
        obs = report.observations[0]
        self.assertEqual(obs.answer, "A")
        self.assertEqual(obs.output_ids, output)
        self.assertTrue(obs.exact_match and obs.boundary_found)


# ---------------------------------------------------------------------------
# Input validation guards.
# ---------------------------------------------------------------------------


class EvaluateValidation(unittest.TestCase):
    def setUp(self):
        self.candidate = make_format()
        self.case = make_fixture("v1", msgs("hello"), "Paris")
        self.backend = ScriptedBackend(Plan(), Event())

    def _raises(self, **overrides):
        params = {
            "backend": self.backend,
            "candidate": self.candidate,
            "cases": (self.case,),
            "boundary": (70, 71),
        }
        params.update(overrides)
        with self.assertRaises(ValueError):
            evaluate(**params)

    def test_candidate_must_be_format(self):
        self._raises(candidate="not-a-format")

    def test_cases_must_be_nonempty_tuple_of_fixtures(self):
        self._raises(cases=())
        self._raises(cases=[self.case])
        self._raises(cases=(self.case, "nope"))

    def test_fixture_ids_must_be_unique(self):
        dup = make_fixture("v1", msgs("other"), "x")
        self._raises(cases=(self.case, dup))

    def test_fixture_ids_unique_after_nfc_normalization(self):
        nfc = make_fixture("café", msgs("one"), "x")
        nfd = make_fixture("café", msgs("two"), "x")
        self.assertNotEqual(nfc.id, nfd.id)  # distinct code points...
        self._raises(cases=(nfc, nfd))  # ...but canonically equivalent

    def test_max_new_tokens_bounds(self):
        for bad in (0, 4097, -3, "64", True, 1.5, None):
            with self.subTest(bad=bad):
                self._raises(max_new_tokens=bad)

    def test_boundary_must_be_nonempty_id_tuple(self):
        for bad in ((), [1, 2], (1, -1), (True,), ("a",), (1.0,), None):
            with self.subTest(bad=bad):
                self._raises(boundary=bad)

    def test_seed_bounds(self):
        for bad in (-1, 2**64, "0", 3.5):
            with self.subTest(bad=bad):
                self._raises(seed=bad)

    def test_cancel_must_be_event_or_none(self):
        for bad in ("nope", 42, object()):
            with self.subTest(bad=bad):
                self._raises(cancel=bad)


# ---------------------------------------------------------------------------
# Seeded deterministic fuzzing.
# ---------------------------------------------------------------------------


class EvaluateFuzz(unittest.TestCase):
    TRIALS = 30

    def test_fuzz_seeded(self):
        for trial in range(self.TRIALS):
            rng = random.Random(0xC0DE20 + trial)  # noqa: S311 -- fixed deterministic fuzz seed, not crypto
            with self.subTest(trial=trial):
                self._run_fuzz_trial(rng, trial)

    def _run_fuzz_trial(self, rng, trial):
        candidate = Format(
            name=f"fuzzfmt{trial}",
            bos=rng.choice(["", "<s>", "BOS:"]),
            user_prefix=rng.choice(["", "U:", "<user>"]),
            assistant_prefix=rng.choice(["A:", "<assistant>", ""]),
            turn_end=rng.choice(["", "</t>", "\n"]),
            repeat_bos=rng.random() < 0.5,
            user_turn_end=rng.choice([None, "", "<|uend|>"]),
            assistant_turn_end=rng.choice([None, "", "<|aend|>"]),
            open_suffix=rng.choice([None, "", "<open>"]),
        )
        blen = rng.randint(1, 3)
        pool = [70, 71, 72, 5, 1088, 1089, 1090, 2500]
        boundary = tuple(rng.choice(pool) for _ in range(blen))
        cap = rng.choice([1, 16, 64])
        seed = rng.choice([0, rng.randrange(2**64)])
        cancel_mode = rng.choices(["never", "preset", "during"], weights=[8, 1, 1])[0]

        n = rng.randint(2, 6)
        vocab = ["hi", "yo", "2+2?", "explain", "a b", "ok", ""]
        answers = ["Paris", "4", "yes", "no", "42"]
        cases = []
        for i in range(n):
            nturns = rng.choice([1, 3, 5])
            contents = [rng.choice(vocab) for _ in range(nturns)]
            cases.append(
                make_fixture(
                    f"t{trial}f{i}", msgs(*contents), rng.choice(answers), group=f"tg{trial}"
                )
            )

        full_kinds = [
            "match",
            "mismatch",
            "no_boundary",
            "error_ok",
            "error_san",
            "raise",
            "invalid",
            "empty_cancel",
            "short_cancel",
            "gen_cancel",
            "decode_fail",
        ]
        small_kinds = ["raise", "invalid", "empty_cancel", "gen_cancel", "error_ok", "error_san"]
        behaviors = []
        kinds_used = []
        decode_raises = set()
        decode_sets_event = set()
        for case in cases:
            kind = rng.choice(full_kinds if cap > 1 else small_kinds)
            behavior = self._script_behavior(rng, kind, case, boundary, cap, decode_raises, answers)
            if behavior is None:  # scripted output did not fit the cap
                kind, behavior = "raise", Behavior("raise")
            behaviors.append(behavior)
            kinds_used.append(kind)

        if cancel_mode == "during":
            j = rng.randrange(n)
            # Force a limit-behavior fixture so the cancel-during-generate path
            # (limit result + event set -> cancelled record with evidence) is hit.
            limit_behavior = self._script_behavior(
                rng, "match", cases[j], boundary, cap, decode_raises, answers
            )
            if limit_behavior is not None:
                behaviors[j] = limit_behavior
                kinds_used[j] = "match"
            behaviors[j] = dataclasses.replace(behaviors[j], set_event_in_generate=True)
        if n >= 2 and rng.random() < 0.25 and kinds_used[0] in ("match", "mismatch", "no_boundary"):
            out0 = behaviors[0].output_ids
            cut0 = oracle_first_boundary(out0, boundary)
            retained0 = out0 if cut0 is None else out0[: cut0 + len(boundary)]
            decode_sets_event.add(retained0)

        plan = plan_for(
            candidate,
            list(zip(cases, behaviors, strict=True)),
            decode_raises=frozenset(decode_raises),
            decode_sets_event=frozenset(decode_sets_event),
        )
        run_scenario(
            self,
            candidate=candidate,
            cases=tuple(cases),
            boundary=boundary,
            plan=plan,
            max_new_tokens=cap,
            seed=seed,
            preset_cancel=(cancel_mode == "preset"),
        )

    @staticmethod
    def _script_behavior(rng, kind, case, boundary, cap, decode_raises, answers):
        ans = case.expected
        if kind == "match":
            out = tid(ans) + boundary
            if rng.random() < 0.5:
                out = out + tid(" tail")
            if len(out) > cap:
                return None
            return Behavior("limit", pad_to(out, cap))
        if kind == "mismatch":
            out = tid(ans + "?") + boundary
            if len(out) > cap:
                return None
            return Behavior("limit", pad_to(out, cap))
        if kind == "no_boundary":
            for _ in range(10):
                out = pad_to(tid(rng.choice(answers)) + tid("z"), cap)
                if oracle_first_boundary(out, boundary) is None:
                    return Behavior("limit", out)
            return None
        if kind == "error_ok":
            return Behavior(
                "error",
                tid("oops")[:cap],
                error=rng.choice(sorted(ORACLE_ALLOWLIST)),
            )
        if kind == "error_san":
            return Behavior("error", tid("oops")[:cap], error=rng.choice(["kaboom", "", None]))
        if kind == "raise":
            return Behavior("raise")
        if kind == "invalid":
            variant = rng.choice(
                ["too_long", "limit_short", "bad_reason", "error_on_limit", "not_gen", "neg_id"]
            )
            if variant == "too_long":
                payload = Generation(tuple(range(cap + 1)), "limit")
            elif variant == "limit_short":
                payload = Generation(tuple(range(max(cap - 1, 0))), "limit")
            elif variant == "bad_reason":
                payload = Generation(tuple(range(cap)), "done")
            elif variant == "error_on_limit":
                payload = Generation(tuple(range(cap)), "limit", error="x")
            elif variant == "not_gen":
                payload = "not-a-generation"
            else:
                payload = Generation((1, -5), "cancelled")
            return Behavior("invalid", invalid_payload=payload)
        if kind == "empty_cancel":
            return Behavior("cancelled", ())
        if kind == "short_cancel":
            return Behavior("cancelled", (1088,) if len(boundary) > 1 else ())
        if kind == "gen_cancel":
            return Behavior("cancelled", tid("part")[:cap])
        if kind == "decode_fail":
            out = tid(ans) + boundary
            if len(out) > cap:
                return None
            out = pad_to(out, cap)
            cut = oracle_first_boundary(out, boundary)
            decode_raises.add(out if cut is None else out[:cut])
            return Behavior("limit", out)
        raise AssertionError(f"unknown kind {kind}")


# ---------------------------------------------------------------------------
# Sensitivity controls: prove the suite catches implementation mutations.
# ---------------------------------------------------------------------------

MUTANTS = {
    "invert_exact_match": (
        'exact_match=status == "completed" and answer == case.expected,',
        'exact_match=status == "completed" and answer != case.expected,',
    ),
    "answer_keeps_boundary_suffix": (
        "answer_ids = output_ids if cut is None else output_ids[:cut]",
        "answer_ids = output_ids",
    ),
    "drop_event_check_after_generate": (
        'if result.reason == "cancelled" or event.is_set():',
        'if result.reason == "cancelled":',
    ),
    "leak_raw_error_text": (
        "            code = (\n"
        "                result.error\n"
        "                if type(result.error) is str and result.error in _BACKEND_ERROR_CODES\n"
        '                else "backend_reported_error"\n'
        "            )",
        "            code = result.error",
    ),
    "invert_boundary_found": (
        "boundary_found=cut is not None,",
        "boundary_found=cut is None,",
    ),
}


def load_mutant(t: unittest.TestCase, name: str, old: str, new: str):
    """Load a mutated copy of core.py with the real data contracts patched in."""
    src = pathlib.Path(core.__file__).read_text(encoding="utf-8")
    count = src.count(old)
    t.assertEqual(count, 1, f"mutation anchor {name!r} found {count}x, want exactly 1")
    # PID-scoped: two pytest processes staging mutants concurrently must not
    # share a file (a fixed name raced -> half-written imports).
    path = pathlib.Path(tempfile.gettempdir()) / f"cycle20_mut_{os.getpid()}_{name}.py"
    path.write_text(src.replace(old, new), encoding="utf-8")
    t.addCleanup(path.unlink, missing_ok=True)
    spec = importlib.util.spec_from_file_location(f"cycle20_mut_{os.getpid()}_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for attr in ("Format", "Fixture", "Message", "Observation", "EvaluationReport", "Generation"):
        setattr(module, attr, getattr(core, attr))
    return module


class EvaluateSensitivity(unittest.TestCase):
    BOUNDARY = (70, 71)

    def _sensitivity_fixtures(self):
        candidate = make_format()
        a = make_fixture("sa", msgs("q"), "Paris")
        b = make_fixture("sb", msgs("q2"), "x")
        c = make_fixture("sc", msgs("q3"), "yo")
        out_a = tid("Paris") + self.BOUNDARY
        cap = len(out_a)
        plan1 = plan_for(
            candidate,
            [
                (a, Behavior("limit", out_a)),
                (b, Behavior("error", tid("zz")[:cap], error="boom-secret-detail")),
            ],
        )
        plan2 = plan_for(
            candidate,
            [
                (a, Behavior("limit", out_a)),
                (
                    c,
                    Behavior(
                        "limit", pad_to(tid("yo") + self.BOUNDARY, cap), set_event_in_generate=True
                    ),
                ),
            ],
        )
        return candidate, (cap, plan1, (a, b)), (cap, plan2, (a, c))

    def _run_both_scenarios(self, evaluate_fn):
        candidate, s1, s2 = self._sensitivity_fixtures()
        for cap, plan, cases in (s1, s2):
            run_scenario(
                self,
                candidate=candidate,
                cases=cases,
                boundary=self.BOUNDARY,
                plan=plan,
                max_new_tokens=cap,
                evaluate_fn=evaluate_fn,
            )

    def test_unmutated_copy_passes_control(self):
        module = load_mutant(
            self,
            "control",
            "boundary_found=cut is not None,",
            "boundary_found=cut is not None,",
        )
        self._run_both_scenarios(module.evaluate)

    def test_mutants_are_caught(self):
        for name, (old, new) in MUTANTS.items():
            with self.subTest(mutant=name):
                module = load_mutant(self, name, old, new)
                caught = False
                candidate, s1, s2 = self._sensitivity_fixtures()
                for cap, plan, cases in (s1, s2):
                    try:
                        run_scenario(
                            self,
                            candidate=candidate,
                            cases=cases,
                            boundary=self.BOUNDARY,
                            plan=plan,
                            max_new_tokens=cap,
                            evaluate_fn=module.evaluate,
                        )
                    except Exception:  # noqa: BLE001 -- any divergence proves the catch
                        caught = True
                self.assertTrue(caught, f"mutant {name!r} was NOT caught")


if __name__ == "__main__":
    unittest.main()
