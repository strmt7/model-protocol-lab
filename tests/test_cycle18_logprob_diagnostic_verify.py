"""Cycle 18: differential verification of the logprob diagnostic.

Target: ``experiments.logprob_diagnostic`` plus ``_diagnostic_observation``,
``_score_expected_answer``, ``_aggregate_fixture_scores`` and
``_interpret_logprob_candidate``.

Method: an INDEPENDENT oracle re-derived from the docstrings only -- it never
calls the implementation's helpers to produce expected values. The oracle
recomputes per-fixture scoring outcomes from a scripted backend whose forward
pass returns known logits (pure-python log-softmax plus struct-based float32
rounding, versus the engine's torch path), then independently applies the
aggregation rules (which fixtures count, how totals/means combine) and the
band interpretation, including the exact boundary semantics (mean >= -2.0
reads "greedy derailed"; mean < -5.0 reads "weights don't know"; between the
bands is indeterminate; highs+lows is mixed).

Brief-vs-docstring discrepancy (docstrings win): the brief says the
diagnostic scores "the expected answer's teacher-forced logprobs on
natively-failed search fixtures". The ``logprob_diagnostic`` docstring says
"For every search candidate and every search fixture, the fixture's expected
answer is teacher-forced", and the code scores every completed observation --
passes included -- in both the per-fixture scores and the aggregate. Only the
*interpretation* step restricts itself to natively-failed fixtures (the
``_interpret_logprob_candidate`` docstring: "Only natively-failed fixtures
with an ``ok`` score classify"). The oracle follows the docstrings.
"""

import json
import math
import random
import struct
import unittest

import torch

from model_protocol_lab.core import Fixture, Message
from model_protocol_lab.experiments import (
    LOGPROB_DERAILED_MEAN_THRESHOLD,
    LOGPROB_UNKNOWN_MEAN_THRESHOLD,
    _aggregate_fixture_scores,
    _diagnostic_observation,
    _interpret_logprob_candidate,
    _score_expected_answer,
    logprob_diagnostic,
)
from model_protocol_lab.scoring import TargetScore

DERAILED = LOGPROB_DERAILED_MEAN_THRESHOLD  # -2.0
UNKNOWN = LOGPROB_UNKNOWN_MEAN_THRESHOLD  # -5.0

_NOT_SCORED = "not_scored"
_INCOMPLETE = "observation_incomplete"

# ---------------------------------------------------------------------------
# Independent oracle, re-derived from the docstrings of experiments.py and
# scoring.py. It never calls the implementation's helpers; it reimplements
# the documented decision trees and the documented log-softmax math with
# separate code (math module instead of torch, struct instead of
# scoring._f32).
# ---------------------------------------------------------------------------


def _f32(value: float) -> float:
    """Independent float32 rounding via struct (cf. the documented
    float32-rounded reporting in scoring.py).
    """
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _oracle_logprob(logits_row: list[float], token_id: int) -> float:
    """Pure-python log-softmax of one logit row at the target token.

    The documented gathering rule: logits at t-1 predict token t, evaluated
    with max-subtracted log-softmax. Same documented math as the engine, a
    separate code path (math.fsum/math.log instead of torch ops).
    """
    peak = max(logits_row)
    log_normalizer = math.log(math.fsum(math.exp(v - peak) for v in logits_row))
    return (logits_row[token_id] - peak) - log_normalizer


# Whitelisted provider reason codes, copied from the documented vocabulary in
# scoring.py (only these pass through; anything else collapses).
_ORACLE_PROVIDER_CODES = frozenset(
    {
        "invalid_input_ids",
        "context_limit_exceeded",
        "local_forward_failed",
        "local_backend_not_initialized",
        "logits_must_be_tensor",
        "logits_must_be_2d",
        "prompt_len_must_be_positive",
        "invalid_target_ids",
        "empty_vocabulary",
        "logits_length_mismatch",
        "token_id_out_of_range",
        "non_finite_logits",
    }
)


def _oracle_provider_code(exc: BaseException) -> str:
    """Oracle provider code."""
    if exc.args and type(exc.args[0]) is str and exc.args[0] in _ORACLE_PROVIDER_CODES:
        return exc.args[0]
    return "forward_failed"


def _ref_encode(text: str) -> tuple[int, ...]:
    """Reference codec for the scripted backend: char <-> ord bijection, so
    the documented alignment checks always hold.
    """
    return tuple(ord(c) for c in text)


def _ref_decode(ids: tuple[int, ...]) -> str:
    """Ref decode."""
    return "".join(chr(t) for t in ids)


def _piece_encode(text: str, pieces: dict[str, int]) -> tuple[int, ...]:
    """Reference longest-match encoder for the merge-behavior backend."""
    ids: list[int] = []
    pos = 0
    ordered = sorted(pieces, key=len, reverse=True)
    while pos < len(text):
        for piece in ordered:
            if text.startswith(piece, pos):
                ids.append(pieces[piece])
                pos += len(piece)
                break
        else:
            raise ValueError("unencodable")
    return tuple(ids)


def _piece_decode(ids: tuple[int, ...], by_id: dict[int, str]) -> str:
    """Piece decode."""
    return "".join(by_id[i] for i in ids)


def _band_logit(mean_p: float, vocab: int) -> float:
    """Closed-form row peak L such that log-softmax(L, 0, ..., 0) == mean_p.

    logprob = L - ln(e^L + (V-1)); solving for L gives
    L = p + ln((V-1) / (1 - e^p)). Lets the tests place a fixture's mean
    logprob exactly on (or just off) a threshold.
    """
    return mean_p + math.log((vocab - 1) / (1.0 - math.exp(mean_p)))


def _logits_for_mode(input_ids: tuple[int, ...], mode: object, vocab: int) -> "torch.Tensor":
    """Build one scripted (T, V) float64 logit tensor for a mode.

    Modes: "high" (peak on the next token: mean ~= -0.0), "low" (peak on id
    0, never a target id in these tests: mean ~= -100), ("band", p) (every
    gathered row scores exactly p nats), "raise" (forward raises),
    "nonfinite" (an inf row). Rows use the causal layout the engine gathers:
    row t predicts input_ids[t+1].
    """
    seq = tuple(input_ids)
    if mode == "raise":
        raise RuntimeError("scripted-forward-boom")
    if mode == "nonfinite":
        rows = [[0.0] * vocab for _ in seq]
        rows[0][0] = float("inf")
        return torch.tensor(rows, dtype=torch.float64)
    rows = []
    for pos in range(len(seq)):
        row = [0.0] * vocab
        if pos + 1 < len(seq):
            nxt = seq[pos + 1]
            if mode == "high":
                row = [-50.0] * vocab
                row[nxt] = 50.0
            elif mode == "low":
                row = [-50.0] * vocab
                row[0] = 50.0
            elif isinstance(mode, tuple) and mode[0] == "band":
                row[nxt] = _band_logit(float(mode[1]), vocab)
            else:
                raise AssertionError(f"unknown scripted mode {mode!r}")
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float64)


class ScriptedBackend:
    """Deterministic scripted backend: bijective char<->ord codec (alignment
    always holds) plus a per-sequence logit plan. A pure test fixture.
    """

    fingerprint = "cycle18-scripted-backend"

    def __init__(
        self,
        plan: dict[tuple[int, ...], object] | None = None,
        *,
        vocab_size: int = 128,
        encode_raises: frozenset[str] = frozenset(),
        encode_as_list: bool = False,
    ) -> None:
        self._plan = dict(plan or {})
        self.vocab_size = vocab_size
        self._encode_raises = frozenset(encode_raises)
        self._encode_as_list = encode_as_list

    def encode(self, text: str) -> tuple[int, ...]:
        """Encode."""
        if text in self._encode_raises:
            raise ValueError("scripted-unencodable")
        ids = _ref_encode(text)
        return list(ids) if self._encode_as_list else ids  # type: ignore[return-value]

    @staticmethod
    def decode(token_ids: tuple[int, ...]) -> str:
        """Decode."""
        return _ref_decode(tuple(token_ids))

    def forward_logits(self, input_ids: tuple[int, ...]) -> "torch.Tensor":
        """Forward logits."""
        mode = self._plan.get(tuple(input_ids), "low")
        return _logits_for_mode(tuple(input_ids), mode, self.vocab_size)


class MergeBackend:
    """Backend whose tokenizer merges across the prompt/answer boundary, so
    the documented alignment check refuses with boundary_merge_mismatch.
    """

    fingerprint = "cycle18-merge-backend"
    vocab_size = 5

    def __init__(self) -> None:
        self._pieces = {"a": 0, "b": 1, "ab": 2, "c": 3, "d": 4}
        self._by_id = {v: k for k, v in self._pieces.items()}

    def encode(self, text: str) -> tuple[int, ...]:
        """Encode."""
        return _piece_encode(text, self._pieces)

    def decode(self, token_ids: tuple[int, ...]) -> str:
        """Decode."""
        return _piece_decode(tuple(token_ids), self._by_id)

    def forward_logits(self, input_ids: tuple[int, ...]) -> "torch.Tensor":
        """Forward logits."""
        return _logits_for_mode(tuple(input_ids), "high", self.vocab_size)


class NoLogitsBackend(ScriptedBackend):
    """Backend with no forward-logits primitive at all."""

    fingerprint = "cycle18-no-logits-backend"
    forward_logits = None  # type: ignore[assignment]


def _oracle_score_expected(
    backend: object,
    prompt_ids: tuple[int, ...],
    expected: str,
    *,
    vocab_size: int,
    encode_raises: frozenset[str] = frozenset(),
    encode_as_list: bool = False,
    ref_encode=_ref_encode,
    ref_decode=_ref_decode,
) -> TargetScore:
    """Independent reimplementation of the documented scoring decision tree
    (``_score_expected_answer`` + ``score_target_logprobs`` docstrings).
    """
    # _score_expected_answer: encode problems become encode_failed.
    if expected in encode_raises or encode_as_list:
        return TargetScore(
            status="unavailable",
            reason="encode_failed",
            per_token_logprobs=(),
            total_logprob=None,
            mean_logprob=None,
            scored_token_count=0,
            target_byte_length=0,
            prompt_token_count=len(prompt_ids),
        )
    target_ids = ref_encode(expected)
    # score_target_logprobs, in documented order.
    if not target_ids:
        return TargetScore(
            status="ok_empty",
            reason=None,
            per_token_logprobs=(),
            total_logprob=0.0,
            mean_logprob=None,
            scored_token_count=0,
            target_byte_length=0,
            prompt_token_count=len(prompt_ids),
        )
    if not prompt_ids:
        return TargetScore(
            status="unavailable",
            reason="empty_prompt",
            per_token_logprobs=(),
            total_logprob=None,
            mean_logprob=None,
            scored_token_count=0,
            target_byte_length=0,
            prompt_token_count=0,
        )
    if (
        type(vocab_size) is int
        and vocab_size >= 1
        and any(t >= vocab_size for t in prompt_ids + target_ids)
    ):
        return TargetScore(
            status="unavailable",
            reason="token_id_out_of_range",
            per_token_logprobs=(),
            total_logprob=None,
            mean_logprob=None,
            scored_token_count=0,
            target_byte_length=0,
            prompt_token_count=len(prompt_ids),
        )
    target_text = ref_decode(target_ids)
    prompt_text = ref_decode(prompt_ids)
    byte_len = len(target_text.encode("utf-8"))
    if ref_encode(prompt_text) != prompt_ids:
        reason = "prompt_roundtrip_mismatch"
    elif ref_encode(target_text) != target_ids:
        reason = "target_roundtrip_mismatch"
    elif ref_encode(prompt_text + target_text) != prompt_ids + target_ids:
        reason = "boundary_merge_mismatch"
    else:
        reason = None
    if reason is not None:
        return TargetScore(
            status="unavailable",
            reason=reason,
            per_token_logprobs=(),
            total_logprob=None,
            mean_logprob=None,
            scored_token_count=0,
            target_byte_length=byte_len,
            prompt_token_count=len(prompt_ids),
        )
    try:
        logits = backend.forward_logits(prompt_ids + target_ids)  # type: ignore[union-attr]
    except Exception as exc:  # noqa: BLE001 -- provider failures become fixed codes
        return TargetScore(
            status="unavailable",
            reason=_oracle_provider_code(exc),
            per_token_logprobs=(),
            total_logprob=None,
            mean_logprob=None,
            scored_token_count=0,
            target_byte_length=byte_len,
            prompt_token_count=len(prompt_ids),
        )
    rows: list[list[float]] = logits.tolist()
    if not all(math.isfinite(v) for row in rows for v in row):
        return TargetScore(
            status="unavailable",
            reason="non_finite_logits",
            per_token_logprobs=(),
            total_logprob=None,
            mean_logprob=None,
            scored_token_count=0,
            target_byte_length=byte_len,
            prompt_token_count=len(prompt_ids),
        )
    per_f64 = [
        _oracle_logprob(rows[len(prompt_ids) - 1 + i], tid) for i, tid in enumerate(target_ids)
    ]
    per_token = tuple(_f32(v) for v in per_f64)
    total = 0.0
    for v in per_f64:
        total += v
    return TargetScore(
        status="ok",
        reason=None,
        per_token_logprobs=per_token,
        total_logprob=_f32(total),
        mean_logprob=_f32(total / len(target_ids)),
        scored_token_count=len(target_ids),
        target_byte_length=byte_len,
        prompt_token_count=len(prompt_ids),
    )


def _oracle_observation(
    backend: object,
    obs: object,
    expected_by_id: dict[str, str],
    **score_kw: object,
) -> tuple[dict, TargetScore | None]:
    """Independent reimplementation of ``_diagnostic_observation``."""
    fixture_id = obs.get("fixture") if type(obs) is dict else None
    label = fixture_id if type(fixture_id) is str else "?"
    native_pass = type(obs) is dict and bool(obs.get("exact_match") and obs.get("boundary_found"))
    template = {
        "fixture": label,
        "status": _NOT_SCORED,
        "reason": _INCOMPLETE,
        "total_logprob": None,
        "mean_logprob": None,
        "scored_token_count": 0,
        "target_byte_length": 0,
        "native_pass": native_pass,
    }
    if type(obs) is not dict or obs.get("status") != "completed":
        return template, None
    raw_ids = obs.get("input_ids")
    prompt_ids = tuple(raw_ids) if isinstance(raw_ids, (list, tuple)) else ()
    if (
        not prompt_ids
        or any(type(t) is not int or t < 0 for t in prompt_ids)
        or type(fixture_id) is not str
        or fixture_id not in expected_by_id
    ):
        return template, None
    score = _oracle_score_expected(backend, prompt_ids, expected_by_id[fixture_id], **score_kw)  # type: ignore[arg-type]
    scored = {
        "fixture": label,
        "status": score.status,
        "reason": score.reason,
        "total_logprob": score.total_logprob,
        "mean_logprob": score.mean_logprob,
        "scored_token_count": score.scored_token_count,
        "target_byte_length": score.target_byte_length,
        "native_pass": native_pass,
    }
    return scored, score


def _oracle_aggregate(scores: list[TargetScore]) -> dict:
    """Independent reimplementation of ``_aggregate_fixture_scores``."""
    ok_scores = [s for s in scores if s.status == "ok"]
    total = 0.0
    with_total = 0
    tokens = 0
    byte_length = 0
    for s in ok_scores:
        tokens += s.scored_token_count
        byte_length += s.target_byte_length
        if type(s.total_logprob) is float:
            with_total += 1
            total += s.total_logprob
    total_out: float | None = None
    mean_out: float | None = None
    if ok_scores and with_total == len(ok_scores) and tokens:
        total_out = total
        mean_out = total / tokens
    if (
        total_out is None
        or mean_out is None
        or not math.isfinite(total_out)
        or not math.isfinite(mean_out)
    ):
        total_out = None
        mean_out = None
    return {
        "scored_fixtures": len(ok_scores),
        "total_logprob": total_out,
        "mean_logprob": mean_out,
        "scored_token_count": tokens,
        "target_byte_length": byte_length,
    }


def _oracle_interpret_kind(fixture_scores: list[dict]) -> str:
    """Independent reimplementation of the band interpretation, from the
    ``_interpret_logprob_candidate`` docstring: mean >= DERAILED reads
    "greedy derailed"; mean < UNKNOWN reads "weights don't know"; between
    the bands the diagnostic withholds judgment; highs+lows is mixed.
    """
    failures = [s for s in fixture_scores if not s["native_pass"]]
    scored = [s for s in failures if s["status"] == "ok" and type(s["mean_logprob"]) is float]
    if not failures:
        return "no_failures"
    if not scored:
        return "unavailable"
    highs = [s for s in scored if s["mean_logprob"] >= DERAILED]
    lows = [s for s in scored if s["mean_logprob"] < UNKNOWN]
    if highs and not lows:
        return "greedy_derailed"
    if lows and not highs:
        return "weights_dont_know"
    if highs and lows:
        return "mixed"
    return "indeterminate"


_METHOD = (
    "Teacher-forced log-likelihood of each fixture's expected answer: "
    "one forward pass per candidate x fixture over prompt_ids + "
    "encode(expected); prompt positions are masked and answer tokens "
    "are gathered with the causal shift (logits at t-1 predict token "
    "t). Diagnostic only; never changes the selection rule."
)


def _oracle_diagnostic(
    backend: object,
    result: dict,
    fixtures: tuple[Fixture, ...] | list[Fixture],
    **score_kw: object,
) -> dict:
    """Independent reimplementation of ``logprob_diagnostic``."""
    if type(result) is not dict or type(result.get("search")) is not list:
        raise ValueError("result must be a run_experiment result dict")
    if isinstance(fixtures, list):
        fixtures = tuple(fixtures)
    if type(fixtures) is not tuple or any(type(item) is not Fixture for item in fixtures):
        raise ValueError("fixtures must be a tuple of Fixture")
    expected_by_id = {item.id: item.expected for item in fixtures}
    payload: dict = {
        "available": False,
        "detail": None,
        "method": _METHOD,
        "thresholds": {
            "derailed_mean_logprob_at_least": DERAILED,
            "unknown_mean_logprob_below": UNKNOWN,
        },
        "candidates": [],
    }
    if not callable(getattr(backend, "forward_logits", None)):
        payload["detail"] = "logits_not_supported"
        return payload
    for entry in result["search"]:
        if type(entry) is not dict:
            continue
        fmt = entry.get("format")
        name = fmt.get("name") if type(fmt) is dict else None
        observations = entry.get("observations")
        if type(name) is not str or type(observations) is not list:
            continue
        fixture_scores: list[dict] = []
        engine_scores: list[TargetScore] = []
        for obs in observations:
            score_dict, engine_score = _oracle_observation(backend, obs, expected_by_id, **score_kw)
            fixture_scores.append(score_dict)
            if engine_score is not None:
                engine_scores.append(engine_score)
        aggregate = _oracle_aggregate(engine_scores)
        aggregate["native_passes"] = sum(1 for s in fixture_scores if s["native_pass"])
        aggregate["native_failures"] = sum(1 for s in fixture_scores if not s["native_pass"])
        aggregate["unavailable_fixtures"] = sum(
            1 for s in fixture_scores if s["status"] in ("unavailable", _NOT_SCORED)
        )
        payload["candidates"].append(
            {
                "name": name,
                "fixture_scores": fixture_scores,
                "aggregate": aggregate,
                "interpretation": {"kind": _oracle_interpret_kind(fixture_scores)},
            }
        )
    if not payload["candidates"]:
        payload["detail"] = "no_scorable_observations"
        return payload
    payload["available"] = True
    return payload


# ---------------------------------------------------------------------------
# Small builders and shared assertion helpers.
# ---------------------------------------------------------------------------


def _make_fixture(fixture_id: str, expected: str, group: str = "g") -> Fixture:
    """Make fixture."""
    return Fixture(
        id=fixture_id,
        messages=(Message(role="user", content="q"),),
        expected=expected,
        group=group,
    )


def _make_obs(
    fixture_id: str,
    prompt_ids: tuple[int, ...] | list[int] | None,
    *,
    status: str = "completed",
    exact_match: bool = False,
    boundary_found: bool = False,
) -> dict:
    """Make obs."""
    return {
        "candidate": "cand",
        "fixture": fixture_id,
        "input_ids": prompt_ids,
        "output_ids": (5, 6),
        "answer": "wrong",
        "exact_match": exact_match,
        "boundary_found": boundary_found,
        "status": status,
    }


def _candidate_entry(name: str, observations: list) -> dict:
    """Candidate entry."""
    return {"format": {"name": name}, "observations": observations}


def _score_dict(
    mean: float | None,
    *,
    native_pass: bool = False,
    status: str = "ok",
    total: float | None = None,
    count: int = 2,
) -> dict:
    """Hand-built fixture-score dict for direct interpretation tests."""
    if total is None and mean is not None:
        total = mean * count
    return {
        "fixture": "f",
        "status": status,
        "reason": None,
        "total_logprob": total,
        "mean_logprob": mean,
        "scored_token_count": count,
        "target_byte_length": 2 * count,
        "native_pass": native_pass,
    }


def _interpretation_mean(fixture_scores: list[dict]) -> float | None:
    """Recompute the token-weighted mean the implementation reports in its
    interpretation text (same selection rules as the implementation).
    """
    failures = [s for s in fixture_scores if not s["native_pass"]]
    scored = [s for s in failures if s["status"] == "ok" and type(s["mean_logprob"]) is float]
    total = sum(s["total_logprob"] for s in scored if type(s["total_logprob"]) is float)
    tokens = sum(s["scored_token_count"] for s in scored if type(s["scored_token_count"]) is int)
    return total / tokens if tokens else None


class DiagnosticCase(unittest.TestCase):
    """Shared differential assertion: implementation vs independent oracle."""

    maxDiff = None

    def assert_diagnostic_matches(
        self, backend: object, result: dict, fixtures, **score_kw: object
    ) -> dict:
        """Assert diagnostic matches."""
        actual = logprob_diagnostic(backend, result, fixtures)
        expected = _oracle_diagnostic(backend, result, fixtures, **score_kw)
        # Top-level payload (everything but the candidate list).
        self.assertEqual(
            {k: v for k, v in actual.items() if k != "candidates"},
            {k: v for k, v in expected.items() if k != "candidates"},
        )
        self.assertEqual(len(actual["candidates"]), len(expected["candidates"]))
        for a_cand, e_cand in zip(actual["candidates"], expected["candidates"], strict=True):
            with self.subTest(candidate=a_cand["name"]):
                self.assertEqual(a_cand["name"], e_cand["name"])
                self.assertEqual(a_cand["fixture_scores"], e_cand["fixture_scores"])
                self.assertEqual(a_cand["aggregate"], e_cand["aggregate"])
                self.assertEqual(
                    a_cand["interpretation"]["kind"],
                    e_cand["interpretation"]["kind"],
                )
                self._check_interpretation_text(a_cand["interpretation"], a_cand["fixture_scores"])
        # results.json is written with allow_nan=False: the payload must
        # always serialize.
        json.dumps(actual, ensure_ascii=True, allow_nan=False)
        return actual

    def _check_interpretation_text(self, interp: dict, fixture_scores: list[dict]) -> None:
        """Check interpretation text."""
        kind = interp["kind"]
        text = interp["text"]
        self.assertIsInstance(text, str)
        self.assertTrue(text)
        if kind == "greedy_derailed":
            self.assertIn("derail", text.lower())
        elif kind == "weights_dont_know":
            self.assertIn("likelihood", text.lower())
        elif kind == "mixed":
            self.assertIn("Mixed causes", text)
        elif kind == "indeterminate":
            self.assertIn("-2.0", text)
            self.assertIn("-5.0", text)
        elif kind == "no_failures":
            self.assertIn("no failure", text.lower())
        elif kind == "unavailable":
            self.assertIn("nothing", text.lower())
        else:
            self.fail(f"unknown interpretation kind {kind!r}")
        if kind in ("greedy_derailed", "weights_dont_know"):
            mean = _interpretation_mean(fixture_scores)
            if mean is None:
                self.assertIn("n/a", text)
            else:
                self.assertIn(f"{mean:.2f}", text)


class ThresholdConstantsTests(unittest.TestCase):
    """Threshold constants tests."""

    def test_threshold_values_pinned(self):
        """Test threshold values pinned."""
        self.assertEqual(LOGPROB_DERAILED_MEAN_THRESHOLD, -2.0)
        self.assertEqual(LOGPROB_UNKNOWN_MEAN_THRESHOLD, -5.0)

    def test_payload_reports_thresholds_and_method(self):
        """Test payload reports thresholds and method."""
        backend = ScriptedBackend()
        fixture = _make_fixture("f1", "c")
        prompt_ids = _ref_encode("ab")
        diag = logprob_diagnostic(
            backend,
            {"search": [_candidate_entry("cand", [_make_obs("f1", prompt_ids)])]},
            (fixture,),
        )
        self.assertEqual(
            diag["thresholds"],
            {"derailed_mean_logprob_at_least": -2.0, "unknown_mean_logprob_below": -5.0},
        )
        self.assertEqual(diag["method"], _METHOD)


class ScoreExpectedAnswerTests(unittest.TestCase):
    """Direct differential tests of ``_score_expected_answer``."""

    def test_ok_path_matches_oracle(self):
        """Test ok path matches oracle."""
        plan = {tuple(_ref_encode("abcd")): "high"}
        backend = ScriptedBackend(plan=plan)
        prompt_ids = _ref_encode("ab")
        actual = _score_expected_answer(backend, prompt_ids, "cd")
        expected = _oracle_score_expected(backend, prompt_ids, "cd", vocab_size=128)
        self.assertEqual(actual, expected)
        self.assertEqual(actual.status, "ok")
        self.assertGreater(actual.mean_logprob, -0.01)

    def test_encode_raising_becomes_encode_failed(self):
        """Test encode raising becomes encode failed."""
        backend = ScriptedBackend(encode_raises=frozenset({"nope"}))
        prompt_ids = _ref_encode("ab")
        actual = _score_expected_answer(backend, prompt_ids, "nope")
        expected = _oracle_score_expected(
            backend, prompt_ids, "nope", vocab_size=128, encode_raises=frozenset({"nope"})
        )
        self.assertEqual(actual, expected)
        self.assertEqual(actual.status, "unavailable")
        self.assertEqual(actual.reason, "encode_failed")
        self.assertEqual(actual.prompt_token_count, len(prompt_ids))
        self.assertEqual(actual.scored_token_count, 0)
        self.assertIsNone(actual.total_logprob)

    def test_encode_returning_non_tuple_becomes_encode_failed(self):
        """Test encode returning non tuple becomes encode failed."""
        backend = ScriptedBackend(encode_as_list=True)
        prompt_ids = _ref_encode("ab")
        actual = _score_expected_answer(backend, prompt_ids, "cd")
        expected = _oracle_score_expected(
            backend, prompt_ids, "cd", vocab_size=128, encode_as_list=True
        )
        self.assertEqual(actual, expected)
        self.assertEqual(actual.reason, "encode_failed")

    def test_empty_expected_is_ok_empty(self):
        """Test empty expected is ok empty."""
        backend = ScriptedBackend()
        prompt_ids = _ref_encode("ab")
        actual = _score_expected_answer(backend, prompt_ids, "")
        expected = _oracle_score_expected(backend, prompt_ids, "", vocab_size=128)
        self.assertEqual(actual, expected)
        self.assertEqual(actual.status, "ok_empty")
        self.assertEqual(actual.total_logprob, 0.0)
        self.assertIsNone(actual.mean_logprob)

    def test_forward_raising_becomes_forward_failed(self):
        """Test forward raising becomes forward failed."""
        prompt_ids = _ref_encode("ab")
        plan = {prompt_ids + _ref_encode("cd"): "raise"}
        backend = ScriptedBackend(plan=plan)
        actual = _score_expected_answer(backend, prompt_ids, "cd")
        expected = _oracle_score_expected(backend, prompt_ids, "cd", vocab_size=128)
        self.assertEqual(actual, expected)
        self.assertEqual(actual.reason, "forward_failed")

    def test_nonfinite_logits_become_non_finite_logits(self):
        """Test nonfinite logits become non finite logits."""
        prompt_ids = _ref_encode("ab")
        plan = {prompt_ids + _ref_encode("cd"): "nonfinite"}
        backend = ScriptedBackend(plan=plan)
        actual = _score_expected_answer(backend, prompt_ids, "cd")
        expected = _oracle_score_expected(backend, prompt_ids, "cd", vocab_size=128)
        self.assertEqual(actual, expected)
        self.assertEqual(actual.reason, "non_finite_logits")

    def test_token_id_out_of_range(self):
        """Test token id out of range."""
        # vocab_size=100 < ord("d")==100: the engine refuses before alignment.
        backend = ScriptedBackend(vocab_size=100)
        prompt_ids = _ref_encode("ab")
        actual = _score_expected_answer(backend, prompt_ids, "d")
        expected = _oracle_score_expected(backend, prompt_ids, "d", vocab_size=100)
        self.assertEqual(actual, expected)
        self.assertEqual(actual.reason, "token_id_out_of_range")

    def test_boundary_merge_mismatch(self):
        """Test boundary merge mismatch."""
        backend = MergeBackend()
        prompt_ids = (0,)  # "a"; "a"+"b" re-tokenizes as the single piece "ab"
        actual = _score_expected_answer(backend, prompt_ids, "b")
        expected = _oracle_score_expected(
            backend,
            prompt_ids,
            "b",
            vocab_size=5,
            ref_encode=lambda t: _piece_encode(t, backend._pieces),
            ref_decode=lambda ids: _piece_decode(ids, backend._by_id),
        )
        self.assertEqual(actual, expected)
        self.assertEqual(actual.reason, "boundary_merge_mismatch")


class AggregateOracleTests(unittest.TestCase):
    """Direct differential tests of ``_aggregate_fixture_scores``."""

    def _check(self, scores: list[TargetScore]) -> dict:
        """Check."""
        actual = _aggregate_fixture_scores(scores)
        expected = _oracle_aggregate(scores)
        self.assertEqual(actual, expected)
        return actual

    def test_empty_list(self):
        """Test empty list."""
        agg = self._check([])
        self.assertEqual(
            agg,
            {
                "scored_fixtures": 0,
                "total_logprob": None,
                "mean_logprob": None,
                "scored_token_count": 0,
                "target_byte_length": 0,
            },
        )

    def test_only_ok_contributes(self):
        """Test only ok contributes."""
        ok = TargetScore("ok", None, (-0.5, -1.5), -2.0, -1.0, 2, 4, 3)
        ok_empty = TargetScore("ok_empty", None, (), 0.0, None, 0, 0, 3)
        refused = TargetScore("unavailable", "forward_failed", (), None, None, 0, 0, 3)
        agg = self._check([ok, ok_empty, refused])
        # ok_empty and unavailable never leak into the sums.
        self.assertEqual(agg["scored_fixtures"], 1)
        self.assertEqual(agg["total_logprob"], -2.0)
        self.assertEqual(agg["mean_logprob"], -1.0)
        self.assertEqual(agg["scored_token_count"], 2)
        self.assertEqual(agg["target_byte_length"], 4)

    def test_token_weighted_mean_across_fixtures(self):
        """Test token weighted mean across fixtures."""
        # Means -1.0 over 2 tokens and -4.0 over 1 token: weighted mean -2.0.
        first = TargetScore("ok", None, (-1.0, -1.0), -2.0, -1.0, 2, 2, 1)
        second = TargetScore("ok", None, (-4.0,), -4.0, -4.0, 1, 1, 1)
        agg = self._check([first, second])
        self.assertEqual(agg["scored_fixtures"], 2)
        self.assertAlmostEqual(agg["total_logprob"], -6.0)
        self.assertAlmostEqual(agg["mean_logprob"], -2.0)
        self.assertEqual(agg["scored_token_count"], 3)

    def test_nonfinite_total_degrades_to_none(self):
        """Test nonfinite total degrades to none."""
        hostile = TargetScore("ok", None, (), float("inf"), float("inf"), 1, 1, 2)
        agg = self._check([hostile])
        self.assertIsNone(agg["total_logprob"])
        self.assertIsNone(agg["mean_logprob"])
        # The counts still report; only the likelihood fields degrade.
        self.assertEqual(agg["scored_fixtures"], 1)
        self.assertEqual(agg["scored_token_count"], 1)
        json.dumps(agg, allow_nan=False)

    def test_nan_total_degrades_to_none(self):
        """Test nan total degrades to none."""
        hostile = TargetScore("ok", None, (), float("nan"), float("nan"), 1, 1, 2)
        agg = self._check([hostile])
        self.assertIsNone(agg["total_logprob"])
        self.assertIsNone(agg["mean_logprob"])
        json.dumps(agg, allow_nan=False)

    def test_none_total_withholds_aggregate(self):
        """Test none total withholds aggregate."""
        # An ok score with a non-float total cannot join the sums.
        hostile = TargetScore("ok", None, (), None, -1.0, 1, 1, 2)
        agg = self._check([hostile])
        self.assertIsNone(agg["total_logprob"])
        self.assertIsNone(agg["mean_logprob"])

    def test_zero_tokens_withholds_totals(self):
        """Test zero tokens withholds totals."""
        hostile = TargetScore("ok", None, (), -2.0, -2.0, 0, 0, 2)
        agg = self._check([hostile])
        self.assertIsNone(agg["total_logprob"])
        self.assertIsNone(agg["mean_logprob"])


class ObservationOracleTests(unittest.TestCase):
    """Direct differential tests of ``_diagnostic_observation``."""

    def _check(self, backend, obs, expected_by_id, **kw) -> None:
        """Check."""
        actual_dict, actual_score = _diagnostic_observation(backend, obs, expected_by_id)
        expected_dict, expected_score = _oracle_observation(backend, obs, expected_by_id, **kw)
        self.assertEqual(actual_dict, expected_dict)
        self.assertEqual(actual_score, expected_score)

    def test_completed_ok_observation(self):
        """Test completed ok observation."""
        prompt_ids = _ref_encode("ab")
        plan = {prompt_ids + _ref_encode("cd"): "high"}
        backend = ScriptedBackend(plan=plan)
        self._check(
            backend,
            _make_obs("f1", prompt_ids),
            {"f1": "cd"},
            vocab_size=128,
        )

    def test_native_pass_flag_combinations(self):
        """Test native pass flag combinations."""
        backend = ScriptedBackend()
        prompt_ids = _ref_encode("ab")
        for exact, boundary in [(True, True), (True, False), (False, True), (False, False)]:
            with self.subTest(exact_match=exact, boundary_found=boundary):
                obs = _make_obs("f1", prompt_ids, exact_match=exact, boundary_found=boundary)
                actual_dict, _ = _diagnostic_observation(backend, obs, {"f1": "c"})
                self.assertEqual(actual_dict["native_pass"], exact and boundary)

    def test_non_dict_observation(self):
        """Test non dict observation."""
        backend = ScriptedBackend()
        for obs in ("junk", None, 42, ["f1"]):
            with self.subTest(obs=obs):
                self._check(backend, obs, {"f1": "c"}, vocab_size=128)
                actual_dict, actual_score = _diagnostic_observation(backend, obs, {"f1": "c"})
                self.assertEqual(actual_dict["status"], "not_scored")
                self.assertEqual(actual_dict["fixture"], "?")
                self.assertIsNone(actual_score)

    def test_non_completed_status_not_scored(self):
        """Test non completed status not scored."""
        backend = ScriptedBackend()
        prompt_ids = _ref_encode("ab")
        for status in ("failed", "cancelled", "running", None):
            with self.subTest(status=status):
                obs = _make_obs("f1", prompt_ids, status=status)
                self._check(backend, obs, {"f1": "c"}, vocab_size=128)

    def test_bad_input_ids_not_scored(self):
        """Test bad input ids not scored."""
        backend = ScriptedBackend()
        cases = [
            (),
            None,
            "ab",
            (97, -1),
            (97, True),  # bool is not int
            (97, 98.0),
        ]
        for input_ids in cases:
            with self.subTest(input_ids=input_ids):
                obs = _make_obs("f1", input_ids)
                self._check(backend, obs, {"f1": "c"}, vocab_size=128)

    def test_list_input_ids_accepted(self):
        """Test list input ids accepted."""
        # isinstance(list/tuple) accepts lists; tuple(raw_ids) normalizes.
        backend = ScriptedBackend()
        prompt_ids = list(_ref_encode("ab"))
        self._check(backend, _make_obs("f1", prompt_ids), {"f1": "c"}, vocab_size=128)

    def test_unknown_fixture_not_scored(self):
        """Test unknown fixture not scored."""
        backend = ScriptedBackend()
        prompt_ids = _ref_encode("ab")
        self._check(backend, _make_obs("ghost", prompt_ids), {"f1": "c"}, vocab_size=128)

    def test_non_string_fixture_label(self):
        """Test non string fixture label."""
        backend = ScriptedBackend()
        prompt_ids = _ref_encode("ab")
        obs = _make_obs(123, prompt_ids)
        actual_dict, actual_score = _diagnostic_observation(backend, obs, {"f1": "c"})
        expected_dict, expected_score = _oracle_observation(
            backend, obs, {"f1": "c"}, vocab_size=128
        )
        self.assertEqual(actual_dict, expected_dict)
        self.assertEqual(actual_score, expected_score)
        self.assertEqual(actual_dict["status"], "not_scored")
        self.assertEqual(actual_dict["fixture"], "?")

    def test_encode_failure_surfaces_in_dict(self):
        """Test encode failure surfaces in dict."""
        backend = ScriptedBackend(encode_raises=frozenset({"nope"}))
        prompt_ids = _ref_encode("ab")
        self._check(
            backend,
            _make_obs("f1", prompt_ids),
            {"f1": "nope"},
            vocab_size=128,
            encode_raises=frozenset({"nope"}),
        )
        actual_dict, _ = _diagnostic_observation(
            backend, _make_obs("f1", prompt_ids), {"f1": "nope"}
        )
        self.assertEqual(actual_dict["status"], "unavailable")
        self.assertEqual(actual_dict["reason"], "encode_failed")


class InterpretationBoundaryTests(unittest.TestCase):
    """Direct differential tests of ``_interpret_logprob_candidate`` with
    hand-built fixture-score dicts -- the exact boundary semantics.
    """

    # (mean_logprob, expected kind): the brief's boundary rules are
    #   highs: mean >= -2.0 ; lows: mean < -5.0 ; between: indeterminate.
    BOUNDARY_CASES = [
        (-2.0, "greedy_derailed"),  # exactly ON the derailed edge counts as high
        (-5.0, "indeterminate"),  # exactly ON the unknown edge does NOT count as low
        (-1.75, "greedy_derailed"),
        (-2.25, "indeterminate"),
        (-4.75, "indeterminate"),
        (-5.25, "weights_dont_know"),
        (-3.5, "indeterminate"),
        (-0.5, "greedy_derailed"),
        (-10.0, "weights_dont_know"),
        (0.0, "greedy_derailed"),
    ]

    COMBO_CASES = [
        ([-0.5, -10.0], "mixed"),
        ([-0.5, -3.5], "greedy_derailed"),
        ([-10.0, -3.5], "weights_dont_know"),
        ([-3.5, -4.0], "indeterminate"),
        ([-2.0, -5.0], "greedy_derailed"),  # on-edge high + on-edge mid
        ([-5.0, -10.0], "weights_dont_know"),  # on-edge mid + low
        ([-2.0, -2.0], "greedy_derailed"),
        ([-5.0, -5.0], "indeterminate"),
    ]

    def _check_kind(self, fixture_scores: list[dict], expected_kind: str) -> None:
        """Check kind."""
        actual = _interpret_logprob_candidate(fixture_scores)
        self.assertEqual(actual["kind"], expected_kind)
        self.assertEqual(actual["kind"], _oracle_interpret_kind(fixture_scores))
        self.assertIsInstance(actual["text"], str)
        self.assertTrue(actual["text"])

    def test_boundary_means(self):
        """Test boundary means."""
        for mean, expected_kind in self.BOUNDARY_CASES:
            with self.subTest(mean=mean):
                self._check_kind([_score_dict(mean)], expected_kind)

    def test_boundary_combos(self):
        """Test boundary combos."""
        for means, expected_kind in self.COMBO_CASES:
            with self.subTest(means=means):
                self._check_kind([_score_dict(m) for m in means], expected_kind)

    def test_native_passes_do_not_classify(self):
        """Test native passes do not classify."""
        # A failing mid-band fixture plus a passing low fixture: the passing
        # fixture must not pull the reading toward weights_dont_know.
        scores = [_score_dict(-3.5), _score_dict(-10.0, native_pass=True)]
        self._check_kind(scores, "indeterminate")

    def test_passing_high_fixture_does_not_derail(self):
        """Test passing high fixture does not derail."""
        scores = [_score_dict(-3.5), _score_dict(-0.5, native_pass=True)]
        self._check_kind(scores, "indeterminate")

    def test_no_failures(self):
        """Test no failures."""
        self._check_kind([_score_dict(-10.0, native_pass=True)], "no_failures")
        self._check_kind([], "no_failures")

    def test_all_unavailable(self):
        """Test all unavailable."""
        scores = [_score_dict(None, status="unavailable")]
        self._check_kind(scores, "unavailable")

    def test_ok_empty_failing_fixture_does_not_classify(self):
        """Test ok empty failing fixture does not classify."""
        # ok_empty is not "ok": a failing empty-answer fixture leaves the
        # diagnostic with nothing scored.
        scores = [_score_dict(None, status="ok_empty", total=0.0, count=0)]
        self._check_kind(scores, "unavailable")

    def test_mixed_statuses(self):
        """Test mixed statuses."""
        scores = [
            _score_dict(-0.5),
            _score_dict(None, status="unavailable"),
            _score_dict(None, status="not_scored"),
        ]
        self._check_kind(scores, "greedy_derailed")

    def test_non_float_mean_does_not_classify(self):
        """Test non float mean does not classify."""
        scores = [_score_dict(None, total=-3.0)]
        self._check_kind(scores, "unavailable")

    def test_zero_token_counts_report_na_mean(self):
        """Test zero token counts report na mean."""
        # Defensive path: scored (ok + float mean) but no tokens, so the
        # reported mean is None and the text says "n/a".
        scores = [_score_dict(-1.0, total=None, count=0)]
        actual = _interpret_logprob_candidate(scores)
        self.assertEqual(actual["kind"], "greedy_derailed")
        self.assertEqual(actual["kind"], _oracle_interpret_kind(scores))
        self.assertIn("n/a", actual["text"])

    def test_reported_mean_is_token_weighted(self):
        """Test reported mean is token weighted."""
        scores = [_score_dict(-1.0, total=-2.0, count=2), _score_dict(-4.0, total=-4.0, count=1)]
        actual = _interpret_logprob_candidate(scores)
        self.assertEqual(actual["kind"], "greedy_derailed")
        # Weighted mean: (-2 + -4) / 3 = -2.00 -> high band edge text.
        self.assertIn("-2.00", actual["text"])


class EngineeredBoundaryEndToEndTests(DiagnosticCase):
    """Boundary semantics through the whole stack: scripted logits place a
    fixture's engine-reported mean logprob exactly on (or just off) a
    threshold, and the diagnostic must read the band correctly.
    """

    PLACEMENTS = [
        (-2.0, "greedy_derailed"),  # exactly ON the derailed edge: >= counts
        (-5.0, "indeterminate"),  # exactly ON the unknown edge: < does not count
        (-1.75, "greedy_derailed"),
        (-2.25, "indeterminate"),
        (-4.75, "indeterminate"),
        (-5.25, "weights_dont_know"),
        (-3.5, "indeterminate"),
        (-0.5, "greedy_derailed"),
        (-10.0, "weights_dont_know"),
    ]

    def _band_case(self, mean_p: float, expected: str, prompt_text: str = "ab"):
        """Band case."""
        prompt_ids = _ref_encode(prompt_text)
        plan = {prompt_ids + _ref_encode(expected): ("band", mean_p)}
        backend = ScriptedBackend(plan=plan)
        # Validate the construction with the oracle (not the implementation):
        # the scripted logits must yield EXACTLY mean_p, otherwise the
        # boundary test would silently test the wrong placement.
        probe = _oracle_score_expected(backend, prompt_ids, expected, vocab_size=128)
        self.assertEqual(probe.status, "ok")
        self.assertEqual(probe.mean_logprob, mean_p)
        return backend, prompt_ids

    def test_exact_and_near_boundary_means_end_to_end(self):
        """Test exact and near boundary means end to end."""
        for mean_p, expected_kind in self.PLACEMENTS:
            with self.subTest(mean=mean_p):
                backend, prompt_ids = self._band_case(mean_p, "c")
                fixture = _make_fixture("f1", "c")
                result = {"search": [_candidate_entry("cand", [_make_obs("f1", prompt_ids)])]}
                diag = self.assert_diagnostic_matches(backend, result, (fixture,), vocab_size=128)
                (cand,) = diag["candidates"]
                self.assertEqual(cand["interpretation"]["kind"], expected_kind)
                self.assertEqual(cand["aggregate"]["mean_logprob"], mean_p)

    def test_multi_token_exact_boundary(self):
        """Test multi token exact boundary."""
        # Three identical per-token logprobs: the n*p/n path must still land
        # exactly on the threshold after float32 rounding.
        for mean_p, expected_kind in [(-2.0, "greedy_derailed"), (-5.0, "indeterminate")]:
            with self.subTest(mean=mean_p):
                backend, prompt_ids = self._band_case(mean_p, "dcd")
                fixture = _make_fixture("f1", "dcd")
                result = {"search": [_candidate_entry("cand", [_make_obs("f1", prompt_ids)])]}
                diag = self.assert_diagnostic_matches(backend, result, (fixture,), vocab_size=128)
                (cand,) = diag["candidates"]
                self.assertEqual(cand["interpretation"]["kind"], expected_kind)
                self.assertEqual(cand["aggregate"]["mean_logprob"], mean_p)
                self.assertEqual(cand["aggregate"]["scored_token_count"], 3)


class InterpretationEndToEndTests(DiagnosticCase):
    """Each of the six interpretation kinds, driven end to end."""

    def _run(self, specs, **backend_kw):
        """specs: (fixture_id, expected, mode, native_pass)."""
        fixtures = []
        observations = []
        plan: dict[tuple[int, ...], object] = {}
        prompt_ids = _ref_encode("ab")
        for fid, expected, mode, native_pass in specs:
            fixtures.append(_make_fixture(fid, expected))
            plan[prompt_ids + _ref_encode(expected)] = mode
            observations.append(
                _make_obs(fid, prompt_ids, exact_match=native_pass, boundary_found=native_pass)
            )
        backend = ScriptedBackend(plan=plan, **backend_kw)
        result = {"search": [_candidate_entry("cand", observations)]}
        diag = self.assert_diagnostic_matches(
            backend, result, tuple(fixtures), vocab_size=128, **backend_kw
        )
        return diag["candidates"][0]

    def test_greedy_derailed(self):
        """Test greedy derailed."""
        cand = self._run([("f1", "c", "high", False)])
        self.assertEqual(cand["interpretation"]["kind"], "greedy_derailed")

    def test_weights_dont_know(self):
        """Test weights dont know."""
        cand = self._run([("f1", "c", "low", False)])
        self.assertEqual(cand["interpretation"]["kind"], "weights_dont_know")

    def test_mixed(self):
        """Test mixed."""
        cand = self._run([("f1", "c", "high", False), ("f2", "d", "low", False)])
        self.assertEqual(cand["interpretation"]["kind"], "mixed")

    def test_indeterminate(self):
        """Test indeterminate."""
        cand = self._run([("f1", "c", ("band", -3.5), False)])
        self.assertEqual(cand["interpretation"]["kind"], "indeterminate")

    def test_no_failures(self):
        """Test no failures."""
        cand = self._run([("f1", "c", "low", True), ("f2", "d", "high", True)])
        self.assertEqual(cand["interpretation"]["kind"], "no_failures")

    def test_unavailable(self):
        """Test unavailable."""
        cand = self._run([("f1", "c", "raise", False), ("f2", "d", "nonfinite", False)])
        self.assertEqual(cand["interpretation"]["kind"], "unavailable")
        agg = cand["aggregate"]
        self.assertEqual(agg["scored_fixtures"], 0)
        self.assertIsNone(agg["mean_logprob"])
        self.assertEqual(agg["unavailable_fixtures"], 2)

    def test_passing_fixtures_excluded_from_interpretation_but_in_aggregate(self):
        """Test passing fixtures excluded from interpretation but in aggregate."""
        # The brief's scope subtlety, pinned: scoring covers every completed
        # observation (aggregate counts both), but only failures classify.
        cand = self._run([("f1", "c", "high", False), ("f2", "d", "low", True)])
        self.assertEqual(cand["interpretation"]["kind"], "greedy_derailed")
        agg = cand["aggregate"]
        self.assertEqual(agg["scored_fixtures"], 2)
        self.assertEqual(agg["native_passes"], 1)
        self.assertEqual(agg["native_failures"], 1)

    def test_ok_empty_failing_fixture_excluded_everywhere(self):
        """Test ok empty failing fixture excluded everywhere."""
        cand = self._run([("f1", "", "high", False), ("f2", "c", "high", False)])
        self.assertEqual(cand["interpretation"]["kind"], "greedy_derailed")
        agg = cand["aggregate"]
        # ok_empty never leaks into the sums (docstring §9.4).
        self.assertEqual(agg["scored_fixtures"], 1)
        self.assertEqual(agg["scored_token_count"], 1)
        self.assertEqual(agg["unavailable_fixtures"], 0)

    def test_candidate_with_no_observations(self):
        """Test candidate with no observations."""
        backend = ScriptedBackend()
        result = {"search": [_candidate_entry("cand", [])]}
        diag = self.assert_diagnostic_matches(
            backend, result, (_make_fixture("f1", "c"),), vocab_size=128
        )
        (cand,) = diag["candidates"]
        self.assertEqual(cand["interpretation"]["kind"], "no_failures")
        self.assertEqual(cand["aggregate"]["scored_fixtures"], 0)

    def test_encode_as_list_backend_all_unavailable(self):
        """Test encode as list backend all unavailable."""
        cand = self._run(
            [("f1", "c", "high", False), ("f2", "d", "high", False)],
            encode_as_list=True,
        )
        self.assertEqual(cand["interpretation"]["kind"], "unavailable")
        for score in cand["fixture_scores"]:
            self.assertEqual(score["status"], "unavailable")
            self.assertEqual(score["reason"], "encode_failed")


class EndToEndFuzzTests(DiagnosticCase):
    """Seeded differential fuzz: random candidates x fixtures x observation
    shapes x logit modes, full payload compared against the oracle.
    """

    N_SEEDS = 25
    EXPECTED_POOL = ["c", "cd", "dcd", "e", "", "UNENCODABLE"]
    PROMPT_POOL = ["ab", "aab", "bca"]
    MODES: list = [
        "high",
        "low",
        ("band", -2.0),
        ("band", -5.0),
        ("band", -1.75),
        ("band", -2.25),
        ("band", -4.75),
        ("band", -5.25),
        ("band", -3.5),
        ("band", -0.5),
        ("band", -10.0),
        "raise",
        "nonfinite",
    ]
    ENCODE_RAISES = frozenset({"UNENCODABLE"})

    def _build_case(self, seed: int):
        """Build case."""
        rng = random.Random(seed)  # noqa: S311 -- deterministic seeded fuzz, not crypto
        fixtures = [
            _make_fixture(f"f{i}", rng.choice(self.EXPECTED_POOL)) for i in range(rng.randint(1, 4))
        ]
        plan: dict[tuple[int, ...], object] = {}
        candidates = []
        for c in range(rng.randint(1, 3)):
            observations: list = []
            for fixture in fixtures:
                prompt_ids = _ref_encode(rng.choice(self.PROMPT_POOL))
                roll = rng.random()
                if roll < 0.10:
                    obs = _make_obs(
                        fixture.id, prompt_ids, status=rng.choice(["failed", "cancelled"])
                    )
                elif roll < 0.16:
                    bad = rng.choice([(), None, "ab", (97, -2), (97, True), (97, 98.0)])
                    obs = _make_obs(fixture.id, bad)
                elif roll < 0.20:
                    obs = _make_obs("ghost", prompt_ids)
                elif roll < 0.23:
                    obs = rng.choice(["junk", None, 7])
                else:
                    ids = prompt_ids if rng.random() < 0.5 else list(prompt_ids)
                    native = rng.random() < 0.35
                    obs = _make_obs(fixture.id, ids, exact_match=native, boundary_found=native)
                    full = tuple(prompt_ids) + _ref_encode(fixture.expected)
                    plan[full] = rng.choice(self.MODES)
                observations.append(obs)
            if rng.random() < 0.30:
                observations.append(rng.choice(["junk", None]))
            candidates.append(_candidate_entry(f"cand{c}", observations))
        # Candidate entries the diagnostic must skip without failing.
        if rng.random() < 0.4:
            candidates.append("not-a-dict")
        if rng.random() < 0.4:
            candidates.append({"format": {"name": 123}, "observations": []})
        if rng.random() < 0.4:
            candidates.append({"format": None, "observations": []})
        if rng.random() < 0.4:
            candidates.append({"format": {"name": "badobs"}, "observations": "nope"})
        result = {"decision": "ambiguous", "search": candidates}
        fixtures_arg = list(fixtures) if rng.random() < 0.5 else tuple(fixtures)
        backend = ScriptedBackend(plan=plan, encode_raises=self.ENCODE_RAISES)
        return backend, result, fixtures_arg

    def test_seeded_fuzz_full_payload(self):
        """Test seeded fuzz full payload."""
        for seed in range(self.N_SEEDS):
            with self.subTest(seed=seed):
                backend, result, fixtures = self._build_case(seed)
                diag = self.assert_diagnostic_matches(
                    backend,
                    result,
                    fixtures,
                    vocab_size=128,
                    encode_raises=self.ENCODE_RAISES,
                )
                self.assertTrue(diag["available"])


class InputValidationTests(DiagnosticCase):
    """Input validation tests."""

    def test_result_must_be_dict_with_search_list(self):
        """Test result must be dict with search list."""
        backend = ScriptedBackend()
        fixtures = (_make_fixture("f1", "c"),)
        for bad in (None, [], {"search": {}}, {"search": None}, {"other": []}):
            with (
                self.subTest(bad=bad),
                self.assertRaisesRegex(ValueError, "run_experiment result dict"),
            ):
                logprob_diagnostic(backend, bad, fixtures)

    def test_fixtures_must_be_tuple_of_fixture(self):
        """Test fixtures must be tuple of fixture."""
        backend = ScriptedBackend()
        result = {"search": []}
        for bad in (None, "x", (1, 2), [_make_fixture("f1", "c"), "nope"]):
            with (
                self.subTest(bad=bad),
                self.assertRaisesRegex(ValueError, "fixtures must be a tuple"),
            ):
                logprob_diagnostic(backend, result, bad)

    def test_fixtures_list_is_accepted(self):
        """Test fixtures list is accepted."""
        backend = ScriptedBackend()
        fixture = _make_fixture("f1", "c")
        prompt_ids = _ref_encode("ab")
        result = {"search": [_candidate_entry("cand", [_make_obs("f1", prompt_ids)])]}
        diag = logprob_diagnostic(backend, result, [fixture])
        self.assertTrue(diag["available"])

    def test_backend_without_forward_logits(self):
        """Test backend without forward logits."""
        backend = NoLogitsBackend()
        fixture = _make_fixture("f1", "c")
        prompt_ids = _ref_encode("ab")
        result = {"search": [_candidate_entry("cand", [_make_obs("f1", prompt_ids)])]}
        diag = logprob_diagnostic(backend, result, (fixture,))
        self.assertFalse(diag["available"])
        self.assertEqual(diag["detail"], "logits_not_supported")
        self.assertEqual(diag["candidates"], [])
        self.assertEqual(diag, _oracle_diagnostic(backend, result, (fixture,), vocab_size=128))

    def test_no_scorable_observations(self):
        """Test no scorable observations."""
        backend = ScriptedBackend()
        result = {
            "search": [
                "junk",
                {"format": {"name": 123}, "observations": []},
                {"format": None, "observations": []},
                {"format": {"name": "badobs"}, "observations": "nope"},
            ]
        }
        diag = logprob_diagnostic(backend, result, (_make_fixture("f1", "c"),))
        self.assertFalse(diag["available"])
        self.assertEqual(diag["detail"], "no_scorable_observations")
        self.assertEqual(diag["candidates"], [])

    def test_deterministic_across_runs(self):
        """Test deterministic across runs."""
        backend = ScriptedBackend()
        fixtures = (_make_fixture("f1", "cd"), _make_fixture("f2", "dc"))
        prompt_ids = _ref_encode("ab")
        result = {
            "search": [
                _candidate_entry(
                    "cand",
                    [_make_obs("f1", prompt_ids), _make_obs("f2", prompt_ids)],
                )
            ]
        }
        first = logprob_diagnostic(backend, result, fixtures)
        second = logprob_diagnostic(backend, result, fixtures)
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))

    def test_input_result_not_mutated(self):
        """Test input result not mutated."""
        backend = ScriptedBackend()
        fixture = _make_fixture("f1", "c")
        prompt_ids = _ref_encode("ab")
        result = {"search": [_candidate_entry("cand", [_make_obs("f1", prompt_ids)])]}
        before = json.dumps(result, sort_keys=True)
        diag = logprob_diagnostic(backend, result, (fixture,))
        self.assertEqual(json.dumps(result, sort_keys=True), before)
        self.assertNotIn("logprob_diagnostic", result)
        self.assertIsNot(result, diag)


if __name__ == "__main__":
    unittest.main()
