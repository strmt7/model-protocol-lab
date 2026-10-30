"""Cycle 10: wire the scoring engine into `run` + the HTML report (§9.3-§9.5).

Tests-first pins for the `--logprob-diagnostic` wiring:

- `experiments.logprob_diagnostic` scores each candidate's expected answer by
  teacher forcing against the recorded search observations; the input result
  dict is never mutated and the selection rule is untouched (diagnostic only).
- Alignment refusals (e.g. a token straddling the prompt/answer boundary)
  surface as status ``unavailable`` with a fixed reason -- never a silent
  guess, never an exception.
- NaN/inf logits are rejected before aggregation; aggregates stay finite so
  ``results.json`` (``allow_nan=False``) can never be poisoned.
- Scoring is deterministic: two runs over the same inputs produce identical
  dicts.
- The CLI flag is opt-in (default off) and appears in ``run --help``.
- The HTML report renders a "Log-likelihood diagnostic" section with
  per-candidate total/mean logprob and a plain-language reading that
  separates "weights don't know" from "greedy derailed"; it degrades
  gracefully when scores are unavailable and renders nothing when the
  diagnostic key is absent (the byte-identical sample-report golden test
  must keep holding).
- Hostile fixture/candidate text cannot inject HTML/JS or terminal escapes.
"""

import io
import json
import unittest
from contextlib import redirect_stdout

import torch

from model_protocol_lab.cli import build_parser
from model_protocol_lab.core import Fixture, Message
from model_protocol_lab.experiments import (
    LOGPROB_DERAILED_MEAN_THRESHOLD,
    LOGPROB_UNKNOWN_MEAN_THRESHOLD,
    logprob_diagnostic,
)
from model_protocol_lab.reporting import render_html
from model_protocol_lab.scoring import TargetScore


class ScriptTokenizer:
    """Char-piece tokenizer with controllable merge behavior (local copy of
    the cycle-9 pattern so this module stays self-contained).
    """

    def __init__(self, pieces: dict[str, int]) -> None:
        self._pieces = dict(pieces)
        self._by_id = {v: k for k, v in pieces.items()}

    def encode(self, text: str) -> tuple[int, ...]:
        """Encode."""
        ids: list[int] = []
        pos = 0
        ordered = sorted(self._pieces, key=len, reverse=True)
        while pos < len(text):
            for piece in ordered:
                if text.startswith(piece, pos):
                    ids.append(self._pieces[piece])
                    pos += len(piece)
                    break
            else:
                raise ValueError("unencodable_text")
        return tuple(ids)

    def decode(self, token_ids: tuple[int, ...]) -> str:
        """Decode."""
        return "".join(self._by_id[token_id] for token_id in token_ids)


class LogitStubBackend:
    """Weight-free backend: scripted tokenizer plus a logit callback."""

    fingerprint = "stub-logprob-backend-v1"

    def __init__(
        self,
        pieces: dict[str, int],
        logit_fn=None,
        *,
        vocab_size: int = 4,
        with_forward_logits: bool = True,
    ) -> None:
        self._tokenizer = ScriptTokenizer(pieces)
        self._logit_fn = logit_fn
        self.vocab_size = vocab_size
        self.forward_calls: list[tuple[int, ...]] = []
        if not with_forward_logits:
            self.forward_logits = None  # type: ignore[assignment]

    def encode(self, text: str) -> tuple[int, ...]:
        """Encode."""
        return self._tokenizer.encode(text)

    def decode(self, token_ids: tuple[int, ...]) -> str:
        """Decode."""
        return self._tokenizer.decode(token_ids)

    def forward_logits(self, input_ids: tuple[int, ...]):
        """Forward logits."""
        self.forward_calls.append(tuple(input_ids))
        if self._logit_fn is None:
            raise RuntimeError("no_logits_scripted")
        return self._logit_fn(tuple(input_ids))


def peaked_fn(vocab_size: int, peak_on_target: bool):
    """Logit callback: peak 50.0 on the next token (high likelihood) or on
    id 0 (very low likelihood for the target, whose ids are 2/3 in these
    tests). Row t predicts token t+1 -- the causal shift the engine
    gathers -- so the peak goes on input_ids[t+1].
    """

    def fn(input_ids: tuple[int, ...]) -> torch.Tensor:
        """Fn."""
        rows = []
        for pos in range(len(input_ids)):
            row = [-50.0] * vocab_size
            nxt = pos + 1
            if nxt < len(input_ids):
                predicted = input_ids[nxt]
                row[predicted if peak_on_target else 0] = 50.0
            else:
                row[0] = 50.0
            rows.append(row)
        return torch.tensor(rows)

    return fn


PLAIN_PIECES = {"a": 0, "b": 1, "c": 2, "d": 3}
MERGE_PIECES = {"a": 0, "b": 1, "ab": 2, "c": 3, "d": 4}


def make_fixture(fixture_id: str, expected: str) -> Fixture:
    """Make fixture."""
    return Fixture(
        id=fixture_id,
        messages=(Message(role="user", content="q"),),
        expected=expected,
        group="g1",
    )


def make_obs(
    fixture_id: str,
    prompt_ids: tuple[int, ...],
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
        "output_ids": (9, 9),
        "answer": "wrong",
        "exact_match": exact_match,
        "boundary_found": boundary_found,
        "status": status,
    }


def make_result(name: str, observations: list[dict]) -> dict:
    """Make result."""
    return {
        "decision": "ambiguous",
        "search": [
            {
                "format": {"name": name},
                "observations": observations,
            }
        ],
    }


def two_token_expected_case(backend, fixture_id="f1", expected="cd", prompt="ab"):
    """Two token expected case."""
    prompt_ids = backend.encode(prompt)
    fixture = make_fixture(fixture_id, expected)
    return fixture, prompt_ids


class DiagnosticEngineTests(unittest.TestCase):
    """Diagnostic engine tests."""

    def test_ok_score_reports_total_mean_counts(self):
        """Test ok score reports total mean counts."""
        backend = LogitStubBackend(PLAIN_PIECES, peaked_fn(4, True))
        fixture, prompt_ids = two_token_expected_case(backend)
        diag = logprob_diagnostic(
            backend, make_result("cand", [make_obs("f1", prompt_ids)]), (fixture,)
        )
        self.assertTrue(diag["available"])
        (cand,) = diag["candidates"]
        self.assertEqual(cand["name"], "cand")
        (score,) = cand["fixture_scores"]
        self.assertEqual(score["status"], "ok")
        self.assertIsNone(score["reason"])
        self.assertEqual(score["scored_token_count"], 2)
        self.assertEqual(score["target_byte_length"], 2)
        # Peaked-on-target logits: mean logprob ≈ 0 (high likelihood).
        self.assertGreater(score["mean_logprob"], -0.01)
        self.assertAlmostEqual(score["total_logprob"], 2 * score["mean_logprob"], places=6)
        agg = cand["aggregate"]
        self.assertEqual(agg["scored_token_count"], 2)
        self.assertAlmostEqual(agg["mean_logprob"], score["mean_logprob"], places=9)

    def test_alignment_refusal_surfaces_as_unavailable(self):
        """Test alignment refusal surfaces as unavailable."""
        # "a" + "b" re-tokenizes as the single piece "ab": a token straddles
        # the prompt/answer boundary, so the engine must refuse.
        backend = LogitStubBackend(MERGE_PIECES, peaked_fn(5, True), vocab_size=5)
        fixture, prompt_ids = two_token_expected_case(backend, expected="b", prompt="a")
        diag = logprob_diagnostic(
            backend, make_result("cand", [make_obs("f1", prompt_ids)]), (fixture,)
        )
        (cand,) = diag["candidates"]
        (score,) = cand["fixture_scores"]
        self.assertEqual(score["status"], "unavailable")
        self.assertEqual(score["reason"], "boundary_merge_mismatch")
        self.assertIsNone(score["total_logprob"])
        self.assertIsNone(score["mean_logprob"])
        # The refusal is reported, never raised and never guessed.
        self.assertTrue(diag["available"])

    def test_nan_inf_logits_rejected_before_aggregation(self):
        """Test nan inf logits rejected before aggregation."""

        def bad_fn(input_ids: tuple[int, ...]) -> torch.Tensor:
            """Bad fn."""
            rows = [[0.0] * 4 for _ in input_ids]
            rows[-1][2] = float("inf")
            return torch.tensor(rows)

        backend = LogitStubBackend(PLAIN_PIECES, bad_fn)
        fixture, prompt_ids = two_token_expected_case(backend)
        diag = logprob_diagnostic(
            backend, make_result("cand", [make_obs("f1", prompt_ids)]), (fixture,)
        )
        (score,) = diag["candidates"][0]["fixture_scores"]
        self.assertEqual(score["status"], "unavailable")
        self.assertEqual(score["reason"], "non_finite_logits")
        # results.json is written with allow_nan=False: the whole diagnostic
        # payload must serialize.
        json.dumps(diag, ensure_ascii=True, allow_nan=False)

    def test_backend_without_forward_logits_is_unavailable(self):
        """Test backend without forward logits is unavailable."""
        backend = LogitStubBackend(PLAIN_PIECES, with_forward_logits=False)
        fixture, prompt_ids = two_token_expected_case(backend)
        diag = logprob_diagnostic(
            backend, make_result("cand", [make_obs("f1", prompt_ids)]), (fixture,)
        )
        self.assertFalse(diag["available"])
        self.assertEqual(diag["detail"], "logits_not_supported")
        self.assertEqual(diag["candidates"], [])

    def test_incomplete_observation_not_scored(self):
        """Test incomplete observation not scored."""
        backend = LogitStubBackend(PLAIN_PIECES, peaked_fn(4, True))
        fixture, _prompt_ids = two_token_expected_case(backend)
        obs = make_obs("f1", (), status="failed")
        diag = logprob_diagnostic(backend, make_result("cand", [obs]), (fixture,))
        (score,) = diag["candidates"][0]["fixture_scores"]
        self.assertEqual(score["status"], "not_scored")
        self.assertEqual(score["reason"], "observation_incomplete")

    def test_empty_expected_is_ok_empty_not_error(self):
        """Test empty expected is ok empty not error."""
        backend = LogitStubBackend(PLAIN_PIECES, peaked_fn(4, True))
        fixture = make_fixture("f1", "")
        prompt_ids = backend.encode("ab")
        diag = logprob_diagnostic(
            backend, make_result("cand", [make_obs("f1", prompt_ids)]), (fixture,)
        )
        (score,) = diag["candidates"][0]["fixture_scores"]
        self.assertEqual(score["status"], "ok_empty")
        self.assertEqual(score["total_logprob"], 0.0)

    def test_deterministic_across_runs(self):
        """Test deterministic across runs."""
        backend = LogitStubBackend(PLAIN_PIECES, peaked_fn(4, True))
        fixtures = (make_fixture("f1", "cd"), make_fixture("f2", "dc"))
        prompt_ids = backend.encode("ab")
        result = make_result("cand", [make_obs("f1", prompt_ids), make_obs("f2", prompt_ids)])
        first = logprob_diagnostic(backend, result, fixtures)
        second = logprob_diagnostic(backend, result, fixtures)
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))

    def test_input_result_not_mutated_selection_untouched(self):
        """Test input result not mutated selection untouched."""
        backend = LogitStubBackend(PLAIN_PIECES, peaked_fn(4, True))
        fixture, prompt_ids = two_token_expected_case(backend)
        result = make_result("cand", [make_obs("f1", prompt_ids)])
        before = json.dumps(result, sort_keys=True)
        diag = logprob_diagnostic(backend, result, (fixture,))
        self.assertEqual(json.dumps(result, sort_keys=True), before)
        self.assertNotIn("logprob_diagnostic", result)
        self.assertIsNot(result, diag)

    def test_thresholds_declared_and_pinned(self):
        """Test thresholds declared and pinned."""
        self.assertEqual(LOGPROB_DERAILED_MEAN_THRESHOLD, -2.0)
        self.assertEqual(LOGPROB_UNKNOWN_MEAN_THRESHOLD, -5.0)
        backend = LogitStubBackend(PLAIN_PIECES, peaked_fn(4, True))
        fixture, prompt_ids = two_token_expected_case(backend)
        diag = logprob_diagnostic(
            backend, make_result("cand", [make_obs("f1", prompt_ids)]), (fixture,)
        )
        self.assertEqual(
            diag["thresholds"],
            {
                "derailed_mean_logprob_at_least": -2.0,
                "unknown_mean_logprob_below": -5.0,
            },
        )


class InterpretationTests(unittest.TestCase):
    """Interpretation tests."""

    @staticmethod
    def _diag(peak_on_target: bool, exact_match: bool):
        """Diag."""
        backend = LogitStubBackend(PLAIN_PIECES, peaked_fn(4, peak_on_target))
        fixture, prompt_ids = two_token_expected_case(backend)
        obs = make_obs("f1", prompt_ids, exact_match=exact_match, boundary_found=exact_match)
        return logprob_diagnostic(backend, make_result("cand", [obs]), (fixture,))

    def test_high_logprob_failed_fixture_reads_greedy_derailed(self):
        """Test high logprob failed fixture reads greedy derailed."""
        diag = self._diag(peak_on_target=True, exact_match=False)
        interp = diag["candidates"][0]["interpretation"]
        self.assertEqual(interp["kind"], "greedy_derailed")
        self.assertIn("derail", interp["text"].lower())

    def test_low_logprob_failed_fixture_reads_weights_dont_know(self):
        """Test low logprob failed fixture reads weights dont know."""
        diag = self._diag(peak_on_target=False, exact_match=False)
        interp = diag["candidates"][0]["interpretation"]
        self.assertEqual(interp["kind"], "weights_dont_know")
        self.assertIn("likelihood", interp["text"].lower())

    def test_no_failures_kind(self):
        """Test no failures kind."""
        diag = self._diag(peak_on_target=True, exact_match=True)
        interp = diag["candidates"][0]["interpretation"]
        self.assertEqual(interp["kind"], "no_failures")

    def test_all_unavailable_kind(self):
        """Test all unavailable kind."""
        backend = LogitStubBackend(MERGE_PIECES, peaked_fn(5, True), vocab_size=5)
        fixture, prompt_ids = two_token_expected_case(backend, expected="b", prompt="a")
        diag = logprob_diagnostic(
            backend, make_result("cand", [make_obs("f1", prompt_ids)]), (fixture,)
        )
        interp = diag["candidates"][0]["interpretation"]
        self.assertEqual(interp["kind"], "unavailable")

    def test_mixed_bands_kind(self):
        """Test mixed bands kind."""
        # One fixture high, one fixture low: the failures have two causes.
        backend = LogitStubBackend(PLAIN_PIECES, None)

        def mixed_fn(input_ids: tuple[int, ...]) -> torch.Tensor:
            """Mixed fn."""
            # First call peaks on target, second peaks on the wrong token.
            calls = len(backend.forward_calls)
            return peaked_fn(4, peak_on_target=(calls == 1))(input_ids)

        backend._logit_fn = mixed_fn
        fixtures = (make_fixture("f1", "cd"), make_fixture("f2", "dc"))
        prompt_ids = backend.encode("ab")
        result = make_result("cand", [make_obs("f1", prompt_ids), make_obs("f2", prompt_ids)])
        diag = logprob_diagnostic(backend, result, fixtures)
        interp = diag["candidates"][0]["interpretation"]
        self.assertEqual(interp["kind"], "mixed")


class AggregateGuardTests(unittest.TestCase):
    """Aggregate guard tests."""

    def test_nonfinite_engine_values_degrade_aggregate_not_json(self):
        """Test nonfinite engine values degrade aggregate not json."""
        # Defense in depth: even a hostile TargetScore with inf values must
        # not poison results.json (written with allow_nan=False).
        from model_protocol_lab.experiments import _aggregate_fixture_scores

        hostile = TargetScore(
            status="ok",
            reason=None,
            per_token_logprobs=(),
            total_logprob=float("inf"),
            mean_logprob=float("inf"),
            scored_token_count=1,
            target_byte_length=1,
            prompt_token_count=2,
        )
        agg = _aggregate_fixture_scores([hostile])
        self.assertIsNone(agg["total_logprob"])
        self.assertIsNone(agg["mean_logprob"])
        json.dumps(agg, allow_nan=False)


class CliFlagTests(unittest.TestCase):
    """Cli flag tests."""

    @staticmethod
    def _parse(extra=()):
        """Parse."""
        return build_parser().parse_args(
            [
                "run",
                "--model",
                "m",
                "--dataset",
                "d.json",
                "--family",
                "chatml",
                "--boundary-ids",
                "1",
                "--output",
                "out",
                *extra,
            ]
        )

    def test_flag_defaults_off(self):
        """Test flag defaults off."""
        self.assertFalse(self._parse().logprob_diagnostic)

    def test_flag_opt_in(self):
        """Test flag opt in."""
        self.assertTrue(self._parse(("--logprob-diagnostic",)).logprob_diagnostic)

    def test_flag_in_run_help(self):
        """Test flag in run help."""
        buf = io.StringIO()
        with redirect_stdout(buf), self.assertRaises(SystemExit) as ctx:
            build_parser().parse_args(["run", "--help"])
        self.assertEqual(ctx.exception.code, 0)
        self.assertIn("--logprob-diagnostic", buf.getvalue())


class ReportSectionTests(unittest.TestCase):
    """Report section tests."""

    @staticmethod
    def _render(diag):
        """Render."""
        result = {
            "decision": "ambiguous",
            "search": [],
            "logprob_diagnostic": diag,
        }
        return render_html(result)

    @staticmethod
    def _ok_diag():
        """Ok diag."""
        backend = LogitStubBackend(PLAIN_PIECES, peaked_fn(4, True))
        fixture, prompt_ids = two_token_expected_case(backend)
        return logprob_diagnostic(
            backend, make_result("cand", [make_obs("f1", prompt_ids)]), (fixture,)
        )

    def test_section_renders_with_totals_and_reading(self):
        """Test section renders with totals and reading."""
        html = self._render(self._ok_diag())
        self.assertIn("Log-likelihood diagnostic", html)
        self.assertIn('id="logprob-diagnostic"', html)
        # Per-candidate total/mean logprob are visible.
        self.assertIn("mean", html.lower())
        # Plain-language reading separates the two failure attributions.
        self.assertIn("derail", html.lower())
        # TOC links to the section.
        self.assertIn('href="#logprob-diagnostic"', html)

    def test_unavailable_state_degrades_gracefully(self):
        """Test unavailable state degrades gracefully."""
        html = self._render(
            {
                "available": False,
                "detail": "logits_not_supported",
                "method": "",
                "thresholds": {},
                "candidates": [],
            }
        )
        self.assertIn("Log-likelihood diagnostic", html)
        self.assertIn("unavailable", html.lower())
        self.assertIn("logits_not_supported", html)

    def test_missing_key_renders_nothing(self):
        """Test missing key renders nothing."""
        html = render_html({"decision": "ambiguous", "search": []})
        self.assertNotIn("logprob-diagnostic", html)
        self.assertNotIn("Log-likelihood diagnostic", html)

    def test_save_report_round_trip_keeps_diagnostic(self):
        """Test save report round trip keeps diagnostic."""
        # results.json must carry the diagnostic (with unavailable refusals
        # intact) through the real save path, which serializes with
        # allow_nan=False.
        import tempfile
        from pathlib import Path

        from model_protocol_lab.reporting import save_report

        backend = LogitStubBackend(MERGE_PIECES, peaked_fn(5, True), vocab_size=5)
        fixture, prompt_ids = two_token_expected_case(backend, expected="b", prompt="a")
        diag = logprob_diagnostic(
            backend, make_result("cand", [make_obs("f1", prompt_ids)]), (fixture,)
        )
        result = {
            "decision": "ambiguous",
            "search": [],
            "logprob_diagnostic": diag,
        }
        with tempfile.TemporaryDirectory() as tmp:
            json_path, html_path = save_report(result, Path(tmp) / "report")
            saved = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(
                saved["logprob_diagnostic"]["candidates"][0]["fixture_scores"][0]["reason"],
                "boundary_merge_mismatch",
            )
            html = html_path.read_text(encoding="utf-8")
            self.assertIn("Log-likelihood diagnostic", html)
            self.assertIn("boundary_merge_mismatch", html)

    def test_hostile_text_is_escaped(self):
        """Test hostile text is escaped."""
        backend = LogitStubBackend(PLAIN_PIECES, peaked_fn(4, True))
        fixture = make_fixture('<script>alert("x")</script>', "cd")
        prompt_ids = backend.encode("ab")
        diag = logprob_diagnostic(
            backend,
            make_result('cand"><img src=x onerror=alert(1)>', [make_obs(fixture.id, prompt_ids)]),
            (fixture,),
        )
        html = self._render(diag)
        self.assertIn("&lt;script&gt;", html)
        self.assertNotIn('<script>alert("x")', html)
        self.assertNotIn("<img src=x", html)
        # The report's own inline sorter script is still exactly one script tag.
        self.assertEqual(html.count("<script>"), 1)


if __name__ == "__main__":
    unittest.main()
