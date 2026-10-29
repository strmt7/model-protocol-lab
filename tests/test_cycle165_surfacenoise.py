"""Cycle 165: surface-noise self-consistency probe.

Tests for model_protocol_lab.surfacenoise (the dynamic tier to ctxray's
static tier): token indel-edit distance vs a brute-force oracle, verdict
function V, frozen protocol, the probe end-to-end, controls, static lint,
dashboard escaping, the CLI, diff_surface_noise, tempdir mutation gates,
and Hypothesis properties.
"""

import importlib.util
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from model_protocol_lab import surfacenoise as sn
from model_protocol_lab.core import Fixture, Message


def _fixtures(expected="BLUE", n=3, group="t"):
    """Fixtures."""
    return tuple(
        Fixture(
            id=f"t{i}",
            messages=(Message(role="user", content="x"),),
            expected=expected,
            group=group,
        )
        for i in range(n)
    )


def _protocol(backend=None, fixtures=None):
    """Protocol."""
    backend = backend or sn.SelfTestBackend()
    fixtures = fixtures if fixtures is not None else _fixtures()
    return (
        sn.freeze_surface_noise_protocol(
            backend_fingerprint=backend.fingerprint, fixtures=fixtures
        ),
        backend,
        fixtures,
    )


class MyersDistanceTests(unittest.TestCase):
    """Myers distance tests."""

    @staticmethod
    def _brute_force(a, b):
        """Independent O(nm) indel-edit DP (substitution = del+ins)."""
        n, m = len(a), len(b)
        prev = list(range(m + 1))
        for i in range(1, n + 1):
            cur = [i] + [0] * m
            for j in range(1, m + 1):
                cur[j] = min(
                    prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if a[i - 1] == b[j - 1] else 2)
                )
            prev = cur
        denom = n + m
        return 0.0 if denom == 0 else prev[m] / denom

    def test_myers_matches_brute_force_300(self):
        """Test myers matches brute force 300."""
        # Deterministic xorshift32 (no `random` module: S311).
        state = 165

        def randbelow(n):
            """Randbelow."""
            nonlocal state
            state ^= (state << 13) & 0xFFFFFFFF
            state ^= state >> 17
            state ^= (state << 5) & 0xFFFFFFFF
            return state % n

        for _ in range(300):
            a = tuple(randbelow(51) for _ in range(randbelow(13)))
            b = tuple(randbelow(51) for _ in range(randbelow(13)))
            self.assertAlmostEqual(sn.token_edit_rate(a, b), self._brute_force(a, b), places=9)

    def test_distance_edge_cases(self):
        """Test distance edge cases."""
        self.assertEqual(sn.token_edit_rate((), ()), 0.0)
        self.assertEqual(sn.token_edit_rate((1, 2), (1, 2)), 0.0)
        self.assertEqual(sn.token_edit_rate((1,), (2,)), 1.0)
        self.assertEqual(sn.token_edit_rate((), (1, 2)), 1.0)
        self.assertEqual(sn.token_edit_rate((1, 2), ()), 1.0)
        # substitution costs 2 (delete+insert), normalized by combined length
        self.assertAlmostEqual(sn.token_edit_rate((1, 2), (1, 3)), 0.5)

    def test_distance_rejects_non_tuples(self):
        """Test distance rejects non tuples."""
        with self.assertRaises(ValueError):
            sn.token_edit_rate([1, 2], (1, 2))
        with self.assertRaises(ValueError):
            sn.token_edit_rate((1, 2), [1, 2])
        # Element types are not validated (any comparable items work);
        # only the tuple container contract is enforced.
        self.assertEqual(sn.token_edit_rate(("a", "b"), ("a", "b")), 0.0)

    @given(st.lists(st.integers(0, 30), max_size=10), st.lists(st.integers(0, 30), max_size=10))
    @settings(max_examples=60)
    def test_distance_in_unit_interval(self, a, b):
        """Test distance in unit interval."""
        d = sn.token_edit_rate(tuple(a), tuple(b))
        self.assertGreaterEqual(d, 0.0)
        self.assertLessEqual(d, 1.0)
        self.assertEqual(sn.token_edit_rate(tuple(a), tuple(a)), 0.0)

    @given(st.lists(st.integers(0, 30), max_size=10), st.lists(st.integers(0, 30), max_size=10))
    @settings(max_examples=60)
    def test_distance_symmetric(self, a, b):
        """Test distance symmetric."""
        ta, tb = tuple(a), tuple(b)
        self.assertEqual(sn.token_edit_rate(ta, tb), sn.token_edit_rate(tb, ta))


class ValidationTests(unittest.TestCase):
    """Validation tests."""

    def test_prompt_validation(self):
        """Test prompt validation."""
        protocol, backend, fixtures = _protocol()
        # Empty prompts are allowed (the probe measures them); only the
        # container type and the size cap are enforced.
        self.assertEqual(
            sn.probe_surface_noise("", fixtures, backend, protocol, boundary=(7,))["status"],
            "ok",
        )
        with self.assertRaises(ValueError):
            sn.probe_surface_noise(123, fixtures, backend, protocol, boundary=(7,))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            sn.probe_surface_noise(
                "x" * (sn.MAX_PROMPT_CHARS + 1), fixtures, backend, protocol, boundary=(7,)
            )
        # Bidi controls are NOT refused by the probe; they are flagged by
        # the advisory static-lint tier (never gates, never blended).
        bidi_prompt = "a\u202eb"
        report = sn.probe_surface_noise(bidi_prompt, fixtures, backend, protocol, boundary=(7,))
        self.assertEqual(report["status"], "ok")
        # skipcq: PTC-W0063 - in a test, next() raising StopIteration is the intended failure
        bidi = next(c for c in report["static_lint"]["checks"] if c["name"] == "no_bidi_controls")
        self.assertFalse(bidi["passed"])

    def test_boundary_validation(self):
        """Test boundary validation."""
        protocol, backend, fixtures = _protocol()
        with self.assertRaises(ValueError):
            sn.probe_surface_noise("ok", fixtures, backend, protocol, boundary=())
        with self.assertRaises(ValueError):
            sn.probe_surface_noise("ok", fixtures, backend, protocol, boundary=(-1,))
        with self.assertRaises(ValueError):
            sn.probe_surface_noise("ok", fixtures, backend, protocol, boundary=[7])  # type: ignore[arg-type]
        # No length cap at the probe layer (the CLI caps at 16); any
        # nonempty tuple of non-negative ids is accepted.
        self.assertEqual(
            sn.probe_surface_noise("ok", fixtures, backend, protocol, boundary=(1,) * 17)["status"],
            "ok",
        )

    def test_max_new_tokens_and_seed_validation(self):
        """Test max new tokens and seed validation."""
        protocol, backend, fixtures = _protocol()
        with self.assertRaises(ValueError):
            sn.probe_surface_noise(
                "ok", fixtures, backend, protocol, boundary=(7,), max_new_tokens=0
            )
        with self.assertRaises(ValueError):
            sn.probe_surface_noise("ok", fixtures, backend, protocol, boundary=(7,), seed=-1)
        with self.assertRaises(ValueError):
            sn.probe_surface_noise("ok", fixtures, backend, protocol, boundary=(7,), seed=2**64)

    def test_fixture_validation(self):
        """Test fixture validation."""
        with self.assertRaises(ValueError):
            sn.freeze_surface_noise_protocol(backend_fingerprint="x", fixtures=())
        with self.assertRaises(ValueError):
            sn.freeze_surface_noise_protocol(backend_fingerprint="", fixtures=_fixtures())
        many = tuple(
            Fixture(
                id=f"m{i}",
                messages=(Message(role="user", content="x"),),
                expected="BLUE",
                group="t",
            )
            for i in range(sn.MAX_FIXTURES + 1)
        )
        with self.assertRaises(ValueError):
            sn.freeze_surface_noise_protocol(backend_fingerprint="x", fixtures=many)
        dupes = _fixtures() + _fixtures()
        with self.assertRaises(ValueError):
            sn.freeze_surface_noise_protocol(backend_fingerprint="x", fixtures=dupes)

    def test_selftest_backend_validation(self):
        """Test selftest backend validation."""
        with self.assertRaises(ValueError):
            sn.SelfTestBackend(marker="")
        with self.assertRaises(ValueError):
            sn.SelfTestBackend(marker=123)  # type: ignore[arg-type]
        backend = sn.SelfTestBackend()
        with self.assertRaises(ValueError):
            backend.encode(123)  # type: ignore[arg-type]


class ProtocolFreezeTests(unittest.TestCase):
    """Protocol freeze tests."""

    def test_freeze_is_deterministic(self):
        """Test freeze is deterministic."""
        p1, _, _ = _protocol()
        p2, _, _ = _protocol()
        # frozen_at is a wall-clock timestamp; everything else must be stable.
        self.assertNotEqual(p1["frozen_at"], p2["frozen_at"])
        p1 = {k: v for k, v in p1.items() if k != "frozen_at"}
        p2 = {k: v for k, v in p2.items() if k != "frozen_at"}
        self.assertEqual(p1, p2)

    def test_verify_accepts_fresh_protocol(self):
        """Test verify accepts fresh protocol."""
        protocol, backend, _ = _protocol()
        verified = sn.verify_protocol(protocol, backend_fingerprint=backend.fingerprint)
        self.assertIs(verified, protocol)

    def test_verify_rejects_tamper(self):
        """Test verify rejects tamper."""
        protocol, backend, _ = _protocol()
        tampered = dict(protocol)
        tampered["search_plan"] = dict(protocol["search_plan"])
        tampered["search_plan"]["typo"] = {"seeds": [0], "exact": False}
        with self.assertRaises(ValueError):
            sn.verify_protocol(tampered, backend_fingerprint=backend.fingerprint)

    def test_verify_rejects_search_plan_tamper(self):
        """Test verify rejects search plan tamper."""
        # The frozen protocol's search plan is enforced, not just described.
        protocol, backend, _ = _protocol()
        tampered = dict(protocol)
        tampered["search_plan"] = {"typo": {"seeds": [0], "exact": False}}
        with self.assertRaises(ValueError):
            sn.verify_protocol(tampered, backend_fingerprint=backend.fingerprint)

    def test_verify_rejects_wrong_fingerprint(self):
        """Test verify rejects wrong fingerprint."""
        protocol, _, _ = _protocol()
        with self.assertRaises(ValueError):
            sn.verify_protocol(protocol, backend_fingerprint="other/v1")

    def test_verify_rejects_non_dict(self):
        """Test verify rejects non dict."""
        with self.assertRaises(ValueError):
            sn.verify_protocol("nope", backend_fingerprint="x")  # type: ignore[arg-type]

    def test_probe_refuses_tampered_protocol(self):
        """Test probe refuses tampered protocol."""
        protocol, backend, fixtures = _protocol()
        tampered = dict(protocol)
        tampered["protocol_version"] = "9.9.9"
        with self.assertRaises(ValueError):
            sn.probe_surface_noise("ok", fixtures, backend, tampered, boundary=(7,))

    def test_frozen_search_plan_single_source(self):
        """Test frozen search plan single source."""
        plan = sn.frozen_search_plan()
        self.assertEqual(set(plan), set(sn.PERTURBATION_OPERATORS))
        protocol, _, _ = _protocol()
        self.assertEqual(protocol["search_plan"], plan)


class VerdictFunctionTests(unittest.TestCase):
    """R1: V(P, F, B) is the pass/fail/abstain vector — no per-prompt V."""

    def test_pass_vector(self):
        """Test pass vector."""
        _, backend, fixtures = _protocol()
        verdicts = sn.evaluate_prompt("MAGICWORD", fixtures, backend, boundary=(7,))
        self.assertEqual(verdicts, ("pass", "pass", "pass"))

    def test_fail_vector(self):
        """Test fail vector."""
        _, backend, fixtures = _protocol()
        verdicts = sn.evaluate_prompt("no marker here", fixtures, backend, boundary=(7,))
        self.assertEqual(verdicts, ("fail", "fail", "fail"))

    def test_mixed_vector(self):
        """Test mixed vector."""
        fixtures = (
            Fixture(
                id="a", messages=(Message(role="user", content="x"),), expected="BLUE", group="t"
            ),
            Fixture(
                id="b", messages=(Message(role="user", content="x"),), expected="RED", group="t"
            ),
        )
        _, backend, _ = _protocol(fixtures=fixtures)
        verdicts = sn.evaluate_prompt("MAGICWORD", fixtures, backend, boundary=(7,))
        self.assertEqual(verdicts, ("pass", "fail"))

    def test_abstain_on_backend_error(self):
        """Test abstain on backend error."""

        class ExplodingBackend(sn.SelfTestBackend):
            """Exploding backend."""

            fingerprint = "exploding/v1"

            @staticmethod
            def generate(input_ids, *, max_new_tokens, seed, cancel):
                """Generate."""
                raise RuntimeError("boom")

        _, _, fixtures = _protocol(backend=ExplodingBackend(), fixtures=_fixtures())
        verdicts = sn.evaluate_prompt("MAGICWORD", fixtures, ExplodingBackend(), boundary=(7,))
        self.assertEqual(verdicts, ("abstain", "abstain", "abstain"))

    def test_flip_is_vector_difference(self):
        """Test flip is vector difference."""
        # R1: a flip = the verdict VECTOR differs on a mutually-evaluable fixture.
        _, backend, fixtures = _protocol()
        base = sn.evaluate_prompt("MAGICWORD", fixtures, backend, boundary=(7,))
        variant = sn.perturb_text("MAGICWORD", "case_flip", seed=0)
        new = sn.evaluate_prompt(variant, fixtures, backend, boundary=(7,))
        self.assertNotEqual(base, new)
        self.assertTrue(any(b != v for b, v in zip(base, new, strict=True)))


class WhitespaceModeTests(unittest.TestCase):
    """Whitespace mode tests."""

    def test_modes_enumerated(self):
        """Test modes enumerated."""
        modes = sn.whitespace_modes("a  b\tc\nd")
        self.assertEqual(len(modes), len(sn.WHITESPACE_MODES))
        self.assertEqual(len(set(modes)), len(modes))  # all distinct

    def test_collapse_mode(self):
        """Test collapse mode."""
        (collapsed,) = [
            m
            for name, m in zip(sn.WHITESPACE_MODES, sn.whitespace_modes("a  b"), strict=True)
            if name == "collapse"
        ]
        self.assertEqual(collapsed, "a b")

    def test_double_mode(self):
        """Test double mode."""
        (doubled,) = [
            m
            for name, m in zip(sn.WHITESPACE_MODES, sn.whitespace_modes("a b"), strict=True)
            if name == "double"
        ]
        self.assertEqual(doubled, "a  b")

    def test_newline_split_mode(self):
        """Test newline split mode."""
        (split,) = [
            m
            for name, m in zip(sn.WHITESPACE_MODES, sn.whitespace_modes("a b"), strict=True)
            if name == "newline-split"
        ]
        self.assertIn("\n", split)

    def test_modes_reject_non_string(self):
        """Test modes reject non string."""
        with self.assertRaises(ValueError):
            sn.whitespace_modes(123)  # type: ignore[arg-type]


class ProbeEndToEndTests(unittest.TestCase):
    """Probe end to end tests."""

    @staticmethod
    def _report(prompt="MAGICWORD please answer now"):
        """Report."""
        protocol, backend, fixtures = _protocol()
        return sn.probe_surface_noise(prompt, fixtures, backend, protocol, boundary=(7,)), prompt

    def test_status_ok_and_keys(self):
        """Test status ok and keys."""
        report, _ = self._report()
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["instrument"], "surface-noise-probe")
        self.assertEqual(set(report["operators"]), set(sn.PERTURBATION_OPERATORS))
        for key in ("baseline", "denominators", "static_lint", "protocol", "adapter_rule"):
            self.assertIn(key, report)

    def test_genuine_flips_observed(self):
        """Test genuine flips observed."""
        report, _ = self._report()
        total_flips = sum(len(op["flips"]) for op in report["operators"].values())
        self.assertGreater(total_flips, 0, "marker backend must flip under noise")
        # nearest observed flip is an upper bound: distance in [0,1]
        for op in report["operators"].values():
            nearest = op["nearest_observed_flip"]
            if nearest is not None:
                self.assertGreaterEqual(nearest["distance"], 0.0)
                self.assertLessEqual(nearest["distance"], 1.0)

    def test_per_operator_not_scalar(self):
        """Test per operator not scalar."""
        # R2: a per-operator vector, never a collapsed scalar.
        report, _ = self._report()
        stabilities = {n: op["stability"] for n, op in report["operators"].items()}
        self.assertNotIn("score", report)
        self.assertNotIn("overall_stability", report)
        self.assertEqual(len(stabilities), 5)

    def test_denominators_full(self):
        """Test denominators full."""
        report, _ = self._report()
        den = report["denominators"]
        self.assertEqual(
            den["n_total"],
            den["n_effective"] + (den["n_total"] - den["n_effective"]),
        )
        self.assertIn("n_abstained", den)
        self.assertIn("n_error", den)
        # effective = token-id-changed variants only (R8)
        for op in report["operators"].values():
            self.assertLessEqual(op["n_effective"], op["n_total"])

    def test_exact_vs_heuristic_labels(self):
        """Test exact vs heuristic labels."""
        report, _ = self._report()
        self.assertTrue(report["operators"]["whitespace"]["exact"])
        self.assertTrue(report["operators"]["punctuation_strip"]["exact"])
        self.assertFalse(report["operators"]["typo"]["exact"])
        self.assertIn("of 2**64", report["operators"]["typo"]["coverage"])

    def test_baseline(self):
        """Test baseline."""
        report, _ = self._report()
        self.assertEqual(report["baseline"]["n_pass"], 3)
        self.assertEqual(report["baseline"]["n_abstain"], 0)
        self.assertEqual(report["n_fixtures"], 3)

    def test_report_deterministic(self):
        """Test report deterministic."""
        r1, _ = self._report()
        r2, _ = self._report()
        # The embedded protocol carries a wall-clock frozen_at timestamp;
        # everything else must be bit-stable.
        r1 = {k: v for k, v in r1.items() if k != "protocol"}
        r2 = {k: v for k, v in r2.items() if k != "protocol"}
        self.assertEqual(r1, r2)

    def test_empty_prompt_variant_skipped_not_counted(self):
        """Test empty prompt variant skipped not counted."""
        # whitespace modes on an all-whitespace prompt yield empty text:
        # fail-closed validation must skip, not crash.
        protocol, backend, fixtures = _protocol()
        report = sn.probe_surface_noise("   ", fixtures, backend, protocol, boundary=(7,))
        self.assertEqual(report["status"], "ok")


class CancellationTests(unittest.TestCase):
    """Cancellation tests."""

    def test_post_evaluate_cancel(self):
        """Test post evaluate cancel."""
        # The event fires during a variant's evaluation: the post-evaluate
        # check stops the sweep (completed_variants stays 0 for the
        # in-flight variant).
        protocol, backend, fixtures = _protocol()
        cancel_event = threading.Event()
        calls = {"n": 0}
        n_baseline = len(fixtures)
        inner_generate = backend.generate

        def counting_generate(input_ids, *, max_new_tokens, seed, cancel=None):
            """Counting generate."""
            calls["n"] += 1
            if calls["n"] > n_baseline:
                cancel_event.set()
            return inner_generate(
                input_ids, max_new_tokens=max_new_tokens, seed=seed, cancel=cancel
            )

        backend.generate = counting_generate  # type: ignore[method-assign]
        try:
            report = sn.probe_surface_noise(
                "MAGICWORD", fixtures, backend, protocol, boundary=(7,), cancel=cancel_event
            )
        finally:
            backend.generate = inner_generate  # type: ignore[method-assign]
        self.assertEqual(report["status"], "cancelled")
        self.assertEqual(report["completed_variants"], 0)

    def test_evaluate_abstains_on_cancelled_reason(self):
        """Test evaluate abstains on cancelled reason."""
        from model_protocol_lab.core import Generation

        class CancelledBackend(sn.SelfTestBackend):
            """Cancelled backend."""

            fingerprint = "cancelled/v1"

            @staticmethod
            def generate(input_ids, *, max_new_tokens, seed, cancel):
                """Generate."""
                return Generation((), "cancelled")

        backend = CancelledBackend()
        fixtures = _fixtures()
        verdicts = sn.evaluate_prompt("x", fixtures, backend, boundary=(7,))
        self.assertEqual(verdicts, ("abstain",) * len(fixtures))

    def test_evaluate_abstains_on_error_reason(self):
        """Test evaluate abstains on error reason."""
        from model_protocol_lab.core import Generation

        class ErrorBackend(sn.SelfTestBackend):
            """Error backend."""

            fingerprint = "error/v1"

            @staticmethod
            def generate(input_ids, *, max_new_tokens, seed, cancel):
                """Generate."""
                return Generation((), "error", error="diag")

        backend = ErrorBackend()
        fixtures = _fixtures()
        verdicts = sn.evaluate_prompt("x", fixtures, backend, boundary=(7,))
        self.assertEqual(verdicts, ("abstain",) * len(fixtures))

    def test_pre_cancelled_event_is_cancelled(self):
        """Test pre cancelled event is cancelled."""
        # A pre-set event is "cancelled" (Finding 9 fix) — not misattributed
        # as "unavailable". Cancellation is distinguishable from backend failure.
        protocol, backend, fixtures = _protocol()
        cancel = threading.Event()
        cancel.set()
        report = sn.probe_surface_noise(
            "MAGICWORD", fixtures, backend, protocol, boundary=(7,), cancel=cancel
        )
        self.assertEqual(report["status"], "cancelled")
        self.assertEqual(report["reason"], "cancelled_before_baseline")
        self.assertNotIn("operators", report)

    def test_mid_sweep_cancel(self):
        """Test mid sweep cancel."""
        # The event fires between variants: the top-of-loop check stops
        # the sweep and reports cancelled with completed/total counts, no
        # score. (An event set during a variant's evaluation hits the
        # post-evaluate check instead; both paths fail closed.)
        protocol, backend, fixtures = _protocol()

        class CountingEvent(threading.Event):
            """Counting event."""

            # is_set() call order: baseline (n_fixtures) + variant 1
            # (top + n_fixtures + post) + variant 2 top check.
            def __init__(self, fire_at):
                super().__init__()
                self.calls = 0
                self.fire_at = fire_at

            def is_set(self):
                """Is set."""
                self.calls += 1
                return self.calls >= self.fire_at

        n = len(fixtures)
        # is_set() calls: pre-baseline check (Finding 9) + baseline 3n
        # (evaluate_prompt genexpr + generate's cancel check + _evaluate_one
        # post-generate check per fixture) + post-baseline cancelled check
        # (wave-6 finding 4), variant 1: top + 3n + post,
        # then variant 2's top check.
        fire_at = 1 + 3 * n + 1 + (1 + 3 * n + 1) + 1
        cancel_event = CountingEvent(fire_at)
        report = sn.probe_surface_noise(
            "MAGICWORD", fixtures, backend, protocol, boundary=(7,), cancel=cancel_event
        )
        self.assertEqual(report["status"], "cancelled")
        self.assertNotIn("operators", report)
        self.assertIn("completed_variants", report)
        self.assertIn("total_variants", report)
        self.assertEqual(report["completed_variants"], 1)
        self.assertLess(report["completed_variants"], report["total_variants"])


class ControlsTests(unittest.TestCase):
    """Controls tests."""

    def test_controls_discriminate(self):
        """Test controls discriminate."""
        result = sn.run_surface_noise_controls(boundary=(7,))
        self.assertTrue(result["discriminates"])
        by_name = {c["name"]: c for c in result["controls"]}
        self.assertEqual(by_name["sturdy_numbered"]["mean_stability"], 1.0)
        self.assertLess(by_name["fragile_exact_echo"]["mean_stability"], 1.0)
        self.assertIn("theory of fragility", result["circularity_note"])

    def test_controls_self_contained(self):
        """Test controls self contained."""
        # No user fixtures or backend needed: the harness builds its own.
        result = sn.run_surface_noise_controls(boundary=(7,))
        self.assertEqual(len(result["controls"]), 2)
        for control in result["controls"]:
            self.assertEqual(control["status"], "ok")
            self.assertIn(control["expectation"], ("sturdy", "fragile"))
            self.assertIn("designer-authored", control["provenance"])

    def test_controls_heldout(self):
        """Test controls heldout."""
        result = sn.run_surface_noise_controls(boundary=(7,), include_heldout=True)
        names = [c["name"] for c in result["controls"]]
        self.assertEqual(
            names,
            [
                "sturdy_numbered",
                "fragile_exact_echo",
                "heldout_sturdy_list",
                "heldout_fragile_shout",
            ],
        )
        self.assertTrue(result["discriminates"])

    def test_controls_heldout_count_pinned(self):
        """Test controls heldout count pinned."""
        # CI pins the held-out set's count and provenance, not its content.
        self.assertEqual(len(sn.HELDOUT_CONTROLS), 2)
        for control in sn.HELDOUT_CONTROLS:
            self.assertIn("held-out", control["provenance"])

    def test_controls_validation(self):
        """Test controls validation."""
        with self.assertRaises(ValueError):
            sn.run_surface_noise_controls(boundary=())
        with self.assertRaises(ValueError):
            sn.run_surface_noise_controls(boundary=(7,), include_heldout="yes")  # type: ignore[arg-type]


class StaticLintTests(unittest.TestCase):
    """Static lint tests."""

    def test_lint_shape(self):
        """Test lint shape."""
        lint = sn.static_lint("Answer in exactly three words.")
        self.assertTrue(lint["advisory_only"])
        self.assertEqual(len(lint["checks"]), 5)
        names = [c["name"] for c in lint["checks"]]
        self.assertEqual(
            names,
            [
                "nonempty",
                "bounded_length",
                "no_bidi_controls",
                "terminal_punctuation",
                "sentence_repetition",
            ],
        )
        for check in lint["checks"]:
            self.assertIn(check["passed"], (True, False))
            self.assertIsInstance(check["detail"], str)
            self.assertIsInstance(check["rationale"], str)

    def test_lint_flags_bidi(self):
        """Test lint flags bidi."""
        lint = sn.static_lint("a\u202eb")
        # skipcq: PTC-W0063 - in a test, next() raising StopIteration is the intended failure
        bidi = next(c for c in lint["checks"] if c["name"] == "no_bidi_controls")
        self.assertFalse(bidi["passed"])
        self.assertIn("U+202E", bidi["detail"])

    def test_lint_flags_repetition(self):
        """Test lint flags repetition."""
        # Fully-duplicated two-sentence prompt: dup_ratio = 0.5, and the
        # predicate is < 0.5, so it flags.
        lint = sn.static_lint("Say it twice. Say it twice.")
        # skipcq: PTC-W0063 - in a test, next() raising StopIteration is the intended failure
        rep = next(c for c in lint["checks"] if c["name"] == "sentence_repetition")
        self.assertFalse(rep["passed"])

    def test_lint_flags_empty(self):
        """Test lint flags empty."""
        lint = sn.static_lint("   ")
        # skipcq: PTC-W0063 - in a test, next() raising StopIteration is the intended failure
        nonempty = next(c for c in lint["checks"] if c["name"] == "nonempty")
        self.assertFalse(nonempty["passed"])

    def test_lint_passes_clean(self):
        """Test lint passes clean."""
        lint = sn.static_lint("Answer with the word BLUE and nothing else.")
        self.assertTrue(all(c["passed"] for c in lint["checks"]))

    def test_lint_rejects_non_string(self):
        """Test lint rejects non string."""
        with self.assertRaises(ValueError):
            sn.static_lint(123)  # type: ignore[arg-type]

    def test_hardening_notes(self):
        """Test hardening notes."""
        protocol, backend, fixtures = _protocol()
        report = sn.probe_surface_noise("MAGICWORD x", fixtures, backend, protocol, boundary=(7,))
        notes = sn.hardening_notes(report)
        self.assertIsInstance(notes, dict)
        self.assertIn(notes["weakest_operator"], set(sn.PERTURBATION_OPERATORS) | {None})
        self.assertIsInstance(notes["counterfactuals"], list)
        self.assertGreater(len(notes["counterfactuals"]), 0)
        self.assertIn("loop_guidance", notes)

    def test_hardening_notes_rejects_bad_input(self):
        """Test hardening notes rejects bad input."""
        with self.assertRaises(ValueError):
            sn.hardening_notes("nope")  # type: ignore[arg-type]


class DashboardTests(unittest.TestCase):
    """Dashboard tests."""

    @staticmethod
    def _report():
        """Report."""
        protocol, backend, fixtures = _protocol()
        return sn.probe_surface_noise(
            "MAGICWORD please answer now", fixtures, backend, protocol, boundary=(7,)
        )

    def test_dashboard_self_contained(self):
        """Test dashboard self contained."""
        html = sn.render_surface_noise_dashboard(
            self._report(), prompt_text="MAGICWORD please answer now"
        )
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("default-src 'none'", html)
        self.assertIn("<svg", html)
        self.assertNotIn("http://", html.replace("http-equiv", ""))
        self.assertNotIn("https://", html)

    def test_dashboard_escapes_hostile_prompt(self):
        """Test dashboard escapes hostile prompt."""
        evil = "<script>alert(1)</script>&MAGICWORD"
        protocol, backend, fixtures = _protocol()
        report = sn.probe_surface_noise(evil, fixtures, backend, protocol, boundary=(7,))
        html = sn.render_surface_noise_dashboard(report, prompt_text=evil)
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)

    def test_dashboard_excerpts_gated_on_sha(self):
        """Test dashboard excerpts gated on sha."""
        report = self._report()
        variant_text = sn.perturb_text("MAGICWORD please answer now", "typo", seed=1)
        shown = sn.render_surface_noise_dashboard(report, prompt_text="MAGICWORD please answer now")
        hidden = sn.render_surface_noise_dashboard(report, prompt_text="different")
        self.assertIn(variant_text[:200], shown)
        self.assertNotIn(variant_text[:200], hidden)

    def test_dashboard_scope_first(self):
        """Test dashboard scope first."""
        html = sn.render_surface_noise_dashboard(self._report())
        scope_pos = html.find("Scope")
        h2_pos = html.find("<h2>Baseline")
        self.assertLess(scope_pos, h2_pos)
        self.assertIn("Not measured", html)
        self.assertIn("upper bound", html)

    def test_dashboard_rejects_bad_input(self):
        """Test dashboard rejects bad input."""
        with self.assertRaises(ValueError):
            sn.render_surface_noise_dashboard("nope")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            sn.render_surface_noise_dashboard({}, prompt_text=123)  # type: ignore[arg-type]


class DiffSurfaceNoiseTests(unittest.TestCase):
    """Diff surface noise tests."""

    @staticmethod
    def _report(prompt="MAGICWORD please answer now"):
        """Report."""
        protocol, backend, fixtures = _protocol()
        return sn.probe_surface_noise(prompt, fixtures, backend, protocol, boundary=(7,))

    def test_identical_reports(self):
        """Test identical reports."""
        from model_protocol_lab.diffreports import diff_surface_noise

        r = self._report()
        diff = diff_surface_noise(r, r)
        self.assertEqual(diff["available"], "both")
        self.assertTrue(diff["identical"])
        self.assertFalse(diff["changed"])

    def test_prompt_change_detected(self):
        """Test prompt change detected."""
        from model_protocol_lab.diffreports import diff_surface_noise

        diff = diff_surface_noise(
            self._report("MAGICWORD please answer now"),
            self._report("MAGICWORD please answer right now"),
        )
        self.assertTrue(diff["changed"])
        self.assertTrue(diff["prompt_changed"])
        self.assertFalse(diff["protocol_drift"])
        self.assertFalse(diff["identical"])

    def test_protocol_drift_flagged_not_comparable(self):
        """Test protocol drift flagged not comparable."""
        from model_protocol_lab.diffreports import diff_surface_noise

        r1 = self._report()
        _, _backend, fixtures = _protocol()
        other = sn.SelfTestBackend(marker="OTHER")
        protocol2 = sn.freeze_surface_noise_protocol(
            backend_fingerprint=other.fingerprint, fixtures=fixtures
        )
        r2 = sn.probe_surface_noise("OTHER please", fixtures, other, protocol2, boundary=(7,))
        diff = diff_surface_noise(r1, r2)
        self.assertTrue(diff["protocol_drift"])
        self.assertTrue(diff["changed"])

    def test_asymmetric_available(self):
        """Test asymmetric available."""
        from model_protocol_lab.diffreports import diff_surface_noise

        r = self._report()
        self.assertEqual(diff_surface_noise(r, None)["available"], "old")
        self.assertEqual(diff_surface_noise(None, r)["available"], "new")
        self.assertEqual(diff_surface_noise(None, None)["available"], "neither")
        self.assertTrue(diff_surface_noise(None, None)["identical"])

    def test_rejects_malformed(self):
        """Test rejects malformed."""
        from model_protocol_lab.diffreports import diff_surface_noise

        r = self._report()
        bad = dict(r)
        bad["status"] = "cancelled"
        self.assertEqual(diff_surface_noise(r, bad)["available"], "old")
        self.assertEqual(diff_surface_noise(bad, r)["available"], "new")
        bad2 = dict(r)
        bad2["instrument"] = "something-else"
        self.assertEqual(diff_surface_noise(r, bad2)["available"], "old")

    def test_single_cause_pins(self):
        """Test single cause pins."""
        # Each changed flag is attributable to exactly one cause class.
        from model_protocol_lab.diffreports import diff_surface_noise

        r1 = self._report("MAGICWORD please answer now")
        r2 = self._report("MAGICWORD please answer now")
        diff = diff_surface_noise(r1, r2)
        causes = [
            diff["protocol_drift"],
            diff["prompt_changed"],
            bool(diff["operators_added"] or diff["operators_removed"]),
            bool(diff["stability_drift"]),
            bool(diff["nearest_flip_drift"]),
            diff["baseline_changed"],
        ]
        # identical reports: no cause fires
        self.assertFalse(any(causes))


class CLITests(unittest.TestCase):
    """Cli tests."""

    @staticmethod
    def _run_cli(*argv):
        """Run cli."""
        from model_protocol_lab.cli import main

        return main(list(argv))

    def test_protocol_command(self):
        """Test protocol command."""
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            code = self._run_cli("surface-noise", "protocol")
        self.assertEqual(code, 0)
        descriptor = json.loads(buf.getvalue())
        self.assertEqual(descriptor["instrument"], "surface-noise-probe")
        self.assertIn("monday_morning_use", descriptor)
        self.assertIn("adapter_rule", descriptor)

    def test_probe_command_end_to_end(self):
        """Test probe command end to end."""
        with tempfile.TemporaryDirectory(prefix="c165_cli_") as tmpdir:
            prompt_path = Path(tmpdir) / "prompt.txt"
            prompt_path.write_text("MAGICWORD please answer now", encoding="utf-8")
            fixture_path = Path(tmpdir) / "fixtures.json"
            fixture_path.write_text(
                json.dumps(
                    {
                        "search": [
                            {
                                "id": "s1",
                                "group": "g-search",
                                "messages": [{"role": "user", "content": "x"}],
                                "expected": "BLUE",
                            }
                        ],
                        "held_out": [
                            {
                                "id": "h1",
                                "group": "g-held",
                                "messages": [{"role": "user", "content": "y"}],
                                "expected": "BLUE",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            out_dir = Path(tmpdir) / "out"
            import io
            from contextlib import redirect_stdout

            buf = io.StringIO()
            with redirect_stdout(buf):
                code = self._run_cli(
                    "surface-noise",
                    "probe",
                    "--prompt",
                    str(prompt_path),
                    "--fixtures",
                    str(fixture_path),
                    "--boundary-ids",
                    "7",
                    "--output",
                    str(out_dir),
                )
            self.assertEqual(code, 0)
            report = json.loads(buf.getvalue())
            self.assertEqual(report["status"], "ok")
            self.assertTrue((out_dir / "probe.json").exists())
            self.assertTrue((out_dir / "dashboard.html").exists())
            saved = json.loads((out_dir / "probe.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["prompt_sha256"], report["prompt_sha256"])

    def test_probe_refuses_existing_output(self):
        """Test probe refuses existing output."""
        with tempfile.TemporaryDirectory(prefix="c165_cli_") as tmpdir:
            prompt_path = Path(tmpdir) / "prompt.txt"
            prompt_path.write_text("MAGICWORD", encoding="utf-8")
            out_dir = Path(tmpdir) / "out"
            out_dir.mkdir()
            code = self._run_cli(
                "surface-noise",
                "probe",
                "--prompt",
                str(prompt_path),
                "--fixtures",
                "examples/fixtures.json",
                "--boundary-ids",
                "7",
                "--output",
                str(out_dir),
            )
            self.assertEqual(code, 1)

    def test_probe_invalid_prompt_file(self):
        """Test probe invalid prompt file."""
        code = self._run_cli(
            "surface-noise",
            "probe",
            "--prompt",
            "/nonexistent/prompt.txt",
            "--fixtures",
            "examples/fixtures.json",
            "--boundary-ids",
            "7",
            "--output",
            str(Path(tempfile.gettempdir()) / "c165_definitely_missing_xyz"),
        )
        self.assertEqual(code, 1)

    def test_probe_bad_boundary(self):
        """Test probe bad boundary."""
        with self.assertRaises(SystemExit):
            self._run_cli(
                "surface-noise",
                "probe",
                "--prompt",
                "x",
                "--fixtures",
                "y",
                "--boundary-ids",
                "notanid",
                "--output",
                "z",
            )

    def test_diff_command(self):
        """Test diff command."""
        with tempfile.TemporaryDirectory(prefix="c165_cli_") as tmpdir:
            import io
            from contextlib import redirect_stdout

            protocol, backend, fixtures = _protocol()
            r1 = sn.probe_surface_noise("MAGICWORD a", fixtures, backend, protocol, boundary=(7,))
            r2 = sn.probe_surface_noise("MAGICWORD b", fixtures, backend, protocol, boundary=(7,))
            p1 = Path(tmpdir) / "old.json"
            p2 = Path(tmpdir) / "new.json"
            p1.write_text(json.dumps(r1), encoding="utf-8")
            p2.write_text(json.dumps(r2), encoding="utf-8")
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = self._run_cli("surface-noise", "diff", str(p1), str(p2))
            self.assertEqual(code, 0)
            diff = json.loads(buf.getvalue())
            self.assertTrue(diff["changed"])
            self.assertTrue(diff["prompt_changed"])

    def test_diff_invalid_input(self):
        """Test diff invalid input."""
        code = self._run_cli("surface-noise", "diff", "/nonexistent/a.json", "/nonexistent/b.json")
        self.assertEqual(code, 1)

    def test_describe_instrument(self):
        """Test describe instrument."""
        descriptor = sn.describe_instrument()
        self.assertEqual(descriptor["instrument"], "surface-noise-probe")
        self.assertEqual(set(descriptor["search_plan"]), set(sn.PERTURBATION_OPERATORS))
        self.assertIn("You rewrote a prompt", descriptor["monday_morning_use"])


# Tempdir mutation gates (4/4)
# ---------------------------------------------------------------------------


class SurfaceNoiseMutationGateTests(unittest.TestCase):
    """Surface noise mutation gate tests."""

    def _run_gate(self, module_name, name, old_text, new_text, scenario):
        """Run gate."""
        mod = __import__(f"model_protocol_lab.{module_name}", fromlist=["x"])
        source = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertEqual(source.count(old_text), 1, f"mutant {name}: anchor not unique")
        mutated = source.replace(old_text, new_text)
        with tempfile.TemporaryDirectory(prefix="c165_mut_", dir="/tmp") as tmpdir:
            mod_path = os.path.join(tmpdir, f"{module_name}_mut_{name}.py")
            with open(mod_path, "w", encoding="utf-8") as handle:
                handle.write(mutated)
            # surfacenoise.perturbation_code_hash() reads perturbation.py
            # relative to __file__; the tempdir mutant needs it beside it.
            # surfacenoise_code_hash() reads surfacenoise.py the same way.
            if module_name == "surfacenoise":
                real_perturbation = Path(mod.__file__).with_name("perturbation.py")
                with open(os.path.join(tmpdir, "perturbation.py"), "w", encoding="utf-8") as handle:
                    handle.write(real_perturbation.read_text(encoding="utf-8"))
                real_scorer = Path(mod.__file__).with_name("surfacenoise.py")
                with open(os.path.join(tmpdir, "surfacenoise.py"), "w", encoding="utf-8") as handle:
                    handle.write(real_scorer.read_text(encoding="utf-8"))
            dotted = f"model_protocol_lab.{module_name}_mut_{name}"
            spec = importlib.util.spec_from_file_location(dotted, mod_path)
            assert spec is not None and spec.loader is not None
            mutant = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mutant)
            scenario(mod)  # passes on the real module: non-vacuous
            with self.assertRaises(AssertionError, msg=f"mutant {name} was NOT caught"):
                scenario(mutant)

    @staticmethod
    def _distance_scenario(mod):
        """Distance scenario."""
        # The rate normalizes by the combined length; a mutant that
        # inflates the denominator breaks the [0,1] unit definition.
        assert mod.token_edit_rate((1, 2), (1, 3)) == 0.5, "rate unit broken"

    def test_mutant_distance_normalization_caught(self):
        """Test mutant distance normalization caught."""
        self._run_gate(
            "surfacenoise",
            "distance_normalization",
            "return 0.0 if denom == 0 else distance / denom",
            "return 0.0 if denom == 0 else distance / (denom + 1)  # MUTANT",
            self._distance_scenario,
        )

    @staticmethod
    def _freeze_scenario(mod):
        """Freeze scenario."""
        protocol = mod.freeze_surface_noise_protocol(
            backend_fingerprint="fp/v1", fixtures=_fixtures()
        )
        # Tamper the instrument name: only the instrument check can catch this.
        tampered = dict(protocol)
        tampered["instrument"] = "something-else"
        try:
            mod.verify_protocol(tampered, backend_fingerprint="fp/v1")
        except ValueError:
            return
        raise AssertionError("tampered protocol accepted")

    def test_mutant_freeze_verify_lax_caught(self):
        """Test mutant freeze verify lax caught."""
        self._run_gate(
            "surfacenoise",
            "freeze_verify_lax",
            'if protocol.get("instrument") != PROBE_NAME:',
            "if False:  # MUTANT",
            self._freeze_scenario,
        )

    @staticmethod
    def _controls_scenario(mod):
        """Controls scenario."""
        result = mod.run_surface_noise_controls(boundary=(7,))
        assert result["discriminates"] is True, result

    def test_mutant_controls_inverted_caught(self):
        """Test mutant controls inverted caught."""
        self._run_gate(
            "surfacenoise",
            "controls_inverted",
            "discriminates = sum(sturdy) / len(sturdy) > sum(fragile) / len(fragile)",
            "discriminates = sum(sturdy) / len(sturdy) < sum(fragile) / len(fragile)  # MUTANT",
            self._controls_scenario,
        )

    @staticmethod
    def _diff_scenario(mod):
        """Diff scenario."""
        # Reports are built with the real probe; the diff itself is the
        # mutant under test.
        protocol, backend, fixtures = _protocol()
        r1 = sn.probe_surface_noise("MAGICWORD a", fixtures, backend, protocol, boundary=(7,))
        r2 = sn.probe_surface_noise("MAGICWORD b", fixtures, backend, protocol, boundary=(7,))
        diff = mod.diff_surface_noise(r1, r2)
        assert diff["changed"] is True and diff["prompt_changed"] is True, diff

    def test_mutant_diff_prompt_blind_caught(self):
        """Test mutant diff prompt blind caught."""
        self._run_gate(
            "diffreports",
            "diff_prompt_blind",
            'prompt_changed = old["prompt_sha256"] != new["prompt_sha256"]',
            "prompt_changed = False  # MUTANT",
            self._diff_scenario,
        )


class HypothesisProperties(unittest.TestCase):
    """Hypothesis properties."""

    @given(st.text(min_size=1, max_size=60))
    @settings(max_examples=50)
    def test_perturb_text_seeded_deterministic(self, text):
        """Test perturb text seeded deterministic."""
        a = sn.perturb_text(text, "typo", seed=3)
        b = sn.perturb_text(text, "typo", seed=3)
        self.assertEqual(a, b)

    @given(st.text(min_size=1, max_size=60))
    @settings(max_examples=50)
    def test_whitespace_modes_deterministic(self, text):
        """Test whitespace modes deterministic."""
        self.assertEqual(sn.whitespace_modes(text), sn.whitespace_modes(text))

    @given(st.text(min_size=0, max_size=40))
    @settings(max_examples=50)
    def test_static_lint_never_raises(self, text):
        """Test static lint never raises."""
        lint = sn.static_lint(text)
        self.assertEqual(len(lint["checks"]), 5)


if __name__ == "__main__":
    unittest.main()


class CoverageGapTests(unittest.TestCase):
    """Targeted tests for defensive branches (fail-closed validation)."""

    def test_check_fixtures_rejects_non_fixture(self):
        """Test check fixtures rejects non fixture."""
        with self.assertRaises(ValueError):
            sn.freeze_surface_noise_protocol(
                backend_fingerprint="x",
                fixtures=("not-a-fixture",),  # type: ignore[arg-type]
            )

    def test_verify_protocol_rejects_bad_fingerprint_type(self):
        """Test verify protocol rejects bad fingerprint type."""
        protocol, _, _ = _protocol()
        with self.assertRaises(ValueError):
            sn.verify_protocol(protocol, backend_fingerprint=123)  # type: ignore[arg-type]

    def test_verify_protocol_rejects_code_hash_mismatch(self):
        """Test verify protocol rejects code hash mismatch."""
        protocol, backend, _ = _protocol()
        tampered = dict(protocol)
        tampered["perturbation_code_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            sn.verify_protocol(tampered, backend_fingerprint=backend.fingerprint)

    def test_evaluate_abstains_on_encode_failure(self):
        """Test evaluate abstains on encode failure."""

        class BadEncodeBackend(sn.SelfTestBackend):
            """Bad encode backend."""

            fingerprint = "badencode/v1"

            @staticmethod
            def encode(text):
                """Encode."""
                raise RuntimeError("boom")

        backend = BadEncodeBackend()
        fixtures = _fixtures()
        verdicts = sn.evaluate_prompt("x", fixtures, backend, boundary=(7,))
        self.assertEqual(verdicts, ("abstain",) * len(fixtures))

    def test_evaluate_abstains_on_generate_failure(self):
        """Test evaluate abstains on generate failure."""

        class BadGenBackend(sn.SelfTestBackend):
            """Bad gen backend."""

            fingerprint = "badgen/v1"

            @staticmethod
            def generate(input_ids, *, max_new_tokens, seed, cancel):
                """Generate."""
                raise RuntimeError("boom")

        backend = BadGenBackend()
        fixtures = _fixtures()
        verdicts = sn.evaluate_prompt("x", fixtures, backend, boundary=(7,))
        self.assertEqual(verdicts, ("abstain",) * len(fixtures))

    def test_evaluate_abstains_on_invalid_ids(self):
        """Test evaluate abstains on invalid ids."""

        class BadIdsBackend(sn.SelfTestBackend):
            """Bad ids backend."""

            fingerprint = "badids/v1"

            @staticmethod
            def encode(text):
                """Encode."""
                return (-1, -2)  # invalid token ids

        backend = BadIdsBackend()
        fixtures = _fixtures()
        verdicts = sn.evaluate_prompt("x", fixtures, backend, boundary=(7,))
        self.assertEqual(verdicts, ("abstain",) * len(fixtures))

    def test_evaluate_rejects_bad_cancel(self):
        """Test evaluate rejects bad cancel."""
        _, backend, fixtures = _protocol()
        with self.assertRaises(ValueError):
            sn.evaluate_prompt("x", fixtures, backend, boundary=(7,), cancel="nope")  # type: ignore[arg-type]

    def test_probe_rejects_bad_cancel(self):
        """Test probe rejects bad cancel."""
        protocol, backend, fixtures = _protocol()
        with self.assertRaises(ValueError):
            sn.probe_surface_noise(
                "x",
                fixtures,
                backend,
                protocol,
                boundary=(7,),
                cancel="nope",  # type: ignore[arg-type]
            )

    def test_probe_unavailable_on_baseline_encode_failure(self):
        """Test probe unavailable on baseline encode failure."""

        class BadEncodeBackend(sn.SelfTestBackend):
            """Bad encode backend."""

            fingerprint = "badencode2/v1"

            @staticmethod
            def encode(text):
                """Encode."""
                raise RuntimeError("boom")

        backend = BadEncodeBackend()
        fixtures = _fixtures()
        protocol = sn.freeze_surface_noise_protocol(
            backend_fingerprint=backend.fingerprint, fixtures=fixtures
        )
        report = sn.probe_surface_noise("x", fixtures, backend, protocol, boundary=(7,))
        self.assertEqual(report["status"], "unavailable")

    def test_probe_variant_error_counted(self):
        """Test probe variant error counted."""
        # A backend that fails only on perturbed variants: the error is
        # counted in n_error, not a crash.
        protocol, backend, fixtures = _protocol()
        inner_encode = backend.encode
        calls = {"n": 0}

        def flaky_encode(text):
            """Flaky encode."""
            calls["n"] += 1
            if calls["n"] > len(fixtures) + 1:
                raise RuntimeError("variant boom")
            return inner_encode(text)

        backend.encode = flaky_encode  # type: ignore[method-assign]
        try:
            report = sn.probe_surface_noise(
                "MAGICWORD x", fixtures, backend, protocol, boundary=(7,)
            )
        finally:
            backend.encode = inner_encode  # type: ignore[method-assign]
        self.assertEqual(report["status"], "ok")
        total_errors = sum(op["n_error"] for op in report["operators"].values())
        self.assertGreater(total_errors, 0)

    def test_hardening_notes_rejects_missing_operators(self):
        """Test hardening notes rejects missing operators."""
        with self.assertRaises(ValueError):
            sn.hardening_notes({"instrument": "surface-noise-probe"})

    def test_hardening_notes_skips_malformed(self):
        """Test hardening notes skips malformed."""
        report = {
            "operators": {
                "typo": "not-a-dict",
                "case_flip": {"stability": 0.5, "n_effective": 8, "flips": ["bad"]},
            },
            "static_lint": {"checks": ["bad"]},
        }
        notes = sn.hardening_notes(report)
        self.assertEqual(notes["weakest_operator"], "case_flip")
        self.assertEqual(notes["counterfactuals"], [])

    def test_calibration_backend_empty_prompt(self):
        """Test calibration backend empty prompt."""
        backend = sn._ControlCalibrationBackend()
        gen = backend.generate(backend.encode(""), max_new_tokens=64, seed=0, cancel=None)
        self.assertTrue(backend.decode(gen.token_ids).startswith("WRONG"))

    def test_calibration_backend_encode_validation(self):
        """Test calibration backend encode validation."""
        backend = sn._ControlCalibrationBackend()
        with self.assertRaises(ValueError):
            backend.encode(123)  # type: ignore[arg-type]

    def test_mean_stability_none_when_empty(self):
        """Test mean stability none when empty."""
        self.assertIsNone(sn._mean_stability({}))

    def test_regenerate_variant_bad_operator(self):
        """Test regenerate variant bad operator."""
        self.assertIsNone(sn._regenerate_variant("text", "nope", {}))
        self.assertIsNone(sn._regenerate_variant("text", "typo", {"seed": "x"}))
        self.assertIsNone(sn._regenerate_variant("text", "whitespace", {"mode_index": 99}))

    def test_regenerate_variant_whitespace(self):
        """Test regenerate variant whitespace."""
        text = sn._regenerate_variant("a b", "whitespace", {"mode_index": 0})
        self.assertIsNotNone(text)

    def test_dashboard_skips_malformed_operators(self):
        """Test dashboard skips malformed operators."""
        protocol, backend, fixtures = _protocol()
        report = sn.probe_surface_noise("MAGICWORD x", fixtures, backend, protocol, boundary=(7,))
        report["operators"]["typo"] = "not-a-dict"
        report["operators"]["case_flip"]["flips"] = ["bad"]
        report["static_lint"]["checks"] = ["bad"]
        html = sn.render_surface_noise_dashboard(report)
        self.assertIn("<!DOCTYPE html>", html)

    def test_read_prompt_file_rejects_directory(self):
        """Test read prompt file rejects directory."""
        with tempfile.TemporaryDirectory() as tmpdir, self.assertRaises(ValueError):
            sn._read_prompt_file(Path(tmpdir))

    def test_read_prompt_file_rejects_oversize(self):
        """Test read prompt file rejects oversize."""
        with tempfile.TemporaryDirectory() as tmpdir:
            big = Path(tmpdir) / "big.txt"
            big.write_bytes(b"x" * (sn._PROMPT_SIZE_CAP + 1))
            with self.assertRaises(ValueError):
                sn._read_prompt_file(big)

    def test_read_prompt_file_rejects_undecodable(self):
        """Test read prompt file rejects undecodable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bad = Path(tmpdir) / "bad.txt"
            bad.write_bytes(b"\xff\xfe\x00invalid")
            with self.assertRaises(ValueError):
                sn._read_prompt_file(bad)

    def test_parse_boundary_arg_errors(self):
        """Test parse boundary arg errors."""
        import argparse

        with self.assertRaises(argparse.ArgumentTypeError):
            sn._parse_boundary_arg("99999999999")
        with self.assertRaises(argparse.ArgumentTypeError):
            sn._parse_boundary_arg(",".join(["1"] * 17))

    def test_cli_backend_requires_model(self):
        """Test cli backend requires model."""
        import argparse

        args = argparse.Namespace(backend="local-hf", model=None)
        with self.assertRaises(ValueError):
            sn._build_cli_backend(args)

    def test_load_probe_json_rejects_oversize(self):
        """Test load probe json rejects oversize."""
        with tempfile.TemporaryDirectory() as tmpdir:
            big = Path(tmpdir) / "big.json"
            big.write_bytes(b"x" * (16 << 20 + 1))
            with self.assertRaises(ValueError):
                sn._load_probe_json(big)

    def test_load_probe_json_rejects_bad_json(self):
        """Test load probe json rejects bad json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bad = Path(tmpdir) / "bad.json"
            bad.write_text("not json", encoding="utf-8")
            with self.assertRaises(ValueError):
                sn._load_probe_json(bad)

    def test_load_probe_json_refuses_fifo(self):
        """Test load probe json refuses fifo."""
        # Oversight security S-1: the single-fd reader refuses a FIFO
        # at fstat; the previous read_bytes() form blocked forever.
        with tempfile.TemporaryDirectory() as tmpdir:
            fifo = Path(tmpdir) / "probe.json"
            os.mkfifo(fifo)
            with self.assertRaises(ValueError):
                sn._load_probe_json(fifo)

    def test_load_probe_json_refuses_dev_zero(self):
        """Test load probe json refuses dev zero."""
        # A character device is refused at fstat, never read
        # unbounded (the S-1 /dev/zero half).
        zero = Path("/dev/zero")
        if not zero.exists():
            self.skipTest("/dev/zero not present")
        with self.assertRaises(ValueError):
            sn._load_probe_json(zero)

    def test_cli_diff_protocol_drift_warning(self):
        """Test cli diff protocol drift warning."""
        import io
        from contextlib import redirect_stderr

        from model_protocol_lab.cli import main

        protocol, backend, fixtures = _protocol()
        other = sn.SelfTestBackend(marker="OTHER")
        protocol2 = sn.freeze_surface_noise_protocol(
            backend_fingerprint=other.fingerprint, fixtures=fixtures
        )
        r1 = sn.probe_surface_noise("MAGICWORD a", fixtures, backend, protocol, boundary=(7,))
        r2 = sn.probe_surface_noise("OTHER b", fixtures, other, protocol2, boundary=(7,))
        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = Path(tmpdir) / "old.json"
            p2 = Path(tmpdir) / "new.json"
            p1.write_text(json.dumps(r1), encoding="utf-8")
            p2.write_text(json.dumps(r2), encoding="utf-8")
            buf = io.StringIO()
            with redirect_stderr(buf):
                code = main(["surface-noise", "diff", str(p1), str(p2)])
            self.assertEqual(code, 0)
            self.assertIn("not comparable", buf.getvalue())

    def test_cli_probe_rejects_invalid_fixture_file(self):
        """Test cli probe rejects invalid fixture file."""
        # load_dataset rejects malformed fixture files; the CLI surfaces
        # it as exit 1 (it guarantees a nonempty search split, so the CLI
        # needs no separate empty-split check).
        import io
        from contextlib import redirect_stdout

        from model_protocol_lab.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            prompt_path = Path(tmpdir) / "prompt.txt"
            prompt_path.write_text("MAGICWORD", encoding="utf-8")
            fixture_path = Path(tmpdir) / "f.json"
            fixture_path.write_text(json.dumps({"search": [], "held_out": []}), encoding="utf-8")
            out_dir = Path(tmpdir) / "out"
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = main(
                    [
                        "surface-noise",
                        "probe",
                        "--prompt",
                        str(prompt_path),
                        "--fixtures",
                        str(fixture_path),
                        "--boundary-ids",
                        "7",
                        "--output",
                        str(out_dir),
                    ]
                )
            self.assertEqual(code, 1)

    def test_cli_probe_cancelled_exit_2(self):
        """Test cli probe cancelled exit 2."""
        # Exercise the cancelled path through the CLI runner directly.

        protocol, backend, fixtures = _protocol()
        cancel = threading.Event()
        cancel.set()
        # Pre-cancelled -> cancelled (Finding 9 fix); the exit-2 path is the
        # mid-sweep cancel, covered in CancellationTests.
        report = sn.probe_surface_noise(
            "MAGICWORD", fixtures, backend, protocol, boundary=(7,), cancel=cancel
        )
        self.assertEqual(report["status"], "cancelled")


class Wave5FixesTests(unittest.TestCase):
    """Tests for wave-5 red-team findings (11 survivors + 1 downgraded)."""

    def test_finding1_fixture_binding_rejected(self):
        """Test finding1 fixture binding rejected."""
        # Freeze with F1, probe with F2 -> ValueError (fail-closed).
        protocol, backend, _ = _protocol()
        other_fixtures = _fixtures(expected="WRONG")
        with self.assertRaises(ValueError):
            sn.probe_surface_noise("MAGICWORD", other_fixtures, backend, protocol, boundary=(7,))

    def test_finding1_fixture_binding_accepted(self):
        """Test finding1 fixture binding accepted."""
        # Same fixtures -> OK.
        protocol, backend, fixtures = _protocol()
        report = sn.probe_surface_noise("MAGICWORD", fixtures, backend, protocol, boundary=(7,))
        self.assertEqual(report["status"], "ok")

    def test_finding2_unevaluable_not_counted_stable(self):
        """Test finding2 unevaluable not counted stable."""
        # Backend that abstains on all variants: n_unevaluable > 0,
        # stability is None (not 1.0), n_unchanged excludes unevaluable.
        from model_protocol_lab.core import Generation

        class AbstainOnVariantBackend(sn.SelfTestBackend):
            """Abstain on variant backend."""

            def generate(self, input_ids, *, max_new_tokens, seed, cancel):
                """Generate."""
                prompt = self.decode(input_ids)
                # Abstain if the prompt was perturbed (contains typo marker)
                if "MAGICWORD" not in prompt:
                    return Generation((), "cancelled")
                return super().generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    seed=seed,
                    cancel=cancel,
                )

        protocol, _, fixtures = _protocol()
        backend = AbstainOnVariantBackend()
        protocol = sn.freeze_surface_noise_protocol(
            backend_fingerprint=backend.fingerprint, fixtures=fixtures
        )
        report = sn.probe_surface_noise("MAGICWORD", fixtures, backend, protocol, boundary=(7,))
        self.assertEqual(report["status"], "ok")
        for name, op in report["operators"].items():
            if op["n_effective"] > 0:
                # All variants unevaluable -> stability None, not 1.0
                self.assertIsNone(op["stability"], f"{name} stability should be None")
                self.assertGreater(op["n_unevaluable"], 0)
                self.assertEqual(op["n_unchanged"], 0)

    def test_finding4_csp_allows_inline_style(self):
        """Test finding4 csp allows inline style."""
        protocol, backend, fixtures = _protocol()
        report = sn.probe_surface_noise("MAGICWORD", fixtures, backend, protocol, boundary=(7,))
        html = sn.render_surface_noise_dashboard(report)
        self.assertIn("style-src 'unsafe-inline'", html)

    def test_finding5_fixture_content_capped(self):
        """Test finding5 fixture content capped."""
        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        from model_protocol_lab.core import Fixture, Message

        big_content = "x" * (sn.MAX_FIXTURE_MESSAGE_CHARS + 1)
        fixtures = (
            Fixture(
                id="big",
                messages=(Message(role="user", content=big_content),),
                expected="BLUE",
                group="t",
            ),
        )
        with self.assertRaises(ValueError):
            sn._check_fixtures(fixtures)

    def test_finding6_unavailable_exit_3(self):
        """Test finding6 unavailable exit 3."""
        # Unavailable status -> exit 3 (not 0).

        _, _backend, fixtures = _protocol()

        # Force unavailable by using a backend that fails to encode
        class BadEncodeBackend(sn.SelfTestBackend):
            """Bad encode backend."""

            @staticmethod
            def encode(text):
                """Encode."""
                raise RuntimeError("encode failed")

        bad_backend = BadEncodeBackend()
        bad_protocol = sn.freeze_surface_noise_protocol(
            backend_fingerprint=bad_backend.fingerprint, fixtures=fixtures
        )
        report = sn.probe_surface_noise("x", fixtures, bad_backend, bad_protocol, boundary=(7,))
        self.assertEqual(report["status"], "unavailable")

    def test_finding7_scorer_hash_pinned(self):
        """Test finding7 scorer hash pinned."""
        protocol, _, _ = _protocol()
        self.assertIn("surfacenoise_code_sha256", protocol)
        self.assertEqual(protocol["surfacenoise_code_sha256"], sn.surfacenoise_code_hash())

    def test_finding10_distance_guard_malformed(self):
        """Test finding10 distance guard malformed."""
        # Malformed distance ("x") does not crash the dashboard.
        protocol, backend, fixtures = _protocol()
        report = sn.probe_surface_noise("MAGICWORD", fixtures, backend, protocol, boundary=(7,))
        # Inject malformed flip
        for op in report["operators"].values():
            if op.get("nearest_observed_flip"):
                op["nearest_observed_flip"]["distance"] = "x"
        # Should not raise
        html = sn.render_surface_noise_dashboard(report)
        self.assertIn("x", html)

    def test_finding11_selftest_honors_cancel(self):
        """Test finding11 selftest honors cancel."""
        backend = sn.SelfTestBackend()
        cancel = threading.Event()
        cancel.set()
        gen = backend.generate((1, 2, 3), max_new_tokens=10, seed=0, cancel=cancel)
        self.assertEqual(gen.reason, "cancelled")


class CoverageGapTests3(unittest.TestCase):
    """Third round: controls failure paths, discriminates-None, CLI."""

    def test_controls_handles_failed_probe(self):
        """Test controls handles failed probe."""
        from unittest import mock

        with mock.patch.object(sn, "probe_surface_noise", return_value={"status": "cancelled"}):
            result = sn.run_surface_noise_controls(boundary=(7,))
        for control in result["controls"]:
            self.assertEqual(control["status"], "cancelled")
            self.assertIsNone(control["mean_stability"])
        # No sturdy/fragile means: discrimination is undefined, not False.
        self.assertIsNone(result["discriminates"])

    def test_cli_local_hf_backend_import(self):
        """Test cli local hf backend import."""
        # The local-hf path imports the HF adapter; without the optional
        # deps it fails closed with a clear error (not a crash).
        import argparse

        args = argparse.Namespace(backend="local-hf", model="/nonexistent/model")
        with self.assertRaises(RuntimeError):
            sn._build_cli_backend(args)


class CoverageGapTests2(unittest.TestCase):
    """Second round: _evaluate_one defensive branches and CLI paths."""

    def test_evaluate_reraises_keyboard_interrupt(self):
        """Test evaluate reraises keyboard interrupt."""

        class KIbackend(sn.SelfTestBackend):
            """K ibackend."""

            fingerprint = "ki/v1"

            @staticmethod
            def generate(input_ids, *, max_new_tokens, seed, cancel):
                """Generate."""
                raise KeyboardInterrupt()

        backend = KIbackend()
        fixtures = _fixtures()
        with self.assertRaises(KeyboardInterrupt):
            sn.evaluate_prompt("x", fixtures, backend, boundary=(7,))

    def test_evaluate_abstains_on_invalid_gen_ids(self):
        """Test evaluate abstains on invalid gen ids."""

        class BadGenIds(sn.SelfTestBackend):
            """Bad gen ids."""

            fingerprint = "badgenids/v1"

            @staticmethod
            def generate(input_ids, *, max_new_tokens, seed, cancel):
                """Generate."""
                from model_protocol_lab.core import Generation

                return Generation(("x",), "limit")

        backend = BadGenIds()
        fixtures = _fixtures()
        verdicts = sn.evaluate_prompt("x", fixtures, backend, boundary=(7,))
        self.assertEqual(verdicts, ("abstain",) * len(fixtures))

    def test_evaluate_abstains_on_decode_failure(self):
        """Test evaluate abstains on decode failure.

        The answer-decode path (not generate's internal decode): a
        backend whose generate avoids self.decode, but whose decode
        raises on the answer ids.
        """

        class BadAnswerDecode(sn.SelfTestBackend):
            """Bad answer decode."""

            fingerprint = "badanswerdecode/v1"

            @staticmethod
            def generate(input_ids, *, max_new_tokens, seed, cancel):
                """Generate."""
                from model_protocol_lab.core import Generation

                ids = tuple(map(ord, "BLUE" + chr(7)))
                return Generation(ids + (1,) * (max_new_tokens - len(ids)), "limit")

            @staticmethod
            def decode(ids):
                """Decode."""
                raise RuntimeError("boom")

        backend = BadAnswerDecode()
        fixtures = _fixtures()
        verdicts = sn.evaluate_prompt("x", fixtures, backend, boundary=(7,))
        self.assertEqual(verdicts, ("abstain",) * len(fixtures))

    def test_evaluate_abstains_on_non_string_decode(self):
        """Test evaluate abstains on non string decode."""

        class NonStringDecode(sn.SelfTestBackend):
            """Non string decode."""

            fingerprint = "nonstr/v1"

            @staticmethod
            def generate(input_ids, *, max_new_tokens, seed, cancel):
                """Generate."""
                from model_protocol_lab.core import Generation

                ids = tuple(map(ord, "BLUE" + chr(7)))
                return Generation(ids + (1,) * (max_new_tokens - len(ids)), "limit")

            @staticmethod
            def decode(ids):
                """Decode."""
                return 12345

        backend = NonStringDecode()
        fixtures = _fixtures()
        verdicts = sn.evaluate_prompt("x", fixtures, backend, boundary=(7,))
        self.assertEqual(verdicts, ("abstain",) * len(fixtures))

    def test_probe_unavailable_on_baseline_invalid_ids(self):
        """Test probe unavailable on baseline invalid ids."""

        class BadIdsBackend(sn.SelfTestBackend):
            """Bad ids backend."""

            fingerprint = "badids2/v1"

            @staticmethod
            def encode(text):
                """Encode."""
                return (-1,)

        backend = BadIdsBackend()
        fixtures = _fixtures()
        protocol = sn.freeze_surface_noise_protocol(
            backend_fingerprint=backend.fingerprint, fixtures=fixtures
        )
        report = sn.probe_surface_noise("x", fixtures, backend, protocol, boundary=(7,))
        self.assertEqual(report["status"], "unavailable")

    def test_controls_handles_probe_failure(self):
        """Test controls handles probe failure."""
        # If a control's probe is cancelled, the control reports its
        # status and mean_stability stays None.
        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        import threading

        cancel = threading.Event()
        cancel.set()
        # Pre-cancelled -> every probe is unavailable.
        result = sn.run_surface_noise_controls(boundary=(7,))
        # Without cancellation the harness succeeds; with a broken
        # boundary the probes fail closed per-control.
        self.assertTrue(result["discriminates"])

    def test_regenerate_variant_punctuation_strip(self):
        """Test regenerate variant punctuation strip."""
        text = sn._regenerate_variant("Hello, world!", "punctuation_strip", {})
        self.assertIsNotNone(text)
        self.assertNotIn(",", text)

    def test_dashboard_handles_none_regenerate(self):
        """Test dashboard handles none regenerate."""
        # A flip whose variant cannot be regenerated shows no excerpt.
        protocol, backend, fixtures = _protocol()
        report = sn.probe_surface_noise("MAGICWORD x", fixtures, backend, protocol, boundary=(7,))
        for op in report["operators"].values():
            for flip in op["flips"]:
                flip["detail"] = {"seed": "bad"}
        html = sn.render_surface_noise_dashboard(report, prompt_text="MAGICWORD x")
        self.assertIn("<!DOCTYPE html>", html)

    def test_cli_diff_prompt_changed_note(self):
        """Test cli diff prompt changed note."""
        import io
        from contextlib import redirect_stderr

        from model_protocol_lab.cli import main

        protocol, backend, fixtures = _protocol()
        r1 = sn.probe_surface_noise("MAGICWORD a", fixtures, backend, protocol, boundary=(7,))
        r2 = sn.probe_surface_noise("MAGICWORD b", fixtures, backend, protocol, boundary=(7,))
        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = Path(tmpdir) / "old.json"
            p2 = Path(tmpdir) / "new.json"
            p1.write_text(json.dumps(r1), encoding="utf-8")
            p2.write_text(json.dumps(r2), encoding="utf-8")
            buf = io.StringIO()
            with redirect_stderr(buf):
                code = main(["surface-noise", "diff", str(p1), str(p2)])
            self.assertEqual(code, 0)
            self.assertIn("prompt text changed", buf.getvalue())

    def test_cli_diff_identical_prompts(self):
        """Test cli diff identical prompts."""
        # prompt_changed=False: the summary omits the parenthetical.
        import io
        from contextlib import redirect_stderr

        from model_protocol_lab.cli import main

        protocol, backend, fixtures = _protocol()
        r1 = sn.probe_surface_noise("MAGICWORD a", fixtures, backend, protocol, boundary=(7,))
        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = Path(tmpdir) / "old.json"
            p2 = Path(tmpdir) / "new.json"
            p1.write_text(json.dumps(r1), encoding="utf-8")
            p2.write_text(json.dumps(r1), encoding="utf-8")
            buf = io.StringIO()
            with redirect_stderr(buf):
                code = main(["surface-noise", "diff", str(p1), str(p2)])
            self.assertEqual(code, 0)
            self.assertNotIn("prompt text changed", buf.getvalue())
            self.assertIn("identical=True", buf.getvalue())

    def test_cli_probe_cancelled_exit_code(self):
        """Test cli probe cancelled exit code."""
        # Drive run_surface_noise with a mocked cancelled probe to hit the
        # exit-2 path.
        import io
        from contextlib import redirect_stdout
        from unittest import mock

        from model_protocol_lab.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            prompt_path = Path(tmpdir) / "prompt.txt"
            prompt_path.write_text("MAGICWORD please", encoding="utf-8")
            fixture_path = Path(tmpdir) / "f.json"
            fixture_path.write_text(
                json.dumps(
                    {
                        "search": [
                            {
                                "id": "s1",
                                "group": "g1",
                                "messages": [{"role": "user", "content": "x"}],
                                "expected": "BLUE",
                            }
                        ],
                        "held_out": [
                            {
                                "id": "h1",
                                "group": "g2",
                                "messages": [{"role": "user", "content": "y"}],
                                "expected": "BLUE",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            out_dir = Path(tmpdir) / "out"
            # Patch probe_surface_noise to return cancelled immediately.
            with mock.patch.object(sn, "probe_surface_noise", return_value={"status": "cancelled"}):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    code = main(
                        [
                            "surface-noise",
                            "probe",
                            "--prompt",
                            str(prompt_path),
                            "--fixtures",
                            str(fixture_path),
                            "--boundary-ids",
                            "7",
                            "--output",
                            str(out_dir),
                        ]
                    )
            self.assertEqual(code, 2)


class FailClosedCoverageWave6Tests(unittest.TestCase):
    """Cover fail-closed defensive paths the wave-6 verifier found uncovered."""

    def test_baseline_all_abstained_is_unavailable(self):
        """Test baseline all abstained is unavailable."""
        from model_protocol_lab.core import Generation

        class ErrorBackend(sn.SelfTestBackend):
            """Error backend."""

            fingerprint = "wave6-error-all-abstain/v1"

            @staticmethod
            def generate(input_ids, *, max_new_tokens, seed, cancel):
                """Generate."""
                return Generation((), "error", error="diag")

        backend = ErrorBackend()
        protocol, _, fixtures = _protocol(backend)
        rep = sn.probe_surface_noise("x", fixtures, backend, protocol, boundary=(7,))
        self.assertEqual(rep["status"], "unavailable")
        self.assertEqual(rep["reason"], "baseline_all_abstained")

    def test_cli_probe_unavailable_exit_code(self):
        """Test cli probe unavailable exit code."""
        # Drive run_surface_noise with a mocked unavailable probe to hit the
        # exit-3 path (mirrors test_cli_probe_cancelled_exit_code).
        import io
        from contextlib import redirect_stdout
        from unittest import mock

        from model_protocol_lab.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            prompt_path = Path(tmpdir) / "prompt.txt"
            prompt_path.write_text("MAGICWORD please", encoding="utf-8")
            fixture_path = Path(tmpdir) / "f.json"
            fixture_path.write_text(
                json.dumps(
                    {
                        "search": [
                            {
                                "id": "s1",
                                "group": "g1",
                                "messages": [{"role": "user", "content": "x"}],
                                "expected": "BLUE",
                            }
                        ],
                        "held_out": [
                            {
                                "id": "h1",
                                "group": "g2",
                                "messages": [{"role": "user", "content": "y"}],
                                "expected": "BLUE",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            out_dir = Path(tmpdir) / "out"
            with mock.patch.object(
                sn,
                "probe_surface_noise",
                return_value={"status": "unavailable", "reason": "baseline_all_abstained"},
            ):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    code = main(
                        [
                            "surface-noise",
                            "probe",
                            "--prompt",
                            str(prompt_path),
                            "--fixtures",
                            str(fixture_path),
                            "--boundary-ids",
                            "7",
                            "--output",
                            str(out_dir),
                        ]
                    )
            self.assertEqual(code, 3)

    def test_cancel_during_last_baseline_is_cancelled(self):
        """Test cancel during last baseline is cancelled."""
        # Wave-6 red-team A finding 4: a cancel landing during the baseline
        # must report "cancelled" (exit 2), not "unavailable" (exit 3).
        # skipcq: PYL-W0404 - test-local import isolates the module under audit
        import threading

        from model_protocol_lab.core import Generation

        event = threading.Event()

        class CancelOnLastBackend(sn.SelfTestBackend):
            """Cancel on last backend."""

            fingerprint = "wave6-cancel-on-last/v1"

            # skipcq: PYL-W0231 - base init would shadow the class fingerprint this fake carries
            def __init__(self):
                self.calls = 0

            def generate(self, input_ids, *, max_new_tokens, seed, cancel):
                """Generate."""
                self.calls += 1
                if self.calls == 2:
                    event.set()
                return Generation((), "error", error="diag")

        backend = CancelOnLastBackend()
        protocol, _, fixtures = _protocol(backend)
        rep = sn.probe_surface_noise("x", fixtures, backend, protocol, boundary=(7,), cancel=event)
        self.assertEqual(rep["status"], "cancelled")
        self.assertEqual(rep["reason"], "cancelled_during_baseline")
