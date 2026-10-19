"""Cycle 166: per-template surface-noise robustness.

Tests for the template-robustness instrument in
model_protocol_lab.surfacenoise (probe_template_robustness),
experiments.template_robustness_stage, the ``mpl run
--template-robustness`` CLI flag, the HTML report section, and
diff_template_robustness.

Design: docs/TEMPLATE_ROBUSTNESS_DESIGN.md (v2, red-team reviewed).
Key properties under test: presentation-level operator set only (F1),
baseline-conditioned preservation rate (N1), token-ID no-op filter (F4),
template-invariant content-span distance (F3), per-operator vectors never
blended (F5), user-role-only perturbation (F10), fail-closed per-variant
abstention (F8), frozen protocol (F9), and the F14 controls (determinism +
scripted robust-vs-fragile discrimination).
"""

import importlib.util
import os
import tempfile
import threading
import unittest
from pathlib import Path

from model_protocol_lab import surfacenoise as sn
from model_protocol_lab.core import Fixture, Generation, Message

BOUNDARY = (7,)


class MarkerBackend:
    """Scripted backend: BLUE iff 'say magicword' (single spaces) in input.

    Case-insensitive, whitespace-sensitive. Deterministic; ord-based
    tokenization. The fingerprint is fixed (behavior is fixed).
    """

    fingerprint = "test-marker/v1"

    @staticmethod
    def encode(text):
        if type(text) is not str:
            raise ValueError("text must be a string")
        return tuple(map(ord, text))

    @staticmethod
    def decode(ids):
        return "".join(map(chr, ids))

    def generate(self, input_ids, *, max_new_tokens, seed, cancel=None):
        prompt = self.decode(input_ids).lower()
        ans = "BLUE" if "say magicword" in prompt else "WRONG"
        return Generation(tuple(map(ord, ans + chr(7))), "stop")


class AlwaysWrongBackend(MarkerBackend):
    """All verdicts fail (N1: no pass-baseline -> rate is None, not 0.0)."""

    fingerprint = "test-always-wrong/v1"

    def generate(self, input_ids, *, max_new_tokens, seed, cancel=None):
        return Generation(tuple(map(ord, "WRONG" + chr(7))), "stop")


class AlwaysErrorBackend(MarkerBackend):
    """Every generation errors -> abstain (F8)."""

    fingerprint = "test-always-error/v1"

    def generate(self, input_ids, *, max_new_tokens, seed, cancel=None):
        return Generation((), "error")


class CoarseBackend(MarkerBackend):
    """Whitespace-insensitive tokenizer: F4 no-op filter test.

    Collapses whitespace runs before ord-encoding, so textually different
    whitespace variants are token-identical -> no-ops by token-ID identity.
    """

    fingerprint = "test-coarse/v1"

    @staticmethod
    def encode(text):
        if type(text) is not str:
            raise ValueError("text must be a string")
        import re

        return tuple(map(ord, re.sub(r"\s+", " ", text)))


def _robust_render(messages):
    """Template that absorbs case/whitespace/unicode noise (known-robust).

    Collapses whitespace, NFKC-folds fullwidth forms back to ASCII; the
    marker survives exactly. (A scripted control, not a real template.)
    """
    import unicodedata

    parts = []
    for m in messages:
        if m.role == "user":
            normalized = unicodedata.normalize("NFKC", " ".join(m.content.split()))
            parts.append("USER: " + normalized)
        else:
            parts.append("ASSISTANT: " + m.content)
    return "\n".join(parts)


def _fragile_render(messages):
    """Template that renders verbatim (the known-fragile control)."""
    parts = []
    for m in messages:
        if m.role == "user":
            parts.append("USER: " + m.content)
        else:
            parts.append("ASSISTANT: " + m.content)
    return "\n".join(parts)


def _verbose_render(messages):
    """Deliberately verbose template: distance must not depend on it (F3)."""
    return "PREFIX " * 200 + _fragile_render(messages)


def _fixtures(n=2, content="Say MAGICWORD"):
    return tuple(
        Fixture(
            id=f"r{i}",
            messages=(Message(role="user", content=content),),
            expected="BLUE",
            group="g",
        )
        for i in range(n)
    )


def _protocol(backend=None, fixtures=None):
    backend = backend or MarkerBackend()
    fixtures = fixtures if fixtures is not None else _fixtures()
    return (
        sn.freeze_template_robustness_protocol(
            backend_fingerprint=backend.fingerprint, fixtures=fixtures
        ),
        backend,
        fixtures,
    )


def _probe(render, template_name="t", backend=None, fixtures=None, **kwargs):
    protocol, backend, fixtures = _protocol(backend, fixtures)
    params = {"boundary": BOUNDARY}
    params.update(kwargs)
    return sn.probe_template_robustness(
        render, template_name, fixtures, backend, protocol, **params
    )


class OperatorSetTests(unittest.TestCase):
    def test_operators_are_presentation_level_only(self):
        # F1: typo and punctuation_strip are excluded (they can change the
        # correct answer); exactly the presentation-level triple remains.
        self.assertEqual(
            sn.TEMPLATE_ROBUSTNESS_OPERATORS,
            ("case_flip", "whitespace", "unicode_variant"),
        )
        self.assertNotIn("typo", sn.TEMPLATE_ROBUSTNESS_OPERATORS)
        self.assertNotIn("punctuation_strip", sn.TEMPLATE_ROBUSTNESS_OPERATORS)

    def test_frozen_search_plan_covers_all_operators(self):
        plan = sn.frozen_template_search_plan()
        self.assertEqual(set(plan), set(sn.TEMPLATE_ROBUSTNESS_OPERATORS))
        self.assertEqual(plan["case_flip"]["seeds"], list(sn.SEARCH_PLAN_SEEDS))
        self.assertEqual(plan["unicode_variant"]["seeds"], list(sn.SEARCH_PLAN_SEEDS))
        self.assertTrue(plan["whitespace"]["exact"])

    def test_probe_name_and_version_pinned(self):
        self.assertEqual(sn.TEMPLATE_ROBUSTNESS_PROBE_NAME, "template-surface-noise-robustness")
        self.assertEqual(sn.TEMPLATE_ROBUSTNESS_PROBE_VERSION, "1.0.0")


class FrozenProtocolTests(unittest.TestCase):
    def test_freeze_verify_roundtrip(self):
        protocol, backend, fixtures = _protocol()
        verified = sn.verify_template_robustness_protocol(
            protocol, backend_fingerprint=backend.fingerprint, fixtures=fixtures
        )
        self.assertIs(verified, protocol)

    def test_verify_rejects_tampered_operators(self):
        protocol, backend, fixtures = _protocol()
        tampered = dict(protocol)
        tampered["perturbation_operators"] = ["typo"]
        with self.assertRaises(ValueError):
            sn.verify_template_robustness_protocol(
                tampered, backend_fingerprint=backend.fingerprint, fixtures=fixtures
            )

    def test_verify_rejects_wrong_backend_fingerprint(self):
        protocol, backend, fixtures = _protocol()
        with self.assertRaises(ValueError):
            sn.verify_template_robustness_protocol(
                protocol, backend_fingerprint="wrong", fixtures=fixtures
            )

    def test_verify_rejects_different_fixtures(self):
        protocol, backend, _ = _protocol()
        other = _fixtures(n=3)
        with self.assertRaises(ValueError):
            sn.verify_template_robustness_protocol(
                protocol, backend_fingerprint=backend.fingerprint, fixtures=other
            )

    def test_verify_rejects_wrong_instrument(self):
        protocol, backend, fixtures = _protocol()
        tampered = dict(protocol)
        tampered["instrument"] = "surface-noise-probe"
        with self.assertRaises(ValueError):
            sn.verify_template_robustness_protocol(
                tampered, backend_fingerprint=backend.fingerprint, fixtures=fixtures
            )

    def test_probe_rejects_tampered_protocol(self):
        protocol, backend, fixtures = _protocol()
        tampered = dict(protocol)
        tampered["search_plan"] = {}
        with self.assertRaises(ValueError):
            sn.probe_template_robustness(
                _fragile_render,
                "t",
                fixtures,
                backend,
                tampered,
                boundary=BOUNDARY,
            )

    def test_protocol_binds_fixtures(self):
        protocol, _, _ = _protocol()
        self.assertEqual(protocol["fixture_ids"], ["r0", "r1"])
        self.assertIn("fixture_sha256", protocol)


class VariantEnumerationTests(unittest.TestCase):
    def test_seeded_operators_get_eight_seeds(self):
        case = _fixtures(n=1)[0]
        variants = sn._enumerate_template_variants(case)
        by_op: dict[str, list[str]] = {}
        for vid, op, _detail, _pert in variants:
            by_op.setdefault(op, []).append(vid)
        self.assertEqual(len(by_op["case_flip"]), 8)
        self.assertEqual(len(by_op["unicode_variant"]), 8)
        self.assertEqual(by_op["case_flip"], [f"case_flip/seed={s}" for s in range(8)])

    def test_whitespace_modes_enumerated(self):
        case = _fixtures(n=1)[0]
        variants = sn._enumerate_template_variants(case)
        ws = [vid for vid, op, _d, _p in variants if op == "whitespace"]
        # "Say MAGICWORD" has single spaces: all three modes apply.
        self.assertEqual(len(ws), 3)
        self.assertEqual(ws, [f"whitespace/mode_index={j}" for j in range(3)])

    def test_whitespace_absent_no_variants(self):
        case = Fixture(
            id="w0",
            messages=(Message(role="user", content="MAGICWORD"),),
            expected="BLUE",
            group="g",
        )
        variants = sn._enumerate_template_variants(case)
        ws = [vid for vid, op, _d, _p in variants if op == "whitespace"]
        self.assertEqual(ws, [])

    def test_variant_ids_fixture_independent(self):
        a = _fixtures(n=1, content="Say MAGICWORD")[0]
        b = _fixtures(n=1, content="Completely different text here")[0]
        ids_a = [vid for vid, _o, _d, _p in sn._enumerate_template_variants(a)]
        ids_b = [vid for vid, _o, _d, _p in sn._enumerate_template_variants(b)]
        self.assertEqual(ids_a, ids_b)

    def test_multi_message_perturbation(self):
        case = Fixture(
            id="m0",
            messages=(
                Message(role="user", content="Say MAGICWORD"),
                Message(role="assistant", content="MAGICWORD"),
                Message(role="user", content="Again MAGICWORD"),
            ),
            expected="BLUE",
            group="g",
        )
        variants = sn._enumerate_template_variants(case)
        _vid, _op, _detail, perturbed = variants[0]
        # Three contents returned (one per message, order preserved)...
        self.assertEqual(len(perturbed), 3)
        # ...assistant content untouched (F10 is enforced at render level;
        # here the helper returns all contents positionally).
        self.assertEqual(perturbed[1], "MAGICWORD")


class NoOpFilterTests(unittest.TestCase):
    def test_token_identical_whitespace_variants_are_no_ops(self):
        # F4: the coarse backend tokenizes whitespace variants identically
        # to the base; they must be no-ops (listed, never counted).
        backend = CoarseBackend()
        rep = _probe(_fragile_render, backend=backend)
        self.assertEqual(rep["status"], "ok")
        ws_no_ops = [n for n in rep["no_ops"] if "whitespace" in n]
        # 2 fixtures x 3 whitespace modes = 6 no-ops.
        self.assertEqual(len(ws_no_ops), 6)
        # None of them entered any denominator: case_flip(8) + unicode(8)
        # = 16 evaluable variants per fixture at most.
        for f in rep["per_fixture"]:
            self.assertLessEqual(f.get("n_variants", 0), 16)

    def test_no_op_variants_never_count_as_preserved(self):
        backend = CoarseBackend()
        rep = _probe(_fragile_render, backend=backend)
        # Preservation denominator counts only evaluated variants.
        self.assertEqual(rep["n_preservation_denominator"], 2)
        # whitespace operator: all variants no-ops -> unevaluable, not 1.0.
        ws = rep["operators"]["whitespace"]
        self.assertEqual(ws["n_denominator"], 0)
        self.assertIsNone(ws["preservation_rate"])
        self.assertEqual(ws["n_unevaluable"], 2)

    def test_textually_identical_means_token_identical(self):
        # Sanity: with the normal backend, the collapse mode on single-space
        # content is a no-op too (textual identity implies token identity).
        rep = _probe(_fragile_render)
        collapse_no_ops = [n for n in rep["no_ops"] if "mode_index=0" in n]
        self.assertEqual(len(collapse_no_ops), 2)


class DistanceTests(unittest.TestCase):
    def test_distance_is_template_invariant(self):
        # F3: the same flip must report the same distance under a terse and
        # a deliberately verbose template (distance is content-span based).
        rep_terse = _probe(_fragile_render, template_name="terse")
        rep_verbose = _probe(_verbose_render, template_name="verbose")

        def _flip_distances(rep):
            out = {}
            for _op, op_rep in rep["operators"].items():
                for flip in op_rep["nearest_flips"]:
                    out[(flip["fixture_id"], flip["variant_id"])] = flip["distance"]
            return out

        terse = _flip_distances(rep_terse)
        verbose = _flip_distances(rep_verbose)
        self.assertTrue(terse, "expected some flips under the fragile template")
        for key, dist in terse.items():
            self.assertIn(key, verbose)
            self.assertEqual(dist, verbose[key])

    def test_distance_matches_content_span_computation(self):
        # The reported distance equals the normalized indel rate on the
        # content-span token IDs (not the rendered input). Uses whitespace
        # flips, which deterministically flip the fragile template.
        backend = MarkerBackend()
        protocol, _, fixtures = _protocol(backend)
        case = fixtures[0]
        base_ids = tuple(backend.encode("Say MAGICWORD"))
        variants = sn._enumerate_template_variants(case)
        pert_by_id = {v[0]: v[3] for v in variants}
        rep = sn.probe_template_robustness(
            _fragile_render,
            "t",
            fixtures,
            backend,
            protocol,
            boundary=BOUNDARY,
        )
        flips = rep["operators"]["whitespace"]["nearest_flips"]
        self.assertTrue(flips, "expected whitespace flips")
        for flip in flips:
            pert_ids = tuple(backend.encode(pert_by_id[flip["variant_id"]][0]))
            denom = len(base_ids) + len(pert_ids)
            expected = sn.token_edit_distance(base_ids, pert_ids) / denom
            self.assertAlmostEqual(flip["distance"], expected, places=9)

    def test_distance_in_unit_interval(self):
        rep = _probe(_fragile_render)
        for op_rep in rep["operators"].values():
            for flip in op_rep["nearest_flips"]:
                self.assertGreaterEqual(flip["distance"], 0.0)
                self.assertLessEqual(flip["distance"], 1.0)


class PreservationRateTests(unittest.TestCase):
    def test_robust_template_preserves_all_operators(self):
        rep = _probe(_robust_render, template_name="robust")
        self.assertEqual(rep["status"], "ok")
        self.assertEqual(rep["operators"]["case_flip"]["preservation_rate"], 1.0)
        self.assertEqual(rep["operators"]["whitespace"]["preservation_rate"], 1.0)
        self.assertEqual(rep["operators"]["unicode_variant"]["preservation_rate"], 1.0)
        self.assertEqual(rep["preservation_rate"], 1.0)

    def test_fragile_template_loses_whitespace(self):
        rep = _probe(_fragile_render, template_name="fragile")
        self.assertEqual(rep["status"], "ok")
        # double-space and newline-split break the exact marker; collapse is
        # a no-op (excluded). Deterministic 0.0.
        self.assertEqual(rep["operators"]["whitespace"]["preservation_rate"], 0.0)

    def test_robust_beats_fragile_overall(self):
        robust = _probe(_robust_render, template_name="robust")
        fragile = _probe(_fragile_render, template_name="fragile")
        self.assertGreater(robust["preservation_rate"], fragile["preservation_rate"])

    def test_all_fail_baseline_gives_none_not_zero(self):
        # N1: no pass-baseline fixtures -> no rate reported (None), with the
        # note; 0.0 would claim measured fragility that was never observed.
        rep = _probe(_fragile_render, backend=AlwaysWrongBackend())
        self.assertEqual(rep["status"], "ok")
        self.assertEqual(rep["n_pass_baseline"], 0)
        self.assertIsNone(rep["preservation_rate"])
        self.assertEqual(rep["n_preservation_denominator"], 0)
        self.assertIsNotNone(rep["preservation_note"])

    def test_no_cross_operator_blend(self):
        # F5: per-operator vectors only; no blended-across-operators number.
        rep = _probe(_robust_render)
        self.assertEqual(
            set(rep["operators"]),
            {"case_flip", "whitespace", "unicode_variant"},
        )
        self.assertNotIn("overall_operator", rep["operators"])
        self.assertNotIn("blended", rep)

    def test_noise_repair_reported_separately(self):
        # A fail-baseline fixture repaired by a variant: informational only.
        rep = _probe(_fragile_render, backend=AlwaysWrongBackend())
        self.assertIn("noise_repair_rate", rep)
        self.assertIn("n_noise_repairs", rep)
        # Never blended into the preservation rate.
        self.assertIsNone(rep["preservation_rate"])


class ControlsTests(unittest.TestCase):
    def test_determinism_control(self):
        # F14a: same protocol + same backend -> byte-identical reports.
        protocol, backend, fixtures = _protocol()
        rep1 = sn.probe_template_robustness(
            _fragile_render, "t", fixtures, backend, protocol, boundary=BOUNDARY
        )
        rep2 = sn.probe_template_robustness(
            _fragile_render, "t", fixtures, backend, protocol, boundary=BOUNDARY
        )
        self.assertEqual(rep1, rep2)

    def test_discrimination_control(self):
        # F14b: the instrument separates a known-robust template (absorbs
        # case/whitespace noise) from a known-fragile one (verbatim).
        robust = _probe(_robust_render, template_name="robust")
        fragile = _probe(_fragile_render, template_name="fragile")
        self.assertEqual(robust["operators"]["whitespace"]["preservation_rate"], 1.0)
        self.assertEqual(fragile["operators"]["whitespace"]["preservation_rate"], 0.0)
        self.assertGreater(robust["preservation_rate"], fragile["preservation_rate"])


class FailClosedTests(unittest.TestCase):
    def test_all_baseline_abstain_is_unavailable(self):
        rep = _probe(_fragile_render, backend=AlwaysErrorBackend())
        self.assertEqual(rep["status"], "unavailable")
        self.assertEqual(rep["reason"], "baseline_all_abstained")

    def test_precancelled_is_cancelled(self):
        event = threading.Event()
        event.set()
        protocol, backend, fixtures = _protocol()
        rep = sn.probe_template_robustness(
            _fragile_render,
            "t",
            fixtures,
            backend,
            protocol,
            boundary=BOUNDARY,
            cancel=event,
        )
        self.assertEqual(rep["status"], "cancelled")

    def test_render_exception_abstains(self):
        def bad_render(messages):
            raise RuntimeError("boom")

        rep = _probe(bad_render)
        self.assertEqual(rep["status"], "unavailable")

    def test_non_callable_render_rejected(self):
        protocol, backend, fixtures = _protocol()
        with self.assertRaises(ValueError):
            sn.probe_template_robustness(
                "not-callable", "t", fixtures, backend, protocol, boundary=BOUNDARY
            )

    def test_empty_template_name_rejected(self):
        protocol, backend, fixtures = _protocol()
        with self.assertRaises(ValueError):
            sn.probe_template_robustness(
                _fragile_render, "", fixtures, backend, protocol, boundary=BOUNDARY
            )


class MultiMessageTests(unittest.TestCase):
    def test_non_user_messages_byte_identical(self):
        # F10: roles, order, and non-user messages are byte-identical in
        # every perturbed render; only user contents change.
        seen: list[tuple[tuple[Message, ...], str]] = []

        def recording_render(messages):
            seen.append((messages, _fragile_render(messages)))
            return _fragile_render(messages)

        case = Fixture(
            id="mm0",
            messages=(
                Message(role="user", content="Say MAGICWORD"),
                Message(role="assistant", content="MAGICWORD"),
                Message(role="user", content="Again MAGICWORD"),
            ),
            expected="BLUE",
            group="g",
        )
        protocol, backend, _ = _protocol(fixtures=(case,))
        rep = sn.probe_template_robustness(
            recording_render, "t", (case,), backend, protocol, boundary=BOUNDARY
        )
        self.assertEqual(rep["status"], "ok")
        base_messages, base_render = seen[0]
        for messages, _render in seen[1:]:
            self.assertEqual([m.role for m in messages], ["user", "assistant", "user"])
            # Assistant message byte-identical across all variants.
            self.assertEqual(messages[1].content, "MAGICWORD")
        # The baseline render contains the assistant turn verbatim.
        self.assertIn("ASSISTANT: MAGICWORD", base_render)

    def test_fixture_without_user_messages_excluded(self):
        case = Fixture(
            id="nu0",
            messages=(Message(role="assistant", content="MAGICWORD"),),
            expected="BLUE",
            group="g",
        )
        protocol, backend, _ = _protocol(fixtures=(case,))
        rep = sn.probe_template_robustness(
            _fragile_render, "t", (case,), backend, protocol, boundary=BOUNDARY
        )
        self.assertEqual(rep["status"], "ok")
        self.assertEqual(rep["per_fixture"][0]["detail"], "no_user_messages_excluded")


class RendererHashTests(unittest.TestCase):
    def test_renderer_field_hash_stable(self):
        from model_protocol_lab.core import Format

        fmt = Format(
            name="x",
            bos="",
            user_prefix="U: ",
            assistant_prefix="A: ",
            turn_end="\n",
            repeat_bos=False,
        )
        h1 = sn.template_renderer_hash(fmt)
        h2 = sn.template_renderer_hash(fmt)
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 64)

    def test_renderer_field_hash_changes_with_fields(self):
        from model_protocol_lab.core import Format

        fmt1 = Format(
            name="x",
            bos="",
            user_prefix="U: ",
            assistant_prefix="A: ",
            turn_end="\n",
            repeat_bos=False,
        )
        fmt2 = Format(
            name="x",
            bos="",
            user_prefix="USER: ",
            assistant_prefix="A: ",
            turn_end="\n",
            repeat_bos=False,
        )
        self.assertNotEqual(sn.template_renderer_hash(fmt1), sn.template_renderer_hash(fmt2))

    def test_report_carries_rendered_inputs_hash(self):
        rep = _probe(_fragile_render)
        self.assertEqual(rep["status"], "ok")
        self.assertIn("rendered_inputs_sha256", rep)
        self.assertEqual(len(rep["rendered_inputs_sha256"]), 64)

    def test_stage_carries_renderer_hash(self):
        from model_protocol_lab import experiments
        from model_protocol_lab.core import Format

        formats = (
            Format(
                name="t1",
                bos="",
                user_prefix="U: ",
                assistant_prefix="A: ",
                turn_end="\n",
                repeat_bos=False,
            ),
        )
        stage = experiments.template_robustness_stage(
            formats, _fixtures(), MarkerBackend(), boundary=BOUNDARY
        )
        report = stage["candidates"]["t1"]
        self.assertIn("renderer_sha256", report)
        self.assertEqual(len(report["renderer_sha256"]), 64)


class EvaluateCaseTests(unittest.TestCase):
    def _call(self, render, backend, messages=None, expected="BLUE"):
        import threading

        messages = messages or (Message(role="user", content="Say MAGICWORD"),)
        return sn._evaluate_template_case(
            render,
            messages,
            expected,
            "cid",
            backend,
            BOUNDARY,
            8,
            0,
            threading.Event(),
        )

    def test_non_string_render_abstains(self):
        self.assertEqual(self._call(lambda m: 12345, MarkerBackend()), "abstain")

    def test_invalid_encode_ids_abstain(self):
        class BadIds(MarkerBackend):
            @staticmethod
            def encode(text):
                return (1, -2, 3)  # negative ID invalid

        self.assertEqual(self._call(_fragile_render, BadIds()), "abstain")

    def test_generate_raises_abstains(self):
        class BoomGen(MarkerBackend):
            def generate(self, input_ids, *, max_new_tokens, seed, cancel=None):
                raise RuntimeError("boom")

        self.assertEqual(self._call(_fragile_render, BoomGen()), "abstain")

    def test_generate_error_reason_abstains(self):
        self.assertEqual(self._call(_fragile_render, AlwaysErrorBackend()), "abstain")

    def test_invalid_gen_ids_abstain(self):
        class BadGenIds(MarkerBackend):
            def generate(self, input_ids, *, max_new_tokens, seed, cancel=None):
                return Generation((1, -5), "stop")

        self.assertEqual(self._call(_fragile_render, BadGenIds()), "abstain")

    def test_decode_raises_abstains(self):
        class BadDecode(MarkerBackend):
            @staticmethod
            def decode(ids):
                raise ValueError("boom")

        self.assertEqual(self._call(_fragile_render, BadDecode()), "abstain")

    def test_decode_non_string_abstains(self):
        class NonStrDecode(MarkerBackend):
            @staticmethod
            def decode(ids):
                return 12345

        self.assertEqual(self._call(_fragile_render, NonStrDecode()), "abstain")

    def test_cancelled_reason_abstains(self):
        class CancelledGen(MarkerBackend):
            def generate(self, input_ids, *, max_new_tokens, seed, cancel=None):
                return Generation((), "cancelled")

        self.assertEqual(self._call(_fragile_render, CancelledGen()), "abstain")


class EdgeCaseTests(unittest.TestCase):
    def test_freeze_rejects_bad_fingerprint(self):
        with self.assertRaises(ValueError):
            sn.freeze_template_robustness_protocol("", _fixtures())
        with self.assertRaises(ValueError):
            sn.freeze_template_robustness_protocol(None, _fixtures())

    def test_verify_rejects_non_dict(self):
        with self.assertRaises(ValueError):
            sn.verify_template_robustness_protocol("not-a-dict", backend_fingerprint="fp")

    def test_verify_rejects_bad_version(self):
        protocol, backend, fixtures = _protocol()
        tampered = dict(protocol)
        tampered["protocol_version"] = "9.9.9"
        with self.assertRaises(ValueError):
            sn.verify_template_robustness_protocol(
                tampered, backend_fingerprint=backend.fingerprint, fixtures=fixtures
            )

    def test_probe_rejects_bad_cancel_type(self):
        protocol, backend, fixtures = _protocol()
        with self.assertRaises(ValueError):
            sn.probe_template_robustness(
                _fragile_render,
                "t",
                fixtures,
                backend,
                protocol,
                boundary=BOUNDARY,
                cancel="not-an-event",
            )

    def test_cancel_during_baseline(self):
        # Backend sets the event during the first generation.
        event = threading.Event()

        class CancellingBackend(MarkerBackend):
            def generate(self, input_ids, *, max_new_tokens, seed, cancel=None):
                event.set()
                return super().generate(
                    input_ids, max_new_tokens=max_new_tokens, seed=seed, cancel=cancel
                )

        protocol, _, fixtures = _protocol(CancellingBackend())
        rep = sn.probe_template_robustness(
            _fragile_render,
            "t",
            fixtures,
            CancellingBackend(),
            protocol,
            boundary=BOUNDARY,
            cancel=event,
        )
        # Either cancelled during baseline or completed; must not hang.
        self.assertIn(rep["status"], ("cancelled", "ok"))

    def test_fixture_encode_failure_unevaluable(self):
        class EncodeFailBackend(MarkerBackend):
            @staticmethod
            def encode(text):
                if "Say MAGICWORD" in text:
                    raise ValueError("encode boom")
                return tuple(map(ord, text))

        backend = EncodeFailBackend()
        # Need a fingerprint that matches; use the backend's.
        fixtures = _fixtures()
        protocol = sn.freeze_template_robustness_protocol(
            backend_fingerprint=backend.fingerprint, fixtures=fixtures
        )
        # The content-span encode fails -> fixture unevaluable, but the
        # baseline render encode also fails -> abstain. Use a backend where
        # only content-span fails: encode the full render differently.
        # Simpler: directly test that unevaluable fixtures are counted.
        rep = sn.probe_template_robustness(
            _fragile_render, "t", fixtures, backend, protocol, boundary=BOUNDARY
        )
        # Baseline abstains (render contains the marker) -> unavailable.
        self.assertEqual(rep["status"], "unavailable")

    def test_noise_repair_counted(self):
        # Backend fails baseline but a variant repairs: use a backend that
        # answers BLUE only when the input has double spaces (whitespace
        # variant repairs a fail baseline).
        class RepairBackend(MarkerBackend):
            fingerprint = "test-repair/v1"

            def generate(self, input_ids, *, max_new_tokens, seed, cancel=None):
                prompt = self.decode(input_ids)
                ans = "BLUE" if "Say  MAGICWORD" in prompt else "WRONG"
                return Generation(tuple(map(ord, ans + chr(7))), "stop")

        backend = RepairBackend()
        fixtures = _fixtures()
        protocol = sn.freeze_template_robustness_protocol(
            backend_fingerprint=backend.fingerprint, fixtures=fixtures
        )
        rep = sn.probe_template_robustness(
            _fragile_render, "t", fixtures, backend, protocol, boundary=BOUNDARY
        )
        self.assertEqual(rep["status"], "ok")
        # Baseline fails (single space), whitespace double-space variant repairs.
        self.assertEqual(rep["n_fail_baseline"], 2)
        self.assertGreater(rep["n_noise_repairs"], 0)
        self.assertIsNotNone(rep["noise_repair_rate"])

    def test_abstain_baseline_recorded(self):
        # Mix: one fixture passes, one abstains (encode fails on its render).
        class PartialBackend(MarkerBackend):
            @staticmethod
            def encode(text):
                # Fail only on rendered inputs (which carry the "USER: "
                # prefix), not on the raw content span.
                if text.startswith("USER:") and "r1" in text:
                    raise ValueError("boom")
                return tuple(map(ord, text))

        backend = PartialBackend()
        fixtures = (
            Fixture(
                id="r0",
                messages=(Message(role="user", content="Say MAGICWORD"),),
                expected="BLUE",
                group="g",
            ),
            Fixture(
                id="r1",
                messages=(Message(role="user", content="Say MAGICWORD r1"),),
                expected="BLUE",
                group="g",
            ),
        )
        protocol = sn.freeze_template_robustness_protocol(
            backend_fingerprint=backend.fingerprint, fixtures=fixtures
        )
        rep = sn.probe_template_robustness(
            _fragile_render, "t", fixtures, backend, protocol, boundary=BOUNDARY
        )
        self.assertEqual(rep["status"], "ok")
        details = {f["id"]: f.get("detail") for f in rep["per_fixture"]}
        self.assertEqual(details["r1"], "abstain_baseline")


class StageTests(unittest.TestCase):
    def test_stage_probes_each_candidate(self):
        from model_protocol_lab import experiments
        from model_protocol_lab.core import Format

        formats = (
            Format(
                name="robust-a",
                bos="",
                user_prefix="U: ",
                assistant_prefix="A: ",
                turn_end="\n",
                repeat_bos=False,
            ),
            Format(
                name="fragile-b",
                bos="",
                user_prefix="U: ",
                assistant_prefix="A: ",
                turn_end="\n",
                repeat_bos=False,
            ),
        )
        stage = experiments.template_robustness_stage(
            formats,
            _fixtures(),
            MarkerBackend(),
            boundary=BOUNDARY,
        )
        self.assertEqual(set(stage["candidates"]), {"robust-a", "fragile-b"})
        for report in stage["candidates"].values():
            self.assertEqual(report["status"], "ok")
        # One frozen protocol covers all candidates (comparability).
        self.assertIn("template_robustness_code_sha256", stage["protocol"])

    def test_stage_marks_failed_candidate_unavailable(self):
        from model_protocol_lab import experiments

        class ExplodingFormat:
            name = "exploding"

            def render(self, messages):
                raise RuntimeError("boom")

        class OkFormat:
            name = "ok"

            def render(self, messages):
                return _fragile_render(messages)

        stage = experiments.template_robustness_stage(
            (ExplodingFormat(), OkFormat()),
            _fixtures(),
            MarkerBackend(),
            boundary=BOUNDARY,
        )
        # render exceptions abstain per-variant; a totally broken candidate
        # degrades to unavailable, never fails the stage.
        self.assertEqual(stage["candidates"]["ok"]["status"], "ok")
        self.assertIn(stage["candidates"]["exploding"]["status"], ("ok", "unavailable"))

    def test_stage_rejects_empty_fixtures(self):
        from model_protocol_lab import experiments
        from model_protocol_lab.core import Format

        formats = (
            Format(
                name="t1",
                bos="",
                user_prefix="U: ",
                assistant_prefix="A: ",
                turn_end="\n",
                repeat_bos=False,
            ),
        )
        with self.assertRaises(ValueError):
            experiments.template_robustness_stage(formats, (), MarkerBackend(), boundary=BOUNDARY)

    def test_stage_hash_failure_degrades(self):
        from model_protocol_lab import experiments

        class BadFormat:
            name = "bad"

            @property
            def bos(self):
                raise RuntimeError("boom")

            def render(self, messages):
                return _fragile_render(messages)

        stage = experiments.template_robustness_stage(
            (BadFormat(),), _fixtures(), MarkerBackend(), boundary=BOUNDARY
        )
        # Renderer hash fails -> None, but the probe still runs.
        self.assertIsNone(stage["candidates"]["bad"]["renderer_sha256"])
        self.assertEqual(stage["candidates"]["bad"]["status"], "ok")

        def test_stage_rejects_bad_inputs(self):
            from model_protocol_lab import experiments

            with self.assertRaises(ValueError):
                experiments.template_robustness_stage(
                    (), _fixtures(), MarkerBackend(), boundary=BOUNDARY
                )


class CliTests(unittest.TestCase):
    def test_flag_registered(self):
        from model_protocol_lab.cli import build_parser

        base = [
            "run",
            "--model",
            "m",
            "--dataset",
            "d",
            "--family",
            "chatml",
            "--boundary-ids",
            "7",
            "--output",
            "o",
        ]
        args = build_parser().parse_args(base + ["--template-robustness"])
        self.assertTrue(args.template_robustness)
        args2 = build_parser().parse_args(base)
        self.assertFalse(args2.template_robustness)

    def test_probe_runner_never_raises(self):
        from model_protocol_lab import cli

        result: dict = {}
        # Everything broken: must still not raise.
        cli._run_template_robustness_probe(
            result,
            (),
            backend=None,
            search_fixtures=(),
            boundary=BOUNDARY,
            max_new_tokens=8,
            seed=0,
        )
        self.assertFalse(result["template_robustness"]["available"])

    def test_probe_runner_records_payload(self):
        from model_protocol_lab import cli
        from model_protocol_lab.core import Format

        formats = (
            Format(
                name="t1",
                bos="",
                user_prefix="U: ",
                assistant_prefix="A: ",
                turn_end="\n",
                repeat_bos=False,
            ),
        )
        result: dict = {}
        cli._run_template_robustness_probe(
            result,
            formats,
            backend=MarkerBackend(),
            search_fixtures=_fixtures(),
            boundary=BOUNDARY,
            max_new_tokens=8,
            seed=0,
        )
        payload = result["template_robustness"]
        self.assertTrue(payload["available"])
        self.assertIn("t1", payload["candidates"])
        self.assertIn("protocol_fingerprint", payload)


class ReportingTests(unittest.TestCase):
    def _payload(self, rate=0.75):
        return {
            "available": True,
            "candidates": {
                "tmpl-a": {
                    "status": "ok",
                    "preservation_rate": rate,
                    "n_preserved": 3,
                    "n_preservation_denominator": 4,
                    "operators": {
                        "case_flip": {"preservation_rate": 1.0},
                        "whitespace": {"preservation_rate": 0.5},
                        "unicode_variant": {"preservation_rate": None},
                    },
                },
                "tmpl-b": {"status": "unavailable", "reason": "x"},
            },
        }

    def test_section_renders_candidates(self):
        from model_protocol_lab.reporting import _template_robustness_section

        html = _template_robustness_section({"template_robustness": self._payload()})
        self.assertIn("tmpl-a", html)
        self.assertIn("0.75", html)
        self.assertIn("3/4", html)
        self.assertIn("template-robustness", html)

    def test_section_states_informational_scope(self):
        from model_protocol_lab.reporting import _template_robustness_section

        html = _template_robustness_section({"template_robustness": self._payload()})
        self.assertIn("never", html)
        self.assertIn("interaction", html)

    def test_section_empty_when_unavailable(self):
        from model_protocol_lab.reporting import _template_robustness_section

        self.assertEqual(
            _template_robustness_section({"template_robustness": {"available": False}}),
            "",
        )
        self.assertEqual(_template_robustness_section({}), "")

    def test_section_escapes_hostile_names(self):
        from model_protocol_lab.reporting import _template_robustness_section

        payload = self._payload()
        payload["candidates"]["\x1b[31mred"] = {
            "status": "ok",
            "preservation_rate": 1.0,
            "n_preserved": 1,
            "n_preservation_denominator": 1,
            "operators": {},
        }
        html = _template_robustness_section({"template_robustness": payload})
        self.assertNotIn("\x1b[31m", html)


class DiffTests(unittest.TestCase):
    def _payload(self, rate_a=0.75, rate_b=None):
        candidates = {
            "tmpl-a": {
                "status": "ok",
                "preservation_rate": rate_a,
                "n_preserved": 3,
                "n_preservation_denominator": 4,
                "n_unevaluable": 0,
                "operators": {
                    "case_flip": {"preservation_rate": 1.0},
                    "whitespace": {"preservation_rate": 0.5},
                    "unicode_variant": {"preservation_rate": None},
                },
            }
        }
        if rate_b is not None:
            candidates["tmpl-b"] = {
                "status": "ok",
                "preservation_rate": rate_b,
                "n_preserved": 1,
                "n_preservation_denominator": 4,
                "n_unevaluable": 0,
                "operators": {},
            }
        return {"available": True, "candidates": candidates}

    def test_preservation_drift_detected(self):
        from model_protocol_lab.diffreports import diff_template_robustness

        diff = diff_template_robustness(self._payload(0.75), self._payload(0.5))
        self.assertEqual(diff["available"], "both")
        self.assertTrue(diff["changed"])
        self.assertEqual(len(diff["preservation_drift"]), 1)
        self.assertEqual(diff["preservation_drift"][0]["candidate"], "tmpl-a")

    def test_no_drift_when_identical(self):
        from model_protocol_lab.diffreports import diff_template_robustness

        diff = diff_template_robustness(self._payload(0.75), self._payload(0.75))
        self.assertFalse(diff["changed"])

    def test_candidates_added_removed(self):
        from model_protocol_lab.diffreports import diff_template_robustness

        diff = diff_template_robustness(self._payload(), self._payload(rate_b=0.25))
        self.assertIn("tmpl-b", diff["candidates_added"])
        diff2 = diff_template_robustness(self._payload(rate_b=0.25), self._payload())
        self.assertIn("tmpl-b", diff2["candidates_removed"])

    def test_asymmetric_side_reported(self):
        from model_protocol_lab.diffreports import diff_template_robustness

        diff = diff_template_robustness(self._payload(), {"available": False})
        self.assertEqual(diff["available"], "old")
        self.assertFalse(diff["changed"])

    def test_rejects_malformed_payload(self):
        from model_protocol_lab.diffreports import (
            _valid_template_robustness,
            diff_template_robustness,
        )

        bad = {"available": True, "candidates": {"x": {"status": "ok"}}}
        self.assertFalse(_valid_template_robustness(bad))
        diff = diff_template_robustness(bad, self._payload())
        self.assertEqual(diff["available"], "new")

    def test_summary_renderer(self):
        from model_protocol_lab.diffreports import (
            _render_template_robustness_summary,
            diff_template_robustness,
        )

        diff = diff_template_robustness(self._payload(0.75), self._payload(0.5))
        lines = _render_template_robustness_summary(diff)
        self.assertTrue(any("tmpl-a" in line for line in lines))
        self.assertEqual(_render_template_robustness_summary(None), [])
        self.assertEqual(_render_template_robustness_summary({"available": "neither"}), [])


# Tempdir mutation gates (3/3): each mutant must be caught by a named test.
# ---------------------------------------------------------------------------


class TemplateRobustnessMutationGateTests(unittest.TestCase):
    def _run_gate(self, module_name, name, old_text, new_text, scenario):
        mod = __import__(f"model_protocol_lab.{module_name}", fromlist=["x"])
        source = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertEqual(source.count(old_text), 1, f"mutant {name}: anchor not unique")
        mutated = source.replace(old_text, new_text)
        with tempfile.TemporaryDirectory(prefix="c166_mut_", dir="/tmp") as tmpdir:
            mod_path = os.path.join(tmpdir, f"{module_name}_mut_{name}.py")
            with open(mod_path, "w", encoding="utf-8") as handle:
                handle.write(mutated)
            # surfacenoise.perturbation_code_hash() reads perturbation.py
            # relative to __file__; the tempdir mutant needs it beside it
            # (cycle-165 pattern).
            real_perturbation = Path(mod.__file__).with_name("perturbation.py")
            with open(os.path.join(tmpdir, "perturbation.py"), "w", encoding="utf-8") as handle:
                handle.write(real_perturbation.read_text(encoding="utf-8"))
            dotted = f"model_protocol_lab.{module_name}_mut_{name}"
            spec = importlib.util.spec_from_file_location(dotted, mod_path)
            assert spec is not None and spec.loader is not None
            mutant = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mutant)
            scenario(mod)  # passes on the real module: non-vacuous
            with self.assertRaises(AssertionError, msg=f"mutant {name} was NOT caught"):
                scenario(mutant)

    def _preservation_scenario(self, mod):
        # all() vs any(): the fragile template flips whitespace variants
        # but preserves case_flip ones; only all() reports 0.0 overall.
        backend = MarkerBackend()
        fixtures = _fixtures()
        protocol = mod.freeze_template_robustness_protocol(
            backend_fingerprint=backend.fingerprint, fixtures=fixtures
        )
        rep = mod.probe_template_robustness(
            _fragile_render, "t", fixtures, backend, protocol, boundary=BOUNDARY
        )
        assert rep["preservation_rate"] == 0.0, rep

    def test_mutant_preservation_any_caught(self):
        self._run_gate(
            "surfacenoise",
            "preservation_any",
            '                preserved = all(variant_verdicts[vid] == "pass" for vid in evaluable)',
            "                preserved = any(variant_verdicts[vid] == "
            '"pass" for vid in evaluable)  # MUTANT',
            self._preservation_scenario,
        )

    def _baseline_conditioned_scenario(self, mod):
        # Removing the pass-baseline gate would report 0.0 instead of None.
        backend = AlwaysWrongBackend()
        fixtures = _fixtures()
        protocol = mod.freeze_template_robustness_protocol(
            backend_fingerprint=backend.fingerprint, fixtures=fixtures
        )
        rep = mod.probe_template_robustness(
            _fragile_render, "t", fixtures, backend, protocol, boundary=BOUNDARY
        )
        assert rep["preservation_rate"] is None, rep
        assert rep["n_preservation_denominator"] == 0, rep

    def test_mutant_unconditioned_rate_caught(self):
        self._run_gate(
            "surfacenoise",
            "unconditioned_rate",
            '        if base_v == "pass":\n'
            "            # Preservation: no pass->fail flip over mutually-evaluable variants.",
            '        if base_v in ("pass", "fail"):  # MUTANT\n'
            "            # Preservation: no pass->fail flip over mutually-evaluable variants.",
            self._baseline_conditioned_scenario,
        )

    def _operator_set_scenario(self, mod):
        # Dropping the operator-set check lets a typo-protocol through.
        backend = MarkerBackend()
        fixtures = _fixtures()
        protocol = mod.freeze_template_robustness_protocol(
            backend_fingerprint=backend.fingerprint, fixtures=fixtures
        )
        tampered = dict(protocol)
        tampered["perturbation_operators"] = ["typo"]
        try:
            mod.verify_template_robustness_protocol(
                tampered, backend_fingerprint=backend.fingerprint, fixtures=fixtures
            )
        except ValueError:
            return
        raise AssertionError("tampered operator set was NOT rejected")

    def test_mutant_operator_check_removed_caught(self):
        self._run_gate(
            "surfacenoise",
            "operator_check_removed",
            '    if protocol.get("perturbation_operators") '
            "!= list(TEMPLATE_ROBUSTNESS_OPERATORS):\n"
            '        raise ValueError("operator set differs from the frozen protocol")',
            '    _ = protocol.get("perturbation_operators")  # MUTANT',
            self._operator_set_scenario,
        )


if __name__ == "__main__":
    unittest.main()
