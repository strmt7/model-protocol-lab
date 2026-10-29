"""Cycle 160: few-shot x template -- demo gain and order sensitivity.

Covers the pure ``fewshot`` core (seeded demo sampling, the two-orders
design, demo-fixture construction, verdict-change rates, order
sensitivity), the ``fewshot_stage`` / ``fewshot_gain`` experiments
wiring, the CLI probe, the HTML section, and the ``diff_fewshot``
comparison -- including 4/4 tempdir mutation gates, a 200-case
brute-force oracle differential for the gain/sensitivity arithmetic,
and single-cause pins for every ``changed`` disjunct.
"""

from __future__ import annotations

import importlib.util
import math
import os
import random
import re
import tempfile
import unittest
from pathlib import Path
from threading import Event

from model_protocol_lab import diffreports
from model_protocol_lab.core import Fixture, Format, Generation, Message
from model_protocol_lab.experiments import fewshot_gain, fewshot_stage
from model_protocol_lab.fewshot import (
    ARMS,
    DETAIL_INSUFFICIENT_DEMO_POOL,
    FEWSHOT_K,
    FEWSHOT_SEED,
    _messages_alternate_user_last,
    build_demo_fixtures,
    build_demo_messages,
    change_rate,
    demo_permutations,
    demo_question_text,
    order_sensitivity,
    original_fixture_id,
    sample_demos,
    validate_demo_pool,
    validate_eval_fixtures,
)

BOUNDARY = (124,)  # ord("|")


def _fixture(
    fid: str,
    expected: str = "WORD",
    content: str = "Say the word.",
) -> Fixture:
    return Fixture(
        id=fid,
        messages=(Message(role="user", content=content),),
        expected=expected,
        group="g",
    )


def _format(name: str, bos: str = "<s>", user_prefix: str = "[user]\n") -> Format:
    return Format(
        name=name,
        bos=bos,
        user_prefix=user_prefix,
        assistant_prefix="[asst]\n",
        turn_end="[end]\n",
    )


class FewshotScriptedBackend:
    """Demo-order policy backend: keyword baseline, recency/primacy demos.

    Prompts without the demo header get the keyword policy ("WORD" when
    the fixture asks for the word, else "WRONG"); the "NOP" marker
    forces WRONG everywhere (a candidate whose baseline already fails).
    Prompts carrying the demo turns answer from the parsed ``A:``
    lines: the last demo's answer, unless "PRIMACY" appears in the
    prompt (a template marker), in which case the first. Deterministic.
    """

    fingerprint = "scripted-fewshot-c160"

    @staticmethod
    def encode(text: str) -> tuple[int, ...]:
        return tuple(map(ord, text))

    @staticmethod
    def decode(token_ids: tuple[int, ...]) -> str:
        return "".join(map(chr, token_ids))

    def generate(
        self,
        input_ids: tuple[int, ...],
        *,
        max_new_tokens: int,
        seed: int,
        cancel: Event,
    ) -> Generation:
        prompt = self.decode(input_ids)
        if "NOP" in prompt:
            answer = "WRONG"
        elif "Example 1 of 2:" in prompt:
            answers = re.findall(r"^A: (\S+)", prompt, re.M)
            if not answers:
                answer = "WRONG"
            elif "PRIMACY" in prompt:
                answer = answers[0]
            else:
                answer = answers[-1]
        elif "Say the word." in prompt:
            answer = "WORD"
        else:
            answer = "WRONG"
        out = tuple(map(ord, (answer + "|").ljust(max_new_tokens, ".")[:max_new_tokens]))
        return Generation(token_ids=out, reason="limit")


class BrokenBackend(FewshotScriptedBackend):
    """generate always raises: every stage cell degrades to None."""

    fingerprint = "scripted-fewshot-broken-c160"

    @staticmethod
    def generate(input_ids, *, max_new_tokens, seed, cancel):
        raise RuntimeError("boom")


def _demo_pool() -> tuple[Fixture, ...]:
    return (
        _fixture("f_d0", expected="D0", content="Demo question zero."),
        _fixture("f_d1", expected="D1", content="Demo question one."),
    )


def _eval_cases() -> tuple[Fixture, ...]:
    return (_fixture("f_e0"), _fixture("f_e1"))


# ---------------------------------------------------------------------------
# Pure core: validation + sampling
# ---------------------------------------------------------------------------


class FewshotValidationTests(unittest.TestCase):
    def test_validate_accepts(self):
        pool = _demo_pool()
        self.assertEqual(validate_demo_pool(pool), pool)
        self.assertEqual(validate_eval_fixtures(pool), pool)

    def test_validate_rejects(self):
        good = _demo_pool()
        for bad in ((), [], "nope", None, 42):
            with self.assertRaises(ValueError):
                validate_demo_pool(bad)  # type: ignore[arg-type]
            with self.assertRaises(ValueError):
                validate_eval_fixtures(bad)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            validate_demo_pool(good + ("nope",))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            validate_eval_fixtures((None,))  # type: ignore[arg-type]


class SampleDemosTests(unittest.TestCase):
    def test_samples_k_in_sampled_order(self):
        demos = sample_demos(_demo_pool())
        self.assertEqual(len(demos), FEWSHOT_K)
        self.assertEqual({d.id for d in demos}, {"f_d0", "f_d1"})

    def test_deterministic(self):
        self.assertEqual(sample_demos(_demo_pool()), sample_demos(_demo_pool()))

    def test_input_order_independent(self):
        pool = _demo_pool()
        self.assertEqual(sample_demos(pool), sample_demos(tuple(reversed(pool))))

    def test_k_subset(self):
        pool = tuple(_fixture(f"f_{i}") for i in range(5))
        demos = sample_demos(pool, k=3)
        self.assertEqual(len(demos), 3)
        self.assertEqual(len({d.id for d in demos}), 3)

    def test_seed_changes_selection(self):
        pool = tuple(_fixture(f"f_{i}") for i in range(6))
        first = sample_demos(pool, k=3, seed=1)
        second = sample_demos(pool, k=3, seed=2)
        self.assertNotEqual([d.id for d in first], [d.id for d in second])
        # Every seed still returns exactly k distinct pool members.
        for demos in (first, second):
            self.assertEqual(len(demos), 3)
            self.assertTrue(all(d.id in {f"f_{i}" for i in range(6)} for d in demos))

    def test_rejects_bad_input(self):
        pool = _demo_pool()
        with self.assertRaises(ValueError):
            sample_demos(pool, k=3)
        with self.assertRaises(ValueError):
            sample_demos(pool, k=0)
        with self.assertRaises(ValueError):
            sample_demos(pool, k=-1)
        with self.assertRaises(ValueError):
            sample_demos(pool, k="2")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            sample_demos(pool, seed=-1)
        with self.assertRaises(ValueError):
            sample_demos(pool, seed=2**64)
        with self.assertRaises(ValueError):
            sample_demos((), k=2)
        with self.assertRaises(ValueError):
            sample_demos("nope")  # type: ignore[arg-type]

    def test_seed_constant_pinned(self):
        self.assertEqual(FEWSHOT_SEED, 20261019)
        self.assertEqual(FEWSHOT_K, 2)


class DemoPermutationsTests(unittest.TestCase):
    def test_two_orders(self):
        d0, d1 = _demo_pool()
        perms = demo_permutations((d0, d1))
        self.assertEqual(len(perms), 2)
        self.assertEqual(perms[0], (d0, d1))
        self.assertEqual(perms[1], (d1, d0))

    def test_single_demo(self):
        (d0,) = _demo_pool()[:1]
        self.assertEqual(demo_permutations((d0,)), ((d0,),))

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            demo_permutations(())
        with self.assertRaises(ValueError):
            demo_permutations("nope")  # type: ignore[arg-type]


class DemoQuestionTextTests(unittest.TestCase):
    def test_joins_user_turns(self):
        demo = Fixture(
            id="d",
            messages=(
                Message(role="user", content="First?"),
                Message(role="assistant", content="no"),
                Message(role="user", content="Second?"),
            ),
            expected="E",
            group="g",
        )
        self.assertEqual(demo_question_text(demo), "First?\nSecond?")

    def test_rejects_no_user_message(self):
        demo = Fixture(
            id="d",
            messages=(Message(role="assistant", content="no"),),
            expected="E",
            group="g",
        )
        with self.assertRaises(ValueError):
            demo_question_text(demo)

    def test_rejects_bad_type(self):
        with self.assertRaises(ValueError):
            demo_question_text("nope")  # type: ignore[arg-type]


class BuildDemoMessagesTests(unittest.TestCase):
    def test_turn_pair_shape(self):
        demo = _fixture("f_d0", expected="D0", content="Demo question zero.")
        user_turn, asst_turn = build_demo_messages(demo, 1, 2)
        self.assertEqual(user_turn.role, "user")
        self.assertEqual(asst_turn.role, "assistant")
        self.assertIn("Example 1 of 2:", user_turn.content)
        self.assertIn("Q: Demo question zero.", user_turn.content)
        self.assertIn("A: D0", user_turn.content)
        self.assertEqual(asst_turn.content, "D0")

    def test_second_index(self):
        demo = _fixture("f_d1", expected="D1", content="Q one.")
        user_turn, _ = build_demo_messages(demo, 2, 2)
        self.assertIn("Example 2 of 2:", user_turn.content)

    def test_rejects_bad_input(self):
        demo = _fixture("f_d0")
        with self.assertRaises(ValueError):
            build_demo_messages("nope", 1, 2)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            build_demo_messages(demo, "1", 2)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            build_demo_messages(demo, 0, 2)
        with self.assertRaises(ValueError):
            build_demo_messages(demo, 3, 2)
        with self.assertRaises(ValueError):
            build_demo_messages(demo, 1, 0)


class MessagesAlternateTests(unittest.TestCase):
    def test_alternation_cases(self):
        user = Message(role="user", content="q")
        asst = Message(role="assistant", content="a")
        self.assertFalse(_messages_alternate_user_last(()))
        self.assertTrue(_messages_alternate_user_last((user,)))
        self.assertFalse(_messages_alternate_user_last((asst,)))
        self.assertFalse(_messages_alternate_user_last((user, asst)))
        self.assertTrue(_messages_alternate_user_last((user, asst, user)))
        self.assertFalse(_messages_alternate_user_last((user, user)))
        self.assertFalse(_messages_alternate_user_last((user, asst, asst, user)))


class BuildDemoFixturesTests(unittest.TestCase):
    def test_prepends_demo_turns(self):
        demos = sample_demos(_demo_pool())
        built = build_demo_fixtures(_eval_cases(), demos, perm_index=0)
        self.assertEqual(len(built), 2)
        first = built[0]
        self.assertEqual(first.id, "fewshot/perm0/f_e0")
        self.assertEqual(first.expected, "WORD")
        self.assertEqual(first.group, "g")
        # 2 demos x 2 turns + 1 eval message
        self.assertEqual(len(first.messages), 5)
        roles = [m.role for m in first.messages]
        self.assertEqual(roles, ["user", "assistant", "user", "assistant", "user"])
        self.assertIn("Example 1 of 2:", first.messages[0].content)
        self.assertEqual(first.messages[-1].content, "Say the word.")

    def test_perm_index_namespaces(self):
        demos = sample_demos(_demo_pool())
        built = build_demo_fixtures(_eval_cases(), demos, perm_index=3)
        self.assertTrue(all(c.id.startswith("fewshot/perm3/") for c in built))

    def test_demo_order_follows_perm(self):
        d0, d1 = _demo_pool()
        built = build_demo_fixtures(_eval_cases(), (d1, d0), perm_index=0)
        self.assertIn("A: D1", built[0].messages[0].content)
        self.assertIn("A: D0", built[0].messages[2].content)

    def test_rejects_non_alternating_eval(self):
        demos = sample_demos(_demo_pool())
        bad = Fixture(
            id="bad",
            messages=(
                Message(role="user", content="q"),
                Message(role="user", content="q2"),
            ),
            expected="E",
            group="g",
        )
        with self.assertRaises(ValueError):
            build_demo_fixtures((bad,), demos, perm_index=0)
        bad_end = Fixture(
            id="bad2",
            messages=(
                Message(role="user", content="q"),
                Message(role="assistant", content="a"),
            ),
            expected="E",
            group="g",
        )
        with self.assertRaises(ValueError):
            build_demo_fixtures((bad_end,), demos, perm_index=0)

    def test_rejects_bad_input(self):
        demos = sample_demos(_demo_pool())
        with self.assertRaises(ValueError):
            build_demo_fixtures((), demos, perm_index=0)
        with self.assertRaises(ValueError):
            build_demo_fixtures(_eval_cases(), (), perm_index=0)
        with self.assertRaises(ValueError):
            build_demo_fixtures(_eval_cases(), demos, perm_index=-1)
        with self.assertRaises(ValueError):
            build_demo_fixtures(_eval_cases(), demos, perm_index="0")  # type: ignore[arg-type]


class OriginalFixtureIdTests(unittest.TestCase):
    def test_strips_namespace(self):
        self.assertEqual(original_fixture_id("fewshot/perm0/f_e0"), "f_e0")
        self.assertEqual(original_fixture_id("fewshot/perm12/abc"), "abc")

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            original_fixture_id("f_e0")
        with self.assertRaises(ValueError):
            original_fixture_id("fewshot/perm0/")
        with self.assertRaises(ValueError):
            original_fixture_id(42)  # type: ignore[arg-type]


class ChangeRateTests(unittest.TestCase):
    def test_basic_counts(self):
        baseline = {"a": True, "b": True, "c": False}
        arm = {"a": False, "b": True, "c": False}
        out = change_rate(baseline, arm)
        self.assertEqual(out["n_evaluable"], 3)
        self.assertEqual(out["n_changed"], 1)
        self.assertAlmostEqual(out["rate"], 1 / 3)

    def test_none_verdicts_excluded_never_counted(self):
        baseline = {"a": True, "b": True}
        arm = {"a": None, "b": False}
        out = change_rate(baseline, arm)
        self.assertEqual(out["n_evaluable"], 1)
        self.assertEqual(out["n_changed"], 1)
        self.assertEqual(out["rate"], 1.0)

    def test_no_evaluable_gives_none_rate(self):
        out = change_rate({"a": None}, {"a": True})
        self.assertEqual(out["n_evaluable"], 0)
        self.assertEqual(out["n_changed"], 0)
        self.assertIsNone(out["rate"])

    def test_unknown_fixture_ids_ignored(self):
        out = change_rate({"a": True}, {"b": False})
        self.assertEqual(out["n_evaluable"], 0)
        self.assertIsNone(out["rate"])

    def test_rejects_malformed(self):
        with self.assertRaises(ValueError):
            change_rate("nope", {})  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            change_rate({}, [])  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            change_rate({1: True}, {})  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            change_rate({"a": "yes"}, {})  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            change_rate({}, {"a": 1})  # type: ignore[arg-type]


class OrderSensitivityTests(unittest.TestCase):
    def test_two_orders(self):
        self.assertAlmostEqual(order_sensitivity((1.0, 0.0)), 0.5)
        self.assertAlmostEqual(order_sensitivity((0.25, 0.75)), 0.25)

    def test_identical_rates_zero(self):
        self.assertEqual(order_sensitivity((0.5, 0.5)), 0.0)

    def test_single_rate_zero(self):
        self.assertEqual(order_sensitivity((0.7,)), 0.0)

    def test_none_skipped(self):
        self.assertEqual(order_sensitivity((None, 0.4)), 0.0)
        self.assertIsNone(order_sensitivity((None, None)))
        self.assertIsNone(order_sensitivity(()))

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            order_sensitivity([0.5, 0.5])  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            order_sensitivity((1.5, 0.0))
        with self.assertRaises(ValueError):
            order_sensitivity((-0.1, 0.0))
        with self.assertRaises(ValueError):
            order_sensitivity(("x", 0.0))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            order_sensitivity((True, 0.0))

    def test_arms_constant(self):
        self.assertEqual(ARMS, ("baseline", "order_a", "order_b"))


# ---------------------------------------------------------------------------
# Experiments wiring: fewshot_stage + fewshot_gain
# ---------------------------------------------------------------------------


class FewshotStageTests(unittest.TestCase):
    @staticmethod
    def _run_stage(backend=None, formats=None):
        backend = backend or FewshotScriptedBackend()
        formats = formats or (
            _format("tmpl_a"),
            _format("tmpl_b", bos="<s>PRIMACY"),
            _format("tmpl_c", user_prefix="[user NOP]\n"),
        )
        return fewshot_stage(
            backend,
            formats,
            _demo_pool(),
            _eval_cases(),
            boundary=BOUNDARY,
            max_new_tokens=16,
            seed=7,
        )

    def test_stage_shape_and_verdicts(self):
        stage = self._run_stage()
        self.assertTrue(stage["available"])
        self.assertEqual(stage["k"], FEWSHOT_K)
        self.assertEqual(set(stage["demo_ids"]), {"f_d0", "f_d1"})
        self.assertEqual(stage["n_eval"], 2)
        self.assertEqual(stage["arms"], ["baseline", "order_a", "order_b"])
        per_candidate = stage["per_candidate"]
        cell_a = per_candidate["tmpl_a"]
        self.assertEqual(cell_a["n_eval"], 2)
        verdicts = cell_a["verdicts"]
        # Baseline: keyword policy answers WORD (correct). Demo arms:
        # demo answers D0/D1 never match WORD -> flipped.
        self.assertEqual(verdicts["f_e0"], {"baseline": True, "order_a": False, "order_b": False})
        self.assertEqual(verdicts["f_e1"], {"baseline": True, "order_a": False, "order_b": False})
        # tmpl_c: NOP marker forces WRONG everywhere -> already failing.
        cell_c = per_candidate["tmpl_c"]["verdicts"]
        self.assertEqual(cell_c["f_e0"], {"baseline": False, "order_a": False, "order_b": False})

    def test_gain_analysis_values(self):
        stage = self._run_stage()
        analysis = fewshot_gain("tmpl_a", "selected", ("tmpl_a", "tmpl_b", "tmpl_c"), stage)
        self.assertTrue(analysis["available"])
        self.assertEqual(analysis["k"], FEWSHOT_K)
        self.assertEqual(set(analysis["demo_ids"]), {"f_d0", "f_d1"})
        per_candidate = analysis["per_candidate"]
        # tmpl_a/tmpl_b: baseline all-correct, demos all-flipped -> gain 1.
        for name in ("tmpl_a", "tmpl_b"):
            entry = per_candidate[name]
            self.assertEqual(entry["gain"], 1.0)
            self.assertEqual(entry["order_sensitivity"], 0.0)
            self.assertEqual(entry["n_evaluable"], 2)
            for arm in ("order_a", "order_b"):
                self.assertEqual(entry["per_order"][arm]["change_rate"], 1.0)
        # tmpl_c: baseline already failing -> demos change nothing.
        self.assertEqual(per_candidate["tmpl_c"]["gain"], 0.0)
        self.assertEqual(per_candidate["tmpl_c"]["order_sensitivity"], 0.0)
        self.assertEqual(analysis["excluded_candidates"], [])
        # Baseline arm: tmpl_a and tmpl_b both pass -> ambiguous.
        baseline_arm = analysis["per_arm"]["baseline"]
        self.assertEqual(baseline_arm["decision"], "ambiguous")
        self.assertIsNone(baseline_arm["winner"])
        # Demo arms: nobody passes -> no_candidate; all three arms flip
        # vs the primary ("selected", "tmpl_a") -- including the
        # ambiguous baseline arm.
        for arm in ("order_a", "order_b"):
            self.assertEqual(analysis["per_arm"][arm]["decision"], "no_candidate")
        self.assertEqual(len(analysis["flips"]), 3)
        self.assertEqual({f["arm"] for f in analysis["flips"]}, {"baseline", "order_a", "order_b"})

    def test_flips_against_primary(self):
        stage = self._run_stage(
            formats=(_format("tmpl_a"), _format("tmpl_c", user_prefix="[user NOP]\n"))
        )
        analysis = fewshot_gain("tmpl_a", "selected", ("tmpl_a", "tmpl_c"), stage)
        self.assertEqual(analysis["per_arm"]["baseline"]["decision"], "selected")
        self.assertEqual(analysis["per_arm"]["baseline"]["winner"], "tmpl_a")
        flips = analysis["flips"]
        self.assertEqual(len(flips), 2)
        for flip in flips:
            self.assertEqual(flip["decision_from"], "selected")
            self.assertEqual(flip["decision_to"], "no_candidate")
            self.assertEqual(flip["winner_from"], "tmpl_a")
            self.assertIsNone(flip["winner_to"])

    def test_primary_decision_mapping(self):
        stage = self._run_stage(
            formats=(_format("tmpl_a"), _format("tmpl_c", user_prefix="[user NOP]\n"))
        )
        names = ("tmpl_a", "tmpl_c")
        mapped = fewshot_gain("tmpl_a", "compatible_on_fixture_set", names, stage)
        self.assertEqual(len(mapped["flips"]), 2)
        held_out = fewshot_gain("tmpl_a", "held_out_failed", names, stage)
        self.assertEqual(len(held_out["flips"]), 2)

    def test_broken_backend_degrades_cells(self):
        stage = self._run_stage(backend=BrokenBackend())
        self.assertTrue(stage["available"])
        self.assertTrue(all(cell is None for cell in stage["per_candidate"].values()))
        analysis = fewshot_gain("tmpl_a", "selected", ("tmpl_a",), stage)
        self.assertTrue(analysis["available"])
        self.assertEqual(analysis["per_candidate"], {})
        self.assertEqual(analysis["excluded_candidates"], ["tmpl_a"])
        for arm in ("baseline", "order_a", "order_b"):
            self.assertEqual(analysis["per_arm"][arm]["decision"], "unavailable")
        self.assertEqual(analysis["flips"], [])

    def test_insufficient_demo_pool_fails_closed(self):
        stage = fewshot_stage(
            FewshotScriptedBackend(),
            (_format("tmpl_a"),),
            (_fixture("only_one"),),
            _eval_cases(),
            boundary=BOUNDARY,
            max_new_tokens=16,
            seed=7,
        )
        self.assertFalse(stage["available"])
        self.assertEqual(stage["detail"], DETAIL_INSUFFICIENT_DEMO_POOL)
        analysis = fewshot_gain("tmpl_a", "selected", ("tmpl_a",), stage)
        self.assertFalse(analysis["available"])
        self.assertEqual(analysis["detail"], DETAIL_INSUFFICIENT_DEMO_POOL)

    def test_stage_rejects_bad_input(self):
        backend = FewshotScriptedBackend()
        formats = (_format("tmpl_a"),)
        good_kwargs = {
            "boundary": BOUNDARY,
            "max_new_tokens": 16,
            "seed": 7,
        }
        with self.assertRaises(ValueError):
            fewshot_stage(None, formats, _demo_pool(), _eval_cases(), **good_kwargs)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            fewshot_stage(backend, (), _demo_pool(), _eval_cases(), **good_kwargs)
        with self.assertRaises(ValueError):
            fewshot_stage(backend, formats, (), _eval_cases(), **good_kwargs)
        with self.assertRaises(ValueError):
            fewshot_stage(backend, formats, ("x",), _eval_cases(), **good_kwargs)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            fewshot_stage(backend, formats, _demo_pool(), (), **good_kwargs)
        with self.assertRaises(ValueError):
            fewshot_stage(backend, formats, _demo_pool(), ("x",), **good_kwargs)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            fewshot_stage(
                backend,
                formats,
                _demo_pool(),
                _eval_cases(),
                boundary=(),
                max_new_tokens=16,
                seed=7,
            )
        with self.assertRaises(ValueError):
            fewshot_stage(
                backend,
                formats,
                _demo_pool(),
                _eval_cases(),
                boundary=BOUNDARY,
                max_new_tokens=0,
                seed=7,
            )
        with self.assertRaises(ValueError):
            fewshot_stage(
                backend,
                formats,
                _demo_pool(),
                _eval_cases(),
                boundary=BOUNDARY,
                max_new_tokens=16,
                seed=-1,
            )

    def test_boundary_map_respected(self):
        stage = fewshot_stage(
            FewshotScriptedBackend(),
            (_format("tmpl_a"),),
            _demo_pool(),
            _eval_cases(),
            boundary=BOUNDARY,
            boundary_map={"tmpl_a": (999999,)},
            max_new_tokens=16,
            seed=7,
        )
        # A boundary id the backend never emits: nothing found, but the
        # stage still completes (verdicts graded on exact_match only).
        self.assertTrue(stage["available"])
        self.assertIsNotNone(stage["per_candidate"]["tmpl_a"])


class FewshotGainMalformedTests(unittest.TestCase):
    @staticmethod
    def _stage(**overrides):
        per_candidate = {
            "c0": {
                "n_eval": 1,
                "verdicts": {"f0": {"baseline": True, "order_a": False, "order_b": True}},
            }
        }
        base = {
            "available": True,
            "k": 2,
            "demo_ids": ["d0", "d1"],
            "n_eval": 1,
            "arms": ["baseline", "order_a", "order_b"],
            "per_candidate": per_candidate,
        }
        base.update(overrides)
        return base

    @staticmethod
    def _call(stage, **kwargs):
        params = {
            "primary_winner": "c0",
            "primary_decision": "selected",
            "candidates": ("c0",),
            "stage": stage,
        }
        params.update(kwargs)
        return fewshot_gain(**params)  # type: ignore[arg-type]

    def test_malformed_candidates(self):
        stage = self._stage()
        for bad in ((), [], "c0", ("c0", "c0"), ("",), (None,), ("c0", 42)):
            out = self._call(stage, candidates=bad)
            self.assertFalse(out["available"])
            self.assertEqual(out["detail"], "invalid_candidates")

    def test_unknown_winner(self):
        out = self._call(self._stage(), primary_winner="ghost")
        self.assertFalse(out["available"])
        self.assertEqual(out["detail"], "primary_winner_unknown")
        out = self._call(self._stage(), primary_winner=42)
        self.assertFalse(out["available"])

    def test_bad_primary_decision(self):
        for bad in ("", None, 42):
            out = self._call(self._stage(), primary_decision=bad)
            self.assertFalse(out["available"])
            self.assertEqual(out["detail"], "primary_decision_invalid")

    def test_bad_stage(self):
        self.assertEqual(self._call("nope")["detail"], "stage_invalid")
        self.assertEqual(self._call(None)["detail"], "stage_invalid")
        unavailable = {"available": False, "detail": "custom_detail"}
        out = self._call(unavailable)
        self.assertFalse(out["available"])
        self.assertEqual(out["detail"], "custom_detail")
        out = self._call({"available": False})
        self.assertEqual(out["detail"], "fewshot_unavailable")

    def test_bad_stage_shape(self):
        for overrides in (
            {"n_eval": 0},
            {"n_eval": "1"},
            {"demo_ids": []},
            {"demo_ids": [1]},
            {"demo_ids": "d0"},
            {"per_candidate": []},
            {"per_candidate": {1: None}},
        ):
            out = self._call(self._stage(**overrides))
            self.assertFalse(out["available"])
            self.assertEqual(out["detail"], "stage_invalid")

    def test_bad_entry_shapes(self):
        for entry in (
            42,
            {"n_eval": 2, "verdicts": {}},
            {"n_eval": 1, "verdicts": []},
            {"n_eval": 1, "verdicts": {1: {}}},
            {"n_eval": 1, "verdicts": {"f0": []}},
            {"n_eval": 1, "verdicts": {"f0": {"baseline": "yes"}}},
            {"n_eval": 1},
        ):
            out = self._call(self._stage(per_candidate={"c0": entry}))
            self.assertFalse(out["available"])
            self.assertEqual(out["detail"], "stage_invalid")

    def test_none_cell_excluded(self):
        stage = self._stage(per_candidate={"c0": None})
        out = self._call(stage)
        self.assertTrue(out["available"])
        self.assertEqual(out["excluded_candidates"], ["c0"])
        self.assertEqual(out["per_candidate"], {})

    def test_all_none_verdicts_excluded(self):
        entry = {
            "n_eval": 1,
            "verdicts": {"f0": {"baseline": None, "order_a": None, "order_b": None}},
        }
        out = self._call(self._stage(per_candidate={"c0": entry}))
        self.assertTrue(out["available"])
        self.assertEqual(out["excluded_candidates"], ["c0"])

    def test_empty_verdicts_excluded(self):
        entry = {"n_eval": 1, "verdicts": {}}
        out = self._call(self._stage(per_candidate={"c0": entry}))
        self.assertTrue(out["available"])
        self.assertEqual(out["excluded_candidates"], ["c0"])

    def test_partial_none_verdicts_included(self):
        # One fixture fully evaluable, one partially: gain from the
        # evaluable fixtures only, never a phantom.
        entry = {
            "n_eval": 2,
            "verdicts": {
                "f0": {"baseline": True, "order_a": False, "order_b": True},
                "f1": {"baseline": True, "order_a": None, "order_b": False},
            },
        }
        out = self._call(self._stage(n_eval=2, per_candidate={"c0": entry}))
        self.assertTrue(out["available"])
        cand = out["per_candidate"]["c0"]
        self.assertEqual(cand["n_evaluable"], 1)
        # order_a: f0 changed (1/1); order_b: f0 unchanged, f1 changed (1/2).
        self.assertEqual(cand["per_order"]["order_a"]["change_rate"], 1.0)
        self.assertEqual(cand["per_order"]["order_b"]["change_rate"], 0.5)
        self.assertAlmostEqual(cand["gain"], 0.75)
        self.assertAlmostEqual(cand["order_sensitivity"], 0.25)

    def test_fully_none_arms_excluded(self):
        entry = {
            "n_eval": 1,
            "verdicts": {"f0": {"baseline": True, "order_a": None, "order_b": None}},
        }
        out = self._call(self._stage(per_candidate={"c0": entry}))
        self.assertTrue(out["available"])
        # No fixture has all three arms: nothing to measure -> excluded.
        self.assertEqual(out["excluded_candidates"], ["c0"])
        self.assertEqual(out["per_candidate"], {})

    def test_included_candidate_always_has_both_rates(self):
        # Invariant the analysis relies on: n_full >= 1 guarantees each
        # order at least one evaluable fixture, so per-order rates and
        # gain are never None for an included candidate.
        entry = {
            "n_eval": 2,
            "verdicts": {
                "f0": {"baseline": True, "order_a": False, "order_b": True},
                "f1": {"baseline": None, "order_a": True, "order_b": True},
            },
        }
        out = self._call(self._stage(n_eval=2, per_candidate={"c0": entry}))
        cand = out["per_candidate"]["c0"]
        self.assertEqual(cand["n_evaluable"], 1)
        self.assertIsNotNone(cand["per_order"]["order_a"]["change_rate"])
        self.assertIsNotNone(cand["per_order"]["order_b"]["change_rate"])
        self.assertIsNotNone(cand["gain"])
        self.assertIsNotNone(cand["order_sensitivity"])


# ---------------------------------------------------------------------------
# 200-case brute-force oracle differential (gain/sensitivity arithmetic)
# ---------------------------------------------------------------------------


class FewshotOracleTests(unittest.TestCase):
    """Independent reimplementation of the gain/sensitivity math.

    Random verdict maps are fed through the real ``fewshot_gain`` and
    through an independently written oracle; every number must match.
    """

    @staticmethod
    def _oracle_rate(baseline, arm):
        pairs = [
            (baseline[f], arm[f])
            for f in baseline
            if f in arm and baseline[f] is not None and arm[f] is not None
        ]
        if not pairs:
            return None
        return sum(1 for b, a in pairs if b != a) / len(pairs)

    @classmethod
    def _oracle_candidate(cls, verdicts):
        baseline = {f: v["baseline"] for f, v in verdicts.items()}
        rates = {}
        for arm in ("order_a", "order_b"):
            arm_v = {f: v[arm] for f, v in verdicts.items()}
            rates[arm] = cls._oracle_rate(baseline, arm_v)
        full = sum(1 for v in verdicts.values() if all(x is not None for x in v.values()))
        if full == 0:
            return None
        valid = [r for r in rates.values() if r is not None]
        gain = sum(valid) / len(valid)
        mean = sum(valid) / len(valid)
        sens = math.sqrt(sum((r - mean) ** 2 for r in valid) / len(valid))
        return {"gain": gain, "order_sensitivity": sens, "n_evaluable": full, "rates": rates}

    @staticmethod
    def _oracle_arm(cands):
        # cands: {name: {fid: bool|None} | None} for one arm
        passing = []
        evaluable = 0
        for name, verdicts in cands.items():
            if (
                type(verdicts) is not dict
                or not verdicts
                or any(type(v) is not bool for v in verdicts.values())
            ):
                continue
            evaluable += 1
            if all(verdicts.values()):
                passing.append(name)
        if evaluable == 0:
            return "unavailable", None
        if len(passing) == 1:
            return "selected", passing[0]
        if passing:
            return "ambiguous", None
        return "no_candidate", None

    def test_200_case_differential(self):
        rng = random.Random(20261019)  # noqa: S311 -- deterministic test seed
        decisions = [
            "selected",
            "ambiguous",
            "no_candidate",
            "compatible_on_fixture_set",
            "held_out_failed",
        ]
        for case in range(200):
            n_cands = rng.randint(1, 3)
            names = tuple(f"c{i}" for i in range(n_cands))
            n_fixtures = rng.randint(1, 4)
            fids = [f"f{i}" for i in range(n_fixtures)]
            per_candidate = {}
            for name in names:
                roll = rng.random()
                if roll < 0.15:
                    per_candidate[name] = None
                    continue
                verdicts = {}
                for fid in fids:
                    cell = {}
                    for arm in ("baseline", "order_a", "order_b"):
                        pick = rng.random()
                        cell[arm] = None if pick < 0.2 else pick < 0.6
                    verdicts[fid] = cell
                per_candidate[name] = {"n_eval": n_fixtures, "verdicts": verdicts}
            stage = {
                "available": True,
                "k": 2,
                "demo_ids": ["d0", "d1"],
                "n_eval": n_fixtures,
                "arms": ["baseline", "order_a", "order_b"],
                "per_candidate": per_candidate,
            }
            decision = rng.choice(decisions)
            winner = rng.choice(names + (None,))
            if decision in ("ambiguous", "no_candidate"):
                winner = None
            got = fewshot_gain(winner, decision, names, stage)
            self.assertTrue(got["available"], f"case {case}")
            # Per-candidate numbers vs the oracle.
            for name in names:
                entry = per_candidate[name]
                expected = None if entry is None else self._oracle_candidate(entry["verdicts"])
                if expected is None:
                    self.assertIn(name, got["excluded_candidates"], f"case {case}")
                    self.assertNotIn(name, got["per_candidate"], f"case {case}")
                    continue
                self.assertNotIn(name, got["excluded_candidates"], f"case {case}")
                cand = got["per_candidate"][name]
                self.assertAlmostEqual(cand["gain"], expected["gain"], msg=f"case {case}")
                self.assertAlmostEqual(
                    cand["order_sensitivity"],
                    expected["order_sensitivity"],
                    msg=f"case {case}",
                )
                self.assertEqual(cand["n_evaluable"], expected["n_evaluable"], f"case {case}")
                for arm in ("order_a", "order_b"):
                    self.assertAlmostEqual(
                        cand["per_order"][arm]["change_rate"],
                        expected["rates"][arm],
                        msg=f"case {case}",
                    )
            self.assertEqual(
                sorted(got["excluded_candidates"]),
                got["excluded_candidates"],
                f"case {case}",
            )
            # Per-arm selection vs the oracle.
            for arm in ("baseline", "order_a", "order_b"):
                arm_verdicts = {
                    name: (
                        None
                        if per_candidate[name] is None
                        else {fid: per_candidate[name]["verdicts"][fid][arm] for fid in fids}
                    )
                    for name in names
                }
                exp_decision, exp_winner = self._oracle_arm(arm_verdicts)
                cell = got["per_arm"][arm]
                self.assertEqual(cell["decision"], exp_decision, f"case {case}")
                self.assertEqual(cell["winner"], exp_winner, f"case {case}")
            # Flips vs the oracle's primary mapping.
            if decision in ("compatible_on_fixture_set", "held_out_failed"):
                primary = ("selected", winner)
            else:
                primary = (decision, winner)
            expected_flips = [
                arm
                for arm in ("baseline", "order_a", "order_b")
                if got["per_arm"][arm]["decision"] != "unavailable"
                and (got["per_arm"][arm]["decision"], got["per_arm"][arm]["winner"]) != primary
            ]
            self.assertEqual([f["arm"] for f in got["flips"]], expected_flips, f"case {case}")


# ---------------------------------------------------------------------------
# Mutation gates (tempdir-module; the real tree is never rewritten)
# ---------------------------------------------------------------------------


class FewshotMutationGateTests(unittest.TestCase):
    """Each gate reverts one safety property; the scenario must catch it."""

    def _run_gate(self, module_name, name, old_text, new_text, scenario):
        mod = __import__(f"model_protocol_lab.{module_name}", fromlist=["x"])
        source = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertEqual(source.count(old_text), 1, f"mutant {name}: anchor not unique")
        mutated = source.replace(old_text, new_text)
        with tempfile.TemporaryDirectory(prefix="c160_mut_", dir="/tmp") as tmpdir:
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

    @staticmethod
    def _gain_mean_scenario(mod):
        stage = {
            "available": True,
            "k": 2,
            "demo_ids": ["d0", "d1"],
            "n_eval": 2,
            "arms": ["baseline", "order_a", "order_b"],
            "per_candidate": {
                "c": {
                    "n_eval": 2,
                    "verdicts": {
                        # order_a flips both (rate 1.0), order_b flips
                        # neither (rate 0.0): mean gain is 0.5.
                        "f0": {"baseline": True, "order_a": False, "order_b": True},
                        "f1": {"baseline": True, "order_a": False, "order_b": True},
                    },
                }
            },
        }
        got = mod.fewshot_gain("c", "selected", ("c",), stage)
        assert got["per_candidate"]["c"]["gain"] == 0.5, got

    def test_mutant_gain_max_caught(self):
        self._run_gate(
            "experiments",
            "gain_max",
            "        gain = sum(valid_rates) / len(valid_rates)",
            "        gain = max(valid_rates)  # MUTANT",
            self._gain_mean_scenario,
        )

    @staticmethod
    def _sensitivity_pstdev_scenario(mod):
        assert mod.order_sensitivity((1.0, 0.0)) == 0.5
        assert mod.order_sensitivity((0.25, 0.75)) == 0.25

    def test_mutant_sensitivity_sample_stdev_caught(self):
        self._run_gate(
            "fewshot",
            "sensitivity_sample_stdev",
            "    return math.sqrt(sum((rate - mean) ** 2 for rate in clean) / len(clean))",
            "    return math.sqrt(sum((rate - mean) ** 2 for rate in clean)"
            " / (len(clean) - 1))  # MUTANT",
            self._sensitivity_pstdev_scenario,
        )

    @staticmethod
    def _change_rate_none_excluded_scenario(mod):
        out = mod.change_rate({"f": True}, {"f": None})
        assert out["n_evaluable"] == 0, out
        assert out["rate"] is None, out

    def test_mutant_none_verdicts_counted_caught(self):
        self._run_gate(
            "fewshot",
            "none_verdicts_counted",
            "    evaluable = [fid for fid in fids"
            " if baseline[fid] is not None and arm[fid] is not None]",
            "    evaluable = [fid for fid in fids]  # MUTANT",
            self._change_rate_none_excluded_scenario,
        )

    @staticmethod
    def _diff_payload(winner):
        return {
            "available": True,
            "k": 2,
            "demo_ids": ["d0", "d1"],
            "n_eval": 1,
            "primary_winner": "c",
            "primary_decision": "selected",
            "per_candidate": {
                "c": {
                    "gain": 1.0,
                    "order_sensitivity": 0.0,
                    "n_evaluable": 1,
                    "per_order": {
                        "order_a": {"change_rate": 1.0, "n_changed": 1, "n_evaluable": 1},
                        "order_b": {"change_rate": 1.0, "n_changed": 1, "n_evaluable": 1},
                    },
                }
            },
            "excluded_candidates": [],
            "per_arm": {
                "baseline": {"decision": "selected", "winner": "c"},
                "order_a": {"decision": "selected", "winner": winner},
                "order_b": {"decision": "selected", "winner": "c"},
            },
            "flips": [],
        }

    def _arm_drift_scenario(self, mod):
        old = self._diff_payload("c")
        new = self._diff_payload(None)
        diff = mod.diff_fewshot(old, new)
        assert diff["available"] == "both", diff
        assert diff["changed"] is True, diff
        # Single cause: only the arm drift disjunct fires.
        assert diff["arm_drift"], diff
        assert not diff["gain_drift"], diff
        assert not diff["sensitivity_drift"], diff

    def test_mutant_diff_arm_drift_ignored_caught(self):
        self._run_gate(
            "diffreports",
            "arm_drift_ignored",
            "        or fewshot_arm_drift\n",
            "        # or fewshot_arm_drift  # MUTANT\n",
            self._arm_drift_scenario,
        )


# ---------------------------------------------------------------------------
# diff_fewshot comparison
# ---------------------------------------------------------------------------


def _diff_candidate(gain=1.0, sensitivity=0.0, n_evaluable=2):
    return {
        "gain": gain,
        "order_sensitivity": sensitivity,
        "n_evaluable": n_evaluable,
        "per_order": {
            "order_a": {"change_rate": gain, "n_changed": 1, "n_evaluable": 1},
            "order_b": {"change_rate": gain, "n_changed": 1, "n_evaluable": 1},
        },
    }


def _diff_payload(gain=1.0, sensitivity=0.0, winner="c", excluded=()):
    return {
        "available": True,
        "k": 2,
        "demo_ids": ["d0", "d1"],
        "n_eval": 1,
        "primary_winner": "c",
        "primary_decision": "selected",
        "per_candidate": {"c": _diff_candidate(gain, sensitivity)},
        "excluded_candidates": list(excluded),
        "per_arm": {
            "baseline": {"decision": "selected", "winner": "c"},
            "order_a": {"decision": "no_candidate", "winner": winner},
            "order_b": {"decision": "no_candidate", "winner": winner},
        },
        "flips": [],
    }


class DiffFewshotTests(unittest.TestCase):
    def test_identical_payloads_unchanged(self):
        diff = diffreports.diff_fewshot(_diff_payload(), _diff_payload())
        self.assertEqual(diff["available"], "both")
        self.assertFalse(diff["changed"])

    def test_availability_matrix(self):
        good = _diff_payload()
        cases = [
            (good, good, "both"),
            (good, {"available": False}, "old"),
            ({"available": False}, good, "new"),
            ({"available": False}, {"available": False}, "neither"),
            (good, None, "old"),
            (None, None, "neither"),
        ]
        for old, new, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(diffreports.diff_fewshot(old, new)["available"], expected)

    def test_single_cause_pins(self):
        # Gain drift only.
        diff = diffreports.diff_fewshot(_diff_payload(gain=1.0), _diff_payload(gain=0.0))
        self.assertTrue(diff["changed"])
        self.assertEqual(len(diff["gain_drift"]), 1)
        self.assertEqual(diff["gain_drift"][0]["candidate"], "c")
        self.assertEqual(diff["gain_drift"][0]["old"], 1.0)
        self.assertEqual(diff["gain_drift"][0]["new"], 0.0)
        self.assertFalse(diff["sensitivity_drift"])
        self.assertFalse(diff["arm_drift"])
        self.assertFalse(diff["candidates_added"])
        # Sensitivity drift only.
        diff = diffreports.diff_fewshot(
            _diff_payload(sensitivity=0.0), _diff_payload(sensitivity=0.5)
        )
        self.assertTrue(diff["changed"])
        self.assertEqual(len(diff["sensitivity_drift"]), 1)
        self.assertFalse(diff["gain_drift"])
        # Arm drift only (the winner param covers both demo-order arms).
        diff = diffreports.diff_fewshot(_diff_payload(winner="c"), _diff_payload(winner=None))
        self.assertTrue(diff["changed"])
        self.assertEqual(len(diff["arm_drift"]), 2)
        self.assertEqual({e["arm"] for e in diff["arm_drift"]}, {"order_a", "order_b"})
        entry = diff["arm_drift"][0]
        self.assertEqual(entry["old_winner"], "c")
        self.assertIsNone(entry["new_winner"])
        self.assertFalse(diff["gain_drift"])
        self.assertFalse(diff["sensitivity_drift"])
        # Candidates added/removed.
        old = _diff_payload()
        new = _diff_payload()
        new["per_candidate"]["c2"] = _diff_candidate()
        diff = diffreports.diff_fewshot(old, new)
        self.assertTrue(diff["changed"])
        self.assertEqual(diff["candidates_added"], ["c2"])
        self.assertFalse(diff["gain_drift"])
        # skipcq: PYL-W1114 - the reversed diff is the assertion: symmetry is what is audited
        diff = diffreports.diff_fewshot(new, old)
        self.assertEqual(diff["candidates_removed"], ["c2"])
        # Excluded added/removed.
        diff = diffreports.diff_fewshot(_diff_payload(), _diff_payload(excluded=["c"]))
        self.assertTrue(diff["changed"])
        self.assertEqual(diff["excluded_added"], ["c"])
        diff = diffreports.diff_fewshot(_diff_payload(excluded=["c"]), _diff_payload())
        self.assertEqual(diff["excluded_removed"], ["c"])

    def test_none_gain_drift_detected(self):
        old = _diff_payload()
        new = _diff_payload()
        new["per_candidate"]["c"]["gain"] = None
        diff = diffreports.diff_fewshot(old, new)
        self.assertTrue(diff["changed"])
        self.assertEqual(diff["gain_drift"][0]["new"], None)

    def test_valid_fewshot_rejects_hostile(self):
        good = _diff_payload()
        self.assertTrue(diffreports._valid_fewshot(good))
        hostile = [
            None,
            "x",
            42,
            [],
            {"available": False},
            {**good, "demo_ids": []},
            {**good, "demo_ids": [1]},
            {**good, "k": "2"},
            {**good, "n_eval": 1.5},
            {**good, "per_candidate": {"c": {**_diff_candidate(), "gain": 1.5}}},
            {**good, "per_candidate": {"c": {**_diff_candidate(), "gain": "high"}}},
            {**good, "per_candidate": {"c": {**_diff_candidate(), "order_sensitivity": 0.6}}},
            {**good, "per_candidate": {"c": {**_diff_candidate(), "order_sensitivity": -0.1}}},
            {**good, "per_candidate": {"c": "nope"}},
            {**good, "per_candidate": {1: _diff_candidate()}},
            {**good, "per_arm": {"baseline": {"decision": 1, "winner": "c"}}},
            {**good, "per_arm": {"baseline": {"decision": "selected", "winner": 2}}},
            {**good, "per_arm": []},
            {**good, "flips": {}},
            {**good, "flips": ["x"]},
            {**good, "excluded_candidates": [1]},
            {**good, "excluded_candidates": "c"},
        ]
        for i, payload in enumerate(hostile):
            with self.subTest(i=i):
                self.assertFalse(diffreports._valid_fewshot(payload))

    def test_diff_feeds_identical(self):
        # A fewshot-only change flips identical; asymmetric probe flips it too.
        def report(**kwargs):
            return {
                "decision": "select_best",
                "selected_candidate": "c",
                "search": [],
                "fewshot": _diff_payload(**kwargs),
            }

        diff = diffreports.diff_reports(report(gain=1.0), report(gain=0.0))
        self.assertTrue(diff["summary"]["fewshot_changed"])
        self.assertFalse(diff["summary"]["identical"])
        diff2 = diffreports.diff_reports(
            report(gain=1.0),
            {"decision": "select_best", "selected_candidate": "c", "search": []},
        )
        self.assertTrue(diff2["summary"]["fewshot_asymmetric"])
        self.assertFalse(diff2["summary"]["identical"])
        diff3 = diffreports.diff_reports(report(gain=1.0), report(gain=1.0))
        self.assertFalse(diff3["summary"]["fewshot_changed"])
        self.assertTrue(diff3["summary"]["identical"])


# ---------------------------------------------------------------------------
# stderr summary rendering
# ---------------------------------------------------------------------------


class FewshotRenderSummaryTests(unittest.TestCase):
    @staticmethod
    def _diff():
        old = _diff_payload()
        new = _diff_payload(gain=0.0, sensitivity=0.5, winner=None, excluded=["c"])
        new["per_candidate"]["c2"] = _diff_candidate()
        return diffreports.diff_fewshot(old, new)

    def test_render_summary_lines(self):
        lines = diffreports._render_fewshot_summary(self._diff())
        text = "\n".join(lines)
        self.assertIn("Few-shot gain drift (1):", text)
        self.assertIn("1.00 -> 0.00", text)
        self.assertIn("order-sensitivity drift", text)
        self.assertIn("Few-shot arm drift (order_a)", text)
        self.assertIn("Few-shot candidates added: c2", text)
        self.assertIn("Few-shot newly excluded: c", text)

    def test_render_summary_sanitizes_hostile_ids(self):
        old = _diff_payload()
        new = _diff_payload(gain=0.0)
        evil = "\x1b[31mRED\x1b[0m"
        new["per_candidate"] = {evil: _diff_candidate(gain=0.0)}
        new["per_candidate"][evil]["gain"] = 0.0
        old["per_candidate"] = {evil: _diff_candidate(gain=1.0)}
        diff = diffreports.diff_fewshot(old, new)
        lines = diffreports._render_fewshot_summary(diff)
        text = "\n".join(lines)
        self.assertNotIn("\x1b", text)

    def test_render_summary_availability(self):
        self.assertEqual(diffreports._render_fewshot_summary(None), [])
        self.assertEqual(diffreports._render_fewshot_summary({"available": "neither"}), [])
        lines = diffreports._render_fewshot_summary({"available": "old"})
        self.assertTrue(any("new run did not record" in line for line in lines))
        lines = diffreports._render_fewshot_summary({"available": "new"})
        self.assertTrue(any("old run did not record" in line for line in lines))


# ---------------------------------------------------------------------------
# CLI probe wiring
# ---------------------------------------------------------------------------


class FewshotCliProbeTests(unittest.TestCase):
    def test_flag_registered(self):
        from model_protocol_lab import cli

        parser = cli.build_parser()
        required = [
            "--model",
            "m",
            "--dataset",
            "d",
            "--family",
            "llama3",
            "--boundary-ids",
            "124",
            "--output",
            "o",
        ]
        args = parser.parse_args(["run", "--few-shot", *required])
        self.assertTrue(args.few_shot)
        args = parser.parse_args(["run", *required])
        self.assertFalse(args.few_shot)

    def test_probe_records_available_payload(self):
        from model_protocol_lab import cli

        formats = (_format("tmpl_a"),)
        result = {"decision": "selected", "selected_candidate": "tmpl_a"}
        cli._run_fewshot_probe(
            FewshotScriptedBackend(),
            formats,
            result,
            _demo_pool(),
            _eval_cases(),
            boundary=BOUNDARY,
            boundary_map=None,
            max_new_tokens=16,
            seed=7,
        )
        payload = result["fewshot"]
        self.assertTrue(payload["available"])
        self.assertEqual(payload["k"], FEWSHOT_K)
        self.assertEqual(set(payload["demo_ids"]), {"f_d0", "f_d1"})
        self.assertEqual(payload["per_candidate"]["tmpl_a"]["gain"], 1.0)
        self.assertEqual(payload["per_arm"]["baseline"]["decision"], "selected")

    def test_probe_never_raises_on_broken_backend(self):
        from model_protocol_lab import cli

        result = {}
        cli._run_fewshot_probe(
            None,
            (),
            result,
            (),
            (),
            boundary=None,
            boundary_map=None,
            max_new_tokens=8,
            seed=7,
        )
        self.assertEqual(result["fewshot"]["detail"], "fewshot_unavailable")


# ---------------------------------------------------------------------------
# Reporting: HTML section + TOC anchor
# ---------------------------------------------------------------------------


class FewshotReportingTests(unittest.TestCase):
    @staticmethod
    def _result():
        return {
            "fewshot": {
                "available": True,
                "k": 2,
                "demo_ids": ["f_d0", "f_d1"],
                "n_eval": 2,
                "primary_winner": "tmpl_a",
                "primary_decision": "selected",
                "per_candidate": {
                    "tmpl_a": {
                        "gain": 1.0,
                        "order_sensitivity": 0.0,
                        "n_evaluable": 2,
                        "per_order": {
                            "order_a": {"change_rate": 1.0, "n_changed": 2, "n_evaluable": 2},
                            "order_b": {"change_rate": 1.0, "n_changed": 2, "n_evaluable": 2},
                        },
                    }
                },
                "excluded_candidates": [],
                "per_arm": {
                    "baseline": {"decision": "selected", "winner": "tmpl_a", "n_evaluable": 1},
                    "order_a": {"decision": "no_candidate", "winner": None, "n_evaluable": 1},
                    "order_b": {"decision": "no_candidate", "winner": None, "n_evaluable": 1},
                },
                "flips": [
                    {
                        "arm": "order_a",
                        "decision_from": "selected",
                        "decision_to": "no_candidate",
                        "winner_from": "tmpl_a",
                        "winner_to": None,
                    }
                ],
            }
        }

    def test_section_renders_with_anchor(self):
        from model_protocol_lab.reporting import _fewshot_section

        html = _fewshot_section(self._result())
        self.assertIn('id="few-shot"', html)
        self.assertIn("1.00", html)
        self.assertIn("tmpl_a", html)
        self.assertIn("f_d0", html)
        self.assertIn("order_a", html)

    def test_section_empty_when_unavailable(self):
        from model_protocol_lab.reporting import _fewshot_section

        self.assertEqual(_fewshot_section({}), "")
        self.assertEqual(_fewshot_section({"fewshot": {"available": False}}), "")
        self.assertEqual(_fewshot_section({"fewshot": {"available": True}}), "")
        self.assertEqual(
            _fewshot_section({"fewshot": {"available": True, "per_candidate": []}}),
            "",
        )

    def test_section_escapes_hostile_names(self):
        from model_protocol_lab.reporting import _fewshot_section

        result = self._result()
        result["fewshot"]["per_candidate"] = {
            "<script>alert(1)</script>": result["fewshot"]["per_candidate"]["tmpl_a"],
        }
        result["fewshot"]["excluded_candidates"] = ["<img src=x onerror=y>"]
        result["fewshot"]["demo_ids"] = ["<b>demo</b>"]
        html = _fewshot_section(result)
        self.assertNotIn("<script>", html)
        self.assertNotIn("<img src=x", html)
        self.assertNotIn("<b>demo</b>", html)

    def test_section_handles_missing_optionals(self):
        from model_protocol_lab.reporting import _fewshot_section

        result = self._result()
        del result["fewshot"]["per_arm"]
        del result["fewshot"]["flips"]
        del result["fewshot"]["demo_ids"]
        html = _fewshot_section(result)
        self.assertIn('id="few-shot"', html)

    def test_toc_links_to_section(self):
        from model_protocol_lab import reporting

        minimal = {
            "config": {},
            "fewshot": self._result()["fewshot"],
        }
        html = reporting.render_html(minimal)
        self.assertIn('href="#few-shot"', html)
        self.assertIn('id="few-shot"', html)


# ---------------------------------------------------------------------------
# Self red-team: hostile inputs, crash-proofing, escape hygiene
# ---------------------------------------------------------------------------


class FewshotRedTeamTests(unittest.TestCase):
    def test_gain_never_raises_on_hostile_stage(self):
        hostile_stages = [
            {"available": True},
            {"available": True, "n_eval": 1, "demo_ids": ["d"], "per_candidate": {"c": None}},
            {
                "available": True,
                "k": 2,
                "demo_ids": ["d0", "d1"],
                "n_eval": 1,
                "arms": ["baseline"],
                "per_candidate": {
                    "c": {
                        "n_eval": 1,
                        "verdicts": {"f": {"baseline": "yes", "order_a": 1, "order_b": []}},
                    }
                },
            },
            {
                "available": True,
                "k": 2,
                "demo_ids": ["d0", "d1"],
                "n_eval": 1,
                "per_candidate": {
                    "\x00\x1b[2Jc": {
                        "n_eval": 1,
                        "verdicts": {"f": {"baseline": True, "order_a": True, "order_b": True}},
                    }
                },
            },
        ]
        for i, stage in enumerate(hostile_stages):
            with self.subTest(i=i):
                out = fewshot_gain("c", "selected", ("c",), stage)
                self.assertIs(type(out), dict)
                self.assertIn("available", out)

    def test_stage_never_raises_on_hostile_backend(self):
        class HostileBackend(FewshotScriptedBackend):
            @staticmethod
            def encode(text):
                raise RuntimeError("encode boom")

            @staticmethod
            def generate(input_ids, *, max_new_tokens, seed, cancel):
                raise RuntimeError("generate boom")

        stage = fewshot_stage(
            HostileBackend(),
            (_format("tmpl_a"),),
            _demo_pool(),
            _eval_cases(),
            boundary=BOUNDARY,
            max_new_tokens=16,
            seed=7,
        )
        self.assertTrue(stage["available"])
        self.assertIsNone(stage["per_candidate"]["tmpl_a"])

    def test_pure_functions_reject_bad_id_types(self):
        with self.assertRaises(ValueError):
            original_fixture_id("fewshot/perm0")  # no fixture part
        with self.assertRaises(ValueError):
            change_rate({b"f": True}, {"f": False})  # bytes key

    def test_control_char_ids_are_plain_data(self):
        out = change_rate({"\x00\x1b[2Jf": True}, {"\x00\x1b[2Jf": False})
        self.assertEqual(out["rate"], 1.0)
        self.assertEqual(original_fixture_id("fewshot/perm0/\x00f"), "\x00f")

    def test_diff_never_raises_on_hostile(self):
        hostile = [
            ({"available": True}, {"available": True}),
            ({"available": True, "per_candidate": {"c": None}}, None),
            ([], {}),
        ]
        for i, (old, new) in enumerate(hostile):
            with self.subTest(i=i):
                diff = diffreports.diff_fewshot(old, new)
                self.assertIn("available", diff)
                lines = diffreports._render_fewshot_summary(diff)
                self.assertIs(type(lines), list)

    def test_stderr_escapes_stripped(self):
        old = _diff_payload()
        new = _diff_payload(gain=0.0)
        evil = "\x1b]0;pwned\x07"
        old["per_candidate"] = {evil: _diff_candidate(gain=1.0)}
        new["per_candidate"] = {evil: _diff_candidate(gain=0.0)}
        diff = diffreports.diff_fewshot(old, new)
        text = "\n".join(diffreports._render_fewshot_summary(diff))
        self.assertNotIn("\x1b", text)
        self.assertNotIn("\x07", text)
