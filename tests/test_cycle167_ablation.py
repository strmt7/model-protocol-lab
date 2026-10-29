"""Cycle 167: template field ablation probe.

Tests the pure ``ablation`` module (field-neutral swaps, arm enumeration,
per-arm flip attribution), ``experiments.ablation_stage`` (top-2 focus,
per-candidate boundaries, fail-closed arms), the CLI runner, the HTML
report section, and ``diffreports.diff_ablation`` -- plus a mutation gate
pinning the ablation ``changed`` line's textual uniqueness.
"""

import unittest

from model_protocol_lab import ablation as ab
from model_protocol_lab.core import Fixture, Format, Generation, Message


def _fmt(name="t", **overrides):
    """Fmt."""
    fields = {
        "name": name,
        "bos": "<s>",
        "user_prefix": "U: ",
        "assistant_prefix": "A: ",
        "turn_end": "\n",
        "repeat_bos": True,
        "user_turn_end": "</u>",
        "assistant_turn_end": "</a>",
        "open_suffix": ">> ",
    }
    fields.update(overrides)
    return Format(**fields)


def _fixtures():
    """Fixtures."""
    return (
        Fixture(
            id="f1",
            messages=(Message(role="user", content="Say MAGICWORD"),),
            expected="BLUE",
            group="g",
        ),
        Fixture(
            id="f2",
            messages=(Message(role="user", content="Say MAGICWORD loudly"),),
            expected="BLUE",
            group="g",
        ),
    )


class PrefixBackend:
    """Scripted backend: BLUE iff the 'U: ' role prefix survived rendering.

    Lets the user_prefix ablation arm flip pass->fail deterministically:
    the base format renders 'U: Say MAGICWORD', the ablated arm renders
    'Say MAGICWORD'.
    """

    fingerprint = "test-prefix/v1"

    @staticmethod
    def encode(text):
        """Encode."""
        if type(text) is not str:
            raise ValueError("text must be a string")
        return tuple(map(ord, text))

    @staticmethod
    def decode(ids):
        """Decode."""
        return "".join(map(chr, ids))

    def generate(self, input_ids, *, max_new_tokens, seed, cancel=None):
        """Generate."""
        prompt = self.decode(input_ids)
        ans = "BLUE" if ("U: " in prompt and "MAGICWORD" in prompt) else "WRONG"
        # reason "limit" requires exactly max_new_tokens ids (core contract);
        # the answer is cut at the boundary so padding is inert.
        ids = tuple(map(ord, ans + chr(7)))
        ids = ids + (1,) * (max_new_tokens - len(ids))
        return Generation(ids, "limit")


class AblationPureTests(unittest.TestCase):
    """Ablation pure tests."""

    def test_ablation_fields_pinned(self):
        """Test ablation fields pinned."""
        self.assertEqual(
            ab.ABLATION_FIELDS,
            (
                "bos",
                "user_prefix",
                "assistant_prefix",
                "turn_end",
                "repeat_bos",
                "user_turn_end",
                "assistant_turn_end",
                "open_suffix",
            ),
        )
        self.assertNotIn("name", ab.ABLATION_FIELDS)

    def test_neutral_values(self):
        """Test neutral values."""
        self.assertEqual(ab.neutral_value("bos"), "")
        self.assertEqual(ab.neutral_value("user_prefix"), "")
        self.assertEqual(ab.neutral_value("repeat_bos"), False)
        self.assertIsNone(ab.neutral_value("user_turn_end"))
        self.assertIsNone(ab.neutral_value("assistant_turn_end"))
        self.assertIsNone(ab.neutral_value("open_suffix"))
        with self.assertRaises(ValueError):
            ab.neutral_value("name")
        with self.assertRaises(ValueError):
            ab.neutral_value(42)

    def test_ablate_format_swaps_one_field(self):
        """Test ablate format swaps one field."""
        base = _fmt()
        for field in ab.ABLATION_FIELDS:
            arm = ab.ablate_format(base, field)
            self.assertIs(type(arm), Format)
            self.assertEqual(getattr(arm, field), ab.NEUTRAL_VALUES[field])
            # Every other field is untouched.
            for other in ab.ABLATION_FIELDS:
                if other != field:
                    self.assertEqual(getattr(arm, other), getattr(base, other))
            self.assertEqual(arm.name, base.name)

    def test_ablate_format_rejects_bad_inputs(self):
        """Test ablate format rejects bad inputs."""
        with self.assertRaises(ValueError):
            ab.ablate_format("not-a-format", "bos")
        with self.assertRaises(ValueError):
            ab.ablate_format(_fmt(), "name")
        with self.assertRaises(ValueError):
            ab.ablation_arms("not-a-format")
        with self.assertRaises(ValueError):
            ab.inapplicable_fields(None)

    def test_ablation_arms_skips_neutral_fields(self):
        """Test ablation arms skips neutral fields."""
        base = _fmt()
        arms = ab.ablation_arms(base)
        fields = [a.field for a in arms]
        # All eight fields are non-neutral -> one arm each, in field order.
        self.assertEqual(fields, list(ab.ABLATION_FIELDS))
        for arm in arms:
            self.assertEqual(arm.original, getattr(base, arm.field))
            self.assertEqual(getattr(arm.ablated, arm.field), ab.NEUTRAL_VALUES[arm.field])
        # A field already at its neutral value yields no arm.
        base2 = _fmt(repeat_bos=False, open_suffix=None)
        fields2 = [a.field for a in ab.ablation_arms(base2)]
        self.assertNotIn("repeat_bos", fields2)
        self.assertNotIn("open_suffix", fields2)
        self.assertEqual(len(fields2), 6)

    def test_ablation_arms_all_neutral(self):
        """Test ablation arms all neutral."""
        base = _fmt(
            bos="",
            user_prefix="",
            assistant_prefix="",
            turn_end="",
            repeat_bos=False,
            user_turn_end=None,
            assistant_turn_end=None,
            open_suffix=None,
        )
        self.assertEqual(ab.ablation_arms(base), ())
        self.assertEqual(set(ab.inapplicable_fields(base)), set(ab.ABLATION_FIELDS))

    def test_inapplicable_fields(self):
        """Test inapplicable fields."""
        base = _fmt(
            repeat_bos=False,
            user_turn_end=None,
            assistant_turn_end=None,
            open_suffix=None,
        )
        inapplicable = ab.inapplicable_fields(base)
        self.assertEqual(
            set(inapplicable),
            {"repeat_bos", "user_turn_end", "assistant_turn_end", "open_suffix"},
        )

    def test_summarize_arm_no_flips(self):
        """Test summarize arm no flips."""
        summary = ab.summarize_arm((True, True), (True, True), ("f1", "f2"))
        self.assertEqual(
            summary,
            {
                "n_fixtures": 2,
                "n_flips": 0,
                "pass_to_fail": 0,
                "fail_to_pass": 0,
                "flipped_fixture_ids": (),
                "selection_bar_held": True,
            },
        )

    def test_summarize_arm_both_directions(self):
        """Test summarize arm both directions."""
        summary = ab.summarize_arm((True, False, True), (False, True, True), ("a", "b", "c"))
        self.assertEqual(summary["n_flips"], 2)
        self.assertEqual(summary["pass_to_fail"], 1)
        self.assertEqual(summary["fail_to_pass"], 1)
        self.assertEqual(summary["flipped_fixture_ids"], ("a", "b"))
        self.assertFalse(summary["selection_bar_held"])

    def test_summarize_arm_selection_bar_held(self):
        """Test summarize arm selection bar held."""
        # Base fails everywhere, arm passes everywhere: bar held by the arm.
        summary = ab.summarize_arm((False, False), (True, True), ("f1", "f2"))
        self.assertTrue(summary["selection_bar_held"])
        self.assertEqual(summary["fail_to_pass"], 2)

    def test_summarize_arm_rejects_bad_inputs(self):
        """Test summarize arm rejects bad inputs."""
        with self.assertRaises(ValueError):
            ab.summarize_arm([True], (True,), ("f1",))
        with self.assertRaises(ValueError):
            ab.summarize_arm((1,), (True,), ("f1",))
        with self.assertRaises(ValueError):
            ab.summarize_arm((True,), (True,), ["f1"])
        with self.assertRaises(ValueError):
            ab.summarize_arm((True,), (True,), (42,))
        with self.assertRaises(ValueError):
            ab.summarize_arm((True, False), (True,), ("f1", "f2"))


class AblationStageTests(unittest.TestCase):
    """Ablation stage tests."""

    def test_user_prefix_is_load_bearing(self):
        """Test user prefix is load bearing."""
        from model_protocol_lab import experiments

        stage = experiments.ablation_stage(
            (_fmt("prefixed"),),
            _fixtures(),
            PrefixBackend(),
            boundary=(7,),
        )
        self.assertEqual(stage["n_focus"], 1)
        rep = stage["candidates"]["prefixed"]
        self.assertEqual(rep["status"], "ok")
        self.assertTrue(rep["base_passes_all"])
        arm = rep["arms"]["user_prefix"]
        self.assertEqual(arm["status"], "ok")
        self.assertEqual(arm["pass_to_fail"], 2)
        self.assertEqual(arm["n_flips"], 2)
        self.assertFalse(arm["selection_bar_held"])
        self.assertIn("user_prefix", rep["load_bearing_fields"])
        self.assertTrue(rep["any_attribution"])
        # A field the backend does not read about causes no flips.
        bos_arm = rep["arms"]["bos"]
        self.assertEqual(bos_arm["n_flips"], 0)
        self.assertTrue(bos_arm["selection_bar_held"])

    def test_top_two_focus_tie_broken_by_name(self):
        """Test top two focus tie broken by name."""
        from model_protocol_lab import experiments

        formats = (_fmt("c"), _fmt("b"), _fmt("a"))
        stage = experiments.ablation_stage(formats, _fixtures(), PrefixBackend(), boundary=(7,))
        self.assertEqual(stage["n_focus"], 2)
        # All three pass everything; ties broken by name -> a, b.
        self.assertEqual(set(stage["candidates"]), {"a", "b"})

    def test_stage_honors_boundary_map(self):
        """Test stage honors boundary map."""
        from model_protocol_lab import experiments

        stage = experiments.ablation_stage(
            (_fmt("x"), _fmt("y")),
            _fixtures(),
            PrefixBackend(),
            boundary=(7,),
            boundary_map={"x": (7,), "y": (8,)},
        )
        self.assertEqual(stage["candidates"]["x"]["boundary"], [7])
        self.assertEqual(stage["candidates"]["y"]["boundary"], [8])

    def test_stage_rejects_bad_inputs(self):
        """Test stage rejects bad inputs."""
        from model_protocol_lab import experiments

        with self.assertRaises(ValueError):
            experiments.ablation_stage((), _fixtures(), PrefixBackend(), boundary=(7,))
        with self.assertRaises(ValueError):
            experiments.ablation_stage((_fmt(),), (), PrefixBackend(), boundary=(7,))

    def test_failed_candidate_degrades(self):
        """Test failed candidate degrades."""
        from model_protocol_lab import experiments

        class ExplodingBackend(PrefixBackend):
            """Exploding backend."""

            fingerprint = "test-exploding/v1"

            @staticmethod
            def generate(input_ids, *, max_new_tokens, seed, cancel=None):
                """Generate."""
                raise RuntimeError("boom")

        stage = experiments.ablation_stage(
            (_fmt("x"),), _fixtures(), ExplodingBackend(), boundary=(7,)
        )
        # No candidate scores -> empty focus.
        self.assertEqual(stage["candidates"], {})
        self.assertEqual(stage["n_focus"], 0)


class AblationCliTests(unittest.TestCase):
    """Ablation cli tests."""

    def test_probe_runner_records_payload(self):
        """Test probe runner records payload."""
        from model_protocol_lab import cli

        result: dict = {}
        cli._run_ablation_probe(
            result,
            (_fmt("t1"),),
            backend=PrefixBackend(),
            search_fixtures=_fixtures(),
            boundary=(7,),
            boundary_map=None,
            max_new_tokens=8,
            seed=0,
        )
        payload = result["ablation"]
        self.assertTrue(payload["available"])
        self.assertIn("t1", payload["candidates"])
        self.assertEqual(payload["n_focus"], 1)

    def test_probe_runner_never_raises(self):
        """Test probe runner never raises."""
        from model_protocol_lab import cli

        result: dict = {}
        cli._run_ablation_probe(
            result,
            (),
            backend=None,
            search_fixtures=(),
            boundary=(7,),
            boundary_map=None,
            max_new_tokens=8,
            seed=0,
        )
        self.assertFalse(result["ablation"]["available"])
        self.assertEqual(result["ablation"]["detail"], "ablation_unavailable")


class AblationReportingTests(unittest.TestCase):
    """Ablation reporting tests."""

    @staticmethod
    def _result(candidates):
        """Result."""
        return {"ablation": {"available": True, "candidates": candidates}}

    def test_section_renders_load_bearing(self):
        """Test section renders load bearing."""
        from model_protocol_lab import reporting

        result = self._result(
            {
                "t1": {
                    "status": "ok",
                    "base_accuracy": 1.0,
                    "base_passes_all": True,
                    "arms": {
                        "user_prefix": {
                            "status": "ok",
                            "original": "U: ",
                            "neutral": "",
                            "n_flips": 2,
                            "pass_to_fail": 2,
                            "fail_to_pass": 0,
                            "flipped_fixture_ids": ("f1", "f2"),
                            "selection_bar_held": False,
                        }
                    },
                    "not_applicable_fields": ["repeat_bos"],
                    "load_bearing_fields": ["user_prefix"],
                    "any_attribution": True,
                }
            }
        )
        html = reporting._ablation_section(result)
        self.assertIn('id="field-ablation"', html)
        self.assertIn("user_prefix", html)
        # Load-bearing star marker.
        self.assertIn("&#9733;", html)

    def test_section_escapes_hostile_names(self):
        """Test section escapes hostile names."""
        from model_protocol_lab import reporting

        evil = 't1"><script>alert(1)</script>'
        result = self._result(
            {
                evil: {
                    "status": "ok",
                    "base_accuracy": 1.0,
                    "base_passes_all": True,
                    "arms": {},
                    "not_applicable_fields": [],
                    "load_bearing_fields": [],
                    "any_attribution": False,
                }
            }
        )
        html = reporting._ablation_section(result)
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)

    def test_section_empty_when_unavailable(self):
        """Test section empty when unavailable."""
        from model_protocol_lab import reporting

        self.assertEqual(reporting._ablation_section({}), "")
        self.assertEqual(reporting._ablation_section({"ablation": {"available": False}}), "")


class AblationDiffTests(unittest.TestCase):
    """Ablation diff tests."""

    @staticmethod
    def _payload(flips=0):
        """Payload."""
        # Producer-consistent story (cycle 176, wave-7 F6d): the
        # base passes every fixture; `flips` pass-to-fail flips
        # break the arm's selection bar, which is exactly what makes
        # the field load-bearing. The strengthened _valid_ablation
        # recomputes these identities, so the fixture cannot assert
        # a story the producer could never emit.
        bar_held = flips == 0
        return {
            "available": True,
            "candidates": {
                "t1": {
                    "status": "ok",
                    "base_passes_all": True,
                    "arms": {
                        "user_prefix": {
                            "status": "ok",
                            "n_fixtures": 4,
                            "n_flips": flips,
                            "pass_to_fail": flips,
                            "fail_to_pass": 0,
                            "flipped_fixture_ids": [f"fx{i}" for i in range(flips)],
                            "selection_bar_held": bar_held,
                        }
                    },
                    "load_bearing_fields": [] if bar_held else ["user_prefix"],
                    "any_attribution": flips > 0,
                }
            },
        }

    def test_identical_payloads(self):
        """Test identical payloads."""
        from model_protocol_lab import diffreports

        diff = diffreports.diff_ablation(self._payload(), self._payload())
        self.assertEqual(diff["available"], "both")
        self.assertFalse(diff["changed"])
        self.assertEqual(diff["attribution_drift"], [])
        self.assertEqual(diff["load_bearing_drift"], [])

    def test_attribution_drift_detected(self):
        """Test attribution drift detected."""
        from model_protocol_lab import diffreports

        diff = diffreports.diff_ablation(self._payload(flips=0), self._payload(flips=2))
        self.assertTrue(diff["changed"])
        self.assertEqual(
            diff["attribution_drift"],
            [
                {
                    "candidate": "t1",
                    "field": "user_prefix",
                    "old_n_flips": 0,
                    "new_n_flips": 2,
                }
            ],
        )

    def test_load_bearing_drift_detected(self):
        """Test load bearing drift detected."""
        from model_protocol_lab import diffreports

        diff = diffreports.diff_ablation(
            self._payload(flips=0),
            self._payload(flips=1),
        )
        self.assertTrue(diff["changed"])
        self.assertEqual(
            diff["load_bearing_drift"],
            [
                {
                    "candidate": "t1",
                    "old_load_bearing": [],
                    "new_load_bearing": ["user_prefix"],
                }
            ],
        )

    def test_candidates_added_removed(self):
        """Test candidates added removed."""
        from model_protocol_lab import diffreports

        old = self._payload()
        new = self._payload()
        new["candidates"]["t2"] = new["candidates"]["t1"]
        diff = diffreports.diff_ablation(old, new)
        self.assertEqual(diff["candidates_added"], ["t2"])
        self.assertTrue(diff["changed"])
        # skipcq: PYL-W1114 - the reversed diff is the assertion: symmetry is what is audited
        diff2 = diffreports.diff_ablation(new, old)
        self.assertEqual(diff2["candidates_removed"], ["t2"])

    def test_asymmetric_and_neither(self):
        """Test asymmetric and neither."""
        from model_protocol_lab import diffreports

        diff = diffreports.diff_ablation(self._payload(), {"available": False})
        self.assertEqual(diff["available"], "old")
        self.assertFalse(diff["changed"])
        diff = diffreports.diff_ablation(None, None)
        self.assertEqual(diff["available"], "neither")

    def test_hostile_payload_rejected(self):
        """Test hostile payload rejected."""
        from model_protocol_lab import diffreports

        bad = self._payload()
        bad["candidates"]["t1"]["arms"]["user_prefix"]["n_flips"] = "many"
        # The hostile side is old -> only the new side counts as available.
        diff = diffreports.diff_ablation(bad, self._payload())
        self.assertEqual(diff["available"], "new")
        self.assertFalse(diff["changed"])


if __name__ == "__main__":
    unittest.main()


class AblationFailClosedTests(unittest.TestCase):
    """Fail-closed defensive paths in ablation_stage (coverage)."""

    def test_arm_evaluation_failed(self):
        """Test arm evaluation failed."""
        from unittest import mock

        from model_protocol_lab import experiments
        from model_protocol_lab.core import evaluate as real_evaluate

        calls = []

        def fake_evaluate(backend, fmt, fixtures, **kwargs):
            """Fake evaluate."""
            calls.append(1)
            if len(calls) >= 3:
                raise RuntimeError("arm boom")
            return real_evaluate(backend, fmt, fixtures, **kwargs)

        with mock.patch("model_protocol_lab.core.evaluate", side_effect=fake_evaluate):
            stage = experiments.ablation_stage(
                (_fmt("x"),), _fixtures(), PrefixBackend(), boundary=(7,)
            )
        rep = stage["candidates"]["x"]
        self.assertEqual(rep["status"], "ok")
        self.assertTrue(rep["arms"])
        self.assertTrue(
            all(a.get("reason") == "arm_evaluation_failed" for a in rep["arms"].values())
        )

    def test_base_evaluation_failed(self):
        """Test base evaluation failed."""
        from unittest import mock

        from model_protocol_lab import experiments
        from model_protocol_lab.core import evaluate as real_evaluate

        calls = []

        def fake_evaluate(backend, fmt, fixtures, **kwargs):
            """Fake evaluate."""
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("base boom")
            return real_evaluate(backend, fmt, fixtures, **kwargs)

        with mock.patch("model_protocol_lab.core.evaluate", side_effect=fake_evaluate):
            stage = experiments.ablation_stage(
                (_fmt("x"),), _fixtures(), PrefixBackend(), boundary=(7,)
            )
        rep = stage["candidates"]["x"]
        self.assertEqual(rep["status"], "unavailable")
        self.assertEqual(rep["reason"], "base_evaluation_failed")

    def test_arm_construction_failed(self):
        """Test arm construction failed."""
        from unittest import mock

        from model_protocol_lab import experiments

        with mock.patch(
            "model_protocol_lab.ablation.ablation_arms",
            side_effect=RuntimeError("construction boom"),
        ):
            stage = experiments.ablation_stage(
                (_fmt("x"),), _fixtures(), PrefixBackend(), boundary=(7,)
            )
        rep = stage["candidates"]["x"]
        self.assertEqual(rep["status"], "unavailable")
        self.assertEqual(rep["reason"], "arm_construction_failed")
