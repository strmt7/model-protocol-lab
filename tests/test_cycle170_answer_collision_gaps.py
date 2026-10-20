"""Cycle 170 gap tests: defensive branches of the answer-collision audit.

Every test here covers a line/branch the main cycle-170 suite leaves
cold under the full CI run (measured via coverage json): validation
rejections, fail-closed emission paths, stage failure taxonomies, the
report-section fallbacks, and the diff validator/renderer edge cases.
"""

from __future__ import annotations

import unittest
from unittest import mock

from model_protocol_lab import collision as col
from model_protocol_lab import core, diffreports, experiments, reporting
from model_protocol_lab.core import Format


class OrdinalBackend:
    fingerprint = "cycle170-gaps-ordinal"

    @staticmethod
    def encode(text):
        return tuple(map(ord, text))

    @staticmethod
    def decode(ids):
        return "".join(map(chr, ids))

    def generate(self, input_ids, *, max_new_tokens, seed, cancel):
        return core.Generation(tuple(input_ids[:max_new_tokens]), "limit")


class SpecialBackend:
    fingerprint = "cycle170-gaps-special"
    MARKER = "<|im_end|>"

    @classmethod
    def encode(cls, text):
        out, i = [], 0
        while i < len(text):
            if text.startswith(cls.MARKER, i):
                out.append(7)
                i += len(cls.MARKER)
            else:
                out.append(ord(text[i]))
                i += 1
        return tuple(out)

    @classmethod
    def decode(cls, ids):
        return "".join(cls.MARKER if x == 7 else chr(x) for x in ids)

    def generate(self, input_ids, *, max_new_tokens, seed, cancel):
        return core.Generation(tuple(input_ids[:max_new_tokens]), "limit")


def _format(name: str) -> Format:
    return Format(name=name, bos="", user_prefix=f"[{name}]", assistant_prefix="A:", turn_end="\n")


def _scored(cls, occurrences=1):
    return {
        "class": cls,
        "cut": 3,
        "exact_match": cls == "intact",
        "verbatim": True,
        "occurrences": occurrences,
    }


def _fixture(fid, reserved, subword, reserved_state="scored", subword_state="no_pair"):
    return {
        "id": fid,
        "reserved": reserved,
        "subword": subword,
        "arm_state": {"reserved": reserved_state, "subword": subword_state},
    }


def _hand_payload(colliding: bool):
    cls_list = ["collision", "intact"] if colliding else ["intact", "intact"]
    per_fixture = [_fixture(f"f{i}", _scored(c), None) for i, c in enumerate(cls_list)]
    cell = col.collision_cell(per_fixture, n_fixtures=2, boundary_len=1)
    cell["protocol_fingerprint"] = col.protocol_fingerprint()
    profile = col.answer_collision_profile(("cand",), {"cand": cell}, col.protocol_dict())
    profile["n_fixtures"] = 2
    return profile


def _base_spec(sid="x", marker="<|m|>", position="interior", expected="a <|m|> b"):
    return {"id": sid, "marker": marker, "position": position, "expected": expected}


class ValidateGapTests(unittest.TestCase):
    def test_clean_with_hostile_string_rejected(self):
        spec = _base_spec(sid="clean", marker=None, position="clean", expected="has <s> here")
        with self.assertRaises(ValueError):
            col.validate_answer_fixtures((spec,), "<|im_end|>")

    def test_non_string_marker_rejected(self):
        for bad_marker in (7, "", None):
            spec = _base_spec(marker=bad_marker)
            with self.assertRaises(ValueError, msg=repr(bad_marker)):
                col.validate_answer_fixtures((spec,), "<|im_end|>")

    def test_position_violations_rejected(self):
        marker = "<|m|>"
        cases = [
            ("start", f"x {marker}"),  # not at start
            ("start", f"{marker} a {marker}"),  # twice
            ("interior", f"{marker} a"),  # at start
            ("interior", f"a {marker}"),  # at end
            ("end", f"a {marker} b"),  # not at end
            ("repeat", f"a {marker} b"),  # only once
            ("hostile", "no marker at all"),
            ("sideways", f"a {marker} b"),
        ]
        for position, expected in cases:
            spec = _base_spec(marker=marker, position=position, expected=expected)
            with self.assertRaises(ValueError, msg=position):
                col.validate_answer_fixtures((spec,), "<|im_end|>")

    def test_empty_marker_text_rejected(self):
        with self.assertRaises(ValueError):
            col.validate_answer_fixtures((_base_spec(),), "")
        with self.assertRaises(ValueError):
            col.build_answer_fixtures("")


class EmissionGapTests(unittest.TestCase):
    def test_reserved_emission_negative_ids_rejected(self):
        ids = col.reserved_emission(lambda t: (-1, 7), OrdinalBackend.decode, "ab")
        self.assertIsNone(ids)

    def test_reserved_emission_decode_raises(self):
        def boom_decode(ids):
            raise RuntimeError("decode boom")

        self.assertIsNone(col.reserved_emission(OrdinalBackend.encode, boom_decode, "ab"))

    def test_subword_emission_non_string_marker(self):
        self.assertIsNone(
            col.subword_emission(SpecialBackend.encode, SpecialBackend.decode, "abc", 7, (7,))
        )

    def test_subword_emission_pair_lookup_raises(self):
        with mock.patch(
            "model_protocol_lab.deliforge.find_encoding_pair",
            side_effect=RuntimeError("lookup boom"),
        ):
            self.assertIsNone(
                col.subword_emission(
                    SpecialBackend.encode,
                    SpecialBackend.decode,
                    "a <|im_end|> b",
                    "<|im_end|>",
                    (7,),
                )
            )

    def test_subword_emission_chunk_encode_raises(self):
        def flaky_encode(text):
            if text == "<|im_end|>":
                return (7,)
            if "z" in text:
                raise RuntimeError("chunk boom")
            return tuple(map(ord, text))

        self.assertIsNone(
            col.subword_emission(
                flaky_encode,
                SpecialBackend.decode,
                "<|im_end|> z <|im_end|>",
                "<|im_end|>",
                (7,),
            )
        )

    def test_subword_emission_invalid_ids(self):
        # The pair lookup succeeds (splits of the marker encode to
        # ordinary ordinals), but re-encoding the chunk " now" yields a
        # negative id: the construction refuses it fail-closed.
        def bad_encode(text):
            if text == "<|im_end|>":
                return (7,)
            if text == " now":
                return (-3,)
            return tuple(map(ord, text))

        self.assertIsNone(
            col.subword_emission(
                bad_encode, SpecialBackend.decode, "say <|im_end|> now", "<|im_end|>", (7,)
            )
        )

    def test_subword_emission_reserved_id_in_control_arm(self):
        # A vocabulary collision: the chunk " then Q done" also encodes
        # through the reserved id 7, so the control arm would contain
        # the very id it controls for -- construction refuses it.
        def colliding_encode(text):
            if text == "<|im_end|>":
                return (7,)
            if "Q" in text:
                return (32, 7, 32)
            return tuple(map(ord, text))

        self.assertIsNone(
            col.subword_emission(
                colliding_encode,
                SpecialBackend.decode,
                "say <|im_end|> then Q done",
                "<|im_end|>",
                (7,),
            )
        )

    def test_subword_emission_final_decode_mismatch(self):
        def short_only_decode(ids):
            if len(ids) > 10:
                return "WRONG"
            return "".join("<|im_end|>" if x == 7 else chr(x) for x in ids)

        self.assertIsNone(
            col.subword_emission(
                SpecialBackend.encode,
                short_only_decode,
                "say <|im_end|> loudly",
                "<|im_end|>",
                (7,),
            )
        )

    def test_subword_emission_final_decode_raises(self):
        def racy_decode(ids):
            if len(ids) > 10:
                raise RuntimeError("decode boom")
            return "".join("<|im_end|>" if x == 7 else chr(x) for x in ids)

        self.assertIsNone(
            col.subword_emission(
                SpecialBackend.encode,
                racy_decode,
                "say <|im_end|> loudly",
                "<|im_end|>",
                (7,),
            )
        )


class StageGapTests(unittest.TestCase):
    def test_repeat_decode_failure_splits_failed_and_anomalous(self):
        # A stateful backend whose decode raises on the SECOND call with
        # the same id tuple: emissions verify fine the first time, then
        # (a) intact fixtures fail inside evaluate's answer decode
        # (status != completed -> the fixture is skipped), and
        # (b) colliding fixtures complete in evaluate (their answer ids
        # differ from the full emission) but the stage's full-output
        # decode raises -> anomalous, never intact.
        class RepeatDecodeBackend(OrdinalBackend):
            def __init__(self):
                self._seen = set()

            def decode(self, ids):
                ids = tuple(ids)
                if ids in self._seen:
                    raise RuntimeError("second decode boom")
                self._seen.add(ids)
                return "".join(map(chr, ids))

        fmt = _format("alpha")
        stage = experiments.answer_collision_stage(
            RepeatDecodeBackend(), (fmt,), boundary=(124,), seed=1
        )
        cell = stage["per_candidate"]["alpha"]
        self.assertIsNotNone(cell)
        # Template prefixes repeat across fixtures, so answer tuples
        # repeat too: both failure lines are exercised (some fixtures
        # fail inside evaluate, the rest complete there and fail the
        # stage's full-output decode), and nothing is scored.
        self.assertGreaterEqual(cell["reserved"]["n_anomalous"], 1)
        self.assertGreaterEqual(cell["reserved"]["n_failed"], 1)
        self.assertEqual(cell["reserved"]["n_scored"], 0)

    def test_evaluate_exception_degrades_fixture(self):
        class PromptBoomBackend(OrdinalBackend):
            @staticmethod
            def encode(text):
                if "Which token ends the turn?" in text:
                    raise RuntimeError("prompt encode boom")
                return tuple(map(ord, text))

        fmt = _format("alpha")
        stage = experiments.answer_collision_stage(
            PromptBoomBackend(), (fmt,), boundary=(124,), seed=1
        )
        cell = stage["per_candidate"]["alpha"]
        self.assertIsNotNone(cell)
        self.assertEqual(cell["reserved"]["n_failed"], 21)

    def test_stage_invalid_forms(self):
        self.assertEqual(
            experiments.answer_collision_profile(("a",), {})["detail"], "stage_invalid"
        )
        self.assertEqual(
            experiments.answer_collision_profile(("a",), {"protocol": None})["detail"],
            "stage_invalid",
        )


class ValidatorGapTests(unittest.TestCase):
    def _arm(self, **over):
        arm = {
            "n_evaluable": 2,
            "n_scored": 2,
            "n_collision": 1,
            "n_ends_at_marker": 0,
            "n_intact": 1,
            "n_anomalous": 0,
            "n_failed": 0,
            "n_roundtrip_excluded": 0,
            "n_no_pair": 0,
            "n_not_applicable": 0,
            "collision_rate": 0.5,
            "ends_rate": 0.0,
        }
        arm.update(over)
        return arm

    def _cell(self, reserved=None, subword=None, **over):
        cell = {
            "n": 2,
            "boundary_ids": 1,
            "reserved": reserved or self._arm(),
            "subword": subword
            or self._arm(
                n_no_pair=2,
                n_evaluable=0,
                n_scored=0,
                n_collision=0,
                n_intact=0,
                collision_rate=None,
                ends_rate=None,
            ),
            "any_collision": True,
            "uniform_intact": False,
            "evaluable": True,
        }
        cell.update(over)
        return cell

    def test_arm_rejections(self):
        for bad in (
            "junk",  # not a dict
            self._arm(n_collision=-1),
            self._arm(n_no_pair=1),  # reserved arm must not carry no_pair
            self._arm(n_evaluable=1),  # class counts exceed evaluable
            self._arm(n_scored=1, n_anomalous=0),  # scored/evaluable mismatch
            self._arm(collision_rate=None),  # rate None with scored > 0
            self._arm(collision_rate=1.5),  # rate out of unit
            self._arm(collision_rate=0.25),  # stale rate
            self._arm(ends_rate=0.25),  # stale ends rate
        ):
            self.assertFalse(
                experiments._answer_collision_arm_valid(bad, control=False), repr(bad)[:60]
            )

    def test_cell_rejections(self):
        self.assertTrue(experiments._answer_collision_cell_valid(None))
        for bad in (
            "junk",
            self._cell(n=-1),
            self._cell(boundary_ids=0),
            self._cell(reserved="junk"),
            self._cell(subword="junk"),
            self._cell(reserved=self._arm(n_failed=3)),  # denominator mismatch
            self._cell(any_collision="yes"),
            self._cell(any_collision=False),  # inconsistent with counts
            self._cell(uniform_intact=True),  # inconsistent with counts
            self._cell(evaluable="yes"),
            self._cell(evaluable=False),  # inconsistent: evaluable is True
        ):
            self.assertFalse(experiments._answer_collision_cell_valid(bad))

    def test_validator_rejections(self):
        for mutate in (
            lambda p: p.update({"protocol": "junk"}),
            lambda p: p.update({"protocol": {"fingerprint": 7}}),
            lambda p: p.update({"n_fixtures": 0}),
            lambda p: p.update({"per_candidate": ["junk"]}),
            lambda p: p.update({"per_candidate": {7: None}}),
            lambda p: p.update({"n_candidates": 99}),
            lambda p: p.update({"n_cells": 99}),
            lambda p: p.update({"n_measurable": 99}),
            lambda p: p.update({"any_collision": "yes"}),
            lambda p: p.update({"uniform_intact": "yes"}),
            lambda p: p.update({"colliding_candidates": "cand"}),
            lambda p: p.update({"colliding_candidates": ["ghost"]}),
            lambda p: p.update({"any_collision": False}),
            lambda p: p.update({"max_collision_rate": 0.01}),
            lambda p: p.update({"worst_candidate": "ghost"}),
        ):
            bad = _hand_payload(True)
            mutate(bad)
            self.assertFalse(diffreports._valid_answer_collision(bad), mutate)

    def test_validator_rates_absent(self):
        # A payload with no measurable cells must carry no rates.
        per_fixture = [_fixture(f"f{i}", None, None, reserved_state="failed") for i in range(2)]
        cell = col.collision_cell(per_fixture, n_fixtures=2, boundary_len=1)
        cell["protocol_fingerprint"] = col.protocol_fingerprint()
        profile = col.answer_collision_profile(("c",), {"c": cell}, col.protocol_dict())
        profile["n_fixtures"] = 2
        self.assertFalse(diffreports._valid_answer_collision(profile))
        profile["available"] = False
        self.assertFalse(diffreports._valid_answer_collision(profile))


class ReportingGapTests(unittest.TestCase):
    def test_section_per_candidate_not_dict(self):
        payload = _hand_payload(True)
        payload["per_candidate"] = ["junk"]
        self.assertEqual(reporting._answer_collision_section({"answer_collision": payload}), "")

    def test_section_none_cell_row(self):
        payload = _hand_payload(True)
        payload["per_candidate"]["cand"] = None
        section = reporting._answer_collision_section({"answer_collision": payload})
        self.assertIn("unavailable", section)

    def test_section_non_numeric_counts(self):
        payload = _hand_payload(True)
        payload["per_candidate"]["cand"]["boundary_ids"] = "seven"
        section = reporting._answer_collision_section({"answer_collision": payload})
        self.assertIn("n/a", section)

    def test_section_uniform_intact_line(self):
        payload = _hand_payload(False)
        section = reporting._answer_collision_section({"answer_collision": payload})
        self.assertIn("Null result", section)


class RendererGapTests(unittest.TestCase):
    def test_protocol_drift_line(self):
        new = _hand_payload(False)
        proto = dict(col.protocol_dict())
        proto["fingerprint"] = "0" * 64
        new["protocol"] = proto
        lines = diffreports._render_answer_collision_summary(
            diffreports.diff_answer_collision(_hand_payload(False), new)
        )
        self.assertTrue(any("protocol drift" in line for line in lines))

    def test_many_drifts_truncated(self):
        old_cells, new_cells = {}, {}
        for n in range(5):
            name = f"cand{n}"
            for bucket, coll in ((old_cells, 1), (new_cells, 3)):
                per_fixture = [
                    _fixture(f"f{i}", _scored("collision" if i < coll else "intact"), None)
                    for i in range(4)
                ]
                cell = col.collision_cell(per_fixture, n_fixtures=4, boundary_len=1)
                cell["protocol_fingerprint"] = col.protocol_fingerprint()
                bucket[name] = cell
        old = col.answer_collision_profile(tuple(old_cells), old_cells, col.protocol_dict())
        old["n_fixtures"] = 4
        new = col.answer_collision_profile(tuple(new_cells), new_cells, col.protocol_dict())
        new["n_fixtures"] = 4
        lines = diffreports._render_answer_collision_summary(
            diffreports.diff_answer_collision(old, new)
        )
        self.assertTrue(any("more rate drifts" in line for line in lines))

    def test_old_side_missing_line(self):
        lines = diffreports._render_answer_collision_summary(
            diffreports.diff_answer_collision(None, _hand_payload(True))
        )
        self.assertTrue(any("did not record" in line for line in lines))

    def test_non_dict_section_returns_empty(self):
        self.assertEqual(diffreports._render_answer_collision_summary(None), [])
        self.assertEqual(diffreports._render_answer_collision_summary("junk"), [])


if __name__ == "__main__":
    unittest.main()
